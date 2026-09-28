"""Inference: turn a forecast cycle into confidence products (PARTS 17-20, 27, 44).

Outputs, per lead day D1..D10:
    bust_probability   calibrated P(large forecast error) in [0, 1]
    confidence         100 * (1 - bust_probability), rounded to an integer

The confidence score is a DERIVED OPERATIONAL INDICATOR, not a probability
that the forecast is correct. The band labels (high/moderate/low/very low)
are user-interface conventions, configurable in configs/default.yaml.
"""
from __future__ import annotations

import datetime as dt
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..analogues.engine import AnalogueEngine
from ..config import Config, load_config, load_regions
from ..explainability.narrative import Explainer, uncertainty_decomposition
from ..features.build import DataAssembler, feature_columns, load_blocks
from ..models.train import REG_TARGETS
from ..regimes.classify import REGIMES
from ..verification.bust import BustDefinition

LAYERS = [
    ("confidence", "Forecast confidence", "0-100"),
    ("bust_probability", "Bust probability", "0-1"),
    ("err_precip", "Predicted rainfall error", "mm/day"),
    ("err_t2m", "Predicted temperature error", "K"),
    ("err_wind", "Predicted wind error", "m/s"),
    ("err_mslp", "Predicted pressure error", "hPa"),
    ("ens_spread_precip", "Ensemble rainfall spread", "mm/day"),
    ("volatility", "Forecast volatility index", "index"),
    ("analogue_bust_rate", "Historical analogue bust rate", "fraction"),
    ("regime", "Weather regime", "class"),
    ("fc_precip", "Forecast rainfall", "mm/day"),
]


def band(score: float, cfg: Config) -> str:
    bands = cfg["confidence"]["bands"]
    for name in ("high", "moderate", "low", "very_low"):
        lo, hi = bands[name]
        if lo <= score <= hi:
            return name
    return "very_low"


class InferenceEngine:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        mdir = self.cfg.path("models")
        self.clf = joblib.load(mdir / "classifiers.joblib")
        self.reg = joblib.load(mdir / "regressors.joblib")
        self.analogues = AnalogueEngine.load(mdir / "analogue_engine.joblib")
        self.bust = BustDefinition.load(mdir / "bust_definition.json")
        self.blocks = load_blocks(self.cfg)
        self.features = self.clf["features"]
        self.assembler = DataAssembler(self.cfg)
        self.manifest = json.loads((self.cfg.path("processed") / "MANIFEST.json").read_text())
        self.card = json.loads((mdir / "model_card.json").read_text()) if (mdir / "model_card.json").exists() else {}
        train_ref = pd.read_parquet(self.cfg.path("labels") / "labelled_table.parquet",
                                    columns=["lead", "split"] + sorted(
                                        {d[1] for d in __import__(
                                            "fbews.explainability.narrative",
                                            fromlist=["DIAGNOSTICS"]).DIAGNOSTICS}
                                        & set(self.features)))
        self.train_ref = train_ref[train_ref["split"] == "train"]
        self.explainer = Explainer(
            model=self.clf["models"]["gradient_boosting"],
            features=self.features,
            blocks=self.blocks,
            train_reference=self.train_ref,
            medians=self.clf["train_medians"],
        )
        self.regions = load_regions()

    # ------------------------------------------------------------------
    def available_cycles(self) -> list[str]:
        return [c.isoformat() for c in self.assembler.cycles()]

    @lru_cache(maxsize=4)
    def frame(self, cycle_iso: str) -> pd.DataFrame:
        """Full-grid feature frame for one cycle, with predictions attached."""
        cycle = dt.date.fromisoformat(cycle_iso)
        df = self.assembler.rows_for_cycle(cycle, stride=1, with_targets=True)
        ana = self.analogues.transform(df)
        df = pd.concat([df, ana], axis=1)
        X = df[self.features]
        df["bust_probability"] = self.clf["models"]["gradient_boosting"].predict_proba(X)
        df["confidence"] = np.clip(np.round(100 * (1 - df["bust_probability"])), 0, 100).astype(int)
        for target in REG_TARGETS:
            df[f"pred_{target}"] = np.clip(self.reg["models"][target].predict(X), 0, None)
        for var in ("precipitation", "temperature", "wind", "pressure"):
            key = f"bust_{var}"
            if key in self.clf["models"]:
                df[f"p_{key}"] = self.clf["models"][key].predict_proba(X)
        df["dominant_variable"] = df[[f"p_bust_{v}" for v in
                                      ("precipitation", "temperature", "wind", "pressure")]] \
            .idxmax(axis=1).str.replace("p_bust_", "", regex=False)
        return df

    # ------------------------------------------------------------------
    def products(self, cycle_iso: str) -> dict:
        cfg = self.cfg
        df = self.frame(cycle_iso)
        lats = self.assembler.lats.tolist()
        lons = self.assembler.lons.tolist()
        nlat, nlon = len(lats), len(lons)
        leads = sorted(df["lead"].unique().tolist())

        def grid(sub: pd.DataFrame, col: str, nd: int = 2) -> list[float]:
            arr = sub.sort_values(["lat", "lon"])[col].to_numpy(dtype="float64")
            return [None if not np.isfinite(v) else round(float(v), nd) for v in arr]

        layers: dict[str, dict[str, list]] = {}
        explain_layers: dict[str, dict[str, list]] = {}
        for lead in leads:
            sub = df[df["lead"] == lead].sort_values(["lat", "lon"])
            bulk = self.explainer.reasons_bulk(sub, int(lead))
            explain_layers[str(lead)] = {
                "keys": bulk["keys"].tolist(),
                "values": [[None if not np.isfinite(v) else round(float(v), 3) for v in row]
                           for row in bulk["values"]],
            }
            layers[str(lead)] = {
                "confidence": grid(sub, "confidence", 0),
                "bust_probability": grid(sub, "bust_probability", 3),
                "err_precip": grid(sub, "pred_err_precip", 2),
                "err_t2m": grid(sub, "pred_err_t2m", 2),
                "err_wind": grid(sub, "pred_err_wind", 2),
                "err_mslp": grid(sub, "pred_err_mslp", 2),
                "ens_spread_precip": grid(sub, "ens_precip_std", 2),
                "volatility": grid(sub, "vol_index", 3),
                "analogue_bust_rate": grid(sub, "ana_bust_rate", 3),
                "regime": grid(sub, "regime", 0),
                "fc_precip": grid(sub, "nwp_precip", 2),
            }

        # ---- KPI summary -------------------------------------------------
        worst = df.loc[df["bust_probability"].idxmax()]
        by_lead = df.groupby("lead", observed=True)
        kpi = {
            "overall_confidence": int(round(float(df["confidence"].mean()))),
            "max_bust_probability": round(float(df["bust_probability"].max()), 3),
            "most_affected_region": str(df.groupby("region", observed=True)["bust_probability"]
                                        .mean().idxmax()),
            "most_uncertain_variable": str(df["dominant_variable"].value_counts().idxmax()),
            "lead_of_max_uncertainty": int(by_lead["bust_probability"].mean().idxmax()),
            "worst_cell": {"lat": float(worst["lat"]), "lon": float(worst["lon"]),
                           "lead_day": int(worst["lead"]),
                           "bust_probability": round(float(worst["bust_probability"]), 3)},
            "confidence_by_lead": {int(k): int(round(v)) for k, v in
                                   by_lead["confidence"].mean().items()},
            "bust_probability_by_lead": {int(k): round(float(v), 3) for k, v in
                                         by_lead["bust_probability"].mean().items()},
        }
        trend = _trend(list(kpi["confidence_by_lead"].values()))

        # ---- region table and watchlist ----------------------------------
        regions = self.region_table(df)
        watchlist = self.watchlist(df, regions)

        return {
            "meta": {
                "forecast_cycle": cycle_iso + "T00:00:00Z",
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "data_mode": self.cfg.data_mode,
                "data_banner": ("DEMO / SAMPLE DATA - SYNTHETIC SANDBOX"
                                if self.cfg.is_sandbox else "REAL DATA"),
                "warning": self.manifest.get("warning", ""),
                "model_version": self.card.get("version", cfg["project"]["version"]),
                "model_name": self.card.get("name", "fbews-bust-classifier"),
                "trained_at": self.card.get("trained_at"),
                "git_commit": self.card.get("git_commit"),
                "lead_days": leads,
                "layers": [{"key": k, "label": l, "units": u} for k, l, u in LAYERS],
                "confidence_bands": cfg["confidence"]["bands"],
                "confidence_note": ("Confidence = 100 x (1 - bust probability). It is a derived "
                                    "operational indicator, not a probability that the forecast "
                                    "is correct."),
                "regimes": REGIMES,
                "reason_templates": self.explainer.templates(),
                "bust_definition": {"method": self.bust.method, "percentile": self.bust.percentile,
                                    "absolute": self.bust.absolute},
            },
            "grid": {"lat": lats, "lon": lons, "shape": [nlat, nlon],
                     "resolution_deg": self.cfg.resolution},
            "kpi": {**kpi, "confidence_trend": trend},
            "layers": layers,
            "explanations": explain_layers,
            "regions": regions,
            "watchlist": watchlist,
        }

    # ------------------------------------------------------------------
    def region_table(self, df: pd.DataFrame) -> list[dict]:
        rows = []
        names = {r["id"]: r["name"] for r in self.regions}
        for (rid, lead), grp in df.groupby(["region", "lead"], observed=True):
            if rid == "other":
                continue
            dom = grp["dominant_variable"].value_counts().idxmax()
            rows.append({
                "region": rid,
                "region_name": names.get(rid, rid),
                "lead_day": int(lead),
                "confidence": int(round(float(grp["confidence"].mean()))),
                "bust_probability": round(float(grp["bust_probability"].mean()), 3),
                "max_bust_probability": round(float(grp["bust_probability"].max()), 3),
                "expected_error": {
                    "precipitation_mm_day": round(float(grp["pred_err_precip"].mean()), 2),
                    "temperature_k": round(float(grp["pred_err_t2m"].mean()), 2),
                    "wind_ms": round(float(grp["pred_err_wind"].mean()), 2),
                    "pressure_hpa": round(float(grp["pred_err_mslp"].mean()), 2),
                },
                "dominant_variable": dom,
                "regime": str(grp["regime_name"].value_counts().idxmax()),
                "ensemble_spread_precip": round(float(grp["ens_precip_std"].mean()), 2),
                "volatility_index": round(float(grp["vol_index"].mean()), 3),
                "analogue_bust_rate": round(float(grp["ana_bust_rate"].mean()), 3),
            })
        rows.sort(key=lambda r: (-r["bust_probability"], r["lead_day"]))
        return rows

    def watchlist(self, df: pd.DataFrame, regions: list[dict], limit: int = 14) -> list[dict]:
        """Forecast Reliability Watchlist (PART 20)."""
        out = []
        for row in regions[:limit]:
            sub = df[(df["region"] == row["region"]) & (df["lead"] == row["lead_day"])]
            cell = sub.loc[sub["bust_probability"].idxmax()]
            reasons = self.explainer.reasons(cell, int(row["lead_day"]), limit=2)
            out.append({
                **{k: row[k] for k in ("region", "region_name", "lead_day", "confidence",
                                       "bust_probability", "dominant_variable", "regime")},
                "top_reason": reasons[0]["text"] if reasons else
                              "No dominant driver; all diagnostics near climatology.",
                "analogue_similarity": round(float(sub["ana_similarity"].mean()), 4),
                "attention_required": bool(row["bust_probability"] >= 0.25),
            })
        return out

    # ------------------------------------------------------------------
    def point(self, cycle_iso: str, lat: float, lon: float, lead: int | None = None) -> dict:
        """Full "why is this area low confidence?" payload for one grid cell."""
        df = self.frame(cycle_iso)
        d2 = (df["lat"] - lat) ** 2 + (df["lon"] - lon) ** 2
        nearest = df.loc[d2.idxmin(), ["lat", "lon"]]
        cell = df[(df["lat"] == nearest["lat"]) & (df["lon"] == nearest["lon"])].sort_values("lead")
        leads = cell["lead"].tolist()
        target_lead = lead if lead in leads else leads[0]
        row = cell[cell["lead"] == target_lead].iloc[0]
        X_row = cell[cell["lead"] == target_lead][self.features]

        expl = self.explainer.explain(X_row, row, target_lead, regime=row["regime_name"])
        neighbours = self.analogues.neighbours(X_row.assign(
            **{c: row[c] for c in self.analogues.columns if c in row.index},
            cycle=row["cycle"]), n=5) if True else []

        series = {
            "lead": leads,
            "confidence": [int(v) for v in cell["confidence"]],
            "bust_probability": [round(float(v), 3) for v in cell["bust_probability"]],
            "ensemble_spread_precip": [round(float(v), 2) for v in cell["ens_precip_std"]],
            "volatility_index": [round(float(v), 3) for v in cell["vol_index"]],
            "predicted_error_precip": [round(float(v), 2) for v in cell["pred_err_precip"]],
            "forecast_precip": [round(float(v), 2) for v in cell["nwp_precip"]],
            "observed_precip": ([round(float(v), 2) for v in cell["obs_precip"]]
                                if "obs_precip" in cell else None),
            "observed_error_precip": ([round(float(v), 2) for v in cell["err_precip"]]
                                      if "err_precip" in cell else None),
        }
        conf = int(row["confidence"])
        return {
            "valid_time": row["valid_date"] + "T00:00:00Z",
            "forecast_cycle": row["cycle"] + "T00:00:00Z",
            "lead_day": int(target_lead),
            "location": {"lat": float(row["lat"]), "lon": float(row["lon"]),
                         "region": row["region"]},
            "confidence": conf,
            "confidence_band": band(conf, self.cfg),
            "bust_probability": round(float(row["bust_probability"]), 3),
            "predicted_error": {
                "precipitation": round(float(row["pred_err_precip"]), 2),
                "temperature": round(float(row["pred_err_t2m"]), 2),
                "wind": round(float(row["pred_err_wind"]), 2),
                "pressure": round(float(row["pred_err_mslp"]), 2),
            },
            "variable_bust_probability": {
                v: round(float(row[f"p_bust_{v}"]), 3)
                for v in ("precipitation", "temperature", "wind", "pressure")
                if f"p_bust_{v}" in row.index
            },
            "dominant_variable": row["dominant_variable"],
            "regime": row["regime_name"],
            "forecast": {
                "precipitation_mm_day": round(float(row["nwp_precip"]), 2),
                "t2m_k": round(float(row["nwp_t2m"]), 2),
                "mslp_hpa": round(float(row["nwp_mslp"]), 2),
                "wind_ms": round(float(row["nwp_wspd10"]), 2),
                "cape_j_kg": round(float(row["nwp_cape"]), 1),
                "tcwv_kg_m2": round(float(row["nwp_tcwv"]), 1),
            },
            "ensemble": {
                "precip_spread_mm_day": round(float(row["ens_precip_std"]), 2),
                "precip_iqr_mm_day": round(float(row["ens_precip_iqr"]), 2),
                "pop_gt10": round(float(row["ens_pop_gt10"]), 2),
                "pop_gt25": round(float(row["ens_pop_gt25"]), 2),
                "wind_spread_ms": round(float(row["ens_wind_std"]), 2),
                "spread_percentile": round(self.explainer.percentile_of(
                    "ens_precip_std", float(row["ens_precip_std"]), int(target_lead)), 1),
            },
            "volatility": {
                "index": round(float(row["vol_index"]), 3),
                "precip_revision_mm_day": round(float(row["vol_precip_rev"]), 2)
                if np.isfinite(row["vol_precip_rev"]) else None,
                "mslp_revision_hpa": round(float(row["vol_mslp_rev"]), 2)
                if np.isfinite(row["vol_mslp_rev"]) else None,
                "displacement_cells": round(float(row["vol_displacement"]), 1)
                if np.isfinite(row["vol_displacement"]) else None,
            },
            "analogue": {
                "similarity": round(float(row["ana_similarity"]), 4),
                "n_analogues": int(row["ana_n_analogues"]),
                "historical_bust_rate": round(float(row["ana_bust_rate"]), 3),
                "historical_mean_error_mm_day": round(float(row["ana_mean_error"]), 2),
                "most_similar": neighbours,
            },
            "explanations": [r["text"] for r in expl["reasons"]] or [expl["summary"]],
            "explanation_detail": expl,
            "uncertainty_decomposition": uncertainty_decomposition(
                row, expl["top_positive_contributors"] + expl["top_negative_contributors"]),
            "series": series,
            "verification": ({
                "observed_precip_mm_day": round(float(row["obs_precip"]), 2),
                "observed_error_precip_mm_day": round(float(row["err_precip"]), 2),
                "was_bust": int(row["bust_overall"]) if "bust_overall" in row.index else None,
            } if "obs_precip" in row.index and np.isfinite(row.get("obs_precip", np.nan)) else None),
            "data_banner": ("DEMO / SAMPLE DATA - SYNTHETIC SANDBOX"
                            if self.cfg.is_sandbox else "REAL DATA"),
        }


def _trend(values: list[float]) -> str:
    if len(values) < 3:
        return "unknown"
    slope = float(np.polyfit(np.arange(len(values)), np.array(values, dtype="float64"), 1)[0])
    if slope < -1.0:
        return "deteriorating"
    if slope > 1.0:
        return "improving"
    return "stable"


def save_products(cycle_iso: str, cfg: Config | None = None,
                  engine: InferenceEngine | None = None) -> Path:
    cfg = cfg or load_config()
    engine = engine or InferenceEngine(cfg)
    prod = engine.products(cycle_iso)
    out = cfg.path("products") / f"products_{cycle_iso}.json"
    out.write_text(json.dumps(prod, separators=(",", ":")))
    return out
