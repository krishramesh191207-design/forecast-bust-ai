"""Ablation study (PART 38) and spatial cross-validation (PART 36).

Ablation answers: does each feature group actually earn its place?
Spatial CV answers: is the model memorising geography?

Both train on the training split and report on the held-out test split, using
the same gradient-boosting configuration throughout so that only the feature
set (or the region split) changes.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..evaluation.metrics import classification_metrics
from ..features.build import feature_columns, load_blocks
from ..models.ladder import CalibratedModel, make_gradient_boosting
from ..models.train import load_table

LADDER = [
    ("A", ["nwp", "ctx"], "NWP state only"),
    ("B", ["nwp", "ctx", "ens"], "+ ensemble spread"),
    ("C", ["nwp", "ctx", "ens", "vol"], "+ forecast volatility"),
    ("D", ["nwp", "ctx", "ens", "vol", "ana"], "+ historical analogues"),
    ("E", ["nwp", "ctx", "ens", "vol", "ana", "regime"], "+ weather regime and systems"),
]


def _fit_eval(df: pd.DataFrame, cols: list[str], cfg: Config, label: str) -> dict:
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    params = {**cfg["models"]["gradient_boosting"], "max_iter": 200}
    mdl = make_gradient_boosting(params)
    mdl.fit(tr[cols], tr["bust_overall"].to_numpy().astype(int))
    wrapped = CalibratedModel(mdl, method=cfg["models"]["calibration"], name=label)
    wrapped.fit_calibration(va[cols], va["bust_overall"].to_numpy().astype(int))
    y = te["bust_overall"].to_numpy().astype(int)
    m = classification_metrics(y, wrapped.predict_proba(te[cols]),
                               climatology=float(tr["bust_overall"].mean()))
    m.pop("curves", None)
    return m


def run_ablation(cfg: Config | None = None, verbose: bool = True) -> list[dict]:
    cfg = cfg or load_config()
    df = load_table(cfg)
    blocks = load_blocks(cfg)
    rows = []
    for tag, use, label in LADDER:
        cols = [c for b in use for c in blocks.get(b, []) if c in df.columns]
        m = _fit_eval(df, cols, cfg, label)
        rows.append({
            "model": f"Model {tag}",
            "description": label,
            "blocks": use,
            "n_features": len(cols),
            "brier": m["brier"],
            "bss": m["brier_skill_score_vs_climatology"],
            "roc_auc": m.get("roc_auc"),
            "pr_auc": m.get("pr_auc"),
            "recall": m.get("recall"),
            "precision": m.get("precision"),
            "ece": m["calibration"]["expected_calibration_error"],
        })
        if verbose:
            print(f"  Model {tag} ({label:32s}) PR-AUC={rows[-1]['pr_auc']} "
                  f"BSS={rows[-1]['bss']} recall={rows[-1]['recall']}", flush=True)
    (cfg.path("models") / "ablation.json").write_text(json.dumps(rows, indent=2))
    return rows


def run_spatial_cv(cfg: Config | None = None, verbose: bool = True) -> dict:
    """Train without the holdout regions, test only on them."""
    cfg = cfg or load_config()
    df = load_table(cfg)
    cols = [c for c in feature_columns(cfg=cfg) if c in df.columns]
    holdout_names = cfg["splits"]["spatial_holdout_regions"]
    from ..config import load_regions

    ids = {r["name"]: r["id"] for r in load_regions()}
    holdout = [ids.get(n, n) for n in holdout_names]

    tr = df[(df["split"] == "train") & (~df["region"].isin(holdout))]
    va = df[(df["split"] == "val") & (~df["region"].isin(holdout))]
    te_in = df[(df["split"] == "test") & (~df["region"].isin(holdout))]
    te_out = df[(df["split"] == "test") & (df["region"].isin(holdout))]

    params = {**cfg["models"]["gradient_boosting"], "max_iter": 200}
    mdl = make_gradient_boosting(params)
    mdl.fit(tr[cols], tr["bust_overall"].to_numpy().astype(int))
    wrapped = CalibratedModel(mdl, method=cfg["models"]["calibration"], name="spatial_cv")
    wrapped.fit_calibration(va[cols], va["bust_overall"].to_numpy().astype(int))

    clim = float(tr["bust_overall"].mean())
    out = {"holdout_regions": holdout, "trained_without_holdout": True}
    for name, part in (("seen_regions", te_in), ("unseen_regions", te_out)):
        if part.empty:
            continue
        m = classification_metrics(part["bust_overall"].to_numpy().astype(int),
                                   wrapped.predict_proba(part[cols]), climatology=clim)
        m.pop("curves", None)
        out[name] = {"n": m["n"], "positive_rate": m["positive_rate"], "brier": m["brier"],
                     "bss": m["brier_skill_score_vs_climatology"], "roc_auc": m.get("roc_auc"),
                     "pr_auc": m.get("pr_auc"), "recall": m.get("recall"),
                     "ece": m["calibration"]["expected_calibration_error"]}
        if verbose:
            print(f"  {name:16s} PR-AUC={out[name]['pr_auc']} ROC-AUC={out[name]['roc_auc']}",
                  flush=True)
    out["interpretation"] = (
        "If performance on regions never seen in training is far below performance on seen "
        "regions, the model is partly memorising geography rather than learning transferable "
        "predictors of forecast failure."
    )
    (cfg.path("models") / "spatial_cv.json").write_text(json.dumps(out, indent=2))
    return out


def extreme_event_evaluation(cfg: Config | None = None, verbose: bool = True) -> dict:
    """Performance stratified by the VERIFIED event type (PART 35)."""
    cfg = cfg or load_config()
    import joblib

    bundle = joblib.load(cfg.path("models") / "classifiers.joblib")
    df = load_table(cfg)
    te = df[df["split"] == "test"].copy()
    cols = bundle["features"]
    te["p"] = bundle["models"]["gradient_boosting"].predict_proba(te[cols])
    clim = float(df[df["split"] == "train"]["bust_overall"].mean())
    out = []
    groups = [("all", te)]
    if "obs_event_type" in te:
        for ev, grp in te.groupby("obs_event_type", observed=True):
            groups.append((str(ev), grp))
    groups.append(("heavy_rainfall_observed", te[te["obs_precip"] > 25]))
    groups.append(("quiet_dry", te[te["obs_precip"] < 1]))
    for name, grp in groups:
        if len(grp) < 200 or grp["bust_overall"].nunique() < 2:
            continue
        m = classification_metrics(grp["bust_overall"].to_numpy().astype(int),
                                   grp["p"].to_numpy(), climatology=clim)
        out.append({"stratum": name, "n": m["n"], "positive_rate": m["positive_rate"],
                    "pr_auc": m.get("pr_auc"), "roc_auc": m.get("roc_auc"),
                    "recall": m.get("recall"), "brier": m["brier"],
                    "bss": m["brier_skill_score_vs_climatology"]})
        if verbose:
            print(f"  {name:26s} n={m['n']:6d} base={m['positive_rate']:.3f} "
                  f"PR-AUC={m.get('pr_auc')} recall={m.get('recall')}", flush=True)
    (cfg.path("models") / "extreme_event_evaluation.json").write_text(json.dumps(out, indent=2))
    return out
