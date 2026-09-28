"""Explainability: "why is confidence low?" (PARTS 16, 50).

Two complementary mechanisms, both grounded in the actual model and the actual
feature values - nothing here is generated from a template unless the
underlying number supports it.

1. **Group attribution (occlusion).** For each feature block (ensemble,
   volatility, analogues, NWP state, regime/systems) the block's features are
   replaced by their training medians and the model is re-run. The drop in
   predicted bust probability is that block's contribution for this specific
   grid cell. This is model-based, exact for the model as evaluated, and needs
   no extra dependency. SHAP can be swapped in for per-feature values when the
   `shap` package is available; the group view is what forecasters actually
   act on.

2. **Diagnostic reasons.** A curated set of meteorological conditions, each
   tied to one named feature, a percentile of that feature in the training
   climatology (stratified by lead time), and a sentence that quotes the real
   value and its units. A reason is only emitted when its feature is genuinely
   in the tail of the climatology, so the system cannot claim "strong moisture
   convergence" when moisture convergence is average.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_BLOCK_LABELS = {
    "ens": "ensemble disagreement",
    "vol": "forecast-cycle instability",
    "ana": "historical analogues",
    "nwp": "forecast atmospheric state",
    "regime": "weather regime and synoptic systems",
    "ctx": "season, location and lead time",
}


@dataclass
class Reason:
    key: str
    feature: str
    text: str
    percentile: float
    value: float
    units: str


# Each entry: (key, feature, minimum percentile to fire, units, sentence)
# {v} is the feature value, {p} the percentile in the training climatology.
DIAGNOSTICS: list[tuple[str, str, float, str, str]] = [
    ("ensemble_precip_spread", "ens_precip_std", 85,  "mm/day",
     "Ensemble members disagree strongly on accumulated rainfall here "
     "(spread {v:.1f} mm/day, {p:.0f}th percentile of this lead time)."),
    ("ensemble_wind_spread", "ens_wind_std", 85, "m/s",
     "Ensemble members disagree on the 10 m wind (spread {v:.1f} m/s, "
     "{p:.0f}th percentile)."),
    ("ensemble_pressure_spread", "ens_mslp_std", 85, "hPa",
     "Ensemble members place the pressure pattern differently "
     "(MSLP spread {v:.2f} hPa, {p:.0f}th percentile)."),
    ("forecast_revision", "vol_precip_rev", 85, "mm/day",
     "The rainfall forecast changed by {v:.1f} mm/day compared with the "
     "previous cycle for the same valid time ({p:.0f}th percentile revision)."),
    ("pressure_revision", "vol_mslp_rev", 85, "hPa",
     "The pressure forecast was revised by {v:.2f} hPa since the previous "
     "cycle, indicating an unsettled synoptic solution."),
    ("forecast_displacement", "vol_displacement", 90, "grid cells",
     "The rainfall pattern shifted by about {v:.0f} grid cells between "
     "consecutive forecast cycles, a track/position uncertainty signal."),
    ("volatility_index", "vol_index", 90, "index",
     "Overall forecast volatility is high ({v:.2f}, {p:.0f}th percentile): "
     "successive cycles are not converging."),
    ("analogue_bust_rate", "ana_bust_rate", 80, "fraction",
     "Similar historical forecast situations busted {v:.0%} of the time."),
    ("analogue_error", "ana_mean_error", 85, "mm/day",
     "Historical analogues of this situation carried a mean rainfall error "
     "of {v:.1f} mm/day."),
    ("rare_configuration", "ana_similarity", -80, "similarity",
     "This atmospheric configuration has few close historical analogues "
     "(similarity {v:.3f}, in the lowest {p:.0f}%), so error statistics are "
     "weakly constrained."),
    ("heavy_precip_regime", "nwp_precip", 90, "mm/day",
     "A heavy-rainfall regime is forecast ({v:.0f} mm/day), where errors are "
     "typically large."),
    ("moisture_convergence", "nwp_mfc", 90, "mm/day",
     "Strong low-level moisture convergence is forecast ({v:.0f} mm/day "
     "equivalent), which organises convection and amplifies rainfall error."),
    ("instability", "nwp_cape", 88, "J/kg",
     "The forecast is convectively unstable (CAPE {v:.0f} J/kg), so "
     "rainfall placement is sensitive to small errors."),
    ("moisture_load", "nwp_tcwv", 90, "kg/m2",
     "The column is very moist ({v:.0f} kg/m2), raising the ceiling on "
     "possible rainfall amounts."),
    ("wind_shear", "nwp_shear", 88, "m/s",
     "Strong vertical wind shear ({v:.0f} m/s) is forecast across the layer."),
    ("pressure_gradient", "nwp_mslp_grad", 90, "hPa/100 km",
     "A tight pressure gradient is forecast ({v:.2f} hPa/100 km)."),
    ("vorticity", "nwp_vorticity850", 90, "1e-5 1/s",
     "Marked low-level cyclonic vorticity ({v:.1f}e-5 /s) is forecast."),
    ("cyclone_proximity", "sys_dist_cyclone", -10, "deg",
     "A forecast tropical cyclone centre lies about {v:.1f} degrees away; "
     "track error dominates local forecast reliability."),
    ("low_proximity", "sys_dist_low", -12, "deg",
     "A forecast monsoon low/depression centre lies about {v:.1f} degrees "
     "away, so small track shifts move the rainfall maximum."),
    ("wd_proximity", "sys_dist_wd", -12, "deg",
     "A forecast western disturbance lies about {v:.1f} degrees away."),
    ("heat_regime", "nwp_t2m", 95, "K",
     "An extreme-heat regime is forecast ({v:.1f} K, {p:.0f}th percentile), "
     "where 2 m temperature errors grow."),
]

REGIME_NOTES = {
    "tropical_cyclone": "Cyclone-specific uncertainty mode: track and intensity errors dominate.",
    "monsoon_depression": "Monsoon-depression mode: rainfall is tied to the depression track.",
    "active_monsoon": "Active monsoon mode: organised convection, large rainfall gradients.",
    "break_monsoon": "Break-monsoon mode: suppressed rainfall, generally smaller absolute errors.",
    "western_disturbance": "Western-disturbance mode: timing and southward extent are uncertain.",
    "heat_wave": "Heat-wave mode: temperature errors are the dominant risk.",
    "heavy_rainfall": "Heavy-rainfall mode: threshold exceedance is highly sensitive to placement.",
    "convective_instability": "Convective mode: rainfall placement is poorly constrained.",
}


class Explainer:
    """Builds explanations from a fitted model and a training-climatology table."""

    def __init__(self, model, features: list[str], blocks: dict[str, list[str]],
                 train_reference: pd.DataFrame, medians: dict[str, float]):
        self.model = model
        self.features = features
        self.blocks = {k: [f for f in v if f in features] for k, v in blocks.items()}
        self.medians = medians
        # climatology of each diagnostic feature, stratified by lead
        self.clim: dict[int, dict[str, np.ndarray]] = {}
        diag_feats = [d[1] for d in DIAGNOSTICS if d[1] in train_reference.columns]
        self.diag_feats = diag_feats
        for lead, grp in train_reference.groupby("lead", observed=True):
            self.clim[int(lead)] = {
                f: np.sort(grp[f].dropna().to_numpy(dtype="float32")) for f in diag_feats
            }

    # ------------------------------------------------------------------
    def percentile_of(self, feature: str, value: float, lead: int) -> float:
        ref = self.clim.get(int(lead), {}).get(feature)
        if ref is None or len(ref) == 0 or not np.isfinite(value):
            return float("nan")
        return float(np.searchsorted(ref, value) / len(ref) * 100.0)

    # ------------------------------------------------------------------
    def block_attribution(self, X: pd.DataFrame) -> list[dict]:
        """Drop in bust probability when each feature block is neutralised."""
        base = self.model.predict_proba(X)
        out = []
        for block, cols in self.blocks.items():
            if not cols:
                continue
            Xm = X.copy()
            for c in cols:
                Xm[c] = self.medians.get(c, np.nanmedian(X[c]))
            p = self.model.predict_proba(Xm)
            out.append({
                "block": block,
                "label": FEATURE_BLOCK_LABELS.get(block, block),
                "contribution": round(float(np.mean(base - p)), 4),
            })
        out.sort(key=lambda d: -d["contribution"])
        return out

    # ------------------------------------------------------------------
    def reasons(self, row: pd.Series, lead: int, limit: int = 4) -> list[dict]:
        found: list[dict] = []
        for key, feature, min_pct, units, template in DIAGNOSTICS:
            if feature not in row or not np.isfinite(row[feature]):
                continue
            value = float(row[feature])
            # proximity diagnostics must not fire on the "nothing detected"
            # sentinel distance
            if key.endswith("proximity") and value > 15.0:
                continue
            pct = self.percentile_of(feature, value, lead)
            if not np.isfinite(pct):
                continue
            if min_pct >= 0:
                fires, strength = pct >= min_pct, pct
            else:                      # negative threshold: low values matter
                fires, strength = pct <= abs(min_pct), 100.0 - pct
            if not fires:
                continue
            found.append({
                "key": key,
                "feature": feature,
                "value": round(value, 4),
                "percentile": round(pct, 1),
                "units": units,
                "text": template.format(v=value, p=pct if min_pct >= 0 else 100 - pct),
                "strength": round(float(strength), 2),
            })
        found.sort(key=lambda d: -d["strength"])
        return found[:limit]

    # ------------------------------------------------------------------
    def reasons_bulk(self, df: pd.DataFrame, lead: int, top: int = 3) -> dict:
        """Vectorised reason ranking for a whole grid (used for map products).

        Returns the indices into DIAGNOSTICS of the strongest reasons per row,
        their feature values and percentiles, so a client can render the same
        sentences without recomputing the model.
        """
        n = len(df)
        strengths = np.full((len(DIAGNOSTICS), n), -1.0, dtype="float32")
        values = np.zeros((len(DIAGNOSTICS), n), dtype="float32")
        pcts = np.zeros((len(DIAGNOSTICS), n), dtype="float32")
        for di, (key, feature, min_pct, _units, _t) in enumerate(DIAGNOSTICS):
            ref = self.clim.get(int(lead), {}).get(feature)
            if ref is None or feature not in df.columns or len(ref) == 0:
                continue
            v = df[feature].to_numpy(dtype="float32")
            pct = np.searchsorted(ref, v) / len(ref) * 100.0
            values[di] = v
            pcts[di] = pct
            if min_pct >= 0:
                fires, strength = pct >= min_pct, pct
            else:
                fires, strength = pct <= abs(min_pct), 100.0 - pct
            if key.endswith("proximity"):
                fires = fires & (v <= 15.0)
            fires = fires & np.isfinite(v)
            strengths[di] = np.where(fires, strength, -1.0)

        order = np.argsort(-strengths, axis=0)[:top]           # (top, n)
        keys, vals = [], []
        for r in range(top):
            idx = order[r]
            st = strengths[idx, np.arange(n)]
            keys.append(np.where(st > 0, idx, -1).astype("int16"))
            vals.append(np.where(st > 0, values[idx, np.arange(n)], np.nan).astype("float32"))
        return {"keys": np.stack(keys, axis=1), "values": np.stack(vals, axis=1)}

    @staticmethod
    def templates() -> list[dict]:
        return [{"key": k, "feature": f, "units": u, "template": t}
                for k, f, _m, u, t in DIAGNOSTICS]

    # ------------------------------------------------------------------
    def explain(self, X_row: pd.DataFrame, row: pd.Series, lead: int,
                regime: str | None = None, limit: int = 4) -> dict:
        reasons = self.reasons(row, lead, limit=limit)
        attribution = self.block_attribution(X_row)
        positive = [a for a in attribution if a["contribution"] > 0][:3]
        negative = [a for a in attribution if a["contribution"] < 0][-2:]
        note = REGIME_NOTES.get(regime or "", None)
        if not reasons:
            summary = ("No single driver stands out: all diagnostics are near their "
                       "climatological values for this lead time.")
        else:
            summary = reasons[0]["text"]
        return {
            "summary": summary,
            "reasons": reasons,
            "top_positive_contributors": positive,
            "top_negative_contributors": negative,
            "regime_note": note,
        }


def uncertainty_decomposition(row: pd.Series, attribution: list[dict]) -> dict:
    """Named, honestly-scoped uncertainty components (PART 15).

    This is NOT a formal epistemic/aleatoric decomposition. Each component is
    a clearly defined, measurable indicator; they are reported side by side
    rather than summed into a single variance budget.
    """
    contrib = {a["block"]: a["contribution"] for a in attribution}
    return {
        "note": ("Indicators, not a variance decomposition. Each entry is a distinct "
                 "measurable source of uncertainty; they are not mutually exclusive "
                 "and do not sum to the total."),
        "ensemble_uncertainty": {
            "precip_spread_mm_day": _num(row.get("ens_precip_std")),
            "wind_spread_ms": _num(row.get("ens_wind_std")),
            "model_contribution": contrib.get("ens"),
        },
        "forecast_instability": {
            "volatility_index": _num(row.get("vol_index")),
            "precip_revision_mm_day": _num(row.get("vol_precip_rev")),
            "model_contribution": contrib.get("vol"),
        },
        "historical_error_uncertainty": {
            "analogue_bust_rate": _num(row.get("ana_bust_rate")),
            "analogue_error_spread": _num(row.get("ana_error_spread")),
            "model_contribution": contrib.get("ana"),
        },
        "regime_uncertainty": {
            "nearest_system_deg": _num(row.get("sys_dist_low")),
            "model_contribution": contrib.get("regime"),
        },
        "data_uncertainty": {
            "note": ("In sandbox mode the verifying analysis is exact by construction, "
                     "so observation uncertainty is zero. With real data this term must "
                     "carry gauge density, satellite retrieval error and reanalysis "
                     "uncertainty - see docs/limitations.md."),
        },
    }


def _num(v):
    try:
        f = float(v)
        return round(f, 4) if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None
