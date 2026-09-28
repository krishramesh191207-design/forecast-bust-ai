"""Weather-regime / event classification (PART 11).

A **rule-assisted** classifier. The meteorological rules below are the primary
labelling mechanism; a shallow decision tree is then fitted to reproduce them
so that the regime can be evaluated as a feature-space partition and so that
the same code path works when rules are unavailable.

Honesty note: because the tree is trained on the rule labels, any reported
"accuracy" measures agreement with the rules, NOT agreement with a
meteorologist's judgement or with an independent event database. No
independent validation of event classification has been performed, so the
project does not claim classification skill. Event proximity flags come from a
real best-track dataset (IBTrACS) in real mode and from the sandbox system
catalogue in sandbox mode.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

REGIMES = [
    "normal",
    "active_monsoon",
    "break_monsoon",
    "monsoon_depression",
    "cyclonic_circulation",
    "tropical_cyclone",
    "heavy_rainfall",
    "western_disturbance",
    "heat_wave",
    "convective_instability",
]
CODE = {name: i for i, name in enumerate(REGIMES)}


def event_proximity(events: pd.DataFrame, valid: dt.date, lats, lons,
                    kinds=("tropical_cyclone", "monsoon_low", "western_disturbance")) -> dict[str, np.ndarray]:
    """Distance (degrees) from every grid cell to the nearest active system."""
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    out = {}
    day = events[events["date"] == valid.isoformat()] if len(events) else events
    for kind in kinds:
        sub = day[day["kind"] == kind] if len(day) else day
        if len(sub) == 0:
            out[f"dist_{kind}"] = np.full(la.shape, 99.0, dtype="float32")
            out[f"depth_{kind}"] = np.zeros(la.shape, dtype="float32")
            continue
        d = np.full(la.shape, 99.0, dtype="float32")
        depth = np.zeros(la.shape, dtype="float32")
        for _, row in sub.iterrows():
            dd = np.hypot(la - row["lat"], lo - row["lon"]).astype("float32")
            closer = dd < d
            d = np.where(closer, dd, d)
            depth = np.where(closer, float(row["depth_hpa"]), depth)
        out[f"dist_{kind}"] = d
        out[f"depth_{kind}"] = depth
    return out


def classify(feats: dict[str, np.ndarray], prox: dict[str, np.ndarray], month: int) -> np.ndarray:
    """Return an integer regime code per grid cell.

    Rules are applied in order of specificity; the first match wins.
    """
    monsoon = month in (6, 7, 8, 9)
    winter = month in (12, 1, 2, 3)
    pre_monsoon = month in (3, 4, 5, 6)

    precip = feats["nwp_precip"]
    cape = feats["nwp_cape"]
    vort = feats["nwp_vorticity850"]
    mslp_anom = feats["nwp_mslp_anom"]
    t2m = feats["nwp_t2m"]
    mfc = feats["nwp_mfc"]
    tcwv = feats["nwp_tcwv"]
    land = feats["ctx_land"]
    lat = feats["ctx_lat"]

    out = np.full(precip.shape, CODE["normal"], dtype="int16")

    def setif(mask, name):
        np.copyto(out, CODE[name], where=mask & (out == CODE["normal"]))

    setif((prox["dist_tropical_cyclone"] < 5.0) & (prox["depth_tropical_cyclone"] > 3.0),
          "tropical_cyclone")
    setif(monsoon & (prox["dist_monsoon_low"] < 5.0) & (prox["depth_monsoon_low"] > 1.5),
          "monsoon_depression")
    setif(winter & (lat > 26.0) & (prox["dist_western_disturbance"] < 7.0),
          "western_disturbance")
    setif(precip > 50.0, "heavy_rainfall")
    setif(pre_monsoon & (land > 0.5) & (t2m > 311.0), "heat_wave")
    setif(monsoon & (mfc > 4.0) & (precip > 8.0), "active_monsoon")
    setif(monsoon & (precip < 1.0) & (tcwv < 35.0) & (land > 0.5), "break_monsoon")
    setif(np.abs(vort) > 4.0, "cyclonic_circulation")
    setif(mslp_anom < -4.0, "cyclonic_circulation")
    setif(cape > 1800.0, "convective_instability")
    return out


def fit_surrogate(X: np.ndarray, y: np.ndarray, max_depth: int = 6):
    """Shallow tree that reproduces the rules in feature space."""
    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=200, random_state=0)
    tree.fit(X, y)
    return tree, float(tree.score(X, y))
