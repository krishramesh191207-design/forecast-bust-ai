"""Synoptic system detection from the FORECAST fields.

Why this module exists
----------------------
An earlier version of the pipeline took cyclone/low/western-disturbance
proximity from the event catalogue of the verifying analysis. Permutation
importance immediately exposed that as target leakage: those columns
dominated the model because they encoded where systems *actually* were at the
valid time, information no forecaster has when the forecast is issued.

Everything here is therefore computed from the forecast ensemble itself:
closed lows are detected as local minima of the forecast mean-sea-level
pressure anomaly, and classified by latitude, land/sea position, depth and
season. The verifying event catalogue (sandbox systems, or IBTrACS in real
mode) is used only for labelling historical events and for stratified
evaluation - never as a predictor.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter

FAR = 99.0


def detect_lows(mslp: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                land: np.ndarray, month: int,
                background_sigma: float = 5.0,
                min_depth_hpa: float = 1.2,
                footprint: int = 5) -> list[dict]:
    """Detect closed lows in a forecast MSLP field.

    Returns a list of centres with position, depth below the smoothed
    background, and a meteorological class.
    """
    background = gaussian_filter(mslp, background_sigma, mode="nearest")
    depth = background - mslp                       # positive where low
    local_min = mslp == minimum_filter(mslp, size=footprint, mode="nearest")
    ii, jj = np.where(local_min & (depth > min_depth_hpa))

    monsoon = month in (6, 7, 8, 9)
    winter = month in (11, 12, 1, 2, 3)
    tc_season = month in (4, 5, 6, 10, 11, 12)

    out = []
    for i, j in zip(ii, jj):
        lat, lon, d = float(lats[i]), float(lons[j]), float(depth[i, j])
        over_sea = not bool(land[i, j])
        if lat < 25.0 and over_sea and d >= 3.0 and tc_season:
            kind = "tropical_cyclone"
        elif lat >= 26.0 and winter:
            kind = "western_disturbance"
        elif monsoon and 12.0 <= lat <= 28.0 and d >= 1.5:
            kind = "monsoon_low"
        else:
            kind = "cyclonic_circulation"
        out.append({"lat": lat, "lon": lon, "depth_hpa": round(d, 3), "kind": kind})
    return out


def proximity_fields(centres: list[dict], lats: np.ndarray, lons: np.ndarray,
                     kinds=("tropical_cyclone", "monsoon_low", "western_disturbance",
                            "cyclonic_circulation")) -> dict[str, np.ndarray]:
    """Distance (deg) and depth (hPa) of the nearest detected system per class."""
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    out: dict[str, np.ndarray] = {}
    for kind in kinds:
        sub = [c for c in centres if c["kind"] == kind]
        dist = np.full(la.shape, FAR, dtype="float32")
        depth = np.zeros(la.shape, dtype="float32")
        for c in sub:
            dd = np.hypot(la - c["lat"], lo - c["lon"]).astype("float32")
            closer = dd < dist
            dist = np.where(closer, dd, dist)
            depth = np.where(closer, c["depth_hpa"], depth)
        out[f"dist_{kind}"] = dist
        out[f"depth_{kind}"] = depth
    return out
