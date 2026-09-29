"""Forecast Stability and confidence trajectory (FEATURE 2).

Everything here is derived from values the pipeline already produces - the
confidence/bust series across lead times for one cell, the existing Forecast
Volatility Index and the cycle-to-cycle confidence revision.  No new model, no
extra training, no invented score.

Stability is decided by three signed contributions, each using a documented
threshold taken from the code it mirrors:

====================  ================================  ====================
contribution          source                           rule
====================  ================================  ====================
lead-time trend       slope of confidence vs lead day  ``|slope| > 1.0``/lead
                                                              (``inference.run._trend``)
cycle revision        confidence now vs previous       ``|delta| > 5`` points
                      cycle, same valid time
volatility            Forecast Volatility Index        ``>= 1.2`` elevated,
                      (0 .. ~3, ``features/engineering``)  ``<= 0.5`` calm
====================  ================================  ====================

Contributions sum to a score in ``{-3..+3}``: ``>= +1`` improving,
``<= -1`` deteriorating, otherwise stable.  With no usable signal at all the
answer is "insufficient data", never a guess.
"""
from __future__ import annotations

import math
from typing import Sequence

RULES: dict[str, float] = {
    "min_points": 3,               # lead-time points needed for a slope
    "slope_points_per_lead": 1.0,  # mirrors inference.run._trend
    "cycle_change_points": 5.0,    # confidence points between cycles
    "volatility_low": 0.5,         # Forecast Volatility Index at/below = calm
    "volatility_elevated": 1.2,    # Forecast Volatility Index at/above = restless
    "volatility_index_max": 3.0,   # documented practical upper bound of the index
}

# Bust Probability = 1 - confidence/100 exactly, so the confidence slope rule
# scales to 1 confidence point expressed as a fraction.
BUST_SLOPE_THRESHOLD = RULES["slope_points_per_lead"] / 100.0

TRAJECTORY_LABELS = {
    "improving": "IMPROVING",
    "stable": "STABLE",
    "deteriorating": "DETERIORATING",
    "insufficient_data": "INSUFFICIENT DATA",
}
STABILITY_LABELS = dict(TRAJECTORY_LABELS)

_LABEL = {"improving": "Improving", "stable": "Stable",
          "deteriorating": "Deteriorating", "insufficient_data": "Insufficient data"}


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def trajectory(values: Sequence, leads: Sequence[int] | None = None, *,
               higher_is_better: bool = True,
               slope_threshold: float | None = None) -> dict:
    """Trajectory of one series across lead times for a single cell.

    ``values`` is aligned with ``leads`` (ascending lead days); missing entries
    may be ``None``/NaN.  ``higher_is_better`` decides whether a rising slope
    is an improvement, so the same routine drives both the Reliability Score
    (higher is better) and its exact complement, Bust Probability (lower is
    better, threshold scaled by 1/100).
    """
    vals = list(values)
    if leads is None:
        leads = list(range(1, len(vals) + 1))
    leads = [int(l) for l in leads]
    thr = RULES["slope_points_per_lead"] if slope_threshold is None else float(slope_threshold)

    pts = []
    for lead, v in zip(leads, vals):
        f = _num(v)
        if f is not None:
            pts.append({"lead": lead, "value": f})

    out: dict = {
        "points": pts,
        "n": len(pts),
        "slope": None,
        "classification": "insufficient_data",
        "label": _LABEL["insufficient_data"],
        "largestDecline": None,
        "largestImprovement": None,
        "latestChange": None,
        "available": False,
    }

    steps = []
    for a, b in zip(pts, pts[1:]):
        steps.append({"fromLead": a["lead"], "toLead": b["lead"],
                      "delta": round(b["value"] - a["value"], 3)})
    if steps:
        down = min(steps, key=lambda s: s["delta"])
        up = max(steps, key=lambda s: s["delta"])
        # "decline" is a decline in *quality*, so it follows the sign convention
        out["largestDecline"] = down if (down["delta"] < 0) == higher_is_better else None
        out["largestImprovement"] = up if (up["delta"] > 0) == higher_is_better else None
        out["latestChange"] = steps[-1]

    if len(pts) >= RULES["min_points"]:
        slope = _slope([p["lead"] for p in pts], [p["value"] for p in pts])
        out["slope"] = round(slope, 3 if higher_is_better else 5)
        adjusted = slope if higher_is_better else -slope
        cls = ("improving" if adjusted > thr
               else "deteriorating" if adjusted < -thr else "stable")
        out["classification"] = cls
        out["label"] = _LABEL[cls]
        out["available"] = True
    return out


def classify_trajectory(values: Sequence, leads: Sequence[int] | None = None, *,
                        higher_is_better: bool = True,
                        slope_threshold: float | None = None) -> str:
    """Just the classification string for ``trajectory``."""
    return trajectory(values, leads, higher_is_better=higher_is_better,
                      slope_threshold=slope_threshold)["classification"]


def volatility_signal(vol_index) -> str:
    """Map the existing Forecast Volatility Index onto a labelled band."""
    v = _num(vol_index)
    if v is None:
        return "unavailable"
    if v >= RULES["volatility_elevated"]:
        return "elevated"
    if v <= RULES["volatility_low"]:
        return "calm"
    return "moderate"


def stability_score(vol_index) -> int | None:
    """Deterministic 0-100 restatement of the Forecast Volatility Index.

    ``stability_score = round(100 * clamp(1 - vol_index / MAX, 0, 1))`` where
    ``MAX = 3.0`` is the documented practical upper bound of the index.  It is a
    transparent transform, not a separate model; ``None`` when the index is
    unavailable.
    """
    v = _num(vol_index)
    if v is None:
        return None
    max_v = RULES["volatility_index_max"]
    return int(round(100 * max(0.0, min(1.0, 1.0 - v / max_v))))


def classify_stability(traj_cls: str, vol_index, cycle_delta) -> dict:
    """Combine trend, cycle revision and volatility into one stability class."""
    trend_pts = {"improving": 1, "stable": 0, "deteriorating": -1}.get(traj_cls, 0)

    delta = _num(cycle_delta)
    if delta is None:
        cycle_pts, cycle_available = 0, False
    else:
        cycle_available = True
        thr = RULES["cycle_change_points"]
        cycle_pts = 1 if delta > thr else -1 if delta < -thr else 0

    v = _num(vol_index)
    if v is None:
        vol_pts, vol_available = 0, False
    else:
        vol_available = True
        if v >= RULES["volatility_elevated"]:
            vol_pts = -1
        elif v <= RULES["volatility_low"]:
            vol_pts = 1
        else:
            vol_pts = 0

    has_signal = traj_cls != "insufficient_data" or vol_available or cycle_available
    total = trend_pts + cycle_pts + vol_pts
    if not has_signal:
        cls = "insufficient_data"
    elif total >= 1:
        cls = "improving"
    elif total <= -1:
        cls = "deteriorating"
    else:
        cls = "stable"

    return {
        "classification": cls,
        "label": _LABEL[cls],
        "score": stability_score(v),
        "volatilitySignal": volatility_signal(v),
        "contributions": {
            "leadTrend": trend_pts,
            "cycleRevision": cycle_pts,
            "volatility": vol_pts,
            "total": total,
            "cycleRevisionAvailable": cycle_available,
            "volatilityAvailable": vol_available,
        },
        "insufficientData": cls == "insufficient_data",
        "rules": RULES,
    }
