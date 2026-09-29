"""Synoptic-system board helpers (Operations Center).

Detection itself lives in :mod:`fbews.features.systems` and is computed from
the **forecast** pressure field only - never from the verifying event
catalogue - so nothing here can leak the answer.  This module adds the
operational dressing: which analysis regions a system influences, how serious
that influence is, and how the board is ordered.

Documented thresholds (there were none in the project, so they are stated
here and nowhere else):

* ``INFLUENCE_RADIUS_DEG`` = 5.0 degrees (~550 km).  A region is "affected" by
  a centre when at least one of its grid cells lies within this radius of the
  centre.  It is an operational convention for the board, not a physical
  radius of the system.
* Impact is read off the **mean Reliability Score of the affected cells**,
  using the same cut points as the confidence bands (40 / 60 / 80):
  mean < 40 -> high, < 60 -> moderate, < 80 -> low, otherwise minimal.
* Open ridges and open troughs are *not* detected.  The detector finds closed
  lows only; anything it cannot find is reported as unavailable rather than
  guessed at.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

INFLUENCE_RADIUS_DEG = 5.0
IMPACT_CUTS: tuple[tuple[float, str], ...] = (
    (40.0, "high"),
    (60.0, "moderate"),
    (80.0, "low"),
    (float("inf"), "minimal"),
)
KIND_LABELS = {
    "tropical_cyclone": "Tropical cyclone",
    "monsoon_low": "Monsoon low",
    "western_disturbance": "Western disturbance",
    "cyclonic_circulation": "Cyclonic circulation",
}
NOT_DETECTED = ("ridge", "trough")


def classify_impact(mean_confidence: float | None) -> tuple[str, str]:
    """(severity, label) for the mean Reliability Score of affected cells."""
    if mean_confidence is None:
        return "unknown", "Impact unavailable"
    for cut, name in IMPACT_CUTS:
        if mean_confidence < cut:
            return name, {"high": "High impact", "moderate": "Moderate impact",
                          "low": "Low impact",
                          "minimal": "Minimal impact"}[name]
    return "minimal", "Minimal impact"


def great_circle_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Angular distance in degrees - enough for a fixed-degree radius test."""
    import math
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    c = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(c))))


def cells_within(lat: float, lon: float, radius_deg: float,
                 cells: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for c in cells:
        if great_circle_deg(lat, lon, c["lat"], c["lon"]) <= radius_deg:
            out.append(c)
    return out


def _mean(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and v == v]
    return round(sum(vals) / len(vals), 2) if vals else None


def build_board(centres: Sequence[dict[str, Any]], regions: Sequence[str],
                cells: Sequence[dict[str, Any]],
                radius_deg: float = INFLUENCE_RADIUS_DEG) -> dict[str, Any]:
    """Turn raw detected centres into board records.

    ``cells`` must already be filtered to one lead time and carry
    ``lat/lon/region/confidence/bust_probability``.
    """
    systems = []
    for i, c in enumerate(sorted(centres,
                                 key=lambda x: -(x.get("depth_hpa") or 0))):
        near = cells_within(c["lat"], c["lon"], radius_deg, cells)
        by_region: dict[str, list[dict[str, Any]]] = {}
        for cell in near:
            by_region.setdefault(cell["region"], []).append(cell)
        region_rows = []
        for rid, group in by_region.items():
            mc = _mean(g.get("confidence") for g in group)
            sev, lab = classify_impact(mc)
            region_rows.append({
                "region": rid,
                "cells": len(group),
                "meanConfidence": mc,
                "maxBustProbability": round(
                    max((g.get("bust_probability") or 0) for g in group), 3),
                "severity": sev,
                "label": lab,
            })
        region_rows.sort(key=lambda r: (r["meanConfidence"] is None,
                                        r["meanConfidence"] or 0))
        overall = _mean(r["meanConfidence"] for r in region_rows
                        if r["meanConfidence"] is not None)
        sev, lab = classify_impact(overall)
        nearest = min((r["region"] for r in region_rows), default=None)
        systems.append({
            "id": f"sys{i + 1}",
            "type": c["kind"],
            "label": KIND_LABELS.get(c["kind"], c["kind"]),
            "lat": round(float(c["lat"]), 3),
            "lon": round(float(c["lon"]), 3),
            "depthHpa": round(float(c["depth_hpa"]), 3) if c.get("depth_hpa") is not None else None,
            "influenceRadiusDeg": radius_deg,
            "regionsAffected": [r["region"] for r in region_rows],
            "nearestRegion": nearest,
            "meanConfidence": overall,
            "severity": sev,
            "labelImpact": lab,
            "regionDetail": region_rows,
            "detectedFrom": "forecast mean sea-level pressure (deterministic)",
        })
    # most influential first, then deepest
    systems.sort(key=lambda s: ({"high": 0, "moderate": 1, "low": 2,
                                 "minimal": 3, "unknown": 4}.get(s["severity"], 4),
                                -(s["depthHpa"] or 0)))
    for i, s in enumerate(systems):
        s["id"] = f"sys{i + 1}"
    return {
        "systems": systems,
        "notDetected": list(NOT_DETECTED),
        "notDetectedReason": (
            "Ridges and open troughs are not identified by the current detector: "
            "fbews.features.systems.detect_lows finds closed lows in the forecast "
            "MSLP field only. These features are reported as not detected rather "
            "than estimated."
        ),
        "influenceRadiusDeg": radius_deg,
    }


def regime_summary(regime_counts: dict[str, int]) -> list[dict[str, Any]]:
    total = sum(regime_counts.values()) or 1
    rows = [{"regime": k, "cells": int(v),
             "share": round(100.0 * v / total, 1)}
            for k, v in sorted(regime_counts.items(), key=lambda kv: -kv[1])]
    return rows


def group_watchlist(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group watchlist rows by dominant variable (the closest thing the data
    has to an event type on a live board)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        key = str(r.get("dominant_variable") or "unknown")
        out.setdefault(key, []).append(dict(r))
    for key in out:
        out[key].sort(key=lambda r: (r.get("confidence") if r.get("confidence") is not None else 101))
    return out
