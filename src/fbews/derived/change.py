"""Cycle-to-cycle forecast change (FEATURE 1 - Forecast Drift map).

Comparison rule
---------------
Two forecasts are only compared when they verify on the **same valid time**.
``features/build.py`` defines ``valid(cycle, lead) = cycle + lead days``, so:

    lead_previous = lead + (current_cycle - previous_cycle).days

The previous cycle is always *the previous cycle present in the dataset* - a
weekly dataset therefore needs lead ``N`` of the current cycle against lead
``N + 7`` of the previous one.  When that lead does not exist the comparison is
**refused** (``available=False``) instead of silently comparing different valid
times; the UI shows why rather than a misleading "no change" map.  This module
never falls back to a naive "yesterday's forecast".

Status codes are returned alongside the raw delta so that every surface (map,
legend, tooltip, analysis panel) classifies change with one shared threshold.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Status codes (single source of truth for change classification)
# ---------------------------------------------------------------------------
STATUS_NO_CHANGE = 0
STATUS_IMPROVED = 1
STATUS_DETERIORATED = 2
STATUS_UNAVAILABLE = 3

STATUS: dict[str, int] = {
    "no_change": STATUS_NO_CHANGE,
    "improved": STATUS_IMPROVED,
    "deteriorated": STATUS_DETERIORATED,
    "unavailable": STATUS_UNAVAILABLE,
}
STATUS_LABELS: dict[int, str] = {
    STATUS_NO_CHANGE: "No Change",
    STATUS_IMPROVED: "Improved",
    STATUS_DETERIORATED: "Deteriorated",
    STATUS_UNAVAILABLE: "No Data",
}

REASON_MESSAGES = {
    "no_previous_cycle": "Comparison unavailable - no previous cycle in current dataset.",
    "same_valid_time_out_of_range": "Comparison unavailable for this lead time.",
    "metric_not_defined": "Forecast Change is not defined for this layer.",
    "grids_not_aligned": "Comparison unavailable - cycle grids are not aligned.",
}


# ---------------------------------------------------------------------------
# Metric configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetricSpec:
    """One layer that can be shown as a cycle-to-cycle change."""

    key: str
    label: str
    column: str          # column name on InferenceEngine.frame()
    units: str
    decimals: int
    higher_is_better: bool | None   # None = non-directional, change undefined
    threshold: float | None         # |delta| <= threshold counts as No Change

    @property
    def is_percentage(self) -> bool:
        return self.key in ("bust_probability", "analogue_bust_rate")


# Thresholds are absolute, expressed in the metric's own unit, and are the only
# place where "did this meaningfully change?" is decided.
METRICS: dict[str, MetricSpec] = {
    "confidence": MetricSpec(
        key="confidence", label="Reliability Score", column="confidence",
        units="0-100", decimals=0, higher_is_better=True, threshold=5.0),
    "bust_probability": MetricSpec(
        key="bust_probability", label="Bust Probability", column="bust_probability",
        units="0-1", decimals=3, higher_is_better=False, threshold=0.05),
    "err_precip": MetricSpec(
        key="err_precip", label="Rainfall Error", column="pred_err_precip",
        units="mm/day", decimals=2, higher_is_better=False, threshold=0.5),
    "err_t2m": MetricSpec(
        key="err_t2m", label="Temperature Error", column="pred_err_t2m",
        units="K", decimals=2, higher_is_better=False, threshold=0.2),
    "err_wind": MetricSpec(
        key="err_wind", label="Wind Error", column="pred_err_wind",
        units="m/s", decimals=2, higher_is_better=False, threshold=0.5),
    "err_mslp": MetricSpec(
        key="err_mslp", label="Pressure Error", column="pred_err_mslp",
        units="hPa", decimals=2, higher_is_better=False, threshold=0.5),
    "ens_spread_precip": MetricSpec(
        key="ens_spread_precip", label="Ensemble Rainfall Spread",
        column="ens_precip_std", units="mm/day", decimals=2,
        higher_is_better=False, threshold=0.5),
    "volatility": MetricSpec(
        key="volatility", label="Forecast Volatility Index", column="vol_index",
        units="index", decimals=3, higher_is_better=False, threshold=0.1),
    "analogue_bust_rate": MetricSpec(
        key="analogue_bust_rate", label="Analogue Bust Rate",
        column="ana_bust_rate", units="0-1", decimals=3,
        higher_is_better=False, threshold=0.05),
    # Forecast rainfall has no universal "better" direction: a revision of the
    # rain field is not an improvement or a deterioration of the forecast.
    "fc_precip": MetricSpec(
        key="fc_precip", label="Forecast Rainfall", column="nwp_precip",
        units="mm/day", decimals=2, higher_is_better=None, threshold=None),
}


def metric_spec(key: str) -> MetricSpec | None:
    return METRICS.get(key)


def change_is_defined(key: str) -> bool:
    spec = METRICS.get(key)
    return bool(spec and spec.higher_is_better is not None and spec.threshold is not None)


# ---------------------------------------------------------------------------
# Cycle / valid-time alignment
# ---------------------------------------------------------------------------
def previous_cycle(cycles: Sequence[str], current: str) -> str | None:
    """Largest cycle strictly before ``current``, else ``None``.

    ``cycles`` may be in any order; never assumes a 1-day cadence.
    """
    prior = [c for c in cycles if c < current]
    return max(prior) if prior else None


def valid_time(cycle: str, lead: int) -> dt.date:
    """The day a forecast verifies on (mirrors ``features/build.py``)."""
    return dt.date.fromisoformat(cycle) + dt.timedelta(days=lead)


def lead_for_previous(previous: str, current: str, lead: int, max_lead: int) -> int | None:
    """Lead of ``previous`` that verifies on the same day as ``current`` + ``lead``.

    Returns ``None`` when the required lead is outside ``1..max_lead``.
    """
    if previous == current:
        return None
    gap = (dt.date.fromisoformat(current) - dt.date.fromisoformat(previous)).days
    wanted = lead + gap
    return wanted if 1 <= wanted <= max_lead else None


def cycle_gap(previous: str, current: str) -> int:
    return (dt.date.fromisoformat(current) - dt.date.fromisoformat(previous)).days


def available_leads(previous: str, current: str, leads: Sequence[int]) -> list[int]:
    """Subset of ``leads`` that can be compared against ``previous``."""
    out = [l for l in leads if lead_for_previous(previous, current, int(l), max(leads)) is not None]
    return sorted(out)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def _finite(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def change_status(spec: MetricSpec | None, current, previous) -> int:
    """Status code for one cell.

    ``current``/``previous`` are raw values in the metric's own unit; the caller
    must not pre-round them.
    """
    if spec is None or spec.higher_is_better is None or spec.threshold is None:
        return STATUS_UNAVAILABLE
    cur, prv = _finite(current), _finite(previous)
    if cur is None or prv is None:
        return STATUS_UNAVAILABLE
    delta = cur - prv
    if abs(delta) <= spec.threshold:
        return STATUS_NO_CHANGE
    improved = delta > 0 if spec.higher_is_better else delta < 0
    return STATUS_IMPROVED if improved else STATUS_DETERIORATED


def aligned_change(current: np.ndarray, previous: np.ndarray,
                   spec: MetricSpec | None) -> tuple[list, list[int]]:
    """Raw delta + status code for two identically ordered grids.

    Returns ``(change, status)`` with ``change`` rounded to the metric's
    decimals and ``None`` where either side is missing.  Non-directional
    metrics still report the arithmetic delta but every status is
    ``STATUS_UNAVAILABLE``.
    """
    cur = np.asarray(current, dtype="float64").ravel()
    prv = np.asarray(previous, dtype="float64").ravel()
    if cur.shape != prv.shape:
        raise ValueError(f"grid shapes differ: {cur.shape} vs {prv.shape}")

    dec = spec.decimals if spec is not None else 3
    both = np.isfinite(cur) & np.isfinite(prv)
    delta = np.where(both, cur - prv, np.nan)

    change: list = [None if not np.isfinite(v) else round(float(v), dec) for v in delta]
    status = [change_status(spec, c, p) for c, p in zip(cur, prv)]
    return change, status


def magnitude_text(spec: MetricSpec, delta: float) -> str:
    """Human text for a delta that respects percentage-fraction units."""
    if spec is None:
        return f"{delta:+.3g}"
    if spec.is_percentage:
        return f"{delta * 100:+.0f} percentage points"
    if spec.key == "confidence":
        return f"{delta:+.0f} points"
    return f"{delta:+.{spec.decimals}f} {spec.units}"
