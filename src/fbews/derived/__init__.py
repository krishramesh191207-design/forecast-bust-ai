"""Cycle-aware derived products shared by the API (FEATURES 1 and 2).

`change`     – FEATURE 1, the Forecast Drift map: how a forecast was revised
               between two forecast cycles that verify on the same day.
`stability`  – FEATURE 2, Forecast Stability and the confidence trajectory
               across lead times.

Both modules are deliberately pure (numpy / stdlib only) so every rule that the
dashboard shows can be unit tested without loading a trained model.
"""
from __future__ import annotations

from .change import (
    METRICS,
    REASON_MESSAGES,
    STATUS,
    STATUS_LABELS,
    MetricSpec,
    aligned_change,
    available_leads,
    change_is_defined,
    change_status,
    cycle_gap,
    lead_for_previous,
    metric_spec,
    previous_cycle,
    valid_time,
)
from .stability import (
    BUST_SLOPE_THRESHOLD,
    RULES,
    STABILITY_LABELS,
    TRAJECTORY_LABELS,
    classify_stability,
    classify_trajectory,
    stability_score,
    trajectory,
    volatility_signal,
)

__all__ = [
    "BUST_SLOPE_THRESHOLD", "METRICS", "REASON_MESSAGES", "RULES", "STATUS",
    "STATUS_LABELS", "STABILITY_LABELS", "TRAJECTORY_LABELS", "MetricSpec",
    "aligned_change", "available_leads", "change_is_defined", "change_status",
    "classify_stability", "classify_trajectory", "cycle_gap",
    "lead_for_previous", "metric_spec", "previous_cycle", "stability_score",
    "trajectory", "valid_time", "volatility_signal",
]
