"""Unit tests for the FEATURE 1 / FEATURE 2 derived rules.

These run without trained models: every rule the dashboard shows lives in
``fbews.derived`` as a pure function.
"""
import datetime as dt

import numpy as np
import pytest

from fbews.derived.change import (
    METRICS,
    STATUS_DETERIORATED,
    STATUS_IMPROVED,
    STATUS_NO_CHANGE,
    STATUS_UNAVAILABLE,
    aligned_change,
    available_leads,
    change_is_defined,
    change_status,
    cycle_gap,
    lead_for_previous,
    magnitude_text,
    metric_spec,
    previous_cycle,
    valid_time,
)
from fbews.derived.stability import (
    BUST_SLOPE_THRESHOLD,
    RULES,
    classify_stability,
    classify_trajectory,
    stability_score,
    trajectory,
    volatility_signal,
)

CONF = METRICS["confidence"]
BUST = METRICS["bust_probability"]
RAIN = METRICS["fc_precip"]


# ---------------------------------------------------------------------------
# cycle / valid-time alignment
# ---------------------------------------------------------------------------
CYCLES = ["2019-06-01", "2019-06-08", "2019-06-15", "2019-06-22"]


def test_previous_cycle_is_strictly_earlier():
    assert previous_cycle(CYCLES, "2019-06-15") == "2019-06-08"


def test_previous_cycle_never_returns_the_same_or_a_later_cycle():
    assert previous_cycle(CYCLES, "2019-06-08") == "2019-06-01"
    assert previous_cycle(CYCLES, "2019-06-01") is None


def test_previous_cycle_works_on_unordered_input():
    assert previous_cycle(list(reversed(CYCLES)), "2019-06-15") == "2019-06-08"


def test_previous_cycle_is_not_naive_minus_one_day():
    # weekly dataset: the previous cycle is 7 days back, never yesterday
    prev = previous_cycle(CYCLES, "2019-06-15")
    assert prev == "2019-06-08"
    assert cycle_gap(prev, "2019-06-15") == 7


def test_valid_time_rule_matches_the_pipeline():
    # features/build.py: valid = cycle + lead days
    assert valid_time("2019-06-08", 3) == dt.date(2019, 6, 11)


def test_previous_lead_hits_the_same_valid_time():
    prev, cur = "2019-06-08", "2019-06-15"
    for lead in (1, 2, 3):
        prev_lead = lead_for_previous(prev, cur, lead, max_lead=10)
        assert prev_lead == lead + 7
        assert valid_time(prev, prev_lead) == valid_time(cur, lead)


def test_previous_lead_refused_when_the_valid_time_is_out_of_range():
    # lead 4 needs day 11 of the previous cycle; only 1..10 exist
    assert lead_for_previous("2019-06-08", "2019-06-15", 4, max_lead=10) is None
    assert lead_for_previous("2019-06-08", "2019-06-15", 10, max_lead=10) is None
    # ...but lead 3 needs day 10, which does exist
    assert lead_for_previous("2019-06-08", "2019-06-15", 3, max_lead=10) == 10


def test_available_leads_lists_only_comparable_leads():
    assert available_leads("2019-06-08", "2019-06-15", list(range(1, 11))) == [1, 2, 3]


# ---------------------------------------------------------------------------
# change classification (the ten scenarios the UI depends on)
# ---------------------------------------------------------------------------
def test_confidence_fall_is_a_deterioration():
    # 72 -> 49 = a 23 point loss of reliability
    assert change_status(CONF, 49, 72) == STATUS_DETERIORATED


def test_confidence_rise_is_an_improvement():
    # 49 -> 72 = a 23 point gain of reliability
    assert change_status(CONF, 72, 49) == STATUS_IMPROVED


def test_bust_probability_rise_is_a_deterioration():
    # 28% -> 51% is +23 percentage points of risk, i.e. worse
    assert change_status(BUST, 0.51, 0.28) == STATUS_DETERIORATED


def test_bust_probability_fall_is_an_improvement():
    assert change_status(BUST, 0.28, 0.51) == STATUS_IMPROVED


def test_change_inside_the_threshold_is_no_change():
    assert change_status(CONF, 72, 69) == STATUS_NO_CHANGE          # 3 points < 5
    assert change_status(CONF, 72, 67) == STATUS_NO_CHANGE          # exactly 5
    assert change_status(CONF, 66, 72) == STATUS_DETERIORATED       # 6 > 5
    assert change_status(BUST, 0.30, 0.26) == STATUS_NO_CHANGE      # 4 pp < 5 pp


def test_non_directional_metric_has_no_change_status():
    assert RAIN.higher_is_better is None
    assert change_status(RAIN, 40.0, 12.0) == STATUS_UNAVAILABLE
    assert not change_is_defined("fc_precip")
    assert change_is_defined("confidence")


def test_missing_values_are_unavailable_never_no_change():
    assert change_status(CONF, 72, None) == STATUS_UNAVAILABLE
    assert change_status(CONF, None, 49) == STATUS_UNAVAILABLE
    assert change_status(CONF, float("nan"), 49) == STATUS_UNAVAILABLE
    assert change_status(CONF, 72, np.nan) == STATUS_UNAVAILABLE
    assert change_status(None, 72, 49) == STATUS_UNAVAILABLE


def test_every_registered_metric_has_metadata():
    for key, spec in METRICS.items():
        assert spec.key == key
        assert spec.decimals >= 0
        if spec.higher_is_better is None:
            assert spec.threshold is None
        else:
            assert spec.threshold > 0


def test_aligned_change_rounds_and_flags_missing_cells():
    cur = np.array([72.0, 49.0, np.nan, 60.0])
    prv = np.array([49.0, 72.0, 50.0, 60.0])
    change, status = aligned_change(cur, prv, CONF)
    assert change == [23.0, -23.0, None, 0.0]
    assert status == [STATUS_IMPROVED, STATUS_DETERIORATED,
                      STATUS_UNAVAILABLE, STATUS_NO_CHANGE]


def test_aligned_change_rejects_mismatched_grids():
    with pytest.raises(ValueError):
        aligned_change(np.zeros(4), np.zeros(5), CONF)


def test_percentage_deltas_are_described_in_points():
    assert magnitude_text(BUST, 0.23) == "+23 percentage points"
    assert magnitude_text(CONF, -23) == "-23 points"
    assert "mm/day" in magnitude_text(METRICS["err_precip"], -1.5)


def test_unknown_metric_lookup_is_none():
    assert metric_spec("not-a-layer") is None


# ---------------------------------------------------------------------------
# trajectory
# ---------------------------------------------------------------------------
LEADS = list(range(1, 11))


def test_trajectory_needs_at_least_three_points():
    assert classify_trajectory([80, 78]) == "insufficient_data"
    assert classify_trajectory([80, 78, 76]) == "deteriorating"


def test_trajectory_classifies_a_clear_decline():
    assert classify_trajectory([95, 92, 88, 83, 77, 70, 62], LEADS[:7]) == "deteriorating"


def test_trajectory_classifies_a_clear_improvement():
    assert classify_trajectory([50, 55, 61, 68, 76, 85], LEADS[:6]) == "improving"


def test_trajectory_classifies_a_flat_series_as_stable():
    assert classify_trajectory([70, 71, 70, 71, 70, 71], LEADS[:6]) == "stable"


def test_trajectory_ignores_missing_entries():
    out = trajectory([95, None, np.nan, 70, 55, 40], LEADS[:6])
    assert out["n"] == 4
    assert out["classification"] == "deteriorating"


def test_trajectory_reports_largest_step_and_latest_change():
    out = trajectory([90, 88, 70, 72, 71], LEADS[:5])
    assert out["largestDecline"] == {"fromLead": 2, "toLead": 3, "delta": -18.0}
    assert out["largestImprovement"] == {"fromLead": 3, "toLead": 4, "delta": 2.0}
    assert out["latestChange"]["delta"] == -1.0


def test_bust_trajectory_is_the_mirror_of_the_confidence_trajectory():
    conf = [90, 84, 78, 70, 62, 55]
    bust = [1 - v / 100 for v in conf]
    assert classify_trajectory(conf, LEADS[:6]) == "deteriorating"
    assert classify_trajectory(bust, LEADS[:6], higher_is_better=False,
                               slope_threshold=BUST_SLOPE_THRESHOLD) == "deteriorating"
    assert BUST_SLOPE_THRESHOLD == RULES["slope_points_per_lead"] / 100


# ---------------------------------------------------------------------------
# volatility / stability
# ---------------------------------------------------------------------------
def test_volatility_signal_bands():
    assert volatility_signal(0.2) == "calm"
    assert volatility_signal(0.5) == "calm"
    assert volatility_signal(0.9) == "moderate"
    assert volatility_signal(1.2) == "elevated"
    assert volatility_signal(np.nan) == "unavailable"
    assert volatility_signal(None) == "unavailable"


def test_stability_score_is_a_bounded_transform_of_the_volatility_index():
    assert stability_score(0.0) == 100
    assert stability_score(1.5) == 50
    assert stability_score(3.0) == 0
    assert stability_score(9.0) == 0          # clamped, never negative
    assert stability_score(np.nan) is None
    assert stability_score(None) is None


def test_stability_needs_some_signal():
    out = classify_stability("insufficient_data", None, None)
    assert out["classification"] == "insufficient_data"
    assert out["insufficientData"] is True
    assert out["score"] is None


def test_calm_volatility_alone_pushes_toward_improving():
    out = classify_stability("stable", 0.2, None)
    assert out["classification"] == "improving"
    assert out["contributions"] == {
        "leadTrend": 0, "cycleRevision": 0, "volatility": 1, "total": 1,
        "cycleRevisionAvailable": False, "volatilityAvailable": True}


def test_elevated_volatility_and_falling_confidence_deteriorate():
    out = classify_stability("deteriorating", 1.8, -12)
    assert out["classification"] == "deteriorating"
    assert out["contributions"]["total"] == -3


def test_mixed_signals_average_to_stable():
    out = classify_stability("improving", 1.8, None)
    assert out["classification"] == "stable"
    assert out["contributions"]["total"] == 0


def test_small_cycle_revision_does_not_count():
    out = classify_stability("stable", None, 4)     # 4 points < 5 point rule
    assert out["classification"] == "stable"
    assert out["contributions"]["cycleRevision"] == 0
    assert out["contributions"]["cycleRevisionAvailable"] is True


def test_stability_carries_the_documented_rules():
    out = classify_stability("stable", 0.4, None)
    assert out["rules"] == RULES
