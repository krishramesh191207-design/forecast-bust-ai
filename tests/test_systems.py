"""Synoptic systems board - pure unit tests (no server, no model)."""
import pytest

from fbews.derived.systems import (
    IMPACT_CUTS,
    INFLUENCE_RADIUS_DEG,
    KIND_LABELS,
    build_board,
    cells_within,
    classify_impact,
    great_circle_deg,
    regime_summary,
)


def test_great_circle_zero_and_one_degree():
    assert great_circle_deg(20.0, 80.0, 20.0, 80.0) == 0
    assert great_circle_deg(20.0, 80.0, 21.0, 80.0) == pytest.approx(1.0, abs=1e-3)
    assert great_circle_deg(20.0, 80.0, 20.0, 85.0) == pytest.approx(4.86, abs=0.2)


def test_cells_within_radius():
    cells = [
        {"lat": 20.0, "lon": 80.0, "region": "a", "confidence": 50,
         "bust_probability": 0.4},
        {"lat": 20.0, "lon": 86.0, "region": "b", "confidence": 70,
         "bust_probability": 0.1},
    ]
    near = cells_within(20.0, 80.0, INFLUENCE_RADIUS_DEG, cells)
    assert [c["region"] for c in near] == ["a"]


def test_impact_cut_points_match_the_confidence_bands():
    assert classify_impact(None)[0] == "unknown"
    assert classify_impact(10)[0] == "high"
    assert classify_impact(39.99)[0] == "high"
    assert classify_impact(40)[0] == "moderate"
    assert classify_impact(59.9)[0] == "moderate"
    assert classify_impact(60)[0] == "low"
    assert classify_impact(79.9)[0] == "low"
    assert classify_impact(80)[0] == "minimal"
    assert [c[0] for c in IMPACT_CUTS] == [40.0, 60.0, 80.0, float("inf")]
    assert classify_impact(45)[1] == "Moderate impact"


def _cells():
    out = []
    for i in range(10):
        out.append({"lat": 20.0, "lon": 80.0 + i * 0.1, "region": "a",
                    "confidence": 30 + i, "bust_probability": 0.9 - i * 0.05})
    out.append({"lat": 10.0, "lon": 70.0, "region": "b",
                "confidence": 90, "bust_probability": 0.05})
    return out


def test_build_board_associates_regions_and_orders_by_impact():
    centres = [
        {"lat": 20.0, "lon": 80.0, "depth_hpa": 2.5, "kind": "monsoon_low"},
        {"lat": 10.0, "lon": 70.0, "depth_hpa": 6.0, "kind": "tropical_cyclone"},
    ]
    board = build_board(centres, [], _cells())
    assert len(board["systems"]) == 2
    # tropical cyclone sits over region "b" (confidence 90, minimal) while the
    # monsoon low sits over region "a" (confidence ~34, high) - high first
    assert board["systems"][0]["severity"] == "high"
    assert board["systems"][0]["type"] == "monsoon_low"
    assert board["systems"][1]["severity"] == "minimal"
    low = board["systems"][0]
    assert low["regionsAffected"] == ["a"]
    assert low["nearestRegion"] == "a"
    assert low["regionDetail"][0]["maxBustProbability"] == 0.9
    assert low["influenceRadiusDeg"] == INFLUENCE_RADIUS_DEG
    assert "forecast" in low["detectedFrom"]


def test_build_board_always_reports_open_features_as_not_detected():
    board = build_board([], [], _cells())
    assert board["systems"] == []
    assert "ridge" in board["notDetected"] and "trough" in board["notDetected"]
    assert "not" in board["notDetectedReason"].lower()
    assert "closed lows" in board["notDetectedReason"]


def test_every_detected_kind_has_a_label():
    for kind in ("tropical_cyclone", "monsoon_low", "western_disturbance",
                 "cyclonic_circulation"):
        assert KIND_LABELS[kind]


def test_regime_summary_shares_sum_to_100():
    rows = regime_summary({"normal": 60, "break_monsoon": 40})
    assert [r["regime"] for r in rows] == ["normal", "break_monsoon"]
    assert abs(sum(r["share"] for r in rows) - 100.0) < 0.2
    assert regime_summary({}) == []
