"""PHASE 5 - warning verification maths and the Operations Center routes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fbews.derived.warning import (
    CONFIDENCE_LTE,
    BUST_GTE,
    Counts,
    earliest_warning,
    scorecard_from_frames,
    select_cycles,
    summarise_earliest,
    warns,
    zone_masks,
)

from fbews.config import load_config  # noqa: E402

cfg = load_config()
pytestmark = pytest.mark.skipif(
    not (cfg.path("models") / "classifiers.joblib").exists(),
    reason="models not trained yet - run `make train`",
)


@pytest.fixture(scope="module")
def client():
    from fbews.api.main import app
    from fastapi.testclient import TestClient
    return TestClient(app)


# --------------------------------------------------------------------- warns
def test_warns_rule_matches_documented_thresholds():
    conf = np.array([CONFIDENCE_LTE - 1, CONFIDENCE_LTE, 50.0, 50.0])
    bust = np.array([0.1, BUST_GTE, BUST_GTE, BUST_GTE - 1e-9])
    w = warns(conf, bust)
    assert w.tolist() == [True, True, True, False]


def test_warns_counts_nan_as_no_warning():
    w = warns(np.array([np.nan]), np.array([np.nan]))
    assert w.tolist() == [False]


# ------------------------------------------------------------------- counts
def test_rates_use_counts_and_null_on_zero_denominator():
    c = Counts(hits=6, misses=4, false_alarms=3, correct_negatives=87)
    r = c.rates()
    assert r["accuracy"] == pytest.approx(0.93)
    assert r["precision"] == pytest.approx(0.6667, abs=1e-3)
    assert r["recall"] == pytest.approx(0.6)
    assert r["falseAlarmRatio"] == pytest.approx(0.3333, abs=1e-3)
    empty = Counts()
    assert all(v is None for v in empty.rates().values())


def test_scorecard_aggregates_leads_and_event_types():
    entries = [
        {"lead": 1,
         "warning": np.array([True, False, True, False]),
         "zone": np.array([True, True, False, False]),
         "kind_zones": {"tc": np.array([True, True, False, False])}},
        {"lead": 2,
         "warning": np.array([True, True, False, False]),
         "zone": np.array([False, True, False, False]),
         "kind_zones": {"tc": np.array([False, True, False, False])}},
    ]
    card = scorecard_from_frames(entries)
    assert card["overall"]["hits"] == 2
    assert card["overall"]["misses"] == 1
    assert card["overall"]["falseAlarms"] == 2
    assert card["overall"]["correctNegatives"] == 3
    assert card["overall"]["total"] == 8
    assert [r["lead"] for r in card["byLead"]] == [1, 2]
    tc = card["byEventType"][0]
    assert tc["eventType"] == "tc"
    assert tc["events"] == 3
    assert "confidence < 40" in card["rule"]


def test_scorecard_keeps_kinds_that_have_no_cells_out():
    entries = [{"lead": 1,
                "warning": np.zeros(4, dtype=bool),
                "zone": np.zeros(4, dtype=bool),
                "kind_zones": {}}]
    assert scorecard_from_frames(entries)["byEventType"] == []


# ------------------------------------------------------------ zone geometry
def test_zone_masks_respects_radius_and_merges_kinds():
    lats = np.arange(0.0, 5.0)
    lons = np.arange(0.0, 5.0)
    events = [{"lat": 2.0, "lon": 2.0, "radius_deg": 1.0, "kind": "a"},
              {"lat": 4.0, "lon": 4.0, "radius_deg": 0.5, "kind": "a"}]
    any_mask, by_kind = zone_masks(events, lats, lons)
    assert any_mask.reshape(5, 5)[2, 2]          # centre of the first radius
    assert not any_mask.reshape(5, 5)[0, 0]
    assert set(by_kind) == {"a"}
    # both events contribute to the same kind mask
    assert by_kind["a"].reshape(5, 5)[4, 4]


def test_distance_deg_zero_at_same_point_and_90_at_quarter_turn():
    from fbews.derived.warning import distance_deg
    assert distance_deg(np.array([10.0]), np.array([20.0]), 10.0, 20.0)[0] == 0.0
    assert distance_deg(np.array([0.0]), np.array([0.0]), 90.0, 0.0)[0] == pytest.approx(90.0)


# ------------------------------------------------------- earliest warning
def _blk(lat, lon, conf, bust, lead=1):
    return pd.DataFrame({"lat": lat, "lon": lon,
                         "confidence": conf, "bust_probability": bust,
                         "lead": lead})


def test_earliest_warning_returns_furthest_ahead_firing_lead():
    # cycle = event date - lead, so lead 7 and lead 3 both exist for 2024-07-13
    frames = {
        "2024-07-06": _blk([10.0], [20.0], [30.0], [0.9], lead=7),
        "2024-07-10": _blk([10.0], [20.0], [90.0], [0.1], lead=3),
    }
    out = earliest_warning({"date": "2024-07-13", "kind": "tc",
                            "lat": 10.0, "lon": 20.0, "depth_hpa": 990.0},
                           frames, list(frames))
    assert out["issued"] is True
    assert out["earliestWarningLead"] == 7          # furthest ahead, first fired
    assert out["leadsEvaluated"] == [3, 7]
    assert out["warningLeads"] == [7]
    assert out["hoursBeforeValid"] == 168


def test_earliest_warning_none_when_no_candidate_cycle_available():
    out = earliest_warning({"date": "2024-07-13", "kind": "tc",
                            "lat": 10.0, "lon": 20.0, "depth_hpa": 990.0},
                           {}, ["2024-01-01"])
    assert out is None


def test_earliest_warning_skips_cells_far_from_the_event():
    frames = {"2024-07-12": _blk([60.0], [60.0], [10.0], [0.99], lead=1)}
    out = earliest_warning({"date": "2024-07-13", "kind": "tc",
                            "lat": 10.0, "lon": 20.0, "depth_hpa": 990.0},
                           frames, list(frames))
    assert out is None


def test_summarise_earliest_reports_rates_and_by_kind():
    rows = [
        {"eventType": "tc", "issued": True, "earliestWarningLead": 4},
        {"eventType": "tc", "issued": True, "earliestWarningLead": 8},
        {"eventType": "wd", "issued": False, "earliestWarningLead": None},
    ]
    s = summarise_earliest(rows)
    assert s["eventsEvaluated"] == 3
    assert s["eventsWithWarning"] == 2
    assert s["detectionRate"] == pytest.approx(0.6667, abs=1e-3)
    assert s["bestLead"] == 8 and s["worstLead"] == 4
    kinds = {k["eventType"]: k for k in s["byEventType"]}
    assert kinds["tc"]["events"] == 2 and kinds["tc"]["maxEarliestLead"] == 8


# ----------------------------------------------------------- cycle picking
def test_select_cycles_covers_every_event_type_deterministically():
    avail = [f"2019-06-{d:02d}" for d in range(1, 29, 7)] + \
            [f"2019-07-{d:02d}" for d in range(1, 29, 7)]
    events = [{"date": "2019-06-05", "kind": "wd"},
              {"date": "2019-07-05", "kind": "tc"},
              {"date": "2019-07-06", "kind": "tc"}]
    picked = select_cycles(avail, events, 4)
    assert len(picked) == 4 == len(set(picked))
    assert set(picked) <= set(avail)
    assert picked == select_cycles(avail, events, 4)      # deterministic
    # round-robin guarantees a cycle covering the wd date is chosen
    assert any("2019-06" in c for c in picked)
    assert any("2019-07" in c for c in picked)


def test_select_cycles_never_exceeds_n():
    avail = [f"2019-06-{d:02d}" for d in range(1, 29)]
    assert len(select_cycles(avail, [], 3)) == 3


# ------------------------------------------------------------------- routes
def test_warning_performance_shape(client):
    r = client.get("/api/warning-performance?sample=4&limit=10")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["available"] is True
    assert "confidence < 40" in d["rule"]
    assert 1 <= len(d["cyclesSampled"]) <= 4
    sc = d["scorecard"]
    assert sc["overall"]["total"] == sum(
        sc["overall"][k] for k in
        ("hits", "misses", "falseAlarms", "correctNegatives"))
    assert [x["lead"] for x in sc["byLead"]] == sorted(
        x["lead"] for x in sc["byLead"])
    assert sc["byEventType"], "at least one event type must be reported"
    for row in sc["byEventType"]:
        assert row["eventType"] and row["total"] >= row["events"]
    assert d["coverage"]["cyclesUsed"] >= 1
    assert d["caveat"] and "synthetic" in d["caveat"].lower()
    es = d["earliest"]["summary"]
    assert es["eventsEvaluated"] == len(d["earliest"]["events"])
    assert es["eventsWithWarning"] <= es["eventsEvaluated"]


def test_event_impact_maps_event_types_to_regions(client):
    r = client.get("/api/event-impact")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["available"] is True
    assert d["rows"], "catalogue should produce impact rows"
    assert [x["events"] for x in d["rows"]] == sorted(
        [x["events"] for x in d["rows"]], reverse=True)
    names = {x["eventType"] for x in d["rows"]}
    assert {"western_disturbance", "monsoon_low", "tropical_cyclone"} <= names
    for row in d["rows"]:
        assert sum(row["regionCounts"].values()) <= row["events"]
        assert set(row["regions"]) == set(row["regionCounts"])
    assert d["method"] and "bounding box" in d["method"]


def test_event_impact_date_filters(client):
    r = client.get("/api/event-impact?start=2024-06-01&end=2024-06-30")
    assert r.status_code == 200, r.text
    assert all(x["eventType"] for x in r.json()["rows"])
