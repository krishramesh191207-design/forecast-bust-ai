"""Forecast Intelligence / Operations Center endpoint tests.

Skipped automatically when the model artefacts have not been built yet, so a
fresh clone can still run the unit suite.
"""
import pytest
from fastapi.testclient import TestClient

from fbews.config import load_config

cfg = load_config()
pytestmark = pytest.mark.skipif(
    not (cfg.path("models") / "classifiers.joblib").exists(),
    reason="models not trained yet - run `make train`",
)

CYCLE = "2024-07-13"


@pytest.fixture(scope="module")
def client():
    from fbews.api.main import app
    return TestClient(app)


def test_ensemble_summary_reads_real_quantiles(client):
    r = client.get(f"/api/ensemble?cycle={CYCLE}&lead=3&lat=19.0&lon=72.9")
    assert r.status_code == 200
    d = r.json()
    assert d["available"] is True
    assert d["cycle"] == CYCLE and d["lead"] == 3
    assert d["valid_time"] == "2024-07-16"
    assert d["location"]["lat"] is not None
    p = d["variables"]["precipitation"]
    assert p["available"] is True and p["unit"] == "mm/day"
    assert p["q25"] is not None and p["q75"] is not None
    assert p["q75"] >= p["q25"]
    assert p["iqr"] == pytest.approx(p["q75"] - p["q25"], abs=0.01)
    assert 0 <= p["extras"]["pop_gt25"] <= 1
    # members are reduced at generation time - the gap must be stated, not hidden
    assert d["members_available"] is False
    assert "member" in d["members_unavailable_reason"].lower()
    assert d["member_count"] == 11


def test_ensemble_reports_missing_cycle_honestly(client):
    r = client.get("/api/ensemble?cycle=1999-01-01&lead=3&lat=19.0&lon=72.9")
    assert r.status_code == 404          # unknown cycle, refused by the router
    assert "cycle" in str(r.json()["detail"]).lower()


def test_ensemble_missing_observation_is_not_invented(client):
    # a valid time with no truth file coverage still returns the forecast stats
    r = client.get(f"/api/ensemble?cycle={CYCLE}&lead=10&lat=5.5&lon=60.5")
    assert r.status_code == 200
    d = r.json()
    assert d["observed_available"] in (True, False)
    if not d["observed_available"]:
        assert d["observed_reason"]


def test_region_stability_covers_every_configured_region(client):
    r = client.get(f"/api/forecast-stability-regions?cycle={CYCLE}&lead=3")
    assert r.status_code == 200
    d = r.json()
    assert d["lead"] == 3 and d["cycle"] == CYCLE
    assert len(d["regions"]) >= 8
    seen = set()
    for row in d["regions"]:
        assert row["readout"]["stability"]["classification"]
        assert 0 <= row["readout"]["stability"]["score"] <= 100
        assert len(row["readout"]["trajectory"]["points"]) >= 2
        seen.add(row["region"])
    assert len(seen) == len(d["regions"])


def test_point_stability_route_still_matches_its_schema(client):
    r = client.get(f"/api/forecast-stability?cycle={CYCLE}&lead=3&lat=19&lon=72.9")
    assert r.status_code == 200
    d = r.json()
    assert d["leadDay"] == 3 and d["metric"] == "confidence"


def test_systems_board_detects_and_states_what_it_cannot(client):
    r = client.get(f"/api/systems?cycle={CYCLE}&lead=3")
    assert r.status_code == 200
    d = r.json()
    assert d["available"] is True
    assert d["validTime"] == "2024-07-16"
    assert "ridge" in d["notDetected"] and "trough" in d["notDetected"]
    assert d["notDetectedReason"]
    assert d["influenceRadiusDeg"] > 0
    assert d["regimeSummaryAvailable"] is True
    assert sum(r2["share"] for r2 in d["regimeSummary"]) == pytest.approx(100, abs=1.0)
    for s in d["systems"]:
        assert s["type"] in ("tropical_cyclone", "monsoon_low",
                             "western_disturbance", "cyclonic_circulation")
        assert s["severity"] in ("high", "moderate", "low", "minimal", "unknown")
        assert "other" not in s["regionsAffected"]
        assert "forecast" in s["detectedFrom"]
        assert "analy" not in s["detectedFrom"]        # never the event catalogue


def test_systems_unavailable_cycle_is_reported_not_guessed(client):
    r = client.get("/api/systems?cycle=1999-01-01&lead=3")
    assert r.status_code == 404 or (r.status_code == 200 and r.json()["available"] is False)


def test_watchlist_groups_covers_attention_rows(client):
    r = client.get(f"/api/watchlist-groups?cycle={CYCLE}")
    assert r.status_code == 200
    d = r.json()
    assert d["counts"]["rows"] > 0
    assert d["counts"]["attention"] <= d["counts"]["rows"]
    by_var = d["groups"]["byDominantVariable"]
    assert sum(len(v) for v in by_var.values()) == d["counts"]["attention"]
    for rows in by_var.values():
        for row in rows:
            assert row["bust_probability"] >= 0.25


def test_alerts_endpoint_shape_and_consistency(client):
    r = client.get(f"/api/alerts?cycle={CYCLE}")
    assert r.status_code == 200
    d = r.json()
    assert d["cycle"] == CYCLE and d["previousCycle"] is not None
    assert d["counts"]["total"] == len(d["alerts"])
    assert d["counts"]["groups"] == len(d["groups"])
    assert sum(d["bySeverity"].values()) == len(d["alerts"])
    assert len(d["rules"]) >= 6
    for a in d["alerts"]:
        assert a["status"] in ("new", "ongoing", "escalated")
        assert a["severity"] in ("info", "advisory", "warning", "critical")
        assert a["condition"]
        assert a["source"]
    for g in d["groups"]:
        assert g["count"] == len(g["regions"])


def test_alerts_lead_filter(client):
    r = client.get(f"/api/alerts?cycle={CYCLE}&lead=3")
    assert r.status_code == 200
    d = r.json()
    assert all(a["lead"] == 3 for a in d["alerts"])


def test_alerts_custom_rules_are_validated(client):
    bad = [{"id": "x", "label": "x", "metric": "nope", "operator": "lt",
            "threshold": 1, "lead": None, "severity": "warning",
            "enabled": True, "source": "user"}]
    r = client.post("/api/alerts/evaluate", json={"cycle": CYCLE, "rules": bad})
    assert r.status_code == 422

    ok = [{"id": "x", "label": "custom", "metric": "confidence", "operator": "lt",
           "threshold": 50, "lead": None, "severity": "info",
           "enabled": True, "source": "user"}]
    r = client.post("/api/alerts/evaluate", json={"cycle": CYCLE, "rules": ok})
    assert r.status_code == 200
    d = r.json()
    assert d["rules"][0]["id"] == "x"
    assert all(a["ruleId"] == "x" for a in d["alerts"])


def test_alerts_empty_rule_set_falls_back_to_defaults(client):
    r = client.post("/api/alerts/evaluate", json={"cycle": CYCLE, "rules": []})
    assert r.status_code == 200
    assert len(r.json()["rules"]) >= 6
