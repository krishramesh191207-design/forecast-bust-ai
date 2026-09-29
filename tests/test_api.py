"""API and trained-model integration tests.

These are skipped automatically when the model artefacts have not been built
yet, so a fresh clone can still run the unit suite.
"""
import pytest
from fastapi.testclient import TestClient

from fbews.config import load_config

cfg = load_config()
pytestmark = pytest.mark.skipif(
    not (cfg.path("models") / "classifiers.joblib").exists(),
    reason="models not trained yet - run `make train`",
)


@pytest.fixture(scope="module")
def client():
    from fbews.api.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["models_loaded"] is True


def test_cycles_and_products(client):
    cycles = client.get("/api/cycles").json()
    assert cycles["count"] > 0
    p = client.get(f"/api/products?cycle={cycles['latest']}").json()
    n = p["grid"]["shape"][0] * p["grid"]["shape"][1]
    for lead in p["meta"]["lead_days"]:
        layer = p["layers"][str(lead)]
        assert len(layer["confidence"]) == n
        assert all(v is None or 0 <= v <= 100 for v in layer["confidence"])
        assert all(v is None or 0 <= v <= 1 for v in layer["bust_probability"])


def test_confidence_is_complement_of_bust(client):
    p = client.get("/api/products").json()
    layer = p["layers"]["3"]
    for c, b in list(zip(layer["confidence"], layer["bust_probability"]))[:400]:
        assert abs(c - round(100 * (1 - b))) <= 1


def test_grid_point_schema_and_explanations(client):
    r = client.get("/api/grid/20.0/80.0?lead=4")
    assert r.status_code == 200
    d = r.json()
    assert 0 <= d["confidence"] <= 100
    assert 0 <= d["bust_probability"] <= 1
    assert d["lead_day"] == 4
    assert set(d["predicted_error"]) == {"precipitation", "temperature", "wind", "pressure"}
    assert all(v >= 0 for v in d["predicted_error"].values())
    assert isinstance(d["explanations"], list) and d["explanations"]
    assert len(d["series"]["lead"]) == len(d["series"]["confidence"])


def test_out_of_domain_is_rejected(client):
    assert client.get("/api/grid/80.0/10.0?lead=3").status_code == 400


def test_unknown_cycle_is_404(client):
    assert client.get("/api/products?cycle=1900-01-01").status_code == 404


def test_invalid_lead_is_422(client):
    assert client.get("/api/grid/20.0/80.0?lead=99").status_code == 422


def test_watchlist_sorted_and_flagged(client):
    rows = client.get("/api/watchlist").json()["rows"]
    probs = [r["bust_probability"] for r in rows]
    assert probs == sorted(probs, reverse=True)
    assert all(r["attention_required"] == (r["bust_probability"] >= 0.25) for r in rows)


def test_model_metrics_reports_baselines(client):
    m = client.get("/api/model-metrics").json()
    ladder = m["classifiers"]["ladder"]
    assert "climatology" in ladder and "gradient_boosting" in ladder
    assert ladder["gradient_boosting"]["pr_auc"] > ladder["climatology"]["pr_auc"]
    assert "caveat" in m


def test_demo_mode_is_labelled(client):
    meta = client.get("/api/meta").json()
    if meta["data_mode"] != "real":
        assert meta["demo"] is True
        assert "SYNTHETIC" in meta["banner"].upper() or "DEMO" in meta["banner"].upper()
        assert meta["warning"]


def test_post_inference(client):
    r = client.post("/api/inference", json={"lat": 19.0, "lon": 85.0, "lead": 6})
    assert r.status_code == 200 and r.json()["lead_day"] == 6


def test_case_study_timeline(client):
    r = client.get("/api/case-study/2024-07-21?lat=19.0&lon=85.0")
    if r.status_code == 404:
        pytest.skip("case-study cycles not generated")
    t = r.json()["timeline"]
    assert len(t) >= 2
    assert [d["lead_day"] for d in t] == sorted([d["lead_day"] for d in t], reverse=True)


def test_csv_export(client):
    r = client.get("/api/export/csv?lead=2")
    assert r.status_code == 200 and r.text.splitlines()[0].startswith("lat,lon,confidence")


# ---------------------------------------------------------------------------
# FEATURES 1 & 2 - forecast comparison and stability
# ---------------------------------------------------------------------------
def test_forecast_comparison_is_aligned_on_valid_time(client):
    import datetime as dt

    cycle = client.get("/api/cycles").json()["latest"]
    r = client.get(f"/api/forecast-comparison?cycle={cycle}&lead=3&metric=confidence")
    assert r.status_code == 200
    b = r.json()
    assert b["available"] is True and b["reason"] is None
    assert b["previousCycle"] < cycle
    assert b["gapDays"] == (dt.date.fromisoformat(cycle)
                            - dt.date.fromisoformat(b["previousCycle"])).days
    assert b["previousLead"] == b["lead"] + b["gapDays"]
    valid = dt.date.fromisoformat(cycle) + dt.timedelta(days=b["lead"])
    assert b["validTime"] == valid.isoformat()
    assert dt.date.fromisoformat(b["previousCycle"]) + dt.timedelta(days=b["previousLead"]) == valid
    assert b["availableLeads"] == [l for l in b["availableLeads"] if l <= b["lead"]]

    n = b["grid"]["shape"][0] * b["grid"]["shape"][1]
    assert len(b["change"]) == len(b["current"]) == len(b["previous"]) == len(b["status"]) == n
    assert set(b["status"]) <= {0, 1, 2, 3}
    assert b["statusLabels"] == {"0": "No Change", "1": "Improved",
                                 "2": "Deteriorated", "3": "No Data"}
    for c, p, d in zip(b["current"], b["previous"], b["change"]):
        if None not in (c, p, d):
            assert d == round(c - p, b["decimals"])


def test_forecast_comparison_refuses_leads_without_a_previous_valid_time(client):
    cycle = client.get("/api/cycles").json()["latest"]
    b = client.get(f"/api/forecast-comparison?cycle={cycle}&lead=9&metric=confidence").json()
    assert b["available"] is False
    assert b["reason"] == "same_valid_time_out_of_range"
    assert b["previousLead"] is None
    assert b["change"] == [] and b["status"] == []
    assert b["availableLeads"]            # some lead of this cycle IS comparable
    assert max(b["availableLeads"]) < 9
    assert "valid time" in b["message"]


def test_forecast_comparison_validates_inputs(client):
    cycle = client.get("/api/cycles").json()["latest"]
    assert client.get(f"/api/forecast-comparison?cycle=1900-01-01&lead=3").status_code == 404
    assert client.get(f"/api/forecast-comparison?cycle={cycle}&lead=99").status_code == 422
    assert client.get(f"/api/forecast-comparison?metric=not_a_layer").status_code == 422


def test_forecast_comparison_marks_non_directional_layers_undefined(client):
    cycle = client.get("/api/cycles").json()["latest"]
    b = client.get(f"/api/forecast-comparison?cycle={cycle}&lead=3&metric=fc_precip").json()
    assert b["available"] is False
    assert b["reason"] == "metric_not_defined"
    assert b["higherIsBetter"] is None and b["threshold"] is None


def test_forecast_stability_payload_and_classification(client):
    cycle = client.get("/api/cycles").json()["latest"]
    r = client.get(f"/api/forecast-stability?lat=21.0&lon=79.0&cycle={cycle}&lead=3")
    assert r.status_code == 200
    b = r.json()
    assert b["metric"] == "confidence" and b["leadDay"] == 3

    traj = b["trajectory"]
    assert [p["lead"] for p in traj["points"]] == sorted(p["lead"] for p in traj["points"])
    assert traj["classification"] in {"improving", "stable", "deteriorating",
                                      "insufficient_data"}
    assert traj["n"] == len(traj["points"])

    vol = b["volatility"]
    assert vol["signal"] in {"calm", "moderate", "elevated", "unavailable"}
    assert vol["available"] == (vol["index"] is not None)

    cc = b["cycleChange"]
    assert cc["validTime"]
    if cc["available"]:
        assert cc["change"] == cc["current"] - cc["previous"]
        assert cc["previousCycle"] < b["cycle"]
    else:
        assert cc["reason"] in {"no_previous_cycle", "same_valid_time_out_of_range", "no_data"}

    st = b["stability"]
    assert st["classification"] in {"improving", "stable", "deteriorating",
                                    "insufficient_data"}
    if st["score"] is not None:
        assert 0 <= st["score"] <= 100
    assert abs(sum(st["contributions"][k] for k in ("leadTrend", "cycleRevision", "volatility"))
               - st["contributions"]["total"]) <= 3
    assert st["rules"]["slope_points_per_lead"] == 1.0


def test_forecast_stability_validation(client):
    cycle = client.get("/api/cycles").json()["latest"]
    base = f"/api/forecast-stability?lat=21.0&lon=79.0&cycle={cycle}&lead=3"
    assert client.get(base + "&metric=nope").status_code == 422
    assert client.get(base.replace(cycle, "1900-01-01")).status_code == 404
    assert client.get(base.replace("lead=3", "lead=99")).status_code == 422
    assert client.get(f"/api/forecast-stability?lat=80.0&lon=10.0").status_code == 400


def test_forecast_stability_bust_metric_is_the_complement(client):
    cycle = client.get("/api/cycles").json()["latest"]
    a = client.get(f"/api/forecast-stability?lat=21.0&lon=79.0&cycle={cycle}&lead=3").json()
    b = client.get(f"/api/forecast-stability?lat=21.0&lon=79.0&cycle={cycle}&lead=3"
                   "&metric=bust_probability").json()
    assert b["metric"] == "bust_probability"
    assert len(a["trajectory"]["points"]) == len(b["trajectory"]["points"])
    for pa, pb in zip(a["trajectory"]["points"], b["trajectory"]["points"]):
        assert pa["lead"] == pb["lead"]
        assert pb["value"] == pytest.approx(1 - pa["value"] / 100, abs=1e-3)
    # stability itself is always decided on the Reliability Score
    assert a["stability"] == b["stability"]
