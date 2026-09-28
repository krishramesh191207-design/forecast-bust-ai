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
