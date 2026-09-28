"""FastAPI backend (PARTS 29, 30).

All endpoints are documented through OpenAPI at /docs. Responses follow the
schemas in schemas.py. CORS origins, host and port come from configuration;
no credentials are ever embedded.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import load_config, load_regions
from ..ingestion.sources import catalogue, probe_all
from ..inference.run import InferenceEngine, band
from ..preprocessing.quality import read_log
from .schemas import GridPointResponse, HealthResponse, InferenceRequest

cfg = load_config()
app = FastAPI(
    title="FBEWS - Forecast Bust Early Warning System",
    version=cfg["project"]["version"],
    description=(
        "Forecast reliability, bust probability and forecast-error prediction for "
        "medium-range NWP over India and the surrounding region.\n\n"
        "**Confidence is a derived operational indicator** (100 x (1 - bust probability)), "
        "not a probability that the forecast is correct."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg["api"]["cors_origins"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_engine: InferenceEngine | None = None
_hits: dict[str, deque] = defaultdict(deque)
FRONTEND = Path(__file__).resolve().parents[3] / "frontend"


def engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine(cfg)
    return _engine


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    limit = int(cfg["api"].get("rate_limit_per_minute", 120))
    key = request.client.host if request.client else "anon"
    now = time.time()
    q = _hits[key]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= limit:
        return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
    q.append(now)
    return await call_next(request)


_default_cycle_cache: str | None = None


def _latest_cycle() -> str:
    global _default_cycle_cache
    if _default_cycle_cache is not None:
        return _default_cycle_cache
    eng = engine()
    cycles = eng.available_cycles()
    if not cfg.is_sandbox:
        _default_cycle_cache = cycles[-1]
        return _default_cycle_cache
    # Sandbox: scan test-period cycles (last ~20%) to find the most interesting
    # one — highest mean bust probability — so the default view shows real
    # spatial variation rather than a quiet period.
    tail = cycles[max(0, int(len(cycles) * 0.8)):]
    best, best_bp = cycles[-1], 0.0
    for c in tail:
        try:
            mean_bp = float(eng.frame(c)["bust_probability"].mean())
            if mean_bp > best_bp:
                best_bp, best = mean_bp, c
        except Exception:
            continue
    _default_cycle_cache = best
    return best


def _cycle_or_latest(cycle: str | None) -> str:
    cycles = engine().available_cycles()
    if cycle is None:
        return _latest_cycle()
    if cycle not in cycles:
        raise HTTPException(404, f"cycle {cycle} not available; see /api/cycles")
    return cycle


def _banner() -> dict:
    return {
        "data_mode": cfg.data_mode,
        "demo": cfg.is_sandbox,
        "banner": "DEMO / SAMPLE DATA - SYNTHETIC SANDBOX" if cfg.is_sandbox else "REAL DATA",
        "warning": engine().manifest.get("warning", "") if cfg.is_sandbox else "",
    }


# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health():
    mdir = cfg.path("models")
    return {
        "status": "ok",
        "version": cfg["project"]["version"],
        "data_mode": cfg.data_mode,
        "models_loaded": (mdir / "classifiers.joblib").exists(),
        "cycles_available": len(engine().available_cycles()),
        "time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


@app.get("/api/meta", tags=["system"])
def meta():
    return {
        **_banner(),
        "project": cfg["project"],
        "domain": cfg.domain,
        "grid_resolution_deg": cfg.resolution,
        "lead_days": cfg.lead_days,
        "confidence_bands": cfg["confidence"]["bands"],
        "model_card": engine().card,
        "dataset_manifest": engine().manifest,
        "splits": cfg["splits"],
    }


@app.get("/api/cycles", tags=["forecast"])
def cycles():
    cs = engine().available_cycles()
    return {"cycles": cs, "latest": cs[-1], "count": len(cs)}


@app.get("/api/forecast/current", tags=["forecast"])
def forecast_current():
    c = _latest_cycle()
    p = engine().products(c)
    return {"meta": p["meta"], "kpi": p["kpi"], "watchlist": p["watchlist"][:8]}


@app.get("/api/products", tags=["forecast"])
def products(cycle: str | None = Query(None, description="YYYY-MM-DD forecast cycle")):
    return engine().products(_cycle_or_latest(cycle))


@app.get("/api/confidence", tags=["products"])
def confidence(cycle: str | None = None, lead: int = Query(3, ge=1, le=10)):
    p = engine().products(_cycle_or_latest(cycle))
    return {"meta": p["meta"], "grid": p["grid"], "lead_day": lead,
            "units": "0-100", "values": p["layers"][str(lead)]["confidence"]}


@app.get("/api/bust-probability", tags=["products"])
def bust_probability(cycle: str | None = None, lead: int = Query(3, ge=1, le=10)):
    p = engine().products(_cycle_or_latest(cycle))
    return {"meta": p["meta"], "grid": p["grid"], "lead_day": lead,
            "units": "probability", "values": p["layers"][str(lead)]["bust_probability"]}


@app.get("/api/error-prediction", tags=["products"])
def error_prediction(cycle: str | None = None, lead: int = Query(3, ge=1, le=10),
                     variable: str = Query("precipitation",
                                           pattern="^(precipitation|temperature|wind|pressure)$")):
    key = {"precipitation": "err_precip", "temperature": "err_t2m",
           "wind": "err_wind", "pressure": "err_mslp"}[variable]
    units = {"precipitation": "mm/day", "temperature": "K", "wind": "m/s", "pressure": "hPa"}[variable]
    p = engine().products(_cycle_or_latest(cycle))
    return {"meta": p["meta"], "grid": p["grid"], "lead_day": lead, "variable": variable,
            "units": units, "values": p["layers"][str(lead)][key]}


@app.get("/api/layers", tags=["products"])
def layers(cycle: str | None = None, lead: int = Query(3, ge=1, le=10)):
    p = engine().products(_cycle_or_latest(cycle))
    return {"grid": p["grid"], "lead_day": lead, "layers": p["layers"][str(lead)],
            "definitions": p["meta"]["layers"]}


@app.get("/api/regions", tags=["products"])
def regions(cycle: str | None = None, lead: int | None = None):
    p = engine().products(_cycle_or_latest(cycle))
    rows = p["regions"] if lead is None else [r for r in p["regions"] if r["lead_day"] == lead]
    return {"definitions": load_regions(), "rows": rows, "meta": p["meta"]}


@app.get("/api/watchlist", tags=["products"])
def watchlist(cycle: str | None = None):
    p = engine().products(_cycle_or_latest(cycle))
    return {"title": "Forecast Reliability Watchlist", "rows": p["watchlist"], "meta": p["meta"]}


@app.get("/api/grid/{lat}/{lon}", response_model=GridPointResponse, tags=["products"])
def grid_point(lat: float, lon: float, cycle: str | None = None,
               lead: int = Query(3, ge=1, le=10)):
    d = cfg.domain
    if not (d["lat_min"] - 1 <= lat <= d["lat_max"] + 1 and d["lon_min"] - 1 <= lon <= d["lon_max"] + 1):
        raise HTTPException(400, "coordinates outside the configured domain")
    return engine().point(_cycle_or_latest(cycle), lat, lon, lead)


@app.get("/api/explanations", tags=["explainability"])
def explanations(lat: float, lon: float, cycle: str | None = None,
                 lead: int = Query(3, ge=1, le=10)):
    pt = engine().point(_cycle_or_latest(cycle), lat, lon, lead)
    return {k: pt[k] for k in ("location", "lead_day", "confidence", "bust_probability",
                               "explanations", "explanation_detail",
                               "uncertainty_decomposition", "regime")}


@app.get("/api/analogs", tags=["explainability"])
def analogs(lat: float, lon: float, cycle: str | None = None, lead: int = Query(3, ge=1, le=10)):
    pt = engine().point(_cycle_or_latest(cycle), lat, lon, lead)
    return {"location": pt["location"], "lead_day": pt["lead_day"], "analogue": pt["analogue"]}


@app.post("/api/inference", tags=["products"])
def inference(req: InferenceRequest):
    return engine().point(_cycle_or_latest(req.cycle), req.lat, req.lon, req.lead)


@app.get("/api/model-metrics", tags=["evaluation"])
def model_metrics():
    mdir = cfg.path("models")
    out = {}
    for name, fn in (("classifiers", "metrics_classifiers.json"),
                     ("regressors", "metrics_regressors.json"),
                     ("feature_importance", "feature_importance.json"),
                     ("ablation", "ablation.json"),
                     ("model_card", "model_card.json"),
                     ("spatial_cv", "spatial_cv.json")):
        p = mdir / fn
        if p.exists():
            out[name] = json.loads(p.read_text())
    if not out:
        raise HTTPException(503, "no evaluation artefacts found - run `make train` first")
    out["caveat"] = (
        "Metrics are computed on the held-out chronological test split. In sandbox mode they "
        "validate the pipeline on synthetic data and are NOT evidence of skill on real forecasts."
    )
    return out


@app.get("/api/historical-events", tags=["events"])
def historical_events(limit: int = 200, kind: str | None = None,
                      start: str | None = None, end: str | None = None):
    path = cfg.path("processed") / "sandbox_events.csv"
    if not path.exists():
        raise HTTPException(503, "no event catalogue available")
    ev = pd.read_csv(path)
    if kind:
        ev = ev[ev["kind"] == kind]
    if start:
        ev = ev[ev["date"] >= start]
    if end:
        ev = ev[ev["date"] <= end]
    grouped = []
    for (kind_, ), grp in ev.groupby(["kind"]):
        peak = grp.sort_values("depth_hpa", ascending=False).head(limit)
        for _, r in peak.iterrows():
            grouped.append({"date": r["date"], "event_type": r["kind"],
                            "lat": float(r["lat"]), "lon": float(r["lon"]),
                            "intensity_hpa": float(r["depth_hpa"]),
                            "age_days": int(r["age_days"])})
    grouped.sort(key=lambda r: (-r["intensity_hpa"]))
    return {"source": ("sandbox synthetic system catalogue" if cfg.is_sandbox
                       else "IBTrACS v04r01 / IMD archives"),
            "count": len(grouped[:limit]), "events": grouped[:limit], **_banner()}


@app.get("/api/case-study/{date}", tags=["events"])
def case_study(date: str, lat: float = Query(...), lon: float = Query(...)):
    """Forecast evolution for one valid date as the event approached."""
    try:
        valid = dt.date.fromisoformat(date)
    except ValueError:
        raise HTTPException(400, "date must be YYYY-MM-DD")
    eng = engine()
    available = set(eng.available_cycles())
    timeline = []
    for back in range(10, 0, -1):
        c = (valid - dt.timedelta(days=back)).isoformat()
        if c not in available:
            continue
        try:
            pt = eng.point(c, lat, lon, lead=back)
        except Exception:
            continue
        timeline.append({
            "forecast_cycle": c,
            "lead_day": back,
            "hours_before_valid": back * 24,
            "confidence": pt["confidence"],
            "confidence_band": pt["confidence_band"],
            "bust_probability": pt["bust_probability"],
            "ensemble_spread_precip": pt["ensemble"]["precip_spread_mm_day"],
            "volatility_index": pt["volatility"]["index"],
            "forecast_precip_mm_day": pt["forecast"]["precipitation_mm_day"],
            "predicted_error_precip": pt["predicted_error"]["precipitation"],
            "observed_precip_mm_day": (pt["verification"] or {}).get("observed_precip_mm_day"),
            "observed_error_precip": (pt["verification"] or {}).get("observed_error_precip_mm_day"),
            "was_bust": (pt["verification"] or {}).get("was_bust"),
            "regime": pt["regime"],
            "top_reason": pt["explanations"][0] if pt["explanations"] else None,
        })
    if not timeline:
        raise HTTPException(404, "no forecast cycles verify at this date/location")
    return {
        "valid_date": date,
        "location": {"lat": lat, "lon": lon},
        "question": "Could the system have warned that this forecast was going to bust?",
        "timeline": timeline,
        **_banner(),
    }


@app.get("/api/datasets", tags=["system"])
def datasets(probe: bool = False):
    out = {"catalogue": catalogue(), **_banner()}
    if probe:
        out["availability"] = probe_all(timeout=2.0)
    return out


@app.get("/api/data-quality", tags=["system"])
def data_quality(limit: int = 50):
    return {"records": read_log(cfg.path("logs"), limit=limit)}


@app.get("/api/boundaries", tags=["system"])
def boundaries():
    p = cfg.path("samples") / "india_states_simplified.geojson"
    if not p.exists():
        raise HTTPException(503, "boundary file not built - run scripts/prepare_boundaries.py")
    return json.loads(p.read_text())


@app.get("/api/geo-labels", tags=["system"])
def geo_labels():
    """Structured geographic metadata used by the dashboard map labels.

    One source of truth for state capitals, reference cities and neighbouring
    country annotations: ``data/samples/india_geo_labels.json``.
    """
    p = cfg.path("samples") / "india_geo_labels.json"
    if not p.exists():
        raise HTTPException(503, "geo label file not built - see data/samples/india_geo_labels.json")
    return json.loads(p.read_text())


@app.get("/api/export/csv", tags=["products"])
def export_csv(cycle: str | None = None, lead: int = Query(3, ge=1, le=10)):
    c = _cycle_or_latest(cycle)
    p = engine().products(c)
    grid, layer = p["grid"], p["layers"][str(lead)]
    lines = ["lat,lon,confidence,bust_probability,err_precip,err_t2m,err_wind,err_mslp,ens_spread_precip,volatility"]
    k = 0
    for la in grid["lat"]:
        for lo in grid["lon"]:
            lines.append(",".join(str(x) for x in [
                la, lo, layer["confidence"][k], layer["bust_probability"][k],
                layer["err_precip"][k], layer["err_t2m"][k], layer["err_wind"][k],
                layer["err_mslp"][k], layer["ens_spread_precip"][k], layer["volatility"][k]]))
            k += 1
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse("\n".join(lines), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="fbews_{c}_D{lead}.csv"'})


# ---------------------------------------------------------------------------
if FRONTEND.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(str(FRONTEND / "index.html"))
