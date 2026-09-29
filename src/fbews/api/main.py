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
from ..derived.change import (
    REASON_MESSAGES,
    STATUS,
    STATUS_LABELS,
    aligned_change,
    available_leads,
    change_status,
    cycle_gap,
    lead_for_previous,
    metric_spec,
    previous_cycle,
    valid_time,
)
from ..derived.stability import (
    BUST_SLOPE_THRESHOLD,
    RULES,
    classify_stability,
    trajectory,
    volatility_signal,
)
from ..ingestion.sources import catalogue, probe_all
from ..inference.run import InferenceEngine, band
from ..preprocessing.quality import read_log
from .schemas import (
    ComparisonGrid,
    CycleChangeReadout,
    ForecastComparisonResponse,
    ForecastStabilityResponse,
    GridPointResponse,
    HealthResponse,
    InferenceRequest,
    Location,
    VolatilityReadout,
)

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


# ---------------------------------------------------------------------------
# FEATURE 1 - Forecast Drift map
# ---------------------------------------------------------------------------
def _comparison_unavailable(base: dict, reason: str, message: str, **extra) -> dict:
    return {**base, "available": False, "reason": reason, "message": message,
            "previousLead": None, "validTime": None, "grid": None,
            "current": [], "previous": [], "change": [], "status": [], **extra}


@app.get("/api/forecast-comparison", response_model=ForecastComparisonResponse,
         tags=["products"])
def forecast_comparison(cycle: str | None = None,
                        lead: int = Query(3, ge=1, le=10),
                        metric: str = Query("confidence")):
    """Cycle-to-cycle change of one layer, aligned on **valid time**.

    ``features/build.py`` defines ``valid = cycle + lead days``, so comparing
    ``cycle`` against its predecessor means using lead ``N`` now and lead
    ``N + gap`` then.  When that lead does not exist the payload reports
    ``available=false`` with a human-readable reason rather than returning a
    misleading "no change" map.
    """
    eng = engine()
    cur = _cycle_or_latest(cycle)
    spec = metric_spec(metric)
    if spec is None:
        raise HTTPException(422, f"unknown metric {metric!r}; see /api/meta -> layers")

    max_lead = max(cfg.lead_days)
    prev = previous_cycle(eng.available_cycles(), cur)
    base = dict(
        cycle=cur, previousCycle=prev, lead=lead, metric=spec.key,
        label=spec.label, units=spec.units, decimals=spec.decimals,
        higherIsBetter=spec.higher_is_better, threshold=spec.threshold,
        availableLeads=[],
        statusLabels={str(k): v for k, v in STATUS_LABELS.items()},
    )

    if prev is None:
        return _comparison_unavailable(base, "no_previous_cycle",
                                       REASON_MESSAGES["no_previous_cycle"])

    gap = cycle_gap(prev, cur)
    leads_now = sorted(int(l) for l in cfg.lead_days)
    usable = available_leads(prev, cur, leads_now)
    ctx = dict(gapDays=gap, availableLeads=usable,
               validTime=valid_time(cur, lead).isoformat())

    if spec.higher_is_better is None or spec.threshold is None:
        return _comparison_unavailable(base, "metric_not_defined",
                                       REASON_MESSAGES["metric_not_defined"], **ctx)

    prev_lead = lead_for_previous(prev, cur, lead, max_lead)
    if prev_lead is None:
        # `availableLeads` carries the "what *can* be compared" hint so the UI
        # renders it once, in one style, instead of repeating it here.
        msg = (f"Comparison unavailable for Day {lead}: the previous cycle {prev} is "
               f"{gap} day(s) earlier, so the same valid time "
               f"({valid_time(cur, lead).isoformat()}) needs Day {lead + gap} of that "
               f"cycle but only Day 1-{max_lead} exists.")
        return _comparison_unavailable(base, "same_valid_time_out_of_range", msg, **ctx)

    if spec.column not in eng.frame(cur).columns or spec.column not in eng.frame(prev).columns:
        return _comparison_unavailable(base, "metric_not_defined",
                                       REASON_MESSAGES["metric_not_defined"], **ctx)

    cur_rows = eng.frame(cur)
    prv_rows = eng.frame(prev)
    sc = cur_rows[cur_rows["lead"] == lead].sort_values(["lat", "lon"])
    sp = prv_rows[prv_rows["lead"] == prev_lead].sort_values(["lat", "lon"])
    lats = [float(v) for v in eng.assembler.lats]
    lons = [float(v) for v in eng.assembler.lons]

    def _bad(reason: str) -> dict:
        return _comparison_unavailable(base, reason, REASON_MESSAGES[reason], **ctx)

    if len(sc) == 0 or len(sc) != len(sp) or len(sc) != len(lats) * len(lons):
        return _bad("grids_not_aligned")
    if not np.allclose(sc[["lat", "lon"]].to_numpy(dtype="float64"),
                       sp[["lat", "lon"]].to_numpy(dtype="float64")):
        return _bad("grids_not_aligned")

    dec = spec.decimals

    def _round(series) -> list:
        arr = series.to_numpy(dtype="float64")
        return [None if not np.isfinite(v) else round(float(v), dec) for v in arr]

    change, status = aligned_change(sc[spec.column].to_numpy(dtype="float64"),
                                    sp[spec.column].to_numpy(dtype="float64"), spec)
    return {**base, **ctx, "available": True, "reason": None, "message": None,
            "previousLead": prev_lead,
            "grid": ComparisonGrid(lat=lats, lon=lons,
                                   shape=[len(lats), len(lons)],
                                   resolution_deg=float(cfg.resolution)),
            "current": _round(sc[spec.column]),
            "previous": _round(sp[spec.column]),
            "change": change, "status": status}


# ---------------------------------------------------------------------------
# FEATURE 2 - Forecast Stability + confidence trajectory
# ---------------------------------------------------------------------------
def _stability_at(eng: InferenceEngine, cur: str, lat: float, lon: float,
                  lead: int, metric: str) -> ForecastStabilityResponse:
    """Stability readout for one cell.  Shared by the point and region routes."""
    df = eng.frame(cur)
    nearest = int(((df["lat"] - lat) ** 2 + (df["lon"] - lon) ** 2).idxmin())
    cell_lat, cell_lon = float(df.at[nearest, "lat"]), float(df.at[nearest, "lon"])
    cell = df[(df["lat"] == cell_lat) & (df["lon"] == cell_lon)].sort_values("lead")
    leads = [int(v) for v in cell["lead"]]
    conf = [int(v) if np.isfinite(v) else None for v in cell["confidence"]]

    traj_conf = trajectory(conf, leads)
    if metric == "bust_probability":
        bp = [None if v is None else round(1 - v / 100.0, 3) for v in conf]
        traj = trajectory(bp, leads, higher_is_better=False,
                          slope_threshold=BUST_SLOPE_THRESHOLD)
    else:
        traj = traj_conf

    vol_at_lead = cell[cell["lead"] == lead]["vol_index"]
    vol_raw = float(vol_at_lead.iloc[0]) if len(vol_at_lead) else np.nan
    vol_index = round(vol_raw, 3) if np.isfinite(vol_raw) else None
    volatility = VolatilityReadout(
        index=vol_index, signal=volatility_signal(vol_index),
        low=RULES["volatility_low"], elevated=RULES["volatility_elevated"],
        available=vol_index is not None)

    # --- revision against the previous cycle, same valid time -------------
    max_lead = max(cfg.lead_days)
    prev = previous_cycle(eng.available_cycles(), cur)
    prev_lead = lead_for_previous(prev, cur, lead, max_lead) if prev else None
    cycle_change = CycleChangeReadout(
        available=False, current=None, previous=None, change=None,
        status=STATUS["unavailable"], label=STATUS_LABELS[STATUS["unavailable"]],
        previousCycle=prev, previousLead=prev_lead,
        validTime=valid_time(cur, lead).isoformat(), reason=None)

    cycle_delta = None
    cur_conf = conf[leads.index(lead)] if lead in leads else None
    if prev is None:
        cycle_change.reason = "no_previous_cycle"
    elif prev_lead is None:
        cycle_change.reason = "same_valid_time_out_of_range"
    else:
        prev_cell = eng.frame(prev)
        prv = prev_cell[(prev_cell["lat"] == cell_lat) & (prev_cell["lon"] == cell_lon)
                        & (prev_cell["lead"] == prev_lead)]["confidence"]
        prv_conf = int(prv.iloc[0]) if len(prv) and np.isfinite(prv.iloc[0]) else None
        if cur_conf is None or prv_conf is None:
            cycle_change.reason = "no_data"
        else:
            status = change_status(metric_spec("confidence"), cur_conf, prv_conf)
            cycle_delta = cur_conf - prv_conf
            cycle_change.available = True
            cycle_change.current = cur_conf
            cycle_change.previous = prv_conf
            cycle_change.change = cycle_delta
            cycle_change.status = status
            cycle_change.label = STATUS_LABELS[status]

    stability = classify_stability(traj_conf["classification"], vol_index, cycle_delta)
    return ForecastStabilityResponse(
        cycle=cur, previousCycle=prev,
        location=Location(lat=cell_lat, lon=cell_lon,
                          region=str(cell["region"].iloc[0]) if len(cell) else None),
        leadDay=lead, metric=metric, trajectory=traj, volatility=volatility,
        cycleChange=cycle_change, stability=stability)


@app.get("/api/forecast-stability", response_model=ForecastStabilityResponse,
         tags=["products"])
def forecast_stability(lat: float = Query(...), lon: float = Query(...),
                       cycle: str | None = None,
                       lead: int = Query(3, ge=1, le=10),
                       metric: str = Query("confidence",
                                           pattern="^(confidence|bust_probability)$")):
    """Stability class and lead-time trajectory for one grid cell.

    Everything returned is derived from values the pipeline already computes:
    the confidence/bust series across leads, the Forecast Volatility Index and
    the revision against the previous cycle at the same valid time.  See
    ``fbews.derived.stability`` for the documented thresholds.
    """
    d = cfg.domain
    if not (d["lat_min"] - 1 <= lat <= d["lat_max"] + 1
            and d["lon_min"] - 1 <= lon <= d["lon_max"] + 1):
        raise HTTPException(400, "coordinates outside the configured domain")
    return _stability_at(engine(), _cycle_or_latest(cycle), lat, lon, lead, metric)


@app.get("/api/forecast-stability-regions", tags=["products"])
def forecast_stability_regions(cycle: str | None = None,
                               lead: int = Query(3, ge=1, le=10)):
    """Stability readout at the centre of every configured analysis region.

    One frame load serves all regions; the centroid is snapped to the nearest
    grid cell by the same code path as the point route.
    """
    eng = engine()
    cur = _cycle_or_latest(cycle)
    out = []
    for r in load_regions():
        bb = r.get("bbox")
        if not bb or len(bb) != 4:
            continue
        la, lo = (bb[0] + bb[1]) / 2.0, (bb[2] + bb[3]) / 2.0
        try:
            ro = _stability_at(eng, cur, la, lo, lead, "confidence")
        except Exception:
            continue
        out.append({"region": r["id"], "name": r.get("name"), "kind": r.get("kind"),
                    "centroid": {"lat": round(la, 3), "lon": round(lo, 3)},
                    "readout": ro})
    return {"cycle": cur, "lead": lead, "regions": out}


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
# FORECAST INTELLIGENCE / OPERATIONS CENTER (mounted late so the shared
# helpers above are already defined when platform.py imports them)
# ---------------------------------------------------------------------------
from .platform import router as platform_router  # noqa: E402

app.include_router(platform_router)


# ---------------------------------------------------------------------------
if FRONTEND.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(str(FRONTEND / "index.html"))
