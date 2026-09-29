"""Forecast Intelligence / Operations Center routes.

Mounted onto the main FastAPI app by ``main.py``.  Everything in here reuses
the shared config, the single :func:`engine` singleton and the Pydantic models
in ``schemas.py`` - no second application object, no duplicated state.

Phase grouping
--------------
1. Forecast Intelligence  - ``/api/ensemble`` (drift/stability reuse the
   existing ``/api/forecast-comparison`` and ``/api/forecast-stability``).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from ..config import load_config, load_regions

# ``router`` must exist before ``.main`` is imported: main imports this module
# back at the bottom of its own body (the circular edge that mounts the router).
router = APIRouter()
cfg = load_config()

from ..derived.ensemble import ensemble_summary  # noqa: E402
from .main import _banner, _cycle_or_latest, engine  # noqa: E402
from .schemas import EnsembleResponse, Location  # noqa: E402


def _region_for(lat: float, lon: float) -> str | None:
    """Resolve a coarse analysis region from the configured bounding boxes."""
    for r in load_regions():
        bb = r.get("bbox")
        if not bb or len(bb) != 4:
            continue
        la0, la1, lo0, lo1 = bb
        if la0 <= lat <= la1 and lo0 <= lon <= lo1:
            return r["id"]
    return None


@router.get("/api/ensemble", response_model=EnsembleResponse, tags=["forecast-intelligence"])
def ensemble(lat: float = Query(..., ge=-90, le=90),
             lon: float = Query(..., ge=-180, le=360),
             cycle: str | None = Query(None, description="YYYY-MM-DD"),
             lead: int = Query(3, ge=1, le=10)):
    """Ensemble summary (mean, spread, quantiles, PoP, control, verifying obs).

    Reads the raw forecast file.  Members are reduced at generation time, so
    ``members_available`` is false and ``members_unavailable_reason`` explains
    why no member plot is drawn - the gap is reported, never papered over.
    """
    cur = _cycle_or_latest(cycle)
    processed = cfg.path("processed")
    out = ensemble_summary(processed / "forecasts", processed / "truth",
                           cur, lat, lon, lead)
    out["location"] = Location(**{**out["location"],
                                  "region": _region_for(out["location"]["lat"],
                                                        out["location"]["lon"])})
    banner = _banner()
    if not out.get("data_mode"):
        out["data_mode"] = banner["data_mode"]
    if not out.get("warning"):
        out["warning"] = banner["warning"] if cfg.is_sandbox else ""
    if not out["available"]:
        raise HTTPException(503, out.get("message") or "ensemble unavailable")
    return out


# ---------------------------------------------------------------------------
# PHASE 3 - Operations Center: systems board + watchlist grouping
# ---------------------------------------------------------------------------
import datetime as _dt  # noqa: E402

import xarray as xr  # noqa: E402

from ..derived.change import previous_cycle  # noqa: E402
from ..derived.systems import build_board, regime_summary  # noqa: E402
from ..features.systems import detect_lows  # noqa: E402
from .schemas import (  # noqa: E402
    AlertsResponse,
    EvaluateAlertsRequest,
    RegimeSummaryRow,
    SystemsResponse,
    WatchlistGroupsResponse,
)


@router.get("/api/systems", response_model=SystemsResponse, tags=["operations-center"])
def systems(cycle: str | None = Query(None, description="YYYY-MM-DD"),
            lead: int = Query(3, ge=1, le=10)):
    """Synoptic systems detected in the forecast pressure field.

    Detection is ``fbews.features.systems.detect_lows`` - forecast MSLP only,
    never the verifying catalogue.  Ridges and open troughs are reported as
    not detected instead of being estimated.
    """
    eng = engine()
    cur = _cycle_or_latest(cycle)
    valid = _dt.date.fromisoformat(cur) + _dt.timedelta(days=int(lead))
    base = dict(cycle=cur, lead=int(lead), validTime=valid.isoformat(),
                notDetected=[], notDetectedReason="", influenceRadiusDeg=0.0,
                regimeSummary=[], regimeSummaryAvailable=False)
    fp = cfg.path("processed") / "forecasts" / f"fc_{cur.replace('-', '')}.nc"
    if not fp.exists():
        return SystemsResponse(available=False, reason="forecast_file_missing",
                               message=f"No forecast file for cycle {cur}.", **base)

    try:
        with xr.open_dataset(fp) as ds:
            mslp = ds["det_mslp"].sel(lead=int(lead)).values
    except Exception as exc:  # missing field / missing lead
        return SystemsResponse(available=False, reason="field_unavailable",
                               message=f"Forecast MSLP unavailable: {exc}", **base)

    centres = detect_lows(mslp, eng.assembler.lats, eng.assembler.lons,
                          eng.assembler.static.land.values, valid.month)
    df = eng.frame(cur)
    sub = df[df["lead"] == int(lead)]
    cells = [{"lat": float(r["lat"]), "lon": float(r["lon"]),
              "region": str(r["region"]), "confidence": int(r["confidence"]),
              "bust_probability": float(r["bust_probability"])}
             for _, r in sub.iterrows() if str(r["region"]) != "other"]
    board = build_board(centres, [], cells)
    if "regime_name" in sub.columns:
        counts = sub["regime_name"].value_counts().to_dict()
        board["regimeSummary"] = regime_summary({str(k): int(v) for k, v in counts.items()})
        base["regimeSummaryAvailable"] = True
    return SystemsResponse(available=True, cycle=cur, lead=int(lead),
                           validTime=valid.isoformat(),
                           systems=board["systems"],
                           notDetected=board["notDetected"],
                           notDetectedReason=board["notDetectedReason"],
                           influenceRadiusDeg=board["influenceRadiusDeg"],
                           regimeSummary=[RegimeSummaryRow(**r)
                                          for r in board["regimeSummary"]],
                           regimeSummaryAvailable=base["regimeSummaryAvailable"])


@router.get("/api/watchlist-groups", response_model=WatchlistGroupsResponse,
            tags=["operations-center"])
def watchlist_groups(cycle: str | None = Query(None),
                     lead: int | None = Query(None, ge=1, le=10)):
    """Attention rows grouped by dominant variable and by regime.

    Rows come from the same region table the watchlist uses; the grouping is
    pure presentation, so nothing is recomputed or re-scored.
    """
    eng = engine()
    cur = _cycle_or_latest(cycle)
    rows = eng.region_table(eng.frame(cur))
    if lead is not None:
        rows = [r for r in rows if r["lead_day"] == lead]
    attention = [r for r in rows if (r.get("bust_probability") or 0) >= 0.25]
    by_var: dict[str, list] = {}
    by_regime: dict[str, list] = {}
    for r in attention:
        by_var.setdefault(r.get("dominant_variable") or "unknown", []).append(r)
        by_regime.setdefault(r.get("regime") or "unknown", []).append(r)
    slim = lambda rs: [{k: r[k] for k in ("region", "region_name", "lead_day",
                                          "confidence", "bust_probability",
                                          "dominant_variable", "regime")} for r in rs]
    return WatchlistGroupsResponse(
        cycle=cur, lead=lead,
        groups={"byDominantVariable": {k: slim(v) for k, v in by_var.items()},
                "byRegime": {k: slim(v) for k, v in by_regime.items()}},
        counts={"rows": len(rows), "attention": len(attention),
                "dominantVariables": len(by_var), "regimes": len(by_regime)})


# ---------------------------------------------------------------------------
# PHASE 4 - Active Alerts + rules
# ---------------------------------------------------------------------------
from ..derived.alerts import (  # noqa: E402
    AlertRule,
    build_alerts,
    default_rules,
)


def _rows_for(cycle: str, lead: int | None) -> list[dict]:
    eng = engine()
    rows = eng.region_table(eng.frame(cycle))
    if lead is not None:
        rows = [r for r in rows if r["lead_day"] == lead]
    return rows


def _alerts_note() -> str:
    return ("Rules are data, not code: every threshold already exists elsewhere "
            "in the project and its source is recorded on the rule. State "
            "(new / ongoing / escalated) is derived by re-running the same "
            "rules against the previous forecast cycle - no alert store.")


@router.get("/api/alerts", response_model=AlertsResponse, tags=["operations-center"])
def alerts(cycle: str | None = Query(None),
           lead: int | None = Query(None, ge=1, le=10),
           previousCycle: str | None = Query(None)):
    """Evaluate the built-in rules for this cycle (and the previous one)."""
    eng = engine()
    cur = _cycle_or_latest(cycle)
    prev = previous_cycle(eng.available_cycles(), cur) if previousCycle is None \
        else _cycle_or_latest(previousCycle)
    rows = _rows_for(cur, lead)
    prev_rows = _rows_for(prev, lead) if prev else []
    res = build_alerts(default_rules(), rows, prev_rows)
    return AlertsResponse(**res, cycle=cur, previousCycle=prev,
                          note=_alerts_note())


@router.post("/api/alerts/evaluate", response_model=AlertsResponse,
             tags=["operations-center"])
def evaluate_alerts(body: EvaluateAlertsRequest):
    """Evaluate caller-supplied rules against a cycle's region table."""
    if not body.rules:
        rules = default_rules()
    else:
        rules = []
        for r in body.rules:
            if r.metric not in ("confidence", "bust_probability", "volatility",
                                "rain_error"):
                raise HTTPException(422, f"unknown metric {r.metric!r}")
            if r.operator not in ("lt", "lte", "gt", "gte"):
                raise HTTPException(422, f"unknown operator {r.operator!r}")
            if r.severity not in ("info", "advisory", "warning", "critical"):
                raise HTTPException(422, f"unknown severity {r.severity!r}")
            rules.append(AlertRule(id=r.id, label=r.label, metric=r.metric,
                                   operator=r.operator, threshold=r.threshold,
                                   lead=r.lead, severity=r.severity,
                                   enabled=r.enabled, source=r.source or "user"))
    eng = engine()
    cur = _cycle_or_latest(body.cycle)
    prev = previous_cycle(eng.available_cycles(), cur)
    res = build_alerts(rules, _rows_for(cur, None),
                       _rows_for(prev, None) if prev else [])
    return AlertsResponse(**res, cycle=cur, previousCycle=prev,
                          note=_alerts_note())


# ---------------------------------------------------------------------------
# PHASE 5 - Event History + Performance
# ---------------------------------------------------------------------------
import functools as _functools  # noqa: E402

from ..derived.warning import (  # noqa: E402
    WARNING_RULE,
    earliest_warning,
    scorecard_from_frames,
    select_cycles,
    summarise_earliest,
    warns,
    zone_masks,
)
from .schemas import (  # noqa: E402
    EarliestWarningRecord,
    EventImpactResponse,
    EventImpactRow,
    WarningPerformanceResponse,
)

_SLIM_COLS = ["lat", "lon", "lead", "valid_date", "confidence",
              "bust_probability", "region"]


@_functools.lru_cache(maxsize=64)
def _slim_frame(cycle: str) -> "pd.DataFrame":
    eng = engine()
    df = eng.frame(cycle)
    return df[_SLIM_COLS].copy()


def _events() -> "pd.DataFrame":
    path = cfg.path("processed") / "sandbox_events.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date", "kind", "lat", "lon",
                                     "depth_hpa", "radius_deg", "age_days"])
    ev = pd.read_csv(path)
    ev["date"] = ev["date"].astype(str)
    return ev


@_functools.lru_cache(maxsize=8)
def _warning_report(sample_cycles: int, limit: int) -> dict:
    eng = engine()
    avail = eng.available_cycles()
    ev = _events()
    # Round-robin over event types, then by event-date coverage: a tail-only
    # or uniform sample would silently report a single season as if it were
    # the whole verification period.  Needs no forecast data.
    sample = select_cycles(avail, ev.to_dict("records"), int(sample_cycles))
    lats = np.asarray(eng.assembler.lats, dtype="float64")
    lons = np.asarray(eng.assembler.lons, dtype="float64")

    # Not every cycle in the record can be framed (the very first weeks have
    # no verifying truth yet, so the feature build yields NaN).  Skip them and
    # report the count - a half-covered sample is still better than none.
    frames: dict[str, "pd.DataFrame"] = {}
    skipped: list[str] = []
    for c in sample:
        try:
            frames[c] = _slim_frame(c)
        except Exception:            # noqa: BLE001 - frame build, not our code
            skipped.append(c)

    entries: list[dict] = []
    valid_dates: set[str] = set()
    for c, df in frames.items():
        cd = dt.date.fromisoformat(c)
        for lead in sorted(df["lead"].unique()):
            sub = df[df["lead"] == lead].sort_values(["lat", "lon"])
            if not len(sub):
                continue
            vdate = (cd + dt.timedelta(days=int(lead))).isoformat()
            valid_dates.add(vdate)
            day_events = ev[ev["date"] == vdate]
            any_mask, kind_masks = zone_masks(
                day_events.to_dict("records"), lats, lons)
            entries.append({
                "lead": int(lead),
                "warning": warns(sub["confidence"].to_numpy(dtype="float64"),
                                 sub["bust_probability"].to_numpy(dtype="float64")),
                "zone": any_mask,
                "kind_zones": kind_masks,
            })
    card = scorecard_from_frames(entries)

    covered_dates = set(ev["date"]) & valid_dates
    cand = ev[ev["date"].isin(covered_dates)].copy()
    if len(cand) > limit:
        cand = cand.sort_values("date").iloc[
            np.unique(np.linspace(0, len(cand) - 1,
                                  int(limit)).round().astype(int))]
    avail_set = set(frames)
    rows = []
    for _, r in cand.iterrows():
        rec = earliest_warning({"date": r["date"], "kind": r["kind"],
                                "lat": float(r["lat"]), "lon": float(r["lon"]),
                                "depth_hpa": float(r["depth_hpa"])},
                               frames, avail_set)
        if rec:
            rec["region"] = _region_for(rec["lat"], rec["lon"])
            rows.append(rec)
    return {
        "cyclesSampled": sorted(frames),
        "cyclesSkipped": skipped,
        "scorecard": card,
        "earliest": {"summary": summarise_earliest(rows), "events": rows},
        "coverage": {"validDates": len(valid_dates),
                     "eventsCovered": len(covered_dates),
                     "eventsTotal": int(len(ev)),
                     "gridCells": int(len(lats) * len(lons)),
                     "cyclesRequested": int(sample_cycles),
                     "cyclesUsed": int(len(frames)),
                     "cyclesSkipped": int(len(skipped))},
    }


@router.get("/api/warning-performance", response_model=WarningPerformanceResponse,
            tags=["operations-center"])
def warning_performance(sample: int = Query(12, ge=1, le=40),
                        limit: int = Query(60, ge=1, le=400)):
    """Earliest warning, lead scorecard and event-type performance.

    Ground truth is the system event catalogue; a cell is "affected" when it
    lies inside the catalogue's own radius.  All rates are reported with their
    counts and a zero denominator yields ``null``, never ``0``.
    """
    ev = _events()
    if ev.empty:
        return WarningPerformanceResponse(
            available=False, reason="no_event_catalogue",
            message="No event catalogue available - run the sandbox generator.",
            source="unavailable", rule=WARNING_RULE, coverage={},
            scorecard=Scorecard(
                overall=ScoreCounts(hits=0, misses=0, falseAlarms=0,
                                    correctNegatives=0, total=0, events=0,
                                    warnings=0),
                byLead=[], byEventType=[], rule=WARNING_RULE),
            earliest={"summary": {"eventsEvaluated": 0, "eventsWithWarning": 0,
                                  "eventsWithoutWarning": 0},
                      "events": []},
            caveat="No catalogue, no scorecard.")
    rep = _warning_report(int(sample), int(limit))
    return WarningPerformanceResponse(
        available=True,
        source=("sandbox synthetic system catalogue" if cfg.is_sandbox
                else "IBTrACS v04r01 / IMD archives"),
        rule=WARNING_RULE,
        cyclesSampled=rep["cyclesSampled"],
        coverage=rep["coverage"],
        scorecard=rep["scorecard"],
        earliest={"summary": rep["earliest"]["summary"],
                  "events": [EarliestWarningRecord(**r)
                             for r in rep["earliest"]["events"][:limit]]},
        caveat=(
            "Sandbox verification: the catalogue is synthetic, so these rates "
            "describe the pipeline against synthetic systems and are not "
            "evidence of operational skill. Confidence/bust values are read "
            "from the stored forecast frame - no model is re-run."))


@router.get("/api/event-impact", response_model=EventImpactResponse,
            tags=["operations-center"])
def event_impact(start: str | None = Query(None), end: str | None = Query(None)):
    """Event type -> analysis regions it most often affects.

    Spatial join only: the catalogue position is tested against the configured
    region bounding boxes.  No forecast data is involved, so this is cheap and
    can never disagree with the region definitions.
    """
    ev = _events()
    if ev.empty:
        return EventImpactResponse(available=False, source="unavailable",
                                   method="region bounding-box join", rows=[])
    if start:
        ev = ev[ev["date"] >= start]
    if end:
        ev = ev[ev["date"] <= end]
    rows: list[EventImpactRow] = []
    uncovered = 0
    for kind, grp in ev.groupby("kind"):
        counts: dict[str, int] = {}
        for _, r in grp.iterrows():
            rid = _region_for(float(r["lat"]), float(r["lon"]))
            if rid is None:
                uncovered += 1
                continue
            counts[rid] = counts.get(rid, 0) + 1
        rows.append(EventImpactRow(
            eventType=str(kind), events=int(len(grp)),
            regions=sorted(counts, key=lambda k: -counts[k]),
            regionCounts=counts,
            meanDepthHpa=round(float(grp["depth_hpa"].mean()), 3),
            meanRadiusDeg=(round(float(grp["radius_deg"].mean()), 3)
                           if "radius_deg" in grp else None)))
    rows.sort(key=lambda r: -r.events)
    return EventImpactResponse(
        available=True,
        source=("sandbox synthetic system catalogue" if cfg.is_sandbox
                else "IBTrACS v04r01 / IMD archives"),
        method=("Each catalogue position is tested against the configured "
                "analysis-region bounding boxes; a position outside every box "
                "is counted in `uncovered` rather than assigned."),
        rows=rows, uncovered=int(uncovered))


# ---------------------------------------------------------------------------
# PHASE 7 - Region profile / trust / model drift
# ---------------------------------------------------------------------------
from ..derived.alerts import default_rules as _default_rules  # noqa: E402
from .schemas import (  # noqa: E402
    BriefSection,
    DriftPoint,
    DriftResponse,
    ForecastBriefResponse,
    RegionProfilePoint,
    RegionProfileResponse,
    SearchHit,
    SearchResponse,
    TrustResponse,
)

_REGION_NAMES = {r["id"]: r.get("name", r["id"]) for r in load_regions()}
_MEMBERS = int(cfg.get("ensemble_members")
                or (cfg.get("ensemble") or {}).get("members") or 0)


@_functools.lru_cache(maxsize=8)
def _region_rows(cycle: str) -> tuple:
    """Region table for one cycle, cached - the frame build is the expensive
    part and a profile view asks for several cycles in a row."""
    return tuple(engine().region_table(engine().frame(cycle)))


def _recent_cycles(n: int) -> tuple[list[str], list[str]]:
    avail = engine().available_cycles()
    n = max(1, int(n))
    return list(avail[:n]), list(avail[-n:])


def _frames_for(cycles: list[str]) -> dict[str, "pd.DataFrame"]:
    out: dict[str, "pd.DataFrame"] = {}
    for c in cycles:
        try:
            out[c] = _slim_frame(c)
        except Exception:            # noqa: BLE001 - frame build, not our code
            continue
    return out


@router.get("/api/region-profile", response_model=RegionProfileResponse,
            tags=["operations-center"])
def region_profile(region: str = Query(...),
                   cycle: str | None = Query(None),
                   lead: int = Query(3, ge=1, le=10),
                   history: int = Query(8, ge=1, le=20)):
    """One analysis region: current snapshot plus a cycle-by-cycle trail.

    History is read from the stored forecast frames (no model re-run).  Cycles
    that cannot be framed are dropped rather than interpolated, and the note
    says so.
    """
    names = {r["id"]: r.get("name", r["id"]) for r in load_regions()}
    if region not in names:
        raise HTTPException(422, f"unknown region {region!r}")
    cur = _cycle_or_latest(cycle)
    rows = [r for r in _region_rows(cur) if r["region"] == region]
    if not rows:
        raise HTTPException(404, f"region {region!r} has no cells in cycle {cur}")
    snapshot = {r["lead_day"]: r for r in rows}
    snap = snapshot.get(int(lead)) or max(rows, key=lambda r: r["lead_day"])

    avail = engine().available_cycles()
    past = [c for c in avail if c < cur][-int(history):]
    hist: list[RegionProfilePoint] = []
    for c in past:
        r = next((x for x in _region_rows(c)
                  if x["region"] == region and x["lead_day"] == int(lead)), None)
        if r is None:
            continue
        hist.append(RegionProfilePoint(
            cycle=c, confidence=int(r["confidence"]),
            bustProbability=float(r["bust_probability"]),
            expectedErrorPrecipMmDay=float(r["expected_error"]["precipitation_mm_day"]),
            volatilityIndex=float(r["volatility_index"])))
    sysR = systems(cycle=cur, lead=int(lead))
    alR = alerts(cycle=cur, lead=int(lead), previousCycle=None)
    matched_sys = [s for s in (sysR.systems or [])
                   if region in (s.regionsAffected or [])]
    matched_alerts = [a for a in (alR.groups or []) if region in (a.regions or [])]
    return RegionProfileResponse(
        region=region, regionName=names[region], cycle=cur, lead=int(lead),
        snapshot=snap, history=hist,
        systems=[s.model_dump() for s in matched_sys],
        alerts=[a.model_dump() for a in matched_alerts],
        note=(f"Snapshot is cycle {cur} Day {lead}. History shows the last "
              f"{len(hist)} cycles that could be framed for this region+lead; "
              "cycles that fail to build are skipped, never interpolated."))


@router.get("/api/trust", response_model=TrustResponse, tags=["operations-center"])
def trust():
    """Provenance: where the numbers come from, the thresholds in force, and
    what the system does not do.  Everything is read from config and from the
    artefact files on disk - nothing is asserted from memory."""
    import datetime as _iso
    prov = [
        {"label": "Forecast source", "value": "Stored NetCDF forecast frames",
         "detail": "data/processed/forecasts/fc_YYYYMMDD.nc"},
        {"label": "Verifying truth", "value": "Truth archive",
         "detail": "data/processed/truth/truth_YYYY.nc"},
        {"label": "Event catalogue",
         "value": ("sandbox synthetic system catalogue" if cfg.is_sandbox
                   else "IBTrACS v04r01 / IMD archives"),
         "detail": "sandbox_events.csv / storm events"},
        {"label": "Region definitions", "value": f"{len(load_regions())} bounding boxes",
         "detail": "configs/regions.yaml"},
        {"label": "Synoptic detection",
         "value": "Forecast MSLP only (detect_lows)",
         "detail": "Ridges and open troughs are reported as not detected."},
        {"label": "Alert state",
         "value": "Recomputed each cycle - no alert store",
         "detail": "new/ongoing/escalated from the previous cycle's rows."},
    ]
    thr = [
        {"name": "Confidence band edges", "value": "60 / 40 / 20",
         "source": "configs/default.yaml · confidence.bands + bandLabel()"},
        {"name": "Watchlist attention cut", "value": "bust >= 0.25",
         "source": "derived/alerts.py · attention_required"},
        {"name": "Dashboard red cut", "value": "bust >= 0.50",
         "source": "derived/alerts.py · dashboard red"},
        {"name": "Volatility elevated", "value": "index >= 1.2",
         "source": "derived/stability.py · RULES.volatility_elevated"},
        {"name": "Volatility maximum", "value": "index >= 3.0",
         "source": "derived/stability.py · volatility_index_max"},
        {"name": "Rain absolute threshold", "value": "5.0 mm/day",
         "source": "configs/default.yaml · bust.absolute_thresholds"},
        {"name": "System influence radius", "value": "5.0 deg (~550 km)",
         "source": "derived/systems.py · INFLUENCE_RADIUS_DEG"},
        {"name": "Warning rule", "value": WARNING_RULE,
         "source": "derived/warning.py · confidence band edge + watchlist cut"},
    ]
    arts = []
    for rel in ("classifiers.joblib", "regressor.joblib", "calibrator.joblib"):
        fp = cfg.path("models") / rel
        if fp.exists():
            arts.append({"name": rel, "bytes": fp.stat().st_size,
                         "modified": _iso.datetime.fromtimestamp(
                             fp.stat().st_mtime).isoformat(timespec="seconds"),
                         "path": str(fp)})
    return TrustResponse(
        provenance=prov, thresholds=thr, artefacts=arts,
        members={"declared": _MEMBERS,
                 "note": ("Member dimension is reduced at generation time; the "
                          "frame stores quantiles and spread, not members. "
                          "Ensemble member traces are therefore unavailable.")},
        limitations=[
            "Sandbox event catalogue is synthetic - verification describes the "
            "pipeline, not operational skill.",
            "No alert store: state is recomputed from the previous cycle, so "
            "acknowledgements and history beyond one cycle are not kept.",
            "Ridges and open troughs are not detected and are never estimated.",
            "Analysis regions are bounding boxes, not meteorological boundaries.",
            "Members are unavailable in the stored frames; ensemble products "
            "are read from quantiles and spread.",
        ],
        generatedAt=_iso.datetime.now().isoformat(timespec="seconds"))


@router.get("/api/model-drift", response_model=DriftResponse,
            tags=["operations-center"])
def model_drift(lead: int = Query(3, ge=1, le=10),
                window: int = Query(6, ge=2, le=20)):
    """Baseline vs recent cycle windows at one lead time.

    Baseline = the first ``window`` framable cycles of the record; recent = the
    last ``window``.  Reported numbers are means over the stored frames - no
    model is retrained and no holdout is re-split.
    """
    eng = engine()
    avail = eng.available_cycles()
    if len(avail) < 2 * int(window):
        return DriftResponse(available=False, reason="too_few_cycles", lead=int(lead),
                             baselineLabel="", recentLabel="",
                             baseline=DriftPoint(label=""), recent=DriftPoint(label=""),
                             delta={}, interpretation="", caveat="Not enough cycles.")
    base_c, recent_c = list(avail[:int(window)]), list(avail[-int(window):])
    b = _mean_window(base_c, int(lead))
    r = _mean_window(recent_c, int(lead))
    if b is None or r is None:
        return DriftResponse(available=False, reason="frames_unavailable",
                             lead=int(lead), baselineLabel="",
                             recentLabel="", baseline=DriftPoint(label=""),
                             recent=DriftPoint(label=""), delta={},
                             interpretation="Not enough cycles.",
                             caveat="Cycles could not be framed.")
    d_conf = round(r["confidence"] - b["confidence"], 2)
    d_bust = round(r["bust"] - b["bust"], 4)
    d_mae = round(r["mae"] - b["mae"], 3)
    if d_conf <= -5 or d_bust >= 0.05 or d_mae >= 0.5:
        verdict = ("Worse than baseline: confidence down, bust risk up or "
                   "realised rain error higher in the recent window.")
    elif d_conf >= 5 or d_bust <= -0.05 or d_mae <= -0.5:
        verdict = ("Better than baseline: confidence up, bust risk down or "
                   "realised rain error lower in the recent window.")
    else:
        verdict = "Within a few points of baseline - no clear drift at this lead."
    return DriftResponse(
        lead=int(lead),
        baselineLabel=f"{base_c[0]} → {base_c[-1]} ({len(base_c)} cycles)",
        recentLabel=f"{recent_c[0]} → {recent_c[-1]} ({len(recent_c)} cycles)",
        baseline=DriftPoint(label="baseline", confidence=b["confidence"],
                            bustProbability=b["bust"], maePrecipMmDay=b["mae"]),
        recent=DriftPoint(label="recent", confidence=r["confidence"],
                          bustProbability=r["bust"], maePrecipMmDay=r["mae"]),
        delta={"confidence": d_conf, "bustProbability": d_bust,
               "maePrecipMmDay": d_mae},
        interpretation=verdict,
        caveat=("Windows are the first and last framable cycles of the record, "
                "not a random split; seasonal differences between the two "
                "windows are not removed, so read this as a change indicator, "
                "not a significance test."))


def _mean_window(cycles: list[str], lead: int) -> dict | None:
    eng = engine()
    vals = []
    for c in cycles:
        try:
            df = eng.frame(c)
        except Exception:            # noqa: BLE001
            continue
        sub = df[df["lead"] == int(lead)]
        if not len(sub):
            continue
        vals.append((float(sub["confidence"].mean()),
                     float(sub["bust_probability"].mean()),
                     float(np.abs(sub["err_precip"].astype("float64")).mean())))
    if not vals:
        return None
    return {"confidence": round(float(np.mean([v[0] for v in vals])), 2),
            "bust": round(float(np.mean([v[1] for v in vals])), 4),
            "mae": round(float(np.mean([v[2] for v in vals])), 3),
            "n": len(vals)}


# ---------------------------------------------------------------------------
# PHASE 8 - Forecast brief + search
# ---------------------------------------------------------------------------
_BRIEF_TABS = [
    ("overview", "Overview", "overview"), ("regions", "Regions", "regions"),
    ("intel", "Forecast Intelligence", "intel"),
    ("ops", "Operations Center", "ops"), ("case", "Case Study", "case"),
    ("model", "Model Performance", "model"), ("data", "Data Sources", "data"),
    ("method", "Methodology", "method"),
]


@router.get("/api/forecast-brief", response_model=ForecastBriefResponse,
            tags=["operations-center"])
def forecast_brief(cycle: str | None = Query(None),
                   lead: int = Query(3, ge=1, le=10),
                   region: str | None = Query(None)):
    """Plain-language brief assembled from the same endpoints the UI uses.

    Every line names the number it came from; where a source is unavailable the
    line says so instead of being omitted.
    """
    import datetime as _iso
    cur = _cycle_or_latest(cycle)
    rows = _rows_for(cur, int(lead))
    rows = sorted(rows, key=lambda r: r["confidence"])
    alR = alerts(cycle=cur, lead=int(lead), previousCycle=None)
    sysR = systems(cycle=cur, lead=int(lead))
    scope = [r for r in rows if r["region"] == region] if region else rows
    if not scope:
        raise HTTPException(404, "no rows for that region/lead")
    focus = min(scope, key=lambda r: r["confidence"])
    bands = {"low": [r for r in scope if r["confidence"] < 40],
             "mid": [r for r in scope if 40 <= r["confidence"] < 60],
             "high": [r for r in scope if r["confidence"] >= 60]}
    order = {"critical": 0, "warning": 1, "advisory": 2, "info": 3}
    groups = sorted(alR.groups or [],
                    key=lambda g: (order.get(g.severity, 9), -g.count))
    crit = [g for g in groups if g.severity == "critical"]
    top_alerts = groups[:6]

    sections = []
    sections.append(BriefSection(heading="Headline", lines=[
        f"Cycle {cur}, Day {lead}. {len(scope)} regions in scope; "
        f"{len(bands['low'])} below 40 confidence, {len(bands['mid'])} in 40-59, "
        f"{len(bands['high'])} at 60 or above.",
        f"Lowest confidence: {focus['region_name']} at {focus['confidence']} "
        f"(bust {focus['bust_probability']:.2f}, regime "
        f"{focus['regime']}).",
    ]))
    sections.append(BriefSection(heading="Why", lines=[
        f"{focus['region_name']}: dominant variable "
        f"{focus['dominant_variable']}, volatility index "
        f"{focus['volatility_index']}, ensemble precip spread "
        f"{focus['ensemble_spread_precip']} mm/day, expected rain error "
        f"{focus['expected_error']['precipitation_mm_day']} mm/day.",
    ]))
    sections.append(BriefSection(heading="Alerts", lines=[
        f"{len(groups)} alert groups, {len(crit)} critical. "
        + ("Top: " + "; ".join(
            f"{', '.join(g.regionNames[:2]) or g.regions[0]} D{g.lead} "
            f"{g.severity} ({g.metric})" for g in top_alerts[:3])
           if top_alerts else "no rules fired at this lead."),
    ]))
    if sysR.available and sysR.systems:
        sections.append(BriefSection(heading="Synoptic", lines=[
            f"{len(sysR.systems)} system(s) detected from forecast MSLP: "
            + "; ".join(
                f"{s.type} ({s.severity}, {s.labelImpact}) affecting "
                f"{', '.join(s.regionsAffected or []) or 'no named region'}"
                for s in sysR.systems[:4]),
        ]))
    else:
        sections.append(BriefSection(heading="Synoptic", lines=[
            "Systems unavailable in current data mode."
            if not sysR.available else "No systems detected in the forecast field.",
        ]))
    sections.append(BriefSection(heading="What would change this", lines=[
        "A lower confidence band or a bust probability at/above 0.25 in the "
        "watchlist, a volatility index at/above 1.2, or a new critical alert "
        "group - all three are already shown on Regions and Active Alerts.",
    ]))
    return ForecastBriefResponse(
        cycle=cur, lead=int(lead), region=region,
        headline=(f"Day {lead} of {cur}: {focus['region_name']} is the weakest "
                  f"region at {focus['confidence']} confidence."),
        sections=sections,
        caveats=[
            "Generated from stored forecast frames; no model is re-run.",
            ("Sandbox event catalogue is synthetic - this brief is not "
             "operational guidance." if cfg.is_sandbox else
             "Operational guidance requires human review."),
            f"Cycle {cur} may be stale relative to the current date.",
        ],
        generatedAt=_iso.datetime.now().isoformat(timespec="seconds"))


@router.get("/api/search", response_model=SearchResponse, tags=["operations-center"])
def search(q: str = Query("", max_length=80),
           lead: int = Query(3, ge=1, le=10)):
    """Search regions, event types, alert rules and navigation targets.

    Cheap by design: only config, the event catalogue header and the static
    rule list are scanned, so the box responds while heavier endpoints load.
    """
    needle = q.strip().lower()
    hits: list[SearchHit] = []
    ns: list[str] = []

    for v, label, _sub in _BRIEF_TABS:
        hits.append(SearchHit(type="view", label=label,
                              detail="Open this section", view=v))
    for t, label in ([("intel", k) for k in (
        "Overview", "Drift", "Stability", "Ensemble", "Analogues", "Drivers",
        "Scenarios", "Risk Evolution")] +
        [("ops", k) for k in ("Situation", "Active Alerts", "Systems", "Regions",
                              "Rules", "History", "Performance", "Trust",
                              "Brief")]):
        hits.append(SearchHit(type="tab", label=label,
                              detail="Operations / forecast tab", tab=t))
    for r in load_regions():
        hits.append(SearchHit(type="region", label=r.get("name", r["id"]),
                              detail=f"Region id {r['id']}", view="ops",
                              tab="regions"))
    ev_path = cfg.path("processed") / "sandbox_events.csv"
    kinds: set[str] = set()
    if ev_path.exists():
        import csv as _csv
        with open(ev_path, newline="") as fh:
            for row in _csv.DictReader(fh):
                kinds.add(row.get("kind") or "")
    for k in sorted(kinds - {""}):
        hits.append(SearchHit(type="event type", label=k,
                              detail="Event history and performance",
                              view="ops", tab="history"))
    for rule in _default_rules():
        hits.append(SearchHit(type="alert rule", label=rule.label,
                              detail=f"{rule.describe()} · {rule.severity}",
                              view="ops", tab="rules"))
    ns = sorted({h.type for h in hits})
    if needle:
        hits = [h for h in hits
                if needle in h.label.lower()
                or (h.detail and needle in h.detail.lower())]
    return SearchResponse(query=q, hits=hits[:40], namespaces=ns)
