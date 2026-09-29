"""Warning verification: earliest warning, lead scorecard, event performance.

Ground truth is the system event catalogue (``sandbox_events.csv`` in sandbox
mode, IBTrACS/IMD in real mode) - the same file the case-study page uses.
A grid cell is treated as "event affected" on a valid date when it lies inside
the catalogue's own ``radius_deg`` for an event on that date; everything else
is a non-event cell.  Nothing is sampled away and nothing is assumed.

Warning rule (existing project thresholds only, see ``fbews.derived.alerts``):
    confidence < 40  OR  bust probability >= 0.25
``40`` is the confidence band edge ("very_low" starts below 40) and ``0.25``
is the watchlist ``attention_required`` cut.

Rates follow the usual contingency definitions and are reported with their
counts so an operator can check the arithmetic:

    accuracy  = (hits + correct negatives) / total
    precision = hits / (hits + false alarms)
    recall    = hits / (hits + misses)      (a.k.a. hit rate)
    false alarm ratio = false alarms / (hits + false alarms)

A rate whose denominator is zero is reported as ``None``, never as 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

CONFIDENCE_LTE = 40.0
BUST_GTE = 0.25
WARNING_RULE = f"confidence < {CONFIDENCE_LTE:g} OR bust_probability >= {BUST_GTE:g}"


@dataclass(frozen=True)
class Counts:
    hits: int = 0
    misses: int = 0
    false_alarms: int = 0
    correct_negatives: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses + self.false_alarms + self.correct_negatives

    @property
    def events(self) -> int:
        return self.hits + self.misses

    @property
    def warnings(self) -> int:
        return self.hits + self.false_alarms

    def rates(self) -> dict[str, float | None]:
        def div(a: int, b: int) -> float | None:
            return round(a / b, 4) if b else None
        return {
            "accuracy": div(self.hits + self.correct_negatives, self.total),
            "precision": div(self.hits, self.warnings),
            "recall": div(self.hits, self.events),
            "falseAlarmRatio": div(self.false_alarms, self.warnings),
            "frequency": div(self.events, self.total),
        }

    def as_dict(self) -> dict[str, Any]:
        return {"hits": self.hits, "misses": self.misses,
                "falseAlarms": self.false_alarms,
                "correctNegatives": self.correct_negatives,
                "total": self.total, "events": self.events,
                "warnings": self.warnings, **self.rates()}


def warns(confidence: np.ndarray, bust: np.ndarray) -> np.ndarray:
    """Warning mask under the documented rule; NaN counts as no warning."""
    c = np.asarray(confidence, dtype="float64")
    b = np.asarray(bust, dtype="float64")
    with np.errstate(invalid="ignore"):
        low_conf = np.where(np.isfinite(c), c < CONFIDENCE_LTE, False)
        high_bust = np.where(np.isfinite(b), b >= BUST_GTE, False)
    return low_conf | high_bust


def distance_deg(lat1: np.ndarray, lon1: np.ndarray,
                 lat2: float, lon2: float) -> np.ndarray:
    """Angular distance in degrees on a sphere (broadcast over arrays)."""
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def zone_masks(events: Sequence[dict], lats: np.ndarray, lons: np.ndarray
               ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(any_event, {kind: mask}) over the flattened grid for one valid date."""
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    la = la.ravel()
    lo = lo.ravel()
    any_mask = np.zeros(la.shape, dtype=bool)
    by_kind: dict[str, np.ndarray] = {}
    for ev in events:
        r = float(ev.get("radius_deg") or 0.0) or 1.0
        m = distance_deg(la, lo, float(ev["lat"]), float(ev["lon"])) <= r
        any_mask |= m
        kind = str(ev.get("kind") or "unknown")
        by_kind.setdefault(kind, np.zeros(la.shape, dtype=bool))
        by_kind[kind] |= m
    return any_mask, by_kind


def _counts(warn: np.ndarray, zone: np.ndarray) -> Counts:
    return Counts(
        hits=int(np.sum(warn & zone)),
        misses=int(np.sum(~warn & zone)),
        false_alarms=int(np.sum(warn & ~zone)),
        correct_negatives=int(np.sum(~warn & ~zone)),
    )


def scorecard_from_frames(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate contingency counts across (cycle, lead) entries.

    Each entry: ``{lead, warning: bool[], zone: bool[], kind_zones: {kind: []}}``
    """
    per_lead: dict[int, Counts] = {}
    per_kind: dict[str, Counts] = {}
    totals = Counts()
    for e in entries:
        lead = int(e["lead"])
        w = np.asarray(e["warning"], dtype=bool)
        z = np.asarray(e["zone"], dtype=bool)
        c = _counts(w, z)
        prev_lead = per_lead.get(lead, Counts())
        per_lead[lead] = Counts(prev_lead.hits + c.hits,
                                prev_lead.misses + c.misses,
                                prev_lead.false_alarms + c.false_alarms,
                                prev_lead.correct_negatives + c.correct_negatives)
        totals = Counts(totals.hits + c.hits, totals.misses + c.misses,
                        totals.false_alarms + c.false_alarms,
                        totals.correct_negatives + c.correct_negatives)
        for kind, km in (e.get("kind_zones") or {}).items():
            kc = _counts(w, np.asarray(km, dtype=bool))
            prev = per_kind.get(kind, Counts())
            per_kind[kind] = Counts(prev.hits + kc.hits, prev.misses + kc.misses,
                                    prev.false_alarms + kc.false_alarms,
                                    prev.correct_negatives + kc.correct_negatives)
    leads = [{"lead": k, **v.as_dict()} for k, v in sorted(per_lead.items())]
    kinds = [{"eventType": k, **v.as_dict()}
             for k, v in sorted(per_kind.items(), key=lambda kv: -kv[1].events)]
    return {"overall": totals.as_dict(), "byLead": leads, "byEventType": kinds,
            "rule": WARNING_RULE}


def earliest_warning(event: dict, frames_by_cycle: dict[str, "pd.DataFrame"],
                      available: Sequence[str]) -> dict | None:
    """Furthest-ahead lead at which the rule first fired for one event.

    ``frames_by_cycle`` maps cycle -> slim frame with ``lat/lon/confidence/
    bust_probability`` per lead.  Each candidate lead uses its own cycle
    (``cycle = event date - lead``), so a warning is always read at the valid
    time of the event.  Returns ``None`` when no candidate cycle exists.
    """
    from datetime import date as _date, timedelta

    date = str(event["date"])
    la, lo = float(event["lat"]), float(event["lon"])
    fired: list[int] = []
    evaluated: list[int] = []
    avail = set(available)
    for lead in range(10, 0, -1):
        cycle = (_date.fromisoformat(date) - timedelta(days=lead)).isoformat()
        if cycle not in avail:
            continue
        df = frames_by_cycle.get(cycle)
        if df is None:
            continue
        sub = df[df["lead"] == lead]
        if not len(sub):
            continue
        d2 = (sub["lat"].to_numpy() - la) ** 2 + (sub["lon"].to_numpy() - lo) ** 2
        idx = int(np.argmin(d2))
        if float(d2[idx]) > 25.0:          # >5 deg: cell is not near the event
            continue
        evaluated.append(lead)
        if bool(warns(np.array([sub["confidence"].to_numpy()[idx]]),
                      np.array([sub["bust_probability"].to_numpy()[idx]]))[0]):
            fired.append(lead)
    if not evaluated:
        return None
    return {
        "eventDate": date,
        "eventType": event.get("kind"),
        "lat": la, "lon": lo,
        "depthHpa": event.get("depth_hpa"),
        "leadsEvaluated": sorted(evaluated),
        "warningLeads": sorted(fired),
        "earliestWarningLead": max(fired) if fired else None,
        "hoursBeforeValid": (max(fired) * 24) if fired else None,
        "issued": bool(fired),
        "latestLeadEvaluated": max(evaluated),
    }


def summarise_earliest(rows: Sequence[dict]) -> dict:
    done = [r for r in rows if r]
    issued = [r for r in done if r["issued"]]
    leads = [r["earliestWarningLead"] for r in issued]
    by_kind: dict[str, list[int]] = {}
    for r in issued:
        by_kind.setdefault(str(r["eventType"]), []).append(int(r["earliestWarningLead"]))
    kind_summary = [{
        "eventType": k,
        "events": len(v),
        "medianEarliestLead": float(np.median(v)),
        "meanEarliestLead": round(float(np.mean(v)), 2),
        "minEarliestLead": int(min(v)),
        "maxEarliestLead": int(max(v)),
    } for k, v in sorted(by_kind.items(), key=lambda kv: -len(kv[1]))]
    return {
        "eventsEvaluated": len(done),
        "eventsWithWarning": len(issued),
        "eventsWithoutWarning": len(done) - len(issued),
        "detectionRate": (round(len(issued) / len(done), 4) if done else None),
        "medianEarliestLead": (float(np.median(leads)) if leads else None),
        "meanEarliestLead": (round(float(np.mean(leads)), 2) if leads else None),
        "bestLead": (int(max(leads)) if leads else None),
        "worstLead": (int(min(leads)) if leads else None),
        "byEventType": kind_summary,
    }


def select_cycles(avail: Sequence[str], events: Sequence[dict],
                  n: int) -> list[str]:
    """Choose ``n`` forecast cycles so the sample covers every event type.

    A cycle covers the valid dates ``cycle+1 .. cycle+10``.  Selection is
    round-robin over event types first (so no type is dropped by chance) and
    then by how many still-uncovered event dates a cycle adds.  Deterministic:
    ties are broken by date order.  Needs no forecast data, so it costs
    nothing before any frame is loaded.
    """
    from datetime import date as _date, timedelta

    avail_set = set(avail)
    by_date: dict[str, set[str]] = {}
    dates_by_kind: dict[str, set[str]] = {}
    for ev in events:
        d = str(ev["date"])
        by_date.setdefault(d, set()).add(str(ev.get("kind") or "unknown"))
        dates_by_kind.setdefault(str(ev.get("kind") or "unknown"), set()).add(d)
    ev_dates = set(by_date)

    def window(c: str) -> set[str]:
        cd = _date.fromisoformat(c)
        return {(cd + timedelta(days=k)).isoformat() for k in range(1, 11)}

    def cycle_for(d: str) -> list[str]:
        dd = _date.fromisoformat(d)
        return [(dd - timedelta(days=k)).isoformat()
                for k in range(1, 11)
                if (dd - timedelta(days=k)).isoformat() in avail_set]

    covered: set[str] = set()
    picked: list[str] = []

    def take(c: str) -> None:
        if c in picked:
            return
        picked.append(c)
        covered.update(window(c) & ev_dates)

    rounds = 0
    while len(picked) < n and rounds < n + 4:
        rounds += 1
        progressed = False
        for kind in sorted(dates_by_kind):
            if len(picked) >= n:
                break
            best, best_gain = None, 0
            for d in sorted(dates_by_kind[kind] - covered):
                for c in cycle_for(d):
                    if c in picked:
                        continue
                    gain = len(window(c) & ev_dates - covered)
                    if gain > best_gain:
                        best, best_gain = c, gain
                if best:
                    break
            if best:
                take(best)
                progressed = True
        if not progressed:
            break

    # Fill any remaining slots with the cycles covering the most event dates.
    if len(picked) < n:
        rest = sorted(
            (c for c in avail if c not in picked),
            key=lambda c: (-len(window(c) & ev_dates - covered), c))
        for c in rest:
            if len(picked) >= n:
                break
            take(c)
    return sorted(picked)
