"""Alert rules and evaluation (Operations Center - Active Alerts).

Rules are data, not code.  Every threshold used by a built-in rule already
exists somewhere else in this project and the source is recorded on the rule
itself, so the alert board can never invent a cut point:

* ``0.25``  - watchlist ``attention_required``
* ``0.50``  - dashboard red cut for bust probability
* ``60 / 40 / 20`` - confidence band edges (``configs/default.yaml``) and the
  ``bandLabel`` cut points in the dashboard
* ``1.2 / 3.0`` - ``derived/stability.RULES["volatility_elevated"]`` and
  ``["volatility_index_max"]``
* ``5.0``   - ``bust.absolute_thresholds.precipitation_mm_day``

Evaluation is a pure function: rules + region rows in, alert records out.
There is no server-side alert store - state (new / ongoing / escalated) is
derived by re-running the same rules against the previous forecast cycle, so
the endpoint stays stateless and reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Sequence

SEVERITIES = ("info", "advisory", "warning", "critical")
SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITIES)}
OPERATORS = ("lt", "lte", "gt", "gte")
OPERATOR_TEXT = {"lt": "<", "lte": "≤", "gt": ">", "gte": "≥"}

METRIC_FIELDS = {
    "confidence": ("confidence", "points"),
    "bust_probability": ("bust_probability", ""),
    "volatility": ("volatility_index", "index"),
    "rain_error": ("expected_error.precipitation_mm_day", "mm/day"),
}


@dataclass(frozen=True)
class AlertRule:
    id: str
    label: str
    metric: str
    operator: str
    threshold: float
    lead: int | None = None
    severity: str = "warning"
    enabled: bool = True
    source: str = "project"

    def describe(self) -> str:
        field_name, unit = METRIC_FIELDS.get(self.metric, (self.metric, ""))
        lead = "any lead" if self.lead is None else f"Day {self.lead}"
        u = f" {unit}" if unit else ""
        return (f"{field_name} {OPERATOR_TEXT[self.operator]} "
                f"{self.threshold:g}{u} · {lead}")


DEFAULT_RULES: tuple[AlertRule, ...] = (
    AlertRule("bust-elevated", "Elevated bust risk", "bust_probability",
              "gte", 0.25, None, "warning",
              source="watchlist attention_required (>= 0.25)"),
    AlertRule("bust-critical", "Critical bust risk", "bust_probability",
              "gte", 0.50, None, "critical",
              source="dashboard red cut for bust probability (>= 0.50)"),
    AlertRule("conf-below-moderate", "Confidence below moderate band",
              "confidence", "lt", 60, None, "advisory",
              source="confidence band edge (moderate starts at 60)"),
    AlertRule("conf-very-low", "Confidence in very-low band", "confidence",
              "lt", 40, None, "warning",
              source="confidence band edge (very_low is 0-39)"),
    AlertRule("conf-critical", "Confidence critically low", "confidence",
              "lt", 20, None, "critical",
              source="bandLabel 'High Bust Risk' cut (< 20)"),
    AlertRule("vol-elevated", "Elevated forecast volatility", "volatility",
              "gte", 1.2, None, "advisory",
              source="derived/stability RULES.volatility_elevated"),
    AlertRule("vol-extreme", "Extreme forecast volatility", "volatility",
              "gte", 3.0, None, "warning",
              source="derived/stability RULES.volatility_index_max"),
    AlertRule("rain-error-high", "Rainfall error above bust floor",
              "rain_error", "gte", 5.0, None, "advisory",
              source="bust.absolute_thresholds.precipitation_mm_day"),
)


def default_rules() -> list[AlertRule]:
    return list(DEFAULT_RULES)


def _value(row: dict[str, Any], metric: str) -> float | None:
    field_name, _ = METRIC_FIELDS.get(metric, (metric, ""))
    if "." in field_name:
        obj: Any = row
        for part in field_name.split("."):
            if not isinstance(obj, dict) or part not in obj:
                return None
            obj = obj[part]
        v = obj
    else:
        v = row.get(field_name)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _holds(op: str, value: float, threshold: float) -> bool:
    if op == "lt":
        return value < threshold
    if op == "lte":
        return value <= threshold
    if op == "gt":
        return value > threshold
    if op == "gte":
        return value >= threshold
    return False


def evaluate(rules: Sequence[AlertRule], rows: Sequence[dict[str, Any]]) -> list[dict]:
    """Breach records for every (rule x region x lead).  No de-duplication."""
    out: list[dict] = []
    for rule in rules:
        if not rule.enabled or rule.metric not in METRIC_FIELDS:
            continue
        for row in rows:
            lead = int(row.get("lead_day") or 0)
            if rule.lead is not None and lead != rule.lead:
                continue
            value = _value(row, rule.metric)
            if value is None:
                continue
            if not _holds(rule.operator, value, rule.threshold):
                continue
            out.append({
                "id": f"{rule.id}|{lead}|{row.get('region')}",
                "ruleId": rule.id,
                "ruleLabel": rule.label,
                "metric": rule.metric,
                "operator": rule.operator,
                "threshold": rule.threshold,
                "condition": rule.describe(),
                "severity": rule.severity,
                "source": rule.source,
                "region": row.get("region"),
                "regionName": row.get("region_name") or row.get("region"),
                "lead": lead,
                "value": round(value, 3),
                "direction": "towards_worse" if rule.operator in ("lt", "lte")
                else "towards_worse",
            })
    return out


def deduplicate(alerts: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    """Keep the highest-severity breach per (metric, region, lead).

    A region sitting at 12% confidence trips three confidence rules at once.
    The board shows the worst one and lists the rest as covered, so an
    operator never reads the same condition three times.
    """
    best: dict[tuple, dict] = {}
    covered: list[dict] = []
    for a in alerts:
        key = (a["metric"], a["region"], a["lead"])
        cur = best.get(key)
        if cur is None:
            best[key] = a
            continue
        keep, drop = (a, cur) if SEVERITY_RANK[a["severity"]] > SEVERITY_RANK[cur["severity"]] else (cur, a)
        best[key] = keep
        covered.append({**drop, "suppressedBecause":
                        f"covered by '{keep['ruleLabel']}' "
                        f"({keep['severity']}) on the same metric"})
    kept = sorted(best.values(),
                  key=lambda a: (-SEVERITY_RANK[a["severity"]], a["lead"], str(a["region"])))
    covered.sort(key=lambda a: (-SEVERITY_RANK[a["severity"]], a["lead"], str(a["region"])))
    return kept, covered


def attach_status(alerts: Sequence[dict],
                  previous: Sequence[dict]) -> list[dict]:
    """new / ongoing / escalated, by comparing against the previous cycle."""
    prev_best: dict[tuple, int] = {}
    for a in previous:
        key = (a["metric"], a["region"], a["lead"])
        prev_best[key] = max(prev_best.get(key, -1), SEVERITY_RANK[a["severity"]])
    out = []
    for a in alerts:
        key = (a["metric"], a["region"], a["lead"])
        prev = prev_best.get(key)
        if prev is None:
            status, label = "new", "New"
        elif SEVERITY_RANK[a["severity"]] > prev:
            status, label = "escalated", "Escalated"
        else:
            status, label = "ongoing", "Ongoing"
        rank = SEVERITY_RANK[a["severity"]]
        out.append({**a, "status": status, "statusLabel": label,
                    "previousSeverity": (SEVERITIES[prev] if prev is not None else None),
                    "severityRank": rank})
    return out


def group_alerts(alerts: Sequence[dict]) -> list[dict]:
    """One row per (rule, lead): which regions, how bad, how many."""
    groups: dict[tuple, dict] = {}
    for a in alerts:
        key = (a["ruleId"], a["lead"])
        g = groups.get(key)
        if g is None:
            g = {"id": f"{a['ruleId']}|{a['lead']}",
                 "ruleId": a["ruleId"], "label": a["ruleLabel"],
                 "metric": a["metric"], "condition": a["condition"],
                 "lead": a["lead"], "severity": a["severity"],
                 "regions": [], "regionNames": [], "count": 0,
                 "worstValue": None, "statuses": set()}
            groups[key] = g
        g["regions"].append(a["region"])
        g["regionNames"].append(a["regionName"])
        g["count"] += 1
        g["statuses"].add(a["status"])
        if g["worstValue"] is None:
            g["worstValue"] = a["value"]
        else:
            worse = (a["value"] < g["worstValue"]) if a["operator"] in ("lt", "lte") \
                else (a["value"] > g["worstValue"])
            if worse:
                g["worstValue"] = a["value"]
        if SEVERITY_RANK[a["severity"]] > SEVERITY_RANK[g["severity"]]:
            g["severity"] = a["severity"]
    out = []
    for g in groups.values():
        g["statuses"] = sorted(g["statuses"])
        g["regions"] = sorted(set(g["regions"]))
        g["regionNames"] = sorted(set(str(x) for x in g["regionNames"]))
        out.append(g)
    out.sort(key=lambda g: (-SEVERITY_RANK[g["severity"]], g["lead"], -g["count"]))
    return out


def build_alerts(rules: Sequence[AlertRule], rows: Sequence[dict[str, Any]],
                 previous_rows: Sequence[dict[str, Any]] | None = None) -> dict:
    current_raw = evaluate(rules, rows)
    previous_raw = evaluate(rules, previous_rows or [])
    current, covered = deduplicate(current_raw)
    previous, _ = deduplicate(previous_raw)
    current = attach_status(current, previous)
    counts = {"new": 0, "ongoing": 0, "escalated": 0}
    by_sev: dict[str, int] = {s: 0 for s in SEVERITIES}
    for a in current:
        counts[a["status"]] = counts.get(a["status"], 0) + 1
        by_sev[a["severity"]] = by_sev.get(a["severity"], 0) + 1
    resolved = [a for a in previous
                if (a["metric"], a["region"], a["lead"])
                not in {(c["metric"], c["region"], c["lead"]) for c in current}]
    return {
        "rules": [asdict(r) for r in rules],
        "alerts": current,
        "groups": group_alerts(current),
        "suppressed": covered,
        "resolved": [{**a, "status": "resolved", "statusLabel": "Resolved"}
                     for a in resolved],
        "counts": {**counts, "total": len(current),
                   "groups": len(group_alerts(current)),
                   "resolved": len(resolved)},
        "bySeverity": by_sev,
        "previousCycleCompared": bool(previous_rows),
    }
