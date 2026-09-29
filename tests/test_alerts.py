"""Alert rule evaluation - pure unit tests (no server, no model)."""
from fbews.derived.alerts import (
    DEFAULT_RULES,
    SEVERITY_RANK,
    AlertRule,
    attach_status,
    build_alerts,
    deduplicate,
    default_rules,
    evaluate,
    group_alerts,
)


def _row(region="west_india", lead=3, conf=50, bust=0.3, vol=2.0, rain=6.0):
    return {
        "region": region, "region_name": region, "lead_day": lead,
        "confidence": conf, "bust_probability": bust, "volatility_index": vol,
        "expected_error": {"precipitation_mm_day": rain},
    }


# ---------------------------------------------------------------------------
# documented thresholds
# ---------------------------------------------------------------------------
def test_default_rule_thresholds_come_from_existing_project_values():
    by_id = {r.id: r for r in default_rules()}
    assert by_id["bust-elevated"].threshold == 0.25      # watchlist attention_required
    assert by_id["bust-critical"].threshold == 0.50      # dashboard red cut
    assert by_id["conf-below-moderate"].threshold == 60  # confidence band edge
    assert by_id["conf-very-low"].threshold == 40        # confidence band edge
    assert by_id["conf-critical"].threshold == 20        # bandLabel High Bust Risk
    assert by_id["vol-elevated"].threshold == 1.2        # stability RULES
    assert by_id["vol-extreme"].threshold == 3.0         # stability RULES
    assert by_id["rain-error-high"].threshold == 5.0     # bust absolute floor
    assert all(r.source for r in default_rules())


def test_rule_severities_are_valid_and_monotonic():
    for r in default_rules():
        assert r.severity in SEVERITY_RANK
        assert r.operator in ("lt", "lte", "gt", "gte")
    # higher confidence thresholds must not be *less* severe than the lower ones
    by_id = {r.id: r for r in default_rules()}
    assert SEVERITY_RANK[by_id["conf-critical"].severity] > \
        SEVERITY_RANK[by_id["conf-very-low"].severity]
    assert SEVERITY_RANK[by_id["bust-critical"].severity] > \
        SEVERITY_RANK[by_id["bust-elevated"].severity]


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def test_evaluate_only_fires_on_breaches():
    rules = [AlertRule("r", "low conf", "confidence", "lt", 40, None, "warning")]
    assert evaluate(rules, [_row(conf=50)]) == []
    hit = evaluate(rules, [_row(conf=30)])
    assert len(hit) == 1
    assert hit[0]["value"] == 30 and hit[0]["lead"] == 3
    assert hit[0]["id"] == "r|3|west_india"


def test_evaluate_skips_missing_values_and_lead_filter():
    rules = [
        AlertRule("a", "a", "confidence", "lt", 40, None, "warning"),
        AlertRule("b", "b", "confidence", "lt", 40, 7, "warning"),
    ]
    out = evaluate(rules, [_row(conf=30, lead=3)])
    assert [a["ruleId"] for a in out] == ["a"]        # lead 7 rule not applied
    no_field = [{"region": "x", "region_name": "x", "lead_day": 3}]
    assert evaluate(rules, no_field) == []


def test_operator_boundaries_are_inclusive_where_documented():
    rules = [
        AlertRule("gte", "gte", "bust_probability", "gte", 0.25, None, "warning"),
        AlertRule("gt", "gt", "bust_probability", "gt", 0.25, None, "warning"),
    ]
    out = evaluate(rules, [_row(bust=0.25)])
    assert [a["ruleId"] for a in out] == ["gte"]


# ---------------------------------------------------------------------------
# de-duplication
# ---------------------------------------------------------------------------
def test_deduplication_keeps_the_highest_severity_per_metric():
    rows = [_row(conf=10)]
    rules = [
        AlertRule("c1", "below 60", "confidence", "lt", 60, None, "advisory"),
        AlertRule("c2", "below 40", "confidence", "lt", 40, None, "warning"),
        AlertRule("c3", "below 20", "confidence", "lt", 20, None, "critical"),
    ]
    raw = evaluate(rules, rows)
    assert len(raw) == 3
    kept, covered = deduplicate(raw)
    assert len(kept) == 1 and kept[0]["ruleId"] == "c3"
    assert len(covered) == 2
    assert all("covered by" in c["suppressedBecause"] for c in covered)


def test_deduplication_keeps_distinct_metrics():
    rows = [_row(conf=10, bust=0.9)]
    rules = [
        AlertRule("c", "conf", "confidence", "lt", 20, None, "critical"),
        AlertRule("b", "bust", "bust_probability", "gte", 0.5, None, "critical"),
    ]
    kept, covered = deduplicate(evaluate(rules, rows))
    assert len(kept) == 2 and covered == []


# ---------------------------------------------------------------------------
# escalation state
# ---------------------------------------------------------------------------
def test_status_is_new_when_absent_from_previous_cycle():
    rules = [AlertRule("r", "r", "confidence", "lt", 40, None, "warning")]
    now = evaluate(rules, [_row(conf=30)])
    out = attach_status(now, [])
    assert [a["status"] for a in out] == ["new"]
    assert [a["statusLabel"] for a in out] == ["New"]


def test_status_is_ongoing_at_the_same_severity():
    rules = [AlertRule("r", "r", "confidence", "lt", 40, None, "warning")]
    now = evaluate(rules, [_row(conf=30)])
    prev = evaluate(rules, [_row(conf=35)])
    out = attach_status(now, prev)
    assert [a["status"] for a in out] == ["ongoing"]


def test_status_is_escalated_when_severity_rises():
    rules = [
        AlertRule("a", "a", "confidence", "lt", 60, None, "advisory"),
        AlertRule("b", "b", "confidence", "lt", 20, None, "critical"),
    ]
    now, _ = deduplicate(evaluate(rules, [_row(conf=10)]))   # critical now
    prev, _ = deduplicate(evaluate(rules, [_row(conf=50)]))  # advisory then
    out = attach_status(now, prev)
    assert [a["status"] for a in out] == ["escalated"]
    assert out[0]["severity"] == "critical"
    assert out[0]["previousSeverity"] == "advisory"


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------
def test_grouping_collapses_regions_for_one_rule_and_lead():
    rows = [_row(region="a", lead=3, conf=30), _row(region="b", lead=3, conf=25),
            _row(region="c", lead=5, conf=30)]
    rules = [AlertRule("r", "low", "confidence", "lt", 40, None, "warning")]
    kept, _ = deduplicate(evaluate(rules, rows))
    kept = attach_status(kept, [])
    groups = group_alerts(kept)
    assert len(groups) == 2                              # one per lead
    g3 = next(g for g in groups if g["lead"] == 3)
    assert g3["count"] == 2 and sorted(g3["regions"]) == ["a", "b"]
    assert g3["label"] == "low"


def test_group_severity_is_the_rule_severity_of_its_members():
    rows = [_row(conf=50), _row(region="b", conf=10)]
    rules = [
        AlertRule("a", "a", "confidence", "lt", 60, None, "advisory"),
        AlertRule("b", "b", "confidence", "lt", 20, None, "critical"),
    ]
    kept, _ = deduplicate(evaluate(rules, rows))     # one record per region
    kept = attach_status(kept, [])
    groups = group_alerts(kept)
    assert {g["severity"] for g in groups} == {"advisory", "critical"}
    for g in groups:
        members = [a for a in kept if a["ruleId"] == g["ruleId"] and a["lead"] == g["lead"]]
        assert all(a["severity"] == g["severity"] for a in members)
        assert g["count"] == len(members)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
def test_build_alerts_counts_are_consistent():
    rows = [_row(region=f"r{i}", lead=3, conf=10 + i, bust=0.6 + i * 0.01)
            for i in range(6)]
    prev = [_row(region=f"r{i}", lead=3, conf=55, bust=0.3) for i in range(6)]
    res = build_alerts(default_rules(), rows, prev)
    assert res["counts"]["total"] == len(res["alerts"])
    assert res["counts"]["groups"] == len(res["groups"])
    assert (res["counts"]["new"] + res["counts"]["ongoing"]
            + res["counts"]["escalated"]) == len(res["alerts"])
    assert res["previousCycleCompared"] is True
    assert res["bySeverity"]["critical"] >= 1
    assert isinstance(res["rules"], list) and res["rules"]


def test_build_alerts_without_previous_cycle_marks_everything_new():
    res = build_alerts(default_rules(), [_row(conf=10, bust=0.9)], None)
    assert res["previousCycleCompared"] is False
    assert all(a["status"] == "new" for a in res["alerts"])


def test_disabled_rules_are_ignored():
    rule = AlertRule("off", "off", "confidence", "lt", 100, None, "warning",
                     enabled=False)
    assert evaluate([rule], [_row(conf=1)]) == []
