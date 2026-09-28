"""Geographic metadata quality checks for the dashboard map labels.

The map label dataset (``data/samples/india_geo_labels.json``) is the single
source of truth for state capitals, reference cities and neighbouring-country
annotations.  These tests make sure it stays consistent with the boundary file
the map actually renders, so no state is silently assigned another state's
capital and no coordinate pair has swapped axes.
"""
import json
import math

import pytest

from fbews.config import load_config

cfg = load_config()
GEO_JSON = cfg.path("samples") / "india_geo_labels.json"
BOUNDARY_JSON = cfg.path("samples") / "india_states_simplified.geojson"

pytestmark = pytest.mark.skipif(
    not (GEO_JSON.exists() and BOUNDARY_JSON.exists()),
    reason="geo label / boundary files not built",
)

# Geographic envelope of the boundary file (lon0, lat0, lon1, lat1).
INDIA_BBOX = (68.0, 6.5, 97.5, 37.5)
# Maximum allowed gap (decimal degrees, ~15 km) between a capital and the
# outline of its own state.  The boundary file is Ramer-Douglas-Peucker
# simplified, so the true border can sit a little away from the polygon.
CAPITAL_TOLERANCE_DEG = 0.15


@pytest.fixture(scope="module")
def geo():
    return json.loads(GEO_JSON.read_text())


@pytest.fixture(scope="module")
def states():
    doc = json.loads(BOUNDARY_JSON.read_text())
    out = []
    for f in doc["features"]:
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "Polygon" else [
            ring for poly in geom["coordinates"] for ring in poly
        ]
        rings = [[tuple(pt) for pt in ring] for ring in polys]
        xs = [p[0] for r in rings for p in r]
        ys = [p[1] for r in rings for p in r]
        out.append({
            "name": f["properties"]["name"],
            "rings": rings,
            "bbox": (min(xs), min(ys), max(xs), max(ys)),
        })
    return out


def _pip(lon, lat, ring):
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _inside(lon, lat, state):
    return any(_pip(lon, lat, r) for r in state["rings"])


def _dist_to_state(lon, lat, state):
    lo0, la0, lo1, la1 = state["bbox"]
    if lon < lo0 - CAPITAL_TOLERANCE_DEG or lon > lo1 + CAPITAL_TOLERANCE_DEG:
        return math.inf
    if lat < la0 - CAPITAL_TOLERANCE_DEG or lat > la1 + CAPITAL_TOLERANCE_DEG:
        return math.inf
    best = math.inf
    for ring in state["rings"]:
        for i in range(len(ring)):
            x1, y1 = ring[i]
            x2, y2 = ring[(i + 1) % len(ring)]
            dx, dy = x2 - x1, y2 - y1
            if dx == 0 and dy == 0:
                d = math.hypot(lon - x1, lat - y1)
            else:
                t = max(0.0, min(1.0, ((lon - x1) * dx + (lat - y1) * dy) / (dx * dx + dy * dy)))
                d = math.hypot(lon - (x1 + t * dx), lat - (y1 + t * dy))
            best = min(best, d)
    return best


def _covers(lon, lat, states):
    return any(_inside(lon, lat, s) for s in states)


def _assert_in_india(lon, lat, where):
    lo0, la0, lo1, la1 = INDIA_BBOX
    assert lo0 <= lon <= lo1, f"{where}: longitude {lon} outside India ({lo0}..{lo1})"
    assert la0 <= lat <= la1, f"{where}: latitude {lat} outside India ({la0}..{la1})"
    # A swapped pair would put a 70-95 degree value in the latitude slot.
    assert lat < lon, f"{where}: latitude {lat} looks like a swapped longitude"


def test_boundary_states_all_have_a_capital(geo, states):
    capitals = {c["state"] for c in geo["capitals"]}
    boundary_names = {s["name"] for s in states}
    missing = boundary_names - capitals
    assert not missing, f"states without a capital entry: {sorted(missing)}"


def test_no_capital_entries_for_unknown_states(geo, states):
    boundary_names = {s["name"] for s in states}
    unknown = {c["state"] for c in geo["capitals"]} - boundary_names
    assert not unknown, f"capital entries for states absent from the map: {sorted(unknown)}"


def test_each_state_appears_once(geo):
    seen = [c["state"] for c in geo["capitals"]]
    dupes = sorted({s for s in seen if seen.count(s) > 1})
    assert not dupes, f"states listed more than once: {dupes}"


def test_shared_capitals_are_only_chandigarh(geo):
    by_capital = {}
    for c in geo["capitals"]:
        by_capital.setdefault(c["capital"], []).append(c["state"])
    shared = {k: v for k, v in by_capital.items() if len(v) > 1}
    # Chandigarh is the administrative seat of Punjab and of the Chandigarh UT.
    allowed = {"Chandigarh": {"Chandigarh", "Punjab", "Haryana"}}
    assert set(shared) <= set(allowed), f"unexpected shared capitals: {shared}"
    for name, states_ in shared.items():
        assert set(states_) <= allowed[name], f"{name} shared by {states_}"


def test_capital_coordinates_are_ordered_and_in_range(geo):
    for c in geo["capitals"]:
        where = f"{c['state']}/{c['capital']}"
        assert isinstance(c["latitude"], (int, float))
        assert isinstance(c["longitude"], (int, float))
        _assert_in_india(c["longitude"], c["latitude"], where)


def test_capitals_sit_in_their_own_state(geo, states):
    by_name = {s["name"]: s for s in states}
    problems = []
    for c in geo["capitals"]:
        state = by_name[c["state"]]
        lon, lat = c["longitude"], c["latitude"]
        if _inside(lon, lat, state):
            continue
        d = _dist_to_state(lon, lat, state)
        if d > CAPITAL_TOLERANCE_DEG:
            nearest = min(states, key=lambda s: _dist_to_state(lon, lat, s))
            problems.append(
                f"{c['capital']} ({c['state']}) is {d:.3f} deg from its own outline; "
                f"nearest outline is {nearest['name']}"
            )
    assert not problems, "capital coordinates do not match their state:\n" + "\n".join(problems)


def test_capital_markers_are_unique(geo):
    seen = {}
    for c in geo["capitals"]:
        key = (round(c["latitude"], 3), round(c["longitude"], 3))
        seen.setdefault(key, []).append(c["capital"])
    for key, names in seen.items():
        assert len(set(names)) == 1, f"one coordinate claimed by {names}"


def test_capitals_do_not_clash_with_reference_cities(geo):
    capitals = {c["capital"] for c in geo["capitals"]}
    overlap = capitals & {c["name"] for c in geo["cities"]}
    assert not overlap, f"listed both as capital and reference city: {sorted(overlap)}"


def test_reference_cities_are_inside_india(geo, states):
    for c in geo["cities"]:
        where = c["name"]
        _assert_in_india(c["longitude"], c["latitude"], where)
        assert _covers(c["longitude"], c["latitude"], states) or min(
            _dist_to_state(c["longitude"], c["latitude"], s) for s in states
        ) <= CAPITAL_TOLERANCE_DEG, f"{where} is not on Indian territory"


def test_country_labels_sit_outside_india(geo, states):
    for c in geo["countries"]:
        assert not _covers(c["longitude"], c["latitude"], states), (
            f"{c['name']} label falls inside India"
        )


def test_importance_is_a_known_tier(geo):
    for group in ("capitals", "cities", "countries"):
        for c in geo[group]:
            assert c["importance"] in (1, 2, 3, 4, 5), f"{c} has an unknown importance"
    assert all(c["importance"] <= 2 for c in geo["capitals"])
    assert all(c["importance"] >= 3 for c in geo["cities"])
    assert all(c["importance"] == 5 for c in geo["countries"])


def test_geo_labels_endpoint_serves_the_dataset():
    from fastapi.testclient import TestClient

    from fbews.api.main import app

    client = TestClient(app)
    r = client.get("/api/geo-labels")
    assert r.status_code == 200
    body = r.json()
    assert len(body["capitals"]) == len(json.loads(GEO_JSON.read_text())["capitals"])
    assert {"state", "capital", "latitude", "longitude"} <= set(body["capitals"][0])
