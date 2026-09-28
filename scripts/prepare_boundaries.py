#!/usr/bin/env python3
"""Download and simplify India state boundaries for the map.

Source: https://github.com/datta07/INDIAN-SHAPEFILES (MIT licence).
Ramer-Douglas-Peucker simplification keeps the file small enough to embed.
"""
import json
import math
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbews.config import load_config  # noqa: E402

URL = ("https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/"
       "INDIA/INDIA_STATES.geojson")


def rdp(pts, eps):
    if len(pts) < 3:
        return pts

    def dist(p, a, b):
        (x, y), (x1, y1), (x2, y2) = p, a, b
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(x - x1, y - y1)
        t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)))
        return math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))

    dmax, idx = 0.0, 0
    for i in range(1, len(pts) - 1):
        d = dist(pts[i], pts[0], pts[-1])
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps:
        return rdp(pts[:idx + 1], eps)[:-1] + rdp(pts[idx:], eps)
    return [pts[0], pts[-1]]


def main(eps: float = 0.05) -> None:
    cfg = load_config()
    raw = cfg.path("raw") / "india_states_raw.geojson"
    if not raw.exists():
        urllib.request.urlretrieve(URL, raw)
    src = json.loads(raw.read_text())
    feats = []
    for f in src["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "Polygon" else [p for mp in g["coordinates"] for p in mp]
        rings = []
        for ring in polys:
            s = rdp([[round(x, 3), round(y, 3)] for x, y in ring], eps)
            if len(s) > 4:
                rings.append(s)
        if rings:
            feats.append({"type": "Feature",
                          "properties": {"name": f["properties"].get("STNAME_SH")},
                          "geometry": {"type": "Polygon", "coordinates": rings}})
    out = cfg.path("samples") / "india_states_simplified.geojson"
    out.write_text(json.dumps({"type": "FeatureCollection",
                               "attribution": "datta07/INDIAN-SHAPEFILES (MIT), simplified",
                               "features": feats}, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, {len(feats)} features)")


if __name__ == "__main__":
    main()
