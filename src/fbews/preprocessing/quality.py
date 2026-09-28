"""Data-quality checks (PART 47).

Every check returns a structured record so results can be written to a
data-quality log that the API exposes at /api/data-quality.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import xarray as xr

# Physically plausible ranges. Values outside these are flagged, never silently
# clipped, so that a broken decode (e.g. precipitation in m instead of mm)
# is caught rather than hidden.
PLAUSIBLE = {
    "precip": (0.0, 2000.0, "mm/day"),
    "t2m": (180.0, 335.0, "K"),
    "mslp": (870.0, 1085.0, "hPa"),
    "u10": (-120.0, 120.0, "m/s"),
    "v10": (-120.0, 120.0, "m/s"),
    "u850": (-150.0, 150.0, "m/s"),
    "v850": (-150.0, 150.0, "m/s"),
    "z500": (4600.0, 6200.0, "m"),
    "tcwv": (0.0, 100.0, "kg/m2"),
    "cape": (0.0, 9000.0, "J/kg"),
}


def check_dataset(ds: xr.Dataset, label: str, expected_leads: list[int] | None = None) -> dict:
    issues: list[dict] = []
    stats: dict[str, dict] = {}

    for name, da in ds.data_vars.items():
        base = name.split("_")[-1]
        vals = da.values
        finite = np.isfinite(vals)
        missing_frac = float(1.0 - finite.mean())
        entry = {
            "missing_fraction": round(missing_frac, 6),
            "min": float(np.nanmin(vals)) if finite.any() else None,
            "max": float(np.nanmax(vals)) if finite.any() else None,
            "mean": float(np.nanmean(vals)) if finite.any() else None,
        }
        stats[name] = entry
        if missing_frac > 0.2:
            issues.append({"severity": "warning", "variable": name,
                           "message": f"{missing_frac:.1%} missing values"})
        if base in PLAUSIBLE and finite.any():
            lo, hi, units = PLAUSIBLE[base]
            if entry["min"] < lo or entry["max"] > hi:
                issues.append({
                    "severity": "error",
                    "variable": name,
                    "message": (f"values outside plausible range [{lo}, {hi}] {units}: "
                                f"observed [{entry['min']:.2f}, {entry['max']:.2f}] - "
                                "check unit conversion or decoding"),
                })

    if "lat" in ds.coords:
        lat = ds.lat.values
        if np.any(np.diff(lat) <= 0):
            issues.append({"severity": "error", "variable": "lat",
                           "message": "latitude is not strictly ascending"})
        if lat.min() < -90 or lat.max() > 90:
            issues.append({"severity": "error", "variable": "lat", "message": "invalid latitude"})
    if "lon" in ds.coords:
        lon = ds.lon.values
        if lon.min() < -180 or lon.max() > 360:
            issues.append({"severity": "error", "variable": "lon", "message": "invalid longitude"})
    if expected_leads is not None and "lead" in ds.coords:
        got = list(map(int, ds.lead.values))
        missing = sorted(set(expected_leads) - set(got))
        if missing:
            issues.append({"severity": "error", "variable": "lead",
                           "message": f"missing forecast lead times: {missing}"})
    if "time" in ds.coords and ds.time.size > 1:
        t = ds.time.values
        if len(np.unique(t)) != len(t):
            issues.append({"severity": "error", "variable": "time",
                           "message": "duplicate time steps"})

    return {
        "label": label,
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_variables": len(ds.data_vars),
        "issues": issues,
        "ok": not any(i["severity"] == "error" for i in issues),
        "stats": stats,
    }


def check_ensemble_completeness(n_members: int, expected: int, label: str) -> dict:
    return {
        "label": label,
        "expected_members": expected,
        "found_members": n_members,
        "ok": n_members == expected,
        "issues": ([] if n_members == expected else
                   [{"severity": "error", "variable": "member",
                     "message": f"{expected - n_members} ensemble members missing"}]),
    }


def append_log(record: dict, log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "data_quality.jsonl"
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")
    return path


def read_log(log_dir: Path, limit: int = 100) -> list[dict]:
    path = log_dir / "data_quality.jsonl"
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()[-limit:]
    return [json.loads(line) for line in lines if line.strip()]
