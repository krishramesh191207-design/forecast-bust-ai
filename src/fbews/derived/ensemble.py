"""Ensemble summary reader.

The forecast NetCDF keeps ensemble *statistics* (mean, spread, quantiles,
probability-of-precipitation) but no member dimension - members are reduced at
generation time.  This module reads the raw forecast and truth files directly
so nothing has to be added to the model feature matrix.

Every value here comes from the files on disk.  Nothing is interpolated and
nothing is estimated; missing inputs are reported as ``None`` plus an explicit
``unavailable`` reason rather than being filled with zeros.
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

# variables a reader cares about -> (forecast var, unit, kind)
_VARIABLES: dict[str, dict[str, Any]] = {
    "precipitation": {
        "label": "Rainfall",
        "unit": "mm/day",
        "det": "det_precip",
        "mean": "ensmean_precip",
        "std": "ensstd_precip",
        "obs": "precip",
        "quantiles": (("q25", "precip_q25"), ("q50", "precip_q50"),
                      ("q75", "precip_q75"), ("q90", "precip_q90")),
        "extras": (("pop_gt10", "pop_gt10"), ("pop_gt25", "pop_gt25"),
                   ("pop_gt50", "pop_gt50")),
    },
    "temperature": {
        "label": "Temperature",
        "unit": "K",
        "det": "det_t2m",
        "mean": "ensmean_t2m",
        "std": "ensstd_t2m",
        "obs": "t2m",
        "quantiles": (),
        "extras": (("range", "t2m_range"),),
    },
    "wind": {
        "label": "Wind",
        "unit": "m/s",
        "det": None,           # u/v components combined below
        "mean": None,
        "std": None,
        "obs": None,
        "quantiles": (),
        "extras": (),
        "components": (("u10", "det_u10", "ensstd_u10", "u10"),
                       ("v10", "det_v10", "ensstd_v10", "v10")),
    },
    "pressure": {
        "label": "Pressure",
        "unit": "hPa",
        "det": "det_mslp",
        "mean": "ensmean_mslp",
        "std": "ensstd_mslp",
        "obs": "mslp",
        "quantiles": (),
        "extras": (("min", "mslp_min"),),
    },
}


def _f(v: Any, nd: int = 3) -> float | None:
    """JSON-safe float, or None for anything non-finite."""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return round(x, nd)


@lru_cache(maxsize=4)
def _open_forecast(path_str: str) -> xr.Dataset:
    return xr.open_dataset(path_str)


@lru_cache(maxsize=4)
def _open_truth(path_str: str) -> xr.Dataset:
    return xr.open_dataset(path_str)


def _close_all() -> None:
    for fn in (_open_forecast, _open_truth):
        try:
            ds = fn.cache_info  # type: ignore[attr-defined]
            del ds
        except Exception:  # pragma: no cover - defensive
            pass
    _open_forecast.cache_clear()
    _open_truth.cache_clear()


def forecast_path(forecasts_dir: str | Path, cycle: str) -> Path:
    return Path(forecasts_dir) / f"fc_{cycle[:10].replace('-', '')}.nc"


def truth_path(truth_dir: str | Path, valid: dt.date) -> Path | None:
    p = Path(truth_dir) / f"truth_{valid.year}.nc"
    return p if p.exists() else None


def ensemble_summary(forecasts_dir: str | Path, truth_dir: str | Path | None,
                     cycle: str, lat: float, lon: float, lead: int) -> dict:
    """Per-variable ensemble summary for one cell and lead time.

    Returns ``available=False`` with a human-readable reason when the forecast
    file is missing, and marks per-field values ``None`` when the field (or the
    verifying observation) is absent.
    """
    cycle_day = cycle[:10]
    fp = forecast_path(forecasts_dir, cycle_day)
    out: dict[str, Any] = {
        "available": False,
        "reason": None,
        "message": None,
        "cycle": cycle_day,
        "lead": int(lead),
        "valid_time": None,
        "location": {"lat": float(lat), "lon": float(lon)},
        "variables": {},
        "member_count": None,
        "members_available": False,
        "members_unavailable_reason": (
            "Member-level ensemble data unavailable. The forecast files store "
            "ensemble statistics (mean, spread, quantiles, probability of "
            "precipitation) only - members are reduced at generation time, so "
            "no member dimension exists to plot."
        ),
        "observed_available": False,
        "observed_reason": None,
        "data_mode": None,
        "warning": None,
    }
    if not fp.exists():
        out["reason"] = "forecast_file_missing"
        out["message"] = f"No forecast file for cycle {cycle_day}."
        return out

    ds = _open_forecast(str(fp))
    out["available"] = True
    out["data_mode"] = ds.attrs.get("data_mode")
    out["warning"] = ds.attrs.get("warning")
    members = ds.attrs.get("ensemble_members")
    out["member_count"] = int(members) if members is not None else None
    out["members_available"] = "member" in ds.dims

    valid = dt.date.fromisoformat(cycle_day) + dt.timedelta(days=int(lead))
    out["valid_time"] = valid.isoformat()

    # nearest grid point (never invent - always snap to a real cell)
    la = float(ds.lat.values[np.argmin(np.abs(ds.lat.values - float(lat)))])
    lo = float(ds.lon.values[np.argmin(np.abs(ds.lon.values - float(lon)))])
    out["location"] = {"lat": la, "lon": lo}

    obs_ds = None
    if truth_dir:
        tp = truth_path(truth_dir, valid)
        if tp is not None:
            try:
                obs_ds = _open_truth(str(tp))
            except Exception:
                obs_ds = None
    obs_sel = None
    if obs_ds is not None:
        try:
            tvals = obs_ds.time.values
            target = np.datetime64(valid.isoformat())
            idx = int(np.argmin(np.abs(tvals - target)))
            same = str(tvals[idx])[:10] == valid.isoformat()
            if same:
                obs_sel = obs_ds.isel(time=idx).sel(lat=la, lon=lo, method="nearest")
                out["observed_available"] = True
        except Exception:
            obs_sel = None
    if not out["observed_available"]:
        out["observed_reason"] = (
            "No verification available for this valid time."
            if obs_ds is None else
            "Verification file present but the valid time is not covered."
        )

    cell = ds.sel(lat=la, lon=lo, method="nearest").sel(lead=int(lead), method="nearest")

    for key, spec in _VARIABLES.items():
        block: dict[str, Any] = {
            "label": spec["label"],
            "unit": spec["unit"],
            "control": None,
            "ens_mean": None,
            "ens_std": None,
            "q25": None, "q50": None, "q75": None, "q90": None,
            "iqr": None,
            "extras": {},
            "observed": None,
            "available": False,
            "reason": None,
        }
        if spec.get("components"):
            spd = []
            for _, dvar, svar, ovar in spec["components"]:
                if dvar not in ds or svar not in ds:
                    continue
                spd.append((float(cell[dvar].values), float(cell[svar].values)))
                if obs_sel is not None and ovar in obs_sel:
                    block["observed"] = None  # combined below
            if spd:
                det = math_hypot([s[0] for s in spd])
                std = math_hypot([s[1] for s in spd])
                block["control"] = _f(det, 2)
                block["ens_mean"] = _f(det, 2)
                block["ens_std"] = _f(std, 2)
                if obs_sel is not None:
                    comp = [float(obs_sel[v].values) for _, _, _, v in spec["components"]
                            if v in obs_sel]
                    if len(comp) == 2:
                        block["observed"] = _f(math_hypot(comp), 2)
                block["available"] = block["control"] is not None
        else:
            for field, var in (("control", spec["det"]), ("ens_mean", spec["mean"]),
                               ("ens_std", spec["std"])):
                if var and var in ds:
                    block[field] = _f(cell[var].values, 3)
            for name, var in spec.get("quantiles", ()):
                if var in ds:
                    block[name] = _f(cell[var].values, 3)
            if block["q25"] is not None and block["q75"] is not None:
                block["iqr"] = _f(block["q75"] - block["q25"], 3)
            for name, var in spec.get("extras", ()):
                if var in ds:
                    block["extras"][name] = _f(cell[var].values, 3)
            if obs_sel is not None and spec["obs"] and spec["obs"] in obs_sel:
                block["observed"] = _f(obs_sel[spec["obs"]].values, 3)
            block["available"] = block["ens_mean"] is not None or block["control"] is not None
            if not block["available"]:
                block["reason"] = "field_missing"
        out["variables"][key] = block
    return out


def math_hypot(vals: list[float]) -> float:
    return float(np.sqrt(sum(float(v) ** 2 for v in vals)))
