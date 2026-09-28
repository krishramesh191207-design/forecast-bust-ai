"""Forecast/observation matching and forecast-error computation (PARTS 4, 48, 49).

Verification rule used everywhere:

    valid_time = forecast_initialisation_time + forecast_lead

All times are UTC. Precipitation is compared as a **24-hour accumulation**
ending at the valid time; the forecast accumulation window and the observation
window must agree. For the sandbox the window is 00-00 UTC on both sides. For
IMD gridded rainfall the observation window is 0830 IST to 0830 IST
(= 03 UTC to 03 UTC), so `obs_window_offset_hours` must be set to 3 and the
forecast accumulation shifted accordingly - see docs/verification.md.

Unit conventions after preprocessing:
    precipitation  mm/day        temperature  K
    wind           m/s           pressure     hPa
    geopotential   m (height, not m2/s2)
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import xarray as xr


def valid_time(cycle: dt.datetime | dt.date, lead_days: int) -> dt.datetime:
    if isinstance(cycle, dt.date) and not isinstance(cycle, dt.datetime):
        cycle = dt.datetime(cycle.year, cycle.month, cycle.day)
    return cycle + dt.timedelta(days=int(lead_days))


def match_truth(truth: xr.Dataset, cycle: dt.date, leads: list[int]) -> xr.Dataset:
    """Select the verifying analysis for each lead of a forecast cycle."""
    times = [np.datetime64(valid_time(cycle, l).date().isoformat()) for l in leads]
    available = set(truth.time.values.astype("datetime64[D]").astype(str))
    keep = [(l, t) for l, t in zip(leads, times) if str(t.astype("datetime64[D]")) in available]
    if not keep:
        raise KeyError(f"no verifying analysis available for cycle {cycle}")
    sel = truth.sel(time=[t for _, t in keep])
    return sel.assign_coords(lead=("time", [l for l, _ in keep])).swap_dims({"time": "lead"})


def compute_errors(fc: xr.Dataset, obs: xr.Dataset, thresholds: list[float]) -> xr.Dataset:
    """Error fields for one forecast cycle on (lead, lat, lon).

    `fc` must contain ensmean_* variables, `obs` the verifying analysis
    aligned on the `lead` dimension (see `match_truth`).
    """
    leads = [int(l) for l in obs.lead.values]
    f = fc.sel(lead=leads)
    out: dict[str, tuple] = {}

    def put(name, arr):
        out[name] = (("lead", "lat", "lon"), np.asarray(arr, dtype="float32"))

    # ---- precipitation -------------------------------------------------
    fp, op = f.ensmean_precip.values, obs.precip.values
    put("obs_precip", op)
    put("fc_precip", fp)
    put("bias_precip", fp - op)
    put("err_precip", np.abs(fp - op))
    put("rel_precip", np.abs(fp - op) / np.maximum(op, 1.0))
    for thr in thresholds:
        fe, oe = fp >= thr, op >= thr
        # +1 false alarm, -1 miss, 0 agreement
        put(f"thr{int(thr)}_precip", fe.astype("float32") - oe.astype("float32"))

    # ---- temperature ----------------------------------------------------
    ft, ot = f.ensmean_t2m.values, obs.t2m.values
    put("obs_t2m", ot)
    put("fc_t2m", ft)
    put("bias_t2m", ft - ot)
    put("err_t2m", np.abs(ft - ot))

    # ---- wind ------------------------------------------------------------
    fu, fv = f.ensmean_u10.values, f.ensmean_v10.values
    ou, ov = obs.u10.values, obs.v10.values
    put("err_u10", np.abs(fu - ou))
    put("err_v10", np.abs(fv - ov))
    put("err_wind", np.hypot(fu - ou, fv - ov))          # vector error
    put("bias_wspd", np.hypot(fu, fv) - np.hypot(ou, ov))

    # ---- pressure ---------------------------------------------------------
    fm, om = f.ensmean_mslp.values, obs.mslp.values
    put("obs_mslp", om)
    put("bias_mslp", fm - om)
    put("err_mslp", np.abs(fm - om))

    # ---- geopotential height ----------------------------------------------
    if "ensmean_z500" in f and "z500" in obs:
        fz, oz = f.ensmean_z500.values, obs.z500.values
        put("bias_z500", fz - oz)
        put("err_z500", np.abs(fz - oz))

    ds = xr.Dataset(
        out,
        coords={"lead": np.array(leads, dtype="int16"), "lat": f.lat, "lon": f.lon},
        attrs={
            **{k: v for k, v in fc.attrs.items() if k in ("data_mode", "warning", "forecast_cycle")},
            "verification_rule": "valid_time = cycle + lead; 24 h precipitation accumulation",
            "units": "precip mm/day, t2m K, wind m/s, mslp hPa, z500 m",
        },
    )
    return ds


def aggregate_error_stats(err: xr.Dataset, dims=("lat", "lon")) -> dict:
    """Domain-mean MAE / RMSE / bias per lead - used for the MAE-by-lead plots."""
    stats = {}
    for var, label in (("err_precip", "precip"), ("err_t2m", "t2m"),
                       ("err_wind", "wind"), ("err_mslp", "mslp")):
        if var not in err:
            continue
        mae = err[var].mean(dim=dims)
        rmse = np.sqrt((err[var] ** 2).mean(dim=dims))
        stats[label] = {
            "lead": [int(x) for x in err.lead.values],
            "mae": [round(float(v), 4) for v in mae.values],
            "rmse": [round(float(v), 4) for v in rmse.values],
        }
        bias_var = f"bias_{label}" if f"bias_{label}" in err else None
        if bias_var:
            stats[label]["bias"] = [round(float(v), 4) for v in err[bias_var].mean(dim=dims).values]
    return stats
