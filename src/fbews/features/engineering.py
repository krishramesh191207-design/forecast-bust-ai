"""Feature engineering (PARTS 8, 9).

Feature blocks, each of which can be switched on and off for the ablation
study (PART 38):

    nwp        forecast atmospheric state and derived dynamics
    ens        ensemble statistics and probabilistic quantities
    vol        forecast-cycle volatility (consecutive-cycle revisions)
    ana        historical analogue statistics (added later in the pipeline)
    regime     weather-regime / event indicators
    ctx        temporal and static context (lead time, season, terrain)

Everything here is computed from information available **at forecast time**.
No verifying observation ever enters a predictor - that is enforced by keeping
error/bust columns strictly out of `FEATURE_BLOCKS`.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import xarray as xr

from ..grid import gradient, laplacian, neighbourhood_std

# ---------------------------------------------------------------------------
# Feature block registry. Filled in by the builders below.
# ---------------------------------------------------------------------------
FEATURE_BLOCKS: dict[str, list[str]] = {
    "nwp": [],
    "ens": [],
    "vol": [],
    "ana": ["ana_similarity", "ana_bust_rate", "ana_mean_error", "ana_n_analogues",
            "ana_error_spread"],
    "regime": [],
    "ctx": [],
}

ANALOGUE_EMBEDDING = [
    "nwp_precip", "nwp_tcwv", "nwp_cape", "nwp_mslp_anom", "nwp_wspd850",
    "nwp_mfc", "nwp_shear", "nwp_z500_anom", "ens_precip_std", "ens_cv_precip",
    "vol_precip_rev", "ctx_sin_doy", "ctx_cos_doy", "ctx_lat", "ctx_lon",
]


def _register(block: str, names: list[str]) -> None:
    for n in names:
        if n not in FEATURE_BLOCKS[block]:
            FEATURE_BLOCKS[block].append(n)


# ---------------------------------------------------------------------------
# NWP state features
# ---------------------------------------------------------------------------
def nwp_state_features(fc: xr.Dataset, lead: int, lats, lons, static: xr.Dataset) -> dict[str, np.ndarray]:
    f = fc.sel(lead=lead)
    g = lambda name: f[name].values.astype("float32")  # noqa: E731

    precip = g("det_precip")
    tcwv = g("det_tcwv")
    cape = g("det_cape")
    mslp = g("det_mslp")
    t2m = g("det_t2m")
    u850, v850 = g("det_u850"), g("det_v850")
    u10, v10 = g("det_u10"), g("det_v10")
    z500 = g("det_z500")
    t850 = g("det_t850")
    rh700 = g("det_rh700")
    shear = g("det_shear")

    dpx, dpy = gradient(mslp, lats, lons)
    dzx, dzy = gradient(z500, lats, lons)
    dtx, dty = gradient(t850, lats, lons)
    dux, duy = gradient(u850, lats, lons)
    dvx, dvy = gradient(v850, lats, lons)
    qu, qv = tcwv * u850, tcwv * v850
    dqux, _ = gradient(qu, lats, lons)
    _, dqvy = gradient(qv, lats, lons)

    feats = {
        "nwp_precip": precip,
        "nwp_log_precip": np.log1p(precip),
        "nwp_t2m": t2m,
        "nwp_tcwv": tcwv,
        "nwp_cape": cape,
        "nwp_rh700": rh700,
        "nwp_mslp": mslp,
        "nwp_mslp_anom": mslp - np.nanmean(mslp),
        "nwp_z500_anom": z500 - np.nanmean(z500),
        "nwp_wspd10": np.hypot(u10, v10),
        "nwp_wspd850": np.hypot(u850, v850),
        "nwp_shear": shear,
        # gradients and dynamics
        "nwp_mslp_grad": np.hypot(dpx, dpy) * 1e5,             # hPa per 100 km
        "nwp_z500_grad": np.hypot(dzx, dzy) * 1e5,
        "nwp_thermal_grad": np.hypot(dtx, dty) * 1e5,
        "nwp_vorticity850": (dvx - duy) * 1e5,
        "nwp_divergence850": (dux + dvy) * 1e5,
        "nwp_mfc": -(dqux + dqvy) * 86400.0,                   # mm/day
        "nwp_mslp_laplacian": laplacian(mslp, lats, lons) * 1e10,
        "nwp_lapse_proxy": t2m - t850,
        "nwp_pw_anom": tcwv - np.nanmean(tcwv),
    }
    _register("nwp", list(feats))
    return feats


# ---------------------------------------------------------------------------
# Ensemble features
# ---------------------------------------------------------------------------
def ensemble_features(fc: xr.Dataset, lead: int) -> dict[str, np.ndarray]:
    f = fc.sel(lead=lead)
    pm = f.ensmean_precip.values.astype("float32")
    ps = f.ensstd_precip.values.astype("float32")
    q25, q75 = f.precip_q25.values, f.precip_q75.values
    q90 = f.precip_q90.values

    feats = {
        "ens_precip_mean": pm,
        "ens_precip_std": ps,
        "ens_precip_var": ps ** 2,
        "ens_precip_iqr": (q75 - q25).astype("float32"),
        "ens_precip_q90": q90.astype("float32"),
        "ens_cv_precip": ps / np.maximum(pm, 0.5),
        "ens_pop_gt10": f.pop_gt10.values.astype("float32"),
        "ens_pop_gt25": f.pop_gt25.values.astype("float32"),
        "ens_pop_gt50": f.pop_gt50.values.astype("float32"),
        "ens_t2m_std": f.ensstd_t2m.values.astype("float32"),
        "ens_t2m_range": f.t2m_range.values.astype("float32"),
        "ens_mslp_std": f.ensstd_mslp.values.astype("float32"),
        "ens_mslp_depth": (f.ensmean_mslp.values - f.mslp_min.values).astype("float32"),
        "ens_u10_std": f.ensstd_u10.values.astype("float32"),
        "ens_v10_std": f.ensstd_v10.values.astype("float32"),
        "ens_wind_std": np.hypot(f.ensstd_u10.values, f.ensstd_v10.values).astype("float32"),
        "ens_z500_std": f.ensstd_z500.values.astype("float32"),
        "ens_tcwv_std": f.ensstd_tcwv.values.astype("float32"),
        "ens_cape_std": f.ensstd_cape.values.astype("float32"),
        # disagreement between the ensemble mean and the control member
        "ens_mean_ctrl_diff_precip": np.abs(pm - f.det_precip.values).astype("float32"),
        "ens_mean_ctrl_diff_mslp": np.abs(f.ensmean_mslp.values - f.det_mslp.values).astype("float32"),
    }
    _register("ens", list(feats))
    return feats


# ---------------------------------------------------------------------------
# Spatial features
# ---------------------------------------------------------------------------
def spatial_features(base: dict[str, np.ndarray], lats, lons) -> dict[str, np.ndarray]:
    precip = base["nwp_precip"]
    spread = base["ens_precip_std"]
    mslp = base["nwp_mslp"]
    feats = {
        "spa_precip_nbhd_std": neighbourhood_std(precip, 1),
        "spa_precip_nbhd_std2": neighbourhood_std(precip, 2),
        "spa_spread_nbhd_std": neighbourhood_std(spread, 1),
        "spa_mslp_nbhd_std": neighbourhood_std(mslp, 1),
        "spa_precip_laplacian": np.abs(laplacian(precip, lats, lons)) * 1e10,
        # precipitation organisation: local max relative to neighbourhood mean
        "spa_precip_peakiness": precip / np.maximum(_nbhd_mean(precip, 2), 0.2),
    }
    _register("nwp", list(feats))
    return feats


def _nbhd_mean(field: np.ndarray, radius: int) -> np.ndarray:
    pad = np.pad(field, radius, mode="edge")
    n = 2 * radius + 1
    acc = np.zeros_like(field, dtype="float32")
    for i in range(n):
        for j in range(n):
            acc += pad[i: i + field.shape[0], j: j + field.shape[1]]
    return acc / (n * n)


# ---------------------------------------------------------------------------
# Forecast volatility features (PART 9)
# ---------------------------------------------------------------------------
def volatility_features(
    fc: xr.Dataset,
    prev: xr.Dataset | None,
    lead: int,
    prev_lead_offset: int,
) -> dict[str, np.ndarray]:
    """Compare the current cycle with the previous cycle **for the same valid
    time**: the previous cycle must be looked at one lead further ahead.

    Returns a Forecast Volatility Index combining the revisions of
    precipitation, pressure, temperature, wind and ensemble spread.
    """
    shape = fc.sel(lead=lead).det_precip.shape
    if prev is None or (lead + prev_lead_offset) not in set(int(x) for x in prev.lead.values):
        nan = np.full(shape, np.nan, dtype="float32")
        feats = {
            "vol_precip_rev": nan, "vol_precip_rev_rel": nan.copy(),
            "vol_mslp_rev": nan.copy(), "vol_t2m_rev": nan.copy(),
            "vol_wind_rev": nan.copy(), "vol_spread_change": nan.copy(),
            "vol_displacement": nan.copy(), "vol_index": nan.copy(),
            "vol_available": np.zeros(shape, dtype="float32"),
        }
        _register("vol", list(feats))
        return feats

    cur = fc.sel(lead=lead)
    old = prev.sel(lead=lead + prev_lead_offset)

    p_rev = (cur.ensmean_precip.values - old.ensmean_precip.values).astype("float32")
    m_rev = (cur.ensmean_mslp.values - old.ensmean_mslp.values).astype("float32")
    t_rev = (cur.ensmean_t2m.values - old.ensmean_t2m.values).astype("float32")
    w_rev = np.hypot(cur.ensmean_u10.values - old.ensmean_u10.values,
                     cur.ensmean_v10.values - old.ensmean_v10.values).astype("float32")
    s_chg = (cur.ensstd_precip.values - old.ensstd_precip.values).astype("float32")

    # spatial displacement of the precipitation pattern: distance to the
    # neighbourhood cell of the previous cycle that best matches the current
    # forecast value (a cheap proxy for storm-track shift)
    disp = _displacement_proxy(cur.ensmean_precip.values, old.ensmean_precip.values, radius=2)

    # Forecast Volatility Index: normalised revision magnitudes, 0..~3
    vol_index = (
        np.abs(p_rev) / 8.0
        + np.abs(m_rev) / 1.5
        + np.abs(t_rev) / 2.0
        + w_rev / 3.0
        + np.abs(s_chg) / 4.0
    ).astype("float32")

    feats = {
        "vol_precip_rev": np.abs(p_rev),
        "vol_precip_rev_rel": np.abs(p_rev) / np.maximum(cur.ensmean_precip.values, 1.0),
        "vol_mslp_rev": np.abs(m_rev),
        "vol_t2m_rev": np.abs(t_rev),
        "vol_wind_rev": w_rev,
        "vol_spread_change": s_chg,
        "vol_displacement": disp,
        "vol_index": vol_index,
        "vol_available": np.ones(shape, dtype="float32"),
    }
    _register("vol", list(feats))
    return feats


def _displacement_proxy(cur: np.ndarray, old: np.ndarray, radius: int = 2) -> np.ndarray:
    """Distance (grid cells) to the best-matching neighbour in the old field."""
    best = np.full(cur.shape, np.inf, dtype="float32")
    dist = np.zeros(cur.shape, dtype="float32")
    pad = np.pad(old, radius, mode="edge")
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            shifted = pad[radius + di: radius + di + cur.shape[0],
                          radius + dj: radius + dj + cur.shape[1]]
            diff = np.abs(cur - shifted)
            better = diff < best
            best = np.where(better, diff, best)
            dist = np.where(better, np.hypot(di, dj), dist)
    # only meaningful where there is rain to displace
    return np.where(cur > 1.0, dist, 0.0).astype("float32")


# ---------------------------------------------------------------------------
# Temporal / static context
# ---------------------------------------------------------------------------
def context_features(cycle: dt.date, lead: int, lats, lons, static: xr.Dataset,
                     indices: dict[str, float]) -> dict[str, np.ndarray]:
    valid = cycle + dt.timedelta(days=lead)
    doy = valid.timetuple().tm_yday
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    ones = np.ones_like(la, dtype="float32")
    season = _season_index(valid.month)
    feats = {
        "ctx_lead": ones * lead,
        "ctx_sin_doy": ones * np.sin(2 * np.pi * doy / 365.25),
        "ctx_cos_doy": ones * np.cos(2 * np.pi * doy / 365.25),
        "ctx_season": ones * season,
        "ctx_lat": la.astype("float32"),
        "ctx_lon": lo.astype("float32"),
        "ctx_terrain": static.terrain.values.astype("float32"),
        "ctx_terrain_grad": np.hypot(*gradient(static.terrain.values.astype("float32"), lats, lons)) * 1e5,
        "ctx_land": static.land.values.astype("float32"),
        "ctx_mjo_amp": ones * float(indices.get("mjo_amplitude", np.nan)),
        "ctx_mjo_sin": ones * float(np.sin(indices.get("mjo_phase", 0.0))),
        "ctx_mjo_cos": ones * float(np.cos(indices.get("mjo_phase", 0.0))),
        "ctx_heat_index": ones * float(indices.get("heat_index", np.nan)),
        "ctx_jet_index": ones * float(indices.get("jet_index", np.nan)),
    }
    _register("ctx", list(feats))
    return feats


def _season_index(month: int) -> int:
    if month in (6, 7, 8, 9):
        return 0    # southwest monsoon
    if month in (10, 11):
        return 1    # post-monsoon
    if month in (12, 1, 2):
        return 2    # winter
    return 3        # pre-monsoon
