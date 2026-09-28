"""Unit and integration tests for FBEWS."""
import datetime as dt
import numpy as np
import pandas as pd
import pytest

from fbews.config import load_config, load_regions
from fbews.grid import gradient, land_mask, make_grid, terrain_height
from fbews.features.systems import detect_lows, proximity_fields
from fbews.preprocessing.quality import check_dataset
from fbews.preprocessing.regrid import _overlap_weights, conservative, normalise_coords
from fbews.verification.bust import BustDefinition
from fbews.verification.errors import valid_time
from fbews.evaluation.metrics import classification_metrics, reliability

cfg = load_config()


# ---- grid -----------------------------------------------------------------
def test_grid_alignment():
    lats, lons = make_grid(cfg.domain, cfg.resolution)
    assert lats[0] == cfg.domain["lat_min"] and lats[-1] <= cfg.domain["lat_max"]
    assert np.all(np.diff(lats) > 0), "latitude must be ascending"
    assert np.all(np.diff(lons) > 0)
    assert abs(float(lats[1] - lats[0]) - cfg.resolution) < 1e-6


def test_land_mask_known_points():
    lats, lons = make_grid(cfg.domain, 1.0)
    mask = land_mask(lats, lons)
    i = int(np.argmin(abs(lats - 20.0)))
    j = int(np.argmin(abs(lons - 78.0)))
    assert mask[i, j], "central India must be land"
    j_sea = int(np.argmin(abs(lons - 65.0)))
    i_sea = int(np.argmin(abs(lats - 15.0)))
    assert not mask[i_sea, j_sea], "Arabian Sea must be ocean"


def test_terrain_is_high_over_himalaya():
    lats, lons = make_grid(cfg.domain, 1.0)
    h = terrain_height(lats, lons)
    i = int(np.argmin(abs(lats - 33.0))); j = int(np.argmin(abs(lons - 78.0)))
    i2 = int(np.argmin(abs(lats - 20.0))); j2 = int(np.argmin(abs(lons - 80.0)))
    assert h[i, j] > 2000 and h[i, j] > h[i2, j2]


def test_gradient_of_linear_field():
    lats, lons = make_grid(cfg.domain, 1.0)
    la, _ = np.meshgrid(lats, lons, indexing="ij")
    dfdx, dfdy = gradient(la.astype("float32"), lats, lons)
    assert np.allclose(dfdy[2:-2, 2:-2], 1.0 / (np.deg2rad(1.0) * 6371000.0), rtol=1e-3)
    assert np.allclose(dfdx[2:-2, 2:-2], 0.0, atol=1e-12)


# ---- lead time / verification ---------------------------------------------
def test_valid_time_rule():
    assert valid_time(dt.date(2024, 7, 20), 5) == dt.datetime(2024, 7, 25)
    assert valid_time(dt.datetime(2024, 1, 1), 10) == dt.datetime(2024, 1, 11)


# ---- regridding ------------------------------------------------------------
def test_conservative_regrid_preserves_mass():
    import xarray as xr
    fine_lat = np.arange(10.0, 20.01, 0.25)
    fine_lon = np.arange(70.0, 80.01, 0.25)
    rng = np.random.default_rng(0)
    data = rng.gamma(2.0, 3.0, size=(len(fine_lat), len(fine_lon)))
    da = xr.DataArray(data, dims=("lat", "lon"), coords={"lat": fine_lat, "lon": fine_lon})
    coarse_lat = np.arange(10.5, 19.51, 1.0)
    coarse_lon = np.arange(70.5, 79.51, 1.0)
    out = conservative(da, coarse_lat, coarse_lon)
    w = np.cos(np.deg2rad(fine_lat))[:, None]
    src_mean = float((data * w).sum() / (w * np.ones_like(data)).sum())
    wc = np.cos(np.deg2rad(coarse_lat))[:, None]
    dst_mean = float((out.values * wc).sum() / (wc * np.ones_like(out.values)).sum())
    assert abs(src_mean - dst_mean) / src_mean < 0.05


def test_overlap_weights_sum():
    src = np.arange(0.0, 10.01, 0.5)
    tgt = np.arange(0.5, 9.51, 1.0)
    w = _overlap_weights(src, tgt, weight_cos=False)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-9)


def test_missing_values_are_not_filled_with_zero():
    import xarray as xr
    lat = np.arange(10.0, 12.01, 0.25); lon = np.arange(70.0, 72.01, 0.25)
    data = np.full((len(lat), len(lon)), np.nan)
    da = xr.DataArray(data, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    out = conservative(da, np.arange(10.5, 11.51, 1.0), np.arange(70.5, 71.51, 1.0))
    assert np.isnan(out.values).all(), "all-missing input must stay missing, never become 0"


def test_normalise_coords_sorts_latitude():
    import xarray as xr
    ds = xr.Dataset({"x": (("latitude", "longitude"), np.zeros((3, 3)))},
                    coords={"latitude": [30.0, 20.0, 10.0], "longitude": [-10.0, 0.0, 10.0]})
    out = normalise_coords(ds)
    assert list(out.lat.values) == [10.0, 20.0, 30.0]
    assert out.lon.min() >= 0


# ---- bust labelling --------------------------------------------------------
def _fake_frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "lead": rng.integers(1, 11, n),
        "err_precip": rng.gamma(1.5, 2.0, n),
        "err_t2m": rng.gamma(2.0, 0.6, n),
        "err_wind": rng.gamma(2.0, 1.2, n),
        "err_mslp": rng.gamma(2.0, 0.4, n),
    })


def test_bust_thresholds_and_labels():
    df = _fake_frame()
    bd = BustDefinition.from_config(cfg).fit(df)
    out = bd.label(df)
    assert "bust_overall" in out and set(out["bust_overall"].unique()) <= {0, 1}
    # hybrid: every labelled bust exceeds both criteria for at least one variable
    hit = out[out["bust_precipitation"] == 1]
    if len(hit):
        assert (hit["err_precip"] > bd.absolute["precipitation_mm_day"]).all()
        assert (hit["err_precip"] > hit["bustthr_precipitation"]).all()
    assert 0.0 <= out["bust_overall"].mean() <= 0.5


def test_bust_percentile_only_mode():
    df = _fake_frame()
    bd = BustDefinition.from_config(cfg)
    bd.method = "percentile"
    bd.fit(df)
    out = bd.label(df)
    rate = out["bust_precipitation"].mean()
    assert 0.05 < rate < 0.15, "percentile mode should flag about 10 percent at p90"


def test_bust_definition_roundtrip(tmp_path):
    bd = BustDefinition.from_config(cfg).fit(_fake_frame())
    p = bd.save(tmp_path / "bd.json")
    again = BustDefinition.load(p)
    assert again.thresholds == bd.thresholds and again.method == bd.method


# ---- system detection ------------------------------------------------------
def test_detect_lows_finds_planted_low():
    lats, lons = make_grid(cfg.domain, 1.0)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    mslp = 1008.0 - 8.0 * np.exp(-(((la - 18.0) ** 2 + (lo - 88.0) ** 2) / (2 * 3.0 ** 2)))
    land = land_mask(lats, lons)
    centres = detect_lows(mslp.astype("float32"), lats, lons, land, month=7)
    assert centres, "a deep planted low must be detected"
    best = min(centres, key=lambda c: (c["lat"] - 18.0) ** 2 + (c["lon"] - 88.0) ** 2)
    assert abs(best["lat"] - 18.0) <= 2 and abs(best["lon"] - 88.0) <= 2
    prox = proximity_fields(centres, lats, lons)
    assert prox["dist_monsoon_low"].min() < 2.0


# ---- metrics ---------------------------------------------------------------
def test_classification_metrics_perfect_and_random():
    y = np.array([0, 0, 1, 1] * 50)
    perfect = y.astype(float) * 0.99 + 0.005
    m = classification_metrics(y, perfect)
    assert m["roc_auc"] == 1.0 and m["brier"] < 0.01
    rng = np.random.default_rng(1)
    m2 = classification_metrics(y, rng.uniform(size=len(y)))
    assert 0.3 < m2["roc_auc"] < 0.7


def test_reliability_perfectly_calibrated():
    rng = np.random.default_rng(3)
    p = rng.uniform(size=40000)
    y = (rng.uniform(size=40000) < p).astype(int)
    r = reliability(y, p)
    assert r["expected_calibration_error"] < 0.02


# ---- config ----------------------------------------------------------------
def test_regions_cover_domain_and_are_valid():
    for r in load_regions():
        a, b, c, d = r["bbox"]
        assert a < b and c < d
        assert -90 <= a <= 90 and 0 <= c <= 360


def test_config_lead_days_are_1_to_10():
    assert cfg.lead_days == list(range(1, 11))


# ---- data quality ----------------------------------------------------------
def test_quality_flags_impossible_values():
    import xarray as xr
    ds = xr.Dataset({"det_t2m": (("lat", "lon"), np.full((3, 3), 1000.0))},
                    coords={"lat": [1.0, 2.0, 3.0], "lon": [10.0, 11.0, 12.0]})
    rec = check_dataset(ds, "bad")
    assert not rec["ok"] and any("plausible range" in i["message"] for i in rec["issues"])


def test_quality_detects_missing_leads():
    import xarray as xr
    ds = xr.Dataset({"det_precip": (("lead", "lat", "lon"), np.zeros((2, 2, 2)))},
                    coords={"lead": [1, 2], "lat": [1.0, 2.0], "lon": [10.0, 11.0]})
    rec = check_dataset(ds, "short", expected_leads=[1, 2, 3])
    assert not rec["ok"]
