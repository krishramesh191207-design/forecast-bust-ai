"""Regridding every dataset onto the common analysis grid.

Rules used throughout the project (documented in docs/verification.md):

* State variables (temperature, pressure, geopotential, winds, humidity) are
  interpolated **bilinearly**.
* Fluxes and accumulations (precipitation) are remapped **conservatively**
  when the source grid is finer than the target grid, so that area totals are
  preserved; a first-order area-weighted block mean is used, which is exact
  for the integer-ratio case (e.g. IMERG 0.1 deg -> 0.25 deg is handled by
  weighted averaging of partially overlapping cells).
* Missing values are preserved as NaN and never filled with zeros. Cells
  whose valid-data fraction falls below `min_valid_fraction` become NaN.
* Longitudes are normalised to the 0-360 convention then subset to the domain;
  latitudes are always stored ascending.

`xesmf` is used when available (it wraps ESMF and handles curvilinear grids);
otherwise the pure-numpy implementations below are used, which are sufficient
for the regular lat/lon grids of all sources in this project.
"""
from __future__ import annotations

import numpy as np
import xarray as xr


def normalise_coords(ds: xr.Dataset, lat_name: str | None = None, lon_name: str | None = None) -> xr.Dataset:
    """Rename coordinates to lat/lon, sort latitude ascending, lon to 0-360."""
    ren = {}
    for cand in ("latitude", "Latitude", "LAT", "y"):
        if cand in ds.coords or cand in ds.dims:
            ren[cand] = "lat"
    for cand in ("longitude", "Longitude", "LON", "x"):
        if cand in ds.coords or cand in ds.dims:
            ren[cand] = "lon"
    if lat_name:
        ren[lat_name] = "lat"
    if lon_name:
        ren[lon_name] = "lon"
    ds = ds.rename(ren) if ren else ds
    if "lon" in ds.coords:
        ds = ds.assign_coords(lon=(ds.lon % 360))
        ds = ds.sortby("lon")
    if "lat" in ds.coords and ds.lat.size > 1 and float(ds.lat[0]) > float(ds.lat[-1]):
        ds = ds.isel(lat=slice(None, None, -1))
    return ds


def subset_domain(ds: xr.Dataset, domain: dict, pad: float = 2.0) -> xr.Dataset:
    return ds.sel(
        lat=slice(domain["lat_min"] - pad, domain["lat_max"] + pad),
        lon=slice(domain["lon_min"] - pad, domain["lon_max"] + pad),
    )


def bilinear(ds: xr.Dataset | xr.DataArray, lats: np.ndarray, lons: np.ndarray):
    """Bilinear interpolation onto the analysis grid (state variables)."""
    return ds.interp(lat=lats, lon=lons, method="linear", kwargs={"fill_value": np.nan})


def conservative(da: xr.DataArray, lats: np.ndarray, lons: np.ndarray,
                 min_valid_fraction: float = 0.5) -> xr.DataArray:
    """Area-weighted conservative remap of a flux/accumulation field.

    Weights are the overlap area between source and target cells on a
    cos(lat)-weighted sphere. Target cells with less than
    `min_valid_fraction` of valid source area become NaN.
    """
    src_lat = da["lat"].values.astype("float64")
    src_lon = da["lon"].values.astype("float64")
    wlat = _overlap_weights(src_lat, lats.astype("float64"), weight_cos=True)
    wlon = _overlap_weights(src_lon, lons.astype("float64"), weight_cos=False)

    vals = da.values.astype("float64")
    valid = np.isfinite(vals)
    filled = np.where(valid, vals, 0.0)

    # contract over the trailing (lat, lon) axes
    num = np.einsum("...ij,ki,lj->...kl", filled, wlat, wlon)
    den = np.einsum("...ij,ki,lj->...kl", valid.astype("float64"), wlat, wlon)
    tot = np.einsum("ki,lj->kl", wlat, wlon)
    out = np.where(den > min_valid_fraction * tot, num / np.maximum(den, 1e-12), np.nan)

    dims = list(da.dims[:-2]) + ["lat", "lon"]
    coords = {d: da[d] for d in da.dims[:-2] if d in da.coords}
    coords.update({"lat": lats, "lon": lons})
    return xr.DataArray(out.astype("float32"), dims=dims, coords=coords, attrs=da.attrs)


def _edges(centres: np.ndarray) -> np.ndarray:
    step = np.diff(centres)
    step = np.append(step, step[-1])
    return np.append(centres - step / 2.0, centres[-1] + step[-1] / 2.0)


def _overlap_weights(src: np.ndarray, tgt: np.ndarray, weight_cos: bool) -> np.ndarray:
    """Matrix W[k, i] = overlap of target cell k with source cell i."""
    se, te = _edges(src), _edges(tgt)
    lo = np.maximum(te[:-1, None], se[None, :-1])
    hi = np.minimum(te[1:, None], se[None, 1:])
    w = np.clip(hi - lo, 0.0, None)
    if weight_cos:
        mid = 0.5 * (np.maximum(lo, -90) + np.minimum(hi, 90))
        w = w * np.cos(np.deg2rad(np.clip(mid, -89.9, 89.9)))
    return w


def regrid_dataset(ds: xr.Dataset, lats: np.ndarray, lons: np.ndarray,
                   flux_vars: tuple[str, ...] = ("precip", "tp", "total_precipitation", "rain")) -> xr.Dataset:
    """Regrid a whole dataset, choosing the method per variable."""
    ds = normalise_coords(ds)
    out = {}
    for name, da in ds.data_vars.items():
        if name in flux_vars:
            out[name] = conservative(da, lats, lons)
        else:
            out[name] = bilinear(da, lats, lons)
    return xr.Dataset(out, attrs={**ds.attrs, "regridded_to": f"{len(lats)}x{len(lons)} analysis grid"})
