"""Common analysis grid, land-sea mask and terrain height.

The land-sea mask is REAL data: it comes from the `global_land_mask` package,
which bundles a 1/100-degree land mask derived from Natural Earth coastlines.

The terrain field is an *idealised* smooth orography used by the sandbox
generator (Himalaya arc, Western/Eastern Ghats, Sulaiman-Kirthar ranges). In
`real` data mode the terrain is read from ERA5 surface geopotential instead;
see `terrain_height()`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # real land mask, bundled with the package
    from global_land_mask import globe

    _HAS_LANDMASK = True
except Exception:  # pragma: no cover - only when optional dep missing
    _HAS_LANDMASK = False

EARTH_R = 6371000.0


def make_grid(domain: dict, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (lats, lons) of the common analysis grid, latitude ascending."""
    lats = np.arange(domain["lat_min"], domain["lat_max"] + 1e-9, resolution)
    lons = np.arange(domain["lon_min"], domain["lon_max"] + 1e-9, resolution)
    return lats.astype("float32"), lons.astype("float32")


def mesh(lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2-D (lat, lon) meshes with shape (nlat, nlon)."""
    return np.meshgrid(lats, lons, indexing="ij")


def land_mask(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Boolean land mask on the analysis grid (True = land)."""
    la, lo = mesh(lats, lons)
    if _HAS_LANDMASK:
        lon180 = np.where(lo > 180.0, lo - 360.0, lo)
        return globe.is_land(la.astype("float64"), lon180.astype("float64"))
    raise RuntimeError(
        "global_land_mask is not installed; install it or supply a land mask "
        "NetCDF at data/raw/land_mask.nc"
    )


def _ridge(la, lo, lat0, lon0, lat1, lon1, width, height):
    """Height contribution of a smooth ridge between two end points."""
    dx, dy = lon1 - lon0, lat1 - lat0
    seg2 = dx * dx + dy * dy
    t = np.clip(((lo - lon0) * dx + (la - lat0) * dy) / seg2, 0.0, 1.0)
    px, py = lon0 + t * dx, lat0 + t * dy
    d2 = (lo - px) ** 2 + (la - py) ** 2
    return height * np.exp(-d2 / (2.0 * width**2))


def terrain_height(lats: np.ndarray, lons: np.ndarray, source: Path | None = None) -> np.ndarray:
    """Terrain height in metres on the analysis grid.

    If `source` points to a NetCDF file containing ERA5 surface geopotential
    (variable `z`), real orography is used. Otherwise an idealised smooth
    orography is returned for use by the sandbox generator.
    """
    if source is not None and Path(source).exists():
        import xarray as xr

        ds = xr.open_dataset(source)
        z = ds["z"].squeeze()
        return (z / 9.80665).interp(latitude=lats, longitude=lons).values.astype("float32")

    la, lo = mesh(lats, lons)
    h = np.zeros_like(la, dtype="float32")
    # Himalaya / Karakoram arc
    h += _ridge(la, lo, 34.5, 73.0, 28.0, 88.0, 1.9, 4200.0)
    h += _ridge(la, lo, 28.0, 88.0, 28.5, 96.0, 1.7, 3400.0)
    # Tibetan plateau shoulder
    h += _ridge(la, lo, 34.0, 80.0, 33.0, 95.0, 3.2, 4000.0)
    # Western Ghats
    h += _ridge(la, lo, 20.5, 73.2, 8.5, 77.2, 0.9, 1250.0)
    # Eastern Ghats
    h += _ridge(la, lo, 19.5, 84.0, 12.0, 78.5, 1.0, 620.0)
    # Sulaiman / Kirthar / Hindu Kush
    h += _ridge(la, lo, 36.0, 70.0, 28.0, 67.0, 1.4, 2600.0)
    # Arakan Yoma
    h += _ridge(la, lo, 25.0, 94.0, 17.0, 94.5, 0.8, 1500.0)
    # Sea level over ocean
    if _HAS_LANDMASK:
        h = np.where(land_mask(lats, lons), np.maximum(h, 20.0), 0.0)
    return h.astype("float32")


def gradient(field: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal gradient (d/dx, d/dy) in units per metre on a lat/lon grid."""
    dlat = np.deg2rad(float(lats[1] - lats[0])) * EARTH_R
    dlon = np.deg2rad(float(lons[1] - lons[0])) * EARTH_R
    coslat = np.cos(np.deg2rad(lats))[:, None]
    dfdy = np.gradient(field, axis=0) / dlat
    dfdx = np.gradient(field, axis=1) / (dlon * np.maximum(coslat, 0.05))
    return dfdx, dfdy


def laplacian(field: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    dfdx, dfdy = gradient(field, lats, lons)
    d2x, _ = gradient(dfdx, lats, lons)
    _, d2y = gradient(dfdy, lats, lons)
    return d2x + d2y


def neighbourhood_std(field: np.ndarray, radius: int = 1) -> np.ndarray:
    """Standard deviation in a (2r+1)^2 neighbourhood, edge-padded."""
    pad = np.pad(field, radius, mode="edge")
    stack = []
    n = 2 * radius + 1
    for i in range(n):
        for j in range(n):
            stack.append(pad[i : i + field.shape[0], j : j + field.shape[1]])
    arr = np.stack(stack, axis=0)
    return arr.std(axis=0)
