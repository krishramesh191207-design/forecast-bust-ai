"""Registry of the real datasets this system is designed to run on.

Nothing here is invented: every URL and access mechanism was verified against
the provider's own documentation (see data/DATASETS.md for the full catalogue
with licensing, resolution and update frequency).

Each entry states the credential environment variables it needs and the python
client used. `probe()` checks whether a source is reachable from the current
environment and returns a structured BLOCKED report when it is not, so the API
and the dashboard can tell the user exactly what is missing and how to enable
it instead of silently substituting data.
"""
from __future__ import annotations

import socket
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Literal

Role = Literal["historical_forecast", "live_forecast", "verification", "events", "indices", "static"]


@dataclass
class Source:
    key: str
    name: str
    provider: str
    url: str
    role: Role
    variables: list[str]
    spatial_resolution: str
    temporal_resolution: str
    period: str
    fmt: str
    access: str
    client: str
    credentials: list[str] = field(default_factory=list)
    licence: str = ""
    host: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


SOURCES: dict[str, Source] = {
    "tigge": Source(
        key="tigge",
        name="TIGGE (THORPEX Interactive Grand Global Ensemble)",
        provider="ECMWF (archive centre) / 13 NWP centres",
        url="https://confluence.ecmwf.int/display/TIGGE",
        role="historical_forecast",
        variables=[
            "total_precipitation", "2t", "10u", "10v", "msl", "gh", "t", "q", "u", "v",
            "cape", "tcw", "sm",
        ],
        spatial_resolution="0.12 to 0.9375 deg depending on centre (retrieved on a 0.5 deg grid)",
        temporal_resolution="6-hourly steps, 1-2 cycles per day per centre",
        period="2006-10 to present",
        fmt="GRIB2",
        access="MARS / ECMWF Data Store (ECDS) request, 48-hour embargo, research licence",
        client="ecmwfapi.ECMWFDataServer",
        credentials=["ECMWF_API_KEY", "ECMWF_API_EMAIL", "ECMWF_API_URL"],
        licence="Non-commercial research/education; registration and licence acceptance required",
        host="apps.ecmwf.int",
        notes="Primary historical multi-centre ensemble archive: ECMWF, NCEP, UKMO, JMA, CMC and others.",
    ),
    "ecmwf_opendata": Source(
        key="ecmwf_opendata",
        name="ECMWF Open Data (IFS / AIFS real-time)",
        provider="ECMWF",
        url="https://www.ecmwf.int/en/forecasts/datasets/open-data",
        role="live_forecast",
        variables=["tp", "2t", "10u", "10v", "msl", "gh", "t", "q", "u", "v"],
        spatial_resolution="0.25 deg",
        temporal_resolution="4 cycles/day (00/06/12/18Z), 3-6 hourly steps to 360 h",
        period="rolling ~4 days of real-time data",
        fmt="GRIB2",
        access="Anonymous HTTPS via the ecmwf-opendata client; no key required",
        client="ecmwf.opendata.Client",
        credentials=[],
        licence="CC-BY-4.0 with attribution (ECMWF Terms of Use)",
        host="data.ecmwf.int",
        notes="Used for live/operational inference. ENS members are available in the open subset.",
    ),
    "gefs": Source(
        key="gefs",
        name="NOAA Global Ensemble Forecast System (GEFS)",
        provider="NOAA NCEP / AWS Open Data (NODD)",
        url="https://registry.opendata.aws/noaa-gefs/",
        role="live_forecast",
        variables=["APCP", "TMP", "UGRD", "VGRD", "PRMSL", "HGT", "RH", "CAPE", "PWAT"],
        spatial_resolution="0.5 deg (pgrb2a) and 0.25 deg for selected fields",
        temporal_resolution="4 cycles/day, 3-hourly to 240 h then 6-hourly to 384 h",
        period="2017 to present in s3://noaa-gefs-pds",
        fmt="GRIB2",
        access="Anonymous S3 / HTTPS, no account required",
        client="httpx + cfgrib (byte-range GRIB index reads)",
        credentials=[],
        licence="Open data, no use restrictions",
        host="noaa-gefs-pds.s3.amazonaws.com",
        notes="21 members (gec00 control + gep01..gep20). Ensemble spread is a primary predictor here.",
    ),
    "gfs": Source(
        key="gfs",
        name="NOAA Global Forecast System (GFS)",
        provider="NOAA NCEP / AWS Open Data",
        url="https://registry.opendata.aws/noaa-gfs-bdp-pds/",
        role="live_forecast",
        variables=["APCP", "TMP", "UGRD", "VGRD", "PRMSL", "HGT", "CAPE", "PWAT"],
        spatial_resolution="0.25 deg",
        temporal_resolution="4 cycles/day, hourly to 120 h then 3-hourly to 384 h",
        period="rolling multi-week archive (noaa-gfs-bdp-pds); longer archives at NCEI",
        fmt="GRIB2",
        access="Anonymous S3 / HTTPS",
        client="httpx + cfgrib",
        credentials=[],
        licence="Open data, no use restrictions",
        host="noaa-gfs-bdp-pds.s3.amazonaws.com",
        notes="Deterministic companion to GEFS; used for multi-model disagreement features.",
    ),
    "era5": Source(
        key="era5",
        name="ERA5 reanalysis (single and pressure levels)",
        provider="ECMWF / Copernicus Climate Data Store",
        url="https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels",
        role="verification",
        variables=[
            "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
            "mean_sea_level_pressure", "total_precipitation", "total_column_water_vapour",
            "convective_available_potential_energy", "geopotential", "temperature",
            "u_component_of_wind", "v_component_of_wind", "specific_humidity",
            "volumetric_soil_water_layer_1",
        ],
        spatial_resolution="0.25 deg",
        temporal_resolution="hourly",
        period="1940 to present (ERA5T within ~5 days)",
        fmt="NetCDF or GRIB",
        access="CDS API, free account and Personal Access Token required",
        client="cdsapi.Client",
        credentials=["CDSAPI_URL", "CDSAPI_KEY"],
        licence="Copernicus Licence (free, attribution required)",
        host="cds.climate.copernicus.eu",
        notes="Reference atmospheric state for verification of all non-precipitation variables.",
    ),
    "imd_gridded": Source(
        key="imd_gridded",
        name="IMD gridded daily rainfall (0.25 deg) and temperature (1.0 deg)",
        provider="India Meteorological Department, Pune",
        url="https://www.imdpune.gov.in/cmpg/Griddata/Public_Data_Download.html",
        role="verification",
        variables=["rain", "tmax", "tmin"],
        spatial_resolution="0.25 deg (rain), 1.0 deg (temperature)",
        temporal_resolution="daily (0830 IST to 0830 IST accumulation for rainfall)",
        period="rainfall 1901-present, temperature 1951-present",
        fmt="IMD binary .grd (converted to NetCDF by imdlib)",
        access="Public download portal; the `imdlib` python package automates it",
        client="imdlib.get_data / imdlib.get_real_data",
        credentials=[],
        licence="IMD data policy - check terms for redistribution",
        host="www.imdpune.gov.in",
        notes=(
            "Primary rainfall verification over India. NOTE the 0830 IST accumulation "
            "window: forecasts must be accumulated over the same window (see "
            "docs/verification.md)."
        ),
    ),
    "imerg": Source(
        key="imerg",
        name="GPM IMERG precipitation (Early / Late / Final)",
        provider="NASA / JAXA, GES DISC",
        url="https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGDF_07/summary",
        role="verification",
        variables=["precipitation"],
        spatial_resolution="0.1 deg",
        temporal_resolution="30-minute and daily",
        period="1998 (V07B, TRMM era) to present",
        fmt="HDF5 / NetCDF",
        access="Earthdata Login required; HTTPS or OPeNDAP from GES DISC",
        client="requests with Earthdata token",
        credentials=["EARTHDATA_TOKEN"],
        licence="NASA open data, free with registration",
        host="disc.gsfc.nasa.gov",
        notes="Latency: Early 4 h, Late 14 h, Final ~3.5 months. Independent of gauge network.",
    ),
    "ibtracs": Source(
        key="ibtracs",
        name="IBTrACS v04r01 tropical cyclone best tracks",
        provider="NOAA NCEI",
        url="https://www.ncei.noaa.gov/products/international-best-track-archive",
        role="events",
        variables=["lat", "lon", "wmo_wind", "wmo_pres", "storm_speed", "nature"],
        spatial_resolution="point tracks",
        temporal_resolution="3-hourly",
        period="1842 to present",
        fmt="CSV / NetCDF / shapefile",
        access="Anonymous HTTPS download, updated three times per week",
        client="pandas.read_csv",
        credentials=[],
        licence="US Government work, no copyright",
        host="www.ncei.noaa.gov",
        notes="North Indian basin subset used for cyclone event labelling.",
    ),
    "mjo_rmm": Source(
        key="mjo_rmm",
        name="RMM Madden-Julian Oscillation index (Wheeler & Hendon)",
        provider="Australian Bureau of Meteorology",
        url="http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt",
        role="indices",
        variables=["RMM1", "RMM2", "phase", "amplitude"],
        spatial_resolution="global index",
        temporal_resolution="daily",
        period="1974 to present",
        fmt="text",
        access="Anonymous HTTP",
        client="pandas.read_csv",
        credentials=[],
        licence="BoM terms, attribution required",
        host="www.bom.gov.au",
        notes="Drives the active/break monsoon regime feature.",
    ),
    "psl_indices": Source(
        key="psl_indices",
        name="Climate indices (ONI/Nino3.4, DMI, NAO, AO)",
        provider="NOAA Physical Sciences Laboratory / Climate Prediction Center",
        url="https://psl.noaa.gov/data/climateindices/list/",
        role="indices",
        variables=["nino34", "dmi", "nao", "ao"],
        spatial_resolution="global index",
        temporal_resolution="monthly (some daily)",
        period="1870/1950 to present depending on index",
        fmt="text",
        access="Anonymous HTTPS",
        client="pandas.read_csv",
        credentials=[],
        licence="US Government work, no copyright",
        host="psl.noaa.gov",
        notes="Slow-varying context features; low weight but cheap to include.",
    ),
    "mosdac": Source(
        key="mosdac",
        name="INSAT-3D/3DR products (QPE, OLR, water vapour, CMV, SST)",
        provider="ISRO / MOSDAC, Space Applications Centre",
        url="https://www.mosdac.gov.in/",
        role="verification",
        variables=["QPE", "OLR", "WV", "CMV", "SST", "cloud products"],
        spatial_resolution="4 km (imager) varying by product",
        temporal_resolution="30-minute",
        period="2013 to present (INSAT-3D), 2016 to present (INSAT-3DR)",
        fmt="HDF5 / NetCDF",
        access="Registered account, web ordering and FTP; no open anonymous API",
        client="manual download / MOSDAC order API",
        credentials=["MOSDAC_USER", "MOSDAC_PASSWORD"],
        licence="MOSDAC data policy, registration required",
        host="www.mosdac.gov.in",
        notes="Optional satellite predictors; registration friction makes this a phase-2 source.",
    ),
}


def probe(key: str, timeout: float = 4.0) -> dict:
    """Check reachability of a source host from this environment."""
    src = SOURCES[key]
    report = {
        "key": key,
        "name": src.name,
        "host": src.host,
        "reachable": False,
        "credentials_present": True,
        "blocked_by": None,
        "reason": None,
        "how_to_enable": None,
    }
    import os

    missing = [c for c in src.credentials if not os.environ.get(c)]
    if missing:
        report["credentials_present"] = False
        report["blocked_by"] = "missing credentials"
        report["reason"] = f"environment variables not set: {', '.join(missing)}"
        report["how_to_enable"] = (
            f"Register at {src.url}, then set {', '.join(missing)} in your .env file."
        )
    try:
        socket.setdefaulttimeout(timeout)
        urllib.request.urlopen(f"https://{src.host}", timeout=timeout)
        report["reachable"] = True
    except (urllib.error.HTTPError,) as exc:  # host answered, path may 403/404
        report["reachable"] = exc.code < 500
    except Exception as exc:
        report["reachable"] = False
        if report["blocked_by"] is None:
            report["blocked_by"] = "network egress"
            report["reason"] = f"{src.host} not reachable: {type(exc).__name__}"
            report["how_to_enable"] = (
                f"Allow outbound HTTPS to {src.host} (or run the pipeline on a host "
                f"with open egress) and re-run `make download`."
            )
    return report


def probe_all(timeout: float = 3.0) -> list[dict]:
    return [probe(k, timeout=timeout) for k in SOURCES]


def catalogue() -> list[dict]:
    return [s.to_dict() for s in SOURCES.values()]
