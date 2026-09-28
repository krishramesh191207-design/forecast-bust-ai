"""Real-data ingestion clients.

These are the code paths used when `data_mode: real`. They are written
against the providers' documented interfaces. In an environment without
network egress or credentials they raise `SourceUnavailable` with an explicit
BLOCKED BY / REASON / HOW TO ENABLE message - they never fall back to
synthetic data.

Optional third-party clients (`ecmwf-opendata`, `cdsapi`, `ecmwfapi`,
`imdlib`, `cfgrib`) are imported lazily so the prototype runs without them.
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path

from .sources import SOURCES


class SourceUnavailable(RuntimeError):
    """Raised when a real data source cannot be used in this environment."""

    def __init__(self, key: str, reason: str, how_to_enable: str):
        self.key, self.reason, self.how_to_enable = key, reason, how_to_enable
        super().__init__(
            f"BLOCKED BY: {key}\nREASON: {reason}\nHOW TO ENABLE: {how_to_enable}"
        )


@dataclass
class Retrieval:
    """Result of a download: the files written plus provenance."""

    source: str
    files: list[Path]
    provenance: dict


def _require(module: str, key: str, pip_name: str | None = None):
    try:
        return __import__(module, fromlist=["*"])
    except ImportError as exc:
        raise SourceUnavailable(
            key,
            f"python client `{module}` is not installed ({exc})",
            f"pip install {pip_name or module}",
        ) from exc


def _require_env(names: list[str], key: str):
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise SourceUnavailable(
            key,
            f"missing credentials: {', '.join(missing)}",
            f"Register at {SOURCES[key].url} and set {', '.join(missing)} in .env",
        )


# ---------------------------------------------------------------------------
# Live forecast: ECMWF Open Data (IFS/AIFS, CC-BY-4.0, no key)
# ---------------------------------------------------------------------------
def fetch_ecmwf_opendata(
    target_dir: Path,
    cycle: dt.datetime,
    stream: str = "enfo",
    params: tuple[str, ...] = ("tp", "2t", "10u", "10v", "msl"),
    steps: tuple[int, ...] = tuple(range(24, 241, 24)),
) -> Retrieval:
    """Download one ECMWF open-data ensemble cycle (GRIB2)."""
    od = _require("ecmwf.opendata", "ecmwf_opendata", "ecmwf-opendata")
    target_dir.mkdir(parents=True, exist_ok=True)
    client = od.Client(source="ecmwf", model="ifs", resol="0p25")
    out = target_dir / f"ecmwf_{stream}_{cycle:%Y%m%d%H}.grib2"
    try:
        result = client.retrieve(
            date=cycle.strftime("%Y%m%d"),
            time=cycle.hour,
            stream=stream,
            type="pf" if stream == "enfo" else "fc",
            param=list(params),
            step=list(steps),
            target=str(out),
        )
    except Exception as exc:  # network or availability failure
        raise SourceUnavailable(
            "ecmwf_opendata",
            f"retrieval failed: {type(exc).__name__}: {exc}",
            "Ensure outbound HTTPS to data.ecmwf.int is allowed; open data needs no key.",
        ) from exc
    return Retrieval(
        source="ecmwf_opendata",
        files=[out],
        provenance={
            "cycle": cycle.isoformat(),
            "stream": stream,
            "params": list(params),
            "steps": list(steps),
            "licence": SOURCES["ecmwf_opendata"].licence,
            "datetime": getattr(result, "datetime", None).__str__(),
        },
    )


# ---------------------------------------------------------------------------
# Live/near-real-time ensemble: NOAA GEFS on AWS (anonymous, open)
# ---------------------------------------------------------------------------
GEFS_BASE = "https://noaa-gefs-pds.s3.amazonaws.com"


def gefs_urls(cycle: dt.datetime, members: int = 21, steps: tuple[int, ...] = tuple(range(24, 241, 24))) -> list[str]:
    """Construct the documented GEFS object keys for a cycle.

    Layout: gefs.YYYYMMDD/HH/atmos/pgrb2ap5/gecXX.tHHz.pgrb2a.0p50.fFFF
    """
    urls = []
    for m in range(members):
        name = "gec00" if m == 0 else f"gep{m:02d}"
        for s in steps:
            urls.append(
                f"{GEFS_BASE}/gefs.{cycle:%Y%m%d}/{cycle:%H}/atmos/pgrb2ap5/"
                f"{name}.t{cycle:%H}z.pgrb2a.0p50.f{s:03d}"
            )
    return urls


def fetch_gefs(target_dir: Path, cycle: dt.datetime, members: int = 21,
               steps: tuple[int, ...] = tuple(range(24, 241, 24))) -> Retrieval:
    import urllib.error
    import urllib.request

    target_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    urls = gefs_urls(cycle, members, steps)
    for url in urls:
        out = target_dir / url.rsplit("/", 1)[-1]
        if out.exists():
            files.append(out)
            continue
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, open(out, "wb") as fh:
                fh.write(resp.read())
        except Exception as exc:
            raise SourceUnavailable(
                "gefs",
                f"cannot download {url}: {type(exc).__name__}: {exc}",
                "Allow outbound HTTPS to noaa-gefs-pds.s3.amazonaws.com (no credentials needed).",
            ) from exc
        files.append(out)
    return Retrieval(
        source="gefs",
        files=files,
        provenance={"cycle": cycle.isoformat(), "members": members, "steps": list(steps),
                    "licence": SOURCES["gefs"].licence},
    )


# ---------------------------------------------------------------------------
# Historical forecast archive: TIGGE via MARS
# ---------------------------------------------------------------------------
def fetch_tigge(
    target_dir: Path,
    start: dt.date,
    end: dt.date,
    origin: str = "ecmf",
    levtype: str = "sfc",
    params: str = "167/165/166/151/228228",
    steps: str = "/".join(str(s) for s in range(24, 241, 24)),
    area: str = "40/60/5/100",
    grid: str = "0.5/0.5",
) -> Retrieval:
    """Submit a TIGGE MARS request (perturbed forecast members).

    Parameter codes above are 2t/10u/10v/msl/total precipitation in the TIGGE
    parameter table. `origin` selects the contributing centre
    (ecmf, kwbc=NCEP, egrr=UKMO, rjtd=JMA, cwao=CMC, ...).
    """
    _require_env(["ECMWF_API_KEY", "ECMWF_API_EMAIL"], "tigge")
    api = _require("ecmwfapi", "tigge", "ecmwf-api-client")
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / f"tigge_{origin}_{start:%Y%m%d}_{end:%Y%m%d}_{levtype}.grib"
    server = api.ECMWFDataServer(
        url=os.environ.get("ECMWF_API_URL", "https://api.ecmwf.int/v1"),
        key=os.environ["ECMWF_API_KEY"],
        email=os.environ["ECMWF_API_EMAIL"],
    )
    request = {
        "class": "ti",
        "dataset": "tigge",
        "date": f"{start:%Y-%m-%d}/to/{end:%Y-%m-%d}",
        "expver": "prod",
        "grid": grid,
        "area": area,
        "levtype": levtype,
        "origin": origin,
        "param": params,
        "step": steps,
        "number": "1/to/50",
        "time": "00:00:00",
        "type": "pf",
        "target": str(out),
    }
    try:
        server.retrieve(request)
    except Exception as exc:
        raise SourceUnavailable(
            "tigge",
            f"MARS retrieval failed: {type(exc).__name__}: {exc}",
            "Accept the TIGGE licence at https://apps.ecmwf.int/datasets/licences/tigge/ "
            "and check ECMWF_API_KEY/ECMWF_API_EMAIL.",
        ) from exc
    return Retrieval("tigge", [out], {"request": request, "licence": SOURCES["tigge"].licence})


# ---------------------------------------------------------------------------
# Verification: ERA5 via the Copernicus CDS API
# ---------------------------------------------------------------------------
ERA5_SINGLE = [
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "mean_sea_level_pressure", "total_precipitation", "total_column_water_vapour",
    "convective_available_potential_energy",
]
ERA5_PRESSURE = ["geopotential", "temperature", "u_component_of_wind",
                 "v_component_of_wind", "relative_humidity"]


def fetch_era5(
    target_dir: Path,
    start: dt.date,
    end: dt.date,
    area: tuple[float, float, float, float] = (40, 60, 5, 100),   # N, W, S, E
    levels: tuple[int, ...] = (200, 300, 500, 700, 850, 925, 1000),
    single_levels: list[str] | None = None,
) -> Retrieval:
    _require_env(["CDSAPI_KEY"], "era5")
    cdsapi = _require("cdsapi", "era5", "cdsapi")
    target_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(
        url=os.environ.get("CDSAPI_URL", "https://cds.climate.copernicus.eu/api"),
        key=os.environ["CDSAPI_KEY"],
    )
    files = []
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    common = {
        "product_type": "reanalysis",
        "year": sorted({f"{d:%Y}" for d in days}),
        "month": sorted({f"{d:%m}" for d in days}),
        "day": sorted({f"{d:%d}" for d in days}),
        "time": [f"{h:02d}:00" for h in range(0, 24, 3)],
        "area": list(area),
        "data_format": "netcdf",
    }
    out_sfc = target_dir / f"era5_sfc_{start:%Y%m%d}_{end:%Y%m%d}.nc"
    out_pl = target_dir / f"era5_pl_{start:%Y%m%d}_{end:%Y%m%d}.nc"
    try:
        client.retrieve(
            "reanalysis-era5-single-levels",
            {**common, "variable": single_levels or ERA5_SINGLE},
            str(out_sfc),
        )
        client.retrieve(
            "reanalysis-era5-pressure-levels",
            {**common, "variable": ERA5_PRESSURE, "pressure_level": [str(p) for p in levels]},
            str(out_pl),
        )
    except Exception as exc:
        raise SourceUnavailable(
            "era5",
            f"CDS retrieval failed: {type(exc).__name__}: {exc}",
            "Create a CDS account, accept the ERA5 licence, and set CDSAPI_KEY.",
        ) from exc
    files += [out_sfc, out_pl]
    return Retrieval("era5", files, {"area": list(area), "levels": list(levels),
                                     "licence": SOURCES["era5"].licence})


# ---------------------------------------------------------------------------
# Verification: IMD gridded rainfall / temperature
# ---------------------------------------------------------------------------
def fetch_imd(target_dir: Path, start_year: int, end_year: int,
              variable: str = "rain") -> Retrieval:
    imdlib = _require("imdlib", "imd_gridded", "imdlib")
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        data = imdlib.get_data(variable, start_year, end_year, fn_format="yearwise",
                               file_dir=str(target_dir))
        out = target_dir / f"imd_{variable}_{start_year}_{end_year}.nc"
        data.to_netcdf(out.name, str(target_dir))
    except Exception as exc:
        raise SourceUnavailable(
            "imd_gridded",
            f"IMD download failed: {type(exc).__name__}: {exc}",
            "Check access to www.imdpune.gov.in; the portal occasionally rate-limits. "
            "Files can also be downloaded manually and placed in data/raw/imd/.",
        ) from exc
    return Retrieval("imd_gridded", [out], {
        "variable": variable, "years": [start_year, end_year],
        "accumulation_window": "0830 IST to 0830 IST",
        "licence": SOURCES["imd_gridded"].licence,
    })


# ---------------------------------------------------------------------------
# Verification: GPM IMERG daily
# ---------------------------------------------------------------------------
def imerg_url(day: dt.date, run: str = "final", version: str = "07B") -> str:
    product = {"final": "GPM_3IMERGDF.07", "late": "GPM_3IMERGDL.07", "early": "GPM_3IMERGDE.07"}[run]
    stem = {"final": "3B-DAY", "late": "3B-DAY-L", "early": "3B-DAY-E"}[run]
    return (
        f"https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/{product}/"
        f"{day:%Y}/{day:%m}/{stem}.MS.MRG.3IMERG.{day:%Y%m%d}-S000000-E235959.V{version}.nc4"
    )


def fetch_imerg(target_dir: Path, start: dt.date, end: dt.date, run: str = "final") -> Retrieval:
    _require_env(["EARTHDATA_TOKEN"], "imerg")
    import urllib.request

    target_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ["EARTHDATA_TOKEN"]
    files = []
    day = start
    while day <= end:
        url = imerg_url(day, run)
        out = target_dir / url.rsplit("/", 1)[-1]
        if not out.exists():
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
            try:
                with urllib.request.urlopen(req, timeout=120) as resp, open(out, "wb") as fh:
                    fh.write(resp.read())
            except Exception as exc:
                raise SourceUnavailable(
                    "imerg",
                    f"GES DISC download failed for {day}: {type(exc).__name__}: {exc}",
                    "Create an Earthdata Login, authorise the GES DISC application, "
                    "and set EARTHDATA_TOKEN.",
                ) from exc
        files.append(out)
        day += dt.timedelta(days=1)
    return Retrieval("imerg", files, {"run": run, "licence": SOURCES["imerg"].licence})


# ---------------------------------------------------------------------------
# Events and indices
# ---------------------------------------------------------------------------
IBTRACS_CSV = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-"
    "stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
)
RMM_TXT = "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt"


def fetch_ibtracs(target_dir: Path) -> Retrieval:
    import urllib.request

    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / "ibtracs.NI.list.v04r01.csv"
    try:
        urllib.request.urlretrieve(IBTRACS_CSV, out)
    except Exception as exc:
        raise SourceUnavailable(
            "ibtracs",
            f"download failed: {type(exc).__name__}: {exc}",
            "Allow outbound HTTPS to www.ncei.noaa.gov (no credentials needed).",
        ) from exc
    return Retrieval("ibtracs", [out], {"basin": "North Indian", "licence": SOURCES["ibtracs"].licence})


def fetch_rmm(target_dir: Path) -> Retrieval:
    import urllib.request

    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / "rmm.74toRealtime.txt"
    try:
        urllib.request.urlretrieve(RMM_TXT, out)
    except Exception as exc:
        raise SourceUnavailable(
            "mjo_rmm",
            f"download failed: {type(exc).__name__}: {exc}",
            "Allow outbound HTTP to www.bom.gov.au.",
        ) from exc
    return Retrieval("mjo_rmm", [out], {"licence": SOURCES["mjo_rmm"].licence})


REGISTRY = {
    "ecmwf_opendata": fetch_ecmwf_opendata,
    "gefs": fetch_gefs,
    "tigge": fetch_tigge,
    "era5": fetch_era5,
    "imd_gridded": fetch_imd,
    "imerg": fetch_imerg,
    "ibtracs": fetch_ibtracs,
    "mjo_rmm": fetch_rmm,
}
