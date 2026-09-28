# Dataset catalogue

Every entry below was checked against the provider's own documentation. No URL
or access method in this file is invented. Reachability from any given machine
can be probed at runtime with `GET /api/datasets?probe=true`.

**Status in this deployment:** none of these sources is reachable from the
development container (network egress is restricted to package registries), so
the prototype runs in a clearly-labelled synthetic sandbox mode. The clients in
`src/fbews/ingestion/clients.py` are real implementations against these
interfaces; see "How to switch to real data" in README.md.

---

## Selection summary (PART 56)

| Priority | Dataset | Purpose | Historical depth | Resolution | Access difficulty | Recommended |
|---|---|---|---|---|---|---|
| 1 | TIGGE (ECMWF archive) | historical ensemble forecasts to learn error patterns | 2006-present | 0.5 deg retrieval | medium (registration + licence) | yes |
| 1 | ERA5 (CDS) | reference atmospheric state for verification | 1940-present | 0.25 deg | medium (free token) | yes |
| 1 | IMD gridded rainfall | primary rainfall verification over India | 1901-present | 0.25 deg | low (`imdlib`) | yes |
| 2 | NOAA GEFS (AWS) | live ensemble + spread for operational inference | 2017-present | 0.5 deg | low (anonymous S3) | yes |
| 2 | ECMWF Open Data | live IFS/AIFS forecasts, CC-BY | rolling days | 0.25 deg | low (no key) | yes |
| 3 | GPM IMERG | gauge-independent precipitation check | 1998-present | 0.1 deg | medium (Earthdata login) | yes, as cross-check |
| 3 | IBTrACS v04r01 | cyclone event labelling and stratified evaluation | 1842-present | point tracks | low | yes |
| 4 | MJO RMM, ONI, DMI | regime context features | 1974/1950-present | index | low | yes, low weight |
| 5 | INSAT-3D/3DR via MOSDAC | satellite QPE/OLR predictors | 2013-present | 4 km | high (registration, no open API) | phase 2 only |
| - | NOAA GFS | deterministic companion for multi-model disagreement | rolling | 0.25 deg | low | optional |

**Primary stack chosen:** TIGGE for historical forecasts, ERA5 + IMD gridded for
verification, GEFS/ECMWF Open Data for live inference, IMERG as an independent
precipitation cross-check, IBTrACS for cyclone events, RMM/ONI/DMI for regime
context. MOSDAC is deliberately deferred: registration friction and the lack of
an open programmatic API make it a poor fit for a reproducible pipeline, and its
predictors overlap with ERA5 moisture/instability fields.

---

## Full catalogue

### TIGGE (THORPEX Interactive Grand Global Ensemble)

| Field | Value |
|---|---|
| Provider | ECMWF (archive centre) / 13 NWP centres |
| Official URL | https://confluence.ecmwf.int/display/TIGGE |
| Role in this project | **historical forecast** |
| Variables | total_precipitation, 2t, 10u, 10v, msl, gh, t, q, u, v, cape, tcw, sm |
| Spatial resolution | 0.12 to 0.9375 deg depending on centre (retrieved on a 0.5 deg grid) |
| Temporal resolution | 6-hourly steps, 1-2 cycles per day per centre |
| Historical period | 2006-10 to present |
| Format | GRIB2 |
| Access mechanism | MARS / ECMWF Data Store (ECDS) request, 48-hour embargo, research licence |
| Python client | `ecmwfapi.ECMWFDataServer` |
| API available | see access |
| Credentials required | ECMWF_API_KEY, ECMWF_API_EMAIL, ECMWF_API_URL |
| Licensing | Non-commercial research/education; registration and licence acceptance required |
| Host probed by /api/datasets | `apps.ecmwf.int` |
| Notes | Primary historical multi-centre ensemble archive: ECMWF, NCEP, UKMO, JMA, CMC and others. |


### ECMWF Open Data (IFS / AIFS real-time)

| Field | Value |
|---|---|
| Provider | ECMWF |
| Official URL | https://www.ecmwf.int/en/forecasts/datasets/open-data |
| Role in this project | **live forecast** |
| Variables | tp, 2t, 10u, 10v, msl, gh, t, q, u, v |
| Spatial resolution | 0.25 deg |
| Temporal resolution | 4 cycles/day (00/06/12/18Z), 3-6 hourly steps to 360 h |
| Historical period | rolling ~4 days of real-time data |
| Format | GRIB2 |
| Access mechanism | Anonymous HTTPS via the ecmwf-opendata client; no key required |
| Python client | `ecmwf.opendata.Client` |
| API available | yes |
| Credentials required | none |
| Licensing | CC-BY-4.0 with attribution (ECMWF Terms of Use) |
| Host probed by /api/datasets | `data.ecmwf.int` |
| Notes | Used for live/operational inference. ENS members are available in the open subset. |


### NOAA Global Ensemble Forecast System (GEFS)

| Field | Value |
|---|---|
| Provider | NOAA NCEP / AWS Open Data (NODD) |
| Official URL | https://registry.opendata.aws/noaa-gefs/ |
| Role in this project | **live forecast** |
| Variables | APCP, TMP, UGRD, VGRD, PRMSL, HGT, RH, CAPE, PWAT |
| Spatial resolution | 0.5 deg (pgrb2a) and 0.25 deg for selected fields |
| Temporal resolution | 4 cycles/day, 3-hourly to 240 h then 6-hourly to 384 h |
| Historical period | 2017 to present in s3://noaa-gefs-pds |
| Format | GRIB2 |
| Access mechanism | Anonymous S3 / HTTPS, no account required |
| Python client | `httpx + cfgrib (byte-range GRIB index reads)` |
| API available | yes |
| Credentials required | none |
| Licensing | Open data, no use restrictions |
| Host probed by /api/datasets | `noaa-gefs-pds.s3.amazonaws.com` |
| Notes | 21 members (gec00 control + gep01..gep20). Ensemble spread is a primary predictor here. |


### NOAA Global Forecast System (GFS)

| Field | Value |
|---|---|
| Provider | NOAA NCEP / AWS Open Data |
| Official URL | https://registry.opendata.aws/noaa-gfs-bdp-pds/ |
| Role in this project | **live forecast** |
| Variables | APCP, TMP, UGRD, VGRD, PRMSL, HGT, CAPE, PWAT |
| Spatial resolution | 0.25 deg |
| Temporal resolution | 4 cycles/day, hourly to 120 h then 3-hourly to 384 h |
| Historical period | rolling multi-week archive (noaa-gfs-bdp-pds); longer archives at NCEI |
| Format | GRIB2 |
| Access mechanism | Anonymous S3 / HTTPS |
| Python client | `httpx + cfgrib` |
| API available | yes |
| Credentials required | none |
| Licensing | Open data, no use restrictions |
| Host probed by /api/datasets | `noaa-gfs-bdp-pds.s3.amazonaws.com` |
| Notes | Deterministic companion to GEFS; used for multi-model disagreement features. |


### ERA5 reanalysis (single and pressure levels)

| Field | Value |
|---|---|
| Provider | ECMWF / Copernicus Climate Data Store |
| Official URL | https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels |
| Role in this project | **verification** |
| Variables | 2m_temperature, 10m_u_component_of_wind, 10m_v_component_of_wind, mean_sea_level_pressure, total_precipitation, total_column_water_vapour, convective_available_potential_energy, geopotential, temperature, u_component_of_wind, v_component_of_wind, specific_humidity, volumetric_soil_water_layer_1 |
| Spatial resolution | 0.25 deg |
| Temporal resolution | hourly |
| Historical period | 1940 to present (ERA5T within ~5 days) |
| Format | NetCDF or GRIB |
| Access mechanism | CDS API, free account and Personal Access Token required |
| Python client | `cdsapi.Client` |
| API available | yes |
| Credentials required | CDSAPI_URL, CDSAPI_KEY |
| Licensing | Copernicus Licence (free, attribution required) |
| Host probed by /api/datasets | `cds.climate.copernicus.eu` |
| Notes | Reference atmospheric state for verification of all non-precipitation variables. |


### IMD gridded daily rainfall (0.25 deg) and temperature (1.0 deg)

| Field | Value |
|---|---|
| Provider | India Meteorological Department, Pune |
| Official URL | https://www.imdpune.gov.in/cmpg/Griddata/Public_Data_Download.html |
| Role in this project | **verification** |
| Variables | rain, tmax, tmin |
| Spatial resolution | 0.25 deg (rain), 1.0 deg (temperature) |
| Temporal resolution | daily (0830 IST to 0830 IST accumulation for rainfall) |
| Historical period | rainfall 1901-present, temperature 1951-present |
| Format | IMD binary .grd (converted to NetCDF by imdlib) |
| Access mechanism | Public download portal; the `imdlib` python package automates it |
| Python client | `imdlib.get_data / imdlib.get_real_data` |
| API available | see access |
| Credentials required | none |
| Licensing | IMD data policy - check terms for redistribution |
| Host probed by /api/datasets | `www.imdpune.gov.in` |
| Notes | Primary rainfall verification over India. NOTE the 0830 IST accumulation window: forecasts must be accumulated over the same window (see docs/verification.md). |


### GPM IMERG precipitation (Early / Late / Final)

| Field | Value |
|---|---|
| Provider | NASA / JAXA, GES DISC |
| Official URL | https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGDF_07/summary |
| Role in this project | **verification** |
| Variables | precipitation |
| Spatial resolution | 0.1 deg |
| Temporal resolution | 30-minute and daily |
| Historical period | 1998 (V07B, TRMM era) to present |
| Format | HDF5 / NetCDF |
| Access mechanism | Earthdata Login required; HTTPS or OPeNDAP from GES DISC |
| Python client | `requests with Earthdata token` |
| API available | yes |
| Credentials required | EARTHDATA_TOKEN |
| Licensing | NASA open data, free with registration |
| Host probed by /api/datasets | `disc.gsfc.nasa.gov` |
| Notes | Latency: Early 4 h, Late 14 h, Final ~3.5 months. Independent of gauge network. |


### IBTrACS v04r01 tropical cyclone best tracks

| Field | Value |
|---|---|
| Provider | NOAA NCEI |
| Official URL | https://www.ncei.noaa.gov/products/international-best-track-archive |
| Role in this project | **events** |
| Variables | lat, lon, wmo_wind, wmo_pres, storm_speed, nature |
| Spatial resolution | point tracks |
| Temporal resolution | 3-hourly |
| Historical period | 1842 to present |
| Format | CSV / NetCDF / shapefile |
| Access mechanism | Anonymous HTTPS download, updated three times per week |
| Python client | `pandas.read_csv` |
| API available | yes |
| Credentials required | none |
| Licensing | US Government work, no copyright |
| Host probed by /api/datasets | `www.ncei.noaa.gov` |
| Notes | North Indian basin subset used for cyclone event labelling. |


### RMM Madden-Julian Oscillation index (Wheeler & Hendon)

| Field | Value |
|---|---|
| Provider | Australian Bureau of Meteorology |
| Official URL | http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt |
| Role in this project | **indices** |
| Variables | RMM1, RMM2, phase, amplitude |
| Spatial resolution | global index |
| Temporal resolution | daily |
| Historical period | 1974 to present |
| Format | text |
| Access mechanism | Anonymous HTTP |
| Python client | `pandas.read_csv` |
| API available | see access |
| Credentials required | none |
| Licensing | BoM terms, attribution required |
| Host probed by /api/datasets | `www.bom.gov.au` |
| Notes | Drives the active/break monsoon regime feature. |


### Climate indices (ONI/Nino3.4, DMI, NAO, AO)

| Field | Value |
|---|---|
| Provider | NOAA Physical Sciences Laboratory / Climate Prediction Center |
| Official URL | https://psl.noaa.gov/data/climateindices/list/ |
| Role in this project | **indices** |
| Variables | nino34, dmi, nao, ao |
| Spatial resolution | global index |
| Temporal resolution | monthly (some daily) |
| Historical period | 1870/1950 to present depending on index |
| Format | text |
| Access mechanism | Anonymous HTTPS |
| Python client | `pandas.read_csv` |
| API available | yes |
| Credentials required | none |
| Licensing | US Government work, no copyright |
| Host probed by /api/datasets | `psl.noaa.gov` |
| Notes | Slow-varying context features; low weight but cheap to include. |


### INSAT-3D/3DR products (QPE, OLR, water vapour, CMV, SST)

| Field | Value |
|---|---|
| Provider | ISRO / MOSDAC, Space Applications Centre |
| Official URL | https://www.mosdac.gov.in/ |
| Role in this project | **verification** |
| Variables | QPE, OLR, WV, CMV, SST, cloud products |
| Spatial resolution | 4 km (imager) varying by product |
| Temporal resolution | 30-minute |
| Historical period | 2013 to present (INSAT-3D), 2016 to present (INSAT-3DR) |
| Format | HDF5 / NetCDF |
| Access mechanism | Registered account, web ordering and FTP; no open anonymous API |
| Python client | `manual download / MOSDAC order API` |
| API available | yes |
| Credentials required | MOSDAC_USER, MOSDAC_PASSWORD |
| Licensing | MOSDAC data policy, registration required |
| Host probed by /api/datasets | `www.mosdac.gov.in` |
| Notes | Optional satellite predictors; registration friction makes this a phase-2 source. |


---

## Verification-window warning

IMD gridded rainfall accumulates 0830 IST to 0830 IST (03 UTC to 03 UTC), while
NWP precipitation is usually accumulated from the 00 UTC cycle. The two windows
differ by three hours and must be aligned before any error is computed. This is
handled by `verification.errors` via the `obs_window_offset_hours` setting and
documented in `docs/verification.md`.
