# ForeSight — Forecast Reliability and Bust Early Warning System

**FBEWS** predicts *where a medium-range weather forecast is likely to fail*, Day 1
to Day 10, over India and the surrounding region — with a calibrated bust
probability, an expected error per variable, and a meteorological explanation of
why confidence is low.

It is not a weather dashboard. It never shows a forecast without also showing how
much to trust it.

| | |
|---|---|
| Backend | FastAPI + scikit-learn, Python 3, served on port **8008** |
| Frontend | static HTML/CSS/JS — no build step, no npm |
| Sandbox grid | 1.0° over 5–40°N, 60–100°E (0.25° is the design target) |
| Forecast range | D1–D10, 314 cycles in the shipped sandbox |
| Tests | 136 (`make test`) |

---

## ⚠️ Read this first: which data is running underneath

This deployment runs in **SANDBOX mode** (`data_mode: sandbox` in
`configs/default.yaml`). Every meteorological field is synthetic, produced by
`src/fbews/sandbox/toy_model.py` — a stochastic toy atmosphere with a monsoon
trough, propagating monsoon lows, tropical cyclones and western disturbances.

The real-data ingestion path is implemented but is not what feeds the running
service. Run `make probe` to have every source checked live from your machine (reasons
and enable instructions abridged here, printed in full by the command):

```
reachable  ECMWF Open Data (IFS / AIFS real-time)
reachable  NOAA Global Ensemble Forecast System (GEFS)
reachable  NOAA Global Forecast System (GFS)
reachable  IBTrACS tropical cyclone best tracks
reachable  RMM Madden-Julian Oscillation index
reachable  Climate indices (ONI/Nino3.4, DMI, NAO, AO)
BLOCKED    TIGGE      - environment variables not set: ECMWF_API_*
BLOCKED    ERA5       - environment variables not set: CDSAPI_*
BLOCKED    GPM IMERG  - environment variables not set: EARTHDATA_TOKEN
BLOCKED    IMD gridded - www.imdpune.gov.in not reachable from this host
BLOCKED    INSAT-3D   - environment variables not set: MOSDAC_*
```

The API, every product payload and the dashboard header all carry a
`DEMO / SAMPLE DATA - SYNTHETIC SANDBOX` banner. Nothing synthetic is ever
presented as an observation or as real NWP output. Two components *are* real: the
land–sea mask (Natural Earth via `global_land_mask`) and the India state
boundaries (`datta07/INDIAN-SHAPEFILES`, MIT).

**Consequently: no number in this repository is a scientific claim about real
forecast busts.** The metrics show the pipeline learns a genuine signal in the
sandbox — nothing more. Read [docs/limitations.md](docs/limitations.md) before
trusting anything.

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
make install      # pip install -r requirements.txt
make all          # sandbox -> boundaries -> features -> labels -> train -> evaluate
make run-api      # dashboard on http://localhost:8008, OpenAPI on /docs
make test         # 136 tests
```

Every Make target runs `python3`, so keep the virtualenv **activated** (or pass
`PY=.venv/bin/python` explicitly). The Makefile already exports
`PYTHONPATH=src`; no `pip install -e` is needed.

`make run-demo` does `all` and `run-api` in one shot. If the dataset and model
artefacts already exist in your checkout, `make run-api` alone is enough.

Docker: `docker compose up --build`. The image runs the full pipeline
(`scripts/cli.py all`) during the *build*, then serves the API on
`${PORT:-8008}` with `./data` and `./artifacts` mounted as volumes and a health
check on `/api/health`.

Open <http://localhost:8008> for the dashboard, <http://localhost:8008/docs> for
the interactive OpenAPI documentation.

---

## What it does

| Capability | Where |
|---|---|
| Bust probability + confidence, D1–D10, every grid cell | `/api/products`, map |
| Predicted error per variable (rain, temp, wind, pressure) | `/api/error-prediction` |
| "Why is confidence low?" in meteorological language | `/api/explanations` |
| Historical analogue search over past forecast situations | `/api/analogs` |
| Forecast Volatility Index (cycle-to-cycle revision) | volatility layer, `/api/forecast-stability` |
| Forecast comparison vs the previous cycle | `/api/forecast-comparison` |
| Ensemble spread / member agreement | `/api/ensemble` |
| Forecast Reliability Watchlist and grouped watchlists | `/api/watchlist`, `/api/watchlist-groups` |
| Alert rules, severity and escalation | `/api/alerts`, `POST /api/alerts/evaluate` |
| Synoptic systems board (monsoon lows, WDs, cyclones) | `/api/systems` |
| Case study: *could we have warned?* | `/api/case-study/{date}` |
| Event impact, region profiles, forecast brief, trust | `/api/event-impact`, `/api/region-profile`, `/api/forecast-brief`, `/api/trust` |
| Model performance, ablation, spatial CV, drift | `/api/model-metrics`, `/api/model-drift` |
| Warning performance and verification | `/api/warning-performance` |
| Dataset catalogue, live probing, data-quality log | `/api/datasets`, `/api/data-quality` |
| Point inference from a JSON body | `POST /api/inference` |
| CSV export of all layers for a cycle/lead | `/api/export/csv` |

## Pipeline

```
sandbox / real ingestion        src/fbews/sandbox, src/fbews/ingestion
   -> regrid + quality checks   src/fbews/preprocessing
   -> forecast/obs verification src/fbews/verification/errors.py
   -> error targets + bust labels  src/fbews/verification/bust.py
   -> features (6 blocks)       src/fbews/features
   -> analogue index            src/fbews/analogues
   -> weather regimes           src/fbews/regimes
   -> model ladder + calibration   src/fbews/models
   -> evaluation + ablation     src/fbews/evaluation
   -> derived layers            src/fbews/derived (change, stability, ensemble,
                                  systems, alerts, warning)
   -> products + explanations   src/fbews/inference, src/fbews/explainability
   -> REST API + dashboard      src/fbews/api, frontend/
```

Every stage is individually runnable and resumable through `scripts/cli.py`:

| Command | Stage | Output |
|---|---|---|
| `make sandbox` | synthetic dataset generation | `data/processed/` |
| `make boundaries` | India state boundaries (downloads on first run) | `data/samples/india_states_simplified.geojson` |
| `make features` | training table | `data/features/` |
| `make labels` | bust labels + analogue enrichment | `data/labels/` |
| `make train` | model ladder, regressors, importance, model card | `artifacts/models/` |
| `make evaluate` | ablation, spatial CV, extreme-event evaluation | `artifacts/models/` |
| `make infer` | products JSON for the latest cycle (optional — the API computes on demand) | `data/products/` |
| `make download` | real-data ingestion (needs network + credentials) | `data/raw/` |
| `make probe` | reachability report for all 11 real sources | stdout |

`python scripts/cli.py sandbox --quick` builds a much shorter period for a
smoke test.

---

## Project layout

```
.
├── Makefile              all pipeline targets (install / all / run-api / test)
├── requirements.txt      numpy … scikit-learn … fastapi … pydantic, ingestion clients
├── configs/
│   ├── default.yaml      master config: mode, paths, domain, grid, splits, bust rules, api
│   └── regions.yaml      region bounding boxes for region-wise aggregation
├── src/fbews/            backend (45 modules, ~8 200 lines)
│   ├── api/              main.py (core API), platform.py (Intelligence/Operations), schemas.py
│   ├── config.py         config + credential loading, FBEWS_ROOT / FBEWS_CONFIG / FBEWS_DATA_MODE
│   ├── grid.py           analysis grid, real land–sea mask, idealised terrain
│   ├── sandbox/          generate.py, toy_model.py
│   ├── ingestion/        clients.py (per-provider fetchers), sources.py (catalogue + probe)
│   ├── preprocessing/    quality.py (checks), regrid.py (conservative / bilinear)
│   ├── features/         build.py, engineering.py, systems.py
│   ├── verification/     errors.py, bust.py, labelling.py
│   ├── analogues/        engine.py
│   ├── regimes/          classify.py
│   ├── models/           ladder.py, train.py
│   ├── evaluation/       ablation.py, metrics.py
│   ├── inference/        run.py (InferenceEngine)
│   ├── explainability/   narrative.py
│   └── derived/          change.py, stability.py, ensemble.py, systems.py, alerts.py, warning.py
├── frontend/             static dashboard — no build step
│   ├── index.html        ~4 800 lines: markup, CSS, core app
│   ├── fbews-platform.js shared selection state + shared metric helpers
│   ├── fbews-intel.js    Forecast Intelligence pane
│   ├── fbews-ops.js      Operations Center pane
│   └── assets/           ncmrwf-logo.png
├── scripts/
│   ├── cli.py            every pipeline stage
│   └── prepare_boundaries.py  fetch + simplify state boundaries
├── tests/                8 files, 136 tests
├── data/                 raw / processed / features / labels / samples / DATASETS.md
├── artifacts/models/     trained models, metrics, model card, ablation, spatial CV
├── docs/                 architecture, methodology, model, verification, api, limitations
└── docker/, docker-compose.yml, render.yaml, railway.toml
```

---

## Dashboard

The frontend is plain static files served by FastAPI at `/` (HTML) and `/app`
(static mount). There is **no `package.json`, no bundler and no npm** — the only
external resource is the Inter webfont from Google Fonts.

Eight views, each a full section of `index.html`:

| View | Contents |
|---|---|
| Overview | map, KPI header, collapsible side cards (watchlist, model, drivers, case), analysis detail panel |
| Regions | region × lead tables |
| Forecast Intelligence | tabs: Overview · Drift · Stability · Ensemble · Analogues · Drivers · Scenarios · Risk Evolution |
| Operations Center | tabs: Situation · Active Alerts · Systems · Regions · Rules · History · Performance · Trust · Brief |
| Case Study | evolution of one valid date |
| Model Performance | ladder, regressors, ablation, spatial CV, feature importance |
| Data Sources | dataset catalogue with live probing |
| Methodology | how every number is produced |

The map is a hand-rolled **Canvas 2D** renderer (no Leaflet/maplibre/d3) drawing
the simplified state boundaries from `/api/boundaries` and labels from
`/api/geo-labels`. The analysis detail panel has five tabs: Overview · Change ·
Drivers · Timeline · Diagnostics.

---

## API

FastAPI app, version and description from `configs/default.yaml`. **38 paths**
(36 GET, 2 POST); the authoritative, always-current list is the OpenAPI schema at
`/docs` / `/openapi.json`. Grouped by tag:

- **system** — `/api/health`, `/api/meta`, `/api/datasets`, `/api/data-quality`,
  `/api/boundaries`, `/api/geo-labels`
- **forecast** — `/api/cycles`, `/api/forecast/current`, `/api/products`
- **products** — `/api/confidence`, `/api/bust-probability`,
  `/api/error-prediction`, `/api/layers`, `/api/regions`, `/api/watchlist`,
  `/api/grid/{lat}/{lon}`, `/api/forecast-comparison`, `/api/forecast-stability`,
  `/api/forecast-stability-regions`, `/api/export/csv`, `POST /api/inference`
- **explainability** — `/api/explanations`, `/api/analogs`
- **evaluation** — `/api/model-metrics`
- **events** — `/api/historical-events`, `/api/case-study/{date}`
- **forecast-intelligence** — `/api/ensemble`
- **operations-center** — `/api/alerts`, `POST /api/alerts/evaluate`,
  `/api/systems`, `/api/watchlist-groups`, `/api/warning-performance`,
  `/api/event-impact`, `/api/region-profile`, `/api/forecast-brief`,
  `/api/model-drift`, `/api/trust`, `/api/search`

Errors use standard HTTP codes: 400 outside the domain, 404 unknown cycle or
case-study date, 422 invalid lead, 429 rate limited (120 requests/minute/IP, see
`api.rate_limit_per_minute`), 503 artefacts not built. CORS origins come from
`api.cors_origins` (default `*`). Full endpoint descriptions and an example point
response: [docs/api.md](docs/api.md).

---

## Headline results (sandbox, held-out test split 2024-06 → 2025-05)

Test split 52,470 rows; training 206,910 rows; 83 features; calibration fitted on
the validation split only. Bust base rate in the test split: **9.67 %** — any
PR-AUC above that beats random ranking.

| Model | Brier ↓ | Brier skill | ROC-AUC | PR-AUC | Recall | ECE |
|---|---|---|---|---|---|---|
| climatology | 0.08734 | 0.0002 | 0.5254 | 0.1022 | 0.0000 | 0.0078 |
| spread only | 0.07229 | 0.1725 | 0.7923 | 0.3846 | 0.1015 | 0.0096 |
| volatility only | 0.07350 | 0.1586 | 0.8247 | 0.3551 | 0.0513 | 0.0121 |
| logistic regression | 0.06190 | 0.2914 | 0.8751 | 0.5254 | 0.2572 | 0.0117 |
| **random forest** | **0.06123** | **0.2991** | **0.8744** | 0.5248 | 0.2846 | 0.0126 |
| gradient boosting | 0.06223 | 0.2876 | 0.8640 | 0.5155 | 0.3000 | 0.0131 |

The random forest wins on Brier and skill; logistic regression is level with it
on PR-AUC (0.5254 vs 0.5248), and gradient boosting buys nothing on either —
so the added complexity is not justified. Full tables, ablation, spatial
cross-validation and extreme-event breakdowns: [docs/model.md](docs/model.md).

---

## Configuration and environment variables

All configuration lives in `configs/default.yaml` (override the file with
`FBEWS_CONFIG=/path/to.yaml`). Three environment switches change behaviour:

| Variable | Purpose |
|---|---|
| `FBEWS_DATA_MODE` | `sandbox` (default) or `real` |
| `FBEWS_CONFIG` | path to a config file, default `configs/default.yaml` |
| `FBEWS_ROOT` | project root override (defaults to the repo checkout) |

Credentials are only needed for real-data ingestion. Copy `.env.example` to
`.env` and fill in what you need:

```bash
cp .env.example .env
export $(grep -v '^#' .env | xargs)
```

| Variable | Source |
|---|---|
| `ECMWF_API_URL`, `ECMWF_API_KEY`, `ECMWF_API_EMAIL` | TIGGE via MARS |
| `CDSAPI_URL`, `CDSAPI_KEY` | Copernicus CDS (ERA5) |
| `EARTHDATA_TOKEN` | NASA Earthdata (GPM IMERG) |
| `MOSDAC_USER`, `MOSDAC_PASSWORD` | MOSDAC (INSAT-3D/3DR) |

ECMWF Open Data, NOAA GEFS/GFS, IBTrACS, the RMM index and the climate indices
need no credentials. Each client degrades to a clear `BLOCKED` message rather
than crashing — see `make probe`.

---

## Switching to real data

1. `cp .env.example .env` and fill in credentials (see table above).
2. `export $(grep -v '^#' .env | xargs)` and set `FBEWS_DATA_MODE=real`.
3. In `configs/default.yaml`, replace `bust.absolute_thresholds` with the
   `bust.real_data_absolute_thresholds` starting values already in the file
   (35 mm/day, 4 K, 6 m/s, 4 hPa) — the defaults are scaled to the sandbox error
   climatology.
4. `make download`, then `make all`.

Read [docs/verification.md](docs/verification.md) first: the IMD 0830 IST
accumulation window and GRIB accumulation conventions will silently corrupt every
error metric if mishandled.

## Retraining

```bash
make features labels train evaluate
```

Splits are chronological and configured in `configs/default.yaml`
(train 2019-06-01 → 2023-05-31, val → 2024-05-31, test → 2025-05-31); never
shuffle weather samples across time. Each run writes a model card to
`artifacts/models/model_card.json` with the period, feature blocks, metrics,
calibration method and git commit. `make clean` removes all generated data and
model artefacts.

## Tests

```bash
source .venv/bin/activate    # the venv must be active, see Quick start
make test                    # python3 -m pytest tests -q  ->  136 passed (~2 min)
```

136 tests across 8 files: pipeline/grid/feature units, derived rules (change,
stability, alerts, systems, warning), geo-label integrity, and API integration
tests for both the core API and the Forecast Intelligence / Operations Center
routes. The API integration tests skip themselves automatically when model
artefacts have not been built, so a fresh clone can still run the unit suite.

There is no linter configured — `lint` appears in `.PHONY` in the Makefile but
has no recipe.

## Deployment

| Path | Detail |
|---|---|
| Docker | `docker/Dockerfile` — python:3.12-slim + eccodes/netCDF4, runs `scripts/cli.py all` at build, `EXPOSE 8008` |
| Compose | `docker compose up --build` — port `8008:8008`, health check on `/api/health`, mounts `./data`, `./artifacts` |
| Render | `render.yaml` — Docker runtime, `healthCheckPath: /api/health` |
| Railway | `railway.toml` — DOCKERFILE builder, `healthcheckPath: /api/health` |

No authentication is implemented, and the in-process rate limiter is a
prototype-grade counter — see [docs/limitations.md](docs/limitations.md).

## Documentation

- [data/DATASETS.md](data/DATASETS.md) — full dataset catalogue and selection
- [docs/architecture.md](docs/architecture.md) — system structure and compute profile
- [docs/methodology.md](docs/methodology.md) — data model, feature blocks, analogue engine, model ladder
- [docs/model.md](docs/model.md) — results, ablation, spatial CV
- [docs/verification.md](docs/verification.md) — units, windows, regridding, leakage rules
- [docs/api.md](docs/api.md) — endpoint reference (see `/docs` for the live schema)
- [docs/limitations.md](docs/limitations.md) — **read before trusting anything**

## Licence and attribution

Code: MIT. Boundaries: datta07/INDIAN-SHAPEFILES (MIT). Land mask: Natural Earth
via `global_land_mask`. Dashboard header uses the NCMRWF logo
(`frontend/assets/ncmrwf-logo.png`) and the Government of India emblem. Real
datasets retain their providers' licences — ECMWF open data CC-BY-4.0, TIGGE
research/education only, ERA5 Copernicus licence, IMD data policy, NOAA products
public domain.
