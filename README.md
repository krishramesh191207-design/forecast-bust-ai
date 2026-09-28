# ForeSight — Forecast Reliability and Bust Early Warning System

Predicts **where a medium-range weather forecast is likely to fail**, at Day 1 to
Day 10, over India and the surrounding region — with a calibrated bust
probability, an expected error per variable, and a meteorological explanation of
why confidence is low.

This is not a weather dashboard. It never shows you a forecast without also
showing you how much to trust it.

---

## ⚠️ Read this first: what data is running underneath

This deployment runs in **SANDBOX mode**. Every meteorological field is
synthetic, produced by `src/fbews/sandbox/toy_model.py` — a stochastic toy
atmosphere with a monsoon trough, propagating monsoon lows, tropical cyclones
and western disturbances.

It exists because the development environment has **no network egress to any
meteorological data provider**:

```
BLOCKED BY:  container network policy
REASON:      egress is limited to package registries; ecmwf.int,
             noaa-*.s3.amazonaws.com, cds.climate.copernicus.eu,
             imdpune.gov.in and ncei.noaa.gov are unreachable, and no
             provider credentials are present
HOW TO ENABLE: run on a host with open egress, fill in .env, set
             FBEWS_DATA_MODE=real, then `make download && make all`
```

Run `make probe` to see this checked live for every source.

The API, every product file and the dashboard header all carry a
`DEMO / SAMPLE DATA — SYNTHETIC SANDBOX` banner. Nothing synthetic is ever
presented as observation or as real NWP output. Two components *are* real: the
land–sea mask (Natural Earth via `global_land_mask`) and the India state
boundaries (`datta07/INDIAN-SHAPEFILES`, MIT).

**Consequently: no result in this repository is a scientific claim about real
forecast busts.** The metrics show the pipeline learns a genuine signal in the
sandbox — nothing more.

---

## Quick start

```bash
pip install -r requirements.txt
make all          # sandbox data -> features -> labels -> train -> evaluate  (~12 min)
make run-api      # dashboard + API on http://localhost:8008
make test         # 32 tests
```

Docker: `docker compose up --build` (first run builds the dataset and trains).

Open <http://localhost:8008> for the ForeSight dashboard, <http://localhost:8008/docs> for
the OpenAPI documentation.

---

## What it does

| Capability | Where |
|---|---|
| Bust probability + confidence, D1–D10, every grid cell | `/api/products`, map |
| Predicted error per variable (rain, temp, wind, pressure) | `/api/error-prediction` |
| "Why is confidence low?" in meteorological language | `/api/explanations` |
| Historical analogue search over past forecast situations | `/api/analogs` |
| Forecast Volatility Index (cycle-to-cycle revision) | volatility layer |
| Forecast Reliability Watchlist | `/api/watchlist` |
| Case study: *could we have warned?* | `/api/case-study/{date}` |
| Model performance, ablation, spatial CV | `/api/model-metrics` |

## Pipeline

```
sandbox / real ingestion        src/fbews/sandbox, src/fbews/ingestion
   -> regrid + quality checks   src/fbews/preprocessing
   -> forecast/obs verification src/fbews/verification/errors.py
   -> error targets + bust labels  src/fbews/verification/bust.py
   -> features (6 blocks)       src/fbews/features
   -> analogue index            src/fbews/analogues
   -> model ladder + calibration   src/fbews/models
   -> evaluation + ablation     src/fbews/evaluation
   -> products + explanations   src/fbews/inference, src/fbews/explainability
   -> REST API + dashboard      src/fbews/api, frontend/
```

## Headline results (sandbox, held-out 2024-06 → 2025-05)

| Model | PR-AUC | ROC-AUC | Brier skill | Calibration error |
|---|---|---|---|---|
| climatology | 0.102 | 0.525 | 0.000 | 0.008 |
| ensemble spread alone | 0.380 | 0.792 | 0.173 | 0.010 |
| forecast volatility alone | 0.351 | 0.825 | 0.159 | 0.012 |
| random forest (best) | 0.525 | 0.874 | 0.299 | 0.013 |

Bust base rate 9.7%. Full tables, ablation, spatial cross-validation and
extreme-event breakdowns: [docs/model.md](docs/model.md).

## How to switch to real data

1. `cp .env.example .env` and fill in credentials (ERA5 and TIGGE need accounts;
   ECMWF Open Data, GEFS, IBTrACS and the RMM index need none).
2. `export $(grep -v '^#' .env | xargs)` and set `FBEWS_DATA_MODE=real`.
3. Switch the bust thresholds in `configs/default.yaml` from the sandbox-scaled
   values to `real_data_absolute_thresholds` (35 mm/day, 4 K, 6 m/s, 4 hPa).
4. `make download` then `make all`.

Read [docs/verification.md](docs/verification.md) first — the IMD 0830 IST
accumulation window and GRIB accumulation conventions will silently corrupt
every error metric if mishandled.

## Retraining

`make features labels train evaluate`. Splits are chronological and configured in
`configs/default.yaml`; never shuffle weather samples across time. Each run
writes a model card to `artifacts/models/model_card.json` with the period,
feature blocks, metrics, calibration method and git commit.

## Documentation

- [data/DATASETS.md](data/DATASETS.md) — full dataset catalogue and selection
- [docs/architecture.md](docs/architecture.md)
- [docs/methodology.md](docs/methodology.md)
- [docs/model.md](docs/model.md)
- [docs/verification.md](docs/verification.md)
- [docs/api.md](docs/api.md)
- [docs/limitations.md](docs/limitations.md) — read before trusting anything

## Licence and attribution

Code: MIT. Boundaries: datta07/INDIAN-SHAPEFILES (MIT). Land mask: Natural Earth
via `global_land_mask`. Real datasets retain their providers' licences — ECMWF
open data CC-BY-4.0, TIGGE research/education only, ERA5 Copernicus licence, IMD
data policy, NOAA products public domain.
