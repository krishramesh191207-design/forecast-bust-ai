# Architecture

```
                    ┌───────────────────────────────────────────┐
                    │ INGESTION                                 │
   real mode ──────▶│ TIGGE · ECMWF Open Data · GEFS · GFS      │
                    │ ERA5 · IMD gridded · IMERG · IBTrACS      │──┐
                    │ MJO RMM · ONI/DMI                         │  │
                    └───────────────────────────────────────────┘  │
                    ┌───────────────────────────────────────────┐  │
 sandbox mode ─────▶│ toy_model.py  (SYNTHETIC, labelled)       │──┤
                    └───────────────────────────────────────────┘  │
                                                                   ▼
        ┌────────────────────────────────────────────────────────────────┐
        │ PREPROCESSING   regrid (bilinear for state, conservative for   │
        │                 fluxes) · unit conversion · quality checks     │
        └────────────────────────────────────────────────────────────────┘
                                      ▼
        ┌────────────────────────────────────────────────────────────────┐
        │ VERIFICATION    valid_time = cycle + lead · 24 h accumulation  │
        │                 error targets: precip, temp, wind, pressure,   │
        │                 z500, threshold exceedance                     │
        │                 bust labels: hybrid percentile + absolute      │
        │                 (fitted on the TRAIN split only)               │
        └────────────────────────────────────────────────────────────────┘
                                      ▼
        ┌────────────────────────────────────────────────────────────────┐
        │ FEATURES  nwp · ens · vol · ana · regime · ctx   (83 columns)  │
        │           systems detected from FORECAST pressure, never from  │
        │           the verifying analysis                               │
        └────────────────────────────────────────────────────────────────┘
                                      ▼
        ┌────────────────────────────────────────────────────────────────┐
        │ MODELS  climatology → spread-only → volatility-only →          │
        │         logistic → random forest → gradient boosting           │
        │         + isotonic calibration (fitted on VAL only)            │
        │         + error-magnitude regressors (4 targets)               │
        │         + variable-specific bust classifiers (4 targets)       │
        └────────────────────────────────────────────────────────────────┘
                                      ▼
        ┌────────────────────────────────────────────────────────────────┐
        │ INFERENCE  full-grid products · confidence · watchlist ·       │
        │            region table · explanations · analogues             │
        └────────────────────────────────────────────────────────────────┘
                                      ▼
              FastAPI  ──────────────────────────▶  dashboard (canvas map)
```

## Storage

Gridded fields live in compressed NetCDF (`data/processed/`), tabular features
in Parquet (`data/features/`, `data/labels/`), products in JSON
(`data/products/`), models in joblib bundles with a JSON model card
(`artifacts/models/`). Large numerical fields are deliberately **not** stored in
a relational database; PostGIS is appropriate only for the metadata, region
polygons and prediction summaries, and the JSON product files map one-to-one
onto such tables if that layer is added.

## Compute profile

The whole sandbox pipeline runs on one CPU core in about 12 minutes:
data generation ~110 s, features ~99 s, labels and analogues ~27 s,
model ladder ~200 s, regressors ~46 s, ablation ~66 s. Inference for one cycle
(1,476 cells x 10 leads, all layers and explanations) takes ~6 s.
