# API

Interactive OpenAPI docs at `/docs`, schema at `/openapi.json`.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | liveness, model load state, cycle count |
| `GET /api/meta` | data mode, domain, bands, model card, manifest |
| `GET /api/cycles` | available forecast cycles |
| `GET /api/forecast/current` | latest cycle: meta, KPIs, top watchlist rows |
| `GET /api/products?cycle=` | everything for one cycle (all layers, D1–D10) |
| `GET /api/confidence?cycle=&lead=` | confidence field |
| `GET /api/bust-probability?cycle=&lead=` | bust probability field |
| `GET /api/error-prediction?cycle=&lead=&variable=` | predicted error field |
| `GET /api/layers?cycle=&lead=` | all layers for one lead |
| `GET /api/regions?cycle=&lead=` | region × lead table |
| `GET /api/watchlist?cycle=` | Forecast Reliability Watchlist |
| `GET /api/grid/{lat}/{lon}?cycle=&lead=` | full point payload |
| `GET /api/explanations?lat=&lon=&lead=` | reasons + attribution + uncertainty |
| `GET /api/analogs?lat=&lon=&lead=` | nearest historical situations |
| `POST /api/inference` | point inference from a JSON body |
| `GET /api/model-metrics` | ladder, diagnostics, ablation, spatial CV, importance |
| `GET /api/historical-events` | event catalogue |
| `GET /api/case-study/{date}?lat=&lon=` | forecast evolution toward one valid date |
| `GET /api/datasets?probe=` | dataset catalogue, optionally probed live |
| `GET /api/data-quality` | data-quality log |
| `GET /api/boundaries` | simplified state boundaries GeoJSON |
| `GET /api/geo-labels` | geographic metadata for map labels (capitals, cities, countries) |
| `GET /api/export/csv?cycle=&lead=` | CSV download of all layers |

Example point response (abridged):

```json
{
  "valid_time": "2024-07-25T00:00:00Z",
  "forecast_cycle": "2024-07-20T00:00:00Z",
  "lead_day": 5,
  "location": {"lat": 19.0, "lon": 85.0, "region": "east_india"},
  "confidence": 42,
  "confidence_band": "low",
  "bust_probability": 0.58,
  "predicted_error": {"precipitation": 6.1, "temperature": 1.2, "wind": 2.4, "pressure": 0.9},
  "dominant_variable": "precipitation",
  "regime": "monsoon_depression",
  "explanations": ["Ensemble members disagree strongly on accumulated rainfall here ..."],
  "analogue": {"similarity": 0.71, "n_analogues": 25, "historical_bust_rate": 0.44},
  "data_banner": "DEMO / SAMPLE DATA - SYNTHETIC SANDBOX"
}
```

Errors use standard HTTP codes: 400 outside the domain, 404 unknown cycle or
case-study date, 422 invalid lead, 429 rate limited, 503 artefacts not built.
