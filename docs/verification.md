# Verification and scientific conventions

## The matching rule

```
valid_time = forecast_initialisation_time + forecast_lead
```

All internal times are UTC. Latitude is stored ascending, longitude in 0–360.

## Precipitation — the part that goes wrong quietly

Distinguish clearly:

- **Rate** (kg m⁻² s⁻¹ or mm h⁻¹): an instantaneous flux.
- **Accumulation** (mm): an integral over a stated window.

GRIB precipitation from ECMWF and NCEP is accumulated **from the start of the
forecast**, so a 24-hour total for Day 5 is `tp(step=120) − tp(step=96)`, and
ECMWF `tp` is in **metres** — multiply by 1000. Getting either wrong produces
error fields that look plausible and are entirely wrong.

### Observation windows

| Product | Window | UTC equivalent |
|---|---|---|
| Sandbox | 00–00 UTC | 00–00 |
| IMD gridded rainfall | 0830–0830 IST | 03–03 UTC |
| GPM IMERG daily | 00–00 UTC | 00–00 |
| ERA5 total precipitation | hourly, accumulate yourself | any |

To verify against IMD you must accumulate the forecast over 03–03 UTC, not
00–00. Set `obs_window_offset_hours: 3`. The three-hour mismatch is small in the
mean and large in individual heavy-rain events — exactly the cases this system
is built for.

## Units after preprocessing

precipitation mm/day · temperature K · wind m/s · pressure hPa · geopotential
**height in metres** (ERA5 gives geopotential in m² s⁻², divide by 9.80665).

## Regridding

State variables bilinear; fluxes and accumulations conservative (area-weighted
with cos φ). Missing data stays NaN and is never filled with zero — a target
cell with less than 50% valid source area becomes NaN.

## Error targets

Per variable: absolute error, bias, RMSE (aggregated), relative error for
precipitation, vector error for wind, and threshold-exceedance agreement at
10 / 25 / 50 / 100 mm/day (configurable; these are conventional Indian heavy-rain
categories, not universal optima).

## Bust definition

Hybrid by default: the error must exceed **both** the 90th percentile of the
training-period error climatology for that lead time **and** an absolute floor.
The percentile stops trivial errors in dry regimes counting as busts; the
absolute floor stops tiny errors in quiet periods counting as busts. Thresholds
are stored in `artifacts/models/bust_definition.json` so any label can be audited.

## Leakage rules enforced in code

1. Percentile thresholds fitted on the training split only.
2. The analogue index contains training rows only, with a ±5-day exclusion window.
3. Calibration fitted on validation only.
4. Splits are chronological, never random.
5. Nothing derived from the verifying analysis may enter `FEATURE_BLOCKS`. This
   rule was violated once (event proximity taken from the truth catalogue) and
   caught by permutation importance; see docs/model.md.
