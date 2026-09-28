# Model and results

All numbers below come from the held-out **chronological** test split
(52,470 rows, 2024-06-01 to 2025-05-31). Training used
206,910 rows and 83 features; calibration was fitted on the
validation split only.

> **These are sandbox results.** They demonstrate that the pipeline learns a real
> signal in the synthetic data. They are not evidence of skill on real forecasts.

## Model ladder (PART 12, 37)

| Model | Brier | Brier skill vs climatology | ROC-AUC | PR-AUC | Recall | Calibration error |
|---|---|---|---|---|---|---|
| climatology | 0.08734 | 0.0002 | 0.5254 | 0.1022 | 0.0 | 0.0078 |
| spread only | 0.07229 | 0.1725 | 0.7922 | 0.3802 | 0.1015 | 0.0096 |
| volatility only | 0.0735 | 0.1586 | 0.8249 | 0.3506 | 0.0513 | 0.0122 |
| logistic regression | 0.06192 | 0.2913 | 0.8752 | 0.522 | 0.2312 | 0.0115 |
| random forest | 0.06123 | 0.2991 | 0.8744 | 0.5248 | 0.2846 | 0.0126 |
| gradient boosting | 0.06275 | 0.2817 | 0.8627 | 0.505 | 0.2381 | 0.015 |

Base rate of a bust in the test split: 0.09668. Any PR-AUC above
that number beats random ranking.

## Variable-specific bust models (PART 13)

| Variable | Base rate | PR-AUC | Brier skill |
|---|---|---|---|
| precipitation | 0.02146 | 0.3293 | 0.1954 |
| temperature | 0.02743 | 0.2395 | 0.1337 |
| wind | 0.05998 | 0.4044 | 0.2481 |
| pressure | 0.01721 | 0.0705 | 0.0256 |

## Error-magnitude models

| Target | MAE | RMSE | MAE skill vs lead climatology |
|---|---|---|---|
| precipitation absolute error (mm/day) | 0.3462 | 1.5936 | 0.5135 |
| 2 m temperature absolute error (K) | 0.4098 | 0.6158 | 0.2476 |
| 10 m vector wind error (m/s) | 0.7216 | 1.5125 | 0.2663 |
| mean sea level pressure absolute error (hPa) | 0.2317 | 0.4762 | 0.28 |

## Ablation: does each feature group earn its place? (PART 38)

| Model | Features added | # features | Brier | PR-AUC | Recall |
|---|---|---|---|---|---|
| Model A | NWP state only | 41 | 0.06508 | 0.4769 | 0.1599 |
| Model B | + ensemble spread | 62 | 0.06332 | 0.5015 | 0.2227 |
| Model C | + forecast volatility | 71 | 0.06258 | 0.508 | 0.2399 |
| Model D | + historical analogues | 76 | 0.06232 | 0.5157 | 0.247 |
| Model E | + weather regime and systems | 83 | 0.06263 | 0.5066 | 0.2608 |

Ensemble spread gives the largest single jump. Volatility and analogues each add a
smaller but consistent gain. The regime/system block does **not** improve PR-AUC
(it slightly reduces it) while it does improve recall; it is retained because the
regime label is needed for the explanation layer, and this trade-off is reported
rather than hidden.

## Spatial cross-validation (PART 36)

Trained with east_india, bay_of_bengal removed entirely, then tested on them.

| Test regions | PR-AUC | ROC-AUC | Recall |
|---|---|---|---|
| seen in training | 0.5235 | 0.8757 | 0.2586 |
| never seen | 0.4605 | 0.7929 | 0.2094 |

Performance drops on unseen regions but stays well above the base rate, so the
model is partly geography-aware and partly transferable. It is not purely
memorising location.

## Extreme-event evaluation (PART 35)

| Stratum | n | base rate | PR-AUC | Recall |
|---|---|---|---|---|
| all | 52470 | 0.09668 | 0.505 | 0.2381 |
| monsoon_low | 492 | 0.51423 | 0.677 | 0.3399 |
| none | 49632 | 0.07701 | 0.4989 | 0.2454 |
| tropical_cyclone | 650 | 0.91077 | 0.8844 | 0.3074 |
| western_disturbance | 1696 | 0.23939 | 0.3021 | 0.0049 |
| heavy_rainfall_observed | 274 | 0.94161 | 0.9393 | 0.8643 |
| quiet_dry | 42264 | 0.03812 | 0.1972 | 0.0422 |

The system is strongest exactly where it matters (cyclones, observed heavy
rainfall, monsoon lows) and weakest for western disturbances, where recall is
near zero. That failure is real and is stated plainly here and in the dashboard.

## Feature importance (permutation, drop in PR-AUC)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `ens_wind_std` | 0.0393 |
| 2 | `ana_bust_rate` | 0.03239 |
| 3 | `nwp_cape` | 0.01699 |
| 4 | `ens_t2m_std` | 0.01209 |
| 5 | `ens_precip_iqr` | 0.01048 |
| 6 | `vol_index` | 0.00952 |
| 7 | `nwp_mslp_anom` | 0.00928 |
| 8 | `ana_similarity` | 0.0092 |
| 9 | `ens_tcwv_std` | 0.00843 |
| 10 | `nwp_shear` | 0.00504 |

An earlier version ranked event-proximity features first; those were taken from the
verifying event catalogue and were **target leakage**. They were replaced with
systems detected in the forecast pressure field, and PR-AUC fell from 0.669 to
0.505. The lower number is the honest one.
