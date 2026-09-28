# Methodology

## The core scientific data model

One training row is one *(forecast cycle, lead time, grid cell)*:

```
cycle T0 + lead L + cell (lat, lon)
  + forecast atmospheric state        (what the model says will happen)
  + ensemble statistics               (how much the members disagree)
  + forecast volatility               (how much the forecast changed since T0-24h)
  + historical analogue statistics    (how similar past situations verified)
  + weather regime / detected systems
  + season, location, terrain, lead time
  ─────────────────────────────────────────────────────────────
  → verifying observation
  → forecast error, per variable
  → bust label, per variable and overall
  → P(bust)
```

## Feature blocks

**nwp** — forecast state plus derived dynamics: 850 hPa vorticity and
divergence, moisture-flux convergence, MSLP and geopotential gradients, thermal
gradient, MSLP Laplacian, lapse-rate proxy, precipitable-water anomaly,
neighbourhood variance and precipitation peakiness.

**ens** — mean, standard deviation, variance, IQR, q90, coefficient of
variation, exceedance probabilities at 10/25/50 mm, temperature range, pressure
depth, wind spread, and ensemble-mean-versus-control disagreement.

**vol** — the distinctive block. Two consecutive cycles are compared *for the
same valid time*: the T−24 h cycle is read at lead+1. Revisions in rainfall,
pressure, temperature and wind, the change in ensemble spread, a pattern
displacement proxy (distance to the best-matching neighbouring cell in the
previous cycle), and a combined **Forecast Volatility Index**. A forecast that
keeps changing is a forecast that is not converged.

**ana** — analogue statistics (below).

**regime** — distance to and depth of the nearest detected tropical cyclone,
monsoon low, western disturbance or cyclonic circulation, plus the number of
systems detected. All detected in the **forecast** pressure field.

**ctx** — lead time, day of year (sin/cos), season, latitude, longitude, terrain
height and gradient, land fraction, and large-scale indices.

## Historical analogue engine

Standardise the embedding → PCA to 8 components → KD-tree over training-period
situations only → k = 25 neighbours, dropping any within ±5 days of the query
cycle. Return mean similarity, the historical bust frequency of those
neighbours, their mean error and its spread. This answers directly: *have we
seen this before, and did the model do badly?* No language model is involved in
any similarity computation.

## Model ladder

Climatology → ensemble spread alone → volatility alone → logistic regression →
random forest → gradient boosting, all evaluated identically. Isotonic
calibration is fitted on the validation split. Alongside the overall classifier
sit four variable-specific bust classifiers and four error-magnitude regressors,
which is what lets the system say *"low confidence mainly because precipitation
uncertainty is high"* rather than just *"low confidence"*.

## Explanation

Two mechanisms, both grounded in real numbers:

1. **Group attribution** — replace a feature block with its training medians,
   re-run the model, and report the drop in bust probability.
2. **Diagnostic reasons** — each candidate reason is tied to one named feature,
   a percentile of that feature in the training climatology *for that lead
   time*, and a sentence quoting the real value and units. A reason only fires
   when the feature is genuinely in the tail, so the system cannot claim
   "strong moisture convergence" when convergence is average.

## Why no deep learning

The ablation shows tabular features are not yet saturated, training data is
~207k rows, and the environment has one CPU core. A ConvLSTM would add
architectural novelty and no demonstrated accuracy, calibration or
interpretability benefit. The brief asks to prioritise those over novelty.
