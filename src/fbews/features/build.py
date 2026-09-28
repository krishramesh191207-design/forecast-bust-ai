"""Build the core scientific data model (PART 3).

One row of the resulting table is:

    forecast cycle T0 + lead L + grid cell
      + forecast atmospheric state
      + ensemble statistics
      + forecast volatility (cycle-to-cycle revision)
      + weather regime / event indicators
      + temporal and static context
      ---------------------------------------------------------------
      -> verifying observation
      -> forecast error (per variable)
      -> forecast bust label (added later by verification.bust)

Feature columns and target columns are kept strictly separate: `FEATURE_COLS`
is derived from the feature-block registry, and nothing derived from the
verifying observation may appear in it.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..config import Config, load_config, load_regions
from ..grid import make_grid
from ..regimes.classify import REGIMES, classify, event_proximity
from .systems import detect_lows, proximity_fields
from ..verification.errors import compute_errors, match_truth
from .engineering import (
    FEATURE_BLOCKS,
    context_features,
    ensemble_features,
    nwp_state_features,
    spatial_features,
    volatility_features,
)

META_COLS = ["cycle", "valid_date", "lead", "lat", "lon", "region", "regime", "split"]
TARGET_COLS = [
    "obs_precip", "fc_precip", "err_precip", "bias_precip", "rel_precip",
    "obs_t2m", "err_t2m", "bias_t2m",
    "err_wind", "err_u10", "err_v10",
    "obs_mslp", "err_mslp", "bias_mslp",
    "err_z500",
]


def blocks_path(cfg: Config) -> Path:
    return cfg.path("features") / "feature_blocks.json"


def save_blocks(cfg: Config) -> Path:
    """Persist the feature-block registry so training can reproduce it."""
    import json

    out = blocks_path(cfg)
    out.write_text(json.dumps(FEATURE_BLOCKS, indent=2))
    return out


def load_blocks(cfg: Config | None = None) -> dict[str, list[str]]:
    import json

    cfg = cfg or load_config()
    p = blocks_path(cfg)
    if p.exists():
        blocks = json.loads(p.read_text())
        for k, v in blocks.items():
            FEATURE_BLOCKS[k] = v
    return FEATURE_BLOCKS


def feature_columns(blocks: list[str] | None = None, cfg: Config | None = None) -> list[str]:
    # the persisted registry (written when the table was built) always wins,
    # so training and inference see exactly the columns the table contains
    registry = load_blocks(cfg)
    blocks = blocks or list(registry)
    cols: list[str] = []
    for b in blocks:
        cols.extend(registry.get(b, []))
    return cols


def assign_regions(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Region id per grid cell; first matching box wins, else 'other'."""
    regions = load_regions()
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    out = np.full(la.shape, "other", dtype=object)
    for r in regions:
        a, b, c, d = r["bbox"]
        m = (la >= a) & (la <= b) & (lo >= c) & (lo <= d) & (out == "other")
        out[m] = r["id"]
    return out


def split_of(day: dt.date, cfg: Config) -> str:
    for name in ("train", "val", "test"):
        lo, hi = cfg["splits"][name]
        if dt.date.fromisoformat(lo) <= day <= dt.date.fromisoformat(hi):
            return name
    return "unused"


class DataAssembler:
    """Holds the shared static inputs and turns cycles into feature rows."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        proc = self.cfg.path("processed")
        self.proc = proc
        self.lats, self.lons = make_grid(self.cfg.domain, self.cfg.resolution)
        self.static = xr.open_dataset(proc / "static.nc")
        self.truth = xr.open_mfdataset(sorted((proc / "truth").glob("truth_*.nc")),
                                       combine="by_coords").load()
        self.events = pd.read_csv(proc / "sandbox_events.csv") if (proc / "sandbox_events.csv").exists() \
            else pd.DataFrame(columns=["date", "kind", "lat", "lon", "depth_hpa"])
        idx = pd.read_csv(proc / "sandbox_indices.csv") if (proc / "sandbox_indices.csv").exists() \
            else pd.DataFrame(columns=["date"])
        self.indices = {r["date"]: r for r in idx.to_dict("records")}
        self.regions = assign_regions(self.lats, self.lons)
        self.cycle_files = sorted((proc / "forecasts").glob("fc_*.nc"))

    # ------------------------------------------------------------------
    def cycles(self) -> list[dt.date]:
        return [dt.date.fromisoformat(p.stem.split("_")[1][:4] + "-" +
                                      p.stem.split("_")[1][4:6] + "-" +
                                      p.stem.split("_")[1][6:8]) for p in self.cycle_files]

    def _open(self, cycle: dt.date) -> tuple[xr.Dataset, xr.Dataset | None]:
        fc = xr.open_dataset(self.proc / "forecasts" / f"fc_{cycle:%Y%m%d}.nc")
        prev_path = self.proc / "forecasts_prev" / f"fcprev_{cycle:%Y%m%d}.nc"
        prev = xr.open_dataset(prev_path) if prev_path.exists() else None
        return fc, prev

    # ------------------------------------------------------------------
    def cycle_fields(self, cycle: dt.date, lead: int) -> dict[str, np.ndarray]:
        """All predictor fields for one cycle/lead on the full grid."""
        fc, prev = self._open(cycle)
        return self._fields(fc, prev, cycle, lead)

    def _fields(self, fc, prev, cycle: dt.date, lead: int) -> dict[str, np.ndarray]:
        feats: dict[str, np.ndarray] = {}
        feats.update(nwp_state_features(fc, lead, self.lats, self.lons, self.static))
        feats.update(ensemble_features(fc, lead))
        feats.update(spatial_features(feats, self.lats, self.lons))
        feats.update(volatility_features(fc, prev, lead, prev_lead_offset=1))
        valid = cycle + dt.timedelta(days=lead)
        feats.update(context_features(cycle, lead, self.lats, self.lons, self.static,
                                      self.indices.get(valid.isoformat(), {})))
        # Systems are detected in the FORECAST pressure field, never taken
        # from the verifying event catalogue (see features/systems.py).
        centres = detect_lows(feats["nwp_mslp"], self.lats, self.lons,
                              self.static.land.values, valid.month)
        prox = proximity_fields(centres, self.lats, self.lons)
        feats["sys_dist_cyclone"] = prox["dist_tropical_cyclone"]
        feats["sys_dist_low"] = prox["dist_monsoon_low"]
        feats["sys_dist_wd"] = prox["dist_western_disturbance"]
        feats["sys_dist_circulation"] = prox["dist_cyclonic_circulation"]
        feats["sys_depth_cyclone"] = prox["depth_tropical_cyclone"]
        feats["sys_depth_low"] = prox["depth_monsoon_low"]
        feats["sys_n_detected"] = np.full(feats["nwp_mslp"].shape, len(centres), dtype="float32")
        for name in ("sys_dist_cyclone", "sys_dist_low", "sys_dist_wd", "sys_dist_circulation",
                     "sys_depth_cyclone", "sys_depth_low", "sys_n_detected"):
            if name not in FEATURE_BLOCKS["regime"]:
                FEATURE_BLOCKS["regime"].append(name)
        feats["_regime"] = classify(feats, prox, valid.month).astype("float32")
        self._last_centres = centres
        return feats

    # ------------------------------------------------------------------
    def rows_for_cycle(self, cycle: dt.date, stride: int = 1, with_targets: bool = True) -> pd.DataFrame:
        fc, prev = self._open(cycle)
        leads = [int(l) for l in fc.lead.values]
        try:
            obs = match_truth(self.truth, cycle, leads) if with_targets else None
        except KeyError:
            obs = None
        err = None
        if obs is not None:
            err = compute_errors(fc, obs, self.cfg["verification"]["variables"]
                                 ["precipitation"]["thresholds_mm_day"])

        ii = np.arange(0, len(self.lats), stride)
        jj = np.arange(0, len(self.lons), stride)
        sel = np.ix_(ii, jj)
        frames = []
        for lead in leads:
            feats = self._fields(fc, prev, cycle, lead)
            valid = cycle + dt.timedelta(days=lead)
            data: dict[str, np.ndarray] = {}
            for name, arr in feats.items():
                data[name.lstrip("_") if name.startswith("_") else name] = arr[sel].ravel()
            n = len(data["ctx_lat"])
            frame = pd.DataFrame(data)
            frame.insert(0, "cycle", cycle.isoformat())
            frame.insert(1, "valid_date", valid.isoformat())
            frame.insert(2, "lead", lead)
            frame.insert(3, "lat", np.repeat(self.lats[ii], len(jj)))
            frame.insert(4, "lon", np.tile(self.lons[jj], len(ii)))
            frame.insert(5, "region", self.regions[sel].ravel())
            frame["regime_name"] = [REGIMES[int(r)] for r in frame["regime"]]
            # verifying event type: METADATA for stratified evaluation and the
            # event explorer. Never used as a predictor.
            frame["obs_event_type"] = self._observed_event_type(valid, frame["lat"].values,
                                                               frame["lon"].values)
            frame["split"] = split_of(cycle, self.cfg)
            if err is not None and lead in set(int(x) for x in err.lead.values):
                e = err.sel(lead=lead)
                for col in TARGET_COLS:
                    if col in e:
                        frame[col] = e[col].values[sel].ravel()
                for thr in self.cfg["verification"]["variables"]["precipitation"]["thresholds_mm_day"]:
                    key = f"thr{int(thr)}_precip"
                    if key in e:
                        frame[key] = e[key].values[sel].ravel()
            assert len(frame) == n
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def _observed_event_type(self, valid: dt.date, lat: np.ndarray, lon: np.ndarray,
                             radius: float = 6.0) -> np.ndarray:
        """Nearest verified system type within `radius` degrees, else 'none'."""
        out = np.full(len(lat), "none", dtype=object)
        if len(self.events) == 0:
            return out
        day = self.events[self.events["date"] == valid.isoformat()]
        if len(day) == 0:
            return out
        best = np.full(len(lat), radius, dtype="float32")
        for _, row in day.iterrows():
            d = np.hypot(lat - row["lat"], lon - row["lon"]).astype("float32")
            closer = d < best
            best = np.where(closer, d, best)
            out[closer] = row["kind"]
        return out

    # ------------------------------------------------------------------
    def build(self, stride: int | None = None, max_cycles: int | None = None,
              verbose: bool = True) -> pd.DataFrame:
        stride = stride or int(self.cfg["grid"]["train_stride"])
        out = []
        cycles = self.cycles()[:max_cycles]
        for i, c in enumerate(cycles):
            out.append(self.rows_for_cycle(c, stride=stride))
            if verbose and (i + 1) % 40 == 0:
                print(f"  features: {i + 1}/{len(cycles)} cycles", flush=True)
        df = pd.concat(out, ignore_index=True)
        for c in df.columns:
            if df[c].dtype == "float64":
                df[c] = df[c].astype("float32")
        return df


def build_and_save(cfg: Config | None = None, stride: int | None = None,
                   max_cycles: int | None = None, verbose: bool = True) -> Path:
    cfg = cfg or load_config()
    asm = DataAssembler(cfg)
    df = asm.build(stride=stride, max_cycles=max_cycles, verbose=verbose)
    out = cfg.path("features") / "training_table.parquet"
    df.to_parquet(out, index=False)
    save_blocks(cfg)
    if verbose:
        print(f"  wrote {out} rows={len(df):,} cols={df.shape[1]}")
    return out
