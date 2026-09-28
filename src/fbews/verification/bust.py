"""Forecast-bust definition and labelling (PART 5).

A "bust" is not a universal quantity - it is a decision threshold on forecast
error. This module keeps that decision explicit and configurable:

    method = "percentile"  error > climatological percentile (per stratum)
    method = "absolute"    error > a fixed physical threshold
    method = "hybrid"      both conditions must hold  (default)

Percentile thresholds are estimated **only on the training split** and are
stratified by the keys in `bust.stratify_by` (lead day by default; region and
season are also supported). The resulting threshold table is stored with the
model so that operational labelling is reproducible and auditable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ERROR_COLUMN = {
    "precipitation": "err_precip",
    "temperature": "err_t2m",
    "wind": "err_wind",
    "pressure": "err_mslp",
}
ABS_KEY = {
    "precipitation": "precipitation_mm_day",
    "temperature": "temperature_k",
    "wind": "wind_ms",
    "pressure": "pressure_hpa",
}


@dataclass
class BustDefinition:
    method: str
    percentile: float
    stratify_by: list[str]
    absolute: dict[str, float]
    variables: list[str]
    climatology_period: str
    thresholds: dict = field(default_factory=dict)   # var -> {stratum key: value}

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "BustDefinition":
        """Estimate percentile thresholds from the training rows only."""
        self.thresholds = {}
        for var in self.variables:
            col = ERROR_COLUMN[var]
            if col not in df:
                continue
            table: dict[str, float] = {}
            if self.stratify_by:
                for key, grp in df.groupby(self.stratify_by, observed=True):
                    k = "|".join(str(x) for x in (key if isinstance(key, tuple) else (key,)))
                    table[k] = float(np.nanpercentile(grp[col].values, self.percentile))
            table["__global__"] = float(np.nanpercentile(df[col].values, self.percentile))
            self.thresholds[var] = table
        return self

    def _stratum(self, df: pd.DataFrame) -> pd.Series:
        if not self.stratify_by:
            return pd.Series(["__global__"] * len(df), index=df.index)
        parts = [df[k].astype(str) for k in self.stratify_by]
        out = parts[0]
        for p in parts[1:]:
            out = out + "|" + p
        return out

    def threshold_series(self, df: pd.DataFrame, var: str) -> np.ndarray:
        table = self.thresholds.get(var, {})
        keys = self._stratum(df)
        default = table.get("__global__", np.nan)
        return keys.map(lambda k: table.get(k, default)).to_numpy(dtype="float64")

    # ------------------------------------------------------------------
    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add bust_<variable> columns plus bust_overall and the thresholds used."""
        out = df.copy()
        any_bust = np.zeros(len(df), dtype=bool)
        for var in self.variables:
            col = ERROR_COLUMN[var]
            if col not in out:
                continue
            err = out[col].to_numpy(dtype="float64")
            pct = self.threshold_series(out, var)
            absolute = float(self.absolute[ABS_KEY[var]])
            if self.method == "percentile":
                flag = err > pct
            elif self.method == "absolute":
                flag = err > absolute
            else:  # hybrid
                flag = (err > pct) & (err > absolute)
            out[f"bust_{var}"] = flag.astype("int8")
            out[f"bustthr_{var}"] = pct
            any_bust |= flag
        out["bust_overall"] = any_bust.astype("int8")
        return out

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "percentile": self.percentile,
            "stratify_by": self.stratify_by,
            "absolute": self.absolute,
            "variables": self.variables,
            "climatology_period": self.climatology_period,
            "thresholds": self.thresholds,
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def from_config(cls, cfg) -> "BustDefinition":
        b = cfg["bust"]
        return cls(
            method=b["method"],
            percentile=float(b["percentile"]),
            stratify_by=list(b["stratify_by"]),
            absolute=b["absolute_thresholds"],
            variables=list(b["overall_variables"]),
            climatology_period=b["climatology_period"],
        )

    @classmethod
    def load(cls, path: Path) -> "BustDefinition":
        d = json.loads(Path(path).read_text())
        obj = cls(
            method=d["method"], percentile=d["percentile"], stratify_by=d["stratify_by"],
            absolute=d["absolute"], variables=d["variables"],
            climatology_period=d["climatology_period"],
        )
        obj.thresholds = d["thresholds"]
        return obj
