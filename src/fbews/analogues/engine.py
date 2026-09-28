"""Historical analogue engine (PART 10).

Question answered: *"have we seen a forecast situation like this before, and
how badly did the model do then?"*

Method (purely numerical - no language model is involved in the similarity
computation):

1. Each forecast situation is embedded as a vector of standardised
   meteorological quantities (`ANALOGUE_EMBEDDING` in features.engineering):
   forecast precipitation, moisture, instability, pressure and height
   anomalies, wind speed, moisture convergence, shear, ensemble spread,
   forecast revision, season and location.
2. The embedding is whitened (z-scored) and projected with PCA so that
   correlated predictors do not dominate the distance.
3. A KD-tree over the **training period only** returns the k nearest
   historical situations. Neighbours within `exclude_same_cycle_window_days`
   of the query cycle are dropped so a situation cannot match itself or its
   own immediate neighbours in time.
4. The verified outcomes of those neighbours (their actual forecast error and
   bust flag) are summarised into analogue features.

FAISS can be substituted for the KD-tree without changing the interface; for
the data volume here (10^5 rows, 8 dimensions) scikit-learn's KD-tree is
faster than the index-building cost of FAISS.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

from ..features.engineering import ANALOGUE_EMBEDDING

ANALOGUE_FEATURES = ["ana_similarity", "ana_bust_rate", "ana_mean_error",
                     "ana_n_analogues", "ana_error_spread"]


@dataclass
class AnalogueEngine:
    k: int = 25
    n_components: int = 8
    exclude_days: int = 5
    max_index_rows: int = 60000
    columns: list[str] = None

    scaler: StandardScaler = None
    pca: PCA = None
    tree: KDTree = None
    ref_error: np.ndarray = None
    ref_bust: np.ndarray = None
    ref_ordinal: np.ndarray = None
    ref_meta: pd.DataFrame = None

    # ------------------------------------------------------------------
    def _embed(self, df: pd.DataFrame) -> np.ndarray:
        cols = self.columns or [c for c in ANALOGUE_EMBEDDING if c in df.columns]
        self.columns = cols
        X = df[cols].to_numpy(dtype="float64", copy=True)
        # volatility can be missing for the first cycle of a stream
        med = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(med, inds[1])
        return X

    def fit(self, train: pd.DataFrame, error_col: str = "err_precip",
            bust_col: str = "bust_overall", seed: int = 0) -> "AnalogueEngine":
        df = train
        if len(df) > self.max_index_rows:
            df = df.sample(self.max_index_rows, random_state=seed).sort_index()
        X = self._embed(df)
        self.scaler = StandardScaler().fit(X)
        self.pca = PCA(n_components=min(self.n_components, X.shape[1]), random_state=seed)
        Z = self.pca.fit_transform(self.scaler.transform(X))
        self.tree = KDTree(Z, leaf_size=48)
        self.ref_error = df[error_col].to_numpy(dtype="float32")
        self.ref_bust = df[bust_col].to_numpy(dtype="float32")
        self.ref_ordinal = np.array([dt.date.fromisoformat(c).toordinal() for c in df["cycle"]],
                                    dtype="int32")
        self.ref_meta = df[["cycle", "valid_date", "lead", "lat", "lon", "regime_name",
                            error_col, bust_col]].reset_index(drop=True)
        return self

    # ------------------------------------------------------------------
    def _query(self, df: pd.DataFrame, k_extra: int = 12):
        X = self._embed(df)
        Z = self.pca.transform(self.scaler.transform(X))
        dist, idx = self.tree.query(Z, k=min(self.k + k_extra, len(self.ref_error)))
        q_ord = np.array([dt.date.fromisoformat(c).toordinal() for c in df["cycle"]], dtype="int32")
        too_close = np.abs(self.ref_ordinal[idx] - q_ord[:, None]) <= self.exclude_days
        return dist, idx, too_close

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the analogue feature block for every row of `df`."""
        dist, idx, too_close = self._query(df)
        k = self.k
        n = len(df)
        sim = np.zeros(n, dtype="float32")
        bust = np.zeros(n, dtype="float32")
        err = np.zeros(n, dtype="float32")
        spread = np.zeros(n, dtype="float32")
        count = np.zeros(n, dtype="float32")

        for i in range(n):
            keep = ~too_close[i]
            sel = idx[i][keep][:k]
            d = dist[i][keep][:k]
            if len(sel) == 0:
                sel, d = idx[i][:k], dist[i][:k]
            sim[i] = float(np.mean(1.0 / (1.0 + d)))
            bust[i] = float(self.ref_bust[sel].mean())
            e = self.ref_error[sel]
            err[i] = float(e.mean())
            spread[i] = float(e.std())
            count[i] = len(sel)

        return pd.DataFrame({
            "ana_similarity": sim,
            "ana_bust_rate": bust,
            "ana_mean_error": err,
            "ana_error_spread": spread,
            "ana_n_analogues": count,
        }, index=df.index)

    def neighbours(self, row: pd.DataFrame, n: int = 5) -> list[dict]:
        """The most similar historical situations, for the UI."""
        dist, idx, too_close = self._query(row)
        keep = ~too_close[0]
        sel = idx[0][keep][:n]
        d = dist[0][keep][:n]
        out = []
        for j, dd in zip(sel, d):
            m = self.ref_meta.iloc[int(j)]
            out.append({
                "cycle": m["cycle"],
                "valid_date": m["valid_date"],
                "lead_day": int(m["lead"]),
                "lat": float(m["lat"]),
                "lon": float(m["lon"]),
                "regime": m["regime_name"],
                "observed_error": round(float(m.iloc[-2]), 3),
                "was_bust": int(m.iloc[-1]),
                "similarity": round(float(1.0 / (1.0 + dd)), 4),
            })
        return out

    # ------------------------------------------------------------------
    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> "AnalogueEngine":
        return joblib.load(path)
