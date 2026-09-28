"""The model ladder (PARTS 12, 13, 14, 46).

Baselines first, then calibrated gradient boosting. Every rung exposes the
same interface so the evaluation code can treat them identically:

    fit(X, y) -> self          predict_proba(X) -> P(bust)

Rungs
-----
climatology      training bust frequency for the stratum (lead day)
spread_only      logistic regression on the ensemble precipitation spread
volatility_only  logistic regression on the forecast volatility index
logistic         regularised logistic regression on all features
random_forest    random forest
gradient_boosting histogram gradient boosting (the production rung)

Calibration is fitted on the **validation split only** (isotonic by default,
Platt available), never on training or test data.

A multi-output regression bank predicts the expected magnitude of the error in
each variable, which is what lets the system say *"low confidence mainly
because precipitation uncertainty is high"* rather than just "low confidence".
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
class ClimatologyModel:
    """P(bust) = training frequency, stratified by lead day."""

    name = "climatology"

    def __init__(self, stratify: str = "ctx_lead"):
        self.stratify = stratify
        self.table: dict[float, float] = {}
        self.base = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.base = float(np.mean(y))
        if self.stratify in X:
            s = pd.Series(y, index=X.index)
            self.table = s.groupby(X[self.stratify]).mean().to_dict()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.stratify in X and self.table:
            p = X[self.stratify].map(self.table).fillna(self.base).to_numpy(dtype="float64")
        else:
            p = np.full(len(X), self.base)
        return np.column_stack([1 - p, p])


class SingleFeatureLogit:
    """Logistic regression on one predictor (spread-only / volatility-only)."""

    def __init__(self, feature: str, name: str):
        self.feature, self.name = feature, name
        self.pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ])

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.pipe.fit(X[[self.feature]], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict_proba(X[[self.feature]])


def make_logistic() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=600, C=0.5, n_jobs=None)),
    ])


def make_random_forest(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(params.get("n_estimators", 120)),
        max_depth=params.get("max_depth"),
        min_samples_leaf=int(params.get("min_samples_leaf", 40)),
        n_jobs=int(params.get("n_jobs", -1)),
        random_state=0,
        class_weight=None,
    )


def make_gradient_boosting(params: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=int(params.get("max_iter", 300)),
        learning_rate=float(params.get("learning_rate", 0.07)),
        max_leaf_nodes=int(params.get("max_leaf_nodes", 31)),
        min_samples_leaf=int(params.get("min_samples_leaf", 40)),
        l2_regularization=float(params.get("l2_regularization", 1.0)),
        early_stopping=bool(params.get("early_stopping", True)),
        validation_fraction=0.1,
        random_state=0,
    )


def make_regressor(max_iter: int = 200) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=max_iter, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, random_state=0, loss="absolute_error",
    )


# ---------------------------------------------------------------------------
class CalibratedModel:
    """Wraps a fitted classifier with a probability calibrator."""

    def __init__(self, model, method: str = "isotonic", name: str = ""):
        self.model, self.method = model, method
        self.name = name or getattr(model, "name", type(model).__name__)
        self.calibrator = None

    def fit_calibration(self, X_val: pd.DataFrame, y_val: np.ndarray) -> "CalibratedModel":
        raw = self.model.predict_proba(X_val)[:, 1]
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(raw, y_val)
        elif self.method == "platt":
            lr = LogisticRegression(max_iter=1000)
            lr.fit(raw.reshape(-1, 1), y_val)
            self.calibrator = lr
        else:
            self.calibrator = None
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(X)[:, 1]
        if self.calibrator is None:
            return raw
        if self.method == "isotonic":
            return np.clip(self.calibrator.predict(raw), 0.0, 1.0)
        return self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
@dataclass
class ModelCard:
    """Model registry entry (PART 46)."""

    name: str
    version: str
    trained_at: str
    data_mode: str
    training_period: list[str]
    validation_period: list[str]
    test_period: list[str]
    features: list[str]
    feature_blocks: list[str]
    n_train_rows: int
    targets: list[str]
    calibration: str
    bust_definition: dict
    metrics: dict = field(default_factory=dict)
    git_commit: str | None = None
    datasets: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def save_bundle(path: Path, **objects) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(objects, path, compress=3)
    return path


def load_bundle(path: Path) -> dict:
    return joblib.load(path)


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
