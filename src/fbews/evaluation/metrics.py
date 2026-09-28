"""Evaluation metrics for probabilistic bust prediction (PARTS 14, 23, 35, 36).

All metrics are computed on held-out data. Poor results are reported as they
are: nothing here selects the best-looking subset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Reliability (calibration) curve plus expected calibration error."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    bins = []
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            bins.append({"bin": b, "p_mean": None, "observed": None, "count": 0})
            continue
        pm, om, c = float(p[m].mean()), float(y[m].mean()), int(m.sum())
        bins.append({"bin": b, "p_mean": round(pm, 4), "observed": round(om, 4), "count": c})
        ece += c / len(y) * abs(pm - om)
    return {"bins": bins, "expected_calibration_error": round(float(ece), 4)}


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5,
                           climatology: float | None = None) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype="float64")
    pred = (p >= threshold).astype(int)
    base = float(np.mean(y)) if climatology is None else float(climatology)
    brier = float(brier_score_loss(y, p))
    brier_ref = float(np.mean((base - y) ** 2))
    out = {
        "n": int(len(y)),
        "positive_rate": round(float(np.mean(y)), 5),
        "brier": round(brier, 5),
        "brier_skill_score_vs_climatology": round(1.0 - brier / brier_ref, 4) if brier_ref > 0 else None,
        "threshold": threshold,
    }
    if len(np.unique(y)) > 1:
        out["roc_auc"] = round(float(roc_auc_score(y, p)), 4)
        out["pr_auc"] = round(float(average_precision_score(y, p)), 4)
        out["precision"] = round(float(precision_score(y, pred, zero_division=0)), 4)
        out["recall"] = round(float(recall_score(y, pred, zero_division=0)), 4)
        out["f1"] = round(float(f1_score(y, pred, zero_division=0)), 4)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        out["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    out["calibration"] = reliability(y, p)
    return out


def curves(y: np.ndarray, p: np.ndarray, max_points: int = 120) -> dict:
    """Thinned ROC and PR curves suitable for plotting in the dashboard."""
    if len(np.unique(y)) < 2:
        return {"roc": [], "pr": []}
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)

    def thin(a, b):
        n = len(a)
        step = max(1, n // max_points)
        return [[round(float(x), 4), round(float(z), 4)] for x, z in zip(a[::step], b[::step])]

    return {"roc": thin(fpr, tpr), "pr": thin(rec, prec)}


def stratified(df: pd.DataFrame, y_col: str, p_col: str, by: str,
               min_count: int = 200, climatology: float | None = None) -> list[dict]:
    rows = []
    for key, grp in df.groupby(by, observed=True):
        if len(grp) < min_count:
            continue
        m = classification_metrics(grp[y_col].to_numpy(), grp[p_col].to_numpy(),
                                   climatology=climatology)
        rows.append({
            by: str(key), "n": m["n"], "positive_rate": m["positive_rate"],
            "brier": m["brier"], "bss": m["brier_skill_score_vs_climatology"],
            "roc_auc": m.get("roc_auc"), "pr_auc": m.get("pr_auc"),
            "recall": m.get("recall"), "precision": m.get("precision"),
            "ece": m["calibration"]["expected_calibration_error"],
        })
    return rows


def regression_metrics(y: np.ndarray, yhat: np.ndarray, reference: np.ndarray | None = None) -> dict:
    y, yhat = np.asarray(y, dtype="float64"), np.asarray(yhat, dtype="float64")
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    mae = float(np.mean(np.abs(y - yhat)))
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    out = {"n": int(len(y)), "mae": round(mae, 4), "rmse": round(rmse, 4),
           "bias": round(float(np.mean(yhat - y)), 4)}
    if len(y) > 2 and np.std(y) > 0 and np.std(yhat) > 0:
        out["pearson_r"] = round(float(np.corrcoef(y, yhat)[0, 1]), 4)
    if reference is not None:
        ref = np.asarray(reference, dtype="float64")[m]
        ref_mae = float(np.mean(np.abs(y - ref)))
        out["mae_skill_score_vs_reference"] = round(1.0 - mae / ref_mae, 4) if ref_mae > 0 else None
    return out


def error_by_lead(df: pd.DataFrame, cols=("err_precip", "err_t2m", "err_wind", "err_mslp")) -> dict:
    out = {}
    for c in cols:
        if c not in df:
            continue
        g = df.groupby("lead", observed=True)[c]
        out[c] = {
            "lead": [int(x) for x in g.mean().index],
            "mae": [round(float(v), 4) for v in g.mean().values],
            "rmse": [round(float(v), 4) for v in np.sqrt(df.groupby("lead")[c].apply(lambda s: (s ** 2).mean())).values],
            "p90": [round(float(v), 4) for v in g.quantile(0.9).values],
        }
    return out
