"""Training pipeline (PARTS 12, 13, 37, 39).

Trains the full model ladder on the chronological training split, calibrates
on validation, evaluates on test, and records everything in a model card.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..evaluation.metrics import (
    classification_metrics,
    curves,
    error_by_lead,
    regression_metrics,
    stratified,
)
from ..features.build import feature_columns
from ..verification.bust import BustDefinition
from .ladder import (
    CalibratedModel,
    ClimatologyModel,
    ModelCard,
    SingleFeatureLogit,
    git_commit,
    make_gradient_boosting,
    make_logistic,
    make_random_forest,
    make_regressor,
    save_bundle,
    utcnow,
)

REG_TARGETS = {
    "err_precip": "precipitation absolute error (mm/day)",
    "err_t2m": "2 m temperature absolute error (K)",
    "err_wind": "10 m vector wind error (m/s)",
    "err_mslp": "mean sea level pressure absolute error (hPa)",
}


def load_table(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(cfg.path("labels") / "labelled_table.parquet")


def splits(df: pd.DataFrame, features: list[str], target: str = "bust_overall"):
    out = {}
    for name in ("train", "val", "test"):
        part = df[df["split"] == name]
        out[name] = (part[features], part[target].to_numpy().astype(int), part)
    return out


def train_ladder(cfg: Config | None = None, verbose: bool = True) -> dict:
    cfg = cfg or load_config()
    df = load_table(cfg)
    feats = [c for c in feature_columns(cfg=cfg) if c in df.columns]
    data = splits(df, feats)
    (Xtr, ytr, tr), (Xva, yva, va), (Xte, yte, te) = data["train"], data["val"], data["test"]
    if verbose:
        print(f"  train={len(Xtr):,} val={len(Xva):,} test={len(Xte):,} features={len(feats)}")

    clim_rate = float(ytr.mean())
    models: dict[str, CalibratedModel] = {}
    calibration = cfg["models"]["calibration"]

    rungs = [
        ("climatology", ClimatologyModel(), "none"),
        ("spread_only", SingleFeatureLogit("ens_precip_std", "spread_only"), calibration),
        ("volatility_only", SingleFeatureLogit("vol_index", "volatility_only"), calibration),
        ("logistic_regression", make_logistic(), calibration),
        ("random_forest", make_random_forest(cfg["models"]["random_forest"]), calibration),
        ("gradient_boosting", make_gradient_boosting(cfg["models"]["gradient_boosting"]), calibration),
    ]

    results: dict[str, dict] = {}
    for name, model, cal in rungs:
        Xf, yf = Xtr, ytr
        if name == "random_forest":
            cap = int(cfg["models"]["max_train_rows_rf"])
            if len(Xtr) > cap:
                idx = np.random.RandomState(0).choice(len(Xtr), cap, replace=False)
                Xf, yf = Xtr.iloc[idx], ytr[idx]
        if name in ("logistic_regression", "random_forest", "gradient_boosting",
                    "spread_only", "volatility_only"):
            Xf = Xf.fillna(Xf.median(numeric_only=True)) if name != "gradient_boosting" else Xf
        model.fit(Xf, yf)
        wrapped = CalibratedModel(model, method=cal, name=name)
        Xv = Xva if name == "gradient_boosting" else Xva.fillna(Xtr.median(numeric_only=True))
        wrapped.fit_calibration(Xv, yva)
        models[name] = wrapped

        Xt = Xte if name == "gradient_boosting" else Xte.fillna(Xtr.median(numeric_only=True))
        p_test = wrapped.predict_proba(Xt)
        p_raw = wrapped.predict_raw(Xt)
        m = classification_metrics(yte, p_test, climatology=clim_rate)
        m["uncalibrated"] = classification_metrics(yte, p_raw, climatology=clim_rate)
        m["curves"] = curves(yte, p_test)
        results[name] = m
        if verbose:
            print(f"  {name:20s} BSS={m['brier_skill_score_vs_climatology']} "
                  f"ROC-AUC={m.get('roc_auc')} PR-AUC={m.get('pr_auc')} "
                  f"ECE={m['calibration']['expected_calibration_error']}", flush=True)

    # ---- best rung, stratified diagnostics ---------------------------
    best = "gradient_boosting"
    te = te.copy()
    te["p_bust"] = models[best].predict_proba(Xte)
    diagnostics = {
        "by_lead": stratified(te, "bust_overall", "p_bust", "lead", climatology=clim_rate),
        "by_region": stratified(te, "bust_overall", "p_bust", "region", climatology=clim_rate),
        "by_regime": stratified(te, "bust_overall", "p_bust", "regime_name", climatology=clim_rate),
        "by_season": stratified(te.assign(season=te["ctx_season"].astype(int).map(
            {0: "monsoon (JJAS)", 1: "post-monsoon (ON)", 2: "winter (DJF)", 3: "pre-monsoon (MAM)"})),
            "bust_overall", "p_bust", "season", climatology=clim_rate) if "ctx_season" in te else [],
        "error_by_lead": error_by_lead(te),
    }

    # ---- variable-specific bust classifiers (PART 13) ------------------
    per_variable = {}
    for target in ("bust_precipitation", "bust_temperature", "bust_wind", "bust_pressure"):
        y_tr = tr[target].to_numpy().astype(int)
        y_va = va[target].to_numpy().astype(int)
        y_te = te[target].to_numpy().astype(int)
        mdl = make_gradient_boosting({**cfg["models"]["gradient_boosting"], "max_iter": 150})
        mdl.fit(Xtr, y_tr)
        wrapped = CalibratedModel(mdl, method=calibration, name=target)
        wrapped.fit_calibration(Xva, y_va)
        models[target] = wrapped
        per_variable[target] = classification_metrics(y_te, wrapped.predict_proba(Xte),
                                                      climatology=float(y_tr.mean()))
        if verbose:
            print(f"  {target:22s} PR-AUC={per_variable[target].get('pr_auc')} "
                  f"BSS={per_variable[target]['brier_skill_score_vs_climatology']}", flush=True)

    bundle = {
        "features": feats,
        "models": models,
        "train_medians": Xtr.median(numeric_only=True).to_dict(),
        "climatology_rate": clim_rate,
    }
    save_bundle(cfg.path("models") / "classifiers.joblib", **bundle)

    out = {"ladder": results, "diagnostics": diagnostics, "per_variable": per_variable,
           "climatology_rate": clim_rate, "n_features": len(feats),
           "n_rows": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)}}
    (cfg.path("models") / "metrics_classifiers.json").write_text(json.dumps(out, indent=2))
    return out


def train_regressors(cfg: Config | None = None, verbose: bool = True) -> dict:
    """Error-magnitude models: one per verification variable (PART 13)."""
    cfg = cfg or load_config()
    df = load_table(cfg)
    feats = [c for c in feature_columns(cfg=cfg) if c in df.columns]
    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"]
    Xtr, Xte = tr[feats], te[feats]

    models, metrics = {}, {}
    for target in REG_TARGETS:
        ytr, yte = tr[target].to_numpy(), te[target].to_numpy()
        mdl = make_regressor(max_iter=180)
        mdl.fit(Xtr, ytr)
        pred = mdl.predict(Xte)
        # reference: predict the training mean error for that lead time
        ref_map = tr.groupby("lead")[target].mean()
        ref = te["lead"].map(ref_map).to_numpy()
        metrics[target] = regression_metrics(yte, pred, reference=ref)
        metrics[target]["description"] = REG_TARGETS[target]
        models[target] = mdl
        if verbose:
            print(f"  {target:12s} MAE={metrics[target]['mae']} "
                  f"skill_vs_lead_climatology={metrics[target]['mae_skill_score_vs_reference']}",
                  flush=True)

    save_bundle(cfg.path("models") / "regressors.joblib", features=feats, models=models)
    (cfg.path("models") / "metrics_regressors.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def feature_importance(cfg: Config | None = None, n_repeats: int = 3, sample: int = 25000,
                       verbose: bool = True) -> dict:
    """Permutation importance of the production classifier (PART 39)."""
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import average_precision_score

    cfg = cfg or load_config()
    bundle = __import__("joblib").load(cfg.path("models") / "classifiers.joblib")
    feats, models = bundle["features"], bundle["models"]
    df = load_table(cfg)
    te = df[df["split"] == "test"]
    if len(te) > sample:
        te = te.sample(sample, random_state=0)
    X, y = te[feats], te["bust_overall"].to_numpy().astype(int)
    model = models["gradient_boosting"].model

    def scorer(est, Xx, yy):
        return average_precision_score(yy, est.predict_proba(Xx)[:, 1])

    res = permutation_importance(model, X, y, scoring=scorer, n_repeats=n_repeats,
                                 random_state=0, n_jobs=1)
    order = np.argsort(res.importances_mean)[::-1]
    out = [{"feature": feats[i], "importance": round(float(res.importances_mean[i]), 5),
            "std": round(float(res.importances_std[i]), 5)} for i in order]
    (cfg.path("models") / "feature_importance.json").write_text(json.dumps(out, indent=2))
    if verbose:
        for row in out[:12]:
            print(f"  {row['feature']:28s} {row['importance']:.5f}")
    return out


def write_model_card(cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    mdir = cfg.path("models")
    clf = json.loads((mdir / "metrics_classifiers.json").read_text())
    reg = json.loads((mdir / "metrics_regressors.json").read_text()) if (mdir / "metrics_regressors.json").exists() else {}
    bust = BustDefinition.load(mdir / "bust_definition.json")
    manifest_path = cfg.path("processed") / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    from ..features.build import load_blocks

    card = ModelCard(
        name="fbews-bust-classifier",
        version=cfg["project"]["version"],
        trained_at=utcnow(),
        data_mode=cfg.data_mode,
        training_period=cfg["splits"]["train"],
        validation_period=cfg["splits"]["val"],
        test_period=cfg["splits"]["test"],
        features=clf.get("n_features"),
        feature_blocks=list(load_blocks(cfg)),
        n_train_rows=clf["n_rows"]["train"],
        targets=["bust_overall"] + list(clf["per_variable"]),
        calibration=cfg["models"]["calibration"],
        bust_definition=bust.to_dict()["method"] and {
            "method": bust.method, "percentile": bust.percentile,
            "absolute": bust.absolute, "stratify_by": bust.stratify_by,
        },
        metrics={
            "test": {k: {kk: v[kk] for kk in ("brier", "brier_skill_score_vs_climatology",
                                              "roc_auc", "pr_auc", "recall", "precision")
                         if kk in v}
                     for k, v in clf["ladder"].items()},
            "regression_test": {k: {"mae": v["mae"], "skill": v.get("mae_skill_score_vs_reference")}
                                for k, v in reg.items()},
        },
        git_commit=git_commit(),
        datasets=(["SANDBOX_SYNTHETIC toy atmosphere"] if cfg.is_sandbox else
                  ["TIGGE", "ERA5", "IMD gridded", "GPM IMERG", "IBTrACS"]),
        notes=manifest.get("warning", ""),
    )
    return card.save(mdir / "model_card.json")
