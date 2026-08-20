"""Stage 2: Bayesian-optimised gradient boosting with a leakage-safe repeated
hold-out protocol, permutation importance, and an elbow-based condition screen.

xgboost + optuna are used when installed; otherwise the module degrades to
scikit-learn's HistGradientBoostingRegressor with random search, so the
pipeline never hard-fails on optional dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from .data import Schema, prepare

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:                                     # pragma: no cover
    _HAS_XGB = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except Exception:                                     # pragma: no cover
    _HAS_OPTUNA = False

DEFAULT_SEEDS = (42, 123, 456, 789, 1024)


def _matrix(df: pd.DataFrame, schema: Schema) -> Tuple[pd.DataFrame, pd.Series]:
    data = prepare(df, schema)
    feats = []
    for c in schema.antecedents + schema.controls:
        feats.append(f"log_{c}" if c in schema.log_transform else c)
    X = data[feats].astype(float)
    y = np.log1p(pd.to_numeric(data[schema.outcome], errors="coerce"))
    keep = X.notna().all(axis=1) & y.notna()
    return X[keep], y[keep]


def _tuned_model(X: pd.DataFrame, y: pd.Series, n_trials: int, seed: int):
    """Tune on the training partition only (5-fold CV RMSE)."""
    if _HAS_XGB and _HAS_OPTUNA:
        def objective(trial):
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                random_state=seed, n_jobs=2, verbosity=0,
            )
            model = XGBRegressor(**params)
            cv = KFold(5, shuffle=True, random_state=seed)
            score = cross_val_score(model, X, y, cv=cv,
                                    scoring="neg_root_mean_squared_error")
            return -score.mean()
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return XGBRegressor(**study.best_params, random_state=seed,
                            n_jobs=2, verbosity=0)
    # fallback: light random search over HistGradientBoosting
    rng = np.random.default_rng(seed)
    best, best_rmse = None, np.inf
    for _ in range(max(10, n_trials // 5)):
        params = dict(max_iter=int(rng.integers(100, 600)),
                      max_depth=int(rng.integers(3, 10)),
                      learning_rate=float(rng.uniform(0.02, 0.3)),
                      random_state=seed)
        model = HistGradientBoostingRegressor(**params)
        cv = KFold(5, shuffle=True, random_state=seed)
        rmse = -cross_val_score(model, X, y, cv=cv,
                                scoring="neg_root_mean_squared_error").mean()
        if rmse < best_rmse:
            best, best_rmse = HistGradientBoostingRegressor(**params), rmse
    return best


@dataclass
class BoostResult:
    metrics: pd.DataFrame            # model-comparison panel
    importance: pd.DataFrame         # mean permutation importance
    retained: List[str]              # antecedents above the elbow cut-off
    cutoff: float


def run_boosting(df: pd.DataFrame, schema: Schema,
                 seeds: Tuple[int, ...] = DEFAULT_SEEDS,
                 n_trials: int = 50,
                 importance_cutoff: Optional[float] = None,
                 min_retain: int = 3) -> BoostResult:
    """Five-times repeated 80/20 hold-out; per-repetition TPE tuning on the
    training fold only; metrics and permutation importance averaged across
    the held-out folds."""
    X, y = _matrix(df, schema)
    metric_rows, importances = [], []

    baselines = {
        "OLS regression": lambda s: LinearRegression(),
        "Random forest (default)": lambda s: RandomForestRegressor(random_state=s),
        "Gradient boosting (default)": lambda s: HistGradientBoostingRegressor(random_state=s),
    }
    tuned_name = "XGBoost (Bayesian-tuned)" if _HAS_XGB and _HAS_OPTUNA \
        else "HistGB (random-search-tuned)"

    for seed in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
        for name, factory in baselines.items():
            m = factory(seed).fit(Xtr, ytr)
            pred = m.predict(Xte)
            metric_rows.append({"model": name, "seed": seed,
                                "RMSE": mean_squared_error(yte, pred) ** 0.5,
                                "MAE": mean_absolute_error(yte, pred),
                                "R2": r2_score(yte, pred)})
        tuned = _tuned_model(Xtr, ytr, n_trials, seed).fit(Xtr, ytr)
        pred = tuned.predict(Xte)
        metric_rows.append({"model": tuned_name, "seed": seed,
                            "RMSE": mean_squared_error(yte, pred) ** 0.5,
                            "MAE": mean_absolute_error(yte, pred),
                            "R2": r2_score(yte, pred)})
        pi = permutation_importance(tuned, Xte, yte, n_repeats=20,
                                    random_state=seed,
                                    scoring="neg_root_mean_squared_error")
        importances.append(pd.Series(pi.importances_mean, index=X.columns))

    metrics = (pd.DataFrame(metric_rows)
               .groupby("model")[["RMSE", "MAE", "R2"]].mean().round(3)
               .sort_values("RMSE", ascending=False).reset_index())
    imp = (pd.concat(importances, axis=1).mean(axis=1)
           .sort_values(ascending=False).rename("permutation_importance"))
    imp = imp.round(4).reset_index().rename(columns={"index": "feature"})

    # map log_x back to x for readability
    imp["feature"] = imp["feature"].str.replace("^log_", "", regex=True)
    ctrl = set(schema.controls)
    imp["role"] = ["control" if f in ctrl else "antecedent" for f in imp["feature"]]

    ant = imp[imp["role"] == "antecedent"].reset_index(drop=True)
    if importance_cutoff is None:
        vals = ant["permutation_importance"].to_numpy()
        gaps = vals[:-1] - vals[1:]
        if len(gaps):
            start = min(max(min_retain - 1, 0), len(gaps) - 1)
            elbow = start + int(np.argmax(gaps[start:]))
            importance_cutoff = float((vals[elbow] + vals[elbow + 1]) / 2)
        else:
            importance_cutoff = 0.0
    retained = ant.loc[ant["permutation_importance"] > importance_cutoff, "feature"].tolist()
    return BoostResult(metrics, imp, retained, importance_cutoff)
