"""Stage 1: parametric baselines - OLS on the log outcome and a
negative-binomial robustness model on the raw count, plus VIF diagnostics."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .data import Schema, prepare


def _design(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    cols = []
    for c in schema.antecedents + schema.controls:
        use = f"log_{c}" if c in schema.log_transform else c
        cols.append(use)
    X = df[cols].copy()
    for c in schema.categorical:
        use = f"log_{c}" if c in schema.log_transform else c
        if use in X.columns:
            dummies = pd.get_dummies(X[use].astype("category"), prefix=use, drop_first=True)
            X = pd.concat([X.drop(columns=[use]), dummies.astype(float)], axis=1)
    return X.astype(float)


def vif_table(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    X = sm.add_constant(_design(prepare(df, schema), schema))
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    rows = [{"variable": X.columns[i],
             "VIF": round(float(variance_inflation_factor(X.values, i)), 2)}
            for i in range(1, X.shape[1])]
    return pd.DataFrame(rows)


def run_regressions(df: pd.DataFrame, schema: Schema) -> Dict[str, pd.DataFrame]:
    """Fit OLS on log1p(outcome) and NB2 on the raw outcome; tidy the results."""
    data = prepare(df, schema)
    X = sm.add_constant(_design(data, schema))
    y_raw = pd.to_numeric(data[schema.outcome], errors="coerce")
    y_log = np.log1p(y_raw)

    ols = sm.OLS(y_log, X, missing="drop").fit(cov_type="HC1")
    nb, nb_label, nb_meta = _fit_count_robustness(y_raw, X)

    def tidy(res, model):
        return pd.DataFrame({
            "model": model,
            "term": res.params.index,
            "estimate": res.params.round(4).values,
            "std_error": res.bse.round(4).values,
            "p_value": res.pvalues.round(4).values,
            "sig": ["***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
                    for p in res.pvalues],
        })

    out = {"ols": tidy(ols, "OLS(log outcome)")}
    out["ols"].attrs["r2"] = round(float(ols.rsquared), 3)
    if nb is not None:
        out["negbin"] = tidy(nb, nb_label)
        out["negbin"].attrs.update(nb_meta)
    out["vif"] = vif_table(df, schema)
    return out

def _fit_count_robustness(y: pd.Series, X: pd.DataFrame):
    """Robustness model for the raw count outcome.

    Ladder: (1) Poisson GLM (numerically stable) supplies fitted means and
    warm-start parameters; (2) the NB2 dispersion alpha is estimated by the
    Cameron-Trivedi auxiliary regression; (3) an NB2 GLM with that alpha and
    the Poisson warm start is accepted only if IRLS converges; otherwise
    (4) the Poisson QMLE with HC1 (sandwich) standard errors is reported -
    the standard robust alternative under overdispersion. All numerical
    warnings are contained inside this fitter, and the choice is recorded in
    the returned metadata rather than silently hidden.
    """
    import warnings as _w
    try:
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            pois = sm.GLM(y, X, family=sm.families.Poisson(), missing="drop").fit()
            mu = pois.fittedvalues
            aux = ((y.loc[mu.index] - mu) ** 2 - mu) / mu
            alpha_hat = float(np.clip(sm.OLS(aux, mu).fit().params.iloc[0], 0.01, 50.0))
            nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha_hat),
                        missing="drop").fit(start_params=pois.params, maxiter=500)
        if nb.converged:
            return nb, f"NegBin(raw outcome, alpha={alpha_hat:.3g})", \
                {"alpha_hat": alpha_hat, "nb_converged": True, "fallback": None}
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            qmle = sm.GLM(y, X, family=sm.families.Poisson(), missing="drop").fit(cov_type="HC1")
        return qmle, "Poisson QMLE robust-SE (raw outcome)", \
            {"alpha_hat": alpha_hat, "nb_converged": False,
             "fallback": "NB IRLS did not converge; Poisson QMLE with HC1 SEs reported"}
    except Exception:
        # extreme magnitudes can break even the Poisson IRLS; rescale the
        # outcome (slopes under a log link are scale-invariant, only the
        # intercept shifts by -log s) and retry once, recording the scale.
        try:
            y_num = pd.to_numeric(y, errors="coerce")
            scale = 10.0 ** max(0, int(np.ceil(np.log10(max(float(y_num.max()), 1.0) / 1e6))))
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                qmle = sm.GLM(y_num / scale, X, family=sm.families.Poisson(),
                              missing="drop").fit(cov_type="HC1")
            label = "Poisson QMLE robust-SE (raw outcome)" if scale == 1 \
                else f"Poisson QMLE robust-SE (outcome/{scale:g})"
            return qmle, label, \
                {"nb_converged": False, "outcome_rescale": scale,
                 "fallback": "raw-scale count fits failed; outcome rescaled "
                             "(slopes invariant, intercept shifted by -log scale)"}
        except Exception as exc:                          # pragma: no cover
            return None, "", {"error": f"{type(exc).__name__}: {exc}"}
