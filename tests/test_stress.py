"""Round-3 boundary and property stress suite."""
import numpy as np
import pandas as pd
import pytest

from examples.make_dataset import make
from mixtriad import Schema, fsqca
from mixtriad.regression import run_regressions


def _schema(**kw):
    base = dict(outcome="engagement",
                antecedents=["creator_reach", "account_topic_focus",
                             "emotional_intensity"],
                controls=["publication_age"],
                log_transform=["creator_reach"])
    base.update(kw)
    return Schema(**base)


def test_missing_column_raises_cleanly():
    df = make(n=50, seed=3).drop(columns=["emotional_intensity"])
    with pytest.raises(KeyError):
        run_regressions(df, _schema())


def test_nan_rows_are_dropped_not_fatal():
    df = make(n=120, seed=4)
    df.loc[:10, "creator_reach"] = np.nan
    out = run_regressions(df, _schema())
    assert "ols" in out and out["ols"]["estimate"].notna().all()


def test_constant_condition_calibration_survives():
    x = pd.Series([5.0] * 40)
    anc = fsqca.percentile_anchors(x)
    m = fsqca.direct_calibrate(x, *anc)
    assert m.between(0, 1).all() and m.notna().all()


def test_single_condition_fsqca_runs():
    rng = np.random.default_rng(11)
    a = pd.Series(rng.uniform(0, 1, 200))
    y = pd.Series(np.clip(a + rng.normal(0, 0.1, 200), 0, 1))
    memb = pd.DataFrame({"A": a})
    tt = fsqca.build_truth_table(memb, y, 0.8, 0.6, 1)
    sols = fsqca.minimize(tt, memb, y)
    assert sols["conservative"].recipes, "single-condition table must minimise"


def test_extreme_outcome_magnitudes_no_warnings():
    import warnings
    df = make(n=200, seed=5)
    df["engagement"] = (df["engagement"].astype(float) * 1e4).astype(np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = run_regressions(df, _schema())
    assert "negbin" in out
    assert out["negbin"].attrs.get("nb_converged") in (True, False)


def test_calibration_is_monotone():
    rng = np.random.default_rng(6)
    x = pd.Series(np.sort(rng.normal(0, 5, 100)))
    m = fsqca.direct_calibrate(x, float(x.quantile(.1)), float(x.median()),
                               float(x.quantile(.9)))
    assert (np.diff(m.to_numpy()) >= -1e-12).all()


def test_measures_bounded_on_random_data():
    rng = np.random.default_rng(8)
    for _ in range(25):
        c = rng.uniform(0, 1, 60)
        y = rng.uniform(0, 1, 60)
        for f in (fsqca.consistency, fsqca.coverage):
            v = f(c, y)
            assert 0.0 <= v <= 1.0 + 1e-9


def test_core_literals_come_from_parsimonious():
    rng = np.random.default_rng(9)
    a, b, c = (rng.uniform(0, 1, 250) for _ in range(3))
    y = np.minimum(a, b)
    memb = pd.DataFrame({"A": a, "B": b, "C": c})
    tt = fsqca.build_truth_table(memb, pd.Series(y), 0.8, 0.6, 1)
    sols = fsqca.minimize(tt, memb, pd.Series(y), directional={"A": 1, "B": 1})
    chart = fsqca.core_periphery(sols["parsimonious"], sols["intermediate"])
    par = set()
    for r in sols["parsimonious"].recipes:
        par |= {(k, v) for k, v in r.literals.items()}
    core = chart[chart["role"] == "core"]
    assert all((row.condition, 1 if row.state == "present" else 0) in par
               for row in core.itertuples())


def test_empty_solution_frame_has_columns():
    from mixtriad.fsqca import Solution
    f = Solution("conservative", [], 0.0, 0.0).to_frame()
    assert list(f.columns) == ["recipe", "consistency", "raw_coverage", "unique_coverage"]
    assert len(f) == 0
