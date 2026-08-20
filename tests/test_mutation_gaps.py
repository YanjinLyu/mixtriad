"""Round-6 targeted tests for gaps exposed by mutation analysis."""
import numpy as np
import pandas as pd
import pytest

from mixtriad import fsqca
from mixtriad.data import Schema, krippendorff_alpha
from mixtriad.regression import run_regressions


def test_calibrate_rejects_equal_anchors():
    with pytest.raises(ValueError):
        fsqca.direct_calibrate(pd.Series([1.0, 2.0]), 5.0, 5.0, 9.0)


def test_percentile_anchors_nudge_left_degenerate():
    # 10th pct == median < 90th pct: anchors must still be strictly ordered
    x = pd.Series([0, 0, 0, 0, 0, 0, 1, 2, 3, 9], dtype=float)
    lo, mid, hi = fsqca.percentile_anchors(x)
    assert lo < mid < hi
    m = fsqca.direct_calibrate(x, lo, mid, hi)
    assert m.between(0, 1).all()


def test_negated_condition_membership_is_one_minus():
    # cases deep in ~A with high outcome: the ~A recipe must be near-perfectly
    # consistent, which pins the 1 - M negation in truth-table membership
    M = pd.DataFrame({"A": [0.1] * 30})
    y = pd.Series([0.9] * 30)
    tt = fsqca.build_truth_table(M, y, 0.8, 0.6, 1)
    sols = fsqca.minimize(tt, M, y)
    rec = sols["conservative"].recipes
    assert len(rec) == 1 and rec[0].literals == {"A": 0}
    assert rec[0].consistency == pytest.approx(1.0, abs=1e-9)
    # pin the truth-table row consistency too (independent negation site)
    assert float(tt.table["raw_consistency"].iloc[0]) == pytest.approx(1.0, abs=1e-6)


def test_unique_coverage_positive_for_sole_recipe():
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 1, 200)
    y = np.clip(a - 0.05, 0, 1)
    M = pd.DataFrame({"A": a})
    tt = fsqca.build_truth_table(M, pd.Series(y), 0.8, 0.6, 1)
    s = fsqca.minimize(tt, M, pd.Series(y))["conservative"]
    assert s.recipes and s.recipes[0].unique_coverage > 0.3
    assert s.recipes[0].unique_coverage == pytest.approx(s.recipes[0].raw_coverage, abs=1e-9)


def test_directional_expectations_filter_remainders():
    # observed sufficient row (1,1); remainders (1,0),(0,1),(0,0);
    # with directional A: presence, only (1,0) is an easy counterfactual,
    # so the intermediate solution must be exactly A
    a = np.concatenate([np.full(40, 0.9), np.full(10, 0.55)])
    b = np.concatenate([np.full(40, 0.9), np.full(10, 0.55)])
    y = np.concatenate([np.full(40, 0.85), np.full(10, 0.6)])
    M = pd.DataFrame({"A": a, "B": b})
    tt_rows = fsqca.TruthTable(["A", "B"],
        pd.DataFrame([{"A": 1, "B": 1, "n_cases": 50, "raw_consistency": 0.95,
                       "PRI": 0.9, "outcome": 1}]), 0.8, 1)
    sols = fsqca.minimize(tt_rows, M, pd.Series(y), directional={"A": 1})
    inter = sols["intermediate"].recipes
    assert any(r.literals == {"A": 1} for r in inter)
    assert all("B" not in r.literals or r.literals.get("A") == 1 for r in inter)


def test_krippendorff_exact_small_table():
    # 4 items x 2 coders, one unit disagreement; interval-alpha hand value
    codes = np.array([[1, 1], [2, 2], [3, 4], [5, 5]], dtype=float)
    a = krippendorff_alpha(codes)
    # independent hand computation of the same estimator
    rows = [r for r in codes]
    Do = sum(((r[0]-r[1])**2 + (r[1]-r[0])**2) / (2-1) for r in rows) / (2*len(rows))
    vals = codes.ravel()
    De = ((vals[:, None]-vals[None, :])**2).sum() / (len(vals)*(len(vals)-1))
    assert a == pytest.approx(1 - Do/De, abs=1e-12)
    assert 0 < a < 1


def test_significance_stars_present_in_tidy_output():
    rng = np.random.default_rng(4)
    n = 400
    x1 = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    df = pd.DataFrame({"x1": x1, "junk": rng.normal(0, 1, n),
                       "engagement": np.round(np.exp(2 + 1.5*x1 + 0.3*noise)).astype(int)})
    schema = Schema(outcome="engagement", antecedents=["x1", "junk"])
    out = run_regressions(df, schema)
    stars = dict(zip(out["ols"]["term"], out["ols"]["sig"]))
    assert stars.get("x1") == "***" and stars.get("junk") == ""


def test_krippendorff_constant_codes_is_one():
    codes = np.array([[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]])
    assert krippendorff_alpha(codes) == 1.0
