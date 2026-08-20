import numpy as np
import pandas as pd
import pytest

from mixtriad import fsqca
from mixtriad.data import krippendorff_alpha


def test_calibration_anchors():
    x = pd.Series([0.0, 5.0, 10.0])
    m = fsqca.direct_calibrate(x, 0, 5, 10)
    assert m.iloc[0] == pytest.approx(0.05, abs=0.01)
    assert m.iloc[1] == pytest.approx(0.50, abs=0.01)
    assert m.iloc[2] == pytest.approx(0.95, abs=0.01)


def test_consistency_bounds():
    x = np.array([0.9, 0.8, 0.2])
    y = np.array([0.95, 0.85, 0.1])
    c = fsqca.consistency(x, y)
    assert 0 <= c <= 1 and c > 0.9


def test_quine_mccluskey_textbook():
    sols = fsqca.quine_mccluskey([(1, 1, 0), (1, 1, 1), (1, 0, 1)])
    assert set(sols) == {(1, 1, "-"), (1, "-", 1)}


def test_qm_with_remainders_simplifies():
    pos = [(1, 0, 0), (1, 1, 0)]
    rem = [(1, 0, 1), (1, 1, 1)]
    sols = fsqca.quine_mccluskey(pos, rem)
    assert (1, "-", "-") in sols


def test_pipeline_recovers_planted_configuration():
    rng = np.random.default_rng(7)
    n = 300
    a = rng.uniform(0, 1, n)
    b = rng.uniform(0, 1, n)
    c = rng.uniform(0, 1, n)
    y = np.minimum(a, b)
    memb = pd.DataFrame({"A": a, "B": b, "C": c})
    tt = fsqca.build_truth_table(memb, pd.Series(y), 0.8, 0.6, 1)
    sols = fsqca.minimize(tt, memb, pd.Series(y))
    labels = [r.label() for r in sols["conservative"].recipes]
    assert any(("A" in lab and "B" in lab) for lab in labels)
    assert sols["conservative"].solution_consistency > 0.85


def test_krippendorff_perfect_agreement():
    codes = np.array([[1, 1, 1], [4, 4, 4], [7, 7, 7]], dtype=float)
    assert krippendorff_alpha(codes) == pytest.approx(1.0)


def test_krippendorff_reasonable_range():
    rng = np.random.default_rng(0)
    truth = rng.integers(1, 8, 60).astype(float)
    codes = np.stack([truth + rng.normal(0, 0.5, 60) for _ in range(3)], axis=1)
    a = krippendorff_alpha(codes)
    assert 0.8 < a <= 1.0
