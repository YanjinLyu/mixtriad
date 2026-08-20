"""Round-5 property-based suite (hypothesis) + D8 minimality regression."""
import itertools

import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st

from mixtriad import fsqca
from mixtriad.fsqca import _covers, quine_mccluskey

settings.register_profile("ci", max_examples=40, deadline=None)
settings.load_profile("ci")


# ---------- brute-force helpers (independent oracle, test-local) ----------
def _cells(imp, k):
    free = [i for i, v in enumerate(imp) if v == "-"]
    out = []
    for bits in itertools.product((0, 1), repeat=len(free)):
        c = list(imp)
        for f, b in zip(free, bits):
            c[f] = b
        out.append(tuple(c))
    return out


def _min_cover_size(pos, dc, k):
    allowed = set(pos) | set(dc)
    imps = [i for i in itertools.product((0, 1, "-"), repeat=k)
            if all(c in allowed for c in _cells(i, k))
            and any(c in set(pos) for c in _cells(i, k))]
    primes = [a for a in imps
              if not any(a != b and all(bv == "-" or bv == av
                                        for av, bv in zip(a, b)) for b in imps)]
    for size in range(1, len(primes) + 1):
        for combo in itertools.combinations(primes, size):
            if all(any(_covers(p, m) for p in combo) for m in pos):
                return size
    return len(primes)


minterm3 = st.tuples(*[st.integers(0, 1)] * 3)


@given(st.sets(minterm3, min_size=1, max_size=8))
def test_qm_exact_minimality_k3(pos):
    sol = quine_mccluskey(list(pos))
    assert len(sol) == _min_cover_size(list(pos), [], 3)


@given(st.sets(minterm3, min_size=2, max_size=8))
def test_qm_cover_and_domain_with_dontcares(cells):
    cells = list(cells)
    pos, dc = cells[: max(1, len(cells) // 2)], cells[max(1, len(cells) // 2):]
    sol = quine_mccluskey(pos, dc)
    allowed = set(pos) | set(dc)
    assert all(any(_covers(p, m) for p in sol) for m in pos)          # covers all
    assert all(set(_cells(p, 3)) <= allowed for p in sol)             # stays in domain


@given(st.lists(st.floats(-1e4, 1e4, allow_nan=False), min_size=5, max_size=60),
       st.floats(0.1, 100.0))
def test_calibration_bounded_and_monotone(xs, spread):
    x = pd.Series(sorted(xs))
    lo, mid, hi = float(x.min()) - spread, float(x.median()), float(x.max()) + spread
    m = fsqca.direct_calibrate(x, lo, mid, hi)
    assert m.between(0, 1).all()
    assert (np.diff(m.to_numpy()) >= -1e-12).all()


@given(st.lists(st.floats(0, 1, allow_nan=False), min_size=3, max_size=50),
       st.lists(st.floats(0, 1, allow_nan=False), min_size=3, max_size=50))
def test_measures_bounded(a, b):
    n = min(len(a), len(b))
    c, y = np.array(a[:n]), np.array(b[:n])
    for f in (fsqca.consistency, fsqca.coverage):
        assert -1e-9 <= f(c, y) <= 1 + 1e-9
