"""Installation self-verification: `mixtriad selfcheck`.

Productised from the project's round-5/6 QA assets, this module lets any
user verify that their installation reproduces the verified behaviour of
the analytic core on their own machine:

  1. Boolean-minimisation oracle - the Quine-McCluskey engine is checked
     against an independent brute-force reference (exhaustive prime
     enumeration + exact minimum cover) over ALL 255 non-empty Boolean
     functions of three conditions: coverage, domain, primality, and
     exact minimality must all hold.
  2. Calibration properties - anchor semantics (0.05 / 0.50 / 0.95),
     exact monotonicity, and the crossover `nextafter` invariant.
  3. Set-measure bounds - consistency/coverage/PRI within [0, 1] on
     seeded random data.
  4. Micro end-to-end - a 120-row synthetic dataset through Stage 1
     (regression) and Stage 3 (fsQCA); `--full` adds a one-seed,
     five-trial Stage-2 boosting pass.

Exit code 0 means every check passed.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from . import fsqca
from .data import Schema
from .fsqca import _covers, quine_mccluskey
from .regression import run_regressions


# ---------------- independent brute-force reference ---------------- #
def _cells(imp, k):
    free = [i for i, v in enumerate(imp) if v == "-"]
    out = []
    for bits in itertools.product((0, 1), repeat=len(free)):
        c = list(imp)
        for f, b in zip(free, bits):
            c[f] = b
        out.append(tuple(c))
    return out


def _brute_min_cover(pos, k):
    allowed = set(pos)
    imps = [i for i in itertools.product((0, 1, "-"), repeat=k)
            if all(c in allowed for c in _cells(i, k))]
    primes = [a for a in imps
              if not any(a != b and all(bv == "-" or bv == av
                                        for av, bv in zip(a, b)) for b in imps)]
    for size in range(1, len(primes) + 1):
        for combo in itertools.combinations(primes, size):
            if all(any(_covers(p, m) for p in combo) for m in pos):
                return size, primes
    return len(primes), primes


def _check_qm_oracle():
    univ = list(itertools.product((0, 1), repeat=3))
    n, fails = 0, 0
    for mask in range(1, 256):
        pos = [m for i, m in enumerate(univ) if (mask >> i) & 1]
        sol = quine_mccluskey(pos)
        mc, primes = _brute_min_cover(pos, 3)
        ok = (all(any(_covers(p, m) for p in sol) for m in pos)      # cover
              and all(set(_cells(p, 3)) <= set(pos) for p in sol)   # domain
              and all(p in primes for p in sol)                      # primality
              and len(sol) == mc)                                    # minimality
        n += 1
        fails += (not ok)
    return {"name": "QM differential oracle (exhaustive k=3)",
            "cases": n, "failures": fails, "passed": fails == 0}


def _check_calibration():
    x = pd.Series([0.0, 5.0, 10.0])
    m = fsqca.direct_calibrate(x, 0, 5, 10)
    anchors_ok = (abs(m.iloc[0] - 0.05) < 0.01 and abs(m.iloc[2] - 0.95) < 0.01
                  and m.iloc[1] > 0.5 and m.iloc[1] - 0.5 < 1e-6)
    g = pd.Series(np.linspace(-8, 8, 400))
    mg = fsqca.direct_calibrate(g, -5, 0, 5)
    mono_ok = bool((np.diff(mg.to_numpy()) >= 0).all())
    no_half = bool((mg != 0.5).all())
    ok = anchors_ok and mono_ok and no_half
    return {"name": "calibration anchors / monotonicity / crossover invariant",
            "cases": 403, "failures": int(not ok), "passed": ok}


def _check_measures(seed=11):
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(50):
        c, y = rng.uniform(0, 1, 40), rng.uniform(0, 1, 40)
        for f in (fsqca.consistency, fsqca.coverage, fsqca.pri):
            v = f(c, y)
            bad += not (-1e-9 <= v <= 1 + 1e-9)
    return {"name": "set-measure bounds on random data",
            "cases": 150, "failures": bad, "passed": bad == 0}


def _micro_data(n=120, seed=5):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 1, n)
    b = rng.uniform(0, 1, n)
    ctrl = rng.integers(1, 60, n)
    y = np.round(np.exp(1.5 + 2.2 * np.minimum(a, b) + rng.normal(0, 0.4, n))).astype(int)
    return pd.DataFrame({"A": a, "B": b, "age": ctrl, "engagement": y})


def _check_end_to_end(full=False):
    df = _micro_data()
    schema = Schema(outcome="engagement", antecedents=["A", "B"], controls=["age"])
    reg = run_regressions(df, schema)
    ok = "ols" in reg and reg["ols"]["estimate"].notna().all()
    memb = pd.DataFrame({c: fsqca.direct_calibrate(df[c], *fsqca.percentile_anchors(df[c]))
                         for c in ("A", "B")})
    yv = fsqca.direct_calibrate(np.log1p(df["engagement"]),
                                *fsqca.percentile_anchors(np.log1p(df["engagement"])))
    tt = fsqca.build_truth_table(memb, yv, 0.75, 0.55, 1)
    sols = fsqca.minimize(tt, memb, yv)
    ok = ok and any(r.literals.get("A") == 1 and r.literals.get("B") == 1
                    for r in sols["conservative"].recipes)
    extra = ""
    if full:
        from .boosting import run_boosting
        res = run_boosting(df, schema, seeds=(42,), n_trials=5)
        ok = ok and {"RMSE", "MAE", "R2"} <= set(res.metrics.columns)
        extra = " + Stage-2 boosting"
    return {"name": f"micro end-to-end (Stage 1 + Stage 3{extra})",
            "cases": 1, "failures": int(not ok), "passed": ok}


def run_selfcheck(full: bool = False, verbose: bool = True) -> dict:
    t0 = time.time()
    checks = [_check_qm_oracle(), _check_calibration(),
              _check_measures(), _check_end_to_end(full)]
    passed = all(c["passed"] for c in checks)
    report = {"passed": passed, "elapsed_s": round(time.time() - t0, 2),
              "mode": "full" if full else "quick", "checks": checks}
    if verbose:
        for c in checks:
            mark = "PASS" if c["passed"] else "FAIL"
            print(f"[{mark}] {c['name']}  ({c['cases']} cases, "
                  f"{c['failures']} failures)")
        print(f"selfcheck {'PASSED' if passed else 'FAILED'} "
              f"in {report['elapsed_s']} s ({report['mode']} mode)")
    return report
