"""Fuzzy-set Qualitative Comparative Analysis (fsQCA) in pure Python.

Implements the standard Ragin workflow: direct calibration, necessity
analysis, truth-table construction with consistency/PRI thresholds, and
Boolean minimization (Quine-McCluskey) yielding conservative (complex),
parsimonious, and directional-expectation-filtered intermediate solutions.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def direct_calibrate(x: pd.Series, full_non: float, crossover: float,
                     full_mem: float) -> pd.Series:
    """Ragin's direct method: map a raw variable to [0, 1] fuzzy membership.

    Uses the log-odds transformation with anchors at full non-membership
    (membership 0.05), crossover (0.50) and full membership (0.95).
    """
    if not (full_non < crossover < full_mem):
        raise ValueError("anchors must satisfy full_non < crossover < full_mem")
    x = pd.to_numeric(x, errors="coerce").astype(float)
    upper = np.log(0.95 / 0.05) / (full_mem - crossover)
    lower = np.log(0.05 / 0.95) / (full_non - crossover)
    dev = x - crossover
    logodds = np.where(dev >= 0, dev * upper, dev * lower)
    mem = 1.0 / (1.0 + np.exp(-logodds))
    # cases exactly at the crossover would drop out of the truth table;
    # lift them by the smallest representable float above 0.5, which keeps
    # calibration exactly monotone (any case strictly above the crossover
    # already has membership >= nextafter(0.5, 1))
    mem = np.where(mem == 0.5, np.nextafter(0.5, 1.0), mem)
    return pd.Series(np.clip(mem, 0.0, 1.0), index=x.index, name=x.name)


def percentile_anchors(x: pd.Series, pcts=(0.10, 0.50, 0.90)) -> Tuple[float, float, float]:
    """Convenience anchors from sample percentiles (10/50/90 by default)."""
    q = pd.to_numeric(x, errors="coerce").quantile(list(pcts)).tolist()
    if not q[0] < q[1] < q[2]:  # degenerate distributions: nudge
        q = [q[1] - abs(q[1]) * 0.5 - 1, q[1], q[1] + abs(q[1]) * 0.5 + 1]
    return tuple(q)


# --------------------------------------------------------------------------- #
# Set-theoretic measures
# --------------------------------------------------------------------------- #
def consistency(cond: np.ndarray, outc: np.ndarray) -> float:
    """Sufficiency consistency: sum(min(X, Y)) / sum(X)."""
    return float(np.minimum(cond, outc).sum() / (cond.sum() + EPS))


def coverage(cond: np.ndarray, outc: np.ndarray) -> float:
    """Sufficiency coverage: sum(min(X, Y)) / sum(Y)."""
    return float(np.minimum(cond, outc).sum() / (outc.sum() + EPS))


def pri(cond: np.ndarray, outc: np.ndarray) -> float:
    """Proportional Reduction in Inconsistency."""
    both = np.minimum(cond, outc).sum()
    neg = np.minimum(np.minimum(cond, outc), 1 - outc).sum()
    denom = cond.sum() - neg
    return float((both - neg) / (denom + EPS))


def necessity_table(memberships: pd.DataFrame, outcome: pd.Series) -> pd.DataFrame:
    """Necessity consistency/coverage for each condition and its negation."""
    y = outcome.to_numpy()
    rows = []
    for c in memberships.columns:
        for label, x in ((c, memberships[c].to_numpy()),
                         (f"~{c}", 1 - memberships[c].to_numpy())):
            nec = np.minimum(x, y).sum() / (y.sum() + EPS)
            cov = np.minimum(x, y).sum() / (x.sum() + EPS)
            rows.append({"condition": label,
                         "necessity_consistency": round(float(nec), 3),
                         "necessity_coverage": round(float(cov), 3)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Truth table
# --------------------------------------------------------------------------- #
@dataclass
class TruthTable:
    conditions: List[str]
    table: pd.DataFrame          # one row per configuration
    consistency_cutoff: float
    frequency_cutoff: int

    @property
    def positive_rows(self) -> List[Tuple[int, ...]]:
        sel = self.table[self.table["outcome"] == 1]
        return [tuple(int(v) for v in r) for r in sel[self.conditions].to_numpy()]

    @property
    def negative_rows(self) -> List[Tuple[int, ...]]:
        sel = self.table[self.table["outcome"] == 0]
        return [tuple(int(v) for v in r) for r in sel[self.conditions].to_numpy()]

    @property
    def remainder_rows(self) -> List[Tuple[int, ...]]:
        observed = {tuple(int(v) for v in r)
                    for r in self.table[self.conditions].to_numpy()}
        k = len(self.conditions)
        return [c for c in itertools.product((0, 1), repeat=k) if c not in observed]


def build_truth_table(memberships: pd.DataFrame, outcome: pd.Series,
                      consistency_cutoff: float = 0.80,
                      pri_cutoff: float = 0.70,
                      frequency_cutoff: int = 1) -> TruthTable:
    """Assign each case to its best-fitting corner of the property space and
    score every observed configuration for sufficiency."""
    conds = list(memberships.columns)
    M = memberships.to_numpy()
    y = outcome.to_numpy()
    corners = (M > 0.5).astype(int)
    df = pd.DataFrame(corners, columns=conds)
    df["_row"] = [tuple(r) for r in corners]

    rows = []
    for combo, idx in df.groupby("_row").groups.items():
        idx = list(idx)
        # membership of every case in this configuration
        memb = np.ones(len(y))
        for j, bit in enumerate(combo):
            col = M[:, j] if bit else 1 - M[:, j]
            memb = np.minimum(memb, col)
        cons = consistency(memb, y)
        p = pri(memb, y)
        n = len(idx)
        keep = n >= frequency_cutoff
        rows.append({**{c: b for c, b in zip(conds, combo)},
                     "n_cases": n, "raw_consistency": round(cons, 3),
                     "PRI": round(p, 3),
                     "outcome": int(keep and cons >= consistency_cutoff and p >= pri_cutoff)
                     if keep else -1})
    table = pd.DataFrame(rows).sort_values("raw_consistency", ascending=False)
    table = table[table["outcome"] != -1].reset_index(drop=True)
    return TruthTable(conds, table, consistency_cutoff, frequency_cutoff)


# --------------------------------------------------------------------------- #
# Quine-McCluskey minimization
# --------------------------------------------------------------------------- #
def _combine(a: Tuple, b: Tuple) -> Optional[Tuple]:
    diff = 0
    out = []
    for x, y in zip(a, b):
        if x == y:
            out.append(x)
        elif x != "-" and y != "-":
            diff += 1
            out.append("-")
        else:
            return None
    return tuple(out) if diff == 1 else None


def _covers(implicant: Tuple, minterm: Tuple[int, ...]) -> bool:
    return all(i == "-" or i == m for i, m in zip(implicant, minterm))


def quine_mccluskey(minterms: Sequence[Tuple[int, ...]],
                    dont_cares: Sequence[Tuple[int, ...]] = ()) -> List[Tuple]:
    """Return the set of prime implicants covering `minterms`, optionally
    using `dont_cares` (logical remainders) during combination."""
    if not minterms:
        return []
    current = {tuple(m) for m in minterms} | {tuple(d) for d in dont_cares}
    primes: set = set()
    while current:
        used, nxt = set(), set()
        for a, b in itertools.combinations(sorted(current, key=lambda t: tuple(map(str, t))), 2):
            c = _combine(a, b)
            if c is not None:
                nxt.add(c)
                used.add(a)
                used.add(b)
        primes |= (current - used)
        current = nxt
    # essential-first greedy cover of the *observed* minterms
    minterms = [tuple(m) for m in minterms]
    chart = {m: [p for p in primes if _covers(p, m)] for m in minterms}
    solution: List[Tuple] = []
    uncovered = set(minterms)
    for m, ps in chart.items():
        if len(ps) == 1 and ps[0] not in solution:
            solution.append(ps[0])
    for s in solution:
        uncovered -= {m for m in uncovered if _covers(s, m)}
    if not uncovered:
        return solution
    # complete the cover: exact minimum-size search on small prime sets
    # (always the practical fsQCA regime), greedy fallback otherwise
    cand = sorted((p for p in primes if p not in solution
                   and any(_covers(p, m) for m in uncovered)),
                  key=lambda t: tuple(map(str, t)))
    greedy, unc_g = [], set(uncovered)
    while unc_g:
        best = max(cand, key=lambda p: sum(_covers(p, m) for m in unc_g))
        greedy.append(best)
        unc_g -= {m for m in unc_g if _covers(best, m)}
    if len(cand) <= 18:
        for size in range(1, len(greedy)):
            for combo in itertools.combinations(cand, size):
                if all(any(_covers(p, m) for p in combo) for m in uncovered):
                    return solution + list(combo)
    return solution + greedy


# --------------------------------------------------------------------------- #
# Solutions
# --------------------------------------------------------------------------- #
@dataclass
class Recipe:
    literals: Dict[str, int]           # condition -> 1 (present) / 0 (absent)
    consistency: float = 0.0
    raw_coverage: float = 0.0
    unique_coverage: float = 0.0

    def label(self) -> str:
        parts = [(c if v else f"~{c}") for c, v in self.literals.items()]
        return " * ".join(parts) if parts else "(empty)"


@dataclass
class Solution:
    kind: str
    recipes: List[Recipe]
    solution_consistency: float
    solution_coverage: float

    def to_frame(self) -> pd.DataFrame:
        cols = ["recipe", "consistency", "raw_coverage", "unique_coverage"]
        rows = [{"recipe": r.label(),
                 "consistency": round(r.consistency, 3),
                 "raw_coverage": round(r.raw_coverage, 3),
                 "unique_coverage": round(r.unique_coverage, 3)} for r in self.recipes]
        df = pd.DataFrame(rows, columns=cols)
        df.attrs["solution_consistency"] = round(self.solution_consistency, 3)
        df.attrs["solution_coverage"] = round(self.solution_coverage, 3)
        return df


def _recipe_membership(recipe: Recipe, memberships: pd.DataFrame) -> np.ndarray:
    memb = np.ones(len(memberships))
    for c, v in recipe.literals.items():
        col = memberships[c].to_numpy()
        memb = np.minimum(memb, col if v else 1 - col)
    return memb


def _score(recipes: List[Recipe], memberships: pd.DataFrame,
           outcome: pd.Series, kind: str) -> Solution:
    y = outcome.to_numpy()
    membs = [_recipe_membership(r, memberships) for r in recipes]
    for i, (r, m) in enumerate(zip(recipes, membs)):
        r.consistency = consistency(m, y)
        r.raw_coverage = coverage(m, y)
        others = [membs[j] for j in range(len(membs)) if j != i]
        rest = np.maximum.reduce(others) if others else np.zeros_like(m)
        r.unique_coverage = float(np.minimum(np.maximum(m - rest, 0), y).sum() / (y.sum() + EPS))
    total = np.maximum.reduce(membs) if membs else np.zeros_like(y)
    return Solution(kind, recipes, consistency(total, y), coverage(total, y))


def _implicants_to_recipes(implicants: List[Tuple], conditions: List[str]) -> List[Recipe]:
    out = []
    for imp in implicants:
        lits = {c: int(v) for c, v in zip(conditions, imp) if v != "-"}
        out.append(Recipe(lits))
    return out


def minimize(tt: TruthTable, memberships: pd.DataFrame, outcome: pd.Series,
             directional: Optional[Dict[str, int]] = None) -> Dict[str, Solution]:
    """Derive conservative, parsimonious, and intermediate solutions.

    `directional` maps a condition name to its expected contribution to the
    outcome (1 = presence, 0 = absence).  Remainders contradicting a
    directional expectation are withheld from the intermediate minimization
    (an 'easy counterfactuals only' policy).
    """
    pos = tt.positive_rows
    rem = tt.remainder_rows
    conds = tt.conditions

    conservative = _implicants_to_recipes(quine_mccluskey(pos, ()), conds)
    parsimonious = _implicants_to_recipes(quine_mccluskey(pos, rem), conds)

    if directional:
        idx = {c: i for i, c in enumerate(conds)}
        def easy(r: Tuple[int, ...]) -> bool:
            return all(r[idx[c]] == v for c, v in directional.items() if c in idx)
        inter_rows = [r for r in rem if easy(r)]
    else:
        inter_rows = []
    intermediate = _implicants_to_recipes(quine_mccluskey(pos, inter_rows), conds)

    return {"conservative": _score(conservative, memberships, outcome, "conservative"),
            "parsimonious": _score(parsimonious, memberships, outcome, "parsimonious"),
            "intermediate": _score(intermediate, memberships, outcome, "intermediate")}


def core_periphery(parsimonious: Solution, intermediate: Solution) -> pd.DataFrame:
    """Classify each literal of the intermediate solution as core (also in the
    parsimonious solution) or peripheral - the basis of configuration charts."""
    par_lits = set()
    for r in parsimonious.recipes:
        par_lits |= {(c, v) for c, v in r.literals.items()}
    rows = []
    for i, r in enumerate(intermediate.recipes, start=1):
        for c, v in r.literals.items():
            rows.append({"configuration": f"C{i}", "condition": c,
                         "state": "present" if v else "absent",
                         "role": "core" if (c, v) in par_lits else "peripheral"})
    return pd.DataFrame(rows)
