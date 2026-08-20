"""Real-world public-data pilot: FakeNewsNet through the full MixTriad pipeline.

Downloads the public FakeNewsNet headline tables (PolitiFact fake+real;
optionally GossipCop fake) at runtime - nothing is redistributed - engineers
transparent headline/source features, and runs all three stages on genuinely
real, messy, heavy-tailed virality data (outcome = number of sharing tweets).

This is an external-validity smoke test, not a substantive study: the
features are shallow headline proxies and the corpus is a convenience
sample. Its purpose is to verify that the pipeline's behaviour on real data
matches everything verified on synthetic data: no crashes on missingness,
a recorded (not warned) count-model decision, plausible importances, and
either interpretable configurations or a correctly refused truth table.
"""
from __future__ import annotations

import io
import re
import sys
import urllib.request

import numpy as np
import pandas as pd

from mixtriad import FsqcaSpec, Pipeline, Schema

BASE = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/"
CUES = ("breaking", "shocking", "exposed", "secret", "bombshell", "revealed",
        "you won", "truth", "banned", "insane", "destroy", "slams")


def fetch(name: str) -> pd.DataFrame:
    with urllib.request.urlopen(BASE + name + ".csv", timeout=60) as r:
        return pd.read_csv(io.BytesIO(r.read()))


def engineer(df: pd.DataFrame, is_fake: int) -> pd.DataFrame:
    t = df["title"].fillna("").astype(str)
    ids = df["tweet_ids"].fillna("").astype(str)
    words = t.str.split()
    out = pd.DataFrame({
        "headline_length": t.str.len(),
        "word_count": words.str.len(),
        "exclamations": t.str.count("!") + t.str.count(r"\?"),
        "caps_ratio": words.apply(
            lambda ws: np.mean([w.isupper() and len(w) > 1 for w in ws]) if ws else 0.0),
        "clickbait_cues": t.str.lower().apply(lambda s: sum(c in s for c in CUES)),
        "is_fake": is_fake,
        "engagement": ids.apply(lambda s: 0 if not s.strip()
                                else len(re.split(r"[\t ]+", s.strip()))),
    })
    return out[out["headline_length"] > 0].reset_index(drop=True)


def run(corpus: str = "politifact", outdir: str = "examples/fnn_out",
        n_trials: int = 10, seeds=(42, 123)) -> Pipeline:
    if corpus == "politifact":
        df = pd.concat([engineer(fetch("politifact_fake"), 1),
                        engineer(fetch("politifact_real"), 0)], ignore_index=True)
        antecedents = ["headline_length", "word_count", "exclamations",
                       "caps_ratio", "clickbait_cues", "is_fake"]
    else:
        df = engineer(fetch("gossipcop_fake"), 1)
        antecedents = ["headline_length", "word_count", "exclamations",
                       "caps_ratio", "clickbait_cues"]

    print(f"[{corpus}] n = {len(df)}, zero-engagement share = "
          f"{(df['engagement'] == 0).mean():.1%}, "
          f"max engagement = {df['engagement'].max():,}")
    schema = Schema(outcome="engagement", antecedents=antecedents, controls=[])
    p = Pipeline(df, schema, outdir=outdir)
    p.stage1_regression()
    nb = p.reg.get("negbin")
    if nb is not None:
        print(f"  count model: {nb['model'].iloc[0]}  "
              f"(meta: {dict(list(nb.attrs.items())[:3])})")
    p.stage2_boosting(seeds=seeds, n_trials=n_trials)
    print(f"  retained conditions: {p.boost.retained}")
    spec = FsqcaSpec(directional={"is_fake": 1, "clickbait_cues": 1},
                     consistency_cutoff=0.80, pri_cutoff=0.60, frequency_cutoff=3)
    q = p.stage3_fsqca(spec)
    inter = q["solutions"]["intermediate"]
    if inter.recipes:
        print(f"  intermediate solution (cons={inter.solution_consistency:.3f}, "
              f"cov={inter.solution_coverage:.3f}):")
        for r in inter.recipes:
            print(f"    {r.label()}  cons={r.consistency:.3f} cov={r.raw_coverage:.3f}")
    else:
        print("  truth table refused a sufficient solution (documented behaviour)")
    p.write_report()
    return p


if __name__ == "__main__":
    run(corpus=sys.argv[1] if len(sys.argv) > 1 else "politifact")
