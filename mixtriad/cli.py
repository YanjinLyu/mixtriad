"""Command-line entry point: `mixtriad run config.yaml`."""
import argparse
import sys

import pandas as pd
import yaml

from .data import Schema
from .pipeline import FsqcaSpec, Pipeline


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mixtriad",
                                 description="Three-stage mixed-methods pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run", help="run the full pipeline from a YAML config")
    runp.add_argument("config")
    runp.add_argument("--outdir", default=None)
    runp.add_argument("--trials", type=int, default=None)
    sc = sub.add_parser("selfcheck", help="verify this installation against the built-in oracle")
    sc.add_argument("--full", action="store_true", help="also run a Stage-2 boosting pass")
    args = ap.parse_args(argv)

    if args.cmd == "selfcheck":
        from .selfcheck import run_selfcheck
        return 0 if run_selfcheck(full=args.full)["passed"] else 1

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    df = pd.read_csv(cfg["data"])
    schema = Schema(outcome=cfg["outcome"],
                    antecedents=cfg["antecedents"],
                    controls=cfg.get("controls", []),
                    dimensions=cfg.get("dimensions", {}),
                    log_transform=cfg.get("log_transform", []),
                    categorical=cfg.get("categorical", []))
    q = cfg.get("fsqca", {})
    spec = FsqcaSpec(anchors={k: tuple(v) for k, v in q.get("anchors", {}).items()} or None,
                     directional=q.get("directional"),
                     consistency_cutoff=q.get("consistency_cutoff", 0.80),
                     pri_cutoff=q.get("pri_cutoff", 0.70),
                     frequency_cutoff=q.get("frequency_cutoff", 1))
    outdir = args.outdir or cfg.get("outdir", "mixtriad_out")
    trials = args.trials or cfg.get("n_trials", 50)
    Pipeline(df, schema, outdir).run_all(spec, n_trials=trials)
    print(f"MixTriad finished; artefacts in {outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
