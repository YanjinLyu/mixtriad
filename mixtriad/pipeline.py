"""One-call orchestration of the three-stage pipeline plus publication-ready
tables and figures."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import fsqca
from .boosting import BoostResult, run_boosting
from .data import Schema
from .regression import run_regressions


@dataclass
class FsqcaSpec:
    anchors: Dict[str, Tuple[float, float, float]] | None = None   # var -> (non, cross, full)
    directional: Dict[str, int] | None = None
    consistency_cutoff: float = 0.80
    pri_cutoff: float = 0.70
    frequency_cutoff: int = 1
    negate_outcome: bool = False


class Pipeline:
    """regression -> tuned boosting -> fsQCA, with artefacts written to disk."""

    def __init__(self, df: pd.DataFrame, schema: Schema, outdir: str = "mixtriad_out"):
        self.df, self.schema, self.outdir = df, schema, outdir
        os.makedirs(outdir, exist_ok=True)
        self.reg: Dict[str, pd.DataFrame] = {}
        self.boost: Optional[BoostResult] = None
        self.qca: Dict[str, object] = {}

    # ---------------- stages ---------------- #
    def stage1_regression(self) -> Dict[str, pd.DataFrame]:
        self.reg = run_regressions(self.df, self.schema)
        for name, t in self.reg.items():
            t.to_csv(os.path.join(self.outdir, f"stage1_{name}.csv"), index=False)
        return self.reg

    def stage2_boosting(self, seeds=(42, 123, 456, 789, 1024),
                        n_trials: int = 50,
                        importance_cutoff: Optional[float] = None) -> BoostResult:
        self.boost = run_boosting(self.df, self.schema, seeds, n_trials,
                                  importance_cutoff)
        self.boost.metrics.to_csv(os.path.join(self.outdir, "stage2_model_comparison.csv"), index=False)
        self.boost.importance.to_csv(os.path.join(self.outdir, "stage2_importance.csv"), index=False)
        self._plot_importance()
        self._plot_model_comparison()
        return self.boost

    def stage3_fsqca(self, spec: FsqcaSpec = FsqcaSpec(),
                     conditions: Optional[List[str]] = None) -> Dict[str, object]:
        conds = conditions or (self.boost.retained if self.boost else self.schema.antecedents)
        memb = pd.DataFrame(index=self.df.index)
        anchors_used = {}
        for c in conds:
            raw = pd.to_numeric(self.df[c], errors="coerce")
            user_anchor = (spec.anchors or {}).get(c)
            anc = user_anchor or fsqca.percentile_anchors(raw)
            q10, q50, q90 = raw.quantile([.10, .50, .90])
            degen = (not user_anchor) and (q10 == q50 or q50 == q90)
            anchors_used[c] = {"anchors": tuple(round(float(a), 4) for a in anc),
                               "degenerate_percentiles": bool(degen)}
            if degen:
                anchors_used[c]["note"] = ("10/50/90 percentiles are not distinct "
                    "(sparse or discrete variable); zero-mass cases sit at the "
                    "crossover and are classified 'present'. Supply theory-based "
                    "anchors via FsqcaSpec(anchors=...) for this condition.")
            memb[c] = fsqca.direct_calibrate(raw, *anc)
        y_raw = pd.to_numeric(self.df[self.schema.outcome], errors="coerce")
        y_anc = fsqca.percentile_anchors(np.log1p(y_raw))
        y = fsqca.direct_calibrate(np.log1p(y_raw), *y_anc)
        if spec.negate_outcome:
            y = 1 - y

        nec = fsqca.necessity_table(memb, y)
        tt = fsqca.build_truth_table(memb, y, spec.consistency_cutoff,
                                     spec.pri_cutoff, spec.frequency_cutoff)
        sols = fsqca.minimize(tt, memb, y, spec.directional)
        chart = fsqca.core_periphery(sols["parsimonious"], sols["intermediate"])

        tag = "negated" if spec.negate_outcome else "outcome"
        nec.to_csv(os.path.join(self.outdir, f"stage3_necessity_{tag}.csv"), index=False)
        tt.table.to_csv(os.path.join(self.outdir, f"stage3_truthtable_{tag}.csv"), index=False)
        for k, s in sols.items():
            f = s.to_frame()
            f.to_csv(os.path.join(self.outdir, f"stage3_solution_{k}_{tag}.csv"), index=False)
        chart.to_csv(os.path.join(self.outdir, f"stage3_chart_{tag}.csv"), index=False)
        with open(os.path.join(self.outdir, f"stage3_anchors_{tag}.json"), "w") as fh:
            json.dump(anchors_used, fh, indent=2)
        self.qca[tag] = {"necessity": nec, "truth_table": tt,
                         "solutions": sols, "chart": chart,
                         "anchors": anchors_used, "conditions": conds}
        self._plot_configuration_chart(chart, sols["intermediate"], tag)
        return self.qca[tag]

    def run_all(self, spec: FsqcaSpec = FsqcaSpec(), n_trials: int = 50) -> Pipeline:
        self.stage1_regression()
        self.stage2_boosting(n_trials=n_trials)
        self.stage3_fsqca(spec)
        neg = FsqcaSpec(**{**asdict(spec), "negate_outcome": True})
        self.stage3_fsqca(neg)
        self.write_report()
        return self

    # ---------------- figures ---------------- #
    def _plot_importance(self):
        imp = self.boost.importance
        fig, ax = plt.subplots(figsize=(7, 4.2))
        colors = ["#4C72B0" if r == "antecedent" else "#999999" for r in imp["role"]]
        ax.barh(imp["feature"][::-1], imp["permutation_importance"][::-1],
                color=colors[::-1])
        ax.axvline(self.boost.cutoff, ls="--", c="crimson", lw=1,
                   label=f"elbow cut-off = {self.boost.cutoff:.3f}")
        ax.set_xlabel("Permutation importance (\u0394RMSE on held-out folds)")
        ax.set_title("Stage 2: permutation importance with elbow screen")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(self.outdir, "fig_importance.png"), dpi=200)
        plt.close(fig)

    def _plot_model_comparison(self):
        m = self.boost.metrics
        fig, ax1 = plt.subplots(figsize=(7, 3.8))
        x = np.arange(len(m))
        ax1.bar(x - 0.18, m["RMSE"], width=0.36, color="#4C72B0", label="RMSE")
        ax1.set_ylabel("RMSE (log outcome)")
        ax2 = ax1.twinx()
        ax2.bar(x + 0.18, m["R2"], width=0.36, color="#DD8452", label="R\u00b2")
        ax2.set_ylabel("R\u00b2")
        ax1.set_xticks(x)
        ax1.set_xticklabels(m["model"], rotation=12, ha="right", fontsize=8)
        ax1.set_title("Stage 2: out-of-sample comparison (5\u00d7 repeated 80/20 hold-out)")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(self.outdir, "fig_model_comparison.png"), dpi=200)
        plt.close(fig)

    def _plot_configuration_chart(self, chart: pd.DataFrame,
                                  inter: fsqca.Solution, tag: str):
        if chart.empty:
            return
        conds = list(dict.fromkeys(chart["condition"]))
        configs = list(dict.fromkeys(chart["configuration"]))
        fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(configs), 0.5 * len(conds) + 1.8))
        for yi, c in enumerate(conds):
            for xi, cf in enumerate(configs):
                row = chart[(chart["condition"] == c) & (chart["configuration"] == cf)]
                if row.empty:
                    continue
                present = row["state"].iloc[0] == "present"
                core = row["role"].iloc[0] == "core"
                size = 320 if core else 140
                if present:
                    ax.scatter(xi, yi, s=size, c="k")
                else:
                    ax.scatter(xi, yi, s=size, facecolors="none", edgecolors="k",
                               linewidths=1.4)
        ax.set_xticks(range(len(configs)))
        labels = [f"{cf}\ncons={r.consistency:.2f}\ncov={r.raw_coverage:.2f}"
                  for cf, r in zip(configs, inter.recipes)]
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels(conds, fontsize=9)
        ax.set_xlim(-0.6, len(configs) - 0.4)
        ax.set_ylim(-0.7, len(conds) - 0.3)
        ax.invert_yaxis()
        which = "high outcome" if tag == "outcome" else "low outcome (~Y)"
        ax.set_title(f"Stage 3: intermediate-solution configurations for {which}\n"
                     f"\u25cf present  \u25cb absent  (large = core, small = peripheral)",
                     fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(self.outdir, f"fig_configurations_{tag}.png"), dpi=200)
        plt.close(fig)

    # ---------------- report ---------------- #
    def write_report(self):
        lines = ["# MixTriad report", ""]
        if self.reg:
            lines += ["## Stage 1 - regression",
                      f"OLS R\u00b2 = {self.reg['ols'].attrs.get('r2')}",
                      self.reg["ols"].to_markdown(index=False), ""]
        if self.boost:
            lines += ["## Stage 2 - tuned gradient boosting",
                      self.boost.metrics.to_markdown(index=False), "",
                      f"Retained antecedents (> {self.boost.cutoff:.3f}): "
                      f"{', '.join(self.boost.retained)}", ""]
        for tag, q in self.qca.items():
            sols = q["solutions"]
            lines += [f"## Stage 3 - fsQCA ({tag})",
                      f"Conditions: {', '.join(q['conditions'])}",
                      "### Intermediate solution",
                      sols["intermediate"].to_frame().to_markdown(index=False),
                      f"Solution consistency = {sols['intermediate'].solution_consistency:.3f}, "
                      f"coverage = {sols['intermediate'].solution_coverage:.3f}", ""]
        with open(os.path.join(self.outdir, "report.md"), "w") as fh:
            fh.write("\n".join(lines))
