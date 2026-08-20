"""Draw the MixTriad architecture / data-flow figure (Fig. 1 of the paper)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(9.2, 4.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")

def box(x, y, w, h, title, lines, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                fc=fc, ec="#333333", lw=1.1))
    ax.text(x + w/2, y + h - 2.9, title, ha="center", va="center",
            fontsize=9.5, fontweight="bold")
    top, bottom = y + h - 6.2, y + 2.2          # text band inside the box
    step = min((top - bottom) / max(len(lines) - 1, 1), 3.4)
    start = (top + bottom + step * (len(lines) - 1)) / 2   # centre the block
    for i, l in enumerate(lines):
        ax.text(x + w/2, start - step * i, l, ha="center", va="center", fontsize=7.8)

def arrow(x1, y1, x2, y2, label=None, dy=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.2, color="#333333"))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + dy, label, ha="center", fontsize=7.2,
                style="italic", color="#333333")

# input layer
box(2, 30, 20, 18, "Input", ["tabular dataset (CSV)", "Schema: outcome,", "antecedents, controls,",
                             "log / categorical roles"], "#EAF2FB")
box(2, 3.5, 20, 18, "data module", ["schema validation", "log1p transforms",
                                  "Krippendorff's \u03b1", "for hand-coded vars"], "#F5F5F5")
arrow(12, 30, 12, 22.2)

# stage 1
box(27, 26, 21, 22, "Stage 1 \u00b7 regression", ["OLS on log outcome", "(HC1 robust SE)",
                                     "negative binomial", "robustness model", "VIF diagnostics"], "#FDEBD0")
# stage 2
box(52, 26, 21, 22, "Stage 2 \u00b7 boosting", ["5\u00d7 repeated 80/20", "hold-out (fixed seeds)",
                                    "TPE-tuned XGBoost", "permutation importance", "elbow condition screen"], "#E8F6E8")
# stage 3
box(77, 26, 21, 22, "Stage 3 \u00b7 fsQCA", ["direct calibration", "necessity analysis",
                                  "truth table (cons/PRI)", "Quine\u2013McCluskey", "3 solution types"], "#F4E7F7")

arrow(22.6, 38, 26.4, 38)
arrow(48.6, 38, 51.4, 38, "significant\nantecedents", 3.2)
arrow(73.6, 38, 76.4, 38, "retained\nconditions", 3.2)

# outputs
box(31, 4, 62, 15, "Outputs (per stage, written to disk)",
    ["tidy CSV tables \u00b7 model-comparison panel \u00b7 importance & configuration figures",
     "necessity / truth-table / solution tables \u00b7 anchors JSON \u00b7 consolidated report.md"],
    "#FFF8DC")
arrow(37.5, 25.3, 45, 19.6); arrow(62.5, 25.3, 62, 19.6); arrow(87.5, 25.3, 79, 19.6)

ax.text(50, 50.6, "MixTriad: three-stage mixed-methods pipeline", ha="center",
        fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("docs/fig_architecture.png", dpi=220)
print("wrote docs/fig_architecture.png")
