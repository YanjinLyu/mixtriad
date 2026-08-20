# MixTriad

**Three-stage mixed-methods triangulation for observational social-media research:
parametric regression → Bayesian-optimised gradient boosting → fuzzy-set QCA (fsQCA), in one reproducible Python pipeline.**

MixTriad packages a complete "net-effects + predictive-importance + configurational"
evidence chain into a single, scriptable workflow. It was developed for studying the
virality of AI-generated misinformation short videos, but nothing in the pipeline is
tied to that domain: any tabular dataset with a continuous or count outcome and a
moderate number of theorised antecedents (technology adoption, policy diffusion,
health behaviour, crowdfunding success, ...) can be analysed with the same three
commands.

## Why

- Researchers increasingly triangulate regression, machine-learning importance, and
  QCA in the same paper, yet today this requires juggling three ecosystems
  (R/Stata → Python → the standalone fsQCA GUI) with manual, error-prone hand-offs.
- fsQCA in particular has no maintained, importable Python implementation. MixTriad
  provides direct calibration, necessity analysis, truth-table construction with
  consistency/PRI/frequency cut-offs, and Quine–McCluskey minimisation producing
  conservative, parsimonious, and intermediate solutions — as ordinary,
  unit-tested Python functions.
- The gradient-boosting stage enforces a leakage-safe protocol (per-repetition TPE
  tuning on the training fold only, metrics averaged over five fixed-seed 80/20
  hold-outs) so that reported RMSE/R² are honest out-of-sample numbers.

## Install

```bash
pip install -e .            # core (numpy, pandas, scikit-learn, statsmodels, matplotlib, pyyaml)
pip install -e ".[all]"     # + xgboost, optuna, shap (recommended)
```

Python ≥ 3.9. Without the optional extras the pipeline transparently falls back to
scikit-learn's `HistGradientBoostingRegressor` with random search.

## Quick start (CLI)

```bash
python examples/make_dataset.py                  # ships a 388-row synthetic corpus
mixtriad run examples/misinfo_config.yaml        # full three-stage pipeline
```

Artefacts appear in `examples/demo_out/`: tidy CSV tables for every stage,
`fig_importance.png`, `fig_model_comparison.png`, `fig_configurations_outcome.png`,
`fig_configurations_negated.png`, and a consolidated `report.md`.

## Quick start (API)

```python
import pandas as pd
from mixtriad import Pipeline, FsqcaSpec, Schema

df = pd.read_csv("examples/misinfo_videos_synthetic.csv")
schema = Schema(
    outcome="engagement",
    antecedents=["creator_reach", "creator_activity", "audience_youthfulness",
                 "topic_category", "video_duration", "headline_length",
                 "hashtag_count", "emotional_intensity", "visual_realism",
                 "av_coherence", "account_topic_focus"],
    controls=["publication_age"],
    log_transform=["creator_reach", "creator_activity"],
    categorical=["topic_category"],
)
spec = FsqcaSpec(directional={"creator_reach": 1, "account_topic_focus": 1,
                              "emotional_intensity": 1})
p = Pipeline(df, schema, outdir="out").run_all(spec, n_trials=50)

p.reg["ols"]                          # stage 1: tidy OLS table (+ negbin, VIF)
p.boost.metrics                       # stage 2: model-comparison panel
p.boost.retained                      # antecedents above the elbow cut-off
p.qca["outcome"]["solutions"]["intermediate"].to_frame()   # stage 3 recipes
```

Each stage is also importable on its own:

```python
from mixtriad import fsqca
m = fsqca.direct_calibrate(x, full_non=10, crossover=50, full_mem=200)
tt = fsqca.build_truth_table(memberships, outcome, consistency_cutoff=0.8)
sols = fsqca.minimize(tt, memberships, outcome, directional={"A": 1})
```

`mixtriad.krippendorff_alpha(codes)` computes inter-coder reliability for
hand-coded ordinal antecedents (items × coders array, NaN = missing).

## Repository layout

```
mixtriad/            package (data, regression, boosting, fsqca, pipeline, cli)
examples/            synthetic dataset generator + YAML config + demo output
tests/               pytest suite (calibration, Quine–McCluskey, end-to-end)
docs/                architecture figure source
```

## Testing

```bash
python -m pytest tests -q
```

## Citing

If you use MixTriad, please cite the accompanying SoftwareX article and this repository:

> Y. Lu, MixTriad: A Python toolkit for triangulating regression, gradient-boosting, and fuzzy-set QCA evidence in observational social-media research, SoftwareX (under review). Repository: https://github.com/USERNAME/mixtriad (see `CITATION.cff`; a Zenodo DOI is minted per `RELEASE_CHECKLIST.md`).

## License

MIT.
