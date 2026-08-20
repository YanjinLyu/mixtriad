# MixTriad User Manual

**Version 1.1.2** · Three-stage mixed-methods triangulation for observational social-media research: regression, tuned gradient boosting, and fuzzy-set QCA in one reproducible Python pipeline

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Requirements](#2-system-requirements)
3. [Installation](#3-installation)
4. [Functional Modules](#4-functional-modules)
5. [API Reference](#5-api-reference)
6. [Operation Guide](#6-operation-guide)
7. [Output Artefacts and Interpretation](#7-output-artefacts-and-interpretation)
8. [Validation and Accuracy](#8-validation-and-accuracy)
9. [Troubleshooting](#9-troubleshooting)
10. [Support and Version Information](#10-support-and-version-information)
11. [Appendix](#11-appendix)

---

## 1. Introduction

### 1.1 Overview

MixTriad answers one question three ways: *what drives an outcome observed in the field?* It chains three research traditions that are normally scattered across ecosystems into a single, scriptable Python workflow:

- **Net-effects evidence** — parametric regression (OLS on the log outcome, plus a negative-binomial robustness model), historically run in R or Stata
- **Predictive-importance evidence** — Bayesian-optimised gradient boosting with permutation importance under a leakage-safe repeated hold-out protocol, historically run in Python
- **Configurational evidence** — fuzzy-set Qualitative Comparative Analysis (fsQCA): direct calibration, necessity analysis, truth-table construction, and Quine–McCluskey minimisation, historically run in the standalone fsQCA GUI or R's `QCA` package

Researchers triangulating all three in one paper currently juggle three tools with manual, error-prone hand-offs. MixTriad runs the full chain from one YAML file or one Python object, writes every table and figure to disk, and — a capability the fsQCA GUI does not offer — exposes the entire fsQCA machinery as ordinary, unit-tested, importable Python functions.

The package was developed for a study of the virality of AI-generated misinformation short videos, but nothing in it is tied to that domain: any tabular dataset with a continuous or count outcome and a moderate number of theorised antecedents (technology adoption, policy diffusion, health behaviour, crowdfunding success, ...) can be analysed with the same commands.

### 1.2 Key Features

- **Stage 1 — regression**: OLS on `log1p(outcome)` with HC1 robust standard errors; a count-data robustness ladder (Poisson warm start → Cameron–Trivedi dispersion estimate → NB2, falling back to Poisson QMLE with sandwich errors when IRLS does not converge, with the choice recorded in metadata rather than hidden); a VIF collinearity table
- **Stage 2 — tuned gradient boosting**: XGBoost tuned by Optuna's TPE sampler when installed, degrading transparently to scikit-learn's `HistGradientBoostingRegressor` with random search; five-times repeated 80/20 hold-out with per-repetition tuning on the training fold only; permutation importance averaged over held-out folds; an elbow-based screen that selects the antecedents carried into Stage 3
- **Stage 3 — complete fsQCA engine**: Ragin's direct calibration, necessity analysis for every condition and its negation, truth-table construction with consistency/PRI/frequency cut-offs, and exact Quine–McCluskey minimisation producing conservative, parsimonious, and intermediate solutions with a core/periphery classification
- **Both outcome directions by default**: `run_all` analyses the outcome and its negation (`~Y`), as fsQCA practice requires
- **Publication-ready artefacts**: tidy CSV tables for every stage, four figures (importance, model comparison, and one configuration chart per outcome direction), and a consolidated `report.md`
- **Built-in verification**: `mixtriad selfcheck` validates the installed copy against an exhaustive Quine–McCluskey oracle and calibration invariants in under a second
- **Inter-coder reliability**: `krippendorff_alpha` for hand-coded ordinal antecedents

### 1.3 Technical Architecture

| Layer | Technology |
|---|---|
| Language | Python ≥ 3.9 |
| Numerical core | NumPy |
| Data handling | pandas (+ tabulate for the Markdown report) |
| Statistical models | statsmodels (Stage 1), scikit-learn (Stage 2) |
| Optional accelerators | xgboost, optuna, shap (`pip install -e ".[all]"`) |
| fsQCA engine | Pure Python/NumPy, no external QCA dependency |
| Figures | matplotlib (Agg backend; safe on headless servers) |
| Packaging | setuptools / PEP 621 (`pyproject.toml`) |
| Continuous integration | GitHub Actions — ruff lint, integrity manifest, Python 3.9–3.13 test matrix, full-stack job, end-to-end demo run |
| Integrity | `MANIFEST.sha256` covering every distributed file |
| License | MIT |

### 1.4 Two Ways to Run an Analysis

|  | Intended for | How to run |
|---|---|---|
| **CLI + YAML config** | Reproducible, declarative runs; users who prefer not to write Python | `mixtriad run config.yaml` |
| **Python API** | Notebook exploration, custom stages, standalone fsQCA | `from mixtriad import Pipeline, Schema, FsqcaSpec` |

Both routes execute identical code paths and write identical artefacts. The YAML route is recommended for the analysis of record, because the config file *is* the complete, shareable description of the run.

---

## 2. System Requirements

### 2.1 Python Package

| Item | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.12 |
| Operating system | Linux, macOS, Windows | any (pure-Python package; CI runs Ubuntu) |
| Memory | 2 GB | 4 GB |
| Disk | 200 MB including dependencies | 500 MB with the optional stack |

**Required dependencies** (installed automatically):

| Package | Minimum version |
|---|---|
| numpy | 1.23 |
| pandas | 1.5 |
| scikit-learn | 1.2 |
| matplotlib | 3.6 |
| statsmodels | 0.14.5 |
| tabulate | 0.9 |
| pyyaml | 6.0 |

**Optional dependency groups:**

| Group | Packages | Needed for |
|---|---|---|
| `boost` | xgboost ≥ 1.7 | XGBoost as the tuned Stage-2 model |
| `tune` | optuna ≥ 3.0 | Bayesian (TPE) hyperparameter search |
| `shap` | shap ≥ 0.42 | optional SHAP analyses in user code |
| `all` | all three | recommended for the analysis of record |

Without the optional extras the pipeline **does not fail**: Stage 2 transparently substitutes scikit-learn's `HistGradientBoostingRegressor` tuned by random search, and the model-comparison table labels the tuned model accordingly (see §8.3 on the practical equivalence of the two paths).

Running the test suite additionally requires `pytest` and `hypothesis`. Reproducing the cross-validation against R (§8.2) requires R ≥ 4.3 with the `QCA` package; nothing in normal operation needs R.

### 2.2 Data Requirements

| Item | Requirement |
|---|---|
| Format | one CSV (or an in-memory `pandas.DataFrame`) |
| Unit of analysis | one row per case |
| Outcome | continuous or count, non-negative (modelled as `log1p`) |
| Antecedents | numeric. **Categorical antecedents must be numerically coded** (integer codes); string columns raise an error in Stage 2 |
| Missing data | rows with missing values in any model column are dropped listwise per stage |
| Sample size | fsQCA convention: intermediate-N designs; the demo corpus has 388 cases |

---

## 3. Installation

### 3.1 From Source

```bash
git clone https://github.com/YanjinLyu/mixtriad
cd mixtriad
pip install -e .            # core (numpy, pandas, scikit-learn, statsmodels, matplotlib, pyyaml, tabulate)
pip install -e ".[all]"     # + xgboost, optuna, shap (recommended)
```

### 3.2 Verify the Installation

```bash
python -c "import mixtriad; print(mixtriad.__version__)"
# 1.1.2

mixtriad selfcheck
# [PASS] QM differential oracle (exhaustive k=3)  (255 cases, 0 failures)
# [PASS] calibration anchors / monotonicity / crossover invariant  (403 cases, 0 failures)
# [PASS] set-measure bounds on random data  (150 cases, 0 failures)
# [PASS] micro end-to-end (Stage 1 + Stage 3)  (1 cases, 0 failures)
# selfcheck PASSED in 0.06 s (quick mode)
```

`mixtriad selfcheck --full` additionally runs a Stage-2 boosting pass (~2 s). The command exits 0 on pass and 1 on failure, so it can gate a script or CI job. The distributed file set can be verified independently with `sha256sum -c MANIFEST.sha256` from the repository root.

### 3.3 Run the Shipped Demo

```bash
python examples/make_dataset.py                  # regenerates the 388-row synthetic corpus (run from the repository root)
mixtriad run examples/misinfo_config.yaml        # full three-stage pipeline (~1–3 min)
```

Artefacts appear in `examples/demo_out/`. If `report.md` and the four PNG figures are produced, the installation is complete. (The corpus CSV also ships pre-generated, so the first command is optional.)

---

## 4. Functional Modules

| Module | Purpose |
|---|---|
| `mixtriad.data` | `Schema` (variable roles), transformations, `krippendorff_alpha` |
| `mixtriad.regression` | Stage 1: OLS, count-data robustness ladder, VIF |
| `mixtriad.boosting` | Stage 2: tuned boosting, permutation importance, elbow screen |
| `mixtriad.fsqca` | Stage 3: calibration, necessity, truth table, Quine–McCluskey |
| `mixtriad.pipeline` | `Pipeline` orchestration, figures, `report.md` |
| `mixtriad.cli` | `mixtriad run` / `mixtriad selfcheck` entry points |
| `mixtriad.selfcheck` | Built-in differential oracle and invariant checks |

### 4.1 `mixtriad.data` — Schema and Reliability

- **`Schema`** — declares the variable roles once, for the whole pipeline: `outcome`, `antecedents`, `controls`, `dimensions` (an optional variable → theoretical-dimension map used for reporting), `log_transform` (variables replaced by `log1p` before modelling), `categorical` (dummy-coded in Stage 1).
- **`krippendorff_alpha`** — inter-coder reliability (interval-data α) for an `(n_items, n_coders)` array with `NaN` marking missing codes. Use it to report agreement on hand-coded antecedents before analysis.

### 4.2 `mixtriad.regression` — Stage 1

- **`run_regressions`** — fits OLS on `log1p(outcome)` with HC1 robust standard errors and dummy-coded categoricals, and a count-data robustness model on the raw outcome. The count model follows a documented ladder: a Poisson GLM supplies fitted means and warm-start parameters; the NB2 dispersion α is estimated by the Cameron–Trivedi auxiliary regression; the NB2 GLM is accepted only if IRLS converges; otherwise the Poisson QMLE with HC1 (sandwich) errors — the standard robust alternative under overdispersion — is reported. Which rung was reached is recorded in the returned table's metadata, never silently hidden. A VIF table screens collinearity.

### 4.3 `mixtriad.boosting` — Stage 2

- **`run_boosting`** — the leakage-safe protocol: for each of five fixed seeds, an 80/20 hold-out split; hyperparameters tuned **on the training fold only** (Optuna TPE over XGBoost when available, otherwise random search over `HistGradientBoostingRegressor`, each scored by 5-fold CV RMSE inside the training fold); RMSE/MAE/R² measured on the untouched test fold and averaged across seeds. Three untuned baselines (OLS, default random forest, default gradient boosting) anchor the comparison. Permutation importance (20 repeats, ΔRMSE) is computed on the held-out fold and averaged.
- **Elbow screen** — importances are sorted and the largest gap (at or beyond `min_retain`) sets a cut-off; antecedents above it become the Stage-3 condition set. The cut-off can be overridden numerically.

### 4.4 `mixtriad.fsqca` — Stage 3

- **`direct_calibrate`** — Ragin's direct method: a log-odds mapping anchored at full non-membership (0.05), crossover (0.50), and full membership (0.95). Cases landing exactly on the crossover are lifted by one representable float so they do not drop out of the truth table, preserving exact monotonicity.
- **`percentile_anchors`** — convenience anchors from the 10th/50th/90th percentiles, with a guarded nudge for degenerate (sparse or discrete) distributions.
- **`consistency` / `coverage` / `pri`** — the standard set-theoretic measures: Σmin(X,Y)/ΣX, Σmin(X,Y)/ΣY, and Proportional Reduction in Inconsistency.
- **`necessity_table`** — necessity consistency and coverage for every condition **and its negation**.
- **`build_truth_table`** — assigns each case to its best-fitting corner of the property space, scores each observed configuration for sufficiency, and applies the consistency, PRI, and frequency cut-offs. Configurations below the frequency cut-off are treated as remainders — the standard treatment.
- **`quine_mccluskey`** — Boolean minimisation with don't-cares. Essential prime implicants are extracted first; the residual cover is found by exhaustive search when the candidate set is small (≤ 18 primes), guaranteeing a minimum cover, with a greedy fallback beyond that.
- **`minimize`** — derives all three solution types in one call: **conservative** (no remainders), **parsimonious** (all remainders as don't-cares), and **intermediate** (easy counterfactuals only — remainders consistent with every stated directional expectation). Each recipe is scored for consistency, raw coverage, and unique coverage.
- **`core_periphery`** — classifies each literal of the intermediate solution as *core* (also present in the parsimonious solution) or *peripheral*, the basis of the configuration chart.

### 4.5 `mixtriad.pipeline` — Orchestration

- **`Pipeline`** — holds the data, schema, and output directory; runs any stage individually or all of them via `run_all`, which analyses the outcome and its negation and writes `report.md`. Every table is written as a tidy CSV the moment its stage completes, so a partially run pipeline still leaves usable artefacts.
- **`FsqcaSpec`** — the Stage-3 configuration object: per-condition calibration anchors, directional expectations, and the three truth-table cut-offs.

---

## 5. API Reference

All public names are importable from the top level: `Schema`, `krippendorff_alpha`, `run_regressions`, `run_boosting`, `BoostResult`, `Pipeline`, `FsqcaSpec`, and the `fsqca` module.

### 5.1 Schema and Reliability

```python
Schema(outcome, antecedents, controls=[], dimensions={},
       log_transform=[], categorical=[])
```

| Field | Type | Description |
|---|---|---|
| `outcome` | str | Column name of the outcome variable |
| `antecedents` | list[str] | Theorised explanatory conditions |
| `controls` | list[str] | Covariates included in Stages 1–2 but never carried into fsQCA |
| `dimensions` | dict[str, str] | Optional variable → theoretical-dimension map (documentation only) |
| `log_transform` | list[str] | Variables replaced by `log1p(x)` (as `log_<name>`) before modelling |
| `categorical` | list[str] | Dummy-coded (drop-first) in Stage 1. Must be **numerically coded**; used as ordinal numeric in Stage 2 |

`Schema.validate(df)` raises `KeyError` listing any declared column missing from the data.

```python
krippendorff_alpha(codes)   # -> float
```

`codes` is an `(n_items, n_coders)` array; `NaN` marks missing codes. Returns interval-data α (1.0 = perfect agreement).

### 5.2 Stage 1 — Regression

```python
run_regressions(df, schema)   # -> dict[str, DataFrame]
```

Returns `{"ols": ..., "negbin": ..., "vif": ...}`. The `ols` and `negbin` tables share tidy columns `model`, `term`, `estimate`, `std_error`, `p_value`, `sig` (`***` < .001, `**` < .01, `*` < .05). `ols.attrs["r2"]` holds the R²; `negbin.attrs` records which rung of the robustness ladder was fitted (NB2 with the estimated α, or Poisson QMLE + HC1). The `vif` table lists one variance inflation factor per design column.

### 5.3 Stage 2 — Boosting

```python
run_boosting(df, schema, seeds=(42, 123, 456, 789, 1024),
             n_trials=50, importance_cutoff=None, min_retain=3)
# -> BoostResult
```

| Parameter | Description |
|---|---|
| `seeds` | One repetition of the 80/20 hold-out per seed |
| `n_trials` | Optuna TPE trials per repetition (fallback: `max(10, n_trials // 5)` random-search draws) |
| `importance_cutoff` | Override the elbow cut-off with a fixed value |
| `min_retain` | The elbow is searched at or beyond this rank, so at least this many antecedents are normally retained |

**`BoostResult`** fields:

| Attribute | Description |
|---|---|
| `.metrics` | Model-comparison panel: mean `RMSE`, `MAE`, `R2` per model across seeds (log-outcome scale) |
| `.importance` | Mean permutation importance per feature, with a `role` column (`antecedent`/`control`) |
| `.retained` | Antecedents above the cut-off, in descending importance — the default Stage-3 condition set |
| `.cutoff` | The elbow (or overridden) importance cut-off |

### 5.4 Stage 3 — fsQCA Functions

```python
fsqca.direct_calibrate(x, full_non, crossover, full_mem)   # -> Series in [0, 1]
```

Anchors must satisfy `full_non < crossover < full_mem` (`ValueError` otherwise). Membership is 0.05 / 0.50 / 0.95 at the three anchors; cases exactly at the crossover are lifted by one representable float.

```python
fsqca.percentile_anchors(x, pcts=(0.10, 0.50, 0.90))   # -> (non, cross, full)
fsqca.consistency(cond, outc)    # sum(min(X, Y)) / sum(X)
fsqca.coverage(cond, outc)       # sum(min(X, Y)) / sum(Y)
fsqca.pri(cond, outc)            # proportional reduction in inconsistency
fsqca.necessity_table(memberships, outcome)   # per condition and negation
```

```python
fsqca.build_truth_table(memberships, outcome, consistency_cutoff=0.80,
                        pri_cutoff=0.70, frequency_cutoff=1)   # -> TruthTable
```

**`TruthTable`**: `.conditions`, `.table` (one row per retained configuration: condition bits, `n_cases`, `raw_consistency`, `PRI`, `outcome` ∈ {0, 1}), `.positive_rows`, `.negative_rows`, `.remainder_rows` (unobserved corners plus configurations dropped by the frequency cut-off).

```python
fsqca.quine_mccluskey(minterms, dont_cares)   # -> list of implicants ('-' = eliminated)
fsqca.minimize(tt, memberships, outcome, directional=None)
# -> {"conservative": Solution, "parsimonious": Solution, "intermediate": Solution}
```

`directional` maps a condition to its expected contribution (1 = presence, 0 = absence). Remainders contradicting any expectation are withheld from the intermediate minimisation (an "easy counterfactuals only" policy). **With no directional expectations the intermediate solution equals the conservative solution.**

**`Solution`**: `.kind`, `.recipes`, `.solution_consistency`, `.solution_coverage`, `.to_frame()` (tidy table; solution-level statistics in `.attrs`). **`Recipe`**: `.literals` (condition → 1/0), `.consistency`, `.raw_coverage`, `.unique_coverage`, `.label()` (e.g. `creator_reach * ~video_duration`).

```python
fsqca.core_periphery(parsimonious, intermediate)
# -> DataFrame: configuration | condition | state (present/absent) | role (core/peripheral)
```

### 5.5 Pipeline Orchestration

```python
FsqcaSpec(anchors=None, directional=None, consistency_cutoff=0.80,
          pri_cutoff=0.70, frequency_cutoff=1, negate_outcome=False)
```

| Field | Description |
|---|---|
| `anchors` | dict: condition → `(full_non, crossover, full_mem)`. Conditions not listed use 10/50/90 percentile anchors. The **outcome** is always calibrated on `log1p` with percentile anchors (not overridable in 1.1.2) |
| `directional` | dict: condition → 1/0 expected direction, for the intermediate solution |
| `consistency_cutoff` / `pri_cutoff` / `frequency_cutoff` | Truth-table thresholds (defaults 0.80 / 0.70 / 1) |
| `negate_outcome` | Analyse `~Y` instead of `Y` (`run_all` does both automatically) |

```python
Pipeline(df, schema, outdir="mixtriad_out")

p.stage1_regression()                       # -> dict of tidy tables; writes stage1_*.csv
p.stage2_boosting(seeds=..., n_trials=50,
                  importance_cutoff=None)   # -> BoostResult; writes stage2_*.csv + 2 figures
p.stage3_fsqca(spec, conditions=None)       # -> dict; writes stage3_*_{outcome|negated}.* + figure
p.run_all(spec, n_trials=50)                # all stages, both outcome directions, report.md
```

`stage3_fsqca` defaults its condition set to `boost.retained` when Stage 2 has run, otherwise to all antecedents; pass `conditions=` to override either. Results stay accessible on the object: `p.reg["ols"]`, `p.boost.metrics`, `p.boost.retained`, `p.qca["outcome"]["solutions"]["intermediate"].to_frame()`.

---

## 6. Operation Guide

### 6.1 Quick Start (CLI)

```bash
mixtriad run examples/misinfo_config.yaml
mixtriad run my_config.yaml --outdir results --trials 100
```

`--outdir` and `--trials` override the corresponding config keys.

### 6.2 YAML Configuration Reference

```yaml
data: path/to/cases.csv          # required
outcome: engagement              # required
antecedents: [a, b, c]           # required
controls: [publication_age]      # optional
log_transform: [a]               # optional; heavy-tailed variables
categorical: [c]                 # optional; numerically coded
dimensions: {a: actor, b: content}   # optional; documentation only
fsqca:                           # optional block
  consistency_cutoff: 0.80
  pri_cutoff: 0.70
  frequency_cutoff: 2
  anchors:                       # theory-based calibration overrides
    a: [10, 50, 200]             # (full_non, crossover, full_mem)
  directional:                   # expectations for the intermediate solution
    a: 1
    b: 0
n_trials: 50                     # Optuna trials per repetition
outdir: results
```

### 6.3 Full Run (Python API)

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

p.reg["ols"]                       # Stage 1 tidy table
p.boost.metrics                    # Stage 2 comparison panel
p.boost.retained                   # antecedents above the elbow
p.qca["outcome"]["solutions"]["intermediate"].to_frame()   # Stage 3 recipes
```

### 6.4 Stage-by-Stage Runs

Each stage is independent — useful in notebooks:

```python
p = Pipeline(df, schema, outdir="out")
p.stage1_regression()
p.stage2_boosting(n_trials=100)
print(p.boost.retained)                       # inspect before committing to Stage 3
p.stage3_fsqca(spec)                          # outcome
p.stage3_fsqca(FsqcaSpec(directional=spec.directional, negate_outcome=True))
```

### 6.5 Theory-Based Calibration Anchors

Percentile anchors are a default, not a doctrine. When theory or measurement supplies substantive thresholds, state them:

```python
spec = FsqcaSpec(anchors={
    "creator_reach": (1_000, 50_000, 1_000_000),   # followers
    "emotional_intensity": (1.5, 3.0, 4.5),        # 5-point coder scale
})
```

Check `stage3_anchors_outcome.json` after every run: any condition flagged `"degenerate_percentiles": true` (sparse or discrete distributions) should be given explicit anchors.

### 6.6 Directional Expectations

```python
spec = FsqcaSpec(directional={"creator_reach": 1, "hashtag_count": 0})
```

State an expectation only where theory genuinely licenses one; leave the rest out. Remainders contradicting any stated expectation are withheld from the intermediate minimisation. With no expectations at all, the intermediate solution simply equals the conservative one — report it as such.

### 6.7 Analysing the Absence of the Outcome

fsQCA is asymmetric: recipes for `~Y` are not the mirror image of recipes for `Y`. `run_all` analyses both automatically (`*_outcome.*` and `*_negated.*` artefacts). For a single direction, set `FsqcaSpec(negate_outcome=True)`.

### 6.8 Standalone fsQCA on Your Own Memberships

The Stage-3 engine has no dependency on Stages 1–2:

```python
import pandas as pd
from mixtriad import fsqca

memb = pd.DataFrame({
    "A": fsqca.direct_calibrate(raw["A"], 10, 50, 200),
    "B": fsqca.direct_calibrate(raw["B"], 0.2, 0.5, 0.8),
})
y = fsqca.direct_calibrate(raw["Y"], 100, 1_000, 50_000)

nec = fsqca.necessity_table(memb, y)
tt = fsqca.build_truth_table(memb, y, consistency_cutoff=0.85, frequency_cutoff=2)
sols = fsqca.minimize(tt, memb, y, directional={"A": 1})
print(sols["intermediate"].to_frame())
```

### 6.9 Inter-Coder Reliability

```python
import numpy as np
from mixtriad import krippendorff_alpha

codes = np.array([[4, 4, 5],
                  [2, 3, 2],
                  [5, 5, np.nan]])    # items x coders; NaN = missing
alpha = krippendorff_alpha(codes)
```

Report α per hand-coded antecedent before analysis; values ≥ 0.80 are conventionally acceptable for interval-level codes.

### 6.10 Example Scripts

| Script | Purpose |
|---|---|
| `examples/make_dataset.py` | Regenerates the 388-row synthetic demo corpus (run from the repository root) |
| `examples/misinfo_config.yaml` | The demo configuration — a complete, annotated template |
| `examples/fakenewsnet_pilot.py` | A pilot on real public data (requires network access) |

---

## 7. Output Artefacts and Interpretation

A full `run_all` writes 24 files to `outdir`. Suffix `_outcome` refers to the analysis of `Y`; `_negated` to `~Y`.

### 7.1 File Inventory

| File | Content |
|---|---|
| `stage1_ols.csv` | OLS(log outcome) tidy coefficients, HC1 errors, significance stars |
| `stage1_negbin.csv` | Count-model coefficients; the model actually fitted (NB2 or Poisson QMLE) is named in the `model` column |
| `stage1_vif.csv` | Variance inflation factors |
| `stage2_model_comparison.csv` | Mean out-of-sample RMSE / MAE / R² per model |
| `stage2_importance.csv` | Mean permutation importance per feature, with role |
| `fig_importance.png` | Importance bars with the elbow cut-off line |
| `fig_model_comparison.png` | RMSE and R² per model |
| `stage3_anchors_*.json` | Anchors used per condition, with degeneracy warnings |
| `stage3_necessity_*.csv` | Necessity consistency/coverage, conditions and negations |
| `stage3_truthtable_*.csv` | Retained configurations: bits, `n_cases`, `raw_consistency`, `PRI`, `outcome` |
| `stage3_solution_conservative_*.csv` | Conservative-solution recipes |
| `stage3_solution_parsimonious_*.csv` | Parsimonious-solution recipes |
| `stage3_solution_intermediate_*.csv` | Intermediate-solution recipes — normally the headline result |
| `stage3_chart_*.csv` | Core/periphery classification underlying the configuration chart |
| `fig_configurations_*.png` | Configuration chart: ● present, ○ absent; large = core, small = peripheral |
| `report.md` | Consolidated Markdown report across all stages |

### 7.2 Reading the Results

- **Convergence is the point.** The triangulation logic: an antecedent that is significant in Stage 1, retained by the Stage 2 elbow, *and* appears as a core condition in Stage 3 is supported by three methodologically independent lenses. Divergence is equally informative — e.g. a variable with no net effect that appears in one configuration signals a conjunctural, not additive, role.
- **Solution statistics.** Report the intermediate solution with its consistency and coverage (in `to_frame().attrs` and `report.md`); conventional floors are consistency ≥ 0.80 and PRI ≥ 0.70 (the defaults). `raw_coverage` is a recipe's empirical breadth; `unique_coverage` the share of the outcome only that recipe explains.
- **Necessity before sufficiency.** A condition with necessity consistency ≥ 0.90 and non-trivial coverage should be discussed as a candidate necessary condition before the sufficiency analysis.
- **Condition order.** Stage-3 columns and recipe literals follow the Stage-2 importance ranking. Two installations (with vs without the optional stack) can rank two near-tied conditions differently, changing display order while every statistic remains identical — see §9.4.

---

## 8. Validation and Accuracy

### 8.1 Built-in Verification (`mixtriad selfcheck`)

Every installation carries its own oracle, runnable in under a second:

| Check | Method | Cases |
|---|---|---|
| Quine–McCluskey differential oracle | Exhaustive k = 3: every non-empty minterm set (255) minimised and compared against a brute-force minimum cover | 255 |
| Calibration invariants | Anchors map to 0.05/0.50/0.95; strict monotonicity; crossover handling | 403 |
| Set-measure bounds | consistency/coverage/PRI ∈ [0, 1] on random data | 150 |
| Micro end-to-end | Stage 1 + Stage 3 on a generated micro-dataset (`--full` adds Stage 2) | 1 |

### 8.2 Cross-Validation Against R (`QCA` 3.25)

The reference implementation for fsQCA is R's `QCA` package. The shipped evidence (`verify/round9/`, produced under R 4.3.3, QCA 3.25, admisc 0.40) compares MixTriad on the demo corpus and on Lipset's classic dataset:

| Quantity | Agreement |
|---|---|
| Direct calibration (63 spot values) | max abs. difference 6.7 × 10⁻¹⁶ (float precision) |
| Truth table (demo, 4 conditions) | 16 of 16 rows matched; identical case counts; consistency/PRI within 4.7 × 10⁻⁴ of R's printed values |
| Conservative solution | identical recipes |
| Parsimonious solution | identical recipes |
| Solution consistency / coverage | identical at 3 decimals (0.908 / 0.530) |
| Parameters of fit, 4 recipes | max difference 3.9 × 10⁻⁴ (R prints 3 decimals) |
| Lipset benchmark | all observed truth-table rows and both solutions matched |

The comparison script (`verify/round9/crosscheck.R`) is shipped and re-runnable.

### 8.3 The Stage-2 Fallback Is Substantively Equivalent

With the optional stack absent, the tuned model is `HistGradientBoostingRegressor` under random search rather than Optuna-tuned XGBoost. On the demo corpus the two paths retain the **same** antecedent set with **identical** downstream fsQCA statistics; only near-tied importance ranks (hence display order) can differ. Absolute Stage-2 metrics differ modestly (demo: tuned RMSE 0.93 with XGBoost vs 1.01 with the fallback). For the analysis of record, install `".[all]"` and report which path ran — the `model` column of `stage2_model_comparison.csv` names it explicitly.

### 8.4 Continuous Integration

Every commit runs: ruff lint; `sha256sum -c MANIFEST.sha256` (file-integrity gate); the pytest suite (35 tests, including Hypothesis property tests) plus `selfcheck` on Python 3.9, 3.10, 3.11, 3.12 and 3.13 with core dependencies and once on 3.12 with the full optional stack; and an end-to-end demo job that executes `mixtriad run` on the shipped corpus, asserts seven key artefacts are non-empty, and uploads the outputs as inspectable CI artefacts.

### 8.5 Scope and Limitations

- **Categorical antecedents** must be numerically coded; string columns raise in Stage 2. Stage 1 dummy-codes them; Stage 2 treats the codes as ordinal numeric — acceptable for tree models, but interpret importances of nominal variables with care.
- **Nominal variables do not belong in fsQCA.** Percentile calibration of an unordered code is not meaningful. If a nominal antecedent survives the Stage-2 screen, exclude it from Stage 3 explicitly via `stage3_fsqca(spec, conditions=[...])`.
- **The outcome calibration is fixed** in 1.1.2: `log1p` with 10/50/90 percentile anchors. Condition anchors are fully overridable; outcome anchors are not.
- **No directional expectations ⇒ intermediate = conservative.** This is by construction (§5.4); state it when reporting.
- **Minimisation is exact** for candidate prime sets up to 18; beyond that a greedy cover (always valid, possibly non-minimal) is used. Four-to-eight-condition analyses — the fsQCA norm — are always exact.
- Stage-1 and Stage-2 estimates are observational associations under the usual caveats; MixTriad triangulates evidence, it does not identify causal effects.

---

## 9. Troubleshooting

### 9.1 `ModuleNotFoundError: No module named 'tabulate'` at Report Time

**Cause:** an installation predating v1.1.2, where `tabulate` (required by `DataFrame.to_markdown`) was not yet a declared dependency.

**Fix:** upgrade to ≥ 1.1.2, or `pip install tabulate`.

### 9.2 Stage 2 Raises `could not convert string to float`

**Cause:** a categorical antecedent stored as text. Stage 2 requires numeric input.

**Fix:** integer-code the column before loading (e.g. `df["topic"] = df["topic"].map({"news": 0, "sports": 1, "meme": 2})`), declare it in `Schema.categorical`, and keep it out of Stage 3 (§8.5).

### 9.3 `stage3_anchors_*.json` Flags `degenerate_percentiles`

**Cause:** the condition's 10th/50th/90th percentiles are not distinct (sparse or discrete variable), so automatic anchors are unreliable; zero-mass cases sit at the crossover and are classified "present".

**Fix:** supply theory-based anchors for that condition via `FsqcaSpec(anchors={...})` or the YAML `fsqca.anchors` block.

### 9.4 Recipe or Column Order Differs Between Machines

**Expected** when one machine has the optional stack and the other uses the fallback: near-tied permutation importances can swap ranks, and Stage-3 output order follows that ranking. Verify that the *set* of retained conditions and every consistency/coverage value match — on the shipped demo they do, exactly.

### 9.5 `make_dataset.py` Raises `FileNotFoundError`

**Cause:** the script writes via a repository-relative path.

**Fix:** run it from the repository root: `python examples/make_dataset.py`.

### 9.6 `--trials 0` Is Silently Ignored

**Cause:** the CLI treats 0 as unset and falls back to the config value.

**Fix:** set `n_trials: 1` (or any positive value) in the YAML instead; a zero-trial run is never meaningful.

### 9.7 The Count Model Is Not NB2

**Cause:** NB2 IRLS did not converge on your data, so the ladder reported the Poisson QMLE with HC1 sandwich errors instead (§4.2).

**Fix:** nothing is wrong — this is the documented robust alternative under overdispersion. The `model` column and the table's `attrs` metadata record the choice; report it as fitted.

### 9.8 The Configuration Chart Is Missing or Empty

**Cause:** no configuration passed the consistency/PRI/frequency cut-offs for that outcome direction, so the solution is empty.

**Fix:** inspect `stage3_truthtable_*.csv`. Consider whether the cut-offs are too strict for your N, whether calibration anchors are sensible (§9.3), or whether the outcome direction simply has no consistent recipe — itself a reportable finding.

### 9.9 `sha256sum -c MANIFEST.sha256` Fails After Editing Files

**Cause:** the manifest pins every distributed file; any edit invalidates its entry.

**Fix:** regenerate after intentional edits, from the repository root:

```bash
find . -type f ! -path "./.git/*" ! -name MANIFEST.sha256 \
  | sed "s|^\./||" | LC_ALL=C sort \
  | while read -r f; do sha256sum "$f"; done > MANIFEST.sha256
```

### 9.10 `mixtriad selfcheck` Fails

A selfcheck failure means the installed copy does not reproduce the built-in oracle — typically a corrupted or partial installation. Reinstall from a verified archive (`sha256sum -c MANIFEST.sha256` first); if the failure persists on a clean install, report it on the issue tracker with the printed check name and your Python/NumPy versions.

---

## 10. Support and Version Information

### 10.1 Contact

- **Repository:** https://github.com/YanjinLyu/mixtriad
- **Issue tracker:** https://github.com/YanjinLyu/mixtriad/issues
- **Archived release:** https://doi.org/10.5281/zenodo.22029183

### 10.2 Version

- **Current version:** 1.1.2
- **Released:** 2026-08-20
- **License:** MIT

### 10.3 Author

| Name | Affiliation |
|---|---|
| Yanjin Lyu (maintainer, contact) | Hebei Education Press Co., Ltd.; Department of Music, University of Sheffield |

### 10.4 Citation

If you use MixTriad in published work, please cite the accompanying SoftwareX article and the archived software release (DOI above). Machine-readable metadata is in `CITATION.cff`; GitHub's "Cite this repository" button renders it directly.

### 10.5 Testing

```bash
python -m pytest tests -q       # 35 tests, incl. Hypothesis property tests
mixtriad selfcheck --full       # built-in oracle, with a Stage-2 pass
```

Continuous integration runs both on every commit across Python 3.9–3.13, verifies the integrity manifest, and executes the full demo pipeline end-to-end (§8.4).

---

## 11. Appendix

### A. Complete Public API

**Top level:** `Schema`, `krippendorff_alpha`, `run_regressions`, `run_boosting`, `BoostResult`, `Pipeline`, `FsqcaSpec`, `fsqca`, `__version__`

**`fsqca` module:** `direct_calibrate`, `percentile_anchors`, `consistency`, `coverage`, `pri`, `necessity_table`, `build_truth_table`, `TruthTable`, `quine_mccluskey`, `minimize`, `Solution`, `Recipe`, `core_periphery`

### B. Output Column Glossary

| Column | Meaning |
|---|---|
| `estimate`, `std_error`, `p_value`, `sig` | Tidy regression output; HC1 robust errors; stars at .05/.01/.001 |
| `RMSE`, `MAE`, `R2` | Out-of-sample metrics on the log-outcome scale, averaged over five hold-outs |
| `permutation_importance` | Mean ΔRMSE on held-out folds when the feature is permuted (20 repeats) |
| `role` | `antecedent` (eligible for Stage 3) or `control` |
| `n_cases` | Cases whose best-fitting corner is this configuration |
| `raw_consistency` | Sufficiency consistency of the configuration, Σmin(X,Y)/ΣX |
| `PRI` | Proportional Reduction in Inconsistency; guards against simultaneous subset relations with Y and ~Y |
| `outcome` | Truth-table coding: 1 = sufficient for the outcome, 0 = not |
| `necessity_consistency` / `necessity_coverage` | Necessity measures per condition (`~x` rows are negations) |
| `recipe` | Boolean conjunction; `*` = AND, `~` = negation |
| `consistency`, `raw_coverage`, `unique_coverage` | Recipe-level parameters of fit |
| `state`, `role` (chart) | `present`/`absent`; `core` (in the parsimonious solution too) or `peripheral` |

### C. Repository Layout

```
mixtriad/
├── mixtriad/                   # package source
│   ├── __init__.py             # public API
│   ├── data.py                 # Schema, transformations, krippendorff_alpha
│   ├── regression.py           # Stage 1
│   ├── boosting.py             # Stage 2
│   ├── fsqca.py                # Stage 3 engine
│   ├── pipeline.py             # orchestration, figures, report
│   ├── cli.py                  # mixtriad run / selfcheck
│   └── selfcheck.py            # built-in oracle
├── examples/                   # dataset generator, YAML config, demo output, real-data pilot
├── tests/                      # pytest suite (35 tests)
├── verify/                     # validation evidence, incl. the R QCA cross-check (round9)
├── docs/                       # architecture figure source
├── .github/workflows/ci.yml    # CI: lint, manifest, test matrix, e2e demo
├── pyproject.toml              # PEP 621 metadata
├── CITATION.cff  ·  .zenodo.json  ·  CHANGELOG.md  ·  RELEASE_CHECKLIST.md
├── MANIFEST.sha256             # integrity manifest for every distributed file
├── LICENSE                     # MIT
└── README.md
```

### D. YAML Key Quick Reference

| Key | Required | Default | Maps to |
|---|---|---|---|
| `data` | yes | — | CSV path |
| `outcome`, `antecedents` | yes | — | `Schema` |
| `controls`, `log_transform`, `categorical`, `dimensions` | no | empty | `Schema` |
| `fsqca.anchors` | no | 10/50/90 percentiles | `FsqcaSpec.anchors` |
| `fsqca.directional` | no | none | `FsqcaSpec.directional` |
| `fsqca.consistency_cutoff` | no | 0.80 | `FsqcaSpec` |
| `fsqca.pri_cutoff` | no | 0.70 | `FsqcaSpec` |
| `fsqca.frequency_cutoff` | no | 1 | `FsqcaSpec` |
| `n_trials` | no | 50 | tuning trials per repetition |
| `outdir` | no | `mixtriad_out` | output directory |

---

*MixTriad 1.1.2 · MIT License · Copyright © 2026 Yanjin Lyu · https://doi.org/10.5281/zenodo.22029183*
