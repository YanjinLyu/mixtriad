# Changelog

## v1.1.2 (2026-08-20)
Metadata and packaging fixes; no runtime change.
- Author metadata corrected in pyproject.toml, CITATION.cff, .zenodo.json,
  LICENSE, and the README citation (name, e-mail, affiliation).
- Release archive cleaned: development artefacts are no longer shipped
  (.coverage, .ruff_cache/, build/ containing stale v1.0.0 sources,
  mixtriad.egg-info/, and the duplicate out/ directory).
- MANIFEST.sha256 regenerated last, over the cleaned tree, so
  `sha256sum -c MANIFEST.sha256` now passes with zero failures (resolves the
  round-10 acceptance-gate self-reference).

## v1.1.1 (2026-08-04)
Real-world public-data pilot (FakeNewsNet, round 8) drove three fixes:
- D10: empty fsQCA solutions now serialise with column headers instead of an
  unparseable one-byte CSV; refusals remain machine-readable.
- Degenerate-anchor diagnostic: when the 10/50/90 percentiles of a condition
  are not distinct (sparse/discrete variables), the anchors JSON now flags it
  and recommends theory-based anchors via FsqcaSpec(anchors=...).
- Count-model fallback label no longer shows a spurious "/1" rescale suffix.

Added
- examples/fakenewsnet_pilot.py: end-to-end pilot on the public FakeNewsNet
  corpus (downloaded at runtime; nothing redistributed). On PolitiFact the
  pipeline reproduces the published direction that false news spreads
  farther (is_fake +1.40 on log shares, p < 0.001) and correctly refuses
  sufficiency claims for shallow headline-only features.

## v1.1.0 (2026-08-03)
Added
- `mixtriad selfcheck` (CLI) / `mixtriad.run_selfcheck()` (API): installation
  self-verification productised from the project's QA programme - an
  exhaustive differential oracle for the Quine-McCluskey engine (all 255
  Boolean functions of three conditions: coverage, domain, primality, exact
  minimality), calibration anchor/monotonicity/crossover invariants,
  set-measure bounds, and a micro end-to-end pipeline (`--full` adds a
  Stage-2 boosting pass). Exit code 0 iff every check passes.

## v1.0.3 (2026-08-03)
Amended (round-6 meta-verification; no runtime change)
- +8 mutation-gap tests (suite now 28) closing every non-equivalent surviving
  mutant; final mutation score 26/37 raw, 26/26 on non-equivalent mutants.
- MANIFEST.sha256 added for release-integrity verification.

Fixed
- Boolean minimisation (D8): the greedy cover-completion step could return
  one implicant more than the exact minimum in ~0.6% of cases (differential
  oracle over 2,055 truth tables). Cover completion now performs an exact
  minimum-size search whenever the candidate prime set is small (always the
  practical fsQCA regime), with the greedy strategy retained as a fallback.
  Post-fix oracle: 2,055/2,055 exactly minimal.
- Calibration (D9, found by property-based testing): cases lying exactly at
  the crossover were lifted by +0.001, which could invert order against
  cases marginally above the crossover. The lift is now
  `np.nextafter(0.5, 1)`, keeping calibration exactly monotone while still
  keeping crossover cases inside the truth table.

Added
- `tests/test_properties.py`: hypothesis-based property suite (QM exact
  minimality vs a test-local brute-force oracle, cover/domain invariants,
  calibration monotonicity, measure bounds); suite now 20 tests.

## v1.0.2 (2026-08-03)
Fixed
- Packaging (D7): the declared `statsmodels>=0.14` floor admitted resolutions
  that break against current SciPy (`_lazywhere` removed in SciPy >=1.15,
  statsmodels <0.14.5 imports it). Floor raised to `statsmodels>=0.14.5`,
  empirically the lowest version importable against SciPy 1.17.
- Removed dead code flagged by static analysis (unused locals in
  `krippendorff_alpha` and `minimize`); no behavioural change.

Added
- `[tool.ruff]` configuration (E/F/W, line-length 100, E501 ignored with
  rationale); codebase lints clean.
- Release artefacts verified: sdist + wheel build, twine metadata check,
  wheel content audit, install-from-wheel smoke in a clean venv.

## v1.0.1 (2026-08-03)
Fixed
- Count-outcome robustness model (K1): replaced the fixed-alpha NB2 GLM,
  which emitted overflow warnings or failed on heavy-tailed counts, with a
  transparent fitting ladder - Poisson warm start, Cameron-Trivedi alpha
  estimation, NB2 accepted only on IRLS convergence, otherwise Poisson QMLE
  with HC1 sandwich errors. The chosen model, alpha_hat, and any fallback
  reason are recorded in the result metadata. Zero warnings under
  `warnings.simplefilter("error")`.

Added
- `tests/test_stress.py`: boundary/adversarial suite (missing columns, NaN
  rows, constant conditions, single-condition fsQCA, extreme-magnitude
  outcomes, calibration monotonicity, measure bounds, core/periphery
  containment).

## v1.0.0 (2026-08-03)
Initial release.
