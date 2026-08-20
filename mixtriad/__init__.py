"""MixTriad: three-stage mixed-methods triangulation for observational data.

Stage 1  parametric regression (OLS on log outcome + negative binomial)
Stage 2  Bayesian-optimised gradient boosting with permutation importance
Stage 3  fuzzy-set QCA (calibration, necessity, truth table, minimisation)
"""
from . import fsqca
from .boosting import BoostResult, run_boosting
from .data import Schema, krippendorff_alpha
from .pipeline import FsqcaSpec, Pipeline
from .regression import run_regressions

__version__ = "1.1.2"
__all__ = [
           "BoostResult",
           "FsqcaSpec",
           "Pipeline",
           "Schema",
           "__version__",
           "fsqca",
           "krippendorff_alpha",
           "run_boosting",
           "run_regressions",
]
