"""Stage 0: dataset schema, transformations, and coder-reliability utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class Schema:
    """Declares variable roles for the whole pipeline."""
    outcome: str
    antecedents: List[str]
    controls: List[str] = field(default_factory=list)
    dimensions: Dict[str, str] = field(default_factory=dict)   # var -> ecology dimension
    log_transform: List[str] = field(default_factory=list)     # log1p'd before modelling
    categorical: List[str] = field(default_factory=list)

    def validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in [self.outcome, *self.antecedents, *self.controls]
                   if c not in df.columns]
        if missing:
            raise KeyError(f"columns missing from data: {missing}")


def prepare(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Apply declared transformations; returns a modelling copy of the data."""
    schema.validate(df)
    out = df.copy()
    for c in schema.log_transform:
        out[f"log_{c}"] = np.log1p(pd.to_numeric(out[c], errors="coerce"))
    return out


def krippendorff_alpha(codes: np.ndarray) -> float:
    """Krippendorff's alpha for interval data.

    `codes` is an (n_items, n_coders) array; NaN marks missing codes.
    """
    codes = np.asarray(codes, dtype=float)
    vals = codes[~np.isnan(codes)]
    # observed disagreement
    Do_num, Do_den = 0.0, 0
    for row in codes:
        r = row[~np.isnan(row)]
        m = len(r)
        if m < 2:
            continue
        diffs = (r[:, None] - r[None, :]) ** 2
        Do_num += diffs.sum() / (m - 1)
        Do_den += m
    Do = Do_num / Do_den if Do_den else 0.0
    De = ((vals[:, None] - vals[None, :]) ** 2).sum() / (len(vals) * (len(vals) - 1))
    return float(1 - Do / De) if De > 0 else 1.0
