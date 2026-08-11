"""Shared loading + reporting for the per-study analyses.

Each study in this directory is standalone: one question, one file, one CSV of
results, and a printed paragraph ready to paste into the manuscript. They all
read the committed source files, so they run offline in seconds and rerun
identically.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "studies"
USABLE = ("include", "include-valid-rows-only")
RNG = np.random.default_rng(42)


def load(kind: str, usable_only: bool = True) -> pd.DataFrame:
    d = pd.read_csv(REPO / "results" / f"{kind}_per_sample.csv.gz", low_memory=False)
    if usable_only:
        d = d[d.analysis_decision.isin(USABLE)]
    return d.copy()


def load_raw(rel: str) -> pd.DataFrame:
    """A raw spreadsheet from the pull, for columns not in the source files."""
    return pd.read_excel(Path.home() / "Downloads/hpc_results" / rel)


def bongard_valid(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d.pred_valid].copy()
    d["correct"] = d.is_correct.astype(float)
    return d


def boot_ci(values, clusters=None, n_boot: int = 2000, alpha: float = 0.05):
    """Percentile CI; resamples whole clusters when given."""
    v = np.asarray(values, dtype=float)
    keep = ~np.isnan(v); v = v[keep]
    if not len(v):
        return np.nan, (np.nan, np.nan)
    if clusters is None:
        boots = [RNG.choice(v, len(v), replace=True).mean() for _ in range(n_boot)]
    else:
        c = np.asarray(clusters)[keep]
        groups = {k: v[c == k] for k in np.unique(c)}
        keys = list(groups)
        boots = [np.concatenate([groups[k] for k in RNG.choice(keys, len(keys), replace=True)]).mean()
                 for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), (float(lo), float(hi))


def save(df: pd.DataFrame, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    df.to_csv(p, index=False)
    return p


def para(title: str, text: str) -> None:
    """Print a manuscript-ready paragraph."""
    print(f"\n{'-' * 74}\nFOR THE PAPER — {title}\n{'-' * 74}")
    print(text.strip())
