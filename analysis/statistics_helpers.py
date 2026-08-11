"""Shared statistical primitives for the analysis scripts.

Deliberately **not** named ``statistics``: that shadows the Python standard
library module of the same name, and which one you get depends on where the
interpreter was started (from ``analysis/`` you get the local file, from the repo
root you get the stdlib). ``analysis/statistics.py`` predates this module and
still carries that hazard; import from here instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)          # fixed seed: reproducible CIs
N_BOOT = 2000


def cluster_bootstrap_ci(values: np.ndarray, clusters: np.ndarray | None = None,
                         n_boot: int = N_BOOT, alpha: float = 0.05):
    """Mean and percentile CI. With ``clusters``, resample whole clusters.

    Bongard-OW queries come in positive/negative pairs from one support set, so
    the cluster is the ``uid``. Measured intra-cluster correlation on this data
    is ~0 (see results/stats/t1_icc.csv), so this rarely differs from a naive
    bootstrap — it is used because it is defensible, not because it changes
    conclusions.
    """
    values = np.asarray(values, dtype=float)
    keep = ~np.isnan(values)
    values = values[keep]
    if len(values) == 0:
        return float("nan"), (float("nan"), float("nan"))
    point = float(values.mean())
    if clusters is None:
        boots = [RNG.choice(values, len(values), replace=True).mean() for _ in range(n_boot)]
    else:
        clusters = np.asarray(clusters)[: len(keep)][keep]
        groups = {c: values[clusters == c] for c in np.unique(clusters)}
        keys = list(groups)
        boots = [np.concatenate([groups[k] for k in RNG.choice(keys, len(keys), replace=True)]).mean()
                 for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, (float(lo), float(hi))


def mcnemar_pair(a: pd.DataFrame, b: pd.DataFrame, metric: str, unit: str) -> dict:
    """Paired McNemar over the instances both conditions answered.

    ``b01`` = a wrong & b right, ``b10`` = a right & b wrong. Exact binomial when
    discordant pairs are few, chi-square with continuity correction otherwise.
    Rows where ``metric`` is NA are dropped, not scored — invalid model outputs
    are missing data, not errors.
    """
    from scipy.stats import binomtest, chi2

    am = a.dropna(subset=[metric]).set_index(unit)[metric]
    bm = b.dropna(subset=[metric]).set_index(unit)[metric]
    common = am.index.intersection(bm.index)
    am, bm = am.loc[common].astype(int), bm.loc[common].astype(int)
    b01 = int(((am == 0) & (bm == 1)).sum())
    b10 = int(((am == 1) & (bm == 0)).sum())
    disc = b01 + b10
    if disc == 0:
        stat, p = 0.0, 1.0
    elif disc < 25:
        p = float(binomtest(min(b01, b10), disc, 0.5).pvalue)
        stat = float("nan")
    else:
        stat = (abs(b01 - b10) - 1) ** 2 / disc
        p = float(chi2.sf(stat, 1))
    return dict(n_pairs=len(common), b01=b01, b10=b10, statistic=stat, pvalue=p,
                acc_a=round(100 * am.mean(), 2), acc_b=round(100 * bm.mean(), 2))


def holm(pvalues) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values, in the input order.

    Step-down: sort ascending, multiply the j-th by (m - j + 1), then take a
    RUNNING MAXIMUM so the sequence is non-decreasing, and cap at 1.

    The running maximum is the part that is easy to get wrong. An earlier
    implementation in analysis/statistics.py used ``np.minimum.accumulate``,
    which yields adjusted p-values *smaller* than Holm permits — i.e.
    anti-conservative, reporting more significant results than the correction
    allows. Verified against statsmodels.stats.multitest.multipletests.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m, dtype=float)
    out[order] = np.clip(adj_sorted, 0, 1)
    return out
