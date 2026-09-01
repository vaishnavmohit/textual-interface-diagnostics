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


def cluster_randomization_pvalue(
    differences: np.ndarray,
    clusters: np.ndarray,
    n_resamples: int = 100_000,
    seed: int = 42,
) -> float:
    """Two-sided paired randomization test with signs exchanged by cluster.

    Under the paired sharp null, exchanging condition labels negates every
    within-cluster difference together.  This preserves the dependence between
    positive and negative Bongard queries that share a support set.  When every
    row is its own cluster (as for Winoground), the procedure reduces to the
    ordinary paired sign-flip randomization test.

    The reported Monte Carlo p-value uses the standard +1 correction and is
    therefore never zero.  A local RNG makes the result reproducible regardless
    of which other analyses ran first.
    """
    diff = np.asarray(differences, dtype=float)
    clu = np.asarray(clusters)
    keep = np.isfinite(diff)
    diff, clu = diff[keep], clu[keep]
    if not len(diff):
        return float("nan")

    cluster_sums = np.asarray([
        diff[clu == key].sum() for key in np.unique(clu)
    ], dtype=float)
    cluster_sums = cluster_sums[cluster_sums != 0]
    if not len(cluster_sums):
        return 1.0

    observed = abs(float(cluster_sums.sum()))
    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = int(n_resamples)
    while remaining:
        batch = min(5_000, remaining)
        signs = rng.integers(0, 2, size=(batch, len(cluster_sums)), dtype=np.int8)
        signs = signs * 2 - 1
        simulated = np.abs(signs @ cluster_sums)
        extreme += int((simulated >= observed - 1e-12).sum())
        remaining -= batch
    return float((extreme + 1) / (n_resamples + 1))


def paired_cluster_test(
    a: pd.DataFrame,
    b: pd.DataFrame,
    metric: str,
    unit: str,
    cluster: str | None,
    n_resamples: int = 100_000,
) -> dict:
    """Cluster-aware paired test over instances answered by both conditions.

    ``b01`` and ``b10`` retain the familiar discordant-case accounting, but the
    p-value is obtained by exchanging condition labels for whole clusters rather
    than by treating every query as an independent Bernoulli trial.
    """
    ad = a.dropna(subset=[metric]).drop_duplicates(unit).set_index(unit)
    bd = b.dropna(subset=[metric]).drop_duplicates(unit).set_index(unit)
    common = ad.index.intersection(bd.index)
    av = ad.loc[common, metric].astype(int)
    bv = bd.loc[common, metric].astype(int)
    if cluster is None or cluster == unit or cluster not in ad.columns:
        clusters = common.to_numpy()
    else:
        clusters = ad.loc[common, cluster].astype(str).to_numpy()
    diff = (bv - av).to_numpy(dtype=float)
    b01 = int(((av == 0) & (bv == 1)).sum())
    b10 = int(((av == 1) & (bv == 0)).sum())
    p = cluster_randomization_pvalue(diff, clusters, n_resamples=n_resamples)
    return dict(
        n_pairs=len(common),
        n_clusters=int(pd.Series(clusters).nunique()),
        b01=b01,
        b10=b10,
        statistic=round(float(diff.mean()), 8) if len(diff) else float("nan"),
        pvalue=p,
        test="cluster_sign_flip",
        acc_a=round(100 * av.mean(), 2),
        acc_b=round(100 * bv.mean(), 2),
    )


def mcnemar_pair(a: pd.DataFrame, b: pd.DataFrame, metric: str, unit: str) -> dict:
    """Backward-compatible unclustered wrapper.

    New analyses should call :func:`paired_cluster_test` and provide the actual
    support-set cluster.  Keeping this wrapper avoids silently breaking legacy
    scripts that operate on genuinely independent units.
    """
    return paired_cluster_test(a, b, metric, unit, cluster=unit)


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
