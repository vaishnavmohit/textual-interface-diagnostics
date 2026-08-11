#!/usr/bin/env python3
"""Statistics for the paper's tables: accuracy + clustered CIs + paired tests.

Implements the preregistered analysis plan (docs, IJCV_SUBMISSION_PLAN §4):
- Accuracy per (experiment, model, program) with a **cluster bootstrap** CI
  (clustered by Bongard concept when a concept/uid column is present, else by
  row). Winoground uses item-level bootstrap over entries.
- Paired **McNemar** tests between two programs on their common instances (for
  the C1-C5 ladder contrasts and paradigm comparisons).

Reads the per-sample tables produced by combine_results.py (or a single
results.xlsx). No plotting here; make_figures.py consumes these.

Usage:
    python analysis/statistics.py --combined ${OUTPUT_DIR}/combined/combined_all.csv \
        --out ${OUTPUT_DIR}/combined/stats
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from statistics_helpers import holm

RNG = np.random.default_rng(42)          # fixed seed: reproducible CIs
N_BOOT = 2000

# metric columns we understand
BONGARD_METRIC = "is_correct"
WINO_METRICS = ("text_score", "image_score", "group_score")
CLUSTER_CANDIDATES = ("commonsense", "concept", "uid", "problem_id")
UNIT_CANDIDATES = ("test_id", "id", "uid")


def _cluster_col(df: pd.DataFrame) -> str | None:
    for c in CLUSTER_CANDIDATES:
        if c in df.columns and df[c].notna().any():
            return c
    return None


def cluster_bootstrap_ci(values: np.ndarray, clusters: np.ndarray | None,
                         n_boot: int = N_BOOT, alpha: float = 0.05):
    """Mean and (lo, hi) percentile CI. If clusters is given, resample whole
    clusters (accounts for correlated Bongard queries from the same concept)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), (float("nan"), float("nan"))
    point = float(values.mean())
    if clusters is None:
        boots = [RNG.choice(values, len(values), replace=True).mean() for _ in range(n_boot)]
    else:
        clusters = np.asarray(clusters)[: len(values)]
        groups = {c: values[clusters == c] for c in np.unique(clusters)}
        keys = list(groups)
        boots = []
        for _ in range(n_boot):
            pick = RNG.choice(keys, len(keys), replace=True)
            boots.append(np.concatenate([groups[k] for k in pick]).mean())
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, (float(lo), float(hi))


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (experiment, model, program) with metric means + CIs."""
    rows = []
    keys = [k for k in ("experiment", "model_name", "program_name") if k in df.columns]
    if not keys:
        df = df.assign(_all="all"); keys = ["_all"]
    for gkey, g in df.groupby(keys):
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        rec = dict(zip(keys, gkey))
        rec["n"] = len(g)
        ccol = _cluster_col(g)
        clusters = g[ccol].to_numpy() if ccol else None
        rec["cluster_by"] = ccol or "row"
        # Detect the schema by which metric actually HAS data in this group
        # (a union-concat leaves the other schema's columns present but all-NaN).
        has_wino = any(m in g.columns and g[m].notna().any() for m in WINO_METRICS)
        metrics = WINO_METRICS if has_wino else (BONGARD_METRIC,)
        for m in metrics:
            if m not in g.columns or not g[m].notna().any():
                continue
            pt, (lo, hi) = cluster_bootstrap_ci(g[m].to_numpy(), clusters)
            rec[f"{m}"] = round(100 * pt, 2)
            rec[f"{m}_ci_lo"] = round(100 * lo, 2)
            rec[f"{m}_ci_hi"] = round(100 * hi, 2)
        rows.append(rec)
    return pd.DataFrame(rows)


def _unit_col(df: pd.DataFrame) -> str | None:
    for c in UNIT_CANDIDATES:
        if c in df.columns:
            return c
    return None


def mcnemar_pair(a: pd.DataFrame, b: pd.DataFrame, metric: str, unit: str):
    """Paired McNemar on the common instances of two programs.

    Returns (n_pairs, b01, b10, statistic, p_value, acc_a, acc_b). b01 = a wrong
    & b right; b10 = a right & b wrong. Uses the exact binomial when discordant
    pairs are few, else the chi-square approximation.
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
        p = binomtest(min(b01, b10), disc, 0.5).pvalue
        stat = float("nan")
    else:
        stat = (abs(b01 - b10) - 1) ** 2 / disc
        p = float(chi2.sf(stat, 1))
    return dict(n_pairs=len(common), b01=b01, b10=b10, statistic=stat, pvalue=p,
                acc_a=round(100 * am.mean(), 2), acc_b=round(100 * bm.mean(), 2))


def paired_tests(df: pd.DataFrame, within=("experiment", "model_name")) -> pd.DataFrame:
    """All program-vs-program McNemar tests within each (experiment, model)."""
    unit = _unit_col(df)
    if unit is None or "program_name" not in df.columns:
        return pd.DataFrame()
    within = [w for w in within if w in df.columns]
    out = []
    for gkey, g in df.groupby(within) if within else [(("all",), df)]:
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        # pick the metric that has data in THIS group (schema-aware)
        metric = ("group_score" if "group_score" in g.columns and g["group_score"].notna().any()
                  else BONGARD_METRIC)
        if metric not in g.columns or not g[metric].notna().any():
            continue
        progs = sorted(g["program_name"].unique())
        for pa, pb in itertools.combinations(progs, 2):
            res = mcnemar_pair(g[g.program_name == pa], g[g.program_name == pb], metric, unit)
            rec = dict(zip(within, gkey), metric=metric, program_a=pa, program_b=pb, **res)
            out.append(rec)
    # Holm correction within each within-group family.
    # NB: this previously used np.minimum.accumulate, which is anti-conservative
    # (it returns adjusted p-values SMALLER than Holm permits, so more results
    # appear significant). Holm is a step-down procedure requiring a running
    # MAXIMUM. Now delegated to statistics_helpers.holm, verified against
    # statsmodels.stats.multitest.multipletests.
    d = pd.DataFrame(out)
    if len(d):
        d = d.sort_values("pvalue")
        d["holm_p"] = holm(d["pvalue"].to_numpy())
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combined", required=True, help="combined_all.csv or a results.xlsx")
    ap.add_argument("--out", required=True, help="output dir for the stats tables")
    args = ap.parse_args()

    p = Path(args.combined)
    df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_excel(p)
    out = Path(args.out).expanduser(); out.mkdir(parents=True, exist_ok=True)

    acc = accuracy_table(df)
    acc.to_csv(out / "accuracy_with_ci.csv", index=False)
    print(f"accuracy table: {len(acc)} rows -> {out}/accuracy_with_ci.csv")

    pairs = paired_tests(df)
    if len(pairs):
        pairs.to_csv(out / "paired_mcnemar.csv", index=False)
        print(f"paired McNemar: {len(pairs)} comparisons -> {out}/paired_mcnemar.csv")
    else:
        print("paired McNemar: skipped (need program_name + a unit id column)")


if __name__ == "__main__":
    main()
