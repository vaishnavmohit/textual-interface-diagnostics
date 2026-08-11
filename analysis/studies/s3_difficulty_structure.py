#!/usr/bin/env python3
"""S3 — Is "difficulty" one thing on this benchmark, and do models fail alike?

Planned as a Rasch (1PL) analysis, which would place items and models on a
single latent scale. Rasch assumes unidimensionality: one ability, one
difficulty, and item ordering that every model agrees on. That assumption is
testable, and here it fails — so this study reports the structure that is
actually present instead of fitting a model whose premise is violated.

Merges the planned S3 (item difficulty) and S8 (error consistency), because once
unidimensionality is rejected they become the same question: if there is no
single difficulty axis, what *does* organise which items a model gets wrong?

Steps:
  1. Unidimensionality, tested against chance by parallel analysis --- the
     eigenvalue spectrum of the observed model-by-model correlation matrix
     compared against matrices built from independent models with the same
     accuracies. With only 8 models a raw eigenvalue ratio proves nothing.
  2. If rejected: how much shared difficulty structure exists at all, measured
     as the excess of the observed first eigenvalue over its null.
  3. What predicts an item being hard, if not a latent trait --- pairwise error
     agreement, and whether it tracks model family or model accuracy.

Degenerate near-constant responders are excluded throughout: a model that always
answers one class is trivially "correct" on half the items and manufactures
spurious agreement.

    python analysis/studies/s3_difficulty_structure.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load, bongard_valid, save, para  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

RNG = np.random.default_rng(42)


def matrix(source: str = "self") -> pd.DataFrame:
    """items x models over one perception condition.

    ``source="self"`` is the original design: every system describes its own
    images, so systems differ in perception AND reasoner. ``source="gpt-4o"``
    holds the descriptions byte-identical, so only the reasoner differs. The two
    answer different questions and are compared directly in step 4.
    """
    d = bongard_valid(load("bongard_ow"))
    ca = d[(d.paradigm == "CA") & (d.components == source) & (d.ablation.isna() | (d.ablation == ""))]
    deg = [m for m, g in ca.groupby("reasoner_model")
           if abs((g.pred_raw == "cat_2").mean() - 0.5) * 2 > 0.6]
    ca = ca[~ca.reasoner_model.isin(deg)]
    w = ca.pivot_table(index="test_id", columns="reasoner_model", values="correct")
    return w.dropna(axis=1, thresh=int(0.9 * len(w))).dropna(), deg


def shared_structure(obs: pd.DataFrame, k: int, n_draw: int = 300,
                     match_to=None) -> dict:
    """Shared-difficulty statistics at a FIXED pool size of k models.

    Pool size is not neutral: eigenvalues and split-half correlations both move
    with the number of models, so a 42-model condition cannot be compared with an
    8-model one directly. Every statistic here is computed on random k-model
    subsets of whichever condition is passed, making the two comparable.
    """
    cols = list(obs.columns)
    rhos, ev_excess, excess_cofail = [], [], []
    for _ in range(n_draw):
        sub = (obs[accuracy_matched_pool(obs, match_to)] if match_to is not None
               else obs[list(RNG.choice(cols, k, replace=False))])
        half = k // 2
        order = RNG.permutation(list(sub.columns))
        a, b = list(order[:half]), list(order[half:2 * half])
        rho = spearmanr(sub[a].mean(axis=1), sub[b].mean(axis=1)).statistic
        rhos.append(rho)
        ev, _, null95 = parallel_analysis(sub, n_iter=60)
        ev_excess.append(ev[0] - null95[0])
        x = sub.to_numpy()
        pairs = []
        for i, j in combinations(range(k), 2):
            bw = float(((x[:, i] == 0) & (x[:, j] == 0)).mean())
            pairs.append(bw - (1 - x[:, i].mean()) * (1 - x[:, j].mean()))
        excess_cofail.append(float(np.median(pairs)))
    rho = float(np.mean(rhos))
    return dict(k=k, n_models_available=len(cols),
                splithalf_rho=round(rho, 3),
                pool_reliability=round(2 * rho / (1 + rho), 3),
                ev1_excess_over_null=round(float(np.mean(ev_excess)), 3),
                median_excess_cofailure=round(100 * float(np.mean(excess_cofail)), 2))


def accuracy_matched_pool(fixed: pd.DataFrame, target_acc, n_near: int = 3):
    """Pick one fixed-interface model per target accuracy, nearest-first.

    Without this the comparison is confounded: the own-description pool is both
    smaller and weaker than the fixed-interface pool, and how much systems agree
    about difficulty depends on how good and how similar they are. Matching the
    accuracy profile leaves the interface as the difference between conditions.
    """
    acc = fixed.mean()
    chosen = []
    for t in target_acc:
        cand = acc.drop(chosen).sub(t).abs().nsmallest(n_near).index
        chosen.append(str(RNG.choice(cand)))
    return chosen


def parallel_analysis(obs: pd.DataFrame, n_iter: int = 500):
    """Compare observed eigenvalues against independent models of equal accuracy.

    The null keeps each model's overall accuracy but destroys any item-level
    agreement, so eigenvalues above the null band indicate genuinely shared
    structure rather than an artefact of matrix size.
    """
    C = np.corrcoef(obs.T.values)
    ev_obs = np.linalg.eigvalsh(C)[::-1]
    rates = obs.mean(axis=0).to_numpy()
    n_items, n_models = obs.shape
    null = np.empty((n_iter, n_models))
    for i in range(n_iter):
        sim = (RNG.random((n_items, n_models)) < rates).astype(float)
        null[i] = np.linalg.eigvalsh(np.corrcoef(sim.T))[::-1]
    return ev_obs, null.mean(axis=0), np.percentile(null, 95, axis=0)


def main() -> None:
    obs, dropped = matrix()
    n_items, n_models = obs.shape
    print(f"item x model matrix: {n_items} items x {n_models} models")
    print(f"excluded as near-constant responders: {dropped}\n")

    # ---- 1/2 dimensionality ---------------------------------------------- #
    ev, null_mean, null_95 = parallel_analysis(obs)
    dim = pd.DataFrame(dict(component=range(1, n_models + 1),
                            eigenvalue=ev.round(3),
                            null_mean=null_mean.round(3),
                            null_p95=null_95.round(3),
                            above_null=(ev > null_95)))
    save(dim, "s3_dimensionality.csv")
    print("=== 1. Unidimensionality, against a parallel-analysis null ===")
    print(f"  {'comp':<6}{'observed':>10}{'null mean':>11}{'null p95':>10}   verdict")
    for _, r in dim.iterrows():
        print(f"  {int(r.component):<6}{r.eigenvalue:>10.3f}{r.null_mean:>11.3f}"
              f"{r.null_p95:>10.3f}   {'ABOVE null' if r.above_null else '-'}")
    n_real = int(dim.above_null.sum())
    var1 = 100 * ev[0] / ev.sum()
    print(f"\n  components above the null: {n_real}")
    print(f"  first component explains {var1:.1f}% of variance (ratio ev1/ev2 = {ev[0]/ev[1]:.2f})")

    # ---- 3 pairwise agreement -------------------------------------------- #
    rows = []
    for a, b in combinations(obs.columns, 2):
        x, y = obs[a].to_numpy(), obs[b].to_numpy()
        both_wrong = float(((x == 0) & (y == 0)).mean())
        # excess co-failure over independence, given each model's error rate
        exp_wrong = float((1 - x.mean()) * (1 - y.mean()))
        rows.append(dict(model_a=a, model_b=b,
                         acc_a=round(100 * x.mean(), 1), acc_b=round(100 * y.mean(), 1),
                         phi=round(float(np.corrcoef(x, y)[0, 1]), 3),
                         co_failure=round(100 * both_wrong, 1),
                         expected_if_independent=round(100 * exp_wrong, 1),
                         excess=round(100 * (both_wrong - exp_wrong), 1)))
    pw = pd.DataFrame(rows).sort_values("phi", ascending=False)
    save(pw, "s3_pairwise_agreement.csv")
    print("\n=== 3. Pairwise error agreement (excess co-failure over independence) ===")
    print(f"  {'pair':<44}{'phi':>7}{'co-fail':>9}{'expected':>10}{'excess':>8}")
    for _, r in pw.head(6).iterrows():
        print(f"  {r.model_a[:20]+' ~ '+r.model_b[:20]:<44}{r.phi:>7.3f}"
              f"{r.co_failure:>8.1f}%{r.expected_if_independent:>9.1f}%{r.excess:>+8.1f}")
    print(f"  ...\n  median phi across {len(pw)} pairs: {pw.phi.median():.3f}   median excess: {pw.excess.median():+.1f} points")

    # does agreement track accuracy similarity?
    pw["acc_gap"] = (pw.acc_a - pw.acc_b).abs()
    rho, pv = spearmanr(pw.acc_gap, pw.phi)
    print(f"  corr(|accuracy gap|, phi) = {rho:+.2f} (p={pv:.3f}) "
          f"-> {'similar-accuracy models agree more' if rho < -0.3 else 'agreement is not explained by accuracy similarity'}")

    # ---- item difficulty distribution ------------------------------------ #
    p = obs.mean(axis=1)
    diff = pd.DataFrame(dict(test_id=obs.index, solve_rate=p.round(3),
                             n_models=n_models,
                             all_correct=(p == 1), all_wrong=(p == 0)))
    save(diff, "s3_item_difficulty.csv")
    print(f"\n=== Item difficulty spread ===")
    print(f"  solve rate: min {p.min():.2f}  median {p.median():.2f}  max {p.max():.2f}  sd {p.std():.2f}")
    print(f"  solved by all {n_models} models: {int((p==1).sum())}/{n_items}"
          f"   failed by all: {int((p==0).sum())}/{n_items}")

    # ---- 4 does holding the interface fixed create shared difficulty? ---- #
    # The condition above lets every system supply its own descriptions, so two
    # systems differ in perception AND reasoner. Repeating it over a byte-identical
    # description artifact isolates the reasoner. Both are subsampled to the same
    # pool size, because eigenvalues and split-half correlations move with it.
    fixed, deg_fixed = matrix("gpt-4o")
    K = min(obs.shape[1], fixed.shape[1])
    own_stats = shared_structure(obs, K)
    fix_stats = shared_structure(fixed, K)
    # and again with the fixed-interface pool matched to the own-condition
    # accuracy profile, so only the interface differs between the two rows
    fix_matched = shared_structure(fixed, K, match_to=list(obs.mean()))
    cmp = pd.DataFrame([dict(condition="own descriptions", **own_stats),
                        dict(condition="fixed descriptions", **fix_stats),
                        dict(condition="fixed, acc-matched", **fix_matched)])
    save(cmp, "s3_condition_comparison.csv")

    print(f"\n=== 4. Shared difficulty, own vs fixed descriptions (both at k={K} models) ===")
    print(f"  {'condition':<22}{'pool':>6}{'split-half rho':>16}{'reliability':>13}"
          f"{'ev1 over null':>15}{'excess co-fail':>16}")
    for _, r in cmp.iterrows():
        print(f"  {r.condition:<22}{int(r.n_models_available):>6}{r.splithalf_rho:>16.3f}"
              f"{r.pool_reliability:>13.3f}{r.ev1_excess_over_null:>15.3f}"
              f"{r.median_excess_cofailure:>+15.2f}%")
    print(f"  (full fixed-description pool is {fixed.shape[1]} models x {fixed.shape[0]} items;"
          f" subsampled to {K} for comparability)")

    para("S3 difficulty structure", f"""
We had intended to place items and models on a common latent scale with a Rasch
model, which presumes that a single difficulty dimension orders the items for
every model. That presumption does not hold here, and testing it is more
informative than fitting the model anyway.

Parallel analysis compares the eigenvalue spectrum of the observed
model-by-model agreement matrix against matrices generated from independent
models with the same accuracies. Only {n_real} component{'s' if n_real != 1 else ''} exceeds the null, and the
leading component accounts for {var1:.1f}\\% of variance (ratio to the second,
{ev[0]/ev[1]:.2f}). There is therefore weak shared structure --- more than chance, but far
from the dominant single factor a unidimensional treatment requires.

The pairwise view says the same thing. Across {len(pw)} model pairs the median excess
co-failure over independence is {pw.excess.median():+.1f} points: models fail together only
slightly more often than their individual error rates predict. No item in the
set is failed by all {n_models} models, while {int((p==1).sum())} of {n_items} are solved by all of them.

That conclusion is real but conditional, and the condition turns out to matter.
In this design each system also describes its own images, so two systems differ
in perception as well as in reasoning, and part of their disagreement about which
items are hard is disagreement about what the items even say. Repeating the
measurement over a byte-identical description artifact isolates the reasoner.

Both conditions are compared at the same pool size of {K} models, and with the
fixed-interface pool additionally matched to the own-description pool's accuracy
profile --- necessary because how much systems agree depends on how good and how
similar they are, and the fixed-interface pool is both larger and stronger.
Holding the interface fixed roughly doubles every measure of shared structure:
split-half difficulty correlation rises from {own_stats["splithalf_rho"]:.2f} to
{fix_matched["splithalf_rho"]:.2f} (pool reliability {own_stats["pool_reliability"]:.2f}
to {fix_matched["pool_reliability"]:.2f}), the leading eigenvalue's excess over
its null from {own_stats["ev1_excess_over_null"]:.2f} to
{fix_matched["ev1_excess_over_null"]:.2f}, and median excess co-failure from
{own_stats["median_excess_cofailure"]:+.1f} to
{fix_matched["median_excess_cofailure"]:+.1f} points. Without the accuracy
matching the apparent effect is roughly twice as large again
({fix_stats["splithalf_rho"]:.2f}), so about half of the raw difference is pool
composition rather than the interface.

The two readings therefore fit together. Difficulty is weakly shared when systems
differ in both components, and substantially shared when they read the same
descriptions --- and at the full pool of {fixed.shape[1]} reasoners the shared
component is stronger still (Section~\\ref{{sec:verbalization-cost}} measures a
split-half correlation of 0.71 over twenty-one-model halves), which is what
increasing the pool size should do. Much of what presents as system-specific
difficulty is thus specific to the perception source, not to the reasoner.

The consequence for interpretation survives that refinement. Aggregate accuracy
still summarises how many items a system happens to solve rather than where it
sits on a shared competence scale, because in practice systems are compared
across differing pipelines, which is the weakly-shared regime. It also means the
benchmark has no saturated core: headroom exists on essentially every item for
some system, so gains remain available without new items.
""")


if __name__ == "__main__":
    main()
