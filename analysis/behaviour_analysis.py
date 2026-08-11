#!/usr/bin/env python3
"""Behavioural analysis of Bongard-OpenWorld predictions.

Reads the canonical per-sample source file (results/bongard_ow_per_sample.csv.gz)
and answers four questions that raw accuracy cannot:

  Q1  Positive vs negative queries — is a model actually discriminating, or is it
      answering one class most of the time? Accuracy alone hides this.
  Q2  Item difficulty — are the failures concentrated on shared hard instances, or
      does each model fail on its own items?
  Q3  Model similarity — which models make the *same* mistakes (agreement beyond
      what their accuracies alone imply)?
  Q4  Item categorisation — which item properties (concept, commonsense flag,
      polarity) predict failure?

Usage:
    python analysis/behaviour_analysis.py [--out results/behaviour]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("results/bongard_ow_per_sample.csv.gz")
# dataset/bongard_ow.py: imagefiles["cat_2"] = positive, ["cat_1"] = negative
POS, NEG = "cat_2", "cat_1"


def load() -> pd.DataFrame:
    # low_memory=False: `ablation` and `decision_reason` are sparse strings, which
    # pandas otherwise chunk-infers into mixed dtypes and warns about.
    t = pd.read_csv(SRC, low_memory=False)
    u = t[t.analysis_decision.isin(["include", "include-valid-rows-only"]) & t.pred_valid].copy()
    u["polarity"] = np.where(u["label_true"] == POS, "positive", "negative")
    return u


def q1_polarity(u: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Positive/negative accuracy, and the degeneracy that accuracy hides."""
    by_par = (u.groupby(["paradigm", "polarity"]).is_correct.mean().unstack() * 100)
    by_par["gap_pos_minus_neg"] = by_par["positive"] - by_par["negative"]

    ca = u[(u.paradigm == "CA") & (u.components == "self")]
    m = (ca.groupby(["reasoner_model", "polarity"]).is_correct.mean().unstack() * 100).dropna()
    m["gap"] = m["positive"] - m["negative"]
    m["accuracy"] = ca.groupby("reasoner_model").is_correct.mean() * 100
    # share of predictions that went to the majority answer: 0.5 = balanced, 1.0 = constant
    pred_pos = ca.groupby("reasoner_model").pred_raw.apply(lambda s: (s == POS).mean())
    m["pred_positive_rate"] = pred_pos * 100
    m["degeneracy"] = (pred_pos - 0.5).abs() * 2      # 0 = balanced, 1 = always one class
    m = m.sort_values("gap")

    by_par.round(1).to_csv(out / "q1_polarity_by_paradigm.csv")
    m.round(1).to_csv(out / "q1_polarity_by_model.csv")
    return m


def q2_difficulty(u: pd.DataFrame, out: Path, drop: list[str] | None = None) -> pd.DataFrame:
    """Per-item difficulty across models, under one fixed condition (CA/self).

    ``drop`` removes degenerate near-constant models. This matters: a model that
    always answers "negative" is trivially correct on every negative item, which
    inflates per-item coverage and would make genuinely hard items look solved.
    """
    ca = u[(u.paradigm == "CA") & (u.components == "self")]
    if drop:
        ca = ca[~ca.reasoner_model.isin(drop)]
    # keep models that saw (nearly) the whole split, so difficulty is comparable
    n_by_model = ca.groupby("reasoner_model").test_id.nunique()
    full = n_by_model[n_by_model >= 450].index
    ca = ca[ca.reasoner_model.isin(full)]

    item = ca.groupby("test_id").agg(
        n_models=("is_correct", "size"),
        n_correct=("is_correct", "sum"),
        concept=("concept", "first"),
        commonsense=("commonsense", "first"),
        polarity=("polarity", "first"),
    )
    item["frac_correct"] = item.n_correct / item.n_models
    item.to_csv(out / "q2_item_difficulty.csv")
    return item


def q3_similarity(u: pd.DataFrame, out: Path):
    """Which models err on the same items — agreement beyond chance."""
    ca = u[(u.paradigm == "CA") & (u.components == "self")]
    w = ca.pivot_table(index="test_id", columns="reasoner_model", values="is_correct")
    w = w.dropna(axis=1, thresh=int(0.9 * len(w))).dropna()
    corr = w.corr()                        # phi coefficient on binary correctness
    corr.round(3).to_csv(out / "q3_model_correlation.csv")
    return w, corr


def q4_categorise(u: pd.DataFrame, item: pd.DataFrame, out: Path):
    """What predicts failure: commonsense flag, polarity, concept."""
    rows = []
    for col in ("commonsense", "polarity"):
        g = item.groupby(col).frac_correct.agg(["mean", "size"])
        for k, r in g.iterrows():
            rows.append({"factor": col, "level": k,
                         "mean_frac_correct": round(r["mean"] * 100, 1), "n_items": int(r["size"])})
    pd.DataFrame(rows).to_csv(out / "q4_factors.csv", index=False)

    # Bongard-OW gives (almost) one distinct concept per problem, so concepts
    # cannot be aggregated as a factor. Group by the concept's head noun instead,
    # which is coarse but does repeat across problems.
    head = item.concept.astype(str).str.lower().str.strip().str.split().str[-1]
    concept = item.assign(head=head).groupby("head").frac_correct.agg(["mean", "size"])
    concept = concept[concept["size"] >= 5].sort_values("mean")
    concept.round(3).to_csv(out / "q4_concept_head_difficulty.csv")
    return pd.DataFrame(rows), concept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/behaviour")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    u = load()
    print(f"loaded {len(u):,} valid predictions across {u.experiment_dir.nunique()} runs\n")

    # ---- Q1 ----
    print("=" * 78); print("Q1  Positive vs negative — accuracy hides degenerate answering"); print("=" * 78)
    m = q1_polarity(u, out)
    print(f"{'model':<24}{'acc':>7}{'pos':>7}{'neg':>7}{'gap':>8}{'%pred pos':>11}  flag")
    for name, r in m.iterrows():
        flag = ("DEGENERATE — near-constant answer" if r.degeneracy > 0.6 else
                "biased" if abs(r.gap) > 15 else "")
        print(f"{name:<24}{r.accuracy:>7.1f}{r['positive']:>7.1f}{r['negative']:>7.1f}"
              f"{r.gap:>+8.1f}{r.pred_positive_rate:>11.1f}  {flag}")

    # ---- Q2 ----
    print("\n" + "=" * 78); print("Q2  Item difficulty — shared or model-specific?"); print("=" * 78)
    degenerate = list(m[m.degeneracy > 0.6].index)
    print(f"  excluding degenerate models (they answer near-constantly): {degenerate}")
    item = q2_difficulty(u, out, drop=degenerate)
    k = int(item.n_models.max())
    print(f"  {len(item)} items x {k} models (full-split CA/self, non-degenerate)")
    all_right = (item.frac_correct == 1).mean() * 100
    all_wrong = (item.frac_correct == 0).mean() * 100
    print(f"  items ALL models got right : {all_right:5.1f}%")
    print(f"  items ALL models got wrong : {all_wrong:5.1f}%   <- the 'universally hard' core")
    print(f"  items with mixed outcomes  : {100-all_right-all_wrong:5.1f}%")
    # If models failed independently, all-wrong and all-right rates would be the
    # product of per-model rates. Excess over that = shared structure.
    per_model = item.n_correct.sum() / (item.n_models.sum())
    exp_all_wrong = (1 - per_model) ** k * 100
    exp_all_right = per_model ** k * 100
    print(f"\n  if models failed INDEPENDENTLY at the same mean rate ({per_model*100:.1f}%):")
    print(f"    expected all-wrong {exp_all_wrong:5.1f}%  vs observed {all_wrong:5.1f}%"
          f"   ({all_wrong/exp_all_wrong:.0f}x)" if exp_all_wrong > 0 else "")
    print(f"    expected all-right {exp_all_right:5.1f}%  vs observed {all_right:5.1f}%"
          f"   ({all_right/exp_all_right:.0f}x)" if exp_all_right > 0 else "")
    verdict = "CONCENTRATED: a shared hard core exists" if all_wrong > 3 * exp_all_wrong else "DIFFUSE: failures are largely model-specific"
    print(f"  -> {verdict}")

    # ---- Q3 ----
    print("\n" + "=" * 78); print("Q3  Which models behave alike (correlation of per-item correctness)"); print("=" * 78)
    w, corr = q3_similarity(u, out)
    print(f"  {w.shape[1]} models over {w.shape[0]} common items")
    pairs = (corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
                 .stack().sort_values(ascending=False))
    print("  most similar pairs:")
    for (a_, b_), v in pairs.head(5).items():
        print(f"    {v:+.3f}  {a_}  ~  {b_}")
    print("  least similar pairs:")
    for (a_, b_), v in pairs.tail(3).items():
        print(f"    {v:+.3f}  {a_}  ~  {b_}")

    # ---- Q4 ----
    print("\n" + "=" * 78); print("Q4  What predicts failure"); print("=" * 78)
    fac, concept = q4_categorise(u, item, out)
    print(fac.to_string(index=False))
    print(f"\n  hardest concepts (>=3 items): {', '.join(concept.head(5).index.astype(str))}")
    print(f"  easiest concepts            : {', '.join(concept.tail(5).index.astype(str))}")
    print(f"\nwrote tables to {out}/")


if __name__ == "__main__":
    main()
