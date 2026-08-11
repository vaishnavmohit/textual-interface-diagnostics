#!/usr/bin/env python3
"""T1 — the statistical rigour the submission gate requires (docs/ANALYSIS_PLAN.md).

Runs on the committed per-sample source files; needs no cluster, GPU or API key.

    python analysis/t1_statistics.py [--out results/stats]

Produces, under --out:

  t1_accuracy_ci.csv        every run: accuracy, cluster-bootstrap 95% CI, n,
                            invalid rate, positive-response rate, degeneracy flag
  t1_paired_tests.csv       McNemar on common instances for the declared
                            contrast families, with Holm-adjusted p per family
  t1_separability.csv       which adjacent orderings survive their CIs
  t1_icc.csv                measured intra-cluster correlation (justifies the
                            clustering choice rather than asserting it)
  T1_SUMMARY.md             what changed, in prose

Design notes
------------
* Clustering is on ``uid`` for Bongard (500 queries come from 251 support sets,
  one positive and one negative each) and on ``id`` for Winoground. §0.1 of the
  plan measured the ICC as ~0, so this widens little — but it is reported, not
  assumed.
* Issued-query accuracy is primary: invalid model outputs in otherwise valid
  runs are scored as errors. Runs marked ``include-valid-rows-only`` because of
  a harness schema fault are excluded from component inference.
* Degenerate models (near-constant answering) are flagged, because an accuracy
  near 50% on a balanced set means something different for them.
* Effect sizes accompany every p-value: accuracy difference, its paired
  bootstrap CI, and the discordant counts that drive McNemar.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from statistics_helpers import cluster_bootstrap_ci, mcnemar_pair, holm

RNG = np.random.default_rng(42)
USABLE = ("include",)
# Anchor to the repo root so the script works from any working directory.
REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
def load(kind: str) -> pd.DataFrame:
    f = REPO / "results" / f"{kind}_per_sample.csv.gz"
    d = pd.read_csv(f, low_memory=False)
    return d[d.analysis_decision.isin(USABLE)].copy()


def _prep_bongard(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["correct"] = d.is_correct.eq(True).astype(float)
    d["answered_positive"] = np.where(
        d.pred_valid, (d.pred_raw == "cat_2").astype(float), np.nan
    )
    return d


# --------------------------------------------------------------------------- #
# T1.1 accuracy + cluster-bootstrap CI, with the degeneracy screen (T2.7)
# --------------------------------------------------------------------------- #
def accuracy_ci(ow: pd.DataFrame, hoi: pd.DataFrame, wino: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bench, d in (("bongard_ow", ow), ("bongard_hoi", hoi)):
        raw = d.copy()
        v = _prep_bongard(d)
        for key, g in v.groupby(["experiment_dir", "paradigm", "components", "reasoner_model", "split"],
                                dropna=False):
            ed, par, comp, model, split = key
            pt, (lo, hi) = cluster_bootstrap_ci(g.correct.to_numpy(), g.uid.to_numpy())
            n_all = len(raw[raw.experiment_dir == ed])
            pos_rate = g.answered_positive.mean()
            rows.append(dict(
                benchmark=bench, experiment_dir=ed, split=split, paradigm=par,
                descriptions_from=comp, reasoner_model=model,
                n_valid=int(g.pred_valid.sum()), n_total=n_all,
                invalid_pct=round(100 * (1 - g.pred_valid.mean()), 1) if n_all else np.nan,
                accuracy=round(100 * pt, 2), ci_lo=round(100 * lo, 2), ci_hi=round(100 * hi, 2),
                ci_width=round(100 * (hi - lo), 2),
                answered_positive_pct=round(100 * pos_rate, 1),
                degenerate=bool(abs(pos_rate - 0.5) * 2 > 0.6),
                cluster_by="uid"))
    for key, g in wino.groupby(["source_file", "perception", "reasoner_model", "condition", "variant"],
                               dropna=False):
        sf, perc, model, cond, var = key
        for metric in ("text_score", "image_score", "group_score"):
            pt, (lo, hi) = cluster_bootstrap_ci(g[metric].to_numpy(), g["id"].to_numpy())
            rows.append(dict(
                benchmark="winoground", experiment_dir=sf, split=metric.replace("_score", ""),
                paradigm=cond, descriptions_from=perc, reasoner_model=model,
                n_valid=len(g), n_total=len(g), invalid_pct=0.0,
                accuracy=round(100 * pt, 2), ci_lo=round(100 * lo, 2), ci_hi=round(100 * hi, 2),
                ci_width=round(100 * (hi - lo), 2),
                answered_positive_pct=np.nan, degenerate=False, cluster_by="id"))
    return pd.DataFrame(rows).sort_values(["benchmark", "paradigm", "accuracy"], ascending=[1, 1, 0])


# --------------------------------------------------------------------------- #
# T1.4 paired effect size — accuracy difference with a paired bootstrap CI
# --------------------------------------------------------------------------- #
def paired_delta_ci(a: pd.Series, b: pd.Series, clusters: np.ndarray, n_boot: int = 2000):
    """CI for mean(b) - mean(a) over paired units, resampling clusters."""
    diff = (b - a).to_numpy(dtype=float)
    groups = {c: diff[clusters == c] for c in np.unique(clusters)}
    keys = list(groups)
    boots = [np.concatenate([groups[k] for k in RNG.choice(keys, len(keys), replace=True)]).mean()
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def _pair(g_a: pd.DataFrame, g_b: pd.DataFrame, unit: str, cluster: str, metric: str):
    a = g_a.dropna(subset=[metric]).drop_duplicates(unit).set_index(unit)
    b = g_b.dropna(subset=[metric]).drop_duplicates(unit).set_index(unit)
    common = a.index.intersection(b.index)
    if len(common) < 20:
        return None
    a, b = a.loc[common], b.loc[common]
    res = mcnemar_pair(a.reset_index(), b.reset_index(), metric, unit)
    # When the cluster IS the unit (Winoground items), set_index has consumed the
    # column — resample the index instead. Each item is then its own cluster,
    # which reduces to an ordinary paired bootstrap, as intended.
    clusters = a.index.to_numpy() if cluster == unit else a[cluster].to_numpy()
    d, lo, hi = paired_delta_ci(a[metric], b[metric], clusters)
    res.update(delta=round(100 * d, 2), delta_ci_lo=round(100 * lo, 2), delta_ci_hi=round(100 * hi, 2))
    return res


# --------------------------------------------------------------------------- #
# T1.2 the declared contrast families
# --------------------------------------------------------------------------- #
def paired_tests(ow: pd.DataFrame, hoi: pd.DataFrame, wino: pd.DataFrame) -> pd.DataFrame:
    out = []

    # F1 — paradigm ladder, per model, Bongard-OW (no ablation arms)
    v = _prep_bongard(ow)
    base = v[v.ablation.isna() | (v.ablation == "")]
    for model, g in base.groupby("reasoner_model"):
        pars = {p: s for p, s in g.groupby("paradigm") if len(s) >= 50}
        for pa, pb in combinations(sorted(pars), 2):
            r = _pair(pars[pa], pars[pb], "test_id", "uid", "correct")
            if r:
                out.append(dict(family="F1 paradigm (OW)", benchmark="bongard_ow",
                                unit_a=f"{model}:{pa}", unit_b=f"{model}:{pb}", **r))

    # F2 — perception swap, per reasoner, on the common instances
    ca = v[v.paradigm == "CA"]
    for model, g in ca.groupby("reasoner_model"):
        srcs = {c: s for c, s in g.groupby("descriptions_from" if "descriptions_from" in g else "components")}
        if "gpt-4o" in srcs and "pixtral" in srcs:
            r = _pair(srcs["pixtral"], srcs["gpt-4o"], "test_id", "uid", "correct")
            if r:
                out.append(dict(family="F2 perception swap (OW)", benchmark="bongard_ow",
                                unit_a=f"{model}:pixtral-desc", unit_b=f"{model}:gpt4o-desc", **r))

    # F3 — CA vs ICA, Winoground, perception and reasoner held fixed
    for (perc, model), g in wino.groupby(["perception", "reasoner_model"]):
        conds = {c: s for c, s in g.groupby("condition")}
        if "CA" in conds and "ICA" in conds:
            for metric in ("text_score", "image_score", "group_score"):
                ica = conds["ICA"]
                for var, sub in ica.groupby("variant"):
                    r = _pair(conds["CA"], sub, "id", "id", metric)
                    if r:
                        out.append(dict(family="F3 CA vs ICA (Wino)", benchmark="winoground",
                                        unit_a=f"{perc}/{model}:CA:{metric}",
                                        unit_b=f"{perc}/{model}:ICA[{var or 'base'}]:{metric}", **r))

    # F4 — rule supplied externally vs CA, same reasoner
    for model, g in v.groupby("reasoner_model"):
        conds = {p: s for p, s in g.groupby("paradigm")}
        if "CA" in conds and "RuleApply" in conds:
            r = _pair(conds["CA"], conds["RuleApply"], "test_id", "uid", "correct")
            if r:
                out.append(dict(family="F4 rule supply (OW)", benchmark="bongard_ow",
                                unit_a=f"{model}:CA", unit_b=f"{model}:RuleApply", **r))

    # F5 — paradigm ladder per HOI split
    h = _prep_bongard(hoi)
    for (split, model), g in h.groupby(["split", "reasoner_model"]):
        pars = {p: s for p, s in g.groupby("paradigm") if len(s) >= 50}
        for pa, pb in combinations(sorted(pars), 2):
            r = _pair(pars[pa], pars[pb], "test_id", "uid", "correct")
            if r:
                out.append(dict(family="F5 paradigm (HOI)", benchmark="bongard_hoi",
                                unit_a=f"{split}/{model}:{pa}", unit_b=f"{split}/{model}:{pb}", **r))

    d = pd.DataFrame(out)
    if not len(d):
        return d
    # T1.3 — Holm within each declared family
    d["holm_p"] = np.nan
    for fam, g in d.groupby("family"):
        d.loc[g.index, "holm_p"] = holm(g.pvalue.to_numpy())
    d["significant_holm"] = d.holm_p < 0.05
    return d.sort_values(["family", "pvalue"])


# --------------------------------------------------------------------------- #
# T1.1b separability — which adjacent orderings survive their CIs
# --------------------------------------------------------------------------- #
def separability(acc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sel = acc[(acc.benchmark == "bongard_ow") & (acc.paradigm == "CA")]
    for comp, g in sel.groupby("descriptions_from"):
        g = g.sort_values("accuracy", ascending=False).reset_index(drop=True)
        for i in range(len(g) - 1):
            a, b = g.loc[i], g.loc[i + 1]
            rows.append(dict(block=f"CA / {comp}", higher=a.reasoner_model, lower=b.reasoner_model,
                             acc_higher=a.accuracy, acc_lower=b.accuracy,
                             gap=round(a.accuracy - b.accuracy, 2),
                             ci_overlap=bool(a.ci_lo <= b.ci_hi),
                             separable=bool(a.ci_lo > b.ci_hi)))
    return pd.DataFrame(rows)


def direction_summary(tests: pd.DataFrame) -> pd.DataFrame:
    """Describe contrast directions without treating them as independent.

    Contrasts within a family reuse models and evaluation cases. Counts of
    positive and negative effects are therefore descriptive and must not be
    passed to a binomial sign test.
    """
    rows = []
    for fam, g in tests.groupby("family"):
        pos = int((g.delta > 0).sum()); neg = int((g.delta < 0).sum())
        n = pos + neg
        if n < 3:
            continue
        rows.append(dict(family=fam, n_contrasts=n, positive=pos, negative=neg,
                         median_delta=round(float(g.delta.median()), 2),
                         n_individually_significant=int(g.significant_holm.sum())))
    return pd.DataFrame(rows)


def icc_table(ow: pd.DataFrame) -> pd.DataFrame:
    """Measured intra-cluster correlation — justifies the clustering choice."""
    v = _prep_bongard(ow)
    rows = []
    for ed, g in v.groupby("experiment_dir"):
        pairs = [x.correct.to_numpy() for _, x in g.groupby("uid") if len(x) == 2]
        if len(pairs) < 30:
            continue
        a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
        if a.std() == 0 or b.std() == 0:
            continue
        rows.append(dict(experiment_dir=ed, n_uid_pairs=len(pairs),
                         icc=round(float(np.corrcoef(a, b)[0, 1]), 3)))
    return pd.DataFrame(rows).sort_values("icc", ascending=False)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/stats")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    ow, hoi, wino = load("bongard_ow"), load("bongard_hoi"), load("winoground")
    print(f"loaded  OW {len(ow):,}  HOI {len(hoi):,}  Winoground {len(wino):,} rows\n")

    acc = accuracy_ci(ow, hoi, wino);      acc.to_csv(out / "t1_accuracy_ci.csv", index=False)
    tests = paired_tests(ow, hoi, wino);   tests.to_csv(out / "t1_paired_tests.csv", index=False)
    sep = separability(acc);               sep.to_csv(out / "t1_separability.csv", index=False)
    icc = icc_table(ow);                   icc.to_csv(out / "t1_icc.csv", index=False)
    directions = direction_summary(tests); directions.to_csv(out / "t1_direction_summary.csv", index=False)

    print(f"T1.1 accuracy + CI      {len(acc):>4} rows   -> t1_accuracy_ci.csv")
    print(f"T1.2 paired tests       {len(tests):>4} rows   -> t1_paired_tests.csv")
    print(f"T1.1b separability      {len(sep):>4} rows   -> t1_separability.csv")
    print(f"T1.x measured ICC       {len(icc):>4} rows   -> t1_icc.csv")
    print(f"T1.4 direction summary {len(directions):>4} rows   -> t1_direction_summary.csv")
    if len(directions):
        print("\n=== contrast directions per family (descriptive; contrasts are dependent) ===")
        for _, r in directions.iterrows():
            print(f"  {r.family:<26} {r.positive}+/{r.negative}-  median Δ={r.median_delta:+6.2f}  "
                  f"({r.n_individually_significant} individually sig.)")

    if len(tests):
        print("\n=== significant after Holm, by family ===")
        s = tests.groupby("family").significant_holm.agg(["sum", "size"])
        for fam, r in s.iterrows():
            print(f"  {fam:<26} {int(r['sum'])}/{int(r['size'])}")
    if len(sep):
        print(f"\n=== separability: {int(sep.separable.sum())}/{len(sep)} adjacent orderings survive their CIs ===")
    if len(icc):
        print(f"\n=== ICC: median {icc.icc.median():+.3f}, max {icc.icc.max():+.3f} "
              f"-> clustering {'matters' if icc.icc.median() > 0.1 else 'is negligible'} ===")
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
