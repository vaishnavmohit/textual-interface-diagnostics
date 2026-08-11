#!/usr/bin/env python3
"""S2 — How much does the description source matter, relative to the reasoner?

The paper's central claim is that the perceptual interface constrains downstream
reasoning more than the reasoner does. The evidence so far is a comparison of
*spreads of point estimates* across separate tables, which cannot be given a
confidence interval and confounds the two factors with the instance sets they
happened to run on.

A fully crossed cell exists in the data and makes the comparison direct:

    7 reasoners x 2 description sources x 137 identical items = 1,918 observations

Every reasoner sees both description sources on exactly the same items, so the
two factors are orthogonal and the design is balanced. Three complementary
estimates are reported, because each has a different failure mode:

  A. Paired within-item contrast (assumption-free)
     For each reasoner, the accuracy change from Pixtral to GPT-4o descriptions
     on the same items, with a cluster bootstrap CI. Then the reasoner spread
     under a *fixed* description source. This is the like-for-like comparison
     the paper wants, and it needs no model.

  B. Variance components (mixed-effects logistic)
     is_correct ~ perception + (1|item) + (1|reasoner)
     Reports how much outcome variance each grouping factor explains. This is
     the standard way to express "which factor matters more", but with only 7
     reasoners the reasoner variance is estimated from few levels, so its
     interval is wide and is reported as such.

  C. Explained-deviance comparison (no distributional assumptions on grouping)
     Nested logistic fits: item-only, +perception, +reasoner. The drop in
     deviance attributable to each factor, as a share, cross-checks B without
     relying on random-effect shrinkage.

If the three disagree, that is reported rather than resolved by choosing the
most favourable one.

    python analysis/studies/s2_variance_decomposition.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load, bongard_valid, boot_ci, save, para  # noqa: E402

RNG = np.random.default_rng(42)


def crossed_cell() -> pd.DataFrame:
    d = bongard_valid(load("bongard_ow"))
    ca = d[(d.paradigm == "CA") & d.components.isin(["gpt-4o", "pixtral"])]
    both = [m for m, g in ca.groupby("reasoner_model") if g.components.nunique() == 2]
    ca = ca[ca.reasoner_model.isin(both)]
    common = set.intersection(*[set(g.test_id) for _, g in ca.groupby(["components", "reasoner_model"])])
    return ca[ca.test_id.isin(common)].copy()


# --------------------------------------------------------------------------- #
# A. paired contrasts — no model assumptions
# --------------------------------------------------------------------------- #
def paired_contrasts(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in x.groupby("reasoner_model"):
        w = g.pivot_table(index=["test_id", "uid"], columns="components", values="correct")
        w = w.dropna()
        diff = (w["gpt-4o"] - w["pixtral"]).to_numpy()
        uids = w.index.get_level_values("uid").to_numpy()
        pt, (lo, hi) = boot_ci(diff, uids)
        rows.append(dict(factor="perception (GPT-4o - Pixtral)", level=m, n_items=len(w),
                         effect=round(100 * pt, 2), ci_lo=round(100 * lo, 2), ci_hi=round(100 * hi, 2)))
    # reasoner spread with the description source held fixed
    for src, g in x.groupby("components"):
        accs = g.groupby("reasoner_model").correct.mean() * 100
        rows.append(dict(factor=f"reasoner spread | {src} descriptions", level="max - min",
                         n_items=int(g.test_id.nunique()),
                         effect=round(accs.max() - accs.min(), 2),
                         ci_lo=np.nan, ci_hi=np.nan))
    return pd.DataFrame(rows)


def spread_ci(x: pd.DataFrame, source: str, n_boot: int = 2000):
    """Bootstrap CI for the reasoner max-min spread under one description source."""
    g = x[x.components == source]
    w = g.pivot_table(index="uid", columns="reasoner_model", values="correct")
    w = w.dropna()
    uids = w.index.to_numpy()
    boots = []
    for _ in range(n_boot):
        pick = RNG.choice(uids, len(uids), replace=True)
        accs = w.loc[pick].mean() * 100
        boots.append(accs.max() - accs.min())
    obs = (w.mean() * 100)
    return obs.max() - obs.min(), np.percentile(boots, [2.5, 97.5])


# --------------------------------------------------------------------------- #
# B/C. model-based decompositions
# --------------------------------------------------------------------------- #
def variance_components(x: pd.DataFrame):
    """Mixed-effects logistic; returns (results_frame, note)."""
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError:
        return None, "statsmodels unavailable"
    d = x.copy()
    d["perc"] = (d.components == "gpt-4o").astype(float)
    d["y"] = d.correct.astype(int)
    try:
        vc = {"item": "0 + C(test_id)", "reasoner": "0 + C(reasoner_model)"}
        m = BinomialBayesMixedGLM.from_formula("y ~ perc", vc, d)
        r = m.fit_vb(verbose=False)
        out = pd.DataFrame({
            "term": ["perception (fixed effect, logit)", "item SD", "reasoner SD"],
            "estimate": [round(float(r.fe_mean[list(r.model.exog_names).index("perc")]), 3),
                         round(float(np.exp(r.vcp_mean[0])), 3),
                         round(float(np.exp(r.vcp_mean[1])), 3)],
        })
        return out, "variational Bayes (fit_vb); SDs on the logit scale"
    except Exception as e:  # pragma: no cover
        return None, f"mixed model failed: {type(e).__name__}: {e}"


def explained_deviance(x: pd.DataFrame) -> pd.DataFrame:
    """Nested logistic fits; share of explainable deviance per factor."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = x.copy()
    d["y"] = d.correct.astype(int)
    d["perc"] = (d.components == "gpt-4o").astype(int)
    fits = {
        "null": "y ~ 1",
        "+item": "y ~ C(test_id)",
        "+item+perception": "y ~ C(test_id) + perc",
        "+item+perception+reasoner": "y ~ C(test_id) + perc + C(reasoner_model)",
    }
    dev = {}
    for k, f in fits.items():
        dev[k] = smf.glm(f, data=d, family=sm.families.Binomial()).fit().deviance
    total = dev["null"] - dev["+item+perception+reasoner"]
    rows = [
        dict(factor="item (which problem)", deviance_drop=round(dev["null"] - dev["+item"], 1),
             share_pct=round(100 * (dev["null"] - dev["+item"]) / total, 1)),
        dict(factor="perception (description source)",
             deviance_drop=round(dev["+item"] - dev["+item+perception"], 1),
             share_pct=round(100 * (dev["+item"] - dev["+item+perception"]) / total, 1)),
        dict(factor="reasoner (which model)",
             deviance_drop=round(dev["+item+perception"] - dev["+item+perception+reasoner"], 1),
             share_pct=round(100 * (dev["+item+perception"] - dev["+item+perception+reasoner"]) / total, 1)),
    ]
    return pd.DataFrame(rows)


def matched_contrasts(x: pd.DataFrame) -> pd.DataFrame:
    """Both factors expressed as 2-level contrasts, so they are comparable.

    This is the correction that matters. The deviance decomposition gives the
    reasoner 6 degrees of freedom and perception only 1, because the data happen
    to contain 7 reasoners and 2 description sources. A 7-level factor
    mechanically absorbs more deviance than a 2-level one, so the raw share
    comparison overstates the reasoner. Reducing both to single contrasts --- one
    perception swap per reasoner, and every reasoner-vs-reasoner pair within a
    fixed description source --- puts them on the same footing.
    """
    import itertools
    rows = []
    for m, g in x.groupby("reasoner_model"):
        w = g.pivot_table(index="test_id", columns="components", values="correct").dropna()
        rows.append(dict(factor="perception", contrast=f"{m}: gpt4o vs pixtral",
                         abs_delta=round(abs((w["gpt-4o"] - w["pixtral"]).mean() * 100), 2)))
    for src, g in x.groupby("components"):
        w = g.pivot_table(index="test_id", columns="reasoner_model", values="correct").dropna()
        for a, b in itertools.combinations(w.columns, 2):
            rows.append(dict(factor="reasoner", contrast=f"{src}: {a} vs {b}",
                             abs_delta=round(abs((w[a] - w[b]).mean() * 100), 2)))
    return pd.DataFrame(rows)


def main() -> None:
    x = crossed_cell()
    n_r, n_i = x.reasoner_model.nunique(), x.test_id.nunique()
    print(f"crossed cell: {n_r} reasoners x 2 description sources x {n_i} items = {len(x)} obs")
    print(f"balanced: {len(x) == n_r * 2 * n_i}\n")

    # ---- A ----
    pc = paired_contrasts(x)
    save(pc, "s2_paired_contrasts.csv")
    print("=== A. Paired within-item contrasts (no model assumptions) ===")
    per = pc[pc.factor.str.startswith("perception")]
    for _, r in per.iterrows():
        print(f"  {r.level:<22} {r.effect:+6.2f}  [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]")
    print(f"  {'median':<22} {per.effect.median():+6.2f}")
    print()
    for src in ("gpt-4o", "pixtral"):
        obs, (lo, hi) = spread_ci(x, src)
        print(f"  reasoner spread | {src:<8} descriptions: {obs:5.2f}  [{lo:.2f}, {hi:.2f}]")

    # ---- C ----
    ed = explained_deviance(x)
    save(ed, "s2_explained_deviance.csv")
    print("\n=== C. Share of explainable deviance ===")
    for _, r in ed.iterrows():
        print(f"  {r.factor:<34} {r.deviance_drop:>8.1f}  {r.share_pct:>5.1f}%")

    # ---- B ----
    vc, note = variance_components(x)
    print(f"\n=== B. Mixed-effects variance components ===\n  ({note})")
    if vc is not None:
        save(vc, "s2_variance_components.csv")
        for _, r in vc.iterrows():
            print(f"  {r.term:<34} {r.estimate:>8.3f}")

    # ---- D. matched 2-level contrasts (the fair comparison) ----
    mc = matched_contrasts(x)
    save(mc, "s2_matched_contrasts.csv")
    pm = mc[mc.factor == "perception"].abs_delta
    rm = mc[mc.factor == "reasoner"].abs_delta
    print("\n=== D. Matched single contrasts (1 df each — the fair comparison) ===")
    print(f"  perception (n={len(pm)}):  median |Δ| = {pm.median():5.2f}   range {pm.min():.1f}–{pm.max():.1f}")
    print(f"  reasoner   (n={len(rm)}):  median |Δ| = {rm.median():5.2f}   range {rm.min():.1f}–{rm.max():.1f}")
    print(f"  ratio of medians = {rm.median()/pm.median():.2f}")
    print(f"  {100*(rm < pm.median()).mean():.0f}% of reasoner pairs differ by less than the median perception effect")

    perc_share = float(ed[ed.factor.str.startswith("perception")].share_pct.iloc[0])
    reas_share = float(ed[ed.factor.str.startswith("reasoner")].share_pct.iloc[0])
    item_share = float(ed[ed.factor.str.startswith("item")].share_pct.iloc[0])
    med = per.effect.median()
    g_obs, (g_lo, g_hi) = spread_ci(x, "gpt-4o")
    ratio_fair = rm.median() / pm.median()

    para("S2 variance decomposition", f"""
Holding the instance set fixed lets the two interventions be compared directly.
Across {n_r} reasoners evaluated on the same {n_i} problems under both description
sources, replacing Pixtral descriptions with GPT-4o descriptions changes accuracy
by a median of {med:+.1f} points, in the same direction for every reasoner.

Comparing that against the reasoner requires care, because the two factors are
not equally sampled: the data contain {n_r} reasoners but only two description
sources, and a seven-level factor mechanically absorbs more deviance than a
two-level one. The raw decomposition of explainable deviance ({item_share:.1f}\\% problem
instance, {perc_share:.1f}\\% description source, {reas_share:.1f}\\% reasoner) therefore overstates the
reasoner. Reducing both factors to single contrasts removes the asymmetry: the
median absolute effect of swapping the description source is {pm.median():.1f} points, against
{rm.median():.1f} points for swapping one reasoner for another ({len(rm)} pairs), a ratio of {ratio_fair:.2f}.
Indeed {100*(rm < pm.median()).mean():.0f}\\% of reasoner pairs differ by less than the median perception
effect.

The defensible statement is therefore weaker than a claim that perception
dominates, and different in kind. Within this cell the two levers are of
comparable magnitude, with the reasoner slightly larger; what distinguishes them
is consistency. Improving the description source helps every reasoner by a
similar amount, whereas the reasoner effect is highly heterogeneous, spanning
{rm.min():.1f} to {rm.max():.1f} points depending on which pair is compared. The interface is
attractive as an intervention point because its benefit is uniform and
transferable, not because it is the larger effect. We note also that both
description sources here are competent VLMs; a wider range of perceptual quality
could yield a larger perception effect, and this cell cannot speak to that.
""")


if __name__ == "__main__":
    main()
