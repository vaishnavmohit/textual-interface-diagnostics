#!/usr/bin/env python3
"""M2 — Can a larger reasoner buy its way past the interface?

Forty-two reasoners read byte-identical GPT-4o descriptions of the same 499
Bongard-OpenWorld problems, and eleven model families appear at two to four
sizes. Holding the descriptions fixed and varying only the reasoner's scale is
therefore a controlled experiment rather than a leaderboard.

M1 turns this into a prediction rather than an exploration. If the description
interface owns a set of failures because it discarded the evidence that would
have settled them, then no downstream reasoner can recover that evidence,
however large --- while on the problems whose evidence survived, scale should
behave normally. So the test is not "does accuracy rise with size" but "does it
rise *differently* on the interface's own failures".

Two design points matter:

- Parameter counts come from a curated table, not from the model tag. Several
  names carry no size (`llama3.2`, `mistral`, `phi4`) and one family mixes
  text-only and vision variants at the same nominal scale.
- The hard stratum for a family is defined by the consensus of every reasoner
  OUTSIDE that family. Defining it on all reasoners would let a family help
  select the items it is then scored on.

    python analysis/studies/m2_reasoner_scaling.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RNG, load, bongard_valid, para, save  # noqa: E402

DESC = "gpt-4o"
HARD_Q = 0.10
N_BOOT = 2000

# Curated: billions of parameters, and the family a size series belongs to.
# Models whose scale is ambiguous are deliberately absent -- deepseek-v2 and
# deepseek-llm are mixture-of-experts or unlabelled, and `qwen:32b` is a lone
# point with no sibling. A family is only usable if its members differ in scale
# and nothing else that we know of.
SPEC: dict[str, tuple[str, float]] = {
    "qwen2.5:7b": ("Qwen2.5", 7), "qwen2.5:14b": ("Qwen2.5", 14),
    "qwen2.5:32b": ("Qwen2.5", 32), "qwen2.5:72b": ("Qwen2.5", 72),
    "gemma3:1b": ("Gemma3", 1), "gemma3:4b": ("Gemma3", 4),
    "gemma3:12b": ("Gemma3", 12), "gemma3:27b": ("Gemma3", 27),
    "deepseek-r1:8b": ("DeepSeek-R1", 8), "deepseek-r1:14b": ("DeepSeek-R1", 14),
    "deepseek-r1:32b": ("DeepSeek-R1", 32), "deepseek-r1:70b": ("DeepSeek-R1", 70),
    "llama2": ("Llama2", 7), "llama2:13b": ("Llama2", 13), "llama2:70b": ("Llama2", 70),
    "llava-llama3": ("LLaVA", 8), "llava:13b": ("LLaVA", 13), "llava:34b": ("LLaVA", 34),
    "qwen2.5vl:3b": ("Qwen2.5-VL", 3), "qwen2.5vl:7b": ("Qwen2.5-VL", 7),
    "qwen2.5vl:32b": ("Qwen2.5-VL", 32),
    "llama3.1:8b": ("Llama3.1", 8), "llama3.1:70b": ("Llama3.1", 70),
    "llama3": ("Llama3", 8), "llama3:70b": ("Llama3", 70),
    "qwen2:7b": ("Qwen2", 7), "qwen2:72b": ("Qwen2", 72),
    "gemma2": ("Gemma2", 9), "gemma2:27b": ("Gemma2", 27),
    "llama3.2-vision11b": ("Llama3.2-V", 11), "llama3.2-vision90b": ("Llama3.2-V", 90),
}


def ca_matrix(u: pd.DataFrame) -> pd.DataFrame:
    ca = u[u.paradigm.eq("CA") & u.components.eq(DESC)]
    ca = ca.drop_duplicates(["reasoner_model", "test_id"])
    return ca.pivot(index="test_id", columns="reasoner_model", values="correct").dropna().astype(float)


def _fit(x, acc, logit: bool) -> float:
    """Slope per doubling, optionally on the log-odds scale.

    Accuracy slopes are not comparable between strata with different base rates:
    a stratum where the strong models sit at 97% has no room to improve, so a
    flat slope there means something different from a flat slope at 35%. The
    log-odds scale removes that compression, and is the scale the hard-versus-rest
    comparison is made on.
    """
    if logit:
        p = np.clip(acc, 0.01, 0.99)
        acc = np.log(p / (1 - p))
    return float(np.polyfit(x, acc, 1)[0])


def slope_ci(sizes, mat, uids, logit: bool = False, n_boot=N_BOOT):
    """Slope per doubling of parameters, with a bootstrap CI.

    ``mat`` is models x items of 0/1 outcomes, in the same order as ``sizes``.
    Resamples problems, clustered on uid, rather than models: the sizes are the
    design, not a sample, so the uncertainty that matters is over items.
    """
    x = np.log2(np.asarray(sizes, dtype=float))
    mat = np.asarray(mat, dtype=float)
    point = _fit(x, mat.mean(axis=1), logit)
    idx = pd.Series(range(mat.shape[1]))
    groups = {k: v.to_numpy() for k, v in idx.groupby(np.asarray(uids))}
    keys = list(groups)
    boots = []
    for _ in range(n_boot):
        pick = np.concatenate([groups[k] for k in RNG.choice(keys, len(keys), replace=True)])
        boots.append(_fit(x, mat[:, pick].mean(axis=1), logit))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, (float(lo), float(hi))


def main() -> None:
    u = bongard_valid(load("bongard_ow"))
    u = u[u.ablation.isna()]
    piv = ca_matrix(u)
    present = [m for m in SPEC if m in piv.columns]
    fam = pd.DataFrame([(m, *SPEC[m]) for m in present], columns=["model", "family", "params"])
    usable = fam.groupby("family").filter(lambda g: len(g) >= 2)
    print(f"basis: {piv.shape[1]} reasoners x {piv.shape[0]} items on identical {DESC} descriptions")
    print(f"       {usable.family.nunique()} families with >=2 sizes, {len(usable)} models")

    uid = u.drop_duplicates("test_id").set_index("test_id").uid.reindex(piv.index)

    rows = []
    for f, g in usable.groupby("family"):
        g = g.sort_values("params")
        models = g.model.tolist()
        # hard stratum from every reasoner OUTSIDE this family
        outside = [c for c in piv.columns if c not in set(fam[fam.family.eq(f)].model)]
        cons = piv[outside].mean(axis=1)
        hard = cons <= cons.quantile(HARD_Q)
        acc_all = piv.loc[:, models].mean().to_numpy()
        s_all, ci_all = slope_ci(g.params, piv[models].T.to_numpy(), uid.values)
        s_h, ci_h = slope_ci(g.params, piv.loc[hard, models].T.to_numpy(), uid[hard].values)
        s_r, ci_r = slope_ci(g.params, piv.loc[~hard, models].T.to_numpy(), uid[~hard].values)
        rows.append(dict(family=f, n_sizes=len(g),
                         sizes="/".join(str(int(p)) for p in g.params),
                         acc_small=round(100 * acc_all[0], 1), acc_large=round(100 * acc_all[-1], 1),
                         slope_all=round(100 * s_all, 2), all_lo=round(100 * ci_all[0], 2),
                         all_hi=round(100 * ci_all[1], 2),
                         slope_hard=round(100 * s_h, 2), hard_lo=round(100 * ci_h[0], 2),
                         hard_hi=round(100 * ci_h[1], 2),
                         slope_rest=round(100 * s_r, 2), rest_lo=round(100 * ci_r[0], 2),
                         rest_hi=round(100 * ci_r[1], 2)))
    res = pd.DataFrame(rows).sort_values("slope_all", ascending=False)
    save(res, "m2_reasoner_scaling.csv")

    print(f"\npoints of accuracy per doubling of parameters (hard decile n={int(hard.sum())})")
    print(f"  {'family':<13}{'sizes(B)':<13}{'small->large':<14}{'overall':>18}{'on hard':>17}{'on rest':>17}")
    for _, r in res.iterrows():
        print(f"  {r.family:<13}{r.sizes:<13}{r.acc_small:>5.1f}->{r.acc_large:<7.1f}"
              f"{r.slope_all:>+7.2f} [{r.all_lo:+.1f},{r.all_hi:+.1f}]"
              f"{r.slope_hard:>+7.2f} [{r.hard_lo:+.1f},{r.hard_hi:+.1f}]"
              f"{r.slope_rest:>+7.2f} [{r.rest_lo:+.1f},{r.rest_hi:+.1f}]")

    # Pooled across families: is the slope on the interface's own failures
    # smaller than on the rest? Paired by family, so family-level differences in
    # architecture and training cancel.
    d = res.slope_rest - res.slope_hard
    stat, p = wilcoxon(d) if len(d) > 5 else (np.nan, np.nan)
    print(f"\npooled over {len(res)} families: mean slope on rest {res.slope_rest.mean():+.2f} "
          f"vs on hard {res.slope_hard.mean():+.2f}")
    print(f"  paired difference {d.mean():+.2f} pts/doubling, Wilcoxon p={p:.4f}, "
          f"{(d > 0).sum()}/{len(d)} families positive")

    # The accuracy scale compresses the 'rest' stratum against its ceiling, and
    # a decile is thin. Re-test on the log-odds scale and at a wider stratum,
    # restricted to families with 3+ sizes where a slope is more than a
    # two-point difference. Reported whatever they show.
    print("\nsensitivity — same comparison on the log-odds scale, by stratum width")
    sens = []
    for q in (0.10, 0.20, 0.30):
        for min_sizes in (2, 3):
            diffs = []
            for f, g in usable.groupby("family"):
                g = g.sort_values("params")
                if len(g) < min_sizes:
                    continue
                models = g.model.tolist()
                outside = [c for c in piv.columns if c not in set(fam[fam.family.eq(f)].model)]
                cons = piv[outside].mean(axis=1)
                hd = cons <= cons.quantile(q)
                sh, _ = slope_ci(g.params, piv.loc[hd, models].T.to_numpy(),
                                 uid[hd].values, logit=True, n_boot=1)
                sr, _ = slope_ci(g.params, piv.loc[~hd, models].T.to_numpy(),
                                 uid[~hd].values, logit=True, n_boot=1)
                diffs.append(sr - sh)
            diffs = np.array(diffs)
            pp = wilcoxon(diffs).pvalue if len(diffs) > 5 else np.nan
            print(f"  stratum {int(q*100):>2}% (n={int((cons <= cons.quantile(q)).sum()):>3}), "
                  f"families with >={min_sizes} sizes: {len(diffs):>2}   "
                  f"mean(rest-hard) {diffs.mean():+.3f} log-odds/doubling   "
                  f"{(diffs > 0).sum()}/{len(diffs)} positive   p={pp:.3f}")
            sens.append(dict(stratum=q, min_sizes=min_sizes, n_families=len(diffs),
                             mean_diff_logodds=round(float(diffs.mean()), 4),
                             n_positive=int((diffs > 0).sum()), p=round(float(pp), 4)))
    save(pd.DataFrame(sens), "m2_stratum_sensitivity.csv")

    s = pd.DataFrame(sens)
    s10 = s[(s.stratum == 0.10) & (s.min_sizes == 2)].iloc[0]
    s10b = s[(s.stratum == 0.10) & (s.min_sizes == 3)].iloc[0]
    s30 = s[(s.stratum == 0.30) & (s.min_sizes == 2)].iloc[0]
    top = res.iloc[0]
    sat = res[res.family.eq("Qwen2.5")].iloc[0]

    para("M2 reasoner scaling", f"""
Holding the descriptions fixed and varying only the reasoner's scale asks what a
larger reasoner can buy at a given interface. {len(usable)} models across
{res.family.nunique()} families read byte-identical GPT-4o descriptions of the
same {piv.shape[0]} problems, at two to four sizes per family.

The first result is that scaling is strongly family-dependent, and its returns
depend on where the family starts. {top.family} gains
{top.slope_all:+.1f} points per doubling, climbing from {top.acc_small:.1f}\\% to
{top.acc_large:.1f}\\%; {sat.family}, which begins at {sat.acc_small:.1f}\\%,
gains {sat.slope_all:+.2f} points per doubling and reaches only
{sat.acc_large:.1f}\\% across a tenfold increase in parameters. Above roughly
90\\% on this interface the reasoner has ceased to be the binding constraint, and
buying a larger one is close to buying nothing.

The second question is sharper, and follows from
Section~\\ref{{sec:verbalization-cost}}: if some problems are hard because the
description discarded the evidence that would have settled them, no reasoner can
recover it by being larger, so scale should pay less on exactly those problems.
Measured in accuracy points the comparison is null ({d.mean():+.2f} points per
doubling, {(d > 0).sum()} of {len(d)} families, $p={p:.2f}$) --- but that
comparison is not sound, because the two strata sit at very different base rates
and the easier one is compressed against its ceiling. On the log-odds scale, the
predicted gap appears: scale is worth {s10.mean_diff_logodds:+.2f} log-odds per
doubling more on the remainder than on the hardest decile
({int(s10.n_positive)} of {int(s10.n_families)} families in the same direction,
$p={s10.p:.3f}$), rising to {s10b.mean_diff_logodds:+.2f}
({int(s10b.n_positive)}/{int(s10b.n_families)}, $p={s10b.p:.3f}$) among the six families
with three or more sizes, where a slope is more than a two-point difference. The
gap weakens monotonically as the stratum widens, to
{s30.mean_diff_logodds:+.2f} at the hardest 30\\%, which is the dose-response the
account predicts.

We report this as consistent with the interface account rather than as
established by it. The direction is unanimous or nearly so under every variant we
examined and the dose-response is orderly, but the evidence rests on six to
eleven families, the significance is borderline, and the conclusion depends on
analysing log-odds rather than accuracy --- a choice that is principled here,
since the strata differ in base rate by roughly fifty points, but a choice
nonetheless. What the data support without qualification is the weaker and still
useful claim: past a competence threshold, scaling the reasoner under a fixed
description interface buys very little, so a decoupled pipeline that
underperforms is unlikely to be repaired by upgrading its reasoner.
""")


if __name__ == "__main__":
    main()
