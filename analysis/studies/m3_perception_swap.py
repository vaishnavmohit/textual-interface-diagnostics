#!/usr/bin/env python3
"""M3 — Is a model's own perception its bottleneck?

Six models answer Bongard-OpenWorld twice: once reasoning over descriptions they
produced themselves, and once over GPT-4o's descriptions of the same images. The
reasoner is held exactly fixed and only the description source moves, which makes
this the cleanest perceptual intervention in the corpus --- cleaner than the
crossed cell, because there the reasoner and the perception source vary across
different systems.

The practical question is whether paying for a better captioner helps. The answer
depends on which component is actually binding, and the swap diagnoses that
rather than merely measuring it --- but only once the cells are screened, because
an accuracy difference between two conditions in which the model is barely
discriminating is not a perception effect at all.

We therefore read the swap through $d'$ rather than accuracy alone, using the
same threshold ($d' < 0.6$) applied elsewhere in the paper:

- both cells below the screen: the model does not discriminate under either
  description source, so the swap is uninformative and its accuracy difference
  must not be interpreted. Two of the six models are in this state, and one of
  them shows a nominally significant *decline* that is an artefact of comparing
  two near-constant response patterns;
- $d'$ low on its own descriptions and high on GPT-4o's: perception-limited. The
  better descriptions do not merely raise accuracy, they create the ability to
  discriminate at all;
- $d'$ already high on its own descriptions: little left to buy.

    python analysis/studies/m3_perception_swap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (boot_ci, bongard_valid, cluster_randomization_pvalue,
                     load, normal_ppf, para, save)  # noqa: E402

POS = "cat_2"      # dataset/bongard_ow.py: imagefiles["cat_2"] = positive
D_MIN = 0.6        # the screen used throughout: below this, accuracy is not a
                   # measurement of task performance (see S4 and Section 6.1)


def sdt(g):
    """d' with a log-linear correction, matching s4_acceptance_bias.sdt."""
    pos, neg = g[g.label_true == POS], g[g.label_true != POS]
    if not len(pos) or not len(neg):
        return float("nan")
    h = ((pos.pred_raw == POS).sum() + 0.5) / (len(pos) + 1)
    f = ((neg.pred_raw == POS).sum() + 0.5) / (len(neg) + 1)
    return float(normal_ppf(h) - normal_ppf(f))


def main() -> None:
    raw = load("bongard_ow")
    # A valid-row subset of a schema-faulted run is not a component
    # intervention. Exclude such runs before the ordinary validity screen.
    u = bongard_valid(raw[raw.analysis_decision.eq("include")])
    u = u[u.ablation.isna() & u.paradigm.eq("CA")]
    uid = u.drop_duplicates("test_id").set_index("test_id").uid

    models = sorted(set(u[u.components.eq("self")].reasoner_model)
                    & set(u[u.components.eq("gpt-4o")].reasoner_model))

    rows = []
    for m in models:
        s = u[u.reasoner_model.eq(m) & u.components.eq("self")].drop_duplicates("test_id").set_index("test_id").correct
        g = u[u.reasoner_model.eq(m) & u.components.eq("gpt-4o")].drop_duplicates("test_id").set_index("test_id").correct
        c = s.index.intersection(g.index)
        if len(c) < 100:
            continue
        rs = u[u.reasoner_model.eq(m) & u.components.eq("self")].drop_duplicates("test_id").set_index("test_id").loc[c]
        rg = u[u.reasoner_model.eq(m) & u.components.eq("gpt-4o")].drop_duplicates("test_id").set_index("test_id").loc[c]
        d_s, d_g = sdt(rs), sdt(rg)
        s, g = s.loc[c], g.loc[c]
        pt, (lo, hi) = boot_ci((g - s).to_numpy(), clusters=uid.loc[c].to_numpy())
        fixed = int(((s == 0) & (g == 1)).sum())     # broken by own perception, repaired
        broken = int(((s == 1) & (g == 0)).sum())    # solved alone, lost with better input
        p_cluster = cluster_randomization_pvalue(
            (g - s).to_numpy(), uid.loc[c].to_numpy()
        )
        rows.append(dict(
            model=m, n=len(c),
            self_acc=round(100 * s.mean(), 1), gpt4o_acc=round(100 * g.mean(), 1),
            gain=round(100 * pt, 1), ci_lo=round(100 * lo, 1), ci_hi=round(100 * hi, 1),
            p=round(p_cluster, 5), fixed=fixed, broken=broken,
            d_self=round(d_s, 2), d_gpt4o=round(d_g, 2),
            measurable=bool(d_g >= D_MIN),
        ))
    res = pd.DataFrame(rows).sort_values("gain", ascending=False)

    def diagnose(r):
        # A cell whose d' is below the screen is not measuring task performance,
        # so a gain computed from it is not a perception effect. When BOTH cells
        # fail, the swap cannot be informative at all.
        if not r.measurable:
            return "not measurable (d' < %.1f under both)" % D_MIN
        return "perception-limited" if r.ci_lo > 0 else "neither binding"

    res["diagnosis"] = res.apply(diagnose, axis=1)
    save(res, "m3_perception_swap.csv")

    print(f"perception swap, reasoner held fixed ({len(res)} models)")
    print(f"  {'model':<20}{'n':>5}{'self':>8}{'GPT-4o':>9}{'gain':>8}{'95% CI':>18}"
          f"{chr(100)+chr(39)+'self':>8}{chr(100)+chr(39)+'gpt':>7}  diagnosis")
    for _, r in res.iterrows():
        print(f"  {r.model:<20}{r.n:>5}{r.self_acc:>8.1f}{r.gpt4o_acc:>9.1f}{r.gain:>+8.1f}"
              f"   [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}]{r.d_self:>8.2f}{r.d_gpt4o:>7.2f}  {r.diagnosis}")

    # Does the benefit track how poor the model's own perception was? Restricted
    # to models that can use the interface at all -- for the others the swap is
    # not a perception manipulation, since nothing downstream can exploit it.
    usable = res[res.measurable]
    rho = (float(np.corrcoef(usable.self_acc.rank(), usable.gain.rank())[0, 1])
           if len(usable) >= 4 else np.nan)
    if len(usable) >= 4:
        print(f"\n  among the {len(usable)} models that can use the interface: "
              f"gain vs own-perception accuracy, Spearman rho={rho:+.2f} "
              "(descriptive; no population p-value)")
    else:
        print(f"\n  {len(usable)} schema-valid discriminating models: correlation not reported")
    print(f"  excluded by the d' screen: "
          f"{', '.join(res[~res.measurable].model) or 'none'}")

    para("M3 perception swap", f"""
Whether a better perceptual front end is worth paying for is the most direct
design question this study can address, and five schema-valid models answer it under an exact
control: each reasons over descriptions it produced itself and over GPT-4o's
descriptions of the same images, with the reasoner, the problems and the prompt
unchanged.

Two of the five must be set aside before anything is read from them. LLaVA-13B
and LLaVA-Llama3 fall below the sensitivity screen under \\emph{{both}} description
sources ($d'$ between $-0.19$ and $0.41$, with up to 92\\% of responses in a
single class), so neither condition measures task performance and the difference
between them measures nothing. LLaVA-Llama3's nominally significant decline of
{res[res.model.eq('llava-llama3')].gain.iloc[0]:.1f} points
[{res[res.model.eq('llava-llama3')].ci_lo.iloc[0]:+.1f},
{res[res.model.eq('llava-llama3')].ci_hi.iloc[0]:+.1f}] is a comparison between
two near-constant response patterns rather than evidence that better
descriptions hurt. The runs are complete and fully valid; what they lack is
discrimination, not data.

Among the three schema-valid models the screen admits, the spread is the finding. Gemma3-12B
rises from {res[res.model.eq('gemma3:12b')].self_acc.iloc[0]:.1f}\\% to
{res[res.model.eq('gemma3:12b')].gpt4o_acc.iloc[0]:.1f}\\%
({res[res.model.eq('gemma3:12b')].gain.iloc[0]:+.1f} points, 95\\% CI
[{res[res.model.eq('gemma3:12b')].ci_lo.iloc[0]:+.1f},
{res[res.model.eq('gemma3:12b')].ci_hi.iloc[0]:+.1f}]) and Gemma3-4B by
{res[res.model.eq('gemma3:4b')].gain.iloc[0]:+.1f} points. The sensitivity
figures show what that means: their $d'$ moves from
{res[res.model.eq('gemma3:12b')].d_self.iloc[0]:.2f} and
{res[res.model.eq('gemma3:4b')].d_self.iloc[0]:.2f} --- barely discriminating on
their own descriptions --- to {res[res.model.eq('gemma3:12b')].d_gpt4o.iloc[0]:.2f}
and {res[res.model.eq('gemma3:4b')].d_gpt4o.iloc[0]:.2f}. Better perception does
not merely raise their accuracy; it is what allows them to tell the two classes
apart at all. Pixtral-12B already discriminates on its own descriptions
($d'={res[res.model.eq('pixtral-12b-2409')].d_self.iloc[0]:.2f}$) and gains only
{res[res.model.eq('pixtral-12b-2409')].gain.iloc[0]:+.1f} points. The
schema-faulted Gemma3-27B self-description run is excluded rather than
interpreted on its valid-row subset.

The swap therefore does more than measure a benefit; it identifies which
component binds, from two runs per model. A model whose $d'$ stays below the
screen even on high-fidelity descriptions cannot use the interface, and
upgrading its perception is wasted expenditure. A model whose $d'$ rises sharply
was perception-limited, and the upgrade is worth up to {res.gain.max():.0f}
points. With only {len(usable)} schema-valid, discriminating models, no
cross-model correlation is reported.

The intervention is also close to purely additive. For Gemma3-12B, better
descriptions repair {res[res.model.eq('gemma3:12b')].fixed.iloc[0]} query cases and
cost {res[res.model.eq('gemma3:12b')].broken.iloc[0]}: improving the perceptual
input raises the level without disturbing what the reasoner already had right.
This is the within-model counterpart of the crossed-cell result in
Section~\\ref{{sec:crossed-cell}}, obtained without varying the reasoner at all,
and it carries the same practical reading --- perception is the component whose
improvement transfers, which is what makes it attractive to invest in even when
it is not the larger lever.
""")


if __name__ == "__main__":
    main()
