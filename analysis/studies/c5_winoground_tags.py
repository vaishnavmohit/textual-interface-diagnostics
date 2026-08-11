#!/usr/bin/env python3
"""C5 — Does the interface lose relational evidence specifically?

Section~\\ref{sec:verbalization-cost} shows the description interface owns a set
of failures the pixels do not share, but not what is in them. Winoground can name
it, because every item is labelled by what distinguishes its two captions: the
objects involved (Object), the relation between them (Relation), or both.

The prediction is directional. A description schema enumerates what is present
more reliably than how the parts are configured, so the evidence it drops should
be relational. Two contrasts test this without needing new runs:

  1. Re-inspection. Letting the reasoner ask for another look at the image
     recovers evidence the description omitted. If the omission is relational,
     the gain should concentrate on Relation items.
  2. Perception source. Swapping which model writes the descriptions, with the
     reasoner fixed, should likewise matter more where relational detail decides
     the item.

Winoground has no usable end-to-end (pixel) condition -- three rows -- so the
direct M1 contrast cannot be replicated here; these two are its available
substitutes and are weaker for it, since both still read a description.

Scores: text (caption-to-image), image (image-to-caption) and group (both
correct). Group is the strict measure and is reported first.

    python analysis/studies/c5_winoground_tags.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RNG, load, para, save  # noqa: E402

SCORES = ["group_score", "text_score", "image_score"]
TAGS = ["Relation", "Object"]


def cell(w: pd.DataFrame, condition: str, reasoner: str, perception: str | None = None):
    s = w[w.condition.eq(condition) & w.reasoner_model.eq(reasoner)]
    if perception is not None:
        s = s[s.perception.eq(perception)]
    return s.drop_duplicates("id").set_index("id")


def interaction(a: pd.DataFrame, b: pd.DataFrame, score: str, n_boot: int = 4000):
    """(b - a) on Relation minus (b - a) on Object, paired by item.

    Bootstrap resamples items within each tag, so the interval reflects item
    sampling rather than an asymptotic approximation on 141-233 items.
    """
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    tag = a.collapsed_tag
    out = {}
    for t in TAGS:
        m = tag.eq(t)
        out[t] = (b.loc[m, score].to_numpy(dtype=float) - a.loc[m, score].to_numpy(dtype=float))
    if not len(out["Relation"]) or not len(out["Object"]):
        return None
    point = out["Relation"].mean() - out["Object"].mean()
    boots = [RNG.choice(out["Relation"], len(out["Relation"]), replace=True).mean()
             - RNG.choice(out["Object"], len(out["Object"]), replace=True).mean()
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(score=score, n_relation=len(out["Relation"]), n_object=len(out["Object"]),
                delta_relation=round(100 * out["Relation"].mean(), 1),
                delta_object=round(100 * out["Object"].mean(), 1),
                interaction=round(100 * point, 1),
                ci_lo=round(100 * lo, 1), ci_hi=round(100 * hi, 1))


def main() -> None:
    w = load("winoground")
    base = w.drop_duplicates("id").set_index("id")
    print("Winoground items by tag:",
          base.collapsed_tag.value_counts().to_dict(), "(Both excluded: ambiguous)")

    rows = []

    # --- 1. re-inspection: ICA - CA, same reasoner, same descriptions --------
    # GPT-4o is the clean pair: both conditions read gpt-4o descriptions and the
    # reasoner label is identical. The Gemini pair is labelled 'gemini-2.0-flash'
    # under ICA and 'gemini-2.0-flash-exp' under CA; we report it but flag it.
    pairs = [("gpt-4o-2024-08-06", "gpt-4o-2024-08-06", "GPT-4o", True),
             ("gemini-2.0-flash-exp", "gemini-2.0-flash", "Gemini 2.0", False)]
    print("\n=== 1. Re-inspection (ICA - CA), by what distinguishes the captions ===")
    print(f"  {'model':<12}{'score':<14}{'Relation':>10}{'Object':>9}{'interaction':>13}{'95% CI':>18}")
    for ca_name, ica_name, label, clean in pairs:
        a = cell(w, "CA", ca_name, "gpt-4o")
        b = cell(w, "ICA", ica_name)
        if a.empty or b.empty:
            continue
        for sc in SCORES:
            r = interaction(a, b, sc)
            if r is None:
                continue
            print(f"  {label:<12}{sc.replace('_score',''):<14}{r['delta_relation']:>+10.1f}"
                  f"{r['delta_object']:>+9.1f}{r['interaction']:>+13.1f}"
                  f"   [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]"
                  f"{'' if clean else '   (label mismatch)'}")
            rows.append(dict(contrast="reinspection", model=label, exact_pair=clean, **r))

    # --- 2. perception source, reasoner fixed --------------------------------
    print("\n=== 2. Perception source (GPT-4o descriptions - Gemini descriptions) ===")
    print(f"  {'model':<12}{'score':<14}{'Relation':>10}{'Object':>9}{'interaction':>13}{'95% CI':>18}")
    for reasoner, label in [("gpt-4o-2024-08-06", "GPT-4o"),
                            ("gemini-2.0-flash-exp", "Gemini 2.0")]:
        a = cell(w, "CA", reasoner, "gemini")
        b = cell(w, "CA", reasoner, "gpt-4o")
        if a.empty or b.empty:
            continue
        for sc in SCORES:
            r = interaction(a, b, sc)
            if r is None:
                continue
            print(f"  {label:<12}{sc.replace('_score',''):<14}{r['delta_relation']:>+10.1f}"
                  f"{r['delta_object']:>+9.1f}{r['interaction']:>+13.1f}"
                  f"   [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]")
            rows.append(dict(contrast="perception_source", model=label, exact_pair=True, **r))

    res = pd.DataFrame(rows)
    save(res, "c5_winoground_tags.csv")

    # Absolute difficulty by tag, for context: are Relation items simply harder?
    ca = cell(w, "CA", "gpt-4o-2024-08-06", "gpt-4o")
    lvl = ca.groupby("collapsed_tag")[SCORES].mean().mul(100).round(1)
    print("\n=== baseline difficulty by tag (GPT-4o, CA) ===")
    print(lvl.to_string())

    g = res[res.score.eq("group_score")]
    ri = g[g.contrast.eq("reinspection") & g.exact_pair]
    ps = g[g.contrast.eq("perception_source")]
    sig = res[(res.ci_lo > 0) | (res.ci_hi < 0)]

    para("C5 Winoground tags", f"""
Section~\\ref{{sec:verbalization-cost}} establishes that the description
interface owns a set of failures the pixels do not share, but not what those
failures contain. Winoground labels every item by what distinguishes its two
captions --- the objects involved, the relation between them, or both --- which
allows a directional test. A schema that enumerates what is present should record
objects more reliably than configurations, so the evidence it drops ought to be
relational, and interventions that restore access to the image ought to pay off
on Relation items specifically.

The prediction is not confirmed. Letting the reasoner request another look at the
image (the interactive condition against componential analysis, same reasoner and
same descriptions) helps Relation items by
{ri.delta_relation.iloc[0]:+.1f} points and Object items by
{ri.delta_object.iloc[0]:+.1f} on the strict group score, an interaction of
{ri.interaction.iloc[0]:+.1f} points (95\\% CI [{ri.ci_lo.iloc[0]:+.1f},
{ri.ci_hi.iloc[0]:+.1f}]). Changing which model writes the descriptions, with the
reasoner held fixed, gives interactions of
{', '.join(f'{v:+.1f}' for v in ps.interaction)} points, likewise with intervals
spanning zero. Across all four model-by-contrast pairs and all three scores,
{len(sig)} of the {len(res)} tag interactions exclude zero.

Relation items are harder --- {lvl.loc['Relation', 'group_score']:.1f}\\% against
{lvl.loc['Object', 'group_score']:.1f}\\% on the group score --- but they are not
\\emph{{differentially}} helped by restoring visual access or by better
descriptions. The interface's cost does not appear to be concentrated in
relational content, at least not along the axis this benchmark labels.

We report this as a negative result and note what it does not establish. Neither
contrast here removes the description: both arms of every comparison still read
text, so this tests where \\emph{{more}} or \\emph{{better}} description helps
rather than what verbalization costs against raw pixels, which is the contrast
that produced the effect on Bongard-OpenWorld. Winoground has no usable
end-to-end condition to supply that comparison. The Object/Relation distinction
is also coarse and was designed to characterise compositional language rather
than perceptual describability. Naming the content of the interface's failure set
therefore remains open, and we identify it as the natural target for the
annotation described in Section~\\ref{{sec:reliability}}.
""")


if __name__ == "__main__":
    main()
