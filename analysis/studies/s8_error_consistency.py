#!/usr/bin/env python3
"""S8 — Which component makes two systems fail on the same items?

S3 established that failures are largely model-specific: there is no shared
difficulty dimension and no item that defeats every model. That leaves the
question of what *does* produce agreement when it occurs.

The crossed cell answers it, because every (description source, reasoner) pair
is observed on the same 137 items. Three kinds of pair exist:

    same reasoner, different description source   -> agreement attributable to
                                                     the reasoner
    same description source, different reasoner   -> agreement attributable to
                                                     the descriptions
    both different                                -> baseline

Whichever kind agrees most identifies the component that carries the error
structure. This is the same decomposition as S2 but on a different quantity: S2
asked which component moves the *accuracy*, this asks which component determines
*which items* are missed. They can disagree, and if they do that is informative:
a component can shift the score without changing what is hard.

Significance is assessed by a permutation test over pair labels, since the
correlations are not independent (they share cells).

    python analysis/studies/s8_error_consistency.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load, bongard_valid, save, para  # noqa: E402

RNG = np.random.default_rng(42)


def crossed_matrix() -> pd.DataFrame:
    d = bongard_valid(load("bongard_ow"))
    ca = d[(d.paradigm == "CA") & d.components.isin(["gpt-4o", "pixtral"])]
    both = [m for m, g in ca.groupby("reasoner_model") if g.components.nunique() == 2]
    ca = ca[ca.reasoner_model.isin(both)]
    common = set.intersection(*[set(g.test_id) for _, g in ca.groupby(["components", "reasoner_model"])])
    ca = ca[ca.test_id.isin(common)]
    return ca.pivot_table(index="test_id", columns=["components", "reasoner_model"], values="correct")


def classify_pairs(w: pd.DataFrame) -> pd.DataFrame:
    C = w.corr()
    cols = list(w.columns)
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            (s1, r1), (s2, r2) = cols[i], cols[j]
            kind = ("same reasoner" if r1 == r2 else
                    "same descriptions" if s1 == s2 else "both differ")
            rows.append(dict(kind=kind, a=f"{s1}/{r1}", b=f"{s2}/{r2}",
                             phi=round(float(C.iloc[i, j]), 4)))
    return pd.DataFrame(rows)


def permutation_test(w: pd.DataFrame, n_iter: int = 2000) -> dict:
    """Is 'same reasoner' agreement higher than 'same descriptions' by chance?

    Permutes the reasoner labels within each description source, which destroys
    the pairing between the two halves of the design while preserving each
    cell's accuracy and the item structure.
    """
    obs = classify_pairs(w)
    stat_obs = (obs[obs.kind == "same reasoner"].phi.mean()
                - obs[obs.kind == "same descriptions"].phi.mean())
    srcs = sorted({c[0] for c in w.columns})
    reas = sorted({c[1] for c in w.columns})
    count = 0
    for _ in range(n_iter):
        perm = w.copy()
        mapping = {s: dict(zip(reas, RNG.permutation(reas))) for s in srcs}
        perm.columns = pd.MultiIndex.from_tuples([(s, mapping[s][r]) for s, r in w.columns])
        p = classify_pairs(perm)
        stat = (p[p.kind == "same reasoner"].phi.mean()
                - p[p.kind == "same descriptions"].phi.mean())
        count += stat >= stat_obs
    return dict(statistic=round(float(stat_obs), 4), p_value=round((count + 1) / (n_iter + 1), 4))


def main() -> None:
    w = crossed_matrix()
    print(f"crossed matrix: {w.shape[0]} items x {w.shape[1]} (description source, reasoner) cells\n")

    pairs = classify_pairs(w)
    save(pairs, "s8_pair_agreement.csv")
    summary = pairs.groupby("kind").phi.agg(["count", "mean", "median", "std"]).round(3)
    summary = summary.reindex(["same reasoner", "same descriptions", "both differ"])
    print("=== Error agreement by what the pair shares ===")
    print(f"  {'shares':<20}{'pairs':>7}{'mean phi':>10}{'median':>9}{'sd':>8}")
    for k, r in summary.iterrows():
        print(f"  {k:<20}{int(r['count']):>7}{r['mean']:>10.3f}{r['median']:>9.3f}{r['std']:>8.3f}")

    pt = permutation_test(w)
    print(f"\n  same-reasoner minus same-descriptions = {pt['statistic']:+.3f}"
          f"   permutation p = {pt['p_value']:.4f}")
    save(pd.DataFrame([pt]), "s8_permutation_test.csv")

    # which reasoner pairs agree most, holding the description source fixed
    sd = pairs[pairs.kind == "same descriptions"].copy()
    sd["src"] = sd.a.str.split("/").str[0]
    print("\n=== Agreement between reasoners, by description source ===")
    for s, g in sd.groupby("src"):
        print(f"  {s:<9} mean phi = {g.phi.mean():+.3f}  (n={len(g)} reasoner pairs)")

    sr = pairs[pairs.kind == "same reasoner"]
    print("\n=== Same reasoner across description sources ===")
    for _, r in sr.sort_values("phi", ascending=False).iterrows():
        print(f"  {r.a.split('/')[1]:<24} phi = {r.phi:+.3f}")

    m_reas = summary.loc["same reasoner", "mean"]
    m_desc = summary.loc["same descriptions", "mean"]
    m_none = summary.loc["both differ", "mean"]

    para("S8 error consistency", f"""
Which component determines \\emph{{which}} problems are missed, as opposed to how
many? Every combination of description source and reasoner in the crossed cell
is evaluated on the same {w.shape[0]} problems, so pairs of systems can be grouped by what
they share.

Systems that share a reasoner but read descriptions from different perception
models agree on their errors at $\\phi={m_reas:.3f}$. Systems that share a description
source but use different reasoners agree at $\\phi={m_desc:.3f}$, and systems sharing
neither at $\\phi={m_none:.3f}$ (permutation test on the difference, $p={pt['p_value']:.3f}$).

The error structure therefore travels with the reasoner more than with the
description source: change the perception model and a given reasoner keeps
missing much the same problems, whereas swapping the reasoner changes the error
set more. Read alongside the accuracy decomposition, in which the two components
have comparable effects, this gives a more precise statement than either alone.
The description source shifts \\emph{{how many}} problems are solved without much
changing \\emph{{which}} ones; the reasoner determines which. Improving
descriptions raises the level uniformly, but the set of problems a pipeline
finds hard is a property of the reasoning stage.

All agreement is modest in absolute terms ($\\phi<0.4$ throughout), consistent
with Section~\\ref{{sec:results-components}}: even systems sharing a component
fail largely independently.
""")


if __name__ == "__main__":
    main()
