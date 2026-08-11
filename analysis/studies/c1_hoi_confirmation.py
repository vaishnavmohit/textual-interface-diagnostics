#!/usr/bin/env python3
"""C1/C3 — Does the Bongard-OpenWorld account hold on a second benchmark?

Bongard-HOI is the confirmation benchmark: six models cross three paradigms and
four splits, and the splits vary seen/unseen object against seen/unseen action
systematically, which Bongard-OpenWorld has no analogue for. Two things are
checked here, both derived from the main experiment rather than exploratory.

C1. The paradigm ordering, paired on common instances, with the sensitivity
    screen applied. Section~\\ref{sec:verbalization-cost} claims the description
    interface is a net aid; on a second benchmark it should still be one.

C3. Whether the advantage differs across the benchmark's four test splits.
    Reported as a null, and NOT as a novelty result. The seen/unseen labels
    partition categories with respect to the TRAINING set of the supervised
    few-shot learners Bongard-HOI was designed for. Nothing here is trained: the
    pipelines are prompted zero-shot and never see that training set, so the
    labels describe nothing about the systems we evaluate. An earlier version of
    this study read the split difference as novelty attenuation; that reading
    imports a training-based semantics into a training-free setting and has been
    withdrawn. The contrast is retained only to record that it was checked, and
    is computed as a proper interaction rather than by comparing two intervals.

Newer-generation runs (GPT-5.1, Gemini-3-Flash) are excluded here; they are the
subject of Section~\\ref{sec:generation-effect} and are not part of the
confirmation.

    python analysis/studies/c1_hoi_confirmation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RNG, boot_ci, bongard_valid, load, para, save  # noqa: E402

D_MIN = 0.6          # below this, accuracy is not a measure of task performance
NEWGEN = ["gpt-5.1", "gemini-3-flash-preview"]
SEEN_OBJ = ["sosa", "soua"]      # s-o = seen object
UNSEEN_OBJ = ["uosa", "uoua"]


POS = "cat_2"        # dataset convention: cat_2 = positive, cat_1 = negative


def sdt_dprime(g: pd.DataFrame) -> float:
    """d' with a log-linear correction, matching s4_acceptance_bias."""
    pos, neg = g[g.label_true == POS], g[g.label_true != POS]
    if not len(pos) or not len(neg):
        return float("nan")
    h = ((pos.pred_raw == POS).sum() + 0.5) / (len(pos) + 1)
    f = ((neg.pred_raw == POS).sum() + 0.5) / (len(neg) + 1)
    return float(norm.ppf(h) - norm.ppf(f))


def cell(h: pd.DataFrame, model: str, paradigm: str, split: str) -> pd.DataFrame:
    s = h[h.reasoner_model.eq(model) & h.paradigm.eq(paradigm) & h.split.eq(split)]
    return s.drop_duplicates("test_id").set_index("test_id")


def paired(h, model, p_a, p_b, splits):
    """(b - a) pooled over the given splits, paired within split."""
    diffs, n = [], 0
    for sp in splits:
        a, b = cell(h, model, p_a, sp), cell(h, model, p_b, sp)
        common = a.index.intersection(b.index)
        if len(common) < 20:
            continue
        diffs.append((b.loc[common].correct - a.loc[common].correct).to_numpy())
        n += len(common)
    if not diffs:
        return None
    d = np.concatenate(diffs)
    pt, (lo, hi) = boot_ci(d)
    return dict(n=n, delta=round(100 * pt, 1), ci_lo=round(100 * lo, 1),
                ci_hi=round(100 * hi, 1))


def split_diffs(h, model, p_a, p_b, splits):
    """Per-instance (b - a) over the given splits, unaggregated."""
    out = []
    for sp in splits:
        a, b = cell(h, model, p_a, sp), cell(h, model, p_b, sp)
        common = a.index.intersection(b.index)
        if len(common) >= 20:
            out.append((b.loc[common].correct - a.loc[common].correct).to_numpy())
    return np.concatenate(out) if out else None


def main() -> None:
    h = bongard_valid(load("bongard_hoi"))
    h = h[~h.reasoner_model.isin(NEWGEN)]

    models = sorted(h.reasoner_model.unique())
    splits = ["sosa", "soua", "uosa", "uoua"]

    # ---- sensitivity screen, per cell ------------------------------------ #
    scr = []
    for m in models:
        for p in ["DVRL", "DRL", "CA"]:
            s = h[h.reasoner_model.eq(m) & h.paradigm.eq(p)]
            if s.empty:
                continue
            dp = sdt_dprime(s.drop_duplicates(["split", "test_id"]))
            scr.append(dict(model=m, paradigm=p, n=s.test_id.nunique(),
                            acc=round(100 * s.correct.mean(), 1), d_prime=round(dp, 2),
                            measurable=bool(dp >= D_MIN)))
    screen = pd.DataFrame(scr)
    save(screen, "c1_hoi_screen.csv")
    print("=== sensitivity screen (all four splits pooled) ===")
    print(f"  {'model':<24}{'paradigm':<7}{'n':>5}{'acc':>7}{'d-prime':>9}   verdict")
    for _, r in screen.iterrows():
        print(f"  {r.model:<24}{r.paradigm:<7}{int(r.n):>5}{r.acc:>7.1f}{r.d_prime:>9.2f}"
              f"   {'' if r.measurable else 'NOT A MEASUREMENT'}")

    ok = {(r.model, r.paradigm) for _, r in screen.iterrows() if r.measurable}

    # ---- C1: paradigm ordering, paired ----------------------------------- #
    rows = []
    for m in models:
        for a, b in [("DVRL", "DRL"), ("DRL", "CA"), ("DVRL", "CA")]:
            if (m, a) not in ok or (m, b) not in ok:
                continue
            r = paired(h, m, a, b, splits)
            if r:
                rows.append(dict(model=m, contrast=f"{b} - {a}", **r))
    c1 = pd.DataFrame(rows)
    save(c1, "c1_paradigm_ordering.csv")
    print("\n=== C1. Paradigm ordering, paired on common instances (screened cells only) ===")
    print(f"  {'model':<24}{'contrast':<14}{'n':>5}{'delta':>8}{'95% CI':>18}")
    for _, r in c1.iterrows():
        print(f"  {r.model:<24}{r.contrast:<14}{int(r.n):>5}{r.delta:>+8.1f}"
              f"   [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}]")

    # ---- C3: does the advantage differ across split groups? -------------- #
    # Tested as an interaction with its own bootstrap interval. Comparing the two
    # groups' separate CIs -- as an earlier version did -- is the overlapping-CI
    # fallacy and can report a difference the data do not support.
    rows = []
    for m in models:
        if (m, "DVRL") not in ok or (m, "CA") not in ok:
            continue
        s_d = split_diffs(h, m, "DVRL", "CA", SEEN_OBJ)
        u_d = split_diffs(h, m, "DVRL", "CA", UNSEEN_OBJ)
        if s_d is None or u_d is None:
            continue
        boots = [RNG.choice(s_d, len(s_d), True).mean() - RNG.choice(u_d, len(u_d), True).mean()
                 for _ in range(4000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(dict(model=m, seen_obj=round(100 * s_d.mean(), 1),
                         unseen_obj=round(100 * u_d.mean(), 1),
                         interaction=round(100 * (s_d.mean() - u_d.mean()), 1),
                         ci_lo=round(100 * lo, 1), ci_hi=round(100 * hi, 1),
                         excludes_zero=bool(lo > 0 or hi < 0)))
    c3 = pd.DataFrame(rows)
    save(c3, "c3_split_group_interaction.csv")
    print("\n=== C3. Does the advantage differ across split groups? (NOT a novelty test) ===")
    print(f"  {'model':<24}{'seen-obj':>10}{'unseen-obj':>12}{'interaction':>13}{'95% CI':>18}")
    for _, r in c3.iterrows():
        print(f"  {r.model:<24}{r.seen_obj:>+10.1f}{r.unseen_obj:>+12.1f}"
              f"{r.interaction:>+13.1f}   [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}]"
              f"{'' if r.excludes_zero else '   n.s.'}")

    g4 = c1[c1.model.eq("gpt-4o-2024-08-06")]
    ca_dv = g4[g4.contrast.eq("CA - DVRL")]
    a3 = c3[c3.model.eq("gpt-4o-2024-08-06")]
    s3 = c3[c3.model.eq('gpt-4o-2024-08-06')]
    gsc = screen[screen.model.eq("gemini-2.0-flash-exp")].set_index("paradigm").d_prime
    gem_dv, gem_drl = gsc.get("DVRL", float("nan")), gsc.get("DRL", float("nan"))

    para("C1/C3 Bongard-HOI confirmation", f"""
The account developed on Bongard-OpenWorld should not be a fact about one
benchmark. Bongard-HOI provides the check, and adds a manipulation
Bongard-OpenWorld lacks: its four splits cross seen and unseen objects with seen
and unseen actions, so the interface can be evaluated against novelty.

The sensitivity screen is applied first, and it is consequential. Gemini~2.0
under direct visual reasoning has $d' = {gem_dv:.2f}$ --- near-constant responding,
not task performance --- and its rule-verbalization cell reaches only
{gem_drl:.2f}, below the same threshold. Both are excluded, which leaves Gemini
with a single admissible cell and no paradigm contrast at all. The confirmation
therefore rests on GPT-4o, and we state that rather than presenting a two-model
replication that the data do not support. We also note that the second exclusion
is marginal: at $d' = {gem_drl:.2f}$ against a threshold of {D_MIN}, it is a cell
the screen declines to trust rather than one that clearly fails, and a slightly
looser criterion would admit it.

The interface advantage replicates. Componential analysis beats direct visual
reasoning for GPT-4o by {ca_dv.delta.iloc[0]:+.1f} points on {int(ca_dv.n.iloc[0])} paired
instances (95\\% CI [{ca_dv.ci_lo.iloc[0]:+.1f}, {ca_dv.ci_hi.iloc[0]:+.1f}]),
against the $+12.8$ measured on Bongard-OpenWorld
(Section~\\ref{{sec:verbalization-cost}}). Describing the images and reasoning
over the text is again better than reasoning from the images, on a second
benchmark, a different task, and different instances.

We pool across the four test splits rather than analysing them separately, and
the reason generalises to any training-free evaluation on this benchmark. The
\\textit{{seen}}/\\textit{{unseen}} designations partition object and action
categories with respect to the \\emph{{training}} set of the supervised few-shot
learners the benchmark was built for. Our pipelines are prompted zero-shot and
never see that training set, so the designations describe nothing about the
systems evaluated here --- the categories are unseen by models we do not run.
Reading the splits as a novelty manipulation would import a training-based
semantics into a setting with no training.

The split difference is in any case not established. The interface advantage is
{s3.seen_obj.iloc[0]:+.1f} points on the seen-object splits and
{s3.unseen_obj.iloc[0]:+.1f} on the unseen-object splits, but the interaction
itself is {s3.interaction.iloc[0]:+.1f} with a 95\\% CI of
[{s3.ci_lo.iloc[0]:+.1f}, {s3.ci_hi.iloc[0]:+.1f}]. Comparing the two groups'
separate intervals, as an earlier version of this analysis did, is not a test of
the difference between them.
""")


if __name__ == "__main__":
    main()
