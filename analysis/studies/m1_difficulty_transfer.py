#!/usr/bin/env python3
"""M1 — Does difficulty survive the interface?

The paper's framework assumes that describing an image and then reasoning over
the description can lose information the pixels carried. Section 10 concedes
this is unmeasured. Bongard-OpenWorld can measure it, because 42 reasoners read
byte-identical GPT-4o descriptions of the same 499 problems while two models
answer the same problems from the raw images.

If the verbalization interface is what makes hard problems hard, the problems
that defeat the description readers should be recoverable by a model that looks
at the pixels. If instead those problems are intrinsically hard, difficulty
should transfer across the change of representation.

The answer is neither, and the four parts below are what it takes to see that:
the interface is a net aid of about twelve points (Part 3) while still owning a
private set of failures that the pixels do not share (Parts 2 and 4).

Three controls carry the argument, and each was added because the version
without it gives a different answer:

1. Selecting the hardest items by consensus and then scoring a different system
   on them guarantees an upward move for purely statistical reasons. So the hard
   set is defined on a random half of the reasoner pool and measured on the
   held-out half.
2. How far a system falls on hard items depends on where it sits: near-chance
   readers here drop ~0 points and competent ones drop 45-66. Comparing a pixel
   model against the pool average therefore flatters it (the naive contrast gives
   +7 where the level-matched one gives +25). We fit drop against level across
   the held-out readers and read off the expectation at the pixel model's own
   accuracy.
3. An asymmetry claim needs the reverse direction, so Part 4 defines the hard set
   from a pixel model's errors and asks the same question backwards.

    python analysis/studies/m1_difficulty_transfer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RNG, boot_ci, bongard_valid, load, para, save  # noqa: E402

DESC = "gpt-4o"          # the shared perception source for the sweep
HARD_Q = 0.10            # bottom decile defines "hard"
N_SPLITS = 400           # random half-splits of the reasoner pool
BAND = 0.05              # match held-out readers to within 5 accuracy points
PIXEL_MODELS = ["gpt-4o", "gemini-2.0-flash-exp"]   # have DVRL on the same items


def ca_matrix(u: pd.DataFrame) -> pd.DataFrame:
    """items x reasoners, 1/0, over the shared description artifact."""
    ca = u[u.paradigm.eq("CA") & u.components.eq(DESC)]
    ca = ca.drop_duplicates(["reasoner_model", "test_id"])
    piv = ca.pivot(index="test_id", columns="reasoner_model", values="correct")
    return piv.dropna().astype(float)      # items every reasoner answered


def pixel_series(u: pd.DataFrame, model: str, items) -> pd.Series:
    dv = u[u.paradigm.eq("DVRL") & u.reasoner_model.eq(model)]
    dv = dv.drop_duplicates("test_id").set_index("test_id").correct
    return dv.reindex(items).dropna()


def main() -> None:
    u = load("bongard_ow")
    u = u[u.analysis_decision.eq("include")].copy()
    # Delivered-system scoring: an invalid model output in an otherwise valid
    # run is an issued query answered incorrectly.
    u["correct"] = u.is_correct.fillna(False).astype(float)
    u = u[u.ablation.isna()]
    piv = ca_matrix(u)
    reasoners = list(piv.columns)
    n_r, n_i = len(reasoners), len(piv)
    print(f"basis: {n_r} reasoners x {n_i} items on identical {DESC} descriptions")

    # ---- Part 1: is item difficulty a shared property at all? -------------
    # If each system had its own idiosyncratic difficulty, split-half consensus
    # would not correlate. This is the precondition for everything after it.
    rhos = []
    for _ in range(N_SPLITS):
        p = RNG.permutation(reasoners)
        a, b = list(p[: n_r // 2]), list(p[n_r // 2:])
        rhos.append(spearmanr(piv[a].mean(axis=1), piv[b].mean(axis=1)).statistic)
    rho = float(np.mean(rhos))
    # Spearman-Brown: correlation of two half-pools -> reliability of the full pool
    rel = 2 * rho / (1 + rho)
    print(f"\nPart 1 — split-half difficulty correlation rho={rho:.3f} "
          f"(full-pool reliability {rel:.3f} by Spearman-Brown)")

    # ---- Part 2: held-out, accuracy-matched control ----------------------
    # Split the pool; define the hard set on half A; measure how far accuracy
    # falls on it for the held-out readers B and for each pixel-seeing model.
    #
    # Two corrections are needed for this to mean anything. Comparing DROPS
    # rather than levels removes the systems' different overall accuracies. And
    # the drop must be compared against readers of COMPARABLE STRENGTH: the pool
    # spans 47% to 93%, so its average drop is dominated by near-chance models
    # that have nothing to lose, which would flatter any competent pixel model.
    pix = {m: pixel_series(u, m, piv.index) for m in PIXEL_MODELS}
    pix = {m: s for m, s in pix.items() if len(s) > 0.9 * n_i}
    print(f"        pixel models with coverage: "
          f"{ {m: len(s) for m, s in pix.items()} }")

    rows = []
    for _ in range(N_SPLITS):
        p = RNG.permutation(reasoners)
        a, b = list(p[: n_r // 2]), list(p[n_r // 2:])
        cons_a = piv[a].mean(axis=1)
        hard = cons_a <= cons_a.quantile(HARD_Q)
        hard_i, easy_i = piv.index[hard], piv.index[~hard]

        # per held-out reasoner: its level (on the easy items) and its drop
        base_b = piv.loc[easy_i, b].mean()
        drop_b = base_b - piv.loc[hard_i, b].mean()

        rec = {"drop_heldout_all": float(drop_b.mean()), "n_hard": int(hard.sum())}
        # How far a reader falls on the hard decile depends on where it sits: a
        # near-chance model has little to lose and a near-ceiling one little room
        # to fall, so the relation is curved. Fit drop against level across all
        # held-out readers and read off the expectation at the pixel model's own
        # level. This uses the whole held-out half rather than the two or three
        # readers that happen to land in a narrow accuracy band.
        coef = np.polyfit(base_b.to_numpy(), drop_b.to_numpy(), 2)
        for m, s in pix.items():
            hi, ei = s.index.intersection(hard_i), s.index.intersection(easy_i)
            base_m = float(s.loc[ei].mean())
            drop_m = base_m - float(s.loc[hi].mean())
            near = (base_b - base_m).abs() <= BAND
            rec[f"drop_{m}"] = drop_m
            rec[f"level_{m}"] = base_m
            rec[f"expected_{m}"] = float(np.polyval(coef, base_m))
            rec[f"excess_{m}"] = rec[f"expected_{m}"] - drop_m
            rec[f"in_range_{m}"] = bool(base_b.min() <= base_m <= base_b.max())
            rec[f"n_band_{m}"] = int(near.sum())
            rec[f"excess_band_{m}"] = (float(drop_b[near].mean()) - drop_m) if near.any() else np.nan
            rec[f"excess_naive_{m}"] = float(drop_b.mean()) - drop_m
        rows.append(rec)

    r = pd.DataFrame(rows)
    save(r, "m1_splithalf_draws.csv")

    print(f"\nPart 2 — accuracy drop on the hard decile (mean over {N_SPLITS} splits, "
          f"n_hard~{int(r.n_hard.mean())})")
    print(f"  {'held-out readers, whole pool':<34}{100*r.drop_heldout_all.mean():6.1f} pts"
          f"   <- confounded: pool spans 47-93%")
    summary = []
    for m in pix:
        d, e = r[f"drop_{m}"], r[f"excess_{m}"]
        lo, hi = np.nanpercentile(e, [2.5, 97.5])
        print(f"  {m + ' (raw pixels)':<34}{100*d.mean():6.1f} pts  at level "
              f"{100*r[f'level_{m}'].mean():.1f}%  (in reader range on "
              f"{100*r[f'in_range_{m}'].mean():.0f}% of splits)")
        print(f"  {'  expected at that level':<34}{100*r[f'expected_{m}'].mean():6.1f} pts"
              f"   interface excess {100*e.mean():+5.1f} [{100*lo:+.1f}, {100*hi:+.1f}]")
        print(f"  {'  (band n~' + str(int(r[f'n_band_{m}'].mean())) + ' / naive)':<34}"
              f"{'':6}   {100*np.nanmean(r[f'excess_band_{m}']):+5.1f} / "
              f"{100*r[f'excess_naive_{m}'].mean():+.1f}")
        summary.append(dict(model=m,
                            level=round(100 * r[f"level_{m}"].mean(), 2),
                            drop_pixel=round(100 * d.mean(), 2),
                            drop_expected_at_level=round(100 * r[f"expected_{m}"].mean(), 2),
                            interface_excess=round(100 * e.mean(), 2),
                            ci_lo=round(100 * lo, 2), ci_hi=round(100 * hi, 2),
                            excess_band=round(100 * np.nanmean(r[f"excess_band_{m}"]), 2),
                            excess_naive=round(100 * r[f"excess_naive_{m}"].mean(), 2),
                            pct_splits_in_range=round(100 * r[f"in_range_{m}"].mean(), 1)))
    save(pd.DataFrame(summary), "m1_interface_excess.csv")

    # ---- Part 3: the same model, with and without the interface ----------
    # gpt-4o and gemini answer these problems both by reading their own
    # descriptions (CA) and by looking at the images (DVRL). Holding the model
    # fixed removes every between-model confound the pool comparison carries.
    print("\nPart 3 — same model, description interface vs raw pixels (paired)")
    within = []
    for m in PIXEL_MODELS:
        ca = u[u.paradigm.eq("CA") & u.reasoner_model.eq(m)].drop_duplicates("test_id")
        dv = u[u.paradigm.eq("DVRL") & u.reasoner_model.eq(m)].drop_duplicates("test_id")
        if ca.empty or dv.empty:
            continue
        a = ca.set_index("test_id").correct
        b = dv.set_index("test_id").correct
        common = a.index.intersection(b.index)
        if len(common) < 50:
            continue
        a, b = a.loc[common], b.loc[common]
        both_wrong = int(((a == 0) & (b == 0)).sum())
        ca_only = int(((a == 1) & (b == 0)).sum())   # description helps
        dv_only = int(((a == 0) & (b == 1)).sum())   # pixels help
        src = ca.components.iloc[0]
        pt, (lo, hi) = boot_ci((b - a).to_numpy(), clusters=ca.set_index("test_id").loc[common].uid)
        print(f"  {m:<22} n={len(common)}  CA({src}) {100*a.mean():.1f}%  "
              f"DVRL {100*b.mean():.1f}%  diff {100*pt:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]")
        print(f"  {'':<22} both wrong {both_wrong}   only-CA-right {ca_only}   "
              f"only-DVRL-right {dv_only}")
        within.append(dict(model=m, n=len(common), ca_source=src,
                           ca=round(100 * a.mean(), 1), dvrl=round(100 * b.mean(), 1),
                           ca_minus_dvrl=round(100 * pt, 1), ci_lo=round(100 * lo, 1),
                           ci_hi=round(100 * hi, 1), both_wrong=both_wrong,
                           only_ca=ca_only, only_dvrl=dv_only))
    save(pd.DataFrame(within), "m1_within_model.csv")

    # ---- Part 4: the same question in reverse ----------------------------
    # Part 2 finds that items hard for description readers are less hard for a
    # pixel model. That is only evidence about the description interface if the
    # reverse does NOT hold. So: define the hard set from one pixel model's
    # errors, and ask whether the other pixel model falls further on it than a
    # description reader of the same accuracy would.
    #
    # This direction is weaker by construction -- the set is defined by a single
    # binary system rather than a 21-model average, so it is noisier -- and it
    # is reported as indicative rather than as a matched counterpart.
    print("\nPart 4 — reverse direction: hard set defined by a pixel model")
    rev = []
    names = list(pix)
    for i, m_def in enumerate(names):
        for m_eval in names[i + 1:]:
            s_def, s_ev = pix[m_def], pix[m_eval]
            hard_i = s_def.index[s_def == 0]
            easy_i = s_def.index[s_def == 1]
            hi, ei = s_ev.index.intersection(hard_i), s_ev.index.intersection(easy_i)
            if len(hi) < 20:
                continue
            base_m = float(s_ev.loc[ei].mean())
            drop_m = base_m - float(s_ev.loc[hi].mean())
            # description readers on the same sets
            base_b = piv.loc[piv.index.intersection(easy_i)].mean()
            drop_b = base_b - piv.loc[piv.index.intersection(hard_i)].mean()
            coef = np.polyfit(base_b.to_numpy(), drop_b.to_numpy(), 2)
            expected = float(np.polyval(coef, base_m))
            print(f"  hard = {m_def} errors (n={len(hi)});  {m_eval} drops "
                  f"{100*drop_m:.1f} pts at level {100*base_m:.1f}%; "
                  f"description readers expected {100*expected:.1f} pts  "
                  f"-> pixel-specific excess {100*(expected-drop_m):+.1f}")
            rev.append(dict(hard_from=m_def, evaluated=m_eval, n_hard=len(hi),
                            level=round(100 * base_m, 2), drop=round(100 * drop_m, 2),
                            expected_description_reader=round(100 * expected, 2),
                            pixel_specific_excess=round(100 * (expected - drop_m), 2)))
    save(pd.DataFrame(rev), "m1_reverse_direction.csv")

    ex = {m: 100 * r[f"excess_{m}"].mean() for m in pix}
    lohi = {m: np.nanpercentile(r[f"excess_{m}"], [2.5, 97.5]) * 100 for m in pix}
    w = pd.DataFrame(within)
    # built outside the f-string: nested quotes of the same kind are a syntax error
    expected_str = " and ".join(f"{100 * r[f'expected_{m}'].mean():.1f}" for m in pix)
    drop_str = " and ".join(f"{100 * r[f'drop_{m}'].mean():.1f}" for m in pix)
    excess_str = "; ".join(
        f"{ex[m]:+.1f} points [{lohi[m][0]:+.1f}, {lohi[m][1]:+.1f}]" for m in pix
    )
    rv = pd.DataFrame(rev)
    rev_str = (f"{rv.iloc[0]['pixel_specific_excess']:+.1f} points"
               if len(rv) else "not estimable")
    para("M1 difficulty transfer", f"""
Section~\\ref{{sec:limitations}} raises the possibility that describing an image
and reasoning over the description discards evidence the pixels carried, and
concedes the concern is unmeasured. On Bongard-OpenWorld it can be measured, in
two ways that agree.

The direct test holds the model fixed and moves only the interface. Both models
that answer these problems end-to-end also answer them by describing the images
first and then reasoning over the text. Neither is harmed by the detour; both
are substantially helped. GPT-4o scores {w.iloc[0]['ca']:.1f}\\% through the
description interface against {w.iloc[0]['dvrl']:.1f}\\% from the images
({w.iloc[0]['ca_minus_dvrl']:+.1f} points, 95\\% CI [{w.iloc[0]['ci_lo']:+.1f},
{w.iloc[0]['ci_hi']:+.1f}]), and Gemini-2.0-Flash {w.iloc[1]['ca']:.1f}\\% against
{w.iloc[1]['dvrl']:.1f}\\% ({w.iloc[1]['ca_minus_dvrl']:+.1f} [{w.iloc[1]['ci_lo']:+.1f},
{w.iloc[1]['ci_hi']:+.1f}]). The exclusive wins are lopsided in the same direction:
for GPT-4o, {int(w.iloc[0]['only_ca'])} problems are solved only through the
description and {int(w.iloc[0]['only_dvrl'])} only from the pixels. Whatever the
verbalization discards, it is outweighed by what it supplies --- a compact,
already-abstracted representation of thirteen images that the model would
otherwise have to hold at once.

The second test asks whether the problems that defeat description readers are
the same problems that defeat a model looking at the images. Item difficulty is
first shown to be a shared property rather than a per-system quirk: splitting the
forty-two reasoners at random and correlating the two halves' item difficulties
gives $\\rho={rho:.2f}$, a full-pool reliability of {rel:.2f}. We then define the
hardest decile on a random half of the pool and measure the fall in accuracy on
it for the held-out half, over {N_SPLITS} splits, so that no system is scored on
items its own errors helped select. How far a reader falls also depends on where
it sits --- a near-chance model has nothing to lose and a near-ceiling one little
room to fall --- so we fit that relation across the held-out readers and read off
the fall expected at each pixel model's own accuracy, rather than comparing
against a pool average that near-chance readers dominate.

Difficulty does not fully transfer. At their own accuracy, description readers
would be expected to lose {expected_str} points on the hard decile; the pixel
models lose only {drop_str} points, an interface-specific excess of {excess_str}.
The problems that defeat the description readers are, to a substantial degree,
made hard by the description rather than by the problem.

The reverse does not hold, which is what makes this a statement about the
interface rather than about difficulty in general. Defining the hard set from one
pixel model's errors and evaluating the other, the excess over a matched
description reader is {rev_str} --- essentially nothing. The description route has
a private set of failures that the pixel route does not share; the pixel route
has no comparable private set. (This direction is noisier, since the set is
defined by a single system rather than a twenty-one-model average, and we read it
as indicative of the asymmetry rather than as a matched counterpart.)

Taken together the two halves are not in tension, and the combination is the
result. Decomposition raises the level and concentrates its losses. It supplies a
compact, already-abstracted representation of thirteen images, which is worth
about twelve points on average; on a minority of problems it discards precisely
the evidence that would have settled them, and those problems are then hard for
every reasoner downstream, however capable, because none of them can recover what
the description did not carry. This is what a verbalization bottleneck looks like
when it is measured rather than assumed: real, asymmetric, confined to a minority
of items, and outweighed in aggregate by the abstraction it buys. It also
predicts the reasoner-scaling behaviour tested in
Section~\\ref{{sec:results-components}} --- past a point, a better reasoner cannot
help, because the missing evidence is missing for all of them.

The scope should be stated plainly: this holds for a schema designed for these
tasks, on benchmarks whose critical evidence is verbalizable. It does not license
the same conclusion for concepts hinging on fine-grained geometry or signal
statistics, which this corpus does not contain.
""")


if __name__ == "__main__":
    main()
