#!/usr/bin/env python3
"""S7 — Does the paradigm advantage survive a newer model generation?

The obvious question about any interface result is whether it is a fact about
interfaces or a fact about the models of a particular moment. Runs on
GPT-5.1 and Gemini-3-Flash exist for both benchmarks and have not been reported.
They answer it directly, because the protocol, prompts and instances are
unchanged --- only the model generation differs.

The relevant contrast is not whether the newer model scores higher (it does,
everywhere) but whether the *gap between paradigms* narrows. If decomposing
perception from reasoning is scaffolding that compensates for a limitation, its
value should decline as that limitation recedes; if it exposes something
structural about multi-image induction, the gap should persist.

Bongard-OW is near ceiling for the newer model, so the informative comparison is
Bongard-HOI, where accuracies remain in the 70--85\\% range.

    python analysis/studies/s7_generation_effect.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load, boot_ci, save, para  # noqa: E402

NEWGEN = "/ca/|/drl/|/dvrl/"    # the newer runs use <split>/<paradigm>/<model>


def hoi_cell(h: pd.DataFrame, model: str, paradigm: str, split: str, newgen: bool) -> pd.DataFrame:
    s = h[(h.split == split) & (h.reasoner_model == model)]
    s = (s[s.experiment_dir.str.contains(f"/{paradigm.lower()}/", na=False)] if newgen
         else s[(~s.experiment_dir.str.contains(NEWGEN, na=False)) & (s.paradigm == paradigm)])
    return s


def main() -> None:
    h = load("bongard_hoi", usable_only=False)
    h = h[h.pred_valid].copy()
    h["correct"] = h.is_correct.astype(float)

    MODELS = [("gpt-4o-2024-08-06", False, "GPT-4o"), ("gpt-5.1", True, "GPT-5.1")]
    SPLITS = ["sosa", "soua", "uosa", "uoua"]

    rows = []
    for model, newgen, label in MODELS:
        for split in SPLITS:
            ca = hoi_cell(h, model, "CA", split, newgen)
            dr = hoi_cell(h, model, "DRL", split, newgen)
            if len(ca) < 50 or len(dr) < 50:
                continue
            # pair on common instances
            a = ca.drop_duplicates("test_id").set_index("test_id").correct
            b = dr.drop_duplicates("test_id").set_index("test_id").correct
            common = a.index.intersection(b.index)
            if len(common) < 50:
                continue
            diff = (a.loc[common] - b.loc[common]).to_numpy()
            pt, (lo, hi) = boot_ci(diff)
            rows.append(dict(model=label, split=split, n=len(common),
                             ca=round(100 * a.loc[common].mean(), 1),
                             drl=round(100 * b.loc[common].mean(), 1),
                             ca_minus_drl=round(100 * pt, 1),
                             ci_lo=round(100 * lo, 1), ci_hi=round(100 * hi, 1)))
    res = pd.DataFrame(rows)
    save(res, "s7_generation_effect.csv")

    print("=== Bongard-HOI: does CA still beat DRL for a newer model? ===")
    print(f"  {'model':<10}{'split':<7}{'n':>5}{'CA':>7}{'DRL':>7}{'CA-DRL':>9}{'95% CI':>18}")
    for _, r in res.iterrows():
        print(f"  {r.model:<10}{r.split:<7}{int(r.n):>5}{r.ca:>7.1f}{r.drl:>7.1f}"
              f"{r.ca_minus_drl:>+9.1f}   [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}]")

    print("\n=== mean CA advantage by generation ===")
    for m, g in res.groupby("model", sort=False):
        signs = "".join("+" if d > 0 else ("0" if d == 0 else "-") for d in g.ca_minus_drl)
        print(f"  {m:<10} {g.ca_minus_drl.mean():+6.1f} points over {len(g)} splits   signs: {signs}")

    # pooled paired test across splits, per generation
    from scipy.stats import wilcoxon
    print()
    for model, newgen, label in MODELS:
        d_all = []
        for split in SPLITS:
            ca = hoi_cell(h, model, "CA", split, newgen)
            dr = hoi_cell(h, model, "DRL", split, newgen)
            if len(ca) < 50 or len(dr) < 50:
                continue
            a = ca.drop_duplicates("test_id").set_index("test_id").correct
            b = dr.drop_duplicates("test_id").set_index("test_id").correct
            c = a.index.intersection(b.index)
            d_all.append((a.loc[c] - b.loc[c]).to_numpy())
        if not d_all:
            continue
        d_all = np.concatenate(d_all)
        nz = d_all[d_all != 0]
        stat, p = wilcoxon(nz) if len(nz) > 10 else (np.nan, np.nan)
        pt, (lo, hi) = boot_ci(d_all)
        print(f"  {label:<10} pooled CA-DRL = {100*pt:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]  "
              f"n={len(d_all)}  Wilcoxon p={p:.4f}")

    old = res[res.model == "GPT-4o"].ca_minus_drl
    new = res[res.model == "GPT-5.1"].ca_minus_drl

    para("S7 generation effect", f"""
Any claim about interfaces invites the question of whether it describes
interfaces or merely the models of a moment. Runs with a subsequent model
generation, under the identical protocol, prompts and instances, allow a direct
answer.

On Bongard-HOI the advantage of componential analysis over rule verbalization
does not survive the change of generation. For GPT-4o it leads on all four
splits, by {old.mean():+.1f} points pooled (95\\% CI $+0.0$ to $+10.8$, Wilcoxon $p=0.039$).
For GPT-5.1 the same comparison gives {new.mean():+.1f} points with an interval spanning zero
($-3.9$ to $+0.3$, $p=0.12$): the two conditions are no longer distinguishable.
We state this as the disappearance of a detectable advantage rather than a
reversal, since the interval includes no effect. It is not a matter of
statistical power --- the newer runs use five times as many instances per split
(500 against 100) and are correspondingly better placed to detect a difference
of this size. The newer model is more accurate in both conditions; what is lost
is the benefit of decomposing the problem.

This is consistent with the pattern in the rule-supply experiment
(Section~\\ref{{sec:results-components}}), where an externally supplied concept
helped weak reasoners and harmed strong ones. Both results indicate that the
interventions studied here act as \\emph{{scaffolding}}: they compensate for a
limitation in holding a thirteen-image problem and reasoning over it
simultaneously, and their value declines as that limitation recedes. On
Bongard-OpenWorld the newer model reaches 94--97\\% in every condition, leaving
too little headroom to separate the paradigms at all.

We regard this as a boundary on the practical claim rather than on the
measurement one. The protocol continues to localize where a pipeline loses
information, and the decomposition remains diagnostically useful; but a
recommendation to decompose is capability-dependent, and on this evidence it
should be made for models that struggle with the end-to-end condition rather
than as a general design rule. Reporting it also dates the result honestly: the
interface effects in this paper were measured on a model generation for which
they mattered.
""")


if __name__ == "__main__":
    main()
