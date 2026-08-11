#!/usr/bin/env python3
"""Two anatomy checks on the perception step's loss (Section 6.8).

1. WHICH ERRORS grow when pixels become text? A d' drop can come from missed
   positives, from false alarms, or both, and they mean different things: a
   miss says the description failed to carry the action that makes the image
   positive; a false alarm says it asserted one that is not there. The
   verb-leak measurement predicts misses.

2. WHOSE failures are they? GPT-5.1 and Gemini 3 Flash write their own
   descriptions independently, in different model families. If the problems
   that fail specifically under CA (DRL right, CA wrong) are the SAME problems
   for both, the loss is a property of those instances under this channel ---
   not an idiosyncrasy of one describer. That overlap is also the natural
   target set for the E6 annotation.

    python analysis/studies/s7e_perception_loss_anatomy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s7c_paradigm_ladder as S  # noqa: E402
from s7c_paradigm_ladder import POS  # noqa: E402

from _common import para, save  # noqa: E402

MODELS = [("GPT-5.1", "gpt-5.1"), ("Gemini 3 Flash", "gemini-3-flash-preview")]


def rates(d: pd.DataFrame) -> tuple[float, float]:
    pos, neg = d.truth == POS, d.truth != POS
    return float((d.pred[pos] == POS).mean()), float((d.pred[neg] == POS).mean())


def ca_specific_fail(key: str) -> pd.Series:
    ca, drl = S.newgen(key, "CA"), S.newgen(key, "DRL")
    ka = ca.set_index(["split", "test_id"])
    kd = drl.set_index(["split", "test_id"])
    common = ka.index.intersection(kd.index)
    return pd.Series((ka.loc[common].correct == 0) & (kd.loc[common].correct == 1),
                     index=common)


def main() -> None:
    print("=== 1. hit / false-alarm decomposition, DRL -> CA ===")
    rows = []
    for name, key in MODELS:
        r = {}
        for p in ("DRL", "CA"):
            hit, fa = rates(S.newgen(key, p))
            r[p] = (hit, fa)
            print(f"  {name:<16}{p:<5} hit {100*hit:5.1f}%   miss {100*(1-hit):5.1f}%"
                  f"   FA {100*fa:5.1f}%")
        dmiss = 100 * ((1 - r["CA"][0]) - (1 - r["DRL"][0]))
        dfa = 100 * (r["CA"][1] - r["DRL"][1])
        print(f"  {'':<16}miss {dmiss:+.1f} pts, FA {dfa:+.1f} pts\n")
        rows.append(dict(model=name, d_miss=round(dmiss, 1), d_fa=round(dfa, 1)))

    print("=== 2. shared CA-specific failures across model families ===")
    f1, f2 = ca_specific_fail("gpt-5.1"), ca_specific_fail("gemini-3-flash-preview")
    common = f1.index.intersection(f2.index)
    a, b = f1.loc[common].astype(bool), f2.loc[common].astype(bool)
    both = int((a & b).sum())
    exp = float(a.mean() * b.mean() * len(common))
    orr, p = fisher_exact(pd.crosstab(a, b).values)
    print(f"  n={len(common)} common problems; failures {a.sum()} / {b.sum()}")
    print(f"  shared {both} vs {exp:.1f} expected under independence")
    print(f"  Fisher OR = {orr:.2f}, p = {p:.1e}")

    shared_idx = common[(a & b).to_numpy()]
    d = S.newgen("gemini-3-flash-preview", "CA").set_index(["split", "test_id"])
    truth = d.loc[d.index.intersection(shared_idx)].truth
    pos_share = float((truth == POS).mean())
    print(f"  {100*pos_share:.0f}% of shared failures are positives (missed action)")

    save(pd.DataFrame(rows), "s7e_miss_fa_decomposition.csv")
    shared = pd.DataFrame({"split": [i[0] for i in shared_idx],
                           "test_id": [i[1] for i in shared_idx]})
    save(shared, "s7e_shared_ca_failures.csv")
    print(f"  shared-failure instance list saved for the E6 annotation "
          f"({len(shared)} problems)")

    para("Perception-loss anatomy", f"""
The loss is made of missed positives, and it is problem-locked. Moving from
rule verbalization to descriptions raises Gemini 3 Flash's miss rate by
{rows[1]['d_miss']:+.1f} points against {rows[1]['d_fa']:+.1f} for false
alarms: when the description omits the action, the reasoner defaults to
rejection. And the problems that fail specifically under CA overlap heavily
across the two model families ({both} shared vs {exp:.0f} expected, odds ratio
{orr:.1f}, p = {p:.0e}), although each model wrote its own descriptions --
the failures follow the instances, not the describer.
""")


if __name__ == "__main__":
    main()
