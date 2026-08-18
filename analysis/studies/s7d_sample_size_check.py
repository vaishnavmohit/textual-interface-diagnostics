#!/usr/bin/env python3
"""Does the paradigm ordering survive at 100 problems per split?

Three of the five Bongard-HOI model configurations were scheduled on 100
problems per split and two on 500. Restricting the larger scheduled runs to the
fixed first-100 manifest makes the intended cohorts commensurable. Some Gemini
3 Flash archive rows are missing, so contrasts are paired on the available
intersection rather than assuming every restricted arm contains 400 valid
predictions. The ID-level accounting is performed by
``s7j_hoi_cohort_audit.py``.

Two questions, kept apart:
  1. Is the DVRL < DRL < CA ordering consistent at n=100, per split and pooled?
  2. Does restricting to 100 change any conclusion, or only the intervals?

The second is the one that matters. If the estimates agree and only the CIs
widen, then n=100 is a presentational choice with a cost and no benefit. If
they disagree, the disagreement is the finding and neither n can be quietly
preferred.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s7c_paradigm_ladder as S  # noqa: E402

SPLITS = S.SPLITS
CONTRASTS = S.CONTRASTS


def first_n(d: pd.DataFrame, keep: dict[str, set]) -> pd.DataFrame:
    return d[[t in keep[s] for s, t in zip(d.split, d.test_id)]]


def main() -> None:
    h = S.bongard_valid(S.load("bongard_hoi"))
    # the shared instance set: the 100 per split the older generation ran
    old = h[h.reasoner_model.eq("gpt-4o-2024-08-06")]
    keep = {sp: set(old[old.split.eq(sp)].test_id.astype(str)) for sp in SPLITS}

    models = [
        ("GPT-4o", lambda p: S.oldgen(h, "gpt-4o-2024-08-06", p), False),
        ("Gemini 2.0", lambda p: S.oldgen(h, "gemini-2.0-flash-exp", p), False),
        ("GPT-5.1", lambda p: S.newgen("gpt-5.1", p), True),
        ("Gemini 3 Flash", lambda p: S.newgen("gemini-3-flash-preview", p), True),
        ("Gemini 3.5 Flash Lite", lambda p: S.newgen("gemini-3.5-flash-lite", p), False),
    ]

    print("=== 1. is DVRL < DRL < CA consistent? (accuracy, per split) ===")
    print("  models marked [500->100] are restricted to the shared instance set\n")
    for name, get, big in models:
        cells = {p: get(p) for p in ("DVRL", "DRL", "CA")}
        if big:
            cells = {p: first_n(d, keep) for p, d in cells.items()}
        tag = "[500->100]" if big else "[100]"
        print(f"  {name} {tag}")
        print(f"    {'split':<8}{'DVRL':>8}{'DRL':>8}{'CA':>8}   ordering")
        for sp in SPLITS + ["POOLED"]:
            vals = {}
            for p, d in cells.items():
                s = d if sp == "POOLED" else d[d.split.eq(sp)]
                vals[p] = 100 * s.correct.mean() if len(s) else np.nan
            ok = vals["DVRL"] <= vals["DRL"] <= vals["CA"]
            mark = "DVRL<DRL<CA" if ok else "VIOLATED"
            print(f"    {sp:<8}{vals['DVRL']:>8.1f}{vals['DRL']:>8.1f}{vals['CA']:>8.1f}   {mark}")
        print()

    print("=== 2. does restricting to 100 change the ladder, or only the CIs? ===")
    print(f"  {'model':<22}{'step':<11}{'n':>6}{'delta':>9}{'95% CI':>20}{'width':>8}")
    rows = []
    for name, get, big in models:
        if not big:
            continue
        for scope in ("full 500", "first 100"):
            cells = {p: get(p) for p in ("DVRL", "DRL", "CA")}
            if scope == "first 100":
                cells = {p: first_n(d, keep) for p, d in cells.items()}
            for a, b, step in CONTRASTS:
                r = S.contrast(cells[a], cells[b])
                if not r:
                    continue
                print(f"  {name+' ('+scope+')':<22}{step:<11}{r['n']:>6}{r['delta']:>+9.2f}"
                      f"   [{r['lo']:>+6.2f},{r['hi']:>+7.2f}]{r['hi']-r['lo']:>8.1f}")
                rows.append(dict(model=name, scope=scope, step=step, **r))
            print()

    r = pd.DataFrame(rows)
    print("=== 3. side by side: does any conclusion flip? ===")
    print(f"  {'model':<18}{'step':<11}{'500':>9}{'100':>9}{'shift':>8}"
          f"{'500 excl 0':>12}{'100 excl 0':>12}")
    for (m, st), g in r.groupby(["model", "step"], sort=False):
        a = g[g.scope == "full 500"].iloc[0]
        b = g[g.scope == "first 100"].iloc[0]
        ea = "yes" if (a.lo > 0 or a.hi < 0) else "no"
        eb = "yes" if (b.lo > 0 or b.hi < 0) else "no"
        flip = "  <-- FLIP" if ea != eb else ""
        print(f"  {m:<18}{st:<11}{a.delta:>+9.2f}{b.delta:>+9.2f}{b.delta-a.delta:>+8.2f}"
              f"{ea:>12}{eb:>12}{flip}")

    out = Path(__file__).resolve().parents[2] / "results/studies/s7d_sample_size_check.csv"
    r.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
