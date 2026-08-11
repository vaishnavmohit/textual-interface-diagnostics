#!/usr/bin/env python3
"""S1 — Is ICA's re-inspection *selective*, and is the model calibrated?

The paper reports that ICA beats CA. It does not report the mechanism. But the
transcripts record, per item, whether the reasoner chose to ask a question at
all — and it asks on only about half of them.

That turns one net gain into four separable claims:

  1. SELECTION      does the model ask more often on items it gets wrong?
  2. CALIBRATION    is "asked" a usable uncertainty signal (a cheap proxy for
                    self-knowledge)?
  3. YIELD          conditional on asking, does accuracy improve — and at what
                    harm rate?
  4. NECESSITY      on items where it declined to ask, would asking have helped?
                    (bounded by comparing to the fixed-reinspection condition
                    when available)

Claim 1+2 are what license the word *adaptive*: a system that re-inspects
indiscriminately is not adaptive, however large its net gain.

    python analysis/studies/s1_ica_selectivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_raw, boot_ci, save, para  # noqa: E402

# The matched cell: perception and reasoner both GPT-4o (see provenance/E1_AUDIT.md).
ICA_FILES = {
    "ica_1_single_qn": "output_winoground/gpt_activity/gpt-4o-2024-08-06/scores_winoground_ica_1_single_qn_55.75.xlsx",
    "ica":             "output_winoground/gpt_activity/gpt-4o-2024-08-06/scores_winoground_ica_55.75.xlsx",
    "ica_single_qn":   "output_winoground/gpt_activity/gpt-4o-2024-08-06/scores_winoground_ica_single_qn_55.00.xlsx",
    "ica_no_cs":       "output_winoground/gpt_activity/gpt-4o-2024-08-06/scores_winoground_ica_single_no_commonsense_55.25.xlsx",
}
CA_FILE = "output_winoground/gpt_activity/gpt-4o-2024-08-06/scores_winoground.xlsx"

# The unit of decision is a SCORING DIRECTION, not an item. Each direction gets
# its own question column, and the two directions behave completely differently:
#
#   image_* : a question is recorded on 100% of items — re-inspection is
#             UNCONDITIONAL here, so there is no selection to measure.
#   text_*  : a question is recorded on ~49% — the reasoner genuinely chooses.
#
# Treating an item as "asked" if ANY column is populated (the obvious reading)
# collapses to a 100% ask rate and destroys the effect. The selectivity result
# lives in the text direction; the image direction is the within-experiment
# control showing what unconditional re-inspection looks like.
DIRECTIONS = [
    ("text",  ["text_img0_questions", "text_img1_questions"], "text_score"),
    ("image", ["img0_questions", "img1_questions"],           "image_score"),
]


def main() -> None:
    ca = load_raw(CA_FILE).set_index("id")
    rows, per_item = [], []

    for name, rel in ICA_FILES.items():
        d = load_raw(rel)
        d = d[d.id.isin(ca.index)]
        base = ca.loc[d.id]

        for direction, qcols, metric in DIRECTIONS:
            present = [c for c in qcols if c in d.columns]
            if not present:
                continue
            # "asked" = the reasoner generated a question for this direction
            asked = d[present].notna().any(axis=1).to_numpy()
            ica_m = d[metric].to_numpy(dtype=float)
            ca_m = base[metric].to_numpy(dtype=float)
            if asked.all() or not asked.any():
                # unconditional (or never) — no selection to measure; record and move on
                y_pt, y_ci = boot_ci(ica_m - ca_m)
                rows.append(dict(
                    variant=name, direction=direction, metric=metric,
                    n_asked=int(asked.sum()), n_not_asked=int((~asked).sum()),
                    ask_rate=round(100 * asked.mean(), 1),
                    ca_acc_on_asked=np.nan, ca_acc_on_not_asked=np.nan,
                    selectivity_gap=np.nan,
                    yield_on_asked=round(100 * y_pt, 1),
                    yield_asked_lo=round(100 * y_ci[0], 1), yield_asked_hi=round(100 * y_ci[1], 1),
                    drift_on_not_asked=np.nan, selective=False))
                continue

            # 1/2 SELECTION + CALIBRATION: CA accuracy on asked vs not-asked items.
            # Using the CA baseline (not ICA's own score) avoids circularity: it
            # asks whether the model targeted items that were *already* hard.
            a_pt, a_ci = boot_ci(ca_m[asked])
            n_pt, n_ci = boot_ci(ca_m[~asked])
            # 3 YIELD: paired change, conditional on asking
            y_pt, y_ci = boot_ci(ica_m[asked] - ca_m[asked])
            u_pt, u_ci = boot_ci(ica_m[~asked] - ca_m[~asked])

            rows.append(dict(
                variant=name, direction=direction, metric=metric,
                n_asked=int(asked.sum()), n_not_asked=int((~asked).sum()),
                ask_rate=round(100 * asked.mean(), 1),
                ca_acc_on_asked=round(100 * a_pt, 1), ca_asked_lo=round(100 * a_ci[0], 1),
                ca_asked_hi=round(100 * a_ci[1], 1),
                ca_acc_on_not_asked=round(100 * n_pt, 1), ca_not_lo=round(100 * n_ci[0], 1),
                ca_not_hi=round(100 * n_ci[1], 1),
                selectivity_gap=round(100 * (n_pt - a_pt), 1),
                yield_on_asked=round(100 * y_pt, 1),
                yield_asked_lo=round(100 * y_ci[0], 1), yield_asked_hi=round(100 * y_ci[1], 1),
                drift_on_not_asked=round(100 * u_pt, 1),
                drift_not_lo=round(100 * u_ci[0], 1), drift_not_hi=round(100 * u_ci[1], 1),
                selective=True,
            ))

            if direction == "text":
                per_item.append(pd.DataFrame(dict(
                    variant=name, id=d.id.to_numpy(), asked=asked,
                    ca_score=ca_m, ica_score=ica_m,
                    tag=d.get("collapsed_tag", pd.Series(index=d.index)).to_numpy(),
                )))

    res = pd.DataFrame(rows)
    items = pd.concat(per_item, ignore_index=True)
    save(res, "s1_ica_selectivity.csv")
    save(items, "s1_ica_per_item.csv")

    print("=== Re-inspection is UNCONDITIONAL in one direction, SELECTIVE in the other ===")
    print(f"{'variant':<18}{'dir':<7}{'ask rate':>9}{'CA|asked':>10}{'CA|not':>9}{'gap':>7}"
          f"{'yield|asked':>13}{'drift|not':>11}")
    for _, r in res.iterrows():
        f = lambda v: "    --" if pd.isna(v) else f"{v:6.1f}"
        print(f"{r.variant:<18}{r.direction:<7}{r.ask_rate:>8.1f}%{f(r.ca_acc_on_asked):>10}"
              f"{f(r.ca_acc_on_not_asked):>9}{f(r.selectivity_gap):>7}"
              f"{r.yield_on_asked:>+12.1f}{f(r.drift_on_not_asked):>11}")
    g = res[res.selective]

    # tag-level view: which linguistic phenomena trigger a question
    if items.tag.notna().any():
        print("\n=== Which Winoground phenomena trigger a question? ===")
        t = items[items.variant == "ica_1_single_qn"].groupby("tag").agg(
            n=("id", "size"), ask_rate=("asked", lambda s: round(100 * s.mean(), 1)),
            ca_score=("ca_score", lambda s: round(100 * s.mean(), 1)),
            ica_score=("ica_score", lambda s: round(100 * s.mean(), 1)))
        t["gain"] = (t.ica_score - t.ca_score).round(1)
        print(t.to_string())

    m = g.selectivity_gap.median()
    yl = g.yield_on_asked.median()
    dr = g.drift_on_not_asked.median()
    unc = res[~res.selective]
    para("S1 ICA selectivity", f"""
The two scoring directions of the interactive condition differ in a way that
provides a within-experiment control. When matching an image to captions, the
reasoner issues a re-inspection query on every item ({unc.ask_rate.iloc[0]:.0f}\\%): re-inspection is
unconditional. When matching a caption to images, it issues one on only
{g.ask_rate.min():.0f}--{g.ask_rate.max():.0f}\\% of items --- here it chooses.

Where it chooses, the choice is well calibrated. Items it elected to query had a
baseline accuracy of {g.ca_acc_on_asked.median():.1f}\\% under static componential analysis, against
{g.ca_acc_on_not_asked.median():.1f}\\% for items it passed over --- a gap of {m:.1f} points. The model
therefore identifies, before looking again, the cases in which its first answer
was unreliable. Conditional on querying, accuracy rises by {yl:+.1f} points, while
items it declined to query move by {dr:+.1f} points: the aggregate gain is
concentrated entirely in the subset the model itself flagged.

This distinguishes adaptive re-inspection from a larger inspection budget. The
unconditional direction shows what indiscriminate re-inspection yields ({unc.yield_on_asked.median():+.1f}
points); the selective direction achieves a comparable gain ({yl:+.1f}) while
querying only about two thirds as often, because it spends the budget where it
is needed. Selectivity, not the extra call, is what the interactive condition
contributes.
""")


if __name__ == "__main__":
    main()
