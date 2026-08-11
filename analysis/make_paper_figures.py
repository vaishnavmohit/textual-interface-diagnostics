#!/usr/bin/env python3
"""The paper's result figures, from the committed study CSVs.

Every value is read from results/studies/; nothing is hard-coded, so a re-run
of the studies re-draws the figures identically. Output is vector PDF into
assets/figures/, which is the manuscript's graphicspath.

  fig_ow_ladder.pdf            C3 - C2 across five reasoners (Section 5)
  fig_hoi_ladder.pdf           two workflow transitions, five models
  fig_perception_vs_dprime.pdf CA-DRL transition vs native sensitivity (no fit
                               line, deliberately -- the ordering is descriptive)

    python analysis/make_paper_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd              # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "results" / "studies"
OUT = ROOT / "assets" / "figures"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

STEP_COLOR = {"rule": "#4878a8", "perception": "#c44e52", "both": "#55a868"}
STEP_LABEL = {"rule": "rule staging (DRL $-$ DVRL)",
              "perception": "description workflow (CA $-$ DRL)",
              "both": "both (CA $-$ DVRL)"}


def ow_ladder() -> None:
    d = pd.read_csv(STUDIES / "ladder_c3_minus_c2_by_reasoner.csv")
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    y = range(len(d))[::-1]
    ax.errorbar(d.delta, list(y),
                xerr=[d.delta - d.lo, d.hi - d.delta],
                fmt="o", color="#4878a8", ecolor="#4878a8",
                elinewidth=1.2, capsize=2.5, ms=4)
    ax.axvline(0, color="0.35", lw=0.8, ls="--")
    ax.set_yticks(list(y))
    ax.set_yticklabels(d.reasoner)
    ax.set_xlabel("C3 $-$ C2, percentage points (95% CI, clustered on uid)")
    fig.savefig(OUT / "fig_ow_ladder.pdf")
    plt.close(fig)


def hoi_ladder() -> None:
    d = pd.read_csv(STUDIES / "s7c_paradigm_ladder.csv")
    d = d[d.scope.eq("pooled")]
    models = ["Gemini 2.0", "Gemini 3.5 Flash Lite", "GPT-4o",
              "GPT-5.1", "Gemini 3 Flash"]  # ordered by DRL d'
    steps = ["rule", "perception", "both"]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ytick, ylab = [], []
    y = 0.0
    for m in models[::-1]:
        g = d[d.model.eq(m)].set_index("step")
        for s in steps[::-1]:
            if s not in g.index:
                continue
            r = g.loc[s]
            ax.errorbar(r.delta, y, xerr=[[r.delta - r.lo], [r.hi - r.delta]],
                        fmt="o", color=STEP_COLOR[s], elinewidth=1.2,
                        capsize=2.5, ms=4)
            y += 1.0
        ytick.append(y - 2.0)
        ylab.append(m)
        y += 0.9
    ax.axvline(0, color="0.35", lw=0.8, ls="--")
    ax.set_yticks(ytick)
    ax.set_yticklabels(ylab)
    ax.set_xlabel("paired difference, percentage points (95% CI)")
    handles = [plt.Line2D([], [], color=STEP_COLOR[s], marker="o", ls="",
                          label=STEP_LABEL[s]) for s in steps]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    fig.savefig(OUT / "fig_hoi_ladder.pdf")
    plt.close(fig)


def perception_vs_dprime() -> None:
    d = pd.read_csv(STUDIES / "s7c_perception_vs_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    ax.axhline(0, color="0.35", lw=0.8, ls="--")
    ax.scatter(d.drl_dprime, d.perception, s=28, color="#c44e52", zorder=3)
    offsets = {"Gemini 2.0": (5, 4), "Gemini 3.5 Flash Lite": (5, 4),
               "GPT-4o": (5, 4), "GPT-5.1": (5, 4), "Gemini 3 Flash": (-8, 6)}
    for _, r in d.iterrows():
        dx, dy = offsets.get(r.model, (5, 4))
        ha = "right" if dx < 0 else "left"
        ax.annotate(r.model, (r.drl_dprime, r.perception),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=7.5, ha=ha)
    ax.set_xlabel("native visual sensitivity: $d'$ under rule verbalization")
    ax.set_ylabel("description-workflow transition\n(CA $-$ DRL, points)")
    ax.set_xlim(0.3, 2.35)
    # no fitted line, deliberately: five models is an ordering, not a law
    fig.savefig(OUT / "fig_perception_vs_dprime.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ow_ladder()
    hoi_ladder()
    perception_vs_dprime()
    for f in ["fig_ow_ladder.pdf", "fig_hoi_ladder.pdf", "fig_perception_vs_dprime.pdf"]:
        print("wrote", OUT / f)
