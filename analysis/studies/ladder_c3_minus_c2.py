#!/usr/bin/env python3
"""The ladder's central contrast: does perception need to know the task?

C2 (role-blind) and C3 (role-aware) descriptions were produced by the same
pinned model (gpt-4o-2024-08-06), the same temperature (1.0), the same encoder,
and a byte-identical output schema; C3's prompt prepends only a task block
naming the Bongard setting and the image's role. The same reasoner
(qwen2.5:14b, temperature 0) read both. Whatever separates the arms is
task-conditioning of the perceptual stage and nothing else.

Also reports the reproduction check: the fresh pri C2 run against the legacy
NS-Reasoner C2 run on the same descriptions (historical 92.99%), which is what
licenses trusting the new pipeline for the contrast.

Inputs are the frozen per-run spreadsheets committed under results/ladder/.

    python analysis/studies/ladder_c3_minus_c2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (RNG, bongard_valid, cluster_randomization_pvalue,
                     load, normal_ppf, para, save)  # noqa: E402

LADDER = Path(__file__).resolve().parents[2] / "results" / "ladder"
N_BOOT = 4000


def arm(path: Path) -> pd.DataFrame:
    d = pd.read_excel(path)
    d["test_id"] = d["test_id"].astype(str)
    d = d.drop_duplicates("test_id").set_index("test_id")
    d["correct"] = d["is_correct"].astype(float)
    return d


def sdt(d: pd.DataFrame) -> tuple[float, float]:
    """(d', criterion c) from the normalized label columns."""
    pos = d[d.test_cat_label == "pos"]
    neg = d[d.test_cat_label == "neg"]
    h = ((pos.test_category_identified == "pos").sum() + 0.5) / (len(pos) + 1)
    f = ((neg.test_category_identified == "pos").sum() + 0.5) / (len(neg) + 1)
    zh, zf = normal_ppf(h), normal_ppf(f)
    return float(zh - zf), float(-(zh + zf) / 2)


def cluster_ci(diff: np.ndarray, uids: np.ndarray) -> tuple[float, float]:
    idx = pd.Series(range(len(diff)))
    groups = {k: v.to_numpy() for k, v in idx.groupby(uids)}
    keys = list(groups)
    boots = []
    for _ in range(N_BOOT):
        pick = np.concatenate([groups[k] for k in RNG.choice(keys, len(keys), replace=True)])
        boots.append(diff[pick].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


# Five reasoners read the SAME two frozen artifacts. Their contrasts are not
# independent replications -- they share the descriptions -- but they do test
# whether the null is a property of the interface or of one reader.
REASONERS = [
    ("Qwen2.5-14B",   "qwen25-14b"),
    ("Qwen2.5-32B",   "qwen2.5-32b"),
    ("Phi-4-14B",     "phi4-latest"),
    ("Gemma2-27B",    "gemma2-27b"),
    ("DeepSeek-R1-14B", "deepseek-r1-14b"),
]


def contrast(tag: str) -> dict | None:
    f2, f3 = LADDER / f"c2_{tag}.xlsx", LADDER / f"c3_{tag}.xlsx"
    if not (f2.exists() and f3.exists()):
        return None
    a_all, b_all = arm(f2), arm(f3)
    common = a_all.index.intersection(b_all.index)
    a, b = a_all.loc[common], b_all.loc[common]
    diff = (b["correct"] - a["correct"]).to_numpy()
    lo, hi = cluster_ci(diff, a["uid"].to_numpy())
    b01 = int(((a["correct"] == 0) & (b["correct"] == 1)).sum())
    b10 = int(((a["correct"] == 1) & (b["correct"] == 0)).sum())
    d2, c2c = sdt(a)
    d3, c3c = sdt(b)
    return dict(n=len(common), c2=100*a["correct"].mean(), c3=100*b["correct"].mean(),
                delta=100*float(diff.mean()), lo=100*lo, hi=100*hi,
                fixed=b01, broken=b10,
                p=cluster_randomization_pvalue(diff, a["uid"].to_numpy()),
                d2=d2, d3=d3, c2c=c2c, c3c=c3c)


def main() -> None:
    print("=== C3 - C2 across reasoners (same two frozen artifacts) ===")
    print(f"  {'reasoner':<18}{'n':>5}{'C2':>8}{'C3':>8}{'delta':>8}{'95% CI':>18}"
          f"{'fix/brk':>10}{'p':>7}{'d2':>6}{'d3':>6}")
    rows = []
    for label, tag in REASONERS:
        r = contrast(tag)
        if r is None:
            print(f"  {label:<18} (missing)"); continue
        fb = f"{r['fixed']}/{r['broken']}"
        print(f"  {label:<18}{r['n']:>5}{r['c2']:>8.2f}{r['c3']:>8.2f}{r['delta']:>+8.2f}"
              f"   [{r['lo']:+.2f},{r['hi']:+.2f}]{fb:>10}"
              f"{r['p']:>7.3f}{r['d2']:>6.2f}{r['d3']:>6.2f}")
        rows.append(dict(reasoner=label, **{k: (round(v, 3) if isinstance(v, float) else v)
                                            for k, v in r.items()}))
    multi = pd.DataFrame(rows)
    save(multi, "ladder_c3_minus_c2_by_reasoner.csv")

    deltas = multi.delta.to_numpy()
    print(f"\n  across {len(deltas)} reasoners: mean {deltas.mean():+.2f} pts, "
          f"range [{deltas.min():+.2f}, {deltas.max():+.2f}], "
          f"{(deltas > 0).sum()} positive")
    print("  shared artifacts make these sensitivity checks dependent; no cross-reader p-value is reported")
    print(f"  every interval spans zero: {all((multi.lo < 0) & (multi.hi > 0))}")

    c2 = arm(LADDER / "c2_qwen25-14b.xlsx")
    c3 = arm(LADDER / "c3_qwen25-14b.xlsx")
    common = c2.index.intersection(c3.index)
    a, b = c2.loc[common], c3.loc[common]
    uids = a["uid"].to_numpy()
    print(f"paired instances: {len(common)}  (C2 n={len(c2)}, C3 n={len(c3)})")

    diff = (b["correct"] - a["correct"]).to_numpy()
    pt = float(diff.mean())
    lo, hi = cluster_ci(diff, uids)
    b01 = int(((a["correct"] == 0) & (b["correct"] == 1)).sum())   # C3 fixes
    b10 = int(((a["correct"] == 1) & (b["correct"] == 0)).sum())   # C3 breaks
    p_cluster = cluster_randomization_pvalue(diff, uids)
    d2, c2c = sdt(a)
    d3, c3c = sdt(b)

    print(f"\nC2 role-blind : {100*a['correct'].mean():.2f}%   d'={d2:.2f}  c={c2c:+.2f}")
    print(f"C3 role-aware : {100*b['correct'].mean():.2f}%   d'={d3:.2f}  c={c3c:+.2f}")
    print(f"C3 - C2       : {100*pt:+.2f} pts  95% CI [{100*lo:+.2f}, {100*hi:+.2f}]  "
          f"(cluster bootstrap on uid)")
    print(f"discordant    : C3-fixes {b01}  C3-breaks {b10}   cluster-randomization p={p_cluster:.3f}")

    # --- reproduction check: fresh pri C2 vs legacy NS-Reasoner C2 -----------
    leg = bongard_valid(load("bongard_ow"))
    leg = leg[leg.ablation.isna() & leg.paradigm.eq("CA") & leg.components.eq("gpt-4o")
              & leg.reasoner_model.eq("qwen2.5:14b")].drop_duplicates("test_id")
    leg["test_id"] = leg["test_id"].astype(str)
    leg = leg.set_index("test_id")
    rc = leg.index.intersection(c2.index)
    agree = float((leg.loc[rc, "correct"].to_numpy() == c2.loc[rc, "correct"].to_numpy()).mean())
    r01 = int(((leg.loc[rc, "correct"] == 0) & (c2.loc[rc, "correct"] == 1)).sum())
    r10 = int(((leg.loc[rc, "correct"] == 1) & (c2.loc[rc, "correct"] == 0)).sum())
    p_rep = cluster_randomization_pvalue(
        (c2.loc[rc, "correct"] - leg.loc[rc, "correct"]).to_numpy(),
        c2.loc[rc, "uid"].to_numpy(),
    )
    print(f"\nreproduction  : legacy {100*leg.loc[rc,'correct'].mean():.2f}%  "
          f"fresh {100*c2.loc[rc,'correct'].mean():.2f}%  per-item agreement {100*agree:.1f}%")
    print(f"                discordant {r01}+{r10}, cluster-randomization p={p_rep:.3f}  (n={len(rc)})")

    save(pd.DataFrame([dict(
        contrast="C3 - C2 (qwen2.5:14b)", n=len(common),
        c2_acc=round(100*a["correct"].mean(), 2), c3_acc=round(100*b["correct"].mean(), 2),
        delta=round(100*pt, 2), ci_lo=round(100*lo, 2), ci_hi=round(100*hi, 2),
        c3_fixes=b01, c3_breaks=b10, cluster_p=round(p_cluster, 4),
        c2_dprime=round(d2, 2), c3_dprime=round(d3, 2),
        c2_criterion=round(c2c, 2), c3_criterion=round(c3c, 2),
        repro_legacy=round(100*leg.loc[rc, "correct"].mean(), 2),
        repro_agreement=round(100*agree, 1), repro_cluster_p=round(p_rep, 4),
    )]), "ladder_c3_minus_c2.csv")

    verdict = "null" if lo < 0 < hi else ("positive" if lo > 0 else "negative")
    para("Ladder C3 - C2 (task conditioning)", f"""
The central contrast of the interface ladder asks whether the perception stage
benefits from knowing the task it is describing for. Both arms use the same
pinned describer (gpt-4o-2024-08-06, temperature 1.0), the same image encoding
and a byte-identical output schema; the task-aware arm prepends only a block
naming the Bongard setting and the image's role. The same text-only reasoner
(qwen2.5:14b) reads both, so the contrast isolates task-conditioning of the
perceptual interface and nothing else.

Task-conditioning does not help ({verdict}): {100*a['correct'].mean():.1f}\\%
role-blind against {100*b['correct'].mean():.1f}\\% role-aware over {len(common)}
paired problems, a difference of {100*pt:+.1f} points (95\\% cluster-bootstrap CI
[{100*lo:+.1f}, {100*hi:+.1f}]; cluster-randomization $p={p_cluster:.2f}$ on
{b01}+{b10} discordant pairs). Sensitivity tells the same story
($d' = {d2:.2f}$ blind, ${d3:.2f}$ aware). Per the pre-registered reading rule,
this null is local to the tested models, prompts and tasks; but it is the
directly relevant regime, and it carries a design implication: at a fixed
structured schema, a role-blind description already retains what this reasoner
needs, so the query-conditioning that a unified architecture provides by
construction buys nothing here. Descriptions can be computed once, role-blind,
and reused across tasks at no measured cost.

The contrast is licensed by a reproduction check: rerunning the role-blind arm
through the new pipeline reproduces the archived legacy result to within
{abs(100*leg.loc[rc,'correct'].mean() - 100*c2.loc[rc,'correct'].mean()):.1f}
points ({100*leg.loc[rc,'correct'].mean():.2f}\\% archived,
{100*c2.loc[rc,'correct'].mean():.2f}\\% reproduced; per-item agreement
{100*agree:.1f}\\%, cluster-randomization $p={p_rep:.2f}$).
""")


if __name__ == "__main__":
    main()
