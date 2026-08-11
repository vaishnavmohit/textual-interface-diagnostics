#!/usr/bin/env python3
"""S5 — Are failures rule-INDUCTION failures or rule-APPLICATION failures?

The paper distinguishes these stages but never separates them empirically. The
data already contain the control that does, and it had been mislabelled as a
generic "rule evaluation" condition.

Reading models/rule_eval/rule_application.py, the RuleApply condition sets

    ground_truth = caption

i.e. the model is handed the benchmark's own ground-truth description of the
concept and asked only to decide whether the query image satisfies it. Induction
is therefore *oracle* in that condition: there is no rule to infer, and every
error is an application error. That makes the comparison

    CA / DRL        (model induces the rule AND applies it)
    RuleApply       (rule given; model only applies it)

a decomposition of the pipeline rather than a comparison of two systems. The
accuracy ceiling under an oracle rule bounds how much of the residual error can
be attributed to induction at all.

The second half of the study uses the free-text rules the models produced. A
lexical-overlap measure asks whether the induced rule mentions the content words
of the ground-truth concept; it is deliberately conservative (semantic
paraphrase counts as a miss), so it gives a LOWER bound on rule quality, and is
used only to compare conditions, never as an absolute score.

    python analysis/studies/s5_induction_vs_application.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load, load_raw, bongard_valid, boot_ci, save, para  # noqa: E402

STOP = {"the", "a", "an", "of", "in", "on", "with", "and", "or", "is", "are", "to",
        "for", "at", "by", "from", "that", "this", "it", "its", "as", "be", "has"}


def content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", str(s).lower()) if w not in STOP and len(w) > 2}


def rule_covers_concept(rule: str, concept: str) -> float:
    """Fraction of the concept's content words that appear in the induced rule."""
    c = content_words(concept)
    if not c:
        return np.nan
    return len(c & content_words(rule)) / len(c)


def main() -> None:
    d = bongard_valid(load("bongard_ow"))
    d = d[d.ablation.isna() | (d.ablation == "")]

    # ---- 1. the oracle-rule ceiling -------------------------------------- #
    print("=== 1. Accuracy when the rule is GIVEN (oracle induction) vs induced ===")
    rows = []
    for par in ("CA", "DRL", "DVRL", "RuleApply"):
        g = d[d.paradigm == par]
        if not len(g):
            continue
        pt, (lo, hi) = boot_ci(g.correct.to_numpy(), g.uid.to_numpy())
        rows.append(dict(condition=par, induction="oracle (given)" if par == "RuleApply" else "model-induced",
                         n_runs=g.experiment_dir.nunique(), n_obs=len(g),
                         accuracy=round(100 * pt, 2), ci_lo=round(100 * lo, 2), ci_hi=round(100 * hi, 2)))
    ceil = pd.DataFrame(rows)
    save(ceil, "s5_oracle_rule_ceiling.csv")
    for _, r in ceil.iterrows():
        print(f"  {r.condition:<11}{r.induction:<18}{r.accuracy:>6.2f}  "
              f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]   {r.n_runs} runs, n={r.n_obs}")

    # paired: same reasoner, rule given vs induced
    print("\n=== 2. Paired within reasoner: does an oracle rule help? ===")
    pair_rows = []
    for m, g in d.groupby("reasoner_model"):
        conds = {p: s for p, s in g.groupby("paradigm")}
        if "RuleApply" not in conds or "CA" not in conds:
            continue
        a = conds["CA"].drop_duplicates("test_id").set_index("test_id").correct
        b = conds["RuleApply"].drop_duplicates("test_id").set_index("test_id").correct
        common = a.index.intersection(b.index)
        if len(common) < 50:
            continue
        diff = (b.loc[common] - a.loc[common]).to_numpy()
        pt, (lo, hi) = boot_ci(diff)
        pair_rows.append(dict(reasoner_model=m, n=len(common),
                              ca=round(100 * a.loc[common].mean(), 1),
                              ruleapply=round(100 * b.loc[common].mean(), 1),
                              delta=round(100 * pt, 2),
                              ci_lo=round(100 * lo, 2), ci_hi=round(100 * hi, 2)))
    pairs = pd.DataFrame(pair_rows)
    if len(pairs):
        save(pairs, "s5_oracle_vs_induced_paired.csv")
        print(f"  {'reasoner':<24}{'CA':>7}{'RuleApply':>11}{'Δ':>8}{'95% CI':>18}")
        for _, r in pairs.iterrows():
            print(f"  {r.reasoner_model:<24}{r.ca:>7.1f}{r.ruleapply:>11.1f}"
                  f"{r.delta:>+8.1f}   [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}]")

    # ---- 3. rule quality vs outcome, for the conditions that induce a rule -- #
    print("\n=== 3. Does a better-stated rule predict a correct answer? ===")
    RUNS = {
        "CA / gpt-4o (self)": "output/Bongard_OpenWorld_single/results.xlsx",
        "DRL / gpt-4o": "output/Bongard-OpenWorld_multi/results.xlsx",
        "DVRL / gpt-4o": "output/Bongard_ow_multi_test/results.xlsx",
        "CA / gemini (self)": "output/bongard_ow_single_gemini/results.xlsx",
    }
    qrows = []
    for label, rel in RUNS.items():
        try:
            r = load_raw(rel)
        except Exception:
            continue
        r = r[r.test_category_identified.astype(str).str.strip().isin(["cat_1", "cat_2"])]
        r["ok"] = (r.test_cat_label.astype(str).str.strip()
                   == r.test_category_identified.astype(str).str.strip())
        r["cover"] = [rule_covers_concept(a, b) for a, b in zip(r.rule_identified, r.concept)]
        hi = r[r.cover >= 0.5]
        lo = r[r.cover < 0.5]
        if not len(hi) or not len(lo):
            continue
        qrows.append(dict(condition=label, n=len(r),
                          median_coverage=round(float(r.cover.median()), 2),
                          acc_rule_covers=round(100 * hi.ok.mean(), 1), n_covers=len(hi),
                          acc_rule_misses=round(100 * lo.ok.mean(), 1), n_misses=len(lo),
                          gap=round(100 * (hi.ok.mean() - lo.ok.mean()), 1)))
    q = pd.DataFrame(qrows)
    if len(q):
        save(q, "s5_rule_quality_vs_outcome.csv")
        print(f"  {'condition':<22}{'median cov':>11}{'acc|covers':>12}{'acc|misses':>12}{'gap':>7}")
        for _, r in q.iterrows():
            print(f"  {r.condition:<22}{r.median_coverage:>11.2f}"
                  f"{r.acc_rule_covers:>11.1f}%{r.acc_rule_misses:>11.1f}%{r.gap:>+7.1f}")

    # ---- 4. the interaction: who benefits from a supplied rule? ---------- #
    from scipy.stats import pearsonr
    r_pear, p_pear = pearsonr(pairs.ca, pairs.delta)
    z = np.polyfit(pairs.ca, pairs.delta, 1)
    crossover = -z[1] / z[0]
    helped = pairs[pairs.delta > 0]
    hurt = pairs[pairs.delta < 0]
    print("\n=== 4. The supplied rule helps weak reasoners and hurts strong ones ===")
    print(f"  correlation(own-rule accuracy, benefit of supplied rule): r = {r_pear:+.3f}, p = {p_pear:.4f}")
    print(f"  crossover at own-rule accuracy ~= {crossover:.1f}%")
    print(f"  helped: {', '.join(helped.reasoner_model)}")
    print(f"  hurt  : {', '.join(hurt.reasoner_model)}")

    try:
        raw = load_raw("output/Bongard_OpenWorld_single/results.xlsx")
        cap_len = raw.caption.astype(str).str.len().median()
        rule_len = raw.rule_identified.astype(str).str.len().median()
    except Exception:
        cap_len = rule_len = float("nan")

    ra = ceil[ceil.condition == "RuleApply"]
    med_gap = q.gap.median() if len(q) else float("nan")

    para("S5 induction vs application", f"""
The rule-application condition hands the model the benchmark's own description of
the concept and asks only whether the query image satisfies it
(\\texttt{{ground\\_truth = caption}} in the implementation), so induction is removed
and every remaining error is an application error. Comparing it against the same
reasoners inducing their own rule separates the two stages directly.

The effect is not uniform, and its structure is the result. Supplying the
concept helps exactly those reasoners that induce poorly and harms those that
induce well: the benefit correlates with own-rule accuracy at $r={r_pear:.2f}$
($p={p_pear:.3f}$, {len(pairs)} reasoners), crossing zero at about {crossover:.0f}\\% accuracy. The four
weakest reasoners gain {helped.delta.min():+.1f} to {helped.delta.max():+.1f} points, while the four strongest lose
{abs(hurt.delta.max()):.1f} to {abs(hurt.delta.min()):.1f}.

The mechanism is visible in the artefacts. The benchmark caption is terse --- a
median of {cap_len:.0f} characters, e.g.\\ ``Fashion magazine.'' --- whereas a rule the model
states for itself runs to {rule_len:.0f} characters and enumerates the discriminating
attributes it intends to check. For a capable reasoner, replacing its own
elaborated rule with the ground-truth label is a loss of usable specification,
not a gain in correctness. A supplied rule is therefore better understood as a
substitute for induction than as an upper bound on it: it caps the pipeline at
the quality of the supplied text.

This reframes what the rule-application condition measures. It does not show
that induction is easy, nor that these models cannot apply a rule; it shows that
the value of an externally supplied rule is conditional on the recipient, and
that for the stronger reasoners in our set the induction stage is already
contributing more than the ground-truth concept label does. Consistent with
this, rule quality retains only limited residual predictive power within a fixed
pipeline (an accuracy gap of {med_gap:+.1f} points between items whose stated rule
lexically covers the ground-truth concept and those where it does not).
""")


if __name__ == "__main__":
    main()
