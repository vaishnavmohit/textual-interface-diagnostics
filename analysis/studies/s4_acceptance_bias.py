#!/usr/bin/env python3
"""S4 — Do the paradigms differ in decision criterion, not just in accuracy?

Bongard query sets are balanced by construction: exactly half the queries belong
to the positive category. A model that discriminates well should answer positive
about half the time. Systematic departure from that is a shift in *decision
criterion* — a willingness to accept the query as matching the rule — which is a
different phenomenon from being more or less accurate, and one that accuracy
alone cannot show.

This matters for the paradigm comparison. If RuleApply is less accurate than CA,
that could mean the reasoner is worse at applying a supplied rule; but if it is
also systematically over-accepting, the deficit is partly a criterion shift, and
the right description is that supplying a rule makes models credulous rather
than incapable.

Reported per run:

  positive-response rate with a binomial CI against the 50% base rate
  sensitivity  d' = z(hit rate) - z(false-alarm rate)      discrimination
  criterion    c  = -[z(hit) + z(false alarm)] / 2         bias, 0 = neutral

d' and c come from signal detection theory and are exactly the decomposition
needed here: they separate how well a model tells the classes apart from where
it places its threshold. Two models can have identical accuracy with very
different c.

    python analysis/studies/s4_acceptance_bias.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load, bongard_valid, save, para  # noqa: E402

POS = "cat_2"   # dataset/bongard_ow.py: imagefiles["cat_2"] = positive


def sdt(g: pd.DataFrame) -> dict:
    """Signal-detection measures with a log-linear correction for extreme rates."""
    pos = g[g.label_true == POS]
    neg = g[g.label_true != POS]
    if not len(pos) or not len(neg):
        return {}
    # hit = says positive when positive; false alarm = says positive when negative
    h = ((pos.pred_raw == POS).sum() + 0.5) / (len(pos) + 1)
    f = ((neg.pred_raw == POS).sum() + 0.5) / (len(neg) + 1)
    zh, zf = norm.ppf(h), norm.ppf(f)
    return dict(hit_rate=round(100 * h, 1), false_alarm_rate=round(100 * f, 1),
                d_prime=round(zh - zf, 3), criterion_c=round(-(zh + zf) / 2, 3))


def analyse(d: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    rows = []
    for (ed, par, model), g in d.groupby(["experiment_dir", "paradigm", "reasoner_model"]):
        n = len(g)
        n_pos = int((g.pred_raw == POS).sum())
        if n < 50:
            continue
        bt = binomtest(n_pos, n, 0.5)
        ci = bt.proportion_ci(0.95)
        rec = dict(benchmark=benchmark, experiment_dir=ed, paradigm=par, reasoner_model=model,
                   n=n, accuracy=round(100 * g.correct.mean(), 1),
                   pos_rate=round(100 * n_pos / n, 1),
                   pos_ci_lo=round(100 * ci.low, 1), pos_ci_hi=round(100 * ci.high, 1),
                   p_vs_balanced=round(bt.pvalue, 6))
        rec.update(sdt(g))
        rec["biased"] = bool(rec["p_vs_balanced"] < 0.05)
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    frames = []
    for kind, label in (("bongard_ow", "bongard_ow"), ("bongard_hoi", "bongard_hoi")):
        d = bongard_valid(load(kind))
        d = d[d.ablation.isna() | (d.ablation == "")]      # main runs only
        frames.append(analyse(d, label))
    res = pd.concat(frames, ignore_index=True)
    save(res, "s4_acceptance_bias.csv")

    print("=== Response bias by paradigm (balanced sets: 50% is neutral) ===")
    print(f"{'benchmark':<13}{'paradigm':<11}{'runs':>5}{'median pos%':>12}{'biased':>8}"
          f"{'median c':>10}{'median d-prime':>15}")
    for (b, p), g in res.groupby(["benchmark", "paradigm"]):
        print(f"{b:<13}{p:<11}{len(g):>5}{g.pos_rate.median():>11.1f}%"
              f"{int(g.biased.sum()):>5}/{len(g):<3}{g.criterion_c.median():>9.2f}"
              f"{g.d_prime.median():>15.2f}")

    ow = res[res.benchmark == "bongard_ow"]
    print("\n=== Bongard-OW runs, sorted by bias ===")
    print(f"{'run':<46}{'par':<10}{'acc':>6}{'pos%':>7}{'c':>7}{'d-prime':>9}")
    for _, r in ow.sort_values("criterion_c").iterrows():
        flag = " *" if r.biased else ""
        print(f"{r.experiment_dir[:45]:<46}{r.paradigm:<10}{r.accuracy:>6.1f}"
              f"{r.pos_rate:>7.1f}{r.criterion_c:>7.2f}{r.d_prime:>9.2f}{flag}")

    ra = ow[ow.paradigm == "RuleApply"]
    ca = ow[ow.paradigm == "CA"]
    dv = ow[ow.paradigm == "DVRL"]
    dr = ow[ow.paradigm == "DRL"]

    # --- the degenerate-condition check: accuracy near chance AND d' near zero --- #
    deg = res[(res.d_prime < 0.6) & (res.n >= 90)]
    if len(deg):
        print("\n=== Runs where accuracy is NOT a measurement (d' < 0.6 = at/near chance) ===")
        print(f"{'run':<42}{'model':<24}{'acc':>6}{'pos%':>7}{'d-prime':>9}")
        for _, r in deg.sort_values("d_prime").iterrows():
            print(f"{r.experiment_dir[:41]:<42}{str(r.reasoner_model)[:23]:<24}"
                  f"{r.accuracy:>6.1f}{r.pos_rate:>7.1f}{r.d_prime:>9.2f}")
        save(deg, "s4_non_discriminating_runs.csv")

    hoi = res[res.benchmark == "bongard_hoi"]
    gem = hoi[(hoi.reasoner_model == "gemini-2.0-flash-exp") & (hoi.paradigm == "DVRL")]
    if len(gem):
        para("S4b DVRL feasibility limit", f"""
One condition fails outright, and reporting it as an accuracy conceals that.
Under direct reasoning over all thirteen images on Bongard-HOI,
Gemini-2.0-Flash attains accuracies of {', '.join(f'{a:.0f}' for a in gem.accuracy)}\\% --- values that read as weak but
measurable. Its sensitivity in the same runs is $d'={gem.d_prime.min():.2f}$--${gem.d_prime.max():.2f}$, i.e.\\
indistinguishable from chance, and it answers ``positive'' on {gem.pos_rate.median():.0f}\\% of a balanced
set. The apparent accuracy is produced entirely by that constant response, not by
discrimination.

Two further observations identify the cause as the input condition rather than
the task. First, the same model discriminates normally on the same benchmark
when the interface changes ($d'={hoi[(hoi.reasoner_model=='gemini-2.0-flash-exp')&(hoi.paradigm=='CA')].d_prime.median():.2f}$ under componential analysis). Second, in
these runs the model sometimes states the problem explicitly, returning
``I lack the ability to analyze a set of images \\ldots I need the 12 images to
perform the analysis'' rather than a label. GPT-4o under the identical condition
retains $d'={hoi[(hoi.reasoner_model=='gpt-4o-2024-08-06')&(hoi.paradigm=='DVRL')].d_prime.median():.2f}$, so the limit is model-specific rather than intrinsic to
thirteen-image contexts.

We therefore report this cell as a failure of the condition for this model, not
as a low score. Including it in a paradigm average would understate direct
reasoning by attributing to it a number that measures an input-handling limit.
""")

    para("S4 acceptance bias", f"""
Because the query sets are balanced by construction, the rate at which a system
answers ``positive'' is interpretable on its own: 50\\% is neutral, and departures
measure a shift in decision criterion rather than in discrimination. Separating
the two with signal-detection measures shows that the paradigms differ in both,
and that the differences do not coincide.

Externally supplied rules make models credulous. Under rule application the
median positive-response rate is {ra.pos_rate.median():.1f}\\% ({int(ra.biased.sum())} of {len(ra)} runs depart significantly from
balance), with median criterion $c={ra.criterion_c.median():+.2f}$; the same reasoners under
componential analysis sit at {ca.pos_rate.median():.1f}\\% and $c={ca.criterion_c.median():+.2f}$. A system given a rule
accepts the query as matching it more readily than one that induced the rule
itself, and this holds even for the most accurate rule-application runs.

The end-to-end paradigms are intermediate: direct reasoning over all images has
median $c={dv.criterion_c.median():+.2f}$ and rule verbalisation $c={dr.criterion_c.median():+.2f}$, the latter being the
best-calibrated condition we measure. Discriminability tells a different story
from criterion --- median $d'$ is {ca.d_prime.median():.2f} for componential analysis against
{ra.d_prime.median():.2f} for rule application --- so the accuracy ordering of the paradigms is not
simply a re-description of their willingness to accept.

Reporting accuracy alone conceals this. Two conditions with the same accuracy can
place their thresholds very differently, and a criterion shift is a design fact
about the interface, not a property of the underlying visual competence.
""")


if __name__ == "__main__":
    main()
