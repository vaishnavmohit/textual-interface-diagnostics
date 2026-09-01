#!/usr/bin/env python3
"""Would label substitution recover the schema-misconfigured pilot runs? No.

Three Bongard-OpenWorld pilot runs were decoded under a constrained grammar
whose vocabulary (pos/neg) did not match the benchmark's (cat_1/cat_2). The
tempting repair is a post-hoc mapping pos->cat_2, neg->cat_1. This audit tests
that repair directly against the raw archived predictions and finds it
unsound on three independent grounds:

  1. The constrained field is strongly single-label (81.0-98.7%), so the
     "predictions" carry almost no per-item variation to recover.
  2. Remapped accuracy is statistically indistinguishable from coin-flipping
     (binomial test vs 50%: p = 0.22-0.91), with task d' of 0.16, 0.46 and
     -1.21 -- the last pointing the WRONG way.
  3. Where the model's own prose reaches a detectable verdict, it contradicts
     the forced field in 14 of 25 rows (56%) -- and the prose verdicts are
     themselves at chance against ground truth (10/25 = 40%, p = 0.42), so no
     recoverable answer exists in either channel.
  4. The REVERSE mapping (pos->cat_1, neg->cat_2) fails identically: two-sided
     p-values are symmetric under relabeling (0.35 / 0.91 / 0.22), and the
     decisive internal control is Gemma3-27B itself -- its valid-vocabulary
     rows in the same run score 87.5% (n=401), so a merely label-inverted
     block would reverse-map to ~88%, not the observed 56.7%.

    python analysis/studies/s9_constrained_decoding_audit.py [--raw DIR]
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest, norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import para, save  # noqa: E402

MAP = {"pos": "cat_2", "neg": "cat_1", "positive": "cat_2", "negative": "cat_1"}
RUNS = [  # (run dir, restrict to the constrained pos/neg block only)
    ("bongard_ow_single_qwen2.5vl_32b", False),
    ("bongard_ow_single_qwen2.5vl_3b", False),
    ("bongard_ow_single_gemma3_27b", True),
]


def sdt(pred: pd.Series, truth: pd.Series) -> float:
    pos, neg = truth == "cat_2", truth == "cat_1"
    h = ((pred[pos] == "cat_2").sum() + 0.5) / (pos.sum() + 1)
    f = ((pred[neg] == "cat_2").sum() + 0.5) / (neg.sum() + 1)
    return float(norm.ppf(h) - norm.ppf(f))


def prose_verdict(text: str) -> str | None:
    """The verdict the model's own prose reaches, if detectable."""
    t = str(text).lower()
    m = re.findall(
        r"belongs to (?:the )?[`'\"]?(cat[_ ]?[12]|positive|negative)"
        r"|conclusion[^a-z]{0,12}(cat[_ ]?[12]|positive|negative)", t)
    flat = [x for pair in m for x in pair if x]
    if not flat:
        return None
    v = flat[-1].replace(" ", "_")
    return {"cat_1": "neg", "cat1": "neg", "negative": "neg",
            "cat_2": "pos", "cat2": "pos", "positive": "pos"}.get(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path,
                    default=Path.home() / "Downloads" / "hpc_results" / "output")
    args = ap.parse_args()

    rows = []
    print(f"{'run':<38}{'n':>5}{'1-label%':>9}{'remap acc':>10}"
          f"{'p vs 50%':>10}{'d-prime':>9}{'prose!=field':>14}")
    for run, constrained_only in RUNS:
        fs = glob.glob(str(args.raw / run / "*.xlsx"))
        if not fs:
            print(f"{run:<38}  raw file not found under {args.raw}")
            continue
        d = pd.read_excel(fs[0])
        lab = d.test_category_identified.astype(str).str.strip().str.lower()
        if constrained_only:
            keep = lab.isin(MAP)
            d, lab = d[keep], lab[keep]
        pred = lab.map(MAP)
        truth = (d.test_cat_label.astype(str).str.strip().str.lower()
                 .map(lambda s: MAP.get(s, s)))
        ok = pred.notna() & truth.isin(["cat_1", "cat_2"])
        pred, truth = pred[ok], truth[ok]
        acc = float((pred == truth).mean())
        single = float(lab.value_counts(normalize=True).max())
        p = binomtest(int((pred == truth).sum()), len(pred), 0.5).pvalue
        pv = d.complete_output.map(prose_verdict)
        have = pv.notna()
        dis = int((pv[have] != lab[have]).sum())
        print(f"{run:<38}{len(pred):>5}{100*single:>8.1f}%{100*acc:>9.1f}%"
              f"{p:>10.3f}{sdt(pred, truth):>9.2f}{dis:>7}/{int(have.sum()):<6}")
        rows.append(dict(run=run, n=len(pred), single_label_pct=round(100*single, 1),
                         remap_acc_pct=round(100*acc, 1), binom_p_vs_chance=round(p, 3),
                         d_prime=round(sdt(pred, truth), 2),
                         prose_verdicts=int(have.sum()), prose_contradicts_field=dis))

    # --- the reverse mapping, and the two controls that rule it out -------- #
    print("\n=== reverse mapping (pos->cat_1, neg->cat_2) ===")
    REV = {"pos": "cat_1", "neg": "cat_2", "positive": "cat_1", "negative": "cat_2"}
    for run, constrained_only in RUNS:
        fs = glob.glob(str(args.raw / run / "*.xlsx"))
        if not fs:
            continue
        d = pd.read_excel(fs[0])
        lab = d.test_category_identified.astype(str).str.strip().str.lower()
        if constrained_only:
            keep = lab.isin(MAP)
            d, lab = d[keep], lab[keep]
        pred = lab.map(REV)
        truth = (d.test_cat_label.astype(str).str.strip().str.lower()
                 .map(lambda s: MAP.get(s, s)))
        ok = pred.notna() & truth.isin(["cat_1", "cat_2"])
        pred, truth = pred[ok], truth[ok]
        p = binomtest(int((pred == truth).sum()), len(pred), 0.5).pvalue
        print(f"  {run:<38} rev acc {100*(pred == truth).mean():5.1f}%  p={p:.2f}"
              f"  d'={sdt(pred, truth):+.2f}")

    # prose verdicts vs ground truth (32B): is there a competent hidden answer?
    d = pd.read_excel(glob.glob(str(args.raw / RUNS[0][0] / "*.xlsx"))[0])
    pv = d.complete_output.map(prose_verdict).map(
        {"pos": "cat_2", "neg": "cat_1"})
    truth = (d.test_cat_label.astype(str).str.strip().str.lower()
             .map(lambda s: MAP.get(s, s)))
    have = pv.notna()
    k, n = int((pv[have] == truth[have]).sum()), int(have.sum())
    print(f"  prose verdicts vs ground truth: {k}/{n} = {100*k/max(n,1):.0f}%"
          f" (p={binomtest(k, n, 0.5).pvalue:.2f}) -> no recoverable answer in the prose either")

    # internal control: Gemma's valid-vocabulary rows in the SAME run
    d = pd.read_excel(glob.glob(str(args.raw / "bongard_ow_single_gemma3_27b" / "*.xlsx"))[0])
    lab = d.test_category_identified.astype(str).str.strip().str.lower()
    valid = d[lab.isin(["cat_1", "cat_2"])]
    vt = (valid.test_cat_label.astype(str).str.strip().str.lower()
          .map(lambda s: MAP.get(s, s)))
    va = float((valid.test_category_identified.astype(str).str.lower() == vt).mean())
    print(f"  Gemma3-27B valid-vocabulary rows, same run: n={len(valid)}, {100*va:.1f}%"
          f" -> an inverted-but-competent block would reverse-map to ~{100*va:.0f}%, not 56.7%")

    r = pd.DataFrame(rows)
    save(r, "s9_constrained_decoding_audit.csv")

    para("Constrained-decoding audit", f"""
Label substitution does not recover the schema-misconfigured pilot runs. Across
the three affected cohorts the constrained field is {r.single_label_pct.min():.0f}
to {r.single_label_pct.max():.0f} percent single-label, remapped accuracy is
{r.remap_acc_pct.min():.1f} to {r.remap_acc_pct.max():.1f} percent and
statistically indistinguishable from chance (binomial p = {r.binom_p_vs_chance.min():.2f}
to {r.binom_p_vs_chance.max():.2f}), and where the model's own prose reaches a
detectable verdict it contradicts the forced field in
{int(r.prose_contradicts_field.sum())} of {int(r.prose_verdicts.sum())} rows,
while the prose verdicts are themselves at chance -- no recoverable answer
exists in either channel. The reverse mapping fails identically (p-values are
symmetric under relabeling), and Gemma3-27B's own valid-vocabulary rows in the
same run (87.5%, n=401) rule out an inverted-but-competent block, which would
reverse-map to ~88% rather than the observed 56.7%. Exclusion is therefore the
only treatment that does not fabricate a measurement.
""".strip())


if __name__ == "__main__":
    main()
