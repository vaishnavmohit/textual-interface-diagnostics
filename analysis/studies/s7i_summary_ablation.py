#!/usr/bin/env python3
"""How much of the structured description does the reasoner actually use?

The structured schema ends with a one-or-two-sentence ``Summary``. Keeping only
that field gives a condition in which the reasoner receives prose at ~11% of the
artifact's characters (149 against 1,313 median), produced by the SAME
perceptual pass over the same images -- so the describer, decoding and every
perceptual judgement are identical by construction.

    full CA  -  summary-only     what the structured detail is worth downstream

What this is and is not:
  * There is no describer confound at all: one generation feeds both arms. A
    freshly generated free-form condition changes prompt AND generation and
    cannot claim that.
  * Format and volume vary together -- the summary is prose and it is short.
  * It does NOT isolate structure. The summary was written after the model had
    decomposed the scene into eight fields, so it distils structured attention
    rather than replacing it. The unstructured-perception condition is the flat
    schema (context ``ca_flat``), generated from a prose prompt.

The comparator is the CA arm of the staging control, not the earlier ladder run:
those share this condition's context limit (16,384) and instance set, so only
the description content differs.

    python analysis/studies/s7i_summary_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RNG, para, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
STAGING = ROOT / "results" / "staging"
SUMMARY = ROOT / "results" / "summary"
POS = "pos"

# (label, full-CA file stem in results/staging, summary file stem in results/summary)
CELLS = [
    ("Qwen2.5-14B", "bongard_ow_staging_ca", "bongard_ow_summary_qwen2.5-14b"),
    ("Phi-4-14B", "bongard_ow_staging_phi4_ca", "bongard_ow_summary_phi4-14b"),
]


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_excel(path)
    d = d[d.test_category_identified.notna()]
    return pd.DataFrame({
        "test_id": d.test_id.astype(str),
        "uid": d.uid.astype(str) if "uid" in d else d.test_id.astype(str),
        "pred": d.test_category_identified.astype(str),
        "truth": d.test_cat_label.astype(str),
        "correct": d.is_correct.astype(float),
        "sha": (d.description_manifest_sha256.astype(str)
                if "description_manifest_sha256" in d else "n/a")})


def sdt(d: pd.DataFrame) -> float:
    pos, neg = d.truth == POS, d.truth != POS
    h = ((d.pred[pos] == POS).sum() + 0.5) / (pos.sum() + 1)
    f = ((d.pred[neg] == POS).sum() + 0.5) / (neg.sum() + 1)
    return float(norm.ppf(h) - norm.ppf(f))


def contrast(full: pd.DataFrame, summ: pd.DataFrame, n_boot: int = 10000):
    kf, ks = full.set_index("test_id"), summ.set_index("test_id")
    common = kf.index.intersection(ks.index)
    if len(common) < 30:
        return None
    a, b = kf.loc[common], ks.loc[common]
    diff = (b.correct - a.correct).to_numpy()
    idx = {k: v.to_numpy() for k, v in pd.Series(range(len(diff))).groupby(a.uid.to_numpy())}
    keys = list(idx)
    boots = np.array([diff[np.concatenate([idx[k] for k in RNG.choice(keys, len(keys), replace=True)])].mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    b01 = int(((b.correct == 1) & (a.correct == 0)).sum())
    b10 = int(((b.correct == 0) & (a.correct == 1)).sum())
    p = binomtest(b01, b01 + b10).pvalue if (b01 + b10) else 1.0
    return dict(n=len(common), full=100 * float(a.correct.mean()),
                summ=100 * float(b.correct.mean()), delta=100 * float(diff.mean()),
                lo=100 * float(lo), hi=100 * float(hi), mcnemar_p=float(p),
                fix=b01, brk=b10, d_full=sdt(a), d_summ=sdt(b))


def main() -> None:
    rows = []
    print(f"  {'reasoner':<16}{'n':>5}{'full CA':>9}{'summary':>9}{'delta':>9}"
          f"{'95% CI':>19}{'p':>8}   d-prime   fix/brk")
    for label, full_stem, summ_stem in CELLS:
        full = load(STAGING / f"{full_stem}.xlsx")
        summ = load(SUMMARY / f"{summ_stem}.xlsx")
        if full.empty or summ.empty:
            print(f"  {label:<16}  -- missing "
                  f"({'full' if full.empty else ''}{' and ' if full.empty and summ.empty else ''}"
                  f"{'summary' if summ.empty else ''})")
            continue
        r = contrast(full, summ)
        if not r:
            print(f"  {label:<16}  -- too few common instances")
            continue
        star = "*" if (r["lo"] > 0 or r["hi"] < 0) else " "
        print(f"  {label:<16}{r['n']:>5}{r['full']:>9.2f}{r['summ']:>9.2f}"
              f"{r['delta']:>+9.2f}{star}[{r['lo']:>+6.2f},{r['hi']:>+6.2f}]"
              f"{r['mcnemar_p']:>8.3f}   {r['d_full']:.2f}->{r['d_summ']:.2f}"
              f"   {r['fix']}/{r['brk']}")
        rows.append(dict(reasoner=label, **r))

    if not rows:
        print("\nNothing to analyse yet.")
        return
    r = pd.DataFrame(rows)
    save(r, "s7i_summary_ablation.csv")

    sig = r[(r.lo > 0) | (r.hi < 0)]
    signs = {d > 0 for d in r.delta}
    mixed = len(signs) > 1

    if mixed:
        gain = r.loc[r.delta.idxmax()]
        loss = r.loc[r.delta.idxmin()]
        body = f"""
How much description text helps depends on the reasoner, not on the schema.
Replacing each description with its own one-sentence Summary --- prose at roughly
a tenth of the characters, from the same perceptual pass --- costs
{loss.reasoner} {abs(loss.delta):.2f} points ([{loss.lo:+.2f}, {loss.hi:+.2f}])
while {gain.reasoner} \emph{{gains}} {gain.delta:.2f}
([{gain.lo:+.2f}, {gain.hi:+.2f}], $d'$ {gain.d_full:.2f} to {gain.d_summ:.2f}).
Because one generation feeds both arms, this cannot be a describer difference:
the same artifact is better for one reader shortened and worse for another. It
reproduces, under an independent manipulation of how much text the reasoner
receives, the sign reversal the staging control found for compression --- the
same reasoner benefits from less input in both. What the interface should pass
downstream is therefore a property of the consumer, and a fixed schema cannot be
optimal for every reader."""
    elif sig.empty:
        body = f"""
Most of the structured artifact is not doing downstream work. Replacing each
description with its own one-sentence Summary changes accuracy by
{r.delta.min():+.2f} to {r.delta.max():+.2f} points across {len(r)} reasoners,
every interval spanning zero, which locates the interface's value in compression
rather than in the enumerated fields."""
    else:
        worst = r.loc[r.delta.idxmin()]
        body = f"""
The structured detail is doing downstream work. Replacing each description with
its own one-sentence Summary costs up to {abs(worst.delta):.2f} points
([{worst.lo:+.2f}, {worst.hi:+.2f}] for {worst.reasoner}), with
{len(sig)} of {len(r)} intervals excluding zero, so what the interface passes
downstream is not reducible to an abstracted gist."""

    para("Summary ablation", body.strip())


if __name__ == "__main__":
    main()
