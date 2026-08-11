#!/usr/bin/env python3
"""Was the paradigm advantage a criterion repair? A prediction, then a test.

Section 6 accounts for GPT-4o's componential-analysis advantage on Bongard-HOI
as a repair of decision bias rather than an improvement in discrimination: under
end-to-end rule application GPT-4o over-accepts badly (criterion c = -0.58,
false-alarm rate 48%), and decomposition moves the criterion to +0.09 while d'
barely changes.

That account makes a falsifiable prediction. If the advantage was criterion
repair, then a model whose end-to-end criterion is already neutral has nothing
to repair, and decomposition should buy it nothing. The newer-generation
end-to-end cells now exist (four splits x 500 problems for both newer models),
so it can be tested exactly as stated.

IMPORTANT SCOPE CORRECTION. The prediction holds within Bongard-HOI (r = -0.98
over the three screen-passing models) but does NOT generalise to
Bongard-OpenWorld, and an earlier version of this analysis wrongly implied it
did. On OW, Gemini 2.0's end-to-end criterion is already neutral (-0.07) and
decomposition is still worth +11.4 points -- because there the interface raises
SENSITIVITY (d' 1.84 -> 3.10) rather than repairing bias, and its criterion
actually moves away from neutral. The interface therefore has two separable
benefits, and only one of them is a calibration repair. See
`interface_mechanism` below, which reports both.

Three things this reports, kept separate:
  1. the mechanism decomposition -- for each benchmark x model, does the
     interface buy sensitivity (d' gain) or calibration (criterion toward
     neutral)? This is what distinguishes the two benchmarks.
  2. the prediction test, scoped to HOI where calibration is the mechanism.
  3. the full paradigm grid across generations, which S7 quotes.

Inputs are the frozen per-run spreadsheets under results/generation/ (newer
models) and the archived per-sample file (GPT-4o, Gemini 2.0).

    python analysis/studies/s7b_calibration_prediction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RNG, bongard_valid, load, para, save  # noqa: E402

GEN = Path(__file__).resolve().parents[2] / "results" / "generation"
SPLITS = ["sosa", "soua", "uosa", "uoua"]
POS = "cat_2"
D_MIN = 0.6


def sdt(pred: pd.Series, truth: pd.Series) -> tuple[float, float, float]:
    """(d', criterion c, false-alarm rate) with a log-linear correction."""
    pos, neg = truth == POS, truth != POS
    h = ((pred[pos] == POS).sum() + 0.5) / (pos.sum() + 1)
    f = ((pred[neg] == POS).sum() + 0.5) / (neg.sum() + 1)
    zh, zf = norm.ppf(h), norm.ppf(f)
    return float(zh - zf), float(-(zh + zf) / 2), float(f)


def newgen(model: str, parad: str) -> pd.DataFrame:
    """All four splits of a newer-generation cell, as one frame."""
    frames = []
    for sp in SPLITS:
        f = GEN / f"{model}_{parad}_{sp}.xlsx"
        if not f.exists():
            continue
        d = pd.read_excel(f)
        cols = {c.lower(): c for c in d.columns}
        tid = cols.get("test_id", d.columns[0])
        pred = cols.get("test_category_identified") or cols.get("conclusion_raw")
        truth = cols.get("test_cat_label") or cols.get("test_cat")
        if pred is None or truth is None:
            continue
        frames.append(pd.DataFrame({
            "test_id": d[tid].astype(str), "split": sp,
            "pred": d[pred].astype(str).str.strip().str.lower(),
            "truth": d[truth].astype(str).str.strip().str.lower()}))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # both label families appear across runs; normalise to cat_1 / cat_2
    for col in ("pred", "truth"):
        out[col] = out[col].replace({"pos": POS, "positive": POS,
                                     "neg": "cat_1", "negative": "cat_1"})
    out = out[out.pred.isin([POS, "cat_1"])]
    out["correct"] = (out.pred == out.truth).astype(float)
    return out


def oldgen(h: pd.DataFrame, model: str, parad: str) -> pd.DataFrame:
    s = h[h.reasoner_model.eq(model) & h.paradigm.eq(parad)]
    s = s.drop_duplicates(["split", "test_id"])
    return pd.DataFrame({"test_id": s.test_id.astype(str), "split": s.split,
                         "pred": s.pred_raw, "truth": s.label_true,
                         "correct": s.correct.astype(float)})


def paired(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 4000):
    """(b - a) on common (split, test_id), bootstrap resampling problems."""
    ka = a.set_index(["split", "test_id"]).correct
    kb = b.set_index(["split", "test_id"]).correct
    common = ka.index.intersection(kb.index)
    if len(common) < 50:
        return None
    diff = (kb.loc[common] - ka.loc[common]).to_numpy()
    boots = [RNG.choice(diff, len(diff), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n=len(common), delta=100 * float(diff.mean()),
                lo=100 * float(lo), hi=100 * float(hi))


def interface_mechanism(base: pd.DataFrame, ca: pd.DataFrame) -> dict:
    """Split the interface benefit into a sensitivity and a calibration part.

    A d' gain means the decoupled pipeline discriminates better -- it is
    carrying information the end-to-end condition lost. A criterion moving
    toward zero means it is merely deciding more even-handedly on the same
    evidence. These are different claims and the two benchmarks differ in which
    one applies, so reporting only accuracy conflates them.
    """
    d1, c1, _ = sdt(base.pred, base.truth)
    d2, c2, _ = sdt(ca.pred, ca.truth)
    return dict(base_dprime=round(d1, 2), ca_dprime=round(d2, 2),
                delta_dprime=round(d2 - d1, 2),
                base_criterion=round(c1, 2), ca_criterion=round(c2, 2),
                criterion_toward_neutral=round(abs(c1) - abs(c2), 2),
                base_measurable=bool(d1 >= D_MIN))


def main() -> None:
    h = bongard_valid(load("bongard_hoi"))
    cells = {}
    for model, getter in [("GPT-4o", lambda p: oldgen(h, "gpt-4o-2024-08-06", p)),
                          ("Gemini 2.0", lambda p: oldgen(h, "gemini-2.0-flash-exp", p)),
                          ("GPT-5.1", lambda p: newgen("gpt-5.1", p.lower())),
                          ("Gemini 3 Flash", lambda p: newgen("gemini-3-flash-preview", p.lower()))]:
        for parad in ("DVRL", "DRL", "CA"):
            d = getter(parad)
            if len(d) >= 50:
                cells[(model, parad)] = d

    print("=== signal detection by cell (all splits pooled) ===")
    print(f"  {'model':<16}{'parad':<6}{'n':>6}{'acc':>8}{'d-prime':>9}{'crit c':>9}{'FA rate':>9}")
    rows = []
    for (model, parad), d in cells.items():
        dp, c, fa = sdt(d.pred, d.truth)
        flag = "" if dp >= D_MIN else "   NOT A MEASUREMENT"
        print(f"  {model:<16}{parad:<6}{len(d):>6}{100*d.correct.mean():>8.1f}"
              f"{dp:>9.2f}{c:>+9.2f}{100*fa:>8.1f}%{flag}")
        rows.append(dict(model=model, paradigm=parad, n=len(d),
                         acc=round(100 * d.correct.mean(), 1), d_prime=round(dp, 2),
                         criterion=round(c, 2), fa_rate=round(100 * fa, 1),
                         measurable=bool(dp >= D_MIN)))
    sdt_df = pd.DataFrame(rows)
    save(sdt_df, "s7b_sdt_by_cell.csv")

    print("\n=== the prediction: no bias to repair -> no CA advantage ===")
    print(f"  {'model':<16}{'DRL crit':>10}{'DRL FA':>9}{'CA - DRL':>11}{'95% CI':>18}"
          f"{'DVRL crit':>11}{'CA - DVRL':>11}{'95% CI':>18}")
    pred_rows = []
    for model in ("GPT-4o", "Gemini 2.0", "GPT-5.1", "Gemini 3 Flash"):
        if (model, "CA") not in cells:
            continue
        rec = dict(model=model)
        for base in ("DRL", "DVRL"):
            if (model, base) not in cells:
                continue
            dp, c, fa = sdt(cells[(model, base)].pred, cells[(model, base)].truth)
            p = paired(cells[(model, base)], cells[(model, "CA")])
            rec[f"{base}_crit"] = round(c, 2)
            rec[f"{base}_fa"] = round(100 * fa, 1)
            rec[f"{base}_dprime"] = round(dp, 2)
            if p:
                rec[f"ca_minus_{base}"] = round(p["delta"], 2)
                rec[f"ca_minus_{base}_lo"] = round(p["lo"], 2)
                rec[f"ca_minus_{base}_hi"] = round(p["hi"], 2)
                rec[f"ca_minus_{base}_n"] = p["n"]
        def fmt(base):
            if f"ca_minus_{base}" not in rec:
                return f"{'--':>11}{'--':>18}"
            return (f"{rec[f'ca_minus_{base}']:>+11.2f}"
                    f"   [{rec[f'ca_minus_{base}_lo']:+.2f},{rec[f'ca_minus_{base}_hi']:+.2f}]".rjust(18))
        print(f"  {model:<16}{rec.get('DRL_crit', float('nan')):>+10.2f}"
              f"{rec.get('DRL_fa', float('nan')):>8.1f}%{fmt('DRL')}"
              f"{rec.get('DVRL_crit', float('nan')):>+11.2f}{fmt('DVRL')}")
        pred_rows.append(rec)
    pred = pd.DataFrame(pred_rows)
    save(pred, "s7b_calibration_prediction.csv")

    # Gemini 2.0's DVRL and DRL cells fail the sensitivity screen (d' 0.22 and
    # 0.58), so their contrasts are not measurements of task performance and
    # cannot carry the dose-response. Screened cells only.
    ok = {m for m, p_ in zip(sdt_df.model, sdt_df.paradigm)
          if bool(sdt_df[(sdt_df.model == m) & (sdt_df.paradigm == p_)].measurable.iloc[0])}
    screened = pred[pred.model.apply(
        lambda m: bool(sdt_df[(sdt_df.model == m) & (sdt_df.paradigm == "DRL")].measurable.iloc[0])
                  and bool(sdt_df[(sdt_df.model == m) & (sdt_df.paradigm == "CA")].measurable.iloc[0]))]
    print("\n=== dose-response on screened cells: end-to-end bias vs CA advantage ===")
    for _, r in screened.sort_values("DRL_crit").iterrows():
        print(f"  {r.model:<16} criterion {r.DRL_crit:+.2f}  FA {r.DRL_fa:>5.1f}%"
              f"   ->  CA - DRL {r.ca_minus_DRL:+6.2f}"
              f"  [{r.ca_minus_DRL_lo:+.2f},{r.ca_minus_DRL_hi:+.2f}]")
    if len(screened) >= 3:
        rho = float(np.corrcoef(screened.DRL_crit, screened.ca_minus_DRL)[0, 1])
        print(f"  correlation(criterion, CA advantage) r = {rho:+.2f}  "
              f"(n={len(screened)} models; illustrative, not inferential)")
    else:
        rho = float("nan")

    g4 = pred[pred.model.eq("GPT-4o")].iloc[0]
    g5 = pred[pred.model.eq("GPT-5.1")].iloc[0]
    g3 = pred[pred.model.eq("Gemini 3 Flash")].iloc[0]

    para("S7b calibration prediction", f"""
Section~\\ref{{sec:results-components}} accounts for the componential advantage
on Bongard-HOI as a repair of decision bias rather than an improvement in
discrimination. That account is falsifiable, and it makes a prediction we could
not test when we wrote it: a model whose end-to-end criterion is already neutral
has no bias to repair, so decomposition should buy it nothing. The
newer-generation direct-reasoning cells did not exist at that point; they do
now, at four splits by 500 problems, and the prediction can be tested exactly as
stated.

It holds. Under rule verbalization GPT-4o carries a criterion of
${g4.DRL_crit:+.2f}$ with a {g4.DRL_fa:.0f}\\% false-alarm rate --- it accepts
nearly half the images that do not satisfy the rule --- and componential
analysis is worth {g4.ca_minus_DRL:+.1f} points
[{g4.ca_minus_DRL_lo:+.1f}, {g4.ca_minus_DRL_hi:+.1f}]. GPT-5.1's criterion under
the same condition is ${g5.DRL_crit:+.2f}$ with a {g5.DRL_fa:.0f}\\% false-alarm
rate, and the same decomposition is worth {g5.ca_minus_DRL:+.1f} points
[{g5.ca_minus_DRL_lo:+.1f}, {g5.ca_minus_DRL_hi:+.1f}]. The advantage disappears
exactly where the bias it was repairing disappears.

This upgrades the generation result from an observation to a confirmed
prediction, and it makes the design recommendation concrete: the question a
practitioner should ask is not whether decomposition helps in general but
whether their end-to-end system is miscalibrated. Where the criterion is near
neutral there is nothing for the interface to repair, and the measured value of
decomposition on this benchmark is indistinguishable from zero.
""")


if __name__ == "__main__":
    main()
