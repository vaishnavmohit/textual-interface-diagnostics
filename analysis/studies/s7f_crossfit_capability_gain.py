#!/usr/bin/env python3
"""Cross-fitted check of the capability--interface-gain association.

The descriptive analysis in s7c correlates DRL sensitivity with CA--DRL
accuracy change. Because DRL appears on both axes, that correlation is exposed
to mathematical coupling. This script estimates the axes on disjoint items.

For each repeated split, native visual sensitivity is computed from one half of
the common DRL/CA items and the perception-step gain from the other half. The
orientation is then reversed. The output is a distribution over item splits,
not a model-population confidence interval; there are still only five models.
"""
from __future__ import annotations

import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import bongard_valid, load, save  # noqa: E402

N_SPLITS = 2000
SEED = 20260804
GEN = Path(__file__).resolve().parents[2] / "results" / "generation"
SPLITS = ["sosa", "soua", "uosa", "uoua"]
POS = "cat_2"


def norm_lab(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip().str.lower()
            .replace({"pos": POS, "positive": POS,
                      "neg": "cat_1", "negative": "cat_1"}))


def newgen(model: str, paradigm: str) -> pd.DataFrame:
    frames = []
    for split in SPLITS:
        path = GEN / f"{model}_{paradigm.lower()}_{split}.xlsx"
        d = pd.read_excel(path)
        pred = "test_category_identified" if "test_category_identified" in d else "conclusion_raw"
        truth = "test_cat_label" if "test_cat_label" in d else "test_cat"
        frames.append(pd.DataFrame({
            "split": split,
            "test_id": d.test_id.astype(str),
            "uid": d.uid.astype(str) if "uid" in d else d.test_id.astype(str),
            "pred": norm_lab(d[pred]),
            "truth": norm_lab(d[truth]),
        }))
    out = pd.concat(frames, ignore_index=True)
    out = out[out.pred.isin([POS, "cat_1"])].copy()
    out["correct"] = (out.pred == out.truth).astype(float)
    return out


def oldgen(h: pd.DataFrame, model: str, paradigm: str) -> pd.DataFrame:
    s = h[h.reasoner_model.eq(model) & h.paradigm.eq(paradigm)]
    s = s.drop_duplicates(["split", "test_id"])
    return pd.DataFrame({
        "split": s.split,
        "test_id": s.test_id.astype(str),
        "uid": s.uid.astype(str),
        "pred": s.pred_raw,
        "truth": s.label_true,
        "correct": s.correct.astype(float),
    }).dropna(subset=["correct"])


def dprime(truth: pd.Series, pred: pd.Series) -> float:
    pos = truth == POS
    neg = ~pos
    hit = ((pred[pos] == POS).sum() + 0.5) / (pos.sum() + 1)
    false_alarm = ((pred[neg] == POS).sum() + 0.5) / (neg.sum() + 1)
    normal = NormalDist()
    return float(normal.inv_cdf(hit) - normal.inv_cdf(false_alarm))


def spearman(x: pd.Series, y: pd.Series) -> float:
    return float(x.rank(method="average").corr(y.rank(method="average")))


def paired_common(drl: pd.DataFrame, ca: pd.DataFrame) -> pd.DataFrame:
    keys = ["split", "test_id"]
    a = drl.drop_duplicates(keys).set_index(keys)
    b = ca.drop_duplicates(keys).set_index(keys)
    common = a.index.intersection(b.index)
    out = a.loc[common, ["uid", "pred", "truth", "correct"]].copy()
    out = out.rename(columns={"pred": "drl_pred", "correct": "drl_correct"})
    out["ca_correct"] = b.loc[common, "correct"].to_numpy()
    return out.reset_index()


def main() -> None:
    archived = bongard_valid(load("bongard_hoi"))
    getters = [
        ("Gemini 2.0", lambda p: oldgen(archived, "gemini-2.0-flash-exp", p)),
        ("Gemini 3.5 Flash Lite", lambda p: newgen("gemini-3.5-flash-lite", p)),
        ("GPT-4o", lambda p: oldgen(archived, "gpt-4o-2024-08-06", p)),
        ("GPT-5.1", lambda p: newgen("gpt-5.1", p)),
        ("Gemini 3 Flash", lambda p: newgen("gemini-3-flash-preview", p)),
    ]
    cells = {}
    for model, getter in getters:
        cells[model] = paired_common(getter("DRL"), getter("CA"))

    rng = np.random.default_rng(SEED)
    rows = []
    for rep in range(N_SPLITS):
        estimates = {0: [], 1: []}
        for model, d in cells.items():
            # Assign whole uid clusters so paired query cases never cross folds.
            uids = d.uid.astype(str).unique()
            fold_by_uid = dict(zip(uids, rng.integers(0, 2, len(uids))))
            fold = d.uid.astype(str).map(fold_by_uid).to_numpy()
            for orientation in (0, 1):
                native = d[fold == orientation]
                gain = d[fold != orientation]
                if len(native) < 40 or len(gain) < 40:
                    continue
                dp = dprime(native.truth, native.drl_pred)
                delta = 100 * (gain.ca_correct - gain.drl_correct).mean()
                estimates[orientation].append((model, dp, delta))

        for orientation, vals in estimates.items():
            if len(vals) != len(cells):
                continue
            frame = pd.DataFrame(vals, columns=["model", "drl_dprime", "gain"])
            rho = spearman(frame.drl_dprime, frame.gain)
            rows.append({"split": rep, "orientation": orientation, "rho": rho})

    out = pd.DataFrame(rows)
    save(out, "s7f_crossfit_capability_gain_draws.csv")
    summary = pd.DataFrame([{
        "n_models": len(cells),
        "n_crossfit_draws": len(out),
        "median_rho": out.rho.median(),
        "q025_rho": out.rho.quantile(0.025),
        "q975_rho": out.rho.quantile(0.975),
        "fraction_negative": (out.rho < 0).mean(),
        "fraction_perfect_negative": (out.rho == -1).mean(),
    }])
    save(summary, "s7f_crossfit_capability_gain_summary.csv")
    print(summary.to_string(index=False))
    print("\nInterpretation: axes are estimated on disjoint item folds. The interval")
    print("describes split sensitivity, not uncertainty over a population of models.")


if __name__ == "__main__":
    main()
