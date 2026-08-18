#!/usr/bin/env python3
"""Audit the two Bongard-HOI cohorts without querying any model.

The paper uses two deliberately different views of the stored runs:

* ``common-100``: the 100 test IDs per split used by the original GPT-4o
  runs.  This is the only cohort on which all five model configurations can
  be shown side by side.
* ``full-run``: every archived valid binary prediction for a model/paradigm.
  Only GPT-5.1 and Gemini 3 Flash were scheduled beyond 100 per split.

This script makes the distinction auditable.  It records archived and valid
counts, verifies set membership against the common manifest, and reports the
actual paired denominator for every workflow contrast.  It performs no model
calls and does not alter source predictions.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "results" / "generation"
OUT = REPO / "results" / "studies"
SPLITS = ["sosa", "soua", "uosa", "uoua"]
PARADIGMS = ["DVRL", "DRL", "CA"]
USABLE = ["include", "include-valid-rows-only"]


def norm_label(x: pd.Series) -> pd.Series:
    return (x.astype(str).str.strip().str.lower()
            .replace({"pos": "cat_2", "positive": "cat_2",
                      "neg": "cat_1", "negative": "cat_1"}))


def archived_old(h: pd.DataFrame, model: str, paradigm: str) -> pd.DataFrame:
    d = h[(h.reasoner_model == model) & (h.paradigm == paradigm)].copy()
    return d.drop_duplicates(["split", "test_id"])


def archived_new(model: str, paradigm: str) -> pd.DataFrame:
    frames = []
    for split in SPLITS:
        path = GEN / f"{model}_{paradigm.lower()}_{split}.xlsx"
        if not path.exists():
            continue
        d = pd.read_excel(path)
        pred = "test_category_identified" if "test_category_identified" in d else "conclusion_raw"
        z = pd.DataFrame({"split": split, "test_id": d.test_id.astype(str),
                          "pred": norm_label(d[pred])})
        z["pred_valid"] = z.pred.isin(["cat_1", "cat_2"])
        frames.append(z)
    return pd.concat(frames, ignore_index=True).drop_duplicates(["split", "test_id"])


def main() -> None:
    h = pd.read_csv(REPO / "results" / "bongard_hoi_per_sample.csv.gz", low_memory=False)
    h = h[h.analysis_decision.isin(USABLE)].copy()
    h["test_id"] = h.test_id.astype(str)

    # GPT-4o is complete in all three paradigms and defines the original
    # 100-per-split manifest.  Fail loudly if that ceases to be true.
    gpt4 = {p: archived_old(h, "gpt-4o-2024-08-06", p) for p in PARADIGMS}
    common = {s: set(gpt4["DVRL"].loc[gpt4["DVRL"].split == s, "test_id"])
              for s in SPLITS}
    for split in SPLITS:
        assert len(common[split]) == 100, (split, len(common[split]))
        for paradigm in PARADIGMS:
            ids = set(gpt4[paradigm].loc[gpt4[paradigm].split == split, "test_id"])
            assert ids == common[split], (split, paradigm)

    models = [
        ("GPT-4o", 100, lambda p: archived_old(h, "gpt-4o-2024-08-06", p)),
        ("Gemini 2.0", 100, lambda p: archived_old(h, "gemini-2.0-flash-exp", p)),
        ("GPT-5.1", 500, lambda p: archived_new("gpt-5.1", p)),
        ("Gemini 3 Flash", 500, lambda p: archived_new("gemini-3-flash-preview", p)),
        ("Gemini 3.5 Flash Lite", 100,
         lambda p: archived_new("gemini-3.5-flash-lite", p)),
    ]

    rows = []
    cached: dict[tuple[str, str], pd.DataFrame] = {}
    for name, scheduled, loader in models:
        for paradigm in PARADIGMS:
            d = loader(paradigm).copy()
            if "pred_valid" not in d:
                d["pred_valid"] = d.pred_valid.fillna(False).astype(bool)
            d["test_id"] = d.test_id.astype(str)
            cached[(name, paradigm)] = d
            for split in SPLITS:
                z = d[d.split == split]
                ids = set(z.test_id)
                valid = set(z.loc[z.pred_valid, "test_id"])
                rows.append({
                    "model": name, "paradigm": paradigm, "split": split,
                    "scheduled_n": scheduled, "archived_n": len(ids),
                    "valid_binary_n": len(valid),
                    "common100_archived_n": len(ids & common[split]),
                    "common100_valid_n": len(valid & common[split]),
                    "outside_common_n": len(ids - common[split]),
                    "common100_complete": common[split].issubset(ids),
                })

    counts = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    counts.to_csv(OUT / "s7j_hoi_cohort_audit.csv", index=False)

    contrasts = [("DVRL", "DRL", "rule staging"),
                 ("DRL", "CA", "description workflow"),
                 ("DVRL", "CA", "complete ladder")]
    paired = []
    for name, scheduled, _ in models:
        for a, b, label in contrasts:
            da, db = cached[(name, a)], cached[(name, b)]
            for scope in ["common-100", "full-run"]:
                total = 0
                per_split = []
                for split in SPLITS:
                    ia = set(da.loc[(da.split == split) & da.pred_valid, "test_id"])
                    ib = set(db.loc[(db.split == split) & db.pred_valid, "test_id"])
                    if scope == "common-100":
                        ia, ib = ia & common[split], ib & common[split]
                    n = len(ia & ib)
                    total += n
                    per_split.append(str(n))
                paired.append({"model": name, "scheduled_per_split": scheduled,
                               "scope": scope, "transition": label,
                               "contrast": f"{b} - {a}", "paired_n": total,
                               "paired_n_by_split": "/".join(per_split)})
    paired = pd.DataFrame(paired)
    paired.to_csv(OUT / "s7j_hoi_paired_denominators.csv", index=False)

    print("Wrote cohort counts and paired denominators.")
    print("\nCommon-100 archival gaps (zero means the intended manifest is complete):")
    gaps = counts.assign(gap=100 - counts.common100_archived_n)
    print(gaps[gaps.gap > 0][["model", "paradigm", "split", "gap"]].to_string(index=False))
    print("\nPaired denominators used by inferential contrasts:")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
