#!/usr/bin/env python3
"""Sensitivity audit for missing Gemini-3-Flash description artifacts.

Uses only frozen benchmark manifests and stored outputs.  It asks whether the
500 scheduled cases lacking a reusable description artifact differ in class
composition or in valid visual-DRL accuracy.  This is a selection audit, not
an imputation and not a new experiment.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[2]
SAMEMODEL = ROOT / "results" / "samemodel"
GEN = ROOT / "results" / "generation"
OUT = ROOT / "results" / "studies" / "s7k_description_missingness.csv"
FILES = {
    "sosa": "seen_obj_seen_act", "soua": "seen_obj_unseen_act",
    "uosa": "unseen_obj_seen_act", "uoua": "unseen_obj_unseen_act",
}


def manifest(split: str) -> pd.DataFrame:
    path = ROOT / "code" / "data" / "bongard_hoi" / f"bongard_hoi_test_{FILES[split]}.json"
    raw = json.loads(path.read_text())[:500]
    return pd.DataFrame({"test_id": [str(x["test_id"]) for x in raw],
                         "positive": [x["testfiles"]["category"] == "cat_2" for x in raw]})


def main() -> None:
    unresolved = json.loads((SAMEMODEL / "unresolved_ids.json").read_text())
    rows = []
    for split in FILES:
        source = manifest(split)
        text = pd.read_excel(SAMEMODEL / f"hoi_{split}_g3f_textdrl.xlsx")
        supported = set(text.test_id.astype(str)) | set(map(str, unresolved.get(split, [])))
        source["artifact_supported"] = source.test_id.isin(supported)

        visual = pd.read_excel(GEN / f"gemini-3-flash-preview_dvrl_{split}.xlsx")
        pred = "test_category_identified" if "test_category_identified" in visual else "conclusion_raw"
        pred_norm = (visual[pred].astype(str).str.strip().str.lower()
                     .replace({"pos": "cat_2", "neg": "cat_1"}))
        truth_norm = visual["test_cat_label"].astype(str).str.strip().str.lower()
        valid = pred_norm.isin(["cat_1", "cat_2"])
        visual = visual.loc[valid, ["test_id"]].copy()
        visual["is_correct"] = (pred_norm[valid].to_numpy() == truth_norm[valid].to_numpy())
        visual["test_id"] = visual.test_id.astype(str)
        source = source.merge(visual, on="test_id", how="left")

        tab = pd.crosstab(source.artifact_supported, source.positive).reindex(
            index=[True, False], columns=[True, False], fill_value=0)
        p_label = fisher_exact(tab.to_numpy()).pvalue
        for available, label in [(True, "supported"), (False, "missing")]:
            g = source[source.artifact_supported == available]
            scored = g[g.is_correct.notna()]
            rows.append({
                "split": split, "artifact_group": label, "scheduled_n": len(g),
                "positive_pct": round(100 * g.positive.mean(), 1),
                "valid_visual_n": len(scored),
                "visual_dvrl_accuracy_pct": round(100 * scored.is_correct.astype(float).mean(), 1),
                "fisher_p_class_composition": p_label,
            })
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
