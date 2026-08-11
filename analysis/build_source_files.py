#!/usr/bin/env python3
"""Rebuild every canonical per-sample source file from the raw HPC pull.

This is the single reproducible step between "spreadsheets copied off the
cluster" and "tables in the paper". Run it and every downstream analysis is
regenerable without cluster access.

    python analysis/build_source_files.py --raw ~/Downloads/hpc_results

Inputs  : the raw pull (see scripts/pull_from_hpc.sh)
Outputs : results/bongard_ow_per_sample.csv.gz
          results/bongard_hoi_per_sample.csv.gz
          results/winoground_per_sample.csv.gz
          provenance/EXPERIMENT_REGISTRY.csv   <- what every run IS

The registry is the point. Directory names on the cluster are historical and
inconsistent; this file encodes, once, which run is which paradigm, which model
produced its descriptions, and whether it may be used. Nothing is deleted:
unusable runs are kept and marked, with the reason.

Naming conventions decoded here (confirmed against the run configs and, for
Winoground, against description content hashes):

  Bongard-OW   *_multi_test*, *dvrl*        -> DVRL   (all images, one pass)
               *_multi*, *drl*              -> DRL    (derive rule, then apply)
               *single_comp_eval_<R>*       -> CA, GPT-4o descriptions
               *single_mistral_comp_eval_*  -> CA, Pixtral descriptions
               *single_<M>*, *_ca_<M>*      -> CA, the model's own descriptions
               *eval_rule*, *groundeval*    -> RuleApply (rule supplied)
               *single_test*                -> CA-describe (emits descriptions)

  Bongard-HOI  <split>/single|multi|multi_test_<M>   -> CA | DRL | DVRL  (n=100)
               <split>/{ca,drl,dvrl}/<M>             -> newer generation

  Winoground   base_dir names the PERCEPTION source, `model` the reasoner:
               gpt_activity/    -> GPT-4o descriptions
               gemini_activity/ -> Gemini descriptions
               winoground_25activity/ -> Gemini descriptions (same artefacts:
                   400/400 description.json identical to gemini_activity)
"""
from __future__ import annotations

import argparse
import hashlib
import re
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# dataset/bongard_ow.py: imagefiles["cat_2"] = positive, ["cat_1"] = negative
VOCAB = {"cat_1", "cat_2"}
SPLITS = {"sosa", "soua", "uosa", "uoua"}


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def classify_bongard(parts: list[str]) -> tuple[str, str]:
    """(paradigm, description source) from an experiment path."""
    toks = [p.lower() for p in parts]
    j = "/".join(toks)
    if "eval_rule" in j or "groundeval" in j:      return "RuleApply", "given rule"
    if any(t == "dvrl" for t in toks):             return "DVRL", "n/a (end-to-end)"
    if any(t == "drl" for t in toks):              return "DRL", "n/a (end-to-end)"
    if any(t == "ca" for t in toks):               return "CA", "self"
    if "multi_test" in j or "dvrl" in j:           return "DVRL", "n/a (end-to-end)"
    if "single_mistral_comp_eval" in j:            return "CA", "pixtral"
    if "comp_eval" in j:                           return "CA", "gpt-4o"
    if "single_test" in j:                         return "CA-describe", "(produces gpt-4o descriptions)"
    if "multi" in j or "drl" in j:                 return "DRL", "n/a (end-to-end)"
    if "single" in j or "_ca_" in j:               return "CA", "self"
    return "?", "?"


def ablation_of(s: str) -> str:
    s = s.lower()
    tags = [("noise", "noise"), ("ablation", "support-size"), ("cot", "CoT"),
            ("_pos", "pos-only"), ("gpt5.1", "GPT-5.1"), ("gpt-5.1", "GPT-5.1"),
            ("_4096", "ctx-4096"), ("_8196", "ctx-8196"), ("rerun", "rerun")]
    out = []
    for k, v in tags:
        if k in s and v not in out:
            out.append(v)
    return "+".join(out)


def failure_mode(bad_values: pd.Series) -> str:
    """Classify *why* outputs were invalid. These are not one phenomenon.

    Three distinct causes appear in the data, and they demand different
    responses — only the first is a harness bug we can fix and re-run:

    schema_vocabulary  the model emitted `pos`/`neg` because Ollama's `format=`
                       enum constrained generation to the wrong vocabulary. The
                       model could reason but not express the answer. Re-runnable
                       under the fixed schema.
    template_echo      the model returned the instruction text itself
                       ("cat_1 or cat_2") instead of choosing. A formatting
                       failure; re-running may fix it.
    capability_refusal the model states it did not receive or cannot process the
                       images ("I need the 12 images to perform the analysis").
                       NOT a harness bug: it is evidence that the condition —
                       13 images in one pass — is at the edge of what the model
                       handles. Re-running would paper over a real finding.

    Two further modes appear at low rates:

    empty_output       no text at all (NaN) — an API or serialisation failure.
    abstention         the model explicitly declined ("neither cat_1 nor cat_2",
                       "unable to determine"). A genuine model behaviour, and
                       arguably the correct response on an ambiguous item.
    parse_failure      the model DID commit to a label inside prose ("so it is
                       categorized as `cat_1`") but the extractor did not pull it
                       out. Recoverable in principle without re-running, so it
                       is counted separately rather than lumped with real
                       failures.
    """
    if not len(bad_values):
        return ""
    v = bad_values.astype(str)
    if v.isin(["pos", "neg"]).mean() > 0.5:
        return "schema_vocabulary"
    low = v.str.lower().str.strip()
    if (low == "nan").mean() > 0.5:
        return "empty_output"
    if low.str.contains("lack the ability|need the .* images|only process one image").any():
        return "capability_refusal"
    if low.str.fullmatch(r"cat_1 or cat_2").any():
        return "template_echo"
    if low.str.contains("neither|unable to determine|does not fit|cannot be determined").any():
        return "abstention"
    # a label is present in the prose but was not extracted
    if low.str.contains(r"cat_[12]").any():
        return "parse_failure"
    return "other"


def decide(exp_dir: str, n: int, invalid_frac: float) -> tuple[str, str]:
    """Whether a run may be used, and why. Author decisions recorded 2026-07-17."""
    d = exp_dir.lower()
    if n <= 20:                        return "exclude", "empty/crashed run (<=20 rows)"
    if "llama8b_delete" in d:          return "exclude", "author decision: skip"
    if "not_working" in d:             return "exclude", "run self-reports failure"
    # Duplicate of uoua/multi_gemini_2_flash_exp under a suffixed name. The paper
    # used the unsuffixed run (65.0); this one gives 63.0, and pooling the two
    # silently averages them.
    if "_8150" in d:                   return "exclude", "duplicate of the unsuffixed run; paper used that one"
    # output_bongard_hoi_rerun: later re-runs that were NOT used in the paper
    # (its sosa/multi_test_gpt4o gives 73.0 where the paper reports 68). The _NN
    # suffixes encode the accuracy each achieved.
    if "rerun" in exp_dir or re.search(r"_\d{2}$", d):
        return "exclude", "re-run not used in the paper; superseded by output_bongard_hoi"
    if invalid_frac > 0.95:            return "exclude", "outputs schema-corrupted (wrong label vocabulary), degenerate"
    if invalid_frac > 0.02:            return "include-valid-rows-only", f"{invalid_frac*100:.1f}% schema-corrupted rows excluded"
    if "gpt5.1" in d or "gpt-5.1" in d: return "exploratory-only", "recent-model demo; barred from common-set tables"
    return "include", ""


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def build_bongard(raw: Path, roots: list[tuple[str, str]], benchmark: str) -> pd.DataFrame:
    frames = []
    for root, _ in roots:
        base = raw / root
        if not base.exists():
            continue
        for f in sorted(base.rglob("results.xlsx")):
            rel = f.relative_to(base)
            parts = list(rel.parts[:-1])
            split = next((p for p in parts if p.lower() in SPLITS), "")
            eff = [p for p in parts if p.lower() not in SPLITS]
            paradigm, components = classify_bongard(eff)
            try:
                d = pd.read_excel(f)
            except Exception:
                continue
            if "test_cat_label" not in d.columns:
                continue
            if paradigm == "CA-describe":            # emits descriptions, not decisions
                continue
            gt = d["test_cat_label"].astype(str).str.strip()
            pr = d["test_category_identified"].astype(str).str.strip()
            valid = pr.isin(VOCAB)
            inv_frac = 1 - valid.mean() if len(d) else 1.0
            exp_dir = "/".join(parts)
            fmode = failure_mode(pr[~valid])
            dec, why = decide(f"{root}/{exp_dir}", len(d), inv_frac)
            if fmode == "capability_refusal" and dec == "include-valid-rows-only":
                why = ("model reported it could not receive/process the images; "
                       "a condition limitation, not a harness fault")
            model = ";".join(sorted(set(d["model"].dropna().astype(str))))[:40] if "model" in d.columns else ""
            o = pd.DataFrame({
                "benchmark": benchmark, "output_root": root, "split": split,
                "experiment_dir": exp_dir, "paradigm": paradigm, "components": components,
                "reasoner_model": model, "ablation": ablation_of(exp_dir + " " + root),
                "analysis_decision": dec, "decision_reason": why,
                "failure_mode": fmode,
                "test_id": d.get("test_id"), "uid": d.get("uid"),
                "concept": d.get("concept"), "commonsense": d.get("commonsense"),
                "label_true": gt, "pred_raw": pr, "pred_valid": valid,
                "is_correct": (pr == gt) & valid,
                "source_sha256": hashlib.sha256(f.read_bytes()).hexdigest()[:16],
            })
            o.loc[~valid, "is_correct"] = pd.NA
            frames.append(o)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def wino_perception(rel: str) -> tuple[str, str]:
    r = rel.lower()
    if "gemini_activity" in r:
        return "gemini", "config base_dir"
    if "gpt_activity" in r:
        return "gpt-4o", "config base_dir"
    if "25activity" in r:
        # 400/400 description.json identical to gemini_activity; the gpt-4o score
        # file even shares an md5 with gemini_activity's copy.
        return "gemini", "config base_dir + identical description content (400/400)"
    if "winoground_activity" in r or "activity_t1" in r:
        return "unknown", "unlabelled directory"
    if "winoground_direct" in r:
        return "n/a (direct)", "end-to-end, no descriptions"
    return "unknown", ""


def wino_variant(rel: str, name: str) -> str:
    n = (rel + " " + name).lower()
    tags = [("wrongentries", "FLAGGED:wrongentries"), ("notworking", "FLAGGED:notworking"),
            ("single_qn", "single-question"), ("no_commonsense", "no-commonsense"),
            ("_multi", "multi-question"), ("_v2", "v2"), ("trial2", "trial2"),
            ("_t1", "temp_t1"), ("_t0", "temp_t0"), ("no worldly", "no-worldly-knowledge")]
    return "+".join(v for k, v in tags if k in n)


def build_winoground(raw: Path) -> pd.DataFrame:
    files = sorted(list((raw / "output_winoground").rglob("*.xlsx"))
                   + list(raw.glob("output/winoground*/**/*.xlsx")))
    frames = []
    for f in files:
        rel = str(f.relative_to(raw))
        try:
            d = pd.read_excel(f)
        except Exception:
            continue
        if "group_score" not in d.columns:
            continue
        model = ";".join(sorted(set(d["model"].dropna().astype(str))))[:40] if "model" in d.columns else ""
        skip = {"output_winoground", "winoground", "winoground_25activity",
                "winoground_activity", "activity_t1", "winoground_direct"}
        reasoner = f.parent.name if f.parent.name not in skip else model
        perc, evidence = wino_perception(rel)
        variant = wino_variant(rel, f.name)
        cond = "ICA" if "_ica" in f.name.lower() else ("DVRL" if "direct" in rel else "CA")
        n = len(d)
        status = "complete" if n >= 390 else ("partial" if n >= 50 else "fragment")
        dec = ("exclude" if ("FLAGGED" in variant or status == "fragment")
               else ("include" if status == "complete" else "partial-run"))
        frames.append(pd.DataFrame({
            "benchmark": "winoground", "source_file": rel, "perception": perc,
            "perception_evidence": evidence, "reasoner_model": reasoner or model,
            "condition": cond, "variant": variant, "n_rows": n, "status": status,
            "analysis_decision": dec,
            "id": d.get("id"), "tag": d.get("tag"), "collapsed_tag": d.get("collapsed_tag"),
            "text_score": d.get("text_score"), "image_score": d.get("image_score"),
            "group_score": d.get("group_score"),
            "has_reinspection": ("img0_questions" in d.columns),
            "source_sha256": hashlib.sha256(f.read_bytes()).hexdigest()[:16],
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def registry(ow: pd.DataFrame, hoi: pd.DataFrame, wino: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, bench in ((ow, "bongard_ow"), (hoi, "bongard_hoi")):
        if not len(t):
            continue
        g = t.groupby(["output_root", "experiment_dir"], dropna=False)
        for (root, ed), s in g:
            rows.append({
                "benchmark": bench, "output_root": root, "experiment_dir": ed,
                "split": s["split"].iloc[0], "paradigm": s["paradigm"].iloc[0],
                "descriptions_from": s["components"].iloc[0],
                "reasoner_model": s["reasoner_model"].iloc[0],
                "ablation": s["ablation"].iloc[0], "n_rows": len(s),
                "n_valid": int(s["pred_valid"].sum()),
                "invalid_pct": round((1 - s["pred_valid"].mean()) * 100, 1),
                "accuracy_valid_pct": round(s.loc[s["pred_valid"], "is_correct"].mean() * 100, 2)
                                       if s["pred_valid"].any() else None,
                "analysis_decision": s["analysis_decision"].iloc[0],
                "decision_reason": s["decision_reason"].iloc[0],
                "source_sha256": s["source_sha256"].iloc[0],
            })
    if len(wino):
        for sf, s in wino.groupby("source_file"):
            rows.append({
                "benchmark": "winoground", "output_root": sf.split("/")[0],
                "experiment_dir": sf, "split": "", "paradigm": s["condition"].iloc[0],
                "descriptions_from": s["perception"].iloc[0],
                "reasoner_model": s["reasoner_model"].iloc[0],
                "ablation": s["variant"].iloc[0], "n_rows": len(s), "n_valid": len(s),
                "invalid_pct": 0.0,
                "accuracy_valid_pct": round(s["group_score"].mean() * 100, 2),
                "analysis_decision": s["analysis_decision"].iloc[0],
                "decision_reason": s["perception_evidence"].iloc[0],
                "source_sha256": s["source_sha256"].iloc[0],
            })
    return pd.DataFrame(rows).sort_values(["benchmark", "paradigm", "descriptions_from", "experiment_dir"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="~/Downloads/hpc_results", help="root of the HPC pull")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    raw = Path(a.raw).expanduser()
    out = Path(a.out); out.mkdir(exist_ok=True)
    Path("provenance").mkdir(exist_ok=True)

    ow = build_bongard(raw, [("output", "")], "bongard_ow")
    ow = ow[ow.benchmark == "bongard_ow"]
    hoi = build_bongard(raw, [("output_bongard_hoi", ""), ("output_bongard_hoi_rerun", "")], "bongard_hoi")
    wino = build_winoground(raw)

    ow.to_csv(out / "bongard_ow_per_sample.csv.gz", index=False, compression="gzip")
    hoi.to_csv(out / "bongard_hoi_per_sample.csv.gz", index=False, compression="gzip")
    wino.to_csv(out / "winoground_per_sample.csv.gz", index=False, compression="gzip")
    reg = registry(ow, hoi, wino)
    reg.to_csv("provenance/EXPERIMENT_REGISTRY.csv", index=False)

    print(f"  bongard_ow   {len(ow):>7,} rows  {ow.experiment_dir.nunique():>3} runs")
    print(f"  bongard_hoi  {len(hoi):>7,} rows  {hoi.experiment_dir.nunique():>3} runs")
    print(f"  winoground   {len(wino):>7,} rows  {wino.source_file.nunique():>3} files")
    print(f"  registry     {len(reg):>7,} runs -> provenance/EXPERIMENT_REGISTRY.csv")
    print()
    print(reg.groupby(["benchmark", "analysis_decision"]).size().to_string())


if __name__ == "__main__":
    main()
