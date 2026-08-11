"""End-to-end multi-image paradigms: DVRL and DRL.

These have no separate perception stage — the VLM sees the images directly — so
there is no frozen description artifact (docs/REPRODUCIBILITY_DESIGN.md, "End-to-end
contexts"). Provenance is recorded inline in results.xlsx.

  context "dvrl": one call over all m+n support images plus the query.
  context "drl" : stage 1 derives a rule from the support set; stage 2 applies it
                  to the query image.

Only models able to ingest all m+n+1 images in one context can run these (the
capability constraint noted in the paper); CA/ICA are the per-image alternative.

Config (module: "pri.endtoend", function: "run"):
    dataset, json_path, image_dir, model, context ('dvrl'|'drl'),
    output_dir, experiment, split_name, num_samples, m, n,
    temperature, max_tokens, num_ctx, think
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from . import prompts as P
from . import provenance as prov
from .io_utils import set_seed
from .llm import REQUIRED_KEYS, llm_inference_multiimage, parse_json_dict
from .pipeline import RESULT_COLUMNS, _build_output_dir, _norm, _unpack

set_seed(42)


def _abs(image_dir: str, rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(image_dir, rel)


def run(
    dataset: str,
    json_path: str,
    image_dir: str,
    model: str,
    context: str,
    output_dir: str,
    experiment: str,
    split_name: str,
    num_samples: int,
    m: int,
    n: int,
    temperature: float = 0.0,
    max_tokens: int = 3000,
    num_ctx: int | None = None,
    think: bool = False,
) -> str:
    if context not in ("dvrl", "drl"):
        raise ValueError(f"endtoend context must be 'dvrl' or 'drl', got '{context}'")

    loader = importlib.import_module(f".datasets.{dataset}", package=__package__)
    git = prov.git_commit()

    # prompt provenance
    if context == "dvrl":
        prompt_sha = P.sha256_text(P.dvrl_prompt(m, n))
    else:
        prompt_sha = P.sha256_text(P.drl_rule_prompt(m, n) + P.drl_apply_prompt(m, n, ""))

    out_dir = _build_output_dir(Path(output_dir) / experiment, model, context, temperature)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.xlsx"

    if results_file.exists():
        rows = pd.read_excel(results_file).to_dict("records")
        done = {str(r.get("test_id")) for r in rows}
    else:
        rows, done = [], set()

    test_ids = loader.list_test_ids(json_path)[:num_samples]
    for test_id in tqdm(test_ids, desc=f"{context} {experiment}/{model}"):
        if str(test_id) in done:
            continue
        image_paths, concept, caption, uid, test_cat, commonsense = loader.read_sample(
            json_path, test_id, m, n)
        abs_paths = [_abs(image_dir, p) for p in image_paths]

        try:
            if context == "dvrl":
                text, _t, usage, dur = _unpack(llm_inference_multiimage(
                    model_name=model, system_prompt="", user_prompt=P.dvrl_prompt(m, n),
                    image_paths=abs_paths, temp=temperature, max_tokens=max_tokens,
                    num_ctx=num_ctx, think=think))
            else:  # drl: derive rule from support, then apply to query
                support, query = abs_paths[:m + n], abs_paths[-1:]
                rule_text, _t0, _u0, _d0 = _unpack(llm_inference_multiimage(
                    model_name=model, system_prompt="", user_prompt=P.drl_rule_prompt(m, n),
                    image_paths=support, temp=temperature, max_tokens=max_tokens,
                    num_ctx=num_ctx, think=think))
                text, _t, usage, dur = _unpack(llm_inference_multiimage(
                    model_name=model, system_prompt="",
                    user_prompt=P.drl_apply_prompt(m, n, str(rule_text)),
                    image_paths=query, temp=temperature, max_tokens=max_tokens,
                    num_ctx=num_ctx, think=think))
        except Exception as e:
            text, usage, dur = f"__ERROR__: {e}", None, None

        decision = parse_json_dict(text, model, required_keys=REQUIRED_KEYS)
        pred = decision.get("Conclusion")
        rows.append({
            "test_id": test_id, "uid": uid, "concept": concept, "caption": caption,
            "commonsense": commonsense, "m": m, "n": n,
            "test_cat_label": _norm(test_cat), "test_category_identified": _norm(pred),
            "is_correct": (_norm(test_cat) == _norm(pred)) if _norm(pred) else None,
            "conclusion_raw": pred, "rule_identified": decision.get("Rule"),
            "test_analysis": decision.get("Analysis"), "complete_output": str(text)[:32000],
            "reasoner_model": model, "perception_model": model,   # same model, end-to-end
            "context": context, "schema_version": "end2end",
            "description_manifest_sha256": "end2end", "reasoner_system_sha256": prompt_sha,
            "temperature": temperature, "max_tokens": max_tokens, "num_ctx": num_ctx,
            "pri_git_commit": git,
            "tokens_used": (usage if isinstance(usage, int) else None),
            "thought_tokens": None,
            "duration_sec": (round(dur / 1e9, 3) if isinstance(dur, (int, float)) else None),
        })
        pd.DataFrame(rows, columns=RESULT_COLUMNS).to_excel(results_file, index=False)

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    acc = df["is_correct"].dropna().mean() if len(df) else float("nan")
    print(f"{context} done: {len(df)} rows, accuracy={acc:.4f}\n  results: {results_file}")
    return str(results_file)
