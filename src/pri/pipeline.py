"""Stage 2 — reasoning: read frozen descriptions, decide, record provenance.

Implements the reasoning half of docs/REPRODUCIBILITY_DESIGN.md. Reads the
immutable description artifact produced by pri.describe **read-only**, runs the
text reasoner, and appends one fully-provenanced row per sample to
``${OUTPUT_DIR}/<experiment>/<reasoner>/<context>_temp_<temp>/results.xlsx``.

Never generates descriptions: if the artifact is absent the run aborts, so a
reasoner swap can never silently diverge from the descriptions used by another
reasoner. Each row records ``description_manifest_sha256`` — identical across a
reasoner swap iff the swap was genuinely controlled.

Config (module: "pri.pipeline", function: "run"):
    dataset, json_path, image_dir      the split (image_dir only for ground truth)
    reasoner_model                     text-only LM
    context                            'ca' (C2) etc.; picks the schema_version
    perception_model                   whose frozen descriptions to consume
    output_dir                         ${OUTPUT_DIR} root (holds descriptions/ and experiments)
    experiment                         results go to output_dir/<experiment>
    split_name                         must match the describe run
    num_samples, m, n, temperature, max_tokens, num_ctx, think
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from . import prompts as P
from . import provenance as prov
from .io_utils import set_seed
from .llm import REQUIRED_KEYS, llm_inference, parse_json_dict

set_seed(42)

RESULT_COLUMNS = [
    "test_id", "uid", "concept", "caption", "commonsense", "m", "n",
    "test_cat_label", "test_category_identified", "is_correct",
    "conclusion_raw", "rule_identified", "test_analysis", "complete_output",
    # provenance
    "reasoner_model", "perception_model", "context", "schema_version",
    "description_manifest_sha256", "reasoner_system_sha256",
    "temperature", "max_tokens", "num_ctx", "pri_git_commit",
    # cost / latency
    "tokens_used", "thought_tokens", "duration_sec",
]


def _norm(label) -> str | None:
    if label is None:
        return None
    s = str(label).lower().strip()
    if "cat_2" in s or ("pos" in s and "neg" not in s) or s in ("1", "1.0", "positive", "true", "yes"):
        return "pos"
    if "cat_1" in s or ("neg" in s and "pos" not in s) or s in ("0", "0.0", "negative", "false", "no"):
        return "neg"
    return None


def _build_output_dir(base: Path, model: str, context: str, temp: float) -> Path:
    return base / model / f"{context}_temp_{temp}"


def _unpack(result):
    """Normalize llm_inference return to (text, thinking, usage, duration_ns)."""
    if isinstance(result, tuple):
        pad = list(result) + [None] * (4 - len(result))
        return pad[0], pad[1], pad[2], pad[3]
    return result, None, None, None


def _load_descriptions(desc_root: Path, test_id: str, count: int) -> list[str]:
    """Read the `count` frozen descriptions for a sample, in image order."""
    sample_dir = desc_root / str(test_id)
    specs = []
    for idx in range(count):
        f = sample_dir / f"img_{idx:02d}.json"
        if not f.exists():
            raise FileNotFoundError(f"missing frozen description {f}")
        rec = json.loads(f.read_text())
        specs.append(rec.get("description", rec.get("error", "")))
    return specs


def run(
    dataset: str,
    json_path: str,
    reasoner_model: str,
    context: str,
    perception_model: str,
    output_dir: str,
    experiment: str,
    split_name: str,
    num_samples: int,
    m: int,
    n: int,
    image_dir: str = "",
    temperature: float = 0.0,
    max_tokens: int = 3000,
    num_ctx: int | None = None,
    think: bool = False,
) -> str:
    if context not in P.PERCEPTION_PROMPTS:
        raise ValueError(f"context '{context}' unknown; known: {list(P.PERCEPTION_PROMPTS)}")
    schema_version = P.PERCEPTION_PROMPTS[context]["schema_version"]

    # Locate the frozen description artifact; abort if absent (never generate).
    desc_root = Path(output_dir) / "descriptions" / perception_model / schema_version / split_name
    manifest = prov.load_frozen_manifest(desc_root, perception_model, context, split_name)
    desc_sha = manifest.get("content_sha256", "unknown")

    loader = importlib.import_module(f".datasets.{dataset}", package=__package__)
    system_prompt = P.system_eval()
    system_sha = P.sha256_text(system_prompt)
    git = prov.git_commit()

    out_dir = _build_output_dir(Path(output_dir) / experiment, reasoner_model, context, temperature)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.xlsx"

    if results_file.exists():
        rows = pd.read_excel(results_file).to_dict("records")
        done = {str(r.get("test_id")) for r in rows}
    else:
        rows, done = [], set()

    # Every raw response is kept alongside the results, so a run can be audited
    # without re-querying: what the model actually returned is otherwise only
    # visible as a truncated cell in the spreadsheet.
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    failed = 0
    missing_desc = 0
    test_ids = loader.list_test_ids(json_path)[:num_samples]
    for test_id in tqdm(test_ids, desc=f"reason {experiment}/{reasoner_model}"):
        if str(test_id) in done:
            continue
        image_paths, concept, caption, uid, test_cat, commonsense = loader.read_sample(
            json_path, test_id, m, n)
        try:
            specs = _load_descriptions(desc_root, test_id, len(image_paths))
        except FileNotFoundError:
            # The imported C2 artifact covers 499/500 problems (one legacy
            # source file never existed). A problem without frozen descriptions
            # cannot be evaluated in this condition; skip it and say so, rather
            # than abort a 500-problem run at whatever position it sits.
            missing_desc += 1
            continue
        user_prompt = P.user_eval(specs, m, n)

        err = None
        try:
            text, _think, usage, dur_ns = _unpack(llm_inference(
                model_name=reasoner_model, user_prompt=user_prompt,
                system_prompt=system_prompt, temp=temperature,
                max_tokens=max_tokens, num_ctx=num_ctx, think=think))
        except Exception as e:
            # ParseFailure carries the last full model output; without it the
            # raw file would record only the error message and lose what the
            # model actually said.
            err = str(e)
            text = getattr(e, "raw_output", "") or ""
            usage, dur_ns = None, None

        decision = parse_json_dict(text, reasoner_model, required_keys=REQUIRED_KEYS)
        pred_raw = decision.get("Conclusion")

        (raw_dir / f"{test_id}.json").write_text(json.dumps({
            "test_id": test_id,
            "reasoner_model": reasoner_model,
            "raw_output": text,
            "error": err,
            "parsed": decision or None,
            "generated_utc": prov.utc_now_iso(),
        }, indent=2, ensure_ascii=False))

        # A row with no prediction is a failed attempt, not a result. Recording
        # it would put the test_id in `done` and make the failure permanent
        # across reruns -- the same trap the describe stage had. Leaving it out
        # means rerunning the identical command retries exactly these.
        if pred_raw is None:
            failed += 1
            continue
        row = {
            "test_id": test_id, "uid": uid, "concept": concept, "caption": caption,
            "commonsense": commonsense, "m": m, "n": n,
            "test_cat_label": _norm(test_cat),
            "test_category_identified": _norm(pred_raw),
            "is_correct": (_norm(test_cat) == _norm(pred_raw)) if _norm(pred_raw) else None,
            "conclusion_raw": pred_raw,
            "rule_identified": decision.get("Rule"),
            "test_analysis": decision.get("Analysis"),
            "complete_output": str(text)[:32000],  # real model text; errors live in raw/
            "reasoner_model": reasoner_model, "perception_model": perception_model,
            "context": context, "schema_version": schema_version,
            "description_manifest_sha256": desc_sha, "reasoner_system_sha256": system_sha,
            "temperature": temperature, "max_tokens": max_tokens, "num_ctx": num_ctx,
            "pri_git_commit": git,
            "tokens_used": (usage if isinstance(usage, int) else None),
            "thought_tokens": None,
            "duration_sec": (round(dur_ns / 1e9, 3) if isinstance(dur_ns, (int, float)) else None),
        }
        rows.append(row)
        pd.DataFrame(rows, columns=RESULT_COLUMNS).to_excel(
            results_file, index=False)          # incremental, resume-safe

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    acc = df["is_correct"].dropna().mean() if len(df) else float("nan")
    print(f"reason done: {len(df)} rows, {failed} unparseable (not saved; "
          f"rerun to retry), {missing_desc} skipped (no frozen description), "
          f"accuracy={acc:.4f}\n"
          f"  results: {results_file}\n  descriptions_sha: {desc_sha[:16]}…")
    return str(results_file)


# ---------------------------------------------------------------------------
# C4 — ca_fixed_reinspect: CA reasoning, then ONE preregistered fixed
# re-inspection of the query image, then a final decision. The budget-matched,
# non-adaptive control for ICA (C5). Reuses the frozen descriptions read-only;
# the re-inspection Q&A is recorded inline. Requires image access at reason time.
# ---------------------------------------------------------------------------

def run_fixed_reinspect(
    dataset: str,
    json_path: str,
    image_dir: str,
    reasoner_model: str,
    perception_model: str,
    output_dir: str,
    experiment: str,
    split_name: str,
    num_samples: int,
    m: int,
    n: int,
    reinspection_model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 3000,
    num_ctx: int | None = None,
    think: bool = False,
) -> str:
    import os
    from .llm import llm_inference_multimodal
    from . import prompts as _P

    schema_version = P.PERCEPTION_PROMPTS["ca"]["schema_version"]   # reads the task-blind CA descriptions
    reinspect = reinspection_model or perception_model
    desc_root = Path(output_dir) / "descriptions" / perception_model / schema_version / split_name
    desc_sha = prov.load_frozen_manifest(
        desc_root, perception_model, "ca", split_name
    ).get("content_sha256", "unknown")

    loader = importlib.import_module(f".datasets.{dataset}", package=__package__)
    system_prompt = P.system_eval(); system_sha = P.sha256_text(system_prompt)
    git = prov.git_commit()
    out_dir = _build_output_dir(Path(output_dir) / experiment, reasoner_model, "ca_fixed_reinspect", temperature)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.xlsx"
    cols = RESULT_COLUMNS + ["reinspection_question", "reinspection_answer", "reinspection_model"]
    if results_file.exists():
        rows = pd.read_excel(results_file).to_dict("records")
        done = {str(r.get("test_id")) for r in rows}
    else:
        rows, done = [], set()

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    missing_desc = 0

    question = _P.fixed_reinspect_question()
    for test_id in tqdm(loader.list_test_ids(json_path)[:num_samples], desc=f"c4 {experiment}/{reasoner_model}"):
        if str(test_id) in done:
            continue
        image_paths, concept, caption, uid, test_cat, commonsense = loader.read_sample(json_path, test_id, m, n)
        try:
            specs = _load_descriptions(desc_root, test_id, len(image_paths))
        except FileNotFoundError:
            missing_desc += 1
            continue

        # Any of the three calls can exhaust the parse/repair ladder; that must
        # cost this problem only, and leave its trace in raw/, not abort the run.
        try:
            # 1. initial CA reasoning
            text, _t, usage, dur = _unpack(llm_inference(
                model_name=reasoner_model, user_prompt=P.user_eval(specs, m, n),
                system_prompt=system_prompt, temp=temperature, max_tokens=max_tokens,
                num_ctx=num_ctx, think=think))
            initial = parse_json_dict(text, reasoner_model, required_keys=REQUIRED_KEYS)
            tok = usage if isinstance(usage, int) else 0
            dr = dur if isinstance(dur, (int, float)) else 0

            # 2. fixed re-inspection of the QUERY image (last image)
            q_abs = image_paths[-1] if os.path.isabs(image_paths[-1]) else os.path.join(image_dir, image_paths[-1])
            ans, _t2, u2, d2 = _unpack(llm_inference_multimodal(
                model_name=reinspect, prompt=question, image_path=q_abs, temp=temperature,
                max_tokens=max_tokens, num_ctx=num_ctx, think=think))
            tok += u2 if isinstance(u2, int) else 0
            dr += d2 if isinstance(d2, (int, float)) else 0

            # 3. final decision incorporating the re-inspection
            fin, _t3, u3, d3 = _unpack(llm_inference(
                model_name=reasoner_model, user_prompt=_P.ca_reinspect_final_prompt(initial, str(ans)),
                system_prompt=system_prompt, temp=temperature, max_tokens=max_tokens,
                num_ctx=num_ctx, think=think))
            tok += u3 if isinstance(u3, int) else 0
            dr += d3 if isinstance(d3, (int, float)) else 0
            decision = (parse_json_dict(fin, reasoner_model, required_keys=REQUIRED_KEYS) or initial)
            pred = decision.get("Conclusion")

            err = None
        except Exception as e:
            err = str(e)
            text = locals().get("text", "")
            ans = locals().get("ans", "")
            fin = getattr(e, "raw_output", "") or locals().get("fin", "")
            decision, pred = {}, None

        # All three calls of the re-inspection round are kept: an unusable final
        # decision is often explained by the question or the answer rather than
        # the reasoning, and that is only visible with the intermediate texts.
        (raw_dir / f"{test_id}.json").write_text(json.dumps({
            "test_id": test_id, "reasoner_model": reasoner_model,
            "initial_raw": text, "reinspect_question": question,
            "reinspect_answer": str(ans), "final_raw": fin,
            "error": err,
            "parsed": decision or None,
            "generated_utc": prov.utc_now_iso(),
        }, indent=2, ensure_ascii=False))

        if pred is None:
            failed += 1
            continue

        rows.append({
            "test_id": test_id, "uid": uid, "concept": concept, "caption": caption,
            "commonsense": commonsense, "m": m, "n": n,
            "test_cat_label": _norm(test_cat), "test_category_identified": _norm(pred),
            "is_correct": (_norm(test_cat) == _norm(pred)) if _norm(pred) else None,
            "conclusion_raw": pred, "rule_identified": decision.get("Rule"),
            "test_analysis": decision.get("Analysis"), "complete_output": str(fin)[:32000],
            "reasoner_model": reasoner_model, "perception_model": perception_model,
            "context": "ca_fixed_reinspect", "schema_version": schema_version,
            "description_manifest_sha256": desc_sha, "reasoner_system_sha256": system_sha,
            "temperature": temperature, "max_tokens": max_tokens, "num_ctx": num_ctx,
            "pri_git_commit": git, "tokens_used": tok or None, "thought_tokens": None,
            "duration_sec": round(dr / 1e9, 3) if dr else None,
            "reinspection_question": question, "reinspection_answer": str(ans)[:8000],
            "reinspection_model": reinspect,
        })
        pd.DataFrame(rows, columns=cols).to_excel(results_file, index=False)

    df = pd.DataFrame(rows, columns=cols)
    acc = df["is_correct"].dropna().mean() if len(df) else float("nan")
    print(f"c4 fixed-reinspect done: {len(df)} rows, accuracy={acc:.4f}\n  results: {results_file}")
    return str(results_file)


# ---------------------------------------------------------------------------
# Textual DRL — the matched control for the CA - DRL contrast.
#
# CA - DRL changes three things at once: the query goes from pixels to text,
# the support evidence goes from pixels to text, and the staging changes (DRL
# compresses the supports into a short rule; CA holds all descriptions jointly).
# This condition holds the FIRST TWO fixed at their CA values -- it reads the
# same frozen descriptions CA reads -- and varies only the third. So
#
#     textual-DRL  -  CA      isolates STAGING (compress-to-rule vs joint context)
#
# over identical evidence, identical reasoner, and identical decoding. If it is
# null, the CA - DRL deficit cannot be a staging artifact and the attribution to
# description content stands on a manipulation rather than on anatomy alone.
#
# Config (module: "pri.pipeline", function: "run_textual_drl"): same keys as
# `run`. Emits two extra columns (rule_summary, stage1_output) and writes the
# stage-1 text to raw/ alongside the decision.
# ---------------------------------------------------------------------------

TEXTUAL_DRL_COLUMNS = RESULT_COLUMNS + ["rule_summary", "stage1_output"]


def run_textual_drl(
    dataset: str,
    json_path: str,
    reasoner_model: str,
    context: str,
    perception_model: str,
    output_dir: str,
    experiment: str,
    split_name: str,
    num_samples: int,
    m: int,
    n: int,
    image_dir: str = "",
    temperature: float = 0.0,
    max_tokens: int = 3000,
    num_ctx: int | None = None,
    think: bool = False,
) -> str:
    if context not in P.PERCEPTION_PROMPTS:
        raise ValueError(f"context '{context}' unknown; known: {list(P.PERCEPTION_PROMPTS)}")
    schema_version = P.PERCEPTION_PROMPTS[context]["schema_version"]

    # Same frozen artifact CA consumes: identical desc_sha is what makes this a
    # controlled comparison rather than two runs that merely resemble each other.
    desc_root = Path(output_dir) / "descriptions" / perception_model / schema_version / split_name
    manifest = prov.load_frozen_manifest(desc_root, perception_model, context, split_name)
    desc_sha = manifest.get("content_sha256", "unknown")

    loader = importlib.import_module(f".datasets.{dataset}", package=__package__)
    # No system prompt: DRL has none (see endtoend.run), and the staging
    # contrast must not smuggle in a reasoner framing CA's comparator lacked.
    prompt_sha = P.sha256_text(P.drl_text_rule_prompt([""], m, n)
                               + P.drl_text_apply_prompt("", m, n, ""))
    git = prov.git_commit()

    out_dir = _build_output_dir(Path(output_dir) / experiment, reasoner_model,
                                f"{context}_textdrl", temperature)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.xlsx"

    if results_file.exists():
        rows = pd.read_excel(results_file).to_dict("records")
        done = {str(r.get("test_id")) for r in rows}
    else:
        rows, done = [], set()

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    failed = 0
    missing_desc = 0
    test_ids = loader.list_test_ids(json_path)[:num_samples]
    for test_id in tqdm(test_ids, desc=f"textdrl {experiment}/{reasoner_model}"):
        if str(test_id) in done:
            continue
        image_paths, concept, caption, uid, test_cat, commonsense = loader.read_sample(
            json_path, test_id, m, n)
        try:
            specs = _load_descriptions(desc_root, test_id, len(image_paths))
        except FileNotFoundError:
            missing_desc += 1
            continue

        err = None
        rule_text = ""
        try:
            # stage 1: rule from the support descriptions, query withheld
            rule_text, _t0, _u0, _d0 = _unpack(llm_inference(
                model_name=reasoner_model,
                user_prompt=P.drl_text_rule_prompt(specs, m, n),
                system_prompt="", temp=temperature, max_tokens=max_tokens,
                num_ctx=num_ctx, think=think))
            # stage 2: apply that rule to the query description alone
            text, _t, usage, dur_ns = _unpack(llm_inference(
                model_name=reasoner_model,
                user_prompt=P.drl_text_apply_prompt(specs[-1], m, n, str(rule_text)),
                system_prompt="", temp=temperature, max_tokens=max_tokens,
                num_ctx=num_ctx, think=think))
        except Exception as e:
            err = str(e)
            text = getattr(e, "raw_output", "") or ""
            usage, dur_ns = None, None

        decision = parse_json_dict(text, reasoner_model, required_keys=REQUIRED_KEYS)
        pred_raw = decision.get("Conclusion")

        (raw_dir / f"{test_id}.json").write_text(json.dumps({
            "test_id": test_id,
            "reasoner_model": reasoner_model,
            "stage1_rule_output": str(rule_text),
            "raw_output": text,
            "error": err,
            "parsed": decision or None,
            "generated_utc": prov.utc_now_iso(),
        }, indent=2, ensure_ascii=False))

        if pred_raw is None:
            failed += 1
            continue
        rows.append({
            "test_id": test_id, "uid": uid, "concept": concept, "caption": caption,
            "commonsense": commonsense, "m": m, "n": n,
            "test_cat_label": _norm(test_cat),
            "test_category_identified": _norm(pred_raw),
            "is_correct": (_norm(test_cat) == _norm(pred_raw)) if _norm(pred_raw) else None,
            "conclusion_raw": pred_raw,
            "rule_identified": decision.get("Rule"),
            "test_analysis": decision.get("Analysis"),
            "complete_output": str(text)[:32000],
            "reasoner_model": reasoner_model, "perception_model": perception_model,
            "context": f"{context}_textdrl", "schema_version": schema_version,
            "description_manifest_sha256": desc_sha, "reasoner_system_sha256": prompt_sha,
            "temperature": temperature, "max_tokens": max_tokens, "num_ctx": num_ctx,
            "pri_git_commit": git,
            "tokens_used": (usage if isinstance(usage, int) else None),
            "thought_tokens": None,
            "duration_sec": (round(dur_ns / 1e9, 3) if isinstance(dur_ns, (int, float)) else None),
            "rule_summary": str(rule_text)[:4000],
            "stage1_output": str(rule_text)[:32000],
        })
        pd.DataFrame(rows, columns=TEXTUAL_DRL_COLUMNS).to_excel(results_file, index=False)

    df = pd.DataFrame(rows, columns=TEXTUAL_DRL_COLUMNS)
    acc = df["is_correct"].dropna().mean() if len(df) else float("nan")
    print(f"textual-DRL done: {len(df)} rows, {failed} unparseable (not saved; "
          f"rerun to retry), {missing_desc} skipped (no frozen description), "
          f"accuracy={acc:.4f}\n"
          f"  results: {results_file}\n  descriptions_sha: {desc_sha[:16]}…")
    return str(results_file)
