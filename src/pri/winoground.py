"""Winoground reasoning: Text / Image / Group scoring over frozen descriptions.

Consumes the immutable description artifact produced by ``pri.describe`` with
``context: winoground`` (activity schema) — img_00 = image_0 description,
img_01 = image_1 description. Captions come from the dataset loader (they are
task text, never seen by perception). Aborts if the artifact is absent.

Per entry, four text-only reasoner calls:
  Text score  : each image description picks a caption; correct when img_0→cat_0
                and img_1→cat_1 (both → 1).
  Image score : each caption picks a description; correct when cap_0→cat_0 and
                cap_1→cat_1 (both → 1).
  Group score : Text ∧ Image.

Writes ``${output_dir}/<experiment>/<reasoner>/winoground_temp_<temp>/results.xlsx``
with per-entry scores and full provenance (incl. description_manifest_sha256).

Config (module: "pri.winoground", function: "run"): dataset ('winoground'),
json_path, reasoner_model, perception_model, output_dir, experiment, split_name,
num_samples, temperature, max_tokens, num_ctx, think.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from . import prompts as P
from . import provenance as prov
from .io_utils import set_seed
from .llm import (llm_inference, llm_inference_multiimage,
                  llm_inference_multimodal, parse_json_dict)
from .pipeline import _build_output_dir, _unpack

set_seed(42)

WINO_SCHEMA_VERSION = "activity_v1"   # must match pri.describe context 'winoground'

WINO_COLUMNS = [
    "id", "caption_0", "caption_1", "image_0", "image_1", "tag",
    "text_score", "image_score", "group_score",
    "text_cat_0", "text_cat_1", "image_cat_0", "image_cat_1",
    "reasoner_model", "perception_model", "schema_version",
    "description_manifest_sha256", "prompt_sha256",
    "temperature", "max_tokens", "num_ctx", "pri_git_commit",
    "tokens_used", "duration_sec",
]


def _choose(model, prompt, temp, max_tokens, num_ctx, think):
    """Run one scoring call; return (category or None, tokens, duration_ns)."""
    text, _t, usage, dur = _unpack(llm_inference(
        model_name=model, user_prompt=prompt, system_prompt="",
        temp=temp, max_tokens=max_tokens, num_ctx=num_ctx, think=think))
    parsed = parse_json_dict(text, model)
    return parsed.get("category"), (usage if isinstance(usage, int) else 0), \
        (dur if isinstance(dur, (int, float)) else 0)


def run(
    dataset: str,
    json_path: str,
    reasoner_model: str,
    perception_model: str,
    output_dir: str,
    experiment: str,
    split_name: str,
    num_samples: int,
    temperature: float = 0.0,
    max_tokens: int = 3000,
    num_ctx: int | None = None,
    think: bool = False,
    **_unused,
) -> str:
    desc_root = Path(output_dir) / "descriptions" / perception_model / WINO_SCHEMA_VERSION / split_name
    desc_sha = prov.load_frozen_manifest(
        desc_root, perception_model, "winoground", split_name
    ).get("content_sha256", "unknown")

    loader = importlib.import_module(f".datasets.{dataset}", package=__package__)
    prompt_sha = P.sha256_text(P.text_score_prompt("", "", "") + P.image_score_prompt("", "", ""))
    git = prov.git_commit()

    out_dir = _build_output_dir(Path(output_dir) / experiment, reasoner_model, "winoground", temperature)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.xlsx"

    if results_file.exists():
        rows = pd.read_excel(results_file).to_dict("records")
        done = {str(r.get("id")) for r in rows}
    else:
        rows, done = [], set()

    def load_desc(entry_id, idx):
        f = desc_root / str(entry_id) / f"img_{idx:02d}.json"
        if not f.exists():
            raise FileNotFoundError(f"missing frozen description {f}")
        rec = json.loads(f.read_text())
        return rec.get("description", rec.get("error", ""))

    for test_id in tqdm(loader.list_test_ids(json_path)[:num_samples], desc=f"winoground/{reasoner_model}"):
        if str(test_id) in done:
            continue
        _imgs, cap0, cap1, eid, tag = loader.read_sample(json_path, test_id)
        d0, d1 = load_desc(eid, 0), load_desc(eid, 1)
        tok, dur = 0, 0

        # Text score: each image description -> which caption
        t0, a, b = _choose(reasoner_model, P.text_score_prompt(json.dumps(d0), cap0, cap1),
                           temperature, max_tokens, num_ctx, think); tok += a; dur += b
        t1, a, b = _choose(reasoner_model, P.text_score_prompt(json.dumps(d1), cap0, cap1),
                           temperature, max_tokens, num_ctx, think); tok += a; dur += b
        text_score = 1 if (t0 == "cat_0" and t1 == "cat_1") else 0

        # Image score: each caption -> which image description
        i0, a, b = _choose(reasoner_model, P.image_score_prompt(cap0, json.dumps(d0), json.dumps(d1)),
                           temperature, max_tokens, num_ctx, think); tok += a; dur += b
        i1, a, b = _choose(reasoner_model, P.image_score_prompt(cap1, json.dumps(d0), json.dumps(d1)),
                           temperature, max_tokens, num_ctx, think); tok += a; dur += b
        image_score = 1 if (i0 == "cat_0" and i1 == "cat_1") else 0

        group_score = 1 if (text_score == 1 and image_score == 1) else 0

        rows.append({
            "id": eid, "caption_0": cap0, "caption_1": cap1,
            "image_0": _imgs[0], "image_1": _imgs[1], "tag": tag,
            "text_score": text_score, "image_score": image_score, "group_score": group_score,
            "text_cat_0": t0, "text_cat_1": t1, "image_cat_0": i0, "image_cat_1": i1,
            "reasoner_model": reasoner_model, "perception_model": perception_model,
            "schema_version": WINO_SCHEMA_VERSION, "description_manifest_sha256": desc_sha,
            "prompt_sha256": prompt_sha, "temperature": temperature, "max_tokens": max_tokens,
            "num_ctx": num_ctx, "pri_git_commit": git,
            "tokens_used": tok or None, "duration_sec": round(dur / 1e9, 3) if dur else None,
        })
        pd.DataFrame(rows, columns=WINO_COLUMNS).to_excel(results_file, index=False)

    df = pd.DataFrame(rows, columns=WINO_COLUMNS)
    if len(df):
        print(f"winoground done: {len(df)} entries — "
              f"Text {100*df.text_score.mean():.1f}  Image {100*df.image_score.mean():.1f}  "
              f"Group {100*df.group_score.mean():.1f}\n  results: {results_file}")
    return str(results_file)


# ---------------------------------------------------------------------------
# Winoground DVRL (direct): the multimodal model scores over the images
# themselves — no describe stage, no frozen artifact. Provenance is inline.
# ---------------------------------------------------------------------------

WINO_DIRECT_COLUMNS = [c for c in WINO_COLUMNS if c != "description_manifest_sha256"] + ["mode"]


def run_direct(
    dataset: str,
    json_path: str,
    image_dir: str,
    model: str,
    output_dir: str,
    experiment: str,
    split_name: str,
    num_samples: int,
    temperature: float = 0.0,
    max_tokens: int = 3000,
    num_ctx: int | None = None,
    think: bool = False,
    **_unused,
) -> str:
    """End-to-end Winoground: images -> Text/Image/Group scores in one stage."""
    loader = importlib.import_module(f".datasets.{dataset}", package=__package__)
    prompt_sha = P.sha256_text(P.mm_text_score_prompt("", "") + P.mm_image_score_prompt(""))
    git = prov.git_commit()

    out_dir = _build_output_dir(Path(output_dir) / experiment, model, "winoground_dvrl", temperature)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.xlsx"
    if results_file.exists():
        rows = pd.read_excel(results_file).to_dict("records")
        done = {str(r.get("id")) for r in rows}
    else:
        rows, done = [], set()

    def _abs(rel):
        return rel if os.path.isabs(rel) else os.path.join(image_dir, rel)

    def _cat_1image(prompt, img):
        text, _t, u, d = _unpack(llm_inference_multimodal(
            model_name=model, prompt=prompt, image_path=img, temp=temperature,
            max_tokens=max_tokens, num_ctx=num_ctx, think=think))
        return (parse_json_dict(text, model)).get("category"), \
            (u if isinstance(u, int) else 0), (d if isinstance(d, (int, float)) else 0)

    def _cat_2image(prompt, imgs):
        text, _t, u, d = _unpack(llm_inference_multiimage(
            model_name=model, system_prompt="", user_prompt=prompt, image_paths=imgs,
            temp=temperature, max_tokens=max_tokens, num_ctx=num_ctx, think=think))
        return (parse_json_dict(text, model)).get("category"), \
            (u if isinstance(u, int) else 0), (d if isinstance(d, (int, float)) else 0)

    for test_id in tqdm(loader.list_test_ids(json_path)[:num_samples], desc=f"winoground-direct/{model}"):
        if str(test_id) in done:
            continue
        imgs, cap0, cap1, eid, tag = loader.read_sample(json_path, test_id)
        i0, i1 = _abs(imgs[0]), _abs(imgs[1])
        tok, dur = 0, 0
        t0, a, b = _cat_1image(P.mm_text_score_prompt(cap0, cap1), i0); tok += a; dur += b
        t1, a, b = _cat_1image(P.mm_text_score_prompt(cap0, cap1), i1); tok += a; dur += b
        text_score = 1 if (t0 == "cat_0" and t1 == "cat_1") else 0
        im0, a, b = _cat_2image(P.mm_image_score_prompt(cap0), [i0, i1]); tok += a; dur += b
        im1, a, b = _cat_2image(P.mm_image_score_prompt(cap1), [i0, i1]); tok += a; dur += b
        image_score = 1 if (im0 == "cat_0" and im1 == "cat_1") else 0
        group_score = 1 if (text_score and image_score) else 0
        rows.append({
            "id": eid, "caption_0": cap0, "caption_1": cap1, "image_0": imgs[0], "image_1": imgs[1],
            "tag": tag, "text_score": text_score, "image_score": image_score, "group_score": group_score,
            "text_cat_0": t0, "text_cat_1": t1, "image_cat_0": im0, "image_cat_1": im1,
            "reasoner_model": model, "perception_model": model, "schema_version": "direct",
            "prompt_sha256": prompt_sha, "temperature": temperature, "max_tokens": max_tokens,
            "num_ctx": num_ctx, "pri_git_commit": git, "tokens_used": tok or None,
            "duration_sec": round(dur / 1e9, 3) if dur else None, "mode": "winoground_dvrl",
        })
        pd.DataFrame(rows, columns=WINO_DIRECT_COLUMNS).to_excel(results_file, index=False)

    df = pd.DataFrame(rows, columns=WINO_DIRECT_COLUMNS)
    if len(df):
        print(f"winoground-direct done: {len(df)} entries — "
              f"Text {100*df.text_score.mean():.1f}  Image {100*df.image_score.mean():.1f}  "
              f"Group {100*df.group_score.mean():.1f}")
    return str(results_file)


# ---------------------------------------------------------------------------
# Winoground ICA (interactive): CA + one reasoner-generated question per
# direction, answered by RE-INSPECTING the real image, then a final decision.
# Base frozen descriptions are untouched; the re-inspection Q&A is recorded in
# the result (docs/REPRODUCIBILITY_DESIGN.md). Needs BOTH the frozen artifact
# (initial reasoning) and image access (re-inspection by the perception model).
# ---------------------------------------------------------------------------

from . import ica_prompts as ICA          # noqa: E402

WINO_ICA_COLUMNS = WINO_COLUMNS + ["text_q0", "text_q1", "image_q0", "image_q1", "reinspection_model"]

# C4 records the fixed question it used, so the C5-C4 contrast is auditable:
# C5 logs the questions it chose, C4 logs the one it was given.
WINO_C4_COLUMNS = WINO_COLUMNS + ["reinspection_model", "reinspect_question"]


def _final_category(reasoner, first_prompt, desc_for_answer, image_path, reinspect_model,
                    final_prompt_fn, temp, max_tokens, num_ctx, think):
    """One ICA direction: reason -> (maybe) question -> re-inspect image -> final.

    Returns (category, question_or_None, tokens, duration_ns).
    """
    text, _t, u, d = _unpack(llm_inference(model_name=reasoner, user_prompt=first_prompt,
                             system_prompt="", temp=temp, max_tokens=max_tokens,
                             num_ctx=num_ctx, think=think))
    tok = u if isinstance(u, int) else 0
    dur = d if isinstance(d, (int, float)) else 0
    parsed = parse_json_dict(text, reasoner)
    question = parsed.get("image_question")
    if not question or str(question).lower() in ("none", "null", ""):
        return parsed.get("category"), None, tok, dur

    # re-inspect the REAL image (perception model) to answer the question
    ans_text, _t2, u2, d2 = _unpack(llm_inference_multimodal(
        model_name=reinspect_model, prompt=ICA.get_visual_answer_ica_prompt(question, desc_for_answer),
        image_path=image_path, temp=temp, max_tokens=max_tokens, num_ctx=num_ctx, think=think))
    tok += u2 if isinstance(u2, int) else 0
    dur += d2 if isinstance(d2, (int, float)) else 0
    qa = (parse_json_dict(ans_text, reinspect_model) or {"answer": ans_text})

    # final decision with the answer
    fin_text, _t3, u3, d3 = _unpack(llm_inference(
        model_name=reasoner, user_prompt=final_prompt_fn(parsed, qa),
        system_prompt="", temp=temp, max_tokens=max_tokens, num_ctx=num_ctx, think=think))
    tok += u3 if isinstance(u3, int) else 0
    dur += d3 if isinstance(d3, (int, float)) else 0
    final = parse_json_dict(fin_text, reasoner)
    return final.get("category", parsed.get("category")), str(question), tok, dur


def _fixed_final_category(reasoner, first_prompt, desc_for_answer, image_path, reinspect_model,
                          temp, max_tokens, num_ctx, think):
    """One C4 direction: reason(desc) -> FIXED re-inspection of the image -> final.

    Deliberately mirrors :func:`_final_category` (C5) call-for-call, with two
    differences that are the whole point of the control:
      * the question is ``P.fixed_reinspect_question()`` — preregistered and
        benchmark-generic — not generated by the reasoner;
      * the re-inspection ALWAYS happens (C5 may decline to ask), so C4 never
        gets fewer looks than C5. If C5 still wins, adaptivity earned it at
        equal-or-lower cost.
    The initial prompt is the plain C2 scoring prompt, so C4-C2 isolates the
    extra look, exactly as C4-C3 does on Bongard-OW.

    Returns (category, tokens, duration_ns).
    """
    text, _t, u, d = _unpack(llm_inference(model_name=reasoner, user_prompt=first_prompt,
                             system_prompt="", temp=temp, max_tokens=max_tokens,
                             num_ctx=num_ctx, think=think))
    tok = u if isinstance(u, int) else 0
    dur = d if isinstance(d, (int, float)) else 0
    initial = parse_json_dict(text, reasoner)

    # re-inspect the REAL image with the FIXED question (same answer prompt as C5)
    question = P.fixed_reinspect_question()
    ans_text, _t2, u2, d2 = _unpack(llm_inference_multimodal(
        model_name=reinspect_model, prompt=ICA.get_visual_answer_ica_prompt(question, desc_for_answer),
        image_path=image_path, temp=temp, max_tokens=max_tokens, num_ctx=num_ctx, think=think))
    tok += u2 if isinstance(u2, int) else 0
    dur += d2 if isinstance(d2, (int, float)) else 0
    qa = (parse_json_dict(ans_text, reinspect_model) or {"answer": ans_text})

    fin_text, _t3, u3, d3 = _unpack(llm_inference(
        model_name=reasoner, user_prompt=P.wino_reinspect_final_prompt(initial, qa),
        system_prompt="", temp=temp, max_tokens=max_tokens, num_ctx=num_ctx, think=think))
    tok += u3 if isinstance(u3, int) else 0
    dur += d3 if isinstance(d3, (int, float)) else 0
    final = parse_json_dict(fin_text, reasoner)
    return final.get("category", initial.get("category")), tok, dur


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
    reinspection_model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 3000,
    num_ctx: int | None = None,
    think: bool = False,
    **_unused,
) -> str:
    """Winoground C4: CA + ONE fixed re-inspection per direction, then a final call.

    The budget-matched control that licenses calling ICA (C5) *adaptive*: same
    one-extra-visual-call structure, but the question is preregistered. Without
    this run, plan §E2/E3 requires ICA to be reported as an additional-inspection
    effect, not adaptivity. Consumes the same frozen `winoground` descriptions as
    C2/C5; reinspection_model defaults to perception_model.
    """
    reinspect = reinspection_model or perception_model
    desc_root = Path(output_dir) / "descriptions" / perception_model / WINO_SCHEMA_VERSION / split_name
    desc_sha = prov.load_frozen_manifest(
        desc_root, perception_model, "winoground", split_name
    ).get("content_sha256", "unknown")

    loader = importlib.import_module(f".datasets.{dataset}", package=__package__)
    prompt_sha = P.sha256_text(P.text_score_prompt("", "", "") + P.image_score_prompt("", "", "")
                               + P.fixed_reinspect_question())
    git = prov.git_commit()

    out_dir = _build_output_dir(Path(output_dir) / experiment, reasoner_model,
                                "winoground_fixed_reinspect", temperature)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.xlsx"
    if results_file.exists():
        rows = pd.read_excel(results_file).to_dict("records")
        done = {str(r.get("id")) for r in rows}
    else:
        rows, done = [], set()

    def load_desc(eid, idx):
        f = desc_root / str(eid) / f"img_{idx:02d}.json"
        if not f.exists():
            raise FileNotFoundError(f"missing frozen description {f}")
        rec = json.loads(f.read_text())
        return rec.get("description", rec.get("error", ""))

    def _abs(rel):
        return rel if os.path.isabs(rel) else os.path.join(image_dir, rel)

    for test_id in tqdm(loader.list_test_ids(json_path)[:num_samples],
                        desc=f"winoground-c4/{reasoner_model}"):
        if str(test_id) in done:
            continue
        imgs, cap0, cap1, eid, tag = loader.read_sample(json_path, test_id)
        d0, d1 = load_desc(eid, 0), load_desc(eid, 1)
        i0, i1 = _abs(imgs[0]), _abs(imgs[1])
        tok = dur = 0

        # Text score: per image, reason(desc) -> fixed re-inspect(image) -> final
        t0, a, b = _fixed_final_category(reasoner_model, P.text_score_prompt(json.dumps(d0), cap0, cap1),
            json.dumps(d0), i0, reinspect, temperature, max_tokens, num_ctx, think); tok += a; dur += b
        t1, a, b = _fixed_final_category(reasoner_model, P.text_score_prompt(json.dumps(d1), cap0, cap1),
            json.dumps(d1), i1, reinspect, temperature, max_tokens, num_ctx, think); tok += a; dur += b
        text_score = 1 if (t0 == "cat_0" and t1 == "cat_1") else 0

        # Image score: per caption, reason(both descs) -> fixed re-inspect -> final
        # (same image targets as C5: cap_0 -> image_0, cap_1 -> image_1)
        im0, a, b = _fixed_final_category(reasoner_model, P.image_score_prompt(cap0, json.dumps(d0), json.dumps(d1)),
            json.dumps(d0), i0, reinspect, temperature, max_tokens, num_ctx, think); tok += a; dur += b
        im1, a, b = _fixed_final_category(reasoner_model, P.image_score_prompt(cap1, json.dumps(d0), json.dumps(d1)),
            json.dumps(d1), i1, reinspect, temperature, max_tokens, num_ctx, think); tok += a; dur += b
        image_score = 1 if (im0 == "cat_0" and im1 == "cat_1") else 0
        group_score = 1 if (text_score and image_score) else 0

        rows.append({
            "id": eid, "caption_0": cap0, "caption_1": cap1, "image_0": imgs[0], "image_1": imgs[1], "tag": tag,
            "text_score": text_score, "image_score": image_score, "group_score": group_score,
            "text_cat_0": t0, "text_cat_1": t1, "image_cat_0": im0, "image_cat_1": im1,
            "reasoner_model": reasoner_model, "perception_model": perception_model,
            "schema_version": WINO_SCHEMA_VERSION, "description_manifest_sha256": desc_sha,
            "prompt_sha256": prompt_sha, "temperature": temperature, "max_tokens": max_tokens,
            "num_ctx": num_ctx, "pri_git_commit": git, "tokens_used": tok or None,
            "duration_sec": round(dur / 1e9, 3) if dur else None,
            "reinspection_model": reinspect, "reinspect_question": P.fixed_reinspect_question(),
        })
        pd.DataFrame(rows, columns=WINO_C4_COLUMNS).to_excel(results_file, index=False)

    df = pd.DataFrame(rows, columns=WINO_C4_COLUMNS)
    if len(df):
        print(f"winoground-c4 (fixed reinspect) done: {len(df)} entries — "
              f"Text {100*df.text_score.mean():.1f}  Image {100*df.image_score.mean():.1f}  "
              f"Group {100*df.group_score.mean():.1f}\n  results: {results_file}")
    return str(results_file)


def run_interactive(
    dataset: str,
    json_path: str,
    image_dir: str,
    reasoner_model: str,
    perception_model: str,
    output_dir: str,
    experiment: str,
    split_name: str,
    num_samples: int,
    reinspection_model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 3000,
    num_ctx: int | None = None,
    think: bool = False,
    **_unused,
) -> str:
    """Winoground ICA. reinspection_model defaults to perception_model."""
    reinspect = reinspection_model or perception_model
    desc_root = Path(output_dir) / "descriptions" / perception_model / WINO_SCHEMA_VERSION / split_name
    desc_sha = prov.load_frozen_manifest(
        desc_root, perception_model, "winoground", split_name
    ).get("content_sha256", "unknown")

    loader = importlib.import_module(f".datasets.{dataset}", package=__package__)
    prompt_sha = P.sha256_text(ICA.text_score_ica_prompt("", "", "") + ICA.image_score_ica_prompt("", "", ""))
    git = prov.git_commit()

    out_dir = _build_output_dir(Path(output_dir) / experiment, reasoner_model, "ica", temperature)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.xlsx"
    if results_file.exists():
        rows = pd.read_excel(results_file).to_dict("records")
        done = {str(r.get("id")) for r in rows}
    else:
        rows, done = [], set()

    def load_desc(eid, idx):
        f = desc_root / str(eid) / f"img_{idx:02d}.json"
        if not f.exists():
            raise FileNotFoundError(f"missing frozen description {f}")
        return json.loads(f.read_text()).get("description", "")

    def _abs(rel):
        return rel if os.path.isabs(rel) else os.path.join(image_dir, rel)

    for test_id in tqdm(loader.list_test_ids(json_path)[:num_samples], desc=f"winoground-ica/{reasoner_model}"):
        if str(test_id) in done:
            continue
        imgs, cap0, cap1, eid, tag = loader.read_sample(json_path, test_id)
        d0, d1 = load_desc(eid, 0), load_desc(eid, 1)
        i0, i1 = _abs(imgs[0]), _abs(imgs[1])
        tok = dur = 0

        # Text score: per image, reason(desc)->question->reinspect(image)->final
        t0, tq0, a, b = _final_category(reasoner_model, ICA.text_score_ica_prompt(json.dumps(d0), cap0, cap1),
            json.dumps(d0), i0, reinspect, ICA.text_score_final_ica_prompt, temperature, max_tokens, num_ctx, think)
        tok += a; dur += b
        t1, tq1, a, b = _final_category(reasoner_model, ICA.text_score_ica_prompt(json.dumps(d1), cap0, cap1),
            json.dumps(d1), i1, reinspect, ICA.text_score_final_ica_prompt, temperature, max_tokens, num_ctx, think)
        tok += a; dur += b
        text_score = 1 if (t0 == "cat_0" and t1 == "cat_1") else 0

        # Image score: per caption, reason(both descs)->question->reinspect->final
        im0, iq0, a, b = _final_category(reasoner_model, ICA.image_score_ica_prompt(cap0, json.dumps(d0), json.dumps(d1)),
            json.dumps(d0), i0, reinspect, ICA.image_score_final_ica_prompt, temperature, max_tokens, num_ctx, think)
        tok += a; dur += b
        im1, iq1, a, b = _final_category(reasoner_model, ICA.image_score_ica_prompt(cap1, json.dumps(d0), json.dumps(d1)),
            json.dumps(d1), i1, reinspect, ICA.image_score_final_ica_prompt, temperature, max_tokens, num_ctx, think)
        tok += a; dur += b
        image_score = 1 if (im0 == "cat_0" and im1 == "cat_1") else 0
        group_score = 1 if (text_score and image_score) else 0

        rows.append({
            "id": eid, "caption_0": cap0, "caption_1": cap1, "image_0": imgs[0], "image_1": imgs[1], "tag": tag,
            "text_score": text_score, "image_score": image_score, "group_score": group_score,
            "text_cat_0": t0, "text_cat_1": t1, "image_cat_0": im0, "image_cat_1": im1,
            "reasoner_model": reasoner_model, "perception_model": perception_model,
            "schema_version": WINO_SCHEMA_VERSION, "description_manifest_sha256": desc_sha,
            "prompt_sha256": prompt_sha, "temperature": temperature, "max_tokens": max_tokens,
            "num_ctx": num_ctx, "pri_git_commit": git, "tokens_used": tok or None,
            "duration_sec": round(dur / 1e9, 3) if dur else None,
            "text_q0": tq0, "text_q1": tq1, "image_q0": iq0, "image_q1": iq1,
            "reinspection_model": reinspect,
        })
        pd.DataFrame(rows, columns=WINO_ICA_COLUMNS).to_excel(results_file, index=False)

    df = pd.DataFrame(rows, columns=WINO_ICA_COLUMNS)
    if len(df):
        nq = df[["text_q0","text_q1","image_q0","image_q1"]].notna().sum().sum()
        print(f"winoground-ica done: {len(df)} entries — Text {100*df.text_score.mean():.1f}  "
              f"Image {100*df.image_score.mean():.1f}  Group {100*df.group_score.mean():.1f}  "
              f"({nq} re-inspection questions issued)")
    return str(results_file)
