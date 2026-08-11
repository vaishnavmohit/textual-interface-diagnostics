"""Stage 1 — perception: generate immutable, hashed per-image descriptions.

Implements the perception artifact of docs/REPRODUCIBILITY_DESIGN.md. A
description is generated **once per image** and never overwritten; a manifest
records everything needed to reproduce it and to prove, downstream, that a
reasoner swap used identical descriptions.

Config (module: "pri.describe", function: "run"):
    dataset            loader name under pri.datasets (e.g. bongard_ow)
    json_path          split JSON
    image_dir          root the loader's relative image paths resolve against
    perception_model   VLM that produces descriptions
    context            perception prompt key in prompts.PERCEPTION_PROMPTS (e.g. ca)
    output_dir         ${OUTPUT_DIR}; artifacts go under output_dir/descriptions/...
    split_name         label for this split (e.g. bongard_ow, winoground)
    num_samples, m, n  sampling + support sizes
    temperature, max_tokens, num_ctx, think, force

Task-blindness is structural: the perception model receives only the image and
the generic perception prompt — never the task, labels, split, or rule.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

from tqdm import tqdm

from . import prompts as P
from . import provenance as prov
from .io_utils import set_seed
from .llm import llm_inference_multimodal

set_seed(42)


def _load_loader(dataset: str):
    return importlib.import_module(f".datasets.{dataset}", package=__package__)


def _descriptions_root(output_dir: str, perception_model: str,
                       schema_version: str, split_name: str) -> Path:
    # model name kept verbatim (colons included), matching the run-tree convention
    return Path(output_dir) / "descriptions" / perception_model / schema_version / split_name


def _is_failed(path: Path) -> bool:
    """True if this file records a failed call rather than a description.

    Unreadable or truncated files count as failed too: a rerun should repair a
    record it cannot parse, not preserve it.
    """
    try:
        return "description" not in json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return True


def run(
    dataset: str,
    json_path: str,
    image_dir: str,
    perception_model: str,
    context: str,
    output_dir: str,
    split_name: str,
    num_samples: int,
    m: int,
    n: int,
    temperature: float = 0.0,
    max_tokens: int = 3000,
    num_ctx: int | None = None,
    think: bool = False,
    force: bool = False,
) -> str:
    if context not in P.PERCEPTION_PROMPTS:
        raise ValueError(f"context '{context}' has no perception prompt; "
                         f"known: {list(P.PERCEPTION_PROMPTS)}")
    spec = P.PERCEPTION_PROMPTS[context]
    schema_version = spec["schema_version"]
    task_aware = spec.get("task_aware", False)
    json_output = spec.get("json_output", True)

    def prompt_for(idx: int) -> str:
        """The perception prompt for image `idx`. Task-aware conditions receive
        the image's Bongard role; task-blind conditions ignore idx."""
        if task_aware:
            return spec["fn"](P.role_for_index(idx, m, n))
        return spec["fn"]()

    # Prompt-identity hash for the manifest. Task-aware prompts vary by role, so
    # hash the full role set deterministically; task-blind hashes the one prompt.
    if task_aware:
        prompt_sha = P.sha256_text("".join(spec["fn"](r) for r in ("positive", "negative", "query")))
    else:
        prompt_sha = P.sha256_text(spec["fn"]())

    loader = _load_loader(dataset)
    root = _descriptions_root(output_dir, perception_model, schema_version, split_name)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"

    # Immutability guard: an existing manifest with a different prompt hash means
    # the prompt changed without a schema_version bump -> abort (silent drift).
    if manifest_path.exists() and not force:
        old = json.loads(manifest_path.read_text())
        if old.get("prompt_sha256") != prompt_sha:
            raise RuntimeError(
                f"Prompt hash mismatch for existing artifact {root}\n"
                f"  existing: {old.get('prompt_sha256')}\n  current:  {prompt_sha}\n"
                f"Bump schema_version in prompts.PERCEPTION_PROMPTS instead of editing in place."
            )

    test_ids = loader.list_test_ids(json_path)[:num_samples]
    generated, skipped, failed = 0, 0, 0

    for test_id in tqdm(test_ids, desc=f"describe {split_name}/{perception_model}"):
        image_paths, *_ = loader.read_sample(json_path, test_id, m, n)
        sample_dir = root / str(test_id)
        sample_dir.mkdir(parents=True, exist_ok=True)
        for idx, rel in enumerate(image_paths):
            out_file = sample_dir / f"img_{idx:02d}.json"
            # Immutable: a real description is never overwritten. A record left by
            # a failed call is not a description, so a rerun retries it -- otherwise
            # one transient API error would be a permanent hole in the artifact.
            if out_file.exists() and not force and not _is_failed(out_file):
                skipped += 1
                continue
            abs_path = rel if os.path.isabs(rel) else os.path.join(image_dir, rel)
            try:
                result = llm_inference_multimodal(
                    model_name=perception_model, prompt=prompt_for(idx), image_path=abs_path,
                    temp=temperature, max_tokens=max_tokens, num_ctx=num_ctx, think=think,
                    json_output=json_output,
                )
                text = result[0] if isinstance(result, tuple) else result
            except Exception as e:                     # record failure, keep going
                failed += 1
                out_file.write_text(json.dumps({
                    "image_index": idx, "image_path": abs_path,
                    "error": str(e), "generated_utc": prov.utc_now_iso(),
                }, indent=2))
                continue
            # Store pretty-printed so every condition's artifact reads the same
            # way, whether it was generated here or imported from an earlier run.
            try:
                text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                pass          # not JSON (e.g. the flat/C1 condition): store verbatim
            rec = {"image_index": idx, "image_path": abs_path,
                   "description": text, "generated_utc": prov.utc_now_iso()}
            if task_aware:
                rec["role"] = P.role_for_index(idx, m, n)
            out_file.write_text(json.dumps(rec, indent=2))
            generated += 1

    # content hash over every description file (proves the frozen set downstream)
    desc_texts = [p.read_text() for p in sorted(root.rglob("img_*.json"))]
    manifest = {
        "perception_model": perception_model,
        "context": context,
        "schema_version": schema_version,
        "prompt_sha256": prompt_sha,
        "dataset": dataset,
        "split_name": split_name,
        "json_path": json_path,
        "json_sha256": prov.sha256_file(json_path),
        "num_samples": len(test_ids),
        "m": m, "n": n,
        "decoding": {"temperature": temperature, "max_tokens": max_tokens,
                     "num_ctx": num_ctx, "think": think, "seed": 42},
        "counts": {"generated": generated, "skipped": skipped, "failed": failed},
        "content_sha256": prov.sha256_texts(desc_texts),
        "pri_git_commit": prov.git_commit(),
        "created_utc": prov.utc_now_iso(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"describe done: {generated} generated, {skipped} skipped, {failed} failed\n"
          f"  artifact: {root}\n  content_sha256: {manifest['content_sha256'][:16]}…")
    return str(root)
