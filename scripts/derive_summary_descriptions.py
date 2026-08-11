#!/usr/bin/env python3
"""Derive a summary-only description artifact from an existing structured one.

The structured (C2) schema ends with a one-or-two-sentence ``Summary`` field.
Keeping only that field yields a condition in which the reasoner receives prose
at roughly a tenth of the full artifact's length, produced by the *same*
perceptual pass over the *same* images.

What this isolates, and what it does not:

  * It DOES isolate the information the interface passes downstream. Both arms
    read one generation, so the describer, decoding, and every perceptual
    judgement are identical by construction -- there is no describer confound
    at all, which a freshly generated free-form condition cannot claim.
  * It DOES vary format and volume together: the summary is prose, and it is
    ~11% of the structured artifact's characters.
  * It does NOT isolate structure. The summary was written *after* the model had
    decomposed the scene into eight categories, so it distils structured
    attention rather than replacing it. A genuine unstructured-perception
    condition has to be generated from a prose prompt (context ``ca_flat``).

So this answers "does the reasoner need the structured detail, or does a short
prose distillation of the same pass suffice?" -- not "does structure help the
describer?".

Nothing is generated: this reads the frozen structured artifact read-only and
writes a new one under a distinct schema_version, so the two can never be
confused. Every record keeps the source path and hash it was derived from.

    python scripts/derive_summary_descriptions.py \
        --out ~/pri_output --model-name gpt-4o-2024-08-06 --split bongard_ow
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SOURCE_SCHEMA = "structured_v1"
TARGET_SCHEMA = "summary_v1"


def sha256_texts(texts) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode())
    return h.hexdigest()


def extract_summary(description: str) -> str | None:
    """The Summary field, or None if the record has no usable one."""
    try:
        obj = json.loads(description)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    for key in ("Summary", "summary", "Overall Summary"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, type=Path, help="OUTPUT_DIR root")
    ap.add_argument("--model-name", required=True,
                    help="perception model whose artifact is the source")
    ap.add_argument("--split", required=True, help="split_name, e.g. bongard_ow")
    args = ap.parse_args()

    src_root = args.out / "descriptions" / args.model_name / SOURCE_SCHEMA / args.split
    dst_root = args.out / "descriptions" / args.model_name / TARGET_SCHEMA / args.split
    if not src_root.exists():
        raise SystemExit(f"source artifact not found: {src_root}")

    src_manifest = src_root / "manifest.json"
    src_hash = "unknown"
    if src_manifest.exists():
        src_hash = json.loads(src_manifest.read_text()).get("content_sha256", "unknown")

    dst_root.mkdir(parents=True, exist_ok=True)
    written = no_summary = no_description = 0

    for f in sorted(src_root.rglob("img_*.json")):
        rec = json.loads(f.read_text())
        desc = rec.get("description")
        if not desc:
            no_description += 1        # a failure record in the source, not a description
            continue
        summary = extract_summary(desc)
        if summary is None:
            no_summary += 1
            continue
        out_dir = dst_root / f.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f.name).write_text(json.dumps({
            "image_index": rec.get("image_index"),
            "image_path": rec.get("image_path"),
            "description": summary,
            "generated_utc": None,
            "derived_from": str(f),
            "derived_from_schema": SOURCE_SCHEMA,
            "derived_from_manifest_sha256": src_hash,
            "derivation": "Summary field extracted verbatim; no model call.",
        }, indent=2, ensure_ascii=False))
        written += 1

    texts = [p.read_text() for p in sorted(dst_root.rglob("img_*.json"))]
    (dst_root / "manifest.json").write_text(json.dumps({
        "perception_model": args.model_name,
        "context": "ca_summary",
        "schema_version": TARGET_SCHEMA,
        "split_name": args.split,
        "num_samples": len({p.parent.name for p in dst_root.rglob("img_*.json")}),
        "derived": True,
        "derived_from_schema": SOURCE_SCHEMA,
        "derived_from_manifest_sha256": src_hash,
        "note": ("Summary-only view of the structured artifact. Prose at ~11% of "
                 "the source length, from the SAME perceptual pass -- a "
                 "content-volume and format ablation, not an unstructured-"
                 "perception condition (that is context ca_flat)."),
        "counts": {"written": written, "no_summary_field": no_summary,
                   "source_failure_records": no_description},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "content_sha256": sha256_texts(texts),
    }, indent=2))

    print(f"derived {written} summary descriptions -> {dst_root}")
    print(f"  source records with no Summary field : {no_summary}")
    print(f"  source failure records skipped       : {no_description}")
    print(f"  source manifest sha                  : {src_hash[:16]}…")


if __name__ == "__main__":
    main()
