"""Provenance helpers: content hashing and environment capture.

Kept separate so both the describe (Stage 1) and reason (Stage 2) stages record
identical, auditable metadata (see docs/REPRODUCIBILITY_DESIGN.md).
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_texts(texts) -> str:
    """Order-independent content hash over a collection of strings."""
    h = hashlib.sha256()
    for t in sorted(texts):
        h.update(t.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def git_commit() -> str:
    """Current pri repo commit, or 'unknown' outside a checkout."""
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_frozen_manifest(desc_root, perception_model: str, context: str, split_name: str) -> dict:
    """Return the description artifact's manifest, or refuse to proceed.

    Two failure modes, both fatal, because a reasoning run must never invent or
    silently reason over an incomplete description set:

    * no manifest -- the describe stage has not been run for this cell;
    * a manifest recording failed calls -- the describe stage ran but did not
      produce every description, so the content hash covers error records rather
      than descriptions. Rerunning describe repairs those records in place
      (``describe._is_failed``), which is why this is a stop rather than a warn.
    """
    from pathlib import Path
    import json

    manifest_path = Path(desc_root) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No description artifact at {desc_root}. Run pri.describe first for "
            f"perception_model={perception_model}, context={context}, split={split_name}."
        )
    manifest = json.loads(manifest_path.read_text())
    failed = manifest.get("counts", {}).get("failed", 0)
    if failed:
        raise RuntimeError(
            f"Description artifact at {desc_root} records {failed} failed call(s): "
            f"it is incomplete, and reasoning over it would silently score missing "
            f"descriptions. Re-run the matching pri.describe config -- failed records "
            f"are retried automatically -- and confirm counts.failed == 0."
        )
    return manifest
