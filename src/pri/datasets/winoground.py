"""Winoground loader.

Each entry is two images and two captions; the correct pairs are
(caption_0, image_0) and (caption_1, image_1). ``read_sample`` returns the two
image paths first (so the shared describe stage can produce one description per
image, as img_00 / img_01) followed by the captions and tag.

Expects a JSON list of entries with fields: id, image_0, image_1, caption_0,
caption_1, and optionally tag.
"""

from __future__ import annotations

import json


def _load(json_path: str):
    with open(json_path) as f:
        return json.load(f)


def read_sample(json_path: str, test_id, m: int = 0, n: int = 0):
    """Return (image_paths[2], caption_0, caption_1, id, tag).

    m/n are unused (kept for a uniform loader signature with the Bongard
    loaders). image_paths order is [image_0, image_1] → described as img_00,
    img_01.
    """
    data = _load(json_path)
    entry = next((e for e in data if str(e.get("id")) == str(test_id)), None)
    if entry is None:
        raise ValueError(f"Winoground entry id {test_id} not found in {json_path}")
    image_paths = [entry["image_0"], entry["image_1"]]
    return image_paths, entry["caption_0"], entry["caption_1"], entry.get("id"), entry.get("tag")


def list_test_ids(json_path: str):
    return [e.get("id") for e in _load(json_path)]
