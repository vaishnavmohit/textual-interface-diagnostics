"""Bongard-HOI loader.

Ported from code/dataset/bongard_hoi.py. Same structure as Bongard-OpenWorld
(m positives + n negatives + one query), for human--object interaction concepts;
concept/caption/commonsense are absent in this benchmark.
"""

from __future__ import annotations

import json


def read_sample(json_path: str, test_id: str, m: int, n: int):
    """Return (image_paths, concept=None, caption=None, uid, test_cat, commonsense=None)."""
    with open(json_path) as f:
        data = json.load(f)
    sample = next((item for item in data if item["test_id"] == test_id), None)
    if sample is None:
        raise ValueError(f"Sample with test_id {test_id} not found in {json_path}")
    positives = sample["imagefiles"]["cat_2"][:m]
    negatives = sample["imagefiles"]["cat_1"][:n]
    image_paths = positives + negatives + [sample["testfiles"]["testimage"]]
    return image_paths, None, None, sample["uid"], sample["testfiles"]["category"], None


def list_test_ids(json_path: str):
    with open(json_path) as f:
        return [item["test_id"] for item in json.load(f)]
