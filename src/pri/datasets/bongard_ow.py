"""Bongard-OpenWorld loader.

Ported from code/dataset/bongard_openworld.py (frozen COLM provenance). A puzzle
is m positive + n negative support images plus one query; the reasoner must
induce the latent concept and classify the query as pos/neg.
"""

from __future__ import annotations

import json


def read_sample(json_path: str, test_id: str, m: int, n: int):
    """Return (image_paths, concept, caption, uid, test_cat, commonsense).

    ``image_paths`` is the m positives, then n negatives, then the query image
    (order matters: the prompt labels them by position). ``test_cat`` is the
    ground-truth pos/neg label of the query.
    """
    with open(json_path) as f:
        data = json.load(f)

    sample = next((item for item in data if item["test_id"] == test_id), None)
    if sample is None:
        raise ValueError(f"Sample with test_id {test_id} not found in {json_path}")

    positives = sample["imagefiles"]["cat_2"][:m]
    negatives = sample["imagefiles"]["cat_1"][:n]
    test_image = sample["testfiles"]["testimage"]
    test_cat = sample["testfiles"]["category"]
    image_paths = positives + negatives + [test_image]

    return (
        image_paths,
        sample["concept"],
        sample["caption"],
        sample["uid"],
        test_cat,
        sample["commonSense"],
    )


def list_test_ids(json_path: str):
    """All test_ids in the split, in file order (for sample selection)."""
    with open(json_path) as f:
        return [item["test_id"] for item in json.load(f)]
