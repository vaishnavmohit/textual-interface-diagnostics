#!/usr/bin/env python3
"""Fast integrity checks for the public research release."""

from __future__ import annotations

import csv
import gzip
import importlib
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "results/bongard_ow_per_sample.csv.gz": (36_967, {"pred_valid", "is_correct", "failure_mode"}),
    "results/bongard_hoi_per_sample.csv.gz": (8_199, {"pred_valid", "is_correct"}),
    "results/winoground_per_sample.csv.gz": (9_751, {"group_score", "analysis_decision"}),
}
REQUIRED_SUBMISSION_AUDITS = (
    "analysis/studies/s7j_hoi_cohort_audit.py",
    "analysis/studies/s7k_description_missingness.py",
    "results/studies/s7j_hoi_cohort_audit.csv",
    "results/studies/s7j_hoi_paired_denominators.csv",
    "results/studies/s7k_description_missingness.csv",
)
PRIVATE_PATH = re.compile(r"/(?:illukas|Users)/[^/]+/", re.IGNORECASE)
TEXT_SUFFIXES = {".cfg", ".csv", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}


def check_csv(relative: str, expected_rows: int, required: set[str]) -> None:
    path = ROOT / relative
    if not path.is_file():
        raise AssertionError(f"missing {relative}")
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = required - columns
        if missing:
            raise AssertionError(f"{relative} lacks columns: {sorted(missing)}")
        rows = 0
        for row in reader:
            rows += 1
            if relative.endswith("bongard_ow_per_sample.csv.gz") and row.get("failure_mode") == "schema_vocabulary":
                raise AssertionError(f"{relative}: contains an excluded schema-misconfigured row")
    if rows != expected_rows:
        raise AssertionError(f"{relative}: expected {expected_rows} rows, found {rows}")


def check_private_paths() -> None:
    hits: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if PRIVATE_PATH.search(text):
            hits.append(str(path.relative_to(ROOT)))
    if hits:
        raise AssertionError(f"private absolute paths found in: {', '.join(hits)}")


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    importlib.import_module("pri")
    for relative, (rows, required) in EXPECTED.items():
        check_csv(relative, rows, required)
    missing_audits = [relative for relative in REQUIRED_SUBMISSION_AUDITS if not (ROOT / relative).is_file()]
    if missing_audits:
        raise AssertionError(f"missing submission-stage audits: {', '.join(missing_audits)}")
    check_private_paths()
    print("release verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
