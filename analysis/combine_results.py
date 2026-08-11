#!/usr/bin/env python3
"""Combine every per-experiment results.xlsx into one table per program.

Walks the pri output tree
    <root>/<experiment>/<model>/<program>/results.xlsx
(``program`` = e.g. ``ca_temp_0.0``, ``ica_temp_0.0``), annotates each row with
its experiment / model / program / source path, and writes one combined file per
program to ``<out>/combined/``. Also writes an all-rows ``combined_all.csv``.

Handles both result schemas automatically (Bongard classification with
``is_correct``; Winoground with ``text_score``/``image_score``/``group_score``) —
they are concatenated with a union of columns.

Ported from hmr's combine_human_context_results.py, adapted to the pri tree.

Usage (on HPC or locally):
    python analysis/combine_results.py --root ${OUTPUT_DIR} --out ${OUTPUT_DIR}
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


def discover(root: Path):
    """Yield (experiment, model, program, results_path) for every results.xlsx
    at the expected 3-level depth. Skips the descriptions/ and combined/ dirs."""
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if exp_dir.name in ("descriptions", "combined"):
            continue
        for model_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            for prog_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                rf = prog_dir / "results.xlsx"
                if rf.is_file():
                    yield exp_dir.name, model_dir.name, prog_dir.name, rf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="output root to scan")
    ap.add_argument("--out", default=None, help="where to write combined/ (default: --root)")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    out = Path(args.out).expanduser() if args.out else root
    combined_dir = out / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    by_program: dict[str, list[pd.DataFrame]] = defaultdict(list)
    found = 0
    for experiment, model, program, rf in discover(root):
        try:
            df = pd.read_excel(rf)
        except Exception as e:  # noqa: BLE001
            print(f"  WARN could not read {rf}: {e}")
            continue
        df["experiment"] = experiment
        df["model_name"] = model
        df["program_name"] = program
        df["source_results_path"] = str(rf)
        by_program[program].append(df)
        found += 1
    print(f"discovered {found} results.xlsx across {len(by_program)} programs")

    all_rows = []
    for program, dfs in sorted(by_program.items()):
        combined = pd.concat(dfs, ignore_index=True, sort=False)
        combined.to_excel(combined_dir / f"{program}_combined.xlsx", index=False)
        all_rows.append(combined)
        print(f"  {program}: {len(combined)} rows from {len(dfs)} files")

    if all_rows:
        allc = pd.concat(all_rows, ignore_index=True, sort=False)
        allc.to_csv(combined_dir / "combined_all.csv", index=False)
        print(f"wrote {combined_dir}/combined_all.csv ({len(allc)} rows) and "
              f"{len(by_program)} per-program workbooks")


if __name__ == "__main__":
    main()
