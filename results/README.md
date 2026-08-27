# Frozen results

This directory contains the frozen model outputs and derived analysis files
used by the accompanying IJCV manuscript. The three canonical per-sample
tables are:

| File | Rows | Benchmark |
|---|---:|---|
| `bongard_ow_per_sample.csv.gz` | 36,967 | Bongard-OpenWorld |
| `bongard_hoi_per_sample.csv.gz` | 8,199 | Bongard-HOI |
| `winoground_per_sample.csv.gz` | 9,751 | Winoground |

The tables retain raw predictions, validity indicators, correctness or score
fields, experimental-condition labels, and source hashes. The
Bongard-OpenWorld table excludes all 1,074 rows from three pilot runs whose
constrained output vocabulary was incompatible with the benchmark. Their run
identities, source hashes, and exclusion decisions remain in
`provenance/EXPERIMENT_REGISTRY.csv`. Other invalid model outputs remain
explicit in the canonical tables.

Additional directories contain per-run workbooks and derived outputs:

- `generation/`: cross-generation benchmark runs.
- `ladder/`: textual-interface ladder conditions.
- `samemodel/` and `staging/`: matched decomposition controls.
- `stats/`: primary confidence intervals, paired tests, separability, and ICC.
- `studies/`: study-specific derived tables.
- `behaviour/`: error and difficulty analyses.

Run `python scripts/verify_release.py` from the repository root to validate the
canonical row counts and required columns. See `provenance/E1_AUDIT.md` for the
audit trail and `provenance/DATA_SOURCE_MANIFEST.csv` for source-file hashes.

Use `pred_valid` and each row's `analysis_decision` when reproducing manuscript
statistics; do not silently coerce a missing `is_correct` value to `False`.
