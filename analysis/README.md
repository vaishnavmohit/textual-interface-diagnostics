# Analysis pipeline

Turns the per-experiment `results.xlsx` files into the paper's numbers, figures,
and LaTeX tables. Three stages, each a standalone script (run on HPC or locally
after copying the outputs down).

```bash
# 1. Combine every results.xlsx into one table per program (+ combined_all.csv)
python analysis/combine_results.py --root ${OUTPUT_DIR}

# 2. Accuracy + cluster bootstrap CIs + cluster-aware paired tests
python analysis/statistics.py \
    --combined ${OUTPUT_DIR}/combined/combined_all.csv --out ${OUTPUT_DIR}/stats

# 3. Publication figures (ladder bars + CI) and LaTeX tables
python analysis/make_figures.py --stats ${OUTPUT_DIR}/stats --out ${OUTPUT_DIR}/report
```

## What each produces

- **combine_results.py** → `${OUTPUT_DIR}/combined/`: `<program>_combined.xlsx`
  per program and `combined_all.csv` (all rows, annotated with
  experiment/model/program/source). Handles the Bongard (`is_correct`) and
  Winoground (`text/image/group_score`) schemas together.
- **statistics.py** → `stats/`: `accuracy_with_ci.csv` (per experiment×model×
  program, with 95% CIs — clustered by Bongard concept when a concept/uid column
  is present) and `paired_cluster_tests.csv` (every program-vs-program
  cluster-level sign-flip test within each experiment×model, with
  Holm-corrected p-values). Seeded (42) so CIs and Monte Carlo tests are
  reproducible.
- **make_figures.py** → `report/`: `figures/<experiment>_ladder.png` (accuracy by
  interface condition with CI error bars, per model) + caption stubs, and
  `tables/accuracy.tex` / `tables/paired_cluster_tests.tex`.

## The contrasts this supports

The paired cluster-test table gives the E2 ladder contrasts directly: C2−C1
(structure), C3−C2 (task-conditioning), C4−C3 (extra look), C5−C4 (adaptivity),
plus DVRL/DRL/CA paradigm comparisons — each on common instances with corrected
significance.

## Notes

- Verified end-to-end on synthetic Bongard + Winoground trees; schema detection
  is data-driven (a union-concat leaves the other schema's columns present as
  NaN, so metrics are chosen by which column actually has data).
- `code/eval/gather_all_results.py` remains the auditor for the *legacy* COLM
  output tree (irregular flat dirs). This pipeline is for the clean 3-level pri
  tree.
