# Per-study analyses

One question per file. Each is standalone, reads only the committed source
files, runs offline in seconds, and prints a manuscript-ready paragraph next to
its CSV output. Results land in `results/studies/`.

```bash
python analysis/studies/s1_ica_selectivity.py
```

The point of splitting these out: the same 56k predictions answer very different
questions depending on the *unit of analysis* and the *conditioning variable*.
Aggregate accuracy uses one number per run; these studies use the per-item,
per-direction, per-tag structure that is already in the data and currently
unreported.

## Done

| Study | Question | Headline |
|---|---|---|
| **S1** `s1_ica_selectivity.py` | Why does ICA beat CA — a second look, or choosing when to look? | Re-inspection is **selective and calibrated**: queried items had 59.5% baseline accuracy vs 91.9% for unqueried, a 32.3-point gap. Gain concentrates entirely in the self-flagged subset. The image direction (100% query rate) is a within-experiment control for indiscriminate re-inspection. → §RQ3 |

## Planned, in value order

| Study | Question | Method | Why it is worth doing |
|---|---|---|---|
| **S2** perception vs reasoner variance | How much of the outcome does the description source explain, relative to the reasoner? | Mixed-effects logistic on the crossed cell: 7 reasoners × 2 perception sources × 137 common items (1,918 paired obs). Variance components with profile CIs, cross-checked against fixed effects. | The paper's central claim currently rests on comparing spreads of point estimates. This makes it a quantity with an interval. |
| **S3** item difficulty (Rasch) | Is "difficulty" a single construct here, and do models differ only in ability? | 1PL via logistic regression with item + model effects; item-fit statistics; eigenvalue check for unidimensionality. | Puts items and models on one scale, and flags items that behave inconsistently — a benchmark-quality signal. If it is *not* unidimensional, that is itself reportable. |
| **S4** acceptance bias | Do paradigms differ in decision criterion, not just accuracy? | Positive-response rate per run with binomial CI against the 50% base rate; paradigm-level test. | Already observed: RuleApply over-accepts in 10/10 runs, DRL is calibrated. A criterion difference is a different claim from a competence difference, and it is new. |
| **S5** rule quality vs outcome | Are failures rule-*induction* or rule-*application* failures? | Embed `rule_identified` against the ground-truth `concept`/`caption`; correlate similarity with correctness; condition on paradigm. | The paper asserts this distinction without measuring it. Traces are local and complete (500/500). Sentence-BERT is already precedented in-project. |
| **S6** description fidelity | Measure the perception difference *directly* rather than inferring it from accuracy. | On the common 137, compare Pixtral vs GPT-4o descriptions for presence of the task-critical fact named in `caption`. | Closes the "model identity as a proxy for perception quality" gap from the opposite side to S2. |
| **S7** support-set size | How does accuracy scale with the number of support images? | `dvrl_ablation_{2x2,4x4,6x6}` (n=200 each) with clustered CIs; pair with the context-length variants. | Uses runs already completed and currently unreported. |
| **S8** error consistency | Do models fail on the same items, and does that structure follow architecture or scale? | Pairwise phi over per-item correctness; hierarchical clustering; compare within-family vs across-family agreement. | Extends the earlier finding (strong models agree, weak ones do not) into a structured claim about *what* drives shared failure. |

## Conventions

- Read from `results/*_per_sample.csv.gz`; never re-read the cluster.
- Filter `analysis_decision` and `pred_valid` before computing anything.
- `is_correct` is NA for invalid outputs — drop the pair, never score it wrong.
- Every estimate carries a CI; every claim names its instance set and n.
- Flag degenerate (near-constant) models wherever their numbers appear.
- If a result contradicts an assumption in the plan, say so in the output rather
  than quietly adjusting the method.
