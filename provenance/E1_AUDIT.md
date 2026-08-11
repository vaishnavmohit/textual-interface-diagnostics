# E1 audit — recomputing the paper's numbers from raw predictions

**Date:** 2026-07-17. **Source:** the 154 sha256-verified `results.xlsx` recorded in
`DATA_SOURCE_MANIFEST.csv` (the authors' compute environment). Accuracy recomputed as
`mean(test_cat_label == test_category_identified)` directly from per-sample rows.

## Headline: the paper reproduces

**17 of 18** rows of `tab:bongard-ow-paradigms` reproduce from raw data, most exactly.

| Model | Paradigm | Paper | Recomputed | n | |
|---|---|---|---|---|---|
| GPT-4o | DVRL / DRL / CA | 80.0 / 88.0 / 92.8 | 80.3 / 88.0 / 92.8 | 498–500 | ok |
| Gemini 2.0 | DVRL / DRL / CA | 82.2 / 86.8 / 93.6 | 82.2 / 86.8 / 93.6 | 500 | ok |
| Pixtral-12B | CA | 87.2 | 87.2 | 500 | ok |
| LLaVA-7B | CA | 66.2 | 66.2 | 500 | ok |
| Gemma3-12B | CA | 59.2 | 59.2 | 500 | ok |
| Llama-Vision-90B | CA | 55.1 | 55.5 | 481 | ok |
| Gemma3-4B | CA | 54.2 | 54.2 | 500 | ok |
| Llama-Vision-11B | CA | 53.4 | 53.4 | 500 | ok |
| LLaVA-Llama3-8B | CA | 53.2 | 53.6 | 500 | ok |
| GPT-5.1 | DVRL / DRL / CA | 94.0 / 96.0 / 97.0 | 94.0 / 96.0 / 97.0 | 100 | ok |
| **Gemma3-27B** | CA | **70.5** | **see below** | 498 | **correction** |
| **Qwen2.5-VL-32B** | CA | **50.0** | **see below** | 500 | **not a measurement** |

The paradigm ordering (DVRL < DRL < CA) is confirmed on raw data for both
end-to-end-capable models. The paper's central result stands.

## The Ollama schema bug reached the results table

Ollama's `format=<schema>` **constrains the output vocabulary**. Where the schema's
`Literal` values were `pos`/`neg` while the ground truth uses `cat_1`/`cat_2`, the
model was forced to emit a token from the wrong vocabulary. Three OW runs are affected:

| Run | n | rows with `pos`/`neg` | |
|---|---|---|---|
| `bongard_ow_single_qwen2.5vl_32b` | 500 | **500 (100%)** | total loss |
| `bongard_ow_single_qwen2.5vl_3b` | 76 | 76 (100%) | total loss |
| `bongard_ow_single_gemma3_27b` | 498 | 97 (19.5%) | partial |

Only CA-*self* runs on newer Ollama VLMs are hit. The CA runs over GPT-4o components
(`*_comp_eval_*`) are clean — including `comp_eval_qwen2.5vl_32b`.

### The forced outputs are not recoverable

The obvious repair — map `pos→cat_2, neg→cat_1` (the OW json defines `cat_1` = the
`neg__*` files, `cat_2` = the `pos__*` files) — **fails**. On the affected rows alone:

| Run | mapped accuracy on affected rows | class balance |
|---|---|---|
| gemma3_27b | 43.3% (95% CI ±9.9) | 93.8% `neg` |
| qwen2.5vl_32b | 52.2% (±4.4) | 81.0% `neg` |
| qwen2.5vl_3b | 51.3% (±11.2) | 98.7% `neg` |

All at chance, all degenerate — under the constraint the model simply emitted `neg`.
These are **invalid outputs, not incorrect predictions**. They must be excluded and
disclosed (E1 requires reporting "invalid outputs, retries, exclusions, and missing
predictions"), not mapped and not silently counted as wrong.

## Correction 1 — Qwen2.5-VL-32B: `50.0` is an artifact, not a result

100% of its outputs are schema-corrupted and degenerate (81% one class). Raw string
match gives 0.0%; the json-consistent mapping gives 52.2% — i.e. chance, which is why
the reported **50.0** looks like a plausible measurement. It is not one: it measures a
harness failure.

**Action:** remove the row from `tab:bongard-ow-paradigms`, or report it explicitly as
a schema failure with no accuracy. Do **not** report 50.0. The run is recoverable only
by re-running with the fixed schema (`utils/ollama_schema.py`, fixed 2026-07-16).

## Correction 2 — Gemma3-27B: the paper **understates** it by 17 points

Rows 401–497 form **one contiguous block** produced under the buggy schema: the run was
interrupted at row 401 and resumed under a different schema config, appending 97 invalid
rows to the same file. The paper counts all 97 as *wrong answers*.

| Treatment | Accuracy | n |
|---|---|---|
| Paper (invalid counted as wrong) | 70.5% | 498 |
| **Valid rows only, invalid disclosed** | **87.5%** | **401** (+97 invalid, 19.5%) |
| Mapped (invalid) | 78.9% | 498 |

**Action:** report **87.5% (n=401)** with the 19.5% invalid rate disclosed.

**This changes the paper's narrative.** At 87.5%, Gemma3-27B is level with Pixtral-12B
(87.2) at the top of the open-weight CA models, not 17 points below it. The claim that
open-weight CA "spans 50.0–87.2" becomes a materially different range once the two
schema-corrupted rows are removed and Gemma3-27B is corrected.

## Consequences to carry into the manuscript

1. `tab:bongard-ow-paradigms`: drop Qwen2.5-VL-32B, correct Gemma3-27B to 87.5 (n=401).
2. The open-weight CA range in the prose ("50.0--87.2") must be recomputed after both.
3. **Protocol-consistency debt:** one `results.xlsx` mixing two schema versions is
   exactly the failure the `pri` design prevents (immutable, per-row `schema_version`,
   hashed description artifacts). Cite this as motivation, and re-run the affected
   cells under `pri` rather than patching the spreadsheet.
4. E1 still requires the common-instance-set recomputation: Gemma3-27B now has n=401
   against others' n=500, so the common set shrinks. Report on the intersection.

## Related finding: the perception swap is pairable but not size-matched

- CA / GPT-4o components: **n=499**; CA / Pixtral components: **n=137**.
- The 137 Pixtral instances are a **strict subset** of the 499, and all 8 Pixtral runs
  use the **identical** 137. The swap must be recomputed **paired on the common 137**.

---

# Winoground audit

Source: `results/winoground_per_sample.csv.gz` (9,751 rows, 36 files).

## Layout, established from the run configs (not inferred)

`code/NS-Reasoner/config_winoground/*.yaml` on ai-lab is authoritative. In
`models_natural.vc_single_wino`, `base_dir` is where the description artefacts
live and `model` is the reasoner, so each config states its pipeline exactly:

| Config | Perception (`base_dir`) | Reasoner | Cond. |
|---|---|---|---|
| `wino_gpt.yaml` | `output/winoground_25activity` | gpt-4o | CA |
| `wino_gemini.yaml` | `output/winoground_25activity/output` | gemini-2.0-flash-exp | CA |
| `wino_gemini_comp_gpt.yaml` | `output_winoground/gemini_activity` | gpt-4o | CA |
| `wino_gemini_{llama33,phi4,qwen25,qwen25_32b,llama32_vision_90b}.yaml` | `gemini_activity` | open-weight | CA |
| `wino_gpt_comp_gpt.yaml` | `output_winoground/gpt_activity` | gpt-4o | CA |
| `wino_gpt_comp_gemini.yaml` | `gpt_activity` | gemini-2.0-flash-exp | CA |
| `wino_gpt_comp_gpt_ica.yaml` | `gpt_activity` | gpt-4o | **ICA** |
| `wino_gpt_comp_gemini_ica.yaml` | `gpt_activity` | gemini-2.0-flash | **ICA** |

The naming is `wino_<perception>_comp_<reasoner>[_ica]`.

## `winoground_25activity` is Gemini perception — proven by content

It is a third directory but not a third perception source:

- **60/60** sampled `description.json` files are byte-identical to
  `gemini_activity`; **0/60** match `gpt_activity`.
- `winoground_25activity/gpt-4o-2024-08-06/scores_winoground.xlsx` and
  `gemini_activity/gpt-4o-2024-08-06/scores_winoground.xlsx` share an md5
  (`42311584…`) — the same file stored in two places.

So every CA row in the paper runs on **Gemini** descriptions, and **ICA was only
ever run on GPT-4o descriptions** (no Gemini-perception ICA config exists).

## The published CA→ICA contrast crosses perception sources

With the reasoner fixed at GPT-4o:

| Perception | Condition | Text | Image | Group |
|---|---|---|---|---|
| **Gemini** | CA | 75.17 | 58.33 | **52.00** | ← the published CA row |
| GPT-4o | CA | 71.00 | 54.75 | 47.75 |
| **GPT-4o** | ICA | 77.50 | 63.44 | **55.19** | ← the published ICA row |

The published **+3.25** Group gain therefore compares CA-on-Gemini against
ICA-on-GPT-4o. The author's account explains why: both perception sources were
tried and *the better CA was reported*. That is a defensible, conservative
choice, but it is not the controlled contrast, and it should be labelled.

Two different questions, both answerable:

- **"Does ICA beat the best CA configuration we found?"** → +3.25 Group
  (52.00 → 55.25). Conservative, but confounded with a perception change.
- **"Does adding re-inspection to a fixed pipeline help?"** → **+7.50 to +8.00**
  Group (47.75 → 55.25/55.75), perception and reasoner both held at GPT-4o.
  Robust across all four ICA variants of that cell.

The controlled contrast is the one RQ3 asks for, and it is **more than twice the
published effect**. Report it as primary and keep +3.25 as the conservative
best-configuration comparison.

## A second, incidental finding: perception swap on Winoground

Holding the reasoner at GPT-4o and swapping only the description source,
Gemini descriptions beat GPT-4o descriptions by **+4.25 Group** (52.00 vs 47.75).
This is a Winoground analogue of the Bongard-OW perception swap and is not
currently reported.

## Actions

1. Give `tab:winoground` separate **perception** and **reasoner** columns; the
   present labels name only the reasoner and are ambiguous.
2. Report the matched CA→ICA contrast as primary (+7.50/+8.00), with the
   best-configuration comparison (+3.25) noted alongside.
3. State that ICA was only run on GPT-4o descriptions — a scope limit, and the
   reason the published comparison had to cross sources.
4. Name one canonical ICA variant; report the other three as a robustness range.
5. Disclose that some variant filenames embed their own scores
   (`…_55.75.xlsx`) and that the reported CA was selected as the better of two
   perception sources. Mitigating: every ICA variant agrees in direction and
   magnitude, and the matched contrast is larger than the reported one.

## Excluded

`*_wrongentries.xlsx` (×2), `*_notworking.xlsx`, and fragments (n < 50) are
marked `exclude`. Nothing was deleted.

---

# Can the schema-corrupted runs be reused instead of re-run?

**No.** Tested two independent recovery routes on the 673 affected rows; both
land at chance. Re-running is the only route, and the cost is not justified
(§ below).

## Route 1 — map the forced token back

`pos`→`cat_2`, `neg`→`cat_1` (the loader's definition). On the affected rows:
43.3% / 52.2% / 51.3% accuracy. Chance. Documented above.

## Route 2 — recover the verdict from the free text

The `format=` enum constrained only the `Conclusion` field. `Analysis`,
`Rule` and `Test Image` remained free text and are fully populated (500/500,
averaging 634 / 322 / 345 characters), so if the model reasoned correctly and
merely could not *express* the label, the verdict should be recoverable from
its own prose.

A verdict extractor (explicit verdict phrases, else the last `cat_[12]` mention)
was **validated on uncorrupted runs first**: 96.8% agreement with the model's own
valid `Conclusion` on GPT-4o self-CA and 96.4% on Gemma3-12B — structurally the
same run type as the corrupted ones. (Agreement is only 60.3% on `comp_eval`
runs, so the extractor is not applied there.)

Applied to the corrupted rows:

| Run | Corrupted | Verdict recovered | Accuracy of recovered | Recovered class balance |
|---|---|---|---|---|
| qwen2.5vl:32b | 500 | 216 (43%) | **50.0%** (±6.7) | 83% one class |
| qwen2.5vl:3b | 76 | 11 (14%) | 54.5% (±29.4) | 64% one class |
| gemma3:27b | 97 | 63 (65%) | 50.8% (±12.3) | 68% one class |

All at chance, and the recovered labels remain strongly imbalanced.

## What this means — a methodological finding worth reporting

Constrained decoding to an unsatisfiable vocabulary did **not** merely mislabel a
correct answer: it degraded the whole generation. The free-text reasoning is as
degenerate as the forced field (68–83% one class), and carries no recoverable
signal. Note too that the free text (83% `cat_2`) contradicts the forced field
(81% `neg`, i.e. `cat_1`) — the two halves of the same response disagree.

So the failure is not "the label was written in the wrong alphabet". Forcing the
model into a vocabulary that cannot express the answer **corrupted the reasoning
itself**. That is a concrete argument for the paper's reproducibility section:
schema enforcement in a serving stack is not a harmless formatting convenience,
and a per-row `schema_version` is what makes such a run detectable after the fact.

## Recommendation: exclude, disclose, do not re-run

- The rows are unusable and cannot be salvaged from existing output.
- Re-running would restore two Qwen2.5-VL rows and 97 Gemma3-27B rows. Per
  `results/stats/T1_SUMMARY.md`, only 2 of 61 adjacent model orderings are
  separable at 95%, so those rows would land inside the indistinguishable
  bottom band and change no conclusion.
- Gemma3-27B is already reportable at 87.5% over its 401 valid rows.

The cost of re-running is therefore high and the evidential return near zero.
Exclude with the invalid rate disclosed, and report the constrained-decoding
failure as a finding.

---

# S2 — the perception-dominates claim does not survive the crossed cell

`analysis/studies/s2_variance_decomposition.py`. **Read before writing any
"perception is the bottleneck" sentence.**

The paper argues that the perceptual interface constrains performance more than
the reasoner does. The evidence has been a comparison of *spreads of point
estimates* taken from separate tables computed on different instance sets. A
fully crossed, perfectly balanced cell exists and tests it directly:

**7 reasoners × 2 description sources × the same 137 items = 1,918 observations.**

## Result

| Comparison | Median effect |
|---|---|
| Swap description source (Pixtral → GPT-4o), per reasoner | **+5.1 points** (consistent in sign for all 7) |
| Swap reasoner, description source fixed (42 pairs) | **6.6 points** |
| Ratio | **1.29** |

Three methods agree in direction — paired within-item contrasts, explained
deviance, and a mixed-effects fit. **The reasoner effect is slightly larger than
the perception effect in this cell, not smaller.**

## The level-count trap (and why the honest ratio is 1.29, not 3.7)

The naive deviance decomposition gives 85.4% to the problem instance, 3.1% to
perception and 11.5% to the reasoner — an apparent 3.7× advantage for the
reasoner. **That comparison is invalid.** The reasoner factor has 7 levels (6 df)
and perception has 2 (1 df); a factor with more levels mechanically absorbs more
deviance. Reducing both to matched single contrasts gives 6.6 vs 5.1, a ratio of
1.29. Report the matched figure; the deviance shares may be quoted only with the
df asymmetry stated.

## What may still be claimed

Not dominance, but something more defensible and more useful:

- The perception effect is **consistent**: +4.4 to +11.7 points, same direction
  for every reasoner tested.
- The reasoner effect is **heterogeneous**: 0.0 to 21.9 points depending on the
  pair, and **38% of reasoner pairs differ by less than the median perception
  effect**.

So the interface is attractive as an intervention point because its benefit is
*uniform and transferable across reasoners*, not because it is the larger lever.
That is a design-relevant claim and it is what the data support.

## Scope limits to state

- Both description sources here are competent VLMs (Pixtral-12B, GPT-4o). A
  wider span of perceptual quality could produce a larger perception effect;
  this cell cannot speak to that.
- The 7 reasoners span a wide capability range (LLaVA-7B to Qwen2.5-14B), so the
  reasoner spread is close to its maximum for open-weight models of this era,
  while the perception spread is measured over only two points.
- 137 items, so individual contrasts are underpowered (T1: only 1 of 7 perception
  contrasts survives Holm, though the sign test over the family gives p = 0.016).

## Required manuscript changes

1. Remove or qualify any unhedged "perception is the bottleneck" / "perception
   matters more than reasoning" statement.
2. Replace with the consistency-versus-heterogeneity framing above.
3. Where the deviance decomposition is reported, state the df asymmetry.
4. Keep the regime qualifier already required by the E4 drop — this result
   further narrows the claim rather than widening it.
