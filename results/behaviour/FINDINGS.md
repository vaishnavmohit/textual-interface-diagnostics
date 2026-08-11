# Behavioural analysis — Bongard-OpenWorld

Produced by `analysis/behaviour_analysis.py` from `results/bongard_ow_per_sample.csv.gz`
(36,297 valid predictions, 84 runs). Accuracy alone answers none of these questions.

Caveat: no confidence intervals yet — these are point estimates on 500-item splits
(E1 adds clustered CIs and paired tests). Treat magnitudes as indicative and
directions as the finding.

---

## 1. Two "weak" models are not weak — they are degenerate

Bongard-OW query sets are 50/50 positive/negative, so a model that always answers
one class scores ~50% without discriminating at all. Splitting accuracy by query
polarity exposes this:

| Model (CA, own components) | acc | pos | neg | gap | % answered positive |
|---|---|---|---|---|---|
| **llava-llama3** | 53.6 | 14.0 | 93.2 | **−79.2** | **10.4** |
| **llava:13b** | 47.7 | 14.0 | 81.3 | **−67.3** | **16.4** |
| gemma3:4b | 54.2 | 49.6 | 58.8 | −9.2 | 45.4 |
| llama3.2-vision:90b | 55.5 | 51.7 | 59.3 | −7.7 | 46.2 |
| pixtral-12b-2409 | 87.4 | 85.6 | 89.2 | −3.6 | 48.2 |
| gemma3:12b | 59.2 | 59.2 | 59.2 | 0.0 | 50.0 |
| gpt-4o | 92.8 | 92.8 | 92.8 | 0.0 | 50.0 |
| llava:7b | 66.2 | 67.2 | 65.2 | +2.0 | 51.0 |
| gemma3:27b | 87.5 | 90.0 | 85.1 | +4.9 | 52.4 |
| gemini-2.0-flash-exp | 93.6 | 96.4 | 90.8 | +5.5 | 52.6 |
| llama3.2-vision | 53.4 | 64.0 | 42.8 | +21.2 | 60.6 |

**LLaVA-Llama3-8B answers "negative" on ~90% of items.** Its 53.6% is what
near-constant answering yields on a balanced set, not a measure of concept
induction. LLaVA-13B is the same. This matters for the paper: the claim that
open-weight CA "spans 53.4–87.5" implies a continuum of reasoning ability, when
the bottom of that range is a model not discriminating at all.

**Recommended:** report the positive-answer rate alongside accuracy, and mark
degenerate rows. A reader cannot distinguish 53.6% "weak reasoning" from 53.6%
"constant answer" without it.

## 2. Paradigms differ in *acceptance bias*, not just accuracy

Across every run, grouped by paradigm (% answered positive; 50% = balanced):

| Paradigm | % answered positive | pos−neg accuracy gap | runs |
|---|---|---|---|
| **DRL** | 51.0 – 55.2 | +2.3 to +10.4 | 3/3 near-balanced |
| **DVRL** | 51.0 – 73.1 | +2.0 to +46.4 | biased in most runs |
| **RuleApply** | 59.2 – 80.2 | +17.6 to +60.4 | **10/10 biased** |

This is systematic, not one bad model: **all ten** RuleApply runs over-answer
positive, including the strongest (GPT-4o groundeval, 88.8% accuracy, still 59.2%
positive). The direction is consistent — when the rule is *given* rather than
derived, models over-accept the query as matching it.

DRL — where the model derives the rule from the support set and then applies it —
is the best calibrated of the three. That is a behavioural argument for the
paradigm ordering that accuracy alone does not make, and it is directly relevant
to RQ2: the paradigms differ in decision criterion, not only in competence.

**Bonus (robustness):** `bongard_ow_dvrl_gpt_noise` answers positive on **100%**
of items. Under image corruption the model does not degrade gracefully toward
chance — it collapses into constant acceptance. Worth stating explicitly if the
noise ablation is reported.

## 3. There is no universally hard core — difficulty is model-specific

Over 503 items × 9 non-degenerate full-split CA models (degenerate models excluded;
a constant answerer is trivially "correct" on half the items and would corrupt this):

| | share of items |
|---|---|
| every model correct | 6.2% |
| **every model wrong** | **0.0%** |
| mixed | 93.8% |

If models failed independently at the observed mean rate (70.8%), we would expect
~0.0% all-wrong and ~4.5% all-right. Observed: 0.0% and 6.2%. Both are close to
the independence prediction.

**Failures do not concentrate on shared hard instances.** There is no subset of
Bongard-OW that defeats all models — which argues against "some items are
impossible" and for genuinely idiosyncratic, model-specific failure modes. That
is a useful negative result: it means an ensemble or a better perception source
has headroom on essentially every item.

## 4. Strong models fail alike; weak models fail randomly

Correlation of per-item correctness (9 models, 478 common items):

| pair | r |
|---|---|
| gemini-2.0-flash ~ gpt-4o | **+0.416** |
| gpt-4o ~ pixtral-12b | +0.361 |
| gemini-2.0-flash ~ pixtral-12b | +0.302 |
| llava:7b ~ pixtral-12b | +0.156 |
| gemma3:12b ~ llama3.2-vision | −0.074 |
| llama3.2-vision ~ llava-llama3 | **−0.187** |

The three strongest models agree on *which* items they get wrong (r ≈ 0.3–0.42),
while weak models show near-zero or negative correlation with everyone. Combined
with §3: strong models track a real, if shallow, item-difficulty signal; weak
models contribute noise. The negative correlations involve the degenerate models
and are an artifact of their constant answering.

## 5. What predicts failure

| Factor | Effect |
|---|---|
| **Query polarity** | **No effect** once degenerate models are excluded (positive 71.2% vs negative 70.6%). The apparent 8-point asymmetry in the raw aggregate was entirely an artifact of the two degenerate models. |
| **Commonsense flag** | No monotonic relationship (level 0: 71.6%, level 4: 76.8%, level 5: 60.0%); most levels have n<20 items, so this is underpowered rather than null. |
| **Concept** | Cannot be tested directly — Bongard-OW gives ~one distinct concept per problem. Grouping by concept head noun (≥5 items): hardest = *field, river, forest, floor, lake*; easiest = *water, snow, plant, sea, sky*. Crude and not to be over-read. |

The polarity result is a caution about the whole exercise: the two most striking
"findings" in the raw aggregate (an 8-point polarity gap, a shared hard core) both
dissolved once degenerate models were removed. Behavioural claims need the
degeneracy check first.

---

## Implications for the manuscript

1. **Disclose degeneracy.** Report % answered positive next to accuracy; flag
   LLaVA-Llama3-8B and LLaVA-13B as near-constant. Soften "spans 53.4–87.5" —
   the low end is not weak reasoning.
2. **§2 is a new result worth reporting.** Acceptance bias by paradigm (DRL
   calibrated, RuleApply systematically over-accepting in 10/10 runs) speaks to
   RQ2 and does not exist in the current draft.
3. **§3 belongs in the limitations/benchmark discussion:** no universally hard
   core, so headroom is broad rather than capped by impossible items.
4. **The noise collapse (100% positive)** should be stated wherever the corruption
   ablation is reported — it is a stronger statement than "accuracy drops".
