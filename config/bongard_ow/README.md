# E2 — the matched interface ladder on Bongard-OW

Grid: **4 conditions × 2 reasoners × 2 perception sources = 16 cells**, plus the
6 describe runs that feed them and the 2 end-to-end baselines.

| Cond. | Config | What it varies |
|---|---|---|
| C1 | `reason_ca_flat_*` | task-blind **free-form** description (no schema) |
| C2 | `reason_ca_*` | task-blind **structured** — the frozen CA schema |
| C3 | `reason_ca_taskaware_*` | **byte-identical C2 schema** + a task block |
| C4 | `reason_ca_fixed_reinspect_*` | C2 + **one fixed** extra visual call |

Contrasts: **C2−C1** = representation structure. **C3−C2** = task conditioning
(schema is identical, so only task info differs). **C4−C3** = one extra visual
inspection.

## Run order — stage 1 before stage 2

Descriptions are **frozen artifacts**. Run each `describe_*` **once**; every
reasoner then consumes the identical descriptions by hash. `reason_*` aborts if
the artifact is missing — it never generates descriptions, because
re-describing between reasoners would break the controlled swap.

```bash
# Stage 1 — 6 runs (3 contexts x 2 perception). C4 reuses the context=ca artifact.
for p in pixtral-12b gpt-4o; do
  for c in ca ca_flat ca_taskaware; do
    python main.py --config config/bongard_ow/describe_${c}_${p}.yaml
  done
done

# Stage 2 — 16 runs
for d in pixtraldesc gpt4odesc; do
  for r in qwen2.5-14b phi4-14b; do
    for c in ca_flat ca ca_taskaware ca_fixed_reinspect; do
      python main.py --config config/bongard_ow/reason_${c}_${d}_${r}.yaml
    done
  done
done
```

Set `num_samples: 500` (full split) before the real runs — the configs ship at 5
for a smoke run.

## Output layout

```
${OUTPUT_DIR}/bongard_ow_<perception>desc/<reasoner>/<condition>_temp_0/results.xlsx
```

`experiment` is keyed by perception source (`bongard_ow_pixtraldesc` vs
`bongard_ow_gpt4odesc`) **deliberately**: the output path is built from
`(experiment, reasoner, context, temp)` and does *not* include
`perception_model`. Without the distinct key both perception sources would write
to the same `results.xlsx`, and since the runner is resume-safe (it skips
test_ids already present) the second run would silently skip the first's rows —
producing one file that mixes two perception sources under one label. Do not
collapse these back to a shared `experiment` name.

## C5 and the "adaptive" claim — on Winoground, not here

**C5 (adaptive reinspection) has no Bongard-OW entry point.** `pri.pipeline` has
`run_fixed_reinspect` (C4) but no adaptive counterpart, so the ladder here stops
at C4 and the contrasts above (C2−C1, C3−C2, C4−C3) are what Bongard-OW supports.

The **C5−C4 contrast lives on Winoground**, where ICA evidence already is:

```
config/winoground/reason_winoground_gpt-4o.yaml        C2  CA
config/winoground/reason_fixed_reinspect_gpt-4o.yaml   C4  fixed question   <- the control
config/winoground/reason_ica_gpt-4o.yaml               C5  adaptive question
```

All three consume the same frozen `winoground` descriptions, so they pair on the
same instances. Per plan §E2/E3, **C5 may only be called *adaptive* if it beats
C4**; otherwise ICA is an additional-inspection effect. Run C4 before making any
adaptivity claim.
