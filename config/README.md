# pri configs — run workflow

Two config types per the reproducibility design: **describe** (Stage 1,
perception → frozen descriptions) then **reason** (Stage 2, text reasoner →
`results.xlsx`). A reason run aborts if its description artifact is absent.

## Smoke run (5 samples) — do this on HPC, where the data lives

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[analysis]"
cp .env.example .env      # set DATA_DIR, OUTPUT_DIR, OLLAMA_HOST / API keys

# Stage 1: Pixtral describes 5 Bongard-OW puzzles (writes the frozen artifact)
python main.py --config config/bongard_ow/describe_ca_pixtral-12b.yaml

# Stage 2: Qwen2.5-14B reasons over those exact descriptions
python main.py --config config/bongard_ow/reason_ca_qwen2.5-14b.yaml
```

Expected: `${OUTPUT_DIR}/descriptions/pixtral:12b/structured_v1/bongard_ow/`
with a `manifest.json`, and
`${OUTPUT_DIR}/bongard_ow/qwen2.5:14b/ca_temp_0/results.xlsx` with per-sample
`is_correct` and the provenance columns.

## Reasoner swap (the A.6 control), correctly

Copy `reason_ca_qwen2.5-14b.yaml`, change **only** `reasoner_model`, run again.
Both results.xlsx will carry the **same** `description_manifest_sha256` — that is
the machine-checkable proof the swap was controlled. Do **not** re-run describe
between reasoners.

## Perception swap (the A.4 control)

Run a second describe config with a different `perception_model` (e.g. GPT-4o),
then a reason config pointing `perception_model` at it, reasoner held fixed. The
two runs will have different `description_manifest_sha256`, attributing any
accuracy change to perception.

## Parity check before trusting new conditions (build increment 8)

Scale `num_samples` to 500 and run `context: ca` with a reasoner + perception
pair that exists in the COLM tables (e.g. Pixtral desc → a 14B reasoner). The
accuracy should match the inherited CA number for that pair within run noise. If
it does, the port is faithful and new conditions (C1/C3/C4, increment 11) can be
trusted. Record the check in `provenance/RUN_REGISTRY.md`.

## The E2 interface ladder (C1–C5) as context sweeps

| Cond. | How to run |
|---|---|
| C1 ca_flat | `describe context=ca_flat` → `pipeline.run context=ca_flat` |
| C2 ca (CA) | `describe context=ca` → `pipeline.run context=ca` |
| C3 ca_taskaware | `describe context=ca_taskaware` → `pipeline.run context=ca_taskaware` |
| C4 ca_fixed_reinspect | `describe context=ca` → `pipeline.run_fixed_reinspect` (fixed query re-inspection) |
| C5 ica | `describe context=ca` → (Winoground) `winoground.run_interactive` |

C1/C2/C3 differ only in the perception artifact (distinct schema_version each);
C4/C5 share the C2 descriptions and add one re-inspection call — C4 fixed
(budget-matched control), C5 adaptive. This makes the ladder contrasts
(C2−C1 structure, C3−C2 task-conditioning, C4−C3 extra-look, C5−C4 adaptivity)
clean by construction.
