# pri — perception–reasoning interfaces (natural-image benchmarks)

Runs LLMs/VLMs over multi-image concept-induction and compositional-matching
benchmarks — **Bongard-OpenWorld, Bongard-HOI, Winoground** — under a spectrum
of perception–reasoning interface conditions.

Architecture ported from `vaishnavmohit/human-machine-reasoning`
(see `docs/DESIGN_NOTES_from_hmr.md`). One parameterized pipeline: the paradigm
is a config parameter, not a module.

## Paradigms and interface conditions (config `context`)

| context | paper name | what perception sees |
|---|---|---|
| `dvrl` | DVRL | all 13 images, end-to-end VLM |
| `drl` | DRL | all 13 images, verbalize-rule-then-apply |
| `ca` | CA / C2 | per-image structured description, task-blind |
| `ca_flat` | C1 | per-image free-form description, task-blind |
| `ca_taskaware` | C3 | structured description + the task |
| `ca_fixed_reinspect` | C4 | C2 + one preregistered extra question |
| `ica` | ICA / C5 | C2 + one reasoner-generated question |

The C1–C5 ladder (E2) is thus a config sweep over `context` on one code path,
budget-matched by construction.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[analysis]"
cp .env.example .env            # fill in keys + DATA_DIR/OUTPUT_DIR
python main.py --config config/bongard_ow/ca_qwen25_14b.yaml
```

## Layout

```
src/pri/
  io_utils.py     image encoding, seeding            (ported)
  schemas.py      pydantic structured-output schemas
  llm.py          unified backend client (OpenAI/Gemini/Mistral/Ollama)  (ported+adapted)
  batch.py        OpenAI/Gemini batch API            (ported)
  prompts.py      per-condition prompts (natural-image benchmarks)
  pipeline.py     the one parameterized run loop -> results.xlsx
  datasets/       per-benchmark loaders (bongard_ow, bongard_hoi, winoground)
```

Outputs `results.xlsx` in the same schema the gather/stats tooling already reads
(`code/eval/gather_all_results.py`), plus token/duration columns for the cost
frontier (E5).
