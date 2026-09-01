# Textual Interface Diagnostics

Code and frozen evaluation outputs for:

> **Diagnosing Textual Interfaces Between Perception and Reasoning in Multimodal Models**
> Mohit Vaishnav and Tanel Tammet

This repository accompanies a manuscript submitted to the *International
Journal of Computer Vision* special issue on Multimodal Large Language Models
for Unified Comprehension and Generation. It studies when a textual interface
between perception and reasoning helps, and when the tested artifacts fail to
retain evidence useful to a joint visual workflow, across Bongard-OpenWorld,
Bongard-HOI, and Winoground.

## What this release contains

- `src/pri/`: maintained experiment pipeline for unified, decomposed, staged,
  and interactive conditions.
- `config/`: portable experiment configurations. Set `DATA_DIR` and
  `OUTPUT_DIR`; benchmark images are not redistributed.
- `analysis/`: scripts used to derive the reported statistics and figures.
- `results/`: frozen per-sample predictions, per-run workbooks, and derived
  analysis tables used in the manuscript.
- `provenance/`: the experiment registry and audit notes.
- `scripts/verify_release.py`: a fast integrity and import check.

The frozen outputs are the reproducibility record for proprietary models,
whose responses cannot be regenerated bit-for-bit. Re-running an API model may
therefore yield slightly different responses even with the same settings.

## Install

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[analysis]"
cp .env.example .env
```

Set `DATA_DIR`, `OUTPUT_DIR`, and only the API keys needed for the selected
backend in `.env`.

## Verify the frozen release

```bash
python scripts/verify_release.py
```

The check imports the maintained package, confirms the three canonical
per-sample files and their row counts, validates key columns, checks the
submission-stage cohort-audit files, and scans for absolute paths from the
authors' compute environment.

## Reproduce analyses

The primary statistics can be regenerated without benchmark images or API
access:

```bash
python analysis/t1_statistics.py
python analysis/make_paper_figures.py
```

Study-specific analyses are in `analysis/studies/`. The checked-in derived
tables under `results/stats/` and `results/studies/` provide reference outputs.

## Run an experiment

Experiments use a two-stage interface when descriptions are frozen:

```bash
python main.py --config config/bongard_ow/describe_ca_pixtral-12b.yaml
python main.py --config config/bongard_ow/reason_ca_qwen2.5-14b.yaml
```

The first stage writes content-hashed descriptions. Reasoner swaps reuse that
same artifact, allowing the perception output to remain fixed. See
`config/README.md` for the available conditions and configuration overrides.

For the Bongard workflows, direct and decomposed conditions process the same
image set. Their different call counts describe execution topology, not a
corresponding ratio of visual-token volume. The decomposed path adds request
and description-generation overhead, but the frozen descriptions can be reused
by multiple downstream reasoners or tasks without re-encoding the images.
Interactive runs retain this reusable textual state and selectively return to
the original image when the reasoner requests missing evidence.

## Data and licensing

No benchmark images are included. Obtain Bongard-OpenWorld, Bongard-HOI/HAKE,
and Winoground from their original distributors and follow their respective
licenses. The software in this repository is released under the MIT License.
The frozen prediction files are research outputs; third-party dataset content
remains governed by its original terms.

## Citation

Citation metadata is provided in `CITATION.cff`. A versioned archival DOI will
be added to the manuscript and this README after the public release is deposited.
