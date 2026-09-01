#!/usr/bin/env python3
"""Publication figures and LaTeX tables from the stats outputs.

Consumes accuracy_with_ci.csv (and optionally paired_cluster_tests.csv) from
statistics.py and writes, under <out>:
  figures/<experiment>_ladder.png   accuracy by condition (bars + 95% CI), per model
  figures/*_caption.txt              a caption stub per figure
  tables/accuracy.tex                the main results table (acc [CI])
  tables/paired_cluster_tests.tex    the contrast table, if present

Styling follows hmr's analysis_pipeline.py (whitegrid, 300 dpi, titles removed —
captions live in LaTeX). No result values are hard-coded; everything is read
from the stats CSVs.

Usage:
    python analysis/make_figures.py --stats ${OUTPUT_DIR}/stats --out ${OUTPUT_DIR}/report
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # headless: render to files
import matplotlib.pyplot as plt           # noqa: E402
import pandas as pd                       # noqa: E402

plt.rcParams.update({"font.size": 12, "figure.dpi": 300, "savefig.dpi": 300})

# stable ordering + labels for the interface ladder, keyed by CONTEXT (the part
# of the program name before "_temp_"), so it is robust to temperature formatting
# (a config with temperature: 0 yields "..._temp_0", 0.0 yields "..._temp_0.0").
# Order matters: bar order within a figure follows this dict, so each ladder is
# listed in contrast order (C1->C5). Keys are the `context` each runner writes.
LADDER = {
    # Bongard-OW: C2-C1 = structure, C3-C2 = task conditioning, C4-C3 = extra look.
    "ca_flat": "C1 flat",
    "ca": "C2 CA",
    "ca_taskaware": "C3 task-aware",
    "ca_fixed_reinspect": "C4 fixed",
    "dvrl": "DVRL",
    "drl": "DRL",
    # Winoground: same shape. "ica" is Wino C5 (pri.winoground.run_interactive
    # writes context "ica"), so it must sort AFTER the C4 control — the C5-C4 gap
    # is what licenses the word "adaptive".
    "winoground": "Wino C2 CA",
    "winoground_fixed_reinspect": "Wino C4 fixed",
    "ica": "Wino C5 ICA",
    "winoground_dvrl": "Wino DVRL",
}


def _context_of(program_name: str) -> str:
    """Strip the trailing _temp_<t>[_ablation] to get the context key."""
    import re
    return re.sub(r"_temp_[0-9.]+.*$", "", str(program_name))


def _metric_cols(acc: pd.DataFrame):
    for m in ("is_correct", "group_score", "text_score", "image_score"):
        if m in acc.columns and acc[m].notna().any():
            return m
    return None


def ladder_figures(acc: pd.DataFrame, out: Path):
    figs = out / "figures"; figs.mkdir(parents=True, exist_ok=True)
    metric = _metric_cols(acc)
    if metric is None or "program_name" not in acc.columns:
        return
    key = "experiment" if "experiment" in acc.columns else None
    groups = acc.groupby(key) if key else [("all", acc)]
    for exp, g in groups:
        sub = g[g[metric].notna()].copy()
        if sub.empty:
            continue
        sub["ctx"] = sub["program_name"].map(_context_of)
        sub["order"] = sub["ctx"].map(lambda c: list(LADDER).index(c) if c in LADDER else 99)
        sub["label"] = sub["ctx"].map(lambda c: LADDER.get(c, c))
        sub = sub.sort_values(["model_name", "order"])
        models = list(sub["model_name"].unique())
        fig, ax = plt.subplots(figsize=(max(6, 1.4 * sub["label"].nunique()), 4))
        width = 0.8 / max(1, len(models))
        labels = list(dict.fromkeys(sub["label"]))
        x = range(len(labels))
        for i, mdl in enumerate(models):
            md = sub[sub.model_name == mdl].set_index("label").reindex(labels)
            vals = md[metric].to_numpy()
            lo = md.get(f"{metric}_ci_lo", md[metric]).to_numpy()
            hi = md.get(f"{metric}_ci_hi", md[metric]).to_numpy()
            err = [vals - lo, hi - vals]
            ax.bar([xi + i * width for xi in x], vals, width, yerr=err, capsize=3, label=mdl)
        ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(f"{metric} (%)"); ax.set_ylim(0, 100)
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)
        fig.tight_layout()
        fig.savefig(figs / f"{exp}_ladder.png", bbox_inches="tight"); plt.close(fig)
        (figs / f"{exp}_ladder_caption.txt").write_text(
            f"Accuracy on {exp} by interface condition ({metric}); error bars are "
            f"95\\% cluster-bootstrap confidence intervals.")
        print(f"  figures/{exp}_ladder.png")


def _tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def accuracy_tex(acc: pd.DataFrame, out: Path):
    tables = out / "tables"; tables.mkdir(parents=True, exist_ok=True)
    metric = _metric_cols(acc)
    if metric is None:
        return
    cols = [c for c in ("experiment", "model_name", "program_name", "n") if c in acc.columns]
    lines = [r"\begin{tabular}{" + "l" * len(cols) + "c}", r"\toprule",
             " & ".join(_tex_escape(c) for c in cols) + r" & acc [95\% CI] \\", r"\midrule"]
    for _, r in acc.sort_values(cols).iterrows():
        if pd.isna(r.get(metric)):
            continue
        cell = f"{r[metric]:.1f} [{r.get(metric+'_ci_lo', float('nan')):.1f}, {r.get(metric+'_ci_hi', float('nan')):.1f}]"
        lines.append(" & ".join(_tex_escape(r[c]) for c in cols) + f" & {cell} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tables / "accuracy.tex").write_text("\n".join(lines))
    print(f"  tables/accuracy.tex ({metric})")


def paired_tex(stats_dir: Path, out: Path):
    src = stats_dir / "paired_cluster_tests.csv"
    if not src.exists():
        return
    d = pd.read_csv(src)
    if d.empty:
        return
    tables = out / "tables"; tables.mkdir(parents=True, exist_ok=True)
    keep = [c for c in ("model_name", "program_a", "program_b", "n_pairs", "b01", "b10",
                        "pvalue", "holm_p", "acc_a", "acc_b") if c in d.columns]
    lines = [r"\begin{tabular}{" + "l" * 3 + "r" * (len(keep) - 3) + "}", r"\toprule",
             " & ".join(_tex_escape(c) for c in keep) + r" \\", r"\midrule"]
    for _, r in d.iterrows():
        cells = []
        for c in keep:
            v = r[c]
            cells.append(f"{v:.3g}" if isinstance(v, float) else _tex_escape(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tables / "paired_cluster_tests.tex").write_text("\n".join(lines))
    print("  tables/paired_cluster_tests.tex")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", required=True, help="dir with accuracy_with_ci.csv (+ paired_cluster_tests.csv)")
    ap.add_argument("--out", required=True, help="output dir for figures/ and tables/")
    args = ap.parse_args()
    stats_dir = Path(args.stats).expanduser(); out = Path(args.out).expanduser()
    acc = pd.read_csv(stats_dir / "accuracy_with_ci.csv")
    ladder_figures(acc, out)
    accuracy_tex(acc, out)
    paired_tex(stats_dir, out)
    print(f"report written to {out}")


if __name__ == "__main__":
    main()
