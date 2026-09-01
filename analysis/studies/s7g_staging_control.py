#!/usr/bin/env python3
"""Is the description-workflow deficit a staging artifact?

Section 7 attributes the CA - DRL deficit to what the descriptions carry. A
reviewer can object that CA - DRL changes three things at once: the query goes
from pixels to text, the support evidence goes from pixels to text, AND the
staging changes (DRL compresses the supports into a short rule; CA holds all
thirteen descriptions jointly). The accuracy gap could be any of the three.

This is the matched control for the third. Textual DRL reads the SAME frozen
descriptions CA reads, with the same reasoner and decoding, and differs from CA
only in staging -- supports compressed to a <20-word rule (image-DRL's own
budget), query classified against that rule alone. So

    textual-DRL  -  CA      isolates compress-to-rule vs joint context.

A null says staging is free in text exactly as it is in pixels (DRL - DVRL is
already ~0 for every screen-passing model), which leaves description content as
the only remaining account of the deficit. A non-null says joint context is
doing real work and the attribution needs qualifying -- also reportable.

Both arms are regenerated at num_ctx 16384. The 8192 used elsewhere silently
truncates the longest problems (measured: CA prompts reach ~8.9k tokens on OW
and ~11k on HOI), and it truncates CA MORE than textual-DRL because CA's single
prompt is the larger one -- which would bias the contrast toward the very result
being tested.

    python analysis/studies/s7g_staging_control.py [--root <OUTPUT_DIR>]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (RNG, cluster_randomization_pvalue, normal_ppf,
                     para, save)  # noqa: E402

POS = "pos"
D_MIN = 0.6

# (label, committed file stem, description source). The frozen result files live
# in results/staging/ so this runs from the repo on any machine.
CELLS = [
    ("Bongard-OW / Qwen2.5-14B", "bongard_ow_staging", "GPT-4o"),
    ("Bongard-OW / Qwen2.5-32B", "bongard_ow_staging_qwen2.5-32b", "GPT-4o"),
    ("Bongard-OW / Phi-4-14B", "bongard_ow_staging_phi4", "GPT-4o"),
    ("Bongard-OW / Gemma2-27B", "bongard_ow_staging_gemma2-27b", "GPT-4o"),
    ("Bongard-OW / DeepSeek-R1-14B", "bongard_ow_staging_deepseek-r1-14b", "GPT-4o"),
    ("Bongard-HOI soua / Qwen2.5-14B", "bongard_hoi_soua_staging", "Gemini-3-Flash"),
]

def load(root: Path, stem: str, arm: str) -> pd.DataFrame:
    """arm is 'ca' or 'ca_textdrl'."""
    f = root / f"{stem}_{arm}.xlsx"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_excel(f)
    d = d[d.test_category_identified.notna()]
    return pd.DataFrame({
        "test_id": d.test_id.astype(str),
        "uid": d.uid.astype(str) if "uid" in d else d.test_id.astype(str),
        "pred": d.test_category_identified.astype(str),
        "truth": d.test_cat_label.astype(str),
        "correct": d.is_correct.astype(float),
        "sha": d.description_manifest_sha256.astype(str)})


def sdt(d: pd.DataFrame) -> tuple[float, float]:
    pos, neg = d.truth == POS, d.truth != POS
    hit = ((d.pred[pos] == POS).sum() + 0.5) / (pos.sum() + 1)
    fa = ((d.pred[neg] == POS).sum() + 0.5) / (neg.sum() + 1)
    zh, zf = normal_ppf(hit), normal_ppf(fa)
    return float(zh - zf), float(-(zh + zf) / 2)


def contrast(ca: pd.DataFrame, td: pd.DataFrame, n_boot: int = 10000):
    ka, kt = ca.set_index("test_id"), td.set_index("test_id")
    common = ka.index.intersection(kt.index)
    if len(common) < 30:
        return None
    a, t = ka.loc[common], kt.loc[common]
    diff = (t.correct - a.correct).to_numpy()
    idx = {k: v.to_numpy() for k, v in pd.Series(range(len(diff))).groupby(a.uid.to_numpy())}
    keys = list(idx)
    boots = np.array([diff[np.concatenate([idx[k] for k in RNG.choice(keys, len(keys), replace=True)])].mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    b01 = int(((t.correct == 1) & (a.correct == 0)).sum())
    b10 = int(((t.correct == 0) & (a.correct == 1)).sum())
    p = cluster_randomization_pvalue(diff, a.uid.to_numpy())
    da, _ = sdt(a)
    dt, _ = sdt(t)
    return dict(n=len(common), ca_acc=100 * float(a.correct.mean()),
                td_acc=100 * float(t.correct.mean()), delta=100 * float(diff.mean()),
                lo=100 * float(lo), hi=100 * float(hi), cluster_p=float(p),
                fix=b01, brk=b10, ca_dprime=da, td_dprime=dt,
                same_artifact=bool(a.sha.iloc[0] == t.sha.iloc[0]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None,
                    help="OUTPUT_DIR holding the staging experiment dirs "
                         "(default: first of $OUTPUT_DIR, ~/pri_output, "
                         "~/Downloads/hpc_results/staging that exists)")
    args = ap.parse_args()
    if args.root is None:
        args.root = Path(__file__).resolve().parents[2] / "results" / "staging"

    rows = []
    print(f"reading from {args.root}\n")
    print(f"  {'cell':<32}{'n':>5}{'CA':>8}{'textDRL':>9}{'delta':>8}"
          f"{'95% CI':>18}{'p':>8}{'d-prime CA->tDRL':>19}")
    for label, stem, source in CELLS:
        ca = load(args.root, stem, "ca")
        td = load(args.root, stem, "ca_textdrl")
        if ca.empty or td.empty:
            print(f"  {label:<32}  -- not run yet "
                  f"({'CA missing' if ca.empty else ''}"
                  f"{' and ' if ca.empty and td.empty else ''}"
                  f"{'textDRL missing' if td.empty else ''})")
            continue
        r = contrast(ca, td)
        if r is None:
            print(f"  {label:<32}  -- too few common instances")
            continue
        flag = "" if r["same_artifact"] else "   !! DIFFERENT DESCRIPTION ARTIFACTS"
        print(f"  {label:<32}{r['n']:>5}{r['ca_acc']:>8.2f}{r['td_acc']:>9.2f}"
              f"{r['delta']:>+8.2f}   [{r['lo']:>+6.2f},{r['hi']:>+7.2f}]"
              f"{r['cluster_p']:>8.3f}   {r['ca_dprime']:.2f}->{r['td_dprime']:.2f}{flag}")
        rows.append(dict(cell=label, description_source=source, **r))

    if not rows:
        print("\nNothing to analyse yet. Run the six configs, sync the outputs, "
              "then re-run this script.")
        return

    r = pd.DataFrame(rows)
    save(r, "s7g_staging_control.csv")

    # The conclusion must follow the data, not the hypothesis that motivated the
    # run. Staging can come back free (intervals spanning zero), costly, or
    # beneficial, and each licenses a different sentence about the deficit.
    sig = r[(r.lo > 0) | (r.hi < 0)]
    costly = r[r.delta < 0]
    verdict = ("free" if sig.empty else
               "costly" if len(costly) == len(r) else "mixed")

    if verdict == "free":
        body = f"""
Staging is approximately free in text, as it already is in pixels. Across
{len(r)} cells the paired difference is {r.delta.min():+.2f} to
{r.delta.max():+.2f} points, every interval spanning zero, so the
description-workflow deficit is not a staging artifact and what the
descriptions carry remains the account that survives."""
    elif verdict == "costly":
        body = f"""
Staging is NOT free, and it runs against the deficit rather than explaining it.
Compressing the supports into a short rule costs {abs(r.delta.max()):.2f} to
{abs(r.delta.min()):.2f} points relative to holding all descriptions jointly,
over identical evidence and an identical reasoner
({len(sig)} of {len(r)} intervals exclude zero). The componential condition
therefore enjoys a staging ADVANTAGE over rule verbalization, which means the
reported CA - DRL gap UNDERSTATES the cost of replacing pixels with text: the
componential condition loses that comparison while holding a favourable
staging position. Correcting for it moves the representation cost away from
zero, not toward it."""
    else:
        body = f"""
Staging is not uniform across cells: paired differences run
{r.delta.min():+.2f} to {r.delta.max():+.2f} points and
{len(sig)} of {len(r)} intervals exclude zero. The staging term is therefore a
real component of the CA - DRL contrast and is reported alongside the
representation term rather than folded into it."""

    para("Staging control", body.strip())


if __name__ == "__main__":
    main()
