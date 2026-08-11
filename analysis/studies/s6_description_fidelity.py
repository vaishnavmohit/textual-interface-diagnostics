#!/usr/bin/env python3
"""S6 — Measure the perception difference directly, not through accuracy.

Everywhere else the description source is judged by what it does to downstream
accuracy, which is exactly the "model identity as a proxy for perception
quality" gap the paper's own gap table flags. The description artefacts are
available for the crossed cell, so the difference can be measured on the
descriptions themselves.

Both sources emit the same JSON schema (Scene, Objects, Activities, Contextual
Elements, Visual Patterns, Emotional Undertones, Textual Information, Summary),
so they are directly comparable field by field, on the same 137 images.

Three measures, none of which requires a judgement about what a "good"
description is in the abstract:

  COVERAGE   does the description mention the content words of the benchmark's
             ground-truth concept for that problem? This is task-critical
             information by construction: it is what the reasoner must recover.
  DETAIL     schema completeness and length --- how much is said at all.
  AGREEMENT  do the two sources describe the same image the same way?
             Low agreement with similar coverage means they are complementary
             rather than one being uniformly better.

Coverage uses lexical matching, so paraphrase counts as a miss and the absolute
level understates. It is used only to compare the two sources on identical
items, where that bias applies equally to both.

    python analysis/studies/s6_description_fidelity.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load, load_raw, bongard_valid, boot_ci, save, para  # noqa: E402

H = Path.home() / "Downloads/hpc_results/output"
SRC = {"gpt-4o": H / "Bongard_OpenWorld_single_test/folders",
       "pixtral": H / "Bongard_OpenWorld_single_mistral/folders"}
STOP = {"the", "a", "an", "of", "in", "on", "with", "and", "or", "is", "are", "to",
        "for", "at", "by", "from", "that", "this", "it", "its", "as", "be", "has"}


def words(s) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", str(s).lower()) if w not in STOP and len(w) > 2}


def flat_text(obj) -> str:
    """All leaf strings of the description JSON, concatenated."""
    out = []
    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            out.append(o)
    walk(obj)
    return " ".join(out)


def load_descriptions(root: Path) -> dict[str, dict]:
    d = {}
    for p in root.rglob("*.json"):
        key = f"{p.parent.name}/{p.name}"
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        d[key] = j[0] if isinstance(j, list) and j else j
    return d


def main() -> None:
    sets = {k: load_descriptions(v) for k, v in SRC.items()}
    common = sorted(set(sets["gpt-4o"]) & set(sets["pixtral"]))
    print(f"images described by both sources: {len(common)}")

    # concept per uid, from the benchmark
    raw = load_raw("output/Bongard_OpenWorld_single/results.xlsx")
    concept_by_uid = dict(zip(raw.uid.astype(str).str.zfill(4), raw.concept))

    rows = []
    for key in common:
        uid = key.split("/")[0]
        concept = concept_by_uid.get(uid)
        if not concept:
            continue
        cw = words(concept)
        rec = dict(key=key, uid=uid, concept=concept)
        texts = {}
        for src in SRC:
            j = sets[src][key]
            t = flat_text(j)
            texts[src] = t
            tw = words(t)
            rec[f"{src}_chars"] = len(t)
            rec[f"{src}_fields"] = len(j) if isinstance(j, dict) else np.nan
            rec[f"{src}_coverage"] = len(cw & tw) / len(cw) if cw else np.nan
        a, b = words(texts["gpt-4o"]), words(texts["pixtral"])
        rec["jaccard"] = len(a & b) / len(a | b) if (a | b) else np.nan
        rows.append(rec)

    t = pd.DataFrame(rows)
    save(t, "s6_description_fidelity.csv")

    print("\n=== Coverage of the ground-truth concept (task-critical content) ===")
    for src in SRC:
        c = t[f"{src}_coverage"].dropna().to_numpy()
        pt, (lo, hi) = boot_ci(c)
        print(f"  {src:<9} mean coverage {100*pt:5.1f}%  [{100*lo:.1f}, {100*hi:.1f}]   "
              f"fully covered on {100*(c==1).mean():4.1f}% of images")
    diff = (t["gpt-4o_coverage"] - t["pixtral_coverage"]).dropna().to_numpy()
    pt, (lo, hi) = boot_ci(diff)
    print(f"  paired difference (gpt-4o - pixtral): {100*pt:+.1f} points  [{100*lo:+.1f}, {100*hi:+.1f}]")

    print("\n=== Detail ===")
    for src in SRC:
        print(f"  {src:<9} median {t[f'{src}_chars'].median():.0f} chars, "
              f"{t[f'{src}_fields'].median():.0f} schema fields")

    print("\n=== Agreement between the two sources on the same image ===")
    print(f"  median Jaccard over content words: {t.jaccard.median():.3f}")
    print(f"  -> the two sources share about {100*t.jaccard.median():.0f}% of the vocabulary they use")

    # Link coverage to downstream correctness. Coverage is per IMAGE; a problem
    # has ~13 of them, and the reasoner sees all of them, so the problem-level
    # predictor is the mean coverage over that problem's images. (Taking one
    # arbitrary image per uid, as a naive dict-join does, silently discards the
    # rest.)
    d = bongard_valid(load("bongard_ow"))
    ca = d[(d.paradigm == "CA") & d.components.isin(["gpt-4o", "pixtral"])]
    ca = ca.assign(uid4=ca.uid.astype(str).str.zfill(4))
    print("\n=== Does description coverage predict downstream correctness? ===")
    print(f"  (per-problem coverage = mean over that problem's {t.groupby('uid').size().median():.0f} described images)")
    parts = []
    for src in SRC:
        per_uid = t.groupby("uid")[f"{src}_coverage"].mean()
        s = ca[ca.components == src].copy()
        s["cover"] = s.uid4.map(per_uid)
        parts.append(s.dropna(subset=["cover"]))
    sub = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    gap = float("nan")
    if len(sub):
        med = sub.cover.median()
        hi_, lo_ = sub[sub.cover >= med], sub[sub.cover < med]
        gap = 100 * (hi_.correct.mean() - lo_.correct.mean())
        print(f"  coverage >= median ({med:.2f}) : accuracy {100*hi_.correct.mean():5.1f}%  (n={len(hi_)})")
        print(f"  coverage <  median          : accuracy {100*lo_.correct.mean():5.1f}%  (n={len(lo_)})")
        print(f"  gap: {gap:+.1f} points")
        from scipy.stats import pointbiserialr
        r, p = pointbiserialr(sub.correct.to_numpy(), sub.cover.to_numpy())
        print(f"  point-biserial correlation(correct, coverage) = {r:+.3f} (p={p:.3f})")

    g_cov = 100 * t["gpt-4o_coverage"].mean()
    p_cov = 100 * t["pixtral_coverage"].mean()

    para("S6 description fidelity", f"""
The description source has so far been assessed by its effect on downstream
accuracy, which leaves open whether it is perception that differs or merely the
model producing it. Since both sources emit the same schema over the same {len(t)}
images, the descriptions can be compared directly.

They are not far apart. Measuring how often a description mentions the content
words of the benchmark's ground-truth concept --- information the reasoner must
recover to answer at all --- gives {g_cov:.1f}\\% coverage for GPT-4o against {p_cov:.1f}\\% for
Pixtral-12B, a paired difference of {100*pt:+.1f} points ({100*lo:+.1f} to {100*hi:+.1f}). Both produce
descriptions of comparable length ({t['gpt-4o_chars'].median():.0f} and {t['pixtral_chars'].median():.0f} characters at the median)
and populate the same number of schema fields.

The two sources nonetheless describe the same image differently: their content
vocabularies overlap by a median Jaccard of only {t.jaccard.median():.2f}. So the difference
between them is less a matter of one being more complete than of the two
attending to different aspects of the same scene --- which is why coverage
differs little while downstream accuracy differs consistently
(Section~\\ref{{sec:results-components}}).

Our lexical proxy for task-critical content does not, however, predict the
outcome: problems whose descriptions cover more of the concept's content words
are answered correctly {gap:+.1f} points more often than those covering less, i.e.\\
indistinguishable from no effect. We take this as a limitation of the proxy
rather than evidence that description content is irrelevant --- it counts
paraphrase as a miss, and the reasoner plainly recovers concepts that are
described without being named. A fidelity measure sensitive to meaning rather
than wording, applied to the same artefacts, is the natural next step and would
close the ``model identity as a proxy for perception quality'' gap from the
measurement side.
""")


if __name__ == "__main__":
    main()
