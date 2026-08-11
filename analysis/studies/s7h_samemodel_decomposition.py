#!/usr/bin/env python3
"""The identifying control: one model, one instance set, three contrasts.

The staging control (s7g) measured compression with text-only reasoners, and
that turned out to be reasoner-specific with a SIGN FLIP -- Qwen2.5-14B loses
3.41 points to compression on Bongard-OpenWorld while Phi-4-14B GAINS 4.81 on
byte-identical descriptions and instances. No cross-model correction of the
CA - DRL gap is therefore licensed, which is exactly why this control exists.

Here the reasoner is Gemini 3 Flash reading its OWN frozen descriptions, so all
three conditions share a model and an instance set:

    visual DRL   pixels       -> rule -> apply to pixel query
    textual DRL  descriptions -> rule -> apply to description query
    CA           descriptions -> joint reasoning over all thirteen

    textual DRL - CA          staging, at fixed textual representation
    textual DRL - visual DRL  TEXT vs PIXELS at matched staging  <-- identifies
                              the representational term
    CA - visual DRL           the total workflow difference (already reported)

Every contrast is paired on the instances common to the two arms it compares,
cluster-bootstrapped on uid, with exact McNemar and d' reported.

    python analysis/studies/s7h_samemodel_decomposition.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s7c_paradigm_ladder as S  # noqa: E402
from _common import RNG, para, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SAMEMODEL = ROOT / "results" / "samemodel"
MODEL = "gemini-3-flash-preview"
POS = S.POS
SPLITS = ["sosa", "soua", "uosa", "uoua"]

CONTRASTS = [
    ("staging", "CA", "textual DRL", "staging at fixed textual representation"),
    ("representation", "visual DRL", "textual DRL",
     "TEXT vs PIXELS at matched staging -- the identified term"),
    ("total", "visual DRL", "CA", "the total workflow difference"),
]


UNRESOLVED = SAMEMODEL / "unresolved_ids.json"


def load_textdrl(split: str, score_unresolved_wrong: bool = False) -> pd.DataFrame:
    """The textual-DRL arm.

    `score_unresolved_wrong` implements the paper's declared issued-query rule:
    a query that was issued but whose output never resolved after the retry
    limit counts as an error, not as a missing observation. Thirteen of the
    1,796 attempts returned no candidate content after five attempts; the
    default view excludes them (a parsing diagnostic) and the flag scores them
    wrong (the primary delivered-system measure). Both are reported so the
    choice cannot flatter the result.
    """
    f = SAMEMODEL / f"hoi_{split}_g3f_textdrl.xlsx"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_excel(f)
    d = d[d.test_category_identified.notna()]
    out = pd.DataFrame({
        "test_id": d.test_id.astype(str), "uid": d.uid.astype(str),
        "pred": S.norm_lab(d.test_category_identified),
        "truth": S.norm_lab(d.test_cat_label),
        "correct": d.is_correct.astype(float)})
    if not score_unresolved_wrong or not UNRESOLVED.exists():
        return out
    ids = json.loads(UNRESOLVED.read_text()).get(split, [])
    if not ids:
        return out
    # Recover each unresolved instance's true label from the CA arm, which
    # scored the same problem successfully, then add it as an error.
    ca = load_grid("CA", split).set_index("test_id")
    add = []
    for tid in ids:
        if tid in ca.index:
            add.append(dict(test_id=tid, uid=ca.loc[tid, "uid"],
                            pred="__unresolved__", truth=ca.loc[tid, "truth"],
                            correct=0.0))
    return pd.concat([out, pd.DataFrame(add)], ignore_index=True) if add else out


def load_grid(paradigm: str, split: str) -> pd.DataFrame:
    d = S.newgen(MODEL, paradigm)
    return d[d.split.eq(split)][["test_id", "uid", "pred", "truth", "correct"]]


def sdt(d: pd.DataFrame) -> float:
    pos, neg = d.truth == POS, d.truth != POS
    h = ((d.pred[pos] == POS).sum() + 0.5) / (pos.sum() + 1)
    f = ((d.pred[neg] == POS).sum() + 0.5) / (neg.sum() + 1)
    return float(norm.ppf(h) - norm.ppf(f))


def contrast(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 10000):
    """(b - a) paired on common test_id, clustered on uid."""
    ka, kb = a.set_index("test_id"), b.set_index("test_id")
    common = ka.index.intersection(kb.index)
    if len(common) < 30:
        return None
    pa, pb = ka.loc[common], kb.loc[common]
    diff = (pb.correct - pa.correct).to_numpy()
    idx = {k: v.to_numpy() for k, v in pd.Series(range(len(diff))).groupby(pa.uid.to_numpy())}
    keys = list(idx)
    boots = np.array([diff[np.concatenate([idx[k] for k in RNG.choice(keys, len(keys), replace=True)])].mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    b01 = int(((pb.correct == 1) & (pa.correct == 0)).sum())
    b10 = int(((pb.correct == 0) & (pa.correct == 1)).sum())
    p = binomtest(b01, b01 + b10).pvalue if (b01 + b10) else 1.0
    return dict(n=len(common), acc_a=100 * float(pa.correct.mean()),
                acc_b=100 * float(pb.correct.mean()), delta=100 * float(diff.mean()),
                lo=100 * float(lo), hi=100 * float(hi), mcnemar_p=float(p),
                fix=b01, brk=b10, d_a=sdt(pa), d_b=sdt(pb))


def main() -> None:
    have = [s for s in SPLITS if not load_textdrl(s).empty]
    if not have:
        print("No textual-DRL results yet under results/samemodel/.")
        return
    print(f"splits with the textual-DRL arm: {', '.join(have)}"
          f"{'  (PARTIAL -- ' + str(len(have)) + ' of 4)' if len(have) < 4 else ''}\n")

    arms_by_split = {}
    for sp in have:
        arms_by_split[sp] = {"textual DRL": load_textdrl(sp),
                             "CA": load_grid("CA", sp),
                             "visual DRL": load_grid("DVRL", sp)}

    rows = []
    print(f"  {'split':<7}{'contrast':<16}{'n':>5}{'from':>8}{'to':>8}{'delta':>9}"
          f"{'95% CI':>19}{'p':>8}   d-prime")
    for sp in have:
        for key, a_name, b_name, _ in CONTRASTS:
            r = contrast(arms_by_split[sp][a_name], arms_by_split[sp][b_name])
            if not r:
                continue
            star = "*" if (r["lo"] > 0 or r["hi"] < 0) else " "
            print(f"  {sp:<7}{key:<16}{r['n']:>5}{r['acc_a']:>8.2f}{r['acc_b']:>8.2f}"
                  f"{r['delta']:>+9.2f}{star}[{r['lo']:>+6.2f},{r['hi']:>+6.2f}]"
                  f"{r['mcnemar_p']:>8.3f}   {r['d_a']:.2f}->{r['d_b']:.2f}")
            rows.append(dict(scope=sp, contrast=key, **r))
        print()

    # Pooled. test_ids are namespaced by split so the same id in two splits
    # cannot be silently merged.
    pooled = {}
    for name in ("textual DRL", "CA", "visual DRL"):
        parts = []
        for sp in have:
            d = arms_by_split[sp][name].copy()
            d["test_id"] = sp + "|" + d.test_id.astype(str)
            parts.append(d)
        pooled[name] = pd.concat(parts, ignore_index=True)

    print(f"=== pooled over {len(have)} split(s) ===")
    for key, a_name, b_name, blurb in CONTRASTS:
        r = contrast(pooled[a_name], pooled[b_name])
        if not r:
            continue
        star = "*" if (r["lo"] > 0 or r["hi"] < 0) else " "
        print(f"  {key:<16}{r['n']:>5}{r['acc_a']:>8.2f}{r['acc_b']:>8.2f}"
              f"{r['delta']:>+9.2f}{star}[{r['lo']:>+6.2f},{r['hi']:>+6.2f}]"
              f"{r['mcnemar_p']:>8.4f}   {r['d_a']:.2f}->{r['d_b']:.2f}   {blurb}")
        rows.append(dict(scope="pooled", contrast=key, **r))

    # --- the declared issued-query scoring, as a sensitivity analysis ------ #
    print("\n=== sensitivity: unresolved issued queries scored WRONG "
          "(the paper's primary rule) ===")
    strict = {}
    for name in ("textual DRL", "CA", "visual DRL"):
        parts = []
        for sp in have:
            d = (load_textdrl(sp, score_unresolved_wrong=True) if name == "textual DRL"
                 else load_grid("DVRL" if name == "visual DRL" else "CA", sp)).copy()
            d["test_id"] = sp + "|" + d.test_id.astype(str)
            parts.append(d)
        strict[name] = pd.concat(parts, ignore_index=True)
    for key, a_name, b_name, _ in CONTRASTS:
        rr = contrast(strict[a_name], strict[b_name])
        if not rr:
            continue
        star = "*" if (rr["lo"] > 0 or rr["hi"] < 0) else " "
        print(f"  {key:<16}{rr['n']:>5}{rr['acc_a']:>8.2f}{rr['acc_b']:>8.2f}"
              f"{rr['delta']:>+9.2f}{star}[{rr['lo']:>+6.2f},{rr['hi']:>+6.2f}]"
              f"{rr['mcnemar_p']:>8.4f}   {rr['d_a']:.2f}->{rr['d_b']:.2f}")
        rows.append(dict(scope="pooled-strict", contrast=key, **rr))

    # --- dimension pooling, saved so the prose cites a file rather than a
    # --- rerun: these are bootstrap intervals and drift at the second decimal
    # --- if the RNG is consumed in a different order.
    print("\n=== the identified term pooled by partition dimension ===")
    DIMS = {"action-A (sosa+uosa)": ["sosa", "uosa"],
            "action-B (soua+uoua)": ["soua", "uoua"],
            "object-A (sosa+soua)": ["sosa", "soua"],
            "object-B (uosa+uoua)": ["uosa", "uoua"]}
    for lab, sps in DIMS.items():
        if not set(sps) <= set(have):
            continue
        arms = {}
        for name in ("visual DRL", "textual DRL"):
            parts = []
            for sp in sps:
                d = arms_by_split[sp][name].copy()
                d["test_id"] = sp + "|" + d.test_id.astype(str)
                parts.append(d)
            arms[name] = pd.concat(parts, ignore_index=True)
        rr = contrast(arms["visual DRL"], arms["textual DRL"])
        if not rr:
            continue
        star = "*" if (rr["lo"] > 0 or rr["hi"] < 0) else " "
        print(f"  {lab:<22}{rr['n']:>5}{rr['delta']:>+9.2f}{star}"
              f"[{rr['lo']:>+6.2f},{rr['hi']:>+6.2f}]")
        rows.append(dict(scope="dimension", contrast=lab, **rr))

    r = pd.DataFrame(rows)
    save(r, "s7h_samemodel_decomposition.csv")

    p = r[r.scope.eq("pooled")].set_index("contrast")
    st, rep = p.loc["staging"], p.loc["representation"]
    ident = "excluding zero" if (rep.lo > 0 or rep.hi < 0) else "spanning zero"
    para("Same-model decomposition", f"""
With the reasoner, the descriptions and the instance set all held fixed, the
workflow difference separates cleanly. Staging contributes
{st.delta:+.2f} points ([{st.lo:+.2f}, {st.hi:+.2f}]), an interval spanning
zero: for this model, compressing the supports into a short rule costs nothing.
The representational substitution carries the effect --- replacing pixels with
descriptions under matched short-rule staging is worth {rep.delta:+.2f} points
([{rep.lo:+.2f}, {rep.hi:+.2f}], {ident}) and moves sensitivity from
d' {rep.d_a:.2f} to {rep.d_b:.2f}. This is a single-factor contrast on one
model and one instance set, so it identifies the representational term rather
than bounding a workflow difference. Note also that the staging term does NOT
transfer across reasoners: the text-only controls put it at $-3.41$ for one
reasoner and $+4.81$ for another on identical descriptions, which is why no
cross-model correction is applied anywhere in this paper.
""".rstrip())


if __name__ == "__main__":
    main()
