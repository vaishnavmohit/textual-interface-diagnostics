#!/usr/bin/env python3
"""CA - DVRL changes two things at once. This separates them.

Section 6 reports componential analysis against direct visual reasoning and
finds the advantage gone -- reversed, on the newest models. But that contrast
moves the rule and the percept out of the model's head simultaneously, and the
third paradigm sits exactly between them:

    DVRL   thirteen images in one context; rule and percept both implicit
    DRL    the model states the rule in words, then applies it -- still sees pixels
    CA     each image is described in text first; a text reasoner induces the rule

So DRL - DVRL isolates externalising the RULE, CA - DRL isolates externalising
PERCEPTION, and the two sum to the reported CA - DVRL. If the deficit sits
entirely in one step, that says what verbalisation actually costs, and the
decomposition is a much stronger claim than the pooled contrast.

It does sit in one step. On Gemini 3 Flash the rule step is free (-0.66, CI
spanning zero) and the perception step carries the whole deficit (-4.31,
d' -0.32). Splitting further, the loss tracks the ACTION dimension of the
benchmark's partition and not the OBJECT dimension -- consistent with the
description channel transmitting the object noun at parity while the verb
leaks, which the schema-elicitation measurement found independently.

A caution carried from Section 4: the sosa/soua/uosa/uoua designations are
defined against the training set of the supervised few-shot learners the
benchmark was built for. Nothing here is trained, so "seen"/"unseen"
characterises no system in this study. The claim is only that the partition's
action dimension separates the deficit and its object dimension does not.

    python analysis/studies/s7c_paradigm_ladder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RNG, bongard_valid, load, para, save  # noqa: E402

GEN = Path(__file__).resolve().parents[2] / "results" / "generation"
SPLITS = ["sosa", "soua", "uosa", "uoua"]
POS = "cat_2"
D_MIN = 0.6
N_BOOT = 10000

# The four splits cross two dimensions. Naming them by dimension rather than by
# "seen"/"unseen" keeps the caution above visible at the point of use.
DIMS = {"action-A": ["sosa", "uosa"], "action-B": ["soua", "uoua"],
        "object-A": ["sosa", "soua"], "object-B": ["uosa", "uoua"]}

CONTRASTS = [("DVRL", "DRL", "rule"), ("DRL", "CA", "perception"),
             ("DVRL", "CA", "both")]


def norm_lab(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip().str.lower()
            .replace({"pos": POS, "positive": POS, "neg": "cat_1", "negative": "cat_1"}))


def newgen(model: str, parad: str) -> pd.DataFrame:
    """A newer-generation cell from the frozen per-run spreadsheets."""
    frames = []
    for sp in SPLITS:
        f = GEN / f"{model}_{parad.lower()}_{sp}.xlsx"
        if not f.exists():
            continue
        d = pd.read_excel(f)
        pred = "test_category_identified" if "test_category_identified" in d else "conclusion_raw"
        truth = "test_cat_label" if "test_cat_label" in d else "test_cat"
        frames.append(pd.DataFrame({
            "split": sp, "test_id": d.test_id.astype(str),
            "uid": d.uid.astype(str) if "uid" in d else d.test_id.astype(str),
            "pred": norm_lab(d[pred]), "truth": norm_lab(d[truth])}))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[out.pred.isin([POS, "cat_1"])]
    out["correct"] = (out.pred == out.truth).astype(float)
    return out


def oldgen(h: pd.DataFrame, model: str, parad: str) -> pd.DataFrame:
    """An archived cell. Rows whose prediction never parsed carry is_correct
    NaN and are dropped rather than scored."""
    s = h[h.reasoner_model.eq(model) & h.paradigm.eq(parad)].drop_duplicates(["split", "test_id"])
    out = pd.DataFrame({"split": s.split, "test_id": s.test_id.astype(str),
                        "uid": s.uid.astype(str), "pred": s.pred_raw,
                        "truth": s.label_true, "correct": s.correct.astype(float)})
    return out.dropna(subset=["correct"])


def sdt(d: pd.DataFrame) -> tuple[float, float, float]:
    """(d', criterion c, false-alarm rate) with a log-linear correction."""
    pos, neg = d.truth == POS, d.truth != POS
    hit = ((d.pred[pos] == POS).sum() + 0.5) / (pos.sum() + 1)
    fa = ((d.pred[neg] == POS).sum() + 0.5) / (neg.sum() + 1)
    zh, zf = norm.ppf(hit), norm.ppf(fa)
    return float(zh - zf), float(-(zh + zf) / 2), float(fa)


def contrast(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = N_BOOT):
    """(b - a) paired on common (split, test_id), cluster-bootstrapped on uid."""
    ka, kb = a.set_index(["split", "test_id"]), b.set_index(["split", "test_id"])
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
    # two-sided bootstrap p, floored at the resolution of the resample count
    p = max(2 * min((boots <= 0).mean(), (boots >= 0).mean()), 1 / n_boot)
    return dict(n=len(common), delta=100 * float(diff.mean()),
                lo=100 * float(lo), hi=100 * float(hi), p=float(p),
                d_shift=sdt(pb)[0] - sdt(pa)[0])


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj.tolist()


def main() -> None:
    h = bongard_valid(load("bongard_hoi"))
    models = [
        ("GPT-4o", lambda p: oldgen(h, "gpt-4o-2024-08-06", p)),
        ("Gemini 2.0", lambda p: oldgen(h, "gemini-2.0-flash-exp", p)),
        ("GPT-5.1", lambda p: newgen("gpt-5.1", p)),
        ("Gemini 3 Flash", lambda p: newgen("gemini-3-flash-preview", p)),
        ("Gemini 3.5 Flash Lite", lambda p: newgen("gemini-3.5-flash-lite", p)),
    ]
    cells: dict[tuple[str, str], pd.DataFrame] = {}
    for name, get in models:
        for par in ("DVRL", "DRL", "CA"):
            d = get(par)
            if len(d) >= 50:
                cells[(name, par)] = d

    print("=== every cell, four splits pooled ===")
    print(f"  {'model':<22}{'parad':<6}{'n':>6}{'acc':>8}{'d-prime':>9}{'crit c':>9}{'FA':>8}")
    rows_cell = []
    for (name, par), d in cells.items():
        dp, c, fa = sdt(d)
        flag = "" if dp >= D_MIN else "   NOT A MEASUREMENT"
        print(f"  {name:<22}{par:<6}{len(d):>6}{100*d.correct.mean():>8.2f}"
              f"{dp:>9.2f}{c:>+9.2f}{100*fa:>7.1f}%{flag}")
        rows_cell.append(dict(model=name, paradigm=par, n=len(d),
                              acc=round(100 * d.correct.mean(), 2), d_prime=round(dp, 2),
                              criterion=round(c, 2), fa_rate=round(100 * fa, 1),
                              measurable=bool(dp >= D_MIN)))
    save(pd.DataFrame(rows_cell), "s7c_cells.csv")

    print("\n=== the ladder, pooled (Holm across the 3 contrasts within a model) ===")
    print("    rule       DRL - DVRL   make the rule explicit, pixels still visible")
    print("    perception CA  - DRL    replace the pixels with text")
    print("    both       CA  - DVRL   the sum, and what Section 6 reports\n")
    print(f"  {'model':<22}{'step':<11}{'n':>6}{'delta':>9}{'95% CI':>20}{'p_holm':>9}{'dprime':>9}")
    rows = []
    for name, _ in models:
        recs = []
        for a, b, step in CONTRASTS:
            if (name, a) not in cells or (name, b) not in cells:
                continue
            r = contrast(cells[(name, a)], cells[(name, b)])
            if r:
                recs.append(dict(step=step, contrast=f"{b} - {a}", **r))
        if not recs:
            continue
        for rec, padj in zip(recs, holm([r["p"] for r in recs])):
            star = "*" if padj < 0.05 else " "
            print(f"  {name:<22}{rec['step']:<11}{rec['n']:>6}{rec['delta']:>+9.2f}"
                  f"   [{rec['lo']:>+6.2f},{rec['hi']:>+7.2f}]{padj:>9.3f}{star}"
                  f"{rec['d_shift']:>+9.2f}")
            rows.append(dict(model=name, scope="pooled", group="all", p_holm=padj, **rec))
        print()

    print("=== does it add up?  (CA-DVRL) vs (DRL-DVRL) + (CA-DRL) ===")
    print("  each contrast pairs on its own common set, so small residuals are expected\n")
    r = pd.DataFrame(rows)
    for name in r.model.unique():
        g = r[(r.model == name) & (r.scope == "pooled")].set_index("step")
        if not {"rule", "perception", "both"} <= set(g.index):
            continue
        part = g.loc["rule", "delta"] + g.loc["perception", "delta"]
        print(f"  {name:<22}rule {g.loc['rule','delta']:+6.2f} + perception "
              f"{g.loc['perception','delta']:+6.2f} = {part:+6.2f}   measured "
              f"{g.loc['both','delta']:+6.2f}   (residual {g.loc['both','delta']-part:+.2f})")

    print("\n=== which dimension of the partition carries the loss? ===")
    print("  the four splits cross an action and an object dimension; if the")
    print("  description channel loses relations, only the action one should move\n")
    print(f"  {'model':<22}{'group':<10}{'n':>6}{'perception':>11}{'95% CI':>20}")
    for name, _ in models:
        if (name, "DRL") not in cells or (name, "CA") not in cells:
            continue
        drl, ca = cells[(name, "DRL")], cells[(name, "CA")]
        for grp, sps in DIMS.items():
            rr = contrast(drl[drl.split.isin(sps)], ca[ca.split.isin(sps)])
            if not rr:
                continue
            print(f"  {name:<22}{grp:<10}{rr['n']:>6}{rr['delta']:>+11.2f}"
                  f"   [{rr['lo']:>+6.2f},{rr['hi']:>+7.2f}]")
            rows.append(dict(model=name, scope="dimension", group=grp, step="perception",
                             contrast="CA - DRL", p_holm=np.nan, **rr))
        print()

    print("=== what orders the perception step? ===")
    print("  index of native visual sensitivity: d' of DRL, the best pixel-based")
    print("  condition that passes the screen for every model (DVRL fails it twice)\n")
    order = []
    for name, _ in models:
        if (name, "DRL") not in cells:
            continue
        step = r[(r.model == name) & (r.scope == "pooled") & (r.step == "perception")]
        if step.empty:
            continue
        order.append(dict(model=name, drl_dprime=round(sdt(cells[(name, "DRL")])[0], 2),
                          perception=round(float(step.delta.iloc[0]), 2)))
    o = pd.DataFrame(order).sort_values("drl_dprime")
    print(f"  {'model':<22}{'DRL d-prime':>12}{'perception step':>17}")
    for _, x in o.iterrows():
        print(f"  {x.model:<22}{x.drl_dprime:>12.2f}{x.perception:>+17.2f}")
    if len(o) >= 4:
        from scipy.stats import pearsonr, spearmanr
        pr, pp = pearsonr(o.drl_dprime, o.perception)
        sr, sp_ = spearmanr(o.drl_dprime, o.perception)
        print(f"\n  Pearson r = {pr:+.3f} (p = {pp:.4f})   "
              f"Spearman rho = {sr:+.2f} (p = {sp_:.4f}), n = {len(o)} models")
        print("  NOTE: five models is a descriptive regularity, not an estimated law.")
        print("  Report the ordering; do not quote a threshold from the fit.")
    save(o, "s7c_perception_vs_sensitivity.csv")

    save(pd.DataFrame(rows), "s7c_paradigm_ladder.csv")

    g3 = r[(r.model == "Gemini 3 Flash") & (r.scope == "pooled")].set_index("step")
    para("Paradigm ladder", f"""
The reported contrast moves two things at once. Componential analysis replaces
the model's implicit rule with a stated one AND its pixels with text, while
rule verbalization changes only the first. Separating them localizes the loss:
on Gemini 3 Flash, stating the rule is free ({g3.loc['rule','delta']:+.2f},
95% CI [{g3.loc['rule','lo']:+.2f}, {g3.loc['rule','hi']:+.2f}]), and the whole
deficit is incurred when the images are replaced by descriptions
({g3.loc['perception','delta']:+.2f} [{g3.loc['perception','lo']:+.2f},
{g3.loc['perception','hi']:+.2f}], d' {g3.loc['perception','d_shift']:+.2f}).
The cost is not verbal reasoning; it is verbal perception, and the model
demonstrates in the same table that it can articulate a rule in words without
penalty.
""")


if __name__ == "__main__":
    main()
