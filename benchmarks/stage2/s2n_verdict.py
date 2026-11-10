"""S2N's verdict.  It chooses nothing.  Judged on points, chat and top-1, never on a loss.  Deltas against ref, paired bootstrap over items (stage 1's).

e2e_row:      stage 1's label and chat gate, reported.
Training:     e2e_row - ks_stk4: E2E PAYS (lower > 0 and margin > GAP), E2E HURTS (upper < 0), NOT SHOWN.  A PAYS with margin <= 2 GAP needs the second data
              order: SECOND ORDER NEEDED until e2e_row2 is in; then PAYS only if e2e_row2 - ks_stk4 PAYS too.
Organs' k = 4 loss:  e2e_row - (bal4 + ks_org), ks_org at k = 8: WINS BACK THE ORGANS' k = 4 LOSS (lower > 0), DOES NOT WIN IT BACK (upper < 0), NOT SHOWN.
              ks_stk4's own is -4.41.  Beside it, generation at k = 4 (nq, tqa, gsm8k, chat), e2e_row - ks_stk4: GENERATION GAINS / LOSES / NOT SHOWN.
k = 8:        arc + mmlu, e2e_row - ks_stk4: HURTS AT k = 8 (upper < 0), else k = 8 HOLDS.
Documents:    dBPB per document, e2e_row - ks_stk4, paired over documents: BETTER ON DOCUMENTS (upper < 0), WORSE ON DOCUMENTS (lower > 0), NOT SHOWN.
Granularity:  e2e_grp - e2e_row: GROUPS PAY (lower > 0 and margin > GAP), else NOT SHOWN.
Trivial:      read only if E2E PAYS.  e2e_row - glob: ROWS PAY (lower > 0, margin > GAP); A GLOBAL GAIN DOES IT if glob - ks_stk4 >= 2/3 of e2e_row - ks_stk4;
              else NOT SHOWN.
Validity:     parity at k = 8 and k = 4 mass 16, recomputed here from the numbers and s2n_train's floors, else VOID; the planted k = 4 mass 0 must fail
              it, else PARITY NOT DISCRIMINATING (said); the subtle plants are reported with their size; damage and recovery (equivariance against damage_ref), else VOID; per arm: held-out
              'acts' top-1 up by ACTS_MARGIN, every effective scale positive, the export read back and kept by the export check, the panels complete on the exported file, else VOID.

Usage: python -I benchmarks/stage2/s2n_verdict.py --selftest
       python -I benchmarks/stage2/s2n_verdict.py --scores-dir D --work W --results DIR --json F
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
sys.path.insert(0, str(HERE.parents[0] / "chatbot50"))
sys.path.insert(0, str(HERE))
from s1_verdict import CHAT_MARGIN, DEGEN_MARGIN, DRAWS, PANELS, REC, SEED, degenerate_count, docs_metrics, label, points, vec  # noqa: E402
from s2z_verdict import GAP  # noqa: E402

TRAINED = ("e2e_row", "e2e_grp", "glob", "e2e_row2")
GEN = ("nq", "tqa", "gsm8k", "chat")
K8P = ("arc", "mmlu")
DAMAGE = 0.8
DAMAGE_MULT = 1 + 0.5 * (1 / DAMAGE - 1)      # half of the way back
DAMAGE_BACK = 0.5
PARITY_TOP1, PARITY_KL = 0.95, 0.05            # s2n_train's; checked equal at run time


def training(m: float, ci: list[float]) -> str:
    if ci[0] > 0 and m > GAP:
        return "E2E PAYS"
    return "E2E HURTS" if ci[1] < 0 else "NOT SHOWN"


def second(first: str, m: float, again: str | None) -> str:
    if first != "E2E PAYS" or m > 2 * GAP:
        return first
    if again is None:
        return "SECOND ORDER NEEDED"
    return "E2E PAYS" if again == "E2E PAYS" else "NOT SHOWN"


def wins_back(ci: list[float]) -> str:
    return "WINS BACK THE ORGANS' k = 4 LOSS" if ci[0] > 0 else "DOES NOT WIN IT BACK" if ci[1] < 0 else "NOT SHOWN"


def generation(ci: list[float]) -> str:
    return "GENERATION GAINS" if ci[0] > 0 else "GENERATION LOSES" if ci[1] < 0 else "NOT SHOWN"


def k8(ci: list[float]) -> str:
    return "HURTS AT k = 8" if ci[1] < 0 else "k = 8 HOLDS"


def documents(ci: list[float]) -> str:
    return "BETTER ON DOCUMENTS" if ci[1] < 0 else "WORSE ON DOCUMENTS" if ci[0] > 0 else "NOT SHOWN"


def groups(m: float, ci: list[float]) -> str:
    return "GROUPS PAY" if ci[0] > 0 and m > GAP else "NOT SHOWN"


def trivial(rows_m: float, rows_ci: list[float], glob_gain: float, row_gain: float) -> str:
    if rows_ci[0] > 0 and rows_m > GAP:
        return "ROWS PAY"
    return "A GLOBAL GAIN DOES IT" if row_gain > 0 and glob_gain >= 2 / 3 * row_gain else "NOT SHOWN"


def parity(rec: dict) -> bool:
    return rec["top1_agreement"] >= PARITY_TOP1 and abs(rec["kl_ratio"] - 1) <= PARITY_KL


def damage(rec: dict, ref: dict, none: dict) -> bool:
    """Repair as equivariance: the damaged run and the reference (same rows, same data, from ks_stk4) land on the same weights.  The damaged run's
    multipliers over the reference's, geometric mean over the rows, at least half of the way to 1 / DAMAGE; and the damaged run ends on held-out acts
    within (1 - DAMAGE_BACK) of the damage's drop from where the reference ends."""
    r = np.asarray(rec["shexp_down_5_14_rows"], dtype=np.float64)
    q = np.asarray(ref["shexp_down_5_14_rows"], dtype=np.float64)
    if r.shape != q.shape or np.any(r <= 0) or np.any(q <= 0):
        return False
    ratio = float(np.exp(np.mean(np.log(r / q))))
    drop = none["after"]["acts"]["mean_top1"] - rec["before"]["acts"]["mean_top1"]
    gap = ref["after"]["acts"]["mean_top1"] - rec["after"]["acts"]["mean_top1"]
    return ratio >= DAMAGE_MULT and drop > 0 and gap <= (1 - DAMAGE_BACK) * drop


def doc_bpb(work: Path, arm: str) -> dict[int, float]:
    sys.path.insert(0, str(HERE.parents[0] / "chatbot50"))
    from c50_compare import byte_lengths
    import c50_a4_checks as A4
    x = np.fromfile(work / arm / "docs.pos.bin", dtype=REC)
    blen = byte_lengths(Path(A4.TOKENIZER))[x["target"]].astype(np.float64)
    return {int(d): float(-x["logp_target"][x["doc"] == d].astype(np.float64).sum() / np.log(2) / blen[x["doc"] == d].sum()) for d in np.unique(x["doc"])}


def selftest() -> int:
    acts = lambda b, a: {"before": {"acts": {"mean_top1": b}}, "after": {"acts": {"mean_top1": a}}}  # noqa: E731
    cases = [(training(2.5, [0.3, 4.7]), "E2E PAYS"), (training(1.5, [0.1, 2.9]), "NOT SHOWN"), (training(-3, [-5, -1]), "E2E HURTS"),
             (second("E2E PAYS", 2.5, None), "SECOND ORDER NEEDED"), (second("E2E PAYS", 2.5, "NOT SHOWN"), "NOT SHOWN"),
             (second("E2E PAYS", 2.5, "E2E PAYS"), "E2E PAYS"), (second("E2E PAYS", 4.0, None), "E2E PAYS"), (second("NOT SHOWN", 1.0, None), "NOT SHOWN"),
             (wins_back([0.2, 5]), "WINS BACK THE ORGANS' k = 4 LOSS"), (wins_back([-6, -0.1]), "DOES NOT WIN IT BACK"), (wins_back([-3, 2]), "NOT SHOWN"),
             (generation([0.1, 3]), "GENERATION GAINS"), (generation([-3, -0.1]), "GENERATION LOSES"), (generation([-1, 1]), "NOT SHOWN"),
             (k8([-3, -0.1]), "HURTS AT k = 8"), (k8([-3, 1]), "k = 8 HOLDS"),
             (documents([-0.03, -0.01]), "BETTER ON DOCUMENTS"), (documents([0.01, 0.02]), "WORSE ON DOCUMENTS"), (documents([-0.01, 0.01]), "NOT SHOWN"),
             (groups(2.0, [0.2, 3.8]), "GROUPS PAY"), (groups(1.0, [0.2, 1.8]), "NOT SHOWN"),
             (trivial(2.0, [0.1, 3.9], 1.0, 4.0), "ROWS PAY"), (trivial(0.5, [-1, 2], 3.0, 4.0), "A GLOBAL GAIN DOES IT"),
             (trivial(0.5, [-1, 2], 2.0, 4.0), "NOT SHOWN"),
             (parity({"top1_agreement": 0.9676, "kl_ratio": 1.002}), True), (parity({"top1_agreement": 0.646, "kl_ratio": 0.819}), False),
             (parity({"top1_agreement": 0.97, "kl_ratio": 1.08}), False),
             (damage({**acts(0.70, 0.83), "shexp_down_5_14_rows": [1.0, 1.25]}, {**acts(0.80, 0.84), "shexp_down_5_14_rows": [0.8, 1.0]},
                     acts(0.80, 0.80)), True),
             (damage({**acts(0.70, 0.78), "shexp_down_5_14_rows": [1.0, 1.25]}, {**acts(0.80, 0.84), "shexp_down_5_14_rows": [0.8, 1.0]},
                     acts(0.80, 0.80)), False),
             (damage({**acts(0.70, 0.83), "shexp_down_5_14_rows": [1.0, 1.05]}, {**acts(0.80, 0.84), "shexp_down_5_14_rows": [1.0, 1.0]},
                     acts(0.80, 0.80)), False),
             (damage({**acts(0.70, 0.83), "shexp_down_5_14_rows": [1.0, -0.2]}, {**acts(0.80, 0.84), "shexp_down_5_14_rows": [0.8, 1.0]},
                     acts(0.80, 0.80)), False),
             (damage({**acts(0.80, 0.84), "shexp_down_5_14_rows": [1.0, 1.25]}, {**acts(0.80, 0.84), "shexp_down_5_14_rows": [0.8, 1.0]},
                     acts(0.80, 0.80)), False)]
    bad = [i for i, (a, b) in enumerate(cases) if a != b]
    print("selftest", "PASS" if not bad else f"FAIL {bad}", len(cases))
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    for x in ("scores-dir", "work", "results", "json"):
        ap.add_argument(f"--{x}", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import s2n_train as T
    if (T.PARITY_TOP1, T.PARITY_KL, T.DAMAGE) != (PARITY_TOP1, PARITY_KL, DAMAGE):
        raise SystemExit("floors differ from s2n_train's")
    R = a.results
    rd = lambda n: json.loads((R / n).read_text(encoding="utf-8"))  # noqa: E731
    have = [x for x in TRAINED if (a.scores_dir / f"{x}.json").exists()]
    names = ["ref", "ks_stk4", "bal4", "ks_org", "ks_org4"] + have
    arms = {x: {p: vec(json.loads((a.scores_dir / f"{x}.json").read_text(encoding="utf-8")), p) for p in PANELS} for x in names}
    n = {p: len(arms["ref"][p]) for p in PANELS}

    def sub(v, idx, ps):
        return float(np.mean([100.0 * (v[p] if idx is None else v[p][idx[p]]).mean() for p in ps]))
    rng = np.random.default_rng(SEED)
    boot = {x: [] for x in names[1:]}
    bgen = {x: [] for x in names[1:]}
    b8 = {x: [] for x in names[1:]}
    for _ in range(DRAWS):
        idx = {p: rng.integers(0, n[p], n[p]) for p in PANELS}
        r, rg, r8 = points(arms["ref"], idx), sub(arms["ref"], idx, GEN), sub(arms["ref"], idx, K8P)
        for x in boot:
            boot[x].append(points(arms[x], idx) - r)
            bgen[x].append(sub(arms[x], idx, GEN) - rg)
            b8[x].append(sub(arms[x], idx, K8P) - r8)
    boot, bgen, b8 = ({x: np.array(v) for x, v in d.items()} for d in (boot, bgen, b8))

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    ref = points(arms["ref"])
    delta = {x: points(arms[x]) - ref for x in boot}
    passes = {x: int(arms[x]["chat"].sum()) for x in names}
    deg = {x: degenerate_count(a.work / x) for x in names if x != "ks_org"}

    par = {f: rd(f"s2n_parity_{f}.json") for f in ("k8", "k4m16", "k4m0")}
    plants = {p: rd(f"s2n_parity_k8_{p}.json") for p in ("noexc", "shexp097") if (R / f"s2n_parity_k8_{p}.json").exists()}
    parity_ok = parity(par["k8"]) and parity(par["k4m16"])
    dmg_ok = damage(rd("s2n_dry_damage.json"), rd("s2n_dry_damage_ref.json"), rd("s2n_dry_none.json"))
    run = rd("s2n_run.json")["arms"]
    valid, checks = {}, {}
    for x in have:
        tr, ex, ec = rd(f"s2n_train_{x}.json"), rd(f"s2n_export_{x}.json"), rd(f"s2n_export_check_{x}.json")
        r = run.get(x, {})
        checks[x] = {"acts": tr["acts"], "scales_positive": (tr.get("multipliers") or {}).get("nonpositive", 1) == 0,
                     "export_read_back": ex["ok"], "export_kept": ec["ok"], "model_is_export": r.get("model_sha256") == ex["out_sha256"],
                     "panels_complete": all(r.get("panels", {}).get(p, {}).get("status") == "COMPLETE" for p in r.get("panel_set", ["missing"]))}
        valid[x] = bool(parity_ok and dmg_ok and all(checks[x].values()))
    m = {x: delta[x] - delta["ks_stk4"] for x in have}
    mci = {x: ci(boot[x] - boot["ks_stk4"]) for x in have}
    out = {"parity": {f: {**r, "parity_here": parity(r)} for f, r in par.items()}, "parity_ok": parity_ok,
           "planted": "CAUGHT" if not parity(par["k4m0"]) else "PARITY NOT DISCRIMINATING",
           "subtle_plants": {p: {"top1_agreement": r["top1_agreement"], "kl_ratio": r["kl_ratio"], "caught": not parity(r)} for p, r in plants.items()},
           "damage_ok": dmg_ok, "checks": checks, "valid": valid, "delta": delta, "ci95": {x: ci(boot[x]) for x in boot}, "chat_passes": passes,
           "degenerate": deg, "panels": {x: {p: 100.0 * float(arms[x][p].mean()) for p in PANELS} for x in names},
           "docs": {x: docs_metrics(a.work, x) for x in ["ks_stk4"] + have}, "over_ks_stk4": m, "over_ks_stk4_ci95": mci}
    if "e2e_row" in have:
        v = valid["e2e_row"]
        first = training(m["e2e_row"], mci["e2e_row"])
        again = training(m["e2e_row2"], mci["e2e_row2"]) if "e2e_row2" in have and valid["e2e_row2"] else None
        t = second(first, m["e2e_row"], again) if v else "VOID"
        wb = delta["e2e_row"] - delta["bal4"] - delta["ks_org"]
        wci = ci(boot["e2e_row"] - boot["bal4"] - boot["ks_org"])
        gci = ci(bgen["e2e_row"] - bgen["ks_stk4"])
        c8 = ci(b8["e2e_row"] - b8["ks_stk4"])
        bs, be = doc_bpb(a.work, "ks_stk4"), doc_bpb(a.work, "e2e_row")
        dd = np.array([be[d] - bs[d] for d in sorted(bs)])
        drng = np.random.default_rng(SEED)
        dci = ci([dd[drng.integers(0, len(dd), len(dd))].mean() for _ in range(DRAWS)])
        out["e2e_row"] = {"label": label(ci(boot["e2e_row"])) if v else "VOID",
                          "chat_ok": passes["e2e_row"] >= passes["ref"] - CHAT_MARGIN and deg["e2e_row"] <= deg["ref"] + DEGEN_MARGIN,
                          "training": t, "first_order": first, "second_order": again,
                          "wins_back": wb, "wins_back_ci95": wci, "wins_back_label": wins_back(wci) if v else "VOID",
                          "ks_stk4_wins_back": delta["ks_stk4"] - delta["bal4"] - delta["ks_org"],
                          "generation_k4": sub(arms["e2e_row"], None, GEN) - sub(arms["ks_stk4"], None, GEN), "generation_k4_ci95": gci,
                          "generation_label": generation(gci) if v else "VOID",
                          "k8_margin": sub(arms["e2e_row"], None, K8P) - sub(arms["ks_stk4"], None, K8P), "k8_ci95": c8, "k8": k8(c8) if v else "VOID",
                          "documents_dbpb": float(dd.mean()), "documents_ci95": dci, "documents": documents(dci) if v else "VOID"}
        if "e2e_grp" in have:
            g = delta["e2e_grp"] - delta["e2e_row"]
            gc = ci(boot["e2e_grp"] - boot["e2e_row"])
            out["granularity"] = {"margin": g, "ci95": gc, "label": groups(g, gc) if v and valid["e2e_grp"] else "VOID"}
        if t == "E2E PAYS" and "glob" in have:
            rm, rci = delta["e2e_row"] - delta["glob"], ci(boot["e2e_row"] - boot["glob"])
            out["trivial"] = {"rows_over_glob": rm, "ci95": rci, "glob_gain": m["glob"], "row_gain": m["e2e_row"],
                              "label": trivial(rm, rci, m["glob"], m["e2e_row"]) if valid["glob"] else "VOID"}
    a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("panels", "ci95", "parity", "docs")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
