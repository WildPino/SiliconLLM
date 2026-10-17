"""C50-S2 verdict: the Ling-mini -> Ling-flash gap map (it chooses nothing beyond the registered rule).

Arms: M0 (Ling-mini Q4_K_M, no imatrix), F0 (Ling-flash Q4_K_M, the same recipe), T0 (P0, reported beside).
Per item-level panel, paired F0 against M0: fixed = F0 right and M0 wrong, broken = the reverse.
Relative error reduction of a family = (fixed - broken) / (M0's errors), pooled over the family's items:
    knowledge K = nq + tqa (closed-book short answers);  reasoning R = gsm8k + arc.
D = RER_K - RER_R, with a percentile CI95 from 10,000 bootstrap draws (seed 20261016) that resample items within
each panel.  Outcome:
    KNOWLEDGE  iff the CI95 of D lies above 0;  REASONING  iff it lies below 0;  MIXED otherwise.
    S3-S5 proceed iff the outcome is not REASONING and the CI95 of RER_K lies above 0 (ROADMAP §5 S2); otherwise
    the quality half of Phase 1 is re-planned before any T4 spend.
    INCONCLUSIVE if M0 has no error in a family.
Reported, not gated: per-panel accuracies of the three arms; MMLU and chat; the documents' paired dBPB (F0 - M0 and
T0 - M0, document bootstrap); the imatrix effect T0 - M0.  The donor-gap panel for S4/S5 lists the items F0 gets
right and T0 gets wrong, and the documents where F0's BPB is below T0's by at least 0.05.
--selftest runs planted cases of the rule on synthetic arms (a knowledge-only gap, a reasoning-only gap, equal gaps,
no gap) and fails unless each gets its registered outcome.

Usage:
    python -I benchmarks/chatbot50/c50_s2_verdict.py --m0 m0.json --f0 f0.json --t0 t0.json \
        --docs-m0 <dir> --docs-f0 <dir> --docs-t0 <dir> --a4-inputs E:/.../a4/inputs --json verdict.json
    python -I benchmarks/chatbot50/c50_s2_verdict.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SEED, DRAWS = 20261016, 10_000
FAMILIES = {"K": ("nq", "tqa"), "R": ("gsm8k", "arc")}
PANELS = ("nq", "tqa", "gsm8k", "arc", "mmlu", "chat")
DOC_MARGIN_BPB = 0.05
REC = np.dtype([("doc", "<i4"), ("pos", "<i4"), ("target", "<i4"), ("top1", "<i4"),
                ("logp_target", "<f4"), ("logp_top1", "<f4"), ("lse", "<f8")])


def vec(scores: dict, panel: str) -> np.ndarray:
    return np.array([bool(i["correct"]) for i in sorted(scores[panel]["items"], key=lambda i: i["item"])])


def rer(m: dict[str, np.ndarray], f: dict[str, np.ndarray], panels, idx=None) -> float:
    fixed = broken = errors = 0
    for p in panels:
        a, b = (m[p], f[p]) if idx is None else (m[p][idx[p]], f[p][idx[p]])
        fixed += int(np.sum(~a & b)); broken += int(np.sum(a & ~b)); errors += int(np.sum(~a))
    return (fixed - broken) / errors if errors else float("nan")


def decide(m: dict[str, np.ndarray], f: dict[str, np.ndarray]) -> dict:
    for fam, ps in FAMILIES.items():
        if sum(int(np.sum(~m[p])) for p in ps) == 0:
            return {"outcome": "INCONCLUSIVE", "why": f"M0 has no error in family {fam}"}
    rng = np.random.default_rng(SEED)
    used = sorted({p for ps in FAMILIES.values() for p in ps})
    dk, dr, dd = [], [], []
    for _ in range(DRAWS):
        idx = {p: rng.integers(0, len(m[p]), len(m[p])) for p in used}
        k, r = rer(m, f, FAMILIES["K"], idx), rer(m, f, FAMILIES["R"], idx)
        dk.append(k); dr.append(r); dd.append(k - r)
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]  # noqa: E731
    out = {"rer_k": rer(m, f, FAMILIES["K"]), "rer_r": rer(m, f, FAMILIES["R"]), "ci95_rer_k": ci(dk), "ci95_rer_r": ci(dr)}
    out["d"], out["ci95_d"] = out["rer_k"] - out["rer_r"], ci(dd)
    out["outcome"] = "KNOWLEDGE" if out["ci95_d"][0] > 0 else "REASONING" if out["ci95_d"][1] < 0 else "MIXED"
    out["proceed_s3_s5"] = out["outcome"] != "REASONING" and out["ci95_rer_k"][0] > 0
    return out


def selftest() -> int:
    rng = np.random.default_rng(1)
    n = {"nq": 600, "tqa": 600, "gsm8k": 100, "arc": 600}

    def arm(acc: dict[str, float]) -> dict[str, np.ndarray]:
        return {p: rng.random(n[p]) < acc[p] for p in n}

    base = {"nq": 0.15, "tqa": 0.30, "gsm8k": 0.60, "arc": 0.55}
    m = arm(base)

    def fix(share: dict[str, float]) -> dict[str, np.ndarray]:   # F0 = M0 with a share of M0's errors fixed
        return {p: m[p] | ((~m[p]) & (rng.random(n[p]) < share[p])) for p in n}

    cases = [("knowledge-only gap", fix({"nq": 0.30, "tqa": 0.30, "gsm8k": 0.0, "arc": 0.0}), "KNOWLEDGE", True),
             ("reasoning-only gap", fix({"nq": 0.0, "tqa": 0.0, "gsm8k": 0.4, "arc": 0.4}), "REASONING", False),
             ("equal gaps", fix({"nq": 0.25, "tqa": 0.25, "gsm8k": 0.25, "arc": 0.25}), "MIXED", True),
             ("no gap", {p: m[p].copy() for p in n}, "MIXED", False)]
    bad = 0
    for name, f, want, proceed in cases:
        d = decide(m, f)
        ok = d["outcome"] == want and d["proceed_s3_s5"] == proceed
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {d['outcome']} proceed={d['proceed_s3_s5']} "
              f"RER_K {d['rer_k']:.3f} RER_R {d['rer_r']:.3f} D CI95 [{d['ci95_d'][0]:.3f}, {d['ci95_d'][1]:.3f}]")
    print("selftest", "FAIL" if bad else "PASS")
    return 1 if bad else 0


def doc_bpb(a4_inputs: Path, arm_dir: Path) -> dict[int, float]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from c50_compare import byte_lengths
    import c50_a4_checks as A4
    r = np.fromfile(arm_dir / "docs.pos.bin", dtype=REC)
    blen = byte_lengths(Path(A4.TOKENIZER))[r["target"]].astype(np.float64)
    out = {}
    for d in np.unique(r["doc"]):
        s = r["doc"] == d
        out[int(d)] = float(-r["logp_target"][s].astype(np.float64).sum() / np.log(2) / blen[s].sum())
    return out


def paired_doc(a: dict[int, float], b: dict[int, float]) -> dict:
    if sorted(a) != sorted(b):
        raise SystemExit("arms do not score the same documents")
    d = np.array([a[k] - b[k] for k in sorted(a)])
    rng = np.random.default_rng(SEED)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(DRAWS)]
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "n": len(d)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    for k in ("m0", "f0", "t0", "docs-m0", "docs-f0", "docs-t0", "a4-inputs", "json"):
        ap.add_argument(f"--{k}")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    s = {a: json.loads(Path(getattr(args, a)).read_text(encoding="utf-8")) for a in ("m0", "f0", "t0")}
    v = {a: {p: vec(s[a], p) for p in PANELS} for a in s}
    out = {"cell": "C50-S2 verdict", "accuracy": {a: {p: float(v[a][p].mean()) for p in PANELS} for a in v},
           "n": {p: int(len(v["m0"][p])) for p in PANELS}}
    out["paired_f0_m0"] = {p: {"fixed": int(np.sum(~v["m0"][p] & v["f0"][p])), "broken": int(np.sum(v["m0"][p] & ~v["f0"][p]))}
                           for p in PANELS}
    out["imatrix_effect_t0_minus_m0"] = {p: float(v["t0"][p].mean() - v["m0"][p].mean()) for p in PANELS}
    out["decision"] = decide({p: v["m0"][p] for p in PANELS}, {p: v["f0"][p] for p in PANELS})
    a4 = Path(args.a4_inputs)
    bpb = {a: doc_bpb(a4, Path(getattr(args, f"docs_{a}"))) for a in ("m0", "f0", "t0")}
    out["docs_dbpb"] = {"f0_minus_m0": paired_doc(bpb["f0"], bpb["m0"]), "t0_minus_m0": paired_doc(bpb["t0"], bpb["m0"])}
    gap = {p: [i for i in range(len(v["f0"][p])) if v["f0"][p][i] and not v["t0"][p][i]] for p in ("nq", "tqa", "mmlu", "arc", "gsm8k")}
    gap["docs"] = [d for d in sorted(bpb["f0"]) if bpb["f0"][d] <= bpb["t0"][d] - DOC_MARGIN_BPB]
    out["donor_gap_panel"] = {"rule": f"F0 right and T0 wrong; documents with F0 BPB <= T0 BPB - {DOC_MARGIN_BPB}",
                              "items": gap, "sizes": {k: len(x) for k, x in gap.items()}}
    d = out["decision"]
    print("accuracy", json.dumps(out["accuracy"]))
    if "rer_k" in d:
        print(f"RER_K {d['rer_k']:.3f} CI95 {d['ci95_rer_k']}  RER_R {d['rer_r']:.3f} CI95 {d['ci95_rer_r']}  D CI95 {d['ci95_d']}")
    print("OUTCOME", d["outcome"], "proceed S3-S5:", d.get("proceed_s3_s5"))
    print("docs dBPB F0-M0", out["docs_dbpb"]["f0_minus_m0"], "T0-M0", out["docs_dbpb"]["t0_minus_m0"])
    print("donor-gap panel sizes", out["donor_gap_panel"]["sizes"])
    Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
