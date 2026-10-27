"""C50-T2 option N: the verdict, from the evaluator's and the trainers' records only (protocol C50_T2_N, rule).

R(X) = (bits_T0 - bits_X) / (bits_T0,S2 - bits_F0,S2) on S4's 36 documents, as in S4: the numerator from the PyTorch
instrument (T0 is MVE-5's evaluation), the denominator S2's llama.cpp gap. Every difference of R comes with a paired
document bootstrap (10,000 draws, seed 20261016) that recomputes both sums (`c50_s4_verdict.delta`). Items are scored
by S2's and A4's functions (`c50_s4_verdict.correctness`) and compared by A4's paired rule.

    python -I benchmarks/chatbot50/c50_t2_n_verdict.py --selftest
    python -I benchmarks/chatbot50/c50_t2_n_verdict.py --evals DIR --train DIR --json OUT
      (--evals holds v0/ n/ n_shuffle/ e2/, each with eval.json; --train holds n/ and e2/, each with the last
       session's train.json and final.pt, and session1.json, the first session's train.json, when the last resumed)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_s4_verdict as V  # noqa: E402

C = V.C
T0_PT = C / "s4/verdict/evals/t0"              # MVE-5's PyTorch T0, the instrument's own baseline
FRESH = C / "t1/fresh/inputs"
V0_SHA = "e490f803"                            # prefix of v0's sha256 (C50-T1 addendum 2's pin of t1_a1_q4_k_m.gguf)
MARGIN = 0.05
FACTS = ("nq", "tqa", "mmlu", "arc")           # fresh items
REST = ("gsm8k", "chat")                       # A4's panels, whole
ARMS = ("v0", "n", "n_shuffle", "e2")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def decide(s: dict) -> dict:
    """s: trained, d_v0 {delta, ci95}, d_e2 {delta, ci95}, facts_ok {panel: bool}, rest_ok {panel: bool},
    shuffle {delta, threshold}. The protocol's outcomes; nothing else is chosen here."""
    if not s["trained"]:
        return {"outcome": "INCONCLUSIVE", "why": "N or E2 did not train"}
    prim = all(s[k]["delta"] >= MARGIN and s[k]["ci95"][0] > 0 for k in ("d_v0", "d_e2"))
    if not prim:
        return {"outcome": "FAIL", "why": "primary not met: R(N) - R(v0) and R(N) - R(E2) must both be >= "
                                          f"{MARGIN} with CI95 above 0"}
    fails = [f"facts worse than v0: {p}" for p, ok in s["facts_ok"].items() if not ok]
    fails += [f"worse than v0: {p}" for p, ok in s["rest_ok"].items() if not ok]
    if s["shuffle"]["delta"] > s["shuffle"]["threshold"]:
        fails.append("the shuffle keeps more than half of the gain")
    if fails:
        return {"outcome": "NOT PASSED", "why": fails}
    return {"outcome": "PASS", "why": "quality PASS, invariance pending"}


def run(a) -> int:
    split = json.loads(V.SPLIT.read_text(encoding="utf-8"))
    docs = split["s4"]["docs"]
    tok = V.A4.Tokenizer.from_file(V.A4.TOKENIZER)
    blen = V.byte_lengths(Path(V.A4.TOKENIZER))
    ev = {k: json.loads((a.evals / k / "eval.json").read_text(encoding="utf-8")) for k in ARMS}
    tr = {k: json.loads((a.train / k / "train.json").read_text(encoding="utf-8")) for k in ("n", "e2")}
    wsha = {k: sha256(a.train / k / "final.pt") for k in ("n", "e2")}
    want = {"docs": len(docs), **{p: len(V.A4.meta(FRESH, p)) for p in FACTS}}
    checks = {}
    for k, e in ev.items():
        checks[f"{k}_gguf_is_v0"] = e["gguf_sha256"].startswith(V0_SHA)
        checks[f"{k}_arm"] = e["arm"] == k.split("_")[0] and (e.get("n_mode") or "full") == \
            ("shuffle" if k == "n_shuffle" else "full")
        checks[f"{k}_weights"] = e["weights_sha256"] == (wsha[k.split("_")[0]] if k != "v0" else None)
        checks[f"{k}_panels"] = set(e["panels_done"]) == {"docs", *FACTS, *REST} and e["limit"] == 0
        checks[f"{k}_docs_count"] = e["panels_done"].get("docs", {}).get("sequences") == want["docs"]
    for k, t in tr.items():
        checks[f"{k}_train_complete"] = t["status"] == "COMPLETE"
        if t["resume"]:
            # A resumed session measures no identity (the checkpoint does not carry it): it comes from session 1, which
            # must have started from step 0, paused, and be the prefix of the final session's log.
            s1 = json.loads((a.train / k / "session1.json").read_text(encoding="utf-8"))
            checks[f"{k}_session1_fresh_paused"] = s1["resume"] is None and s1["status"] == "PAUSED"
            checks[f"{k}_session1_is_prefix"] = t["log"][:len(s1["log"])] == s1["log"] and \
                t["vals"][:len(s1["vals"])] == s1["vals"] and "-s1/" in t["resume"]
            t = s1
        checks[f"{k}_identity_step0"] = (t.get("identity_step0") or {}).get("max_abs_logit") == 0.0
    rec = {"cell": "C50-T2 option N verdict", "checks": checks, "weights_sha256": wsha}
    if not all(checks.values()):
        rec["verdict"] = "STOP: " + ", ".join(k for k, v in checks.items() if not v)
        Path(a.json).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(rec, indent=1))
        return 0
    gap = V.doc_bits(V.LLAMA_T0["a4"], blen, docs)[0] - V.doc_bits(V.LLAMA_F0, blen, docs)[0]
    t0 = V.doc_bits(T0_PT, blen, docs)[0]
    bits = {k: V.doc_bits(a.evals / k, blen, docs)[0] for k in ARMS}
    R = {k: V.recovered(t0, bits[k], gap) for k in ARMS}
    d_v0 = V.delta(t0, bits["n"], bits["v0"], gap)
    d_e2 = V.delta(t0, bits["n"], bits["e2"], gap)
    d_sh = V.delta(t0, bits["n_shuffle"], bits["v0"], gap)
    e2_v0 = V.delta(t0, bits["e2"], bits["v0"], gap)
    inputs = {**{p: FRESH for p in FACTS}, **{p: V.INPUTS[p] for p in REST}}
    right = {k: V.correctness({p: a.evals / k for p in (*FACTS, *REST)}, tok, (*FACTS, *REST), inputs)
             for k in ("v0", "n", "e2", "n_shuffle")}
    items = {}
    for p in (*FACTS, *REST):
        it = sorted(right["v0"][p])
        assert all(sorted(right[k][p]) == it for k in right), p
        col = {k: np.array([right[k][p][i] for i in it]) for k in right}
        items[p] = {"n": len(it), "right": {k: int(v.sum()) for k, v in col.items()},
                    "n_vs_v0": V.paired_ok(col["v0"], col["n"]), "e2_vs_v0": V.paired_ok(col["v0"], col["e2"]),
                    "n_vs_e2": V.paired_ok(col["e2"], col["n"]), "shuffle_vs_v0": V.paired_ok(col["v0"], col["n_shuffle"])}
        if p in FACTS:
            assert items[p]["n"] == want[p], (p, items[p]["n"], want[p])
    s = {"trained": all(t["trained"] for t in tr.values()), "d_v0": d_v0, "d_e2": d_e2,
         "facts_ok": {p: items[p]["n_vs_v0"]["ok"] for p in FACTS},
         "rest_ok": {p: items[p]["n_vs_v0"]["ok"] for p in REST},
         "shuffle": {"delta": d_sh["delta"], "threshold": d_v0["delta"] / 2}}
    rec.update({"R": R, "n_minus_v0": d_v0, "n_minus_e2": d_e2, "shuffle_minus_v0": d_sh, "e2_minus_v0": e2_v0,
                "items": items, "decision_inputs": s, **decide(s),
                "fire_docs": ev["n"]["panels_done"]["docs"].get("fire"),
                "fire_by_panel": {p: ev["n"]["panels_done"][p].get("fire") for p in ev["n"]["panels_done"]},
                "val": {k: [t["val_first"], t["val_last"]] for k, t in tr.items()}})
    Path(a.json).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("R", "outcome", "why", "fire_docs")}, indent=1))
    for k in ("n_minus_v0", "n_minus_e2", "shuffle_minus_v0", "e2_minus_v0"):
        print(k, f"{rec[k]['delta']:+.4f}", rec[k]["ci95"])
    return 0


def selftest() -> int:
    """Planted cases: each must give the stated outcome."""
    rng = np.random.default_rng(1)
    n = 36
    gap = rng.uniform(80, 400, n)
    t0 = rng.uniform(4000, 9000, n)
    v0 = t0 - 0.67 * gap * rng.uniform(0.9, 1.1, n)
    ok_facts = {p: True for p in FACTS}
    ok_rest = {p: True for p in REST}

    def case(r_n, r_e2, r_sh, facts=ok_facts, rest=ok_rest, trained=True):
        bn = t0 - r_n * gap * rng.uniform(0.95, 1.05, n)
        be2 = t0 - r_e2 * gap * rng.uniform(0.95, 1.05, n)
        bsh = t0 - r_sh * gap * rng.uniform(0.95, 1.05, n)
        d_v0 = V.delta(t0, bn, v0, gap)
        return {"trained": trained, "d_v0": d_v0, "d_e2": V.delta(t0, bn, be2, gap), "facts_ok": facts,
                "rest_ok": rest, "shuffle": {"delta": V.delta(t0, bsh, v0, gap)["delta"], "threshold": d_v0["delta"] / 2}}

    cases = [
        ("a gain over v0 and over E2, erased by the shuffle", case(0.80, 0.68, 0.66), "PASS"),
        ("the same gain in E2 (more training, not capacity)", case(0.80, 0.78, 0.66), "FAIL"),
        ("no gain over v0", case(0.68, 0.66, 0.66), "FAIL"),
        ("a gain that survives the shuffle", case(0.80, 0.68, 0.78), "NOT PASSED"),
        ("a fact regression", case(0.80, 0.68, 0.66, facts={**ok_facts, "nq": False}), "NOT PASSED"),
        ("a gsm8k regression", case(0.80, 0.68, 0.66, rest={**ok_rest, "gsm8k": False}), "NOT PASSED"),
        ("an untrained arm", case(0.80, 0.68, 0.66, trained=False), "INCONCLUSIVE"),
    ]
    ok = True
    for name, s, want in cases:
        got = decide(s)["outcome"]
        ok &= got == want
        print(f"{'ok ' if got == want else 'BAD'} {name}: {got} (want {want}), N-v0 {s['d_v0']['delta']:+.3f} "
              f"ci95 [{s['d_v0']['ci95'][0]:+.3f}, {s['d_v0']['ci95'][1]:+.3f}], N-E2 {s['d_e2']['delta']:+.3f}")
    # the bootstrap must see noise: a +0.06 mean gain carried by 3 documents out of 36 has a CI that reaches 0
    bn = v0.copy()
    bn[:3] -= 0.06 * gap.sum() / 3
    d = V.delta(t0, bn, v0, gap)
    thin = decide({"trained": True, "d_v0": d, "d_e2": d, "facts_ok": ok_facts, "rest_ok": ok_rest,
                   "shuffle": {"delta": 0.0, "threshold": d["delta"] / 2}})["outcome"]
    ok &= thin == "FAIL"
    print(f"{'ok ' if thin == 'FAIL' else 'BAD'} a gain carried by 3 documents: {thin} (want FAIL), "
          f"delta {d['delta']:+.3f} ci95 [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--evals", type=Path)
    ap.add_argument("--train", type=Path)
    ap.add_argument("--json")
    a = ap.parse_args()
    raise SystemExit(selftest() if a.selftest else run(a))
