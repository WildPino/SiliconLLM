"""C50-T2 N-fire: the verdict, from the evaluation and training records only (protocol C50_T2_NF, rule).

Arm NF is option N with a fire cost (ReMoE's adaptive L1, fire target 0.25), trained on block 1 like N. It is judged
against the N cell's own records (v0, N, N-shuffle, E2 in `t2/n/evals`), with option N's verdict functions: R on S4's
36 documents with a paired document bootstrap, items by A4's paired rule.

    python -I benchmarks/chatbot50/c50_t2_nf_verdict.py --selftest
    python -I benchmarks/chatbot50/c50_t2_nf_verdict.py --evals DIR --train DIR --json OUT
      (--evals holds nf/ nf_shuffle/ ndocs/, each with eval.json; --train holds nf/ with the last session's
       train.json and final.pt, and session1.json when the last session resumed)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_s4_eval as E  # noqa: E402
import c50_s4_verdict as V  # noqa: E402
import c50_t2_n_verdict as NV  # noqa: E402

C = V.C
N_EVALS = C / "t2/n/evals"                     # the N cell's records: v0, n, n_shuffle, e2
N_FINAL = C / "t2/n/train/n/final.pt"          # N's weights (sha256 d045b913...)
TARGET, BAND = 0.25, 0.05                      # the fire target and the controller's tolerance on the documents
ARMS = ("nf", "nf_shuffle", "ndocs")


def decide(s: dict) -> dict:
    """s: trained, fire_docs, keep {delta, ci95}, keep_threshold, shuffle {delta}, facts_ok, rest_ok."""
    if not s["trained"] or abs(s["fire_docs"] - TARGET) > BAND:
        return {"outcome": "INCONCLUSIVE", "why": "NF did not train, or its fire share on the documents is outside "
                                                  f"{TARGET} +- {BAND}"}
    fails = []
    if not (s["keep"]["delta"] >= s["keep_threshold"] and s["keep"]["ci95"][0] > 0):
        fails.append("keep: R(NF) - R(v0) below half of R(N) - R(v0), or its CI95 reaches 0")
    if s["shuffle"]["delta"] > s["keep"]["delta"] / 2:
        fails.append("specific: the shuffle keeps more than half of NF's gain")
    fails += [f"worse than v0: {p}" for p, ok in {**s["facts_ok"], **s["rest_ok"]}.items() if not ok]
    if fails:
        return {"outcome": "NOT ADOPTED", "why": fails}
    return {"outcome": "ADOPT", "why": "the fire cost keeps half of N's gain at a quarter of the fire, and the gain "
                                       "lives in what each expert holds"}


def fire_split(fire_u8: Path, v0_pos: Path) -> dict:
    """Mean share of layers fired at positions where v0 is sure of the next token (p >= 0.5) and where it is not."""
    f = np.fromfile(fire_u8, dtype=np.uint8).astype(np.float64) / 19
    r = np.fromfile(v0_pos, dtype=E.REC)
    assert len(f) == len(r), (len(f), len(r))
    sure = r["logp_target"] >= math.log(0.5)
    return {"sure": round(float(f[sure].mean()), 5), "unsure": round(float(f[~sure].mean()), 5),
            "sure_share": round(float(sure.mean()), 5), "all": round(float(f.mean()), 5)}


def run(a) -> int:
    split = json.loads(V.SPLIT.read_text(encoding="utf-8"))
    docs = split["s4"]["docs"]
    tok = V.A4.Tokenizer.from_file(V.A4.TOKENIZER)
    blen = V.byte_lengths(Path(V.A4.TOKENIZER))
    ev = {k: json.loads((a.evals / k / "eval.json").read_text(encoding="utf-8")) for k in ARMS}
    t = json.loads((a.train / "nf" / "train.json").read_text(encoding="utf-8"))
    wsha = NV.sha256(a.train / "nf" / "final.pt")
    nsha = NV.sha256(N_FINAL)
    full = {"docs", *NV.FACTS, *NV.REST}
    checks = {}
    for k, e in ev.items():
        checks[f"{k}_gguf_is_v0"] = e["gguf_sha256"].startswith(NV.V0_SHA)
        checks[f"{k}_arm"] = e["arm"] == "n" and e["n_mode"] == ("shuffle" if k == "nf_shuffle" else "full")
        checks[f"{k}_weights"] = e["weights_sha256"] == (nsha if k == "ndocs" else wsha)
        checks[f"{k}_panels"] = set(e["panels_done"]) == ({"docs"} if k == "ndocs" else full) and e["limit"] == 0
        checks[f"{k}_docs_count"] = e["panels_done"]["docs"]["sequences"] == len(docs)
        checks[f"{k}_fire_trace"] = (a.evals / k / "docs.fire.u8").exists()
    # the trace must change nothing: N's documents again, with the trace on, bit for bit as in the N cell
    checks["trace_changes_nothing"] = (a.evals / "ndocs/docs.pos.bin").read_bytes() == \
        (N_EVALS / "n/docs.pos.bin").read_bytes()
    checks["nf_train_complete"] = t["status"] == "COMPLETE" and t["arm"] == "nf" and t["fire_target"] == TARGET
    if t["resume"]:
        s1 = json.loads((a.train / "nf" / "session1.json").read_text(encoding="utf-8"))
        checks["nf_session1_fresh_paused"] = s1["resume"] is None and s1["status"] == "PAUSED"
        checks["nf_session1_is_prefix"] = t["log"][:len(s1["log"])] == s1["log"] and \
            t["vals"][:len(s1["vals"])] == s1["vals"] and "-s1/" in t["resume"]
        t1 = s1
    else:
        t1 = t
    checks["nf_identity_step0"] = (t1.get("identity_step0") or {}).get("max_abs_logit") == 0.0
    rec = {"cell": "C50-T2 N-fire verdict", "checks": checks, "weights_sha256": {"nf": wsha, "n": nsha}}
    if not all(checks.values()):
        rec["verdict"] = "STOP: " + ", ".join(k for k, v in checks.items() if not v)
        Path(a.json).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(rec, indent=1))
        return 0
    gap = V.doc_bits(V.LLAMA_T0["a4"], blen, docs)[0] - V.doc_bits(V.LLAMA_F0, blen, docs)[0]
    t0 = V.doc_bits(NV.T0_PT, blen, docs)[0]
    dirs = {"v0": N_EVALS / "v0", "n": N_EVALS / "n", "n_shuffle": N_EVALS / "n_shuffle", "e2": N_EVALS / "e2",
            "nf": a.evals / "nf", "nf_shuffle": a.evals / "nf_shuffle"}
    bits = {k: V.doc_bits(d, blen, docs)[0] for k, d in dirs.items()}
    R = {k: V.recovered(t0, b, gap) for k, b in bits.items()}
    keep = V.delta(t0, bits["nf"], bits["v0"], gap)
    n_v0 = V.delta(t0, bits["n"], bits["v0"], gap)
    sh = V.delta(t0, bits["nf_shuffle"], bits["v0"], gap)
    inputs = {**{p: NV.FRESH for p in NV.FACTS}, **{p: V.INPUTS[p] for p in NV.REST}}
    right = {k: V.correctness({p: dirs[k] for p in (*NV.FACTS, *NV.REST)}, tok, (*NV.FACTS, *NV.REST), inputs)
             for k in ("v0", "n", "nf", "nf_shuffle")}
    items = {}
    for p in (*NV.FACTS, *NV.REST):
        it = sorted(right["v0"][p])
        assert all(sorted(right[k][p]) == it for k in right), p
        col = {k: np.array([right[k][p][i] for i in it]) for k in right}
        items[p] = {"n": len(it), "right": {k: int(v.sum()) for k, v in col.items()},
                    "nf_vs_v0": V.paired_ok(col["v0"], col["nf"]), "nf_vs_n": V.paired_ok(col["n"], col["nf"]),
                    "shuffle_vs_v0": V.paired_ok(col["v0"], col["nf_shuffle"])}
    s = {"trained": t["trained"], "fire_docs": ev["nf"]["panels_done"]["docs"]["fire"], "keep": keep,
         "keep_threshold": n_v0["delta"] / 2, "shuffle": sh,
         "facts_ok": {p: items[p]["nf_vs_v0"]["ok"] for p in NV.FACTS},
         "rest_ok": {p: items[p]["nf_vs_v0"]["ok"] for p in NV.REST}}
    v0_pos = N_EVALS / "v0/docs.pos.bin"
    rec.update({"R": R, "nf_minus_v0": keep, "n_minus_v0": n_v0, "nf_shuffle_minus_v0": sh,
                "nf_minus_n": V.delta(t0, bits["nf"], bits["n"], gap),
                "nf_minus_e2": V.delta(t0, bits["nf"], bits["e2"], gap), "items": items,
                "decision_inputs": {k: v for k, v in s.items() if k not in ("keep", "shuffle")}, **decide(s),
                "fire_by_panel": {p: ev["nf"]["panels_done"][p].get("fire") for p in ev["nf"]["panels_done"]},
                "fire_by_layer_docs": ev["nf"]["panels_done"]["docs"].get("fire_by_layer"),
                "fire_sure_vs_unsure": {"n": fire_split(a.evals / "ndocs/docs.fire.u8", v0_pos),
                                        "nf": fire_split(a.evals / "nf/docs.fire.u8", v0_pos)},
                "val": [t["val_first"], t["val_last"]]})
    Path(a.json).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("R", "outcome", "why", "fire_sure_vs_unsure")}, indent=1))
    for k in ("nf_minus_v0", "n_minus_v0", "nf_shuffle_minus_v0", "nf_minus_n", "nf_minus_e2"):
        print(k, f"{rec[k]['delta']:+.4f}", rec[k]["ci95"])
    return 0


def selftest() -> int:
    """Planted cases: each must give the stated outcome."""
    rng = np.random.default_rng(2)
    n = 36
    gap = rng.uniform(80, 400, n)
    t0 = rng.uniform(4000, 9000, n)
    v0 = t0 - 0.673 * gap * rng.uniform(0.9, 1.1, n)
    ok_f = {p: True for p in NV.FACTS}
    ok_r = {p: True for p in NV.REST}

    def case(g_nf, g_sh, fire=0.25, facts=ok_f, rest=ok_r, trained=True, n_gain=0.0103):
        # paired, as in the N cell: each arm is v0 minus a per-document gain (N - v0 there: +0.0103, CI +-0.0027)
        bnf = v0 - g_nf * gap * rng.uniform(0.5, 1.5, n)
        bsh = v0 - g_sh * gap * rng.uniform(0.5, 1.5, n)
        return {"trained": trained, "fire_docs": fire, "keep": V.delta(t0, bnf, v0, gap), "keep_threshold": n_gain / 2,
                "shuffle": V.delta(t0, bsh, v0, gap), "facts_ok": facts, "rest_ok": rest}

    cases = [
        ("N's gain kept at a quarter of the fire, erased by the shuffle", case(0.010, 0.001), "ADOPT"),
        ("the gain kept but generic (the shuffle keeps it)", case(0.010, 0.008), "NOT ADOPTED"),
        ("the cost erases the gain", case(0.002, 0.001), "NOT ADOPTED"),
        ("a fact regression", case(0.010, 0.001, facts={**ok_f, "tqa": False}), "NOT ADOPTED"),
        ("the controller missed the target", case(0.010, 0.001, fire=0.45), "INCONCLUSIVE"),
        ("NF did not train", case(0.010, 0.001, trained=False), "INCONCLUSIVE"),
    ]
    ok = True
    for name, s, want in cases:
        got = decide(s)["outcome"]
        ok &= got == want
        print(f"{'ok ' if got == want else 'BAD'} {name}: {got} (want {want}), NF-v0 {s['keep']['delta']:+.4f} "
              f"ci95 [{s['keep']['ci95'][0]:+.4f}, {s['keep']['ci95'][1]:+.4f}], shuffle {s['shuffle']['delta']:+.4f}")
    # fire_split on a planted trace: fired only where v0 is unsure
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        r = np.zeros(1000, dtype=E.REC)
        r["logp_target"] = np.where(np.arange(1000) < 300, math.log(0.9), math.log(0.1))
        f = np.where(np.arange(1000) < 300, 0, 19).astype(np.uint8)
        r.tofile(Path(d) / "v0.bin")
        f.tofile(Path(d) / "f.u8")
        fs = fire_split(Path(d) / "f.u8", Path(d) / "v0.bin")
    good = fs["sure"] == 0.0 and fs["unsure"] == 1.0 and fs["sure_share"] == 0.3
    ok &= good
    print(f"{'ok ' if good else 'BAD'} fire split on a planted trace: {fs}")
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
