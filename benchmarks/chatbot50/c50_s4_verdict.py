"""C50-S4: MVE-5's criterion and S4's verdict, from the evaluator's records only (S4 protocol, gates).

Items are scored by S2's and A4's own functions (`c50_s2_checks.qa_check`, `c50_a4_checks`), on the items present in
the evaluator's outputs (the S5 half is never produced). Documents: bits = -sum log2 p(target), bytes from the
tokenizer, as in S2.

    python -I benchmarks/chatbot50/c50_s4_verdict.py --selftest
    python -I benchmarks/chatbot50/c50_s4_verdict.py mve5 --t0 DIR --json OUT
    python -I benchmarks/chatbot50/c50_s4_verdict.py s4 --evals DIR --train DIR --json OUT
      (--evals holds t0/ c48/ c496/ a/ b/ b_remove/ b_shuffle/ b_mask/ a_shuffle/; --train holds <arm>/train.json)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_a4_checks as A4  # noqa: E402
from c50_compare import byte_lengths  # noqa: E402
from c50_s2_checks import qa_check  # noqa: E402

C = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
A4_IN, S2_IN = C / "a4/inputs", C / "s2/inputs"
LLAMA_T0 = {"a4": C / "a4/p0", "s2": C / "s2/t0"}       # S2's T0: A4's P0 outputs and S2's new T0 panels
LLAMA_F0 = C / "s2/f0"
SPLIT = Path(__file__).resolve().parent / "results/c50_s4_split.json"
SEED, DRAWS = 20261016, 10_000
MARGIN, DOC_FIDELITY_BPB = 0.20, 0.010
ITEM_PANELS = ("nq", "tqa", "mmlu", "arc", "gsm8k")
GENERAL = ITEM_PANELS + ("chat",)
INPUTS = {"nq": S2_IN, "tqa": S2_IN, "mmlu": S2_IN, "arc": A4_IN, "gsm8k": A4_IN, "chat": A4_IN, "docs": A4_IN}
PAIRS = {"a": ("c496", ["a_shuffle"]), "b": ("c48", ["b_remove", "b_shuffle", "b_mask"])}


# ---------------------------------------------------------------- reading
def likelihood_present(arm: Path, name: str, inputs=None) -> dict[int, bool]:
    recs = np.fromfile(arm / f"{name}.pos.bin", dtype=A4.REC)
    by = {}
    for d in np.unique(recs["doc"]):
        s = recs["doc"] == d
        by[int(d)] = (recs["pos"][s], recs["logp_target"][s].astype(np.float64))
    out = {}
    for m in A4.meta(inputs or INPUTS[name], name):
        if m["item"] * 8 not in by:
            continue
        mean = []
        for k, slen in enumerate(m["suffix_lens"]):
            pos, lp = by[m["item"] * 8 + k]
            sel = pos >= m["prefix_len"]
            assert int(sel.sum()) == slen, (name, m["item"], k)
            mean.append(float(-lp[sel].sum()) / slen)
        out[m["item"]] = int(np.argmin(mean)) == m["label"]
    return out


def correctness(dirs: dict[str, Path], tok, panels=None, inputs=None) -> dict[str, dict[int, bool]]:
    """dirs: panel -> the directory holding its output. Returns panel -> {item: right}, for `panels` (default all).
    inputs: panel -> its input set (default INPUTS)."""
    want = set(panels or ("arc", "mmlu", "nq", "tqa", "gsm8k", "chat"))
    inp = inputs or INPUTS
    res = {}
    for name in ("arc", "mmlu"):
        if name in want:
            res[name] = likelihood_present(dirs[name], name, inp[name])
    for name in ("nq", "tqa"):
        if name not in want:
            continue
        rows = A4.gen_rows(dirs[name], name)
        res[name] = {m["item"]: qa_check(A4.decode(tok, rows[m["item"]]["gen"]), m["aliases"])["match"]
                     for m in A4.meta(inp[name], name) if m["item"] in rows}
    if "gsm8k" not in want and "chat" not in want:
        return res
    rows = A4.gen_rows(dirs["gsm8k"], "gsm8k")
    res["gsm8k"] = {m["item"]: A4.num_equal(A4.gsm8k_answer(A4.decode(tok, rows[m["item"]]["gen"])), m["answer"])
                    for m in A4.meta(INPUTS["gsm8k"], "gsm8k") if m["item"] in rows}
    rows = A4.gen_rows(dirs["chat"], "chat")
    res["chat"] = {m["item"]: A4.chat_check(m, A4.decode(tok, rows[m["item"]]["gen"]), rows[m["item"]]["stop"] == "EOS")["pass"]
                   for m in A4.meta(INPUTS["chat"], "chat") if m["item"] in rows}
    return res


def doc_bits(arm: Path, blen: np.ndarray, docs: list[int]) -> tuple[np.ndarray, np.ndarray]:
    r = np.fromfile(arm / "docs.pos.bin", dtype=A4.REC)
    bits, nbytes = [], []
    for d in docs:
        s = r["doc"] == d
        assert s.any(), (arm, d)
        bits.append(float(-r["logp_target"][s].astype(np.float64).sum() / math.log(2)))
        nbytes.append(float(blen[r["target"][s]].sum()))
    return np.array(bits), np.array(nbytes)


def llama_dirs(base: dict[str, Path]) -> dict[str, Path]:
    return {"nq": base["s2"], "tqa": base["s2"], "mmlu": base["s2"], "arc": base["a4"], "gsm8k": base["a4"],
            "chat": base["a4"], "docs": base["a4"]}


# ---------------------------------------------------------------- rules
def paired_ok(ref: np.ndarray, x: np.ndarray) -> dict:
    """A4's paired rule: b = ref right and x wrong, c = the reverse; not worse iff b - c <= max(2 sqrt(b + c), 1)."""
    b, c = int((ref & ~x).sum()), int((x & ~ref).sum())
    return {"b": b, "c": c, "ok": b - c <= max(2 * math.sqrt(b + c), 1)}


def recovered(bits_t0: np.ndarray, bits_x: np.ndarray, gap: np.ndarray, idx=None) -> float:
    i = slice(None) if idx is None else idx
    return float((bits_t0[i] - bits_x[i]).sum() / gap[i].sum())


def delta(bits_t0, bits_x, bits_c, gap) -> dict:
    """Delta = R(X) - R(C) on the documents, with a paired document bootstrap that recomputes both sums."""
    d = recovered(bits_t0, bits_x, gap) - recovered(bits_t0, bits_c, gap)
    rng = np.random.default_rng(SEED)
    n = len(gap)
    boot = []
    for _ in range(DRAWS):
        i = rng.integers(0, n, n)
        boot.append(recovered(bits_t0, bits_x, gap, i) - recovered(bits_t0, bits_c, gap, i))
    return {"delta": d, "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def decide_arm(s: dict) -> dict:
    """s: delta, ci95, items_ok, general_ok {panel: bool}, controls {name: {delta, threshold}}, trained.
    PASS / NOT PASSED / FAIL / INCONCLUSIVE, as fixed in the protocol."""
    if not s["trained"]:
        return {"outcome": "INCONCLUSIVE", "why": "the arm or its control did not train"}
    primary = s["delta"] >= MARGIN and s["ci95"][0] > 0
    if not primary:
        return {"outcome": "FAIL", "why": "primary not met"}
    fails = []
    if not s["items_ok"]:
        fails.append("S4 donor-gap items worse than the control")
    fails += [f"general panel regression: {p}" for p, ok in s["general_ok"].items() if not ok]
    fails += [f"control {k} does not erase the gain" for k, c in s["controls"].items() if c["delta"] > c["threshold"]]
    if fails:
        return {"outcome": "NOT PASSED", "why": fails}
    return {"outcome": "PASS", "why": "quality PASS, invariance pending"}


def decide_s4(arms: dict) -> str:
    o = [arms[k]["outcome"] for k in ("a", "b")]
    if "PASS" in o:
        return "PASS"
    if all(x == "FAIL" for x in o):
        return "FAIL"
    return "NOT PASSED"


def control_threshold(name: str, d: float, mask_share: float | None) -> float:
    return d * (1 - mask_share / 2) if name == "b_mask" else d / 2


# ---------------------------------------------------------------- commands
def cmd_mve5(a) -> int:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    tok = A4.Tokenizer.from_file(A4.TOKENIZER)
    blen = byte_lengths(Path(A4.TOKENIZER))
    docs = split["s4"]["docs"]
    pt_bits, nb = doc_bits(a.t0, blen, docs)
    ll_bits, nb2 = doc_bits(LLAMA_T0["a4"], blen, docs)
    assert np.array_equal(nb, nb2)
    dbpb = float((pt_bits - ll_bits).sum() / nb.sum())
    pt = correctness({p: a.t0 for p in GENERAL}, tok)
    ll = correctness(llama_dirs(LLAMA_T0), tok)
    gap_items = {p: set(split["s4"][p]) for p in ITEM_PANELS}
    general, donor = {}, {}
    for p in GENERAL:
        items = [i for i in pt[p] if i not in gap_items.get(p, set())]
        general[p] = paired_ok(np.array([ll[p][i] for i in items]), np.array([pt[p][i] for i in items]))
        general[p]["n"] = len(items)
    for p in ITEM_PANELS:
        donor[p] = {"n": len(gap_items[p]), "pytorch_t0_right": int(sum(pt[p][i] for i in gap_items[p]))}
    ok = abs(dbpb) <= DOC_FIDELITY_BPB and all(g["ok"] for g in general.values())
    rec = {"cell": "C50-S4 MVE-5 verdict", "docs": {"n": len(docs), "dbpb_pytorch_minus_llama": dbpb,
           "threshold": DOC_FIDELITY_BPB}, "general": general, "donor_gap_s4_reported": donor,
           "verdict": "PASS" if ok else "FAIL"}
    Path(a.json).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


def cmd_s4(a) -> int:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    tok = A4.Tokenizer.from_file(A4.TOKENIZER)
    blen = byte_lengths(Path(A4.TOKENIZER))
    docs = split["s4"]["docs"]
    gap = doc_bits(LLAMA_T0["a4"], blen, docs)[0] - doc_bits(LLAMA_F0, blen, docs)[0]
    names = ["t0", "c48", "c496", "a", "b", "b_remove", "b_shuffle", "b_mask", "a_shuffle"]
    bits = {n: doc_bits(a.evals / n, blen, docs)[0] for n in names}
    right = {n: correctness({p: a.evals / n for p in GENERAL}, tok) for n in ("t0", "c48", "c496", "a", "b")}
    trained = {n: json.loads((a.train / n / "train.json").read_text(encoding="utf-8")) for n in ("a", "b", "c48", "c496")}
    mask_share = json.loads((a.evals / "b_mask" / "eval.json").read_text(encoding="utf-8"))["mask_lookup_share"]
    R = {n: recovered(bits["t0"], bits[n], gap) for n in names}
    gap_items = {p: split["s4"][p] for p in ITEM_PANELS}
    arms = {}
    for x, (c, ctrls) in PAIRS.items():
        d = delta(bits["t0"], bits[x], bits[c], gap)
        ref = np.concatenate([[right[c][p][i] for i in gap_items[p]] for p in ITEM_PANELS])
        xx = np.concatenate([[right[x][p][i] for i in gap_items[p]] for p in ITEM_PANELS])
        items = paired_ok(ref.astype(bool), xx.astype(bool))
        general = {}
        for p in GENERAL:
            it = [i for i in right[x][p] if i not in set(gap_items.get(p, []))]
            general[p] = paired_ok(np.array([right["t0"][p][i] for i in it]), np.array([right[x][p][i] for i in it]))
        controls = {k: {"delta": R[k] - R[c], "threshold": control_threshold(k, d["delta"], mask_share)} for k in ctrls}
        ok_train = all(trained[n]["status"] == "COMPLETE" and trained[n]["val_last"] < trained[n]["val_first"]
                       for n in (x, c))
        s = {"delta": d["delta"], "ci95": d["ci95"], "items_ok": items["ok"], "general_ok": {p: g["ok"] for p, g in general.items()},
             "controls": controls, "trained": ok_train}
        arms[x] = {**s, "control": c, "items": items, "general": general, **decide_arm(s)}
    rec = {"cell": "C50-S4 verdict", "R": R, "mask_lookup_share": mask_share, "arms": arms, "s4": decide_s4(arms),
           "copy_share_a": [e["copy_share"] for e in trained["a"]["log"]][-1:],
           "val": {n: [trained[n]["val_first"], trained[n]["val_last"]] for n in trained}}
    Path(a.json).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("R", "s4")}, indent=1))
    for x in arms:
        print(x, arms[x]["outcome"], arms[x]["why"], f"delta {arms[x]['delta']:+.4f} ci95 {arms[x]['ci95']}")
    return 0


def selftest() -> int:
    """Planted cases (protocol): each must give the stated outcome."""
    rng = np.random.default_rng(1)
    n = 36
    gap = rng.uniform(80, 400, n)                       # bits per document
    t0 = rng.uniform(4000, 9000, n)
    good = {"items_ok": True, "general_ok": {p: True for p in GENERAL}, "trained": True}

    def arm(share_x, share_c, ctrl_share, mask=None):
        bx, bc = t0 - share_x * gap * rng.uniform(0.8, 1.2, n), t0 - share_c * gap * rng.uniform(0.8, 1.2, n)
        d = delta(t0, bx, bc, gap)
        rc = recovered(t0, bc, gap)
        ctl = {"shuffle": {"delta": recovered(t0, t0 - ctrl_share * gap, gap) - rc,
                           "threshold": control_threshold("b_shuffle", d["delta"], None)}}
        if mask is not None:
            ctl["b_mask"] = {"delta": recovered(t0, t0 - mask * gap, gap) - rc,
                             "threshold": control_threshold("b_mask", d["delta"], 0.5)}
        return {**good, **d, "controls": ctl}

    cases = [
        ("gain the control lacks, erased by its controls", arm(0.40, 0.05, 0.05, mask=0.20), "PASS"),
        ("the same gain in the control", arm(0.30, 0.28, 0.05), "FAIL"),
        ("a gain that survives the shuffle", arm(0.40, 0.05, 0.38), "NOT PASSED"),
        ("a gain that survives the mask", arm(0.40, 0.05, 0.05, mask=0.40), "NOT PASSED"),
        ("a general-panel regression", {**arm(0.40, 0.05, 0.05), "general_ok": {**good["general_ok"], "nq": False}},
         "NOT PASSED"),
        ("an untrained arm", {**arm(0.40, 0.05, 0.05), "trained": False}, "INCONCLUSIVE"),
    ]
    ok = True
    for name, s, want in cases:
        got = decide_arm(s)["outcome"]
        ok &= got == want
        print(f"{'ok ' if got == want else 'BAD'} {name}: {got} (want {want}), delta {s['delta']:+.3f} ci95 "
              f"[{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]")
    pr = paired_ok(np.array([1, 1, 1, 0] * 10, bool), np.array([1, 1, 0, 0] * 10, bool))
    ok &= not pr["ok"]                                  # 10 broken, 0 fixed: worse
    print("ok " if not pr["ok"] else "BAD", "paired rule flags 10 broken / 0 fixed:", pr)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    m = sub.add_parser("mve5")
    m.add_argument("--t0", type=Path, required=True)
    m.add_argument("--json", required=True)
    s = sub.add_parser("s4")
    s.add_argument("--evals", type=Path, required=True)
    s.add_argument("--train", type=Path, required=True)
    s.add_argument("--json", required=True)
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    raise SystemExit({"mve5": cmd_mve5, "s4": cmd_s4}[a.cmd](a))
