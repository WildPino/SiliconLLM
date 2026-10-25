"""C50-T1 MVE-0 verdict, from the records only: K0, K1, L, and gate M (the merge keeps the LoRA's gain).

R_llama(T1) uses S4's 36 documents and S2's llama.cpp gap, with the S4 verdict's own functions; the general panel and
the donor-gap items are reported against S2's T0 by A4's paired rule.
Protocol: docs/research/CHATBOT50_ROADMAP_20261016/C50_T1_PROTOCOL_20261024.md

    python -I benchmarks/chatbot50/c50_t1_verdict.py mve0 --k0 K0.json --k1 K1.json --merge MERGE.json --layout L.json \
        --run RUN.json --json OUT
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

RES = Path(__file__).resolve().parent / "results"
P0_PIN = "44de73157f11165c0551fd06714f4f043913292a79b07683e6a34b11d3b0d2f4"
KEEP = 0.80


def both_halves(split: dict) -> dict[str, set]:
    """Both halves of the donor-gap items, left out of every general panel. T1's runs use S2's full panels, which hold
    S5's half too; the first readers left out S4's half only, so S5's items entered the general counts (fixed 24 October).
    """
    return {p: set(split["s4"][p]) | set(split["s5"][p]) for p in V.ITEM_PANELS}


def load(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def mve0(a) -> int:
    k0, k1, mg, lay, run = (load(p) for p in (a.k0, a.k1, a.merge, a.layout, a.run))
    s4 = load(RES / "c50_s4_verdict.json")
    post = load(RES / "c50_s4_posthoc.json")
    checks = {"K0": k0.get("sha256") == P0_PIN, "K1": bool(k1.get("K1")), "L": bool(lay.get("L")),
              "run_complete": all(r.get("status") == "COMPLETE" for r in run["panels"].values()) and len(run["panels"]) == 7,
              "same_lora": mg["lora_sha256"] == k1["lora_sha256"] and not mg["zero"],
              "model_is_layout_new": Path(run["model"]) == Path(lay["new"])}
    rec = {"cell": "C50-T1 MVE-0 verdict", "checks": checks}
    if all(checks.values()):
        split = json.loads(V.SPLIT.read_text(encoding="utf-8"))
        tok = V.A4.Tokenizer.from_file(V.A4.TOKENIZER)
        blen = V.byte_lengths(Path(V.A4.TOKENIZER))
        docs = split["s4"]["docs"]
        t1dir = Path(run["panels"]["docs"]["dir"])
        b_t0, nb = V.doc_bits(V.LLAMA_T0["a4"], blen, docs)
        b_f0 = V.doc_bits(V.LLAMA_F0, blen, docs)[0]
        b_t1 = V.doc_bits(t1dir, blen, docs)[0]
        gap = b_t0 - b_f0
        d = V.delta(b_t0, b_t1, b_t0, gap)                 # R(T1) - R(T0) = R(T1), with its paired bootstrap
        r_pt = s4["R"]["c48"]
        right_t0 = V.correctness(V.llama_dirs(V.LLAMA_T0), tok)
        right_t1 = V.correctness({p: t1dir for p in V.GENERAL}, tok)
        gap_items = {p: split["s4"][p] for p in V.ITEM_PANELS}
        ref = np.concatenate([[right_t0[p][i] for i in gap_items[p]] for p in V.ITEM_PANELS]).astype(bool)
        xx = np.concatenate([[right_t1[p][i] for i in gap_items[p]] for p in V.ITEM_PANELS]).astype(bool)
        general, excl = {}, both_halves(split)
        for p in V.GENERAL:
            it = [i for i in right_t1[p] if i not in excl.get(p, set())]
            t, y = np.array([right_t0[p][i] for i in it]), np.array([right_t1[p][i] for i in it])
            general[p] = {**V.paired_ok(t, y), "n": len(it), "t0_right": int(t.sum()), "t1_right": int(y.sum()),
                          "c48_pytorch_vs_pytorch_t0": post["arms"]["c48"]["general_vs_t0"][p]}
        rec.update({"bpb": {"t0": float(b_t0.sum() / nb.sum()), "f0": float(b_f0.sum() / nb.sum()),
                            "t1": float(b_t1.sum() / nb.sum())},
                    "R_llama_t1": d["delta"], "R_llama_t1_ci95": d["ci95"], "R_pytorch_c48": r_pt,
                    "threshold": KEEP * r_pt, "M": d["delta"] >= KEEP * r_pt,
                    "donor_items_vs_t0": {**V.paired_ok(ref, xx), "n": int(len(ref)), "t0_right": int(ref.sum()),
                                          "t1_right": int(xx.sum())},
                    "general_vs_t0": general,
                    "bf16_kept_median": mg.get("kept_median"), "bf16_cos_median": mg.get("cos_median"),
                    "q4_kept_median": lay.get("quant_kept_median"), "q4_cos_median": lay.get("quant_cos_median")})
        rec["verdict"] = "PASS" if rec["M"] else "FAIL (M)"
    else:
        rec["verdict"] = "STOP: " + ", ".join(k for k, v in checks.items() if not v)
    Path(a.json).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: rec[k] for k in rec if k not in ("general_vs_t0",)}, indent=1))
    return 0


def part2a(a) -> int:
    """Addendum 1 (part 2a): partial merges of c48, read on the documents, nq and tqa against T0 and against the full
    T1. Reported; the decision rule for part 2b's first arm is applied as registered."""
    split = json.loads(V.SPLIT.read_text(encoding="utf-8"))
    tok = V.A4.Tokenizer.from_file(V.A4.TOKENIZER)
    blen = V.byte_lengths(Path(V.A4.TOKENIZER))
    docs = split["s4"]["docs"]
    b_t0 = V.doc_bits(V.LLAMA_T0["a4"], blen, docs)[0]
    gap = b_t0 - V.doc_bits(V.LLAMA_F0, blen, docs)[0]
    right_t0 = V.correctness(V.llama_dirs(V.LLAMA_T0), tok)
    gap_items = both_halves(split)
    full = load(RES / "c50_t1_mve0.json")
    rec = {"cell": "C50-T1 part 2a", "full_t1": {"R": full["R_llama_t1"],
                                                 "nq_ok": full["general_vs_t0"]["nq"]["ok"],
                                                 "tqa_ok": full["general_vs_t0"]["tqa"]["ok"]}, "variants": {}}
    for name, lay_p, run_p in a.variant:
        lay, run = load(Path(lay_p)), load(Path(run_p))
        ok = (lay.get("L") and all(run["panels"].get(p, {}).get("status") == "COMPLETE" for p in ("docs", "nq", "tqa"))
              and Path(run["model"]) == Path(lay["new"]))
        v = {"checks_ok": bool(ok)}
        if ok:
            d_dir = Path(run["panels"]["docs"]["dir"])
            d = V.delta(b_t0, V.doc_bits(d_dir, blen, docs)[0], b_t0, gap)
            keep = KEEP * load(RES / "c50_s4_verdict.json")["R"]["c48"]
            v.update({"R": d["delta"], "R_ci95": d["ci95"], "keeps_gain": d["delta"] >= keep})
            for p in ("nq", "tqa"):                          # the S4 verdict's own nq/tqa scoring, on these two panels
                rows = V.A4.gen_rows(d_dir, p)
                rx = {m["item"]: V.qa_check(V.A4.decode(tok, rows[m["item"]]["gen"]), m["aliases"])["match"]
                      for m in V.A4.meta(V.INPUTS[p], p) if m["item"] in rows}
                it = [i for i in rx if i not in gap_items[p]]
                t, y = np.array([right_t0[p][i] for i in it]), np.array([rx[i] for i in it])
                v[p] = {**V.paired_ok(t, y), "n": len(it), "t0_right": int(t.sum()), "x_right": int(y.sum())}
        rec["variants"][name] = v
    va = rec["variants"]
    def clean(n):
        x = va.get(n, {})
        return bool(x.get("checks_ok") and x["keeps_gain"] and x["nq"]["ok"] and x["tqa"]["ok"])
    if clean("attn") or clean("ffn"):
        rec["part2b_first_arm"] = "train the LoRA on the group whose partial merge is clean: " + ",".join(n for n in ("attn", "ffn") if clean(n))
    else:
        rec["part2b_first_arm"] = "no group is clean alone: the anchored arm (loss anchored to T0) comes first"
    Path(a.json).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


def part2b(a) -> int:
    """Part 2b: one trained arm, merged, against T0 on the seven panels; gates L, G1, G2, G3 as registered."""
    mg, lay, run = load(a.merge), load(a.layout), load(a.run)
    split = json.loads(V.SPLIT.read_text(encoding="utf-8"))
    checks = {"L": bool(lay.get("L")) and len(lay["differs"]) == 100,
              "run_complete": len(run["panels"]) == 7 and all(r.get("status") == "COMPLETE" for r in run["panels"].values()),
              "merge_full_nonzero": mg.get("only", "all") == "all" and not mg["zero"],   # MVE-0's record predates `only`
              "lora_is_the_trained_arm": mg["lora_sha256"] == a.lora_sha256,
              "model_is_layout_new": Path(run["model"]) == Path(lay["new"])}
    rec = {"cell": "C50-T1 part 2b verdict", "arm": a.arm, "checks": checks}
    if all(checks.values()):
        tok = V.A4.Tokenizer.from_file(V.A4.TOKENIZER)
        blen = V.byte_lengths(Path(V.A4.TOKENIZER))
        docs = split["s4"]["docs"]
        xdir = Path(run["panels"]["docs"]["dir"])
        b_t0 = V.doc_bits(V.LLAMA_T0["a4"], blen, docs)[0]
        gap = b_t0 - V.doc_bits(V.LLAMA_F0, blen, docs)[0]
        d = V.delta(b_t0, V.doc_bits(xdir, blen, docs)[0], b_t0, gap)
        right_t0 = V.correctness(V.llama_dirs(V.LLAMA_T0), tok)
        right_x = V.correctness({p: xdir for p in V.GENERAL}, tok)
        excl = both_halves(split)
        general = {}
        for p in V.GENERAL:
            it = [i for i in right_x[p] if i not in excl.get(p, set())]
            t, y = np.array([right_t0[p][i] for i in it]), np.array([right_x[p][i] for i in it])
            general[p] = {**V.paired_ok(t, y), "n": len(it), "t0_right": int(t.sum()), "x_right": int(y.sum())}
        s4i = {p: split["s4"][p] for p in V.ITEM_PANELS}
        ref = np.concatenate([[right_t0[p][i] for i in s4i[p]] for p in V.ITEM_PANELS]).astype(bool)
        xx = np.concatenate([[right_x[p][i] for i in s4i[p]] for p in V.ITEM_PANELS]).astype(bool)
        c48dir = Path(load(RES / "c50_t1_run.json")["panels"]["docs"]["dir"])
        right_c48 = V.correctness({p: c48dir for p in V.GENERAL}, tok)
        vs_c48 = {}
        for p in ("nq", "tqa"):
            it = [i for i in right_x[p] if i not in excl[p]]
            lost_c48 = {i for i in it if right_t0[p][i] and not right_c48[p][i]}
            lost_x = {i for i in it if right_t0[p][i] and not right_x[p][i]}
            vs_c48[p] = {"lost_c48": len(lost_c48), "lost_x": len(lost_x), "c48_lost_kept_by_x": len(lost_c48 - lost_x),
                         **V.paired_ok(np.array([right_c48[p][i] for i in it]), np.array([right_x[p][i] for i in it]))}
        gates = {"L": checks["L"], "G1": d["delta"] >= KEEP * load(RES / "c50_s4_verdict.json")["R"]["c48"],
                 "G2": general["nq"]["ok"] and general["tqa"]["ok"],
                 "G3": all(general[p]["ok"] for p in ("mmlu", "arc", "gsm8k", "chat"))}
        rec.update({"R_llama": d["delta"], "R_ci95": d["ci95"], "threshold": KEEP * load(RES / "c50_s4_verdict.json")["R"]["c48"],
                    "bpb_x": float(V.doc_bits(xdir, blen, docs)[0].sum() / V.doc_bits(xdir, blen, docs)[1].sum()),
                    "general_vs_t0": general, "vs_c48_t1": vs_c48,
                    "donor_items_vs_t0": {**V.paired_ok(ref, xx), "n": int(len(ref)), "t0_right": int(ref.sum()),
                                          "x_right": int(xx.sum())}, "gates": gates})
        rec["verdict"] = "PASS" if all(gates.values()) else "FAIL (" + ", ".join(k for k, v in gates.items() if not v) + ")"
    else:
        rec["verdict"] = "STOP: " + ", ".join(k for k, v in checks.items() if not v)
    Path(a.json).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


A1_LORA = "05e185aed7f6b7524c061806d0fcc4699e6593ff3fe6f2e6533fec6fd0ce3a54"
NULL_PANELS = ("nq", "tqa", "mmlu", "arc")


def null(a) -> int:
    """Part 2b review, addendum 1: null LoRAs of A1's per-tensor size, merged as A1 was, on the four panels A1 fails.
    The outcome rule is the registered one; this function applies it and chooses nothing else."""
    split = json.loads(V.SPLIT.read_text(encoding="utf-8"))
    a1 = load(RES / "c50_t1_part2b.json")
    a1_flips = sum(a1["general_vs_t0"][p]["b"] + a1["general_vs_t0"][p]["c"] for p in NULL_PANELS)
    rec = {"cell": "C50-T1 part 2b review, addendum 1: null controls", "a1_flips": a1_flips, "variants": {}}
    tok = V.A4.Tokenizer.from_file(V.A4.TOKENIZER)
    blen = V.byte_lengths(Path(V.A4.TOKENIZER))
    docs = split["s4"]["docs"]
    b_t0 = V.doc_bits(V.LLAMA_T0["a4"], blen, docs)[0]
    gap = b_t0 - V.doc_bits(V.LLAMA_F0, blen, docs)[0]
    right_t0 = V.correctness(V.llama_dirs(V.LLAMA_T0), tok)
    excl = both_halves(split)
    for name, nr_p, mg_p, lay_p, run_p in a.variant:
        nr, mg, lay, run = (load(Path(x)) for x in (nr_p, mg_p, lay_p, run_p))
        checks = {"null_ok": bool(nr.get("ok")) and nr["kind"] in ("rand", "rot"),
                  "null_from_a1": nr["lora_sha256"] == A1_LORA,
                  "merged_is_null": mg["lora_sha256"] == nr["out_sha256"] and mg.get("only", "all") == "all" and not mg["zero"],
                  "L": bool(lay.get("L")) and len(lay["differs"]) == 100,
                  "run_complete": all(run["panels"].get(p, {}).get("status") == "COMPLETE" for p in NULL_PANELS + ("docs",)),
                  "model_is_layout_new": Path(run["model"]) == Path(lay["new"])}
        v = {"kind": nr["kind"], "checks": checks}
        if all(checks.values()):
            xdir = Path(run["panels"]["docs"]["dir"])
            d = V.delta(b_t0, V.doc_bits(xdir, blen, docs)[0], b_t0, gap)
            right_x = V.correctness({p: xdir for p in NULL_PANELS}, tok, NULL_PANELS)   # the null run has only these
            general = {}
            for p in NULL_PANELS:
                it = [i for i in right_x[p] if i not in excl[p]]
                t, y = np.array([right_t0[p][i] for i in it]), np.array([right_x[p][i] for i in it])
                general[p] = {**V.paired_ok(t, y), "n": len(it), "t0_right": int(t.sum()), "x_right": int(y.sum())}
            flips = sum(g["b"] + g["c"] for g in general.values())
            v.update({"R_llama": d["delta"], "R_ci95": d["ci95"], "general_vs_t0": general, "flips": flips,
                      "flips_vs_a1": flips / a1_flips})
        rec["variants"][name] = v
    vs = list(rec["variants"].values())
    if not all(all(v["checks"].values()) for v in vs) or {v["kind"] for v in vs} != {"rand", "rot"}:
        rec["outcome"] = "STOP: a check failed, or the two kinds are not both present"
    else:
        g2 = lambda v: v["general_vs_t0"]["nq"]["ok"] and v["general_vs_t0"]["tqa"]["ok"]
        allp = lambda v: all(g["ok"] for g in v["general_vs_t0"].values())
        rot = next(v for v in vs if v["kind"] == "rot")
        if not all(g2(v) for v in vs):
            rec["outcome"] = ("2: an update of A1's size without its learned map already fails nq or tqa; G2 as written "
                              "measures perturbation, so the gates need a null baseline (an amendment; the choice is the user's)")
        elif all(allp(v) for v in vs) and rot["flips_vs_a1"] >= 0.5:
            rec["outcome"] = "1: the loss is specific to the learned map; the gates stand; next is option B"
        elif all(allp(v) for v in vs):
            rec["outcome"] = "1w: the nulls pass but move the panels less than half as much as A1; not matched in effect; review"
        else:
            rec["outcome"] = "3: the nulls pass nq and tqa but fail mmlu or arc; G3's failure is reachable by perturbation; review"
    Path(a.json).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


A1_Q4 = "e490f80351e336c11a7e87c4df75cde13ee3425e4bcef2b84f2f56e10f9c5369"
NROT_Q4 = "793e5dd459a6f3cf7b721e067173182b905538dcb31e65b382b453bac42ccc56"
PLANT = 30


def fresh(a) -> int:
    """Addendum 2: on fresh items, A1 against the matched null N-rot by A4's paired rule (the amended G2 and G3).
    T0 is reported beside. The rule is the registered one; this function applies it and chooses nothing else."""
    inp = Path(a.inputs)
    man = hashlib.sha256((inp / "manifest.json").read_bytes()).hexdigest()
    runs = {k: load(Path(p)) for k, p in (("t0", a.t0), ("a1", a.a1), ("nrot", a.nrot))}
    pins = {"t0": P0_PIN, "a1": A1_Q4, "nrot": NROT_Q4}
    checks = {}
    for k, r in runs.items():
        checks[f"{k}_model_pin"] = r["model_sha256"] == pins[k]
        checks[f"{k}_inputs"] = r["inputs_manifest_sha256"] == {"a4": man, "s2": man}
        checks[f"{k}_complete"] = all(r["panels"].get(p, {}).get("status") == "COMPLETE" for p in NULL_PANELS)
    rec = {"cell": "C50-T1 part 2b review, addendum 2: A1 against N-rot on fresh items", "inputs_manifest_sha256": man,
           "checks": checks}
    if all(checks.values()):
        tok = V.A4.Tokenizer.from_file(V.A4.TOKENIZER)
        right = {k: V.correctness({p: Path(r["panels"][p]["dir"]) for p in NULL_PANELS}, tok, NULL_PANELS,
                                  {p: inp for p in NULL_PANELS}) for k, r in runs.items()}
        rng = np.random.default_rng(20261025)
        res, planted = {}, {}
        for p in NULL_PANELS:
            it = sorted(right["nrot"][p])
            assert sorted(right["a1"][p]) == it == sorted(right["t0"][p]), p
            t0, a1, nr = (np.array([right[k][p][i] for i in it]) for k in ("t0", "a1", "nrot"))
            res[p] = {"n": len(it), "right": {"t0": int(t0.sum()), "a1": int(a1.sum()), "nrot": int(nr.sum())},
                      "a1_vs_nrot": V.paired_ok(nr, a1), "a1_vs_t0": V.paired_ok(t0, a1), "nrot_vs_t0": V.paired_ok(t0, nr)}
            # planted controls: the rule must pass a copy of the null and fail the null with PLANT of its right items broken
            broken = nr.copy()
            broken[rng.choice(np.flatnonzero(nr), PLANT, replace=False)] = False
            planted[p] = {"copy_ok": V.paired_ok(nr, nr.copy())["ok"], "broken_ok": V.paired_ok(nr, broken)["ok"]}
        rec.update({"panels": res, "planted": planted})
        if not all(v["copy_ok"] and not v["broken_ok"] for v in planted.values()):
            rec["verdict"] = "STOP: a planted control did not behave"
        elif all(res[p]["a1_vs_nrot"]["ok"] for p in NULL_PANELS):
            rec["verdict"] = ("PASS: A1 is not worse than a matched null on any of the four panels; step 1 of decision C "
                              "closes, and step 2 (capacity on the residual) gets its own protocol")
        else:
            rec["verdict"] = ("FAIL: A1 is worse than a matched null on " +
                              ", ".join(p for p in NULL_PANELS if not res[p]["a1_vs_nrot"]["ok"]) + "; review")
    else:
        rec["verdict"] = "STOP: " + ", ".join(k for k, v in checks.items() if not v)
    Path(a.json).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("part2b")
    for k in ("arm", "merge", "layout", "run", "lora-sha256", "json"):
        b.add_argument(f"--{k}", required=True)
    b.set_defaults(f=part2b)
    m = sub.add_parser("mve0")
    for k in ("k0", "k1", "merge", "layout", "run", "json"):
        m.add_argument(f"--{k}", required=True)
    m.set_defaults(f=mve0)
    q = sub.add_parser("part2a")
    q.add_argument("--variant", nargs=3, action="append", required=True, metavar=("NAME", "LAYOUT", "RUN"))
    q.add_argument("--json", required=True)
    q.set_defaults(f=part2a)
    n = sub.add_parser("null")
    n.add_argument("--variant", nargs=5, action="append", required=True,
                   metavar=("NAME", "NULL_REC", "MERGE", "LAYOUT", "RUN"))
    n.add_argument("--json", required=True)
    n.set_defaults(f=null)
    f = sub.add_parser("fresh")
    for k in ("inputs", "t0", "a1", "nrot", "json"):
        f.add_argument(f"--{k}", required=True)
    f.set_defaults(f=fresh)
    a = ap.parse_args()
    return a.f(a)


if __name__ == "__main__":
    raise SystemExit(main())
