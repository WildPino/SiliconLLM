"""C50-T1 MVE-0 verdict, from the records only: K0, K1, L, and gate M (the merge keeps the LoRA's gain).

R_llama(T1) uses S4's 36 documents and S2's llama.cpp gap, with the S4 verdict's own functions; the general panel and
the donor-gap items are reported against S2's T0 by A4's paired rule.
Protocol: docs/research/CHATBOT50_ROADMAP_20261016/C50_T1_PROTOCOL_20261024.md

    python -I benchmarks/chatbot50/c50_t1_verdict.py mve0 --k0 K0.json --k1 K1.json --merge MERGE.json --layout L.json \
        --run RUN.json --json OUT
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_s4_verdict as V  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
P0_PIN = "44de73157f11165c0551fd06714f4f043913292a79b07683e6a34b11d3b0d2f4"
KEEP = 0.80


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
        general = {}
        for p in V.GENERAL:
            it = [i for i in right_t1[p] if i not in set(gap_items.get(p, []))]
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


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mve0")
    for k in ("k0", "k1", "merge", "layout", "run", "json"):
        m.add_argument(f"--{k}", required=True)
    m.set_defaults(f=mve0)
    a = ap.parse_args()
    return a.f(a)


if __name__ == "__main__":
    raise SystemExit(main())
