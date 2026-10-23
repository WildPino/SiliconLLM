"""C50-S4, after the verdict: numbers the protocol did not register, reported and never gated.

The verdict judges the capacity arms against their LoRA controls; it does not judge the controls against T0. Here,
for every trained arm X: R(X) with its CI95 (the verdict's bootstrap, with T0 as the control), and X against T0 on
the S4 donor-gap items and on each general panel, by A4's paired rule (the verdict's own functions).

    python -I benchmarks/chatbot50/c50_s4_posthoc.py --evals DIR --json OUT
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_s4_verdict as V  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=Path, required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    split = json.loads(V.SPLIT.read_text(encoding="utf-8"))
    tok = V.A4.Tokenizer.from_file(V.A4.TOKENIZER)
    blen = V.byte_lengths(Path(V.A4.TOKENIZER))
    docs = split["s4"]["docs"]
    gap = V.doc_bits(V.LLAMA_T0["a4"], blen, docs)[0] - V.doc_bits(V.LLAMA_F0, blen, docs)[0]
    names = ["c48", "c496", "a", "b"]
    bits = {n: V.doc_bits(a.evals / n, blen, docs)[0] for n in ["t0"] + names}
    right = {n: V.correctness({p: a.evals / n for p in V.GENERAL}, tok) for n in ["t0"] + names}
    gap_items = {p: split["s4"][p] for p in V.ITEM_PANELS}
    out = {}
    for x in names:
        d = V.delta(bits["t0"], bits[x], bits["t0"], gap)
        ref = np.concatenate([[right["t0"][p][i] for i in gap_items[p]] for p in V.ITEM_PANELS])
        xx = np.concatenate([[right[x][p][i] for i in gap_items[p]] for p in V.ITEM_PANELS])
        general = {}
        for p in V.GENERAL:
            it = [i for i in right[x][p] if i not in set(gap_items.get(p, []))]
            t, y = np.array([right["t0"][p][i] for i in it]), np.array([right[x][p][i] for i in it])
            general[p] = {**V.paired_ok(t, y), "n": len(it), "t0_right": int(t.sum()), "x_right": int(y.sum())}
        out[x] = {"R": d["delta"], "R_ci95": d["ci95"],
                  "donor_items_vs_t0": {**V.paired_ok(ref.astype(bool), xx.astype(bool)), "n": int(len(ref)),
                                        "t0_right": int(ref.sum()), "x_right": int(xx.sum())},
                  "general_vs_t0": general}
    rec = {"cell": "C50-S4 post-hoc (not registered, not gated)", "arms": out}
    Path(a.json).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    for x, r in out.items():
        g = " ".join(f"{p}:{v['x_right']}/{v['t0_right']}{'' if v['ok'] else '!'}" for p, v in r["general_vs_t0"].items())
        di = r["donor_items_vs_t0"]
        print(f"{x:5s} R {r['R']:+.4f} ci95 [{r['R_ci95'][0]:+.4f}, {r['R_ci95'][1]:+.4f}]  donor {di['x_right']}/{di['t0_right']} "
              f"of {di['n']}  general (X/T0, ! = paired-rule regression) {g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
