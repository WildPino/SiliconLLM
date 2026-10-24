"""C50-T1: the general panels against T0 with both halves of the donor-gap items left out (correction, 24 October).

S4's verdict defined a general panel as every item outside S4's half of the donor-gap items, so S5's half (items
where llama.cpp T0 is wrong and F0 right) stayed inside it. Those items can only add to c (X right, T0 wrong) and
make "not worse" easier. Here the general panels exclude both halves; no gap item of either half is read.

    python -I benchmarks/chatbot50/c50_t1_general_clean.py --s4-evals DIR --json OUT
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_s4_verdict as V  # noqa: E402

T1 = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/t1")


def general(right_t0: dict, right_x: dict, gap: dict, panels) -> dict:
    out = {}
    for p in panels:
        it = [i for i in right_x[p] if i not in gap.get(p, set())]
        t, y = np.array([right_t0[p][i] for i in it]), np.array([right_x[p][i] for i in it])
        out[p] = {**V.paired_ok(t, y), "n": len(it), "t0_right": int(t.sum()), "x_right": int(y.sum())}
    return out


def qa_only(d: Path, tok) -> dict:
    res = {}
    for p in ("nq", "tqa"):
        rows = V.A4.gen_rows(d, p)
        res[p] = {m["item"]: V.qa_check(V.A4.decode(tok, rows[m["item"]]["gen"]), m["aliases"])["match"]
                  for m in V.A4.meta(V.INPUTS[p], p) if m["item"] in rows}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s4-evals", type=Path, required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    split = json.loads(V.SPLIT.read_text(encoding="utf-8"))
    tok = V.A4.Tokenizer.from_file(V.A4.TOKENIZER)
    gap = {p: set(split["s4"][p]) | set(split["s5"][p]) for p in V.ITEM_PANELS}
    ll_t0 = V.correctness(V.llama_dirs(V.LLAMA_T0), tok)
    rec = {"cell": "C50-T1 general panels, both halves of the donor-gap items excluded",
           "excluded": {p: len(v) for p, v in gap.items()}, "llama_vs_llama_t0": {}, "pytorch_vs_pytorch_t0": {}}
    rec["llama_vs_llama_t0"]["t1_full"] = general(ll_t0, V.correctness({p: T1 / "t1" for p in V.GENERAL}, tok),
                                                  gap, V.GENERAL)
    for v in ("attn", "ffn"):
        rec["llama_vs_llama_t0"][f"t1_{v}"] = general(ll_t0, qa_only(T1 / f"t1_{v}", tok), gap, ("nq", "tqa"))
    pt_t0 = V.correctness({p: a.s4_evals / "t0" for p in V.GENERAL}, tok)
    for x in ("c48", "c496", "a", "b"):
        rec["pytorch_vs_pytorch_t0"][x] = general(pt_t0, V.correctness({p: a.s4_evals / x for p in V.GENERAL}, tok),
                                                  gap, V.GENERAL)
    Path(a.json).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    for side, arms in rec.items():
        if not isinstance(arms, dict) or side == "excluded":
            continue
        for x, g in arms.items():
            print(side, x, " ".join(f"{p}:{v['x_right']}/{v['t0_right']}(b{v['b']},c{v['c']}){'' if v['ok'] else '!'}"
                                    for p, v in g.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
