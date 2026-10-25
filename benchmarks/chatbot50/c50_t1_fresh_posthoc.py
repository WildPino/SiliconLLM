"""C50-T1 part 2b review, addendum 2, post-hoc (not registered): does leaving the donor-gap items out bias the old panels?

The old general panels leave out both halves of S2's donor-gap items: the items F0 gets right and T0 wrong. Those
are where an update toward F0 can win an item, so leaving them out removes c (merge right, T0 wrong) but not b.
This reads the same old records twice, with the donor-gap items out (as the gates did) and in (a plain draw, as
the fresh items are), for A1, N-rand and N-rot against T0, by A4's paired rule; and S4's arms the same way, with
only S4's half of those items, the one S4's evaluator produced.

    python -I benchmarks/chatbot50/c50_t1_fresh_posthoc.py results/c50_t1_fresh_posthoc.json
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_s4_verdict as V
P = ("nq", "tqa", "mmlu", "arc")
split = json.loads(V.SPLIT.read_text(encoding="utf-8"))
tok = V.A4.Tokenizer.from_file(V.A4.TOKENIZER)
T = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/t1")
r_t0 = V.correctness(V.llama_dirs(V.LLAMA_T0), tok, P)
r = {k: V.correctness({p: d for p in P}, tok, P) for k, d in
     (("a1", T / "a1"), ("rand", T / "null" / "n_rand"), ("rot", T / "null" / "n_rot"))}
out = {}
for p in P:
    gi = set(split["s4"].get(p, [])) | set(split["s5"].get(p, []))
    out[p] = {}
    for name, keep in (("gap_out", lambda i: i not in gi), ("gap_in", lambda i: True)):
        it = sorted(i for i in r["a1"][p] if keep(i) and i in r_t0[p])
        t0 = np.array([r_t0[p][i] for i in it])
        row = {"n": len(it), "gap_items": sum(1 for i in it if i in gi), "t0_right": int(t0.sum())}
        for k in r:
            x = np.array([r[k][p][i] for i in it])
            b, c = int((t0 & ~x).sum()), int((~t0 & x).sum())
            row[k] = {"right": int(x.sum()), "b": b, "c": c, "ok": bool(b - c <= max(2 * (b + c) ** 0.5, 1))}
        out[p][name] = row

# S4's arms against the PyTorch T0, on S4's evaluator records: only S4's half of the donor-gap items was produced,
# so "gap_in" here puts back that half (the S5 half is never produced) and is still partly censored.
E = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/s4/verdict/evals")
S4P = P + ("gsm8k",)
rs = {n: V.correctness({p: E / n for p in S4P + ("chat",)}, tok, S4P) for n in ("t0", "c48", "c496", "a", "b")}
s4 = {}
for p in S4P:
    gi = set(split["s4"].get(p, []))
    s4[p] = {}
    for name, keep in (("gap_out", lambda i: i not in gi), ("s4_half_in", lambda i: True)):
        it = sorted(i for i in rs["a"][p] if keep(i))
        t0 = np.array([rs["t0"][p][i] for i in it])
        row = {"n": len(it), "gap_items": sum(1 for i in it if i in gi), "t0_right": int(t0.sum())}
        for k in ("c48", "c496", "a", "b"):
            x = np.array([rs[k][p][i] for i in it])
            b, c = int((t0 & ~x).sum()), int((~t0 & x).sum())
            row[k] = {"right": int(x.sum()), "b": b, "c": c, "ok": bool(b - c <= max(2 * (b + c) ** 0.5, 1))}
        s4[p][name] = row

out = {"cell": "C50-T1 fresh post-hoc (not registered): old panels with the donor-gap items out and in",
       "t1": out, "s4": s4}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
for tag, blk in (("T1", out["t1"]), ("S4", s4)):
    for p, d in blk.items():
        for name, o in d.items():
            print(tag, p, name, o["n"], o["gap_items"], "T0", o["t0_right"],
                  *(f"{k} {v['right']} ({v['b']},{v['c']}) {'ok' if v['ok'] else 'WORSE'}"
                    for k, v in o.items() if isinstance(v, dict)))
