"""C50-T1 part 2a post-hoc (not registered): do the full and partial merges of c48 lose the same nq/tqa items?

    python -I benchmarks/chatbot50/c50_t1_part2a_posthoc.py results/c50_t1_part2a_posthoc.json
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_s4_verdict as V
split = json.loads(V.SPLIT.read_text(encoding="utf-8"))
tok = V.A4.Tokenizer.from_file(V.A4.TOKENIZER)
right_t0 = V.correctness(V.llama_dirs(V.LLAMA_T0), tok)
T = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/t1")
dirs = {"full": T / "t1", "attn": T / "t1_attn", "ffn": T / "t1_ffn"}
out = {}
for p in ("nq", "tqa"):
    gi = set(split["s4"][p])
    rx = {}
    for k, d in dirs.items():
        rows = V.A4.gen_rows(d, p)
        rx[k] = {m["item"]: V.qa_check(V.A4.decode(tok, rows[m["item"]]["gen"]), m["aliases"])["match"]
                 for m in V.A4.meta(V.INPUTS[p], p) if m["item"] in rows}
    it = [i for i in rx["full"] if i not in gi]
    lost = {k: {i for i in it if right_t0[p][i] and not rx[k][i]} for k in dirs}
    won = {k: {i for i in it if not right_t0[p][i] and rx[k][i]} for k in dirs}
    n_right = sum(1 for i in it if right_t0[p][i])
    out[p] = {"n_t0_right": n_right, "lost": {k: len(v) for k, v in lost.items()},
              "lost_attn&ffn": len(lost["attn"] & lost["ffn"]),
              "expected_if_independent": round(len(lost["attn"]) * len(lost["ffn"]) / n_right, 1),
              "lost_full&attn": len(lost["full"] & lost["attn"]), "lost_full&ffn": len(lost["full"] & lost["ffn"]),
              "lost_full_in_either": len(lost["full"] & (lost["attn"] | lost["ffn"])),
              "won": {k: len(v) for k, v in won.items()}, "won_attn&ffn": len(won["attn"] & won["ffn"])}
out = {"cell": "C50-T1 part 2a post-hoc (not registered): overlap of the nq/tqa items lost against T0", **out}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, indent=1))
