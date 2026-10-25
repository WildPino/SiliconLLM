"""C50-T1 part 2b review, addendum 1, post-hoc (not registered): do the nulls lose the same items as A1?

Per panel A1 fails (both halves of the donor-gap items out): items T0 gets right and each of A1, N-rand, N-rot gets
wrong, their overlaps with A1's, and A1 against each null by A4's paired rule.

    python -I benchmarks/chatbot50/c50_t1_null_posthoc.py results/c50_t1_null_posthoc.json
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
    gi = set(split["s4"].get(p, [])) | set(split["s5"].get(p, []))   # both halves of the donor-gap items stay out
    it = sorted(i for i in r["a1"][p] if i not in gi)
    t0 = np.array([r_t0[p][i] for i in it])
    x = {k: np.array([r[k][p][i] for i in it]) for k in r}
    lost = {k: t0 & ~v for k, v in x.items()}
    won = {k: ~t0 & v for k, v in x.items()}
    nt = max(int(t0.sum()), 1)
    out[p] = {"n": len(it), "t0_right": int(t0.sum()),
              "lost": {k: int(v.sum()) for k, v in lost.items()},
              "won": {k: int(v.sum()) for k, v in won.items()},
              "lost_with_a1": {k: int((lost[k] & lost["a1"]).sum()) for k in ("rand", "rot")},
              "lost_with_a1_expected_if_independent": {k: round(float(lost[k].sum() * lost["a1"].sum() / nt), 1)
                                                      for k in ("rand", "rot")},
              "lost_rand_with_rot": int((lost["rand"] & lost["rot"]).sum()),
              "a1_vs_null": {k: V.paired_ok(x[k], x["a1"]) for k in ("rand", "rot")}}
out = {"cell": "C50-T1 null post-hoc (not registered): lost items of A1 and the nulls, A1 against each null", **out}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, indent=1))
