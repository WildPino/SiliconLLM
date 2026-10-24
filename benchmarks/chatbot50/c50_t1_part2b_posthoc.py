"""C50-T1 part 2b post-hoc (not registered): which items A1 loses and gains against T0, and against c48's T1.

Per general panel (both halves of the donor-gap items out): A1's lost and won items against T0, how many of them c48's
T1 (MVE-0) also lost or won, and A1 against c48's T1 by A4's paired rule.

    python -I benchmarks/chatbot50/c50_t1_part2b_posthoc.py results/c50_t1_part2b_posthoc.json
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_s4_verdict as V
split = json.loads(V.SPLIT.read_text(encoding="utf-8"))
tok = V.A4.Tokenizer.from_file(V.A4.TOKENIZER)
T = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/t1")
r_t0 = V.correctness(V.llama_dirs(V.LLAMA_T0), tok)
r = {k: V.correctness({p: T / k for p in V.GENERAL + ("docs",)}, tok) for k in ("t1", "a1")}
out = {}
for p in V.GENERAL:
    gi = set(split["s4"].get(p, [])) | set(split["s5"].get(p, []))   # both halves of the donor-gap items stay out
    it = sorted(i for i in r["a1"][p] if i not in gi)
    t0, c48, a1 = (np.array([d[p][i] for i in it]) for d in (r_t0, r["t1"], r["a1"]))
    lost_a1, lost_c48 = t0 & ~a1, t0 & ~c48
    won_a1, won_c48 = ~t0 & a1, ~t0 & c48
    out[p] = {"n": len(it), "t0_right": int(t0.sum()),
              "lost": {"a1": int(lost_a1.sum()), "c48": int(lost_c48.sum()), "both": int((lost_a1 & lost_c48).sum()),
                       "expected_if_independent": round(float(lost_a1.sum() * lost_c48.sum() / max(t0.sum(), 1)), 1)},
              "won": {"a1": int(won_a1.sum()), "c48": int(won_c48.sum()), "both": int((won_a1 & won_c48).sum())},
              "a1_vs_c48": V.paired_ok(c48, a1)}


def margins(arm: Path, name: str) -> dict[int, float]:
    """T0's margin on a likelihood item: the runner-up's mean NLL per token minus the label's (> 0 iff right)."""
    recs = np.fromfile(arm / f"{name}.pos.bin", dtype=V.A4.REC)
    by = {}
    for d in np.unique(recs["doc"]):
        sel = recs["doc"] == d
        by[int(d)] = (recs["pos"][sel], recs["logp_target"][sel].astype(np.float64))
    res = {}
    for m in V.A4.meta(V.INPUTS[name], name):
        if m["item"] * 8 not in by:
            continue
        nll = []
        for k, slen in enumerate(m["suffix_lens"]):
            pos, lp = by[m["item"] * 8 + k]
            nll.append(float(-lp[pos >= m["prefix_len"]].sum()) / slen)
        res[m["item"]] = min(x for j, x in enumerate(nll) if j != m["label"]) - nll[m["label"]]
    return res


# Are the lost items T0's fragile ones? T0's margin on its right items, lost by A1 against kept.
for p in ("mmlu", "arc"):
    mg = margins(V.llama_dirs(V.LLAMA_T0)[p], p)
    gi = set(split["s4"].get(p, [])) | set(split["s5"].get(p, []))
    right = [i for i in r["a1"][p] if i not in gi and r_t0[p][i]]
    lost = np.array([mg[i] for i in right if not r["a1"][p][i]])
    kept = np.array([mg[i] for i in right if r["a1"][p][i]])
    q = lambda a: [round(float(x), 4) for x in np.quantile(a, [0.25, 0.5, 0.75])]
    out[p]["t0_margin"] = {"lost_by_a1_q25_50_75": q(lost), "kept_q25_50_75": q(kept),
                           "kept_share_below_lost_median": round(float((kept < np.median(lost)).mean()), 3)}
out = {"cell": "C50-T1 part 2b post-hoc (not registered): A1's lost and won items against T0 and c48's T1", **out}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, indent=1))
