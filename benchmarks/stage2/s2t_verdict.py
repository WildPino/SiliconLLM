"""Stage 2 test 2b verdict, by the rule registered in S2D_TWO_PASS_HEAD_PROTOCOL_20261104.md.  It reads the sweeps, the
compositions and the panels, and chooses nothing: the cells (preview, K, tail offset) were named in the protocol.

Recall is judged at each cell's own K and at 4,096, on three sets: the panel's documents, the generated positions of ref's
answers, and the documents on the model with ternary experts.  On the documents the miss that counts is the one the panel
tool itself counted, on llama.cpp's kernels (TWOPASS); the verdict checks it equals 1 - top-1 agreement.

Controls: C1 the planted preview misses (sweep); C2 the trivial list and a preview just under the line miss, or recall does
not decide alone; C3 the float recomputation agrees with the panel tool on each set; C4 the replay (K = the vocabulary)
equals ref on its documents, and the planted two-pass panel falls.

Usage: python -I benchmarks/stage2/s2t_verdict.py --selftest
       python -I benchmarks/stage2/s2t_verdict.py --docs F --gen F --tern F --compose F --compose-w F --run F --json F
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
sys.path.insert(0, str(HERE))

LINE_MISS = 1.0                # per 1,000: the stage's own line
CELLS = {"two_pass": ("tern_tr", 1024), "two_pass_w": ("wsvd256", 4096)}
TAIL = 1
LINE_KL, LINE_KL_FULL, LINE_DBPB = 0.02, 0.02, 0.005
CHAT_MIN, DEGEN_MAX = 27, 1
PLANT_MISS = 900.0
PARITY_MISS = 0.1              # per 1,000 of the panel tool's top-1 outside the float head's first 16
PLANT_TOP1 = 0.5
NEAR = "wsvd128"               # a preview just under the line on dev (2.34): the line must be able to reject it
IDENTITY = 1e-9


def miss_at(sweep: dict, name: str, k: int, stratum: str) -> float:
    return sweep["previews"][name][str(k)]["miss_per_1000"][stratum]


def judge_cell(cell: str, docs: dict, gen: dict, tern: dict, panel: dict, chat: dict, comp: dict, kernel: dict) -> dict:
    name, k = CELLS[cell]
    miss = {"docs_kernel": kernel["miss_per_1000"],
            "docs_4096": miss_at(docs, name, 4096, "all"),
            "generated": miss_at(gen, name, k, "generated"), "generated_4096": miss_at(gen, name, 4096, "generated"),
            "tern": miss_at(tern, name, k, "all"), "tern_4096": miss_at(tern, name, 4096, "all")}
    full = next(c for c in comp["cells"] if c["k"] == k and c["tail_offset"] == TAIL)
    identity = abs(1 - panel["top1_agreement"] - kernel["miss_per_1000"] / 1000) <= IDENTITY
    recall = all(v <= LINE_MISS for v in miss.values())
    panels = (panel["kl_top32_nats"] <= LINE_KL and full["kl_nats"] <= LINE_KL_FULL and panel["dbpb_ci95"][1] <= LINE_DBPB
              and chat["passes"] >= CHAT_MIN and chat["degenerate"] <= DEGEN_MAX)
    return {"cell": {"preview": name, "k": k, "tail_offset": TAIL}, "miss_per_1000": miss, "identity": identity,
            "miss_kernel_ci95": kernel["ci95"], "kl_full_vocab": full["kl_nats"], "recall_within": recall,
            "panels_within": panels, "docs": panel, "chat": chat}


def decide(docs: dict, gen: dict, tern: dict, first: dict, second: dict, controls: dict) -> dict:
    plant = miss_at(docs, "shuf", 4096, "all")
    c1 = plant >= PLANT_MISS
    parity = {"docs": miss_at(docs, "exact", 16, "all"), "gen": miss_at(gen, "exact", 16, "all"),
              "tern": miss_at(tern, "exact", 16, "all")}
    c3 = all(v <= PARITY_MISS for v in parity.values())
    c4 = controls["replay_equal"] and controls["plant_top1"] < PLANT_TOP1
    freq, near = miss_at(docs, "freq", 4096, "all"), miss_at(docs, NEAR, 4096, "all")
    c2 = freq > LINE_MISS and near > LINE_MISS
    ident = first["identity"] and second["identity"]
    out = {"controls": {"C1_planted_miss": plant, "C1": c1, "C2_freq_miss": freq, "C2_near_miss": near, "C2": c2,
                        "C3_parity": parity, "C3": c3, "C4": c4, "identity": ident},
           "first": first, "second": second}
    if not (c1 and c3 and c4 and ident):
        out["outcome"] = "VOID"
    elif not first["recall_within"]:
        out["outcome"] = "FAIL: THE WORD STAYS OUT OF THE CANDIDATES"
    elif not first["panels_within"]:
        out["outcome"] = "FAIL: THE PANELS LEAVE THE LINE"
    elif not c2:
        out["outcome"] = "PASS, AND A WEAKER LIST PASSES TOO: RECALL DOES NOT DECIDE ALONE"
    else:
        out["outcome"] = "PASS: THE HEAD READS IN TWO PASSES"
    out["second_label"] = ("THE UNTRAINED PREVIEW PASSES TOO" if second["recall_within"] and second["panels_within"]
                           else "ONLY THE TRAINED PREVIEW IS WITHIN THE LINE")
    return out


def selftest() -> int:
    global CELLS
    CELLS = {"two_pass": ("p", 1024), "two_pass_w": ("w", 4096)}

    def sweep(p=0.0, w=0.5, plant=970.0, freq=30.0, near=2.0, par=0.0):
        def row(v, g=None):
            return {"miss_per_1000": {"all": v, "generated": v if g is None else g}}
        return {"previews": {"p": {"1024": row(p), "4096": row(p)}, "w": {"1024": row(w), "4096": row(w)},
                             "shuf": {"4096": row(plant)}, "exact": {"16": row(par)}, "freq": {"4096": row(freq)},
                             NEAR: {"4096": row(near)}}}
    pn = {"top1_agreement": 0.9995, "kl_top32_nats": 0.001, "dbpb_ci95": [0.0, 0.001]}
    ch = {"passes": 28, "degenerate": 0}
    co = {"cells": [{"k": 1024, "tail_offset": 1, "kl_nats": 0.004}, {"k": 4096, "tail_offset": 1, "kl_nats": 0.012}]}
    ke = {"miss_per_1000": 0.5, "ci95": [0.1, 0.9]}
    ct = {"replay_equal": True, "plant_top1": 0.1}

    def case(d=None, g=None, t=None, p1=None, c1=None, k1=None, co1=None, p2=None, c2=None, k2=None, ctl=None):
        d, g, t = d or sweep(), g or sweep(), t or sweep()
        f = judge_cell("two_pass", d, g, t, p1 or pn, c1 or ch, co1 or co, k1 or ke)
        s = judge_cell("two_pass_w", d, g, t, p2 or pn, c2 or ch, co, k2 or ke)
        return decide(d, g, t, f, s, ctl or ct)
    o = {"pass": "PASS: THE HEAD READS IN TWO PASSES", "word": "FAIL: THE WORD STAYS OUT OF THE CANDIDATES",
         "panel": "FAIL: THE PANELS LEAVE THE LINE", "void": "VOID",
         "alone": "PASS, AND A WEAKER LIST PASSES TOO: RECALL DOES NOT DECIDE ALONE"}
    bad = {"miss_per_1000": 1.5, "ci95": [1, 2]}
    cases = [
        (case(), o["pass"]),
        (case(d=sweep(p=1.5)), o["word"]),
        (case(g=sweep(p=1.5)), o["word"]),
        (case(t=sweep(p=1.5)), o["word"]),
        (case(p1=dict(pn, top1_agreement=0.9985), k1=bad), o["word"]),
        (case(p1=dict(pn, kl_top32_nats=0.05)), o["panel"]),
        (case(p1=dict(pn, dbpb_ci95=[0.0, 0.02])), o["panel"]),
        (case(c1=dict(ch, passes=20)), o["panel"]),
        (case(c1=dict(ch, degenerate=3)), o["panel"]),
        (case(co1={"cells": [{"k": 1024, "tail_offset": 1, "kl_nats": 0.05}, {"k": 4096, "tail_offset": 1, "kl_nats": 0.012}]}), o["panel"]),
        (case(d=sweep(plant=500.0)), o["void"]),
        (case(g=sweep(par=5.0)), o["void"]),
        (case(t=sweep(par=5.0)), o["void"]),
        (case(p1=dict(pn, top1_agreement=0.9)), o["void"]),          # the panel's top-1 and the tool's count disagree
        (case(ctl=dict(ct, replay_equal=False)), o["void"]),
        (case(ctl=dict(ct, plant_top1=0.9)), o["void"]),
        (case(d=sweep(freq=0.5)), o["alone"]),
        (case(d=sweep(near=0.5)), o["alone"]),
    ]
    for i, (got, want) in enumerate(cases):
        if got["outcome"] != want:
            print("FAIL", i, got["outcome"], "!=", want)
            return 1
    second = [(case(), "THE UNTRAINED PREVIEW PASSES TOO"),
              (case(d=sweep(w=1.5)), "ONLY THE TRAINED PREVIEW IS WITHIN THE LINE"),
              (case(t=sweep(w=1.5)), "ONLY THE TRAINED PREVIEW IS WITHIN THE LINE"),
              (case(p2=dict(pn, kl_top32_nats=0.05)), "ONLY THE TRAINED PREVIEW IS WITHIN THE LINE"),
              (case(c2=dict(ch, passes=20)), "ONLY THE TRAINED PREVIEW IS WITHIN THE LINE")]
    for i, (got, want) in enumerate(second):
        if got["second_label"] != want or got["outcome"] != o["pass"]:
            print("FAIL second", i, got["second_label"], got["outcome"])
            return 1
    print(f"selftest PASS, {len(cases) + len(second)} cases")
    return 0


def kernel_miss(work: Path, arm: str, run_arm: dict) -> dict:
    """The panel tool's own count, and its CI95 by document from the records against ref."""
    import numpy as np
    from s2t_head import REC, boot_ci
    a = np.fromfile(work / arm / "docs.pos.bin", REC)
    r = np.fromfile(work / "ref" / "docs.pos.bin", REC)
    if len(a) != len(r) or not np.array_equal(a["doc"], r["doc"]):
        raise SystemExit(f"{arm}: records do not align with ref")
    line = run_arm["panels"]["docs"]["two_pass"][0]
    f = dict(x.split("=") for x in line.split()[1:])
    if int(f["positions"]) != len(a):
        raise SystemExit(f"{arm}: TWOPASS counted other positions")
    _, inv = np.unique(a["doc"], return_inverse=True)
    wrong = a["top1"] != r["top1"]
    if int(wrong.sum()) != int(f["missed"]):
        raise SystemExit(f"{arm}: the records' misses differ from TWOPASS")
    return {"miss_per_1000": 1000 * int(f["missed"]) / len(a),
            "ci95": boot_ci(np.bincount(inv, weights=wrong), np.bincount(inv))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    for x in ("docs", "gen", "tern", "compose", "compose-w", "run", "json"):
        ap.add_argument(f"--{x}", type=Path)
    ap.add_argument("--work", type=Path, default=Path("E:/SiliconLLM_Artifacts/stage1/s1a"))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    from s1_verdict import docs_metrics
    from s2h_verdict import chat
    from s2t_run import PREVIEWS

    def read(p: Path) -> dict:
        return json.loads(p.read_text(encoding="utf-8"))
    docs, gen, tern, run = read(a.docs), read(a.gen), read(a.tern), read(a.run)
    comp = {"two_pass": read(a.compose), "two_pass_w": read(a.compose_w)}
    res = {}
    for arm, (name, k) in CELLS.items():
        cell = run["arms"][arm]
        if (cell["k"], cell["tail_offset"], cell["preview_sha256"]) != (k, TAIL, PREVIEWS[name][1]) or cell.get("docs_limit"):
            raise SystemExit(f"{arm}: the run's cell is not the registered one")
        if comp[arm]["set"] != "docs" or comp[arm]["preview"] != name:
            raise SystemExit(f"{arm}: the composition is not this cell's")
        res[arm] = judge_cell(arm, docs, gen, tern, docs_metrics(a.work, arm), chat(a.work, arm), comp[arm],
                              kernel_miss(a.work, arm, cell))
    out = decide(docs, gen, tern, res["two_pass"], res["two_pass_w"], run["controls"])
    print(json.dumps(out, indent=1))
    a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
