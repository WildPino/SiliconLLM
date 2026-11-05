"""Stage 2 test 2b-bis: the verdict, by the registered rule.  Cell 1 of test 2b with the tail a * preview + b, on the 96 new documents.

    PASS: THE HEAD READS IN TWO PASSES ON NEW TEXT     every line below holds
    FAIL: THE TAIL DOES NOT HOLD ON NEW TEXT           dBPB or KL out
    FAIL: THE WORD STAYS OUT OF THE CANDIDATES         miss out
Line: miss <= 1 per 1,000 (tool's count and float), KL top-32 <= 0.02, KL over the vocabulary <= 0.02, dBPB CI95 upper <= +0.005, chat >= 27,
degenerate <= 1.  VOID: the replay differs, chat's answers differ from test 2b's, the float dBPB is more than 0.001 from the tool's, the planted
tail stays inside the line, or the tail's change does nothing (the registered offset reads within 0.0005 of the affine tail).

`second_label` is test 2b's, so that test 3b's verdict can read this file as its --head.

Usage: python -I benchmarks/stage2/s2u_verdict.py --run F --two-pass F --json F
       python -I benchmarks/stage2/s2u_verdict.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
sys.path.insert(0, str(HERE.parents[0] / "chatbot50"))
sys.path.insert(0, str(HERE))

WORK = Path("E:/SiliconLLM_Artifacts/stage2/s2u")
STAGE1 = Path("E:/SiliconLLM_Artifacts/stage1/s1a")
ARM = "two_pass_a"
LINE_MISS, LINE_KL, LINE_DBPB, CHAT_MIN, DEGEN_MAX = 1.0, 0.02, 0.005, 27, 1
PARITY, MOVES = 0.001, 0.0005


def decide(fl: dict, panel: dict, miss_tool: float, chat: dict, replay: bool, chat_equal: bool, second: str) -> dict:
    t = fl["tails"]
    c = {"replay": replay, "chat_equal": chat_equal, "float_parity": abs(t["affine"]["dbpb"] - panel["dbpb"]) <= PARITY,
         "plant_out": t["plant"]["ci95"][1] > LINE_DBPB, "tail_moves": abs(t["offset"]["dbpb"] - t["affine"]["dbpb"]) > MOVES}
    miss_ok = miss_tool <= LINE_MISS and fl["miss_per_1000"] <= LINE_MISS
    tail_ok = panel["kl_top32_nats"] <= LINE_KL and t["affine"]["kl_full_vocab"] <= LINE_KL and panel["dbpb_ci95"][1] <= LINE_DBPB
    chat_ok = chat["passes"] >= CHAT_MIN and chat["degenerate"] <= DEGEN_MAX
    out = {"controls": c, "miss_tool_per_1000": miss_tool, "miss_float_per_1000": fl["miss_per_1000"], "docs": panel, "chat": chat,
           "kl_full_vocab": t["affine"]["kl_full_vocab"], "second_label": second}
    if not all(c.values()):
        out["outcome"] = "VOID"
    elif not miss_ok:
        out["outcome"] = "FAIL: THE WORD STAYS OUT OF THE CANDIDATES"
    elif not (tail_ok and chat_ok):
        out["outcome"] = "FAIL: THE TAIL DOES NOT HOLD ON NEW TEXT"
    else:
        out["outcome"] = "PASS: THE HEAD READS IN TWO PASSES ON NEW TEXT"
    return out


def selftest() -> int:
    def fl(aff=0.003, off=0.009, plant=0.05, miss=0.0, kl=0.006):
        def r(v):
            return {"dbpb": v, "ci95": [v - 0.001, v + 0.001], "kl_full_vocab": kl}
        return {"miss_per_1000": miss, "tails": {"affine": r(aff), "offset": r(off), "plant": r(plant)}}

    def pn(d=0.003, up=0.004, kl=0.0005):
        return {"dbpb": d, "dbpb_ci95": [d - 0.001, up], "kl_top32_nats": kl}
    ch, s = {"passes": 28, "degenerate": 0}, "x"
    cases = [("pass", (fl(), pn(), 0.0, ch, True, True, s), "PASS: THE HEAD READS IN TWO PASSES ON NEW TEXT"),
             ("dbpb", (fl(aff=0.006), pn(0.006, 0.0062), 0.0, ch, True, True, s), "FAIL: THE TAIL DOES NOT HOLD ON NEW TEXT"),
             ("kl", (fl(kl=0.03), pn(), 0.0, ch, True, True, s), "FAIL: THE TAIL DOES NOT HOLD ON NEW TEXT"),
             ("chat", (fl(), pn(), 0.0, {"passes": 26, "degenerate": 0}, True, True, s), "FAIL: THE TAIL DOES NOT HOLD ON NEW TEXT"),
             ("miss", (fl(), pn(), 1.2, ch, True, True, s), "FAIL: THE WORD STAYS OUT OF THE CANDIDATES"),
             ("replay", (fl(), pn(), 0.0, ch, False, True, s), "VOID"), ("chat differs", (fl(), pn(), 0.0, ch, True, False, s), "VOID"),
             ("parity", (fl(aff=0.0045), pn(0.003), 0.0, ch, True, True, s), "VOID"), ("plant in", (fl(plant=0.002), pn(), 0.0, ch, True, True, s), "VOID"),
             ("tail still", (fl(off=0.0032), pn(), 0.0, ch, True, True, s), "VOID")]
    ok = True
    for name, args, want in cases:
        got = decide(*args)["outcome"]
        print("ok " if got == want else "BAD", name, got)
        ok &= got == want
    print("selftest", "PASS" if ok else "FAIL", len(cases))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    for x in ("run", "two-pass", "json"):
        ap.add_argument(f"--{x}", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import re
    from s1_verdict import docs_metrics
    from s2h_verdict import chat
    r = json.loads(a.run.read_text(encoding="utf-8"))
    t2 = json.loads(a.two_pass.read_text(encoding="utf-8"))
    line = r["panels"]["docs"]["two_pass"][0]
    pos, missed = (int(x) for x in re.search(r"positions=(\d+) missed=(\d+)", line).groups())
    rows = lambda p: [json.loads(x)["gen"] for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]  # noqa: E731
    out = decide(r["float"]["reserved"], docs_metrics(WORK, ARM), 1000 * missed / pos, chat(WORK, ARM), r["replay"]["equal_to_two_pass"],
                 rows(WORK / ARM / "chat.jsonl") == rows(STAGE1 / "two_pass" / "chat.jsonl"), t2["second_label"])
    out["reported"] = {k: r["float"][k] for k in r["float"]}
    print(json.dumps({k: v for k, v in out.items() if k != "reported"}, indent=1))
    a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
