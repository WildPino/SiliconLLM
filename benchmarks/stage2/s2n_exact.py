"""S2N amendment 4: the trained stack with the top expert read exact where it weighs, CPU only.  Two points of the ms / recall curve.

    sx19  e2e_rowx's file (organs IQ2_KS and bal experts, both with e2e_rowx's scales), top slot exact from P0 in layers 1-19 (x19k4's form, 3.92 desk ms)
    sx9   the same, layers 9-19 (x9's layers, 2.27 desk ms)
Both generate at k = 4 (mass 16, decode only) as e2e_rowx; arc, mmlu and docs score at k = 8 with the same exact slot.  The slot is test 3b's hook
(s2x_mix_hook.h) in the tools with unaligned moves (S2K, amendment 3), unchanged: each process's CONFIG and MIX lines must confirm model, mode and layers.

run      the seven panels of one arm (resumable: a COMPLETE panel is skipped).
verdict  numbers and labels, chosen by nothing else (paired bootstrap, stage 1's seed and draws):
    Per arm: stage 1's label and chat gate (WITHIN BUDGET needs CI95 lower >= -5, chat >= ref - 4, degenerate <= ref + 3).
    Slot on the stack, sx - e2e_rowx, all panels and recall (mean of nq and tqa): PAYS (lower > 0) / COSTS (upper < 0) / NOT SHOWN.
    Depth, sx19 - sx9 on recall and on all panels: DEPTH PAYS (lower > 0) / DEPTH COSTS (upper < 0) / NOT SHOWN.
    Carry-over, all(sx19) - [all(rowx_org) + all(x19k4) - all(top4dm16)]: BELOW THE PARTS (upper < 0) / ABOVE THE PARTS (lower > 0) / ADDS UP.
    Desk: ms and tok/s per arm, from the S2M row (ks_stk4, 43.05 ms) plus the slot's ms; points and recall per ms of slot.
    VOID if an arm's model or P0 differs from its pin, a panel is not COMPLETE, or a MIX line does not confirm the slot.

Usage: python -I benchmarks/stage2/s2n_exact.py run --arm sx19|sx9 --out benchmarks/stage2/results/s2n_exact_run.json
       python -I benchmarks/stage2/s2n_exact.py verdict --json benchmarks/stage2/results/s2n_exact_verdict.json
       python -I benchmarks/stage2/s2n_exact.py selftest
       python -I benchmarks/stage2/s2n_exact.py chain      (each arm's panels and scores, then the verdict; resumable)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "stage1"))
from s2x_run import GEN, INPUTS, MAX_PROJECTED_S, PANELS, SCORE, STOP, WORK, check_inputs, launch, sha256  # noqa: E402
from s1_verdict import CHAT_MARGIN, DEGEN_MARGIN, DRAWS, PANELS as JUDGED, SEED, degenerate_count, label, points, vec  # noqa: E402

MODEL = Path("E:/SiliconLLM_Artifacts/stage2/s2n/q4/e2e_rowx.gguf")
MODEL_SHA = "cc047475472ab952729824894037a94e345c71724c07d06ae206d31331e387f1"     # s2n_export_e2e_rowx.json
P0 = Path("C:/SiliconLLM_fast/s2t/m/p0.gguf")     # copied back to the SSD on 2026-11-27: read from E: (HDD) the hook trips the time guard
P0_SHA = "44de73157f11165c0551fd06714f4f043913292a79b07683e6a34b11d3b0d2f4"         # s2x_run.P0_SHA
ARMS = {"sx19": "1-19", "sx9": "9-19"}
DESK_STACK, DESK_SLOT, LINE_MS = 43.05, {"sx19": 3.92, "sx9": 2.27}, 40.49      # S2M protocol; S2F protocol; 24.7 tok/s
UA = {"score": SCORE.with_name(SCORE.name.replace("_avx2", "_ua_avx2")), "generate": GEN.with_name(GEN.name.replace("_avx2", "_ua_avx2"))}
RECALL = ("nq", "tqa")
SCORES = Path("E:/SiliconLLM_Artifacts/stage1/s1a/scores")
ROOT = HERE.parents[1]
A4, S2 = "E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/a4/inputs", "E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/s2/inputs"


def command(arm: str, panel: str, src: str, tool: str, extra: list[str]) -> tuple[list[str], list[str]]:
    flag = "--ids" if tool == "score" else "--prompts"
    layers = ARMS[arm]
    cmd = [str(UA[tool]), "--model", str(MODEL), flag, str(INPUTS[src] / f"{panel}.tsv"), "--out", str(WORK / arm / panel), "--threads", "6",
           "--ubatch", "512", "--max-projected-s", str((6 if tool == "score" else 3) * MAX_PROJECTED_S)] + extra   # the hook's one-off P0 load sits in the first document's time
    cmd += ["--exact-from", str(P0), "--exact-mode", "top", "--exact-layers", layers]
    k4 = tool == "generate"
    cmd += ["--moe-mode", "top", "--moe-k", "4", "--moe-mass", "16", "--moe-decode-only"] if k4 else ["--moe-mode", "none"]
    if panel == "docs":
        cmd += ["--ref-top", str(WORK / "ref" / "docs.top.bin")]
    if tool == "generate":
        cmd += ["--stop", STOP]
    need = [f"model={MODEL} ", "flash_attn=disabled", "moe_k=4 " if k4 else "moe_mode=none ", "exact_mode=top ", f"exact_layers={layers} ",
            f"exact_from={P0} ", "hook=1 "]
    return cmd, need


def mix_ok(r: dict) -> str:
    if r["status"] != "COMPLETE":
        return r["status"]
    if len(r["mix"]) != 1:
        return "STOPPED: no MIX line"
    if "mode=top " not in r["mix"][0]:
        return "STOPPED: MIX does not confirm the mode"
    return "COMPLETE"


def run(arm: str, out: Path) -> int:
    report = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"cell": "S2N-amendment-4-exact-slot", "arms": {}}
    man = {k: check_inputs(p) for k, p in INPUTS.items()}
    if report.setdefault("inputs_manifest_sha256", man) != man:
        raise SystemExit("inputs changed since the run began")
    if sha256(MODEL) != MODEL_SHA or sha256(P0) != P0_SHA:
        raise SystemExit("a model differs from its pin")
    rec = report["arms"].setdefault(arm, {"panels": {}, "model": str(MODEL), "model_sha256": MODEL_SHA, "exact_from": str(P0), "p0_sha256": P0_SHA,
                                          "layers": ARMS[arm], "mode": "top"})
    (WORK / arm).mkdir(parents=True, exist_ok=True)
    for panel, src, tool, extra in PANELS:
        if rec["panels"].get(panel, {}).get("status") == "COMPLETE":
            continue
        cmd, need = command(arm, panel, src, tool, extra)
        r = launch(cmd, need)
        r["status"] = mix_ok(r)
        rec["panels"][panel] = r
        out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(arm, panel, r["status"], r["wall_s"], r["mix"], flush=True)
        if r["status"] != "COMPLETE":
            return 1
    return 0


def ci(v) -> list[float]:
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def three(c: list[float], pos: str, neg: str, mid: str) -> str:
    return pos if c[0] > 0 else neg if c[1] < 0 else mid


NAMES = ("ref", "top4dm16", "x19k4", "e2e_rowx", "rowx_org", "sx19", "sx9")


def judge(arms: dict, deg: dict, valid: dict) -> dict:
    """arms: name -> panel -> bool vector; deg: name -> degenerate count; valid: sx arm -> bool."""
    n = {p: len(arms["ref"][p]) for p in JUDGED}
    for x in NAMES:
        if {p: len(arms[x][p]) for p in JUDGED} != n:
            raise SystemExit(f"{x} does not score the same items as ref")

    def rec(x, idx=None):
        return float(np.mean([100.0 * (arms[x][p] if idx is None else arms[x][p][idx[p]]).mean() for p in RECALL]))
    rng = np.random.default_rng(SEED)
    keys = ("d_sx19", "d_sx9", "all_sx19", "all_sx9", "rec_sx19", "rec_sx9", "depth_rec", "depth_all", "carry")
    b = {k: [] for k in keys}
    for _ in range(DRAWS):
        idx = {p: rng.integers(0, n[p], n[p]) for p in JUDGED}
        a = {x: points(arms[x], idx) for x in NAMES}
        r = {x: rec(x, idx) for x in ("e2e_rowx", "sx19", "sx9")}
        for s in ("sx19", "sx9"):
            b[f"d_{s}"].append(a[s] - a["ref"])
            b[f"all_{s}"].append(a[s] - a["e2e_rowx"])
            b[f"rec_{s}"].append(r[s] - r["e2e_rowx"])
        b["depth_rec"].append(r["sx19"] - r["sx9"])
        b["depth_all"].append(a["sx19"] - a["sx9"])
        b["carry"].append(a["sx19"] - (a["rowx_org"] + a["x19k4"] - a["top4dm16"]))
    a = {x: points(arms[x]) for x in NAMES}
    r = {x: rec(x) for x in NAMES}
    chat = {x: int(arms[x]["chat"].sum()) for x in NAMES}
    ok = all(valid.values())
    out = {"valid": valid, "all_vs_ref": {x: a[x] - a["ref"] for x in NAMES[1:]}, "recall_vs_ref": {x: r[x] - r["ref"] for x in NAMES[1:]},
           "chat": chat, "degenerate": deg, "arms": {}}
    for s in ("sx19", "sx9"):
        c = ci(b[f"d_{s}"])
        chat_ok = chat[s] >= chat["ref"] - CHAT_MARGIN and deg[s] <= deg["ref"] + DEGEN_MARGIN
        ms = DESK_STACK + DESK_SLOT[s]
        lab = label(c) if ok else "VOID"
        out["arms"][s] = {"delta": a[s] - a["ref"], "ci95": c, "label": lab, "chat_ok": chat_ok,
                          "fits": ok and lab in ("WITHIN SHARE", "WITHIN BUDGET") and chat_ok,
                          "slot_all": {"gain": a[s] - a["e2e_rowx"], "ci95": ci(b[f"all_{s}"]),
                                       "label": three(ci(b[f"all_{s}"]), "PAYS", "COSTS", "NOT SHOWN") if ok else "VOID"},
                          "slot_recall": {"gain": r[s] - r["e2e_rowx"], "ci95": ci(b[f"rec_{s}"]),
                                          "label": three(ci(b[f"rec_{s}"]), "PAYS", "COSTS", "NOT SHOWN") if ok else "VOID"},
                          "desk_ms": ms, "desk_tok_s": 1000.0 / ms, "over_line_ms": ms - LINE_MS,
                          "points_per_slot_ms": (a[s] - a["e2e_rowx"]) / DESK_SLOT[s], "recall_per_slot_ms": (r[s] - r["e2e_rowx"]) / DESK_SLOT[s]}
    out["depth"] = {"recall": r["sx19"] - r["sx9"], "recall_ci95": ci(b["depth_rec"]), "all": a["sx19"] - a["sx9"], "all_ci95": ci(b["depth_all"]),
                    "label": three(ci(b["depth_rec"]), "DEPTH PAYS", "DEPTH COSTS", "NOT SHOWN") if ok else "VOID"}
    out["carry_over"] = {"value": a["sx19"] - (a["rowx_org"] + a["x19k4"] - a["top4dm16"]), "ci95": ci(b["carry"]),
                         "estimate": a["rowx_org"] + a["x19k4"] - a["top4dm16"] - a["ref"],
                         "label": three(ci(b["carry"]), "ABOVE THE PARTS", "BELOW THE PARTS", "ADDS UP") if ok else "VOID"}
    out["desk"] = {"stack_ms": DESK_STACK, "stack_tok_s": 1000.0 / DESK_STACK, "line_ms": LINE_MS}
    return out


def verdict(results: Path) -> dict:
    arms = {x: {p: vec(json.loads((SCORES / f"{x}.json").read_text(encoding="utf-8")), p) for p in JUDGED} for x in NAMES}
    deg = {x: degenerate_count(WORK / x) for x in NAMES}
    run_ = json.loads((results / "s2n_exact_run.json").read_text(encoding="utf-8"))["arms"]
    valid = {}
    for s in ARMS:
        ru = run_.get(s, {})
        p = ru.get("panels", {})
        valid[s] = bool(ru.get("model_sha256") == MODEL_SHA and ru.get("p0_sha256") == P0_SHA and ru.get("layers") == ARMS[s]
                        and all(p.get(x[0], {}).get("status") == "COMPLETE" for x in PANELS)
                        and all(f"exact_layers={ARMS[s]} " in p[x[0]]["config"][0] for x in PANELS))
    return judge(arms, deg, valid)


def selftest() -> int:
    rng = np.random.default_rng(1)
    n = {"nq": 300, "tqa": 300, "gsm8k": 100, "arc": 300, "mmlu": 300, "chat": 40}
    base = {p: rng.random(k) < 0.5 for p, k in n.items()}

    def shift(v, p_flip, up):                    # flip a share of the items one way
        out = {}
        for p, x in v.items():
            x = x.copy()
            m = rng.random(len(x)) < p_flip
            x[m] = up
            out[p] = x
        return out
    arms = {x: {p: v.copy() for p, v in base.items()} for x in NAMES}
    arms["ref"] = shift(base, 0.15, True)
    arms["sx19"] = shift(base, 0.25, True)        # the slot pays
    deg = {x: 0 for x in NAMES}
    o = judge(arms, deg, {"sx19": True, "sx9": True})
    cases = [(o["arms"]["sx19"]["slot_all"]["label"], "PAYS"), (o["arms"]["sx9"]["slot_all"]["label"], "NOT SHOWN"),
             (o["depth"]["label"], "DEPTH PAYS"), (o["arms"]["sx9"]["slot_recall"]["label"], "NOT SHOWN")]
    arms2 = dict(arms, sx19=shift(base, 0.25, False))  # the slot costs, and sits below the parts
    o2 = judge(arms2, deg, {"sx19": True, "sx9": True})
    cases += [(o2["arms"]["sx19"]["slot_all"]["label"], "COSTS"), (o2["carry_over"]["label"], "BELOW THE PARTS")]
    o3 = judge(arms, dict(deg, sx19=9), {"sx19": True, "sx9": False})
    cases += [(o3["arms"]["sx19"]["label"], "VOID"), (o3["depth"]["label"], "VOID"), (o["arms"]["sx19"]["chat_ok"], True)]
    o4 = judge(arms, dict(deg, sx19=9), {"sx19": True, "sx9": True})
    cases += [(o4["arms"]["sx19"]["chat_ok"], False), (o4["arms"]["sx19"]["fits"], False)]
    cmd, need = command("sx9", "nq", "s2", "generate", ["--max-new", "32"])
    cases += [("9-19" in cmd and "--moe-k" in cmd and str(P0) in cmd, True), ("exact_layers=9-19 " in need, True)]
    cmd, need = command("sx19", "arc", "a4", "score", ["--ctx", "4096"])
    cases += [("--moe-k" in cmd, False), ("moe_mode=none " in need and "exact_layers=1-19 " in need, True)]
    fake = {"status": "COMPLETE", "mix": []}
    cases += [(mix_ok(fake), "STOPPED: no MIX line"), (mix_ok(dict(fake, mix=["MIX mode=random layers=11"])), "STOPPED: MIX does not confirm the mode"),
              (mix_ok(dict(fake, mix=["MIX mode=top layers=11"])), "COMPLETE")]
    cases += [(abs(o["arms"]["sx19"]["desk_ms"] - 46.97) < 1e-9, True), (abs(o["arms"]["sx9"]["desk_ms"] - 45.32) < 1e-9, True)]
    bad = [i for i, (x, y) in enumerate(cases) if x != y]
    print("selftest", "PASS" if not bad else f"FAIL {bad} {[cases[i] for i in bad]}", len(cases))
    return 0 if not bad else 1


def step(*cmd: str) -> None:
    print(">", " ".join(cmd), flush=True)
    if subprocess.run([sys.executable, "-I", *cmd], cwd=ROOT).returncode:
        raise SystemExit(f"stopped: {' '.join(cmd)}")


def chain() -> None:
    R = HERE / "results"
    for arm in ARMS:
        if not (SCORES / f"{arm}.json").exists():
            step("benchmarks/stage2/s2n_exact.py", "run", "--arm", arm, "--out", str(R / "s2n_exact_run.json"))
            step("benchmarks/chatbot50/c50_s2_checks.py", "--a4-inputs", A4, "--s2-inputs", S2, "--arm-a4", str(WORK / arm), "--arm-s2", str(WORK / arm),
                 "--out", str(SCORES / f"{arm}.json"))
    step("benchmarks/stage2/s2n_exact.py", "verdict", "--json", str(R / "s2n_exact_verdict.json"))
    print("S2N EXACT DONE", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "verdict", "selftest", "chain"))
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--results", type=Path, default=HERE / "results")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "chain":
        chain()
        return 0
    if a.cmd == "run":
        return run(a.arm, a.out)
    out = verdict(a.results)
    a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
