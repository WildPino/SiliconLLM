"""Stage 1 (a) runner: fewer routed experts per token on Ling-mini, no training.  Numbers only;
c50_s2_checks.py scores each arm's items and s1_verdict.py applies the registered rule.

The panel is C50-S2's (nq, tqa, gsm8k, arc, mmlu, docs, chat) on its frozen inputs, through the stage 1 tools
(pinned llama.cpp 5b335f4, AVX2 libraries) with the expert mode of each arm.  Every arm, the reference included,
runs through the same callback path (s1_moe_hook.h).  The documents panel writes the reference's top-32 and, for
every other arm, its log-probabilities on the reference's top-32 (KL on that partition).
replay is the reference again at ubatch 256: the panel's noise under a numerically different but equivalent run.
top4d applies k = 4 to generated tokens only; its likelihood panels (arc, mmlu, docs) are prompt positions only, so
the runner copies the reference's outputs for them, identical by construction.
Resumable: a panel recorded COMPLETE is skipped; a re-run keeps the earlier record under "superseded".
Protocol: docs/research/STAGE1_20261030/S1A_FEWER_EXPERTS_PROTOCOL_20261030.md

Usage:
    python -I benchmarks/stage1/s1_run.py --work E:/SiliconLLM_Artifacts/stage1/s1a \
        --out benchmarks/stage1/results/s1a_run.json [--arms ref,top4,...] [--budget-h 30]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
MODEL = (CAMPAIGN / "gguf" / "ling-mini-2.0-a810f64-q4_k_m.gguf",
         "44de73157f11165c0551fd06714f4f043913292a79b07683e6a34b11d3b0d2f4")
INPUTS = {"a4": CAMPAIGN / "a4" / "inputs", "s2": CAMPAIGN / "s2" / "inputs"}
STOP = "156892,156895"
SEED = 20261030
P_TOPP = 0.65   # from the calibration run (s1_calibrate.py, results/s1a_calibrate.json): mean kept 3.9892 on PIQA
MOE_LAYERS = 19   # Ling-mini: 20 layers, the first dense
# arm -> (mode, k, uses p, seeded)
ARMS = {"ref": ("none", 0, False, False), "replay": ("none", 0, False, False),
        "top4": ("top", 4, False, False), "rnd4": ("random", 4, False, True), "topw4": ("topw", 4, False, False),
        "top4d": ("top", 4, False, False),
        "topp": ("topp", 0, True, False), "rndp": ("randp", 0, True, True),
        "top6": ("top", 6, False, False), "rnd6": ("random", 6, False, True),
        "top2": ("top", 2, False, False), "rnd2": ("random", 2, False, True)}
# (panel, input set, tool, extra args), in C50-S2's order
PANELS = [("nq", "s2", "generate", ["--max-new", "32"]), ("tqa", "s2", "generate", ["--max-new", "32"]),
          ("gsm8k", "a4", "generate", ["--max-new", "384"]), ("arc", "a4", "score", ["--ctx", "4096"]),
          ("mmlu", "s2", "score", ["--ctx", "4096"]), ("docs", "a4", "score", ["--ctx", "8192", "--top", "32"]),
          ("chat", "a4", "generate", ["--max-new", "256"])]
MAX_PROJECTED_S = 3 * 3600
UBATCH = {"replay": 256}
DECODE_ONLY = {"top4d"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def check_inputs(inputs: Path) -> str:
    manifest = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
    for name, rec in manifest["files"].items():
        if sha256(inputs / f"{name}.tsv") != rec["tsv_sha256"]:
            raise SystemExit(f"input {inputs / name}.tsv differs from its manifest")
    return sha256(inputs / "manifest.json")


def moe_args(arm: str) -> list[str]:
    mode, k, uses_p, seeded = ARMS[arm]
    a = ["--moe-mode", mode]
    if k:
        a += ["--moe-k", str(k)]
    if uses_p:
        if P_TOPP is None:
            raise SystemExit("P_TOPP is not registered yet")
        a += ["--moe-p", f"{P_TOPP:.2f}"]
    if seeded:
        a += ["--moe-seed", str(SEED)]
    if arm in DECODE_ONLY:
        a += ["--moe-decode-only"]
    return a


def confirms(arm: str, tool: str, config: str, moe: str) -> bool:
    """The CONFIG line must carry the registered settings and the MOE line must show the mode acting."""
    mode, k, uses_p, seeded = ARMS[arm]
    ok = ("AVX2 = 1" in config and "threads=6 " in config and f"ubatch={UBATCH.get(arm, 512)} " in config
          and "flash_attn=disabled" in config and "hook=1 " in config and f"moe_mode={mode} " in config
          and f"moe_k={k} " in config and (tool == "score" or f"stop={STOP} " in config)
          and (not uses_p or f"moe_p={P_TOPP:.4f} " in config) and (not seeded or f"moe_seed={SEED} " in config))
    if tool == "generate":
        ok = ok and f"decode_only={int(arm in DECODE_ONLY)} " in config and "override_experts=0 " in config
    else:
        ok = ok and "override_experts=0 " in config
    m = re.search(r"n_expert_used=(\d+) layers=(\d+) input_tokens=(\d+) tensors=(\d+) slots=(\d+) kept=(\d+) "
                  r"mean_kept=([\d.]+) decode_slots=(\d+) decode_mean_kept=([\d.]+)", moe)
    if not ok or not m or f"mode={mode} " not in moe:
        return False
    n_used, layers, inputs, slots = (int(m.group(i)) for i in (1, 2, 3, 5))
    c = re.search(r"items=(\d+) prompt_tokens=(\d+)", config)
    rows = inputs if c is None else inputs - int(c.group(2)) + int(c.group(1))
    if n_used != 8 or layers != MOE_LAYERS or slots != n_used * ((MOE_LAYERS - 1) * inputs + rows):
        return False   # every MoE layer saw every input token once; the last only the rows with logits
    mean_kept, d_mean = float(m.group(7)), float(m.group(9))
    if mode == "none":
        return abs(mean_kept - 8.0) < 1e-9
    if arm in DECODE_ONLY:
        return abs(d_mean - k) < 1e-9 and int(m.group(8)) > 0 and mean_kept > k
    if mode in ("top", "topw", "random"):
        return abs(mean_kept - k) < 1e-9
    return 1.0 <= mean_kept < 8.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--budget-h", type=float, default=30.0)
    args = ap.parse_args()
    work = Path(args.work)
    man = {k: check_inputs(p) for k, p in INPUTS.items()}
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(outp.read_text(encoding="utf-8")) if outp.exists() else {
        "cell": "S1A", "inputs_manifest_sha256": man, "model": str(MODEL[0]), "p_topp": P_TOPP, "seed": SEED, "arms": {}}
    if report["inputs_manifest_sha256"] != man or report["p_topp"] != P_TOPP:
        raise SystemExit("inputs or registered p changed since the run began")
    if "model_sha256" not in report:
        report["model_sha256"] = sha256(MODEL[0])
    if report["model_sha256"] != MODEL[1]:
        raise SystemExit("model hash differs from its pin")
    report.setdefault("sessions", []).append({"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "arms": args.arms,
                                              "budget_h": args.budget_h})
    t_start = time.time()
    for arm in args.arms.split(","):
        a = report["arms"].setdefault(arm, {"panels": {}})
        (work / arm).mkdir(parents=True, exist_ok=True)
        for panel, src, tool, extra in PANELS:
            if a["panels"].get(panel, {}).get("status", "") == "COMPLETE":
                continue
            if time.time() - t_start > args.budget_h * 3600:
                a["panels"][panel] = {"status": "NOT STARTED: session budget spent"}
                continue
            if panel in a["panels"]:
                a.setdefault("superseded", {}).setdefault(panel, []).append(a["panels"][panel])
            if arm in DECODE_ONLY and tool == "score":   # prompt positions only: the reference's outputs
                ref_rec = report["arms"].get("ref", {}).get("panels", {}).get(panel, {})
                if ref_rec.get("status") != "COMPLETE":
                    raise SystemExit(f"{arm} {panel}: the reference's panel must be COMPLETE first")
                for suffix in (".pos.bin", ".top.bin", ".json"):
                    src_f = work / "ref" / f"{panel}{suffix}"
                    if src_f.exists():
                        shutil.copyfile(src_f, work / arm / f"{panel}{suffix}")
                a["panels"][panel] = {"status": "COMPLETE", "copied_from_ref": True,
                                      "sha256": sha256(work / arm / f"{panel}.pos.bin")}
                outp.write_text(json.dumps(report, indent=1), encoding="utf-8")
                print(arm, panel, "COMPLETE (copied from ref)", flush=True)
                continue
            exe = HERE / "bin" / f"s1_llama_{tool}_avx2.exe"
            flag = "--ids" if tool == "score" else "--prompts"
            cmd = [str(exe), "--model", str(MODEL[0]), flag, str(INPUTS[src] / f"{panel}.tsv"),
                   "--out", str(work / arm / panel), "--threads", "6", "--ubatch", str(UBATCH.get(arm, 512)),
                   "--max-projected-s", str(MAX_PROJECTED_S)] + extra + moe_args(arm)
            if panel == "docs" and arm not in ("ref",):
                ref_top = work / "ref" / "docs.top.bin"
                if report["arms"].get("ref", {}).get("panels", {}).get("docs", {}).get("status") != "COMPLETE":
                    raise SystemExit("the reference's documents panel must be COMPLETE before any other arm's")
                cmd += ["--ref-top", str(ref_top)]
            if tool == "generate":
                cmd += ["--stop", STOP]
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            lines = proc.stdout.splitlines()
            config = [ln for ln in lines if ln.startswith("CONFIG ")]
            moe = [ln for ln in lines if ln.startswith("MOE ")]
            rec = {"cmd": cmd, "exe_sha256": sha256(exe), "returncode": proc.returncode, "wall_s": time.time() - t0,
                   "config": config, "moe": moe, "stdout_tail": lines[-3:], "stderr_tail": proc.stderr[-600:]}
            ok = (proc.returncode == 0 and len(config) == 1 and len(moe) == 1
                  and confirms(arm, tool, config[0], moe[0]))
            rec["status"] = ("COMPLETE" if ok else "STOPPED by the projected-time rule" if proc.returncode == 3
                             else "VOID: run failed or CONFIG/MOE does not confirm the registered settings")
            a["panels"][panel] = rec
            outp.write_text(json.dumps(report, indent=1), encoding="utf-8")
            print(arm, panel, rec["status"], f"{rec['wall_s']:.0f}s", moe[0] if moe else "", flush=True)
    report["sessions"][-1]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    outp.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
