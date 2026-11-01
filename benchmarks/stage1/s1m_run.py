"""Stage 1 (a) part 1b runner: k = 4 with the routing mass decoupled from the kept experts (--moe-mass), on the same
panel as part 1, into part 1's work directory (its ref outputs give the documents' top-32 partition and top4dm16's
prompt-only panels).  It prints numbers, never a verdict: s1m_verdict.py decides.  Resumable: a COMPLETE panel is
skipped, every earlier record of a panel it reruns is kept under "superseded".

Usage:
    python -I benchmarks/stage1/s1m_run.py --work E:/SiliconLLM_Artifacts/stage1/s1a \
        --out benchmarks/stage1/results/s1m_run.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s1_run import (HERE, INPUTS, MAX_PROJECTED_S, MODEL, MOE_LAYERS, PANELS, SEED, STOP,  # noqa: E402
                    check_inputs, sha256)

# arm -> (mode, k, mass_k, seeded, decode_only), in run order
ARMS = {"top4m16": ("top", 4, 16, False, False), "rnd4m16": ("random", 4, 16, True, False),
        "top4m8": ("top", 4, 8, False, False), "top4dm16": ("top", 4, 16, False, True)}


def moe_args(arm: str) -> list[str]:
    mode, k, mass, seeded, dec = ARMS[arm]
    a = ["--moe-mode", mode, "--moe-k", str(k), "--moe-mass", str(mass)]
    if seeded:
        a += ["--moe-seed", str(SEED)]
    if dec:
        a += ["--moe-decode-only"]
    return a


def confirms(arm: str, tool: str, config: str, moe: str, massl: str) -> bool:
    """CONFIG must carry the registered settings; MOE must show 4 kept (on decode only for the decode-only arm) over
    every layer; MASS must count one row per MoE layer per token, with factor 1 at K = 8 and below 1 at K = 16."""
    mode, k, mass, seeded, dec = ARMS[arm]
    ok = ("AVX2 = 1" in config and "threads=6 " in config and "ubatch=512 " in config
          and "flash_attn=disabled" in config and "hook=1 " in config and f"moe_mode={mode} " in config
          and f"moe_k={k} " in config and f"moe_mass={mass} " in config and "override_experts=0 " in config
          and (tool == "score" or f"stop={STOP} " in config) and (not seeded or f"moe_seed={SEED} " in config)
          and (tool == "score" or f"decode_only={int(dec)} " in config))
    m = re.search(r"n_expert_used=(\d+) layers=(\d+) input_tokens=(\d+) tensors=(\d+) slots=(\d+) kept=(\d+) "
                  r"mean_kept=([\d.]+) decode_slots=(\d+) decode_mean_kept=([\d.]+)", moe)
    s = re.search(r"MASS mass_k=(\d+) rows=(\d+) mean_factor=([\d.]+)", massl)
    if not ok or not m or not s or f"mode={mode} " not in moe:
        return False
    n_used, layers, inputs, slots = (int(m.group(i)) for i in (1, 2, 3, 5))
    c = re.search(r"items=(\d+) prompt_tokens=(\d+)", config)
    rows = inputs if c is None else inputs - int(c.group(2)) + int(c.group(1))
    if n_used != 8 or layers != MOE_LAYERS or slots != n_used * ((MOE_LAYERS - 1) * inputs + rows):
        return False
    if int(s.group(1)) != mass or int(s.group(2)) * n_used != slots:
        return False
    factor = float(s.group(3))
    if (mass == 8 and factor != 1.0) or (mass > 8 and not 0.0 < factor < 1.0):
        return False
    if dec:
        return abs(float(m.group(9)) - k) < 1e-9 and int(m.group(8)) > 0 and float(m.group(7)) > k
    return abs(float(m.group(7)) - k) < 1e-9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--budget-h", type=float, default=12.0)
    args = ap.parse_args()
    work = Path(args.work)
    man = {k: check_inputs(p) for k, p in INPUTS.items()}
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(outp.read_text(encoding="utf-8")) if outp.exists() else {
        "cell": "S1B", "inputs_manifest_sha256": man, "model": str(MODEL[0]), "seed": SEED, "arms": {}}
    if report["inputs_manifest_sha256"] != man:
        raise SystemExit("inputs changed since the run began")
    if "model_sha256" not in report:
        report["model_sha256"] = sha256(MODEL[0])
    if report["model_sha256"] != MODEL[1]:
        raise SystemExit("model hash differs from its pin")
    for f in ("docs.pos.bin", "docs.top.bin"):
        if not (work / "ref" / f).exists():
            raise SystemExit(f"part 1's ref/{f} is missing from the work directory")
    report.setdefault("sessions", []).append({"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "arms": args.arms,
                                              "budget_h": args.budget_h})
    t_start = time.time()
    for arm in args.arms.split(","):
        a = report["arms"].setdefault(arm, {"panels": {}})
        (work / arm).mkdir(parents=True, exist_ok=True)
        dec = ARMS[arm][4]
        for panel, src, tool, extra in PANELS:
            if a["panels"].get(panel, {}).get("status", "") == "COMPLETE":
                continue
            if time.time() - t_start > args.budget_h * 3600:
                a["panels"][panel] = {"status": "NOT STARTED: session budget spent"}
                continue
            if panel in a["panels"]:
                a.setdefault("superseded", {}).setdefault(panel, []).append(a["panels"][panel])
            if dec and tool == "score":   # prompt positions only: part 1's reference outputs
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
                   "--out", str(work / arm / panel), "--threads", "6", "--ubatch", "512",
                   "--max-projected-s", str(MAX_PROJECTED_S)] + extra + moe_args(arm)
            if panel == "docs":
                cmd += ["--ref-top", str(work / "ref" / "docs.top.bin")]
            if tool == "generate":
                cmd += ["--stop", STOP]
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            lines = proc.stdout.splitlines()
            config = [ln for ln in lines if ln.startswith("CONFIG ")]
            moe = [ln for ln in lines if ln.startswith("MOE ")]
            massl = [ln for ln in lines if ln.startswith("MASS ")]
            rec = {"cmd": cmd, "exe_sha256": sha256(exe), "returncode": proc.returncode, "wall_s": time.time() - t0,
                   "config": config, "moe": moe, "mass": massl, "stdout_tail": lines[-3:],
                   "stderr_tail": proc.stderr[-600:]}
            ok = (proc.returncode == 0 and len(config) == 1 and len(moe) == 1 and len(massl) == 1
                  and confirms(arm, tool, config[0], moe[0], massl[0]))
            rec["status"] = ("COMPLETE" if ok else "STOPPED by the projected-time rule" if proc.returncode == 3
                             else "VOID: run failed or CONFIG/MOE/MASS does not confirm the registered settings")
            a["panels"][panel] = rec
            outp.write_text(json.dumps(report, indent=1), encoding="utf-8")
            print(arm, panel, rec["status"], f"{rec['wall_s']:.0f}s", massl[0] if massl else "", flush=True)
    report["sessions"][-1]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    outp.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
