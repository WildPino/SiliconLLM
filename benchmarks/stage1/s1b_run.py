"""Stage 1 (b) runner: Ling-mini with ternary routed experts (s1t_train.py --ternary, s1t_export.py --ternary) on part
1's panel, into part 1's work directory.  It prints numbers, never a verdict: s1b_verdict.py decides.  Resumable as
s1t_run.py: a COMPLETE panel is skipped, earlier records go under "superseded".

Every panel runs at k = 8 (mode none): the experts change, not the routing.  One arm per call:
    --arm NAME --model <gguf> --model-sha <sha256> [--panels all|docs]
Registered arms (S1B protocol): tern (all panels), tern_ph and tern_l10 (documents only).
Each panel is checked by part 1's rule for mode none (CONFIG and MOE lines).
Usage:
    python -I benchmarks/stage1/s1b_run.py --work E:/SiliconLLM_Artifacts/stage1/s1a --arm tern --model <q4.gguf> \
        --model-sha <sha256> --out benchmarks/stage1/results/s1b_run.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1_run  # noqa: E402
from s1_run import HERE, INPUTS, MAX_PROJECTED_S, PANELS, STOP, check_inputs, sha256  # noqa: E402

s1_run.ARMS["k8"] = ("none", 0, False, False)
ARMS = {"tern": "all", "tern_ph": "docs", "tern_l10": "docs"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-sha", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-h", type=float, default=8.0)
    args = ap.parse_args()
    work = Path(args.work)
    man = {k: check_inputs(p) for k, p in INPUTS.items()}
    outp = Path(args.out)
    report = json.loads(outp.read_text(encoding="utf-8")) if outp.exists() else {
        "cell": "S1B-ternary", "inputs_manifest_sha256": man, "arms": {}}
    if report["inputs_manifest_sha256"] != man:
        raise SystemExit("inputs changed since the run began")
    if not (work / "ref" / "docs.top.bin").exists():
        raise SystemExit("part 1's ref/docs.top.bin is missing from the work directory")
    report.setdefault("sessions", []).append({"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "arm": args.arm,
                                              "budget_h": args.budget_h})
    t_start = time.time()
    arm, model = args.arm, Path(args.model)
    a = report["arms"].setdefault(arm, {"panels": {}, "panel_set": ARMS[arm]})
    a.setdefault("model", str(model))
    a.setdefault("model_sha256", sha256(model))
    if a["model_sha256"] != args.model_sha:
        raise SystemExit(f"{arm}: model hash differs from its pin")
    (work / arm).mkdir(parents=True, exist_ok=True)
    for panel, src, tool, extra in PANELS:
        if ARMS[arm] == "docs" and panel != "docs":
            continue
        if a["panels"].get(panel, {}).get("status", "") == "COMPLETE":
            continue
        if time.time() - t_start > args.budget_h * 3600:
            a["panels"][panel] = {"status": "NOT STARTED: session budget spent"}
            continue
        if panel in a["panels"]:
            a.setdefault("superseded", {}).setdefault(panel, []).append(a["panels"][panel])
        exe = HERE / "bin" / f"s1_llama_{tool}_avx2.exe"
        flag = "--ids" if tool == "score" else "--prompts"
        cmd = [str(exe), "--model", str(model), flag, str(INPUTS[src] / f"{panel}.tsv"),
               "--out", str(work / arm / panel), "--threads", "6", "--ubatch", "512",
               "--max-projected-s", str(MAX_PROJECTED_S)] + extra + s1_run.moe_args("k8")
        if panel == "docs":
            cmd += ["--ref-top", str(work / "ref" / "docs.top.bin")]
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
              and s1_run.confirms("k8", tool, config[0], moe[0]))
        rec["status"] = ("COMPLETE" if ok else "STOPPED by the projected-time rule" if proc.returncode == 3
                         else "VOID: run failed or CONFIG/MOE does not confirm the registered settings")
        a["panels"][panel] = rec
        outp.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(arm, panel, rec["status"], f"{rec['wall_s']:.0f}s", flush=True)
    report["sessions"][-1]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    outp.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
