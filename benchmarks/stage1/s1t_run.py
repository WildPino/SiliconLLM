"""Stage 1 (a) part 2 runner: decode-only k = 2 with k2 = 16, untrained (P0) and after training (s1t_export.py), on
part 1's panel, into part 1's work directory.  It prints numbers, never a verdict: s1t_verdict.py decides.  Resumable as
s1m_run.py: a COMPLETE panel is skipped, earlier records go under "superseded".

Arms:
    top2dm16   P0; generation panels decode-only k = 2, k2 = 16; score panels are ref's outputs (prompt positions
               only, so the form does not touch them), copied as in part 1.
    trained    the trained export (its sha256 must equal --model-sha); generation panels as top2dm16; score panels run
               at k = 8 (mode none), because training changed the weights the prompt runs on.
Generation panels are checked by part 1b's rule (CONFIG, MOE decode-only, MASS); score panels by part 1's rule for
mode none.
Usage:
    python -I benchmarks/stage1/s1t_run.py --work E:/SiliconLLM_Artifacts/stage1/s1a --arms top2dm16 \
        --out benchmarks/stage1/results/s1t_run.json
    python -I benchmarks/stage1/s1t_run.py --work ... --arms trained --model <q4.gguf> --model-sha <sha256> --out ...
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1_run  # noqa: E402
import s1m_run  # noqa: E402
from s1_run import HERE, INPUTS, MAX_PROJECTED_S, MODEL, PANELS, STOP, check_inputs, sha256  # noqa: E402

K, MASS = 2, 16
s1m_run.ARMS["dec"] = ("top", K, MASS, False, True)      # the generation form of both arms
s1_run.ARMS["k8"] = ("none", 0, False, False)            # the trained arm's score panels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--model-sha", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-h", type=float, default=8.0)
    args = ap.parse_args()
    work = Path(args.work)
    man = {k: check_inputs(p) for k, p in INPUTS.items()}
    outp = Path(args.out)
    report = json.loads(outp.read_text(encoding="utf-8")) if outp.exists() else {
        "cell": "S1C", "inputs_manifest_sha256": man, "k": K, "mass": MASS, "arms": {}}
    if report["inputs_manifest_sha256"] != man:
        raise SystemExit("inputs changed since the run began")
    if not (work / "ref" / "docs.top.bin").exists():
        raise SystemExit("part 1's ref/docs.top.bin is missing from the work directory")
    report.setdefault("sessions", []).append({"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "arms": args.arms,
                                              "budget_h": args.budget_h})
    t_start = time.time()
    for arm in args.arms.split(","):
        if arm == "top2dm16":
            model, pin = MODEL
        elif arm == "trained":
            model, pin = Path(args.model), args.model_sha
        else:
            raise SystemExit(f"unknown arm {arm}")
        a = report["arms"].setdefault(arm, {"panels": {}})
        a.setdefault("model", str(model))
        a.setdefault("model_sha256", sha256(Path(model)))
        if a["model_sha256"] != pin:
            raise SystemExit(f"{arm}: model hash differs from its pin")
        (work / arm).mkdir(parents=True, exist_ok=True)
        for panel, src, tool, extra in PANELS:
            if a["panels"].get(panel, {}).get("status", "") == "COMPLETE":
                continue
            if time.time() - t_start > args.budget_h * 3600:
                a["panels"][panel] = {"status": "NOT STARTED: session budget spent"}
                continue
            if panel in a["panels"]:
                a.setdefault("superseded", {}).setdefault(panel, []).append(a["panels"][panel])
            if tool == "score" and arm == "top2dm16":
                for suffix in (".pos.bin", ".top.bin", ".json"):
                    f = work / "ref" / f"{panel}{suffix}"
                    if f.exists():
                        shutil.copyfile(f, work / arm / f"{panel}{suffix}")
                a["panels"][panel] = {"status": "COMPLETE", "copied_from_ref": True,
                                      "sha256": sha256(work / arm / f"{panel}.pos.bin")}
                outp.write_text(json.dumps(report, indent=1), encoding="utf-8")
                print(arm, panel, "COMPLETE (copied from ref)", flush=True)
                continue
            moe_args = s1_run.moe_args("k8") if tool == "score" else s1m_run.moe_args("dec")
            exe = HERE / "bin" / f"s1_llama_{tool}_avx2.exe"
            flag = "--ids" if tool == "score" else "--prompts"
            cmd = [str(exe), "--model", str(model), flag, str(INPUTS[src] / f"{panel}.tsv"),
                   "--out", str(work / arm / panel), "--threads", "6", "--ubatch", "512",
                   "--max-projected-s", str(MAX_PROJECTED_S)] + extra + moe_args
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
            if tool == "score":
                ok = len(massl) == 0 and len(config) == 1 and len(moe) == 1 and s1_run.confirms("k8", tool, config[0], moe[0])
            else:
                ok = (len(config) == 1 and len(moe) == 1 and len(massl) == 1
                      and s1m_run.confirms("dec", tool, config[0], moe[0], massl[0]))
            ok = ok and proc.returncode == 0
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
