"""C50-S2 runner: the gap-map arms through the pinned llama.cpp tools on the same IDs.  Numbers only;
c50_s2_checks.py scores each arm and c50_s2_verdict.py applies the rule.  Resumable as the A4 runner.

Arms: m0 (Ling-mini Q4_K_M, no imatrix), f0 (Ling-flash-2.0 Q4_K_M, no imatrix, the same recipe), t0 (P0).
Panels: the A4 inputs (docs, arc, gsm8k, chat) and the S2 inputs (nq, tqa, mmlu).  t0's A4 panels are not re-run:
the runner reuses A4's P0 outputs after checking that A4 recorded them COMPLETE with the same executable and the same
input manifest.
Protocol: docs/research/CHATBOT50_ROADMAP_20261016/C50_S2_GAP_MAP_PROTOCOL_20261017.md

Usage:
    python -I benchmarks/chatbot50/c50_s2_run.py --work E:/.../s2 --out benchmarks/chatbot50/results/c50_s2_run.json \
        [--arms t0,m0,f0] [--budget-h 20]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
GGUF = CAMPAIGN / "gguf"
A4_WORK, A4_RUN = CAMPAIGN / "a4", HERE / "results" / "c50_a4_run.json"
MODELS = {"t0": (GGUF / "ling-mini-2.0-a810f64-q4_k_m.gguf", "44de73157f11165c0551fd06714f4f043913292a79b07683e6a34b11d3b0d2f4"),
          "m0": (GGUF / "ling-mini-2.0-a810f64-q4_k_m-noimatrix.gguf", "1f54536855ab10520d6b21c73986cc7e8cfc2f8ad394a2da766872168022b77e"),
          # F0 is staged on the NVMe C: (ROADMAP §12, 17 October): 63 GB load faster than from the USB disk E:
          "f0": (Path("C:/SiliconLLM_stage/s2/ling-flash-2.0-96f9b7a-q4_k_m-noimatrix.gguf"), "5348c73fe6497c5152e16fd792d6c7b5aeefa15de8b15b8c8e18bef691704b8e")}
STOP = "156892,156895"
# (panel, input set, tool, extra args), in run order: the decision panels first
PANELS = [("nq", "s2", "generate", ["--max-new", "32"]), ("tqa", "s2", "generate", ["--max-new", "32"]),
          ("gsm8k", "a4", "generate", ["--max-new", "384"]), ("arc", "a4", "score", ["--ctx", "4096"]),
          ("mmlu", "s2", "score", ["--ctx", "4096"]), ("docs", "a4", "score", ["--ctx", "8192"]),
          ("chat", "a4", "generate", ["--max-new", "256"])]
MAX_PROJECTED_S = 3 * 3600
# Addendum 1 (18 October, before F0's second session): the tools project a process's time from its first item, which
# on F0 carries the cold start of a 63 GB model.  For the arms listed here the tool's projection is off and the runner
# bounds each process by a wall-time limit instead; the other arms keep the registered guard.  A panel re-run after a
# non-COMPLETE record keeps that record under "superseded".
WALL_LIMIT_S = {"f0": 6 * 3600}


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="t0,m0,f0")
    ap.add_argument("--budget-h", type=float, default=20.0)
    ap.add_argument("--a4-inputs", default=str(A4_WORK / "inputs"))
    args = ap.parse_args()
    work = Path(args.work)
    inputs = {"a4": Path(args.a4_inputs), "s2": work / "inputs"}
    man = {k: check_inputs(p) for k, p in inputs.items()}
    outp = Path(args.out)
    report = json.loads(outp.read_text(encoding="utf-8")) if outp.exists() else {
        "cell": "C50-S2", "inputs_manifest_sha256": man, "arms": {}}
    if report["inputs_manifest_sha256"] != man:
        raise SystemExit("inputs changed since the run began")
    report.setdefault("sessions", []).append({"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "arms": args.arms,
                                              "budget_h": args.budget_h, "wall_limit_s": WALL_LIMIT_S})
    a4 = json.loads(A4_RUN.read_text(encoding="utf-8")) if A4_RUN.exists() else None
    t_start = time.time()
    for arm in args.arms.split(","):
        model, pin = MODELS[arm]
        a = report["arms"].setdefault(arm, {"model": str(model), "panels": {}})
        if "model_sha256" not in a:
            a["model_sha256"] = sha256(model)
        if a["model_sha256"] != pin:
            raise SystemExit(f"{arm}: model hash differs from its pin")
        (work / arm).mkdir(parents=True, exist_ok=True)
        for panel, src, tool, extra in PANELS:
            if a["panels"].get(panel, {}).get("status", "").startswith("COMPLETE"):
                continue
            exe = HERE / "bin" / f"c50_llama_{tool}_avx2.exe"
            if arm == "t0" and src == "a4":   # reuse A4's P0 outputs: same model, executable and inputs
                rec = (a4 or {}).get("arms", {}).get("p0", {}).get("panels", {}).get(panel, {})
                same = (a4 and a4.get("inputs_manifest_sha256") == man["a4"] and rec.get("status") == "COMPLETE"
                        and rec.get("exe_sha256") == sha256(exe))
                a["panels"][panel] = {"status": "COMPLETE (reused from A4 p0)" if same else "VOID: A4 p0 output not reusable",
                                      "dir": str(A4_WORK / "p0"), "a4_record": rec}
                outp.write_text(json.dumps(report, indent=1), encoding="utf-8")
                print(arm, panel, a["panels"][panel]["status"], flush=True)
                continue
            if time.time() - t_start > args.budget_h * 3600:
                a["panels"][panel] = {"status": "NOT STARTED: session budget spent"}
                continue
            if panel in a["panels"]:
                a.setdefault("superseded", {}).setdefault(panel, []).append(a["panels"][panel])
            limit = WALL_LIMIT_S.get(arm)
            flag = "--ids" if tool == "score" else "--prompts"
            cmd = [str(exe), "--model", str(model), flag, str(inputs[src] / f"{panel}.tsv"), "--out", str(work / arm / panel),
                   "--threads", "6", "--ubatch", "512", "--max-projected-s", "0" if limit else str(MAX_PROJECTED_S)] + extra
            if tool == "generate":
                cmd += ["--stop", STOP]
            t0 = time.time()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=limit)
                rc, out, err = proc.returncode, proc.stdout, proc.stderr
            except subprocess.TimeoutExpired as e:   # the child is killed; its partial outputs stay on disk
                rc = "timeout"
                out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
                err = e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            lines = out.splitlines()
            config = [ln for ln in lines if ln.startswith("CONFIG ")]
            rec = {"cmd": cmd, "exe_sha256": sha256(exe), "returncode": rc, "wall_s": time.time() - t0,
                   "config": config, "stdout_tail": lines[-3:], "stderr_tail": err[-600:], "dir": str(work / arm),
                   "guard": {"wall_limit_s": limit} if limit else {"max_projected_s": MAX_PROJECTED_S}}
            c = config[0] if len(config) == 1 else ""
            ok = (rc == 0 and "AVX2 = 1" in c and "threads=6 " in c and "ubatch=512 " in c
                  and "flash_attn=disabled" in c and (tool == "score" or f"stop={STOP} " in c))
            rec["status"] = ("COMPLETE" if ok else "STOPPED by the projected-time rule" if rc == 3
                             else "STOPPED by the wall-time limit (addendum 1)" if rc == "timeout"
                             else "VOID: run failed or CONFIG does not confirm the registered settings")
            a["panels"][panel] = rec
            outp.write_text(json.dumps(report, indent=1), encoding="utf-8")
            print(arm, panel, rec["status"], f"{rec['wall_s']:.0f}s", flush=True)
    report["sessions"][-1]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    outp.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
