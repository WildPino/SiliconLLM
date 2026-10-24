"""C50-T1 MVE-0 runner: one merged model through S2's seven panels, with S2's executables, settings and checks.

Numbers only; `c50_t1_verdict.py` reads the outputs. Resumable: a COMPLETE panel is not re-run.
Protocol: docs/research/CHATBOT50_ROADMAP_20261016/C50_T1_PROTOCOL_20261024.md

    python -I benchmarks/chatbot50/c50_t1_run.py --model T1.gguf --pin SHA256 --work E:/.../t1 --out results/c50_t1_run.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c50_s2_run import A4_WORK, HERE, MAX_PROJECTED_S, PANELS, STOP, check_inputs, sha256  # noqa: E402

S2_INPUTS = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/s2/inputs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--pin", required=True, help="the model's sha256, recorded when it was built")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", default="t1")
    ap.add_argument("--panels", default=",".join(p for p, *_ in PANELS), help="a subset of S2's panels, in S2's order")
    a = ap.parse_args()
    inputs = {"a4": A4_WORK / "inputs", "s2": S2_INPUTS}
    man = {k: check_inputs(p) for k, p in inputs.items()}
    rep = json.loads(a.out.read_text(encoding="utf-8")) if a.out.exists() else {
        "cell": "C50-T1 MVE-0 run", "inputs_manifest_sha256": man, "arm": a.arm, "model": str(a.model), "panels": {}}
    if rep["inputs_manifest_sha256"] != man:
        raise SystemExit("inputs changed since the run began")
    rep.setdefault("model_sha256", sha256(a.model))
    if rep["model_sha256"] != a.pin:
        raise SystemExit("model hash differs from its pin")
    want = a.panels.split(",")
    assert set(want) <= {p for p, *_ in PANELS}, want
    print("CONFIG", json.dumps({"model": str(a.model), "sha256": a.pin, "panels": want}), flush=True)
    out = a.work / a.arm
    out.mkdir(parents=True, exist_ok=True)
    for panel, src, tool, extra in PANELS:
        if panel not in want:
            continue
        if rep["panels"].get(panel, {}).get("status") == "COMPLETE":
            continue
        if panel in rep["panels"]:
            rep.setdefault("superseded", {}).setdefault(panel, []).append(rep["panels"][panel])
        exe = HERE / "bin" / f"c50_llama_{tool}_avx2.exe"
        flag = "--ids" if tool == "score" else "--prompts"
        cmd = [str(exe), "--model", str(a.model), flag, str(inputs[src] / f"{panel}.tsv"), "--out", str(out / panel),
               "--threads", "6", "--ubatch", "512", "--max-projected-s", str(MAX_PROJECTED_S)] + extra
        if tool == "generate":
            cmd += ["--stop", STOP]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        lines = proc.stdout.splitlines()
        config = [ln for ln in lines if ln.startswith("CONFIG ")]
        c = config[0] if len(config) == 1 else ""
        ok = (proc.returncode == 0 and "AVX2 = 1" in c and "threads=6 " in c and "ubatch=512 " in c
              and "flash_attn=disabled" in c and (tool == "score" or f"stop={STOP} " in c))
        rec = {"cmd": cmd, "exe_sha256": sha256(exe), "returncode": proc.returncode, "wall_s": time.time() - t0,
               "config": config, "stdout_tail": lines[-3:], "stderr_tail": proc.stderr[-600:], "dir": str(out),
               "status": "COMPLETE" if ok else "STOPPED by the projected-time rule" if proc.returncode == 3
               else "VOID: run failed or CONFIG does not confirm the registered settings"}
        rep["panels"][panel] = rec
        a.out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
        print(a.arm, panel, rec["status"], f"{rec['wall_s']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
