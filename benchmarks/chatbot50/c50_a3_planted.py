"""C50-A3 planted-fault control of the system gate (A3 addendum 3).

Each planted binary is c50_engine.c built with one fault macro; it scores the A2b positions, is
compared with llama.cpp R3, and goes through c50_a3_verdict.py exactly as the engine did.  The gate is
expected to FAIL every planted fault; a planted fault that PASSES marks a fault class the gate cannot
see.  Numbers only; the outcome is read off the verdict files.

Usage:
    python -I benchmarks/chatbot50/c50_a3_planted.py --model <q4_k_m.gguf> --work E:/.../a2b \
        --out benchmarks/chatbot50/results/c50_a3_planted.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKENIZER = "E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/sources/Ling-mini-2.0_a810f64/tokenizer.json"
IDS_SHA256 = "cf76819b8b5ee1636feea8badd7ae4b52d08b1bcc5dffcd87d8704722eeaa460"
FLOOR = HERE / "results" / "c50_a2b_floor.json"
# (name, binary, substring the CONFIG line must carry)
PLANTS = [("PLANT_WSCALE", "c50_engine_plant_wscale.exe", "weights_scale=2.25 "),
          ("PLANT_ROPE", "c50_engine_plant_rope.exe", "rope_base=500000 ")]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    work, model = Path(args.work), Path(args.model)
    ids = work / "a2b_ids.tsv"
    if sha256(ids) != IDS_SHA256:
        raise SystemExit("A2b position file differs from the registered SHA-256")
    report = {"cell": "C50-A3 planted control", "model_sha256": sha256(model),
              "source_sha256": sha256(HERE / "c50_engine.c"), "plants": {},
              "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for name, exe_name, expect in PLANTS:
        exe = HERE / "bin" / exe_name
        rec = {"exe": exe_name, "exe_sha256": sha256(exe), "expect_config": expect}
        report["plants"][name] = rec
        t0 = time.time()
        proc = subprocess.run([str(exe), "--model", str(model), "--score", str(ids), "--out", str(work / name),
                               "--dump-logprobs"], capture_output=True, text=True)
        lines = proc.stdout.splitlines()
        config = [ln for ln in lines if ln.startswith("CONFIG ")]
        rec.update({"returncode": proc.returncode, "wall_s": time.time() - t0, "config": config,
                    "stdout_tail": lines[-2:], "stderr_tail": proc.stderr[-600:]})
        if (proc.returncode != 0 or len(config) != 1 or expect not in config[0]
                or "positions=26363 " not in config[0] or "dump=1 " not in config[0]):
            rec["status"] = "VOID: run failed or CONFIG does not carry the planted value"
            Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
            continue
        cmp_path = work / f"compare_R3_{name}.json"
        subprocess.run([sys.executable, "-I", str(HERE / "c50_compare.py"), "--ref", str(work / "R3"), "--other",
                        str(work / name), "--ids", str(ids), "--tokenizer", TOKENIZER, "--json", str(cmp_path)],
                       check=True)
        run_path = work / f"run_{name}.json"
        run_path.write_text(json.dumps({"status": "COMPLETE", "pairs": {"R3-ENGINE": json.loads(
            cmp_path.read_text(encoding="utf-8"))}}), encoding="utf-8")
        verdict_path = work / f"verdict_{name}.json"
        subprocess.run([sys.executable, "-I", str(HERE / "c50_a3_verdict.py"), str(run_path), str(FLOOR),
                        "--json", str(verdict_path)], check=True)
        rec["compare"] = json.loads(cmp_path.read_text(encoding="utf-8"))
        rec["verdict"] = json.loads(verdict_path.read_text(encoding="utf-8"))
        rec["status"] = "COMPLETE"
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
