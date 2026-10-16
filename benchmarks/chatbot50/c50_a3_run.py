"""C50-A3 runner: engine self-test, engine scoring of the A2b positions, comparison with the
llama.cpp arms R3 (reference) and R0 (reported).  Numbers only; c50_a3_verdict.py applies
the gate with the A2b floor.

Protocol: docs/research/CHATBOT50_ROADMAP_20261016/C50_A3_ENGINE_PORT_PROTOCOL_20261016.md

Usage:
    python -I benchmarks/chatbot50/c50_a3_run.py --model <q4_k_m.gguf> --work E:/.../a2b \
        --out benchmarks/chatbot50/results/c50_a3_run.json
(--work is the A2b work directory holding a2b_ids.tsv, R3.* and R0.*)
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
    work, exe, model = Path(args.work), HERE / "bin" / "c50_engine.exe", Path(args.model)
    ids = work / "a2b_ids.tsv"
    if sha256(ids) != IDS_SHA256:
        raise SystemExit("A2b position file differs from the registered SHA-256")
    report = {"cell": "C50-A3", "exe_sha256": sha256(exe), "model": str(model), "model_sha256": sha256(model),
              "source_sha256": {p: sha256(HERE / p) for p in ("c50_engine.c", "c50_gguf.h", "c50_kquants_a3.h", "c50_kquants.h")},
              "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    st = subprocess.run([str(exe), "--selftest"], capture_output=True, text=True)
    report["selftest"] = st.stdout
    if st.returncode != 0 or "selftest PASS" not in st.stdout:
        report["status"] = "VOID: engine self-test failed"
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        raise SystemExit(report["status"])
    prefix = work / "ENGINE"
    t0 = time.time()
    proc = subprocess.run([str(exe), "--model", str(model), "--score", str(ids), "--out", str(prefix), "--dump-logprobs"],
                          capture_output=True, text=True)
    lines = proc.stdout.splitlines()
    config = [ln for ln in lines if ln.startswith("CONFIG ")]
    report["engine"] = {"returncode": proc.returncode, "wall_s": time.time() - t0, "config": config,
                        "stdout_tail": lines[-2:], "stderr_tail": proc.stderr[-600:]}
    if proc.returncode != 0 or len(config) != 1 or "positions=26363 " not in config[0] or "dump=1 " not in config[0]:
        report["status"] = "VOID: engine run failed or CONFIG does not confirm the registered run"
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        raise SystemExit(report["status"])
    report["pairs"] = {}
    for ref in ("R3", "R0"):
        out = work / f"compare_{ref}_ENGINE.json"
        subprocess.run([sys.executable, "-I", str(HERE / "c50_compare.py"), "--ref", str(work / ref), "--other",
                        str(prefix), "--ids", str(ids), "--tokenizer", TOKENIZER, "--json", str(out)], check=True)
        report["pairs"][f"{ref}-ENGINE"] = json.loads(out.read_text(encoding="utf-8"))
    report["status"] = "COMPLETE"
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
