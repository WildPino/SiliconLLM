"""S2N's parity, llama.cpp's side at k = 4 mass 16: S2M's documents panel on ks_stk4 with S2L's score tool and arguments, routing top 4 with mass 16 on every
position (a scored position is a prompt position).  Its docs.pos.bin is what s2n_train.py --parity --form k4m16 (and the planted k4m0) is read against;
k = 8's is S2M's own (E:/SiliconLLM_Artifacts/stage1/s1a/ks_stk4/docs.pos.bin).  CPU.

Usage: python -I benchmarks/stage2/s2n_llama_docs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from s2x_run import INPUTS, MAX_PROJECTED_S, PANELS, WORK, check_inputs, launch, sha256  # noqa: E402
from s2y_run import UA  # noqa: E402

KS = Path("E:/SiliconLLM_Artifacts/stage2/s2y/q4/ks_stk4.gguf")
OUT = Path("E:/SiliconLLM_Artifacts/stage2/s2n/llama/ks_stk4_k4m16")
REPORT = HERE / "results" / "s2n_llama_docs.json"


def main() -> int:
    want = json.loads((HERE / "results" / "s2z_export_ks_stk4.json").read_text(encoding="utf-8"))["q4_sha256"]
    got = sha256(KS)
    if got != want:
        raise SystemExit(f"ks_stk4 differs from S2M's export record ({got[:8]})")
    panel = next(p for p in PANELS if p[0] == "docs")
    _, src, tool, extra = panel
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = [str(UA[tool]), "--model", str(KS), "--ids", str(INPUTS[src] / "docs.tsv"), "--out", str(OUT / "docs"), "--threads", "6", "--ubatch", "512",
           "--max-projected-s", str(3 * MAX_PROJECTED_S)] + extra + ["--moe-mode", "top", "--moe-k", "4", "--moe-mass", "16",
                                                                      "--ref-top", str(WORK / "ref" / "docs.top.bin")]
    r = launch(cmd, [f"model={KS} ", "flash_attn=disabled", "moe_mode=top ", "moe_k=4 ", "moe_mass=16 "])
    r.update(model_sha256=got, inputs_manifest_sha256=check_inputs(INPUTS[src]))
    REPORT.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    print("S2N LLAMA DOCS", r["status"], r["wall_s"], flush=True)
    return 0 if r["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
