"""Stage 1 (a) part 2: put trained routers and routed experts into Ling-mini's GGUF and quantize it as P0 was.

P0 (`ling-mini-2.0-a810f64-q4_k_m.gguf`, sha256 44de7315...) was made from the bf16 GGUF by llama-quantize (commit
5b335f4) as Q4_K_M with the imatrix `ling-mini-2.0-a810f64-imatrix-fit24.gguf`.  The export copies the bf16 GGUF,
overwrites in place, at their offsets, the tensors that training changed (blk.L.ffn_{gate,up,down}_exps.weight in bf16,
blk.L.ffn_gate_inp.weight in f32), and runs the same quantize command.

    check     the bf16 GGUF's tensors equal the HF checkpoint's, byte for byte (layout of the stacks), for every layer
    patch     --trained DIR (layer_L.pt files from s1t_train.py; none = the untrained control) -> --out-bf16, --out-q4
Control registered with the protocol: `patch` without --trained must give P0's sha256.
Usage: python -I s1t_export.py check | patch [--trained DIR] --out-bf16 F --out-q4 F --json F
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1t_model as M  # noqa: E402

CAMPAIGN = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
LLAMA = CAMPAIGN / "tools" / "llama.cpp-5b335f4"
sys.path.insert(0, str(LLAMA / "gguf-py"))
import gguf  # noqa: E402

BF16 = CAMPAIGN / "gguf" / "ling-mini-2.0-a810f64-bf16.gguf"
IMATRIX = CAMPAIGN / "gguf" / "ling-mini-2.0-a810f64-imatrix-fit24.gguf"
QUANTIZE = LLAMA / "build-avx2" / "bin" / "llama-quantize.exe"
# The binary needs the MinGW runtime (WinLibs, installed by winget): a shell whose PATH finds an older libstdc++ first
# (Git Bash) makes it fail with 0xC0000139, so the runtime directory goes first on its PATH.
RUNTIME = (Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
           / "BrechtSanders.WinLibs.POSIX.MSVCRT_Microsoft.Winget.Source_8wekyb3d8bbwe" / "mingw64" / "bin")
P0_SHA = "44de73157f11165c0551fd06714f4f043913292a79b07683e6a34b11d3b0d2f4"
HF = {"ffn_gate_exps": "mlp.experts.gate_proj", "ffn_up_exps": "mlp.experts.up_proj",
      "ffn_down_exps": "mlp.experts.down_proj", "ffn_gate_inp": "mlp.gate.weight"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def tensors() -> dict[str, tuple[int, int, str]]:
    r = gguf.GGUFReader(str(BF16))
    return {t.name: (int(t.data_offset), int(t.n_bytes), t.tensor_type.name) for t in r.tensors}


def as_bytes(t: torch.Tensor, kind: str) -> bytes:
    if kind == "BF16":
        return t.contiguous().to(torch.bfloat16).view(torch.int16).numpy().tobytes()
    assert kind == "F32"
    return t.contiguous().float().numpy().astype("<f4").tobytes()


def check() -> dict:
    tt, W, out = tensors(), M.Weights(), {}
    with BF16.open("rb") as f:
        for i in range(1, M.N_LAYER):
            w = W.layer(i)
            for g, h in HF.items():
                off, n, kind = tt[f"blk.{i}.{g}.weight"]
                f.seek(off)
                out[f"{i}.{g}"] = f.read(n) == as_bytes(w[h], kind)
            off, n, _ = tt[f"blk.{i}.exp_probs_b.bias"]
            f.seek(off)
            out[f"{i}.exp_probs_b"] = np.frombuffer(f.read(n), "<f4").tobytes() == as_bytes(w["mlp.gate.expert_bias"], "F32")
    return {"all_equal": all(out.values()), "n": len(out), "unequal": [k for k, v in out.items() if not v]}


def patch(trained: str, out_bf16: Path, out_q4: Path) -> dict:
    t0 = time.time()
    shutil.copyfile(BF16, out_bf16)
    changed = []
    if trained:
        tt = tensors()
        with out_bf16.open("r+b") as f:
            for p in sorted(Path(trained).glob("layer_*.pt")):
                i = int(p.stem.split("_")[1])
                st = torch.load(p, map_location="cpu", weights_only=True)
                for g, h in HF.items():
                    off, n, kind = tt[f"blk.{i}.{g}.weight"]
                    b = as_bytes(st[h], kind)
                    assert len(b) == n, (i, g)
                    f.seek(off)
                    f.write(b)
                    changed.append(f"blk.{i}.{g}.weight")
    # forward slashes, as P0 was made: the imatrix path is written into the output's metadata (quantize.imatrix.file)
    cmd = [str(QUANTIZE), "--imatrix", IMATRIX.as_posix(), out_bf16.as_posix(), out_q4.as_posix(), "Q4_K_M", "6"]
    env = {**os.environ, "PATH": os.pathsep.join([str(QUANTIZE.parent), str(RUNTIME), os.environ.get("PATH", "")])}
    q = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    (out_q4.parent / (out_q4.name + ".quantize.log")).write_text(q.stdout + q.stderr, encoding="utf-8")
    if q.returncode:
        raise SystemExit(f"quantize failed ({q.returncode})")
    s = sha256(out_q4)
    return {"trained": trained or None, "changed_tensors": len(changed), "bf16": str(out_bf16), "q4": str(out_q4),
            "q4_sha256": s, "equals_p0": s == P0_SHA, "seconds": round(time.time() - t0, 1), "cmd": cmd}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["check", "patch"])
    ap.add_argument("--trained", default="")
    ap.add_argument("--out-bf16", type=Path)
    ap.add_argument("--out-q4", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    r = check() if a.what == "check" else patch(a.trained, a.out_bf16, a.out_q4)
    print(json.dumps(r))
    if a.json:
        a.json.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
