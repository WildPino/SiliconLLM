"""Stage 1 (a) part 2: put trained routers and routed experts into Ling-mini's GGUF and quantize it as P0 was.

P0 (`ling-mini-2.0-a810f64-q4_k_m.gguf`, sha256 44de7315...) was made from the bf16 GGUF by llama-quantize (commit
5b335f4) as Q4_K_M with the imatrix `ling-mini-2.0-a810f64-imatrix-fit24.gguf`.  The export copies the bf16 GGUF,
overwrites in place, at their offsets, the tensors that training changed (blk.L.ffn_{gate,up,down}_exps.weight in bf16,
blk.L.ffn_gate_inp.weight in f32), and runs the same quantize command.

    check     the bf16 GGUF's tensors equal the HF checkpoint's, byte for byte (layout of the stacks), for every layer
    patch     --trained DIR (layer_L.pt files from s1t_train.py; none = the untrained control) -> --out-bf16, --out-q4
Control registered with the protocol: `patch` without --trained must give P0's sha256.
Stage 1 (b), --ternary: the trained layers' routed experts are quantized to TQ2_0 instead (s * q in bf16, one scale per
256-weight block, so TQ2_0's block maximum is s and the conversion is exact); the output is read back and every such
tensor must dequantize to the trained values bit for bit (ternary_exact).
Usage: python -I s1t_export.py check | patch [--trained DIR] [--ternary] --out-bf16 F --out-q4 F --json F
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


def ternary_exact(out_q4: Path, trained: str, layers: list[int]) -> dict:
    """Stage 1 (b) control: every TQ2_0 expert tensor of the output dequantizes to the trained s * q, bit for bit."""
    r = gguf.GGUFReader(str(out_q4))
    by = {t.name: t for t in r.tensors}
    out = {}
    for i in layers:
        st = torch.load(Path(trained) / f"layer_{i}.pt", map_location="cpu", weights_only=True)
        for g in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            t = by[f"blk.{i}.{g}.weight"]
            ok = t.tensor_type.name == "TQ2_0"
            if ok:
                got = gguf.quants.dequantize(t.data, t.tensor_type).reshape(st[HF[g]].shape)
                ok = bool(np.array_equal(got, st[HF[g]].float().numpy()))
            out[f"{i}.{g}"] = ok
    return {"all_exact": all(out.values()), "n": len(out), "not_exact": [k for k, v in out.items() if not v]}


def bal_levels(st: dict, h: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The trained stack back as Q4_0 fields, group 256 in blocks of 32: (d per block fp32, q in -3..3 per weight).  Four-level groups: d = s / 2, q = 2 * level;
    outlier groups: d = s, q = level.  Q4_0 holds both without error, unless d leaves fp16's normal range."""
    w = st[h].float()
    key = h.split(".")[-1]
    s = st[f"ternary_scale.{key}"].float()
    big = st[f"bal_big.{key}"]
    d = torch.where(big, s, s / 2).unsqueeze(-1)
    x = w.view(*s.shape, -1) / torch.where(d > 0, d, 1.0)
    return d.expand(*s.shape, 8), x.round()


BF16_HALF_ULP = 2.0 ** -8


def bal_blocks(d: torch.Tensor, q: torch.Tensor) -> tuple[np.ndarray, int]:
    """Q4_0 blocks (fp16 d, then x[j] | x[j + 16] << 4) and how many d fall under fp16's normal range."""
    dd = d.reshape(-1).numpy().astype(np.float16)
    n = (q.reshape(-1, 32).numpy() + 8).astype(np.uint8)
    b = np.empty((n.shape[0], 18), dtype=np.uint8)
    b[:, :2] = dd.view(np.uint8).reshape(-1, 2)
    b[:, 2:] = n[:, :16] | (n[:, 16:] << 4)
    return b, int(((dd != 0) & (np.abs(dd) < 2.0 ** -14)).sum())


def bal_error(got: np.ndarray, want: np.ndarray) -> float:
    """Worst relative error per weight; a zero must stay a zero."""
    nz = want != 0
    return float(max(np.abs(got[nz] / want[nz] - 1).max() if nz.any() else 0.0, np.inf if np.any(got[~nz] != 0) else 0.0))


def write_bal(out_q4: Path, trained: str, layers: list[int]) -> dict:
    """The quantizer rounds Q4_0 by the block's maximum and would miss this grid, so the trained tensors' bytes are written over its output, and read back."""
    r = gguf.GGUFReader(str(out_q4))
    by = {t.name: (t.tensor_type.name, int(t.data_offset), int(t.n_bytes)) for t in r.tensors}
    del r                       # its map must be closed before the file is written
    out, under = {}, 0
    with out_q4.open("r+b") as f:
        for i in layers:
            st = torch.load(Path(trained) / f"layer_{i}.pt", map_location="cpu", weights_only=True)
            for g in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
                name = f"blk.{i}.{g}.weight"
                kind, offset, size = by[name]
                if kind != "Q4_0":
                    raise SystemExit(f"{name} is {kind}")
                blocks, low = bal_blocks(*bal_levels(st, HF[g]))
                under += low
                if blocks.nbytes != size:
                    raise SystemExit(f"{name}: {blocks.nbytes} bytes against {size}")
                f.seek(offset)
                f.write(blocks.tobytes())
                out[f"{i}.{g}"] = name
    r2 = gguf.GGUFReader(str(out_q4))
    by2 = {t.name: t for t in r2.tensors}
    worst = 0.0
    for i in layers:                # per weight: the file against the stored bf16 s * level, which rounds 1.5 s
        st = torch.load(Path(trained) / f"layer_{i}.pt", map_location="cpu", weights_only=True)
        for g in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            got = gguf.quants.dequantize(by2[f"blk.{i}.{g}.weight"].data, by2[f"blk.{i}.{g}.weight"].tensor_type).reshape(st[HF[g]].shape)
            want = st[HF[g]].float().numpy()
            worst = max(worst, bal_error(got, want))
    return {"tensors": len(out), "blocks_under_fp16_normal": under, "worst_rel_per_weight": worst, "exact_to_bf16": worst <= BF16_HALF_ULP}


def patch(trained: str, out_bf16: Path, out_q4: Path, ternary: bool = False, bal: bool = False) -> dict:
    t0 = time.time()
    shutil.copyfile(BF16, out_bf16)
    changed, layers = [], []
    if trained:
        tt = tensors()
        with out_bf16.open("r+b") as f:
            for p in sorted(Path(trained).glob("layer_*.pt")):
                i = int(p.stem.split("_")[1])
                layers.append(i)
                st = torch.load(p, map_location="cpu", weights_only=True)
                for g, h in HF.items():
                    off, n, kind = tt[f"blk.{i}.{g}.weight"]
                    b = as_bytes(st[h], kind)
                    assert len(b) == n, (i, g)
                    f.seek(off)
                    f.write(b)
                    changed.append(f"blk.{i}.{g}.weight")
    # forward slashes, as P0 was made: the imatrix path is written into the output's metadata (quantize.imatrix.file)
    cmd = [str(QUANTIZE), "--imatrix", IMATRIX.as_posix()]
    if ternary:                 # stage 1 (b): the trained layers' routed experts go to TQ2_0 (block 256 = the groups)
        for i in sorted(layers):
            cmd += ["--tensor-type", rf"blk\.{i}\.ffn_(gate|up|down)_exps=tq2_0"]
    if bal:                     # the fourth code, quality only: Q4_0 holds the grid exactly; write_bal puts the bytes
        for i in sorted(layers):
            cmd += ["--tensor-type", rf"blk\.{i}\.ffn_(gate|up|down)_exps=q4_0"]
    cmd += [out_bf16.as_posix(), out_q4.as_posix(), "Q4_K_M", "6"]
    env = {**os.environ, "PATH": os.pathsep.join([str(QUANTIZE.parent), str(RUNTIME), os.environ.get("PATH", "")])}
    q = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    (out_q4.parent / (out_q4.name + ".quantize.log")).write_text(q.stdout + q.stderr, encoding="utf-8")
    if q.returncode:
        raise SystemExit(f"quantize failed ({q.returncode})")
    bal_rec = write_bal(out_q4, trained, sorted(layers)) if bal else None
    s = sha256(out_q4)
    rec = {"bal_exact": bal_rec, "trained": trained or None, "changed_tensors": len(changed), "bf16": str(out_bf16), "q4": str(out_q4),
           "q4_sha256": s, "equals_p0": s == P0_SHA, "seconds": round(time.time() - t0, 1), "cmd": cmd}
    if ternary:
        rec["ternary_exact"] = ternary_exact(out_q4, trained, layers)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["check", "patch"])
    ap.add_argument("--trained", default="")
    ap.add_argument("--out-bf16", type=Path)
    ap.add_argument("--out-q4", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--ternary", action="store_true", help="stage 1 (b): the trained layers' experts as TQ2_0")
    ap.add_argument("--bal", action="store_true", help="the fourth code: the trained layers' experts as exact Q4_0")
    a = ap.parse_args()
    r = check() if a.what == "check" else patch(a.trained, a.out_bf16, a.out_q4, a.ternary, a.bal)
    print(json.dumps(r))
    if a.json:
        a.json.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
