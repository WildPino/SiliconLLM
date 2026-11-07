"""S2L, IQ2_KS on the organs, no training.  ik_llama.cpp's quantizer writes attention, shared expert and dense FFN in IQ2_KS with the fit24 imatrix (converted to its legacy format), then
requantizes the file to BF16, which is how its values reach a file llama.cpp reads.  It runs twice, with and without the imatrix.  Control: over the organs, the imatrix-weighted weight
error must be lower with the imatrix than without it, or the imatrix did not arrive.

Usage: python -I benchmarks/stage2/s2y_ks.py --json benchmarks/stage2/results/s2y_ks.json
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
from s1t_export import BF16, IMATRIX, RUNTIME, gguf  # noqa: E402

IK = Path("E:/SiliconLLM_Artifacts/wallmap/tools/ik_llama.cpp/build-fa/bin/llama-quantize.exe")
Y = Path("E:/SiliconLLM_Artifacts/stage2/s2y")
CUSTOM = r"attn_qkv=iq2_ks,attn_output=iq2_ks,ffn_gate_shexp=iq2_ks,ffn_up_shexp=iq2_ks,ffn_down_shexp=iq2_ks,blk\.0\.ffn_gate\.weight=iq2_ks,blk\.0\.ffn_up\.weight=iq2_ks,blk\.0\.ffn_down\.weight=iq2_ks"


def imatrix() -> dict[str, np.ndarray]:
    r = {t.name: np.asarray(t.data, dtype=np.float32) for t in gguf.GGUFReader(str(IMATRIX)).tensors}
    return {k.removesuffix(".in_sum2"): v.reshape(-1) / float(r[k.removesuffix(".in_sum2") + ".counts"].reshape(-1)[0])
            for k, v in r.items() if k.endswith(".in_sum2") and "_exps" not in k}


def legacy(data: dict[str, np.ndarray], path: Path) -> None:
    with path.open("wb") as f:
        f.write(struct.pack("<i", len(data)))
        for n, v in data.items():
            b = n.encode()
            f.write(struct.pack("<i", len(b)) + b + struct.pack("<ii", 1, v.size) + v.astype("<f4").tobytes())
        name = b"fit24"
        f.write(struct.pack("<ii", 1, len(name)) + name)    # last call, then the dataset's name, which the loader reads unchecked


def run(cmd: list[str], log: Path) -> None:
    env = {**os.environ, "PATH": os.pathsep.join([str(RUNTIME), os.environ.get("PATH", "")])}
    q = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    log.write_text(q.stdout + q.stderr, encoding="utf-8")
    if q.returncode:
        raise SystemExit(f"failed ({q.returncode}): {' '.join(cmd[:4])}; see {log}")


def make(tag: str, dat: Path | None) -> Path:
    ks, bf = Y / f"ks-{tag}.gguf", Y / f"ks-{tag}-bf16.gguf"
    if not bf.exists():
        run([str(IK), *(["--imatrix", str(dat)] if dat else []), "--ignore-imatrix-rules", "--custom-q", CUSTOM, str(BF16), str(ks), "Q8_0", "6"], Y / f"ks-{tag}.log")
        run([str(IK), "--allow-requantize", "--pure", str(ks), str(bf), "BF16", "6"], Y / f"ks-{tag}-bf16.log")
        ks.unlink()
    return bf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    Y.mkdir(parents=True, exist_ok=True)
    data = imatrix()
    dat = Y / "imatrix-organs.dat"
    legacy(data, dat)
    files = {"im": make("im", dat), "noim": make("noim", None)}
    src = {t.name: t for t in gguf.GGUFReader(str(BF16)).tensors}
    got = {k: {t.name: t for t in gguf.GGUFReader(str(p)).tensors} for k, p in files.items()}
    organs = [n for n in src if any(x in n for x in ("attn_qkv", "attn_output", "_shexp")) or n.startswith("blk.0.ffn_")]

    def f32(t) -> np.ndarray:
        return (np.asarray(t.data).reshape(-1).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    out = {}
    for n in organs:
        o = f32(src[n])
        cols = data[n.removesuffix(".weight") + ".weight"].size
        wgt = np.tile(data[n], o.size // cols)
        rec = {}
        for k in files:
            d = f32(got[k][n])
            e = (d - o) ** 2
            rec[k] = {"rel": float(e.sum() / (o * o).sum()), "weighted": float((wgt * e).sum() / (wgt * o * o).sum())}
        out[n] = rec
    win = sum(r["im"]["weighted"] < r["noim"]["weighted"] for r in out.values())
    rec = {"organ_tensors": len(out), "imatrix_lower_on": win, "control": win == len(out), "mean_rel_im": float(np.mean([r["im"]["rel"] for r in out.values()])),
           "mean_rel_noim": float(np.mean([r["noim"]["rel"] for r in out.values()])), "mean_weighted_im": float(np.mean([r["im"]["weighted"] for r in out.values()])),
           "mean_weighted_noim": float(np.mean([r["noim"]["weighted"] for r in out.values()])), "tensors": out, "seconds": round(time.time() - t0, 1)}
    a.json.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k != "tensors"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
