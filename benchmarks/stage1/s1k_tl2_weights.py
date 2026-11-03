"""Stage 1 (c): weights in bitnet.cpp's TL2 layout for s1k_bench (BITNET build with BITNET_X86_TL2), one random ternary
matrix per Ling-flash shape, packed by bitnet.cpp's own converter (utils/convert-hf-to-gguf-bitnet.py,
transform_to_tl2) with the kernel config that utils/codegen_tl2.py wrote for these shapes.

Per shape (out x in) it writes <out>_<in>.tl2 (the packed bytes), <out>_<in>.sign (int8 {-1, 0, +1}, row-major
[out, in], for the correctness check) and <out>_<in>.json (scale, byte count, sha256).  s1k_bench copies each shape's
bytes into every matrix of that shape: the values do not change the kernel's speed, and the copies still stream from RAM.

Usage: python -I benchmarks/stage1/s1k_tl2_weights.py --bitnet <BitNet clone> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

SHAPES = [(1024, 4096), (4096, 1024), (4096, 4096)]
SEED = 20261103


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitnet", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    root = a.bitnet.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    out = a.out.resolve()
    sys.path.insert(0, str(root / "utils"))
    sys.path.insert(0, str(root / "3rdparty" / "llama.cpp" / "gguf-py"))  # its gguf, not the installed one
    # the converter's vocabulary imports are not used for packing weights; a stub stands in for sentencepiece
    sys.modules.setdefault("sentencepiece", types.SimpleNamespace(SentencePieceProcessor=None))
    spec = importlib.util.spec_from_file_location("bitnet_convert", root / "utils" / "convert-hf-to-gguf-bitnet.py")
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    os.chdir(root)  # preprocess_weights_tl2 reads include/kernel_config.ini relative to the working directory
    rng = np.random.default_rng(SEED)
    for m, k in SHAPES:
        u = rng.random((m, k))
        sign = np.where(u < 0.4, 0, np.where(u < 0.7, 1, -1)).astype(np.int8)
        scale = 0.02
        packed, s = conv.transform_to_tl2(sign.astype(np.float32) * scale)
        b = np.ascontiguousarray(packed).tobytes()
        (out / f"{m}_{k}.tl2").write_bytes(b)
        (out / f"{m}_{k}.sign").write_bytes(sign.tobytes())
        rec = {"m": m, "k": k, "scale": float(s), "bytes": len(b), "bits_per_weight": 8 * len(b) / (m * k),
               "sha256": hashlib.sha256(b).hexdigest()}
        (out / f"{m}_{k}.json").write_text(json.dumps(rec), encoding="utf-8")
        print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
