"""C50-A6 addendum 1: copy a GGUF with every MoE router (ffn_gate_inp.weight) rewritten from F32 to BF16.

The pinned llama-quantize never quantizes ffn_gate_inp (src/llama-quant.cpp:306), so a --tensor-type override
for it is silently ignored.  This tool makes the P4 router step with the pinned gguf-py: metadata, tensor order
and alignment are copied; each router is rounded F32 -> BF16 (round to nearest even, gguf-py BF16, the same
rule as ggml_compute_fp32_to_bf16); every other tensor is copied byte for byte.  After writing, the output is
re-read and checked: same tensors in the same order, routers equal to the BF16 rounding of the source, all
other tensors byte-identical.  It prints numbers only.

Usage:
    python -I benchmarks/chatbot50/c50_a6_router_bf16.py --src in.gguf --dst out.gguf --out record.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

LLAMA_PIN = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/tools/llama.cpp-5b335f4")
ROUTER = "ffn_gate_inp.weight"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(LLAMA_PIN / "gguf-py"))
    import gguf
    from gguf.quants import quantize, dequantize

    BF16, F32 = gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F32
    src = gguf.GGUFReader(args.src)
    arch = src.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    w = gguf.GGUFWriter(args.dst, arch=arch, endianess=src.endianess)
    al = src.get_field(gguf.Keys.General.ALIGNMENT)
    if al is not None:
        w.data_alignment = al.contents()
    for f in src.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        t = f.types[0]
        w.add_key_value(f.name, f.contents(), t, sub_type=f.types[-1] if t == gguf.GGUFValueType.ARRAY else None)
    data = []
    for t in src.tensors:
        if t.name.endswith(ROUTER):
            if t.tensor_type != F32:
                raise SystemExit(f"{t.name}: expected F32, found {t.tensor_type.name}")
            q = quantize(np.asarray(t.data, dtype=np.float32), BF16)
            w.add_tensor_info(t.name, q.shape, q.dtype, q.nbytes, BF16)
            data.append(q)
        else:
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            data.append(t.data)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for d in data:
        w.write_tensor_data(d, tensor_endianess=src.endianess)
    w.close()

    dst = gguf.GGUFReader(args.dst)
    if [t.name for t in dst.tensors] != [t.name for t in src.tensors]:
        raise SystemExit("tensor list differs")
    n_router = n_same = 0
    rel = []
    for a, b in zip(src.tensors, dst.tensors):
        if a.name.endswith(ROUTER):
            x = np.asarray(a.data, dtype=np.float32)
            y = dequantize(np.asarray(b.data), BF16).reshape(x.shape)
            if b.tensor_type != BF16 or not np.array_equal(y, dequantize(quantize(x, BF16), BF16).reshape(x.shape)):
                raise SystemExit(f"{a.name}: router is not the BF16 rounding of the source")
            rel.append(float(np.max(np.abs(y - x)) / np.max(np.abs(x))))
            n_router += 1
        else:
            if b.tensor_type != a.tensor_type or not np.array_equal(np.asarray(a.data).view(np.uint8),
                                                                    np.asarray(b.data).view(np.uint8)):
                raise SystemExit(f"{a.name}: tensor changed")
            n_same += 1
    if n_router == 0:
        raise SystemExit("no router tensor found")
    rec = {"src": args.src, "dst": args.dst, "routers_rewritten": n_router, "tensors_byte_identical": n_same,
           "router_max_abs_err_over_max_abs": max(rel)}
    Path(args.out).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("routers", n_router, "identical", n_same, "max rel err", f"{max(rel):.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
