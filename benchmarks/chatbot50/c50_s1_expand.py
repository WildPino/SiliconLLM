"""C50-S1 expander: copy T0's GGUF with every MoE layer's expert pool grown from 256 to E experts by byte copies.

Layout (the engine's, c50_engine.c): 8 groups of G = E/8 slots; slot g*G + 32k + j holds copy k of T0 expert 32g + j
(G >= 32, so copy 0 of every expert exists and sits at g*G + j).  Per MoE layer:
    ffn_gate_exps / ffn_up_exps / ffn_down_exps   slot s = the source expert's bytes, unchanged
    exp_probs_b.bias                              slot s = the source expert's bias
    ffn_gate_inp.weight (F32)                     copy 0 = T0's row; copy k >= 1 = the row times (1 + sigma * z),
                                                  z ~ N(0, 1) from a fixed seed per (layer, slot), rounded to BF16
                                                  (round to nearest even) and stored as F32, so that every router
                                                  value stays exactly representable in BF16
bailingmoe2.expert_count = E.  Every other key and tensor is copied unchanged.  After writing, the output is re-read
and checked: the tensor list, every expert slot against its source, copy 0 of the router and the bias against T0,
every router value exactly BF16, every other tensor byte-identical.  It prints numbers only.

Usage:
    python -I benchmarks/chatbot50/c50_s1_expand.py --src T0.gguf --experts 536 --sigma S --seed 0 \
        --dst out.gguf --out record.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

LLAMA_PIN = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/tools/llama.cpp-5b335f4")
NE0, NG, GE0 = 256, 8, 32
EXPERT_TENSORS = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
ROUTER, BIAS, KEY = "ffn_gate_inp.weight", "exp_probs_b.bias", "bailingmoe2.expert_count"


def slot_sources(e: int) -> np.ndarray:
    """source T0 expert of every slot"""
    g_size = e // NG
    s = np.arange(e)
    g, r = s // g_size, s % g_size
    return (g * GE0 + r % GE0).astype(np.int64)


def slot_copy(e: int) -> np.ndarray:
    return (np.arange(e) % (e // NG)) // GE0


def bf16_round(x: np.ndarray) -> np.ndarray:
    """F32 -> nearest-even BF16, returned as F32 with the low 16 bits zero"""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def router_rows(src: np.ndarray, e: int, layer: int, sigma: float, seed: int) -> np.ndarray:
    srcs, copy = slot_sources(e), slot_copy(e)
    out = src[srcs].astype(np.float32, copy=True)
    for s in np.nonzero(copy > 0)[0]:
        z = np.random.default_rng([seed, layer, int(s)]).standard_normal(src.shape[1]).astype(np.float32)
        out[s] = bf16_round(out[s] * (np.float32(1) + np.float32(sigma) * z))
    return out


def layer_of(name: str) -> int | None:
    return int(name.split(".")[1]) if name.startswith("blk.") else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--experts", type=int, required=True)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hash", action="store_true", help="also record the sha256 of the output file")
    args = ap.parse_args()
    e = args.experts
    if e % NG or e // NG < GE0:
        raise SystemExit(f"--experts {e}: must be a multiple of {NG} with at least {GE0} per group")
    sys.path.insert(0, str(LLAMA_PIN / "gguf-py"))
    import gguf

    F32 = gguf.GGMLQuantizationType.F32
    t0 = time.time()
    src = gguf.GGUFReader(args.src)
    if int(src.get_field(KEY).contents()) != NE0:
        raise SystemExit("the source is not a 256-expert T0")
    srcs = slot_sources(e)
    arch = src.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    w = gguf.GGUFWriter(args.dst, arch=arch, endianess=src.endianess)
    al = src.get_field(gguf.Keys.General.ALIGNMENT)
    if al is not None:
        w.data_alignment = al.contents()
    for f in src.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        t = f.types[0]
        val = e if f.name == KEY else f.contents()
        w.add_key_value(f.name, val, t, sub_type=f.types[-1] if t == gguf.GGUFValueType.ARRAY else None)

    def grown(t):
        """(shape, dtype, nbytes) of the output tensor and a function that materializes its data"""
        if layer_of(t.name) is not None and t.name.endswith(EXPERT_TENSORS):
            per = t.data.nbytes // t.data.shape[0]
            return (e,) + tuple(t.data.shape[1:]), t.data.dtype, per * e, lambda: np.ascontiguousarray(t.data[srcs])
        if t.name.endswith(ROUTER):
            if t.tensor_type != F32:
                raise SystemExit(f"{t.name}: expected F32")
            l = layer_of(t.name)
            return (e, t.data.shape[1]), np.float32, 4 * e * t.data.shape[1], \
                lambda: router_rows(np.asarray(t.data, dtype=np.float32), e, l, args.sigma, args.seed)
        if t.name.endswith(BIAS):
            return (e,), np.float32, 4 * e, lambda: np.ascontiguousarray(np.asarray(t.data, dtype=np.float32)[srcs])
        return tuple(t.data.shape), t.data.dtype, t.data.nbytes, lambda: t.data

    plan = []
    for t in src.tensors:
        shape, dtype, nbytes, make = grown(t)
        w.add_tensor_info(t.name, shape, dtype, nbytes, t.tensor_type)
        plan.append(make)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for make in plan:
        w.write_tensor_data(make(), tensor_endianess=src.endianess)
    w.close()
    t_write = time.time() - t0

    dst = gguf.GGUFReader(args.dst)
    if [t.name for t in dst.tensors] != [t.name for t in src.tensors] or int(dst.get_field(KEY).contents()) != e:
        raise SystemExit("tensor list or expert count differs")
    copy = slot_copy(e)
    checked = {"expert_tensors": 0, "routers": 0, "biases": 0, "other_identical": 0}
    max_rel = 0.0
    for a, b in zip(src.tensors, dst.tensors):
        if b.tensor_type != a.tensor_type:
            raise SystemExit(f"{a.name}: type changed")
        if layer_of(a.name) is not None and a.name.endswith(EXPERT_TENSORS):
            if not np.array_equal(np.asarray(b.data).view(np.uint8), np.asarray(a.data)[srcs].view(np.uint8)):
                raise SystemExit(f"{a.name}: an expert slot differs from its source")
            checked["expert_tensors"] += 1
        elif a.name.endswith(ROUTER):
            x, y = np.asarray(a.data, dtype=np.float32), np.asarray(b.data, dtype=np.float32)
            if not np.array_equal(y[copy == 0].view(np.uint32), x[srcs[copy == 0]].view(np.uint32)):
                raise SystemExit(f"{a.name}: copy 0 differs from T0")
            if np.any(y.view(np.uint32) & 0xFFFF):
                raise SystemExit(f"{a.name}: a router value is not exactly BF16")
            if np.any(copy > 0):
                d = np.abs(y[copy > 0] - x[srcs[copy > 0]]).max() / np.abs(x).max()
                max_rel = max(max_rel, float(d))
            checked["routers"] += 1
        elif a.name.endswith(BIAS):
            if not np.array_equal(np.asarray(b.data, dtype=np.float32), np.asarray(a.data, dtype=np.float32)[srcs]):
                raise SystemExit(f"{a.name}: bias differs")
            checked["biases"] += 1
        else:
            if not np.array_equal(np.asarray(a.data).view(np.uint8), np.asarray(b.data).view(np.uint8)):
                raise SystemExit(f"{a.name}: tensor changed")
            checked["other_identical"] += 1
    copies = np.bincount(np.bincount(srcs, minlength=NE0))
    rec = {"src": args.src, "dst": args.dst, "experts": e, "group_size": e // NG, "sigma": args.sigma, "seed": args.seed,
           "copies_per_expert_histogram": {str(k): int(v) for k, v in enumerate(copies) if v},
           "checked": checked, "router_copy_max_abs_change_over_max_abs": max_rel,
           "dst_bytes": Path(args.dst).stat().st_size, "write_s": round(t_write, 1), "verify_s": round(time.time() - t0 - t_write, 1)}
    if args.hash:
        h = hashlib.sha256()
        with open(args.dst, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        rec["dst_sha256"] = h.hexdigest()
    Path(args.out).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
