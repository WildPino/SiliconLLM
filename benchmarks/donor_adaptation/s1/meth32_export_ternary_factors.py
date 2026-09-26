#!/usr/bin/env python3
"""METH-32: weight-only optimal-row-ternary A/B factors in LUT code order."""
import argparse
import json
from pathlib import Path
import time

import psutil
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

import meth15_zero_residual_expert_smoke as M15
import meth20_half_adapter_generation as M20
import meth25_fresh_r8_artifact_audit as M25


ROOT = Path(__file__).resolve().parents[3]
MAX_SECONDS = 15 * 60
MAX_RSS_BYTES = 20 * (1 << 30)


def check_budget(start):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    if elapsed > MAX_SECONDS:
        raise TimeoutError(f"METH-32 export wall stop: {elapsed:.1f}s")
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"METH-32 export RSS stop: {rss}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss}


def quantize_rows(weight):
    assert weight.dtype == torch.float32 and weight.ndim == 3
    experts, outputs, inputs = weight.shape
    assert inputs % 2 == 0
    rows = weight.reshape(-1, inputs)
    values, order = torch.sort(rows.abs(), dim=1, descending=True, stable=True)
    sums = values.cumsum(dim=1)
    k = torch.arange(1, inputs + 1, dtype=torch.float32)
    gains = sums.square() / k
    best = gains.argmax(dim=1)
    scales = sums.gather(1, best[:, None]).squeeze(1) / (best + 1)
    chosen = torch.arange(inputs)[None, :] <= best[:, None]
    sorted_codes = rows.gather(1, order).sign().to(torch.int8) * chosen.to(torch.int8)
    codes = torch.zeros_like(sorted_codes).scatter_(1, order, sorted_codes)
    reconstructed = codes.float() * scales[:, None]
    error = (rows - reconstructed).square().sum().item()
    norm = rows.square().sum().item()
    assert bool((codes >= -1).all()) and bool((codes <= 1).all())
    code = codes.reshape(experts, outputs, inputs).contiguous()
    scale = scales.reshape(experts, outputs).contiguous()
    return code, scale, {"weight_elements": rows.numel(),
                         "nonzero_codes": int((codes != 0).sum()),
                         "squared_error": error, "weight_squared_norm": norm,
                         "relative_squared_error": error / norm if norm else 0.0}


def pack_codes(q, pad_outputs):
    assert q.dtype == torch.int8 and q.ndim == 3
    experts, outputs, inputs = q.shape
    assert inputs % 2 == 0 and pad_outputs >= outputs and pad_outputs % 32 == 0
    pairs = q.reshape(experts, outputs, inputs // 2, 2).to(torch.int16)
    codes = (pairs[..., 0] + 1) * 3 + pairs[..., 1] + 1
    packed = torch.full((experts, inputs // 2, pad_outputs), 4, dtype=torch.uint8)
    packed[:, :, :outputs] = codes.permute(0, 2, 1).to(torch.uint8)
    assert bool((packed <= 8).all())
    return packed.contiguous()


def unpack_codes(packed, outputs):
    assert packed.dtype == torch.uint8 and packed.ndim == 3
    experts, tiles, pad_outputs = packed.shape
    assert outputs <= pad_outputs and bool((packed <= 8).all())
    pairs = packed[:, :, :outputs].permute(0, 2, 1).to(torch.int16)
    out = torch.stack((pairs // 3 - 1, pairs % 3 - 1), dim=-1)
    return out.reshape(experts, outputs, tiles * 2).to(torch.int8).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    start = time.monotonic()
    torch.set_num_threads(6)
    assert M15.M13.sha256(M25.CORE) == M25.CORE_SHA
    assert M15.M13.sha256(M20.ADAPTER) == M20.ADAPTER_SHA
    with safe_open(str(M20.ADAPTER), framework="pt", device="cpu") as archive:
        original_meta = archive.metadata()
        assert original_meta["model_sha256"] == M15.M13.MODEL_SHA
        assert original_meta["output_factor"] == "0.5"
    source = load_file(str(M20.ADAPTER), device="cpu")
    assert len(source) == 3 * M15.M13.L
    tensors = {}
    totals = {organ: {"weight_elements": 0, "nonzero_codes": 0,
                      "squared_error": 0.0, "weight_squared_norm": 0.0}
              for organ in ("a", "b")}
    for li in range(M15.M13.L):
        for organ, shape, pad in (("a", (M15.E, M15.R, M15.M13.D), 32),
                                  ("b", (M15.E, M15.M13.D, M15.R), M15.M13.D)):
            key = f"layers.{li}.{organ}"
            weight = source[key]
            assert weight.dtype == torch.float32 and tuple(weight.shape) == shape
            q, scale, stats = quantize_rows(weight)
            packed = pack_codes(q, pad)
            assert torch.equal(unpack_codes(packed, shape[1]), q)
            tensors[key + ".code"] = packed
            tensors[key + ".scale"] = scale
            for field in totals[organ]:
                totals[organ][field] += stats[field]
        router = source[f"layers.{li}.router"]
        assert router.dtype == torch.float32
        assert tuple(router.shape) == (M15.E, M15.M13.D)
        tensors[f"layers.{li}.router"] = router.contiguous()
        check_budget(start)
        if (li + 1) % 6 == 0:
            print(f"quantized {li+1}/{M15.M13.L} layers", flush=True)
    for organ in totals:
        row = totals[organ]
        row["relative_squared_error"] = row["squared_error"] / row["weight_squared_norm"]
        row["nonzero_fraction"] = row["nonzero_codes"] / row["weight_elements"]
    assert len(tensors) == M15.M13.L * 5
    assert all(t.is_contiguous() for t in tensors.values())
    metadata = {"format": "QWEN25_R8_E128_TERNARY_LUT_V1",
                "core_sha256": M25.CORE_SHA,
                "source_adapter_sha256": M20.ADAPTER_SHA,
                "source_model_sha256": M15.M13.MODEL_SHA,
                "output_factor": "0.5",
                "quantizer": "optimal_row_sse_ternary_tie_smallest_k",
                "code_mapping": "(q0+1)*3+(q1+1)",
                "code_layout": "expert,tile,output_padded"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out), metadata=metadata)
    with safe_open(str(args.out), framework="pt", device="cpu") as archive:
        assert archive.metadata() == metadata
        assert len(archive.keys()) == len(tensors)
        for key, original in tensors.items():
            assert torch.equal(archive.get_tensor(key), original), key
    code_bytes = sum(t.numel() * t.element_size() for k, t in tensors.items()
                     if k.endswith(".code"))
    scale_bytes = sum(t.numel() * t.element_size() for k, t in tensors.items()
                      if k.endswith(".scale"))
    router_bytes = sum(t.numel() * t.element_size() for k, t in tensors.items()
                       if k.endswith(".router"))
    runtime = check_budget(start)
    report = {"experiment": "METH-32-export", "format": metadata["format"],
              "core_sha256": M25.CORE_SHA,
              "source_adapter_sha256": M20.ADAPTER_SHA,
              "packed_adapter_sha256": M15.M13.sha256(args.out),
              "packed_adapter_bytes": args.out.stat().st_size,
              "tensor_count": len(tensors), "factor_stats": totals,
              "byte_ledger": {"factor_code_bytes": code_bytes,
                              "factor_scale_bytes": scale_bytes,
                              "router_bytes": router_bytes,
                              "selected_factor_codes_per_token": M15.M13.L * M15.K *
                              (M15.M13.D // 2 * 32 + M15.R // 2 * M15.M13.D),
                              "selected_factor_scales_per_token": M15.M13.L * M15.K *
                              (M15.R + M15.M13.D) * 4},
              "runtime": runtime,
              "scope": "weight-only packed export; quality and C parity untested"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
