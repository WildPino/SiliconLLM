#!/usr/bin/env python3
"""METH-34: export trained rank-8 factors as per-weight 15-level LUT codes."""
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


MAX_SECONDS = 15 * 60
MAX_RSS_BYTES = 20 * (1 << 30)


def check_budget(start):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    if elapsed > MAX_SECONDS:
        raise TimeoutError(f"METH-34 export wall stop: {elapsed:.1f}s")
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"METH-34 export RSS stop: {rss}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss}


def quantize_rows(weight):
    assert weight.dtype == torch.float32 and weight.ndim == 3
    experts, outputs, inputs = weight.shape
    rows = weight.reshape(-1, inputs)
    scale = rows.abs().amax(dim=1) / 7.0
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.round(rows / safe[:, None]).clamp(-7, 7).to(torch.int8)
    reconstructed = q.float() * scale[:, None]
    error = (rows - reconstructed).square().sum().item()
    norm = rows.square().sum().item()
    assert bool((q >= -7).all()) and bool((q <= 7).all())
    return (q.reshape(experts, outputs, inputs).contiguous(),
            scale.reshape(experts, outputs).contiguous(),
            {"weight_elements": rows.numel(),
             "nonzero_codes": int((q != 0).sum()),
             "squared_error": error, "weight_squared_norm": norm,
             "relative_squared_error": error / norm if norm else 0.0})


def pack_codes(q, pad_outputs):
    assert q.dtype == torch.int8 and q.ndim == 3
    experts, outputs, inputs = q.shape
    assert pad_outputs >= outputs and pad_outputs % 32 == 0
    packed = torch.full((experts, inputs, pad_outputs), 7, dtype=torch.uint8)
    packed[:, :, :outputs] = (q.permute(0, 2, 1).to(torch.int16) + 7).to(torch.uint8)
    assert bool((packed <= 14).all())
    return packed.contiguous()


def unpack_codes(packed, outputs):
    assert packed.dtype == torch.uint8 and packed.ndim == 3
    assert outputs <= packed.shape[2] and bool((packed <= 14).all())
    return (packed[:, :, :outputs].permute(0, 2, 1).to(torch.int16) - 7
            ).to(torch.int8).contiguous()


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
        meta = archive.metadata()
        assert meta["model_sha256"] == M15.M13.MODEL_SHA
        assert meta["output_factor"] == "0.5"
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
    metadata = {"format": "QWEN25_R8_E128_I4_LUT_V1",
                "core_sha256": M25.CORE_SHA,
                "source_adapter_sha256": M20.ADAPTER_SHA,
                "source_model_sha256": M15.M13.MODEL_SHA,
                "output_factor": "0.5",
                "quantizer": "per_row_max_abs_div7_round_even_clamp_minus7_plus7",
                "code_mapping": "q+7; 15 reserved",
                "code_layout": "expert,input,output_padded"}
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
    assert (code_bytes, scale_bytes, router_bytes) == (110100480, 11108352, 11010048)
    selected_codes = M15.M13.L * M15.K * (M15.M13.D * 32 + M15.R * M15.M13.D)
    selected_scales = M15.M13.L * M15.K * (M15.R + M15.M13.D) * 4
    assert selected_codes + selected_scales == 3787776
    runtime = check_budget(start)
    report = {"experiment": "METH-34-export", "format": metadata["format"],
              "core_sha256": M25.CORE_SHA,
              "source_adapter_sha256": M20.ADAPTER_SHA,
              "packed_adapter_sha256": M15.M13.sha256(args.out),
              "packed_adapter_bytes": args.out.stat().st_size,
              "tensor_count": len(tensors), "factor_stats": totals,
              "byte_ledger": {"factor_code_bytes": code_bytes,
                              "factor_scale_bytes": scale_bytes,
                              "router_bytes": router_bytes,
                              "selected_factor_codes_per_token": selected_codes,
                              "selected_factor_scales_per_token": selected_scales,
                              "selected_factor_total_per_token": selected_codes + selected_scales},
              "runtime": runtime,
              "scope": "weight-only LUT-indexed export; activation LUT and quality untested"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
