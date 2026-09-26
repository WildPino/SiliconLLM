#!/usr/bin/env python3
"""METH-36: freeze disjoint prompts and export stored rank-64 int8 router sketches."""
import argparse
import json
from pathlib import Path
import time

import psutil
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth19_half_residual_independent_audit as M19
import meth20_half_adapter_generation as M20
import meth27_r8_fresh_generation as M27
import meth35_low_rank_router as M35


ROOT = Path(__file__).resolve().parents[3]
M17_MANIFEST = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth17_fresh_document_manifest.json"
M35_RESULT = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth35_low_rank_router_result.json"
M35_SHA = "f701f1968987db0d107fe4383166837e07be985771754ee1bce36de6615710d7"
SEED = "meth36-36036"
RANK = 64
MAX_SECONDS = 20 * 60
MAX_RSS = 20 * (1 << 30)


def check_budget(start):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    if elapsed > MAX_SECONDS or rss > MAX_RSS:
        raise RuntimeError(f"METH-36 export budget: {elapsed:.1f}s, RSS {rss}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss}


def select_prompts(tokenizer):
    assert M17.sha(M17_MANIFEST.read_bytes()) == M19.M17_MANIFEST_SHA
    prior, items = M17.build_selection()
    assert json.loads(M17_MANIFEST.read_text(encoding="utf-8")) == prior
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    _, old_prompts = M27.select_prompts(tokenizer)
    old_ids = {p["source_id"] for p in old_prompts}
    chosen = []
    for category in ("code", "prose", "technical_general"):
        candidates = [x for x in items if x["category"] == category]
        candidates.sort(key=lambda x: M17.sha((SEED + "|" + x["source_id"]).encode()))
        assert len(candidates) >= 8
        for item in candidates[:8]:
            assert item["source_id"] not in old_ids
            ids = tokenizer.encode(item["text"], add_special_tokens=False)
            assert len(ids) >= 256
            chosen.append({"category": category, "source_id": item["source_id"],
                           "text_sha256": M17.sha(item["text"].encode("utf-8")),
                           "prompt_ids_sha256": M17.sha(
                               torch.tensor(ids[:256], dtype=torch.int32).numpy().tobytes()),
                           "prompt_ids": ids[:256]})
    assert len(chosen) == 24 and len({x["source_id"] for x in chosen}) == 24
    manifest = {"experiment": "METH-36", "seed": SEED,
                "m17_manifest_sha256": M19.M17_MANIFEST_SHA,
                "m35_result_sha256": M35_SHA,
                "core_sha256": M35.M25.CORE_SHA,
                "adapter_sha256": M20.ADAPTER_SHA,
                "tokenizer_fingerprint": M15.M13.TOK_FP,
                "prompt_tokens": 256,
                "selected": [{k: v for k, v in x.items() if k != "prompt_ids"}
                             for x in chosen]}
    return manifest, chosen


def quantize(sketch):
    assert sketch.dtype == torch.float32 and sketch.shape == (M15.E, RANK)
    scales = sketch.abs().amax(dim=1) / 127.0
    safe = torch.where(scales > 0, scales, torch.ones_like(scales))
    codes = torch.round(sketch / safe[:, None]).clamp(-127, 127).to(torch.int8)
    assert bool((codes >= -127).all()) and bool((codes <= 127).all())
    error = (sketch - codes.float() * scales[:, None]).square().sum().item()
    norm = sketch.square().sum().item()
    return codes.contiguous(), scales.contiguous(), error, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    start = time.monotonic()
    torch.set_num_threads(6)
    assert M17.sha(M35_RESULT.read_bytes()) == M35_SHA
    assert M15.M13.sha256(M20.ADAPTER) == M20.ADAPTER_SHA
    assert M15.M13.sha256(M35.M25.CORE) == M35.M25.CORE_SHA
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    source = hf_hub_download(M15.M13.MODEL, "model.safetensors",
                             revision=M15.M13.REV, local_files_only=True)
    assert M15.M13.sha256(source) == M15.M13.MODEL_SHA
    tokenizer = AutoTokenizer.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, local_files_only=True)
    manifest, _ = select_prompts(tokenizer)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    adapter = load_file(str(M20.ADAPTER), device="cpu")
    assert len(adapter) == 3 * M15.M13.L
    tensors = {}
    error = norm = 0.0
    for li in range(M15.M13.L):
        router = adapter[f"layers.{li}.router"]
        assert router.dtype == torch.float32 and router.shape == (M15.E, M15.M13.D)
        _, _, vh = torch.linalg.svd(router, full_matrices=False)
        basis = vh[:RANK].T.contiguous()
        sketch = (router @ basis).contiguous()
        codes, scales, layer_error, layer_norm = quantize(sketch)
        tensors[f"layers.{li}.basis"] = basis
        tensors[f"layers.{li}.codes"] = codes
        tensors[f"layers.{li}.scales"] = scales
        error += layer_error
        norm += layer_norm
        check_budget(start)
    assert len(tensors) == 3 * M15.M13.L
    metadata = {"format": "QWEN25_E128_ROUTER_R64_INT8_V1",
                "adapter_sha256": M20.ADAPTER_SHA,
                "core_sha256": M35.M25.CORE_SHA,
                "rank": str(RANK),
                "quantizer": "row_max_abs_div127_round_even_clamp_minus127_plus127"}
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.artifact), metadata=metadata)
    with safe_open(str(args.artifact), framework="pt", device="cpu") as archive:
        assert archive.metadata() == metadata
        assert len(archive.keys()) == len(tensors)
        for key, value in tensors.items():
            assert torch.equal(archive.get_tensor(key), value), key
    bytes_by_organ = {suffix: sum(t.numel() * t.element_size()
                                  for key, t in tensors.items() if key.endswith(suffix))
                      for suffix in (".basis", ".codes", ".scales")}
    runtime = check_budget(start)
    report = {"experiment": "METH-36-export",
              "manifest_sha256": M17.sha(args.manifest.read_bytes()),
              "artifact_sha256": M15.M13.sha256(args.artifact),
              "artifact_bytes": args.artifact.stat().st_size,
              "adapter_sha256": M20.ADAPTER_SHA,
              "core_sha256": M35.M25.CORE_SHA,
              "byte_ledger": bytes_by_organ,
              "relative_sketch_squared_error": error / norm,
              "runtime": runtime}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
