#!/usr/bin/env python3
"""METH-81: price packed grouped-Q4 donor core from bound tensor headers."""

import argparse
import json
from pathlib import Path
import time

from huggingface_hub import hf_hub_download
import psutil
from safetensors import safe_open

import meth15_zero_residual_expert_smoke as M15
import meth24_export_r8_core as M24
import meth42_instruct_prompt_manifest as M42
import meth57_product_key_external_audit as M57


ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
M60_LEDGER = DIR / "meth60_r8_core_organ_ablation_result.json"
M60_LEDGER_SHA = "22bac64fa592f6e56c77111082ea51fc10e7476db2d37ed9b445e0fc01d28df3"
GROUP = 64
LIMIT_BYTES = 560_000_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    source = Path(hf_hub_download(M42.MODEL, "model.safetensors",
                                  revision=M42.REV, local_files_only=True))
    assert M15.M13.sha256(source) == M57.MODEL_SHA
    assert M15.M13.sha256(M60_LEDGER) == M60_LEDGER_SHA
    ledger = json.loads(M60_LEDGER.read_text(encoding="utf-8"))["byte_ledger"]["all_r8"]
    assert ledger["router_bytes"] == 5_652_480
    assert ledger["selected_factor_bytes"] == 2_752_512
    overhead = ledger["router_bytes"] + ledger["selected_factor_bytes"]
    organs = {name: {"tensors": 0, "parameters": 0, "q4_bytes": 0,
                     "bf16_bytes": 0, "control_bytes": 0}
              for name in ("tied_head", "ffn", "attention", "control")}
    matrices = []
    with safe_open(str(source), framework="pt", device="cpu") as archive:
        names = list(archive.keys())
        for name in names:
            shape = tuple(archive.get_slice(name).get_shape())
            organ = M24.classify(name, shape)
            record = organs[organ]
            record["tensors"] += 1
            parameters = 1
            for dimension in shape:
                parameters *= dimension
            record["parameters"] += parameters
            if organ == "control":
                record["control_bytes"] += 4 * parameters
                continue
            rows, width = shape
            code_bytes = rows * ((width + 1) // 2)
            scale_bytes = rows * ((width + GROUP - 1) // GROUP) * 2
            q4_bytes = code_bytes + scale_bytes
            bf16_bytes = 2 * parameters
            record["q4_bytes"] += q4_bytes
            record["bf16_bytes"] += bf16_bytes
            matrices.append({"name": name, "organ": organ, "shape": shape,
                             "code_bytes": code_bytes,
                             "scale_bytes": scale_bytes,
                             "q4_bytes": q4_bytes,
                             "bf16_bytes": bf16_bytes,
                             "width_divisible_by_64": width % GROUP == 0})
    assert len(names) == 290 and len(matrices) == 169
    assert {name: row["tensors"] for name, row in organs.items()} == {
        "tied_head": 1, "ffn": 72, "attention": 96, "control": 121}
    assert sum(row["parameters"] for row in organs.values()) == 494_032_768
    assert sum(row["parameters"] for name, row in organs.items()
               if name != "control") == 493_961_216
    arms = {}
    for name, bf16_organs in {
        "all_grouped_q4": (),
        "bf16_attention": ("attention",),
        "bf16_tied_head": ("tied_head",),
        "bf16_tied_head_attention": ("tied_head", "attention"),
    }.items():
        core = organs["control"]["control_bytes"]
        core += sum(row["bf16_bytes"] if organ in bf16_organs else row["q4_bytes"]
                    for organ, row in organs.items() if organ != "control")
        total = core + overhead
        arms[name] = {"bf16_organs": bf16_organs, "core_bytes": core,
                      "router_bytes": ledger["router_bytes"],
                      "selected_factor_bytes": ledger["selected_factor_bytes"],
                      "ideal_addressed_total_bytes": total,
                      "under_560m": total <= LIMIT_BYTES}
    leading = arms["bf16_tied_head_attention"]
    result = {
        "experiment": "METH-81-grouped-Q4-core-payload-preflight",
        "model": M42.MODEL, "revision": M42.REV,
        "model_sha256": M57.MODEL_SHA,
        "meth60_ledger_sha256": M60_LEDGER_SHA,
        "format": {"codes": "signed symmetric int4 packed two per byte",
                   "group_width": GROUP, "scale_dtype": "float16",
                   "control_dtype": "float32", "bf16_override_dtype": "bfloat16"},
        "organs": organs, "matrices": matrices, "arms": arms,
        "all_matrix_widths_divisible_by_64": all(
            row["width_divisible_by_64"] for row in matrices),
        "decision": ("proceed_to_export_and_fresh_development_composition"
                     if leading["under_560m"] else "stop_q4_core_map_over_budget"),
        "runtime": {"elapsed_seconds": time.monotonic() - start,
                    "rss_end_bytes": psutil.Process().memory_info().rss},
    }
    assert result["runtime"]["elapsed_seconds"] <= 120
    assert result["runtime"]["rss_end_bytes"] <= 2 * (1 << 30)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"],
                      "all_widths_divisible": result["all_matrix_widths_divisible_by_64"],
                      "arms": arms, "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
