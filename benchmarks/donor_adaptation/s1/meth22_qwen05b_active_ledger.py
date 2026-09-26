#!/usr/bin/env python3
"""METH-22: exact Qwen0.5B organ counts and optimistic active-byte floors."""
import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig

import meth15_zero_residual_expert_smoke as M15
import meth20_half_adapter_generation as M20


BANDWIDTH_BYTES_PER_SEC = 40_000_000_000
DESIGN_BYTES_PER_TOKEN = 560_000_000
RATE_MS = 20.0
ROOT = Path(__file__).resolve().parents[3]


def count_organs(path):
    count = {"tied_head_embed": 0, "ffn_matrix": 0,
             "attention_matrix": 0, "control_vector": 0}
    tensors = {key: 0 for key in count}
    with safe_open(path, framework="pt", device="cpu") as source:
        names = source.keys()
        assert "model.embed_tokens.weight" in names
        assert "lm_head.weight" not in names
        for name in names:
            shape = source.get_slice(name).get_shape()
            n = 1
            for dim in shape:
                n *= dim
            if name == "model.embed_tokens.weight":
                organ = "tied_head_embed"
            elif ".mlp." in name and len(shape) == 2:
                organ = "ffn_matrix"
            elif ".self_attn." in name and len(shape) == 2:
                organ = "attention_matrix"
            else:
                assert len(shape) == 1, (name, shape)
                organ = "control_vector"
            count[organ] += n
            tensors[organ] += 1
    assert count == {"tied_head_embed": 136134656,
                     "ffn_matrix": 313786368,
                     "attention_matrix": 44040192,
                     "control_vector": 71552}, count
    return count, tensors


def scenario(count, adapter, head_bytes, body_bytes):
    assert all((count[key] * scale).is_integer()
               for key, scale in (("tied_head_embed", head_bytes),
                                  ("ffn_matrix", body_bytes),
                                  ("attention_matrix", body_bytes)))
    head = int(count["tied_head_embed"] * head_bytes)
    ffn = int(count["ffn_matrix"] * body_bytes)
    attention = int(count["attention_matrix"] * body_bytes)
    control = count["control_vector"] * 4
    total = head + ffn + attention + control + sum(adapter.values())
    return {"head_bytes": head, "ffn_bytes": ffn,
            "attention_bytes": attention, "control_bytes": control,
            "adapter_router_bytes": adapter["router"],
            "adapter_selected_factor_bytes": adapter["selected_factors"],
            "active_bytes_per_token_lower_bound": total,
            "floor_ms_at_40GBps": total / BANDWIDTH_BYTES_PER_SEC * 1000,
            "under_20ms_payload_only": total <= RATE_MS / 1000 * BANDWIDTH_BYTES_PER_SEC,
            "under_560MB_design_payload": total <= DESIGN_BYTES_PER_TOKEN}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    source = hf_hub_download(M15.M13.MODEL, "model.safetensors",
                             revision=M15.M13.REV, local_files_only=True)
    assert M15.M13.sha256(source) == M15.M13.MODEL_SHA
    assert M15.M13.sha256(M20.ADAPTER) == M20.ADAPTER_SHA
    config = AutoConfig.from_pretrained(M15.M13.MODEL, revision=M15.M13.REV,
                                        local_files_only=True)
    assert config.tie_word_embeddings is True
    assert (config.hidden_size, config.intermediate_size,
            config.num_hidden_layers, config.vocab_size) == (896, 4864, 24, 151936)
    count, tensors = count_organs(source)
    assert sum(count.values()) == 494032768
    router = M15.M13.L * M15.E * M15.M13.D * 4
    selected = M15.M13.L * M15.K * (2 * M15.M13.D * M15.R) * 4
    assert router == 11010048 and selected == 5505024
    adapter = {"router": router, "selected_factors": selected}
    cases = {
        "bf16_core_fp32_adapter": scenario(count, adapter, 2.0, 2.0),
        "fp32_core_fp32_adapter": scenario(count, adapter, 4.0, 4.0),
        "ideal_one_byte_body_fp32_tied_head": scenario(count, adapter, 4.0, 1.0),
        "ideal_half_byte_body_fp32_tied_head": scenario(count, adapter, 4.0, 0.5),
        "ideal_one_byte_all_matrices": scenario(count, adapter, 1.0, 1.0),
        "ideal_half_byte_head_one_byte_body": scenario(count, adapter, 0.5, 1.0),
    }
    report = {"experiment": "METH-22", "model": M15.M13.MODEL,
              "model_revision": M15.M13.REV,
              "model_sha256": M15.M13.MODEL_SHA,
              "adapter_sha256": M20.ADAPTER_SHA,
              "shape": {"hidden": config.hidden_size,
                        "intermediate": config.intermediate_size,
                        "layers": config.num_hidden_layers,
                        "vocab": config.vocab_size,
                        "tied_head": config.tie_word_embeddings},
              "organ_parameters": count, "organ_tensors": tensors,
              "total_donor_parameters": sum(count.values()),
              "adapter": {**adapter, "E": M15.E, "K": M15.K,
                          "rank": M15.R, "layers": M15.M13.L,
                          "stored_fp32_factor_parameters": M15.M13.L * M15.E *
                          2 * M15.M13.D * M15.R,
                          "stored_fp32_router_parameters": M15.M13.L * M15.E *
                          M15.M13.D},
              "assumptions": {"bandwidth_bytes_per_second": BANDWIDTH_BYTES_PER_SEC,
                              "design_bytes_per_token": DESIGN_BYTES_PER_TOKEN,
                              "target_ms_per_token": RATE_MS,
                              "format_overheads_omitted": True,
                              "latency_and_DRAM_traffic_measured": False},
              "scenarios": cases,
              "decision": "one_byte_body_fp32_head_exceeds_20ms_at_40GBps_streaming"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"organ_parameters": count,
                      "adapter": adapter,
                      "scenarios": cases,
                      "decision": report["decision"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
