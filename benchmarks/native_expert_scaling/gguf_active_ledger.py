#!/usr/bin/env python3
"""Price active GGUF weight payload for pinned sparse donor topologies.

This is a descriptor calculation. It does not execute a model or measure DRAM
traffic. It accepts a full local GGUF or a validated HTTP Range prefix
containing the complete header; the caller must separately verify the Range
response and full-file identity. Routed tensors store all experts along the
last GGUF dimension; one token selects ``expert_used_count`` of them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


MLA = {
    "attn_q.weight", "attn_kv_a_mqa.weight", "attn_k_b.weight",
    "attn_v_b.weight", "attn_output.weight",
}
CONTROLS = {"attn_norm.weight", "attn_kv_a_norm.weight", "ffn_norm.weight"}
DENSE = {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}
ROUTED = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}
SHARED = {"ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"}
MOE_EXTRA = {"ffn_gate_inp.weight", "exp_probs_b.bias"}
ROOT = {"output.weight", "token_embd.weight", "output_norm.weight"}
PINNED_LARGE_ACTIVE_ELEMENTS = 1_628_078_080
GRANITE_ATTENTION_LAYERS = {5, 15, 25, 35}
GRANITE_ATTENTION = {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"}
GRANITE_SSM = {"ssm_a", "ssm_d", "ssm_conv1d.weight", "ssm_conv1d.bias", "ssm_dt.bias",
               "ssm_in.weight", "ssm_norm.weight", "ssm_out.weight"}
GRANITE_COMMON = {"attn_norm.weight", "ffn_norm.weight", "ffn_gate_inp.weight"} | ROUTED | SHARED
GRANITE_ACTIVE_CONFIG_ELEMENTS = 1_465_350_720


class LedgerError(ValueError):
    pass


def integer_field(reader: Any, name: str) -> int:
    field = reader.get_field(name)
    if field is None:
        raise LedgerError(f"missing GGUF field: {name}")
    return int(field.contents())


def expected_names(layers: int, dense_layers: int) -> set[str]:
    names = set(ROOT)
    for layer in range(layers):
        prefix = f"blk.{layer}."
        suffixes = MLA | CONTROLS | (DENSE if layer < dense_layers else ROUTED | SHARED | MOE_EXTRA)
        names.update(prefix + suffix for suffix in suffixes)
    return names


def expected_granite_names() -> set[str]:
    names = {"output_norm.weight", "token_embd.weight"}
    for layer in range(40):
        suffixes = GRANITE_COMMON | (GRANITE_ATTENTION if layer in GRANITE_ATTENTION_LAYERS else GRANITE_SSM)
        names.update(f"blk.{layer}.{suffix}" for suffix in suffixes)
    return names


def price_tensor(tensor: Any, experts: int, selected: int, vocab: int) -> tuple[str, int, int]:
    """Return organ, addressed bytes and active elements for one token."""
    name = tensor.name
    size = int(tensor.n_bytes)
    elements = int(tensor.n_elements)
    shape = tuple(int(dim) for dim in tensor.shape)
    if size <= 0 or elements <= 0:
        raise LedgerError(f"empty tensor: {name}")
    if name in ROOT:
        if name == "output.weight":
            return "head", size, elements
        if name == "token_embd.weight":
            if vocab not in shape or size % vocab or elements % vocab:
                raise LedgerError("embedding rows do not divide evenly by vocab size")
            return "embedding", size // vocab, elements // vocab
        return "control", size, elements
    suffix = name.split(".", 2)[-1]
    if suffix in MLA:
        return "mla", size, elements
    if suffix in ROUTED:
        if not shape or shape[-1] != experts or size % experts or elements % experts:
            raise LedgerError(f"routed tensor has no even expert axis: {name}")
        return "routed", size // experts * selected, elements // experts * selected
    if suffix in SHARED:
        return "shared", size, elements
    if suffix in DENSE:
        return "dense0", size, elements
    if suffix == "ffn_gate_inp.weight":
        return "router", size, elements
    if suffix in CONTROLS or suffix == "exp_probs_b.bias":
        return "control", size, elements
    raise LedgerError(f"unclassified tensor: {name}")


def calculate(reader: Any, file_size: int) -> dict[str, Any]:
    arch = reader.get_field("general.architecture")
    if arch is None or arch.contents() != "deepseek2":
        raise LedgerError("this profile requires a deepseek2 GGUF")
    layers = integer_field(reader, "deepseek2.block_count")
    dense_layers = integer_field(reader, "deepseek2.leading_dense_block_count")
    experts = integer_field(reader, "deepseek2.expert_count")
    selected = integer_field(reader, "deepseek2.expert_used_count")
    vocab = integer_field(reader, "deepseek2.vocab_size")
    if (layers, dense_layers, experts, selected, vocab) != (26, 1, 64, 4, 128256):
        raise LedgerError("GGUF topology differs from pinned GigaChat 3.1 base profile")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if len(tensors) != len(reader.tensors):
        raise LedgerError("duplicate tensor name")
    expected = expected_names(layers, dense_layers)
    if set(tensors) != expected:
        raise LedgerError(f"tensor inventory differs: missing={sorted(expected-set(tensors))}, extra={sorted(set(tensors)-expected)}")

    organs: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "tensor_count": 0, "stored_bytes": 0, "active_payload_bytes": 0,
        "active_elements": 0, "active_payload_by_type": defaultdict(int),
    })
    for name, tensor in sorted(tensors.items()):
        organ, active_bytes, active_elements = price_tensor(tensor, experts, selected, vocab)
        row = organs[organ]
        row["tensor_count"] += 1
        row["stored_bytes"] += int(tensor.n_bytes)
        row["active_payload_bytes"] += active_bytes
        row["active_elements"] += active_elements
        row["active_payload_by_type"][tensor.tensor_type.name] += active_bytes
    large = ("mla", "routed", "shared", "dense0", "router", "head")
    large_elements = sum(organs[name]["active_elements"] for name in large)
    if large_elements != PINNED_LARGE_ACTIVE_ELEMENTS:
        raise LedgerError(f"active large-matrix element count {large_elements} disagrees with organ audit")
    stored = sum(row["stored_bytes"] for row in organs.values())
    active = sum(row["active_payload_bytes"] for row in organs.values())
    if stored > file_size:
        raise LedgerError("tensor payload exceeds GGUF file size")
    return {
        "schema": "gguf_active_ledger_gigachat31_base_v1",
        "basis": "GGUF descriptor n_bytes; selected expert slices; one embedding row; no cache assumption",
        "scope": "compressed weight payload addressed once per autoregressive token; not measured DRAM traffic, latency, or quality",
        "topology": {"layers": layers, "leading_dense_layers": dense_layers,
                     "experts_per_moe_layer": experts, "selected_experts_per_token": selected,
                     "vocab": vocab, "tensor_count": len(tensors)},
        "file_bytes": file_size,
        "stored_tensor_payload_bytes": stored,
        "active_large_matrix_elements": large_elements,
        "active_payload_bytes_per_token": active,
        "payload_gb_per_second_at_50_tok_s": active * 50 / 1e9,
        "payload_ms_at_40_gb_s": active / 40e9 * 1000,
        "organs": {name: {**row, "active_payload_by_type": dict(sorted(row["active_payload_by_type"].items()))}
                   for name, row in sorted(organs.items())},
    }


def calculate_granite(reader: Any, file_size: int) -> dict[str, Any]:
    arch = reader.get_field("general.architecture")
    if arch is None or arch.contents() != "granitehybrid":
        raise LedgerError("this profile requires a granitehybrid GGUF")
    layers = integer_field(reader, "granitehybrid.block_count")
    experts = integer_field(reader, "granitehybrid.expert_count")
    selected = integer_field(reader, "granitehybrid.expert_used_count")
    vocab = integer_field(reader, "granitehybrid.vocab_size")
    shared_width = integer_field(reader, "granitehybrid.expert_shared_feed_forward_length")
    if (layers, experts, selected, vocab, shared_width) != (40, 64, 6, 100352, 1024):
        raise LedgerError("GGUF topology differs from pinned Granite H Tiny base profile")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if len(tensors) != len(reader.tensors):
        raise LedgerError("duplicate tensor name")
    expected = expected_granite_names()
    if set(tensors) != expected:
        raise LedgerError(f"tensor inventory differs: missing={sorted(expected-set(tensors))}, extra={sorted(set(tensors)-expected)}")

    organs: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "tensor_count": 0, "stored_bytes": 0, "active_payload_bytes": 0,
        "active_elements": 0, "active_payload_by_type": defaultdict(int),
    })
    for name, tensor in sorted(tensors.items()):
        size = int(tensor.n_bytes)
        elements = int(tensor.n_elements)
        shape = tuple(int(dim) for dim in tensor.shape)
        if size <= 0 or elements <= 0:
            raise LedgerError(f"empty tensor: {name}")
        suffix = name.split(".", 2)[-1]
        if name == "token_embd.weight":
            # Tied output head: all vocabulary rows are read each decoded token.
            organ = "tied_head_embedding"
        elif suffix in ROUTED:
            if not shape or shape[-1] != experts or size % experts or elements % experts:
                raise LedgerError(f"routed tensor has no even expert axis: {name}")
            organ = "routed"
            size = size // experts * selected
            elements = elements // experts * selected
        elif suffix in SHARED:
            organ = "shared"
        elif suffix == "ffn_gate_inp.weight":
            organ = "router"
        elif suffix in GRANITE_SSM:
            organ = "ssm"
        elif suffix in GRANITE_ATTENTION:
            organ = "attention"
        elif name == "output_norm.weight" or suffix in ("attn_norm.weight", "ffn_norm.weight"):
            organ = "control"
        else:
            raise LedgerError(f"unclassified tensor: {name}")
        row = organs[organ]
        row["tensor_count"] += 1
        row["stored_bytes"] += int(tensor.n_bytes)
        row["active_payload_bytes"] += size
        row["active_elements"] += elements
        row["active_payload_by_type"][tensor.tensor_type.name] += size
    stored = sum(row["stored_bytes"] for row in organs.values())
    active = sum(row["active_payload_bytes"] for row in organs.values())
    active_elements = sum(row["active_elements"] for row in organs.values())
    if stored > file_size:
        raise LedgerError("tensor payload exceeds GGUF file size")
    delta = active_elements - GRANITE_ACTIVE_CONFIG_ELEMENTS
    if abs(delta) > GRANITE_ACTIVE_CONFIG_ELEMENTS * 0.0001:
        raise LedgerError("GGUF active element count disagrees with independent config analyzer")
    return {
        "schema": "gguf_active_ledger_granite_h_tiny_base_v1",
        "basis": "GGUF descriptor n_bytes; six expert slices; tied head read in full; no cache assumption",
        "scope": "compressed weight payload addressed once per autoregressive token; not measured DRAM traffic, latency, or quality",
        "topology": {"layers": layers, "attention_layers": sorted(GRANITE_ATTENTION_LAYERS),
                     "mamba_layers": layers-len(GRANITE_ATTENTION_LAYERS),
                     "experts_per_moe_layer": experts, "selected_experts_per_token": selected,
                     "vocab": vocab, "tensor_count": len(tensors)},
        "file_bytes": file_size,
        "stored_tensor_payload_bytes": stored,
        "active_elements_per_token": active_elements,
        "config_analyzer_active_elements_per_token": GRANITE_ACTIVE_CONFIG_ELEMENTS,
        "config_analyzer_delta_elements": delta,
        "active_payload_bytes_per_token": active,
        "payload_gb_per_second_at_50_tok_s": active * 50 / 1e9,
        "payload_ms_at_40_gb_s": active / 40e9 * 1000,
        "organs": {name: {**row, "active_payload_by_type": dict(sorted(row["active_payload_by_type"].items()))}
                   for name, row in sorted(organs.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="complete local GGUF")
    source.add_argument("--header-file", type=Path, help="validated HTTP Range prefix containing complete GGUF header")
    parser.add_argument("--profile", choices=("gigachat31_base", "granite_h_tiny_base"), default="gigachat31_base")
    parser.add_argument("--gguf-py", required=True, type=Path, help="pinned llama.cpp gguf-py directory")
    parser.add_argument("--expect-file-bytes", type=int, default=0)
    parser.add_argument("--hash-file", action="store_true", help="stream and verify full file identity in output")
    parser.add_argument("--out", type=Path, help="new JSON result; refuses overwrite")
    args = parser.parse_args()
    input_file = args.model or args.header_file
    if not input_file.is_file() or not (args.gguf_py / "gguf").is_dir():
        parser.error("GGUF input or gguf-py directory is unavailable")
    if args.header_file and (not args.expect_file_bytes or args.hash_file):
        parser.error("header-only input requires --expect-file-bytes and cannot use --hash-file")
    file_size = args.expect_file_bytes if args.header_file else input_file.stat().st_size
    if args.model and args.expect_file_bytes and file_size != args.expect_file_bytes:
        parser.error("GGUF file size differs from pinned artifact")
    sys.path.insert(0, str(args.gguf_py))
    from gguf import GGUFReader
    if args.header_file:
        import numpy as np

        class HeaderOnlyReader(GGUFReader):
            def _get(self, offset: int, dtype: Any, count: int = 1, override_order: Any = None) -> Any:
                if hasattr(self, "data_offset") and offset >= self.data_offset:
                    values = np.broadcast_to(np.zeros(1, dtype=dtype), (int(count),))
                    return values.view(values.dtype.newbyteorder(self.byte_order if override_order is None else override_order))
                return super()._get(offset, dtype, count, override_order)

        reader = HeaderOnlyReader(str(input_file))
        if reader.data_offset > input_file.stat().st_size:
            raise LedgerError("Range prefix ends before GGUF payload start")
    else:
        reader = GGUFReader(str(input_file))

    report = (calculate(reader, file_size) if args.profile == "gigachat31_base"
              else calculate_granite(reader, file_size))
    if args.header_file and reader.data_offset + report["stored_tensor_payload_bytes"] != file_size:
        raise LedgerError("reported full size does not reconcile with header offset and tensor payloads")
    report["model"] = str(args.model.resolve()) if args.model else None
    report["header_file"] = str(args.header_file.resolve()) if args.header_file else None
    report["header_bytes_local"] = input_file.stat().st_size if args.header_file else None
    report["header_sha256"] = hashlib.sha256(input_file.read_bytes()).hexdigest() if args.header_file else None
    report["full_file_sha256"] = None
    if args.hash_file:
        digest = hashlib.sha256()
        with args.model.open("rb", buffering=0) as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        report["full_file_sha256"] = digest.hexdigest()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        if args.out.exists():
            parser.error("output already exists")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
