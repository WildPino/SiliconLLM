#!/usr/bin/env python3
"""Bind GigaChat BF16 GGUF and plan an organ-selective low-bit target.

This emits a per-tensor llama-quantize override file and a descriptor-derived
payload ledger. It does not quantize weights or claim retained quality/speed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from gguf_active_ledger import expected_names, integer_field, price_tensor


EXPECTED_SOURCE_BYTES = 21_356_264_448
EXPECTED_SOURCE_SHA256 = "fabc8056f57e230ae9e6aadceb45f4abf9e5d7031fbfe8eca2671d871ef35d47"
TARGET_TYPES = {
    "mla": "IQ2_XS", "routed": "IQ2_XS", "shared": "IQ2_XS",
    "dense0": "Q4_K", "head": "Q3_K", "embedding": "Q4_K",
    "router": "F32", "control": "F32",
}
REQUIRES_IMATRIX = {"IQ2_XS"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gguf-py", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tensor-types-out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.tensor_types_out.exists():
        parser.error("one or both output files already exist")
    if args.model.stat().st_size != EXPECTED_SOURCE_BYTES:
        parser.error("source BF16 GGUF size mismatch")
    if sha256(args.model) != EXPECTED_SOURCE_SHA256:
        parser.error("source BF16 GGUF SHA-256 mismatch")
    if not (args.gguf_py / "gguf").is_dir():
        parser.error("gguf-py directory unavailable")
    sys.path.insert(0, str(args.gguf_py))
    from gguf import GGUFReader
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

    reader = GGUFReader(str(args.model))
    arch = reader.get_field("general.architecture")
    if arch is None or arch.contents() != "deepseek2":
        raise ValueError("unexpected GGUF architecture")
    layers = integer_field(reader, "deepseek2.block_count")
    dense_layers = integer_field(reader, "deepseek2.leading_dense_block_count")
    experts = integer_field(reader, "deepseek2.expert_count")
    selected = integer_field(reader, "deepseek2.expert_used_count")
    vocab = integer_field(reader, "deepseek2.vocab_size")
    if (layers, dense_layers, experts, selected, vocab) != (26, 1, 64, 4, 128256):
        raise ValueError("unexpected GigaChat topology")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if len(tensors) != len(reader.tensors) or set(tensors) != expected_names(layers, dense_layers):
        raise ValueError("source tensor inventory differs from pinned GigaChat base")
    source_rows = defaultdict(lambda: {"tensor_count": 0, "stored_bytes": 0, "active_bytes": 0})
    target_rows = defaultdict(lambda: {"tensor_count": 0, "stored_bytes": 0, "active_bytes": 0,
                                       "type_counts": defaultdict(int)})
    overrides = []
    imatrix_names = []
    fallback_names = []
    for name, tensor in sorted(tensors.items()):
        organ, old_active_bytes, active_elements = price_tensor(tensor, experts, selected, vocab)
        target_name = TARGET_TYPES[organ]
        # attn_k_b has a 128-element GGUF innermost dimension. IQ2_XS uses
        # 256-element blocks, so use the compatible 32-element Q4_0 format.
        if name.endswith("attn_k_b.weight"):
            if organ != "mla" or int(tensor.shape[0]) != 128:
                raise ValueError("unexpected MLA key matrix shape")
            target_name = "Q4_0"
            fallback_names.append(name)
        target_type = GGMLQuantizationType[target_name]
        block, type_bytes = GGML_QUANT_SIZES[target_type]
        elements = int(tensor.n_elements)
        if int(tensor.shape[0]) % block or elements % block or active_elements % block:
            raise ValueError(f"{name} cannot use {target_name} block size {block}")
        stored_bytes = elements // block * type_bytes
        active_bytes = active_elements // block * type_bytes
        source_rows[organ]["tensor_count"] += 1
        source_rows[organ]["stored_bytes"] += int(tensor.n_bytes)
        source_rows[organ]["active_bytes"] += old_active_bytes
        target_rows[organ]["tensor_count"] += 1
        target_rows[organ]["stored_bytes"] += stored_bytes
        target_rows[organ]["active_bytes"] += active_bytes
        target_rows[organ]["type_counts"][target_name] += 1
        if target_name in REQUIRES_IMATRIX:
            imatrix_names.append(name)
        overrides.append(f"^{re.escape(name)}$={target_name.lower()}")
    source_active = sum(row["active_bytes"] for row in source_rows.values())
    target_active = sum(row["active_bytes"] for row in target_rows.values())
    source_stored = sum(row["stored_bytes"] for row in source_rows.values())
    target_stored = sum(row["stored_bytes"] for row in target_rows.values())
    if source_stored + reader.data_offset != args.model.stat().st_size:
        raise ValueError("source header/payload bytes do not reconcile")
    result = {
        "schema": "meth04_gigachat_lowbit_preflight_v1",
        "scope": "descriptor arithmetic and explicit type map only; no transformed weights, quality or C rate",
        "source": {"path": str(args.model.resolve()), "bytes": EXPECTED_SOURCE_BYTES,
                   "sha256": EXPECTED_SOURCE_SHA256, "gguf_data_offset": int(reader.data_offset),
                   "tensor_count": len(tensors), "expert_count": experts,
                   "expert_used_count": selected},
        "target_types": TARGET_TYPES,
        "mla_attn_k_b_exception": {"type": "Q4_0", "tensor_count": len(fallback_names),
                                     "reason": "GGUF innermost dimension 128 is shorter than IQ2_XS 256-element block"},
        "source_active_payload_bytes_per_token": source_active,
        "target_active_payload_bytes_per_token": target_active,
        "target_margin_below_560mb_bytes": 560_000_000 - target_active,
        "target_payload_ms_at_40_gb_s": target_active / 40e9 * 1000,
        "source_stored_tensor_payload_bytes": source_stored,
        "target_stored_tensor_payload_bytes": target_stored,
        "imatrix_required_tensor_count": len(imatrix_names),
        "imatrix_required_tensors_sha256": hashlib.sha256("\n".join(imatrix_names).encode()).hexdigest(),
        "source_organs": {key: value for key, value in sorted(source_rows.items())},
        "target_organs": {key: {**value, "type_counts": dict(value["type_counts"])}
                          for key, value in sorted(target_rows.items())},
        "tensor_type_file": str(args.tensor_types_out.resolve()),
        "tensor_type_file_sha256": None,
        "converter_requirements": "pinned llama-quantize with --tensor-type-file; IQ2_XS requires a donor-specific importance matrix; source BF16 avoids requantization",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tensor_types_out.parent.mkdir(parents=True, exist_ok=True)
    args.tensor_types_out.write_text("\n".join(overrides) + "\n", encoding="utf-8", newline="\n")
    result["tensor_type_file_sha256"] = sha256(args.tensor_types_out)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("source active", source_active, "target active", target_active,
          "margin", 560_000_000 - target_active, "IQ2 imatrix tensors", len(imatrix_names))


if __name__ == "__main__":
    main()
