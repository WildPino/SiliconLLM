#!/usr/bin/env python3
"""Price organ-specific Q4_K counterfactuals for the failed GigaChat IQ2 map.

The generated arms are causal diagnostics, not eligible 540 MB/token targets.
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
from plan_gigachat_lowbit import EXPECTED_SOURCE_BYTES, EXPECTED_SOURCE_SHA256, sha256


BASE_MAP_SHA256 = "a55334bf64f39a506efba1350a3eaae034f6d4453de47d5a2e5e227b7bb44128"
BASE_ACTIVE_BYTES = 534_024_800
ARMS = {"mla_q4k": {"mla"}, "experts_q4k": {"routed", "shared"}}


def read_base_map(path: Path, names: set[str]) -> dict[str, str]:
    if sha256(path) != BASE_MAP_SHA256:
        raise ValueError("METH-04 tensor-type map SHA-256 mismatch")
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        expression, sep, quant = line.partition("=")
        if not sep or not expression.startswith("^") or not expression.endswith("$"):
            raise ValueError(f"invalid type-map line: {line}")
        name = re.sub(r"\\(.)", r"\1", expression[1:-1])
        if name in result:
            raise ValueError(f"duplicate tensor: {name}")
        result[name] = quant.upper()
    if set(result) != names:
        raise ValueError("METH-04 tensor inventory mismatch")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-map", type=Path, required=True)
    parser.add_argument("--gguf-py", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tensor-types-out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.tensor_types_out.exists():
        parser.error("output already exists")
    if args.model.stat().st_size != EXPECTED_SOURCE_BYTES or sha256(args.model) != EXPECTED_SOURCE_SHA256:
        parser.error("source BF16 GGUF identity mismatch")
    sys.path.insert(0, str(args.gguf_py))
    from gguf import GGUFReader
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

    reader = GGUFReader(str(args.model))
    if reader.get_field("general.architecture").contents() != "deepseek2":
        raise ValueError("unexpected architecture")
    topology = tuple(integer_field(reader, key) for key in (
        "deepseek2.block_count", "deepseek2.leading_dense_block_count",
        "deepseek2.expert_count", "deepseek2.expert_used_count", "deepseek2.vocab_size"))
    if topology != (26, 1, 64, 4, 128256):
        raise ValueError("unexpected topology")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if len(tensors) != 414 or set(tensors) != expected_names(26, 1):
        raise ValueError("unexpected tensor inventory")
    base = read_base_map(args.base_map, set(tensors))
    new = dict(base)
    changed: list[str] = []
    organs: dict[str, dict[str, int]] = defaultdict(lambda: {"base_active": 0, "arm_active": 0,
                                                               "base_stored": 0, "arm_stored": 0})
    for name, tensor in sorted(tensors.items()):
        organ, _, active_elements = price_tensor(tensor, 64, 4, 128256)
        if organ in ARMS[args.arm] and base[name] == "IQ2_XS":
            new[name] = "Q4_K"
            changed.append(name)
        elements = int(tensor.n_elements)
        for quant, prefix in ((base[name], "base"), (new[name], "arm")):
            block, type_bytes = GGML_QUANT_SIZES[GGMLQuantizationType[quant]]
            if int(tensor.shape[0]) % block or elements % block or active_elements % block:
                raise ValueError(f"{name} cannot use {quant} block size {block}")
            organs[organ][f"{prefix}_active"] += active_elements // block * type_bytes
            organs[organ][f"{prefix}_stored"] += elements // block * type_bytes
    if not changed or any(base[name] != "IQ2_XS" for name in changed):
        raise ValueError("arm did not replace the intended IQ2 tensors")
    base_active = sum(row["base_active"] for row in organs.values())
    if base_active != BASE_ACTIVE_BYTES:
        raise ValueError("base active bytes do not reproduce METH-04/METH-06")
    arm_active = sum(row["arm_active"] for row in organs.values())
    arm_stored = sum(row["arm_stored"] for row in organs.values())
    lines = [f"^{re.escape(name)}$={new[name].lower()}" for name in sorted(tensors)]
    args.tensor_types_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tensor_types_out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    report = {
        "schema": "meth07_gigachat_organ_precision_plan_v1",
        "scope": "BF16-source type-map and descriptor preflight; diagnostic arm, not a cost-qualified target",
        "arm": args.arm,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "base_map_sha256": BASE_MAP_SHA256,
        "arm_map_sha256": sha256(args.tensor_types_out),
        "changed_tensors": changed,
        "changed_tensor_count": len(changed),
        "base_active_bytes_per_token": base_active,
        "arm_active_bytes_per_token": arm_active,
        "arm_stored_tensor_bytes": arm_stored,
        "active_over_540mb_bytes": arm_active - 540_000_000,
        "arm_payload_ms_at_40_gb_s": arm_active / 40e9 * 1000,
        "organs": dict(sorted(organs.items())),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: report[key] for key in ("arm", "changed_tensor_count", "arm_active_bytes_per_token",
                                                    "arm_stored_tensor_bytes", "active_over_540mb_bytes")}, indent=2))


if __name__ == "__main__":
    main()
