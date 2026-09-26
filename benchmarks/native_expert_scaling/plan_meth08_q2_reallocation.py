#!/usr/bin/env python3
"""Price a budgeted GigaChat Q2_K expert/head/dense precision swap."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from gguf_active_ledger import expected_names, integer_field, price_tensor
from plan_gigachat_lowbit import EXPECTED_SOURCE_BYTES, EXPECTED_SOURCE_SHA256, sha256
from plan_meth07_organ_precision import BASE_ACTIVE_BYTES, BASE_MAP_SHA256, read_base_map


TARGET_QUANT = {"routed": "Q2_K", "shared": "Q2_K", "head": "Q2_K", "dense0": "Q2_K"}
EXPECTED_CHANGED = {"routed": 75, "shared": 75, "head": 1, "dense0": 3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-map", type=Path, required=True)
    parser.add_argument("--gguf-py", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tensor-types-out", type=Path, required=True)
    parser.add_argument("--experts-only", action="store_true",
                        help="METH-09 diagnostic: change experts to Q2_K without head/dense payment")
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
    arm = dict(base)
    target_quant = ({key: value for key, value in TARGET_QUANT.items() if key in ("routed", "shared")}
                    if args.experts_only else TARGET_QUANT)
    expected_changed = ({"routed": 75, "shared": 75} if args.experts_only else EXPECTED_CHANGED)
    changed: dict[str, list[str]] = defaultdict(list)
    organs: dict[str, dict[str, int]] = defaultdict(lambda: {"base_active": 0, "arm_active": 0,
                                                               "base_stored": 0, "arm_stored": 0})
    for name, tensor in sorted(tensors.items()):
        organ, _, active_elements = price_tensor(tensor, 64, 4, 128256)
        if organ in target_quant:
            arm[name] = target_quant[organ]
            if arm[name] != base[name]:
                changed[organ].append(name)
        elements = int(tensor.n_elements)
        for quant, prefix in ((base[name], "base"), (arm[name], "arm")):
            block, type_bytes = GGML_QUANT_SIZES[GGMLQuantizationType[quant]]
            if int(tensor.shape[0]) % block or elements % block or active_elements % block:
                raise ValueError(f"{name} cannot use {quant} block size {block}")
            organs[organ][f"{prefix}_active"] += active_elements // block * type_bytes
            organs[organ][f"{prefix}_stored"] += elements // block * type_bytes
    if {key: len(value) for key, value in changed.items()} != expected_changed:
        raise ValueError("unexpected organ/tensor replacements")
    base_active = sum(row["base_active"] for row in organs.values())
    if base_active != BASE_ACTIVE_BYTES:
        raise ValueError("base active bytes do not reproduce METH-04/METH-06")
    active = sum(row["arm_active"] for row in organs.values())
    stored = sum(row["arm_stored"] for row in organs.values())
    lines = [f"^{re.escape(name)}$={arm[name].lower()}" for name in sorted(tensors)]
    args.tensor_types_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tensor_types_out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    report = {
        "schema": ("meth09_gigachat_q2_experts_only_plan_v1" if args.experts_only
                   else "meth08_gigachat_q2_reallocation_plan_v1"),
        "scope": "BF16-source descriptor preflight; quality and native speed unmeasured",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "base_map_sha256": BASE_MAP_SHA256,
        "arm_map_sha256": sha256(args.tensor_types_out),
        "changed_tensors_by_organ": dict(sorted(changed.items())),
        "changed_tensor_count": sum(map(len, changed.values())),
        "base_active_bytes_per_token": base_active,
        "arm_active_bytes_per_token": active,
        "arm_stored_tensor_bytes": stored,
        "active_margin_below_540mb_bytes": 540_000_000 - active,
        "arm_payload_ms_at_40_gb_s": active / 40e9 * 1000,
        "organs": dict(sorted(organs.items())),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: report[key] for key in ("changed_tensor_count", "arm_active_bytes_per_token",
                                                    "arm_stored_tensor_bytes", "active_margin_below_540mb_bytes")}, indent=2))


if __name__ == "__main__":
    main()
