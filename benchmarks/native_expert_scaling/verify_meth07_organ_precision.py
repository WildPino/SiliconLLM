#!/usr/bin/env python3
"""Check a GigaChat organ-precision GGUF against its frozen preflight."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gguf_active_ledger import calculate
from verify_meth05_gguf import sha256, type_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--tensor-types", type=Path, required=True)
    parser.add_argument("--gguf-py", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    plan_schema = plan.get("schema")
    if plan_schema not in ("meth07_gigachat_organ_precision_plan_v1",
                           "meth08_gigachat_q2_reallocation_plan_v1",
                           "meth09_gigachat_q2_experts_only_plan_v1"):
        raise ValueError("unexpected organ-precision plan schema")
    map_sha = sha256(args.tensor_types)
    if map_sha != plan["arm_map_sha256"]:
        raise ValueError("tensor map does not match the frozen plan")
    sys.path.insert(0, str(args.gguf_py))
    from gguf import GGUFReader

    reader = GGUFReader(str(args.model))
    expected = type_map(args.tensor_types)
    actual = {tensor.name: tensor for tensor in reader.tensors}
    wrong = [
        {"tensor": name, "expected": quant, "actual": actual[name].tensor_type.name}
        for name, quant in sorted(expected.items())
        if name in actual and actual[name].tensor_type.name.upper() != quant
    ]
    ledger = calculate(reader, args.model.stat().st_size)
    stored = ledger["stored_tensor_payload_bytes"]
    active = ledger["active_payload_bytes_per_token"]
    header_ok = (len(actual) == 414 and set(actual) == set(expected) and not wrong
                 and stored == plan["arm_stored_tensor_bytes"]
                 and active == plan["arm_active_bytes_per_token"])
    result = {
        "schema": ("meth07_gigachat_organ_precision_verification_v1" if plan_schema.startswith("meth07")
                   else "meth08_gigachat_q2_reallocation_verification_v1" if plan_schema.startswith("meth08")
                   else "meth09_gigachat_q2_experts_only_verification_v1"),
        "arm": plan.get("arm", "q2_experts_only" if plan_schema.startswith("meth09") else "q2_reallocation"),
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "model_bytes": args.model.stat().st_size,
        "plan_sha256": sha256(args.plan),
        "tensor_type_file_sha256": map_sha,
        "tensor_count": len(actual),
        "missing_tensors": sorted(set(expected) - set(actual)),
        "unexpected_tensors": sorted(set(actual) - set(expected)),
        "wrong_types": wrong,
        "planned_stored_tensor_bytes": plan["arm_stored_tensor_bytes"],
        "actual_stored_tensor_bytes": stored,
        "planned_active_bytes_per_token": plan["arm_active_bytes_per_token"],
        "actual_active_bytes_per_token": active,
        "active_limit_bytes_per_token": 540_000_000,
        "passes_diagnostic_header": header_ok,
        "passes_native_active_cost": active <= 540_000_000,
        "organ_ledger": ledger["organs"],
    }
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "organ_ledger"}, indent=2))
    if not header_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
