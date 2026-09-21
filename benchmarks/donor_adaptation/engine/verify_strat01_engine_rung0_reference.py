"""Compare the C rung-0 inventory with the pinned GGUF reader and BF16 binding."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT = (
    ROOT / "benchmarks" / "donor_adaptation" / "engine" / "results"
    / "strat01_gigachat_engine_rung0_20260921"
)
DEFAULT_MODEL = (
    ROOT / "benchmarks" / "donor_adaptation" / "density" / "results"
    / "strat01_gigachat_q4_97045b2" / "GigaChat3.1-10B-A1.8B-q4_K_M.gguf"
)
DEFAULT_BINDING = (
    ROOT / "benchmarks" / "donor_adaptation" / "density" / "results"
    / "strat01_gigachat_source_binding_v1" / "source_binding_report.json"
)
DEFAULT_GGUF_PY = (
    Path.home() / "AppData" / "Local" / "Temp"
    / "siliconllm-llama-bind-5b335f4" / "gguf-py"
)


def digest_descriptors(items: list[dict[str, object]]) -> str:
    canonical = json.dumps(items, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_RESULT / "inventory.json")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source-binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--gguf-py", type=Path, default=DEFAULT_GGUF_PY)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT / "reference_compare.json")
    args = parser.parse_args()

    sys.path.insert(0, str(args.gguf_py.resolve()))
    from gguf import GGUFReader  # type: ignore[import-not-found]  # pinned local checkout

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    c_items = [
        {
            "name": item["name"],
            "dims": item["dims"],
            "type": item["type"],
            "offset": item["offset"],
            "byte_span": item["byte_span"],
            "file_offset": item["file_offset"],
        }
        for item in inventory["tensors"]
    ]

    reader = GGUFReader(args.model.resolve(), "r")
    reference_items = [
        {
            "name": tensor.name,
            "dims": [int(value) for value in tensor.shape],
            "type": tensor.tensor_type.name,
            "offset": int(tensor.data_offset) - int(reader.data_offset),
            "byte_span": int(tensor.n_bytes),
            "file_offset": int(tensor.data_offset),
        }
        for tensor in reader.tensors
    ]
    descriptor_mismatches = [
        {"index": index, "c": c_item, "reference": reference_item}
        for index, (c_item, reference_item) in enumerate(zip(c_items, reference_items, strict=False))
        if c_item != reference_item
    ]
    if len(c_items) != len(reference_items):
        descriptor_mismatches.append(
            {"count_mismatch": {"c": len(c_items), "reference": len(reference_items)}}
        )

    binding = json.loads(args.source_binding.read_text(encoding="utf-8"))
    bound_shapes = {
        tensor["name"]: [int(value) for value in tensor["shape"]]
        for tensor in binding["payload"]["tensors"]
    }
    c_shapes = {item["name"]: item["dims"] for item in c_items}
    source_shape_mismatches = [
        {"name": name, "c": c_shapes.get(name), "bound_bf16": shape}
        for name, shape in bound_shapes.items()
        if c_shapes.get(name) != shape
    ]
    source_shape_mismatches.extend(
        {"name": name, "c": shape, "bound_bf16": None}
        for name, shape in c_shapes.items()
        if name not in bound_shapes
    )

    errors: list[str] = []
    if int(reader.data_offset) != inventory["gguf"]["data_offset"]:
        errors.append("data_offset")
    if descriptor_mismatches:
        errors.append("descriptor_mismatch")
    if source_shape_mismatches:
        errors.append("source_binding_name_shape_mismatch")
    result = {
        "schema": "strat01_gigachat_engine_rung0_reference_compare_v1",
        "status": "PASS_REFERENCE_COMPARE" if not errors else "FAIL_REFERENCE_COMPARE",
        "gguf_reader_path": str(args.gguf_py.resolve()),
        "gguf_reader_binding_commit": "5b335f413e4f73b0809c4fe39af894efbcc6a0d2",
        "inventory_path": str(args.inventory.resolve()),
        "model_path": str(args.model.resolve()),
        "source_binding_path": str(args.source_binding.resolve()),
        "data_offset": {"c": inventory["gguf"]["data_offset"], "reference": int(reader.data_offset)},
        "descriptor_count": {"c": len(c_items), "reference": len(reference_items), "bound_bf16": len(bound_shapes)},
        "descriptor_sha256": {"c": digest_descriptors(c_items), "reference": digest_descriptors(reference_items)},
        "descriptor_mismatches": descriptor_mismatches,
        "source_binding_name_shape_mismatches": source_shape_mismatches,
        "errors": errors,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": errors}, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
