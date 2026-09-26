"""Verify the actual GigaChat low-bit GGUF against the frozen METH-04 map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from gguf_active_ledger import calculate


PLANNED_STORED = 3_202_277_120
PLANNED_ACTIVE = 534_024_800
ACTIVE_LIMIT = 540_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def type_map(path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        pattern, kind = line.split("=", 1)
        if not pattern.startswith("^") or not pattern.endswith("$"):
            raise ValueError(f"tensor override not anchored: {pattern}")
        name = pattern[1:-1].replace(r"\.", ".")
        if "\\" in name or name in expected:
            raise ValueError(f"invalid or repeated tensor override: {pattern}")
        expected[name] = kind.upper()
    if len(expected) != 414:
        raise ValueError(f"expected 414 override entries, got {len(expected)}")
    return expected


def verify(model: Path, tensor_types: Path, gguf_py: Path) -> dict:
    sys.path.insert(0, str(gguf_py))
    from gguf import GGUFReader

    reader = GGUFReader(str(model))
    expected = type_map(tensor_types)
    actual = {tensor.name: tensor for tensor in reader.tensors}
    wrong_types = [
        {"tensor": name, "expected": kind, "actual": actual[name].tensor_type.name}
        for name, kind in sorted(expected.items())
        if name in actual and actual[name].tensor_type.name.upper() != kind
    ]
    ledger = calculate(reader, model.stat().st_size)
    stored = ledger["stored_tensor_payload_bytes"]
    active = ledger["active_payload_bytes_per_token"]
    return {
        "schema": "meth05_gigachat_converted_gguf_verification_v1",
        "model": str(model),
        "model_sha256": sha256(model),
        "tensor_type_file_sha256": sha256(tensor_types),
        "model_bytes": model.stat().st_size,
        "tensor_count": len(actual),
        "missing_tensors": sorted(set(expected) - set(actual)),
        "unexpected_tensors": sorted(set(actual) - set(expected)),
        "wrong_types": wrong_types,
        "planned_stored_tensor_bytes": PLANNED_STORED,
        "actual_stored_tensor_bytes": stored,
        "planned_active_bytes_per_token": PLANNED_ACTIVE,
        "actual_active_bytes_per_token": active,
        "active_limit_bytes_per_token": ACTIVE_LIMIT,
        "organ_ledger": ledger["organs"],
        "passes_header_gate": len(actual) == 414 and not wrong_types
        and set(actual) == set(expected) and stored == PLANNED_STORED
        and active == PLANNED_ACTIVE and active <= ACTIVE_LIMIT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tensor-types", type=Path, required=True)
    parser.add_argument("--gguf-py", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = verify(args.model, args.tensor_types, args.gguf_py)
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "organ_ledger"}, indent=2))
    if not report["passes_header_gate"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
