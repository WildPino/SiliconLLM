"""Audit low-bit GigaChat imatrix tensor and routed-expert coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def required_names(type_file: Path) -> set[str]:
    names: set[str] = set()
    for line in type_file.read_text(encoding="utf-8").splitlines():
        pattern, kind = line.split("=", 1)
        if kind.lower() != "iq2_xs":
            continue
        if not pattern.startswith("^") or not pattern.endswith("$"):
            raise ValueError(f"non-anchored tensor pattern: {pattern}")
        name = pattern[1:-1].replace(r"\.", ".")
        if "\\" in name or name in names:
            raise ValueError(f"invalid or duplicate tensor pattern: {pattern}")
        names.add(name)
    if len(names) != 254:
        raise ValueError(f"expected 254 IQ2_XS tensors, got {len(names)}")
    return names


def audit(imatrix_path: Path, type_file: Path, gguf_py: Path, minimum_exposure: int = 1) -> dict:
    sys.path.insert(0, str(gguf_py))
    import gguf
    import numpy as np

    expected = required_names(type_file)
    reader = gguf.GGUFReader(str(imatrix_path))
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    missing: list[str] = []
    nonfinite: list[str] = []
    partial: list[dict] = []
    underexposed: list[dict] = []
    min_expert_count: int | None = None
    expert_instances = 0
    for name in sorted(expected):
        value = tensors.get(f"{name}.in_sum2")
        counts = tensors.get(f"{name}.counts")
        if value is None or counts is None:
            missing.append(name)
            continue
        if not np.isfinite(value.data).all() or not np.isfinite(counts.data).all():
            nonfinite.append(name)
            continue
        flat = counts.data.reshape(-1)
        if flat.size == 64:
            expert_instances += flat.size
            smallest = int(flat.min())
            min_expert_count = smallest if min_expert_count is None else min(min_expert_count, smallest)
            zeros = int(np.count_nonzero(flat == 0))
            if zeros:
                partial.append({"tensor": name, "zero_experts": zeros, "minimum_count": smallest})
            for expert_id, count in enumerate(flat):
                if count < minimum_exposure:
                    underexposed.append({"tensor": name, "expert_id": expert_id, "count": int(count)})
        elif flat.size != 1 or flat[0] <= 0:
            partial.append({"tensor": name, "count_entries": int(flat.size), "minimum_count": float(flat.min())})

    report = {
        "schema": "meth05_gigachat_imatrix_audit_v1",
        "imatrix": str(imatrix_path),
        "imatrix_sha256": sha256(imatrix_path),
        "imatrix_bytes": imatrix_path.stat().st_size,
        "tensor_type_file_sha256": sha256(type_file),
        "chunk_count": int(reader.fields["imatrix.chunk_count"].parts[-1][0]),
        "chunk_size": int(reader.fields["imatrix.chunk_size"].parts[-1][0]),
        "required_tensors": len(expected),
        "present_required_tensors": len(expected) - len(missing),
        "missing_tensors": missing,
        "nonfinite_tensors": nonfinite,
        "routed_expert_instances": expert_instances,
        "minimum_routed_expert_count": min_expert_count,
        "required_minimum_routed_expert_count": minimum_exposure,
        "partial_expert_tensors": partial,
        "underexposed_expert_slices": underexposed,
        "passes_coverage_gate": not (missing or nonfinite or partial)
        and not underexposed and min_expert_count is not None and min_expert_count >= minimum_exposure,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imatrix", type=Path, required=True)
    parser.add_argument("--tensor-types", type=Path, required=True)
    parser.add_argument("--gguf-py", type=Path, required=True)
    parser.add_argument("--minimum-exposure", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.minimum_exposure < 1:
        parser.error("--minimum-exposure must be positive")
    report = audit(args.imatrix, args.tensor_types, args.gguf_py, args.minimum_exposure)
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("partial_expert_tensors", "underexposed_expert_slices")}, indent=2))


if __name__ == "__main__":
    main()
