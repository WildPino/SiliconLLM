"""Validate all frozen METH-54 CPU arms and compute the decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


ARMS = ((8, 16, 256), (128, 214, 64), (512, 534, 32))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    parsed: dict[str, dict] = {}
    for a, b, tokens in ARMS:
        rows = []
        for threads in (1, 6):
            path = args.raw_dir / f"meth54_a{a}_b{b}_t{threads}.json"
            row = json.loads(path.read_text(encoding="utf-8-sig"))
            assert row["method"] == "METH54"
            assert (row["a"], row["b"], row["experts"]) == (a, b, a * b)
            assert (row["threads"], row["tokens"], row["layers"], row["dim"], row["rank"]) == (
                threads, tokens, 24, 896, 64
            )
            assert row["projection_bytes"] == 24 * 64 * 896 * 4
            assert row["key_pool_bytes"] == 24 * (a + b) * 64 * 4
            assert row["addressed_bytes_per_token"] == row["projection_bytes"] + row["key_pool_bytes"]
            assert row["oracle_cases"] == (384 if a == 8 else 24)
            assert row["dot_max_abs_error"] < 1e-4
            assert row["rss_bytes"] < 1024**3
            assert len(row["total_ms_reps"]) == 4
            assert abs(row["total_ms_median"] - statistics.median(row["total_ms_reps"])) < 1e-8
            row["raw_file"] = path.name
            row["raw_sha256"] = sha256(path)
            rows.append(row)
        assert rows[0]["checksum"] == rows[1]["checksum"]
        parsed[str(a * b)] = {str(row["threads"]): row for row in rows}

    six_ratio = parsed["273408"]["6"]["total_ms_median"] / parsed["27392"]["6"]["total_ms_median"]
    fastest_large = min(parsed["273408"][str(t)]["total_ms_median"] for t in (1, 6))
    gates = {
        "large_router_at_most_3_ms": fastest_large <= 3.0,
        "six_thread_10x_ratio_at_most_2": six_ratio <= 2.0,
    }
    gates["retain_r64_product_key_cpu_candidate"] = all(gates.values())
    result = {
        "method": "METH54",
        "source_sha256": sha256(args.source),
        "binary_sha256": sha256(args.binary),
        "arms": parsed,
        "six_thread_10x_ratio": six_ratio,
        "fastest_large_ms_per_token": fastest_large,
        "gates": gates,
        "scope": "synthetic CPU router; no trained keys or experts, donor quality, or engine.c parity",
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"six_ratio": six_ratio, "fastest_large_ms": fastest_large, "gates": gates}))


if __name__ == "__main__":
    main()
