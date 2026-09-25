#!/usr/bin/env python3
"""Score the frozen NES-01 C-engine greedy continuations and retain samples."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "phase55"))
from phase55_ssm import load_meta  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_generations(path: Path, vocab: int) -> list[tuple[int, list[int]]]:
    with path.open("rb") as stream:
        header = stream.read(16)
        if len(header) != 16:
            raise ValueError(f"short generation header: {path}")
        magic, count, prefix_len, gen_len = struct.unpack("<4I", header)
        if (magic, count, prefix_len, gen_len) != (0x4E473031, 16, 128, 128):
            raise ValueError(f"unexpected NES-01 generation shape: {path}")
        rows = []
        for index in range(count):
            raw = stream.read(4 + 2 * gen_len)
            if len(raw) != 4 + 2 * gen_len:
                raise ValueError(f"short generation row: {path}")
            offset = struct.unpack_from("<I", raw)[0]
            tokens = np.frombuffer(raw, dtype="<u2", offset=4).astype(int).tolist()
            if offset != 512 + 2048 * index or max(tokens) >= vocab:
                raise ValueError(f"bad offset or token ID in row {index}: {path}")
            rows.append((offset, tokens))
        if stream.read(1):
            raise ValueError(f"trailing bytes: {path}")
    return rows


def max_ngram_count(tokens: list[int], n: int = 8) -> int:
    return max(Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)).values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e32", type=Path, required=True)
    parser.add_argument("--e128", type=Path, required=True)
    parser.add_argument("--mode", choices=("lut-fast", "fp32-exact"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")

    ids_path = ROOT / "results" / "phase55" / "ids.u16"
    meta_path = ROOT / "results" / "phase55" / "meta.bin"
    vocab, _, id_to_bytes = load_meta(str(meta_path))
    ids = np.fromfile(ids_path, dtype="<u2")
    val = ids[int(len(ids) * 0.9):]
    arms = {}
    for name, path in (("e32", args.e32), ("e128", args.e128)):
        rows = read_generations(path, vocab)
        samples = []
        for offset, tokens in rows:
            prefix_ids = val[offset:offset + 128].astype(int).tolist()
            if len(prefix_ids) != 128:
                raise ValueError("validation prefix is incomplete")
            maximum = max_ngram_count(tokens)
            samples.append({
                "validation_offset": offset,
                "prefix_text": b"".join(id_to_bytes[token] for token in prefix_ids).decode("utf-8", "replace"),
                "continuation_ids": tokens,
                "continuation_text": b"".join(id_to_bytes[token] for token in tokens).decode("utf-8", "replace"),
                "max_repeated_token_8gram_count": maximum,
                "has_token_8gram_repeated_at_least_three_times": maximum >= 3,
            })
        arms[name] = {
            "generation_file_sha256": sha256(path),
            "triple_8gram_continuations": sum(row["has_token_8gram_repeated_at_least_three_times"] for row in samples),
            "samples": samples,
        }
    result = {
        "schema": "nes01_greedy_generation_v1",
        "scope": f"C engine {args.mode} outputs; token 8-gram counted with overlapping occurrences in each 128-token continuation",
        "validation_ids_sha256": sha256(ids_path),
        "tokenizer_meta_sha256": sha256(meta_path),
        "prefix_positions": "last 10% of ids.u16; offsets 512+2048*i, i=0..15; 128-token prefixes",
        "generation": "128 greedy tokens per prefix; context state reset for each prefix",
        "arms": arms,
        "gate": {"e128_at_most_e32_plus_one": arms["e128"]["triple_8gram_continuations"]
                 <= arms["e32"]["triple_8gram_continuations"] + 1},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"E32={arms['e32']['triple_8gram_continuations']}/16, "
          f"E128={arms['e128']['triple_8gram_continuations']}/16, "
          f"gate={'PASS' if result['gate']['e128_at_most_e32_plus_one'] else 'FAIL'}")


if __name__ == "__main__":
    main()
