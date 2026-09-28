#!/usr/bin/env python3
"""Merge forty verified donor continuations without changing prompt order."""

import argparse
import json
from pathlib import Path

import meth158_run_teacher_shards as R158


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    assert R158.digest(R158.PROMPTS) == R158.PROMPTS_SHA
    manifest = json.loads(R158.PROMPTS.read_text(encoding="utf-8"))
    prompts = manifest["chat_rows"]
    assert len(prompts) == R158.COUNT
    shards, rows = [], []
    for first in range(0, R158.COUNT, R158.SHARD):
        path = args.shard_dir / f"shard_{first:04d}_{first + R158.SHARD:04d}.json"
        assert path.is_file(), path
        shards.append(R158.verify(path, first, prompts))
        item = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(item["rows"])
    assert len(shards) == 40 and len(rows) == 2560
    assert [row["index"] for row in rows] == list(range(2560))
    assert len({row["train_row"] for row in rows}) == 2560
    lengths = [len(row["continuation_ids"]) for row in rows]
    result = {"experiment": "METH-158-independent-teacher-merged",
              "prompt_manifest_sha256": R158.PROMPTS_SHA,
              "shards": shards,
              "summary": {"rows": len(rows),
                          "min_response_tokens": min(lengths),
                          "max_response_tokens": max(lengths),
                          "mean_response_tokens": sum(lengths) / len(lengths),
                          "total_response_tokens": sum(lengths),
                          "eos_terminated": sum(row["eos_terminated"] for row in rows),
                          "repeated_8gram_3x": sum(row["repeated_8gram_3x"] for row in rows),
                          "new_chat_train_rows": len({row["train_row"] for row in rows})},
              "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sha256": R158.digest(args.out),
                      "summary": result["summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
