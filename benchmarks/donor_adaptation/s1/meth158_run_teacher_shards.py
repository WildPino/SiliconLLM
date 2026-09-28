#!/usr/bin/env python3
"""Resume and verify the forty immutable METH-158 donor-response shards."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
PROMPTS = DOC / "meth158_independent_training_manifest.json"
PROMPTS_SHA = "09086fc7c27877e12ae7c122365d67eef5d78d3cda442e5c3a5d36e8d452ed69"
SHARD_SCRIPT = Path(__file__).with_name("meth158_independent_teacher_shard.py")
COUNT = 2560
SHARD = 64


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path, start, prompts):
    item = json.loads(path.read_text(encoding="utf-8"))
    assert item["experiment"] == "METH-158-independent-teacher-shard"
    assert item["prompt_manifest_sha256"] == PROMPTS_SHA
    assert item["shard_start"] == start and item["shard_stop"] == start + SHARD
    assert len(item["rows"]) == SHARD
    assert item["summary"]["rows"] == SHARD
    for index, row in enumerate(item["rows"], start):
        assert row["index"] == index
        assert row["train_row"] == prompts[index]["train_row"]
        assert row["prompt_ids_sha256"] == prompts[index]["prompt_ids_sha256"]
        assert 1 <= len(row["continuation_ids"]) <= 128
        import numpy as np
        bits = np.asarray(row["continuation_ids"], dtype=np.int32).tobytes()
        assert hashlib.sha256(bits).hexdigest() == row["continuation_ids_sha256"]
    return {"start": start, "stop": start + SHARD,
            "path": str(path.resolve()), "sha256": digest(path),
            "response_tokens": sum(len(r["continuation_ids"]) for r in item["rows"]),
            "eos_terminated": item["summary"]["eos_terminated"],
            "repeated_8gram_3x": item["summary"]["repeated_8gram_3x"],
            "generation_seconds": item["runtime"]["seconds"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--progress", type=Path, required=True)
    ap.add_argument("--max-new-shards", type=int, default=40)
    args = ap.parse_args()
    assert 1 <= args.max_new_shards <= 40
    assert digest(PROMPTS) == PROMPTS_SHA
    manifest = json.loads(PROMPTS.read_text(encoding="utf-8"))
    prompts = manifest["chat_rows"]
    assert len(prompts) == COUNT
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed = []
    new_shards = 0
    for first in range(0, COUNT, SHARD):
        path = args.out_dir / f"shard_{first:04d}_{first + SHARD:04d}.json"
        if not path.exists():
            if new_shards >= args.max_new_shards:
                break
            subprocess.run([sys.executable, str(SHARD_SCRIPT),
                            "--start", str(first), "--stop", str(first + SHARD),
                            "--out", str(path)], cwd=ROOT, check=True)
            new_shards += 1
        row = verify(path, first, prompts)
        completed.append(row)
        report = {"experiment": "METH-158-teacher-shard-acquisition-progress",
                  "prompt_manifest_sha256": PROMPTS_SHA,
                  "completed_shards": len(completed), "total_shards": COUNT // SHARD,
                  "completed_rows": len(completed) * SHARD,
                  "response_tokens": sum(x["response_tokens"] for x in completed),
                  "eos_terminated": sum(x["eos_terminated"] for x in completed),
                  "repeated_8gram_3x": sum(x["repeated_8gram_3x"] for x in completed),
                  "shards": completed,
                  "runner_elapsed_seconds": time.monotonic() - started}
        temporary = args.progress.with_suffix(args.progress.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.progress)
        print(json.dumps({"completed_shards": len(completed),
                          "completed_rows": report["completed_rows"],
                          "new_shards_this_run": new_shards,
                          "response_tokens": report["response_tokens"]}), flush=True)
    assert completed
    print(json.dumps({"complete": len(completed) == COUNT // SHARD,
                      "completed_shards": len(completed),
                      "progress_sha256": digest(args.progress)}), flush=True)


if __name__ == "__main__":
    main()
