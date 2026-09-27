#!/usr/bin/env python3
"""Merge four complete, hash-bound METH-106 response shards."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
PROMPTS = DIR / "meth43_instruct_chat_train_manifest.json"
PROMPTS_SHA = "17921cb83c1d10793e2a56df6aa899039bd794757eeff43921068b8de2096055"
PARENT_SHA = "8371262a1461a44fcedf12d129059d8b8ab1fd9860e032df8661a30eff187072"
MODEL_SHA = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs=4, type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert sha(PROMPTS) == PROMPTS_SHA
    prompt_rows = json.loads(PROMPTS.read_text(encoding="utf-8"))["rows"]
    assert len(prompt_rows) == 256
    shards = []
    rows = []
    for expected_start, path in zip(range(0, 256, 64), args.shards):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["experiment"] == "METH-106-long-E128-teacher-chat"
        assert (data["shard_start"], data["shard_stop"]) == (expected_start, expected_start + 64)
        assert data["prompt_manifest_sha256"] == PROMPTS_SHA
        assert data["parent_checkpoint_sha256"] == PARENT_SHA
        assert data["model_sha256"] == MODEL_SHA
        assert data["max_new_tokens"] == 128 and len(data["rows"]) == 64
        for prompt, row in zip(prompt_rows[expected_start:expected_start + 64], data["rows"]):
            assert row["train_row"] == prompt["train_row"]
            assert row["prompt_ids_sha256"] == prompt["prompt_ids_sha256"]
            continuation = row["continuation_ids"]
            assert 1 <= len(continuation) <= 128
            assert row["window_ids"] == (prompt["prompt_ids"] + continuation)[-129:]
            assert len(row["assistant_target_mask"]) == 128
            assert sum(row["assistant_target_mask"]) == len(continuation)
            digest = hashlib.sha256(np.asarray(row["window_ids"],
                                            dtype=np.int32).tobytes()).hexdigest()
            assert row["window_ids_sha256"] == digest
        rows.extend(data["rows"])
        shards.append({"path": str(path.resolve()), "sha256": sha(path),
                       "runtime": data["runtime"]})
    assert len(rows) == 256 and len({r["train_row"] for r in rows}) == 256
    lengths = [len(r["continuation_ids"]) for r in rows]
    output = {"experiment": "METH-106-long-E128-teacher-chat-merged",
              "prompt_manifest_sha256": PROMPTS_SHA,
              "parent_checkpoint_sha256": PARENT_SHA,
              "model_sha256": MODEL_SHA, "max_new_tokens": 128,
              "shards": shards,
              "summary": {"rows": 256, "min_response_tokens": min(lengths),
                          "max_response_tokens": max(lengths),
                          "mean_response_tokens": sum(lengths) / 256,
                          "responses_over_64_tokens": sum(n > 64 for n in lengths),
                          "eos_terminated": sum(r["eos_terminated"] for r in rows),
                          "repeated_8gram_3x": sum(r["repeated_8gram_3x"] for r in rows)},
              "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sha256": sha(args.out), "summary": output["summary"]}, indent=2))


if __name__ == "__main__":
    main()
