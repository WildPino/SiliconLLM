#!/usr/bin/env python3
"""Bind and permute 3,840 source-distinct paired training updates."""

import argparse
import json
from pathlib import Path

import numpy as np

import meth136_sparse_experts  # inserts S1 import path
import meth136_matched_sparse_train as M136
import meth158_training_inputs as M158


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
OLD_MANIFEST_SHA = "09086fc7c27877e12ae7c122365d67eef5d78d3cda442e5c3a5d36e8d452ed69"
OLD_TEACHER_SHA = "cdcb22a4273148dede97a17eac81345c4fc5f8e40f09cb18ecac4e35115e437c"
NEW_MANIFEST = DOC / "meth165_expanded_training_manifest.json"
NEW_MANIFEST_SHA = "2875ac50a4de005b56ad6403c6bd30e358b87eb21d7dbb1918738957855068ba"
NEW_TEACHER = DOC / "meth165_expanded_teacher_merged.json"
NEW_TEACHER_SHA = "9843b8d98d0bbe322cbbf2567f72d5c3b09a34bfa774b90785c22a9079076ed9"
ROUTE = DOC / "meth173_fresh_recurrence_route_result.json"
ROUTE_SHA = "8cf875ade5a2996e7f2e9db352f5dde1afb65efe537e0038a0e61033c97770e3"
CPU = DOC / "meth174_native_low_recurrence_cost_result.json"
CPU_SHA = "55db1199b54a84463efc15f62c38d6e3896d08312bb31d872ef53d94f6c33e3c"
TABLE = DOC / "meth172_threshold8_shared_table.json"
TABLE_SHA = "e89ff2cecbf887b85b5e361d47a8b918d8ccb93bd5c385a580ba993b93ac72db"
SEED = 175175
UPDATES = 3840


def inputs():
    for path, sha in ((M158.MANIFEST, OLD_MANIFEST_SHA),
                      (M158.TEACHER, OLD_TEACHER_SHA),
                      (NEW_MANIFEST, NEW_MANIFEST_SHA),
                      (NEW_TEACHER, NEW_TEACHER_SHA),
                      (ROUTE, ROUTE_SHA), (CPU, CPU_SHA), (TABLE, TABLE_SHA)):
        assert M136.digest(path) == sha, path
    assert json.loads(ROUTE.read_text(encoding="utf-8"))["decision"] == (
        "prospective_low_recurrence_route_pass_native_cpu_pending")
    assert json.loads(CPU.read_text(encoding="utf-8"))["decision"] == (
        "native_low_recurrence_route_cost_pass_quality_and_full_rate_open")
    raw_ids, chat, old_draws = M158.load(OLD_TEACHER_SHA)
    assert len(chat) == len(old_draws) == 2560
    manifest = json.loads(NEW_MANIFEST.read_text(encoding="utf-8"))
    teacher = json.loads(NEW_TEACHER.read_text(encoding="utf-8"))
    assert manifest["experiment"] == "METH-165-expanded-independent-training-manifest"
    assert teacher["experiment"] == "METH-165-expanded-teacher-merged"
    assert teacher["manifest_sha256"] == NEW_MANIFEST_SHA
    assert len(manifest["raw_rows"]) == len(manifest["chat_rows"]) == 1280
    assert len(teacher["rows"]) == 1280
    old_raw = {r["raw_row"] for r in old_draws}
    old_chat = {r["chat_train_row"] for r in old_draws}
    new_raw = {r["train_row"] for r in manifest["raw_rows"]}
    new_chat = {r["train_row"] for r in manifest["chat_rows"]}
    assert len(old_raw) == len(old_chat) == 2560
    assert len(new_raw) == len(new_chat) == 1280
    assert not (old_raw & old_chat or old_raw & new_raw or old_raw & new_chat or
                old_chat & new_raw or old_chat & new_chat or new_raw & new_chat)
    pairs = [dict(row, origin="meth158", origin_index=i)
             for i, row in enumerate(old_draws)]
    for i, (raw, prompt, response) in enumerate(zip(
            manifest["raw_rows"], manifest["chat_rows"], teacher["rows"])):
        assert response["index"] == i and response["train_row"] == prompt["train_row"]
        assert response["prompt_ids_sha256"] == prompt["prompt_ids_sha256"]
        offset = raw["offset"]
        assert 96 <= offset <= 384
        window = raw_ids[raw["train_row"], offset:offset + 128]
        assert len(window) == 128 and window.tolist() == raw["window_ids"]
        assert M136.M17.sha(window.tobytes()) == raw["window_ids_sha256"]
        prompt_ids = np.asarray(prompt["prompt_ids"], dtype=np.int32)
        continuation = np.asarray(response["continuation_ids"], dtype=np.int32)
        assert M136.M17.sha(prompt_ids.tobytes()) == prompt["prompt_ids_sha256"]
        assert M136.M17.sha(continuation.tobytes()) == response["continuation_ids_sha256"]
        assert 1 <= len(continuation) <= 128
        ids = prompt["prompt_ids"] + response["continuation_ids"]
        mask = [0] * (len(prompt_ids) - 1) + [1] * len(continuation)
        assert len(mask) == len(ids) - 1
        chat.append((prompt["train_row"], ids, mask))
        pairs.append({"origin": "meth165", "origin_index": i,
                      "raw_row": raw["train_row"], "raw_offset": offset,
                      "chat_index": 2560 + i,
                      "chat_train_row": prompt["train_row"],
                      "raw_window_ids_sha256": raw["window_ids_sha256"],
                      "chat_prompt_ids_sha256": prompt["prompt_ids_sha256"],
                      "teacher_continuation_ids_sha256":
                      response["continuation_ids_sha256"]})
    assert len(pairs) == len(chat) == UPDATES
    order = np.random.default_rng(SEED).permutation(UPDATES).tolist()
    draws = []
    for update, index in enumerate(order, start=1):
        row = dict(pairs[index])
        row["update"] = update
        draws.append(row)
    assert [d["update"] for d in draws] == list(range(1, UPDATES + 1))
    assert len({(d["origin"], d["origin_index"]) for d in draws}) == UPDATES
    input_tokens = sum(127 + len(chat[d["chat_index"]][1]) - 1 for d in draws)
    assert input_tokens == 1330096
    return raw_ids, chat, draws, input_tokens


def load(draws_sha):
    """Regenerate inputs and verify the frozen draw manifest by SHA."""
    path = DOC / "meth175_training_draws.json"
    assert M136.digest(path) == draws_sha
    record = json.loads(path.read_text(encoding="utf-8"))
    raw_ids, chat, draws, input_tokens = inputs()
    assert record["seed"] == SEED and record["updates"] == UPDATES
    assert record["input_tokens"] == input_tokens
    assert record["draws"] == draws
    return raw_ids, chat, draws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    _, _, draws, input_tokens = inputs()
    record = {"experiment": "METH-175-matched-long-training-draws",
              "old_manifest_sha256": OLD_MANIFEST_SHA,
              "old_teacher_sha256": OLD_TEACHER_SHA,
              "expanded_manifest_sha256": NEW_MANIFEST_SHA,
              "expanded_teacher_sha256": NEW_TEACHER_SHA,
              "route_result_sha256": ROUTE_SHA,
              "cpu_result_sha256": CPU_SHA,
              "table_sha256": TABLE_SHA,
              "seed": SEED, "updates": UPDATES,
              "input_tokens": input_tokens,
              "draws": draws}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sha256": M136.digest(args.out),
                      "updates": UPDATES, "input_tokens": input_tokens,
                      "old_pairs": 2560, "expanded_pairs": 1280}, indent=2))


if __name__ == "__main__":
    main()
