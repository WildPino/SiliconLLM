"""Bind the independent METH-158 raw/chat pairs for matched training.

This loader does not run training. It must receive the SHA-256 of the
verified 40-shard teacher merge before it can return any draws.
"""

import json
from pathlib import Path

import numpy as np

import meth136_sparse_experts  # inserts the donor-adaptation S1 import path
import meth136_matched_sparse_train as M136


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
MANIFEST = DOC / "meth158_independent_training_manifest.json"
MANIFEST_SHA = "09086fc7c27877e12ae7c122365d67eef5d78d3cda442e5c3a5d36e8d452ed69"
TEACHER = DOC / "meth158_independent_teacher_merged.json"
COUNT = 2560


def load(teacher_sha):
    """Return (source_ids, chat, draws) in the frozen paired update order."""
    assert len(teacher_sha) == 64
    assert M136.digest(MANIFEST) == MANIFEST_SHA
    assert M136.digest(TEACHER) == teacher_sha
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    teacher = json.loads(TEACHER.read_text(encoding="utf-8"))
    assert manifest["experiment"] == "METH-158-independent-tenfold-training-manifest"
    assert teacher["experiment"] == "METH-158-independent-teacher-merged"
    assert teacher["prompt_manifest_sha256"] == MANIFEST_SHA
    assert manifest["source_file_sha256"] == M136.M15.TRAIN_FILE_SHA
    assert manifest["source_ids_sha256"] == M136.M15.TRAIN_IDS_SHA
    assert manifest["source_shape"] == [31250, 512]
    assert manifest["count_per_part"] == COUNT
    assert len(manifest["raw_rows"]) == len(manifest["chat_rows"]) == COUNT
    assert len(teacher["rows"]) == COUNT
    assert len(teacher["shards"]) == 40
    assert M136.digest(M136.M15.TRAIN_PATH) == M136.M15.TRAIN_FILE_SHA
    with np.load(M136.M15.TRAIN_PATH, allow_pickle=False) as archive:
        source_ids = archive["ids"].copy()
    assert source_ids.shape == (31250, 512) and source_ids.dtype == np.int32
    assert M136.M15.sha_bytes(source_ids.tobytes()) == M136.M15.TRAIN_IDS_SHA

    raw_ids = [row["train_row"] for row in manifest["raw_rows"]]
    chat_ids = [row["train_row"] for row in manifest["chat_rows"]]
    assert len(set(raw_ids)) == len(set(chat_ids)) == COUNT
    assert not set(raw_ids).intersection(chat_ids)
    assert all(0 <= row < source_ids.shape[0] for row in raw_ids + chat_ids)

    chat, draws = [], []
    for index, (raw, prompt, response) in enumerate(zip(
            manifest["raw_rows"], manifest["chat_rows"], teacher["rows"])):
        offset = raw["offset"]
        assert 96 <= offset <= 384
        window = source_ids[raw["train_row"], offset:offset + M136.M15.SEQ]
        assert len(window) == M136.M15.SEQ == 128
        assert window.tolist() == raw["window_ids"]
        assert M136.M17.sha(window.tobytes()) == raw["window_ids_sha256"]
        assert response["index"] == index
        assert response["train_row"] == prompt["train_row"]
        assert response["prompt_ids_sha256"] == prompt["prompt_ids_sha256"]
        prompt_ids = np.asarray(prompt["prompt_ids"], dtype=np.int32)
        continuation_ids = np.asarray(response["continuation_ids"], dtype=np.int32)
        assert M136.M17.sha(prompt_ids.tobytes()) == prompt["prompt_ids_sha256"]
        assert M136.M17.sha(continuation_ids.tobytes()) == response["continuation_ids_sha256"]
        assert 1 <= len(continuation_ids) <= 128
        ids = prompt["prompt_ids"] + response["continuation_ids"]
        mask = [0] * (len(prompt_ids) - 1) + [1] * len(continuation_ids)
        assert len(mask) == len(ids) - 1
        chat.append((prompt["train_row"], ids, mask))
        draws.append({"update": index + 1, "raw_row": raw["train_row"],
                      "raw_offset": offset, "chat_index": index,
                      "chat_train_row": prompt["train_row"],
                      "raw_window_ids_sha256": raw["window_ids_sha256"],
                      "chat_prompt_ids_sha256": prompt["prompt_ids_sha256"],
                      "teacher_continuation_ids_sha256": response["continuation_ids_sha256"]})
    assert len(chat) == len(draws) == COUNT
    assert [row["update"] for row in draws] == list(range(1, COUNT + 1))
    return source_ids, chat, draws
