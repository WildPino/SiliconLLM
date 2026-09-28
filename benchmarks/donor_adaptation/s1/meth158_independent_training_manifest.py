#!/usr/bin/env python3
"""Freeze 2,560 new chat prompts and disjoint raw windows from pinned H0 IDs."""

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth42_instruct_prompt_manifest as M42
import meth43_instruct_chat_train_manifest as M43
import meth44_instruct_full_chat_smoke as M44
import meth56_product_key_retention as M56
import meth133_q15_external_manifest as M133
import meth136_matched_sparse_train as M136


ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
SEED = "meth158-independent-train-158158"
COUNT = 2560
PRIOR = M133.PRIOR + (
    (DOC / "meth133_q15_external_manifest.json", "ec6f839a2c9b1803dc8d5658aafa8fd9191b59fa2607961850f5a8666bc35674"),
    (DOC / "meth143_fresh_hash_route_manifest.json", "e6c4fb6a92929eb29332c55a0887adb6627c9528f80b21c09f7759139070ccff"),
    (DOC / "meth150_shared_route_manifest.json", "6e911e65529be53dba65fbe382229f0f43e510eb76ab83f4404d4be0368ff5d9"),
    (DOC / "meth153_shared_external_manifest.json", "c9a3dbaa231905b261f72f55ab4e3a89dbcb16ff0138ee0717ebabb1a2c21555"),
    (DOC / "meth156_factorized_external_manifest.json", "a3c370f285dce386e94570ac889c7941d9181a87122a2491e9c47905ff832ca1"))


def rank(part, row):
    return M17.sha(f"{SEED}|{part}|{row}".encode("ascii"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    assert M17.sha(Path(M15.TRAIN_PATH).read_bytes()) == M15.TRAIN_FILE_SHA
    with np.load(M15.TRAIN_PATH, allow_pickle=False) as archive:
        raw_ids = archive["ids"].copy()
    assert raw_ids.shape == (31250, 512) and raw_ids.dtype == np.int32
    assert M17.sha(raw_ids.tobytes()) == M15.TRAIN_IDS_SHA
    assert M17.sha(M44.TRAIN_CHAT_PATH.read_bytes()) == M44.TRAIN_CHAT_SHA
    old_chat = json.loads(M44.TRAIN_CHAT_PATH.read_text(encoding="utf-8"))
    assert len(old_chat["rows"]) == 256
    assert M17.sha((DOC / "meth155_matched_factorized_train_result.json").read_bytes()) == (
        "759d9044c7187c9bfd8ed3fbd6cc8b6d6f3be663c81bf9549b62200668dd4e2a")
    old_train = json.loads((DOC / "meth155_matched_factorized_train_result.json").read_text(encoding="utf-8"))
    excluded = {row["train_row"] for row in old_chat["rows"]}
    excluded.update(draw["raw_row"] for draw in old_train["draws"])
    for path, digest in ((M56.DEV_PATH, M56.DEV_SHA),
                         (M56.OLD_DEV_PATH, M56.OLD_DEV_SHA),
                         (M44.PROMPT_PATH, M44.PROMPT_SHA),
                         (M56.M47_DEV_PATH, M56.M47_DEV_SHA)):
        assert M17.sha(path.read_bytes()) == digest
        excluded.update(row["train_row"] for row in json.loads(
            path.read_text(encoding="utf-8"))["rows"])
    prior_bytes = []
    for path, digest in PRIOR:
        assert M17.sha(path.read_bytes()) == digest, path
        manifest = json.loads(path.read_text(encoding="utf-8"))
        prior_bytes.extend(item["text"].encode("utf-8") for item in manifest["items"])
    tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                               local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    chat, raw = [], []
    rejected = {"excluded_row": 0, "short_excerpt": 0,
                "overlap": 0, "long_prompt": 0}
    for part in ("chat", "raw"):
        selected = chat if part == "chat" else raw
        for row_index in sorted(range(raw_ids.shape[0]),
                                key=lambda row: rank(part, row)):
            if row_index in excluded:
                rejected["excluded_row"] += 1
                continue
            if part == "chat":
                source = raw_ids[row_index, :96]
                excerpt = tokenizer.decode(source, skip_special_tokens=True)
                if len(excerpt.strip()) < 20:
                    rejected["short_excerpt"] += 1
                    continue
                if M17.fragment_overlap(excerpt.encode("utf-8"), prior_bytes):
                    rejected["overlap"] += 1
                    continue
                message = M43.TEMPLATE.format(excerpt=excerpt)
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": message}],
                    tokenize=True, add_generation_prompt=True)["input_ids"]
                if not 1 <= len(ids) <= 256:
                    rejected["long_prompt"] += 1
                    continue
                selected.append({"train_row": row_index,
                    "source_prefix_ids_sha256": M17.sha(source.tobytes()),
                    "excerpt_sha256": M17.sha(excerpt.encode("utf-8")),
                    "prompt_ids_sha256": M17.sha(np.asarray(ids, dtype=np.int32).tobytes()),
                    "prompt_ids": ids})
            else:
                offset = 96 + int(rank("offset", row_index)[:8], 16) % 289
                source = raw_ids[row_index, offset:offset + M15.SEQ]
                assert source.size == M15.SEQ
                excerpt = tokenizer.decode(source, skip_special_tokens=True)
                if len(excerpt.strip()) < 20:
                    rejected["short_excerpt"] += 1
                    continue
                if M17.fragment_overlap(excerpt.encode("utf-8"), prior_bytes):
                    rejected["overlap"] += 1
                    continue
                selected.append({"train_row": row_index, "offset": offset,
                    "window_ids_sha256": M17.sha(source.tobytes()),
                    "window_ids": source.tolist()})
            excluded.add(row_index)
            if len(selected) == COUNT:
                break
        if len(selected) != COUNT:
            raise RuntimeError(f"insufficient new {part} rows: {len(selected)}")
    assert len({r["train_row"] for r in chat + raw}) == 2 * COUNT
    result = {"experiment": "METH-158-independent-tenfold-training-manifest",
              "seed": SEED, "source_file_sha256": M15.TRAIN_FILE_SHA,
              "source_ids_sha256": M15.TRAIN_IDS_SHA,
              "source_shape": list(raw_ids.shape),
              "prior_manifest_sha256": {path.name: digest for path, digest in PRIOR},
              "prior_training_chat_sha256": M44.TRAIN_CHAT_SHA,
              "prior_meth155_result_sha256": "759d9044c7187c9bfd8ed3fbd6cc8b6d6f3be663c81bf9549b62200668dd4e2a",
              "model": M42.MODEL, "revision": M42.REV,
              "tokenizer_fingerprint": M15.M13.TOK_FP,
              "template": M43.TEMPLATE, "source_prefix_tokens": 96,
              "count_per_part": COUNT, "rejected": rejected,
              "chat_rows": chat, "raw_rows": raw}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"sha256": M17.sha(args.out.read_bytes()),
                      "chat": len(chat), "raw": len(raw),
                      "rejected": rejected,
                      "max_prompt_tokens": max(len(x["prompt_ids"]) for x in chat)},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
