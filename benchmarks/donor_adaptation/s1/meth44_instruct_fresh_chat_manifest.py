#!/usr/bin/env python3
"""Freeze 24 unused training-corpus excerpts for a new chat retention screen."""
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth42_instruct_prompt_manifest as M42
import meth43_instruct_chat_train_manifest as M43


ROOT = Path(__file__).resolve().parents[3]
TRAIN_MANIFEST = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth43_instruct_chat_train_manifest.json"
TRAIN_MANIFEST_SHA = "17921cb83c1d10793e2a56df6aa899039bd794757eeff43921068b8de2096055"
SEED = 444444
COUNT = 24


def build():
    assert M17.sha(TRAIN_MANIFEST.read_bytes()) == TRAIN_MANIFEST_SHA
    selected = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))
    excluded = {row["train_row"] for row in selected["rows"]}
    assert len(excluded) == 256
    assert M15.M13.sha256(M15.TRAIN_PATH) == M15.TRAIN_FILE_SHA
    with np.load(M15.TRAIN_PATH, allow_pickle=False) as archive:
        train_ids = archive["ids"].copy()
    assert train_ids.shape == (31250, 512) and train_ids.dtype == np.int32
    assert M17.sha(train_ids.tobytes()) == M15.TRAIN_IDS_SHA
    tok = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                        local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tok) == M15.M13.TOK_FP
    remaining = np.asarray([i for i in range(len(train_ids)) if i not in excluded])
    indices = np.random.default_rng(SEED).choice(remaining, size=COUNT,
                                                  replace=False).tolist()
    rows = []
    for row_index in indices:
        raw_ids = train_ids[row_index, :M43.PREFIX_TOKENS]
        excerpt = tok.decode(raw_ids, skip_special_tokens=True)
        assert len(excerpt.strip()) >= 20
        prompt_ids = tok.apply_chat_template(
            [{"role": "user", "content": M43.TEMPLATE.format(excerpt=excerpt)}],
            tokenize=True, add_generation_prompt=True)["input_ids"]
        assert 1 <= len(prompt_ids) <= 256
        rows.append({"train_row": row_index,
                     "source_ids_sha256": M17.sha(raw_ids.tobytes()),
                     "prompt_ids_sha256": M17.sha(np.asarray(
                         prompt_ids, dtype=np.int32).tobytes()),
                     "prompt_ids": prompt_ids})
    return {"experiment": "METH-44-chat-development", "seed": SEED,
            "source_file_sha256": M15.TRAIN_FILE_SHA,
            "source_ids_sha256": M15.TRAIN_IDS_SHA,
            "excluded_train_manifest_sha256": TRAIN_MANIFEST_SHA,
            "model": M42.MODEL, "revision": M42.REV,
            "tokenizer_fingerprint": M15.M13.TOK_FP,
            "template": M43.TEMPLATE, "count": COUNT, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sha256": M17.sha(args.out.read_bytes()),
                      "count": len(result["rows"])}, indent=2))


if __name__ == "__main__":
    main()
