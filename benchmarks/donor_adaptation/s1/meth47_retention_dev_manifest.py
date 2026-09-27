#!/usr/bin/env python3
"""Freeze another 24 disjoint chat development prompts for retention repair."""
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
M43_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth43_instruct_chat_train_manifest.json"
M43_SHA = "17921cb83c1d10793e2a56df6aa899039bd794757eeff43921068b8de2096055"
M44_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth44_instruct_fresh_chat_manifest.json"
M44_SHA = "703abca3f83a983d9380ba84c1a79744d15881d638174b064cf38a47453f47ba"
SEED = 474747
COUNT = 24


def build():
    assert M17.sha(M43_PATH.read_bytes()) == M43_SHA
    assert M17.sha(M44_PATH.read_bytes()) == M44_SHA
    m43 = json.loads(M43_PATH.read_text(encoding="utf-8"))
    m44 = json.loads(M44_PATH.read_text(encoding="utf-8"))
    excluded = {r["train_row"] for r in m43["rows"] + m44["rows"]}
    assert len(excluded) == 280
    assert M15.M13.sha256(M15.TRAIN_PATH) == M15.TRAIN_FILE_SHA
    with np.load(M15.TRAIN_PATH, allow_pickle=False) as archive:
        train_ids = archive["ids"].copy()
    assert train_ids.shape == (31250, 512) and train_ids.dtype == np.int32
    assert M17.sha(train_ids.tobytes()) == M15.TRAIN_IDS_SHA
    tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                               local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    pool = np.asarray([i for i in range(len(train_ids)) if i not in excluded])
    indices = np.random.default_rng(SEED).choice(pool, size=COUNT,
                                                  replace=False).tolist()
    rows = []
    for index in indices:
        raw_ids = train_ids[index, :M43.PREFIX_TOKENS]
        excerpt = tokenizer.decode(raw_ids, skip_special_tokens=True)
        assert len(excerpt.strip()) >= 20
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": M43.TEMPLATE.format(excerpt=excerpt)}],
            tokenize=True, add_generation_prompt=True)["input_ids"]
        assert 1 <= len(ids) <= 256
        rows.append({"train_row": index,
                     "source_ids_sha256": M17.sha(raw_ids.tobytes()),
                     "prompt_ids_sha256": M17.sha(np.asarray(
                         ids, dtype=np.int32).tobytes()),
                     "prompt_ids": ids})
    return {"experiment": "METH-47-chat-development", "seed": SEED,
            "excluded_train_manifest_sha256": M43_SHA,
            "excluded_development_manifest_sha256": M44_SHA,
            "source_file_sha256": M15.TRAIN_FILE_SHA,
            "source_ids_sha256": M15.TRAIN_IDS_SHA,
            "model": M42.MODEL, "revision": M42.REV,
            "tokenizer_fingerprint": M15.M13.TOK_FP,
            "template": M43.TEMPLATE, "count": COUNT, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    manifest = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sha256": M17.sha(args.out.read_bytes()),
                      "count": COUNT,
                      "min_prompt_tokens": min(len(x["prompt_ids"])
                                               for x in manifest["rows"]),
                      "max_prompt_tokens": max(len(x["prompt_ids"])
                                               for x in manifest["rows"])}, indent=2))


if __name__ == "__main__":
    main()
