#!/usr/bin/env python3
"""Freeze disjoint E128 product-key development prompts before training."""
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
DOCS = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
PRIOR = (
    ("meth43_instruct_chat_train_manifest.json", "17921cb83c1d10793e2a56df6aa899039bd794757eeff43921068b8de2096055"),
    ("meth44_instruct_fresh_chat_manifest.json", "703abca3f83a983d9380ba84c1a79744d15881d638174b064cf38a47453f47ba"),
    ("meth47_retention_dev_manifest.json", "0195852ce7a46af5a2d9442b38e687a19393a943168b61a8f8cdb0c66ec78a60"),
)
SEED = 555555
COUNT = 24


def build():
    manifests = []
    for name, expected_sha in PRIOR:
        path = DOCS / name
        assert M17.sha(path.read_bytes()) == expected_sha, name
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    excluded = {row["train_row"] for manifest in manifests for row in manifest["rows"]}
    assert len(excluded) == 304
    assert M15.M13.sha256(M15.TRAIN_PATH) == M15.TRAIN_FILE_SHA
    with np.load(M15.TRAIN_PATH, allow_pickle=False) as archive:
        train_ids = archive["ids"].copy()
    assert train_ids.shape == (31250, 512) and train_ids.dtype == np.int32
    assert M17.sha(train_ids.tobytes()) == M15.TRAIN_IDS_SHA
    tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV, local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    pool = np.asarray([i for i in range(len(train_ids)) if i not in excluded])
    indices = np.random.default_rng(SEED).choice(pool, size=COUNT, replace=False).tolist()
    rows = []
    for index in indices:
        raw_ids = train_ids[index, :M43.PREFIX_TOKENS]
        excerpt = tokenizer.decode(raw_ids, skip_special_tokens=True)
        assert len(excerpt.strip()) >= 20
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": M43.TEMPLATE.format(excerpt=excerpt)}],
            tokenize=True, add_generation_prompt=True)["input_ids"]
        assert 1 <= len(prompt_ids) <= 256
        rows.append({
            "train_row": index,
            "source_ids_sha256": M17.sha(raw_ids.tobytes()),
            "prompt_ids_sha256": M17.sha(np.asarray(prompt_ids, dtype=np.int32).tobytes()),
            "prompt_ids": prompt_ids,
        })
    return {
        "experiment": "METH-55-product-key-chat-development",
        "seed": SEED,
        "prior_manifest_sha256": {name: digest for name, digest in PRIOR},
        "source_file_sha256": M15.TRAIN_FILE_SHA,
        "source_ids_sha256": M15.TRAIN_IDS_SHA,
        "model": M42.MODEL,
        "revision": M42.REV,
        "tokenizer_fingerprint": M15.M13.TOK_FP,
        "template": M43.TEMPLATE,
        "count": COUNT,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sha256": M17.sha(args.out.read_bytes()), "count": len(result["rows"])}))


if __name__ == "__main__":
    main()
