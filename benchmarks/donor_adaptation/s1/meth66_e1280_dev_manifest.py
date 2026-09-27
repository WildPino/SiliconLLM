#!/usr/bin/env python3
"""Freeze disjoint E128/E1280 development prompts before training."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

import meth15_zero_residual_expert_smoke as M15
import meth42_instruct_prompt_manifest as M42
import meth43_instruct_chat_train_manifest as M43


ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
PRIOR = (
    ("meth43_instruct_chat_train_manifest.json", "17921cb83c1d10793e2a56df6aa899039bd794757eeff43921068b8de2096055"),
    ("meth44_instruct_fresh_chat_manifest.json", "703abca3f83a983d9380ba84c1a79744d15881d638174b064cf38a47453f47ba"),
    ("meth47_retention_dev_manifest.json", "0195852ce7a46af5a2d9442b38e687a19393a943168b61a8f8cdb0c66ec78a60"),
    ("meth55_product_key_dev_manifest.json", "269322564f0e4d26aae58314c465c1f742e386656e294e93574062f88ccb3338"),
    ("meth56_product_key_dev_manifest.json", "46b6c5b8c14894e0525febb8677b055a3874293b4da0edac2e8801ff918b6a31"),
)
SEED = 666666
COUNT = 24


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    manifests = []
    for name, expected in PRIOR:
        path = DOCS / name
        assert sha(path.read_bytes()) == expected
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    excluded = {row["train_row"] for manifest in manifests for row in manifest["rows"]}
    assert len(excluded) == 256 + 4 * 24
    assert M15.M13.sha256(M15.TRAIN_PATH) == M15.TRAIN_FILE_SHA
    with np.load(M15.TRAIN_PATH, allow_pickle=False) as archive:
        ids = archive["ids"].copy()
    assert ids.shape == (31250, 512) and ids.dtype == np.int32
    assert sha(ids.tobytes()) == M15.TRAIN_IDS_SHA
    tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                               local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    pool = np.asarray([i for i in range(len(ids)) if i not in excluded])
    selected = np.random.default_rng(SEED).choice(pool, size=COUNT, replace=False).tolist()
    rows = []
    for index in selected:
        raw = ids[index, :M43.PREFIX_TOKENS]
        excerpt = tokenizer.decode(raw, skip_special_tokens=True)
        assert len(excerpt.strip()) >= 20
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": M43.TEMPLATE.format(excerpt=excerpt)}],
            tokenize=True, add_generation_prompt=True)["input_ids"]
        assert 1 <= len(prompt_ids) <= 256
        rows.append({"train_row": index, "source_ids_sha256": sha(raw.tobytes()),
                     "prompt_ids_sha256": sha(np.asarray(prompt_ids, dtype=np.int32).tobytes()),
                     "prompt_ids": prompt_ids})
    result = {"experiment": "METH-66-E128-E1280-development",
              "seed": SEED, "count": COUNT,
              "prior_manifest_sha256": dict(PRIOR),
              "source_file_sha256": M15.TRAIN_FILE_SHA,
              "source_ids_sha256": M15.TRAIN_IDS_SHA,
              "model": M42.MODEL, "revision": M42.REV,
              "tokenizer_fingerprint": M15.M13.TOK_FP,
              "template": M43.TEMPLATE, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sha256": sha(args.out.read_bytes()), "count": len(rows)}))


if __name__ == "__main__":
    main()
