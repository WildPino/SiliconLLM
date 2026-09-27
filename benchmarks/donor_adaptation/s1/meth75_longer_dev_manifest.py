#!/usr/bin/env python3
"""Freeze METH-75 prompts outside the entire METH-71 256-update draw stream."""

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

import meth15_zero_residual_expert_smoke as M15
import meth42_instruct_prompt_manifest as M42
import meth43_instruct_chat_train_manifest as M43
import meth70_dense_scale_dev_manifest as M70


ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
PRIOR = M70.PRIOR + (("meth70_dense_scale_dev_manifest.json",
                      "fa77b745fa63c953cb8f50963e6e6f93a9dc26d7bc8f0152c93380b4a9f91607"),)
M71_NAME = "meth71_balanced_dev_manifest.json"
M71_SHA = "4bb5eb221c0754e6769f0faff5f8487d7f20cfa5dac029531ad174cede7840a8"
SEED = 757575
COUNT = 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    seen = set()
    for name, expected in PRIOR:
        path = DOCS / name
        assert M70.sha(path.read_bytes()) == expected, name
        seen.update(row["train_row"] for row in json.loads(path.read_text(encoding="utf-8"))["rows"])
    assert len(seen) == 256 + 6 * 24

    # Recreate the exact METH-70 sampling calls, including chat draws which
    # advance the same RNG. Both E arms consumed this identical stream.
    prior70 = json.loads((DOCS / PRIOR[-1][0]).read_text(encoding="utf-8"))
    prior70_seen = set(seen) - {row["train_row"] for row in prior70["rows"]}
    assert len(prior70_seen) == 256 + 5 * 24
    prior70_pool = np.asarray([i for i in range(31250) if i not in prior70_seen])
    rng70 = np.random.default_rng(707071)
    sampled70 = []
    for _ in range(16):
        for micro in range(4):
            if micro < 2:
                sampled70.append(int(rng70.choice(prior70_pool)))
                rng70.integers(512 - M15.SEQ + 1)
            else:
                rng70.integers(256)
    assert len(sampled70) == 32
    seen.update(sampled70)
    m71_path = DOCS / M71_NAME
    assert M70.sha(m71_path.read_bytes()) == M71_SHA
    m71 = json.loads(m71_path.read_text(encoding="utf-8"))
    assert len(m71["rows"]) == 24
    assert set(sampled70).isdisjoint(row["train_row"] for row in m71["rows"])
    seen.update(row["train_row"] for row in m71["rows"])
    assert len(seen) == 256 + 7 * 24 + 32
    raw_pool71 = np.asarray([i for i in range(31250) if i not in seen])
    rng71 = np.random.default_rng(717172)
    raw_draws = []
    chat_draws = []
    state_after64 = None
    for update in range(1, 257):
        for micro in range(4):
            if micro < 2:
                raw_draws.append({"row": int(rng71.choice(raw_pool71)),
                                  "offset": int(rng71.integers(512 - M15.SEQ + 1))})
            else:
                chat_draws.append(int(rng71.integers(256)))
        if update == 64:
            state_after64 = rng71.bit_generator.state
    assert len(raw_draws) == len(chat_draws) == 512
    seen.update(draw["row"] for draw in raw_draws)

    assert M15.M13.sha256(M15.TRAIN_PATH) == M15.TRAIN_FILE_SHA
    with np.load(M15.TRAIN_PATH, allow_pickle=False) as archive:
        ids = archive["ids"].copy()
    assert ids.shape == (31250, 512) and ids.dtype == np.int32
    assert M70.sha(ids.tobytes()) == M15.TRAIN_IDS_SHA
    tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                               local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    pool = np.asarray([i for i in range(len(ids)) if i not in seen])
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
        rows.append({"train_row": index,
                     "source_ids_sha256": M70.sha(raw.tobytes()),
                     "prompt_ids_sha256": M70.sha(np.asarray(prompt_ids, dtype=np.int32).tobytes()),
                     "prompt_ids": prompt_ids})
    result = {"experiment": "METH-75-balanced-continuation-development",
              "seed": SEED, "count": COUNT,
              "prior_manifest_sha256": dict(PRIOR),
              "meth71_dev_manifest_sha256": M71_SHA,
              "meth70_sampled_raw_rows": sorted(set(sampled70)),
              "meth70_sampled_raw_draws": sampled70,
              "meth71_raw_pool_count": len(raw_pool71),
              "meth71_raw_training_draws_256": raw_draws,
              "meth71_chat_training_draws_256": chat_draws,
              "meth71_training_rng_state_after_64": state_after64,
              "source_file_sha256": M15.TRAIN_FILE_SHA,
              "source_ids_sha256": M15.TRAIN_IDS_SHA,
              "model": M42.MODEL, "revision": M42.REV,
              "tokenizer_fingerprint": M15.M13.TOK_FP,
              "template": M43.TEMPLATE, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sha256": M70.sha(args.out.read_bytes()), "count": len(rows),
                      "meth70_sampled_raw_unique": len(set(sampled70)),
                      "meth71_sampled_raw_unique": len({d["row"] for d in raw_draws})}))


if __name__ == "__main__":
    main()
