#!/usr/bin/env python3
"""Freeze METH-45 source-disjoint documents and chat prompts."""
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth41_fresh_c96_manifest as M41
import meth42_instruct_prompt_manifest as M42


ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth41_fresh_c96_manifest.json"
OLD_SHA = "8fff79f2502612d1f49083d1c48dd36d2718acec6335d676ccb5b8ded726365f"
SEED = "meth45-external-45045"
COUNTS = {"code": 4, "prose": 4, "technical_general": 4}


def build():
    assert M17.sha(OLD.read_bytes()) == OLD_SHA
    old = json.loads(OLD.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        M42.MODEL, revision=M42.REV, local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    rebuilt_old, old_items = M41.select(tokenizer)
    assert rebuilt_old == old
    selection, selected = M41.select(tokenizer, seed=SEED,
                                      counts=COUNTS,
                                      extra_exclusions=old_items)
    old_ids = {row["source_id"] for row in old_items}
    rows = []
    for row in selected:
        assert row["source_id"] not in old_ids
        excerpt = row["text"][:384]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": M42.REQUEST.format(excerpt=excerpt)}],
            tokenize=True, add_generation_prompt=True)["input_ids"]
        assert 1 <= len(prompt) <= 1024
        rows.append({"category": row["category"],
                     "source_id": row["source_id"],
                     "source_kind": row["source_kind"],
                     "source_ref": row["source_ref"],
                     "source_sha256": row["source_sha256"],
                     "span_start_byte": row["span_start_byte"],
                     "text_sha256": row["text_sha256"],
                     "text": row["text"],
                     "bytes": row["bytes"],
                     "document_ids_sha256": M17.sha(np.asarray(
                         row["ids"], dtype=np.int32).tobytes()),
                     "document_ids": row["ids"],
                     "prompt_ids_sha256": M17.sha(np.asarray(
                         prompt, dtype=np.int32).tobytes()),
                     "prompt_ids": prompt})
    assert len(rows) == sum(COUNTS.values()) == 12
    return {"experiment": "METH-45-fresh-external",
            "previous_selection_sha256": OLD_SHA,
            "seed": SEED, "selected_counts": COUNTS,
            "model": M42.MODEL, "revision": M42.REV,
            "tokenizer_fingerprint": M15.M13.TOK_FP,
            "chat_template": M42.REQUEST,
            "excerpt_characters": 384,
            "selection_provenance": selection,
            "items": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    manifest = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({"sha256": M17.sha(args.out.read_bytes()),
                      "counts": COUNTS,
                      "max_document_tokens": max(len(x["document_ids"])
                                                 for x in manifest["items"]),
                      "max_prompt_tokens": max(len(x["prompt_ids"])
                                               for x in manifest["items"])}, indent=2))


if __name__ == "__main__":
    main()
