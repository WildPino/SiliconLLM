#!/usr/bin/env python3
"""Freeze METH-57 source-disjoint documents and chat prompts."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth41_fresh_c96_manifest as M41
import meth42_instruct_prompt_manifest as M42


ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth41_fresh_c96_manifest.json"
OLD_SHA = "8fff79f2502612d1f49083d1c48dd36d2718acec6335d676ccb5b8ded726365f"
M45 = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth45_fresh_external_manifest.json"
M45_SHA = "15a68db9a43a8066e70b1b261f819cab3d6779d8b52633bcce7c725d8d54acfb"
SEED = "meth57-external-57057"
COUNTS = {"code": 8, "prose": 8, "technical_general": 8}
MAIN_COUNTS = {"code": 8, "prose": 3}
PARQUET_REL = "data/external/pg19/data/test-00000-of-00001-29a571947c0b5ccc.parquet"
PARQUET_SHA = "9aae5ddf035760257458cff08d2575d78a15f84eff867af7a87eff0681b01bfc"
EXTRA_SEED = "meth57-pg19-extra-57057"
TECH_REF = "91b49fba9c83d4397ed78cfb253262deec5eef7c"
TECH_SEED = "meth57-tech-extra-57057"


def prior_context(selected, old_items, m45_items):
    _, m17_items = M17.build_selection()
    _, m19_items = M41.M19.select()
    _, m25_items = M41.M25.select()
    prior = m17_items + m19_items + m25_items + old_items + m45_items + selected
    excluded_ids = {row["source_id"] for row in prior}
    base = [(M17.CORPUS / name).read_bytes() for name in ("calib.txt", "heldout.txt")]
    assert [M17.sha(x) for x in base] == [M17.CALIB_SHA, M17.HELDOUT_SHA]
    base.extend(row["text"].encode("utf-8") for row in prior)
    return excluded_ids, base


def extra_prose(tokenizer, excluded_ids, base):
    parquet_path = ROOT / PARQUET_REL
    assert M17.sha(parquet_path.read_bytes()) == PARQUET_SHA
    candidates = []
    seen = 0
    row_index = 0
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=8, columns=["text"]):
        for value in batch.column("text").to_pylist():
            source_id = f"pg19:{PARQUET_REL}:row={row_index}"
            row_index += 1
            seen += 1
            if source_id in excluded_ids or not isinstance(value, str):
                continue
            raw = value.encode("utf-8")
            if len(raw) < M17.SPAN:
                continue
            digest = M17.sha((EXTRA_SEED + "|" + source_id).encode("utf-8"))
            start = int(digest[:16], 16) % (len(raw) - M17.SPAN + 1)
            text = raw[start:start + M17.SPAN].decode("utf-8", errors="ignore")
            data = text.encode("utf-8")
            if len(data) < 4000 or M17.fragment_overlap(data, base):
                continue
            candidates.append({"category": "prose", "source_id": source_id,
                               "source_kind": "pg19", "source_ref": PARQUET_REL,
                               "source_file_sha256": PARQUET_SHA,
                               "source_sha256": M17.sha(raw),
                               "span_start_byte": start, "text": text,
                               "selection_rank": digest})
    assert seen == 100, seen
    chosen = []
    for row in sorted(candidates, key=lambda x: x["selection_rank"]):
        data = row["text"].encode("utf-8")
        if M17.fragment_overlap(data, [x["text"].encode("utf-8") for x in chosen]):
            continue
        ids = tokenizer.encode(row["text"], add_special_tokens=False)
        if len(ids) < 256:
            continue
        chosen.append({k: v for k, v in row.items() if k != "selection_rank"} | {
            "ids": ids, "text_sha256": M17.sha(data), "bytes": len(data)})
        if len(chosen) == 5:
            break
    assert len(chosen) == 5, (len(chosen), len(candidates))
    assert not ({x["source_id"] for x in chosen} & excluded_ids)
    return chosen, {"parquet_sha256": PARQUET_SHA, "seed": EXTRA_SEED,
                    "rows_seen": seen, "candidates_after_prior_filter": len(candidates),
                    "chosen_source_ids": [x["source_id"] for x in chosen]}


def extra_technical(tokenizer, excluded_ids, base):
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", TECH_REF, "docs"],
        cwd=ROOT, text=True).splitlines()
    candidates = [path for path in paths
                  if path.endswith(".md")
                  and not path.startswith("docs/research/NATIVE_EXPERT_SCALING_20260925/")
                  and path not in excluded_ids]
    def rank(path):
        return M17.sha((TECH_SEED + "|" + path).encode("utf-8"))
    chosen = []
    for path in sorted(candidates, key=rank):
        raw = subprocess.check_output(["git", "show", f"{TECH_REF}:{path}"], cwd=ROOT)
        if len(raw) < M17.SPAN:
            continue
        digest = rank(path)
        start = int(digest[:16], 16) % (len(raw) - M17.SPAN + 1)
        text = raw[start:start + M17.SPAN].decode("utf-8", errors="ignore")
        data = text.encode("utf-8")
        if len(data) < 4000 or M17.fragment_overlap(data, base):
            continue
        if M17.fragment_overlap(data, [x["text"].encode("utf-8") for x in chosen]):
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 256:
            continue
        chosen.append({"category": "technical_general", "source_id": path,
                       "source_kind": "git", "source_ref": TECH_REF,
                       "source_sha256": M17.sha(raw),
                       "span_start_byte": start, "text": text,
                       "text_sha256": M17.sha(data), "bytes": len(data),
                       "ids": ids})
        if len(chosen) == 8:
            break
    assert len(chosen) == 8, (len(chosen), len(candidates))
    return chosen, {"git_commit": TECH_REF, "seed": TECH_SEED,
                    "candidate_paths": len(candidates), "chosen_source_ids":
                    [x["source_id"] for x in chosen]}


def build():
    assert M17.sha(OLD.read_bytes()) == OLD_SHA
    assert M17.sha(M45.read_bytes()) == M45_SHA
    old = json.loads(OLD.read_text(encoding="utf-8"))
    m45 = json.loads(M45.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        M42.MODEL, revision=M42.REV, local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    rebuilt_old, old_items = M41.select(tokenizer)
    assert rebuilt_old == old
    selection, selected = M41.select(tokenizer, seed=SEED,
                                      counts=MAIN_COUNTS,
                                      extra_exclusions=old_items + m45["items"])
    excluded_ids, base = prior_context(selected, old_items, m45["items"])
    prose, prose_selection = extra_prose(tokenizer, excluded_ids, base)
    selected.extend(prose)
    excluded_ids.update(row["source_id"] for row in prose)
    base.extend(row["text"].encode("utf-8") for row in prose)
    technical, tech_selection = extra_technical(tokenizer, excluded_ids, base)
    selected.extend(technical)
    old_ids = {row["source_id"] for row in old_items + m45["items"]}
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
    assert len(rows) == sum(COUNTS.values()) == 24
    assert all(sum(row["category"] == cat for row in rows) == count
               for cat, count in COUNTS.items())
    return {"experiment": "METH-57-product-key-external",
            "previous_selection_sha256": OLD_SHA,
            "previous_external_sha256": M45_SHA,
            "seed": SEED, "selected_counts": COUNTS,
            "model": M42.MODEL, "revision": M42.REV,
            "tokenizer_fingerprint": M15.M13.TOK_FP,
            "chat_template": M42.REQUEST,
            "excerpt_characters": 384,
            "selection_provenance": {"main": selection,
                                     "additional_prose": prose_selection,
                                     "additional_technical": tech_selection},
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
