#!/usr/bin/env python3
"""Freeze unseen PG19-shard-12 rows for semantic-boundary route validation."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import psutil
import pyarrow.parquet as pq
from transformers import AutoTokenizer

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth42_instruct_prompt_manifest as M42
import meth43_instruct_chat_train_manifest as M43


ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
SOURCE = ROOT / "data/external/pg19/data/train-00012-of-00023-78d820574bddc6bb.parquet"
SOURCE_SHA = "646d8e8dbe535cbdbab611463a446c32852ee570731635a7109232d34c702bcf"
TABLE = DOC / "meth172_threshold8_shared_table.json"
TABLE_SHA = "e89ff2cecbf887b85b5e361d47a8b918d8ccb93bd5c385a580ba993b93ac72db"
M172 = DOC / "meth172_low_recurrence_route_result.json"
SEED = "meth173-fresh-recurrence-route-173173"
COUNTS = {"chat": 512, "raw": 512, "document": 24}
MAX_SECONDS = 5 * 60
MAX_RSS = 8 * (1 << 30)


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def budget(start):
    result = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss}
    if result["seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS:
        raise RuntimeError(f"METH-173 manifest resource stop: {result}")
    return result


def rank(part, row):
    return M17.sha(f"{SEED}|{part}|{row}".encode("ascii"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--candidate-sha", required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    start = time.monotonic()
    assert file_sha(M172) == args.candidate_sha
    candidate = json.loads(M172.read_text(encoding="utf-8"))
    assert candidate["decision"] == "postfailure_in_sample_candidate_pass_fresh_source_pending"
    assert file_sha(SOURCE) == SOURCE_SHA
    assert file_sha(TABLE) == TABLE_SHA
    tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                               local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    source = pq.ParquetFile(SOURCE)
    assert source.metadata.num_rows == 1244
    texts = source.read(columns=["text"]).column("text").to_pylist()
    assert len(texts) == 1244
    selected = set()
    rows = {part: [] for part in COUNTS}
    rejected = {part: {"empty": 0, "short_source": 0,
                       "short_excerpt": 0, "long_prompt": 0}
                for part in COUNTS}
    for part, count in COUNTS.items():
        for row in sorted(range(len(texts)), key=lambda i: rank(part, i)):
            if row in selected:
                continue
            text = texts[row]
            if not isinstance(text, str) or not text:
                rejected[part]["empty"] += 1
                continue
            offset = 256 + int(rank("offset", row)[:8], 16) % 512
            ids = tokenizer.encode(text[:24000], add_special_tokens=False)
            if len(ids) < offset + 1024:
                rejected[part]["short_source"] += 1
                continue
            length = {"chat": 96, "raw": 128, "document": 1024}[part]
            span = np.asarray(ids[offset:offset + length], dtype=np.int32)
            entry = {"source_row": row, "source_text_sha256":
                     M17.sha(text.encode("utf-8")),
                     "source_token_offset": offset,
                     "span_ids_sha256": M17.sha(span.tobytes()),
                     "span_ids": span.tolist()}
            if part == "chat":
                excerpt = tokenizer.decode(span, skip_special_tokens=True)
                if len(excerpt.strip()) < 20:
                    rejected[part]["short_excerpt"] += 1
                    continue
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": M43.TEMPLATE.format(excerpt=excerpt)}],
                    tokenize=True, add_generation_prompt=True)["input_ids"]
                if not 1 <= len(prompt) <= 256:
                    rejected[part]["long_prompt"] += 1
                    continue
                entry["excerpt_sha256"] = M17.sha(excerpt.encode("utf-8"))
                entry["prompt_ids_sha256"] = M17.sha(np.asarray(
                    prompt, dtype=np.int32).tobytes())
                entry["prompt_ids"] = prompt
            rows[part].append(entry)
            selected.add(row)
            if len(rows[part]) == count:
                break
            budget(start)
        assert len(rows[part]) == count, (part, len(rows[part]))
    assert len(selected) == sum(COUNTS.values())
    result = {"experiment": "METH-173-fresh-recurrence-route-input-manifest",
              "source": str(SOURCE.relative_to(ROOT)),
              "source_sha256": SOURCE_SHA, "source_rows": len(texts),
              "candidate_meth172_result_sha256": args.candidate_sha,
              "table_sha256": TABLE_SHA, "seed": SEED,
              "model": M42.MODEL, "revision": M42.REV,
              "tokenizer_fingerprint": M15.M13.TOK_FP,
              "template": M43.TEMPLATE,
              "counts": COUNTS, "rejected": rejected,
              "chat_rows": rows["chat"], "raw_rows": rows["raw"],
              "document_rows": rows["document"],
              "runtime": budget(start)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"sha256": file_sha(args.out), "counts": COUNTS,
                      "rejected": rejected, "runtime": result["runtime"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()

