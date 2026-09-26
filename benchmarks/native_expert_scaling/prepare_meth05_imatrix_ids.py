"""Prepare source-tokenizer IDs for the pinned GigaChat imatrix instrument.

The input is the existing non-held-out STRAT-01 calibration split. The output
is a local numerical exchange file, never the quality-evaluation heldout.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


CALIB_SHA256 = "68d9327a823328b1104c843afe986efeefcdd88c911c02895d577f65bffd8f81"
TOKENIZER_SHA256 = "b4b3d90c67830a4e566296ad8d0f6b5ac5a5cdbfd331b200d0ee8263aaaea1fe"
EXPECTED_CATEGORIES = {"code": 16, "prose": 16, "technical_general": 16}
VOCAB_SIZE = 128_256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(calib: Path, tokenizer_json: Path, output: Path) -> dict:
    if sha256(calib) != CALIB_SHA256:
        raise ValueError("calibration corpus SHA-256 mismatch")
    if sha256(tokenizer_json) != TOKENIZER_SHA256:
        raise ValueError("source tokenizer SHA-256 mismatch")
    if output.exists() or output.with_suffix(output.suffix + ".manifest.json").exists():
        raise FileExistsError("refusing to overwrite source-ID output or manifest")

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    special = {int(entry["id"]) for entry in json.loads(tokenizer_json.read_text(encoding="utf-8"))["added_tokens"]
               if entry.get("special")}
    counts: Counter[str] = Counter()
    grouped: dict[str, list[dict]] = {name: [] for name in EXPECTED_CATEGORIES}
    seen: set[str] = set()
    for raw in calib.read_text(encoding="utf-8").splitlines():
        row = json.loads(raw)
        item_id = row["source_document_id"]
        category = row["category"]
        content = row["text"]
        if not isinstance(item_id, str) or item_id in seen or not isinstance(content, str):
            raise ValueError("invalid or duplicate calibration document")
        if category not in grouped:
            raise ValueError(f"unexpected category: {category}")
        seen.add(item_id)
        grouped[category].append(row)
        counts[category] += 1
    if dict(counts) != EXPECTED_CATEGORIES:
        raise ValueError(f"unexpected category inventory: {dict(counts)}")

    # A bounded prefix now includes each domain rather than only code.
    order = ("code", "prose", "technical_general")
    rows = [grouped[category][index] for index in range(16) for category in order]
    item_ids: list[str] = []
    tokens: list[int] = []
    text_bytes = 0
    for row in rows:
        item_id = row["source_document_id"]
        content = row["text"]
        item_ids.append(item_id)
        encoded = tokenizer.encode(content, add_special_tokens=False).ids
        if not encoded or any(t < 0 or t >= VOCAB_SIZE or t in special for t in encoded):
            raise ValueError(f"invalid payload IDs in {item_id}")
        if tokenizer.decode(encoded, skip_special_tokens=False) != content:
            raise ValueError(f"source tokenizer round-trip failed in {item_id}")
        tokens.extend(encoded)
        text_bytes += len(content.encode("utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii", newline="\n") as handle:
        for token in tokens:
            handle.write(f"{token}\n")
    manifest = {
        "schema": "meth05_gigachat_source_imatrix_ids_v1",
        "source_corpus": str(calib),
        "source_corpus_sha256": CALIB_SHA256,
        "source_tokenizer": str(tokenizer_json),
        "source_tokenizer_sha256": TOKENIZER_SHA256,
        "output": str(output),
        "output_sha256": sha256(output),
        "documents": len(item_ids),
        "document_ids": item_ids,
        "category_counts": dict(counts),
        "document_order": "round_robin_code_prose_technical_general",
        "text_bytes": text_bytes,
        "token_count": len(tokens),
        "first_ids": tokens[:16],
        "last_ids": tokens[-16:],
        "special_id_count": 0,
        "round_trip_documents": len(item_ids),
        "limitation": "English/code-domain calibration pilot; zero Cyrillic in source corpus",
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.calib, args.tokenizer_json, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
