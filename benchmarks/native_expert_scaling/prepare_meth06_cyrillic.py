"""Freeze distinct RU/UK calibration spans and source-tokenizer IDs for METH-06."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


OLD_CALIB_SHA256 = "68d9327a823328b1104c843afe986efeefcdd88c911c02895d577f65bffd8f81"
HELDOUT_SHA256 = "04687034c1054e0985e24efa87ba37a3fb11031de5b2880e5b8ac1742f80ec6e"
TOKENIZER_SHA256 = "b4b3d90c67830a4e566296ad8d0f6b5ac5a5cdbfd331b200d0ee8263aaaea1fe"
SPAN_BYTES = 4095
MIN_CYRILLIC = 100
EXPECTED_LANGUAGES = {"ru": 74, "uk": 16}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cyrillic_count(text: str) -> int:
    return sum("\u0400" <= char <= "\u04ff" for char in text)


def best_span(raw: bytes) -> tuple[int, int, str, int]:
    candidates = []
    for offset in range(0, len(raw) - SPAN_BYTES + 1, SPAN_BYTES):
        start = offset
        end = offset + SPAN_BYTES
        while start < end and raw[start] & 0xC0 == 0x80:
            start += 1
        while end > start and end < len(raw) and raw[end] & 0xC0 == 0x80:
            end -= 1
        try:
            text = raw[start:end].decode("utf-8")
        except UnicodeDecodeError:
            continue
        candidates.append((cyrillic_count(text), -start, start, end, text))
    if not candidates:
        raise ValueError("source has no valid UTF-8 span")
    count, _, start, end, text = max(candidates)
    return start, end, text, count


def excluded_sources(calib: Path, heldout: Path) -> tuple[set[str], set[str]]:
    if sha256(calib) != OLD_CALIB_SHA256 or sha256(heldout) != HELDOUT_SHA256:
        raise ValueError("pinned calibration/heldout SHA-256 mismatch")
    paths: set[str] = set()
    hashes: set[str] = set()
    for source in (calib, heldout):
        for line in source.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            paths.add(row["source_path"].replace("\\", "/"))
            hashes.add(row["source_content_sha256"])
    return paths, hashes


def prepare(root: Path, calib: Path, heldout: Path, tokenizer_json: Path,
            corpus_out: Path, ids_out: Path, manifest_out: Path) -> dict:
    if any(path.exists() for path in (corpus_out, ids_out, manifest_out)):
        raise FileExistsError("refusing to overwrite METH-06 artifacts")
    if sha256(tokenizer_json) != TOKENIZER_SHA256:
        raise ValueError("source tokenizer SHA-256 mismatch")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    special = {int(entry["id"]) for entry in json.loads(tokenizer_json.read_text(encoding="utf-8"))["added_tokens"]
               if entry.get("special")}
    excluded_paths, excluded_hashes = excluded_sources(calib, heldout)
    groups: dict[str, list[dict]] = {language: [] for language in EXPECTED_LANGUAGES}
    for language in EXPECTED_LANGUAGES:
        language_dir = root / "data/external/markdown_corpus/kubernetes_website/content" / language
        for path in sorted(language_dir.rglob("*.md")):
            raw = path.read_bytes()
            if len(raw) < SPAN_BYTES + 1:
                continue
            relative = path.relative_to(root).as_posix()
            source_hash = hashlib.sha256(raw).hexdigest()
            if relative in excluded_paths or source_hash in excluded_hashes:
                raise ValueError(f"candidate source overlaps existing calibration/heldout: {relative}")
            start, end, content, cyrillic = best_span(raw)
            if cyrillic < MIN_CYRILLIC:
                continue
            ids = tokenizer.encode(content, add_special_tokens=False).ids
            if not ids or any(token < 0 or token >= 128_256 or token in special for token in ids):
                raise ValueError(f"invalid source-tokenizer IDs: {relative}")
            if tokenizer.decode(ids, skip_special_tokens=False) != content:
                raise ValueError(f"source tokenizer round-trip failed: {relative}")
            groups[language].append({
                "source_document_id": f"file:{relative}:byte={start}-{end}",
                "source_path": relative,
                "source_content_sha256": source_hash,
                "source_file_bytes": len(raw),
                "span_start_byte": start,
                "span_end_byte": end,
                "span_sha256": hashlib.sha256(raw[start:end]).hexdigest(),
                "language": language,
                "cyrillic_characters": cyrillic,
                "text": content,
                "ids": ids,
            })
    counts = {language: len(rows) for language, rows in groups.items()}
    if counts != EXPECTED_LANGUAGES:
        raise ValueError(f"unexpected eligible source inventory: {counts}")
    ordered: list[dict] = []
    for index in range(max(counts.values())):
        for language in ("ru", "uk"):
            if index < counts[language]:
                ordered.append(groups[language][index])
    if len({row["source_content_sha256"] for row in ordered}) != len(ordered):
        raise ValueError("duplicate source content among new calibration files")
    if len({row["source_document_id"] for row in ordered}) != len(ordered):
        raise ValueError("duplicate source IDs in new calibration")

    corpus_out.parent.mkdir(parents=True, exist_ok=True)
    ids_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with corpus_out.open("x", encoding="utf-8", newline="\n") as handle:
        for row in ordered:
            record = {key: value for key, value in row.items() if key != "ids"}
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    with ids_out.open("x", encoding="ascii", newline="\n") as handle:
        for row in ordered:
            for token in row["ids"]:
                handle.write(f"{token}\n")
    report = {
        "schema": "meth06_cyrillic_calibration_v1",
        "selection": "all RU/UK Kubernetes Markdown files >=4096 bytes; one nonoverlapping 4095-byte window with maximum Cyrillic count per file, earliest tie",
        "ordering": "round_robin_ru_uk_then_remaining_ru",
        "minimum_cyrillic_characters_per_span": MIN_CYRILLIC,
        "source_tokenizer_sha256": TOKENIZER_SHA256,
        "excluded_old_calib_sha256": OLD_CALIB_SHA256,
        "excluded_heldout_sha256": HELDOUT_SHA256,
        "files_by_language": counts,
        "documents": len(ordered),
        "source_paths_disjoint": True,
        "source_content_sha256_disjoint": True,
        "total_span_bytes": sum(len(row["text"].encode("utf-8")) for row in ordered),
        "total_cyrillic_characters": sum(row["cyrillic_characters"] for row in ordered),
        "token_count": sum(len(row["ids"]) for row in ordered),
        "corpus": str(corpus_out),
        "corpus_sha256": sha256(corpus_out),
        "ids": str(ids_out),
        "ids_sha256": sha256(ids_out),
        "documents_manifest": [
            {key: row[key] for key in ("source_document_id", "source_path", "source_content_sha256",
                                       "span_sha256", "span_start_byte", "span_end_byte", "language",
                                       "cyrillic_characters")}
            for row in ordered
        ],
    }
    with manifest_out.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-calib", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--corpus-out", type=Path, required=True)
    parser.add_argument("--ids-out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.root.resolve(), args.old_calib, args.heldout,
                     args.tokenizer_json, args.corpus_out, args.ids_out, args.manifest_out)
    print(json.dumps({key: value for key, value in result.items() if key != "documents_manifest"}, indent=2))


if __name__ == "__main__":
    main()
