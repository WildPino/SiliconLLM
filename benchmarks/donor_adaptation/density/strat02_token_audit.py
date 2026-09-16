#!/usr/bin/env python3
"""Deterministic, read-only STRAT-02 corpus/tokenizer gate audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DONOR = "allenai/StdMoE_1b14b_1T_Preanneal"
REVISION = "d2a4949c9d4ad6cf47fbac131f7e020077332b21"
EOS_ID, MAX_CONTEXT = 100257, 4096
ASSETS = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")
EXPECTED_MANIFEST_SHA256 = "56f3d707040785b21e657d7ba721814fb32fe63e85feb418fafe38889d8ca749"
EXPECTED_ASSET_SHA256 = {
    "tokenizer.json": "73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca",
    "tokenizer_config.json": "733b2ec0f743cf206dd20335eacbd4427c85487fd6f5216bb361f92a85314315",
    "special_tokens_map.json": "4cb2b52960bababa8ec27153831e405ef7811e3fad1226106e09c28e715cfb21",
    "vocab.json": "9e14712c91b37c7aab74b1306baa46ac342d620637a4b44523cdc3aec7d24195",
    "merges.txt": "b6fe424e334903f7fb84d3a106d9730455f4744b9fe3c21ee136d97a00e72502",
}
EXPECTED_IDS_SHA256 = {"calib": "430e44946673ed06cc2afcbfb373611c366dd2c363a3af652ffa42bf5a0869f4",
                       "heldout": "5a4cf31ec7db0f3bc541f2e75a78b133a50cd3d6174abb26ef81e08c45e65289"}
EXPECTED_TOTALS = {"calib": 88756, "heldout": 181385}
TRANSFORMERS_VERSION, TOKENIZERS_VERSION = "4.57.1", "0.22.2"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def read_rows(path: Path, split: str, errors: list[str]) -> list[dict]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        fail(errors, f"{split}: cannot read {path}: {exc}")
        return []
    rows = []
    try:
        for number, line in enumerate(raw.splitlines(), 1):
            if not line:
                fail(errors, f"{split}: blank JSONL line {number}")
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("row is not an object")
            rows.append(row)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        fail(errors, f"{split}: invalid JSONL: {exc}")
    return rows


def aggregate(rows: list[dict]) -> str:
    return sha("".join(row.get("item_sha256", "") + "\n" for row in rows).encode("ascii"))


def split_overlaps(rows_by_split: dict[str, list[dict]]) -> tuple[set, set]:
    calib, heldout = rows_by_split["calib"], rows_by_split["heldout"]
    return ({row.get("source_document_id") for row in calib} & {row.get("source_document_id") for row in heldout},
            {row.get("source_content_sha256") for row in calib} & {row.get("source_content_sha256") for row in heldout})


def check_corpus(manifest_path: Path, paths: dict[str, Path], errors: list[str]) -> tuple[dict, dict[str, list[dict]]]:
    try:
        manifest_raw = manifest_path.read_bytes()
        if sha(manifest_raw) != EXPECTED_MANIFEST_SHA256:
            fail(errors, "manifest: SHA-256 differs from STRAT-02 pin")
        manifest = json.loads(manifest_raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(errors, f"manifest: invalid {manifest_path}: {exc}")
        return {}, {name: [] for name in paths}
    rows_by_split = {split: read_rows(path, split, errors) for split, path in paths.items()}
    for split, rows in rows_by_split.items():
        part = manifest.get("parts", {}).get(split)
        if not isinstance(part, dict):
            fail(errors, f"manifest: missing parts.{split}")
            continue
        try:
            raw = paths[split].read_bytes()
            if sha(raw) != part.get("jsonl_sha256"):
                fail(errors, f"{split}: JSONL SHA-256 differs from manifest")
        except OSError:
            pass
        if len(rows) != part.get("item_count"):
            fail(errors, f"{split}: item count differs from manifest")
        counts = dict(sorted(Counter(row.get("category") for row in rows).items()))
        if counts != part.get("category_counts"):
            fail(errors, f"{split}: category counts differ from manifest")
        if sum(len(str(row.get("text", "")).encode("utf-8")) for row in rows) != part.get("total_span_bytes"):
            fail(errors, f"{split}: total UTF-8 span bytes differ from manifest")
        if aggregate(rows) != part.get("items_aggregate_sha256"):
            fail(errors, f"{split}: item aggregate SHA-256 differs from manifest")
        for number, row in enumerate(rows, 1):
            if row.get("schema") != manifest.get("schema"):
                fail(errors, f"{split}:{number}: schema mismatch")
            if row.get("split") != split:
                fail(errors, f"{split}:{number}: split mismatch")
            if row.get("category") not in {"code", "prose", "technical_general"}:
                fail(errors, f"{split}:{number}: invalid category")
            text = row.get("text")
            if not isinstance(text, str) or text.encode("utf-8").decode("utf-8") != text:
                fail(errors, f"{split}:{number}: text is not exact UTF-8")
                continue
            raw_text = text.encode("utf-8")
            if len(raw_text) != row.get("span_byte_count"):
                fail(errors, f"{split}:{number}: span byte count mismatch")
            unsigned = dict(row)
            observed = unsigned.pop("item_sha256", None)
            if not isinstance(observed, str) or sha(canonical(unsigned)) != observed:
                fail(errors, f"{split}:{number}: item SHA-256 mismatch")
    document_overlap, content_overlap = split_overlaps(rows_by_split)
    if document_overlap:
        fail(errors, f"document overlap across splits: {len(document_overlap)}")
    if content_overlap:
        fail(errors, f"content overlap across splits: {len(content_overlap)}")
    return manifest, rows_by_split


def load_tokenizer(snapshot: Path):
    if snapshot.name != REVISION or not snapshot.is_dir():
        raise RuntimeError(f"tokenizer snapshot must be the pinned local revision directory {REVISION}")
    missing = [name for name in ASSETS if not (snapshot / name).is_file()]
    if missing:
        raise RuntimeError(f"tokenizer snapshot missing assets: {', '.join(missing)}")
    import tokenizers
    import transformers
    from transformers import GPT2TokenizerFast
    return GPT2TokenizerFast.from_pretrained(str(snapshot), local_files_only=True), GPT2TokenizerFast, transformers.__version__, tokenizers.__version__


def check_tokens(snapshot: Path, rows_by_split: dict[str, list[dict]], errors: list[str]) -> dict:
    asset_sha256 = {}
    for name in ASSETS:
        try:
            asset_sha256[name] = sha((snapshot / name).read_bytes())
        except OSError as exc:
            fail(errors, f"tokenizer asset {name}: {exc}")
    if asset_sha256 != EXPECTED_ASSET_SHA256:
        fail(errors, "tokenizer: asset SHA-256 differs from STRAT-02 pin")
    try:
        tokenizer, fast_class, transformers_version, tokenizers_version = load_tokenizer(snapshot)
    except Exception as exc:
        fail(errors, f"tokenizer: {exc}")
        return {"snapshot": str(snapshot), "asset_sha256": asset_sha256}
    if transformers_version != TRANSFORMERS_VERSION or tokenizers_version != TOKENIZERS_VERSION:
        fail(errors, f"tokenizer: require transformers=={TRANSFORMERS_VERSION}, tokenizers=={TOKENIZERS_VERSION}; got {transformers_version}, {tokenizers_version}")
    if type(tokenizer) is not fast_class: fail(errors, "tokenizer: exact class is not GPT2TokenizerFast")
    if tokenizer.bos_token_id is not None or tokenizer.eos_token_id != EOS_ID or tokenizer.model_max_length != MAX_CONTEXT:
        fail(errors, "tokenizer: expected bos=None, eos=100257, max_context=4096")
    special_ids = set(tokenizer.all_special_ids)
    stats = {}
    for split, rows in rows_by_split.items():
        counts = []
        ids_by_doc = []
        for number, row in enumerate(rows, 1):
            text = row.get("text", "")
            ids = tokenizer.encode(text, add_special_tokens=False)
            ids_by_doc.append(ids)
            if tokenizer.decode(ids, clean_up_tokenization_spaces=False) != text:
                fail(errors, f"{split}:{number}: tokenizer roundtrip mismatch")
            if special_ids.intersection(ids):
                fail(errors, f"{split}:{number}: payload contains special token ID")
            scored = 1 + len(ids)  # prepend EOS; score every raw-text payload token.
            if scored > MAX_CONTEXT:
                fail(errors, f"{split}:{number}: context {scored} exceeds {MAX_CONTEXT}")
            counts.append(len(ids))
        ids_sha256 = sha(json.dumps(ids_by_doc, separators=(",", ":")).encode("ascii"))
        if ids_sha256 != EXPECTED_IDS_SHA256[split]:
            fail(errors, f"{split}: ordered token-ID SHA-256 differs from STRAT-02 pin")
        if sum(counts) != EXPECTED_TOTALS[split]:
            fail(errors, f"{split}: token total differs from STRAT-02 pin")
        stats[split] = {"documents": len(counts), "payload_tokens_min": min(counts, default=0),
                        "payload_tokens_max": max(counts, default=0), "payload_tokens_total": sum(counts),
                        "ordered_token_ids_sha256": ids_sha256, "max_context_tokens": (max(counts, default=-1) + 1)}
    return {"donor": DONOR, "revision": REVISION, "snapshot": str(snapshot), "asset_sha256": asset_sha256,
            "versions": {"transformers": transformers_version, "tokenizers": tokenizers_version},
            "protocol": {"prepend_token_id": EOS_ID, "encode_add_special_tokens": False,
                         "payload_tokens_scored": True, "denominator": "raw_utf8_bytes", "truncation": False, "stride": False},
            "splits": stats}


def audit(args: argparse.Namespace) -> dict:
    errors: list[str] = []
    manifest, rows = check_corpus(args.manifest, {"calib": args.calib, "heldout": args.heldout}, errors)
    token = check_tokens(args.tokenizer_snapshot, rows, errors)
    return {"ok": not errors, "donor": DONOR, "revision": REVISION, "manifest": str(args.manifest),
            "schema": manifest.get("schema"), "tokenizer": token, "errors": errors}


def selftest() -> None:
    assert sha(b"x") == hashlib.sha256(b"x").hexdigest()
    row = {"schema": "s", "split": "calib", "category": "code", "text": "a\u00e9", "span_byte_count": 3}
    row["item_sha256"] = sha(canonical(row))
    assert sha(canonical({k: v for k, v in row.items() if k != "item_sha256"})) == row["item_sha256"]
    changed = dict(row); changed["text"] = "b"
    assert sha(canonical({k: v for k, v in changed.items() if k != "item_sha256"})) != changed["item_sha256"]
    assert split_overlaps({"calib": [dict(row, source_document_id="a", source_content_sha256="x")],
                           "heldout": [dict(row, source_document_id="a", source_content_sha256="x")]}) == ({"a"}, {"x"})
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x.jsonl"; p.write_text(json.dumps(row) + "\n", encoding="utf-8")
        errors: list[str] = []
        assert read_rows(p, "calib", errors) == [row] and not errors
    print(json.dumps({"ok": True, "selftest": True}, sort_keys=True))


def main() -> int:
    corpus = HERE / "corpus" / "strat02_document_holdout_v1"
    default_cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    default_snapshot = default_cache / "models--allenai--StdMoE_1b14b_1T_Preanneal" / "snapshots" / REVISION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=corpus / "manifest.json")
    parser.add_argument("--calib", type=Path, default=corpus / "calib.jsonl")
    parser.add_argument("--heldout", type=Path, default=corpus / "heldout.jsonl")
    parser.add_argument("--tokenizer-snapshot", type=Path, default=default_snapshot)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest(); return 0
    report = audit(args)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
