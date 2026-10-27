#!/usr/bin/env python3
"""Build the R1 document-preserving calibration/heldout text manifests.

Unlike ``build_calib.py``, this tool never chunks or concatenates documents before
the split.  Code and technical-markdown documents are assigned from their complete
UTF-8 content SHA-256, then a bounded, UTF-8-valid span is selected from each chosen
document.  PG19 rows retain the publisher file split (``train`` -> calibration,
``test`` -> heldout); that routing is provenance, not proof that the publishers'
row IDs or their semantic content are disjoint.  The generated manifest therefore
checks both IDs and exact content hashes across the two outputs.

The byte count recorded here is the raw *span* length, not automatically the
BPB denominator: a causal scorer must compute the UTF-8 bytes of its actually
scored continuation after tokenizer/BOS/stride policy is frozen.

Raw text can have redistribution restrictions.  ``build`` deliberately has no
default output location: pass an explicit local/ignored ``--out-dir`` and do not
commit its JSONL files.  A reviewed manifest can be copied separately if appropriate.

Scientific limitation: exact SHA-256 deduplication prevents only byte-identical
crossovers.  It does not establish absence of near duplicates, translations, or
other semantic overlap.  The publisher PG19 train/test naming is likewise not an
assumption of row-level disjointness; the emitted IDs and content hashes are checked.

Examples:
  python build_document_holdout.py preflight --config r1_preflight_v1
  python build_document_holdout.py preflight --full-content --config r1_preflight_v1
  python build_document_holdout.py build --out-dir D:/local-results/r1-document-holdout
  python build_document_holdout.py verify --out-dir D:/local-results/r1-document-holdout
  python build_document_holdout.py selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SCHEMA = "r1_document_holdout_v1"
CATEGORIES = ("code", "technical_general", "prose")

# This named config is intentionally small enough for a preflight.  A later brief
# can freeze its literal name and values in a gate protocol.
PRESETS = {
    "r1_preflight_v1": {
        "seed": 20260916,
        "span_bytes": 8192,
        "min_source_bytes": 4096,
        "calib": {"code": 16, "technical_general": 16, "prose": 16},
        "heldout": {"code": 32, "technical_general": 32, "prose": 32},
    },
}


class InvariantError(RuntimeError):
    """Raised when a manifest would not be safe for the document-level gate."""


@dataclass(frozen=True)
class DocumentRef:
    category: str
    source_kind: str  # ``file`` or ``pg19``
    source_document_id: str
    source_path: str  # repository-relative POSIX path
    source_content_sha256: str
    source_byte_count: int
    publisher_split: str | None = None
    row_index: int | None = None


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def _strict_text(raw: bytes, source: Path) -> str | None:
    """Return only source documents whose stored bytes are valid UTF-8."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not text or text.encode("utf-8") != raw:
        return None
    return text


def _tree_refs(root: Path, suffixes: tuple[str, ...], category: str,
               counters: Counter[str]) -> Iterable[DocumentRef]:
    if not root.is_dir():
        raise FileNotFoundError(f"source directory is missing: {root}")
    paths = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in suffixes)
    for path in paths:
        counters[f"{category}.seen"] += 1
        try:
            raw = path.read_bytes()
        except OSError:
            counters[f"{category}.unreadable"] += 1
            continue
        text = _strict_text(raw, path)
        if text is None:
            counters[f"{category}.non_utf8_or_empty"] += 1
            continue
        rel = _repo_path(path)
        counters[f"{category}.utf8_documents"] += 1
        yield DocumentRef(
            category=category,
            source_kind="file",
            source_document_id=f"file:{rel}",
            source_path=rel,
            source_content_sha256=_sha(raw),
            source_byte_count=len(raw),
        )


def _pg19_refs(split: str, counters: Counter[str]) -> Iterable[DocumentRef]:
    """Yield PG19 references without retaining book text in memory.

    ``row_index`` is the physical row number within the parquet file and is part of
    the stable source-document ID.  We intentionally do not invent a publisher ID
    beyond that local, reproducible identifier.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment error path
        raise RuntimeError("pyarrow is required to read local PG19 parquet files") from exc

    data_dir = REPO / "data" / "external" / "pg19" / "data"
    paths = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no PG19 {split}-*.parquet files under {data_dir}")
    for path in paths:
        rel = _repo_path(path)
        parquet = pq.ParquetFile(path)
        row_index = 0
        for batch in parquet.iter_batches(batch_size=64, columns=["text"]):
            for value in batch.column("text").to_pylist():
                this_row = row_index
                row_index += 1
                counters[f"prose.{split}.seen"] += 1
                if not isinstance(value, str) or not value:
                    counters[f"prose.{split}.empty_or_nonstring"] += 1
                    continue
                raw = value.encode("utf-8")
                if raw.decode("utf-8").encode("utf-8") != raw:
                    # Defensive; valid Arrow strings should always satisfy this.
                    counters[f"prose.{split}.non_roundtrip"] += 1
                    continue
                counters[f"prose.{split}.utf8_documents"] += 1
                yield DocumentRef(
                    category="prose",
                    source_kind="pg19",
                    source_document_id=f"pg19:{rel}:row={this_row}",
                    source_path=rel,
                    source_content_sha256=_sha(raw),
                    source_byte_count=len(raw),
                    publisher_split=split,
                    row_index=this_row,
                )


def discover_documents() -> tuple[list[DocumentRef], Counter[str]]:
    """Hash local source documents once; no raw corpus is generated or written."""
    counters: Counter[str] = Counter()
    refs = list(_tree_refs(REPO / "data" / "external" / "the_stack_python",
                           (".py",), "code", counters))
    refs.extend(_tree_refs(REPO / "data" / "external" / "markdown_corpus",
                           (".md", ".markdown", ".mdx"), "technical_general", counters))
    refs.extend(_pg19_refs("train", counters))
    refs.extend(_pg19_refs("test", counters))
    return refs, counters


def estimate_source_scan_cost() -> dict:
    """Return a metadata-only cost estimate; it never reads document text.

    An exact global dedup audit cannot safely stop once quotas are reached: an
    unseen later document can be byte-identical to a selected document or rank ahead
    of it.  ``--full-content`` consequently scans every registered source document.
    This estimate-only preflight is deliberately not a gate-readiness assertion.
    """
    def tree_estimate(root: Path, suffixes: tuple[str, ...]) -> dict:
        paths = [path for path in root.rglob("*")
                 if path.is_file() and path.suffix.lower() in suffixes]
        return {"files": len(paths), "filesystem_bytes": sum(path.stat().st_size for path in paths)}

    result = {
        "code": tree_estimate(REPO / "data" / "external" / "the_stack_python", (".py",)),
        "technical_general": tree_estimate(REPO / "data" / "external" / "markdown_corpus",
                                            (".md", ".markdown", ".mdx")),
        "prose": {},
        "integrity_note": (
            "Exact global deduplication and deterministic top-rank selection require a full "
            "content scan. Stopping after a quota is reached is not integrity-preserving."
        ),
    }
    data_dir = REPO / "data" / "external" / "pg19" / "data"
    for split in ("train", "test"):
        paths = sorted(data_dir.glob(f"{split}-*.parquet"))
        result["prose"][split] = {
            "parquet_files": len(paths),
            "filesystem_bytes": sum(path.stat().st_size for path in paths),
            "row_count": None,
        }
        try:
            import pyarrow.parquet as pq
            result["prose"][split]["row_count"] = sum(
                pq.ParquetFile(path).metadata.num_rows for path in paths)
        except ImportError:
            pass
    return result


def _deduplicate_globally(refs: Iterable[DocumentRef]) -> tuple[list[DocumentRef], dict[str, int]]:
    """Keep one deterministic owner for every exact UTF-8 content SHA globally."""
    owners: dict[str, DocumentRef] = {}
    duplicate_groups: set[str] = set()
    duplicates_removed = 0
    for ref in refs:
        old = owners.get(ref.source_content_sha256)
        if old is None:
            owners[ref.source_content_sha256] = ref
            continue
        duplicate_groups.add(ref.source_content_sha256)
        duplicates_removed += 1
        if (ref.source_document_id, ref.source_path) < (old.source_document_id, old.source_path):
            owners[ref.source_content_sha256] = ref
    return sorted(owners.values(), key=lambda item: item.source_document_id), {
        "unique_documents": len(owners),
        "duplicate_content_groups": len(duplicate_groups),
        "duplicates_removed": duplicates_removed,
    }


def _eligible_refs(refs: Iterable[DocumentRef], min_source_bytes: int) -> tuple[list[DocumentRef], dict]:
    """Reject short source documents before deduplication, split, or selection."""
    eligible: list[DocumentRef] = []
    rejected: Counter[str] = Counter()
    for ref in refs:
        if ref.source_byte_count < min_source_bytes:
            rejected[ref.category] += 1
        else:
            eligible.append(ref)
    return eligible, {
        "min_source_bytes": min_source_bytes,
        "rejected_short_source_documents": dict(sorted(rejected.items())),
        "eligible_documents_before_exact_dedup": len(eligible),
    }


def _content_split(content_sha256: str) -> str:
    """Stable half split derived solely from complete document content."""
    return "calib" if int(content_sha256[:16], 16) % 2 == 0 else "heldout"


def _selection_key(ref: DocumentRef, split: str, seed: int) -> str:
    return _sha(f"{seed}:{split}:{ref.category}:{ref.source_content_sha256}".encode("ascii"))


def plan_documents(refs: Iterable[DocumentRef], config: dict) -> tuple[dict[str, list[DocumentRef]], dict]:
    """Deduplicate, assign document splits, and select bounded item counts.

    Code and markdown assignment occurs from the full content hash before any text
    span is extracted.  PG19 is routed only by its publisher file label; all later
    checks remain content- and ID-based rather than trusting that label.
    """
    eligible, eligibility = _eligible_refs(refs, config["min_source_bytes"])
    unique, dedup = _deduplicate_globally(eligible)
    pools: dict[str, dict[str, list[DocumentRef]]] = {
        split: {category: [] for category in CATEGORIES} for split in ("calib", "heldout")
    }
    for ref in unique:
        if ref.category in ("code", "technical_general"):
            split = _content_split(ref.source_content_sha256)
        elif ref.category == "prose":
            split = "calib" if ref.publisher_split == "train" else "heldout"
        else:  # pragma: no cover - protects future source additions
            raise InvariantError(f"unknown source category: {ref.category}")
        pools[split][ref.category].append(ref)

    chosen: dict[str, list[DocumentRef]] = {"calib": [], "heldout": []}
    availability: dict[str, dict[str, int]] = {"calib": {}, "heldout": {}}
    for split in ("calib", "heldout"):
        for category in CATEGORIES:
            candidates = sorted(pools[split][category],
                                key=lambda ref: (_selection_key(ref, split, config["seed"]),
                                                 ref.source_document_id))
            need = config[split][category]
            availability[split][category] = len(candidates)
            if len(candidates) < need:
                raise InvariantError(
                    f"insufficient {split}/{category} documents after exact global dedup: "
                    f"need {need}, found {len(candidates)}")
            chosen[split].extend(candidates[:need])
        chosen[split].sort(key=lambda ref: (ref.category,
                                            _selection_key(ref, split, config["seed"]),
                                            ref.source_document_id))
    return chosen, {"eligibility": eligibility, "deduplication": dedup, "availability": availability}


def _hydrate_selected(refs: Iterable[DocumentRef]) -> tuple[dict[str, str], dict[str, str]]:
    """Load only selected texts for JSONL; source scans above never retain a corpus."""
    selected = list(refs)
    texts: dict[str, str] = {}
    source_file_hashes: dict[str, str] = {}
    pg_by_path: dict[str, list[DocumentRef]] = defaultdict(list)
    for ref in selected:
        source = REPO / ref.source_path
        if ref.source_kind == "file":
            raw = source.read_bytes()
            text = _strict_text(raw, source)
            if text is None or _sha(raw) != ref.source_content_sha256:
                raise InvariantError(f"source changed after discovery: {ref.source_path}")
            texts[ref.source_document_id] = text
            source_file_hashes[ref.source_path] = _sha(raw)
        else:
            pg_by_path[ref.source_path].append(ref)

    if pg_by_path:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - environment error path
            raise RuntimeError("pyarrow is required to hydrate local PG19 rows") from exc
        for rel, wanted_refs in sorted(pg_by_path.items()):
            wanted = {ref.row_index: ref for ref in wanted_refs}
            path = REPO / rel
            source_file_hashes[rel] = _sha_file(path)
            row_index = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["text"]):
                for value in batch.column("text").to_pylist():
                    ref = wanted.get(row_index)
                    if ref is not None:
                        if not isinstance(value, str) or _sha(value.encode("utf-8")) != ref.source_content_sha256:
                            raise InvariantError(f"PG19 row changed after discovery: {ref.source_document_id}")
                        texts[ref.source_document_id] = value
                    row_index += 1
            missing = {ref.source_document_id for ref in wanted_refs} - texts.keys()
            if missing:
                raise InvariantError(f"PG19 rows disappeared after discovery: {sorted(missing)}")
    return texts, source_file_hashes


def _bounded_span(text: str, content_sha256: str, seed: int, span_bytes: int) -> tuple[str, int]:
    """Return a deterministic contiguous span without cutting an UTF-8 code point."""
    raw = text.encode("utf-8")
    if not raw:
        raise InvariantError("cannot score an empty document")
    if len(raw) <= span_bytes:
        span = raw
        start = 0
    else:
        raw_start = int(_sha(f"{seed}:{content_sha256}".encode("ascii"))[:16], 16)
        start = raw_start % (len(raw) - span_bytes + 1)
        while start < len(raw) and raw[start] & 0xC0 == 0x80:
            start += 1
        end = min(len(raw), start + span_bytes)
        while end > start and end < len(raw) and raw[end] & 0xC0 == 0x80:
            end -= 1
        span = raw[start:end]
    try:
        selected = span.decode("utf-8")
    except UnicodeDecodeError as exc:  # pragma: no cover - guarded by boundary adjustment
        raise InvariantError("span boundary was not UTF-8 valid") from exc
    if not selected or selected.encode("utf-8") != span:
        raise InvariantError("span is empty or fails UTF-8 roundtrip")
    return selected, start


def _item(ref: DocumentRef, split: str, text: str, source_file_sha256: str, config: dict) -> dict:
    if ref.source_byte_count < config["min_source_bytes"]:
        raise InvariantError(f"short source document reached extraction: {ref.source_document_id}")
    span, start = _bounded_span(text, ref.source_content_sha256, config["seed"], config["span_bytes"])
    raw_span = span.encode("utf-8")
    item = {
        "schema": SCHEMA,
        "split": split,
        "category": ref.category,
        "source_document_id": ref.source_document_id,
        "source_path": ref.source_path,
        "source_kind": ref.source_kind,
        "publisher_split": ref.publisher_split,
        "source_file_sha256": source_file_sha256,
        "source_content_sha256": ref.source_content_sha256,
        "source_byte_count": ref.source_byte_count,
        "span_start_byte": start,
        "span_byte_count": len(raw_span),
        "text": span,
    }
    item["item_sha256"] = _sha(_canonical(item))
    return item


def _assert_cross_split_disjoint(items_by_split: dict[str, list[dict]]) -> None:
    """Assert the two required leakage barriers independently."""
    ids = {split: {item["source_document_id"] for item in items}
           for split, items in items_by_split.items()}
    contents = {split: {item["source_content_sha256"] for item in items}
                for split, items in items_by_split.items()}
    overlap_ids = ids["calib"] & ids["heldout"]
    overlap_contents = contents["calib"] & contents["heldout"]
    if overlap_ids:
        raise InvariantError(f"source-document leakage across splits: {sorted(overlap_ids)}")
    if overlap_contents:
        raise InvariantError(f"exact-content leakage across splits: {sorted(overlap_contents)}")


def _validate_items(items_by_split: dict[str, list[dict]], config: dict) -> None:
    _assert_cross_split_disjoint(items_by_split)
    for split, items in items_by_split.items():
        category_counts = Counter(item["category"] for item in items)
        if dict(category_counts) != config[split]:
            raise InvariantError(f"wrong {split} category counts: {dict(category_counts)}")
        for item in items:
            raw = item["text"].encode("utf-8")
            if not raw or raw.decode("utf-8") != item["text"]:
                raise InvariantError(f"empty or invalid UTF-8 span: {item['source_document_id']}")
            if item["source_byte_count"] < config["min_source_bytes"]:
                raise InvariantError(f"short source document was emitted: {item['source_document_id']}")
            if len(raw) != item["span_byte_count"] or len(raw) > config["span_bytes"]:
                raise InvariantError(f"incorrect span byte count: {item['source_document_id']}")
            unsigned = dict(item)
            observed = unsigned.pop("item_sha256", None)
            if observed != _sha(_canonical(unsigned)):
                raise InvariantError(f"item hash mismatch: {item['source_document_id']}")


def _jsonl_bytes(items: list[dict]) -> bytes:
    return b"".join(_canonical(item) + b"\n" for item in items)


def _aggregate_hash(items: list[dict]) -> str:
    return _sha("".join(item["item_sha256"] + "\n" for item in items).encode("ascii"))


def _manifest(items_by_split: dict[str, list[dict]], config_name: str, config: dict,
              planning: dict, source_counters: Counter[str]) -> dict:
    source_files = sorted({(item["source_path"], item["source_file_sha256"])
                           for items in items_by_split.values() for item in items})
    parts = {}
    for split, items in items_by_split.items():
        raw = _jsonl_bytes(items)
        parts[split] = {
            "path": f"{split}.jsonl",
            "item_count": len(items),
            "category_counts": dict(sorted(Counter(item["category"] for item in items).items())),
            "total_span_bytes": sum(item["span_byte_count"] for item in items),
            "jsonl_sha256": _sha(raw),
            "items_aggregate_sha256": _aggregate_hash(items),
        }
    return {
        "schema": SCHEMA,
        "config_name": config_name,
        "config": config,
        "source_provenance": {
            "code": "data/external/the_stack_python/**/*.py",
            "technical_general": "data/external/markdown_corpus/**/*.{md,markdown,mdx}",
            "prose_calib": "data/external/pg19/data/train-*.parquet",
            "prose_heldout": "data/external/pg19/data/test-*.parquet",
            "license_or_redistribution_rights": "not asserted by this tool; review source provenance separately",
        },
        "split_logic": {
            "deduplication": "exact UTF-8 content SHA-256 globally; deterministic smallest source_document_id owner",
            "code_and_technical_general": "int(first_16_hex(content_sha256), 16) % 2 == 0 -> calib, else heldout; before span extraction",
            "prose": "local PG19 publisher filename train -> calib and test -> heldout; post-build ID/content checks required",
            "span": "one contiguous UTF-8-valid span per document; start derives from SHA-256(seed:content_sha256)",
            "limitation": "exact deduplication does not detect near duplicates or semantic overlap",
        },
        "source_discovery_counts": dict(sorted(source_counters.items())),
        "source_scan_policy": "full-content scan; required for exact global deduplication",
        "planning": planning,
        "source_files": [{"path": path, "sha256": digest} for path, digest in source_files],
        "parts": parts,
    }


def _write_output(out_dir: Path, items_by_split: dict[str, list[dict]], manifest: dict) -> None:
    """Create a new directory only; any existing destination is an error."""
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    for split, items in items_by_split.items():
        with (out_dir / f"{split}.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for item in items:
                handle.write(_canonical(item).decode("utf-8") + "\n")
    with (out_dir / "manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _resolve_config(args: argparse.Namespace) -> tuple[str, dict]:
    config_name = args.config
    config = json.loads(json.dumps(PRESETS[config_name]))
    for split in ("calib", "heldout"):
        for category in CATEGORIES:
            value = getattr(args, f"{split}_{category}")
            if value is not None:
                if value < 1:
                    raise ValueError(f"--{split}-{category.replace('_', '-')} must be positive")
                config[split][category] = value
    if args.span_bytes is not None:
        if args.span_bytes < 1:
            raise ValueError("--span-bytes must be positive")
        config["span_bytes"] = args.span_bytes
    if args.min_source_bytes is not None:
        if args.min_source_bytes < 1:
            raise ValueError("--min-source-bytes must be positive")
        config["min_source_bytes"] = args.min_source_bytes
    if args.seed is not None:
        config["seed"] = args.seed
    return config_name, config


def build(args: argparse.Namespace) -> None:
    config_name, config = _resolve_config(args)
    out_dir = Path(args.out_dir).expanduser()
    print("WARNING: JSONL contains raw source text; use an ignored/local destination and do not commit it.", file=sys.stderr)
    refs, counters = discover_documents()
    chosen, planning = plan_documents(refs, config)
    all_chosen = chosen["calib"] + chosen["heldout"]
    texts, source_hashes = _hydrate_selected(all_chosen)
    items = {
        split: [_item(ref, split, texts[ref.source_document_id], source_hashes[ref.source_path], config)
                for ref in chosen[split]]
        for split in ("calib", "heldout")
    }
    _validate_items(items, config)
    manifest = _manifest(items, config_name, config, planning, counters)
    _write_output(out_dir, items, manifest)
    print(f"wrote {out_dir}")
    for split in ("calib", "heldout"):
        part = manifest["parts"][split]
        print(f"  {split}: {part['item_count']} items, {part['total_span_bytes']} span bytes, "
              f"sha256={part['jsonl_sha256']}")


def verify(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser()
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise InvariantError(f"unexpected manifest schema: {manifest.get('schema')!r}")
    config = manifest["config"]
    items: dict[str, list[dict]] = {}
    for split in ("calib", "heldout"):
        part = manifest["parts"][split]
        raw = (out_dir / part["path"]).read_bytes()
        if _sha(raw) != part["jsonl_sha256"]:
            raise InvariantError(f"JSONL hash mismatch: {split}")
        items[split] = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
        if len(items[split]) != part["item_count"]:
            raise InvariantError(f"item count mismatch: {split}")
        if _aggregate_hash(items[split]) != part["items_aggregate_sha256"]:
            raise InvariantError(f"aggregate item hash mismatch: {split}")
    _validate_items(items, config)
    if args.check_source_files:
        for source in manifest["source_files"]:
            actual = _sha_file(REPO / source["path"])
            if actual != source["sha256"]:
                raise InvariantError(f"source file hash mismatch: {source['path']}")
    print(f"verified {out_dir}")


def _toy_ref(category: str, identifier: str, wanted_split: str | None = None,
             publisher_split: str | None = None) -> tuple[DocumentRef, str]:
    """Create a deterministic in-memory fixture; it never touches local source pools."""
    nonce = 0
    while True:
        text = f"{category} fixture {identifier} {nonce} — UTF-8 ✅\n"
        digest = _sha(text.encode("utf-8"))
        if wanted_split is None or _content_split(digest) == wanted_split:
            return DocumentRef(category, "file", f"toy:{identifier}", f"toy/{identifier}",
                               digest, len(text.encode("utf-8")), publisher_split=publisher_split), text
        nonce += 1


def selftest(_args: argparse.Namespace) -> None:
    """Pure toy self-test: no local corpora, models, weights, or network access."""
    config = {
        "seed": 17, "span_bytes": 19, "min_source_bytes": 20,
        "calib": {"code": 1, "technical_general": 1, "prose": 1},
        "heldout": {"code": 1, "technical_general": 1, "prose": 1},
    }
    refs: list[DocumentRef] = []
    texts: dict[str, str] = {}
    for category in ("code", "technical_general"):
        for split in ("calib", "heldout"):
            ref, text = _toy_ref(category, f"{category}-{split}", split)
            refs.append(ref); texts[ref.source_document_id] = text
    for split in ("train", "test"):
        ref, text = _toy_ref("prose", f"pg19-{split}", publisher_split=split)
        refs.append(DocumentRef("prose", "pg19", ref.source_document_id, ref.source_path,
                                ref.source_content_sha256, ref.source_byte_count, split, 0))
        texts[ref.source_document_id] = text
    # A train/test duplicate must be collapsed globally rather than crossing splits.
    duplicate_text = "exact duplicate across publisher file labels\n"
    duplicate_sha = _sha(duplicate_text.encode("utf-8"))
    for split in ("train", "test"):
        ref = DocumentRef("prose", "pg19", f"toy:duplicate-{split}", f"toy/{split}",
                          duplicate_sha, len(duplicate_text.encode("utf-8")), split, 1)
        refs.append(ref); texts[ref.source_document_id] = duplicate_text
    short_text = "tiny\n"
    refs.append(DocumentRef("code", "file", "toy:short", "toy/short", _sha(short_text.encode("utf-8")),
                            len(short_text.encode("utf-8"))))
    texts["toy:short"] = short_text

    chosen, planning = plan_documents(refs, config)
    assert planning["deduplication"]["duplicates_removed"] == 1
    assert planning["eligibility"]["rejected_short_source_documents"] == {"code": 1}
    assert all(ref.source_document_id != "toy:short" for refs_ in chosen.values() for ref in refs_)
    source_hashes = {ref.source_path: _sha(ref.source_path.encode("utf-8"))
                     for refs_ in chosen.values() for ref in refs_}
    items = {
        split: [_item(ref, split, texts[ref.source_document_id], source_hashes[ref.source_path], config)
                for ref in refs_]
        for split, refs_ in chosen.items()
    }
    _validate_items(items, config)

    # Planted controls must FIRE: the first represents old chunk-level leakage,
    # the second represents exact duplicate content with different document IDs.
    planted_chunk_leak = {"calib": [{"source_document_id": "toy:long-doc", "source_content_sha256": "a"}],
                          "heldout": [{"source_document_id": "toy:long-doc", "source_content_sha256": "b"}]}
    planted_duplicate = {"calib": [{"source_document_id": "toy:a", "source_content_sha256": "same"}],
                         "heldout": [{"source_document_id": "toy:b", "source_content_sha256": "same"}]}
    for planted in (planted_chunk_leak, planted_duplicate):
        try:
            _assert_cross_split_disjoint(planted)
        except InvariantError:
            pass
        else:  # pragma: no cover - the selftest must fail if either guard is removed
            raise AssertionError("planted leakage control did not fire")

    with tempfile.TemporaryDirectory(prefix="r1-document-holdout-selftest-") as temp:
        out_dir = Path(temp) / "toy-output"
        manifest = _manifest(items, "selftest", config, planning, Counter({"toy": len(refs)}))
        _write_output(out_dir, items, manifest)
        verify(argparse.Namespace(out_dir=str(out_dir), check_source_files=False))
        try:
            _write_output(out_dir, items, manifest)
        except FileExistsError:
            pass
        else:  # pragma: no cover - safety control
            raise AssertionError("write-once control did not fire")
    print("selftest: OK (toy-only; chunk leakage and duplicate crossover controls fired)")


def _add_config_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", choices=sorted(PRESETS), default="r1_preflight_v1")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--span-bytes", type=int)
    parser.add_argument("--min-source-bytes", type=int,
                        help="reject shorter source documents before split/selection")
    for split in ("calib", "heldout"):
        for category in CATEGORIES:
            parser.add_argument(f"--{split}-{category.replace('_', '-')}", type=int,
                                dest=f"{split}_{category}",
                                help=f"override {split} item count for {category}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight_parser = commands.add_parser("preflight", help="estimate source cost by default; writes nothing")
    _add_config_options(preflight_parser)
    preflight_parser.add_argument("--full-content", action="store_true",
                                  help="perform exact dedup/split availability audit over all source documents")
    build_parser = commands.add_parser("build", help="write new raw-text JSONL and manifest")
    _add_config_options(build_parser)
    build_parser.add_argument("--out-dir", required=True,
                              help="new ignored/local directory; raw JSONL must not be committed")
    verify_parser = commands.add_parser("verify", help="verify manifest, JSONL, and leakage barriers")
    verify_parser.add_argument("--out-dir", required=True)
    verify_parser.add_argument("--check-source-files", action="store_true",
                               help="also rehash selected local source files")
    commands.add_parser("selftest", help="run pure toy tests; no source-pool scan")
    return parser.parse_args(argv)


def preflight(args: argparse.Namespace) -> None:
    config_name, config = _resolve_config(args)
    estimate = estimate_source_scan_cost()
    if not args.full_content:
        print(f"preflight ESTIMATE ({config_name}); no document text was read and no output was written")
        print(json.dumps({"source_scan_estimate": estimate, "config": config,
                          "not_gate_ready": "Run with --full-content for exact dedup/split eligibility."},
                         indent=2, sort_keys=True))
        return
    refs, counters = discover_documents()
    try:
        _chosen, planning = plan_documents(refs, config)
    except InvariantError as exc:
        print(f"preflight FAILED ({config_name}): {exc}")
        print(json.dumps(dict(sorted(counters.items())), indent=2, sort_keys=True))
        raise
    print(f"preflight OK ({config_name}); full content was scanned, no output was written")
    print(json.dumps({"source_scan_estimate": estimate, "source_discovery_counts": dict(sorted(counters.items())),
                      "planning": planning, "config": config}, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        if args.command == "preflight":
            preflight(args)
        elif args.command == "build":
            build(args)
        elif args.command == "verify":
            verify(args)
        else:
            selftest(args)
    except (FileNotFoundError, FileExistsError, InvariantError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
