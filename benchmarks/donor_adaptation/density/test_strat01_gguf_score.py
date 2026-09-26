#!/usr/bin/env python3
"""No-model tests plus build/run wrapper for ``strat01_gguf_score.cpp``.

The scoring entry point is deliberately explicit:

  python test_strat01_gguf_score.py --build
  python test_strat01_gguf_score.py --score --binary <scorer.exe> --model <model.gguf> \\
      --corpus <heldout.jsonl> --tokenizer-json <tokenizer.json> --output <scores.jsonl>

``--score`` first writes a short-lived base64 exchange file, invokes the C++
core without exposing document text in stdout/stderr, verifies the returned
DocumentScore records, and persists a sibling provenance manifest.  It does
not score at import time and the unit tests do not open a model.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


HERE = Path(__file__).resolve().parent
CORE = HERE / "strat01_gguf_score.cpp"
PINNED_LLAMA = Path(r"C:\Users\giosa\AppData\Local\Temp\siliconllm-llama-mtp-5b335f4")
MAX_CONTEXT = 4096
MAX_DOCUMENTS = 256
MAX_DOCUMENT_BYTES = 1 << 20
SCORE_FIELDS = ("source_document_id", "category", "tokens", "bytes", "bits")
CALIB_SHA256 = "f1ed84f64284d2cd6ffd59f2373849f41a3c33bdb327eaa75f0f2b7e0e3d998f"
CATEGORIES = ("code", "technical_general", "prose")
TOKENIZER_SHA256 = "b4b3d90c67830a4e566296ad8d0f6b5ac5a5cdbfd331b200d0ee8263aaaea1fe"
MODEL_SHA256 = {
    "GigaChat3.1-10B-A1.8B-bf16.gguf": "e7a6409be0ac197babf21c48cfc8a96486d035c1e7a784feadd817b7b883c08e",
    "GigaChat3.1-10B-A1.8B-q4_K_M.gguf": "68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb",
}


class ScorerError(RuntimeError):
    """A deterministic preflight or runtime violation; document text is omitted."""


@dataclass(frozen=True)
class PreparedDocument:
    source_document_id: str
    category: str
    text: str
    raw_bytes: bytes
    ids: tuple[int, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _require_new(path: Path, label: str) -> None:
    if path.exists():
        raise ScorerError(f"{label} already exists; refusing to overwrite it")


def _read_jsonl(corpus: Path) -> list[dict[str, Any]]:
    try:
        lines = corpus.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ScorerError("cannot read UTF-8 corpus JSONL") from exc
    if not lines:
        raise ScorerError("corpus has no JSONL rows")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, 1):
        if not line:
            raise ScorerError(f"corpus has a blank line at {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScorerError(f"corpus JSON is invalid at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ScorerError(f"corpus row {line_number} is not an object")
        identifier, category, text = row.get("source_document_id"), row.get("category"), row.get("text")
        if not isinstance(identifier, str) or not identifier or not isinstance(category, str) or not category or not isinstance(text, str):
            raise ScorerError(f"corpus row {line_number} lacks a valid identity/category/text")
        if identifier in seen:
            raise ScorerError(f"corpus has duplicate source_document_id at line {line_number}")
        seen.add(identifier)
        rows.append(row)
    if len(rows) > MAX_DOCUMENTS:
        raise ScorerError("corpus exceeds the scorer's bounded document count")
    return rows


def _selected_rows(rows: Sequence[dict[str, Any]], per_category: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for category in CATEGORIES:
        candidates = sorted(
            (row for row in rows if row["category"] == category),
            key=lambda row: row["source_document_id"],
        )
        if len(candidates) < per_category:
            raise ScorerError(f"calibration corpus has fewer than {per_category} documents in {category}")
        selected.extend(candidates[:per_category])
    return selected


def select_calibration(corpus: Path, output: Path, stage: str) -> None:
    if _sha256(corpus) != CALIB_SHA256:
        raise ScorerError("calibration source SHA-256 differs from the frozen corpus")
    per_category = {"instrument": 1, "pilot": 3}[stage]
    rows = _selected_rows(_read_jsonl(corpus), per_category)
    manifest = output.with_suffix(output.suffix + ".selection.json")
    _require_new(output, "selected corpus")
    _require_new(manifest, "selection manifest")
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    record = {
        "schema": "strat01_gguf_calib_selection_v1",
        "stage": stage,
        "source": {"path": str(corpus), "sha256": CALIB_SHA256},
        "selected": {"path": str(output), "sha256": _sha256(output),
                     "source_document_ids": [row["source_document_id"] for row in rows],
                     "documents_per_category": per_category},
    }
    with manifest.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _special_ids(tokenizer: Any) -> set[int]:
    decoder = getattr(tokenizer, "get_added_tokens_decoder", lambda: {})()
    return {int(token_id) for token_id, token in decoder.items() if bool(getattr(token, "special", False))}


def _compare_vocab_mapping(source: dict[str, Any], gguf_tokens: Sequence[str], *,
                           base_count: int = 128_000, added_count: int = 15,
                           model_vocab_size: int = 128_256) -> dict[str, Any]:
    vocab = source.get("model", {}).get("vocab")
    added = source.get("added_tokens")
    if not isinstance(vocab, dict) or len(vocab) != base_count or not isinstance(added, list) or len(added) != added_count:
        raise ScorerError("source tokenizer has an unexpected vocabulary shape")
    if len(gguf_tokens) != model_vocab_size or not all(isinstance(token, str) for token in gguf_tokens):
        raise ScorerError("GGUF tokenizer has an unexpected vocabulary shape")
    by_id: dict[int, str] = {}
    for token, identifier in vocab.items():
        if not isinstance(token, str) or type(identifier) is not int or identifier in by_id:
            raise ScorerError("source tokenizer has invalid or duplicate base IDs")
        by_id[identifier] = token
    if set(by_id) != set(range(base_count)):
        raise ScorerError("source tokenizer base IDs are not contiguous")
    for entry in added:
        if not isinstance(entry, dict) or type(entry.get("id")) is not int or not isinstance(entry.get("content"), str):
            raise ScorerError("source tokenizer has an invalid added token")
        identifier, token = entry["id"], entry["content"]
        if identifier in by_id and by_id[identifier] != token:
            raise ScorerError("source tokenizer added token disagrees with its base ID")
        by_id[identifier] = token
    for identifier, token in by_id.items():
        if identifier < 0 or identifier >= len(gguf_tokens) or gguf_tokens[identifier] != token:
            raise ScorerError(f"source/GGUF vocabulary mapping differs at ID {identifier}")
    canonical = json.dumps(sorted(by_id.items()), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {"base_entries": len(vocab), "added_entries": len(added), "gguf_slots": len(gguf_tokens),
            "verified_distinct_ids": len(by_id), "id_to_token_sha256": hashlib.sha256(canonical).hexdigest(),
            "all_defined_ids_match": True}


def _verify_vocab_mapping(tokenizer_json: Path, model: Path) -> dict[str, Any]:
    if _sha256(tokenizer_json) != TOKENIZER_SHA256:
        raise ScorerError("source tokenizer SHA-256 differs from the pinned GigaChat asset")
    try:
        source = json.loads(tokenizer_json.read_text(encoding="utf-8"))
        gguf_package = PINNED_LLAMA / "gguf-py"
        if str(gguf_package) not in sys.path:
            sys.path.insert(0, str(gguf_package))
        from gguf import GGUFReader
        reader = GGUFReader(str(model))
        field = reader.get_field("tokenizer.ggml.tokens")
        if field is None:
            raise ScorerError("GGUF has no tokenizer token list")
        tokens = field.contents()
        return _compare_vocab_mapping(source, tokens)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ImportError, ValueError, IndexError) as exc:
        raise ScorerError("cannot verify full source/GGUF vocabulary mapping") from exc


def _prepare_documents(rows: Iterable[dict[str, Any]], tokenizer: Any, n_ctx: int) -> list[PreparedDocument]:
    special_ids = _special_ids(tokenizer)
    prepared: list[PreparedDocument] = []
    for index, row in enumerate(rows, 1):
        identifier, category, text = row["source_document_id"], row["category"], row["text"]
        try:
            raw_bytes = text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ScorerError(f"corpus row {index} text is not valid UTF-8") from exc
        if not raw_bytes or len(raw_bytes) > MAX_DOCUMENT_BYTES:
            raise ScorerError(f"corpus row {index} violates the bounded raw-byte contract")
        encoded = tokenizer.encode(text, add_special_tokens=False)
        ids = getattr(encoded, "ids", None)
        if not isinstance(ids, list) or not ids or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in ids):
            raise ScorerError(f"tokenizer emitted invalid payload IDs at corpus row {index}")
        if special_ids.intersection(ids):
            raise ScorerError(f"payload contains a tokenizer special ID at corpus row {index}")
        try:
            roundtrip = tokenizer.decode(ids, skip_special_tokens=False)
        except Exception as exc:  # tokenizers raises type-specific exceptions across versions.
            raise ScorerError(f"tokenizer cannot decode corpus row {index}") from exc
        if roundtrip != text:
            raise ScorerError(f"Python tokenizer roundtrip differs at corpus row {index}")
        if len(ids) + 1 > n_ctx:
            raise ScorerError(f"corpus row {index} exceeds the no-truncation context bound")
        prepared.append(PreparedDocument(identifier, category, text, raw_bytes, tuple(ids)))
    return prepared


def _write_private_exchange(path: Path, documents: Sequence[PreparedDocument]) -> None:
    _require_new(path, "private exchange file")
    with path.open("x", encoding="ascii", newline="\n") as handle:
        for document in documents:
            fields = (
                _b64(document.source_document_id.encode("utf-8")),
                _b64(document.category.encode("utf-8")),
                str(len(document.raw_bytes)),
                _b64(document.raw_bytes),
                ",".join(str(token) for token in document.ids),
            )
            handle.write("\t".join(fields) + "\n")


def _read_score_jsonl(path: Path, documents: Sequence[PreparedDocument]) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(documents):
        raise ScorerError("C++ scorer returned the wrong document count")
    expected = {document.source_document_id: document for document in documents}
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScorerError(f"C++ score output has invalid JSON at line {line_number}") from exc
        if not isinstance(record, dict) or tuple(record) != SCORE_FIELDS:
            raise ScorerError(f"C++ score output violates the DocumentScore schema at line {line_number}")
        identifier = record["source_document_id"]
        document = expected.pop(identifier, None)
        if document is None or record["category"] != document.category:
            raise ScorerError(f"C++ score identity/category mismatch at line {line_number}")
        if type(record["tokens"]) is not int or record["tokens"] != len(document.ids):
            raise ScorerError(f"C++ score token-count mismatch at line {line_number}")
        if type(record["bytes"]) is not int or record["bytes"] != len(document.raw_bytes):
            raise ScorerError(f"C++ score byte-count mismatch at line {line_number}")
        if type(record["bits"]) not in (int, float) or not math.isfinite(float(record["bits"])) or float(record["bits"]) < 0.0:
            raise ScorerError(f"C++ score bits are invalid at line {line_number}")
        result.append(record)
    if expected:
        raise ScorerError("C++ scorer omitted at least one source document")
    return result


def _read_runtime_metadata(path: Path, expected_documents: int) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScorerError("C++ scorer did not emit valid runtime metadata") from exc
    if not isinstance(metadata, dict) or metadata.get("schema") != "strat01_gguf_score_runtime_v1":
        raise ScorerError("C++ runtime metadata has the wrong schema")
    mismatches = metadata.get("runtime_tokenizer_mismatch_documents")
    if (metadata.get("documents") != expected_documents or metadata.get("source_token_ids_mode") is not True
            or type(mismatches) is not int or not 0 <= mismatches <= expected_documents):
        raise ScorerError("C++ runtime metadata does not confirm the source-ID scoring mode")
    if metadata.get("kv_reset_per_document") is not True or metadata.get("payload_eos_appended") is not False:
        raise ScorerError("C++ runtime metadata violates the per-document protocol")
    if (metadata.get("context_marker_policy") != "bos"
            or metadata.get("model_bos_id") != 1
            or metadata.get("context_marker_id") != 1
            or metadata.get("model_add_bos") is not True):
        raise ScorerError("GGUF runtime did not use GigaChat's pinned BOS=1 prefix")
    return metadata


def _git_head() -> str | None:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE.parents[2], text=True, capture_output=True, check=False)
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and len(value) == 40 else None


def _cmake_project(source: Path, llama_source: Path) -> str:
    return f'''cmake_minimum_required(VERSION 3.20)
project(strat01_gguf_score LANGUAGES C CXX)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
add_subdirectory("{llama_source.as_posix()}" llama-cpp)
add_executable(strat01_gguf_score "{source.as_posix()}")
target_compile_features(strat01_gguf_score PRIVATE cxx_std_17)
target_link_libraries(strat01_gguf_score PRIVATE llama)
'''


def build(binary_dir: Path, llama_source: Path, configuration: str) -> Path:
    if not CORE.is_file() or not (llama_source / "CMakeLists.txt").is_file():
        raise ScorerError("core source or pinned llama.cpp source is unavailable")
    binary_dir.mkdir(parents=True, exist_ok=True)
    cmake_lists = binary_dir / "CMakeLists.txt"
    cmake_lists.write_text(_cmake_project(CORE, llama_source), encoding="utf-8", newline="\n")
    configure = subprocess.run(["cmake", "-S", str(binary_dir), "-B", str(binary_dir / "cmake-build")], text=True, capture_output=True, check=False)
    if configure.returncode:
        raise ScorerError("CMake configure failed; inspect the build directory, not corpus data")
    compile_command = ["cmake", "--build", str(binary_dir / "cmake-build"), "--target", "strat01_gguf_score", "--config", configuration]
    compiled = subprocess.run(compile_command, text=True, capture_output=True, check=False)
    if compiled.returncode:
        raise ScorerError("CMake build failed; inspect the build directory, not corpus data")
    names = ("strat01_gguf_score.exe", "strat01_gguf_score")
    candidates = [path for name in names for path in (binary_dir / "cmake-build").rglob(name)]
    if len(candidates) != 1:
        raise ScorerError("CMake build did not produce one scorer binary")
    return candidates[0]


def score(args: argparse.Namespace) -> None:
    if args.n_ctx < 2 or args.n_ctx > MAX_CONTEXT or args.n_batch < 1 or args.n_batch > args.n_ctx:
        raise ScorerError("n-ctx/n-batch violate the bounded scorer contract")
    for path, label in ((args.binary, "scorer binary"), (args.model, "GGUF model"), (args.corpus, "corpus"), (args.tokenizer_json, "tokenizer JSON")):
        if not path.is_file():
            raise ScorerError(f"{label} is unavailable")
    expected_model_sha = MODEL_SHA256.get(args.model.name)
    explicit_sha = args.expected_model_sha256
    if explicit_sha is not None:
        explicit_sha = explicit_sha.lower()
        if len(explicit_sha) != 64 or any(ch not in "0123456789abcdef" for ch in explicit_sha):
            raise ScorerError("--expected-model-sha256 must be a 64-digit hex digest")
        if expected_model_sha is not None and explicit_sha != expected_model_sha:
            raise ScorerError("explicit digest conflicts with the pinned STRAT-01 arm")
        expected_model_sha = explicit_sha
    if expected_model_sha is None:
        raise ScorerError("GGUF name is not a pinned arm; supply --expected-model-sha256")
    model_stat = args.model.stat()
    model_sha = _sha256(args.model)
    if model_sha != expected_model_sha:
        raise ScorerError("GGUF SHA-256 differs from the pinned STRAT-01 arm")
    vocab_mapping = _verify_vocab_mapping(args.tokenizer_json, args.model)
    _require_new(args.output, "score output")
    manifest = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    _require_new(manifest, "score manifest")
    runtime_metadata = args.output.with_suffix(args.output.suffix + ".runtime.json")
    _require_new(runtime_metadata, "runtime metadata")
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ScorerError("--score requires the tokenizers package") from exc
    rows = _read_jsonl(args.corpus)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    documents = _prepare_documents(rows, tokenizer, args.n_ctx)
    exchange = args.output.with_suffix(args.output.suffix + ".private.tsv")
    _write_private_exchange(exchange, documents)
    try:
        command = [
            str(args.binary), "--model", str(args.model), "--input", str(exchange), "--output", str(args.output),
            "--metadata-out", str(runtime_metadata), "--n-ctx", str(args.n_ctx), "--n-batch", str(args.n_batch),
            "--context-marker", args.context_marker, "--source-token-ids",
        ]
        if args.threads is not None:
            command.extend(("--threads", str(args.threads)))
        completed = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, text=True, check=False)
        if completed.returncode:
            diagnostic = completed.stderr.strip()
            if not diagnostic.startswith("strat01_gguf_score: ") or "\n" in diagnostic:
                diagnostic = "unrecognized core diagnostic; inspect the apparatus before retrying"
            raise ScorerError(f"C++ scorer failed with exit status {completed.returncode}: {diagnostic}")
    finally:
        if not args.keep_exchange:
            exchange.unlink(missing_ok=True)
    records = _read_score_jsonl(args.output, documents)
    runtime = _read_runtime_metadata(runtime_metadata, len(documents))
    after_stat = args.model.stat()
    if (model_stat.st_size, model_stat.st_mtime_ns) != (after_stat.st_size, after_stat.st_mtime_ns):
        raise ScorerError("GGUF file changed during scoring")
    manifest_record = {
        "schema": "strat01_gguf_score_manifest_v1",
        "corpus": {"path": str(args.corpus), "sha256": _sha256(args.corpus), "documents": len(documents)},
        "tokenizer": {"path": str(args.tokenizer_json), "sha256": TOKENIZER_SHA256,
                      "source_gguf_vocab_mapping": vocab_mapping},
        "model": {"path": str(args.model), "sha256": model_sha},
        "scorer": {"binary": str(args.binary), "binary_sha256": _sha256(args.binary), "source": str(CORE), "source_sha256": _sha256(CORE)},
        "output": {"path": str(args.output), "sha256": _sha256(args.output), "records": len(records)},
        "runtime_metadata": {"path": str(runtime_metadata), "sha256": _sha256(runtime_metadata), "record": runtime},
        "protocol": {"python_tokenizer_add_special_tokens": False, "python_tokenizer_roundtrip": True,
                     "source_ids_used_for_model_input": True,
                     "llama_text_tokenizer_mismatch_is_diagnostic": True, "context_marker": args.context_marker,
                     "payload_eos_appended": False, "context_reset_per_document": True},
        "repository_head": _git_head(),
    }
    with manifest.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest_record, handle, sort_keys=True, indent=2)
        handle.write("\n")


class FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> FakeEncoding:
        assert add_special_tokens is False
        return FakeEncoding([ord(character) for character in text])

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        return "".join(chr(identifier) for identifier in ids)

    def get_added_tokens_decoder(self) -> dict[int, object]:
        return {}


class Strat01GgufScoreTests(unittest.TestCase):
    def test_full_vocab_mapping_check_rejects_one_wrong_id(self) -> None:
        source = {"model": {"vocab": {"a": 0, "b": 1}},
                  "added_tokens": [{"id": 2, "content": "<s>"}]}
        result = _compare_vocab_mapping(source, ["a", "b", "<s>"], base_count=2, added_count=1, model_vocab_size=3)
        self.assertEqual(result["verified_distinct_ids"], 3)
        with self.assertRaisesRegex(ScorerError, "differs at ID 1"):
            _compare_vocab_mapping(source, ["a", "wrong", "<s>"], base_count=2, added_count=1, model_vocab_size=3)

    def test_selection_is_lexical_and_category_balanced(self) -> None:
        rows = [
            {"source_document_id": f"{category}-b", "category": category, "text": "b"}
            for category in CATEGORIES
        ] + [
            {"source_document_id": f"{category}-a", "category": category, "text": "a"}
            for category in CATEGORIES
        ]
        selected = _selected_rows(rows, 1)
        self.assertEqual([row["source_document_id"] for row in selected], [f"{category}-a" for category in CATEGORIES])

    def test_private_exchange_is_deterministic_and_text_is_base64_only(self) -> None:
        document = PreparedDocument("doc-α", "prose", "A\nB", b"A\nB", (65, 10, 66))
        with tempfile.TemporaryDirectory(prefix="strat01-gguf-score-") as directory:
            path = Path(directory) / "private.tsv"
            _write_private_exchange(path, [document])
            line = path.read_text(encoding="ascii").strip()
            self.assertNotIn("doc-α", line)
            self.assertNotIn("A\nB", line)
            fields = line.split("\t")
            self.assertEqual(base64.b64decode(fields[0]), b"doc-\xce\xb1")
            self.assertEqual(base64.b64decode(fields[3]), b"A\nB")
            self.assertEqual(fields[2:], ["3", fields[3], "65,10,66"])

    def test_fake_tokenizer_preflight_preserves_raw_utf8_and_context_bound(self) -> None:
        rows = [{"source_document_id": "one", "category": "code", "text": "abc"}]
        prepared = _prepare_documents(rows, FakeTokenizer(), 4)
        self.assertEqual(prepared[0].raw_bytes, b"abc")
        self.assertEqual(prepared[0].ids, (97, 98, 99))
        with self.assertRaisesRegex(ScorerError, "no-truncation"):
            _prepare_documents(rows, FakeTokenizer(), 3)

    def test_core_has_required_scientific_guards(self) -> None:
        source = CORE.read_text(encoding="utf-8")
        for required in ("llama_tokenize", "llama_memory_clear", "stable_logsumexp", "payload_eos_appended\\\":false",
                         "runtime_tokenizer_mismatch_documents", "--metadata-out", "--source-token-ids"):
            self.assertIn(required, source)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="test/build/run wrapper for the bounded STRAT-01 GGUF scorer")
    parser.add_argument("--build", action="store_true", help="create a standalone CMake build directory and compile the C++ core")
    parser.add_argument("--select-stage", choices=("instrument", "pilot"), help="write the frozen calibration subset and selection manifest")
    parser.add_argument("--build-dir", type=Path, default=Path(tempfile.gettempdir()) / "siliconllm-strat01-gguf-score-build")
    parser.add_argument("--llama-source", type=Path, default=PINNED_LLAMA)
    parser.add_argument("--configuration", default="Release")
    parser.add_argument("--score", action="store_true", help="run the already-built C++ scorer; never runs implicitly")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--expected-model-sha256", help="content-address a new GGUF arm")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--n-ctx", type=int, default=MAX_CONTEXT)
    parser.add_argument("--n-batch", type=int, default=128)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--context-marker", choices=("bos",), default="bos", help="frozen GigaChat document prefix: BOS=1")
    parser.add_argument("--keep-exchange", action="store_true", help="retain the base64 private exchange file for apparatus debugging")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        parser.error("unrecognized arguments")
    if sum(bool(item) for item in (args.build, args.score, args.select_stage)) > 1:
        parser.error("choose only one of --build, --score, or --select-stage")
    if args.select_stage:
        if args.corpus is None or args.output is None:
            parser.error("--select-stage requires --corpus and --output")
        select_calibration(args.corpus, args.output, args.select_stage)
        return 0
    if args.build:
        binary = build(args.build_dir, args.llama_source, args.configuration)
        print(binary)
        return 0
    if args.score:
        required = (args.binary, args.model, args.corpus, args.tokenizer_json, args.output)
        if any(item is None for item in required):
            parser.error("--score requires --binary --model --corpus --tokenizer-json --output")
        score(args)
        return 0
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Strat01GgufScoreTests)
    outcome = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if outcome.wasSuccessful() else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScorerError as exc:
        print(f"strat01_gguf_score: {exc}", file=sys.stderr)
        raise SystemExit(2)
