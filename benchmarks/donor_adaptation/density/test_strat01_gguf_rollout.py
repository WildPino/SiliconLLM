#!/usr/bin/env python3
"""Build, run, bind, and adjudicate the STRAT-01 GGUF document rollout."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
CORE = HERE / "strat01_gguf_rollout.cpp"
PINNED_LLAMA = Path(r"C:\Users\giosa\AppData\Local\Temp\siliconllm-llama-bind-5b335f4")
PINNED_LLAMA_HEAD = "5b335f413e4f73b0809c4fe39af894efbcc6a0d2"
TOKENIZER_SHA256 = "b4b3d90c67830a4e566296ad8d0f6b5ac5a5cdbfd331b200d0ee8263aaaea1fe"
HELDOUT_SHA256 = "04687034c1054e0985e24efa87ba37a3fb11031de5b2880e5b8ac1742f80ec6e"
MODEL_SHA256 = {
    "GigaChat3.1-10B-A1.8B-bf16.gguf": "e7a6409be0ac197babf21c48cfc8a96486d035c1e7a784feadd817b7b883c08e",
    "GigaChat3.1-10B-A1.8B-q4_K_M.gguf": "68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb",
}
BOS_ID = 1
EOS_ID = 2
MODEL_VOCAB = 128_256
DOCUMENTS = 96
PREFIX_TOKENS = 256
MAX_NEW_TOKENS = 256
DEFAULT_CONTEXT = 4096
CATEGORIES = ("code", "technical_general", "prose")
CORE_FIELDS = {
    "index", "item_id", "prompt_ids", "generated_ids", "stop_reason",
    "generated_count", "elapsed_seconds", "run32", "loop8x3", "empty",
}
PUBLIC_FIELDS = (
    "index", "source_document_id", "category", "prompt_ids_sha256",
    "generated_ids", "decoded_text", "emitted_utf8_bytes", "stop_reason",
    "generated_count", "elapsed_seconds", "run32", "loop8x3", "empty", "degenerate",
)


class RolloutError(RuntimeError):
    """A frozen input, apparatus invariant, or paired artifact is invalid."""


@dataclass(frozen=True)
class PreparedDocument:
    index: int
    source_document_id: str
    category: str
    source_content_sha256: str
    item_sha256: str
    prompt_ids: tuple[int, ...]
    prompt_ids_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True, capture_output=True, check=False
    )
    if completed.returncode or len(completed.stdout.strip()) != 40:
        raise RolloutError("cannot bind the repository HEAD")
    return completed.stdout.strip()


def _require_new(path: Path, label: str) -> None:
    if path.exists():
        raise RolloutError(f"{label} already exists; refusing to overwrite it")


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _read_heldout(path: Path) -> list[dict[str, Any]]:
    if _sha256(path) != HELDOUT_SHA256:
        raise RolloutError("fresh-heldout JSONL SHA-256 differs from the frozen corpus")
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RolloutError("cannot read fresh-heldout JSONL as UTF-8") from exc
    if not raw or not raw.endswith(b"\n"):
        raise RolloutError("fresh-heldout JSONL must be non-empty and newline-terminated")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RolloutError(f"fresh-heldout:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise RolloutError(f"fresh-heldout:{line_number}: row is not an object")
        identifier, category, payload = row.get("source_document_id"), row.get("category"), row.get("text")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise RolloutError(f"fresh-heldout:{line_number}: source_document_id is invalid or duplicate")
        if category not in CATEGORIES or not isinstance(payload, str) or not payload:
            raise RolloutError(f"fresh-heldout:{line_number}: category/text violates the frozen schema")
        for key in ("source_content_sha256", "item_sha256"):
            if not isinstance(row.get(key), str) or len(row[key]) != 64:
                raise RolloutError(f"fresh-heldout:{line_number}: {key} is invalid")
        seen.add(identifier)
        rows.append(row)
    if len(rows) != DOCUMENTS or Counter(row["category"] for row in rows) != Counter({name: 32 for name in CATEGORIES}):
        raise RolloutError("fresh-heldout must contain 96 rows, 32 per category")
    return rows


def _special_ids(tokenizer_json: Path) -> set[int]:
    try:
        payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RolloutError("cannot read source tokenizer JSON") from exc
    added = payload.get("added_tokens")
    if not isinstance(added, list):
        raise RolloutError("source tokenizer has no valid added-token table")
    result = {
        entry["id"] for entry in added
        if isinstance(entry, dict) and type(entry.get("id")) is int and entry.get("special") is True
    }
    if BOS_ID not in result or EOS_ID not in result:
        raise RolloutError("source tokenizer does not bind BOS=1 and EOS=2 as special")
    return result


def _defined_source_ids(tokenizer_json: Path) -> set[int]:
    try:
        payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
        vocabulary = payload["model"]["vocab"]
        added = payload["added_tokens"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RolloutError("cannot derive defined source-tokenizer IDs") from exc
    if not isinstance(vocabulary, dict) or not isinstance(added, list):
        raise RolloutError("source tokenizer vocabulary tables are malformed")
    result = {identifier for identifier in vocabulary.values() if type(identifier) is int}
    result.update(entry["id"] for entry in added if isinstance(entry, dict) and type(entry.get("id")) is int)
    if not result or min(result) < 0 or max(result) >= MODEL_VOCAB:
        raise RolloutError("source tokenizer IDs exceed the pinned model vocabulary")
    return result


def prepare_documents(
    rows: Sequence[dict[str, Any]], tokenizer: Any, special_ids: set[int], n_ctx: int
) -> list[PreparedDocument]:
    if n_ctx < 1 + PREFIX_TOKENS + MAX_NEW_TOKENS:
        raise RolloutError("n-ctx cannot hold the frozen 257-token prompt plus 256 generated positions")
    prepared: list[PreparedDocument] = []
    for index, row in enumerate(rows):
        text = row["text"]
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            ids = getattr(encoded, "ids", None)
        except Exception as exc:
            raise RolloutError(f"source tokenizer failed on heldout row {index}") from exc
        if not isinstance(ids, list) or len(ids) < PREFIX_TOKENS or any(type(token) is not int or token < 0 for token in ids):
            raise RolloutError(f"source tokenizer emitted insufficient or invalid IDs at heldout row {index}")
        if special_ids.intersection(ids):
            raise RolloutError(f"heldout payload contains a source special ID at row {index}")
        try:
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
        except Exception as exc:
            raise RolloutError(f"source tokenizer cannot decode heldout row {index}") from exc
        if decoded != text:
            raise RolloutError(f"source tokenizer round-trip differs at heldout row {index}")
        prompt = (BOS_ID, *ids[:PREFIX_TOKENS])
        binding = {
            "index": index,
            "source_document_id": row["source_document_id"],
            "category": row["category"],
            "prompt_ids": prompt,
        }
        prepared.append(PreparedDocument(
            index=index,
            source_document_id=row["source_document_id"],
            category=row["category"],
            source_content_sha256=row["source_content_sha256"],
            item_sha256=row["item_sha256"],
            prompt_ids=tuple(prompt),
            prompt_ids_sha256=_canonical_hash(binding),
        ))
    return prepared


def _write_exchange(path: Path, documents: Sequence[PreparedDocument]) -> None:
    _require_new(path, "private rollout exchange")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            record = {
                "index": document.index,
                "item_id": document.source_document_id,
                "prompt_ids": list(document.prompt_ids),
                "prompt_ids_sha256": document.prompt_ids_sha256,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def rollout_flags(ids: Sequence[int]) -> tuple[bool, bool, bool]:
    tokens = list(ids)
    if any(type(token) is not int or token < 0 or token == EOS_ID for token in tokens):
        raise RolloutError("rollout flags require non-negative, non-EOS generated IDs")
    run32 = any(len(set(tokens[start:start + 32])) == 1 for start in range(max(0, len(tokens) - 31)))
    loop8x3 = any(
        tokens[start:start + 8] == tokens[start + 8:start + 16] == tokens[start + 16:start + 24]
        for start in range(max(0, len(tokens) - 23))
    )
    return run32, loop8x3, not tokens


def _read_core_results(path: Path, documents: Sequence[PreparedDocument]) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise RolloutError("cannot read rollout core JSONL") from exc
    if len(lines) != len(documents):
        raise RolloutError("rollout core JSONL has the wrong item count")
    output: list[dict[str, Any]] = []
    for expected, line in zip(documents, lines, strict=True):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RolloutError(f"rollout core JSON is invalid at item {expected.index}") from exc
        if not isinstance(record, dict) or set(record) != CORE_FIELDS:
            raise RolloutError(f"rollout core schema differs at item {expected.index}")
        if (record["index"] != expected.index or record["item_id"] != expected.source_document_id
                or record["prompt_ids"] != list(expected.prompt_ids)):
            raise RolloutError(f"rollout core identity/prompt binding differs at item {expected.index}")
        ids = record["generated_ids"]
        if (not isinstance(ids, list) or len(ids) > MAX_NEW_TOKENS
                or any(type(token) is not int or token < 0 or token >= MODEL_VOCAB or token == EOS_ID for token in ids)):
            raise RolloutError(f"rollout core emitted invalid IDs at item {expected.index}")
        if record["stop_reason"] not in {"EOS", "CAP"} or record["generated_count"] != len(ids):
            raise RolloutError(f"rollout core stop/count fields differ at item {expected.index}")
        if ((record["stop_reason"] == "CAP") is not (len(ids) == MAX_NEW_TOKENS)
                or (record["stop_reason"] == "EOS" and len(ids) >= MAX_NEW_TOKENS)):
            raise RolloutError(f"rollout core stop semantics differ at item {expected.index}")
        elapsed = record["elapsed_seconds"]
        if type(elapsed) not in (int, float) or not math.isfinite(float(elapsed)) or elapsed < 0:
            raise RolloutError(f"rollout core elapsed time is invalid at item {expected.index}")
        flags = rollout_flags(ids)
        if tuple(record[name] for name in ("run32", "loop8x3", "empty")) != flags:
            raise RolloutError(f"rollout core degeneration flags differ at item {expected.index}")
        output.append(record)
    return output


def _read_runtime(
    path: Path, expected_items: int, n_ctx: int, n_batch: int, n_threads: int
) -> dict[str, Any]:
    try:
        runtime = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RolloutError("rollout core emitted invalid runtime metadata") from exc
    required = {
        "schema": "strat01_gguf_rollout_runtime_v1",
        "items": expected_items,
        "model_bos_id": BOS_ID,
        "model_eos_id": EOS_ID,
        "model_vocab_size": MODEL_VOCAB,
        "logical_n_ctx": n_ctx,
        "n_batch": n_batch,
        "n_threads": n_threads,
        "source_token_ids_mode": True,
        "greedy": True,
        "tie_policy": "lowest_id",
        "kv_reset_per_document": True,
        "generation_max_new_tokens": MAX_NEW_TOKENS,
    }
    if not isinstance(runtime, dict) or any(runtime.get(key) != value for key, value in required.items()):
        raise RolloutError("rollout runtime metadata violates the frozen protocol")
    physical = runtime.get("physical_n_ctx")
    if type(physical) is not int or physical < n_ctx:
        raise RolloutError("rollout runtime physical context is smaller than requested")
    elapsed = runtime.get("total_elapsed_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(float(elapsed)) or elapsed < 0:
        raise RolloutError("rollout runtime total elapsed time is invalid")
    return runtime


def _clean_pinned_llama(source: Path) -> dict[str, Any]:
    if not (source / "CMakeLists.txt").is_file():
        raise RolloutError("pinned llama.cpp source is unavailable")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, text=True, capture_output=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=source, text=True, capture_output=True, check=False)
    if head.returncode or head.stdout.strip() != PINNED_LLAMA_HEAD or status.returncode or status.stdout.strip():
        raise RolloutError("llama.cpp source is not the clean pinned commit")
    return {"path": str(source), "commit": PINNED_LLAMA_HEAD, "clean": True}


def _cmake_project(source: Path, llama_source: Path) -> str:
    return f'''cmake_minimum_required(VERSION 3.20)
project(strat01_gguf_rollout LANGUAGES C CXX)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
add_subdirectory("{llama_source.as_posix()}" llama-cpp)
add_executable(strat01_gguf_rollout "{source.as_posix()}")
target_compile_features(strat01_gguf_rollout PRIVATE cxx_std_17)
target_link_libraries(strat01_gguf_rollout PRIVATE llama)
'''


def build(binary_dir: Path, llama_source: Path, configuration: str) -> Path:
    _clean_pinned_llama(llama_source)
    if not CORE.is_file():
        raise RolloutError("rollout C++ core is unavailable")
    binary_dir.mkdir(parents=True, exist_ok=True)
    (binary_dir / "CMakeLists.txt").write_text(_cmake_project(CORE, llama_source), encoding="utf-8", newline="\n")
    configured = subprocess.run(
        ["cmake", "-S", str(binary_dir), "-B", str(binary_dir / "cmake-build")],
        text=True, capture_output=True, check=False,
    )
    if configured.returncode:
        raise RolloutError("rollout CMake configure failed; inspect the build directory")
    compiled = subprocess.run(
        ["cmake", "--build", str(binary_dir / "cmake-build"), "--target", "strat01_gguf_rollout", "--config", configuration],
        text=True, capture_output=True, check=False,
    )
    if compiled.returncode:
        tail = "\n".join(compiled.stdout.splitlines()[-10:] + compiled.stderr.splitlines()[-10:])
        raise RolloutError(f"rollout CMake build failed; tail follows:\n{tail}")
    candidates = [path for name in ("strat01_gguf_rollout.exe", "strat01_gguf_rollout")
                  for path in (binary_dir / "cmake-build").rglob(name)]
    if len(candidates) != 1:
        raise RolloutError("rollout build did not produce exactly one binary")
    return candidates[0]


def _write_token_manifest(path: Path, documents: Sequence[PreparedDocument]) -> tuple[list[dict[str, Any]], str]:
    _require_new(path, "rollout token manifest")
    records = [
        {"index": item.index, "source_document_id": item.source_document_id, "category": item.category,
         "source_content_sha256": item.source_content_sha256, "item_sha256": item.item_sha256,
         "prompt_tokens": len(item.prompt_ids), "prompt_ids_sha256": item.prompt_ids_sha256}
        for item in documents
    ]
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return records, _canonical_hash(records)


def score(args: argparse.Namespace) -> None:
    if args.n_ctx < 1 + PREFIX_TOKENS + MAX_NEW_TOKENS or args.n_ctx > DEFAULT_CONTEXT:
        raise RolloutError("n-ctx violates the bounded rollout contract")
    if args.n_batch < 1 or args.n_batch > args.n_ctx:
        raise RolloutError("n-batch violates the bounded rollout contract")
    if type(args.threads) is not int or args.threads < 1:
        raise RolloutError("threads must be an explicit positive integer")
    for path, label in ((args.binary, "rollout binary"), (args.model, "GGUF model"),
                        (args.tokenizer_json, "source tokenizer"), (args.corpus, "fresh-heldout corpus")):
        if not path.is_file():
            raise RolloutError(f"{label} is unavailable")
    expected_model_sha = MODEL_SHA256.get(args.model.name)
    if expected_model_sha is None or _sha256(args.model) != expected_model_sha:
        raise RolloutError("GGUF is not one of the two pinned STRAT-01 arms")
    if _sha256(args.tokenizer_json) != TOKENIZER_SHA256:
        raise RolloutError("source tokenizer SHA-256 differs from the pinned asset")
    rows = _read_heldout(args.corpus)
    if args.limit is not None:
        if args.limit < 1 or args.limit > DOCUMENTS:
            raise RolloutError("--limit must be in 1..96")
        rows = rows[:args.limit]
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RolloutError("rollout scoring requires the tokenizers package") from exc
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    documents = prepare_documents(rows, tokenizer, _special_ids(args.tokenizer_json), args.n_ctx)
    defined_ids = _defined_source_ids(args.tokenizer_json)

    output = args.output
    manifest = args.manifest or output.with_suffix(output.suffix + ".manifest.json")
    runtime_path = output.with_suffix(output.suffix + ".runtime.json")
    core_output = output.with_suffix(output.suffix + ".core.jsonl")
    token_manifest = output.with_suffix(output.suffix + ".tokens.jsonl")
    exchange = output.with_suffix(output.suffix + ".private.jsonl")
    for path, label in ((output, "rollout output"), (manifest, "rollout manifest"),
                        (runtime_path, "rollout runtime metadata"), (core_output, "rollout core output"),
                        (token_manifest, "rollout token manifest"), (exchange, "private rollout exchange")):
        _require_new(path, label)
    _write_exchange(exchange, documents)
    model_stat = args.model.stat()
    try:
        command = [str(args.binary), "--model", str(args.model), "--input", str(exchange),
                   "--output", str(core_output), "--metadata-out", str(runtime_path),
                   "--n-ctx", str(args.n_ctx), "--n-batch", str(args.n_batch),
                   "--threads", str(args.threads)]
        completed = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, text=True, check=False)
        if completed.returncode:
            diagnostic = completed.stderr.strip()
            if not diagnostic.startswith("strat01_gguf_rollout: ") or "\n" in diagnostic:
                diagnostic = "unrecognized core diagnostic; inspect the preserved apparatus state"
            raise RolloutError(f"rollout core failed with exit status {completed.returncode}: {diagnostic}")
    finally:
        if not args.keep_exchange:
            exchange.unlink(missing_ok=True)
    core_records = _read_core_results(core_output, documents)
    runtime = _read_runtime(runtime_path, len(documents), args.n_ctx, args.n_batch, args.threads)
    if (model_stat.st_size, model_stat.st_mtime_ns) != (args.model.stat().st_size, args.model.stat().st_mtime_ns):
        raise RolloutError("GGUF changed during rollout")
    public_records: list[dict[str, Any]] = []
    for item, raw in zip(documents, core_records, strict=True):
        ids = raw["generated_ids"]
        undefined = [token for token in ids if token not in defined_ids]
        if undefined:
            raise RolloutError(f"model emitted source-undefined token ID {undefined[0]} at item {item.index}; core output preserved")
        try:
            text = tokenizer.decode(ids, skip_special_tokens=False)
        except Exception as exc:
            raise RolloutError(f"source tokenizer failed to decode item {item.index}; core output preserved") from exc
        flags = tuple(raw[name] for name in ("run32", "loop8x3", "empty"))
        public_records.append({
            "index": item.index, "source_document_id": item.source_document_id, "category": item.category,
            "prompt_ids_sha256": item.prompt_ids_sha256, "generated_ids": ids, "decoded_text": text,
            "emitted_utf8_bytes": len(text.encode("utf-8")), "stop_reason": raw["stop_reason"],
            "generated_count": len(ids), "elapsed_seconds": raw["elapsed_seconds"],
            "run32": flags[0], "loop8x3": flags[1], "empty": flags[2], "degenerate": any(flags),
        })
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for record in public_records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    token_records, token_binding = _write_token_manifest(token_manifest, documents)
    manifest_record = {
        "schema": "strat01_gigachat_rollout_manifest_v1",
        "corpus": {"path": str(args.corpus), "sha256": HELDOUT_SHA256, "records": DOCUMENTS},
        "selection": {"items": len(documents), "full_gate": len(documents) == DOCUMENTS,
                      "original_order_prefix": True, "categories": dict(Counter(item.category for item in documents))},
        "tokenizer": {"path": str(args.tokenizer_json), "sha256": TOKENIZER_SHA256,
                      "defined_source_ids": len(defined_ids)},
        "token_binding": {"path": str(token_manifest), "sha256": _sha256(token_manifest),
                          "records": len(token_records), "all_item_hashes_sha256": token_binding},
        "model": {"path": str(args.model), "sha256": expected_model_sha},
        "scorer": {"binary": str(args.binary), "binary_sha256": _sha256(args.binary),
                   "source": str(CORE), "source_sha256": _sha256(CORE),
                   "llama_cpp": _clean_pinned_llama(args.llama_source)},
        "output": {"path": str(output), "sha256": _sha256(output), "records": len(public_records)},
        "core_output": {"path": str(core_output), "sha256": _sha256(core_output)},
        "runtime_metadata": {"path": str(runtime_path), "sha256": _sha256(runtime_path), "record": runtime},
        "protocol": {"prompt": "[BOS=1] + first 256 source payload IDs", "max_new_tokens": MAX_NEW_TOKENS,
                     "eos_id": EOS_ID, "eos_retained": False, "greedy": True,
                     "tie_policy": "lowest_id", "source_token_decode": True,
                     "degeneration_flags": ["run32", "loop8x3", "empty"]},
        "repository_head": _git_head(),
    }
    with manifest.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest_record, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _read_public(path: Path, *, require_full: bool) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise RolloutError(f"cannot read rollout artifact {path}") from exc
    if not lines or (require_full and len(lines) != DOCUMENTS):
        raise RolloutError(f"rollout artifact {path.name} has the wrong item count")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RolloutError(f"rollout artifact {path.name} has invalid JSON at {index}") from exc
        if not isinstance(record, dict) or tuple(record) != PUBLIC_FIELDS or record["index"] != index:
            raise RolloutError(f"rollout artifact {path.name} schema/order differs at {index}")
        ids = record["generated_ids"]
        if (not isinstance(record["source_document_id"], str) or not record["source_document_id"]
                or record["category"] not in CATEGORIES or not isinstance(record["prompt_ids_sha256"], str)
                or len(record["prompt_ids_sha256"]) != 64 or not isinstance(ids, list)
                or len(ids) > MAX_NEW_TOKENS
                or any(type(token) is not int or token < 0 or token >= MODEL_VOCAB or token == EOS_ID for token in ids)
                or not isinstance(record["decoded_text"], str)
                or record["stop_reason"] not in {"EOS", "CAP"}
                or ((record["stop_reason"] == "CAP") is not (len(ids) == MAX_NEW_TOKENS))
                or (record["stop_reason"] == "EOS" and len(ids) >= MAX_NEW_TOKENS)
                or type(record["elapsed_seconds"]) not in (int, float)
                or not math.isfinite(float(record["elapsed_seconds"])) or record["elapsed_seconds"] < 0):
            raise RolloutError(f"rollout artifact {path.name} primitive fields differ at {index}")
        flags = rollout_flags(ids)
        if (record["generated_count"] != len(ids) or record["emitted_utf8_bytes"] != len(record["decoded_text"].encode("utf-8"))
                or tuple(record[name] for name in ("run32", "loop8x3", "empty")) != flags
                or record["degenerate"] is not any(flags)):
            raise RolloutError(f"rollout artifact {path.name} derived fields differ at {index}")
        records.append(record)
    return records


def _load_manifest(
    path: Path,
    expected_model_sha: str,
    *,
    expected_items: int = DOCUMENTS,
    require_full: bool = True,
) -> dict[str, Any]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RolloutError(f"cannot read bound manifest for {path.name}") from exc
    if (manifest.get("schema") != "strat01_gigachat_rollout_manifest_v1"
            or manifest.get("selection", {}).get("full_gate") is not require_full
            or manifest.get("selection", {}).get("items") != expected_items
            or manifest.get("corpus", {}).get("sha256") != HELDOUT_SHA256
            or manifest.get("tokenizer", {}).get("sha256") != TOKENIZER_SHA256
            or manifest.get("model", {}).get("sha256") != expected_model_sha
            or manifest.get("output", {}).get("sha256") != _sha256(path)):
        raise RolloutError(f"rollout manifest binding failed for {path.name}")
    for key in ("core_output", "runtime_metadata", "token_binding"):
        bound = manifest.get(key, {})
        bound_path = Path(bound.get("path", ""))
        if not bound_path.is_file() or bound.get("sha256") != _sha256(bound_path):
            raise RolloutError(f"rollout manifest {key} binding failed for {path.name}")
    return manifest


def _common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def verify_apparatus(args: argparse.Namespace) -> None:
    _require_new(args.output, "rollout apparatus verification")
    arms = {
        "bf16_batch128": (args.bf16_batch128, MODEL_SHA256["GigaChat3.1-10B-A1.8B-bf16.gguf"], 128),
        "bf16_batch64": (args.bf16_batch64, MODEL_SHA256["GigaChat3.1-10B-A1.8B-bf16.gguf"], 64),
        "q4_batch128": (args.q4_batch128, MODEL_SHA256["GigaChat3.1-10B-A1.8B-q4_K_M.gguf"], 128),
        "q4_batch64": (args.q4_batch64, MODEL_SHA256["GigaChat3.1-10B-A1.8B-q4_K_M.gguf"], 64),
    }
    records: dict[str, dict[str, Any]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    for name, (path, model_sha, expected_batch) in arms.items():
        if path is None or not path.is_file():
            raise RolloutError(f"apparatus arm {name} is unavailable")
        rows = _read_public(path, require_full=False)
        if len(rows) != 1:
            raise RolloutError(f"apparatus arm {name} must contain exactly one item")
        manifest = _load_manifest(path, model_sha, expected_items=1, require_full=False)
        runtime = manifest["runtime_metadata"]["record"]
        if runtime.get("n_batch") != expected_batch or type(runtime.get("n_threads")) is not int or runtime["n_threads"] < 1:
            raise RolloutError(f"apparatus arm {name} has invalid execution geometry")
        records[name] = rows[0]
        manifests[name] = manifest
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        artifacts[name] = {
            "path": str(path), "sha256": _sha256(path),
            "manifest_path": str(manifest_path), "manifest_sha256": _sha256(manifest_path),
            "elapsed_seconds": runtime["total_elapsed_seconds"],
        }

    first_manifest = manifests["bf16_batch128"]
    binding = first_manifest["token_binding"]["all_item_hashes_sha256"]
    for name, manifest in manifests.items():
        if manifest["token_binding"]["all_item_hashes_sha256"] != binding:
            raise RolloutError(f"apparatus arm {name} has a different token binding")
        for key in ("scorer", "protocol"):
            if manifest.get(key) != first_manifest.get(key):
                raise RolloutError(f"apparatus arm {name} has a different {key} binding")

    parity_fields = tuple(field for field in PUBLIC_FIELDS if field != "elapsed_seconds")
    for arm in ("bf16", "q4"):
        left, right = records[f"{arm}_batch128"], records[f"{arm}_batch64"]
        if any(left[field] != right[field] for field in parity_fields):
            raise RolloutError(f"{arm} rollout differs between batch128 and batch64")

    teacher, candidate = records["bf16_batch128"], records["q4_batch128"]
    identity = ("source_document_id", "category", "prompt_ids_sha256")
    if any(teacher[field] != candidate[field] for field in identity):
        raise RolloutError("paired apparatus arm identity differs")
    q4_only_degenerate = candidate["degenerate"] and not teacher["degenerate"]
    report = {
        "schema": "strat01_gigachat_rollout_apparatus_v1",
        "status": "PASS_APPARATUS" if not q4_only_degenerate else "FAIL_APPARATUS_Q4_ONLY_DEGENERATION",
        "item": {field: teacher[field] for field in identity},
        "batch_parity": {"bf16_exact": True, "q4_exact": True, "fields": list(parity_fields)},
        "paired_smoke": {
            "bf16_degenerate": teacher["degenerate"],
            "q4_degenerate": candidate["degenerate"],
            "q4_only_degenerate": q4_only_degenerate,
            "common_prefix_tokens": _common_prefix(teacher["generated_ids"], candidate["generated_ids"]),
        },
        "bindings": {
            "corpus_sha256": HELDOUT_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "token_ids": binding,
        },
        "artifacts": artifacts,
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def adjudicate(args: argparse.Namespace) -> None:
    _require_new(args.output, "rollout adjudication report")
    teacher = _read_public(args.teacher, require_full=True)
    candidate = _read_public(args.candidate, require_full=True)
    teacher_manifest = _load_manifest(args.teacher, MODEL_SHA256["GigaChat3.1-10B-A1.8B-bf16.gguf"])
    candidate_manifest = _load_manifest(args.candidate, MODEL_SHA256["GigaChat3.1-10B-A1.8B-q4_K_M.gguf"])
    if teacher_manifest["token_binding"]["all_item_hashes_sha256"] != candidate_manifest["token_binding"]["all_item_hashes_sha256"]:
        raise RolloutError("teacher and candidate rollout token bindings differ")
    for key in ("scorer", "protocol"):
        if teacher_manifest.get(key) != candidate_manifest.get(key):
            raise RolloutError(f"teacher and candidate rollout {key} bindings differ")
    categories = {name: {"teacher_degenerate": 0, "candidate_degenerate": 0, "new_candidate_degenerate": 0}
                  for name in CATEGORIES}
    new_degenerate: list[str] = []
    exact = 0
    prefix_lengths: list[int] = []
    for index, (base, trial) in enumerate(zip(teacher, candidate, strict=True)):
        identity = ("source_document_id", "category", "prompt_ids_sha256")
        if any(base[key] != trial[key] for key in identity):
            raise RolloutError(f"paired rollout identity differs at item {index}")
        category = base["category"]
        if base["degenerate"]:
            categories[category]["teacher_degenerate"] += 1
        if trial["degenerate"]:
            categories[category]["candidate_degenerate"] += 1
        if trial["degenerate"] and not base["degenerate"]:
            categories[category]["new_candidate_degenerate"] += 1
            new_degenerate.append(base["source_document_id"])
        if base["generated_ids"] == trial["generated_ids"]:
            exact += 1
        prefix_lengths.append(_common_prefix(base["generated_ids"], trial["generated_ids"]))
    catastrophic = len(new_degenerate) >= 3
    report = {
        "schema": "strat01_gigachat_rollout_adjudication_v1",
        "status": "FAIL_CATASTROPHIC_ROLLOUT" if catastrophic else "PASS_DOCUMENT_ROLLOUT",
        "documents": DOCUMENTS,
        "gate": {"fail_if_new_candidate_degenerate_at_least": 3,
                 "new_candidate_degenerate_count": len(new_degenerate), "passed": not catastrophic},
        "categories": categories,
        "new_candidate_degenerate_document_ids": new_degenerate,
        "diagnostics": {
            "exact_generated_id_sequences": exact,
            "mean_common_prefix_tokens": sum(prefix_lengths) / len(prefix_lengths),
            "teacher_eos_documents": sum(item["stop_reason"] == "EOS" for item in teacher),
            "candidate_eos_documents": sum(item["stop_reason"] == "EOS" for item in candidate),
            "teacher_mean_generated_tokens": sum(item["generated_count"] for item in teacher) / DOCUMENTS,
            "candidate_mean_generated_tokens": sum(item["generated_count"] for item in candidate) / DOCUMENTS,
        },
        "teacher": {"path": str(args.teacher), "sha256": _sha256(args.teacher),
                    "manifest_sha256": _sha256(args.teacher.with_suffix(args.teacher.suffix + ".manifest.json"))},
        "candidate": {"path": str(args.candidate), "sha256": _sha256(args.candidate),
                      "manifest_sha256": _sha256(args.candidate.with_suffix(args.candidate.suffix + ".manifest.json"))},
        "bindings": {"corpus_sha256": HELDOUT_SHA256, "tokenizer_sha256": TOKENIZER_SHA256,
                     "token_ids": teacher_manifest["token_binding"]["all_item_hashes_sha256"]},
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
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
        return "".join(chr(token) for token in ids)


class RolloutTests(unittest.TestCase):
    def test_flags_match_frozen_semantics(self) -> None:
        self.assertEqual(rollout_flags([]), (False, False, True))
        self.assertEqual(rollout_flags([7] * 32), (True, True, False))
        loop = list(range(10, 18)) * 3
        self.assertEqual(rollout_flags(loop), (False, True, False))
        self.assertEqual(rollout_flags(list(range(100, 140))), (False, False, False))

    def test_preparation_uses_bos_plus_exact_first_256(self) -> None:
        text = "".join(chr(1000 + index) for index in range(300))
        row = {"source_document_id": "x", "category": "code", "text": text,
               "source_content_sha256": "a" * 64, "item_sha256": "b" * 64}
        item = prepare_documents([row], FakeTokenizer(), {BOS_ID, EOS_ID}, 513)[0]
        self.assertEqual(item.prompt_ids[0], BOS_ID)
        self.assertEqual(item.prompt_ids[1:], tuple(ord(character) for character in text[:256]))
        self.assertEqual(len(item.prompt_ids), 257)

    def test_exchange_contains_ids_and_binding(self) -> None:
        item = PreparedDocument(0, "doc", "prose", "a" * 64, "b" * 64, (1, 5, 6), "c" * 64)
        with tempfile.TemporaryDirectory(prefix="strat01-rollout-") as directory:
            path = Path(directory) / "private.jsonl"
            _write_exchange(path, [item])
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record, {"index": 0, "item_id": "doc", "prompt_ids": [1, 5, 6],
                                      "prompt_ids_sha256": "c" * 64})

    def test_core_source_has_required_guards(self) -> None:
        source = CORE.read_text(encoding="utf-8")
        for required in ("--self-test", "llama_memory_clear", "lowest_id", "run32", "loop8x3",
                         "prompt_ids", "total_elapsed_seconds"):
            self.assertIn(required, source)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="STRAT-01 GigaChat GGUF document-rollout apparatus")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--build", action="store_true")
    modes.add_argument("--score", action="store_true")
    modes.add_argument("--verify-apparatus", action="store_true")
    modes.add_argument("--adjudicate", action="store_true")
    parser.add_argument("--build-dir", type=Path,
                        default=Path(tempfile.gettempdir()) / "siliconllm-strat01-gguf-rollout-build")
    parser.add_argument("--llama-source", type=Path, default=PINNED_LLAMA)
    parser.add_argument("--configuration", default="Release")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--n-ctx", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--n-batch", type=int, default=128)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--limit", type=int, help="apparatus-only original-order prefix; full gate omits this")
    parser.add_argument("--keep-exchange", action="store_true")
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--bf16-batch128", type=Path)
    parser.add_argument("--bf16-batch64", type=Path)
    parser.add_argument("--q4-batch128", type=Path)
    parser.add_argument("--q4-batch64", type=Path)
    args = parser.parse_args(argv)
    if args.build:
        print(build(args.build_dir, args.llama_source, args.configuration))
        return 0
    if args.score:
        required = (args.binary, args.model, args.tokenizer_json, args.corpus, args.output)
        if any(path is None for path in required):
            parser.error("--score requires --binary --model --tokenizer-json --corpus --output")
        if args.threads is None:
            parser.error("--score requires an explicit --threads value")
        score(args)
        return 0
    if args.verify_apparatus:
        required = (args.bf16_batch128, args.bf16_batch64, args.q4_batch128, args.q4_batch64, args.output)
        if any(path is None for path in required):
            parser.error("--verify-apparatus requires all four batch-arm paths and --output")
        verify_apparatus(args)
        return 0
    if args.adjudicate:
        if args.teacher is None or args.candidate is None or args.output is None:
            parser.error("--adjudicate requires --teacher --candidate --output")
        adjudicate(args)
        return 0
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RolloutTests)
    outcome = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if outcome.wasSuccessful() else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RolloutError as exc:
        print(f"strat01_gguf_rollout: {exc}", file=sys.stderr)
        raise SystemExit(2)
