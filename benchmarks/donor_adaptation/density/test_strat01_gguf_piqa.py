#!/usr/bin/env python3
"""Build, preflight, run, and adjudicate the STRAT-01 GigaChat PIQA gate.

The model-facing C++ process receives source-tokenizer IDs only.  Raw PIQA
text remains confined to the pinned source files and process memory; score
artifacts contain item indices, token counts, token-ID hashes, and NLLs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from test_strat01_gguf_score import (  # type: ignore[import-not-found]
    MODEL_SHA256,
    TOKENIZER_SHA256,
    _git_head,
    _sha256,
    _verify_vocab_mapping,
)
from strat02_task_rollout import (  # type: ignore[import-not-found]
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    PIQA_COUNT,
    PiqaRecord,
    audit_pinned_sources,
    paired_accuracy_bootstrap,
)


HERE = Path(__file__).resolve().parent
CORE = HERE / "strat01_gguf_piqa.cpp"
PINNED_LLAMA = Path(r"C:\Users\giosa\AppData\Local\Temp\siliconllm-llama-bind-5b335f4")
PINNED_LLAMA_HEAD = "5b335f413e4f73b0809c4fe39af894efbcc6a0d2"
MAX_CONTEXT = 4096
BOS_ID = 1
RESULT_FIELDS = (
    "index", "label", "prefix_tokens",
    "option0_tokens", "option0_nll", "option0_mean_nll",
    "option1_tokens", "option1_nll", "option1_mean_nll",
    "choice_mean_nll", "choice_total_nll", "correct_mean_nll", "correct_total_nll",
)


class PiqaScorerError(RuntimeError):
    """The frozen PIQA apparatus or one of its inputs is invalid."""


@dataclass(frozen=True)
class PreparedItem:
    index: int
    label: int
    prefix_ids: tuple[int, ...]
    suffix_ids: tuple[tuple[int, ...], tuple[int, ...]]
    token_ids_sha256: str


def _require_new(path: Path, label: str) -> None:
    if path.exists():
        raise PiqaScorerError(f"{label} already exists; refusing to overwrite it")


def _encode_exact(tokenizer: Any, text: str, label: str, special_ids: set[int]) -> tuple[int, ...]:
    try:
        encoding = tokenizer.encode(text, add_special_tokens=False)
        ids = getattr(encoding, "ids", None)
    except Exception as exc:
        raise PiqaScorerError(f"source tokenizer failed on {label}") from exc
    if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids):
        raise PiqaScorerError(f"source tokenizer emitted invalid IDs for {label}")
    if special_ids.intersection(ids):
        raise PiqaScorerError(f"source tokenizer emitted a special ID inside {label}")
    try:
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
    except Exception as exc:
        raise PiqaScorerError(f"source tokenizer cannot round-trip {label}") from exc
    if decoded != text:
        raise PiqaScorerError(f"source tokenizer round-trip differs for {label}")
    return tuple(ids)


def _special_ids(tokenizer_json: Path) -> set[int]:
    try:
        payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PiqaScorerError("cannot read source tokenizer JSON") from exc
    added = payload.get("added_tokens")
    if not isinstance(added, list):
        raise PiqaScorerError("source tokenizer has no valid added-token table")
    result: set[int] = set()
    for entry in added:
        if not isinstance(entry, dict) or type(entry.get("id")) is not int:
            raise PiqaScorerError("source tokenizer has an invalid added-token entry")
        if entry.get("special") is True:
            result.add(entry["id"])
    if BOS_ID not in result:
        raise PiqaScorerError("source tokenizer does not mark pinned BOS=1 as special")
    return result


def _item_hash(index: int, label: int, prefix: Sequence[int], suffixes: Sequence[Sequence[int]]) -> str:
    canonical = json.dumps(
        {"index": index, "label": label, "prefix_ids": list(prefix),
         "suffix0_ids": list(suffixes[0]), "suffix1_ids": list(suffixes[1])},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def prepare_items(records: Sequence[PiqaRecord], tokenizer: Any, special_ids: set[int], n_ctx: int) -> list[PreparedItem]:
    prepared: list[PreparedItem] = []
    for expected_index, record in enumerate(records):
        if record.index != expected_index or record.label not in (0, 1):
            raise PiqaScorerError(f"PIQA source order/label mismatch at item {expected_index}")
        prefix = _encode_exact(tokenizer, f"Question: {record.goal}\nAnswer:", f"PIQA {expected_index} prefix", special_ids)
        suffixes = (
            _encode_exact(tokenizer, f" {record.sol1}", f"PIQA {expected_index} option 0", special_ids),
            _encode_exact(tokenizer, f" {record.sol2}", f"PIQA {expected_index} option 1", special_ids),
        )
        if any(1 + len(prefix) + len(suffix) > n_ctx for suffix in suffixes):
            raise PiqaScorerError(f"PIQA item {expected_index} exceeds the no-truncation context bound")
        prepared.append(PreparedItem(expected_index, record.label, prefix, suffixes,
                                     _item_hash(expected_index, record.label, prefix, suffixes)))
    return prepared


def _write_exchange(path: Path, items: Sequence[PreparedItem]) -> None:
    _require_new(path, "private PIQA exchange")
    with path.open("x", encoding="ascii", newline="\n") as handle:
        for item in items:
            fields = (
                str(item.index), str(item.label),
                ",".join(map(str, item.prefix_ids)),
                ",".join(map(str, item.suffix_ids[0])),
                ",".join(map(str, item.suffix_ids[1])),
            )
            handle.write("\t".join(fields) + "\n")


def _token_manifest_records(items: Sequence[PreparedItem]) -> list[dict[str, Any]]:
    return [
        {"index": item.index, "label": item.label, "prefix_tokens": len(item.prefix_ids),
         "option0_tokens": len(item.suffix_ids[0]), "option1_tokens": len(item.suffix_ids[1]),
         "token_ids_sha256": item.token_ids_sha256}
        for item in items
    ]


def _write_token_manifest(path: Path, items: Sequence[PreparedItem]) -> None:
    _require_new(path, "PIQA token-hash manifest")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in _token_manifest_records(items):
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def _read_results(path: Path, items: Sequence[PreparedItem] | None = None, *, require_full: bool = False) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise PiqaScorerError("cannot read PIQA score JSONL") from exc
    if not lines or (require_full and len(lines) != PIQA_COUNT) or (items is not None and len(lines) != len(items)):
        raise PiqaScorerError("PIQA score JSONL has the wrong item count")
    results: list[dict[str, Any]] = []
    for expected_index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PiqaScorerError(f"PIQA score JSON is invalid at item {expected_index}") from exc
        if not isinstance(record, dict) or tuple(record) != RESULT_FIELDS:
            raise PiqaScorerError(f"PIQA result schema differs at item {expected_index}")
        if record["index"] != expected_index or record["label"] not in (0, 1):
            raise PiqaScorerError(f"PIQA result identity differs at item {expected_index}")
        if items is not None:
            item = items[expected_index]
            expected_counts = (len(item.prefix_ids), len(item.suffix_ids[0]), len(item.suffix_ids[1]))
            observed_counts = (record["prefix_tokens"], record["option0_tokens"], record["option1_tokens"])
            if record["label"] != item.label or observed_counts != expected_counts:
                raise PiqaScorerError(f"PIQA result token binding differs at item {expected_index}")
        for option in (0, 1):
            tokens = record[f"option{option}_tokens"]
            total = record[f"option{option}_nll"]
            mean = record[f"option{option}_mean_nll"]
            if type(tokens) is not int or tokens <= 0 or type(total) not in (int, float) or type(mean) not in (int, float):
                raise PiqaScorerError(f"PIQA NLL fields are invalid at item {expected_index}")
            if not math.isfinite(float(total)) or float(total) < 0.0 or not math.isfinite(float(mean)):
                raise PiqaScorerError(f"PIQA NLL is non-finite at item {expected_index}")
            if not math.isclose(float(mean), float(total) / tokens, rel_tol=2e-15, abs_tol=2e-15):
                raise PiqaScorerError(f"PIQA mean/total NLL differs at item {expected_index}")
        expected_mean = 0 if record["option0_mean_nll"] <= record["option1_mean_nll"] else 1
        expected_total = 0 if record["option0_nll"] <= record["option1_nll"] else 1
        if (record["choice_mean_nll"] != expected_mean or record["choice_total_nll"] != expected_total
                or record["correct_mean_nll"] is not (expected_mean == record["label"])
                or record["correct_total_nll"] is not (expected_total == record["label"])):
            raise PiqaScorerError(f"PIQA derived decision fields differ at item {expected_index}")
        results.append(record)
    return results


def _read_runtime(path: Path, expected_items: int, replay_prefix: bool) -> dict[str, Any]:
    try:
        runtime = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PiqaScorerError("PIQA core emitted invalid runtime metadata") from exc
    if (not isinstance(runtime, dict) or runtime.get("schema") != "strat01_gguf_piqa_runtime_v1"
            or runtime.get("items") != expected_items or runtime.get("model_bos_id") != BOS_ID
            or runtime.get("source_token_ids_mode") is not True or runtime.get("kv_reset_per_item") is not True
            or runtime.get("prefix_cache_shared") is not (not replay_prefix)
            or runtime.get("replay_prefix_reference") is not replay_prefix
            or runtime.get("choice_metric") != "mean_suffix_nll" or runtime.get("tie_policy") != "option_0"):
        raise PiqaScorerError("PIQA runtime metadata violates the frozen protocol")
    return runtime


def _clean_pinned_llama(source: Path) -> dict[str, Any]:
    if not (source / "CMakeLists.txt").is_file():
        raise PiqaScorerError("pinned llama.cpp source is unavailable")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, text=True, capture_output=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=source, text=True, capture_output=True, check=False)
    if head.returncode or head.stdout.strip() != PINNED_LLAMA_HEAD or status.returncode or status.stdout.strip():
        raise PiqaScorerError("llama.cpp source is not the clean pinned commit")
    return {"path": str(source), "commit": PINNED_LLAMA_HEAD, "clean": True}


def _cmake_project(source: Path, llama_source: Path) -> str:
    return f'''cmake_minimum_required(VERSION 3.20)
project(strat01_gguf_piqa LANGUAGES C CXX)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
add_subdirectory("{llama_source.as_posix()}" llama-cpp)
add_executable(strat01_gguf_piqa "{source.as_posix()}")
target_compile_features(strat01_gguf_piqa PRIVATE cxx_std_17)
target_link_libraries(strat01_gguf_piqa PRIVATE llama)
'''


def build(binary_dir: Path, llama_source: Path, configuration: str) -> Path:
    _clean_pinned_llama(llama_source)
    if not CORE.is_file():
        raise PiqaScorerError("PIQA C++ core is unavailable")
    binary_dir.mkdir(parents=True, exist_ok=True)
    (binary_dir / "CMakeLists.txt").write_text(_cmake_project(CORE, llama_source), encoding="utf-8", newline="\n")
    configured = subprocess.run(["cmake", "-S", str(binary_dir), "-B", str(binary_dir / "cmake-build")],
                                text=True, capture_output=True, check=False)
    if configured.returncode:
        raise PiqaScorerError("PIQA CMake configure failed; inspect the build directory")
    compiled = subprocess.run(["cmake", "--build", str(binary_dir / "cmake-build"), "--target", "strat01_gguf_piqa",
                               "--config", configuration], text=True, capture_output=True, check=False)
    if compiled.returncode:
        diagnostic = "\n".join(compiled.stdout.splitlines()[-10:] + compiled.stderr.splitlines()[-10:])
        raise PiqaScorerError(f"PIQA CMake build failed; tail follows:\n{diagnostic}")
    candidates = [path for name in ("strat01_gguf_piqa.exe", "strat01_gguf_piqa")
                  for path in (binary_dir / "cmake-build").rglob(name)]
    if len(candidates) != 1:
        raise PiqaScorerError("PIQA build did not produce exactly one scorer binary")
    return candidates[0]


def score(args: argparse.Namespace) -> None:
    if args.n_ctx < 2 or args.n_ctx > MAX_CONTEXT or args.n_batch < 1 or args.n_batch > args.n_ctx:
        raise PiqaScorerError("n-ctx/n-batch violate the bounded PIQA contract")
    for path, label in ((args.binary, "PIQA scorer"), (args.model, "GGUF model"),
                        (args.tokenizer_json, "source tokenizer"), (args.humaneval_gz, "HumanEval source"),
                        (args.piqa_valid, "PIQA source"), (args.piqa_labels, "PIQA labels")):
        if not path.is_file():
            raise PiqaScorerError(f"{label} is unavailable")
    expected_model_sha = MODEL_SHA256.get(args.model.name)
    if expected_model_sha is None or _sha256(args.model) != expected_model_sha:
        raise PiqaScorerError("GGUF is not one of the two pinned STRAT-01 arms")
    if _sha256(args.tokenizer_json) != TOKENIZER_SHA256:
        raise PiqaScorerError("source tokenizer SHA-256 differs from the pinned asset")
    vocab_mapping = _verify_vocab_mapping(args.tokenizer_json, args.model)
    audit = audit_pinned_sources(args.humaneval_gz, args.piqa_valid, args.piqa_labels)
    records: list[PiqaRecord] = audit["piqa"]
    if args.limit is not None:
        if args.limit < 1 or args.limit > PIQA_COUNT:
            raise PiqaScorerError("--limit must be in 1..1838")
        records = records[:args.limit]
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise PiqaScorerError("PIQA scoring requires the tokenizers package") from exc
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    items = prepare_items(records, tokenizer, _special_ids(args.tokenizer_json), args.n_ctx)

    output = args.output
    manifest = args.manifest or output.with_suffix(output.suffix + ".manifest.json")
    runtime_path = output.with_suffix(output.suffix + ".runtime.json")
    token_manifest = output.with_suffix(output.suffix + ".tokens.jsonl")
    exchange = output.with_suffix(output.suffix + ".private.tsv")
    for path, label in ((output, "PIQA score output"), (manifest, "PIQA score manifest"),
                        (runtime_path, "PIQA runtime metadata"), (token_manifest, "PIQA token manifest")):
        _require_new(path, label)
    _write_exchange(exchange, items)
    model_stat = args.model.stat()
    try:
        command = [str(args.binary), "--model", str(args.model), "--input", str(exchange), "--output", str(output),
                   "--metadata-out", str(runtime_path), "--n-ctx", str(args.n_ctx), "--n-batch", str(args.n_batch)]
        if args.threads is not None:
            command.extend(("--threads", str(args.threads)))
        if args.replay_prefix:
            command.append("--replay-prefix")
        completed = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, text=True, check=False)
        if completed.returncode:
            diagnostic = completed.stderr.strip()
            if not diagnostic.startswith("strat01_gguf_piqa: ") or "\n" in diagnostic:
                diagnostic = "unrecognized core diagnostic; inspect the apparatus before retrying"
            raise PiqaScorerError(f"PIQA core failed with exit status {completed.returncode}: {diagnostic}")
    finally:
        if not args.keep_exchange:
            exchange.unlink(missing_ok=True)
    results = _read_results(output, items)
    runtime = _read_runtime(runtime_path, len(items), args.replay_prefix)
    if (model_stat.st_size, model_stat.st_mtime_ns) != (args.model.stat().st_size, args.model.stat().st_mtime_ns):
        raise PiqaScorerError("GGUF changed during PIQA scoring")
    _write_token_manifest(token_manifest, items)
    token_records = _token_manifest_records(items)
    manifest_record = {
        "schema": "strat01_gguf_piqa_manifest_v1",
        "source_audit": audit["sources"],
        "selection": {"items": len(items), "full_gate": len(items) == PIQA_COUNT, "original_order_prefix": True},
        "tokenizer": {"path": str(args.tokenizer_json), "sha256": TOKENIZER_SHA256,
                      "source_gguf_vocab_mapping": vocab_mapping},
        "token_binding": {"path": str(token_manifest), "sha256": _sha256(token_manifest),
                          "records": len(token_records),
                          "all_item_hashes_sha256": hashlib.sha256(
                              json.dumps(token_records, sort_keys=True, separators=(",", ":")).encode("ascii")
                          ).hexdigest()},
        "model": {"path": str(args.model), "sha256": expected_model_sha},
        "scorer": {"binary": str(args.binary), "binary_sha256": _sha256(args.binary),
                   "source": str(CORE), "source_sha256": _sha256(CORE),
                   "llama_cpp": _clean_pinned_llama(args.llama_source)},
        "output": {"path": str(output), "sha256": _sha256(output), "records": len(results)},
        "runtime_metadata": {"path": str(runtime_path), "sha256": _sha256(runtime_path), "record": runtime},
        "protocol": {"format": "Question: {goal}\\nAnswer:", "suffix_leading_space": True,
                     "bos_id": BOS_ID, "eos_appended": False, "separate_prefix_suffix_tokenization": True,
                     "choice": "minimum_mean_suffix_nll", "exact_tie": "option_0",
                     "prefix_cache_shared": not args.replay_prefix},
        "repository_head": _git_head(),
    }
    with manifest.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest_record, handle, sort_keys=True, indent=2)
        handle.write("\n")


def _load_bound_manifest(score_path: Path, expected_model_sha: str) -> dict[str, Any]:
    manifest_path = score_path.with_suffix(score_path.suffix + ".manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PiqaScorerError(f"cannot read bound manifest for {score_path.name}") from exc
    if (manifest.get("schema") != "strat01_gguf_piqa_manifest_v1"
            or manifest.get("selection", {}).get("full_gate") is not True
            or manifest.get("selection", {}).get("items") != PIQA_COUNT
            or manifest.get("model", {}).get("sha256") != expected_model_sha
            or manifest.get("tokenizer", {}).get("sha256") != TOKENIZER_SHA256
            or manifest.get("output", {}).get("sha256") != _sha256(score_path)):
        raise PiqaScorerError(f"score manifest binding failed for {score_path.name}")
    return manifest


def adjudicate(args: argparse.Namespace) -> None:
    _require_new(args.output, "PIQA adjudication report")
    teacher = _read_results(args.teacher, require_full=True)
    candidate = _read_results(args.candidate, require_full=True)
    teacher_manifest = _load_bound_manifest(args.teacher, MODEL_SHA256["GigaChat3.1-10B-A1.8B-bf16.gguf"])
    candidate_manifest = _load_bound_manifest(args.candidate, MODEL_SHA256["GigaChat3.1-10B-A1.8B-q4_K_M.gguf"])
    if (teacher_manifest["token_binding"]["all_item_hashes_sha256"]
            != candidate_manifest["token_binding"]["all_item_hashes_sha256"]):
        raise PiqaScorerError("teacher and candidate PIQA token bindings differ")
    teacher_correct = [record["correct_mean_nll"] for record in teacher]
    candidate_correct = [record["correct_mean_nll"] for record in candidate]
    paired = paired_accuracy_bootstrap(teacher_correct, candidate_correct, expected_count=PIQA_COUNT)
    teacher_count = sum(teacher_correct)
    candidate_count = sum(candidate_correct)
    required_candidate = math.ceil(0.98 * teacher_count)
    if teacher_count <= PIQA_COUNT / 2:
        status = "INCONCLUSIVE_NO_BASELINE_ABILITY"
    elif candidate_count >= required_candidate:
        status = "PASS_PIQA"
    else:
        status = "FAIL_PIQA"
    report = {
        "schema": "strat01_gigachat_piqa_adjudication_v1",
        "status": status,
        "items": PIQA_COUNT,
        "teacher": {"path": str(args.teacher), "sha256": _sha256(args.teacher), "correct": teacher_count,
                    "accuracy": teacher_count / PIQA_COUNT},
        "candidate": {"path": str(args.candidate), "sha256": _sha256(args.candidate), "correct": candidate_count,
                      "accuracy": candidate_count / PIQA_COUNT},
        "gate": {"teacher_must_exceed_accuracy": 0.5, "candidate_fraction_of_teacher": 0.98,
                 "candidate_required_correct": required_candidate, "passed": status == "PASS_PIQA"},
        "paired_bootstrap_descriptive": paired,
        "bindings": {"token_ids": teacher_manifest["token_binding"]["all_item_hashes_sha256"],
                     "tokenizer_sha256": TOKENIZER_SHA256,
                     "teacher_manifest_sha256": _sha256(args.teacher.with_suffix(args.teacher.suffix + ".manifest.json")),
                     "candidate_manifest_sha256": _sha256(args.candidate.with_suffix(args.candidate.suffix + ".manifest.json"))},
        "protocol": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                     "bootstrap_is_descriptive_not_gate": True},
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, sort_keys=True, indent=2)
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


class PiqaScorerTests(unittest.TestCase):
    def test_preparation_keeps_prefix_and_suffix_separate(self) -> None:
        records = [PiqaRecord(0, "g", "a", "bb", 1)]
        prepared = prepare_items(records, FakeTokenizer(), set(), 64)
        self.assertEqual(prepared[0].prefix_ids, tuple(map(ord, "Question: g\nAnswer:")))
        self.assertEqual(prepared[0].suffix_ids[0], tuple(map(ord, " a")))
        self.assertEqual(prepared[0].suffix_ids[1], tuple(map(ord, " bb")))
        self.assertEqual(len(prepared[0].token_ids_sha256), 64)

    def test_context_bound_includes_bos(self) -> None:
        records = [PiqaRecord(0, "g", "a", "b", 0)]
        full = 1 + len("Question: g\nAnswer:") + len(" a")
        prepare_items(records, FakeTokenizer(), set(), full)
        with self.assertRaisesRegex(PiqaScorerError, "no-truncation"):
            prepare_items(records, FakeTokenizer(), set(), full - 1)

    def test_exchange_contains_ids_not_text(self) -> None:
        item = PreparedItem(0, 1, (10, 11), ((20,), (21, 22)), "a" * 64)
        with tempfile.TemporaryDirectory(prefix="strat01-piqa-") as directory:
            path = Path(directory) / "private.tsv"
            _write_exchange(path, [item])
            self.assertEqual(path.read_text(encoding="ascii"), "0\t1\t10,11\t20\t21,22\n")

    def test_core_contains_alignment_and_prefix_replay_guards(self) -> None:
        source = CORE.read_text(encoding="utf-8")
        for required in ("suffix[target_index - 1]", "suffix[first_target + offset]", "llama_memory_seq_cp",
                         "--replay-prefix", "tie_policy\\\":\\\"option_0", "llama_memory_clear"):
            self.assertIn(required, source)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="STRAT-01 GigaChat GGUF PIQA apparatus")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--build", action="store_true")
    modes.add_argument("--score", action="store_true")
    modes.add_argument("--adjudicate", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=Path(tempfile.gettempdir()) / "siliconllm-strat01-gguf-piqa-build")
    parser.add_argument("--llama-source", type=Path, default=PINNED_LLAMA)
    parser.add_argument("--configuration", default="Release")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--humaneval-gz", type=Path)
    parser.add_argument("--piqa-valid", type=Path)
    parser.add_argument("--piqa-labels", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--n-ctx", type=int, default=MAX_CONTEXT)
    parser.add_argument("--n-batch", type=int, default=128)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--limit", type=int, help="apparatus-only original-order prefix; full gate omits this")
    parser.add_argument("--replay-prefix", action="store_true", help="reference mode: independently replay each option prefix")
    parser.add_argument("--keep-exchange", action="store_true")
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args(argv)
    if args.build:
        print(build(args.build_dir, args.llama_source, args.configuration))
        return 0
    if args.score:
        required = (args.binary, args.model, args.tokenizer_json, args.humaneval_gz,
                    args.piqa_valid, args.piqa_labels, args.output)
        if any(path is None for path in required):
            parser.error("--score requires --binary --model --tokenizer-json --humaneval-gz --piqa-valid --piqa-labels --output")
        score(args)
        return 0
    if args.adjudicate:
        if args.teacher is None or args.candidate is None or args.output is None:
            parser.error("--adjudicate requires --teacher --candidate --output")
        adjudicate(args)
        return 0
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PiqaScorerTests)
    outcome = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if outcome.wasSuccessful() else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PiqaScorerError as exc:
        print(f"strat01_gguf_piqa: {exc}", file=sys.stderr)
        raise SystemExit(2)
