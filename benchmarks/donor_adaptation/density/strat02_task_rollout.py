#!/usr/bin/env python3
"""Frozen STRAT-02 task and rollout primitives (no model loader, no code execution).

The command line intentionally has no forward path.  It can only audit the
three pinned external sources or run planted, in-memory tests.  Callers which
have *already* loaded the exact eval model and tokenizer may use the pure
functions below to obtain JSON-serialisable per-item artifacts.

HumanEval completions are never imported, compiled, parsed as Python, or run
by this module.  Their functional result is consequently always
``PENDING_SANDBOX`` until a separately verified isolated executor adjudicates
the saved artifacts.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence


EOS_ID = 100257
MAX_CONTEXT = 4096
HUMANEVAL_COUNT = 164
PIQA_COUNT = 1838
ROLLOUT_COUNT = 96
ROLLOUT_PREFIX_TOKENS = 256
HUMANEVAL_MAX_NEW_TOKENS = 512
ROLLOUT_MAX_NEW_TOKENS = 256
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20_260_916
ROLLOUT_CATEGORIES = ("code", "technical_general", "prose")
PIQA_LOGIT_CHUNK = 32


class ProtocolError(ValueError):
    """An input violates a frozen STRAT-02 requirement."""


class SourceAuditError(ProtocolError):
    """A pinned benchmark file is absent, altered, malformed, or miscounted."""


class NumericFailure(RuntimeError):
    """A model produced non-finite logits or an invalid vocabulary result."""


@dataclass(frozen=True)
class SourcePins:
    humaneval_gz_sha256: str
    piqa_valid_sha256: str
    piqa_labels_sha256: str
    humaneval_count: int = HUMANEVAL_COUNT
    piqa_count: int = PIQA_COUNT


PINNED_SOURCES = SourcePins(
    humaneval_gz_sha256="b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef",
    piqa_valid_sha256="93503cc97c679e459b065c3d13e848282e44b2a25213985bed3e5d458abef72d",
    piqa_labels_sha256="b4192dc3a2a0363d9d60ccf79800cbbe2f32ebb17726efdde6970e0b8131bceb",
)


@dataclass(frozen=True)
class HumanEvalRecord:
    task_id: str
    prompt: str


@dataclass(frozen=True)
class PiqaRecord:
    index: int
    goal: str
    sol1: str
    sol2: str
    label: int


@dataclass(frozen=True)
class OptionScore:
    option_index: int
    suffix_ids: tuple[int, ...]
    nll_total: float
    nll_mean: float

    def json_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PiqaArtifact:
    index: int
    label: int
    prefix_ids: tuple[int, ...]
    options: tuple[OptionScore, OptionScore]
    choice_mean_nll: int
    choice_total_nll: int
    correct_mean_nll: bool
    correct_total_nll: bool

    def json_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationArtifact:
    item_id: str
    prompt_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    decoded_text: str
    emitted_utf8_bytes: int
    stop_reason: str
    elapsed_seconds: float
    numeric_error: str | None = None
    run32: bool = False
    loop8x3: bool = False
    empty: bool = False

    def json_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HumanEvalArtifact:
    task_id: str
    prompt: str
    generation: GenerationArtifact
    functional_status: str = "PENDING_SANDBOX"
    functional_passed: None = None

    def json_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RolloutArtifact:
    source_document_id: str
    category: str
    generation: GenerationArtifact

    @property
    def degenerate(self) -> bool:
        return self.generation.run32 or self.generation.loop8x3 or self.generation.empty

    def json_record(self) -> dict[str, Any]:
        result = asdict(self)
        result["degenerate"] = self.degenerate
        return result


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError("STRAT-02 task scoring requires PyTorch for model calls") from exc
    return torch


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise SourceAuditError(f"cannot read {label} at {path}: {exc}") from exc


def _require_ids(ids: Any, label: str, *, allow_empty: bool = False) -> list[int]:
    if not isinstance(ids, (list, tuple)):
        raise ProtocolError(f"{label} must be a sequence of token IDs")
    checked = list(ids)
    if not checked and not allow_empty:
        raise ProtocolError(f"{label} must not be empty")
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in checked):
        raise ProtocolError(f"{label} must contain only non-negative integer token IDs")
    return checked


def _parse_jsonl(raw: bytes, label: str) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceAuditError(f"{label} is not UTF-8") from exc
    if not text or not text.endswith("\n"):
        raise SourceAuditError(f"{label} must be a non-empty newline-terminated JSONL file")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line:
            raise SourceAuditError(f"{label}:{number}: blank JSONL line")
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SourceAuditError(f"{label}:{number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(item, dict):
            raise SourceAuditError(f"{label}:{number}: each JSONL item must be an object")
        records.append(item)
    return records


def _parse_humaneval(raw_gz: bytes) -> list[HumanEvalRecord]:
    try:
        raw = gzip.decompress(raw_gz)
    except (OSError, EOFError) as exc:
        raise SourceAuditError("HumanEval.jsonl.gz is not a valid gzip stream") from exc
    parsed = _parse_jsonl(raw, "HumanEval.jsonl.gz payload")
    records: list[HumanEvalRecord] = []
    seen: set[str] = set()
    for number, item in enumerate(parsed, 1):
        task_id, prompt = item.get("task_id"), item.get("prompt")
        if not isinstance(task_id, str) or not task_id:
            raise SourceAuditError(f"HumanEval:{number}: task_id must be a non-empty string")
        if task_id in seen:
            raise SourceAuditError(f"HumanEval:{number}: duplicate task_id {task_id!r}")
        if not isinstance(prompt, str):
            raise SourceAuditError(f"HumanEval:{number}: prompt must be a string")
        seen.add(task_id)
        records.append(HumanEvalRecord(task_id=task_id, prompt=prompt))
    return records


def _parse_piqa(valid_raw: bytes, labels_raw: bytes) -> list[PiqaRecord]:
    records = _parse_jsonl(valid_raw, "PIQA valid.jsonl")
    try:
        labels_text = labels_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceAuditError("PIQA valid-labels.lst is not UTF-8") from exc
    if not labels_text or not labels_text.endswith("\n"):
        raise SourceAuditError("PIQA valid-labels.lst must be non-empty and newline-terminated")
    labels: list[int] = []
    for number, line in enumerate(labels_text.splitlines(), 1):
        if line not in {"0", "1"}:
            raise SourceAuditError(f"PIQA labels:{number}: expected exactly 0 or 1")
        labels.append(int(line))
    if len(records) != len(labels):
        raise SourceAuditError(f"PIQA records/labels length mismatch ({len(records)} != {len(labels)})")
    output: list[PiqaRecord] = []
    for index, (item, label) in enumerate(zip(records, labels, strict=True)):
        values = (item.get("goal"), item.get("sol1"), item.get("sol2"))
        if not all(isinstance(value, str) for value in values):
            raise SourceAuditError(f"PIQA:{index + 1}: goal, sol1, and sol2 must all be strings")
        output.append(PiqaRecord(index=index, goal=values[0], sol1=values[1], sol2=values[2], label=label))
    return output


def audit_pinned_sources(
    humaneval_gz: Path,
    piqa_valid: Path,
    piqa_labels: Path,
    *,
    pins: SourcePins = PINNED_SOURCES,
) -> dict[str, Any]:
    """Read, hash, and structurally count the three immutable external files.

    This routine is read-only: it neither downloads nor caches benchmark data.
    It returns parsed records only for callers that deliberately need them;
    callers should persist no benchmark text in this repository.
    """
    human_raw = _read_bytes(humaneval_gz, "HumanEval.jsonl.gz")
    valid_raw = _read_bytes(piqa_valid, "PIQA valid.jsonl")
    labels_raw = _read_bytes(piqa_labels, "PIQA valid-labels.lst")
    observed = {
        "humaneval_gz": _sha256(human_raw),
        "piqa_valid": _sha256(valid_raw),
        "piqa_labels": _sha256(labels_raw),
    }
    expected = {
        "humaneval_gz": pins.humaneval_gz_sha256,
        "piqa_valid": pins.piqa_valid_sha256,
        "piqa_labels": pins.piqa_labels_sha256,
    }
    mismatches = [name for name in expected if observed[name] != expected[name]]
    if mismatches:
        detail = ", ".join(f"{name} expected {expected[name]}, got {observed[name]}" for name in mismatches)
        raise SourceAuditError(f"pinned source SHA-256 mismatch: {detail}")
    human = _parse_humaneval(human_raw)
    piqa = _parse_piqa(valid_raw, labels_raw)
    if len(human) != pins.humaneval_count:
        raise SourceAuditError(f"HumanEval requires {pins.humaneval_count} records; got {len(human)}")
    if len(piqa) != pins.piqa_count:
        raise SourceAuditError(f"PIQA requires {pins.piqa_count} records; got {len(piqa)}")
    return {
        "ok": True,
        "protocol": "STRAT-02 Stage 0 task rollout",
        "sources": {
            "humaneval_gz": {"path": str(humaneval_gz), "sha256": observed["humaneval_gz"], "records": len(human)},
            "piqa_valid": {"path": str(piqa_valid), "sha256": observed["piqa_valid"], "records": len(piqa)},
            "piqa_labels": {"path": str(piqa_labels), "sha256": observed["piqa_labels"], "records": len(piqa)},
        },
        "humaneval": human,
        "piqa": piqa,
    }


def validate_strat02_tokenizer(tokenizer: Any) -> None:
    """Require the exact tokenizer class, versions, EOS, and native context."""
    try:
        import tokenizers
        import transformers
        from transformers import GPT2TokenizerFast
    except ImportError as exc:  # pragma: no cover - actionable outside test env.
        raise RuntimeError("STRAT-02 requires transformers and tokenizers") from exc
    if transformers.__version__ != "4.57.1" or tokenizers.__version__ != "0.22.2":
        raise RuntimeError(
            "STRAT-02 requires transformers==4.57.1 and tokenizers==0.22.2; "
            f"got {transformers.__version__} and {tokenizers.__version__}"
        )
    if type(tokenizer) is not GPT2TokenizerFast:
        raise RuntimeError("STRAT-02 requires the exact GPT2TokenizerFast class")
    if tokenizer.bos_token_id is not None or tokenizer.eos_token_id != EOS_ID:
        raise RuntimeError("STRAT-02 tokenizer must have bos=None and eos=100257")
    if tokenizer.model_max_length != MAX_CONTEXT:
        raise RuntimeError("STRAT-02 tokenizer model_max_length must be 4096")


def _encode(tokenizer: Any, text: str, label: str) -> list[int]:
    if not isinstance(text, str):
        raise ProtocolError(f"{label} must be a string")
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception as exc:
        raise ProtocolError(f"tokenizer failed to encode {label}: {exc}") from exc
    result = _require_ids(ids, label)
    if EOS_ID in result or set(result).intersection(getattr(tokenizer, "all_special_ids", ())):
        raise ProtocolError(f"{label} contains a special token ID")
    return result


def _model_device(model: Any, torch: Any) -> Any:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        weight = getattr(getattr(model, "lm_head", None), "weight", None)
        return weight.device if weight is not None else torch.device("cpu")


def _tokenizer_vocab_size(tokenizer: Any) -> int | None:
    """Return a declared tokenizer vocabulary size when the API exposes one."""
    value = getattr(tokenizer, "vocab_size", None)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    getter = getattr(tokenizer, "get_vocab", None)
    if callable(getter):
        vocabulary = getter()
        if isinstance(vocabulary, Mapping) and vocabulary:
            return max(vocabulary.values()) + 1
    return None


def _last_hidden_state(output: Any) -> Any:
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, (tuple, list)) and output:
        hidden = output[0]
    if hidden is None:
        raise ProtocolError("model.model did not return last_hidden_state")
    return hidden


def _require_eval_causal_model(model: Any) -> None:
    if getattr(model, "training", None) is not False:
        raise ProtocolError("model must be in eval mode before STRAT-02 task scoring")
    if not callable(getattr(model, "model", None)) or not callable(getattr(model, "lm_head", None)):
        raise TypeError("expected an eval CausalLM-like model exposing model and lm_head")


def _forward_hidden(model: Any, input_ids: Sequence[int]) -> Any:
    torch = _torch()
    _require_eval_causal_model(model)
    ids = _require_ids(input_ids, "model input")
    if len(ids) > MAX_CONTEXT:
        raise ProtocolError(f"native context exceeded: {len(ids)} > {MAX_CONTEXT}")
    device = _model_device(model, torch)
    inputs = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        output = model.model(input_ids=inputs, use_cache=False, return_dict=True)
        hidden = _last_hidden_state(output)
    if hidden.ndim != 3 or hidden.shape[:2] != (1, len(ids)):
        raise ProtocolError("model.model returned hidden states with an unexpected shape")
    return hidden, output


def _nll_for_targets(
    model: Any, hidden: Any, start: int, targets: Sequence[int], *, chunk_size: int = PIQA_LOGIT_CHUNK
) -> tuple[float, list[float]]:
    """Score causal suffix targets in bounded chunks of vocabulary logits.

    A chunk has at most ``[1, 32, vocab]`` float32 logits (about 12.8 MB for
    a 100k vocabulary), avoiding both a full-sequence head projection and one
    Python/linear-head call per suffix token.
    """
    torch = _torch()
    target_ids = _require_ids(targets, "suffix token IDs")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ProtocolError("PIQA logits chunk_size must be a positive integer")
    if start < 0 or start + len(target_ids) > hidden.shape[1]:
        raise ProtocolError("suffix target positions are outside returned hidden states")
    total = 0.0
    per_token: list[float] = []
    with torch.inference_mode(), torch.autocast(device_type=hidden.device.type, enabled=False):
        for offset in range(0, len(target_ids), chunk_size):
            stop = min(offset + chunk_size, len(target_ids))
            targets_chunk = target_ids[offset:stop]
            logits = model.lm_head(hidden[:, start + offset : start + stop, :])
            if (
                logits.ndim != 3
                or logits.shape[:2] != (1, len(targets_chunk))
                or logits.shape[-1] <= max(targets_chunk)
            ):
                raise ProtocolError("lm_head returned invalid shape or target is outside its vocabulary")
            logits = logits.to(dtype=torch.float32)
            if not bool(torch.isfinite(logits).all().item()):
                raise NumericFailure("non-finite LM-head logits")
            target_tensor = torch.tensor([targets_chunk], dtype=torch.long, device=logits.device)
            selected = logits.gather(-1, target_tensor.unsqueeze(-1)).squeeze(-1)
            nll = torch.logsumexp(logits, dim=-1) - selected
            if not bool(torch.isfinite(nll).all().item()):
                raise NumericFailure("non-finite token NLL")
            total += float(nll.to(dtype=torch.float64).sum(dtype=torch.float64).item())
            per_token.extend(float(value) for value in nll[0].detach().cpu().tolist())
            del logits, target_tensor, selected, nll
    if not math.isfinite(total):
        raise NumericFailure("non-finite total NLL")
    return total, per_token


def score_piqa_option(
    model: Any,
    tokenizer: Any,
    goal: str,
    solution: str,
    *,
    option_index: int,
    verify_tokenizer: bool = True,
) -> tuple[tuple[int, ...], OptionScore]:
    """Score one PIQA option with separately tokenized fixed prefix/suffix."""
    if option_index not in (0, 1):
        raise ProtocolError("PIQA option_index must be 0 or 1")
    if verify_tokenizer:
        validate_strat02_tokenizer(tokenizer)
    prefix = _encode(tokenizer, f"Question: {goal}\nAnswer:", "PIQA prefix")
    suffix = _encode(tokenizer, f" {solution}", "PIQA suffix")
    # The full conceptual sequence is EOS + prefix + suffix; never merge BPE
    # strings before encoding, and never append an EOS answer token.
    if 1 + len(prefix) + len(suffix) > MAX_CONTEXT:
        raise ProtocolError("PIQA EOS + separately encoded prefix + suffix exceeds native context 4096")
    inputs = [EOS_ID, *prefix, *suffix[:-1]]
    hidden, output = _forward_hidden(model, inputs)
    try:
        nll_total, _ = _nll_for_targets(model, hidden, len(prefix), suffix)
    finally:
        del hidden, output
    return tuple(prefix), OptionScore(
        option_index=option_index,
        suffix_ids=tuple(suffix),
        nll_total=nll_total,
        nll_mean=nll_total / len(suffix),
    )


def score_piqa_item(
    model: Any, tokenizer: Any, record: PiqaRecord, *, verify_tokenizer: bool = True
) -> PiqaArtifact:
    if record.label not in (0, 1):
        raise ProtocolError("PIQA label must be 0 or 1")
    prefix0, option0 = score_piqa_option(
        model, tokenizer, record.goal, record.sol1, option_index=0, verify_tokenizer=verify_tokenizer
    )
    prefix1, option1 = score_piqa_option(
        model, tokenizer, record.goal, record.sol2, option_index=1, verify_tokenizer=False
    )
    if prefix0 != prefix1:
        raise ProtocolError("PIQA separately encoded prefix changed between options")
    # ``<=`` intentionally makes exact ties option 0, without floating epsilon.
    mean_choice = 0 if option0.nll_mean <= option1.nll_mean else 1
    total_choice = 0 if option0.nll_total <= option1.nll_total else 1
    return PiqaArtifact(
        index=record.index,
        label=record.label,
        prefix_ids=prefix0,
        options=(option0, option1),
        choice_mean_nll=mean_choice,
        choice_total_nll=total_choice,
        correct_mean_nll=(mean_choice == record.label),
        correct_total_nll=(total_choice == record.label),
    )


def score_piqa_records(
    model: Any, tokenizer: Any, records: Sequence[PiqaRecord], *, verify_tokenizer: bool = True
) -> list[PiqaArtifact]:
    if len(records) != PIQA_COUNT:
        raise ProtocolError(f"PIQA requires exactly {PIQA_COUNT} records; got {len(records)}")
    if verify_tokenizer:
        validate_strat02_tokenizer(tokenizer)
    result = [score_piqa_item(model, tokenizer, record, verify_tokenizer=False) for record in records]
    if [item.index for item in result] != list(range(PIQA_COUNT)):
        raise ProtocolError("PIQA records must remain in pinned original order 0..1837")
    return result


def _forward_cached_hidden(model: Any, input_ids: Sequence[int], past_key_values: Any = None) -> tuple[Any, Any]:
    """Run either the one-time prefill or one cached token update."""
    torch = _torch()
    _require_eval_causal_model(model)
    ids = _require_ids(input_ids, "cached model input")
    if len(ids) > MAX_CONTEXT:
        raise ProtocolError(f"native context exceeded: {len(ids)} > {MAX_CONTEXT}")
    if past_key_values is not None and len(ids) != 1:
        raise ProtocolError("cached generation updates must process exactly one token")
    device = _model_device(model, torch)
    inputs = torch.tensor([ids], dtype=torch.long, device=device)
    kwargs: dict[str, Any] = {"input_ids": inputs, "use_cache": True, "return_dict": True}
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        output = model.model(**kwargs)
        hidden = _last_hidden_state(output)
    if hidden.ndim != 3 or hidden.shape[:2] != (1, len(ids)):
        raise ProtocolError("model.model returned cached hidden states with an unexpected shape")
    if getattr(output, "past_key_values", None) is None:
        raise ProtocolError("VOID_APPARATUS: model.model did not return past_key_values with use_cache=True")
    return hidden, output


def _greedy_id_from_last_hidden(model: Any, hidden: Any) -> int:
    """Return the least ID among finite equal maxima from one final hidden state."""
    torch = _torch()
    with torch.inference_mode(), torch.autocast(device_type=hidden.device.type, enabled=False):
        logits = model.lm_head(hidden[:, -1:, :])
        if logits.ndim != 3 or logits.shape[:2] != (1, 1) or logits.shape[-1] <= EOS_ID:
            raise ProtocolError("lm_head returned invalid generation logits or lacks EOS ID")
        logits = logits.to(dtype=torch.float32)[0, 0]
        if not bool(torch.isfinite(logits).all().item()):
            raise NumericFailure("non-finite generation logits")
        maximum = logits.max()
        candidates = torch.nonzero(logits == maximum, as_tuple=False).flatten()
        if candidates.numel() == 0:  # Defensive: finite logits always have a max.
            raise NumericFailure("no finite argmax candidate")
        choice = int(candidates.min().item())
        if choice < 0 or choice >= logits.shape[0]:
            raise NumericFailure("stable argmax selected an out-of-vocabulary ID")
        return choice


def rollout_flags(ids: Sequence[int]) -> tuple[bool, bool, bool]:
    """Return (run32, loop8x3, empty) over only non-EOS generated IDs."""
    tokens = _require_ids(ids, "generated token IDs", allow_empty=True)
    if EOS_ID in tokens:
        raise ProtocolError("rollout_flags expects generated IDs with EOS already removed")
    run32 = any(all(tokens[start + offset] == tokens[start] for offset in range(32)) for start in range(max(0, len(tokens) - 31)))
    loop8x3 = any(
        tokens[start : start + 8] == tokens[start + 8 : start + 16] == tokens[start + 16 : start + 24]
        for start in range(max(0, len(tokens) - 23))
    )
    return run32, loop8x3, not tokens


def generate_greedy(
    model: Any,
    tokenizer: Any,
    *,
    item_id: str,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    rollout_checks: bool,
) -> GenerationArtifact:
    """Greedily generate from one cached prefill plus one-token cache updates."""
    if not isinstance(item_id, str) or not item_id:
        raise ProtocolError("item_id must be a non-empty string")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ProtocolError("max_new_tokens must be a positive integer")
    prompt = _require_ids(prompt_ids, "generation prompt")
    if len(prompt) + max_new_tokens > MAX_CONTEXT:
        raise ProtocolError(
            f"generation prompt plus fixed cap exceeds native context: {len(prompt)} + {max_new_tokens} > {MAX_CONTEXT}"
        )
    start = time.perf_counter()
    generated: list[int] = []
    stop_reason = "CAP"
    numeric_error: str | None = None
    hidden, output = _forward_cached_hidden(model, prompt)
    cache = getattr(output, "past_key_values")
    try:
        while len(generated) < max_new_tokens:
            try:
                token = _greedy_id_from_last_hidden(model, hidden)
            except NumericFailure as exc:
                stop_reason = "FAIL_NUMERIC"
                numeric_error = str(exc)
                break
            vocabulary_size = _tokenizer_vocab_size(tokenizer)
            if vocabulary_size is not None and token >= vocabulary_size:
                stop_reason = "FAIL_NUMERIC"
                numeric_error = f"generation selected token ID {token} outside tokenizer vocabulary {vocabulary_size}"
                break
            if token == EOS_ID:
                stop_reason = "EOS"
                break
            generated.append(token)
            if len(generated) == max_new_tokens:
                break
            # The cache represents the prompt plus prior emitted tokens.  Feed
            # exactly the latest emitted token once to obtain the next logits.
            del hidden, output
            hidden, output = _forward_cached_hidden(model, [token], cache)
            cache = getattr(output, "past_key_values")
    finally:
        try:
            del hidden, output, cache
        except UnboundLocalError:  # pragma: no cover - defensive cleanup only.
            pass
    try:
        decoded = tokenizer.decode(generated, clean_up_tokenization_spaces=False)
    except Exception as exc:
        raise ProtocolError(f"tokenizer failed to decode generated IDs: {exc}") from exc
    if not isinstance(decoded, str):
        raise ProtocolError("tokenizer.decode must return a string")
    run32, loop8x3, empty = rollout_flags(generated) if rollout_checks else (False, False, not generated)
    return GenerationArtifact(
        item_id=item_id,
        prompt_ids=tuple(prompt),
        generated_ids=tuple(generated),
        decoded_text=decoded,
        emitted_utf8_bytes=len(decoded.encode("utf-8")),
        stop_reason=stop_reason,
        elapsed_seconds=time.perf_counter() - start,
        numeric_error=numeric_error,
        run32=run32,
        loop8x3=loop8x3,
        empty=empty,
    )


def generate_humaneval(
    model: Any, tokenizer: Any, records: Sequence[HumanEvalRecord], *, verify_tokenizer: bool = True
) -> list[HumanEvalArtifact]:
    """Produce raw 164-item HumanEval artifacts; never execute their source."""
    if len(records) != HUMANEVAL_COUNT:
        raise ProtocolError(f"HumanEval requires exactly {HUMANEVAL_COUNT} records; got {len(records)}")
    if verify_tokenizer:
        validate_strat02_tokenizer(tokenizer)
    output: list[HumanEvalArtifact] = []
    seen: set[str] = set()
    for record in records:
        if record.task_id in seen:
            raise ProtocolError(f"duplicate HumanEval task_id {record.task_id!r}")
        seen.add(record.task_id)
        prompt = _encode(tokenizer, record.prompt, f"HumanEval prompt {record.task_id}")
        if 1 + len(prompt) > MAX_CONTEXT:
            raise ProtocolError(f"HumanEval {record.task_id}: EOS + prompt exceeds native context 4096")
        generation = generate_greedy(
            model, tokenizer, item_id=record.task_id, prompt_ids=[EOS_ID, *prompt],
            max_new_tokens=HUMANEVAL_MAX_NEW_TOKENS, rollout_checks=False,
        )
        output.append(HumanEvalArtifact(task_id=record.task_id, prompt=record.prompt, generation=generation))
    return output


def generate_document_rollouts(
    model: Any, tokenizer: Any, documents: Sequence[Mapping[str, Any]], *, verify_tokenizer: bool = True
) -> list[RolloutArtifact]:
    """Generate the fixed 256-token continuation for each 96-item heldout prefix."""
    if len(documents) != ROLLOUT_COUNT:
        raise ProtocolError(f"document rollout requires exactly {ROLLOUT_COUNT} heldout documents; got {len(documents)}")
    if verify_tokenizer:
        validate_strat02_tokenizer(tokenizer)
    output: list[RolloutArtifact] = []
    seen: set[str] = set()
    for number, document in enumerate(documents, 1):
        identifier, category, text = document.get("source_document_id"), document.get("category"), document.get("text")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ProtocolError(f"heldout:{number}: source_document_id must be unique and non-empty")
        if category not in ROLLOUT_CATEGORIES:
            raise ProtocolError(f"heldout:{number}: invalid category {category!r}")
        seen.add(identifier)
        payload = _encode(tokenizer, text, f"heldout text {identifier}")
        if len(payload) < ROLLOUT_PREFIX_TOKENS:
            raise ProtocolError(f"heldout {identifier}: payload has fewer than {ROLLOUT_PREFIX_TOKENS} tokens")
        prefix = payload[:ROLLOUT_PREFIX_TOKENS]
        generation = generate_greedy(
            model, tokenizer, item_id=identifier, prompt_ids=[EOS_ID, *prefix],
            max_new_tokens=ROLLOUT_MAX_NEW_TOKENS, rollout_checks=True,
        )
        output.append(RolloutArtifact(source_document_id=identifier, category=category, generation=generation))
    return output


def paired_accuracy_bootstrap(
    teacher_correct: Sequence[bool], candidate_correct: Sequence[bool], *, expected_count: int
) -> dict[str, Any]:
    """Fixed-seed 20k paired bootstrap for a task accuracy difference."""
    if len(teacher_correct) != expected_count or len(candidate_correct) != expected_count:
        raise ProtocolError(f"paired accuracy requires exactly {expected_count} items per arm")
    if any(type(value) is not bool for value in [*teacher_correct, *candidate_correct]):
        raise ProtocolError("paired accuracy inputs must be bool, not numeric truthy values")
    import numpy as np

    teacher = np.asarray(teacher_correct, dtype=np.float64)
    candidate = np.asarray(candidate_correct, dtype=np.float64)
    delta = candidate - teacher
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # Do not materialize a 20,000 x 1,838 index matrix for PIQA.  Chunking
    # preserves the same deterministic RNG stream and all 20,000 draws.
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    draw_chunk = 256
    for start in range(0, BOOTSTRAP_DRAWS, draw_chunk):
        stop = min(start + draw_chunk, BOOTSTRAP_DRAWS)
        sampled = rng.integers(0, expected_count, size=(stop - start, expected_count), endpoint=False)
        draws[start:stop] = delta[sampled].mean(axis=1, dtype=np.float64)
    return {
        "items": expected_count,
        "teacher_correct": int(teacher.sum(dtype=np.float64)),
        "candidate_correct": int(candidate.sum(dtype=np.float64)),
        "teacher_accuracy": float(teacher.mean()),
        "candidate_accuracy": float(candidate.mean()),
        "point_delta_accuracy": float(delta.mean()),
        "paired_bootstrap_ci95_delta_accuracy": [
            float(np.quantile(draws, 0.025, method="linear")),
            float(np.quantile(draws, 0.975, method="linear")),
        ],
        "protocol": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "resampling": "paired_items_with_replacement",
            "quantile_method": "linear",
        },
    }


def _canonical_piqa_correctness(item: PiqaArtifact, expected_index: int) -> tuple[bool, bool]:
    """Recompute choices from raw option NLLs and reject altered derived fields."""
    if item.index != expected_index or item.label not in (0, 1):
        raise ProtocolError(f"PIQA artifact mismatch at index {expected_index}")
    _require_ids(item.prefix_ids, f"PIQA {expected_index} prefix")
    if len(item.options) != 2:
        raise ProtocolError(f"PIQA {expected_index}: expected exactly two option artifacts")
    for option_index, option in enumerate(item.options):
        if option.option_index != option_index:
            raise ProtocolError(f"PIQA {expected_index}: option IDs are not 0, 1 in order")
        suffix = _require_ids(option.suffix_ids, f"PIQA {expected_index} option {option_index} suffix")
        if (
            isinstance(option.nll_total, bool)
            or isinstance(option.nll_mean, bool)
            or not isinstance(option.nll_total, (int, float))
            or not isinstance(option.nll_mean, (int, float))
            or not (math.isfinite(option.nll_total) and math.isfinite(option.nll_mean) and option.nll_total >= 0)
        ):
            raise ProtocolError(f"PIQA {expected_index}: option NLLs must be finite and total NLL non-negative")
        expected_mean = option.nll_total / len(suffix)
        if not math.isclose(option.nll_mean, expected_mean, rel_tol=0.0, abs_tol=0.0):
            raise ProtocolError(f"PIQA {expected_index}: stored mean NLL does not match raw total/length")
    mean_choice = 0 if item.options[0].nll_mean <= item.options[1].nll_mean else 1
    total_choice = 0 if item.options[0].nll_total <= item.options[1].nll_total else 1
    mean_correct, total_correct = mean_choice == item.label, total_choice == item.label
    if (
        item.choice_mean_nll != mean_choice
        or item.choice_total_nll != total_choice
        or item.correct_mean_nll is not mean_correct
        or item.correct_total_nll is not total_correct
    ):
        raise ProtocolError(f"PIQA {expected_index}: stored choices/correctness do not match raw option scores and label")
    return mean_correct, total_correct


def _validate_generation_artifact(
    artifact: GenerationArtifact, *, expected_item_id: str, rollout_checks: bool
) -> bool:
    """Validate IDs and recompute degeneracy fields before any arm verdict."""
    if artifact.item_id != expected_item_id:
        raise ProtocolError(f"generation item ID mismatch ({artifact.item_id!r} != {expected_item_id!r})")
    prompt = _require_ids(artifact.prompt_ids, f"generation {expected_item_id} prompt IDs")
    if prompt[0] != EOS_ID:
        raise ProtocolError(f"generation {expected_item_id}: persisted prompt must start with the EOS prefix")
    generated = _require_ids(artifact.generated_ids, f"generation {expected_item_id} IDs", allow_empty=True)
    if EOS_ID in generated:
        raise ProtocolError(f"generation {expected_item_id}: EOS must not be retained in generated IDs")
    if artifact.stop_reason not in {"EOS", "CAP", "FAIL_NUMERIC"}:
        raise ProtocolError(f"generation {expected_item_id}: invalid stop reason {artifact.stop_reason!r}")
    if artifact.stop_reason == "FAIL_NUMERIC" and not isinstance(artifact.numeric_error, str):
        raise ProtocolError(f"generation {expected_item_id}: FAIL_NUMERIC requires its numeric error")
    if artifact.stop_reason != "FAIL_NUMERIC" and artifact.numeric_error is not None:
        raise ProtocolError(f"generation {expected_item_id}: numeric error attached to non-numeric stop")
    run32, loop8x3, empty = rollout_flags(generated)
    expected_flags = (run32, loop8x3, empty) if rollout_checks else (False, False, empty)
    stored_flags = (artifact.run32, artifact.loop8x3, artifact.empty)
    if stored_flags != expected_flags:
        raise ProtocolError(f"generation {expected_item_id}: stored degeneration flags do not match generated IDs")
    return artifact.stop_reason == "FAIL_NUMERIC"


def _validate_decoded_generation(
    tokenizer: Any, artifact: GenerationArtifact, *, max_new_tokens: int, expected_item_id: str, rollout_checks: bool
) -> bool:
    """Bind an artifact's IDs, decoded text, bytes, cap, and flags to a tokenizer."""
    numeric_failure = _validate_generation_artifact(
        artifact, expected_item_id=expected_item_id, rollout_checks=rollout_checks
    )
    generated = _require_ids(artifact.generated_ids, f"generation {expected_item_id} IDs", allow_empty=True)
    if len(generated) > max_new_tokens:
        raise ProtocolError(f"generation {expected_item_id}: emitted more than fixed cap {max_new_tokens}")
    if artifact.stop_reason == "CAP" and len(generated) != max_new_tokens:
        raise ProtocolError(f"generation {expected_item_id}: CAP requires exactly {max_new_tokens} non-EOS IDs")
    try:
        decoded = tokenizer.decode(generated, clean_up_tokenization_spaces=False)
    except Exception as exc:
        raise ProtocolError(f"tokenizer failed to decode persisted generation {expected_item_id}: {exc}") from exc
    if not isinstance(decoded, str) or decoded != artifact.decoded_text:
        raise ProtocolError(f"generation {expected_item_id}: decoded text does not match generated IDs")
    if len(decoded.encode("utf-8")) != artifact.emitted_utf8_bytes:
        raise ProtocolError(f"generation {expected_item_id}: stored UTF-8 byte count does not match decoded text")
    return numeric_failure


def validate_pinned_task_artifacts(
    *,
    tokenizer: Any,
    humaneval_records: Sequence[HumanEvalRecord],
    piqa_records: Sequence[PiqaRecord],
    heldout_documents: Sequence[Mapping[str, Any]],
    teacher_piqa: Sequence[PiqaArtifact],
    candidate_piqa: Sequence[PiqaArtifact],
    teacher_humaneval: Sequence[HumanEvalArtifact],
    candidate_humaneval: Sequence[HumanEvalArtifact],
    teacher_rollouts: Sequence[RolloutArtifact],
    candidate_rollouts: Sequence[RolloutArtifact],
    verify_tokenizer: bool = True,
) -> dict[str, Any]:
    """Bind persisted task artifacts to already-audited pinned source records.

    ``audit_pinned_sources`` and the document-manifest/tokenizer audit must be
    completed by the caller first.  This function performs no I/O and no model
    forward: it checks that every persisted teacher/candidate artifact is for
    precisely those in-order source records, tokens, and decoded outputs.
    ``verify_tokenizer=False`` exists only for the in-memory planted tests.
    """
    if verify_tokenizer:
        validate_strat02_tokenizer(tokenizer)
    expected_lengths = {
        "HumanEval records": (len(humaneval_records), HUMANEVAL_COUNT),
        "PIQA records": (len(piqa_records), PIQA_COUNT),
        "heldout documents": (len(heldout_documents), ROLLOUT_COUNT),
        "teacher PIQA": (len(teacher_piqa), PIQA_COUNT),
        "candidate PIQA": (len(candidate_piqa), PIQA_COUNT),
        "teacher HumanEval": (len(teacher_humaneval), HUMANEVAL_COUNT),
        "candidate HumanEval": (len(candidate_humaneval), HUMANEVAL_COUNT),
        "teacher rollouts": (len(teacher_rollouts), ROLLOUT_COUNT),
        "candidate rollouts": (len(candidate_rollouts), ROLLOUT_COUNT),
    }
    for label, (actual, expected) in expected_lengths.items():
        if actual != expected:
            raise ProtocolError(f"pinned artifact binding requires {expected} {label}; got {actual}")

    for index, source in enumerate(piqa_records):
        if source.index != index or source.label not in (0, 1):
            raise ProtocolError(f"PIQA source record order/label mismatch at index {index}")
        expected_prefix = tuple(_encode(tokenizer, f"Question: {source.goal}\nAnswer:", f"PIQA source {index} prefix"))
        expected_suffixes = (
            tuple(_encode(tokenizer, f" {source.sol1}", f"PIQA source {index} option 0 suffix")),
            tuple(_encode(tokenizer, f" {source.sol2}", f"PIQA source {index} option 1 suffix")),
        )
        for arm, artifact in (("teacher", teacher_piqa[index]), ("candidate", candidate_piqa[index])):
            _canonical_piqa_correctness(artifact, index)
            if artifact.label != source.label or tuple(artifact.prefix_ids) != expected_prefix:
                raise ProtocolError(f"PIQA {arm} artifact {index}: label or separately encoded prefix differs from pinned source")
            if tuple(artifact.options[0].suffix_ids) != expected_suffixes[0] or tuple(artifact.options[1].suffix_ids) != expected_suffixes[1]:
                raise ProtocolError(f"PIQA {arm} artifact {index}: separately encoded suffix differs from pinned source")

    for index, source in enumerate(humaneval_records):
        if not isinstance(source.task_id, str) or not source.task_id or not isinstance(source.prompt, str):
            raise ProtocolError(f"HumanEval source record malformed at index {index}")
        expected_prompt_ids = (EOS_ID, *_encode(tokenizer, source.prompt, f"HumanEval source {source.task_id} prompt"))
        for arm, artifact in (("teacher", teacher_humaneval[index]), ("candidate", candidate_humaneval[index])):
            if artifact.task_id != source.task_id or artifact.prompt != source.prompt:
                raise ProtocolError(f"HumanEval {arm} artifact {index}: task ID or literal prompt differs from pinned source")
            if tuple(artifact.generation.prompt_ids) != expected_prompt_ids:
                raise ProtocolError(f"HumanEval {arm} artifact {source.task_id}: prompt IDs differ from pinned tokenizer")
            _validate_decoded_generation(
                tokenizer, artifact.generation, max_new_tokens=HUMANEVAL_MAX_NEW_TOKENS,
                expected_item_id=source.task_id, rollout_checks=False,
            )

    for index, source in enumerate(heldout_documents):
        identifier, category, text = source.get("source_document_id"), source.get("category"), source.get("text")
        if not isinstance(identifier, str) or not identifier or category not in ROLLOUT_CATEGORIES or not isinstance(text, str):
            raise ProtocolError(f"heldout source record malformed at index {index}")
        payload = _encode(tokenizer, text, f"heldout source {identifier} text")
        if len(payload) < ROLLOUT_PREFIX_TOKENS:
            raise ProtocolError(f"heldout source {identifier}: payload is shorter than fixed rollout prefix")
        expected_prompt_ids = (EOS_ID, *payload[:ROLLOUT_PREFIX_TOKENS])
        for arm, artifact in (("teacher", teacher_rollouts[index]), ("candidate", candidate_rollouts[index])):
            if artifact.source_document_id != identifier or artifact.category != category:
                raise ProtocolError(f"rollout {arm} artifact {index}: document ID or category differs from pinned source")
            if tuple(artifact.generation.prompt_ids) != expected_prompt_ids:
                raise ProtocolError(f"rollout {arm} artifact {identifier}: 256-token prefix IDs differ from pinned tokenizer")
            _validate_decoded_generation(
                tokenizer, artifact.generation, max_new_tokens=ROLLOUT_MAX_NEW_TOKENS,
                expected_item_id=identifier, rollout_checks=True,
            )
    return {
        "ok": True,
        "binding": "pinned_sources_tokenizer_and_manifest_rows",
        "items": {"humaneval": HUMANEVAL_COUNT, "piqa": PIQA_COUNT, "rollouts": ROLLOUT_COUNT},
    }


def adjudicate_piqa(teacher: Sequence[PiqaArtifact], candidate: Sequence[PiqaArtifact]) -> dict[str, Any]:
    """Unbound diagnostic PIQA count gate; use pinned binding before final verdicts."""
    if len(teacher) != PIQA_COUNT or len(candidate) != PIQA_COUNT:
        raise ProtocolError(f"PIQA adjudication requires exactly {PIQA_COUNT} artifacts per arm")
    teacher_correctness: list[tuple[bool, bool]] = []
    candidate_correctness: list[tuple[bool, bool]] = []
    for index, (base, trial) in enumerate(zip(teacher, candidate, strict=True)):
        base_correctness = _canonical_piqa_correctness(base, index)
        trial_correctness = _canonical_piqa_correctness(trial, index)
        if base.label != trial.label:
            raise ProtocolError(f"PIQA paired artifact mismatch at index {index}")
        teacher_correctness.append(base_correctness)
        candidate_correctness.append(trial_correctness)
    summary = paired_accuracy_bootstrap(
        [values[0] for values in teacher_correctness],
        [values[0] for values in candidate_correctness],
        expected_count=PIQA_COUNT,
    )
    teacher_correct, candidate_correct = summary["teacher_correct"], summary["candidate_correct"]
    threshold = math.ceil(0.98 * teacher_correct)
    if summary["teacher_accuracy"] <= 0.5:
        verdict = "INCONCLUSIVE_NO_BASELINE_ABILITY"
    elif candidate_correct >= threshold:
        verdict = "PASS_COUNT_GATE"
    else:
        verdict = "FAIL_COUNT_GATE"
    summary.update({
        "candidate_required_correct": threshold,
        "verdict": verdict,
        "teacher_total_nll_accuracy": sum(values[1] for values in teacher_correctness) / PIQA_COUNT,
        "candidate_total_nll_accuracy": sum(values[1] for values in candidate_correctness) / PIQA_COUNT,
    })
    return summary


def adjudicate_humaneval_pending(teacher: Sequence[HumanEvalArtifact], candidate: Sequence[HumanEvalArtifact]) -> dict[str, Any]:
    """Validate paired raw artifacts while refusing to manufacture functional pass@1."""
    if len(teacher) != HUMANEVAL_COUNT or len(candidate) != HUMANEVAL_COUNT:
        raise ProtocolError(f"HumanEval adjudication requires exactly {HUMANEVAL_COUNT} artifacts per arm")
    numeric_failures = {"teacher": [], "candidate": []}
    for number, (base, trial) in enumerate(zip(teacher, candidate, strict=True)):
        if base.task_id != trial.task_id:
            raise ProtocolError(f"HumanEval paired task mismatch at position {number}")
        if base.functional_status != "PENDING_SANDBOX" or trial.functional_status != "PENDING_SANDBOX":
            raise ProtocolError("this module only accepts PENDING_SANDBOX HumanEval artifacts")
        if base.functional_passed is not None or trial.functional_passed is not None:
            raise ProtocolError("HumanEval functional pass@1 must not be inferred in this module")
        if _validate_generation_artifact(base.generation, expected_item_id=base.task_id, rollout_checks=False):
            numeric_failures["teacher"].append(base.task_id)
        if _validate_generation_artifact(trial.generation, expected_item_id=trial.task_id, rollout_checks=False):
            numeric_failures["candidate"].append(trial.task_id)
    return {
        "items": HUMANEVAL_COUNT,
        "verdict": "PENDING_SANDBOX",
        "functional_pass_at_1": None,
        "paired_bootstrap_ci95_delta_pass_at_1": None,
        "reason": "Generated code was neither imported nor executed on this host.",
        "numeric_failures": numeric_failures,
        "arm_verdicts": {
            arm: "FAIL_NUMERIC" if failures else "PENDING_SANDBOX" for arm, failures in numeric_failures.items()
        },
    }


def adjudicate_rollouts(teacher: Sequence[RolloutArtifact], candidate: Sequence[RolloutArtifact]) -> dict[str, Any]:
    """Unbound free-run diagnostic; bind source rows/tokens before final use."""
    if len(teacher) != ROLLOUT_COUNT or len(candidate) != ROLLOUT_COUNT:
        raise ProtocolError(f"rollout adjudication requires exactly {ROLLOUT_COUNT} artifacts per arm")
    new_degenerate: list[str] = []
    numeric_failures = {"teacher": [], "candidate": []}
    by_category: dict[str, dict[str, int]] = {
        category: {"teacher_degenerate": 0, "candidate_degenerate": 0, "new_candidate_degenerate": 0}
        for category in ROLLOUT_CATEGORIES
    }
    for number, (base, trial) in enumerate(zip(teacher, candidate, strict=True)):
        if base.source_document_id != trial.source_document_id or base.category != trial.category:
            raise ProtocolError(f"rollout paired document mismatch at position {number}")
        category = base.category
        if _validate_generation_artifact(
            base.generation, expected_item_id=base.source_document_id, rollout_checks=True
        ):
            numeric_failures["teacher"].append(base.source_document_id)
        if _validate_generation_artifact(
            trial.generation, expected_item_id=trial.source_document_id, rollout_checks=True
        ):
            numeric_failures["candidate"].append(trial.source_document_id)
        # Degeneration is derived again from IDs, not trusted from the stored property.
        base_degenerate = base.generation.run32 or base.generation.loop8x3 or base.generation.empty
        trial_degenerate = trial.generation.run32 or trial.generation.loop8x3 or trial.generation.empty
        if base_degenerate:
            by_category[category]["teacher_degenerate"] += 1
        if trial_degenerate:
            by_category[category]["candidate_degenerate"] += 1
        if trial_degenerate and not base_degenerate:
            by_category[category]["new_candidate_degenerate"] += 1
            new_degenerate.append(base.source_document_id)
    return {
        "documents": ROLLOUT_COUNT,
        "categories": by_category,
        "new_candidate_degenerate_document_ids": new_degenerate,
        "new_candidate_degenerate_count": len(new_degenerate),
        "catastrophic_new_rollout_degeneracy": len(new_degenerate) >= 3,
        "numeric_failures": numeric_failures,
        "arm_verdicts": {
            arm: "FAIL_NUMERIC" if failures else "VALID" for arm, failures in numeric_failures.items()
        },
    }


def adjudicate_task_rollout(
    teacher_piqa: Sequence[PiqaArtifact],
    candidate_piqa: Sequence[PiqaArtifact],
    teacher_humaneval: Sequence[HumanEvalArtifact],
    candidate_humaneval: Sequence[HumanEvalArtifact],
    teacher_rollouts: Sequence[RolloutArtifact],
    candidate_rollouts: Sequence[RolloutArtifact],
) -> dict[str, Any]:
    """Combine unbound diagnostics only; no prompt/text provenance is asserted here."""
    piqa = adjudicate_piqa(teacher_piqa, candidate_piqa)
    humaneval = adjudicate_humaneval_pending(teacher_humaneval, candidate_humaneval)
    rollout = adjudicate_rollouts(teacher_rollouts, candidate_rollouts)
    numeric_failures = {
        arm: humaneval["numeric_failures"][arm] + rollout["numeric_failures"][arm] for arm in ("teacher", "candidate")
    }
    piqa_loss_10pp = piqa["candidate_accuracy"] <= piqa["teacher_accuracy"] - 0.10
    arm_verdicts = {
        arm: "FAIL_NUMERIC" if failures else "PENDING_SANDBOX" for arm, failures in numeric_failures.items()
    }
    return {
        "piqa": piqa,
        "humaneval": humaneval,
        "rollout": rollout,
        "numeric_failures": numeric_failures,
        "arm_verdicts": arm_verdicts,
        "catastrophic": bool(rollout["catastrophic_new_rollout_degeneracy"] or piqa_loss_10pp),
        "verdict": "FAIL_NUMERIC" if numeric_failures["teacher"] or numeric_failures["candidate"] else "PENDING_SANDBOX",
    }


def adjudicate_pinned_task_rollout(
    *,
    tokenizer: Any,
    humaneval_records: Sequence[HumanEvalRecord],
    piqa_records: Sequence[PiqaRecord],
    heldout_documents: Sequence[Mapping[str, Any]],
    teacher_piqa: Sequence[PiqaArtifact],
    candidate_piqa: Sequence[PiqaArtifact],
    teacher_humaneval: Sequence[HumanEvalArtifact],
    candidate_humaneval: Sequence[HumanEvalArtifact],
    teacher_rollouts: Sequence[RolloutArtifact],
    candidate_rollouts: Sequence[RolloutArtifact],
    verify_tokenizer: bool = True,
) -> dict[str, Any]:
    """Validate pinned provenance first, then return the only final task verdict API."""
    binding = validate_pinned_task_artifacts(
        tokenizer=tokenizer,
        humaneval_records=humaneval_records,
        piqa_records=piqa_records,
        heldout_documents=heldout_documents,
        teacher_piqa=teacher_piqa,
        candidate_piqa=candidate_piqa,
        teacher_humaneval=teacher_humaneval,
        candidate_humaneval=candidate_humaneval,
        teacher_rollouts=teacher_rollouts,
        candidate_rollouts=candidate_rollouts,
        verify_tokenizer=verify_tokenizer,
    )
    result = adjudicate_task_rollout(
        teacher_piqa, candidate_piqa, teacher_humaneval, candidate_humaneval, teacher_rollouts, candidate_rollouts
    )
    result["artifact_binding"] = binding
    return result


def assert_full_incremental_parity(
    model: Any, input_ids: Sequence[int], *, atol: float = 1e-5, rtol: float = 1e-5
) -> None:
    """Check a planted full/cache last-position logit parity before real use.

    The fixed tolerances are part of this implementation rather than chosen
    after observing a benchmark result.  It is intentionally not called by a
    CLI path because that would be a real model forward.
    """
    torch = _torch()
    ids = _require_ids(input_ids, "parity input")
    if len(ids) < 2:
        raise ProtocolError("parity input needs at least two tokens")
    if len(ids) > MAX_CONTEXT:
        raise ProtocolError("parity input exceeds native context")
    if not (math.isfinite(atol) and math.isfinite(rtol) and atol >= 0 and rtol >= 0):
        raise ProtocolError("parity tolerances must be finite non-negative numbers")
    _require_eval_causal_model(model)
    device = _model_device(model, torch)
    full_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        full_output = model.model(input_ids=full_ids, use_cache=False, return_dict=True)
        full_hidden = _last_hidden_state(full_output)[:, -1:, :]
        full_logits = model.lm_head(full_hidden).to(torch.float32)
        prefix_output = model.model(input_ids=full_ids[:, :-1], use_cache=True, return_dict=True)
        cache = getattr(prefix_output, "past_key_values", None)
        if cache is None:
            raise ProtocolError("model.model did not return cache for incremental parity")
        incremental_output = model.model(
            input_ids=full_ids[:, -1:], past_key_values=cache, use_cache=True, return_dict=True
        )
        incremental_hidden = _last_hidden_state(incremental_output)[:, -1:, :]
        incremental_logits = model.lm_head(incremental_hidden).to(torch.float32)
    if (
        not bool(torch.isfinite(full_hidden).all().item())
        or not bool(torch.isfinite(incremental_hidden).all().item())
        or not bool(torch.isfinite(full_logits).all().item())
        or not bool(torch.isfinite(incremental_logits).all().item())
    ):
        raise NumericFailure("non-finite hidden states or logits during full/incremental parity")
    if not bool(torch.allclose(full_hidden, incremental_hidden, atol=atol, rtol=rtol)):
        max_abs = float((full_hidden - incremental_hidden).abs().max().item())
        raise ProtocolError(f"VOID_APPARATUS: full/incremental hidden states differ (max_abs={max_abs}, atol={atol}, rtol={rtol})")
    if not bool(torch.allclose(full_logits, incremental_logits, atol=atol, rtol=rtol)):
        max_abs = float((full_logits - incremental_logits).abs().max().item())
        raise ProtocolError(f"VOID_APPARATUS: full/incremental logits differ (max_abs={max_abs}, atol={atol}, rtol={rtol})")


def _fake_model_for_selftest(mode: str = "score") -> Any:
    """Small deterministic model used solely by planted tests; no external data."""
    torch = _torch()

    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.processed_token_positions = 0
            self.cache_lengths: list[int] = []
            self.used_cache_updates = 0

        def forward(
            self, *, input_ids: Any, use_cache: bool, return_dict: bool, past_key_values: Any = None, **_: Any
        ) -> Any:
            if past_key_values is None:
                previous_length = 0
            else:
                if not isinstance(past_key_values, SimpleNamespace) or not isinstance(past_key_values.length, int):
                    raise AssertionError("fake cache update received an invalid cache object")
                previous_length = past_key_values.length
                self.used_cache_updates += 1
            token_count = int(input_ids.shape[1])
            self.processed_token_positions += token_count
            token_hidden = torch.nn.functional.one_hot(input_ids.clamp(max=8), num_classes=9).to(torch.float32)
            positions = torch.arange(previous_length, previous_length + token_count, device=input_ids.device)
            position_hidden = positions.to(torch.float32).view(1, token_count, 1) / 100.0
            hidden = torch.cat((token_hidden, position_hidden), dim=-1)
            cache = SimpleNamespace(length=previous_length + token_count) if use_cache else None
            if use_cache:
                self.cache_lengths.append(cache.length)
            return SimpleNamespace(last_hidden_state=hidden, past_key_values=cache)

    class Fake(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Backbone()
            self.lm_head = torch.nn.Linear(10, EOS_ID + 1, bias=False)
            with torch.no_grad():
                self.lm_head.weight.fill_(-10.0)
                self.lm_head.weight[:, 9].zero_()  # Position is a cache-parity sentinel, not a score feature.
                if mode == "score":
                    # Prefix ends in 8.  A one-token option 3 has lower total
                    # NLL, while option 4,5 has lower mean NLL: planted BPE
                    # length-normalisation and first-suffix-token sentinels.
                    self.lm_head.weight[:, 8].zero_()
                    self.lm_head.weight[4, 8] = 1.0
                    self.lm_head.weight[:, 4].zero_()
                    self.lm_head.weight[5, 4] = 1.0
                elif mode == "eos":
                    self.lm_head.weight[EOS_ID, :].fill_(1.0)
                elif mode == "cap":
                    self.lm_head.weight[2, :].fill_(1.0)
                    self.lm_head.weight[2, 9] = 0.0
                elif mode == "tie":
                    self.lm_head.weight[2, :].fill_(1.0)
                    self.lm_head.weight[3, :].fill_(1.0)
                    self.lm_head.weight[2, 9] = self.lm_head.weight[3, 9] = 0.0
                elif mode == "loop":
                    for token in range(1, 8):
                        self.lm_head.weight[token + 1, token] = 1.0
                    self.lm_head.weight[1, 8] = 1.0
                    self.lm_head.weight[1, EOS_ID if EOS_ID < 9 else 0] = 1.0
                elif mode == "nan":
                    self.lm_head.weight.fill_(float("nan"))

        def forward(self, *, input_ids: Any, use_cache: bool, return_dict: bool) -> Any:
            return SimpleNamespace(logits=self.lm_head(self.model(input_ids=input_ids, use_cache=use_cache, return_dict=return_dict).last_hidden_state))

    return Fake().eval()


class _TinyTokenizer:
    eos_token_id = EOS_ID
    bos_token_id = None
    model_max_length = MAX_CONTEXT
    vocab_size = EOS_ID + 1
    all_special_ids: tuple[int, ...] = ()

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        table = {
            "Question: goal\nAnswer:": [7, 8],
            " a": [3],
            " b": [4, 5],
            " c": [3],
            "prompt": [1],
        }
        return list(table.get(text, [1] * max(1, len(text))))

    def decode(self, ids: Iterable[int], *, clean_up_tokenization_spaces: bool) -> str:
        assert clean_up_tokenization_spaces is False
        return "".join(f"<{token}>" for token in ids)


def selftest() -> dict[str, Any]:
    """Run planted alignment, tie, flags, numeric, and source-corruption tests."""
    import tempfile

    tokenizer = _TinyTokenizer()
    score_model = _fake_model_for_selftest("score")
    assert_full_incremental_parity(score_model, [EOS_ID, 7, 8])
    assert score_model.model.used_cache_updates == 1 and score_model.model.cache_lengths[-2:] == [2, 3]
    record = PiqaRecord(index=0, goal="goal", sol1="a", sol2="b", label=1)
    item = score_piqa_item(score_model, tokenizer, record, verify_tokenizer=False)
    assert item.prefix_ids == (7, 8)
    assert item.options[0].suffix_ids == (3,) and item.options[1].suffix_ids == (4, 5)
    assert item.choice_mean_nll == 1 and item.choice_total_nll == 0, "length normalisation sentinel failed"
    tied = score_piqa_item(score_model, tokenizer, PiqaRecord(0, "goal", "a", "c", 0), verify_tokenizer=False)
    assert tied.choice_mean_nll == 0 and tied.choice_total_nll == 0, "exact tie must choose option 0"
    chunk_model = _fake_model_for_selftest("score")
    chunk_widths: list[int] = []
    hook = chunk_model.lm_head.register_forward_hook(lambda _module, inputs, _output: chunk_widths.append(inputs[0].shape[1]))
    chunk_hidden, chunk_output = _forward_hidden(chunk_model, [EOS_ID, *([8] * 32)])
    try:
        _nll_for_targets(chunk_model, chunk_hidden, 0, [3] * 33)
    finally:
        hook.remove()
        del chunk_hidden, chunk_output
    assert chunk_widths == [PIQA_LOGIT_CHUNK, 1], "PIQA head projection must use bounded 32-token chunks"

    eos_model = _fake_model_for_selftest("eos")
    cap_model = _fake_model_for_selftest("cap")
    loop_model = _fake_model_for_selftest("loop")
    eos = generate_greedy(eos_model, tokenizer, item_id="eos", prompt_ids=[EOS_ID], max_new_tokens=2, rollout_checks=True)
    cap = generate_greedy(cap_model, tokenizer, item_id="cap", prompt_ids=[EOS_ID], max_new_tokens=2, rollout_checks=True)
    tied_generation = generate_greedy(_fake_model_for_selftest("tie"), tokenizer, item_id="tie", prompt_ids=[EOS_ID], max_new_tokens=1, rollout_checks=True)
    loop = generate_greedy(loop_model, tokenizer, item_id="loop", prompt_ids=[1], max_new_tokens=24, rollout_checks=True)
    nan = generate_greedy(_fake_model_for_selftest("nan"), tokenizer, item_id="nan", prompt_ids=[EOS_ID], max_new_tokens=1, rollout_checks=True)
    assert eos.stop_reason == "EOS" and eos.empty
    assert cap.stop_reason == "CAP" and cap.generated_ids == (2, 2)
    assert tied_generation.generated_ids == (2,), "stable generation argmax must choose the smaller tied ID"
    assert eos_model.model.processed_token_positions == 1 and eos_model.model.cache_lengths == [1]
    assert cap_model.model.processed_token_positions == 2 and cap_model.model.cache_lengths == [1, 2]
    assert cap_model.model.used_cache_updates == 1
    assert loop_model.model.processed_token_positions == 24 and loop_model.model.cache_lengths[-1] == 24
    assert loop_model.model.used_cache_updates == 23, "cached generation must be linear, never triangular"
    assert loop.stop_reason == "CAP" and loop.loop8x3 and not loop.run32
    assert nan.stop_reason == "FAIL_NUMERIC" and nan.numeric_error
    try:
        _canonical_piqa_correctness(replace(item, correct_mean_nll=False), 0)
    except ProtocolError:
        pass
    else:  # pragma: no cover - would accept a tampered derived result.
        raise AssertionError("tampered PIQA correctness was accepted")
    try:
        _validate_generation_artifact(replace(cap, run32=True), expected_item_id="cap", rollout_checks=True)
    except ProtocolError:
        pass
    else:  # pragma: no cover - would accept a tampered degeneration flag.
        raise AssertionError("tampered rollout flags were accepted")
    try:
        _validate_generation_artifact(replace(cap, item_id="other"), expected_item_id="cap", rollout_checks=True)
    except ProtocolError:
        pass
    else:  # pragma: no cover - would accept a tampered persisted ID.
        raise AssertionError("tampered generation ID was accepted")
    context_model = _fake_model_for_selftest("cap")
    try:
        generate_greedy(
            context_model, tokenizer, item_id="context", prompt_ids=[EOS_ID] * (MAX_CONTEXT - ROLLOUT_MAX_NEW_TOKENS + 1),
            max_new_tokens=ROLLOUT_MAX_NEW_TOKENS, rollout_checks=True,
        )
    except ProtocolError:
        pass
    else:  # pragma: no cover - would forward an invalid fixed-cap prompt.
        raise AssertionError("over-context prompt was accepted")
    assert context_model.model.processed_token_positions == 0, "context rejection must happen before any forward"

    def planted_generation(item_id: str, prompt_ids: tuple[int, ...], cap: int, *, rollout_checks: bool) -> GenerationArtifact:
        generated_ids = (2,) * cap
        decoded = tokenizer.decode(generated_ids, clean_up_tokenization_spaces=False)
        run32, loop8x3, empty = rollout_flags(generated_ids) if rollout_checks else (False, False, False)
        return GenerationArtifact(
            item_id=item_id, prompt_ids=prompt_ids, generated_ids=generated_ids, decoded_text=decoded,
            emitted_utf8_bytes=len(decoded.encode("utf-8")), stop_reason="CAP", elapsed_seconds=0.0,
            run32=run32, loop8x3=loop8x3, empty=empty,
        )

    bound_piqa_sources = [PiqaRecord(index, "goal", "a", "b", 1) for index in range(PIQA_COUNT)]
    bound_piqa = [replace(item, index=index) for index in range(PIQA_COUNT)]
    bound_human_sources = [HumanEvalRecord(f"h{index}", "prompt") for index in range(HUMANEVAL_COUNT)]
    bound_human = [
        HumanEvalArtifact(
            source.task_id, source.prompt,
            planted_generation(source.task_id, (EOS_ID, 1), HUMANEVAL_MAX_NEW_TOKENS, rollout_checks=False),
        )
        for source in bound_human_sources
    ]
    bound_documents = [
        {"source_document_id": f"d{index}", "category": "code", "text": "x" * ROLLOUT_PREFIX_TOKENS}
        for index in range(ROLLOUT_COUNT)
    ]
    bound_rollouts = [
        RolloutArtifact(
            source["source_document_id"], source["category"],
            planted_generation(source["source_document_id"], (EOS_ID,) + (1,) * ROLLOUT_PREFIX_TOKENS,
                               ROLLOUT_MAX_NEW_TOKENS, rollout_checks=True),
        )
        for source in bound_documents
    ]

    def assert_binding_rejects(**changed: Any) -> None:
        bound = dict(
            tokenizer=tokenizer, humaneval_records=bound_human_sources, piqa_records=bound_piqa_sources,
            heldout_documents=bound_documents, teacher_piqa=bound_piqa, candidate_piqa=bound_piqa,
            teacher_humaneval=bound_human, candidate_humaneval=bound_human,
            teacher_rollouts=bound_rollouts, candidate_rollouts=bound_rollouts, verify_tokenizer=False,
        )
        bound.update(changed)
        try:
            validate_pinned_task_artifacts(**bound)
        except ProtocolError:
            return
        raise AssertionError("pinned artifact binding accepted a planted mutation")

    assert validate_pinned_task_artifacts(
        tokenizer=tokenizer, humaneval_records=bound_human_sources, piqa_records=bound_piqa_sources,
        heldout_documents=bound_documents, teacher_piqa=bound_piqa, candidate_piqa=bound_piqa,
        teacher_humaneval=bound_human, candidate_humaneval=bound_human,
        teacher_rollouts=bound_rollouts, candidate_rollouts=bound_rollouts, verify_tokenizer=False,
    )["ok"]
    bad_human = list(bound_human); bad_human[0] = replace(bad_human[0], prompt="wrong prompt")
    assert_binding_rejects(teacher_humaneval=bad_human)
    bad_piqa = list(bound_piqa)
    bad_piqa[0] = replace(bad_piqa[0], options=(replace(bad_piqa[0].options[0], suffix_ids=(5,)), bad_piqa[0].options[1]))
    assert_binding_rejects(teacher_piqa=bad_piqa)
    bad_text = list(bound_rollouts)
    bad_text[0] = replace(bad_text[0], generation=replace(bad_text[0].generation, decoded_text="tampered"))
    assert_binding_rejects(teacher_rollouts=bad_text)
    bad_category = list(bound_rollouts); bad_category[0] = replace(bad_category[0], category="prose")
    assert_binding_rejects(teacher_rollouts=bad_category)
    bad_cap = list(bound_rollouts)
    short_ids = bad_cap[0].generation.generated_ids[:-1]
    short_text = tokenizer.decode(short_ids, clean_up_tokenization_spaces=False)
    bad_cap[0] = replace(
        bad_cap[0], generation=replace(
            bad_cap[0].generation, generated_ids=short_ids, decoded_text=short_text,
            emitted_utf8_bytes=len(short_text.encode("utf-8")),
        )
    )
    assert_binding_rejects(teacher_rollouts=bad_cap)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        human_path, valid_path, labels_path = root / "HumanEval.jsonl.gz", root / "valid.jsonl", root / "valid-labels.lst"
        human_payload = b'{"task_id":"t","prompt":"p"}\n'
        human_path.write_bytes(gzip.compress(human_payload))
        valid_path.write_bytes(b'{"goal":"g","sol1":"x","sol2":"y"}\n')
        labels_path.write_bytes(b"0\n")
        pins = SourcePins(_sha256(human_path.read_bytes()), _sha256(valid_path.read_bytes()), _sha256(labels_path.read_bytes()), 1, 1)
        audit = audit_pinned_sources(human_path, valid_path, labels_path, pins=pins)
        assert audit["ok"] and len(audit["humaneval"]) == len(audit["piqa"]) == 1
        valid_path.write_bytes(b'{"goal":"g","sol1":"x","sol2":"z"}\n')
        try:
            audit_pinned_sources(human_path, valid_path, labels_path, pins=pins)
        except SourceAuditError:
            pass
        else:  # pragma: no cover - would prove corruption was accepted.
            raise AssertionError("source corruption was not rejected")
    return {"ok": True, "selftest": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--selftest", action="store_true", help="run only in-memory planted tests")
    modes.add_argument("--preflight-source", action="store_true", help="read-only SHA/count audit of pinned benchmark files")
    parser.add_argument("--humaneval-gz", type=Path, help="pinned HumanEval.jsonl.gz")
    parser.add_argument("--piqa-valid", type=Path, help="pinned PIQA valid.jsonl")
    parser.add_argument("--piqa-labels", type=Path, help="pinned PIQA valid-labels.lst")
    args = parser.parse_args(argv)
    if args.selftest:
        print(json.dumps(selftest(), sort_keys=True))
        return 0
    paths = (args.humaneval_gz, args.piqa_valid, args.piqa_labels)
    if any(path is None for path in paths):
        parser.error("--preflight-source requires --humaneval-gz, --piqa-valid, and --piqa-labels")
    try:
        report = audit_pinned_sources(args.humaneval_gz, args.piqa_valid, args.piqa_labels)
    except SourceAuditError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    # Do not echo source benchmark text in the preflight report.
    report.pop("humaneval")
    report.pop("piqa")
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
