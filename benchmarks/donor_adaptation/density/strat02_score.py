#!/usr/bin/env python3
"""Read-only STRAT-02 per-document scoring core and paired adjudicator.

This module deliberately does not load a checkpoint.  A caller that has already
constructed an ``EmoForCausalLM`` can use ``score_document``; the command line
only runs the no-model self-test, the existing tokenizer/corpus preflight, or
adjudication of two persisted per-document score files.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
EOS_ID = 100257
MAX_CONTEXT = 4096
CHUNK_SIZE = 128
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20_260_916
CATEGORIES = ("code", "technical_general", "prose")
DOCUMENTS_PER_CATEGORY = 32


@dataclass(frozen=True)
class DocumentScore:
    """The persisted STRAT-02 score schema for one source document."""

    source_document_id: str
    category: str
    tokens: int
    bytes: int
    bits: float

    def json_record(self) -> dict[str, Any]:
        return asdict(self)


def _torch():
    """Import torch lazily so --adjudicate remains a NumPy-only operation."""
    import torch

    return torch


def _require_int(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _model_device(model: Any, torch: Any):
    """Find the model's device without retaining a parameter iterator."""
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        weight = getattr(getattr(model, "lm_head", None), "weight", None)
        return weight.device if weight is not None else torch.device("cpu")


def _last_hidden_state(output: Any) -> Any:
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, (tuple, list)) and output:
        hidden = output[0]
    if hidden is None:
        raise ValueError("model.model did not return last_hidden_state")
    return hidden


def validate_strat02_tokenizer(tokenizer: Any) -> None:
    """Require the exact tokenizer runtime fixed by the STRAT-02 protocol."""
    try:
        import tokenizers
        import transformers
        from transformers import GPT2TokenizerFast
    except ImportError as exc:  # pragma: no cover - makes the failure actionable.
        raise RuntimeError("STRAT-02 scoring requires transformers and tokenizers") from exc
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


def _encode_payload(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(ids, list) or any(isinstance(token, bool) or not isinstance(token, int) for token in ids):
        raise ValueError("tokenizer.encode must return a list of integer IDs")
    if tokenizer.decode(ids, clean_up_tokenization_spaces=False) != text:
        raise ValueError("tokenizer roundtrip differs from the exact document text")
    if set(ids).intersection(getattr(tokenizer, "all_special_ids", ())):
        raise ValueError("payload contains a special token ID")
    return ids


def score_payload_ids(
    model: Any,
    payload_ids: Sequence[int],
    *,
    byte_count: int,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[float, int]:
    """Return ``(bits, token_count)`` for already validated payload token IDs.

    ``model.model`` is used directly so the full vocabulary logits tensor is
    never materialized.  The one unavoidable hidden-state tensor is released
    before returning; each LM-head chunk is immediately reduced to its NLL.
    """
    torch = _torch()
    byte_count = _require_int(byte_count, "byte_count", positive=True)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    ids = list(payload_ids)
    if not ids:
        raise ValueError("STRAT-02 documents must contain at least one payload token")
    if len(ids) + 1 > MAX_CONTEXT:
        raise ValueError(f"STRAT-02 forbids truncation: context {len(ids) + 1} exceeds {MAX_CONTEXT}")
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in ids):
        raise ValueError("payload_ids must be non-negative integer IDs")
    if getattr(model, "training", None) is not False:
        raise RuntimeError("model must be in eval mode before STRAT-02 scoring")
    if not callable(getattr(model, "model", None)) or not callable(getattr(model, "lm_head", None)):
        raise TypeError("expected an EmoForCausalLM-like object with model and lm_head")

    device = _model_device(model, torch)
    inputs = torch.tensor([[EOS_ID, *ids[:-1]]], dtype=torch.long, device=device)
    targets = torch.tensor([ids], dtype=torch.long, device=device)
    nll_total = torch.zeros((), dtype=torch.float64, device=device)

    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        backbone_output = model.model(input_ids=inputs, use_cache=False, return_dict=True)
        hidden = _last_hidden_state(backbone_output)
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != len(ids):
            raise ValueError(
                "model.model hidden-state length must exactly equal the number of scored payload tokens"
            )
        for start in range(0, len(ids), chunk_size):
            stop = min(start + chunk_size, len(ids))
            logits = model.lm_head(hidden[:, start:stop, :])
            if logits.ndim != 3 or logits.shape[:2] != (1, stop - start):
                raise ValueError("lm_head returned logits with an unexpected shape")
            if logits.shape[-1] <= int(targets[:, start:stop].max().item()):
                raise ValueError("selected target token is outside the LM-head vocabulary")
            logits_f32 = logits.to(dtype=torch.float32)
            if not bool(torch.isfinite(logits_f32).all().item()):
                raise FloatingPointError("non-finite LM-head logits")
            selected = logits_f32.gather(-1, targets[:, start:stop].unsqueeze(-1)).squeeze(-1)
            token_nll = torch.logsumexp(logits_f32, dim=-1) - selected
            if not bool(torch.isfinite(token_nll).all().item()):
                raise FloatingPointError("non-finite per-token NLL")
            nll_total += token_nll.to(dtype=torch.float64).sum(dtype=torch.float64)
            del logits, logits_f32, selected, token_nll
        del hidden, backbone_output

    nll = float(nll_total.detach().cpu().item())
    if not math.isfinite(nll):
        raise FloatingPointError("non-finite total NLL")
    return nll / math.log(2.0), len(ids)


def score_document(
    model: Any,
    tokenizer: Any,
    document: Mapping[str, Any],
    *,
    chunk_size: int = CHUNK_SIZE,
) -> DocumentScore:
    """Score exactly one heldout JSONL document under the frozen STRAT-02 rule."""
    validate_strat02_tokenizer(tokenizer)
    source_document_id = document.get("source_document_id")
    category = document.get("category")
    text = document.get("text")
    if not isinstance(source_document_id, str) or not source_document_id:
        raise ValueError("document.source_document_id must be a non-empty string")
    if category not in CATEGORIES:
        raise ValueError(f"document.category must be one of {CATEGORIES}")
    if not isinstance(text, str):
        raise ValueError("document.text must be a string")
    raw_bytes = text.encode("utf-8")
    if not raw_bytes:
        raise ValueError("STRAT-02 documents must contain at least one raw UTF-8 byte")
    payload_ids = _encode_payload(tokenizer, text)
    bits, tokens = score_payload_ids(model, payload_ids, byte_count=len(raw_bytes), chunk_size=chunk_size)
    return DocumentScore(
        source_document_id=source_document_id,
        category=category,
        tokens=tokens,
        bytes=len(raw_bytes),
        bits=bits,
    )


def _number(value: Any, field: str, path: Path, line_number: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}:{line_number}: {field} must be a finite JSON number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{path}:{line_number}: {field} must be finite")
    return value


def read_score_jsonl(path: Path) -> dict[str, DocumentScore]:
    """Read and fully validate one persisted scorer output without mutating it."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    scores: dict[str, DocumentScore] = {}
    required = {"source_document_id", "category", "tokens", "bytes", "bits"}
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank JSONL line")
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"{path}:{line_number}: expected fields {sorted(required)}")
        identifier = item["source_document_id"]
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{path}:{line_number}: invalid source_document_id")
        if identifier in scores:
            raise ValueError(f"{path}:{line_number}: duplicate source_document_id {identifier!r}")
        category = item["category"]
        if category not in CATEGORIES:
            raise ValueError(f"{path}:{line_number}: invalid category {category!r}")
        tokens = _require_int(item["tokens"], f"{path}:{line_number}: tokens", positive=True)
        byte_count = _require_int(item["bytes"], f"{path}:{line_number}: bytes", positive=True)
        bits = _number(item["bits"], "bits", path, line_number)
        if bits < 0:
            raise ValueError(f"{path}:{line_number}: bits must be non-negative")
        scores[identifier] = DocumentScore(identifier, category, tokens, byte_count, bits)
    if not scores:
        raise ValueError(f"{path}: no score records")
    return scores


def _validate_paired_scores(
    teacher: Mapping[str, DocumentScore], candidate: Mapping[str, DocumentScore]
) -> dict[str, list[tuple[DocumentScore, DocumentScore]]]:
    teacher_ids, candidate_ids = set(teacher), set(candidate)
    if teacher_ids != candidate_ids:
        missing = sorted(teacher_ids - candidate_ids)
        unexpected = sorted(candidate_ids - teacher_ids)
        detail = []
        if missing:
            detail.append(f"missing candidate documents: {missing[:5]}")
        if unexpected:
            detail.append(f"unexpected candidate documents: {unexpected[:5]}")
        raise ValueError("paired document sets differ; " + "; ".join(detail))
    grouped: dict[str, list[tuple[DocumentScore, DocumentScore]]] = {category: [] for category in CATEGORIES}
    for identifier in sorted(teacher_ids):
        base, trial = teacher[identifier], candidate[identifier]
        if base.category != trial.category:
            raise ValueError(f"{identifier}: category mismatch ({base.category!r} != {trial.category!r})")
        if base.bytes != trial.bytes:
            raise ValueError(f"{identifier}: raw UTF-8 byte-count mismatch ({base.bytes} != {trial.bytes})")
        if base.tokens != trial.tokens:
            raise ValueError(f"{identifier}: payload token-count mismatch ({base.tokens} != {trial.tokens})")
        grouped[base.category].append((base, trial))
    for category, pairs in grouped.items():
        if len(pairs) != DOCUMENTS_PER_CATEGORY:
            raise ValueError(
                f"heldout {category!r} requires exactly {DOCUMENTS_PER_CATEGORY} documents; got {len(pairs)}"
            )
    return grouped


def _summary_for_pairs(
    pairs: Sequence[tuple[DocumentScore, DocumentScore]], rng: np.random.Generator
) -> dict[str, Any]:
    teacher_bits = np.asarray([pair[0].bits for pair in pairs], dtype=np.float64)
    candidate_bits = np.asarray([pair[1].bits for pair in pairs], dtype=np.float64)
    byte_counts = np.asarray([pair[0].bytes for pair in pairs], dtype=np.float64)
    sampled = rng.integers(0, len(pairs), size=(BOOTSTRAP_DRAWS, len(pairs)), endpoint=False)
    bootstrap_numerator = (candidate_bits - teacher_bits)[sampled].sum(axis=1, dtype=np.float64)
    bootstrap_denominator = byte_counts[sampled].sum(axis=1, dtype=np.float64)
    bootstrap_delta = bootstrap_numerator / bootstrap_denominator
    return {
        "documents": len(pairs),
        "tokens": int(sum(pair[0].tokens for pair in pairs)),
        "bytes": int(byte_counts.sum(dtype=np.float64)),
        "teacher_bpb": float(teacher_bits.sum(dtype=np.float64) / byte_counts.sum(dtype=np.float64)),
        "candidate_bpb": float(candidate_bits.sum(dtype=np.float64) / byte_counts.sum(dtype=np.float64)),
        "point_delta_bpb": float((candidate_bits - teacher_bits).sum(dtype=np.float64) / byte_counts.sum(dtype=np.float64)),
        "upper_one_sided_ci95_delta_bpb": float(np.quantile(bootstrap_delta, 0.95, method="linear")),
        "_bootstrap_numerator": bootstrap_numerator,
        "_bootstrap_denominator": bootstrap_denominator,
    }


def adjudicate(teacher: Mapping[str, DocumentScore], candidate: Mapping[str, DocumentScore]) -> dict[str, Any]:
    """Compute the fixed paired, document-stratified STRAT-02 adjudication."""
    grouped = _validate_paired_scores(teacher, candidate)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    categories: dict[str, dict[str, Any]] = {}
    all_bootstrap_numerator = np.zeros(BOOTSTRAP_DRAWS, dtype=np.float64)
    all_bootstrap_denominator = np.zeros(BOOTSTRAP_DRAWS, dtype=np.float64)
    all_pairs: list[tuple[DocumentScore, DocumentScore]] = []
    for category in CATEGORIES:
        summary = _summary_for_pairs(grouped[category], rng)
        all_bootstrap_numerator += summary.pop("_bootstrap_numerator")
        all_bootstrap_denominator += summary.pop("_bootstrap_denominator")
        categories[category] = summary
        all_pairs.extend(grouped[category])
    total_bytes = sum(pair[0].bytes for pair in all_pairs)
    all_bootstrap_delta = all_bootstrap_numerator / all_bootstrap_denominator
    teacher_bits = sum(pair[0].bits for pair in all_pairs)
    candidate_bits = sum(pair[1].bits for pair in all_pairs)
    return {
        "protocol": {
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "documents_per_category": DOCUMENTS_PER_CATEGORY,
            "categories": list(CATEGORIES),
            "resampling": "paired_document_stratified_with_replacement",
            "quantile_method": "linear",
            "ci": "upper_one_sided_95_percent",
        },
        "documents": len(all_pairs),
        "tokens": int(sum(pair[0].tokens for pair in all_pairs)),
        "bytes": int(total_bytes),
        "teacher_bpb": float(teacher_bits / total_bytes),
        "candidate_bpb": float(candidate_bits / total_bytes),
        "point_delta_bpb": float((candidate_bits - teacher_bits) / total_bytes),
        "upper_one_sided_ci95_delta_bpb": float(np.quantile(all_bootstrap_delta, 0.95, method="linear")),
        "categories": categories,
    }


def adjudicate_files(teacher_path: Path, candidate_path: Path) -> dict[str, Any]:
    return adjudicate(read_score_jsonl(teacher_path), read_score_jsonl(candidate_path))


def validate_scores_against_heldout(
    scores: Mapping[str, DocumentScore], heldout_rows: Sequence[Mapping[str, Any]], tokenizer: Any, *, arm: str
) -> None:
    """Bind persisted score records to the exact pinned heldout documents.

    The caller is responsible for running ``strat02_token_audit.audit`` first,
    which verifies the corpus, tokenizer assets, runtime, round trips, and
    ordered token-ID hashes.  This focused check then binds every score row to
    that verified heldout identity rather than merely to its opposite arm.
    """
    expected: dict[str, tuple[str, int, int]] = {}
    for number, row in enumerate(heldout_rows, 1):
        identifier = row.get("source_document_id")
        category = row.get("category")
        text = row.get("text")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"pinned heldout:{number}: invalid source_document_id")
        if identifier in expected:
            raise ValueError(f"pinned heldout:{number}: duplicate source_document_id {identifier!r}")
        if category not in CATEGORIES or not isinstance(text, str):
            raise ValueError(f"pinned heldout:{number}: invalid category or text")
        payload_ids = tokenizer.encode(text, add_special_tokens=False)
        if not isinstance(payload_ids, list):
            raise ValueError(f"pinned heldout:{number}: tokenizer did not return a list")
        expected[identifier] = (category, len(text.encode("utf-8")), len(payload_ids))

    actual_ids, expected_ids = set(scores), set(expected)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        detail = []
        if missing:
            detail.append(f"missing {arm} documents: {missing[:5]}")
        if extra:
            detail.append(f"extra {arm} documents: {extra[:5]}")
        raise ValueError("pinned heldout score records differ; " + "; ".join(detail))
    for identifier in sorted(expected_ids):
        score = scores[identifier]
        category, byte_count, token_count = expected[identifier]
        if score.category != category:
            raise ValueError(f"{arm}:{identifier}: category differs from pinned heldout")
        if score.bytes != byte_count:
            raise ValueError(f"{arm}:{identifier}: raw UTF-8 byte count differs from pinned heldout")
        if score.tokens != token_count:
            raise ValueError(f"{arm}:{identifier}: payload token count differs from pinned heldout")


def adjudicate_pinned_files(
    teacher_path: Path,
    candidate_path: Path,
    heldout_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    """Read, bind, then adjudicate score files after the caller's preflight."""
    teacher = read_score_jsonl(teacher_path)
    candidate = read_score_jsonl(candidate_path)
    validate_scores_against_heldout(teacher, heldout_rows, tokenizer, arm="teacher")
    validate_scores_against_heldout(candidate, heldout_rows, tokenizer, arm="candidate")
    return adjudicate(teacher, candidate)


def _fake_model_for_selftest() -> Any:
    torch = _torch()

    class FakeBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, *, input_ids: Any, use_cache: bool, return_dict: bool) -> Any:
            assert use_cache is False and return_dict is True
            # EOS is a legal input ID but never a target.  Map it and the
            # three payload-prefix IDs onto a tiny hidden basis, avoiding a
            # dense 100258-wide fake LM head.
            hidden = torch.zeros((*input_ids.shape, 4), dtype=torch.float32, device=input_ids.device)
            for feature, token in enumerate((EOS_ID, 3, 1, 5)):
                hidden[..., feature] = input_ids.eq(token).to(torch.float32)
            return SimpleNamespace(last_hidden_state=hidden)

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = FakeBackbone()
            self.lm_head = torch.nn.Linear(4, 6, bias=False)
            with torch.no_grad():
                self.lm_head.weight.zero_()
                # The inputs used below are EOS, 3, 1, and 5.  First and last
                # target tokens are deliberately distinctive off-by-one sentinels.
                self.lm_head.weight[3, 0] = -2.75
                self.lm_head.weight[1, 1] = 0.50
                self.lm_head.weight[5, 2] = 1.25
                self.lm_head.weight[2, 3] = 3.50

        def forward(self, *, input_ids: Any, use_cache: bool, return_dict: bool) -> Any:
            hidden = self.model(input_ids=input_ids, use_cache=use_cache, return_dict=return_dict).last_hidden_state
            return SimpleNamespace(logits=self.lm_head(hidden))

    return FakeModel().eval()


def selftest() -> dict[str, Any]:
    """Exercise numerical alignment without any tokenizer or checkpoint download."""
    torch = _torch()
    model = _fake_model_for_selftest()
    payload = [3, 1, 5, 2]
    text = "Aé😀"
    byte_count = len(text.encode("utf-8"))
    assert byte_count == 7, "self-test requires a non-ASCII UTF-8 denominator"

    # Independent full-vocabulary reference: do not call score_payload_ids here.
    inputs = torch.tensor([[EOS_ID, *payload[:-1]]], dtype=torch.long)
    targets = torch.tensor(payload, dtype=torch.long)
    with torch.inference_mode():
        full_logits = model(input_ids=inputs, use_cache=False, return_dict=True).logits.to(torch.float32)[0]
        full_nll = torch.nn.functional.cross_entropy(full_logits, targets, reduction="sum")
        individual_nll = torch.nn.functional.cross_entropy(full_logits, targets, reduction="none")
    expected_bits = float(full_nll.to(torch.float64).item() / math.log(2.0))
    one_chunk_bits, tokens = score_payload_ids(model, payload, byte_count=byte_count, chunk_size=1)
    wide_chunk_bits, wide_tokens = score_payload_ids(model, payload, byte_count=byte_count, chunk_size=CHUNK_SIZE)
    assert tokens == wide_tokens == len(payload)
    assert math.isclose(one_chunk_bits, expected_bits, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(wide_chunk_bits, expected_bits, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(one_chunk_bits, wide_chunk_bits, rel_tol=0.0, abs_tol=1e-8)
    total_nll = float(full_nll.item())
    assert not math.isclose(total_nll, float(individual_nll[1:].sum().item()), rel_tol=0.0, abs_tol=1e-6)
    assert not math.isclose(total_nll, float(individual_nll[:-1].sum().item()), rel_tol=0.0, abs_tol=1e-6)

    fake_scores: dict[str, DocumentScore] = {}
    fake_candidate: dict[str, DocumentScore] = {}
    for category_index, category in enumerate(CATEGORIES):
        for document_index in range(DOCUMENTS_PER_CATEGORY):
            identifier = f"{category}-{document_index:02d}"
            bytes_for_doc = 7 + document_index
            teacher_score = DocumentScore(identifier, category, 4, bytes_for_doc, 10.0 + document_index)
            fake_scores[identifier] = teacher_score
            fake_candidate[identifier] = DocumentScore(
                identifier, category, 4, bytes_for_doc, teacher_score.bits + 0.1 * (category_index + 1)
            )
    report_a = adjudicate(fake_scores, fake_candidate)
    report_b = adjudicate(fake_scores, fake_candidate)
    assert report_a == report_b, "bootstrap must be deterministic under the pinned seed"

    class FakeTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return list(range(1, (len(text) % 5) + 2))

    pinned_rows = []
    pinned_teacher: dict[str, DocumentScore] = {}
    pinned_candidate: dict[str, DocumentScore] = {}
    fake_tokenizer = FakeTokenizer()
    for category_index, category in enumerate(CATEGORIES):
        for document_index in range(DOCUMENTS_PER_CATEGORY):
            identifier = f"pinned-{category}-{document_index:02d}"
            text_value = f"{category}:{document_index}:é"
            pinned_rows.append({"source_document_id": identifier, "category": category, "text": text_value})
            pinned_teacher[identifier] = DocumentScore(
                identifier, category, len(fake_tokenizer.encode(text_value, add_special_tokens=False)),
                len(text_value.encode("utf-8")), 20.0 + document_index,
            )
            pinned_candidate[identifier] = DocumentScore(
                identifier, category, pinned_teacher[identifier].tokens, pinned_teacher[identifier].bytes,
                pinned_teacher[identifier].bits + 0.25,
            )
    validate_scores_against_heldout(pinned_teacher, pinned_rows, fake_tokenizer, arm="teacher")
    for mutation, expected_error in (
        ("missing", "missing candidate documents"),
        ("extra", "extra candidate documents"),
        ("category", "category differs"),
        ("bytes", "byte count differs"),
        ("tokens", "token count differs"),
    ):
        changed_scores = dict(pinned_candidate)
        changed = changed_scores["pinned-code-00"]
        if mutation == "missing":
            changed_scores.pop(changed.source_document_id)
        elif mutation == "extra":
            changed_scores["not-pinned"] = DocumentScore("not-pinned", "code", 1, 1, 0.0)
        elif mutation == "category":
            changed_scores[changed.source_document_id] = DocumentScore(changed.source_document_id, "prose", changed.tokens, changed.bytes, changed.bits)
        elif mutation == "bytes":
            changed_scores[changed.source_document_id] = DocumentScore(changed.source_document_id, changed.category, changed.tokens, changed.bytes + 1, changed.bits)
        else:
            changed_scores[changed.source_document_id] = DocumentScore(changed.source_document_id, changed.category, changed.tokens + 1, changed.bytes, changed.bits)
        try:
            validate_scores_against_heldout(changed_scores, pinned_rows, fake_tokenizer, arm="candidate")
        except ValueError as exc:
            assert expected_error in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"pinned-heldout {mutation} mutation was not rejected")

    weighted_teacher: dict[str, DocumentScore] = {}
    weighted_candidate: dict[str, DocumentScore] = {}
    for category in CATEGORIES:
        for document_index in range(DOCUMENTS_PER_CATEGORY):
            identifier = f"weighted-{category}-{document_index:02d}"
            byte_count_for_doc = 1 if identifier == "weighted-code-00" else 100
            weighted_teacher[identifier] = DocumentScore(identifier, category, 1, byte_count_for_doc, 0.0)
            weighted_candidate[identifier] = DocumentScore(
                identifier, category, 1, byte_count_for_doc, 1.0 if identifier == "weighted-code-00" else 0.0
            )
    weighted = adjudicate(weighted_teacher, weighted_candidate)
    expected_weighted_delta = 1.0 / (1.0 + 95.0 * 100.0)
    mean_document_delta = 1.0 / 96.0
    assert math.isclose(weighted["point_delta_bpb"], expected_weighted_delta, rel_tol=0.0, abs_tol=1e-15)
    assert not math.isclose(weighted["point_delta_bpb"], mean_document_delta, rel_tol=0.0, abs_tol=1e-12)

    duplicate = dict(fake_scores)
    duplicate.pop("code-00")
    try:
        adjudicate(duplicate, fake_candidate)
    except ValueError as exc:
        assert "document sets differ" in str(exc)
    else:  # pragma: no cover - assertion makes the mutation test explicit.
        raise AssertionError("missing-document mutation was not rejected")
    bad_bytes = dict(fake_candidate)
    changed = bad_bytes["code-00"]
    bad_bytes["code-00"] = DocumentScore(changed.source_document_id, changed.category, changed.tokens, changed.bytes + 1, changed.bits)
    try:
        adjudicate(fake_scores, bad_bytes)
    except ValueError as exc:
        assert "byte-count mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("byte-count mutation was not rejected")
    bad_category = dict(fake_candidate)
    changed = bad_category["code-00"]
    bad_category["code-00"] = DocumentScore(changed.source_document_id, "prose", changed.tokens, changed.bytes, changed.bits)
    try:
        adjudicate(fake_scores, bad_category)
    except ValueError as exc:
        assert "category mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("category mutation was not rejected")
    return {"ok": True, "selftest": True, "utf8_bytes": byte_count, "tokens": tokens}


def _run_preflight(args: argparse.Namespace, audit_module: Any | None = None) -> dict[str, Any]:
    """Run the existing pinned, read-only tokenizer/corpus audit."""
    module = audit_module if audit_module is not None else _load_audit_module()
    return module.audit(
        argparse.Namespace(
            manifest=args.manifest,
            calib=args.calib,
            heldout=args.heldout,
            tokenizer_snapshot=args.tokenizer_snapshot,
        )
    )


def _load_audit_module() -> Any:
    audit_path = HERE / "strat02_token_audit.py"
    spec = importlib.util.spec_from_file_location("strat02_token_audit", audit_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load existing audit {audit_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def main(argv: Sequence[str] | None = None) -> int:
    corpus = HERE / "corpus" / "strat02_document_holdout_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--selftest", action="store_true", help="run the no-model numerical self-test")
    action.add_argument("--adjudicate", nargs=2, metavar=("TEACHER_JSONL", "CANDIDATE_JSONL"))
    action.add_argument("--preflight", action="store_true", help="run the existing pinned tokenizer/corpus audit")
    parser.add_argument("--manifest", type=Path, default=corpus / "manifest.json")
    parser.add_argument("--calib", type=Path, default=corpus / "calib.jsonl")
    parser.add_argument("--heldout", type=Path, default=corpus / "heldout.jsonl")
    parser.add_argument("--tokenizer-snapshot", type=Path)
    args = parser.parse_args(argv)
    if args.tokenizer_snapshot is None:
        import os

        cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
        args.tokenizer_snapshot = cache / "models--allenai--StdMoE_1b14b_1T_Preanneal" / "snapshots" / "d2a4949c9d4ad6cf47fbac131f7e020077332b21"
    try:
        if args.selftest:
            result = selftest()
        elif args.preflight:
            result = _run_preflight(args)
        else:
            teacher_path, candidate_path = map(Path, args.adjudicate)
            audit_module = _load_audit_module()
            preflight = _run_preflight(args, audit_module)
            if not preflight["ok"]:
                raise RuntimeError("pinned preflight failed: " + "; ".join(preflight["errors"]))
            read_errors: list[str] = []
            heldout_rows = audit_module.read_rows(args.heldout, "heldout", read_errors)
            if read_errors:
                raise RuntimeError("cannot reread pinned heldout: " + "; ".join(read_errors))
            tokenizer, _, _, _ = audit_module.load_tokenizer(args.tokenizer_snapshot)
            result = adjudicate_pinned_files(teacher_path, candidate_path, heldout_rows, tokenizer)
    except (OSError, RuntimeError, TypeError, ValueError, FloatingPointError, AssertionError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
