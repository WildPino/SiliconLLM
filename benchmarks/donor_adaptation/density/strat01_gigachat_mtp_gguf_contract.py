"""Read-only contract preflight for the pinned GigaChat 3.1 MTP GGUF draft.

This module reads only the small JSON manifest emitted by
``strat01_gigachat_mtp_extract.py``.  It neither opens the 1.6 GiB sidecar
payload nor writes a GGUF.  Its purpose is deliberately narrower: prove that
the manifest describes exactly the MTP block expected by a future writer, and
emit the deterministic tensor transformation plan that writer must follow.

The GGUF names and logical dimensions below are taken from llama.cpp commit
5b335f413e4f73b0809c4fe39af894efbcc6a0d2:

* ``src/models/deepseek2.cpp`` (loader dimensions and NextN names), and
* ``conversion/deepseek.py`` (expert stacking and MLA ``kv_b`` split).

The pinned runtime still rejects this particular GigaChat MTP graph because
its lite/direct-``q_proj`` configuration leaves q_lora_rank at zero while the
MTP graph asserts a Q-LoRA path.  This contract reports that blocker; it does
not disguise a metadata pass as a runnable GGUF.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY = "ai-sage/GigaChat3.1-10B-A1.8B-bf16"
REVISION = "189fff27a1dee68473960c3d5bca53e0e07a3191"
SHARD = "model-00000-of-00005.safetensors"
SOURCE_LFS_SHA256 = "612f84d11aca544694ef674e58781d0c39ae4b2046f9cadeada79f1d37e581b6"
SIDECAR_NAME = "strat01_gigachat31_mtp_bf16_189fff27.safetensors"
SIDECAR_SHA256 = "c340ed41c1355441207357723b984fc336082e7c0e8a00deeb6f4fce15c46119"
SIDECAR_BYTES = 1_614_456_643
ROOT_NORM_BF16_SHA256 = "12935320d4033c560cec87fc07cc5d3f0f28ef2cb5b0f509d7a2a05246b0e44b"
MTP_LAYER = 26
N_EXPERTS = 64
HIDDEN = 1536
VOCAB = 128256
EXPERT_FF = 1280
N_HEADS = 32
QK_NOPE = 128
V_HEAD = 192
KV_LORA = 512

PINNED_LLAMA_CPP = "5b335f413e4f73b0809c4fe39af894efbcc6a0d2"
PINNED_DEEPSEEK2_CPP = (
    "https://github.com/ggml-org/llama.cpp/blob/"
    + PINNED_LLAMA_CPP
    + "/src/models/deepseek2.cpp"
)
PINNED_DEEPSEEK_CONVERTER = (
    "https://github.com/ggml-org/llama.cpp/blob/"
    + PINNED_LLAMA_CPP
    + "/conversion/deepseek.py"
)

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent
    / "results"
    / "strat01_gigachat31_mtp_bf16_189fff27.safetensors.manifest.json"
)


class ContractError(ValueError):
    """Raised when the manifest cannot prove the draft metadata contract."""


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str = "BF16"


def _name(suffix: str) -> str:
    return f"model.layers.{MTP_LAYER}.{suffix}"


NONEXPERT_SPECS: dict[str, TensorSpec] = {
    _name("eh_proj.weight"): TensorSpec((HIDDEN, 2 * HIDDEN)),
    _name("embed_tokens.weight"): TensorSpec((VOCAB, HIDDEN)),
    _name("enorm.weight"): TensorSpec((HIDDEN,)),
    _name("hnorm.weight"): TensorSpec((HIDDEN,)),
    _name("input_layernorm.weight"): TensorSpec((HIDDEN,)),
    _name("post_attention_layernorm.weight"): TensorSpec((HIDDEN,)),
    _name("self_attn.kv_a_layernorm.weight"): TensorSpec((KV_LORA,)),
    _name("self_attn.kv_a_proj_with_mqa.weight"): TensorSpec((KV_LORA + 64, HIDDEN)),
    _name("self_attn.kv_b_proj.weight"): TensorSpec((N_HEADS * (QK_NOPE + V_HEAD), KV_LORA)),
    _name("self_attn.o_proj.weight"): TensorSpec((HIDDEN, N_HEADS * V_HEAD)),
    _name("self_attn.q_proj.weight"): TensorSpec((N_HEADS * (QK_NOPE + 64), HIDDEN)),
    _name("shared_head.head.weight"): TensorSpec((VOCAB, HIDDEN)),
    _name("shared_head.norm.weight"): TensorSpec((HIDDEN,)),
    _name("mlp.gate.e_score_correction_bias"): TensorSpec((N_EXPERTS,)),
    _name("mlp.gate.weight"): TensorSpec((N_EXPERTS, HIDDEN)),
    _name("mlp.shared_experts.down_proj.weight"): TensorSpec((HIDDEN, EXPERT_FF)),
    _name("mlp.shared_experts.gate_proj.weight"): TensorSpec((EXPERT_FF, HIDDEN)),
    _name("mlp.shared_experts.up_proj.weight"): TensorSpec((EXPERT_FF, HIDDEN)),
}


def _expert_name(expert: int, projection: str) -> str:
    return _name(f"mlp.experts.{expert}.{projection}.weight")


def expected_specs() -> dict[str, TensorSpec]:
    """Return the exact 210-tensor, layer-26 manifest key set."""
    specs = dict(NONEXPERT_SPECS)
    for expert in range(N_EXPERTS):
        specs[_expert_name(expert, "down_proj")] = TensorSpec((HIDDEN, EXPERT_FF))
        specs[_expert_name(expert, "gate_proj")] = TensorSpec((EXPERT_FF, HIDDEN))
        specs[_expert_name(expert, "up_proj")] = TensorSpec((EXPERT_FF, HIDDEN))
    return specs


EXPECTED_SPECS = expected_specs()
assert len(NONEXPERT_SPECS) == 18
assert len(EXPECTED_SPECS) == 210


def _require(value: Mapping[str, Any], key: str, expected: Any, errors: list[str], where: str) -> None:
    if value.get(key) != expected:
        errors.append(f"{where}.{key}: expected {expected!r}, got {value.get(key)!r}")


def _as_object(value: Any, name: str, errors: list[str]) -> Mapping[str, Any]:
    if isinstance(value, dict):
        return value
    errors.append(f"{name}: expected object")
    return {}


def _tensor_index(manifest: Mapping[str, Any], errors: list[str]) -> dict[str, Mapping[str, Any]]:
    raw = manifest.get("tensors")
    if not isinstance(raw, list):
        errors.append("tensors: expected list")
        return {}
    by_name: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            errors.append(f"tensors[{index}]: expected object")
            continue
        name = item.get("name")
        if not isinstance(name, str):
            errors.append(f"tensors[{index}].name: expected string")
            continue
        if name in by_name:
            errors.append(f"tensors: duplicate name {name!r}")
            continue
        by_name[name] = item
    return by_name


def _operation(
    source: str | list[str],
    target: str,
    ggml_shape: Sequence[int],
    transform: str,
    *,
    required: bool = True,
) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "source_dtype": "BF16",
        "ggml_logical_shape": list(ggml_shape),
        "transform": transform,
        "required": required,
    }


def gguf_mapping_plan() -> Mapping[str, Any]:
    """Plan only; it is not an instruction to write or mutate a GGUF."""
    p = _name
    block = "blk.26"
    operations = [
        _operation(p("eh_proj.weight"), f"{block}.nextn.eh_proj.weight", (2 * HIDDEN, HIDDEN), "copy_as_ggml_matrix"),
        _operation(p("enorm.weight"), f"{block}.nextn.enorm.weight", (HIDDEN,), "copy"),
        _operation(p("hnorm.weight"), f"{block}.nextn.hnorm.weight", (HIDDEN,), "copy"),
        _operation(p("input_layernorm.weight"), f"{block}.attn_norm.weight", (HIDDEN,), "copy"),
        _operation(p("self_attn.q_proj.weight"), f"{block}.attn_q.weight", (HIDDEN, N_HEADS * (QK_NOPE + 64)), "copy_as_ggml_matrix_direct_q_proj"),
        _operation(p("self_attn.kv_a_proj_with_mqa.weight"), f"{block}.attn_kv_a_mqa.weight", (HIDDEN, KV_LORA + 64), "copy_as_ggml_matrix"),
        _operation(p("self_attn.kv_a_layernorm.weight"), f"{block}.attn_kv_a_norm.weight", (KV_LORA,), "copy"),
        _operation(
            p("self_attn.kv_b_proj.weight"),
            f"{block}.attn_k_b.weight",
            (QK_NOPE, KV_LORA, N_HEADS),
            "view_source_as_[32,320,512]; slice_axis1_[0:128]; transpose_axes_1_2",
        ),
        _operation(
            p("self_attn.kv_b_proj.weight"),
            f"{block}.attn_v_b.weight",
            (KV_LORA, V_HEAD, N_HEADS),
            "view_source_as_[32,320,512]; slice_axis1_[128:320]",
        ),
        _operation(p("self_attn.o_proj.weight"), f"{block}.attn_output.weight", (N_HEADS * V_HEAD, HIDDEN), "copy_as_ggml_matrix"),
        _operation(p("post_attention_layernorm.weight"), f"{block}.ffn_norm.weight", (HIDDEN,), "copy"),
        _operation(p("mlp.gate.weight"), f"{block}.ffn_gate_inp.weight", (HIDDEN, N_EXPERTS), "copy_as_ggml_matrix"),
        _operation(p("mlp.gate.e_score_correction_bias"), f"{block}.ffn_exp_probs_b.bias", (N_EXPERTS,), "copy"),
        _operation(
            [_expert_name(expert, "gate_proj") for expert in range(N_EXPERTS)],
            f"{block}.ffn_gate_exps.weight",
            (HIDDEN, EXPERT_FF, N_EXPERTS),
            "stack_experts_ascending_0_to_63_axis0; reinterpret_as_ggml_dims",
        ),
        _operation(
            [_expert_name(expert, "up_proj") for expert in range(N_EXPERTS)],
            f"{block}.ffn_up_exps.weight",
            (HIDDEN, EXPERT_FF, N_EXPERTS),
            "stack_experts_ascending_0_to_63_axis0; reinterpret_as_ggml_dims",
        ),
        _operation(
            [_expert_name(expert, "down_proj") for expert in range(N_EXPERTS)],
            f"{block}.ffn_down_exps.weight",
            (EXPERT_FF, HIDDEN, N_EXPERTS),
            "stack_experts_ascending_0_to_63_axis0; reinterpret_as_ggml_dims",
        ),
        _operation(p("mlp.shared_experts.gate_proj.weight"), f"{block}.ffn_gate_shexp.weight", (HIDDEN, EXPERT_FF), "copy_as_ggml_matrix"),
        _operation(p("mlp.shared_experts.up_proj.weight"), f"{block}.ffn_up_shexp.weight", (HIDDEN, EXPERT_FF), "copy_as_ggml_matrix"),
        _operation(p("mlp.shared_experts.down_proj.weight"), f"{block}.ffn_down_shexp.weight", (EXPERT_FF, HIDDEN), "copy_as_ggml_matrix"),
        _operation(p("shared_head.norm.weight"), f"{block}.nextn.shared_head_norm.weight", (HIDDEN,), "copy"),
        _operation(p("embed_tokens.weight"), f"{block}.nextn.embed_tokens.weight", (HIDDEN, VOCAB), "optional_alias_only; loader_falls_back_to_root_token_embd", required=False),
        _operation(p("shared_head.head.weight"), f"{block}.nextn.shared_head_head.weight", (HIDDEN, VOCAB), "optional_alias_only; loader_falls_back_to_root_output", required=False),
    ]
    return {
        "pinned_llama_cpp_commit": PINNED_LLAMA_CPP,
        "sources": {
            "loader": PINNED_DEEPSEEK2_CPP,
            "converter": PINNED_DEEPSEEK_CONVERTER,
        },
        "block": MTP_LAYER,
        "operations": operations,
        "required_external_root_tensors": [
            {
                "target": "token_embd.weight",
                "ggml_logical_shape": [HIDDEN, VOCAB],
                "status": "REQUIRED_PHYSICAL_TENSOR_IN_STANDALONE_DRAFT_GGUF",
                "sidecar_copy": p("embed_tokens.weight"),
                "alias_status": "PROVEN_BYTE_IDENTICAL; MTP_BLOCK_COPY_OPTIONAL_IF_ROOT_IS_MATERIALIZED",
            },
            {
                "target": "output.weight",
                "ggml_logical_shape": [HIDDEN, VOCAB],
                "status": "REQUIRED_PHYSICAL_TENSOR_IN_STANDALONE_DRAFT_GGUF",
                "sidecar_copy": p("shared_head.head.weight"),
                "alias_status": "PROVEN_BYTE_IDENTICAL; MTP_BLOCK_COPY_OPTIONAL_IF_ROOT_IS_MATERIALIZED",
            },
            {
                "target": "output_norm.weight",
                "ggml_logical_shape": [HIDDEN],
                "status": "REQUIRED_PHYSICAL_TENSOR_IN_STANDALONE_DRAFT_GGUF; MISSING_FROM_210_TENSOR_SIDECAR",
                "required_source_tensor": "model.norm.weight",
                "required_bf16_bytes": HIDDEN * 2,
                "required_bf16_sha256": ROOT_NORM_BF16_SHA256,
                "not_equal_to": p("shared_head.norm.weight"),
            },
        ],
    }


def preflight_manifest(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate only manifest metadata and return a deterministic plan/report."""
    errors: list[str] = []
    _require(manifest, "schema", "strat01_gigachat_mtp_sidecar_manifest_v1", errors, "manifest")
    source = _as_object(manifest.get("source"), "source", errors)
    sidecar = _as_object(manifest.get("sidecar"), "sidecar", errors)
    _require(source, "repository", REPOSITORY, errors, "source")
    _require(source, "revision", REVISION, errors, "source")
    _require(source, "shard", SHARD, errors, "source")
    _require(source, "lfs_sha256_expected", SOURCE_LFS_SHA256, errors, "source")
    _require(sidecar, "name", SIDECAR_NAME, errors, "sidecar")
    _require(sidecar, "sha256", SIDECAR_SHA256, errors, "sidecar")
    _require(sidecar, "bytes", SIDECAR_BYTES, errors, "sidecar")
    _require(sidecar, "self_read_parity", "PASS_names_shapes_dtypes_offsets", errors, "sidecar")

    actual = _tensor_index(manifest, errors)
    expected_names = set(EXPECTED_SPECS)
    actual_names = set(actual)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing:
        errors.append("tensors: missing exact expected names: " + ", ".join(missing))
    if extra:
        errors.append("tensors: unexpected names: " + ", ".join(extra))
    for name in sorted(expected_names & actual_names):
        spec = EXPECTED_SPECS[name]
        record = actual[name]
        if record.get("dtype") != spec.dtype:
            errors.append(f"{name}.dtype: expected {spec.dtype!r}, got {record.get('dtype')!r}")
        if record.get("shape") != list(spec.shape):
            errors.append(f"{name}.shape: expected {list(spec.shape)!r}, got {record.get('shape')!r}")

    plan = gguf_mapping_plan()
    return {
        "schema": "strat01_gigachat_mtp_gguf_contract_v1",
        "metadata_contract": "PASS" if not errors else "FAIL",
        "manifest_tensor_count": len(actual),
        "expected_tensor_count": len(EXPECTED_SPECS),
        "errors": errors,
        "plan": plan,
        "runtime_blockers": [
            {
                "status": "BLOCKED_IN_PINNED_LLAMA_CPP",
                "reason": "DeepSeek2 graph_mtp asserts q_lora_rank > 0, but n_layer==26 is_lite keeps direct q_proj and q_lora_rank==0.",
                "source": PINNED_DEEPSEEK2_CPP,
            }
        ],
        "limitations": [
            "READ_ONLY_METADATA_PREFLIGHT: the sidecar safetensors payload was not opened or hashed here.",
            "NO_GGUF_WRITTEN.",
            "NO_LOGITS_OR_NUMERICAL_FIDELITY_CLAIM.",
            "NO_ACCEPTANCE_MEASUREMENT.",
            "NO_SPEED_MEASUREMENT.",
            "NO_MODEL_EXECUTION_OR_T4_USE.",
        ],
    }


def load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"manifest {path}: expected JSON object")
    return value


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="read-only GigaChat 3.1 MTP GGUF metadata contract preflight")
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST, help="optional sidecar manifest path")
    args = parser.parse_args(argv)
    try:
        report = preflight_manifest(load_manifest(args.manifest))
    except ContractError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["metadata_contract"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(_main())
