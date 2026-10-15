#!/usr/bin/env python3
"""Header-only, pinned Granite 4 H Tiny instruct active-byte preflight.

No tensor payload is requested. Byte scenarios are analytical, not an engine
format, observed DRAM traffic, throughput, or donor-quality result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

REPO = "ibm-granite/granite-4.0-h-tiny"
ORIGIN = "https://huggingface.co/"
MAX_FILE = 2 * 1024 * 1024
MAX_TOTAL = 16 * 1024 * 1024
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1}
LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


class PreflightError(RuntimeError):
    pass


class Budget:
    def __init__(self):
        self.used = 0

    def take(self, n):
        if n < 0 or n > MAX_FILE or self.used + n > MAX_TOTAL:
            raise PreflightError(f"metadata cap exceeded: request={n}, consumed={self.used}")
        self.used += n


def read_json(url, budget):
    # The API/config/index are small text metadata. Enforce the limit while
    # reading; a missing or deceptive Content-Length cannot bypass the cap.
    with urlopen(Request(url, headers={"Accept-Encoding": "identity"}), timeout=30) as h:
        size = h.headers.get("Content-Length")
        if size is not None and int(size) > MAX_FILE:
            raise PreflightError(f"oversize JSON metadata: {url}")
        remaining = min(MAX_FILE, MAX_TOTAL - budget.used)
        data = h.read(remaining + 1)
    if len(data) > remaining:
        raise PreflightError(f"oversize JSON metadata: {url}")
    budget.take(len(data))
    return json.loads(data), data


def strict_range(url, first, last, budget):
    """Check 206 and exact Content-Range before reading any response body."""
    length = last - first + 1
    if first < 0 or length <= 0:
        raise PreflightError("invalid range")
    budget.take(length)
    req = Request(url, headers={"Range": f"bytes={first}-{last}", "Accept-Encoding": "identity"})
    with urlopen(req, timeout=30) as h:
        raw = h.headers.get("Content-Range", "")
        match = RANGE.fullmatch(raw)
        if h.status != 206 or match is None or (int(match[1]), int(match[2])) != (first, last):
            raise PreflightError(f"range refused before body read: status={h.status}, Content-Range={raw!r}")
        if h.headers.get("Content-Encoding", "identity") != "identity":
            raise PreflightError("range response encoded")
        total = int(match[3])
        if total <= last:
            raise PreflightError("invalid remote total size")
        declared = h.headers.get("Content-Length")
        if declared is not None and int(declared) != length:
            raise PreflightError("range response length header mismatch")
        body = h.read(length + 1)
    if len(body) != length:
        raise PreflightError("range body length mismatch")
    return body, total


def product(shape):
    if not isinstance(shape, list) or not shape or any(type(x) is not int or x <= 0 for x in shape):
        raise PreflightError(f"invalid tensor shape: {shape}")
    return math.prod(shape)


def header_for(shard, rev, budget):
    url = ORIGIN + REPO + "/resolve/" + rev + "/" + shard
    prefix, total = strict_range(url, 0, 7, budget)
    header_length = struct.unpack("<Q", prefix)[0]
    if header_length <= 0 or header_length + 8 > MAX_FILE:
        raise PreflightError(f"oversize/empty safetensors header: {shard}: {header_length}")
    body, second_total = strict_range(url, 8, header_length + 7, budget)
    if second_total != total:
        raise PreflightError(f"remote file size changed during header read: {shard}")
    descriptor = json.loads(body)
    if not isinstance(descriptor, dict):
        raise PreflightError(f"invalid safetensors header: {shard}")
    intervals = []
    for name, item in descriptor.items():
        if name == "__metadata__":
            continue
        if not isinstance(item, dict) or item.get("dtype") not in DTYPE_BYTES:
            raise PreflightError(f"unsupported tensor dtype/descriptor: {name}")
        shape = item.get("shape")
        offsets = item.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
            raise PreflightError(f"invalid tensor offsets: {name}")
        a, b = offsets
        if a < 0 or b <= a or b - a != product(shape) * DTYPE_BYTES[item["dtype"]]:
            raise PreflightError(f"tensor byte arithmetic mismatch: {name}")
        intervals.append((a, b))
    intervals.sort()
    cursor = 0
    for a, b in intervals:
        if a != cursor:
            raise PreflightError(f"gap/overlap in {shard} at {cursor}")
        cursor = b
    if cursor + header_length + 8 != total:
        raise PreflightError(f"safetensors payload/file size mismatch: {shard}")
    return descriptor, {"shard": shard, "remote_bytes": total, "header_bytes": header_length,
                        "header_sha256": hashlib.sha256(body).hexdigest(), "tensor_count": len(intervals)}, body


def classify(name, shape, config):
    hidden = config["hidden_size"]
    experts = config["num_local_experts"]
    expert_hidden = config["intermediate_size"]
    shared_hidden = config["shared_intermediate_size"]
    if name == "model.embed_tokens.weight":
        if shape != [config["vocab_size"], config["hidden_size"]]:
            raise PreflightError("embedding dimensions differ from config")
        return "head_embedding", 1, True
    if name == "lm_head.weight":
        if config["tie_word_embeddings"]:
            raise PreflightError("explicit lm_head tensor with tied embeddings needs alias audit")
        return "head_embedding", 1, True
    if name == "model.norm.weight":
        if shape != [hidden]:
            raise PreflightError("final norm shape mismatch")
        return "protected_norm_state", 1, False
    match = LAYER.fullmatch(name)
    if match is None:
        raise PreflightError(f"unclassified tensor: {name}")
    layer, suffix = int(match[1]), match[2]
    if layer >= config["num_hidden_layers"]:
        raise PreflightError(f"out-of-range layer: {name}")
    layer_type = config["layer_types"][layer]
    if suffix in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        if shape != [hidden]:
            raise PreflightError(f"layer norm shape mismatch: {name}")
        return "protected_norm_state", 1, False
    if suffix == "block_sparse_moe.router.layer.weight":
        if shape != [config["num_local_experts"], config["hidden_size"]]:
            raise PreflightError(f"router shape mismatch: {name}")
        return "router", 1, True
    if suffix in ("block_sparse_moe.input_linear.weight", "block_sparse_moe.output_linear.weight"):
        expected = ([experts, 2 * expert_hidden, hidden] if suffix.endswith("input_linear.weight")
                    else [experts, hidden, expert_hidden])
        if shape != expected:
            raise PreflightError(f"routed expert shape mismatch: {name}")
        return "routed", config["num_experts_per_tok"], True
    if suffix in ("shared_mlp.input_linear.weight", "shared_mlp.output_linear.weight"):
        expected = ([2 * shared_hidden, hidden] if suffix.endswith("input_linear.weight")
                    else [hidden, shared_hidden])
        if shape != expected:
            raise PreflightError(f"shared expert shape mismatch: {name}")
        return "shared", 1, True
    if suffix in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight"):
        if layer_type != "attention":
            raise PreflightError(f"attention tensor at {layer_type} layer: {name}")
        kv_width = config["num_key_value_heads"] * (hidden // config["num_attention_heads"])
        expected = [kv_width if suffix in ("self_attn.k_proj.weight", "self_attn.v_proj.weight") else hidden, hidden]
        if shape != expected:
            raise PreflightError(f"attention projection shape mismatch: {name}")
        return "attention", 1, True
    if suffix in ("mamba.in_proj.weight", "mamba.out_proj.weight"):
        if layer_type != "mamba":
            raise PreflightError(f"Mamba tensor at {layer_type} layer: {name}")
        expanded = hidden * config["mamba_expand"]
        in_width = 2 * expanded + 2 * config["mamba_n_groups"] * config["mamba_d_state"] + config["mamba_n_heads"]
        expected = [in_width, hidden] if suffix == "mamba.in_proj.weight" else [hidden, expanded]
        if shape != expected:
            raise PreflightError(f"Mamba projection shape mismatch: {name}")
        return "ssm_projection", 1, True
    if suffix in ("mamba.A_log", "mamba.D", "mamba.conv1d.bias", "mamba.conv1d.weight",
                  "mamba.dt_bias", "mamba.norm.weight"):
        if layer_type != "mamba":
            raise PreflightError(f"Mamba control at {layer_type} layer: {name}")
        expanded = hidden * config["mamba_expand"]
        conv_width = expanded + 2 * config["mamba_n_groups"] * config["mamba_d_state"]
        expected = {
            "mamba.A_log": [config["mamba_n_heads"]],
            "mamba.D": [config["mamba_n_heads"]],
            "mamba.dt_bias": [config["mamba_n_heads"]],
            "mamba.conv1d.bias": [conv_width],
            "mamba.conv1d.weight": [conv_width, 1, config["mamba_d_conv"]],
            "mamba.norm.weight": [expanded],
        }[suffix]
        if shape != expected:
            raise PreflightError(f"Mamba control shape mismatch: {name}")
        return "protected_norm_state", 1, False
    raise PreflightError(f"unclassified tensor: {name}")


def packed_bytes(shape, bits):
    # One row is the last dimension, with 128-wide F32 scale groups.
    rows, columns = math.prod(shape[:-1]), shape[-1]
    return rows * ((columns * bits + 7) // 8 + 4 * ((columns + 127) // 128))


def runtime_state_scenarios(config):
    """Logical cache accounting for a proposed F32-state/F16-KV runtime.

    These are addressed reads/writes, not a claim that every byte crosses
    DRAM: cache residence, compiler stores and native formats are unmeasured.
    """
    recurrent = config["layer_types"].count("mamba")
    attention = config["layer_types"].count("attention")
    channels = config["mamba_n_heads"] * config["mamba_d_head"]
    if channels != config["hidden_size"] * config["mamba_expand"]:
        raise PreflightError("Mamba head/channel dimensions disagree")
    state = recurrent * channels * config["mamba_d_state"] * 4
    conv_channels = channels + 2 * config["mamba_n_groups"] * config["mamba_d_state"]
    ring = recurrent * conv_channels * config["mamba_d_conv"] * 4
    kv_row = attention * 2 * config["num_key_value_heads"] * (config["hidden_size"] // config["num_attention_heads"]) * 2
    return {
        "assumptions": "F32 SSM state and full conv ring read/write once; F16 full-attention KV. Logical accounting, not source execution or physical DRAM measurement.",
        "persistent_ssm_state_bytes": state, "persistent_conv_ring_bytes": ring,
        "ssm_state_logical_read_write_bytes_per_token": 2 * state,
        "conv_ring_full_logical_read_write_bytes_per_token": 2 * ring,
        "attention_KV_bytes_per_cached_position": kv_row,
        "contexts": {str(context): {
            "KV_logical_read_bytes_per_token": context * kv_row,
            "KV_append_bytes_per_token": kv_row,
            "extra_logical_bytes_per_token": 2 * state + 2 * ring + (context + 1) * kv_row,
        } for context in (128, 512, 2048, 8192)},
        "unpriced": ["activation scratch", "quantizer or LUT construction", "dispatch and sampling", "intermediate cache traffic", "actual memory-controller traffic"],
    }


def ledger(config, index, headers, api):
    required = {"hidden_size": 1536, "num_hidden_layers": 40, "num_local_experts": 64,
                "num_experts_per_tok": 6, "tie_word_embeddings": True, "vocab_size": 100352}
    for key, expected in required.items():
        if config.get(key) != expected:
            raise PreflightError(f"unexpected config {key}: {config.get(key)!r}")
    if config.get("model_type") != "granitemoehybrid" or len(config.get("layer_types", [])) != 40:
        raise PreflightError("unexpected model/layer topology")
    if config["layer_types"].count("attention") != 4 or config["layer_types"].count("mamba") != 36:
        raise PreflightError("unexpected mixer counts")
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise PreflightError("missing/empty safetensors weight_map")
    indexed_shards = set(mapping.values())
    if indexed_shards != set(headers):
        raise PreflightError("incomplete or extra index shards")
    actual = {}
    for shard, header in headers.items():
        for name, item in header.items():
            if name == "__metadata__":
                continue
            if name in actual:
                raise PreflightError(f"duplicate tensor across shards: {name}")
            actual[name] = (shard, item)
    if set(actual) != set(mapping):
        raise PreflightError(f"index/header tensor-set mismatch: index={len(mapping)}, header={len(actual)}")
    if any(actual[name][0] != shard for name, shard in mapping.items()):
        raise PreflightError("index/header shard mapping mismatch")
    common = {"block_sparse_moe.input_linear.weight", "block_sparse_moe.output_linear.weight",
              "block_sparse_moe.router.layer.weight", "input_layernorm.weight",
              "post_attention_layernorm.weight", "shared_mlp.input_linear.weight",
              "shared_mlp.output_linear.weight"}
    mamba = {"mamba.A_log", "mamba.D", "mamba.conv1d.bias", "mamba.conv1d.weight",
             "mamba.dt_bias", "mamba.in_proj.weight", "mamba.norm.weight", "mamba.out_proj.weight"}
    attention = {"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                 "self_attn.o_proj.weight"}
    expected_names = {"model.embed_tokens.weight", "model.norm.weight"}
    for layer, kind in enumerate(config["layer_types"]):
        expected_names.update(f"model.layers.{layer}.{suffix}" for suffix in common | (mamba if kind == "mamba" else attention))
    if set(actual) != expected_names:
        raise PreflightError(f"topology tensor-set mismatch: expected={len(expected_names)}, actual={len(actual)}")
    totals = defaultdict(lambda: defaultdict(int))
    scenarios = {fmt: defaultdict(int) for fmt in ("BF16", "I8", "I4", "I3", "I2")}
    total_params = 0
    stored_expert_params = 0
    for name, (_, item) in actual.items():
        shape, dtype = item["shape"], item["dtype"]
        n = product(shape)
        role, selected, quantizable = classify(name, shape, config)
        if role == "routed":
            stored_expert_params += n
            active = n // config["num_local_experts"] * selected
            active_shape = [selected] + shape[1:]
        else:
            active = n
            active_shape = shape
            if role == "shared":
                stored_expert_params += n
        total_params += n
        totals[role]["stored_coefficients"] += n
        totals[role]["active_coefficients"] += active
        totals[role]["original_active_bytes"] += active * DTYPE_BYTES[dtype]
        for fmt, role_bytes in scenarios.items():
            if quantizable:
                if fmt == "BF16":
                    value = active * 2
                else:
                    value = packed_bytes(active_shape, int(fmt[1:]))
            else:
                value = active * DTYPE_BYTES[dtype]
            role_bytes[role] += value
    expected_total = api.get("safetensors", {}).get("total")
    if total_params != expected_total:
        raise PreflightError(f"API safetensors parameter mismatch: {total_params} != {expected_total}")
    if index.get("metadata", {}).get("total_size") != sum(
        product(item["shape"]) * DTYPE_BYTES[item["dtype"]] for _, item in actual.values()
    ):
        raise PreflightError("index total_size differs from tensor headers")
    if config["tie_word_embeddings"] and "lm_head.weight" in actual:
        raise PreflightError("tied head unexpectedly explicit")
    if "model.embed_tokens.weight" not in actual:
        raise PreflightError("tied head embedding tensor missing")
    result_scenarios = {}
    for fmt, roles in scenarios.items():
        active = sum(roles.values())
        result_scenarios[fmt] = {
            "active_bytes": active,
            "by_role_bytes": dict(sorted(roles.items())),
            "ideal_stream_GB_per_s_for_14ms": active / 0.014 / 1e9,
            "ideal_stream_GB_per_s_for_20ms": active / 0.020 / 1e9,
            "ideal_read_ms_at_40GB_per_s": active / 40e9 * 1000,
            "below_560MB_14ms_subbudget": active <= 560_000_000,
            "below_800MB_20ms_total_ideal": active <= 800_000_000,
        }
    # Preserve the source router representation and source top-k in these
    # hypotheses. Unlike uniform rows, they expose which learned functions
    # must tolerate aggressive compression to leave any execution margin.
    policies = {
        "experts2_core3_head4_source_router": {
            "routed": "I2", "shared": "I3", "ssm_projection": "I3",
            "head_embedding": "I4", "attention": "I4", "router": "BF16",
            "protected_norm_state": "BF16"},
        "experts3_core2_head4_source_router": {
            "routed": "I3", "shared": "I3", "ssm_projection": "I2",
            "head_embedding": "I4", "attention": "I4", "router": "BF16",
            "protected_norm_state": "BF16"},
    }
    mixed = {}
    for label, policy in policies.items():
        roles = {role: scenarios[fmt][role] for role, fmt in policy.items()}
        active = sum(roles.values())
        mixed[label] = {
            "formats_by_role": policy, "by_role_bytes": roles,
            "active_bytes": active, "source_top_k_unchanged": True,
            "ideal_read_ms_at_40GB_per_s": active / 40e9 * 1000,
            "ideal_stream_GB_per_s_for_14ms": active / 0.014 / 1e9,
            "below_560MB_14ms_subbudget": active <= 560_000_000,
            "implemented_format": False, "quality_known": False,
            "decision": "BYTE_SCREEN_ONLY_REQUIRES_VALUE_CALIBRATION_AND_KERNEL",
        }
    return {"tensor_count": len(actual), "total_distinct_parameters": total_params,
            "distinct_expert_parameters_including_shared": stored_expert_params,
            "active_coefficients_per_token": sum(x["active_coefficients"] for x in totals.values()),
            "by_role_coefficients": {role: dict(row) for role, row in sorted(totals.items())},
            "scenarios": result_scenarios, "mixed_scenarios": mixed,
            "runtime_state_scenarios": runtime_state_scenarios(config),
            "semantics": {key: config[key] for key in (
                "layer_types", "num_local_experts", "num_experts_per_tok", "shared_intermediate_size",
                "intermediate_size", "tie_word_embeddings", "vocab_size", "hidden_size", "mamba_d_state",
                "mamba_d_conv", "mamba_expand", "mamba_n_heads", "mamba_n_groups", "attention_multiplier",
                "residual_multiplier", "embedding_multiplier", "logits_scaling", "position_embedding_type")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/native_expert_scaling/conditional_donor_preflight_20261015"))
    parser.add_argument("--revision", help="Immutable official checkpoint revision; omit only for initial discovery")
    args = parser.parse_args()
    if args.out.exists():
        raise PreflightError("output already exists; use a new observation directory")
    budget = Budget()
    # No files are written until all headers, mappings and byte arithmetic pass.
    if args.revision is not None and re.fullmatch(r"[0-9a-f]{40}", args.revision) is None:
        raise PreflightError("revision must be a complete 40-hex commit")
    api_url = ORIGIN + "api/models/" + REPO
    if args.revision is not None:
        api_url += "/revision/" + args.revision
    api, api_bytes = read_json(api_url, budget)
    rev = api.get("sha")
    if not isinstance(rev, str) or re.fullmatch(r"[0-9a-f]{40}", rev) is None:
        raise PreflightError("official API did not return an immutable 40-hex revision")
    if args.revision is not None and rev != args.revision:
        raise PreflightError("official API returned a different revision")
    prefix = ORIGIN + REPO + "/resolve/" + rev + "/"
    config, config_bytes = read_json(prefix + "config.json", budget)
    index, index_bytes = read_json(prefix + "model.safetensors.index.json", budget)
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise PreflightError("missing complete shard index")
    shards = sorted(set(mapping.values()))
    if not shards or any(re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", s) is None for s in shards):
        raise PreflightError("unexpected shard names")
    headers, manifests, raw_headers = {}, [], {}
    for shard in shards:
        headers[shard], manifest, raw_headers[shard] = header_for(shard, rev, budget)
        manifests.append(manifest)
    prices = ledger(config, index, headers, api)
    output = {"schema": "CONDITIONAL_DONOR_METADATA_V2", "status": "complete_header_only", "repo": REPO, "revision": rev,
              "metadata_bytes_read": budget.used, "shards": manifests,
              "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
              "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
              "method": "Pinned HF safetensors headers only; all weight_map names/shards, dtype/shape/offset bytes, and complete file lengths checked. All matrix format scenarios use last-axis groups of 128 with one F32 scale/group and row byte padding. Protected state/norm/conv remain source dtype. Tied embedding is charged once as a full output head; no second full embedding stream.",
              **prices}
    # Each successful observation has an exclusive destination. Do not erase
    # a previous completed ledger when rediscovering a moving API revision.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    work = args.out.with_name(args.out.name + ".staging-" + stamp)
    work.mkdir(parents=True, exist_ok=False)
    (work / "api.json").write_bytes(api_bytes)
    (work / "config.json").write_bytes(config_bytes)
    (work / "model.safetensors.index.json").write_bytes(index_bytes)
    for shard, raw in raw_headers.items():
        (work / (shard + ".header.json")).write_bytes(raw)
    (work / "preflight.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    work.rename(args.out)
    print(json.dumps({"status": output["status"], "revision": rev,
                      "active_coefficients": output["active_coefficients_per_token"],
                      "scenario_bytes": {k: v["active_bytes"] for k, v in output["scenarios"].items()},
                      "out": str(args.out)}))


if __name__ == "__main__":
    try:
        main()
    except (PreflightError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        # Preserve failures beside the requested output, never inside an
        # existing completed observation. Retain any failed staging files.
        failure_args = argparse.ArgumentParser(add_help=False)
        failure_args.add_argument("--out", type=Path, default=Path("results/native_expert_scaling/conditional_donor_preflight_20261015"))
        failure_out, _ = failure_args.parse_known_args()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        receipt_dir = failure_out.out.with_name(failure_out.out.name + ".failed-" + timestamp)
        receipt_dir.mkdir(parents=True, exist_ok=False)
        (receipt_dir / "failure.json").write_text(json.dumps({
            "status": "failed_attempt_not_admitted", "error_type": type(exc).__name__,
            "error": str(exc), "timestamp_utc": timestamp, "requested_output": str(failure_out.out),
            "staging_policy": "Any staging files are retained as incomplete; no partial observation is published.",
        }, indent=2) + "\n", encoding="utf-8")
        print(f"PREFLIGHT FAILED (no partial ledger): {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
