#!/usr/bin/env python3
"""Export the pinned original Qwen2.5-0.5B-Instruct into recipient_qwen_entry.c's packet.

This is an offline format conversion, not an inference or quality test. It reads
only a locally present, revision-pinned BF16 safetensors snapshot. The archive is
an ordered concatenation of 291 named tensor payloads, with zero bytes inserted
so every payload begins at a 64-byte offset. No archive header is stored.

Catalog dtype codes: 1=F32, 2=BF16, 3=W8, 4=W4G32. Vectors have rows=1.
W8 row: little-endian F32 maxabs/127 scale, then signed I8 codes from nearest-
even round(W/scale), clipped to [-127,127]. All-zero rows have scale and codes 0.
W4 group of 32 consecutive input columns: little-endian FP16-rounded maxabs/7
scale, then 16 packed bytes. The low nibble codes input 0..15 and the high
nibble codes input 16..31; code=nearest-even round(W/decoded FP16 scale),
clipped [-7,7], plus 8. All-zero blocks have scale 0 and 0x88 bytes. A nonzero
block whose FP16 scale underflows to zero is rejected. The original tied BF16
embedding is retained and is also the source of an explicit W8 deployment head.

CLI: --source MODEL.SAFETENSORS --out ARCHIVE --catalog HEADER --manifest JSON
     [--threads 6]
Output paths must be distinct and absent. A receipt is published last; its
absence means that no emitted artifact set has completed. No network or model
execution occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

# The frozen controller uses Python -I -S: bundled dependencies are not on
# sys.path unless this workspace-local site directory is inserted explicitly.
ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "results" / "native_expert_scaling" / "chatbot_source_runtime" / "site"
if not SITE.is_dir():
    raise RuntimeError(f"missing bundled Python site: {SITE}")
sys.path.insert(0, str(SITE))

import numpy as np
import torch
from safetensors import safe_open

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SOURCE_SHA = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
L, D, H, V, KV = 24, 896, 4864, 151936, 128
SCHEMA = "RECIPIENT_MIXED_CATALOG_V1"


def demand(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fields() -> list[dict]:
    spec = [dict(name="model.embed_tokens.weight", source_name="model.embed_tokens.weight", dtype=2, rows=V, cols=D),
            dict(name="model.norm.weight", source_name="model.norm.weight", dtype=1, rows=1, cols=D),
            dict(name="lm_head.weight", source_name="model.embed_tokens.weight", dtype=3, rows=V, cols=D)]
    for layer in range(L):
        prefix = f"model.layers.{layer}."
        for suffix, dtype, rows, cols in (
            ("self_attn.q_proj.weight", 3, D, D),
            ("self_attn.k_proj.weight", 3, KV, D),
            ("self_attn.v_proj.weight", 3, KV, D),
            ("self_attn.o_proj.weight", 3, D, D),
            ("self_attn.q_proj.bias", 1, 1, D),
            ("self_attn.k_proj.bias", 1, 1, KV),
            ("self_attn.v_proj.bias", 1, 1, KV),
            ("input_layernorm.weight", 1, 1, D),
            ("post_attention_layernorm.weight", 1, 1, D),
            ("mlp.gate_proj.weight", 4, H, D),
            ("mlp.up_proj.weight", 4, H, D),
            ("mlp.down_proj.weight", 4, D, H),
        ):
            name = prefix + suffix
            spec.append(dict(name=name, source_name=name, dtype=dtype, rows=rows, cols=cols))
    demand(len(spec) == 291 and len({s["name"] for s in spec}) == 291, "role count/uniqueness")
    return spec


def expected_bytes(field: dict) -> int:
    rows, cols, kind = (field[x] for x in ("rows", "cols", "dtype"))
    if kind == 1:
        return rows * cols * 4
    if kind == 2:
        return rows * cols * 2
    if kind == 3:
        return rows * (cols + 4)
    demand(cols % 32 == 0, f"unaligned W4 columns: {field['name']}")
    return rows * (cols // 32) * 18


def source_config(snapshot: Path) -> dict:
    config_path = snapshot / "config.json"
    demand(config_path.is_file(), "missing local config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = dict(model_type="qwen2", hidden_size=D, intermediate_size=H,
                    num_hidden_layers=L, num_attention_heads=14, num_key_value_heads=2,
                    vocab_size=V, tie_word_embeddings=True, hidden_act="silu",
                    rms_norm_eps=1e-6, rope_theta=1_000_000.0,
                    use_sliding_window=False)
    for key, value in expected.items():
        demand(config.get(key) == value, f"config mismatch: {key}")
    demand(D // config["num_attention_heads"] == 64, "head dimension")
    return config


def tensor(source: safe_open, field: dict) -> torch.Tensor:
    value = source.get_tensor(field["source_name"])
    shape = (field["rows"], field["cols"]) if field["dtype"] != 1 else (field["cols"],)
    demand(tuple(value.shape) == shape, f"source shape: {field['source_name']}")
    demand(value.dtype == torch.bfloat16 and value.device.type == "cpu", f"source BF16: {field['source_name']}")
    return value.contiguous()


def encode_f32(value: torch.Tensor, field: dict):
    values = value.float().numpy().astype("<f4", copy=False)
    demand(np.isfinite(values).all(), f"nonfinite source: {field['source_name']}")
    yield values.tobytes(order="C")


def encode_bf16(value: torch.Tensor, field: dict):
    for start in range(0, value.shape[0], 1024):
        chunk = value[start:start + 1024].contiguous()
        demand(torch.isfinite(chunk.float()).all().item(), f"nonfinite source: {field['source_name']}")
        yield chunk.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes(order="C")


def encode_w8(value: torch.Tensor, field: dict):
    cols = field["cols"]
    for start in range(0, value.shape[0], 256):
        w = value[start:start + 256].float().numpy()
        demand(np.isfinite(w).all(), f"nonfinite source: {field['source_name']}")
        maximum = np.max(np.abs(w), axis=1).astype(np.float32)
        scale = (maximum / np.float32(127.0)).astype("<f4")
        safe = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)
        q = np.clip(np.rint(w / safe[:, None]), -127, 127).astype(np.int8)
        q[scale == 0] = 0
        packed = np.empty((w.shape[0], cols + 4), dtype=np.uint8)
        packed[:, :4] = scale.view(np.uint8).reshape(-1, 4)
        packed[:, 4:] = q.view(np.uint8)
        yield packed.tobytes(order="C")


def encode_w4(value: torch.Tensor, field: dict):
    groups = field["cols"] // 32
    for start in range(0, value.shape[0], 64):
        w = value[start:start + 64].float().numpy().reshape(-1, groups, 32)
        demand(np.isfinite(w).all(), f"nonfinite source: {field['source_name']}")
        maximum = np.max(np.abs(w), axis=2).astype(np.float32)
        scale16 = (maximum / np.float32(7.0)).astype("<f2")
        scale = scale16.astype(np.float32)
        demand(np.isfinite(scale).all() and np.all((maximum == 0) | (scale > 0)),
               f"FP16 W4 scale underflow/overflow: {field['source_name']}")
        safe = np.where(scale > 0, scale, np.float32(1.0))
        q = np.clip(np.rint(w / safe[:, :, None]), -7, 7).astype(np.int8) + np.int8(8)
        packed = np.empty((w.shape[0], groups, 18), dtype=np.uint8)
        packed[:, :, :2] = scale16.view(np.uint8).reshape(w.shape[0], groups, 2)
        packed[:, :, 2:] = q[:, :, :16].astype(np.uint8) | (q[:, :, 16:].astype(np.uint8) << 4)
        yield packed.tobytes(order="C")


def write_archive(path: Path, source: safe_open, spec: list[dict]) -> tuple[int, str]:
    encoders = {1: encode_f32, 2: encode_bf16, 3: encode_w8, 4: encode_w4}
    overall = hashlib.sha256()
    with path.open("xb") as stream:
        for field in spec:
            offset = stream.tell()
            padding = (-offset) % 64
            if padding:
                zero = bytes(padding)
                stream.write(zero)
                overall.update(zero)
            field["offset"] = stream.tell()
            checksum = hashlib.sha256()
            count = 0
            value = tensor(source, field)
            for chunk in encoders[field["dtype"]](value, field):
                stream.write(chunk)
                checksum.update(chunk)
                overall.update(chunk)
                count += len(chunk)
            field["bytes"] = count
            field["sha256"] = checksum.hexdigest()
            demand(count == expected_bytes(field), f"emitted byte count: {field['name']}")
            del value
        stream.flush()
        os.fsync(stream.fileno())
        total = stream.tell()
    return total, overall.hexdigest()


def header_text(spec: list[dict], total: int, archive_sha: str) -> str:
    lines = ["#ifndef RECIPIENT_MIXED_CATALOG_H", "#define RECIPIENT_MIXED_CATALOG_H",
             "#include <stdint.h>",
             "typedef struct { const char *name; uint32_t dtype; uint64_t rows, cols, offset, bytes; } RcField;",
             f"#define RC_FIELD_COUNT {len(spec)}",
             f"#define RC_ARCHIVE_BYTES UINT64_C({total})",
             f'#define RC_ARCHIVE_SHA "{archive_sha}"',
             "static const RcField rc_fields[RC_FIELD_COUNT] = {"]
    for field in spec:
        lines.append('    {"%s", %d, UINT64_C(%d), UINT64_C(%d), UINT64_C(%d), UINT64_C(%d)},' %
                     tuple(field[key] for key in ("name", "dtype", "rows", "cols", "offset", "bytes")))
    return "\n".join(lines + ["};", "#endif", ""])


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def write_bytes(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def publish(temp: Path, final: Path) -> None:
    # Hard links provide an exclusive publication step on the same volume.
    os.link(temp, final)
    temp.unlink()


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    # Keep the snapshot-path spelling even when an HF cache entry is a symlink
    # into blobs; Path.resolve() would erase the pinned revision directory.
    source_path = Path(os.path.abspath(args.source))
    snapshot = source_path.parent
    demand(snapshot.is_dir() and snapshot.name == REVISION, "unpinned snapshot directory")
    demand(source_path.is_file() and source_path.name == "model.safetensors", "missing local model.safetensors")
    actual_source_sha = sha_file(source_path)
    demand(actual_source_sha == SOURCE_SHA, "source SHA mismatch")
    source_config(snapshot)
    config_path = snapshot / "config.json"
    spec = fields()
    destinations = [args.out, args.catalog, args.manifest]
    resolved = [path.resolve(strict=False) for path in destinations]
    demand(len(set(resolved)) == 3, "output paths must be distinct")
    for target in destinations:
        demand(target.parent.is_dir() and not target.exists(), f"output exists or parent missing: {target}")
    script_path = Path(__file__).resolve(strict=True)
    script_sha = sha_file(script_path)
    with safe_open(str(source_path), framework="pt", device="cpu") as source:
        source_keys = set(source.keys())
        expected = {field["source_name"] for field in spec}
        demand(len(expected) == 290 and source_keys == expected,
               f"source tensor inventory mismatch: missing={sorted(expected-source_keys)}, extra={sorted(source_keys-expected)}")
        with tempfile.TemporaryDirectory(prefix="recipient-mixed-", dir=str(args.out.parent)) as temporary:
            stage = Path(temporary)
            temp_archive = stage / "archive.bin"
            total, archive_sha = write_archive(temp_archive, source, spec)
            public_fields = [{key: field[key] for key in ("name", "dtype", "rows", "cols", "offset", "bytes")}
                             for field in spec]
            manifest = dict(schema=SCHEMA, source_model=MODEL, source_revision=REVISION,
                            sourcepath=str(source_path), source_sha256=SOURCE_SHA,
                            config_path=str(config_path), config_sha256=sha_file(config_path),
                            archive_path=str(args.out.resolve()), archive_bytes=total,
                            archive_sha256=archive_sha, fields=public_fields,
                            field_sha256={field["name"]: field["sha256"] for field in spec},
                            source_tensor_count=len(source_keys),
                            codec_counts={str(kind): sum(field["dtype"] == kind for field in spec)
                                          for kind in (1, 2, 3, 4)},
                            exporter_path=str(script_path),
                            exporter_sha256=script_sha, numerical_test_performed=False)
            temp_header = stage / "catalog.h"
            temp_manifest = stage / "manifest.json"
            write_bytes(temp_header, header_text(spec, total, archive_sha).encode("ascii"))
            manifest["catalog_path"] = str(args.catalog.resolve())
            manifest["catalog_bytes"] = temp_header.stat().st_size
            manifest["catalog_sha256"] = sha_file(temp_header)
            write_bytes(temp_manifest, json_bytes(manifest))
            # Verify all staged bytes before making any output visible.
            demand(sha_file(temp_archive) == archive_sha, "staged archive SHA")
            for temp, final in ((temp_archive, args.out), (temp_header, args.catalog),
                                (temp_manifest, args.manifest)):
                publish(temp, final)
    print(json.dumps({"status": "COMPLETE", "archive_sha256": archive_sha,
                      "archive_bytes": total, "roles": len(spec), "manifest": str(args.manifest.resolve())}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=6, choices=range(1, 7))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
