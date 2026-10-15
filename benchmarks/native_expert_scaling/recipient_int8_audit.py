#!/usr/bin/env python3
"""Audit a complete QWENDON1/R8 Instruct export and saved native logits.

This is a CPU-only, offline decoded-wire check. It does not execute the C engine,
assess chatbot quality, or measure decoding speed.
"""

import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import sys
import time

# Set these before importing any numerical/model package, including transitive imports.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                       "results/native_expert_scaling/chatbot_source_runtime/site"))

import numpy as np
import psutil
import torch
from safetensors import safe_open


REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SOURCE_SHA = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
GEOMETRY = (896, 4864, 24, 14, 2, 64, 151936, 0, 5)
HEADER = struct.Struct("<8s9i2f")
TIME_CAP = 900.0
RSS_CAP = 8 * (1 << 30)
OUTPUT_CAP = 2 * (1 << 20)


def sha256(path, guard):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
            guard()
    return digest.hexdigest()


def fail_unless(condition, message):
    if not condition:
        raise ValueError(message)


def source_header(path, guard):
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        fail_unless(0 < length < (1 << 22), "invalid safetensors header length")
        raw = handle.read(length)
    header = json.loads(raw)
    tensors = {name: spec for name, spec in header.items() if name != "__metadata__"}
    fail_unless(len(tensors) == 290, f"source tensor count {len(tensors)} != 290")
    extents = []
    for name, spec in tensors.items():
        fail_unless(spec["dtype"] == "BF16", f"non-BF16 source tensor: {name}")
        shape = tuple(spec["shape"])
        lo, hi = spec["data_offsets"]
        fail_unless(lo >= 0 and hi - lo == 2 * int(np.prod(shape)), f"source extent: {name}")
        extents.append((lo, hi))
    extents.sort()
    cursor = 0
    for lo, hi in extents:
        fail_unless(lo == cursor, "source tensor gap or overlap")
        cursor = hi
    fail_unless(8 + length + cursor == path.stat().st_size, "source trailing bytes")
    guard()
    return tensors


def expected_shapes():
    d, f, layers, nh, nkv, hd, vocab, _, _ = GEOMETRY
    shapes = {"model.embed_tokens.weight": (vocab, d), "model.norm.weight": (d,)}
    for layer in range(layers):
        pre = f"model.layers.{layer}."
        shapes[pre + "input_layernorm.weight"] = (d,)
        shapes[pre + "post_attention_layernorm.weight"] = (d,)
        for organ, out, inn in (
            ("q_proj", nh * hd, d), ("k_proj", nkv * hd, d),
            ("v_proj", nkv * hd, d), ("o_proj", d, nh * hd),
        ):
            shapes[pre + "self_attn." + organ + ".weight"] = (out, inn)
            if organ in ("q_proj", "k_proj", "v_proj"):
                shapes[pre + "self_attn." + organ + ".bias"] = (out,)
        for organ, out, inn in (("gate_proj", f, d), ("up_proj", f, d), ("down_proj", d, f)):
            shapes[pre + "mlp." + organ + ".weight"] = (out, inn)
    fail_unless(len(shapes) == 290, "internal shape census")
    return shapes


class Archive:
    def __init__(self, path, guard):
        self.handle = path.open("rb")
        self.mm = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)
        self.size = len(self.mm)
        self.pos = 0
        self.guard = guard
        fail_unless(self.size >= HEADER.size, "archive too short")
        magic, *fields = HEADER.unpack_from(self.mm)
        fail_unless(magic == b"QWENDON1", "wrong archive magic")
        fail_unless(tuple(fields[:9]) == GEOMETRY, f"wrong geometry/quant/tied: {fields[:9]}")
        fail_unless(abs(fields[9] - 1e-6) <= 1e-12, "wrong RMS epsilon")
        fail_unless(fields[10] == 1e6, "wrong RoPE theta")
        self.pos = HEADER.size
        self.extents = []

    def take(self, name, dtype, shape):
        count = int(np.prod(shape))
        length = count * np.dtype(dtype).itemsize
        end = self.pos + length
        fail_unless(end <= self.size, f"archive truncated at {name}")
        result = np.frombuffer(self.mm, dtype=dtype, count=count, offset=self.pos).reshape(shape)
        self.extents.append((name, self.pos, end))
        self.pos = end
        self.guard()
        return result

    def finish(self):
        fail_unless(self.pos == self.size, f"unparsed archive tail: {self.size - self.pos} bytes")
        return len(self.extents)


def source_rows(handle, name, lo, hi):
    # Safetensors indexes the mapped BF16 source. Only a bounded row chunk is materialized.
    return handle.get_slice(name)[lo:hi].to(torch.float32).contiguous()


def verify_vector(handle, name, archive, shape, guard):
    actual = archive.take(name, "<f4", shape)
    fail_unless(np.isfinite(actual).all(), f"nonfinite archive vector {name}")
    original = handle.get_tensor(name).float().numpy()
    fail_unless(np.array_equal(actual, original), f"vector widening differs: {name}")
    guard()
    return torch.from_numpy(np.array(actual, copy=True))


def verify_matrix(handle, source_name, archive, shape, guard, archive_name=None):
    name = archive_name or source_name
    out, inn = shape
    codes = archive.take(name + ".codes", "i1", shape)
    scales = archive.take(name + ".scales", "<f4", (out,))
    fail_unless(np.isfinite(scales).all() and np.all(scales >= 1e-12), f"invalid scales {name}")
    fail_unless(np.all((codes >= -127) & (codes <= 127)), f"invalid R8 codes {name}")
    for first in range(0, out, 64):
        last = min(out, first + 64)
        source = source_rows(handle, source_name, first, last)
        scale = (source.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)
        expected = torch.round(source / scale).clamp(-127, 127).to(torch.int8).numpy()
        fail_unless(np.array_equal(scales[first:last], scale[:, 0].numpy()),
                    f"R8 scale derivation differs: {name} rows {first}:{last}")
        fail_unless(np.array_equal(codes[first:last], expected),
                    f"R8 codes differ: {name} rows {first}:{last}")
        guard()
    decoded = codes.astype(np.float32) * scales[:, None]
    fail_unless(np.isfinite(decoded).all(), f"nonfinite decoded matrix: {name}")
    return torch.from_numpy(decoded)


def run(args):
    started = time.monotonic()
    process = psutil.Process()
    peak = 0

    def guard():
        nonlocal peak
        elapsed = time.monotonic() - started
        rss = process.memory_info().rss
        peak = max(peak, rss)
        fail_unless(elapsed <= TIME_CAP, f"time cap {TIME_CAP}s exceeded")
        fail_unless(rss <= RSS_CAP, f"RSS cap {RSS_CAP} exceeded: {rss}")

    fail_unless(args.source.is_file() and args.artifact.is_file(), "missing source/artifact")
    fail_unless(args.ids.is_file() and args.logits.is_file(), "missing IDs/logits")
    fail_unless(args.source.parent.name == REVISION, "source snapshot revision path mismatch")
    config_path = args.source.parent / "config.json"
    fail_unless(config_path.is_file(), "missing local pinned config")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fail_unless((config["hidden_size"], config["intermediate_size"],
                 config["num_hidden_layers"], config["num_attention_heads"],
                 config["num_key_value_heads"], config["vocab_size"],
                 config["tie_word_embeddings"]) == (896, 4864, 24, 14, 2, 151936, True),
                "local config geometry mismatch")
    before = {"source": sha256(args.source, guard), "artifact": sha256(args.artifact, guard),
              "ids": sha256(args.ids, guard), "logits": sha256(args.logits, guard),
              "config": sha256(config_path, guard)}
    fail_unless(before["source"] == SOURCE_SHA, "source SHA mismatch")
    fail_unless(before["artifact"] == args.artifact_sha.lower(), "artifact SHA mismatch")
    specs = source_header(args.source, guard)
    shapes = expected_shapes()
    fail_unless(set(specs) == set(shapes), "source tensor name set mismatch")
    for name, shape in shapes.items():
        fail_unless(tuple(specs[name]["shape"]) == shape, f"source tensor shape {name}")
    ids = np.fromfile(args.ids, dtype="<i4")
    fail_unless(len(ids) == 32 and args.ids.stat().st_size == 32 * 4, "expected 32 canonical IDs")
    fail_unless(np.all((ids >= 0) & (ids < 151936)), "token ID out of range")
    stride = 4 + 151936 * 4
    fail_unless(args.logits.stat().st_size == 16 + len(ids) * stride, "native logits extent mismatch")
    with args.logits.open("rb") as f:
        fail_unless(f.read(8) == b"QWIP0001", "native probe magic")
        fail_unless(struct.unpack("<2I", f.read(8)) == (len(ids), 151936), "native probe shape")
        native = np.empty((len(ids), 151936), dtype=np.float32)
        for i in range(len(ids)):
            fail_unless(struct.unpack("<I", f.read(4))[0] == i, "native probe position")
            native[i] = np.frombuffer(f.read(151936 * 4), dtype="<f4")
        fail_unless(f.read(1) == b"", "native probe tail")
    fail_unless(np.isfinite(native).all(), "nonfinite native logits")

    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    from transformers import Qwen2Config, Qwen2ForCausalLM

    hf_config = Qwen2Config.from_pretrained(str(args.source.parent), local_files_only=True)
    hf_config.tie_word_embeddings = False  # the archive has an explicit quantized head
    hf_config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen2ForCausalLM(hf_config)
    archive = Archive(args.artifact, guard)
    state = {}
    vectors = 0
    matrices = 0
    with safe_open(str(args.source), framework="pt", device="cpu") as source:
        state["model.embed_tokens.weight"] = verify_vector(
            source, "model.embed_tokens.weight", archive, (151936, 896), guard)
        vectors += 1
        for layer in range(24):
            pre = f"model.layers.{layer}."
            for organ in ("input_layernorm.weight",):
                name = pre + organ
                state[name] = verify_vector(source, name, archive, shapes[name], guard)
                vectors += 1
            for organ in ("q_proj", "k_proj", "v_proj"):
                name = pre + "self_attn." + organ + ".weight"
                state[name] = verify_matrix(source, name, archive, shapes[name], guard)
                matrices += 1
                bias = pre + "self_attn." + organ + ".bias"
                state[bias] = verify_vector(source, bias, archive, shapes[bias], guard)
                vectors += 1
            name = pre + "self_attn.o_proj.weight"
            state[name] = verify_matrix(source, name, archive, shapes[name], guard)
            matrices += 1
            name = pre + "post_attention_layernorm.weight"
            state[name] = verify_vector(source, name, archive, shapes[name], guard)
            vectors += 1
            for organ in ("gate_proj", "up_proj", "down_proj"):
                name = pre + "mlp." + organ + ".weight"
                state[name] = verify_matrix(source, name, archive, shapes[name], guard)
                matrices += 1
        state["model.norm.weight"] = verify_vector(
            source, "model.norm.weight", archive, (896,), guard)
        vectors += 1
        # The original source ties this matrix to embed_tokens. The archive deliberately unties it.
        state["lm_head.weight"] = verify_matrix(
            source, "model.embed_tokens.weight", archive, (151936, 896), guard,
            archive_name="lm_head.weight")
        matrices += 1
    extent_count = archive.finish()
    fail_unless((vectors, matrices, extent_count) == (122, 169, 460),
                f"unexpected component/extent counts: {(vectors, matrices, extent_count)}")
    fail_unless(set(model.state_dict()) == set(state), "decoded state tensor set mismatch")
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.eval()
    fail_unless(next(model.parameters()).device.type == "cpu", "model is not CPU")
    guard()
    with torch.inference_mode():
        reference = model(torch.from_numpy(ids.astype(np.int64))[None],
                          use_cache=False).logits[0].float().numpy()
    fail_unless(reference.shape == native.shape and np.isfinite(reference).all(),
                "reference logits invalid")
    rows = []
    for position in range(len(ids)):
        a = native[position].astype(np.float64)
        b = reference[position].astype(np.float64)
        delta = a - b
        rows.append({"position": position, "native_argmax": int(np.argmax(a)),
                     "reference_argmax": int(np.argmax(b)),
                     "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(b)),
                     "maximum_absolute_difference": float(np.max(np.abs(delta)))})
    worst = max(row["relative_l2"] for row in rows)
    matches = sum(row["native_argmax"] == row["reference_argmax"] for row in rows)
    after = {"source": sha256(args.source, guard), "artifact": sha256(args.artifact, guard),
             "ids": sha256(args.ids, guard), "logits": sha256(args.logits, guard),
             "config": sha256(config_path, guard)}
    fail_unless(after == before, "input seal changed during audit")
    result = {"schema": "QWEN_RECIPIENT_INT8_AUDIT_V1", "status": "PASS",
              "scope": "R8 source derivation and decoded FP32 full-prefix C-logit parity only",
              "source_revision": REVISION, "input_sha256_before_after": before,
              "auditor_sha256": sha256(Path(__file__), guard),
              "source_tensors": len(specs), "vectors_checked": vectors,
              "matrices_checked": matrices, "artifact_extents_checked": extent_count,
              "artifact_bytes_consumed": archive.pos, "artifact_bytes": archive.size,
              "ids_count": len(ids), "logits_shape": list(native.shape),
              "reference_logits_sha256": hashlib.sha256(reference.astype("<f4").tobytes()).hexdigest(),
              "rows": rows, "top1_matches": matches, "worst_relative_l2": worst,
              "seconds": time.monotonic() - started, "peak_sampled_rss_bytes": peak,
              "torch_version": torch.__version__, "numpy_version": np.__version__}
    fail_unless(matches == len(ids) and worst <= 1e-3,
                f"native/reference parity failed: top1 {matches}/{len(ids)}, worst L2 {worst}")
    guard()
    result["seconds"] = time.monotonic() - started
    result["peak_sampled_rss_bytes"] = peak
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "artifact", "ids", "logits", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--artifact-sha", required=True)
    args = parser.parse_args()
    failure = args.out.with_name(args.out.name + ".failure.json")
    fail_unless(not args.out.exists() and not failure.exists(), "output namespace already exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        fail_unless(len(args.artifact_sha) == 64 and
                    all(c in "0123456789abcdefABCDEF" for c in args.artifact_sha),
                    "artifact SHA must be 64 hex digits")
        result = run(args)
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        fail_unless(len(payload.encode("utf-8")) <= OUTPUT_CAP, "output cap exceeded")
        with args.out.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except BaseException as exc:
        payload = {"schema": "QWEN_RECIPIENT_INT8_AUDIT_V1", "status": "FAIL",
                   "error_type": type(exc).__name__, "error": str(exc)}
        with failure.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        raise


if __name__ == "__main__":
    main()
