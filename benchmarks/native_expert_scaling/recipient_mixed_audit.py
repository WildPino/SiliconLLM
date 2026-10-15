#!/usr/bin/env python3
"""Offline source-byte and decoded-wire audit of the mixed Qwen recipient.

The reference graph explicitly applies runtime activation quantization. It does
not execute the native engine, measure speed, or evaluate chatbot quality.
"""

import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import struct
import sys
import time

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
D, H, L, V, NK, NH, HD = 896, 4864, 24, 151936, 2, 14, 64
KD = NK * HD
POSITIONS = (0, 7, 15, 31)
TIME_CAP = 900.0
RSS_CAP = 8 * (1 << 30)
OUTPUT_CAP = 2 * (1 << 20)


def require(ok, message):
    if not ok:
        raise ValueError(message)


class AuditFailure(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def digest_file(path, guard):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
            guard()
    return h.hexdigest()


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def source_header(path, guard):
    with path.open("rb") as f:
        rawlen = f.read(8)
        require(len(rawlen) == 8, "source header length")
        length = struct.unpack("<Q", rawlen)[0]
        require(0 < length < (1 << 22), "source header bound")
        raw = f.read(length)
        require(len(raw) == length, "source header truncated")
    specs = {k: v for k, v in json.loads(raw).items() if k != "__metadata__"}
    require(len(specs) == 290, "source tensor count")
    extents = []
    for name, value in specs.items():
        require(value["dtype"] == "BF16", f"source dtype: {name}")
        shape = value["shape"]
        lo, hi = value["data_offsets"]
        require(hi - lo == 2 * int(np.prod(shape)) and lo >= 0, f"source extent: {name}")
        extents.append((lo, hi))
    cursor = 0
    for lo, hi in sorted(extents):
        require(lo == cursor, "source tensor gap or overlap")
        cursor = hi
    require(8 + length + cursor == path.stat().st_size, "source trailing bytes")
    guard()
    return specs


def roles():
    expected = {"model.embed_tokens.weight": (2, V, D, "model.embed_tokens.weight"),
                "model.norm.weight": (1, 1, D, "model.norm.weight"),
                "lm_head.weight": (3, V, D, "model.embed_tokens.weight")}
    for layer in range(L):
        pre = f"model.layers.{layer}."
        for tail in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            expected[pre + tail] = (1, 1, D, pre + tail)
        for organ, rows in (("q_proj", D), ("k_proj", KD), ("v_proj", KD),
                            ("o_proj", D)):
            name = pre + "self_attn." + organ + ".weight"
            expected[name] = (3, rows, D, name)
            if organ != "o_proj":
                bias = pre + "self_attn." + organ + ".bias"
                expected[bias] = (1, 1, rows, bias)
        for organ, rows, cols in (("gate_proj", H, D), ("up_proj", H, D),
                                  ("down_proj", D, H)):
            name = pre + "mlp." + organ + ".weight"
            expected[name] = (4, rows, cols, name)
    require(len(expected) == 291, "internal role census")
    return expected


def source_shapes(expected):
    shapes = {}
    for _, rows, cols, source_name in expected.values():
        shape = (rows, cols) if source_name.endswith(".weight") and (
            source_name == "model.embed_tokens.weight" or
            ".self_attn." in source_name or ".mlp." in source_name) else (cols,)
        previous = shapes.setdefault(source_name, shape)
        require(previous == shape, f"conflicting source shape: {source_name}")
    require(len(shapes) == 290, "internal source census")
    return shapes


class Catalog:
    def __init__(self, manifest_path, header_path, archive_path, archive_sha,
                 source_path, config_path, expected, guard):
        self.doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        d = self.doc
        require(d["schema"] == "RECIPIENT_MIXED_CATALOG_V1", "catalog schema")
        require(d["source_model"] == "Qwen/Qwen2.5-0.5B-Instruct" and
                d["source_revision"] == REVISION and d["source_sha256"] == SOURCE_SHA,
                "catalog source binding")
        require(Path(d["sourcepath"]).resolve() == source_path.resolve() and
                Path(d["config_path"]).resolve() == config_path.resolve() and
                Path(d["archive_path"]).resolve() == archive_path.resolve() and
                Path(d["catalog_path"]).resolve() == header_path.resolve(),
                "manifest input paths")
        require(d["config_sha256"] == digest_file(config_path, guard) and
                d["catalog_sha256"] == digest_file(header_path, guard) and
                d["catalog_bytes"] == header_path.stat().st_size,
                "manifest config/header seals")
        require(d["archive_sha256"] == archive_sha, "catalog archive SHA")
        require(d["archive_bytes"] == archive_path.stat().st_size, "catalog archive length")
        require(len(d["fields"]) == len(expected), "catalog field count")
        require(d["source_tensor_count"] == 290 and
                d["codec_counts"] == {"1": 121, "2": 1, "3": 97, "4": 72},
                "manifest tensor/codec counts")
        hashes = d["field_sha256"]
        require(set(hashes) == set(expected), "manifest field hash key set")
        header = header_path.read_text(encoding="ascii")
        require(re.search(r"#define RC_FIELD_COUNT 291\b", header) and
                re.search(rf"#define RC_ARCHIVE_BYTES UINT64_C\({d['archive_bytes']}\)", header) and
                f'#define RC_ARCHIVE_SHA "{archive_sha}"' in header,
                "native catalog constants")
        pattern = (r'\{"([^"]+)", (\d+), UINT64_C\((\d+)\), UINT64_C\((\d+)\), '
                   r'UINT64_C\((\d+)\), UINT64_C\((\d+)\)\},')
        header_fields = [(m[0], *map(int, m[1:])) for m in re.findall(pattern, header)]
        manifest_fields = [(f["name"], f["dtype"], f["rows"], f["cols"],
                            f["offset"], f["bytes"]) for f in d["fields"]]
        require(header_fields == manifest_fields, "native header/manifest field mismatch")
        self.handle = archive_path.open("rb")
        self.mm = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)
        self.fields = {}
        cursor = 0
        for f in d["fields"]:
            name = f["name"]
            require(name in expected and name not in self.fields, f"field name: {name}")
            dtype, rows, cols, _ = expected[name]
            require((f["dtype"], f["rows"], f["cols"]) ==
                    (dtype, rows, cols), f"field geometry: {name}")
            length = rows * cols * (4 if dtype == 1 else 2) if dtype in (1, 2) else (
                rows * (cols + 4) if dtype == 3 else rows * (cols // 32) * 18)
            require(f["bytes"] == length and (dtype != 4 or cols % 32 == 0),
                    f"field length: {name}")
            lo, hi = f["offset"], f["offset"] + length
            require(lo % 64 == 0 and lo >= cursor and hi <= len(self.mm) and
                    (bool(self.fields) or lo == 0), f"field extent: {name}")
            require(not any(self.mm[cursor:lo]), f"nonzero field padding: {name}")
            require(digest_bytes(memoryview(self.mm)[lo:hi]) == hashes[name],
                    f"field SHA: {name}")
            self.fields[name] = f
            cursor = hi
            guard()
        require(set(self.fields) == set(expected), "catalog role set")
        require(not any(self.mm[cursor:]), "nonzero archive trailing padding")

    def raw(self, name):
        f = self.fields[name]
        lo = f["offset"]
        return memoryview(self.mm)[lo:lo + f["bytes"]]


def source_rows(handle, name, lo, hi):
    return handle.get_slice(name)[lo:hi].to(torch.float32).contiguous().numpy()


def verify_fields(catalog, source, expected, guard):
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for name, (dtype, rows, cols, source_name) in expected.items():
        raw = catalog.raw(name)
        if dtype == 1:
            actual = np.frombuffer(raw, dtype="<f4")
            reference = source.get_tensor(source_name).float().numpy().reshape(-1)
            require(np.isfinite(actual).all() and np.array_equal(actual, reference),
                    f"F32 widening: {name}")
        elif dtype == 2:
            # Raw BF16 embedding bytes are retained, including all vocabulary rows.
            for lo in range(0, rows, 256):
                hi = min(rows, lo + 256)
                ref = source.get_slice(source_name)[lo:hi].contiguous().view(torch.uint16).numpy()
                actual = np.frombuffer(raw[lo * cols * 2:hi * cols * 2], dtype="<u2")
                require(np.array_equal(actual, ref.reshape(-1)), f"BF16 bytes: {name} {lo}:{hi}")
                require(np.isfinite((actual.astype(np.uint32) << 16).view("<f4")).all(),
                        f"nonfinite BF16: {name}")
                guard()
        elif dtype == 3:
            pitch = cols + 4
            for lo in range(0, rows, 64):
                hi = min(rows, lo + 64)
                src = source_rows(source, source_name, lo, hi)
                maximum = np.max(np.abs(src), axis=1).astype(np.float32)
                scale = np.where(maximum == 0, np.float32(0),
                                 (maximum / np.float32(127)).astype(np.float32))
                quant = np.zeros(src.shape, dtype=np.int8)
                nz = scale > 0
                quant[nz] = np.clip(np.rint(src[nz] / scale[nz, None]), -127, 127).astype(np.int8)
                wire = np.frombuffer(raw[lo * pitch:hi * pitch], dtype="u1").reshape(hi - lo, pitch)
                require(np.isfinite(scale).all() and np.array_equal(
                    wire[:, :4].copy().view("<f4").reshape(-1), scale),
                    f"W8 scale derivation: {name} {lo}:{hi}")
                require(np.array_equal(wire[:, 4:].view("i1"), quant),
                        f"W8 code derivation: {name} {lo}:{hi}")
                guard()
        else:
            groups = cols // 32
            pitch = groups * 18
            for lo in range(0, rows, 32):
                hi = min(rows, lo + 32)
                src = source_rows(source, source_name, lo, hi).reshape(hi - lo, groups, 32)
                maximum = np.max(np.abs(src), axis=2).astype(np.float32)
                half_scale = (maximum / np.float32(7)).astype("<f2")
                scale = half_scale.astype(np.float32)
                require(np.all((maximum == 0) | (scale > 0)), f"W4 underflow: {name}")
                quant = np.zeros(src.shape, dtype=np.int8)
                nz = scale > 0
                quant[nz] = np.clip(np.rint(src[nz] / scale[nz, None]), -7, 7).astype(np.int8)
                nibble = (quant + 8).astype(np.uint8)
                packed = nibble[:, :, :16] | (nibble[:, :, 16:] << 4)
                wire = np.frombuffer(raw[lo * pitch:hi * pitch], dtype="u1").reshape(hi - lo, groups, 18)
                require(np.array_equal(wire[:, :, :2].copy().view("<f2").reshape(hi-lo, groups),
                                       half_scale), f"W4 scale derivation: {name} {lo}:{hi}")
                require(np.array_equal(wire[:, :, 2:], packed),
                        f"W4 code derivation: {name} {lo}:{hi}")
                guard()
        counts[dtype] += 1
    require(counts == {1: 121, 2: 1, 3: 97, 4: 72}, f"role counts: {counts}")
    return counts


def check_request(path, ids):
    require(path.stat().st_size == 16 + 4 * len(POSITIONS) + 4 * len(ids),
            "QWCP request length")
    with path.open("rb") as f:
        require(f.read(8) == b"QWCP0001", "QWCP magic")
        require(struct.unpack("<2I", f.read(8)) == (len(ids), len(POSITIONS)),
                "QWCP counts")
        require(struct.unpack("<4I", f.read(4 * len(POSITIONS))) == POSITIONS,
                "QWCP positions")
        request_ids = np.frombuffer(f.read(4 * len(ids)), dtype="<u4")
        require(np.array_equal(request_ids, ids.astype("<u4")), "QWCP IDs")
        require(f.read(1) == b"", "QWCP trailing bytes")


def read_probe(path, guard):
    size = 24 + len(POSITIONS) * (4 + 4 * D + 4 * V)
    require(path.stat().st_size == size, "QWCT probe length")
    norms, logits = {}, {}
    with path.open("rb") as f:
        require(f.read(8) == b"QWCT0001", "QWCT magic")
        require(struct.unpack("<4I", f.read(16)) == (V, len(POSITIONS), L, D), "QWCT header")
        for position in POSITIONS:
            require(struct.unpack("<I", f.read(4))[0] == position, "QWCT position")
            norm = np.frombuffer(f.read(4 * D), dtype="<f4").copy()
            logit = np.frombuffer(f.read(4 * V), dtype="<f4").copy()
            require(len(norm) == D and len(logit) == V and
                    np.isfinite(norm).all() and np.isfinite(logit).all(), "QWCT finite/extent")
            norms[position], logits[position] = norm, logit
            guard()
        require(f.read(1) == b"", "QWCT trailing bytes")
    return norms, logits


class Reference:
    def __init__(self, catalog, source, guard):
        self.c = catalog
        self.source = source
        self.guard = guard
        self.weights = {}
        self.buffers = {}
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)

    def vector(self, name):
        if name not in self.buffers:
            self.buffers[name] = np.frombuffer(self.c.raw(name), dtype="<f4").copy()
        return self.buffers[name]

    def embedding(self, token):
        raw = self.c.raw("model.embed_tokens.weight")
        bits = np.frombuffer(raw[token * D * 2:(token + 1) * D * 2], dtype="<u2")
        return (bits.astype(np.uint32) << 16).view("<f4").copy()

    def weight(self, name):
        if name in self.weights:
            return self.weights[name]
        f = self.c.fields[name]
        rows, cols, dtype = f["rows"], f["cols"], f["dtype"]
        raw = self.c.raw(name)
        if dtype == 3:
            wire = np.frombuffer(raw, dtype="u1").reshape(rows, cols + 4)
            scales = wire[:, :4].copy().view("<f4").reshape(rows)
            codes = torch.from_numpy(wire[:, 4:].copy().view("i1").astype(np.float32))
            value = (dtype, codes, scales)
        else:
            require(dtype == 4, f"reference matrix codec: {name}")
            groups = cols // 32
            wire = np.frombuffer(raw, dtype="u1").reshape(rows, groups, 18)
            scales = wire[:, :, :2].copy().view("<f2").reshape(rows, groups).astype(np.float32)
            packed = wire[:, :, 2:]
            codes = np.empty((rows, groups, 32), dtype=np.float32)
            codes[:, :, :16] = (packed & 15).astype(np.int16) - 8
            codes[:, :, 16:] = (packed >> 4).astype(np.int16) - 8
            # bmm sees independent exact integer sums for each group.
            value = (dtype, torch.from_numpy(codes.transpose(1, 0, 2).copy()), scales)
        self.weights[name] = value
        self.guard()
        return value

    @staticmethod
    def quant(x, group):
        x = np.asarray(x, dtype=np.float32)
        require(x.size % group == 0 and np.isfinite(x).all(), "activation quant input")
        blocks = x.reshape(-1, group)
        maximum = np.max(np.abs(blocks), axis=1)
        scale = np.where(maximum > 0, maximum / np.float32(127), np.float32(1)).astype(np.float32)
        q = np.clip(np.rint(blocks / scale[:, None]), -127, 127).astype(np.int8)
        return q, scale

    def project(self, name, q, scale):
        dtype, codes, ws = self.weight(name)
        if dtype == 3:
            require(scale.size == 1, "W8 activation scale count")
            dot = torch.mv(codes, torch.from_numpy(q.reshape(-1).astype(np.float32))).numpy()
            # Integer sums are exactly representable in F32 for this geometry.
            result = (dot.astype(np.float64) * ws.astype(np.float64) *
                      np.float64(scale[0])).astype(np.float32)
        else:
            qg = torch.from_numpy(q.astype(np.float32)).unsqueeze(2)
            dot = torch.bmm(codes, qg).squeeze(2).transpose(0, 1).numpy()
            result = np.sum(dot.astype(np.float64) * ws.astype(np.float64) *
                            scale.astype(np.float64)[None, :], axis=1).astype(np.float32)
        require(np.isfinite(result).all(), f"reference projection finite: {name}")
        return result

    @staticmethod
    def norm(x, weight):
        inverse = 1.0 / np.sqrt(np.sum(x.astype(np.float64) ** 2) / D + 1e-6)
        return (x.astype(np.float64) * inverse * weight.astype(np.float64)).astype(np.float32)

    @staticmethod
    def rope(x, heads, cosine, sine):
        z = x.reshape(heads, HD).copy()
        a, b = z[:, :32].copy(), z[:, 32:].copy()
        z[:, :32] = a * cosine - b * sine
        z[:, 32:] = b * cosine + a * sine
        return z.reshape(-1)

    def attention(self, q, keys, values, pos):
        attn = np.empty(D, dtype=np.float32)
        for h in range(NH):
            kvh = h // 7
            score = np.empty(pos + 1, dtype=np.float32)
            for t in range(pos + 1):
                dot = np.sum(q[h * HD:(h + 1) * HD].astype(np.float64) *
                             keys[t, kvh * HD:(kvh + 1) * HD].astype(np.float64))
                score[t] = np.float32(dot * 0.125)
            score = np.exp((score - np.max(score)).astype(np.float32)).astype(np.float32)
            score = (score.astype(np.float64) / np.sum(score.astype(np.float64))).astype(np.float32)
            for j in range(HD):
                attn[h * HD + j] = np.float32(np.sum(score.astype(np.float64) *
                    values[:pos+1, kvh * HD + j].astype(np.float64)))
        return attn

    def head(self, norm):
        q, scale = self.quant(norm, D)
        logits = self.project("lm_head.weight", q, scale)
        top = np.lexsort((np.arange(V, dtype=np.int32), -logits))[:8]
        for token in top:
            logits[token] = np.float32(np.sum(self.embedding(int(token)).astype(np.float64) *
                                                 norm.astype(np.float64)))
        return logits, top.tolist()

    def run(self, ids):
        keys = np.zeros((L, len(ids), KD), dtype=np.float32)
        values = np.zeros_like(keys)
        norms, logits, top8 = {}, {}, {}
        with torch.inference_mode():
            for pos, token in enumerate(ids):
                x = self.embedding(int(token))
                j = np.arange(32, dtype=np.float64)
                phase = pos * np.power(1e6, -(2 * j) / HD)
                cosine = np.cos(phase).astype(np.float32)
                sine = np.sin(phase).astype(np.float32)
                for li in range(L):
                    pre = f"model.layers.{li}."
                    norm = self.norm(x, self.vector(pre + "input_layernorm.weight"))
                    qa, sa = self.quant(norm, D)
                    att = pre + "self_attn."
                    q = self.project(att + "q_proj.weight", qa, sa)
                    k = self.project(att + "k_proj.weight", qa, sa)
                    v = self.project(att + "v_proj.weight", qa, sa)
                    q = (q + self.vector(att + "q_proj.bias")).astype(np.float32)
                    k = (k + self.vector(att + "k_proj.bias")).astype(np.float32)
                    v = (v + self.vector(att + "v_proj.bias")).astype(np.float32)
                    q = self.rope(q, NH, cosine, sine)
                    k = self.rope(k, NK, cosine, sine)
                    keys[li, pos] = k
                    values[li, pos] = v
                    a = self.attention(q, keys[li], values[li], pos)
                    qa, sa = self.quant(a, D)
                    x = (x + self.project(att + "o_proj.weight", qa, sa)).astype(np.float32)
                    norm = self.norm(x, self.vector(pre + "post_attention_layernorm.weight"))
                    qa, sa = self.quant(norm, 32)
                    gate = self.project(pre + "mlp.gate_proj.weight", qa, sa)
                    up = self.project(pre + "mlp.up_proj.weight", qa, sa)
                    positive = gate >= 0
                    exp = np.exp(-np.abs(gate)).astype(np.float32)
                    silu = np.where(positive, gate / (np.float32(1) + exp),
                                    gate * exp / (np.float32(1) + exp)).astype(np.float32)
                    active = (silu * up).astype(np.float32)
                    qa, sa = self.quant(active, 32)
                    x = (x + self.project(pre + "mlp.down_proj.weight", qa, sa)).astype(np.float32)
                    require(np.isfinite(x).all(), f"reference residual finite: {li}/{pos}")
                if pos in POSITIONS:
                    norm = self.norm(x, self.vector("model.norm.weight"))
                    logit, chosen = self.head(norm)
                    norms[pos], logits[pos], top8[pos] = norm, logit, chosen
                self.guard()
        return norms, logits, top8


def compare(reference, native):
    rows = []
    for pos in POSITIONS:
        a = native[pos].astype(np.float64)
        b = reference[pos].astype(np.float64)
        relative = float(np.linalg.norm(a - b) / np.linalg.norm(b))
        rows.append({"position": pos, "native_argmax": int(np.argmax(a)),
                     "reference_argmax": int(np.argmax(b)), "relative_l2": relative,
                     "maximum_absolute_difference": float(np.max(np.abs(a-b)))})
    return rows


def run(args):
    start = time.monotonic()
    process = psutil.Process()
    peak = 0

    def guard():
        nonlocal peak
        seconds = time.monotonic() - start
        peak = max(peak, process.memory_info().rss)
        require(seconds <= TIME_CAP, f"time cap exceeded: {seconds:.1f}s")
        require(peak <= RSS_CAP, f"RSS cap exceeded: {peak}")

    for path in (args.source, args.archive, args.catalog, args.manifest,
                 args.ids, args.request, args.probe):
        require(path.is_file(), f"missing input: {path}")
    require(args.source.parent.name == REVISION, "source revision path")
    config_path = args.source.parent / "config.json"
    require(config_path.is_file(), "pinned config absent")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require((config["hidden_size"], config["intermediate_size"],
             config["num_hidden_layers"], config["num_attention_heads"],
             config["num_key_value_heads"], config["vocab_size"],
             config["tie_word_embeddings"]) == (D, H, L, NH, NK, V, True),
            "pinned config geometry")
    require(config["model_type"] == "qwen2" and config["hidden_act"] == "silu" and
            config["use_sliding_window"] is False and
            config["rms_norm_eps"] == 1e-6 and config["rope_theta"] == 1e6,
            "pinned config decoder semantics")
    paths = {"source": args.source, "archive": args.archive, "catalog": args.catalog,
             "manifest": args.manifest, "ids": args.ids, "request": args.request,
             "probe": args.probe,
             "config": config_path}
    before = {k: digest_file(p, guard) for k, p in paths.items()}
    require(before["source"] == SOURCE_SHA, "source SHA mismatch")
    require(before["archive"] == args.archive_sha.lower(), "archive SHA mismatch")
    expected = roles()
    specs = source_header(args.source, guard)
    shapes = source_shapes(expected)
    require(set(specs) == set(shapes), "source tensor names")
    for name, shape in shapes.items():
        require(tuple(specs[name]["shape"]) == shape, f"source shape: {name}")
    ids = np.fromfile(args.ids, dtype="<i4")
    require(args.ids.stat().st_size == 32 * 4 and len(ids) == 32 and
            np.all((ids >= 0) & (ids < V)), "canonical 32-token IDs")
    check_request(args.request, ids)
    native_norm, native_logits = read_probe(args.probe, guard)
    catalog = Catalog(args.manifest, args.catalog, args.archive, before["archive"],
                      args.source, config_path, expected, guard)
    with safe_open(str(args.source), framework="pt", device="cpu") as source:
        counts = verify_fields(catalog, source, expected, guard)
        reference = Reference(catalog, source, guard)
        norm, logits, top8 = reference.run(ids)
    norm_rows = compare(norm, native_norm)
    logit_rows = compare(logits, native_logits)
    after = {k: digest_file(p, guard) for k, p in paths.items()}
    require(after == before, "input seals changed during audit")
    result = {"schema": "QWEN_RECIPIENT_MIXED_AUDIT_V1", "status": "PASS",
            "scope": "Exact source-derived 291-role mixed wire plus saved QWCT decoded numerical parity; no quality/rate claim",
            "source_revision": REVISION, "input_sha256_before_after": before,
            "auditor_sha256": digest_file(Path(__file__), guard),
            "field_counts_by_dtype": counts, "field_count": len(expected),
            "source_tensor_count": len(specs), "archive_bytes": len(catalog.mm),
            "ids_count": len(ids), "selected_positions": list(POSITIONS),
            "reference_top8_before_refinement": top8,
            "reference_logits_sha256": {str(p): digest_bytes(logits[p].astype("<f4").tobytes())
                                         for p in POSITIONS},
            "final_norm_rows": norm_rows, "full_logits_rows": logit_rows,
            "seconds": time.monotonic() - start, "peak_sampled_rss_bytes": peak,
            "torch_version": torch.__version__, "numpy_version": np.__version__}
    result["gates"] = {
        "all_selected_argmax_equal": all(r["native_argmax"] == r["reference_argmax"]
                                         for r in logit_rows),
        "all_selected_full_logits_relative_l2_le_1e-3": all(
            r["relative_l2"] <= 1e-3 for r in logit_rows),
        "all_selected_final_norm_relative_l2_le_1e-3": all(
            r["relative_l2"] <= 1e-3 for r in norm_rows)}
    if not all(result["gates"].values()):
        result["status"] = "FAIL"
        raise AuditFailure("native/reference numerical parity failed", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "archive", "catalog", "manifest", "ids", "request",
                "probe", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--archive-sha", required=True)
    args = parser.parse_args()
    failure = args.out.with_name(args.out.name + ".failure.json")
    require(not args.out.exists() and not failure.exists(), "output namespace already exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        require(len(args.archive_sha) == 64 and
                all(c in "0123456789abcdefABCDEF" for c in args.archive_sha),
                "archive SHA must be 64 hex digits")
        result = run(args)
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        require(len(payload.encode("utf-8")) <= OUTPUT_CAP, "audit output cap")
        with args.out.open("x", encoding="utf-8") as f:
            f.write(payload)
    except BaseException as exc:
        report = {"schema": "QWEN_RECIPIENT_MIXED_AUDIT_V1", "status": "FAIL",
                  "error_type": type(exc).__name__, "error": str(exc)}
        if isinstance(exc, AuditFailure):
            report["diagnostics"] = exc.diagnostics
        with failure.open("x", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        raise


if __name__ == "__main__":
    main()
