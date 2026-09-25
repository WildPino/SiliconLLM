#!/usr/bin/env python3
"""Frozen NES-03 int8 router shortlist feasibility on real E128 inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "phase60"))
sys.path.insert(0, str(ROOT / "benchmarks" / "phase55"))
from e4_reference import build  # noqa: E402
from phase55_ssm import IDS  # noqa: E402
from phase59_moe import moe_forward  # noqa: E402


EXPECTED_SHA256 = {
    "ckpt": "58c84a8730b03c37e98481b8508f5cc84e15757de5513c3e6354f09b975cd462",
    "e128": "259e1aa73593e8fa8f72a997aaa63d30961342676fa0be587a22d920c804d2cf",
    "e1280": "25f99a300a26ed90948d04990ed75df9f94ab9a288ef48e6ea0c46fc0a3dcf2e",
    "ids": "33b8cba2a26653599f7f87a4d8e05b38be051ba850d4cbc5d09b561aae133889",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def router_arrays(path: Path) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    source = np.memmap(path, mode="r", dtype=np.uint8)
    h = struct.unpack("<16I", source[:64])
    magic, vocab, width, state, heads, layers, dn, dt_rank, conv, win, swa, experts, hidden, topk, packed, reserved = h
    if (magic, vocab, width, state, heads, layers, dn, dt_rank, conv, win, swa, hidden, topk, packed, reserved) != (
            0x45344D31, 1024, 256, 96, 8, 6, 512, 16, 4, 128, 5, 128, 8, 1, 0):
        raise ValueError(f"unexpected E4M1 topology in {path}")
    offset = 64 + vocab * width * 4
    weights, biases = [], []
    for layer in range(layers):
        offset += width * 4  # mixer input norm
        if layer == swa:
            mixer_elements = 4 * width * width
        else:
            mixer_elements = (2 * dn * width + dn * conv + dn +
                              (dt_rank + 2 * state) * dn + dn * dt_rank + dn +
                              dn * state + dn + width * dn)
        offset += mixer_elements * 4 + width * 4  # mixer and MoE norm
        n = experts * width
        weights.append(np.frombuffer(source[offset:offset + n * 4], dtype="<f4").reshape(experts, width))
        offset += n * 4
        biases.append(np.frombuffer(source[offset:offset + experts * 4], dtype="<f4"))
        offset += experts * 4
        offset += 3 * experts * hidden * width * 4  # fp32 reference expert matrices
    return weights, biases, experts


def capture(ckpt: Path, windows: int, offset: int, device: str) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, cfg = build(str(ckpt))
    model.to(device).eval()
    modules = [block.mlp for block in model.blocks if getattr(block, "use_mlp", False)]
    if cfg["E"] != 128 or len(modules) != 6:
        raise ValueError("source checkpoint is not the frozen E128/L6 model")
    for module in modules:
        module._cap = True
    ids = np.fromfile(IDS, dtype="<u2")
    val = ids[int(len(ids) * 0.9):]
    seq = 512
    if offset + windows * seq + 1 > len(val):
        raise ValueError("frozen windows exceed validation IDs")
    inputs: list[list[np.ndarray]] = [[] for _ in modules]
    exact_ids: list[list[np.ndarray]] = [[] for _ in modules]
    started = time.perf_counter()
    with torch.inference_mode():
        for w in range(windows):
            chunk = torch.from_numpy(val[offset + w * seq:offset + (w + 1) * seq].astype(np.int64))
            moe_forward(model, chunk[None].to(device), False)
            for layer, module in enumerate(modules):
                inputs[layer].append(module._xin.reshape(seq, cfg["D"]).copy())
                exact_ids[layer].append(module._topi.reshape(seq, 8).copy())
            print(f"captured window {w + 1}/{windows}", flush=True)
    elapsed = time.perf_counter() - started
    return ([np.concatenate(chunks) for chunks in inputs],
            [np.concatenate(chunks) for chunks in exact_ids], elapsed)


def analyze_layer(x: np.ndarray, w: np.ndarray, b: np.ndarray, captured_top8: np.ndarray | None) -> dict:
    # Each integer dot fits exactly in float32: 256*127*127 < 2**24.
    row_max = np.max(np.abs(w), axis=1)
    row_scale = np.where(row_max > 0, row_max / 127.0, 1.0).astype(np.float32)
    qw = np.rint(w / row_scale[:, None]).clip(-127, 127).astype(np.int8)
    input_max = np.max(np.abs(x), axis=1)
    input_scale = np.where(input_max > 0, input_max / 63.0, 1.0).astype(np.float32)
    qx = np.rint(x / input_scale[:, None]).clip(-63, 63).astype(np.int8)
    xu = (qx.astype(np.int16) + 64).astype(np.uint8)
    sumw = qw.astype(np.int32).sum(axis=1)
    approx_dot = xu.astype(np.float32) @ qw.astype(np.float32).T
    approx = (approx_dot - 64.0 * sumw[None, :]) * input_scale[:, None] * row_scale[None, :] + b[None, :]
    exact = x @ w.T + b[None, :]
    exact_order = np.argsort(-exact, axis=1, kind="stable")
    sketch_order = np.argsort(-approx, axis=1, kind="stable")
    true_top8 = exact_order[:, :8]
    rank = np.empty_like(sketch_order)
    rank[np.arange(len(x))[:, None], sketch_order] = np.arange(w.shape[0])[None, :]
    exact_ranks = np.take_along_axis(rank, true_top8, axis=1)
    row = {
        "positions": int(len(x)),
        "experts": int(len(w)),
        "exact_eighth_ninth_margin_mean": float(np.mean(np.take_along_axis(exact, exact_order[:, 7:8], axis=1) -
                                                      np.take_along_axis(exact, exact_order[:, 8:9], axis=1))),
        "max_sketch_rank_of_exact_top8_p50_p95_p99_max": [float(v) for v in np.percentile(exact_ranks.max(axis=1) + 1,
                                                                                           [50, 95, 99, 100])],
        "shortlists": {},
    }
    if captured_top8 is not None:
        hits = (np.sort(captured_top8, axis=1) == np.sort(true_top8, axis=1)).sum()
        row["pytorch_exact_top8_id_recall"] = float(hits / (len(x) * 8))
        row["pytorch_exact_all8_fraction"] = float(np.mean(np.all(np.sort(captured_top8, axis=1) ==
                                                                    np.sort(true_top8, axis=1), axis=1)))
    for count in (32, 64):
        candidate = sketch_order[:, :count]
        candidate_exact = np.take_along_axis(exact, candidate, axis=1)
        selected = np.take_along_axis(candidate, np.argsort(-candidate_exact, axis=1, kind="stable")[:, :8], axis=1)
        contained = (exact_ranks < count)
        exact_ids_recovered = int(contained.sum())
        row["shortlists"][str(count)] = {
            "top8_id_recall": float(exact_ids_recovered / (len(x) * 8)),
            "all8_contained_fraction": float(np.mean(np.all(contained, axis=1))),
            "changed_final_route_positions": int(np.sum(np.any(np.sort(selected, axis=1) !=
                                                                 np.sort(true_top8, axis=1), axis=1))),
        }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--e128-export", type=Path, required=True)
    parser.add_argument("--e1280-export", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument("--offset", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.windows < 1 or args.offset < 0:
        parser.error("windows must be positive and offset nonnegative")
    if args.out_dir.exists():
        parser.error("output directory already exists")
    paths = {"ckpt": args.ckpt, "e128": args.e128_export, "e1280": args.e1280_export,
             "ids": Path(IDS)}
    for key, path in paths.items():
        actual = sha256(path)
        if actual != EXPECTED_SHA256[key]:
            parser.error(f"{key} SHA-256 mismatch: {actual}")
    args.out_dir.mkdir(parents=True)
    x, captured_top8, capture_seconds = capture(args.ckpt, args.windows, args.offset, args.device)
    input_path = args.out_dir / "router_inputs_e128.npz"
    np.savez(input_path, **{f"x_{i}": value for i, value in enumerate(x)},
             **{f"top8_{i}": value for i, value in enumerate(captured_top8)})
    arms = {}
    for name in ("e128", "e1280"):
        weights, biases, experts = router_arrays(paths[name])
        arm = []
        for layer in range(6):
            result = analyze_layer(x[layer], weights[layer], biases[layer],
                                   captured_top8[layer] if name == "e128" else None)
            arm.append(result)
            print(name, "layer", layer, "shortlists", result["shortlists"], flush=True)
        arms[name] = {"experts": experts, "layers": arm}
    apparatus_pass = all(layer["pytorch_exact_top8_id_recall"] >= .999 for layer in arms["e128"]["layers"])
    ladder = {}
    for count in (32, 64):
        ladder[str(count)] = all(
            layer["shortlists"][str(count)]["top8_id_recall"] >= .999 and
            layer["shortlists"][str(count)]["all8_contained_fraction"] >= .99
            for arm in arms.values() for layer in arm["layers"])
    chosen = next((count for count in (32, 64) if apparatus_pass and ladder[str(count)]), None)
    result = {
        "schema": "nes03_int8_router_probe_v1",
        "scope": "E128 pretrained-free native checkpoint inputs; E1280 synthetic router on E128 inputs, not self-consistent quality",
        "identity_sha256": EXPECTED_SHA256,
        "validation_offset": args.offset,
        "windows": args.windows,
        "seq": 512,
        "capture_device": args.device,
        "capture_seconds": capture_seconds,
        "captured_inputs_bytes": input_path.stat().st_size,
        "captured_inputs_sha256": sha256(input_path),
        "apparatus_pass": apparatus_pass,
        "shortlist_gates": ladder,
        "chosen_shortlist": chosen,
        "advance_to_native": chosen is not None and args.windows == 16 and args.offset == 8192,
        "arms": arms,
    }
    out = args.out_dir / "probe.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("apparatus", apparatus_pass, "shortlist gates", ladder,
          "chosen", chosen, "advance", result["advance_to_native"], flush=True)


if __name__ == "__main__":
    main()
