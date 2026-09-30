#!/usr/bin/env python3
"""Ablate/amplify only trained E12800 content-child deviations."""

import argparse
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import psutil
import torch

import meth179_content_route_function as P


M136, M151, M156, M175, M176 = P.M136, P.M151, P.M156, P.M175, P.M176
DOC = P.DOC
PRIOR = DOC / "meth179_content_route_function_result.json"
PRIOR_SHA = "32bc1760dbdb4ce3efcaa98fa8d5f266fa773cc22d9f434369bc75ae772fef4a"
ARMS = (1, 0, 4)
MAX_SECONDS = 20 * 60
MAX_RSS = 35 * (1 << 30)
MAX_GPU = int(10.5 * (1 << 30))
MAX_DISK = 1_000_000_000


def budget(start, device):
    row = {"seconds": time.monotonic() - start,
           "rss_bytes": psutil.Process().memory_info().rss,
           "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if row["seconds"] > MAX_SECONDS or row["rss_bytes"] > MAX_RSS or row[
            "gpu_peak_allocated_bytes"] > MAX_GPU:
        raise RuntimeError(f"METH-202 resource stop: {row}")
    return row


def apply_amplitude(wrappers, original_bits, amplitude, start, device):
    assert amplitude in (0, 4)
    assert original_bits.shape == (24, 12800, 896, 8)
    for layer, wrapper in enumerate(wrappers):
        original = torch.from_numpy(M136.bf16_to_f32(original_bits[layer]).copy())
        assert original.shape == wrapper.cpu_bank.shape == (12800, 896, 8)
        view = original.view(1280, 10, 896, 8)
        source_content = view[:, 1:]
        mean = source_content.mean(dim=1, keepdim=True)
        transformed = (mean + amplitude * (source_content - mean)).to(
            torch.bfloat16).float()
        with torch.no_grad():
            wrapper.cpu_bank.copy_(original)
            wrapper.cpu_bank.view(1280, 10, 896, 8)[:, 1:].copy_(transformed)
            assert torch.equal(wrapper.cpu_bank.view(1280, 10, 896, 8)[:, 0],
                               view[:, 0])
            assert torch.equal(wrapper.cpu_bank.view(1280, 10, 896, 8)[:, 1:]
                               .to(torch.bfloat16).float(), transformed)
        del original, view, source_content, mean, transformed
        if layer % 4 == 3:
            gc.collect()
            budget(start, device)


def summarize(rows):
    bytes_total = sum(row["bytes"] for row in rows)
    assert len(rows) == 24 and bytes_total > 0
    bpb = {str(arm): sum(row["nats"][str(arm)] for row in rows) /
           (math.log(2) * bytes_total) for arm in ARMS}
    gain = bpb["0"] - bpb["4"]
    rng = np.random.default_rng(202202)
    nats = np.asarray([row["nats"]["0"] - row["nats"]["4"]
                       for row in rows], dtype=np.float64)
    byte_counts = np.asarray([row["bytes"] for row in rows], dtype=np.float64)
    draws = rng.integers(0, len(rows), size=(10000, len(rows)))
    boot = nats[draws].sum(axis=1) / (math.log(2) * byte_counts[draws].sum(axis=1))
    p05 = float(np.quantile(boot, .05))
    gates = {"lambda4_gain_at_least_0_0002_bpb": gain >= .0002,
             "paired_bootstrap_p05_positive": p05 > 0}
    return {"bytes": bytes_total, "bpb_by_amplitude": bpb,
            "lambda0_minus_lambda1_bpb": bpb["0"] - bpb["1"],
            "lambda0_minus_lambda4_bpb": gain,
            "bootstrap": {"seed": 202202, "draws": 10000,
                          "lambda0_minus_lambda4_p05_bpb": p05,
                          "lambda0_minus_lambda4_median_bpb": float(np.median(boot))},
            "gates": gates,
            "decision": "child_direction_signal_pass" if all(gates.values()) else
                        "child_direction_signal_fail_change_route_training_coupling"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    stage = "bindings"
    started = time.monotonic()
    device = None
    try:
        assert P.digest(P.MANIFEST) == P.MANIFEST_SHA
        assert P.digest(P.CANDIDATE) == P.CANDIDATE_SHA
        assert P.digest(PRIOR) == PRIOR_SHA
        manifest = json.loads(P.MANIFEST.read_text(encoding="utf-8"))
        candidate = json.loads(P.CANDIDATE.read_text(encoding="utf-8"))
        prior = json.loads(PRIOR.read_text(encoding="utf-8"))
        assert manifest["table_sha256"] == M175.TABLE_SHA
        assert manifest["tokenizer_fingerprint"] == M136.M15.M13.TOK_FP
        assert prior["tokenizer_fingerprint"] == manifest["tokenizer_fingerprint"]
        bank = Path(candidate["artifact"]["path"])
        assert P.digest(bank) == candidate["artifact"]["sha256"]
        assert prior["candidate_bank_sha256"] == candidate["artifact"]["sha256"]
        items = manifest["document_rows"]
        assert len(items) == len(prior["document_rows"]) == 24
        rows = []
        for item, old in zip(items, prior["document_rows"]):
            ids = np.asarray(item["span_ids"], dtype=np.int32)
            assert ids.shape == (1024,)
            assert P.M17.sha(ids.tobytes()) == item["span_ids_sha256"]
            assert old["source_row"] == item["source_row"]
            assert old["span_ids_sha256"] == item["span_ids_sha256"]
            rows.append({"source_row": item["source_row"],
                         "span_ids_sha256": item["span_ids_sha256"],
                         "bytes": old["bytes"], "nats": {}})
        torch.set_num_threads(6)
        torch.set_grad_enabled(False)
        matches = [i for i in range(torch.cuda.device_count())
                   if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
        assert len(matches) == 1
        device = torch.device(f"cuda:{matches[0]}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        P.M17.MAX_SECONDS = M176.MAX_SECONDS = M136.MAX_SECONDS = MAX_SECONDS
        P.M17.MAX_RSS_BYTES = M176.MAX_RSS = M136.MAX_RSS = MAX_RSS
        P.M17.MAX_GPU_BYTES = M176.MAX_GPU = M136.MAX_GPU = MAX_GPU
        table = M175.load_table()
        stage = "initial_parity"
        parent, child = M136.bind_inputs()
        source_a, source_b, prefixes = M136.load_factor_bank()
        parity = M176.initial_parity(parent, child, source_a, source_b,
                                     prefixes, table, device, started)
        stage = "load_trained_bank"
        model = M136.model_shell(device).eval()
        original_class = M151.SharedStructureSparseExperts
        P.ShiftedContentExperts.route_table = table
        M151.SharedStructureSparseExperts = P.ShiftedContentExperts
        try:
            wrappers, _ = M136.make_wrappers(
                model, parent["expert_state"], child["expert_state"],
                source_a, source_b, prefixes, device, "shared")
        finally:
            M151.SharedStructureSparseExperts = original_class
        binding = M156.install_bank(wrappers, bank, 12800)
        assert binding["sha256"] == candidate["artifact"]["sha256"]
        del parent, child, source_a, source_b
        gc.collect()
        stage = "amplitude_scores"
        layer_bytes = 12800 * 896 * 8 * 2
        assert bank.stat().st_size == M136.OUT_HEADER.size + 24 * layer_bytes
        original_bits = np.memmap(bank, dtype="<u2", mode="r",
                                  offset=M136.OUT_HEADER.size,
                                  shape=(24, 12800, 896, 8))
        with torch.inference_mode():
            for amplitude in ARMS:
                if amplitude != 1:
                    apply_amplitude(wrappers, original_bits, amplitude,
                                    started, device)
                for row, item, old in zip(rows, items, prior["document_rows"]):
                    row["nats"][str(amplitude)] = M176.score_candidate_doc(
                        model, item["span_ids"], wrappers, device, started)
                    assert math.isfinite(row["nats"][str(amplitude)])
                    if amplitude == 1:
                        assert row["nats"]["1"] == old["nats"]["0"]
                    budget(started, device)
                print(json.dumps({"completed_amplitude": amplitude,
                                  "budget": budget(started, device)}), flush=True)
        stage = "summary"
        for row in rows:
            row["bpb"] = {arm: value / (math.log(2) * row["bytes"])
                          for arm, value in row["nats"].items()}
        summary = summarize(rows)
        result = {"experiment": "METH-202-trained-child-deviation-gain",
                  "manifest_sha256": P.MANIFEST_SHA,
                  "candidate_result_sha256": P.CANDIDATE_SHA,
                  "candidate_bank_sha256": binding["sha256"],
                  "prior_route_result_sha256": PRIOR_SHA,
                  "tokenizer_fingerprint": manifest["tokenizer_fingerprint"],
                  "initial_bf16_parity": parity,
                  "bank_readback": binding,
                  "document_rows": rows,
                  "summary": summary,
                  "runtime": {**budget(started, device),
                              "gpu": torch.cuda.get_device_name(device)},
                  "scope": "Already viewed METH-173 sources; child-gain mechanism only"}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        assert args.out.stat().st_size < MAX_DISK
        print(json.dumps({"summary": summary,
                          "runtime": result["runtime"]}), flush=True)
    except BaseException as error:
        failure = args.out.with_name(args.out.stem + ".failure.json")
        failure.write_text(json.dumps({"experiment": "METH-202-failure",
                                       "stage": stage, "error": repr(error),
                                       "elapsed_seconds": time.monotonic() - started,
                                       "rss_bytes": psutil.Process().memory_info().rss,
                                       "gpu_peak_allocated_bytes":
                                           torch.cuda.max_memory_allocated(device)
                                           if device is not None else None},
                                      indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
