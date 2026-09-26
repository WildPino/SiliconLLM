#!/usr/bin/env python3
"""METH-21: full PIQA paired accuracy for donor and exported half adapter."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import psutil
from safetensors import safe_open
from safetensors.torch import load_file
import torch

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth20_half_adapter_generation as M20


ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = ROOT / "benchmarks/donor_adaptation/density/results/strat01_task_sources_20260919"
PIQA = TASK_DIR / "piqa_valid.jsonl"
LABELS = TASK_DIR / "piqa_valid-labels.lst"
PIQA_SHA = "93503cc97c679e459b065c3d13e848282e44b2a25213985bed3e5d458abef72d"
LABELS_SHA = "b4192dc3a2a0363d9d60ccf79800cbbe2f32ebb17726efdde6970e0b8131bceb"
ITEMS = 1838
MAX_CONTEXT = 4096
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 212121
MAX_SECONDS = 25 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def check_budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS:
        raise TimeoutError(f"METH-21 wall-time stop: {elapsed:.1f}s")
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"METH-21 RSS stop: {rss}")
    if peak > MAX_GPU_BYTES:
        raise MemoryError(f"METH-21 GPU stop: {peak}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss,
            "gpu_peak_allocated_bytes": peak}


def bind_data(tokenizer):
    assert M17.sha(PIQA.read_bytes()) == PIQA_SHA
    assert M17.sha(LABELS.read_bytes()) == LABELS_SHA
    raw_rows = [json.loads(line) for line in PIQA.read_text(encoding="utf-8").splitlines()]
    labels = [int(line) for line in LABELS.read_text(encoding="utf-8").splitlines()]
    assert len(raw_rows) == len(labels) == ITEMS
    rows = []
    meta = []
    for index, (source, label) in enumerate(zip(raw_rows, labels)):
        assert label in (0, 1)
        assert set(source) == {"goal", "sol1", "sol2"}
        assert all(isinstance(source[key], str) and source[key]
                   for key in ("goal", "sol1", "sol2"))
        prefix = tokenizer.encode(f"Question: {source['goal']}\nAnswer:",
                                  add_special_tokens=False)
        suffixes = [tokenizer.encode(" " + source[key], add_special_tokens=False)
                    for key in ("sol1", "sol2")]
        assert prefix and all(suffixes)
        assert all(1 + len(prefix) + len(suffix) <= MAX_CONTEXT
                   for suffix in suffixes)
        row = {"index": index, "label": label,
               "prefix": prefix, "suffixes": suffixes}
        rows.append(row)
        meta.append({"index": index, "label": label,
                     "source_sha256": M17.sha(json.dumps(source, sort_keys=True,
                                                          ensure_ascii=False).encode("utf-8")),
                     "prefix_ids_sha256": M17.sha(np.asarray(prefix, dtype=np.int32).tobytes()),
                     "suffix_ids_sha256": [M17.sha(np.asarray(x, dtype=np.int32).tobytes())
                                            for x in suffixes],
                     "prefix_tokens": len(prefix),
                     "suffix_tokens": [len(x) for x in suffixes]})
    manifest = {"experiment": "METH-21", "source_sha256": PIQA_SHA,
                "labels_sha256": LABELS_SHA, "adapter_sha256": M20.ADAPTER_SHA,
                "model_sha256": M15.M13.MODEL_SHA,
                "tokenizer_fingerprint": M15.M13.TOK_FP,
                "count": ITEMS, "eos_id": tokenizer.eos_token_id,
                "prompt_format": "EOS + separately tokenized 'Question: {goal}\\nAnswer:' + separately tokenized ' {solution}'",
                "primary_metric": "choice with lower mean suffix NLL; exact ties choose option 0",
                "items": meta}
    return manifest, rows


@torch.inference_mode()
def score_option(model, prefix, suffix, device):
    eos = model.config.eos_token_id
    inputs = torch.as_tensor([eos, *prefix, *suffix[:-1]],
                             dtype=torch.long, device=device).unsqueeze(0)
    output = model(inputs, use_cache=False).logits[0, len(prefix):len(prefix)+len(suffix)]
    output = output.float()
    targets = torch.as_tensor(suffix, dtype=torch.long, device=device)
    selected = output.gather(1, targets[:, None]).squeeze(1)
    nll = torch.logsumexp(output, dim=-1) - selected
    total = float(nll.sum())
    assert math.isfinite(total) and total >= 0
    return total, total / len(suffix)


def summarize(rows):
    out = {"items": len(rows)}
    for arm in ("donor", "half"):
        primary = [r[arm]["choice_mean"] == r["label"] for r in rows]
        secondary = [r[arm]["choice_total"] == r["label"] for r in rows]
        out[arm] = {"correct_mean": sum(primary), "accuracy_mean": sum(primary)/len(rows),
                    "correct_total": sum(secondary), "accuracy_total": sum(secondary)/len(rows),
                    "gold_mean_nll": sum(r[arm]["options"][r["label"]]["mean_nll"]
                                         for r in rows) / len(rows)}
    out["accuracy_delta_half_minus_donor"] = (
        out["half"]["accuracy_mean"] - out["donor"]["accuracy_mean"])
    out["discordance"] = {
        "donor_correct_half_wrong": sum(r["donor"]["choice_mean"] == r["label"]
                                        and r["half"]["choice_mean"] != r["label"]
                                        for r in rows),
        "donor_wrong_half_correct": sum(r["donor"]["choice_mean"] != r["label"]
                                        and r["half"]["choice_mean"] == r["label"]
                                        for r in rows)}
    paired = np.asarray([int(r["half"]["choice_mean"] == r["label"])
                         - int(r["donor"]["choice_mean"] == r["label"])
                         for r in rows], dtype=np.int8)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [float(paired[rng.integers(0, len(rows), len(rows))].mean())
             for _ in range(BOOTSTRAP_DRAWS)]
    out["paired_bootstrap_lower95"] = float(np.quantile(draws, 0.05))
    out["bootstrap_seed"] = BOOTSTRAP_SEED
    out["bootstrap_draws"] = BOOTSTRAP_DRAWS
    out["pass"] = bool(out["accuracy_delta_half_minus_donor"] >= -0.02
                       and out["paired_bootstrap_lower95"] >= -0.05)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--manifest-sha256")
    ap.add_argument("--out")
    args = ap.parse_args()
    torch.set_num_threads(6)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = hf_hub_download(M15.M13.MODEL, "model.safetensors",
                             revision=M15.M13.REV, local_files_only=True)
    assert M15.M13.sha256(source) == M15.M13.MODEL_SHA
    tokenizer = AutoTokenizer.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    manifest, items = bind_data(tokenizer)
    manifest_path = Path(args.manifest)
    if args.prepare:
        assert args.manifest_sha256 is None and args.out is None
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"manifest_sha256": M17.sha(manifest_path.read_bytes()),
                          "items": len(items),
                          "max_sequence_tokens": max(1 + len(r["prefix"]) +
                                                     max(map(len, r["suffixes"]))
                                                     for r in items)}, indent=2), flush=True)
        return
    assert args.manifest_sha256 and args.out
    assert M17.sha(manifest_path.read_bytes()) == args.manifest_sha256
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert M15.M13.sha256(M20.ADAPTER) == M20.ADAPTER_SHA
    with safe_open(str(M20.ADAPTER), framework="pt", device="cpu") as archive:
        meta = archive.metadata()
    assert meta["model_sha256"] == M15.M13.MODEL_SHA
    assert meta["source_checkpoint_sha256"] == M17.CHECKPOINT_SHA
    assert meta["output_factor"] == "0.5"
    tensors = load_file(str(M20.ADAPTER), device="cpu")
    start = time.monotonic()
    torch.set_grad_enabled(False)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = AutoModelForCausalLM.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    assert model.config.eos_token_id == tokenizer.eos_token_id
    for p in model.parameters():
        p.requires_grad_(False)
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = M15.ResidualExperts(layer.mlp, li).to(device)
        layer.mlp = wrapper
        for key in ("a", "b", "router"):
            getattr(wrapper, key).copy_(tensors[f"layers.{li}.{key}"].to(device))
        wrappers.append(wrapper)
    assert len(tensors) == 3 * M15.M13.L
    del tensors
    rows = []
    with torch.inference_mode():
        for i, item in enumerate(items):
            result = {"index": i, "label": item["label"]}
            for enabled, arm in ((False, "donor"), (True, "half")):
                for w in wrappers:
                    w.enabled = enabled
                scores = [score_option(model, item["prefix"], suffix, device)
                          for suffix in item["suffixes"]]
                mean_choice = 0 if scores[0][1] <= scores[1][1] else 1
                total_choice = 0 if scores[0][0] <= scores[1][0] else 1
                result[arm] = {"choice_mean": mean_choice,
                               "choice_total": total_choice,
                               "options": [{"total_nll": total,
                                            "mean_nll": mean,
                                            "tokens": len(suffix)}
                                           for (total, mean), suffix in zip(scores, item["suffixes"])]}
            rows.append(result)
            check_budget(start, device)
            if (i+1) % 128 == 0:
                print(f"scored {i+1}/{len(items)} PIQA items", flush=True)
    summary = summarize(rows)
    runtime = check_budget(start, device)
    result = {"experiment": "METH-21", "manifest_sha256": args.manifest_sha256,
              "adapter_sha256": M20.ADAPTER_SHA,
              "model_sha256": M15.M13.MODEL_SHA,
              "rows": rows, "summary": summary,
              "decision": "paired_task_gate_pass" if summary["pass"]
                          else "paired_task_gate_fail",
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary,
                      "decision": result["decision"],
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
