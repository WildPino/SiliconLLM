#!/usr/bin/env python3
"""METH-19: independent document audit of a fixed half-residual candidate."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import psutil
import torch

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17


ROOT = Path(__file__).resolve().parents[3]
EXTERNAL = M17.CORPUS / "strat01_gigachat_fresh_v2/heldout.jsonl"
EXTERNAL_SHA = "04687034c1054e0985e24efa87ba37a3fb11031de5b2880e5b8ac1742f80ec6e"
M17_MANIFEST = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth17_fresh_document_manifest.json"
M17_MANIFEST_SHA = "7e0593d6c56c28398e3440f9b80db15d506a31131a21480a87578c23b0a043a8"
CHECKPOINT = ROOT / "results/native_expert_scaling/meth16_checkpoints/meth16_update1024.pt"
SEED = "meth19-19019"
COUNTS = {"code": 24, "prose": 24, "technical_general": 8}
ARMS = ("donor", "half", "full", "permuted_half")
PERM_SEED = 1919
BOOTSTRAP_SEED = 191919
BOOTSTRAP_DRAWS = 20000
MAX_SECONDS = 15 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def rank(key):
    return M17.sha((SEED + "|" + key).encode("utf-8"))


def select():
    assert M17.sha(EXTERNAL.read_bytes()) == EXTERNAL_SHA
    old_manifest, old_items = M17.build_selection()
    assert M17.sha(M17_MANIFEST.read_bytes()) == M17_MANIFEST_SHA
    assert json.loads(M17_MANIFEST.read_text(encoding="utf-8")) == old_manifest
    old_ids = {item["source_id"] for item in old_items}
    base = [(M17.CORPUS / name).read_bytes()
            for name in ("calib.txt", "heldout.txt")]
    assert [M17.sha(x) for x in base] == [M17.CALIB_SHA, M17.HELDOUT_SHA]
    base += [item["text"].encode("utf-8") for item in old_items]
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", M17.CODE_REF],
        cwd=ROOT, text=True).splitlines()
    pool = defaultdict(list)
    considered = defaultdict(int)
    excluded_overlap = defaultdict(int)
    excluded_old_source = defaultdict(int)
    for path in paths:
        if not (path.startswith("benchmarks/phase") and path.endswith((".c", ".py"))):
            continue
        raw = M17.git_bytes(path)
        if len(raw) < M17.SPAN:
            continue
        if path in old_ids:
            excluded_old_source["code"] += 1
            continue
        considered["code"] += 1
        start = int(rank(path)[:16], 16) % (len(raw) - M17.SPAN + 1)
        selected = raw[start:start+M17.SPAN].decode("utf-8", errors="ignore")
        data = selected.encode("utf-8")
        if len(data) < 4000 or M17.fragment_overlap(data, base):
            excluded_overlap["code"] += 1
            continue
        pool["code"].append({"category": "code", "source_id": path,
                             "source_kind": "git", "source_ref": M17.CODE_REF,
                             "source_sha256": M17.sha(raw),
                             "span_start_byte": start, "text": selected})
    for line in EXTERNAL.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        category = row["category"]
        if category not in ("prose", "technical_general"):
            continue
        if row["source_document_id"] in old_ids:
            excluded_old_source[category] += 1
            continue
        considered[category] += 1
        full = row["text"].encode("utf-8")
        if M17.fragment_overlap(full, base):
            excluded_overlap[category] += 1
            continue
        selected = full[:M17.SPAN].decode("utf-8", errors="ignore")
        pool[category].append({"category": category,
                               "source_id": row["source_document_id"],
                               "source_kind": row["source_kind"],
                               "source_ref": EXTERNAL_SHA,
                               "source_sha256": row["source_content_sha256"],
                               "span_start_byte": row["span_start_byte"],
                               "text": selected})
    items = []
    for category, count in COUNTS.items():
        candidates = sorted(pool[category], key=lambda x: rank(x["source_id"]))
        assert len(candidates) >= count, (category, len(candidates), count)
        items += candidates[:count]
    assert len({x["source_id"] for x in items}) == len(items)
    assert len({M17.sha(x["text"].encode()) for x in items}) == len(items)
    meta = []
    for item in items:
        row = {k: v for k, v in item.items() if k != "text"}
        row["text_sha256"] = M17.sha(item["text"].encode("utf-8"))
        row["text_bytes"] = len(item["text"].encode("utf-8"))
        meta.append(row)
    manifest = {"experiment": "METH-19", "seed": SEED,
                "external_sha256": EXTERNAL_SHA, "code_ref": M17.CODE_REF,
                "prior_manifest_sha256": M17_MANIFEST_SHA,
                "excluded_corpus_sha256": {"calib": M17.CALIB_SHA,
                                            "heldout": M17.HELDOUT_SHA},
                "overlap_rule": "first, middle and final 256 bytes absent from each Qwen corpus file and all METH-17 text spans",
                "source_candidates": dict(considered),
                "excluded_old_source": dict(excluded_old_source),
                "excluded_overlap": dict(excluded_overlap),
                "selected_counts": COUNTS, "span_max_bytes": M17.SPAN,
                "items": meta}
    return manifest, items


def check_budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS:
        raise TimeoutError(f"METH-19 wall-time stop: {elapsed:.1f}s")
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"METH-19 RSS stop: {rss}")
    if peak > MAX_GPU_BYTES:
        raise MemoryError(f"METH-19 GPU stop: {peak}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss,
            "gpu_peak_allocated_bytes": peak}


def summarize(rows):
    groups = {"pooled": rows}
    groups.update({category: [r for r in rows if r["category"] == category]
                   for category in COUNTS})

    def bpb(group, arm):
        return sum(r["nats"][arm] for r in group) / (
            math.log(2) * sum(r["bytes"] for r in group))

    summary = {}
    for name, group in groups.items():
        summary[name] = {"docs": len(group), "bytes": sum(r["bytes"] for r in group),
                         "tokens": sum(r["tokens"] for r in group),
                         "donor_bpb": bpb(group, "donor"),
                         "half_bpb": bpb(group, "half"),
                         "full_bpb": bpb(group, "full"),
                         "permuted_half_bpb": bpb(group, "permuted_half")}
        summary[name]["half_delta_bpb"] = (
            summary[name]["half_bpb"] - summary[name]["donor_bpb"])
        summary[name]["full_delta_bpb"] = (
            summary[name]["full_bpb"] - summary[name]["donor_bpb"])
        summary[name]["half_route_utility_bpb"] = (
            summary[name]["permuted_half_bpb"] - summary[name]["half_bpb"])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = []
        for category in COUNTS:
            group = groups[category]
            sampled += [group[i] for i in rng.integers(0, len(group), size=len(group))]
        draws.append(bpb(sampled, "half") - bpb(sampled, "donor"))
    summary["pooled"]["half_ci95_upper"] = float(np.quantile(draws, 0.95))
    summary["pooled"]["bootstrap_seed"] = BOOTSTRAP_SEED
    summary["pooled"]["bootstrap_draws"] = BOOTSTRAP_DRAWS
    passed = (summary["pooled"]["half_delta_bpb"] <= 0.01
              and summary["pooled"]["half_ci95_upper"] <= 0.02
              and summary["code"]["half_delta_bpb"] <= 0.01
              and summary["prose"]["half_delta_bpb"] <= 0.03
              and summary["technical_general"]["half_delta_bpb"] <= 0.03
              and summary["pooled"]["half_route_utility_bpb"] >= 0.002)
    return summary, passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--manifest-sha256")
    ap.add_argument("--out")
    args = ap.parse_args()
    manifest, items = select()
    manifest_path = Path(args.manifest)
    if args.prepare:
        assert args.manifest_sha256 is None and args.out is None
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"manifest_sha256": M17.sha(manifest_path.read_bytes()),
                          "counts": COUNTS, "bytes": sum(x["text_bytes"] for x in manifest["items"]),
                          "source_candidates": manifest["source_candidates"],
                          "excluded_old_source": manifest["excluded_old_source"],
                          "excluded_overlap": manifest["excluded_overlap"]}, indent=2))
        return
    assert args.manifest_sha256 and args.out
    assert M17.sha(manifest_path.read_bytes()) == args.manifest_sha256
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert M15.M13.sha256(CHECKPOINT) == M17.CHECKPOINT_SHA
    start = time.monotonic()
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = hf_hub_download(M15.M13.MODEL, "model.safetensors",
                             revision=M15.M13.REV, local_files_only=True)
    assert M15.M13.sha256(source) == M15.M13.MODEL_SHA
    tok = AutoTokenizer.from_pretrained(M15.M13.MODEL, revision=M15.M13.REV,
                                        local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tok) == M15.M13.TOK_FP
    model = AutoModelForCausalLM.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = M15.ResidualExperts(layer.mlp, li).to(device)
        layer.mlp = wrapper
        wrappers.append(wrapper)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert checkpoint["updates"] == 1024
    assert checkpoint["source_sha256"] == M15.M13.MODEL_SHA
    assert checkpoint["train_ids_sha256"] == M15.TRAIN_IDS_SHA
    for w, values in zip(wrappers, checkpoint["expert_state"]):
        w.a.copy_(values["a"].to(device))
        w.b.copy_(values["b"].to(device))
        w.router.copy_(values["router"].to(device))
    del checkpoint
    original_b = [w.b.detach().clone() for w in wrappers]
    original_router = [w.router.detach().clone() for w in wrappers]
    perm_rng = np.random.default_rng(PERM_SEED)
    permuted_router = [r[torch.as_tensor(perm_rng.permutation(M15.E),
                                         dtype=torch.long, device=device)]
                       for r in original_router]

    def set_arm(arm):
        for w, b, router, permuted in zip(wrappers, original_b,
                                          original_router, permuted_router):
            if arm != "donor":
                w.b.copy_(b if arm == "full" else b * 0.5)
            w.router.copy_(permuted if arm == "permuted_half" else router)

    rows = []
    for i, item in enumerate(items):
        ids = tok.encode(item["text"], add_special_tokens=False)
        row = {"index": i, "category": item["category"],
               "source_id": item["source_id"],
               "text_sha256": M17.sha(item["text"].encode("utf-8")),
               "bytes": len(item["text"].encode("utf-8")),
               "tokens": len(ids), "nats": {}}
        for arm in ARMS:
            set_arm(arm)
            row["nats"][arm] = M17.score_doc(model, ids, wrappers,
                                               arm != "donor", device, start)
        rows.append(row)
        check_budget(start, device)
        if (i + 1) % 8 == 0:
            print(f"scored {i+1}/{len(items)} documents", flush=True)
    summary, passed = summarize(rows)
    runtime = check_budget(start, device)
    result = {"experiment": "METH-19", "manifest_sha256": args.manifest_sha256,
              "source": {"model": M15.M13.MODEL, "revision": M15.M13.REV,
                         "model_sha256": M15.M13.MODEL_SHA,
                         "tokenizer_fingerprint": M15.M13.TOK_FP,
                         "checkpoint_sha256": M17.CHECKPOINT_SHA},
              "arms": ARMS, "half_factor": 0.5,
              "permutation_seed": PERM_SEED,
              "rows": rows, "summary": summary,
              "document_gate_pass": passed,
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "document_gate_pass": passed,
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
