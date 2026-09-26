#!/usr/bin/env python3
"""METH-17: bind disjoint documents, then score donor/adapter by document."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F

import meth15_zero_residual_expert_smoke as M15


ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "benchmarks/donor_adaptation/density/corpus"
EXTERNAL = CORPUS / "strat02_document_holdout_v1/heldout.jsonl"
EXTERNAL_SHA = "450da27e25755bb7c71215e148f5863d197af6893385033e6100bb212deafd0e"
CALIB_SHA = "10d4d28102625f399f715b6aa220b234c5015dcff199997ccafa06e5c59c89d0"
HELDOUT_SHA = "f46b0310c15faec59ca805d5688317d53b7655ae7099008fe5c3439460d58312"
CODE_REF = "45afab6847cfd328c105f73256441f36748dc323"
CHECKPOINT_SHA = "a9e74c9ceeb8a91fd04f9b9e35a981965db9ab824d2d25ea728dd940ea5365a4"
SELECT_SEED = "meth17-17017"
SPAN = 4095
N_SELECT = {"code": 24, "prose": 24, "technical_general": 12}
STRIDE = 512
CONTEXT = 512
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 171717
MAX_SECONDS = 30 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_bytes(path):
    return subprocess.check_output(["git", "show", f"{CODE_REF}:{path}"], cwd=ROOT)


def fragment_overlap(data, base_parts):
    marks = (data[:256], data[len(data)//2-128:len(data)//2+128], data[-256:])
    return any(mark in base for mark in marks for base in base_parts)


def ranking(key):
    return sha((SELECT_SEED + "|" + key).encode("utf-8"))


def build_selection():
    assert sha(EXTERNAL.read_bytes()) == EXTERNAL_SHA
    bases = [(CORPUS / name).read_bytes() for name in ("calib.txt", "heldout.txt")]
    assert [sha(x) for x in bases] == [CALIB_SHA, HELDOUT_SHA]
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", CODE_REF],
        cwd=ROOT, text=True).splitlines()
    pool = defaultdict(list)
    considered = defaultdict(int)
    excluded_overlap = defaultdict(int)
    for path in paths:
        if not (path.startswith("benchmarks/phase") and path.endswith((".c", ".py"))):
            continue
        raw = git_bytes(path)
        if len(raw) < SPAN:
            continue
        considered["code"] += 1
        start = int(ranking(path)[:16], 16) % (len(raw) - SPAN + 1)
        selected = raw[start:start+SPAN].decode("utf-8", errors="ignore")
        data = selected.encode("utf-8")
        if len(data) < 4000 or fragment_overlap(data, bases):
            excluded_overlap["code"] += 1
            continue
        pool["code"].append({"category": "code", "source_id": path,
                             "source_kind": "git", "source_ref": CODE_REF,
                             "source_sha256": sha(raw), "span_start_byte": start,
                             "text": selected})
    for line in EXTERNAL.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        category = row["category"]
        if category not in ("prose", "technical_general"):
            continue
        considered[category] += 1
        full = row["text"].encode("utf-8")
        if fragment_overlap(full, bases):
            excluded_overlap[category] += 1
            continue
        selected = full[:SPAN].decode("utf-8", errors="ignore")
        pool[category].append({"category": category,
                               "source_id": row["source_document_id"],
                               "source_kind": row["source_kind"],
                               "source_ref": EXTERNAL_SHA,
                               "source_sha256": row["source_content_sha256"],
                               "span_start_byte": row["span_start_byte"],
                               "text": selected})
    selected = []
    for category, amount in N_SELECT.items():
        candidates = sorted(pool[category], key=lambda x: ranking(x["source_id"]))
        assert len(candidates) >= amount, (category, len(candidates), amount)
        selected += candidates[:amount]
    assert len({x["source_id"] for x in selected}) == len(selected)
    assert len({sha(x["text"].encode()) for x in selected}) == len(selected)
    meta = []
    for x in selected:
        y = {k: v for k, v in x.items() if k != "text"}
        y["text_sha256"] = sha(x["text"].encode("utf-8"))
        y["text_bytes"] = len(x["text"].encode("utf-8"))
        meta.append(y)
    manifest = {"experiment": "METH-17", "selection_seed": SELECT_SEED,
                "code_ref": CODE_REF, "external_sha256": EXTERNAL_SHA,
                "excluded_corpus_sha256": {"calib": CALIB_SHA,
                                            "heldout": HELDOUT_SHA},
                "overlap_rule": "first, middle and final 256 bytes absent from both Qwen corpus files",
                "source_candidates": dict(considered),
                "excluded_overlap": dict(excluded_overlap),
                "selected_counts": N_SELECT, "span_max_bytes": SPAN,
                "items": meta}
    return manifest, selected


def budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS:
        raise TimeoutError(f"METH-17 time stop: {elapsed:.1f}s")
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"METH-17 RSS stop: {rss}")
    if peak > MAX_GPU_BYTES:
        raise MemoryError(f"METH-17 allocated GPU stop: {peak}")
    return elapsed, rss, peak


@torch.no_grad()
def score_doc(model, ids, wrappers, enabled, device, start):
    for w in wrappers:
        w.enabled = enabled
    prefix = torch.tensor([model.config.eos_token_id] + ids,
                          dtype=torch.long, device=device)
    nats = 0.0
    for first in range(0, len(ids), STRIDE):
        end = min(first + STRIDE, len(ids))
        lo = max(0, first - CONTEXT)
        window = prefix[lo:end].unsqueeze(0)
        positions = torch.arange(lo, end, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(window, position_ids=positions, use_cache=False).logits
        selected_logits = logits[0, first-lo:end-lo].float()
        lp = F.log_softmax(selected_logits, dim=-1)
        targets = torch.as_tensor(ids[first:end], dtype=torch.long, device=device)
        nats += float(-lp.gather(1, targets[:, None]).sum())
        budget(start, device)
    return nats


def summarize(rows):
    def bpb(group, key):
        return sum(r[key] for r in group) / (math.log(2) * sum(r["bytes"] for r in group))
    groups = {category: [r for r in rows if r["category"] == category]
              for category in N_SELECT}
    out = {}
    for category, items in [("pooled", rows), *groups.items()]:
        donor = bpb(items, "donor_nats")
        student = bpb(items, "student_nats")
        out[category] = {"docs": len(items), "bytes": sum(r["bytes"] for r in items),
                         "tokens": sum(r["tokens"] for r in items),
                         "donor_bpb": donor, "student_bpb": student,
                         "delta_bpb": student-donor}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = []
        for group in groups.values():
            sampled += [group[i] for i in rng.integers(0, len(group), size=len(group))]
        draws.append(bpb(sampled, "student_nats") - bpb(sampled, "donor_nats"))
    out["pooled"]["one_sided_ci95_upper"] = float(np.quantile(draws, 0.95))
    out["pooled"]["bootstrap_seed"] = BOOTSTRAP_SEED
    out["pooled"]["bootstrap_draws"] = BOOTSTRAP_DRAWS
    out["pass"] = bool(out["pooled"]["delta_bpb"] <= 0.01
                       and out["pooled"]["one_sided_ci95_upper"] <= 0.02
                       and all(out[c]["delta_bpb"] <= 0.05 for c in N_SELECT))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--manifest-sha256")
    ap.add_argument("--checkpoint")
    ap.add_argument("--out")
    args = ap.parse_args()
    manifest, items = build_selection()
    manifest_path = Path(args.manifest)
    if args.prepare:
        assert args.checkpoint is None and args.out is None
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"manifest_sha256": sha(manifest_path.read_bytes()),
                          "counts": N_SELECT,
                          "candidate_counts": manifest["source_candidates"],
                          "excluded_overlap": manifest["excluded_overlap"],
                          "bytes": sum(x["text_bytes"] for x in manifest["items"])},
                         indent=2), flush=True)
        return
    assert args.manifest_sha256 and args.checkpoint and args.out
    assert sha(manifest_path.read_bytes()) == args.manifest_sha256
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert M15.M13.sha256(args.checkpoint) == CHECKPOINT_SHA
    torch.set_grad_enabled(False)
    torch.set_num_threads(6)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
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
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint["updates"] == 1024
    assert checkpoint["source_sha256"] == M15.M13.MODEL_SHA
    assert checkpoint["train_ids_sha256"] == M15.TRAIN_IDS_SHA
    for w, values in zip(wrappers, checkpoint["expert_state"]):
        w.a.data.copy_(values["a"].to(device))
        w.b.data.copy_(values["b"].to(device))
        w.router.data.copy_(values["router"].to(device))
    del checkpoint
    rows = []
    for i, item in enumerate(items):
        ids = tok.encode(item["text"], add_special_tokens=False)
        assert len(ids) >= 2 and max(ids) < 151936
        donor = score_doc(model, ids, wrappers, False, device, start)
        student = score_doc(model, ids, wrappers, True, device, start)
        rows.append({"index": i, "category": item["category"],
                     "source_id": item["source_id"],
                     "text_sha256": sha(item["text"].encode()),
                     "bytes": len(item["text"].encode()), "tokens": len(ids),
                     "ids_sha256": sha(np.asarray(ids, dtype=np.int32).tobytes()),
                     "donor_nats": donor, "student_nats": student,
                     "delta_bpb": (student-donor)/(math.log(2)*len(item["text"].encode()))})
        if (i+1) % 6 == 0:
            print(f"scored {i+1}/{len(items)} documents", flush=True)
    summary = summarize(rows)
    elapsed, rss, peak = budget(start, device)
    result = {"experiment": "METH-17", "manifest_sha256": args.manifest_sha256,
              "source": {"model": M15.M13.MODEL, "revision": M15.M13.REV,
                         "model_sha256": M15.M13.MODEL_SHA,
                         "tokenizer_fingerprint": M15.M13.TOK_FP,
                         "checkpoint_sha256": CHECKPOINT_SHA},
              "scoring": {"prefix_token": model.config.eos_token_id,
                          "stride": STRIDE, "left_context": CONTEXT,
                          "position_ids": "document absolute positions",
                          "score_all_text_tokens": True},
              "rows": rows, "summary": summary,
              "runtime": {"torch": torch.__version__,
                          "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "elapsed_seconds": elapsed,
                          "rss_end_bytes": rss,
                          "gpu_peak_allocated_bytes": peak},
              "decision": "fresh_document_quality_pass" if summary["pass"]
                          else "fresh_document_quality_fail_or_inconclusive"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "runtime": result["runtime"],
                      "decision": result["decision"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
