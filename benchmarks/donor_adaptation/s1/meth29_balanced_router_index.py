#!/usr/bin/env python3
"""METH-29: post-hoc balanced coarse router index on real Qwen E128 inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth20_half_adapter_generation as M20
import meth24_export_r8_core as M24
import meth25_fresh_r8_artifact_audit as M25
import meth25_fresh_r8_manifest as M25M
import meth27_r8_fresh_generation as M27


ROOT = Path(__file__).resolve().parents[3]
PROMPT_MANIFEST = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth27_r8_fresh_generation_manifest.json"
PROMPT_MANIFEST_SHA = "94abd7544d4f887e476286774b2ce51bbc0c5c68f079907b88b3984589d62ec6"
GROUP_SIZE = 8
LEAVES = M15.E // GROUP_SIZE
ARMS = {"candidates32": 4, "candidates64": 8}
MAX_SECONDS = 20 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def check_budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS:
        raise TimeoutError(f"METH-29 wall-time stop: {elapsed:.1f}s")
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"METH-29 RSS stop: {rss}")
    if peak > MAX_GPU_BYTES:
        raise MemoryError(f"METH-29 GPU stop: {peak}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss,
            "gpu_peak_allocated_bytes": peak}


def balanced_groups(router_cpu):
    assert router_cpu.shape == (M15.E, M15.M13.D)
    rows = router_cpu.float().contiguous()

    def split(ids):
        if len(ids) == GROUP_SIZE:
            return [ids]
        assert len(ids) > GROUP_SIZE and len(ids) % 2 == 0
        sub = rows[ids]
        centered = sub - sub.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        projection = (centered @ vh[0]).numpy()
        order = np.lexsort((np.asarray(ids, dtype=np.int32), projection))
        arranged = [ids[int(j)] for j in order]
        middle = len(ids) // 2
        return split(arranged[:middle]) + split(arranged[middle:])

    leaves = split(list(range(M15.E)))
    assert len(leaves) == LEAVES
    assert all(len(group) == GROUP_SIZE for group in leaves)
    assert sorted(x for group in leaves for x in group) == list(range(M15.E))
    ids = torch.as_tensor(leaves, dtype=torch.long)
    means = rows[ids].mean(dim=1)
    return leaves, ids, means


def empty_count():
    return {"positions": 0, "exact_ids": 0, "included_exact_ids": 0,
            "full_set_matches": 0, "missed_gate_mass_sum": 0.0,
            "worst_missed_gate_mass": 0.0}


def add_count(dst, src):
    for key in ("positions", "exact_ids", "included_exact_ids", "full_set_matches"):
        dst[key] += src[key]
    dst["missed_gate_mass_sum"] += src["missed_gate_mass_sum"]
    dst["worst_missed_gate_mass"] = max(dst["worst_missed_gate_mass"],
                                        src["worst_missed_gate_mass"])


def finalize_count(row):
    row = dict(row)
    row["exact_id_inclusion_fraction"] = row["included_exact_ids"] / row["exact_ids"]
    row["full_set_match_fraction"] = row["full_set_matches"] / row["positions"]
    row["mean_missed_gate_mass"] = row["missed_gate_mass_sum"] / row["positions"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert M17.sha(PROMPT_MANIFEST.read_bytes()) == PROMPT_MANIFEST_SHA
    assert M15.M13.sha256(M25.CORE) == M25.CORE_SHA
    assert M15.M13.sha256(M20.ADAPTER) == M20.ADAPTER_SHA
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    start = time.monotonic()
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
    tokenizer = AutoTokenizer.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    prompt_manifest, prompts = M27.select_prompts(tokenizer)
    assert json.loads(PROMPT_MANIFEST.read_text(encoding="utf-8")) == prompt_manifest
    assert len(prompts) == 24
    with safe_open(str(M20.ADAPTER), framework="pt", device="cpu") as archive:
        meta = archive.metadata()
        assert meta["model_sha256"] == M15.M13.MODEL_SHA
        assert meta["output_factor"] == "0.5"
    adapter = load_file(str(M20.ADAPTER), device="cpu")
    model = AutoModelForCausalLM.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    original_params = dict(model.named_parameters())
    assert len(original_params) == 290
    for p in model.parameters():
        p.requires_grad_(False)
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = M15.ResidualExperts(layer.mlp, li).to(device)
        layer.mlp = wrapper
        for key in ("a", "b", "router"):
            getattr(wrapper, key).copy_(adapter[f"layers.{li}.{key}"].to(device))
        wrapper.enabled = True
        wrappers.append(wrapper)
    assert len(adapter) == 3 * M15.M13.L
    del adapter
    with safe_open(str(M25.CORE), framework="pt", device="cpu") as archive:
        meta = archive.metadata()
        assert meta["format"] == "QWEN25_R8_CORE_V1"
        assert meta["model_sha256"] == M15.M13.MODEL_SHA
        assert meta["adapter_sha256"] == M20.ADAPTER_SHA
        assert meta["tied_head"] == "true"
        assert len(archive.keys()) == 459
        matrix_count = control_count = 0
        for name, param in original_params.items():
            organ = M24.classify(name, tuple(param.shape))
            if organ == "control":
                stored = archive.get_tensor(name)
                assert stored.dtype == torch.float32
                assert torch.equal(stored.to(torch.bfloat16), param.detach().cpu())
                control_count += 1
            else:
                codes = archive.get_tensor(name + ".q")
                scale = archive.get_tensor(name + ".scale")
                assert codes.dtype == torch.int8 and tuple(codes.shape) == tuple(param.shape)
                assert scale.dtype == torch.float32 and tuple(scale.shape) == (param.shape[0],)
                assert bool((codes >= -127).all()) and bool((codes <= 127).all())
                param.copy_((codes.to(device).float() *
                             scale.to(device).unsqueeze(1)).to(torch.bfloat16))
                matrix_count += 1
        assert matrix_count == 169 and control_count == 121
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    check_budget(start, device)
    print("stored R8 core and E128 adapter loaded", flush=True)

    indices = []
    for li, wrapper in enumerate(wrappers):
        groups, group_ids, centers = balanced_groups(wrapper.router.detach().cpu())
        indices.append({"layer": li, "groups": groups,
                        "group_ids": group_ids.to(device),
                        "centers": centers.to(device)})
        if (li+1) % 6 == 0:
            print(f"built {li+1}/24 balanced indices", flush=True)
            check_budget(start, device)

    counts = {arm: {category: [empty_count() for _ in range(M15.M13.L)]
                    for category in M25M.COUNTS}
              for arm in ARMS}
    stream_hashes = {arm: [hashlib.sha256() for _ in range(M15.M13.L)]
                     for arm in ("exact", *ARMS)}
    active_category = [None]

    def hook_for_layer(li):
        index = indices[li]
        router = wrappers[li].router.detach()

        def hook(module, input_tuple):
            x = input_tuple[0].detach().reshape(-1, M15.M13.D).float()
            n = x.shape[0]
            assert n == M27.PROMPT_TOKENS
            scores = F.linear(x, router)
            exact = torch.topk(scores, M15.K, dim=-1)
            exact_ids = exact.indices
            exact_gate = F.softmax(exact.values, dim=-1)
            stream_hashes["exact"][li].update(
                exact_ids.cpu().numpy().astype("<i2", copy=False).tobytes())
            coarse = F.linear(x, index["centers"])
            for arm, nleaves in ARMS.items():
                selected_groups = torch.topk(coarse, nleaves, dim=-1).indices
                candidates = index["group_ids"][selected_groups].reshape(
                    n, nleaves * GROUP_SIZE)
                selected_scores = (router[candidates] * x.unsqueeze(1)).sum(dim=-1)
                approximate = candidates.gather(
                    1, torch.topk(selected_scores, M15.K, dim=-1).indices)
                hits = (exact_ids.unsqueeze(2) == candidates.unsqueeze(1)).any(dim=2)
                matches = (torch.sort(exact_ids, dim=-1).values ==
                           torch.sort(approximate, dim=-1).values).all(dim=-1)
                mass = (exact_gate * (~hits)).sum(dim=-1)
                row = counts[arm][active_category[0]][li]
                row["positions"] += n
                row["exact_ids"] += n * M15.K
                row["included_exact_ids"] += int(hits.sum())
                row["full_set_matches"] += int(matches.sum())
                row["missed_gate_mass_sum"] += float(mass.sum())
                row["worst_missed_gate_mass"] = max(
                    row["worst_missed_gate_mass"], float(mass.max()))
                stream_hashes[arm][li].update(
                    approximate.cpu().numpy().astype("<i2", copy=False).tobytes())
        return hook

    handles = [wrapper.register_forward_pre_hook(hook_for_layer(li))
               for li, wrapper in enumerate(wrappers)]
    with torch.inference_mode():
        for i, prompt in enumerate(prompts):
            active_category[0] = prompt["category"]
            ids = torch.as_tensor(prompt["prompt_ids"], dtype=torch.long,
                                  device=device).unsqueeze(0)
            model(ids, use_cache=False)
            check_budget(start, device)
            if (i+1) % 4 == 0:
                print(f"audited {i+1}/{len(prompts)} real prompts", flush=True)
    for handle in handles:
        handle.remove()
    assert all(sum(row["positions"] for cat in M25M.COUNTS
                   for row in counts[arm][cat]) == 24 * 256 * 24
               for arm in ARMS)

    summaries = {}
    for arm, nleaves in ARMS.items():
        categories = {}
        pooled = empty_count()
        layers = [empty_count() for _ in range(M15.M13.L)]
        for category in M25M.COUNTS:
            aggregate = empty_count()
            for li, row in enumerate(counts[arm][category]):
                add_count(aggregate, row)
                add_count(pooled, row)
                add_count(layers[li], row)
            categories[category] = finalize_count(aggregate)
        pooled = finalize_count(pooled)
        passed = (pooled["exact_id_inclusion_fraction"] >= 0.999
                  and pooled["full_set_match_fraction"] >= 0.99
                  and all(r["exact_id_inclusion_fraction"] >= 0.99
                          for r in categories.values()))
        summaries[arm] = {"selected_leaves": nleaves,
                          "candidate_rows": nleaves * GROUP_SIZE,
                          "coarse_rows": LEAVES,
                          "row_dot_equivalents": LEAVES + nleaves * GROUP_SIZE,
                          "fraction_of_exhaustive_row_dots":
                          (LEAVES + nleaves * GROUP_SIZE) / M15.E,
                          "pooled": pooled, "categories": categories,
                          "layers": [finalize_count(r) for r in layers],
                          "diagnostic_route_gate_pass": passed}
    runtime = check_budget(start, device)
    result = {"experiment": "METH-29", "prompt_manifest_sha256": PROMPT_MANIFEST_SHA,
              "core_sha256": M25.CORE_SHA,
              "adapter_sha256": M20.ADAPTER_SHA,
              "donor_sha256": M15.M13.MODEL_SHA,
              "tokenizer_fingerprint": M15.M13.TOK_FP,
              "group_size": GROUP_SIZE,
              "grouping_rule": "recursive balanced leading-right-singular-direction split; expert ID tie-break",
              "group_assignments": [x["groups"] for x in indices],
              "summary": summaries,
              "route_stream_sha256": {arm: [h.hexdigest() for h in hashes]
                                      for arm, hashes in stream_hashes.items()},
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__},
              "decision": "candidate_recall_pass_test_quality_and_cpu_next"
              if any(x["diagnostic_route_gate_pass"] for x in summaries.values())
              else "posthoc_balanced_grouping_route_gate_fail"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summaries, "decision": result["decision"],
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
