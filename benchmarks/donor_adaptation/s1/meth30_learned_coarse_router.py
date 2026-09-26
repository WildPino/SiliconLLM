#!/usr/bin/env python3
"""METH-30: distill a bounded coarse group gate from the frozen E128 router."""
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
from safetensors.torch import load_file, save_file

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth20_half_adapter_generation as M20
import meth24_export_r8_core as M24
import meth25_fresh_r8_artifact_audit as M25
import meth27_r8_fresh_generation as M27
import meth29_balanced_router_index as M29


ROOT = Path(__file__).resolve().parents[3]
INDEX_RESULT = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth29_balanced_router_index_result.json"
INDEX_RESULT_SHA = "249836830ed41e77b9e2c6036292e71838359221e430eac29295e6bb7944d502"
MANIFEST = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth30_learned_coarse_router_manifest.json"
SEED = "meth30-30030"
TORCH_SEED = 303030
TRAIN_WINDOWS = 32
VALID_WINDOWS = 8
WINDOW_TOKENS = 256
HIDDEN = 64
EPOCHS = 20
BATCH = 512
LR = 0.003
MAX_SECONDS = 25 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)
ARMS = {"candidates32": 4, "candidates64": 8}


def check_budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS:
        raise TimeoutError(f"METH-30 wall-time stop: {elapsed:.1f}s")
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"METH-30 RSS stop: {rss}")
    if peak > MAX_GPU_BYTES:
        raise MemoryError(f"METH-30 GPU stop: {peak}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss,
            "gpu_peak_allocated_bytes": peak}


def select_manifest():
    assert M15.M13.sha256(M15.TRAIN_PATH) == M15.TRAIN_FILE_SHA
    with np.load(M15.TRAIN_PATH, allow_pickle=False) as archive:
        ids = archive["ids"].copy()
    assert ids.shape == (31250, 512) and ids.dtype == np.int32
    assert M15.sha_bytes(ids.tobytes()) == M15.TRAIN_IDS_SHA
    assert M15.M13.sha256(INDEX_RESULT) == INDEX_RESULT_SHA
    prior = json.loads(INDEX_RESULT.read_text(encoding="utf-8"))
    groups = prior["group_assignments"]
    assert len(groups) == M15.M13.L
    assert all(len(g) == M29.LEAVES and all(len(x) == M29.GROUP_SIZE for x in g)
               for g in groups)
    ranked = sorted(range(ids.shape[0]),
                    key=lambda i: hashlib.sha256(f"{SEED}|{i}".encode()).hexdigest())
    chosen = ranked[:TRAIN_WINDOWS + VALID_WINDOWS]
    rows = [{"split": "train" if j < TRAIN_WINDOWS else "internal_validation",
             "source_row": int(i),
             "first256_ids_sha256": M17.sha(ids[i, :WINDOW_TOKENS].tobytes())}
            for j, i in enumerate(chosen)]
    assert len({x["first256_ids_sha256"] for x in rows}) == len(rows)
    manifest = {"experiment": "METH-30", "selection_seed": SEED,
                "train_file_sha256": M15.TRAIN_FILE_SHA,
                "train_ids_sha256": M15.TRAIN_IDS_SHA,
                "index_result_sha256": INDEX_RESULT_SHA,
                "external_prompt_manifest_sha256": M29.PROMPT_MANIFEST_SHA,
                "train_windows": TRAIN_WINDOWS,
                "internal_validation_windows": VALID_WINDOWS,
                "window_tokens": WINDOW_TOKENS, "items": rows}
    return manifest, ids, chosen, groups


def empty_count():
    return M29.empty_count()


def gate_forward(x_norm, w1, b1, w2, b2):
    hidden = F.silu(torch.bmm(x_norm, w1) + b1[:, None, :])
    return torch.bmm(hidden, w2) + b2[:, None, :]


def summarize(counts, categories):
    result = {}
    for arm, nleaves in ARMS.items():
        per_category = {}
        pooled = empty_count()
        per_layer = [empty_count() for _ in range(M15.M13.L)]
        for category in categories:
            cat_total = empty_count()
            for li, row in enumerate(counts[arm][category]):
                M29.add_count(cat_total, row)
                M29.add_count(pooled, row)
                M29.add_count(per_layer[li], row)
            per_category[category] = M29.finalize_count(cat_total)
        pooled = M29.finalize_count(pooled)
        gate = (pooled["exact_id_inclusion_fraction"] >= 0.999
                and pooled["full_set_match_fraction"] >= 0.99
                and all(x["exact_id_inclusion_fraction"] >= 0.99
                        for x in per_category.values()))
        result[arm] = {"selected_leaves": nleaves,
                       "candidate_rows": nleaves * M29.GROUP_SIZE,
                       "pooled": pooled, "categories": per_category,
                       "layer_min_id_inclusion": min(
                           M29.finalize_count(x)["exact_id_inclusion_fraction"]
                           for x in per_layer),
                       "route_gate_pass": gate}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--manifest-sha256")
    ap.add_argument("--out")
    ap.add_argument("--gate-out")
    args = ap.parse_args()
    manifest, train_ids, chosen, group_lists = select_manifest()
    if args.prepare:
        assert args.manifest_sha256 is None and args.out is None and args.gate_out is None
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"manifest_sha256": M17.sha(MANIFEST.read_bytes()),
                          "train_rows": TRAIN_WINDOWS,
                          "validation_rows": VALID_WINDOWS}, indent=2), flush=True)
        return
    assert args.manifest_sha256 and args.out and args.gate_out
    assert M17.sha(MANIFEST.read_bytes()) == args.manifest_sha256
    assert json.loads(MANIFEST.read_text(encoding="utf-8")) == manifest
    assert M15.M13.sha256(M25.CORE) == M25.CORE_SHA
    assert M15.M13.sha256(M20.ADAPTER) == M20.ADAPTER_SHA
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    torch.manual_seed(TORCH_SEED)
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
    external_manifest, prompts = M27.select_prompts(tokenizer)
    assert M17.sha(M29.PROMPT_MANIFEST.read_bytes()) == M29.PROMPT_MANIFEST_SHA
    assert json.loads(M29.PROMPT_MANIFEST.read_text(encoding="utf-8")) == external_manifest
    with safe_open(str(M20.ADAPTER), framework="pt", device="cpu") as archive:
        assert archive.metadata()["model_sha256"] == M15.M13.MODEL_SHA
        assert archive.metadata()["output_factor"] == "0.5"
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
                param.copy_((codes.to(device).float() *
                             scale.to(device).unsqueeze(1)).to(torch.bfloat16))
                matrix_count += 1
        assert matrix_count == 169 and control_count == 121
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    check_budget(start, device)

    groups = torch.as_tensor(group_lists, dtype=torch.long, device=device)
    assert groups.shape == (M15.M13.L, M29.LEAVES, M29.GROUP_SIZE)
    group_of_expert = torch.full((M15.M13.L, M15.E), -1,
                                 dtype=torch.long, device=device)
    for li in range(M15.M13.L):
        for gi in range(M29.LEAVES):
            group_of_expert[li, groups[li, gi]] = gi
    assert bool((group_of_expert >= 0).all())
    captured = [[] for _ in range(M15.M13.L)]
    labels = [[] for _ in range(M15.M13.L)]

    def capture_hook(li):
        router = wrappers[li].router.detach()
        inv = group_of_expert[li]

        def hook(module, input_tuple):
            x = input_tuple[0].detach().reshape(-1, M15.M13.D).float()
            assert x.shape[0] == WINDOW_TOKENS
            top = torch.topk(F.linear(x, router), M15.K, dim=-1).indices
            marked = torch.zeros((WINDOW_TOKENS, M29.LEAVES),
                                 dtype=torch.float32, device=device)
            marked.scatter_(1, inv[top], 1.0)
            captured[li].append(x.to(torch.float16).cpu())
            labels[li].append(marked.to(torch.uint8).cpu())
        return hook

    handles = [w.register_forward_pre_hook(capture_hook(li))
               for li, w in enumerate(wrappers)]
    with torch.inference_mode():
        for j, source_row in enumerate(chosen):
            ids = torch.as_tensor(train_ids[source_row, :WINDOW_TOKENS],
                                  dtype=torch.long, device=device).unsqueeze(0)
            model(ids, use_cache=False)
            check_budget(start, device)
            if (j+1) % 8 == 0:
                print(f"captured {j+1}/{len(chosen)} source windows", flush=True)
    for handle in handles:
        handle.remove()
    assert all(len(x) == len(chosen) for x in captured)
    assert all(len(x) == len(chosen) for x in labels)
    x_cpu = torch.stack([torch.cat(x, dim=0) for x in captured])
    y_cpu = torch.stack([torch.cat(x, dim=0) for x in labels])
    del captured, labels
    assert x_cpu.shape == (M15.M13.L, len(chosen) * WINDOW_TOKENS, M15.M13.D)
    assert y_cpu.shape == (M15.M13.L, len(chosen) * WINDOW_TOKENS, M29.LEAVES)
    train_n = TRAIN_WINDOWS * WINDOW_TOKENS
    val_n = VALID_WINDOWS * WINDOW_TOKENS
    x_train = x_cpu[:, :train_n].to(device=device, dtype=torch.float32)
    y_train = y_cpu[:, :train_n].to(device=device, dtype=torch.float32)
    x_val = x_cpu[:, train_n:].to(device=device, dtype=torch.float32)
    del x_cpu, y_cpu
    mean = x_train.mean(dim=1, keepdim=True)
    rms = ((x_train - mean).square().mean(dim=(1, 2), keepdim=True)
           .sqrt().clamp_min(1e-6))
    x_train = (x_train - mean) / rms
    x_val_norm = (x_val - mean) / rms
    assert torch.isfinite(x_train).all() and torch.isfinite(x_val_norm).all()
    check_budget(start, device)

    L, D, C = M15.M13.L, M15.M13.D, M29.LEAVES
    w1 = torch.nn.Parameter(torch.randn((L, D, HIDDEN), device=device) * 0.02)
    b1 = torch.nn.Parameter(torch.zeros((L, HIDDEN), device=device))
    w2 = torch.nn.Parameter(torch.randn((L, HIDDEN, C), device=device) * 0.02)
    b2 = torch.nn.Parameter(torch.zeros((L, C), device=device))
    optimizer = torch.optim.AdamW([w1, b1, w2, b2], lr=LR, weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(TORCH_SEED)
    train_losses = []
    torch.set_grad_enabled(True)
    for epoch in range(EPOCHS):
        order = torch.randperm(train_n, generator=generator)
        total = 0.0
        for begin in range(0, train_n, BATCH):
            selected = order[begin:begin+BATCH].to(device)
            xb = x_train[:, selected]
            yb = y_train[:, selected]
            predicted = gate_forward(xb, w1, b1, w2, b2)
            loss = F.binary_cross_entropy_with_logits(predicted, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * selected.numel()
            check_budget(start, device)
        train_losses.append(total / train_n)
        if (epoch+1) % 5 == 0:
            print(f"trained epoch {epoch+1}/{EPOCHS} loss={train_losses[-1]:.6f}",
                  flush=True)
    torch.set_grad_enabled(False)
    model.eval()

    def make_counts(categories):
        return {arm: {cat: [empty_count() for _ in range(L)] for cat in categories}
                for arm in ARMS}

    def score_inputs(li, x, logits, category, counts):
        router = wrappers[li].router.detach()
        n = x.shape[0]
        exact = torch.topk(F.linear(x, router), M15.K, dim=-1)
        exact_ids = exact.indices
        exact_gate = F.softmax(exact.values, dim=-1)
        for arm, nleaves in ARMS.items():
            selected_groups = torch.topk(logits, nleaves, dim=-1).indices
            candidates = groups[li, selected_groups].reshape(
                n, nleaves * M29.GROUP_SIZE)
            scores = (router[candidates] * x.unsqueeze(1)).sum(dim=-1)
            approximate = candidates.gather(
                1, torch.topk(scores, M15.K, dim=-1).indices)
            hits = (exact_ids.unsqueeze(2) == candidates.unsqueeze(1)).any(dim=2)
            matches = (torch.sort(exact_ids, dim=-1).values ==
                       torch.sort(approximate, dim=-1).values).all(dim=-1)
            mass = (exact_gate * (~hits)).sum(dim=-1)
            row = counts[arm][category][li]
            row["positions"] += n
            row["exact_ids"] += n * M15.K
            row["included_exact_ids"] += int(hits.sum())
            row["full_set_matches"] += int(matches.sum())
            row["missed_gate_mass_sum"] += float(mass.sum())
            row["worst_missed_gate_mass"] = max(
                row["worst_missed_gate_mass"], float(mass.max()))

    internal_counts = make_counts(("internal_validation",))
    with torch.inference_mode():
        for begin in range(0, val_n, BATCH):
            xv = x_val[:, begin:begin+BATCH]
            norm = x_val_norm[:, begin:begin+BATCH]
            logits = gate_forward(norm, w1, b1, w2, b2)
            for li in range(L):
                score_inputs(li, xv[li], logits[li], "internal_validation",
                             internal_counts)
            check_budget(start, device)
    internal_summary = summarize(internal_counts, ("internal_validation",))
    print(json.dumps({"internal_validation": {
        arm: {"id_inclusion": x["pooled"]["exact_id_inclusion_fraction"],
              "full_set_match": x["pooled"]["full_set_match_fraction"]}
        for arm, x in internal_summary.items()}}, indent=2), flush=True)

    external_counts = make_counts(("code", "prose", "technical_general"))
    active_category = [None]

    def external_hook(li):
        def hook(module, input_tuple):
            x = input_tuple[0].detach().reshape(-1, D).float()
            assert x.shape[0] == WINDOW_TOKENS
            norm = (x - mean[li, 0]) / rms[li, 0, 0]
            hidden = F.silu(norm @ w1[li] + b1[li])
            logits = hidden @ w2[li] + b2[li]
            score_inputs(li, x, logits, active_category[0], external_counts)
        return hook

    handles = [w.register_forward_pre_hook(external_hook(li))
               for li, w in enumerate(wrappers)]
    with torch.inference_mode():
        for j, prompt in enumerate(prompts):
            active_category[0] = prompt["category"]
            ids = torch.as_tensor(prompt["prompt_ids"], dtype=torch.long,
                                  device=device).unsqueeze(0)
            model(ids, use_cache=False)
            check_budget(start, device)
            if (j+1) % 4 == 0:
                print(f"audited {j+1}/{len(prompts)} external prompts", flush=True)
    for handle in handles:
        handle.remove()
    external_summary = summarize(external_counts,
                                 ("code", "prose", "technical_general"))

    tensors = {"w1": w1.detach().cpu().contiguous(),
               "b1": b1.detach().cpu().contiguous(),
               "w2": w2.detach().cpu().contiguous(),
               "b2": b2.detach().cpu().contiguous(),
               "input_mean": mean[:, 0].detach().cpu().contiguous(),
               "input_rms": rms[:, 0, 0].detach().cpu().contiguous()}
    metadata = {"experiment": "METH-30", "core_sha256": M25.CORE_SHA,
                "adapter_sha256": M20.ADAPTER_SHA,
                "index_result_sha256": INDEX_RESULT_SHA,
                "manifest_sha256": args.manifest_sha256,
                "architecture": "per-layer D896-SiLU64-C16", "epochs": str(EPOCHS)}
    gate_out = Path(args.gate_out)
    gate_out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(gate_out), metadata=metadata)
    reloaded = load_file(str(gate_out), device="cpu")
    assert reloaded.keys() == tensors.keys()
    assert all(torch.equal(reloaded[k], v) for k, v in tensors.items())
    with safe_open(str(gate_out), framework="pt", device="cpu") as archive:
        assert archive.metadata() == metadata
    runtime = check_budget(start, device)
    result = {"experiment": "METH-30", "manifest_sha256": args.manifest_sha256,
              "core_sha256": M25.CORE_SHA, "adapter_sha256": M20.ADAPTER_SHA,
              "index_result_sha256": INDEX_RESULT_SHA,
              "coarse_gate_sha256": M15.M13.sha256(gate_out),
              "coarse_gate_bytes": gate_out.stat().st_size,
              "training": {"seed": TORCH_SEED, "epochs": EPOCHS,
                           "batch_positions": BATCH, "learning_rate": LR,
                           "train_positions": train_n,
                           "internal_validation_positions": val_n,
                           "loss_per_epoch": train_losses},
              "internal_validation": internal_summary,
              "external_prompt_audit": external_summary,
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__},
              "decision": "coarse_route_diagnostic_pass_fresh_quality_and_cpu_next"
              if any(x["route_gate_pass"] for x in external_summary.values())
              else "fixed_learned_coarse_route_gate_fail"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"external_prompt_audit": external_summary,
                      "decision": result["decision"],
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
