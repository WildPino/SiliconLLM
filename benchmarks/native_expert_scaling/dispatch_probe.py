#!/usr/bin/env python3
"""Bounded E128 sparse-dispatch equivalence and training-step feasibility probe.

This is an apparatus check, not an expert-count quality result.
Run: python benchmarks/native_expert_scaling/dispatch_probe.py --device cuda:0
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "phase57"))
from phase59_moe import MoEMLP, SparseMoEMLP, build_model, moe_forward  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equivalence(device, experts):
    torch.manual_seed(19)
    dense = MoEMLP(256, 128, experts, 8, 0.01, 0.0, device.type).to(device)
    sparse = SparseMoEMLP(256, 128, experts, 8, 0.01, 0.0, device.type).to(device)
    sparse.load_state_dict(dense.state_dict())
    x0 = torch.randn(2, 16, 256, device=device)
    outputs = {}
    for name, model in (("compute_all", dense), ("sparse", sparse)):
        x = x0.detach().clone().requires_grad_(True)
        out, aux = model(x)
        (out.square().mean() + aux).backward()
        outputs[name] = (out.detach(), x.grad.detach(),
                         {key: p.grad.detach() for key, p in model.named_parameters()})
    a, b = outputs.values()
    grad_rel = {}
    for key in a[2]:
        scale = max(a[2][key].abs().max().item(), 1e-8)
        grad_rel[key] = (a[2][key] - b[2][key]).abs().max().item() / scale
    return {
        "max_abs_output": (a[0] - b[0]).abs().max().item(),
        "max_abs_input_gradient": (a[1] - b[1]).abs().max().item(),
        "max_relative_parameter_gradient": max(grad_rel.values()),
        "worst_parameter": max(grad_rel, key=grad_rel.get),
        "pass_2e_4": (a[0] - b[0]).abs().max().item() < 2e-4
        and max(grad_rel.values()) < 2e-4,
    }


def training_step(device, experts, steps, batch, sequence):
    torch.manual_seed(19)
    arch = dict(D=256, N=96, H=8, L=6, swa_layer=5, use_mlp=True,
                mlp_mult=4, dt_rank=16)
    model, _ = build_model(1024, arch, "moe-gran", 0.01, 0.0,
                           device, device.type, experts=experts, sparse_moe=True)
    model.train()
    ids = torch.randint(0, 1024, (batch, sequence), device=device)
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    elapsed = []
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, aux = moe_forward(model, ids, use_ckpt=True)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 1024).float(), ids.reshape(-1)) + aux
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed.append(time.perf_counter() - t0)
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "batch": batch, "sequence": sequence, "steps": steps,
        "seconds_per_step": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda" else None,
        "final_loss": float(loss.detach()),
        "all_gradients_finite": all(p.grad is None or torch.isfinite(p.grad).all().item()
                                    for p in model.parameters()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--sequence", type=int, default=128)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    device = torch.device(args.device)
    result = {
        "scope": "apparatus_only",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "phase59_sha256": digest(ROOT / "benchmarks" / "phase57" / "phase59_moe.py"),
        "mve_model_sha256": digest(ROOT / "benchmarks" / "phase64" / "mve" / "mve_model.py"),
        "experts": args.experts, "topk": 8,
        "equivalence": equivalence(device, args.experts),
        "training_step": training_step(device, args.experts, args.steps,
                                       args.batch, args.sequence),
    }
    rendered = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not result["equivalence"]["pass_2e_4"] or not result["training_step"]["all_gradients_finite"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
