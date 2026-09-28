#!/usr/bin/env python3
"""Bound a CPU-master sparse-gradient path for a third E12800 expert tier."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
BANK = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth126_shared_a_factor_bank.bin"
BANK_SHA = "1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1"
VECTORS = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth125_e1280_vectors.bin"
VECTORS_SHA = "f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699"
HEADER = struct.Struct("<8s8I")
VHEADER = struct.Struct("<8s4I")
LAYERS, WIDTH, RANK, NA, NB, CHILD_RANK, CHILDREN, FACTOR_RANK = (
    24, 896, 64, 8, 16, 32, 10, 8)
E1280 = NA * NB * CHILDREN
E12800 = E1280 * 10
SEED = 134134
MAX_SECONDS = 600
MAX_RSS = 4 * (1 << 30)
MAX_GPU = 4 * (1 << 30)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bf16_to_f32(bits):
    return (np.asarray(bits, dtype="<u2").astype("<u4") << 16).view("<f4")


def budget(start, device):
    result = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (result["seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS
            or result["gpu_peak_allocated_bytes"] > MAX_GPU):
        raise RuntimeError(f"METH-134 resource stop: {result}")
    return result


class SparseCollector:
    def __init__(self):
        self.parts = []

    def add(self, ids, gradients):
        self.parts.append((ids.clone(), gradients.contiguous()))

    def materialize(self):
        assert self.parts
        ids = torch.cat([part[0] for part in self.parts])
        gradients = torch.cat([part[1] for part in self.parts])
        unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
        summed = torch.zeros((unique.numel(), *gradients.shape[1:]),
                             dtype=gradients.dtype, device="cpu")
        summed.index_add_(0, inverse, gradients)
        return unique, summed, int(gradients.numel() * gradients.element_size())


class SparseCpuGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, trigger, ids, bank, collector):
        assert bank.device.type == "cpu" and bank.dtype == torch.float32
        flat = ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        ctx.ids = flat
        ctx.collector = collector
        ctx.trigger_shape = tuple(trigger.shape)
        gathered = bank.index_select(0, flat).to(device=trigger.device)
        return gathered.reshape(*ids.shape, *bank.shape[1:])

    @staticmethod
    def backward(ctx, gradient):
        ctx.collector.add(ctx.ids,
                          gradient.detach().reshape(ctx.ids.numel(),
                                                    *gradient.shape[-2:]).to("cpu"))
        return torch.zeros(ctx.trigger_shape, device=gradient.device,
                           dtype=gradient.dtype), None, None, None


def small_gradient_oracle(device):
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    reference = torch.randn((64, 16, 8), generator=generator)
    ids = torch.randint(0, 64, (11, 4), generator=generator)
    ids[0, 1] = ids[0, 0]
    ids[1, 0] = ids[0, 0]
    hidden = torch.randn((11, 4, 8), generator=generator).to(device)
    dense = reference.to(device).detach().requires_grad_(True)
    dense_out = torch.einsum("nkr,nkdr->nd", hidden, dense[ids.to(device)])
    dense_loss = dense_out.square().mean()
    dense_loss.backward()
    collector = SparseCollector()
    trigger = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
    sparse = reference.clone()
    gathered = SparseCpuGather.apply(trigger, ids.to(device), sparse, collector)
    sparse_out = torch.einsum("nkr,nkdr->nd", hidden, gathered)
    sparse_loss = sparse_out.square().mean()
    sparse_loss.backward()
    unique, gradient, raw_gradient_bytes = collector.materialize()
    dense_gradient = dense.grad.detach().cpu().index_select(0, unique)
    gradient_max_abs = float((dense_gradient - gradient).abs().max())
    output_max_abs = float((dense_out - sparse_out).abs().max())
    learning_rate = 0.01
    with torch.no_grad():
        dense.add_(dense.grad, alpha=-learning_rate)
        sparse.index_add_(0, unique, -learning_rate * gradient)
    updated_max_abs = float((dense.detach().cpu() - sparse).abs().max())
    other = torch.ones(64, dtype=torch.bool)
    other[unique] = False
    untouched_max_abs = float((sparse[other] - reference[other]).abs().max())
    assert ids[0, 0] == ids[0, 1] == ids[1, 0]
    assert output_max_abs <= 1e-6 and gradient_max_abs <= 1e-6
    assert updated_max_abs <= 1e-6 and untouched_max_abs == 0
    return {"rows": 64, "selected_occurrences": int(ids.numel()),
            "unique_selected_rows": int(unique.numel()),
            "duplicate_occurrences": int(ids.numel() - unique.numel()),
            "output_max_abs_error": output_max_abs,
            "selected_gradient_max_abs_error": gradient_max_abs,
            "one_step_bank_max_abs_error": updated_max_abs,
            "unselected_change_max_abs": untouched_max_abs,
            "raw_sparse_gradient_bytes": raw_gradient_bytes}


def read_bound_layer():
    assert digest(BANK) == BANK_SHA and digest(VECTORS) == VECTORS_SHA
    bank = np.memmap(BANK, dtype=np.uint8, mode="r")
    assert HEADER.unpack_from(bank) == (b"M126FB01", LAYERS, WIDTH, RANK,
                                        NA, NB, CHILD_RANK, CHILDREN,
                                        FACTOR_RANK)
    router_floats = RANK * WIDTH + (NA + NB) * RANK + CHILD_RANK * WIDTH + E1280 * CHILD_RANK
    router_bytes = router_floats * 4
    a_bytes = NA * NB * FACTOR_RANK * WIDTH * 2
    b_bytes = E1280 * WIDTH * FACTOR_RANK * 2
    assert bank.size == HEADER.size + LAYERS * (router_bytes + a_bytes + b_bytes)
    cursor = HEADER.size

    def fp32(count, shape):
        nonlocal cursor
        array = np.frombuffer(bank, dtype="<f4", count=count,
                              offset=cursor).copy().reshape(shape)
        cursor += count * 4
        return torch.from_numpy(array)

    parent_projection = fp32(RANK * WIDTH, (RANK, WIDTH))
    axis_a = fp32(NA * RANK, (NA, RANK))
    axis_b = fp32(NB * RANK, (NB, RANK))
    child_projection = fp32(CHILD_RANK * WIDTH, (CHILD_RANK, WIDTH))
    child_keys = fp32(E1280 * CHILD_RANK,
                      (NA * NB, CHILDREN, CHILD_RANK))
    assert cursor == HEADER.size + router_bytes
    a_bits = np.frombuffer(bank, dtype="<u2", count=a_bytes // 2,
                           offset=cursor).reshape(NA * NB, FACTOR_RANK, WIDTH)
    shared_a = torch.from_numpy(bf16_to_f32(a_bits).copy())
    cursor += a_bytes
    b_bits = np.frombuffer(bank, dtype="<u2", count=b_bytes // 2,
                           offset=cursor).reshape(E1280, WIDTH, FACTOR_RANK)
    source_b = torch.from_numpy(bf16_to_f32(b_bits).copy())
    vector_data = np.memmap(VECTORS, dtype=np.uint8, mode="r")
    assert VHEADER.unpack_from(vector_data) == (b"M125HX01", LAYERS, 256,
                                               WIDTH, E1280)
    assert vector_data.size == VHEADER.size + 256 * LAYERS * WIDTH * 2
    vectors = np.frombuffer(vector_data, dtype="<u2", count=256 * LAYERS * WIDTH,
                            offset=VHEADER.size).reshape(256, LAYERS, WIDTH)
    states = torch.from_numpy(bf16_to_f32(vectors[:, 0, :]).copy())
    return (parent_projection, axis_a, axis_b, child_projection,
            child_keys, shared_a, source_b, states)


def route_bound_states(x, matrices, device):
    parent_projection, axis_a, axis_b, child_projection, child_keys = (
        matrix.to(device) for matrix in matrices)
    q = F.linear(x, parent_projection)
    av, ai = F.linear(q, axis_a).topk(4, dim=-1)
    bv, bi = F.linear(q, axis_b).topk(4, dim=-1)
    candidate_scores = (av[:, :, None] + bv[:, None, :]).reshape(-1, 16)
    candidate_ids = (ai[:, :, None] * NB + bi[:, None, :]).reshape(-1, 16)
    selected_scores, where = candidate_scores.topk(4, dim=-1)
    parents = candidate_ids.gather(1, where)
    child_q = F.linear(x, child_projection)
    keys = child_keys[parents]
    child_scores = torch.einsum("nr,nkcr->nkc", child_q, keys)
    children = parents * CHILDREN + child_scores.argmax(dim=-1)
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    grand_projection = (torch.randn((CHILD_RANK, WIDTH), generator=generator)
                        * (0.1 / math.sqrt(WIDTH))).to(device)
    grand_keys = (torch.randn((E1280, 10, CHILD_RANK), generator=generator)
                  * (0.1 / math.sqrt(CHILD_RANK))).to(device)
    grand_q = F.linear(x, grand_projection)
    chosen_keys = grand_keys[children]
    grand_scores = torch.einsum("nr,nkcr->nkc", grand_q, chosen_keys)
    grandchildren = children * 10 + grand_scores.argmax(dim=-1)
    assert torch.equal(grandchildren // 10, children)
    return parents, children, grandchildren, F.softmax(selected_scores, dim=-1)


def check_unselected(bank, source, selected):
    chosen = np.zeros(E12800, dtype=np.bool_)
    chosen[selected.numpy()] = True
    array = bank.numpy()
    original = source.numpy()
    unchanged = 0
    changed_selected = 0
    for start in range(0, E12800, 128):
        end = min(start + 128, E12800)
        expected = original[np.arange(start, end) // 10]
        rows_changed = np.any(array[start:end] != expected, axis=(1, 2))
        assert not np.any(rows_changed & ~chosen[start:end])
        unchanged += int(np.count_nonzero(~rows_changed & ~chosen[start:end]))
        changed_selected += int(np.count_nonzero(rows_changed & chosen[start:end]))
    assert unchanged == E12800 - int(selected.numel())
    assert changed_selected > 0
    return changed_selected, unchanged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    start = time.monotonic()
    torch.set_num_threads(6)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    oracle = small_gradient_oracle(device)
    budget(start, device)
    (parent_projection, axis_a, axis_b, child_projection, child_keys,
     shared_a, source_b, states) = read_bound_layer()
    assert source_b.shape == (E1280, WIDTH, FACTOR_RANK)
    bank_array = np.repeat(source_b.numpy(), 10, axis=0)
    assert bank_array.nbytes == E12800 * WIDTH * FACTOR_RANK * 4
    bank = torch.from_numpy(bank_array)
    assert torch.equal(bank[::10], source_b) and torch.equal(bank[9::10], source_b)
    x = states.to(device)
    parents, children, grandchildren, gates = route_bound_states(
        x, (parent_projection, axis_a, axis_b,
            child_projection, child_keys), device)
    a = shared_a.to(device)[parents]
    hidden = F.silu(torch.einsum("nd,nkrd->nkr", x, a))
    exact = source_b.to(device)[children]
    collector = SparseCollector()
    trigger = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
    selected = SparseCpuGather.apply(trigger, grandchildren, bank, collector)
    assert torch.equal(exact, selected)
    exact_output = (torch.einsum("nkr,nkdr->nkd", hidden, exact)
                    * gates[:, :, None]).sum(dim=1)
    expanded_output = (torch.einsum("nkr,nkdr->nkd", hidden, selected)
                       * gates[:, :, None]).sum(dim=1)
    output_max_abs = float((exact_output - expanded_output).abs().max())
    assert output_max_abs == 0
    budget(start, device)
    loss = expanded_output.square().mean()
    loss.backward()
    unique, gradient, raw_gradient_bytes = collector.materialize()
    grad_norm = float(torch.linalg.vector_norm(gradient))
    assert torch.isfinite(gradient).all() and math.isfinite(grad_norm) and grad_norm > 0
    learning_rate = 0.01
    with torch.no_grad():
        bank.index_add_(0, unique, -learning_rate * gradient)
    assert torch.isfinite(bank).all()
    changed_selected, unchanged_unselected = check_unselected(bank, source_b, unique)
    runtime = budget(start, device)
    result = {"experiment": "METH-134-CPU-master-sparse-E12800-apparatus",
              "source_bank_sha256": BANK_SHA, "source_vectors_sha256": VECTORS_SHA,
              "layer": 0, "states": int(x.shape[0]), "seed": SEED,
              "dimensions": {"source_children": E1280, "grandchildren": E12800,
                             "width": WIDTH, "factor_rank": FACTOR_RANK,
                             "active_experts": 4},
              "source_b_bytes_fp32": int(source_b.numel() * source_b.element_size()),
              "expanded_b_cpu_bytes_fp32": int(bank.numel() * bank.element_size()),
              "full_24_layer_b_cpu_bytes_projected": int(bank.numel() * bank.element_size() * LAYERS),
              "full_24_layer_b_bf16_bytes_projected": int(bank.numel() * 2 * LAYERS),
              "small_dense_sparse_oracle": oracle,
              "route": {"selected_occurrences": int(children.numel()),
                        "unique_source_children": int(torch.unique(children).numel()),
                        "unique_grandchildren": int(torch.unique(grandchildren).numel()),
                        "grandchild_parent_identity": True,
                        "untrained_third_tier": True},
              "clone_output_max_abs_error": output_max_abs,
              "one_step": {"loss": float(loss.detach()),
                           "gradient_l2": grad_norm,
                           "unique_gradient_rows": int(unique.numel()),
                           "raw_gradient_transfer_bytes": raw_gradient_bytes,
                           "summed_unique_gradient_bytes": int(gradient.numel() * gradient.element_size()),
                           "forward_selected_B_transfer_bytes": int(selected.numel() * selected.element_size()),
                           "changed_selected_rows": changed_selected,
                           "unchanged_unselected_rows": unchanged_unselected,
                           "full_dense_gpu_bank_allocated": False},
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "torch": torch.__version__, "numpy": np.__version__},
              "decision": "apparatus_pass_training_and_quality_not_tested"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
