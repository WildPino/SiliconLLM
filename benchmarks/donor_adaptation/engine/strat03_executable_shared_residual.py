#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STRAT-03 -- CPU-only executable shared-residual diagnostic.

Frozen preregistration:
docs/research/donor_adaptation/briefs/BRIEF_STRAT_03_EXECUTABLE_SHARED_RESIDUAL.md

This runner is deliberately local to five donor FFN layers.  It makes no
claim about BPB, generation, throughput, cache residency, engine code, or a
10B model.  Donor loading is cache-only: it will fail rather than download a
weight or tokenizer.

    python strat03_executable_shared_residual.py --selftest
    python strat03_executable_shared_residual.py --smoke
    python strat03_executable_shared_residual.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HERE = os.path.dirname(os.path.abspath(__file__))
DENSITY = os.path.abspath(os.path.join(HERE, "..", "density"))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if DENSITY not in sys.path:
    sys.path.insert(0, DENSITY)

import e68_shared_residual as E68


HF = "Qwen/Qwen2.5-1.5B"
REV = "8faed761d45a263340a0528343f099c05c9a4323"
LAYERS = (1, 7, 14, 21, 27)
D_MODEL = 1536
E_GROUPS = 256
GROUP_NEURONS = 35
K = 3
FFN_DIM = E_GROUPS * GROUP_NEURONS
LINEAR_RANK = 96
NONLINEAR_WIDTH = 64
THREADS = 6
TOKEN_CHUNK = 32
RECON_REL_TOL = 1e-5
ANCHOR_REL_TOL = 1e-5
CALIB_N_SEQ, CALIB_SEQLEN, CALIB_SEED = 8, 512, 424242
CALIB_IDS_SHA = "92fc41840c8feb0595758e540a048c1661bf3f618b3355d637ecf418b6c00157"
HELDOUT_N_SEQ, HELDOUT_SEQLEN, HELDOUT_SEED = 24, 512, 1234
HELDOUT_IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
HELDOUT_SELECTED_IDS_SHA = "6e4160a27d10288f24d6a60fded1b0537c97fd28df8798d2a65dac235124a4b6"
HELDOUT_START, HELDOUT_STOP = 384, 512
TRAIN_SEED = 20260916
TRAIN_STEPS = 300
TRAIN_BATCH = 256
TRAIN_LR = 1e-3
LOSS_STEPS = (0, 50, 100, 200, 300)

BRIEF_REL = "docs/research/donor_adaptation/briefs/BRIEF_STRAT_03_EXECUTABLE_SHARED_RESIDUAL.md"
BRIEF = os.path.join(REPO, *BRIEF_REL.split("/"))
RUNNER_REL = "benchmarks/donor_adaptation/engine/strat03_executable_shared_residual.py"
BRIEF_COMMIT = "4f4ee4b"
BRIEF_SHA = "386c3bf121c8d0a5243561a8f14405d07d858fa21baa8aa8d610125d85cd2fff"
E68_RUNNER = os.path.join(HERE, "e68_shared_residual.py")
E68_RUNNER_SHA = "0fd10804742ce856f626314a2a251a8982f4a9ff6d63da2d48f79ca815df867e"
E68_RESULT = os.path.join(HERE, "results", "e68_shared_residual.json")
E68_RESULT_SHA = "02e5ae986ad3f59241d78602378bb7e17575c3bbf38f78db13244442da91be47"
LABELS_PATH = os.path.join(DENSITY, "results", "d0c_labels", "labels_E256.npz")
LABELS_SHA = "c39d0740b7f754daf168d811c77120c5394ae677e3882fef208d331d8c333c9c"
LABEL_LOADER = os.path.join(DENSITY, "e19_carve_rank.py")
LABEL_LOADER_SHA = "31c595efd5ccc12bf37922e54a38939a69ece96566adcde257340656d75e9525"
COMMON = os.path.join(DENSITY, "common.py")
COMMON_SHA = "7ac00b31e91d0018c775e2e9b26120c0ec025075d7913218ea3cdcfe3196e461"
RESULTS = os.path.join(HERE, "results")
MASS_ANCHORS = {
    1: 26876.563427, 7: 154032.840704, 14: 113665.947309,
    21: 446756.691130, 27: 7247181.889910,
}
ARMS = ("MASS", "O-L96", "O-N64", "X-L96", "X-N64")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: str) -> str:
    with open(path, "rb") as handle:
        return _sha_bytes(handle.read())


def _sha_tensor(tensor: torch.Tensor) -> str:
    return _sha_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError("not JSON serialisable: %r" % (type(value),))


def _write_json_once(path: str, payload: Mapping[str, object]) -> None:
    """Write atomically enough for this single-file protocol; never overwrite evidence."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise RuntimeError("result already exists; STRAT-03 refuses to overwrite: %s" % path) from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=_json_default)
            handle.write("\n")
    except Exception:
        # A partial evidence file must remain visible rather than being silently replaced.
        raise


def _stable_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    return E68._stable_topk(scores, k)


def _multi_hot(selected: torch.Tensor, n_groups: int = E_GROUPS) -> torch.Tensor:
    if selected.ndim != 2 or selected.shape[1] != K:
        raise RuntimeError("router target must contain exactly three oracle labels")
    target = torch.zeros(selected.shape[0], n_groups, dtype=torch.float32, device=selected.device)
    target.scatter_(1, selected.long(), 1.0)
    if not torch.equal(target.sum(dim=1), torch.full((selected.shape[0],), float(K), device=selected.device)):
        raise RuntimeError("router BCE target is not three-hot")
    return target


def _relative_error(reference: torch.Tensor, actual: torch.Tensor) -> float:
    denom = reference.double().square().sum(dim=1).sqrt()
    if torch.any(denom <= 0):
        raise RuntimeError("zero-norm reference prevents selective reconstruction control")
    return float(((reference.double() - actual.double()).square().sum(dim=1).sqrt() / denom).max().item())


class _RSS:
    """Operational observation only, explicitly not a model-rate measurement."""

    def __init__(self):
        self.peak = 0
        self.method = "unavailable"
        try:
            import psutil
            self.process = psutil.Process(os.getpid())
            self.method = "psutil.memory_info.peak_wset (or rss fallback)"
        except ImportError:
            self.process = None

    def observe(self) -> None:
        if self.process is not None:
            info = self.process.memory_info()
            self.peak = max(self.peak, int(getattr(info, "peak_wset", info.rss)), int(info.rss))


class Router(nn.Module):
    """The only X-router interface: selection accepts x (and x) alone, never dense z."""

    def __init__(self, d_model: int = D_MODEL, groups: int = E_GROUPS):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_model, groups))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, normalized_x: torch.Tensor) -> torch.Tensor:
        return normalized_x @ self.weight

    def select_x_only(self, normalized_x: torch.Tensor) -> torch.Tensor:
        """Deliberately has no z/y/weight argument: guard against a hidden dense-z router."""
        return _stable_topk(self.forward(normalized_x), K)


class SharedSwiGLU(nn.Module):
    """Bias-free D -> 64 SwiGLU -> D shared residual map."""

    def __init__(self, d_model: int = D_MODEL, width: int = NONLINEAR_WIDTH):
        super().__init__()
        self.gate = nn.Parameter(torch.empty(width, d_model))
        self.up = nn.Parameter(torch.empty(width, d_model))
        self.down = nn.Parameter(torch.empty(d_model, width))
        nn.init.kaiming_uniform_(self.gate, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.up, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.down, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.silu(F.linear(x, self.gate)) * F.linear(x, self.up), self.down)


@dataclass
class _Metrics:
    n_seq: int
    arms: Sequence[str] = ARMS

    def __post_init__(self):
        self.sse = {arm: 0.0 for arm in self.arms}
        self.seq_sse = {arm: np.zeros(self.n_seq, dtype=np.float64) for arm in self.arms}
        self.token_ratios = {arm: [] for arm in self.arms}
        self.energy = 0.0
        self.seq_energy = np.zeros(self.n_seq, dtype=np.float64)
        self.tokens = 0

    def add(self, arm: str, y: torch.Tensor, yhat: torch.Tensor, sequence: int, reference: bool = False) -> None:
        if arm not in self.sse or not (0 <= sequence < self.n_seq):
            raise RuntimeError("invalid metric arm or sequence")
        err = (y.double() - yhat.double()).square().sum(dim=1)
        energy = y.double().square().sum(dim=1)
        if torch.any(energy <= 0):
            raise RuntimeError("zero-norm FFN output prevents SSE/energy metrics")
        err_np, energy_np = err.cpu().numpy(), energy.cpu().numpy()
        self.sse[arm] += float(err_np.sum())
        self.seq_sse[arm][sequence] += float(err_np.sum())
        self.token_ratios[arm].extend((err_np / (energy_np + 1e-12)).tolist())
        if reference:
            self.energy += float(energy_np.sum())
            self.seq_energy[sequence] += float(energy_np.sum())
            self.tokens += int(y.shape[0])

    def result(self, expected_tokens: int) -> Dict[str, object]:
        if self.tokens != expected_tokens or self.energy <= 0 or np.any(self.seq_energy <= 0):
            raise RuntimeError("incomplete or zero-energy metric accounting")
        return {
            arm: {
                "sum_sse": self.sse[arm],
                "sum_sse_over_sum_energy": self.sse[arm] / self.energy,
                "median_token_sse_over_energy": float(np.median(self.token_ratios[arm])),
                "p95_token_sse_over_energy": float(np.percentile(self.token_ratios[arm], 95)),
                "per_sequence_sum_sse_over_sum_energy": (self.seq_sse[arm] / self.seq_energy).tolist(),
            }
            for arm in self.arms
        }


class _RouterMetrics:
    def __init__(self):
        self.tokens = 0
        self.intersection = 0
        self.exact = 0
        self.oracle_counts = Counter()
        self.router_counts = Counter()

    def add(self, oracle: torch.Tensor, routed: torch.Tensor) -> None:
        if oracle.shape != routed.shape or oracle.ndim != 2 or oracle.shape[1] != K:
            raise RuntimeError("router/oracle selections must both be [tokens,3]")
        for oracle_row, routed_row in zip(oracle.cpu().tolist(), routed.cpu().tolist()):
            oracle_set, routed_set = set(oracle_row), set(routed_row)
            if len(oracle_set) != K or len(routed_set) != K:
                raise RuntimeError("selector returned duplicate groups")
            self.tokens += 1
            self.intersection += len(oracle_set & routed_set)
            self.exact += int(oracle_set == routed_set)
            self.oracle_counts.update(oracle_row)
            self.router_counts.update(routed_row)

    def result(self) -> Dict[str, object]:
        if self.tokens <= 0:
            raise RuntimeError("no router metrics recorded")
        return {
            "tokens": self.tokens,
            "top3_recall": self.intersection / float(self.tokens * K),
            "exact_set_match": self.exact / float(self.tokens),
            "oracle_group_frequency": [self.oracle_counts[group] for group in range(E_GROUPS)],
            "router_group_frequency": [self.router_counts[group] for group in range(E_GROUPS)],
        }


class _SelectiveFFN:
    """Computes only the gate/up/down rows named by an already-made top-3 selection."""

    def __init__(self, mlp: torch.nn.Module, labels: torch.Tensor, d_model: int = D_MODEL,
                 groups: int = E_GROUPS, group_neurons: int = GROUP_NEURONS):
        if labels.ndim != 1 or labels.numel() != groups * group_neurons:
            raise RuntimeError("labels have wrong dimension for selective FFN")
        counts = torch.bincount(labels.long(), minlength=groups)
        if counts.numel() != groups or not torch.equal(counts, torch.full_like(counts, group_neurons)):
            raise RuntimeError("selective FFN labels are not equal-sized groups")
        self.d_model, self.groups, self.group_neurons = d_model, groups, group_neurons
        self.gate = mlp.gate_proj.weight.detach()
        self.up = mlp.up_proj.weight.detach()
        self.down = mlp.down_proj.weight.detach()
        if self.gate.shape != (groups * group_neurons, d_model) or self.up.shape != self.gate.shape:
            raise RuntimeError("unexpected donor gate/up dimensions")
        if self.down.shape != (d_model, groups * group_neurons):
            raise RuntimeError("unexpected donor down dimension")
        labels = labels.long().to(self.gate.device)
        self.indices = [(labels == group).nonzero(as_tuple=False).squeeze(1) for group in range(groups)]
        if any(index.numel() != group_neurons for index in self.indices):
            raise RuntimeError("non-equal group map prevents selective FFN calculation")

    def output(self, x: torch.Tensor, selected: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
        if x.ndim != 2 or x.shape[1] != self.d_model or selected.shape != (x.shape[0], K):
            raise RuntimeError("bad x/selection shape for selective FFN")
        if selected.device != x.device:
            selected = selected.to(x.device)
        output = torch.zeros(x.shape[0], self.d_model, dtype=x.dtype, device=x.device)
        groups_evaluated = 0
        for slot in range(K):
            groups = selected[:, slot]
            for group in torch.unique(groups, sorted=True).tolist():
                mask = groups == group
                index = self.indices[int(group)]
                x_group = x[mask]
                z_group = F.silu(F.linear(x_group, self.gate[index])) * F.linear(x_group, self.up[index])
                output[mask] += F.linear(z_group, self.down[:, index])
                groups_evaluated += int(mask.sum().item())
        return output, {"selected_group_evaluations": groups_evaluated,
                        "gate_up_down_rows_per_token": K * self.group_neurons}


class _FitCapture:
    """Calibration-only capture.  Held-out values can never reach its lists or fit methods."""

    def __init__(self, model: torch.nn.Module, layer: int, labels: torch.Tensor, rss: _RSS):
        self.layer, self.rss, self.sequence = layer, rss, None
        self.pending_x = None
        self.x_parts: List[torch.Tensor] = []
        self.residual_parts: List[torch.Tensor] = []
        self.y_parts: List[torch.Tensor] = []
        self.oracle_parts: List[torch.Tensor] = []
        self.selective_max = 0.0
        mlp = model.model.layers[layer].mlp
        self.reference = E68._MassProjector(mlp.down_proj.weight, labels)
        self.selective = _SelectiveFFN(mlp, labels)
        self.handles = [mlp.register_forward_pre_hook(self._mlp_hook),
                        mlp.down_proj.register_forward_pre_hook(self._down_hook)]

    def _mlp_hook(self, _module, args):
        if self.sequence is not None:
            x = args[0].detach()
            if x.shape != (1, CALIB_SEQLEN, D_MODEL):
                raise RuntimeError("bad calibration MLP input at layer %d" % self.layer)
            self.pending_x = x[0].cpu().float().contiguous()

    def _down_hook(self, _module, args):
        if self.sequence is None:
            return
        if self.pending_x is None:
            raise RuntimeError("calibration down hook without x at layer %d" % self.layer)
        z_all = args[0].detach()
        if z_all.shape != (1, CALIB_SEQLEN, FFN_DIM):
            raise RuntimeError("bad calibration down input at layer %d" % self.layer)
        for start in range(0, CALIB_SEQLEN, TOKEN_CHUNK):
            stop = min(start + TOKEN_CHUNK, CALIB_SEQLEN)
            x = self.pending_x[start:stop].to(z_all.device)
            y, oracle_dense, oracle, _ = self.reference.sparse_output(z_all[0, start:stop])
            oracle_selective, _ = self.selective.output(x, oracle)
            self.selective_max = max(self.selective_max, _relative_error(oracle_dense, oracle_selective))
            self.x_parts.append(x.cpu().contiguous())
            self.residual_parts.append((y - oracle_dense).cpu().float().contiguous())
            self.y_parts.append(y.cpu().float().contiguous())
            self.oracle_parts.append(oracle.cpu().long().contiguous())
        self.pending_x = None
        self.rss.observe()

    def detach(self) -> None:
        while self.handles:
            self.handles.pop().remove()

    def materialize(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
        if self.pending_x is not None:
            raise RuntimeError("unconsumed calibration x at layer %d" % self.layer)
        expected = CALIB_N_SEQ * CALIB_SEQLEN
        x, residual, y, oracle = (torch.cat(parts, dim=0) for parts in
                                  (self.x_parts, self.residual_parts, self.y_parts, self.oracle_parts))
        if x.shape != (expected, D_MODEL) or residual.shape != x.shape or y.shape != x.shape or oracle.shape != (expected, K):
            raise RuntimeError("calibration capture shape mismatch at layer %d" % self.layer)
        if self.selective_max >= RECON_REL_TOL:
            raise RuntimeError("calibration selective-vs-dense control failed at layer %d: %.3e" %
                               (self.layer, self.selective_max))
        self.x_parts.clear(); self.residual_parts.clear(); self.y_parts.clear(); self.oracle_parts.clear()
        return x, residual, y, oracle, {
            "fit_partition": "calib only", "fit_tokens": expected,
            "selective_vs_dense_max_relative_rms": self.selective_max,
            "selective_vs_dense_relative_tolerance": RECON_REL_TOL,
            "selective_vs_dense_fires": True,
        }


def _remove_captures(captures: Mapping[int, _FitCapture]) -> None:
    for capture in captures.values():
        capture.detach()


def _load_labels(n_layers: int, selected_layers: Iterable[int]) -> Tuple[Dict[int, torch.Tensor], Dict[str, object]]:
    import e19_carve_rank as e19

    if _sha_file(LABEL_LOADER) != LABEL_LOADER_SHA:
        raise RuntimeError("label loader hash mismatch -- STOP")
    if _sha_file(LABELS_PATH) != LABELS_SHA:
        raise RuntimeError("labels hash mismatch -- STOP")
    learned_np, _ = e19.load_labels(E_GROUPS)
    if set(learned_np) != set(range(n_layers)):
        raise RuntimeError("labels do not name exactly every donor layer")
    labels, per_layer = {}, {}
    for layer in selected_layers:
        value = torch.from_numpy(np.asarray(learned_np[layer], dtype=np.int64)).contiguous()
        E68._validate_labels(value)
        expected = E68.LEARNED_LABEL_SHA[layer]
        if _sha_tensor(value) != expected:
            raise RuntimeError("learned label hash mismatch at layer %d -- STOP" % layer)
        labels[layer] = value
        per_layer[str(layer)] = expected
    return labels, {"path": LABELS_PATH, "labels_E256_npz_sha256": LABELS_SHA,
                    "learned_per_layer": per_layer, "E": E_GROUPS,
                    "neurons_per_group": GROUP_NEURONS}


def _git_brief_sha() -> str:
    result = subprocess.run(["git", "show", "%s:%s" % (BRIEF_COMMIT, BRIEF_REL)], cwd=REPO,
                            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return _sha_bytes(result.stdout)


def _git_runner_sha() -> str:
    result = subprocess.run(["git", "show", "HEAD:%s" % RUNNER_REL], cwd=REPO,
                            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return _sha_bytes(result.stdout)


def _preflight(smoke: bool) -> Tuple[str, Dict[str, object]]:
    out_path = os.path.join(RESULTS, "strat03_executable_shared_residual%s.json" % ("_smoke" if smoke else ""))
    if os.path.exists(out_path):
        raise RuntimeError("result already exists; STRAT-03 refuses to overwrite: %s" % out_path)
    checks = {
        "brief_worktree_sha256": _sha_file(BRIEF),
        "brief_git_4f4ee4b_sha256": _git_brief_sha(),
        "e68_runner_sha256": _sha_file(E68_RUNNER),
        "e68_full_result_sha256": _sha_file(E68_RESULT),
        "labels_sha256": _sha_file(LABELS_PATH),
        "density_common_sha256": _sha_file(COMMON),
        "label_loader_sha256": _sha_file(LABEL_LOADER), "runner_sha256": _sha_file(__file__),
        "runner_head_blob_sha256": _git_runner_sha(),
    }
    expected = {
        "brief_worktree_sha256": BRIEF_SHA, "brief_git_4f4ee4b_sha256": BRIEF_SHA,
        "e68_runner_sha256": E68_RUNNER_SHA, "e68_full_result_sha256": E68_RESULT_SHA,
        "labels_sha256": LABELS_SHA, "density_common_sha256": COMMON_SHA,
        "label_loader_sha256": LABEL_LOADER_SHA,
    }
    bad = {key: {"actual": checks[key], "expected": expected[key]} for key in expected if checks[key] != expected[key]}
    if bad:
        raise RuntimeError("frozen provenance hash mismatch -- STOP: %s" % bad)
    if checks["runner_sha256"] != checks["runner_head_blob_sha256"]:
        raise RuntimeError("runner worktree differs from committed HEAD blob -- STOP")
    e68 = json.load(open(E68_RESULT, encoding="utf-8"))
    saved_rank96 = all("96" in e68["layers"][str(layer)]["fit"]["factors"]
                       ["residual_target_y_minus_y_sparse"] for layer in LAYERS)
    return out_path, {"expected": expected, "observed": checks,
                      "e68_rank96_anchor": {"saved": saved_rank96,
                          "action": "compare if saved" if saved_rank96 else
                                    "no anchor invented; pinned E68 result has no saved rank-96 factor"},
                      "result_write_once": True}


def _configure_determinism(out_path: str | None = None, provenance: Mapping[str, object] | None = None) -> Dict[str, object]:
    try:
        torch.set_num_threads(THREADS)
        torch.use_deterministic_algorithms(True)
    except Exception as exc:
        if out_path is not None:
            _write_json_once(out_path, {"brief": "BRIEF_STRAT_03_EXECUTABLE_SHARED_RESIDUAL.md",
                                        "apparatus_status": "VOID_APPARATUS",
                                        "reason": "deterministic CPU configuration failed",
                                        "error": "%s: %s" % (type(exc).__name__, exc),
                                        "provenance": provenance or {}, "smoke": out_path.endswith("_smoke.json")})
        raise RuntimeError("VOID_APPARATUS: deterministic CPU configuration failed: %s" % exc) from exc
    return {"device": "cpu", "torch_num_threads": THREADS, "use_deterministic_algorithms": True,
            "train_seed": TRAIN_SEED, "optimizer": "AdamW(weight_decay=0)",
            "minibatch_order": "calibration rows in capture order; batch (update-1) mod 16"}


def _load_donor_cache_only() -> Tuple[torch.nn.Module, object]:
    """No cache miss may turn into a Hugging Face network request."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF, revision=REV, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(HF, revision=REV, dtype=torch.float32,
                                                  attn_implementation="eager", local_files_only=True)
    model.eval()
    if next(model.parameters()).device.type != "cpu":
        raise RuntimeError("CPU-only protocol loaded a non-CPU donor")
    return model, tokenizer


def _train_router(x: torch.Tensor, oracle: torch.Tensor) -> Tuple[Router, torch.Tensor, Dict[str, object]]:
    scale = x.square().mean(dim=0).sqrt().clamp_min(1e-6).float()
    normalized = x.float() / scale
    target = _multi_hot(oracle)
    with torch.enable_grad():
        torch.manual_seed(TRAIN_SEED)
        router = Router().cpu()
        optimizer = torch.optim.AdamW(router.parameters(), lr=TRAIN_LR, weight_decay=0.0)
        positive = torch.full((E_GROUPS,), float(E_GROUPS) / K)
        curves = {}
        for update in range(TRAIN_STEPS + 1):
            if update in LOSS_STEPS:
                with torch.no_grad():
                    curves[str(update)] = float(F.binary_cross_entropy_with_logits(
                        router(normalized), target, pos_weight=positive, reduction="mean").item())
            if update == TRAIN_STEPS:
                break
            begin = (update % (x.shape[0] // TRAIN_BATCH)) * TRAIN_BATCH
            end = begin + TRAIN_BATCH
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(router(normalized[begin:end]), target[begin:end],
                                                     pos_weight=positive, reduction="mean")
            loss.backward()
            if not all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                       for parameter in router.parameters()):
                raise RuntimeError("router gradient was null or non-finite")
            optimizer.step()
    return router, scale, {"loss": "BCEWithLogits(mean all elements; positive weight E/k)",
                           "positive_weight": float(E_GROUPS) / K, "loss_curve": curves,
                           "updates_applied": TRAIN_STEPS, "learning_rate": TRAIN_LR,
                           "weight_decay": 0.0, "normalization_scale_shape": [D_MODEL],
                           "normalization_scale_dtype": "float32"}


def _train_swiglu(x: torch.Tensor, residual: torch.Tensor) -> Tuple[SharedSwiGLU, Dict[str, object]]:
    with torch.enable_grad():
        torch.manual_seed(TRAIN_SEED)
        shared = SharedSwiGLU().cpu()
        optimizer = torch.optim.AdamW(shared.parameters(), lr=TRAIN_LR, weight_decay=0.0)
        curves = {}
        for update in range(TRAIN_STEPS + 1):
            if update in LOSS_STEPS:
                with torch.no_grad():
                    curves[str(update)] = float(F.mse_loss(shared(x.float()), residual.float(), reduction="mean").item())
            if update == TRAIN_STEPS:
                break
            begin = (update % (x.shape[0] // TRAIN_BATCH)) * TRAIN_BATCH
            end = begin + TRAIN_BATCH
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(shared(x[begin:end].float()), residual[begin:end].float(), reduction="mean")
            loss.backward()
            if not all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                       for parameter in shared.parameters()):
                raise RuntimeError("SwiGLU gradient was null or non-finite")
            optimizer.step()
    return shared, {"loss": "mean MSE per residual element", "loss_curve": curves,
                    "updates_applied": TRAIN_STEPS, "learning_rate": TRAIN_LR, "weight_decay": 0.0,
                    "width": NONLINEAR_WIDTH, "bias": False}


def _calibration_metrics(x: torch.Tensor, residual: torch.Tensor, y: torch.Tensor, oracle: torch.Tensor,
                         selective: _SelectiveFFN, linear: E68.FactorPair, nonlinear: SharedSwiGLU,
                         router: Router, scale: torch.Tensor) -> Tuple[Dict[str, object], Dict[str, object]]:
    metrics, routing = _Metrics(CALIB_N_SEQ), _RouterMetrics()
    oracle_sparse = y - residual
    a, vt = linear.score_factors_fp32()
    with torch.no_grad():
        for sequence in range(CALIB_N_SEQ):
            start, stop = sequence * CALIB_SEQLEN, (sequence + 1) * CALIB_SEQLEN
            xs, ys, sparse, target = x[start:stop], y[start:stop], oracle_sparse[start:stop], oracle[start:stop]
            routed = router.select_x_only(xs / scale)
            routed_sparse, _ = selective.output(xs, routed)
            linear_shared = (xs @ a) @ vt
            nonlinear_shared = nonlinear(xs)
            metrics.add("MASS", ys, sparse, sequence, reference=True)
            metrics.add("O-L96", ys, sparse + linear_shared, sequence)
            metrics.add("O-N64", ys, sparse + nonlinear_shared, sequence)
            metrics.add("X-L96", ys, routed_sparse + linear_shared, sequence)
            metrics.add("X-N64", ys, routed_sparse + nonlinear_shared, sequence)
            routing.add(target, routed)
    return metrics.result(CALIB_N_SEQ * CALIB_SEQLEN), routing.result()


class _ScoreCapture:
    """Held-out scorer: the router consumes normalized x only; z remains oracle/control-only."""

    def __init__(self, model: torch.nn.Module, layer: int, labels: torch.Tensor, n_seq: int,
                 linear: E68.FactorPair, nonlinear: SharedSwiGLU, router: Router, scale: torch.Tensor,
                 rss: _RSS):
        self.layer, self.n_seq, self.rss = layer, n_seq, rss
        self.linear, self.nonlinear, self.router, self.scale = linear, nonlinear, router, scale
        self.sequence, self.pending_x, self.selective_max = None, None, 0.0
        self.metrics, self.routing = _Metrics(n_seq), _RouterMetrics()
        mlp = model.model.layers[layer].mlp
        self.reference = E68._MassProjector(mlp.down_proj.weight, labels)
        self.selective = _SelectiveFFN(mlp, labels)
        self.handles = [mlp.register_forward_pre_hook(self._mlp_hook),
                        mlp.down_proj.register_forward_pre_hook(self._down_hook)]

    def _mlp_hook(self, _module, args):
        if self.sequence is not None:
            x = args[0].detach()
            if x.shape != (1, HELDOUT_SEQLEN, D_MODEL):
                raise RuntimeError("bad held-out MLP input at layer %d" % self.layer)
            self.pending_x = x[0, HELDOUT_START:HELDOUT_STOP].cpu().float().contiguous()

    def _down_hook(self, _module, args):
        if self.sequence is None:
            return
        if self.pending_x is None:
            raise RuntimeError("held-out down hook without x at layer %d" % self.layer)
        z_all = args[0].detach()
        if z_all.shape != (1, HELDOUT_SEQLEN, FFN_DIM):
            raise RuntimeError("bad held-out down input at layer %d" % self.layer)
        z_all = z_all[0, HELDOUT_START:HELDOUT_STOP]
        a, vt = self.linear.score_factors_fp32()
        with torch.no_grad():
            for start in range(0, z_all.shape[0], TOKEN_CHUNK):
                stop = min(start + TOKEN_CHUNK, z_all.shape[0])
                x = self.pending_x[start:stop].to(z_all.device)
                y, oracle_dense, oracle, _ = self.reference.sparse_output(z_all[start:stop])
                oracle_sparse, oracle_rows = self.selective.output(x, oracle)
                self.selective_max = max(self.selective_max, _relative_error(oracle_dense, oracle_sparse))
                # This selection call has no path to z_all, y, down weights, or oracle labels.
                routed = self.router.select_x_only(x / self.scale.to(x.device))
                routed_sparse, routed_rows = self.selective.output(x, routed)
                if oracle_rows["gate_up_down_rows_per_token"] != K * GROUP_NEURONS or \
                        routed_rows["gate_up_down_rows_per_token"] != K * GROUP_NEURONS:
                    raise RuntimeError("selective FFN did not charge exactly 105 rows")
                linear_shared = (x @ a.to(x.device)) @ vt.to(x.device)
                nonlinear_shared = self.nonlinear(x)
                self.metrics.add("MASS", y, oracle_sparse, self.sequence, reference=True)
                self.metrics.add("O-L96", y, oracle_sparse + linear_shared, self.sequence)
                self.metrics.add("O-N64", y, oracle_sparse + nonlinear_shared, self.sequence)
                self.metrics.add("X-L96", y, routed_sparse + linear_shared, self.sequence)
                self.metrics.add("X-N64", y, routed_sparse + nonlinear_shared, self.sequence)
                self.routing.add(oracle, routed)
        self.pending_x = None
        self.rss.observe()

    def close(self) -> Tuple[Dict[str, object], Dict[str, object]]:
        while self.handles:
            self.handles.pop().remove()
        if self.pending_x is not None:
            raise RuntimeError("unconsumed held-out x at layer %d" % self.layer)
        if self.selective_max >= RECON_REL_TOL:
            raise RuntimeError("held-out selective-vs-dense control failed at layer %d: %.3e" %
                               (self.layer, self.selective_max))
        return self.metrics.result(self.n_seq * (HELDOUT_STOP - HELDOUT_START)), {
            "router": self.routing.result(),
            "selective_vs_dense_max_relative_rms": self.selective_max,
            "selective_vs_dense_relative_tolerance": RECON_REL_TOL,
            "selective_vs_dense_fires": True,
            "selection": "X arms call Router.select_x_only(normalized_x); no dense z is supplied to it",
        }


def _score(model: torch.nn.Module, ids: torch.Tensor, labels: Mapping[int, torch.Tensor],
           fit: Mapping[int, Mapping[str, object]], layers: Sequence[int], smoke: bool, rss: _RSS) -> Dict[str, object]:
    n_seq = 1 if smoke else 4
    captures = {}
    try:
        for layer in layers:
            row = fit[layer]
            captures[layer] = _ScoreCapture(model, layer, labels[layer], n_seq, row["linear"], row["nonlinear"],
                                            row["router"], row["scale"], rss)
        with torch.inference_mode():
            for sequence in range(n_seq):
                for capture in captures.values():
                    capture.sequence = sequence
                model.model(ids[sequence:sequence + 1])
                print("STRAT-03 held-out sequence %d/%d scored" % (sequence + 1, n_seq), flush=True)
                rss.observe()
    finally:
        for capture in captures.values():
            capture.sequence = None
    return {str(layer): dict(zip(("heldout", "heldout_controls"), captures[layer].close())) for layer in layers}


def _aggregate(layers: Mapping[str, Mapping[str, object]], arm: str) -> float:
    value = sum(float(row["heldout"][arm]["sum_sse"]) for row in layers.values())
    if not np.isfinite(value) or value <= 0:
        raise RuntimeError("invalid aggregate SSE for %s" % arm)
    return value


def _gates(layers: Mapping[str, Mapping[str, object]], smoke: bool) -> Dict[str, object]:
    if smoke:
        return {"name": "NOT_APPLICABLE_SMOKE",
                "reason": "all three preregistered gates require five layers and four held-out sequences"}
    totals = {arm: _aggregate(layers, arm) for arm in ARMS}
    per = {layer: {arm: float(row["heldout"][arm]["sum_sse"]) for arm in ARMS}
           for layer, row in layers.items()}
    nonlinear_ratio = totals["O-N64"] / totals["O-L96"]
    nonlinear_layer_not_worse = {layer: values["O-N64"] <= values["O-L96"] for layer, values in per.items()}
    nonlinear_layer_cap = {layer: values["O-N64"] <= 1.05 * values["O-L96"] for layer, values in per.items()}
    router_ratio = totals["X-N64"] / totals["O-N64"]
    router_layer_cap = {layer: values["X-N64"] <= 1.50 * values["O-N64"] for layer, values in per.items()}
    recalls = [row["heldout_controls"]["router"]["top3_recall"] * row["heldout_controls"]["router"]["tokens"]
               for row in layers.values()]
    router_tokens = sum(row["heldout_controls"]["router"]["tokens"] for row in layers.values())
    mean_recall = sum(recalls) / router_tokens
    executable_ratio = totals["X-N64"] / totals["MASS"]
    executable_layer_80 = {layer: values["X-N64"] / values["MASS"] <= 0.80 for layer, values in per.items()}
    executable_layer_105 = {layer: values["X-N64"] / values["MASS"] <= 1.05 for layer, values in per.items()}
    executable_p95 = {layer: row["heldout"]["X-N64"]["p95_token_sse_over_energy"] <=
                       row["heldout"]["MASS"]["p95_token_sse_over_energy"] for layer, row in layers.items()}
    nonlinear = {"aggregate_ratio_O_N64_over_O_L96": nonlinear_ratio,
                 "aggregate_at_most_0_80": nonlinear_ratio <= 0.80,
                 "per_layer_O_N64_not_above_O_L96": nonlinear_layer_not_worse,
                 "layers_not_above_count": sum(nonlinear_layer_not_worse.values()),
                 "at_least_four_layers_not_above": sum(nonlinear_layer_not_worse.values()) >= 4,
                 "per_layer_O_N64_at_most_1_05_O_L96": nonlinear_layer_cap,
                 "no_layer_above_1_05": all(nonlinear_layer_cap.values())}
    nonlinear["fires"] = all((nonlinear["aggregate_at_most_0_80"], nonlinear["at_least_four_layers_not_above"],
                               nonlinear["no_layer_above_1_05"]))
    transfer = {"aggregate_ratio_X_N64_over_O_N64": router_ratio,
                "aggregate_at_most_1_20": router_ratio <= 1.20,
                "per_layer_X_N64_at_most_1_50_O_N64": router_layer_cap,
                "no_layer_above_1_50": all(router_layer_cap.values()),
                "mean_top3_recall": mean_recall, "mean_top3_recall_at_least_0_50": mean_recall >= 0.50}
    transfer["fires"] = all((transfer["aggregate_at_most_1_20"], transfer["no_layer_above_1_50"],
                             transfer["mean_top3_recall_at_least_0_50"]))
    executable = {"aggregate_ratio_X_N64_over_MASS": executable_ratio,
                  "aggregate_at_most_0_50": executable_ratio <= 0.50,
                  "per_layer_X_N64_over_MASS_at_most_0_80": executable_layer_80,
                  "layers_at_most_0_80_count": sum(executable_layer_80.values()),
                  "at_least_four_layers_at_most_0_80": sum(executable_layer_80.values()) >= 4,
                  "per_layer_X_N64_over_MASS_at_most_1_05": executable_layer_105,
                  "no_layer_above_1_05": all(executable_layer_105.values()),
                  "per_layer_p95_X_N64_not_above_MASS": executable_p95,
                  "p95_not_above_MASS_count": sum(executable_p95.values()),
                  "p95_not_above_MASS_in_at_least_four_layers": sum(executable_p95.values()) >= 4}
    executable["fires"] = all((executable["aggregate_at_most_0_50"], executable["at_least_four_layers_at_most_0_80"],
                                executable["no_layer_above_1_05"], executable["p95_not_above_MASS_in_at_least_four_layers"]))
    return {"aggregate_sse": totals, "NONLINEAR_BENEFIT": nonlinear,
            "ROUTER_TRANSFER": transfer, "EXECUTABLE_LOCAL_SIGNAL": executable}


def _selftest() -> int:
    """No model/cache access: protocol controls for ties, targets, gradients, count, and selective rows."""
    torch.manual_seed(3003)
    tied = torch.tensor([[4.0, 4.0, 1.0]], dtype=torch.float64)
    assert _stable_topk(tied, 2).tolist() == [[0, 1]]
    labels = torch.tensor([[0, 2, 3], [1, 2, 0]], dtype=torch.long)
    target = _multi_hot(labels, 4)
    assert target.tolist() == [[1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]]
    with torch.enable_grad():
        logits = torch.zeros_like(target, requires_grad=True)
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.full((4,), 1.5))
        loss.backward()
        assert logits.grad is not None and bool(torch.isfinite(logits.grad).all()) and bool((logits.grad != 0).any())
    assert 2 * D_MODEL * LINEAR_RANK == 3 * D_MODEL * NONLINEAR_WIDTH == 294912
    assert 3 * D_MODEL * K * GROUP_NEURONS == 483840
    assert 3 * D_MODEL * K * GROUP_NEURONS + 294912 + D_MODEL * E_GROUPS == 1171968
    with torch.enable_grad():
        router = Router(4, 4)
        shared = SharedSwiGLU(4, 2)
        x = torch.randn(5, 4)
        tiny_target = torch.tensor([[0, 1, 2]] * 5)
        router_loss = F.binary_cross_entropy_with_logits(router(x), _multi_hot(tiny_target, 4),
                                                          pos_weight=torch.ones(4))
        router_loss.backward()
        shared_loss = F.mse_loss(shared(x), torch.randn(5, 4))
        shared_loss.backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool((p.grad != 0).any())
                   for p in list(router.parameters()) + list(shared.parameters()))
    # Small planted selective reconstruction: the selective function sees x and selected groups only.
    class _Tiny: pass
    tiny = _Tiny(); tiny.gate_proj = nn.Linear(4, 6, bias=False); tiny.up_proj = nn.Linear(4, 6, bias=False)
    tiny.down_proj = nn.Linear(6, 4, bias=False)
    toy_labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    selective = _SelectiveFFN(tiny, toy_labels, d_model=4, groups=3, group_neurons=2)
    x = torch.randn(3, 4)
    z = F.silu(tiny.gate_proj(x)) * tiny.up_proj(x)
    selected = torch.tensor([[0, 2, 1], [1, 0, 2], [2, 1, 0]])
    sparse, rows = selective.output(x, selected)
    dense = torch.zeros_like(sparse)
    for row, groups in enumerate(selected.tolist()):
        indices = torch.cat([selective.indices[group] for group in groups])
        dense[row] = F.linear(z[row:row + 1, indices], tiny.down_proj.weight[:, indices]).squeeze(0)
    assert _relative_error(dense, sparse) < 1e-6 and rows["gate_up_down_rows_per_token"] == 3 * 2
    print("STRAT-03 selftest: PASS")
    print("  stable lower-label ties, BCE multi-label targets, finite non-null gradients")
    print("  294,912 shared parameters in both arms; selective-only rows reproduce dense selected output")
    return 0


def run(smoke: bool = False) -> str:
    out_path, provenance = _preflight(smoke)
    determinism = _configure_determinism(out_path, provenance)
    _selftest()
    import common as c

    started, rss = time.time(), _RSS()
    rss.observe()
    model, tokenizer = _load_donor_cache_only()
    architecture = c.arch(model)
    if (architecture["revision"] != REV or architecture["d_model"] != D_MODEL or
            architecture["d_ffn"] != FFN_DIM or architecture["n_layers"] <= max(LAYERS)):
        raise RuntimeError("pinned donor architecture mismatch: %s" % architecture)
    calib_ids, _, calib_meta = c.get_slice(tokenizer, "calib", CALIB_N_SEQ, CALIB_SEQLEN, CALIB_SEED)
    heldout_ids, _, heldout_meta = c.get_slice(tokenizer, "heldout", HELDOUT_N_SEQ, HELDOUT_SEQLEN, HELDOUT_SEED)
    if calib_meta["ids_sha256"] != CALIB_IDS_SHA or heldout_meta["ids_sha256"] != HELDOUT_IDS_SHA:
        raise RuntimeError("frozen calibration or held-out slice hash mismatch -- STOP")
    if calib_meta["part"] == heldout_meta["part"] or CALIB_IDS_SHA == HELDOUT_IDS_SHA:
        raise RuntimeError("fit and score partitions are not disjoint -- STOP")
    layers = (27,) if smoke else LAYERS
    heldout_n_seq = 1 if smoke else 4
    selected_ids = heldout_ids[:heldout_n_seq, HELDOUT_START:HELDOUT_STOP].contiguous()
    expected_selected = ("271647d218110b267fd4d01a41f87fca42da239757c94fd5d80ccc9ad6302ba9"
                         if smoke else HELDOUT_SELECTED_IDS_SHA)
    if _sha_tensor(selected_ids) != expected_selected:
        raise RuntimeError("selected E67/E68 held-out token IDs hash mismatch -- STOP")
    labels, labels_meta = _load_labels(architecture["n_layers"], layers)

    captures = {layer: _FitCapture(model, layer, labels[layer], rss) for layer in layers}
    try:
        with torch.inference_mode():
            for sequence in range(CALIB_N_SEQ):
                for capture in captures.values():
                    capture.sequence = sequence
                model.model(calib_ids[sequence:sequence + 1])
                print("STRAT-03 calibration sequence %d/%d captured" % (sequence + 1, CALIB_N_SEQ), flush=True)
    finally:
        for capture in captures.values():
            capture.sequence = None
        _remove_captures(captures)

    fit: Dict[int, Mapping[str, object]] = {}
    layers_out: Dict[str, Dict[str, object]] = {}
    for layer in layers:
        x, residual, y, oracle, capture_meta = captures[layer].materialize()
        basis = E68._ridge_basis(x.double())
        factors, ridge_meta = E68._fit_reduced_rank_path(basis, residual.double(), ranks=(LINEAR_RANK,))
        linear = factors[LINEAR_RANK]
        router, scale, router_meta = _train_router(x, oracle)
        nonlinear, nonlinear_meta = _train_swiglu(x, residual)
        if sum(parameter.numel() for parameter in nonlinear.parameters()) != 2 * D_MODEL * LINEAR_RANK:
            raise RuntimeError("shared parameter-count equality control failed")
        calibration, calibration_router = _calibration_metrics(x, residual, y, oracle,
                                                                captures[layer].selective, linear, nonlinear, router, scale)
        fit[layer] = {"linear": linear, "nonlinear": nonlinear, "router": router, "scale": scale}
        layers_out[str(layer)] = {"fit": {"capture": capture_meta,
                                            "linear": {"rank": LINEAR_RANK, "ridge": ridge_meta,
                                                       "factor": linear.metadata()},
                                            "nonlinear": nonlinear_meta, "router": router_meta,
                                            "shared_parameter_count": {"linear": 2 * D_MODEL * LINEAR_RANK,
                                                                       "swiglu": sum(p.numel() for p in nonlinear.parameters()),
                                                                       "equal": True},
                                            "fit_target": "R = y - y_sparse_oracle; router selection does not refit shared"},
                                   "calibration": calibration, "calibration_router": calibration_router}
        rss.observe()

    heldout = _score(model, heldout_ids, labels, fit, layers, smoke, rss)
    for layer in layers:
        row = heldout[str(layer)]
        mass = float(row["heldout"]["MASS"]["sum_sse"])
        anchor = MASS_ANCHORS[layer]
        relative = abs(mass - anchor) / max(abs(anchor), 1.0)
        if not smoke and relative > ANCHOR_REL_TOL:
            raise RuntimeError("MASS E68 anchor mismatch at layer %d: %.17g vs %.17g" % (layer, mass, anchor))
        layers_out[str(layer)].update(row)
        layers_out[str(layer)]["mass_e68_anchor"] = {"expected_sum_sse": anchor, "observed_sum_sse": mass,
                                                       "relative_difference": relative,
                                                       "relative_tolerance": ANCHOR_REL_TOL,
                                                       "fires": (relative <= ANCHOR_REL_TOL) if not smoke else None}
    rss.observe()
    output = {
        "brief": "BRIEF_STRAT_03_EXECUTABLE_SHARED_RESIDUAL.md", "brief_commit": BRIEF_COMMIT,
        "scope": "CPU-only local FFN residual diagnostic; no donor full run, T4, BPB, generation, engine, speed, or rate claim",
        "smoke": smoke, "planted_selftests": "PASS", "apparatus_status": "VALID",
        "donor": {"model": HF, "revision": REV, "precision": "fp32", "attention": "eager",
                  "cache_only_offline": True},
        "determinism": determinism, "architecture": architecture, "provenance": provenance,
        "slices": {"calibration": {"part": "calib", "n_sequences": CALIB_N_SEQ, "sequence_length": CALIB_SEQLEN,
                                      "seed": CALIB_SEED, "ids_sha256": calib_meta["ids_sha256"],
                                      "expected_ids_sha256": CALIB_IDS_SHA, "positions": [0, CALIB_SEQLEN - 1]},
                   "heldout": {"part": "heldout", "full_n_sequences": HELDOUT_N_SEQ, "sequence_length": HELDOUT_SEQLEN,
                                "seed": HELDOUT_SEED, "full_ids_sha256": heldout_meta["ids_sha256"],
                                "expected_full_ids_sha256": HELDOUT_IDS_SHA,
                                "selected_sequences": list(range(heldout_n_seq)),
                                "selected_positions": [HELDOUT_START, HELDOUT_STOP - 1],
                                "selected_ids_sha256": _sha_tensor(selected_ids),
                                "expected_selected_ids_sha256": expected_selected},
                   "disjoint_fit_and_heldout": True, "heldout_used_for_fit": False},
        "labels": labels_meta,
        "protocol": {"layers": list(layers), "groups": E_GROUPS, "k": K, "selected_neurons_per_token": K * GROUP_NEURONS,
                     "arms": {"MASS": "zero shared + dense-z mass oracle", "O-L96": "rank-96 shared + mass oracle",
                              "O-N64": "width-64 shared SwiGLU + mass oracle", "X-L96": "same rank-96 + x-only router",
                              "X-N64": "same width-64 + x-only router"},
                     "fit_score_split": "all fits use calibration only; score is the fixed E67/E68 held-out positions",
                     "minimum_weight_accounting_per_layer": {"selected_ffn": 3 * D_MODEL * K * GROUP_NEURONS,
                                                                "shared": 294912, "router": D_MODEL * E_GROUPS,
                                                                "rms_scales": D_MODEL, "total_large_organs": 1171968,
                                                                "dense_ffn": 3 * D_MODEL * FFN_DIM,
                                                                "hypothetical_0_5_byte_payload": 585984}},
        "layers": layers_out, "gates": _gates(layers_out, smoke),
        "operational_observations_only": {"elapsed_seconds": time.time() - started, "peak_rss_bytes": rss.peak,
                                            "peak_rss_method": rss.method},
    }
    _write_json_once(out_path, output)
    print("wrote %s" % out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="run toy controls only; never load donor or write results")
    parser.add_argument("--smoke", action="store_true", help="fixed calibration, one held-out sequence/layer; cache-only donor")
    args = parser.parse_args()
    if args.selftest and args.smoke:
        parser.error("--selftest and --smoke are mutually exclusive")
    if args.selftest:
        return _selftest()
    run(smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
