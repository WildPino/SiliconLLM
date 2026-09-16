#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E68 -- shared low-rank residual diagnostic for E67's mass carve.

Frozen design: docs/research/donor_adaptation/briefs/BRIEF_E68_SHARED_RESIDUAL.md

This is a CPU-only, local FFN-output diagnostic.  The dense ``z`` used to
select the mass groups is an oracle; this script neither implements a router
nor makes a BPB, generation, throughput, cache-residency, or rate claim.

    python e68_shared_residual.py --selftest
    python e68_shared_residual.py --smoke
    python e68_shared_residual.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


HERE = os.path.dirname(os.path.abspath(__file__))
DENSITY = os.path.abspath(os.path.join(HERE, "..", "density"))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if DENSITY not in sys.path:
    sys.path.insert(0, DENSITY)

HF = "Qwen/Qwen2.5-1.5B"
REV = "8faed761d45a263340a0528343f099c05c9a4323"
LAYERS = (1, 7, 14, 21, 27)
D_MODEL = 1536
E_GROUPS = 256
GROUP_NEURONS = 35
K = 3
FFN_DIM = E_GROUPS * GROUP_NEURONS
RANKS = (32, 64, 128)
PRIMARY_RANK = 64
RIDGE_SCALE = 0.001
THREADS = 6
TOKEN_CHUNK = 32
RECON_REL_RMS_TOL = 1e-5
OBJECTIVE_REL_TOL = 1e-10
ANCHOR_REL_TOL = 1e-5

CALIB_N_SEQ, CALIB_SEQLEN, CALIB_SEED = 8, 512, 424242
CALIB_IDS_SHA = "92fc41840c8feb0595758e540a048c1661bf3f618b3355d637ecf418b6c00157"
HELDOUT_N_SEQ, HELDOUT_SEQLEN, HELDOUT_SEED = 24, 512, 1234
HELDOUT_IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
HELDOUT_SELECTED_IDS_SHA = "6e4160a27d10288f24d6a60fded1b0537c97fd28df8798d2a65dac235124a4b6"
HELDOUT_START, HELDOUT_STOP = 384, 512

LABELS_PATH = os.path.join(DENSITY, "results", "d0c_labels", "labels_E256.npz")
LABELS_NPZ_SHA = "c39d0740b7f754daf168d811c77120c5394ae677e3882fef208d331d8c333c9c"
LEARNED_LABEL_SHA = {
    1: "20846b99936ee7c2ab90416af0d325ff7b27dfabdd086bfc1fada206bf46c5c5",
    7: "40902a27f43d449475f1f011255c598fb667153302bc893c1aa8088575518886",
    14: "878e8c2e62391401f5da3786d64838cba9852b4bb940b8c24e73a211c54ba1a3",
    21: "67427f39250809542a810a224e7b5d0be245d9e30839ce0fd2d3caf8988c9d72",
    27: "a61eb3a72b95172d3344ff155717d5172129afd350cb39a4e114bc3f419d188e",
}
E67_RUNNER = os.path.join(HERE, "e67_output_aware_selection.py")
E67_RESULT = os.path.join(HERE, "results", "e67_output_aware_selection.json")
E67_SMOKE_RESULT = os.path.join(HERE, "results", "e67_output_aware_selection_smoke.json")
E67_RUNNER_SHA = "4c5914646876b91c992fd056793314d9d8b05cebdfeb25a00fbfee2e17f9d88f"
E67_RESULT_SHA = "98d1b63ec6b259dbf1902d01e696fc78199e4709e244cbbae4e4ca2c4b4a33aa"
E67_SMOKE_RESULT_SHA = "ca299ccc3c948b2bf20e489561050de1bd449014e77a7252ca6ecf5977419c58"
E19_LABEL_LOADER = os.path.join(DENSITY, "e19_carve_rank.py")
E19_LABEL_LOADER_SHA = "31c595efd5ccc12bf37922e54a38939a69ece96566adcde257340656d75e9525"
BRIEF = os.path.join(REPO, "docs", "research", "donor_adaptation", "briefs",
                     "BRIEF_E68_SHARED_RESIDUAL.md")
BRIEF_SHA = "eb72a48086521f809c08149d84ed9cae521a8e571268dede5ac7186f2a98c5b5"
DENSITY_COMMON_SHA = "7ac00b31e91d0018c775e2e9b26120c0ec025075d7913218ea3cdcfe3196e461"
RESULTS = os.path.join(HERE, "results")

COMBINED_R0 = "combined_r0_sparse_only"
RESIDUAL_ARMS = tuple("residual_rank_%d" % rank for rank in RANKS)
SHARED_ARMS = tuple("shared_only_rank_%d" % rank for rank in RANKS)
ALL_ARMS = (COMBINED_R0,) + RESIDUAL_ARMS + SHARED_ARMS


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


def _stable_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Select largest scores, resolving exact ties to the lower group label."""
    if scores.ndim != 2 or not (0 < k <= scores.shape[1]):
        raise ValueError("invalid score matrix/k: %s, %s" % (tuple(scores.shape), k))
    return torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]


def _validate_labels(labels: torch.Tensor) -> None:
    if labels.ndim != 1 or labels.numel() != FFN_DIM:
        raise RuntimeError("labels have wrong FFN dimension: %s" % (tuple(labels.shape),))
    if labels.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("labels are not integral")
    counts = torch.bincount(labels.long(), minlength=E_GROUPS)
    expected = torch.full_like(counts, GROUP_NEURONS)
    if counts.numel() != E_GROUPS or not torch.equal(counts, expected):
        raise RuntimeError("labels are not 256 equal groups of 35")


def _mass_select(z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """E67's mass selector: float64 group reduction and stable tie order."""
    mass = torch.zeros(z.shape[0], E_GROUPS, dtype=torch.float64, device=z.device)
    mass.index_add_(1, labels, z.double().square())
    return _stable_topk(mass, K)


def _relative_reconstruction_error(y: torch.Tensor, reconstructed: torch.Tensor) -> float:
    denom = y.double().square().sum(dim=1).sqrt()
    if torch.any(denom <= 0):
        raise RuntimeError("zero-norm FFN output prevents reconstruction control")
    return float(((y.double() - reconstructed.double()).square().sum(dim=1).sqrt() / denom).max())


def _check_mass_selection(selected: torch.Tensor) -> None:
    """The one sparse path must select three distinct 35-neuron groups per token."""
    if selected.ndim != 2 or selected.shape[1] != K:
        raise RuntimeError("mass selector did not return exactly three groups")
    ordered = selected.sort(dim=1).values
    if torch.any(ordered[:, 1:] == ordered[:, :-1]):
        raise RuntimeError("mass selector returned duplicate groups")


class _RSS:
    """Peak-RSS observation only; it is not a speed or deployment measurement."""

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
        if self.process is None:
            return
        info = self.process.memory_info()
        self.peak = max(self.peak, int(getattr(info, "peak_wset", info.rss)), int(info.rss))


@dataclass
class FactorPair:
    """One low-rank map represented exactly as x @ A @ Vt."""

    rank: int
    a64: torch.Tensor
    vt64: torch.Tensor

    def predict64(self, x64: torch.Tensor) -> torch.Tensor:
        return (x64 @ self.a64) @ self.vt64

    def score_factors_fp32(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The frozen donor remains fp32; only these fitted factors are cast for score."""
        return self.a64.float().contiguous(), self.vt64.float().contiguous()

    def metadata(self) -> Dict[str, object]:
        finite = bool(torch.isfinite(self.a64).all() and torch.isfinite(self.vt64).all())
        return {
            "rank": self.rank,
            "A_shape": list(self.a64.shape),
            "Vt_shape": list(self.vt64.shape),
            "fit_dtype": "float64",
            "score_factor_conversion": "A and Vt cast float64 -> float32 before fp32 donor scoring",
            "finite": finite,
        }


@dataclass
class RidgeBasis:
    """The shared X-dependent Cholesky state, reused for both frozen targets."""

    x64: torch.Tensor
    lam: float
    chol_l: torch.Tensor


def _ridge_basis(x64: torch.Tensor) -> RidgeBasis:
    if x64.ndim != 2 or x64.dtype != torch.float64 or x64.shape[1] != D_MODEL:
        raise ValueError("X must be [N,%d] float64" % D_MODEL)
    trace = float(x64.square().sum().item())
    lam = RIDGE_SCALE * trace / float(x64.shape[1])
    if not np.isfinite(lam) or lam <= 0.0:
        raise RuntimeError("non-positive or non-finite ridge lambda: %r" % lam)
    c = x64.T @ x64
    c.diagonal().add_(lam)
    try:
        chol_l = torch.linalg.cholesky(c)
    except RuntimeError as exc:
        raise RuntimeError("ridge Cholesky failed") from exc
    if not bool(torch.isfinite(chol_l).all()):
        raise RuntimeError("ridge Cholesky produced non-finite values")
    return RidgeBasis(x64=x64, lam=lam, chol_l=chol_l)


def _fit_reduced_rank_path(basis: RidgeBasis, target64: torch.Tensor,
                           ranks: Iterable[int] = RANKS) -> Tuple[Dict[int, FactorPair], Dict[str, object]]:
    """Frozen reduced-rank ridge formula: C=LLt, M=L^-1 X^T R, B=A Vt."""
    x64, l = basis.x64, basis.chol_l
    if target64.shape != x64.shape or target64.dtype != torch.float64:
        raise ValueError("target must match X and be float64")
    try:
        m = torch.linalg.solve_triangular(l, x64.T @ target64, upper=False)
        u, sigma, vh = torch.linalg.svd(m, full_matrices=False)
    except RuntimeError as exc:
        raise RuntimeError("reduced-rank ridge solve/SVD failed") from exc
    if not bool(torch.isfinite(u).all() and torch.isfinite(sigma).all() and torch.isfinite(vh).all()):
        raise RuntimeError("reduced-rank ridge SVD produced non-finite values")

    factors: Dict[int, FactorPair] = {}
    objectives: Dict[str, float] = {}
    for rank in ranks:
        rank = int(rank)
        if not (0 <= rank <= min(m.shape)):
            raise ValueError("invalid rank %d for %s" % (rank, tuple(m.shape)))
        if rank == 0:
            a = torch.empty(D_MODEL, 0, dtype=torch.float64)
            vt = torch.empty(0, D_MODEL, dtype=torch.float64)
        else:
            # A = L^-T U_r Sigma_r, with C = L L^T exactly as preregistered.
            a = torch.linalg.solve_triangular(l.T, u[:, :rank] * sigma[:rank], upper=True)
            vt = vh[:rank, :]
        pair = FactorPair(rank=rank, a64=a.contiguous(), vt64=vt.contiguous())
        b = pair.a64 @ pair.vt64
        objective = (target64 - x64 @ b).square().sum() + basis.lam * b.square().sum()
        if not bool(torch.isfinite(b).all() and torch.isfinite(objective)):
            raise RuntimeError("non-finite reduced-rank ridge factor/objective")
        factors[rank] = pair
        objectives[str(rank)] = float(objective.item())
    ordered = sorted(factors)
    for low, high in zip(ordered, ordered[1:]):
        a, b = objectives[str(low)], objectives[str(high)]
        if b > a + max(1e-8, OBJECTIVE_REL_TOL * abs(a)):
            raise RuntimeError("ridge objective increased from rank %d to %d: %.17g -> %.17g" %
                               (low, high, a, b))
    return factors, {
        "lambda": basis.lam,
        "cholesky_success": True,
        "svd_success": True,
        "M_shape": list(m.shape),
        "singular_values_finite": bool(torch.isfinite(sigma).all()),
        "objective_by_rank": objectives,
        "objective_nonincreasing_by_rank": True,
    }


class MetricAccumulator:
    """Float64 SSE accounting; the first arm supplies the common output energy."""

    def __init__(self, n_seq: int, arms: Sequence[str]):
        self.n_seq = n_seq
        self.arms = tuple(arms)
        self.error_sum = {arm: 0.0 for arm in arms}
        self.sequence_error = {arm: np.zeros(n_seq, dtype=np.float64) for arm in arms}
        self.token_relative = {arm: [] for arm in arms}
        self.energy_sum = 0.0
        self.sequence_energy = np.zeros(n_seq, dtype=np.float64)
        self.reference_tokens = 0

    def add_error(self, arm: str, y: torch.Tensor, error: torch.Tensor,
                  sequence: int, reference: bool = False) -> None:
        if arm not in self.error_sum or not (0 <= sequence < self.n_seq):
            raise RuntimeError("invalid metric arm/sequence")
        err = error.double().square().sum(dim=1)
        energy = y.double().square().sum(dim=1)
        if torch.any(energy <= 0):
            raise RuntimeError("zero-norm FFN output prevents relative SSE accounting")
        err_np = err.detach().cpu().numpy()
        energy_np = energy.detach().cpu().numpy()
        self.error_sum[arm] += float(err_np.sum())
        self.sequence_error[arm][sequence] += float(err_np.sum())
        self.token_relative[arm].extend((err_np / energy_np).tolist())
        if reference:
            self.energy_sum += float(energy_np.sum())
            self.sequence_energy[sequence] += float(energy_np.sum())
            self.reference_tokens += int(error.shape[0])

    def result(self, expected_tokens: int) -> Dict[str, object]:
        if self.reference_tokens != expected_tokens:
            raise RuntimeError("accounted %d tokens, expected %d" %
                               (self.reference_tokens, expected_tokens))
        if self.energy_sum <= 0 or np.any(self.sequence_energy <= 0):
            raise RuntimeError("invalid metric energy")
        out = {}
        for arm in self.arms:
            out[arm] = {
                "sum_sse_over_sum_energy": self.error_sum[arm] / self.energy_sum,
                "median_token_sse_over_energy": float(np.median(self.token_relative[arm])),
                "per_sequence_sum_sse_over_sum_energy":
                    (self.sequence_error[arm] / self.sequence_energy).tolist(),
                "sum_sse": self.error_sum[arm],
            }
        return out


class _MassProjector:
    """E67-identical group contribution construction, bounded to TOKEN_CHUNK tokens."""

    def __init__(self, down_weight: torch.Tensor, labels: torch.Tensor):
        _validate_labels(labels)
        if down_weight.shape != (D_MODEL, FFN_DIM):
            raise RuntimeError("unexpected down weight shape %s" % (tuple(down_weight.shape),))
        self.weight = down_weight.detach()
        self.labels = labels.detach().long().to(self.weight.device)
        order = torch.argsort(self.labels, stable=True)
        self.grouped_weight = self.weight[:, order].reshape(D_MODEL, E_GROUPS, GROUP_NEURONS)

    def sparse_output(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Return y, y_sparse, chosen groups and the dense reconstruction control value."""
        order = torch.argsort(self.labels, stable=True)
        grouped_z = z[:, order].reshape(z.shape[0], E_GROUPS, GROUP_NEURONS)
        contribution = torch.einsum("teg,deg->ted", grouped_z, self.grouped_weight)
        y = F.linear(z, self.weight)
        reconstructed = contribution.sum(dim=1)
        reconstruction_error = _relative_reconstruction_error(y, reconstructed)
        if reconstruction_error >= RECON_REL_RMS_TOL:
            raise RuntimeError("E67 dense reconstruction control failed: %.3e >= %.3e" %
                               (reconstruction_error, RECON_REL_RMS_TOL))
        selected = _mass_select(z, self.labels)
        _check_mass_selection(selected)
        chosen = contribution.gather(1, selected.unsqueeze(-1).expand(-1, -1, D_MODEL)).sum(dim=1)
        return y, chosen, selected, reconstruction_error


class FitCapture:
    """Collect one layer's calibration X, residual R, and y in float64."""

    def __init__(self, model: torch.nn.Module, layer: int, labels: torch.Tensor,
                 n_seq: int, rss: _RSS):
        self.layer, self.n_seq, self.rss = layer, n_seq, rss
        self.sequence = None
        self.pending_x = None
        self.x_parts: List[torch.Tensor] = []
        self.residual_parts: List[torch.Tensor] = []
        self.y_parts: List[torch.Tensor] = []
        self.reconstruction_max = 0.0
        down = model.model.layers[layer].mlp.down_proj
        self.projector = _MassProjector(down.weight, labels)
        mlp = model.model.layers[layer].mlp
        self.handles = []
        try:
            self.handles.append(mlp.register_forward_pre_hook(self._mlp_hook))
            self.handles.append(down.register_forward_pre_hook(self._down_hook))
        except Exception:
            self.detach()
            raise

    def _mlp_hook(self, _module, args):
        if self.sequence is None:
            return None
        x = args[0].detach()
        if x.ndim != 3 or x.shape[0] != 1 or x.shape[1:] != (CALIB_SEQLEN, D_MODEL):
            raise RuntimeError("unexpected layer %d MLP input shape %s" % (self.layer, tuple(x.shape)))
        self.pending_x = x[0].cpu().double().contiguous()
        return None

    def _down_hook(self, _module, args):
        if self.sequence is None:
            return None
        if self.pending_x is None:
            raise RuntimeError("layer %d down hook fired without its MLP input" % self.layer)
        z_all = args[0].detach()
        if z_all.ndim != 3 or z_all.shape[0] != 1 or z_all.shape[1:] != (CALIB_SEQLEN, FFN_DIM):
            raise RuntimeError("unexpected layer %d down input shape %s" % (self.layer, tuple(z_all.shape)))
        for start in range(0, CALIB_SEQLEN, TOKEN_CHUNK):
            stop = min(start + TOKEN_CHUNK, CALIB_SEQLEN)
            y, sparse, selected, reconstruction_error = self.projector.sparse_output(z_all[0, start:stop])
            if selected.shape != (stop - start, K):
                raise RuntimeError("mass selector did not return three groups")
            self.reconstruction_max = max(self.reconstruction_max, reconstruction_error)
            self.x_parts.append(self.pending_x[start:stop])
            self.residual_parts.append((y.double() - sparse.double()).cpu().contiguous())
            self.y_parts.append(y.double().cpu().contiguous())
        self.pending_x = None
        self.rss.observe()
        return None

    def detach(self) -> None:
        """Remove hooks without replacing an in-flight forward failure with cleanup noise."""
        _remove_handles(self.handles, "calibration layer %d" % self.layer)

    def materialize(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
        """Return and release this layer's capture lists; factors are fitted afterwards."""
        if self.pending_x is not None:
            raise RuntimeError("unconsumed MLP input at layer %d" % self.layer)
        try:
            expected = self.n_seq * CALIB_SEQLEN
            x = torch.cat(self.x_parts, dim=0)
            residual = torch.cat(self.residual_parts, dim=0)
            y = torch.cat(self.y_parts, dim=0)
            if x.shape != (expected, D_MODEL) or residual.shape != x.shape or y.shape != x.shape:
                raise RuntimeError("calibration capture has wrong shape at layer %d" % self.layer)
            return x, residual, y, {
                "capture": "MLP forward-pre-hook for x; down_proj forward-pre-hook for z",
                "fit_positions": [0, CALIB_SEQLEN - 1],
                "fit_tokens": expected,
                "reconstruction_max_relative_rms": self.reconstruction_max,
                "reconstruction_tolerance": RECON_REL_RMS_TOL,
                "reconstruction_fires": True,
            }
        finally:
            # The returned concatenated tensors own their storage.  Drop the chunk lists before
            # the per-layer ridge solve so no duplicated calibration activations remain live.
            self.x_parts.clear()
            self.residual_parts.clear()
            self.y_parts.clear()

    def discard(self) -> None:
        """Best-effort memory release for an aborted calibration collection."""
        self.pending_x = None
        self.x_parts.clear()
        self.residual_parts.clear()
        self.y_parts.clear()


def _remove_handles(handles: List[object], context: str) -> None:
    """Remove every hook, preserving a pre-existing exception as the root cause."""
    errors = []
    while handles:
        handle = handles.pop()
        try:
            handle.remove()
        except Exception as exc:  # pragma: no cover - PyTorch hook removal normally cannot fail.
            errors.append(exc)
    if errors:
        message = "failed to remove %d hook(s) for %s: %s" % (len(errors), context, errors[0])
        if sys.exc_info()[0] is not None:
            print("WARN E68 cleanup: " + message, file=sys.stderr, flush=True)
        else:
            raise RuntimeError(message) from errors[0]


class FitCaptureSet:
    """Collect all registered layers in the same eight frozen calibration forwards."""

    def __init__(self, model: torch.nn.Module, layers: Sequence[int], labels: Mapping[int, torch.Tensor],
                 n_seq: int, rss: _RSS):
        self.layers = tuple(layers)
        self.captures: Dict[int, FitCapture] = {}
        try:
            for layer in self.layers:
                self.captures[layer] = FitCapture(model, layer, labels[layer], n_seq, rss)
        except Exception:
            self.detach()
            self.discard()
            raise

    def collect(self, model: torch.nn.Module, ids: torch.Tensor) -> None:
        """Run exactly one full-model forward per frozen calibration sequence."""
        try:
            with torch.inference_mode():
                for sequence in range(CALIB_N_SEQ):
                    for capture in self.captures.values():
                        capture.sequence = sequence
                    model.model(ids[sequence:sequence + 1])
                    print("E68 calibration sequence %d/%d captured for layers %s" %
                          (sequence + 1, CALIB_N_SEQ, list(self.layers)), flush=True)
        finally:
            for capture in self.captures.values():
                capture.sequence = None
            self.detach()

    def detach(self) -> None:
        for capture in self.captures.values():
            capture.detach()

    def discard(self) -> None:
        for capture in self.captures.values():
            capture.discard()


class ScoreCapture:
    """Score fixed factors on held-out data; this class has no fit storage or fit method."""

    def __init__(self, model: torch.nn.Module, layer: int, labels: torch.Tensor,
                 factors_fp32: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
                 n_seq: int, seq_len: int, start: int, stop: int, rss: _RSS):
        self.layer, self.n_seq, self.seq_len = layer, n_seq, seq_len
        self.start, self.stop, self.rss = start, stop, rss
        self.sequence = None
        self.pending_x = None
        self.reconstruction_max = 0.0
        self.metrics = MetricAccumulator(n_seq, ALL_ARMS)
        down = model.model.layers[layer].mlp.down_proj
        self.projector = _MassProjector(down.weight, labels)
        self.factors_fp32 = dict(factors_fp32)
        for arm, (a, vt) in self.factors_fp32.items():
            if a.dtype != torch.float32 or vt.dtype != torch.float32:
                raise RuntimeError("%s score factors are not fp32" % arm)
            if a.shape[0] != D_MODEL or vt.shape[1] != D_MODEL or a.shape[1] != vt.shape[0]:
                raise RuntimeError("%s score factor shapes do not compose" % arm)
        mlp = model.model.layers[layer].mlp
        self.handles = []
        try:
            self.handles.append(mlp.register_forward_pre_hook(self._mlp_hook))
            self.handles.append(down.register_forward_pre_hook(self._down_hook))
        except Exception:
            self.detach()
            raise

    def _mlp_hook(self, _module, args):
        if self.sequence is None:
            return None
        x = args[0].detach()
        if x.ndim != 3 or x.shape[0] != 1 or x.shape[1:] != (self.seq_len, D_MODEL):
            raise RuntimeError("unexpected layer %d held-out MLP input shape %s" %
                               (self.layer, tuple(x.shape)))
        self.pending_x = x[0, self.start:self.stop].contiguous()
        return None

    def _down_hook(self, _module, args):
        if self.sequence is None:
            return None
        if self.pending_x is None:
            raise RuntimeError("layer %d held-out down hook fired without MLP input" % self.layer)
        z_all = args[0].detach()
        if z_all.ndim != 3 or z_all.shape[0] != 1 or z_all.shape[1:] != (self.seq_len, FFN_DIM):
            raise RuntimeError("unexpected layer %d held-out down input shape %s" %
                               (self.layer, tuple(z_all.shape)))
        z_all = z_all[0, self.start:self.stop]
        for begin in range(0, z_all.shape[0], TOKEN_CHUNK):
            end = min(begin + TOKEN_CHUNK, z_all.shape[0])
            y, sparse, selected, reconstruction_error = self.projector.sparse_output(z_all[begin:end])
            if selected.shape != (end - begin, K):
                raise RuntimeError("held-out mass selector did not return three groups")
            self.reconstruction_max = max(self.reconstruction_max, reconstruction_error)
            x = self.pending_x[begin:end]
            self.metrics.add_error(COMBINED_R0, y, y - sparse, self.sequence, reference=True)
            for arm, (a, vt) in self.factors_fp32.items():
                shared = (x @ a) @ vt
                yhat = sparse + shared if arm.startswith("residual_") else shared
                self.metrics.add_error(arm, y, y - yhat, self.sequence)
        self.pending_x = None
        self.rss.observe()
        return None

    def detach(self) -> None:
        _remove_handles(self.handles, "held-out layer %d" % self.layer)

    def close(self) -> Tuple[Dict[str, object], Dict[str, object]]:
        self.detach()
        if self.pending_x is not None:
            raise RuntimeError("unconsumed held-out MLP input at layer %d" % self.layer)
        expected = self.n_seq * (self.stop - self.start)
        return self.metrics.result(expected), {
            "capture": "MLP forward-pre-hook for x; down_proj forward-pre-hook for z",
            "score_positions": [self.start, self.stop - 1],
            "score_tokens": expected,
            "reconstruction_max_relative_rms": self.reconstruction_max,
            "reconstruction_tolerance": RECON_REL_RMS_TOL,
            "reconstruction_fires": True,
        }


def _load_labels(n_layers: int, selected_layers: Iterable[int]) -> Tuple[Dict[int, torch.Tensor], Dict[str, object]]:
    import e19_carve_rank as e19

    if _sha_file(E19_LABEL_LOADER) != E19_LABEL_LOADER_SHA:
        raise RuntimeError("E19 label-loader source hash mismatch -- STOP")
    if not os.path.exists(LABELS_PATH) or _sha_file(LABELS_PATH) != LABELS_NPZ_SHA:
        raise RuntimeError("frozen D0c labels hash mismatch -- STOP")
    learned_np, _random_np = e19.load_labels(E_GROUPS)
    if set(learned_np) != set(range(n_layers)):
        raise RuntimeError("E19 learned labels do not name exactly all donor layers")
    learned = {}
    hashes = {}
    for layer in selected_layers:
        label = torch.from_numpy(np.asarray(learned_np[layer], dtype=np.int64)).contiguous()
        _validate_labels(label)
        digest = _sha_tensor(label)
        if digest != LEARNED_LABEL_SHA[layer]:
            raise RuntimeError("frozen E67 label hash mismatch at layer %d -- STOP" % layer)
        learned[layer] = label
        hashes[str(layer)] = digest
    return learned, {
        "path": LABELS_PATH,
        "labels_E256_npz_sha256": LABELS_NPZ_SHA,
        "learned_per_layer": hashes,
        "E": E_GROUPS,
        "neurons_per_group": GROUP_NEURONS,
    }


def _load_e67_anchors(smoke: bool) -> Dict[int, float]:
    path = E67_SMOKE_RESULT if smoke else E67_RESULT
    expected_hash = E67_SMOKE_RESULT_SHA if smoke else E67_RESULT_SHA
    if _sha_file(E67_RUNNER) != E67_RUNNER_SHA:
        raise RuntimeError("E67 runner source hash mismatch -- STOP")
    if not os.path.exists(path) or _sha_file(path) != expected_hash:
        raise RuntimeError("E67 result hash mismatch -- STOP")
    data = json.load(open(path, encoding="utf-8"))
    expected_layers = (27,) if smoke else LAYERS
    if data["slice"]["full_ids_sha256"] != HELDOUT_IDS_SHA:
        raise RuntimeError("E67 result does not name the pinned held-out slice")
    return {layer: float(data["layers"][str(layer)]["selectors"]["mass"]["sum_sse"])
            for layer in expected_layers}


def _check_e67_r0(layer: int, heldout: Mapping[str, object], expected_sse: float) -> Dict[str, object]:
    actual = float(heldout[COMBINED_R0]["sum_sse"])
    relative = abs(actual - expected_sse) / max(abs(expected_sse), 1.0)
    if relative > ANCHOR_REL_TOL:
        raise RuntimeError("E67 r=0 mass anchor mismatch at layer %d: %.17g vs %.17g (rel %.3e)" %
                           (layer, actual, expected_sse, relative))
    return {"expected_e67_mass_sum_sse": expected_sse, "observed_r0_sum_sse": actual,
            "relative_difference": relative, "relative_tolerance": ANCHOR_REL_TOL, "fires": True}


def _calibration_metrics(x: torch.Tensor, residual: torch.Tensor, y: torch.Tensor,
                         residual_factors: Mapping[int, FactorPair],
                         shared_factors: Mapping[int, FactorPair]) -> Dict[str, object]:
    if x.shape[0] != CALIB_N_SEQ * CALIB_SEQLEN:
        raise RuntimeError("calibration metric input count is not 4096")
    metrics = MetricAccumulator(CALIB_N_SEQ, ALL_ARMS)
    for sequence in range(CALIB_N_SEQ):
        start, stop = sequence * CALIB_SEQLEN, (sequence + 1) * CALIB_SEQLEN
        xs, rs, ys = x[start:stop], residual[start:stop], y[start:stop]
        # Rank zero is exactly y_sparse: its output error is exactly the residual.
        metrics.add_error(COMBINED_R0, ys, rs, sequence, reference=True)
        for rank, factor in residual_factors.items():
            metrics.add_error("residual_rank_%d" % rank, ys, rs - factor.predict64(xs), sequence)
        for rank, factor in shared_factors.items():
            metrics.add_error("shared_only_rank_%d" % rank, ys, ys - factor.predict64(xs), sequence)
    return metrics.result(CALIB_N_SEQ * CALIB_SEQLEN)


def _fit_layer(capture: FitCapture, layer: int, rss: _RSS) -> Tuple[Dict[int, FactorPair],
                                                                      Dict[int, FactorPair],
                                                                      Dict[str, object]]:
    """Fit one already-captured layer, then release its 4096x1536 float64 matrices."""
    x = residual = y = basis = None
    try:
        x, residual, y, capture_meta = capture.materialize()
        # X, R and y are the only materialized calibration activations for this fit.
        basis = _ridge_basis(x)
        residual_factors, residual_fit = _fit_reduced_rank_path(basis, residual)
        shared_factors, shared_fit = _fit_reduced_rank_path(basis, y)
        calibration = _calibration_metrics(x, residual, y, residual_factors, shared_factors)
        factor_meta = {
            "residual_target_y_minus_y_sparse": {str(rank): pair.metadata()
                                                    for rank, pair in residual_factors.items()},
            "shared_only_target_y": {str(rank): pair.metadata()
                                      for rank, pair in shared_factors.items()},
        }
        out = {
            "capture": capture_meta,
            "ridge": {
                "formula": "C=X^T X+lambda I=L L^T; M=L^-1 X^T R=U Sigma V^T; "
                           "A=L^-T U_r Sigma_r; prediction=(x A) V^T",
                "lambda": basis.lam,
                "lambda_rule": "0.001 * trace(X^T X) / D",
                "fit_shape_X": list(x.shape),
                "fit_target_shape": list(residual.shape),
                "float64_accumulations_and_linear_algebra": True,
                "residual_fit": residual_fit,
                "shared_only_fit": shared_fit,
            },
            "factors": factor_meta,
            "calibration": calibration,
            "memory_discipline": "all registered layers are captured in eight forwards; this layer's "
                                 "X/residual/y matrices are materialized, fitted, and released sequentially",
        }
        return residual_factors, shared_factors, out
    finally:
        # Retain only returned factors.  If fitting fails, this also drops the materialized tensors
        # before the original exception propagates.
        x = residual = y = basis = None
        rss.observe()


def _score_layers(model: torch.nn.Module, ids: torch.Tensor, labels: Mapping[int, torch.Tensor],
                  residual: Mapping[int, Mapping[int, FactorPair]],
                  shared: Mapping[int, Mapping[int, FactorPair]], layers: Sequence[int],
                  smoke: bool, rss: _RSS) -> Dict[str, Dict[str, object]]:
    n_seq = 1 if smoke else 4
    captures = {}
    try:
        for layer in layers:
            factor_fp32 = {}
            for rank in RANKS:
                factor_fp32["residual_rank_%d" % rank] = residual[layer][rank].score_factors_fp32()
                factor_fp32["shared_only_rank_%d" % rank] = shared[layer][rank].score_factors_fp32()
            captures[layer] = ScoreCapture(model, layer, labels[layer], factor_fp32, n_seq,
                                           HELDOUT_SEQLEN, HELDOUT_START, HELDOUT_STOP, rss)
        with torch.inference_mode():
            for sequence in range(n_seq):
                for capture in captures.values():
                    capture.sequence = sequence
                model.model(ids[sequence:sequence + 1])
                print("E68 held-out sequence %d/%d scored" % (sequence + 1, n_seq), flush=True)
                rss.observe()
    finally:
        for capture in captures.values():
            capture.sequence = None
            capture.detach()
    out = {}
    for layer, capture in captures.items():
        heldout, control = capture.close()
        out[str(layer)] = {"heldout": heldout, "score_capture": control}
    return out


def _aggregate_sse(layers: Mapping[str, Mapping[str, object]], arm: str) -> float:
    value = sum(float(row["heldout"][arm]["sum_sse"]) for row in layers.values())
    if not np.isfinite(value) or value <= 0.0:
        raise RuntimeError("invalid aggregate SSE for %s" % arm)
    return value


def _verdict(layers: Mapping[str, Mapping[str, object]], smoke: bool) -> Dict[str, object]:
    if smoke:
        return {"name": "NOT_APPLICABLE_SMOKE",
                "reason": "frozen verdict requires all five registered layers and four held-out sequences"}
    mass = _aggregate_sse(layers, COMBINED_R0)
    combined = _aggregate_sse(layers, "residual_rank_%d" % PRIMARY_RANK)
    shared = _aggregate_sse(layers, "shared_only_rank_%d" % PRIMARY_RANK)
    combined_over_mass = combined / mass
    combined_over_shared = combined / shared
    shared_over_mass = shared / mass
    layer_ratios = {layer: float(row["heldout"]["residual_rank_64"]["sum_sse"]) /
                    float(row["heldout"][COMBINED_R0]["sum_sse"])
                    for layer, row in layers.items()}
    complementary = (combined_over_mass <= 0.50 and combined_over_shared <= 0.80 and
                     sum(value <= 0.70 for value in layer_ratios.values()) >= 4 and
                     all(value <= 1.05 for value in layer_ratios.values()))
    if complementary:
        name = "COMPLEMENTARY_SHARED_SIGNAL"
    elif shared_over_mass <= 0.50:
        name = "SHARED_ONLY_SIGNAL"
    elif combined_over_mass <= 0.80:
        name = "PARTIAL_LOCAL_SIGNAL"
    else:
        name = "NO_LOCAL_SIGNAL"
    return {
        "name": name,
        "primary_rank": PRIMARY_RANK,
        "aggregate_sse": {COMBINED_R0: mass, "residual_rank_64": combined,
                            "shared_only_rank_64": shared},
        "aggregate_ratios": {"combined_over_mass": combined_over_mass,
                               "combined_over_shared_only": combined_over_shared,
                               "shared_only_over_mass": shared_over_mass},
        "per_layer_combined_over_mass": layer_ratios,
        "scope": "local linear FFN-output reconstruction only; no rate, BPB, engine, cache, or deployment claim",
    }


def _run_e67_planted_controls() -> None:
    """Use E67's frozen synthetic mass/tie/cancellation controls without loading a donor."""
    from e67_output_aware_selection import _selftest as e67_selftest
    if e67_selftest() != 0:
        raise RuntimeError("E67 planted controls did not pass")


def _selftest() -> int:
    """Pure synthetic tests: formula/factors, objective monotonicity, and exact r=0."""
    torch.manual_seed(68001)
    # Keep D local here so the test remains small and never touches a donor/model/cache.
    n, d = 19, 7
    x = torch.randn(n, d, dtype=torch.float64)
    target = torch.randn(n, d, dtype=torch.float64)
    trace = float(x.square().sum())
    lam = RIDGE_SCALE * trace / d
    c = x.T @ x + lam * torch.eye(d, dtype=torch.float64)
    l = torch.linalg.cholesky(c)
    m = torch.linalg.solve_triangular(l, x.T @ target, upper=False)
    u, sigma, vh = torch.linalg.svd(m, full_matrices=False)
    objectives = []
    for rank in range(d + 1):
        if rank:
            a = torch.linalg.solve_triangular(l.T, u[:, :rank] * sigma[:rank], upper=True)
            vt = vh[:rank]
        else:
            a, vt = torch.empty(d, 0, dtype=torch.float64), torch.empty(0, d, dtype=torch.float64)
        b_from_factors = a @ vt
        # Independently construct the prescribed B_r expression for this planted problem.
        b_formula = (torch.linalg.solve_triangular(l.T, u[:, :rank] * sigma[:rank], upper=True) @ vh[:rank]
                     if rank else torch.zeros(d, d, dtype=torch.float64))
        assert torch.allclose(b_from_factors, b_formula, atol=1e-11, rtol=1e-11)
        objective = (target - x @ b_from_factors).square().sum() + lam * b_from_factors.square().sum()
        objectives.append(float(objective))
    full_ridge = torch.linalg.solve(c, x.T @ target)
    assert torch.allclose((torch.linalg.solve_triangular(l.T, u * sigma, upper=True) @ vh), full_ridge,
                          atol=1e-10, rtol=1e-10)
    assert all(b <= a + max(1e-10, OBJECTIVE_REL_TOL * abs(a))
               for a, b in zip(objectives, objectives[1:])), objectives

    # r=0 is not a zero-output control in the combined arm: it is bit-identical sparse-only.
    sparse = torch.randn(5, d, dtype=torch.float64)
    zero_a, zero_vt = torch.empty(d, 0, dtype=torch.float64), torch.empty(0, d, dtype=torch.float64)
    r0_prediction = sparse + (torch.randn(5, d, dtype=torch.float64) @ zero_a) @ zero_vt
    assert torch.equal(r0_prediction, sparse)

    # E67-compatible float64 mass tie and dense reconstruction controls, also entirely synthetic.
    z = torch.tensor([[3.0, 3.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    selected = _stable_topk(torch.stack((z.double().square()[:, 0], z.double().square()[:, 1],
                                          z.double().square()[:, 2]), dim=1), 1)
    assert selected.tolist() == [[0]]
    y = torch.tensor([[1.5, -2.0]], dtype=torch.float32)
    assert _relative_reconstruction_error(y, y.clone()) == 0.0
    print("E68 selftest: PASS")
    print("  reduced-rank ridge factors reproduce B=A V^T and full ridge at rank D")
    print("  synthetic ridge objective is non-increasing from rank 0 through rank D")
    print("  combined r=0 is exactly sparse-only; float64 lower-label tie and reconstruction fire")
    return 0


def run(smoke: bool = False) -> str:
    """Run the frozen full protocol (or a clearly non-verdict one-layer smoke check)."""
    import common as c

    _selftest()
    _run_e67_planted_controls()
    torch.set_num_threads(THREADS)
    torch.set_grad_enabled(False)
    started = time.time()
    rss = _RSS()
    rss.observe()
    model, tokenizer = c.load_model(dtype=torch.float32)
    model.eval()
    if next(model.parameters()).device.type != "cpu":
        raise RuntimeError("E68 is CPU-only; donor was not loaded on CPU")
    architecture = c.arch(model)
    if _sha_file(BRIEF) != BRIEF_SHA:
        raise RuntimeError("frozen E68 brief hash mismatch -- STOP")
    if _sha_file(c.__file__) != DENSITY_COMMON_SHA:
        raise RuntimeError("frozen density common.py hash mismatch -- STOP")
    if (architecture["revision"] != REV or architecture["d_model"] != D_MODEL or
            architecture["d_ffn"] != FFN_DIM or architecture["n_layers"] <= max(LAYERS)):
        raise RuntimeError("pinned donor architecture mismatch: %s" % architecture)

    calib_ids, _calib_bytes, calib_meta = c.get_slice(tokenizer, "calib", CALIB_N_SEQ,
                                                       CALIB_SEQLEN, CALIB_SEED)
    if calib_meta["ids_sha256"] != CALIB_IDS_SHA:
        raise RuntimeError("CALIB SLICE HASH MISMATCH -- STOP: %s" % calib_meta["ids_sha256"])
    heldout_ids, _heldout_bytes, heldout_meta = c.get_slice(tokenizer, "heldout", HELDOUT_N_SEQ,
                                                             HELDOUT_SEQLEN, HELDOUT_SEED)
    if heldout_meta["ids_sha256"] != HELDOUT_IDS_SHA:
        raise RuntimeError("HELDOUT SLICE HASH MISMATCH -- STOP: %s" % heldout_meta["ids_sha256"])
    if calib_meta["part"] == heldout_meta["part"] or CALIB_IDS_SHA == HELDOUT_IDS_SHA:
        raise RuntimeError("fit and held-out partitions are not disjoint -- STOP")
    run_layers = (27,) if smoke else LAYERS
    heldout_n_seq = 1 if smoke else 4
    selected_heldout = heldout_ids[:heldout_n_seq, HELDOUT_START:HELDOUT_STOP].contiguous()
    selected_hash = _sha_tensor(selected_heldout)
    expected_selected_hash = ("271647d218110b267fd4d01a41f87fca42da239757c94fd5d80ccc9ad6302ba9"
                              if smoke else HELDOUT_SELECTED_IDS_SHA)
    if selected_hash != expected_selected_hash:
        raise RuntimeError("selected E67 held-out positions hash mismatch -- STOP: %s" % selected_hash)
    anchors = _load_e67_anchors(smoke)
    labels, labels_meta = _load_labels(architecture["n_layers"], run_layers)

    residual_factors: Dict[int, Dict[int, FactorPair]] = {}
    shared_factors: Dict[int, Dict[int, FactorPair]] = {}
    layer_results: Dict[str, Dict[str, object]] = {}
    fit_captures = FitCaptureSet(model, run_layers, labels, CALIB_N_SEQ, rss)
    try:
        # One hook set spans all registered layers, so full mode is exactly eight calibration
        # forwards (and smoke remains one layer with the same eight frozen fit forwards).
        fit_captures.collect(model, calib_ids)
        for layer in run_layers:
            res, shared, fit_result = _fit_layer(fit_captures.captures[layer], layer, rss)
            residual_factors[layer] = res
            shared_factors[layer] = shared
            layer_results[str(layer)] = {"fit": fit_result}
            rss.observe()
    finally:
        # This is intentionally non-throwing memory cleanup, so it cannot hide a collection or
        # numerical-fit failure that is already propagating.
        fit_captures.discard()

    heldout_results = _score_layers(model, heldout_ids, labels, residual_factors, shared_factors,
                                    run_layers, smoke, rss)
    for layer in run_layers:
        row = heldout_results[str(layer)]
        row["r0_e67_mass_anchor"] = _check_e67_r0(layer, row["heldout"], anchors[layer])
        layer_results[str(layer)].update(row)
    rss.observe()

    source_hashes = {
        "runner_sha256": _sha_file(__file__),
        "brief_sha256": _sha_file(BRIEF),
        "density_common_sha256": _sha_file(c.__file__),
        "e19_carve_rank_sha256": _sha_file(E19_LABEL_LOADER),
        "e67_runner_sha256": _sha_file(E67_RUNNER),
        "e67_result_sha256": _sha_file(E67_SMOKE_RESULT if smoke else E67_RESULT),
    }
    output = {
        "brief": "BRIEF_E68_SHARED_RESIDUAL.md",
        "scope": "CPU-only local linear shared-plus-block-sparse FFN-output diagnostic; no BPB, "
                 "generation, engine, cache-residency, throughput, or rate claim",
        "smoke": smoke,
        "planted_selftests": "PASS",
        "donor": {"model": HF, "revision": REV, "precision": "fp32", "attention": "eager"},
        "threads": THREADS,
        "architecture": architecture,
        "slices": {
            "calibration": {"part": "calib", "n_sequences": CALIB_N_SEQ,
                            "sequence_length": CALIB_SEQLEN, "seed": CALIB_SEED,
                            "ids_sha256": calib_meta["ids_sha256"], "expected_ids_sha256": CALIB_IDS_SHA,
                            "positions": [0, CALIB_SEQLEN - 1], "token_count_per_layer": 4096},
            "heldout": {"part": "heldout", "full_n_sequences": HELDOUT_N_SEQ,
                         "sequence_length": HELDOUT_SEQLEN, "seed": HELDOUT_SEED,
                         "full_ids_sha256": heldout_meta["ids_sha256"],
                         "expected_full_ids_sha256": HELDOUT_IDS_SHA,
                         "selected_sequences": list(range(heldout_n_seq)),
                         "selected_positions": [HELDOUT_START, HELDOUT_STOP - 1],
                         "selected_ids_sha256": selected_hash,
                         "expected_selected_ids_sha256": expected_selected_hash,
                         "selected_token_count_per_layer": int(selected_heldout.numel())},
            "disjoint_fit_and_heldout": True,
            "heldout_used_for_fit": False,
        },
        "labels": labels_meta,
        "source_hashes": source_hashes,
        "protocol": {
            "layers": list(run_layers), "k": K, "groups": E_GROUPS,
            "neurons_per_selected_group": GROUP_NEURONS,
            "selected_neurons_per_token": K * GROUP_NEURONS,
            "mass_selector": "float64 sum(z_i^2), stable descending sort, lower label wins exact ties",
            "capture": "x from MLP forward-pre-hook; z from down_proj forward-pre-hook",
            "ranks": list(RANKS), "primary_rank": PRIMARY_RANK,
            "fit_order": "calibration only; held-out scores are never supplied to the fit",
            "rank_zero": "combined r=0 is exactly sparse-only, not a fitted zero-output arm",
            "shared_only": "rank 32/64/128 fit y directly and omit selected groups at score time; "
                           "only rank 64 participates in the preregistered verdict",
        },
        "layers": layer_results,
        "verdict": _verdict(layer_results, smoke),
        "operational_observations_only": {"elapsed_seconds": time.time() - started,
                                            "peak_rss_bytes": rss.peak,
                                            "peak_rss_method": rss.method},
    }
    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, "e68_shared_residual%s.json" % ("_smoke" if smoke else ""))
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, default=_json_default)
    print("wrote %s" % out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="pure synthetic checks only; never loads donor")
    parser.add_argument("--smoke", action="store_true", help="full fixed calibration, one held-out sequence and layer 27; loads donor")
    args = parser.parse_args()
    if args.selftest and args.smoke:
        parser.error("--selftest and --smoke are mutually exclusive")
    if args.selftest:
        return _selftest()
    run(smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
