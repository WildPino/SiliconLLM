#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E67 -- local FFN reconstruction under output-aware group selection.

Frozen protocol:
docs/research/donor_adaptation/briefs/BRIEF_E67_OUTPUT_AWARE_SELECTION.md

This is deliberately a local, dense-information diagnostic.  Every sparse
selector constructs every group contribution first; it is not a router or a
speed measurement.  It never runs BPB or generation.

    python e67_output_aware_selection.py --selftest
    python e67_output_aware_selection.py --smoke
    python e67_output_aware_selection.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


HERE = os.path.dirname(os.path.abspath(__file__))
DENSITY = os.path.abspath(os.path.join(HERE, "..", "density"))
if DENSITY not in sys.path:
    sys.path.insert(0, DENSITY)

HF = "Qwen/Qwen2.5-1.5B"
REV = "8faed761d45a263340a0528343f099c05c9a4323"
EXPECT_IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
E_GROUPS = 256
GROUP_NEURONS = 35
K = 3
LAYERS = (1, 7, 14, 21, 27)
N_SEQ, SEQLEN, SEED = 24, 512, 1234
POSITION_START, POSITION_STOP = 384, 512
TOKEN_CHUNK = 32
THREADS = 6
RECON_REL_RMS_TOL = 1e-5
RESULTS = os.path.join(HERE, "results")

ARMS = ("mass", "output_norm", "output_greedy", "random_label_greedy", "dense_identity")
SPARSE_ARMS = ARMS[:4]


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: str) -> str:
    with open(path, "rb") as f:
        return _sha_bytes(f.read())


def _sha_tensor(tensor: torch.Tensor) -> str:
    return _sha_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError("not JSON serialisable: %r" % (type(value),))


def _stable_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k scores, with the lower numeric label winning exact ties.

    The columns are labels in ascending order.  A stable descending sort thus
    preserves that order for equal scores, unlike ``topk`` whose tie order is
    not a protocol guarantee.
    """
    if scores.ndim != 2 or not (0 < k <= scores.shape[1]):
        raise ValueError("invalid score matrix/k: %s, %s" % (tuple(scores.shape), k))
    return torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]


def _mass_tie_counts(scores: torch.Tensor, k: int) -> Dict[str, int]:
    """Count exact float64 mass ties, including the selection-boundary subset."""
    ordered = torch.sort(scores, dim=1, descending=True, stable=True).values
    adjacent_equal = ordered[:, 1:] == ordered[:, :-1]
    boundary = ordered[:, k - 1] == ordered[:, k] if k < scores.shape[1] else torch.zeros(
        scores.shape[0], dtype=torch.bool, device=scores.device)
    return {
        "tokens_with_any_exact_tie": int(adjacent_equal.any(dim=1).sum().item()),
        "adjacent_equal_pairs": int(adjacent_equal.sum().item()),
        "tokens_with_boundary_tie": int(boundary.sum().item()),
    }


def _greedy_select(v: torch.Tensor, k: int) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Greedy local squared-error minimisation from the current residual.

    ``v`` is [tokens, labels, d_model].  At each step this evaluates
    2 r.v_g - ||v_g||^2 for *unselected* labels, selects with a stable
    lower-label tie break, then subtracts the chosen contribution from r.
    """
    if v.ndim != 3:
        raise ValueError("v must be [tokens, labels, hidden]")
    n_tok, n_labels, _ = v.shape
    if not (0 < k <= n_labels):
        raise ValueError("invalid greedy k")
    residual = v.sum(dim=1)
    selected = torch.zeros(n_tok, n_labels, dtype=torch.bool, device=v.device)
    choices = []
    tie_tokens = 0
    tie_competitors = 0
    norm_sq = v.square().sum(dim=2)
    for _ in range(k):
        scores = 2.0 * (residual.unsqueeze(1) * v).sum(dim=2) - norm_sq
        scores = scores.masked_fill(selected, -torch.inf)
        pick = _stable_topk(scores, 1).squeeze(1)
        picked_score = scores.gather(1, pick[:, None])
        equal_available = (scores == picked_score) & ~selected
        n_equal = equal_available.sum(dim=1)
        tie_tokens += int((n_equal > 1).sum().item())
        tie_competitors += int((n_equal - 1).sum().item())
        selected.scatter_(1, pick[:, None], True)
        residual = residual - v[torch.arange(n_tok, device=v.device), pick]
        choices.append(pick)
    return torch.stack(choices, dim=1), {
        "selection_tie_token_steps": tie_tokens,
        "selection_tie_extra_competitors": tie_competitors,
    }


def _selected_output(v: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum selected [token, label] contributions without changing label order."""
    return v.gather(1, labels.unsqueeze(-1).expand(-1, -1, v.shape[2])).sum(dim=1)


@dataclass
class LayerMetrics:
    """SSE/energy accounting with the per-sequence values mandated by E67."""

    n_seq: int
    arms: Iterable[str] = ARMS
    error_sum: Dict[str, float] = field(init=False)
    energy_sum: float = 0.0
    sequence_error: Dict[str, np.ndarray] = field(init=False)
    sequence_energy: np.ndarray = field(init=False)
    token_relative_sse: Dict[str, List[float]] = field(init=False)

    def __post_init__(self):
        self.error_sum = {arm: 0.0 for arm in self.arms}
        self.sequence_error = {arm: np.zeros(self.n_seq, dtype=np.float64) for arm in self.arms}
        self.sequence_energy = np.zeros(self.n_seq, dtype=np.float64)
        self.token_relative_sse = {arm: [] for arm in self.arms}

    def add(self, arm: str, y: torch.Tensor, yhat: torch.Tensor, sequence: int) -> torch.Tensor:
        err = (y.double() - yhat.double()).square().sum(dim=1)
        energy = y.double().square().sum(dim=1)
        if torch.any(energy <= 0):
            raise RuntimeError("zero-norm FFN output prevents relative-error accounting")
        err_np = err.detach().cpu().numpy()
        energy_np = energy.detach().cpu().numpy()
        self.error_sum[arm] += float(err_np.sum())
        self.sequence_error[arm][sequence] += float(err_np.sum())
        if arm == ARMS[0]:
            self.energy_sum += float(energy_np.sum())
            self.sequence_energy[sequence] += float(energy_np.sum())
        self.token_relative_sse[arm].extend((err_np / energy_np).tolist())
        return err

    def result(self) -> Dict[str, object]:
        if self.energy_sum <= 0 or np.any(self.sequence_energy <= 0):
            raise RuntimeError("invalid zero energy in layer accounting")
        rows = {}
        for arm in self.arms:
            rows[arm] = {
                "sum_sse_over_sum_energy": self.error_sum[arm] / self.energy_sum,
                "median_token_sse_over_energy": float(np.median(self.token_relative_sse[arm])),
                "per_sequence_sum_sse_over_sum_energy": (
                    self.sequence_error[arm] / self.sequence_energy
                ).tolist(),
                "sum_sse": self.error_sum[arm],
            }
        return rows


class _RSS:
    """Process peak RSS observation; Windows exposes peak_wset via psutil."""

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


class LayerAnalyzer:
    """Consumes intact down-projection inputs for one E67 layer at a time."""

    def __init__(self, layer: int, down_weight: torch.Tensor, labels: torch.Tensor,
                 null_labels: torch.Tensor, n_seq: int, rss: _RSS):
        self.layer = layer
        self.weight = down_weight.detach()
        self.labels = labels.detach().long()
        self.null_labels = null_labels.detach().long()
        self.n_seq = n_seq
        self.rss = rss
        self.metrics = LayerMetrics(n_seq)
        self.mass_ties = defaultdict(int)
        self.greedy_ties = defaultdict(int)
        self.random_greedy_ties = defaultdict(int)
        self.reconstruction = {"tokens": 0, "max_relative_rms": 0.0,
                               "dense_identity_max_relative_rms": 0.0}
        self.selection_checks = {arm: {"tokens": 0, "all_distinct": True,
                                       "all_groups_size_35": True, "ffn_weights": set()}
                                 for arm in SPARSE_ARMS}
        self.dense_labels_checked = 0
        self.greedy_vs_mass = {"wins": 0, "losses": 0, "ties": 0}

        self._validate_labels(self.labels, "D0c")
        self._validate_labels(self.null_labels, "D0c random null")

    @staticmethod
    def _validate_labels(labels: torch.Tensor, name: str) -> None:
        if labels.ndim != 1 or labels.numel() != E_GROUPS * GROUP_NEURONS:
            raise RuntimeError("%s labels have wrong FFN dimension: %s" % (name, tuple(labels.shape)))
        if labels.dtype not in (torch.int64, torch.int32):
            raise RuntimeError("%s labels are not integral" % name)
        counts = torch.bincount(labels.long(), minlength=E_GROUPS)
        if counts.numel() != E_GROUPS or not torch.equal(counts, torch.full_like(counts, GROUP_NEURONS)):
            raise RuntimeError("%s labels are not 256 groups of 35" % name)

    def _contributions(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return v_g = W_down[:, label g] @ z_g as [token, group, hidden]."""
        order = torch.argsort(labels, stable=True)
        grouped_z = z[:, order].reshape(z.shape[0], E_GROUPS, GROUP_NEURONS)
        grouped_w = self.weight[:, order].reshape(self.weight.shape[0], E_GROUPS, GROUP_NEURONS)
        # einsum is equivalent to a separate down projection for every group, but only materialises
        # [token_chunk, 256, hidden], never all 512 sampled tokens at once.
        return torch.einsum("teg,deg->ted", grouped_z, grouped_w)

    def _check_sparse_selection(self, arm: str, selected: torch.Tensor, labels: torch.Tensor) -> None:
        check = self.selection_checks[arm]
        check["tokens"] += int(selected.shape[0])
        check["all_distinct"] &= bool(torch.all(selected.sort(dim=1).values[:, 1:] !=
                                                  selected.sort(dim=1).values[:, :-1]))
        sizes = torch.bincount(labels, minlength=E_GROUPS)[selected]
        check["all_groups_size_35"] &= bool(torch.all(sizes == GROUP_NEURONS))
        check["ffn_weights"].update(int(v) for v in sizes.sum(dim=1).unique().tolist())

    def process(self, z_all: torch.Tensor, sequence: int) -> None:
        if z_all.ndim != 2 or z_all.shape[1] != self.labels.numel():
            raise RuntimeError("layer %d down input dimension mismatch: %s" %
                               (self.layer, tuple(z_all.shape)))
        if z_all.shape[0] != POSITION_STOP - POSITION_START:
            raise RuntimeError("layer %d captured %d selected positions, expected %d" %
                               (self.layer, z_all.shape[0], POSITION_STOP - POSITION_START))
        for start in range(0, z_all.shape[0], TOKEN_CHUNK):
            z = z_all[start:start + TOKEN_CHUNK]
            # E41 §0 / §9 repair: this is intentionally float64, including the group reduction.
            mass = torch.zeros(z.shape[0], E_GROUPS, dtype=torch.float64, device=z.device)
            mass.index_add_(1, self.labels, z.double().square())
            mass_selected = _stable_topk(mass, K)
            for key, value in _mass_tie_counts(mass, K).items():
                self.mass_ties[key] += value

            v = self._contributions(z, self.labels)
            y = F.linear(z, self.weight)
            reconstructed = v.sum(dim=1)
            denom = y.double().square().sum(dim=1).sqrt()
            rel_rms = (reconstructed.double() - y.double()).square().sum(dim=1).sqrt() / denom
            max_rel = float(rel_rms.max().item())
            self.reconstruction["tokens"] += int(z.shape[0])
            self.reconstruction["max_relative_rms"] = max(self.reconstruction["max_relative_rms"], max_rel)
            self.reconstruction["dense_identity_max_relative_rms"] = max(
                self.reconstruction["dense_identity_max_relative_rms"], max_rel)
            if max_rel >= RECON_REL_RMS_TOL:
                raise RuntimeError("G-E67 reconstruction failed at layer %d: %.3e >= %.3e" %
                                   (self.layer, max_rel, RECON_REL_RMS_TOL))

            norm_selected = _stable_topk(v.square().sum(dim=2), K)
            greedy_selected, greedy_ties = _greedy_select(v, K)
            v_null = self._contributions(z, self.null_labels)
            null_selected, null_greedy_ties = _greedy_select(v_null, K)
            all_labels = torch.arange(E_GROUPS, device=z.device).expand(z.shape[0], E_GROUPS)

            selected = {
                "mass": (mass_selected, self.labels, v),
                "output_norm": (norm_selected, self.labels, v),
                "output_greedy": (greedy_selected, self.labels, v),
                "random_label_greedy": (null_selected, self.null_labels, v_null),
            }
            for arm, (choice, arm_labels, contribution) in selected.items():
                self._check_sparse_selection(arm, choice, arm_labels)
                self.metrics.add(arm, y, _selected_output(contribution, choice), sequence)
            mass_error = self.metrics.add("dense_identity", y, _selected_output(v, all_labels), sequence)
            # Dense identity is additionally tested against the direct projection above, before metrics.
            self.dense_labels_checked += int(all_labels.shape[0])

            # Recompute just the two compared errors for exact per-token win/loss accounting.
            mass_yhat = _selected_output(v, mass_selected)
            greedy_yhat = _selected_output(v, greedy_selected)
            mass_sse = (y.double() - mass_yhat.double()).square().sum(dim=1)
            greedy_sse = (y.double() - greedy_yhat.double()).square().sum(dim=1)
            self.greedy_vs_mass["wins"] += int((greedy_sse < mass_sse).sum().item())
            self.greedy_vs_mass["losses"] += int((greedy_sse > mass_sse).sum().item())
            self.greedy_vs_mass["ties"] += int((greedy_sse == mass_sse).sum().item())
            for key, value in greedy_ties.items():
                self.greedy_ties[key] += value
            for key, value in null_greedy_ties.items():
                self.random_greedy_ties[key] += value
            self.rss.observe()

    def close(self) -> Dict[str, object]:
        for arm, check in self.selection_checks.items():
            if not (check["all_distinct"] and check["all_groups_size_35"] and check["ffn_weights"] == {K * GROUP_NEURONS}):
                raise RuntimeError("G-E67 selection control failed at layer %d, arm %s: %s" %
                                   (self.layer, arm, check))
            check["ffn_weights"] = sorted(check["ffn_weights"])
        expected = self.n_seq * (POSITION_STOP - POSITION_START)
        if self.reconstruction["tokens"] != expected or self.dense_labels_checked != expected:
            raise RuntimeError("layer %d did not account for all selected tokens" % self.layer)
        return {
            "selectors": self.metrics.result(),
            "mass_ties_float64": dict(self.mass_ties),
            "output_greedy_ties": dict(self.greedy_ties),
            "random_label_output_greedy_ties": dict(self.random_greedy_ties),
            "reconstruction": dict(self.reconstruction, tolerance=RECON_REL_RMS_TOL, fires=True),
            "selection_controls": self.selection_checks,
            "dense_identity": {"labels_per_token": E_GROUPS, "fires": True},
            "output_greedy_vs_mass_per_token": self.greedy_vs_mass,
        }


class Capture:
    """Captures intact z at down_proj pre-hooks and analyses only E67's five layers."""

    def __init__(self, model: torch.nn.Module, learned: Dict[int, torch.Tensor],
                 random_labels: Dict[int, torch.Tensor], layers: Iterable[int], n_seq: int, rss: _RSS):
        self.model = model
        self.layers = tuple(layers)
        self.sequence = None
        self.analyzers = {
            layer: LayerAnalyzer(layer, model.model.layers[layer].mlp.down_proj.weight,
                                 learned[layer], random_labels[layer], n_seq, rss)
            for layer in self.layers
        }
        self.handles = [model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(
            self._hook(layer)) for layer in self.layers]

    def _hook(self, layer: int):
        def capture(_module, args):
            if self.sequence is None:
                return None
            z = args[0].detach()
            if z.ndim != 3 or z.shape[0] != 1 or z.shape[1] != SEQLEN:
                raise RuntimeError("unexpected layer %d down input shape %s" % (layer, tuple(z.shape)))
            self.analyzers[layer].process(z[0, POSITION_START:POSITION_STOP], self.sequence)
            return None
        return capture

    def close(self) -> Dict[str, object]:
        for handle in self.handles:
            handle.remove()
        return {str(layer): analyzer.close() for layer, analyzer in self.analyzers.items()}


def _load_labels(n_layers: int) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], str]:
    """Read D0c's learned and stored random equal partitions; never refit either."""
    import e19_carve_rank as E19

    learned_np, random_np = E19.load_labels(E_GROUPS)
    expected = set(range(n_layers))
    if set(learned_np) != expected or set(random_np) != expected:
        raise RuntimeError("label file does not name exactly all model layers")
    label_path = os.path.join(DENSITY, "results", "d0c_labels", "labels_E256.npz")
    if not os.path.exists(label_path):
        raise RuntimeError("missing frozen D0c labels: %s" % label_path)
    learned = {layer: torch.from_numpy(np.asarray(learned_np[layer], dtype=np.int64)).contiguous()
               for layer in range(n_layers)}
    random_labels = {layer: torch.from_numpy(np.asarray(random_np[layer], dtype=np.int64)).contiguous()
                     for layer in range(n_layers)}
    return learned, random_labels, label_path


def _selftest() -> int:
    """Pure-torch planted controls: unequal columns, cancellation, and lower-label ties."""
    # Brief's counterexample: z=(10,1), W_down=(.001,1), k=1.
    z = torch.tensor([[10.0, 1.0]])
    w = torch.tensor([[0.001, 1.0]])
    v = torch.stack((z[:, 0] * w[:, 0], z[:, 1] * w[:, 1]), dim=1).unsqueeze(-1)
    mass = z.double().square()
    mass_choice = _stable_topk(mass, 1)
    output_choice = _stable_topk(v.square().sum(dim=2), 1)
    y = v.sum(dim=1)
    mass_error = float((y - _selected_output(v, mass_choice)).square().sum())
    output_error = float((y - _selected_output(v, output_choice)).square().sum())
    assert mass_choice.tolist() == [[0]] and output_choice.tolist() == [[1]]
    assert abs(mass_error - 1.0) < 1e-7 and abs(output_error - 1e-4) < 1e-8

    # The +/- x contributions cancel.  After choosing label 0, current residual is (0,2),
    # so true greedy chooses label 2; a stale, one-shot score would choose label 1 instead.
    cancellation_v = torch.tensor([[[5.0, 0.0], [1.0, 0.0], [0.0, 2.0], [-1.0, 0.0]]])
    greedy, _ = _greedy_select(cancellation_v, 2)
    initial_y = cancellation_v.sum(dim=1)
    stale_scores = 2 * (initial_y.unsqueeze(1) * cancellation_v).sum(dim=2) - cancellation_v.square().sum(dim=2)
    stale_first = _stable_topk(stale_scores, 1)
    stale_scores.scatter_(1, stale_first, -torch.inf)
    stale_second = _stable_topk(stale_scores, 1)
    assert greedy.tolist() == [[0, 2]], greedy.tolist()
    assert torch.cat((stale_first, stale_second), dim=1).tolist() == [[0, 1]]

    tied = torch.tensor([[9.0, 9.0, 1.0]], dtype=torch.float64)
    assert _stable_topk(tied, 1).tolist() == [[0]]
    assert _mass_tie_counts(tied, 1)["tokens_with_boundary_tie"] == 1
    print("E67 selftest: PASS")
    print("  unequal-column: mass=[0] SSE=1.000000; output-aware=[1] SSE=0.000100")
    print("  cancellation: current-residual greedy=[0, 2]; stale-score control=[0, 1]")
    print("  float64 tie: lower label 0 selected; boundary ties=1")
    return 0


def _aggregate_ratio(layers: Dict[str, object], numerator: str, denominator: str) -> float:
    total_num = sum(layers[str(layer)]["selectors"][numerator]["sum_sse"] for layer in layers)
    total_den = sum(layers[str(layer)]["selectors"][denominator]["sum_sse"] for layer in layers)
    if total_den <= 0:
        raise RuntimeError("cannot form E67 aggregate ratio from zero mass SSE")
    return total_num / total_den


def _stopping_band(layers: Dict[str, object], smoke: bool) -> Dict[str, object]:
    """Apply E67's frozen local-only stopping rule; never infer a BPB result."""
    if smoke:
        return {"name": "NOT_APPLICABLE_SMOKE",
                "reason": "the frozen bands require all five registered layers"}
    per_layer = {}
    for layer, result in layers.items():
        mass_sse = result["selectors"]["mass"]["sum_sse"]
        greedy_sse = result["selectors"]["output_greedy"]["sum_sse"]
        if mass_sse <= 0:
            raise RuntimeError("cannot apply E67 stopping band with zero mass SSE")
        per_layer[layer] = {"mass_to_output_greedy_sse_ratio": greedy_sse / mass_sse,
                            "sse_reduction": 1.0 - greedy_sse / mass_sse}
    aggregate_ratio = _aggregate_ratio(layers, "output_greedy", "mass")
    aggregate_reduction = 1.0 - aggregate_ratio
    reductions = [row["sse_reduction"] for row in per_layer.values()]
    strong = (aggregate_reduction >= 0.50 and sum(value >= 0.50 for value in reductions) >= 4
              and all(value >= -0.20 for value in reductions))
    if strong:
        name = "STRONG_LOCAL_SIGNAL"
    elif aggregate_reduction >= 0.20:
        name = "PARTIAL_LOCAL_SIGNAL"
    else:
        name = "NO_LOCAL_SIGNAL"
    return {"name": name, "aggregate_sse_reduction": aggregate_reduction,
            "per_layer": per_layer,
            "strong_gate": {"aggregate_reduction_at_least_50pct": aggregate_reduction >= 0.50,
                            "layers_at_least_50pct": sum(value >= 0.50 for value in reductions),
                            "no_layer_worse_by_more_than_20pct": all(value >= -0.20 for value in reductions)},
            "scope": "local FFN reconstruction only; this band does not promote a donor or assert BPB/rate."}


def run(smoke: bool = False) -> str:
    """Run the frozen local diagnostic.  This intentionally loads the pinned donor."""
    import common as C

    _selftest()  # planted controls are mandatory, including on the real run
    torch.set_num_threads(THREADS)
    torch.set_grad_enabled(False)
    started = time.time()
    rss = _RSS()
    rss.observe()
    model, tokenizer = C.load_model(dtype=torch.float32)
    model.eval()
    architecture = C.arch(model)
    if architecture["revision"] != REV or architecture["d_ffn"] != E_GROUPS * GROUP_NEURONS:
        raise RuntimeError("pinned donor architecture mismatch: %s" % architecture)
    if architecture["n_layers"] <= max(LAYERS):
        raise RuntimeError("donor lacks required E67 layers")

    ids, _bytes, slice_meta = C.get_slice(tokenizer, "heldout", N_SEQ, SEQLEN, SEED)
    if slice_meta["ids_sha256"] != EXPECT_IDS_SHA:
        raise RuntimeError("SLICE HASH MISMATCH -- STOP: %s" % slice_meta["ids_sha256"])
    selected_sequences = [0] if smoke else list(range(4))
    run_layers = (LAYERS[-1],) if smoke else LAYERS
    selected_ids = ids[selected_sequences, POSITION_START:POSITION_STOP].contiguous()
    selected_ids_sha = _sha_tensor(selected_ids)
    learned, random_labels, label_path = _load_labels(architecture["n_layers"])
    label_hashes = {
        "labels_E256_npz_sha256": _sha_file(label_path),
        "learned_per_layer": {str(layer): _sha_tensor(learned[layer]) for layer in run_layers},
        "random_equal_partition_per_layer": {str(layer): _sha_tensor(random_labels[layer]) for layer in run_layers},
    }
    capture = Capture(model, learned, random_labels, run_layers, len(selected_sequences), rss)
    try:
        with torch.inference_mode():
            for local_sequence, source_sequence in enumerate(selected_sequences):
                capture.sequence = local_sequence
                model.model(ids[source_sequence:source_sequence + 1])
                print("E67 sequence %d/%d analysed" % (local_sequence + 1, len(selected_sequences)), flush=True)
                rss.observe()
    finally:
        capture.sequence = None
        layer_results = capture.close()
    rss.observe()

    greedy_over_mass = _aggregate_ratio(layer_results, "output_greedy", "mass")
    mass_to_greedy_reduction = 1.0 - greedy_over_mass
    source_hashes = {
        "runner_sha256": _sha_file(__file__),
        "density_common_sha256": _sha_file(C.__file__),
        "e19_carve_rank_sha256": _sha_file(os.path.join(DENSITY, "e19_carve_rank.py")),
    }
    output = {
        "brief": "BRIEF_E67_OUTPUT_AWARE_SELECTION.md",
        "scope": "local FFN-output squared-error diagnostic only; no BPB, generation, or speed claim",
        "planted_selftest": "PASS",
        "selector_charge": "Every sparse selector computes all dense group contributions first; "
                           "these are capacity diagnostics, not executable routers and are not priced at E63 mixed-byte rate.",
        "donor": {"model": HF, "revision": REV, "precision": "fp32", "attention": "eager"},
        "threads": THREADS,
        "smoke": smoke,
        "architecture": architecture,
        "slice": {
            "part": "heldout", "full_n_sequences": N_SEQ, "sequence_length": SEQLEN, "seed": SEED,
            "full_ids_sha256": slice_meta["ids_sha256"], "expected_full_ids_sha256": EXPECT_IDS_SHA,
            "selected_sequences": selected_sequences, "selected_positions": [POSITION_START, POSITION_STOP - 1],
            "selected_ids_sha256": selected_ids_sha, "selected_token_count": int(selected_ids.numel()),
        },
        "labels": {"path": label_path, "E": E_GROUPS, "neurons_per_group": GROUP_NEURONS,
                   "hashes": label_hashes},
        "source_hashes": source_hashes,
        "protocol": {
            "layers": list(run_layers), "k": K, "token_chunk": TOKEN_CHUNK,
            "mass": "float64 sum(z_i^2), stable descending order, lower label wins exact ties",
            "output_greedy": "per step score = 2*r.dot(v_g)-||v_g||^2; r -= selected v_g; lower label wins exact ties",
            "aggregate_metric": "sum ||y-yhat||^2 / sum ||y||^2",
            "median_metric": "median per-token ||y-yhat||^2 / ||y||^2",
        },
        "layers": layer_results,
        "mass_to_output_greedy": {
            "summed_sse_ratio": greedy_over_mass,
            "summed_sse_reduction": mass_to_greedy_reduction,
        },
        "stopping_band": _stopping_band(layer_results, smoke),
        "operational_observations_only": {"elapsed_seconds": time.time() - started,
                                            "peak_rss_bytes": rss.peak, "peak_rss_method": rss.method},
    }
    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, "e67_output_aware_selection%s.json" % ("_smoke" if smoke else ""))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=_json_default)
    print("wrote %s" % out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="run planted controls only; never load donor")
    parser.add_argument("--smoke", action="store_true", help="one sequence and one layer; still loads donor")
    args = parser.parse_args()
    if args.selftest and args.smoke:
        parser.error("--selftest and --smoke are mutually exclusive")
    if args.selftest:
        return _selftest()
    run(smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
