"""Empirical downstream Fisher geometry, with independent held-out probes.

Only run tensor/model operations on Kaggle. Public logits_with_delta is shared
by the other two sessions. This module performs no work when imported.
"""
from __future__ import annotations

import json
from pathlib import Path
import time


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def _finite(torch, value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Nonfinite {name}")


def _event(output, **value):
    print("geometry " + json.dumps(value, allow_nan=False), flush=True)
    with (Path(output) / "geometry_events.jsonl").open("a", encoding="utf8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def logits_with_delta(torch, donor, ids, delta, position, layer_end, query_positions=None):
    """Inject delta[D] once at one source-span output position.

    Returns [1,T,V], or [1,Q,V] when query_positions is supplied. Token IDs
    contain the actual teacher-forced continuation, never generated history.
    The additive tensor stays on the autograd graph; donor parameters should
    have requires_grad=False. Hooks are removed even when forward fails.
    """
    if ids.ndim != 2 or ids.shape[0] != 1 or delta.ndim != 1:
        raise ValueError("Expected ids[1,T] and delta[D]")
    if not 0 <= position < ids.shape[1] or not 0 < layer_end <= len(donor.model.layers):
        raise ValueError("Injection position/layer is outside the source")
    if query_positions is not None and any(not 0 <= p < ids.shape[1] for p in query_positions):
        raise ValueError("Query position is outside the sequence")

    def inject(module, args, value):
        hidden = value[0] if isinstance(value, tuple) else value
        if hidden.shape[-1] != delta.shape[0]:
            raise ValueError("Delta width does not match source hidden width")
        selector = torch.arange(hidden.shape[1], device=hidden.device) == position
        addition = selector[None, :, None] * delta.to(device=hidden.device, dtype=hidden.dtype)[None, None, :]
        perturbed = hidden + addition
        return (perturbed,) + value[1:] if isinstance(value, tuple) else perturbed

    handle = donor.model.layers[layer_end - 1].register_forward_hook(inject)
    try:
        hidden = donor.model(ids, use_cache=False, return_dict=True).last_hidden_state
        if query_positions is not None:
            hidden = hidden[:, query_positions]
        return donor.lm_head(hidden).float()
    finally:
        handle.remove()


def _score_gradients(torch, donor, documents, cfg, output, budget, device, split, samples):
    anchor = cfg.get("anchor", 63)
    queries = [anchor, anchor + 4, anchor + 16]
    dim = donor.config.hidden_size
    gradients, metadata = [], []
    split_seed = {"train": 0, "dev": 1000000, "test": 2000000}[split]
    for document, window in enumerate(documents):
        budget.check()
        ids = torch.tensor([window[:-1]], device=device)
        delta = torch.zeros(dim, device=device, dtype=torch.float32, requires_grad=True)
        logits = logits_with_delta(torch, donor, ids, delta, anchor, cfg["layer_end"], queries)
        _finite(torch, logits, "unperturbed source logits")
        logp = logits.log_softmax(dim=-1)
        probabilities = logp.detach().exp()
        generator = torch.Generator(device=device).manual_seed(cfg["seed"] + split_seed + document)
        # Multinomial samples come from the complete exact teacher distribution;
        # there is no top-k truncation, argmax label, or data-dependent resampling.
        chosen = [torch.multinomial(probabilities[0, q], samples, replacement=True,
                                    generator=generator).tolist() for q in range(len(queries))]
        for qi, query in enumerate(queries):
            for sample, token in enumerate(chosen[qi]):
                budget.check()
                last = qi == len(queries) - 1 and sample == samples - 1
                gradient = torch.autograd.grad(logp[0, qi, token], delta, retain_graph=not last)[0]
                _finite(torch, gradient, "categorical score gradient")
                gradients.append(gradient.detach().cpu())
                metadata.append(dict(document=document, query=query, sample=sample, token_id=token,
                                     teacher_probability=probabilities[0, qi, token].item(),
                                     gradient_norm=gradient.detach().norm().item()))
        del logits, logp, probabilities, delta
        _event(output, phase="gradient_capture", split=split, document=document + 1,
               total_documents=len(documents), captured_gradients=len(gradients))
    result = dict(gradients=torch.stack(gradients), metadata=metadata, queries=queries,
                  samples=samples, documents=len(documents))
    if result["gradients"].square().sum().item() <= 0:
        raise RuntimeError(f"All {split} score gradients are zero")
    torch.save(result, Path(output) / f"geometry_gradients_{split}.pt")
    return result


def _finite_difference(torch, donor, documents, fit, records, cfg, output, budget, device):
    checks = []
    for document, query in ((0, cfg.get("anchor", 63)), (1, cfg.get("anchor", 63) + 16)):
        budget.check()
        index = next(i for i, meta in enumerate(fit["metadata"])
                     if meta["document"] == document and meta["query"] == query and meta["sample"] == 0)
        token = fit["metadata"][index]["token_id"]
        generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 3000000 + document)
        direction = torch.randn(donor.config.hidden_size, generator=generator, device=device)
        direction /= direction.norm()
        hidden_norm = records["train"]["y"][document, cfg.get("anchor", 63)].float().norm().item()
        epsilon = 0.005 * hidden_norm
        if epsilon <= 0:
            raise RuntimeError("Zero source hidden norm in derivative check")
        with torch.no_grad():
            plus = logits_with_delta(torch, donor, torch.tensor([documents[document][:-1]], device=device),
                                     direction * epsilon, cfg.get("anchor", 63), cfg["layer_end"], [query])
            minus = logits_with_delta(torch, donor, torch.tensor([documents[document][:-1]], device=device),
                                      -direction * epsilon, cfg.get("anchor", 63), cfg["layer_end"], [query])
            # Double precision readout arithmetic avoids cancellation in logp.
            difference = (plus.double().log_softmax(-1)[0, 0, token] -
                          minus.double().log_softmax(-1)[0, 0, token]).item() / (2 * epsilon)
        analytic = (fit["gradients"][index].to(device) * direction).sum().item()
        error = abs(difference - analytic)
        passed = error <= max(1e-4, 0.10 * abs(analytic))
        checks.append(dict(document=document, query=query, token_id=token, epsilon=epsilon,
                           analytic=analytic, finite_difference=difference, absolute_error=error,
                           pass_check=passed, tolerance="max(1e-4,0.10*abs(analytic))"))
    _json(Path(output) / "geometry_derivative_checks.json", checks)
    if not all(row["pass_check"] for row in checks):
        raise RuntimeError("Directional finite-difference derivative validation failed")
    return checks


def _bases(torch, fit, records, cfg, output, budget, device):
    budget.check()
    gradients = fit["gradients"].to(device=device, dtype=torch.float32)
    _, singular, vh = torch.linalg.svd(gradients, full_matrices=False)
    # The score mean is not subtracted: empirical Fisher is E[g g^T].
    tolerance = max(gradients.shape) * torch.finfo(gradients.dtype).eps * singular[0]
    numerical_rank = int((singular > tolerance).sum().item())
    ranks = cfg.get("ranks", [8, 32, 128])
    # PCA uses FIT source output states, including multiple causal positions per
    # document. It never sees DEV/TEST states or their gradients while fitting.
    activation = records["train"]["y"][:, cfg["prefix"]:].reshape(-1, gradients.shape[-1]).to(device).float()
    center = activation.mean(0)
    activation = activation - center
    _, activation_singular, activation_vh = torch.linalg.svd(activation, full_matrices=False)
    activation_tolerance = max(activation.shape) * torch.finfo(activation.dtype).eps * activation_singular[0]
    pca_numerical_rank = int((activation_singular > activation_tolerance).sum().item())
    valid_ranks = [rank for rank in ranks if rank <= min(numerical_rank, pca_numerical_rank)]
    if not valid_ranks:
        raise RuntimeError("No requested matched rank is supported by both FIT Fisher and PCA")
    maximum = max(valid_ranks)
    fisher = vh[:maximum].T.contiguous()
    pca = activation_vh[:maximum].T.contiguous()
    generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 4000000)
    random_basis = torch.linalg.qr(torch.randn(gradients.shape[-1], maximum, device=device,
                                             generator=generator), mode="reduced").Q
    bases = dict(fisher=fisher, pca=pca, random=random_basis)
    for name, basis in bases.items():
        _finite(torch, basis, name + " basis")
        orthogonal_error = (basis.T @ basis - torch.eye(maximum, device=device)).abs().max().item()
        if orthogonal_error > 1e-4:
            raise RuntimeError(f"Nonorthogonal {name} basis: {orthogonal_error}")
    torch.save(dict(bases={name: value.cpu() for name, value in bases.items()},
                    fisher_singular_values=singular.cpu(), pca_singular_values=activation_singular.cpu(),
                    pca_center=center.cpu(), numerical_rank=numerical_rank, valid_ranks=valid_ranks),
               Path(output) / "geometry_fit_bases.pt")
    receipt = dict(fit_gradient_count=len(gradients), fisher_numerical_rank=numerical_rank,
                   pca_numerical_rank=pca_numerical_rank,
                   requested_ranks=ranks, valid_ranks=valid_ranks,
                   unsupported_ranks=[rank for rank in ranks if rank not in valid_ranks],
                   covariance="uncentered equal-weight score outer products, joint teacher-forced queries",
                   queries=fit["queries"],
                   rank_limit_warning="FIT rank is at most the number of sampled scores; only independent held-out probes assess concentration",
                   pca_fit_states=len(activation), fit_documents=fit["documents"])
    _json(Path(output) / "geometry_basis_receipt.json", receipt)
    return bases, valid_ranks, receipt


def _bootstrap(torch, numerator, denominator, seed):
    numerator = torch.tensor(numerator, dtype=torch.float64)
    denominator = torch.tensor(denominator, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(numerator), (2000, len(numerator)), generator=generator)
    ratios = numerator[indices].sum(1) / denominator[indices].sum(1).clamp(min=1e-30)
    return [torch.quantile(ratios, q).item() for q in (0.025, 0.5, 0.975)]


def _projection_evaluation(torch, probes, bases, ranks, cfg, device):
    gradient = probes["gradients"].to(device)
    totals = gradient.square().sum(-1)
    result = {}
    for query in probes["queries"]:
        selected = [i for i, meta in enumerate(probes["metadata"]) if meta["query"] == query]
        selected_totals = totals[selected]
        per_doc_total = []
        for document in range(probes["documents"]):
            doc_indices = [i for i in selected if probes["metadata"][i]["document"] == document]
            per_doc_total.append(totals[doc_indices].sum().item())
        query_result = {}
        projected_by_basis = {}
        for name, basis in bases.items():
            rank_result = {}
            for rank in ranks:
                energy = (gradient @ basis[:, :rank]).square().sum(-1)
                if bool((energy > totals * 1.0001 + 1e-12).any()):
                    raise FloatingPointError("Projection energy exceeds full gradient energy")
                per_doc_energy = []
                for document in range(probes["documents"]):
                    doc_indices = [i for i in selected if probes["metadata"][i]["document"] == document]
                    per_doc_energy.append(energy[doc_indices].sum().item())
                rank_result[str(rank)] = dict(trace_fraction=energy[selected].sum().item() / selected_totals.sum().item(),
                                               ci95=_bootstrap(torch, per_doc_energy, per_doc_total, cfg["seed"]),
                                               document_projected_energy=per_doc_energy,
                                               document_total_energy=per_doc_total,
                                               zero_gradient_probes=int((selected_totals == 0).sum().item()))
                projected_by_basis[name, rank] = per_doc_energy
            query_result[name] = rank_result
        if 32 in ranks:
            for control in ("random", "pca"):
                difference = [a - b for a, b in zip(projected_by_basis["fisher", 32],
                                                    projected_by_basis[control, 32])]
                query_result["fisher_minus_" + control + "_rank32_ci95"] = _bootstrap(
                    torch, difference, per_doc_total, cfg["seed"] + 1)
        result[str(query)] = query_result
    return result


def _noise_tests(torch, donor, documents, probes, bases, records, cfg, output, budget, device, split):
    rank = 32
    if any(basis.shape[1] < rank for basis in bases.values()):
        return dict(status="UNSUPPORTED_RANK32", rows=[])
    anchor, queries = cfg.get("anchor", 63), probes["queries"]
    rows = []
    for document, window in enumerate(documents):
        budget.check()
        ids = torch.tensor([window[:-1]], device=device)
        zero = torch.zeros(donor.config.hidden_size, device=device)
        with torch.no_grad():
            teacher = logits_with_delta(torch, donor, ids, zero, anchor, cfg["layer_end"], queries)
            teacher_log = teacher.double().log_softmax(-1)
            teacher_prob = teacher_log.exp()
        generator = torch.Generator(device=device).manual_seed(
            cfg["seed"] + (5000000 if split == "dev" else 6000000) + document)
        direction = torch.randn(donor.config.hidden_size, device=device, generator=generator)
        direction /= direction.norm()
        directions = {"isotropic": direction}
        for name, basis in bases.items():
            projection = basis[:, :rank] @ (basis[:, :rank].T @ direction)
            for kind, vector in (("retained", projection), ("null", direction - projection)):
                norm = vector.norm()
                if norm <= 1e-12:
                    raise RuntimeError(f"Degenerate {name}/{kind} probe direction")
                directions[name + "_" + kind] = vector / norm
        source_norm = records[split]["y"][document, anchor].float().norm().item()
        for name, vector in directions.items():
            for ratio in (0.005, 0.02):
                for sign in (-1, 1):
                    budget.check()
                    delta = vector * (sign * ratio * source_norm)
                    with torch.no_grad():
                        candidate = logits_with_delta(torch, donor, ids, delta, anchor, cfg["layer_end"], queries)
                        logq = candidate.double().log_softmax(-1)
                        divergences = (teacher_prob * (teacher_log - logq)).sum(-1)[0]
                    _finite(torch, divergences, "finite-perturbation KL")
                    if divergences.min().item() < -1e-10:
                        raise FloatingPointError("Materially negative KL divergence")
                    for qi, query in enumerate(queries):
                        indices = [i for i, meta in enumerate(probes["metadata"])
                                   if meta["document"] == document and meta["query"] == query]
                        fresh_gradients = probes["gradients"][indices].to(device)
                        quadratic = 0.5 * (fresh_gradients @ delta).square().mean().item()
                        rows.append(dict(document=document, query=query, direction=name, rank=rank,
                                         source_norm=source_norm, delta_norm=delta.norm().item(),
                                         relative_norm=ratio, sign=sign, kl=divergences[qi].item(),
                                         independent_score_quadratic_estimate=quadratic))
        _json(Path(output) / f"geometry_noise_{split}.json", dict(status="RUNNING", rows=rows))
        _event(output, phase="finite_noise", split=split, document=document + 1,
               total_documents=len(documents), measured_query_perturbations=len(rows))
    result = dict(status="COMPLETE", rows=rows,
                  interpretation="Teacher-forced conditional distributions; equal L2 retained/null directions; signed finite perturbations",
                  no_generation_claim=True)
    _json(Path(output) / f"geometry_noise_{split}.json", result)
    return result


def _decision(projections, noise, anchor):
    primary = projections[str(anchor + 16)]
    if "32" not in primary["fisher"] or noise["status"] != "COMPLETE":
        return dict(status="INCONCLUSIVE", reason="Rank32/finite-noise target incomplete")
    capture = primary["fisher"]["32"]["trace_fraction"]
    random_advantage = primary["fisher_minus_random_rank32_ci95"][0]
    pca_advantage = primary["fisher_minus_pca_rank32_ci95"][0]
    selected = [row for row in noise["rows"] if row["query"] == anchor + 16 and row["relative_norm"] == 0.02]
    null = [row["kl"] for row in selected if row["direction"] == "fisher_null"]
    retained = [row["kl"] for row in selected if row["direction"] == "fisher_retained"]
    if not null or len(null) != len(retained):
        return dict(status="INCONCLUSIVE", reason="Missing paired finite-noise primary rows")
    ratio = sum(null) / max(sum(retained), 1e-30)
    checks = dict(trace_capture_at_least_half=capture >= 0.5,
                  positive_paired_advantage_over_random=random_advantage > 0,
                  positive_paired_advantage_over_pca=pca_advantage > 0,
                  equal_norm_null_kl_at_most_quarter_retained=ratio <= 0.25)
    return dict(status="SUPPORTED_EXPLORATORY" if all(checks.values()) else "NOT_SUPPORTED",
                primary_query=anchor + 16, primary_rank=32, checks=checks,
                heldout_trace_capture=capture, null_over_retained_kl=ratio,
                limitation="Empirical teacher-forced local predictive sensitivity; no preservation, reusable dynamics, or speed admission")


def _run(torch, donor, records, windows, cfg, output, budget, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if any(parameter.requires_grad for parameter in donor.parameters()):
        raise ValueError("Donor parameters must be frozen; only additive deltas require gradients")
    if any(parameter.dtype != torch.float32 for parameter in donor.parameters()):
        raise ValueError("Derivative geometry requires an FP32 donor")
    if len(windows["train"]) < 2:
        raise ValueError("Two distinct FIT documents are required for derivative validation")
    fit_samples = cfg.get("geometry_fit_per_query", 2)
    eval_samples = cfg.get("geometry_eval_per_query", 4)
    prospective = dict(injection_position=cfg.get("anchor", 63), layer_end=cfg["layer_end"],
                       fit_per_query=fit_samples, heldout_per_query=eval_samples,
                       ranks=cfg.get("ranks", [8, 32, 128]), noise_scales=[0.005, 0.02],
                       primary_rank=32, primary_query=cfg.get("anchor", 63) + 16,
                       primary_checks=["heldout trace>=0.5", "paired Fisher-random CI lower>0",
                                       "paired Fisher-PCA CI lower>0", "null/retained equal-norm KL<=0.25"],
                       teacher_forced_future=True, independent_heldout_scores=True)
    _json(output / "geometry_prospective.json", prospective)
    fit = _score_gradients(torch, donor, windows["train"], cfg, output, budget, device,
                           "train", fit_samples)
    expected_fit = len(windows["train"]) * 3 * fit_samples
    if len(fit["gradients"]) != expected_fit:
        raise RuntimeError("Incomplete FIT gradient capture")
    checks = _finite_difference(torch, donor, windows["train"], fit, records, cfg, output, budget, device)
    bases, ranks, basis_receipt = _bases(torch, fit, records, cfg, output, budget, device)
    reports = {}
    # DEV decision is fixed before even sampling TEST categorical labels.
    for split in ("dev", "test"):
        probes = _score_gradients(torch, donor, windows[split], cfg, output, budget, device,
                                  split, eval_samples)
        expected_heldout = len(windows[split]) * 3 * eval_samples
        if len(probes["gradients"]) != expected_heldout:
            raise RuntimeError(f"Incomplete {split} gradient capture")
        projections = _projection_evaluation(torch, probes, bases, ranks, cfg, device)
        noise = _noise_tests(torch, donor, windows[split], probes, bases, records, cfg,
                             output, budget, device, split)
        decision = _decision(projections, noise, cfg.get("anchor", 63))
        reports[split] = dict(projection=projections, decision=decision,
                              score_probe_count=len(probes["gradients"]))
        _json(output / f"geometry_{split}_decision.json", reports[split])
    report = dict(status="COMPLETE", elapsed_seconds=time.monotonic() - started,
                  dev=reports["dev"]["decision"], test_confirmation=reports["test"]["decision"],
                  basis_receipt=basis_receipt, derivative_checks=checks,
                  actual_fit_score_probes=len(fit["gradients"]),
                  actual_dev_score_probes=reports["dev"]["score_probe_count"],
                  actual_test_score_probes=reports["test"]["score_probe_count"],
                  no_learned_surrogate=True, no_generation_or_cpu_speed_claim=True)
    _json(output / "geometry_report.json", report)
    return report


def run(torch, donor, records, windows, cfg, output, budget, device):
    """Record incomplete/invalid work explicitly before propagating failure."""
    import traceback
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    _json(output / "geometry_report.json", dict(status="RUNNING"))
    try:
        return _run(torch, donor, records, windows, cfg, output, budget, device)
    except Exception as error:
        _json(output / "geometry_report.json", dict(
            status="INCONCLUSIVE", error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), elapsed_seconds=time.monotonic() - started,
            no_positive_claim=True))
        raise


def self_test(torch):
    """Remote CPU smoke test; no checkpoint, source acquisition, or training."""
    from types import SimpleNamespace

    class Layer(torch.nn.Module):
        def forward(self, hidden):
            return (hidden * 2,)

    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(), Layer()])

        def forward(self, ids, **kwargs):
            hidden = torch.nn.functional.one_hot(ids, 4).float()
            for layer in self.layers:
                hidden = layer(hidden)[0]
            return SimpleNamespace(last_hidden_state=hidden)

    class Donor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Body()
            self.lm_head = torch.nn.Identity()

    donor = Donor()
    ids = torch.tensor([[0, 1, 2]])
    delta = torch.ones(4, requires_grad=True)
    value = logits_with_delta(torch, donor, ids, delta, 1, 1)
    base = logits_with_delta(torch, donor, ids, torch.zeros(4), 1, 1)
    assert torch.equal(value[:, 0], base[:, 0])
    assert torch.equal(value[:, 2], base[:, 2])
    assert torch.equal(value[:, 1] - base[:, 1], torch.full((1, 4), 2.0))
    assert torch.equal(torch.autograd.grad(value[:, 1].sum(), delta)[0], torch.full((4,), 2.0))
    selected = logits_with_delta(torch, donor, ids, torch.zeros(4), 1, 1, [2, 0])
    assert torch.equal(selected, base[:, [2, 0]])
    assert all(len(layer._forward_hooks) == 0 for layer in donor.model.layers)
    # Orthogonal projection energy is bounded and decomposes into retained+null.
    matrix = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    q = torch.linalg.qr(torch.eye(4)[:, :2], mode="reduced").Q
    retained = (matrix @ q) @ q.T
    assert torch.allclose(matrix.square().sum(), retained.square().sum() + (matrix - retained).square().sum())
    print("geometry CPU hook/autograd/projection self-test passed", flush=True)
