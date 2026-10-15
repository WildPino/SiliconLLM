"""Cross-document reuse of downstream-sensitive subspaces.

Tensor operations are intended exclusively for the authorized Kaggle session.
Archive construction uses FIT only. Selection uses source-entry prefix states,
strictly before the injection, and no downstream states or gradient targets.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import statistics

try:
    from .geometry import logits_with_delta
except ImportError:
    from geometry import logits_with_delta


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _finite(torch, tensor):
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError("Nonfinite cross-document reuse value")


def _summary(values):
    """Two-sided Student t interval with articles as independent units."""
    values = [float(v) for v in values]
    n = len(values)
    if n < 2:
        raise ValueError("At least two independent articles are required")
    critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
                11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
                15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
                19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
                23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
                27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}
    # Larger samples use conservative df=30, avoiding an extra dependency.
    t = critical[min(n - 1, 30)]
    mean = statistics.mean(values)
    radius = t * statistics.stdev(values) / math.sqrt(n)
    return {"n_articles": n, "mean": mean, "ci95": [mean - radius, mean + radius],
            "interval": "paired article-level Student t for differences; descriptive, unadjusted"}


def _basis(torch, gradients, rank):
    _finite(torch, gradients)
    if gradients.ndim != 2 or min(gradients.shape) < rank:
        raise ValueError("Insufficient gradient dimensions for requested rank")
    _, singular, vh = torch.linalg.svd(gradients, full_matrices=False)
    if singular[0].item() <= 0:
        raise ValueError("Zero FIT sensitivity bank")
    # Repeated categorical samples can make the empirical bank rank deficient.
    # SVD supplies orthonormal completion so every method retains rank8 storage;
    # effective rank is reported rather than silently claiming eight learned modes.
    return vh[:rank].T.contiguous()


def _energy(torch, gradients, basis):
    total = gradients.square().sum()
    if not total.item() > 0:
        raise ValueError("Zero gradient energy cannot define captured fraction")
    value = (gradients @ basis).square().sum() / total
    _finite(torch, value)
    if not -1e-5 <= value.item() <= 1 + 1e-5:
        raise ValueError("Projection energy outside orthogonal-projection bounds")
    return float(value.clamp(0, 1).item())


def _key(torch, x, anchor, device):
    # End is exclusive: neither the anchor token nor source span output is read.
    key = x[anchor - 32:anchor].to(device=device, dtype=torch.float32).mean(0)
    _finite(torch, key)
    norm = key.norm()
    if norm.item() <= 1e-12:
        raise ValueError("Zero causal prefix key")
    return key / norm


def _probes(torch, donor, ids, width, count, seed, anchor, queries,
            layer_end, budget, device):
    """Unbiased Monte Carlo categorical score probes at delta=0.

    Each row is grad_delta log p(token | teacher-forced prefix), with token
    sampled independently from the frozen source distribution for that query.
    A single graph is reused for the small probe bank, then fully released.
    """
    budget.check()
    generator = torch.Generator(device=device).manual_seed(seed)
    delta = torch.zeros(width, device=device, dtype=torch.float32, requires_grad=True)
    with torch.enable_grad():
        logits = logits_with_delta(torch, donor, ids, delta, anchor, layer_end,
                                   query_positions=list(queries))[0]
        log_probs = torch.log_softmax(logits, -1)
        result, sampled_tokens = [], []
        for qi in range(len(queries)):
            tokens = torch.multinomial(log_probs[qi].detach().exp(), count,
                                       replacement=True, generator=generator)
            sampled_tokens.append(tokens.detach().cpu())
            gradients = []
            for k, token in enumerate(tokens):
                budget.check()
                last = qi == len(queries) - 1 and k == count - 1
                gradient, = torch.autograd.grad(log_probs[qi, token], delta,
                                                 retain_graph=not last)
                gradients.append(gradient.detach())
            result.append(torch.stack(gradients))
    for gradients in result:
        _finite(torch, gradients)
    return result, sampled_tokens


def run(torch, donor, records, windows, cfg, output, budget, device):
    """Run the predeclared rank-eight FIT archive reuse experiment."""
    anchor, queries, rank, fit_k, eval_k = 63, (63, 79), 8, 16, 8
    seed = int(cfg.get("seed", 20261015)) + 32000
    layer_end = int(cfg.get("layer_end", cfg.get("source_end", 12)))
    names = {name: next((k for k in records if k.lower() == name or
                        (name == "fit" and k.lower() == "train")), None)
             for name in ("fit", "dev", "test")}
    if any(v is None for v in names.values()):
        raise ValueError("Reuse requires disjoint fit/dev/test records")
    fit = records[names["fit"]]
    if len(fit["x"]) != 64:
        raise ValueError("The preregistered FIT archive requires 64 articles")
    if any(len(records[names[s]]["x"]) != 12 for s in ("dev", "test")):
        raise ValueError("The preregistered evaluation requires 12 DEV and 12 TEST articles")
    width = fit["x"].shape[-1]
    if width != 896:
        raise ValueError("Expected the pinned donor's 896-dimensional source states")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    donor.eval()
    if any(p.requires_grad for p in donor.parameters()):
        raise ValueError("Freeze donor parameters before collecting reuse gradients")

    keys, gradient_banks, local_banks = [], [[], []], [[], []]
    effective_ranks = [[], []]
    custody_directory = output / "reuse_derivative_custody"
    custody_directory.mkdir(parents=True, exist_ok=True)
    for doc in range(64):
        budget.check()
        keys.append(_key(torch, fit["x"][doc], anchor, device))
        ids = torch.as_tensor(windows[names["fit"]][doc][:cfg["context"]], device=device,
                              dtype=torch.long).reshape(1, -1)
        if ids.shape[1] <= max(queries):
            raise ValueError("Actual-token window does not cover evaluation queries")
        probes, sampled_tokens = _probes(torch, donor, ids, width, fit_k, seed + doc,
                         anchor, queries, layer_end, budget, device)
        torch.save({"split": names["fit"], "article_index": doc,
                    "seed": seed + doc, "anchor": anchor, "queries": queries,
                    "sampled_token_ids": sampled_tokens,
                    "gradients": [g.detach().cpu() for g in probes],
                    "input_token_ids": ids.detach().cpu()},
                   custody_directory / f"fit_{doc:03d}.pt")
        for qi, gradients in enumerate(probes):
            gradient_banks[qi].append(gradients)
            local_banks[qi].append(_basis(torch, gradients, rank))
            effective_ranks[qi].append(int(torch.linalg.matrix_rank(gradients).item()))
    keys = torch.stack(keys)
    locals_by_query = [torch.stack(bank) for bank in local_banks]
    global_bases = {n: [_basis(torch, torch.cat(bank[:n]), rank)
                        for bank in gradient_banks] for n in (16, 64)}
    generator = torch.Generator(device=device).manual_seed(seed + 90000)
    random_basis = torch.linalg.qr(torch.randn(width, rank, device=device,
                                  generator=generator), mode="reduced").Q
    torch.save({"local_bases_by_query": [b.detach().cpu() for b in locals_by_query],
                "global_bases_by_archive_size": {
                    n: [b.detach().cpu() for b in bases] for n, bases in global_bases.items()},
                "random_basis": random_basis.detach().cpu(),
                "random_basis_seed": seed + 90000,
                "causal_keys": keys.detach().cpu(), "rank": rank,
                "anchor": anchor, "queries": queries},
               custody_directory / "fit_archives.pt")
    del gradient_banks, local_banks
    rows = {"dev": [], "test": []}
    audit = {"key_source": "source-entry-8 x; mean of positions31 through62 inclusive; L2 normalization",
             "fit_split": names["fit"], "fit_keys": keys.detach().cpu().tolist(),
             "fit_token_ids": [list(map(int, ids)) for ids in windows[names["fit"]]],
             "evaluation": {"dev": [], "test": []}}
    duplicate_index_equal = True
    duplicate_energy_equal = True
    for split_i, split in enumerate(("dev", "test")):
        data = records[names[split]]
        for doc in range(12):
            budget.check()
            key = _key(torch, data["x"][doc], anchor, device)
            audit["evaluation"][split].append({"article_index": doc,
                "causal_key": key.detach().cpu().tolist(),
                "token_ids": list(map(int, windows[names[split]][doc]))})
            nearest = {n: int((keys[:n] @ key).argmax().item()) for n in (16, 64)}
            duplicated = keys[:16].repeat(4, 1)
            duplicate = int((duplicated @ key).argmax().item()) % 16
            duplicate_index_equal &= duplicate == nearest[16]
            rng = random.Random(seed + 100000 + split_i * 1000 + doc)
            random_local = {n: rng.randrange(n) for n in (16, 64)}
            ids = torch.as_tensor(windows[names[split]][doc][:cfg["context"]], device=device,
                                  dtype=torch.long).reshape(1, -1)
            probe_seed = seed + 200000 + split_i * 1000 + doc
            targets, sampled_tokens = _probes(torch, donor, ids, width, eval_k,
                              probe_seed,
                              anchor, queries, layer_end, budget, device)
            torch.save({"split": split, "article_index": doc,
                        "seed": probe_seed, "anchor": anchor, "queries": queries,
                        "sampled_token_ids": sampled_tokens,
                        "gradients": [g.detach().cpu() for g in targets],
                        "input_token_ids": ids.detach().cpu(),
                        "nearest_selection": nearest, "random_local_selection": random_local,
                        "random_local_seed": seed + 100000 + split_i * 1000 + doc},
                       custody_directory / f"{split}_{doc:03d}.pt")
            for qi, query in enumerate(queries):
                gradients = targets[qi]
                values = {"random_rank8": _energy(torch, gradients, random_basis)}
                for n in (16, 64):
                    values[f"causal_nearest_{n}"] = _energy(
                        torch, gradients, locals_by_query[qi][nearest[n]])
                    values[f"random_local_{n}"] = _energy(
                        torch, gradients, locals_by_query[qi][random_local[n]])
                    values[f"global_{n}"] = _energy(torch, gradients, global_bases[n][qi])
                    # DEV-only privileged diagnostic. Never a deployable selector,
                    # never used to alter keys, archives, controls, or TEST routing.
                    if split == "dev":
                        values[f"privileged_target_oracle_{n}_NOT_DEPLOYABLE"] = max(
                            _energy(torch, gradients, basis)
                            for basis in locals_by_query[qi][:n])
                values["duplicate64_from16"] = _energy(
                    torch, gradients, locals_by_query[qi][duplicate])
                duplicate_energy_equal &= (values["duplicate64_from16"] ==
                                            values["causal_nearest_16"])
                rows[split].append({"article_index": doc, "query": query,
                                    "selected_fit_article": nearest,
                                    "random_fit_article": random_local,
                                    "captured_energy": values})
            _write(output / "reuse_partial.json", {"status": "IN_PROGRESS", "rows": rows})
    if not duplicate_index_equal or not duplicate_energy_equal:
        raise AssertionError("Duplicating the 16-entry archive changed nearest selection")
    summaries = {}
    for split, split_rows in rows.items():
        summaries[split] = {}
        for query in queries:
            articles = [r for r in split_rows if r["query"] == query]
            methods = list(articles[0]["captured_energy"])
            measures = {m: _summary([r["captured_energy"][m] for r in articles])
                        for m in methods}
            differences = {}
            for n in (16, 64):
                selected = f"causal_nearest_{n}"
                for control in (f"global_{n}", "random_rank8", f"random_local_{n}"):
                    differences[f"{selected}_minus_{control}"] = _summary([
                        r["captured_energy"][selected] - r["captured_energy"][control]
                        for r in articles])
            for method in ("causal_nearest", "global", "random_local"):
                differences[f"{method}_64_minus_16"] = _summary([
                    r["captured_energy"][f"{method}_64"] -
                    r["captured_energy"][f"{method}_16"] for r in articles])
            summaries[split][str(query)] = {"captured_energy": measures,
                                            "paired_differences": differences}
    report = {"status": "COMPLETE", "test": "cross_document_fisher_subspace_reuse",
              "anchor": anchor, "queries": list(queries), "rank": rank,
              "fit_articles": 64, "fit_probes_per_query_article": fit_k,
              "fit_empirical_gradient_rank_by_query": {
                  str(query): effective_ranks[qi] for qi, query in enumerate(queries)},
              "rank_deficiency": "Repeated categorical samples may reduce empirical rank; rank8 SVD orthonormal completion then includes directions with no fitted sensitivity evidence",
              "evaluation_probes_per_query_article": eval_k,
              "derivative_custody": "reuse_derivative_custody/*.pt stores exact FIT/DEV/TEST gradients, sampled token IDs, per-article seeds, exact input tokens, and frozen archive/global/random bases",
              "key": "L2-normalized mean source-entry-8 states x[31:63]; cosine full scan",
              "archive": "FIT-only rank8 score-gradient bases per article and query horizon",
              "evaluation": "Fresh independent categorical score probes; fraction of total gradient energy captured",
              "teacher_forcing": "Future query79 uses actual causal continuation tokens through position79",
              "oracle": "DEV-only target-aware best archive projection, NOT DEPLOYABLE; upward-biased finite-probe upper reference",
              "test_targets": "TEST gradients score frozen selections only and never create keys or archives",
              "selection_cost": "Exact full archive scan; no bounded-search or runtime-scalability claim",
              "selection_arithmetic": {str(n): {
                  "cosine_dot_products": n, "multiply_add_pairs": n * width,
                  "prefix_mean_source_values": 32 * width,
                  "selected_bases": 1} for n in (16, 64)},
              "storage_float32": {str(n): {"selected_basis_bytes_per_query": width * rank * 4,
                    "key_scan_bytes": n * width * 4,
                    "archive_basis_bytes_both_queries": n * width * rank * 4 * len(queries),
                    "archive_key_bytes": n * width * 4} for n in (16, 64)},
              "duplicate64_from16": {"same_nearest_index": duplicate_index_equal,
                                       "same_captured_energy": duplicate_energy_equal},
              "uncertainty": "12 independent article units per split/query; paired t intervals are descriptive and not multiplicity-adjusted",
              "scope": "Local downstream geometry transfer on a frozen teacher-forced donor; no quality, speedup, own-history, or capacity-scaling admission",
              "summaries": summaries, "rows": rows}
    _write(output / "reuse.json", report)
    _write(output / "reuse_causal_key_audit.json", audit)
    return report


def self_test(torch, device="cpu"):
    """Small algebra checks; invoke only in the authorized Kaggle environment."""
    generator = torch.Generator(device=device).manual_seed(21)
    gradients = torch.randn(16, 32, device=device, generator=generator)
    basis = _basis(torch, gradients, 8)
    error = (basis.T @ basis - torch.eye(8, device=device)).abs().max().item()
    if error > 1e-5:
        raise AssertionError("Basis orthogonality failed")
    value = _energy(torch, gradients, basis)
    keys = torch.randn(16, 32, device=device, generator=generator)
    query = torch.randn(32, device=device, generator=generator)
    first = (keys @ query).argmax().item()
    repeated = (keys.repeat(4, 1) @ query).argmax().item() % 16
    if first != repeated or not 0 <= value <= 1:
        raise AssertionError("Projection or duplicate archive invariant failed")
    return {"orthogonality_max_error": error, "projection_fraction": value,
            "duplicate_nearest_equal": True}
