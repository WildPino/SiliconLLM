"""Temporal persistence of downstream categorical-score sensitivity on Kaggle.

The experiment compares subspaces in the donor's fixed hidden coordinate basis.
It is an oracle derivative diagnostic, without token-quality or runtime admission.
No tensor library is imported or numerical work performed at module import.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import time


SPEC = dict(anchors=[63, 79, 95], query_offsets=[0, 16], basis_samples=16,
            evaluation_samples=8, rank=8, documents_per_partition=8,
            bootstrap_replicates=2000, rank_relative_tolerance=1e-6)


def seed_for(seed, *parts):
    text = ":".join(str(part) for part in (seed, *parts))
    return int(hashlib.sha256(text.encode("utf8")).hexdigest()[:15], 16)


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def mean(values):
    if not values:
        raise ValueError("Empty document collection")
    return sum(values) / len(values)


def paired_interval(differences, seed, replicates=2000):
    """Document-level paired bootstrap; no token/sample independence assumption."""
    if not differences or not all(math.isfinite(x) for x in differences):
        raise ValueError("Finite nonempty paired differences required")
    generator = random.Random(seed)
    n = len(differences)
    samples = sorted(mean([differences[generator.randrange(n)] for _ in range(n)])
                     for _ in range(replicates))
    return dict(mean=mean(differences), ci95=[samples[int(.025 * (replicates - 1))],
                                           samples[int(.975 * (replicates - 1))]],
                paired_documents=n, bootstrap_replicates=replicates)


def finite(torch, tensor, label):
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError("Nonfinite temporal " + label)


def score_gradients(torch, donor, ids, anchor, cfg, budget, seed):
    """One differentiated forward; independently sampled fitting/response scores.

    For each query, samples come from the unperturbed categorical distribution.
    A score is d log p(sampled token | fixed token prefix) / d delta_anchor.
    The held-out categorical draws use a separate random stream from the basis.
    """
    from geometry import logits_with_delta

    delta = torch.zeros(donor.config.hidden_size, device=ids.device,
                        dtype=torch.float32, requires_grad=True)
    queries = [anchor + offset for offset in SPEC["query_offsets"]]
    result = {}
    with torch.enable_grad():
        logits = logits_with_delta(torch, donor, ids, delta, anchor,
                                   cfg["layer_end"], query_positions=queries)
        if tuple(logits.shape[:2]) != (1, len(queries)):
            raise RuntimeError("Unexpected selective-logits helper shape")
        finite(torch, logits, "logits")
        logp = logits.float().log_softmax(-1)
        for query_index, offset in enumerate(SPEC["query_offsets"]):
            budget.check()
            probabilities = logp[0, query_index].detach().exp().cpu().double()
            groups = {}
            for group, count in (("basis", SPEC["basis_samples"]),
                                 ("evaluation", SPEC["evaluation_samples"])):
                sampling_seed = seed_for(seed, anchor, offset, group)
                generator = torch.Generator(device="cpu").manual_seed(sampling_seed)
                token_ids = torch.multinomial(probabilities, count, replacement=True,
                                             generator=generator).tolist()
                gradients = []
                for token in token_ids:
                    budget.check()
                    gradient, = torch.autograd.grad(logp[0, query_index, token], delta,
                                                   retain_graph=True, create_graph=False)
                    finite(torch, gradient, "categorical score gradient")
                    gradients.append(gradient.detach().cpu())
                groups[group] = dict(gradients=torch.stack(gradients),
                                     sampled_token_ids=token_ids, seed=sampling_seed)
            result[offset] = groups
    return result


def make_basis(torch, gradients):
    finite(torch, gradients, "basis samples")
    _, singular_values, vh = torch.linalg.svd(gradients.float(), full_matrices=False)
    dimension = min(SPEC["rank"], vh.shape[0], vh.shape[1])
    if dimension != SPEC["rank"]:
        raise RuntimeError("Insufficient dimensions for prescribed rank-eight basis")
    maximum = singular_values[0].item()
    effective_rank = int((singular_values > maximum * SPEC["rank_relative_tolerance"]).sum()) if maximum > 0 else 0
    basis = vh[:dimension].T.contiguous()
    finite(torch, basis, "basis")
    return basis, dict(basis_dimension=dimension, effective_sample_rank=effective_rank,
                       rank_eight_supported=effective_rank >= dimension,
                       singular_values=singular_values.tolist(),
                       zero_or_degenerate_samples=maximum == 0)


def random_basis(torch, dim, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(dim, SPEC["rank"], generator=generator), mode="reduced")
    return q


def projection(torch, gradients, basis):
    energy = gradients.double().square().sum(-1)
    captured = (gradients.double() @ basis.double()).square().sum(-1)
    finite(torch, energy, "held-out energy")
    finite(torch, captured, "held-out projection")
    if bool((energy <= 0).any()):
        raise FloatingPointError("Zero held-out categorical-score sensitivity")
    fractions = captured / energy
    if bool((fractions > 1.00001).any()) or bool((fractions < 0).any()):
        raise FloatingPointError("Invalid orthogonal projection fraction")
    return dict(mean_sample_fraction=fractions.mean().item(),
                pooled_energy_fraction=(captured.sum() / energy.sum()).item(),
                sample_energy=energy.tolist(), captured_energy=captured.tolist(),
                sample_fraction=fractions.tolist())


def overlap(torch, first, second):
    cosines = torch.linalg.svdvals(first.double().T @ second.double()).clamp(0, 1)
    finite(torch, cosines, "principal cosines")
    return dict(principal_cosines=cosines.tolist(),
                mean_squared_cosine=cosines.square().mean().item())


def run(torch, donor, records, windows, cfg, output, budget, device):
    """Run the fixed temporal study; fitting and responses stay article separated.

    `records` is accepted for the common experiment interface and is deliberately
    unused: score derivatives come from fresh unperturbed source-model forwards.
    """
    del records
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(SPEC, layer_end=cfg["layer_end"], seed=cfg["seed"],
                  source_model=cfg.get("model_id"), source_revision=cfg.get("model_revision"),
                  derivative="categorical log-probability score wrt one span-output position",
                  coordinates="Unchanged donor hidden basis; no learned semantic alignment")
    write(output / "temporal_config.json", config)
    if donor.training or any(p.requires_grad for p in donor.parameters()):
        raise RuntimeError("Temporal derivatives require eval-mode frozen donor parameters")
    if next(donor.parameters()).dtype != torch.float32:
        raise RuntimeError("Temporal score gradients require the prescribed FP32 donor")
    all_rows, summaries, training_docs = [], {}, None
    try:
        for split in ("train", "dev", "test"):
            if len(windows[split]) < SPEC["documents_per_partition"]:
                raise RuntimeError("Require eight documents per temporal partition")
            docs = []
            for doc, raw in enumerate(windows[split][:SPEC["documents_per_partition"]]):
                budget.check()
                token_ids = list(raw)[:cfg.get("context", 192)]
                if len(token_ids) <= max(SPEC["anchors"]) + max(SPEC["query_offsets"]):
                    raise ValueError("Temporal input too short for prescribed anchor/query pairs")
                ids = torch.tensor([token_ids], dtype=torch.long, device=device)
                document = dict(index=doc,
                                token_sha256=hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
                                anchors={})
                for anchor in SPEC["anchors"]:
                    budget.check()
                    responses = score_gradients(torch, donor, ids, anchor, cfg, budget,
                                                seed_for(cfg["seed"], split, doc))
                    local = {}
                    for offset, groups in responses.items():
                        basis, receipt = make_basis(torch, groups["basis"]["gradients"])
                        local[offset] = dict(basis=basis, receipt=receipt,
                                             basis_score_gradients=groups["basis"]["gradients"],
                                             heldout=groups["evaluation"]["gradients"],
                                             sampling={name: {key: value for key, value in group.items()
                                                              if key != "gradients"}
                                                       for name, group in groups.items()})
                    document["anchors"][anchor] = local
                docs.append(document)
                # CPU tensors retain the exact observed basis/response evidence;
                # these small cloud artifacts permit analysis without another run.
                torch.save(docs, output / ("temporal_evidence_" + split + ".pt"))
            # DEV/TEST cross-document controls use the frozen TRAIN basis pool.
            # Conditional on that pool, held-out rows do not depend on one another.
            # The TRAIN descriptive rows use a nonidentity circular pairing.
            if split == "train":
                training_docs = docs
            for doc, document in enumerate(docs):
                other = training_docs[(doc + 1) % len(training_docs)]
                for offset in SPEC["query_offsets"]:
                    old = document["anchors"][SPEC["anchors"][0]][offset]
                    cross = other["anchors"][SPEC["anchors"][0]][offset]
                    for anchor in SPEC["anchors"]:
                        budget.check()
                        current = document["anchors"][anchor][offset]
                        random_seed = seed_for(cfg["seed"], split, doc, anchor, offset, "random")
                        random = random_basis(torch, donor.config.hidden_size, random_seed)
                        controls = dict(old=old["basis"], current=current["basis"],
                                        cross_document_old=cross["basis"], random=random)
                        row = dict(split=split, document=doc,
                                   token_sha256=document["token_sha256"], anchor=anchor,
                                   query=anchor + offset, query_offset=offset,
                                   old_anchor=SPEC["anchors"][0], cross_document=(doc + 1) % len(training_docs),
                                   cross_document_source_split="train",
                                   sampling=current["sampling"], basis_receipts=dict(
                                       current=current["receipt"], old=old["receipt"],
                                       cross_document_old=cross["receipt"]), random_seed=random_seed,
                                   projection={name: projection(torch, current["heldout"], basis)
                                               for name, basis in controls.items()},
                                   overlap_with_current={name: overlap(torch, basis, current["basis"])
                                                         for name, basis in controls.items()},
                                   rank_qualified=all(item["receipt"]["rank_eight_supported"]
                                                      for item in (old, current, cross)))
                        all_rows.append(row)
                        write(output / "temporal_documents.json", all_rows)
            summaries[split] = {}
            for anchor in SPEC["anchors"]:
                for offset in SPEC["query_offsets"]:
                    subset = [row for row in all_rows if row["split"] == split and
                              row["anchor"] == anchor and row["query_offset"] == offset]
                    contrasts = {}
                    for first, second in (("old", "random"), ("old", "cross_document_old"),
                                          ("current", "old"), ("current", "random")):
                        differences = [row["projection"][first]["pooled_energy_fraction"] -
                                       row["projection"][second]["pooled_energy_fraction"] for row in subset]
                        contrasts[first + "_minus_" + second] = paired_interval(
                            differences, seed_for(cfg["seed"], split, anchor, offset, first, second),
                            SPEC["bootstrap_replicates"])
                    summaries[split][f"anchor{anchor}_query_plus{offset}"] = dict(
                        documents=len(subset), rank_qualified_documents=sum(row["rank_qualified"] for row in subset),
                        mean_document_projection={name: mean([row["projection"][name]["pooled_energy_fraction"]
                                                             for row in subset]) for name in controls},
                        paired_contrasts=contrasts)
            write(output / "temporal_summary.json", summaries)
        report = dict(status="COMPLETE_ORACLE_DERIVATIVE_DIAGNOSTIC", config=config,
                      partitions=summaries, elapsed_seconds=time.monotonic() - started,
                      differentiated_forwards=3 * 8 * 3,
                      categorical_gradient_evaluations=3 * 8 * 3 * 2 * (16 + 8),
                      rank_warning="Rank-deficient basis samples retain eight SVD directions but are explicitly unqualified; null directions have no identified predictive meaning.",
                      limitations=["Only eight documents per partition; bootstrap intervals are exploratory.",
                                   "Train rows describe fit-partition behavior; held-out claims use dev/test articles.",
                                   "DEV/TEST paired intervals condition on the fixed training basis pool; TRAIN circular cross-document controls share articles and their intervals are descriptive.",
                                   "Old/current subspaces use independent basis and response draws at each anchor; old/current coincide at anchor63 as a sanity control.",
                                   "Coordinates are the donor's unchanged D-dimensional hidden basis.",
                                   "Offset16 basis construction sees the fixed teacher-forced future tokens up to its query; it is an offline oracle reference, not a causally available program at the anchor.",
                                   "Categorical score derivatives characterize local source-model sensitivity, not chatbot quality or cheap execution."])
        write(output / "temporal_report.json", report)
        return report
    except BaseException as exc:
        write(output / "temporal_report.json", dict(status="INCOMPLETE", error_type=type(exc).__name__,
                                                    error=str(exc), elapsed_seconds=time.monotonic() - started))
        raise
