"""Cheap, conditional follow-ups using source-input captures on Kaggle.

These are local mechanism diagnostics, not an autonomous routing policy or
capacity-scaling admission. All learned fits and archives use training only.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def finite(torch, value):
    if not torch.isfinite(value).all().item():
        raise FloatingPointError("nonfinite follow-up diagnostic")


def predictor(torch, model, records, cfg, output, budget, device, prefix_features):
    rows = {}
    for split, data in records.items():
        collected = []
        for doc in range(len(data["x"])):
            budget.check()
            x = data["x"][doc:doc + 1].to(device=device, dtype=torch.float32)
            y = data["y"][doc:doc + 1].to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                fresh = model(x, prefix_features(torch, x, cfg["prefix"], 16))
                stale = model(x, prefix_features(torch, x, cfg["prefix"], 64))
                for start in range(cfg["prefix"] + 16, x.shape[1] - 15, 16):
                    if (start - cfg["prefix"]) % 64 == 0:
                        continue
                    p = x[:, start - cfg["prefix"]:start]
                    prev = x[:, max(0, start - cfg["prefix"] - 16):start - 16]
                    mean = p.mean(1)
                    features = [1., start / cfg["context"], p.square().mean().sqrt().item(),
                                mean.square().mean().sqrt().item(), p.std(unbiased=False).item(),
                                (mean - prev.mean(1)).square().mean().sqrt().item(),
                                (p[:, -1] - mean).square().mean().sqrt().item()]
                    section = slice(start, start + 16)
                    energy = (y[:, section] - x[:, section]).square().mean().clamp(min=1e-12)
                    gain = ((stale[:, section] - y[:, section]).square().mean() -
                            (fresh[:, section] - y[:, section]).square().mean()) / energy
                    finite(torch, gain)
                    collected.append({"doc": doc, "position": start,
                                      "features": features, "gain": gain.item()})
        rows[split] = collected
    tx = torch.tensor([r["features"] for r in rows["train"]], dtype=torch.float64)
    ty = torch.tensor([r["gain"] for r in rows["train"]], dtype=torch.float64)
    center, scale = tx.mean(0), tx.std(0).clamp(min=1e-8)
    center[0], scale[0] = 0., 1.
    z = (tx - center) / scale
    weight = torch.linalg.solve(z.T @ z + 1e-3 * torch.eye(z.shape[1]), z.T @ ty)
    finite(torch, weight)
    # Freeze an approximately 25% refresh threshold using training predictions.
    threshold = torch.quantile(z @ weight, .75).item()
    report = {"status": "LOCAL_PROXY_COMPLETE", "threshold": threshold,
              "weight": weight.tolist(), "center": center.tolist(), "scale": scale.tolist(),
              "limitation": "Independent counterfactual opportunities on source inputs; stale64 schedule is fixed. No closed-loop refresh-policy or own-history claim.",
              "controls": "Periodic/random control counts are matched retrospectively to causal learned decisions; they do not use benefit labels.",
              "active_predictor_values": int(weight.numel()),
              "execution": "Batched local proxy calls repeat program generator per position; no amortized runtime claim.",
              "splits": {}}
    for split, data in rows.items():
        features = torch.tensor([r["features"] for r in data], dtype=torch.float64)
        gains = torch.tensor([r["gain"] for r in data], dtype=torch.float64)
        pred = ((features - center) / scale) @ weight
        selected = pred > threshold
        n, k = len(data), int(selected.sum())
        periodic = torch.zeros(n, dtype=torch.bool)
        random_mask = torch.zeros(n, dtype=torch.bool)
        oracle = torch.zeros(n, dtype=torch.bool)
        if k:
            periodic[((torch.arange(k, dtype=torch.float64) + .5) * n / k).long()] = True
            g = torch.Generator().manual_seed(cfg["seed"] + 77)
            random_mask[torch.randperm(n, generator=g)[:k]] = True
            oracle[torch.topk(gains, k).indices] = True
        matched = {name: float(gains[mask].sum() / n) for name, mask in
                   (("learned", selected), ("periodic", periodic), ("random", random_mask), ("oracle", oracle))}
        report["splits"][split] = {"opportunities": n, "refresh_count": k,
                                  "refresh_fraction": k / n, "benefit_per_opportunity": matched,
                                  "learned_exceeds_both_controls": matched["learned"] > max(matched["periodic"], matched["random"])}
    write(output / "exception_opportunities.json", rows)
    write(output / "exception_predictor.json", report)
    return report


def archive_test(torch, model, records, cfg, output, budget, device, prefix_features):
    bank = []
    identities = []
    for doc in range(len(records["train"]["x"])):
        x = records["train"]["x"][doc:doc + 1].float()
        p = prefix_features(torch, x, cfg["prefix"], 16)
        for start in range(cfg["prefix"], x.shape[1] - 15, 16):
            bank.append(p[0, start].clone())
            identities.append({"training_doc": doc, "position": start})
    bank = torch.stack(bank)
    g = torch.Generator().manual_seed(cfg["seed"] + 103)
    order = torch.randperm(len(bank), generator=g)
    bank = bank[order]
    identities = [identities[i] for i in order.tolist()]
    if len(bank) < 256:
        return {"status": "INCONCLUSIVE", "reason": "fewer than 256 training-only entries"}
    # Same normalized prefix representation for the key and contextual value.
    # Full scanning is explicitly charged and NOT a scalable-addressing claim.
    bank = bank[:256].to(device)
    keys = torch.nn.functional.normalize(bank, dim=-1)
    key_bytes = bank.element_size() * bank.numel()
    output_rows = []
    for split in ("dev", "test"):
        data = records[split]
        for doc in range(len(data["x"])):
            budget.check()
            x = data["x"][doc:doc + 1].to(device=device, dtype=torch.float32)
            y = data["y"][doc:doc + 1].to(device=device, dtype=torch.float32)
            own = prefix_features(torch, x, cfg["prefix"], 16)
            with torch.inference_mode():
                for start in range(cfg["prefix"], x.shape[1] - 15, 16):
                    budget.check()
                    query = own[:, start]
                    scores = torch.nn.functional.normalize(query, dim=-1) @ keys.T
                    i64 = int(scores[0, :64].argmax())
                    i256 = int(scores[0].argmax())
                    # Deterministic random control independent of targets/query.
                    rand = int(hashlib.sha256(f"{cfg['seed']}:{split}:{doc}:{start}".encode()).hexdigest()[:8], 16) % 256
                    section = x[:, start:start + 16]
                    target = y[:, start:start + 16]
                    values = {"fresh_privileged": query, "zero_program_input": torch.zeros_like(query),
                              "retrieved64": bank[i64:i64 + 1], "retrieved256": bank[i256:i256 + 1],
                              "random256": bank[rand:rand + 1], "duplicated256": bank[i64:i64 + 1]}
                    errors = {}
                    for name, p in values.items():
                        prediction = model(section, p[:, None, :].expand(-1, 16, -1))
                        error = (prediction - target).square().sum()
                        finite(torch, error)
                        errors[name] = error.item()
                    output_rows.append({"split": split, "doc": doc, "position": start,
                                        "id64": i64, "id256": i256, "random_id": rand, "sse": errors})
    summaries = {}
    for split in ("dev", "test"):
        subset = [r for r in output_rows if r["split"] == split]
        summaries[split] = {name: sum(r["sse"][name] for r in subset) for name in subset[0]["sse"]}
    report = {"status": "LOCAL_PROXY_COMPLETE", "splits": summaries,
              "archive_entries": [64, 256], "active_entries": 1,
              "key_scan_bytes_per_lookup_256": key_bytes,
              "selected_prefix_value_bytes": int(bank.shape[1] * bank.element_size()),
              "archive_representation": "Training-only causal prefix prototypes; program generator still constructs modulation after retrieval.",
              "execution": "Local proxy calls repeat the generator per position; actual diagnostic work is not amortized.",
              "zero_control": "Zero prefix input, not removal of all generator biases.",
              "duplicate_control": "256 slots repeating the same64 entries; stable first tie yields the identical retrieved64 value.",
              "limitation": "Full bank scans and source-input queries; no bounded addressing, compiled-program storage, whole-model retrieval quality or large-family scaling admission."}
    write(output / "archive_identities.json", identities[:256])
    write(output / "archive_rows.json", output_rows)
    write(output / "archive_report.json", report)
    return report


def run(torch, model, records, cfg, output, budget, device, prefix_features):
    summary = {"status": "RUNNING", "scope": "Conditional exploratory source-input diagnostics"}
    write(output / "followup_status.json", summary)
    with torch.inference_mode():
        summary["exceptions"] = predictor(torch, model, records, cfg, output, budget, device, prefix_features)
        summary["archive"] = archive_test(torch, model, records, cfg, output, budget, device, prefix_features)
    summary["status"] = "COMPLETE_LOCAL_DIAGNOSTICS"
    write(output / "followup_status.json", summary)
    return summary


def self_test():
    """Tiny cloud-only tensor smoke test before spending the training budget."""
    import tempfile
    import torch
    from pilot import Budget, build_classes, prefix_features
    torch.manual_seed(17)
    cfg = {"seed": 17, "prefix": 32, "context": 256}
    records = {}
    for split, count in (("train", 20), ("dev", 2), ("test", 2)):
        x = torch.randn(count, 256, 4)
        records[split] = {"x": x, "y": x + .1 * torch.randn_like(x)}
    model = build_classes(torch)(4, 3, 2, "conditional", 1.)
    with tempfile.TemporaryDirectory() as directory:
        result = run(torch, model, records, cfg, Path(directory), Budget(120, 10),
                     torch.device("cpu"), prefix_features)
        assert result["status"] == "COMPLETE_LOCAL_DIAGNOSTICS"
        assert result["archive"]["status"] == "LOCAL_PROXY_COMPLETE"
        for split in ("dev", "test"):
            r = result["archive"]["splits"][split]
            assert r["retrieved64"] == r["duplicated256"]
    print("Cloud follow-up shapes, finite checks and duplicate control passed")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] != ["--self-test"]:
        raise SystemExit("Use --self-test inside Kaggle; real execution is controlled by pilot.py")
    if not Path("/kaggle").exists():
        raise SystemExit("Tensor smoke tests are restricted to Kaggle")
    self_test()
