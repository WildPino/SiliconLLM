"""CPU-only readback of sealed group-oracle outputs; no contribution replay.

Recompute branch metrics and selected residual consistency. The source-group
construction/optimal-mass execution remains a separate producer witness.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "results/native_expert_scaling/chatbot_source_runtime/site"))
import torch
import psutil
torch.set_num_threads(6)
torch.set_num_interop_threads(1)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_files(pins):
    for d in pins:
        p = Path(d["path"])
        assert p.stat().st_size == d["bytes"] and sha(p) == d["sha256"], str(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--binding", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    start = time.monotonic()
    assert not args.report.exists()
    out = args.source.resolve()
    r = json.loads((out / "result.json").read_text())
    b = json.loads(args.binding.read_text())
    assert r["binding_sha256"] == sha(args.binding)
    assert r["status"] == "ONE_GROUP_ORACLE_LOWER_BOUND_FAIL"
    assert r["producer_status"] == "FAILED_GPU_RELEASE" and r["admitted"] is False
    pins = list(b["files"].values()) + r["owned_outputs"]
    own = {"code": sha(__file__), "result": sha(out / "result.json"),
           "binding": sha(args.binding), "terminal": sha(out / "terminal.json")}
    check_files(pins)
    m = json.loads(Path(b["files"]["manifest"]["path"]).read_text())
    common = m["common_prefix_length"]
    rows = {s: {k: [] for k in ("oracle", "all_group")} for s in ("FIT", "DEV")}
    for c in m["cases"]:
        assert time.monotonic() - start < 600
        q = torch.load(out / ("case_" + c["id"] + ".pt"), weights_only=True,
                       map_location="cpu")
        truth = torch.load(b["files"]["source_" + c["id"]]["path"],
                           weights_only=True, map_location="cpu")["branch"].double()
        n = len(c["input_ids"])
        assert q["input_ids"] == c["input_ids"] and q["case_id"] == c["id"]
        assert q["split"] == c["split"] and q["common_prefix_length"] == common
        residuals = q["analytic_residual2_f64"]
        masses = q["mass_f64"]
        ids = q["selected_group_id"].long()
        assert residuals.shape == (n, 16) and residuals.dtype == torch.float64
        assert masses.shape == ids.shape == (n,) and masses.dtype == torch.float64
        assert torch.isfinite(residuals).all() and torch.isfinite(masses).all()
        assert (residuals >= 0).all() and torch.equal(ids, residuals.argmin(1))
        for kind, key in (("oracle", "oracle_branch_f64"),
                          ("all_group", "all_group_branch_f64")):
            branch = q[key]
            assert branch.shape == truth.shape and branch.dtype == torch.float64
            assert torch.isfinite(branch).all()
            token_errors = (branch - truth).square().sum(1)
            if kind == "oracle":
                ref2 = truth.square().sum(1).clamp_min(1e-30)
                assert ((token_errors - residuals[torch.arange(n), ids]).abs()
                        <= 1e-10 * ref2).all()
            e2 = float(token_errors[common:].sum())
            n2 = float(truth[common:].square().sum())
            rows[c["split"]][kind].append((e2, n2, math.sqrt(e2 / max(n2, 1e-30))))
        assert psutil.Process().memory_info().peak_wset < 3 << 30
    metrics = {}
    for split in rows:
        metrics[split] = {}
        for kind, values in rows[split].items():
            actual = {"pooled_branch_l2": math.sqrt(sum(x[0] for x in values) /
                                                   sum(x[1] for x in values)),
                      "sequence_l2_p95": sorted(x[2] for x in values)[math.ceil(.95 * len(values)) - 1]}
            for key, val in actual.items():
                assert abs(val - r["metrics"][split][kind][key]) < 1e-10
            metrics[split][kind] = actual
        assert metrics[split]["all_group"]["pooled_branch_l2"] <= .01
        assert metrics[split]["all_group"]["sequence_l2_p95"] <= .05
        assert metrics[split]["oracle"]["pooled_branch_l2"] > .01
        assert metrics[split]["oracle"]["sequence_l2_p95"] > .05
    check_files(pins)
    assert own == {"code": sha(__file__), "result": sha(out / "result.json"),
                   "binding": sha(args.binding), "terminal": sha(out / "terminal.json")}
    report = {"schema": "SOURCE_GROUP_METRIC_READBACK_V1", "status": "PASS",
              "source_status": r["status"], "pins_sha256": own, "metrics": metrics,
              "cases": len(m["cases"]), "seconds": time.monotonic() - start,
              "peak_wset_bytes": psutil.Process().memory_info().peak_wset,
              "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
              "limits": "Checks saved branches, residuals, IDs and seals; does not replay source-group construction or optimal masses from weights."}
    with args.report.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "seconds": report["seconds"]}))


if __name__ == "__main__":
    main()
