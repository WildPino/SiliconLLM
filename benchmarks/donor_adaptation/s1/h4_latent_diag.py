#!/usr/bin/env python3
"""H4D: evaluate the frozen H4 terminal fp32 masters without ternarization.

Descriptive CPU diagnostic only; never adjudicates H4's ternary gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F

import h4_eval as E
import h4_qat as H4

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "results" / "h4"
TERMINAL = OUTDIR / "stage_a_v2_kaggle" / "h4_trained.npz"
TERNARY = OUTDIR / "h4_eval_stage_a_v2_terminal.json"
INIT = OUTDIR / "h4_eval_init.json"
DEFAULT_OUT = OUTDIR / "h4_latent_diagnostic.json"
PIN = {
    TERMINAL: "11a483419e6b8795164e4651547f6b3765fc325adaa3283153d9cd5276d4cefb",
    TERNARY: "23c3792f4a04efedf90ec0617654a529f0e2a115419f55cd71ad0d55d48e48f2",
    INIT: "140a8968ccc49ebe4bdf8518bd6f901b0b5c16e76ed5df87a069bc2f7c1e461c",
    HERE / "h4_eval.py": "0b536b46a23e2290172824d139c8564378331e98eb2cb93a1eccbf9a3d9e59b3",
    HERE / "h4_qat.py": "8a3b5c7be26e3b21f9c223d69706fb36c5ed7d41b2d260dfed70b8e1778b7bb7",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assert_committed_runner() -> dict:
    rel = "benchmarks/donor_adaptation/s1/h4_latent_diag.py"
    repo = HERE.parents[2]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    committed = subprocess.check_output(
        ["git", "rev-parse", "HEAD:" + rel], cwd=repo, text=True
    ).strip()
    current = subprocess.check_output(
        ["git", "hash-object", str(Path(__file__).resolve())], cwd=repo, text=True
    ).strip()
    if current != committed:
        raise SystemExit("H4D runner differs from the committed blob")
    return {"head_commit": head, "runner_git_blob": committed, "runner_sha256": sha256_file(Path(__file__))}


def latent_forward(self: H4.TernaryLowRank, x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cpu" or x.dtype != torch.float32:
        raise RuntimeError("H4D accepts CPU fp32 only")
    h = F.linear(x, self.B) * self.s
    y = F.linear(h, self.A)
    return y if self.bias is None else y + self.bias_buf


def selftest() -> None:
    a = torch.tensor([[0.2, -0.5], [0.7, 0.3], [-0.4, 0.9]])
    b = torch.tensor([[0.4, -0.2, 0.8, 0.5], [0.1, 0.6, -0.3, 0.7]])
    s = torch.tensor([0.8, -1.2])
    bias = torch.tensor([0.1, 0.2, -0.1])
    mod = H4.TernaryLowRank(a, s, b, torch.ones(4), torch.ones(2), bias)
    x = torch.tensor([[0.5, -1.0, 0.25, 0.4]])
    want = F.linear(F.linear(x, b) * s, a) + bias
    if not torch.equal(latent_forward(mod, x), want):
        raise SystemExit("latent-forward selftest failed")
    if torch.equal(mod(x), want):
        raise SystemExit("quantized control unexpectedly equals the latent arm")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out = args.out.resolve()
    raw = out.with_name(out.stem + ".internal_raw.json")
    if out.exists() or raw.exists():
        raise SystemExit("H4D output already exists; refusing overwrite")
    if out == TERNARY or out == INIT or out.parent != OUTDIR:
        raise SystemExit("H4D output must be a new file in results/h4")

    provenance = assert_committed_runner()
    for path, expected in PIN.items():
        if sha256_file(path) != expected:
            raise SystemExit("H4D pinned input changed: " + str(path))
    meta = json.loads(TERMINAL.with_suffix(".json").read_text(encoding="utf-8"))
    if not (meta.get("stage") == "H4" and meta.get("model") == H4.MODEL_ID
            and meta.get("revision") == H4.REVISION and meta.get("rank") == 48
            and meta.get("status") == "TIME_CAP" and meta.get("terminal_checkpoint") is True
            and meta.get("actual_steps") == 1148 and meta.get("applied_steps") == 1145
            and meta.get("origin_factors_sha256") == "7ae0d2ceeff628ff9b076b5df5806def54340f193e6cae5eb8e8013817959371"
            and meta.get("checkpoint_sha256") == PIN[TERMINAL]):
        raise SystemExit("H4D terminal metadata mismatch")
    selftest()
    reference = json.loads(TERNARY.read_text(encoding="utf-8"))

    original_forward = H4.TernaryLowRank.forward
    original_argv = sys.argv
    try:
        H4.TernaryLowRank.forward = latent_forward
        sys.argv = [str(HERE / "h4_eval.py"), "--factors", str(TERMINAL),
                    "--tag", "h4d_fp32_latent", "--out", str(raw)]
        rc = E.main()
    finally:
        H4.TernaryLowRank.forward = original_forward
        sys.argv = original_argv
    if rc != 0:
        raise SystemExit("frozen evaluator failed with status " + str(rc))
    measured = json.loads(raw.read_text(encoding="utf-8"))
    if measured.get("factors_sha256") != PIN[TERMINAL] or measured.get("counted") != 160:
        raise SystemExit("H4D evaluator provenance/count mismatch")

    # The frozen evaluator's H4 band formula is deliberately inapplicable to
    # this fp32 diagnostic. Do not publish those boolean values as H4 gates.
    measured.pop("bands", None)
    measured["mode"] = "latent_fp32_posthoc_diagnostic"
    measured["eligible_for_H4_gates"] = False
    measured["question"] = "same trained masters, remove only inference-time ternarization of A/B"
    measured["reference_ternary"] = {k: reference[k] for k in
        ("bpb", "free", "teacher_forced", "mean_rank", "median_rank", "rank_le5")}
    measured["delta_latent_minus_ternary"] = {k: measured[k] - reference[k] for k in
        ("bpb", "free", "teacher_forced", "mean_rank", "median_rank", "rank_le5")}
    measured["predictions"] = {
        "bpb_lower_by_at_least_0p05": measured["bpb"] <= reference["bpb"] - 0.05,
        "teacher_forced_up_by_at_least_10": measured["teacher_forced"] >= reference["teacher_forced"] + 10,
        "free_remains_at_floor": measured["free"] <= 14,
    }
    measured["provenance"] = provenance
    measured["pinned_sha256"] = {str(k.relative_to(HERE)): v for k, v in PIN.items() if k.is_relative_to(HERE)}
    measured["non_promotion"] = "Descriptive fp32 arm; no H4 re-adjudication, Stage B license, 10B or rate claim"
    with out.open("x", encoding="utf-8") as f:
        json.dump(measured, f, indent=1)
    raw.unlink()
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
