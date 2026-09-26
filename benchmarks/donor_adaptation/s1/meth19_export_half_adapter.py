#!/usr/bin/env python3
"""Export the METH-19 half-amplitude E128 factors as a bound adapter file."""
import argparse
import json
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17


ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "results/native_expert_scaling/meth16_checkpoints/meth16_update1024.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert M15.M13.sha256(CHECKPOINT) == M17.CHECKPOINT_SHA
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert checkpoint["updates"] == 1024
    assert checkpoint["source_sha256"] == M15.M13.MODEL_SHA
    assert checkpoint["train_ids_sha256"] == M15.TRAIN_IDS_SHA
    assert len(checkpoint["expert_state"]) == M15.M13.L
    tensors = {}
    for li, state in enumerate(checkpoint["expert_state"]):
        assert set(state) == {"a", "b", "router"}
        for key in ("a", "b", "router"):
            source = state[key]
            assert source.dtype == torch.float32
            tensors[f"layers.{li}.{key}"] = (
                (source * 0.5 if key == "b" else source).contiguous())
    metadata = {"experiment": "METH-19", "model": M15.M13.MODEL,
                "model_revision": M15.M13.REV,
                "model_sha256": M15.M13.MODEL_SHA,
                "tokenizer_fingerprint": M15.M13.TOK_FP,
                "source_checkpoint_sha256": M17.CHECKPOINT_SHA,
                "expert_count": str(M15.E), "top_k": str(M15.K),
                "rank": str(M15.R), "layers": str(M15.M13.L),
                "output_factor": "0.5"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=metadata)
    reloaded = load_file(str(out), device="cpu")
    assert reloaded.keys() == tensors.keys()
    assert all(torch.equal(reloaded[key], value) for key, value in tensors.items())
    result = {"path": str(out.resolve()), "sha256": M15.M13.sha256(out),
              "bytes": out.stat().st_size, "tensor_count": len(tensors),
              "factor_count": sum(x.numel() for x in tensors.values()),
              "metadata": metadata, "exact_reload_equal": True}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
