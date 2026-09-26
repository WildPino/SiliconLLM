#!/usr/bin/env python3
"""METH-14 CPU diagnosis: fitted versus current-activation oracle routing."""
import argparse
import json
import math
import os
import time

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

import meth13_qwen05b_preflight as M13


LABEL_SHA = "1c6840399ace72afee3e7200ad21a5a1519b12a688545d5035143168d8c637b1"
ROUTER_SHA = "55f01e1d5c8c5ef9df5db8a813036d48d889f4ad9286046ca6fefb191e509518"
DONOR_BPB = 1.0830219361346318
FITTED_BPB = 2.4434964126822423
LIMIT_SECONDS = 900
LIMIT_RSS = 20 * (1 << 30)


def budget(start):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    if elapsed > LIMIT_SECONDS:
        raise TimeoutError("METH-14 15-minute CPU stop")
    if rss > LIMIT_RSS:
        raise MemoryError("METH-14 20-GiB process RAM stop")
    return elapsed, rss


class RoutedFFN(nn.Module):
    def __init__(self, base, labels, router):
        super().__init__()
        self.base = base
        self.register_buffer("labels", torch.as_tensor(labels.copy(), dtype=torch.long))
        self.register_buffer("router", torch.as_tensor(router.copy(), dtype=torch.float32))
        self.register_buffer("membership", F.one_hot(self.labels, M13.E).float())
        self.mode = "all"

    def forward(self, x):
        b = self.base
        z = b.act_fn(b.gate_proj(x)) * b.up_proj(x)
        if self.mode == "all":
            return b.down_proj(z)
        if self.mode == "fitted":
            scores = F.linear(x, self.router)
        elif self.mode == "oracle":
            scores = z.square() @ self.membership
        else:
            raise ValueError(self.mode)
        chosen = M13.stable_topk(scores, M13.K)
        active = torch.zeros(*chosen.shape[:-1], M13.E,
                             dtype=z.dtype, device=z.device)
        active.scatter_(-1, chosen, 1)
        return b.down_proj(z * active[..., self.labels])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--assets", required=True)
    args = ap.parse_args()
    start = time.monotonic()
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    weight_path = hf_hub_download(M13.MODEL, "model.safetensors",
                                  revision=M13.REV, local_files_only=True)
    assert M13.sha256(weight_path) == M13.MODEL_SHA
    tok = AutoTokenizer.from_pretrained(M13.MODEL, revision=M13.REV,
                                        local_files_only=True)
    assert M13.C.tok_fingerprint(tok) == M13.TOK_FP
    ids, byts, meta = M13.C.get_slice(tok, "heldout", 2, 512, 1234)
    assert meta["ids_sha256"] == M13.ID_SHA["quality"]
    assert ids.shape == (2, 512) and int(byts.sum()) == 4396
    asset_dir = os.path.abspath(args.assets)
    label_path = os.path.join(asset_dir, "meth13_labels_E128.npz")
    router_path = os.path.join(asset_dir, "meth13_routers_E128.npz")
    assert M13.sha256(label_path) == LABEL_SHA
    assert M13.sha256(router_path) == ROUTER_SHA
    with np.load(label_path, allow_pickle=False) as archive:
        labels = [archive[f"c{i}"].copy() for i in range(M13.L)]
    with np.load(router_path, allow_pickle=False) as archive:
        routers = [archive[f"r{i}"].copy() for i in range(M13.L)]
    for lab, router in zip(labels, routers):
        assert lab.shape == (M13.FF,) and router.shape == (M13.E, M13.D)
        assert np.array_equal(np.bincount(lab, minlength=M13.E),
                              np.full(M13.E, M13.G))

    model = AutoModelForCausalLM.from_pretrained(
        M13.MODEL, revision=M13.REV, dtype=torch.float32,
        attn_implementation="eager", local_files_only=True).eval()
    cfg = model.config
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size,
            cfg.vocab_size, cfg.num_key_value_heads) == (M13.L, M13.D,
                                                         M13.FF, 151936, 2)
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = RoutedFFN(layer.mlp, labels[li], routers[li])
        wrappers.append(wrapper)
        layer.mlp = wrapper
    budget(start)
    with torch.no_grad():
        all_logits = model(ids[:1, :16]).logits[0].float()
        for layer, wrapper in zip(model.model.layers, wrappers):
            layer.mlp = wrapper.base
        donor_bpb, donor_first = M13.bpb(model, ids, byts)
        identity = float((all_logits - donor_first).abs().max())
        assert identity <= 1e-4
        assert abs(donor_bpb - DONOR_BPB) <= 1e-4
        budget(start)
        for layer, wrapper in zip(model.model.layers, wrappers):
            layer.mlp = wrapper
            wrapper.mode = "fitted"
        fitted_bpb, fitted_first = M13.bpb(model, ids, byts)
        fitted_change = float((fitted_first - donor_first).abs().max())
        assert abs(fitted_bpb - FITTED_BPB) <= 1e-4 and fitted_change > 1e-4
        budget(start)
        for wrapper in wrappers:
            wrapper.mode = "oracle"
        oracle_bpb, oracle_first = M13.bpb(model, ids, byts)
        oracle_change = float((oracle_first - donor_first).abs().max())
        assert math.isfinite(oracle_bpb) and oracle_change > 1e-4
    elapsed, rss = budget(start)
    oracle_delta = oracle_bpb - donor_bpb
    decision = ("prioritize_router_training_objective" if oracle_delta <= 0.40
                else "stop_router_only_repair_of_this_geometry")
    result = {
        "experiment": "METH-14",
        "source": {"model": M13.MODEL, "revision": M13.REV,
                   "model_sha256": M13.MODEL_SHA,
                   "tokenizer_fingerprint": M13.TOK_FP},
        "assets": {"labels_sha256": LABEL_SHA, "routers_sha256": ROUTER_SHA,
                   "labels_path": label_path, "routers_path": router_path},
        "data": {"part": "heldout", "n_seq": 2, "seq_len": 512,
                 "seed": 1234, "ids_sha256": M13.ID_SHA["quality"],
                 "scored_bytes": int(byts.sum())},
        "runtime": {"torch": torch.__version__, "numpy": np.__version__,
                    "threads": torch.get_num_threads(), "device": "cpu",
                    "elapsed_seconds": elapsed, "rss_at_end_bytes": rss},
        "arms": {"donor_bpb": donor_bpb, "fitted_top32_bpb": fitted_bpb,
                 "oracle_top32_bpb": oracle_bpb,
                 "fitted_delta_bpb": fitted_bpb - donor_bpb,
                 "oracle_delta_bpb": oracle_delta,
                 "oracle_minus_fitted_bpb": oracle_bpb - fitted_bpb,
                 "identity_max_abs": identity,
                 "fitted_change_max_abs": fitted_change,
                 "oracle_change_max_abs": oracle_change},
        "decision": decision,
    }
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result["arms"] | {"decision": decision}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
