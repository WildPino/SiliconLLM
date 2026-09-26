#!/usr/bin/env python3
"""METH-11 frozen CPU-fp32 screen; see METH_11_SHARED_RESIDUAL_PROTOCOL_20260926.md."""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "density")))
import common as C

BUNDLE = os.path.join(HERE, "_h1_bundle")
ROUTERS = "D:/_ktmp/e37/e37_routers_E256.npz"
EXPECTED_ROUTER_SHA = "42b12cb9d4bda2da7833cd0e179a9dab8e3f43264ce640e2b5db31d5fd043d8a"
EXPECTED_IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
DONOR_REV = "8faed761d45a263340a0528343f099c05c9a4323"
N_GROUPS, N_SHARED, N_RESIDUAL_ACTIVE, GROUP_SIZE = 256, 64, 16, 35


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


class SharedResidualFFN(nn.Module):
    def __init__(self, base, labels, shared, router):
        super().__init__()
        self.base = base
        self.register_buffer("labels", torch.as_tensor(labels.copy(), dtype=torch.long))
        self.register_buffer("shared", torch.as_tensor(shared.copy(), dtype=torch.long))
        residual = np.setdiff1d(np.arange(N_GROUPS), shared, assume_unique=True)
        self.register_buffer("residual", torch.as_tensor(residual, dtype=torch.long))
        self.register_buffer("router", torch.as_tensor(router.copy(), dtype=torch.float32))
        self.all_active = False

    def forward(self, x):
        b = self.base
        h = b.act_fn(b.gate_proj(x)) * b.up_proj(x)
        if not self.all_active:
            rows = x.reshape(-1, x.shape[-1])
            scores = F.linear(rows, self.router.index_select(0, self.residual))
            # residual IDs are ascending. Stable descending sort gives lower-ID ties.
            selected = self.residual[torch.argsort(scores, dim=1, descending=True,
                                                   stable=True)[:, :N_RESIDUAL_ACTIVE]]
            group_mask = torch.zeros(rows.shape[0], N_GROUPS, dtype=h.dtype)
            group_mask[:, self.shared] = 1
            group_mask.scatter_(1, selected, 1)
            h = h * group_mask[:, self.labels].reshape_as(h)
        return b.down_proj(h)


@torch.no_grad()
def score(model, ids, byte_counts):
    nats = 0.0
    first_logits = None
    for i in range(ids.shape[0]):
        logits = model(ids[i:i + 1]).logits.float()
        if i == 0:
            first_logits = logits[0, :16].clone()
        lp = F.log_softmax(logits[:, :-1], dim=-1)
        nats += float(-lp.gather(-1, ids[i:i + 1, 1:].unsqueeze(-1)).sum())
        print("scored window", i + 1, flush=True)
    return nats / (math.log(2) * float(byte_counts.sum())), first_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(6)
    assert C.REVISION == DONOR_REV
    assert sha256(ROUTERS) == EXPECTED_ROUTER_SHA
    t0 = time.time()
    model, tok = C.load_model(dtype=torch.float32)
    ids, byts, meta = C.get_slice(tok, "heldout", 24, 512, 1234)
    assert meta["ids_sha256"] == EXPECTED_IDS_SHA
    frozen = np.load(os.path.join(BUNDLE, "h1_heldout.npz"))
    assert np.array_equal(ids.numpy(), frozen["ids"])
    ids, byts = ids[:2], byts[:2]
    assert ids.shape == (2, 512) and (byts > 0).all()
    labels_np = np.load(os.path.join(BUNDLE, "labels_E256.npz"))
    stats_np = np.load(os.path.join(BUNDLE, "h1_actstats.npz"))
    router_np = np.load(ROUTERS)
    plan = []
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        labels = labels_np[f"c{li}"].astype(np.int64)
        rms = stats_np[f"L{li:02d}.rms_h"].astype(np.float64)
        router = router_np[f"r{li}"].astype(np.float32)
        assert labels.shape == rms.shape == (8960,)
        assert router.shape == (N_GROUPS, 1536)
        assert np.array_equal(np.bincount(labels, minlength=N_GROUPS),
                              np.full(N_GROUPS, GROUP_SIZE))
        down = layer.mlp.down_proj.weight.detach()
        importance = (down.square().sum(0).double().numpy() * rms**2)
        group_weight = np.bincount(labels, weights=importance, minlength=N_GROUPS)
        assert np.isfinite(group_weight).all()
        shared = np.lexsort((np.arange(N_GROUPS), -group_weight))[:N_SHARED]
        shared.sort()
        plan.append({"layer": li, "shared_original_group_ids": shared.tolist(),
                     "shared_proxy_mass_fraction": float(group_weight[shared].sum() /
                                                         group_weight.sum())})
        wrapper = SharedResidualFFN(layer.mlp, labels, shared, router)
        wrappers.append(wrapper)
        layer.mlp = wrapper
    result = {
        "experiment": "METH-11", "status": "plan-only" if args.plan_only else "pilot",
        "donor": C.MODEL_ID, "revision": C.REVISION,
        "source_ids_sha256": meta["ids_sha256"], "pilot_windows": 2,
        "pilot_scored_bytes": int(byts.sum()), "precision": "CPU-fp32",
        "labels_sha256": sha256(os.path.join(BUNDLE, "labels_E256.npz")),
        "stats_sha256": sha256(os.path.join(BUNDLE, "h1_actstats.npz")),
        "router_sha256": EXPECTED_ROUTER_SHA,
        "geometry": {"groups_per_layer": N_GROUPS, "neurons_per_group": GROUP_SIZE,
                     "shared_groups": N_SHARED, "residual_experts": N_GROUPS-N_SHARED,
                     "active_residual_experts": N_RESIDUAL_ACTIVE,
                     "active_ffn_neurons_per_layer": (N_SHARED+N_RESIDUAL_ACTIVE)*GROUP_SIZE},
        "plan": plan,
    }
    if not args.plan_only:
        # Start with the exact planted control, then the actual sparse arm.
        for w in wrappers:
            w.all_active = True
        all_logits = model(ids[:1, :16]).logits[0].float()
        for li, layer in enumerate(model.model.layers):
            layer.mlp = wrappers[li].base
        dense_bpb, dense_first = score(model, ids, byts)
        control_max = float((all_logits - dense_first).abs().max())
        assert math.isfinite(control_max) and control_max <= 1e-4, control_max
        for li, layer in enumerate(model.model.layers):
            layer.mlp = wrappers[li]
        for w in wrappers:
            w.all_active = False
        conditional_bpb, _ = score(model, ids, byts)
        assert math.isfinite(dense_bpb) and math.isfinite(conditional_bpb)
        result["scores"] = {"donor_bpb": dense_bpb,
                            "shared_residual_bpb": conditional_bpb,
                            "delta_bpb": conditional_bpb-dense_bpb,
                            "all_active_max_abs_logit_diff": control_max}
        result["decision"] = ("consider_bounded_joint_training" if conditional_bpb-dense_bpb <= 0.20
                              else "reject_this_geometry_before_training")
    result["elapsed_seconds"] = time.time()-t0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "scores", "decision") if k in result},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
