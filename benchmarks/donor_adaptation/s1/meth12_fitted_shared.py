#!/usr/bin/env python3
"""METH-12 CPU transfer screen; frozen protocol is in docs/research/NATIVE_EXPERT_SCALING_20260925."""
import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.abspath(os.path.join(HERE, "..", "engine"))
DEN = os.path.abspath(os.path.join(HERE, "..", "density"))
for path in (ENG, DEN):
    sys.path.insert(0, path)
import common as C
from e68_shared_residual import _ridge_basis, _fit_reduced_rank_path

BUNDLE = os.path.join(HERE, "_h1_bundle")
ROUTERS = "D:/_ktmp/e37/e37_routers_E256.npz"
EXPECTED = {
    "labels": "c39d0740b7f754daf168d811c77120c5394ae677e3882fef208d331d8c333c9c",
    "routers": "42b12cb9d4bda2da7833cd0e179a9dab8e3f43264ce640e2b5db31d5fd043d8a",
    "calib_ids": "3075c14d95a05dd69387e3df040370b053c16108bac409fa582db038f89be46e",
    "heldout_ids": "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65",
}
E, K, G, D, RANK = 256, 16, 35, 1536, 256
CALIB_FIT, CALIB_VAL = 12, 4
DEADLINE = 1800.0
RAM_MAX = 40 * (1 << 30)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_budget(t0):
    if time.monotonic() - t0 > DEADLINE:
        raise TimeoutError("METH-12 30-minute CPU stop")
    rss = psutil.Process().memory_info().rss
    if rss > RAM_MAX:
        raise MemoryError("METH-12 40-GiB process RAM stop: %d" % rss)
    return rss


def selected_mask(x, labels, router):
    flat = x.reshape(-1, D)
    scores = F.linear(flat, router)
    # Router rows follow original group IDs 0..255, so stable sort breaks ties low.
    chosen = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :K]
    groups = torch.zeros(flat.shape[0], E, dtype=x.dtype)
    groups.scatter_(1, chosen, 1)
    return groups[:, labels].reshape(*x.shape[:-1], E * G), chosen


class Capture:
    def __init__(self, layer, labels, router):
        self.layer = layer
        self.labels = labels
        self.router = router
        self.x_parts = []
        self.r_parts = []
        self.pending_x = None
        self.pending_z = None
        self.max_identity = 0.0
        mlp = layer.mlp
        self.handles = [mlp.register_forward_pre_hook(self.on_input),
                        mlp.down_proj.register_forward_pre_hook(self.on_hidden),
                        mlp.register_forward_hook(self.on_output)]

    def on_input(self, _mod, args):
        assert self.pending_x is None and self.pending_z is None
        self.pending_x = args[0].detach()

    def on_hidden(self, _mod, args):
        assert self.pending_x is not None and self.pending_z is None
        self.pending_z = args[0].detach()

    def on_output(self, _mod, _args, y):
        x, z = self.pending_x, self.pending_z
        assert x.shape == (1, 512, D) and z.shape == (1, 512, E * G)
        exact = F.linear(z, self.layer.mlp.down_proj.weight)
        self.max_identity = max(self.max_identity, float((exact - y).abs().max()))
        if self.max_identity > 1e-4:
            raise RuntimeError("dense FFN capture identity failed: %g" % self.max_identity)
        mask, _ = selected_mask(x, self.labels, self.router)
        sparse = F.linear(z * mask, self.layer.mlp.down_proj.weight)
        self.x_parts.append(x[0].cpu().clone())
        self.r_parts.append((y - sparse)[0].cpu().clone())
        self.pending_x = self.pending_z = None

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


class FittedResidualFFN(nn.Module):
    def __init__(self, base, labels, router, a, vt):
        super().__init__()
        self.base = base
        self.register_buffer("labels", labels.clone())
        self.register_buffer("router", router.clone())
        self.register_buffer("a", a.clone())
        self.register_buffer("vt", vt.clone())
        self.all_active = False

    def forward(self, x):
        b = self.base
        z = b.act_fn(b.gate_proj(x)) * b.up_proj(x)
        if self.all_active:
            return b.down_proj(z)
        mask, _ = selected_mask(x, self.labels, self.router)
        sparse = b.down_proj(z * mask)
        shared = (x @ self.a) @ self.vt
        return sparse + shared


@torch.no_grad()
def score(model, ids, byts):
    nats = 0.0
    first = None
    for i in range(ids.shape[0]):
        logits = model(ids[i:i+1]).logits.float()
        if i == 0:
            first = logits[0, :16].clone()
        lp = F.log_softmax(logits[:, :-1], dim=-1)
        nats += float(-lp.gather(-1, ids[i:i+1, 1:].unsqueeze(-1)).sum())
        print("scored window", i + 1, flush=True)
    return nats / (math.log(2) * float(byts.sum())), first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--factors", required=True)
    args = ap.parse_args()
    t0 = time.monotonic()
    torch.set_num_threads(6)
    assert C.REVISION == "8faed761d45a263340a0528343f099c05c9a4323"
    label_path = os.path.join(BUNDLE, "labels_E256.npz")
    assert sha256(label_path) == EXPECTED["labels"]
    assert sha256(ROUTERS) == EXPECTED["routers"]
    labels_np, routers_np = np.load(label_path), np.load(ROUTERS)
    model, tok = C.load_model(dtype=torch.float32)
    calib, _, cmeta = C.get_slice(tok, "calib", CALIB_FIT+CALIB_VAL, 512, 424242)
    heldout, byts, hmeta = C.get_slice(tok, "heldout", 24, 512, 1234)
    assert cmeta["ids_sha256"] == EXPECTED["calib_ids"]
    assert hmeta["ids_sha256"] == EXPECTED["heldout_ids"]
    assert np.array_equal(heldout.numpy(), np.load(os.path.join(BUNDLE, "h1_heldout.npz"))["ids"])
    assert calib.shape == (16, 512) and heldout.shape == (24, 512)
    assert int(byts[:2].sum()) == 4396
    check_budget(t0)
    captures = []
    labels, routers = [], []
    for li, layer in enumerate(model.model.layers):
        lab = torch.as_tensor(labels_np[f"c{li}"].copy(), dtype=torch.long)
        router = torch.as_tensor(routers_np[f"r{li}"].copy(), dtype=torch.float32)
        assert lab.shape == (E*G,) and router.shape == (E,D)
        assert torch.equal(torch.bincount(lab, minlength=E), torch.full((E,), G))
        labels.append(lab)
        routers.append(router)
        captures.append(Capture(layer, lab, router))
    try:
        with torch.no_grad():
            for i in range(calib.shape[0]):
                model(calib[i:i+1])
                print("captured calibration", i+1, "/", len(calib), flush=True)
                check_budget(t0)
    finally:
        for cap in captures:
            cap.close()
    fit_rows = []
    factor_arrays = {}
    wrappers = []
    for li, cap in enumerate(captures):
        xtr = torch.cat(cap.x_parts[:CALIB_FIT], dim=0).double()
        rtr = torch.cat(cap.r_parts[:CALIB_FIT], dim=0).double()
        xv = torch.cat(cap.x_parts[CALIB_FIT:], dim=0)
        rv = torch.cat(cap.r_parts[CALIB_FIT:], dim=0)
        basis = _ridge_basis(xtr)
        pairs, fit_meta = _fit_reduced_rank_path(basis, rtr, ranks=(0, RANK))
        a, vt = pairs[RANK].score_factors_fp32()
        pred = (xv @ a) @ vt
        sse0 = float(rv.square().sum())
        sse1 = float((rv-pred).square().sum())
        assert math.isfinite(sse0) and math.isfinite(sse1) and sse0 > 0
        assert torch.isfinite(a).all() and torch.isfinite(vt).all()
        factor_arrays[f"L{li:02d}.a"] = a.numpy()
        factor_arrays[f"L{li:02d}.vt"] = vt.numpy()
        wrappers.append(FittedResidualFFN(model.model.layers[li].mlp,
                                          labels[li], routers[li], a, vt))
        fit_rows.append({"layer": li, "validation_sparse_sse": sse0,
                         "validation_fitted_sse": sse1,
                         "validation_ratio": sse1/sse0,
                         "lambda": fit_meta["lambda"],
                         "dense_capture_max_abs": cap.max_identity})
        cap.x_parts.clear(); cap.r_parts.clear()
        del xtr, rtr, xv, rv, basis, pairs, pred, a, vt
        gc.collect()
        print("fit layer", li, "validation ratio", round(sse1/sse0, 5), flush=True)
        check_budget(t0)
    os.makedirs(os.path.dirname(os.path.abspath(args.factors)), exist_ok=True)
    np.savez_compressed(args.factors, **factor_arrays)
    factor_sha = sha256(args.factors)
    del factor_arrays
    gc.collect()
    check_budget(t0)
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            layer.mlp = wrappers[li]
        for w in wrappers:
            w.all_active = True
        full_logits = model(heldout[:1, :16]).logits[0].float()
        for li, layer in enumerate(model.model.layers):
            layer.mlp = wrappers[li].base
        dense_bpb, dense_first = score(model, heldout[:2], byts[:2])
        max_control = float((full_logits - dense_first).abs().max())
        assert max_control <= 1e-4, max_control
        for li, layer in enumerate(model.model.layers):
            layer.mlp = wrappers[li]
        for w in wrappers:
            w.all_active = False
        compact_bpb, _ = score(model, heldout[:2], byts[:2])
    delta = compact_bpb-dense_bpb
    result = {
        "experiment": "METH-12", "donor": C.MODEL_ID, "revision": C.REVISION,
        "protocol": "METH_12_FITTED_SHARED_PROTOCOL_20260926.md",
        "source_ids": {"calib_16x512": cmeta["ids_sha256"],
                       "heldout_24x512": hmeta["ids_sha256"]},
        "source_hashes": {"labels_npz": EXPECTED["labels"],
                          "routers_npz": EXPECTED["routers"],
                          "factors_npz": factor_sha},
        "factors_path": os.path.abspath(args.factors),
        "geometry": {"E": E, "top_k": K, "neurons_per_expert": G,
                     "shared_rank": RANK, "calibration_fit_windows": CALIB_FIT,
                     "calibration_validation_windows": CALIB_VAL},
        "fit_validation": fit_rows,
        "pilot": {"windows": 2, "scored_bytes": int(byts[:2].sum()),
                  "donor_bpb": dense_bpb, "fitted_shared_bpb": compact_bpb,
                  "delta_bpb": delta, "all_active_max_abs_logit_diff": max_control},
        "decision": "consider_joint_training_design" if delta <= 0.20 else
                    "reject_this_geometry_before_training",
        "elapsed_seconds": time.monotonic()-t0,
        "rss_at_end_bytes": psutil.Process().memory_info().rss,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"pilot": result["pilot"], "decision": result["decision"],
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
