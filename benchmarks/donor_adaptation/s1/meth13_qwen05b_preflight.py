#!/usr/bin/env python3
"""METH-13 CPU preflight for a progressive Qwen0.5B E128 upcycle."""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import psutil
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DEN = os.path.abspath(os.path.join(HERE, "..", "density"))
TERN = os.path.abspath(os.path.join(HERE, "..", "ternary"))
ENG = os.path.abspath(os.path.join(HERE, "..", "engine"))
for p in (DEN, TERN, ENG):
    sys.path.insert(0, p)
import common as C
import d0_layout as D0
import e23_router as E23
import carve_common as CV

MODEL = "Qwen/Qwen2.5-0.5B"
REV = "060db6499f32faf8b98477b0a26969ef7d8b9987"
MODEL_SHA = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
TOK_FP = "4efeeb9382a77a06"
ID_SHA = {
    "label": "40f40c2f29a7c0507bf41d1ae35d4342151870ebafd12f2d59d7e101997a7509",
    "fit": "b3c06b88955d9ab6593a52df2ff6005ccca2532ff69f844038b399a4fbe56ea8",
    "route": "938abee0680d9c15df08d4f669e7a8dd9a158621779763464cc0d05a65055259",
    "quality": "3808624ff6fc5cb8fa5911c9fbfb37a8bcbf4f08db83bb5b146887c267a9e62c",
}
E, K, G, D, FF, L = 128, 32, 38, 896, 4864, 24
ACTIVE_NEURONS = 487
CLUSTER_SEED = 4242
DEADLINE = 1800.0
RAM_MAX = 20 * (1 << 30)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def budget(t0):
    elapsed = time.monotonic() - t0
    rss = psutil.Process().memory_info().rss
    if elapsed > DEADLINE:
        raise TimeoutError("METH-13 30-minute CPU stop")
    if rss > RAM_MAX:
        raise MemoryError("METH-13 20-GiB process RAM stop: %d" % rss)
    return rss


def stable_topk(x, k):
    return torch.argsort(x, dim=-1, descending=True, stable=True)[..., :k]


def labels_from_calib(model, ids, t0):
    parts = {i: [] for i in range(L)}
    hooks = []
    for li, layer in enumerate(model.model.layers):
        def mk(li=li):
            def hook(_mod, args):
                z = args[0].detach().reshape(-1, FF)
                idx = stable_topk(z.abs(), ACTIVE_NEURONS)
                mask = torch.zeros(z.shape[0], FF, dtype=torch.bool)
                mask.scatter_(1, idx, True)
                parts[li].append(mask)
            return hook
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(mk()))
    try:
        with torch.no_grad():
            for i in range(ids.shape[0]):
                model(ids[i:i+1])
                print("label capture", i+1, "/", ids.shape[0], flush=True)
                budget(t0)
    finally:
        for h in hooks:
            h.remove()
    labels = {}
    for li in range(L):
        b = torch.cat(parts[li], dim=0)
        assert b.shape == (4*256, FF)
        lab = np.asarray(D0.balanced_labels(b, E, CLUSTER_SEED), dtype=np.int64)
        assert lab.shape == (FF,)
        assert np.array_equal(np.bincount(lab, minlength=E), np.full(E, G))
        labels[li] = lab
        parts[li].clear()
        print("balanced labels layer", li, flush=True)
        budget(t0)
    return labels


def route_audit(model, ids, labels, routers, t0):
    oh = {}
    rnd = {}
    sums = {}
    pending = {}
    hooks = []
    for li in range(L):
        lab = torch.as_tensor(labels[li], dtype=torch.long)
        mat = F.one_hot(lab, E).float()
        oh[li] = mat
        rnd[li] = torch.as_tensor(CV.router_weights(D, E, 26, li))
        sums[li] = {"fitted_recall": 0.0, "random_recall": 0.0,
                    "fitted_mass": 0.0, "random_mass": 0.0, "tokens": 0}

        def mk_in(li=li):
            def hook(_mod, args):
                pending[li] = args[0].detach().reshape(-1, D)
            return hook

        def mk_hidden(li=li):
            def hook(_mod, args):
                x = pending.pop(li)
                z = args[0].detach().reshape(-1, FF)
                mass = (z.square() @ oh[li]).clamp_min(0)
                oracle = stable_topk(mass, K)
                total = mass.gather(1, oracle).sum(1).clamp_min(1e-30)
                for name, wr in (("fitted", routers[li]), ("random", rnd[li])):
                    selected = stable_topk(F.linear(x, wr), K)
                    hit = (selected.unsqueeze(2) == oracle.unsqueeze(1)).any(2).float().sum(1) / K
                    got = mass.gather(1, selected).sum(1) / total
                    sums[li][name + "_recall"] += float(hit.sum())
                    sums[li][name + "_mass"] += float(got.sum())
                sums[li]["tokens"] += x.shape[0]
            return hook
        layer = model.model.layers[li]
        hooks.append(layer.mlp.register_forward_pre_hook(mk_in()))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(mk_hidden()))
    try:
        with torch.no_grad():
            for i in range(ids.shape[0]):
                model(ids[i:i+1])
                print("route audit", i+1, "/", ids.shape[0], flush=True)
                budget(t0)
    finally:
        for h in hooks:
            h.remove()
    rows = []
    for li in range(L):
        n = sums[li].pop("tokens")
        assert n == 4*256
        rows.append({"layer": li, "tokens": n,
                     **{k: v/n for k, v in sums[li].items()}})
    avg = {key: sum(row[key] for row in rows)/L
           for key in ("fitted_recall", "random_recall", "fitted_mass", "random_mass")}
    avg["mass_advantage"] = avg["fitted_mass"] - avg["random_mass"]
    avg["pass"] = bool(avg["mass_advantage"] >= 0.15 and avg["fitted_mass"] >= 0.60)
    return rows, avg


class CarvedFFN(nn.Module):
    def __init__(self, base, lab, router):
        super().__init__()
        self.base = base
        self.register_buffer("labels", torch.as_tensor(lab.copy(), dtype=torch.long))
        self.register_buffer("router", router.clone())
        self.all_active = False

    def forward(self, x):
        b = self.base
        z = b.act_fn(b.gate_proj(x)) * b.up_proj(x)
        if self.all_active:
            return b.down_proj(z)
        chosen = stable_topk(F.linear(x, self.router), K)
        active = torch.zeros(*chosen.shape[:-1], E, dtype=z.dtype)
        active.scatter_(-1, chosen, 1)
        return b.down_proj(z * active[..., self.labels])


@torch.no_grad()
def bpb(model, ids, byts):
    nats = 0.0
    first = None
    for i in range(len(ids)):
        logits = model(ids[i:i+1]).logits.float()
        if i == 0:
            first = logits[0, :16].clone()
        lp = F.log_softmax(logits[:, :-1], dim=-1)
        nats += float(-lp.gather(-1, ids[i:i+1, 1:].unsqueeze(-1)).sum())
        print("BPB window", i+1, flush=True)
    return nats/(math.log(2)*float(byts.sum())), first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--assets", required=True)
    args = ap.parse_args()
    t0 = time.monotonic()
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    weight_path = hf_hub_download(MODEL, "model.safetensors", revision=REV, local_files_only=True)
    assert sha256(weight_path) == MODEL_SHA
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, local_files_only=True)
    assert C.tok_fingerprint(tok) == TOK_FP
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REV, dtype=torch.float32, attn_implementation="eager",
        local_files_only=True).eval()
    cfg = model.config
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size,
            cfg.vocab_size, cfg.num_key_value_heads) == (L,D,FF,151936,2)
    data = {}
    for name, part, n, length, seed in (
        ("label", "calib", 4,256,42424), ("fit", "calib", 16,256,42424),
        ("route", "heldout", 4,256,1234), ("quality", "heldout", 2,512,1234)):
        ids, byts, meta = C.get_slice(tok, part, n, length, seed)
        assert meta["ids_sha256"] == ID_SHA[name] and ids.shape == (n,length)
        data[name] = (ids, byts, meta)
    assert int(data["quality"][1].sum()) == 4396
    budget(t0)
    labels = labels_from_calib(model, data["label"][0], t0)
    routers_raw, router_fit_diag, _ = E23.fit_routers(
        model, data["fit"][0], list(range(L)), labels)
    routers = {li: routers_raw[li].T.contiguous() for li in range(L)}
    print("ridge routers fitted on 16x256", flush=True)
    budget(t0)
    route_rows, route_mean = route_audit(model, data["route"][0], labels, routers, t0)
    print("route mean", json.dumps(route_mean), flush=True)
    budget(t0)
    asset_dir = os.path.abspath(args.assets)
    os.makedirs(asset_dir, exist_ok=True)
    lp = os.path.join(asset_dir, "meth13_labels_E128.npz")
    rp = os.path.join(asset_dir, "meth13_routers_E128.npz")
    np.savez_compressed(lp, **{f"c{li}": labels[li] for li in range(L)})
    np.savez_compressed(rp, **{f"r{li}": routers[li].numpy() for li in range(L)})
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        w = CarvedFFN(layer.mlp, labels[li], routers[li])
        wrappers.append(w)
        layer.mlp = w
        w.all_active = True
    quality_ids, quality_byts, _ = data["quality"]
    with torch.no_grad():
        all_logits = model(quality_ids[:1,:16]).logits[0].float()
        for li, layer in enumerate(model.model.layers):
            layer.mlp = wrappers[li].base
        donor_bpb, donor_first = bpb(model, quality_ids, quality_byts)
        identity = float((all_logits-donor_first).abs().max())
        assert identity <= 1e-4
        for li, layer in enumerate(model.model.layers):
            layer.mlp = wrappers[li]
        for w in wrappers:
            w.all_active = False
        sparse_bpb, sparse_first = bpb(model, quality_ids, quality_byts)
    changed = float((sparse_first-donor_first).abs().max())
    assert changed > 1e-4 and math.isfinite(sparse_bpb)
    delta = sparse_bpb-donor_bpb
    decision = ("eligible_for_frozen_joint_training_design" if route_mean["pass"] and delta <= 0.40
                else "stop_this_E128_top32_geometry_before_GPU")
    result = {
        "experiment": "METH-13", "source": {"model": MODEL, "revision": REV,
            "model_sha256": MODEL_SHA, "tokenizer_fingerprint": TOK_FP},
        "runtime": {"torch": torch.__version__, "numpy": np.__version__,
                    "sklearn": sklearn.__version__},
        "data": {name: {"part": meta["part"], "n_seq": meta["n_seq"],
                           "seq_len": meta["seq_len"], "seed": meta["seed"],
                           "ids_sha256": meta["ids_sha256"],
                           "scored_bytes": int(byts.sum())}
                 for name, (_,byts,meta) in data.items()},
        "geometry": {"layers": L,"D": D,"F": FF,"E": E,"k": K,
            "group_width": G,"label_activity_top_neurons": ACTIVE_NEURONS,
            "cluster_seed": CLUSTER_SEED,"router_ridge_frac": E23.RIDGE_FRAC},
        "assets": {"labels_path": lp,"labels_sha256": sha256(lp),
                   "routers_path": rp,"routers_sha256": sha256(rp)},
        "router_fit": {str(li): router_fit_diag[li] for li in range(L)},
        "route_rows": route_rows,"route_mean": route_mean,
        "pilot": {"donor_bpb": donor_bpb,"hard_top32_bpb": sparse_bpb,
                  "delta_bpb": delta,"identity_max_abs": identity,
                  "sparse_change_max_abs": changed},
        "decision": decision,"elapsed_seconds": time.monotonic()-t0,
        "rss_at_end_bytes": psutil.Process().memory_info().rss,
    }
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out),exist_ok=True)
    with open(out,"w",encoding="utf-8") as f:
        json.dump(result,f,indent=2)
        f.write("\n")
    print(json.dumps({"route_mean": route_mean,"pilot": result["pilot"],
                      "decision": decision,"elapsed_seconds": result["elapsed_seconds"]},indent=2),flush=True)


if __name__ == "__main__":
    main()
