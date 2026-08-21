#!/usr/bin/env python3
"""D0 (reframed, 2026-08-21) -- is the donor's active set LAYOUT-compatible with the engine?

The coordinator's prior-art note retired the original question.  MoEfication's own Table 2
shows a random neuron split scoring 95.9/87.3/80.0 against clustered 96.3/89.1/80.8 -- so
"do neurons cluster by co-activation" does not decide anything; the router is the benefit.
What decides it for THIS engine is the rho-law: bulk-contiguous reads of >= 48 KB.

So the question measured here is:

  Q1  In the donor's NATIVE neuron ordering, how many BYTES does an active set arrive in per
      contiguous run?  What fraction of active bytes lands in runs >= 48 / 16 / 4 KB?
  Q2  What does the engine actually FETCH at a >= 48 KB granularity, i.e. what is the
      EFFECTIVE activation fraction after block granularity is charged?  (D5: BS 8/16/32 too.)
  Q3  How much does REORDERING help?  native vs co-activation-clustered vs a random
      permutation NULL.  Published comparator: Neuralink (ASPLOS '25) gets 1.80x by solving
      exactly this as Hamiltonian-path placement; we would need ~5.6x.
  Q4  The engine-legal MoE view: at expert sizes that are themselves >= 48 KB (so contiguity
      is free by construction), what activation fraction k/E reaches what output fidelity --
      co-activation clustering vs the mandatory RANDOM-clustering control.

Byte accounting is read off the artefact and stated, never assumed:
  one FFN hidden neuron costs  gate row + up row + down column = 3 * d_model weights.
  At the engine design point of 0.5 B/weight that is 3*1536*0.5 = 2304 B (interleaved layout)
  or 768 B per organ if the three matrices are streamed separately.  48 KB is therefore
  21.3 neurons interleaved / 64 neurons per organ.

PLANTED CONTROLS (both directions, logged):
  * a mask with runs planted at a known length must be reported at that length;
  * a uniformly random mask at the same density must come back at the analytic run length
    1/(1-p), i.e. the instrument must NOT invent contiguity.
"""
from __future__ import annotations

import sys, os, json, math, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

torch.set_num_threads(int(os.environ.get("D_THREADS", "3")))

LAYERS = [1, 7, 14, 21, 27]
P_ACTIVE = 0.10
BYTES_PER_WEIGHT = 0.5           # engine design point (4-bit codes), stated not assumed
BLOCK_SIZES = [8, 16, 32, 64]    # D5 asks for 8/16/32; 64 is the 48 KB per-organ block
E_LIST = [32, 64, 128]
K_LIST = [1, 2, 4, 8, 16]
N_CAL, N_EVAL, SEQ_LEN = 12, 6, 512
SEED = 909


# ------------------------------------------------------------------ capture / recompute
def capture_mlp_inputs(model, ids, layers, batch=2):
    buf = {L: [] for L in layers}
    hooks = []

    def mk(L):
        def hook(mod, args):
            buf[L].append(args[0].detach().reshape(-1, args[0].shape[-1]).clone())
        return hook

    for L in layers:
        hooks.append(model.model.layers[L].mlp.register_forward_pre_hook(mk(L)))
    for i in range(0, ids.shape[0], batch):
        model.model(ids[i:i + batch])
    for h in hooks:
        h.remove()
    return {L: torch.cat(v, 0) for L, v in buf.items()}


def hidden(model, L, X, chunk=1024):
    mlp = model.model.layers[L].mlp
    return torch.cat([torch.nn.functional.silu(mlp.gate_proj(X[i:i + chunk]))
                      * mlp.up_proj(X[i:i + chunk]) for i in range(0, X.shape[0], chunk)], 0)


def topp_mask(H, p):
    k = max(1, int(round(p * H.shape[1])))
    idx = H.abs().topk(k, dim=1).indices
    B = torch.zeros_like(H, dtype=torch.bool)
    B.scatter_(1, idx, True)
    return B


# ------------------------------------------------------------------ layout instrument
def run_lengths(mask_row: np.ndarray) -> np.ndarray:
    """Lengths (in neurons) of maximal runs of consecutive active neurons."""
    x = np.concatenate(([0], mask_row.astype(np.int8), [0]))
    d = np.diff(x)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return ends - starts


def layout_stats(B: np.ndarray, bytes_per_neuron: float, block_sizes=BLOCK_SIZES,
                 max_tokens: int = 512) -> dict:
    """Run-length + block-granularity accounting for an active mask [T, N]."""
    T, N = B.shape
    T = min(T, max_tokens)
    B = B[:T]
    runs = np.concatenate([run_lengths(B[t]) for t in range(T)])
    run_bytes = runs * bytes_per_neuron
    active = B.sum()
    tot_run_bytes = run_bytes.sum()

    out = {
        "n_tokens": int(T), "n_neurons": int(N),
        "bytes_per_neuron": bytes_per_neuron,
        "active_fraction": float(active / (T * N)),
        "n_runs_per_token": float(len(runs) / T),
        "mean_run_neurons": float(runs.mean()),
        "median_run_neurons": float(np.median(runs)),
        "p99_run_neurons": float(np.percentile(runs, 99)),
        "max_run_neurons": int(runs.max()),
        "mean_bytes_per_read": float(tot_run_bytes / len(runs)),
        "frac_active_bytes_in_runs_ge_4KB": float(run_bytes[run_bytes >= 4096].sum() / tot_run_bytes),
        "frac_active_bytes_in_runs_ge_16KB": float(run_bytes[run_bytes >= 16384].sum() / tot_run_bytes),
        "frac_active_bytes_in_runs_ge_48KB": float(run_bytes[run_bytes >= 49152].sum() / tot_run_bytes),
        "blocks": {},
    }
    for bs in block_sizes:
        nb = N // bs
        Bb = B[:, :nb * bs].reshape(T, nb, bs)
        touched = Bb.any(axis=2)                       # block must be fetched
        eff = touched.mean()                           # fraction of blocks fetched
        out["blocks"][str(bs)] = {
            "block_bytes": bs * bytes_per_neuron,
            "block_skippable_fraction": float(1.0 - eff),
            "effective_activation_fraction": float(eff),
            "byte_amplification_vs_ideal": float(eff / out["active_fraction"]),
            "speedup_vs_dense": float(1.0 / eff),
        }
    return out


# ------------------------------------------------------------------ orderings
def clustered_order(B: torch.Tensor, E: int, seed: int, dim: int = 64):
    """Co-activation clustering, then lay the neurons out cluster-by-cluster."""
    from sklearn.cluster import KMeans
    X = B.float(); X = X - X.mean(0, keepdim=True)
    _, _, Vh = torch.svd_lowrank(X, q=dim, niter=4)
    emb = Vh.numpy()
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    lab = KMeans(n_clusters=E, n_init=4, random_state=seed).fit(emb).labels_
    order = np.argsort(lab, kind="stable")
    return order, lab


def greedy_path_order(B: torch.Tensor, seed: int, n_sample: int = 512):
    """Cheap stand-in for Neuralink's Hamiltonian-path placement: greedy nearest-neighbour
    over the co-activation graph.  An optimistic-but-tractable reordering."""
    Bf = B[:n_sample].float()
    p = Bf.mean(0)
    Cm = (Bf.T @ Bf) / Bf.shape[0]
    Em = p[:, None] * p[None, :]
    S = (Cm / Em.clamp_min(1e-12))                 # lift as affinity
    N = S.shape[0]
    S.fill_diagonal_(-1e9)
    order = [int(torch.argmax(p))]
    used = torch.zeros(N, dtype=torch.bool); used[order[0]] = True
    cur = order[0]
    for _ in range(N - 1):
        row = S[cur].clone()
        row[used] = -1e9
        nxt = int(torch.argmax(row))
        order.append(nxt); used[nxt] = True; cur = nxt
    return np.array(order)


# ------------------------------------------------------------------ MoE routing view
def balanced_labels(B: torch.Tensor, E: int, seed: int, dim: int = 64) -> np.ndarray:
    """Size-BALANCED co-activation clustering.

    Unbalanced clusters would game the gate: top-k over clusters of wildly different sizes
    is not an activation fraction of k/E.  Every expert here holds exactly d_ffn/E neurons,
    so `activation_fraction = k/E` is a fact about the bytes read, not an average.
    """
    from sklearn.cluster import KMeans
    X = B.float(); X = X - X.mean(0, keepdim=True)
    _, _, Vh = torch.svd_lowrank(X, q=dim, niter=4)
    emb = Vh.numpy()
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    km = KMeans(n_clusters=E, n_init=4, random_state=seed).fit(emb)
    N = emb.shape[0]
    cap = N // E
    cost = ((emb[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(-1)   # [N, E]
    order = np.argsort(cost, axis=None)
    lab = np.full(N, -1, dtype=np.int64)
    load = np.zeros(E, dtype=np.int64)
    for flat in order:
        n, e = divmod(int(flat), E)
        if lab[n] == -1 and load[e] < cap:
            lab[n] = e; load[e] += 1
    left = np.flatnonzero(lab < 0)
    for n in left:                                   # remainder neurons -> emptiest cluster
        e = int(np.argmin(load)); lab[n] = e; load[e] += 1
    return lab


def routed_error(model, L, X, labels, k):
    """Relative L2 error of the FFN output when only the top-k experts are computed, with an
    ORACLE router (picks the k experts with the largest ||h restricted||).  The oracle is an
    UPPER BOUND on any trainable router: if it cannot recover the output, nothing can.

    Also returns the TRUE fraction of neurons actually read, which is k/E only when the
    clusters are equal-sized -- reported so an unbalanced partition cannot flatter the result.
    """
    mlp = model.model.layers[L].mlp
    E = int(labels.max()) + 1
    lab = torch.from_numpy(labels.astype(np.int64))
    onehot = torch.zeros(len(lab), E)
    onehot[torch.arange(len(lab)), lab] = 1.0
    sizes = torch.tensor(np.bincount(labels, minlength=E), dtype=torch.float32)
    num = den = 0.0
    kept = 0.0
    ntok = 0
    for i in range(0, X.shape[0], 512):
        x = X[i:i + 512]
        h = torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x)
        y = mlp.down_proj(h)
        sel = ((h ** 2) @ onehot).topk(k, dim=1).indices
        keep = torch.zeros(x.shape[0], E, dtype=torch.bool)
        keep.scatter_(1, sel, True)
        kept += float((keep.float() @ sizes).sum()); ntok += x.shape[0]
        yh = mlp.down_proj(h * keep[:, lab])
        num += float(((yh - y) ** 2).sum()); den += float((y ** 2).sum())
    return math.sqrt(num / den), kept / (ntok * len(lab))


# ------------------------------------------------------------------ planted controls
def planted_controls(bytes_per_neuron):
    rng = np.random.default_rng(0)
    T, N = 256, 8960
    p = P_ACTIVE
    out = []

    # (+) known-positive: runs of exactly 128 neurons planted at 10% density.
    # Starts are drawn from alternate 128-slots so no two planted runs can merge, which
    # makes the expected mean run length exactly 128 with no tolerance needed.
    B = np.zeros((T, N), dtype=bool)
    runlen = 128
    nslots = N // (2 * runlen)
    nruns = max(1, int(round(p * N / runlen)))
    for t in range(T):
        starts = rng.choice(nslots, nruns, replace=False) * 2 * runlen
        for s in starts:
            B[t, s:s + runlen] = True
    st = layout_stats(B, bytes_per_neuron)
    out.append({"name": "planted_runs_128", "expect": "mean_run_neurons == 128 exactly",
                "measured_mean_run": st["mean_run_neurons"],
                "measured_active_fraction": st["active_fraction"],
                "block64_skippable": st["blocks"]["64"]["block_skippable_fraction"],
                "fired": abs(st["mean_run_neurons"] - runlen) < 1e-9})

    # (-) known-negative: i.i.d. mask at the same density must give the analytic run length
    Bn = rng.random((T, N)) < p
    stn = layout_stats(Bn, bytes_per_neuron)
    analytic = 1.0 / (1.0 - p)
    out.append({"name": "iid_mask_same_density",
                "expect": f"mean_run_neurons ~= 1/(1-p) = {analytic:.3f}",
                "measured_mean_run": stn["mean_run_neurons"], "analytic": analytic,
                "block64_skippable": stn["blocks"]["64"]["block_skippable_fraction"],
                "fired": abs(stn["mean_run_neurons"] - analytic) < 0.05 * analytic})
    return out


# ------------------------------------------------------------------ main
def main():
    m, tok = C.load_model()
    A = C.arch(m)
    d_model, d_ffn = A["d_model"], A["d_ffn"]
    bpn_organ = d_model * BYTES_PER_WEIGHT              # one row of one organ
    bpn_joint = 3 * d_model * BYTES_PER_WEIGHT          # gate row + up row + down col

    log = {"arch": A, "bytes_per_weight": BYTES_PER_WEIGHT,
           "bytes_per_neuron_per_organ": bpn_organ,
           "bytes_per_neuron_interleaved": bpn_joint,
           "neurons_per_48KB_per_organ": 49152 / bpn_organ,
           "neurons_per_48KB_interleaved": 49152 / bpn_joint,
           "p_active": P_ACTIVE, "block_sizes": BLOCK_SIZES,
           "controls": [], "layers": []}
    print(f"48 KB = {49152/bpn_organ:.1f} consecutive neurons per organ "
          f"({49152/bpn_joint:.1f} interleaved) at {BYTES_PER_WEIGHT} B/weight", flush=True)

    print("== D0 planted controls (layout instrument) ==", flush=True)
    for c in planted_controls(bpn_organ):
        log["controls"].append(c)
        print(f"  {c['name']:26s} mean_run={c['measured_mean_run']:.3f} "
              f"block64_skippable={c['block64_skippable']:.4f} -> "
              f"{'FIRED' if c['fired'] else 'DID NOT FIRE'}  [{c['expect']}]", flush=True)
    C.dump("d0_layout.json", log)

    cal_ids, _, cal_meta = C.get_slice(tok, "calib", N_CAL, SEQ_LEN, SEED)
    ev_ids, _, ev_meta = C.get_slice(tok, "heldout", N_EVAL, SEQ_LEN, SEED + 1)
    log["cal_slice"], log["eval_slice"] = cal_meta, ev_meta
    with C.Timer("capture"):
        Xcal = capture_mlp_inputs(m, cal_ids, LAYERS)
        Xev = capture_mlp_inputs(m, ev_ids, LAYERS)

    rng = np.random.default_rng(SEED)
    for L in LAYERS:
        t0 = time.time()
        H = hidden(m, L, Xcal[L])
        B = topp_mask(H, P_ACTIVE)
        Bnp = B.numpy()
        rec = {"layer": L, "orderings": {}}

        # ---- Q1/Q2: native order -------------------------------------------------
        rec["orderings"]["native"] = layout_stats(Bnp, bpn_organ)

        # ---- Q3: reorderings, with a random permutation as the NULL ---------------
        perm = rng.permutation(Bnp.shape[1])
        rec["orderings"]["random_permutation_NULL"] = layout_stats(Bnp[:, perm], bpn_organ)

        for E in (64, 128):
            order, lab = clustered_order(B, E, SEED)
            rec["orderings"][f"coactivation_clustered_E{E}"] = layout_stats(Bnp[:, order], bpn_organ)
        with C.Timer(f"L{L} greedy path"):
            gorder = greedy_path_order(B, SEED)
        rec["orderings"]["greedy_coactivation_path"] = layout_stats(Bnp[:, gorder], bpn_organ)

        for name, st in rec["orderings"].items():
            b = st["blocks"]
            print(f"  L{L:02d} {name:32s} mean_read={st['mean_bytes_per_read']:8.0f}B "
                  f">=48KB={st['frac_active_bytes_in_runs_ge_48KB']:.4f} "
                  f"skip@BS8={b['8']['block_skippable_fraction']:.3f} "
                  f"BS16={b['16']['block_skippable_fraction']:.3f} "
                  f"BS32={b['32']['block_skippable_fraction']:.3f} "
                  f"BS64={b['64']['block_skippable_fraction']:.3f} "
                  f"speedup@BS64={b['64']['speedup_vs_dense']:.2f}x", flush=True)

        # ---- Q4: engine-legal MoE view -------------------------------------------
        rec["routing"] = []
        for E in E_LIST:
            parts = {}
            _order, lab_km = clustered_order(B, E, SEED)
            parts["coactivation_kmeans"] = lab_km
            lab_bal = balanced_labels(B, E, SEED)
            parts["coactivation_balanced"] = lab_bal
            lab_rnd = lab_bal.copy(); rng.shuffle(lab_rnd)
            parts["random_balanced_NULL"] = lab_rnd
            for pname, lab in parts.items():
                sizes = np.bincount(lab, minlength=E)
                for k in K_LIST:
                    if k >= E:
                        continue
                    err, kept = routed_error(m, L, Xev[L], lab, k)
                    rec["routing"].append({
                        "E": E, "k": k, "partition": pname,
                        "nominal_activation_fraction": k / E,
                        "true_activation_fraction": kept,
                        "expert_bytes_per_organ": (d_ffn / E) * bpn_organ,
                        "engine_legal_48KB": bool((d_ffn / E) * bpn_organ >= 49152),
                        "relerr": err,
                        "cluster_size_min": int(sizes.min()),
                        "cluster_size_max": int(sizes.max())})
                    print(f"  L{L:02d} MoE E={E:3d} k={k:2d} {pname:22s} "
                          f"nom={k/E:6.3f} true={kept:6.3f} "
                          f"expert={(d_ffn/E)*bpn_organ/1024:5.1f}KB relerr={err:.4f} "
                          f"sizes[{sizes.min()},{sizes.max()}]", flush=True)
        log["layers"].append(rec)
        C.dump("d0_layout.json", log)
        print(f"  L{L:02d} done {time.time()-t0:.0f}s", flush=True)

    print("wrote", C.dump("d0_layout.json", log))


if __name__ == "__main__":
    main()
