#!/usr/bin/env python3
# Canonical val-set + fp32-CPU re-evaluation of every promoted checkpoint (P4.1).
#
# WHY: the probe-era BPB numbers were produced by per-phase harnesses (different eval slices,
# some under autocast bf16 on GPU). Same-slice DELTAS in each probe are valid; the ABSOLUTE
# numbers are slice- and dtype-sensitive. This script freezes ONE canonical slice and ONE
# harness (fp32, CPU, no autocast) and re-evaluates every promoted checkpoint on it.
# The resulting table is the single source of public absolute BPB numbers.
#
# ANTI-GOODHART RULE (non-negotiable): pre-registered gates are read with the SAME harness
# that produced their anchor. Phase 61's gate (BPB <= 0.8799+0.010) is read with the
# phase61 training-script val (the harness that produced the 0.8799 anchor) — NEVER with
# this canonical harness. A gate is never compared across two harnesses. The canonical
# number is the public absolute; it anchors FUTURE gates only.
#
# CANONICAL SLICE (frozen):
#   ids     = results/phase55/ids.u16 (uint16 BPE-1024 token stream; md5 printed & recorded)
#   split   = train ids[:int(n*0.9)] | val ids[int(n*0.9):]   (the project-wide 90/10 split)
#   slice   = first 390 windows of 512 tokens of val (targets val[pos+1..pos+512], pos=0,512,...)
#           = 199,680 evaluated tokens; BPB = sum CE(bits) / sum target-byte-lengths
#   harness = fp32 weights, CPU, no autocast, torch.no_grad, windows batched (batch dim only;
#             math identical to per-window eval)
#
# Run:  python scripts/canonical_eval.py            (all promoted checkpoints)
#       python scripts/canonical_eval.py --only sp58_base moe_gran
# Writes: results/canonical/canonical_bpb.json + prints the table (doc: docs/CANONICAL_EVAL.md)
import argparse, hashlib, json, math, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase55"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase57"))
from phase55_ssm import ArchA, load_meta, IDS, META            # noqa: E402
from phase57_ternary import ternarize_mlp                      # noqa: E402
from phase57_sparse import sparsify_mlp                        # noqa: E402
from phase59_moe import MoEMLP, DenseExpert, moe_forward      # noqa: E402

WIN = 512
NWIN = 390          # 390*512 = 199,680 tokens — the canonical slice
OUTDIR = os.path.join(ROOT, "results", "canonical")

# label -> (path relative to ROOT, builder kind, provenance note)
CKPTS = [
    ("archA_5m_fp32",    "results/phase55/archA_5m.pt",          "plain",   "probe-1 fp32 arm (phase-55 5M, plain SiLU MLP)"),
    ("archA_5m_ternary", "results/phase57/archA_5m_ternmlp.pt",  "ternary", "probe-1 ternary arm (BitLinear158 MLP)"),
    ("sp_silu",          "results/phase57/sp_silu.pt",           "sparse",  "probe-2 SiLU(SwiGLU) arm"),
    ("sp_drelu",         "results/phase57/sp_drelu.pt",          "sparse",  "probe-2 dReLU arm"),
    ("sp58_base",        "results/phase57/sp58_base.pt",         "sparse",  "phase-58 base (= probe-4 arm A, engine E1-E3.5 target)"),
    ("sp58_reg",         "results/phase57/sp58_reg.pt",          "sparse",  "phase-58 +coherence arm"),
    ("moe_dense_big",    "results/phase57/moe_dense_big.pt",     "moe",     "probe-4 arm B (dense-4096)"),
    ("moe_gran",         "results/phase57/moe_gran.pt",          "moe",     "probe-4 arm C (E32xh128 top-8, engine E4 target)"),
    ("moe_coarse",       "results/phase57/moe_coarse.pt",        "moe",     "probe-4 arm D (E8xh512 top-2)"),
    # p61_* enter this table only when Phase 61 closes (its GATE is read on the phase61
    # training-script harness, per the anti-Goodhart rule above).
]

ARCH_KEYS = ("D", "N", "H", "L", "swa_layer", "use_mlp", "mlp_mult", "dt_rank")


def md5_file(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_model(kind, V, cfg):
    ac = {k: cfg[k] for k in ARCH_KEYS if k in cfg}
    model = ArchA(V, **ac)
    D = ac["D"]
    if kind == "plain":
        pass
    elif kind == "ternary":
        ternarize_mlp(model, per_row=(cfg.get("scale", "per-row") == "per-row"))
    elif kind == "sparse":
        hid = ac.get("mlp_mult", 4) * D
        sparsify_mlp(model, D, hid, act=cfg["act"], gated=cfg["gated"],
                     topk=cfg.get("topk", 0.0), ternary=(cfg.get("mlp_precision", "ternary") == "ternary"))
    elif kind == "moe":
        if cfg.get("arm") == "dense-big":
            # the phase59 trainer wraps the dense arm in DenseExpert (keys blocks.*.mlp.mlp.*)
            for b in model.blocks:
                if getattr(b, "use_mlp", False):
                    b.mlp = DenseExpert(D, cfg["total_hid"])
        else:
            for b in model.blocks:
                if getattr(b, "use_mlp", False):
                    b.mlp = MoEMLP(D, cfg["hid_e"], cfg["E"], cfg["topk"],
                                   load_w=cfg.get("load_balance_w", 0.01),
                                   lam_coh=cfg.get("lam_coh", 0.0), dev_type="cpu")
    else:
        raise ValueError(kind)
    return model


def forward_logits(model, kind, cfg, x):
    if kind == "moe":                       # MoEMLP and DenseExpert both return (out, aux)
        logits, _ = moe_forward(model, x, use_ckpt=False)
        return logits
    return model(x)


def eval_bpb(model, kind, cfg, val, el_np, V, batch=24):
    model = model.float().cpu().eval()
    wins = np.stack([val[p:p + WIN + 1] for p in range(0, NWIN * WIN, WIN)])   # (NWIN, 513)
    bits = 0.0
    nbytes = 0
    ntok = 0
    with torch.no_grad():
        for i in range(0, NWIN, batch):
            blk = wins[i:i + batch]
            x = torch.from_numpy(blk[:, :-1])
            y = torch.from_numpy(blk[:, 1:])
            logits = forward_logits(model, kind, cfg, x)
            ce = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="sum")
            bits += float(ce) / math.log(2)
            nbytes += int(el_np[y.reshape(-1).numpy()].sum())
            ntok += y.numel()
    return bits / nbytes, ntok, nbytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="subset of checkpoint labels")
    ap.add_argument("--batch", type=int, default=24)
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() - 2))

    V, exp_len, _ = load_meta(META)
    el_np = np.array(exp_len, dtype=np.int64)
    ids = np.fromfile(IDS, dtype=np.uint16).astype(np.int64)
    n = len(ids)
    ntr = int(n * 0.9)
    val = ids[ntr:]
    ids_md5 = md5_file(IDS)
    slice_desc = dict(ids_file="results/phase55/ids.u16", ids_md5=ids_md5, n_ids=int(n),
                      split="train=[0,int(n*0.9)) val=[int(n*0.9),n)", val_offset=int(ntr),
                      window=WIN, n_windows=NWIN, eval_tokens=NWIN * WIN,
                      harness="fp32 CPU, no autocast, torch.no_grad; CE bits / target byte-lengths")
    print(f"canonical slice: ids md5={ids_md5} n={n} ntr={ntr} | {NWIN} windows x {WIN} = {NWIN*WIN} tok")

    rows = []
    for label, rel, kind, note in CKPTS:
        if a.only and label not in a.only:
            continue
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print(f"  [skip] {label}: {rel} not found")
            continue
        sd = torch.load(path, map_location="cpu")
        cfg = sd.get("cfg", {})
        model = build_model(kind, V, cfg)
        missing, unexpected = model.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
        if missing or unexpected:
            print(f"  [WARN] {label}: missing={list(missing)} unexpected={list(unexpected)}")
        t0 = time.time()
        bpb, ntok, nbytes = eval_bpb(model, kind, cfg, val, el_np, V, batch=a.batch)
        dt = time.time() - t0
        train_bpb = cfg.get("bpb", None)
        print(f"  {label:18s} canonical BPB = {bpb:.6f}  (training-harness bpb: "
              f"{train_bpb if train_bpb is not None else 'n/a'})  [{ntok} tok, {dt:.0f}s]")
        rows.append(dict(label=label, path=rel, kind=kind, note=note,
                         canonical_bpb=round(bpb, 6),
                         training_harness_bpb=(round(float(train_bpb), 6) if train_bpb is not None else None),
                         eval_tokens=ntok, eval_bytes=nbytes,
                         ckpt_md5=md5_file(path)))
        del model, sd

    os.makedirs(OUTDIR, exist_ok=True)
    outp = os.path.join(OUTDIR, "canonical_bpb.json")
    if a.only and os.path.exists(outp):     # partial run: merge into the existing table by label
        with open(outp) as f:
            prev = json.load(f)
        merged = {r["label"]: r for r in prev.get("results", [])}
        for r in rows:
            merged[r["label"]] = r
        order = [c[0] for c in CKPTS]
        rows = sorted(merged.values(), key=lambda r: order.index(r["label"]) if r["label"] in order else 99)
    out = dict(slice=slice_desc, results=rows)
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {outp}")
    print("\n| checkpoint | canonical BPB (fp32 CPU) | training-harness BPB | note |")
    print("|---|---|---|---|")
    for r in rows:
        th = f"{r['training_harness_bpb']:.4f}" if r["training_harness_bpb"] is not None else "n/a"
        print(f"| `{r['label']}` | **{r['canonical_bpb']:.4f}** | {th} | {r['note']} |")
    print("\nSTOP. Canonical table above. Gates are still read on their original harnesses (see header).")


if __name__ == "__main__":
    main()
