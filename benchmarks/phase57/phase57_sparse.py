#!/usr/bin/env python3
# Phase 57 / Probe-2 (Finding-2) - SPARSITA DI ATTIVAZIONE: does a from-scratch sparse-activation MLP reach
#   high activation sparsity (target 80-90%, TurboSparse/Q-Sparse style) WITHOUT losing quality, so that per
#   token only a small fraction of MLP rows must be streamed? Composes with probe-1 ternary on an INDEPENDENT axis
#   (ternary cuts bytes/weight; sparsity cuts WHICH weights you touch). magic: "SPRS" 0x53505253
#
#   ONE VARIABLE = the MLP activation. Everything else = the validated probe-1 5M Arch-A (MLP linears TERNARY via
#   BitLinear158; SSM/SWA/head/emb/norms fp32). Arms:
#     baseline : --gated --act silu     = SwiGLU gated MLP (dense activation)  [the matched control]
#     dReLU    : --gated --act drelu    = dual-ReLU gated MLP  down(relu(gate) * relu(up))   [headline]
#     plain    : (no --gated) --act relu2|relu = plain MLP up->act->down (lighter; skips DOWN only)
#     Q-Sparse : --topk F  forces sparsity F by zeroing all but the top-(1-F) magnitudes per token (escalation)
#
#   Honest scope (like probe-1): validates the MECHANISM (high sparsity + quality held + large SKIPPABLE fraction),
#   NOT a realized bandwidth win (sandbox model is cache-resident; the fused sparse-ternary kernel is built later in
#   the C engine). No kernel here. Speed follows from skippable-fraction x the already-validated LUT.
#
#   Three numbers reported: (1) per-block activation sparsity; (2) val BPB vs ternary baseline 0.8382 (& fp32 0.8104);
#   (3) SKIPPABLE MLP-weight fraction computed per the REAL structure:
#       - plain (up->act->down): post-act zeros skip only DOWN columns  -> skippable = s_h * down_share.
#       - gated (gate,up,down):  gate zeros skip UP rows AND DOWN cols   -> skippable ~ s_gate * (up+down)_share.
#       (skipping UP on a plain MLP needs a PREDICTOR = Finding-7, future probe, not here.)
#
# Smoke (CPU): python benchmarks/phase57/phase57_sparse.py --smoke --gated --act drelu
# A/B (short): python benchmarks/phase57/phase57_sparse.py --gated --act drelu --steps 4000 --seq 512 --batch 16 --fp16 --compile --save results/phase57/sp_drelu.pt
#         ...  python benchmarks/phase57/phase57_sparse.py --gated --act silu  --steps 4000 --seq 512 --batch 16 --fp16 --compile --save results/phase57/sp_silu.pt
import argparse, math, os, sys, time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase55"))
sys.path.insert(0, HERE)
from phase55_ssm import ArchA, load_meta, IDS, META            # frozen baseline arch + data
from phase57_ternary import BitLinear158                        # probe-1 ternary linear (STE, per-row absmean)

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT  = os.path.join(ROOT, "results", "phase57")
try: os.makedirs(OUT, exist_ok=True)      # read-only mount: import for symbols must still succeed
except OSError: pass

# ---------------- sparse-activation MLP ----------------
class SparseMLP(nn.Module):
    def __init__(s, D, hid, act="drelu", gated=True, topk=0.0, ternary=True):
        super().__init__()
        Lin = (lambda i, o: BitLinear158(i, o)) if ternary else (lambda i, o: nn.Linear(i, o, bias=False))
        s.gated, s.act, s.topk = gated, act, topk
        if gated: s.gate = Lin(D, hid); s.up = Lin(D, hid)
        else:     s.up = Lin(D, hid)
        s.down = Lin(hid, D)
        s._cap = False; s._s_h = 0.0; s._s_g = 0.0     # captured activation sparsity (hidden, gate)
    def _act1(s, t):
        if s.act == "relu":  return F.relu(t)
        if s.act == "relu2": return F.relu(t) ** 2
        if s.act == "silu":  return F.silu(t)
        raise ValueError(f"plain act must be relu/relu2/silu, got {s.act}")
    def forward(s, x):
        if s.gated:
            g = s.gate(x); u = s.up(x)
            if   s.act == "drelu": h = F.relu(g) * F.relu(u)
            elif s.act == "silu":  h = F.silu(g) * u            # SwiGLU
            else:                  h = s._act1(g) * u
            gate_pos = (F.relu(g) > 0) if s.act != "silu" else (F.silu(g) != 0)
        else:
            h = s._act1(s.up(x)); gate_pos = (h != 0)
        if s.topk and s.topk > 0.0:                              # Q-Sparse: keep top-(1-topk) magnitudes per token
            keep = max(1, int(round(h.shape[-1] * (1.0 - s.topk))))
            thr = h.abs().kthvalue(h.shape[-1] - keep + 1, dim=-1, keepdim=True).values
            h = torch.where(h.abs() >= thr, h, torch.zeros_like(h))
        if s._cap:
            with torch.no_grad():
                s._s_h = (h == 0).float().mean().item()
                s._s_g = (~gate_pos).float().mean().item() if s.gated else s._s_h
        return s.down(h)
    def shares(s):                                               # weight-share of each matrix within the MLP
        return (3 if s.gated else 2)

def sparsify_mlp(model, D, hid, act, gated, topk, ternary):
    n = 0
    for blk in model.blocks:
        if getattr(blk, "use_mlp", False):
            blk.mlp = SparseMLP(D, hid, act, gated, topk, ternary); n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--act", choices=["silu", "relu", "relu2", "drelu"], default="drelu")
    ap.add_argument("--gated", action="store_true", help="gated MLP (gate/up/down); needed for drelu & the up+down skip")
    ap.add_argument("--topk", type=float, default=0.0, help="Q-Sparse: force this sparsity fraction via per-token top-K (0=off)")
    ap.add_argument("--mlp-precision", choices=["fp32", "ternary"], default="ternary")
    ap.add_argument("--mlp-mult", type=int, default=4, help="MLP hidden multiplier (hid=mult*D); same for both arms = one variable")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true", help="fp16 autocast (T4/Turing: use this, NOT bf16)")
    ap.add_argument("--eval-tok", type=int, default=200000)
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--time-budget-min", type=float, default=0.0)
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--device", type=str, default="auto")
    a = ap.parse_args()
    if a.smoke: a.steps = 60; a.seq = 64; a.batch = 4; a.eval_tok = 20000
    if a.act == "drelu" and not a.gated: print("  [note] --act drelu requires --gated; enabling gated."); a.gated = True
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dev_type = "cuda" if dev.startswith("cuda") else "cpu"
    amp_dtype = torch.float16 if a.fp16 else (torch.bfloat16 if a.bf16 else None)
    torch.manual_seed(0); np.random.seed(0)
    if dev_type == "cuda": torch.cuda.manual_seed_all(0)

    V, exp_len, id2bytes = load_meta(META)
    ids = np.fromfile(IDS, dtype=np.uint16).astype(np.int64)
    n = len(ids); ntr = int(n * 0.9); train = ids[:ntr]; val = ids[ntr:]
    el_t = torch.tensor(exp_len)

    D = 256; AC = dict(D=D, N=96, H=8, L=6, swa_layer=5, use_mlp=True, mlp_mult=a.mlp_mult, dt_rank=16)
    hid = a.mlp_mult * D
    model = ArchA(V, **AC).to(dev); model.use_ckpt = not a.no_ckpt
    nsw = sparsify_mlp(model, D, hid, a.act, a.gated, a.topk, ternary=(a.mlp_precision == "ternary"))
    model = model.to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    nmlp = sum(p.numel() for blk in model.blocks if getattr(blk, "use_mlp", False) for p in blk.mlp.parameters())
    struct = "gated(gate,up,down)" if a.gated else "plain(up,down)"
    print(f"Phase57 SPARSE | act={a.act} struct={struct} topk={a.topk} mlp_precision={a.mlp_precision} | sparsified {nsw} MLP blocks -> SparseMLP")
    print(f"  Arch-A 5M: params={nparam/1e6:.3f}M (MLP={nmlp/1e6:.3f}M={100*nmlp/nparam:.0f}%, hid={hid}) | dev={dev} amp={amp_dtype}")
    print(f"  config: steps={a.steps} seq={a.seq} batch={a.batch} lr={a.lr} | baselines: ternary 0.8382, fp32 0.8104")

    mlp_blocks = [blk.mlp for blk in model.blocks if getattr(blk, "use_mlp", False)]

    def autocast_ctx():
        return torch.autocast(dev_type, dtype=amp_dtype) if amp_dtype is not None else torch.autocast(dev_type, enabled=False)

    def val_bpb(cap):
        model.eval(); bits = 0.0; nb = 0
        with torch.no_grad():
            W = a.seq; lim = min(cap, len(val) - 1); pos = 0
            while pos + W + 1 <= lim:
                x = torch.from_numpy(val[pos:pos+W][None, :]).to(dev); y = torch.from_numpy(val[pos+1:pos+1+W][None, :]).to(dev)
                ce = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1), reduction="sum")
                bits += ce.item() / math.log(2); nb += int(el_t[y.reshape(-1).cpu()].sum().item()); pos += W
        model.train(); return bits / max(nb, 1)

    def eval_sparsity(nbatch=8):
        for m in mlp_blocks: m._cap = True
        sh = [[] for _ in mlp_blocks]; sg = [[] for _ in mlp_blocks]
        model.eval()
        with torch.no_grad():
            for b in range(nbatch):
                pos = b * a.seq
                if pos + a.seq + 1 > len(val): break
                x = torch.from_numpy(val[pos:pos+a.seq][None, :]).to(dev)
                model(x)
                for i, m in enumerate(mlp_blocks): sh[i].append(m._s_h); sg[i].append(m._s_g)
        for m in mlp_blocks: m._cap = False
        model.train()
        sh = [float(np.mean(v)) for v in sh]; sg = [float(np.mean(v)) for v in sg]
        return sh, sg

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    base = model
    if a.compile:
        try:
            cm = torch.compile(model); warm = torch.zeros(a.batch, a.seq, dtype=torch.long, device=dev)
            with torch.no_grad(), autocast_ctx(): cm(warm)
            model = cm; print("  torch.compile enabled")
        except Exception as e:
            print(f"  torch.compile failed -> eager ({type(e).__name__}: {str(e)[:100]})")
    def get_batch(src):
        ix = np.random.randint(0, len(src) - a.seq - 1, size=a.batch)
        return (torch.from_numpy(np.stack([src[i:i+a.seq] for i in ix])).to(dev),
                torch.from_numpy(np.stack([src[i+1:i+1+a.seq] for i in ix])).to(dev))

    print("== training (CE only) ==")
    model.train(); t0 = time.time(); stopped = a.steps
    for step in range(a.steps):
        ts = time.time(); opt.zero_grad(); lsum = 0.0
        for _ in range(a.accum):
            x, y = get_batch(train)
            with autocast_ctx():
                loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)) / a.accum
            loss.backward(); lsum += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step == 0 or (step + 1) % max(1, a.steps // 20) == 0:
            if dev_type == "cuda": torch.cuda.synchronize()
            ms = (time.time() - ts) * 1000; vmsg = ""
            if (step + 1) % max(1, a.steps // 10) == 0 or step == 0:
                sh, _ = eval_sparsity(4); vmsg = f"  | val BPB={val_bpb(min(a.eval_tok,16000)):.4f}  sparsity~{np.mean(sh)*100:.0f}%"
            print(f"  step {step+1:5d}/{a.steps}  loss={lsum:.4f}  gnorm={gn:.2f}  bits/tok={lsum/math.log(2):.4f}  ({ms:.0f} ms){vmsg}")
        if a.time_budget_min > 0 and (time.time() - t0) / 60.0 > a.time_budget_min:
            print(f"  [time-budget hit @ step {step+1}]"); stopped = step + 1; break

    # ---- final readout: the three numbers ----
    bpb = val_bpb(a.eval_tok)
    sh, sg = eval_sparsity(16)
    print("== RESULTS ==")
    print(f"  (2) QUALITY: val BPB={bpb:.4f}  | vs ternary-baseline 0.8382 -> {bpb-0.8382:+.4f} | vs fp32 0.8104 -> {bpb-0.8104:+.4f}")
    print(f"  (1) ACTIVATION SPARSITY per-block (hidden zeros/token): " + " ".join(f"{v*100:4.0f}%" for v in sh) + f"  | mean {np.mean(sh)*100:.1f}%")
    if a.gated:
        print(f"      gate sparsity per-block (drives UP+DOWN skip):     " + " ".join(f"{v*100:4.0f}%" for v in sg) + f"  | mean {np.mean(sg)*100:.1f}%")
    # ---- (3) skippable MLP-weight fraction per REAL structure ----
    sh_m, sg_m = float(np.mean(sh)), float(np.mean(sg))
    if a.gated:
        skippable = sg_m * (2.0 / 3.0)          # gate zeros skip UP and DOWN (each 1/3); GATE itself computed (predictor=Finding-7)
        active = 1.0 - skippable
        detail = f"gate skips up+down: {sg_m*100:.0f}% x 2/3"
    else:
        skippable = sh_m * 0.5                   # post-act zeros skip DOWN only (down=1/2 of plain MLP)
        active = 1.0 - skippable
        detail = f"post-act skips down: {sh_m*100:.0f}% x 1/2"
    print(f"  (3) SKIPPABLE MLP-weight fraction/token = {skippable*100:.1f}%  ({detail}) -> active MLP = {active*100:.1f}% -> shrink {1.0/active:.2f}x")
    print(f"      (NOTE: skipping UP on a plain MLP, or GATE on a gated MLP, needs a predictor = Finding-7, future probe.)")

    if a.save:
        os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
        torch.save({"model": base.state_dict(),
                    "cfg": dict(V=V, **AC, win=128, expand=2, conv=4, seq=a.seq, steps=stopped, bpb=bpb,
                                act=a.act, gated=a.gated, topk=a.topk, mlp_precision=a.mlp_precision,
                                sparsity_mean=sh_m, gate_sparsity_mean=sg_m, skippable=skippable)}, a.save)
        print(f"  saved -> {a.save}")
    print("STOP. three numbers above (sparsity / BPB / skippable). No kernel, no commit.")

if __name__ == "__main__":
    main()
