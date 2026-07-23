#!/usr/bin/env python3
# Phase 59 / Probe-4 (MoE) - MoE-ify the Phase-58 gated-dReLU-ternary MLP. Does MoE buy capacity at iso-active-cost
#   WITHOUT costing quality, and does the routing CONCENTRATE (working-set sub-linear) + PREDICT (router recall)?
#   magic: "P59M" 0x50353_4D
#
#   ONE VARIABLE = the MLP structure (dense vs MoE, and granularity). Everything else = the validated foundation:
#   each expert is a mini gated-dReLU-ternary MLP (phase57 SparseMLP, the Phase-58 recipe, IDENTICAL); SSM/SWA/head/
#   emb/norms fp32; router fp32 (small, decision-critical, always-resident, like SSM/head); experts ternary.
#   Load-balancing aux-loss (Switch-style) is APPARATUS (not the variable) - without it the router collapses.
#
#   OOM lesson from 58.B (inherited): the aux-loss must NOT be stashed on a module and read outside the block (that
#   pins the whole forward graph and ANNULS gradient-checkpointing -> OOM). Here moe_forward checkpoints ONLY the SSM
#   mix (the dA memory hog) and the MoE MLP RETURNS its aux cleanly (out, aux) -> checkpointing survives -> bigger batch
#   fits. Experts use a BLOCKED layout (gate/up = one big BitLinear; per-expert down via batched einsum; routing = a
#   structured hidden-mask) so there is NO Python expert-loop -> GPU-optimal, ~dense-big speed (a per-expert loop is
#   launch-bound on GPU: 8.8s vs 6.4s). --compile is DEAD (Inductor/Triton chokes on the unrolled seq-512 SSM scan). bf16 on the 3060.
#
#   Arms (active-hidden matched to 1024; total 4096 where MoE):
#     dense-big   : dense hidden 4096                         (B; upper bound = what 4x active buys)
#     moe-gran    : E=32 x hid_e=128, top-8  (active 1024)    (C; candidate - Phase 58 favors fine)
#     moe-coarse  : E=8  x hid_e=512, top-2  (active 1024)    (D; granularity A/B, secondary)
#   Arm A (dense-small 1024) already EXISTS = results/phase57/sp58_base.pt (BPB 0.8799). Capture=(A-C)/(A-B), head=B-A.
#
#   Five measures (cheap, CPU, like 58.A; expert-granularity reuse of the phase58_predict ridge probe):
#     1 quality val BPB ; 2 balance (dead experts, max/mean load) ; 3 working-set |union experts|/E over N positions
#     (sub-linear -> L3-fit plausible) ; 4 routing predictability in-place (ridge x_t->expert-set, held-out) + ahead
#     (x_{t-1}) ; 5 expert persistence overlap(experts[t],experts[t-1]).
#   Honest scope: sandbox 5M is cache-resident -> we measure the PROPERTY (pays quality? routing concentrates+predicts?)
#   NOT realized L3 bandwidth (= the C engine). Working-set/balance/predictability are the L3-fit proxies.
#
# Smoke (CPU): python benchmarks/phase57/phase59_moe.py --smoke --arm moe-gran
# Train (3060): python benchmarks/phase57/phase59_moe.py --arm moe-gran --steps 4000 --seq 512 --batch 4 --accum 4 --bf16 --save results/phase57/moe_gran.pt
import argparse, math, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint as ckpt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase55"))
sys.path.insert(0, HERE)
from phase55_ssm import ArchA, load_meta, IDS, META
from phase57_ternary import BitLinear158                         # noqa: F401
from phase57_sparse import SparseMLP
from phase58_predict import ridge_fit, ridge_apply, recall_at_k  # reuse the held-out ridge probe

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "phase57")
try: os.makedirs(OUT, exist_ok=True)      # read-only mount: import for symbols must still succeed
except OSError: pass

ARMS = {  # arm -> (is_moe, E, hid_e, k, dense_hid)
    "dense-big":  (False, 0,  0,   0, 4096),
    "moe-gran":   (True, 32, 128,  8, 0),
    "moe-coarse": (True,  8, 512,  2, 0),
}


class DenseExpert(nn.Module):
    """Arm B: a single dense gated-dReLU-ternary MLP. Returns (out, aux=0) for the moe_forward interface."""
    def __init__(s, D, hid):
        super().__init__(); s.mlp = SparseMLP(D, hid, act="drelu", gated=True, topk=0.0, ternary=True)
    def forward(s, x):
        return s.mlp(x), x.new_zeros(())


class MoEMLP(nn.Module):
    """Token-choice top-k MoE, BLOCKED layout (no Python expert-loop -> GPU-optimal, ~dense-big speed). The E experts
    are contiguous blocks of one big gated-dReLU-ternary MLP: gate/up = a single BitLinear158(D, E*hid_e) whose
    per-row ternary scale (over D) is INDEPENDENT per row -> byte-identical to E separate expert gates/ups. Routing =
    a structured mask on the hidden (block e scaled by expert e's top-k weight, 0 if not selected). down keeps a
    PER-EXPERT ternary scale via a batched (E,D,hid_e) param + one einsum -> identical to separate expert downs.
    Compute-all (= dense-big FLOPs; the true active<total sparse-compute is the C engine). (out, aux) - aux NEVER
    stashed (58.B OOM lesson). Self-tested vs a per-expert reference (_ref) in --smoke."""
    def __init__(s, D, hid_e, E, k, load_w, lam_coh, dev_type):
        super().__init__()
        s.D, s.E, s.k, s.hid_e, s.H = D, E, k, hid_e, E * hid_e
        s.load_w, s.lam_coh, s.dev_type = load_w, lam_coh, dev_type
        s.router = nn.Linear(D, E)                                # fp32, always-resident (decision-critical)
        s.gate = BitLinear158(D, s.H)                            # all experts' gate rows (per-row ternary over D == separate)
        s.up = BitLinear158(D, s.H)
        s.Wd = nn.Parameter(torch.empty(E, D, hid_e)); nn.init.kaiming_uniform_(s.Wd, a=math.sqrt(5))  # per-expert down
        s._cap = False; s._topi = None; s._xin = None; s._last_load = 0.0; s._last_coh = 0.0
    @staticmethod
    def _tern(W):                                                # per-row (last dim) absmean ternary STE (== BitLinear158)
        scale = W.abs().mean(-1, keepdim=True).clamp_min(1e-5)
        wq = (W / scale).round().clamp(-1, 1)
        return W + (wq * scale - W).detach()
    def _route_hidden(s, x):                                     # shared gate/up/dReLU + routing mask -> (masked hidden, routing)
        B, T, D = x.shape; xf = x.reshape(-1, D); N = xf.shape[0]
        with torch.autocast(s.dev_type, enabled=False):
            probs = torch.softmax(s.router(xf.float()), dim=-1)  # (N,E)
        topv, topi = probs.topk(s.k, dim=-1)
        topw = topv / topv.sum(dim=-1, keepdim=True)
        gatefull = torch.zeros_like(probs).scatter(-1, topi, topw)                 # (N,E) top-k weights, 0 else
        h = F.relu(s.gate(xf)) * F.relu(s.up(xf))                # (N,H) dReLU over all experts
        h = h * gatefull.repeat_interleave(s.hid_e, dim=1).to(h.dtype)             # structured mask: block e *= weight_e
        return xf, probs, topi, topw, h.reshape(N, s.E, s.hid_e), B, T
    def forward(s, x):
        xf, probs, topi, topw, hE, B, T = s._route_hidden(x)
        out = torch.einsum('neh,edh->nd', hE, s._tern(s.Wd).to(hE.dtype))          # per-expert down, batched (no loop)
        sel = torch.zeros_like(probs).scatter(-1, topi, torch.ones_like(topw))     # Switch aux: E*sum f_e*P_e / k
        load = s.E * (sel.mean(0) * probs.mean(0)).sum() / s.k
        aux = s.load_w * load
        coh = xf.new_zeros(())
        if s.lam_coh > 0 and T > 1:                              # optional routing-persistence pressure (foundation +coh)
            pr = probs.reshape(B, T, s.E); coh = (pr[:, 1:] - pr[:, :-1]).abs().mean(); aux = aux + s.lam_coh * coh
        s._last_load = float(load.detach()); s._last_coh = float(coh.detach())
        if s._cap:                                               # measurement only (no_grad) - safe to store, no graph
            s._topi = topi.reshape(B, T, s.k).detach().cpu().numpy()
            s._xin = x.detach().float().cpu().numpy()
        return out.reshape(B, T, s.D), aux
    @torch.no_grad()
    def _ref(s, x):                                             # per-expert loop over the SAME params (equivalence test)
        xf, probs, topi, topw, hE, B, T = s._route_hidden(x)
        Wd = s._tern(s.Wd)
        out = torch.zeros(xf.shape[0], s.D, device=xf.device, dtype=torch.float32)
        for e in range(s.E):
            out = out + hE[:, e, :].float() @ Wd[e].float().t()
        return out.reshape(B, T, s.D)


def build_model(V, AC, arm, load_w, lam_coh, dev, dev_type):
    is_moe, E, hid_e, k, dense_hid = ARMS[arm]
    model = ArchA(V, **AC).to(dev)
    n = 0
    for b in model.blocks:
        if getattr(b, "use_mlp", False):
            b.mlp = MoEMLP(AC["D"], hid_e, E, k, load_w, lam_coh, dev_type) if is_moe else DenseExpert(AC["D"], dense_hid)
            n += 1
    return model.to(dev), n


def moe_forward(model, idx, use_ckpt):
    """Manual block loop so the MoE aux returns CLEANLY (no stash). Checkpoints only the SSM mix (the dA memory hog),
    NOT the MoE MLP -> checkpointing is not defeated by the aux. Returns (logits, aux_total)."""
    x = model.emb(idx)
    aux_total = x.new_zeros(())
    for b in model.blocks:
        normed = b.norm(x)
        mixed = ckpt.checkpoint(b.mix, normed, use_reentrant=False) if (use_ckpt and model.training) else b.mix(normed)
        x = x + mixed
        if getattr(b, "use_mlp", False):
            out, aux = b.mlp(b.norm2(x)); x = x + out; aux_total = aux_total + aux
    return model.head(model.norm_f(x)), aux_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arm", choices=list(ARMS.keys()), default="moe-gran")
    ap.add_argument("--load-balance-w", type=float, default=0.01, help="Switch aux weight (apparatus, not the variable)")
    ap.add_argument("--lam-coh", type=float, default=0.0, help="routing-persistence coherence (58.B was unpromoted; off by default)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4, help="eff batch = batch*accum (MoE is heavier; drop batch first)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no-ckpt", action="store_true", help="disable SSM-mix checkpointing (rarely needed; ckpt now survives MoE)")
    ap.add_argument("--eval-tok", type=int, default=200000)
    ap.add_argument("--measure-batches", type=int, default=24)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--measure-only", action="store_true", help="load --ckpt and only run the 5 measures (no training)")
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--device", type=str, default="auto")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.seq, a.batch, a.accum, a.eval_tok, a.measure_batches = 60, 64, 4, 1, 20000, 8
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dev_type = "cuda" if dev.startswith("cuda") else "cpu"
    amp_dtype = torch.float16 if a.fp16 else (torch.bfloat16 if a.bf16 else None)
    torch.manual_seed(0); np.random.seed(0)
    if dev_type == "cuda": torch.cuda.manual_seed_all(0)

    V, exp_len, id2bytes = load_meta(META)
    ids = np.fromfile(IDS, dtype=np.uint16).astype(np.int64)
    n = len(ids); ntr = int(n * 0.9); train = ids[:ntr]; val = ids[ntr:]
    el_t = torch.tensor(exp_len)

    D = 256; AC = dict(D=D, N=96, H=8, L=6, swa_layer=5, use_mlp=True, mlp_mult=4, dt_rank=16)
    if a.measure_only and a.ckpt:
        sd = torch.load(a.ckpt, map_location=dev); cfg = sd.get("cfg", {})
        a.arm = cfg.get("arm", a.arm); a.load_balance_w = cfg.get("load_balance_w", a.load_balance_w); a.lam_coh = cfg.get("lam_coh", a.lam_coh)
    model, nsw = build_model(V, AC, a.arm, a.load_balance_w, a.lam_coh, dev, dev_type)
    model.use_ckpt = not a.no_ckpt
    use_ckpt = model.use_ckpt
    if a.measure_only and a.ckpt:
        model.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
    is_moe, E, hid_e, k, dense_hid = ARMS[a.arm]
    act_hid = (k * hid_e) if is_moe else dense_hid
    tot_hid = (E * hid_e) if is_moe else dense_hid
    npar = sum(p.numel() for p in model.parameters())
    print(f"Phase59 MoE | arm={a.arm} | active_hid={act_hid} total_hid={tot_hid} (E={E} hid_e={hid_e} top-{k}) | dev={dev} amp={amp_dtype}")
    print(f"  params={npar/1e6:.3f}M | load_balance_w={a.load_balance_w} lam_coh={a.lam_coh} | sparsified {nsw} MLP blocks | ckpt(mix)={use_ckpt}")
    print(f"  baselines: A dense-1024 (sp58_base) BPB 0.8799 | ternary 0.8382 | probe-2 dReLU 0.8813")

    moe_layers = [b.mlp for b in model.blocks if isinstance(b.mlp, MoEMLP)]

    if a.smoke and moe_layers:                                   # PROVE sparse-dispatch == masked-all (no behavior change)
        m0 = moe_layers[0]
        with torch.no_grad():
            xt = torch.randn(2, 8, D, device=dev)
            o_blk, _ = m0(xt); o_ref = m0._ref(xt)
            dmax = (o_blk - o_ref).abs().max().item()
        print(f"  [self-test] blocked-einsum vs per-expert-loop: max_abs_diff={dmax:.2e} (~0 -> identical, no Python loop -> dense-big speed)")

    def autocast_ctx():
        return torch.autocast(dev_type, dtype=amp_dtype) if amp_dtype is not None else torch.autocast(dev_type, enabled=False)

    def val_bpb(cap):
        model.eval(); bits = 0.0; nb = 0
        with torch.no_grad():
            W = a.seq; lim = min(cap, len(val) - 1); pos = 0
            while pos + W + 1 <= lim:
                x = torch.from_numpy(val[pos:pos+W][None, :]).to(dev); y = torch.from_numpy(val[pos+1:pos+1+W][None, :]).to(dev)
                with autocast_ctx(): logits, _ = moe_forward(model, x, use_ckpt=False)
                ce = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="sum")
                bits += ce.item() / math.log(2); nb += int(el_t[y.reshape(-1).cpu()].sum().item()); pos += W
        model.train(); return bits / max(nb, 1)

    def measure(nbatch, fit_frac=0.6, Ns=(8, 16, 32)):
        if not moe_layers:
            print("  (dense arm: no routing measures - BPB only)"); return
        for m in moe_layers: m._cap = True
        topis = [[] for _ in moe_layers]; xins = [[] for _ in moe_layers]
        model.eval()
        with torch.no_grad():
            for b in range(nbatch):
                pos = b * a.batch * a.seq
                if pos + a.batch * a.seq + 1 > len(val): break
                blk = val[pos:pos + a.batch * a.seq].reshape(a.batch, a.seq)
                with autocast_ctx(): moe_forward(model, torch.from_numpy(blk).to(dev), use_ckpt=False)
                for i, m in enumerate(moe_layers): topis[i].append(m._topi); xins[i].append(m._xin)
        for m in moe_layers: m._cap = False
        model.train()
        hdr = f"  {'layer':>5} {'E':>4} {'k':>3} {'dead':>5} {'max/mean':>9} {'persist%':>9} " + " ".join(f"ws@{N:<2}" for N in Ns) + f" {'rec_in%':>8} {'rec_ahead%':>10}"
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for i, m in enumerate(moe_layers):
            TOPI = np.concatenate(topis[i], 0); XIN = np.concatenate(xins[i], 0)   # (rows,T,k) , (rows,T,D)
            rows, T, kk = TOPI.shape; Ei = m.E
            SEL = np.zeros((rows, T, Ei), bool)
            ri = np.arange(rows)[:, None, None]; ti = np.arange(T)[None, :, None]
            SEL[ri, ti, TOPI] = True
            counts = SEL.reshape(-1, Ei).sum(0)
            dead = int((counts == 0).sum()); maxmean = counts.max() / max(counts.mean(), 1e-9)
            St, Stm = SEL[:, 1:, :], SEL[:, :-1, :]
            persist = float(((St & Stm).sum(-1) / np.maximum(St.sum(-1), 1)).mean())
            ws = []
            for N in Ns:
                nw = T // N
                if nw < 1: ws.append(float("nan")); continue
                u = SEL[:, :nw*N, :].reshape(rows, nw, N, Ei).any(2)            # (rows,nw,Ei)
                ws.append(float((u.sum(-1) / Ei).mean()))
            nrf = max(1, int(round(rows * fit_frac)))
            Xf = XIN[:nrf].reshape(-1, D).astype(np.float64); Yf = SEL[:nrf].reshape(-1, Ei).astype(np.float64)
            Xe = XIN[nrf:].reshape(-1, D).astype(np.float64); Ye = SEL[nrf:].reshape(-1, Ei)
            if Xe.shape[0] == 0: Xe, Ye = Xf, SEL[:nrf].reshape(-1, Ei)
            Wp = ridge_fit(Xf, Yf, a.ridge); rec_in = recall_at_k(ridge_apply(Xe, Wp), Ye)
            Xf2 = XIN[:nrf, :-1].reshape(-1, D).astype(np.float64); Yf2 = SEL[:nrf, 1:].reshape(-1, Ei).astype(np.float64)
            Xe2 = XIN[nrf:, :-1].reshape(-1, D).astype(np.float64); Ye2 = SEL[nrf:, 1:].reshape(-1, Ei)
            if Xe2.shape[0] == 0: Xe2, Ye2 = Xf2, SEL[:nrf, 1:].reshape(-1, Ei)
            Wp2 = ridge_fit(Xf2, Yf2, a.ridge); rec_ah = recall_at_k(ridge_apply(Xe2, Wp2), Ye2)
            wss = " ".join(f"{v*100:4.0f}%" for v in ws)
            print(f"  {i:>5} {Ei:>4} {kk:>3} {dead:>5} {maxmean:>8.2f}x {persist*100:>8.1f}% {wss} {rec_in*100:>7.1f}% {rec_ah*100:>9.1f}%")
        print("  reading: dead=0 & max/mean<~3x = router OK | ws=|union experts|/E over N pos (sub-linear -> L3-fit) | rec_in = router predictable in-place")

    if a.measure_only:
        bpb = val_bpb(a.eval_tok)
        print(f"== MEASURE-ONLY ==\n  val BPB={bpb:.4f}  | vs A(0.8799) {bpb-0.8799:+.4f}")
        measure(a.measure_batches)
        print("STOP. measures above. No commit."); return

    use_scaler = (amp_dtype == torch.float16 and dev_type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)

    def get_batch(src):
        ix = np.random.randint(0, len(src) - a.seq - 1, size=a.batch)
        return (torch.from_numpy(np.stack([src[i:i+a.seq] for i in ix])).to(dev),
                torch.from_numpy(np.stack([src[i+1:i+1+a.seq] for i in ix])).to(dev))

    print("== training (CE + Switch load-balance aux) ==")
    model.train(); t0 = time.time()
    for step in range(a.steps):
        ts = time.time(); opt.zero_grad(); ce_v = aux_v = 0.0
        for _ in range(a.accum):
            x, y = get_batch(train)
            with autocast_ctx():
                logits, aux = moe_forward(model, x, use_ckpt=use_ckpt)
                ce = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            loss = (ce + aux) / a.accum
            scaler.scale(loss).backward()
            ce_v += ce.item() / a.accum; aux_v += float(aux.detach()) / a.accum
        scaler.unscale_(opt); gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if step == 0 or (step + 1) % max(1, a.steps // 20) == 0:
            if dev_type == "cuda": torch.cuda.synchronize()
            ms = (time.time() - ts) * 1000
            lmsg = ""
            if moe_layers:
                lmsg = f" load={np.mean([m._last_load for m in moe_layers]):.3f}"
                if a.lam_coh > 0: lmsg += f" coh={np.mean([m._last_coh for m in moe_layers]):.3f}"
            vmsg = f"  | val BPB={val_bpb(min(a.eval_tok,16000)):.4f}" if ((step+1) % max(1, a.steps//10) == 0 or step == 0) else ""
            print(f"  step {step+1:5d}/{a.steps}  ce={ce_v:.4f} aux={aux_v:.4f}{lmsg}  gnorm={gn:.2f}  ({ms:.0f} ms){vmsg}")

    bpb = val_bpb(a.eval_tok)
    print(f"== RESULTS ==\n  arm={a.arm}  val BPB={bpb:.4f}  | vs A dense-1024 (0.8799) -> {bpb-0.8799:+.4f}  (capture/headroom: Architect vs B)")
    measure(a.measure_batches)
    if a.save:
        os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(),
                    "cfg": dict(V=V, **AC, win=128, arm=a.arm, E=E, hid_e=hid_e, topk=k, active_hid=act_hid, total_hid=tot_hid,
                                load_balance_w=a.load_balance_w, lam_coh=a.lam_coh, bpb=bpb, mlp_precision="ternary")}, a.save)
        print(f"  saved -> {a.save}")
    print("STOP. quality + 5 measures above. No kernel, no commit.")


if __name__ == "__main__":
    main()
