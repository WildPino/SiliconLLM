#!/usr/bin/env python3
# Phase 62 -> E5.1 - LEARNED DRAFTER vs n-gram floor (DESIGN-GATE, pre-registered rule). magic: "E51M" 0x45353149
#   E5.0 showed a free n-gram drafter is weakest exactly on the product domain (code): its acceptance tracks
#   SURFACE repetition, not the model's DEEP structure (code recall 83% / BPB 1.24). Falsifiable hypothesis:
#   a LEARNED drafter converts the deep structure into acceptance. This tests it the cheapest honest way.
#
#   Arm (Architect): Medusa-1-style MTP heads on a FROZEN backbone (catA_code.pt untouched, no co-training) ->
#   isolates "does structure convert?" without confounding the backbone. Head j (j=1..4) predicts token t+j from
#   the pre-head hidden h_t (D). Head = zero-init residual block + the FROZEN tied lm_head -> each head starts as
#   the backbone's own next-token predictor and specializes. Trainable = 4 x (D*D+D) ~ 0.26M; lm_head shared/frozen.
#
#   PRE-REGISTERED DECISION RULE (fixed before any result): spec-AR stays a live E5 design-gate arm IFF
#   tokens-per-pass on code-val >= 2.0 (low edge of R-F 2-4, ~ DeepSeek 1.8-1.9). Below -> speculative-AR is out,
#   carve/block-decode becomes the primary E5 lane. Measured on the SAME cached greedy positions as E5.0.
#
# Smoke (CPU): .venv/Scripts/python.exe benchmarks/phase62/e5_1_mtp.py --smoke
# Train (3060): .venv/Scripts/python.exe benchmarks/phase62/e5_1_mtp.py --steps 4000 --seq 512 --batch 16 --bf16 --save results/phase62/mtp_code.pt
#   (requires the E5.0 greedy cache: run e5_0_acceptance.py --ckpt results/phase62/catA_code.pt first, same P/C)
import argparse, os, sys, math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from e5_0_acceptance import build_model, load_data, get_or_compute_greedy, acceptance_table, build_ngrams, draft_ngram, KMAX, P62

class MedusaHeads(nn.Module):
    """Head j: logits = lm_head( h + SiLU(res_j(h)) ). res_j zero-init -> starts == backbone next-token head."""
    def __init__(s, D, lm_head, nheads=4):
        super().__init__(); s.res = nn.ModuleList([nn.Linear(D, D) for _ in range(nheads)])
        for l in s.res: nn.init.zeros_(l.weight); nn.init.zeros_(l.bias)
        s.lm_head = lm_head; s.nheads = nheads          # lm_head is the FROZEN backbone head (shared ref)
    def forward(s, h):
        return [s.lm_head(h + F.silu(s.res[j](h))) for j in range(s.nheads)]

def capture_hidden(model):
    """Hook the pre-head hidden h_t = norm_f(x). Returns (handle, store-list)."""
    store = []
    h = model.norm_f.register_forward_hook(lambda m, i, o: store.append(o.detach()))
    return h, store

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=os.path.join(P62, "catA_code.pt"))
    ap.add_argument("--nheads", type=int, default=4); ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seq", type=int, default=512); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--bf16", action="store_true"); ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--positions", type=int, default=300); ap.add_argument("--context", type=int, default=128)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--device", default="auto")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    if a.smoke: a.steps=60; a.seq=64; a.batch=4; a.positions=20; a.context=64
    dev = a.device if a.device!="auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dev_type = "cuda" if dev.startswith("cuda") else "cpu"
    amp = torch.float16 if a.fp16 else (torch.bfloat16 if a.bf16 else None)
    torch.manual_seed(0); np.random.seed(0)

    model, cfg, V = build_model(a.backbone, dev)
    for p in model.parameters(): p.requires_grad_(False)          # FROZEN backbone
    D = cfg["D"]; dom, train, val, el, id2b = load_data(cfg)
    heads = MedusaHeads(D, model.head, a.nheads).to(dev)
    npar = sum(p.numel() for p in heads.parameters() if p.requires_grad)
    flops_head = D*D + D*V                                          # per head per token (res D*D + tied lm_head D*V)
    print(f"E5.1 MTP | backbone={os.path.basename(a.backbone)} FROZEN domain={dom} | nheads={a.nheads} | dev={dev} amp={amp}")
    print(f"  heads trainable={npar/1e3:.1f}K params | FLOPs/head/token={flops_head/1e3:.0f}K (res {D*D/1e3:.0f}K + tied lm_head {D*V/1e3:.0f}K)")
    print(f"  RULE: spec-AR alive IFF tokens/pass(code-val) >= 2.0")

    hook, store = capture_hidden(model)
    def actx(): return torch.autocast(dev_type, dtype=amp) if amp is not None else torch.autocast(dev_type, enabled=False)
    def get_batch():
        ix = np.random.randint(0, len(train)-a.seq-1, size=a.batch)
        return torch.from_numpy(np.stack([train[i:i+a.seq] for i in ix])).long().to(dev)
    opt = torch.optim.AdamW(heads.parameters(), lr=a.lr, betas=(0.9,0.95), weight_decay=0.0)
    print("== training MTP heads (head-only, backbone frozen) ==")
    for step in range(a.steps):
        x = get_batch(); store.clear()
        with torch.no_grad(): model(x)                             # fills store with h (B,T,D)
        h = store[0].float()
        with actx():
            logits = heads(h); loss = 0.0
            for j in range(a.nheads):                              # head j predicts token t+(j+1)
                off = j+1
                lg = logits[j][:, :a.seq-off, :]; tgt = x[:, off:]
                loss = loss + F.cross_entropy(lg.reshape(-1,V), tgt.reshape(-1))
            loss = loss / a.nheads
        loss.backward(); opt.step(); opt.zero_grad()
        if step==0 or (step+1)%max(1,a.steps//10)==0: print(f"  step {step+1:5d}/{a.steps} loss={loss.item():.4f}")
    hook.remove()

    # ---- measure on the SAME cached greedy positions as E5.0 ----
    pos, ctx_np, G = get_or_compute_greedy(model, val, a.backbone, a.positions, a.context, dev)
    P = len(pos)
    hook, store = capture_hidden(model); store.clear()
    with torch.no_grad():
        model(torch.from_numpy(ctx_np).long().to(dev))            # one batched backbone pass over contexts
        hlast = store[0][:, -1, :].float()                        # (P,D) hidden after last ctx token
        hl = heads(hlast)                                         # list nheads x (P,V)
        mtp = torch.stack([hl[j].argmax(-1) for j in range(a.nheads)], 1).cpu().numpy()  # (P,nheads)
    hook.remove()
    # pad mtp draft to KMAX with -1 (never matches) so acceptance_table's LCP works; K in KLIST clipped by nheads
    Dr = -np.ones((P, KMAX), dtype=np.int64); Dr[:, :a.nheads] = mtp
    # n-gram references on the SAME G for direct comparison
    tab, ufb = build_ngrams(train[:4_000_000 if not a.smoke else 300_000], [4, 8])
    drafts = {"MTP": Dr,
              "N4": np.stack([draft_ngram(tab, ufb, 4, ctx_np[i].tolist(), KMAX) for i in range(P)]),
              "N8": np.stack([draft_ngram(tab, ufb, 8, ctx_np[i].tolist(), KMAX) for i in range(P)])}
    res = acceptance_table(G, drafts, P, f"{dom} (MTP vs n-gram)")
    tpp_mtp_k4 = res[("MTP", 4)][1]
    print(f"\n  >>> MTP tokens/pass @K=4 = {tpp_mtp_k4:.3f}  vs RULE 2.0 -> {'spec-AR ALIVE' if tpp_mtp_k4>=2.0 else 'spec-AR OUT (carve becomes primary)'}")
    if a.save:
        torch.save({"heads": heads.state_dict(), "cfg": cfg, "nheads": a.nheads}, a.save); print(f"  saved -> {a.save}")
    print("STOP. E5.1 MTP acceptance (design-gate, rule pre-registered). No commit.")

if __name__ == "__main__":
    main()
