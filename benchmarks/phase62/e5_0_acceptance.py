#!/usr/bin/env python3
# Phase 62 -> E5.0 - ZERO-TRAINING ACCEPTANCE (characterization, NO gate). magic: "E50A" 0x45353041
#   Design-gate E5 asks: how many tokens can a CHEAP drafter propose before the real model diverges? Here the
#   drafter is the cheapest possible: an n-gram counted on TRAIN ids only (order N, simple backoff). The floor.
#
#   METRIC CONVENTION (Architect, R-F comparable): the number to compare with the R-F ceiling a~2-4 and DeepSeek
#   1.8 is TOKENS-PER-PASS = mean_accepted + 1 (the verify pass always commits >=1 token -- the model's token at
#   the first mismatch; greedy-vs-greedy is lossless). We report BOTH mean_a and tokens/pass.
#
#   Verify = the model's OWN greedy self-continuation under LOCKED hygiene (greedy, rep-pen 1.2 / win 128). Key
#   identity: accepted prefix == where drafter matches the model greedy self-continuation -> a = LCP(greedy, draft).
#   The greedy G (per position) is CACHED to results/phase62/greedy_<ckpt>_<P>_<C>.npz so a swapped proposer
#   (E5.1 MTP heads) is measured on the SAME positions with no repeated 700s CPU pass. This module is importable.
#
# Smoke : .venv/Scripts/python.exe benchmarks/phase62/e5_0_acceptance.py --ckpt results/phase62/catA_code.pt --smoke
# Code  : .venv/Scripts/python.exe benchmarks/phase62/e5_0_acceptance.py --ckpt results/phase62/catA_code.pt --nlist 2,3,4,5,6,8
# Prose : .venv/Scripts/python.exe benchmarks/phase62/e5_0_acceptance.py --ckpt results/phase57/sp58_base.pt --label prose-control --nlist 2,3,4,5,6,8
import argparse, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase55")); sys.path.insert(0, os.path.join(HERE, "..", "phase57"))
from phase55_ssm import ArchA, load_meta
from phase57_sparse import sparsify_mlp
ROOT = os.path.abspath(os.path.join(HERE, "..", "..")); P62 = os.path.join(ROOT, "results", "phase62")
KLIST = [2, 4, 8, 16]; KMAX = 16

def build_model(ckpt, dev):
    d = torch.load(ckpt, map_location=dev, weights_only=False); cfg = d["cfg"]; V = cfg["V"]
    AC = {k: cfg[k] for k in ("D", "N", "H", "L", "swa_layer", "use_mlp", "mlp_mult", "dt_rank")}
    m = ArchA(V, **AC); hid = cfg["mlp_mult"] * cfg["D"]
    sparsify_mlp(m, cfg["D"], hid, cfg.get("act", "drelu"), cfg.get("gated", True), cfg.get("topk", 0.0),
                 ternary=(cfg.get("mlp_precision", "ternary") == "ternary"))
    m.load_state_dict(d["model"], strict=False); m.to(dev).eval(); m.use_ckpt = False
    return m, cfg, V

def load_data(cfg):
    """Cat-A ckpt -> per-domain ids/meta; prose ckpt (no 'domain') -> TinyStories ids (90/10)."""
    if "domain" in cfg:
        dom = cfg["domain"]; V, el, id2b = load_meta(os.path.join(P62, f"{dom}.meta"))
        tr = np.fromfile(os.path.join(P62, f"{dom}_train.u16"), dtype=np.uint16).astype(np.int64)
        va = np.fromfile(os.path.join(P62, f"{dom}_val.u16"), dtype=np.uint16).astype(np.int64)
        return dom, tr, va, el, id2b
    V, el, id2b = load_meta(os.path.join(ROOT, "results", "phase55", "meta.bin"))
    ids = np.fromfile(os.path.join(ROOT, "results", "phase55", "ids.u16"), dtype=np.uint16).astype(np.int64)
    n = int(len(ids) * 0.9); return "prose", ids[:n], ids[n:], el, id2b

@torch.no_grad()
def model_greedy(model, ctx, K, dev, rep=1.2, win=128):
    """Batched greedy self-continuation (P,K) under locked hygiene. ctx: (P,C) LongTensor."""
    seq = ctx.clone(); P = seq.shape[0]; gen = torch.zeros(P, K, dtype=torch.long)
    for step in range(K):
        lg = model(seq)[:, -1, :].float()
        for r in range(P):
            recent = torch.unique(seq[r, -win:]); sv = lg[r, recent]
            lg[r, recent] = torch.where(sv > 0, sv / rep, sv * rep)
        nx = lg.argmax(-1); gen[:, step] = nx.cpu(); seq = torch.cat([seq, nx[:, None]], 1)
    return gen.numpy()

def get_or_compute_greedy(model, val, ckpt, P, C, dev, seed=0):
    """Sample P positions, return (pos, ctx_np, G) with disk cache keyed by ckpt/P/C/seed. Ensures the MTP
       proposer (E5.1) is scored on the IDENTICAL positions/greedy without recomputing the CPU pass."""
    tag = os.path.splitext(os.path.basename(ckpt))[0]
    cache = os.path.join(P62, f"greedy_{tag}_{P}_{C}_s{seed}.npz")
    if os.path.exists(cache):
        z = np.load(cache); print(f"  greedy cache HIT {os.path.basename(cache)}")
        return z["pos"], z["ctx"], z["G"]
    rng = np.random.default_rng(seed); pos = rng.integers(C, len(val) - 1, size=P)
    ctx_np = np.stack([val[p - C:p] for p in pos])
    t1 = time.time(); G = model_greedy(model, torch.from_numpy(ctx_np).long().to(dev), KMAX, dev)
    print(f"  model greedy ({P}x{KMAX}) in {time.time()-t1:.1f}s -> cached")
    np.savez(cache, pos=pos, ctx=ctx_np, G=G); return pos, ctx_np, G

def acceptance_table(G, drafts, P, label):
    """drafts: dict name -> (P,>=KMAX) int array. Prints mean_a AND tokens/pass (=mean_a+1) per (name,K)."""
    print(f"\n== ACCEPTANCE - domain={label} | tokens/pass = mean_a + 1 (R-F comparable) ==")
    print(f"  {'draft':>8} {'K':>3} {'mean_a':>7} {'tok/pass':>8} {'a/K':>5}   dist(a=0..K)")
    out = {}
    for name, D in drafts.items():
        lcp = np.array([next((m for m in range(KMAX) if G[i, m] != D[i, m]), KMAX) for i in range(P)])
        for K in KLIST:
            aK = np.minimum(lcp, K); dist = np.bincount(aK, minlength=K + 1)[:K + 1]
            dstr = " ".join(f"{x/P*100:.0f}" for x in dist)
            print(f"  {name:>8} {K:>3} {aK.mean():>7.3f} {aK.mean()+1:>8.3f} {aK.mean()/K:>5.2f}   [{dstr}]")
            out[(name, K)] = (aK.mean(), aK.mean() + 1)
    return out

# ---- n-gram drafter ----
def build_ngrams(train, orders):
    from collections import defaultdict, Counter
    cnt = {o: defaultdict(Counter) for o in orders}; uni = Counter(); T = train.tolist()
    for i in range(len(T)):
        uni[T[i]] += 1
        for o in orders:
            if i >= o - 1: cnt[o][tuple(T[i - o + 1:i])][T[i]] += 1
    tab = {o: {c: max(nx.items(), key=lambda kv: kv[1])[0] for c, nx in cnt[o].items()} for o in orders}
    return tab, max(uni.items(), key=lambda kv: kv[1])[0]

def draft_ngram(tab, ufb, N, ctx_tokens, K):
    seq = list(ctx_tokens); out = []
    for _ in range(K):
        nxt = next((tab[o][tuple(seq[-(o-1):])] for o in range(N, 1, -1) if tuple(seq[-(o-1):]) in tab.get(o, {})), ufb)
        out.append(nxt); seq.append(nxt)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--label", default="")
    ap.add_argument("--nlist", default="2,3,4"); ap.add_argument("--positions", type=int, default=300)
    ap.add_argument("--context", type=int, default=128); ap.add_argument("--train-cap", type=int, default=4_000_000)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.smoke: a.positions = 20; a.context = 64; a.train_cap = 300_000; a.nlist = "2,3,4"
    NLIST = [int(x) for x in a.nlist.split(",")]
    model, cfg, V = build_model(a.ckpt, a.device)
    dom, train, val, el, id2b = load_data(cfg); label = a.label or dom
    note = "  [ckpt OVERFIT -> interpretation limited]" if dom == "log" else ""
    print(f"E5.0 acceptance | ckpt={os.path.basename(a.ckpt)} domain={label}{note} | V={V} | dev={a.device}")
    print(f"  drafter: n-gram N={NLIST} (train-only, backoff) | verify: greedy rep1.2/win128 | ctx={a.context} pos={a.positions}")
    t0 = time.time(); tab, ufb = build_ngrams(train[:a.train_cap], NLIST); print(f"  n-gram tables built in {time.time()-t0:.1f}s")
    pos, ctx_np, G = get_or_compute_greedy(model, val, a.ckpt, a.positions, a.context, a.device)
    P = len(pos)
    drafts = {f"N{N}": np.stack([draft_ngram(tab, ufb, N, ctx_np[i].tolist(), KMAX) for i in range(P)]) for N in NLIST}
    acceptance_table(G, drafts, P, label)
    print("STOP. E5.0 acceptance characterization (no gate). No commit.")

if __name__ == "__main__":
    main()
