#!/usr/bin/env python3
# Phase 64.4 / MVE — components (C,D,E,F): the full curriculum end-to-end, at pilot scale, with the control arms.
#   magic: "MVEC" 0x4D564543
#
#   C  KD pre-train fp        : span-mapping KD at the boundary anchors + CE everywhere else
#   D  QAT ternary            : hot-swap MLP -> BitLinear158 (KD-first-then-QAT), keep training
#   E  MoE upcycle + recall   : dense MLP -> E32 x h128 top8 (seeded from the dense hidden) + recall tier w/ InfoNCE
#   F  reverse-KL             : short fine-tune with the KL direction flipped at the anchors
#
# THE ARMS (one variable each, as pre-registered in MVE_PREREG.md):
#   --arm kd  vs  --arm ce        : gate D3 -- does cross-tokenizer span KD beat plain CE on the SAME text?
#   --recall on vs --recall off   : gate D4c2 -- does the InfoNCE/recall stage destabilize the curriculum?
#
# The KD target (span mapping, D3): at an anchor, the teacher's top-K over the TEACHER vocab is projected to the
# STUDENT vocab through t2s (teacher token -> student first-token of its bytes), summing the probs of teacher tokens
# that land on the same student token, then renormalizing over the mapped mass. Elsewhere: plain CE.
#
# Chunked data path: batches are drawn from the student range covered by the RESIDENT logit chunks (2 at a time,
# advancing, delete-behind) -- i.e. the production generate->train->delete pipeline, exercised at pilot scale rather
# than simulated. Sampling is therefore chunk-local, not globally i.i.d.: a declared consequence of the design.
#
# Run (local smoke):  .venv/Scripts/python.exe benchmarks/phase64/mve/mve_train.py --smoke --arm kd
# Run (real, owner):  .venv/Scripts/python.exe benchmarks/phase64/mve/mve_train.py --tag full --arm kd --steps 20000 ...
# DDP (2xT4):         torchrun --nproc_per_node=2 benchmarks/phase64/mve/mve_train.py --tag full --arm kd ...
import argparse, json, math, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from mve_model import MVEStudent, S0, MOE, param_report   # noqa: E402

DATA = os.path.join(ROOT, "results", "phase64", "mve")
CKPT = os.path.join(ROOT, "results", "phase64", "mve", "ckpt")
STAGE_SPLIT = dict(C=0.55, D=0.20, E=0.20, F=0.05)         # plan §4 token split


# ------------------------------------------------------------------ chunked KD store -----------------
class KDChunks:
    """Rolling window of teacher-logit chunks. Holds <=2 resident, advances, and (--delete-behind) removes the
    consumed file -- the production storage discipline, not a simulation of it."""
    def __init__(s, tag, anchors, resident=2, delete_behind=False):
        s.dir = os.path.join(DATA, f"logits_{tag}")
        s.man = json.load(open(os.path.join(s.dir, "manifest.json")))
        s.chunks = s.man["chunks"]; s.K = s.man["K"]
        s.anchors = anchors; s.resident = resident; s.delete_behind = delete_behind
        s.pos = 0; s.cache = {}
        s.a_valid = anchors >= 0
        s.a_sorted = anchors.copy(); s.a_sorted[~s.a_valid] = np.iinfo(np.int64).max
    def _load(s, i):
        if i in s.cache: return s.cache[i]
        c = s.chunks[i]; z = np.load(os.path.join(s.dir, c["path"]))
        s.cache[i] = (int(z["j0"]), z["ids"].astype(np.int64), z["pq"])
        return s.cache[i]
    def window(s):
        """(student position range, dict row->(ids,pq)) for the currently resident chunks."""
        idx = [(s.pos + d) % len(s.chunks) for d in range(min(s.resident, len(s.chunks)))]
        loaded = [s._load(i) for i in idx]
        j0 = min(l[0] for l in loaded); j1 = max(l[0] + len(l[1]) for l in loaded)
        # anchors are NON-DECREASING in p (student and teacher tokenize the same byte stream in order) -> the student
        # positions whose anchor lands in [j0,j1) form a contiguous range. searchsorted on the -1-masked copy.
        p_lo = int(np.searchsorted(s.a_sorted, j0, "left"))
        p_hi = int(np.searchsorted(s.a_sorted, j1, "left"))
        return p_lo, p_hi, idx, loaded
    def advance(s):
        old = s.pos
        s.pos = (s.pos + 1) % len(s.chunks)
        drop = [i for i in list(s.cache) if i not in [(s.pos + d) % len(s.chunks) for d in range(s.resident)]]
        for i in drop:
            s.cache.pop(i, None)
            if s.delete_behind:
                p = os.path.join(s.dir, s.chunks[i]["path"])
                if os.path.exists(p): os.remove(p)
        return old
    def gather(s, rows, loaded):
        """rows (M,) teacher indices -> (ids (M,K) int64, p (M,K) float32). Rows outside the resident set -> mask 0."""
        M = len(rows)
        ids = np.zeros((M, s.K), dtype=np.int64); pq = np.zeros((M, s.K), dtype=np.uint8)
        ok = np.zeros(M, dtype=bool)
        for j0, cid, cpq in loaded:
            m = (rows >= j0) & (rows < j0 + len(cid))
            if not m.any(): continue
            r = rows[m] - j0
            ids[m] = cid[r]; pq[m] = cpq[r]; ok |= m
        return ids, pq.astype(np.float32) / 255.0, ok


# ------------------------------------------------------------------ losses ---------------------------
def kd_targets(ids, p, t2s, V, dev):
    """ANCHOR KD (the sealed D3 design): project teacher top-K onto the student vocab through the FIRST student token
    of each teacher token's bytes: q[s] = sum_{t: t2s[t]=s} p[t], renormalized over the mapped mass.
    Measured property (kd_information.py): this delivers ~15% of H(teacher) -- the many-to-one collapse onto the
    first token sums away most of the teacher's uncertainty. Kept as the default because it is what is PRE-REGISTERED."""
    tid = torch.from_numpy(ids).to(dev)                        # (M,K)
    pp = torch.from_numpy(p).to(dev)                           # (M,K)
    s = t2s[tid]                                               # (M,K) student id or -1
    m = (s >= 0).float()
    q = torch.zeros(tid.shape[0], V, device=dev, dtype=torch.float32)
    q.scatter_add_(1, s.clamp_min(0), pp * m)
    q = q / q.sum(1, keepdim=True).clamp_min(1e-8)
    return q


def kd_targets_span(ids, p, step, prefix, pmask, dtok, dlen, V, dev):
    """SPAN KD (the alternative, same stored rows): at student position with segment-step s, keep only the teacher
    candidates whose student decomposition still AGREES with the s tokens already emitted inside the segment, and
    whose decomposition is longer than s. Renormalize; the target is their (s+1)-th student token.

    By the chain rule the teacher's uncertainty factorizes across the student tokens of the span -- the choice between
    ' the'/' then'/' they' is not made at ' t', it is made at the token after it. Anchor KD supervises only the first
    (near-deterministic) token and hands the informative interior positions a hard CE label. This reads the SAME K=32
    rows at every interior position instead. Measured: ~83-86% of H(teacher) vs ~15%. Zero extra teacher compute.

      ids    (M,K) teacher candidate ids     p      (M,K) their probs
      step   (M,)  segment step s            prefix (M,S) student tokens already emitted in the segment
      pmask  (M,S) which prefix slots are valid (in-window)
    """
    tid = torch.from_numpy(ids).to(dev); pp = torch.from_numpy(p).to(dev)
    st = torch.from_numpy(step).to(dev)                        # (M,)
    pf = torch.from_numpy(prefix).to(dev); pm = torch.from_numpy(pmask).to(dev)
    M, K = tid.shape; S = dtok.shape[1]
    dt = dtok[tid]                                             # (M,K,S) student decomposition of each candidate
    dl = dlen[tid]                                             # (M,K)
    u = torch.arange(S, device=dev)[None, None, :]             # (1,1,S)
    need = (u < st[:, None, None]) & pm[:, None, :]            # prefix slots that must agree
    agree = ((dt == pf[:, None, :]) | ~need).all(-1)           # (M,K)
    alive = agree & (dl > st[:, None])
    nxt = dt.gather(2, st.clamp(0, S - 1)[:, None, None].expand(M, K, 1)).squeeze(2)   # (M,K) the (s+1)-th token
    w = pp * alive.float()
    tot = w.sum(1, keepdim=True)
    q = torch.zeros(M, V, device=dev, dtype=torch.float32)
    q.scatter_add_(1, nxt.clamp_min(0).long(), w)
    ok = (tot.squeeze(1) > 1e-6)                               # no surviving candidate -> plain CE at that position
    q = q / tot.clamp_min(1e-8)
    return q, ok


def kd_loss(logits_a, q, reverse=False):
    """forward KL (stage C-E): CE(q, student) = -sum q log p  [= KL(q||p) + H(q), H(q) const wrt params]
       reverse KL (stage F): KL(p||q) restricted to q's support, with p renormalized on that support."""
    logp = F.log_softmax(logits_a.float(), dim=-1)
    if not reverse:
        return -(q * logp).sum(-1)
    sup = q > 0
    p = logp.exp() * sup
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-8)
    return (p * (torch.log(p.clamp_min(1e-8)) - torch.log(q.clamp_min(1e-8)))).sum(-1)


# ------------------------------------------------------------------ main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--arm", choices=["kd", "ce"], default="kd", help="kd = teacher KD; ce = the D3 control arm")
    ap.add_argument("--kd", choices=["anchor", "span"], default="anchor",
                    help="anchor = the SEALED D3 design (top-K projected onto the segment's first student token; "
                         "measured to deliver ~15%% of H(teacher)). span = prefix-conditioned over the whole segment "
                         "from the SAME stored rows (~83-86%% of H(teacher), zero extra teacher compute). "
                         "Default is the sealed design: switching it is the Architect's call, not the Builder's.")
    ap.add_argument("--recall", choices=["on", "off"], default="on", help="off = the D4-clause-2 control arm")
    ap.add_argument("--stages", default="CDEF")
    ap.add_argument("--steps", type=int, default=20000, help="TOTAL steps, split across stages by STAGE_SPLIT")
    ap.add_argument("--seq", type=int, default=512); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--alpha", type=float, default=0.5,
                    help="KD weight at anchor positions: loss = (1-a)*CE + a*KD there, CE elsewhere")
    ap.add_argument("--lam-nce", type=float, default=0.05); ap.add_argument("--load-w", type=float, default=0.01)
    ap.add_argument("--fp16", action="store_true"); ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--eval-tok", type=int, default=200_000)
    ap.add_argument("--ckpt-min", type=float, default=30.0); ap.add_argument("--resume", default="")
    ap.add_argument("--delete-behind", action="store_true")
    ap.add_argument("--chunk-steps", type=int, default=200, help="steps before advancing the resident chunk window")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="auto"); ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.smoke: a.steps = 120; a.seq = 128; a.batch = 4; a.eval_tok = 20_000; a.chunk_steps = 20

    # ---- DDP (torchrun) ----------------------------------------------------------------------------
    ws = int(os.environ.get("WORLD_SIZE", 1)); rank = int(os.environ.get("RANK", 0))
    lrank = int(os.environ.get("LOCAL_RANK", 0)); ddp = ws > 1
    if ddp:
        bk = "nccl" if (torch.cuda.is_available() and torch.distributed.is_nccl_available()) else "gloo"
        torch.distributed.init_process_group(backend=bk)
        if torch.cuda.is_available(): torch.cuda.set_device(lrank)
    dev = a.device if a.device != "auto" else (f"cuda:{lrank}" if torch.cuda.is_available() else "cpu")
    dev_type = "cuda" if dev.startswith("cuda") else "cpu"
    amp = torch.float16 if a.fp16 else (torch.bfloat16 if a.bf16 else None)
    scaler = torch.amp.GradScaler(dev_type, enabled=(amp == torch.float16))   # T4 = fp16, no bf16 -> loss scaling
    torch.manual_seed(a.seed); np.random.seed(a.seed + rank)
    P0 = (rank == 0)
    def log(*x):
        if P0: print(*x, flush=True)

    # ---- data ---------------------------------------------------------------------------------------
    meta = json.load(open(os.path.join(DATA, f"meta_{a.tag}.json")))
    V = meta["V_student"]
    ids = np.fromfile(os.path.join(DATA, f"ts_{a.tag}.u16"), dtype=np.uint16).astype(np.int64)
    anchors = np.fromfile(os.path.join(DATA, f"anchors_{a.tag}.i32"), dtype=np.int32).astype(np.int64)
    t2s_np = np.fromfile(os.path.join(DATA, f"t2s_{a.tag}.i32"), dtype=np.int32).astype(np.int64)
    t2s = torch.from_numpy(t2s_np).to(dev)
    dz = np.load(os.path.join(DATA, f"decomp_{a.tag}.npz"))
    dtok = torch.from_numpy(dz["tok"].astype(np.int64)).to(dev)      # (Vt,S) student decomposition of each teacher tok
    dlen = torch.from_numpy(dz["len"].astype(np.int64)).to(dev)      # (Vt,)
    SPAN_S = int(dz["S"])
    # segment map: for every student position, which teacher row governs it and how far into the segment it sits.
    _last = np.where(anchors >= 0, np.arange(len(anchors)), -1)
    _last = np.maximum.accumulate(_last)
    seg_row = np.where(_last >= 0, anchors[np.clip(_last, 0, None)], -1)
    seg_step = np.where(_last >= 0, np.arange(len(anchors)) - _last, 0)
    sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase62"))
    from cartography import Bpe
    bpe = Bpe.load(os.path.join(DATA, f"bpe{V}_ts.bin"))
    el = torch.tensor(bpe.exp_len, dtype=torch.long)
    ntr = meta["n_train_tok"]; train_hi = ntr; val = ids[ntr:]
    kdc = KDChunks(a.tag, anchors, delete_behind=a.delete_behind) if a.arm == "kd" else None

    # ---- model --------------------------------------------------------------------------------------
    model = MVEStudent(V, **S0).to(dev)
    pr = param_report(model)
    log(f"Phase64.4 MVE | arm={a.arm} recall={a.recall} stages={a.stages} | dev={dev} amp={amp} ddp={ws}")
    log(f"  student S0: D{S0['D']} N{S0['N']} L{S0['L']} | params={pr['total']/1e6:.2f}M (MLP {pr['mlp']/1e6:.2f}M) "
        f"| V={V} | corpus {meta['bytes']/2**20:.0f}MiB, {meta['n_student_tok']} tok, anchors {100*meta['anchor_frac']:.1f}%")
    log(f"  teacher={meta['teacher']} K={kdc.K if kdc else 0} | alpha={a.alpha} lam_nce={a.lam_nce} | seq={a.seq} batch={a.batch} steps={a.steps}")

    # ---- KD TARGET AGREEMENT (the D3 mechanism check; run BEFORE any training, costs seconds) ---------
    # The span map is only worth training on if the projected teacher distribution actually points at the true next
    # student token. If q[y_true] were ~0 the projection would be broken (bad t2s, or an anchor off-by-one) and the
    # KD arm would be training on noise. This is the instrument that separates "D3 is hard" from "D3 is buggy".
    if kdc is not None and P0:
        rr = np.random.default_rng(7)
        w_lo, w_hi, idx0, loaded0 = kdc.window()                 # sample INSIDE the resident chunk window
        w_hi = min(w_hi, train_hi - a.seq - 1)
        p0 = rr.integers(w_lo, max(w_hi, w_lo + 1), size=8)
        anc = np.stack([anchors[i:i+a.seq] for i in p0]).reshape(-1)
        yy = np.stack([ids[i+1:i+1+a.seq] for i in p0]).reshape(-1)
        selq = np.nonzero(anc >= 0)[0]
        ti_, pr_, ok_ = kdc.gather(anc[selq], loaded0)
        selq = selq[ok_]
        if len(selq):
            q = kd_targets(ti_[ok_], pr_[ok_], t2s, V, dev)
            yt = torch.from_numpy(yy[selq]).to(dev)
            mass = q.gather(1, yt[:, None]).squeeze(1)
            top1 = (q.argmax(1) == yt).float().mean().item()
            log(f"  KD-target check on {len(selq)} anchors: mass on TRUE next token = {mass.mean().item():.3f} "
                f"(median {mass.median().item():.3f}) | argmax(q)==y: {100*top1:.1f}% | mapped mass/row: "
                f"{float((pr_[ok_].sum(1)).mean()):.3f}")
            # THE D3 QUESTION, quantified. KD is only worth its cost if the target carries MORE than the hard label
            # ("dark knowledge"). The span projection maps every teacher token to the FIRST student token of its
            # bytes -- and first-tokens are low-entropy (" the"/" then"/" they" collapse onto the same student token).
            # If H(q) << H(teacher), the projection has destroyed precisely the information KD exists to transfer,
            # and the KD arm is training on an expensive near-one-hot. Report both entropies in bits.
            pt = torch.from_numpy(pr_[ok_]).to(dev)                      # (M,K) teacher top-K probs
            pt = pt / pt.sum(1, keepdim=True).clamp_min(1e-8)
            h_t = -(pt * pt.clamp_min(1e-9).log2()).sum(1)               # teacher entropy on its own top-K
            h_q = -(q * q.clamp_min(1e-9).log2()).sum(1)                 # projected-target entropy
            log(f"    dark-knowledge: H(teacher top-K) = {h_t.mean().item():.3f} bits  ->  H(q projected) = "
                f"{h_q.mean().item():.3f} bits   (retained {100*h_q.mean().item()/max(h_t.mean().item(),1e-6):.0f}%); "
                f"q is near-one-hot on {100*(mass>0.95).float().mean().item():.0f}% of anchors")

    # ONE optimizer factory: the stage transitions (D/E) rebuild the optimizer after the model surgery, and they must
    # rebuild the SAME kind -- otherwise the curriculum silently drops back to fp32 Adam states after stage C and the
    # plan's 8-bit memory budget stops holding where it matters most (the MoE stage, the biggest parameter count).
    use8 = [dev_type == "cuda"]
    def make_opt(lr):
        if use8[0]:
            try:
                import bitsandbytes as bnb
                return bnb.optim.AdamW8bit(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
            except Exception as e:
                use8[0] = False; log(f"  8-bit optimizer unavailable ({str(e)[:50]}) -> fp32 AdamW")
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    opt = make_opt(a.lr)
    log(f"  optimizer: {type(opt).__name__}")

    def wrap(m):
        if not ddp: return m
        return nn.parallel.DistributedDataParallel(m, device_ids=[lrank] if dev_type == "cuda" else None)
    net = wrap(model)

    def val_bpb(cap):
        model.eval(); bits = 0.0; nb = 0
        with torch.no_grad():
            W = a.seq; pos = 0; lim = min(cap, len(val) - 1)
            while pos + W + 1 <= lim:
                x = torch.from_numpy(val[pos:pos+W][None]).to(dev); y = torch.from_numpy(val[pos+1:pos+1+W][None]).to(dev)
                lg, _ = model(x, y)
                bits += F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction="sum").item() / math.log(2)
                nb += int(el[y.reshape(-1).cpu()].sum()); pos += W
        model.train(); return bits / max(nb, 1)

    # ---- the step ------------------------------------------------------------------------------------
    rng = np.random.default_rng(a.seed + 1000 * rank)
    state = dict(p_lo=0, p_hi=0, idx=[], loaded=[])
    def refresh():
        if kdc is None:
            state.update(p_lo=0, p_hi=train_hi - a.seq - 1, idx=[], loaded=[]); return
        lo, hi, idx, loaded = kdc.window()
        hi = min(hi, train_hi - a.seq - 1)
        if hi - lo < a.seq + 2:   # degenerate window (tiny smoke corpus) -> fall back to the whole train range
            lo, hi = 0, train_hi - a.seq - 1
        state.update(p_lo=lo, p_hi=hi, idx=idx, loaded=loaded)
    refresh()

    def step_loss(reverse=False):
        p = rng.integers(state["p_lo"], max(state["p_hi"], state["p_lo"] + 1), size=a.batch)
        x = torch.from_numpy(np.stack([ids[i:i+a.seq] for i in p])).to(dev)
        y = torch.from_numpy(np.stack([ids[i+1:i+1+a.seq] for i in p])).to(dev)
        ctx = torch.autocast(dev_type, dtype=amp) if amp is not None else torch.autocast(dev_type, enabled=False)
        with ctx:
            logits, aux = net(x, y)
        ce = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1), reduction="none")   # (B*T,)
        nkd = 0; kd_m = torch.zeros((), device=dev)
        if kdc is not None and a.alpha > 0:
            pf = (p[:, None] + np.arange(a.seq)[None, :]).reshape(-1)            # global student index of each slot
            if a.kd == "anchor":
                row = anchors[pf]                                                # KD only AT the segment start
                sel = np.nonzero(row >= 0)[0]
            else:
                row = seg_row[pf]; stp = seg_step[pf]                            # KD at EVERY position of the segment
                sel = np.nonzero(row >= 0)[0]
            if len(sel):
                tids_r, pr_r, ok = kdc.gather(row[sel], state["loaded"])
                sel = sel[ok]; tids_r = tids_r[ok]; pr_r = pr_r[ok]
            if len(sel):
                if a.kd == "anchor":
                    q = kd_targets(tids_r, pr_r, t2s, V, dev); okm = None
                else:
                    ss = stp[sel]
                    gi = np.clip((pf[sel] - ss + 1)[:, None] + np.arange(SPAN_S)[None, :], 0, len(ids) - 1)
                    pref = ids[gi].astype(np.int64)                              # tokens already emitted in the segment
                    pmask = np.arange(SPAN_S)[None, :] < ss[:, None]
                    q, okm = kd_targets_span(tids_r, pr_r, ss, pref, pmask, dtok, dlen, V, dev)
                si = torch.from_numpy(sel).to(dev)
                if okm is not None:                                              # drop positions with no live candidate
                    si = si[okm]; q = q[okm]
                if si.numel():
                    kdv = kd_loss(logits.reshape(-1, V)[si], q, reverse=reverse)
                    nkd = int(si.numel())
                    ce = ce.clone(); ce[si] = ce[si] * (1.0 - a.alpha)
                    kd_m = kdv.sum() * a.alpha
        loss = (ce.sum() + kd_m) / (a.batch * a.seq) + aux
        return loss, nkd, float(ce.mean())

    # ---- the curriculum -------------------------------------------------------------------------------
    stages = [c for c in a.stages]
    tot_w = sum(STAGE_SPLIT[c] for c in stages)
    budget = {c: max(1, int(a.steps * STAGE_SPLIT[c] / tot_w)) for c in stages}
    rows = []; t_ck = time.time(); gstep = 0
    hist = []      # (stage, step, loss) -- the loss-curve, for the DDP parity check and the QAT-switch read

    for st in stages:
        # ---- stage transitions (the model surgery) ----
        if st == "D":
            n = model.qat_ternary(); model.to(dev); net = wrap(model)
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr * 0.5, betas=(0.9, 0.95), weight_decay=0.1)
            log(f"\n== [D] QAT switch: {n} MLP linears -> BitLinear158 (weights carried over), lr x0.5 ==")
        if st == "E":
            n = model.upcycle_moe(dev_type=dev_type, load_w=a.load_w, seed=a.seed); model.to(dev)
            msg = f"\n== [E] MoE upcycle: {n} MLPs -> E{MOE['E']}xh{MOE['hid_e']} top{MOE['k']} (seeded from dense hidden)"
            if a.recall == "on":
                nr = model.add_recall(lam_nce=a.lam_nce); msg += f" + recall slot ({nr/1e3:.1f}K par, gate=0 -> identity at insertion)"
            model.to(dev); net = wrap(model)
            opt = make_opt(a.lr * 0.5)
            pr = param_report(model)
            log(msg + f" ==\n   params total={pr['total']/1e6:.2f}M active={pr['active']/1e6:.2f}M (active/total={pr['active']/pr['total']:.2f})")
        if st == "F":
            for g in opt.param_groups: g["lr"] = a.lr * 0.1
            log("\n== [F] reverse-KL fine-tune (KL direction flipped at anchors), lr x0.1 ==")

        b0 = val_bpb(min(a.eval_tok, 20000))
        log(f"\n== stage {st}: {budget[st]} steps | val BPB at entry = {b0:.4f} ==")
        model.train(); t0 = time.time(); ntok_seen = 0
        for i in range(budget[st]):
            if kdc is not None and gstep and gstep % a.chunk_steps == 0:
                kdc.advance(); refresh()
            opt.zero_grad(set_to_none=True)
            loss, nkd, ce_m = step_loss(reverse=(st == "F"))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            gstep += 1; ntok_seen += a.batch * a.seq * ws
            hist.append((st, gstep, float(loss)))
            if i == 0 or (i + 1) % max(1, budget[st] // 5) == 0:
                nce = model.recall._last_nce if model.recall is not None else 0.0
                gate = float(model.recall.gate.detach()) if model.recall is not None else 0.0
                log(f"  [{st}] {i+1:5d}/{budget[st]}  loss={float(loss):.4f} ce={ce_m:.4f} gnorm={float(gn):.2f} "
                    f"kd@{nkd}/{a.batch*a.seq}" + (f" nce={nce:.3f} gate={gate:+.4f}" if model.recall else ""))
            if (time.time() - t_ck) / 60.0 > a.ckpt_min and P0:
                os.makedirs(CKPT, exist_ok=True)                       # preemption insurance (Kaggle): ~30 min cadence
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), stage=st, gstep=gstep,
                                arm=a.arm, recall=a.recall, cfg=vars(a)), os.path.join(CKPT, "last.pt"))
                t_ck = time.time(); log(f"  [ckpt] saved at step {gstep}")
        dt = time.time() - t0
        b1 = val_bpb(a.eval_tok)
        tps = ntok_seen / max(dt, 1e-9)
        rows.append(dict(stage=st, steps=budget[st], bpb_in=b0, bpb_out=b1, d=b1 - b0, tok_s=tps, s=dt))
        log(f"  -> stage {st} done: val BPB {b0:.4f} -> {b1:.4f} ({b1-b0:+.4f}) | {tps:.0f} tok/s | {dt/60:.1f} min")

    # ---- report ---------------------------------------------------------------------------------------
    if P0:
        os.makedirs(CKPT, exist_ok=True)
        out = a.out or os.path.join(CKPT, f"mve_{a.arm}_{a.recall}_{a.tag}.pt")
        torch.save(dict(model=model.state_dict(), cfg=vars(a), rows=rows, hist=hist), out)
        log(f"\n==== MVE run [{a.tag}] arm={a.arm} recall={a.recall} ====")
        log(f"  {'stage':6s} {'steps':>7s} {'BPB in':>9s} {'BPB out':>9s} {'delta':>8s} {'tok/s':>9s} {'min':>7s}")
        for r in rows:
            log(f"  {r['stage']:6s} {r['steps']:7d} {r['bpb_in']:9.4f} {r['bpb_out']:9.4f} {r['d']:+8.4f} "
                f"{r['tok_s']:9.0f} {r['s']/60:7.1f}")
        log(f"  saved -> {out}")
    if ddp: torch.distributed.destroy_process_group()
    log("STOP. MVE curriculum above. No commit.")


if __name__ == "__main__":
    main()
