#!/usr/bin/env python3
# Sparse Capacity, phase 2 (D-0, D-R): one training run of one arm, then its evaluation.
# Protocol: docs/research/SPARSE_CAPACITY/F2_BANK_PROTOCOL.md. The verdict is computed by sc2_verdict.py.
#
#   python benchmarks/sparse_capacity/sc2_train.py --selftest
#   python benchmarks/sparse_capacity/sc2_train.py --dry  --arm m64                  (real shapes, a few steps, no data)
#   python benchmarks/sparse_capacity/sc2_train.py --run  --arm m16 --seed 1 --data-dir <dir with the m0 package>
#   python benchmarks/sparse_capacity/sc2_train.py --run  ... --resume                (after a crash or a freeze)
#
# Arms: dense (ternary + engine activations, the rung-1 MLP), m4 / m16 / m64 (bank of 128*M experts of h8, top-128),
# dense-fp (the dense arm without quantization: the precision reading). Everything else is identical across arms.
import argparse, hashlib, json, math, os, platform, random, subprocess, sys, time
import numpy as np
import torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase62"))
import sc2_model as SM      # noqa: E402
import sc2_probes as PR     # noqa: E402
from cartography import Bpe  # noqa: E402

# m1 = 128 experts, all active (E = k): the bank's quantization, gating and router with no extra capacity, the D-0
# baseline (review item 2). m64 runs micro-batch 4 (12 GB), the others 16: the load-balance loss of m64 sees 2,048
# tokens per micro-batch instead of 8,192 (a noisier coefficient, not a bias; protocol section 8).
ARMS = {"dense": dict(M=0, quant=True, mb=16), "dense-fp": dict(M=0, quant=False, mb=16),
        "m1": dict(M=1, quant=True, mb=16), "m4": dict(M=4, quant=True, mb=16), "m16": dict(M=16, quant=True, mb=16),
        "m64": dict(M=64, quant=True, mb=4)}
SEQ, GLOBAL_SEQS = 512, 16                      # 8192 tokens per optimizer step (the rung-1 DDP step)
TOKENS = 60_000_000
LR, WARMUP, LR_FLOOR, WD, BETAS, CLIP = 3e-3, 200, 0.1, 0.1, (0.9, 0.95), 1.0
P_K, P_R = 0.04, 0.06                           # share of training sequences that are probe sequences
VAL_TOK, VAL_CURVE_TOK, CURVE_EVERY = 2_000_000, 262_144, 1000
TEMPS = (0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3)
K_EVAL = (256, 2048, 4096)                      # entries scored per book (all, all, a fixed sample of 16384)
R_PER_CELL = 500
CKPT_MIN = 30
DRY = dict(steps=12, val=16_384, curve=8_192, curve_every=6, k_eval=(64, 64, 64), r_cell=16, unseen=64)


def git(*a):
    return subprocess.run(["git", "-C", ROOT] + list(a), capture_output=True, text=True).stdout.strip()


def sha256(path, limit=None):
    h = hashlib.sha256(); n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b: break
            h.update(b); n += len(b)
            if limit and n >= limit: break
    return h.hexdigest()


def emit(fh, rec):
    fh.write(json.dumps(rec) + "\n"); fh.flush()


# ------------------------------------------------------------------------------------------------------------ data
class Data:
    """Code windows from the rung-1 m0 package (ts_m0.u16, meta_m0.json, bpe2048_ts.bin), plus the two probes."""
    def __init__(s, data_dir, seed, dry=False):
        s.dry = dry
        if dry:
            s.bpe = Bpe.byte_raw(); s.V = 2048
            rng = np.random.default_rng(0)
            s.ids = rng.integers(32, 127, 4_000_000).astype(np.uint16)          # byte-like noise, no corpus
            s.ntr = 3_500_000; s.meta = dict(dry=True); s.p62 = None; s.ts_sha = "dry"
        else:
            s.meta = json.load(open(os.path.join(data_dir, "meta_m0.json")))
            s.V = int(s.meta["V_student"])
            path = os.path.join(data_dir, "ts_m0.u16")
            if os.path.getsize(path) != 2 * int(s.meta["n_student_tok"]):
                sys.exit("ERROR: ts_m0.u16 does not match meta_m0.json")
            s.ids = np.memmap(path, dtype=np.uint16, mode="r")
            s.ntr = int(s.meta["n_train_tok"])
            s.bpe = Bpe.load(os.path.join(data_dir, f"bpe{s.V}_ts.bin"))
            p62 = os.path.join(data_dir, "p62_m0.u16")
            s.p62 = np.fromfile(p62, dtype=np.uint16) if os.path.exists(p62) else None
            s.ts_sha = sha256(path, limit=1 << 28)                                 # first 256 MiB: identity, cheap
        s.el = np.array(s.bpe.exp_len + [1] * max(0, 2048 - len(s.bpe.exp_len)), dtype=np.int64)
        s.rng = random.Random(seed); s.nrng = np.random.default_rng(seed)
        s.books, s.unseen = PR.make_books()
        s.k_tok = [[(s.enc(PR.k_prefix(n)), s.enc(PR.k_answer(v))) for n, v in b.items()] for b in s.books]
        s.r_test, keys = PR.r_test_set(per_cell=R_PER_CELL if not dry else DRY["r_cell"])
        s.r_gen = PR.r_train_stream(seed + 1000, keys); s.r_count = 0
        s.filler = PR.r_train_stream(99, keys)                                     # context before an R test item

    def enc(s, text):
        return s.bpe.encode(text.encode())

    def code_seq(s):
        p = int(s.nrng.integers(0, s.ntr - SEQ - 1))
        return np.asarray(s.ids[p:p + SEQ + 1], dtype=np.int64)

    def k_seq(s):
        out = []
        while len(out) < SEQ + 1:
            b = s.k_tok[s.rng.randrange(len(s.k_tok))]
            pre, ans = b[s.rng.randrange(len(b))]
            out += pre + ans
        off = s.rng.randrange(0, len(out) - SEQ) if len(out) > SEQ + 1 else 0
        return np.asarray(out[off:off + SEQ + 1], dtype=np.int64)

    def r_seq(s):
        out = []
        while len(out) < SEQ + 1:
            out += s.enc(next(s.r_gen)[1]); s.r_count += 1
        off = s.rng.randrange(0, len(out) - SEQ) if len(out) > SEQ + 1 else 0
        return np.asarray(out[off:off + SEQ + 1], dtype=np.int64)

    def batch(s, n):
        rows, kinds = [], []
        for _ in range(n):
            u = s.rng.random()
            if u < P_K: rows.append(s.k_seq()); kinds.append("k")
            elif u < P_K + P_R: rows.append(s.r_seq()); kinds.append("r")
            else: rows.append(s.code_seq()); kinds.append("c")
        return torch.from_numpy(np.stack(rows)), kinds

    def state(s):
        return dict(rng=s.rng.getstate(), nrng=s.nrng.bit_generator.state, r_count=s.r_count)

    def skip_r(s, n):
        """Advance the R training generator by n instances (resume: training instances stay fresh, never repeated)."""
        for _ in range(n):
            next(s.r_gen)
        s.r_count = n


# ------------------------------------------------------------------------------------------------------------ eval
@torch.no_grad()
def eval_stream(model, stream, el, dev, mb, temps=(1.0,)):
    """BPB and top-1 over a token stream in full windows of SEQ (every target scored once). For each temperature the
    bits are accumulated separately on the two halves of the stream (the temperature control fits on half A and
    reports half B). Returns dict."""
    model.eval()
    n = (len(stream) - 1) // SEQ
    half = n // 2
    bits = np.zeros((2, len(temps))); nb = np.zeros(2); top1 = np.zeros(2); ntok = np.zeros(2)
    T = torch.tensor(temps, device=dev).view(-1, 1, 1)
    for i in range(0, n, mb):
        k = min(mb, n - i)
        rows = np.stack([np.asarray(stream[(i + j) * SEQ:(i + j) * SEQ + SEQ + 1], dtype=np.int64) for j in range(k)])
        xy = torch.from_numpy(rows).to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            lg, _ = model(xy[:, :-1])
        lg = lg.float(); y = xy[:, 1:]
        for j in range(k):
            hf = 0 if (i + j) < half else 1
            l = lg[j]                                                       # (SEQ,V)
            lp = torch.log_softmax(l.unsqueeze(0) / T, -1).gather(-1, y[j].view(1, -1, 1).expand(len(temps), -1, 1))
            bits[hf] += (-lp.squeeze(-1).sum(-1) / math.log(2)).cpu().numpy()
            top1[hf] += float((l.argmax(-1) == y[j]).sum()); ntok[hf] += SEQ
            nb[hf] += float(el[y[j].cpu().numpy()].sum())
    model.train()
    i1 = temps.index(1.0)
    best = int(np.argmin(bits[0] / nb[0]))
    tot_bits = bits[:, i1].sum(); tot_b = nb.sum()
    return dict(bpb=tot_bits / tot_b, top1=top1.sum() / ntok.sum(), bytes=int(tot_b), tokens=int(ntok.sum()),
                temp_fit=temps[best], bpb_B_t1=bits[1, i1] / nb[1], bpb_B_tfit=bits[1, best] / nb[1])


@torch.no_grad()
def score_items(model, items, dev, mb=32):
    """items: list of (context_tokens, answer_tokens). Exact match of every answer token as the argmax given the
    prefix (= greedy decoding of the answer). Right-aligned in a window of SEQ tokens."""
    model.eval(); ok = []
    for i in range(0, len(items), mb):
        chunk = items[i:i + mb]
        rows, spans = [], []
        for ctx, ans in chunk:
            seq = (ctx + ans)[-(SEQ + 1):]
            pad = SEQ + 1 - len(seq)
            assert pad == 0, "probe context shorter than the window: fill it with probe lines, never pad"
            rows.append(seq)
            a0 = SEQ + 1 - len(ans)                                           # first answer position in the row
            spans.append((a0, len(ans)))
        xy = torch.tensor(rows, device=dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            lg, _ = model(xy[:, :-1])
        pred = lg.argmax(-1)                                                  # pred[t] predicts xy[t+1]
        for j, (a0, na) in enumerate(spans):
            ok.append(bool((pred[j, a0 - 1:a0 - 1 + na] == xy[j, a0:a0 + na]).all()))
    model.train()
    return ok


def k_context(data, rng, need):
    """Phone-book lines drawn as in training (any of the three books), at least `need` tokens."""
    ctx = []
    while len(ctx) < need:
        b = data.k_tok[rng.randrange(len(data.k_tok))]
        p, a = b[rng.randrange(len(b))]; ctx += p + a
    return ctx


def k_items(data, rng, book_i, entries):
    out = []
    for e in entries:
        pre, ans = data.k_tok[book_i][e]
        out.append((k_context(data, rng, SEQ + 1) + pre, ans))
    return out


def r_context(data, need):
    ctx = []
    while len(ctx) < need:
        ctx += data.enc(next(data.filler)[1])
    return ctx


def eval_probes(model, data, dev, mb, dry=False):
    rng = random.Random(4242)
    res = dict(k={}, r={}); k_all = []
    ke = DRY["k_eval"] if dry else K_EVAL
    for bi, size in enumerate(PR.K_SIZES):
        n = min(ke[bi], size)
        entries = list(range(size)) if n == size else random.Random(bi).sample(range(size), n)
        ok = score_items(model, k_items(data, rng, bi, entries), dev, mb)
        res["k"][str(size)] = sum(ok) / len(ok); k_all += [int(v) for v in ok]
    # control: names never in a book, written with a random number (must be ~0)
    un = data.unseen[:DRY["unseen"]] if dry else data.unseen
    items = [(k_context(data, rng, SEQ + 1) + data.enc(PR.k_prefix(nm)), data.enc(PR.k_answer(v))) for nm, v in un]
    ok = score_items(model, items, dev, mb)
    res["k_unseen"] = sum(ok) / len(ok)                                     # sanity check of the scorer, not a control
    res["k_primary"] = float(np.mean([res["k"][str(s)] for s in PR.K_SIZES]))
    res["k_items"] = k_all
    # R: normal, shuffled-graph control (the edge list of another test instance with the same n), no-model heuristic
    by_n = {}
    for d in data.r_test:
        by_n.setdefault(d["n"], []).append(d)
    items, sh_items, cells, heur, look, walk = [], [], [], [], [], []
    crng = random.Random(77)
    for d in data.r_test:
        filler = r_context(data, SEQ + 1)
        items.append((filler + data.enc(PR.r_prefix(d["adj"], d["s"], d["t"])), data.enc(PR.r_answer(d["path"]))))
        pool = by_n[d["n"]]
        other = pool[rng.randrange(len(pool))]
        while other is d:
            other = pool[rng.randrange(len(pool))]
        sh_items.append((filler + data.enc(PR.r_prefix(other["adj"], d["s"], d["t"])), data.enc(PR.r_answer(d["path"]))))
        cells.append((d["n"], d["L"])); heur.append(PR.r_heuristic(d["adj"], d["s"], d["t"]) == d["path"])
        look.append(PR.r_lookahead(d["adj"], d["s"], d["t"], crng) == d["path"])
        walk.append(PR.r_walk(d["adj"], d["s"], d["t"], crng) == d["path"])
    ok = score_items(model, items, dev, mb); oks = score_items(model, sh_items, dev, mb)
    for key in sorted(set(cells)):
        ix = [i for i, c in enumerate(cells) if c == key]
        res["r"][f"n{key[0]}_L{key[1]}"] = dict(acc=float(np.mean([ok[i] for i in ix])),
                                               shuffled=float(np.mean([oks[i] for i in ix])),
                                               heuristic=float(np.mean([heur[i] for i in ix])),
                                               lookahead=float(np.mean([look[i] for i in ix])),
                                               walk=float(np.mean([walk[i] for i in ix])), n=len(ix))
    pix = [i for i, c in enumerate(cells) if c in PR.R_PRIMARY]
    res["r_primary"] = float(np.mean([ok[i] for i in pix]))
    res["r_primary_shuffled"] = float(np.mean([oks[i] for i in pix]))
    res["r_primary_heuristic"] = float(np.mean([heur[i] for i in pix]))
    res["r_primary_lookahead"] = float(np.mean([look[i] for i in pix]))
    res["r_primary_walk"] = float(np.mean([walk[i] for i in pix]))
    res["r_primary_items"] = [int(ok[i]) for i in pix]                    # same items in every run: paired test
    return res


@torch.no_grad()
def expert_usage(model, stream, dev, mb, ntok):
    """Fraction of experts that receive at least one token over ntok validation tokens, and max/mean load."""
    if model.M == 0:
        return None
    for b in model.blocks: b.mlp.count_acc = None
    model.eval()
    n = min((len(stream) - 1) // SEQ, max(1, ntok // SEQ))
    for i in range(0, n, mb):
        k = min(mb, n - i)
        x = torch.from_numpy(np.stack([np.asarray(stream[(i + j) * SEQ:(i + j) * SEQ + SEQ], dtype=np.int64)
                                       for j in range(k)])).to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            model(x)
    model.train()
    used = [float((b.mlp.count_acc > 0).float().mean()) for b in model.blocks]
    mx = [float(b.mlp.count_acc.max() / b.mlp.count_acc.mean()) for b in model.blocks]
    eff = []                                                              # exp(entropy of the load) / E (review item 8)
    for b in model.blocks:
        p = b.mlp.count_acc / b.mlp.count_acc.sum()
        eff.append(float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum()) / b.mlp.E))
    return dict(used_min=min(used), used_mean=float(np.mean(used)), max_over_mean=max(mx),
                eff_min=min(eff), eff_mean=float(np.mean(eff)), eff_by_layer=eff)


# ------------------------------------------------------------------------------------------------------------ selftest
def engine_ref_dense(x, gate, up, down):
    """numpy float64 replica of engine.c: quant_i8 (absmax/63, round half even), int products, fp scales after."""
    def qi8(v):
        s = np.maximum(np.abs(v).max(-1, keepdims=True) / 63.0, 1e-8)
        return np.clip(np.round(v / s), -63, 63), s
    def tern(w):
        s = np.maximum(np.abs(w).mean(-1, keepdims=True), 1e-5)
        return np.clip(np.round(w / s), -1, 1), s
    qx, sx = qi8(x); qg, sg = tern(gate); qu, su = tern(up); qd, sd = tern(down)
    g = (qx @ qg.T) * sx * sg.T; u = (qx @ qu.T) * sx * su.T
    hdn = np.maximum(g, 0) * np.maximum(u, 0)
    qh, sh = qi8(hdn)
    return (qh @ qd.T) * sh * sd.T


def engine_ref_bank(x, bank):
    D = x.shape[1]
    Wr = bank.router.weight.detach().double().cpu().numpy(); br = bank.router.bias.detach().double().cpu().numpy()
    Wg = bank.Wg.detach().double().cpu().numpy(); Wu = bank.Wu.detach().double().cpu().numpy()
    Wd = bank.Wd.detach().double().cpu().numpy()
    def qi8(v):
        s = np.maximum(np.abs(v).max(-1, keepdims=True) / 63.0, 1e-8)
        return np.clip(np.round(v / s), -63, 63), s
    def tern(w):
        s = np.maximum(np.abs(w).mean(-1, keepdims=True), 1e-5)
        return np.clip(np.round(w / s), -1, 1), s
    out = np.zeros_like(x)
    for n in range(x.shape[0]):
        z = Wr @ x[n] + br; p = np.exp(z - z.max()); p /= p.sum()
        top = np.argsort(-p, kind="stable")[:bank.k]; w = p[top] / p[top].sum() * bank.k
        qx, sx = qi8(x[n:n + 1])
        for e, we in zip(top, w):
            qg, sg = tern(Wg[e]); qu, su = tern(Wu[e]); qd, sd = tern(Wd[e])
            g = (qx @ qg.T) * sx * sg.T; u = (qx @ qu.T) * sx * su.T
            hh = np.maximum(g, 0) * np.maximum(u, 0) * we
            qh, sh = qi8(hh)
            out[n] += ((qh @ qd.T) * sh * sd.T)[0]
    return out


def selftest(dev):
    res = []
    def check(name, val, tol):
        ok = bool(val <= tol); res.append(ok)
        print(f"  {'ok  ' if ok else 'FAIL'} {name:<58} {val:.3e}  (tol {tol:.0e})")
    torch.manual_seed(0); np.random.seed(0)
    # act_q: integer valued, |q| <= 63, round half to even at exact .5 (engine lrintf)
    x = torch.tensor([[63.0, 0.5, 1.5, 2.5, -2.5, 10.0]], device=dev)
    q, s = SM.act_q(x)
    check("act_q round-half-even and range", float((q - torch.tensor([[63, 0, 2, 2, -2, 10.]], device=dev)).abs().max()), 0)
    # dense: torch vs engine replica
    m = SM.DenseMLP(256, 1024).to(dev)
    xx = torch.randn(64, 256, device=dev)
    y = m(xx).detach().double().cpu().numpy()
    ref = engine_ref_dense(xx.double().cpu().numpy(), m.gate.weight.detach().double().cpu().numpy(),
                           m.up.weight.detach().double().cpu().numpy(), m.down.weight.detach().double().cpu().numpy())
    check("dense MLP vs engine replica (rel max)", float(np.abs(y - ref).max() / np.abs(ref).max()), 1e-5)
    # bank: torch vs per-token engine replica, small E
    b = SM.Bank(256, E=48, h=8, k=16).to(dev)
    xx = torch.randn(1, 40, 256, device=dev)
    y = b(xx)[0].detach().double().cpu().numpy()
    ref = engine_ref_bank(xx[0].double().cpu().numpy(), b)
    check("bank vs engine replica, per token loop (rel max)", float(np.abs(y - ref).max() / np.abs(ref).max()), 1e-5)
    # bank gradient: unselected experts exactly zero, selected nonzero
    xx = torch.randn(1, 3, 256, device=dev)
    b.zero_grad(); (b(xx).sum() + b.aux).backward()
    with torch.no_grad():
        topi = torch.softmax(b.router(xx.reshape(-1, 256)), -1).topk(b.k, -1)[1].unique()
    gn = b.Wg.grad.abs().sum((1, 2))
    mask = torch.ones(b.E, dtype=torch.bool, device=dev); mask[topi] = False
    check("bank: grad of unselected experts (abs sum)", float(gn[mask].sum()), 0)
    # a selected expert whose 8 units are all off on its tokens (dReLU) has zero gradient legitimately; so the check is
    # that gate and down gradients agree: down grad nonzero <=> some unit active <=> gate grad nonzero
    dn = b.Wd.grad.abs().sum((1, 2))
    check("bank: selected experts, gate/down gradient disagree (count)",
          float(((gn[topi] > 0) != (dn[topi] > 0)).sum()), 0)
    check("bank: selected experts with nonzero gradient (1 - fraction)", 1 - float((dn[topi] > 0).float().mean()), 0.5)
    # determinism: two identical fwd+bwd, bit equal
    def once():
        torch.manual_seed(3)
        mm = SM.Student(512, M=1).to(dev)
        xi = torch.randint(0, 512, (2, 65), device=dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            lg, aux = mm(xi[:, :-1])
            loss = F.cross_entropy(lg.float().reshape(-1, 512), xi[:, 1:].reshape(-1)) + aux
        loss.backward()
        return torch.cat([p.grad.flatten() for p in mm.parameters() if p.grad is not None])
    g1, g2 = once(), once()
    check("determinism: two runs, max abs grad difference", float((g1 - g2).abs().max()), 0)
    # probes
    pst = PR.selftest()
    check("probes: shortest path not unique or wrong (count)", pst["r_bad_instances"], 0)
    check("probes: books overlap (1 = yes)", 0 if pst["k_books_disjoint"] else 1, 0)
    check("probes: train instance equal to a test instance (count)", pst["leak"], 0)
    # BPE alignment: prefix+answer encoded apart == encoded together (chunks split at spaces, so they must agree)
    bpe = Bpe.byte_raw()
    pre, ans = PR.k_prefix("abcde"), PR.k_answer(12345678)
    check("probes: K encode(prefix)+encode(answer) == encode(line)",
          0 if bpe.encode(pre.encode()) + bpe.encode(ans.encode()) == bpe.encode((pre + ans).encode()) else 1, 0)
    # scorer: a planted oracle scores 1, a planted constant scores 0
    class Oracle(torch.nn.Module):
        def __init__(s, nxt): super().__init__(); s.nxt = nxt; s.M = 0
        def forward(s, x):
            lg = torch.full((x.shape[0], x.shape[1], 300), -9.0, device=x.device)
            for i in range(x.shape[0]):
                lg[i, torch.arange(x.shape[1] - 1), x[i, 1:]] = 9.0
                lg[i, -1, s.nxt[i]] = 9.0
            return lg, None
    items = [([1, 2, 3, 4] * 130, [5, 6, 7]), ([9, 8] * 260, [7, 6])]   # contexts fill the window: no padding
    o = Oracle([7, 6])                     # sees the input row; its last position must name the final answer token
    ok = score_items(o, items, dev)
    check("scorer: planted oracle (1 - accuracy)", 1 - float(np.mean(ok)), 0)
    class Const(torch.nn.Module):
        M = 0
        def forward(s, x):
            lg = torch.zeros((x.shape[0], x.shape[1], 300), device=x.device); lg[..., 0] = 1.0
            return lg, None
    ok = score_items(Const(), items, dev)
    check("scorer: planted constant (accuracy)", float(np.mean(ok)), 0)
    try:
        score_items(Const(), [([1, 2], [3])], dev); refused = 0
    except AssertionError:
        refused = 1
    check("scorer: refuses a context shorter than the window (0 = refuses)", 1 - refused, 0)
    bk = SM.Bank(256, E=256, h=8, k=128)
    check("bank: down init bound equals the dense one, 1/sqrt(1024) (abs diff)",
          abs(float(bk.Wd.abs().max()) - 1 / 32) if float(bk.Wd.abs().max()) <= 1 / 32 else 1.0, 1e-3)
    print("SELFTEST", "PASS" if all(res) else "FAIL", f"({sum(res)}/{len(res)})")
    return all(res)


# ------------------------------------------------------------------------------------------------------------ train
def lr_at(i, steps):
    if i < WARMUP:
        return LR * (i + 1) / WARMUP
    t = (i - WARMUP) / max(1, steps - WARMUP)
    return LR * (LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * t)))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--selftest", action="store_true"); g.add_argument("--dry", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--arm", choices=list(ARMS), default="dense")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "sparse_capacity"))
    ap.add_argument("--ckpt-dir", default="", help="large checkpoints (not in git); default: <out>/<run>/ckpt")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if a.selftest:
        sys.exit(0 if selftest(dev) else 1)
    dry = a.dry
    arm = ARMS[a.arm]
    steps = DRY["steps"] if dry else TOKENS // (SEQ * GLOBAL_SEQS)
    accum = GLOBAL_SEQS // arm["mb"]
    tag = f"f2{'_dry' if dry else ''}_{a.arm}_s{a.seed}"
    run_dir = os.path.join(a.out, tag)
    ckpt_dir = a.ckpt_dir or os.path.join(run_dir, "ckpt")
    os.makedirs(run_dir, exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "last.pt")

    head, upstream = git("rev-parse", "HEAD"), git("rev-parse", "@{u}")
    dirty = git("status", "--porcelain", "--", "benchmarks/sparse_capacity", "docs/research/SPARSE_CAPACITY",
                "benchmarks/phase55", "benchmarks/phase62")
    srcs = {f: sha256(os.path.join(HERE, f)) for f in ("sc2_model.py", "sc2_probes.py", "sc2_train.py")}
    data = Data(a.data_dir, a.seed, dry=dry)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = SM.Student(data.V, M=arm["M"], quant=arm["quant"]).to(dev)
    counts = model.counts()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WD, fused=True)
    cfg = dict(mode="DRY (not a measurement)" if dry else "RUN", arm=a.arm, seed=a.seed, git_head=head,
               pushed=(head == upstream), dirty_files=dirty.splitlines(), src_sha256=srcs,
               python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0), cpu=platform.processor(), core=SM.S0, xproj_rank=SM.XPROJ_RANK,
               M=arm["M"], E=128 * arm["M"], h=SM.H_EXP, k=SM.ACTIVE // SM.H_EXP, active_hidden=SM.ACTIVE,
               quant=("ternary + act_quant=engine (Q=63)" if arm["quant"] else "none (bf16 autocast)"),
               params=counts, load_w=SM.LOAD_W, seq=SEQ, global_seqs=GLOBAL_SEQS, micro_batch=arm["mb"], accum=accum,
               tokens=steps * SEQ * GLOBAL_SEQS, steps=steps, lr=LR, warmup=WARMUP, lr_floor=LR_FLOOR, wd=WD,
               betas=BETAS, clip=CLIP, p_k=P_K, p_r=P_R, k_sizes=PR.K_SIZES, r_n=PR.R_N, r_l=PR.R_L,
               r_primary=PR.R_PRIMARY, r_ood=PR.R_OOD, val_tok=(DRY["val"] if dry else VAL_TOK), temps=TEMPS,
               tf32=torch.backends.cuda.matmul.allow_tf32, data_dir=a.data_dir, ts_sha256_first256MiB=data.ts_sha,
               meta=data.meta, p62=(data.p62 is not None))
    for k, v in cfg.items():
        print(f"CONFIG {k} = {v}")
    refuse = []
    if not dry:
        if dirty: refuse.append("uncommitted changes in the apparatus")
        if head != upstream: refuse.append("HEAD is not pushed (protocol must be pushed before the run)")
        if "3060" not in cfg["gpu"]: refuse.append("not the RTX 3060 of the protocol")
        if not selftest(dev): refuse.append("selftest failed")
    if refuse:
        sys.exit("REFUSED: " + "; ".join(refuse))
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1, default=str)

    start = 0; prev_sec, prev_tok = 0.0, 0
    if a.resume and os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location=dev, weights_only=False)
        if st["cfg_key"] != [a.arm, a.seed, head]:
            sys.exit(f"REFUSED: checkpoint is {st['cfg_key']}, this run is {[a.arm, a.seed, head]}")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start = st["step"]
        prev_sec, prev_tok = st.get("train_sec", 0.0), st.get("tok_done", 0)
        data.rng.setstate(st["data"]["rng"]); data.nrng.bit_generator.state = st["data"]["nrng"]
        data.skip_r(st["data"]["r_count"])
        torch.set_rng_state(st["torch_rng"]); torch.cuda.set_rng_state(st["cuda_rng"])
        print(f"RESUMED at step {start}")
    log = open(os.path.join(run_dir, "train.jsonl"), "a")
    emit(log, dict(kind="start", step=start, time=time.time(), resumed=bool(start)))
    val = data.ids[data.ntr:data.ntr + (DRY["val"] if dry else VAL_TOK) + 1]
    curve_val = val[:(DRY["curve"] if dry else VAL_CURVE_TOK) + 1]
    curve_every = DRY["curve_every"] if dry else CURVE_EVERY
    t_last_ck = time.time(); t0 = time.time(); tok_done = 0
    train_sec = lambda: prev_sec + (time.time() - t0)
    model.train()
    for i in range(start, steps):
        for gp in opt.param_groups: gp["lr"] = lr_at(i, steps)
        tot = 0.0; kinds_all = []
        for _ in range(accum):
            xy, kinds = data.batch(arm["mb"]); kinds_all += kinds
            xy = xy.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                lg, aux = model(xy[:, :-1])
                loss = F.cross_entropy(lg.float().reshape(-1, data.V), xy[:, 1:].reshape(-1))
                full = loss + (aux if aux is not None else 0.0)
            (full / accum).backward()
            tot += float(loss.detach()) / accum
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP))
        if not (math.isfinite(tot) and math.isfinite(gn)):
            emit(log, dict(kind="nonfinite", step=i, loss=tot, gnorm=gn))
            sys.exit(f"STOP: non-finite loss or gradient at step {i} (PLAN section 3)")
        opt.step(); opt.zero_grad(set_to_none=True)
        tok_done += SEQ * GLOBAL_SEQS
        if i % 50 == 0 or i == steps - 1:
            rec = dict(kind="step", step=i, loss=tot, gnorm=gn, lr=lr_at(i, steps), tok_s=(prev_tok + tok_done) / train_sec(),
                       k_share=kinds_all.count("k") / len(kinds_all), r_share=kinds_all.count("r") / len(kinds_all))
            if arm["M"]:
                rec["load"] = float(np.mean([float(b.mlp.stats["load"]) for b in model.blocks]))
                rec["used_batch"] = float(min(float(b.mlp.stats["used"]) for b in model.blocks))
            emit(log, rec)
            print(f"step {i:6d}/{steps} loss {tot:.4f} gn {gn:.2f} {rec['tok_s']:.0f} tok/s", flush=True)
        if (i + 1) % curve_every == 0 or i == steps - 1:
            ev = eval_stream(model, curve_val, data.el, dev, arm["mb"])
            emit(log, dict(kind="curve", step=i + 1, tokens=(i + 1) * SEQ * GLOBAL_SEQS, bpb=ev["bpb"], top1=ev["top1"]))
        if time.time() - t_last_ck > CKPT_MIN * 60 or i == steps - 1:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=i + 1, cfg_key=[a.arm, a.seed, head],
                            data=data.state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                            train_sec=train_sec(), tok_done=prev_tok + tok_done),
                       ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path); t_last_ck = time.time()

    # ---------------------------------------------------------------- final evaluation
    res = dict(arm=a.arm, seed=a.seed, git_head=head, mode=cfg["mode"], params=counts, steps=steps,
               tokens=steps * SEQ * GLOBAL_SEQS, train_hours=train_sec() / 3600)
    res["val"] = eval_stream(model, val, data.el, dev, arm["mb"], temps=TEMPS)
    if data.p62 is not None and not dry:
        res["p62"] = eval_stream(model, data.p62, data.el, dev, arm["mb"])
    res["probes"] = eval_probes(model, data, dev, arm["mb"], dry=dry)
    res["experts"] = expert_usage(model, val, dev, arm["mb"], 262_144 if not dry else 4096)
    json.dump(res, open(os.path.join(run_dir, "result.json"), "w"), indent=1)
    w_sha = None
    if not dry:
        wpath = os.path.join(ckpt_dir, "final_model.pt")
        torch.save(model.state_dict(), wpath); w_sha = sha256(wpath)
    emit(log, dict(kind="done", time=time.time(), final_weights_sha256=w_sha))
    print("RESULT", json.dumps(res))
    print(f"DONE -> {run_dir}")


if __name__ == "__main__":
    main()
