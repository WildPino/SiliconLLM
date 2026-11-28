#!/usr/bin/env python3
# Sparse Capacity, phase 4 (D-M): a bank beyond the VRAM, and two banks inside it with the same router and the same
# tokens. Protocol: docs/research/SPARSE_CAPACITY/F4_BEYOND_VRAM_PROTOCOL.md. The verdict is computed by sc4_verdict.py.
#
#   arm  E per layer  bank (8 layers)  where the masters and AdamW live
#   e2k      2,048       101 M         GPU (phase 2's path: fused AdamW on the GPU)
#   e8k      8,192       403 M         GPU
#   e32k    32,768     1,611 M         host RAM, 19.3 GB with the AdamW state (+6.4 GB pinned gradient); the GPU holds
#                                       the ternary codes and scales (0.43 bytes per weight), sent once per step
# Same core (rung-1 S0, phase 2), same product-key router, same expert maths (sc4_model), same recipe (phase 2's AdamW,
# lr 3e-3, 200 warmup, cosine to 10%, clip 1), same data stream and batches per seed. Readings in the stream: K
# (phase 2's book), P named (64, 512) and the explicit book (sc_proc), with digits separated by spaces.
#
#   python benchmarks/sparse_capacity/sc4_train.py --selftest
#   python benchmarks/sparse_capacity/sc4_train.py --dry --arm <arm> --data-dir <m0>      (real shapes; not a measurement)
#   python benchmarks/sparse_capacity/sc4_train.py --equiv --data-dir <m0>               (host path = GPU path, one step)
#   python benchmarks/sparse_capacity/sc4_train.py --run --arm <arm> --data-dir <m0> [--resume]   (after registration)
import argparse, json, math, os, platform, random, sys, time
import numpy as np
import torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sc2_model as SM      # noqa: E402
import sc2_train as T       # noqa: E402
import sc4_model as M4      # noqa: E402
import sc4_host as H4       # noqa: E402
import sc4_adamw as KA      # noqa: E402
import sc_proc as PP        # noqa: E402
import sc_rt_train as RT    # noqa: E402

ROOT = T.ROOT
CKPT_ROOT = "E:/SiliconLLM_Artifacts/sparse_capacity/f4"
REGISTERED = False            # set to True only in the commit that registers the protocol
ARMS = {"e2k": dict(E=2048, host=False, mb=16), "e8k": dict(E=8192, host=False, mb=8), "e32k": dict(E=32768, host=True, mb=16)}
TOKENS = 240_000_000          # per arm (protocol section 3)
SEQ, MB = T.SEQ, 16           # 16 x 512 = 8,192 tokens per step; e8k in two micro-batches of 8 (its masters and AdamW
                              # take 6.4 GB of the GPU); the host arm in one (its gradient is not accumulated)
S_K, S_P, S_E = 0.04, 0.12, 0.04
P_BOOKS = (64, 512)           # named books in the stream (the 4,096 book of sc_proc is not used)
CURVE_EVERY = 2000            # ~16 M tokens
TIME_EVERY = 50
CKPT_MIN = {False: 60, True: 180}
NTHR = 6


class Data4:
    """Phase 2's code windows and K book, plus the P streams (named books 64 and 512, and the explicit book). The
    attribute names are those the readings of sc_rt_train expect."""
    def __init__(s, data_dir, seed):
        s.d = T.Data(data_dir, 5000 + seed)
        s.V = s.d.V; s.el = s.d.el; s.enc = s.d.enc; s.ids = s.d.ids; s.ntr = s.d.ntr; s.p62 = s.d.p62; s.ts_sha = s.d.ts_sha
        knames = {n for b in s.d.books for n in b} | {n for n, _ in s.d.unseen}
        books, s.punseen, s.ptest = PP.make_pbooks(forbidden=knames)
        s.pbooks = books                                                      # all three, for the readings' indexing
        s.etest, ekeys = PP.e_test_set()
        s.pgen = PP.p_train_stream(books[:len(P_BOOKS)], s.ptest, 6000 + seed)
        s.egen = PP.e_train_stream(ekeys, 6500 + seed)
        s.pfill = PP.p_train_stream(books[:len(P_BOOKS)], s.ptest, 99); s.efill = PP.e_train_stream(ekeys, 99)
        s.rng = random.Random(8000 + seed); s.counts = dict(c=0, k=0, p=0, e=0); s.pn = s.en = 0

    def _lines(s, gen, attr):
        out = []
        while len(out) < SEQ + 1:
            out += s.enc(next(gen)); setattr(s, attr, getattr(s, attr) + 1)
        off = s.rng.randrange(0, len(out) - SEQ) if len(out) > SEQ + 1 else 0
        return np.asarray(out[off:off + SEQ + 1], dtype=np.int64)

    def batch(s, n):
        rows = []
        for _ in range(n):
            u = s.rng.random()
            if u < S_K: rows.append(s.d.k_seq()); s.counts["k"] += 1
            elif u < S_K + S_P: rows.append(s._lines(s.pgen, "pn")); s.counts["p"] += 1
            elif u < S_K + S_P + S_E: rows.append(s._lines(s.egen, "en")); s.counts["e"] += 1
            else: rows.append(s.d.code_seq()); s.counts["c"] += 1
        return torch.from_numpy(np.stack(rows))

    def state(s):
        return dict(rng=s.rng.getstate(), drng=s.d.rng.getstate(), nrng=s.d.nrng.bit_generator.state, pn=s.pn, en=s.en,
                    counts=dict(s.counts))

    def restore(s, st):
        s.rng.setstate(st["rng"]); s.d.rng.setstate(st["drng"]); s.d.nrng.bit_generator.state = st["nrng"]
        for _ in range(st["pn"]): next(s.pgen)
        for _ in range(st["en"]): next(s.egen)
        s.pn, s.en, s.counts = st["pn"], st["en"], dict(st["counts"])


def lr_at(i, steps):
    return T.lr_at(i, steps)                                                  # phase 2's schedule over this run's steps


def build_arm(arm, V, seed, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    if arm["host"]:
        m, hws = H4.build(V, arm["E"], dev, seed, NTHR)
        m = m.to(dev)
    else:
        m = M4.build(V, arm["E"], None, seed).to(dev); hws = []
    opt = torch.optim.AdamW(m.parameters(), lr=T.LR, betas=T.BETAS, weight_decay=T.WD, fused=True)
    m.updater = H4.Updater(hws)
    return m, opt, hws


def train_step(m, opt, hws, xy, V, i, steps, timed=False, skip_host_layer=None, mb=MB, pipelined=True):
    """One optimizer step on MB sequences, in micro-batches of mb (the mean loss of each, divided by their number, as
    phase 2's accumulation). Returns (loss, grad norm before the clip, timings)."""
    accum = xy.shape[0] // mb
    assert accum * mb == xy.shape[0] and (accum == 1 or not hws)
    lr = lr_at(i, steps)
    for gp in opt.param_groups: gp["lr"] = lr
    for hw in hws:
        hw.timed = timed; hw.t["grad_out"] = 0.0                            # gradients were zeroed by the last update
    if timed:
        torch.cuda.synchronize(); t0 = time.perf_counter()
    tot = 0.0
    for c in range(accum):
        x = xy[c * mb:(c + 1) * mb]
        with torch.autocast(xy.device.type, dtype=torch.bfloat16):
            lg, aux = m(x[:, :-1])
            loss = F.cross_entropy(lg.float().reshape(-1, V), x[:, 1:].reshape(-1))
            full = loss + aux
        (full / accum).backward(); tot += float(loss.detach()) / accum
    if hws:
        total, coef = H4.clip_(m, hws, T.CLIP)
    else:
        total = torch.nn.utils.clip_grad_norm_(m.parameters(), T.CLIP); coef = None
    gn = float(total)
    if timed:
        torch.cuda.synchronize(); t1 = time.perf_counter()
    opt.step(); opt.zero_grad(set_to_none=True)
    if hws:
        torch.cuda.synchronize()                                             # the GPU is done with the old codes
        m.updater.start(i + 1, lr, float(coef), T.BETAS, T.WD, KA.EPS, skip_layer=skip_host_layer, sync=not pipelined)
    if timed:
        torch.cuda.synchronize(); t2 = time.perf_counter()
        t = dict(fwd_bwd=t1 - t0, update_issue=t2 - t1)
        if hws:
            t["grad_out"] = sum(hw.t["grad_out"] for hw in hws)
        return tot, gn, t
    return tot, gn, None


def host_times(hws):
    """The host update times of the last finished update (call after Updater.wait)."""
    return dict(host_adam=sum(hw.t["host_adam"] for hw in hws), upload_issue=sum(hw.t["upload"] for hw in hws))


def readings(m, data, dev, final, mb=MB):
    """K, P named (64, 512), explicit: the readings of sc_rt_train on this arm."""
    out = dict(p=RT.eval_p(m, data, dev, mb, (64, 512, 0) if final else (64, 128, 0)),
               e=RT.eval_e(m, data, dev, mb, None if final else 512))
    if final:
        out["k"] = RT.eval_k(m, data, dev, mb)
        items = []
        for nm, rule in list(data.punseen.items()):
            for x in sorted(data.ptest[nm]):
                items.append((RT.fill(data, data.pfill, SEQ + 1) + data.enc(PP.p_prefix(nm, x)),
                              data.enc(PP.p_answer(PP.apply_rule(rule, x)))))
        out["p"]["unseen"] = float(np.mean(T.score_items(m, items, dev, mb)))
    return out


def usage(m, stream, dev, mb):
    """Expert usage over 1 M validation tokens (sc2_train.expert_usage), plus the KL of the load to the uniform
    distribution, the usage measure of PKM and PEER (Lample et al. 2019 section 4.2; PEER table 2)."""
    out = T.expert_usage(m, stream, dev, mb, 1 << 20)
    kl = []
    for b in m.blocks:
        p = b.mlp.count_acc / b.mlp.count_acc.sum(); nz = p[p > 0]
        kl.append(float(math.log(b.mlp.E) + (nz * nz.log()).sum()))
    out["kl_uniform_by_layer"] = kl; out["kl_uniform_max"] = max(kl)
    return out


def save_ckpt(path, m, opt, hws, i, data, cfg_key, train_sec):
    m.updater.wait()
    st = dict(model=m.state_dict(), opt=opt.state_dict(), step=i, cfg_key=cfg_key, data=data.state(),
              torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), train_sec=train_sec,
              host=[hw.state() for hw in hws])
    torch.save(st, path + ".tmp"); os.replace(path + ".tmp", path)


def load_ckpt(path, m, opt, hws, data, dev):
    m.updater.wait()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict({k: v.to(dev) for k, v in ck["model"].items()}); opt.load_state_dict(ck["opt"])
    for hw, hs in zip(hws, ck["host"]):
        hw.load_state(hs)
    data.restore(ck["data"]); torch.set_rng_state(ck["torch_rng"]); torch.cuda.set_rng_state(ck["cuda_rng"])
    return ck


# ------------------------------------------------------------------------------------------------------- checks
def equiv(a, dev, K=20):
    """The host path against the GPU path at E = 2,048, from the same init and the same batches, for K steps.
    The two paths are the same maths up to float rounding: the kernel's per-row scale sums |w| in another order than
    torch's mean (up to ~2.7e-7 relative, kernel selftest), and AdamW runs on the CPU instead of CUDA. The yardstick
    is a null twin: the GPU path again, with every w_q scale multiplied by (1 + 2.4e-7), the same size of rounding.
    PASS: the host path stays as close to the GPU path as the null twin does (within 3x), and a planted negative (one
    host layer never updated) lands far outside. Resume: host and GPU state saved and loaded give the same step,
    bitwise."""
    data = Data4(a.data_dir, 0)
    res = []
    def check(name, val, tol):
        ok = bool(val <= tol); res.append(ok); print(f"  {'ok  ' if ok else 'FAIL'} {name:<84} {val:.3e}  (tol {tol:.3g})")
    E, steps = 2048, 1000
    xs = [data.batch(MB).to(dev) for _ in range(K + 1)]
    def traj(host, perturb=False, skip=None):
        m, o, hws = build_arm(dict(E=E, host=host), data.V, 0, dev)
        wq = SM.w_q
        if perturb:
            def wq2(w, dim=-1):
                q, sc = wq(w, dim); return q, sc * (1 + 2.4e-7)
            SM.w_q = wq2
        losses = []
        try:
            for i in range(K):
                losses.append(train_step(m, o, hws, xs[i], data.V, i, steps, skip_host_layer=skip)[0])
            m.updater.wait()
        finally:
            SM.w_q = wq
        return m, o, hws, np.array(losses)
    mg, og, _, lg = traj(False)
    _, _, _, ln = traj(False, perturb=True)
    mh, oh, hws, lh = traj(True)
    _, _, _, ls = traj(True, skip=3)
    d_null, d_host, d_neg = (float(np.abs(x - lg).max()) for x in (ln, lh, ls))
    print(f"  losses over {K} steps: GPU {lg[0]:.4f} -> {lg[-1]:.4f}; max |diff| null twin {d_null:.2e}, host {d_host:.2e}, planted {d_neg:.2e}")
    check("host path vs GPU path, max loss difference over the steps / (3 x the null twin's)", d_host / max(3 * d_null, 1e-12), 1)
    check("planted negative (one host layer never updated) far outside: 10 x host diff / planted diff", 10 * d_host / max(d_neg, 1e-12), 1)
    pg = dict(mg.named_parameters())
    core = max(float((pg[n] - q).abs().max()) for n, q in mh.named_parameters())
    bank = max(float((getattr(bg.mlp, n).detach().cpu() - hw.p[n]).abs().max()) for bg, hw in zip(mg.blocks, hws) for n in H4.NAMES)
    print(f"  after {K} steps, max weight difference host vs GPU: core and routers {core:.2e}, bank {bank:.2e} (lr {T.LR:.0e} at most per step)")
    # resume: save, one step, reload, the same step again: bitwise
    import copy
    st = [copy.deepcopy(hw.state()) for hw in hws]; ms = copy.deepcopy(mh.state_dict()); os_ = copy.deepcopy(oh.state_dict())
    l1 = train_step(mh, oh, hws, xs[K], data.V, K, steps)[0]; mh.updater.wait()
    after = [hw.p["Wg"].clone() for hw in hws]; core1 = [p.detach().clone() for p in mh.parameters()]
    for hw, s_ in zip(hws, st): hw.load_state(s_)
    mh.load_state_dict(ms); oh.load_state_dict(copy.deepcopy(os_))   # the optimizer aliases the loaded tensors
    l2 = train_step(mh, oh, hws, xs[K], data.V, K, steps)[0]; mh.updater.wait()
    # the pipelined update against the synchronous one: the same step, bitwise
    for hw, s_ in zip(hws, st): hw.load_state(s_)
    mh.load_state_dict(ms); oh.load_state_dict(copy.deepcopy(os_))   # the optimizer aliases the loaded tensors
    train_step(mh, oh, hws, xs[K], data.V, K, steps, pipelined=False)
    check("pipelined host update = synchronous one (max abs on every layer's Wg)",
          max(float((a_ - hw.p["Wg"]).abs().max()) for a_, hw in zip(after, hws)), 0)
    check("resume: the same step after reloading, loss (abs)", abs(l1 - l2), 0)
    check("resume: the same step after reloading, host Wg and GPU parameters (max abs)",
          max([float((a_ - hw.p["Wg"]).abs().max()) for a_, hw in zip(after, hws)] +
              [float((u - v).abs().max()) for u, v in zip(core1, mh.parameters())]), 0)
    print("EQUIV (host path = GPU path)", "PASS" if all(res) else "FAIL", f"({sum(res)}/{len(res)})")
    return all(res)


def pace_cell(a, dev, warm=20, n=200):
    """Gate G1's measurement: tokens per second of one arm over n steps after warm, real data and real shapes, no
    evaluation, no checkpoint. Before the registration it runs as a dry (estimate, file marked so)."""
    arm = ARMS[a.arm]; steps = TOKENS // (SEQ * MB)
    head, upstream = T.git("rev-parse", "HEAD"), T.git("rev-parse", "@{u}")
    dirty = T.git("status", "--porcelain", "--", "benchmarks/sparse_capacity", "docs/research/SPARSE_CAPACITY")
    official = REGISTERED and head == upstream and not dirty and n == 200 and warm == 20
    data = Data4(a.data_dir, a.seed)
    m, opt, hws = build_arm(arm, data.V, a.seed, dev)
    torch.manual_seed(9000 + a.seed); m.train()
    xs = [data.batch(MB) for _ in range(warm + n)]                           # batches made before the clock
    for i in range(warm + n):
        if i == warm:
            m.updater.wait(); torch.cuda.synchronize(); t0 = time.perf_counter()
        tot, gn, _ = train_step(m, opt, hws, xs[i].to(dev, non_blocking=True), data.V, i, steps, mb=arm["mb"])
        if not (math.isfinite(tot) and math.isfinite(gn)):
            sys.exit(f"STOP: non-finite at step {i}")
    m.updater.wait(); torch.cuda.synchronize(); sec = time.perf_counter() - t0
    out = dict(arm=a.arm, official=official, git_head=head, dirty=dirty.splitlines(), steps=n, sec=sec,
               step_s=sec / n, tok_s=n * SEQ * MB / sec, host=dict(m.updater.last), peak_gb=torch.cuda.max_memory_allocated() / 2**30,
               gpu=torch.cuda.get_device_name(0), time=time.time())
    path = os.path.join(a.out, f"f4_time_{a.arm}_s{a.seed}" + ("" if official else "_dry") + ".json")
    json.dump(out, open(path, "w"), indent=1); print("PACE", json.dumps(out)); print("->", path)
    return True


def selftest(dev):
    ok = M4.selftest(dev) and KA.selftest() and PP.selftest()
    print("SELFTEST (phase 4)", "PASS" if ok else "FAIL")
    return ok


# ------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--selftest", action="store_true"); g.add_argument("--dry", action="store_true")
    g.add_argument("--equiv", action="store_true"); g.add_argument("--run", action="store_true")
    g.add_argument("--time", action="store_true", help="the pace cell: 20 warm-up + 200 measured steps (machine free)")
    ap.add_argument("--arm", choices=list(ARMS), default="e2k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--dry-steps", type=int, default=30)
    ap.add_argument("--pace-n", type=int, default=200, help="measured steps of the pace cell (official only at 200)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "sparse_capacity"))
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if a.selftest:
        sys.exit(0 if selftest(dev) else 1)
    if a.equiv:
        sys.exit(0 if equiv(a, dev) else 1)
    if a.run and not REGISTERED:
        sys.exit("REFUSED: the protocol is not registered yet")
    if a.time:
        sys.exit(0 if pace_cell(a, dev, warm=20 if a.pace_n == 200 else 5, n=a.pace_n) else 1)
    arm = ARMS[a.arm]
    steps = TOKENS // (SEQ * MB)
    tag = f"f4_{a.arm}_s{a.seed}" + ("_dry" if a.dry else "")
    run_dir = os.path.join(a.out, tag); ckpt_dir = os.path.join(CKPT_ROOT, tag)
    os.makedirs(run_dir, exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "last.pt")
    head, upstream = T.git("rev-parse", "HEAD"), T.git("rev-parse", "@{u}")
    dirty = T.git("status", "--porcelain", "--", "benchmarks/sparse_capacity", "docs/research/SPARSE_CAPACITY")
    srcs = {f: T.sha256(os.path.join(HERE, f)) for f in ("sc2_model.py", "sc2_train.py", "sc4_model.py", "sc4_host.py",
                                                          "sc4_adamw.py", "sc4_train.py", "sc_proc.py", "sc_rt_train.py",
                                                          os.path.join("csrc", "adamw_tern.c"))}
    data = Data4(a.data_dir, a.seed)
    m, opt, hws = build_arm(arm, data.V, a.seed, dev)
    cnt = M4.counts(m)
    cfg = dict(mode="DRY (not a measurement)" if a.dry else "RUN", arm=a.arm, seed=a.seed, git_head=head,
               pushed=(head == upstream), dirty_files=dirty.splitlines(), src_sha256=srcs, registered=REGISTERED,
               python=sys.version.split()[0], torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
               cpu=platform.processor(), E=arm["E"], bank_on_host=arm["host"], params=cnt,
               host_bytes=sum(hw.nbytes()["host"] for hw in hws), gpu_bank_bytes=sum(hw.nbytes()["gpu"] for hw in hws),
               router="product keys, dk 128, n1 x n2 = %d x %d" % M4.pk_shape(arm["E"]), k=SM.ACTIVE // SM.H_EXP,
               micro_batch=arm["mb"], accum=MB // arm["mb"], tokens=steps * SEQ * MB, steps=steps, lr=T.LR, warmup=T.WARMUP,
               lr_floor=T.LR_FLOOR, schedule="phase 2 (linear warmup, cosine to the floor)", wd=T.WD, betas=T.BETAS,
               eps=KA.EPS, clip=T.CLIP, shares=dict(k=S_K, p=S_P, e=S_E), p_books=P_BOOKS, p_digits_spaced=True,
               tokens_per_expert=steps * SEQ * MB * (SM.ACTIVE // SM.H_EXP) / arm["E"],
               data_dir=a.data_dir, ts_sha256_first256MiB=data.ts_sha, threads=NTHR)
    for k, v in cfg.items():
        print(f"CONFIG {k} = {v}")
    refuse = []
    if dirty and not a.dry: refuse.append("uncommitted changes in the apparatus")
    if head != upstream and not a.dry: refuse.append("HEAD is not pushed")
    if "3060" not in cfg["gpu"]: refuse.append("not the RTX 3060")
    if PP.check_tokens(data.enc): refuse.append("P digits are not one token each with this BPE")
    if not a.dry and not selftest(dev): refuse.append("selftest failed")
    if refuse:
        sys.exit("REFUSED: " + "; ".join(refuse))
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1, default=str)
    if a.dry:
        steps_run = a.dry_steps
    else:
        steps_run = steps
    start, prev_sec = 0, 0.0
    torch.manual_seed(9000 + a.seed)
    if a.resume and os.path.exists(ckpt_path):
        ck = load_ckpt(ckpt_path, m, opt, hws, data, dev)
        if ck["cfg_key"] != [a.arm, a.seed, head]:
            sys.exit(f"REFUSED: checkpoint is {ck['cfg_key']}")
        start, prev_sec = ck["step"], ck["train_sec"]; print(f"RESUMED at step {start}")
    log = open(os.path.join(run_dir, "train.jsonl"), "a")
    T.emit(log, dict(kind="start", step=start, time=time.time(), resumed=bool(start)))
    val = data.ids[data.ntr:data.ntr + T.VAL_TOK + 1]; curve_val = val[:T.VAL_CURVE_TOK + 1]
    if start == 0 and not a.dry:
        T.emit(log, dict(kind="curve", step=0, bpb=T.eval_stream(m, curve_val, data.el, dev, arm["mb"])["bpb"], **readings(m, data, dev, False, arm["mb"])))
    t0 = time.time(); t_ck = time.time(); train_sec = lambda: prev_sec + (time.time() - t0)
    t_win, i_win, eval_win = time.time(), start, 0.0
    torch.cuda.reset_peak_memory_stats()
    m.train()
    for i in range(start, steps_run):
        xy = data.batch(MB).to(dev, non_blocking=True)
        timed = (i % TIME_EVERY == 0) or a.dry
        tot, gn, tm = train_step(m, opt, hws, xy, data.V, i, steps, timed, mb=arm["mb"])
        if not (math.isfinite(tot) and math.isfinite(gn)):
            T.emit(log, dict(kind="nonfinite", step=i, loss=tot, gnorm=gn)); sys.exit(f"STOP: non-finite at step {i}")
        if timed:
            us = [float(b.mlp.stats["used"]) for b in m.blocks]
            now = time.time()
            win = dict(steps=i - i_win, sec=now - t_win - eval_win,
                       tok_s=(i - i_win) * SEQ * MB / max(1e-9, now - t_win - eval_win)) if i > i_win else None
            t_win, i_win, eval_win = now, i, 0.0
            T.emit(log, dict(kind="step", step=i, loss=tot, gnorm=gn, lr=lr_at(i, steps), times=tm, host=dict(m.updater.last),
                             window=win, counts=dict(data.counts), used_min=min(us), peak_gb=torch.cuda.max_memory_allocated() / 2**30))
            print(f"step {i:6d}/{steps} loss {tot:.4f} gn {gn:.2f} " + " ".join(f"{k} {v:.3f}" for k, v in {**tm, **m.updater.last}.items())
                  + (f" tok/s {win['tok_s']:.0f}" if win else ""), flush=True)
        if not a.dry and ((i + 1) % CURVE_EVERY == 0 or i == steps - 1):
            te = time.time()
            T.emit(log, dict(kind="curve", step=i + 1, bpb=T.eval_stream(m, curve_val, data.el, dev, arm["mb"])["bpb"], **readings(m, data, dev, False, arm["mb"])))
            m.train(); eval_win += time.time() - te
        if not a.dry and (time.time() - t_ck > CKPT_MIN[arm["host"]] * 60 or i == steps - 1):
            te = time.time(); save_ckpt(ckpt_path, m, opt, hws, i + 1, data, [a.arm, a.seed, head], train_sec())
            t_ck = time.time(); eval_win += t_ck - te
    if a.dry:                                                                # one curve, one checkpoint round trip
        t1 = time.time(); bpb = T.eval_stream(m, curve_val, data.el, dev, arm["mb"])["bpb"]; rd = readings(m, data, dev, False, arm["mb"])
        t_curve = time.time() - t1
        t1 = time.time(); save_ckpt(ckpt_path, m, opt, hws, steps_run, data, [a.arm, a.seed, head], train_sec()); t_save = time.time() - t1
        before = [p.detach().clone() for p in m.parameters()] + [hw.p["Wd"].clone() for hw in hws]
        t1 = time.time(); load_ckpt(ckpt_path, m, opt, hws, data, dev); t_load = time.time() - t1
        after = [p.detach() for p in m.parameters()] + [hw.p["Wd"] for hw in hws]
        same = max(float((u.cpu() - v.cpu()).abs().max()) for u, v in zip(before, after))
        rec = dict(kind="dry_done", peak_gb=torch.cuda.max_memory_allocated() / 2**30, curve_s=t_curve, ckpt_save_s=t_save,
                   ckpt_load_s=t_load, ckpt_gb=os.path.getsize(ckpt_path) / 2**30, ckpt_roundtrip_maxabs=same, curve_bpb=bpb, readings=rd)
        T.emit(log, rec); print("DRY DONE", json.dumps({k: v for k, v in rec.items() if k != "readings"})); os.remove(ckpt_path); return
    res = dict(arm=a.arm, seed=a.seed, git_head=head, steps=steps, train_hours=train_sec() / 3600, counts=dict(data.counts))
    mb = arm["mb"]
    res["val"] = T.eval_stream(m, val, data.el, dev, mb, temps=T.TEMPS)
    if data.p62 is not None:
        res["p62"] = T.eval_stream(m, data.p62, data.el, dev, mb)
    res["usage"] = usage(m, val, dev, mb)
    res.update(readings(m, data, dev, True, mb))
    json.dump(res, open(os.path.join(run_dir, "result.json"), "w"), indent=1)
    T.emit(log, dict(kind="done", time=time.time()))
    print("RESULT", json.dumps(res)); print(f"DONE -> {run_dir}")


if __name__ == "__main__":
    main()
