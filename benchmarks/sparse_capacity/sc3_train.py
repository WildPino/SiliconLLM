#!/usr/bin/env python3
# Sparse Capacity, phase 3 (the method): phase 2's m16 s0 trained again with the bank in host RAM, streamed.
# Protocol: docs/research/SPARSE_CAPACITY/F3_STREAM_PROTOCOL.md. The verdict is computed by sc3_verdict.py.
#
#   python benchmarks/sparse_capacity/sc3_train.py --selftest
#   python benchmarks/sparse_capacity/sc3_train.py --dry --data-dir <m0 package>   (real shapes, real data, the trained
#                                        reference state: one step against the reference, planted negative, resume, pace)
#   python benchmarks/sparse_capacity/sc3_train.py --run --data-dir <m0 package> --ckpt-dir <dir> [--resume]
#
# Everything that is not the bank is phase 2's code (sc2_train: data, schedule, evaluation, probes), so the only
# difference from the reference run f2_m16_s0 is where the experts live and how they are updated.
import argparse, json, math, os, platform, sys, tempfile, time
import numpy as np
import torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sc2_model as SM      # noqa: E402
import sc2_train as T       # noqa: E402
import sc3_stream as ST     # noqa: E402

ROOT = T.ROOT
ARM, SEED = "m16", 0                     # the phase 2 arm this phase repeats (PLAN, phase 3)
REF_DIR = os.path.join(ROOT, "results", "sparse_capacity", "f2_m16_s0")
REF_CKPT = "E:/SiliconLLM_Artifacts/sparse_capacity/f2/f2_m16_s0/last.pt"    # the reference at step 7324, AdamW state
NTHR = 6                                 # host threads for the AdamW of the bank
STEP0_RTOL = 1e-6                        # planted: step-0 loss and grad norm equal to the reference's (review item 1)
EQ_RTOL = 1e-6                           # one step from the same state (selftest, dry)
TIME_EVERY = 50


def build(V, M, seed, dev, stream):
    torch.manual_seed(seed); np.random.seed(seed)
    m = SM.Student(V, M=M, quant=True)                                        # built on the host, as in phase 2
    counts = m.counts()
    if stream:
        m = ST.convert(m, dev)
        og = torch.optim.AdamW(ST.gpu_params(m), lr=T.LR, betas=T.BETAS, weight_decay=T.WD, fused=True)
        oh = torch.optim.AdamW(ST.host_params(m), lr=T.LR, betas=T.BETAS, weight_decay=T.WD, fused=True)
    else:
        m = m.to(dev)
        og = torch.optim.AdamW(m.parameters(), lr=T.LR, betas=T.BETAS, weight_decay=T.WD, fused=True); oh = None
    return m, og, oh, counts


def step(m, og, oh, xy, V, lr, stream, timed=False, skip_host=None):
    """One optimizer step on one micro-batch (accumulation 1: m16 runs micro-batch 16 = the global batch). Returns
    (loss, grad norm before the clip, timings)."""
    for o in (og, oh):
        if o is not None:
            for gp in o.param_groups: gp["lr"] = lr
    t = {}
    if timed:
        torch.cuda.synchronize(); t0 = time.perf_counter()
    if stream:
        for sb in ST.banks(m):
            sb.begin_step(); sb.timed = timed; sb.t = dict(fetch=0.0, grad_out=0.0, n_fetch=0)
    with torch.autocast(xy.device.type, dtype=torch.bfloat16):
        lg, aux = m(xy[:, :-1])
        loss = F.cross_entropy(lg.float().reshape(-1, V), xy[:, 1:].reshape(-1))
        full = loss + (aux if aux is not None else 0.0)
    full.backward()
    gn = float(ST.clip_(m, T.CLIP)) if stream else float(torch.nn.utils.clip_grad_norm_(m.parameters(), T.CLIP))
    if timed:
        torch.cuda.synchronize(); t1 = time.perf_counter()
    og.step(); og.zero_grad(set_to_none=True)
    if timed:
        torch.cuda.synchronize(); t2 = time.perf_counter()
    if oh is not None:
        if skip_host is not None:                                             # planted negative: one bank goes stale
            saved = [p.data.clone() for p in skip_host]
        oh.step()
        if skip_host is not None:
            for p, v in zip(skip_host, saved): p.data.copy_(v)
    if timed:
        t3 = time.perf_counter()
        t = dict(fwd_bwd=t1 - t0, gpu_adam=t2 - t1, host_adam=t3 - t2, step=t3 - t0)
        if stream:
            t["fetch"] = sum(sb.t["fetch"] for sb in ST.banks(m)); t["grad_out"] = sum(sb.t["grad_out"] for sb in ST.banks(m))
            sel = [v for sb in ST.banks(m) for v in sb.n_sel]
            t["n_sel_min"], t["n_sel_max"] = min(sel), max(sel)
            t["fetches_per_layer"] = ST.banks(m)[0].t["n_fetch"]
    return float(loss.detach()), gn, t


# ---------------------------------------------------------------------------------------------- state and checkpoints
def bank_tensors(m, stream):
    if stream:
        return [p.detach().cpu() for sb in ST.banks(m) for p in sb.host_params()]
    return [getattr(b.mlp, n).detach().cpu() for b in m.blocks for n in ST.NAMES]


def ref_bank_names(m):
    return [f"blocks.{i}.mlp.{n}" for i, b in enumerate(m.blocks) if isinstance(b.mlp, (SM.Bank, ST.StreamBank))
            for n in ST.NAMES]


def sync(mr, ogr, mq, ogq, ohq):
    """Copy the reference model and its AdamW state into the stream model (same tensors, same moments, same step)."""
    nr = dict(mr.named_parameters()); sr = ogr.state
    with torch.no_grad():
        for k, p in mq.named_parameters():
            p.copy_(nr[k])
            ogq.state[p] = {kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in sr[nr[k]].items()}
        for k, p in zip(ref_bank_names(mr), ST.host_params(mq)):
            p.data.copy_(nr[k].detach().cpu())
            ohq.state[p] = {kk: v.detach().to("cpu", copy=True) for kk, v in sr[nr[k]].items()}


def save_ckpt(path, m, og, oh, extra):
    torch.save(dict(model={k: v.cpu() for k, v in m.state_dict().items()}, host=[p.detach() for p in ST.host_params(m)],
                    og=og.state_dict(), oh=oh.state_dict(), **extra), path + ".tmp")
    os.replace(path + ".tmp", path)


def load_ckpt(path, m, og, oh, dev):
    st = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict({k: v.to(dev) for k, v in st["model"].items()})
    for p, v in zip(ST.host_params(m), st["host"]): p.data.copy_(v)
    og.load_state_dict(st["og"]); oh.load_state_dict(st["oh"])
    return st


def rel(a, b):
    return float((a - b).detach().abs().max() / b.detach().abs().max().clamp_min(1e-12))


def compare_from(mr, ogr, V, M, dev, test, lr, broken=False):
    """The method's step against the reference trainer's from the reference's current state: the model and its AdamW
    moments are copied into a stream model, then both take one step on `test`. Returns (relative loss difference,
    relative grad-norm difference, max relative difference of the bank after the step, of the rest; relative = max abs
    difference over max abs value, per tensor). The reference (mr, ogr) is advanced by one step.
    A multi-step trajectory is not a check: ternary rounding flips on 1e-7 differences and two right trajectories part
    after a few steps (protocol section 3)."""
    mq, ogq, ohq, _ = build(V, M, 0, dev, True)
    sync(mr, ogr, mq, ogq, ohq)
    skip = ST.banks(mq)[0].host_params() if broken else None
    lr_, gr_, _ = step(mr, ogr, None, test, V, lr, False)
    lq, gq, _ = step(mq, ogq, ohq, test, V, lr, True, skip_host=skip)
    br, bq = bank_tensors(mr, False), bank_tensors(mq, True)
    nr, nq = dict(mr.named_parameters()), dict(mq.named_parameters())
    dw = max(rel(a, b) for a, b in zip(bq, br))
    do = max(rel(nq[k], nr[k]) for k in nq)
    del mq, ogq, ohq; torch.cuda.empty_cache()
    return abs(lq - lr_) / abs(lr_), abs(gq - gr_) / abs(gr_), dw, do


def resume_check(m, og, oh, V, M, dev, batches, lr):
    """Take one step, save a checkpoint with the run's code, take another step (A); rebuild from scratch, load the
    checkpoint, take the same step (B). A and B must be equal: host weights, both optimizers and the model."""
    step(m, og, oh, batches[0], V, lr, True)
    path = os.path.join(tempfile.mkdtemp(), "ck.pt")
    save_ckpt(path, m, og, oh, dict(step=1))
    la = step(m, og, oh, batches[1], V, lr, True)[0]
    A = bank_tensors(m, True) + [p.detach().cpu() for p in m.parameters()]
    m2, og2, oh2, _ = build(V, M, 1234, dev, True)                            # other seed: the load must overwrite it
    load_ckpt(path, m2, og2, oh2, dev)
    lb = step(m2, og2, oh2, batches[1], V, lr, True)[0]
    B = bank_tensors(m2, True) + [p.detach().cpu() for p in m2.parameters()]
    os.remove(path)
    return abs(la - lb), max(float((a - b).abs().max()) for a, b in zip(A, B))


# ------------------------------------------------------------------------------------------------------------ selftest
def selftest(dev):
    res = []
    def check(name, val, tol):
        ok = bool(val <= tol); res.append(ok)
        print(f"  {'ok  ' if ok else 'FAIL'} {name:<70} {val:.3e}  (tol {tol:.0e})")
    torch.manual_seed(0)
    # 1. StreamBank against Bank, a selection smaller than E: forward and gradients (sent to the host in the backward)
    b = SM.Bank(256, E=48, h=8, k=16)
    sb = ST.StreamBank(b, dev); b = b.to(dev); sb.router = b.router
    x = torch.randn(1, 3, 256, device=dev)
    y_ref = b(x); (y_ref.sum() + b.aux).backward()
    sb.train(); sb.begin_step()
    y = sb(x); (y.sum() + sb.aux).backward()
    with torch.no_grad():
        U = torch.softmax(b.router(x.reshape(-1, 256)), -1).topk(16, -1)[1].unique().cpu()
    check("stream bank: |U| < E in this test (1 = no)", 0 if U.numel() < 48 else 1, 0)
    check("stream bank vs bank, forward (rel max)", rel(y, y_ref), 1e-6)
    check("stream bank vs bank, gradients on the host (rel max)",
          max(rel(sb.host[n].grad, getattr(b, n).grad.cpu()) for n in ST.NAMES), 1e-6)
    mask = torch.ones(48, dtype=torch.bool); mask[U] = False
    check("stream bank: host gradient outside U (abs sum)", float(sum(sb.host[n].grad[mask].abs().sum() for n in ST.NAMES)), 0)
    check("stream bank: weights fetched twice (forward, backward), not kept (count - 2)", abs(sb.t["n_fetch"] - 2), 0)
    # 2. activation checkpointing (recompute) gives the same gradients as no checkpointing
    def grads(use_ckpt):
        m, og, oh, _ = build(512, 2, 0, dev, True); m.use_ckpt = use_ckpt
        xi = torch.randint(0, 512, (2, 65), generator=torch.Generator().manual_seed(1)).to(dev)
        for sb_ in ST.banks(m): sb_.begin_step()
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            lg, aux = m(xi[:, :-1]); (F.cross_entropy(lg.float().reshape(-1, 512), xi[:, 1:].reshape(-1)) + aux).backward()
        return torch.cat([p.grad.flatten().cpu() for p in ST.host_params(m)] + [p.grad.flatten().cpu() for p in m.parameters()])
    check("checkpointing on vs off, all gradients (max abs)", float((grads(True) - grads(False)).abs().max()), 1e-6)
    # 3. one step from the same state (M = 2: E = 256 > k, so |U| < E happens), and the planted negative
    g = torch.Generator().manual_seed(2)
    batches = [torch.randint(0, 512, (2, 65), generator=g).to(dev) for _ in range(5)]
    def warm():
        mr, ogr, _, _ = build(512, 2, 0, dev, False)
        for xy in batches[:3]: step(mr, ogr, None, xy, 512, T.LR, False)
        return mr, ogr
    dl, dg, dw, do = compare_from(*warm(), 512, 2, dev, batches[3], T.LR)
    check("one step from the same state: loss (relative)", dl, EQ_RTOL)
    check("one step from the same state: grad norm before the clip (relative)", dg, EQ_RTOL)
    check("one step from the same state: bank (max abs diff / max abs, per tensor)", dw, EQ_RTOL)
    check("one step from the same state: other parameters (same, relative)", do, EQ_RTOL)
    _, _, dwb, _ = compare_from(*warm(), 512, 2, dev, batches[3], T.LR, broken=True)
    check("planted negative (one bank stale): bank relative difference above 1e-3 (1e-3/x)", 1e-3 / max(dwb, 1e-12), 1.0)
    # 4. resume: a checkpoint written and read by the run's code continues bit-equal
    m, og, oh, _ = build(512, 2, 0, dev, True)
    dlr, dpr = resume_check(m, og, oh, 512, 2, dev, batches[3:5], T.LR)
    check("resume: next step after save and load, loss (abs)", dlr, 0)
    check("resume: next step after save and load, all parameters (max abs)", dpr, 0)
    # 5. the final weights load into the phase-2 Student strictly and give the same logits
    ref = SM.Student(512, M=2).to(dev); ref.load_state_dict(ST.full_state_dict(m), strict=True)
    m.eval(); ref.eval()
    with torch.no_grad():
        a = m(batches[1][:, :-1])[0]; c = ref(batches[1][:, :-1])[0]
    check("full state dict into the phase-2 Student: logits (rel max)", rel(a, c), 1e-5)
    print("SELFTEST (phase 3)", "PASS" if all(res) else "FAIL", f"({sum(res)}/{len(res)})")
    return all(res)


# ------------------------------------------------------------------------------------------------------------ reference
def ref_cfg():
    return json.load(open(os.path.join(REF_DIR, "config.json")))


def ref_step0():
    for r in map(json.loads, open(os.path.join(REF_DIR, "train.jsonl"))):
        if r.get("kind") == "step" and r["step"] == 0:
            return r["loss"], r["gnorm"]
    return None, None


def identity_problems(data):
    """The run must use the reference's phase-2 code, torch and data (review item 7)."""
    rc = ref_cfg(); bad = []
    for f, h in rc["src_sha256"].items():
        if T.sha256(os.path.join(HERE, f)) != h: bad.append(f"{f} differs from the reference's")
    if torch.__version__ != rc["torch"]: bad.append(f"torch {torch.__version__} != {rc['torch']}")
    if data.ts_sha != rc["ts_sha256_first256MiB"]: bad.append("training data differ from the reference's")
    return bad


def main():
    ap = argparse.ArgumentParser()
    gx = ap.add_mutually_exclusive_group(required=True)
    gx.add_argument("--selftest", action="store_true"); gx.add_argument("--dry", action="store_true")
    gx.add_argument("--run", action="store_true")
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "sparse_capacity"))
    ap.add_argument("--ckpt-dir", default="")
    ap.add_argument("--ref-ckpt", default=REF_CKPT)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(NTHR)
    if a.selftest:
        sys.exit(0 if (selftest(dev) and T.selftest(dev)) else 1)
    M = T.ARMS[ARM]["M"]; mb = T.ARMS[ARM]["mb"]
    assert mb == T.GLOBAL_SEQS, "this trainer runs accumulation 1 only"
    if a.dry:
        dry(a, dev, M, mb); return

    steps = T.TOKENS // (T.SEQ * T.GLOBAL_SEQS)
    tag = f"f3_stream_{ARM}_s{SEED}"
    run_dir = os.path.join(a.out, tag)
    ckpt_dir = a.ckpt_dir or os.path.join(run_dir, "ckpt")
    os.makedirs(run_dir, exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "last.pt")
    head, upstream = T.git("rev-parse", "HEAD"), T.git("rev-parse", "@{u}")
    dirty = T.git("status", "--porcelain", "--", "benchmarks/sparse_capacity", "docs/research/SPARSE_CAPACITY",
                  "benchmarks/phase55", "benchmarks/phase62")
    srcs = {f: T.sha256(os.path.join(HERE, f)) for f in ("sc2_model.py", "sc2_probes.py", "sc2_train.py",
                                                          "sc3_stream.py", "sc3_train.py")}
    data = T.Data(a.data_dir, SEED)
    m, og, oh, counts = build(data.V, M, SEED, dev, True)
    l0, g0 = ref_step0()
    cfg = dict(mode="RUN", arm=ARM, seed=SEED, method="bank in host RAM (fp32 master, gradient, AdamW state), streamed; "
               "re-fetched in the backward", git_head=head, pushed=(head == upstream), dirty_files=dirty.splitlines(),
               src_sha256=srcs, python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0), cpu=platform.processor(), host_threads=NTHR, params=counts,
               host_bank_params=sum(p.numel() for p in ST.host_params(m)),
               pinned_stage_params=sum(t.numel() for t in ST.banks(m)[0].stage.buf.values()), steps=steps,
               micro_batch=mb, accum=1, tokens=steps * T.SEQ * T.GLOBAL_SEQS, lr=T.LR, warmup=T.WARMUP,
               lr_floor=T.LR_FLOOR, wd=T.WD, betas=T.BETAS, clip=T.CLIP, host_optimizer="torch AdamW fused (CPU)",
               gpu_optimizer="torch AdamW fused", tf32=torch.backends.cuda.matmul.allow_tf32, data_dir=a.data_dir,
               ts_sha256_first256MiB=data.ts_sha, reference=REF_DIR, reference_step0=dict(loss=l0, gnorm=g0))
    for k, v in cfg.items():
        print(f"CONFIG {k} = {v}")
    refuse = identity_problems(data)
    if dirty: refuse.append("uncommitted changes in the apparatus")
    if head != upstream: refuse.append("HEAD is not pushed (protocol must be pushed before the run)")
    if "3060" not in cfg["gpu"]: refuse.append("not the RTX 3060 of the protocol")
    if l0 is None: refuse.append("reference run f2_m16_s0 not found")
    if not (selftest(dev) and T.selftest(dev)): refuse.append("selftest failed")
    if refuse:
        sys.exit("REFUSED: " + "; ".join(refuse))
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1, default=str)

    start = 0; prev_sec, prev_tok = 0.0, 0
    if a.resume and os.path.exists(ckpt_path):
        st = load_ckpt(ckpt_path, m, og, oh, dev)
        if st["cfg_key"] != [ARM, SEED, head]:
            sys.exit(f"REFUSED: checkpoint is {st['cfg_key']}, this run is {[ARM, SEED, head]}")
        start = st["step"]; prev_sec, prev_tok = st["train_sec"], st["tok_done"]
        data.rng.setstate(st["data"]["rng"]); data.nrng.bit_generator.state = st["data"]["nrng"]
        data.skip_r(st["data"]["r_count"])
        torch.set_rng_state(st["torch_rng"]); torch.cuda.set_rng_state(st["cuda_rng"])
        print(f"RESUMED at step {start}")
    log = open(os.path.join(run_dir, "train.jsonl"), "a")
    T.emit(log, dict(kind="start", step=start, time=time.time(), resumed=bool(start)))
    val = data.ids[data.ntr:data.ntr + T.VAL_TOK + 1]
    curve_val = val[:T.VAL_CURVE_TOK + 1]
    t_last_ck = time.time(); t0 = time.time(); tok_done = 0
    train_sec = lambda: prev_sec + (time.time() - t0)
    m.train()
    for i in range(start, steps):
        xy, kinds = data.batch(mb)
        xy = xy.to(dev, non_blocking=True)
        timed = (i % TIME_EVERY == 0)
        loss, gn, tm = step(m, og, oh, xy, data.V, T.lr_at(i, steps), True, timed=timed)
        if not (math.isfinite(loss) and math.isfinite(gn)):
            T.emit(log, dict(kind="nonfinite", step=i, loss=loss, gnorm=gn))
            sys.exit(f"STOP: non-finite loss or gradient at step {i} (PLAN section 3)")
        if i == 0 and start == 0:
            dl, dg = abs(loss - l0) / abs(l0), abs(gn - g0) / abs(g0)
            ok = dl <= STEP0_RTOL and dg <= STEP0_RTOL
            T.emit(log, dict(kind="planted_step0", loss=loss, gnorm=gn, ref_loss=l0, ref_gnorm=g0, rel_loss=dl,
                             rel_gnorm=dg, rtol=STEP0_RTOL, ok=ok))
            print(f"PLANTED step 0: loss {loss:.7f} vs {l0:.7f}, grad norm {gn:.7f} vs {g0:.7f}: {'ok' if ok else 'FAIL'}",
                  flush=True)
            if not ok:
                sys.exit("STOP: the first step is not the reference's (VOID, protocol section 5)")
        tok_done += T.SEQ * T.GLOBAL_SEQS
        if timed or i == steps - 1:
            rec = dict(kind="step", step=i, loss=loss, gnorm=gn, lr=T.lr_at(i, steps),
                       tok_s=(prev_tok + tok_done) / train_sec(), times=tm,
                       load=float(np.mean([float(b.mlp.stats["load"]) for b in m.blocks])),
                       used_batch=float(min(float(b.mlp.stats["used"]) for b in m.blocks)))
            T.emit(log, rec)
            print(f"step {i:6d}/{steps} loss {loss:.4f} gn {gn:.2f} {rec['tok_s']:.0f} tok/s", flush=True)
        if (i + 1) % T.CURVE_EVERY == 0 or i == steps - 1:
            ev = T.eval_stream(m, curve_val, data.el, dev, mb)
            T.emit(log, dict(kind="curve", step=i + 1, tokens=(i + 1) * T.SEQ * T.GLOBAL_SEQS, bpb=ev["bpb"], top1=ev["top1"]))
        if time.time() - t_last_ck > T.CKPT_MIN * 60 or i == steps - 1:
            save_ckpt(ckpt_path, m, og, oh, dict(step=i + 1, cfg_key=[ARM, SEED, head], data=data.state(),
                                                 torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                                                 train_sec=train_sec(), tok_done=prev_tok + tok_done))
            t_last_ck = time.time()

    res = dict(arm=ARM, seed=SEED, method="stream", git_head=head, mode="RUN", params=counts, steps=steps,
               tokens=steps * T.SEQ * T.GLOBAL_SEQS, train_hours=train_sec() / 3600)
    res["val"] = T.eval_stream(m, val, data.el, dev, mb, temps=T.TEMPS)
    if data.p62 is not None:
        res["p62"] = T.eval_stream(m, data.p62, data.el, dev, mb)
    res["probes"] = T.eval_probes(m, data, dev, mb)
    res["experts"] = T.expert_usage(m, val, dev, mb, 262_144)
    json.dump(res, open(os.path.join(run_dir, "result.json"), "w"), indent=1)
    wpath = os.path.join(ckpt_dir, "final_model.pt")
    torch.save(ST.full_state_dict(m), wpath)
    T.emit(log, dict(kind="done", time=time.time(), final_weights_sha256=T.sha256(wpath)))
    print("RESULT", json.dumps(res))
    print(f"DONE -> {run_dir}\nverdict: python benchmarks/sparse_capacity/sc3_verdict.py")


def dry(a, dev, M, mb):
    """Real shapes, real data, the trained reference state (review item 2). Saves its numbers to
    results/sparse_capacity/f3_dry_<stamp>.json (committed: the protocol cites them)."""
    data = T.Data(a.data_dir, SEED)
    out = dict(ref_ckpt=a.ref_ckpt, git_head=T.git("rev-parse", "HEAD"), identity=identity_problems(data))
    batches = [data.batch(mb)[0].to(dev) for _ in range(8)]
    lr = T.lr_at(7000, 7324)
    def ref_from_ckpt():
        mr, ogr, _, _ = build(data.V, M, SEED, dev, False)
        st = torch.load(a.ref_ckpt, map_location=dev, weights_only=False)
        mr.load_state_dict(st["model"]); ogr.load_state_dict(st["opt"])
        return mr, ogr
    mr, ogr = ref_from_ckpt()
    eq = []
    for xy in batches[:3]:                                                    # three steps, each from the reference's state
        eq.append(compare_from(mr, ogr, data.V, M, dev, xy, lr))
    out["one_step"] = [dict(loss=r[0], gnorm=r[1], bank=r[2], other=r[3]) for r in eq]
    print("DRY one step from the trained reference state (relative diffs: loss, grad norm, bank, rest):",
          [[f"{v:.1e}" for v in r] for r in eq])
    neg = compare_from(mr, ogr, data.V, M, dev, batches[3], lr, broken=True)
    out["planted_negative"] = dict(loss=neg[0], bank=neg[2])
    print(f"DRY planted negative (bank 0 stale): loss {neg[0]:.1e}, bank {neg[2]:.1e}")
    mq, ogq, ohq, counts = build(data.V, M, SEED, dev, True)
    sync(mr, ogr, mq, ogq, ohq)
    del mr, ogr; torch.cuda.empty_cache()
    dlr, dpr = resume_check(mq, ogq, ohq, data.V, M, dev, batches[4:6], lr)
    out["resume"] = dict(loss=dlr, params=dpr)
    print(f"DRY resume after save and load: loss diff {dlr:.1e}, parameter diff {dpr:.1e}")
    ts = [step(mq, ogq, ohq, xy, data.V, lr, True, timed=True)[2] for xy in batches]
    med = {k: float(np.median([t[k] for t in ts[2:]])) for k in ts[-1]}
    out["timings_median_s"] = med; out["tok_s_no_eval"] = T.SEQ * T.GLOBAL_SEQS / med["step"]
    out["host_bank_params"] = sum(p.numel() for p in ST.host_params(mq)); out["counts"] = counts
    out["vram_max_alloc_GB"] = torch.cuda.max_memory_allocated() / 2**30
    print("DRY stream step timings (median, s):", {k: round(v, 4) for k, v in med.items()},
          f"-> {out['tok_s_no_eval']:.0f} tok/s without evaluation; identity problems: {out['identity']}")
    path = os.path.join(a.out, "f3_dry_" + time.strftime("%Y%m%d_%H%M%S") + ".json")
    json.dump(out, open(path, "w"), indent=1, default=str)
    print("DRY ->", path)


if __name__ == "__main__":
    main()
