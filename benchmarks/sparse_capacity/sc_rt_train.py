#!/usr/bin/env python3
# Sparse Capacity, the trained R probes (D-R): continued training of the ten phase-2 models with two new probes in the
# stream. Protocol: docs/research/SPARSE_CAPACITY/RT_PROC_PROTOCOL.md.
#
#   P  "rubrica di procedure" (sc_proc.py): many named rules on digit strings, tested on inputs never seen with them
#   T  variable binding (sc_trace.py, trained version): a reading
#   K  phase 2's phone book, kept in the stream: knowledge, read beside P
#
# Stream: 76% code, 4% K, 10% P, 10% T. 15 M tokens from the phase-2 state at step 7324 (weights and AdamW moments),
# constant lr 3e-4 after 100 warmup steps. The same batches for every arm with the same seed.
#
#   python benchmarks/sparse_capacity/sc_rt_train.py --selftest
#   python benchmarks/sparse_capacity/sc_rt_train.py --tune --seed {0,1} --data-dir <m0>   (m1 only: the calibration)
#   python benchmarks/sparse_capacity/sc_rt_train.py --run --arm <arm> --seed <s> --data-dir <m0>   (after registration)
import argparse, json, math, os, platform, random, sys, time
import numpy as np
import torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sc2_model as SM      # noqa: E402
import sc2_train as T       # noqa: E402
import sc_proc as PP        # noqa: E402
import sc_trace as TR       # noqa: E402

ROOT = T.ROOT
F2_CKPT = "E:/SiliconLLM_Artifacts/sparse_capacity/f2"
RT_CKPT = "E:/SiliconLLM_Artifacts/sparse_capacity/rt"
REGISTERED = False            # set to True only in the commit that registers the calibration (protocol section 5)
TOKENS = 15_000_000
LR, WARMUP = 3e-4, 100
S_K, S_P, S_T = 0.04, 0.10, 0.10
CURVE_EVERY = 250             # ~2 M tokens: probe curves (P on the small books, T, BPB on the curve val)
P_EVAL = (256, 2048, 2048)    # functions scored per book (all, all, a fixed sample of 16384), P_TEST_PER_FN inputs each
P_CURVE = (256, 512, 0)
SEQ = T.SEQ


class RtData:
    """Phase 2's data (code windows, K books, BPE) plus the P and T streams; one mixing RNG per seed, the same for every
    arm."""
    def __init__(s, data_dir, seed):
        s.d = T.Data(data_dir, 5000 + seed)
        s.V = s.d.V; s.el = s.d.el; s.enc = s.d.enc; s.ids = s.d.ids; s.ntr = s.d.ntr; s.p62 = s.d.p62
        s.ts_sha = s.d.ts_sha
        knames = {n for b in s.d.books for n in b} | {n for n, _ in s.d.unseen}
        s.pbooks, s.punseen, s.ptest = PP.make_pbooks(forbidden=knames)
        s.pool = TR.single_token_pool(s.enc)
        s.ttest, tkeys = TR.t_test_set(s.pool)
        s.seed = seed
        s.pgen = PP.p_train_stream(s.pbooks, s.ptest, 6000 + seed)
        s.tgen = TR.t_train_stream(s.pool, tkeys, 7000 + seed)
        s.pfill = PP.p_train_stream(s.pbooks, s.ptest, 99)                 # context before a P test item
        s.tfill = TR.t_train_stream(s.pool, tkeys, 99)
        s.rng = random.Random(8000 + seed); s.counts = dict(c=0, k=0, p=0, t=0); s.pn = s.tn = 0

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
            elif u < S_K + S_P + S_T: rows.append(s._lines(s.tgen, "tn")); s.counts["t"] += 1
            else: rows.append(s.d.code_seq()); s.counts["c"] += 1
        return torch.from_numpy(np.stack(rows))

    def state(s):
        return dict(rng=s.rng.getstate(), drng=s.d.rng.getstate(), nrng=s.d.nrng.bit_generator.state, pn=s.pn, tn=s.tn,
                    counts=dict(s.counts))

    def restore(s, st):
        s.rng.setstate(st["rng"]); s.d.rng.setstate(st["drng"]); s.d.nrng.bit_generator.state = st["nrng"]
        for _ in range(st["pn"]): next(s.pgen)
        for _ in range(st["tn"]): next(s.tgen)
        s.pn, s.tn, s.counts = st["pn"], st["tn"], dict(st["counts"])


# ----------------------------------------------------------------------------------------------------------- probes
@torch.no_grad()
def score_tokens(model, items, dev, mb=32):
    """Fraction of answer tokens predicted right (teacher-forced argmax), per item: a partial-credit reading of P."""
    model.eval(); out = []
    for i in range(0, len(items), mb):
        ch = items[i:i + mb]; rows, spans = [], []
        for ctx, ans in ch:
            seq = (ctx + ans)[-(SEQ + 1):]; assert len(seq) == SEQ + 1
            rows.append(seq); spans.append((SEQ + 1 - len(ans), len(ans)))
        xy = torch.tensor(rows, device=dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            lg, _ = model(xy[:, :-1])
        pred = lg.argmax(-1)
        for j, (a0, na) in enumerate(spans):
            out.append(float((pred[j, a0 - 1:a0 - 1 + na] == xy[j, a0:a0 + na]).float().mean()))
    model.train()
    return out



def fill(data, gen, need, avoid=None):
    ctx = []
    while len(ctx) < need:
        line = next(gen)
        if avoid and avoid in line:
            continue
        ctx += data.enc(line)
    return ctx


def eval_p(model, data, dev, mb, per_book=P_EVAL):
    """P per book: exact match of the answer on held-out inputs; the context never shows the tested function. Controls:
    unseen functions (model), copy of the input, the book's most common rule (no model)."""
    res = {}; allok = []
    for bi, (size, n) in enumerate(zip(PP.P_SIZES, per_book)):
        if n == 0:
            continue
        b = data.pbooks[bi]; names = list(b)
        names = names if n >= size else random.Random(bi).sample(names, n)
        items, copy, mcr = [], [], []
        rule0, share = PP.most_common_rule(b)
        for nm in names:
            for x in sorted(data.ptest[nm]):
                y = PP.apply_rule(b[nm], x)
                items.append((fill(data, data.pfill, SEQ + 1, f"f_{nm}(") + data.enc(PP.p_prefix(nm, x)), data.enc(PP.p_answer(y))))
                copy.append(x == y); mcr.append(PP.apply_rule(rule0, x) == y)
        ok = T.score_items(model, items, dev, mb)
        res[str(size)] = dict(acc=float(np.mean(ok)), n=len(ok), ctl_copy=float(np.mean(copy)), ctl_common_rule=float(np.mean(mcr)),
                              common_rule_share=share, tok_acc=float(np.mean(score_tokens(model, items, dev, mb))))
        allok.append(float(np.mean(ok)))
    if per_book is P_EVAL:
        items = []
        for nm, rule in data.punseen.items():
            for x in sorted(data.ptest[nm]):
                items.append((fill(data, data.pfill, SEQ + 1) + data.enc(PP.p_prefix(nm, x)), data.enc(PP.p_answer(PP.apply_rule(rule, x)))))
        res["unseen"] = float(np.mean(T.score_items(model, items, dev, mb)))
        res["unseen_tok_acc"] = float(np.mean(score_tokens(model, items, dev, mb)))
        res["primary"] = float(np.mean(allok))
    return res


def eval_t(model, data, dev, mb, limit=None):
    """T (reading): greedy exact match of the literal (and newline), per cell, with the no-model controls."""
    tests = data.ttest if limit is None else data.ttest[::max(1, len(data.ttest) // limit)]
    items, cells, ctl = [], [], []
    crng = random.Random(3)
    for p in tests:
        # only the literal: in the stream it is followed by the blank line, which the BPE merges into one "

" token
        items.append((fill(data, data.tfill, SEQ + 1) + data.enc(TR.program_text(p, "assert")), data.enc(TR.answer_text(p, "assert"))[:1]))
        cells.append(f"m{p['m']}_d{p['d']}" + ("_rb" if p["rebind"] else ""))
        ctl.append({k: f(p, crng) == p["ans"] for k, f in TR.CONTROLS.items()})
    ok = T.score_items(model, items, dev, mb)
    out = {}
    for c in sorted(set(cells)):
        ix = [i for i, x in enumerate(cells) if x == c]
        r = dict(acc=float(np.mean([ok[i] for i in ix])), n=len(ix))
        triv = [k for k in TR.CONTROLS if k != "final_binding" or c.endswith("_rb")]
        r.update({"ctl_" + k: float(np.mean([ctl[i][k] for i in ix])) for k in triv})
        out[c] = r
    return out


def eval_k(model, data, dev, mb):
    rng = random.Random(4242); res = {}
    for bi, size in enumerate((256, 2048, 16384)):
        n = min(T.K_EVAL[bi], size)
        entries = list(range(size)) if n == size else random.Random(bi).sample(range(size), n)
        ok = T.score_items(model, T.k_items(data.d, rng, bi, entries), dev, mb)
        res[str(size)] = float(np.mean(ok))
    res["primary"] = float(np.mean(list(res.values())))
    return res


def lr_at(i):
    return LR * min(1.0, (i + 1) / WARMUP)


# ------------------------------------------------------------------------------------------------------------ main
def selftest(dev):
    ok = PP.selftest() and TR.selftest() and T.selftest(dev)
    print("SELFTEST (continuation)", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--selftest", action="store_true"); g.add_argument("--tune", action="store_true")
    g.add_argument("--run", action="store_true"); g.add_argument("--dry", action="store_true")
    g.add_argument("--rescore", action="store_true", help="probes again on the final weights of a finished run")
    ap.add_argument("--arm", choices=list(T.ARMS), default="m1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "sparse_capacity"))
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if a.selftest:
        sys.exit(0 if selftest(dev) else 1)
    if a.rescore:
        rescore(a, dev); return
    if (a.tune or a.dry) and a.arm != "m1":
        sys.exit("REFUSED: the calibration looks at m1 only (user rule)")
    if a.run and not REGISTERED:
        sys.exit("REFUSED: the calibration is not registered yet (protocol section 5)")
    global CURVE_EVERY, P_EVAL, P_CURVE
    arm = T.ARMS[a.arm]; accum = T.GLOBAL_SEQS // arm["mb"]
    steps = TOKENS // (SEQ * T.GLOBAL_SEQS)
    tag = f"rt_{a.arm}_s{a.seed}"
    if a.dry:                                     # smoke test of the whole path: not a measurement, never committed
        steps, CURVE_EVERY, P_EVAL, P_CURVE = 8, 4, (16, 16, 16), (16, 16, 0)
        T.VAL_TOK, T.VAL_CURVE_TOK, T.K_EVAL = 16_384, 8_192, (16, 16, 16)
        TR.T_PER_CELL = 4
        tag = "rt_dry_" + tag[3:]
    run_dir = os.path.join(a.out, tag); ckpt_dir = os.path.join(RT_CKPT, tag)
    os.makedirs(run_dir, exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "last.pt")
    head, upstream = T.git("rev-parse", "HEAD"), T.git("rev-parse", "@{u}")
    dirty = T.git("status", "--porcelain", "--", "benchmarks/sparse_capacity", "docs/research/SPARSE_CAPACITY")
    srcs = {f: T.sha256(os.path.join(HERE, f)) for f in ("sc2_model.py", "sc2_probes.py", "sc2_train.py", "sc_proc.py",
                                                          "sc_trace.py", "sc_rt_train.py")}
    data = RtData(a.data_dir, a.seed)
    model = SM.Student(data.V, M=arm["M"], quant=arm["quant"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=T.BETAS, weight_decay=T.WD, fused=True)
    f2 = os.path.join(F2_CKPT, f"f2_{a.arm}_s{a.seed}", "last.pt")
    st = torch.load(f2, map_location=dev, weights_only=False)
    if st["step"] != 7324 or st["cfg_key"][:2] != [a.arm, a.seed]:
        sys.exit(f"REFUSED: {f2} is {st['cfg_key']} at step {st['step']}")
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); del st
    line_tok = float(np.mean([len(data.enc(PP.p_line(n, r, "3812"))) for n, r in list(data.pbooks[0].items())[:500]]))
    p_lines = TOKENS * S_P / line_tok
    cfg = dict(mode="DRY (not a measurement)" if a.dry else "TUNE" if a.tune else "RUN", arm=a.arm, seed=a.seed, git_head=head, pushed=(head == upstream),
               dirty_files=dirty.splitlines(), src_sha256=srcs, registered=REGISTERED, start=f2, python=sys.version.split()[0],
               torch=torch.__version__, gpu=torch.cuda.get_device_name(0), cpu=platform.processor(), M=arm["M"],
               micro_batch=arm["mb"], accum=accum, tokens=steps * SEQ * T.GLOBAL_SEQS, steps=steps, lr=LR, warmup=WARMUP,
               schedule="linear warmup then constant", wd=T.WD, betas=T.BETAS, clip=T.CLIP, shares=dict(k=S_K, p=S_P, t=S_T),
               p_sizes=PP.P_SIZES, p_ops=len(PP.OPS), p_tokens_per_line=line_tok,
               p_exposures_per_function={str(n): p_lines / 3 / n for n in PP.P_SIZES}, t_cells=TR.T_CELLS, t_ood=TR.T_OOD,
               pool=data.pool, data_dir=a.data_dir, ts_sha256_first256MiB=data.ts_sha)
    for k, v in cfg.items():
        print(f"CONFIG {k} = {v}")
    refuse = []
    if dirty and not a.dry: refuse.append("uncommitted changes in the apparatus")
    if head != upstream and not a.dry: refuse.append("HEAD is not pushed")
    if "3060" not in cfg["gpu"]: refuse.append("not the RTX 3060")
    if not selftest(dev): refuse.append("selftest failed")
    if refuse:
        sys.exit("REFUSED: " + "; ".join(refuse))
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1, default=str)
    start = 0; prev_sec = 0.0
    torch.manual_seed(9000 + a.seed)
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        if ck["cfg_key"] != [a.arm, a.seed, head]:
            sys.exit(f"REFUSED: checkpoint is {ck['cfg_key']}")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start = ck["step"]; prev_sec = ck["train_sec"]
        data.restore(ck["data"]); torch.set_rng_state(ck["torch_rng"]); torch.cuda.set_rng_state(ck["cuda_rng"])
        print(f"RESUMED at step {start}")
    log = open(os.path.join(run_dir, "train.jsonl"), "a")
    T.emit(log, dict(kind="start", step=start, time=time.time(), resumed=bool(start)))
    val = data.ids[data.ntr:data.ntr + T.VAL_TOK + 1]; curve_val = val[:T.VAL_CURVE_TOK + 1]
    if start == 0:                                                           # the starting point, same code as the curves
        T.emit(log, dict(kind="curve", step=0, bpb=T.eval_stream(model, curve_val, data.el, dev, arm["mb"])["bpb"],
                         p=eval_p(model, data, dev, arm["mb"], P_CURVE), t=eval_t(model, data, dev, arm["mb"], 600)))
    t0 = time.time(); t_ck = time.time()
    train_sec = lambda: prev_sec + (time.time() - t0)
    model.train()
    for i in range(start, steps):
        for gp in opt.param_groups: gp["lr"] = lr_at(i)
        tot = 0.0
        for _ in range(accum):
            xy = data.batch(arm["mb"]).to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                lg, aux = model(xy[:, :-1])
                loss = F.cross_entropy(lg.float().reshape(-1, data.V), xy[:, 1:].reshape(-1))
                full = loss + (aux if aux is not None else 0.0)
            (full / accum).backward(); tot += float(loss.detach()) / accum
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), T.CLIP))
        if not (math.isfinite(tot) and math.isfinite(gn)):
            T.emit(log, dict(kind="nonfinite", step=i, loss=tot, gnorm=gn)); sys.exit(f"STOP: non-finite at step {i}")
        opt.step(); opt.zero_grad(set_to_none=True)
        if i % 50 == 0 or i == steps - 1:
            T.emit(log, dict(kind="step", step=i, loss=tot, gnorm=gn, lr=lr_at(i), counts=dict(data.counts),
                             tok_s=(i + 1 - start) * SEQ * T.GLOBAL_SEQS / max(1e-9, time.time() - t0)))
            print(f"step {i:5d}/{steps} loss {tot:.4f} gn {gn:.2f}", flush=True)
        if (i + 1) % CURVE_EVERY == 0 or i == steps - 1:
            T.emit(log, dict(kind="curve", step=i + 1, bpb=T.eval_stream(model, curve_val, data.el, dev, arm["mb"])["bpb"],
                             p=eval_p(model, data, dev, arm["mb"], P_CURVE), t=eval_t(model, data, dev, arm["mb"], 600)))
            model.train()
        if time.time() - t_ck > T.CKPT_MIN * 60 or i == steps - 1:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=i + 1, cfg_key=[a.arm, a.seed, head],
                            data=data.state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                            train_sec=train_sec()), ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path); t_ck = time.time()
    res = dict(arm=a.arm, seed=a.seed, mode=cfg["mode"], git_head=head, steps=steps, train_hours=train_sec() / 3600,
               counts=dict(data.counts))
    res["val"] = T.eval_stream(model, val, data.el, dev, arm["mb"], temps=T.TEMPS)
    if data.p62 is not None:
        res["p62"] = T.eval_stream(model, data.p62, data.el, dev, arm["mb"])
    res["k"] = eval_k(model, data, dev, arm["mb"])
    res["p"] = eval_p(model, data, dev, arm["mb"])
    res["t"] = eval_t(model, data, dev, arm["mb"])
    json.dump(res, open(os.path.join(run_dir, "result.json"), "w"), indent=1)
    wpath = os.path.join(ckpt_dir, "final_model.pt"); torch.save(model.state_dict(), wpath)
    T.emit(log, dict(kind="done", time=time.time(), final_weights_sha256=T.sha256(wpath)))
    print("RESULT", json.dumps(res)); print(f"DONE -> {run_dir}")


def rescore(a, dev):
    """The probes on the final weights of a finished run, with the code at this commit (written to result_rescore.json;
    result.json stays as the run wrote it). m1 only until the registration, as the calibration."""
    if a.arm != "m1" and not REGISTERED:
        sys.exit("REFUSED: the calibration looks at m1 only (user rule)")
    tag = f"rt_{a.arm}_s{a.seed}"; run_dir = os.path.join(a.out, tag)
    wpath = os.path.join(RT_CKPT, tag, "final_model.pt")
    done = [json.loads(l) for l in open(os.path.join(run_dir, "train.jsonl")) if '"done"' in l][-1]
    if T.sha256(wpath) != done["final_weights_sha256"]:
        sys.exit("REFUSED: final weights do not match the run's log")
    arm = T.ARMS[a.arm]; data = RtData(a.data_dir, a.seed)
    model = SM.Student(data.V, M=arm["M"], quant=arm["quant"]).to(dev)
    model.load_state_dict(torch.load(wpath, map_location=dev, weights_only=True)); model.eval()
    res = dict(arm=a.arm, seed=a.seed, git_head=T.git("rev-parse", "HEAD"), weights_sha256=done["final_weights_sha256"],
               p=eval_p(model, data, dev, arm["mb"]), t=eval_t(model, data, dev, arm["mb"]))
    json.dump(res, open(os.path.join(run_dir, "result_rescore.json"), "w"), indent=1)
    print("RESCORE", json.dumps(res))


if __name__ == "__main__":
    main()
