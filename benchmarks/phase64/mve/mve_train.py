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
    def __init__(s, tag, seg_row, resident=2, delete_behind=False, logits_dir=None):
        # logits_dir lets the 17.976 GB of teacher logits live in their OWN shared dataset rather than be
        # copied into every arm's package -- three alpha arms read the SAME logits, so a per-package copy
        # would triple 18 GB to 54 GB for no reason. Defaults to DATA (the --data-dir) so nothing that
        # shipped logits inside the package breaks.
        s.dir = os.path.join(logits_dir or DATA, f"logits_{tag}")
        s.man = json.load(open(os.path.join(s.dir, "manifest.json")))
        s.chunks = s.man["chunks"]; s.K = s.man["K"]
        s.resident = resident; s.delete_behind = delete_behind
        s.pos = 0; s.cache = {}
        # KEY for the window search = seg_row (the teacher row GOVERNING each student position, forward-filled).
        # It is non-decreasing, which is what searchsorted requires. The raw `anchors` array is NOT a legal key:
        # 56% of its entries are -1 and, however they are remapped, they sit INTERSPERSED between the increasing
        # valid values -> the array is unsorted -> searchsorted returns garbage and the sampling window silently
        # stops matching the resident chunks. (That bug cost the first MVE run its KD coverage.)
        s.key = seg_row
        assert np.all(np.diff(s.key) >= 0), "window key must be non-decreasing"
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
        # student and teacher tokenize the same byte stream in order -> the positions governed by teacher rows
        # [j0,j1) form a CONTIGUOUS range, found by bisecting the non-decreasing key.
        p_lo = int(np.searchsorted(s.key, j0, "left"))
        p_hi = int(np.searchsorted(s.key, j1, "left"))
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
    ap.add_argument("--arm", choices=["kd", "ce"], default="ce",
                    help="ce = CE-PRIMARY, the rung-1 default after gate D3 failed at the MVE (plain CE beat "
                         "span-KD at all four stage exits, final -0.0112 = 2.2 sigma). kd = the challenger arm, "
                         "which must now be asked for explicitly rather than inherited.")
    ap.add_argument("--kd", choices=["anchor", "span"], default="anchor",
                    help="anchor = the SEALED D3 design (top-K projected onto the segment's first student token; "
                         "measured to deliver ~15%% of H(teacher)). span = prefix-conditioned over the whole segment "
                         "from the SAME stored rows (~83-86%% of H(teacher), zero extra teacher compute). "
                         "Default is the sealed design: switching it is the Architect's call, not the Builder's.")
    ap.add_argument("--recall", choices=["on", "off"], default="on", help="off = the D4-clause-2 control arm")
    ap.add_argument("--stages", default="CDEF")
    ap.add_argument("--steps", type=int, default=20000, help="TOTAL steps, split across stages by STAGE_SPLIT")
    ap.add_argument("--seq", type=int, default=512); ap.add_argument("--batch", type=int, default=16,
                    help="MICRO-batch. Effective batch = batch*accum. On <=16GB use --batch 8 --accum 2 to keep "
                         "effective 16 and fit. NOTE: stage E was long assumed to be the memory peak; WS2 measured "
                         "otherwise -- the SSM scan is (stage C alone peaks at 4.86GB, the MoE is ~3%% of peak). "
                         "Stage E was the last drop, not the load, so accum is a whole-run choice, not an E fix.")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (effective batch = batch*accum)")
    ap.add_argument("--sparse-moe", action="store_true",
                    help="stage-E: active-only MoE dispatch instead of compute-all (grad-equivalent, "
                         "deterministic; the win scales with E -- 1.5x at E32, 4.3x at E128)")
    ap.add_argument("--compile-scan", action="store_true",
                    help="regional torch.compile on the SSM mixer only (Linux; needs Triton). The scan is the "
                         "measured bottleneck: it materializes two (B,L,Dn,N) tensors per layer. Compiles the "
                         "forward FUNCTION, not the module, so state_dict keys -- and therefore resume -- are "
                         "unchanged. Fails loudly if Triton is missing rather than silently no-op'ing.")
    ap.add_argument("--fp32-opt", action="store_true",
                    help="force fp32 AdamW instead of AdamW-8bit (the A/B control for the optimizer)")
    ap.add_argument("--expect-slice-sha", default="",
                    help="fail unless the logits manifest carries this slice fingerprint. The hash is sealed in the "
                         "pre-registration, so a corpus or a slice that changed under a registered run cannot pass "
                         "unnoticed. Verified at every start.")
    ap.add_argument("--restrict-to-slice", action="store_true",
                    help="sample ONLY the positions covered by the teacher logits, whatever the arm. Required for "
                         "the screening: without it the CE control trains on the full corpus while the KD arm sees "
                         "only the covered slice, and the comparison stacks subset-vs-corpus on top of KD-vs-CE.")
    ap.add_argument("--dense-paired", action="store_true",
                    help="stage E keeps the dense ternary MLP instead of upcycling to MoE: the ACTIVE-param-matched "
                         "control (~11.0M dense vs ~11.2M active MoE). Branch it from the same stage-D artefact as "
                         "the MoE arm. Any result must be cited as 'at equal active cost', with the total-param "
                         "ratio alongside -- active-matching favours the MoE by construction (2.7x total params).")
    ap.add_argument("--xproj-rank", type=int, default=0,
                    help="0 = dense control; r>0 factorizes every SSM x_proj as U.V of rank r (Inventor S1: "
                         "r=26 beat dense 3/3 seeds at 17.6%% of the bytes). Set at construction, not a stage switch.")
    ap.add_argument("--branch-from", default="",
                    help="fork from a stage-exit artefact and run only the stages AFTER it. Both arms of a "
                         "stage-boundary A/B fork from the SAME file, so the fork point is byte-identical and "
                         "only the downstream stages cost twice.")
    ap.add_argument("--save-stage-ckpt", default="",
                    help="dir for PERMANENT stage-exit checkpoints (model+cfg, no optimizer). Separate from "
                         "--resume-ckpt, which is transient and deleted on completion. Prerequisite for the "
                         "rung-1 branch-from-D A/B.")
    ap.add_argument("--qat-alpha", type=int, default=0,
                    help="stage-D epsilon-identity: ramp alpha 0->1 over N steps (0 = hot-swap, the MVE behaviour). "
                         "Keep N <= ~10%% of stage D or the QAT pressure ramps in too late.")
    ap.add_argument("--max-nonfinite", type=int, default=50,
                    help="abort after this many CONSECUTIVE non-finite loss steps (fp16 forward-overflow deadlock)")
    ap.add_argument("--require-p62", action="store_true",
                    help="refuse to start unless the P62 code-val stream is present. The screening's "
                         "deciding metric; without this flag a package missing it completes and decides "
                         "nothing. The rung-1 cells set it.")
    ap.add_argument("--warmup", type=int, default=0,
                    help="linear LR warmup steps at each stage entry (0 = off, the run-2 pre-registered behaviour)")
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--alpha", type=float, default=0.5,
                    help="KD weight at anchor positions: loss = (1-a)*CE + a*KD there, CE elsewhere")
    ap.add_argument("--lam-nce", type=float, default=0.05); ap.add_argument("--load-w", type=float, default=0.01)
    ap.add_argument("--fp16", action="store_true"); ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--eval-tok", type=int, default=200_000)
    ap.add_argument("--ckpt-min", type=float, default=30.0, help="resume-checkpoint cadence (min)")
    ap.add_argument("--resume", action="store_true", help="resume from --resume-ckpt if it exists")
    ap.add_argument("--resume-ckpt", default="", help="resume/checkpoint path (default: CKPT/resume_<arm>_<kd>_<recall>_<tag>.pt)")
    ap.add_argument("--time-budget-min", type=float, default=0.0,
                    help="stop cleanly after this wall-clock (save a resume checkpoint, exit MVE-INCOMPLETE). "
                         "0=off. Kaggle: set ~660 to land under the 12h session cap.")
    ap.add_argument("--delete-behind", action="store_true")
    ap.add_argument("--chunk-steps", type=int, default=0,
                    help="steps before advancing the resident chunk window. 0 = DERIVE from the budget "
                         "(steps // n_chunks // chunk-sweeps) so the window sweeps the ring representatively; "
                         "a positive value is an explicit override for tests. The launch cells leave it 0.")
    ap.add_argument("--chunk-sweeps", type=int, default=3,
                    help="how many times the resident window sweeps the whole chunk ring over the budget "
                         "when chunk-steps is derived. K=3 clears the ring-end taper with margin.")
    ap.add_argument("--logits-dir", default="",
                    help="directory holding logits_<tag>/ when the teacher logits ship as their own shared "
                         "dataset instead of inside the arm's data package. Empty = read from --data-dir.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="auto"); ap.add_argument("--out", default="")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="permit training on CPU. Without it a missing CUDA device is a hard error, "
                         "because a silent CPU fall back is indistinguishable from a hung run.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="", help="override input dir (Kaggle: /kaggle/input/<ds>/data)")
    ap.add_argument("--ckpt-dir", default="", help="override checkpoint/output dir (Kaggle: /kaggle/working)")
    a = ap.parse_args()
    global DATA, CKPT
    if a.data_dir: DATA = a.data_dir            # KDChunks + all data loads read this module global
    if a.ckpt_dir: CKPT = a.ckpt_dir
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
    # A silent fall back to CPU is indistinguishable from a hung run: measured here, ten minutes without a
    # single step because the interpreter on PATH carried a CPU-only torch. On a real Kaggle session that
    # burns the session budget and produces nothing, and "slow" and "wrong environment" look identical from
    # the outside. Refusing is cheap; --allow-cpu makes the CPU path a declared choice instead of an
    # accident (the package smokes use it deliberately).
    if dev_type == "cpu" and not a.allow_cpu:
        sys.exit("ERROR: no CUDA device -- refusing to train on CPU.\n"
                 f"  torch {torch.__version__}, cuda available = {torch.cuda.is_available()}\n"
                 "  A CPU fall back is not slow training, it is a wrong environment that LOOKS like slow\n"
                 "  training, and on a metered session it costs the whole budget before anyone notices.\n"
                 "  Deliberate CPU run: pass --allow-cpu.")
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
    # ---- CONDITIONS (a) + (b) of prereg v7, ASSERTED AT EVERY RUN START, not verified once by hand ----
    # (b) The two vocab arms tokenize the same corpus differently, so "the last 2% of TOKENS" is a
    # different byte range in each: the arms would evaluate different text and produce a healthy-looking
    # comparison of nothing. pack_kaggle cuts the split in BYTE space at a document boundary and records
    # val_split_byte; this is the consumer that makes that record load-bearing. The invariant is exact,
    # not approximate -- the expected byte lengths of the first n_train_tok tokens must sum to precisely
    # that offset, in BOTH vocabularies, which is what "byte-identical" means operationally.
    # (a) The same line states which stream the tail is cut from: val_split_byte < meta['bytes'] means it
    # is carved out of the covered slice the arms train on, not out of some other corpus.
    _vsb = meta.get("val_split_byte")
    if _vsb is not None:
        _eln = np.asarray(bpe.exp_len, dtype=np.int64)
        # bincount, not fancy-indexing: a gather over ~185 M positions would allocate ~1.5 GB for a
        # number we already know how to get in V buckets. Kaggle has 13 GB and a trainer to load.
        _cnt = np.bincount(ids[:ntr], minlength=len(_eln))
        _got = int((_cnt * _eln[:len(_cnt)]).sum())
        if _got != int(_vsb) or int(_vsb) >= int(meta["bytes"]):
            sys.exit(
                f"ERROR: the validation split is not the one this arm was registered against. Refusing to "
                f"produce a gate number.\n"
                f"  first {ntr} tokens of V={V} cover {_got} bytes; the package declares {_vsb} "
                f"(delta {_got - int(_vsb)}).\n"
                f"  slice is {meta['bytes']} bytes, so the split must fall strictly inside it.\n"
                f"  A mismatch here means the arms evaluate different text: the comparison would run "
                f"cleanly and mean nothing.")
        log(f"  val split VERIFIED byte-identical: train {_got} B == declared {_vsb} B ({ntr} tok at V={V}); "
            f"tail {int(meta['bytes']) - int(_vsb)} B = "
            f"{100*(int(meta['bytes'])-int(_vsb))/int(meta['bytes']):.3f}% cut FROM the covered slice")
    # P62 code-val: the DECIDING metric for the screening (prereg v7). External, fixed, temporally held out,
    # byte-identical across arms by construction. Absent in the MVE packages, so it stays optional.
    _p62 = os.path.join(DATA, f"p62_{a.tag}.u16")
    p62 = np.fromfile(_p62, dtype=np.uint16).astype(np.int64) if os.path.isfile(_p62) else None
    # FAIL FAST when the deciding metric is required but absent. Rung-1 stage-1 ran to completion twice on
    # a package that predated the p62 stream: the file was missing, p62 stayed None, the [DECIDING] line
    # never printed, and .done was still written -- 18 GPU-hours that answered only the forbidden tail val.
    # A screening that cannot compute its decider must not look like a finished screening. Optional stays
    # the default for the MVE-era packages; the cell passes --require-p62 so a repeat is a startup error,
    # not a silent 9-hour no-op.
    if a.require_p62 and p62 is None:
        sys.exit(f"ERROR: --require-p62 set but no {os.path.basename(_p62)} in {DATA}.\n"
                 f"  This is the DECIDING metric. Without it the run would complete and decide nothing --\n"
                 f"  which is exactly how the first stage-1 attempt wasted two full sessions. The data\n"
                 f"  package is a generation behind: rebuild/re-upload it with the p62 stream present.")
    # The CE control must see the SAME positions as the KD arm, so build the chunk index whenever the slice is in
    # play -- even for --arm ce, which then uses it only for the sampling window and never reads a logit.
    need_win = a.arm == "kd" or a.restrict_to_slice
    kdc = KDChunks(a.tag, seg_row, delete_behind=a.delete_behind, logits_dir=a.logits_dir) if need_win else None
    if kdc is not None:
        man = kdc.man
        # DERIVE the chunk-advance cadence from the budget, do not take it by hand. The window advances one
        # chunk every chunk_steps steps; over the budget it must sweep the whole ring several times, or the
        # alpha trend is measured on a PREFIX of the slice (measured: chunk_steps=200 over 15106 steps swept
        # 0.62 of the ring -- 44 of 121 chunks never sampled). steps // (n_chunks * K) guarantees K sweeps and,
        # because the three alpha arms share one budget, guarantees they share one cadence -> one sampling
        # domain -> a one-variable curve. A hand-set value could desync if someone edits one arm's budget.
        # ws3_chunk_residence.py is the independent gate: deriving is not verifying.
        if a.chunk_steps <= 0:
            a.chunk_steps = max(1, a.steps // (len(kdc.chunks) * a.chunk_sweeps))
            log(f"  chunk-steps DERIVED: {a.steps} steps / ({len(kdc.chunks)} chunks x {a.chunk_sweeps} sweeps) "
                f"= {a.chunk_steps}  (window sweeps the ring {a.steps/(a.chunk_steps*len(kdc.chunks)):.2f}x)")
        else:
            log(f"  chunk-steps EXPLICIT: {a.chunk_steps} (override; the launch cells derive it)")
        sha = man.get("slice_sha256", "")
        if a.expect_slice_sha:
            if sha != a.expect_slice_sha:
                sys.exit(f"ERROR: slice fingerprint mismatch.\n"
                         f"  manifest : {sha or '(absent -- produced before slicing existed)'}\n"
                         f"  expected : {a.expect_slice_sha}\n"
                         f"  The corpus or the slice is not the one this run was registered against. Refusing.")
            log(f"  slice verified: {man.get('n_windows_selected','?')}/{man.get('n_windows_total','?')} windows, "
                f"seed {man.get('slice_seed','?')}, sha {sha[:16]}...")
        elif sha:
            log(f"  slice: {man.get('n_windows_selected','?')}/{man.get('n_windows_total','?')} windows, "
                f"seed {man.get('slice_seed','?')}, sha {sha[:16]}... (not verified: --expect-slice-sha unset)")
    kd_active = a.arm == "kd"          # whether KD LOSS is applied; the window is a separate concern

    # ---- model --------------------------------------------------------------------------------------
    model = MVEStudent(V, **S0)
    if a.xproj_rank > 0:
        # Applied BEFORE .to(dev) and before any stage surgery: this is a parameterization of the model, decided
        # once at construction, not a curriculum switch. It changes state_dict KEYS (x_proj.weight becomes
        # x_proj.Vl/.Ul), so a resume or a branch whose flag disagrees would fail on load -- the guard below makes
        # that a clear message instead of a shape-mismatch traceback.
        nx, dn, ln = model.lowrank_xproj(a.xproj_rank)
        log(f"  x_proj low-rank r={a.xproj_rank}: {nx} swapped | {ln}/{dn} params = "
            f"{100*ln/max(dn,1):.1f}% of dense")
    model = model.to(dev)
    if a.compile_scan:
        # Compile the mixer's forward FUNCTION, not the module. torch.compile(module) returns an OptimizedModule
        # whose state_dict keys gain an "_orig_mod." prefix, which would silently break every resume and every
        # cross-run checkpoint comparison we rely on. Patching .forward leaves parameters and keys untouched.
        import importlib.util
        if importlib.util.find_spec("triton") is None:
            raise SystemExit("--compile-scan needs Triton (Linux). It is absent here, and a silent no-op would "
                             "put an unmeasured 'compiled' label on an uncompiled run. Refusing to continue.")
        nc = 0
        for blk in model.blocks:
            mx = getattr(blk, "mix", None)
            if mx is not None and hasattr(mx, "A_log"):        # the SSM mixer, not the SWA layers
                # mode="default" PINNED, and max-autotune deliberately not used: autotuning benchmarks kernels at
                # compile time and can pick a different winner per process, so session 2 would compile differently
                # from session 1 and a resume would not be numerically continuous. Determinism was bought with a
                # real bug fix in WS2; it is not handed back to Inductor for a throughput number.
                mx.forward = torch.compile(mx.forward, dynamic=False, mode="default"); nc += 1
        log(f"  [compile-scan] regional torch.compile on {nc} SSM mixers (state_dict keys unchanged)")
    pr = param_report(model)
    log(f"Phase64.4 MVE | arm={a.arm} recall={a.recall} stages={a.stages} | dev={dev} amp={amp} ddp={ws}")
    log(f"  student S0: D{S0['D']} N{S0['N']} L{S0['L']} | params={pr['total']/1e6:.2f}M (MLP {pr['mlp']/1e6:.2f}M) "
        f"| V={V} | corpus {meta['bytes']/2**20:.0f}MiB, {meta['n_student_tok']} tok, anchors {100*meta['anchor_frac']:.1f}%")
    log(f"  teacher={meta['teacher']} K={kdc.K if (kd_active and kdc) else 0} | alpha={a.alpha} lam_nce={a.lam_nce} | seq={a.seq} batch={a.batch} steps={a.steps}")

    # ---- KD TARGET AGREEMENT (the D3 mechanism check; run BEFORE any training, costs seconds) ---------
    # The span map is only worth training on if the projected teacher distribution actually points at the true next
    # student token. If q[y_true] were ~0 the projection would be broken (bad t2s, or an anchor off-by-one) and the
    # KD arm would be training on noise. This is the instrument that separates "D3 is hard" from "D3 is buggy".
    #
    # It measures THE MODE ACTUALLY BEING TRAINED. It used to always report the `anchor` projection even under
    # `--kd span`, so every run-2/run-3 log carries "retained 13%" while the trainer was in fact retaining ~85% --
    # a number that was not wrong about anchor-KD, merely about this run. A gate-bearing log may not describe a
    # code path that is switched off.
    if kd_active and kdc is not None and P0:
        rr = np.random.default_rng(7)
        w_lo, w_hi, idx0, loaded0 = kdc.window()                 # sample INSIDE the resident chunk window
        w_hi = min(w_hi, train_hi - a.seq - 1)
        p0 = rr.integers(w_lo, max(w_hi, w_lo + 1), size=8)
        pf0 = (p0[:, None] + np.arange(a.seq)[None, :]).reshape(-1)     # global student index of each sampled slot
        yy = np.stack([ids[i+1:i+1+a.seq] for i in p0]).reshape(-1)
        # exactly the selection step_loss() performs for this mode: anchors only, or every position of the segment
        row0 = anchors[pf0] if a.kd == "anchor" else seg_row[pf0]
        selq = np.nonzero(row0 >= 0)[0]
        ti_, pr_, ok_ = kdc.gather(row0[selq], loaded0)
        selq = selq[ok_]; ti_ = ti_[ok_]; pr_ = pr_[ok_]
        if len(selq):
            if a.kd == "anchor":
                q = kd_targets(ti_, pr_, t2s, V, dev); keep = np.ones(len(selq), dtype=bool)
            else:
                ss0 = seg_step[pf0][selq]
                gi0 = np.clip((pf0[selq] - ss0 + 1)[:, None] + np.arange(SPAN_S)[None, :], 0, len(ids) - 1)
                q, okm0 = kd_targets_span(ti_, pr_, ss0, ids[gi0].astype(np.int64),
                                          np.arange(SPAN_S)[None, :] < ss0[:, None], dtok, dlen, V, dev)
                keep = okm0.cpu().numpy() if torch.is_tensor(okm0) else okm0
                q = q[okm0]; selq = selq[keep]; pr_ = pr_[keep]
            yt = torch.from_numpy(yy[selq]).to(dev)
            mass = q.gather(1, yt[:, None]).squeeze(1)
            top1 = (q.argmax(1) == yt).float().mean().item()
            unit = "anchors" if a.kd == "anchor" else "span positions"
            log(f"  KD-target check [{a.kd}] on {len(selq)} {unit}: mass on TRUE next token = {mass.mean().item():.3f} "
                f"(median {mass.median().item():.3f}) | argmax(q)==y: {100*top1:.1f}% | mapped mass/row: "
                f"{float(pr_.sum(1).mean()):.3f}")
            # THE D3 QUESTION, quantified. KD is only worth its cost if the target carries MORE than the hard label
            # ("dark knowledge"). The span projection maps every teacher token to the FIRST student token of its
            # bytes -- and first-tokens are low-entropy (" the"/" then"/" they" collapse onto the same student token).
            # If H(q) << H(teacher), the projection has destroyed precisely the information KD exists to transfer,
            # and the KD arm is training on an expensive near-one-hot. Report both entropies in bits.
            pt = torch.from_numpy(pr_).to(dev)                           # (M,K) teacher top-K probs
            pt = pt / pt.sum(1, keepdim=True).clamp_min(1e-8)
            h_t = -(pt * pt.clamp_min(1e-9).log2()).sum(1)               # teacher entropy on its own top-K
            h_q = -(q * q.clamp_min(1e-9).log2()).sum(1)                 # projected-target entropy
            # PER-POSITION ratio -- NOT the 83-86% chain-rule figure from kd_information.py, which is the total
            # recoverable information SUMMED over a span. Conditioning on the prefix necessarily lowers the entropy
            # still outstanding at each interior position (that is the mechanism, not a defect), so the per-position
            # mean is smaller than the span total by construction. Labelled explicitly because the two numbers have
            # already been printed side by side once and invite exactly the wrong comparison.
            log(f"    dark-knowledge [{a.kd}]: H(teacher top-K) = {h_t.mean().item():.3f} bits  ->  H(q projected) = "
                f"{h_q.mean().item():.3f} bits   (retained {100*h_q.mean().item()/max(h_t.mean().item(),1e-6):.0f}% "
                f"PER-POSITION; not the summed-over-span figure); "
                f"q is near-one-hot on {100*(mass>0.95).float().mean().item():.0f}% of {unit}")

    # ONE optimizer factory: the stage transitions (D/E) rebuild the optimizer after the model surgery, and they must
    # rebuild the SAME kind -- otherwise the curriculum silently drops back to fp32 Adam states after stage C and the
    # plan's 8-bit memory budget stops holding where it matters most (the MoE stage, the biggest parameter count).
    use8 = [dev_type == "cuda" and not a.fp32_opt]
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

    def bpb_on(stream, cap=0):
        """BPB over an arbitrary token stream. cap=0 means the WHOLE stream, final partial window included.

        Covering the whole stream matters for the P62 metric: stepping by seq and dropping the remainder
        evaluates a different number of BYTES under each tokenizer (the token counts differ), so two arms
        would be compared over slightly different text while both looked fine. With cap=0 every target is
        scored exactly once and the byte total is a function of the file, not of the vocabulary."""
        model.eval(); bits = 0.0; nb = 0
        with torch.no_grad():
            n = (min(cap, len(stream) - 1) if cap else len(stream) - 1)
            # Full-length windows are batched; only the ragged remainder is evaluated alone. Windows are
            # independent sequences either way, so batching changes the arithmetic not at all -- it only
            # stops 1.5 MB of val from costing thousands of sequential forwards on every stage exit.
            W = a.seq
            full = n // W
            for i in range(0, full, a.batch):
                k = min(a.batch, full - i)
                st = [(i + j) * W for j in range(k)]
                x = torch.from_numpy(np.stack([stream[p0:p0+W] for p0 in st])).to(dev)
                y = torch.from_numpy(np.stack([stream[p0+1:p0+1+W] for p0 in st])).to(dev)
                lg, _ = model(x, y)
                bits += F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction="sum").item() / math.log(2)
                nb += int(el[y.reshape(-1).cpu()].sum())
            rem = n - full * W
            if rem > 0:
                p0 = full * W
                x = torch.from_numpy(stream[p0:p0+rem][None]).to(dev)
                y = torch.from_numpy(stream[p0+1:p0+1+rem][None]).to(dev)
                lg, _ = model(x, y)
                bits += F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction="sum").item() / math.log(2)
                nb += int(el[y.reshape(-1).cpu()].sum())
        model.train(); return bits / max(nb, 1), nb

    def val_bpb(cap):
        return bpb_on(val, cap)[0]

    def p62_bpb():
        """The DECIDING metric (prereg v7). Returns (bpb, bytes_evaluated, ok).

        PLANTED INVARIANT: bytes_evaluated plus the first token's own bytes must equal the declared file
        size EXACTLY. The first token is never predicted, so it is the only byte that may be missing; any
        other shortfall means the harness is normalising over something that is not this file, which is
        the eval-side form of training on the wrong package. It is checked, not assumed."""
        if p62 is None: return float("nan"), 0, False
        b, nb = bpb_on(p62, 0)
        total = nb + int(el[int(p62[0])])
        return b, nb, total == int(meta.get("p62_bytes", -1))

    # ---- the step ------------------------------------------------------------------------------------
    # Batch sampling is a PURE FUNCTION of (seed, rank, gstep, micro) rather than a carried RNG stream. Two reasons,
    # both load-bearing: (1) under DDP every rank would otherwise reload rank-0's saved RNG state on resume and all
    # ranks would draw the SAME batch -- silently collapsing the effective batch back to one rank's worth;
    # (2) it makes resume exactly bit-faithful at any world size with no RNG state in the checkpoint.
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

    def step_loss(reverse=False, gs=0, mi=0):
        r = np.random.default_rng([a.seed, rank, gs, mi])
        p = r.integers(state["p_lo"], max(state["p_hi"], state["p_lo"] + 1), size=a.batch)
        x = torch.from_numpy(np.stack([ids[i:i+a.seq] for i in p])).to(dev)
        y = torch.from_numpy(np.stack([ids[i+1:i+1+a.seq] for i in p])).to(dev)
        ctx = torch.autocast(dev_type, dtype=amp) if amp is not None else torch.autocast(dev_type, enabled=False)
        with ctx:
            logits, aux = net(x, y)
        ce = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1), reduction="none")   # (B*T,)
        nkd = 0; kd_m = torch.zeros((), device=dev)
        if kd_active and kdc is not None and a.alpha > 0:
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

    # ---- the curriculum (RESUMABLE across sessions — gate #2) -----------------------------------------
    # The model changes SHAPE between stages (dense fp -> ternary -> MoE+recall). A resume therefore cannot just
    # load a state_dict onto a fresh model: it must first replay the structural surgery of every stage up to and
    # including the one being resumed, THEN load the checkpoint, THEN continue from the saved step. That is exactly
    # what makes Kaggle's 12h cap survivable, and it is gate #2 of the pre-registration.
    stages = [c for c in a.stages]
    tot_w = sum(STAGE_SPLIT[c] for c in stages)
    budget = {c: max(1, int(a.steps * STAGE_SPLIT[c] / tot_w)) for c in stages}
    STAGE_LR = {"C": a.lr, "D": a.lr * 0.5, "E": a.lr * 0.5, "F": a.lr * 0.1}

    def stage_surgery(st):
        """The structural surgery applied at a stage's ENTRY (idempotent per stage). Returns a log line ('' if none)."""
        if st == "D":
            n = model.qat_ternary(alpha_sched=a.qat_alpha > 0); model.to(dev)
            if a.qat_alpha > 0:
                model.set_qat_alpha(0.0)                 # exact fp32 identity at the instant of the switch
                return (f"[D] QAT switch: {n} MLP linears -> BitLinear158Alpha (weights carried), lr x0.5 | "
                        f"alpha 0->1 over {a.qat_alpha} steps (epsilon-identity)")
            return f"[D] QAT switch: {n} MLP linears -> BitLinear158 (weights carried), lr x0.5"
        if st == "E" and a.dense_paired:
            # ACTIVE-PARAM-MATCHED control: stage E runs with its full step budget and the same recall insertion,
            # but the MLP stays dense-ternary. ~11.0M dense vs ~11.2M active in the MoE arm -- matched on what the
            # engine actually pays per token, which is the claim the product thesis makes (active != total).
            # Deliberately NOT total-matched: a 30M dense answers a different, more academic question.
            msg = "[E] dense-paired control: MoE upcycle SKIPPED (MLP stays dense-ternary, active-param-matched)"
            if a.recall == "on":
                nr = model.add_recall(lam_nce=a.lam_nce); msg += f" + recall slot ({nr/1e3:.1f}K par, gate=0)"
            model.to(dev)
            return msg
        if st == "E":
            n = model.upcycle_moe(dev_type=dev_type, load_w=a.load_w, seed=a.seed, sparse=a.sparse_moe)
            msg = (f"[E] MoE upcycle: {n} MLPs -> E{MOE['E']}xh{MOE['hid_e']} top{MOE['k']} (magnitude-matched seed"
                   + (", ACTIVE-ONLY dispatch)" if a.sparse_moe else ", compute-all)"))
            if a.recall == "on":
                nr = model.add_recall(lam_nce=a.lam_nce); msg += f" + recall slot ({nr/1e3:.1f}K par, gate=0)"
            model.to(dev)
            return msg
        return ""   # C entry = none; F entry = lr change only (handled by STAGE_LR), no structural surgery

    rpath = a.resume_ckpt or os.path.join(CKPT, f"resume_{a.arm}_{a.kd}_{a.recall}_{a.tag}.pt")
    donepath = (a.out or os.path.join(CKPT, f"mve_{a.arm}_{a.kd}_{a.recall}_{a.tag}.pt")) + ".done"
    if a.resume and os.path.exists(donepath):
        log(f"MVE-DONE already: {donepath} exists. Nothing to do.");
        if ddp: torch.distributed.destroy_process_group()
        return

    def restore_qat_alpha(start_si):
        """alpha is a plain attribute, NOT a state_dict entry, so loading weights does not restore it. Replaying the
        stage-D surgery sets it to 0 (the epsilon-identity at the switch), and only stage D's own loop ramps it to 1
        -- so any restart PAST stage D leaves the model sitting at alpha=0, i.e. not ternary at all.

        This stays invisible whenever stage E upcycles, because the upcycle REPLACES the BitLinear158Alpha modules
        outright. It bites exactly one arm: --dense-paired, which keeps them. That arm would have trained and been
        reported in fp32 against a ternary MoE arm, with a perfectly plausible BPB and nothing to flag it.
        Stage D is complete at any such restart, so the correct value is exactly 1.0."""
        if a.qat_alpha <= 0 or "D" not in stages: return
        if stages.index("D") < start_si:
            n = model.set_qat_alpha(1.0)
            if n: log(f"  [alpha] stage D already complete -> alpha=1.0 restored on {n} modules")

    rows = []; hist = []; gstep = 0; start_si = 0; start_step = 0; resumed_b0 = float("nan")
    t_run = time.time()

    # ---- resume path ----
    if a.resume and os.path.exists(rpath):
        ck = torch.load(rpath, map_location=dev)
        if ck.get("fmt") != "mve-resume-2":
            sys.exit(f"ERROR: {rpath} is a STALE checkpoint (fmt={ck.get('fmt')}, this apparatus writes mve-resume-2). "
                     f"It was produced by a superseded apparatus -- delete it (and any final_*.pt / *.done next to it) "
                     f"and start this arm from scratch. Resuming across an apparatus fix would silently mix two runs.")
        if ck.get("stages") != stages or ck["cfg"]["steps"] != a.steps or ck["cfg"]["arm"] != a.arm \
           or ck["cfg"]["kd"] != a.kd or ck["cfg"]["recall"] != a.recall \
           or ck["cfg"].get("xproj_rank", 0) != a.xproj_rank:
            sys.exit(f"ERROR: resume checkpoint {rpath} does not match this command "
                     f"(stages/steps/arm/kd/recall/xproj_rank).")
        # Vocabulary, checked separately because it is not an argument. `ck.get("V", V)` passes for
        # checkpoints written before this field existed -- MVE-era files, the only way to stay permissive
        # there without weakening anything: every rung-1 checkpoint carries V from its first write, so the
        # hole closes for exactly the runs this guard exists for.
        if int(ck.get("V", V)) != int(V):
            sys.exit(f"ERROR: resume checkpoint {rpath} was written by the V={ck.get('V')} arm; this run is "
                     f"V={V}.\n  The stage-1 arms are identical in every flag -- they differ only in the "
                     f"vocabulary, so this is the mix-up a file name alone cannot prevent.")
        start_si = ck["stage_idx"]; start_step = ck["step_in_stage"]; gstep = ck["gstep"]
        rows = ck["rows"]; hist = ck["hist"]; resumed_b0 = ck.get("stage_b0", float("nan"))
        for st in stages[:start_si + 1]:
            m = stage_surgery(st)
            if m: log(f"  [resume] replay surgery: {m}")
        net = wrap(model)
        opt = make_opt(STAGE_LR[stages[start_si]])
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        if kdc is not None: kdc.pos = ck["kdc_pos"]; refresh()
        restore_qat_alpha(start_si)
        log(f"  [resume] {rpath}: stage {stages[start_si]} step {start_step}/{budget[stages[start_si]]} gstep {gstep}")

    elif a.branch_from:
        # BRANCH-FROM-STAGE: fork both A/B arms of a stage-boundary experiment from ONE shared upstream state, so
        # only the stages after the fork run twice. The fork must be byte-identical on both arms: same weights,
        # same gstep (batch sampling is a pure function of it), same KD chunk window, same GradScaler. The
        # optimizer is deliberately NOT restored -- stage entry rebuilds it on the baseline path too, so a fresh
        # one is what the baseline itself would have had.
        bk = torch.load(a.branch_from, map_location=dev, weights_only=False)
        if bk.get("fmt") != "mve-stage-2":
            sys.exit(f"ERROR: {a.branch_from} has fmt={bk.get('fmt')!r}, expected 'mve-stage-2'. A stage artefact "
                     f"from the older format lacks the scaler/kdc state and would fork a NON-identical run.")
        bsi = bk["stage_idx"]
        if bsi + 1 >= len(stages):
            sys.exit(f"ERROR: {a.branch_from} is stage {bk['stage']}, which is the last in --stages {a.stages}.")
        start_si = bsi + 1; start_step = 0; gstep = bk["gstep"]
        rows = list(bk["rows"]); hist = list(bk["hist"])
        for st in stages[:bsi + 1]:
            m = stage_surgery(st)
            if m: log(f"  [branch] replay surgery: {m}")
        model.load_state_dict(bk["model"]); scaler.load_state_dict(bk["scaler"])
        if kdc is not None: kdc.pos = bk["kdc_pos"]; refresh()
        restore_qat_alpha(start_si)
        net = wrap(model)

        # FORK-POINT ASSERTION -- a general detector, not a fix for the last bug.
        # Three times now, state living OUTSIDE state_dict has failed to survive a checkpoint boundary: kdc.pos,
        # the GradScaler, and the alpha of BitLinear158Alpha. The failure mode is identical and maximally
        # insidious every time: the restored model is wrong yet trains to a perfectly plausible loss. The alpha
        # case would have compared an fp32 control against a ternary arm and called it "at equal active cost".
        # Test coverage could not be relied on to catch these -- the alpha bug was invisible to every arm whose
        # stage E replaces the alpha modules, so coverage depended on which arm happened to exist.
        # So: before the new stage does anything, the fork must reproduce its parent's exit BPB. Same weights,
        # same deterministic eval -> the numbers must agree. Whatever state we have not yet discovered we carry,
        # this catches it.
        fb = val_bpb(a.eval_tok)
        if abs(fb - bk["bpb"]) > 1e-4:
            sys.exit(f"ERROR: fork point does not reproduce its parent.\n"
                     f"  parent exit BPB = {bk['bpb']:.6f}   fork entry BPB = {fb:.6f}   delta = {fb-bk['bpb']:+.6f}\n"
                     f"  Some state outside state_dict was not restored. This branch is VOID -- do not train on it.")
        log(f"  [fork-check] reproduces parent exit BPB {fb:.4f} (delta {fb-bk['bpb']:+.6f}) -- branch valid")
        log(f"  [branch] forked from {a.branch_from} (end of stage {bk['stage']}, BPB {bk['bpb']:.4f}, "
            f"gstep {gstep}) -> resuming at stage {stages[start_si]}")

    def save_resume(si, step_next, sb0):
        if not P0: return
        os.makedirs(os.path.dirname(rpath) or ".", exist_ok=True)
        # V is stored because it is NOT in cfg: the vocabulary comes from the data directory, not from a
        # flag, so the resume guard could not see it. The two stage-1 screening arms are identical in
        # every argument it does check -- same --arm ce, same steps, same stages -- and differ only in the
        # vocabulary. On Kaggle, where resume files are carried between sessions by hand, that is a
        # reachable mix-up, and without this it surfaces as an opaque shape error deep inside
        # load_state_dict instead of a named refusal.
        torch.save(dict(fmt="mve-resume-2", V=V, stages=stages, stage_idx=si, step_in_stage=step_next,
                        gstep=gstep,
                        stage_b0=float(sb0), model=model.state_dict(), opt=opt.state_dict(),
                        scaler=scaler.state_dict(),   # no RNG state: sampling is a pure fn of (seed,rank,gstep,micro)
                        kdc_pos=(kdc.pos if kdc else 0), rows=rows, hist=hist, cfg=vars(a)), rpath + ".tmp")
        os.replace(rpath + ".tmp", rpath)   # atomic: a preemption mid-write can never corrupt the resume file

    for si in range(start_si, len(stages)):
        st = stages[si]
        resuming_here = (si == start_si and start_step > 0)
        if resuming_here:
            log(f"\n== stage {st}: RESUMING at step {start_step}/{budget[st]} ==")
            b0 = resumed_b0; i0 = start_step
        else:
            m = stage_surgery(st)
            if m: log("\n== " + m + " ==")
            net = wrap(model); opt = make_opt(STAGE_LR[st])
            if st == "E":
                pr = param_report(model)
                log(f"   params total={pr['total']/1e6:.2f}M active={pr['active']/1e6:.2f}M (active/total={pr['active']/pr['total']:.2f})")
            # SAME eval slice as the stage-exit read. They used to differ (entry 20k vs exit --eval-tok 200k), which
            # made every stage-transition delta a comparison of two different measurements -- and the transition
            # delta IS gate #4 (KD->QAT stability). One eval cap everywhere, or the gate reads noise.
            b0 = val_bpb(a.eval_tok); i0 = 0
            log(f"\n== stage {st}: {budget[st]} steps | val BPB at entry = {b0:.4f} ==")
        model.train(); t0 = time.time(); ntok_seen = 0; t_ck = time.time(); n_bad = 0
        # PER-STAGE peak. max_memory_allocated is a process-wide high-water mark, so without this reset every
        # stage after the first would report the largest stage seen so far -- the WS2 failure, where two models
        # resident at once printed one identical number for both and it read as a measurement.
        if dev_type == "cuda": torch.cuda.reset_peak_memory_stats(dev)
        for i in range(i0, budget[st]):
            if kdc is not None and gstep and gstep % a.chunk_steps == 0:
                kdc.advance(); refresh()
            # Optional linear warmup at stage entry. OFF by default: run 2 was pre-registered with a flat lr and the
            # arms must stay comparable. It exists because the flat 3e-3-from-step-1 is what let the CE arm reach an
            # overflowing activation regime by step 24 -- the KD arms only survived it because alpha halves their CE.
            # alpha-QAT: ramp the ternarization in instead of teleporting to it. Driven every step (not only inside
            # the ramp) so that a resume landing mid-stage-D restores the right alpha from step_in_stage alone --
            # alpha is a pure function of the step, so it needs no checkpoint state and cannot desync across ranks.
            if st == "D" and a.qat_alpha > 0:
                model.set_qat_alpha(min(1.0, (i + 1) / a.qat_alpha))
            if a.warmup > 0 and i < a.warmup:
                for g in opt.param_groups: g["lr"] = STAGE_LR[st] * (i + 1) / a.warmup
            elif a.warmup > 0 and i == a.warmup:
                for g in opt.param_groups: g["lr"] = STAGE_LR[st]
            opt.zero_grad(set_to_none=True)
            loss_v = 0.0; nkd = 0; ce_m = 0.0                    # gradient accumulation (effective batch = batch*accum)
            for _mi in range(a.accum):
                loss, nkd_i, ce_i = step_loss(reverse=(st == "F"), gs=gstep, mi=_mi)
                scaler.scale(loss / a.accum).backward()
                loss_v += float(loss) / a.accum; nkd += nkd_i; ce_m += ce_i / a.accum
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            gstep += 1; ntok_seen += a.batch * a.seq * a.accum * ws
            hist.append((st, gstep, loss_v))

            # ONE collective per step carrying BOTH stop conditions. Divergence and the time budget are decided
            # together and identically on every rank, because any rank that leaves the loop while another is still
            # in it strands the survivor in a collective with no partner -- which is exactly how run 3's arm A died:
            # rank 0 hit the budget and returned while rank 1 entered the next step, then sat in an all-reduce until
            # the 600 s NCCL watchdog aborted it. The checkpoint had already been written, so it was survivable, but
            # it must not recur. Merging the two flags into one all-reduce also keeps the per-step cost at one.
            #
            # The divergence half: an fp16 forward that overflows gives inf logits -> NaN loss -> GradScaler skips
            # the step -> the weights never change -> the same forward overflows again. A self-sustaining deadlock
            # the scaler cannot break (it rescales GRADIENTS; the overflow is in the forward). Run 2's CE arm fell
            # into it at step 24 and burned ~11 T4-hours emitting NaN before printing MVE-DONE on a model frozen at
            # step 23. No .done is written here, so a diverged run can never be mistaken for a completed one.
            bad = 0.0 if math.isfinite(loss_v) else 1.0
            timeup = 1.0 if (a.time_budget_min > 0 and (time.time() - t_run) / 60.0 > a.time_budget_min) else 0.0
            if ddp:
                ft = torch.tensor([bad, timeup], device=dev)
                torch.distributed.all_reduce(ft, op=torch.distributed.ReduceOp.MAX)
                bad, timeup = float(ft[0].item()), float(ft[1].item())
            n_bad = n_bad + 1 if bad > 0 else 0
            if n_bad >= a.max_nonfinite:
                log(f"\n[DIVERGED] loss non-finite for {n_bad} consecutive steps (stage {st}, gstep {gstep}).")
                log("  fp16 forward overflow: the scaler cannot recover this -- the weights are frozen and the run is dead.")
                log(f"  last finite loss was at gstep {gstep - n_bad}. NO .done file written.")
                log("MVE-DIVERGED: apparatus stop. Do NOT read gates from this run.")
                if ddp: torch.distributed.destroy_process_group()
                sys.exit(2)
            if i == i0 or (i + 1) % max(1, budget[st] // 5) == 0:
                nce = model.recall._last_nce if model.recall is not None else 0.0
                gate = float(model.recall.gate.detach()) if model.recall is not None else 0.0
                log(f"  [{st}] {i+1:5d}/{budget[st]}  loss={loss_v:.4f} ce={ce_m:.4f} gnorm={float(gn):.2f} "
                    f"kd@{nkd}/{a.batch*a.seq*a.accum}" + (f" nce={nce:.3f} gate={gate:+.4f}" if model.recall else ""))
            if (time.time() - t_ck) / 60.0 > a.ckpt_min:
                save_resume(si, i + 1, b0); t_ck = time.time(); log(f"  [ckpt] resume-point at gstep {gstep}")
            if timeup > 0:                              # agreed by all ranks above, never decided locally
                save_resume(si, i + 1, b0)
                log(f"\n[TIME-BUDGET {a.time_budget_min:.0f}min HIT at gstep {gstep}] resume checkpoint -> {rpath}")
                log("MVE-INCOMPLETE: re-launch the IDENTICAL command with --resume to continue.")
                if ddp: torch.distributed.destroy_process_group()
                return
        dt = time.time() - t0
        b1 = val_bpb(a.eval_tok)
        tps = ntok_seen / max(dt, 1e-9)
        rows.append(dict(stage=st, steps=budget[st], bpb_in=b0, bpb_out=b1, d=b1 - b0, tok_s=tps, s=dt))
        pk = (f" | peak {torch.cuda.max_memory_allocated(dev)/2**20:.0f} MiB"
              if dev_type == "cuda" else "")
        if p62 is not None:
            pb, pn, pok = p62_bpb()
            # The invariant is reported AFFIRMATIVELY, not only on failure. Printed only when it breaks, a
            # healthy log says nothing and "it held" has to be inferred from silence -- the same weakness as
            # a control that has only ever passed. This is the planted control of the DECIDING metric, so
            # every arm's log carries the reconciliation in full and an auditor reads it instead of
            # trusting that a warning would have appeared. The residue is the first token, which cannot be
            # scored because nothing precedes it, and its byte length differs per vocabulary -- so the raw
            # counts across arms differ by a byte or two BY CONSTRUCTION while the total is exact.
            _pd = int(meta.get("p62_bytes", -1))
            log(f"  -> stage {st} P62 code-val BPB {pb:.4f}  [DECIDING]  over {pn} bytes"
                + (f" + {_pd - pn} B unscored first token = {_pd} declared  [byte invariant HOLDS]" if pok
                   else f"   << BYTE-COUNT INVARIANT FAILED: {pn} evaluated vs {_pd} declared; the harness "
                        f"is not normalising over this file and the number is NOT comparable"))
        log(f"  -> stage {st} done: val BPB {b0:.4f} -> {b1:.4f} ({b1-b0:+.4f}) | {tps:.0f} tok/s | "
            f"{dt/60:.1f} min{pk}")
        # STAGE-EXIT ARCHIVE -- deliberately NOT the resume file. The resume file is transient state that is
        # deleted on completion, which is why run 3 left no stage-D checkpoint and WS6 could not probe one.
        # These are permanent artefacts, and they are the prerequisite for the rung-1 branch-from-D A/B: both
        # upcycle arms fork from ONE shared stage-D state, so E+F (~25% of the tokens) is all that runs twice.
        # No optimizer state, which is where the bulk of a resume file's bytes live -- see the note below for
        # exactly which cross-boundary state IS carried, and why the optimizer is not part of it.
        if a.save_stage_ckpt and P0:
            sp = os.path.join(a.save_stage_ckpt, f"stage_{st}_{a.arm}_{a.kd}_{a.recall}_{a.tag}.pt")
            os.makedirs(a.save_stage_ckpt, exist_ok=True)
            # Everything that SURVIVES a stage boundary must be here, or a branch is not byte-identical to the
            # baseline at the fork. Optimizer moments are NOT in that set: stage entry rebuilds the optimizer
            # (make_opt), so they are zero on both sides by construction -- model-only is right for them.
            # The three that do survive: gstep (batch sampling is a pure function of it), kdc.pos (the KD chunk
            # window), and the GradScaler (created once, never rebuilt -- a fresh one would restart at 65536 and
            # skip steps the baseline did not skip). rows/hist ride along so the branch's report is continuous.
            torch.save(dict(fmt="mve-stage-2", stage=st, stage_idx=si, gstep=gstep, bpb=b1,
                            model=model.state_dict(), scaler=scaler.state_dict(),
                            kdc_pos=(kdc.pos if kdc else 0), rows=rows, hist=hist, cfg=vars(a)), sp + ".tmp")
            os.replace(sp + ".tmp", sp)
            log(f"     [stage-ckpt] {sp}  ({os.path.getsize(sp)/2**20:.0f} MiB)")
        start_step = 0

    # ---- report ---------------------------------------------------------------------------------------
    if P0:
        out = a.out or os.path.join(CKPT, f"mve_{a.arm}_{a.kd}_{a.recall}_{a.tag}.pt")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        torch.save(dict(model=model.state_dict(), cfg=vars(a), rows=rows, hist=hist), out)
        with open(out + ".done", "w") as f: f.write("done\n")
        if os.path.exists(rpath): os.remove(rpath)   # resume file no longer needed
        log(f"\n==== MVE run [{a.tag}] arm={a.arm} kd={a.kd} recall={a.recall} ====")
        log(f"  {'stage':6s} {'steps':>7s} {'BPB in':>9s} {'BPB out':>9s} {'delta':>8s} {'tok/s':>9s} {'min':>7s}")
        for r in rows:
            log(f"  {r['stage']:6s} {r['steps']:7d} {r['bpb_in']:9.4f} {r['bpb_out']:9.4f} {r['d']:+8.4f} "
                f"{r['tok_s']:9.0f} {r['s']/60:7.1f}")
        log(f"  saved -> {out}")
        log("MVE-DONE: all stages complete.")
    if ddp: torch.distributed.destroy_process_group()
    log("STOP. MVE curriculum above. No commit.")


if __name__ == "__main__":
    main()
