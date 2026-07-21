#!/usr/bin/env python3
# Phase 64.4 / MVE — component (B): offline teacher top-K logits, CHUNKED (generate -> train -> delete).
#   magic: "MVEB" 0x4D564542
#
# THE LOAD-BEARING OBSERVATION, stated up front so the STOP-B re-price can use it: offline KD needs the teacher's
# distribution over a corpus we ALREADY HAVE. That is TEACHER-FORCED SCORING (prefill only) -- NOT autoregressive
# generation. There is no KV-cache decode loop, no sampling, nothing sequential per token: it is one batched forward
# per window. The T1 "gen tok/s" column (batch-1 decode, physically inverted 0.5B<1.5B) was never the right meter for
# this cost. The right meter is SCORING tok/s, measured here.
#
# Format (V-agnostic by construction): one row per TEACHER token j, holding the teacher's predictive distribution FOR
# token j (i.e. the model's output at input position j-1), as top-K=32 ids (int32) + probs quantized to uint8.
#   ~ K*(4+1) = 160 B/teacher-token. Indexing by TEACHER token -- not by student anchor -- keeps the student vocab V
#   an OPEN A/B at rung-1 (anchors are a function of V; teacher tokens are not) AND is cheaper: teacher tokens are
#   0.5x student tokens on this corpus.
#
# Backends: --backend hf   (batched HF forward; works everywhere incl. Windows/3060 -> the local smoke)
#           --backend vllm (Linux/Kaggle; same output format, same chunk files)
#
# Run: .venv/Scripts/python.exe benchmarks/phase64/mve/mve_logits.py --tag smoke --backend hf --quant fp16
import argparse, glob, hashlib, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "results", "phase64", "mve")
K = 32


def pick_gpu(want_bf16=False):
    import torch
    if not torch.cuda.is_available(): return None
    n = torch.cuda.device_count()
    g = max(range(n), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
    p = torch.cuda.get_device_properties(g)
    if n > 1:
        oth = ", ".join(f"cuda:{i}={torch.cuda.get_device_properties(i).name}" for i in range(n) if i != g)
        print(f"NOTE: {n} GPUs; pinning ALL layers to cuda:{g} = {p.name} ({p.total_memory/2**30:.0f}GB). Ignored: {oth}",
              file=sys.stderr)
    if want_bf16 and not torch.cuda.is_bf16_supported():
        sys.exit(f"ERROR: bf16 requested but cuda:{g} ({p.name}) has no bf16.")
    return g


class ChunkWriter:
    """Rolling chunk files: gen chunk i+1 while training on chunk i, delete behind (the >=500 GB scratch discipline,
    exercised here at pilot scale so the production path is the same code)."""
    def __init__(s, out, tag, rows_per_chunk, resume=False):
        s.dir = os.path.join(out, f"logits_{tag}"); os.makedirs(s.dir, exist_ok=True)
        s.rpc = rows_per_chunk; s.chunks = []; s.buf_ids = None; s.buf_pq = None; s.j0 = None
        if resume:
            # Adopt the chunks already on disk. Rows are written in order and each chunk records its own
            # j0, so the resume point is simply the end of the last complete chunk -- no bookkeeping file
            # to fall out of sync with the data. A run that dies at hour 20 of 22 otherwise starts over,
            # which is the same argument that made the fetch resumable, only more expensive here.
            for p in sorted(glob.glob(os.path.join(s.dir, "chunk_*.npz"))):
                try:
                    z = np.load(p)
                    s.chunks.append(dict(path=os.path.basename(p), j0=int(z["j0"]),
                                         rows=int(len(z["ids"])), bytes=int(os.path.getsize(p))))
                except Exception:
                    print(f"  resume: ignoring unreadable {os.path.basename(p)} (partial write?)")
                    break

    def resume_at(s):
        """First teacher row NOT yet stored. 1 when nothing is on disk (row 0 has no predictive context)."""
        return max([c["j0"] + c["rows"] for c in s.chunks], default=1)
    def _flush(s):
        if s.j0 is None: return
        p = os.path.join(s.dir, f"chunk_{len(s.chunks):04d}.npz")   # continues past adopted chunks on resume
        np.savez(p, ids=s.buf_ids, pq=s.buf_pq, j0=np.int64(s.j0))
        s.chunks.append(dict(path=os.path.basename(p), j0=int(s.j0), rows=int(len(s.buf_ids)),
                             bytes=int(os.path.getsize(p))))
        s.buf_ids = s.buf_pq = None; s.j0 = None
    def add_rows(s, j, ids, pq):
        """ids (R,K) int32, pq (R,K) uint8 for consecutive rows j..j+R-1 (vectorized; no per-row Python)."""
        if s.j0 is None: s.j0 = j; s.buf_ids = []; s.buf_pq = []
        assert s.j0 + sum(len(b) for b in s.buf_ids) == j, "rows must arrive in order"
        s.buf_ids.append(ids); s.buf_pq.append(pq)
        if sum(len(b) for b in s.buf_ids) >= s.rpc:
            s.buf_ids = np.concatenate(s.buf_ids); s.buf_pq = np.concatenate(s.buf_pq); s._flush()
    def close(s, meta):
        if s.buf_ids is not None and len(s.buf_ids):
            s.buf_ids = np.concatenate(s.buf_ids); s.buf_pq = np.concatenate(s.buf_pq); s._flush()
        meta = dict(meta, K=K, chunks=s.chunks, total_bytes=sum(c["bytes"] for c in s.chunks))
        with open(os.path.join(s.dir, "manifest.json"), "w") as f: json.dump(meta, f, indent=2)
        return meta


SLICE_SEED = 20260719          # sealed in the rung-1 pre-registration (the seal date, so an auditor can verify
                               # the seed PRECEDES the corpus it samples). Never chosen at sampling time.


def slice_windows(starts, frac, seed):
    """Uniform random subsample of the scoring windows, plus the fingerprint that makes it auditable.

    WHY uniform and WHY declared: which tokens carry teacher signal must not become a hidden second variable.
    Any value-based selection ("highest-value code") would confound the KD-vs-CE comparison with a data-quality
    choice, and the criterion for it is undefined in the plan anyway. That experiment is a separate one, later.

    Returns (selected_starts, fingerprint_dict). The fingerprint goes in the manifest and is re-verified at every
    training start, so a corpus that silently changed under a run cannot go unnoticed.
    """
    st = np.asarray(sorted(starts), dtype=np.int64)
    if frac >= 1.0:
        sel = st
    else:
        rng = np.random.default_rng(seed)                     # seed only: reproducible from the prereg alone
        k = max(1, int(round(len(st) * frac)))
        sel = np.sort(rng.choice(len(st), size=k, replace=False))
        sel = st[sel]
    h = hashlib.sha256(sel.tobytes()).hexdigest()
    return sel.tolist(), dict(slice_frac=float(frac), slice_seed=int(seed),
                              n_windows_total=int(len(st)), n_windows_selected=int(len(sel)),
                              slice_sha256=h)


def windows(n, ctx, stride):
    """Uniform-length windows covering [1,n): the last one is CLAMPED to start at n-ctx rather than truncated, so
    every row is scored exactly once (prev_end skips the overlap) and every batch element has the same length."""
    L = min(ctx, n)
    st, seen = [], set()
    for b in range(0, max(n - 1, 1), stride):
        b = min(b, n - L)
        if b not in seen: seen.add(b); st.append(b)
        if b == n - L: break
    return st, L


def preflight(model, tids, dev, L, ctx_windows=2):
    """Verify the alignment BEFORE spending the hours, not after.

    The post-scoring self-check below is the right test in the wrong place: a run that is going to fail
    it fails it after producing a full logit store. This is the same test on two windows, costing
    seconds, positioned so that a bad configuration cannot buy sixteen hours of garbage first.

    The concrete trap it was built for, stated precisely because the first diagnosis was wrong. Measured
    on this teacher at ctx=2048:

        eager + fp16  ->  NaN logits, hit@32 = 13%      sdpa + fp16  ->  clean, hit@32 = 97.8%
        eager + bf16  ->  clean, hit@32 = 97.6%         sdpa + bf16  ->  clean, hit@32 = 97.9%

    So it is the COMBINATION, not fp16: SDPA accumulates the softmax in fp32 and is stable, eager is
    not. This producer uses sdpa and was therefore never exposed -- but eager is the default elsewhere
    in this repo (teacher_bpb.py, set there to dodge a cutlass crash), so the combination is reachable.
    NaN propagates into topk as arbitrary indices, and the stored file is then well-formed, the right
    size, and meaningless: no exception, no warning, only a low hit@K that reads like a data problem.
    """
    import torch
    n = min(len(tids), ctx_windows * L)
    x = torch.from_numpy(np.stack([tids[i * L:(i + 1) * L] for i in range(max(1, n // L))]
                                  ).astype(np.int64)).to(dev)
    with torch.no_grad():
        lg = model(x).logits.float()
    nan = bool(torch.isnan(lg).any())
    lp = torch.log_softmax(lg, -1)
    tgt = x[:, 1:]
    _, ti = lp[:, :-1].topk(K, dim=-1)
    hit = 100.0 * float((ti == tgt.unsqueeze(-1)).any(-1).float().mean())
    pt = float(lp[:, :-1].gather(-1, tgt.unsqueeze(-1)).exp().mean())
    print(f"  preflight        hit@{K} = {hit:.2f}%  mean p_true = {pt:.3f}  nan={nan}", flush=True)
    if nan or hit < 70.0:
        raise SystemExit(
            f"PREFLIGHT FAILED (hit@{K}={hit:.2f}%, nan={nan}). Refusing to score.\n"
            f"  A healthy teacher puts the true next token in its own top-{K} for ~95% of positions.\n"
            f"  Most likely cause: --quant fp16 on this teacher, which overflows to NaN. Use bf16 on\n"
            f"  Ampere or newer; fp16 is only for Turing (T4), which has no bf16.\n"
            f"  Second cause: the teacher-id stream does not match the tokenizer that produced it.")


def run_hf(a, tids, W):
    import torch
    from transformers import AutoModelForCausalLM
    gpu = pick_gpu(a.quant == "bf16")
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.quant]
    if a.quant == "fp16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("  NOTE: --quant fp16 on a bf16-capable device. Safe with sdpa (used here), NOT with eager "
              "attention -- see preflight(). bf16 is the better default on Ampere+.", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.teacher, torch_dtype=dt, attn_implementation="sdpa",
                                                 device_map={"": gpu} if gpu is not None else None)
    model.eval()
    dev = model.device
    preflight(model, tids, dev, a.ctx)
    n = len(tids)
    bs = a.batch
    starts, L = windows(n, a.ctx, a.stride)
    starts, _fp = slice_windows(starts, a.slice_frac, a.slice_seed)
    prev_end = W.resume_at()
    if prev_end > 1:
        print(f"  RESUME: {len(W.chunks)} chunk(s) on disk, {prev_end-1} rows already stored; "
              f"continuing at teacher row {prev_end}", flush=True)
    t0 = time.time(); scored = 0
    with torch.no_grad():
        starts = [b for b in starts if b + L > prev_end]      # drop windows wholly already stored
        for bi in range(0, len(starts), bs):
            grp = starts[bi:bi + bs]
            x = torch.from_numpy(np.stack([tids[b:b + L] for b in grp]).astype(np.int64)).to(dev)
            # only the tail of each window carries NEW rows -> ask the lm_head for those positions only. The full
            # (B, L, 151665) logits tensor is what OOMs the card; this is what lets the batch grow.
            keep = min(L, max((b + L) - max(prev_end, b + 1) + 1 for b in grp))
            logits = model(x, logits_to_keep=keep).logits               # (B, keep, Vt)
            off = L - keep                                              # kept position i corresponds to window i+off
            for gi, b in enumerate(grp):
                end = b + L
                lo = max(prev_end, b + 1)                              # rows j newly scored by this window
                if lo >= end: continue
                sl = logits[gi, lo - b - 1 - off:end - b - 1 - off]    # logits[i] predicts token b+i+1
                # top-K probs WITHOUT materializing the full (rows x 151665) softmax: topk is order-preserving, so
                # take it on the logits and normalize with the row logsumexp. This is the throughput lever -- the
                # full-vocab softmax, not the batch, is what saturates the card.
                lse = torch.logsumexp(sl.float(), dim=-1, keepdim=True)
                tv, ti = sl.topk(K, dim=-1)
                pv = torch.exp(tv.float() - lse)
                W.add_rows(lo, ti.to(torch.int32).cpu().numpy(),
                           (pv.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy())
                scored += end - lo; prev_end = end
            del logits
            if bi % (20 * bs) == 0:
                el = time.time() - t0
                print(f"  scored {scored}/{n} teacher tok  {scored/max(el,1e-9):.0f} tok/s  ({el:.0f}s)", flush=True)
    return scored, time.time() - t0


def run_vllm(a, tids, W):
    """Linux/Kaggle path. Same windows, same output; vLLM returns prompt_logprobs (top-K) per prompt position."""
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.teacher, dtype=a.quant, gpu_memory_utilization=0.90, max_model_len=a.ctx,
              enforce_eager=False)
    n = len(tids)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=K, temperature=0.0)
    starts, L = windows(n, a.ctx, a.stride)
    starts, _fp = slice_windows(starts, a.slice_frac, a.slice_seed)
    prev_end = W.resume_at()
    if prev_end > 1:
        print(f"  RESUME: {len(W.chunks)} chunk(s) on disk, {prev_end-1} rows already stored; "
              f"continuing at teacher row {prev_end}", flush=True)
    t0 = time.time(); scored = 0
    for bi in range(0, len(starts), a.batch):
        grp = starts[bi:bi + a.batch]
        outs = llm.generate([dict(prompt_token_ids=tids[b:b + L].tolist()) for b in grp], sp)
        for b, o in zip(grp, outs):
            end = b + L; lo = max(prev_end, b + 1)
            if lo >= end: continue
            pl = o.prompt_logprobs                                     # list[pos] -> {id: Logprob}; pos0 = None
            R = end - lo
            ids = np.zeros((R, K), dtype=np.int32); pq = np.zeros((R, K), dtype=np.uint8)
            for r, j in enumerate(range(lo, end)):
                for c, (tid, lp) in enumerate(sorted(pl[j - b].items(), key=lambda kv: -kv[1].logprob)[:K]):
                    ids[r, c] = tid; pq[r, c] = min(255, max(0, round(math.exp(lp.logprob) * 255)))
            W.add_rows(lo, ids, pq)
            scored += R; prev_end = end
    return scored, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--quant", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--rows-per-chunk", type=int, default=1_000_000)   # ~160 MB/chunk
    ap.add_argument("--slice-frac", type=float, default=1.0,
                    help="score only this fraction of the windows, sampled UNIFORMLY at random. Both arms of the "
                         "screening then train on the covered slice, so KD-vs-CE is not confounded with "
                         "subset-vs-full-corpus.")
    ap.add_argument("--slice-seed", type=int, default=SLICE_SEED,
                    help="sealed in the pre-registration; do not choose this at sampling time")
    ap.add_argument("--max-tok", type=int, default=0, help="cap teacher tokens (smoke)")
    ap.add_argument("--resume", action="store_true",
                    help="adopt chunks already on disk and continue. The long scoring run must survive a crash, a reboot or a closed laptop.")
    ap.add_argument("--data-dir", default="", help="where teacher_ids_<tag>.i32 lives and logits are written; "
                                                   "rung-1 data is outside the MVE tree")
    a = ap.parse_args()

    out_dir = a.data_dir or OUT
    tids = np.fromfile(os.path.join(out_dir, f"teacher_ids_{a.tag}.i32"), dtype=np.int32)
    if a.max_tok: tids = tids[:a.max_tok]
    print(f"[B] teacher={a.teacher} backend={a.backend} quant={a.quant} ctx={a.ctx} stride={a.stride} batch={a.batch}")
    print(f"    scoring {len(tids)} teacher tokens (TEACHER-FORCED prefill; no autoregressive decode)", flush=True)

    # recomputed here (deterministic in starts/frac/seed) so the FINGERPRINT lands in the manifest: the trainer
    # re-verifies it against the value sealed in the pre-registration at every start.
    _st, _L = windows(len(tids), a.ctx, a.stride)
    _sel, fp = slice_windows(_st, a.slice_frac, a.slice_seed)
    print(f"    slice: {fp['n_windows_selected']}/{fp['n_windows_total']} windows "
          f"(frac={fp['slice_frac']}, seed={fp['slice_seed']}) sha256={fp['slice_sha256'][:16]}...", flush=True)

    W = ChunkWriter(out_dir, a.tag, a.rows_per_chunk, resume=a.resume)
    scored, el = (run_hf if a.backend == "hf" else run_vllm)(a, tids, W)
    meta = W.close(dict(teacher=a.teacher, backend=a.backend, quant=a.quant, ctx=a.ctx, stride=a.stride,
                        batch=a.batch, n_teacher_tok=int(len(tids)), n_scored=int(scored),
                        scoring_tok_s=scored / max(el, 1e-9), wall_s=el, **fp))
    # ---- self-check: the stored row j must be the distribution PREDICTING teacher token j. If the off-by-one were
    # wrong, the true token would stop showing up in its own top-K. hit@32 on a real teacher must be high (~90%+).
    hits = tot = 0; ptrue = 0.0
    for c in meta["chunks"][:2]:
        z = np.load(os.path.join(W.dir, c["path"]))
        ids, pq, j0 = z["ids"], z["pq"], int(z["j0"])
        true = tids[j0:j0 + len(ids)]
        m = (ids == true[:, None])
        hits += int(m.any(1).sum()); tot += len(ids)
        ptrue += float((pq * m).sum()) / 255.0
    hit = 100.0 * hits / max(tot, 1)
    gb = meta["total_bytes"] / 2**30
    print(f"\n==== (B) teacher logits [{a.tag}] ====")
    print(f"  self-check       true token in its own top-{K}: {hit:.1f}%  (mean p_true={ptrue/max(tot,1):.3f}) "
          f"{'OK' if hit > 70 else '<< SUSPECT: off-by-one in the row->position map'}")
    print(f"  scored           {scored} rows in {el:.0f}s")
    print(f"  SCORING tok/s    {scored/max(el,1e-9):.0f}   <- the KD economic input (supersedes batch-1 gen tok/s)")
    stored = sum(c["rows"] for c in meta["chunks"])      # ALL rows on disk, not just this run's -- on a
    print(f"  chunks           {len(meta['chunks'])} x <= {a.rows_per_chunk} rows | {gb:.3f} GB total "
          f"({meta['total_bytes']/max(stored,1):.0f} B/teacher-token over {stored} stored rows)")
    print(f"  manifest         {os.path.join(W.dir,'manifest.json')}")
    print("STOP. (B) built. No commit.")


if __name__ == "__main__":
    main()
