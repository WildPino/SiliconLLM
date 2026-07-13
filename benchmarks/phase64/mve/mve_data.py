#!/usr/bin/env python3
# Phase 64.4 / MVE — component (A): data + tokenizer + the span-mapping ANCHOR table.
#   magic: "MVEA" 0x4D564541
#
# Builds, from the pinned TinyStories corpus, everything the KD path needs:
#   1. raw bytes        <- decode results/phase55/ids.u16 through its meta (the pilot corpus, 64 MiB)
#   2. student BPE      <- cartography.train_bpe V=1024 on a 300 KB sample -- THE SAME CLASS/RECIPE as the P62 code
#                          BPE. We deliberately do NOT reuse P55's own id stream: retokenizing with the production
#                          Bpe means the MVE exercises the *production* tokenizer path (train/encode/exp_len/anchor),
#                          which is the whole point of a minimum viable EXERCISE. Same V=1024, same recipe.
#   3. teacher tokens   <- Qwen2.5-Coder-1.5B fast tokenizer with offsets -> byte-space start of every teacher token
#   4. ANCHORS          <- student position p is an anchor iff the byte where student token p+1 STARTS is also the
#                          start of a teacher token j. Then the teacher's predictive distribution FOR token j is a
#                          legitimate target for the student's prediction at p. anchors[p] = j, else -1.
#   5. t2s              <- projection table teacher_id -> student FIRST token id of its bytes. This is what turns a
#                          teacher top-K over the teacher vocab into a distribution over the student vocab.
#
# Declared approximations (they are the D3 mechanism on trial, not hidden costs):
#   - t2s encodes each teacher token's bytes IN ISOLATION. In context, a student merge could cross the teacher token
#     boundary, so an alternative's true first-token may differ. The GROUND-TRUTH student token at an anchor always
#     comes from the real corpus stream, so only the K-1 counterfactual alternatives carry this approximation.
#   - teacher tokenization is done in newline-aligned ~1 MB blocks (offsets are exact; only the tokenizer's context
#     resets at block edges -- negligible at 64 MiB, and the scoring windows are ctx=2048 anyway).
#
# Run:  .venv/Scripts/python.exe benchmarks/phase64/mve/mve_data.py [--smoke]
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase62"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase55"))
from cartography import Bpe, train_bpe          # noqa: E402
from phase55_ssm import load_meta               # noqa: E402

OUT = os.path.join(ROOT, "results", "phase64", "mve")
TEACHER = "Qwen/Qwen2.5-Coder-1.5B"
VSTU = 1024
BPE_SAMPLE = 300_000        # same sample size the P62 code BPE was trained on
BLOCK = 1 << 20             # tokenizer block (newline-aligned)

_BPE = None
def _enc_block(b):
    return _BPE.encode(b)
def _init(bpe):
    global _BPE; _BPE = bpe


def nl_blocks(raw, size):
    """Split raw bytes into ~size blocks, each ending on a newline (so BPE whitespace-chunks never straddle)."""
    out, i, n = [], 0, len(raw)
    while i < n:
        j = min(i + size, n)
        if j < n:
            k = raw.rfind(b"\n", i, j)
            j = k + 1 if k > i else j
        out.append(raw[i:j]); i = j
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 MB slice, everything else identical")
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = "smoke" if a.smoke else "full"

    # ---- 1. raw bytes -------------------------------------------------------------------------
    t0 = time.time()
    V55, el55, id2b = load_meta(os.path.join(ROOT, "results", "phase55", "meta.bin"))
    ids55 = np.fromfile(os.path.join(ROOT, "results", "phase55", "ids.u16"), dtype=np.uint16)
    raw = b"".join(id2b[i] for i in ids55)
    if a.smoke: raw = raw[:2 << 20]
    N = len(raw)
    print(f"[A1] TinyStories raw = {N} bytes ({N/2**20:.1f} MiB)  ({time.time()-t0:.0f}s)", flush=True)

    # ---- 2. student BPE (production class/recipe) ---------------------------------------------
    bp = os.path.join(OUT, f"bpe{VSTU}_ts.bin")
    if os.path.exists(bp):
        bpe = Bpe.load(bp); print(f"[A2] student BPE V={bpe.vocab} loaded {bp}")
    else:
        t0 = time.time(); bpe = train_bpe(raw[:BPE_SAMPLE], VSTU, verbose=False); bpe.save(bp)
        print(f"[A2] student BPE V={bpe.vocab} trained on {BPE_SAMPLE}B in {time.time()-t0:.0f}s -> {bp}")

    # ---- 3. encode corpus (parallel over newline-aligned blocks; deterministic, order-preserving)
    ip = os.path.join(OUT, f"ts_{tag}.u16")
    if os.path.exists(ip):
        ids = np.fromfile(ip, dtype=np.uint16).astype(np.int64)
        print(f"[A3] student ids loaded {ip} ({len(ids)} tok)")
    else:
        t0 = time.time(); blocks = nl_blocks(raw, BLOCK)
        with Pool(a.jobs, initializer=_init, initargs=(bpe,)) as p:
            parts = p.map(_enc_block, blocks, chunksize=1)
        ids = np.array([t for part in parts for t in part], dtype=np.int64)
        ids.astype(np.uint16).tofile(ip)
        print(f"[A3] encoded {len(ids)} student tok in {time.time()-t0:.0f}s ({a.jobs} jobs) -> {ip}")
    exp_len = np.array(bpe.exp_len, dtype=np.int64)
    ends = np.cumsum(exp_len[ids])                       # byte offset AFTER student token p
    assert ends[-1] == N, (ends[-1], N)
    ntok = len(ids)

    # ---- 4. teacher tokens + byte starts -------------------------------------------------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.teacher)
    t0 = time.time(); tids = []; tstart = []; base = 0
    for blk in nl_blocks(raw, BLOCK):
        txt = blk.decode("utf-8", errors="strict")
        enc = tok(txt, add_special_tokens=False, return_offsets_mapping=True)   # chat-template OFF
        cb = np.zeros(len(txt) + 1, dtype=np.int64); acc = 0
        for i, ch in enumerate(txt):
            acc += len(ch.encode("utf-8")); cb[i + 1] = acc
        for tid, (cs, ce) in zip(enc["input_ids"], enc["offset_mapping"]):
            if ce <= cs: continue
            tids.append(tid); tstart.append(base + int(cb[cs]))
        base += len(blk)
    tids = np.array(tids, dtype=np.int32); tstart = np.array(tstart, dtype=np.int64)
    tids.tofile(os.path.join(OUT, f"teacher_ids_{tag}.i32"))
    tstart.tofile(os.path.join(OUT, f"teacher_start_{tag}.i64"))
    print(f"[A4] teacher tok = {len(tids)} ({N/len(tids):.2f} B/tok) in {time.time()-t0:.0f}s")

    # ---- 5. anchors: student p -> teacher j starting at the same byte -------------------------
    # student token p+1 starts at byte ends[p]; the target of the student's prediction at p.
    t0 = time.time()
    b2j = np.full(N + 1, -1, dtype=np.int64); b2j[tstart] = np.arange(len(tids))
    anchors = b2j[ends[:-1]]                                     # (ntok-1,) teacher index or -1
    anchors = np.where(anchors > 0, anchors, -1)                 # j=0 has no predictive row (no left context) -> drop
    nanc = int((anchors >= 0).sum())
    anchors.astype(np.int32).tofile(os.path.join(OUT, f"anchors_{tag}.i32"))
    print(f"[A5] anchors = {nanc}/{ntok-1} student positions ({100.0*nanc/(ntok-1):.2f}%)  ({time.time()-t0:.0f}s)")

    # ---- 6. projection tables ------------------------------------------------------------------
    # t2s     : teacher token -> student FIRST token of its bytes   (the SEALED anchor-KD projection)
    # decomp  : teacher token -> the FULL student token sequence of its bytes (first S), + its length.
    #           This is what a prefix-conditioned span KD needs, and it costs nothing extra to build or to store:
    #           it is derived from the same teacher vocabulary, and the teacher logit rows are untouched. t2s is
    #           just decomp[:,0], so the sealed design remains available bit-for-bit.
    S = 8
    t0 = time.time(); Vt = int(tok.vocab_size) + len(tok.get_added_vocab())
    t2s = np.full(Vt, -1, dtype=np.int32); nmap = 0
    dtok = np.zeros((Vt, S), dtype=np.int32); dlen = np.zeros(Vt, dtype=np.int8)
    for t in range(Vt):
        try: b = tok.convert_tokens_to_string([tok.convert_ids_to_tokens(t)]).encode("utf-8")
        except Exception: b = b""
        if not b: continue
        e = bpe.encode(b)
        if not e: continue
        t2s[t] = e[0]; nmap += 1
        L = min(len(e), S); dtok[t, :L] = e[:L]; dlen[t] = L
    t2s.tofile(os.path.join(OUT, f"t2s_{tag}.i32"))
    np.savez(os.path.join(OUT, f"decomp_{tag}.npz"), tok=dtok, len=dlen, S=np.int32(S))
    ml = float(dlen[dlen > 0].mean())
    print(f"[A6] t2s: {nmap}/{Vt} teacher tokens map to a student first-token  ({time.time()-t0:.0f}s)")
    print(f"[A6] decomp: mean {ml:.2f} student tokens per teacher token (S={S} cap) -> the span-KD read of the SAME rows")

    # ---- 7. split + meta ------------------------------------------------------------------------
    ntr = int(ntok * 0.9)
    meta = dict(corpus="TinyStories(P55)", bytes=N, teacher=a.teacher, V_student=int(bpe.vocab),
                n_student_tok=int(ntok), n_teacher_tok=int(len(tids)), bytes_per_student_tok=N / ntok,
                bytes_per_teacher_tok=N / len(tids), anchors=nanc, anchor_frac=nanc / (ntok - 1),
                t2s_mapped=nmap, t2s_vocab=Vt, n_train_tok=ntr, n_val_tok=ntok - ntr, tag=tag,
                seg_bytes=N / max(nanc, 1), seg_student_tok=ntok / max(nanc, 1), seg_teacher_tok=len(tids) / max(nanc, 1))
    with open(os.path.join(OUT, f"meta_{tag}.json"), "w") as f: json.dump(meta, f, indent=2)
    print(f"\n==== (A) MVE data ready [{tag}] ====")
    for k, v in meta.items(): print(f"  {k:24s} {v}")
    print(f"\n  KD anchor rate = {100*meta['anchor_frac']:.1f}% of student positions; segments tile 100% of bytes at "
          f"{meta['seg_bytes']:.1f} B / {meta['seg_student_tok']:.1f} student tok / {meta['seg_teacher_tok']:.1f} teacher tok.")
    print("STOP. (A) built. No commit.")


if __name__ == "__main__":
    main()
