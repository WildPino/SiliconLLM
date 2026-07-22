#!/usr/bin/env python3
"""WS3 / rung-1 component (A) on the REAL corpus: student ids + teacher tokens + the span-mapping anchors.

The code-corpus counterpart of `mve/mve_data.py`. Same construction, same file formats (the trainer and
the logit producer read these unchanged); three things differ, and each is deliberate:

  corpus   the Python training split, with the reserved validation band excluded via
           `fetch_corpus.is_val` -- imported, never re-derived.
  BPE      loaded, not trained: the tokenizers come from the pipeline and were retrained on the
           training split. Training a tokenizer here would silently fork the vocabulary away from the
           one in the corpus manifest.
  vocab    built for BOTH V=2048 and V=4096, because anchors and the projection tables are functions of
           the student vocabulary. The TEACHER tokens are not -- they are scored once and shared by both
           arms, which is why the logits were stored per teacher token in the first place.

Documents are joined with a trailing newline and the boundaries are recorded: a document boundary is a
place where the teacher's context is meaningless, and a scoring window that straddles one is measuring
the teacher's confusion rather than the corpus.

Run: python benchmarks/phase64/data/code_data.py --gb 0.5
"""
import argparse, glob, hashlib, json, os, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase62"))
from fetch_corpus import is_val                  # noqa: E402
from cartography import Bpe                      # noqa: E402

RAW = os.path.join(ROOT, "data", "phase64", "raw_python")
BPEDIR = os.path.join(ROOT, "data", "phase64", "corpus")
OUT = os.path.join(ROOT, "results", "phase64", "rung1")
TEACHER = "Qwen/Qwen2.5-Coder-1.5B"
BLOCK = 1 << 20
S = 8

_BPE = None
def _enc(b): return _BPE.encode(b)
def _init(bpe):
    global _BPE; _BPE = bpe


def nl_blocks(raw, size):
    """~size blocks, each ending on a newline, so BPE whitespace chunks never straddle a block edge."""
    out, i, n = [], 0, len(raw)
    while i < n:
        j = min(i + size, n)
        if j < n:
            k = raw.rfind(b"\n", i, j)
            j = k + 1 if k > i else j
        out.append(raw[i:j]); i = j
    return out


def build_raw(target_bytes, tag):
    """Concatenate training-split documents until the byte target is met. Deterministic: shard order is
    sorted and document order within a shard is the fetch order."""
    p = os.path.join(OUT, f"raw_{tag}.bin")
    bp = os.path.join(OUT, f"docbound_{tag}.i64")
    if os.path.isfile(p) and os.path.isfile(bp):
        raw = open(p, "rb").read()
        print(f"[A1] raw reused {p} ({len(raw)/2**20:.1f} MiB)")
        return raw, np.fromfile(bp, dtype=np.int64)
    parts, bounds, nb, nd, nskip = [], [0], 0, 0, 0
    for sh in sorted(glob.glob(os.path.join(RAW, "shard*.jsonl"))):
        for line in open(sh, encoding="utf-8"):
            d = json.loads(line)
            if is_val(d["blob_id"]): nskip += 1; continue
            b = d["text"].encode("utf-8")
            if not b.endswith(b"\n"): b += b"\n"
            parts.append(b); nb += len(b); nd += 1; bounds.append(nb)
            if nb >= target_bytes: break
        if nb >= target_bytes: break
    raw = b"".join(parts)
    os.makedirs(OUT, exist_ok=True)
    open(p, "wb").write(raw)
    bd = np.array(bounds, dtype=np.int64); bd.tofile(bp)
    print(f"[A1] raw = {len(raw)/2**20:.1f} MiB from {nd} training docs "
          f"({nskip} reserved-band docs skipped)  sha {hashlib.sha256(raw).hexdigest()[:16]}...")
    return raw, bd


def teacher_tokens(raw, tag, jobs):
    """Teacher ids + the byte offset each one starts at. Vocabulary-independent, so this runs ONCE and
    both student-vocab arms share it."""
    ip = os.path.join(OUT, f"teacher_ids_{tag}.i32"); sp = os.path.join(OUT, f"teacher_start_{tag}.i64")
    if os.path.isfile(ip) and os.path.isfile(sp):
        tids = np.fromfile(ip, dtype=np.int32); tstart = np.fromfile(sp, dtype=np.int64)
        print(f"[A2] teacher tokens reused ({len(tids)} tok)")
        return tids, tstart
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEACHER, token=os.environ.get("HF_TOKEN"))
    t0 = time.time(); tids, tstart, base = [], [], 0
    for k, blk in enumerate(nl_blocks(raw, BLOCK)):
        txt = blk.decode("utf-8", errors="strict")
        enc = tok(txt, add_special_tokens=False, return_offsets_mapping=True)
        cb = np.zeros(len(txt) + 1, dtype=np.int64); acc = 0
        for i, ch in enumerate(txt):
            acc += len(ch.encode("utf-8")); cb[i + 1] = acc
        for tid, (cs, ce) in zip(enc["input_ids"], enc["offset_mapping"]):
            if ce <= cs: continue
            tids.append(tid); tstart.append(base + int(cb[cs]))
        base += len(blk)
        if k % 50 == 0:
            print(f"     block {k}  {len(tids)} teacher tok  {time.time()-t0:.0f}s", flush=True)
    tids = np.array(tids, dtype=np.int32); tstart = np.array(tstart, dtype=np.int64)
    tids.tofile(ip); tstart.tofile(sp)
    print(f"[A2] teacher tok = {len(tids)} ({len(raw)/len(tids):.3f} B/tok) in {time.time()-t0:.0f}s")
    return tids, tstart


def build_vocab(raw, tids, tstart, V, tag, jobs, docbound):
    """Student ids, anchors and the projection tables for ONE student vocabulary."""
    bpe = Bpe.load(os.path.join(BPEDIR, f"bpe{V}_code.bin"))
    N = len(raw)
    ip = os.path.join(OUT, f"ids_V{V}_{tag}.u16")
    if os.path.isfile(ip):
        ids = np.fromfile(ip, dtype=np.uint16).astype(np.int64)
    else:
        t0 = time.time()
        with Pool(jobs, initializer=_init, initargs=(bpe,)) as pool:
            parts = pool.map(_enc, nl_blocks(raw, BLOCK), chunksize=1)
        ids = np.array([t for part in parts for t in part], dtype=np.int64)
        ids.astype(np.uint16).tofile(ip)
        print(f"[A3/V{V}] encoded {len(ids)} student tok in {time.time()-t0:.0f}s")
    exp_len = np.array(bpe.exp_len, dtype=np.int64)
    ends = np.cumsum(exp_len[ids])
    assert ends[-1] == N, (ends[-1], N)
    ntok = len(ids)

    # anchors: student p is an anchor iff student token p+1 starts where a teacher token starts
    b2j = np.full(N + 1, -1, dtype=np.int64); b2j[tstart] = np.arange(len(tids))
    anchors = b2j[ends[:-1]]
    anchors = np.where(anchors > 0, anchors, -1)
    nanc = int((anchors >= 0).sum())
    anchors.astype(np.int32).tofile(os.path.join(OUT, f"anchors_V{V}_{tag}.i32"))

    # segment decomposition: how many student tokens does one anchored segment actually span?
    # This is what the span chain-rule consumes. If segments are ~1 student token the span read degrades
    # to the anchor read, which is the 13-15% degenerate case already measured and rejected.
    ap = np.flatnonzero(anchors >= 0)
    seg_stu = np.diff(ap) if len(ap) > 1 else np.array([0])
    seg_tea = np.diff(anchors[ap]) if len(ap) > 1 else np.array([0])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEACHER, token=os.environ.get("HF_TOKEN"))
    Vt = int(tok.vocab_size) + len(tok.get_added_vocab())
    t2s = np.full(Vt, -1, dtype=np.int32); dtok = np.zeros((Vt, S), dtype=np.int32)
    dlen = np.zeros(Vt, dtype=np.int8); nmap = 0
    for t in range(Vt):
        try: b = tok.convert_tokens_to_string([tok.convert_ids_to_tokens(t)]).encode("utf-8")
        except Exception: b = b""
        if not b: continue
        e = bpe.encode(b)
        if not e: continue
        t2s[t] = e[0]; nmap += 1
        L = min(len(e), S); dtok[t, :L] = e[:L]; dlen[t] = L
    t2s.tofile(os.path.join(OUT, f"t2s_V{V}_{tag}.i32"))
    np.savez(os.path.join(OUT, f"decomp_V{V}_{tag}.npz"), tok=dtok, len=dlen, S=np.int32(S))

    return dict(V=V, n_student_tok=int(ntok), bytes_per_student_tok=N / ntok,
                anchors=nanc, anchor_frac=nanc / (ntok - 1),
                seg_student_tok_mean=float(seg_stu.mean()), seg_student_tok_med=float(np.median(seg_stu)),
                seg_teacher_tok_mean=float(seg_tea.mean()),
                seg_stu_eq1_frac=float((seg_stu == 1).mean()),
                t2s_mapped=nmap, t2s_vocab=Vt,
                mean_decomp=float(dlen[dlen > 0].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=0.5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    a = ap.parse_args()
    tag = a.tag or f"{a.gb:g}gb"
    os.makedirs(OUT, exist_ok=True)

    raw, docbound = build_raw(int(a.gb * 2**30), tag)
    tids, tstart = teacher_tokens(raw, tag, a.jobs)
    rows = [build_vocab(raw, tids, tstart, V, tag, a.jobs, docbound) for V in (2048, 4096)]

    N = len(raw)
    print(f"\n==== rung-1 data [{tag}] ====")
    print(f"  corpus {N/2**20:.1f} MiB, {len(docbound)-1} docs, teacher {len(tids)} tok "
          f"({N/len(tids):.3f} B/tok)\n")
    print(f"  {'V':>6} {'student tok':>13} {'B/tok':>7} {'anchors':>11} {'anchor%':>8} "
          f"{'seg stu':>8} {'seg=1':>7} {'seg tea':>8}")
    for r in rows:
        print(f"  {r['V']:6d} {r['n_student_tok']:13d} {r['bytes_per_student_tok']:7.3f} "
              f"{r['anchors']:11d} {100*r['anchor_frac']:7.2f}% {r['seg_student_tok_mean']:8.2f} "
              f"{100*r['seg_stu_eq1_frac']:6.1f}% {r['seg_teacher_tok_mean']:8.2f}")
    meta = dict(corpus="TheStackV2-dedup/Python strict-permissive (rung-1 slice)", bytes=N,
                teacher=TEACHER, n_teacher_tok=int(len(tids)), bytes_per_teacher_tok=N / len(tids),
                n_docs=int(len(docbound) - 1), tag=tag, vocabs=rows,
                raw_sha256=hashlib.sha256(raw).hexdigest())
    json.dump(meta, open(os.path.join(OUT, f"meta_{tag}.json"), "w"), indent=1)
    print(f"\n  wrote {os.path.join(OUT, f'meta_{tag}.json')}")
    print("\nSTOP. Numbers above; scoring has NOT been run. No commit.")


if __name__ == "__main__":
    main()
