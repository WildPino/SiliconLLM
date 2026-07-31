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
SUMCHUNK = 1 << 24            # elements per pass of the byte-length sum; bounds its temporary at ~128 MiB

_BPE = None
def _enc(b): return _BPE.encode(b)
def _init(bpe):
    global _BPE; _BPE = bpe


def peak_gib():
    """Peak working set of THIS process (the Pool children hold one block each and are not counted).

    The 5.5 GB build died twice without ever saying how much memory it had taken, which is why the second
    wall -- twelve lines below the first, in the same file -- was found by reading rather than by running.
    A projected peak that nobody measures is a projection of one array, not of the path."""
    try:
        if os.name == "nt":
            import ctypes, ctypes.wintypes as wt
            class PMC(ctypes.Structure):
                _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            # argtypes/restype are NOT optional here. GetCurrentProcess returns the pseudo-handle -1; with
            # ctypes' default int marshalling that is truncated to 32 bits on a 64-bit HANDLE, the call
            # fails, and the struct stays zeroed. The first version of this function had no argtypes and
            # printed "0.0 GiB" for a build that used tens of gigabytes -- a plausible artefact, not an
            # error, which is the failure mode this project keeps paying for. Hence the return check.
            k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
            k32.GetCurrentProcess.restype = wt.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
            psapi.GetProcessMemoryInfo.restype = wt.BOOL
            c = PMC(); c.cb = ctypes.sizeof(PMC)
            if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                raise OSError(ctypes.get_last_error())
            return c.PeakWorkingSetSize / 2**30
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    except Exception:
        return float("nan")


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


def build_raw(target_bytes, tag, out=None):
    """Concatenate training-split documents until the byte target is met. Deterministic: shard order is
    sorted and document order within a shard is the fetch order."""
    out = out or OUT
    p = os.path.join(out, f"raw_{tag}.bin")
    bp = os.path.join(out, f"docbound_{tag}.i64")
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
    os.makedirs(out, exist_ok=True)
    open(p, "wb").write(raw)
    bd = np.array(bounds, dtype=np.int64); bd.tofile(bp)
    print(f"[A1] raw = {len(raw)/2**20:.1f} MiB from {nd} training docs "
          f"({nskip} reserved-band docs skipped)  sha {hashlib.sha256(raw).hexdigest()[:16]}...")
    return raw, bd


def total_gib():
    """Physical RAM, so the projection below is measured against the machine rather than a constant."""
    try:
        if os.name == "nt":
            import ctypes, ctypes.wintypes as wt
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS(); m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 2**30
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except Exception:
        return float("nan")


def teacher_tokens(raw, tag, jobs, out=None):
    """Teacher ids + the byte offset each one starts at. Vocabulary-independent, so this runs ONCE and
    both student-vocab arms share it.

    THIS FUNCTION STILL ACCUMULATES IN PYTHON LISTS. That is deliberate: it is the code that produced the
    s0 artifacts, and the identity gate for the ids rewrite compares against them, so rewriting it here
    would put a second unproven change inside the control. It is also why it is now behind
    --kd-apparatus, which no arm at this rung sets. The projection below refuses rather than letting it
    die at 2000 s for the third time; the remedy it names is the same one applied to the student ids."""
    out = out or OUT
    ip = os.path.join(out, f"teacher_ids_{tag}.i32"); sp = os.path.join(out, f"teacher_start_{tag}.i64")
    if os.path.isfile(ip) and os.path.isfile(sp):
        tids = np.fromfile(ip, dtype=np.int32); tstart = np.fromfile(sp, dtype=np.int64)
        print(f"[A2] teacher tokens reused ({len(tids)} tok)")
        return tids, tstart
    # 4.005 B/teacher-tok is the s0 measurement; 76 B/tok is sys.getsizeof(int) x2 plus two list slots.
    proj = (len(raw) / 4.005) * 76 / 2**30
    tot = total_gib()
    print(f"[A2] projected peak for the list accumulator: {proj:.1f} GiB on a {tot:.0f} GiB machine")
    if proj > 0.6 * tot:
        sys.exit(f"ERROR: teacher tokenization would need ~{proj:.0f} GiB of Python list objects for "
                 f"{len(raw)/2**30:.1f} GiB of corpus, against {tot:.0f} GiB of RAM. It will be killed "
                 f"partway and write nothing (the arrays are only flushed on completion).\n"
                 f"  REMEDY: stream tids/tstart to their output files per block, as build_vocab now does "
                 f"for the student ids, and read them back with np.fromfile.\n"
                 f"  This is NOT silenced by lowering --gb: that changes the artifact. If you do not need "
                 f"the KD apparatus, drop --kd-apparatus and this function is not called at all.")
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


def build_vocab(raw, tids, tstart, V, tag, jobs, docbound, out=None, kd=True):
    """Student ids for ONE student vocabulary; anchors and the projection tables only when `kd`.

    MEMORY. Every array here is O(corpus). At 0.5 GB that is invisible and at 5.5 GB it is the whole
    problem, so the shape of each one is stated where it is allocated. `ids` stays uint16 -- the dtype it
    has on disk -- because promoting it to int64 quadruples the resident cost of the one array that must
    survive the whole function, and nothing downstream needs the width."""
    out = out or OUT
    bpe = Bpe.load(os.path.join(BPEDIR, f"bpe{V}_code.bin"))
    N = len(raw)
    ip = os.path.join(out, f"ids_V{V}_{tag}.u16")
    if os.path.isfile(ip):
        ids = np.fromfile(ip, dtype=np.uint16)
    else:
        # STREAMED. The previous form was `parts = pool.map(...)` followed by a flattening
        # comprehension: a list of every student token as a Python int (~30 B each, plus an 8 B pointer),
        # and briefly TWO of them, since the comprehension is fully materialized before np.array consumes
        # it. At 2.3 B tokens that is ~77 GiB per copy. imap yields blocks IN ORDER, so writing each block
        # as it arrives produces the identical byte stream while never holding more than one block.
        t0 = time.time(); ntok = 0
        with open(ip + ".part", "wb") as fh:
            with Pool(jobs, initializer=_init, initargs=(bpe,)) as pool:
                for part in pool.imap(_enc, nl_blocks(raw, BLOCK), chunksize=1):
                    a = np.asarray(part, dtype=np.uint16)
                    a.tofile(fh); ntok += len(a)
        os.replace(ip + ".part", ip)
        print(f"[A3/V{V}] encoded {ntok} student tok in {time.time()-t0:.0f}s  (peak {peak_gib():.1f} GiB)")
        ids = np.fromfile(ip, dtype=np.uint16)
    ntok = len(ids)

    # The byte-length invariant: the student tokens must re-expand to exactly the corpus. Summed in
    # chunks because `exp_len[ids]` materializes an int64 array as long as the corpus (17 GiB at 5.5 GB)
    # and np.cumsum then materializes a second one. Under --kd-apparatus the running sum IS the prefix
    # sum's last element; the full prefix is built only where it has a consumer, which is `anchors`.
    exp_len = np.array(bpe.exp_len, dtype=np.int64)
    tot = 0
    for i in range(0, ntok, SUMCHUNK):
        tot += int(exp_len[ids[i:i + SUMCHUNK]].sum())
    assert tot == N, (tot, N)

    if not kd:
        print(f"[A3/V{V}] {ntok} student tok, {N/ntok:.3f} B/tok  (peak {peak_gib():.1f} GiB); "
              f"KD apparatus SKIPPED")
        return dict(V=V, n_student_tok=int(ntok), bytes_per_student_tok=N / ntok, kd_apparatus=False)

    ends = np.cumsum(exp_len[ids])
    assert ends[-1] == N, (ends[-1], N)

    # anchors: student p is an anchor iff student token p+1 starts where a teacher token starts
    b2j = np.full(N + 1, -1, dtype=np.int64); b2j[tstart] = np.arange(len(tids))
    anchors = b2j[ends[:-1]]
    anchors = np.where(anchors > 0, anchors, -1)
    nanc = int((anchors >= 0).sum())
    anchors.astype(np.int32).tofile(os.path.join(out, f"anchors_V{V}_{tag}.i32"))

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
    t2s.tofile(os.path.join(out, f"t2s_V{V}_{tag}.i32"))
    np.savez(os.path.join(out, f"decomp_V{V}_{tag}.npz"), tok=dtok, len=dlen, S=np.int32(S))

    return dict(V=V, n_student_tok=int(ntok), bytes_per_student_tok=N / ntok, kd_apparatus=True,
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
    ap.add_argument("--vocabs", default="2048,4096",
                    help="which student vocabularies to materialize. Both by default, which is what the "
                         "stage-1 A/B needed. That A/B is DECIDED (V=2048 adopted by the sealed sigma_seed "
                         "rule), so building V4096 for the main run is half the work for an arm nobody will "
                         "run -- pass --vocabs 2048.")
    ap.add_argument("--out", default=OUT, help="output directory; overridable so the identity gate can "
                                               "build into a scratch tree without touching the real one")
    ap.add_argument("--kd-apparatus", action="store_true",
                    help="also build the teacher tokens, anchors, t2s and decomp. OFF by default: the "
                         "main run and all three branches of prereg section 6 are CE, so nothing reads "
                         "them, and building them costs a teacher tokenization pass plus the arrays that "
                         "do not fit at 5.5 GB. A rung-2 KD reopening rebuilds the corpus anyway -- "
                         "SCALEUP_ARCHITECTURE section 5.1 requires randomized membership, and these "
                         "arrays are functions of the sequential order the order probe priced at +0.0339.")
    a = ap.parse_args()
    vocabs = tuple(int(v) for v in a.vocabs.split(","))
    tag = a.tag or f"{a.gb:g}gb"
    out = a.out
    os.makedirs(out, exist_ok=True)

    raw, docbound = build_raw(int(a.gb * 2**30), tag, out)
    tids = tstart = None
    if a.kd_apparatus:
        tids, tstart = teacher_tokens(raw, tag, a.jobs, out)
    rows = [build_vocab(raw, tids, tstart, V, tag, a.jobs, docbound, out, a.kd_apparatus) for V in vocabs]

    N = len(raw)
    print(f"\n==== rung-1 data [{tag}] ====")
    tea = (f"teacher {len(tids)} tok ({N/len(tids):.3f} B/tok)" if a.kd_apparatus
           else "teacher tokens NOT built (--kd-apparatus off)")
    print(f"  corpus {N/2**20:.1f} MiB, {len(docbound)-1} docs, {tea}\n")
    if a.kd_apparatus:
        print(f"  {'V':>6} {'student tok':>13} {'B/tok':>7} {'anchors':>11} {'anchor%':>8} "
              f"{'seg stu':>8} {'seg=1':>7} {'seg tea':>8}")
        for r in rows:
            print(f"  {r['V']:6d} {r['n_student_tok']:13d} {r['bytes_per_student_tok']:7.3f} "
                  f"{r['anchors']:11d} {100*r['anchor_frac']:7.2f}% {r['seg_student_tok_mean']:8.2f} "
                  f"{100*r['seg_stu_eq1_frac']:6.1f}% {r['seg_teacher_tok_mean']:8.2f}")
    else:
        print(f"  {'V':>6} {'student tok':>13} {'B/tok':>7}")
        for r in rows:
            print(f"  {r['V']:6d} {r['n_student_tok']:13d} {r['bytes_per_student_tok']:7.3f}")
    meta = dict(corpus="TheStackV2-dedup/Python strict-permissive (rung-1 slice)", bytes=N,
                teacher=TEACHER if a.kd_apparatus else None,
                n_teacher_tok=int(len(tids)) if a.kd_apparatus else None,
                bytes_per_teacher_tok=(N / len(tids)) if a.kd_apparatus else None,
                kd_apparatus=bool(a.kd_apparatus),
                n_docs=int(len(docbound) - 1), tag=tag, vocabs=rows,
                raw_sha256=hashlib.sha256(raw).hexdigest())
    json.dump(meta, open(os.path.join(out, f"meta_{tag}.json"), "w"), indent=1)
    print(f"\n  wrote {os.path.join(out, f'meta_{tag}.json')}")
    print(f"  PEAK working set for this build: {peak_gib():.1f} GiB "
          f"(machine has {total_gib():.0f} GiB)")
    print("\nSTOP. Numbers above; scoring has NOT been run. No commit.")


if __name__ == "__main__":
    main()
