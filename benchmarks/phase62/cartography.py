#!/usr/bin/env python3
# Phase 62 / Task 2 - UNIT CARTOGRAPHY (mantra-guard, BEFORE committing to a unit).
#
#   The BPE-1024 was chosen for TinyStories PROSE. This measures, per Cat-A domain, whether that
#   choice transfers or whether the silicon would prefer a different unit -- the judge stays BPB-in-BYTES
#   regardless of unit (a unit that needs fewer bits/byte to model the SAME bytes is the better unit).
#
#   Three units, ONE machinery (the only variable is the merge table -> honest apples-to-apples):
#     (1) byte-raw          : vocab 256, no merges (fertility == 1 tok/byte by definition).
#     (2) BPE-1024-TinyStories : the project's existing weights/bpe1024.bin (magic 'BPE1'), AS-IS.
#     (3) BPE-1024-relearned   : same byte-level algorithm (space/non-space chunk + greedy lowest-rank
#                                merge), retrained on THIS domain's sample -> drop-in for the C engine's
#                                bpe_codec.h if the Architect picks it.
#
#   Per (domain x unit) we report: fertility (tok/byte), single-byte-fallback frac + mean tok len
#   (coverage signal), pure-digit token stats (the log timestamp/number saturation the Architect flagged),
#   and TWO train-free BPB PROJECTIONS: order-0 (unigram) and order-1 (bigram-conditional) bits/byte.
#   The order-1 bits/byte is the key relative proxy: it is a FLOOR-style proxy (a real SSM captures longer
#   range, so trained BPB < this), valid for RANKING units within a domain, NOT as the final BPB.
#
#   No gate (characterization). No commit. Freezes results/phase62/corpus_manifest.json (sha256 of the
#   EXACT sampled bytes + upstream git commits) so the measurement is reproducible against a pinned artifact.
#
# Run (smoke, fast): .venv/Scripts/python.exe benchmarks/phase62/cartography.py --smoke
# Run (full)       : .venv/Scripts/python.exe benchmarks/phase62/cartography.py
import os, sys, struct, json, math, argparse, hashlib, subprocess, time
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXT  = os.path.join(ROOT, "data", "external")
BPE_TS = os.path.join(ROOT, "weights", "bpe1024.bin")
OUT  = os.path.join(ROOT, "results", "phase62")
try: os.makedirs(OUT, exist_ok=True)      # read-only mount: import for symbols must still succeed
except OSError: pass

BPE1_MAGIC = 0x42504531  # 'BPE1' (little-endian u32), same as archive/benchmarks/phase50/bpe_codec.h

# ------------------------------------------------------------------ codec (port of bpe_codec.h) --
def is_space(c): return c in (0x20, 0x09, 0x0A, 0x0D)  # ' ' '\t' '\n' '\r'

def chunk_spans(buf):
    """Yield (start,end) of maximal space / non-space runs (the C pre-tokenization)."""
    i, n = 0, len(buf)
    while i < n:
        j = i; sp = is_space(buf[i])
        while j < n and is_space(buf[j]) == sp: j += 1
        yield i, j; i = j

class Bpe:
    """Byte-level BPE: id<256 == raw byte, id>=256 == learned merge. Greedy leftmost-lowest-rank encode
       (bit-identical to the C engine's bpe_encode_region)."""
    def __init__(self, nmerge, mA, mB):
        self.nmerge = nmerge; self.mA = mA; self.mB = mB; self.vocab = 256 + nmerge
        self.rank = {(mA[r], mB[r]): r for r in range(nmerge)}
        # token byte-length expansions (only lengths needed for accounting)
        self.exp_len = [1] * 256
        self.exp_bytes = [bytes([t]) for t in range(256)]
        for r in range(nmerge):
            a, b = mA[r], mB[r]
            self.exp_bytes.append(self.exp_bytes[a] + self.exp_bytes[b])
            self.exp_len.append(self.exp_len[a] + self.exp_len[b])

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            m, v, n = struct.unpack("<III", f.read(12))
            assert m == BPE1_MAGIC, f"bad bpe magic {m:#x}"
            mA = [0]*n; mB = [0]*n
            for r in range(n): mA[r], mB[r] = struct.unpack("<II", f.read(8))
        return cls(n, mA, mB)

    @classmethod
    def byte_raw(cls): return cls(0, [], [])

    def save(self, path):
        with open(path, "wb") as f:
            f.write(struct.pack("<III", BPE1_MAGIC, self.vocab, self.nmerge))
            for r in range(self.nmerge): f.write(struct.pack("<II", self.mA[r], self.mB[r]))

    def _encode_chunk(self, s):
        rank = self.rank
        while True:
            best = 1 << 30; bp = -1
            for t in range(len(s) - 1):
                r = rank.get((s[t], s[t+1]))
                if r is not None and r < best: best = r; bp = t
            if bp < 0: break
            s = s[:bp] + [256 + best] + s[bp+2:]
        return s

    def encode(self, buf):
        out = []
        for a, b in chunk_spans(buf):
            out.extend(self._encode_chunk(list(buf[a:b])))
        return out

# ------------------------------------------------------------------ trainer (same algorithm) ------
def train_bpe(buf, target_vocab=1024, verbose=False):
    """Sennrich-style BPE over whitespace-chunk frequencies, byte-level start, producing rank-ordered
       merges consumed by the SAME greedy encoder above. nmerge = target_vocab - 256."""
    words = Counter()
    for a, b in chunk_spans(buf): words[buf[a:b]] += 1
    wl = [list(w) for w in words.keys()]         # symbol lists
    wf = list(words.values())                    # frequencies
    nmerge = target_vocab - 256
    mA, mB = [], []
    # incremental global pair stats + inverted index word->set(pairs) rebuilt lazily
    def pair_stats():
        st = Counter()
        for sym, f in zip(wl, wf):
            for i in range(len(sym) - 1): st[(sym[i], sym[i+1])] += f
        return st
    stats = pair_stats()
    for r in range(nmerge):
        if not stats: break
        (a, b), _ = max(stats.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
        nid = 256 + r; mA.append(a); mB.append(b)
        # apply merge to affected words, delta-update stats
        for wi, sym in enumerate(wl):
            if a not in sym: continue
            i = 0; changed = False
            while i < len(sym) - 1:
                if sym[i] == a and sym[i+1] == b:
                    f = wf[wi]
                    if i > 0:
                        stats[(sym[i-1], a)] -= f; stats[(sym[i-1], nid)] += f
                    if i + 2 < len(sym):
                        stats[(b, sym[i+2])] -= f; stats[(nid, sym[i+2])] += f
                    stats[(a, b)] -= f
                    sym[i:i+2] = [nid]; changed = True
                else:
                    i += 1
            if changed: wl[wi] = sym
        # prune zeros occasionally
        if r % 64 == 0: stats = Counter({k: v for k, v in stats.items() if v > 0})
        if verbose and r % 128 == 0: print(f"    merge {r}/{nmerge} pair=({a},{b})", file=sys.stderr)
    return Bpe(len(mA), mA, mB)

# ------------------------------------------------------------------ metrics -----------------------
def entropy_bits(counter):
    tot = sum(counter.values());
    if tot == 0: return 0.0
    return -sum((c/tot) * math.log2(c/tot) for c in counter.values())

def cond_entropy_bits(ids):
    """order-1 conditional entropy H(next|cur) in bits/token."""
    uni = Counter(ids); big = defaultdict(Counter)
    for i in range(len(ids) - 1): big[ids[i]][ids[i+1]] += 1
    tot = len(ids) - 1
    if tot <= 0: return 0.0
    H = 0.0
    for cur, nxt in big.items():
        pc = uni[cur] / tot  # approx marginal over positions that have a successor
        H += pc * entropy_bits(nxt)
    return H

def analyze(name, codec, data):
    t0 = time.time(); ids = codec.encode(data); enc_s = time.time() - t0
    n_bytes = len(data); n_tok = len(ids)
    fert = n_tok / n_bytes
    lens = [codec.exp_len[i] for i in ids]
    single = sum(1 for l in lens if l == 1) / n_tok
    mean_len = sum(lens) / n_tok
    # digit saturation: pure-ASCII-digit tokens
    def is_digit_tok(i): b = codec.exp_bytes[i]; return len(b) > 0 and all(48 <= c <= 57 for c in b)
    digit_toks = [i for i in ids if is_digit_tok(i)]
    digit_frac = len(digit_toks) / n_tok
    digit_mean_len = (sum(codec.exp_len[i] for i in digit_toks) / len(digit_toks)) if digit_toks else 0.0
    uni = Counter(ids)
    H0 = entropy_bits(uni); H1 = cond_entropy_bits(ids)
    return dict(name=name, n_bytes=n_bytes, n_tok=n_tok, fertility=fert, bytes_per_tok=1/fert,
                single_byte_frac=single, mean_tok_len=mean_len, vocab_used=len(uni), vocab_size=codec.vocab,
                digit_frac=digit_frac, digit_mean_len=digit_mean_len,
                bpb0=H0*fert, bpb1=H1*fert, H0_bits_tok=H0, H1_bits_tok=H1, enc_s=enc_s)

# ------------------------------------------------------------------ corpus sampling ---------------
def git_commit(path):
    try: return subprocess.check_output(["git","-C",path,"rev-parse","HEAD"], text=True).strip()
    except Exception: return "n/a"

def sample_code(max_bytes):
    """Deterministic .py sample from cpython + django (permissive: PSF / BSD-3)."""
    roots = [os.path.join(EXT,"the_stack_python","cpython"), os.path.join(EXT,"the_stack_python","django")]
    files = []
    for root in roots:
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.endswith(".py"): files.append(os.path.join(dp, fn))
    files.sort()  # deterministic order
    buf = bytearray(); used = []
    for fp in files:
        if len(buf) >= max_bytes: break
        try: b = open(fp, "rb").read()
        except Exception: continue
        buf += b + b"\n"; used.append(fp)
    return bytes(buf[:max_bytes]), used, [git_commit(r) for r in roots]

def sample_log(max_bytes):
    """BGL + Linux LogHub samples (restricted per Architect; research-use, manifest-only, never released)."""
    srcs = [os.path.join(EXT,"log_corpus","loghub","BGL","BGL_2k.log"),
            os.path.join(EXT,"log_corpus","loghub","Linux","Linux_2k.log")]
    buf = bytearray(); used = []
    for fp in srcs:
        if os.path.exists(fp): buf += open(fp,"rb").read() + b"\n"; used.append(fp)
    return bytes(buf[:max_bytes]), used, [git_commit(os.path.join(EXT,"log_corpus","loghub"))]

def sample_prose(max_bytes):
    """TinyStories anchor (the already-measured prose point of the structure gradient; source of ids.u16)."""
    fp = os.path.join(ROOT, "data", "corpora", "tinystories_64mb.txt")
    if not os.path.exists(fp): return b"", [], []
    return open(fp, "rb").read(max_bytes), [fp], []

def sha(b): return hashlib.sha256(b).hexdigest()

# ------------------------------------------------------------------ main --------------------------
def fmt_row(r):
    return (f"  {r['name']:<22} fert={r['fertility']:.3f} tok/B  B/tok={r['bytes_per_tok']:.2f}  "
            f"1byte={r['single_byte_frac']*100:4.1f}%  meanlen={r['mean_tok_len']:.2f}  "
            f"digit={r['digit_frac']*100:4.1f}%(len{r['digit_mean_len']:.2f})  "
            f"bpb0={r['bpb0']:.3f}  bpb1={r['bpb1']:.3f}  vocab_used={r['vocab_used']}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny samples + tiny retrain (fast sanity)")
    ap.add_argument("--code-bytes", type=int, default=3_000_000)
    ap.add_argument("--log-bytes",  type=int, default=1_000_000)
    ap.add_argument("--prose-bytes",type=int, default=1_000_000)
    ap.add_argument("--train-bytes",type=int, default=1_500_000, help="sample size for retraining BPE")
    a = ap.parse_args()
    if a.smoke:
        a.code_bytes=a.log_bytes=a.prose_bytes=120_000; a.train_bytes=120_000

    bpe_ts = Bpe.load(BPE_TS)
    print(f"BPE-TinyStories loaded: vocab={bpe_ts.vocab} nmerge={bpe_ts.nmerge}", file=sys.stderr)

    domains = {}
    domains["code"]  = sample_code(a.code_bytes)
    domains["log"]   = sample_log(a.log_bytes)
    domains["prose"] = sample_prose(a.prose_bytes)

    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "policy": "corpora NEVER committed/released; manifest+scripts+hash only (data/ gitignored). "
                          "LogHub = research-use (cite Zhu et al. ISSRE 2023); log-trained ckpts research-only, "
                          "out of release assets by default. Split (Task3): by-REPO hash (all files of a repo -> "
                          "one split) to avoid intra-repo leakage.",
                "domains": {}}
    print("\n==== PHASE 62 / TASK 2 - UNIT CARTOGRAPHY (train-free BPB projections; order-1 = ranking proxy, not final BPB) ====")
    for dom, (data, used, commits) in domains.items():
        if len(data) == 0:
            print(f"\n[{dom}] NO DATA on disk -- skipped"); continue
        # retrain BPE on this domain's own sample
        train_slice = data[:a.train_bytes]
        t0=time.time(); bpe_re = train_bpe(train_slice, 1024, verbose=not a.smoke); tr=time.time()-t0
        rows = [analyze("byte-raw", Bpe.byte_raw(), data),
                analyze("BPE-1024-TinyStories", bpe_ts, data),
                analyze("BPE-1024-relearned", bpe_re, data)]
        print(f"\n[{dom}]  sample={len(data)} bytes  ({len(used)} src files)  retrain={tr:.1f}s on {len(train_slice)}B")
        for r in rows: print(fmt_row(r))
        # save the relearned tokenizer for the engine (if picked)
        re_path = os.path.join(OUT, f"bpe1024_{dom}.bin"); bpe_re.save(re_path)
        manifest["domains"][dom] = {
            "sample_bytes": len(data), "sha256_sample": sha(data),
            "n_src_files": len(used), "src_first": used[:3], "upstream_commits": commits,
            "relearned_bpe": os.path.relpath(re_path, ROOT), "sha256_relearned_bpe": sha(open(re_path,'rb').read()),
            "metrics": {r["name"]: {k: r[k] for k in ("fertility","single_byte_frac","mean_tok_len",
                        "digit_frac","digit_mean_len","bpb0","bpb1","vocab_used")} for r in rows}}
    mpath = os.path.join(OUT, "corpus_manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=2)
    print(f"\nmanifest (frozen hashes) -> {os.path.relpath(mpath, ROOT)}")
    print("STOP (unit cartography). No gate, no commit. Architect chooses the unit; MM commits manifest before Task-3 run.")

if __name__ == "__main__":
    main()
