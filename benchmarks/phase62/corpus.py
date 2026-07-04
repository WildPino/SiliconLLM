#!/usr/bin/env python3
# Phase 62 / corpus builder (Task-1 decision + Task-2 adjustments). Assembles the Cat-A training corpora,
#   relearns the PER-DOMAIN BPE-1024 on the TRAIN split only (no val leakage), tokenizes to ids+meta in the
#   phase55 format (so the existing trainer path is reused), and FREEZES results/phase62/corpus_manifest.json.
#
#   CODE (primary): 6 permissive repos, DIRECT git clones (cleaner than the HF gated pull -- upstream license,
#     pinned commit): cpython(PSF) django(BSD-3) flask(BSD-3) requests(Apache-2) pip(MIT) numpy(BSD-3).
#   Split (Architect fix): held-out by TOP-LEVEL DIRECTORY across all repos (~10% -> val). by-repo degenerates
#     to domain-shift with few repos; file-level would leak sibling boilerplate. Group = (repo, first path
#     component); a whole group goes to one split -> in-distribution val, zero file-level leakage.
#   LOG (secondary, research-use, manifest-only NEVER released): LogHub BGL+Linux. Block-dedup = cap repeats of
#     any exact line to K (logs are natively redundant; uncapped -> BPB is inflated by copy-paste). % reported.
#     NB: only the *_2k.log samples are on disk; FULL BGL+Linux need the Zenodo pull (command printed at STOP).
#
# Build (smoke): .venv/Scripts/python.exe benchmarks/phase62/corpus.py --smoke
# Build (full) : .venv/Scripts/python.exe benchmarks/phase62/corpus.py
import os, sys, json, struct, time, hashlib, subprocess, argparse, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cartography import train_bpe, Bpe, chunk_spans

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXT  = os.path.join(ROOT, "data", "external")
OUT  = os.path.join(ROOT, "results", "phase62"); os.makedirs(OUT, exist_ok=True)

CODE_REPOS = [  # (name, relpath under the_stack_python, license)
    ("cpython",  "cpython",  "PSF-2.0"),   ("django",   "django",   "BSD-3-Clause"),
    ("flask",    "flask",    "BSD-3-Clause"), ("requests", "requests", "Apache-2.0"),
    ("pip",      "pip",      "MIT"),        ("numpy",    "numpy",    "BSD-3-Clause"),
]
LOG_SRCS = [("BGL", "BGL/BGL_2k.log"), ("Linux", "Linux/Linux_2k.log")]

def git_commit(path):
    try: return subprocess.check_output(["git","-C",path,"rev-parse","HEAD"], text=True).strip()
    except Exception: return "n/a"
def sha(b): return hashlib.sha256(b).hexdigest()
def h10(s): return zlib.crc32(s.encode()) % 10          # deterministic 0..9 bucket

# ------------------------------------------------------------------ CODE ---------------------------
def build_code(cap_train, cap_val, val_bucket=0):
    """Return (train_bytes, val_bytes, meta). Split held-out by (repo, top-level-dir) group hash."""
    groups = {}  # key -> list[path]
    repo_meta = []
    for name, rel, lic in CODE_REPOS:
        root = os.path.join(EXT, "the_stack_python", rel)
        if not os.path.isdir(root): continue
        repo_meta.append({"repo": name, "license": lic, "commit": git_commit(root)})
        for dp, _, fns in os.walk(root):
            if os.sep + ".git" in dp: continue
            for fn in fns:
                if not fn.endswith(".py"): continue
                fp = os.path.join(dp, fn); rp = os.path.relpath(fp, root)
                top = rp.split(os.sep)[0] if os.sep in rp else "__root__"
                groups.setdefault((name, top), []).append(fp)
    tr, va, n_tr_g, n_va_g = bytearray(), bytearray(), 0, 0
    for key in sorted(groups):
        is_val = (h10(f"{key[0]}/{key[1]}") == val_bucket)
        dst = va if is_val else tr
        n_va_g += is_val; n_tr_g += (not is_val)
        for fp in sorted(groups[key]):
            if len(tr) >= cap_train and len(va) >= cap_val: break
            if (is_val and len(va) >= cap_val) or (not is_val and len(tr) >= cap_train): continue
            try: dst += open(fp, "rb").read() + b"\n"
            except Exception: pass
    meta = {"repos": repo_meta, "n_groups_train": n_tr_g, "n_groups_val": n_va_g,
            "split_rule": "held-out by (repo, top-level-dir) group; crc32%%10==%d -> val" % val_bucket}
    return bytes(tr[:cap_train]), bytes(va[:cap_val]), meta

# ------------------------------------------------------------------ LOG ----------------------------
def block_dedup(data, cap=5):
    """Cap exact-line repeats to `cap` occurrences (logs are natively redundant). Returns (bytes, pct_removed)."""
    seen = {}; out = []; kept = 0; total = 0
    for line in data.split(b"\n"):
        total += 1; c = seen.get(line, 0)
        if c < cap: out.append(line); kept += 1
        seen[line] = c + 1
    ded = b"\n".join(out)
    pct = 100.0 * (1 - len(ded) / max(len(data), 1))
    return ded, pct

def build_log(cap_train, cap_val, dedup_cap=5):
    raw = bytearray(); present = []
    for name, rel in LOG_SRCS:
        fp = os.path.join(EXT, "log_corpus", "loghub", rel)
        if os.path.exists(fp): raw += open(fp, "rb").read() + b"\n"; present.append(name)
    ded, pct = block_dedup(bytes(raw), dedup_cap)
    # split: contiguous tail 10% -> val (held-out region, no template overlap with train head)
    cut = int(len(ded) * 0.9)
    tr, va = ded[:cut][:cap_train], ded[cut:][:cap_val]
    meta = {"sources": present, "commit": git_commit(os.path.join(EXT, "log_corpus", "loghub")),
            "dedup_line_cap": dedup_cap, "dedup_pct_removed": round(pct, 2),
            "note": "ONLY *_2k.log samples on disk; FULL BGL+Linux via Zenodo (see STOP)"}
    return tr, va, meta

# ------------------------------------------------------------------ tokenize + write ---------------
def write_meta(path, bpe):
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x54444D50, bpe.vocab, 0))     # magic 'PMDT'(phase55), V, nt(0=unused here)
        f.write(bytes(bpe.exp_len[i] for i in range(bpe.vocab)))
        for i in range(bpe.vocab): f.write(bpe.exp_bytes[i])

def tokenize_write(dom, train, val, train_meta, bpe_train_bytes, manifest):
    t0 = time.time(); bpe = train_bpe(train[:bpe_train_bytes], 1024, verbose=False); ttr = time.time()-t0
    bpe.save(os.path.join(OUT, f"bpe1024_{dom}.bin"))
    import numpy as np
    def enc_to(ids_path, data):
        ids = np.asarray(bpe.encode(data), dtype="<u2"); ids.tofile(ids_path); return len(ids)
    t1 = time.time()
    ntr = enc_to(os.path.join(OUT, f"{dom}_train.u16"), train)
    nva = enc_to(os.path.join(OUT, f"{dom}_val.u16"),   val)
    write_meta(os.path.join(OUT, f"{dom}.meta"), bpe)
    tenc = time.time()-t1
    fert = (ntr+nva) / max(len(train)+len(val), 1)
    print(f"[{dom}] train={len(train)}B/{ntr}tok  val={len(val)}B/{nva}tok  fert={fert:.3f} tok/B  "
          f"(bpe {ttr:.1f}s, enc {tenc:.1f}s)")
    manifest["domains"][dom] = {**train_meta,
        "train_bytes": len(train), "val_bytes": len(val), "train_tok": ntr, "val_tok": nva,
        "fertility": round(fert, 4), "sha256_train": sha(train), "sha256_val": sha(val),
        "bpe": f"results/phase62/bpe1024_{dom}.bin", "sha256_bpe": sha(open(os.path.join(OUT,f'bpe1024_{dom}.bin'),'rb').read())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--code-train", type=int, default=14_000_000); ap.add_argument("--code-val", type=int, default=1_500_000)
    ap.add_argument("--log-train",  type=int, default=2_000_000);  ap.add_argument("--log-val",  type=int, default=400_000)
    ap.add_argument("--bpe-train",  type=int, default=3_000_000)
    a = ap.parse_args()
    if a.smoke:
        a.code_train=a.log_train=300_000; a.code_val=a.log_val=60_000; a.bpe_train=300_000

    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy": "corpora NEVER committed/released (data/ gitignored); manifest+scripts+hash only. LogHub=research-use "
                  "(cite Zhu et al. ISSRE 2023); log-trained ckpts research-only, out of release assets by default. "
                  "Unit=BPE-1024 relearned PER-DOMAIN on TRAIN split (foundation recipe: dReLU-gated-ternary MLP, fp32 proj).",
        "domains": {}}
    print("==== PHASE 62 corpus build (per-domain BPE relearn, top-level-dir split, log block-dedup) ====")
    c_tr, c_va, c_meta = build_code(a.code_train, a.code_val)
    print(f"code: {c_meta['n_groups_train']} train-groups / {c_meta['n_groups_val']} val-groups over {len(c_meta['repos'])} repos")
    tokenize_write("code", c_tr, c_va, c_meta, a.bpe_train, manifest)
    l_tr, l_va, l_meta = build_log(a.log_train, a.log_val)
    print(f"log: dedup removed {l_meta['dedup_pct_removed']}%  sources={l_meta['sources']}")
    tokenize_write("log", l_tr, l_va, l_meta, a.bpe_train, manifest)
    # manifest lives in the COMMITTABLE scripts dir (results/ is gitignored with the corpora it indexes)
    json.dump(manifest, open(os.path.join(os.path.dirname(__file__), "corpus_manifest.json"), "w"), indent=2)
    print(f"\nmanifest FROZEN -> benchmarks/phase62/corpus_manifest.json (committable; corpora stay gitignored)")
    print("Zenodo FULL logs (Task-3, user-launched heavy pull), then re-run this builder:")
    print("  BGL:   https://zenodo.org/records/8196385/files/BGL.zip     Linux: https://zenodo.org/records/8196385/files/Linux.zip")
    print("STOP (corpus built + frozen). No commit -> MM commits manifest+scripts BEFORE the Task-3 run.")

if __name__ == "__main__":
    main()
