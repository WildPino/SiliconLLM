"""
Phase 31: External Compressor Atlas

Compares SEE V1.0 (general + prose profiles) against classical compressors:
  zlib-9, bz2-9, lzma, zstd-3, zstd-22, brotli-11

Corpus: 14 files covering 7 domains (internal + external Phase 29C clean catalog).

Metrics per (corpus, compressor):
  bpb         true bits per byte (actual compressed size / original size * 8)
  encode_ms   encode wall-clock time
  decode_ms   decode wall-clock time

Output:
  results/phase31_atlas.json  — full structured data
  results/phase31_atlas.csv   — flat table for spreadsheet
  Console: formatted atlas table + research Q&A

Usage:
  python scripts/phase31_atlas.py
  python scripts/phase31_atlas.py --skip-see      (classical compressors only)
  python scripts/phase31_atlas.py --corpus c_real.c prose_real.txt  (subset)
"""

import os
import re
import sys
import json
import math
import time
import bz2
import lzma
import zlib
import hashlib
import argparse
import tempfile
import subprocess
import io
from collections import Counter

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    print("[WARN] zstandard not installed — zstd compressors skipped", flush=True)

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEE     = os.path.join(ROOT, "see.exe")
WEIGHTS = os.path.join(ROOT, "weights", "entropy_weights_factors_r16.bin")
DATA    = os.path.join(ROOT, "data")
EXT_DIR = os.path.join(DATA, "external")
RESULTS = os.path.join(ROOT, "results")

TMP_ENC = os.path.join(ROOT, "tmp_31.see")
TMP_DEC = os.path.join(ROOT, "tmp_31.dec")
TMP_IN  = os.path.join(ROOT, "tmp_31.in")
TMP_BRO = os.path.join(ROOT, "tmp_31.br")

# ---------------------------------------------------------------------------
# Corpus catalog
# ---------------------------------------------------------------------------

CORPUS = [
    # (short_name, path, domain, origin)
    ("natural_text",  os.path.join(DATA, "natural_text.txt"),             "prose_synth",  "internal"),
    ("markdown_docs", os.path.join(DATA, "markdown_docs.md"),             "markdown",     "internal"),
    ("c_code",        os.path.join(DATA, "c_code.c"),                     "code",         "internal"),
    ("shuffled",      os.path.join(DATA, "shuffled.bin"),                  "random",       "internal"),
    ("json_synth",    os.path.join(DATA, "json_synth.json"),               "json",         "internal"),
    ("c_header",      os.path.join(DATA, "c_header_synth.h"),              "code",         "internal"),
    ("notes_it",      os.path.join(DATA, "project_notes_it.txt"),          "prose_it",     "internal"),
    ("md_mixed",      os.path.join(DATA, "repo_markdown_mixed.md"),        "markdown",     "internal"),
    ("log_synth",     os.path.join(DATA, "log_synth.log"),                 "log",          "internal"),
    ("promessi_sposi",os.path.join(DATA, "promessi_sposi.txt"),            "prose_it_lit", "internal"),
    ("prose_real",    os.path.join(EXT_DIR, "prose_real.txt"),             "prose_en_lit", "external"),
    ("c_real",        os.path.join(EXT_DIR, "c_real.c"),                   "code",         "external"),
    ("log_real",      os.path.join(EXT_DIR, "log_real.log"),               "log",          "external"),
    ("readme_real",   os.path.join(EXT_DIR, "readme_real.md"),             "markdown",     "external"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unigram_bpb(data: bytes) -> float:
    if not data:
        return 8.0
    counts = Counter(data)
    n = len(data)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def brotli_available() -> bool:
    try:
        r = subprocess.run(["brotli", "--version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


HAS_BROTLI = brotli_available()
if not HAS_BROTLI:
    print("[WARN] brotli CLI not found — brotli compressor skipped", flush=True)


# ---------------------------------------------------------------------------
# Classical compressors (in-memory)
# ---------------------------------------------------------------------------

def _time_classical(compress_fn, decompress_fn, data):
    t0 = time.perf_counter()
    compressed = compress_fn(data)
    enc_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    decompress_fn(compressed)
    dec_ms = (time.perf_counter() - t0) * 1000
    return len(compressed), enc_ms, dec_ms


def run_zlib(data):
    return _time_classical(
        lambda d: zlib.compress(d, level=9),
        zlib.decompress,
        data,
    )


def run_bz2(data):
    return _time_classical(
        lambda d: bz2.compress(d, compresslevel=9),
        bz2.decompress,
        data,
    )


def run_lzma(data):
    return _time_classical(
        lzma.compress,
        lzma.decompress,
        data,
    )


def run_zstd(data, level=3):
    if not HAS_ZSTD:
        return None
    cctx = zstandard.ZstdCompressor(level=level)
    dctx = zstandard.ZstdDecompressor()
    t0 = time.perf_counter()
    compressed = cctx.compress(data)
    enc_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    dctx.decompress(compressed)
    dec_ms = (time.perf_counter() - t0) * 1000
    return len(compressed), enc_ms, dec_ms


def run_brotli_cli(data):
    if not HAS_BROTLI:
        return None
    with open(TMP_IN, "wb") as f:
        f.write(data)
    t0 = time.perf_counter()
    r = subprocess.run(
        ["brotli", "--quality=11", "--force", TMP_IN, f"--output={TMP_BRO}"],
        capture_output=True, timeout=300,
    )
    enc_ms = (time.perf_counter() - t0) * 1000
    if r.returncode != 0 or not os.path.exists(TMP_BRO):
        return None
    comp_size = os.path.getsize(TMP_BRO)
    t0 = time.perf_counter()
    dec_out = TMP_BRO + ".dec"
    r2 = subprocess.run(
        ["brotli", "--decompress", "--force", TMP_BRO, f"--output={dec_out}"],
        capture_output=True, timeout=300,
    )
    dec_ms = (time.perf_counter() - t0) * 1000
    for p in [TMP_IN, TMP_BRO]:
        try:
            os.remove(p)
        except OSError:
            pass
    if os.path.exists(dec_out):
        os.remove(dec_out)
    if r2.returncode != 0:
        return None
    return comp_size, enc_ms, dec_ms


# ---------------------------------------------------------------------------
# SEE compressor
# ---------------------------------------------------------------------------

def run_see(filepath, profile_args, timeout=900):
    for p in [TMP_ENC, TMP_DEC]:
        try:
            os.remove(p)
        except OSError:
            pass

    enc_cmd = [SEE, "encode", filepath, TMP_ENC, "--weights", WEIGHTS] + profile_args
    t0 = time.perf_counter()
    r = subprocess.run(enc_cmd, capture_output=True, timeout=timeout)
    enc_ms = (time.perf_counter() - t0) * 1000

    if r.returncode != 0 or not os.path.exists(TMP_ENC):
        return None

    comp_size = os.path.getsize(TMP_ENC)

    dec_cmd = [SEE, "decode", TMP_ENC, TMP_DEC, "--weights", WEIGHTS]
    t0 = time.perf_counter()
    r2 = subprocess.run(dec_cmd, capture_output=True, timeout=timeout)
    dec_ms = (time.perf_counter() - t0) * 1000

    for p in [TMP_ENC, TMP_DEC]:
        try:
            os.remove(p)
        except OSError:
            pass

    if r2.returncode != 0:
        return None

    return comp_size, enc_ms, dec_ms


# ---------------------------------------------------------------------------
# Build compressor list
# ---------------------------------------------------------------------------

def build_compressors(skip_see=False):
    comps = []
    if not skip_see:
        comps += [
            ("SEE-general", lambda path, _: run_see(path, ["--expert-profile", "general"]), True),
            ("SEE-prose",   lambda path, _: run_see(path, ["--expert-profile", "prose"]),   True),
        ]
    comps += [
        ("zlib-9",    lambda path, data: run_zlib(data),     False),
        ("bz2-9",     lambda path, data: run_bz2(data),      False),
        ("lzma",      lambda path, data: run_lzma(data),     False),
    ]
    if HAS_ZSTD:
        comps += [
            ("zstd-3",  lambda path, data: run_zstd(data, level=3),  False),
            ("zstd-22", lambda path, data: run_zstd(data, level=22), False),
        ]
    if HAS_BROTLI:
        comps += [
            ("brotli-11", lambda path, data: run_brotli_cli(data), False),
        ]
    return comps


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_atlas(corpus_filter=None, skip_see=False):
    compressors = build_compressors(skip_see=skip_see)
    comp_names = [c[0] for c in compressors]

    atlas = {}

    for name, path, domain, origin in CORPUS:
        if corpus_filter and name not in corpus_filter:
            continue
        if not os.path.exists(path):
            print(f"[SKIP] {name}: file not found at {path}", flush=True)
            continue

        data = open(path, "rb").read()
        size = len(data)
        uni_bpb = unigram_bpb(data)
        sha = sha256_bytes(data)

        print(f"\n{'='*70}", flush=True)
        print(f"  {name}  [{domain}]  {size:,} bytes  unigram={uni_bpb:.3f} BPB", flush=True)
        print(f"{'='*70}", flush=True)

        entry = {
            "domain": domain,
            "origin": origin,
            "size_bytes": size,
            "sha256": sha,
            "unigram_bpb": round(uni_bpb, 6),
            "compressors": {},
        }

        for cname, cfn, uses_path in compressors:
            print(f"  {cname:<14}", end=" ", flush=True)
            try:
                if uses_path:
                    result = cfn(path, data)
                else:
                    result = cfn(path, data)
            except subprocess.TimeoutExpired:
                print("TIMEOUT", flush=True)
                entry["compressors"][cname] = None
                continue
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                entry["compressors"][cname] = None
                continue

            if result is None:
                print("FAILED", flush=True)
                entry["compressors"][cname] = None
                continue

            comp_size, enc_ms, dec_ms = result
            bpb = (comp_size * 8) / size
            print(f"BPB={bpb:.4f}  enc={enc_ms:6.0f}ms  dec={dec_ms:6.0f}ms", flush=True)

            entry["compressors"][cname] = {
                "bpb":        round(bpb, 6),
                "comp_size":  comp_size,
                "enc_ms":     round(enc_ms, 1),
                "dec_ms":     round(dec_ms, 1),
            }

        atlas[name] = entry

    return atlas, comp_names


# ---------------------------------------------------------------------------
# Atlas table printer
# ---------------------------------------------------------------------------

def print_atlas(atlas, comp_names):
    W = 14
    print("\n\n" + "=" * 80, flush=True)
    print("PHASE 31 — EXTERNAL COMPRESSOR ATLAS", flush=True)
    print("=" * 80, flush=True)

    header = f"{'corpus':<20} {'domain':<15} {'size':>8}  {'unigram':>7}"
    for cn in comp_names:
        header += f"  {cn:>{W}}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for name, entry in atlas.items():
        size_kb = entry["size_bytes"] / 1024
        row = f"{name:<20} {entry['domain']:<15} {size_kb:>7.1f}K  {entry['unigram_bpb']:>7.3f}"
        comp_vals = []
        for cn in comp_names:
            r = entry["compressors"].get(cn)
            if r is None:
                comp_vals.append(None)
                row += f"  {'N/A':>{W}}"
            else:
                comp_vals.append(r["bpb"])
                row += f"  {r['bpb']:>{W}.4f}"
        print(row, flush=True)

    print("\n\nENCODE TIME (ms)\n" + "-" * 60, flush=True)
    header2 = f"{'corpus':<20}"
    for cn in comp_names:
        header2 += f"  {cn:>{W}}"
    print(header2, flush=True)
    for name, entry in atlas.items():
        row = f"{name:<20}"
        for cn in comp_names:
            r = entry["compressors"].get(cn)
            row += f"  {str(round(r['enc_ms'])) if r else 'N/A':>{W}}"
        print(row, flush=True)


# ---------------------------------------------------------------------------
# Research questions
# ---------------------------------------------------------------------------

def answer_questions(atlas, comp_names):
    classical = [c for c in comp_names if not c.startswith("SEE")]
    see_general = "SEE-general"
    see_prose   = "SEE-prose"

    print("\n\n" + "=" * 80, flush=True)
    print("RESEARCH QUESTIONS", flush=True)
    print("=" * 80, flush=True)

    # Q1: SEE vince solo su shuffled/marginali o anche su certi log/config?
    print("\nQ1: On which domains does SEE-general WIN vs best classical?", flush=True)
    for name, entry in atlas.items():
        sg = entry["compressors"].get(see_general)
        if sg is None:
            continue
        see_bpb = sg["bpb"]
        classical_bpbs = [
            entry["compressors"][c]["bpb"]
            for c in classical
            if entry["compressors"].get(c) is not None
        ]
        if not classical_bpbs:
            continue
        best_classical = min(classical_bpbs)
        delta = see_bpb - best_classical
        flag = "WIN" if delta < 0 else ("TIE" if abs(delta) < 0.01 else "LOSS")
        best_name = classical[classical_bpbs.index(best_classical)]
        print(
            f"  {name:<20} [{entry['domain']:<14}]  SEE={see_bpb:.4f}  "
            f"best_classical={best_classical:.4f} ({best_name})  "
            f"delta={delta:+.4f}  {flag}",
            flush=True,
        )

    # Q2: TOKPFX regge su testi mai visti? (SEE-prose on external prose)
    print("\nQ2: TOKPFX on unseen prose — SEE-prose vs SEE-general on external prose corpora", flush=True)
    prose_domains = {"prose_en_lit", "prose_it_lit", "prose_synth", "prose_it"}
    for name, entry in atlas.items():
        if entry["domain"] not in prose_domains:
            continue
        sg = entry["compressors"].get(see_general)
        sp = entry["compressors"].get(see_prose)
        if sg is None or sp is None:
            continue
        delta = sp["bpb"] - sg["bpb"]
        origin = entry["origin"]
        print(
            f"  {name:<20} [{entry['domain']:<14}] [{origin:<8}]  "
            f"general={sg['bpb']:.4f}  prose={sp['bpb']:.4f}  delta={delta:+.4f}  "
            f"{'PROSE WINS' if delta < 0 else 'NO GAIN'}",
            flush=True,
        )

    # Q3: LZ6 basta su codice reale?
    print("\nQ3: SEE-general vs best classical on real code (external code corpora)", flush=True)
    for name, entry in atlas.items():
        if entry["domain"] != "code" or entry["origin"] != "external":
            continue
        sg = entry["compressors"].get(see_general)
        if sg is None:
            continue
        see_bpb = sg["bpb"]
        rows = []
        for c in classical:
            r = entry["compressors"].get(c)
            if r:
                rows.append((c, r["bpb"]))
        rows.sort(key=lambda x: x[1])
        print(f"  {name}:", flush=True)
        for c, b in rows:
            marker = " <-- best" if c == rows[0][0] else ""
            print(f"    {c:<14} {b:.4f}{marker}", flush=True)
        print(f"    {'SEE-general':<14} {see_bpb:.4f}  delta vs best={see_bpb - rows[0][1]:+.4f}", flush=True)

    # Q4: markdown resta il buco peggiore?
    print("\nQ4: SEE-general delta vs best classical, by domain (sorted by gap)", flush=True)
    domain_gaps = {}
    for name, entry in atlas.items():
        sg = entry["compressors"].get(see_general)
        if sg is None:
            continue
        classical_bpbs = [
            entry["compressors"][c]["bpb"]
            for c in classical
            if entry["compressors"].get(c) is not None
        ]
        if not classical_bpbs:
            continue
        domain = entry["domain"]
        delta = sg["bpb"] - min(classical_bpbs)
        if domain not in domain_gaps:
            domain_gaps[domain] = []
        domain_gaps[domain].append((name, delta))

    domain_avg = {
        d: sum(x[1] for x in rows) / len(rows)
        for d, rows in domain_gaps.items()
    }
    for domain, avg in sorted(domain_avg.items(), key=lambda x: -x[1]):
        files = ", ".join(f"{n}({d:+.3f})" for n, d in domain_gaps[domain])
        print(f"  {domain:<15}  avg_gap={avg:+.4f}  files: {files}", flush=True)

    # Q5: quale gap è più grande per BPB × volume?
    print("\nQ5: Largest absolute gap (delta_bpb × size_bytes / 8 = extra bytes vs best classical)", flush=True)
    gaps = []
    for name, entry in atlas.items():
        sg = entry["compressors"].get(see_general)
        if sg is None:
            continue
        classical_bpbs = [
            (c, entry["compressors"][c]["bpb"])
            for c in classical
            if entry["compressors"].get(c) is not None
        ]
        if not classical_bpbs:
            continue
        best_c, best_bpb = min(classical_bpbs, key=lambda x: x[1])
        delta_bpb = sg["bpb"] - best_bpb
        extra_bytes = delta_bpb * entry["size_bytes"] / 8
        gaps.append((name, entry["domain"], delta_bpb, extra_bytes, best_c))

    gaps.sort(key=lambda x: -x[3])
    for name, domain, delta_bpb, extra_bytes, best_c in gaps:
        sign = "extra" if extra_bytes > 0 else "saved"
        print(
            f"  {name:<20} [{domain:<14}]  "
            f"delta={delta_bpb:+.4f} BPB  {abs(extra_bytes):>10.0f} bytes {sign} vs {best_c}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(atlas, comp_names, path):
    import csv
    fields = ["corpus", "domain", "origin", "size_bytes", "unigram_bpb"] + \
             [f"{c}_bpb" for c in comp_names] + \
             [f"{c}_enc_ms" for c in comp_names] + \
             [f"{c}_dec_ms" for c in comp_names]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, entry in atlas.items():
            row = {
                "corpus": name,
                "domain": entry["domain"],
                "origin": entry["origin"],
                "size_bytes": entry["size_bytes"],
                "unigram_bpb": entry["unigram_bpb"],
            }
            for c in comp_names:
                r = entry["compressors"].get(c)
                row[f"{c}_bpb"]    = r["bpb"]    if r else ""
                row[f"{c}_enc_ms"] = r["enc_ms"] if r else ""
                row[f"{c}_dec_ms"] = r["dec_ms"] if r else ""
            w.writerow(row)
    print(f"\nCSV saved: {path}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 31: External Compressor Atlas")
    parser.add_argument("--skip-see", action="store_true", help="Skip SEE compressors")
    parser.add_argument("--corpus", nargs="+", metavar="NAME", help="Run only these corpus names")
    args = parser.parse_args()

    corpus_filter = set(args.corpus) if args.corpus else None

    print("Phase 31: External Compressor Atlas", flush=True)
    print(f"SEE: {SEE}", flush=True)
    print(f"WEIGHTS: {WEIGHTS}", flush=True)
    if not os.path.exists(SEE) and not args.skip_see:
        print("[WARN] see.exe not found — SEE compressors will fail", flush=True)
    print("", flush=True)

    atlas, comp_names = run_atlas(corpus_filter=corpus_filter, skip_see=args.skip_see)

    if not atlas:
        print("No corpora processed.", flush=True)
        return

    print_atlas(atlas, comp_names)
    answer_questions(atlas, comp_names)

    os.makedirs(RESULTS, exist_ok=True)

    json_path = os.path.join(RESULTS, "phase31_atlas.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"comp_names": comp_names, "corpora": atlas}, f, indent=2)
    print(f"\nJSON saved: {json_path}", flush=True)

    csv_path = os.path.join(RESULTS, "phase31_atlas.csv")
    save_csv(atlas, comp_names, csv_path)


if __name__ == "__main__":
    main()
