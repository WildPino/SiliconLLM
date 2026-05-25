"""
Phase 25: External Compression Positioning
Compares SEE against standard compressors on all datasets.
Goal: map where SEE loses and why — not to win, but to identify the next gap.

Output: results/phase25_report.txt + results/phase25_gap_analysis.txt
"""

import os
import sys
import re
import time
import zlib
import lzma
import bz2
import hashlib
import subprocess

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEE     = os.path.join(ROOT, "see.exe")
WEIGHTS = os.path.join(ROOT, "weights", "entropy_weights.bin")
RESULTS = os.path.join(ROOT, "results")

DATASETS = [
    ("c_code",        os.path.join(ROOT, "data", "c_code.c")),
    ("natural_text",  os.path.join(ROOT, "data", "natural_text.txt")),
    ("markdown_docs", os.path.join(ROOT, "data", "markdown_docs.md")),
    ("shuffled",      os.path.join(ROOT, "data", "shuffled.bin")),
    ("multi_domain",  os.path.join(ROOT, "data", "multi_domain.bin")),
]

# ── Optional compressors ──────────────────────────────────────────────────────

def try_import_zstd():
    try:
        import zstandard as zstd
        return zstd
    except ImportError:
        return None

def try_import_brotli():
    try:
        import brotli
        return brotli
    except ImportError:
        return None

# ── Utilities ─────────────────────────────────────────────────────────────────

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def bpb_from_bytes(compressed: bytes, original_len: int) -> float:
    return len(compressed) * 8.0 / original_len

def compress_timed(fn, data: bytes):
    t0 = time.perf_counter()
    compressed = fn(data)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return compressed, elapsed_ms

# ── Compressor runners ─────────────────────────────────────────────────────────

def run_zlib(data: bytes, level: int):
    compressed, ms = compress_timed(lambda d: zlib.compress(d, level), data)
    return len(compressed), bpb_from_bytes(compressed, len(data)), ms

def run_lzma(data: bytes):
    compressed, ms = compress_timed(lambda d: lzma.compress(d, preset=6), data)
    return len(compressed), bpb_from_bytes(compressed, len(data)), ms

def run_bz2(data: bytes):
    compressed, ms = compress_timed(lambda d: bz2.compress(d, compresslevel=9), data)
    return len(compressed), bpb_from_bytes(compressed, len(data)), ms

def run_zstd(data: bytes, zstd_mod, level: int = 3):
    cctx = zstd_mod.ZstdCompressor(level=level)
    compressed, ms = compress_timed(lambda d: cctx.compress(d), data)
    return len(compressed), bpb_from_bytes(compressed, len(data)), ms

def run_brotli(data: bytes, brotli_mod, quality: int = 11):
    compressed, ms = compress_timed(lambda d: brotli_mod.compress(d, quality=quality), data)
    return len(compressed), bpb_from_bytes(compressed, len(data)), ms

# ── SEE audit runner ───────────────────────────────────────────────────────────

def run_see_audit(filepath: str):
    """Run see.exe audit and parse output. Returns dict of metrics."""
    if not os.path.exists(SEE):
        return None
    if not os.path.exists(WEIGHTS):
        return None

    t0 = time.perf_counter()
    result = subprocess.run(
        [SEE, "audit", filepath, "--weights", WEIGHTS, "--blend", "moe"],
        capture_output=True, text=True, timeout=300
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    out = result.stdout + result.stderr

    m = {}
    def grab(pattern):
        found = re.search(pattern, out)
        return float(found.group(1)) if found else None

    m["bpb"]        = grab(r"Model BPB:\s+([\d.]+)")
    m["quant_bpb"]  = grab(r"Quantized BPB:\s+([\d.]+)")
    m["unigram_bpb"]= grab(r"Unigram BPB:\s+([\d.]+)")
    m["see_only"]   = grab(r"SEE Only:\s+([\d.]+)")
    m["uni_only"]   = grab(r"UNI Only:\s+([\d.]+)")
    m["bi_only"]    = grab(r"BI\s+Only:\s+([\d.]+)")
    m["lz_only"]    = grab(r"LZ\s+Only:\s+([\d.]+)")

    avg_weights = re.search(r"Avg\s+\[SEE UNI BI\s+LZ\s*\]:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", out)
    if avg_weights:
        m["w_see"] = float(avg_weights.group(1))
        m["w_uni"] = float(avg_weights.group(2))
        m["w_bi"]  = float(avg_weights.group(3))
        m["w_lz"]  = float(avg_weights.group(4))
    else:
        m["w_see"] = m["w_uni"] = m["w_bi"] = m["w_lz"] = None

    m["elapsed_ms"] = elapsed_ms
    m["raw_output"] = out
    return m

# ── Gap analysis ───────────────────────────────────────────────────────────────

def analyze_gap(ds_name: str, see_bpb: float, competitors: dict) -> list[str]:
    """
    Apply the Phase 25 reading rules:
      - loses to zstd on repetitions → LZ window too small
      - loses on natural text        → word/subword model or higher-order PPM
      - loses on code                → syntactic context or longer history
      - loses on binary              → domain may not be target for V0
    """
    lines = []
    worst_gap = None
    worst_name = None

    for name, (_, cmp_bpb, _) in competitors.items():
        if cmp_bpb is None:
            continue
        gap = see_bpb - cmp_bpb  # positive = SEE loses
        if worst_gap is None or gap > worst_gap:
            worst_gap = gap
            worst_name = name

    if worst_gap is None or worst_gap <= 0:
        lines.append("  SEE competitive on all measured compressors.")
        return lines

    lines.append(f"  Worst gap: +{worst_gap:.4f} BPB vs {worst_name}")

    is_text    = "natural" in ds_name or "markdown" in ds_name
    is_code    = "c_code"  in ds_name
    is_binary  = "shuffled" in ds_name or "multi_domain" in ds_name
    is_multi   = "multi_domain" in ds_name

    # Check gap vs zstd specifically (repetition indicator)
    zstd_gap = None
    for name, (_, cmp_bpb, _) in competitors.items():
        if "zstd" in name and cmp_bpb is not None:
            zstd_gap = see_bpb - cmp_bpb

    if zstd_gap is not None and zstd_gap > 0.3:
        lines.append(f"  [GAP vs zstd ({zstd_gap:+.3f})] SEE LZ window likely too shallow for long-range repetitions.")
        lines.append(    "    Hypothesis: increase LZ hash coverage or add a second-order match tier.")

    if is_text and worst_gap > 0.2:
        lines.append(f"  [DOMAIN=text] gap suggests missing word/subword statistics.")
        lines.append(    "    Hypothesis: a token-boundary BI expert or higher-order (4-gram) SEE layer.")

    if is_code and worst_gap > 0.15:
        lines.append(f"  [DOMAIN=code] gap suggests syntactic structure not captured.")
        lines.append(    "    Hypothesis: longer SEE context (identifier-aware) or indentation-aware BI.")

    if "shuffled" in ds_name and worst_gap < 0.05:
        lines.append(    "  [DOMAIN=shuffled] gap small as expected -- purely entropic data.")

    if is_multi and worst_gap > 0.2:
        lines.append(    "  [DOMAIN=multi] compound gap -- run per-segment breakdown to isolate.")

    # Check brotli specifically for text
    brotli_gap = None
    for name, (_, cmp_bpb, _) in competitors.items():
        if "brotli" in name and cmp_bpb is not None:
            brotli_gap = see_bpb - cmp_bpb
    if brotli_gap is not None and brotli_gap > 0.3 and is_text:
        lines.append(f"  [GAP vs brotli ({brotli_gap:+.3f})] brotli uses static text dictionary.")
        lines.append(    "    Hypothesis: SEE needs a static prior or bigram frequency table for common tokens.")

    return lines

# ── Formatting helpers ─────────────────────────────────────────────────────────

def fmt_bpb(v):   return f"{v:.4f}" if v is not None else "  N/A "
def fmt_kb(n):    return f"{n/1024:.1f} KB" if n else "N/A"
def fmt_ms(v):    return f"{v:.0f} ms" if v is not None else "N/A"

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS, exist_ok=True)

    zstd_mod   = try_import_zstd()
    brotli_mod = try_import_brotli()

    lines = []  # report lines
    gap_lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    p("=" * 80)
    p("PHASE 25: External Compression Positioning")
    p("=" * 80)
    p(f"SEE binary:  {SEE}  {'[FOUND]' if os.path.exists(SEE) else '[MISSING]'}")
    p(f"Weights:     {WEIGHTS}  {'[FOUND]' if os.path.exists(WEIGHTS) else '[MISSING]'}")
    p(f"zstd:        {'available (pip install zstandard)' if zstd_mod   else 'NOT INSTALLED'}")
    p(f"brotli:      {'available (pip install brotli)'    if brotli_mod else 'NOT INSTALLED'}")
    p()

    all_results = {}

    for ds_name, ds_path in DATASETS:
        if not os.path.exists(ds_path):
            p(f"[SKIP] {ds_name}: file not found at {ds_path}")
            continue

        with open(ds_path, "rb") as f:
            data = f.read()

        raw_bytes = len(data)
        sha = sha256_bytes(data)

        p("=" * 80)
        p(f"DATASET: {ds_name}  ({raw_bytes/1024:.1f} KB)")
        p(f"SHA-256: {sha}")
        p("=" * 80)

        competitors = {}

        # ── stdlib compressors ─────────────────────────────────────────────
        sz, bpb_val, ms = run_zlib(data, 1)
        competitors["zlib-1"]  = (sz, bpb_val, ms)

        sz, bpb_val, ms = run_zlib(data, 9)
        competitors["zlib-9"]  = (sz, bpb_val, ms)

        sz, bpb_val, ms = run_lzma(data)
        competitors["lzma-6"]  = (sz, bpb_val, ms)

        sz, bpb_val, ms = run_bz2(data)
        competitors["bz2-9"]   = (sz, bpb_val, ms)

        # ── optional compressors ───────────────────────────────────────────
        if zstd_mod:
            sz, bpb_val, ms = run_zstd(data, zstd_mod, level=3)
            competitors["zstd-3"]  = (sz, bpb_val, ms)
            sz, bpb_val, ms = run_zstd(data, zstd_mod, level=19)
            competitors["zstd-19"] = (sz, bpb_val, ms)

        if brotli_mod:
            sz, bpb_val, ms = run_brotli(data, brotli_mod, quality=11)
            competitors["brotli-11"] = (sz, bpb_val, ms)

        # ── SEE audit ─────────────────────────────────────────────────────
        see_metrics = run_see_audit(ds_path)

        # ── Print table ───────────────────────────────────────────────────
        p(f"\n{'Compressor':<16} {'BPB':>8}  {'Comp.Size':>12}  {'Time':>8}")
        p("-" * 52)

        for name, (sz, bpb_val, ms) in sorted(competitors.items()):
            p(f"{name:<16} {fmt_bpb(bpb_val):>8}  {fmt_kb(sz):>12}  {fmt_ms(ms):>8}")

        if see_metrics:
            see_bpb_model = see_metrics.get("bpb")
            see_bpb_quant = see_metrics.get("quant_bpb")
            see_ms        = see_metrics.get("elapsed_ms")
            p(f"{'SEE (model)':<16} {fmt_bpb(see_bpb_model):>8}  {'(no encode)':>12}  {fmt_ms(see_ms):>8}")
            p(f"{'SEE (quant)':<16} {fmt_bpb(see_bpb_quant):>8}  {'(no encode)':>12}  {'':>8}")
            p(f"{'unigram':<16} {fmt_bpb(see_metrics.get('unigram_bpb')):>8}")
        else:
            p(f"{'SEE':<16} {'N/A':>8}  {'see.exe missing':>12}")

        # ── SEE internal breakdown ─────────────────────────────────────────
        if see_metrics and see_metrics.get("w_see") is not None:
            p()
            p("  SEE internal breakdown:")
            p(f"    Expert BPB:  SEE={fmt_bpb(see_metrics['see_only'])}  UNI={fmt_bpb(see_metrics['uni_only'])}  BI={fmt_bpb(see_metrics['bi_only'])}  LZ={fmt_bpb(see_metrics['lz_only'])}")
            p(f"    Avg weights: SEE={see_metrics['w_see']:.4f}  UNI={see_metrics['w_uni']:.4f}  BI={see_metrics['w_bi']:.4f}  LZ={see_metrics['w_lz']:.4f}")

            # LZ contribution
            if see_metrics.get("lz_only") and see_metrics["lz_only"] < 7.5:
                p(f"    LZ is ACTIVE (only-BPB={see_metrics['lz_only']:.4f}, avg_w={see_metrics['w_lz']:.4f})")
            else:
                p(f"    LZ is MARGINAL (lz_only={fmt_bpb(see_metrics.get('lz_only'))} — high = sparse matches)")

        # ── Gap analysis ───────────────────────────────────────────────────
        if see_metrics and see_metrics.get("quant_bpb") is not None:
            see_bpb_for_gap = see_metrics["quant_bpb"]
            gaps = analyze_gap(ds_name, see_bpb_for_gap, competitors)
            p()
            p("  Gap analysis (SEE quant_bpb vs competitors):")
            for gl in gaps:
                p(gl)

            gap_lines.append(f"\n### {ds_name} ({raw_bytes/1024:.1f} KB)")
            gap_lines.extend(gaps)

        all_results[ds_name] = {
            "raw_bytes":    raw_bytes,
            "competitors":  competitors,
            "see":          see_metrics,
        }
        p()

    # ── Summary table across all datasets ─────────────────────────────────
    p()
    p("=" * 80)
    p("SUMMARY: BPB across datasets")
    p("=" * 80)

    ds_names = [n for n, _ in DATASETS if n in all_results]
    compressor_names = []
    for ds_name in ds_names:
        for cn in all_results[ds_name]["competitors"]:
            if cn not in compressor_names:
                compressor_names.append(cn)

    header = f"{'Compressor':<16}" + "".join(f"  {n[:12]:>12}" for n in ds_names)
    p(header)
    p("-" * len(header))

    for cn in compressor_names:
        row = f"{cn:<16}"
        for ds_name in ds_names:
            v = all_results[ds_name]["competitors"].get(cn)
            row += f"  {fmt_bpb(v[1] if v else None):>12}"
        p(row)

    # SEE rows
    for label, key in [("SEE (model)", "bpb"), ("SEE (quant)", "quant_bpb")]:
        row = f"{label:<16}"
        for ds_name in ds_names:
            see = all_results[ds_name].get("see")
            v = see.get(key) if see else None
            row += f"  {fmt_bpb(v):>12}"
        p(row)

    # ── Gap summary ────────────────────────────────────────────────────────
    p()
    p("=" * 80)
    p("GAP ANALYSIS SUMMARY")
    p("=" * 80)
    for gl in gap_lines:
        p(gl)

    p()
    p("=" * 80)
    p("NEXT STEPS (from gap map)")
    p("=" * 80)
    p("  Read results above to identify the largest systematic gap.")
    p("  Possible directions (ordered by evidence, not assumption):")
    p("  1. If SEE loses on repetitions vs zstd: LZ window / hash depth")
    p("  2. If SEE loses on text vs brotli/zstd: word-boundary BI or PPM order +1")
    p("  3. If SEE loses on code vs zstd: syntactic context or longer history")
    p("  4. If gap is uniform across domains: MoE mixing is the bottleneck")
    p("  5. If gap only on binary: binary not a V0 target — accept and document")
    p()

    # ── Write reports ──────────────────────────────────────────────────────
    report_path = os.path.join(RESULTS, "phase25_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport written: {report_path}")

    # Save raw SEE outputs for inspection
    for ds_name in ds_names:
        see = all_results[ds_name].get("see")
        if see and see.get("raw_output"):
            raw_path = os.path.join(RESULTS, f"phase25_see_raw_{ds_name}.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(see["raw_output"])

if __name__ == "__main__":
    main()
