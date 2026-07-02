"""
Phase 26: Multi-Scale LZ Context-Length Tribunal
Tests LZ4, LZ6, LZ8 (standalone) and LZ4+LZ8 (dual) across all datasets.
Goal: measure whether wider context helps, domain-by-domain, and what the MoE decides.

Output: results/phase26_tribunal_report.txt
"""

import os
import re
import subprocess
import time

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

# Tribunal configurations: (label, extra_flags)
CONFIGS = [
    ("LZ4",      ["--lz-key", "4"]),
    ("LZ6",      ["--lz-key", "6"]),
    ("LZ8",      ["--lz-key", "8"]),
    ("LZ4+LZ8",  ["--lz-dual"]),
]

# Phase 25 reference results (competitors) for context in report
P25_REF = {
    "c_code":        {"zlib-9": 1.2450, "lzma-6": 1.0441, "bz2-9": 1.1016},
    "natural_text":  {"zlib-9": 2.4430, "lzma-6": 2.1531, "bz2-9": 1.7444},
    "markdown_docs": {"zlib-9": 2.6295, "lzma-6": 2.4104, "bz2-9": 2.3997},
    "shuffled":      {"zlib-9": 5.5349, "lzma-6": 5.1283, "bz2-9": 5.7320},
    "multi_domain":  {"zlib-9": 3.0238, "lzma-6": 2.6964, "bz2-9": 3.0050},
}

def run_audit(filepath, extra_args):
    t0 = time.perf_counter()
    result = subprocess.run(
        [SEE, "audit", filepath, "--weights", WEIGHTS, "--blend", "moe"] + extra_args,
        capture_output=True, text=True, timeout=300
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    out = result.stdout + result.stderr

    def grab(pattern):
        m = re.search(pattern, out)
        return float(m.group(1)) if m else None

    def grab_all(pattern):
        m = re.search(pattern, out)
        if not m: return None
        return [float(m.group(i+1)) for i in range(len(m.groups()))]

    m = {}
    m["bpb"]       = grab(r"Model BPB:\s+([\d.]+)")
    m["quant_bpb"] = grab(r"Quantized BPB:\s+([\d.]+)")
    m["see_only"]  = grab(r"SEE Only:\s+([\d.]+)")
    m["uni_only"]  = grab(r"UNI Only:\s+([\d.]+)")
    m["bi_only"]   = grab(r"BI\s+Only:\s+([\d.]+)")
    m["lz_only"]   = grab(r"LZ\s+Only:\s+([\d.]+)")
    m["lz8_only"]  = grab(r"LZ8 Only:\s+([\d.]+)")
    m["elapsed_ms"] = elapsed_ms

    avg = re.search(r"Avg\s+\[SEE UNI BI\s+LZ\s*(?:LZ8)?\]:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s+([\d.]+))?", out)
    if avg:
        m["w_see"] = float(avg.group(1))
        m["w_uni"] = float(avg.group(2))
        m["w_bi"]  = float(avg.group(3))
        m["w_lz"]  = float(avg.group(4))
        m["w_lz8"] = float(avg.group(5)) if avg.group(5) else 0.0
    else:
        m["w_see"] = m["w_uni"] = m["w_bi"] = m["w_lz"] = m["w_lz8"] = None

    return m

def fmt(v, f=".4f"):
    return f"{v:{f}}" if v is not None else "  N/A "

def delta(a, b):
    if a is None or b is None: return " N/A"
    return f"{a-b:+.4f}"

def main():
    os.makedirs(RESULTS, exist_ok=True)
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    p("=" * 80)
    p("PHASE 26: Multi-Scale LZ Context-Length Tribunal")
    p("=" * 80)
    p()

    all_results = {}  # (ds_name, cfg_label) -> metrics

    for ds_name, ds_path in DATASETS:
        if not os.path.exists(ds_path):
            p(f"[SKIP] {ds_name}: not found")
            continue

        raw_bytes = os.path.getsize(ds_path)
        p("=" * 80)
        p(f"DATASET: {ds_name}  ({raw_bytes/1024:.1f} KB)")
        p("=" * 80)

        for cfg_label, cfg_args in CONFIGS:
            print(f"  Running {cfg_label}...", end="", flush=True)
            m = run_audit(ds_path, cfg_args)
            all_results[(ds_name, cfg_label)] = m
            print(f" {fmt(m.get('quant_bpb'))} BPB  ({m['elapsed_ms']:.0f} ms)")

        # Print comparison table
        p()
        p(f"  {'Config':<12} {'BPB(model)':>10}  {'BPB(quant)':>10}  {'LZ-only':>8}  {'LZ8-only':>8}  {'W_LZ':>6}  {'W_LZ8':>6}  {'dBPB vs LZ4':>12}")
        p("  " + "-" * 82)

        baseline_bpb = all_results.get((ds_name, "LZ4"), {}).get("quant_bpb")
        for cfg_label, _ in CONFIGS:
            m = all_results.get((ds_name, cfg_label), {})
            d = delta(m.get("quant_bpb"), baseline_bpb) if cfg_label != "LZ4" else "  baseline"
            w_lz8_str = fmt(m.get("w_lz8"), ".4f") if m.get("w_lz8", 0) > 0 else "   ---"
            lz8_only_str = fmt(m.get("lz8_only"), ".4f") if m.get("lz8_only") is not None else "    ---"
            p(f"  {cfg_label:<12} {fmt(m.get('bpb')):>10}  {fmt(m.get('quant_bpb')):>10}  {fmt(m.get('lz_only'), '.4f'):>8}  {lz8_only_str:>8}  {fmt(m.get('w_lz'), '.4f'):>6}  {w_lz8_str:>6}  {d:>12}")

        # Phase 25 reference
        p()
        ref = P25_REF.get(ds_name, {})
        if ref:
            p(f"  Phase 25 reference:")
            for name, bpb in sorted(ref.items(), key=lambda x: x[1]):
                best_see = all_results.get((ds_name, "LZ4+LZ8"), {}).get("quant_bpb")
                gap = f"{(best_see - bpb):+.4f}" if best_see is not None else "N/A"
                p(f"    {name:<10} {bpb:.4f} BPB  (gap from best SEE: {gap})")
        p()

    # ── Summary table: all datasets × all configs ──────────────────────────────
    p()
    p("=" * 80)
    p("TRIBUNAL VERDICT: BPB (quantized) across all domains")
    p("=" * 80)

    valid_ds = [(n, p_) for n, p_ in DATASETS if os.path.exists(p_)]

    header = f"{'Config':<12}" + "".join(f"  {n[:12]:>12}" for n, _ in valid_ds)
    p(header)
    p("-" * len(header))

    for cfg_label, _ in CONFIGS:
        row = f"{cfg_label:<12}"
        for ds_name, _ in valid_ds:
            m = all_results.get((ds_name, cfg_label), {})
            v = m.get("quant_bpb")
            row += f"  {fmt(v, '.4f'):>12}"
        p(row)

    # Delta vs LZ4 baseline row
    p()
    p("Delta vs LZ4 (quant BPB, negative = improvement):")
    for cfg_label, _ in CONFIGS:
        if cfg_label == "LZ4": continue
        row = f"{cfg_label:<12}"
        for ds_name, _ in valid_ds:
            base = all_results.get((ds_name, "LZ4"), {}).get("quant_bpb")
            m    = all_results.get((ds_name, cfg_label), {}).get("quant_bpb")
            row += f"  {delta(m, base):>12}"
        p(row)

    # ── Domain-by-domain tribunal reading ─────────────────────────────────────
    p()
    p("=" * 80)
    p("TRIBUNAL READING: what the MoE decided")
    p("=" * 80)

    for ds_name, _ in valid_ds:
        lz4  = all_results.get((ds_name, "LZ4"), {})
        lz6  = all_results.get((ds_name, "LZ6"), {})
        lz8  = all_results.get((ds_name, "LZ8"), {})
        dual = all_results.get((ds_name, "LZ4+LZ8"), {})

        p(f"\n  {ds_name}:")

        lz4_bpb  = lz4.get("quant_bpb")
        lz6_bpb  = lz6.get("quant_bpb")
        lz8_bpb  = lz8.get("quant_bpb")
        dual_bpb = dual.get("quant_bpb")

        if all(v is not None for v in [lz4_bpb, lz6_bpb, lz8_bpb, dual_bpb]):
            best_solo = min(lz4_bpb, lz6_bpb, lz8_bpb)
            best_name = ["LZ4", "LZ6", "LZ8"][[lz4_bpb, lz6_bpb, lz8_bpb].index(best_solo)]

            if lz8_bpb < lz4_bpb:
                gain = lz4_bpb - lz8_bpb
                p(f"    LZ8 beats LZ4 by {gain:.4f} BPB -> long context helps on this domain")
            elif lz4_bpb < lz8_bpb:
                loss = lz8_bpb - lz4_bpb
                p(f"    LZ4 beats LZ8 by {loss:.4f} BPB -> short context is more reliable here")
            else:
                p(f"    LZ4 and LZ8 roughly equivalent")

            if lz6_bpb < min(lz4_bpb, lz8_bpb):
                p(f"    LZ6 is the sweet spot ({lz6_bpb:.4f} BPB)")

            dual_gain = lz4_bpb - dual_bpb
            if dual_gain > 0.001:
                w_lz8 = dual.get("w_lz8", 0)
                p(f"    Dual (LZ4+LZ8) gains {dual_gain:.4f} BPB vs LZ4 alone (LZ8 avg weight: {w_lz8:.4f})")
            else:
                p(f"    Dual mode gives no meaningful gain ({dual_gain:+.4f} BPB)")

    # ── Final verdict ──────────────────────────────────────────────────────────
    p()
    p("=" * 80)
    p("VERDICT")
    p("=" * 80)
    p()
    p("  Promotion criteria (defined before tribunal):")
    p("  - markdown must improve")
    p("  - c_code must improve")
    p("  - shuffled must not regress significantly (MoE should mute wide LZ)")
    p("  - natural_text tells us if word-patterns need more context")
    p()

    for crit_ds in ["markdown_docs", "c_code", "shuffled", "natural_text"]:
        lz4_bpb  = all_results.get((crit_ds, "LZ4"), {}).get("quant_bpb")
        dual_bpb = all_results.get((crit_ds, "LZ4+LZ8"), {}).get("quant_bpb")
        if lz4_bpb is None or dual_bpb is None:
            p(f"  {crit_ds:<16}: no data")
            continue
        delta_v = dual_bpb - lz4_bpb
        if crit_ds == "shuffled":
            verdict = "OK" if delta_v < 0.02 else "REGRESSED"
        else:
            verdict = "IMPROVED" if delta_v < -0.001 else "NO GAIN"
        p(f"  {crit_ds:<16}: dual={dual_bpb:.4f}  lz4={lz4_bpb:.4f}  delta={delta_v:+.4f}  [{verdict}]")

    p()
    p("  Next: based on above, decide whether to promote dual LZ as default,")
    p("  promote LZ6/LZ8 as replacement for LZ4, or archive tribunal without change.")
    p()

    report_path = os.path.join(RESULTS, "phase26_tribunal_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport: {report_path}")

if __name__ == "__main__":
    main()
