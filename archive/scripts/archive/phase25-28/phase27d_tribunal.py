"""
Phase 27D: TOK_PREV Tribunal
Tests TOK_PREV (inter-token start expert) vs LZ6+TOKPFX baseline.

Configurations tested:
  LZ6+TOKPFX             -- current default baseline
  LZ6+TOKPFX+TOKPREV     -- candidate
  LZ6+TOKPFX+TOKPREVMUTED -- empty arm to measure MoE tax
  LZ6+TOKPREV            -- optional, without TOKPFX
"""

import os
import re
import subprocess
import time

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEE     = os.path.join(ROOT, "see.exe")
WEIGHTS = os.path.join(ROOT, "weights", "entropy_weights_factors_r16.bin")
RESULTS = os.path.join(ROOT, "results")
DATA    = os.path.join(ROOT, "data")

DATASETS = [
    ("c_code",        os.path.join(DATA, "c_code.c")),
    ("natural_text",  os.path.join(DATA, "natural_text.txt")),
    ("markdown_docs", os.path.join(DATA, "markdown_docs.md")),
    ("shuffled",      os.path.join(DATA, "shuffled.bin")),
    ("multi_domain",  os.path.join(DATA, "multi_domain.bin")),
]

def run_audit(filepath, extra_args):
    t0 = time.perf_counter()
    result = subprocess.run(
        [SEE, "audit", filepath, "--weights", WEIGHTS, "--blend", "moe"] + extra_args,
        capture_output=True, text=True, timeout=600
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    out = result.stdout + result.stderr

    def grab(pattern):
        m = re.search(pattern, out)
        return float(m.group(1)) if m else None

    m = {}
    m["bpb"]        = grab(r"Model BPB:\s+([\d.]+)")
    m["quant_bpb"]  = grab(r"Quantized BPB:\s+([\d.]+)")
    m["lz_only"]    = grab(r"LZ\s+Only:\s+([\d.]+)")
    m["tokpfx_only"] = grab(r"TOKPFX Only:\s+([\d.]+)")
    m["tokprev_only"] = grab(r"TOKPREV Only:\s+([\d.]+)")
    m["elapsed_ms"] = elapsed_ms

    # Avg weights line
    avg_line = re.search(r"Avg\s+\[(.*?)\]:\s+(.*)", out)
    if avg_line:
        labels = avg_line.group(1).split()
        vals = list(map(float, avg_line.group(2).split()))
        m["w_tokpfx"] = vals[labels.index("TOKPFX")] if "TOKPFX" in labels else 0.0
        m["w_tokprev"] = vals[labels.index("TKPREV")] if "TKPREV" in labels else 0.0
    else:
        m["w_tokpfx"] = m["w_tokprev"] = None

    return m

def fmt(v, spec=".4f"):
    return f"{v:{spec}}" if v is not None else "  N/A "

def delta_str(a, b):
    if a is None or b is None: return "   N/A"
    d = a - b
    return f"{d:+.4f}"

def main():
    os.makedirs(RESULTS, exist_ok=True)
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    p("=" * 80)
    p("PHASE 27D: TOK_PREV Tribunal")
    p("=" * 80)

    all_results = {}

    CONFIGS = [
        ("LZ6+TOKPFX",          ["--tok-prefix"]),
        ("LZ6+TOKPFX+TOKPREV",  ["--tok-prefix", "--tok-prev"]),
        ("LZ6+TOKPFX+MUTED",    ["--tok-prefix", "--tok-prev-mute"]),
        ("LZ6+TOKPREV",         ["--tok-prev"]),
    ]

    for ds_name, ds_path in DATASETS:
        if not os.path.exists(ds_path):
            p(f"[SKIP] {ds_name}: not found")
            continue

        raw_bytes = os.path.getsize(ds_path)
        p("=" * 80)
        p(f"DATASET: {ds_name}  ({raw_bytes / 1024:.1f} KB)")
        p("=" * 80)

        for cfg_label, cfg_args in CONFIGS:
            print(f"  Running {cfg_label}...", end="", flush=True)
            m = run_audit(ds_path, cfg_args)
            all_results[(ds_name, cfg_label)] = m
            print(f" {fmt(m.get('quant_bpb'))} BPB  ({m['elapsed_ms']:.0f} ms)")

        base = all_results.get((ds_name, "LZ6+TOKPFX"), {})

        p()
        p(f"  {'Config':<20} {'BPB(quant)':>10}  {'dBPB':>8}  {'W_TOKPFX':>9}  {'W_TOKPREV':>9}")
        p("  " + "-" * 62)

        for cfg_label, _ in CONFIGS:
            m  = all_results.get((ds_name, cfg_label), {})
            base_q = base.get("quant_bpb")
            d  = delta_str(m.get("quant_bpb"), base_q) if cfg_label != "LZ6+TOKPFX" else "baseline"
            w_tok    = fmt(m.get("w_tokpfx"),    ".4f")
            w_prev   = fmt(m.get("w_tokprev"),   ".4f")
            p(f"  {cfg_label:<20} {fmt(m.get('quant_bpb')):>10}  {d:>8}  {w_tok:>9}  {w_prev:>9}")

        p()

    p()
    p("=" * 80)
    p("TRIBUNAL VERDICT: BPB (quantized) across all domains")
    p("=" * 80)

    valid_ds = [(n, dp) for n, dp in DATASETS if os.path.exists(dp)]

    hdr = f"{'Config':<20}" + "".join(f"  {n[:12]:>12}" for n, _ in valid_ds)
    p(hdr)
    p("-" * len(hdr))
    for cfg_label, _ in CONFIGS:
        row = f"{cfg_label:<20}"
        for ds_name, _ in valid_ds:
            m = all_results.get((ds_name, cfg_label), {})
            row += f"  {fmt(m.get('quant_bpb'), '.4f'):>12}"
        p(row)

    p()
    p("Delta vs LZ6+TOKPFX (negative = improvement):")
    for cfg_label, _ in CONFIGS:
        if cfg_label == "LZ6+TOKPFX": continue
        row = f"{cfg_label[:20]:20}"
        for ds_name, _ in valid_ds:
            base = all_results.get((ds_name, "LZ6+TOKPFX"), {}).get("quant_bpb")
            tok  = all_results.get((ds_name, cfg_label), {}).get("quant_bpb")
            row += f"  {delta_str(tok, base):>12}"
        p(row)

    report_path = os.path.join(RESULTS, "phase27d_tribunal_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport -> {report_path}")

if __name__ == "__main__":
    main()
