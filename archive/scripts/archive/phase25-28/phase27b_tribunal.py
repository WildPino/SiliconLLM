"""
Phase 27B: Token-LZ Expert Tribunal
Tests TOK_PREFIX (inside-token prefix expert) vs LZ6 baseline across all domains.

Promotion criteria (pre-defined):
  - markdown_docs: must improve by > 0.05 BPB vs LZ6 baseline
  - natural_text:  must not worsen (delta < +0.01 BPB)
  - shuffled:      MoE must mute it (avg TOKPFX weight < 0.05)
  - c_code:        neutral or improved
  - TOKPFX-only BPB must be meaningfully lower than LZ-only (proves expert has signal)

Configurations tested:
  LZ6          -- current default (Phase 26 verdict)
  LZ6+TOKPFX  -- adds inside-token prefix expert as 5th MoE arm

Output: results/phase27b_tribunal_report.txt
"""

import os
import re
import subprocess
import time

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEE     = os.path.join(ROOT, "see.exe")
WEIGHTS = os.path.join(ROOT, "weights", "entropy_weights.bin")
RESULTS = os.path.join(ROOT, "results")
DATA    = os.path.join(ROOT, "data")

DATASETS = [
    ("c_code",        os.path.join(DATA, "c_code.c")),
    ("natural_text",  os.path.join(DATA, "natural_text.txt")),
    ("markdown_docs", os.path.join(DATA, "markdown_docs.md")),
    ("shuffled",      os.path.join(DATA, "shuffled.bin")),
    ("multi_domain",  os.path.join(DATA, "multi_domain.bin")),
]

# Promotion criteria for TOKPFX (pre-defined, not post-hoc)
CRITERIA = {
    "markdown_docs": ("improve",  0.05),   # must gain > 0.05 BPB
    "natural_text":  ("neutral",  0.01),   # must not worsen by > 0.01 BPB
    "c_code":        ("neutral",  0.02),   # neutral or improved
    "shuffled":      ("muted",    0.05),   # avg TOKPFX weight < 0.05
    "multi_domain":  ("neutral",  0.03),   # neutral or improved
}

# Phase 25 reference results for context
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
    m["see_only"]   = grab(r"SEE Only:\s+([\d.]+)")
    m["uni_only"]   = grab(r"UNI Only:\s+([\d.]+)")
    m["bi_only"]    = grab(r"BI\s+Only:\s+([\d.]+)")
    m["lz_only"]    = grab(r"LZ\s+Only:\s+([\d.]+)")
    m["tokpfx_only"] = grab(r"TOKPFX Only:\s+([\d.]+)")
    m["elapsed_ms"] = elapsed_ms

    # Avg weights line — handles both 4-expert and 5-expert format
    avg5 = re.search(
        r"Avg\s+\[SEE UNI BI\s+LZ\s+TOKPFX\]:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        out)
    avg4 = re.search(
        r"Avg\s+\[SEE UNI BI\s+LZ\s*\]:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        out)
    if avg5:
        m["w_see"]    = float(avg5.group(1))
        m["w_uni"]    = float(avg5.group(2))
        m["w_bi"]     = float(avg5.group(3))
        m["w_lz"]     = float(avg5.group(4))
        m["w_tokpfx"] = float(avg5.group(5))
    elif avg4:
        m["w_see"]    = float(avg4.group(1))
        m["w_uni"]    = float(avg4.group(2))
        m["w_bi"]     = float(avg4.group(3))
        m["w_lz"]     = float(avg4.group(4))
        m["w_tokpfx"] = 0.0
    else:
        m["w_see"] = m["w_uni"] = m["w_bi"] = m["w_lz"] = m["w_tokpfx"] = None

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
    p("PHASE 27B: Token-LZ Expert Tribunal")
    p("TOK_PREFIX — inside-token prefix expert (with MACRO support for LaTeX)")
    p("=" * 80)
    p()
    p("Promotion criteria (pre-defined):")
    p("  markdown_docs: gain > 0.05 BPB vs LZ6 baseline")
    p("  natural_text:  must not worsen by > 0.01 BPB")
    p("  c_code:        neutral or improved (delta < +0.02)")
    p("  shuffled:      avg TOKPFX weight < 0.05 (MoE must mute it)")
    p("  multi_domain:  neutral or improved (delta < +0.03)")
    p()

    all_results = {}  # (ds_name, config) -> metrics

    CONFIGS = [
        ("LZ6",        []),
        ("LZ6+TOKPFX", ["--tok-prefix"]),
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

        lz6  = all_results.get((ds_name, "LZ6"), {})
        tokp = all_results.get((ds_name, "LZ6+TOKPFX"), {})

        p()
        p(f"  {'Config':<14} {'BPB(model)':>10}  {'BPB(quant)':>10}  {'dBPB':>8}  "
          f"{'TOKPFX-only':>11}  {'W_TOKPFX':>9}  {'W_BI':>6}")
        p("  " + "-" * 74)

        for cfg_label, _ in CONFIGS:
            m  = all_results.get((ds_name, cfg_label), {})
            base_q = lz6.get("quant_bpb")
            d  = delta_str(m.get("quant_bpb"), base_q) if cfg_label != "LZ6" else "baseline"
            tok_only = fmt(m.get("tokpfx_only"), ".4f") if m.get("tokpfx_only") else "     ---"
            w_tok    = fmt(m.get("w_tokpfx"),    ".4f") if m.get("w_tokpfx") is not None else "   ---"
            w_bi     = fmt(m.get("w_bi"),         ".4f") if m.get("w_bi")     is not None else "   ---"
            p(f"  {cfg_label:<14} {fmt(m.get('bpb')):>10}  {fmt(m.get('quant_bpb')):>10}  "
              f"{d:>8}  {tok_only:>11}  {w_tok:>9}  {w_bi:>6}")

        # Expert weight evolution (final vs avg)
        if tokp.get("w_tokpfx") is not None:
            p()
            p(f"  MoE weight trajectory (LZ6+TOKPFX):")
            p(f"    Avg  TOKPFX: {fmt(tokp.get('w_tokpfx'), '.4f')}   "
              f"BI: {fmt(tokp.get('w_bi'), '.4f')}   "
              f"LZ: {fmt(tokp.get('w_lz'), '.4f')}")

        # TOKPFX vs LZ comparison
        tok_only_bpb = tokp.get("tokpfx_only")
        lz_only_bpb  = tokp.get("lz_only")
        if tok_only_bpb is not None and lz_only_bpb is not None:
            signal = lz_only_bpb - tok_only_bpb
            p(f"  TOKPFX-only vs LZ-only: {signal:+.4f} BPB "
              f"({'TOKPFX has signal' if signal > 0 else 'LZ is better'})")

        p()

    # ── Cross-dataset summary ──────────────────────────────────────────────────
    p()
    p("=" * 80)
    p("TRIBUNAL VERDICT: BPB (quantized) across all domains")
    p("=" * 80)

    valid_ds = [(n, dp) for n, dp in DATASETS if os.path.exists(dp)]

    hdr = f"{'Config':<14}" + "".join(f"  {n[:12]:>12}" for n, _ in valid_ds)
    p(hdr)
    p("-" * len(hdr))
    for cfg_label, _ in CONFIGS:
        row = f"{cfg_label:<14}"
        for ds_name, _ in valid_ds:
            m = all_results.get((ds_name, cfg_label), {})
            row += f"  {fmt(m.get('quant_bpb'), '.4f'):>12}"
        p(row)

    p()
    p("Delta LZ6+TOKPFX vs LZ6 (negative = improvement):")
    row = f"{'delta':14}"
    for ds_name, _ in valid_ds:
        base = all_results.get((ds_name, "LZ6"),        {}).get("quant_bpb")
        tok  = all_results.get((ds_name, "LZ6+TOKPFX"), {}).get("quant_bpb")
        row += f"  {delta_str(tok, base):>12}"
    p(row)

    # ── Promotion verdict ──────────────────────────────────────────────────────
    p()
    p("=" * 80)
    p("PROMOTION VERDICT")
    p("=" * 80)
    p()

    all_pass = True
    for ds_name, (criterion, threshold) in CRITERIA.items():
        lz6_m  = all_results.get((ds_name, "LZ6"),        {})
        tokp_m = all_results.get((ds_name, "LZ6+TOKPFX"), {})
        if not lz6_m or not tokp_m:
            p(f"  {ds_name:<16}: no data  [SKIP]")
            continue

        base_q = lz6_m.get("quant_bpb")
        tok_q  = tokp_m.get("quant_bpb")
        w_tok  = tokp_m.get("w_tokpfx", 0.0)

        if base_q is None or tok_q is None:
            p(f"  {ds_name:<16}: missing metrics  [SKIP]")
            continue

        delta = tok_q - base_q  # negative = improvement

        if criterion == "improve":
            ok = delta < -threshold
            verdict = "PASS" if ok else "FAIL"
            detail = f"delta={delta:+.4f} (need < -{threshold})"
        elif criterion == "neutral":
            ok = delta < threshold
            verdict = "PASS" if ok else "FAIL"
            detail = f"delta={delta:+.4f} (need < +{threshold})"
        elif criterion == "muted":
            ok = (w_tok is not None and w_tok < threshold)
            verdict = "PASS" if ok else "FAIL"
            detail = f"avg_w_tokpfx={w_tok:.4f} (need < {threshold})"
        else:
            ok = True
            verdict = "SKIP"
            detail = ""

        if not ok:
            all_pass = False
        p(f"  {ds_name:<16}: {verdict}  {detail}")

    p()
    if all_pass:
        p("  RESULT: TOKPFX PROMOTED")
        p("  All criteria met. Token-prefix expert earns real credit on structured text.")
        p("  Recommended: add --tok-prefix to standard audit profile.")
    else:
        p("  RESULT: TOKPFX NOT PROMOTED (some criteria failed)")
        p("  Review failures above before deciding next step.")

    p()
    p("  Next steps:")
    p("  - If promoted: make --tok-prefix default for audit, add to PLAN")
    p("  - Implement TOK_PREV (last-token -> first-byte-of-next) and re-run tribunal")
    p("  - Profile cycles/byte overhead vs gain")
    p()

    # Phase 25 reference for gap context
    p("=" * 80)
    p("GAP vs REFERENCE COMPRESSORS (best SEE config = LZ6+TOKPFX)")
    p("=" * 80)
    for ds_name, _ in valid_ds:
        best = all_results.get((ds_name, "LZ6+TOKPFX"), {}).get("quant_bpb")
        ref  = P25_REF.get(ds_name, {})
        if best is None or not ref:
            continue
        p(f"\n  {ds_name}:")
        for rname, rbpb in sorted(ref.items(), key=lambda x: x[1]):
            gap = best - rbpb
            p(f"    {rname:<10} {rbpb:.4f} BPB  gap={gap:+.4f}")

    report_path = os.path.join(RESULTS, "phase27b_tribunal_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport -> {report_path}")


if __name__ == "__main__":
    main()
