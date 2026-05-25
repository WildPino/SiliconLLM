"""
Phase 28B: SPANPFX Expert Tribunal

Configurations:
  BASE       -- expert-profile general (LZ6+TOKPFX)
  SPAN_MUTE  -- +span-pfx-mute  (MoE tax of adding a muted arm)
  SPAN_PFX   -- +span-pfx       (gated, event-driven)

Promotion criteria:
  markdown  gain >= 0.025 BPB
  natural_text / c_code regression <= 0.005 BPB
  shuffled: muted (W_SPAN ~ floor)
"""

import os, re, subprocess, time

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEE     = os.path.join(ROOT, "see.exe")
WEIGHTS = os.path.join(ROOT, "weights", "entropy_weights_factors_r16.bin")
RESULTS = os.path.join(ROOT, "results")
DATA    = os.path.join(ROOT, "data")

DATASETS = [
    ("natural_text",  os.path.join(DATA, "natural_text.txt")),
    ("markdown_docs", os.path.join(DATA, "markdown_docs.md")),
    ("c_code",        os.path.join(DATA, "c_code.c")),
    ("shuffled",      os.path.join(DATA, "shuffled.bin")),
]

CONFIGS = [
    ("BASE",      ["--expert-profile", "general"]),
    ("SPAN_MUTE", ["--expert-profile", "general", "--span-pfx-mute"]),
    ("SPAN_PFX",  ["--expert-profile", "general", "--span-pfx"]),
]


def run_audit(path, extra):
    t0 = time.perf_counter()
    r = subprocess.run(
        [SEE, "audit", path, "--weights", WEIGHTS, "--blend", "moe"] + extra,
        capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    elapsed = (time.perf_counter() - t0) * 1000

    def g(pat, cast=float):
        m = re.search(pat, out)
        return cast(m.group(1)) if m else None

    m = {}
    m["quant_bpb"] = g(r"Quantized BPB:\s+([\d.]+)")
    m["span_bpb"]  = g(r"SPANPFX Only:\s+([\d.]+)")
    m["elapsed"]   = elapsed

    avg = re.search(r"Avg\s+\[(.*?)\]:\s+(.*)", out)
    if avg:
        labels = avg.group(1).split()
        vals   = list(map(float, avg.group(2).split()))
        m["w_span"] = vals[labels.index("SPANPFX")] if "SPANPFX" in labels else None
    else:
        m["w_span"] = None

    m["pct_elig"] = g(r"Eligible bytes:\s+\d+ / \d+ \(([\d.]+)%\)")
    m["w_when_elig"] = g(r"W_SPANPFX global:[\d.]+\s+when-eligible:\s+([\d.]+)")
    return m


def fmt(v): return f"{v:.4f}" if v is not None else "  N/A"
def d(a, b): return f"{a-b:+.4f}" if (a is not None and b is not None) else "   N/A"


def main():
    os.makedirs(RESULTS, exist_ok=True)
    lines = []
    def p(s=""): print(s); lines.append(s)

    p("=" * 80)
    p("PHASE 28B: SPANPFX Expert Tribunal")
    p("=" * 80)

    all_r = {}
    for ds, path in DATASETS:
        if not os.path.exists(path):
            p(f"[SKIP] {ds}"); continue
        p(); p(f"DATASET: {ds}  ({os.path.getsize(path)//1024} KB)")
        p("-" * 60)
        for cfg, args in CONFIGS:
            print(f"  {cfg}...", end="", flush=True)
            m = run_audit(path, args)
            all_r[(ds, cfg)] = m
            print(f" {fmt(m['quant_bpb'])} BPB  ({m['elapsed']:.0f} ms)")

        base = all_r.get((ds, "BASE"), {})
        mute = all_r.get((ds, "SPAN_MUTE"), {})
        span = all_r.get((ds, "SPAN_PFX"), {})
        p(f"  MoE tax (MUTE vs BASE):  {d(mute.get('quant_bpb'), base.get('quant_bpb'))}")
        p(f"  Net gain (SPAN vs BASE): {d(span.get('quant_bpb'), base.get('quant_bpb'))}")
        if span.get("pct_elig") is not None:
            p(f"  Eligible: {span['pct_elig']:.1f}%  "
              f"W_global={fmt(span.get('w_span'))}  W_when_elig={fmt(span.get('w_when_elig'))}")

    p(); p("=" * 80)
    p("TRIBUNAL VERDICT")
    p("=" * 80)
    p()
    header = f"{'Config':<12}" + "".join(f"  {n[:12]:>12}" for n, _ in DATASETS if os.path.exists(_))
    p(header); p("-" * len(header))
    valid = [(n, dp) for n, dp in DATASETS if os.path.exists(dp)]
    for cfg, _ in CONFIGS:
        row = f"{cfg:<12}" + "".join(f"  {fmt(all_r.get((n,cfg),{}).get('quant_bpb')):>12}" for n, _ in valid)
        p(row)
    p()
    p("Delta vs BASE:")
    for cfg, _ in CONFIGS:
        if cfg == "BASE": continue
        row = f"{cfg:<12}" + "".join(
            f"  {d(all_r.get((n,cfg),{}).get('quant_bpb'), all_r.get((n,'BASE'),{}).get('quant_bpb')):>12}"
            for n, _ in valid)
        p(row)

    p()
    md_gain   = -(all_r.get(("markdown_docs","SPAN_PFX"),{}).get("quant_bpb",0) or 0) + (all_r.get(("markdown_docs","BASE"),{}).get("quant_bpb",0) or 0)
    nt_regr   =  (all_r.get(("natural_text","SPAN_PFX"),{}).get("quant_bpb",0) or 0) - (all_r.get(("natural_text","BASE"),{}).get("quant_bpb",0) or 0)
    cc_regr   =  (all_r.get(("c_code","SPAN_PFX"),{}).get("quant_bpb",0) or 0) - (all_r.get(("c_code","BASE"),{}).get("quant_bpb",0) or 0)
    p(f"  markdown gain:     {md_gain:+.4f} BPB  (criterion >= 0.025)")
    p(f"  natural_text regr: {nt_regr:+.4f} BPB  (criterion <= 0.005)")
    p(f"  c_code regr:       {cc_regr:+.4f} BPB  (criterion <= 0.005)")
    p()
    if md_gain >= 0.025 and nt_regr <= 0.005 and cc_regr <= 0.005:
        p("  RESULT: SPANPFX PROMOTED.")
        p("  Candidate for --expert-profile markdown profile.")
    elif md_gain >= 0.010:
        p("  RESULT: SPANPFX PARTIAL — gain real but below criterion.")
        p("  Investigate: span volume too small, or LZ table too sparse?")
    else:
        p("  RESULT: SPANPFX REJECTED — insufficient markdown gain.")

    rp = os.path.join(RESULTS, "phase28b_span_tribunal.txt")
    with open(rp, "w", encoding="utf-8") as f: f.write("\n".join(lines))
    print(f"\nReport -> {rp}")

if __name__ == "__main__":
    main()
