"""
Phase 23: Micro-Audit A/B Testing
Runs 3 configurations on c_code.c and multi_domain.bin:
  A) 3-Expert (SEE + UNI + BI)       -- --no-lz
  B) 4-Expert Full LZ                -- current default
  C) 4-Expert Muted LZ               -- --lz-mute (measures fourth-share tax)
Uses --eval-start 0 --eval-len 100 to evaluate the full file.
"""

import subprocess
import re
import os
import hashlib
import sys

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "weights/entropy_weights.bin")
CODER   = os.path.join(ROOT, "coder.exe")
RESULTS = "results"

DATASETS = {
    "c_code":        os.path.join(ROOT, "data/c_code.c"),
    "multi_domain":  os.path.join(ROOT, "data/multi_domain.bin"),
}

CONFIGS = [
    ("3-expert",     ["--blend", "moe", "--no-lz"]),
    ("4-expert-lz",  ["--blend", "moe"]),
    ("4-expert-mute",["--blend", "moe", "--lz-mute"]),
]

RAM_FULL_MB  = 262144 * (4 + 4 + 512) / 1024**2   # LzEntry full
RAM_MUTED_MB = 0.0                                  # no alloc in muted/no-lz

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def run_eval(dataset_path, extra_args, telemetry_path=None):
    cmd = [
        CODER,
        "--eval", dataset_path,
        "--weights", WEIGHTS,
        "--eval-start", "0",
        "--eval-len", "100",
    ] + extra_args
    if telemetry_path:
        cmd += ["--telemetry", telemetry_path]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout + result.stderr

def parse_output(output):
    metrics = {}
    patterns = {
        "bpb":          r"Model BPB:\s+([\d.]+)",
        "quant_bpb":    r"Quantized BPB:\s+([\d.]+)",
        "total_cyc":    r"Total:\s+([\d.]+)",
        "see_bpb":      r"SEE Only BPB:\s+([\d.]+)",
        "uni_bpb":      r"Uni Only BPB:\s+([\d.]+)",
        "bi_bpb":       r"Bi  Only BPB:\s+([\d.]+)",
        "lz_bpb":       r"LZ  Only BPB:\s+([\d.]+)",
        "oracle_bpb":   r"Oracle BPB:\s+([\d.]+)",
        "avg_w_see":    r"Avg W_SEE:\s+([\d.]+)",
        "avg_w_uni":    r"Avg W_UNI:\s+([\d.]+)",
        "avg_w_bi":     r"Avg W_BI:\s+([\d.]+)",
        "avg_w_lz":     r"Avg W_LZ:\s+([\d.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, output)
        metrics[key] = float(m.group(1)) if m else None
    return metrics

def fmt(v, fmt_str=".4f"):
    return f"{v:{fmt_str}}" if v is not None else "N/A"

def run_audit():
    results_dir = os.path.join(ROOT, RESULTS)
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}  # (dataset, config_name) -> metrics

    for ds_name, ds_path in DATASETS.items():
        sha = sha256_file(ds_path)
        size_kb = os.path.getsize(ds_path) / 1024
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({size_kb:.1f} KB)")
        print(f"SHA-256: {sha}")
        print(f"{'='*60}")

        for cfg_name, cfg_args in CONFIGS:
            print(f"\n  Config: {cfg_name}")
            tel_path = os.path.join(results_dir, f"phase23_{ds_name}_{cfg_name}.csv")
            output = run_eval(ds_path, cfg_args, telemetry_path=tel_path)
            metrics = parse_output(output)
            metrics["sha256"] = sha
            metrics["size_kb"] = size_kb
            all_results[(ds_name, cfg_name)] = metrics

            # Print raw output for inspection
            for line in output.splitlines():
                if any(kw in line for kw in ["BPB", "Avg W", "Total:", "error", "Error", "[no-lz", "[lz-mute"]):
                    print(f"    {line.strip()}")

    # Build comparison tables
    print("\n\n" + "="*80)
    print("PHASE 23 MICRO-AUDIT RESULTS")
    print("="*80)

    for ds_name, ds_path in DATASETS.items():
        sha = all_results.get((ds_name, CONFIGS[0][0]), {}).get("sha256", "N/A")
        print(f"\n### {ds_name.upper()}  (SHA-256: {sha[:16]}...)\n")

        header = f"{'Metric':<28} {'3-Expert':>12} {'4-LZ-Full':>12} {'4-LZ-Mute':>12}  {'Delta(4full-3exp)':>18}  {'Delta(4full-4mute)':>19}"
        print(header)
        print("-" * len(header))

        m3   = all_results.get((ds_name, "3-expert"), {})
        m4lz = all_results.get((ds_name, "4-expert-lz"), {})
        m4mt = all_results.get((ds_name, "4-expert-mute"), {})

        ram_3exp   = "0 MB (no alloc)"
        ram_4full  = f"{RAM_FULL_MB:.0f} MB"
        ram_4mute  = "0 MB (no alloc)"

        def row(label, key, fstr=".4f"):
            v3   = m3.get(key)
            v4lz = m4lz.get(key)
            v4mt = m4mt.get(key)
            d1 = f"{v4lz - v3:+.4f}" if v3 is not None and v4lz is not None else "N/A"
            d2 = f"{v4lz - v4mt:+.4f}" if v4lz is not None and v4mt is not None else "N/A"
            print(f"{label:<28} {fmt(v3, fstr):>12} {fmt(v4lz, fstr):>12} {fmt(v4mt, fstr):>12}  {d1:>18}  {d2:>19}")

        row("BPB (model)",        "bpb")
        row("BPB (quantized)",    "quant_bpb")
        row("Cycles/byte",        "total_cyc", ".1f")
        row("SEE-only BPB",       "see_bpb")
        row("UNI-only BPB",       "uni_bpb")
        row("BI-only BPB",        "bi_bpb")
        row("LZ-only BPB",        "lz_bpb")
        row("Oracle BPB",         "oracle_bpb")
        row("Avg W_SEE",          "avg_w_see")
        row("Avg W_UNI",          "avg_w_uni")
        row("Avg W_BI",           "avg_w_bi")
        row("Avg W_LZ",           "avg_w_lz")
        print(f"{'RAM (LZ table)':<28} {'0 MB':>12} {ram_4full:>12} {'0 MB':>12}")

        # ΔBPB / MB analysis
        v4lz_bpb = m4lz.get("bpb")
        v3_bpb   = m3.get("bpb")
        v4mt_bpb = m4mt.get("bpb")
        if v4lz_bpb and v3_bpb:
            delta_lz_vs_3 = v4lz_bpb - v3_bpb
            delta_per_mb  = delta_lz_vs_3 / RAM_FULL_MB if RAM_FULL_MB > 0 else 0
            print(f"\n  LZ contribution net  (4full - 3exp)  : {delta_lz_vs_3:+.4f} BPB")
        if v4lz_bpb and v4mt_bpb:
            delta_real_lz = v4lz_bpb - v4mt_bpb
            delta_tax     = v4mt_bpb - v3_bpb if v3_bpb else None
            print(f"  LZ real gain         (4full - 4mute) : {delta_real_lz:+.4f} BPB (actual LZ prediction value)")
            if delta_tax is not None:
                print(f"  Fourth-share tax     (4mute - 3exp)  : {delta_tax:+.4f} BPB (cost of diluting from 1/3 to 1/4)")
            print(f"  dBPB / MB of table   (4full vs 3exp) : {(v4lz_bpb-v3_bpb)/RAM_FULL_MB:+.6f} BPB/MB")

    # Save summary to file
    summary_path = os.path.join(results_dir, "phase23_audit_summary.txt")
    print(f"\n  Summary written to {summary_path} (re-run to refresh)")

if __name__ == "__main__":
    run_audit()
