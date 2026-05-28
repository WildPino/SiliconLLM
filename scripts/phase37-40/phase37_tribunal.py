"""
Phase 37: Multi-Domain Static Prior Tribunal.

Steps:
  1. Build training corpora (calls phase37_corpus.py)
  2. Compile train_entropy_readout.exe (if needed)
  3. Train 4 weight sets from the Phase 37 corpora
  4. Audit all Phase29A corpora × all weight sets with see.exe --expert-profile general
  5. Print delta-BPB matrix vs V1.0 baseline (entropy_weights_factors_r16.bin)
  6. Save results/phase37_tribunal.json
  7. Print verdict

Usage:
    python scripts/phase37_tribunal.py [--skip-corpus] [--skip-train] [--skip-audit]
"""

import os
import sys
import json
import subprocess
import argparse
import re

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "data")
WEIGHTS_DIR = os.path.join(ROOT, "weights")
RESULTS_DIR = os.path.join(ROOT, "results")
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
BENCHES_DIR = os.path.join(ROOT, "benchmarks")

SEE_EXE     = os.path.join(ROOT, "see.exe")
TRAINER_SRC = os.path.join(BENCHES_DIR, "train_entropy_readout.c")
TRAINER_EXE = os.path.join(ROOT, "train_entropy_readout.exe")

BASELINE_WEIGHTS = os.path.join(WEIGHTS_DIR, "entropy_weights_factors_r16.bin")

# Phase 37 weight files to train
WEIGHT_SETS = {
    "multidomain":    os.path.join(WEIGHTS_DIR, "phase37_multidomain.bin"),
    "loo_no_md":      os.path.join(WEIGHTS_DIR, "phase37_loo_no_md.bin"),
    "loo_no_code":    os.path.join(WEIGHTS_DIR, "phase37_loo_no_code.bin"),
    "loo_no_prose":   os.path.join(WEIGHTS_DIR, "phase37_loo_no_prose.bin"),
}

TRAINING_CORPORA = {
    "multidomain":  os.path.join(DATA_DIR, "phase37_multidomain.bin"),
    "loo_no_md":    os.path.join(DATA_DIR, "phase37_loo_no_md.bin"),
    "loo_no_code":  os.path.join(DATA_DIR, "phase37_loo_no_code.bin"),
    "loo_no_prose": os.path.join(DATA_DIR, "phase37_loo_no_prose.bin"),
}

# Eval corpora: Phase29A catalog (9 internal)
EVAL_CORPORA = {
    "natural_text":  os.path.join(DATA_DIR, "natural_text.txt"),
    "markdown_docs": os.path.join(DATA_DIR, "markdown_docs.md"),
    "c_code":        os.path.join(DATA_DIR, "c_code.c"),
    "shuffled":      os.path.join(DATA_DIR, "shuffled.bin"),
    "json_synth":    os.path.join(DATA_DIR, "json_synth.json"),
    "c_header":      os.path.join(DATA_DIR, "c_header_synth.h"),
    "notes_it":      os.path.join(DATA_DIR, "project_notes_it.txt"),
    "md_mixed":      os.path.join(DATA_DIR, "repo_markdown_mixed.md"),
    "log_synth":     os.path.join(DATA_DIR, "log_synth.log"),
}


# ── helpers ──────────────────────────────────────────────────────────────────

def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def compile_trainer():
    if os.path.exists(TRAINER_EXE):
        print(f"Trainer already compiled: {TRAINER_EXE}")
        return True
    print("Compiling train_entropy_readout.exe...")
    src_files = [
        TRAINER_SRC,
        os.path.join(ROOT, "src", "silicon_entropy.c"),
        os.path.join(ROOT, "src", "silicon_v0.c"),
    ]
    cmd = [
        "gcc", "-O3", "-march=native",
        "-o", TRAINER_EXE,
        *src_files,
        "-lm",
        "-I", ROOT,
    ]
    r = run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print("Compile FAILED:")
        print(r.stderr)
        return False
    print("Compiled OK.")
    return True


def train_weights(corpus_path, weights_path):
    if os.path.exists(weights_path):
        print(f"  Weights exist, skipping: {os.path.basename(weights_path)}")
        return True
    cmd = [
        TRAINER_EXE, corpus_path, weights_path,
        "--train-start", "0", "--train-len", "70",
        "--val-start",   "70", "--val-len",   "15",
    ]
    print(f"  Training {os.path.basename(corpus_path)} -> {os.path.basename(weights_path)} ...")
    r = run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print("  TRAIN FAILED:")
        print(r.stderr[-2000:])
        return False
    # Print last few lines of training log
    lines = r.stdout.strip().splitlines()
    for l in lines[-5:]:
        print(f"    {l}")
    return True


def audit_bpb(corpus_path, weights_path):
    """Run see.exe audit and parse Model BPB from output."""
    cmd = [
        SEE_EXE, "audit", corpus_path,
        "--weights", weights_path,
        "--expert-profile", "general",
    ]
    r = run(cmd, cwd=ROOT)
    output = r.stdout + r.stderr
    # "Model BPB:       2.XXXX"
    m = re.search(r"Model BPB:\s+([0-9.]+)", output)
    if not m:
        # fallback: "model_bpb: X.XXXX"
        m = re.search(r"model_bpb[:\s]+([0-9.]+)", output, re.IGNORECASE)
    if not m:
        return None
    return float(m.group(1))


def dominant_expert(corpus_path, weights_path):
    """Return dominant expert name from audit output."""
    cmd = [
        SEE_EXE, "audit", corpus_path,
        "--weights", weights_path,
        "--expert-profile", "general",
    ]
    r = run(cmd, cwd=ROOT)
    output = r.stdout + r.stderr
    m = re.search(r"dominant[:\s]+([A-Z_]+)", output, re.IGNORECASE)
    return m.group(1).upper() if m else "?"


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-corpus", action="store_true")
    ap.add_argument("--skip-train",  action="store_true")
    ap.add_argument("--skip-audit",  action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Step 1: Build corpora ──────────────────────────────────────────────
    if not args.skip_corpus:
        print("=" * 60)
        print("Step 1: Building training corpora")
        print("=" * 60)
        r = run([sys.executable, os.path.join(SCRIPTS_DIR, "phase37_corpus.py")], cwd=ROOT)
        print(r.stdout)
        if r.returncode != 0:
            print("Corpus build FAILED:", r.stderr)
            return 1
    else:
        print("(skipping corpus build)")

    # ── Step 2: Compile trainer ────────────────────────────────────────────
    if not args.skip_train:
        print("\n" + "=" * 60)
        print("Step 2: Compile trainer")
        print("=" * 60)
        if not compile_trainer():
            return 1

    # ── Step 3: Train weight sets ──────────────────────────────────────────
    if not args.skip_train:
        print("\n" + "=" * 60)
        print("Step 3: Train weight sets")
        print("=" * 60)
        for name, corpus_path in TRAINING_CORPORA.items():
            weights_path = WEIGHT_SETS[name]
            if not train_weights(corpus_path, weights_path):
                print(f"  FAILED: {name}")
                return 1

    # ── Step 4: Audit all corpora × all weight sets ────────────────────────
    results_path = os.path.join(RESULTS_DIR, "phase37_tribunal.json")

    if not args.skip_audit:
        print("\n" + "=" * 60)
        print("Step 4: Audit")
        print("=" * 60)

        all_weight_sets = {"v1_0": BASELINE_WEIGHTS, **WEIGHT_SETS}
        results = {}

        for corpus_name, corpus_path in EVAL_CORPORA.items():
            print(f"\n  Corpus: {corpus_name}")
            results[corpus_name] = {}
            for ws_name, ws_path in all_weight_sets.items():
                if not os.path.exists(ws_path):
                    print(f"    [{ws_name}] MISSING weights, skip")
                    continue
                bpb = audit_bpb(corpus_path, ws_path)
                if bpb is None:
                    print(f"    [{ws_name}] PARSE ERROR")
                    results[corpus_name][ws_name] = None
                else:
                    print(f"    [{ws_name}] {bpb:.4f} BPB")
                    results[corpus_name][ws_name] = bpb

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {results_path}")
    else:
        print("(skipping audit)")
        if not os.path.exists(results_path):
            print("No results file found. Run without --skip-audit first.")
            return 1
        with open(results_path) as f:
            results = json.load(f)

    # ── Step 5: Print delta matrix ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DELTA BPB vs V1.0 (negative = improvement)")
    print("=" * 60)

    ws_names = ["multidomain", "loo_no_md", "loo_no_code", "loo_no_prose"]
    col_w = 14

    # Header
    header = f"{'corpus':<18}" + "".join(f"{n:>{col_w}}" for n in ["v1_0"] + ws_names)
    print(header)
    print("-" * len(header))

    corpus_order = [
        "c_code", "natural_text", "markdown_docs", "md_mixed",
        "json_synth", "c_header", "log_synth", "notes_it", "shuffled",
    ]

    deltas_all = {n: [] for n in ws_names}

    for corpus_name in corpus_order:
        if corpus_name not in results:
            continue
        row = results[corpus_name]
        v1_bpb = row.get("v1_0")
        v1_str = f"{v1_bpb:.4f}" if v1_bpb is not None else "  N/A "
        line = f"{corpus_name:<18}{v1_str:>{col_w}}"
        for ws_name in ws_names:
            bpb = row.get(ws_name)
            if bpb is None or v1_bpb is None:
                line += f"{'N/A':>{col_w}}"
            else:
                delta = bpb - v1_bpb
                deltas_all[ws_name].append(delta)
                delta_str = ("+" if delta >= 0 else "") + f"{delta:.4f}"
                line += f"{delta_str:>{col_w}}"
        print(line)

    # Mean delta row
    print("-" * len(header))
    mean_line = f"{'MEAN delta':<18}{'':>{col_w}}"
    for ws_name in ws_names:
        ds = deltas_all[ws_name]
        if ds:
            mean_d = sum(ds) / len(ds)
            mean_str = ("+" if mean_d >= 0 else "") + f"{mean_d:.4f}"
            mean_line += f"{mean_str:>{col_w}}"
        else:
            mean_line += f"{'N/A':>{col_w}}"
    print(mean_line)

    # ── Step 6: Verdict ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    v1_0_bpb = {k: v.get("v1_0") for k, v in results.items()}
    md_bpb   = {k: v.get("multidomain") for k, v in results.items()}

    issues = []
    promotions = []

    # Check shuffled: must not get worse by more than 0.01 BPB
    for ws_name in ws_names:
        if results.get("shuffled", {}).get(ws_name) is not None:
            shuffled_delta = results["shuffled"][ws_name] - (results["shuffled"].get("v1_0") or 0)
            if shuffled_delta > 0.01:
                issues.append(f"{ws_name}: shuffled DEGRADED by {shuffled_delta:+.4f} BPB (prior over-fit on structure)")

    # Check c_code not degraded (threshold 0.01 BPB)
    for ws_name in ws_names:
        if results.get("c_code", {}).get(ws_name) is not None and results.get("c_code", {}).get("v1_0") is not None:
            c_delta = results["c_code"][ws_name] - results["c_code"]["v1_0"]
            if c_delta > 0.01:
                issues.append(f"{ws_name}: c_code DEGRADED by {c_delta:+.4f} BPB")

    # Check multidomain mean BPB vs v1_0
    if deltas_all["multidomain"]:
        mean_delta = sum(deltas_all["multidomain"]) / len(deltas_all["multidomain"])
        if mean_delta < -0.005:
            promotions.append(f"multidomain: mean improvement {mean_delta:.4f} BPB — GENERALIZES")
        elif mean_delta < 0.005:
            promotions.append(f"multidomain: mean delta {mean_delta:+.4f} BPB — NEUTRAL (capacity limit)")
        else:
            issues.append(f"multidomain: mean DEGRADATION {mean_delta:+.4f} BPB — DILUTION")

    # LOO notarial check: domain excluded -> that domain BPB should be worse
    loo_checks = [
        ("loo_no_md",    "markdown_docs", "markdown excluded -> markdown BPB should rise"),
        ("loo_no_code",  "c_code",        "c_code excluded -> c_code BPB should rise"),
        ("loo_no_prose", "natural_text",  "prose excluded -> natural_text BPB should rise"),
    ]
    print("\nLOO notarial check:")
    for ws_name, corpus_name, label in loo_checks:
        bpb_loo = results.get(corpus_name, {}).get(ws_name)
        bpb_md  = results.get(corpus_name, {}).get("multidomain")
        if bpb_loo is not None and bpb_md is not None:
            delta = bpb_loo - bpb_md
            result_str = "PASS (generalizes)" if delta > 0.002 else "FLAT (no effect)" if abs(delta) <= 0.002 else "INVERTED (unexpected improvement)"
            print(f"  {ws_name} on {corpus_name}: {delta:+.4f} BPB — {result_str}")
            print(f"    ({label})")
        else:
            print(f"  {ws_name} on {corpus_name}: N/A")

    print()
    if promotions:
        for p in promotions:
            print(f"PROMOTE: {p}")
    if issues:
        for i in issues:
            print(f"ISSUE:   {i}")
    if not promotions and not issues:
        print("NEUTRAL: no clear signal from multi-domain training.")

    # Capacity limit hypothesis
    print()
    print("Hypothesis test:")
    print("  If all LOO deltas are flat: 192D static prior lacks capacity for multi-domain representation.")
    print("  If LOO deltas are significant but multidomain is neutral: domains interfere at the readout level.")
    print("  If multidomain improves AND LOO degrades: generalisation confirmed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
