"""
Phase 34: Credit-Only Regime Router Tribunal

Research question:
    Can compression credit alone (per-expert win rates, weight velocity)
    replace hand-coded char-class thresholds for regime detection?

Compares three configurations on the frozen Phase 29C catalog:
  A. baseline      — general/prose, no regime router
  B. regime-prior  — Phase 33B (char-class EMA + hardcoded thresholds)
  C. regime-credit — Phase 34 (win-rate EMA, entropy-gated injection)

For each corpus × profile, reports:
  - BPB for A, B, C
  - delta B vs A  (Phase 33B gain, the target to beat)
  - delta C vs A  (Credit Router gain)
  - delta C vs B  (direct comparison)

Pass criteria (same as Phase 33B verdict):
  - markdown_docs  / general:  C vs A < 0.0 BPB  (improvement)
  - prose_real     / prose:    C vs A < 0.0 BPB
  - shuffled       / general:  C vs A < +0.005 BPB  (no regression)
  - c_real         / general:  C vs A < +0.005 BPB
  - log_real       / general:  neutral or better

Output:
  results/phase34_credit_router.txt  — full comparison table + verdict
"""

import os
import re
import sys
import io
import subprocess

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEE     = os.path.join(ROOT, "see_research.exe")  # research binary
WEIGHTS = os.path.join(ROOT, "weights", "entropy_weights_factors_r16.bin")
DATA    = os.path.join(ROOT, "data")
EXT_DIR = os.path.join(DATA, "external")
RESULTS = os.path.join(ROOT, "results")

os.makedirs(RESULTS, exist_ok=True)

# --- Corpus catalog (Phase 29C frozen + key external files) -----------------

CORPORA = [
    # (label, path, profile, target_improvement)
    # target_improvement: "up" = want negative delta, "flat" = tolerate ±0.005
    ("markdown_docs",  os.path.join(DATA,    "markdown_docs.md"),    "general", "up"),
    ("md_mixed",       os.path.join(DATA,    "repo_markdown_mixed.md"), "general", "up"),
    ("natural_text",   os.path.join(DATA,    "natural_text.txt"),    "general", "up"),
    ("prose_real",     os.path.join(EXT_DIR, "prose_real.txt"),      "prose",   "up"),
    ("c_code",         os.path.join(DATA,    "c_code.c"),            "general", "flat"),
    ("c_real",         os.path.join(EXT_DIR, "c_real.c"),            "general", "flat"),
    ("shuffled",       os.path.join(DATA,    "shuffled.bin"),        "general", "flat"),
    ("log_synth",      os.path.join(DATA,    "log_synth.log"),       "general", "flat"),
    ("log_real",       os.path.join(EXT_DIR, "log_real.log"),        "general", "flat"),
    ("json_synth",     os.path.join(DATA,    "json_synth.json"),     "general", "flat"),
]

PROFILE_BASE_ARGS = {
    "general": ["--expert-profile", "general"],
    "prose":   ["--expert-profile", "prose"],
}

IMPROVEMENT_THRESH = -0.001   # must beat baseline by at least this (BPB)
REGRESSION_THRESH  =  0.005   # must not regress by more than this (BPB)


def run_audit(fpath, profile, extra_flags=()):
    base_args = PROFILE_BASE_ARGS.get(profile, ["--expert-profile", profile])
    cmd = [SEE, "audit", fpath, "--weights", WEIGHTS, "--blend", "moe"] + base_args + list(extra_flags)
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        out = res.stdout + res.stderr
        m = re.search(r"Quantized BPB:\s+([\d.]+)", out)
        return float(m.group(1)) if m else None
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def fmt_delta(d):
    if d is None:
        return "  N/A  "
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:+.4f}"


def verdict_symbol(delta_c_vs_a, target):
    if delta_c_vs_a is None:
        return "?"
    if target == "up":
        return "PASS" if delta_c_vs_a < IMPROVEMENT_THRESH else "FAIL"
    else:  # flat
        return "PASS" if delta_c_vs_a < REGRESSION_THRESH else "FAIL"


lines = []
def log(s=""):
    print(s)
    lines.append(s)


log("=" * 78)
log("Phase 34: Credit-Only Regime Router Tribunal")
log("=" * 78)
log(f"Binary:  {SEE}")
log(f"Weights: {WEIGHTS}")
log()
log(f"  {'Corpus':<22}  {'Profile':<8}  {'A:base':>8}  {'B:prior':>8}  {'C:credit':>8}  "
    f"{'C-A':>8}  {'B-A':>8}  {'Verdict'}")
log("-" * 78)

results = {}
n_pass = n_fail = 0

for label, fpath, profile, target in CORPORA:
    if not os.path.exists(fpath):
        log(f"  {label:<22}  {profile:<8}  [SKIP — file not found]")
        continue

    bpb_a = run_audit(fpath, profile, [])
    bpb_b = run_audit(fpath, profile, ["--regime-prior"])
    bpb_c = run_audit(fpath, profile, ["--regime-credit"])

    d_c_a = (bpb_c - bpb_a) if (bpb_c is not None and bpb_a is not None) else None
    d_b_a = (bpb_b - bpb_a) if (bpb_b is not None and bpb_a is not None) else None
    verdict = verdict_symbol(d_c_a, target)

    if verdict == "PASS":
        n_pass += 1
    else:
        n_fail += 1

    a_s = f"{bpb_a:.4f}" if bpb_a is not None else "  N/A"
    b_s = f"{bpb_b:.4f}" if bpb_b is not None else "  N/A"
    c_s = f"{bpb_c:.4f}" if bpb_c is not None else "  N/A"

    log(f"  {label:<22}  {profile:<8}  {a_s:>8}  {b_s:>8}  {c_s:>8}  "
        f"{fmt_delta(d_c_a):>8}  {fmt_delta(d_b_a):>8}  {verdict}")

    results[label] = {
        "profile": profile, "target": target,
        "bpb_baseline": bpb_a, "bpb_regime_prior": bpb_b, "bpb_regime_credit": bpb_c,
        "delta_credit_vs_base": d_c_a, "delta_prior_vs_base": d_b_a,
        "verdict": verdict,
    }

log("-" * 78)
log(f"  PASS: {n_pass}   FAIL: {n_fail}   TOTAL: {n_pass + n_fail}")
log()

# Summary: where does credit beat prior, where does it lose?
beats = [(k, v) for k, v in results.items()
         if v.get("delta_credit_vs_base") is not None and v.get("delta_prior_vs_base") is not None
         and v["delta_credit_vs_base"] < v["delta_prior_vs_base"]]
loses = [(k, v) for k, v in results.items()
         if v.get("delta_credit_vs_base") is not None and v.get("delta_prior_vs_base") is not None
         and v["delta_credit_vs_base"] > v["delta_prior_vs_base"]]

if beats:
    log("Credit BEATS prior on:")
    for k, v in beats:
        log(f"  {k:<22}  C-A={v['delta_credit_vs_base']:+.4f}  B-A={v['delta_prior_vs_base']:+.4f}")
if loses:
    log("Credit LOSES to prior on:")
    for k, v in loses:
        log(f"  {k:<22}  C-A={v['delta_credit_vs_base']:+.4f}  B-A={v['delta_prior_vs_base']:+.4f}")

log()
overall = "PROMOTED" if n_fail == 0 else "NEEDS WORK" if n_fail <= 2 else "FAIL"
log(f"Verdict: {overall}")
log()
log("Tunables used:")
log("  CREDIT_WIN_WINDOW     = 64")
log("  CREDIT_VEL_WINDOW     = 32")
log("  CREDIT_CHECK_EVERY    = 16")
log("  CREDIT_WARMUP         = 128")
log("  CREDIT_GAMMA          = 0.25")
log("  CREDIT_ENTROPY_THRESH = 0.72")

out_path = os.path.join(RESULTS, "phase34_credit_router.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"\n[saved to {out_path}]")
