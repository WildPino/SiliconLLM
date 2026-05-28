"""
gen_baseline_29a.py  —  Save Phase 29A tribunal results as a regression baseline.

Reads the Phase 29A results from already-computed all_r dict by re-running the
audit on all corpora (or loading from the existing report if available), then
writes data/baselines/phase29a_baseline.json.

The baseline JSON is used by future tribunals to detect regressions:
  new_bpb > baseline_bpb + REGRESSION_THRESH  →  flag regression

Format:
  {
    "version": "29A",
    "date": "2026-05-25",
    "profiles": ["general", "prose", "span-pfx", "no-token"],
    "corpora": {
      "<corpus_name>": {
        "<profile>": {
          "quant_bpb": float,
          "model_bpb": float,
          "unigram_bpb": float,
          "cycles_byte": float,
          "dominant": str,
          "avg_weights": {label: float, ...}
        }
      }
    }
  }
"""

import os
import re
import json
import subprocess
import time
import sys
import io

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEE      = os.path.join(ROOT, "see.exe")
WEIGHTS  = os.path.join(ROOT, "weights", "entropy_weights_factors_r16.bin")
DATA     = os.path.join(ROOT, "data")
BASELINE_DIR = os.path.join(DATA, "baselines")
BASELINE_OUT = os.path.join(BASELINE_DIR, "phase29a_baseline.json")

CORPORA = [
    ("natural_text",  os.path.join(DATA, "natural_text.txt")),
    ("markdown_docs", os.path.join(DATA, "markdown_docs.md")),
    ("c_code",        os.path.join(DATA, "c_code.c")),
    ("shuffled",      os.path.join(DATA, "shuffled.bin")),
    ("json_synth",    os.path.join(DATA, "json_synth.json")),
    ("c_header",      os.path.join(DATA, "c_header_synth.h")),
    ("notes_it",      os.path.join(DATA, "project_notes_it.txt")),
    ("md_mixed",      os.path.join(DATA, "repo_markdown_mixed.md")),
    ("log_synth",     os.path.join(DATA, "log_synth.log")),
]

PROFILES = [
    ("general",  ["--expert-profile", "general"]),
    ("prose",    ["--expert-profile", "prose"]),
    ("span-pfx", ["--expert-profile", "general", "--span-pfx"]),
    ("no-token", ["--expert-profile", "experimental", "--lz-key", "6"]),
]


def run_audit(filepath, extra):
    res = subprocess.run(
        [SEE, "audit", filepath, "--weights", WEIGHTS, "--blend", "moe"] + extra,
        capture_output=True, text=True, timeout=900)
    out = res.stdout + res.stderr

    def g(pat, cast=float):
        m = re.search(pat, out)
        return cast(m.group(1)) if m else None

    r = {}
    r["quant_bpb"]   = g(r"Quantized BPB:\s+([\d.]+)")
    r["model_bpb"]   = g(r"Model BPB:\s+([\d.]+)")
    r["unigram_bpb"] = g(r"Unigram BPB:\s+([\d.]+)")
    r["cycles_byte"] = g(r"Cycles/byte:\s+([\d.]+)")

    avg_m = re.search(r"Avg\s+\[(.*?)\]:\s+(.*)", out)
    r["avg_weights"] = {}
    r["dominant"]    = None
    if avg_m:
        labels = avg_m.group(1).split()
        vals   = list(map(float, avg_m.group(2).split()))
        r["avg_weights"] = dict(zip(labels, vals))
        if labels:
            r["dominant"] = labels[vals.index(max(vals))]
    return r


def main():
    os.makedirs(BASELINE_DIR, exist_ok=True)

    if os.path.exists(BASELINE_OUT):
        print(f"Baseline already exists: {BASELINE_OUT}")
        print("Delete it and re-run to regenerate.")
        return

    baseline = {
        "version": "29A",
        "date": "2026-05-25",
        "note": ("Phase 29A regression baseline. "
                 "Profile 'prose' replaces 'text' (text kept as alias). "
                 "Tolerance for regression checks: 0.005 BPB."),
        "regression_threshold_bpb": 0.005,
        "profiles": [p for p, _ in PROFILES],
        "corpora": {},
    }

    valid = [(n, p) for n, p in CORPORA if os.path.exists(p)]
    total = len(valid) * len(PROFILES)
    done  = 0

    for ds_name, ds_path in valid:
        baseline["corpora"][ds_name] = {}
        sz_kb = os.path.getsize(ds_path) // 1024
        for prof_name, prof_args in PROFILES:
            done += 1
            print(f"[{done}/{total}] {ds_name} / {prof_name} ({sz_kb} KB)...",
                  end="", flush=True)
            t0 = time.perf_counter()
            m  = run_audit(ds_path, prof_args)
            elapsed = time.perf_counter() - t0
            baseline["corpora"][ds_name][prof_name] = m
            print(f" {m.get('quant_bpb', 'N/A'):.4f} BPB  dom={m.get('dominant','?')}  ({elapsed:.1f}s)")

    with open(BASELINE_OUT, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)

    print(f"\nBaseline saved: {BASELINE_OUT}")
    print(f"Corpora: {len(baseline['corpora'])}  Profiles: {len(PROFILES)}")


if __name__ == "__main__":
    main()
