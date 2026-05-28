"""
Phase 29C: Catalog Hygiene & Baseline Freeze

Goals:
  1. Replace mislabeled catalog entries (json_real was a README, not JSON)
  2. Add real prose corpus from Project Gutenberg pg19 (Jennie Gerhardt, 512KB)
  3. Run full 4-profile matrix on clean external catalog
  4. Freeze results as data/baselines/phase29c_baseline.json

Catalog design rules:
  - Each file must have a domain label that matches its actual content
  - domain: prose | code | markdown | log | json | binary
  - Mislabeled files from 29B are excluded from domain conclusions
  - Files must be >= 64 KB for MoE to reach steady state (readme_real.md logged as SMALL)

Clean catalog (29C):
  prose_real.txt   Jennie Gerhardt (512KB) — English narrative prose, Dreiser 1911
  c_real.c         zlib inflate.c  (51KB)  — real C source, Zlib license
  log_real.log     Apache log      (2.3MB) — real server log, highly repetitive
  readme_real.md   LZ4 README      (4KB)   — markdown, SMALL flag (< 64KB)

Excluded from 29C (catalog errors from 29B):
  json_real.json   awesome-json-datasets README.md — was markdown, NOT JSON. Excluded.

Files already tested in 29A (historical, internal):
  natural_text.txt / markdown_docs.md / c_code.c / shuffled.bin / ...

Baseline freeze format (data/baselines/phase29c_baseline.json):
  {
    "version": "29C",
    "tolerance_bpb": 0.005,
    "excluded": ["json_real.json"],
    "corpora": {
      "<name>": {
        "domain": "<label>",
        "size_bytes": N,
        "small_flag": bool,
        "<profile>": {"quant_bpb": N, "dominant": "X", ...},
        ...
      }
    }
  }
"""

import os
import re
import json
import subprocess
import hashlib
import sys
import io

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEE     = os.path.join(ROOT, "see.exe")
WEIGHTS = os.path.join(ROOT, "weights", "entropy_weights_factors_r16.bin")
EXT_DIR = os.path.join(ROOT, "data", "external")
BASEDIR = os.path.join(ROOT, "data", "baselines")
RESULTS = os.path.join(ROOT, "results")

SMALL_THRESHOLD = 64 * 1024  # bytes — below this MoE hasn't enough context

# Clean catalog: (filename, domain, notes)
CATALOG = [
    ("prose_real.txt", "prose",    "Jennie Gerhardt by Theodore Dreiser (1911), 512KB from pg19"),
    ("c_real.c",       "code",     "zlib inflate.c — real C source, complex control flow, Zlib license"),
    ("log_real.log",   "log",      "Apache access log, real server log, 2.3MB, Apache-2.0"),
    ("readme_real.md", "markdown", "LZ4 README.md, BSD-2-Clause — SMALL FLAG (<64KB)"),
]

EXCLUDED_29B = {
    "json_real.json": "CATALOG ERROR — was awesome-json-datasets README.md (markdown), not JSON. "
                      "Excluded from all domain conclusions."
}

PROFILES = [
    ("general",  ["--expert-profile", "general"]),
    ("prose",    ["--expert-profile", "prose"]),
    ("span-pfx", ["--expert-profile", "general", "--span-pfx"]),
    ("no-token", ["--expert-profile", "experimental", "--lz-key", "6"]),
]

REGRESSION_THRESH = 0.005


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


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


def fmt(v, spec=".4f"):
    return f"{v:{spec}}" if v is not None else "  N/A "

def d(a, b):
    if a is None or b is None: return "   N/A"
    return f"{a - b:+.4f}"


def main():
    os.makedirs(BASEDIR, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    lines = []
    def p(s=""): print(s); lines.append(s)

    p("=" * 90)
    p("PHASE 29C: Catalog Hygiene & Baseline Freeze")
    p("=" * 90)
    p()

    # ── Catalog error report ───────────────────────────────────────────────────
    p("CATALOG CORRECTIONS (vs Phase 29B)")
    p("-" * 60)
    for fname, reason in EXCLUDED_29B.items():
        p(f"  EXCLUDED: {fname}")
        p(f"    Reason: {reason}")
    p()

    # ── Verify files exist ─────────────────────────────────────────────────────
    p("CATALOG VERIFICATION")
    p("-" * 60)
    valid = []
    for fname, domain, notes in CATALOG:
        path = os.path.join(EXT_DIR, fname)
        if not os.path.exists(path):
            p(f"  [MISSING] {fname} — {path}")
            continue
        sz = os.path.getsize(path)
        sha = sha256_file(path)
        small = sz < SMALL_THRESHOLD
        flag = "  [SMALL]" if small else ""
        p(f"  [OK]{flag}  {fname:<22}  {sz//1024:5d} KB  domain={domain:<9}  sha={sha[:12]}...")
        valid.append((fname, path, domain, notes, sz, small))
    p()

    # ── Audit matrix ───────────────────────────────────────────────────────────
    all_r = {}

    for fname, fpath, domain, notes, sz_bytes, small_flag in valid:
        p(f"{'='*90}")
        p(f"CORPUS: {fname}  domain={domain}  size={sz_bytes//1024}KB"
          + ("  [SMALL — MoE context limited]" if small_flag else ""))
        p(f"  {notes}")
        p(f"{'-'*90}")

        for prof_name, prof_args in PROFILES:
            print(f"  {prof_name:<10}...", end="", flush=True)
            m = run_audit(fpath, prof_args)
            all_r[(fname, prof_name)] = m
            dom = m.get("dominant") or "?"
            cyc = f"{m['cycles_byte']:.0f}" if m.get("cycles_byte") else "?"
            print(f" {fmt(m['quant_bpb'])} BPB  dom={dom:<7}  cyc/B={cyc}")

        gen  = all_r.get((fname, "general"),  {}).get("quant_bpb")
        pros = all_r.get((fname, "prose"),    {}).get("quant_bpb")
        notk = all_r.get((fname, "no-token"), {}).get("quant_bpb")
        uni  = all_r.get((fname, "general"),  {}).get("unigram_bpb")

        p()
        p(f"  prose vs general:    {d(pros, gen)}")
        p(f"  no-token vs general: {d(notk, gen)}  (TOKPFX value)")
        if uni and gen:
            p(f"  general vs unigram:  {d(gen, uni)}  (total gain)")

    # ── Grand matrix ───────────────────────────────────────────────────────────
    p()
    p("=" * 90)
    p("EXTERNAL CORPUS MATRIX (29C clean catalog) — Quantized BPB")
    p("=" * 90)
    col_w = 12
    header = f"{'corpus':<24}{'domain':<10}" + "".join(f"{pn:>{col_w}}" for pn, _ in PROFILES) + f"{'dominant':>{col_w}}"
    p(header)
    p("-" * len(header))
    for fname, _, domain, _, _, small_flag in valid:
        dom = all_r.get((fname, "general"), {}).get("dominant") or "?"
        sflag = "*" if small_flag else " "
        row = (f"{fname+sflag:<24}{domain:<10}"
               + "".join(f"{fmt(all_r.get((fname, pn), {}).get('quant_bpb')):>{col_w}}"
                         for pn, _ in PROFILES)
               + f"{dom:>{col_w}}")
        p(row)
    p("  * = SMALL flag (< 64KB), MoE context limited")

    p()
    p("Delta vs general (prose / span-pfx / no-token):")
    p(f"{'corpus':<24}" + "".join(f"{'D'+pn:>{col_w}}" for pn, _ in PROFILES if pn != "general"))
    p("-" * (24 + col_w * (len(PROFILES) - 1)))
    for fname, _, domain, _, _, _ in valid:
        gen_bpb = all_r.get((fname, "general"), {}).get("quant_bpb")
        row = f"{fname:<24}" + "".join(
            f"{d(all_r.get((fname, pn), {}).get('quant_bpb'), gen_bpb):>{col_w}}"
            for pn, _ in PROFILES if pn != "general")
        p(row)

    # ── Domain verdicts ────────────────────────────────────────────────────────
    p()
    p("=" * 90)
    p("DOMAIN VERDICTS")
    p("=" * 90)
    p()

    for fname, _, domain, _, sz_bytes, small_flag in valid:
        gen  = all_r.get((fname, "general"),  {}).get("quant_bpb")
        pros = all_r.get((fname, "prose"),    {}).get("quant_bpb")
        notk = all_r.get((fname, "no-token"), {}).get("quant_bpb")
        uni  = all_r.get((fname, "general"),  {}).get("unigram_bpb")
        dom  = all_r.get((fname, "general"),  {}).get("dominant") or "?"

        prose_delta = (pros - gen) if pros and gen else None
        tokpfx_val  = (notk - gen) if notk and gen else None

        p(f"  {fname}  [{domain.upper()}]")
        if small_flag:
            p(f"    [SMALL] Results indicative only — corpus < 64KB")
        if prose_delta is not None:
            verdict = "OK" if prose_delta <= 0.01 else "WARN"
            p(f"    prose delta: {prose_delta:+.4f}  [{verdict}]  (limit: +0.010)")
        if tokpfx_val is not None:
            p(f"    TOKPFX saves: {tokpfx_val:+.4f}")
        p(f"    dominant expert: {dom}")
        p()

    p("Excluded from domain verdicts:")
    for fname, reason in EXCLUDED_29B.items():
        p(f"  {fname}: {reason}")

    # ── TOKPFX aggregate ───────────────────────────────────────────────────────
    p()
    p("TOKPFX value summary (external, clean catalog only):")
    gains = []
    for fname, _, domain, _, _, small_flag in valid:
        gen_b  = all_r.get((fname, "general"),  {}).get("quant_bpb")
        notk_b = all_r.get((fname, "no-token"), {}).get("quant_bpb")
        if gen_b and notk_b:
            gains.append((fname, domain, notk_b - gen_b, small_flag))
    for fname, domain, gain, small_flag in sorted(gains, key=lambda x: -x[2]):
        flag = "  [SMALL]" if small_flag else ""
        p(f"  {fname:<24}  domain={domain:<9}  TOKPFX saves {gain:+.4f} BPB{flag}")
    valid_gains = [g for _, _, g, sf in gains if not sf]
    if valid_gains:
        avg = sum(valid_gains) / len(valid_gains)
        p(f"  Mean (non-small corpora): {avg:+.4f} BPB")

    # ── Baseline freeze ────────────────────────────────────────────────────────
    p()
    p("=" * 90)
    p("BASELINE FREEZE — data/baselines/phase29c_baseline.json")
    p("=" * 90)

    freeze = {
        "version": "29C",
        "description": "External corpus baseline — clean catalog, 4 profiles",
        "tolerance_bpb": REGRESSION_THRESH,
        "excluded_catalog_errors": {
            k: v for k, v in EXCLUDED_29B.items()
        },
        "corpora": {}
    }

    for fname, _, domain, notes, sz_bytes, small_flag in valid:
        entry = {
            "domain": domain,
            "notes": notes,
            "size_bytes": sz_bytes,
            "small_flag": small_flag,
        }
        for prof_name, _ in PROFILES:
            m = all_r.get((fname, prof_name), {})
            entry[prof_name] = {
                "quant_bpb":   m.get("quant_bpb"),
                "model_bpb":   m.get("model_bpb"),
                "unigram_bpb": m.get("unigram_bpb"),
                "cycles_byte": m.get("cycles_byte"),
                "dominant":    m.get("dominant"),
                "avg_weights": m.get("avg_weights", {}),
            }
        freeze["corpora"][fname] = entry

    baseline_path = os.path.join(BASEDIR, "phase29c_baseline.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(freeze, f, indent=2)
    p(f"  Written: {baseline_path}")
    p(f"  Corpora frozen: {len(freeze['corpora'])}")
    p(f"  Profiles: {[pn for pn, _ in PROFILES]}")
    p(f"  Tolerance: {REGRESSION_THRESH} BPB")

    # ── Summary ────────────────────────────────────────────────────────────────
    p()
    p("=" * 90)
    p("PHASE 29C SUMMARY")
    p("=" * 90)
    p()
    p("  Catalog corrections:")
    p(f"    EXCLUDED: json_real.json (29B catalog error — was markdown, not JSON)")
    p(f"    ADDED:    prose_real.txt (Jennie Gerhardt, 512KB, Dreiser 1911)")
    p()
    p("  Baseline freeze: data/baselines/phase29c_baseline.json")
    p("  This is the authoritative external baseline for future regression testing.")
    p()
    p("  Reminder: external baselines are stress test confirmation.")
    p("  Phase 29A internal baseline (phase29a_baseline.json) drives architecture decisions.")

    rp = os.path.join(RESULTS, "phase29c_catalog_freeze.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport -> {rp}")
    print(f"Baseline -> {baseline_path}")


if __name__ == "__main__":
    main()
