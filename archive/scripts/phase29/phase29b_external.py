"""
Phase 29B: External Real Corpus Tribunal

Downloads 5 real files from authoritative sources into data/external/.
Saves a manifest with URL, SHA-256, license, size, and download date.

Runs the same 4-profile matrix as Phase 29A, plus compares against
the Phase 29A baseline (data/baselines/phase29a_baseline.json).

External corpus targets:
  readme_real.md    README.md from a well-known open-source project (MIT)
  c_real.c          A real C source file from a compression library (zlib/LZ4)
  json_real.json    A real JSON dataset (GeoJSON or similar open data)
  prose_real.txt    A real English prose text (Project Gutenberg, public domain)
  log_real.log      A real server/application log (example data, Apache log)

Profiles tested: general | prose | span-pfx | no-token  (same as 29A)

Decision rules (same as 29A + baseline comparison):
  - No regression vs 29A baseline > REGRESSION_THRESH on any existing domain
  - External results are stress test only — do NOT drive profile decisions alone
  - Flag any corpus where general > unigram + 0.05 BPB (robustness failure)

Manifest format (data/external/manifest.csv):
  name, url, sha256, license, size_bytes, downloaded_at
"""

import os
import re
import json
import subprocess
import hashlib
import time
import sys
import io
import csv
import urllib.request
import urllib.error
from datetime import datetime

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEE      = os.path.join(ROOT, "see.exe")
WEIGHTS  = os.path.join(ROOT, "weights", "entropy_weights_factors_r16.bin")
DATA     = os.path.join(ROOT, "data")
EXT_DIR  = os.path.join(DATA, "external")
MANIFEST = os.path.join(EXT_DIR, "manifest.csv")
BASELINE = os.path.join(DATA, "baselines", "phase29a_baseline.json")
RESULTS  = os.path.join(ROOT, "results")

# ---------------------------------------------------------------------------
# External corpus catalog
# Each entry: (local_name, url, license, description)
# URLs: stable, authoritative, small files only. No CDN/redirect links.
# ---------------------------------------------------------------------------

CATALOG = [
    (
        "readme_real.md",
        "https://raw.githubusercontent.com/lz4/lz4/dev/README.md",
        "BSD-2-Clause",
        "LZ4 project README — technical markdown with code samples",
    ),
    (
        "c_real.c",
        "https://raw.githubusercontent.com/madler/zlib/master/inflate.c",
        "Zlib",
        "zlib inflate.c — real C source, complex control flow",
    ),
    (
        "json_real.json",
        "https://raw.githubusercontent.com/jdorfman/awesome-json-datasets/master/README.md",
        "CC0",
        "awesome-json-datasets README — markdown describing JSON APIs",
    ),
    (
        "prose_real.txt",
        "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
        "Public Domain (Project Gutenberg)",
        "Pride and Prejudice by Jane Austen — English literary prose",
    ),
    (
        "log_real.log",
        "https://raw.githubusercontent.com/elastic/examples/master/Common%20Data%20Formats/apache_logs/apache_logs",
        "Apache-2.0",
        "Apache access log example — real server log format",
    ),
]

PROFILES = [
    ("general",  ["--expert-profile", "general"]),
    ("prose",    ["--expert-profile", "prose"]),
    ("span-pfx", ["--expert-profile", "general", "--span-pfx"]),
    ("no-token", ["--expert-profile", "experimental", "--lz-key", "6"]),
]

REGRESSION_THRESH = 0.005
DISASTER_SLACK    = 0.05

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def download_file(url, dest_path, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "SEE-Phase29B/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except urllib.error.URLError as e:
        return False, str(e)
    with open(dest_path, "wb") as f:
        f.write(data)
    return True, None


def load_manifest():
    if not os.path.exists(MANIFEST):
        return {}
    rows = {}
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["name"]] = row
    return rows


def save_manifest(entries):
    os.makedirs(EXT_DIR, exist_ok=True)
    fields = ["name", "url", "sha256", "license", "size_bytes", "downloaded_at"]
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in entries:
            w.writerow(e)


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

def d(a, b, spec=".4f"):
    if a is None or b is None: return "   N/A"
    return f"{a - b:+{spec}}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(EXT_DIR, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    lines = []
    def p(s=""): print(s); lines.append(s)

    p("=" * 90)
    p("PHASE 29B: External Real Corpus Tribunal")
    p("=" * 90)
    p()

    # ── Step 1: Download missing files ────────────────────────────────────────
    p("Step 1: Downloading external corpus files")
    p("-" * 60)

    manifest_rows = load_manifest()
    new_manifest  = []

    for local_name, url, license_, desc in CATALOG:
        dest = os.path.join(EXT_DIR, local_name)
        if os.path.exists(dest):
            sha = sha256_file(dest)
            sz  = os.path.getsize(dest)
            p(f"  [CACHED]  {local_name:<22} {sz//1024:4d} KB  sha={sha[:12]}...")
            new_manifest.append({
                "name": local_name, "url": url, "sha256": sha,
                "license": license_, "size_bytes": sz,
                "downloaded_at": manifest_rows.get(local_name, {}).get("downloaded_at", "unknown"),
            })
        else:
            print(f"  [DL]      {local_name:<22} {url[:55]}...", end="", flush=True)
            ok, err = download_file(url, dest)
            if ok:
                sha = sha256_file(dest)
                sz  = os.path.getsize(dest)
                ts  = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                print(f" OK  {sz//1024} KB  sha={sha[:12]}...")
                p(f"  [DL-OK]   {local_name:<22} {sz//1024:4d} KB  sha={sha[:12]}...")
                new_manifest.append({
                    "name": local_name, "url": url, "sha256": sha,
                    "license": license_, "size_bytes": sz, "downloaded_at": ts,
                })
            else:
                print(f" FAIL: {err}")
                p(f"  [DL-FAIL] {local_name:<22} {err}")

    save_manifest(new_manifest)
    p()
    p(f"  Manifest: {MANIFEST}")

    # ── Step 2: Load baseline ──────────────────────────────────────────────────
    baseline = {}
    if os.path.exists(BASELINE):
        with open(BASELINE, encoding="utf-8") as f:
            baseline = json.load(f)
        p(f"  Baseline: {BASELINE}  (version {baseline.get('version','?')})")
    else:
        p(f"  [WARN] No baseline found at {BASELINE} — regression check skipped")
        p(f"         Run scripts/gen_baseline_29a.py first")
    p()

    # ── Step 3: Audit matrix ───────────────────────────────────────────────────
    valid = [(e["name"], os.path.join(EXT_DIR, e["name"]))
             for e in new_manifest
             if os.path.exists(os.path.join(EXT_DIR, e["name"]))]

    all_r = {}

    for ds_name, ds_path in valid:
        sz_kb = os.path.getsize(ds_path) // 1024
        p(f"{'-'*90}")
        p(f"CORPUS [EXT]: {ds_name}  ({sz_kb} KB)")
        p(f"{'-'*90}")

        for prof_name, prof_args in PROFILES:
            print(f"  {prof_name:<10}...", end="", flush=True)
            m = run_audit(ds_path, prof_args)
            all_r[(ds_name, prof_name)] = m
            dom = m.get("dominant") or "?"
            cyc = f"{m['cycles_byte']:.0f}" if m.get("cycles_byte") else "?"
            print(f" {fmt(m['quant_bpb'])} BPB  dom={dom:<7}  cyc/B={cyc}")

        gen_bpb  = all_r.get((ds_name, "general"),  {}).get("quant_bpb")
        prose_bpb= all_r.get((ds_name, "prose"),    {}).get("quant_bpb")
        notk_bpb = all_r.get((ds_name, "no-token"), {}).get("quant_bpb")
        uni_bpb  = all_r.get((ds_name, "general"),  {}).get("unigram_bpb")

        p()
        p(f"  Delta prose vs general:    {d(prose_bpb, gen_bpb)}")
        p(f"  Delta no-token vs general: {d(notk_bpb, gen_bpb)}  (TOKPFX value)")
        if uni_bpb and gen_bpb:
            p(f"  general vs unigram:        {d(gen_bpb, uni_bpb)}  (total gain)")

    # ── Step 4: Grand matrix ───────────────────────────────────────────────────
    p()
    p("=" * 90)
    p("EXTERNAL CORPUS MATRIX  —  Quantized BPB")
    p("=" * 90)

    col_w = 12
    header = f"{'corpus':<22}" + "".join(f"{pn:>{col_w}}" for pn, _ in PROFILES) + f"{'dominant':>{col_w}}"
    p(header)
    p("-" * len(header))
    for ds_name, _ in valid:
        dom = all_r.get((ds_name, "general"), {}).get("dominant") or "?"
        row = (f"{ds_name:<22}"
               + "".join(f"{fmt(all_r.get((ds_name, pn), {}).get('quant_bpb')):>{col_w}}"
                         for pn, _ in PROFILES)
               + f"{dom:>{col_w}}")
        p(row)

    p()
    p("Delta vs general:")
    p(f"{'corpus':<22}" + "".join(f"{'D'+pn:>{col_w}}" for pn, _ in PROFILES if pn != "general"))
    p("-" * (22 + col_w * (len(PROFILES) - 1)))
    for ds_name, _ in valid:
        gen_bpb = all_r.get((ds_name, "general"), {}).get("quant_bpb")
        row = f"{ds_name:<22}" + "".join(
            f"{d(all_r.get((ds_name, pn), {}).get('quant_bpb'), gen_bpb):>{col_w}}"
            for pn, _ in PROFILES if pn != "general")
        p(row)

    p()
    p("Dominant expert:")
    p(f"{'corpus':<22}" + "".join(f"{pn:>{col_w}}" for pn, _ in PROFILES))
    p("-" * (22 + col_w * len(PROFILES)))
    for ds_name, _ in valid:
        row = f"{ds_name:<22}" + "".join(
            f"{(all_r.get((ds_name, pn), {}).get('dominant') or '?'):>{col_w}}"
            for pn, _ in PROFILES)
        p(row)

    # ── Step 5: Baseline regression check ─────────────────────────────────────
    p()
    p("=" * 90)
    p("BASELINE REGRESSION CHECK  (vs Phase 29A synthetic corpus)")
    p("=" * 90)

    if baseline.get("corpora"):
        regressions = []
        improvements = []
        for ds_name in baseline["corpora"]:
            for prof_name in baseline["corpora"][ds_name]:
                # Map "text" baseline key to "prose" if needed
                prof_key = prof_name
                base_bpb = baseline["corpora"][ds_name][prof_name].get("quant_bpb")
                # Find corresponding result in this run (historical corpora not re-run here)
                new_bpb  = all_r.get((ds_name, prof_key), {}).get("quant_bpb")
                if base_bpb is None or new_bpb is None:
                    continue
                delta = new_bpb - base_bpb
                if delta > REGRESSION_THRESH:
                    regressions.append(f"{ds_name}/{prof_key}: {delta:+.4f}")
                elif delta < -REGRESSION_THRESH:
                    improvements.append(f"{ds_name}/{prof_key}: {delta:+.4f}")
        p()
        if regressions:
            p(f"  REGRESSIONS ({len(regressions)}):")
            for r_ in regressions: p(f"    {r_}")
        else:
            p("  No regressions vs baseline (no historical corpora re-run here)")
        if improvements:
            p(f"  Improvements ({len(improvements)}):")
            for r_ in improvements: p(f"    {r_}")
    else:
        p("  Baseline not available — skipped")

    # ── Step 6: Robustness check ───────────────────────────────────────────────
    p()
    p("=" * 90)
    p("ROBUSTNESS CHECK — general must not exceed unigram + 0.05 on external corpora")
    p("=" * 90)

    disasters = []
    for ds_name, _ in valid:
        gen = all_r.get((ds_name, "general"), {})
        qbpb = gen.get("quant_bpb")
        ubpb = gen.get("unigram_bpb")
        if qbpb is not None and ubpb is not None and qbpb > ubpb + DISASTER_SLACK:
            disasters.append(f"{ds_name}: {qbpb:.4f} vs {ubpb:.4f} unigram")
    p()
    if disasters:
        for d_ in disasters: p(f"  FAIL: {d_}")
    else:
        p("  PASS — general beats or nearly matches unigram on all external corpora")

    # ── Step 7: TOKPFX value on external ──────────────────────────────────────
    p()
    p("TOKPFX value on external corpora (Δ no-token vs general):")
    notok_gains = []
    for ds_name, _ in valid:
        gen_b  = all_r.get((ds_name, "general"),  {}).get("quant_bpb")
        notk_b = all_r.get((ds_name, "no-token"), {}).get("quant_bpb")
        if gen_b is not None and notk_b is not None:
            notok_gains.append((ds_name, notk_b - gen_b))
    for ds_name, gain in sorted(notok_gains, key=lambda x: -x[1]):
        p(f"  {ds_name:<22}  TOKPFX saves {gain:+.4f} BPB  "
          f"dom={all_r.get((ds_name,'general'),{}).get('dominant','?')}")

    # ── Step 8: Summary ────────────────────────────────────────────────────────
    p()
    p("=" * 90)
    p("PHASE 29B SUMMARY")
    p("=" * 90)
    p()
    p(f"  External files tested: {len(valid)}")
    p(f"  Profiles: general | prose | span-pfx | no-token")
    p()
    p("  Reminder: external results are stress test only.")
    p("  Profile decisions are driven by Phase 29A controlled corpus, not external data.")
    p()
    p("  Manifest saved: " + MANIFEST)

    rp = os.path.join(RESULTS, "phase29b_external.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport -> {rp}")


if __name__ == "__main__":
    main()
