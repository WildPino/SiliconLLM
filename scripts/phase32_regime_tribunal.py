"""
Phase 32: Regime Segmentation Tribunal

Research question:
    Do local regimes exist in the byte stream such that switching routing
    BEFORE prediction reduces entropy?

Method (offline, no codec modification):
  1. Run SEE audit --telemetry on a selection of corpora.
  2. Parse per-byte features: expert weights, per-expert losses, char class.
  3. Derive oracle_gap = moe_loss - min(expert_losses) per byte.
     This is the routing opportunity: how many bits would a perfect oracle router save?
  4. Build 32-byte windows with per-window feature averages.
  5. Cluster windows with k-means (k=4..8), pick best silhouette.
  6. Per-cluster analysis:
     - mean oracle_gap (routing opportunity)
     - dominant expert (lowest mean loss)
     - char-class distribution
     - corpus origin distribution
  7. Verdict: if oracle_gap varies significantly between clusters AND clusters
     have identifiable character → routing opportunity exists.

Corpora (from Phase 29A/29C catalog):
  Markdown (gap target): markdown_docs.md, repo_markdown_mixed.md
  Contrast: natural_text.txt, c_code.c, log_synth.log, shuffled.bin

Telemetry CSV columns (general profile):
  i, target, loss_see, loss_uni, loss_bi, loss_lz, loss_lz8(=TOKPFX),
  loss_tokprev(=0), loss_actual, w_see, w_uni, w_bi, w_lz, w_lz8(=TOKPFX),
  w_tokprev(=0), loss_span, w_span

Output:
  results/phase32_regime_tribunal.txt   — full report
  results/phase32_regime_data.json      — cluster stats for downstream use
"""

import os
import sys
import io
import json
import math
import time
import subprocess
import numpy as np
from collections import Counter

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEE     = os.path.join(ROOT, "see.exe")
WEIGHTS = os.path.join(ROOT, "weights", "entropy_weights_factors_r16.bin")
DATA    = os.path.join(ROOT, "data")
EXT_DIR = os.path.join(DATA, "external")
RESULTS = os.path.join(ROOT, "results")
TEL_DIR = os.path.join(ROOT, "data", "phase32_tel")

CORPORA = [
    ("markdown_docs",  os.path.join(DATA, "markdown_docs.md"),          "markdown"),
    ("md_mixed",       os.path.join(DATA, "repo_markdown_mixed.md"),    "markdown"),
    ("natural_text",   os.path.join(DATA, "natural_text.txt"),          "prose"),
    ("c_code",         os.path.join(DATA, "c_code.c"),                  "code"),
    ("log_synth",      os.path.join(DATA, "log_synth.log"),             "log"),
    ("shuffled",       os.path.join(DATA, "shuffled.bin"),              "random"),
]

WINDOW   = 32    # bytes per window
STRIDE   = 8     # stride between windows
K_VALUES = [4, 5, 6, 7, 8]

# ---------------------------------------------------------------------------
# Char class
# ---------------------------------------------------------------------------

def char_class(b: int) -> int:
    """0=alphanum, 1=space/tab, 2=newline, 3=punct/ascii, 4=other"""
    if (48 <= b <= 57) or (65 <= b <= 90) or (97 <= b <= 122):
        return 0
    if b in (32, 9):
        return 1
    if b in (10, 13):
        return 2
    if 33 <= b <= 126:
        return 3
    return 4


# ---------------------------------------------------------------------------
# Telemetry collection
# ---------------------------------------------------------------------------

def collect_telemetry(corpora, force=False):
    os.makedirs(TEL_DIR, exist_ok=True)
    paths = {}
    for name, path, domain in corpora:
        tel_path = os.path.join(TEL_DIR, f"{name}.csv")
        paths[name] = tel_path
        if os.path.exists(tel_path) and not force:
            print(f"  [cached] {name}", flush=True)
            continue
        if not os.path.exists(path):
            print(f"  [SKIP] {name}: file not found", flush=True)
            paths[name] = None
            continue
        print(f"  [audit] {name} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        r = subprocess.run(
            [SEE, "audit", path, "--weights", WEIGHTS,
             "--expert-profile", "general", "--blend", "moe",
             "--telemetry", tel_path],
            capture_output=True, text=True, timeout=900,
        )
        elapsed = time.perf_counter() - t0
        if r.returncode != 0 or not os.path.exists(tel_path):
            print(f"FAILED: {r.stderr[:200]}", flush=True)
            paths[name] = None
        else:
            print(f"done ({elapsed:.1f}s)", flush=True)
    return paths


# ---------------------------------------------------------------------------
# Parse telemetry CSV
# ---------------------------------------------------------------------------

def load_telemetry(csv_path, name, domain):
    """Return list of per-byte dicts."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        # Expected: i,target,loss_see,loss_uni,loss_bi,loss_lz,loss_lz8,
        #           loss_tokprev,loss_actual,w_see,w_uni,w_bi,w_lz,w_lz8,
        #           w_tokprev,loss_span,w_span
        idx = {col: i for i, col in enumerate(header)}

        def g(row, col):
            return float(row[idx[col]]) if col in idx else 0.0

        for line in f:
            row = line.strip().split(",")
            if len(row) < 9:
                continue
            b = int(row[idx["target"]])

            # Expert losses (cap at 8.0 = unigram ceiling)
            l_see   = min(g(row, "loss_see"),  8.0)
            l_uni   = min(g(row, "loss_uni"),  8.0)
            l_bi    = min(g(row, "loss_bi"),   8.0)
            l_lz    = min(g(row, "loss_lz"),   8.0)
            l_tok   = min(g(row, "loss_lz8"),  8.0)  # TOKPFX in general profile
            l_moe   = min(g(row, "loss_actual"), 8.0)

            oracle  = min(l_see, l_uni, l_bi, l_lz, l_tok)
            gap     = l_moe - oracle           # routing opportunity (bits)
            assert gap >= -0.001, f"oracle_gap negative: {gap}"

            # Expert weights
            w_see   = g(row, "w_see")
            w_uni   = g(row, "w_uni")
            w_bi    = g(row, "w_bi")
            w_lz    = g(row, "w_lz")
            w_tok   = g(row, "w_lz8")          # TOKPFX weight

            # Derived
            dom_exp_loss = min(l_see, l_uni, l_bi, l_lz, l_tok)
            dom_idx  = [l_see, l_uni, l_bi, l_lz, l_tok].index(dom_exp_loss)
            lz_adv   = l_uni - l_lz            # positive = LZ beats unigram
            cc       = char_class(b)

            rows.append({
                "i": int(row[idx["i"]]),
                "corpus": name,
                "domain": domain,
                "byte": b,
                "cc": cc,
                # losses
                "l_see": l_see, "l_uni": l_uni, "l_bi": l_bi,
                "l_lz": l_lz,  "l_tok": l_tok, "l_moe": l_moe,
                "oracle": oracle,
                "gap": max(gap, 0.0),
                # weights
                "w_see": w_see, "w_uni": w_uni, "w_bi": w_bi,
                "w_lz": w_lz,  "w_tok": w_tok,
                # derived
                "dom_exp": dom_idx,        # 0=SEE 1=UNI 2=BI 3=LZ 4=TOK
                "lz_adv": lz_adv,
            })
    return rows


# ---------------------------------------------------------------------------
# Window features
# ---------------------------------------------------------------------------

EXP_NAMES = ["SEE", "UNI", "BI", "LZ", "TOKPFX"]

def make_windows(rows, window=WINDOW, stride=STRIDE):
    """Return array of per-window feature vectors + metadata."""
    n = len(rows)
    windows = []
    for start in range(0, n - window + 1, stride):
        chunk = rows[start : start + window]
        meta = {
            "corpus": chunk[0]["corpus"],
            "domain": chunk[0]["domain"],
            "start_i": chunk[0]["i"],
            "text_sample": bytes(r["byte"] for r in chunk),
        }

        # Feature vector (all normalized to similar scale)
        feats = [
            # Expert weight profile (who is trusted)
            np.mean([r["w_see"] for r in chunk]),
            np.mean([r["w_uni"] for r in chunk]),
            np.mean([r["w_bi"]  for r in chunk]),
            np.mean([r["w_lz"]  for r in chunk]),
            np.mean([r["w_tok"] for r in chunk]),
            # Oracle gap (routing opportunity)
            np.mean([r["gap"]   for r in chunk]),
            # MoE loss level
            np.mean([r["l_moe"] for r in chunk]),
            # LZ advantage (positive = LZ has matches)
            np.mean([r["lz_adv"] for r in chunk]),
            # Char class fractions
            sum(1 for r in chunk if r["cc"] == 0) / window,  # alphanum
            sum(1 for r in chunk if r["cc"] == 1) / window,  # space/tab
            sum(1 for r in chunk if r["cc"] == 2) / window,  # newline
            sum(1 for r in chunk if r["cc"] == 3) / window,  # punct/ascii
            sum(1 for r in chunk if r["cc"] == 4) / window,  # other
            # Loss std (regime stability)
            np.std([r["l_moe"] for r in chunk]),
            # Dominant expert count
            sum(1 for r in chunk if r["dom_exp"] == 3) / window,  # LZ dominant frac
        ]

        meta["feats"]     = feats
        meta["oracle_gap"] = np.mean([r["gap"] for r in chunk])
        meta["l_moe"]      = np.mean([r["l_moe"] for r in chunk])
        meta["l_lz"]       = np.mean([r["l_lz"] for r in chunk])
        meta["l_bi"]       = np.mean([r["l_bi"] for r in chunk])
        meta["l_tok"]      = np.mean([r["l_tok"] for r in chunk])
        meta["l_uni"]      = np.mean([r["l_uni"] for r in chunk])
        meta["l_see"]      = np.mean([r["l_see"] for r in chunk])
        meta["dom_exp"]    = Counter(r["dom_exp"] for r in chunk).most_common(1)[0][0]
        meta["cc_dist"]    = Counter(r["cc"] for r in chunk)
        windows.append(meta)
    return windows


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def choose_k(X_scaled, k_values):
    best_k, best_sil, best_labels = k_values[0], -1, None
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) < k:
            continue
        sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
        print(f"    k={k}  silhouette={sil:.4f}", flush=True)
        if sil > best_sil:
            best_sil, best_k, best_labels = sil, k, labels
    return best_k, best_sil, best_labels


# ---------------------------------------------------------------------------
# Cluster analysis
# ---------------------------------------------------------------------------

def analyze_clusters(windows, labels, k):
    clusters = {i: [] for i in range(k)}
    for w, lbl in zip(windows, labels):
        clusters[lbl].append(w)

    stats = {}
    for cid, members in clusters.items():
        if not members:
            continue
        n = len(members)

        oracle_gaps  = [m["oracle_gap"] for m in members]
        l_moe_vals   = [m["l_moe"]      for m in members]
        dom_exps     = [m["dom_exp"]     for m in members]
        corpus_dist  = Counter(m["corpus"] for m in members)
        domain_dist  = Counter(m["domain"] for m in members)
        cc_total     = Counter()
        for m in members:
            cc_total += m["cc_dist"]

        # Mean expert losses
        mean_losses = {
            "SEE":    np.mean([m["l_see"] for m in members]),
            "UNI":    np.mean([m["l_uni"] for m in members]),
            "BI":     np.mean([m["l_bi"]  for m in members]),
            "LZ":     np.mean([m["l_lz"]  for m in members]),
            "TOKPFX": np.mean([m["l_tok"] for m in members]),
        }
        oracle_expert = min(mean_losses, key=mean_losses.get)

        # Sample windows (readable bytes only)
        samples = []
        for m in members[:5]:
            try:
                s = m["text_sample"].decode("utf-8", errors="replace")
                s = s.replace("\n", "↵").replace("\r", "").replace("\t", "→")
                samples.append(s[:40])
            except Exception:
                pass

        stats[cid] = {
            "n": n,
            "mean_oracle_gap": float(np.mean(oracle_gaps)),
            "std_oracle_gap":  float(np.std(oracle_gaps)),
            "mean_moe_loss":   float(np.mean(l_moe_vals)),
            "mean_losses":     {k: round(v, 4) for k, v in mean_losses.items()},
            "oracle_expert":   oracle_expert,
            "dom_exp_dist":    {EXP_NAMES[e]: c for e, c in Counter(dom_exps).most_common()},
            "corpus_dist":     dict(corpus_dist.most_common()),
            "domain_dist":     dict(domain_dist.most_common()),
            "cc_dist":         {str(k): v for k, v in cc_total.most_common()},
            "samples":         samples,
        }
    return stats


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

CC_LABEL = {0: "alphanum", 1: "space/tab", 2: "newline", 3: "punct", 4: "other"}

def print_report(stats, best_k, best_sil, windows, labels):
    lines = []
    def p(s=""):
        lines.append(s)
        print(s, flush=True)

    p("=" * 80)
    p("PHASE 32: REGIME SEGMENTATION TRIBUNAL")
    p("=" * 80)
    p(f"Windows: {len(windows)}  (size={WINDOW}B, stride={STRIDE}B)")
    p(f"Best k: {best_k}  silhouette: {best_sil:.4f}")
    p()

    total_oracle_gap = np.mean([m["oracle_gap"] for m in windows])
    p(f"Global mean oracle_gap: {total_oracle_gap:.4f} bits/byte")
    p(f"  (bits saved per byte if routing were perfect across all windows)")
    p()

    for cid in sorted(stats):
        s = stats[cid]
        p(f"─── Cluster {cid}  (n={s['n']}, {100*s['n']/len(windows):.1f}%) ─────────────────────────")
        p(f"  oracle_gap:  {s['mean_oracle_gap']:.4f} ± {s['std_oracle_gap']:.4f} bits/byte")
        p(f"  moe_loss:    {s['mean_moe_loss']:.4f} bits/byte")
        p(f"  oracle_exp:  {s['oracle_expert']}")
        losses_str = "  ".join(f"{e}={v:.3f}" for e, v in sorted(s['mean_losses'].items(), key=lambda x: x[1]))
        p(f"  exp_losses:  {losses_str}")
        dom = s["dom_exp_dist"]
        dom_str = "  ".join(f"{e}={c}" for e, c in list(dom.items())[:4])
        p(f"  dom_exp:     {dom_str}")
        # char class (top 3)
        cc_items = sorted(s["cc_dist"].items(), key=lambda x: -int(x[1]))[:3]
        cc_str = "  ".join(f"{CC_LABEL.get(int(k), k)}={v}" for k, v in cc_items)
        p(f"  char_class:  {cc_str}")
        # domain distribution
        dom_dist = "  ".join(f"{d}={c}" for d, c in list(s["domain_dist"].items())[:4])
        p(f"  domain_dist: {dom_dist}")
        if s["samples"]:
            p(f"  samples:")
            for smp in s["samples"]:
                p(f"    [{smp}]")
        p()

    # Verdict
    p("=" * 80)
    p("VERDICT")
    p("=" * 80)

    gaps = [s["mean_oracle_gap"] for s in stats.values()]
    gap_range = max(gaps) - min(gaps)
    gap_ratio = max(gaps) / max(min(gaps), 0.001)

    oracle_experts = Counter(s["oracle_expert"] for s in stats.values())
    n_distinct_experts = len(oracle_experts)

    p(f"Oracle-gap range across clusters: {gap_range:.4f} bits/byte")
    p(f"Oracle-gap ratio (max/min):        {gap_ratio:.2f}x")
    p(f"Distinct oracle experts:           {n_distinct_experts} / {best_k}")
    p(f"Oracle experts by cluster:         {dict(oracle_experts)}")
    p()

    # Decision
    ROUTING_THRESHOLD_RANGE = 0.10   # gap range must be > 0.1 BPB to matter
    ROUTING_THRESHOLD_RATIO  = 1.5   # max/min must be > 1.5x
    EXPERT_THRESHOLD         = 2     # at least 2 different oracle experts

    routing_opportunity = (
        gap_range >= ROUTING_THRESHOLD_RANGE and
        gap_ratio  >= ROUTING_THRESHOLD_RATIO and
        n_distinct_experts >= EXPERT_THRESHOLD
    )

    if routing_opportunity:
        p("VERDICT: ROUTING OPPORTUNITY EXISTS")
        p(f"  Gap varies by {gap_range:.3f} BPB across clusters — regime routing is worth pursuing.")
        p(f"  {n_distinct_experts} different experts are optimal in different clusters.")
        p("  → Phase 33 hypothesis: a lightweight regime detector before the MoE router")
        p("    can reduce oracle_gap from the global mean toward near-zero.")
    else:
        p("VERDICT: ROUTING OPPORTUNITY NOT CONFIRMED")
        if gap_range < ROUTING_THRESHOLD_RANGE:
            p(f"  Oracle-gap is uniformly low across clusters (range={gap_range:.4f} < {ROUTING_THRESHOLD_RANGE}).")
            p("  The current MoE already tracks near-oracle. A regime router would not help.")
        elif n_distinct_experts < EXPERT_THRESHOLD:
            p(f"  Only {n_distinct_experts} oracle expert(s) across all clusters.")
            p("  Clusters differ in content but the same expert wins everywhere — no routing value.")
        else:
            p(f"  Weak signal: gap range {gap_range:.4f}, ratio {gap_ratio:.2f}x.")
            p("  Not enough evidence to justify a regime router in Phase 33.")

    p()
    p("Note: oracle_gap measured over the CURRENT expert set (SEE, UNI, BI, LZ, TOKPFX).")
    p("A different expert set might raise or lower the ceiling.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-run audit even if telemetry cached")
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    print("Phase 32: Regime Segmentation Tribunal", flush=True)
    print(f"Window={WINDOW}B  Stride={STRIDE}B  K={K_VALUES}", flush=True)
    print()

    # 1. Collect telemetry
    print("Step 1: Collect telemetry", flush=True)
    tel_paths = collect_telemetry(CORPORA, force=args.force)

    # 2. Load and build windows
    print("\nStep 2: Load telemetry and build windows", flush=True)
    all_rows = []
    for name, path, domain in CORPORA:
        tel = tel_paths.get(name)
        if not tel or not os.path.exists(tel):
            continue
        rows = load_telemetry(tel, name, domain)
        print(f"  {name:<20} {len(rows):>7} bytes → ", end="", flush=True)
        wins = make_windows(rows)
        print(f"{len(wins)} windows", flush=True)
        all_rows.extend(wins)

    if not all_rows:
        print("No data. Exiting.", flush=True)
        return

    print(f"\nTotal windows: {len(all_rows)}", flush=True)

    # 3. Build feature matrix
    X = np.array([w["feats"] for w in all_rows])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Cluster
    print("\nStep 3: K-means clustering", flush=True)
    best_k, best_sil, labels = choose_k(X_scaled, K_VALUES)
    print(f"  → Selected k={best_k}  silhouette={best_sil:.4f}", flush=True)

    # 5. Analyze
    print("\nStep 4: Cluster analysis", flush=True)
    stats = analyze_clusters(all_rows, labels, best_k)

    # 6. Report
    print("\n", flush=True)
    report = print_report(stats, best_k, best_sil, all_rows, labels)

    report_path = os.path.join(RESULTS, "phase32_regime_tribunal.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport: {report_path}", flush=True)

    json_path = os.path.join(RESULTS, "phase32_regime_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "k": best_k, "silhouette": best_sil,
            "n_windows": len(all_rows),
            "global_oracle_gap": float(np.mean([m["oracle_gap"] for m in all_rows])),
            "clusters": {str(k): v for k, v in stats.items()},
        }, f, indent=2, default=str)
    print(f"Data:   {json_path}", flush=True)


if __name__ == "__main__":
    main()
