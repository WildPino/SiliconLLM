"""
Phase 27C: Post-Token Loss Cartography
---------------------------------------
After TOKPFX promotion (Phase 27B), map WHERE the remaining loss lives:
  - Per byte-category breakdown with TOKPFX weight and oracle gap
  - Worst-span analysis: high-loss regions where TOKPFX is muted
  - Cartography verdict: what the next expert must cover

Tokenizer mirrors tok_lz.c exactly -- state replay from CSV target bytes.
Output: results/phase27c_cartography.txt
"""

import os
import csv
import re
import subprocess
import time

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEE     = os.path.join(ROOT, "see.exe")
WEIGHTS = os.path.join(ROOT, "weights", "entropy_weights.bin")
RESULTS = os.path.join(ROOT, "results")
TEL_DIR = os.path.join(RESULTS, "phase27c_tel")

# Phase 25 bz2-9 reference BPB
P25_BZ2 = {
    "natural_text": 1.7444,
    "markdown":     2.3997,
    "c_code":       1.1016,
    "multi_domain": 3.0050,
}

DATASETS = [
    ("natural_text", os.path.join(ROOT, "data", "natural_text.txt")),
    ("markdown",     os.path.join(ROOT, "data", "markdown_docs.md")),
    ("c_code",       os.path.join(ROOT, "data", "c_code.c")),
    ("multi_domain", os.path.join(ROOT, "data", "multi_domain.bin")),
]

# ---- Tokenizer: mirrors tok_lz.c classify_byte / is_token_boundary ----------

TOKTYPE_NONE  = 0
TOKTYPE_ALNUM = 1
TOKTYPE_MACRO = 2
TOKTYPE_SPACE = 3
TOKTYPE_NL    = 4
TOKTYPE_DELIM = 5

def _classify(b, prev):
    if b == 92:  # backslash
        return TOKTYPE_MACRO
    if (48 <= b <= 57) or (65 <= b <= 90) or (97 <= b <= 122):
        if prev == TOKTYPE_ALNUM or prev == TOKTYPE_MACRO:
            return prev
        return TOKTYPE_ALNUM
    if b == 10 or b == 13:
        return TOKTYPE_NL
    if b == 32 or b == 9:
        return TOKTYPE_SPACE
    return TOKTYPE_DELIM

def _is_boundary(new_type, prev):
    if new_type != prev:
        return True
    return prev in (TOKTYPE_SPACE, TOKTYPE_NL, TOKTYPE_DELIM)

# ---- Byte category (by post-advance token membership) -----------------------

_MD_BYTES      = set(b'#*>`|!_~')
_MATH_BYTES    = set(b'$^&')
_BRACKET_BYTES = set(b'[](){}')
_CODE_BYTES    = set(b';:=+/<,%-')

CATEGORIES = [
    "ALNUM_CONT",
    "ALNUM_START",
    "MACRO_CONT",
    "MACRO_START",
    "SPACE",
    "NL",
    "DELIM_MD",
    "DELIM_MATH",
    "DELIM_BRACKET",
    "DELIM_CODE",
    "DELIM_OTHER",
]

def byte_category(b, tok_type_before, new_type, boundary):
    if new_type == TOKTYPE_ALNUM:
        return "ALNUM_START" if boundary else "ALNUM_CONT"
    if new_type == TOKTYPE_MACRO:
        return "MACRO_START" if b == 92 else "MACRO_CONT"
    if new_type == TOKTYPE_SPACE:
        return "SPACE"
    if new_type == TOKTYPE_NL:
        return "NL"
    if b in _MD_BYTES:
        return "DELIM_MD"
    if b in _MATH_BYTES:
        return "DELIM_MATH"
    if b in _BRACKET_BYTES:
        return "DELIM_BRACKET"
    if b in _CODE_BYTES:
        return "DELIM_CODE"
    return "DELIM_OTHER"

# ---- Telemetry runner --------------------------------------------------------

def run_telemetry(ds_path, tel_path):
    t0 = time.perf_counter()
    r = subprocess.run(
        [SEE, "audit", ds_path,
         "--weights", WEIGHTS, "--blend", "moe",
         "--tok-prefix", "--telemetry", tel_path],
        capture_output=True, text=True, timeout=600,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return elapsed, r.stdout + r.stderr

def parse_quant_bpb(output):
    m = re.search(r"Quantized BPB:\s+([\d.]+)", output)
    return float(m.group(1)) if m else None

# ---- CSV loader -------------------------------------------------------------

def load_telemetry(tel_path):
    rows = []
    with open(tel_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "b":           int(row["target"]),
                "loss_see":    float(row["loss_see"]),
                "loss_uni":    float(row["loss_uni"]),
                "loss_bi":     float(row["loss_bi"]),
                "loss_lz":     float(row["loss_lz"]),
                "loss_tok":    float(row["loss_lz8"]),
                "loss_actual": float(row["loss_actual"]),
                "w_tok":       float(row["w_lz8"]),
            })
    return rows

# ---- Per-category accumulator -----------------------------------------------

def _empty_acc():
    return {"n": 0, "loss_actual": 0.0, "loss_tok": 0.0,
            "loss_lz": 0.0, "oracle": 0.0, "w_tok": 0.0}

def _finalize(acc):
    n = acc["n"]
    if n == 0:
        return None
    return {
        "count":      n,
        "actual_bpb": acc["loss_actual"] / n,
        "tok_bpb":    acc["loss_tok"]    / n,
        "lz_bpb":     acc["loss_lz"]     / n,
        "oracle_bpb": acc["oracle"]      / n,
        "moe_gap":    (acc["loss_actual"] - acc["oracle"]) / n,
        "w_tok_pct":  100.0 * acc["w_tok"] / n,
    }

# ---- Main analysis ----------------------------------------------------------

def analyze(rows):
    accs = {cat: _empty_acc() for cat in CATEGORIES}
    annotated = []

    tok_type = TOKTYPE_NONE
    tok_pos  = 0

    for row in rows:
        b        = row["b"]
        new_type = _classify(b, tok_type)
        boundary = _is_boundary(new_type, tok_type)
        cat      = byte_category(b, tok_type, new_type, boundary)

        oracle = min(row["loss_see"], row["loss_uni"], row["loss_bi"],
                     row["loss_lz"], row["loss_tok"])

        if cat in accs:
            a = accs[cat]
            a["n"]           += 1
            a["loss_actual"] += row["loss_actual"]
            a["loss_tok"]    += row["loss_tok"]
            a["loss_lz"]     += row["loss_lz"]
            a["oracle"]      += oracle
            a["w_tok"]       += row["w_tok"]

        annotated.append({"b": b, "cat": cat, "loss": row["loss_actual"],
                          "w_tok": row["w_tok"], "oracle": oracle})

        if boundary:
            tok_pos  = 1
            tok_type = new_type
        else:
            tok_pos += 1

    cat_stats = {cat: _finalize(accs[cat]) for cat in CATEGORIES}

    # Worst spans: windows of 32 bytes where TOKPFX is muted and loss is high
    WINDOW   = 32
    MIN_LOSS = 3.2
    MAX_WTOK = 0.10
    candidates = []
    n = len(annotated)
    for start in range(0, n - WINDOW):
        w  = annotated[start:start + WINDOW]
        ml = sum(r["loss"]  for r in w) / WINDOW
        mt = sum(r["w_tok"] for r in w) / WINDOW
        if ml >= MIN_LOSS and mt <= MAX_WTOK:
            text = bytes([r["b"] for r in w])
            cats = sorted(set(r["cat"] for r in w))
            candidates.append({"start": start, "mean_loss": ml,
                                "mean_w_tok": mt, "bytes": text, "cats": cats})

    candidates.sort(key=lambda x: -x["mean_loss"])

    # De-duplicate overlapping spans
    worst_spans = []
    used = set()
    for sp in candidates:
        if any(abs(sp["start"] - u) < 64 for u in used):
            continue
        used.add(sp["start"])
        worst_spans.append(sp)
        if len(worst_spans) >= 8:
            break

    return cat_stats, worst_spans

# ---- Formatting -------------------------------------------------------------

def _render(b):
    if 32 <= b < 127:
        return chr(b)
    return f"\\x{b:02x}"

def _text(data):
    return "".join(_render(b) for b in data)

# ---- Main -------------------------------------------------------------------

def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(TEL_DIR, exist_ok=True)

    lines = []
    def p(s=""):
        print(s)
        lines.append(s)

    p("=" * 80)
    p("PHASE 27C: Post-Token Loss Cartography (--tok-prefix mode)")
    p("=" * 80)
    p()

    all_summary = {}

    for ds_name, ds_path in DATASETS:
        if not os.path.exists(ds_path):
            p(f"[SKIP] {ds_name}: not found")
            continue

        raw_size  = os.path.getsize(ds_path)
        tel_path  = os.path.join(TEL_DIR, f"{ds_name}.csv")
        audit_out = ""

        if not os.path.exists(tel_path):
            p(f"Running audit --tok-prefix on {ds_name} ({raw_size//1024} KB)...")
            elapsed, audit_out = run_telemetry(ds_path, tel_path)
            p(f"  Completed in {elapsed:.0f} ms")
        else:
            p(f"Using cached telemetry: {ds_name}")

        if not os.path.exists(tel_path):
            p("  [ERROR] telemetry file missing -- audit may have failed")
            continue

        rows = load_telemetry(tel_path)
        if not rows:
            p("  [ERROR] empty telemetry CSV")
            continue

        p(f"  {len(rows)} rows loaded")

        # Derive global stats from CSV (avoids a second audit invocation)
        n = len(rows)
        bpb_model    = sum(r["loss_actual"] for r in rows) / n
        w_tok_global = sum(r["w_tok"]       for r in rows) / n
        quant_bpb    = parse_quant_bpb(audit_out) if audit_out else None
        summary      = {"bpb": bpb_model, "quant_bpb": quant_bpb,
                        "w_tok": w_tok_global}
        all_summary[ds_name] = summary

        cat_stats, worst_spans = analyze(rows)

        # ---- Per-dataset report ----------------------------------------------
        p()
        p("=" * 80)
        p(f"DATASET: {ds_name}  ({raw_size//1024} KB, {n} bytes evaluated)")
        bz2_ref = P25_BZ2.get(ds_name)
        bpb_display = quant_bpb if quant_bpb else bpb_model
        bpb_label   = "Quant" if quant_bpb else "Model"
        if bz2_ref:
            gap = bpb_display - bz2_ref
            p(f"  {bpb_label} BPB: {bpb_display:.4f}  (bz2-9 ref: {bz2_ref:.4f}  gap: {gap:+.4f})")
        else:
            p(f"  {bpb_label} BPB: {bpb_display:.4f}")
        p(f"  TOKPFX avg weight (global): {w_tok_global*100:.1f}%")
        p("=" * 80)
        p()

        # Category breakdown table
        hdr = (f"  {'Category':<16} {'N':>7}  {'Actual':>7}  "
               f"{'TOKPFX':>7}  {'LZ':>7}  {'Oracle':>7}  {'MoEgap':>7}  {'w_tok%':>7}")
        p(hdr)
        p("  " + "-" * 76)

        for cat in CATEGORIES:
            st = cat_stats.get(cat)
            if st is None:
                continue
            p(f"  {cat:<16} {st['count']:>7}  "
              f"{st['actual_bpb']:>7.3f}  "
              f"{st['tok_bpb']:>7.3f}  "
              f"{st['lz_bpb']:>7.3f}  "
              f"{st['oracle_bpb']:>7.3f}  "
              f"{st['moe_gap']:>+7.3f}  "
              f"{st['w_tok_pct']:>6.1f}%")

        p()

        alnum_cont   = cat_stats.get("ALNUM_CONT")
        alnum_start  = cat_stats.get("ALNUM_START")
        delim_md     = cat_stats.get("DELIM_MD")
        delim_math   = cat_stats.get("DELIM_MATH")
        nl           = cat_stats.get("NL")
        macro_cont   = cat_stats.get("MACRO_CONT")

        p("  Key findings:")
        if alnum_cont:
            p(f"    ALNUM_CONT:  TOKPFX {alnum_cont['w_tok_pct']:.0f}%  "
              f"actual={alnum_cont['actual_bpb']:.3f}  oracle={alnum_cont['oracle_bpb']:.3f}  "
              f"MoE gap={alnum_cont['moe_gap']:+.3f}")
        if alnum_start:
            p(f"    ALNUM_START: TOKPFX {alnum_start['w_tok_pct']:.0f}%  "
              f"actual={alnum_start['actual_bpb']:.3f}  oracle={alnum_start['oracle_bpb']:.3f}  "
              f"[word-boundary -- TOKPFX context = prev-token hash]")
        if macro_cont:
            p(f"    MACRO_CONT:  TOKPFX {macro_cont['w_tok_pct']:.0f}%  "
              f"actual={macro_cont['actual_bpb']:.3f}  oracle={macro_cont['oracle_bpb']:.3f}")
        if delim_md:
            p(f"    DELIM_MD:    TOKPFX {delim_md['w_tok_pct']:.0f}%  "
              f"actual={delim_md['actual_bpb']:.3f}  oracle={delim_md['oracle_bpb']:.3f}  "
              f"MoE gap={delim_md['moe_gap']:+.3f}  <- MD structure blindspot")
        if delim_math:
            p(f"    DELIM_MATH:  TOKPFX {delim_math['w_tok_pct']:.0f}%  "
              f"actual={delim_math['actual_bpb']:.3f}  oracle={delim_math['oracle_bpb']:.3f}")
        if nl:
            p(f"    NL:          TOKPFX {nl['w_tok_pct']:.0f}%  "
              f"actual={nl['actual_bpb']:.3f}  oracle={nl['oracle_bpb']:.3f}")

        p()

        if worst_spans:
            p("  Worst spans (TOKPFX muted, loss >= 3.2 BPB):")
            for j, sp in enumerate(worst_spans[:5]):
                text = _text(sp["bytes"])
                p(f"  [{j+1}] @+{sp['start']:>6}  loss={sp['mean_loss']:.3f}  w_tok={sp['mean_w_tok']:.3f}")
                p(f"        {repr(text)}")
                p(f"        cats: {', '.join(sp['cats'])}")
        else:
            p("  No high-loss / TOKPFX-muted spans found -- remaining loss is spread.")

        p()

    # ---- Cross-dataset verdict -----------------------------------------------
    p()
    p("=" * 80)
    p("CARTOGRAPHY VERDICT")
    p("=" * 80)
    p()
    p("  TOKPFX signal map:")
    p("    ALNUM_CONT   -> TOKPFX wins: inside-word prefix hash, strong predictive power")
    p("    ALNUM_START  -> TOKPFX neutral: uses prev-token context -- TOK_PREV covers this")
    p("    MACRO_CONT   -> TOKPFX wins on LaTeX macro continuations (\\mathbf, \\begin ...)")
    p("    MACRO_START  -> single-byte token, no within-token history")
    p("    SPACE / NL   -> single-byte: TOKPFX context = prev-token hash (LZ dominates)")
    p("    DELIM_*      -> single-byte: structural patterns; no expert wins here")
    p()
    p("  Diagnosis rules:")
    p("    ALNUM_START: high oracle_bpb + low w_tok")
    p("      -> word-transition gap: TOK_PREV (last-token hash -> first-byte-of-next) is the fix")
    p()
    p("    DELIM_MD: high oracle_bpb (no expert helps) + high count on markdown")
    p("      -> MD structure expert needed: line-start type context (BOL), not token-based")
    p()
    p("    moe_gap > 0.2 for any category:")
    p("      -> MoE mixing overhead, not expert gap (oracle << actual)")
    p("      -> Consider faster convergence or domain-specific eta")
    p()
    p("    DELIM_MATH oracle_bpb >= 6.0:")
    p("      -> LaTeX math is irrecoverable for streaming experts -- accept as floor")
    p()

    report_path = os.path.join(RESULTS, "phase27c_cartography.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport: {report_path}")

if __name__ == "__main__":
    main()
