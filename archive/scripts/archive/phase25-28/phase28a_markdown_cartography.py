"""
Phase 28A: Markdown Line Cartography

Does NOT implement any new expert. Asks three questions first:

  Q-VOL.  How many bytes are at each line-position bucket?
          (col 0, col 1-4, col 5-16, col 17+)
          If BOL is <3% of bytes, even a perfect oracle won't give -0.05 BPB.

  Q-GAP.  Where does the MoE mixture actually lose?
          actual_bpb, oracle_bpb, moe_gap, blind_bpb per bucket.
          Buckets: line_pos x {after_newline, after_blank_line, interior}.

  Q-SPAN. What are the worst loss spans in the markdown file?
          Shows the actual bytes, so we can read what structure they represent.

  Q-ORA.  Oracle estimate for a simple BOL predictor:
          If we predicted perfectly every col-0 byte, how many BPB do we save?

Outputs a plain-text report and a CSV for further analysis.
"""

import os, csv, math, collections, subprocess, time

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEE     = os.path.join(ROOT, "see.exe")
WEIGHTS = os.path.join(ROOT, "weights", "entropy_weights_factors_r16.bin")
RESULTS = os.path.join(ROOT, "results")
DATA    = os.path.join(ROOT, "data")
MD_PATH = os.path.join(DATA, "markdown_docs.md")
TEL_CSV = os.path.join(ROOT, "tmp_28a_tel.csv")


# ── Line-position state machine ───────────────────────────────────────────────

def col_bucket(col):
    if col == 0:   return "col_0"
    if col <= 4:   return "col_1_4"
    if col <= 16:  return "col_5_16"
    return "col_17plus"

def line_context(col, prev_line_blank, after_newline):
    ctx = col_bucket(col)
    if after_newline:
        ctx += "|after_nl"
    if prev_line_blank:
        ctx += "|after_blank"
    return ctx


# ── Telemetry CSV columns ─────────────────────────────────────────────────────
# byte_i, target, l_SEE, l_UNI, l_BI, l_LZ, l_TOKPFX, l_TOKPREV,
# sample_loss, w_SEE, w_UNI, w_BI, w_LZ, w_TOKPFX, w_TOKPREV

COL_TARGET     = 1
COL_L_SEE      = 2
COL_L_UNI      = 3
COL_L_BI       = 4
COL_L_LZ       = 5
COL_L_TOKPFX   = 6
COL_L_TOKPREV  = 7
COL_SAMPLE     = 8

N_EXPERTS      = 5   # SEE UNI BI LZ TOKPFX (general profile — no TOKPREV)
EXPERT_COLS    = [COL_L_SEE, COL_L_UNI, COL_L_BI, COL_L_LZ, COL_L_TOKPFX]
EXPERT_NAMES   = ["SEE", "UNI", "BI", "LZ", "TOKPFX"]


# ── Bucket accumulator ────────────────────────────────────────────────────────

class Bucket:
    def __init__(self):
        self.n            = 0
        self.sum_actual   = 0.0
        self.sum_oracle   = 0.0
        self.sum_blind    = 0.0   # -log2(1/256) = 8.0 per byte
        self.expert_wins  = [0] * N_EXPERTS

    def add(self, actual, oracle, expert_idx):
        self.n           += 1
        self.sum_actual  += actual
        self.sum_oracle  += oracle
        self.sum_blind   += 8.0
        self.expert_wins[expert_idx] += 1

    def bpb(self, s):
        return s / self.n if self.n else 0.0

    def report(self, name, total_bytes):
        if self.n == 0:
            return f"  {name:<30}  n=0"
        actual = self.bpb(self.sum_actual)
        oracle = self.bpb(self.sum_oracle)
        blind  = 8.0
        gap    = actual - oracle
        pct    = 100.0 * self.n / total_bytes
        wins   = "  ".join(f"{EXPERT_NAMES[e]}={self.expert_wins[e]}" for e in range(N_EXPERTS))
        return (f"  {name:<30}  n={self.n:>7} ({pct:5.1f}%)  "
                f"actual={actual:.4f}  oracle={oracle:.4f}  gap={gap:+.4f}  [{wins}]")


def run_telemetry():
    print("Running telemetry audit on markdown_docs.md (--expert-profile general)...")
    t0 = time.perf_counter()
    r = subprocess.run(
        [SEE, "audit", MD_PATH, "--weights", WEIGHTS, "--blend", "moe",
         "--expert-profile", "general", "--telemetry", TEL_CSV],
        capture_output=True, text=True, timeout=300
    )
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")
    return r.stdout + r.stderr


def main():
    os.makedirs(RESULTS, exist_ok=True)

    if not os.path.exists(MD_PATH):
        print(f"ERROR: markdown dataset not found at {MD_PATH}")
        return

    # ── Step 1: run telemetry ─────────────────────────────────────────────────
    audit_out = run_telemetry()

    # ── Step 2: load raw markdown bytes ──────────────────────────────────────
    with open(MD_PATH, "rb") as f:
        raw = f.read()

    # ── Step 3: parse telemetry CSV ──────────────────────────────────────────
    # The telemetry eval window starts after byte index 2 (the 2 seed bytes).
    # CSV row i corresponds to raw[i + 2].
    # We need to reconstruct line state from raw[] from byte 0.

    print("Parsing telemetry CSV...")
    rows = []
    with open(TEL_CSV, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 9:
                try:
                    rows.append([float(x) for x in row])
                except ValueError:
                    pass  # skip header or malformed lines
    print(f"  {len(rows)} rows loaded")

    # ── Step 4: reconstruct line state for each telemetry byte ────────────────
    # Walk raw[] from byte 0 to build col / prev_line_blank / after_newline
    # for every position. Then slice to the eval window [2 : 2+len(rows)].

    col              = [0] * len(raw)
    after_newline    = [False] * len(raw)
    prev_line_blank  = [False] * len(raw)

    cur_col          = 0
    cur_after_nl     = False
    cur_prev_blank   = False
    cur_line_empty   = True   # current line has had no non-NL chars yet

    for idx, b in enumerate(raw):
        col[idx]             = cur_col
        after_newline[idx]   = cur_after_nl
        prev_line_blank[idx] = cur_prev_blank

        if b == ord('\n'):
            cur_prev_blank  = cur_line_empty
            cur_col         = 0
            cur_after_nl    = True
            cur_line_empty  = True
        else:
            cur_col         = cur_col + 1
            cur_after_nl    = False
            cur_line_empty  = False

    # ── Step 5: aggregate into buckets ───────────────────────────────────────

    buckets      = collections.defaultdict(Bucket)
    worst_spans  = []   # (loss, byte_idx, context_key)

    # BOL oracle accumulator (col == 0 bytes only)
    bol_n         = 0
    bol_oracle    = 0.0
    bol_actual    = 0.0

    total_actual  = 0.0
    total_n       = 0
    total_oracle  = 0.0

    EVAL_OFFSET = 2  # first 2 bytes are seed, not in telemetry

    for row_i, row in enumerate(rows):
        raw_idx = row_i + EVAL_OFFSET
        if raw_idx >= len(raw):
            break

        actual  = row[COL_SAMPLE]
        losses  = [row[c] for c in EXPERT_COLS]
        oracle  = min(losses)
        best_e  = losses.index(oracle)

        c       = col[raw_idx]
        an      = after_newline[raw_idx]
        pb      = prev_line_blank[raw_idx]

        ctx_key = line_context(c, pb, an)
        buckets[ctx_key].add(actual, oracle, best_e)

        # Flat col-bucket (ignoring after_nl / after_blank distinction)
        buckets["_col:" + col_bucket(c)].add(actual, oracle, best_e)

        # Flat after_nl bucket
        if an:
            buckets["_after_nl:yes"].add(actual, oracle, best_e)
        else:
            buckets["_after_nl:no"].add(actual, oracle, best_e)

        # Flat after_blank bucket
        if pb:
            buckets["_after_blank:yes"].add(actual, oracle, best_e)
        else:
            buckets["_after_blank:no"].add(actual, oracle, best_e)

        total_actual += actual
        total_oracle += oracle
        total_n      += 1

        # BOL tracking
        if c == 0:
            bol_n      += 1
            bol_actual += actual
            bol_oracle += oracle

        # Worst spans (top-100 individual bytes by loss)
        worst_spans.append((actual, raw_idx, ctx_key, chr(raw[raw_idx]) if 32 <= raw[raw_idx] < 127 else f"\\x{raw[raw_idx]:02x}"))

    worst_spans.sort(key=lambda x: -x[0])

    # ── Step 6: build report ─────────────────────────────────────────────────

    lines = []
    def p(s=""):
        print(s)
        lines.append(s)

    p("=" * 88)
    p("PHASE 28A: Markdown Line Cartography")
    p("=" * 88)

    global_actual = total_actual / total_n if total_n else 0
    global_oracle = total_oracle / total_n if total_n else 0
    p()
    p(f"Dataset:        {MD_PATH}")
    p(f"Bytes in eval:  {total_n}")
    p(f"Global actual:  {global_actual:.4f} BPB")
    p(f"Global oracle:  {global_oracle:.4f} BPB")
    p(f"Global gap:     {global_actual - global_oracle:+.4f} BPB")

    # ── Q-VOL / Q-GAP: col-bucket breakdown ──────────────────────────────────
    p()
    p("─" * 88)
    p("Q-VOL / Q-GAP: Breakdown by line-position bucket")
    p("─" * 88)
    for key in ["col_0", "col_1_4", "col_5_16", "col_17plus"]:
        bk = buckets.get("_col:" + key)
        if bk:
            p(bk.report("col:" + key, total_n))

    # ── After-newline breakdown ───────────────────────────────────────────────
    p()
    p("─" * 88)
    p("Breakdown by after_newline state")
    p("─" * 88)
    for key in ["_after_nl:yes", "_after_nl:no"]:
        bk = buckets.get(key)
        if bk:
            p(bk.report(key[1:], total_n))

    # ── After-blank-line breakdown ────────────────────────────────────────────
    p()
    p("─" * 88)
    p("Breakdown by after_blank_line state")
    p("─" * 88)
    for key in ["_after_blank:yes", "_after_blank:no"]:
        bk = buckets.get(key)
        if bk:
            p(bk.report(key[1:], total_n))

    # ── Combined col × context breakdown ─────────────────────────────────────
    p()
    p("─" * 88)
    p("Combined: col_bucket × {after_nl, after_blank, interior}")
    p("─" * 88)
    col_keys = ["col_0", "col_1_4", "col_5_16", "col_17plus"]
    ctx_sfx  = ["|after_nl", "|after_blank", "|after_nl|after_blank", ""]
    # Collect all keys that exist, sorted
    all_keys = sorted([k for k in buckets if not k.startswith("_")])
    for k in all_keys:
        p(buckets[k].report(k, total_n))

    # ── Q-ORA: BOL oracle estimate ────────────────────────────────────────────
    p()
    p("─" * 88)
    p("Q-ORA: BOL Oracle Estimate")
    p("─" * 88)
    if bol_n > 0:
        bol_actual_bpb  = bol_actual / bol_n
        bol_oracle_bpb  = bol_oracle / bol_n
        bol_gap_bpb     = bol_actual_bpb - bol_oracle_bpb
        bol_pct         = 100.0 * bol_n / total_n
        # If BOL oracle matched actual for non-BOL, how much global BPB do we save?
        # Saving = bol_gap * (bol_n / total_n)
        # This is the maximum a perfect BOL expert could give if it closed the full gap.
        max_global_saving = bol_gap_bpb * (bol_n / total_n)
        p(f"  BOL bytes (col=0):    {bol_n} / {total_n}  ({bol_pct:.1f}%)")
        p(f"  BOL actual BPB:       {bol_actual_bpb:.4f}")
        p(f"  BOL oracle BPB:       {bol_oracle_bpb:.4f}")
        p(f"  BOL gap:              {bol_gap_bpb:+.4f} BPB")
        p(f"  Max global saving:    {max_global_saving:+.4f} BPB")
        p(f"  (= BOL gap × BOL fraction = {bol_gap_bpb:.4f} × {bol_pct/100:.3f})")
        if max_global_saving >= 0.05:
            p(f"  => BOL headroom SUFFICIENT for -0.05 BPB target (headroom = {max_global_saving:.4f})")
        elif max_global_saving >= 0.02:
            p(f"  => BOL headroom MARGINAL ({max_global_saving:.4f} < 0.05): partial gain possible")
        else:
            p(f"  => BOL headroom INSUFFICIENT ({max_global_saving:.4f}): BOL expert unlikely to hit target")

    # ── Q-SPAN: Worst 30 loss bytes with context ──────────────────────────────
    p()
    p("─" * 88)
    p("Q-SPAN: Worst 30 individual bytes (by actual loss)")
    p("─" * 88)
    p(f"  {'rank':>4}  {'loss':>6}  {'col':>5}  {'ctx':<35}  char  snippet")
    p("  " + "-" * 75)
    for rank, (loss, raw_idx, ctx, ch) in enumerate(worst_spans[:30], 1):
        c_val = col[raw_idx]
        an_val = after_newline[raw_idx]
        # Show 10 bytes of context around this position
        lo = max(0, raw_idx - 5)
        hi = min(len(raw), raw_idx + 6)
        snippet = raw[lo:hi].decode("utf-8", errors="replace").replace("\n", "\\n").replace("\t", "\\t")
        p(f"  {rank:>4}  {loss:>6.2f}  {c_val:>5}  {ctx:<35}  {ch!r:>4}  |{snippet}|")

    # ── Worst 10 consecutive 50-byte spans ───────────────────────────────────
    p()
    p("─" * 88)
    p("Worst 10 spans of 50 consecutive bytes (by average loss)")
    p("─" * 88)
    SPAN = 50
    if len(rows) >= SPAN:
        losses_all = [row[COL_SAMPLE] for row in rows]
        window = sum(losses_all[:SPAN])
        span_scores = [(window / SPAN, 0)]
        for i in range(1, len(rows) - SPAN):
            window += losses_all[i + SPAN - 1] - losses_all[i - 1]
            span_scores.append((window / SPAN, i))
        span_scores.sort(key=lambda x: -x[0])
        seen = set()
        shown = 0
        for avg_loss, start_i in span_scores:
            if shown >= 10: break
            # Deduplicate overlapping spans
            if any(abs(start_i - s) < SPAN for s in seen):
                continue
            seen.add(start_i)
            raw_start = start_i + EVAL_OFFSET
            raw_end   = raw_start + SPAN
            snippet   = raw[raw_start:raw_end].decode("utf-8", errors="replace")
            snippet   = snippet.replace("\n", "\\n").replace("\t", "\\t")
            c0        = col[raw_start] if raw_start < len(col) else "?"
            ctx0      = line_context(c0, prev_line_blank[raw_start] if raw_start < len(prev_line_blank) else False,
                                         after_newline[raw_start]  if raw_start < len(after_newline) else False)
            p(f"  [{shown+1:>2}] avg={avg_loss:.4f} BPB  raw[{raw_start}:{raw_end}]  ctx={ctx0}")
            p(f"       |{snippet[:70]}|")
            shown += 1

    # ── Summary verdict ───────────────────────────────────────────────────────
    p()
    p("=" * 88)
    p("PHASE 28A VERDICT")
    p("=" * 88)
    p()
    bk_col0 = buckets.get("_col:col_0")
    bk_anl  = buckets.get("_after_nl:yes")
    if bk_col0 and bol_n > 0:
        max_sav = bol_gap_bpb * (bol_n / total_n)
        pct_bol = 100.0 * bol_n / total_n
        p(f"  BOL bytes:     {pct_bol:.1f}%  max saving: {max_sav:+.4f} BPB")
        if max_sav >= 0.05:
            p("  DECISION: BOL has sufficient volume and headroom.")
            p("  => Proceed to Phase 28B: BOL Expert Tribunal.")
        elif max_sav >= 0.02:
            p("  DECISION: BOL headroom marginal. Check worst spans.")
            p("  => If worst spans are BOL, still worth a minimal tribunal.")
            p("  => If worst spans are math/inline, investigate phrase expert instead.")
        else:
            p("  DECISION: BOL insufficient. Investigate worst span structure.")
            p("  => Next candidate: math/macro phrase expert or delimiter-pair expert.")
    p()
    p("  Full breakdown saved to: " + os.path.join(RESULTS, "phase28a_cartography.txt"))

    # ── Write report ──────────────────────────────────────────────────────────
    report_path = os.path.join(RESULTS, "phase28a_cartography.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if os.path.exists(TEL_CSV):
        os.remove(TEL_CSV)

    print(f"\nReport -> {report_path}")


if __name__ == "__main__":
    main()
