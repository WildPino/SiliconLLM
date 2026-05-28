"""
Phase 28A.2: Delimited Span Cartography

Pure measurement — no C changes. Questions:

  Q-VOL.  % bytes inside backtick spans  `...`
          % bytes inside dollar spans    $...$
          (If volume * gap < 0.05 BPB globally, SPANPFX is not worth building.)

  Q-GAP.  actual BPB, oracle BPB, moe_gap per span class.

  Q-MAX.  Max global saving = gap_in_span × fraction_in_span.
          This is the ceiling for a perfect SPANPFX expert.

  Q-SPAN. Sample bytes per class so we can verify the state machine is correct.

State machine (single-pass, streaming-compatible):
  BACKTICK: entered on ` not inside $, exited on matching `.
  DOLLAR:   entered on $ not inside `, exited on matching $.
  Nested/escaped handling: minimal — no lookback, no escaped-char tracking.
  Code fences (```) are NOT tracked here (Phase 28B+).
"""

import os, csv, math, collections, subprocess, time

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEE     = os.path.join(ROOT, "see.exe")
WEIGHTS = os.path.join(ROOT, "weights", "entropy_weights_factors_r16.bin")
RESULTS = os.path.join(ROOT, "results")
DATA    = os.path.join(ROOT, "data")
MD_PATH = os.path.join(DATA, "markdown_docs.md")
TEL_CSV = os.path.join(ROOT, "tmp_28a2_tel.csv")

EVAL_OFFSET = 2  # first 2 bytes are seed, not in telemetry

COL_TARGET   = 1
COL_L_SEE    = 2
COL_L_UNI    = 3
COL_L_BI     = 4
COL_L_LZ     = 5
COL_L_TOKPFX = 6
COL_SAMPLE   = 8
EXPERT_COLS  = [COL_L_SEE, COL_L_UNI, COL_L_BI, COL_L_LZ, COL_L_TOKPFX]
EXPERT_NAMES = ["SEE", "UNI", "BI", "LZ", "TOKPFX"]

SPAN_NONE    = 0
SPAN_BACKTICK = 1
SPAN_DOLLAR  = 2
SPAN_NAMES   = {SPAN_NONE: "NONE", SPAN_BACKTICK: "BACKTICK", SPAN_DOLLAR: "DOLLAR"}


# ── Span state machine ────────────────────────────────────────────────────────

def build_span_state(raw: bytes) -> list:
    """Return span class for each byte position (SPAN_NONE/BACKTICK/DOLLAR).

    Rules (single-pass, minimal):
    - ` toggles BACKTICK when not in DOLLAR span.
    - $ toggles DOLLAR when not in BACKTICK span.
    - ``` (three consecutive backticks) opens a code-fence block that closes
      on the next ``` — these bytes are classified as NONE to keep this
      cartography focused on inline spans only.
    - \n resets any open span (malformed inline span cannot cross line).
    """
    n      = len(raw)
    state  = [SPAN_NONE] * n
    cur    = SPAN_NONE

    i = 0
    while i < n:
        b = raw[i]

        # Newline always resets
        if b == ord('\n'):
            cur = SPAN_NONE
            state[i] = SPAN_NONE
            i += 1
            continue

        if cur == SPAN_NONE:
            if b == ord('`'):
                # Check for code fence (``` at column 0 or after only spaces)
                if i + 2 < n and raw[i+1] == ord('`') and raw[i+2] == ord('`'):
                    # Skip entire code fence block (find closing ```)
                    j = i + 3
                    while j + 2 < n:
                        if raw[j] == ord('`') and raw[j+1] == ord('`') and raw[j+2] == ord('`'):
                            j += 3
                            break
                        j += 1
                    for k in range(i, min(j, n)):
                        state[k] = SPAN_NONE
                    i = j
                    continue
                else:
                    cur = SPAN_BACKTICK
            elif b == ord('$'):
                cur = SPAN_DOLLAR

        elif cur == SPAN_BACKTICK:
            if b == ord('`'):
                state[i] = SPAN_BACKTICK  # mark the closing ` too
                cur = SPAN_NONE
                i += 1
                continue

        elif cur == SPAN_DOLLAR:
            if b == ord('$'):
                state[i] = SPAN_DOLLAR    # mark the closing $ too
                cur = SPAN_NONE
                i += 1
                continue

        state[i] = cur
        i += 1

    return state


# ── Bucket ────────────────────────────────────────────────────────────────────

class Bucket:
    def __init__(self, name):
        self.name         = name
        self.n            = 0
        self.sum_actual   = 0.0
        self.sum_oracle   = 0.0
        self.expert_wins  = [0] * len(EXPERT_NAMES)
        self.samples      = []   # (byte_val, actual_loss) for up to 20 examples

    def add(self, actual, oracle, best_e, byte_val):
        self.n           += 1
        self.sum_actual  += actual
        self.sum_oracle  += oracle
        self.expert_wins[best_e] += 1
        if len(self.samples) < 20:
            self.samples.append((byte_val, actual))

    def actual_bpb(self): return self.sum_actual / self.n if self.n else 0.0
    def oracle_bpb(self): return self.sum_oracle / self.n if self.n else 0.0
    def gap(self):        return self.actual_bpb() - self.oracle_bpb()

    def max_global_saving(self, total_n):
        return self.gap() * (self.n / total_n) if total_n else 0.0


def run_telemetry():
    print("Running telemetry audit (--expert-profile general)...")
    t0 = time.perf_counter()
    r = subprocess.run(
        [SEE, "audit", MD_PATH, "--weights", WEIGHTS, "--blend", "moe",
         "--expert-profile", "general", "--telemetry", TEL_CSV],
        capture_output=True, text=True, timeout=300
    )
    print(f"  Done in {time.perf_counter()-t0:.1f}s")
    return r.stdout + r.stderr


def main():
    os.makedirs(RESULTS, exist_ok=True)

    if not os.path.exists(MD_PATH):
        print(f"ERROR: {MD_PATH} not found"); return

    audit_out = run_telemetry()

    with open(MD_PATH, "rb") as f:
        raw = f.read()

    print("Building span state machine...")
    span_state = build_span_state(raw)

    # Sanity-check: count span bytes
    counts = collections.Counter(span_state)
    total_raw = len(raw)
    print(f"  NONE={counts[SPAN_NONE]} ({100*counts[SPAN_NONE]/total_raw:.1f}%)  "
          f"BACKTICK={counts[SPAN_BACKTICK]} ({100*counts[SPAN_BACKTICK]/total_raw:.1f}%)  "
          f"DOLLAR={counts[SPAN_DOLLAR]} ({100*counts[SPAN_DOLLAR]/total_raw:.1f}%)")

    print("Parsing telemetry CSV...")
    rows = []
    with open(TEL_CSV, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 9:
                try:
                    rows.append([float(x) for x in row])
                except ValueError:
                    pass
    print(f"  {len(rows)} rows")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    buckets = {k: Bucket(SPAN_NAMES[k]) for k in [SPAN_NONE, SPAN_BACKTICK, SPAN_DOLLAR]}
    total_n = 0
    total_actual = 0.0
    total_oracle = 0.0

    for row_i, row in enumerate(rows):
        raw_idx = row_i + EVAL_OFFSET
        if raw_idx >= len(raw): break

        actual  = row[COL_SAMPLE]
        losses  = [row[c] for c in EXPERT_COLS]
        oracle  = min(losses)
        best_e  = losses.index(oracle)
        sp      = span_state[raw_idx]

        buckets[sp].add(actual, oracle, best_e, raw[raw_idx])
        total_n      += 1
        total_actual += actual
        total_oracle += oracle

    # ── Report ─────────────────────────────────────────────────────────────────
    lines = []
    def p(s=""):
        print(s)
        lines.append(s)

    p("=" * 80)
    p("PHASE 28A.2: Delimited Span Cartography")
    p("=" * 80)
    p()
    p(f"Dataset:       {MD_PATH}")
    p(f"Bytes in eval: {total_n}")
    g_actual = total_actual / total_n
    g_oracle = total_oracle / total_n
    p(f"Global actual: {g_actual:.4f} BPB")
    p(f"Global oracle: {g_oracle:.4f} BPB")
    p(f"Global gap:    {g_actual - g_oracle:+.4f} BPB")

    p()
    p("─" * 80)
    p("Q-VOL / Q-GAP / Q-MAX: Per span-class breakdown")
    p("─" * 80)
    p(f"  {'Class':<12}  {'n':>8}  {'%':>5}  {'actual':>7}  {'oracle':>7}  {'gap':>7}  {'max_save_global':>16}")
    p("  " + "-" * 72)

    threshold_met = False
    for sp_key in [SPAN_NONE, SPAN_BACKTICK, SPAN_DOLLAR]:
        bk   = buckets[sp_key]
        pct  = 100.0 * bk.n / total_n if total_n else 0
        ms   = bk.max_global_saving(total_n)
        flag = " <== TARGET" if ms >= 0.05 else (" <-- marginal" if ms >= 0.02 else "")
        if ms >= 0.05: threshold_met = True
        p(f"  {bk.name:<12}  {bk.n:>8}  {pct:>5.1f}%  "
          f"{bk.actual_bpb():>7.4f}  {bk.oracle_bpb():>7.4f}  "
          f"{bk.gap():>+7.4f}  {ms:>+16.4f}{flag}")

    p()
    p("  Expert oracle wins per class:")
    for sp_key in [SPAN_NONE, SPAN_BACKTICK, SPAN_DOLLAR]:
        bk = buckets[sp_key]
        if bk.n == 0: continue
        wins = "  ".join(f"{EXPERT_NAMES[e]}={bk.expert_wins[e]}" for e in range(len(EXPERT_NAMES)))
        p(f"  {bk.name:<12}  {wins}")

    # ── Combined headroom ──────────────────────────────────────────────────────
    p()
    p("─" * 80)
    p("Q-MAX: Combined SPANPFX headroom (BACKTICK + DOLLAR)")
    p("─" * 80)
    bt = buckets[SPAN_BACKTICK]
    dl = buckets[SPAN_DOLLAR]
    combined_n    = bt.n + dl.n
    combined_save = bt.max_global_saving(total_n) + dl.max_global_saving(total_n)
    combined_pct  = 100.0 * combined_n / total_n if total_n else 0
    p(f"  Combined bytes:  {combined_n} ({combined_pct:.1f}%)")
    p(f"  Combined max saving: {combined_save:+.4f} BPB")
    if combined_save >= 0.05:
        p(f"  => SUFFICIENT headroom for -0.05 BPB target.")
        p(f"  => Proceed to Phase 28B: SPANPFX Expert Tribunal.")
    elif combined_save >= 0.02:
        p(f"  => MARGINAL headroom ({combined_save:.4f}). A well-calibrated expert")
        p(f"     might capture 50-70% of oracle — borderline for target.")
        p(f"  => Consider proceeding to 28B with adjusted criterion (>0.02 BPB).")
    else:
        p(f"  => INSUFFICIENT headroom ({combined_save:.4f} < 0.02).")
        p(f"  => Gap is too dispersed for a span-delimited expert.")
        p(f"  => Markdown gap may be a streaming architecture limit. Accept it.")

    # ── Sample bytes per class ─────────────────────────────────────────────────
    p()
    p("─" * 80)
    p("Q-SPAN: Sample bytes per class (verify state machine)")
    p("─" * 80)
    for sp_key in [SPAN_BACKTICK, SPAN_DOLLAR]:
        bk = buckets[sp_key]
        if bk.n == 0:
            p(f"  {bk.name}: no bytes found")
            continue
        sample_str = "".join(
            chr(b) if 32 <= b < 127 else f"\\x{b:02x}"
            for b, _ in bk.samples
        )
        p(f"  {bk.name} (first {len(bk.samples)} bytes): |{sample_str}|")

    # Show example span contexts from raw file
    p()
    p("  Example spans from raw file:")
    for sp_key, delim_open, delim_close in [(SPAN_BACKTICK, '`', '`'), (SPAN_DOLLAR, '$', '$')]:
        bk = buckets[sp_key]
        if bk.n == 0: continue
        # Find first 5 complete spans in raw
        found = 0
        i = 0
        while i < len(raw) and found < 5:
            if span_state[i] == sp_key:
                j = i
                while j < len(raw) and span_state[j] == sp_key:
                    j += 1
                snippet = raw[i:j].decode("utf-8", errors="replace")
                snippet = snippet.replace("\n", "\\n")
                p(f"    {bk.name}[{found+1}]: |{delim_open}{snippet[:60]}{delim_close}|")
                found += 1
                i = j
            else:
                i += 1

    # ── Verdict ────────────────────────────────────────────────────────────────
    p()
    p("=" * 80)
    p("PHASE 28A.2 VERDICT")
    p("=" * 80)
    p()
    p(f"  BACKTICK bytes: {bt.n} ({100*bt.n/total_n:.1f}%)  "
      f"gap={bt.gap():+.4f}  max_save={bt.max_global_saving(total_n):+.4f} BPB")
    p(f"  DOLLAR bytes:   {dl.n} ({100*dl.n/total_n:.1f}%)  "
      f"gap={dl.gap():+.4f}  max_save={dl.max_global_saving(total_n):+.4f} BPB")
    p(f"  Combined max:   {combined_save:+.4f} BPB")
    p()
    if combined_save >= 0.05:
        p("  RESULT: Volume + headroom PASS.")
        p("  Phase 28B: implement SPANPFX expert (event-driven, BACKTICK+DOLLAR).")
        p("  Eligibility: inside span only. Frozen outside. Separate key seed per span_type.")
    elif combined_save >= 0.02:
        p("  RESULT: Volume + headroom MARGINAL.")
        p("  Consider Phase 28B with adjusted BPB threshold (>0.02 instead of >0.05).")
        p("  Or: investigate whether the gap is inside long spans vs short spans.")
    else:
        p("  RESULT: SPANPFX INSUFFICIENT.")
        p("  Markdown gap is dispersed — not localized in delimited spans.")
        p("  Accept markdown gap as streaming architecture limit for now.")

    report_path = os.path.join(RESULTS, "phase28a2_span_cartography.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport -> {report_path}")

    for p_ in [TEL_CSV]:
        if os.path.exists(p_): os.remove(p_)


if __name__ == "__main__":
    main()
