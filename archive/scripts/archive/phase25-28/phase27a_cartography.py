"""
Phase 27A: Loss Cartography
Maps where SEE loses entropy — by byte class, position context, and expert profile.
No decisions here. Cartography first, surgery later.

Outputs:
  results/phase27a_cartography_report.txt
  results/phase27a_cartography_{dataset}.csv  (annotated per-byte)

Oracle definitions:
  oracle_expert = min(active expert losses per byte)   — best available expert
  actual_gap    = loss_actual - oracle_expert           — MoE credit assignment gap
  blind_gap     = oracle_expert                         — true new-expert headroom

Reading key (apply after seeing data):
  blind HIGH + actual_gap LOW   -> oracle knows; MoE didn't assign credit
  blind HIGH + actual_gap HIGH  -> no expert has it; new organ candidate
  blind LOW  (oracle < 1.5 BPB) -> existing experts already cover this region

Candidate mapping:
  LETTER/DIGIT inside alnum run, blind high -> Token-LZ / word-prefix expert
  NEWLINE / col0, blind high               -> line-state expert
  MD_MARKER, blind high                    -> markup-structure expert
  any class, actual_gap >> blind           -> MoE tuning, not new expert
"""

import os
import csv
import subprocess
import tempfile
from collections import defaultdict

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEE     = os.path.join(ROOT, "see.exe")
WEIGHTS = os.path.join(ROOT, "weights", "entropy_weights.bin")
RESULTS = os.path.join(ROOT, "results")
DATA    = os.path.join(ROOT, "data")

DATASETS = [
    ("c_code",        os.path.join(DATA, "c_code.c")),
    ("natural_text",  os.path.join(DATA, "natural_text.txt")),
    ("markdown_docs", os.path.join(DATA, "markdown_docs.md")),
    ("shuffled",      os.path.join(DATA, "shuffled.bin")),
    ("multi_domain",  os.path.join(DATA, "multi_domain.bin")),
]

# Markdown structural characters per user spec
MD_MARKER_BYTES = set(ord(c) for c in '#*_`[]()!|>-+=:')

EXPERT_KEYS = ['loss_see', 'loss_uni', 'loss_bi', 'loss_lz']
EXPERT_NAMES = ['SEE', 'UNI', 'BI', 'LZ']

SPAN_WINDOW = 48
TOP_SPANS   = 20


# ── Byte classification ────────────────────────────────────────────────────────

def byte_class(b):
    if b == 10: return 'NEWLINE'
    if b == 32: return 'SPACE'
    if b in (9, 13): return 'WHITESPACE'
    if 48 <= b <= 57: return 'DIGIT'
    if (65 <= b <= 90) or (97 <= b <= 122): return 'LETTER'
    if b in MD_MARKER_BYTES: return 'MD_MARKER'
    if 33 <= b <= 126: return 'PUNCT'
    if b < 32: return 'CONTROL'
    return 'HIGH'

def lp_bucket(pos):
    if pos == 0: return 'col0'
    if pos <= 4: return 'col1-4'
    if pos <= 16: return 'col5-16'
    return 'col17+'

def sanitize(byte_list):
    return ''.join(chr(b) if 32 <= b <= 126 else '.' for b in byte_list)


# ── Telemetry I/O ──────────────────────────────────────────────────────────────

def run_telemetry(filepath):
    fd, tel_path = tempfile.mkstemp(suffix='.csv')
    os.close(fd)
    try:
        result = subprocess.run(
            [SEE, 'audit', filepath, '--weights', WEIGHTS, '--blend', 'moe',
             '--telemetry', tel_path],
            capture_output=True, text=True, timeout=600
        )
        if os.path.exists(tel_path) and os.path.getsize(tel_path) > 0:
            return tel_path, result.stdout + result.stderr
        return None, result.stdout + result.stderr
    except Exception as e:
        return None, str(e)


def load_and_annotate(tel_path):
    """Load telemetry CSV and add context features + oracle metrics per byte."""
    rows = []
    line_pos = 0
    prev_b = None
    prev_class = 'NONE'

    # First pass: compute avg w_lz8 to decide if LZ8 is an active expert
    w_lz8_sum = 0.0
    row_count = 0
    with open(tel_path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            w_lz8_sum += float(r['w_lz8'])
            row_count += 1
    lz8_active = (row_count > 0) and (w_lz8_sum / row_count > 0.05)

    active_loss_keys  = EXPERT_KEYS + (['loss_lz8'] if lz8_active else [])
    active_exp_names  = EXPERT_NAMES + (['LZ8']      if lz8_active else [])

    with open(tel_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            target  = int(float(row['target']))
            loss_actual = float(row['loss_actual'])

            bc = byte_class(target)
            after_nl    = (prev_b == 10) if prev_b is not None else False
            inside_alnum = (bc in ('LETTER', 'DIGIT') and
                            prev_class in ('LETTER', 'DIGIT'))

            # Oracle: best available expert at this byte
            active_losses = [float(row[k]) for k in active_loss_keys]
            oracle        = min(active_losses)
            dom_idx       = active_losses.index(oracle)
            dom_expert    = active_exp_names[dom_idx]

            # actual_gap: how much the MoE over-spent vs oracle
            # Can be 0 (MoE = oracle or better via mixture benefit)
            actual_gap = max(0.0, loss_actual - oracle)
            # blind_gap: how blind the best expert is — true new-organ space
            blind_gap  = oracle

            rows.append({
                'i':           int(row['i']),
                'target':      target,
                'prev_target': prev_b if prev_b is not None else -1,
                'byte_class':  bc,
                'prev_class':  prev_class,
                'line_pos':    line_pos,
                'lp_bucket':   lp_bucket(line_pos),
                'after_newline':  after_nl,
                'inside_alnum':   inside_alnum,
                'loss_actual':    loss_actual,
                'oracle':         oracle,
                'actual_gap':     actual_gap,
                'blind_gap':      blind_gap,
                'dom_expert':     dom_expert,
            })

            # Advance position state
            if target == 10:
                line_pos = 0
            else:
                line_pos += 1
            prev_b    = target
            prev_class = bc

    return rows, lz8_active


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate(rows, key_fn):
    groups = defaultdict(lambda: {
        'count': 0,
        'loss_sum': 0.0, 'oracle_sum': 0.0,
        'actual_gap_sum': 0.0, 'blind_gap_sum': 0.0,
        'expert_wins': defaultdict(int),
    })
    for r in rows:
        k = key_fn(r)
        g = groups[k]
        g['count']           += 1
        g['loss_sum']        += r['loss_actual']
        g['oracle_sum']      += r['oracle']
        g['actual_gap_sum']  += r['actual_gap']
        g['blind_gap_sum']   += r['blind_gap']
        g['expert_wins'][r['dom_expert']] += 1
    return groups


def fmt_rows(groups, total, sort_by='blind_gap_avg', top=None):
    out = []
    for k, g in groups.items():
        n = g['count']
        if n == 0:
            continue
        top_exp = max(g['expert_wins'], key=g['expert_wins'].get) if g['expert_wins'] else '-'
        out.append({
            'key':             k,
            'count':           n,
            'pct':             100.0 * n / total if total else 0.0,
            'loss_avg':        g['loss_sum']       / n,
            'oracle_avg':      g['oracle_sum']     / n,
            'actual_gap_avg':  g['actual_gap_sum'] / n,
            'blind_gap_avg':   g['blind_gap_sum']  / n,
            'top_expert':      top_exp,
        })
    out.sort(key=lambda x: -x[sort_by])
    return out[:top] if top else out


# ── Report formatting ──────────────────────────────────────────────────────────

_HDR = f"  {'Category':<22} {'N':>7} {'%':>5}  {'loss':>6}  {'oracle':>6}  {'moe_gap':>7}  {'blind':>6}  {'top_exp'}"
_SEP = "  " + "-" * (len(_HDR) - 2)

def print_table(rows_out, p):
    p(_HDR)
    p(_SEP)
    for r in rows_out:
        p(f"  {str(r['key']):<22} {r['count']:>7} {r['pct']:>4.1f}%  "
          f"{r['loss_avg']:>6.3f}  {r['oracle_avg']:>6.3f}  "
          f"{r['actual_gap_avg']:>7.3f}  {r['blind_gap_avg']:>6.3f}  "
          f"{r['top_expert']}")


# ── Worst-span detection ───────────────────────────────────────────────────────

def find_top_spans(rows, window=SPAN_WINDOW, topk=TOP_SPANS):
    n = len(rows)
    if n < window:
        return []

    loss_vals   = [r['loss_actual'] for r in rows]
    oracle_vals = [r['oracle']      for r in rows]

    # Sliding window sums
    win_loss = sum(loss_vals[:window])
    scores = [(0, win_loss / window)]
    for i in range(1, n - window + 1):
        win_loss += loss_vals[i + window - 1] - loss_vals[i - 1]
        scores.append((i, win_loss / window))

    scores.sort(key=lambda x: -x[1])

    selected = []
    covered  = set()
    for start, avg_loss in scores:
        if any(j in covered for j in range(start, start + window)):
            continue
        span_rows = rows[start:start + window]
        cats    = defaultdict(int)
        experts = defaultdict(int)
        oracle_sum = 0.0
        for r in span_rows:
            cats[r['byte_class']] += 1
            experts[r['dom_expert']] += 1
            oracle_sum += r['oracle']
        dom_cat    = max(cats,    key=cats.get)
        dom_expert = max(experts, key=experts.get)
        snippet    = sanitize([r['target'] for r in span_rows])
        selected.append({
            'start':      start,
            'end':        start + window - 1,
            'avg_loss':   avg_loss,
            'avg_oracle': oracle_sum / window,
            'dom_expert': dom_expert,
            'dom_class':  dom_cat,
            'snippet':    snippet,
        })
        covered.update(range(start, start + window))
        if len(selected) >= topk:
            break

    return selected


# ── Candidate hypothesis generator ────────────────────────────────────────────

def generate_hypotheses(class_groups, lp_groups, total, p):
    p()
    p("  CANDIDATE HYPOTHESES")
    p("  " + "-" * 74)
    p("  Not verdicts. Read with the key below and the tables above.")
    p()
    p("  Key:")
    p("    blind HIGH + moe_gap LOW   -> oracle knows; MoE underassigns credit")
    p("    blind HIGH + moe_gap HIGH  -> no expert has it; new organ candidate")
    p("    blind LOW                  -> existing experts already cover this")
    p()

    class_table = fmt_rows(class_groups, total, sort_by='blind_gap_avg')
    lp_table    = fmt_rows(lp_groups,    total, sort_by='blind_gap_avg')

    candidates = []

    for r in class_table:
        k    = r['key']
        blind = r['blind_gap_avg']
        mgap  = r['actual_gap_avg']
        pct   = r['pct']
        if pct < 0.3:
            continue  # too rare to matter

        if blind > 2.5 and mgap > 0.4:
            if k in ('LETTER', 'DIGIT'):
                candidates.append(
                    f"  [TOKEN-LZ / WORD-PREFIX]  {k}: blind={blind:.3f}  moe_gap={mgap:.3f}  ({pct:.1f}%)")
            elif k == 'NEWLINE':
                candidates.append(
                    f"  [LINE-STATE EXPERT]        {k}: blind={blind:.3f}  moe_gap={mgap:.3f}  ({pct:.1f}%)")
            elif k == 'MD_MARKER':
                candidates.append(
                    f"  [MARKUP-STRUCTURE EXPERT]  {k}: blind={blind:.3f}  moe_gap={mgap:.3f}  ({pct:.1f}%)")
            else:
                candidates.append(
                    f"  [UNKNOWN PATTERN]          {k}: blind={blind:.3f}  moe_gap={mgap:.3f}  ({pct:.1f}%)")
        elif blind > 1.8 and mgap > 0.5:
            candidates.append(
                f"  [MOE TUNING, not new organ]  {k}: oracle knows "
                f"(blind={blind:.3f}) but MoE gap={mgap:.3f}  ({pct:.1f}%)")

    for r in lp_table:
        k    = r['key']
        blind = r['blind_gap_avg']
        mgap  = r['actual_gap_avg']
        pct   = r['pct']
        if pct < 0.5:
            continue
        if k == 'col0' and blind > 2.0:
            candidates.append(
                f"  [LINE-START EXPERT]  col0: blind={blind:.3f}  moe_gap={mgap:.3f}  ({pct:.1f}%)")

    if candidates:
        for c in candidates:
            p(c)
    else:
        p("  No dominant structural gap detected (blind_gap < thresholds for all major classes).")
        p("  Gap may be distributional or uniformly spread — no clear new-organ target.")

    p()
    p("  VALIDATION: candidates that appear on markdown/text but NOT on shuffled are")
    p("  structural. Candidates that appear on shuffled too are distributional noise.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS, exist_ok=True)
    lines = []

    def p(s=''):
        print(s)
        lines.append(s)

    p("=" * 80)
    p("PHASE 27A: Loss Cartography — Silicon Entropy Engine")
    p("=" * 80)
    p()
    p("  loss     = avg loss_actual (mixture BPB at each byte)")
    p("  oracle   = avg min(expert losses) — best available expert")
    p("  moe_gap  = loss_actual - oracle — credit assignment gap")
    p("  blind    = oracle itself — where even best expert fails (new-expert headroom)")
    p()

    for ds_name, ds_path in DATASETS:
        if not os.path.exists(ds_path):
            p(f"[SKIP] {ds_name}: file not found at {ds_path}")
            continue

        file_size = os.path.getsize(ds_path)
        p()
        p("=" * 80)
        p(f"DATASET: {ds_name}  ({file_size / 1024:.1f} KB)")
        p("=" * 80)

        print(f"  Generating telemetry for {ds_name}...", end='', flush=True)
        tel_path, see_out = run_telemetry(ds_path)
        print(" done." if tel_path else " FAILED.")

        if tel_path is None:
            p(f"  [ERROR] Telemetry failed.")
            p(f"  see.exe output: {see_out[:300]}")
            continue

        rows, lz8_active = load_and_annotate(tel_path)
        os.unlink(tel_path)

        if not rows:
            p("  [ERROR] No telemetry rows — file may be too small.")
            continue

        total = len(rows)
        avg = lambda key: sum(r[key] for r in rows) / total

        overall_loss   = avg('loss_actual')
        overall_oracle = avg('oracle')
        overall_mgap   = avg('actual_gap')
        overall_blind  = avg('blind_gap')

        p(f"  Bytes: {total:,}   LZ8 active: {'yes' if lz8_active else 'no (standard mode)'}")
        p(f"  Overall loss:   {overall_loss:.4f} BPB")
        p(f"  Overall oracle: {overall_oracle:.4f} BPB  (best-expert lower bound)")
        p(f"  MoE gap:        {overall_mgap:.4f} BPB  (credit assignment gap)")
        p(f"  Blind gap:      {overall_blind:.4f} BPB  (true new-expert headroom)")
        p()

        # [1] By byte class
        class_groups = aggregate(rows, lambda r: r['byte_class'])
        p("  [1] BY BYTE CLASS  (sorted by blind_gap)")
        print_table(fmt_rows(class_groups, total, sort_by='blind_gap_avg'), p)
        p()

        # [2] By line position
        lp_groups = aggregate(rows, lambda r: r['lp_bucket'])
        p("  [2] BY LINE POSITION  (sorted by blind_gap)")
        print_table(fmt_rows(lp_groups, total, sort_by='blind_gap_avg'), p)
        p()

        # [3] Compound: class × lp_bucket, top 15
        compound_groups = aggregate(rows, lambda r: f"{r['byte_class']}/{r['lp_bucket']}")
        p("  [3] COMPOUND (class × line_pos)  top 15 by blind_gap")
        print_table(fmt_rows(compound_groups, total, sort_by='blind_gap_avg', top=15), p)
        p()

        # [4] Context flags: after_newline × inside_alnum
        flag_groups = aggregate(
            rows,
            lambda r: f"after_nl={int(r['after_newline'])}, alnum_run={int(r['inside_alnum'])}"
        )
        p("  [4] CONTEXT FLAGS  (after_newline × inside_alnum_run)  sorted by blind_gap")
        print_table(fmt_rows(flag_groups, total, sort_by='blind_gap_avg'), p)
        p()

        # [5] Top worst spans
        spans = find_top_spans(rows)
        p(f"  [5] TOP-{TOP_SPANS} WORST SPANS  ({SPAN_WINDOW}-byte window, non-overlapping)")
        p("  " + "-" * 78)
        p(f"  {'Bytes':>14}  {'loss':>6}  {'oracle':>6}  {'expert':>6}  {'class':>10}  Snippet (first 40 chars)")
        p("  " + "-" * 78)
        for s in spans:
            rng     = f"[{s['start']:>6},{s['end']:>6}]"
            snippet = s['snippet'][:40]
            p(f"  {rng}  {s['avg_loss']:>6.3f}  {s['avg_oracle']:>6.3f}  "
              f"{s['dom_expert']:>6}  {s['dom_class']:>10}  {snippet!r}")
        p()

        # [6] Candidate hypotheses
        generate_hypotheses(class_groups, lp_groups, total, p)

        # Save annotated CSV
        csv_out = os.path.join(RESULTS, f"phase27a_cartography_{ds_name}.csv")
        fieldnames = [
            'i', 'target', 'prev_target', 'byte_class', 'prev_class',
            'line_pos', 'lp_bucket', 'after_newline', 'inside_alnum',
            'loss_actual', 'oracle', 'actual_gap', 'blind_gap', 'dom_expert',
        ]
        with open(csv_out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        p(f"  Annotated CSV -> {csv_out}")
        p()

    report_path = os.path.join(RESULTS, "phase27a_cartography_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nReport -> {report_path}")


if __name__ == '__main__':
    main()
