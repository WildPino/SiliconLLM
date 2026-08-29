"""P1 Stage 2 aggregation.

Primary estimator is the per-rep MEAN inside each invocation (never min-of-reps: a minimum over reps is a
maximum order statistic and biases upward by construction). This script then aggregates ACROSS invocations
and reports the between-invocation dispersion, which is the number Amendment 1 A1.6 asks for.

MOVED bytes are the reporting convention. CHARGED bytes are carried in their own column and are never
combined with moved in a ratio.
"""
import csv
import io
import math
import statistics as st
import sys

RAW = 'benchmarks/donor_adaptation/p1/results/stage2/stage2_raw.csv'
COLS = ("inv tag shape M K arm stride_pad stride_used planes threads_req threads_ach reps replicas "
        "pool_MiB mean_us median_us cv_pct sd_us matvecs_per_s moved_B moved_GBs charged_B charged_GBs").split()

rows = []
for line in io.open(RAW, encoding='utf-8', errors='replace'):
    line = line.strip()
    if not line.startswith('P1S2CSV,'):
        continue
    p = line.split(',')[1:]
    r = dict(zip(COLS, p))
    for k in ('inv', 'M', 'K', 'stride_pad', 'stride_used', 'planes', 'threads_req',
              'threads_ach', 'reps', 'replicas'):
        r[k] = int(r[k])
    for k in ('pool_MiB', 'mean_us', 'median_us', 'cv_pct', 'sd_us', 'matvecs_per_s',
              'moved_GBs', 'charged_GBs'):
        r[k] = float(r[k])
    for k in ('moved_B', 'charged_B'):
        r[k] = int(r[k])
    rows.append(r)

if not rows:
    print('NO ROWS'); sys.exit(2)

invs = sorted({r['inv'] for r in rows})
print('invocations aggregated: %s (n=%d)' % (invs, len(invs)))
print('rows: %d' % len(rows))


def key(r):
    return (r['tag'], r['shape'], r['stride_pad'], r['threads_req'], r['arm'])


groups = {}
for r in rows:
    groups.setdefault(key(r), []).append(r)


def agg(g):
    """Aggregate one cell across invocations. Returns dict of central value + dispersion."""
    mv = [x['matvecs_per_s'] for x in g]
    gb = [x['moved_GBs'] for x in g]
    cg = [x['charged_GBs'] for x in g]
    cv = [x['cv_pct'] for x in g]
    n = len(mv)
    d = {
        'n': n,
        'mv': st.mean(mv), 'mv_sd': st.stdev(mv) if n > 1 else 0.0,
        'mv_min': min(mv), 'mv_max': max(mv),
        'gb': st.mean(gb), 'gb_sd': st.stdev(gb) if n > 1 else 0.0,
        'gb_min': min(gb), 'gb_max': max(gb),
        'cg': st.mean(cg),
        'within_cv': st.mean(cv),
        'moved_B': g[0]['moved_B'], 'charged_B': g[0]['charged_B'],
        'reps': g[0]['reps'], 'replicas': g[0]['replicas'],
        'threads_ach': g[0]['threads_ach'], 'stride_used': g[0]['stride_used'],
        'planes': g[0]['planes'],
    }
    d['mv_cv'] = 100.0 * d['mv_sd'] / d['mv'] if d['mv'] else 0.0
    d['gb_cv'] = 100.0 * d['gb_sd'] / d['gb'] if d['gb'] else 0.0
    return d


def emit(tag, title):
    ks = sorted({(k[1], k[2], k[3]) for k in groups if k[0] == tag})
    if not ks:
        return
    print('\n### %s' % title)
    print('\n| shape | stride pad | thr (ach) | arm | reps/inv | replicas | matvecs/s (mean of inv means) '
          '| between-inv CV | within-inv CV | moved B | **moved GB/s** | charged B | charged GB/s |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for shape, sp, thr in ks:
        for arm in ('byte', 'nibble'):
            g = groups.get((tag, shape, sp, thr, arm))
            if not g:
                continue
            a = agg(g)
            print('| %s | +%d | %d (%d) | %s | %d | %d | **%.1f** | %.1f%% | %.1f%% | %d | **%.2f** | %d | %.2f |'
                  % (shape, sp, thr, a['threads_ach'], arm, a['reps'], a['replicas'],
                     a['mv'], a['mv_cv'], a['within_cv'], a['moved_B'], a['gb'],
                     a['charged_B'], a['cg']))
    print('\n**Nibble / byte, per cell — the A1.3 discriminator:**\n')
    print('| shape | stride pad | thr | matvecs/s ratio | (range over inv) | moved GB/s ratio | (range over inv) | reading |')
    print('|---|---|---|---|---|---|---|---|')
    for shape, sp, thr in ks:
        gb_ = groups.get((tag, shape, sp, thr, 'byte'))
        gn_ = groups.get((tag, shape, sp, thr, 'nibble'))
        if not gb_ or not gn_:
            continue
        b, n = agg(gb_), agg(gn_)
        rmv = n['mv'] / b['mv']
        rgb = n['gb'] / b['gb']
        rmv_lo, rmv_hi = n['mv_min'] / b['mv_max'], n['mv_max'] / b['mv_min']
        rgb_lo, rgb_hi = n['gb_min'] / b['gb_max'], n['gb_max'] / b['gb_min']
        # A1.3's table has exactly two poles. A moved-GB/s ratio ABOVE ~1 is off that table entirely:
        # you cannot beat a byte ceiling by moving fewer bytes if bytes were the binding constraint.
        # It means the byte arm was limited per-ACCESS, not per-byte, so halving the LOAD COUNT (which
        # this intervention does as well as halving the bytes) bought more than the bytes could.
        if rgb > 1.15:
            rd = 'OFF-TABLE: byte arm access-limited, not byte-limited'
        elif rmv >= 1.80 and 0.85 <= rgb <= 1.15:
            rd = 'bandwidth-limited'
        elif rmv <= 1.15 and rgb <= 0.58:
            rd = 'compute/port-limited'
        else:
            rd = 'JOINTLY limited (intermediate)'
        print('| %s | +%d | %d | **%.3fx** | %.3f-%.3f | **%.3fx** | %.3f-%.3f | %s |'
              % (shape, sp, thr, rmv, rmv_lo, rmv_hi, rgb, rgb_lo, rgb_hi, rd))


emit('d5cd_resident', 'd5cd control - RESIDENT (single replica, block fits L3)')
emit('d5cd_streamed', 'd5cd control - STREAMED (pool >> L3)')
emit('main', 'Main sweep - stride pad +0 B (every Mpad a power of two / multiple of 4096)')
emit('main_stride64', 'Main sweep - stride pad +64 B (A1.5 stride-conflict separation)')

# --- A1.5: does +64 recover a fall? compare same arm/shape/threads across stride arms ---
print('\n### A1.5 stride-conflict separation: +64 B vs +0 B, same arm, same shape, same threads\n')
print('| shape | thr | arm | matvecs/s +0 | matvecs/s +64 | +64 / +0 | moved GB/s +0 | moved GB/s +64 | +64 / +0 |')
print('|---|---|---|---|---|---|---|---|---|')
shapes = []
for k in groups:
    if k[0] == 'main' and k[1] not in shapes:
        shapes.append(k[1])
for shape in shapes:
    for thr in (1, 6):
        for arm in ('byte', 'nibble'):
            g0 = groups.get(('main', shape, 0, thr, arm))
            g6 = groups.get(('main_stride64', shape, 64, thr, arm))
            if not g0 or not g6:
                continue
            a0, a6 = agg(g0), agg(g6)
            print('| %s | %d | %s | %.1f | %.1f | **%.3fx** | %.2f | %.2f | %.3fx |'
                  % (shape, thr, arm, a0['mv'], a6['mv'], a6['mv'] / a0['mv'],
                     a0['gb'], a6['gb'], a6['gb'] / a0['gb']))

# --- d5cd honesty: resident vs streamed ---
print('\n### d5cd meter-honesty: resident vs streamed, same arm, donor shape k/v 1024x8192\n')
print('| arm | thr | matvecs/s resident | matvecs/s streamed | resident / streamed | moved GB/s resident | moved GB/s streamed |')
print('|---|---|---|---|---|---|---|')
for arm in ('byte', 'nibble'):
    for thr in (1, 6):
        gr = groups.get(('d5cd_resident', 'llama70b_kv', 0, thr, arm))
        gs = groups.get(('d5cd_streamed', 'llama70b_kv', 0, thr, arm))
        if not gr or not gs:
            continue
        ar, as_ = agg(gr), agg(gs)
        print('| %s | %d | %.1f | %.1f | **%.2fx** | %.2f | %.2f |'
              % (arm, thr, ar['mv'], as_['mv'], ar['mv'] / as_['mv'], ar['gb'], as_['gb']))
