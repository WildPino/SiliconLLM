"""Independent Boolean support/clique witness audit, no main math import."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth522_operations import Context, DOC, ROOT, wire


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--binding-sha', required=True); ap.add_argument('--main-sha', required=True)
    args = ap.parse_args(); ctx = None
    try:
        ctx = Context('audit', args.binding_sha)
        assert ctx.digest(DOC / 'meth522_main_result.json') == args.main_sha
        raw = json.loads((DOC / 'meth522_main_result.json').read_bytes())
        assert raw['binding_sha256'] == args.binding_sha and all(raw['gates'].values())
        folder = ROOT / 'results/native_expert_scaling/meth522_bound'
        for v in raw['output_inventory']: assert Path(v['path']).stat().st_size == v['bytes'] and ctx.digest(v['path']) == v['sha256']
        assert json.loads((folder / 'terminal_resource.json').read_bytes())['result_sha256'] == args.main_sha
        ctx.r['gates']['completed_main_ALL_current_output_and_terminal_SHA'] = True
        np = ctx.numpy()
        ctl = json.loads((folder / 'controls.json').read_bytes()); sets = [{0, 1, 2}, {0, 3, 4}, {0, 5, 6}, {0, 1}]
        assert all(len(sets[a] | sets[b]) == 5 > 3 for a in range(3) for b in range(a))
        assert sum(len(v) for v in sets[:3]) == 9 and len(set.union(*sets[:3])) == 7 and 8 + 9 - 7 == 10 > 9
        assert ctl['H8_B3_clique']['clique_UIDs'] == [0, 1, 2] and ctl['H8_B3_clique']['lower_bound_incidence_copies'] == 10
        assert ctl['trace'][-1] == [3, 2, 0, 1, 0, 3] and ctl['union_integer'] == 127
        assert ctl['support_exceeds_width'] == {'UID': 11, 'support': 4, 'width': 3}
        ctx.r['gates']['independent_nontrivial_finite_sets_clique_copy_width_and_tie_controls'] = True
        ut = np.dtype([('m', '<u4', (13,)), ('hash', 'u1', (128,))])
        uid = wire(np, ctx.data('uid'), b'M493U001', 180, 0, ut, 17540); m = uid['m']
        occ = wire(np, ctx.data('occurrences'), b'M493O001', 68, 0, np.dtype(('<u4', (17,))), 19962)
        q = np.load(ctx.data('codes'), mmap_mode='r', allow_pickle=False); metric = np.load(ctx.data('metrics'), mmap_mode='r', allow_pickle=False)
        assert q.shape == (17540, 3072) and q.dtype == np.dtype('<i2') and metric.shape == (17540,)
        dev = (m[:, 4] & 5) != 0; val = (m[:, 4] & 2) != 0
        assert np.array_equal(m[:, 0], np.arange(17540)) and np.all(m[:, 2] == 11) and np.all(m[:, 3] < 128) and np.all(m[:, 6] == 2)
        assert (int(dev.sum()), int(val.sum())) == (11721, 5819) and np.all(dev ^ val)
        counts = np.bincount(m[dev, 3], minlength=128); assert counts[0] == 0 and np.all(counts[1:] > 0) and counts.max() == 308
        assert np.array_equal(occ[:, 0], np.arange(19962)) and np.all(occ[:, 1] < 17540) and np.all(occ[:, 3] == 128) and np.all(occ[:, 8] == 11) and np.all(occ[:, 12] == 1)
        for oi, mi in ((14, 1), (11, 3), (15, 12), (16, 10)): assert np.array_equal(occ[:, oi], m[occ[:, 1], mi])
        assert np.array_equal(np.bincount(occ[:, 1], minlength=17540), m[:, 7])
        card = np.array([np.flatnonzero(row).size for row in q], '<u2')
        assert np.all((q >= 0) & (q <= 32767)) and np.array_equal(card, metric['support'])
        assert np.array_equal(card, np.load(folder / 'support_cardinality.npy', allow_pickle=False))
        ctx.r['gates']['ALL17540_independent_support_cardinality_saved500_metrics_and19962_role_joins'] = True
        clique_saved = np.load(folder / 'clique_UIDs.npy', allow_pickle=False)
        union_saved = np.load(folder / 'clique_union.npy', allow_pickle=False)
        trace_saved = np.load(folder / 'clique_trace.npy', allow_pickle=False)
        order_saved = np.load(folder / 'development_order.npy', allow_pickle=False)
        assert clique_saved.shape == (128, 308) and clique_saved.dtype == np.dtype('<u4')
        assert union_saved.shape == (128, 3072) and union_saved.dtype == np.dtype('u1')
        assert trace_saved.shape == (11721,) and order_saved.shape == (11721,)
        freeze = json.loads((folder / 'development_witness_freeze.json').read_bytes())
        assert freeze['before_consumed_threshold_reports'] and freeze['construction_uses_only_own_development_supports'] and freeze['cases'] == raw['cases']
        cursor = 0; pair_checks = 0
        for e in range(128):
            ids = np.flatnonzero(dev & (m[:, 3] == e)); case = raw['cases'][e]; assert case['expert'] == e
            if not len(ids):
                assert case['status'] == 'EMPTY_UNCOMPILED_UNPROMOTED' and not np.any(m[:, 3] == e)
                assert np.all(clique_saved[e] == 0xffffffff) and not union_saved[e].any(); continue
            support = q[ids] > 0; sizes = np.sum(support, axis=1)
            order = np.lexsort((ids, -sizes.astype('<i8'))); picked = []; union = np.zeros(3072, bool); total = 0
            for local in order:
                row = trace_saved[cursor]; uid = int(ids[local]); checked = 0; blocker = None; bsize = 0
                for other in picked:
                    checked += 1; bsize = int(np.count_nonzero(np.logical_or(support[local], support[other])))
                    if bsize <= 512: blocker = int(ids[other]); break
                accepted = blocker is None
                assert int(order_saved[cursor]) == uid and int(row['UID']) == uid and int(row['expert']) == e
                assert int(row['size']) == int(sizes[local]) and bool(row['accepted']) == accepted and int(row['checked']) == checked
                assert int(row['blocker_UID']) == (0xffffffff if accepted else blocker) and int(row['blocker_union']) == (0 if accepted else bsize)
                assert row['reserved'] == row['padding'] == 0
                if accepted: picked.append(int(local)); union = np.logical_or(union, support[local]); total += int(sizes[local])
                cursor += 1
            selected = ids[picked].astype('<u4')
            assert selected.tolist() == case['clique_UIDs'] and np.array_equal(clique_saved[e, :len(picked)], selected)
            assert np.all(clique_saved[e, len(picked):] == 0xffffffff) and np.array_equal(union_saved[e], union)
            minimum = None
            for k, at in enumerate(picked):
                for other in picked[:k]:
                    size = int(np.count_nonzero(np.logical_or(support[at], support[other]))); assert size > 512
                    minimum = size if minimum is None else min(minimum, size); pair_checks += 1
            distinct = int(np.count_nonzero(union)); lower = 3072 + total - distinct
            wide = ids[sizes > 512].tolist(); copybad = lower > 6144
            assert case['development'] == len(ids) and case['consumed'] == int(np.sum(val & (m[:, 3] == e)))
            assert case['max_development_support'] == int(sizes.max()) and case['development_support_above512_UIDs'] == wide
            assert case['clique_support_copy_sum'] == total and case['clique_support_union'] == distinct
            assert case['lower_bound_incidence_copies'] == lower and case['lower_bound_branches'] == max(len(picked), 6)
            assert case['minimum_clique_pair_union'] == minimum and case['logical_dense_I8_coefficient_bytes_lower_bound'] == lower * 1536
            assert case['atom_incidence_rho_lower_bound'] == lower / 3072 and case['width_obstructed'] == bool(wide) and case['copy_budget_obstructed'] == copybad
            assert case['status'] == ('ACTIVE_WIDTH_OBSTRUCTED' if wide else 'COPY_BUDGET_OBSTRUCTED' if copybad else 'NECESSARY_BOUNDS_FIT_INCONCLUSIVE')
            ctx.guard()
        assert cursor == 11721
        ctx.r['gates']['ALL127_independent_Boolean_clique_trace_pair_checks_and_exact_copy_lower_witnesses'] = True
        # Shared report layout only; independently count every saved selector.
        def verify(row, ids):
            cc = [int(card[k]) for k in ids]
            assert row['count'] == len(cc) and row['sum_source_support'] == sum(cc)
            assert row['max_source_support'] == max(cc, default=0) and row['source_support_above512'] == sum(v > 512 for v in cc)
        reps = raw['views']['source_support_domains']
        for row in reps['uid_roles']: verify(row, np.flatnonzero(dev if row['split'] == 'development' else val))
        for row in reps['cells']: verify(row, np.flatnonzero((dev if row['split'] == 'development' else val) & (m[:, 3] == row['expert'])))
        for row in reps['rare']:
            c = row['development_class']; group = counts == 0 if c == '0' else (counts >= 1) & (counts <= 4) if c == '1..4' else (counts >= 5) & (counts <= 15) if c == '5..15' else counts >= 16
            verify(row, np.flatnonzero((dev if row['split'] == 'development' else val) & group[m[:, 3]]))
        for row in reps['book_mode']: verify(row, occ[(occ[:, 4] == row['book']) & (occ[:, 6] == row['mode']), 1])
        for row in reps['role_mode']: verify(row, occ[(occ[:, 7] == row['role']) & (occ[:, 6] == row['mode']), 1])
        eligibility = {'ALL127_development_supports_fit512': all(not v['width_obstructed'] for v in raw['cases'][1:]),
                       'ALL127_clique_copy_lower_bounds_le6144': all(not v['copy_budget_obstructed'] for v in raw['cases'][1:])}
        assert eligibility == raw['eligibility']
        summary = raw['views']['development']; cases = raw['cases'][1:]
        assert summary == {'UIDs': 11721, 'exposed_parents': 127, 'width_obstructed_parents': sum(v['width_obstructed'] for v in cases),
            'copy_bound_obstructed_parents': sum(v['copy_budget_obstructed'] for v in cases),
            'necessary_bounds_fit_parents': sum(v['status'] == 'NECESSARY_BOUNDS_FIT_INCONCLUSIVE' for v in cases),
            'maximum_clique_size': max(len(v['clique_UIDs']) for v in cases), 'maximum_incidence_copy_lower_bound': max(v['lower_bound_incidence_copies'] for v in cases)}
        decision = 'EXACT_SUPPORT_B512_RHO2_OBSTRUCTED_ON_DEVELOPMENT' if not all(eligibility.values()) else 'NECESSARY_BOUNDS_FIT_NO_COVER_EXISTENCE_PROOF'
        assert decision == raw['decision']
        ctx.r['gates']['ALL_reports_parent_rare_book_mode_denominators_and_frozen_necessary_decisions_independently_verified'] = True
        ctx.finish({'main_sha256': args.main_sha, 'views': raw['views'], 'eligibility': eligibility, 'decision': decision,
                    'clique_pair_checks': pair_checks, 'source_response_function_calls': 0, 'model_calls': 0, 'native_calls': 0,
                    'physical_DRAM_verified': False, 'scope': 'Independent Boolean audit of all finite support/copy witnesses; no new functions, packing or selector and no whole/goal promotion.'})
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
