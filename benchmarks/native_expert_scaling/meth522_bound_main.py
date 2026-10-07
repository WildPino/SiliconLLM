"""One complete support-only incompatibility/copy lower witness."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth522_operations import Context, write


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--binding-sha', required=True); args = ap.parse_args(); ctx = None
    try:
        ctx = Context('main', args.binding_sha); np = ctx.numpy()
        import meth522_bound_math as M
        import meth522_support as S
        write(ctx.out / 'controls.json', M.controls())
        m, occ, q, card, counts = S.load(ctx); dev = (m[:, 4] & 5) != 0
        ctx.r['gates']['NEW_nontrivial_clique_copy_and_width_controls_ALL_original_UID_support_metric_joins'] = True
        cliques = np.full((128, 308), 0xffffffff, '<u4'); unions = np.zeros((128, 3072), 'u1')
        order = []; trace = []; cases = []
        for e in range(128):
            ids = np.flatnonzero(dev & (m[:, 3] == e)); allids = np.flatnonzero(m[:, 3] == e)
            if not len(ids):
                assert e == 0 and not len(allids)
                cases.append({'expert': e, 'development': 0, 'consumed': 0, 'status': 'EMPTY_UNCOMPILED_UNPROMOTED',
                              'clique_UIDs': [], 'lower_bound_incidence_copies': None, 'lower_bound_branches': None})
                continue
            support = q[ids] != 0
            bits = [int.from_bytes(v.tobytes(), 'little') for v in np.packbits(support, axis=1, bitorder='little')]
            rec, steps, union = M.clique(bits, list(map(int, ids)), 512, 3072)
            chosen = rec['clique_UIDs']; cliques[e, :len(chosen)] = chosen
            unions[e] = np.unpackbits(np.frombuffer(union.to_bytes(384, 'little'), 'u1'), bitorder='little')
            assert int(unions[e].sum()) == rec['clique_support_union']
            order.extend(rec.pop('ordered_UIDs'))
            for k, size, accepted, checked, blocker, bsize in steps:
                trace.append((k, e, size, accepted, 0, checked, 0xffffffff if blocker < 0 else blocker, bsize, 0))
            wide = ids[card[ids] > 512].astype('<u4').tolist(); copybad = rec['lower_bound_incidence_copies'] > 6144
            status = 'ACTIVE_WIDTH_OBSTRUCTED' if wide else 'COPY_BUDGET_OBSTRUCTED' if copybad else 'NECESSARY_BOUNDS_FIT_INCONCLUSIVE'
            cases.append({'expert': e, 'development': len(ids), 'consumed': len(allids) - len(ids),
                          'max_development_support': int(card[ids].max()), 'development_support_above512_UIDs': wide,
                          'width_obstructed': bool(wide), 'copy_budget_obstructed': copybad, **rec,
                          'logical_dense_I8_coefficient_bytes_lower_bound': 1536 * rec['lower_bound_incidence_copies'],
                          'atom_incidence_rho_lower_bound': rec['lower_bound_incidence_copies'] / 3072,
                          'status': status})
            ctx.guard()
        assert len(order) == len(trace) == 11721 and len(cases) == 128
        np.save(ctx.out / 'support_cardinality.npy', card, allow_pickle=False)
        np.save(ctx.out / 'clique_UIDs.npy', cliques, allow_pickle=False)
        np.save(ctx.out / 'clique_union.npy', unions, allow_pickle=False)
        np.save(ctx.out / 'development_order.npy', np.array(order, '<u4'), allow_pickle=False)
        np.save(ctx.out / 'clique_trace.npy', np.array(trace, S.TRACE), allow_pickle=False)
        write(ctx.out / 'development_witness_freeze.json', {'cases': cases, 'before_consumed_threshold_reports': True,
              'construction_uses_only_own_development_supports': True})
        ctx.r['gates']['ALL127_deterministic_development_cliques_complete_trace_pairwise_width_and_copy_witnesses'] = True
        reports = S.reports(m, occ, card, counts); exposed = cases[1:]
        eligibility = {'ALL127_development_supports_fit512': all(not v['width_obstructed'] for v in exposed),
                       'ALL127_clique_copy_lower_bounds_le6144': all(not v['copy_budget_obstructed'] for v in exposed)}
        views = {'development': {'UIDs': 11721, 'exposed_parents': 127,
                 'width_obstructed_parents': sum(v['width_obstructed'] for v in exposed),
                 'copy_bound_obstructed_parents': sum(v['copy_budget_obstructed'] for v in exposed),
                 'necessary_bounds_fit_parents': sum(v['status'] == 'NECESSARY_BOUNDS_FIT_INCONCLUSIVE' for v in exposed),
                 'maximum_clique_size': max(len(v['clique_UIDs']) for v in exposed),
                 'maximum_incidence_copy_lower_bound': max(v['lower_bound_incidence_copies'] for v in exposed)},
                 'source_support_domains': reports}
        ctx.r['gates']['ALL_original_parent_rare_384_book_mode_six_role_mode_denominators_retained_after_witness_freeze'] = True
        ctx.r['gates']['frozen_B512_rho2_necessary_bound_decisions_no_packing_selector_source_or_quality_claim'] = True
        ctx.finish({'cases': cases, 'views': views, 'eligibility': eligibility,
                    'source_response_function_calls': 0, 'model_calls': 0, 'native_calls': 0, 'readout_fits': 0,
                    'physical_DRAM_verified': False,
                    'decision': 'EXACT_SUPPORT_B512_RHO2_OBSTRUCTED_ON_DEVELOPMENT' if not all(eligibility.values()) else 'NECESSARY_BOUNDS_FIT_NO_COVER_EXISTENCE_PROOF',
                    'scope': 'Finite development incompatibility/copy necessary witness. Copy count is logical incidence; coefficient bytes assume dense separately stored branch copies. No approximate-function impossibility, cover, routing, quality/rate, useful n/DRAM/family or goal promotion.'})
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
