"""One dev-only hierarchical innovation plan, positive rank and stability screen."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth520_operations import Context, load_geometry, write


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--binding-sha', required=True); args = ap.parse_args(); ctx = None
    try:
        ctx = Context('main', args.binding_sha); np = ctx.numpy()
        import meth520_math as M
        write(ctx.out / 'controls.json', M.controls())
        uid, occ, ids, codes, alpha, h, kreg = load_geometry(np, ctx); m = uid['m']; L = 513
        t = np.linalg.cholesky(kreg); factor = M.envelope(t, t.T, kreg)
        z = np.linalg.solve(t, h.T).T; white = M.envelope(t, z.T, h.T)
        finite = M.field(codes, alpha.view('<u4'))
        where = np.full(17540, -1, '<i4'); where[ids] = np.arange(len(ids))
        ctx.r['gates']['original_ALL_UID_occurrence_roles_dev_only_feature_metric_whitening_and_NEW_controls'] = True
        print(json.dumps({'phase': 'pooled_development_selection', 'seconds': ctx.resources()['seconds']}), flush=True)
        selected, innovation, relative, maximum = M.greedy(z, ids, np.empty((0, L)), np.array([], '<u4'), ctx.guard)
        global_ids = ids[selected]; global_cert = M.minor(finite[selected], ctx.guard)
        global_stats, global_singular = M.stability(z[selected])
        np.save(ctx.out / 'global_UID_order.npy', global_ids.astype('<u4'), allow_pickle=False)
        np.save(ctx.out / 'global_innovation.npy', innovation, allow_pickle=False)
        np.save(ctx.out / 'global_greedy_maximum.npy', maximum, allow_pickle=False)
        np.save(ctx.out / 'global_singular.npy', global_singular, allow_pickle=False)
        write(ctx.out / 'global_basis_freeze.json', {'UIDs': global_ids.tolist(), 'certificate': global_cert, 'stability': global_stats,
                                                    'minimum_relative_innovation': float(relative.min()),
                                                    'only_development': True, 'before_all_parent_complements_and_responses': True})
        candidate_ids = np.sort(global_ids); candidate_pos = where[candidate_ids]; candidate_z = z[candidate_pos]
        augmented = np.empty((128, L), '<u4'); new_innovation = np.full((128, L), -1., '<f8')
        greedy_max = np.full((128, L), -1., '<f8'); spectra = np.empty((128, L), '<f8')
        cases = []; pairs = []
        for e in range(128):
            own = ids[m[ids, 3] == e]; own_pos = where[own]; count = len(own)
            chosen, inv, rel, maxscore = M.greedy(candidate_z, candidate_ids, z[own_pos], own, ctx.guard)
            added = candidate_ids[chosen]; aug = np.concatenate((own, added)); assert len(aug) == L and len(np.unique(aug)) == L
            pos = where[aug]; assert np.all(pos >= 0)
            cert = M.minor(finite[pos], ctx.guard); stats, singular = M.stability(z[pos])
            augmented[e] = aug; new_innovation[e, count:] = inv; greedy_max[e, count:] = maxscore; spectra[e] = singular
            cases.append({'parent': e, 'original_development_UIDs': count, 'new_pairs': len(added),
                          'certificate': cert, 'stability': stats, 'minimum_new_relative_innovation': float(rel.min())})
            pairs.extend((e, int(m[k, 3]), int(k)) for k in added)
            ctx.guard()
            if e % 8 == 7: print(json.dumps({'parent_complement_terminal': e, 'seconds': ctx.resources()['seconds']}), flush=True)
        pair_array = np.array(pairs, dtype=[('expert', '<u2'), ('input_owner', '<u2'), ('UID', '<u4')])
        assert len(pairs) == 128 * L - len(ids) == 53943
        np.save(ctx.out / 'augmented_UID_order.npy', augmented, allow_pickle=False)
        np.save(ctx.out / 'parent_new_innovation.npy', new_innovation, allow_pickle=False)
        np.save(ctx.out / 'parent_greedy_maximum.npy', greedy_max, allow_pickle=False)
        np.save(ctx.out / 'parent_singular.npy', spectra, allow_pickle=False)
        np.save(ctx.out / 'query_pairs.npy', pair_array, allow_pickle=False)
        provenance = np.empty(len(ids), dtype=[('UID', '<u4'), ('input_code_pair_SHA', 'u1', (96,))])
        provenance['UID'] = ids; provenance['input_code_pair_SHA'] = uid['hash'][ids, :96]
        np.save(ctx.out / 'development_query_sources.npy', provenance, allow_pickle=False)
        write(ctx.out / 'complement_certificates.json', {'prime': M.P, 'cases': cases, 'global_certificate': global_cert})
        eligibility = {'pooled_full513_real_rank_positive_certificate': global_cert['positive_real_full_rank_certificate'],
                       'ALL128_augmented_full513_real_rank_positive_certificates': all(v['certificate']['positive_real_full_rank_certificate'] for v in cases),
                       'global_AND_ALL128_condition2_le1e6': global_stats['condition2'] <= 1e6 and all(v['stability']['condition2'] <= 1e6 for v in cases),
                       'global_AND_ALL128_new_relative_innovation_ge1e-6': relative.min() >= 1e-6 and all(v['minimum_new_relative_innovation'] >= 1e-6 for v in cases)}
        ctx.r['gates'].update(complete_dev_only_pooled_basis_ALL128_complement_manifest_positive_or_inconclusive_minor_receipts=True,
                             ALL128_full_numerical_QR_spectra_innovation_and_exact53943_pair_provenance=True,
                             frozen_rank_stability_decisions_and_zero_source_response_calls=True)
        views = {'development': {'UIDs': len(ids), 'pooled_real_rank_lower_bound': L if eligibility['pooled_full513_real_rank_positive_certificate'] else 308,
                                 'pooled_real_rank_upper_bound': L, 'new_pairs_in_manifest': len(pairs), 'parents': 128},
                 'stability': {'global_condition2': global_stats['condition2'], 'maximum_parent_condition2': max(v['stability']['condition2'] for v in cases),
                               'minimum_new_relative_innovation': min(float(relative.min()), *(v['minimum_new_relative_innovation'] for v in cases))}}
        ctx.finish({'views': views, 'eligibility': eligibility, 'global_certificate': global_cert, 'global_stability': global_stats,
                    'global_minimum_relative_innovation': float(relative.min()), 'cases': cases, 'metric_factor_envelope': factor, 'whitening_envelope': white,
                    'selected_pairs': len(pairs), 'prospective_source_function_MACs': len(pairs) * 4718592,
                    'prospective_F32_targets_bytes': len(pairs) * 3072, 'source_response_function_calls': 0,
                    'readout_coefficient_solves': 0, 'new_model_native_capture_calls': 0, 'physical_DRAM_verified': False,
                    'decision': 'ELIGIBLE_FOR_SEPARATELY_FROZEN_SOURCE_RESPONSES' if all(eligibility.values()) else 'COMPLEMENT_PLAN_RANK_OR_NUMERICAL_ELIGIBILITY_FAIL',
                    'scope': 'Development geometry/source-query manifest only; positive minors prove rank, numerical receipts qualify stability. No fitted functions, routing, quality/rate or goal completion.'})
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
