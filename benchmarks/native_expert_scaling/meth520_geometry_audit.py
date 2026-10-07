"""Independent Fraction minors, full QR tail-innovation and query-plan audit."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth520_operations import Context, DOC, ROOT, load_geometry
P = 2147483647


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--binding-sha', required=True); ap.add_argument('--main-sha', required=True)
    args = ap.parse_args(); ctx = None
    try:
        ctx = Context('audit', args.binding_sha); path = DOC / 'meth520_main_result.json'
        assert ctx.digest(path) == args.main_sha; raw = json.loads(path.read_bytes())
        assert all(raw['gates'].values()) and raw['binding_sha256'] == args.binding_sha
        for v in raw['output_inventory']: assert ctx.digest(v['path']) == v['sha256']
        np = ctx.numpy(); uid, occ, ids, codes, alpha, h, kreg = load_geometry(np, ctx); m = uid['m']; L = 513
        folder = ROOT / 'results/native_expert_scaling/meth520_geometry'
        control = json.loads((folder / 'controls.json').read_bytes())
        assert all(P % d for d in range(2, math.isqrt(P) + 1)) and (P - 1) ** 2 < 2 ** 62
        def mapped(bits):
            value = struct.unpack('<f', struct.pack('<I', int(bits)))[0]
            rational = Fraction.from_float(value); assert rational > 0 and rational.denominator & (rational.denominator - 1) == 0
            return rational.numerator * pow(rational.denominator, -1, P) % P
        assert control['prime'] == P and control['new_alpha_bits'] == [0x3e800000, 0x3fc00000, 0x3f000000]
        assert control['new_alpha_mod'] == [mapped(v) for v in control['new_alpha_bits']]
        tiny = [[Fraction(1, 4), Fraction(0), Fraction(1)], [Fraction(0), Fraction(3, 2), Fraction(1)],
                [Fraction(-1, 2), Fraction(1, 2), Fraction(1)]]
        d = tiny[0][0] * (tiny[1][1] * tiny[2][2] - tiny[1][2] * tiny[2][1]) + tiny[0][2] * (-tiny[1][1] * tiny[2][0])
        assert d == 1 and control['new_rational_minor']['determinant_mod_prime'] == 1
        assert not control['duplicate_minor']['positive_real_full_rank_certificate']
        assert control['global_tie_order'] == [2, 1, 0] and control['extension_order'] == [1, 0]
        a = np.array([mapped(b) for b in alpha.view('<u4')], '<i8')
        field = np.ones((len(ids), L), '<i8'); field[:, :-1] = (a[:, None] * codes.astype('<i8')) % P
        where = np.full(17540, -1, '<i4'); where[ids] = np.arange(len(ids))
        def determinant(matrix):
            matrix = matrix.copy(); determinant = 1
            for col in range(len(matrix)):
                nonzero = np.flatnonzero(matrix[col:, col])
                if not len(nonzero): return 0
                row = col + int(nonzero[-1])
                if row != col: matrix[[col, row]] = matrix[[row, col]]; determinant = -determinant % P
                pivot = int(matrix[col, col]); determinant = determinant * pivot % P
                if col + 1 < len(matrix):
                    factors = matrix[col + 1:, col] * pow(pivot, -1, P) % P
                    matrix[col + 1:, col + 1:] = (matrix[col + 1:, col + 1:] - factors[:, None] * matrix[col, None, col + 1:]) % P
                    matrix[col + 1:, col] = 0
                if col % 8 == 0: ctx.guard()
            return determinant
        def minor_check(pos, cert):
            matrix = field[pos]
            assert cert['dimension'] == L and hashlib.sha256(matrix.tobytes()).hexdigest() == cert['field_SHA256']
            value = determinant(matrix)
            assert value == cert['determinant_mod_prime'] and bool(value) == cert['positive_real_full_rank_certificate']
            return bool(value)
        t = np.linalg.cholesky(kreg)
        solution = np.empty((L, len(h)), '<f8')
        for i in range(L):
            solution[i] = (h[:, i] - t[i, :i] @ solution[:i]) / t[i, i]
            if i % 32 == 0: ctx.guard()
        error = t @ solution - h.T; bound = 5e-12 * np.maximum(1, np.abs(t) @ np.abs(solution) + np.abs(h.T))
        assert np.max(np.abs(error) / bound) <= 1; z = solution.T
        ctx.r['gates']['dev_only_wire_roles_NEW_Fraction_controls_and_independent_forward_whitening'] = True
        def numeric(pos, recorded, stored):
            matrix = z[pos].T; Q, R = np.linalg.qr(matrix)
            err = Q @ R - matrix; bound = 5e-12 * np.maximum(1, np.abs(Q) @ np.abs(R) + np.abs(matrix))
            assert np.max(np.abs(err) / bound) <= 1 and np.max(np.abs(Q.T @ Q - np.eye(L))) <= 5e-11
            values = np.linalg.svd(R, compute_uv=False)
            assert np.isfinite(values).all() and np.all(values > 0)
            assert np.all(np.abs(values - stored) <= 1e-9 * np.maximum(1, np.abs(stored)))
            condition = float(values[0] / values[-1])
            assert abs(condition - recorded['condition2']) <= 3e-8 * max(1, condition)
            assert abs(float(values[0]) - recorded['sigma_max']) <= 1e-9 * max(1, float(values[0]))
            assert abs(float(values[-1]) - recorded['sigma_min']) <= 1e-9 * max(1, float(values[-1]))
            return Q, R, condition
        def selection(candidate_z, candidate_ids, chosen_ids, initial, Q, R, inv, maximum):
            coefficients = candidate_z @ Q
            tail = np.cumsum((coefficients * coefficients)[:, ::-1], axis=1)[:, ::-1]
            available = ~np.isin(candidate_ids, initial)
            lookup = {int(uid): i for i, uid in enumerate(candidate_ids)}
            norms = np.einsum('ij,ij->i', candidate_z, candidate_z)
            relative = []
            for offset, chosen in enumerate(chosen_ids):
                step = len(initial) + offset; row = lookup[int(chosen)]; assert available[row]
                best = float(np.max(tail[available, step])); value = float(tail[row, step]); tolerance = 1e-9 * max(1, best)
                assert value >= best - tolerance and abs(float(inv[step]) - value) <= tolerance and abs(float(maximum[step]) - best) <= tolerance
                assert abs(float(R[step, step] ** 2) - value) <= tolerance
                relative.append(math.sqrt(value / float(norms[row]))); available[row] = False
                if offset % 32 == 0: ctx.guard()
            return min(relative)
        global_ids = np.load(folder / 'global_UID_order.npy', allow_pickle=False)
        assert global_ids.dtype == np.dtype('<u4') and global_ids.shape == (L,) and len(np.unique(global_ids)) == L
        pos = where[global_ids]; assert np.all(pos >= 0)
        freeze = json.loads((folder / 'global_basis_freeze.json').read_bytes())
        assert freeze['UIDs'] == global_ids.tolist() and freeze['only_development'] and freeze['before_all_parent_complements_and_responses']
        assert freeze['certificate'] == raw['global_certificate'] and freeze['stability'] == raw['global_stability']
        global_positive = minor_check(pos, raw['global_certificate'])
        Q, R, global_condition = numeric(pos, raw['global_stability'], np.load(folder / 'global_singular.npy', allow_pickle=False))
        global_relative = selection(z, ids, global_ids, np.array([], '<u4'), Q, R,
                                    np.load(folder / 'global_innovation.npy', allow_pickle=False), np.load(folder / 'global_greedy_maximum.npy', allow_pickle=False))
        assert abs(global_relative - raw['global_minimum_relative_innovation']) <= 1e-9
        augmented = np.load(folder / 'augmented_UID_order.npy', allow_pickle=False)
        inv = np.load(folder / 'parent_new_innovation.npy', allow_pickle=False)
        maxima = np.load(folder / 'parent_greedy_maximum.npy', allow_pickle=False)
        spectra = np.load(folder / 'parent_singular.npy', allow_pickle=False)
        assert augmented.shape == inv.shape == maxima.shape == spectra.shape == (128, L)
        assert augmented.dtype == np.dtype('<u4') and inv.dtype == maxima.dtype == spectra.dtype == np.dtype('<f8')
        certificates = json.loads((folder / 'complement_certificates.json').read_bytes())
        assert certificates['cases'] == raw['cases'] and certificates['global_certificate'] == raw['global_certificate'] and certificates['prime'] == P
        candidate_ids = np.sort(global_ids); candidate_z = z[where[candidate_ids]]
        conditions = []; relatives = []; positives = []; pairs = []
        for e, case in enumerate(raw['cases']):
            own = ids[m[ids, 3] == e]; count = len(own); aug = augmented[e]; added = aug[count:]
            assert case['parent'] == e and case['original_development_UIDs'] == count and case['new_pairs'] == L - count
            assert aug[:count].tobytes() == own.astype('<u4').tobytes() and len(np.unique(aug)) == L and np.all(where[aug] >= 0)
            assert np.all(np.isin(added, candidate_ids)) and not np.isin(added, own).any()
            assert np.all(inv[e, :count] == -1) and np.all(maxima[e, :count] == -1)
            positives.append(minor_check(where[aug], case['certificate']))
            Q, R, condition = numeric(where[aug], case['stability'], spectra[e])
            relative = selection(candidate_z, candidate_ids, added, own, Q, R, inv[e], maxima[e])
            assert abs(relative - case['minimum_new_relative_innovation']) <= 1e-9
            conditions.append(condition); relatives.append(relative)
            pairs.extend((e, int(m[k, 3]), int(k)) for k in added)
            if e % 8 == 7: print(json.dumps({'audit_complement_terminal': e, 'seconds': ctx.resources()['seconds']}), flush=True)
            ctx.guard()
        dtype = np.dtype([('expert', '<u2'), ('input_owner', '<u2'), ('UID', '<u4')])
        rebuilt = np.array(pairs, dtype=dtype); recorded = np.load(folder / 'query_pairs.npy', allow_pickle=False)
        assert rebuilt.dtype == recorded.dtype and rebuilt.tobytes() == recorded.tobytes() and len(pairs) == 53943
        assert np.all(rebuilt['expert'] != rebuilt['input_owner'])
        provenance = np.load(folder / 'development_query_sources.npy', allow_pickle=False)
        assert provenance.dtype == np.dtype([('UID', '<u4'), ('input_code_pair_SHA', 'u1', (96,))])
        assert provenance['UID'].tobytes() == ids.astype('<u4').tobytes() and provenance['input_code_pair_SHA'].tobytes() == uid['hash'][ids, :96].tobytes()
        eligibility = {'pooled_full513_real_rank_positive_certificate': global_positive,
                       'ALL128_augmented_full513_real_rank_positive_certificates': all(positives),
                       'global_AND_ALL128_condition2_le1e6': global_condition <= 1e6 and max(conditions) <= 1e6,
                       'global_AND_ALL128_new_relative_innovation_ge1e-6': global_relative >= 1e-6 and min(relatives) >= 1e-6}
        assert eligibility == raw['eligibility']
        decision = 'ELIGIBLE_FOR_SEPARATELY_FROZEN_SOURCE_RESPONSES' if all(eligibility.values()) else 'COMPLEMENT_PLAN_RANK_OR_NUMERICAL_ELIGIBILITY_FAIL'
        assert decision == raw['decision'] and raw['source_response_function_calls'] == raw['readout_coefficient_solves'] == 0
        assert raw['selected_pairs'] == len(pairs) and raw['prospective_source_function_MACs'] == len(pairs) * 4718592 and raw['prospective_F32_targets_bytes'] == len(pairs) * 3072
        ctx.r['gates'].update(ALL129_independent_last_pivot_Fraction_positive_or_inconclusive_full_minor_checks=True,
                             ALL_global_and_parent_full_QR_tail_max_innovation_and_R_spectra_checks=True,
                             ALL53943_counterfactual_dev_query_pairs_original_rows_and_input_hash_provenance=True,
                             ALL_rank_stability_frozen_decisions_and_zero_source_response_calls=True)
        ctx.finish({'views': raw['views'], 'independent_numeric_views': {'global_condition2': global_condition, 'maximum_parent_condition2': max(conditions),
                    'minimum_relative_innovation': min(global_relative, *relatives)}, 'eligibility': eligibility, 'decision': decision,
                    'main_sha256': args.main_sha, 'full_minors_checked': 129, 'selected_query_pairs_audited': len(pairs),
                    'source_response_function_calls': 0, 'readout_coefficient_solves': 0, 'physical_DRAM_verified': False,
                    'scope': 'Independent development geometry/full minors/tail-innovation stability/complete query manifest; numerical agreement within frozen envelopes, no functions, quality/rate or goal completion.'})
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
