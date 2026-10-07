"""Independent IEEE decoding and scalar integer verification of ALL certificate cells."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import struct
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth526_operations import Context, DOC, ROOT, write


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--binding-sha', required=True); ap.add_argument('--main-sha', required=True)
    args = ap.parse_args(); ctx = None
    try:
        ctx = Context('audit', args.binding_sha); rp = DOC / 'meth526_main_result.json'
        assert ctx.digest(rp) == args.main_sha
        raw = json.loads(rp.read_bytes()); assert raw['binding_sha256'] == args.binding_sha and all(raw['gates'].values())
        folder = ROOT / 'results/native_expert_scaling/meth526_exact_projector'
        for v in raw['output_inventory']: assert Path(v['path']).stat().st_size == v['bytes'] and ctx.digest(v['path']) == v['sha256']
        assert json.loads((folder / 'terminal_resource.json').read_bytes())['result_sha256'] == args.main_sha
        ctx.r['gates']['ALL_main_fixed_integer_arrays_and_terminal_SHA_before_independent_exact_verification'] = True
        np = ctx.numpy(); import meth526_contract as S
        ctl = json.loads((folder / 'controls.json').read_bytes())
        assert ctl['Q_integer'] == [[1], [1]] and ctl['C_integer'] == [[1, 1]]
        assert ctl['Q_power'] == 1 and ctl['C_power'] == 0
        assert ctl['gram_defect'] == [[1 * 1 + 1 * 1 - 4]] == [[-2]]
        assert ctl['CQ'] == [[2]] and ctl['rowspace_residual'] == [[4 - 2, 4 - 2]] == [[2, 2]]
        assert ctl['norm_square_sum'] == '8' and Fraction(ctl['norm_upper']) == Fraction(3, 4)
        assert 3 ** 2 >= 8 and 2 ** 2 < 8 and ctl['I512_roundtrips']
        assert Fraction.from_float(ctl['upward_one_third']) >= Fraction(1, 3)
        assert Fraction.from_float(math.nextafter(ctl['upward_one_third'], -math.inf)) < Fraction(1, 3)
        ctx.r['gates']['independent_exact_toy_Gram_residual_integer_root_and_directed_conversion_controls'] = True
        q = np.load(ctx.data('basis'), allow_pickle=False)
        assert q.shape == (768, 128) and q.dtype == np.dtype('<f8') and np.isfinite(q).all()
        legacy = ctx.b['legacy_projection']; assert legacy == raw['legacy_projection']
        assert legacy['rank'] == 128 and legacy['kept_rows'] == list(range(128)) and legacy['minimum_singular_lower'] > 0
        rwords = struct.unpack('<98304I', S.organ_bytes(ctx, 'router'))
        gwords = struct.unpack('<768I', S.organ_bytes(ctx, 'norm'))
        qwords = struct.unpack('<98304Q', q.tobytes())
        ctx.r['gates']['SAME_original_router_norm_Q_bytes_and_reused_qualified_singular_lower'] = True
        def normalize(num, shift):
            if num == 0: return 0, 0
            tz = (abs(num) & -abs(num)).bit_length() - 1
            num //= 1 << tz; shift += tz
            return (num << shift, 0) if shift >= 0 else (num, -shift)
        def decode(word, bits, fractional, bias):
            expmask = (1 << (bits - fractional - 1)) - 1
            exponent = (word >> fractional) & expmask; assert exponent != expmask
            mantissa = word & ((1 << fractional) - 1)
            if exponent: mantissa |= 1 << fractional
            if word >> (bits - 1): mantissa = -mantissa
            shift = exponent - bias - fractional if exponent else 1 - bias - fractional
            return normalize(mantissa, shift)
        gpairs = [decode(w, 32, 23, 127) for w in gwords]; assert all(n != 0 for n, p in gpairs)
        qpairs = [decode(w, 64, 52, 1023) for w in qwords]
        cpairs = []
        for i, w in enumerate(rwords):
            rn, re = decode(w, 32, 23, 127); gn, ge = gpairs[i % 768]
            cpairs.append(normalize(rn * gn, -(re + ge)))
        qp = max(p for n, p in qpairs); cp = max(p for n, p in cpairs)
        avec = [n << (qp - p) for n, p in qpairs]; bvec = [n << (cp - p) for n, p in cpairs]
        qb = max(abs(v).bit_length() for v in avec); cb = max(abs(v).bit_length() for v in bvec)
        assert qp <= 128 and cp <= 160 and qb <= 127 and cb <= 127
        encoding = dict(Q=dict(power=qp, maximum_operand_bits=qb), C=dict(power=cp, maximum_operand_bits=cb),
                        wire_signed_bits=512, wire_integer_bytes=64, no_pickle=True)
        assert encoding == raw['encoding']
        a = S.read_matrix(folder / 'Q_numerators.bin', 'Q_numerators', 768, 128, qp)
        b = S.read_matrix(folder / 'C_numerators.bin', 'C_numerators', 128, 768, cp)
        assert [v for row in a for v in row] == avec and [v for row in b for v in row] == bvec
        ctx.r['gates']['independent_F32_F64_IEEE_bit_decoding_ALL_exact_common_units_and_fixed_wire_limits'] = True
        gram = S.read_matrix(folder / 'Q_gram_defect.bin', 'Q_gram_defect', 128, 128, 2 * qp)
        cq = S.read_matrix(folder / 'CQ_numerators.bin', 'CQ_numerators', 128, 128, cp + qp)
        residue = S.read_matrix(folder / 'C_rowspace_residual.bin', 'C_rowspace_residual', 128, 768, cp + 2 * qp)
        columns = list(zip(*a)); verified = 0
        for i in range(128):
            for j in range(128):
                expected = sum(x * y for x, y in zip(columns[i], columns[j])) - ((1 << (2 * qp)) if i == j else 0)
                assert expected == gram[i][j]
                assert sum(x * y for x, y in zip(b[i], columns[j])) == cq[i][j]
                verified += 2 * 768
            ctx.guard()
            if i % 32 == 31: print('audited_exact_Gram_CQ_rows=' + str(i + 1), flush=True)
        for i in range(128):
            for j in range(768):
                expected = (b[i][j] << (2 * qp)) - sum(x * y for x, y in zip(cq[i], a[j]))
                assert expected == residue[i][j]; verified += 128
            ctx.guard()
            if i % 32 == 31: print('audited_exact_residual_rows=' + str(i + 1), flush=True)
        assert verified == raw['exact_integer_product_terms'] == 37748736
        ctx.r['gates']['ALL_Gram_CQ_and_rowspace_residual_integer_cells_scalar_exactly_verified'] = True
        values = {}
        for key, array, power in (('orthogonality', gram, 2 * qp), ('rowspace_residual', residue, cp + 2 * qp), ('source_Frobenius', b, cp)):
            square = sum(sum(v * v for v in row) for row in array); rec = raw['exact_norms'][key]
            root = int(rec['ceil_integer_root'])
            assert int(rec['square_sum']) == square and rec['denominator_power'] == power
            assert root * root >= square and (root == 0 or (root - 1) ** 2 < square)
            bound = Fraction(root, 1 << power)
            assert [str(bound.numerator), str(bound.denominator)] == rec['upper_rational']; values[key] = bound
        oq, rb, cn = values['orthogonality'], values['rowspace_residual'], values['source_Frobenius']
        assert oq < 1; sl = Fraction.from_float(legacy['minimum_singular_lower'])
        exact_upper = 4 * (rb + cn * oq / (1 - oq)) / sl + 4 * oq / (1 - oq)
        report = raw['exact_certificate']
        assert report['reused_source_singular_lower'] == legacy['minimum_singular_lower']
        assert report['projector_upper_rational'] == [str(exact_upper.numerator), str(exact_upper.denominator)]
        for key, value in (('orthogonality_upper', oq), ('rowspace_residual_upper', rb), ('source_Frobenius_upper', cn), ('projector_error_bound', exact_upper)):
            f = report[key]; assert math.isfinite(f) and Fraction.from_float(f) >= value
            assert Fraction.from_float(math.nextafter(f, -math.inf)) < value
        assert raw['exact_norm_square_terms'] == 212992
        ctx.r['gates']['ALL_integer_square_sums_ceil_roots_exact_rational_bound_and_least_upward_F64_independent'] = True
        eligibility = dict(exact_finite_dyadic_encoding_and_ALL_integer_identities_verified=True,
            original_positive_singular_lower_and_orthogonality_upper_lt_1=report['orthogonality_upper'] < 1,
            SAME_representation_projector_bound_strictly_below_legacy=report['projector_error_bound'] < legacy['projector_error_bound'])
        decision = 'EXACT_PROJECTOR_CERTIFICATE_ELIGIBLE_FOR_BOUND_REFINEMENT_NOT_QUERY_PROMOTION' if all(eligibility.values()) else 'EXACT_PROJECTOR_CERTIFICATE_NO_STRICT_BOUND_IMPROVEMENT'
        assert raw['eligibility'] == eligibility and raw['decision'] == decision and len(raw['gates']) == 7
        ctx.r['gates']['unchanged_representation_and_fixed_bound_improvement_with_zero_query_decision'] = True
        write(ctx.out / 'verified_exact_certificate.json', dict(main_sha256=args.main_sha, verified_integer_product_terms=verified,
            verified_certificate_integer_cells=16384 + 16384 + 98304, exact_certificate=report))
        ctx.finish(dict(main_sha256=args.main_sha, encoding=encoding, legacy_projection=legacy, exact_certificate=report, exact_norms=raw['exact_norms'],
            exact_integer_product_terms=verified, verified_certificate_integer_cells=131072, exact_norm_square_terms=212992,
            eligibility=eligibility, decision=decision, views=[], constructed_queries=0,
            new_router_score_vectors=0, selected_source_integer_dot_checks=0,
            source_response_function_calls=0, native_calls=0, model_calls=0, readout_fits=0,
            full_source_WI_projection_rows=0, old_native_router_queries_replayed=0, consumed_rows_for_selection=0,
            old525_geometry_or_queries_recomputed=0, physical_DRAM_verified=False,
            scope='Independent exact represented-projector identities only; no525 query replay, bound-refined ordering, new source labels, useful n, fresh quality/rate or whole claim.'))
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
