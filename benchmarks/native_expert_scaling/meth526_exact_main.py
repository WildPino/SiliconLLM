"""Exact certificate of SAME represented Q; no525 geometry/query replay."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
from fractions import Fraction
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth526_operations import Context, write


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--binding-sha', required=True); args = ap.parse_args(); ctx = None
    try:
        ctx = Context('main', args.binding_sha); np = ctx.numpy()
        import meth526_contract as S
        import meth526_exact_math as M
        # Different finite dyadic control: Q=(1/2,1/2), C=(1,1).
        tq = np.array([[1], [1]], dtype=object); tc = np.array([[1, 1]], dtype=object)
        tg = tq.T @ tq - 4 * np.eye(1, dtype=object); td = tc @ tq; te = 4 * tc - td @ tq.T
        assert tg.tolist() == [[-2]] and td.tolist() == [[2]] and te.tolist() == [[2, 2]]
        bound, nr = M.norm_upper(te, 2); assert bound == Fraction(3, 4) and nr['square_sum'] == '8'
        assert M.upward(Fraction(1, 3)) >= 1 / 3
        samples = [-(1 << 511), -1, 0, 1, (1 << 511) - 1]
        assert all(int.from_bytes(v.to_bytes(64, 'little', signed=True), 'little', signed=True) == v for v in samples)
        write(ctx.out / 'controls.json', dict(Q_integer=tq.tolist(), C_integer=tc.tolist(), Q_power=1, C_power=0,
            gram_defect=tg.tolist(), CQ=td.tolist(), rowspace_residual=te.tolist(), norm_upper='3/4',
            norm_square_sum='8', I512_roundtrips=True, upward_one_third=M.upward(Fraction(1, 3))))
        ctx.r['gates']['NEW_dyadic_Gram_residual_integer_norm_upward_and_I512_controls'] = True
        router = np.frombuffer(S.organ_bytes(ctx, 'router'), '<f4').reshape(128, 768)
        norm = np.frombuffer(S.organ_bytes(ctx, 'norm'), '<f4')
        q = np.load(ctx.data('basis'), allow_pickle=False)
        assert q.shape == (768, 128) and q.dtype == np.dtype('<f8') and np.isfinite(q).all()
        assert np.isfinite(router).all() and np.isfinite(norm).all() and np.all(norm != 0)
        c = router.astype('<f8') * norm.astype('<f8')
        legacy = ctx.b['legacy_projection']; assert legacy['rank'] == 128 and legacy['kept_rows'] == list(range(128))
        ctx.r['gates']['SAME_retained_Q_actual_original_C_and_admitted_singular_lower_contract'] = True
        a, qp, qb = M.common_units(q); b, cp, cb = M.common_units(c)
        assert qp <= 128 and cp <= 160 and qb <= 127 and cb <= 127
        encoding = dict(Q=dict(power=qp, maximum_operand_bits=qb), C=dict(power=cp, maximum_operand_bits=cb),
                        wire_signed_bits=512, wire_integer_bytes=64, no_pickle=True)
        ctx.r['gates']['exact_F64_ratio_common_dyadic_units_and_prospective_operand_wire_limits'] = True
        gram = a.T @ a - (1 << (2 * qp)) * np.eye(128, dtype=object); ctx.guard()
        cq = b @ a; ctx.guard()
        residue = (1 << (2 * qp)) * b - cq @ a.T; ctx.guard()
        for name, array, power in (('Q_numerators', a, qp), ('C_numerators', b, cp),
            ('Q_gram_defect', gram, 2 * qp), ('CQ_numerators', cq, cp + qp),
            ('C_rowspace_residual', residue, cp + 2 * qp)):
            S.write_matrix(ctx.out / (name + '.bin'), name, array, power); ctx.guard()
        ctx.r['gates']['ALL_three_exact_integer_products_and_fixed_I512_certificate_arrays_retained'] = True
        report, norms = M.certificate(gram, residue, b, qp, cp, legacy['minimum_singular_lower'])
        write(ctx.out / 'exact_certificate.json', dict(encoding=encoding, exact_certificate=report, exact_norms=norms,
            retained_Q_changed=False, legacy_geometry_replayed=False, source_function_labels_acquired=0))
        ctx.r['gates']['exact_integer_square_sums_ceil_roots_rational_projection_inequality_and_upward_F64'] = True
        eligible = dict(exact_finite_dyadic_encoding_and_ALL_integer_identities_verified=True,
            original_positive_singular_lower_and_orthogonality_upper_lt_1=report['orthogonality_upper'] < 1,
            SAME_representation_projector_bound_strictly_below_legacy=report['projector_error_bound'] < legacy['projector_error_bound'])
        decision = 'EXACT_PROJECTOR_CERTIFICATE_ELIGIBLE_FOR_BOUND_REFINEMENT_NOT_QUERY_PROMOTION' if all(eligible.values()) else 'EXACT_PROJECTOR_CERTIFICATE_NO_STRICT_BOUND_IMPROVEMENT'
        ctx.r['gates']['unchanged_representation_frozen_strict_bound_improvement_and_zero_query_decision'] = True
        ctx.finish(dict(encoding=encoding, legacy_projection=legacy, exact_certificate=report, exact_norms=norms,
            exact_integer_product_terms=3 * 128 * 128 * 768, exact_norm_square_terms=16384 + 98304 + 98304,
            eligibility=eligible, decision=decision, views=[], constructed_queries=0,
            new_router_score_vectors=0, selected_source_integer_dot_checks=0,
            source_response_function_calls=0, native_calls=0, model_calls=0, readout_fits=0,
            full_source_WI_projection_rows=0, old_native_router_queries_replayed=0, consumed_rows_for_selection=0,
            old525_geometry_or_queries_recomputed=0, physical_DRAM_verified=False,
            scope='Exact represented-projector certificate only; no bound-refined plane ordering, new query, source label, natural quality/rate, useful n or whole transfer claim.'))
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
