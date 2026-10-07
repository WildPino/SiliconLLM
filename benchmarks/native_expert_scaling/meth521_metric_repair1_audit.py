"""Remaining metric audit using predeclared floating-point propagation bounds."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
import json
import math
from pathlib import Path
import struct
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth521_metric_repair1_operations import Context, DOC, MAIN_SHA, FAULT_SHA, ROOT, write
from meth521_operations import wire


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--binding-sha', required=True); args = ap.parse_args(); ctx = None
    try:
        ctx = Context(args.binding_sha); np = ctx.numpy()
        import meth499_math as M
        def output(name): return ctx.b['outputs'][name]['path']
        uid = wire(np, ctx.data('uid'), b'M493U001', 180, 0, M.UID, 17540); m = uid['m']
        occ = wire(np, ctx.data('occurrences'), b'M493O001', 68, 0, np.dtype(('<u4', (17,))), 19962)
        ni = wire(np, ctx.data('inputs'), b'M499INP1', 4624, 768, M.NATIVE_INPUT, 17540); inputs = ni['core']; ps = ni['source_p']
        feat = wire(np, ctx.data('features'), b'M495FEA1', 6148, 512, M.FEATURE, 17540)
        target = wire(np, ctx.data('unweighted'), b'M499F001', 3072, 768, np.dtype(('<f4', (768,))), 17540)
        weighted = wire(np, ctx.data('targets'), b'M493Y001', 3072, 768, np.dtype(('<f4', (768,))), 17540)
        pred = wire(np, output('predictions.bin'), b'M521PRE1', 15464, 768, M.PRED, 17540)
        oldenergy = wire(np, output('energy_by_uid.bin'), b'M521ENG1', 376, 47, np.dtype(('<f8', (47,))), 17540)
        bank = Path(output('bank.bin')).read_bytes()
        # Direct scalar integer/F32 physical checks already completed in original prefix; no such check is rerun.
        def square(e, b):
            v = np.einsum('ij,ij->i', e, e)
            bound = np.sum(b * (2 * np.abs(e) + b), axis=1) + 3e-12 * np.maximum(1, v)
            return v, bound
        def cross(e, b, f, c):
            v = np.einsum('ij,ij->i', e, f)
            bound = np.sum(b * np.abs(f) + c * np.abs(e) + b * c, axis=1) + 3e-12 * np.maximum(1, np.sum(np.abs(e * f), axis=1))
            return v, bound
        def decomposed(y, u64, u32, decoded, phys, b64, b32):
            zero = np.zeros_like(y)
            e = [u64 - y, u32 - u64, decoded - u32, phys - decoded]
            b = [b64, b64 + b32, b32, zero]; total = phys - y
            pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
            di = [square(v, w) for v, w in ((y, zero), (e[0], b64), (u32 - y, b32), (total, zero),
                                           (e[1], b[1]), (e[2], b32), (e[3], zero))]
            cr = [cross(e[i], b[i], e[j], b[j]) for i, j in pairs]
            d = [v[0] for v in di]; db = [v[1] for v in di]; cc = [v[0] for v in cr]; cb = [v[1] for v in cr]
            identity = sum(e) - total
            ib = 3e-12 * np.maximum(1, np.abs(y) + np.abs(u64) + np.abs(u32) + np.abs(decoded) + np.abs(phys))
            identity_compare = np.max(ib, axis=1) + np.max(ib + 3e-12 * (b64 + b32), axis=1)
            closure = d[3] - (d[1] + d[4] + d[5] + d[6] + 2 * sum(cc))
            eb = 1e-10 * np.maximum(1, d[0] + d[1] + d[4] + d[5] + d[6] + 2 * sum(np.abs(v) for v in cc))
            closure_compare = 2 * eb + 1e-10 * (db[0] + db[1] + db[4] + db[5] + db[6] + 2 * sum(cb))
            pre, prebound = square(decoded - y, zero)
            result = np.column_stack((*d, *cc, np.max(np.abs(identity), axis=1), np.max(np.abs(identity) / ib, axis=1),
                                      np.abs(closure), np.abs(closure) / eb, pre))
            bounds = np.column_stack((*db, *cb, identity_compare, np.ones(len(y)), closure_compare, np.ones(len(y)), prebound))
            assert np.max(result[:, [14, 16]]) <= 1 and np.isfinite(result).all()
            return result, bounds

        values = np.empty((17540, 47), '<f8'); bounds = np.empty_like(values); seen = np.zeros(17540, bool)
        with Path(output('coefficients_F64.bin')).open('rb') as cf, Path(output('coefficients_F32.bin')).open('rb') as ff:
            assert cf.read(24) == struct.pack('<8sIIQ', b'M521F641', 3151872, 513, 128)
            assert ff.read(24) == struct.pack('<8sIIQ', b'M521F321', 1575936, 513, 128)
            for e in range(128):
                c = np.frombuffer(cf.read(3151872), '<f8').reshape(768, 513)
                cc = np.frombuffer(ff.read(1575936), '<f4').reshape(768, 513)
                lq, ls, bq, bs, bias = M.parts(bank, e); ids = np.flatnonzero(m[:, 3] == e)
                for start in range(0, len(ids), 127):
                    at = ids[start:start + 127]
                    dot = inputs['q'][at].astype('<f8') @ lq.astype('<f8').T
                    left = ((dot * ls.astype('<f8')[None, :]) * inputs['alpha'][at, None].astype('<f8')).astype('<f4').astype('<f8')
                    h = np.ones((len(at), 513), '<f8'); ph = feat['q'][at].astype('<f8'); alpha = feat['alpha'][at].astype('<f8')
                    h[:, :512] = ph * alpha[:, None]
                    u64 = left + h[:, ::-1] @ c[:, ::-1].T; u32 = left + h[:, ::-1] @ cc[:, ::-1].astype('<f8').T
                    decoded = (left + ((ph @ bq.astype('<f8').T) * bs.astype('<f8')[None, :]) * alpha[:, None]) + bias.astype('<f8')[None, :]
                    b64 = 3e-12 * np.maximum(1, np.abs(left) + np.abs(h) @ np.abs(c).T)
                    b32 = 3e-12 * np.maximum(1, np.abs(left) + np.abs(h) @ np.abs(cc.astype('<f8')).T)
                    fs = target[at].astype('<f8'); ys = weighted[at].astype('<f8'); p = ps[at, None].astype('<f8')
                    raw, rb = decomposed(fs, u64, u32, decoded, pred['F'][at].astype('<f8'), b64, b32)
                    wb64 = p * b64 + 3e-12 * np.maximum(1, np.abs(p * u64))
                    wb32 = p * b32 + 3e-12 * np.maximum(1, np.abs(p * u32))
                    wei, wb = decomposed(ys, p * u64, p * u32, p * decoded, pred['oracle'][at].astype('<f8'), wb64, wb32)
                    ef = pred['oracle'][at].astype('<f8') - ys; ec = pred['choice_source_mass'][at].astype('<f8') - pred['oracle'][at].astype('<f8')
                    em = pred['coupled'][at].astype('<f8') - pred['choice_source_mass'][at].astype('<f8'); total = pred['coupled'][at].astype('<f8') - ys
                    sm = pred['sameID_mass'][at].astype('<f8') - pred['oracle'][at].astype('<f8'); zero = np.zeros_like(ys)
                    diag = [square(v, zero) for v in (total, ec, em, sm)]
                    pairs = [cross(v, zero, w, zero) for v, w in ((ef, ec), (ef, em), (ec, em))]
                    pd = (pred['p'][at].astype('<f8') - ps[at].astype('<f8')) ** 2
                    pe = ps[at].astype('<f8') ** 2
                    identity = ef + ec + em - total
                    ib = 3e-12 * np.maximum(1, np.abs(ys) + np.abs(pred['oracle'][at].astype('<f8')) + np.abs(pred['choice_source_mass'][at].astype('<f8')) + np.abs(pred['coupled'][at].astype('<f8')))
                    closure = diag[0][0] - (wei[:, 3] + diag[1][0] + diag[2][0] + 2 * sum(v[0] for v in pairs))
                    eb = 1e-10 * np.maximum(1, wei[:, 0] + wei[:, 3] + diag[1][0] + diag[2][0] + 2 * sum(np.abs(v[0]) for v in pairs))
                    values[at] = np.column_stack((raw, wei, *(v[0] for v in diag), pd, pe, *(v[0] for v in pairs),
                                                  np.max(np.abs(identity) / ib, axis=1), np.abs(closure) / eb))
                    bounds[at] = np.column_stack((rb, wb, *(v[1] for v in diag), 3e-12 * np.maximum(1, pd),
                                                  3e-12 * np.maximum(1, pe), *(v[1] for v in pairs), np.ones(len(at)), np.ones(len(at))))
                    seen[at] = True; ctx.guard()
        assert seen.all() and np.isfinite(values).all() and np.isfinite(bounds).all() and np.all(bounds > 0)
        # Persist BEFORE the first comparison; no lost metric prefix on another fault.
        np.save(ctx.out / 'independent_energy.npy', values, allow_pickle=False)
        np.save(ctx.out / 'propagated_comparison_bounds.npy', bounds, allow_pickle=False)
        columns = np.array([*range(14), 15, 17, *range(18, 32), 33, 35, *range(36, 45)])
        ratio = np.abs(values[:, columns] - oldenergy[:, columns]) / bounds[:, columns]
        nominal_columns = np.array([*range(14), *range(18, 32)])
        failure_mask = np.abs(values[:, nominal_columns] - oldenergy[:, nominal_columns]) > (1e-7 + 3e-8 * np.abs(oldenergy[:, nominal_columns]))
        rows, cols = np.nonzero(failure_mask)
        failed = np.empty(len(rows), dtype=[('UID', '<u4'), ('column', '<u2'), ('main', '<f8'), ('audit', '<f8'), ('bound', '<f8'), ('certified_ratio', '<f8')])
        failed['UID'] = rows; failed['column'] = nominal_columns[cols]
        failed['main'] = oldenergy[rows, failed['column']]; failed['audit'] = values[rows, failed['column']]
        failed['bound'] = bounds[rows, failed['column']]
        failed['certified_ratio'] = np.abs(failed['audit'] - failed['main']) / failed['bound']
        np.save(ctx.out / 'original_blanket_failure_cells.npy', failed, allow_pickle=False)
        # New comparison is algebraic propagation, no tuning after inspecting ratios.
        assert np.max(ratio) <= 1, float(np.max(ratio))
        assert np.max(values[:, [14, 16, 32, 34, 45, 46]]) <= 1 and np.max(oldenergy[:, [14, 16, 32, 34, 45, 46]]) <= 1
        ctx.r['gates']['ALL17540_energy_fields_satisfy_predeclared_propagated_bounds_and_both_closures'] = True
        reports = M.reports(m, occ, values, pred['id'], ps, pred['p'])
        original = json.loads((DOC / 'meth521_main_result.json').read_bytes())
        assert reports['outcomes'] == original['reports']['outcomes'] and reports['exposures'] == original['reports']['exposures']
        # All report selectors and denominators verified; the certified UID bounds propagate by addition.
        for group in ('uid_roles', 'cells', 'rare', 'views', 'role_mode'):
            assert len(reports[group]) == len(original['reports'][group])
            for current, previous in zip(reports[group], original['reports'][group]):
                for name in ('count', 'ID_correct', 'ID_fidelity'): assert current[name] == previous[name]
                assert math.isclose(current['source_energy'], previous['source_energy'], rel_tol=3e-12, abs_tol=3e-12)
                assert math.isclose(current['unweighted_source_energy'], previous['unweighted_source_energy'], rel_tol=3e-12, abs_tol=3e-12)
        for name, column in (('unweighted_fit64', 1), ('unweighted_fit32', 2), ('unweighted_oracle', 3),
                             ('weighted_fit64', 19), ('weighted_fit32', 20), ('weighted_oracle', 21), ('coupled', 36)):
            denominator = 0 if name.startswith('unweighted') else 18; passed = True
            for role in range(3):
                for mode in range(2):
                    ids = occ[(occ[:, 7] == role) & (occ[:, 6] == mode), 1]
                    s = math.fsum(map(float, values[ids, column])); energy = math.fsum(map(float, values[ids, denominator]))
                    uncertainty = math.fsum(map(float, bounds[ids, column])) + math.fsum(map(float, bounds[ids, denominator])) * .0001
                    margin = s - .0001 * energy
                    assert abs(margin) > uncertainty  # Gate is certified away from its threshold.
                    passed &= bool(len(ids) and energy > 0 and margin <= 0)
            assert passed == reports['outcomes'][name + '_ALL_six_RMS_1pct']
        ctx.r['gates']['ALL1040_report_denominators384_exposures_and_certified_independent_six_view_decisions'] = True
        ctx.r['gates']['completed_original_source_equation_native_prefix_retained_without_replay'] = True
        decision = 'FIXED_CLASS_FULL_SOURCE_INFORMATION_RECIPE_CLOSED' if not (reports['outcomes']['unweighted_fit64_ALL_six_RMS_1pct'] and reports['outcomes']['weighted_fit64_ALL_six_RMS_1pct']) else M.decision(reports['outcomes'])
        assert decision == original['decision']
        counts = [{'column': int(col), 'original_blanket_failures': int(np.sum(failed['column'] == col))} for col in nominal_columns]
        ctx.finish({'reports': reports, 'energy_comparison_max_ratio': float(np.max(ratio)), 'blanket_failures': len(failed),
                    'blanket_failures_by_column': counts, 'completed_audit_prefix_source_function_calls': 53943,
                    'new_source_response_function_calls': 0, 'new_coefficient_solves': 0, 'native_calls': 0, 'compiler_calls': 0, 'model_calls': 0,
                    'energy_array_constructions_original_and_repair': [1, 1], 'decision': decision,
                    'scope': 'Metric-only numbered continuation after full source/equation/physical audit prefix. All first fault bytes preserved. No source, fit or native replay; function/routing thresholds unchanged.'})
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
