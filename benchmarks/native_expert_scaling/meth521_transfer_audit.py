"""Independent source reconstruction, direct full-rank equations and physical metrics."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth521_operations import Context, DOC, ROOT, wire, write


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--binding-sha', required=True); ap.add_argument('--main-sha', required=True)
    args = ap.parse_args(); ctx = None
    try:
        ctx = Context('audit', args.binding_sha)
        assert ctx.digest(DOC / 'meth521_main_result.json') == args.main_sha
        raw = json.loads((DOC / 'meth521_main_result.json').read_bytes())
        assert raw['binding_sha256'] == args.binding_sha and all(raw['gates'].values())
        folder = ROOT / 'results/native_expert_scaling/meth521_transfer'
        for v in raw['output_inventory']: assert Path(v['path']).stat().st_size == v['bytes'] and ctx.digest(v['path']) == v['sha256']
        terminal = json.loads((folder / 'terminal_resource.json').read_bytes()); assert terminal['result_sha256'] == args.main_sha
        ctx.r['gates']['complete_main_ALL_outputs_and_terminal_SHA_before_independent_arithmetic'] = True
        np = ctx.numpy()
        # Inherited report definitions and wire layouts are specifications only.
        import meth521_math as S
        M = S.M

        def check(a, x, rhs, tol):
            d = a @ x - rhs; b = tol * np.maximum(1, np.abs(a) @ np.abs(x) + np.abs(rhs))
            ratio = float(np.max(np.abs(d) / b)); assert np.isfinite(ratio) and ratio <= 1
            return ratio

        def nearest(c, limit):
            c = np.asarray(c, dtype='<f4'); mx = np.max(np.abs(c), axis=1)
            scale = (mx.astype('<f8') / limit).astype('<f4'); scale[mx == 0] = 1
            z = (c.astype('<f8') / scale[:, None].astype('<f8')).astype('<f4').astype('<f8')
            low = np.floor(z); rest = z - low
            code = np.clip(low + ((rest > .5) | ((rest == .5) & (low % 2 != 0))), -limit, limit)
            assert np.isfinite(scale).all() and np.all(scale > 0)
            return code.astype('<i2' if limit == 32767 else '<i1'), scale

        def physical(q, w, scales, alpha):
            dot = q.astype('<f8') @ w.astype('<f8').T
            # Every product and every partial sum is an integer smaller than2^53.
            bound = q.shape[1] * 128 * 32767
            assert bound < 2 ** 53 and np.array_equal(dot, dot.astype('<i8')) and np.all(np.abs(dot) <= bound)
            return ((dot * scales.astype('<f8')[None, :]) * alpha.astype('<f8')[:, None]).astype('<f4')

        ctl = json.loads((folder / 'controls.json').read_bytes())
        exact = [[Fraction(5, 2), Fraction(3, 2), Fraction(-5, 2)], [Fraction(-4), Fraction(-3), Fraction(8)]]
        hsmall = [[2, -1, 1], [0, 3, 1], [1, 0, 1]]; ysmall = [[1, 2, 0], [3, -1, 4]]
        for row, labels in zip(exact, ysmall):
            assert [sum(a * b for a, b in zip(hr, row)) for hr in hsmall] == labels
        assert abs(Fraction(2) * 3 - Fraction(1) * 3 - 1) == 2  # Exact determinant.
        assert np.max(np.abs(np.array(ctl['unique_coefficients']) - np.array([[float(v) for v in row] for row in exact]))) <= 1e-13
        tie = np.array([[32767, -32767, 2.5, 3.5, -2.5, -3.5, 0, 1]], '<f4')
        qt, at = nearest(tie, 32767); assert qt.tolist() == [[32767, -32767, 2, 4, -2, -4, 0, 1]] and at.tolist() == [1.]
        qz, az = nearest(np.zeros((1, 3072), '<f4'), 32767); assert not qz.any() and az.tolist() == [1.]
        native = [json.loads(v) for v in (folder / 'new_controls.stdout').read_text().splitlines()]
        assert native == ctl['native'] and native[0]['source_full_width_I64'] == -128 * 32767 * 3072 < -(2 ** 31)
        assert native[0]['zero_alpha'] == 1. and native[0]['codes'] == qt[0].tolist() and native[0]['alpha_bits'] == 0x3f800000
        assert native[0]['RNE'] == 0 and native[0]['CPU10'] == 1024 and not native[0]['MXCSR'] & 0x8040
        x = [32765, -19875, 301, -101]
        assert native[1] == {'new_table81': [sum(((p // (3 ** j)) % 3 - 1) * x[j] for j in range(4)) for p in range(81)]}
        pb = [2, 2, 2, 4, 0x80000002, 0x3e000003]
        assert native[2] == {'new_quarter_product_bits': pb}
        for k, out in zip((7, 9, 10, 14), pb[:4]):
            v = Fraction(k, 4); assert out - Fraction(1, 2) <= v <= out + Fraction(1, 2)
            if v in (out - Fraction(1, 2), out + Fraction(1, 2)): assert out % 2 == 0
        assert pb[4] == (0x80000000 | pb[2]) and pb[5] == 0x3f000003 - (2 << 23)
        assert 192 * 2 * 128 * 32767 < 2 ** 31 and 3072 * 128 * 32767 < 2 ** 53
        assert (folder / 'negative_magic.stderr').read_bytes().replace(b'\r\n', b'\n') == b'factor521_error:wire_magic\n'
        assert not (folder / 'negative_predictions.bin').exists()
        write(ctx.out / 'independent_controls.json', {'exact_unique_coefficients': [[str(v) for v in row] for row in exact],
              'exact_determinant': 2, 'source_AVX_lane_bound': 192 * 2 * 128 * 32767, 'quarter_bits': pb, 'native': native})
        ctx.r['gates']['independent_Fraction_unique_coefficients_RNE_LUT_fullwidth_integer_and_negative_controls'] = True
        uid, occ, inputs, ps, feat, target, y, pairs, aug, kreg = S.load(ctx); m = uid['m']
        response = wire(np, folder / 'source_responses.bin', b'M521SR01', 9236, 768, S.RESPONSE, 53943)
        source_calls = 0
        with Path(ctx.b['payload']['path']).open('rb') as f:
            def original(v, scale=False):
                f.seek(v['scale_offset'] if scale else v['offset'])
                n = v['scale_bytes'] if scale else v['bytes']; rawbytes = f.read(n); assert len(rawbytes) == n
                if scale: return np.frombuffer(rawbytes, '<f4')
                return np.frombuffer(rawbytes, '<i1').reshape(v['shape'])
            for e, parent in enumerate(ctx.b['parents']):
                wi, ws = original(parent['wi']), original(parent['wi'], True)
                wo, oscale = original(parent['wo']), original(parent['wo'], True)
                ids = np.flatnonzero(pairs['expert'] == e)
                for start in range(0, len(ids), 127):
                    at = ids[start:start + 127]; p = pairs[at]; ks = p['UID']; recorded = response[at]
                    assert np.array_equal(recorded['UID'], ks) and np.all(recorded['expert'] == e)
                    assert np.array_equal(recorded['input_owner'], p['input_owner']) and np.array_equal(inputs['e'][ks], p['input_owner'])
                    qx, ax = nearest(inputs['x'][ks], 32767)
                    assert qx.tobytes() == inputs['q'][ks].tobytes() and ax.tobytes() == inputs['alpha'][ks].tobytes()
                    hidden = physical(qx, wi, ws, ax); hidden[hidden < 0] = 0
                    qh, ah = nearest(hidden, 32767); down = physical(qh, wo, oscale, ah)
                    assert qh.tobytes() == recorded['codes'].tobytes() and ah.tobytes() == recorded['alpha'].tobytes()
                    assert np.array_equal(np.count_nonzero(qh, axis=1), recorded['nonzero']) and down.tobytes() == recorded['F'].tobytes()
                    source_calls += len(at); ctx.guard()
                if e % 8 == 7: print(json.dumps({'independent_source_parent': e, 'source_functions': source_calls, 'seconds': ctx.resources()['seconds']}), flush=True)
        assert source_calls == 53943
        ctx.r['gates']['ALL53943_independent_exact_integer_F64_BLAS_source_WI_ReLU_hidden_codes_scales_and_WO_BYTES'] = True
        bank = (folder / 'bank.bin').read_bytes()
        assert bank[:40] == struct.pack('<8s8I', b'M521BNK1', 768, 3072, 512, 128, 11, 4, 8, 16) and len(bank) == 127232136
        assert hashlib.sha256(bank[8:223368]).hexdigest() == raw['fixed_dictionary_keys_sha256']
        pred = wire(np, folder / 'predictions.bin', b'M521PRE1', 15464, 768, M.PRED, 17540)
        baseline = wire(np, ctx.data('baseline'), b'M499PRE1', 15464, 768, M.PRED, 17540)
        for field in ('id', 'p', 'logits'): assert pred[field].tobytes() == baseline[field].tobytes()
        del baseline
        # Reconstruct all fixed A16 feature bytes from the original physical trits.
        packed = np.frombuffer(bank, 'u1', 512 * 192, 40).reshape(512, 192)
        assert np.all(packed <= 80)
        a = np.empty((512, 768), '<i1')
        for j in range(4): a[:, j::4] = ((packed.astype('<i2') // 3 ** j) % 3 - 1).astype('<i1')
        sigma = np.frombuffer(bank, '<f4', 512, 40 + 512 * 192)
        for start in range(0, 17540, 193):
            ix = np.arange(start, min(start + 193, 17540))
            phi = np.abs(physical(inputs['q'][ix], a, sigma, inputs['alpha'][ix]))
            q, alpha = nearest(phi, 32767)
            assert phi.tobytes() == feat['phi'][ix].tobytes() and q.tobytes() == feat['q'][ix].tobytes() and alpha.tobytes() == feat['alpha'][ix].tobytes(); ctx.guard()

        def decomposed(f, u64, u32, decoded, phys):
            errors = [u64 - f, u32 - u64, decoded - u32, phys - decoded]; total = phys - f
            diag = [np.einsum('ij,ij->i', v, v) for v in (f, errors[0], u32 - f, total, errors[1], errors[2], errors[3])]
            cross = [np.einsum('ij,ij->i', errors[i], errors[j]) for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]
            residual = sum(errors) - total; vb = 3e-12 * np.maximum(1, np.abs(f) + np.abs(u64) + np.abs(u32) + np.abs(decoded) + np.abs(phys))
            closure = diag[3] - (diag[1] + diag[4] + diag[5] + diag[6] + 2 * sum(cross))
            eb = 1e-10 * np.maximum(1, diag[0] + diag[1] + diag[4] + diag[5] + diag[6] + 2 * sum(np.abs(v) for v in cross))
            pre = decoded - f
            out = np.column_stack((*diag, *cross, np.max(np.abs(residual), axis=1), np.max(np.abs(residual) / vb, axis=1),
                                  np.abs(closure), np.abs(closure) / eb, np.einsum('ij,ij->i', pre, pre)))
            assert np.isfinite(out).all() and np.max(out[:, [14, 16]]) <= 1
            return out

        values = np.empty((17540, 47), '<f8'); seen = np.zeros(17540, bool); cases = []
        with (folder / 'coefficients_F64.bin').open('rb') as cf, (folder / 'coefficients_F32.bin').open('rb') as ff, Path(ctx.data('bank')).open('rb') as oldbank:
            assert cf.read(24) == struct.pack('<8sIIQ', b'M521F641', 3151872, 513, 128)
            assert ff.read(24) == struct.pack('<8sIIQ', b'M521F321', 1575936, 513, 128)
            for e in range(128):
                c = np.frombuffer(cf.read(3151872), '<f8').reshape(768, 513)
                cc = np.frombuffer(ff.read(1575936), '<f4').reshape(768, 513)
                assert c.astype('<f4').tobytes() == cc.tobytes() and np.isfinite(c).all()
                pos = 223368 + 992256 * e; oldbank.seek(pos); assert oldbank.read(592896) == bank[pos:pos + 592896]
                lq, ls, bq, bs, bias = M.parts(bank, e); code, scale = nearest(cc[:, :512], 127)
                assert code.tobytes() == bq.tobytes() and scale.tobytes() == bs.tobytes() and bias.tobytes() == cc[:, 512].tobytes()
                own = np.flatnonzero(((m[:, 4] & 5) != 0) & (m[:, 3] == e)); added = np.flatnonzero(pairs['expert'] == e)
                labels = np.vstack((target[own], response['F'][added])).astype('<f8'); ids = aug[e]
                assert np.array_equal(ids, np.concatenate((own, pairs['UID'][added])))
                h = np.ones((513, 513), '<f8'); h[:, :512] = feat['q'][ids].astype('<f8') * feat['alpha'][ids, None].astype('<f8')
                left = physical(inputs['q'][ids], lq, ls, inputs['alpha'][ids]).astype('<f8'); rhs = labels - left
                # Direct unwhitened solve, reversed coordinates, independent of main QR/triangular solver.
                hp = h[:, ::-1].copy(); ca = np.linalg.solve(hp, rhs).T[:, ::-1]
                independent_equation = check(h, ca.T, rhs, 3e-10); main_equation = check(h, c.T, rhs, 3e-10)
                agreement = float(np.max(np.abs(ca - c) / (3e-8 * np.maximum(1, np.abs(ca) + np.abs(c)))))
                assert agreement <= 1
                residual = h @ c.T - rhs; rms = math.sqrt(float(np.einsum('ij,ij->', residual, residual)) / float(np.einsum('ij,ij->', labels, labels)))
                assert rms <= 1e-7 and raw['solve_cases'][e]['original_development'] == len(own)
                cases.append({'expert': e, 'direct_equation_ratio': independent_equation, 'main_equation_ratio': main_equation,
                              'coefficient_agreement_ratio': agreement, 'augmented_F64_fit_RMS': rms})
                allids = np.flatnonzero(m[:, 3] == e)
                for start in range(0, len(allids), 127):
                    at = allids[start:start + 127]; left = physical(inputs['q'][at], lq, ls, inputs['alpha'][at])
                    right = physical(feat['q'][at], bq, bs, feat['alpha'][at])
                    phys = ((left.astype('<f8') + right.astype('<f8')).astype('<f4').astype('<f8') + bias.astype('<f8')[None, :]).astype('<f4')
                    assert phys.tobytes() == pred['F'][at].tobytes()
                    assert (phys.astype('<f8') * ps[at].astype('<f8')[:, None]).astype('<f4').tobytes() == pred['oracle'][at].tobytes()
                    assert (phys.astype('<f8') * pred['p'][at].astype('<f8')[:, None]).astype('<f4').tobytes() == pred['sameID_mass'][at].tobytes()
                    h = np.ones((len(at), 513), '<f8'); ph = feat['q'][at].astype('<f8'); alpha = feat['alpha'][at].astype('<f8'); h[:, :512] = ph * alpha[:, None]
                    u64 = left.astype('<f8') + h[:, ::-1] @ c[:, ::-1].T
                    u32 = left.astype('<f8') + h[:, ::-1] @ cc[:, ::-1].astype('<f8').T
                    decoded = (left.astype('<f8') + ((ph @ bq.astype('<f8').T) * bs.astype('<f8')[None, :]) * alpha[:, None]) + bias.astype('<f8')[None, :]
                    fs = target[at].astype('<f8'); ys = y[at].astype('<f8'); p = ps[at].astype('<f8')[:, None]
                    decomp = decomposed(fs, u64, u32, decoded, pred['F'][at].astype('<f8'))
                    weighted = decomposed(ys, p * u64, p * u32, p * decoded, pred['oracle'][at].astype('<f8'))
                    ef = pred['oracle'][at].astype('<f8') - ys; ec = pred['choice_source_mass'][at].astype('<f8') - pred['oracle'][at].astype('<f8')
                    em = pred['coupled'][at].astype('<f8') - pred['choice_source_mass'][at].astype('<f8'); total = pred['coupled'][at].astype('<f8') - ys
                    sm = pred['sameID_mass'][at].astype('<f8') - pred['oracle'][at].astype('<f8')
                    diag = [np.einsum('ij,ij->i', v, v) for v in (total, ec, em, sm)]
                    cross = [np.einsum('ij,ij->i', v, w) for v, w in ((ef, ec), (ef, em), (ec, em))]
                    residual = ef + ec + em - total; vb = 3e-12 * np.maximum(1, np.abs(ys) + np.abs(pred['oracle'][at].astype('<f8')) + np.abs(pred['choice_source_mass'][at].astype('<f8')) + np.abs(pred['coupled'][at].astype('<f8')))
                    closure = diag[0] - (weighted[:, 3] + diag[1] + diag[2] + 2 * sum(cross))
                    eb = 1e-10 * np.maximum(1, weighted[:, 0] + weighted[:, 3] + diag[1] + diag[2] + 2 * sum(np.abs(v) for v in cross))
                    values[at] = np.column_stack((decomp, weighted, *diag, (pred['p'][at].astype('<f8') - ps[at].astype('<f8')) ** 2,
                        ps[at].astype('<f8') ** 2, *cross, np.max(np.abs(residual) / vb, axis=1), np.abs(closure) / eb))
                    seen[at] = True; ctx.guard()
                if e % 8 == 7: print(json.dumps({'independent_readout_parent': e, 'seconds': ctx.resources()['seconds']}), flush=True)
            assert not cf.read(1) and not ff.read(1)
        assert seen.all() and np.isfinite(values).all() and np.max(values[:, [45, 46]]) <= 1
        for e in range(128):
            ids = np.flatnonzero(pred['id'] == e); lq, ls, bq, bs, bias = M.parts(bank, e)
            for start in range(0, len(ids), 127):
                at = ids[start:start + 127]; left = physical(inputs['q'][at], lq, ls, inputs['alpha'][at]); right = physical(feat['q'][at], bq, bs, feat['alpha'][at])
                fc = ((left.astype('<f8') + right.astype('<f8')).astype('<f4').astype('<f8') + bias.astype('<f8')[None, :]).astype('<f4')
                assert (fc.astype('<f8') * ps[at].astype('<f8')[:, None]).astype('<f4').tobytes() == pred['choice_source_mass'][at].tobytes()
                assert (fc.astype('<f8') * pred['p'][at].astype('<f8')[:, None]).astype('<f4').tobytes() == pred['coupled'][at].tobytes(); ctx.guard()
        retained = wire(np, folder / 'energy_by_uid.bin', b'M521ENG1', 376, 47, np.dtype(('<f8', (47,))), 17540)
        assert np.allclose(values[:, :14], retained[:, :14], rtol=3e-8, atol=1e-7)
        assert np.allclose(values[:, 18:32], retained[:, 18:32], rtol=3e-8, atol=1e-7)
        assert np.allclose(values[:, 36:45], retained[:, 36:45], rtol=3e-8, atol=1e-7)
        assert np.max(values[:, [14, 16, 32, 34, 45, 46]]) <= 1 and np.max(retained[:, [14, 16, 32, 34, 45, 46]]) <= 1
        # Independent reduction order changes cancellation-sized residuals; certify all energies and both closures.
        for col in (17, 35): assert np.allclose(values[:, col], retained[:, col], rtol=3e-8, atol=1e-7)
        ctx.r['gates']['ALL128_direct_unique_equations_codec_fixed_features_L0_native_and_ALL17540_independent_energy_controls'] = True
        reports = M.reports(m, occ, values, pred['id'], ps, pred['p'])
        for group in ('uid_roles', 'cells', 'rare', 'views', 'role_mode'):
            assert len(reports[group]) == len(raw['reports'][group])
            for current, previous in zip(reports[group], raw['reports'][group]):
                for key, value in current.items():
                    other = previous[key]
                    if isinstance(value, float):
                        if any(v in key for v in ('identity', 'closure')): continue
                        assert math.isclose(value, other, rel_tol=3e-8, abs_tol=1e-7), (group, key, value, other)
                    else: assert value == other, (group, key)
        assert reports['outcomes'] == raw['reports']['outcomes'] and reports['exposures'] == raw['reports']['exposures']
        # Recompute each declared six-view decision independently from UID energies.
        for name, column in (('unweighted_fit64', 1), ('unweighted_fit32', 2), ('unweighted_oracle', 3),
                             ('weighted_fit64', 19), ('weighted_fit32', 20), ('weighted_oracle', 21), ('coupled', 36)):
            denominator = 0 if name.startswith('unweighted') else 18; passed = True
            for role in range(3):
                for mode in range(2):
                    ids = occ[(occ[:, 7] == role) & (occ[:, 6] == mode), 1]
                    numerator = math.fsum(map(float, values[ids, column])); energy = math.fsum(map(float, values[ids, denominator]))
                    passed &= bool(len(ids) and energy > 0 and math.sqrt(numerator / energy) <= .01)
            assert passed == reports['outcomes'][name + '_ALL_six_RMS_1pct']
        ctx.r['gates']['ALL1040_reports384_exposures_and_independent_six_view_function_decisions'] = True
        decision = S.decision(reports['outcomes']); assert decision == raw['decision']
        ctx.finish({'main_sha256': args.main_sha, 'reports': reports, 'equation_cases': cases,
                    'source_response_function_calls': source_calls, 'readout_coefficient_solves': 128,
                    'model_calls': 0, 'native_calls': 0, 'compiler_calls': 0, 'physical_DRAM_verified': False,
                    'decision': decision, 'scope': 'All new source functions independently reconstructed once; full equations/physical metrics audited. Shared wire/report definitions; independent source, solver, codec, arithmetic, energy reductions and six-view decisions. No new corpus/whole quality or goal promotion.'})
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
