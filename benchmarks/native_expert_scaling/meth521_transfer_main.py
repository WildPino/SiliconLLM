"""One new source acquisition, fixed full-information readout and native candidate."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth521_operations import Context, DOC, ROOT, wire, write


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--binding-sha', required=True); args = ap.parse_args(); ctx = None
    try:
        ctx = Context('main', args.binding_sha); np = ctx.numpy()
        import meth521_math as S
        M = S.M
        binary = ctx.out / 'meth521_transfer_cpu.exe'
        ctx.run([ctx.b['compiler']['path'], '-O3', '-std=c11', '-march=x86-64-v3', '-fno-fast-math', '-ffp-contract=off',
                 ROOT / 'benchmarks/native_expert_scaling/meth521_transfer_cpu.c', '-o', binary], 'compile')
        binary_desc = {'path': str(binary), 'bytes': binary.stat().st_size, 'sha256': ctx.digest(binary)}
        native = [json.loads(v) for v in ctx.run([binary, 'controls'], 'new_controls')]
        assert len(native) == 3; write(ctx.out / 'controls.json', S.controls(native))
        bad = ctx.out / 'negative_bank.bin'
        with bad.open('xb') as f: f.write(b'BADMAGIC' + struct.pack('<8I', 768, 3072, 512, 128, 11, 4, 8, 16))
        negative = ctx.out / 'negative_predictions.bin'
        ctx.run([binary, 'predict', bad, 'unused_source_path', negative], 'negative_magic', 2)
        assert not negative.exists() and (ctx.out / 'negative_magic.stderr').read_bytes().replace(b'\r\n', b'\n') == b'factor521_error:wire_magic\n'
        ctx.r['gates']['NEW_unique_QR_I8_LUT_A16_source_I64_RNE_products_and_negative_wire_controls'] = True
        uid, occ, inputs, ps, feat, target, y, pairs, aug, kreg = S.load(ctx); m = uid['m']
        ctx.r['gates']['ALL17540_unweighted_F_input_hash_mass_19962_joins_and_immutable53943_dev_only_pairs'] = True
        pairpath = ctx.out / 'source_pairs.bin'
        with pairpath.open('xb') as f:
            f.write(struct.pack('<8sIIQ', b'M521PA01', 8, 768, 53943)); f.write(pairs.tobytes())
        extentpath = ctx.out / 'source_extents.bin'
        with extentpath.open('xb') as f:
            f.write(struct.pack('<8sIIQ', b'M521EX01', 32, 768, 128))
            for e, parent in enumerate(ctx.b['parents']):
                assert parent['parent'] == e
                f.write(struct.pack('<4Q', parent['wi']['offset'], parent['wi']['scale_offset'], parent['wo']['offset'], parent['wo']['scale_offset']))
        responsepath = ctx.out / 'source_responses.bin'
        rows = ctx.run([binary, 'capture', ctx.b['payload']['path'], extentpath, ctx.data('inputs'), pairpath, responsepath], 'new_source_capture')
        assert json.loads(rows[-1]) == {'new_source_function_calls': 53943, 'source_response_width': 9236}
        response = wire(np, responsepath, b'M521SR01', 9236, 768, S.RESPONSE, 53943)
        assert np.array_equal(response['UID'], pairs['UID']) and np.array_equal(response['expert'], pairs['expert'])
        assert np.array_equal(response['input_owner'], pairs['input_owner'])
        assert np.all(np.isfinite(response['alpha']) & (response['alpha'] > 0))
        for start in range(0, 53943, 256):
            part = response[start:start + 256]
            assert np.all(part['codes'] >= 0) and np.isfinite(part['F']).all()
            assert np.array_equal(np.count_nonzero(part['codes'], axis=1), part['nonzero']); ctx.guard()
        ctx.r['gates']['ONE_NEW_native_source_all53943_requested_function_owner_UID_and_hidden_codec_records'] = True
        bank = bytearray(Path(ctx.data('bank')).read_bytes())
        assert len(bank) == 127232136 and bank[:40] == struct.pack('<8s8I', b'M499BNK1', 768, 3072, 512, 128, 11, 4, 8, 16)
        prefix = hashlib.sha256(bank[8:223368]).hexdigest(); bank[:8] = b'M521BNK1'
        t = np.linalg.cholesky(kreg); metric_factor = M.envelope(t, t.T, kreg, 5e-12); summaries = []
        with (ctx.out / 'coefficients_F64.bin').open('xb') as f64, (ctx.out / 'coefficients_F32.bin').open('xb') as f32:
            f64.write(struct.pack('<8sIIQ', b'M521F641', 3151872, 513, 128))
            f32.write(struct.pack('<8sIIQ', b'M521F321', 1575936, 513, 128))
            for e in range(128):
                ids = aug[e]; labels, own = S.augmented_labels(target, response, pairs, aug, m, e)
                lq, ls, _, _, _ = M.parts(bank, e)
                left = M.block(inputs['q'][ids], lq, ls, inputs['alpha'][ids]).astype('<f8')
                c, record = S.unique(S.design(feat, ids), labels - left, t, float(np.sum(labels * labels)), ctx.guard)
                cc = c.astype('<f4'); assert np.isfinite(cc).all()
                bq, bs = M.quant(cc[:, :512]); at = 223368 + 992256 * e + 592896
                bank[at:at + 393216] = bq.tobytes(); bank[at + 393216:at + 396288] = bs.tobytes()
                bank[at + 396288:at + 399360] = cc[:, 512].tobytes()
                f64.write(c.tobytes()); f32.write(cc.tobytes())
                summaries.append({'expert': e, 'original_development': own, 'new_source_pairs': 513 - own, **record})
                ctx.guard()
                if e % 8 == 7: print(json.dumps({'full_information_parent_fit': e, 'seconds': ctx.resources()['seconds']}), flush=True)
        assert hashlib.sha256(bank[8:223368]).hexdigest() == prefix
        oldbank = Path(ctx.data('bank')).open('rb')
        try:
            for e in range(128):
                at = 223368 + 992256 * e; oldbank.seek(at)
                assert oldbank.read(592896) == bank[at:at + 592896]
        finally: oldbank.close()
        bankpath = ctx.out / 'bank.bin'
        with bankpath.open('xb') as f: f.write(bank)
        ctx.r['gates']['ALL128_unique_augmented_readouts_fixed_A_keys_L0_rank_metric_and_I8_bank'] = True
        pp = ctx.out / 'predictions.bin'; ctx.run([binary, 'predict', bankpath, ctx.data('inputs'), pp], 'new_candidate_predict')
        pred = wire(np, pp, b'M521PRE1', 15464, 768, M.PRED, 17540)
        baseline = wire(np, ctx.data('baseline'), b'M499PRE1', 15464, 768, M.PRED, 17540)
        for field in ('id', 'p', 'logits'): assert pred[field].tobytes() == baseline[field].tobytes()
        del baseline
        allowed = {Path(v['path']).resolve() for v in ctx.b['catalog']} | {binary.resolve()}
        for cmd in ctx.r['commands']: assert {Path(v).resolve() for v in cmd['observed_modules']} <= allowed, cmd
        assert ctx.digest(binary) == binary_desc['sha256']
        ctx.r['gates']['actual_native_commands_module_samples_and_unchanged_ID_mass_logit_BYTES'] = True
        data = np.empty((17540, 47), '<f8'); seen = np.zeros(17540, bool)
        with (ctx.out / 'coefficients_F64.bin').open('rb') as f64, (ctx.out / 'coefficients_F32.bin').open('rb') as f32:
            assert f64.read(24) == struct.pack('<8sIIQ', b'M521F641', 3151872, 513, 128)
            assert f32.read(24) == struct.pack('<8sIIQ', b'M521F321', 1575936, 513, 128)
            for e in range(128):
                c = np.frombuffer(f64.read(3151872), '<f8').reshape(768, 513)
                cc = np.frombuffer(f32.read(1575936), '<f4').reshape(768, 513)
                assert c.astype('<f4').tobytes() == cc.tobytes()
                lq, ls, bq, bs, bias = M.parts(bank, e); ids = np.flatnonzero(m[:, 3] == e)
                for start in range(0, len(ids), 256):
                    at = ids[start:start + 256]; left = M.block(inputs['q'][at], lq, ls, inputs['alpha'][at])
                    right = M.block(feat['q'][at], bq, bs, feat['alpha'][at])
                    physical = np.add(np.add(left, right, dtype=np.float32), bias, dtype=np.float32)
                    assert physical.tobytes() == pred['F'][at].tobytes()
                    assert np.multiply(physical, ps[at, None], dtype=np.float32).tobytes() == pred['oracle'][at].tobytes()
                    assert np.multiply(physical, pred['p'][at, None], dtype=np.float32).tobytes() == pred['sameID_mass'][at].tobytes()
                    h = S.design(feat, at); l = left.astype('<f8'); alpha = feat['alpha'][at].astype('<f8')
                    u64 = l + h @ c.T; u32 = l + h @ cc.astype('<f8').T
                    integer = feat['q'][at].astype('<f8') @ bq.astype('<f8').T
                    decoded = (l + ((integer * bs.astype('<f8')[None, :]) * alpha[:, None])) + bias.astype('<f8')[None, :]
                    data[at] = M.energies(target[at].astype('<f8'), y[at].astype('<f8'), u64, u32, decoded, pred[at], ps[at].astype('<f8'))
                    seen[at] = True; ctx.guard()
        assert seen.all()
        fc = M.emit(bank, inputs, feat, pred['id'], ctx.guard)
        assert np.multiply(fc, ps[:, None], dtype=np.float32).tobytes() == pred['choice_source_mass'].tobytes()
        assert np.multiply(fc, pred['p'][:, None], dtype=np.float32).tobytes() == pred['coupled'].tobytes(); del fc
        with (ctx.out / 'energy_by_uid.bin').open('xb') as f:
            f.write(struct.pack('<8sIIQ', b'M521ENG1', 376, 47, 17540)); f.write(data.tobytes())
        ctx.r['gates']['ALL17540_native_products_four_function_levels_and_nonorthogonal_error_accounting'] = True
        reports = M.reports(m, occ, data, pred['id'], ps, pred['p'])
        old = json.loads((DOC / 'meth499_factor_fit_result.json').read_bytes())['reports']
        for label in ('uid_roles', 'cells', 'rare', 'views', 'role_mode'):
            assert len(reports[label]) == len(old[label])
            for current, previous in zip(reports[label], old[label]):
                for key in ('count', 'ID_correct', 'ID_fidelity'): assert current[key] == previous[key]
                for key in ('source_energy', 'unweighted_source_energy'):
                    assert np.isclose(current[key], previous[key], rtol=1e-10, atol=1e-8)
        assert reports['exposures'] == old['exposures'] and sum(len(reports[v]) for v in ('uid_roles', 'cells', 'rare', 'views', 'role_mode')) == 1040
        assert sum(v['count'] for v in reports['role_mode']) == 19962
        ctx.r['gates']['ALL1040_reports_384_exposures_rare_every_book_and_original499_comparison_contract'] = True
        ctx.finish({'reports': reports, 'solve_cases': summaries, 'metric_factor_ratio': metric_factor,
                    'fixed_dictionary_keys_sha256': prefix, 'native_binary': binary_desc,
                    'source_response_function_calls': 53943, 'new_source_MACs': 254535008256,
                    'readout_coefficient_solves': 128, 'optimizer_updates': 0,
                    'compiler_calls': 1, 'native_calls': 4, 'model_calls': 0, 'UIDs': 17540, 'occurrences': 19962,
                    'physical_bank_bytes': 127232136, 'logical_weight_bytes_per_12_bank_token': 14587008,
                    'physical_DRAM_verified': False, 'decision': S.decision(reports['outcomes']),
                    'scope': 'New development-only counterfactual responses at fixed features. Consumed roles reused as diagnostics, not fresh heldout quality. Known routing/mass fails remain; no whole/rate/n/family promotion.'})
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
