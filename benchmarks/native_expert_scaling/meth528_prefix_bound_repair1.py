"""First C verification of retained parents1..12; exact Python integer bound."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
import ctypes
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth528_prefix_io_repair1 import PrefixContext, ROOT, DOC, OLD, build, pointer, save, write


def lifted(bits):
    exp = (bits >> 23) & 255
    assert exp != 255
    value = (bits & 0x7fffff) if exp == 0 else ((bits & 0x7fffff) | 0x800000) << (exp - 1)
    return -value if bits >> 31 else value


def energy(bits, guess=None):
    if guess is None:
        return sum(lifted(int(v)) ** 2 for v in bits)
    return sum((lifted(int(v)) - lifted(int(w))) ** 2 for v, w in zip(bits, guess))


def limbs(np, value):
    assert 0 <= value < 1 << 640
    return np.frombuffer(value.to_bytes(80, 'little'), '<u8')


def lower_factor(count):
    u = Fraction(1, 1 << 53)
    row = Fraction(767, (1 << 53) - 767)
    domain = Fraction(max(0, count - 1), (1 << 53) - max(0, count - 1))
    # Difference twice after squaring, square, division, squared sqrt: six.
    return (1 - u) ** 6 * (1 - row) * (1 - domain) / ((1 + u) * (1 + row) * (1 + domain))


def sqrt_controls():
    values = [0.0, 1.0, 2.0, 0.01 ** 2, math.nextafter(0.01 ** 2, math.inf), 2.0 ** -580, 2.0 ** 580]
    result = []
    for value in values:
        y = math.sqrt(value)
        if value:
            fy = Fraction.from_float(y)
            lo = (fy + Fraction.from_float(math.nextafter(y, -math.inf))) / 2
            hi = (fy + Fraction.from_float(math.nextafter(y, math.inf))) / 2
            assert lo ** 2 <= Fraction.from_float(value) <= hi ** 2
        else:
            assert y == 0
        result.append(dict(x_hex=value.hex(), sqrt_hex=y.hex()))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest-sha', required=True)
    ap.add_argument('--freeze', required=True)
    args = ap.parse_args()
    ctx = None
    try:
        ctx = PrefixContext('prefix_bound_repair1', args.manifest_sha, args.freeze)
        np = ctx.numpy()
        import meth528_contract as S
        m, occ, dev, counts = S.metadata(ctx)
        dll, dll_sha = build(ctx, ROOT / 'benchmarks/native_expert_scaling/meth528_verifier.c', 'first_prefix_scalar')
        lib = ctypes.CDLL(str(dll))
        lib.m528_quant.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 3
        lib.m528_quant.restype = ctypes.c_int
        lib.m528_eval.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 11
        lib.m528_eval.restype = ctypes.c_int
        calls = 0

        def quant(values):
            nonlocal calls
            x = np.array(values, '<f4'); q = np.empty(len(x), '<i2'); a = np.empty(1, '<f4')
            calls += 1
            ctx.r['C_response_calls_entered'] = calls
            assert lib.m528_quant(len(x), pointer(x), pointer(q), pointer(a)) == 0
            ctx.r['C_response_calls_returned_zero'] = calls
            return q, a

        def evaluate(dots, alpha, si, wo, so):
            nonlocal calls
            inputs = [np.ascontiguousarray(a, dtype) for a, dtype in zip((dots, alpha, si, wo, so), ('<i8', '<f4', '<f4', 'i1', '<f4'))]
            n, b = inputs[0].shape
            q = np.empty((n, b), '<i2'); a = np.empty(n, '<f4')
            integer = np.empty((n, 768), '<i8'); physical = np.empty((n, 768), '<f4')
            calls += 1
            ctx.r['C_response_calls_entered'] = calls
            assert lib.m528_eval(n, b, *[pointer(v) for v in [*inputs, q, a, integer, physical, None, None]]) == 0
            ctx.r['C_response_calls_returned_zero'] = calls
            return physical, integer, q, a

        ctl = json.loads((OLD / 'controls.json').read_bytes())
        q, a = quant([32767, -32767, 1.5, 2.5, 4.5, 5.5, -1.5, -4.5])
        assert [q.tolist()] == ctl['tie_codes'] and a.tolist() == [1.0]
        q, a = quant([0, 0, 0, 0]); assert not q.any() and a.tolist() == [1.0] == [ctl['zero_alpha']]
        wide = evaluate(np.full((1, 3072), 32767, '<i8'), np.ones(1, '<f4'), np.ones(3072, '<f4'), np.full((768, 3072), 127, 'i1'), np.ones(768, '<f4'))
        assert np.all(wide[1] == ctl['wide_WO_integer'] == 3072 * 127 * 32767) and ctl['wide_WO_integer'] > 2 ** 31
        assert np.all(wide[0] == np.float32(ctl['wide_WO_physical']))
        sqrt_receipt = sqrt_controls()
        write(ctx.out / 'first_C_controls.json', dict(C_calls=calls, tie_codes=ctl['tie_codes'], zero_alpha=ctl['zero_alpha'], wide_integer=ctl['wide_WO_integer'], sqrt_controls=sqrt_receipt))
        ctx.r['gates']['first_C_tie_zero_wide_controls_and_local_sqrt_midpoint_controls'] = True
        prefix = np.flatnonzero((m[:,3] >= 1) & (m[:,3] <= 12))
        position = {int(uid): i for i, uid in enumerate(prefix)}
        physical = np.empty((len(prefix), 2, 768), '<f4')
        freeze = json.loads((OLD / 'development_mask_freeze.json').read_bytes())
        cases = []; terms = 0
        for e in range(1, 13):
            ids = np.flatnonzero(m[:,3] == e)
            inp = S.records(ctx, 'inputs', ids)
            assert np.all(inp['e'] == e) and np.all(inp['accept'] == 1)
            assert inp['p'].tobytes() == m[ids,10].tobytes() and inp['alpha'].tobytes() == m[ids,12].tobytes()
            wi, si, wo, so = S.weights(ctx, e)
            u = S.old(ctx, 'old_hinges', e).astype(int); signs = S.old(ctx, 'old_signs', e); anchor = int(S.old(ctx, 'old_anchors', e))
            assert dev[anchor] and m[anchor,3] == e and len(set(int(v) for v in u)) == 512
            selected = np.array(sorted(set(int(v) for v in u) | {j for j in range(3072) if signs[j]}), '<u2')
            case = dict(expert=e, anchor_UID=anchor, development=int(counts[e]), width=len(selected), extra_anchor_positive_rows=len(selected)-512, candidate_bank_bytes=3104+1542*len(selected), active_MACs=1536*len(selected), rotation=1536)
            assert case == freeze['cases'][e-1]
            saved = S.unpack(OLD / f'e{e:03d}_a0.bin', e, 0)
            expected = selected, si[selected], so, wi[selected], wo[:,selected]
            assert all(a.tobytes(order='C') == b.tobytes(order='C') for a, b in zip(saved, expected))
            dots = S.old(ctx, 'old_signed_dots', ids)
            assert dots.dtype == np.dtype('<i8') and np.max(np.abs(dots)) <= 768*128*32767
            answer, integer, codes, ah = evaluate(dots[:,selected], inp['alpha'], saved[1], saved[4], saved[2])
            for name, value in (('physical', answer), ('integer', integer), ('codes', codes), ('alpha', ah)):
                save(np, ctx.out, f'e{e:03d}_a0_verified_' + name, value)
            assert answer.tobytes() == S.get(OLD, 'physical', ids)[:,0].tobytes()
            assert integer.tobytes() == S.get(OLD, 'integer_WO', ids)[:,0].tobytes()
            assert ah.tobytes() == S.get(OLD, 'hidden_alpha', ids)[:,0].tobytes()
            old_codes = np.load(OLD / f'e{e:03d}_a0_hidden_codes.npy', allow_pickle=False)
            assert old_codes.shape == codes.shape and old_codes.tobytes() == codes.tobytes()
            at = [position[int(uid)] for uid in ids]
            physical[at,0] = answer
            physical[at,1] = np.multiply(answer, inp['p'][:,None], dtype=np.float32)
            terms += len(ids)*768*len(selected)
            cases.append(dict(**case, UID_count=len(ids)))
            write(ctx.out / f'e{e:03d}_verification.json', dict(**cases[-1], all_packed_and_physical_integer_alpha_codes_BYTE=True, cumulative_C_calls=calls))
            ctx.r['completed_C_BYTE_parent_prefix'] = e
            ctx.guard()
            print('first_C_prefix_parent=' + str(e), flush=True)
        assert calls == 15
        save(np, ctx.out, 'prefix_UIDs', prefix.astype('<u4'))
        save(np, ctx.out, 'verified_physical', physical)
        write(ctx.out / 'first_C_verification.json', dict(C_calls=calls, response_parent_calls=12, quant_controls=2, wide_control=1, C_verifier_sha256=dll_sha, cases=cases, exact_selected_WO_terms=terms, sqrt_controls=sqrt_receipt))
        ctx.r['gates']['ALL_completed_1_to_12_candidate_packed_source_and_I64_F32_A16_BYTE'] = True
        del dots, wi, si, wo, so, old_codes, answer, integer, codes, ah, wide
        den = np.empty((17540,2,10), '<u8'); err = np.zeros_like(den)
        ref = S.records(ctx, 'unweighted', slice(None)); target = S.records(ctx, 'targets', slice(None))
        p = m[:,10].copy().view('<f4')
        assert np.isfinite(ref).all() and np.isfinite(target).all() and np.isfinite(physical).all()
        assert np.multiply(ref, p[:,None], dtype=np.float32).tobytes() == target.tobytes()
        den_int = [[], []]; err_int = [[0]*17540, [0]*17540]
        for arm, source in enumerate((ref, target)):
            bits = source.view('<u4'); candidate = physical[:,arm].view('<u4')
            for uid in range(17540):
                value = energy(bits[uid].tolist())
                den_int[arm].append(value); den[uid,arm] = limbs(np, value)
                if uid in position:
                    value = energy(bits[uid].tolist(), candidate[position[uid]].tolist())
                    err_int[arm][uid] = value; err[uid,arm] = limbs(np, value)
                if uid % 512 == 0:
                    ctx.guard()
            print('exact_source_and_prefix_energies_arm=' + str(arm), flush=True)
        save(np, ctx.out, 'source_U640', den); save(np, ctx.out, 'prefix_error_U640', err)
        threshold = Fraction.from_float(0.01)
        views = []
        for label, ids in S.domain_ids(m, occ, counts):
            role = label['kind'] == 'role'
            rare = label['kind'] == 'rare' and label['split'] == 'consumed_validation' and label['development_class'] in ('1..4', '5..15')
            if not (role or rare): continue
            factor = lower_factor(len(ids))
            view = dict(**label, count=len(ids), prefix_occurrences=sum(int(v) in position for v in ids), rounding_lower_squared_factor_numerator=str(factor.numerator), rounding_lower_squared_factor_denominator=str(factor.denominator))
            for arm, name in enumerate(('unweighted', 'weighted')):
                D = sum(den_int[arm][int(v)] for v in ids); L = sum(err_int[arm][int(v)] for v in ids)
                left = L*factor.numerator*threshold.denominator**2
                right = D*factor.denominator*threshold.numerator**2
                reject = bool(len(ids) and (D == 0 or left > right))
                view[name] = dict(full_denominator_integer=str(D), prefix_error_integer=str(L), guarded_cross_left=str(left), guarded_cross_right=str(right), strict_original_gate_failure=reject, real_prefix_RMS_lower_display=math.sqrt(L/D) if D else None, nonempty_zero_denominator_original_failure=bool(len(ids) and D == 0))
            views.append(view)
        assert len(views) == 8
        decision = 'NECESSARY_FIXED_DOMAIN_FIDELITY_FAILURE_CERTIFIED' if any(v[a]['strict_original_gate_failure'] for v in views for a in ('unweighted','weighted')) else 'INCONCLUSIVE_PREFIX_BOUND'
        ctx.r['gates']['exact_full_source_and_verified_prefix_U640_energies_and_original_fixed_domains'] = True
        write(ctx.out / 'exact_bound.json', dict(views=views, decision=decision, original_binary_threshold_numerator=str(threshold.numerator), original_binary_threshold_denominator=str(threshold.denominator)))
        ctx.finish(dict(decision=decision, independently_energy_audited=False, first_C_verifier_sha256=dll_sha, new_C_response_calls=calls, cases=cases, prefix_UID_count=len(prefix), original_UID_count=17540, original_occurrence_count=19962, exact_source_scalar_terms=2*17540*768, exact_prefix_error_scalar_terms=2*len(prefix)*768, views=views, floating_contract='IEEE754 binary64 nearest-even basic arithmetic and correctly rounded sqrt; exact midpoint controls are qualification samples, not a proof of the library for all inputs.', full528_eligibility='UNKNOWN; original main resource FALSE', no_completed_source_model_native_replay=True))
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__':
    main()
