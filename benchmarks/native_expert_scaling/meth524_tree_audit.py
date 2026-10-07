"""Independent reversed reductions, exact controls and integer source routing."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth524_operations import Context, DOC, ROOT, write


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--binding-sha', required=True); ap.add_argument('--main-sha', required=True)
    args = ap.parse_args(); ctx = None
    try:
        ctx = Context('audit', args.binding_sha); mainpath = DOC / 'meth524_main_result.json'
        assert ctx.digest(mainpath) == args.main_sha
        raw = json.loads(mainpath.read_bytes()); assert raw['binding_sha256'] == args.binding_sha and all(raw['gates'].values())
        folder = ROOT / 'results/native_expert_scaling/meth524_tree'
        for v in raw['output_inventory']: assert Path(v['path']).stat().st_size == v['bytes'] and ctx.digest(v['path']) == v['sha256']
        assert json.loads((folder / 'terminal_resource.json').read_bytes())['result_sha256'] == args.main_sha
        ctx.r['gates']['complete_main_ALL_output_terminal_and_freeze_SHA'] = True
        np = ctx.numpy(); import meth524_contract as S
        # Import only wire/constants/I/O. No main or main-math imports.
        ctl = json.loads((folder / 'controls.json').read_bytes())
        for arm in range(2):
            r = [[Fraction(v) for v in row] for row in ctl['residual']]
            rho = [Fraction(v) for v in ctl['rho']]; w = [Fraction(1) if arm == 0 else Fraction(v) ** 2 for v in ctl['p']]
            source = [[Fraction(v) for v in row] for row in ctl['source']]
            def exact(ids):
                denominator = sum(w[i] * rho[i] ** 2 for i in ids)
                sums = [sum(w[i] * rho[i] * r[i][t] for i in ids) for t in range(2)]
                return sum(v * v for v in sums) / denominator
            energy = sum(w[i] * v * v for i, row in enumerate(source) for v in row)
            for j in range(3):
                plus = [i for i in range(4) if ctl['sign'][i][j]]; minus = [i for i in range(4) if not ctl['sign'][i][j]]
                gain = (exact(plus) + exact(minus) - exact(range(4))) / energy
                assert abs(float(gain) - ctl['panel'][j][arm]) <= ctl['panel'][j][3]
        assert ctl['representatives'] == [0, 1] and ctl['positive_counts'] == [2, 2, 2]
        assert all(ctl['sign'][i][0] != ctl['sign'][i][2] for i in range(4)) and ctl['zero_sign_branch'] == 'nonpositive'
        scores = [v[2] for v in ctl['panel']]; assert ctl['winner'] == max([0, 1], key=lambda j: (scores[j], -j))
        mass = ctl['hard_child_mass_fraction_control']; parents = [Fraction(v) for v in mass['parents']]
        children = [[Fraction(0)] * 3 for _ in parents]
        for i, value in enumerate(parents): children[i][mass['children'][i]] = value
        assert sum(v for row in children for v in row) == sum(parents) == Fraction(mass['sum']) == 1
        assert max((children[e][l], -e, -l) for e in range(3) for l in range(3))[1] == -parents.index(max(parents))
        assert not (sum([0, 0]) > 0)
        ctx.r['gates']['independent_Fraction_radial_gain_complement_and_hard_mass_identity_controls'] = True
        m, occ, inputs, dev, counts = S.load(ctx)
        ctx.r['gates']['ALL_original_UID_parent_probability_and_occurrence_contracts'] = True
        freeze = json.loads((folder / 'development_freeze.json').read_bytes())
        assert ctx.digest(folder / 'development_freeze.json') == raw['development_freeze_sha256']
        assert freeze['cases'] == raw['cases'] and len(freeze['cases']) == 128
        for ref in freeze['panels']: assert ctx.digest(ref['path']) == ref['sha256']
        eps = 2. ** -53
        def gam(n): return (n * eps) / (1 - n * eps)
        def independent_panel(sign, r, rho, p, source):
            # Reverse UID order, transpose BLAS reductions, fsum parent sums.
            sign, r, rho, p, source = sign[::-1], r[::-1], rho[::-1], p[::-1], source[::-1]
            n, dim = r.shape; a = np.zeros((3072, 4), '<f8')
            for arm, weight in enumerate((np.ones(n), p ** 2)):
                v = r * (rho * weight)[:, None]; t = rho * rho * weight
                ds = gam(n + 3) * np.sum(np.abs(v), axis=0)
                mask = sign.astype('<f8'); inverse = (~sign).astype('<f8')
                sp, sm = v.T @ mask, v.T @ inverse
                dp = np.einsum('ij,i->j', mask, t, optimize=False)
                dm = np.einsum('ij,i->j', inverse, t, optimize=False)
                st = np.array([math.fsum(v[:, k]) for k in range(dim)], '<f8')
                dt = math.fsum(t)
                def term(sums, denominator):
                    numerator = np.einsum('ij,ij->j', sums, sums, optimize=False) if sums.ndim == 2 else float(np.dot(sums, sums))
                    value = np.divide(numerator, denominator, out=np.zeros_like(np.asarray(numerator)), where=np.asarray(denominator) > 0)
                    delta_num = (np.sum(2 * np.abs(sums) * ds[:, None] + ds[:, None] ** 2, axis=0)
                                 if sums.ndim == 2 else np.sum(2 * np.abs(sums) * ds + ds ** 2))
                    delta_num += gam(dim + 2) * numerator
                    delta_den = gam(n + 3) * denominator
                    error = np.divide(delta_num + value * delta_den, denominator - delta_den,
                        out=np.zeros_like(np.asarray(value)), where=np.asarray(denominator) > 0) + eps * np.abs(value)
                    return value, error
                ep, bp = term(sp, dp); em, bm = term(sm, dm); et, bt = term(st, dt)
                gain = ep - et + em
                error = bp + bm + bt + gam(3) * (np.abs(ep) + np.abs(em) + abs(float(et)))
                energy = math.fsum(float(weight[i]) * float(z) * float(z) for i, row in enumerate(source) for z in row)
                if energy == 0: assert not np.any(r); continue
                energy_error = gam(n * dim + 4) * energy
                a[:, arm] = gain / energy
                a[:, 3] += error / (energy - energy_error) + np.abs(gain) * energy_error / (energy * (energy - energy_error)) + eps * np.abs(a[:, arm])
            a[:, 2] = a[:, 0] + a[:, 1]
            a[:, 3] = np.nextafter(2 * (a[:, 3] + eps * (np.abs(a[:, 0]) + np.abs(a[:, 1]))), np.inf)
            return a
        maximum_gain_ratio = 0.; verified_nodes = verified_candidates = 0; copied_tests = 0
        for case in freeze['cases'][1:]:
            e = case['expert']; di = np.flatnonzero(dev & (m[:, 3] == e))
            assert case['development'] == len(di) and case['fallback_in_future_function_test'] == bool(counts[e] < 16)
            dots = S.saved(ctx, 'signed_dots', di); f = S.saved(ctx, 'predictions', di, 0); g = S.saved(ctx, 'predictions', di, 2)
            r = np.subtract(f, g); qq = inputs['q'][di].astype('<i8')
            norm2 = np.einsum('ij,ij->i', qq, qq, dtype=np.int64, optimize=False)
            rho = np.array([math.sqrt(int(z)) for z in norm2]) * inputs['alpha'][di].astype('<f8')
            p = inputs['p'][di].astype('<f8')
            a = np.load(folder / f'e{e:03d}_gains.npy', allow_pickle=False)
            pc = np.load(folder / f'e{e:03d}_positive_counts.npy', allow_pickle=False)
            flags = np.load(folder / f'e{e:03d}_balanced.npy', allow_pickle=False)
            assert a.shape == (3, 3072, 4) and a.dtype == np.dtype('<f8') and pc.shape == flags.shape == (3, 3072)
            assert pc.dtype == np.dtype('<u2') and flags.dtype == np.dtype('u1')
            expected_nodes = {0}; seen_nodes = set(); leaf_uids = []
            for node in case['nodes']:
                number = node['node']; path = node['path']; assert number in expected_nodes and number not in seen_nodes
                seen_nodes.add(number); assert node['depth'] == len(path) <= 2
                selected = np.ones(len(di), bool); heap = 0; used = []
                for step in path:
                    selected &= (dots[:, step['neuron']] > 0) == step['positive']
                    heap = 2 * heap + 1 + int(step['positive']); used.append(step['neuron'])
                assert heap == number and len(set(used)) == len(used)
                local = np.flatnonzero(selected); assert di[local].tolist() == node['development_UIDs']
                if len(path) == 2:
                    assert not node['candidate_panel'] and node['status'] == 'leaf_depth_limit'
                elif len(local) < 4:
                    assert not node['candidate_panel'] and node['status'] == 'leaf_fewer_than4_UIDs'
                    assert not a[number].any() and not pc[number].any() and not flags[number].any()
                else:
                    assert node['candidate_panel']; sign = dots[local] > 0
                    actual_counts = np.count_nonzero(sign, axis=0); valid = (actual_counts >= 2) & (actual_counts <= len(local) - 2)
                    assert np.array_equal(pc[number], actual_counts) and np.array_equal(flags[number], valid)
                    expected = independent_panel(sign, r[local], rho[local], p[local], f[local])
                    bound = a[number, :, 3] + expected[:, 3]
                    discrepancy = np.max(np.abs(a[number, :, :3] - expected[:, :3]), axis=1)
                    assert np.all(discrepancy <= bound)
                    maximum_gain_ratio = max(maximum_gain_ratio, float(np.max(discrepancy / bound)))
                    verified_nodes += 1; verified_candidates += 3072
                    assert np.all(a[number, valid, 2] >= -a[number, valid, 3])
                    signatures = set(); reps = []
                    for j in np.flatnonzero(valid):
                        if int(j) in used: continue
                        one = tuple(np.flatnonzero(sign[:, j])); zero = tuple(np.flatnonzero(~sign[:, j]))
                        key = min(one, zero)
                        if key not in signatures: signatures.add(key); reps.append(int(j))
                    assert node['balanced_candidates'] == int(valid.sum()) and node['unique_partitions'] == len(reps)
                    leading = None
                    if not reps: status = 'no_balanced_partition'
                    else:
                        order = sorted(reps, key=lambda j: (-float(a[number, j, 2]), j)); leading = order[0]
                        lo = float(a[number, leading, 2] - a[number, leading, 3])
                        if lo <= 1e-12:
                            status = 'no_resolved_positive_gain' if a[number, leading, 2] + a[number, leading, 3] <= 1e-12 else 'unresolved_positive_gain'
                        elif len(order) > 1 and lo <= max(float(a[number, j, 2] + a[number, j, 3]) for j in order[1:]): status = 'unresolved_gain_order'
                        else: status = 'split'
                    assert node['status'] == status
                    if leading is not None: assert node['leading_neuron'] == leading and node['leading_panel'] == a[number, leading].tolist()
                    if status == 'split':
                        assert node['neuron'] == leading; expected_nodes.update((2 * number + 1, 2 * number + 2)); copied_tests += 1
                if node['status'] != 'split': leaf_uids.extend(node['development_UIDs'])
            assert expected_nodes == seen_nodes and sorted(leaf_uids) == di.tolist()
            assert case['leaves'] == [v['node'] for v in case['nodes'] if v['status'] != 'split']
            ctx.guard()
            if e % 32 == 0: print('audit_development_parent_complete=' + str(e), flush=True)
        ctx.r['gates']['ALL_development_panels_reversed_reductions_balanced_partition_classes_and_resolved_orders'] = True
        rowdata = (folder / 'source_test_rows.bin').read_bytes(); assert len(rowdata) == 772 * copied_tests
        got_leaves = np.zeros(S.N, 'u1'); got_depths = np.zeros(S.N, 'u1'); occupancy = []; dot_checks = 0; offsets = []
        for case in freeze['cases'][1:]:
            e = case['expert']; ix = np.flatnonzero(m[:, 3] == e); dots = S.saved(ctx, 'signed_dots', ix)
            qq = inputs['q'][ix].astype('<i8'); here = np.zeros(len(ix), '<u2'); depth = np.zeros(len(ix), 'u1')
            tensor = next(v for v in ctx.b['parents'] if v['parent'] == e)['wi']
            for node in case['nodes']:
                if node['status'] != 'split': continue
                j = node['neuron']; off = node['source_row_offset']; offsets.append(off)
                row = rowdata[off:off + 772]
                assert node['source_row_bytes'] == 772 and hashlib.sha256(row).hexdigest() == node['source_row_sha256']
                with Path(ctx.b['payload']['path']).open('rb') as file:
                    file.seek(tensor['offset'] + j * 768); codes = file.read(768)
                    file.seek(tensor['scale_offset'] + j * 4); scale = file.read(4)
                assert codes + scale == row and np.frombuffer(scale, dtype='<f4')[0] > 0
                integer = np.einsum('ij,j->i', qq, np.frombuffer(codes, dtype='i1').astype('<i8'), dtype=np.int64, optimize=False)
                assert np.array_equal(integer, dots[:, j]); dot_checks += len(ix)
                mask = here == node['node']; here[mask] = np.where(integer[mask] > 0, 2 * node['node'] + 2, 2 * node['node'] + 1)
                depth[mask] += 1
            for leaf in case['nodes']:
                if leaf['status'] != 'split': assert ix[dev[ix] & (here == leaf['node'])].tolist() == leaf['development_UIDs']
            assert set(int(v) for v in here).issubset(case['leaves'])
            got_leaves[ix] = here; got_depths[ix] = depth
            occupancy.append(dict(expert=e, leaves=[dict(node=t, development=int(np.count_nonzero(dev[ix] & (here == t))),
                consumed=int(np.count_nonzero((~dev[ix]) & (here == t)))) for t in case['leaves']]))
            ctx.guard()
        assert sorted(offsets) == list(range(0, len(rowdata), 772)) and dot_checks == raw['selected_source_predicate_dot_checks']
        for name, expected in (('leaf_ids', got_leaves), ('depths', got_depths),
            ('expanded_ids', (7 * m[:, 3] + got_leaves).astype('<u2')), ('selected_parent_mass', np.array(inputs['p'], copy=True))):
            assert np.load(folder / (name + '.npy'), allow_pickle=False).tobytes() == expected.tobytes()
        assert occupancy == raw['occupancy'] and np.all(got_depths <= 2)
        ctx.r['gates']['ALL_original_source_test_BYTES_independent_I64_routes_and_onehot_selected_mass_BYTES'] = True
        assert S.SOURCE_BYTES == 2 * 768 * 3072 + 4 * (3072 + 768) and S.SOURCE_MACS == 2 * 768 * 3072
        assert S.LEAF_BYTES == 2 * 768 ** 2 + 2 * 768 * 512 + 4 * (2 * 768 + 512) + 2 * 512
        assert S.LEAF_MACS == 768 ** 2 + 2 * 768 * 512 and S.NODE_BYTES == 768 + 4 + 16
        fallback = counts[m[:, 3]] < 16
        costs = np.full(S.N, S.SOURCE_BYTES, '<i8'); macs = np.full(S.N, S.SOURCE_MACS, '<i8')
        costs[~fallback] = S.LEAF_BYTES + S.NODE_BYTES * got_depths[~fallback].astype('<i8')
        macs[~fallback] = S.LEAF_MACS + 768 * got_depths[~fallback].astype('<i8')
        views = []
        for label, ids in S.domains(m, occ, counts):
            byte_sum = sum(int(costs[i]) for i in ids); mac_sum = sum(int(macs[i]) for i in ids)
            v = {**label, 'count': len(ids), 'fallback_rows': sum(bool(fallback[i]) for i in ids),
                 'active_bytes_sum': byte_sum, 'active_MACs_sum': mac_sum}
            if len(ids): v.update(logical_active_byte_ratio=byte_sum / (S.SOURCE_BYTES * len(ids)), logical_active_MAC_ratio=mac_sum / (S.SOURCE_MACS * len(ids)))
            views.append(v)
        assert views == raw['views'] and len(views) == 656
        eligible_cases = [c for c in freeze['cases'][1:] if counts[c['expert']] >= 16]
        nf = 127 - len(eligible_cases); nl = sum(len(c['leaves']) for c in eligible_cases)
        nt = sum(v['status'] == 'split' for c in eligible_cases for v in c['nodes'])
        price = dict(source_exposed_bank_bytes=127 * S.SOURCE_BYTES, source_fallback_parents=nf,
            one_leaf_functions=len(eligible_cases), multi_leaf_functions=nl, copied_selected_tests_all_parents=copied_tests,
            selected_tests_in_future_nonfallback=nt, one_leaf_bank_bytes=nf * S.SOURCE_BYTES + len(eligible_cases) * S.LEAF_BYTES + 2048,
            multi_leaf_bank_bytes=nf * S.SOURCE_BYTES + nl * S.LEAF_BYTES + nt * S.NODE_BYTES + 2048,
            node_record_bytes=16, copied_WI_row_bytes=772, header_bytes=2048,
            original_parent_winner_and_full_normalization='retained control, FULL cost still owed', actual_leaf_banks_compiled=False, physical_DRAM_verified=False)
        assert price == raw['price']
        eligibility = dict(ALL_nonfallback_parents_resolved_positive_balanced_root=bool(eligible_cases) and all(c['nodes'][0]['status'] == 'split' for c in eligible_cases),
            ALL_nonfallback_node_orders_resolved=all(not v['status'].startswith('unresolved') for c in eligible_cases for v in c['nodes']),
            planned_occupied_nonfallback_leaf_count_increases=nl > len(eligible_cases),
            ALL_six_logical_active_bytes_le_75pct=all(v['logical_active_byte_ratio'] <= .75 for v in views if v['kind'] == 'role'),
            ALL_nonempty_books_logical_active_bytes_le_source=all(v['logical_active_byte_ratio'] <= 1 for v in views if v['kind'] == 'book' and v['count']),
            planned_multileaf_bank_le_twice_exposed_source=price['multi_leaf_bank_bytes'] <= 2 * price['source_exposed_bank_bytes'])
        decision = 'ELIGIBLE_ONE_MATCHED_LEAF_FUNCTION_TEST_NOT_QUALITY_PASS' if all(eligibility.values()) else 'DEPTH2_RADIAL_INFORMATION_TREE_CRITERION_CLOSED'
        assert eligibility == raw['eligibility'] and decision == raw['decision'] and len(raw['gates']) == 6
        ctx.r['gates']['ALL656_cost_views_dimensional_prices_full_fallback_and_decisions_independently_reconstructed'] = True
        write(ctx.out / 'verified_tree_information.json', dict(main_sha256=args.main_sha, gain_panels=verified_nodes,
            gains_checked=verified_candidates, maximum_gain_bound_ratio=maximum_gain_ratio, selected_predicate_dot_checks=dot_checks))
        ctx.finish(dict(main_sha256=args.main_sha, cases=freeze['cases'], occupancy=occupancy, views=views, price=price,
            eligibility=eligibility, decision=decision, verified_gain_panels=verified_nodes, verified_candidates=verified_candidates,
            maximum_gain_bound_ratio=maximum_gain_ratio, selected_source_predicate_dot_checks=dot_checks,
            source_response_function_calls=0, full_signed_WI_projection_rows=0, continuous_source_shadows=0,
            candidate_function_vectors=0, native_calls=0, model_calls=0, readout_fits=0, physical_DRAM_verified=False,
            scope='Independent finite dev gain, selected source routing and priced fallback audit only; no function-quality or useful large-n claim.'))
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
