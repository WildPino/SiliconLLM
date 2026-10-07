"""One depth2 development tree; route occupancy and priced fallback afterward."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth524_operations import Context, write


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--binding-sha', required=True); args = ap.parse_args(); ctx = None
    try:
        ctx = Context('main', args.binding_sha); np = ctx.numpy()
        import meth524_contract as S
        import meth524_math as M
        write(ctx.out / 'controls.json', M.controls())
        ctx.r['gates']['NEW_radial_gain_complement_tie_zero_sign_and_mass_controls'] = True
        m, occ, inputs, dev, counts = S.load(ctx)
        ctx.r['gates']['ALL_UID_original_parent_probability_role_and_occurrence_contracts'] = True
        cases = [dict(expert=0, development=0, status='UNEXPOSED_UNSUPPORTED', nodes=[])]
        panel_refs = []; copied = 0
        with (ctx.out / 'source_test_rows.bin').open('xb') as rowfile:
            for e in range(1, 128):
                di = np.flatnonzero(dev & (m[:, 3] == e))
                dots = S.saved(ctx, 'signed_dots', di)
                f = S.saved(ctx, 'predictions', di, 0); g = S.saved(ctx, 'predictions', di, 2)
                r = f - g; q = inputs['q'][di].astype('<i8')
                norm2 = np.sum(q * q, axis=1, dtype=np.int64)
                rho = inputs['alpha'][di].astype('<f8') * np.sqrt(norm2.astype('<f8'))
                p = inputs['p'][di].astype('<f8')
                assert np.all(norm2 >= 0) and np.all(np.abs(dots) <= 768 * 128 * 32767)
                panels = np.zeros((3, 3072, 4), '<f8'); pc = np.zeros((3, 3072), '<u2'); flags = np.zeros((3, 3072), 'u1')
                nodes = []
                def grow(node, local_ids, path):
                    depth = len(path)
                    info = dict(node=node, depth=depth, development_UIDs=di[local_ids].tolist(),
                                path=path, candidate_panel=False)
                    nodes.append(info)
                    if depth == 2: info['status'] = 'leaf_depth_limit'; return
                    if len(local_ids) < 4: info['status'] = 'leaf_fewer_than4_UIDs'; return
                    sign = dots[local_ids] > 0
                    a, c, valid = M.panel(sign, r[local_ids], rho[local_ids], p[local_ids], f[local_ids])
                    panels[node], pc[node], flags[node] = a, c, valid
                    reps = M.representatives(sign, valid, [v['neuron'] for v in path])
                    j, state, leading = M.choose(a, reps)
                    info.update(candidate_panel=True, balanced_candidates=int(valid.sum()), unique_partitions=len(reps), status=state)
                    if leading is not None: info.update(leading_neuron=leading, leading_panel=a[leading].tolist())
                    if j is None: return
                    row = S.source_row(ctx, e, j); assert np.frombuffer(row[768:], dtype='<f4')[0] > 0
                    off = rowfile.tell(); rowfile.write(row)
                    info.update(neuron=j, source_row_offset=off, source_row_bytes=len(row),
                                source_row_sha256=__import__('hashlib').sha256(row).hexdigest())
                    bit = sign[:, j]
                    for positive, child in ((False, 2 * node + 1), (True, 2 * node + 2)):
                        grow(child, local_ids[bit == positive], path + [dict(neuron=j, positive=positive)])
                grow(0, np.arange(len(di)), [])
                nodes.sort(key=lambda v: v['node'])
                case = dict(expert=e, development=len(di), fallback_in_future_function_test=bool(counts[e] < 16),
                            nodes=nodes, leaves=[v['node'] for v in nodes if v['status'] != 'split'])
                assert sorted(uid for v in nodes if v['status'] != 'split' for uid in v['development_UIDs']) == di.tolist()
                cases.append(case); copied += sum(v['status'] == 'split' for v in nodes)
                for name, array in (('gains', panels), ('positive_counts', pc), ('balanced', flags)):
                    out = ctx.out / f'e{e:03d}_{name}.npy'
                    with out.open('xb') as file: np.save(file, array, allow_pickle=False)
                    panel_refs.append(dict(path=str(out), bytes=out.stat().st_size, sha256=ctx.digest(out)))
                ctx.guard()
                if e % 32 == 0: print('development_parent_complete=' + str(e), flush=True)
        freeze_path = ctx.out / 'development_freeze.json'
        write(freeze_path, dict(cases=cases, panels=panel_refs, rule='ONE depth2; BOTH gains; balanced>=2; deduplicate complements; resolved max; threshold1e-12'))
        freeze_sha = ctx.digest(freeze_path)
        ctx.r['gates']['ALL127_development_trees_and_gain_panels_frozen_BEFORE_consumed_routing'] = True
        leaves = np.zeros(S.N, 'u1'); depths = np.zeros(S.N, 'u1'); seen = np.zeros(S.N, bool)
        selected_test_dot_checks = 0; occupancy = []
        rows = (ctx.out / 'source_test_rows.bin').read_bytes()
        for case in cases[1:]:
            e = case['expert']; ix = np.flatnonzero(m[:, 3] == e); dots = S.saved(ctx, 'signed_dots', ix)
            q = inputs['q'][ix].astype('<i8'); here = np.zeros(len(ix), '<u2'); dep = np.zeros(len(ix), 'u1')
            for node in case['nodes']:
                if node['status'] != 'split': continue
                j = node['neuron']; off = node['source_row_offset']; row = rows[off:off + 772]
                # New selected-predicate dot checks ONLY; not a full WI replay.
                exact = q @ np.frombuffer(row[:768], dtype='i1').astype('<i8')
                assert np.array_equal(exact, dots[:, j]); selected_test_dot_checks += len(ix)
                selected = here == node['node']; here[selected] = 2 * node['node'] + 1 + (dots[selected, j] > 0)
                dep[selected] += 1
            assert all(int(t) in case['leaves'] for t in here)
            for node in case['nodes']:
                if node['status'] == 'split': continue
                got = ix[dev[ix] & (here == node['node'])]
                assert got.tolist() == node['development_UIDs']
            leaves[ix], depths[ix], seen[ix] = here.astype('u1'), dep, True
            occupancy.append(dict(expert=e, leaves=[dict(node=t, development=int(np.sum(dev[ix] & (here == t))),
                consumed=int(np.sum((~dev[ix]) & (here == t)))) for t in case['leaves']]))
            ctx.guard()
        assert seen.all() and np.all(depths <= 2) and ctx.digest(freeze_path) == freeze_sha
        expanded = (m[:, 3] * 7 + leaves).astype('<u2'); mass = np.array(inputs['p'], copy=True)
        assert mass.tobytes() == inputs['p'].tobytes() and np.array_equal(expanded // 7, m[:, 3])
        for name, array in (('leaf_ids', leaves), ('depths', depths), ('expanded_ids', expanded), ('selected_parent_mass', mass)):
            with (ctx.out / (name + '.npy')).open('xb') as file: np.save(file, array, allow_pickle=False)
        ctx.r['gates']['ALL17540_selected_source_integer_predicates_routes_dev_UIDs_and_parent_mass_BYTES'] = True
        fallback = counts[m[:, 3]] < 16
        active_bytes = np.where(fallback, S.SOURCE_BYTES, S.LEAF_BYTES + depths.astype('<i8') * S.NODE_BYTES)
        active_macs = np.where(fallback, S.SOURCE_MACS, S.LEAF_MACS + depths.astype('<i8') * S.D)
        views = []
        for label, ids in S.domains(m, occ, counts):
            v = {**label, 'count': len(ids), 'fallback_rows': int(np.sum(fallback[ids])),
                 'active_bytes_sum': int(np.sum(active_bytes[ids], dtype=np.int64)),
                 'active_MACs_sum': int(np.sum(active_macs[ids], dtype=np.int64))}
            if len(ids): v.update(logical_active_byte_ratio=v['active_bytes_sum'] / (S.SOURCE_BYTES * len(ids)),
                                logical_active_MAC_ratio=v['active_MACs_sum'] / (S.SOURCE_MACS * len(ids)))
            views.append(v)
        assert len(views) == 656 and sum(v['count'] for v in views if v['kind'] == 'book') == 19962
        fallback_parents = sum(c['fallback_in_future_function_test'] for c in cases[1:])
        eligible_cases = [c for c in cases[1:] if not c['fallback_in_future_function_test']]
        tests = sum(v['status'] == 'split' for c in eligible_cases for v in c['nodes'])
        planned_leaves = sum(len(c['leaves']) for c in eligible_cases)
        price = dict(source_exposed_bank_bytes=127 * S.SOURCE_BYTES, source_fallback_parents=fallback_parents,
            one_leaf_functions=len(eligible_cases), multi_leaf_functions=planned_leaves,
            copied_selected_tests_all_parents=copied, selected_tests_in_future_nonfallback=tests,
            one_leaf_bank_bytes=fallback_parents * S.SOURCE_BYTES + len(eligible_cases) * S.LEAF_BYTES + 128 * 16,
            multi_leaf_bank_bytes=fallback_parents * S.SOURCE_BYTES + planned_leaves * S.LEAF_BYTES + tests * S.NODE_BYTES + 128 * 16,
            node_record_bytes=16, copied_WI_row_bytes=772, header_bytes=128 * 16,
            original_parent_winner_and_full_normalization='retained control, FULL cost still owed',
            actual_leaf_banks_compiled=False, physical_DRAM_verified=False)
        eligibility = dict(
            ALL_nonfallback_parents_resolved_positive_balanced_root=bool(eligible_cases) and all(c['nodes'][0]['status'] == 'split' for c in eligible_cases),
            ALL_nonfallback_node_orders_resolved=all(not v['status'].startswith('unresolved') for c in eligible_cases for v in c['nodes']),
            planned_occupied_nonfallback_leaf_count_increases=planned_leaves > len(eligible_cases),
            ALL_six_logical_active_bytes_le_75pct=all(v['logical_active_byte_ratio'] <= .75 for v in views if v['kind'] == 'role'),
            ALL_nonempty_books_logical_active_bytes_le_source=all(v['logical_active_byte_ratio'] <= 1 for v in views if v['kind'] == 'book' and v['count']),
            planned_multileaf_bank_le_twice_exposed_source=price['multi_leaf_bank_bytes'] <= 2 * price['source_exposed_bank_bytes'])
        decision = 'ELIGIBLE_ONE_MATCHED_LEAF_FUNCTION_TEST_NOT_QUALITY_PASS' if all(eligibility.values()) else 'DEPTH2_RADIAL_INFORMATION_TREE_CRITERION_CLOSED'
        ctx.r['gates']['ALL656_UID_occurrence_cost_views_full_priced_fallback_and_frozen_decisions'] = True
        ctx.finish(dict(cases=cases, occupancy=occupancy, development_freeze_sha256=freeze_sha, views=views, price=price,
            eligibility=eligibility, decision=decision, selected_source_predicate_dot_checks=selected_test_dot_checks,
            source_response_function_calls=0, full_signed_WI_projection_rows=0, continuous_source_shadows=0,
            candidate_function_vectors=0, native_calls=0, model_calls=0, readout_fits=0,
            physical_DRAM_verified=False, scope='Dev radial-residual information screen and routing/cost diagnostics only. No new leaf functions, useful count, kernel, fresh quality/rate or whole goal promotion.'))
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
