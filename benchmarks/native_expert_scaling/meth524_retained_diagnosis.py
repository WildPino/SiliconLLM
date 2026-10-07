"""Post-hoc JSON-only copy of failed roots and admitted source metric evidence."""
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth524_operations import DOC, ROOT, Meter, write


def main():
    ctx = None
    try:
        ctx = Meter('retained_diagnosis', 60, 128 << 20)
        refs = []
        for name, sha in (
            ('ADMISSION_523_20261007.json', '246b21ea1725af54199f1ccf58b45bc7b20ec19c86be4157a084f9735027b939'),
            ('meth523_main_result.json', '0688e17034df2ae58e88dc76eba06b39509ad705cdc7fca77e62d3d0d4c71cd2'),
            ('ADMISSION_524_20261007.json', 'ddab3001bfc4419991fef0c87d66710637689495ea46215f477717eaf0173be1'),
            ('meth524_main_result.json', 'b70ae865aecd95ea00feefd7de1f276d66073c4b717c1a846db82d3381b3f1de'),
            ('meth524_audit_result.json', 'a5dbf23fe11667fbaa8ffa81a6546e303e1a5cec188e3a61875b982520eaf7e3')):
            p = DOC / name; assert ctx.digest(p) == sha
            refs.append(dict(path=str(p), bytes=p.stat().st_size, sha256=sha))
        old = json.loads((DOC / 'meth523_main_result.json').read_bytes())
        raw = json.loads((DOC / 'meth524_main_result.json').read_bytes())
        audit = json.loads((DOC / 'meth524_audit_result.json').read_bytes())
        assert all(audit['gates'].values()) and audit['cases'] == raw['cases']
        scientific = []
        for p in (Path(__file__).resolve(), DOC / 'METH_524_RETAINED_DIAGNOSIS_PROTOCOL_20261007.md'):
            rel = p.relative_to(ROOT).as_posix(); saved = subprocess.check_output(['git', 'show', 'HEAD:' + rel], cwd=ROOT)
            assert p.read_bytes().replace(b'\r\n', b'\n') == saved.replace(b'\r\n', b'\n')
            scientific.append(dict(path=str(p), sha256=ctx.digest(p)))
        cases = [c for c in raw['cases'][1:] if not c['fallback_in_future_function_test']]
        failed = [c for c in cases if c['nodes'][0]['status'] != 'split']
        rows = []
        for c in failed:
            root = c['nodes'][0]; entries = []
            for v in old['views']:
                if v['kind'] != 'parent' or v['expert'] != c['expert']: continue
                entries.append(dict(split=v['split'], count=v['count'],
                    unweighted_source_energy=v['unweighted_source_energy'], weighted_source_energy=v['weighted_source_energy'],
                    unweighted_omitted_sign_energy=v['sums'][8], weighted_omitted_sign_energy=v['sums'][24],
                    unweighted_continuous_hybrid_RMS=v['unweighted_continuous_hybrid_RMS'],
                    unweighted_physical_reference_RMS=v['unweighted_physical_reference_RMS'],
                    weighted_physical_reference_RMS=v['weighted_physical_reference_RMS']))
            assert len(entries) == 2
            rows.append(dict(expert=c['expert'], development=c['development'], status=root['status'],
                balanced_candidates=root['balanced_candidates'], unique_partitions=root['unique_partitions'],
                leading_neuron=root['leading_neuron'], leading_panel=root['leading_panel'], qualified523_parent_views=entries))
        ctx.r['gates'].update(ALL_known_523_524_admission_and_original_records_SHA=True,
            committed_JSON_only_diagnosis_source_no_arrays_or_new_choices=True,
            copied_failed_root_and_prior_parent_fields_without_scientific_replay=True)
        ctx.r.update(experiment='METH524_RETAINED_DIAGNOSIS', post_hoc=True, parents=rows, eligible_parent_count=len(cases),
            split_root_count=len(cases) - len(failed), input_records=refs, scientific=scientific,
            freeze_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            original_524_decision=raw['decision'], decision_changed=False,
            consumed_rows_used_for_new_selection=0, array_reads=0, new_gain_route_function_metric_or_source_calls=0,
            inference='A residual-driven dev split can have no signal despite unseen source-function error. This does not prove the full source routing cone is covered.',
            resource_before_serialization=ctx.resources())
        write(ctx.raw, ctx.r); sha = ctx.digest(ctx.raw)
        write(ROOT / 'results/native_expert_scaling/meth524_retained_diagnosis_resource.json',
              dict(result_sha256=sha, process_instance=ctx.r['process_instance'], **ctx.resources()))
        ctx.guard(); ctx.timer.cancel(); print(json.dumps(dict(result_sha256=sha, parents=len(rows), resource=ctx.resources())))
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
