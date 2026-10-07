"""Bind original527 scalar inputs and admitted525/526; no norm/energy observations."""
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth527_operations import BIND, DOC, ROOT, Meter, write


def main():
    ctx = None
    try:
        ctx = Meter('binding', 180, 512 << 20); catalog = {}
        def item(path, expected=None):
            p = Path(path).resolve(); sha = ctx.digest(p)
            if expected: assert sha == expected, str(p)
            v = dict(path=str(p), bytes=p.stat().st_size, sha256=sha); catalog[str(p)] = v; return v
        refs = {}
        for stage, sha in ((525, '78db99d3c33ae8afa85d0f9c89df7679852927e28c93fc21db4e0316bffc66f3'),
                           (526, '50f5faa4bd1d605d41d85313c720affae2e9c42e13c9cce62181beba62253294')):
            admission = json.loads(Path(item(DOC / f'ADMISSION_{stage}_20261007.json', sha)['path']).read_bytes())
            assert admission['main_completed'] and admission['independent_audit_completed'] and all(admission['gates'].values())
            refs[str(stage)] = {k: item(admission[k]['path'], admission[k]['sha256']) for k in ('binding', 'raw', 'audit', 'retention', 'windows')}
        old = json.loads(Path(refs['525']['binding']['path']).read_bytes())
        geometry = json.loads(Path(refs['525']['raw']['path']).read_bytes())
        cert = json.loads(Path(refs['526']['raw']['path']).read_bytes())
        assert geometry['decision'] == 'ROUTER_NEUTRAL_HALF_ENDPOINT_GEOMETRY_CRITERION_CLOSED' and geometry['constructed_queries'] == 109
        assert all(cert['eligibility'].values()) and cert['decision'] == 'EXACT_PROJECTOR_CERTIFICATE_ELIGIBLE_FOR_BOUND_REFINEMENT_NOT_QUERY_PROMOTION'
        legacy_outputs = geometry['output_inventory'] + cert['output_inventory']
        for v in legacy_outputs: item(v['path'], v['sha256'])
        for stage, folder in ((525, 'meth525_geometry'), (526, 'meth526_exact_projector')):
            v = item(ROOT / 'results/native_expert_scaling' / folder / 'terminal_resource.json')
            assert json.loads(Path(v['path']).read_bytes())['result_sha256'] == refs[str(stage)]['raw']['sha256']
        names = {v['wi']['name'] for v in old['parents']}
        extents = [v for v in old['source_extents'] if v['name'].endswith(':codes') and v['name'].removesuffix(':codes') in names]
        assert len(extents) == 127 and sum(v['bytes'] for v in extents) == 299630592
        norm = old['organs']['norm']; extents.append(dict(path=old['payload']['path'], offset=norm['offset'], bytes=norm['bytes'], sha256=norm['sha256'], name=norm['name']))
        for v in extents: assert ctx.digest(v['path'], v['offset'], v['bytes']) == v['sha256']
        runtime = [item(v['path'], v['sha256']) for v in old['runtime_files']]
        old526 = json.loads(Path(refs['526']['binding']['path']).read_bytes())
        for v in old['scientific'] + old526['scientific']: item(v['path'], v['sha256'])
        scientific = []
        for p in sorted((ROOT / 'benchmarks/native_expert_scaling').glob('meth527*')) + [DOC / 'METH_527_RETAINED_BOUND_PROTOCOL_20261007.md']:
            saved = subprocess.check_output(['git', 'show', 'HEAD:' + p.relative_to(ROOT).as_posix()], cwd=ROOT)
            assert p.read_bytes().replace(b'\r\n', b'\n') == saved.replace(b'\r\n', b'\n'); scientific.append(item(p))
        for rel, sha in old['preserved'].items(): item(ROOT / rel, sha)
        assert sys.version == old['python'] and str(Path(sys.executable).resolve()) == old['executable']
        ctx.r['gates'].update(admitted525_fixed_failure_AND526_exact_bound_before_refinement=True,
            ALL_original_panels_query_and_certificate_records_SHA_preserved=True,
            fresh_actual_norm_ALL127_WI_code_extents_runtime_and_committed_source_without_new_values=True)
        write(BIND, {**ctx.r, 'freeze_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'python': sys.version, 'executable': old['executable'], 'packages': old['packages'], 'catalog': list(catalog.values()),
            'data': {}, 'legacy': refs, 'legacy_cases': geometry['cases'], 'legacy_output_inventory': legacy_outputs,
            'exact_certificate': cert['exact_certificate'], 'payload': old['payload'], 'payload_stat': old['payload_stat'],
            'parents': old['parents'], 'organs': old['organs'], 'source_extents': extents, 'runtime_files': runtime,
            'scientific': scientific, 'preserved': old['preserved'],
            'limits': {'binding': [180, 512 << 20], 'main': [180, 512 << 20], 'audit': [240, 512 << 20], 'combined_output_bytes': 32 << 20},
            'resource_before_serialization': ctx.resources()})
        sha = ctx.digest(BIND)
        write(ROOT / 'results/native_expert_scaling/meth527_binding_resource.json',
            dict(binding_sha256=sha, process_instance=ctx.r['process_instance'], **ctx.resources()))
        ctx.guard(); ctx.timer.cancel(); print(json.dumps(dict(binding_sha256=sha, resource=ctx.resources())))
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
