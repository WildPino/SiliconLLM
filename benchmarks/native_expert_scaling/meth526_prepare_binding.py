"""Bind the admitted525 representation; no numerical certificate calculation."""
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth526_operations import BIND, DOC, ROOT, Meter, write


def main():
    ctx = None
    try:
        ctx = Meter('binding', 180, 512 << 20); catalog = {}
        def item(path, expected=None):
            p = Path(path).resolve(); sha = ctx.digest(p)
            if expected: assert sha == expected, str(p)
            v = dict(path=str(p), bytes=p.stat().st_size, sha256=sha); catalog[str(p)] = v; return v
        admission = json.loads(Path(item(DOC / 'ADMISSION_525_20261007.json',
            '78db99d3c33ae8afa85d0f9c89df7679852927e28c93fc21db4e0316bffc66f3')['path']).read_bytes())
        assert admission['main_completed'] and admission['independent_audit_completed'] and all(admission['gates'].values())
        assert admission['decision'] == 'ROUTER_NEUTRAL_HALF_ENDPOINT_GEOMETRY_CRITERION_CLOSED'
        assert admission['constructed_queries'] == 109 and admission['verified_source_planes'] == 390144
        refs = {k: item(admission[k]['path'], admission[k]['sha256']) for k in ('binding', 'raw', 'audit', 'retention', 'windows')}
        old = json.loads(Path(refs['binding']['path']).read_bytes()); main_raw = json.loads(Path(refs['raw']['path']).read_bytes())
        for v in main_raw['output_inventory']: item(v['path'], v['sha256'])
        terminal = item(ROOT / 'results/native_expert_scaling/meth525_geometry/terminal_resource.json')
        assert json.loads(Path(terminal['path']).read_bytes())['result_sha256'] == refs['raw']['sha256']
        q = next(v for v in main_raw['output_inventory'] if Path(v['path']).name == 'basis.npy')
        projection = main_raw['projection']
        assert projection['kept_rows'] == list(range(128)) and projection['rank'] == 128 and projection['null_dimensions'] == 640
        assert projection['minimum_singular_lower'] > 0
        extents = []
        for v in old['organs'].values():
            extents.append(dict(path=old['payload']['path'], offset=v['offset'], bytes=v['bytes'], sha256=v['sha256'], name=v['name']))
        assert len(extents) == 2 and sum(v['bytes'] for v in extents) == 396288
        for v in extents: assert ctx.digest(v['path'], v['offset'], v['bytes']) == v['sha256']
        runtime = [item(v['path'], v['sha256']) for v in old['runtime_files']]
        for v in old['scientific']: item(v['path'], v['sha256'])
        scientific = []
        for p in sorted((ROOT / 'benchmarks/native_expert_scaling').glob('meth526*')) + [DOC / 'METH_526_EXACT_PROJECTOR_PROTOCOL_20261007.md']:
            rel = p.relative_to(ROOT).as_posix(); saved = subprocess.check_output(['git', 'show', 'HEAD:' + rel], cwd=ROOT)
            assert p.read_bytes().replace(b'\r\n', b'\n') == saved.replace(b'\r\n', b'\n'); scientific.append(item(p))
        for rel, sha in old['preserved'].items(): item(ROOT / rel, sha)
        assert sys.version == old['python'] and str(Path(sys.executable).resolve()) == old['executable']
        ctx.r['gates'].update(admitted525_fixed_failure_and_ALL109_physical_records_preserved=True,
            fresh_retained_Q_actual_router_norm_runtime_and_committed_source_before_integer_values=True,
            no_exact_encoding_Gram_residual_or_new_query_observation=True)
        write(BIND, {**ctx.r, 'freeze_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'python': sys.version, 'executable': old['executable'], 'packages': old['packages'],
            'catalog': list(catalog.values()), 'data': {'basis': q}, 'legacy': refs,
            'legacy_projection': projection, 'legacy_output_inventory': main_raw['output_inventory'],
            'payload': old['payload'], 'payload_stat': old['payload_stat'], 'organs': old['organs'],
            'source_extents': extents, 'runtime_files': runtime, 'scientific': scientific, 'preserved': old['preserved'],
            'limits': {'binding': [180, 512 << 20], 'main': [180, 512 << 20], 'audit': [240, 512 << 20], 'combined_output_bytes': 32 << 20},
            'resource_before_serialization': ctx.resources()})
        sha = ctx.digest(BIND)
        write(ROOT / 'results/native_expert_scaling/meth526_binding_resource.json',
            dict(binding_sha256=sha, process_instance=ctx.r['process_instance'], **ctx.resources()))
        ctx.guard(); ctx.timer.cancel(); print(json.dumps(dict(binding_sha256=sha, resource=ctx.resources())))
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
