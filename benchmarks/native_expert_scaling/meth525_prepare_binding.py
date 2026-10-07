"""Freeze actual F32 router/norm, source WI and previously admitted anchor data."""
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth525_operations import BIND, DOC, ROOT, Meter, write


def main():
    ctx = None
    try:
        ctx = Meter('binding', 180, 512 << 20); catalog = {}
        def item(path, expected=None):
            p = Path(path).resolve(); sha = ctx.digest(p)
            if expected: assert sha == expected, str(p)
            v = dict(path=str(p), bytes=p.stat().st_size, sha256=sha); catalog[str(p)] = v; return v
        a = json.loads(Path(item(DOC / 'ADMISSION_524_20261007.json',
            'ddab3001bfc4419991fef0c87d66710637689495ea46215f477717eaf0173be1')['path']).read_bytes())
        assert a['main_completed'] and a['independent_audit_completed'] and all(a['gates'].values())
        assert a['decision'] == 'DEPTH2_RADIAL_INFORMATION_TREE_CRITERION_CLOSED'
        for key in ('raw', 'audit', 'retention'): item(a[key]['path'], a[key]['sha256'])
        oldref = item(DOC / 'meth523_binding.json', '7cced083d598522bef6b9c382b96b5761e4c26f37130e814241572c5d9603b0e')
        old = json.loads(Path(oldref['path']).read_bytes())
        data = {k: item(old['data'][k]['path'], old['data'][k]['sha256']) for k in ('uid', 'occurrences', 'inputs')}
        raw = json.loads(Path(item(DOC / 'meth523_main_result.json',
            '0688e17034df2ae58e88dc76eba06b39509ad705cdc7fca77e62d3d0d4c71cd2')['path']).read_bytes())
        audited = json.loads(Path(item(DOC / 'ADMISSION_523_20261007.json',
            '246b21ea1725af54199f1ccf58b45bc7b20ec19c86be4157a084f9735027b939')['path']).read_bytes())
        assert audited['main_numerical_prefix_completed'] and audited['independent_audit_completed'] and not audited['main_completed']
        for key in ('anchors', 'hinges', 'signs'):
            v = next(v for v in raw['output_inventory'] if Path(v['path']).name == key + '.npy')
            data[key] = item(v['path'], v['sha256'])
        retained = json.loads(Path(item(DOC / 'meth493_r2_binding.json')['path']).read_bytes())
        v = retained['data']['unique_inputs.bin']; assert v['sha256'] == 'f80e46251c7a13ea86483883f49cfa3d15ceeff08ad95fe9b7928a3debfa1c57'
        data['canonical_scores'] = item(v['path'], v['sha256'])
        export = json.loads(Path(item(DOC / 'meth380_switch_base128_export_result.json',
            '6be77281f7ecca73c47eab7f318fed2cb27699f18071b3f3781ab9bce54881ee')['path']).read_bytes())
        organs = {k: dict(name=name, **export['tensors'][name]) for k, name in (
            ('router', 'decoder.block.11.layer.2.mlp.router.classifier.weight'),
            ('norm', 'decoder.block.11.layer.2.layer_norm.weight'))}
        assert organs['router']['shape'] == [128, 768] and organs['norm']['shape'] == [768]
        assert all(v['encoding'] == 0 for v in organs.values())
        names = {p['wi']['name'] for p in old['parents']}
        extents = [v for v in old['source_extents'] if v['name'].removesuffix(':codes').removesuffix(':scales') in names]
        assert len(extents) == 254
        for name, v in organs.items(): extents.append(dict(path=old['payload']['path'], offset=v['offset'], bytes=v['bytes'], sha256=v['sha256'], name=v['name']))
        for v in extents: assert ctx.digest(v['path'], v['offset'], v['bytes']) == v['sha256']
        runtime = [item(v['path'], v['sha256']) for v in old['runtime_files']]
        source_contracts = [item(ROOT / 'benchmarks/native_expert_scaling' / name) for name in (
            'meth393_switch_router_audit.c', 'meth393_switch_router_trace.h', 'meth479_source_router.c',
            'meth461_switch_common_input.c', 'meth524_contract.py', 'meth524_operations.py')]
        scientific = []
        for p in sorted((ROOT / 'benchmarks/native_expert_scaling').glob('meth525*')) + [DOC / 'METH_525_ROUTER_NEUTRAL_GEOMETRY_PROTOCOL_20261007.md']:
            rel = p.relative_to(ROOT).as_posix(); saved = subprocess.check_output(['git', 'show', 'HEAD:' + rel], cwd=ROOT)
            assert p.read_bytes().replace(b'\r\n', b'\n') == saved.replace(b'\r\n', b'\n'); scientific.append(item(p))
        for rel, sha in old['preserved'].items(): item(ROOT / rel, sha)
        assert sys.version == old['python'] and str(Path(sys.executable).resolve()) == old['executable']
        ctx.r['gates'].update(admitted523_prefix_and524_fixed_failure_before_new_geometry=True,
            fresh_original_cached_scores_anchor_wire_ALL_WI_router_norm_extents_runtime_and_frozen_source=True,
            no_geometry_rank_or_new_router_or_source_function_observation=True)
        write(BIND, {**ctx.r, 'freeze_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'python': sys.version, 'executable': old['executable'], 'packages': old['packages'],
            'catalog': list(catalog.values()), 'data': data, 'payload': old['payload'], 'parents': old['parents'],
            'payload_stat': old['payload_stat'], 'source_extents': extents, 'runtime_files': runtime,
            'source_contracts': source_contracts, 'organs': organs, 'scientific': scientific, 'preserved': old['preserved'],
            'limits': {'binding': [180, 512 << 20], 'main': [180, 512 << 20], 'audit': [240, 512 << 20], 'combined_output_bytes': 32 << 20},
            'resource_before_serialization': ctx.resources()})
        sha = ctx.digest(BIND)
        write(ROOT / 'results/native_expert_scaling/meth525_binding_resource.json',
              dict(binding_sha256=sha, process_instance=ctx.r['process_instance'], **ctx.resources()))
        ctx.guard(); ctx.timer.cancel(); print(json.dumps(dict(binding_sha256=sha, resource=ctx.resources())))
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
