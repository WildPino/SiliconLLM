"""Bind admitted labels and exact source extents before new signed-margin data."""
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth523_operations import BIND, DOC, ROOT, Meter, write


def main():
    ctx = None
    try:
        ctx = Meter('binding', 180, 512 << 20); catalog = {}
        def item(path, expected=None):
            p = Path(path).resolve(); sha = ctx.digest(p)
            if expected: assert sha == expected, str(p)
            v = dict(path=str(p), bytes=p.stat().st_size, sha256=sha); catalog[str(p)] = v; return v
        for name, sha in (
            ('ADMISSION_500_20261006.json', '03e5078073f0a78958d80662de3890fec38be4c0ef925ffc72e228a926942558'),
            ('ADMISSION_521_20261007.json', '805197715cf05b71021b328063ae94513f61d882d187740a31e7080987ab95f8'),
            ('ADMISSION_522_20261007.json', 'cd8a253fac64cef9b25f7f49105752677a491ffaddf841756be871a67490fd60')):
            v = item(DOC / name, sha); a = json.loads(Path(v['path']).read_bytes())
            assert a['main_completed'] and all(a['gates'].values())
            for key in ('raw', 'retention'): item(a[key]['path'], a[key]['sha256'])
        old = json.loads(Path(item(DOC / 'meth521_binding.json', '97990040545bc2669431390e7f1221b52a998c08b90a5ea91db60a858856076d')['path']).read_bytes())
        data = {k: item(old['data'][k]['path'], old['data'][k]['sha256']) for k in ('uid', 'occurrences', 'inputs', 'unweighted', 'targets')}
        raw = json.loads((DOC / 'meth500_overlap_result.json').read_bytes())
        for key, name in (('hidden', 'hidden.npy'), ('codes', 'hidden_codes.npy'), ('scales', 'hidden_scales.npy')):
            v = next(v for v in raw['output_inventory'] if Path(v['path']).name == name)
            data[key] = item(v['path'], v['sha256'])
        export = item(DOC / 'meth380_switch_base128_export_result.json', '6be77281f7ecca73c47eab7f318fed2cb27699f18071b3f3781ab9bce54881ee')
        entries = json.loads(Path(export['path']).read_bytes())['tensors']
        parents = old['parents'][1:]; assert [v['parent'] for v in parents] == list(range(1, 128))
        extents = []
        for p in parents:
            for organ, shape in (('wi', [3072, 768]), ('wo', [768, 3072])):
                v = p[organ]; assert v == {'name': v['name'], **entries[v['name']]}
                assert v['shape'] == shape and v['encoding'] == 1
                for scale in (False, True):
                    off, size, sha = (v['scale_offset'], v['scale_bytes'], v['scale_sha256']) if scale else (v['offset'], v['bytes'], v['sha256'])
                    assert ctx.digest(old['payload']['path'], off, size) == sha
                    extents.append(dict(path=old['payload']['path'], offset=off, bytes=size, sha256=sha, name=v['name'] + (':scales' if scale else ':codes')))
        assert len(extents) == 508 and sum(v['bytes'] for v in extents) == 601211904
        runtime = [item(v['path'], v['sha256']) for v in old['runtime_files']]
        sources = sorted((ROOT / 'benchmarks/native_expert_scaling').glob('meth523*'))
        sources += [DOC / 'METH_523_SOURCE_FOLDING_HINGES_PROTOCOL_20261007.md']
        scientific = []
        for p in sources:
            rel = p.relative_to(ROOT).as_posix(); saved = subprocess.check_output(['git', 'show', 'HEAD:' + rel], cwd=ROOT)
            assert p.read_bytes().replace(b'\r\n', b'\n') == saved.replace(b'\r\n', b'\n'); scientific.append(item(p))
        for rel, sha in old['preserved'].items(): item(ROOT / rel, sha)
        assert sys.version == old['python'] and str(Path(sys.executable).resolve()) == old['executable']
        ctx.r['gates'].update(admitted_original500_labels_and521_522_closures=True,
            fresh_ALL_used_source508_extents_inputs_hidden_labels_runtime_and_committed_science=True,
            no_new_signed_margin_fold_selection_or_function_observation=True)
        value = {**ctx.r, 'freeze_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'python': sys.version, 'executable': old['executable'], 'packages': old['packages'],
            'catalog': list(catalog.values()), 'data': data, 'payload': old['payload'], 'parents': parents,
            'payload_stat': dict(bytes=Path(old['payload']['path']).stat().st_size, mtime_ns=Path(old['payload']['path']).stat().st_mtime_ns),
            'source_extents': extents, 'runtime_files': runtime, 'scientific': scientific, 'preserved': old['preserved'],
            'limits': {'binding': [180, 512 << 20], 'main': [900, 1536 << 20], 'audit': [1200, 1536 << 20],
                'combined_output_bytes': 2 << 30}, 'resource_before_serialization': ctx.resources()}
        write(BIND, value); sha = ctx.digest(BIND)
        write(ROOT / 'results/native_expert_scaling/meth523_binding_resource.json',
            dict(binding_sha256=sha, process_instance=ctx.r['process_instance'], **ctx.resources()))
        ctx.guard(); ctx.timer.cancel(); print(json.dumps(dict(binding_sha256=sha, resource=ctx.resources())), flush=True)
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
