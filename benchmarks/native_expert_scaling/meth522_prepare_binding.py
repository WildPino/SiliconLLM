"""Fresh used support/runtime metadata; no new support cardinality or clique."""
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth522_operations import BIND, DOC, ROOT, Meter, write


def main():
    ctx = None
    try:
        ctx = Meter('binding', 60, 128 << 20); catalog = {}
        def item(path, expected=None):
            p = Path(path).resolve(); sha = ctx.digest(p)
            if expected: assert sha == expected, str(p)
            v = {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha}; catalog[str(p)] = v; return v
        refs = {'ADMISSION_500_20261006.json': '03e5078073f0a78958d80662de3890fec38be4c0ef925ffc72e228a926942558',
                'ADMISSION_501_20261006.json': 'cd8cc8f6800c1e59f4b2a0fecdf43b2c763868542e9b37479b7439af8ed6e931',
                'ADMISSION_502_20261006.json': '34326908fba8582ede30350b9c97b31c6b25de03b5e95b7a6968e0f74a8ab402',
                'ADMISSION_503_20261006.json': '6979e3b371bbb8ac8cdfa9bc0718aa318aab5c644193fc2b079af03a612ec175',
                'ADMISSION_521_20261007.json': '805197715cf05b71021b328063ae94513f61d882d187740a31e7080987ab95f8'}
        for name, sha in refs.items():
            item(DOC / name, sha); adm = json.loads((DOC / name).read_bytes())
            assert adm['main_completed'] and all(adm['gates'].values())
            for key in ('raw', 'retention'): item(adm[key]['path'], adm[key]['sha256'])
        old = json.loads((DOC / 'meth501_binding.json').read_bytes()); item(DOC / 'meth501_binding.json')
        recentref = item(DOC / 'meth521_binding.json', '97990040545bc2669431390e7f1221b52a998c08b90a5ea91db60a858856076d')
        recent = json.loads(Path(recentref['path']).read_bytes())
        data = {k: item(old['data'][k]['path'], old['data'][k]['sha256']) for k in ('uid', 'codes', 'metrics')}
        v = recent['data']['occurrences']; data['occurrences'] = item(v['path'], v['sha256'])
        runtime = [item(v['path'], v['sha256']) for v in recent['runtime_files']]
        scientific = []
        for p in sorted((ROOT / 'benchmarks/native_expert_scaling').glob('meth522*')) + [DOC / 'METH_522_REDUNDANCY_LOWER_BOUND_PROTOCOL_20261007.md']:
            rel = p.relative_to(ROOT).as_posix(); saved = subprocess.check_output(['git', 'show', 'HEAD:' + rel], cwd=ROOT)
            assert p.read_bytes().replace(b'\r\n', b'\n') == saved.replace(b'\r\n', b'\n'); scientific.append(item(p))
        for rel, sha in recent['preserved'].items(): item(ROOT / rel, sha)
        assert sys.version == recent['python'] and str(Path(sys.executable).resolve()) == recent['executable']
        ctx.r['gates'].update(admitted500_source_support_and501_502_503_recipe_521_closure_anchors=True,
                              fresh_ALL_used_support_metadata_runtime_and_frozen_science_before_NumPy=True,
                              no_source_payload_weights_labels_compiler_native_or_support_observation=True)
        value = {**ctx.r, 'experiment': 'METH522 exact incompatibility clique copy lower bound',
                 'freeze_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                 'python': sys.version, 'executable': str(Path(sys.executable).resolve()), 'packages': recent['packages'],
                 'catalog': list(catalog.values()), 'data': data, 'runtime_files': runtime, 'scientific': scientific,
                 'preserved': recent['preserved'], 'limits': {'binding': [60, 128 << 20], 'main': [60, 512 << 20],
                     'audit': [60, 512 << 20], 'metadata': [60, 128 << 20], 'combined_output_bytes': 16 << 20},
                 'resource_before_serialization': ctx.resources()}
        write(BIND, value); sha = ctx.digest(BIND)
        write(ROOT / 'results/native_expert_scaling/meth522_binding_resource.json',
              {'binding_sha256': sha, 'process_instance': ctx.r['process_instance'], **ctx.resources()})
        ctx.guard(); ctx.timer.cancel(); print(json.dumps({'binding_sha256': sha, 'resource': ctx.resources()}), flush=True)
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
