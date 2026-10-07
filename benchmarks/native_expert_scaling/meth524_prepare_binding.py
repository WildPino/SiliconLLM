"""Bind audited 523 numerical data, retaining its failed original time gate."""
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth524_operations import BIND, DOC, ROOT, Meter, write


def main():
    ctx = None
    try:
        ctx = Meter('binding', 180, 512 << 20); catalog = {}
        def item(path, expected=None):
            p = Path(path).resolve(); sha = ctx.digest(p)
            if expected: assert sha == expected, str(p)
            v = dict(path=str(p), bytes=p.stat().st_size, sha256=sha); catalog[str(p)] = v; return v
        a = json.loads(Path(item(DOC / 'ADMISSION_523_20261007.json',
            '246b21ea1725af54199f1ccf58b45bc7b20ec19c86be4157a084f9735027b939')['path']).read_bytes())
        assert not a['main_completed'] and a['main_numerical_prefix_completed'] and a['independent_audit_completed']
        assert not a['gates']['original_main_900s_envelope']
        assert all(v for k, v in a['gates'].items() if k != 'original_main_900s_envelope')
        for key in ('raw', 'audit', 'original_failure', 'metadata_repair', 'retention'):
            item(a[key]['path'], a[key]['sha256'])
        old = json.loads(Path(item(DOC / 'meth523_binding.json',
            '7cced083d598522bef6b9c382b96b5761e4c26f37130e814241572c5d9603b0e')['path']).read_bytes())
        data = {k: item(old['data'][k]['path'], old['data'][k]['sha256']) for k in ('uid', 'occurrences', 'inputs')}
        raw = json.loads(Path(a['raw']['path']).read_bytes())
        for key in ('signed_dots', 'predictions'):
            v = next(v for v in raw['output_inventory'] if Path(v['path']).name == key + '.npy')
            data[key] = item(v['path'], v['sha256'])
        # Names are source tensor identifiers; select by exact parent metadata.
        names = {p['wi']['name'] for p in old['parents']}
        extents = [v for v in old['source_extents'] if v['name'].removesuffix(':codes').removesuffix(':scales') in names]
        assert len(extents) == 254 and sum(v['bytes'] for v in extents) == 301191168
        for v in extents: assert ctx.digest(v['path'], v['offset'], v['bytes']) == v['sha256']
        runtime = [item(v['path'], v['sha256']) for v in old['runtime_files']]
        sources = sorted((ROOT / 'benchmarks/native_expert_scaling').glob('meth524*'))
        sources += [DOC / 'METH_524_SOURCE_HYPERPLANE_TREE_PROTOCOL_20261007.md']
        scientific = []
        for p in sources:
            rel = p.relative_to(ROOT).as_posix(); saved = subprocess.check_output(['git', 'show', 'HEAD:' + rel], cwd=ROOT)
            assert p.read_bytes().replace(b'\r\n', b'\n') == saved.replace(b'\r\n', b'\n'); scientific.append(item(p))
        for rel, sha in old['preserved'].items(): item(ROOT / rel, sha)
        assert sys.version == old['python'] and str(Path(sys.executable).resolve()) == old['executable']
        ctx.r['gates'].update(audited_original523_prefix_with_time_failure_explicit=True,
            fresh_used_vectors_inputs_ALL254_WI_extents_runtime_and_committed_science=True,
            no_gain_tree_or_new_function_observation=True)
        write(BIND, {**ctx.r, 'freeze_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'python': sys.version, 'executable': old['executable'], 'packages': old['packages'],
            'catalog': list(catalog.values()), 'data': data, 'payload': old['payload'], 'parents': old['parents'],
            'payload_stat': old['payload_stat'], 'source_extents': extents, 'runtime_files': runtime,
            'scientific': scientific, 'preserved': old['preserved'],
            'limits': {'binding': [180, 512 << 20], 'main': [180, 512 << 20], 'audit': [240, 512 << 20],
                       'combined_output_bytes': 64 << 20}, 'resource_before_serialization': ctx.resources()})
        sha = ctx.digest(BIND)
        write(ROOT / 'results/native_expert_scaling/meth524_binding_resource.json',
              dict(binding_sha256=sha, process_instance=ctx.r['process_instance'], **ctx.resources()))
        ctx.guard(); ctx.timer.cancel(); print(json.dumps(dict(binding_sha256=sha, resource=ctx.resources())), flush=True)
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
