"""521 metadata admission and fresh file/extent SHA before source responses."""
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth521_operations import BIND, DOC, ROOT, Meter, utc, write


def main():
    ctx = None
    try:
        ctx = Meter('binding', 180, 512 << 20); catalog = {}
        def item(path, expected=None):
            p = Path(path).resolve(); sha = ctx.digest(p)
            if expected: assert sha == expected, str(p)
            v = {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha}; catalog[str(p)] = v
            return v
        refs = {'meth499_binding.json': 'ce3cf895f4093d221a2d096b7b3246d599f04e4f84b5ce3e0f459b31697a1576',
                'ADMISSION_499_20261006.json': 'a7852bfbc67a695a3696006b5098a1a06afb5e7401a0fb3b045adf4d3a288c6d',
                'meth519_binding.json': '759330331caca87bd6c358ae1f7fd00bc31896ab64ed117569141d1caaa899b0',
                'meth520_binding.json': 'abcabcd44e8adc2e8dce79300663fccbbe1ca3aadfd70e3a1bfa41ff1a2e128a',
                'ADMISSION_520_20261007.json': '4b31002b7dae8d29d43d25a31a90b4b197dedb578888de7ae3379018b88ebab2'}
        for name, sha in refs.items(): item(DOC / name, sha)
        for name in ('ADMISSION_499_20261006.json', 'ADMISSION_520_20261007.json'):
            adm = json.loads((DOC / name).read_bytes()); assert adm['main_completed'] and all(adm['gates'].values())
            for key in ('raw', 'retention'): item(adm[key]['path'], adm[key]['sha256'])
        old = json.loads((DOC / 'meth499_binding.json').read_bytes())
        recent = json.loads((DOC / 'meth520_binding.json').read_bytes())
        source = json.loads((DOC / 'meth519_binding.json').read_bytes())
        data = {k: item(old['data'][k]['path'], old['data'][k]['sha256'])
                for k in ('uid', 'occurrences', 'features', 'geometry', 'targets')}
        raw = json.loads((DOC / 'meth499_factor_fit_result.json').read_bytes())
        for key, name in (('bank', 'bank.bin'), ('inputs', 'inputs.bin'), ('unweighted', 'unweighted_targets.bin'),
                          ('baseline', 'predictions.bin')):
            v = next(v for v in raw['output_inventory'] if Path(v['path']).name == name)
            data[key] = item(v['path'], v['sha256'])
        plan = json.loads((DOC / 'meth520_main_result.json').read_bytes())
        assert all(plan['eligibility'].values()) and plan['selected_pairs'] == 53943
        for key, name in (('pairs', 'query_pairs.npy'), ('augmented', 'augmented_UID_order.npy'),
                          ('provenance', 'development_query_sources.npy')):
            v = next(v for v in plan['output_inventory'] if Path(v['path']).name == name)
            data[key] = item(v['path'], v['sha256'])
        payload = source['payload']; s = Path(payload['path']).stat()
        assert (s.st_size, s.st_mtime_ns) == (payload['bytes'], payload['mtime_ns'])
        extents = []
        for parent in source['parents']:
            for which in ('wi', 'wo'):
                v = parent[which]
                assert v['shape'] == ([3072, 768] if which == 'wi' else [768, 3072]) and v['encoding'] == 1
                for part in ('codes', 'scales'):
                    scale = part == 'scales'; off = v['scale_offset'] if scale else v['offset']
                    length = v['scale_bytes'] if scale else v['bytes']; sha = v['scale_sha256'] if scale else v['sha256']
                    assert ctx.digest(payload['path'], off, length) == sha
                    extents.append({'name': v['name'] + ':' + part, 'path': payload['path'],
                                    'offset': off, 'bytes': length, 'sha256': sha})
        assert len(extents) == 512 and sum(v['bytes'] for v in extents) == 605945856
        runtime = [item(v['path'], v['sha256']) for v in recent['runtime_files']]
        compiler = [item(v['path'], v['sha256']) for v in old['compiler_snapshot']]
        modules = [item(v['path'], v['sha256']) for v in old['native_modules']]
        accounting_module = item(Path(os.environ['SystemRoot']) / 'System32/psapi.dll')
        scientific = []
        sources = sorted((ROOT / 'benchmarks/native_expert_scaling').glob('meth521*'))
        sources += [ROOT / 'benchmarks/native_expert_scaling' / name for name in ('meth498_math.py', 'meth499_math.py')]
        sources += [DOC / 'METH_521_SOURCE_INFORMATION_TRANSFER_PROTOCOL_20261007.md']
        for p in sources:
            rel = p.relative_to(ROOT).as_posix(); saved = subprocess.check_output(['git', 'show', 'HEAD:' + rel], cwd=ROOT)
            assert p.read_bytes().replace(b'\r\n', b'\n') == saved.replace(b'\r\n', b'\n')
            scientific.append(item(p))
        for rel, sha in recent['preserved'].items(): item(ROOT / rel, sha)
        assert sys.version == recent['python'] and str(Path(sys.executable).resolve()) == recent['executable']
        ctx.r['gates'].update(admitted499_function_class_and520_fullrank_fixed_queries=True,
                              fresh_ALL_used_input_label_bank_plan_source512_runtime_compiler_and_science=True,
                              no_new_numeric_source_function_or_model_call=True)
        value = {**ctx.r, 'experiment': 'METH521 source information at fixed feature class',
                 'freeze_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                 'python': sys.version, 'executable': str(Path(sys.executable).resolve()), 'packages': recent['packages'],
                 'catalog': list(catalog.values()), 'data': data, 'payload': payload,
                 'parents': source['parents'], 'source_extents': extents,
                 'scientific': scientific, 'runtime_files': runtime, 'compiler_snapshot': compiler,
                 'compiler': old['compiler'], 'native_modules': modules, 'accounting_module': accounting_module,
                 'preserved': recent['preserved'],
                 'limits': {'binding': [180, 512 << 20], 'main': [900, 2 << 30], 'audit': [900, 2 << 30],
                            'combined_output_bytes': 2 << 30},
                 'resource_before_serialization': ctx.resources(), 'ended_compute_utc': utc()}
        write(BIND, value); sha = ctx.digest(BIND)
        write(ROOT / 'results/native_expert_scaling/meth521_binding_resource.json',
              {'binding_sha256': sha, 'process_instance': ctx.r['process_instance'], **ctx.resources(), 'ended_utc': utc()})
        ctx.guard(); ctx.timer.cancel()
        print(json.dumps({'binding_sha256': sha, 'catalog_files': len(catalog), 'resource': ctx.resources()}), flush=True)
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
