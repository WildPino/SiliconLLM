"""520 fresh used geometry/runtime bytes; no source-response mathematics."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth520_operations import BIND, DOC, ROOT, Meter, write


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
                'ADMISSION_497_20261006.json': '6121e58ce83bb53fa12eaf3cda554c3d33288027c5ab7957ee35f97d780a35b5',
                'meth519_binding.json': '759330331caca87bd6c358ae1f7fd00bc31896ab64ed117569141d1caaa899b0',
                'ADMISSION_519_20261007.json': '9ed10d86565b23c80521d02f0da327269cbd1f3ec76b814544bda9068b4a1a75'}
        for name, sha in refs.items(): item(DOC / name, sha)
        for name in ['ADMISSION_499_20261006.json', 'ADMISSION_497_20261006.json']:
            adm = json.loads((DOC / name).read_bytes()); assert adm['main_completed'] and all(adm['gates'].values())
            for key in ['raw', 'retention']: item(adm[key]['path'], adm[key]['sha256'])
        old = json.loads((DOC / 'meth499_binding.json').read_bytes())
        recent = json.loads((DOC / 'meth519_binding.json').read_bytes())
        data = {k: item(old['data'][k]['path'], old['data'][k]['sha256']) for k in ['uid', 'occurrences', 'features', 'geometry']}
        raw = json.loads((DOC / 'meth499_factor_fit_result.json').read_bytes())
        bank = next(v for v in raw['output_inventory'] if Path(v['path']).name == 'bank.bin')
        p = Path(bank['path']); stat = p.stat(); assert stat.st_size == bank['bytes'] == 127232136
        prefix = {'path': str(p), 'offset': 8, 'bytes': 223360, 'sha256': ctx.digest(p, 8, 223360)}
        assert prefix['sha256'] == raw['fixed_dictionary_keys_sha256']
        runtime = [item(v['path'], v['sha256']) for v in recent['runtime_files']]
        scientific = []
        for p in sorted((ROOT / 'benchmarks/native_expert_scaling').glob('meth520*')) + [DOC / 'METH_520_SOURCE_INFORMATION_ELIGIBILITY_PROTOCOL_20261007.md']:
            rel = p.relative_to(ROOT).as_posix(); saved = subprocess.check_output(['git', 'show', 'HEAD:' + rel], cwd=ROOT)
            assert p.read_bytes().replace(b'\r\n', b'\n') == saved.replace(b'\r\n', b'\n')
            scientific.append(item(p))
        for rel, sha in recent['preserved'].items(): item(ROOT / rel, sha)
        assert sys.version == recent['python'] and str(Path(sys.executable).resolve()) == recent['executable']
        ctx.r['gates'].update(admitted499_function_class_497_rank_and519_terminal_anchors=True,
                              fresh_used_development_features_metric_dictionary_prefix_runtime_science_SHA=True,
                              no_targets_compiler_source_payload_or_response_calls=True)
        value = {**ctx.r, 'experiment': 'METH520 development complement source-information eligibility',
                 'freeze_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                 'python': sys.version, 'executable': str(Path(sys.executable).resolve()), 'packages': recent['packages'],
                 'catalog': list(catalog.values()), 'data': data, 'dictionary_prefix': prefix,
                 'dictionary_file': {'path': prefix['path'], 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                                     'whole_SHA_scope': 'retained499 bank inventory; fresh fixed223360B prefix only'},
                 'scientific': scientific, 'runtime_files': runtime, 'preserved': recent['preserved'],
                 'limits': {'binding': [180, 512 << 20], 'main': [900, 1536 << 20], 'audit': [900, 1536 << 20], 'combined_output_bytes': 32 << 20},
                 'resource_before_serialization': ctx.resources(), 'ended_compute_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        write(BIND, value); sha = ctx.digest(BIND)
        write(ROOT / 'results/native_expert_scaling/meth520_binding_resource.json',
              {'binding_sha256': sha, 'process_instance': ctx.r['process_instance'], **ctx.resources(),
               'ended_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        ctx.guard(); ctx.timer.cancel()
        print(json.dumps({'binding_sha256': sha, 'catalog_files': len(catalog), 'resource': ctx.resources()}), flush=True)
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
