"""Fresh used retained-data, original bank11 extents and isolated runtime binding."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth516_operations import BIND, DOC, ROOT, Meter, write


def main():
    ctx = None
    try:
        ctx = Meter('binding', 180, 512 << 20)
        catalog = {}

        def item(path, expected=None):
            p = Path(path).resolve()
            sha = ctx.digest(p)
            if expected is not None:
                assert sha == expected, str(p)
            v = {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha}
            catalog[str(p)] = v
            return v

        oldpath = DOC / 'meth513_r1_binding.json'
        item(oldpath, 'e149336d994ab20bdf667154c97132be975eac07cbd27268f21e50464afd3c7b')
        old = json.loads(oldpath.read_bytes())
        admissionpath = DOC / 'ADMISSION_504_20261006.json'
        item(admissionpath)
        admission = json.loads(admissionpath.read_bytes())
        assert admission['main_completed'] and all(admission['gates'].values())
        for key in ['raw', 'binding', 'retention']:
            item(admission[key]['path'], admission[key]['sha256'])
        base = json.loads(Path(admission['binding']['path']).read_bytes())
        data = {key: item(base['data'][key]['path'], base['data'][key]['sha256'])
                for key in ['uid', 'occurrences', 'inputs', 'hidden', 'codes', 'scales']}
        item(base['data']['manifest']['path'], base['data']['manifest']['sha256'])
        for key in ['query_norm2.npy'] + [f'row_norm2_e{e:03}.npy' for e in range(1, 128)]:
            data[key] = item(old['data'][key]['path'], old['data'][key]['sha256'])
        sourcepath = DOC / 'meth380_switch_base128_export_result.json'
        source_record = item(sourcepath)
        source = json.loads(sourcepath.read_bytes())
        assert all(source['gates'].values())
        assert source['original_config']['dense_act_fn'] == 'relu'
        assert source['original_config']['num_experts'] == 128 and source['artifact']['source_unique_parameters'] == 7415217408
        payload = old['retained_complete_original_fallback_payload']
        stat = Path(payload['path']).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (payload['bytes'], payload['mtime_ns'])
        assert payload['path'] == source['artifact']['payload'] and payload['sha256'] == source['artifact']['sha256']
        parents, extents = [], []
        for e in range(128):
            parent = {'parent': e}
            for operator, shape in [('wi', [3072, 768]), ('wo', [768, 3072])]:
                name = f'decoder.block.11.layer.2.mlp.experts.expert_{e}.{operator}.weight'
                v = source['tensors'][name]
                assert v['shape'] == shape and v['encoding'] == 1 and v['elements'] == v['bytes'] == 3072 * 768
                assert v['scale_bytes'] == 4 * shape[0]
                parent[operator] = {'name': name, **v}
                for offset, length, expected, role in [(v['offset'], v['bytes'], v['sha256'], 'code'),
                                                       (v['scale_offset'], v['scale_bytes'], v['scale_sha256'], 'scale')]:
                    assert ctx.digest(payload['path'], offset, length) == expected
                    extents.append({'parent': e, 'operator': operator, 'role': role,
                                    'offset': offset, 'bytes': length, 'sha256': expected})
            parents.append(parent)
        assert len(extents) == 512 and sum(v['bytes'] for v in extents) == 605945856
        spans = sorted((v['offset'], v['offset'] + v['bytes']) for v in extents)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
        runtime = [item(v['path'], v['sha256']) for v in old['runtime_files']]
        scientific = []
        for p in sorted((ROOT / 'benchmarks/native_expert_scaling').glob('meth516*')) + [DOC / 'METH_516_DIRECTIONAL_WI_PROTOCOL_20261007.md']:
            assert p.is_file()
            rel = p.relative_to(ROOT).as_posix()
            committed = subprocess.check_output(['git', 'show', 'HEAD:' + rel], cwd=ROOT)
            assert p.read_bytes().replace(b'\r\n', b'\n') == committed.replace(b'\r\n', b'\n')
            scientific.append(item(p))
        for rel, preserved_sha in old['preserved'].items():
            item(ROOT / rel, preserved_sha)
        assert sys.version == old['python'] and str(Path(sys.executable).resolve()) == old['executable']
        ctx.r['gates'].update(retained504_admission_and_original128_relu_contract=True,
                              all512_original_WI_WO_code_scale_extents_fresh_SHA=True,
                              used_input_cache_norm_runtime_science_foreign_fresh_SHA=True)
        value = {**ctx.r, 'experiment': 'METH516 exact current-query fixed64 WI certificate',
                 'freeze_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                 'python': sys.version, 'executable': str(Path(sys.executable).resolve()), 'packages': old['packages'],
                 'catalog': list(catalog.values()), 'data': data, 'parents': parents, 'extents': extents,
                 'source_record': source_record, 'payload': {**payload, 'full_SHA_scope': 'Retained whole-payload SHA/stat; fresh ALL512 bank11 code/scale extent SHA only.'},
                 'scientific': scientific, 'runtime_files': runtime, 'preserved': old['preserved'],
                 'limits': {'binding': [180, 512 << 20], 'main': [300, 2 << 30], 'audit': [300, 2 << 30], 'combined_output_bytes': 64 << 20},
                 'resource_before_serialization': ctx.resources(), 'ended_compute_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        write(BIND, value)
        result_sha = ctx.digest(BIND)
        write(ROOT / 'results/native_expert_scaling/meth516_binding_resource.json',
              {'binding_sha256': result_sha, 'process_instance': ctx.r['process_instance'],
               **ctx.resources(), 'ended_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        ctx.guard()
        ctx.timer.cancel()
        print(json.dumps({'binding_sha256': result_sha, 'files': len(catalog), 'extents': len(extents), 'resource': ctx.resources()}), flush=True)
    except BaseException:
        if ctx:
            ctx.fail()
        raise


if __name__ == '__main__':
    main()
