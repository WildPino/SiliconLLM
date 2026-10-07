"""521 numbered metric-only continuation; original source/equation audit stays frozen."""
import json
from pathlib import Path
import subprocess
import sys
from meth521_operations import DOC, ROOT, Meter, Context as Original, output_bytes, utc, write

BIND = DOC / 'meth521_metric_repair1_binding_result.json'
FAULT_SHA = 'ba79eee3693ed3feb9203a733d2f3776b01515ac27cd876622139eb0a1d4f218'
MAIN_SHA = '12201afe3df9f4521bb1fe345330faf92f01c3f00197c93630e6c51fef6b5f2d'
ORIGINAL_BINDING_SHA = '97990040545bc2669431390e7f1221b52a998c08b90a5ea91db60a858856076d'


class Context(Meter):
    numpy = Original.numpy
    data = Original.data
    finish = Original.finish
    payload_stat = Original.payload_stat

    def guard(self):
        super().guard()
        if hasattr(self, 'out'):
            size = sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
            assert size <= 64 << 20 and output_bytes() + size <= 2 << 30

    def __init__(self, sha):
        super().__init__('audit_repair1', 300, 1536 << 20)
        try:
            self.out = ROOT / 'results/native_expert_scaling/meth521_audit_repair1'
            assert not self.out.exists(); self.out.mkdir()
            assert self.digest(BIND) == sha; self.b = json.loads(BIND.read_bytes())
            assert sys.version == self.b['python'] and str(Path(sys.executable).resolve()) == self.b['executable']
            for v in self.b['catalog']: assert self.digest(v['path']) == v['sha256'], v['path']
            self.payload_stat()
            for rel, expected in self.b['preserved'].items(): assert self.digest(ROOT / rel) == expected
            self.r.update(binding_sha256=sha, original_binding_sha256=ORIGINAL_BINDING_SHA,
                          main_sha256=MAIN_SHA, original_audit_fault_sha256=FAULT_SHA)
            self.r['gates']['fresh_used_metric_inputs_runtime_frozen_prefix_failure_and_repair_science'] = True
        except BaseException:
            self.fail(); raise


def bind():
    ctx = None
    try:
        ctx = Meter('metric_repair1_binding', 180, 512 << 20); catalog = {}
        def item(path, expected=None):
            p = Path(path).resolve(); digest = ctx.digest(p)
            if expected: assert digest == expected
            v = {'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest}; catalog[str(p)] = v; return v
        oldref = item(DOC / 'meth521_binding.json', ORIGINAL_BINDING_SHA); old = json.loads(Path(oldref['path']).read_bytes())
        rawref = item(DOC / 'meth521_main_result.json', MAIN_SHA); raw = json.loads(Path(rawref['path']).read_bytes())
        faultref = item(DOC / 'meth521_audit_result.failure.json', FAULT_SHA); fault = json.loads(Path(faultref['path']).read_bytes())
        assert all(raw['gates'].values()) and all(fault['gates'].values())
        assert fault['gates']['ALL53943_independent_exact_integer_F64_BLAS_source_WI_ReLU_hidden_codes_scales_and_WO_BYTES']
        assert 'line 203, in main' in fault['traceback'] and 'np.allclose(values[:, :14], retained[:, :14]' in fault['traceback']
        # Frozen control flow reaches this assertion only after all128 direct solves/17540 physical products.
        data = {k: item(old['data'][k]['path'], old['data'][k]['sha256'])
                for k in ('uid', 'occurrences', 'features', 'inputs', 'unweighted', 'targets')}
        outputs = {}
        for v in raw['output_inventory']: outputs[Path(v['path']).name] = item(v['path'], v['sha256'])
        item(ROOT / 'results/native_expert_scaling/meth521_audit/independent_controls.json')
        runtime = [item(v['path'], v['sha256']) for v in old['runtime_files']]
        scientific = []
        paths = [ROOT / 'benchmarks/native_expert_scaling' / name for name in
                 ('meth521_operations.py', 'meth521_transfer_audit.py', 'meth498_math.py', 'meth499_math.py')]
        paths += sorted((ROOT / 'benchmarks/native_expert_scaling').glob('meth521_metric_repair1*'))
        paths += [DOC / 'METH_521_METRIC_REPAIR1_PROTOCOL_20261007.md']
        for p in paths:
            rel = p.relative_to(ROOT).as_posix(); saved = subprocess.check_output(['git', 'show', 'HEAD:' + rel], cwd=ROOT)
            assert p.read_bytes().replace(b'\r\n', b'\n') == saved.replace(b'\r\n', b'\n'); scientific.append(item(p))
        for rel, expected in old['preserved'].items(): item(ROOT / rel, expected)
        ctx.r['gates'].update(first_fault_full_source_and_equation_prefix_preserved=True,
                              fresh_used_files_and_runtime_science_before_metric_continuation=True,
                              no_source_function_fit_native_or_compiler_replay=True)
        value = {**ctx.r, 'freeze_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                 'python': old['python'], 'executable': old['executable'], 'packages': old['packages'],
                 'catalog': list(catalog.values()), 'runtime_files': runtime, 'data': data, 'outputs': outputs,
                 'scientific': scientific, 'payload': old['payload'], 'preserved': old['preserved'],
                 'original_binding': oldref, 'main': rawref, 'first_audit_fault': faultref,
                 'resource_before_serialization': ctx.resources(), 'ended_compute_utc': utc()}
        write(BIND, value); sha = ctx.digest(BIND)
        write(ROOT / 'results/native_expert_scaling/meth521_metric_repair1_binding_resource.json',
              {'binding_sha256': sha, 'process_instance': ctx.r['process_instance'], **ctx.resources(), 'ended_utc': utc()})
        ctx.guard(); ctx.timer.cancel(); print(json.dumps({'binding_sha256': sha, 'resource': ctx.resources()}), flush=True)
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': bind()
