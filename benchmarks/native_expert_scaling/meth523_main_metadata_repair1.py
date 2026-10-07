"""Seal completed numerical prefix after its original terminal hashing time fault."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth523_operations import BIND, DOC, ROOT, Meter, write

FAULT_SHA = 'e8f24213b092c7bf31d3b4203a6d28f853733b72fc7d6927a3c64a6d5ddb84c5'
BIND_SHA = '7cced083d598522bef6b9c382b96b5761e4c26f37130e814241572c5d9603b0e'


def main():
    ctx = None
    try:
        ctx = Meter('main_metadata_repair1', 180, 128 << 20)
        def item(path, expected=None):
            p = Path(path).resolve(); sha = ctx.digest(p)
            if expected: assert sha == expected, str(p)
            return dict(path=str(p), bytes=p.stat().st_size, sha256=sha)
        fault_ref = item(DOC / 'meth523_main_result.failure.json', FAULT_SHA); b = json.loads(Path(item(BIND, BIND_SHA)['path']).read_bytes())
        f = json.loads(Path(fault_ref['path']).read_bytes())
        assert len(f['gates']) == 7 and all(f['gates'].values()) and len(f['cases']) == 128 and len(f['views']) == 656
        assert f['binding_sha256'] == BIND_SHA and f['resource']['seconds'] > 900 and f['resource']['OS_peak_bytes'] <= 1536 << 20
        assert 'finish' in f['traceback'] and 'inventory' in f['traceback'] and 'self.guard()' in f['traceback']
        metadata_science = []
        for p in (Path(__file__).resolve(), DOC / 'METH_523_MAIN_METADATA_REPAIR1_PROTOCOL_20261007.md'):
            rel = p.relative_to(ROOT).as_posix(); saved = subprocess.check_output(['git', 'show', 'HEAD:' + rel], cwd=ROOT)
            assert p.read_bytes().replace(b'\r\n', b'\n') == saved.replace(b'\r\n', b'\n')
            metadata_science.append(item(p))
        for v in b['catalog']: item(v['path'], v['sha256'])
        for v in b['source_extents']: assert ctx.digest(v['path'], v['offset'], v['bytes']) == v['sha256']
        folder = ROOT / 'results/native_expert_scaling/meth523_fold'
        assert not (folder / 'terminal_resource.json').exists() and not (DOC / 'meth523_main_result.json').exists()
        inventory = [item(p) for p in sorted(folder.iterdir()) if p.is_file()]
        assert len(inventory) == 16 and sum(v['bytes'] for v in inventory) < 2 << 30
        assert {(v['path'], v['bytes']) for v in inventory} == {(v['path'], v['bytes']) for v in f['partial_outputs']}
        # No array parsing/NumPy/selection/projection/encoding/function/metric call.
        m = {k: v for k, v in f.items() if k not in ('traceback', 'resource', 'partial_outputs')}
        m.update(output_inventory=inventory, original_failure=fault_ref, original_executor_exit=dict(tool_chunk_id='2afaca', exit_code=1),
            original_terminal_resource=f['resource'], original_resource_gate_passed=False,
            numerical_prefix_completed=True, original_main_success=False,
            metadata_seal_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(DOC / 'meth523_main_result.json', m); raw = item(DOC / 'meth523_main_result.json')
        write(folder / 'terminal_resource.json', dict(result_sha256=raw['sha256'], process_instance=f['process_instance'],
            **f['resource'], original_resource_gate_passed=False, original_executor_exit_code=1,
            metadata_seal_process_instance=ctx.r['process_instance'], metadata_seal_resource=ctx.resources()))
        ctx.r['gates'].update(original_complete7_scientific_prefix_and_exact_hashing_time_fault_retained=True,
            original_ALL_inputs_source_runtime_and_ALL16_output_SHA_sealed_without_numerical_calls=True,
            canonical_copy_explicitly_retains_original_exit1_and_failed_900s_gate=True)
        ctx.r.update(original_failure=fault_ref, binding_sha256=BIND_SHA, canonical_main=raw, new_numerical_calls=0,
            metadata_science=metadata_science, repair_freeze_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            original_resource_gate_passed=False, resource=ctx.resources())
        write(ctx.raw, ctx.r); job = item(ctx.raw)
        write(ROOT / 'results/native_expert_scaling/meth523_main_metadata_repair1_resource.json',
            dict(result_sha256=job['sha256'], process_instance=ctx.r['process_instance'], **ctx.resources()))
        ctx.guard(); ctx.timer.cancel(); print(json.dumps(dict(main_sha256=raw['sha256'], repair_sha256=job['sha256'], resource=ctx.resources())), flush=True)
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
