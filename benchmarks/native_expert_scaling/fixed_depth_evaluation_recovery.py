"""Recover journaled evaluation data without replaying the failed GPU producer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='6')
ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0] = [str(B), str(ROOT / 'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent, write
from fixed_depth_quality import load, check_inputs, quality

ORIGINAL = DOC / 'fixed_depth_evaluation6_result_20261015.json'
BINDING = DOC / 'fixed_depth_evaluation6_recovery_binding_20261015.json'
REPORT = DOC / 'fixed_depth_evaluation6_recovered_result_20261015.json'
AUDIT = DOC / 'fixed_depth_evaluation6_recovered_audit_20261015.json'
CLOSURE = DOC / 'fixed_depth_evaluation6_recovery_closure_20261015.json'
PIPELINE = DOC / 'fixed_depth_evaluation6_recovery_pipeline_20261015.json'
SOURCE_BINDING = DOC / 'fixed_depth_quality_io_binding_20261014.json'
SELECTION = DOC / 'fixed_depth_fit_selection_20261015.json'
BASE_AUDITOR = B / 'fixed_depth_quality_audit.py'
PROTOCOL = DOC / 'FIXED_DEPTH_EVALUATION_RECOVERY_20261015.md'
SECONDS = 3600


def check(pins, guard):
    for item in pins:
        guard()
        assert extent(item['path']) == item, item['path']


def receipts():
    import psutil
    worker = load(ORIGINAL.with_suffix('.worker_terminal.json'))
    holder = load(ORIGINAL.with_suffix('.terminal.json'))
    assert worker['exit_code'] == holder['exit_code'] == holder['worker_exit_code'] == 1
    assert worker['stage'] == 'after_hash' and worker['phase'] == 'evaluation' and worker['depth'] == 6
    assert worker['error'] == "AssertionError('worker total deadline')"
    assert worker['completed_histories'] == 24 and worker['completed_controls'] == 16
    assert holder['worker_pid'] == worker['pid']
    assert abs(holder['worker_creation_time'] - worker['creation_time']) < .001
    for row in (worker, holder):
        if psutil.pid_exists(row['pid']):
            assert abs(psutil.Process(row['pid']).create_time() - row['creation_time']) > .001
    return worker, holder


def auditor_source():
    """Replace custody admission only; preserve all mathematical checks verbatim."""
    source = BASE_AUDITOR.read_text(encoding='utf8')
    before = """    holder=load(a.report.with_suffix('.terminal.json'));worker=load(a.report.with_suffix('.worker_terminal.json'))
    assert holder['exit_code']==worker['exit_code']==holder['worker_exit_code']==0
    assert holder['result']==worker['result']==extent(a.report)
"""
    after = """    assert r['schema']=='FIXED_DEPTH_QUALITY_RECOVERED_EVALUATION_V1'
    assert r['phase']=='evaluation' and r['depth']==6 and r['nonselected_defect_diagnostic']
    assert not r['original_posthash_complete'] and not r['before_after_hashes_exact']
    assert r['reconciliation_current_inputs_outputs_exact']
    assert r['recovery']['binding']==extent(RECOVERY_BINDING)
    rb=load(RECOVERY_BINDING);assert rb['source_binding']==r['binding']
    check_inputs({'inputs':rb['inputs']},PIPELINE_GUARD)
    worker,holder=RECOVERY_RECEIPTS()
"""
    substitutions = [
        (before, after),
        ("    def guard():\n", "    def guard():\n        PIPELINE_GUARD()\n"),
        ("    fixed=[extent(a.report),r['binding'],extent(a.report.with_suffix('.terminal.json')),extent(a.report.with_suffix('.worker_terminal.json'))]",
         "    fixed=[extent(a.report),r['binding'],extent(RECOVERY_BINDING)]+rb['inputs']"),
        ("assert r['all411_source_parameters_exact'] and r['before_after_hashes_exact'] and not r['timing_admitted']",
         "assert r['all411_source_parameters_exact'] and r['reconciliation_current_inputs_outputs_exact'] and not r['timing_admitted']"),
        ("        before_after_hashes_exact=True,seconds=", "        before_after_hashes_exact=True,original_posthash_complete=False,recovery_binding=extent(RECOVERY_BINDING),seconds="),
    ]
    for old, new in substitutions:
        assert source.count(old) == 1, old
        source = source.replace(old, new, 1)
    # This entire block includes native/state, F64, metrics, cached prefixes,
    # selection, tokenization, generated argmax IDs and complete quality checks.
    marker = '    sequences=[q[\'sequence\']'
    end = '    check();guard()'
    original = BASE_AUDITOR.read_text(encoding='utf8')
    math = original[original.index(marker):original.index(end)]
    assert math == source[source.index(marker):source.index(end)]
    compile(source, str(BASE_AUDITOR), 'exec')
    return source, hashlib.sha256(math.encode('utf8')).hexdigest()


def main(freeze, guard):
    import numpy as np
    for path in (BINDING, REPORT, AUDIT, CLOSURE, PIPELINE):
        assert not path.exists(), path
    assert not ORIGINAL.exists(), 'Do not replace an original report'
    for path in (Path(__file__), PROTOCOL):
        blob = subprocess.check_output(['git', 'show', freeze + ':' + path.relative_to(ROOT).as_posix()], cwd=ROOT)
        assert blob.replace(b'\r\n', b'\n') == path.read_bytes().replace(b'\r\n', b'\n')
    source, math_sha = auditor_source()
    worker, holder = receipts()
    b = load(SOURCE_BINDING)
    directory = Path(b['family']) / 'evaluation6'
    journal = load(directory / 'numerical_completion.json')
    assert journal['phase'] == 'evaluation' and journal['depth'] == 6
    assert journal['posthash_complete'] is False
    assert all(journal[k] for k in ('all411_source_parameters_exact', 'identities_versions_exact', 'MuP_before_after_exact'))
    assert len(journal['records']) == 24 and len(journal['tasks']) == len(journal['prefix_controls']) == 16
    assert journal['followup'] is not None and journal['GPU_phase_seconds'] < 2700
    selection = load(SELECTION)
    assert selection['selected_depth'] is None and selection['evaluation_depth'] == 6
    assert selection['nonselected_defect_diagnostic'] and selection['DEV_queries'] == selection['task_choice_queries'] == 0
    pins = [extent(p) for p in (Path(__file__), PROTOCOL, BASE_AUDITOR, SOURCE_BINDING, SELECTION,
            ORIGINAL.with_suffix('.terminal.json'), ORIGINAL.with_suffix('.worker_terminal.json'),
            ORIGINAL.with_suffix('.worker.log'), ORIGINAL.with_suffix('.owned_process.json'))]
    pins += selection['inputs']
    check_inputs(b, guard)
    check(pins, guard)
    outputs = []
    for path in sorted(directory.iterdir()):
        guard()
        assert path.is_file(), path
        outputs.append(extent(path))
    assert len(outputs) == 994 and sum(e['bytes'] for e in outputs) == 3346887358
    assert load(directory / 'parameter_before.json') == load(directory / 'parameter_after.json')
    for record in journal['records']:
        assert load(directory / (record['id'] + '.metrics.json')) == record
    for task in journal['tasks']:
        assert load(directory / (task['id'] + '.task.json')) == task
    for control in journal['prefix_controls']:
        assert load(directory / (control['id'] + '.prefix_control.json')) == control
    assert load(directory / 'own_answer_followup.json') == journal['followup']
    write(BINDING, dict(schema='FIXED_DEPTH_EVALUATION_RECOVERY_BINDING_V1', freeze=freeze,
        source_binding=extent(SOURCE_BINDING), inputs=pins, original_posthash_complete=False,
        original_worker=extent(ORIGINAL.with_suffix('.worker_terminal.json')),
        original_holder=extent(ORIGINAL.with_suffix('.terminal.json')),
        mathematical_audit_block_sha256=math_sha, seconds_cap=SECONDS, OS_bytes_cap=8 << 30,
        directory=str(directory), original_family=outputs,
        lineage='Original worker/holder exit1 after numerical completion and GPU release; current CPU seal is separate.'))
    records, tasks, controls, followup = (journal[k] for k in ('records', 'tasks', 'prefix_controls', 'followup'))
    sequences = [r['sequence'] for r in records + tasks + [followup]]
    sequences += [c[k] for c in controls for k in ('wrapper', 'continuation')]
    processed = sum(s['seen'] for s in sequences)
    assert len(sequences) == 73 and processed <= b['processed_bounds']['evaluation_processed_tokens']
    result = dict(schema='FIXED_DEPTH_QUALITY_RECOVERED_EVALUATION_V1', binding=extent(SOURCE_BINDING),
        phase='evaluation', depth=6, selection=extent(SELECTION), nonselected_defect_diagnostic=True,
        records=records, tasks=tasks, prefix_controls=controls, followup=followup, FIT_quality=None,
        quality=quality(tasks, followup, records, controls, b['gates']), parent_24_exact=None,
        processed_tokens=processed, all411_source_parameters_exact=True,
        before_after_hashes_exact=False, original_posthash_complete=False,
        reconciliation_current_inputs_outputs_exact=True, outputs=outputs, optimizer_updates=0,
        RESERVED_queries=0, omitted_source_operator_calls=0, timing_admitted=False,
        seconds=worker['seconds'], OS_peak=worker['OS_peak'], GPU_allocated_peak=worker['GPU_allocated_peak'],
        GPU_reserved_peak=worker['GPU_reserved_peak'], GPU_phase_seconds=journal['GPU_phase_seconds'],
        GPU_released_before_posthash=True, output_bytes=sum(e['bytes'] for e in outputs),
        recovery=dict(binding=extent(BINDING)), scope=b['scope'], custody='Explicit original deadline failure; late CPU reconciliation.')
    write(REPORT, result)
    print(json.dumps(dict(stage='reconciled', report=extent(REPORT), quality=result['quality'])), flush=True)
    namespace = dict(__name__='recovered_evaluation_auditor', __file__=str(BASE_AUDITOR),
        RECOVERY_BINDING=BINDING, PIPELINE_GUARD=guard, RECOVERY_RECEIPTS=receipts)
    exec(compile(source, str(BASE_AUDITOR), 'exec'), namespace)
    namespace['main'](argparse.Namespace(report=REPORT, out=AUDIT))
    import psutil
    proc = psutil.Process()
    write(AUDIT.with_suffix('.terminal.json'), dict(command=sys.orig_argv, pid=proc.pid,
        creation_time=proc.create_time(), exit_code=0, result=extent(AUDIT),
        execution='Inline full mathematical audit with explicit recovery custody branch.'))
    audited = load(AUDIT)
    assert audited['all_numeric_pass'] and audited['decision'] == 'INDEPENDENT_STORED_NUMERIC_PASS'
    assert not audited['quality_admissible'] and audited['original_posthash_complete'] is False
    assert audited['report'] == extent(REPORT) and audited['recovery_binding'] == extent(BINDING)
    receipts()
    check(pins + [extent(BINDING), extent(REPORT), extent(AUDIT)], guard)
    assert {str(p.resolve()): p.stat().st_size for p in directory.iterdir()} == {e['path']: e['bytes'] for e in outputs}
    write(CLOSURE, dict(schema='FIXED_DEPTH_RECOVERED_EVALUATION_CLOSURE_V1', report=extent(REPORT),
        audit=extent(AUDIT), recovery_binding=extent(BINDING), original_worker_exit_code=1,
        original_holder_exit_code=1, original_posthash_complete=False, current_inputs_outputs_exact=True,
        phase_family_files=len(outputs), phase_family_bytes=result['output_bytes'], quality_admissible=False,
        decision='STORED_DIAGNOSTIC_QUALIFIED_WITH_EXPLICIT_ORIGINAL_POSTHASH_FAILURE', goal_status='ACTIVE_INCOMPLETE'))
    write(PIPELINE, dict(report=extent(REPORT), audit=extent(AUDIT), closure=extent(CLOSURE),
        original_posthash_complete=False, decision='RECOVERED_STORED_DIAGNOSTIC_PIPELINE_PASS'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze')
    parser.add_argument('--check-adapter', action='store_true')
    args = parser.parse_args()
    if args.check_adapter:
        print(json.dumps(dict(adapter_compiles=True, unchanged_math_sha256=auditor_source()[1])))
    else:
        assert args.freeze
        import psutil
        proc = psutil.Process()
        proc.cpu_affinity(list(range(6)))
        start = time.monotonic()
        def guard():
            assert time.monotonic() - start <= SECONDS, 'CPU recovery deadline'
            assert proc.memory_info().peak_wset <= 8 << 30, 'OS cap'
            assert not proc.children(recursive=True), 'unexpected child'
            if 'torch' in sys.modules:
                assert not sys.modules['torch'].cuda.is_initialized()
            assert sum(p.stat().st_size for p in DOC.glob('fixed_depth_evaluation6_recover*') if p.is_file()) <= 32 << 20
        terminal = dict(command=sys.orig_argv, pid=proc.pid, creation_time=proc.create_time(),
            seconds_cap=SECONDS, OS_bytes_cap=8 << 30, GPU_calls=0)
        try:
            assert not PIPELINE.with_suffix('.terminal.json').exists()
            write(PIPELINE.with_suffix('.owned_process.json'), terminal)
            main(args.freeze, guard)
            guard()
            terminal.update(exit_code=0, error=None, result=extent(PIPELINE))
        except BaseException as error:
            terminal.update(exit_code=1, error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            terminal.update(seconds=time.monotonic() - start, OS_peak=proc.memory_info().peak_wset)
            write(PIPELINE.with_suffix('.terminal.json'), terminal)
