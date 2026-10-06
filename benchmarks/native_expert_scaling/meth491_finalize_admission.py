"""Metadata link of actual executions and independently validated witnesses."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST = DOC / 'ADMISSION_491_20261006.json'
assert not DEST.exists()
def receipt(path):
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

path = DOC / 'meth491_mass_interval_result.json'
completed = path.exists()
if not completed:
    path = path.with_suffix('.failure.json')
raw = json.loads(path.read_bytes())
retpath = DOC / 'RETENTION_491_20261006.json'; ret = json.loads(retpath.read_bytes())
registrations = []; events = []
for label, record, recordpath in (('main', raw, path), ('audit', ret, retpath)):
    registrationpath = DOC / ('meth491_'+label+'_sole_first_registration.json')
    registration = json.loads(registrationpath.read_bytes())
    assert registration['attempt'] == 1 and registration['process_instance'] == record['process_instance']
    assert registration['record_sha256'] == receipt(recordpath)['sha256']
    assert registration['actual_exit_code'] == 0 if completed or label == 'audit' else registration['actual_exit_code'] != 0
    eventpath = ROOT / ('results/native_expert_scaling/meth491_'+('audit_' if label == 'audit' else '')+'windows_terminal.json')
    event = json.loads(eventpath.read_bytes())
    assert event['query_available'] and event['query_error'] is None and not event['matching_scientific_events']
    assert event['instances'][0]['pid'] == record['process_instance']['pid']
    assert event['instances'][0]['create_time_unix'] == record['process_instance']['create_time_unix']
    assert all(datetime.datetime.fromisoformat(event['instances'][0][key]) == datetime.datetime.fromisoformat(record[key]) for key in ('start_utc', 'end_utc'))
    registrations.append(receipt(registrationpath)); events.append(receipt(eventpath))
assert ret['raw']['sha256'] == receipt(path)['sha256'] and all(ret['gates'].values())
assert ret['main_windows_sha256'] == events[0]['sha256']
if completed:
    assert all(raw['gates'].values()) and ret['main_completed'] and ret['witnesses_audited'] == 12
    assert raw['optimizer_updates'] == raw['native_calls'] == ret['optimizer_updates'] == ret['native_calls'] == ret['SVD_calls'] == 0
gates = {'sole_first_actual_main_and_audit': True, 'raw_retention_digest_link': True,
    'ALL_independent_retention_gates': True, 'BOTH_actual_Windows_without_application_fault': True,
    'no_primal_physical_quality_or_unbounded_class_promotion_from_nonzero_residual': True}
value = {'experiment': 'METH491 parametric affine root-mass witnesses admission', 'main_completed': completed,
    'raw': receipt(path), 'retention': receipt(retpath), 'registrations': registrations, 'Windows_events': events,
    'inherited_mass_admission': receipt(DOC / 'ADMISSION_490_R1_20261006.json'), 'gates': gates,
    'decision': ret['decision'], 'scope': 'Only exact subset dual/parametric outcomes; full goal active/incomplete.'}
with DEST.open('xb') as stream:
    stream.write((json.dumps(value, indent=2, allow_nan=False)+'\n').replace('\n', '\r\n').encode())
print(json.dumps({'admission': str(DEST), 'sha256': receipt(DEST)['sha256'], 'gates': gates, 'decision': value['decision']}))
