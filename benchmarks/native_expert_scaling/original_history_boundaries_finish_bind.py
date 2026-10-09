"""Bind missing-only continuation of actual captured residual boundaries."""
import json
from pathlib import Path
import sys
from original_history_boundaries import ROOT, B, DOC, SITE, extent, sha, write, energy


def bind(a):
    import numpy as np
    import psutil
    parent_path=DOC/'original_history_boundaries_binding_r2_20261009.json'
    parent=json.loads(parent_path.read_bytes())
    assert sha(parent_path)=='2d565b4e0e4779f5ea2fb0cf28bc4cc793cc952fda0949c7cfbe712b36ea60b2'
    ns=ROOT/'results/native_expert_scaling/original_history_boundaries_20261009'
    fault_path=ns/'first_fault.json';fault=json.loads(fault_path.read_bytes())
    failure_path=DOC/'original_history_boundaries_result_20261009.launcher_failure.json'
    failure=json.loads(failure_path.read_bytes())
    assert failure['exit_code']==1 and failure['worker_pid']==25640
    if psutil.pid_exists(failure['worker_pid']):
        assert psutil.Process(failure['worker_pid']).create_time()!=failure['worker_creation_time'],'parent live'
    assert fault['stage']=='broad_fit_smol_magpie_ultra_022/h4' and fault['error']=="AssertionError('GPU reserved cap')"
    assert len(fault['completed_ids'])==17 and fault['source_base_forwards']==17 and fault['source_base_attempts']==18
    assert fault['exact_output_witnesses']==103
    archive=DOC/'original_history_boundaries_first_fault_20261009.json'
    if not archive.exists():archive.write_bytes(fault_path.read_bytes())
    assert archive.read_bytes()==fault_path.read_bytes()
    adopted=[json.loads((ns/(identifier+'.json')).read_bytes()) for identifier in fault['completed_ids']]
    assert all(len(rec['boundaries'])==7 and rec['exact_output_witnesses']==parent['witness_sites'] for rec in adopted)
    partial='broad_fit_smol_magpie_ultra_022';row=next(rec for rec in parent['witnesses'] if rec['id']==partial)
    boundaries=[]
    for boundary in (0,4):
        hp=ns/f'{partial}.h{boundary:02d}.bf16';zp=ns/f'{partial}.h{boundary:02d}.P256.f32'
        bits=np.fromfile(hp,dtype='<u2').reshape(row['history'],2048)
        h=(bits.astype('<u4')<<16).view('<f4');z=np.fromfile(zp,dtype='<f4').reshape(row['history'],256)
        assert np.isfinite(h).all() and np.isfinite(z).all()
        eh=energy(h);ez=energy(z)
        boundaries.append(dict(boundary=boundary,h=dict(**extent(hp),shape=list(h.shape),dtype='uint16'),
            z=dict(**extent(zp),shape=list(z.shape),dtype='float32'),source_energy=eh,projected_energy=ez,
            discarded_energy_difference=eh['total']-ez['total'],retained_energy_fraction=ez['total']/max(eh['total'],1e-24),
            centered_retained_fraction=ez['centered']/max(eh['centered'],1e-24)))
    old_outputs=[extent(path) for path in sorted(ns.iterdir()) if path.is_file()]
    assert len(old_outputs)==262
    files=[Path(item['path']) for item in parent['inputs']]+[parent_path,failure_path,archive,
        DOC/'original_history_boundaries_result_20261009.worker.log',Path(__file__),
        B/'original_history_boundaries_finish.py',DOC/'ORIGINAL_HISTORY_BOUNDARIES_FINISH_PROTOCOL_20261009.md']
    files += [Path(item['path']) for item in old_outputs]
    result=dict(parent,schema='ORIGINAL_HISTORY_BOUNDARIES_FINISH_BINDING_V1',
        worker_path=str((B/'original_history_boundaries_finish.py').resolve()),
        adopted_records=adopted,partial_id=partial,partial_boundaries=boundaries,
        remaining_witnesses=[rec for rec in parent['witnesses'] if rec['id'] not in fault['completed_ids']],
        parent_P=extent(ns/'P256.f32'),parent_fault=extent(fault_path),parent_failure=extent(failure_path),
        parent_held_seconds=failure['elapsed_seconds'],old_output_files=old_outputs,
        continuation_scope='17 complete cases/2 partial boundaries adopted;resume depth from saved BF16 h4,'
            'remaining20 decoder layers for partial case and30 new full calls;no earlier source-layer replay.'
            'Release only unused CUDA cache between cases. Parent resource/parameter aggregate gaps retained.')
    result['limits']=dict(parent['limits'],seconds=1660,output_bytes=(1<<30)-sum(item['bytes'] for item in old_outputs))
    assert len(result['remaining_witnesses'])==31 and result['remaining_witnesses'][0]['id']==partial
    assert result['parent_held_seconds']+result['limits']['seconds']<1800
    result['inputs']=[extent(path) for path in dict.fromkeys(files)]
    write(a.out,result)
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(result['inputs']),
        old_files=len(old_outputs),old_bytes=sum(item['bytes'] for item in old_outputs))),flush=True)
