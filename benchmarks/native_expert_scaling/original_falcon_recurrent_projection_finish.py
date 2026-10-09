"""Bind actual24 bases/18 metrics and finish missing126 with bounded SSD storage."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';sys.path.insert(0,str(B))
from chatbot_falcon_usability import sha,write
from original_falcon_whole_recovery import extent


def bind(a):
    path=DOC/'original_falcon_recurrent_projection_binding_20261009.json'
    parent=json.loads(path.read_bytes());assert sha(path)=='b47fb412f0100ce8ad51f00ac3167ae93a8543ebeb94192990f24c700329d59c'
    ns=ROOT/'results/native_expert_scaling/original_falcon_recurrent_projection_20261009'
    fault=ns/'first_fault.json';r=json.loads(fault.read_bytes())
    failure=DOC/'original_falcon_recurrent_projection_result_20261009.launcher_failure.json'
    f=json.loads(failure.read_bytes());assert f['exit_code']==1 and r['error']=="AssertionError('GPU reserved cap')"
    assert len(r['completed_bases'])==24 and len(r['completed_measurements'])==18
    assert all(v['reconstruction']['passed'] for v in r['completed_measurements'])
    frozen=DOC/'original_falcon_recurrent_projection_frozen_20261009.py.txt'
    inputs=[]
    for i in parent['inputs']:
        item=extent(frozen if Path(i['path']).resolve()==Path(parent['worker_path']).resolve() else i['path'])
        assert (item['bytes'],item['sha256'])==(i['bytes'],i['sha256']),i['path'];inputs.append(item)
    files=[path,fault,failure,Path(__file__),B/'original_falcon_recurrent_projection.py',
        DOC/'original_falcon_recurrent_projection_result_20261009.worker.log',
        DOC/'ORIGINAL_FALCON_RECURRENT_PROJECTION_FINISH_PROTOCOL_20261009.md']
    for packet in r['completed_bases']:
        assert extent(packet['basis']['path'])==packet['basis'];files.append(Path(packet['basis']['path']))
    for row in r['completed_measurements']:
        metric=ns/f"{row['id']}.site{row['site']:02d}.metrics.json"
        assert json.loads(metric.read_bytes())==row;files.append(metric)
    inputs += [extent(p) for p in files];inputs=list({i['path']:i for i in inputs}.values())
    result=dict(parent,adopted_bases=r['completed_bases'],adopted_measurements=r['completed_measurements'],
        inputs=inputs,limits=dict(seconds=900,reserve_seconds=60,OS_bytes=6<<30,GPU_allocated_bytes=4<<30,
            GPU_reserved_bytes=5<<30,output_bytes=4<<20),
        parent_fault=extent(fault),parent_failure=extent(failure),parent_family_seconds=f['elapsed_seconds'],
        original_worker_alias=extent(frozen),
        runtime_binding_scope=parent['runtime_binding_scope']+' Completed24 bases/18 case-sites reused;'
            'only missing126 local comparisons;Y_diag chunk storage and empty_cache,new family same GPU caps.')
    write(a.out,result);print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(inputs),
        adopted_bases=24,adopted_case_sites=18,new_case_sites=126,parent_family_seconds=f['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);bind(p.parse_args())
