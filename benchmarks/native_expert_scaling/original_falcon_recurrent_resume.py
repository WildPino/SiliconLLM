"""Adopt45 durable source histories, complete only3 missing after lost process."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]; B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from chatbot_falcon_usability import SITE,sha,write
from original_falcon_whole_recovery import extent,check_inputs
sys.path.insert(0,str(SITE))


def bind(a):
    parent_path=DOC/'original_falcon_recurrent_capture_binding_20261009.json'
    parent=json.loads(parent_path.read_bytes())
    assert sha(parent_path)=='aa1c57ccf7925f2f60e4a81104ce5ef96929d2912de29377395bd7d8668e6dc2'
    ns=ROOT/'results/native_expert_scaling/original_falcon_recurrent_capture_20261009'
    frozen=DOC/'original_falcon_recurrent_capture_frozen_20261009.py.txt'
    old_worker=Path(parent['worker_path']).resolve()
    parent_inputs=[]
    for item in parent['inputs']:
        actual=extent(frozen if Path(item['path']).resolve()==old_worker else item['path'])
        assert (actual['bytes'],actual['sha256'])==(item['bytes'],item['sha256']),item['path']
        parent_inputs.append(actual)
    corpus=json.loads(Path(parent['corpus']).read_bytes())['records'];byid={r['id']:r for r in corpus}
    fields=parent['fields'];done=[];files=[]
    for identifier in parent['ordered_ids']:
        packet=ns/(identifier+'.json')
        if not packet.exists():continue
        row=json.loads(packet.read_bytes());rec=byid[identifier]
        assert row['input_ids']==rec['student_input_ids'] and row['positions']==rec['positions']
        assert row['history']==len(row['input_ids']) and len(row['sites'])==24
        for site in row['sites']:
            assert set(site['fields'])==set(fields)
            for name,item in site['fields'].items():
                assert item['shape']==[row['history'],fields[name]['width']] and item['dtype']==fields[name]['dtype']
                assert extent(item['path'])=={k:item[k] for k in ('path','bytes','sha256')}
                files.append(Path(item['path']))
        done.append(row);files.append(packet)
    assert len(done)==45
    done_ids={r['id'] for r in done};missing=[i for i in parent['ordered_ids'] if i not in done_ids]
    assert missing==['broad_dev_smol_summarize_039','broad_dev_systemchats_30k_034','broad_dev_systemchats_30k_039']
    partial=[]
    for identifier in missing:
        for p in sorted(ns.glob(identifier+'.*')):
            assert p.suffix in ('.bf16','.f32'),p
            partial.append(extent(p));files.append(p)
    assert len(partial)==63 and all('broad_dev_smol_summarize_039.site' in i['path'] for i in partial)
    files += [Path(__file__),B/'original_falcon_recurrent_capture.py',parent_path,frozen,
        DOC/'ORIGINAL_FALCON_RECURRENT_RESUME_PROTOCOL_20261009.md',
        DOC/'original_falcon_recurrent_capture_result_20261009.worker.log']
    inputs=parent_inputs+[extent(p) for p in dict.fromkeys(files)]
    inputs=list({i['path']:i for i in inputs}.values())
    histories=sum(len(byid[i]['student_input_ids']) for i in missing)
    assert histories==2167
    result=dict(parent,worker_path=str((B/'original_falcon_recurrent_capture.py').resolve()),
        ordered_ids=missing,cases=3,history_tokens=histories,expected_payload_bytes=histories*24*29792,
        limits=dict(seconds=600,reserve_seconds=60,OS_bytes=12<<30,GPU_allocated_bytes=10<<30,
            GPU_reserved_bytes=11<<30,output_bytes=3<<30),inputs=inputs,
        interrupted_parent=dict(namespace=str(ns.resolve()),binding=extent(parent_path),
            old_worker_byte_alias=extent(frozen),completed_records=done,unadmitted_partial_files=partial,
            completed_source_base_forwards=45,source_base_attempts_lower_bound=46,
            last_recorded_worker_seconds=1267.25,process_handle='Missing session29367; no matching Python process observed',
            exit_code=None,held_family_seconds=None,GPU_peaks=None,worker_OS_peak=None,launcher_OS_peak=None,
            final_parameter_identity_version_check_missing=True),
        runtime_binding_scope='New source3-case completion; adopt45 durable packets/63 unadmitted prefix fields, '
            'old code exact-byte alias. Parent exit/resource/identity aggregate records are missing, not reconstructed.')
    write(a.out,result)
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(inputs),adopted_cases=45,
        new_cases=3,partial_witness_files=63,payload_bytes=result['expected_payload_bytes'])),flush=True)


def adopt(a):
    import numpy as np
    binding=DOC/'original_falcon_recurrent_resume_binding_20261009.json'; b=json.loads(binding.read_bytes())
    raw=DOC/'original_falcon_recurrent_resume_result_20261009.json';r=json.loads(raw.read_bytes())
    terminal=raw.with_suffix('.terminal.json');t=json.loads(terminal.read_bytes())
    assert t['exit_code']==0 and t['resource_gates'] and sha(raw)==t['result_sha256']
    assert sha(binding)==r['binding_sha256']==t['binding_sha256']
    assert r['cases']==r['source_base_forwards']==3 and r['source_parameter_identities_versions_unchanged']
    check_inputs(b)
    for item in t['output_files']:assert extent(item['path'])==item
    records=b['interrupted_parent']['completed_records']+r['records']
    assert len(records)==48 and len({i['id'] for i in records})==48
    for row in records:
        for site in row['sites']:
            for name,item in site['fields'].items():
                values=np.fromfile(item['path'],dtype='<u2' if item['dtype']=='BF16' else '<f4')
                if item['dtype']=='BF16':values=(values.astype('<u4')<<16).view('<f4')
                assert values.size==row['history']*b['fields'][name]['width']
                assert np.isfinite(values).all()
    new_files={Path(item['path']).name:item for row in r['records'] for site in row['sites'] for item in site['fields'].values()}
    for item in b['interrupted_parent']['unadmitted_partial_files']:
        new=new_files[Path(item['path']).name]
        assert (new['bytes'],new['sha256'])==(item['bytes'],item['sha256']),item['path']
    check_inputs(b)
    total=sum(f['bytes'] for row in records for site in row['sites'] for f in site['fields'].values())
    assert total==16121285376
    result=dict(r,schema='ORIGINAL_FALCON_RECURRENT_CAPTURE_RESULT_V1',
        decision='ALL_RECURRENT_OPERANDS_ADOPTED_WITH_INTERRUPTED_PARENT_RUNTIME_GAPS',
        cases=48,records=records,history_tokens=22547,payload_bytes=total,
        source_base_forwards=48,source_base_attempts_lower_bound=49,new_source_base_forwards=3,
        adopted_completed_source_base_forwards=45,partial_prefix_witness_files_equal=63,
        source_parameter_identities_versions_unchanged=None,
        completion_parameter_identities_versions_unchanged=True,original_parameter_aggregate_check_missing=True,
        original_resource_records_missing=True,original_held_family_seconds=None,
        completion_held_family_seconds=t['elapsed_seconds'],combined_family_seconds=None,
        completion_raw_result=extent(raw),completion_terminal=extent(terminal),
        adoption_binding=extent(binding),adoption_code=extent(Path(__file__)),
        inherited_parent=b['interrupted_parent'],
        runtime_scope='Completion resource peaks/elapsed/identity check describe ONLY new3-case family; '
            'old45 completed histories+one partial attempt have no held terminal/peak/exit records.',
        stored_finite_extent_hash_adoption=True,adoption_new_source_forwards=0,
        adoption_new_predictions=0,adoption_GPU_calls=0)
    write(a.out,result)
    print(json.dumps(dict(result=str(a.out),sha256=sha(a.out),cases=48,payload_bytes=total,
        partial_prefix_witness_files_equal=63,combined_family_seconds=None)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--bind',action='store_true');m.add_argument('--adopt',action='store_true')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.bind:bind(a)
    else:adopt(a)
