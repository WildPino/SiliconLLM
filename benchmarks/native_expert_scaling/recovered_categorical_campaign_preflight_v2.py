"""Small recovery-format faults and exact inherited mathematical-core check."""
import argparse,ast,hashlib,json,os,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'


def extent(path):
    p=Path(path);return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main(out):
    start=time.monotonic();flags={}
    names=['recovered_categorical_campaign_audit_v2.py','recovered_categorical_campaign_launch_v2.py','recovered_categorical_campaign_preflight_v2.py']
    sources={n:extent(B/n) for n in names}
    trees={n:ast.parse((B/n).read_text(encoding='utf-8')) for n in names};flags['AST_PASS']=True
    original=(B/'original_categorical_campaign_audit.py').read_text(encoding='utf-8')
    expected=original[original.index('        masters=changed_state('):original.index('        tokenizer=ChatTokenizer')].replace("r['","stored['")
    assert expected.count("s['binding_sha256']==a.binding_sha")==1
    expected=expected.replace("s['binding_sha256']==a.binding_sha","s['binding_sha256']==binding['original_numeric_binding']['sha256']")
    adapted=(B/names[0]).read_text(encoding='utf-8')
    actual=adapted[adapted.index('        masters=changed_state('):adapted.index('        # Original stream stage')]
    assert expected==actual;flags['original_math_core_only_index_and_original_provenance_adapter']=True
    source_binding=json.loads((ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925/original_categorical_campaign_binding_20261010.json').read_bytes())
    prior=next(x for x in source_binding['inputs'] if Path(x['path']).resolve()==B/'original_categorical_campaign_audit.py')
    assert extent(B/'original_categorical_campaign_audit.py')==prior
    sources['original_mathematical_auditor']=prior
    # Execute only the stdlib indexing function, without importing torch or the
    # worker. These synthetic bytes are format fixtures, not numerical evidence.
    node=next(x for x in trees[names[0]].body if isinstance(x,ast.FunctionDef) and x.name=='recovered_index')
    namespace=dict(Path=Path,json=json,extent=extent)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(B/names[0]),'exec'),namespace)
    index=namespace['recovered_index']
    header_node=next(x for x in trees[names[0]].body if isinstance(x,ast.FunctionDef) and x.name=='check_snapshot_header')
    exec(compile(ast.Module(body=[header_node],type_ignores=[]),str(B/names[0]),'exec'),namespace)
    header=namespace['check_snapshot_header']
    original_sha='original-binding';audit_sha='different-audit-binding'
    steps=[dict(selection=dict(accepted=True),selected_parameters=dict(weight={})) for _ in range(24)]
    state=dict(schema='ORIGINAL_CATEGORICAL_CAMPAIGN_STATE_V1',binding_sha256=original_sha,directions=6,optimizer_updates=0,accepted_displacements=6,model=dict(weight=None))
    header(state,6,steps,original_sha);flags['distinct_original_and_audit_binding_accepted']=True
    for key,value in [('binding_sha256',audit_sha),('schema','wrong'),('directions',12),('optimizer_updates',1),('accepted_displacements',5),('model',{})]:
        changed=dict(state);changed[key]=value
        try:header(changed,6,steps,original_sha)
        except AssertionError:flags['snapshot_'+key+'_fault_rejected']=True
        else:raise RuntimeError(('snapshot fault accepted',key))
    with tempfile.TemporaryDirectory(prefix='recovered_campaign_format_') as tmp:
        directory=Path(tmp);fit=[f'fit_{i}' for i in range(24)]
        binding=dict(FIT_order=fit,alphas=[.0001,.001],milestones=[6,12,18,24],records=[dict(id=f'case_{i}') for i in range(48)])
        def write(path,value):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')
        for ordinal,identifier in enumerate(fit,1):
            folder=directory/f'step_{ordinal:02d}';history=dict(reused=ordinal==1)
            trials=[dict(alpha=alpha) for alpha in binding['alphas']]
            write(folder/'step.json',dict(ordinal=ordinal,id=identifier,history=history,trials=trials,selected_packed=dict(sha256='fixture'),selected_parameters=dict()))
            write(folder/'history.json',history)
            for row in trials:write(folder/f"alpha_{row['alpha']:g}"/'trial.json',row)
            if ordinal in binding['milestones']:(folder/'masters.pt').write_bytes(b'format fixture only')
        for rec in binding['records']:write(directory/('final_'+rec['id']+'.json'),rec)
        def seal():return dict(outputs=[extent(p) for p in directory.rglob('*') if p.is_file()])
        frozen=seal();valid=index(binding,frozen,directory)
        assert len(valid['steps'])==24 and len(valid['final_native'])==48 and valid['pending'] is None
        assert not any(k in valid for k in ('GPU_allocated_peak','GPU_reserved_peak','history_attempts','proposal_attempts','decision','freeze','binding_sha256'))
        flags['complete_durable_index_without_invented_runtime_receipt']=True
        def rejected():
            try:index(binding,seal(),directory)
            except (AssertionError,KeyError):return True
            return False
        path=directory/'step_01'/'step.json';saved=path.read_bytes();row=json.loads(saved);row['ordinal']=2;write(path,row)
        assert rejected();path.write_bytes(saved);flags['wrong_ordinal_rejected']=True
        path=directory/'step_01'/'alpha_0.0001'/'trial.json';saved=path.read_bytes();write(path,dict(alpha=.9))
        assert rejected();path.write_bytes(saved);flags['inconsistent_duplicate_trial_rejected']=True
        path=directory/'step_24'/'step.json';saved=path.read_bytes();path.unlink()
        assert rejected();path.write_bytes(saved);flags['missing_final_step_rejected']=True
        path=directory/'final.stream.receipt.json';write(path,dict())
        assert rejected();path.unlink();flags['unexpected_stream_receipt_rejected']=True
        path=directory/'final_case_0.json';saved=path.read_bytes();write(path,dict(id='changed'))
        try:index(binding,frozen,directory);raise RuntimeError('changed extent accepted')
        except AssertionError:pass
        path.write_bytes(saved);flags['changed_sealed_extent_rejected']=True
    # Read actual checkpoint headers without materializing tensor values. This
    # diagnoses the adapter; full file hashes and parameter bits stay mandatory
    # in the independently bound complete holder/worker audit.
    sys.path.insert(0,str(B))
    from original_packed_capacity import SITE
    sys.path.insert(0,str(SITE));os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch,psutil
    proc=psutil.Process();proc.cpu_affinity([1]);torch.set_num_threads(1)
    directory=ROOT/'results/native_expert_scaling/original_categorical_campaign_20261010'
    actual_steps=[json.loads((directory/f'step_{i:02d}/step.json').read_bytes()) for i in range(1,25)]
    original_binding=extent(ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925/original_categorical_campaign_binding_20261010.json')
    observed_headers=[]
    for ordinal in (6,12,18,24):
        checkpoint=torch.load(directory/f'step_{ordinal:02d}/masters.pt',weights_only=True,map_location='cpu',mmap=True)
        header(checkpoint,ordinal,actual_steps,original_binding['sha256'])
        observed_headers.append(dict(ordinal=ordinal,parameters=len(checkpoint['model']),**{k:v for k,v in checkpoint.items() if k!='model'}))
        del checkpoint
        assert time.monotonic()-start<120 and proc.memory_info().peak_wset<=2<<30
    flags['all_four_actual_snapshot_headers_match_original_binding']=True
    failed=json.loads((ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925/recovered_categorical_campaign_stored_adjudication_20261011.terminal.json').read_bytes())
    assert failed['exit_code']==1 and failed['error'] is not None
    original_failed=json.loads((ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925/original_categorical_campaign_result_20261010.terminal.json').read_bytes())
    identity=lambda x:(x['pid'],x['creation_time'])
    owned={identity(row):row for path in directory.rglob('native.receipt.json') for row in [json.loads(path.read_bytes())]}
    observed={identity(row):row for row in original_failed['observed_native_children']}
    assert len(owned)==len(observed)==115 and set(owned)==set(observed)
    for key,row in owned.items():assert row['command']==observed[key]['command'] and row['exit_code']==0
    flags['actual_completed_prefix_receipt_identities_match_original_observations']=True
    assert time.monotonic()-start<120 and proc.memory_info().peak_wset<=2<<30
    assert all(flags.values()) and not out.exists()
    result=dict(schema='RECOVERED_CATEGORICAL_CAMPAIGN_PREFLIGHT_V2',flags=flags,sources=sources,
        independent_math_core_sha256=hashlib.sha256(expected.encode()).hexdigest(),seconds=time.monotonic()-start,
        actual_snapshot_headers=observed_headers,original_binding=original_binding,OS_peak=proc.memory_info().peak_wset,
        full_snapshot_hashes_checked=False,snapshot_parameter_bits_checked=False,
        model_loads=0,history_forwards=0,backwards=0,native_calls=0,GPU_calls=0,optimizer_updates=0,
        scope='Small format and inherited-source checks only; actual complete audit remains unexecuted.')
    with out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
