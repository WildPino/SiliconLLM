"""528 terminal admission, including exact retained-handle compiler receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parent))
from meth528_operations import ROOT,DOC,write,output_bytes


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding-sha',required=True);ap.add_argument('--main-sha',required=True);ap.add_argument('--audit-sha',required=True);ap.add_argument('--exit-receipts',required=True);args=ap.parse_args()
    start=time.monotonic();proc=psutil.Process();hashed=0
    def item(path,expected=None):
        nonlocal hashed
        p=Path(path).resolve();st=p.stat();h=hashlib.sha256()
        with p.open('rb') as f:
            while data:=f.read(8<<20):
                h.update(data);hashed+=len(data);assert time.monotonic()-start<=60 and proc.memory_info().peak_wset<=128<<20
        assert (p.stat().st_size,p.stat().st_mtime_ns)==(st.st_size,st.st_mtime_ns)
        if expected:assert h.hexdigest()==expected,str(p)
        return dict(path=str(p),bytes=st.st_size,sha256=h.hexdigest())
    refs=[item(DOC/name,sha) for name,sha in (('meth528_binding.json',args.binding_sha),('meth528_main_result.json',args.main_sha),('meth528_audit_result.json',args.audit_sha))]
    b,m,a=[json.loads(Path(v['path']).read_bytes()) for v in refs]
    assert all(v['gates'] and all(v['gates'].values()) for v in (b,m,a)) and len(m['gates'])==7 and len(a['gates'])==8
    assert m['binding_sha256']==a['binding_sha256']==args.binding_sha and a['main_sha256']==args.main_sha
    assert not b['upstream_original_main_completed'] and not m['upstream_original523_main_completed'] and not a['upstream_original523_main_completed']
    assert len(a['compiler']['commands'])==len(a['compiler']['observed_instances'])==3 and all(v['exit_code']==0 for v in a['compiler']['commands'])
    winref=item(DOC/'meth528_windows_terminal.json');win=json.loads(Path(winref['path']).read_text(encoding='utf-8-sig'))
    assert len(win['stages'])==6 and set(win['positive_control_ids'])=={179810,179791} and all(v['query_available'] and not v['relevant_events'] for v in win['stages'])
    closed=[]
    instances=[(kind,raw['process_instance'],ref['sha256']) for kind,raw,ref in zip(('binding','main','audit'),(b,m,a),refs)]
    instances += [('compiler-'+v['label'],v,args.audit_sha) for v in a['compiler']['observed_instances']]
    for kind,inst,sha in instances:
        w=next(v for v in win['stages'] if v['kind']==kind);assert w['raw_sha256']==sha
        assert w['process_instance']['pid']==inst['pid'] and w['process_instance']['create_time_unix']==inst['create_time_unix']
        try:live=psutil.Process(inst['pid']);assert abs(live.create_time()-inst['create_time_unix'])>.002
        except psutil.NoSuchProcess:pass
        closed.append(dict(kind=kind,instance=inst,closed=True))
    resources=[]
    for kind,path,sha in (('binding',ROOT/'results/native_expert_scaling/meth528_binding_resource.json',args.binding_sha),('main',ROOT/'results/native_expert_scaling/meth528_atoms/terminal_resource.json',args.main_sha),('audit',ROOT/'results/native_expert_scaling/meth528_audit/terminal_resource.json',args.audit_sha)):
        ref=item(path);r=json.loads(path.read_bytes());assert r.get('binding_sha256',r.get('result_sha256'))==sha
        assert r['seconds']<=r['limits'][0] and r['parent_plus_compiler_peak_bound']<=r['limits'][1];resources.append(dict(kind=kind,receipt=ref,resource=r))
    for rel,sha in b['preserved'].items():item(ROOT/rel,sha)
    for v in b['legacy_output_inventory']:item(v['path'],v['sha256'])
    assert not any(p.is_file() for p in (ROOT/'results/native_expert_scaling/meth511_runtime/empty_cache').rglob('*'))
    retained=[item(v['path'],v['sha256']) for v in m['output_inventory']+a['output_inventory']]
    for k in ('cases','views','eligibility','decision','candidate_functions','control_functions','candidate_bank_bytes','new_candidate_physical_vectors','new_continuous_atom_vectors','new_negative_fold_vectors','new_exact_selected_WO_product_terms','omitted_zero_BYTE_equal_UIDs','global_hidden_max_preserved_UIDs'):assert m[k]==a[k]
    zero=('source_response_function_calls','full_source_WI_projection_rows','old_native_source_WO_replays','model_calls','readout_fits','consumed_rows_for_selection')
    assert all(m[k]==a[k]==0 for k in zero) and m['native_calls']==0 and a['native_calls']==a['new_C_verifier_calls']==384
    exits=json.loads(args.exit_receipts);assert len(exits)==3 and {v['kind'] for v in exits}=={'binding','main','audit'} and all(v['exit_code']==0 for v in exits)
    ret=DOC/'RETENTION_528_20261007.json';write(ret,dict(experiment='METH528',binding=refs[0],main=refs[1],audit=refs[2],local_outputs=retained,resources=resources,
        scope='One new atom main and one full separate scalar-C audit on retained consumed inputs; original523 resource failure remains FALSE.'))
    admission=DOC/'ADMISSION_528_20261007.json';write(admission,dict(experiment='METH528',main_completed=True,independent_audit_completed=True,
        gates=dict(all7_main_packed_original_atoms_quantizers_and_stats=True,all8_full_independent_scalar_C_byte_and_continuous_controls=True,
            six_actual_instances_closed_typed_UTC_zero_faults=True,original_output_foreign_cache_and_all_resource_bounds_preserved=True,
            unchanged_fixed_six_rare_byte_rotation_eligibility_and_upstream_failure=True),binding=refs[0],raw=refs[1],audit=refs[2],retention=item(ret),windows=winref,
        executor_exit_receipts=exits,closed_instances=closed,resources=resources,eligibility=m['eligibility'],decision=m['decision'],views=m['views'],
        candidate_bank_bytes=m['candidate_bank_bytes'],candidate_functions=127,control_functions=127,new_C_verifier_calls=384,upstream_original523_main_completed=False,
        **{k:0 for k in zero},physical_DRAM_verified=False,first_faults_or_repairs=[],
        scope='Local new variable-width source functions on previously consumed diagnostic data; no fresh quality, original source/prefix replay, useful large-n, engine speed/DRAM or whole promotion.'))
    size=output_bytes();assert size<=2<<30;print(json.dumps(dict(admission=item(admission),combined_output_bytes=size,closed_instances=len(closed),resource=dict(seconds=time.monotonic()-start,OS_peak_bytes=proc.memory_info().peak_wset,bytes_hashed=hashed))))


if __name__=='__main__':main()
