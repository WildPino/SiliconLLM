"""Independent all-layer input/data/source coefficient witness and continuity audit."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
START=time.monotonic()


def main(a):
    import numpy as np,psutil
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));receipt=dict(schema='TRUE_GENERATOR_INPUT_CAPTURE_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False);rows=[];counts=dict(H_words=0,continuity_words=0,parameter_words=0);stage='startup'
    try:
        re=extent(a.report);te=extent(a.terminal);assert re['sha256']==a.report_sha and te['sha256']==a.terminal_sha;r=json.loads(a.report.read_bytes());t=json.loads(a.terminal.read_bytes());assert t['exit_code']==t['worker_exit_code']==0 and t['result']==re;b=json.loads(a.binding.read_bytes());be=extent(a.binding);assert be['sha256']==a.binding_sha==r['binding_sha256'] and not a.out.exists() and Path(sys.executable).resolve()==Path(b['python']).resolve();sources=[extent(Path(__file__)),extent(B/'falcon_auxiliary_s6_transport.py'),extent(DOC/'TRUE_GENERATOR_INPUT_CAPTURE_AUDIT_PROTOCOL_20261013.md')]
        def guard():assert time.monotonic()-START<=b['limits']['audit_seconds'] and proc.memory_info().peak_wset<=b['limits']['audit_OS_bytes'] and not proc.children(recursive=True)
        def check():
            for e in b['inputs']+sources+[be,re,te,r['parameter_before'],r['parameter_after'],r['source_method']]+[v for q in r['records'] for s in q['sites'] for v in s['fields'].values()]:guard();assert extent(e['path'])=={k:e[k] for k in ['path','bytes','sha256']},e['path']
        def raw(e):
            assert Path(e['path']).stat().st_size==e['bytes'];z=np.fromfile(e['path'],'<u2' if e['dtype']=='BF16' else '<f4').reshape(e['shape']);x=(z.astype('<u4')<<16).view('<f4').astype('f8') if e['dtype']=='BF16' else z.astype('f8');assert np.isfinite(x).all();return z,x
        def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
        check();old=json.loads(Path(b['prior_capture']['path']).read_bytes());prior={q['id']:q for q in old['records']};corpus={q['id']:q for q in json.loads(Path(b['corpus']).read_bytes())['records']};assert len(r['records'])==48 and [q['id'] for q in r['records']]==b['ordered_ids'];before=json.loads(Path(r['parameter_before']['path']).read_bytes());after=json.loads(Path(r['parameter_after']['path']).read_bytes());assert before==after and before['exact_source_BF16_values'] and len(before['tensors'])==411 and before['canonical_coefficient_words']==1554863488 and {q['name'] for q in before['tensors']}==set(b['checkpoint_tensors'])
        for q in before['tensors']:
            z=b['checkpoint_tensors'][q['name']];assert q['shape']==z['shape'] and q['canonical_BF16_sha256']==z['sha256'] and q['runtime_dtype'] in ['torch.bfloat16','torch.float32'];counts['parameter_words']+=int(np.prod(q['shape']))
        assert counts['parameter_words']==1554863488 and json.loads(Path(r['source_method']['path']).read_bytes())==json.loads(Path(b['prior_source_method']['path']).read_bytes());payload=0;input_sites=0;flags=[]
        for rec in r['records']:
            stage=rec['id'];guard();cp=corpus[rec['id']];pr=prior[rec['id']];assert rec['input_ids']==cp['student_input_ids']==pr['input_ids'] and rec['positions']==cp['positions']==pr['positions'] and rec['history']==len(rec['input_ids']) and rec['split']==cp['split'] and rec['domain']==cp['domain'] and [q['site'] for q in rec['sites']]==list(range(24));n=rec['history']
            for packet in rec['sites']:
                site=packet['site'];fields=packet['fields'];assert set(fields)==({'H'}|set(b['continuity_fields']) if site in b['continuity_sites'] else {'H'});e=fields['H'];assert e['dtype']=='BF16' and e['shape']==[n,2048] and e['bytes']==n*4096;hb,h=raw(e);counts['H_words']+=hb.size;input_sites+=1;payload+=e['bytes'];mm={}
                if site in b['continuity_sites']:
                    for key in b['continuity_fields']:
                        e=fields[key];pe=pr['sites'][site]['fields'][key];assert e['shape']==pe['shape'] and e['dtype']==pe['dtype'] and e['bytes']==pe['bytes'];bits,x=raw(e);pbits,y=raw(pe);counts['continuity_words']+=bits.size;payload+=e['bytes'];rr=rel(x,y);mm[key]=dict(relative_RMS=rr,centered_relative_RMS=rel(x-x.mean(0),y-y.mean(0)),label_relative_RMS=rel(x[rec['positions']],y[rec['positions']]),bit_mismatches=int(np.count_nonzero(bits.view('u2' if e['dtype']=='BF16' else 'u4')!=pbits.view('u2' if pe['dtype']=='BF16' else 'u4'))));flags.append(dict(id=rec['id'],site=site,field=key,relative=rr,threshold=b['criteria']['continuity_each_case_field_relative_RMS'],passed=rr<=b['criteria']['continuity_each_case_field_relative_RMS']))
                    rows.append(dict(id=rec['id'],site=site,split=rec['split'],domain=rec['domain'],history=n,metrics=mm))
            print(json.dumps(dict(stage=stage,complete=len(rows)//3,H_fields=input_sites,seconds=time.monotonic()-START)),flush=True)
        assert input_sites==r['H_fields']==1152 and payload==r['payload_bytes']==b['expected_payload_bytes'] and len(flags)==1008 and r['source_base_forwards']==48 and r['source_parameter_identities_versions_unchanged'] and r['source_parameter_bytes_exact_before_after'] and r['MuP_all24_BF16_bits_exact'] and all(r[k]==0 for k in ['LM_head_calls','source_generations','optimizer_updates','native_calls','reserved_queries']);aggregates={}
        for site in [0,12,23]:
            aggregates[str(site)]={}
            for split in ['FIT','DEV']:
                rr=[q for q in rows if q['site']==site and q['split']==split];assert len(rr)==24;aggregates[str(site)][split]={}
                for key in b['continuity_fields']:
                    vals=[q['metrics'][key]['relative_RMS'] for q in rr];dm={d:float(np.mean([q['metrics'][key]['relative_RMS'] for q in rr if q['domain']==d])) for d in {q['domain'] for q in rr}};aggregates[str(site)][split][key]=dict(mean=float(np.mean(vals)),worst=max(vals),worst_domain_mean=max(dm.values()),domain_means=dm,bit_mismatches=sum(q['metrics'][key]['bit_mismatches'] for q in rr))
        check();guard();result=dict(schema='TRUE_GENERATOR_INPUT_CAPTURE_AUDIT_V1',freeze=a.freeze,report=re,terminal=te,binding=be,records=rows,counts=counts,flags=flags,aggregates=aggregates,continuity_all_pass=all(q['passed'] for q in flags),decision='INDEPENDENT_DATA_AND_SOURCE_CONTINUITY_PASS' if all(q['passed'] for q in flags) else 'INDEPENDENT_DATA_CONTINUITY_FAILURE',before_after_hashes_exact=True,seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,scope='All1152 actual BF16 H arrays/all144 three-site full recurrent observations/all finite shapes/extents/hash/source before-after canonical coefficient witnesses/all continuity vectors and metrics. No independent CUDA execution repeat or all-layer generator/causal/chatbot/speed admission.');write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(decision=result['decision'],counts=counts,seconds=result['seconds'])),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),stage=stage,traceback=traceback.format_exc());raise
    finally:
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['report','terminal','binding','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['report-sha','terminal-sha','binding-sha','freeze']:p.add_argument('--'+name,required=True)
    main(p.parse_args())
