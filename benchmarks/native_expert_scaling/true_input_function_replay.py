"""Actual captured BF16 module inputs:source generator replay and inverse diagnosis."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
from gate_input_recovery import replay,decode
START=time.monotonic();CLOCK=time.perf_counter();FIELDS=['gate','x','B','C','delta']


def prepare(a):
    import numpy as np,psutil
    cp=DOC/'true_generator_input_capture_result_v3_20261013.json';ca=DOC/'true_generator_input_capture_adjudication_20261013.json';op=DOC/'gate_input_recovery_result_20261013.json';oa=DOC/'gate_input_recovery_adjudication_20261013.json';r=json.loads(cp.read_bytes());ad=json.loads(ca.read_bytes());old=json.loads(op.read_bytes());oldad=json.loads(oa.read_bytes());assert ad['decision']=='INDEPENDENT_DATA_AND_SOURCE_CONTINUITY_PASS' and ad['report']==extent(cp) and oldad['decision']=='INDEPENDENT_STORED_PASS' and oldad['report']==extent(op);ob=json.loads(Path(old['binding']['path']).read_bytes());cbpath=DOC/'true_generator_input_capture_binding_v4_20261013.json';cb=json.loads(cbpath.read_bytes());assert cb['MuP_bits']==ob['semantics']['MuP_bits'] and r['MuP_all24_BF16_bits_exact'];storage=DOC/'true_input_function_replay_storage_20261013.json';st=json.loads(storage.read_text(encoding='utf-8-sig'));assert not a.out.exists() and not a.directory.exists() and str(a.directory.resolve())==str(Path(st['new_output_directory']).resolve());paths=[Path(__file__),B/'true_input_function_replay_audit.py',B/'gate_input_recovery.py',B/'falcon_auxiliary_s6_transport.py',DOC/'TRUE_INPUT_FUNCTION_REPLAY_PROTOCOL_20261013.md',DOC/'TRUE_INPUT_FUNCTION_REPLAY_AUDIT_PROTOCOL_20261013.md',storage,cp,ca,op,oa,Path(old['binding']['path']),cbpath,Path(ob['source_weights']),Path(ob['source_config']),Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),Path(np._core._multiarray_umath.__file__),Path(np.linalg._umath_linalg.__file__),Path(psutil.__file__)];nr=Path(np.__file__).resolve().parent;pr=Path(psutil.__file__).resolve().parent;paths+=list((nr.parent/'numpy.libs').glob('*.dll'))+list(pr.glob('*.pyd'));paths += [Path(q['file']['path']) for q in old['records']]
    for rec in r['records']:
        for site in [0,12,23]:paths += [Path(rec['sites'][site]['fields'][key]['path']) for key in ['H',*FIELDS]]
    paths += [ROOT/p for p in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']];inputs=[extent(p) for p in dict.fromkeys(paths)];assert time.monotonic()-START<900 and psutil.Process().memory_info().peak_wset<6<<30
    b=dict(schema='TRUE_INPUT_FUNCTION_REPLAY_BINDING_V1',freeze=a.freeze,python=str(Path(sys.executable).resolve()),versions=dict(numpy=np.__version__,psutil=psutil.__version__),capture=extent(cp),capture_adjudication=extent(ca),inverse_report=extent(op),inverse_adjudication=extent(oa),source_weights=ob['source_weights'],source_config=ob['source_config'],semantics=ob['semantics'],sites=[0,12,23],fields=FIELDS,inputs=inputs,output_directory=str(a.directory.resolve()),numeric=dict(audit_vector_relative=1e-6,audit_scalar_relative=1e-8),consistency=dict(each_case_each_field_relative_RMS=.01),limits=dict(seconds=900,audit_seconds=1200,OS_bytes=6<<30,output_bytes=2<<30),scope='Actual observed source BF16 H,AFTER .625/BEFORE in_proj,all144 source generator/conv/activation/delta replays;inverse H diagnostics only,no input choice/fit/source/GPU/native/RESERVED/causal/quality/speed admission. Explicit promoted+rounded equations are not automatically original CUDA reduction bits.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.perf_counter()-CLOCK)),flush=True)


def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil'] and not a.directory.exists() and not a.out.exists() and str(a.directory.resolve())==b['output_directory'];proc=psutil.Process();proc.cpu_affinity(list(range(6)));rows=[];stream=None;stage='startup';receipt=dict(schema='TRUE_INPUT_FUNCTION_REPLAY_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir())<=b['limits']['output_bytes']
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    def raw(e):assert e['dtype']=='BF16';return decode(np.fromfile(e['path'],'<u2').reshape(e['shape']),np)
    def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    def mm(x,y,pos):return dict(relative_RMS=rel(x,y),centered_relative_RMS=rel(x-x.mean(0),y-y.mean(0)),label_relative_RMS=rel(x[pos],y[pos]))
    try:
        check();a.directory.mkdir();get,stream=tensor_reader(b['source_weights'],np);capture=json.loads(Path(b['capture']['path']).read_bytes())['records'];old=json.loads(Path(b['inverse_report']['path']).read_bytes());oldrows={(q['site'],q['id']):q for q in old['records']};mu=decode(np.array(b['semantics']['MuP_bits'],'u2'),np)
        for site in b['sites']:
            guard();prefix=f'model.layers.{site}.mamba.';W=get(prefix+'in_proj.weight');cw=get(prefix+'conv1d.weight')[:,0];cb=get(prefix+'conv1d.bias');db=get(prefix+'dt_bias')
            for rec in capture:
                stage=f'{site}/{rec["id"]}';guard();n=rec['history'];f=rec['sites'][site]['fields'];H=raw(f['H']);assert H.shape==(n,2048);prior=oldrows[(site,rec['id'])]
                with np.load(prior['file']['path'],allow_pickle=False) as z:oh=z['H_unrounded'].copy();ob=z['H_candidate_BF16'].copy()
                inv=dict(unrounded=mm(oh,H,rec['positions']),rounded=mm(decode(ob,np),H,rec['positions']),BF16_bit_mismatches=int(np.count_nonzero(ob!=np.fromfile(f['H']['path'],'<u2').reshape(n,2048))));pred=replay(H,W,cw,cb,db,mu,b['semantics']['delta_min'],b['semantics']['delta_max'],np);metrics={};flags={};mismatch={}
                for key in FIELDS:
                    expected=raw(f[key]);metrics[key]=mm(decode(pred[key],np),expected,rec['positions']);flags[key]=metrics[key]['relative_RMS']<=b['consistency']['each_case_each_field_relative_RMS'];mismatch[key]=int(np.count_nonzero(pred[key]!=np.fromfile(f[key]['path'],'<u2').reshape(f[key]['shape'])))
                path=a.directory/(rec['id']+f'.site{site:02d}.npz');np.savez(path,**{'predicted_'+key:value for key,value in pred.items()});rows.append(dict(id=rec['id'],site=site,split=rec['split'],domain=rec['domain'],history=n,H_actual=f['H'],inverse_file=prior['file'],inverse_diagnostic=inv,file=extent(path),metrics=metrics,bit_mismatches=mismatch,consistency_flags=flags));print(json.dumps(dict(stage=stage,complete=len(rows),flags=flags,seconds=time.perf_counter()-CLOCK)),flush=True)
        stream.close();stream=None;check();guard();assert len(rows)==144;aa={}
        for site in b['sites']:
            aa[str(site)]={}
            for split in ['FIT','DEV']:
                rr=[q for q in rows if q['site']==site and q['split']==split];assert len(rr)==24;aa[str(site)][split]={}
                for key in FIELDS:
                    vals=[q['metrics'][key]['relative_RMS'] for q in rr];dm={d:float(np.mean([q['metrics'][key]['relative_RMS'] for q in rr if q['domain']==d])) for d in {q['domain'] for q in rr}};aa[str(site)][split][key]=dict(mean=float(np.mean(vals)),worst=max(vals),worst_domain_mean=max(dm.values()),domain_means=dm,failed_cases=sum(not q['consistency_flags'][key] for q in rr),bit_mismatches=sum(q['bit_mismatches'][key] for q in rr))
        success=all(all(q['consistency_flags'].values()) for q in rows);r=dict(schema='TRUE_INPUT_FUNCTION_REPLAY_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,records=rows,aggregates=aa,calibration_consistency_all_fields=success,before_after_hashes_exact=True,actual_module_inputs=144,source_forwards=0,optimizer_updates=0,native_calls=0,recovery_updates=0,decision='COMPLETE_PENDING_INDEPENDENT_AUDIT',output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()),seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope=b['scope']);write(a.out,r);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(result=extent(a.out),consistent=success)),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),stage=stage,traceback=traceback.format_exc())
        if a.directory.exists():write(a.directory/'first_fault.json',dict(stage=stage,error=repr(e),records=rows))
        raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    for name in ['binding','directory','out']:p.add_argument('--'+name,type=Path)
    for name in ['binding-sha','freeze']:p.add_argument('--'+name)
    a=p.parse_args()
    if a.run:run(a)
    else:
        import psutil
        proc=psutil.Process();receipt=dict(schema='TRUE_INPUT_FUNCTION_REPLAY_PREPARATION_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
        try:prepare(a);receipt.update(exit_code=0,error=None,result=extent(a.out))
        except BaseException as e:
            import traceback
            receipt.update(exit_code=1,error=repr(e),traceback=traceback.format_exc());raise
        finally:
            receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.preparation_terminal.json'),receipt)
