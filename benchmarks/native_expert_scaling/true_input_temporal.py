"""Fixed actual-input regenerated operands through the already qualified native state96 operator."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write
from full_channel_native import interface,buffers,ptrs,raw
START=time.monotonic();CLOCK=time.perf_counter();MODES=['full_source_denominator','reconstructed_denominator']


def aggregate(rows,np):
    aa={};gates={}
    for site in [0,12,23]:
        aa[str(site)]={};gates[str(site)]={}
        for mode in MODES:
            aa[str(site)][mode]={}
            for split in ['FIT','DEV']:
                rr=[q for q in rows if q['site']==site and q['split']==split];assert len(rr)==24
                for group in ['source','baseline']:
                    vals=[q['metrics'][group][mode]['output_relative_RMS'] for q in rr];center=float(np.mean([q['metrics'][group][mode]['centered_output_relative_RMS'] for q in rr]));dm={d:float(np.mean([q['metrics'][group][mode]['output_relative_RMS'] for q in rr if q['domain']==d])) for d in {q['domain'] for q in rr}}
                    aa[str(site)][mode].setdefault(split,{})[group]=dict(mean=float(np.mean(vals)),worst=max(vals),centered_mean=center,worst_domain_mean=max(dm.values()),domain_means=dm)
                if split=='DEV':
                    q=aa[str(site)][mode][split]['source'];gates[str(site)][mode]=dict(mean=q['mean']<=.1,worst=q['worst']<=.2,centered=q['centered_mean']<=.1,domain=q['worst_domain_mean']<=.2)
        for split in ['FIT','DEV']:
            rr=[q for q in rows if q['site']==site and q['split']==split];vals=[q['baseline_vectors']['final_state'] for q in rr];aa[str(site)].setdefault('final_state',{})[split]=dict(mean=float(np.mean(vals)),worst=max(vals),failed_cases=sum(not q['temporal_flags']['final_state'] for q in rr))
    return aa,gates


def prepare(a):
    import numpy as np,psutil
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));rp=DOC/'true_input_function_replay_result_20261013.json';ra=DOC/'true_input_function_replay_adjudication_20261013.json';npth=DOC/'full_channel_native_result_20261013.json';na=DOC/'full_channel_native_adjudication_20261013.json'
    r=json.loads(rp.read_bytes());ad=json.loads(ra.read_bytes());nr=json.loads(npth.read_bytes());nad=json.loads(na.read_bytes());assert ad['decision']==nad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(rp) and nad['report']==extent(npth) and r['calibration_consistency_all_fields'];rb=json.loads(Path(r['binding']['path']).read_bytes());nb=json.loads(Path(nr['binding']['path']).read_bytes());assert nb['sites']==[0,12,23] and nb['modes']==MODES
    actual=json.loads(Path(rb['capture']['path']).read_bytes())['records'];original=json.loads(Path(nb['capture']['path']).read_bytes())['records'];orig={q['id']:q for q in original};assert len(actual)==len(original)==48
    for q in actual:
        oq=orig[q['id']];assert all(q[k]==oq[k] for k in ['history','input_ids','positions','split','domain'])
        for site in [0,12,23]:
            for key in ['x','B','C','gate','delta','y','output']:
                ef=q['sites'][site]['fields'][key];of=oq['sites'][site]['fields'][key];assert all(ef[k]==of[k] for k in ['sha256','bytes','shape','dtype'])
    assert len(r['records'])==len(nr['records'])==144 and not a.out.exists() and not a.directory.exists();protocol=DOC/'TRUE_INPUT_TEMPORAL_PROTOCOL_20261013.md';storage=DOC/'true_input_temporal_storage_20261013.json';st=json.loads(storage.read_text(encoding='utf-8-sig'));assert str(a.directory.resolve())==str(Path(st['new_output_directory']).resolve())
    paths=[Path(__file__),B/'true_input_temporal_audit.py',DOC/'TRUE_INPUT_TEMPORAL_AUDIT_PROTOCOL_20261013.md',B/'full_channel_native.py',B/'composed_operator_mixed.py',B/'falcon_auxiliary_s6_transport.py',protocol,storage,rp,ra,npth,na,Path(r['binding']['path']),Path(nr['binding']['path'])];extra=[extent(q) for q in paths]+[q['file'] for q in r['records']]+[q['file'] for q in nr['records']];inputs=list({q['path']:q for q in nb['inputs']+rb['inputs']+extra}.values())
    for e in inputs:assert extent(e['path'])==e,e['path']
    assert time.monotonic()-START<=900 and proc.memory_info().peak_wset<=6<<30 and not proc.children(recursive=True)
    b=dict(schema='TRUE_INPUT_TEMPORAL_BINDING_V1',python=rb['python'],versions=dict(numpy=np.__version__,psutil=psutil.__version__),freeze=a.freeze,native_dll=nb['native_dll'],reference_dll=nb['reference_dll'],exports=nb['exports'],capture=nb['capture'],parent=nb['parent'],state_report=nb['state_report'],native_binding=nr['binding'],native_report=extent(npth),native_adjudication=extent(na),generator_binding=r['binding'],generator_report=extent(rp),generator_adjudication=extent(ra),sites=[0,12,23],modes=MODES,inputs=inputs,output_directory=str(a.directory.resolve()),numeric=nb['numeric'],source_criteria=nb['criteria'],temporal_criteria=dict(each_case_own_output_relative=.01,each_case_final_state_relative=.01),limits=dict(seconds=900,audit_seconds=1200,OS_bytes=6<<30,output_bytes=6<<30),scope='All144 histories,only supplied source x/B/C/delta/gate replaced by frozen actual-input replay arrays;SAME native DLL/FIT state maps/actual source full skip/norm/output;source denominator fixed to original y/gate. No fit/new recovery/source/GPU/RESERVED/T4/causal/chatbot/speed/n admission.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.perf_counter()-CLOCK)),flush=True)


def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil'] and not a.out.exists() and not a.directory.exists() and str(a.directory.resolve())==b['output_directory'];proc=psutil.Process();proc.cpu_affinity([0]);rows=[];stage='startup'
    receipt=dict(schema='TRUE_INPUT_TEMPORAL_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=b['limits']['output_bytes']
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    def load(e):
        with np.load(e['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def dec(z):assert z.dtype==np.dtype('<u2');return (z.astype('<u4')<<16).view('<f4').astype('f8')
    def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    def metrics(x,y,pos):return dict(output_relative_RMS=rel(x,y),centered_output_relative_RMS=rel(x-x.mean(0),y-y.mean(0)),label_output_relative_RMS=rel(x[pos],y[pos]))
    try:
        check();a.directory.mkdir();dll,fn=interface(b);capture=json.loads(Path(b['capture']['path']).read_bytes())['records'];recov=json.loads(Path(b['generator_report']['path']).read_bytes());base=json.loads(Path(b['native_report']['path']).read_bytes());ri={(q['site'],q['id']):q for q in recov['records']};bi={(q['site'],q['id']):q for q in base['records']}
        for packet in b['exports']:
            site=packet['site'];params=load(packet['parameters']);w=ptrs([params[k] for k in ['Ws','Vs','Alog','D','gamma','O','in_proj']])
            for rec in capture:
                stage=f'{site}/{rec["id"]}';guard();n=rec['history'];f=rec['sites'][site]['fields'];re=ri[(site,rec['id'])];be=bi[(site,rec['id'])];rr=load(re['file']);zz=load(be['file']);inp=[np.ascontiguousarray(dec(rr['predicted_'+k]),dtype='f4') for k in ['x','B','C','delta','gate']];original_gate=raw(f['gate'],np).astype('f8');y=raw(f['y'],np).astype('f8');g=original_gate*np.exp(-np.logaddexp(0.,-original_gate));inv=np.ascontiguousarray(1/np.sqrt(np.mean((y*g)**2,axis=1)+1e-5));floats,bits=buffers(n,np);times=np.zeros(4,'f8');rc=fn(n,0,w,ptrs(inp),inv.ctypes.data,ptrs(floats),ptrs(bits),times.ctypes.data,0);assert rc==0 and all(np.isfinite(q).all() for q in floats);arrays={k:q for k,q in zip(['state96_B','state96_C','latent','final_state'],floats[:4])};vectors={k:rel(q.astype('f8'),zz[k].astype('f8')) for k,q in arrays.items()};ref=raw(f['output'],np).astype('f8');mm=dict(source={},baseline={})
                for i,mode in enumerate(MODES):
                    arrays[mode+'__normalized']=bits[i];arrays[mode+'__raw_scores']=floats[4+i];arrays[mode+'__output']=bits[2+i];out=dec(bits[2+i]);mm['source'][mode]=metrics(out,ref,rec['positions']);mm['baseline'][mode]=metrics(out,dec(zz[mode+'__output']),rec['positions']);vectors[mode+'__normalized']=rel(dec(bits[i]),dec(zz[mode+'__normalized']));vectors[mode+'__raw_scores']=rel(floats[4+i].astype('f8'),zz[mode+'__raw_scores'].astype('f8'))
                flags=dict(own_output=mm['baseline']['reconstructed_denominator']['output_relative_RMS']<=b['temporal_criteria']['each_case_own_output_relative'],final_state=vectors['final_state']<=b['temporal_criteria']['each_case_final_state_relative']);path=a.directory/(rec['id']+f'.site{site:02d}.npz');np.savez(path,**arrays);rows.append(dict(id=rec['id'],site=site,split=rec['split'],domain=rec['domain'],history=n,generator=re['file'],baseline=be['file'],file=extent(path),metrics=mm,baseline_vectors=vectors,temporal_flags=flags));print(json.dumps(dict(stage=stage,complete=len(rows),flags=flags,seconds=time.perf_counter()-CLOCK)),flush=True)
        check();guard();assert len(rows)==144;aa,gates=aggregate(rows,np);tf={key:all(q['temporal_flags'][key] for q in rows) for key in ['own_output','final_state']};r=dict(schema='TRUE_INPUT_TEMPORAL_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,records=rows,aggregates=aa,gates=gates,temporal_gates=tf,temporal_all_pass=all(tf.values()),before_after_hashes_exact=True,native_history_calls=144,new_native_scans=144,source_forwards=0,optimizer_updates=0,basis_fits=0,generator_updates=0,whole_model_quality_admitted=False,whole_model_speed_admitted=False,decision='COMPLETE_PENDING_FULL_AUDIT',output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()),seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope=b['scope']);write(a.out,r);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(result=extent(a.out),temporal_gates=tf,source_gates=gates)),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),stage=stage,traceback=traceback.format_exc())
        if a.directory.exists():write(a.directory/'first_fault.json',dict(stage=stage,error=repr(e),records=rows))
        raise
    finally:
        receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    for name in ['binding','directory','out']:p.add_argument('--'+name,type=Path)
    for name in ['binding-sha','freeze']:p.add_argument('--'+name)
    a=p.parse_args()
    if a.run:run(a)
    else:
        import psutil
        proc=psutil.Process();receipt=dict(schema='TRUE_INPUT_TEMPORAL_PREPARATION_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
        try:prepare(a);receipt.update(exit_code=0,error=None,result=extent(a.out))
        except BaseException as e:
            import traceback
            receipt.update(exit_code=1,error=repr(e),traceback=traceback.format_exc());raise
        finally:
            receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.preparation_terminal.json'),receipt)
