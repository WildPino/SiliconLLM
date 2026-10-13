"""Ordered real-arithmetic state-error mechanism;independent survival-sum audit."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write
START=time.monotonic();CLOCK=time.perf_counter();TERMS=['decay','delta','x','B']


def main(a):
    import numpy as np,psutil
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));stage='startup';rows=[];numeric=[];b=None
    receipt=dict(schema='TEMPORAL_ERROR_DECOMPOSITION_TERMINAL_V1',mode=a.mode,command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def load(e):
        with np.load(e['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def dec(z):assert z.dtype==np.dtype('<u2');return (z.astype('<u4')<<16).view('<f4').astype('f8')
    def raw(e):assert e['dtype']=='BF16';return dec(np.fromfile(e['path'],'<u2').reshape(e['shape']))
    def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    def guard():assert time.monotonic()-START<=300 and proc.memory_info().peak_wset<=2<<30 and not proc.children(recursive=True)
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    def inputs(item):
        rr=load(item['recovery']);old=load(item['baseline']);new=load(item['native']);params=load(item['parameters']);x=raw(item['x']);dt=raw(item['delta']);xh=dec(rr['predicted_x']);dh=dec(rr['predicted_delta']);bp=old['state96_B'].astype('f8');bh=new['state96_B'].astype('f8');aa=-np.exp(params['Alog'].astype('f8'));assert x.shape==xh.shape==(item['history'],3072) and dt.shape==dh.shape==(item['history'],48) and bp.shape==bh.shape==(item['history'],96);return old,new,x,dt,xh,dh,bp,bh,aa
    try:
        assert not a.out.exists()
        if a.mode=='prepare':
            rp=DOC/'recovered_input_temporal_result_20261013.json';ap=DOC/'recovered_input_temporal_adjudication_20261013.json';r=json.loads(rp.read_bytes());ad=json.loads(ap.read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(rp) and ad['temporal_gates']==r['temporal_gates'];tb=json.loads(Path(r['binding']['path']).read_bytes());capture=json.loads(Path(tb['capture']['path']).read_bytes())['records'];cases={q['id']:q for q in capture};packets={q['site']:q for q in tb['exports']};failed=[q for q in r['records'] if not all(q['temporal_flags'].values())];assert failed;assert not a.directory.exists();storage=json.loads((DOC/'temporal_error_decomposition_storage_20261013.json').read_text(encoding='utf-8-sig'));assert str(a.directory.resolve())==str(Path(storage['new_output_directory']).resolve());items=[]
            paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',DOC/'TEMPORAL_ERROR_DECOMPOSITION_PROTOCOL_20261013.md',DOC/'temporal_error_decomposition_storage_20261013.json',rp,ap,Path(r['binding']['path']),Path(tb['capture']['path']),Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),Path(np._core._multiarray_umath.__file__),Path(np.linalg._umath_linalg.__file__),Path(psutil.__file__)];nr=Path(np.__file__).resolve().parent;pr=Path(psutil.__file__).resolve().parent;paths+=list((nr.parent/'numpy.libs').glob('*.dll'))+list(pr.glob('*.pyd'))
            for q in failed:
                f=cases[q['id']]['sites'][q['site']]['fields'];item=dict(id=q['id'],site=q['site'],history=q['history'],split=q['split'],domain=q['domain'],parameters=packets[q['site']]['parameters'],recovery=q['recovery'],baseline=q['baseline'],native=q['file'],x=f['x'],delta=f['delta'],original_temporal_flags=q['temporal_flags']);items.append(item)
                paths += [Path(item[k]['path']) for k in ['parameters','recovery','baseline','native','x','delta']]
            b=dict(schema='TEMPORAL_ERROR_DECOMPOSITION_BINDING_V1',python=str(Path(sys.executable).resolve()),versions=dict(numpy=np.__version__,psutil=psutil.__version__),freeze=a.freeze,parent_report=extent(rp),parent_adjudication=extent(ap),items=items,terms=TERMS,inputs=[extent(p) for p in dict.fromkeys(paths)],output_directory=str(a.directory.resolve()),limits=dict(seconds=300,OS_bytes=2<<30,output_bytes=128<<20),numeric=dict(component_relative=1e-10,identity_source_state_relative=1e-10,native_source_state_relative=1e-4,scalar_relative=1e-8),scope='All failed temporal histories only,diagnostic ordered F64 forcing/decay decomposition,independent final survival-weight sums/native error in original-state units. No model choice/fit/source/GPU/RESERVED/true-input/causal qualification. Contributions depend on declared order;cross energies retained.')
            check();guard();write(a.out,b);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(binding=extent(a.out),cases=len(items))),flush=True);return
        assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert b['versions']==dict(numpy=np.__version__,psutil=psutil.__version__) and Path(sys.executable).resolve()==Path(b['python']).resolve();check()
        if a.mode=='run':assert not a.directory.exists() and str(a.directory.resolve())==b['output_directory'];a.directory.mkdir()
        if a.mode=='audit':
            re=extent(a.report);te=extent(a.terminal);assert re['sha256']==a.report_sha and te['sha256']==a.terminal_sha;r=json.loads(a.report.read_bytes());t=json.loads(a.terminal.read_bytes());assert t['exit_code']==0 and t['result']==re and r['binding']==extent(a.binding) and len(r['records'])==len(b['items'])
        for item in b['items']:
            stage=f'{item["site"]}/{item["id"]}';guard();old,new,x,dt,xh,dh,bp,bh,aa=inputs(item);n=item['history'];orig_a=np.exp(dt*aa);hat_a=np.exp(dh*aa);ids=np.repeat(np.arange(48),64)
            if a.mode=='run':
                so=np.zeros((3072,96));sh=so.copy();components={k:so.copy() for k in TERMS}
                for ti in range(n):
                    if ti%32==0:guard()
                    av=orig_a[ti,ids,None];hv=hat_a[ti,ids,None];d=dt[ti,ids,None];hd=dh[ti,ids,None];xx=x[ti,:,None];hx=xh[ti,:,None];bb=bp[ti,None,:];hb=bh[ti,None,:]
                    components['decay']=hv*components['decay']+(hv-av)*so;components['delta']=hv*components['delta']+(hd-d)*xx*bb;components['x']=hv*components['x']+hd*(hx-xx)*bb;components['B']=hv*components['B']+hd*hx*(hb-bb);so=av*so+d*xx*bb;sh=hv*sh+hd*hx*hb
                arrays=dict(S_original=so,S_regenerated=sh,error=sh-so,**components);base_norm=float(np.linalg.norm(old['final_state']));assert base_norm>0;gram=np.array([[float(np.vdot(components[i],components[j]))/base_norm**2 for j in TERMS] for i in TERMS]);arrays['gram']=gram;identity=float(np.linalg.norm(sum(components.values())-(sh-so))/base_norm);native_error=new['final_state'].astype('f8')-old['final_state'].astype('f8');numeric_rows=dict(identity_source_state_relative=identity,original_native_relative=rel(so,old['final_state'].astype('f8')),regenerated_native_relative=rel(sh,new['final_state'].astype('f8')),error_native_source_state_relative=float(np.linalg.norm(arrays['error']-native_error)/base_norm));metrics=dict(F64_error_relative=float(np.linalg.norm(arrays['error'])/base_norm),native_error_relative=float(np.linalg.norm(native_error)/base_norm),term_norms_source_relative={k:float(np.linalg.norm(v)/base_norm) for k,v in components.items()},cross_gram=gram.tolist(),numeric=numeric_rows);path=a.directory/(item['id']+f'.site{item["site"]:02d}.npz');np.savez(path,**arrays);rows.append(dict(id=item['id'],site=item['site'],file=extent(path),metrics=metrics))
            else:
                record=next(q for q in r['records'] if (q['id'],q['site'])==(item['id'],item['site']));assert extent(record['file']['path'])==record['file'];saved=load(record['file']);assert set(saved)=={'S_original','S_regenerated','error','gram',*TERMS};fresh={k:np.zeros((3072,96)) for k in ['S_original','S_regenerated',*TERMS]};wo=np.ones_like(orig_a);wh=np.ones_like(hat_a);wo[:-1]=np.cumprod(orig_a[:0:-1],axis=0)[::-1];wh[:-1]=np.cumprod(hat_a[:0:-1],axis=0)[::-1]
                for h in range(48):
                    guard();sl=slice(h*64,(h+1)*64);dd=dt[:,h,None];hd=dh[:,h,None];o=wo[:,h,None];v=wh[:,h,None];xx=x[:,sl];hx=xh[:,sl];fresh['S_original'][sl]=(o*dd*xx).T@bp;fresh['S_regenerated'][sl]=(v*hd*hx).T@bh;fresh['decay'][sl]=((v-o)*dd*xx).T@bp;fresh['delta'][sl]=(v*(hd-dd)*xx).T@bp;fresh['x'][sl]=(v*hd*(hx-xx)).T@bp;fresh['B'][sl]=(v*hd*hx).T@(bh-bp)
                fresh['error']=fresh['S_regenerated']-fresh['S_original'];base_norm=float(np.linalg.norm(old['final_state']));assert base_norm>0
                for key,value in fresh.items():
                    assert saved[key].shape==(3072,96) and saved[key].dtype==np.dtype('<f8') and np.isfinite(saved[key]).all() and np.isfinite(value).all();e=rel(saved[key],value);numeric.append(dict(case=stage,control=key,relative=e,threshold=b['numeric']['component_relative'],passed=e<=b['numeric']['component_relative']))
                gg=np.array([[float(np.vdot(saved[i],saved[j]))/base_norm**2 for j in TERMS] for i in TERMS]);assert saved['gram'].shape==(4,4) and np.allclose(saved['gram'],gg,rtol=0,atol=1e-12);ne=new['final_state'].astype('f8')-old['final_state'].astype('f8');identity=float(np.linalg.norm(sum(saved[k] for k in TERMS)-saved['error'])/base_norm);values=dict(F64_error_relative=float(np.linalg.norm(saved['error'])/base_norm),native_error_relative=float(np.linalg.norm(ne)/base_norm));scalars=record['metrics']
                for key,value in values.items():assert abs(scalars[key]-value)/max(abs(value),1)<=1e-8
                for k in TERMS:assert abs(scalars['term_norms_source_relative'][k]-np.linalg.norm(saved[k])/base_norm)<=1e-8
                assert np.array_equal(np.array(scalars['cross_gram']),gg);checks=dict(identity_source_state_relative=identity,original_native_relative=rel(saved['S_original'],old['final_state'].astype('f8')),regenerated_native_relative=rel(saved['S_regenerated'],new['final_state'].astype('f8')),error_native_source_state_relative=float(np.linalg.norm(saved['error']-ne)/base_norm))
                for key,value in checks.items():
                    assert abs(scalars['numeric'][key]-value)<=1e-8;tol=b['numeric']['identity_source_state_relative'] if key=='identity_source_state_relative' else b['numeric']['native_source_state_relative'];numeric.append(dict(case=stage,control=key,relative=value,threshold=tol,passed=value<=tol))
                assert abs(float(np.sum(gg))-values['F64_error_relative']**2)<=1e-12;rows.append(dict(id=item['id'],site=item['site'],word_count=7*3072*96,metrics=scalars))
        check();guard()
        if a.mode=='run':
            size=sum(p.stat().st_size for p in a.directory.iterdir());assert size<=b['limits']['output_bytes'];result=dict(schema='TEMPORAL_ERROR_DECOMPOSITION_RESULT_V1',binding=extent(a.binding),records=rows,decision='COMPLETE_PENDING_INDEPENDENT_SURVIVAL_AUDIT',before_after_hashes_exact=True,output_bytes=size,seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope=b['scope'])
        else:
            assert extent(a.report)==re and extent(a.terminal)==te;result=dict(schema='TEMPORAL_ERROR_DECOMPOSITION_AUDIT_V1',binding=extent(a.binding),report=re,terminal=te,records=rows,numeric_flags=numeric,decision='INDEPENDENT_STORED_PASS' if all(q['passed'] for q in numeric) else 'INDEPENDENT_STORED_NUMERIC_FAILURE',before_after_hashes_exact=True,seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope=b['scope'])
        write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(result=extent(a.out),decision=result['decision'],records=rows)),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),stage=stage,traceback=traceback.format_exc());raise
    finally:
        receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.'+a.mode+'_terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','run','audit'],required=True)
    for name in ['binding','directory','report','terminal','out']:p.add_argument('--'+name,type=Path)
    for name in ['binding-sha','report-sha','terminal-sha','freeze']:p.add_argument('--'+name)
    main(p.parse_args())
