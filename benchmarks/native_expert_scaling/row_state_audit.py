"""Full row ABI archive identity,all24 source export and arbitrary-input math audit."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic()

def main(a):
    import numpy as np,psutil
    re,te=extent(a.report),extent(a.terminal);assert re['sha256']==a.report_sha and te['sha256']==a.terminal_sha;r=json.loads(a.report.read_bytes());t=json.loads(a.terminal.read_bytes());assert t['exit_code']==0 and t['result']==re and extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and dict(numpy=np.__version__,psutil=psutil.__version__)==b['versions'] and not a.out.exists();proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None;stage='startup';counts=dict(exported_sites=0,q_coefficients=0,core_and_conv_coefficients=0,archive_words=0,arbitrary_field_words=0,arbitrary_latent_words=0,arbitrary_dot_words=0,dot_mismatches=0,raw_bit_mismatches=0,BF16_mismatches=0);maximum={};gates={};receipt=dict(schema='ROW_STATE_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,model_admitted=False)
    def guard():assert time.monotonic()-START<=b['limits']['audit_seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
    def check():
        for e in b['inputs']+[r['binding'],re,te]+[q['file'] for q in r['records']+r['operator_controls']]:guard();assert extent(e['path'])==e,e['path']
    def dec(x):return (x.astype('u4')*65536).view('f4').astype('f8')
    def bf(x):
        u=x.astype('f4').view('u4');lo=u&65535;hi=u>>16;return (hi+((lo>32768)|((lo==32768)&((hi&1)!=0))).astype('u4')).astype('u2')
    def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    def test(k,x,y,tol):
        err=rel(x,y);maximum[k]=max(maximum.get(k,0.),err);gates[k]=bool(gates.get(k,True) and np.isfinite(err) and err<=tol)
    try:
        check();pa=json.loads(Path(b['parent_adjudication']['path']).read_bytes());assert pa['decision']=='INDEPENDENT_STORED_PASS' and pa['report']==b['parent_report'];assert r['nonfinite_input_rejected'] and r['all48_incremental_bits_exact'] and len(r['records'])==48;refs={q['id']:q for q in b['reference_records']}
        for rec in r['records']:
            stage=rec['id'];guard();ref=refs[rec['id']];assert rec['reference']==ref['file'] and all(rec[k]==ref[k] for k in ['id','history','split','domain'])
            with np.load(rec['file']['path'],allow_pickle=False) as z,np.load(ref['file']['path'],allow_pickle=False) as old:
                assert set(z.files)=={'Bp','Cp','latent','fields','norm','output','r','dots','a','raw','final_state','final_conv','tags','seen'};same=True
                for k in z.files:
                    if k=='tags':same=same and np.array_equal(z[k],old['token_IDs'])
                    elif k=='seen':same=same and z[k].dtype==np.dtype('u8') and z[k].shape==(1,) and z[k][0]==rec['history']
                    else:same=same and z[k].shape==old[k].shape and z[k].dtype==old[k].dtype and np.array_equal(z[k].view('u1'),old[k].view('u1'))
                    counts['archive_words']+=z[k].size
                assert rec['all_archive_bits_exact']==same;gates['archive_bits']=bool(gates.get('archive_bits',True) and same)
        get,stream=tensor_reader(b['source_weights'],np);table=np.memmap(b['table']['path'],dtype='u2',mode='r',shape=(65537,6704));tags=np.array(b['operator_control']['tags'],'u4');source_rows=dec(table[tags].copy());heads=np.repeat(np.arange(48),64);assert len(b['exports'])==len(r['operator_controls'])==24
        for e in b['exports']:
            site=e['site'];stage=f'parameters/{site}';guard();pre=f'model.layers.{site}.mamba.'
            with np.load(e['file']['path'],allow_pickle=False) as z:p={k:z[k].copy() for k in z.files}
            assert set(p)=={'Ws','Vs','Alog','D','gamma','cw','cb','db','q','scale'}
            with np.load(e['basis']['path'],allow_pickle=False) as z:W=z['W'].copy();V=z['V'].copy()
            assert W.shape==V.shape==(256,96) and np.max(np.abs(W.T@V-np.eye(96)))<=1e-8 and np.array_equal(p['Ws'],W.T.astype('f4')) and np.array_equal(p['Vs'],V.T.astype('f4'))
            for key,name,dtype in [('Alog','A_log','f4'),('D','D','f4'),('gamma','norm.weight','f4'),('cb','conv1d.bias','f8'),('db','dt_bias','f8')]:assert np.array_equal(p[key],get(pre+name).astype(dtype));counts['core_and_conv_coefficients']+=p[key].size
            assert np.array_equal(p['cw'],get(pre+'conv1d.weight')[:,0]);counts['core_and_conv_coefficients']+=p['cw'].size+p['Ws'].size+p['Vs'].size;O=get(pre+'out_proj.weight');s=np.max(np.abs(O),axis=1)/364;ratio=np.divide(O,s[:,None],out=np.zeros_like(O),where=s[:,None]!=0);absr=np.abs(ratio);fl=np.floor(absr);frac=absr-fl;qr=np.clip(np.sign(ratio)*(fl+((frac>.5)|((frac==.5)&((fl%2)!=0)))),-364,364).astype('i2');assert np.array_equal(qr,p['q']) and np.array_equal(s.view('u8'),p['scale'].view('u8'));counts['q_coefficients']+=qr.size;counts['exported_sites']+=1
            control=next(q for q in r['operator_controls'] if q['site']==site);assert control['parameters']==e['file'] and control['input_scope']==b['operator_control']['scope']
            with np.load(control['file']['path'],allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
            assert np.array_equal(v['tags'],tags) and v['seen'][0]==16;bc=source_rows[:,3072:6656];pad=np.pad(bc,((3,0),(0,0)));conv=np.stack([pad[t:t+16] for t in range(4)],axis=2);c=dec(bf(np.einsum('tck,ck->tc',conv,p['cw'],optimize=True)+p['cb']));sig=np.empty_like(c);positive=c>=0;sig[positive]=1/(1+np.exp(-c[positive]));ex=np.exp(c[~positive]);sig[~positive]=ex/(1+ex);act=dec(bf(c*sig));d=dec(bf(source_rows[:,6656:]+p['db']));sp=np.maximum(d,0)+np.log1p(np.exp(-np.abs(d)));sp[d>20]=d[d>20];fields=np.concatenate([bf(source_rows[:,:3072]),bf(act),bf(sp)],axis=1);test('fields',dec(v['fields']),dec(fields),b['numeric']['field_relative']);counts['arbitrary_field_words']+=fields.size;decoded=dec(v['fields']);x=decoded[:,3072:6144];bb=decoded[:,6144:6400];cc=decoded[:,6400:6656];dt=decoded[:,6656:];gate=decoded[:,:3072];test('B_projection',v['Bp'].astype('f8'),bb@p['Ws'].astype('f8').T,b['numeric']['projected_relative']);test('C_projection',v['Cp'].astype('f8'),cc@p['Vs'].astype('f8').T,b['numeric']['projected_relative']);state=np.zeros((3072,96));lat=np.empty((16,3072));A=-np.exp(p['Alog'].astype('f8'))
            for t0 in range(16):dd=dt[t0,heads];state*=np.exp(dd*A[heads])[:,None];state+=(dd*x[t0])[:,None]*v['Bp'][t0].astype('f8')[None,:];lat[t0]=np.sum(state*v['Cp'][t0].astype('f8')[None,:],axis=1)+p['D'][heads]*x[t0]
            test('scan',v['latent'].astype('f8'),lat,b['numeric']['scan_relative']);test('state',v['final_state'].astype('f8'),state,b['numeric']['state_relative']);counts['arbitrary_latent_words']+=lat.size;ug=v['latent'].astype('f8')*gate*np.exp(-np.logaddexp(0.,-gate));norm=bf((ug/np.sqrt(np.mean(ug**2,axis=1,keepdims=True)+1e-5))*p['gamma']);test('norm',dec(v['norm']),dec(norm),b['numeric']['controlled_relative']);Z=dec(v['norm']);aa=np.max(np.abs(Z),axis=1)/32767;rr=np.clip(np.rint(np.divide(Z,aa[:,None],out=np.zeros_like(Z),where=aa[:,None]!=0)),-32767,32767).astype('i2');assert np.array_equal(rr,v['r']) and np.array_equal(aa.view('u8'),v['a'].view('u8'));exact=rr.astype('f8')@qr.astype('f8').T;assert np.array_equal(exact,np.rint(exact)) and np.max(np.abs(exact))<=36640321536;counts['dot_mismatches']+=int(np.count_nonzero(v['dots']!=exact.astype('i8')));gold=exact*aa[:,None]*s;counts['raw_bit_mismatches']+=int(np.count_nonzero(v['raw'].view('u8')!=gold.view('u8')));counts['BF16_mismatches']+=int(np.count_nonzero(v['output']!=bf(gold)));counts['arbitrary_dot_words']+=exact.size;print(json.dumps(dict(audited_site=site,seconds=time.monotonic()-START)),flush=True)
        assert r['all48_archive_bits_exact']==gates['archive_bits'];gates['integer_exact']=counts['dot_mismatches']==0;gates['raw_exact']=counts['raw_bit_mismatches']==0;gates['BF16_exact']=counts['BF16_mismatches']==0;assert r['actual_initial_layer_positions']==22547 and r['arbitrary_operator_positions']==384 and all(r[k]==0 for k in ['source_model_forwards','GPU_calls','optimizer_updates']);stream.close();stream=None;check();guard();ad=dict(schema='ROW_STATE_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,report=re,terminal=te,binding=r['binding'],counts=counts,maximum=maximum,numeric_gates=gates,all_numeric_pass=all(gates.values()),before_after_hashes_exact=True,decision='INDEPENDENT_STORED_PASS' if all(gates.values()) else 'INDEPENDENT_STORED_NUMERIC_FAILURE',seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,scope='Every actual source/basis/q/core/conv export and48 initial-layer new-row history words checked to complete admitted archive;all24 arbitrary16-row controls independently verify projections/scan/norm/exact integers. Producer continuation and nonfinite controls checked as receipts;no later21 actual-input/model-quality/GPU/speed admission.');write(a.out,ad);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(adjudication=extent(a.out),decision=ad['decision'],counts=counts,maximum=maximum)),flush=True)
    except BaseException as er:
        import traceback
        receipt.update(exit_code=1,error=repr(er),stage=stage,traceback=traceback.format_exc());raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['report','terminal','out']:p.add_argument('--'+k,type=Path,required=True)
    for k in ['report-sha','terminal-sha','freeze']:p.add_argument('--'+k,required=True)
    main(p.parse_args())
