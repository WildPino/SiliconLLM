"""Independent all-field/core/integer/continuation audit; no producer import."""
import argparse,ctypes,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
from full_channel_native import interface as prefix_interface,buffers as prefix_buffers,ptrs
START=time.monotonic()

def main(a):
    import numpy as np,psutil
    re,te=extent(a.report),extent(a.terminal);assert re['sha256']==a.report_sha and te['sha256']==a.terminal_sha;r=json.loads(a.report.read_bytes());t=json.loads(a.terminal.read_bytes());assert t['exit_code']==0 and t['result']==re and extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and dict(numpy=np.__version__,psutil=psutil.__version__)==b['versions'] and not a.out.exists();proc=psutil.Process();proc.cpu_affinity(list(range(6)));stage='startup';maximum={};gates={};rows=[];counts=dict(field_words=0,projected_words=0,latent_words=0,final_state_words=0,norm_words=0,integer_input_words=0,integer_dot_words=0,raw_words=0,BF16_words=0,integer_mismatches=0,raw_bit_mismatches=0,BF16_mismatches=0,token_IDs=0,masses=0);receipt=dict(schema='TOKEN_STATE_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():assert time.monotonic()-START<=b['limits']['audit_seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
    def check():
        for e in b['inputs']+[r['binding'],re,te]+[q['file'] for q in r['records']]:guard();assert extent(e['path'])==e,e['path']
    def dec(x):return (x.astype('u4')*65536).view('f4').astype('f8')
    def bf(x):
        u=x.astype('f4').view('u4');lo=u&65535;hi=u>>16;return (hi+((lo>32768)|((lo==32768)&((hi&1)!=0))).astype('u4')).astype('u2')
    def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    def measure(k,v,tol):maximum[k]=max(maximum.get(k,0.),float(v));gates[k]=bool(gates.get(k,True) and v<=tol)
    def meta(x,y):
        if isinstance(y,dict):
            assert x.keys()==y.keys()
            for k in y:meta(x[k],y[k])
        elif isinstance(y,(bool,int,str)):assert x==y,(stage,x,y)
        else:measure('metadata',abs(x-y)/max(abs(y),1.),b['numeric']['metadata_absolute_or_relative'])
    def metrics(x,y,pos):return dict(relative_RMS=rel(x,y),centered_relative_RMS=rel(x-x.mean(0),y-y.mean(0)),label_relative_RMS=rel(x[pos],y[pos]))
    try:
        check();ga=json.loads(Path(b['generator_adjudication']['path']).read_bytes());assert ga['decision']=='INDEPENDENT_STORED_PASS' and ga['report']==b['generator_report']
        with np.load(b['core_parameters']['path'],allow_pickle=False) as z:p={k:z[k].copy() for k in z.files}
        with np.load(b['readout_packet']['path'],allow_pickle=False) as z:q=z['q'].copy();scale=z['scale'].copy()
        assert q.shape==(2048,3072) and np.max(np.abs(q.astype('i4')))<=364;qd=q.astype('f8');table=np.memmap(b['table']['path'],dtype='u2',mode='r',shape=(65537,6704));wp=ptrs([p[k] for k in ['Ws','Vs','Alog','D','gamma','O','in_proj']]);prefix_dll,prefix=prefix_interface(dict(native_dll=b['prefix_native_dll']));cases={e['id']:e for e in b['records']};assert len(r['records'])==48 and {e['id'] for e in r['records']}==set(cases);heads=np.repeat(np.arange(48),64)
        for e in r['records']:
            stage=e['id'];guard();rec=cases[e['id']];n=rec['history'];assert all(e[k]==rec[k] for k in ['id','history','split','domain']) and e['seen']==n
            with np.load(e['file']['path'],allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
            assert set(v)=={'Bp','Cp','latent','fields','norm','output','r','dots','a','raw','final_state','final_conv','token_IDs','implicit_masses'};ids=np.array(rec['input_ids'],'u4');assert v['token_IDs'].dtype==np.dtype('u4') and np.array_equal(v['token_IDs'],ids) and v['implicit_masses'].dtype==np.dtype('f4') and np.array_equal(v['implicit_masses'],np.ones(n,'f4'));counts['token_IDs']+=n;counts['masses']+=n
            dims={'Bp':((n,96),'f4'),'Cp':((n,96),'f4'),'latent':((n,3072),'f4'),'fields':((n,6704),'u2'),'norm':((n,3072),'u2'),'output':((n,2048),'u2'),'r':((n,3072),'i2'),'dots':((n,2048),'i8'),'a':((n,),'f8'),'raw':((n,2048),'f8'),'final_state':((3072,96),'f4'),'final_conv':((3,3584),'u2')}
            for k,(shape,dt) in dims.items():assert v[k].shape==shape and v[k].dtype==np.dtype(dt) and np.isfinite(v[k]).all()
            ring=np.zeros((3,3584),'u2')
            for t0 in range(max(0,n-3),n):ring[t0%3]=table[ids[t0],3072:6656]
            assert np.array_equal(ring,v['final_conv']);gen=next(x for x in b['generator_records'] if x['id']==e['id']);assert e['generator_reference']==gen['file'];bounds=[0,3072,6144,6400,6656,6704];names=['gate','x','B','C','delta'];fields={k:dec(v['fields'][:,bounds[i]:bounds[i+1]]) for i,k in enumerate(names)};errors={};differences={}
            with np.load(gen['file']['path'],allow_pickle=False) as z:
                for i,k in enumerate(names):errors[k]=rel(fields[k],dec(z[k]));differences[k]=int(np.count_nonzero(v['fields'][:,bounds[i]:bounds[i+1]]!=z[k]));measure('field_'+k,errors[k],b['numeric']['field_relative'])
            meta(e['field_reference_relative'],errors);assert e['field_reference_bit_differences']==differences;counts['field_words']+=v['fields'].size
            # All old-qualified prefix bits independently replayed on actual native fields.
            f,bb=prefix_buffers(n,np);inp=[np.ascontiguousarray(fields[k],dtype='f4') for k in ['x','B','C','delta','gate']];unused=np.zeros(n,'f8');times=np.zeros(4,'f8');proc.cpu_affinity([0]);assert prefix(n,1,wp,ptrs(inp),unused.ctypes.data,ptrs(f),ptrs(bb),times.ctypes.data,0)==0;proc.cpu_affinity(list(range(6)))
            exact=all(np.array_equal(v[k].view('u4'),x.view('u4')) for k,x in [('Bp',f[0]),('Cp',f[1]),('latent',f[2]),('final_state',f[3])]) and np.array_equal(v['norm'],bb[1]);gates['prefix_core_bits']=bool(gates.get('prefix_core_bits',True) and exact)
            bp=fields['B']@p['Ws'].astype('f8').T;cp=fields['C']@p['Vs'].astype('f8').T;measure('B_projection',rel(v['Bp'].astype('f8'),bp),b['numeric']['projected_relative']);measure('C_projection',rel(v['Cp'].astype('f8'),cp),b['numeric']['projected_relative']);counts['projected_words']+=v['Bp'].size+v['Cp'].size;state=np.zeros((3072,96),'f8');lat=np.empty((n,3072),'f8');A=-np.exp(p['Alog'].astype('f8'))
            for t0 in range(n):
                if t0%32==0:guard()
                dt=fields['delta'][t0,heads];state*=np.exp(dt*A[heads])[:,None];state+=(dt*fields['x'][t0])[:,None]*v['Bp'][t0].astype('f8')[None,:];lat[t0]=np.sum(state*v['Cp'][t0].astype('f8')[None,:],axis=1)+p['D'].astype('f8')[heads]*fields['x'][t0]
            measure('scan',rel(v['latent'].astype('f8'),lat),b['numeric']['scan_relative']);measure('final_state',rel(v['final_state'].astype('f8'),state),b['numeric']['state_relative']);counts['latent_words']+=lat.size;counts['final_state_words']+=state.size;gate=fields['gate'];ug=v['latent'].astype('f8')*(gate*np.exp(-np.logaddexp(0.,-gate)));norm=bf((ug/np.sqrt(np.mean(ug**2,axis=1,keepdims=True)+1e-5))*p['gamma'].astype('f8'));measure('own_norm_smooth',rel(dec(v['norm']),dec(norm)),b['numeric']['controlled_relative']);counts['norm_words']+=norm.size
            Z=dec(v['norm']);aa=np.max(np.abs(Z),axis=1)/32767;rr=np.clip(np.rint(np.divide(Z,aa[:,None],out=np.zeros_like(Z),where=aa[:,None]!=0)),-32767,32767).astype('i2');assert np.array_equal(v['r'],rr) and np.array_equal(v['a'].view('u8'),aa.view('u8'));counts['integer_input_words']+=rr.size;gold=np.empty_like(v['raw'])
            for lo in range(0,n,32):gold[lo:lo+32]=np.einsum('td,od->to',rr[lo:lo+32].astype('f8'),qd,optimize=True)
            assert np.array_equal(gold,np.rint(gold)) and np.max(np.abs(gold))<=36640321536;dm=int(np.count_nonzero(v['dots']!=gold.astype('i8')));gold=gold*aa[:,None]*scale;ym=int(np.count_nonzero(v['raw'].view('u8')!=gold.view('u8')));bm=int(np.count_nonzero(v['output']!=bf(gold)));counts['integer_mismatches']+=dm;counts['raw_bit_mismatches']+=ym;counts['BF16_mismatches']+=bm;counts['integer_dot_words']+=v['dots'].size;counts['raw_words']+=v['raw'].size;counts['BF16_words']+=v['output'].size
            # Re-execute complete continuation in a separate process/program.
            native=ctypes.CDLL(b['compile']['dll']['path']);create=native.ts_create;create.argtypes=[];create.restype=ctypes.c_void_p;destroy=native.ts_destroy;destroy.argtypes=[ctypes.c_void_p];destroy.restype=None;block=native.ts_block;block.argtypes=[ctypes.c_void_p,ctypes.c_int]+[ctypes.c_void_p]*12;block.restype=ctypes.c_int;snapshot=native.ts_snapshot;snapshot.argtypes=[ctypes.c_void_p]*4;snapshot.restype=ctypes.c_int;last=native.ts_last;last.argtypes=[ctypes.c_void_p];last.restype=ctypes.c_uint32
            with np.load(b['conv_parameters']['path'],allow_pickle=False) as z:cv={k:z[k].copy() for k in z.files}
            w=ptrs([p[k] for k in ['Ws','Vs','Alog','D','gamma']]);conv=ptrs([cv[k] for k in ['cw','cb','db']]);tmp={k:np.empty_like(v[k]) for k in ['Bp','Cp','latent','fields','norm','output','r','dots','a','raw']};ctx=create();assert ctx;continuation=True;proc.cpu_affinity([0])
            try:
                for ti in range(n):
                    if ti%64==0:guard()
                    ff=ptrs([tmp[k][ti:ti+1] for k in ['Bp','Cp','latent']]);bbptr=ptrs([tmp[k][ti:ti+1] for k in ['fields','norm','output']]);args=[tmp[k][ti:ti+1].ctypes.data for k in ['r','dots','a','raw']];assert block(ctx,1,ids[ti:ti+1].ctypes.data,table.ctypes.data,w,conv,q.ctypes.data,scale.ctypes.data,ff,bbptr,*args)==0;continuation=continuation and last(ctx)==ids[ti]
                st=np.empty((3072,96),'f4');rg=np.empty((3,3584),'u2');seen=np.zeros(1,'u8');assert snapshot(ctx,st.ctypes.data,rg.ctypes.data,seen.ctypes.data)==0;continuation=continuation and all(np.array_equal(tmp[k].view('u1'),v[k].view('u1')) for k in tmp) and np.array_equal(st.view('u4'),v['final_state'].view('u4')) and np.array_equal(rg,v['final_conv']) and seen[0]==n
            finally:destroy(ctx);proc.cpu_affinity(list(range(6)))
            gates['incremental_bits']=bool(gates.get('incremental_bits',True) and continuation);meta(e['numeric_gates'],dict(fields=all(val<=b['numeric']['field_relative'] for val in errors.values()),prefix_core_bits=bool(exact),incremental_bits=bool(continuation)))
            old=next(x for x in b['temporal_records'] if x['id']==e['id']);ir=next(x for x in b['integer_records'] if x['id']==e['id']);assert e['temporal_reference']==old['file'] and e['integer_reference']==ir['file']
            with np.load(old['file']['path'],allow_pickle=False) as z:stateerr=rel(v['final_state'].astype('f8'),z['final_state'].astype('f8'))
            with np.load(ir['file']['path'],allow_pickle=False) as z:base=metrics(dec(v['output']),dec(z['output_BF16']),rec['positions'])
            se=rec['sites'][0]['fields']['output'];source=np.fromfile(se['path'],'u2').reshape(se['shape']);sm=metrics(dec(v['output']),dec(source),rec['positions']);meta(e['metrics'],dict(source=sm,baseline=base));meta(e['final_state_relative'],stateerr);qg=dict(own_output=base['relative_RMS']<=.01,final_state=stateerr<=.01);assert e['quality_gates']==qg;rows.append(dict(split=e['split'],domain=e['domain'],metrics=dict(source=sm,baseline=base),quality_gates=qg));print(json.dumps(dict(complete=len(rows),seconds=time.monotonic()-START)),flush=True)
        aggregates={};donor={}
        for split in ['FIT','DEV']:
            rr=[q for q in rows if q['split']==split];assert len(rr)==24;aggregates[split]={}
            for kind in ['source','baseline']:
                vals=[q['metrics'][kind]['relative_RMS'] for q in rr];dm={d:float(np.mean([q['metrics'][kind]['relative_RMS'] for q in rr if q['domain']==d])) for d in sorted({q['domain'] for q in rr})};aggregates[split][kind]=dict(mean=float(np.mean(vals)),worst=max(vals),centered_mean=float(np.mean([q['metrics'][kind]['centered_relative_RMS'] for q in rr])),domain_means=dm,worst_domain_mean=max(dm.values()))
            d=aggregates[split]['source'];donor[split]=dict(mean=d['mean']<=.1,worst=d['worst']<=.2,centered=d['centered_mean']<=.1,domain=d['worst_domain_mean']<=.2)
        meta(r['aggregates'],aggregates);assert donor==r['donor_gates'] and r['quality_all_pass']==all(all(q['quality_gates'].values()) for q in rows) and r['controls']==dict(invalid_ID_rejected=True,token65536_supported=True);gates['integer_exact']=counts['integer_mismatches']==0;gates['raw_exact']=counts['raw_bit_mismatches']==0;gates['BF16_exact']=counts['BF16_mismatches']==0;assert r['token_steps']==22547 and r['initial_state_zero_histories']==48 and all(r[k]==0 for k in ['source_model_forwards','GPU_calls','optimizer_updates']);check();guard();ad=dict(schema='TOKEN_STATE_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,report=re,terminal=te,binding=r['binding'],counts=counts,maximum=maximum,numeric_gates=gates,numeric_all_pass=all(gates.values()),quality_all_pass=r['quality_all_pass'],donor_gates=donor,before_after_hashes_exact=True,decision='INDEPENDENT_STORED_PASS' if all(gates.values()) else 'INDEPENDENT_STORED_NUMERIC_FAILURE',seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,scope='Every stored field/core/norm/I16/I64/raw/BF16 word/ring/token ID/implicit mass/source and baseline metric verified. Full separate native continuation and exact old-qualified prefix core replay plus independent F64 projections/scan/norm/exact integer equations. Producer boundary-ID controls checked as receipts. Initial SSM branch only,no attention/FFN/all-layer chatbot/cost/DRAM/useful-n admission.');write(a.out,ad);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(adjudication=extent(a.out),decision=ad['decision'],counts=counts,maximum=maximum,quality_all_pass=ad['quality_all_pass'])),flush=True)
    except BaseException as er:
        import traceback
        receipt.update(exit_code=1,error=repr(er),stage=stage,traceback=traceback.format_exc());raise
    finally:receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['report','terminal','out']:p.add_argument('--'+k,type=Path,required=True)
    for k in ['report-sha','terminal-sha','freeze']:p.add_argument('--'+k,required=True)
    main(p.parse_args())
