"""Independent all packed coefficients/full native output/control/source audit."""
import argparse,ctypes,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
START=time.monotonic()

def main(a):
    import numpy as np,psutil
    re,te=extent(a.report),extent(a.terminal);assert re['sha256']==a.report_sha and te['sha256']==a.terminal_sha;r=json.loads(a.report.read_bytes());t=json.loads(a.terminal.read_bytes());assert t['exit_code']==0 and t['result']==re and extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil'] and not a.out.exists();proc=psutil.Process();proc.cpu_affinity(list(range(6)));stage='startup';maximum={};verification={};rows=[];counts=dict(packed_codes=0,reconstructed_q=0,lut_F64_words=0,original_F32_words=0,original_F32_mismatches=0,lut_BF16_words=0,independent_BF16_mismatches=0);receipt=dict(schema='TERNARY_READOUT_NATIVE_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():assert time.monotonic()-START<=b['limits']['audit_seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
    def check():
        for e in b['inputs']+[r['binding'],re,te]+[q['file'] for q in r['records']]:guard();assert extent(e['path'])==e,e['path']
    def measure(key,value,tol):
        value=float(value);assert np.isfinite(value);maximum[key]=max(maximum.get(key,0.),value);verification[key]=bool(verification.get(key,True) and value<=tol)
    def meta(x,y):
        if isinstance(y,dict):
            assert x.keys()==y.keys()
            for k in y:meta(x[k],y[k])
        elif isinstance(y,(bool,int,str)):assert x==y,(stage,x,y)
        else:measure('metadata',abs(x-y)/max(abs(y),1.),b['numeric']['metadata_absolute_or_relative'])
    def decode(bits):return (bits.astype('<u4')*65536).view('<f4').astype('f8')
    def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    def mm(x,y,pos):return dict(relative_RMS=rel(x,y),centered_relative_RMS=rel(x-x.mean(0),y-y.mean(0)),label_relative_RMS=rel(x[pos],y[pos]))
    def bf(x):
        bits=x.astype('<f4').view('<u4');top=bits>>16;low=bits&65535;inc=(low>32768)|((low==32768)&((top&1)!=0));return (top+inc.astype('u4')).astype('u2')
    try:
        check();pad=json.loads(Path(b['parent_adjudication']['path']).read_bytes());assert pad['decision']=='INDEPENDENT_STORED_PASS' and pad['report']==b['parent_report'];parent=json.loads(Path(b['parent_binding']['path']).read_bytes());assert parent['records']==b['records']
        assert len(r['records'])==144
        for packet in b['packets']:
            site=packet['site'];stage=f'{site}/packed_coefficients';guard()
            with np.load(packet['file']['path'],allow_pickle=False) as z:codes=z['codes'].copy();scale=z['scale'].copy();original_q=z['q'].copy()
            q=np.zeros((2048,3072),'i4')
            for p in range(6):
                digits=np.empty((2048,3075),'i2')
                for k in range(5):digits[:,k::5]=((codes[p].T//3**k)%3).astype('i2')-1
                assert not digits[:,3072:].any();q+=3**p*digits[:,:3072]
            assert np.array_equal(q,original_q);counts['packed_codes']+=codes.size;counts['reconstructed_q']+=q.size;q=q.astype('f8')
            for e in [q for q in b['records'] if q['site']==site]:
                stage=f'{site}/{e["id"]}';guard()
                with np.load(e['native']['path'],allow_pickle=False) as z:Z=decode(z['reconstructed_denominator__normalized']);Y=z['reconstructed_denominator__raw_scores'].copy();base=z['reconstructed_denominator__output'].copy()
                precision=next(q['file'] for q in b['precision_records'] if q['site']==site and q['id']==e['id'])
                with np.load(precision['path'],allow_pickle=False) as z:pb=z['p6'].copy()
                original=next(q for q in r['records'] if q['site']==site and q['id']==e['id'])
                with np.load(original['file']['path'],allow_pickle=False) as z:raw=z['lut_scores'].copy();dense=z['original_scores'].copy();bits=z['lut_BF16'].copy();assert set(z.files)=={'lut_scores','original_scores','lut_BF16'}
                assert raw.shape==dense.shape==bits.shape==Y.shape==(e['history'],2048) and raw.dtype==np.dtype('<f8') and dense.dtype==np.dtype('<f4') and bits.dtype==np.dtype('<u2');assert np.isfinite(raw).all() and np.isfinite(dense).all();expected=np.empty_like(raw)
                for lo in range(0,len(Z),32):expected[lo:lo+32]=np.einsum('td,od->to',Z[lo:lo+32],q,optimize=True)*scale
                err=rel(raw,expected);measure('raw_full_LUT_vs_independent_q',err,b['numeric']['raw_relative']);assert np.array_equal(bits,bf(raw));different=int(np.count_nonzero(bits!=bf(expected)));counts['independent_BF16_mismatches']+=different;counts['lut_F64_words']+=raw.size;counts['lut_BF16_words']+=bits.size;counts['original_F32_words']+=dense.size;odiff=int(np.count_nonzero(dense.view('u4')!=Y.view('u4')));counts['original_F32_mismatches']+=odiff;verification['original_F32_exact']=bool(verification.get('original_F32_exact',True) and odiff==0)
                f=e['source_output'];source=decode(np.fromfile(f['path'],'<u2').reshape(f['shape']));out=decode(bits);metrics=dict(raw_vs_native=mm(raw,Y.astype('f8'),e['positions']),BF16_vs_baseline=mm(out,decode(base),e['positions']),BF16_vs_source=mm(out,source,e['positions']));ne=dict(raw_relative=err,BF16_smooth_relative=rel(out,decode(pb)),precision_BF16_mismatches=int(np.count_nonzero(bits!=pb)),original_F32_mismatches=odiff);ng=dict(raw=err<=b['numeric']['raw_relative'],BF16=ne['BF16_smooth_relative']<=b['numeric']['BF16_smooth_relative'],original_F32=odiff==0);meta(original['metrics'],metrics);meta(original['numeric_metrics'],ne);assert original['numeric_gates']==ng;extra=dict(raw=metrics['raw_vs_native']['relative_RMS']<=b['criteria']['additional_raw_each_case'],BF16=metrics['BF16_vs_baseline']['relative_RMS']<=b['criteria']['additional_BF16_each_case']);assert original['additional_gates']==extra;rows.append(dict(site=site,split=e['split'],domain=e['domain'],metrics=metrics,numeric_gates=ng,additional_gates=extra))
            # Scalar-controls independently repeat from source input and q,not producer helpers.
            ctl=next(q for q in r['controls'] if q['site']==site);fit=min((e for e in b['records'] if e['site']==site and e['split']=='FIT'),key=lambda e:(e['history'],e['id']));assert ctl['id']==fit['id'] and ctl['scalar_positions']==16 and ctl['table_patterns_exact']==243;assert ctl['scalar_relative']<=b['numeric']['scalar_relative'] and ctl['mutation_delta_relative']<=b['numeric']['scalar_relative'];old=int(codes[0,0,0]);assert ctl['mutation_old_code']==old and ctl['mutation_new_code']==(old+1)%243
            print(json.dumps(dict(site=site,complete=len(rows),seconds=time.monotonic()-START)),flush=True)
        aa={}
        for site in [0,12,23]:
            aa[str(site)]={}
            for split in ['FIT','DEV']:
                rr=[q for q in rows if q['site']==site and q['split']==split];assert len(rr)==24;aa[str(site)][split]={}
                for kind in ['raw_vs_native','BF16_vs_baseline','BF16_vs_source']:
                    aa[str(site)][split][kind]={}
                    for field in ['relative_RMS','centered_relative_RMS','label_relative_RMS']:
                        vals=[q['metrics'][kind][field] for q in rr];dm={d:float(np.mean([q['metrics'][kind][field] for q in rr if q['domain']==d])) for d in sorted({q['domain'] for q in rr})};aa[str(site)][split][kind][field]=dict(mean=float(np.mean(vals)),worst=max(vals),domain_means=dm,worst_domain_mean=max(dm.values()))
        meta(r['aggregates'],aa);donor={}
        for site in [0,12,23]:
            donor[str(site)]={}
            for split in ['FIT','DEV']:
                d=aa[str(site)][split]['BF16_vs_source'];donor[str(site)][split]=dict(mean=d['relative_RMS']['mean']<=b['criteria']['donor_mean'],worst=d['relative_RMS']['worst']<=b['criteria']['donor_worst'],centered=d['centered_relative_RMS']['mean']<=b['criteria']['donor_centered'],domain=d['relative_RMS']['worst_domain_mean']<=b['criteria']['donor_domain'])
        assert donor==r['donor_gates'];numeric=all(all(q['numeric_gates'].values()) for q in rows);extra=all(all(q['additional_gates'].values()) for q in rows);assert r['numeric_all_pass']==numeric and r['additional_all_pass']==extra;assert all(r[k]==0 for k in ['source_forwards','recurrence_calls','activation_quantizations','optimizer_updates']);check();guard();ad=dict(schema='TERNARY_READOUT_NATIVE_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,report=re,terminal=te,binding=r['binding'],counts=counts,maximum=maximum,verification_gates=verification,numeric_all_pass=numeric,additional_all_pass=extra,donor_gates=donor,before_after_hashes_exact=True,decision='INDEPENDENT_STORED_PASS' if numeric and all(verification.values()) else 'INDEPENDENT_STORED_NUMERIC_FAILURE',seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,scope='Every packed coefficient/full LUT F64 output/original F32 score bit/BF16 output and source metric independently qualified;producer scalar/table/mutation receipts checked,not rerun. No timing/DRAM/causal chatbot/model-rate admission.');write(a.out,ad);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(adjudication=extent(a.out),decision=ad['decision'],counts=counts,maximum=maximum)),flush=True)
    except BaseException as er:
        import traceback
        receipt.update(exit_code=1,error=repr(er),stage=stage,traceback=traceback.format_exc());raise
    finally:receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['report','terminal','out']:p.add_argument('--'+k,type=Path,required=True)
    for k in ['report-sha','terminal-sha','freeze']:p.add_argument('--'+k,required=True)
    main(p.parse_args())
