"""Independent full I16 quantizer/exact I64 dots/scaled scores/BF16/source audit."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
START=time.monotonic()

def main(a):
    import numpy as np,psutil
    re,te=extent(a.report),extent(a.terminal);assert re['sha256']==a.report_sha and te['sha256']==a.terminal_sha;r=json.loads(a.report.read_bytes());t=json.loads(a.terminal.read_bytes());assert t['exit_code']==0 and t['result']==re and extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil'] and not a.out.exists();proc=psutil.Process();proc.cpu_affinity(list(range(6)));stage='startup';maximum={};verification={};rows=[];counts=dict(q_weights=0,input_I16_words=0,input_scales=0,integer_dot_words=0,integer_dot_mismatches=0,score_F64_words=0,BF16_words=0,BF16_mismatches=0);receipt=dict(schema='INTEGER_READOUT_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
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
        v=x.astype('<f4').view('<u4');top=v>>16;low=v&65535;return (top+((low>32768)|((low==32768)&((top&1)!=0))).astype('u4')).astype('u2')
    try:
        check();ad=json.loads(Path(b['parent_adjudication']['path']).read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==b['parent_report'];assert b['bounds']==dict(weight=364,input=32767,block=128,block_abs_sum=364*32767*128,full_abs_sum=364*32767*3072,exact_F64_integer_limit=2**53);assert b['bounds']['block_abs_sum']<2**31 and b['bounds']['full_abs_sum']<2**53;assert len(r['records'])==144
        for packet in b['packets']:
            site=packet['site'];guard()
            with np.load(packet['file']['path'],allow_pickle=False) as z:original_q=z['q'].copy();codes=z['codes'].copy();scale=z['scale'].copy()
            reconstructed=np.zeros((2048,3072),'i4')
            for p in range(6):
                d=np.empty((2048,3075),'i2')
                for k in range(5):d[:,k::5]=((codes[p].T//3**k)%3).astype('i2')-1
                assert not d[:,3072:].any();reconstructed+=3**p*d[:,:3072]
            assert np.array_equal(reconstructed,original_q) and np.max(np.abs(reconstructed))<=364;counts['q_weights']+=reconstructed.size;qd=reconstructed.astype('f8')
            for e in [q for q in b['records'] if q['site']==site]:
                stage=f'{site}/{e["id"]}';guard()
                with np.load(e['native']['path'],allow_pickle=False) as z:Z=decode(z['reconstructed_denominator__normalized']);Y=z['reconstructed_denominator__raw_scores'].astype('f8');base=z['reconstructed_denominator__output'].copy()
                original=next(q for q in r['records'] if q['site']==site and q['id']==e['id'])
                with np.load(original['file']['path'],allow_pickle=False) as z:qr=z['input_I16'].copy();aa=z['input_scale'].copy();dots=z['integer_dots'].copy();raw=z['scores'].copy();bits=z['output_BF16'].copy();assert set(z.files)=={'input_I16','input_scale','integer_dots','scores','output_BF16'}
                assert qr.dtype==np.dtype('<i2') and qr.shape==Z.shape and aa.dtype==np.dtype('<f8') and aa.shape==(len(Z),);assert dots.dtype==np.dtype('<i8') and raw.dtype==np.dtype('<f8') and bits.dtype==np.dtype('<u2') and dots.shape==raw.shape==bits.shape==Y.shape;assert np.isfinite(raw).all() and np.isfinite(aa).all()
                expected_a=np.max(np.abs(Z),axis=1)/32767;measure('activation_scale',rel(aa,expected_a),b['numeric']['scale_and_score_relative']);ratio=np.divide(Z,expected_a[:,None],out=np.zeros_like(Z),where=expected_a[:,None]!=0);mag=np.abs(ratio);lo=np.floor(mag);part=mag-lo;inc=(part>.5)|((part==.5)&((lo%2)!=0));expected_r=np.clip((lo+inc)*np.sign(ratio),-32767,32767).astype('i2');assert np.array_equal(qr,expected_r) and np.max(np.abs(qr.astype('i4')))<=32767
                counts['input_I16_words']+=qr.size;counts['input_scales']+=aa.size;exact=np.empty_like(raw)
                for lo in range(0,len(Z),32):exact[lo:lo+32]=np.einsum('td,od->to',expected_r[lo:lo+32].astype('f8'),qd,optimize=True)
                assert np.array_equal(exact,np.rint(exact)) and np.max(np.abs(exact))<=b['bounds']['full_abs_sum'];expected_dot=exact.astype('i8');different=int(np.count_nonzero(dots!=expected_dot));counts['integer_dot_mismatches']+=different;counts['integer_dot_words']+=dots.size;verification['integer_exact']=bool(verification.get('integer_exact',True) and different==0);gold=exact*expected_a[:,None]*scale;scoreerr=rel(raw,gold);measure('scaled_score',scoreerr,b['numeric']['scale_and_score_relative']);bf_diff=int(np.count_nonzero(bits!=bf(gold)));counts['BF16_mismatches']+=bf_diff;counts['BF16_words']+=bits.size;counts['score_F64_words']+=raw.size;verification['BF16_exact']=bool(verification.get('BF16_exact',True) and bf_diff==0)
                nz=expected_a>0;bound=float(np.max(np.abs(Z[nz]-expected_r[nz]*expected_a[nz,None])/(expected_a[nz,None]*.5))) if nz.any() else 0.;qm=dict(zero_rows=int((~nz).sum()),half_step_max=bound,relative_RMS=rel(Z,expected_r*expected_a[:,None]));meta(original['quantization'],qm)
                ref=next(q['file'] for q in b['reference_records'] if q['site']==site and q['id']==e['id'])
                with np.load(ref['path'],allow_pickle=False) as z:p6=z['lut_scores'].copy();pb=z['lut_BF16'].copy()
                increment=rel(raw,p6);f=e['source_output'];source=decode(np.fromfile(f['path'],'u2').reshape(f['shape']));out=decode(bits);metrics=dict(raw_vs_native=mm(raw,Y,e['positions']),BF16_vs_baseline=mm(out,decode(base),e['positions']),BF16_vs_source=mm(out,source,e['positions']));ng=dict(integer=different==0,scaled_score=scoreerr<=b['numeric']['scale_and_score_relative'],BF16=bf_diff==0,activation_bound=bound<=1+b['numeric']['activation_half_step_slack']);extra=dict(raw=metrics['raw_vs_native']['relative_RMS']<=b['criteria']['additional_raw_each_case'],BF16=metrics['BF16_vs_baseline']['relative_RMS']<=b['criteria']['additional_BF16_each_case'],activation=increment<=b['criteria']['incremental_activation_each_case']);meta(original['metrics'],metrics);meta(original['scaled_score_relative'],scoreerr);meta(original['incremental_activation_relative'],increment);assert original['integer_mismatches']==different and original['p6_BF16_mismatches']==int(np.count_nonzero(bits!=pb)) and original['numeric_gates']==ng and original['additional_gates']==extra;rows.append(dict(site=site,split=e['split'],domain=e['domain'],metrics=metrics,numeric_gates=ng,additional_gates=extra))
            print(json.dumps(dict(site=site,complete=len(rows),seconds=time.monotonic()-START)),flush=True)
        for ctl,(sw,sx) in zip(r['controls'][:4],[(1,1),(1,-1),(-1,1),(-1,-1)]):assert ctl==dict(weight_sign=sw,input_sign=sx,exact_dot=sw*sx*b['bounds']['full_abs_sum'],all2048=True)
        assert r['controls'][4:]==[dict(single_coordinate_mutation_exact=True),dict(zero_input_exact=True)]
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
        assert donor==r['donor_gates'];numeric=all(all(q['numeric_gates'].values()) for q in rows);extra=all(all(q['additional_gates'].values()) for q in rows);assert r['numeric_all_pass']==numeric and r['additional_all_pass']==extra and all(r[k]==0 for k in ['source_forwards','recurrence_calls','optimizer_updates']);check();guard();ad=dict(schema='INTEGER_READOUT_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,report=re,terminal=te,binding=r['binding'],counts=counts,maximum=maximum,verification_gates=verification,numeric_all_pass=numeric,additional_all_pass=extra,donor_gates=donor,before_after_hashes_exact=True,decision='INDEPENDENT_STORED_PASS' if numeric and all(verification.values()) else 'INDEPENDENT_STORED_NUMERIC_FAILURE',seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,scope='Every input I16/scale/exact native I64 dot/F64 output/BF16/source gate independently verified;producer extrema/mutation receipts distinct from all-history field audit. New activation quantization isolated,no model/DRAM/rate admission.');write(a.out,ad);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(adjudication=extent(a.out),decision=ad['decision'],counts=counts,maximum=maximum)),flush=True)
    except BaseException as er:
        import traceback
        receipt.update(exit_code=1,error=repr(er),stage=stage,traceback=traceback.format_exc());raise
    finally:receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['report','terminal','out']:p.add_argument('--'+k,type=Path,required=True)
    for k in ['report-sha','terminal-sha','freeze']:p.add_argument('--'+k,required=True)
    main(p.parse_args())
