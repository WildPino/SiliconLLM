"""Independent quantizer/packed digits/Gram spectra and every BF16 response/metric."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic()


def main(a):
    import numpy as np,psutil
    re,te=extent(a.report),extent(a.terminal)
    assert re['sha256']==a.report_sha and te['sha256']==a.terminal_sha
    r=json.loads(a.report.read_bytes())
    t=json.loads(a.terminal.read_bytes())
    assert t['exit_code']==0 and t['result']==re and extent(r['binding']['path'])==r['binding']
    b=json.loads(Path(r['binding']['path']).read_bytes())
    assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    assert np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil']
    assert b['planes']==[4,5,6] and b['sites']==[0,12,23] and not a.out.exists()
    proc=psutil.Process()
    proc.cpu_affinity(list(range(6)))
    stage='startup'
    stream=None
    maximum={}
    verification={}
    counts=dict(q_coefficients=0,packed_codes=0,scale_words=0,singular_values=0,
                stored_BF16_words=0,independent_BF16_mismatches=0,source_Z_Y_words=0)
    rows=[]
    models=[]
    num=b['numeric']
    receipt=dict(schema='TERNARY_READOUT_PRECISION_AUDIT_TERMINAL_V1',command=sys.orig_argv,
                 pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['audit_seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes']
        assert not proc.children(recursive=True)
    def check():
        files=[r['binding'],re,te]+[q['file'] for q in r['records']]
        files+=[q['file'] for m in r['models'] for q in m['arms']]
        for e in b['inputs']+files:
            guard()
            assert extent(e['path'])==e,e['path']
    def metric(key,value,tol):
        value=float(value)
        assert np.isfinite(value)
        maximum[key]=max(maximum.get(key,0.),value)
        verification[key]=bool(verification.get(key,True) and value<=tol)
    def meta(x,y):
        if isinstance(y,dict):
            assert x.keys()==y.keys()
            for k in y:
                meta(x[k],y[k])
        elif isinstance(y,list):
            assert len(x)==len(y)
            for v,z in zip(x,y):
                meta(v,z)
        elif isinstance(y,(bool,int,str)) or y is None:
            assert x==y,(stage,x,y)
        else:
            metric('metadata_absolute_or_relative',abs(x-y)/max(abs(y),1.),num['metadata_absolute_or_relative'])
    def decode(bits):
        return (bits.astype('<u4')*65536).view('<f4').astype('f8')
    def rounded(X):
        bits=X.astype('<f4').view('<u4')
        upper,lower=bits>>16,bits&65535
        increment=(lower>32768)|((lower==32768)&((upper&1)!=0))
        return (upper+increment.astype('u4')).astype('<u2')
    def rel(x,y):
        return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    def measurements(x,y,pos):
        return dict(relative_RMS=rel(x,y),centered_relative_RMS=rel(x-x.mean(0),y-y.mean(0)),label_relative_RMS=rel(x[pos],y[pos]))
    def spectrum(M,stored,key):
        G=np.zeros((2048,2048))
        for lo in range(0,3072,128):
            block=M[:,lo:lo+128]
            G+=block@block.T
        ev=np.linalg.eigvalsh((G+G.T)*.5)[::-1]
        assert stored.shape==(2048,) and np.all(np.diff(stored)<=0) and (stored>=0).all()
        metric(key,float(np.max(np.abs(stored**2-ev))/stored[0]**2),num['singular_sigma_max_relative'])
        metric('Gram_negative_relative',max(0.,float(-ev[-1]))/float(ev[0]),1e-10)
        counts['singular_values']+=2048
        return dict(raw_Gram_minimum=float(ev[-1]),raw_Gram_negative_modes=int((ev<0).sum()))
    try:
        check()
        ad=json.loads(Path(b['parent_adjudication']['path']).read_bytes())
        assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==b['parent_report'] and ad['binding']==b['parent_binding']
        parent=json.loads(Path(b['parent_binding']['path']).read_bytes())
        assert b['records']==parent['records'] and b['exports']==parent['exports']
        get,stream=tensor_reader(b['source_weights'],np)
        for model in r['models']:
            site=model['site']
            stage=f'{site}/full_source_quantizer_packing'
            guard()
            O=get(f'model.layers.{site}.mamba.out_proj.weight')
            with np.load(next(e['parameters']['path'] for e in b['exports'] if e['site']==site),allow_pickle=False) as z:
                assert np.array_equal(O.astype('f4'),z['O'])
            params={}
            mr=[]
            original_reference=None
            original_gram=None
            for entry in model['arms']:
                p=entry['planes']
                assert p in b['planes']
                with np.load(entry['file']['path'],allow_pickle=False) as z:
                    arrays={k:z[k].copy() for k in z.files}
                assert set(arrays)=={'q','scale','codes','singular_values','original_singular_values'}
                q,scale,codes=arrays['q'],arrays['scale'],arrays['codes']
                assert q.dtype==np.dtype('<i2') and q.shape==(2048,3072)
                assert scale.dtype==np.dtype('<f8') and scale.shape==(2048,)
                assert codes.dtype==np.dtype('u1') and codes.shape==(p,615,2048) and (codes<=242).all()
                limit=(3**p-1)//2
                expected_scale=np.max(np.abs(O),axis=1)/limit
                metric('scale_relative',rel(scale,expected_scale),num['quantizer_scale_relative'])
                ratio=np.divide(O,expected_scale[:,None],out=np.zeros_like(O),where=expected_scale[:,None]!=0)
                absolute=np.abs(ratio)
                floor=np.floor(absolute)
                fraction=absolute-floor
                inc=(fraction>.5)|((fraction==.5)&((floor%2)!=0))
                nearest=(floor+inc)*np.sign(ratio)
                expected_q=np.clip(nearest,-limit,limit).astype('i2')
                assert np.array_equal(q,expected_q)
                rebuilt=np.zeros(q.shape,'i4')
                hist=[]
                for plane in range(p):
                    digits=np.empty((2048,3075),'i2')
                    for k in range(5):
                        digits[:,k::5]=((codes[plane].T//(3**k))%3).astype('i2')-1
                    assert not digits[:,3072:].any()
                    actual=digits[:,:3072]
                    rebuilt+=3**plane*actual
                    hist.append({str(v):int(np.count_nonzero(actual==v)) for v in [-1,0,1]})
                assert np.array_equal(rebuilt,q)
                meta(entry['digit_histograms'],hist)
                counts['q_coefficients']+=q.size
                counts['packed_codes']+=codes.size
                counts['scale_words']+=scale.size
                assert entry['packed_bytes']==codes.nbytes and entry['scale_bytes']==scale.nbytes
                err=O-q*scale[:,None]
                nonzero=scale>0
                bound=float(np.max(np.abs(err[nonzero])/(scale[nonzero,None]*.5))) if nonzero.any() else 0.
                sv=arrays['singular_values']
                original=arrays['original_singular_values']
                if original_reference is None:
                    original_gram=spectrum(O,original,'original_Gram_vs_squared_SVD')
                    original_reference=original.copy()
                else:
                    assert np.array_equal(original,original_reference)
                    counts['singular_values']+=2048
                gm=original_gram
                qm=spectrum(q*scale[:,None],sv,'quantized_Gram_vs_squared_SVD')
                rank=int(np.count_nonzero(sv>num['robust_sigma_relative']*sv[0]))
                cm=dict(coefficient_relative_RMS=float(np.linalg.norm(err)/np.linalg.norm(O)),
                        max_error_over_half_step=bound,zero_rows=int((~nonzero).sum()),clipped_coefficients=int(np.count_nonzero(nearest!=q)),
                        exact_ratio_half_ties=int(np.count_nonzero(fraction==.5)),robust_rank=rank,
                        sigma_min=float(sv[-1]),sigma_max=float(sv[0]),original_sigma_min=float(original[-1]),original_sigma_max=float(original[0]))
                meta(entry['coefficient_metrics'],cm)
                ng=dict(coefficient_bound=bound<=1+num['coefficient_half_step_slack'],full_robust_rank=rank==2048)
                assert entry['numeric_gates']==ng
                mr.append(dict(planes=p,numeric_gates=ng,original_Gram=gm,quantized_Gram=qm))
                params[str(p)]=(rebuilt.astype('f8'),scale)
                print(json.dumps(dict(stage=stage,planes=p,packed_coefficients_exact=True,seconds=time.monotonic()-START)),flush=True)
            models.append(dict(site=site,arms=mr))
            for e in [q for q in b['records'] if q['site']==site]:
                stage=f'{site}/{e["id"]}/all_outputs'
                guard()
                with np.load(e['native']['path'],allow_pickle=False) as z:
                    Z=decode(z['reconstructed_denominator__normalized'])
                    Y=z['reconstructed_denominator__raw_scores'].astype('f8')
                    basebits=z['reconstructed_denominator__output'].copy()
                f=e['source_output']
                source=decode(np.fromfile(f['path'],'<u2').reshape(f['shape']))
                counts['source_Z_Y_words']+=Z.size+Y.size
                base=decode(basebits)
                original=next(q for q in r['records'] if q['site']==site and q['id']==e['id'])
                with np.load(original['file']['path'],allow_pickle=False) as z:
                    outputs={k:z[k].copy() for k in z.files}
                assert set(outputs)=={'p4','p5','p6'}
                dense=np.empty_like(Y)
                predictions={key:np.empty_like(Y) for key in params}
                for lo in range(0,len(Z),32):
                    block=Z[lo:lo+32]
                    dense[lo:lo+32]=np.einsum('td,od->to',block,O,optimize=True)
                    for key,(q,scale) in params.items():
                        predictions[key][lo:lo+32]=np.einsum('td,od->to',block,q,optimize=True)*scale
                out=decode(rounded(dense))
                mm=dict(dense=dict(raw_vs_native=measurements(dense,Y,e['positions']),
                                   BF16_vs_baseline=measurements(out,base,e['positions']),BF16_vs_source=measurements(out,source,e['positions'])))
                flags={}
                mismatch={}
                bounds={}
                for key,pred in predictions.items():
                    bits=outputs['p'+key]
                    assert bits.dtype==np.dtype('<u2') and bits.shape==Y.shape
                    counts['stored_BF16_words']+=bits.size
                    independent=rounded(pred)
                    different=int(np.count_nonzero(bits!=independent))
                    counts['independent_BF16_mismatches']+=different
                    out=decode(bits)
                    metric('independent_BF16_smooth_relative',rel(out,decode(independent)),num['independent_BF16_smooth_relative'])
                    mm[key]=dict(raw_vs_native=measurements(pred,Y,e['positions']),
                                 BF16_vs_baseline=measurements(out,base,e['positions']),BF16_vs_source=measurements(out,source,e['positions']))
                    flags[key]=dict(additional_raw=mm[key]['raw_vs_native']['relative_RMS']<=b['criteria']['additional_raw_each_case'],
                                    additional_BF16=mm[key]['BF16_vs_baseline']['relative_RMS']<=b['criteria']['additional_BF16_each_case'])
                    mismatch[key]=int(np.count_nonzero(bits!=basebits))
                    q,scale=params[key]
                    bd=.5*np.linalg.norm(scale)*np.sum(np.abs(Z),axis=1)
                    co=next(q['coefficient_metrics']['coefficient_relative_RMS'] for q in model['arms'] if str(q['planes'])==key)
                    bounds[key]=dict(conservative_error_bound_relative=float(np.linalg.norm(bd)/np.linalg.norm(dense)),
                                     observed_quant_vs_real_relative=float(np.linalg.norm(pred-dense)/np.linalg.norm(dense)),
                                     coefficient_to_response_amplification=float(mm[key]['raw_vs_native']['relative_RMS']/co) if co>0 else None)
                meta(original['metrics'],mm)
                meta(original['bounds'],bounds)
                assert original['additional_flags']==flags and original['BF16_mismatches']==mismatch
                dense_gate=mm['dense']['raw_vs_native']['relative_RMS']<=num['dense_vs_native_relative']
                assert original['dense_numeric_gate']==dense_gate
                rows.append(dict(id=e['id'],site=site,split=e['split'],domain=e['domain'],metrics=mm,additional_flags=flags,dense_numeric_gate=dense_gate))
                if len(rows)%24==0:
                    print(json.dumps(dict(complete=len(rows),seconds=time.monotonic()-START)),flush=True)
            del params,O
        stream.close()
        stream=None
        aa={}
        for site in b['sites']:
            aa[str(site)]={}
            for split in ['FIT','DEV']:
                rr=[q for q in rows if q['site']==site and q['split']==split]
                assert len(rr)==24
                aa[str(site)][split]={}
                for key in ['dense','4','5','6']:
                    aa[str(site)][split][key]={}
                    for kind in ['raw_vs_native','BF16_vs_baseline','BF16_vs_source']:
                        sub={}
                        for field in ['relative_RMS','centered_relative_RMS','label_relative_RMS']:
                            vals=[q['metrics'][key][kind][field] for q in rr]
                            dm={d:float(np.mean([q['metrics'][key][kind][field] for q in rr if q['domain']==d]))
                                for d in sorted({q['domain'] for q in rr})}
                            sub[field]=dict(mean=float(np.mean(vals)),worst=max(vals),domain_means=dm,worst_domain_mean=max(dm.values()))
                        aa[str(site)][split][key][kind]=sub
        meta(r['aggregates'],aa)
        donor={}
        for site in b['sites']:
            donor[str(site)]={}
            for p in b['planes']:
                key=str(p)
                donor[str(site)][key]={}
                for split in ['FIT','DEV']:
                    d=aa[str(site)][split][key]['BF16_vs_source']
                    donor[str(site)][key][split]=dict(mean=d['relative_RMS']['mean']<=b['criteria']['donor_mean'],
                        worst=d['relative_RMS']['worst']<=b['criteria']['donor_worst'],centered=d['centered_relative_RMS']['mean']<=b['criteria']['donor_centered'],
                        domain=d['relative_RMS']['worst_domain_mean']<=b['criteria']['donor_domain'])
        assert r['donor_gates']==donor
        fits=[p for p in b['planes'] if all(all(q['additional_flags'][str(p)].values()) for q in rows if q['split']=='FIT')
              and all(all(donor[str(site)][str(p)]['FIT'].values()) for site in b['sites'])]
        selected=min(fits) if fits else None
        heldout=None if selected is None else (all(all(q['additional_flags'][str(selected)].values()) for q in rows if q['split']=='DEV')
            and all(all(donor[str(site)][str(selected)]['DEV'].values()) for site in b['sites']))
        assert r['FIT_selected_planes']==selected and r['selected_DEV_all_pass']==heldout
        numeric=all(q['dense_numeric_gate'] for q in rows) and all(all(q['numeric_gates'].values()) for m in models for q in m['arms'])
        assert r['numeric_all_pass']==numeric and r['source_only_artifacts_before_responses']
        assert all(r[k]==0 for k in ['source_forwards','native_calls','activation_quantizations','optimizer_updates','new_ternary_expert_banks'])
        check()
        guard()
        result=dict(schema='TERNARY_READOUT_PRECISION_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,
                    report=re,terminal=te,binding=r['binding'],models=models,counts=counts,maximum=maximum,verification_gates=verification,
                    FIT_selected_planes=selected,selected_DEV_all_pass=heldout,numeric_all_pass=numeric,before_after_hashes_exact=True,
                    decision='INDEPENDENT_STORED_PASS' if numeric and all(verification.values()) else 'INDEPENDENT_STORED_NUMERIC_FAILURE',
                    seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
                    scope='All source integer quantizer coordinates/digits/packed bytes/scales/full Gram spectra;all stored BF16 words '
                          'and every unsaved raw-score metric independently recomputed from actual normalized inputs. '
                          'No actual LUT kernel/activation quantization/model/DRAM/quality-speed admission.')
        write(a.out,result)
        receipt.update(exit_code=0,error=None,result=extent(a.out))
        print(json.dumps(dict(adjudication=extent(a.out),decision=result['decision'],counts=counts,maximum=maximum)),flush=True)
    except BaseException as ex:
        import traceback
        receipt.update(exit_code=1,error=repr(ex),stage=stage,traceback=traceback.format_exc())
        raise
    finally:
        if stream is not None:
            stream.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['report','terminal','out']:
        p.add_argument('--'+k,type=Path,required=True)
    for k in ['report-sha','terminal-sha','freeze']:
        p.add_argument('--'+k,required=True)
    main(p.parse_args())
