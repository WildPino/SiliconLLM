"""Source-weight-only full-size balanced ternary representation and response assay."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic()
PLANES=[4,5,6]


def decode(q,np):
    return (q.astype('<u4')<<16).view('<f4').astype('f8')


def bf(q,np):
    u=q.astype('<f4').view('<u4')
    return ((u+0x7fff+((u>>16)&1))>>16).astype('<u2')


def metrics(x,y,pos,np):
    def relative(a,b):
        return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
    return dict(relative_RMS=relative(x,y),centered_relative_RMS=relative(x-x.mean(0),y-y.mean(0)),
                label_relative_RMS=relative(x[pos],y[pos]))


def aggregate(rows,np):
    aa={}
    for site in [0,12,23]:
        aa[str(site)]={}
        for split in ['FIT','DEV']:
            rr=[q for q in rows if q['site']==site and q['split']==split]
            assert len(rr)==24
            aa[str(site)][split]={}
            for name in ['dense',*[str(p) for p in PLANES]]:
                aa[str(site)][split][name]={}
                for kind in ['raw_vs_native','BF16_vs_baseline','BF16_vs_source']:
                    out={}
                    for field in ['relative_RMS','centered_relative_RMS','label_relative_RMS']:
                        vv=[q['metrics'][name][kind][field] for q in rr]
                        dm={d:float(np.mean([q['metrics'][name][kind][field] for q in rr if q['domain']==d]))
                            for d in sorted({q['domain'] for q in rr})}
                        out[field]=dict(mean=float(np.mean(vv)),worst=max(vv),domain_means=dm,worst_domain_mean=max(dm.values()))
                    aa[str(site)][split][name][kind]=out
    return aa


def quantize(O,p,np):
    limit=(3**p-1)//2
    scale=np.max(np.abs(O),axis=1)/limit
    ratio=np.divide(O,scale[:,None],out=np.zeros_like(O),where=scale[:,None]!=0)
    nearest=np.rint(ratio)
    q=np.clip(nearest,-limit,limit).astype('<i2')
    remaining=q.astype('i4')
    codes=np.empty((p,615,2048),'u1')
    hist=[]
    for plane in range(p):
        digit=remaining%3
        digit[digit==2]=-1
        remaining=(remaining-digit)//3
        hist.append({str(k):int(np.count_nonzero(digit==k)) for k in [-1,0,1]})
        pad=np.pad(digit,((0,0),(0,3)))
        index=np.sum((pad.reshape(2048,615,5)+1)*np.array([1,3,9,27,81]),axis=2)
        codes[plane]=index.T.astype('u1')
    assert not np.any(remaining)
    err=np.abs(O-q*scale[:,None])
    nonzero=scale>0
    bound=float(np.max(err[nonzero]/(scale[nonzero,None]*.5))) if nonzero.any() else 0.
    return q,scale,codes,hist,dict(coefficient_relative_RMS=float(np.linalg.norm(err)/np.linalg.norm(O)),
                                  max_error_over_half_step=bound,zero_rows=int((~nonzero).sum()),
                                  clipped_coefficients=int(np.count_nonzero(nearest!=q)),
                                  exact_ratio_half_ties=int(np.count_nonzero(np.abs(ratio)-np.floor(np.abs(ratio))==.5)))


def prepare(a):
    import numpy as np,psutil
    rp=DOC/'conditional_output_function_result_20261013.json'
    ap=DOC/'conditional_output_function_adjudication_20261013.json'
    r=json.loads(rp.read_bytes())
    ad=json.loads(ap.read_bytes())
    assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(rp)
    assert extent(r['binding']['path'])==r['binding']
    pb=json.loads(Path(r['binding']['path']).read_bytes())
    stp=DOC/'ternary_readout_precision_storage_20261013.json'
    st=json.loads(stp.read_text(encoding='utf-8-sig'))
    assert not a.out.exists() and not a.directory.exists() and a.directory.resolve()==Path(st['new_output_directory']).resolve()
    paths=[Path(__file__),B/'ternary_readout_precision_audit.py',B/'falcon_auxiliary_s6_transport.py',
           DOC/'TERNARY_READOUT_PRECISION_PROTOCOL_20261013.md',stp,rp,ap,Path(r['binding']['path']),
           Path(pb['source_weights']),Path(pb['source_config']),Path(sys.executable),
           Path(sys.executable).parent/'python312.dll',Path(np.__file__),Path(np._core._multiarray_umath.__file__),
           Path(np.linalg._umath_linalg.__file__),Path(psutil.__file__)]
    nr=Path(np.__file__).resolve().parent
    pr=Path(psutil.__file__).resolve().parent
    paths+=list((nr.parent/'numpy.libs').glob('*.dll'))+list(pr.glob('*.pyd'))
    paths+=[Path(q['parameters']['path']) for q in pb['exports']]
    paths+=[Path(q[k]['path']) for q in pb['records'] for k in ['native','source_output']]
    paths+=[ROOT/p for p in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json',
           'benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']]
    paths+=[DOC/f'meth{i}_{stem}_result.json' for i,stem in [(398,'additive_integer_lut'),
           (399,'integer_lut_cost_profile'),(400,'blocked_integer_lut')]]
    b=dict(schema='TERNARY_READOUT_PRECISION_BINDING_V1',freeze=a.freeze,python=str(Path(sys.executable).resolve()),
           versions=dict(numpy=np.__version__,psutil=psutil.__version__),parent_report=extent(rp),parent_adjudication=extent(ap),
           parent_binding=r['binding'],source_weights=pb['source_weights'],source_config=pb['source_config'],
           exports=pb['exports'],records=pb['records'],planes=PLANES,sites=[0,12,23],
           numeric=dict(coefficient_half_step_slack=1e-12,quantizer_scale_relative=1e-12,
                        singular_sigma_max_relative=1e-8,robust_sigma_relative=1e-10,
                        metadata_absolute_or_relative=1e-10,dense_vs_native_relative=1e-6,
                        independent_BF16_smooth_relative=1e-4),
           criteria=dict(additional_raw_each_case=.01,additional_BF16_each_case=.01,
                         donor_mean=.1,donor_worst=.2,donor_centered=.1,donor_domain=.2),
           limits=dict(seconds=900,audit_seconds=1200,OS_bytes=6<<30,output_bytes=1<<30),
           output_directory=str(a.directory.resolve()),inputs=[extent(p) for p in dict.fromkeys(paths)],
           candidate_rule='Smallest p passing BOTH extra raw/BF16 each-case1% and all donor local FIT gates '
                          'at ALL3 sites;DEV cannot choose upgrade. All3 diagnostics retained.',
           scope='Source-weight-only balanced ternary p4/5/6/full-size q and five-trit uint8 pattern packing. '
                 'Actual own-norm BF16 inputs,integer-weight promotedF64 dot then F64 row scale/F32/BF16. '
                 'No activation quantization/new native LUT/source/GPU/RESERVED/teacher/optimizer/causal '
                 'chatbot/DRAM/model-rate/conditional-n admission. Full stored BF16 words;raw F64 arrays '
                 'not retained,their metrics independently recomputed.')
    assert time.monotonic()-START<900 and psutil.Process().memory_info().peak_wset<6<<30
    write(a.out,b)
    print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)


def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha
    b=json.loads(a.binding.read_bytes())
    assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    assert np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil']
    assert not a.directory.exists() and not a.out.exists() and a.directory.resolve()==Path(b['output_directory']).resolve()
    proc=psutil.Process()
    proc.cpu_affinity(list(range(6)))
    stage='startup'
    stream=None
    models=[]
    rows=[]
    receipt=dict(schema='TERNARY_READOUT_PRECISION_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,
                 creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes']
        assert not proc.children(recursive=True)
        if a.directory.exists():
            assert sum(p.stat().st_size for p in a.directory.iterdir())<=b['limits']['output_bytes']
    def check():
        for e in b['inputs']:
            guard()
            assert extent(e['path'])==e,e['path']
    try:
        check()
        a.directory.mkdir()
        get,stream=tensor_reader(b['source_weights'],np)
        # ALL source-weight-only artifacts built before consuming any response.
        for site in b['sites']:
            O=get(f'model.layers.{site}.mamba.out_proj.weight')
            with np.load(next(e['parameters']['path'] for e in b['exports'] if e['site']==site),allow_pickle=False) as z:
                assert np.array_equal(O.astype('f4'),z['O'])
            assert O.shape==(2048,3072)
            stage=f'{site}/source_spectrum'
            guard()
            original=np.linalg.svd(O,compute_uv=False)
            model=dict(site=site,arms=[])
            for p in PLANES:
                stage=f'{site}/p{p}/source_representation'
                guard()
                q,scale,codes,hist,cm=quantize(O,p,np)
                quantized=np.linalg.svd(q*scale[:,None],compute_uv=False)
                rank=int(np.count_nonzero(quantized>b['numeric']['robust_sigma_relative']*quantized[0]))
                cm.update(robust_rank=rank,sigma_min=float(quantized[-1]),sigma_max=float(quantized[0]),
                          original_sigma_min=float(original[-1]),original_sigma_max=float(original[0]))
                path=a.directory/f'site{site:02d}.p{p}.parameters.npz'
                np.savez(path,q=q,scale=scale,codes=codes,singular_values=quantized,original_singular_values=original)
                flags=dict(coefficient_bound=cm['max_error_over_half_step']<=1+b['numeric']['coefficient_half_step_slack'],
                           full_robust_rank=rank==2048)
                model['arms'].append(dict(planes=p,file=extent(path),digit_histograms=hist,coefficient_metrics=cm,
                                          numeric_gates=flags,packed_bytes=codes.nbytes,scale_bytes=scale.nbytes))
                print(json.dumps(dict(stage=stage,coefficient_metrics=cm,seconds=time.monotonic()-START)),flush=True)
                del q,scale,codes,quantized
            models.append(model)
            del O,original
        for model in models:
            site=model['site']
            O=get(f'model.layers.{site}.mamba.out_proj.weight')
            arms={}
            for e in model['arms']:
                with np.load(e['file']['path'],allow_pickle=False) as z:
                    arms[str(e['planes'])]=(z['q'].astype('f8'),z['scale'].copy())
            for rec in [r for r in b['records'] if r['site']==site]:
                stage=f'{site}/{rec["id"]}'
                guard()
                with np.load(rec['native']['path'],allow_pickle=False) as z:
                    Z=decode(z['reconstructed_denominator__normalized'],np)
                    Y=z['reconstructed_denominator__raw_scores'].astype('f8')
                    basebits=z['reconstructed_denominator__output'].copy()
                f=rec['source_output']
                source=decode(np.fromfile(f['path'],'<u2').reshape(f['shape']),np)
                assert Z.shape==(rec['history'],3072) and Y.shape==source.shape==(rec['history'],2048)
                assert np.isfinite(Z).all() and np.isfinite(Y).all()
                base=decode(basebits,np)
                dense=Z@O.T
                out=decode(bf(dense,np),np)
                mm=dict(dense=dict(raw_vs_native=metrics(dense,Y,rec['positions'],np),
                                   BF16_vs_baseline=metrics(out,base,rec['positions'],np),
                                   BF16_vs_source=metrics(out,source,rec['positions'],np)))
                arrays={}
                flags={}
                mismatch={}
                bounds={}
                for key,(q,scale) in arms.items():
                    pred=(Z@q.T)*scale
                    bits=bf(pred,np)
                    out=decode(bits,np)
                    arrays['p'+key]=bits
                    mm[key]=dict(raw_vs_native=metrics(pred,Y,rec['positions'],np),
                                 BF16_vs_baseline=metrics(out,base,rec['positions'],np),
                                 BF16_vs_source=metrics(out,source,rec['positions'],np))
                    flags[key]=dict(additional_raw=mm[key]['raw_vs_native']['relative_RMS']<=b['criteria']['additional_raw_each_case'],
                                    additional_BF16=mm[key]['BF16_vs_baseline']['relative_RMS']<=b['criteria']['additional_BF16_each_case'])
                    mismatch[key]=int(np.count_nonzero(bits!=basebits))
                    bd=.5*np.linalg.norm(scale)*np.sum(np.abs(Z),axis=1)
                    co=next(e['coefficient_metrics']['coefficient_relative_RMS'] for e in model['arms'] if str(e['planes'])==key)
                    bounds[key]=dict(conservative_error_bound_relative=float(np.linalg.norm(bd)/np.linalg.norm(dense)),
                                     observed_quant_vs_real_relative=float(np.linalg.norm(pred-dense)/np.linalg.norm(dense)),
                                     coefficient_to_response_amplification=float(mm[key]['raw_vs_native']['relative_RMS']/co) if co>0 else None)
                path=a.directory/f'{rec["id"]}.site{site:02d}.npz'
                np.savez(path,**arrays)
                rows.append(dict(id=rec['id'],site=site,split=rec['split'],domain=rec['domain'],history=rec['history'],
                                 file=extent(path),metrics=mm,additional_flags=flags,BF16_mismatches=mismatch,
                                 bounds=bounds,dense_numeric_gate=mm['dense']['raw_vs_native']['relative_RMS']<=b['numeric']['dense_vs_native_relative']))
                if len(rows)%12==0:
                    print(json.dumps(dict(complete=len(rows),seconds=time.monotonic()-START)),flush=True)
            del arms,O
        stream.close()
        stream=None
        check()
        for m in models:
            for e in m['arms']:
                assert extent(e['file']['path'])==e['file']
        guard()
        aa=aggregate(rows,np)
        donor={}
        for site in b['sites']:
            donor[str(site)]={}
            for p in PLANES:
                key=str(p)
                donor[str(site)][key]={}
                for split in ['FIT','DEV']:
                    d=aa[str(site)][split][key]['BF16_vs_source']
                    donor[str(site)][key][split]=dict(mean=d['relative_RMS']['mean']<=b['criteria']['donor_mean'],
                        worst=d['relative_RMS']['worst']<=b['criteria']['donor_worst'],
                        centered=d['centered_relative_RMS']['mean']<=b['criteria']['donor_centered'],
                        domain=d['relative_RMS']['worst_domain_mean']<=b['criteria']['donor_domain'])
        admitted=[p for p in PLANES if all(all(q['additional_flags'][str(p)].values()) for q in rows if q['split']=='FIT')
                  and all(all(donor[str(site)][str(p)]['FIT'].values()) for site in b['sites'])]
        selected=min(admitted) if admitted else None
        heldout=None if selected is None else (all(all(q['additional_flags'][str(selected)].values()) for q in rows if q['split']=='DEV')
                 and all(all(donor[str(site)][str(selected)]['DEV'].values()) for site in b['sites']))
        numeric=all(q['dense_numeric_gate'] for q in rows) and all(all(e['numeric_gates'].values()) for m in models for e in m['arms'])
        result=dict(schema='TERNARY_READOUT_PRECISION_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,models=models,
                    records=rows,aggregates=aa,donor_gates=donor,FIT_selected_planes=selected,selected_DEV_all_pass=heldout,
                    numeric_all_pass=numeric,before_after_hashes_exact=True,source_only_artifacts_before_responses=True,
                    source_forwards=0,native_calls=0,activation_quantizations=0,optimizer_updates=0,new_ternary_expert_banks=0,
                    decision='COMPLETE_PENDING_FULL_INDEPENDENT_AUDIT',output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()),
                    seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,scope=b['scope'])
        write(a.out,result)
        receipt.update(exit_code=0,error=None,result=extent(a.out))
        print(json.dumps(dict(result=extent(a.out),FIT_selected_planes=selected,DEV_all_pass=heldout,numeric_all_pass=numeric)),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),stage=stage,traceback=traceback.format_exc())
        raise
    finally:
        if stream is not None:
            stream.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--prepare',action='store_true')
    m.add_argument('--run',action='store_true')
    for k in ['binding','directory','out']:
        p.add_argument('--'+k,type=Path)
    for k in ['binding-sha','freeze']:
        p.add_argument('--'+k)
    a=p.parse_args()
    if a.run:
        run(a)
    else:
        import psutil
        proc=psutil.Process()
        receipt=dict(schema='TERNARY_READOUT_PRECISION_PREPARATION_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,
                     creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
        try:
            prepare(a)
            receipt.update(exit_code=0,error=None,result=extent(a.out))
        except BaseException as e:
            import traceback
            receipt.update(exit_code=1,error=repr(e),traceback=traceback.format_exc())
            raise
        finally:
            receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset)
            write(a.out.with_suffix('.preparation_terminal.json'),receipt)
