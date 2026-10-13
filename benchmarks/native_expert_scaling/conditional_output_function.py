"""Input-routed redundant affine source output functions, FIT objects frozen before DEV."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic()
ARMS=[('tree16_r128',16,128),('tree64_r32',64,32)]


def decode(q,np):
    return (q.astype('<u4')<<16).view('<f4').astype('<f8')


def rounded(q,np):
    u=q.astype('<f4').view('<u4')
    return ((u+0x7fff+((u>>16)&1))>>16).astype('<u2')


def relative(x,y,np):
    return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))


def metrics(x,y,positions,np):
    return dict(relative_RMS=relative(x,y,np),
                centered_relative_RMS=relative(x-x.mean(0),y-y.mean(0),np),
                label_relative_RMS=relative(x[positions],y[positions],np))


def sketch(Z,indices,signs,np):
    Q=np.zeros((len(Z),16))
    for j in range(16):
        for k in range(64):
            Q[:,j]+=signs[j,k]*Z[:,indices[j,k]]
    return Q*.125


def route(Q,features,cuts,K,np):
    nodes=np.zeros(len(Q),'i4')
    for _ in range(K.bit_length()-1):
        go=Q[np.arange(len(Q)),features[nodes]]>cuts[nodes]
        nodes=nodes*2+1+go.astype('i4')
    return (nodes-(K-1)).astype('<u4')


def build_tree(Q,weights,K,rank,np):
    features=np.full(K-1,-1,'i4')
    cuts=np.zeros(K-1)
    stack=[(0,np.arange(len(Q)),K.bit_length()-1)]
    while stack:
        node,rows,depth=stack.pop()
        if depth==0:
            assert len(rows)>=rank+1
            continue
        w=weights[rows]
        mass=w.sum()
        vals=Q[rows]
        mean=np.sum(vals*w[:,None],axis=0)/mass
        variance=np.sum((vals-mean)**2*w[:,None],axis=0)/mass
        required=(rank+1)*(1<<(depth-1))
        choice=None
        for f in sorted(range(16),key=lambda j:(-float(variance[j]),j)):
            if variance[f]<=0:
                continue
            order=np.argsort(vals[:,f],kind='stable')
            values=vals[order,f]
            end=np.flatnonzero(values[:-1]!=values[1:])
            end=end[(end+1>=required)&(len(rows)-end-1>=required)]
            if len(end)==0:
                continue
            cumulative=np.cumsum(w[order])
            cut_at=int(end[np.argmin(np.abs(cumulative[end]-mass*.5))])
            choice=(f,float(values[cut_at]))
            break
        if choice is None:
            raise ValueError(f'UNSUPPORTED_PARTITION node={node} depth={depth} rows={len(rows)}')
        f,cut=choice
        features[node]=f
        cuts[node]=cut
        left=rows[Q[rows,f]<=cut]
        right=rows[Q[rows,f]>cut]
        assert len(left)>=required and len(right)>=required
        stack.extend([(2*node+2,right,depth-1),(2*node+1,left,depth-1)])
    IDs=route(Q,features,cuts,K,np)
    return features,cuts,IDs


def fit_bank(Z,Y,w,Q,O,K,rank,np,guard):
    features,cuts,IDs=build_tree(Q,w,K,rank,np)
    U=np.empty((K,2048,rank))
    BB=np.empty((K,rank,3072))
    bias=np.empty((K,2048))
    mz=np.empty((K,3072))
    my=np.empty((K,2048))
    spectra=[]
    offsets=[0]
    leaves=[]
    for j in range(K):
        guard()
        keep=IDs==j
        ww=w[keep]
        mass=ww.sum()
        zz=Z[keep]
        yy=Y[keep]
        mz[j]=np.sum(zz*ww[:,None],axis=0)/mass
        my[j]=np.sum(yy*ww[:,None],axis=0)/mass
        M=(yy-my[j])*np.sqrt(ww/mass)[:,None]
        _,sing,Vh=np.linalg.svd(M,full_matrices=False)
        u=Vh[:rank].T.copy()
        for col in range(rank):
            if u[np.argmax(np.abs(u[:,col])),col]<0:
                u[:,col]*=-1
        U[j]=u
        BB[j]=u.T@O
        bias[j]=my[j]-u@(BB[j]@mz[j])
        energy=float(np.sum(M*M))
        eigen=sing[:rank]**2
        orth=float(np.max(np.abs(u.T@u-np.eye(rank))))
        residual=float(np.linalg.norm(M.T@(M@u)-u*eigen)/max(energy,1e-30))
        tail=float(np.sum(sing[rank:]**2))
        leaves.append(dict(id=j,positions=int(keep.sum()),mass=float(mass),
                           centered_energy=energy,discarded_energy=tail,
                           orthogonal_max=orth,eigen_residual_relative=residual,
                           zero_singular_values=int((sing==0).sum())))
        spectra.append(sing)
        offsets.append(offsets[-1]+len(sing))
    arrays=dict(features=features,cuts=cuts,fit_IDs=IDs,U=U,B=BB,bias=bias,
                mean_Z=mz,mean_Y=my,singular_values=np.concatenate(spectra),
                singular_offsets=np.array(offsets,'i4'))
    return arrays,leaves


def evaluate(bank,Z,Y,Q,K,np):
    IDs=route(Q,bank['features'],bank['cuts'],K,np)
    out=np.empty_like(Y)
    projection=np.empty_like(Y)
    means=np.empty_like(Y)
    for j in range(K):
        keep=IDs==j
        if not keep.any():
            continue
        u=bank['U'][j]
        out[keep]=(Z[keep]@bank['B'][j].T)@u.T+bank['bias'][j]
        projection[keep]=(Y[keep]-bank['mean_Y'][j])@u@u.T+bank['mean_Y'][j]
        means[keep]=bank['mean_Y'][j]
    return IDs,out,projection,means


def aggregate(rows,np):
    result={}
    for site in [0,12,23]:
        result[str(site)]={}
        for split in ['FIT','DEV']:
            rr=[r for r in rows if r['site']==site and r['split']==split]
            assert len(rr)==24
            result[str(site)][split]={}
            for name in ['dense_real','global128',*[a[0] for a in ARMS]]:
                result[str(site)][split][name]={}
                for kind in rr[0]['metrics'][name]:
                    rec={}
                    for metric in ['relative_RMS','centered_relative_RMS','label_relative_RMS']:
                        ds={d:float(np.mean([r['metrics'][name][kind][metric] for r in rr if r['domain']==d]))
                            for d in sorted({r['domain'] for r in rr})}
                        vals=[r['metrics'][name][kind][metric] for r in rr]
                        rec[metric]=dict(mean=float(np.mean(vals)),worst=float(max(vals)),
                                         domain_means=ds,worst_domain_mean=float(max(ds.values())))
                    result[str(site)][split][name][kind]=rec
    return result


def prepare(a):
    import numpy as np,psutil
    tp=DOC/'true_input_temporal_result_20261013.json'
    ta=DOC/'true_input_temporal_adjudication_20261013.json'
    rp=DOC/'source_function_rank_result_20261013.json'
    ra=DOC/'source_function_rank_adjudication_v2_20261013.json'
    t=json.loads(tp.read_bytes())
    ad=json.loads(ta.read_bytes())
    ranks=json.loads(rp.read_bytes())
    rad=json.loads(ra.read_bytes())
    assert ad['decision']==rad['decision']=='INDEPENDENT_STORED_PASS'
    assert ad['report']==extent(tp) and rad['report']==extent(rp)
    tb=json.loads(Path(t['binding']['path']).read_bytes())
    rb=json.loads(Path(ranks['binding']['path']).read_bytes())
    capture=json.loads(Path(tb['capture']['path']).read_bytes())
    old={r['id']:r for r in capture['records']}
    storage=DOC/'conditional_output_function_storage_20261013.json'
    st=json.loads(storage.read_text(encoding='utf-8-sig'))
    assert not a.directory.exists() and not a.out.exists()
    assert a.directory.resolve()==Path(st['new_output_directory']).resolve()
    records=[]
    for r in t['records']:
        q=old[r['id']]
        assert (r['history'],r['domain'],r['split'])==(q['history'],q['domain'],q['split'])
        records.append(dict(id=r['id'],site=r['site'],history=r['history'],split=r['split'],
                            domain=r['domain'],positions=q['positions'],native=r['file'],
                            source_output=q['sites'][r['site']]['fields']['output']))
    assert len(records)==144
    paths=[Path(__file__),B/'conditional_output_function_audit.py',
           B/'falcon_auxiliary_s6_transport.py',DOC/'CONDITIONAL_OUTPUT_FUNCTION_PROTOCOL_20261013.md',
           storage,tp,ta,rp,ra,Path(t['binding']['path']),Path(ranks['binding']['path']),
           Path(tb['capture']['path']),Path(rb['source_weights']),Path(rb['source_config']),
           Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),
           Path(np._core._multiarray_umath.__file__),Path(np.linalg._umath_linalg.__file__),Path(psutil.__file__)]
    nr=Path(np.__file__).resolve().parent
    pr=Path(psutil.__file__).resolve().parent
    paths+=list((nr.parent/'numpy.libs').glob('*.dll'))+list(pr.glob('*.pyd'))
    paths+=[Path(e['file']['path']) for e in ranks['records']]
    paths+=[Path(e['parameters']['path']) for e in tb['exports']]
    paths+=[Path(r[k]['path']) for r in records for k in ['native','source_output']]
    paths+=[ROOT/p for p in ['benchmarks/phase60/engine.c',
           'benchmarks/donor_adaptation/configs/_manifest.json',
           'benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']]
    inputs=[extent(p) for p in dict.fromkeys(paths)]
    b=dict(schema='CONDITIONAL_OUTPUT_FUNCTION_BINDING_V1',freeze=a.freeze,
           python=str(Path(sys.executable).resolve()),versions=dict(numpy=np.__version__,psutil=psutil.__version__),
           temporal=extent(tp),temporal_adjudication=extent(ta),rank_report=extent(rp),rank_adjudication=extent(ra),
           capture=tb['capture'],exports=tb['exports'],rank_files=ranks['records'],
           source_weights=rb['source_weights'],source_config=rb['source_config'],records=records,
           sites=[0,12,23],arms=ARMS,seed=20261013,sketch_features=16,sketch_support=64,
           numeric=dict(array_relative=1e-10,scalar_absolute_or_relative=1e-10,
                        orthogonal_max=1e-10,eigen_residual_relative=1e-10,
                        squared_spectrum_relative=1e-8,dense_vs_native_relative=1e-6),
           criteria=dict(additional_raw_each_case=.01,additional_BF16_each_case=.01,
                         donor_mean=.1,donor_worst=.2,donor_centered=.1,donor_domain=.2),
           limits=dict(seconds=900,audit_seconds=1200,OS_bytes=6<<30,output_bytes=3<<30),
           output_directory=str(a.directory.resolve()),inputs=inputs,
           scope='Observable normalized3072 input routed through sparse ternary sketch/binary tree '
                 'to one affine F64 output expert. FIT-only objects;all48 held-out/source metrics '
                 'with explicit BF16 output rounding. No new source/state/generator/native kernel '
                 'or OWN-history/chatbot/ternary bank/DRAM/model-speed/n admission.')
    assert time.monotonic()-START<900 and psutil.Process().memory_info().peak_wset<6<<30
    write(a.out,b)
    print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)


def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha
    b=json.loads(a.binding.read_bytes())
    assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    assert np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil']
    assert not a.directory.exists() and not a.out.exists()
    assert a.directory.resolve()==Path(b['output_directory']).resolve()
    proc=psutil.Process()
    proc.cpu_affinity(list(range(6)))
    stage='startup'
    stream=None
    rows=[]
    models=[]
    receipt=dict(schema='CONDITIONAL_OUTPUT_FUNCTION_TERMINAL_V1',command=sys.orig_argv,
                 pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)

    def guard():
        assert time.monotonic()-START<=b['limits']['seconds']
        assert proc.memory_info().peak_wset<=b['limits']['OS_bytes']
        assert not proc.children(recursive=True)
        if a.directory.exists():
            assert sum(p.stat().st_size for p in a.directory.iterdir())<=b['limits']['output_bytes']

    def check():
        for e in b['inputs']:
            guard()
            assert extent(e['path'])==e,e['path']

    def data(r):
        with np.load(r['native']['path'],allow_pickle=False) as z:
            Z=decode(z['reconstructed_denominator__normalized'],np)
            Y=z['reconstructed_denominator__raw_scores'].astype('f8')
            bits=z['reconstructed_denominator__output'].copy()
        assert Z.shape==(r['history'],3072) and Y.shape==bits.shape==(r['history'],2048)
        assert np.isfinite(Z).all() and np.isfinite(Y).all()
        return Z,Y,bits

    try:
        check()
        a.directory.mkdir()
        get,stream=tensor_reader(b['source_weights'],np)
        rng=np.random.default_rng(b['seed'])
        indices=np.array([rng.choice(3072,64,replace=False) for _ in range(16)],'i4')
        signs=(rng.integers(0,2,size=(16,64))*2-1).astype('i1')
        # All three sites' FIT objects are saved before reading any DEV array.
        for site in b['sites']:
            stage=f'{site}/FIT_models'
            guard()
            rr=[r for r in b['records'] if r['site']==site and r['split']=='FIT']
            assert len(rr)==24 and sum(r['history'] for r in rr)==11148
            zs,ys,weights=[],[],[]
            for r in rr:
                Z,Y,_=data(r)
                zs.append(Z)
                ys.append(Y)
                weights.append(np.full(r['history'],1/(24*r['history'])))
            Z=np.concatenate(zs)
            Y=np.concatenate(ys)
            w=np.concatenate(weights)
            del zs,ys,weights
            Q=sketch(Z,indices,signs,np)
            O=get(f'model.layers.{site}.mamba.out_proj.weight')
            exp=next(e for e in b['exports'] if e['site']==site)
            with np.load(exp['parameters']['path'],allow_pickle=False) as z:
                assert np.array_equal(O.astype('f4'),z['O'])
            assert O.shape==(2048,3072)
            rf=next(e['file'] for e in b['rank_files'] if e['site']==site)
            with np.load(rf['path'],allow_pickle=False) as z:
                gu=z['output_eigenvectors'][:,-128:].copy()
            gb=gu.T@O
            gp=a.directory/f'site{site:02d}.global.npz'
            np.savez(gp,indices=indices,signs=signs,U=gu,B=gb)
            total=float(np.sum(np.sum(Y*Y,axis=1)*w))
            model=dict(site=site,global_file=extent(gp),banks=[],FIT_uncentered_output_energy=total)
            for name,K,rank in ARMS:
                stage=f'{site}/{name}/FIT'
                bank,leaves=fit_bank(Z,Y,w,Q,O,K,rank,np,guard)
                mean_energy=float(sum(q['mass']*np.sum(bank['mean_Y'][q['id']]**2) for q in leaves))
                within=float(sum(q['mass']*q['centered_energy'] for q in leaves))
                discarded=float(sum(q['mass']*q['discarded_energy'] for q in leaves))
                p=a.directory/f'site{site:02d}.{name}.bank.npz'
                np.savez(p,**bank)
                decomposition=dict(mean_energy=mean_energy,within_energy=within,discarded_energy=discarded,
                                   energy_identity_relative=float(abs(total-mean_energy-within)/total),
                                   projected_relative_RMS=float(np.sqrt(discarded/total)))
                ng=dict(orthogonality=all(q['orthogonal_max']<=b['numeric']['orthogonal_max'] for q in leaves),
                        eigen_residual=all(q['eigen_residual_relative']<=b['numeric']['eigen_residual_relative'] for q in leaves),
                        energy_identity=decomposition['energy_identity_relative']<=b['numeric']['scalar_absolute_or_relative'])
                model['banks'].append(dict(name=name,K=K,rank=rank,file=extent(p),leaves=leaves,
                                           decomposition=decomposition,numeric_gates=ng,
                                           active_coefficients=rank*5120+2048,total_coefficients=K*(rank*5120+2048)))
                print(json.dumps(dict(stage=stage,bank_complete=True,projection_FIT_RMS=decomposition['projected_relative_RMS'],
                                      seconds=time.monotonic()-START)),flush=True)
                del bank
            models.append(model)
            del Z,Y,Q,w,O,gu,gb
        print(json.dumps(dict(all_FIT_objects_frozen=True,DEV_arrays_read_so_far=0)),flush=True)
        for model in models:
            site=model['site']
            O=get(f'model.layers.{site}.mamba.out_proj.weight')
            with np.load(model['global_file']['path'],allow_pickle=False) as z:
                global_bank={k:z[k].copy() for k in z.files}
            banks={}
            for q in model['banks']:
                with np.load(q['file']['path'],allow_pickle=False) as z:
                    banks[q['name']]={k:z[k].copy() for k in z.files}
            projection_fit_energy={name:0. for name,_,_ in ARMS}
            for r in [e for e in b['records'] if e['site']==site]:
                stage=f'{site}/{r["id"]}/evaluate'
                guard()
                Z,Y,basebits=data(r)
                f=r['source_output']
                source=decode(np.fromfile(f['path'],'<u2').reshape(f['shape']),np)
                base=decode(basebits,np)
                mm={}
                dense=Z@O.T
                gpred=(Z@global_bank['B'].T)@global_bank['U'].T
                for name,pred in [('dense_real',dense),('global128',gpred)]:
                    out=decode(rounded(pred,np),np)
                    mm[name]=dict(raw_vs_native=metrics(pred,Y,r['positions'],np),
                                  BF16_vs_baseline=metrics(out,base,r['positions'],np),
                                  BF16_vs_source=metrics(out,source,r['positions'],np))
                arrays=dict(mass=np.ones(r['history'],'f4'))
                flags={}
                mismatches={}
                Q=sketch(Z,global_bank['indices'],global_bank['signs'],np)
                for name,K,rank in ARMS:
                    IDs,pred,proxy,means=evaluate(banks[name],Z,Y,Q,K,np)
                    bits=rounded(pred,np)
                    out=decode(bits,np)
                    mm[name]=dict(raw_vs_native=metrics(pred,Y,r['positions'],np),
                                  BF16_vs_baseline=metrics(out,base,r['positions'],np),
                                  BF16_vs_source=metrics(out,source,r['positions'],np),
                                  projection_vs_native=metrics(proxy,Y,r['positions'],np),
                                  mean_vs_native=metrics(means,Y,r['positions'],np),
                                  factor_vs_projection=metrics(pred,proxy,r['positions'],np))
                    flags[name]=dict(additional_raw=mm[name]['raw_vs_native']['relative_RMS']<=b['criteria']['additional_raw_each_case'],
                                     additional_BF16=mm[name]['BF16_vs_baseline']['relative_RMS']<=b['criteria']['additional_BF16_each_case'])
                    arrays[name+'__scores']=pred
                    arrays[name+'__IDs']=IDs
                    mismatches[name]=int(np.count_nonzero(bits!=basebits))
                    if r['split']=='FIT':
                        projection_fit_energy[name]+=float(np.sum((proxy-Y)**2)/(24*r['history']))
                p=a.directory/f'{r["id"]}.site{site:02d}.npz'
                np.savez(p,**arrays)
                rows.append(dict(id=r['id'],site=site,split=r['split'],domain=r['domain'],history=r['history'],
                                 file=extent(p),metrics=mm,additional_flags=flags,BF16_mismatches=mismatches,
                                 dense_numeric_gate=mm['dense_real']['raw_vs_native']['relative_RMS']<=b['numeric']['dense_vs_native_relative']))
                print(json.dumps(dict(stage=stage,complete=len(rows),additional_flags=flags,
                                      seconds=time.monotonic()-START)),flush=True)
            for bank in model['banks']:
                observed=projection_fit_energy[bank['name']]
                expected=bank['decomposition']['discarded_energy']
                bank['decomposition']['replayed_projection_energy']=observed
                bank['decomposition']['replayed_energy_absolute_or_relative']=abs(observed-expected)/max(abs(expected),1.)
                bank['numeric_gates']['replayed_energy']=bank['decomposition']['replayed_energy_absolute_or_relative']<=b['numeric']['scalar_absolute_or_relative']
            del O,banks,global_bank
        stream.close()
        stream=None
        check()
        for model in models:
            for e in [model['global_file'],*[q['file'] for q in model['banks']]]:
                assert extent(e['path'])==e
        guard()
        aa=aggregate(rows,np)
        gates={}
        for site in b['sites']:
            gates[str(site)]={}
            for name,_,_ in ARMS:
                gates[str(site)][name]={}
                for split in ['FIT','DEV']:
                    d=aa[str(site)][split][name]['BF16_vs_source']
                    gates[str(site)][name][split]=dict(mean=d['relative_RMS']['mean']<=b['criteria']['donor_mean'],
                            worst=d['relative_RMS']['worst']<=b['criteria']['donor_worst'],
                            centered=d['centered_relative_RMS']['mean']<=b['criteria']['donor_centered'],
                            domain=d['relative_RMS']['worst_domain_mean']<=b['criteria']['donor_domain'])
        ng=all(q['dense_numeric_gate'] for q in rows) and all(all(q['numeric_gates'].values()) for m in models for q in m['banks'])
        additional={name:all(all(q['additional_flags'][name].values()) for q in rows) for name,_,_ in ARMS}
        result=dict(schema='CONDITIONAL_OUTPUT_FUNCTION_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,
                    models=models,records=rows,aggregates=aa,donor_gates=gates,numeric_all_pass=ng,
                    additional_all_cases=additional,all_FIT_objects_frozen_before_DEV=True,
                    source_forwards=0,native_calls=0,optimizer_updates=0,new_ternary_expert_banks=0,
                    before_after_hashes_exact=True,decision='COMPLETE_PENDING_FULL_INDEPENDENT_AUDIT',
                    output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()),seconds=time.monotonic()-START,
                    OS_peak=proc.memory_info().peak_wset,scope=b['scope'])
        write(a.out,result)
        receipt.update(exit_code=0,error=None,result=extent(a.out))
        print(json.dumps(dict(result=extent(a.out),numeric_all_pass=ng,additional_all_cases=additional)),flush=True)
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
        receipt=dict(schema='CONDITIONAL_OUTPUT_FUNCTION_PREPARATION_TERMINAL_V1',command=sys.orig_argv,
                     pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
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
