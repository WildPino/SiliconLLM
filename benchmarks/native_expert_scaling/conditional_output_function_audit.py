"""Independent input-route/tree/local-spectrum/factor/full-prediction audit."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
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
    assert b['arms']==[['tree16_r128',16,128],['tree64_r32',64,32]] and not a.out.exists()
    proc=psutil.Process()
    proc.cpu_affinity(list(range(6)))
    receipt=dict(schema='CONDITIONAL_OUTPUT_FUNCTION_AUDIT_TERMINAL_V1',command=sys.orig_argv,
                 pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    stage='startup'
    stream=None
    maxima={}
    gates={}
    rows=[]
    models=[]
    counts=dict(bank_float_words=0,bank_integer_words=0,predicted_F64_words=0,
                routed_IDs=0,masses=0,FIT_Z_Y_words=0,evaluated_Z_Y_words=0,
                rounded_BF16_words=0,independent_BF16_mismatches=0,full_leaf_singular_values=0)
    num=b['numeric']

    def guard():
        assert time.monotonic()-START<=b['limits']['audit_seconds']
        assert proc.memory_info().peak_wset<=b['limits']['OS_bytes']
        assert not proc.children(recursive=True)

    def check():
        ext=[r['binding'],re,te]
        for m in r['models']:
            ext+=[m['global_file']]+[q['file'] for q in m['banks']]
        ext+=[q['file'] for q in r['records']]
        for e in b['inputs']+ext:
            guard()
            assert extent(e['path'])==e,e['path']

    def metric(key,error,tol):
        error=float(error)
        assert np.isfinite(error),(stage,key)
        maxima[key]=max(maxima.get(key,0.),error)
        gates[key]=bool(gates.get(key,True) and error<=tol)
        return error

    def rel(x,y):
        return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))

    def close(key,x,y):
        assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
        return metric(key,rel(x,y),num['array_relative'])

    def meta(x,y):
        if isinstance(y,dict):
            assert x.keys()==y.keys(),(stage,x.keys(),y.keys())
            for k in y:
                meta(x[k],y[k])
        elif isinstance(y,list):
            assert len(x)==len(y)
            for v,z in zip(x,y):
                meta(v,z)
        elif isinstance(y,(bool,int,str)):
            assert x==y,(stage,x,y)
        else:
            metric('metadata_absolute_or_relative',abs(x-y)/max(abs(y),1.),num['scalar_absolute_or_relative'])

    def read(e):
        with np.load(e['path'],allow_pickle=False) as z:
            arrays={k:z[k].copy() for k in z.files}
        for v in arrays.values():
            assert np.isfinite(v).all()
            if v.dtype.kind=='f':
                counts['bank_float_words']+=v.size
            else:
                counts['bank_integer_words']+=v.size
        return arrays

    def decode(bits):
        return (bits.astype('<u4')*65536).view('<f4').astype('f8')

    def bf(X):
        bits=X.astype('<f4').view('<u4')
        upper=bits>>16
        lower=bits&65535
        inc=(lower>32768)|((lower==32768)&((upper&1)!=0))
        return (upper+inc.astype('u4')).astype('<u2')

    def raw_data(e):
        with np.load(e['native']['path'],allow_pickle=False) as z:
            zbits=z['reconstructed_denominator__normalized'].copy()
            yy=z['reconstructed_denominator__raw_scores'].copy()
            ybits=z['reconstructed_denominator__output'].copy()
        assert zbits.dtype==ybits.dtype==np.dtype('<u2') and yy.dtype==np.dtype('<f4')
        assert zbits.shape==(e['history'],3072) and yy.shape==ybits.shape==(e['history'],2048)
        Z,Y=decode(zbits),yy.astype('f8')
        assert np.isfinite(Z).all() and np.isfinite(Y).all()
        return Z,Y,ybits

    def sketches(Z,indices,signs):
        # Explicit fixed-order addition independently simulates the observable router.
        out=[]
        for ix,sg in zip(indices,signs):
            acc=np.zeros(len(Z))
            for coordinate,sign in zip(ix,sg):
                acc=np.add(acc,np.multiply(Z[:,coordinate],int(sign)))
            out.append(acc/8.)
        return np.column_stack(out)

    def assignments(Q,features,cuts,K):
        ids=np.empty(len(Q),'u4')
        for lo in range(0,len(Q),32):
            nodes=np.zeros(min(32,len(Q)-lo),'i4')
            for _ in range(K.bit_length()-1):
                values=Q[lo:lo+len(nodes)][np.arange(len(nodes)),features[nodes]]
                nodes=2*nodes+np.where(values<=cuts[nodes],1,2)
            ids[lo:lo+len(nodes)]=nodes-(K-1)
        return ids

    def tree(Q,w,K,rank):
        features=np.full(K-1,-1,'i4')
        cuts=np.zeros(K-1)
        def node(at,positions,depth):
            guard()
            if depth==0:
                assert len(positions)>=rank+1
                return
            weights=w[positions]
            mass=float(weights.sum())
            values=Q[positions]
            av=np.array([np.dot(weights,values[:,j])/mass for j in range(16)])
            var=np.array([np.dot(weights,(values[:,j]-av[j])**2)/mass for j in range(16)])
            required=(rank+1)*2**(depth-1)
            for feature in sorted(range(16),key=lambda j:(-float(var[j]),j)):
                if var[feature]<=0:
                    continue
                order=np.lexsort((np.arange(len(positions)),values[:,feature]))
                vv=values[order,feature]
                boundary=np.nonzero(vv[:-1]<vv[1:])[0]
                valid=[int(i) for i in boundary if i+1>=required and len(order)-i-1>=required]
                if not valid:
                    continue
                partial=np.add.accumulate(weights[order])
                best=min(valid,key=lambda i:(abs(float(partial[i])-mass*.5),i))
                features[at]=feature
                cuts[at]=vv[best]
                left=positions[values[:,feature]<=cuts[at]]
                right=positions[values[:,feature]>cuts[at]]
                node(2*at+1,left,depth-1)
                node(2*at+2,right,depth-1)
                return
            raise ValueError(f'UNSUPPORTED_PARTITION node={at}')
        node(0,np.arange(len(Q)),K.bit_length()-1)
        return features,cuts,assignments(Q,features,cuts,K)

    def measurements(X,Y,pos):
        return dict(relative_RMS=rel(X,Y),centered_relative_RMS=rel(X-X.mean(0),Y-Y.mean(0)),
                    label_relative_RMS=rel(X[pos],Y[pos]))

    def predict(bank,Z,Y,Q,K):
        ids=assignments(Q,bank['features'],bank['cuts'],K)
        output=np.empty_like(Y)
        proxy=np.empty_like(Y)
        means=np.empty_like(Y)
        for j in range(K):
            positions=np.flatnonzero(ids==j)
            u=bank['U'][j]
            for lo in range(0,len(positions),32):
                ix=positions[lo:lo+32]
                low=np.einsum('td,rd->tr',Z[ix],bank['B'][j],optimize=True)
                output[ix]=np.einsum('tr,dr->td',low,u,optimize=True)+bank['bias'][j]
                p=np.einsum('td,dr->tr',Y[ix]-bank['mean_Y'][j],u,optimize=True)
                proxy[ix]=np.einsum('tr,dr->td',p,u,optimize=True)+bank['mean_Y'][j]
                means[ix]=bank['mean_Y'][j]
        return ids,output,proxy,means

    try:
        check()
        cad=json.loads(Path(b['temporal_adjudication']['path']).read_bytes())
        rad=json.loads(Path(b['rank_adjudication']['path']).read_bytes())
        assert cad['decision']==rad['decision']=='INDEPENDENT_STORED_PASS'
        assert cad['report']==b['temporal'] and rad['report']==b['rank_report']
        native=json.loads(Path(b['temporal']['path']).read_bytes())
        capture=json.loads(Path(b['capture']['path']).read_bytes())
        byid={q['id']:q for q in capture['records']}
        expected=[]
        for e in native['records']:
            q=byid[e['id']]
            assert (e['history'],e['domain'],e['split'])==(q['history'],q['domain'],q['split'])
            expected.append(dict(id=e['id'],site=e['site'],history=e['history'],split=e['split'],domain=e['domain'],
                                 positions=q['positions'],native=e['file'],source_output=q['sites'][e['site']]['fields']['output']))
        assert expected==b['records'] and len(r['records'])==144
        rng=np.random.default_rng(b['seed'])
        indices=np.array([rng.choice(3072,64,replace=False) for _ in range(16)],'i4')
        signs=(rng.integers(0,2,size=(16,64))*2-1).astype('i1')
        get,stream=tensor_reader(b['source_weights'],np)
        for model in r['models']:
            site=model['site']
            stage=f'{site}/FIT_model_audit'
            guard()
            ff=[e for e in b['records'] if e['site']==site and e['split']=='FIT']
            ZZ,YY,WW=[],[],[]
            for e in ff:
                Z,Y,_=raw_data(e)
                ZZ.append(Z)
                YY.append(Y)
                WW.append(np.full(e['history'],1/(24*e['history'])))
            Z,Y,w=np.concatenate(ZZ),np.concatenate(YY),np.concatenate(WW)
            assert len(Z)==11148 and len(ff)==24
            del ZZ,YY,WW
            counts['FIT_Z_Y_words']+=Z.size+Y.size
            Q=sketches(Z,indices,signs)
            O=get(f'model.layers.{site}.mamba.out_proj.weight')
            with np.load(next(e['parameters']['path'] for e in b['exports'] if e['site']==site),allow_pickle=False) as z:
                assert np.array_equal(O.astype('f4'),z['O'])
            glob=read(model['global_file'])
            assert set(glob)=={'indices','signs','U','B'}
            assert glob['indices'].dtype==np.dtype('<i4') and glob['signs'].dtype==np.dtype('i1')
            assert glob['U'].dtype==glob['B'].dtype==np.dtype('<f8')
            assert np.array_equal(glob['indices'],indices) and np.array_equal(glob['signs'],signs)
            rank_file=next(e['file']['path'] for e in b['rank_files'] if e['site']==site)
            with np.load(rank_file,allow_pickle=False) as z:
                assert np.array_equal(glob['U'],z['output_eigenvectors'][:,-128:])
            close('global_B',glob['B'],glob['U'].T@O)
            total=float(sum(float(np.dot(yy,yy))*wt for yy,wt in zip(Y,w)))
            meta(model['FIT_uncentered_output_energy'],total)
            banks={}
            leaf_gates=[]
            for entry in model['banks']:
                name,K,rank=entry['name'],entry['K'],entry['rank']
                assert [name,K,rank] in b['arms']
                stage=f'{site}/{name}/full_leaf_audit'
                bank=read(entry['file'])
                assert set(bank)=={'features','cuts','fit_IDs','U','B','bias','mean_Z','mean_Y','singular_values','singular_offsets'}
                assert bank['features'].dtype==bank['singular_offsets'].dtype==np.dtype('<i4')
                assert bank['fit_IDs'].dtype==np.dtype('<u4')
                assert all(bank[k].dtype==np.dtype('<f8') for k in ['cuts','U','B','bias','mean_Z','mean_Y','singular_values'])
                features,cuts,ids=tree(Q,w,K,rank)
                assert np.array_equal(bank['features'],features) and np.array_equal(bank['cuts'],cuts)
                assert np.array_equal(bank['fit_IDs'],ids)
                assert bank['U'].shape==(K,2048,rank) and bank['B'].shape==(K,rank,3072)
                assert bank['mean_Z'].shape==(K,3072) and bank['mean_Y'].shape==bank['bias'].shape==(K,2048)
                assert bank['singular_offsets'].shape==(K+1,) and bank['singular_offsets'][0]==0
                leafmeta=[]
                for j in range(K):
                    guard()
                    ix=np.flatnonzero(ids==j)
                    weights=w[ix]
                    mass=float(weights.sum())
                    mz=np.einsum('t,td->d',weights,Z[ix],optimize=True)/mass
                    my=np.einsum('t,td->d',weights,Y[ix],optimize=True)/mass
                    close('mean_Z',bank['mean_Z'][j],mz)
                    close('mean_Y',bank['mean_Y'][j],my)
                    M=(Y[ix]-my)*np.sqrt(weights/mass)[:,None]
                    direct=np.linalg.svd(M,compute_uv=False)
                    lo,hi=map(int,bank['singular_offsets'][j:j+2])
                    sv=bank['singular_values'][lo:hi]
                    assert sv.shape==direct.shape and np.all(np.diff(sv)<=0) and (sv>=0).all()
                    counts['full_leaf_singular_values']+=sv.size
                    metric('local_direct_squared_spectrum',np.max(np.abs(sv**2-direct**2))/max(direct[0]**2,1e-30),num['squared_spectrum_relative'])
                    u=bank['U'][j]
                    orth=float(np.max(np.abs(u.T@u-np.eye(rank))))
                    energy=float(np.einsum('td,td->',M,M))
                    residual=float(np.linalg.norm(M.T@(M@u)-u*sv[:rank]**2)/max(energy,1e-30))
                    metric('local_orthogonal_max',orth,num['orthogonal_max'])
                    metric('local_eigen_residual',residual,num['eigen_residual_relative'])
                    expected_B=u.T@O
                    close('local_B',bank['B'][j],expected_B)
                    projected_mean=u@(expected_B@mz)
                    metric('local_bias_equation_backward',np.linalg.norm(bank['bias'][j]-(my-projected_mean))/
                           max(np.linalg.norm(my)+np.linalg.norm(projected_mean),1e-30),num['array_relative'])
                    leafmeta.append(dict(id=j,positions=len(ix),mass=mass,centered_energy=energy,
                                         discarded_energy=float(np.sum(sv[rank:]**2)),orthogonal_max=orth,
                                         eigen_residual_relative=residual,zero_singular_values=int((sv==0).sum())))
                assert bank['singular_offsets'][-1]==len(bank['singular_values'])
                meta(entry['leaves'],leafmeta)
                mean_energy=float(sum(q['mass']*np.dot(bank['mean_Y'][q['id']],bank['mean_Y'][q['id']]) for q in leafmeta))
                within=float(sum(q['mass']*q['centered_energy'] for q in leafmeta))
                discarded=float(sum(q['mass']*q['discarded_energy'] for q in leafmeta))
                decomp=dict(mean_energy=mean_energy,within_energy=within,discarded_energy=discarded,
                            energy_identity_relative=float(abs(total-mean_energy-within)/total),
                            projected_relative_RMS=float(np.sqrt(discarded/total)))
                for key,value in decomp.items():
                    meta(entry['decomposition'][key],value)
                assert entry['active_coefficients']==rank*5120+2048
                assert entry['total_coefficients']==K*(rank*5120+2048)
                leaf_gates.append(dict(name=name,orthogonality=all(q['orthogonal_max']<=num['orthogonal_max'] for q in leafmeta),
                                       eigen_residual=all(q['eigen_residual_relative']<=num['eigen_residual_relative'] for q in leafmeta),
                                       energy_identity=decomp['energy_identity_relative']<=num['scalar_absolute_or_relative']))
                banks[name]=bank
                print(json.dumps(dict(stage=stage,leaves_verified=K,seconds=time.monotonic()-START)),flush=True)
            del Z,Y,Q,w
            energy_replay={name:0. for name,_,_ in b['arms']}
            for e in [q for q in b['records'] if q['site']==site]:
                stage=f'{site}/{e["id"]}/full_prediction_audit'
                guard()
                Z,Y,basebits=raw_data(e)
                counts['evaluated_Z_Y_words']+=Z.size+Y.size
                source=e['source_output']
                source=decode(np.fromfile(source['path'],'<u2').reshape(source['shape']))
                base=decode(basebits)
                original=next(q for q in r['records'] if q['site']==site and q['id']==e['id'])
                with np.load(original['file']['path'],allow_pickle=False) as z:
                    arrays={k:z[k].copy() for k in z.files}
                assert set(arrays)=={'mass',*[name+'__'+kind for name,_,_ in b['arms'] for kind in ['scores','IDs']]}
                assert arrays['mass'].dtype==np.dtype('<f4') and np.array_equal(arrays['mass'],np.ones(e['history'],'f4'))
                counts['masses']+=e['history']
                mm={}
                dense=np.empty_like(Y)
                global_pred=np.empty_like(Y)
                for lo in range(0,len(Z),32):
                    z=Z[lo:lo+32]
                    dense[lo:lo+32]=z@O.T
                    global_pred[lo:lo+32]=(z@glob['B'].T)@glob['U'].T
                for name,pred in [('dense_real',dense),('global128',global_pred)]:
                    out=decode(bf(pred))
                    mm[name]=dict(raw_vs_native=measurements(pred,Y,e['positions']),
                                  BF16_vs_baseline=measurements(out,base,e['positions']),
                                  BF16_vs_source=measurements(out,source,e['positions']))
                Q=sketches(Z,indices,signs)
                flags={}
                mismatches={}
                for name,K,rank in b['arms']:
                    ids,pred,proxy,means=predict(banks[name],Z,Y,Q,K)
                    stored=arrays[name+'__scores']
                    assert stored.dtype==np.dtype('<f8') and stored.shape==Y.shape
                    assert arrays[name+'__IDs'].dtype==np.dtype('<u4')
                    assert np.array_equal(ids,arrays[name+'__IDs'])
                    counts['routed_IDs']+=len(ids)
                    counts['predicted_F64_words']+=stored.size
                    close('full_factor_prediction',stored,pred)
                    bits=bf(stored)
                    different=int(np.count_nonzero(bits!=bf(pred)))
                    counts['rounded_BF16_words']+=bits.size
                    counts['independent_BF16_mismatches']+=different
                    gates['independent_BF16_bits']=bool(gates.get('independent_BF16_bits',True) and different==0)
                    out=decode(bits)
                    mm[name]=dict(raw_vs_native=measurements(stored,Y,e['positions']),
                                  BF16_vs_baseline=measurements(out,base,e['positions']),
                                  BF16_vs_source=measurements(out,source,e['positions']),
                                  projection_vs_native=measurements(proxy,Y,e['positions']),
                                  mean_vs_native=measurements(means,Y,e['positions']),
                                  factor_vs_projection=measurements(stored,proxy,e['positions']))
                    flags[name]=dict(additional_raw=mm[name]['raw_vs_native']['relative_RMS']<=b['criteria']['additional_raw_each_case'],
                                     additional_BF16=mm[name]['BF16_vs_baseline']['relative_RMS']<=b['criteria']['additional_BF16_each_case'])
                    mismatches[name]=int(np.count_nonzero(bits!=basebits))
                    if e['split']=='FIT':
                        energy_replay[name]+=float(np.sum((proxy-Y)**2)/(24*e['history']))
                meta(original['metrics'],mm)
                assert original['additional_flags']==flags and original['BF16_mismatches']==mismatches
                dense_gate=mm['dense_real']['raw_vs_native']['relative_RMS']<=num['dense_vs_native_relative']
                assert original['dense_numeric_gate']==dense_gate
                rows.append(dict(id=e['id'],site=site,split=e['split'],domain=e['domain'],metrics=mm,
                                 additional_flags=flags,dense_numeric_gate=dense_gate))
            for entry,lg in zip(model['banks'],leaf_gates):
                observed=energy_replay[entry['name']]
                expected=entry['decomposition']['discarded_energy']
                err=abs(observed-expected)/max(abs(expected),1.)
                meta(entry['decomposition']['replayed_projection_energy'],observed)
                meta(entry['decomposition']['replayed_energy_absolute_or_relative'],err)
                lg['replayed_energy']=err<=num['scalar_absolute_or_relative']
                assert entry['numeric_gates']=={k:v for k,v in lg.items() if k!='name'}
            models.append(dict(site=site,banks=leaf_gates))
            del banks,glob,O
        stream.close()
        stream=None
        # Independent grouping of every held-out/source metric and acceptance flag.
        aa={}
        for site in b['sites']:
            aa[str(site)]={}
            for split in ['FIT','DEV']:
                rr=[q for q in rows if q['site']==site and q['split']==split]
                assert len(rr)==24
                aa[str(site)][split]={}
                for name in ['dense_real','global128',*[q[0] for q in b['arms']]]:
                    aa[str(site)][split][name]={}
                    for kind in rr[0]['metrics'][name]:
                        summary={}
                        for field in ['relative_RMS','centered_relative_RMS','label_relative_RMS']:
                            vals=[q['metrics'][name][kind][field] for q in rr]
                            dm={d:float(np.mean([q['metrics'][name][kind][field] for q in rr if q['domain']==d]))
                                for d in sorted({q['domain'] for q in rr})}
                            summary[field]=dict(mean=float(np.mean(vals)),worst=max(vals),domain_means=dm,worst_domain_mean=max(dm.values()))
                        aa[str(site)][split][name][kind]=summary
        meta(r['aggregates'],aa)
        donor={}
        for site in b['sites']:
            donor[str(site)]={}
            for name,_,_ in b['arms']:
                donor[str(site)][name]={}
                for split in ['FIT','DEV']:
                    d=aa[str(site)][split][name]['BF16_vs_source']
                    donor[str(site)][name][split]=dict(mean=d['relative_RMS']['mean']<=b['criteria']['donor_mean'],
                            worst=d['relative_RMS']['worst']<=b['criteria']['donor_worst'],
                            centered=d['centered_relative_RMS']['mean']<=b['criteria']['donor_centered'],
                            domain=d['relative_RMS']['worst_domain_mean']<=b['criteria']['donor_domain'])
        assert r['donor_gates']==donor
        numeric=all(q['dense_numeric_gate'] for q in rows) and all(all(v for k,v in q.items() if k!='name') for m in models for q in m['banks'])
        assert r['numeric_all_pass']==numeric
        additional={name:all(all(q['additional_flags'][name].values()) for q in rows) for name,_,_ in b['arms']}
        assert r['additional_all_cases']==additional and r['all_FIT_objects_frozen_before_DEV']
        assert all(r[k]==0 for k in ['source_forwards','native_calls','optimizer_updates','new_ternary_expert_banks'])
        check()
        guard()
        ad=dict(schema='CONDITIONAL_OUTPUT_FUNCTION_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,
                report=re,terminal=te,binding=r['binding'],models=models,counts=counts,maximum=maxima,
                verification_gates=gates,numeric_all_pass=numeric,additional_all_cases=additional,donor_gates=donor,
                before_after_hashes_exact=True,
                decision='INDEPENDENT_STORED_PASS' if numeric and all(gates.values()) else 'INDEPENDENT_STORED_NUMERIC_FAILURE',
                seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
                scope='Complete reconstructed input-only trees/IDs/masses/FIT local weighted means/spectra/eigenbasis/factors;'
                      'every stored F64 prediction/all144 histories/independent rounded BF16 outputs/source metrics and gates. '
                      'Stored F64 operator pilot;no native kernels/DRAM/causal/chatbot/quality-speed or useful ternary-n admission.')
        write(a.out,ad)
        receipt.update(exit_code=0,error=None,result=extent(a.out))
        print(json.dumps(dict(adjudication=extent(a.out),decision=ad['decision'],counts=counts,maximum=maxima)),flush=True)
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
    for k in ['report','terminal','out']:
        p.add_argument('--'+k,type=Path,required=True)
    for k in ['report-sha','terminal-sha','freeze']:
        p.add_argument('--'+k,required=True)
    main(p.parse_args())
