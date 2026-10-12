"""Independent coefficient/quadratic reconstruction; no producer/refit imports."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic()


def main(a):
    import numpy as np,psutil
    assert extent(a.report)['sha256']==a.report_sha and extent(a.terminal)['sha256']==a.terminal_sha
    r=json.loads(a.report.read_bytes());t=json.loads(a.terminal.read_bytes())
    assert t['exit_code']==0 and t['error'] is None and t['result']==extent(a.report)
    assert r['schema']=='SOURCE_FUNCTION_COORDINATES_RESULT_V1' and not a.out.exists()
    b=json.loads(Path(r['binding']['path']).read_bytes());assert extent(r['binding']['path'])==r['binding']
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None;maxdelta=0.;maxnumeric=0.;maxscore=0.
    receipt=dict(schema='SOURCE_FUNCTION_COORDINATES_AUDIT_TERMINAL_V1',command=sys.orig_argv,
        pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,source=extent(__file__),timing_control_admitted=False)
    audit_inputs=[extent(a.report),extent(a.terminal),extent(__file__),
        extent(DOC/'SOURCE_FUNCTION_COORDINATES_STORED_AUDIT_PROTOCOL_20261012.md')]
    def guard():
        assert time.monotonic()-START<1200 and proc.memory_info().peak_wset<=6<<30
        assert not proc.children(recursive=True)
    def check():
        for x in b['inputs']+r['outputs']+[r['binding']]+audit_inputs:
            guard();assert extent(x['path'])==x,x['path']
    def compare(x,y,tol=1e-10):
        nonlocal maxnumeric
        delta=float(np.linalg.norm(x-y)/max(float(np.linalg.norm(y)),1e-300));maxnumeric=max(maxnumeric,delta)
        assert delta<=tol,delta
    def scalar(x,y,tol=1e-8):
        nonlocal maxdelta
        delta=abs(float(x)-float(y));maxdelta=max(maxdelta,delta)
        assert delta<=tol*max(1.,abs(float(y))),(x,y,delta)
    def raw(rec,bd):
        x=next(x['h'] for x in rec['boundaries'] if x['boundary']==bd)
        return (np.fromfile(x['path'],'<u2').astype('<u4')<<16).view('<f4').reshape(rec['history'],2048).astype('f8')
    def partmul(x,A):
        # Source-width summation, independent of the producer's full BLAS/projection shortcut.
        out=np.zeros((*x.shape[:-1],A.shape[0]),'f8')
        for lo in range(0,2048,256):out+=x[...,lo:lo+256]@A[:,lo:lo+256].T
        return out
    try:
        check();get,stream=tensor_reader(Path(b['source'])/'model.safetensors',np)
        cfg=json.loads((Path(b['source'])/'config.json').read_bytes());allmetrics={};allobs=[];base_checks=[]
        for bd in b['boundaries']:
            guard();record=next(x for x in r['bases'] if x['boundary']==bd)
            prior=next(x for x in b['covariances'] if x['boundary']==bd)
            with np.load(prior['file']['path'],allow_pickle=False) as p: C=p['gram'];ce=p['eigenvalues'];cq=p['eigenvectors'];R=p['basis']
            with np.load(record['file']['path'],allow_pickle=False) as pack:
                S=pack['root'];J=pack['inverse'];M=pack['M'];K=pack['K'];eig=pack['eigenvalues'];U=pack['eigenvectors'];V=pack['V'];W=pack['W'];G=pack['norm_metric']
                assert np.array_equal(pack['V_f32'],V.astype('f4')) and np.array_equal(pack['W_f32'],W.astype('f4'))
            keep=ce>ce[0]*b['covariance_support_cut'];assert int(keep.sum())==record['support_dimension']
            compare(S,(cq[:,keep]*np.sqrt(ce[keep]))@cq[:,keep].T)
            compare(J,(cq[:,keep]/np.sqrt(ce[keep]))@cq[:,keep].T)
            compare(V,S@U[:,:256]);compare(W,J@U[:,:256]);compare(W.T@V,np.eye(256),1e-8);compare(G,V.T@V)
            compare(U.T@U,np.eye(2048));compare(K,U@np.diag(eig)@U.T)
            assert np.all(eig[:-1]>=eig[1:]) and eig[-1]>=-1e-10*max(1.,eig[0])
            Mtest=np.zeros((2048,2048),'f8');gs=[]
            if bd<24:
                base=f'model.layers.{bd}.';gamma=get(base+'input_layernorm.weight');proj=get(base+'mamba.in_proj.weight')
                bounds=[0,3072,6144,6400,6656,6704]
                for i,name in enumerate(['gate','x','B','C','dt']):
                    A=proj[bounds[i]:bounds[i+1]]*gamma*cfg['ssm_in_multiplier']*cfg['ssm_multipliers'][i];gs.append((name,A))
                for name in ['q','k','v']:
                    A=get(base+'self_attn.'+name+'_proj.weight')*gamma*cfg['attention_in_multiplier']
                    if name=='k':A=A*cfg['key_multiplier']
                    gs.append((name,A))
                for (name,A),desc in zip(gs,record['groups']):
                    gg=A.T@A;energy=float(np.trace(gg@C));scalar(energy,desc['FIT_energy']);Mtest+=gg/energy/8
                    assert desc['name']==name and desc['shape']==list(A.shape)
                del gg
            else:
                qbar=np.fromfile(b['qbar']['path'],'<f8');gamma=get('model.final_layernorm.weight');mean=np.zeros(2048,'f8')
                for lo in range(0,65537,1024):
                    guard();hi=min(65537,lo+1024);A=get('lm_head.weight',list(range(lo,hi)))*gamma*cfg['lm_head_multiplier']
                    weighted=A*np.sqrt(qbar[lo:hi,None]);Mtest+=weighted.T@weighted;mean+=qbar[lo:hi]@A
                Mtest-=np.outer(mean,mean);energy=float(np.trace(Mtest@C));scalar(energy,record['groups'][0]['FIT_energy']);Mtest/=energy
            compare(M,Mtest);compare(K,S@Mtest@S)
            scalar(np.trace(C@M),1.,1e-8);MV=M@V;VMV=V.T@MV;MR=M@R;RMR=R.T@MR
            selected=[]
            for rec in b['records']:
                guard();H=raw(rec,bd);u=H/np.sqrt(np.mean(H**2,axis=1,keepdims=True)+b['epsilon']);z=u@W
                # Quadratic expansion avoids producer explicit reconstruction and error-vector multiplication.
                truth=np.sum((u@M)*u,axis=1);cross=np.sum((u@MV)*z,axis=1);pred=np.sum((z@VMV)*z,axis=1)
                rawz=H@W;rad=np.sqrt(np.sum((rawz@G)*rawz,axis=1)/2048+b['epsilon'])
                s=np.sqrt(np.mean(H**2,axis=1)+b['epsilon'])/rad
                rz=u@R;rcross=np.sum((u@MR)*rz,axis=1);rpred=np.sum((rz@RMR)*rz,axis=1)
                rr=next(x for x in r['rows'] if x['boundary']==bd and x['id']==rec['id']);scalar(rr['true_to_own_radius_mean'],s.mean())
                for mode,terms in [('oracle_norm',truth-2*cross+pred),('own_norm',truth-2*s*cross+s*s*pred),('angular_oracle_norm',truth-2*rcross+rpred)]:
                    error=float(math.fsum(float(x) for x in terms));source=float(math.fsum(float(x) for x in truth))
                    assert error>=-1e-8*max(source,1.);error=max(error,0.);val=math.sqrt(error/max(source,1e-300))
                    vv=rr['values'][mode];scalar(vv['error_energy_per_token'],error/len(H));scalar(vv['source_energy_per_token'],source/len(H));scalar(vv['metric_relative_RMS'],val)
                    allmetrics[(bd,rec['split'],mode,rec['id'])]=val
                ids=[0,len(H)//2,len(H)-1];oracle=(u[ids]@W)@V.T
                selected.append(np.stack([u[ids],oracle,oracle*s[ids,None]]))
            samples=np.concatenate(selected,axis=1)
            for ss in [x for x in r['summary'] if x['boundary']==bd]:
                vals=[v for (bb,split,mode,_),v in allmetrics.items() if bb==bd and split=='DEV' and mode==ss['mode']]
                scalar(math.fsum(vals)/24,ss['DEV_metric_relative_RMS_mean']);scalar(max(vals),ss['DEV_metric_relative_RMS_worst'])
                assert ss['proxy_gates']=={k:ss[k]<=limit for k,limit in b['proxy'].items()}
            fit=[x for x in r['rows'] if x['boundary']==bd and x['split']=='FIT']
            scalar(record['FIT_observed_error_energy'],math.fsum(x['values']['oracle_norm']['error_energy_per_token'] for x in fit)/24)
            scalar(record['FIT_spectral_tail'],eig[256:].sum());scalar(record['FIT_metric_retained_fraction'],eig[:256].sum()/eig.sum())
            if bd<24:
                with np.load(record['samples_file']['path'],allow_pickle=False) as obs:
                    compare(obs['normalized'],samples)
                    for name,A in gs:
                        actual=partmul(samples,A);compare(obs[name],actual)
                        for split in ['FIT','DEV']:
                            mask=np.array([x['split']==split for x in record['sample_labels']]);truth=actual[0,mask]
                            for i,mode in [(1,'oracle_norm'),(2,'own_norm')]:
                                err=float(np.linalg.norm(actual[i,mask]-truth)/np.linalg.norm(truth))
                                row=next(x for x in r['observables'] if x['boundary']==bd and x['group']==name and x['mode']==mode and x['split']==split)
                                scalar(err,row['relative_RMS']);allobs.append(row)
            base_checks.append(dict(boundary=bd,full_history_cases=48,observed_positions=sum(rec['history'] for rec in b['records'])))
            print(json.dumps(dict(audited_boundary=bd,seconds=time.monotonic()-START)),flush=True)
        rd=r['readout'];ref=np.fromfile(rd['reference']['path'],'<f8').reshape(72,65537)
        with np.load(rd['file']['path'],allow_pickle=False) as stored:
            H=stored['raw_states'];z=stored['encoded'];head=stored['head'];scores=stored['scores']
        actualH=np.concatenate([raw(next(x for x in b['records'] if x['id']==s['id']),24)[s['positions']] for s in b['readout_samples']])
        assert np.array_equal(H,actualH);compare(z,H@W)
        radius=np.sqrt(np.mean(H**2,axis=1,keepdims=True)+b['epsilon']);recon=z@V.T
        own=np.sqrt(np.mean(recon**2,axis=1,keepdims=True)+b['epsilon'])
        inp=np.stack([H/radius,recon/radius,recon/own])
        for lo in range(0,65537,1024):
            guard();hi=min(65537,lo+1024);A=get('lm_head.weight',list(range(lo,hi)))*gamma*cfg['lm_head_multiplier']
            compare(head[lo:hi],partmul(V.T,A).T);calc=partmul(inp,A)
            compare(ref[:,lo:hi],calc[0]);compare(scores[:,:,lo:hi],calc[1:]);maxscore=max(maxscore,float(np.max(np.abs(scores[:,:,lo:hi]-calc[1:]))))
        for i,mode in [(0,'oracle_norm'),(1,'own_norm')]:
            losses=[];wrongs=[]
            for k,label in enumerate(rd['labels']):
                qt=ref[k]-np.logaddexp.reduce(ref[k]);pt=scores[i,k]-np.logaddexp.reduce(scores[i,k])
                loss=float(np.exp(qt)@(qt-pt));wrong=int(np.argmax(ref[k])!=np.argmax(scores[i,k]));losses.append(loss);wrongs.append(wrong)
                row=next(x for x in rd['rows'] if x['mode']==mode and all(x[f]==label[f] for f in label))
                scalar(loss,row['KL']);assert wrong==row['argmax_disagreement']
            ss=next(x for x in rd['summary'] if x['mode']==mode)
            scalar(ss['mean_KL'],math.fsum(losses)/72);scalar(ss['worst_KL'],max(losses));scalar(ss['disagreement_rate'],sum(wrongs)/72)
            assert ss['proxy_gates']=={k:ss[k]<=limit for k,limit in b['readout_proxy'].items()}
        assert np.array_equal(np.argmax(scores[0],axis=1),np.argmax(scores[1],axis=1))
        for item in r['transport']:
            src=next(x for x in r['bases'] if x['boundary']==item['source']);dst=next(x for x in r['bases'] if x['boundary']==item['destination'])
            with np.load(src['file']['path'],allow_pickle=False) as x,np.load(dst['file']['path'],allow_pickle=False) as y:
                value=y['W'].T@x['V'];actual=np.fromfile(item['file']['path'],'<f8').reshape(256,256);compare(actual,value)
                singular=np.linalg.svd(value,compute_uv=False);scalar(item['max_singular'],singular[0]);scalar(item['min_singular'],singular[-1])
        expected='PROXY_PASS' if all(all(x['proxy_gates'].values()) for x in r['summary'] if x['mode']=='own_norm') and all(all(x['proxy_gates'].values()) for x in rd['summary']) else 'COMPLETED_PROXY_FAILURE'
        assert expected==r['decision'];stream.close();stream=None;check();guard()
        result=dict(schema='SOURCE_FUNCTION_COORDINATES_STORED_AUDIT_V1',source=extent(__file__),freeze=a.freeze,
            report=extent(a.report),terminal=extent(a.terminal),binding=r['binding'],
            all7_source_metrics_dual_maps_complete_spectra_and336_histories_checked=True,
            all6_times8_full_observable_packets_and192_metrics_checked=len(allobs)==192,
            all72_actual_source_states_heads_scores144_probability_rows_and6_transports_checked=True,
            max_numeric_relative=maxnumeric,max_scalar_delta=maxdelta,max_score_absolute=maxscore,
            before_after_hashes_exact=True,base_checks=base_checks,seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
            decision='INDEPENDENT_STORED_PASS',scientific_decision=expected,
            scope='Independent coefficients and quadratic history expansion, source-width partial sums for all observables/head. '
                'Reuses already audited C/qbar; no covariance refit, eigensolver refit, model/optimizer/native/GPU/T4 replay. SVD of skips only.')
        assert result['all6_times8_full_observable_packets_and192_metrics_checked']
        write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['report','terminal','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['report-sha','terminal-sha','freeze']:p.add_argument('--'+name,required=True)
    main(p.parse_args())
