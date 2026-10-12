"""FIT-only per-boundary angular PCA; DEV reconstruction/radius/transport.

Residual-state coordinates, not recurrent-channel pruning or a chatbot fit.
"""
import argparse,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write
START=time.monotonic()


def prepare(a):
    import numpy as np,psutil
    source=DOC/'original_history_boundaries_finish_result_20261009.json'
    audit=DOC/'original_history_boundaries_stored_adjudication_20261009.json'
    r=json.loads(source.read_bytes());ad=json.loads(audit.read_bytes())
    assert extent(source)['sha256']==ad['result_sha256'] and ad['decision']=='STORED_BOUNDARIES_PROJECTION_PASS'
    assert len(r['records'])==48 and r['boundaries']==[0,4,8,12,16,20,24]
    config=ROOT/'results/native_expert_scaling/falcon_1p5b_source_repair1_20261008/config.json'
    assert json.loads(config.read_bytes())['rms_norm_eps']==1e-5
    paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',Path(sys.executable),source,audit,config,
        Path(r['basis']['path']),DOC/'SOURCE_RESIDUAL_COORDINATE_FEASIBILITY_PROTOCOL_20261012.md',
        Path(np.__file__),Path(np._core._multiarray_umath.__file__),Path(np.linalg._umath_linalg.__file__),Path(psutil.__file__),
        Path(sys.executable).parent/'python312.dll']
    paths+=list((SITE/'numpy.libs').glob('*.dll'))+list((SITE/'psutil').glob('*.pyd'))
    for record in r['records']:
        for field in record['boundaries']:paths.extend((Path(field['h']['path']),Path(field['z']['path'])))
    inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in paths)]
    b=dict(schema='SOURCE_RESIDUAL_COORDINATE_FEASIBILITY_BINDING_V1',python=str(Path(sys.executable).resolve()),
        source_report=extent(source),source_prior_audit=extent(audit),records=r['records'],basis=r['basis'],
        source_D=2048,target_D=256,epsilon=1e-5,boundaries=[0,4,8,12,16,20,24],
        fit='Equal24 FIT cases;uncentered covariance of per-token RMS-normalized RAW source residuals,top256.',
        numeric=dict(orthogonality=1e-10,eigen_residual=1e-10,trace_relative=1e-10,F32_basis_metric_relative=1e-5),
        proxy=dict(DEV_angular_relative_RMS_mean=.10,DEV_angular_relative_RMS_worst=.20,
                   DEV_centered_relative_RMS_mean=.10),
        limits=dict(seconds=900,OS_bytes=4<<30,output_bytes=512<<20),inputs=inputs,
        scope='New residual-basis variable;no source/shadow/generator/optimizer/native/GPU/T4 call. '
            'Spectral/reconstruction proxy is not conditional-function cost, preserved task information or whole-model admission.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),inputs=len(inputs),seconds=time.monotonic()-START)),flush=True)


def run(a):
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes())
    assert b['schema']=='SOURCE_RESIDUAL_COORDINATE_FEASIBILITY_BINDING_V1'
    assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    assert not a.out.exists() and not a.directory.exists() and not a.out.with_suffix('.terminal.json').exists()
    import numpy as np,psutil
    assert (np.__version__,psutil.__version__)==('2.4.6','7.2.2')
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));stage='before_hash';rows=[];bases=[]
    receipt=dict(schema='SOURCE_RESIDUAL_COORDINATE_FEASIBILITY_TERMINAL_V1',binding=extent(a.binding),
        freeze=a.freeze,command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),timing_control_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'],'deadline'
        assert proc.memory_info().peak_wset<=b['limits']['OS_bytes'],'OS cap'
        assert not proc.children(recursive=True),'unexpected child'
    def check():
        for item in b['inputs']:guard();assert extent(item['path'])==item,item['path']
    def raw(record,boundary):
        item=next(x for x in record['boundaries'] if x['boundary']==boundary)['h']
        z=(np.fromfile(item['path'],'<u2').astype('<u4')<<16).view('<f4').reshape(record['history'],2048).astype('f8')
        assert np.isfinite(z).all();return z
    def angular(x):return x/np.sqrt(np.mean(x*x,axis=1,keepdims=True)+b['epsilon'])
    def measure(x,basis):
        u=angular(x);projection=(u@basis)@basis.T
        error=float(np.linalg.norm(u-projection)/max(float(np.linalg.norm(u)),1e-300))
        centered=u-u.mean(0);cp=projection-projection.mean(0)
        ce=float(np.linalg.norm(centered-cp)/max(float(np.linalg.norm(centered)),1e-300))
        xp=(x@basis)@basis.T
        full=np.sqrt(np.mean(x*x,axis=1)+b['epsilon']);restricted=np.sqrt(np.mean(xp*xp,axis=1)+b['epsilon'])
        return dict(angular_relative_RMS=error,centered_relative_RMS=ce,
            angular_energy_fraction=float(np.sum(projection*projection)/max(float(np.sum(u*u)),1e-300)),
            true_to_restricted_RMS_ratio_mean=float(np.mean(full/restricted)),
            true_to_restricted_RMS_ratio_worst=float(np.max(full/restricted)))
    try:
        check();a.directory.mkdir();old=np.fromfile(b['basis']['path'],'<f4').reshape(2048,256).astype('f8')
        fit=[r for r in b['records'] if r['split']=='FIT'];assert len(fit)==24
        identity=np.eye(2048);rankeye=np.eye(256)
        for boundary in b['boundaries']:
            guard();stage=f'FIT_basis{boundary}'
            gram=np.zeros((2048,2048),dtype='f8');mean=np.zeros(2048,dtype='f8')
            for record in fit:
                guard();u=angular(raw(record,boundary));gram+=(u.T@u)/(24*len(u));mean+=u.mean(0)/24
            eigen,Q=np.linalg.eigh(gram);order=np.arange(2047,-1,-1);eigen=eigen[order];Q=Q[:,order]
            signs=np.where(Q[np.argmax(np.abs(Q),axis=0),np.arange(2048)]<0,-1.,1.);Q*=signs
            basis=Q[:,:256].copy();orth=float(np.max(np.abs(Q.T@Q-identity)))
            resid=float(np.linalg.norm(gram@Q-Q*eigen)/max(float(np.linalg.norm(gram)),1e-300))
            trace=float(abs(np.trace(gram)-eigen.sum())/max(float(np.trace(gram)),1e-300))
            assert orth<=b['numeric']['orthogonality'] and resid<=b['numeric']['eigen_residual'] and trace<=b['numeric']['trace_relative']
            assert eigen[-1]>=-1e-10*max(float(eigen[0]),1)
            path=a.directory/f'boundary{boundary:02d}.basis.npz'
            np.savez(path,gram=gram,eigenvalues=eigen,eigenvectors=Q,basis=basis,basis_f32=basis.astype('<f4'),mean_angular=mean)
            bases.append(dict(boundary=boundary,file=extent(path),orthogonality_max=orth,eigen_residual_relative=resid,
                trace_relative=trace,FIT_spectral_retained_fraction=float(eigen[:256].sum()/eigen.sum())))
            for record in b['records']:
                guard();x=raw(record,boundary);new=measure(x,basis);oldvalue=measure(x,old)
                roundtrip=measure(x,basis.astype('f4').astype('f8'))
                rounding=max(abs(roundtrip[key]-new[key]) for key in ('angular_relative_RMS','centered_relative_RMS','angular_energy_fraction'))
                assert rounding<=b['numeric']['F32_basis_metric_relative']
                rows.append(dict(id=record['id'],split=record['split'],domain=record['domain'],boundary=boundary,
                    positions=record['history'],new=new,old=oldvalue,F32_basis_metric_delta=rounding))
            print(json.dumps(dict(boundary=boundary,FIT_retention=bases[-1]['FIT_spectral_retained_fraction'],seconds=time.monotonic()-START)),flush=True)
            del gram,Q,eigen,basis,u,x
        stage='residual_transport';transport=[]
        for first,second in zip(bases,bases[1:]):
            with np.load(first['file']['path'],allow_pickle=False) as x,np.load(second['file']['path'],allow_pickle=False) as y:
                T=y['basis'].T@x['basis'];singular=np.linalg.svd(T,compute_uv=False)
                path=a.directory/f"transport{first['boundary']:02d}_{second['boundary']:02d}.f64";T.astype('<f8').tofile(path)
                transport.append(dict(source_boundary=first['boundary'],destination_boundary=second['boundary'],
                    file=extent(path),mean_squared_singular=float(np.mean(singular*singular)),
                    min_singular=float(singular[-1]),max_singular=float(singular[0]),
                    identity_difference_relative=float(np.linalg.norm(T-rankeye)/math.sqrt(256))))
        summary=[]
        for boundary in b['boundaries']:
            dev=[r for r in rows if r['split']=='DEV' and r['boundary']==boundary];assert len(dev)==24
            values=dict(boundary=boundary,cases=24,
                DEV_angular_relative_RMS_mean=math.fsum(r['new']['angular_relative_RMS'] for r in dev)/24,
                DEV_angular_relative_RMS_worst=max(r['new']['angular_relative_RMS'] for r in dev),
                DEV_centered_relative_RMS_mean=math.fsum(r['new']['centered_relative_RMS'] for r in dev)/24,
                old_DEV_angular_relative_RMS_mean=math.fsum(r['old']['angular_relative_RMS'] for r in dev)/24,
                DEV_true_to_restricted_RMS_ratio_mean=math.fsum(r['new']['true_to_restricted_RMS_ratio_mean'] for r in dev)/24)
            values['proxy_gates']={key:values[key]<=threshold for key,threshold in b['proxy'].items()};summary.append(values)
        assert len(rows)==336 and len(bases)==7 and len(transport)==6
        stage='after_hash';check();guard();outputs=[extent(p) for p in sorted(a.directory.iterdir())]
        assert sum(r['bytes'] for r in outputs)<=b['limits']['output_bytes']
        result=dict(schema='SOURCE_RESIDUAL_COORDINATE_FEASIBILITY_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,
            bases=bases,rows=rows,summary=summary,transport=transport,outputs=outputs,
            inputs_before_after_exact=True,seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
            decision='PROXY_ALL_BOUNDARIES_PASS' if all(all(r['proxy_gates'].values()) for r in summary) else 'COMPLETED_PROXY_FAILURE',
            source_shadow_model_generator_optimizer_native_GPU_T4_calls=0,
            scope=b['scope'])
        write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc(),stage=stage,completed_boundaries=bases);raise
    finally:
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze')
    p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.prepare:prepare(a)
    else:assert all((a.binding,a.binding_sha,a.freeze,a.directory));run(a)
