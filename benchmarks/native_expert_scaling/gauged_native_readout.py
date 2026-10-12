"""Exact decoded-map gauge and original scalar-norm native F32 head export."""
import argparse,ctypes,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
from broad_exact_residual_correction import metrics,summarize
START=time.monotonic()


def prepare(a):
    import numpy as np,psutil
    report=DOC/'broad_exact_residual_correction_result_20261012.json';audit=DOC/'broad_exact_residual_stored_adjudication_v2_20261012.json'
    r=json.loads(report.read_bytes());ad=json.loads(audit.read_bytes());assert ad['report']==extent(report) and ad['decision']=='INDEPENDENT_STORED_PASS'
    parent=json.loads(Path(r['binding']['path']).read_bytes());prep_path=DOC/'native_scalar_readout_preparation_20261012.json';prep=json.loads(prep_path.read_bytes());assert prep['compile_exit_code']==0
    assert extent(prep['engine']['path'])==prep['engine']
    paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',B/'broad_exact_residual_correction.py',report,audit,Path(r['binding']['path']),prep_path,
        DOC/'GAUGED_NATIVE_READOUT_PROTOCOL_20261012.md',Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),Path(np._core._multiarray_umath.__file__),
        Path(np.linalg._umath_linalg.__file__),Path(psutil.__file__),Path(parent['source'])/'model.safetensors',Path(parent['source'])/'config.json',
        Path(r['selected']['parameters']['path']),Path(r['selected']['file']['path']),Path(r['DEV_selected']['file']['path']),Path(r['FIT_dataset']['path']),Path(r['DEV_dataset']['path'])]
    paths += [Path(prep[k]['path']) for k in ['source','dll','log','engine','compiler','source_tool']]
    paths+=list((SITE/'numpy.libs').glob('*.dll'))+list((SITE/'psutil').glob('*.pyd'))
    b=dict(schema='GAUGED_NATIVE_READOUT_BINDING_V1',python=str(Path(sys.executable).resolve()),parent=r,source=parent['source'],native=prep,
        Ds=2048,d=256,V=65537,epsilon_source=parent['epsilon'],batch=64,quality=parent['quality'],
        numeric=dict(orthogonal_max=1e-12,gauge_vector_relative=1e-10,gauge_radius_relative=1e-10,gauge_score_sup_relative=1e-10,gauge_source_KL_delta=1e-9,gauge_argmax_changes=0,
            native_norm_relative=2e-5,native_inverse_relative=2e-5,native_score_absolute=1e-4,native_to_factor_mean_KL=1e-6,native_argmax_changes=0),
        limits=dict(seconds=900,OS_bytes=6<<30,output_bytes=3<<30),inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in paths)],
        scope='Actual selected fully audited source-factor predictor,all8808 source labels. Exact-real QR gauge/scalar epsilon/normalized bias and F32 original-engine head kernels. '
            'Teacher-injected codes,not causal source acquisition/whole engine/chatbot or useful-n/same-artifact50 admission. No source model/GPU/T4/RESERVED/longalign.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)


def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    assert not a.out.exists() and not a.directory.exists();r=b['parent'];proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None;mapping=None;stage='before_hash'
    receipt=dict(schema='GAUGED_NATIVE_READOUT_TERMINAL_V1',binding=extent(a.binding),freeze=a.freeze,command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),timing_control_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
        if a.directory.exists():assert sum(x.stat().st_size for x in a.directory.iterdir())<=b['limits']['output_bytes']
    def check():
        for x in b['inputs']:guard();assert extent(x['path'])==x,x['path']
    def load(x):
        with np.load(x['path'],allow_pickle=False) as p:return {k:p[k].copy() for k in p.files}
    def save(name,**fields):
        guard();p=a.directory/(name+'.npz');np.savez(p,**fields);return extent(p)
    def rawfile(name,x):
        p=a.directory/name;np.asarray(x).tofile(p);return extent(p)
    try:
        check();a.directory.mkdir();stage='gauge';params=load(r['selected']['parameters']);V,W,mu=params['V'],params['W'],params['mu'];Q,R=np.linalg.qr(V,mode='reduced')
        sign=np.where(np.diag(R)<0,-1.,1.);Q*=sign;R*=sign[:,None];E=W@R.T;c=mu@Q;perp=mu-c@Q.T;bias_code=c-mu@E
        epsilon=(float(perp@perp)+2048*b['epsilon_source'])/256;eps32=np.float32(epsilon)
        orth=float(np.max(np.abs(Q.T@Q-np.eye(256))));reconstruct=float(np.linalg.norm(Q@R-V)/np.linalg.norm(V));perpdef=float(np.linalg.norm(Q.T@perp)/max(1.,float(np.linalg.norm(perp))))
        gaugefile=save('gauge',Q=Q,R=R,E=E,code_bias=bias_code,center_in_code=c,center_perpendicular=perp,epsilon=np.array(epsilon),epsilon_f32=np.array(eps32))
        get,stream=tensor_reader(Path(b['source'])/'model.safetensors',np);cfg=json.loads((Path(b['source'])/'config.json').read_bytes());gamma=get('model.final_layernorm.weight');A=np.empty((65537,2048),'f8');C=np.empty((65537,256),'f8');beta=np.empty(65537,'f8')
        for lo in range(0,65537,1024):
            guard();hi=min(65537,lo+1024);A[lo:hi]=get('lm_head.weight',list(range(lo,hi)))*gamma*cfg['lm_head_multiplier'];C[lo:hi]=math.sqrt(8)*(A[lo:hi]@Q);beta[lo:hi]=math.sqrt(8)*(A[lo:hi]@perp)
        stream.close();stream=None;C32=np.ascontiguousarray(C,dtype='<f4');beta32=np.ascontiguousarray(beta,dtype='<f4')
        head64=save('head_F64',head=C,bias=beta);head32=rawfile('head.f32',C32);bias32=rawfile('bias.f32',beta32);epsilon32=rawfile('epsilon.f32',np.array([eps32],'<f4'))
        stage='native_ABI';dll=ctypes.CDLL(b['native']['dll']['path']);abi=dll.scalar_readout_abi;abi.argtypes=[ctypes.c_int];abi.restype=ctypes.c_int
        assert abi(0)==256 and abi(1)==65537
        fn=dll.scalar_readout;fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_float,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p];fn.restype=ctypes.c_int
        splits=[];native_seconds=0.;calls=0
        for split,dataset,point,meta in [('FIT',r['FIT_dataset'],r['selected'],r['FIT_cases']),('DEV',r['DEV_dataset'],r['DEV_selected'],r['DEV_cases'])]:
            guard();stage=split;data=load(dataset);H=data['H'];old=load(point['file']);Z=old['z'];Y=mu+Z@V.T;t64=Z@R.T+c;alternative=H@E+bias_code
            vector_rel=float(np.linalg.norm((t64@Q.T+perp)-Y)/np.linalg.norm(Y));code_rel=float(np.linalg.norm(t64-alternative)/max(float(np.linalg.norm(t64)),1e-300))
            radius=np.sqrt(np.mean(t64*t64,axis=1)+epsilon);source_radius=np.sqrt(np.mean(Y*Y,axis=1)+b['epsilon_source']);radius_rel=float(np.max(np.abs(radius/math.sqrt(8)-source_radius)/source_radius))
            t32=np.ascontiguousarray(t64,dtype='<f4');normalized=np.empty_like(t32);inverse=np.empty(len(H),'<f4');nativevalues=np.empty((len(H),4),'f8');gaugevalues=np.empty_like(nativevalues);preservation=np.empty((len(H),4),'f8')
            outpath=a.directory/(split+'_scores.f32');mapping=np.memmap(outpath,mode='w+',dtype='<f4',shape=(len(H),65537));batches=[]
            for lo in range(0,len(H),64):
                guard();hi=min(lo+64,len(H));n=hi-lo;out=np.empty((n,65537),'<f4');t0=time.monotonic()
                rc=fn(C32.ctypes.data,beta32.ctypes.data,t32[lo:hi].ctypes.data,n,float(eps32),normalized[lo:hi].ctypes.data,inverse[lo:hi].ctypes.data,out.ctypes.data)
                elapsed=time.monotonic()-t0;native_seconds+=elapsed;calls+=1;assert rc==0 and np.isfinite(out).all();mapping[lo:hi]=out
                original=(Z[lo:hi]@old['head'].T+old['bias'])/old['radius'][lo:hi,None];gauged=(t64[lo:hi]@C.T+beta)/radius[lo:hi,None]
                ref=(H[lo:hi]/np.sqrt(np.mean(H[lo:hi]**2,axis=1,keepdims=True)+b['epsilon_source']))@A.T
                nv=out.astype('f8');nativevalues[lo:hi]=metrics(nv,ref,np);gaugevalues[lo:hi]=metrics(gauged,ref,np);preservation[lo:hi]=metrics(nv,original,np)
                norm_ref=t64[lo:hi]/radius[lo:hi,None];nerr=float(np.linalg.norm(normalized[lo:hi].astype('f8')-norm_ref)/max(float(np.linalg.norm(norm_ref)),1e-300));ierr=float(np.max(np.abs(inverse[lo:hi].astype('f8')*radius[lo:hi]-1)))
                batches.append(dict(lo=lo,hi=hi,native_seconds=elapsed,gauge_score_sup_relative=float(np.max(np.abs(gauged-original))/max(1.,float(np.max(np.abs(original))))),
                    gauge_source_KL_delta=float(np.max(np.abs(gaugevalues[lo:hi,0]-old['metrics'][lo:hi,0]))),gauge_argmax_changes=int(np.count_nonzero(np.argmax(gauged,axis=1)!=np.argmax(original,axis=1))),
                    native_norm_relative=nerr,native_inverse_relative=ierr,native_score_absolute=float(np.max(np.abs(nv-original))),native_argmax_changes=int(np.count_nonzero(np.argmax(nv,axis=1)!=np.argmax(original,axis=1)))))
            mapping.flush();mapping._mmap.close();mapping=None
            packet=save(split+'_codes',t=t64,t_f32=t32,normalized_f32=normalized,inverse_f32=inverse,native_metrics=nativevalues,gauge_metrics=gaugevalues,preservation_metrics=preservation)
            summary=summarize(nativevalues,meta,b['quality']);meanpres=float(np.mean(preservation[:,0]));limits=b['numeric']
            numeric=dict(orthogonal_max=orth<=limits['orthogonal_max'],gauge_vector_relative=max(vector_rel,code_rel)<=limits['gauge_vector_relative'],gauge_radius_relative=radius_rel<=limits['gauge_radius_relative'],
                gauge_score_sup_relative=max(x['gauge_score_sup_relative'] for x in batches)<=limits['gauge_score_sup_relative'],gauge_source_KL_delta=max(x['gauge_source_KL_delta'] for x in batches)<=limits['gauge_source_KL_delta'],
                gauge_argmax_changes=sum(x['gauge_argmax_changes'] for x in batches)<=limits['gauge_argmax_changes'],native_norm_relative=max(x['native_norm_relative'] for x in batches)<=limits['native_norm_relative'],
                native_inverse_relative=max(x['native_inverse_relative'] for x in batches)<=limits['native_inverse_relative'],native_score_absolute=max(x['native_score_absolute'] for x in batches)<=limits['native_score_absolute'],
                native_to_factor_mean_KL=meanpres<=limits['native_to_factor_mean_KL'],native_argmax_changes=sum(x['native_argmax_changes'] for x in batches)<=limits['native_argmax_changes'])
            splits.append(dict(split=split,labels=len(H),codes=packet,scores=extent(outpath),decoded_vector_relative=vector_rel,code_relation_relative=code_rel,radius_relative=radius_rel,
                preservation_mean_KL=meanpres,batches=batches,numeric_gates=numeric,quality=summary))
            print(json.dumps(dict(split=split,numeric_gates=numeric,source_quality=summary['summary'],seconds=time.monotonic()-START)),flush=True)
        stage='after_hash';check();guard();outputs=[extent(p) for p in sorted(a.directory.iterdir())]
        result=dict(schema='GAUGED_NATIVE_READOUT_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,gauge=gaugefile,head_F64=head64,head_F32=head32,bias_F32=bias32,epsilon_F32=epsilon32,
            orthogonal_max=orth,QR_reconstruction_relative=reconstruct,perpendicular_defect_relative=perpdef,epsilon=epsilon,epsilon_f32=float(eps32),source_reencoding_identity_defect=float(np.max(np.abs(E.T@Q-np.eye(256)))),
            splits=splits,native_calls=calls,native_component_call_seconds=native_seconds,component_timing_admitted=False,outputs=outputs,inputs_before_after_exact=True,
            seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,scope=b['scope'],decision='GAUGE_AND_NATIVE_PRECISION_PASS' if all(all(s['numeric_gates'].values()) for s in splits) else 'COMPLETED_GAUGE_OR_NATIVE_PRECISION_FAILURE')
        write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc(),stage=stage);raise
    finally:
        if stream is not None:stream.close()
        if mapping is not None:mapping.flush();mapping._mmap.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.prepare:prepare(a)
    else:assert all((a.binding,a.binding_sha,a.freeze,a.directory));run(a)
