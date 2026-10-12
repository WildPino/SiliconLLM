"""Matrix-only input identifiability:exact source gate rows,not recovered states."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic();CLOCK=time.perf_counter()


def prepare(a):
    import numpy as np,psutil
    priorpath=DOC/'full_channel_native_binding_20261013.json';p=json.loads(priorpath.read_bytes());storagepath=DOC/'gate_input_operator_storage_20261013.json';st=json.loads(storagepath.read_text(encoding='utf-8-sig'));paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',DOC/'GATE_INPUT_OPERATOR_PROTOCOL_20261013.md',storagepath,priorpath,Path(p['parent']['source_weights']),Path(p['parent']['source_config']),SITE/'transformers/models/falcon_h1/modeling_falcon_h1.py']
    np_root=Path(np.__file__).resolve().parent;ps_root=Path(psutil.__file__).resolve().parent;paths += [Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),Path(np._core._multiarray_umath.__file__),Path(np.linalg._umath_linalg.__file__),Path(psutil.__file__)]+list((np_root.parent/'numpy.libs').glob('*.dll'))+list(ps_root.glob('*.pyd'))
    runtime=[e for e in p['inputs'] if e['path'].startswith(str(Path(sys.executable).parent)) or e['path'].startswith(str(SITE.resolve())) or e['path'].startswith('C:\\Windows\\System32')];inputs=list({e['path']:e for e in runtime+[extent(q) for q in paths]}.values())
    for e in inputs:assert extent(e['path'])==e
    b=dict(schema='GATE_INPUT_OPERATOR_BINDING_V1',python=p['python'],versions=p['versions'],source_weights=p['parent']['source_weights'],source_config=p['parent']['source_config'],sites=[0,12,23],rows=3072,columns=2048,scaled_matrix=False,numeric=dict(Gram_relative=1e-12,eigenvector_orthogonal_max=1e-10,eigen_residual_relative=1e-10,spectrum_sigma_max_relative=1e-8,negative_eigenvalue_relative=1e-10),selection=dict(robust_sigma_relative=1e-6,condition_max=1e6),limits=dict(seconds=900,audit_seconds=1200,OS_bytes=6<<30,output_bytes=512<<20),output_directory=st['new_output_directory'],inputs=inputs,scope='Exact BF16-promoted unscaled source in_proj gate row matrices3072x2048/all3 sites. Rank/condition/inverse noise-gain necessary geometry only;MuP nonzero scalar does not alter rank/condition. No reconstructed hidden/input/corpus fitting/source model/native/causal/quality/50 admission.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.perf_counter()-CLOCK)),flush=True)


def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and not a.directory.exists() and not a.out.exists() and str(a.directory.resolve())==str(Path(b['output_directory']).resolve());proc=psutil.Process();proc.cpu_affinity(list(range(6)));rows=[];stream=None;stage='startup';receipt=dict(schema='GATE_INPUT_OPERATOR_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir())<=b['limits']['output_bytes']
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    try:
        check();a.directory.mkdir();get,stream=tensor_reader(b['source_weights'],np);cfg=json.loads(Path(b['source_config']).read_bytes());gate_multiplier=cfg['ssm_multipliers'][0];assert gate_multiplier!=0 and cfg['ssm_in_multiplier']!=0 and cfg['hidden_size']==2048
        for site in b['sites']:
            stage=str(site);guard();G=get(f'model.layers.{site}.mamba.in_proj.weight')[:3072].copy();assert G.shape==(3072,2048) and np.isfinite(G).all();gram=G.T@G;ev,U=np.linalg.eigh((gram+gram.T)/2);orth=float(np.max(np.abs(U.T@U-np.eye(2048))));residual=float(np.linalg.norm(gram@U-U*ev)/np.linalg.norm(gram));negative=float(max(0.,-ev[0])/max(ev[-1],1e-300));assert orth<=b['numeric']['eigenvector_orthogonal_max'] and residual<=b['numeric']['eigen_residual_relative'] and negative<=b['numeric']['negative_eigenvalue_relative'];sigma=np.sqrt(np.maximum(ev,0));rank=int(np.count_nonzero(sigma>b['selection']['robust_sigma_relative']*sigma[-1]));condition=float(sigma[-1]/sigma[0]) if sigma[0]>0 else None;inverse_gain=float(1/sigma[0]) if sigma[0]>0 else None;eligible=rank==2048 and condition is not None and condition<=b['selection']['condition_max'];path=a.directory/f'site{site:02d}.gate_operator.npz';np.savez(path,Gram=gram,eigenvalues=ev,eigenvectors=U,singular_values_ascending=sigma);rows.append(dict(site=site,file=extent(path),source_key=f'model.layers.{site}.mamba.in_proj.weight',rows=3072,columns=2048,sigma_min=float(sigma[0]),sigma_max=float(sigma[-1]),condition=condition,unscaled_inverse_gain=inverse_gain,robust_rank=rank,eigenvector_orthogonal_max=orth,eigen_residual_relative=residual,negative_eigenvalue_relative=negative,recovery_preflight_eligible=eligible));print(json.dumps(dict(site=site,condition=condition,rank=rank,seconds=time.perf_counter()-CLOCK)),flush=True)
        stream.close();stream=None;check();guard();r=dict(schema='GATE_INPUT_OPERATOR_PREFLIGHT_V1',binding=extent(a.binding),freeze=a.freeze,records=rows,gate_multiplier_config=gate_multiplier,input_multiplier_config=cfg['ssm_in_multiplier'],before_after_hashes_exact=True,source_forwards=0,hidden_recoveries=0,optimizer_updates=0,native_calls=0,decision='MATRIX_PREFLIGHT_COMPLETE_PENDING_AUDIT',output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()),seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope=b['scope']);write(a.out,r);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(result=extent(a.out))),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),stage=stage,traceback=traceback.format_exc())
        if a.directory.exists():write(a.directory/'first_fault.json',dict(stage=stage,error=repr(e),records=rows))
        raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    for name in ['binding','directory','out']:p.add_argument('--'+name,type=Path)
    for name in ['binding-sha','freeze']:p.add_argument('--'+name)
    a=p.parse_args()
    if a.run:run(a)
    else:
        import psutil
        proc=psutil.Process();receipt=dict(schema='GATE_INPUT_OPERATOR_PREPARATION_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
        try:prepare(a);receipt.update(exit_code=0,error=None,result=extent(a.out))
        except BaseException as e:
            import traceback
            receipt.update(exit_code=1,error=repr(e),traceback=traceback.format_exc());raise
        finally:
            receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.preparation_terminal.json'),receipt)
