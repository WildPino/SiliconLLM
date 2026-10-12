"""Independent block Gram/direct rectangular source SVD for all gate operators."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic();CLOCK=time.perf_counter()


def main(a):
    import numpy as np,psutil
    re=extent(a.report);te=extent(a.terminal);assert re['sha256']==a.report_sha and te['sha256']==a.terminal_sha;r=json.loads(a.report.read_bytes());t=json.loads(a.terminal.read_bytes());assert t['exit_code']==0 and t['result']==re and extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil'] and not a.out.exists();proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None;stage='startup';maximum={};rows=[];words=0;receipt=dict(schema='GATE_INPUT_OPERATOR_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False);sources=[extent(Path(__file__)),extent(B/'falcon_auxiliary_s6_transport.py'),extent(DOC/'GATE_INPUT_OPERATOR_AUDIT_PROTOCOL_20261013.md')]
    def guard():assert time.monotonic()-START<=b['limits']['audit_seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
    def check():
        for e in b['inputs']+sources+[r['binding'],re,te]+[q['file'] for q in r['records']]:guard();assert extent(e['path'])==e,e['path']
    def close(x,y,tol,key):
        err=float(abs(x-y)/max(abs(y),1.));maximum[key]=max(maximum.get(key,0.),err);assert err<=tol,(stage,key,x,y)
    try:
        check();get,stream=tensor_reader(b['source_weights'],np);cfg=json.loads(Path(b['source_config']).read_bytes());assert cfg['ssm_multipliers'][0]==r['gate_multiplier_config']!=0 and cfg['ssm_in_multiplier']==r['input_multiplier_config']!=0;assert len(r['records'])==3 and {q['site'] for q in r['records']}==set(b['sites'])
        for q in r['records']:
            stage=str(q['site']);guard();G=get(q['source_key'])[:3072].copy();assert G.shape==(3072,2048);gram=np.zeros((2048,2048))
            for lo in range(0,3072,256):guard();block=G[lo:lo+256];gram+=np.einsum('ri,rj->ij',block,block,optimize=True)
            with np.load(q['file']['path'],allow_pickle=False) as z:sg=z['Gram'].copy();U=z['eigenvectors'].copy();ev=z['eigenvalues'].copy();sv=z['singular_values_ascending'].copy();assert len(z.files)==4
            err=float(np.linalg.norm(sg-gram)/np.linalg.norm(gram));close(err,0,b['numeric']['Gram_relative'],'Gram_relative');assert sg.shape==U.shape==(2048,2048) and ev.shape==sv.shape==(2048,) and np.all(np.diff(ev)>=0);orth=float(np.max(np.abs(U.T@U-np.eye(2048))));res=float(np.linalg.norm(gram@U-U*ev)/np.linalg.norm(gram));negative=float(max(0.,-ev[0])/max(ev[-1],1e-300));assert orth<=b['numeric']['eigenvector_orthogonal_max'] and res<=b['numeric']['eigen_residual_relative'] and negative<=b['numeric']['negative_eigenvalue_relative'];close(q['eigenvector_orthogonal_max'],orth,1e-10,'metadata');close(q['eigen_residual_relative'],res,1e-10,'metadata');close(q['negative_eigenvalue_relative'],negative,1e-12,'metadata');assert np.array_equal(sv,np.sqrt(np.maximum(ev,0)))
            direct=np.linalg.svd(G,compute_uv=False)[::-1];serr=float(np.max(np.abs(direct-sv))/direct[-1]);maximum['rectangular_SVD_sigma_max_relative']=max(maximum.get('rectangular_SVD_sigma_max_relative',0.),serr);assert serr<=b['numeric']['spectrum_sigma_max_relative'];rank=int(np.count_nonzero(direct>b['selection']['robust_sigma_relative']*direct[-1]));condition=float(direct[-1]/direct[0]) if direct[0]>0 else None;inverse=float(1/direct[0]) if direct[0]>0 else None;eligible=rank==2048 and condition is not None and condition<=b['selection']['condition_max'];assert q['robust_rank']==rank and q['recovery_preflight_eligible']==eligible;close(q['sigma_min'],float(direct[0]),1e-8,'sigma_min');close(q['sigma_max'],float(direct[-1]),1e-8,'sigma_max')
            if condition is not None:close(q['condition'],condition,1e-8,'condition');close(q['unscaled_inverse_gain'],inverse,1e-8,'inverse_gain')
            else:assert q['condition'] is None and q['unscaled_inverse_gain'] is None
            rows.append(dict(site=q['site'],robust_rank=rank,condition=condition,unscaled_inverse_gain=inverse,recovery_preflight_eligible=eligible,Gram_relative=err,rectangular_sigma_relative=serr));words+=sg.size+U.size+ev.size+sv.size;print(json.dumps(dict(site=q['site'],complete=len(rows),seconds=time.perf_counter()-CLOCK)),flush=True)
        stream.close();stream=None;check();guard();assert all(r[k]==0 for k in ['source_forwards','hidden_recoveries','optimizer_updates','native_calls']);ad=dict(schema='GATE_INPUT_OPERATOR_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,report=re,terminal=te,binding=r['binding'],sites=rows,all_matrix_spectrum_words_verified=words,maximum=maximum,before_after_hashes_exact=True,decision='INDEPENDENT_STORED_PASS',seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope='Complete blockwise Gram/eigenvectors/eigenvalues/direct rectangular source SVD all2048 singular values/site/classification/inverse gain/config/hash. Necessary unrounded information geometry only;no reconstructed source hidden/quantization/generator/causal quality/speed admission.')
        write(a.out,ad);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(ad),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),stage=stage,traceback=traceback.format_exc());raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['report','terminal','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['report-sha','terminal-sha','freeze']:p.add_argument('--'+name,required=True)
    main(p.parse_args())
