"""Independent QR input witness and full window-convolution/field consistency audit."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic();CLOCK=time.perf_counter()


def main(a):
    import numpy as np,psutil
    re=extent(a.report);te=extent(a.terminal);assert re['sha256']==a.report_sha and te['sha256']==a.terminal_sha;r=json.loads(a.report.read_bytes());t=json.loads(a.terminal.read_bytes());assert t['exit_code']==0 and t['result']==re and extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil'] and not a.out.exists();proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None;stage='startup';rows=[];maxima={};numeric=[];counts=dict(inverse=0,H_unrounded=0,H_candidate=0,predicted_fields=0)
    receipt=dict(schema='GATE_INPUT_RECOVERY_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False);sources=[extent(Path(__file__)),extent(B/'falcon_auxiliary_s6_transport.py'),extent(DOC/'GATE_INPUT_RECOVERY_AUDIT_PROTOCOL_20261013.md')]
    def guard():assert time.monotonic()-START<=b['limits']['audit_seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
    def check():
        for e in b['inputs']+sources+[r['binding'],re,te]+[q['file'] for q in r['recovery_maps']]+[q['file'] for q in r['records']]:guard();assert extent(e['path'])==e,e['path']
    def load(e):
        with np.load(e['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def dec(q):assert q.dtype==np.dtype('<u2');return (q.astype('<u4')<<16).view('<f4').astype('f8')
    def bf(q):
        f=np.asarray(q,dtype='<f4');assert np.isfinite(f).all();u=f.view('<u4');return ((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype('<u2')
    def raw(e):assert e['dtype']=='BF16';return dec(np.fromfile(e['path'],'<u2').reshape(e['shape']))
    def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    def vector(x,y,tol,key):
        assert np.isfinite(x).all() and np.isfinite(y).all();d=rel(x,y);maxima[key]=max(maxima.get(key,0.),d);numeric.append(dict(case=stage,control=key,relative=d,threshold=tol,passed=bool(d<=tol)))
    def scalar(x,y):
        d=float(abs(x-y)/max(abs(y),1.));maxima['scalar']=max(maxima.get('scalar',0.),d);assert d<=b['numeric']['audit_scalar_relative'],(stage,x,y)
    def sigmoid(z):
        out=np.empty_like(z);positive=z>=0;out[positive]=1/(1+np.exp(-z[positive]));e=np.exp(z[~positive]);out[~positive]=e/(1+e);return out
    def metrics(x,y,pos):return dict(relative_RMS=rel(x,y),centered_relative_RMS=rel(x-x.mean(0),y-y.mean(0)),label_relative_RMS=rel(x[pos],y[pos]))
    try:
        check();get,stream=tensor_reader(b['source_weights'],np);cfg=json.loads(Path(b['source_config']).read_bytes());mu=dec(np.array(b['semantics']['MuP_bits'],'u2'));expected_mu=bf(np.repeat(np.array(cfg['ssm_multipliers']),[3072,3072,256,256,48]));assert np.array_equal(expected_mu,np.array(b['semantics']['MuP_bits'],'u2')) and b['semantics']['MuP_dtype']=='BF16' and b['semantics']['CPU_primitive_controls_bit_exact'] and not b['semantics']['GPU_initialized'];capture=json.loads(Path(b['capture']['path']).read_bytes())['records'];cases={q['id']:q for q in capture};assert len(r['records'])==144 and {(q['site'],q['id']) for q in r['records']}=={(s,i) for s in b['sites'] for i in cases};maps={q['site']:q for q in r['recovery_maps']}
        for site in b['sites']:
            stage=f'map/{site}';guard();prefix=f'model.layers.{site}.mamba.';W=get(prefix+'in_proj.weight');G=W[:3072];cw=get(prefix+'conv1d.weight')[:,0];cb=get(prefix+'conv1d.bias');db=get(prefix+'dt_bias');Q,R=np.linalg.qr(G,mode='reduced');Pqr=np.linalg.solve(R,Q.T);saved=load(maps[site]['file']);assert set(saved)=={'P'};P=saved['P'];assert P.shape==(2048,3072) and P.dtype==np.dtype('<f8');vector(P,Pqr,b['numeric']['audit_inverse_relative'],'inverse_rectangular_QR');identity=float(np.max(np.abs(P@G-np.eye(2048))));assert identity<=b['numeric']['inverse_left_identity_max'];scalar(maps[site]['left_identity_max'],identity);counts['inverse']+=P.size
            for item in [q for q in r['records'] if q['site']==site]:
                stage=f'{site}/{item["id"]}';guard();rec=cases[item['id']];n=rec['history'];fields=rec['sites'][site]['fields'];gate=raw(fields['gate']);z=load(item['file']);assert set(z)=={'H_unrounded','H_candidate_BF16',*[f'predicted_{k}' for k in b['fields']]} and (item['split'],item['domain'],item['history'])==(rec['split'],rec['domain'],n);H=z['H_unrounded'];bits=z['H_candidate_BF16'];assert H.shape==bits.shape==(n,2048) and H.dtype==np.dtype('<f8') and np.array_equal(bits,bf(H));candidate=dec(bits);Hqr=(gate/mu[0])@Pqr.T;vector(H,Hqr,b['numeric']['audit_H_relative'],'H_rectangular_QR');counts['H_unrounded']+=H.size;counts['H_candidate']+=bits.size;res=H@G.T-gate/mu[0];normal=float(np.linalg.norm(res@G)/max(np.linalg.norm(gate/mu[0])*np.linalg.norm(G),1e-30));scalar(item['normal_equation_residual_relative'],normal);scalar(item['BF16_candidate_vs_unrounded_relative'],rel(candidate,H))
                # Independent full contraction,row blocks;all fields use rounded candidate.
                projected=np.empty((n,6704))
                for lo in range(0,6704,256):guard();hi=min(lo+256,6704);projected[:,lo:hi]=dec(bf(np.dot(candidate,W[lo:hi].T)))
                scaled=dec(bf(projected*mu));bc=scaled[:,3072:6656];padded=np.pad(bc,((3,0),(0,0)));conv=np.empty_like(bc)
                for lo in range(0,n,128):
                    guard();hi=min(lo+128,n);window=np.stack([padded[lo+tap:hi+tap] for tap in range(4)],axis=2);conv[lo:hi]=np.einsum('tck,ck->tc',window,cw,optimize=True)+cb
                activated_input=dec(bf(conv));activated=dec(bf(activated_input*sigmoid(activated_input)));d=dec(bf(scaled[:,6656:]+db));sp=np.maximum(d,0)+np.log1p(np.exp(-np.abs(d)));sp[d>20]=d[d>20];sp=dec(bf(sp));delta=np.maximum(sp,b['semantics']['delta_min'])
                if b['semantics']['delta_max'] is not None:delta=np.minimum(delta,b['semantics']['delta_max'])
                fresh=dict(gate=bf(scaled[:,:3072]),x=bf(activated[:,:3072]),B=bf(activated[:,3072:3328]),C=bf(activated[:,3328:]),delta=bf(delta));mm={};flags={}
                for key in b['fields']:
                    actual=z['predicted_'+key];assert actual.dtype==np.dtype('<u2') and actual.shape==tuple(fields[key]['shape']);vector(dec(actual),dec(fresh[key]),b['numeric']['audit_vector_relative'],'replay_'+key);counts['predicted_fields']+=actual.size;expected=raw(fields[key]);mm[key]=metrics(dec(actual),expected,rec['positions']);flags[key]=mm[key]['relative_RMS']<=b['consistency']['each_case_each_field_relative_RMS'];assert item['consistency_flags'][key]==flags[key]
                    for metric,value in mm[key].items():scalar(item['metrics'][key][metric],value)
                rows.append(dict(id=rec['id'],site=site,split=rec['split'],domain=rec['domain'],metrics=mm,flags=flags));print(json.dumps(dict(stage=stage,complete=len(rows),seconds=time.perf_counter()-CLOCK)),flush=True)
        for site in b['sites']:
            for split in ['FIT','DEV']:
                rr=[q for q in rows if q['site']==site and q['split']==split];assert len(rr)==24
                for key in b['fields']:
                    vals=[q['metrics'][key]['relative_RMS'] for q in rr];actual=r['aggregates'][str(site)][split][key];scalar(actual['mean'],float(np.mean(vals)));scalar(actual['worst'],max(vals));dm={d:float(np.mean([q['metrics'][key]['relative_RMS'] for q in rr if q['domain']==d])) for d in {q['domain'] for q in rr}};assert set(dm)==set(actual['domain_means'])
                    for domain,value in dm.items():scalar(actual['domain_means'][domain],value)
                    scalar(actual['worst_domain_mean'],max(dm.values()));assert actual['failed_cases']==sum(not q['flags'][key] for q in rr)
        independent=all(all(q['flags'][k] for k in ['x','B','C','delta']) for q in rows);success=all(all(q['flags'].values()) for q in rows);assert r['calibration_consistency_all_independent_fields']==independent and r['calibration_consistency_all_fields']==success and r['approximate_module_inputs']==144 and all(r[k]==0 for k in ['source_forwards','native_calls','optimizer_updates','true_hidden_bits_recovered']);stream.close();stream=None;check();guard();ad=dict(schema='GATE_INPUT_RECOVERY_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,report=re,terminal=te,binding=r['binding'],case_sites=144,counts=counts,maximum=maxima,numeric_flags=numeric,before_after_hashes_exact=True,consistency_all_independent_fields=independent,consistency_all_fields=success,decision='INDEPENDENT_STORED_PASS' if all(q['passed'] for q in numeric) else 'INDEPENDENT_STORED_NUMERIC_FAILURE',seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope='Independent rectangular QR recovery,all H/candidate bits/shifted-window rounded convolution/activation/delta/predicted source fields/all metrics/flags/hash. Fidelity failure distinct from reproduction;no actual input bits/CUDA/source-model/causal/quality/50 admission.')
        write(a.out,ad);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(decision=ad['decision'],consistent=success,maximum=maxima,counts=counts,seconds=ad['seconds'])),flush=True)
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
