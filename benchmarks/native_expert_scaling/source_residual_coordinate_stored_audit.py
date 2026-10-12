"""Independent full covariance/eigensystem/quadratic reconstruction audit."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'benchmarks/native_expert_scaling'),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
START=time.monotonic()


def main(a):
    import numpy as np,psutil
    assert extent(a.report)['sha256']==a.report_sha and extent(a.terminal)['sha256']==a.terminal_sha
    r=json.loads(a.report.read_bytes());t=json.loads(a.terminal.read_bytes())
    assert t['exit_code']==0 and t['error'] is None and t['result']==extent(a.report)
    assert r['schema']=='SOURCE_RESIDUAL_COORDINATE_FEASIBILITY_RESULT_V1'
    assert extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes())
    proc=psutil.Process();proc.cpu_affinity(list(range(6)))
    inputs=[extent(__file__),extent(a.report),extent(a.terminal),r['binding'],*b['inputs'],*r['outputs']]
    def guard():
        assert time.monotonic()-START<=900 and proc.memory_info().peak_wset<=4<<30
        assert not proc.children(recursive=True)
    def check():
        for item in inputs:guard();assert extent(item['path'])==item,item['path']
    def load(rec,boundary):
        item=next(x for x in rec['boundaries'] if x['boundary']==boundary)['h']
        raw=(np.fromfile(item['path'],'<u2').astype('<u4')<<16).view('<f4')
        return raw.reshape(rec['history'],2048).astype('f8')
    def normalize(x):return x/np.sqrt(np.sum(x*x,axis=1,keepdims=True)/2048+b['epsilon'])
    def quadratic(x,R):
        u=normalize(x);codes=u@R;gram=R.T@R
        total=float(np.sum(u*u));cross=float(np.sum(codes*codes));reconstruction=float(np.sum((codes@gram)*codes))
        error=math.sqrt(max(0,total-2*cross+reconstruction)/max(total,1e-300))
        uc=u-u.mean(0);cc=codes-codes.mean(0)
        centered_total=float(np.sum(uc*uc));c_cross=float(np.sum(cc*cc));c_recon=float(np.sum((cc@gram)*cc))
        ce=math.sqrt(max(0,centered_total-2*c_cross+c_recon)/max(centered_total,1e-300))
        raw_codes=x@R;recon_energy=np.sum((raw_codes@gram)*raw_codes,axis=1)
        true=np.sqrt(np.sum(x*x,axis=1)/2048+b['epsilon']);restricted=np.sqrt(np.maximum(recon_energy,0)/2048+b['epsilon'])
        return dict(angular_relative_RMS=error,centered_relative_RMS=ce,angular_energy_fraction=reconstruction/max(total,1e-300),
            true_to_restricted_RMS_ratio_mean=float(np.mean(true/restricted)),true_to_restricted_RMS_ratio_worst=float(np.max(true/restricted)))
    check();old=np.fromfile(b['basis']['path'],'<f4').reshape(2048,256).astype('f8')
    fit=[x for x in b['records'] if x['split']=='FIT'];assert len(fit)==24
    lookup={(x['id'],x['boundary']):x for x in r['rows']};checked=[];case_rows=[];max_delta=0
    for basis in r['bases']:
        guard();boundary=basis['boundary']
        with np.load(basis['file']['path'],allow_pickle=False) as z:
            G=z['gram'];Q=z['eigenvectors'];ev=z['eigenvalues'];R=z['basis'];F32=z['basis_f32']
            assert G.shape==Q.shape==(2048,2048) and R.shape==(2048,256)
            assert np.array_equal(R,Q[:,:256]) and np.array_equal(F32,R.astype('f4'))
            assert np.isfinite(G).all() and np.isfinite(Q).all() and np.isfinite(ev).all()
            assert (np.diff(ev)<=0).all() and ev[-1]>=-1e-10*max(float(ev[0]),1)
            full=np.zeros_like(G);mean=np.zeros(2048)
            for rec in fit:
                guard();u=normalize(load(rec,boundary));full+=(u.T@u)/len(u);mean+=u.mean(0)
            full/=24;mean/=24
            gram_error=float(np.linalg.norm(full-G)/max(float(np.linalg.norm(full)),1e-300))
            orth=float(np.max(np.abs(Q.T@Q-np.eye(2048))))
            reconstruction=float(np.linalg.norm((Q*ev)@Q.T-full)/max(float(np.linalg.norm(full)),1e-300))
            assert gram_error<=1e-10 and orth<=1e-10 and reconstruction<=1e-10
            assert np.max(np.abs(mean-z['mean_angular']))<=1e-10
            assert abs(float(ev[:256].sum()/ev.sum())-basis['FIT_spectral_retained_fraction'])<=1e-12
            for rec in b['records']:
                guard();x=load(rec,boundary);row=lookup[(rec['id'],boundary)]
                actual=quadratic(x,R);baseline=quadratic(x,old);rounded=quadratic(x,F32.astype('f8'))
                for kind,values in [('new',actual),('old',baseline)]:
                    for key,value in values.items():max_delta=max(max_delta,abs(value-row[kind][key]))
                rounding=max(abs(rounded[k]-actual[k]) for k in ('angular_relative_RMS','centered_relative_RMS','angular_energy_fraction'))
                max_delta=max(max_delta,abs(rounding-row['F32_basis_metric_delta']));assert rounding<=1e-5
                case_rows.append(dict(id=rec['id'],split=rec['split'],boundary=boundary,new=actual,old=baseline))
            checked.append(dict(boundary=boundary,covariance_relative=gram_error,orthogonality_max=orth,full_spectral_reconstruction_relative=reconstruction))
            print(json.dumps(dict(boundary=boundary,max_metric_delta=max_delta,seconds=time.monotonic()-START)),flush=True)
    for tr,first,second in zip(r['transport'],r['bases'],r['bases'][1:]):
        with np.load(first['file']['path'],allow_pickle=False) as x,np.load(second['file']['path'],allow_pickle=False) as y:
            T=np.fromfile(tr['file']['path'],'<f8').reshape(256,256);expected=y['basis'].T@x['basis']
            assert np.max(np.abs(T-expected))<=1e-12
            sigma=np.linalg.svd(T,compute_uv=False)
            assert abs(float(np.mean(sigma*sigma))-tr['mean_squared_singular'])<=1e-12
            assert abs(float(sigma[-1])-tr['min_singular'])<=1e-12 and abs(float(sigma[0])-tr['max_singular'])<=1e-12
            assert abs(float(np.linalg.norm(T-np.eye(256))/math.sqrt(256))-tr['identity_difference_relative'])<=1e-12
    for summary in r['summary']:
        dev=[x for x in case_rows if x['boundary']==summary['boundary'] and x['split']=='DEV'];assert len(dev)==24
        values=dict(DEV_angular_relative_RMS_mean=math.fsum(x['new']['angular_relative_RMS'] for x in dev)/24,
            DEV_angular_relative_RMS_worst=max(x['new']['angular_relative_RMS'] for x in dev),
            DEV_centered_relative_RMS_mean=math.fsum(x['new']['centered_relative_RMS'] for x in dev)/24)
        for key,value in values.items():max_delta=max(max_delta,abs(value-summary[key]))
        assert {key:values[key]<=bound for key,bound in b['proxy'].items()}==summary['proxy_gates']
    assert len(case_rows)==336 and max_delta<=1e-8;check();guard()
    out=dict(schema='SOURCE_RESIDUAL_COORDINATE_STORED_AUDIT_V1',source=extent(__file__),report=extent(a.report),terminal=extent(a.terminal),
        all7_covariances_full_eigensystems_and336_case_metrics_checked=True,all6_residual_transports_checked=True,
        all_inputs_outputs_before_after_exact=True,max_quadratic_metric_delta=max_delta,checks=checked,
        decision=r['decision'],seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
        scope='Independent moment accumulation/full spectral reconstruction and quadratic projection metrics; '
            'no eigensolver/refit/source/shadow/GPU/native replay, no capacity/behavior/throughput admission.')
    write(a.out,out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--report-sha',required=True)
    p.add_argument('--terminal',type=Path,required=True);p.add_argument('--terminal-sha',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    try:main(a)
    except BaseException as error:
        import traceback
        write(a.out.with_suffix('.fault.json'),dict(error=repr(error),traceback=traceback.format_exc(),command=sys.orig_argv,
            source=extent(__file__),seconds=time.monotonic()-START));raise
