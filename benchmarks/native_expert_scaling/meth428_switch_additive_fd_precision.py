"""Same-anchor additive tiny FD precision diagnostic, no fit license."""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits, threadpool_info
import meth324_switch_reference as M
import meth422_switch_function_autograd as G
import meth422_switch_function_gradient_contract as R
import meth424_switch_anchored_reference as L
import meth424_switch_matched_primal_contract as K
import meth427_switch_additive_math as H
forward = H.forward
PROTOCOL=M.DOC/'METH_428_SWITCH_ADDITIVE_FD_PRECISION_PROTOCOL_20261004.md'
OUT=M.ROOT/'results/native_expert_scaling/meth428_switch_additive_fd_precision'
FAILURE=M.DOC/'meth427_switch_additive_pilot_result.failure.json'
FAILURE_SHA='8c36e4efb071516475fe6487e0bdde92661f90a38c932ab24083d0bd876e7717'


def diagnose(arm, label, pre, base, scores, chosen, ff, fn, wi, wo, head, parameters, target, tiny, path, expected_logits=None):
    values = {k: z.detach().numpy().copy() for k, z in parameters.items()}
    tx = torch.tensor(pre.copy(), requires_grad=True); tb = torch.tensor(base.copy(), requires_grad=True); ts = torch.tensor(scores.copy(), requires_grad=True)
    actual = forward(arm, tx, tb, ts, chosen, ff, fn, wi, wo, head, parameters)
    loss = G.prediction_loss(actual['logits'], torch.tensor(target)); loss.backward()
    native_grad = {k: z.grad.numpy().copy() for k, z in parameters.items()}
    native_grad.update(pre=tx.grad.numpy().copy(), base=tb.grad.numpy().copy(), scores=ts.grad.numpy().copy())
    if tiny: assert all(np.linalg.norm(native_grad[k]) > 0 for k in ('A', 'B', 'C', 'D'))
    else:
        live = ('C',) if arm == 'real' else ('A', 'C')
        zero = ('A', 'B', 'D') if arm == 'real' else ('B', 'D')
        assert all(np.linalg.norm(native_grad[k]) > 1e-10 for k in live)
        assert all(np.count_nonzero(native_grad[k]) == 0 for k in zero)
        R.C.exact(actual['post'].detach().numpy(), base)
        if expected_logits is not None: R.C.exact(actual['logits'].detach().numpy(), expected_logits)
    # Independent literal NumPy native evaluation, including every correction node.
    anchor = {}; anchor['input'] = L.native_rms(pre, ff)
    dot = lambda w, x: (w.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)
    anchor['Bx'] = dot(values['B'], anchor['input']); anchor['Ax'] = dot(values['A'], anchor['Bx'])
    if arm == 'real':
        anchor['corrected_input'] = anchor['input']+anchor['Ax']; anchor['up_raw'] = wi.native(anchor['corrected_input'])
        anchor['up'] = np.where(anchor['up_raw'] < 0, np.float32(0), anchor['up_raw']); anchor['feature'] = wo.native(anchor['up'])
        anchor['Dy'] = dot(values['D'], anchor['feature']); anchor['Cy'] = dot(values['C'], anchor['Dy']); anchor['correction'] = anchor['Cy'].copy()
    else:
        anchor['Dy'] = dot(values['D'], anchor['input']); anchor['Cy'] = dot(values['C'], anchor['Dy']); anchor['correction'] = anchor['Ax']+anchor['Cy']
    exp = np.exp((scores-scores[chosen]).astype(np.float64)).astype(np.float32).astype(np.float64)
    anchor['probability'] = np.float32(1/np.sum(exp)); anchor['weighted'] = anchor['probability']*anchor['correction']
    anchor['post'] = base+anchor['weighted']; anchor['final'] = L.native_rms(anchor['post'], fn)
    scale = float(np.float32(1/np.sqrt(len(pre))))
    anchor['head_input'] = anchor['final']*np.float32(scale); anchor['logits'] = head.native(anchor['head_input'])
    for k, z in actual.items(): R.C.exact(z.detach().numpy(), anchor[k])
    q = {k: np.asarray(z, np.float64) for k, z in anchor.items()}; pv = {k: z.astype(np.float64) for k, z in values.items()}
    x = pre.astype(np.float64); b = base.astype(np.float64); s = scores.astype(np.float64); f = ff.astype(np.float64); n = fn.astype(np.float64)
    wh = head.smooth_weight().numpy(); ww = wi.smooth_weight().numpy() if arm == 'real' else None; ow = wo.smooth_weight().numpy() if arm == 'real' else None
    at_native = {'input': L.rms(x, f), 'Bx': pv['B']@q['input'], 'Ax': pv['A']@q['Bx']}
    if arm == 'real':
        at_native.update(corrected_input=q['input']+q['Ax'], up_raw=ww@q['corrected_input'], up=np.maximum(q['up_raw'], 0), feature=ow@q['up'],
                         Dy=pv['D']@q['feature'], Cy=pv['C']@q['Dy'], correction=q['Cy'])
    else: at_native.update(Dy=pv['D']@q['input'], Cy=pv['C']@q['Dy'], correction=q['Ax']+q['Cy'])
    at_native.update(probability=L.selected_probability(s, chosen), weighted=q['probability']*q['correction'], post=b+q['weighted'],
                     final=L.rms(q['post'], n), head_input=q['final']*scale, logits=wh@q['head_input'])
    offsets = {k: q[k]-at_native[k] for k in q}; fingerprints = {k: v.tobytes() for k, v in offsets.items()}
    def numpy_path(xx, bb, ss, pp, off=offsets):
        z = {}; z['input'] = L.rms(xx, f)+off['input']; z['Bx'] = pp['B'].dot(z['input'])+off['Bx']; z['Ax'] = pp['A'].dot(z['Bx'])+off['Ax']
        if arm == 'real':
            z['corrected_input'] = z['input']+z['Ax']+off['corrected_input']; z['up_raw'] = ww.dot(z['corrected_input'])+off['up_raw']
            z['up'] = np.maximum(z['up_raw'], 0)+off['up']; z['feature'] = ow.dot(z['up'])+off['feature']
            z['Dy'] = pp['D'].dot(z['feature'])+off['Dy']; z['Cy'] = pp['C'].dot(z['Dy'])+off['Cy']; z['correction'] = z['Cy']+off['correction']
        else:
            z['Dy'] = pp['D'].dot(z['input'])+off['Dy']; z['Cy'] = pp['C'].dot(z['Dy'])+off['Cy']; z['correction'] = z['Ax']+z['Cy']+off['correction']
        z['probability'] = L.selected_probability(ss, chosen)+off['probability']; z['weighted'] = z['probability']*z['correction']+off['weighted']
        z['post'] = bb+z['weighted']+off['post']; z['final'] = L.rms(z['post'], n)+off['final']; z['head_input'] = z['final']*scale+off['head_input']
        z['logits'] = wh.dot(z['head_input'])+off['logits']; return z
    fx = torch.tensor(x, requires_grad=True); fb = torch.tensor(b, requires_grad=True); fs = torch.tensor(s, requires_grad=True)
    fp = {k: torch.tensor(v, requires_grad=True) for k, v in pv.items()}; off = {k: torch.tensor(v) for k, v in offsets.items()}
    z = {}; z['input'] = fx*torch.rsqrt(torch.mean(fx*fx)+1e-6)*torch.tensor(f)+off['input']
    z['Bx'] = fp['B']@z['input']+off['Bx']; z['Ax'] = fp['A']@z['Bx']+off['Ax']
    if arm == 'real':
        z['corrected_input'] = z['input']+z['Ax']+off['corrected_input']; z['up_raw'] = wi.smooth_weight()@z['corrected_input']+off['up_raw']
        z['up'] = torch.relu(z['up_raw'])+off['up']; z['feature'] = wo.smooth_weight()@z['up']+off['feature']
        z['Dy'] = fp['D']@z['feature']+off['Dy']; z['Cy'] = fp['C']@z['Dy']+off['Cy']; z['correction'] = z['Cy']+off['correction']
    else:
        z['Dy'] = fp['D']@z['input']+off['Dy']; z['Cy'] = fp['C']@z['Dy']+off['Cy']; z['correction'] = z['Ax']+z['Cy']+off['correction']
    z['probability'] = torch.softmax(fs, dim=0)[chosen]+off['probability']; z['weighted'] = z['probability']*z['correction']+off['weighted']
    z['post'] = fb+z['weighted']+off['post']; z['final'] = z['post']*torch.rsqrt(torch.mean(z['post']*z['post'])+1e-6)*torch.tensor(n)+off['final']
    z['head_input'] = z['final']*scale+off['head_input']; z['logits'] = head.smooth_weight()@z['head_input']+off['logits']
    npout = numpy_path(x, b, s, pv); endpoints = {}
    for k in q:
        errors = (R.relative(npout[k], q[k]), R.relative(z[k].detach().numpy(), q[k]))
        absolute = max(float(np.max(np.abs(npout[k]-q[k]))), float(np.max(np.abs(z[k].detach().numpy()-q[k]))))
        assert max(errors) <= 1e-12 and absolute <= 1e-9, ('additive_endpoint', label, k, errors, absolute)
        endpoints[k] = {'relative_max': max(errors), 'absolute_max': absolute}
    G.prediction_loss(z['logits'], torch.tensor(target)).backward()
    ref = {k: v.grad.numpy().copy() for k, v in fp.items()}; ref.update(pre=fx.grad.numpy().copy(), base=fb.grad.numpy().copy(), scores=fs.grad.numpy().copy())
    comparisons = {k: R.relative(native_grad[k], ref[k]) for k in ref}; assert max(comparisons.values()) <= 1e-3, ('additive_gradient', label, comparisons)
    base_logits=npout['logits'].copy(); prob0=R.probability(base_logits); kmax=int(np.argmax(base_logits)); mask=anchor.get('up_raw',np.ones(1))>0
    def imaginary_path(xx,bb,ss,pp):
        zz={}; zz['input']=L.rms(xx,f)+offsets['input']; zz['Bx']=pp['B'].dot(zz['input'])+offsets['Bx']; zz['Ax']=pp['A'].dot(zz['Bx'])+offsets['Ax']
        if arm=='real':
            zz['corrected_input']=zz['input']+zz['Ax']+offsets['corrected_input']; zz['up_raw']=ww.dot(zz['corrected_input'])+offsets['up_raw']
            zz['up']=np.where(mask,zz['up_raw'],0)+offsets['up']; zz['feature']=ow.dot(zz['up'])+offsets['feature']
            zz['Dy']=pp['D'].dot(zz['feature'])+offsets['Dy']; zz['Cy']=pp['C'].dot(zz['Dy'])+offsets['Cy']; zz['correction']=zz['Cy']+offsets['correction']
        else:
            zz['Dy']=pp['D'].dot(zz['input'])+offsets['Dy']; zz['Cy']=pp['C'].dot(zz['Dy'])+offsets['Cy']; zz['correction']=zz['Ax']+zz['Cy']+offsets['correction']
        exp=np.exp(ss-float(s[chosen])); zz['probability']=exp[chosen]/np.sum(exp)+offsets['probability']
        zz['weighted']=zz['probability']*zz['correction']+offsets['weighted']; zz['post']=bb+zz['weighted']+offsets['post']
        zz['final']=L.rms(zz['post'],n)+offsets['final']; zz['head_input']=zz['final']*scale+offsets['head_input']; zz['logits']=wh.dot(zz['head_input'])+offsets['logits']; return zz
    assert R.relative(imaginary_path(x,b,s,pv)['logits'],base_logits)<=1e-12
    def loss_delta(logits):
        delta=logits-base_logits; delta-=delta[kmax]
        return np.log1p(np.sum(prob0*np.expm1(delta)))-np.dot(target,delta)
    # Independent standard-library Decimal(80) reference, same native offsets and FD h.
    cv=lambda value: np.vectorize(lambda v: Decimal(float(v)),otypes=[object])(value)
    dx,db,ds,df,dn,dw,do,dh=cv(x),cv(b),cv(s),cv(f),cv(n),cv(ww) if ww is not None else None,cv(ow) if ow is not None else None,cv(wh)
    dp={k:cv(v) for k,v in pv.items()}; od={k:cv(v) for k,v in offsets.items()}; dscale=Decimal(scale); dt=cv(target)
    def decimal_rms(xx,weight):
        denom=(sum(v*v for v in xx)/Decimal(xx.size)+Decimal(1e-6)).sqrt(); return xx/denom*weight
    def decimal_path(xx,bb,ss,pp):
        zz={}; zz['input']=decimal_rms(xx,df)+od['input']; zz['Bx']=pp['B'].dot(zz['input'])+od['Bx']; zz['Ax']=pp['A'].dot(zz['Bx'])+od['Ax']
        if arm=='real':
            zz['corrected_input']=zz['input']+zz['Ax']+od['corrected_input']; zz['up_raw']=dw.dot(zz['corrected_input'])+od['up_raw']
            zz['up']=np.where(mask,zz['up_raw'],Decimal(0))+od['up']; zz['feature']=do.dot(zz['up'])+od['feature']
            zz['Dy']=pp['D'].dot(zz['feature'])+od['Dy']; zz['Cy']=pp['C'].dot(zz['Dy'])+od['Cy']; zz['correction']=zz['Cy']+od['correction']
        else:
            zz['Dy']=pp['D'].dot(zz['input'])+od['Dy']; zz['Cy']=pp['C'].dot(zz['Dy'])+od['Cy']; zz['correction']=zz['Ax']+zz['Cy']+od['correction']
        ex=np.array([(v-ds[chosen]).exp() for v in ss],object); zz['probability']=ex[chosen]/sum(ex)+od['probability']
        zz['weighted']=zz['probability']*zz['correction']+od['weighted']; zz['post']=bb+zz['weighted']+od['post']
        zz['final']=decimal_rms(zz['post'],dn)+od['final']; zz['head_input']=zz['final']*dscale+od['head_input']; zz['logits']=dh.dot(zz['head_input'])+od['logits']; return zz
    checks=[]; arrays={}; crossings=0
    with localcontext() as context:
        context.prec=80; decimal_anchor=decimal_path(dx,db,ds,dp)['logits']; shifted=decimal_anchor-decimal_anchor[kmax]
        dex=np.array([v.exp() for v in shifted],object); dp0=dex/sum(dex)
        assert R.relative(np.array([float(v) for v in decimal_anchor]),base_logits)<=1e-12
        for field in ('A','B','C','D','pre','scores','base'):
            value=x if field=='pre' else b if field=='base' else s if field=='scores' else pv[field]
            def evaluate(v, mode):
                xx,bb,ss,pp=(dx,db,ds,dp) if mode=='decimal' else (x,b,s,pv)
                if field=='pre': xx=v
                elif field=='base': bb=v
                elif field=='scores': ss=v
                else: pp=pp | {field:v}
                return (decimal_path if mode=='decimal' else imaginary_path if mode=='imaginary' else numpy_path)(xx,bb,ss,pp)
            def objective(v):
                nonlocal crossings
                zz=evaluate(v,'F64')
                if arm=='real': crossings+=int(np.count_nonzero((zz['up_raw']>0)!=mask))
                return loss_delta(zz['logits'])
            fd=R.finite_difference(objective,value,1e-4); imaginary=np.zeros_like(value); decimal_fd=np.zeros_like(value)
            for index in np.ndindex(value.shape):
                v=value.astype(np.complex128); v[index]+=1j*1e-20
                imaginary[index]=loss_delta(evaluate(v,'imaginary')['logits']).imag/1e-20
                vv=cv(value); h=Decimal(1e-4); derivatives=[]
                for sign in (1,-1):
                    perturbed=vv.copy(); perturbed[index]+=Decimal(sign)*h
                    dz=evaluate(perturbed,'decimal')['logits']-decimal_anchor; dz-=dz[kmax]
                    derivatives.append((Decimal(1)+sum(dp0[j]*(dz[j].exp()-Decimal(1)) for j in range(len(dz)))).ln()-sum(dt[j]*dz[j] for j in range(len(dz))))
                decimal_fd[index]=float((derivatives[0]-derivatives[1])/(Decimal(2)*h))
            errors={'F64_FD_reference':R.relative(ref[field],fd),'F64_FD_native':R.relative(native_grad[field],fd),
                    'imaginary_reference':R.relative(ref[field],imaginary),'imaginary_native':R.relative(native_grad[field],imaginary),
                    'decimal80_FD_reference':R.relative(ref[field],decimal_fd),'decimal80_FD_native':R.relative(native_grad[field],decimal_fd),
                    'decimal80_FD_imaginary':R.relative(imaginary,decimal_fd)}
            checks.append({'field':field,'coordinates':value.size,'reference_gradient_norm':float(np.linalg.norm(ref[field])),'errors':errors})
            arrays['F64_FD_'+field]=fd; arrays['imaginary_'+field]=imaginary; arrays['decimal80_FD_'+field]=decimal_fd
        if arm=='real': assert checks[0]['errors']['F64_FD_reference']==0.0005427976577140341
    assert crossings==0 and all(v.tobytes()==fingerprints[k] for k,v in offsets.items())
    for k in q: arrays['native_'+k]=anchor[k]; arrays['continuation_'+k]=npout[k]; arrays['offset_'+k]=offsets[k]
    for k in ref: arrays['native_gradient_'+k]=native_grad[k]; arrays['reference_gradient_'+k]=ref[k]
    np.savez(path,**arrays,**{'factor_'+k:v for k,v in values.items()},pre=pre,base=base,scores=scores,target=target)
    return {'arm':arm,'endpoints':endpoints,'native_reference_gradients':comparisons,'checks':checks,'ReLU_crossings':crossings,'offsets_fixed':True,'archive_path':str(path)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic(); peak=0; hashed=0; result={'experiment':'METH-428-SAME-tiny-additive-FD-precision-diagnostic','points':[]}; stage='bindings'
    def guard():
        nonlocal peak
        mi=psutil.Process().memory_info(); peak=max(peak,mi.rss,getattr(mi,'peak_wset',0))
        assert peak<=1<<30 and time.monotonic()-start<=120,'120sec_1GiB_diagnostic'
    def sha(p):
        nonlocal hashed
        data=Path(p).read_bytes(); hashed+=len(data); guard(); return hashlib.sha256(data).hexdigest()
    try:
        helpers=[Path(__file__),PROTOCOL,FAILURE,Path(H.__file__),Path(K.__file__),Path(L.__file__),Path(G.__file__),Path(R.__file__),Path(M.__file__)]
        result['helper_sha256']={}
        for p in helpers: M.committed(p); result['helper_sha256'][str(p)]=sha(p)
        assert sha(FAILURE)==FAILURE_SHA; failure=json.loads(FAILURE.read_text(encoding='utf-8')); assert failure['updates']==[] and 'real.tiny' in failure['error']
        for p,expected in failure['helper_sha256'].items(): M.committed(Path(p)); assert sha(p)==expected
        source=Path(H.__file__).read_text(encoding='utf-8'); own_source=Path(__file__).read_text(encoding='utf-8')
        expected=source[source.index('def qualify('):source.index('    base_logits =')].replace('def qualify(','def diagnose(',1)
        assert own_source[own_source.index('def diagnose('):own_source.index('    base_logits=')]==expected
        own={os.getpid(),*(p.pid for p in psutil.Process().parents())}
        for p in psutil.process_iter(['name','cmdline']):
            if p.pid in own: continue
            name=(p.info['name'] or '').lower(); argv=p.info['cmdline'] or []
            if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve(): continue
            assert not(name.startswith('python') or ('meth' in name and name.endswith('.exe'))),('concurrent_job',p.pid,name)
        psutil.Process().cpu_affinity([0]); torch.set_num_threads(1); torch.set_num_interop_threads(1)
        assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6'; OUT.mkdir(parents=True)
        with threadpool_limits(limits=1):
            result['runtime']={'torch':torch.__version__,'numpy':np.__version__,'BLAS':threadpool_info(),'CPU_affinity':[0],'GPU':False}
            tiny=K.tiny_inputs()
            with torch.no_grad(): base=G.forward(torch.from_numpy(tiny[0]),torch.from_numpy(tiny[1]),tiny[2],*tiny[3:5],*tiny[5:8])['post'].numpy()
            for arm in ('real','adapter'):
                stage=arm; result['points'].append(diagnose(arm,arm+'.tiny',tiny[0],base,tiny[1],tiny[2],*tiny[3:5],*tiny[5:8],G.corrections(7,2,419,zero=False),tiny[9],True,OUT/(arm+'.tiny.npz'))); guard()
        checks=[c for point in result['points'] for c in point['checks']]
        result['gates']={'original_first_FD_error_exact':True,'same_native_and_reference_prefix_exact':True,'fixed_offsets_no_ReLU_crossings':True,
            'ALL_imaginary_reference_at_most1e_5':all(c['errors']['imaginary_reference']<=1e-5 for c in checks),
            'ALL_imaginary_native_at_most1e_3':all(c['errors']['imaginary_native']<=1e-3 for c in checks),
            'ALL_decimal80_same_h_FD_reference_at_most1e_5':all(c['errors']['decimal80_FD_reference']<=1e-5 for c in checks),
            'ALL_decimal80_same_h_FD_native_at_most1e_3':all(c['errors']['decimal80_FD_native']<=1e-3 for c in checks),
            'ALL_decimal80_FD_imaginary_at_most1e_5':all(c['errors']['decimal80_FD_imaginary']<=1e-5 for c in checks)}
        result['output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.glob('*.npz'))]
        result['resource']={'seconds':time.monotonic()-start,'peak_bytes':peak,'bytes_hashed':hashed,'output_bytes':sum(a['bytes'] for a in result['output_inventory'])}
        result['scope']='SAME tiny nonzero two-arm additive local continuation precision diagnostic only. No real model source scan/forward, fit, C export, fresh quality/rate/LUT/DRAM or promotion of427. Decimal80 SAME1e-4 step; imaginary1e-20 independent fixed-mask control. All thresholds unchanged.'
        result['decision']='eligible_for_NEW_tiny_FD_precision_only_repair' if all(result['gates'].values()) else 'stop_diagnostic_no_fit_license'
        guard(); M.write(args.out,result); print(json.dumps({'gates':result['gates'],'resource':result['resource'],'sha256':M.digest(args.out)}))
    except BaseException as error:
        result.update(failure_stage=stage,error=repr(error),seconds=time.monotonic()-start,peak_bytes=peak,bytes_hashed=hashed)
        if OUT.exists(): result['partial_output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':M.digest(p)} for p in sorted(OUT.glob('*.npz'))]
        M.write(args.out.with_suffix('.failure.json'),result); raise

if __name__=='__main__': main()
