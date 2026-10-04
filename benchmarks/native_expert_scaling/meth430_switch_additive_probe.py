"""Saturated training-target vs fixed numerical-probe diagnostic; no fitting."""
import argparse
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
import meth422_switch_function_gradient_contract as R
import meth422_switch_function_autograd as G
import meth427_switch_additive_math as H
PROTOCOL=M.DOC/'METH_430_SWITCH_ADDITIVE_PROBE_PROTOCOL_20261004.md'
FAILURE=M.DOC/'meth429_switch_additive_pilot_result.failure.json'
FAILURE_SHA='7d82c7cdd7197df42c14a179301deb72743848c6d7db39a30c45b506e57b6a55'
OUT=M.ROOT/'results/native_expert_scaling/meth430_switch_additive_probe'


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic(); peak=0; hashed=0; result={'experiment':'METH-430-fixed-real-additive-numerical-probe','actual_mixture_gradients':{},'probe_qualifications':[]}; stage='bindings'
    def guard():
        nonlocal peak
        mi=psutil.Process().memory_info(); peak=max(peak,mi.rss,getattr(mi,'peak_wset',0))
        assert peak<=3<<30 and time.monotonic()-start<=120,'120sec_3GiB_probe_diagnostic'
    def sha(p):
        nonlocal hashed
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            while block:=f.read(4<<20):h.update(block); hashed+=len(block); guard()
        return h.hexdigest()
    try:
        result['helper_sha256']={}
        for p in (Path(__file__),PROTOCOL,FAILURE,Path(H.__file__),Path(G.__file__),Path(R.__file__),Path(H.L.__file__),Path(M.__file__)):
            M.committed(p); result['helper_sha256'][str(p)]=sha(p)
        assert sha(FAILURE)==FAILURE_SHA; fail=json.loads(FAILURE.read_text(encoding='utf-8')); assert fail['updates']==[] and len(fail['qualifications'])==3
        for p,expected in fail['helper_sha256'].items():M.committed(Path(p)); assert sha(p)==expected
        for a in fail['partial_output_inventory']:assert sha(a['path'])==a['sha256']
        archive=next(a for a in fail['partial_output_inventory'] if Path(a['path']).name=='real.real.npz')
        assert archive['sha256']=='dd1e637da954ec525777861ce9ed644481f2fd5f3d800f6faed34e275027cc24'
        own={os.getpid(),*(p.pid for p in psutil.Process().parents())}
        for p in psutil.process_iter(['name','cmdline']):
            if p.pid in own:continue
            name=(p.info['name'] or '').lower(); argv=p.info['cmdline'] or []
            if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():continue
            assert not(name.startswith('python') or ('meth' in name and name.endswith('.exe'))),('concurrent_job',p.pid,name)
        psutil.Process().cpu_affinity([0]);torch.set_num_threads(1);torch.set_num_interop_threads(1)
        assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6'
        mapped={};entries={}
        for n,(name,expected) in R.C.U.EXPORT.items():
            p=M.DOC/name;M.committed(p);assert sha(p)==expected;e=json.loads(p.read_text(encoding='utf-8'));a=e['artifact']
            assert a==fail['artifacts'][str(n)] and sha(a['payload'])==a['sha256'] and sha(a['manifest'])==a['manifest_sha256']
            R.B.read_manifest(a['manifest'],e['original_config'],e['tensors'],Path(a['payload']));mapped[n]=np.memmap(a['payload'],dtype='u1',mode='r');entries[n]=e['tensors']
        with np.load(archive['path']) as z:pre=z['pre'].copy();base=z['base'].copy();scores=z['scores'].copy();target=z['target'].copy();original=z['native_logits'].copy();saved_grad=z['native_gradient_base'].copy()
        assert pre.size==768 and scores.size==256 and np.argmax(scores)==5
        norm=tuple(R.C.tensor(mapped[256],entries[256],k) for k in ('decoder.block.11.layer.2.layer_norm.weight','decoder.final_layer_norm.weight'))
        def op(n,name):return G.I8Operator(R.C.tensor(mapped[n],entries[n],name+'.weight'),R.C.tensor(mapped[n],entries[n],name+'.weight','scales'))
        wi=op(128,'decoder.block.11.layer.2.mlp.experts.expert_25.wi');wo=op(128,'decoder.block.11.layer.2.mlp.experts.expert_25.wo');head=op(256,'lm_head')
        k=(int(np.argmax(original))+original.size//2)%original.size;probe=np.zeros(original.size,np.float64);probe[k]=1.
        result['probe_target']={'one_hot_index':k,'original_argmax':int(np.argmax(original)),'rule':'(argmax_original + vocabulary//2)%vocabulary','training_target_unchanged':True}
        OUT.mkdir(parents=True)
        with threadpool_limits(limits=1):
            result['runtime']={'torch':torch.__version__,'numpy':np.__version__,'CPU_affinity':[0],'BLAS':threadpool_info(),'GPU':False}
            for arm in ('real','adapter'):
                stage=arm+'_actual_mixture';parameters=G.corrections(768,8,25772,zero=True);tx=torch.tensor(pre,requires_grad=True);tb=torch.tensor(base,requires_grad=True);ts=torch.tensor(scores,requires_grad=True)
                actual=H.forward(arm,tx,tb,ts,5,*norm,wi,wo,head,parameters);R.C.exact(actual['logits'].detach().numpy(),original)
                loss=G.prediction_loss(actual['logits'],torch.tensor(target));loss.backward()
                gradients={k:v.grad.numpy().copy() for k,v in parameters.items()};gradients.update(pre=tx.grad.numpy().copy(),base=tb.grad.numpy().copy(),scores=ts.grad.numpy().copy())
                if arm=='real':R.C.exact(gradients['base'],saved_grad)
                norms={k:float(np.linalg.norm(v)) for k,v in gradients.items()};result['actual_mixture_gradients'][arm]={'norms':norms,'loss':float(loss.detach()),'live_A_C_gt1e_10':all(norms[k]>1e-10 for k in ('A','C'))}
                np.savez(OUT/(arm+'.actual_mixture.npz'),**{'gradient_'+k:v for k,v in gradients.items()},pre=pre,base=base,scores=scores,target=target,logits=original)
                del actual,parameters,tx,tb,ts,loss;guard();stage=arm+'_probe'
                result['probe_qualifications'].append(H.qualify(arm,arm+'.probe',pre,base,scores,5,*norm,wi,wo,head,G.corrections(768,8,25772,zero=True),probe,False,OUT/(arm+'.probe.npz'),expected_logits=original));guard()
        result['gates']={'original429_real_base_gradient_exact':True,'actual_adapter_live_guard_failure_reproduced':not result['actual_mixture_gradients']['adapter']['live_A_C_gt1e_10'],
            'both_fixed_probe_FULL_qualification_SAME_bounds':len(result['probe_qualifications'])==2,'no_training_objective_or_forward_change':True}
        result['output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.glob('*.npz'))]
        assert sum(a['bytes'] for a in result['output_inventory'])<=8<<20
        result['resource']={'seconds':time.monotonic()-start,'peak_bytes':peak,'bytes_hashed':hashed,'output_bytes':sum(a['bytes'] for a in result['output_inventory'])}
        result['decision']='eligible_for_NEW_numeric_probe_only_pilot_prerequisite' if all(result['gates'].values()) else 'stop_no_fit_license'
        result['scope']='SAME first actual core256/source128 function25, seed25772/rank8/zero factors. Artificial one-hot target ONLY numerical derivative probe; actual pilot teacher-mixture objective/data/classifier/capacity unchanged. No fit/selector/export/new whole quality/rate/DRAM/LUT or promotion of429.'
        guard();M.write(args.out,result);print(json.dumps({'gates':result['gates'],'actual_gradients':result['actual_mixture_gradients'],'resource':result['resource'],'sha256':M.digest(args.out)}))
    except BaseException as error:
        result.update(failure_stage=stage,error=repr(error),seconds=time.monotonic()-start,peak_bytes=peak,bytes_hashed=hashed)
        if OUT.exists():result['partial_output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':M.digest(p)} for p in sorted(OUT.glob('*.npz'))]
        M.write(args.out.with_suffix('.failure.json'),result);raise

if __name__=='__main__':main()
