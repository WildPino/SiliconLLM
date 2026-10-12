"""Separate smaller-scale derivative study; parent acceptance remains false."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
from exact_categorical_residual_correction import selected,quality
START=time.monotonic()


def prepare(a):
    import numpy as np,psutil
    report=DOC/'exact_categorical_residual_correction_result_20261012.json';audit=DOC/'exact_categorical_residual_correction_stored_adjudication_20261012.json'
    r=json.loads(report.read_bytes());ad=json.loads(audit.read_bytes());assert ad['report']==extent(report) and ad['decision']=='INDEPENDENT_STORED_PASS' and not r['accepted']
    parent=json.loads(Path(r['binding']['path']).read_bytes())
    paths=[Path(x['path']) for x in parent['inputs']+r['outputs']]+[report,audit,Path(r['binding']['path']),Path(__file__),
        B/'exact_categorical_residual_correction.py',DOC/'EXACT_CATEGORICAL_RESIDUAL_REFINEMENT_PROTOCOL_20261012.md']
    b=dict(schema='EXACT_CATEGORICAL_RESIDUAL_REFINEMENT_BINDING_V1',parent=r,parent_binding=parent,parent_audit=extent(audit),
        epsilon=2e-7,tolerance=parent['finite_derivative_relative_absolute'],limits=dict(seconds=300,OS_bytes=4<<30,output_bytes=1<<30),
        inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in paths)],
        scope='Separate derivative convergence study; parent13/14 original gates and NONacceptance remain unchanged. '
            'Require all7 parent2e-6 and all7 new2e-7 controls at unchanged2e-6 tolerance; reuse original4 frozen FIT candidates only. '
            'One independently qualified correction, no new gradient/alpha/epoch, no source model/GPU/native/T4 or causal admission.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)


def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());r=b['parent'];pb=b['parent_binding']
    assert not a.out.exists() and not a.directory.exists();proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None
    receipt=dict(schema='EXACT_CATEGORICAL_RESIDUAL_REFINEMENT_TERMINAL_V1',binding=extent(a.binding),freeze=a.freeze,
        command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),timing_control_admitted=False)
    def guard():assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
    def check():
        for item in b['inputs']:guard();assert extent(item['path'])==item,item['path']
    def load(item):
        with np.load(item['path'],allow_pickle=False) as pack:return {k:pack[k].copy() for k in pack.files}
    def save(name,**arrays):
        guard();path=a.directory/(name+'.npz');np.savez(path,**arrays);return extent(path)
    try:
        check();a.directory.mkdir();get,stream=tensor_reader(Path(pb['source'])/'model.safetensors',np);cfg=json.loads((Path(pb['source'])/'config.json').read_bytes());gamma=get('model.final_layernorm.weight');A=np.empty((65537,2048),'f8')
        for lo in range(0,65537,1024):guard();hi=min(65537,lo+1024);A[lo:hi]=get('lm_head.weight',list(range(lo,hi)))*gamma*cfg['lm_head_multiplier']
        stream.close();stream=None;fit=load(r['FIT_reference']);H,ref=fit['H'],fit['scores'];theta=load(r['initial_parameters']);grad=load(r['gradient'])
        frames={f['name']:f for f in r['finite']};rows=[]
        for name,old in frames.items():
            guard();frame=load(old['direction']);points=[]
            for sign in [-1,1]:
                pp={k:theta[k]+sign*b['epsilon']*frame[k] for k in theta};z=(H-pp['mu'])@pp['W'];y=pp['mu']+z@pp['V'].T;rad=np.sqrt(np.mean(y*y,axis=1)+pb['epsilon']);scores=(y/rad[:,None])@A.T
                loss,ss=quality(scores,ref,np);file=save('finite_'+name+'_'+str(sign),z=z,decoded=y,radius=rad,scores=scores,KL=loss);points.append(dict(file=file,summary=ss))
            analytic=math.fsum(float(np.sum(grad[k]*frame[k])) for k in theta);derivative=(points[1]['summary']['mean_KL']-points[0]['summary']['mean_KL'])/(2*b['epsilon']);delta=abs(derivative-analytic)/max(1.,abs(analytic))
            coarse=next(f for f in r['finite'] if f['name']==name and f['epsilon']==1e-5);fine=next(f for f in r['finite'] if f['name']==name and f['epsilon']==2e-6)
            richardson=(25*fine['finite']-coarse['finite'])/24
            rows.append(dict(name=name,direction=old['direction'],epsilon=b['epsilon'],minus=points[0],plus=points[1],analytic=analytic,finite=derivative,delta=delta,gate=delta<=b['tolerance'],
                prior_fine_gate=fine['gate'],Richardson_two_prior_scales=richardson,Richardson_relative_absolute_delta=abs(richardson-analytic)/max(1.,abs(analytic))))
            print(json.dumps(dict(direction=name,delta=delta,gate=rows[-1]['gate'],seconds=time.monotonic()-START)),flush=True)
        assert len(rows)==7;best=min(r['candidates'],key=lambda x:x['summary']['mean_KL']);accepted=all(f['prior_fine_gate'] and f['gate'] for f in rows) and r['FIT_relative_improvement']>=pb['FIT_relative_improvement_required']
        chosen=best['parameters'] if accepted else r['initial_parameters'];seal=a.directory/'selection.json';write(seal,dict(parent_acceptance_unchanged=False,refined_acceptance=accepted,
            parameters=chosen,alpha=best['alpha'] if accepted else 0.,rule='All parent2e-6 and new2e-7 controls;unchanged tolerance;original FIT-only ladder. Sealed before new selected DEV calculation.'))
        selection=extent(seal);pp=load(chosen);devH,labels=selected(pb['DEV'],np);assert labels==pb['DEV_labels'];devref=np.fromfile(pb['DEV_reference']['path'],'<f8').reshape(72,65537)
        z=(devH-pp['mu'])@pp['W'];y=pp['mu']+z@pp['V'].T;rad=np.sqrt(np.mean(y*y,axis=1)+pb['epsilon']);scores=(y/rad[:,None])@A.T;loss,ss=quality(scores,devref,np)
        ss['proxy_gates']={k:ss[k]<=v for k,v in pb['probability_proxy'].items()};devfile=save('DEV_selected',H=devH,z=z,decoded=y,radius=rad,scores=scores,KL=loss)
        assert extent(seal)==selection;with_norm=float(np.max(np.abs(pp['W'].T@pp['V']-np.eye(256))))
        check();guard();outputs=[extent(p) for p in sorted(a.directory.iterdir())];assert sum(x['bytes'] for x in outputs)<=b['limits']['output_bytes']
        result=dict(schema='EXACT_CATEGORICAL_RESIDUAL_REFINEMENT_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,rows=rows,
            parent_acceptance_stays_false=True,refined_acceptance=accepted,selected_parameters=chosen,selected_alpha=best['alpha'] if accepted else 0.,selection=selection,
            FIT_before=r['baseline']['summary'],FIT_selected=best['summary'] if accepted else r['baseline']['summary'],DEV=dict(file=devfile,labels=labels,summary=ss),
            selected_WtV_identity_defect=with_norm,outputs=outputs,inputs_before_after_exact=True,seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
            decision='REFINED_DERIVATIVE_VERIFIED_FIT_DESCENT' if accepted else 'COMPLETED_REFINED_NONACCEPTANCE',scope=b['scope'])
        write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.prepare:prepare(a)
    else:assert all((a.binding,a.binding_sha,a.freeze,a.directory));run(a)
