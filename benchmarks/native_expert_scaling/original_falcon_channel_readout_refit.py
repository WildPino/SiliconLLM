"""Frozen-channel FIT-only closed-form readout repair, no whole-model update."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';sys.path.insert(0,str(B))
from original_falcon_channel_transport import SITE,sha,write,extent,launch
sys.path.insert(0,str(SITE))


def bind(a):
    parent=DOC/'original_falcon_channel_transport_binding_20261009.json'
    b=json.loads(parent.read_bytes());result=DOC/'original_falcon_channel_transport_result_20261009.json'
    terminal=result.with_suffix('.terminal.json');t=json.loads(terminal.read_bytes())
    assert t['exit_code']==0 and t['resource_gates'] and sha(result)==t['result_sha256']
    r=json.loads(result.read_bytes());assert r['decision']=='CHANNEL_HEAD_REPRESENTATION_BUDGET_FAIL'
    for i in b['inputs']:assert extent(i['path'])==i
    files=[Path(__file__),parent,result,terminal,DOC/'ORIGINAL_FALCON_CHANNEL_READOUT_REFIT_PROTOCOL_20261009.md']
    for s in r['selections']:
        assert s['stored_selection_replay_exact'] and s['FIT_cases']==24
        assert extent(s['moments']['path'])==s['moments'];files.append(Path(s['moments']['path']))
    inputs=list({i['path']:i for i in b['inputs']+[extent(p) for p in files]}.values())
    b.update(worker_path=str(Path(__file__).resolve()),inputs=inputs,experiment='CHANNEL_READOUT_REFIT_V1',
        parent_result=str(result.resolve()),ridge_trace_fraction=1e-6,
        limits=dict(seconds=900,reserve_seconds=90,OS_bytes=6<<30,GPU_allocated_bytes=4<<30,
            GPU_reserved_bytes=5<<30,output_bytes=256<<20))
    b['criteria'].update(normal_equation_F64=1e-8,normal_equation_export_F32=1e-4)
    write(a.out,b);print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(inputs))),flush=True)


def worker(a):
    start=time.monotonic();assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes());assert b['experiment']=='CHANNEL_READOUT_REFIT_V1'
    a.directory.mkdir(exist_ok=False);stage='startup';selections=[];rows=[];torch=None;proc=None
    try:
        import numpy as np
        import torch
        import psutil
        from safetensors import safe_open
        assert (torch.__version__,np.__version__,psutil.__version__)==('2.6.0+cu124','2.4.6','7.2.2')
        torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        proc=psutil.Process();proc.cpu_affinity(list(range(6)))
        def guard():
            assert time.monotonic()-start<=b['limits']['seconds']-b['limits']['reserve_seconds']
            assert proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
            assert torch.cuda.max_memory_allocated()<=b['limits']['GPU_allocated_bytes']
            assert torch.cuda.max_memory_reserved()<=b['limits']['GPU_reserved_bytes']
        def event(**kw):
            guard();print(json.dumps(dict(seconds=time.monotonic()-start,**kw)),flush=True)
        records=json.loads(Path(b['capture']).read_bytes())['records'];fit=[r for r in records if r['split']=='FIT']
        parent=json.loads(Path(b['parent_result']).read_bytes());eps=json.loads(Path(b['source_config']).read_bytes())['rms_norm_eps']
        def raw(item):
            if item['dtype']=='BF16':z=(np.fromfile(item['path'],dtype='<u2').astype('<u4')<<16).view('<f4')
            else:assert item['dtype']=='F32';z=np.fromfile(item['path'],dtype='<f4')
            z=z.reshape(item['shape']);assert np.isfinite(z).all();return torch.from_numpy(z).cuda()
        def error(x,y):
            x=x.double();y=y.double();return float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(y).clamp_min(1e-12))
        with safe_open(b['source_weights'],framework='pt',device='cpu') as sf,torch.inference_mode():
            for site in b['sites']:
                p=next(s for s in parent['selections'] if s['site']==site)
                weight=sf.get_tensor(f'model.layers.{site}.mamba.norm.weight').float().cuda()
                original_out=sf.get_tensor(f'model.layers.{site}.mamba.out_proj.weight').cuda()
                ids={name:torch.tensor(p['layouts'][name]['channels'],device='cuda') for name in b['layouts']}
                combos=[(name,mode) for name in b['layouts'] for mode in b['modes']]
                gg={key:torch.zeros((384,384),device='cuda',dtype=torch.float64) for key in combos}
                cc={key:torch.zeros((384,2048),device='cuda',dtype=torch.float64) for key in combos}
                cache={};target_energy=0.
                def features(rec):
                    fields=rec['sites'][site]['fields'];y=raw(fields['y']);gate=raw(fields['gate']);ref=raw(fields['output']).bfloat16()
                    u=y*torch.nn.functional.silu(gate);den=u.square().mean(-1,keepdim=True)+eps
                    full=(u*torch.rsqrt(den)*weight).bfloat16()
                    baseline=error(torch.nn.functional.linear(full,original_out),ref)
                    assert baseline<=b['criteria']['reconstruction_relative_RMS'],('baseline',site,rec['id'],baseline)
                    ff={}
                    for name,index in ids.items():
                        us=u[:,index]
                        ff[(name,b['modes'][0])]=full[:,index]
                        ff[(name,b['modes'][1])]=(us*torch.rsqrt(us.square().mean(-1,keepdim=True)+eps)*weight[index]).bfloat16()
                        ff[(name,b['modes'][2])]=(us/p['constant_denominator']*weight[index]).bfloat16()
                    return ref,ff,baseline
                for rec in fit:
                    stage=f"FIT_moments/{site}/{rec['id']}";ref,ff,baseline=features(rec)
                    rr=ref.double();target_energy+=float(rr.square().sum()/len(rr))/24
                    for key,f in ff.items():
                        xx=f.double();gg[key]+=xx.T@xx/len(xx)/24;cc[key]+=xx.T@rr/len(xx)/24
                    cache[rec['id']]=dict(ref=ref.cpu(),features={key:x.cpu() for key,x in ff.items()},baseline=baseline)
                    event(stage='FIT_moments',site=site,id=rec['id']);del ref,ff,rr,xx,f;torch.cuda.empty_cache()
                fitted={};store={};diagnostics={};predicted_objective={};actual_objective={key:0. for key in combos}
                for key in combos:
                    stage=f'FIT_solve/{site}/{key}';g=gg[key].cpu();cross=cc[key].cpu();g=(g+g.T)/2
                    eigen,vectors=torch.linalg.eigh(g);lam=b['ridge_trace_fraction']*float(torch.trace(g))/384
                    assert lam>0 and float(eigen.min())>=-1e-10*float(eigen.max())
                    beta=(vectors/(eigen+lam))@(vectors.T@cross)
                    normal=g+lam*torch.eye(384,dtype=torch.float64)
                    resid=float(torch.linalg.vector_norm(normal@beta-cross)/torch.linalg.vector_norm(cross))
                    exported=beta.float();export_resid=float(torch.linalg.vector_norm(normal@exported.double()-cross)/torch.linalg.vector_norm(cross))
                    assert resid<=b['criteria']['normal_equation_F64'] and export_resid<=b['criteria']['normal_equation_export_F32']
                    assert torch.isfinite(exported).all()
                    fitted[key]=exported.cuda();tag='__'.join(key)
                    store.update({tag+'__G':g.numpy(),tag+'__C':cross.numpy(),tag+'__beta_F32':exported.numpy(),tag+'__eigenvalues':eigen.numpy()})
                    diagnostics[tag]=dict(ridge_lambda=lam,normal_residual_F64=resid,normal_residual_export_F32=export_resid,
                        regularized_condition=float((eigen.max()+lam)/(eigen.min()+lam)),F32_weight_L2=float(torch.linalg.matrix_norm(exported.double(),ord=2)))
                    eb=exported.double();predicted_objective[key]=float(target_energy-2*(eb*cross).sum()+(eb*(g@eb)).sum())
                    event(stage='FIT_solve',site=site,layout=key[0],mode=key[1],normal_residual=resid)
                path=a.directory/f'site{site:02d}.readouts.npz'
                with path.open('xb') as f:np.savez(f,**store)
                packet=dict(site=site,layouts=p['layouts'],constant_denominator=p['constant_denominator'],
                    fitted_readouts=extent(path),FIT_cases=24,diagnostics=diagnostics,objective_witness={})
                selections.append(packet);del gg,cc,store,g,cross,eigen,vectors,beta,normal,exported,eb
                for rec in records:
                    stage=f"evaluation/{site}/{rec['id']}";guard()
                    if rec['split']=='FIT':
                        c=cache.pop(rec['id']);ref=c['ref'].cuda();ff={key:f.cuda() for key,f in c['features'].items()};baseline=c['baseline'];del c
                    else:ref,ff,baseline=features(rec)
                    arms={name:{} for name in b['layouts']};label=torch.tensor(rec['positions'],device='cuda')
                    for key in combos:
                        prediction=ff[key].float()@fitted[key];assert torch.isfinite(prediction).all()
                        pr=prediction.double();rr=ref.double()
                        if rec['split']=='FIT':actual_objective[key]+=float((pr-rr).square().sum()/len(rr))/24
                        arms[key[0]][key[1]]=dict(output_relative_RMS=error(pr,rr),
                            centered_output_relative_RMS=error(pr-pr.mean(0),rr-rr.mean(0)),label_output_relative_RMS=error(pr[label],rr[label]))
                    row=dict(id=rec['id'],site=site,split=rec['split'],domain=rec['domain'],history=rec['history'],
                        reconstruction_relative_RMS=baseline,arms=arms)
                    rows.append(row);write(a.directory/f"{rec['id']}.site{site:02d}.metrics.json",row)
                    event(stage='evaluation',site=site,id=rec['id'],complete=len(rows));del ref,ff,pr,rr,prediction;torch.cuda.empty_cache()
                for key in combos:
                    delta=abs(predicted_objective[key]-actual_objective[key])/target_energy
                    assert delta<=b['criteria']['moment_objective_relative_delta']
                    packet['objective_witness']['__'.join(key)]=dict(predicted=predicted_objective[key],direct=actual_objective[key],delta_relative_to_target_energy=delta)
                write(a.directory/f'site{site:02d}.fit.json',packet)
                assert not cache;del fitted,weight,original_out;torch.cuda.empty_cache()
        aggregates={};domains={};gates={};viable=[]
        for site in b['sites']:
            aggregates[str(site)]={};domains[str(site)]={}
            for split in ('FIT','DEV'):
                sr=[r for r in rows if r['site']==site and r['split']==split];assert len(sr)==24
                aggregates[str(site)][split]={};domains[str(site)][split]={}
                for name,mode in combos:
                    aa={key:dict(mean=sum(r['arms'][name][mode][key] for r in sr)/24,
                        worst=max(r['arms'][name][mode][key] for r in sr)) for key in sr[0]['arms'][name][mode]}
                    dd={d:{key:dict(mean=sum(r['arms'][name][mode][key] for r in sr if r['domain']==d)/2,
                        worst=max(r['arms'][name][mode][key] for r in sr if r['domain']==d)) for key in aa} for d in sorted({r['domain'] for r in sr})}
                    aggregates[str(site)][split].setdefault(name,{})[mode]=aa;domains[str(site)][split].setdefault(name,{})[mode]=dd
        for name in b['layouts']:
            gates[name]={}
            for mode in b['modes'][1:]:
                sg={}
                for site in b['sites']:
                    g=aggregates[str(site)]['DEV'][name][mode];d=domains[str(site)]['DEV'][name][mode];c=b['criteria']
                    sg[str(site)]=dict(mean=g['output_relative_RMS']['mean']<=c['DEV_output_mean'],case_worst=g['output_relative_RMS']['worst']<=c['DEV_case_worst'],
                        domain_mean_worst=max(x['output_relative_RMS']['mean'] for x in d.values())<=c['DEV_domain_mean_worst'],centered_mean=g['centered_output_relative_RMS']['mean']<=c['DEV_centered_mean'])
                passed=all(all(v.values()) for v in sg.values());gates[name][mode]=dict(sites=sg,passed=passed)
                if passed:viable.append((max(aggregates[str(s)]['DEV'][name][mode]['output_relative_RMS']['mean'] for s in b['sites']),
                    sum(aggregates[str(s)]['DEV'][name][mode]['output_relative_RMS']['mean'] for s in b['sites']),name,mode))
        chosen=min(viable) if viable else None
        result=dict(schema='ORIGINAL_FALCON_CHANNEL_TRANSPORT_RESULT_V1',experiment=b['experiment'],freeze=a.freeze,binding_sha256=a.binding_sha,
            decision='READOUT_REFIT_LOCAL_LAYOUT_PROVISIONALLY_SELECTED' if chosen else 'CHANNEL_READOUT_REFIT_BUDGET_FAIL',
            selected_layout=dict(layout=chosen[2],mode=chosen[3]) if chosen else None,selections=selections,records=rows,aggregates=aggregates,domains=domains,gates=gates,
            source_forwards=0,optimizer_updates=0,native_calls=0,closed_form_FIT_readout_fits=18,channel_reselections=0,
            quality_admission=False,speed_admission=False,scope='Fixed actual source-state256 channels/gate/features;FIT-only linear ridge output repair. '
                'Full source norm is oracle diagnostic;no P256/state96/depth/ternary/SSM-generator/native/chatbot admission.',
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            worker_OS_peak_snapshot=proc.memory_info().peak_wset,elapsed_seconds=time.monotonic()-start)
        assert len(rows)==144 and len(selections)==3;write(a.out,result);event(stage='complete',decision=result['decision'],sha256=sha(a.out))
    except BaseException as e:
        write(a.directory/'first_fault.json',dict(stage=stage,error=repr(e),selections=selections,completed_measurements=rows,
            elapsed_seconds=time.monotonic()-start,GPU_allocated_peak=torch.cuda.max_memory_allocated() if torch is not None else None,
            GPU_reserved_peak=torch.cuda.max_memory_reserved() if torch is not None else None,
            OS_peak=proc.memory_info().peak_wset if proc is not None else None));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--bind',action='store_true');m.add_argument('--launch',action='store_true');m.add_argument('--worker',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.bind:bind(a)
    elif a.launch:launch(a)
    else:worker(a)
