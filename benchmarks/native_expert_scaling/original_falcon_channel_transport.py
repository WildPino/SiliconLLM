"""Stored-only output-aware384-channel/head selection and norm controls.

Keeps source state256/gate/generators; distinct from rank96 experiment.
No source-model, optimizer, native or chatbot generation calls.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT/'benchmarks/native_expert_scaling'
DOC = ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0, str(B))
from chatbot_falcon_usability import SITE, sha, write
from original_falcon_whole_recovery import extent, check_inputs, memory_reader
sys.path.insert(0, str(SITE))


def bind(a):
    capture = DOC/'original_falcon_recurrent_adopted_result_20261009.json'
    assert sha(capture) == 'bdec178a948e49bbec1d9eb4c01076bec9c182be8347306eb5dc02e0c873c849'
    r = json.loads(capture.read_bytes())
    assert r['cases']==48 and r['sites_per_case']==24 and r['stored_finite_extent_hash_adoption']
    old = json.loads(Path(r['adoption_binding']['path']).read_bytes())
    source = Path(old['source']); sites = [0,12,23]
    files = [Path(__file__), B/'chatbot_falcon_usability.py', B/'chatbot_falcon_usability_launch.py',
        B/'original_falcon_whole_recovery.py', capture, Path(sys.executable), source/'model.safetensors',
        source/'config.json', DOC/'ORIGINAL_FALCON_CHANNEL_TRANSPORT_PROTOCOL_20261009.md',
        DOC/'original_falcon_recurrent_projection_finish_result_20261009.json',
        DOC/'original_falcon_recurrent_projection_audit_result_20261009.json']
    for rec in r['records']:
        assert len(rec['positions']) and rec['history']>max(rec['positions'])
        for site in sites:
            for name in ('y','gate','output'):
                item = rec['sites'][site]['fields'][name]
                assert extent(item['path']) == {k:item[k] for k in ('path','bytes','sha256')}
                files.append(Path(item['path']))
    for split in ('FIT','DEV'):
        rows = [i for i in r['records'] if i['split']==split]
        assert len(rows)==24 and len({i['domain'] for i in rows})==12
    files += [SITE/'torch'/p for p in ('__init__.py','_C.cp312-win_amd64.pyd',
        'lib/torch_cpu.dll','lib/torch_cuda.dll','cuda/__init__.py')]
    files += [SITE/'numpy/__init__.py', SITE/'psutil/__init__.py', SITE/'safetensors/torch.py']
    files += [ROOT/p for p in ('benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json',
        'benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md')]
    write(a.out, dict(schema='ORIGINAL_FALCON_CHANNEL_TRANSPORT_BINDING_V1',
        python=str(Path(sys.executable).resolve()),worker_path=str(Path(__file__).resolve()),
        capture=str(capture.resolve()),source_weights=str((source/'model.safetensors').resolve()),
        source_config=str((source/'config.json').resolve()),sites=sites,active_channels=384,
        layouts=['all48_heads_x8','selected6_heads_x64'],
        modes=['full_source_denominator','subset_RMS_calibrated','constant_RMS_calibrated'],
        selection='FIT only,equal case weights;FP32 CUDA contractions with TF32 disabled,F64 case accumulation;'
            'greedy fixed-unit output residual reduction,lowest index breaks ties;8 per head or6 whole heads.',
        criteria=dict(reconstruction_relative_RMS=1e-4,moment_objective_relative_delta=1e-4,
            DEV_output_mean=.10,DEV_case_worst=.20,DEV_domain_mean_worst=.20,DEV_centered_mean=.10),
        limits=dict(seconds=1800,reserve_seconds=90,OS_bytes=6<<30,GPU_allocated_bytes=4<<30,
            GPU_reserved_bytes=5<<30,output_bytes=32<<20),
        inputs=[extent(p) for p in dict.fromkeys(files)],
        runtime_binding_scope='All144 y/gate/output packets,pinned coefficients/config,actual adopted capture,'
            'principal runtime binaries,worker/protocol and unchanged original engine;not a full DLL tree.'))
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(dict.fromkeys(files)))),flush=True)


def launch(a):
    import psutil
    start=time.monotonic(); assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes());assert b['schema']=='ORIGINAL_FALCON_CHANNEL_TRANSPORT_BINDING_V1'
    assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    proc=psutil.Process();proc.cpu_affinity([11]);own={proc.pid,*(p.pid for p in proc.parents())}
    for p in psutil.process_iter(['name','cmdline']):
        name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
        if p.pid in own:continue
        if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():continue
        assert not name.startswith(('python','gcc','clang','engine','packed_original')),('overlap',p.pid,name)
    log=a.out.with_suffix('.worker.log');terminal=a.out.with_suffix('.terminal.json')
    assert not a.directory.exists() and not any(p.exists() for p in (a.out,log,terminal,a.out.with_suffix('.launcher_failure.json')))
    worker=None;peak=0;reader=memory_reader()
    record=dict(freeze=a.freeze,binding_sha256=a.binding_sha,launcher_pid=proc.pid)
    def memory():
        nonlocal peak
        peak=max(peak,reader(worker))
    def guard():
        assert time.monotonic()-start<=b['limits']['seconds'],'family deadline'
        assert peak+proc.memory_info().peak_wset<=b['limits']['OS_bytes'],'family OS cap'
        assert log.stat().st_size<=4<<20,'log cap'
    try:
        check_inputs(b)
        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
            OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        argv=[b['python'],'-I','-S','-B','-X','utf8',b['worker_path'],'--worker','--binding',str(a.binding.resolve()),
            '--binding-sha',a.binding_sha,'--freeze',a.freeze,'--directory',str(a.directory.resolve()),'--out',str(a.out.resolve())]
        record['command']=argv
        with log.open('xb') as stream:
            worker=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,creationflags=8)
            record.update(worker_pid=worker.pid,worker_creation_time=psutil.Process(worker.pid).create_time());offset=0
            while worker.poll() is None:
                memory();guard()
                try:assert not psutil.Process(worker.pid).children(recursive=True),'unexpected child'
                except psutil.NoSuchProcess:pass
                with log.open('rb') as f:
                    f.seek(offset);chunk=f.read();end=chunk.rfind(b'\n')+1
                    if end:print(chunk[:end].decode('utf8',errors='replace'),end='',flush=True);offset+=end
                time.sleep(.1)
        memory();guard();record.update(exit_code=worker.returncode,worker_OS_peak_through_exit=peak,
            launcher_OS_peak_snapshot=proc.memory_info().peak_wset)
        assert worker.returncode==0,log.read_text(errors='replace')[-6000:]
        check_inputs(b);r=json.loads(a.out.read_bytes())
        assert r['schema']=='ORIGINAL_FALCON_CHANNEL_TRANSPORT_RESULT_V1' and len(r['records'])==144 and len(r['selections'])==3
        assert r['source_forwards']==r['optimizer_updates']==r['native_calls']==0
        assert r['GPU_allocated_peak']<=b['limits']['GPU_allocated_bytes'] and r['GPU_reserved_peak']<=b['limits']['GPU_reserved_bytes']
        outputs=[extent(p) for p in sorted(a.directory.iterdir()) if p.is_file()]
        assert sum(i['bytes'] for i in outputs)<=b['limits']['output_bytes']
        record.update(elapsed_seconds=time.monotonic()-start,result_sha256=sha(a.out),output_files=outputs,
            decision=r['decision'],resource_gates=True)
        write(terminal,record);print(json.dumps(dict(terminal=str(terminal),worker_pid=worker.pid,
            exit_code=worker.returncode,seconds=record['elapsed_seconds'],decision=r['decision'])),flush=True)
    except BaseException as error:
        if worker is not None:
            if worker.poll() is None:worker.kill();worker.wait()
            memory();record.update(exit_code=worker.returncode,worker_OS_peak_through_exit=peak)
        record.update(fault=repr(error),elapsed_seconds=time.monotonic()-start,
            launcher_OS_peak_snapshot=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.launcher_failure.json'),record);raise


def worker(a):
    start=time.monotonic();assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes());a.directory.mkdir(exist_ok=False)
    stage='startup';selections=[];measured=[];torch=None;proc=None
    try:
        import numpy as np
        import psutil
        import torch
        from safetensors import safe_open
        assert (torch.__version__,np.__version__,psutil.__version__)==('2.6.0+cu124','2.4.6','7.2.2')
        torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        proc=psutil.Process();proc.cpu_affinity(list(range(6)))
        def guard():
            lim=b['limits']
            assert time.monotonic()-start<=lim['seconds']-lim['reserve_seconds'],'worker reserve'
            assert proc.memory_info().peak_wset<=lim['OS_bytes'],'OS cap'
            assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'],'GPU allocated cap'
            assert torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes'],'GPU reserved cap'
            assert not proc.children(recursive=True),'unexpected child'
        def event(**fields):
            guard();print(json.dumps(dict(seconds=time.monotonic()-start,**fields)),flush=True)
        records=json.loads(Path(b['capture']).read_bytes())['records']
        fit=[r for r in records if r['split']=='FIT'];dev=[r for r in records if r['split']=='DEV']
        config=json.loads(Path(b['source_config']).read_bytes());eps=config['rms_norm_eps']
        assert not config['mamba_proj_bias'] and not config['mamba_norm_before_gate'] and config['mamba_n_groups']==1
        def raw(item):
            if item['dtype']=='BF16':
                z=(np.fromfile(item['path'],dtype='<u2').astype('<u4')<<16).view('<f4')
            else:
                assert item['dtype']=='F32';z=np.fromfile(item['path'],dtype='<f4')
            z=z.reshape(item['shape']);assert np.isfinite(z).all()
            return torch.from_numpy(z).cuda()
        def operands(rec,site,weight):
            fields=rec['sites'][site]['fields']
            yy=raw(fields['y']);gate=raw(fields['gate']);ref=raw(fields['output']).bfloat16()
            u=yy*torch.nn.functional.silu(gate)
            denom=torch.sqrt(u.square().mean(-1,keepdim=True)+eps)
            full=(u*torch.rsqrt(u.square().mean(-1,keepdim=True)+eps)*weight).bfloat16()
            assert torch.isfinite(full).all()
            return u,denom,full,ref
        def norm_error(value,ref):
            value=value.double();ref=ref.double()
            return float(torch.sqrt((value-ref).square().mean())/torch.sqrt(ref.square().mean()).clamp_min(1e-12))
        def metrics(value,ref,positions):
            x=value.double();y=ref.double()
            label=torch.tensor(positions,device='cuda')
            return dict(output_relative_RMS=norm_error(x,y),centered_output_relative_RMS=norm_error(x-x.mean(0),y-y.mean(0)),
                label_output_relative_RMS=norm_error(x[label],y[label]))
        with safe_open(b['source_weights'],framework='pt',device='cpu') as sf,torch.inference_mode():
            for site in b['sites']:
                prefix=f'model.layers.{site}.mamba.'
                weight=sf.get_tensor(prefix+'norm.weight').cuda().float()
                out_weight=sf.get_tensor(prefix+'out_proj.weight').cuda();ow=out_weight.float()
                assert tuple(ow.shape)==(2048,3072) and len(weight)==3072
                covariance=np.zeros((3072,3072),dtype='f8');target_b=np.zeros(3072,dtype='f8')
                target_energy=0.;denom_square=0.
                for rec in fit:
                    stage=f"moments/site{site:02d}/{rec['id']}";guard()
                    u,d,f,ref=operands(rec,site,weight);ff=f.float();oo=ref.float()
                    covariance+=(ff.T@ff/len(ff)).cpu().numpy().astype('f8')/24
                    target_b+=((ff.T@oo/len(ff))*ow.T).sum(-1).cpu().numpy().astype('f8')/24
                    target_energy+=float(oo.double().square().sum()/len(oo))/24
                    denom_square+=float(d.double().square().mean())/24
                    event(stage='FIT_moments',site=site,id=rec['id'])
                    del u,d,f,ref,ff,oo;torch.cuda.empty_cache()
                ww=(ow.T@ow).cpu().numpy().astype('f8')
                gram=covariance*ww;gram=(gram+gram.T)/2;del covariance,ww
                assert np.isfinite(gram).all() and np.isfinite(target_b).all() and target_energy>0 and denom_square>0
                diag=np.diag(gram).copy();residual_b=target_b.copy();used=np.zeros(3072,dtype=bool)
                quotas=np.zeros(48,dtype='i8');channel_order=[];channel_gains=[]
                for step in range(384):
                    gain=2*residual_b-diag;gain[used | (quotas[np.arange(3072)//64]>=8)]=-np.inf
                    chosen=int(np.argmax(gain));assert math.isfinite(float(gain[chosen]))
                    channel_order.append(chosen);channel_gains.append(float(gain[chosen]))
                    used[chosen]=True;quotas[chosen//64]+=1;residual_b-=gram[chosen]
                assert np.array_equal(quotas,np.full(48,8))
                head_gram=gram.reshape(48,64,48,64).sum(axis=(1,3))
                head_b=target_b.reshape(48,64).sum(1);hb=head_b.copy();hu=np.zeros(48,dtype=bool)
                head_order=[];head_gains=[]
                for step in range(6):
                    gain=2*hb-np.diag(head_gram);gain[hu]=-np.inf
                    chosen=int(np.argmax(gain));head_order.append(chosen);head_gains.append(float(gain[chosen]))
                    hu[chosen]=True;hb-=head_gram[chosen]
                ids={'all48_heads_x8':np.sort(np.asarray(channel_order,dtype='i8')),
                    'selected6_heads_x64':np.sort(np.concatenate([np.arange(h*64,(h+1)*64) for h in head_order]))}
                path=a.directory/f'site{site:02d}.selection_moments.npz'
                with path.open('xb') as f:
                    np.savez(f,selected_gram_rows=gram[channel_order],initial_b=target_b,diag=diag,
                        channel_order=np.array(channel_order),channel_gains=np.array(channel_gains),
                        head_gram=head_gram,head_b=head_b,head_order=np.array(head_order),head_gains=np.array(head_gains),
                        target_energy=np.array(target_energy),constant_denominator=np.array(math.sqrt(denom_square)))
                packet=dict(site=site,moments=extent(path),constant_denominator=math.sqrt(denom_square),
                    layouts={name:dict(channels=index.tolist(),head_count=len(set((index//64).tolist()))) for name,index in ids.items()},
                    FIT_cases=24,FIT_positions=sum(r['history'] for r in fit),
                    moment_method=b['selection'],calibration={},moment_objective_witness={})
                torch_ids={name:torch.from_numpy(index).cuda() for name,index in ids.items()}
                selected_weights={name:out_weight[:,index] for name,index in torch_ids.items()}
                # Cache every FIT restricted BF16 output once; evaluate cached outputs after fitting scalars.
                cache={};stats={name:{mode:dict(xx=0.,xy=0.) for mode in b['modes'][1:]} for name in ids}
                objective_actual={name:0. for name in ids}
                reconstruction_failures=[]
                def outputs(rec):
                    u,d,full,ref=operands(rec,site,weight)
                    reconstructed=torch.nn.functional.linear(full,out_weight)
                    err=norm_error(reconstructed,ref)
                    assert err<=b['criteria']['reconstruction_relative_RMS'],('baseline',rec['id'],site,err)
                    row_outputs={};energy={};denominators={}
                    for name,index in torch_ids.items():
                        us=u[:,index];ds=torch.sqrt(us.square().mean(-1,keepdim=True)+eps)
                        fsub=(us*torch.rsqrt(us.square().mean(-1,keepdim=True)+eps)*weight[index]).bfloat16()
                        fconst=(us/packet['constant_denominator']*weight[index]).bfloat16()
                        row_outputs[name]={
                            'full_source_denominator':torch.nn.functional.linear(full[:,index],selected_weights[name]),
                            'subset_RMS_calibrated':torch.nn.functional.linear(fsub,selected_weights[name]),
                            'constant_RMS_calibrated':torch.nn.functional.linear(fconst,selected_weights[name])}
                        fractions=us.double().square().sum(-1)/u.double().square().sum(-1).clamp_min(1e-12)
                        energy[name]=dict(mean=float(fractions.mean()),min=float(fractions.min()),max=float(fractions.max()))
                        denominators[name]=ds
                    return ref,row_outputs,energy,denominators,d,err,full
                for rec in fit:
                    stage=f"calibration/site{site:02d}/{rec['id']}";guard()
                    ref,outs,energy,ds,d,err,full=outputs(rec)
                    for name in ids:
                        for mode in b['modes'][1:]:
                            val=outs[name][mode].double();rr=ref.double()
                            stats[name][mode]['xx']+=float(val.square().sum()/len(val))/24
                            stats[name][mode]['xy']+=float((val*rr).sum()/len(val))/24
                        # Independent direct FP32 objective, matching moment contraction convention.
                        prediction=full[:,torch_ids[name]].float()@ow[:,torch_ids[name]].T
                        objective_actual[name]+=float((prediction.double()-ref.double()).square().sum()/len(ref))/24
                    cache[rec['id']]=dict(ref=ref.cpu(),outputs={name:{mode:v.cpu() for mode,v in vv.items()} for name,vv in outs.items()},
                        energy=energy,denominators={name:dd.cpu() for name,dd in ds.items()},full_denominator=d.cpu(),reconstruction=err)
                    event(stage='FIT_calibration',site=site,id=rec['id'])
                    del ref,outs,ds,d,full,val,rr,prediction;torch.cuda.empty_cache()
                for name,index in ids.items():
                    for mode,s in stats[name].items():
                        assert s['xx']>0
                        alpha=max(0.,s['xy']/s['xx']);assert math.isfinite(alpha)
                        packet['calibration'].setdefault(name,{})[mode]=dict(alpha=alpha,**s)
                    formula=float(target_energy-2*target_b[index].sum()+gram[np.ix_(index,index)].sum())
                    error=abs(formula-objective_actual[name])/target_energy
                    assert error<=b['criteria']['moment_objective_relative_delta'],('moment objective',name,site,error)
                    packet['moment_objective_witness'][name]=dict(moment_objective=formula,direct_FP32_objective=objective_actual[name],
                        delta_relative_to_target_energy=error)
                # Independent F64 replay of selected greedy steps from durable sufficient-statistic rows only.
                with np.load(path) as z:
                    rb=z['initial_b'].copy();mask=np.zeros(3072,dtype=bool);counts=np.zeros(48,dtype='i8')
                    for k,i in enumerate(z['channel_order']):
                        scores=2*rb-z['diag'];scores[mask | (counts[np.arange(3072)//64]>=8)]=-np.inf
                        assert int(np.argmax(scores))==int(i) and float(scores[i])==float(z['channel_gains'][k])
                        mask[i]=True;counts[i//64]+=1;rb-=z['selected_gram_rows'][k]
                    rb=z['head_b'].copy();mask=np.zeros(48,dtype=bool)
                    for k,h in enumerate(z['head_order']):
                        scores=2*rb-np.diag(z['head_gram']);scores[mask]=-np.inf
                        assert int(np.argmax(scores))==int(h) and float(scores[h])==float(z['head_gains'][k])
                        mask[h]=True;rb-=z['head_gram'][h]
                packet['stored_selection_replay_exact']=True;selections.append(packet)
                write(a.directory/f'site{site:02d}.selection.json',packet)
                event(stage='selection_frozen',site=site,calibration=packet['calibration'])
                del gram,target_b,diag,residual_b,head_gram,head_b,hb,rb,scores
                for rec in records:
                    stage=f"evaluation/site{site:02d}/{rec['id']}";guard()
                    if rec['split']=='FIT':
                        cached=cache.pop(rec['id']);ref=cached['ref'].cuda()
                        outs={name:{mode:v.cuda() for mode,v in vv.items()} for name,vv in cached['outputs'].items()}
                        energy=cached['energy'];ds={name:dd.cuda() for name,dd in cached['denominators'].items()}
                        d=cached['full_denominator'].cuda();err=cached['reconstruction']
                        del cached
                    else:
                        ref,outs,energy,ds,d,err,full=outputs(rec);del full
                    arms={}
                    for name in ids:
                        arms[name]={}
                        for mode in b['modes']:
                            alpha=1. if mode==b['modes'][0] else packet['calibration'][name][mode]['alpha']
                            val=outs[name][mode].float()*alpha
                            item=metrics(val,ref,rec['positions'])
                            estimate=ds[name]/alpha if mode==b['modes'][1] and alpha>0 else (
                                torch.full_like(d,packet['constant_denominator']/alpha) if mode==b['modes'][2] and alpha>0 else None)
                            item['implied_full_denominator_relative_RMS']=norm_error(estimate,d) if estimate is not None else (0. if mode==b['modes'][0] else None)
                            assert all(v is None or math.isfinite(v) for v in item.values())
                            arms[name][mode]=item
                    row=dict(id=rec['id'],site=site,split=rec['split'],domain=rec['domain'],history=rec['history'],
                        reconstruction_relative_RMS=err,selected_gated_energy_fraction=energy,arms=arms)
                    measured.append(row);write(a.directory/f"{rec['id']}.site{site:02d}.metrics.json",row)
                    event(stage='evaluation',site=site,id=rec['id'],complete=len(measured))
                    del ref,outs,ds,d,val,estimate;torch.cuda.empty_cache()
                assert not cache
                del weight,out_weight,ow,selected_weights,torch_ids;torch.cuda.empty_cache()
        aggregates={};gates={};domains={}
        for site in b['sites']:
            aggregates[str(site)]={};domains[str(site)]={}
            for split in ('FIT','DEV'):
                rows=[r for r in measured if r['site']==site and r['split']==split];assert len(rows)==24
                aggregate={};domain_result={}
                for name in b['layouts']:
                    aggregate[name]={};domain_result[name]={}
                    for mode in b['modes']:
                        keys=rows[0]['arms'][name][mode]
                        aggregate[name][mode]={key:dict(mean=sum(r['arms'][name][mode][key] for r in rows)/24,
                            worst=max(r['arms'][name][mode][key] for r in rows)) for key in keys if all(r['arms'][name][mode][key] is not None for r in rows)}
                        domain_result[name][mode]={domain:{key:dict(mean=sum(r['arms'][name][mode][key] for r in rows if r['domain']==domain)/2,
                            worst=max(r['arms'][name][mode][key] for r in rows if r['domain']==domain)) for key in ('output_relative_RMS','centered_output_relative_RMS','label_output_relative_RMS')}
                            for domain in sorted({r['domain'] for r in rows})}
                aggregates[str(site)][split]=aggregate;domains[str(site)][split]=domain_result
        viable=[]
        for name in b['layouts']:
            gates[name]={}
            for mode in b['modes'][1:]:
                site_gates={}
                for site in b['sites']:
                    g=aggregates[str(site)]['DEV'][name][mode];dd=domains[str(site)]['DEV'][name][mode];c=b['criteria']
                    site_gates[str(site)]=dict(mean=g['output_relative_RMS']['mean']<=c['DEV_output_mean'],
                        case_worst=g['output_relative_RMS']['worst']<=c['DEV_case_worst'],
                        domain_mean_worst=max(d['output_relative_RMS']['mean'] for d in dd.values())<=c['DEV_domain_mean_worst'],
                        centered_mean=g['centered_output_relative_RMS']['mean']<=c['DEV_centered_mean'])
                passed=all(all(g.values()) for g in site_gates.values());gates[name][mode]=dict(sites=site_gates,passed=passed)
                if passed:
                    viable.append((max(aggregates[str(s)]['DEV'][name][mode]['output_relative_RMS']['mean'] for s in b['sites']),
                        sum(aggregates[str(s)]['DEV'][name][mode]['output_relative_RMS']['mean'] for s in b['sites']),name,mode))
        chosen=min(viable) if viable else None
        result=dict(schema='ORIGINAL_FALCON_CHANNEL_TRANSPORT_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
            decision='ONE_LOCAL_CHANNEL_LAYOUT_PROVISIONALLY_SELECTED' if chosen else 'CHANNEL_HEAD_REPRESENTATION_BUDGET_FAIL',
            selected_layout=dict(layout=chosen[2],mode=chosen[3]) if chosen else None,
            selections=selections,records=measured,aggregates=aggregates,domains=domains,gates=gates,
            source_forwards=0,optimizer_updates=0,native_calls=0,state_rank_experiment_replays=0,
            local_scope='Actual state256/source y/source gate functions retained;384-column output restriction only. '
                'Full denominator is an optimistic diagnostic;subset norm is an unimplemented core extension;'
                'constant norm omits runtime norm. FIT scalars applied in F32 after BF16 out_proj. '
                'No dual96/P256/depth/FFN combination,chatbot quality,native parity or speed admission.',
            quality_admission=False,speed_admission=False,worker_OS_peak_snapshot=proc.memory_info().peak_wset,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.monotonic()-start)
        assert len(measured)==144 and len(selections)==3
        write(a.out,result);event(stage='complete',decision=result['decision'],sha256=sha(a.out))
    except BaseException as error:
        write(a.directory/'first_fault.json',dict(stage=stage,error=repr(error),completed_selections=selections,
            completed_measurements=measured,elapsed_seconds=time.monotonic()-start,
            GPU_allocated_peak=torch.cuda.max_memory_allocated() if torch is not None else None,
            GPU_reserved_peak=torch.cuda.max_memory_reserved() if torch is not None else None,
            OS_peak=proc.memory_info().peak_wset if proc is not None else None))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--bind',action='store_true');mode.add_argument('--launch',action='store_true');mode.add_argument('--worker',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.bind:bind(a)
    elif a.launch:launch(a)
    else:worker(a)
