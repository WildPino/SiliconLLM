"""FIT-only dual channel bases; fixed/adaptive384 actual packed SSD histories.

Independent-time read/write objective is a diagnostic, not chatbot loss.
Uses retained operands and immutable held launcher/scan. No model forwards.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]; B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'; sys.path.insert(0,str(B))
from original_falcon_channel_transport import SITE,sha,write,extent,launch
from original_falcon_mixed_history import scan
sys.path.insert(0,str(SITE))


def bind(a):
    parent=DOC/'original_falcon_mixed_history_binding_20261009.json'
    b=json.loads(parent.read_bytes())
    for i in b['inputs']: assert extent(i['path'])==i
    files=[Path(__file__),parent,DOC/'ORIGINAL_FALCON_BALANCED_HEADS_PROTOCOL_20261009.md',
        DOC/'original_falcon_mixed_history_result_20261009.json',
        DOC/'original_falcon_mixed_history_result_20261009.terminal.json']
    b.update(worker_path=str(Path(__file__).resolve()),experiment='BALANCED_HEAD_CHANNELS_V1',
        layouts=['fixed8','adaptive384'],selection='FIT-only unnormalized equal-case GY/GM; dual SVD; min1/head and336 largest remaining squared singular values',
        limits=dict(seconds=1800,reserve_seconds=90,OS_bytes=6<<30,GPU_allocated_bytes=4<<30,
            GPU_reserved_bytes=5<<30,output_bytes=256<<20),
        inputs=list({i['path']:i for i in b['inputs']+[extent(p) for p in files]}.values()))
    b.pop('rank_per_head',None)
    b['criteria'].update(PSD_relative_negative=1e-10,minimum_retained_singular=1e-12,
        biorthogonality=1e-8,independent_kernel_relative=1e-8,tail_relative_delta=1e-10)
    write(a.out,b);print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(b['inputs']))),flush=True)


def worker(a):
    start=time.monotonic(); assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes()); assert b['experiment']=='BALANCED_HEAD_CHANNELS_V1'
    a.directory.mkdir(exist_ok=False);stage='startup';packets=[];rows=[];scalar=[];torch=None;proc=None
    try:
        import numpy as np
        import psutil
        import torch
        from safetensors import safe_open
        from transformers.models.falcon_h1 import modeling_falcon_h1 as code
        assert (torch.__version__,np.__version__,psutil.__version__)==('2.6.0+cu124','2.4.6','7.2.2')
        torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        proc=psutil.Process();proc.cpu_affinity(list(range(6)));criteria=b['criteria']
        def guard():
            assert time.monotonic()-start<=b['limits']['seconds']-b['limits']['reserve_seconds'],'worker reserve'
            assert proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True),'OS/children'
            assert torch.cuda.max_memory_allocated()<=b['limits']['GPU_allocated_bytes'],'GPU allocated'
            assert torch.cuda.max_memory_reserved()<=b['limits']['GPU_reserved_bytes'],'GPU reserved'
        def event(**kw):
            guard();print(json.dumps(dict(seconds=time.monotonic()-start,**kw)),flush=True)
        records=json.loads(Path(b['capture']).read_bytes())['records'];fit=[r for r in records if r['split']=='FIT']
        assert len(fit)==24
        eps=json.loads(Path(b['source_config']).read_bytes())['rms_norm_eps']
        def raw_np(item):
            if item['dtype']=='BF16':z=(np.fromfile(item['path'],dtype='<u2').astype('<u4')<<16).view('<f4')
            else:assert item['dtype']=='F32';z=np.fromfile(item['path'],dtype='<f4')
            z=z.reshape(item['shape']);assert np.isfinite(z).all();return z
        def raw(item):return torch.from_numpy(raw_np(item)).cuda()
        def error(x,y):
            return float(torch.linalg.vector_norm(x.double()-y.double())/torch.linalg.vector_norm(y.double()).clamp_min(1e-12))
        def roots(g):
            e,v=torch.linalg.eigh((g+g.transpose(-2,-1))/2)
            scale=e.abs().amax(-1).clamp_min(1e-300)
            assert float((-e.amin(-1)/scale).max())<=criteria['PSD_relative_negative']
            return (v*e.clamp_min(0).sqrt()[:,None,:])@v.transpose(-2,-1),e
        def numpy_root(g):
            e,v=np.linalg.eigh((g+g.T)/2)
            assert -e.min()/max(abs(e).max(),1e-300)<=criteria['PSD_relative_negative']
            return (v*np.sqrt(np.maximum(e,0)))@v.T
        maps={}
        with safe_open(b['source_weights'],framework='pt',device='cpu') as sf,torch.inference_mode():
            for site in b['sites']:
                prefix=f'model.layers.{site}.mamba.'
                weight=sf.get_tensor(prefix+'norm.weight').cuda().float();ow=sf.get_tensor(prefix+'out_proj.weight').cuda()
                oh=ow.double().reshape(2048,48,64).permute(1,0,2)
                og=oh.transpose(-2,-1)@oh
                gy=torch.zeros((48,64,64),device='cuda',dtype=torch.float64);gq=torch.zeros_like(gy)
                for rec in fit:
                    stage=f"FIT_basis/{site}/{rec['id']}";guard();f=rec['sites'][site]['fields'];n=rec['history']
                    y=raw(f['y']);gate=raw(f['gate']);sg=torch.nn.functional.silu(gate)
                    inv=torch.rsqrt((y*sg).square().mean(-1,keepdim=True)+eps)
                    q=(sg*inv*weight).reshape(n,48,64).permute(1,0,2).double()
                    yh=y.reshape(n,48,64).permute(1,0,2).double()
                    gy+=(yh.transpose(-2,-1)@yh)/(24*n);gq+=(q.transpose(-2,-1)@q)/(24*n)
                    del y,gate,sg,inv,q,yh;event(stage='FIT_basis',site=site,id=rec['id']);torch.cuda.empty_cache()
                gm=(og*gq).cpu();gy=gy.cpu();sy,ey=roots(gy);sm,em=roots(gm)
                hh=sm@sy;uu,ss,qh=torch.linalg.svd(hh,full_matrices=False);qq=qh.transpose(-2,-1)
                ranks={'fixed8':[8]*48};adaptive=[1]*48;scores=ss.numpy()**2
                for _ in range(336):
                    head=min((h for h in range(48) if adaptive[h]<64),key=lambda h:(-scores[h,adaptive[h]],h))
                    adaptive[head]+=1
                # Independent global marginal sorting must preserve all per-head prefixes.
                chosen=sorted([(scores[h,r],h,r) for h in range(48) for r in range(1,64)],key=lambda x:(-x[0],x[1],x[2]))[:336]
                independent=[1+sum(h==j for _,h,_ in chosen) for j in range(48)]
                assert adaptive==independent and sum(adaptive)==384
                assert all({r for _,j,r in chosen if j==h}==set(range(1,adaptive[h])) for h in range(48))
                ranks['adaptive384']=adaptive
                arrays=dict(GY=gy.numpy(),GM=gm.numpy(),GQ=gq.cpu().numpy(),OG=og.cpu().numpy(),
                    sqrt_GY=sy.numpy(),sqrt_GM=sm.numpy(),GY_eigenvalues=ey.numpy(),GM_eigenvalues=em.numpy(),
                    U=uu.numpy(),Q=qq.numpy(),singular_values=ss.numpy())
                variants={};maps[site]={}
                for name,rs in ranks.items():
                    vv=[];ww=[];audits=[];offsets=[0]
                    for h,r in enumerate(rs):
                        stage=f'basis_audit/{site}/{name}/{h}';guard()
                        sigma=ss[h,:r];assert float(sigma.min())>criteria['minimum_retained_singular']
                        v=(sy[h]@qq[h,:,:r])/sigma.sqrt();w=(sm[h]@uu[h,:,:r])/sigma.sqrt()
                        bio=float((w.T@v-torch.eye(r,dtype=torch.float64)).abs().max());assert bio<=criteria['biorthogonality']
                        nsy=numpy_root(arrays['GY'][h]);nsm=numpy_root(arrays['GM'][h]);nk=nsm@nsy
                        nu,ns,nqt=np.linalg.svd(nk,full_matrices=False)
                        reference=(nu[:,:r]*ns[:r])@nqt[:r]
                        actual=nsm@v.numpy()@w.numpy().T@nsy
                        kerr=float(np.linalg.norm(actual-reference)/max(np.linalg.norm(reference),1e-300))
                        terr=float(abs(np.linalg.norm(nk-actual)**2-np.sum(ns[r:]**2))/max(np.linalg.norm(nk)**2,1e-300))
                        assert kerr<=criteria['independent_kernel_relative'] and terr<=criteria['tail_relative_delta']
                        audits.append(dict(head=h,rank=r,biorthogonality=bio,kernel_relative=kerr,tail_relative_delta=terr,
                            minimum_retained_singular=float(sigma.min()),V_spectral_norm=float(torch.linalg.matrix_norm(v,ord=2)),
                            W_spectral_norm=float(torch.linalg.matrix_norm(w,ord=2)),weighted_tail_energy=float((ss[h,r:]**2).sum())))
                        arrays[f'{name}_V_{h}']=v.numpy();arrays[f'{name}_W_{h}']=w.numpy()
                        vv.append(v.float().cuda());ww.append(w.float().cuda());offsets.append(offsets[-1]+r)
                    groups={r:[h for h in range(48) if rs[h]==r] for r in sorted(set(rs))}
                    maps[site][name]=dict(V=vv,W=ww,ranks=rs,offsets=offsets,groups=groups)
                    variants[name]=dict(ranks=rs,offsets=offsets,groups=groups,audits=audits,
                        weighted_pair_tail_energy=sum(x['weighted_tail_energy'] for x in audits),carried_channels=sum(rs))
                path=a.directory/f'site{site:02d}.balanced_bases.npz'
                with path.open('xb') as f:np.savez(f,**arrays)
                packet=dict(site=site,basis=extent(path),FIT_cases=24,FIT_positions=sum(r['history'] for r in fit),
                    variants=variants,adaptive_rank_independent_check=True,
                    GY_negative_eigenvalues=int((ey<0).sum()),GM_negative_eigenvalues=int((em<0).sum()))
                packets.append(packet);write(a.directory/f'site{site:02d}.basis.json',packet)
                event(stage='basis_frozen',site=site,adaptive_ranks=adaptive)
                del weight,ow,oh,og,gy,gq,gm,sy,sm,ey,em,hh,uu,ss,qh,qq,arrays;torch.cuda.empty_cache()
            for site in b['sites']:
                prefix=f'model.layers.{site}.mamba.'
                alog=sf.get_tensor(prefix+'A_log').cuda();dd=sf.get_tensor(prefix+'D').cuda()
                weight=sf.get_tensor(prefix+'norm.weight').cuda().float();ow=sf.get_tensor(prefix+'out_proj.weight').cuda()
                for rec in records:
                    stage=f"evaluation/{site}/{rec['id']}";guard();f=rec['sites'][site]['fields'];n=rec['history']
                    xx=raw(f['x']).reshape(n,48,64);bb=raw(f['B']);cc=raw(f['C']);dt=raw(f['delta']).bfloat16()
                    yy=raw(f['y']);gate=raw(f['gate']);ref=raw(f['output']).bfloat16()
                    sg=torch.nn.functional.silu(gate);u=yy*sg;source_inv=torch.rsqrt(u.square().mean(-1,keepdim=True)+eps)
                    source_out=torch.nn.functional.linear((u*source_inv*weight).bfloat16(),ow)
                    baseline=error(source_out,ref);assert baseline<=criteria['reconstruction_relative_RMS']
                    arms={};measurements={};label=torch.tensor(rec['positions'],device='cuda')
                    for name in b['layouts']:
                        m=maps[site][name];packed=torch.empty((n,384),device='cuda');px=torch.empty_like(packed)
                        expected=torch.empty_like(packed);rec_expected=torch.empty_like(packed);skip=torch.empty_like(packed)
                        yhat=torch.empty_like(xx);scan_seconds=0.
                        for r,heads in m['groups'].items():
                            w=torch.stack([m['W'][h] for h in heads]);v=torch.stack([m['V'][h] for h in heads])
                            xm=torch.einsum('thc,hcr->thr',xx[:,heads],w)
                            torch.cuda.synchronize();ts=time.monotonic()
                            ym=scan(torch,code,xm,bb,cc,dt[:,heads],alog[heads],dd[heads])
                            torch.cuda.synchronize();scan_seconds+=time.monotonic()-ts
                            source_head=yy.reshape(n,48,64)[:,heads]
                            ex=torch.einsum('thc,hcr->thr',source_head,w)
                            rex=torch.einsum('thc,hcr->thr',source_head-xx[:,heads]*dd.float()[None,heads,None],w)
                            yh=torch.einsum('thr,hcr->thc',ym,v)
                            for j,h in enumerate(heads):
                                sl=slice(m['offsets'][h],m['offsets'][h+1]);packed[:,sl]=ym[:,j];px[:,sl]=xm[:,j]
                                expected[:,sl]=ex[:,j];rec_expected[:,sl]=rex[:,j];skip[:,sl]=xm[:,j]*dd[h].float();yhat[:,h]=yh[:,j]
                            del w,v,xm,ym,source_head,ex,rex,yh;torch.cuda.empty_cache()
                        pe=error(packed,expected);re=error(packed-skip,rec_expected)
                        assert pe<=criteria['projected_scan_relative_RMS'] and re<=criteria['projected_recurrent_only_relative_RMS'],('projection',site,rec['id'],name,pe,re)
                        if rec['id']=='broad_fit_smol_magpie_ultra_022':
                            heads=np.array([0,7,13,19,25,31,37,47]);coords=[min(j,m['ranks'][int(h)]-1) for j,h in enumerate(heads)]
                            columns=[m['offsets'][int(h)]+c for h,c in zip(heads,coords)]
                            x64=px.cpu().numpy()[:,columns].astype('f8');dt64=dt.cpu().float().numpy()[:,heads].astype('f8')
                            b64=bb.cpu().numpy().astype('f8');c64=cc.cpu().numpy().astype('f8')
                            aa=-np.exp(alog.cpu().float().numpy().astype('f8'))[heads];d64=dd.cpu().float().numpy().astype('f8')[heads]
                            state=np.zeros((8,256),dtype='f8');direct=[]
                            for t in range(n):
                                state=state*np.exp(dt64[t,:,None]*aa[:,None])+dt64[t,:,None]*x64[t,:,None]*b64[t]
                                direct.append((state*c64[t]).sum(-1)+d64*x64[t])
                            actual=packed.cpu().numpy()[:,columns].astype('f8');direct=np.asarray(direct)
                            er=float(np.linalg.norm(direct-actual)/max(np.linalg.norm(actual),1e-12));assert er<=criteria['scalar_F64_relative_RMS']
                            scalar.append(dict(id=rec['id'],site=site,layout=name,heads=heads.tolist(),coordinates=coords,history=n,relative_RMS=er))
                        yhat=yhat.reshape(n,3072);uh=yhat*sg;arms[name]={}
                        for mode,inv in ((b['modes'][0],source_inv),(b['modes'][1],torch.rsqrt(uh.square().mean(-1,keepdim=True)+eps))):
                            op=torch.nn.functional.linear((uh*inv*weight).bfloat16(),ow)
                            arms[name][mode]=dict(output_relative_RMS=error(op,ref),
                                centered_output_relative_RMS=error(op.double()-op.double().mean(0),ref.double()-ref.double().mean(0)),label_output_relative_RMS=error(op[label],ref[label]))
                        lp=a.directory/f"{rec['id']}.site{site:02d}.{name}.f32";arr=packed.cpu().numpy().astype('<f4');assert np.isfinite(arr).all()
                        with lp.open('xb') as stream:stream.write(arr.tobytes(order='C'))
                        measurements[name]=dict(latent=dict(**extent(lp),shape=[n,384],dtype='F32'),projected_scan_relative_RMS=pe,
                            projected_recurrent_only_relative_RMS=re,ungated_reconstruction_relative_RMS=error(yhat,yy),
                            scan_seconds=scan_seconds,scan_groups=len(m['groups']),carried_channels=384)
                        del packed,px,expected,rec_expected,skip,yhat,uh,op,inv,arr;torch.cuda.empty_cache()
                    row=dict(id=rec['id'],site=site,split=rec['split'],domain=rec['domain'],history=n,
                        reconstruction_relative_RMS=baseline,measurements=measurements,arms=arms)
                    rows.append(row);write(a.directory/f"{rec['id']}.site{site:02d}.metrics.json",row)
                    event(stage='evaluation',site=site,id=rec['id'],complete=len(rows))
                    del xx,bb,cc,dt,yy,gate,ref,sg,u,source_inv,source_out;torch.cuda.empty_cache()
                del alog,dd,weight,ow;torch.cuda.empty_cache()
        aggregates={};domains={};gates={};viable=[]
        for site in b['sites']:
            aggregates[str(site)]={};domains[str(site)]={}
            for split in ('FIT','DEV'):
                sr=[r for r in rows if r['site']==site and r['split']==split];assert len(sr)==24
                aggregates[str(site)][split]={};domains[str(site)][split]={}
                for name in b['layouts']:
                    aggregates[str(site)][split][name]={};domains[str(site)][split][name]={}
                    for mode in b['modes']:
                        aa={k:dict(mean=sum(r['arms'][name][mode][k] for r in sr)/24,worst=max(r['arms'][name][mode][k] for r in sr)) for k in sr[0]['arms'][name][mode]}
                        dr={d:{k:dict(mean=sum(r['arms'][name][mode][k] for r in sr if r['domain']==d)/2,
                            worst=max(r['arms'][name][mode][k] for r in sr if r['domain']==d)) for k in aa} for d in sorted({r['domain'] for r in sr})}
                        aggregates[str(site)][split][name][mode]=aa;domains[str(site)][split][name][mode]=dr
        for name in b['layouts']:
            gates[name]={}
            for mode in b['modes']:
                sgates={}
                for site in b['sites']:
                    g=aggregates[str(site)]['DEV'][name][mode];dr=domains[str(site)]['DEV'][name][mode]
                    sgates[str(site)]=dict(mean=g['output_relative_RMS']['mean']<=criteria['DEV_output_mean'],
                        case_worst=g['output_relative_RMS']['worst']<=criteria['DEV_case_worst'],
                        domain_mean_worst=max(d['output_relative_RMS']['mean'] for d in dr.values())<=criteria['DEV_domain_mean_worst'],
                        centered_mean=g['centered_output_relative_RMS']['mean']<=criteria['DEV_centered_mean'])
                passed=all(all(v.values()) for v in sgates.values());gates[name][mode]=dict(sites=sgates,passed=passed)
                if passed:viable.append((max(aggregates[str(s)]['DEV'][name][mode]['output_relative_RMS']['mean'] for s in b['sites']),
                    sum(aggregates[str(s)]['DEV'][name][mode]['output_relative_RMS']['mean'] for s in b['sites']),name,mode))
        chosen=min(viable) if viable else None
        result=dict(schema='ORIGINAL_FALCON_CHANNEL_TRANSPORT_RESULT_V1',experiment=b['experiment'],freeze=a.freeze,binding_sha256=a.binding_sha,
            decision='BALANCED_HEADS_MERIT_CONDITIONAL_GENERATOR_STUDY' if chosen else 'BALANCED_HEADS_LINEAR_RECONSTRUCTION_BUDGET_FAIL',
            selected_layout=dict(layout=chosen[2],mode=chosen[3]) if chosen else None,selections=packets,records=rows,scalar_F64=scalar,
            aggregates=aggregates,domains=domains,gates=gates,basis_head_fits=144,source_forwards=0,optimizer_updates=0,native_calls=0,
            quality_admission=False,speed_admission=False,deployable_decoder=False,
            scope='Two actual packed384 channel scans;source state256/all nonlinear generators/full3072 gate retained. Separable independent-time pair optimum only;not coupled output/chatbot optimum. Grouped diagnostic compute is not native rate.',
            actual_latent_payload_bytes=sum(m['latent']['bytes'] for r in rows for m in r['measurements'].values()),
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            worker_OS_peak_snapshot=proc.memory_info().peak_wset,elapsed_seconds=time.monotonic()-start)
        assert len(rows)==144 and len(packets)==3 and len(scalar)==6
        write(a.out,result);event(stage='complete',decision=result['decision'],sha256=sha(a.out))
    except BaseException as e:
        write(a.directory/'first_fault.json',dict(stage=stage,error=repr(e),selections=packets,completed_measurements=rows,scalar_F64=scalar,
            elapsed_seconds=time.monotonic()-start,GPU_allocated_peak=torch.cuda.max_memory_allocated() if torch is not None else None,
            GPU_reserved_peak=torch.cuda.max_memory_reserved() if torch is not None else None,OS_peak=proc.memory_info().peak_wset if proc is not None else None));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--bind',action='store_true');m.add_argument('--launch',action='store_true');m.add_argument('--worker',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.bind:bind(a)
    elif a.launch:launch(a)
    else:worker(a)
