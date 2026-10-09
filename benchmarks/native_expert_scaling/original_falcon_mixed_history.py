"""FIT-only per-head rank8 ungated history mixtures and downstream gate checks.

Source operands only; no model/optimizer/native forwards. Saves actual384 F32
projected recurrence values for later conditional decoder studies.
"""
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
    b=json.loads(parent.read_bytes());r=json.loads(Path(b['capture']).read_bytes())
    for i in b['inputs']:assert extent(i['path'])==i
    files=[Path(__file__),parent,DOC/'ORIGINAL_FALCON_MIXED_HISTORY_PROTOCOL_20261009.md',
        DOC/'original_falcon_channel_readout_refit_result_20261009.json',
        DOC/'original_falcon_channel_readout_refit_result_20261009.terminal.json',
        SITE/'transformers/models/falcon_h1/modeling_falcon_h1.py']
    for rec in r['records']:
        for site in b['sites']:
            for name in ('x','B','C','delta'):
                item=rec['sites'][site]['fields'][name]
                assert extent(item['path'])=={k:item[k] for k in ('path','bytes','sha256')};files.append(Path(item['path']))
    b.update(worker_path=str(Path(__file__).resolve()),experiment='MIXED_HISTORY_RANK8_V1',rank_per_head=8,
        layouts=['all48_heads_mixed8'],modes=['full_source_denominator','reconstructed_denominator'],
        limits=dict(seconds=1800,reserve_seconds=90,OS_bytes=6<<30,GPU_allocated_bytes=4<<30,
            GPU_reserved_bytes=5<<30,output_bytes=128<<20),
        inputs=list({i['path']:i for i in b['inputs']+[extent(p) for p in files]}.values()))
    b['criteria'].update(orthogonality=1e-10,projected_scan_relative_RMS=1e-4,
        projected_recurrent_only_relative_RMS=1e-4,scalar_F64_relative_RMS=1e-4)
    write(a.out,b);print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(b['inputs']))),flush=True)


def scan(torch,code,x,bb,cc,delta,A_log,dskip):
    """Source chunk128 SSD reductions with8 mixed channels per head."""
    n=len(x);heads=x.shape[1];width=x.shape[2];rank=bb.shape[-1];pad=(128-n%128)%128
    x=x.reshape(1,n,heads,width).float()
    bb=bb.reshape(1,n,1,rank).float().repeat_interleave(heads,dim=2)
    cc=cc.reshape(1,n,1,rank).float().repeat_interleave(heads,dim=2);delta=delta.reshape(1,n,heads)
    d_residual=dskip[...,None]*code.pad_tensor_by_size(x,pad)
    x=x*delta[...,None];aa=-torch.exp(A_log.float())*delta
    x,aa,bb,cc=[code.reshape_into_chunks(t,pad,128) for t in (x,aa,bb,cc)]
    aa=aa.permute(0,3,1,2);acum=torch.cumsum(aa,dim=-1);decay=torch.exp(code.segment_sum(aa))
    pair=torch.cat([(cc[:,i:i+1,:,None,:,:]*bb[:,i:i+1,None,:,:,:]).sum(-1) for i in range(cc.shape[1])],dim=1)
    kernel=(pair[...,None]*decay.permute(0,2,3,4,1)[...,None]).sum(-1)
    diag=torch.cat([(kernel[:,i:i+1,...,None]*x[:,i:i+1,None]).sum(dim=3) for i in range(kernel.shape[1])],dim=1)
    ds=torch.exp(acum[:,:,:,-1:]-acum);bd=bb*ds.permute(0,-2,-1,1)[...,None]
    states=torch.cat([(bd[:,i:i+1,...,None,:]*x[:,i:i+1,...,None]).sum(dim=2) for i in range(bd.shape[1])],dim=1)
    states=torch.cat([torch.zeros_like(states[:,:1]),states],dim=1)
    dc=torch.exp(code.segment_sum(torch.nn.functional.pad(acum[:,:,:,-1],(1,0)))).transpose(1,3)
    next_states=(dc[...,None,None]*states[:,:,None,...]).sum(dim=1)[:,:-1]
    reduced=torch.cat([(cc[:,i:i+1,...,None,:]*next_states[:,i:i+1,None,...]).sum(-1) for i in range(cc.shape[1])],dim=1)
    off=reduced*torch.exp(acum).permute(0,2,3,1)[...,None]
    yy=(diag+off).reshape(1,-1,heads,width)+d_residual
    return yy[:,:n].reshape(n,heads,width)


def worker(a):
    start=time.monotonic();assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes());assert b['experiment']=='MIXED_HISTORY_RANK8_V1'
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
        proc=psutil.Process();proc.cpu_affinity(list(range(6)))
        def guard():
            assert time.monotonic()-start<=b['limits']['seconds']-b['limits']['reserve_seconds'],'worker reserve'
            assert proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True),'OS/children'
            assert torch.cuda.max_memory_allocated()<=b['limits']['GPU_allocated_bytes'],'GPU allocated'
            assert torch.cuda.max_memory_reserved()<=b['limits']['GPU_reserved_bytes'],'GPU reserved'
        def event(**kw):
            guard();print(json.dumps(dict(seconds=time.monotonic()-start,**kw)),flush=True)
        records=json.loads(Path(b['capture']).read_bytes())['records'];fit=[r for r in records if r['split']=='FIT']
        eps=json.loads(Path(b['source_config']).read_bytes())['rms_norm_eps']
        def raw_np(item):
            if item['dtype']=='BF16':z=(np.fromfile(item['path'],dtype='<u2').astype('<u4')<<16).view('<f4')
            else:assert item['dtype']=='F32';z=np.fromfile(item['path'],dtype='<f4')
            z=z.reshape(item['shape']);assert np.isfinite(z).all();return z
        def raw(item):return torch.from_numpy(raw_np(item)).cuda()
        def error(x,y):
            x=x.double();y=y.double();return float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(y).clamp_min(1e-12))
        maps={}
        for site in b['sites']:
            gram=torch.zeros((48,64,64),device='cuda',dtype=torch.float64);centered=torch.zeros_like(gram)
            for rec in fit:
                stage=f"FIT_basis/{site}/{rec['id']}";guard()
                y=raw(rec['sites'][site]['fields']['y']).reshape(rec['history'],48,64).double().permute(1,0,2)
                gg=y.transpose(-2,-1)@y;tr=gg.diagonal(dim1=-2,dim2=-1).sum(-1);assert (tr>0).all()
                gram+=gg/tr[:,None,None]/24
                yc=y-y.mean(1,keepdim=True);cg=yc.transpose(-2,-1)@yc;ctr=cg.diagonal(dim1=-2,dim2=-1).sum(-1)
                assert (ctr>0).all();centered+=cg/ctr[:,None,None]/24
                event(stage='FIT_basis',site=site,id=rec['id']);del y,gg,yc,cg,tr,ctr;torch.cuda.empty_cache()
            eig,vec=torch.linalg.eigh(gram.cpu());rr=vec[:,:,-8:].contiguous()
            orth=float((rr.transpose(-2,-1)@rr-torch.eye(8,dtype=torch.float64)).abs().max())
            assert orth<=b['criteria']['orthogonality']
            retained=(rr.transpose(-2,-1)@gram.cpu()@rr).diagonal(dim1=-2,dim2=-1).sum(-1)
            centered_retained=(rr.transpose(-2,-1)@centered.cpu()@rr).diagonal(dim1=-2,dim2=-1).sum(-1)
            path=a.directory/f'site{site:02d}.head_bases.npz'
            with path.open('xb') as f:np.savez(f,R=rr.numpy(),G=gram.cpu().numpy(),G_centered=centered.cpu().numpy(),eigenvalues=eig.numpy())
            packet=dict(site=site,basis=extent(path),heads=48,rank_per_head=8,orthogonality_error=orth,
                FIT_cases=24,FIT_positions=sum(r['history'] for r in fit),
                uncentered_retention=retained.tolist(),centered_retention=centered_retained.tolist())
            packets.append(packet);maps[site]=rr.float().cuda();write(a.directory/f'site{site:02d}.basis.json',packet)
            event(stage='basis_frozen',site=site,orthogonality_error=orth,mean_retention=float(retained.mean()),mean_centered_retention=float(centered_retained.mean()))
            del gram,centered,eig,vec,rr,retained,centered_retained;torch.cuda.empty_cache()
        with safe_open(b['source_weights'],framework='pt',device='cpu') as sf,torch.inference_mode():
            for site in b['sites']:
                prefix=f'model.layers.{site}.mamba.';rr=maps[site]
                alog=sf.get_tensor(prefix+'A_log').cuda();dd=sf.get_tensor(prefix+'D').cuda()
                weight=sf.get_tensor(prefix+'norm.weight').cuda().float();ow=sf.get_tensor(prefix+'out_proj.weight').cuda()
                for rec in records:
                    stage=f"evaluation/{site}/{rec['id']}";guard();fields=rec['sites'][site]['fields'];n=rec['history']
                    xx=raw(fields['x']);bb=raw(fields['B']);cc=raw(fields['C']);dt=raw(fields['delta']).bfloat16()
                    yy=raw(fields['y']);gate=raw(fields['gate']);ref=raw(fields['output']).bfloat16()
                    xm=torch.einsum('thc,hcr->thr',xx.reshape(n,48,64),rr)
                    ym=scan(torch,code,xm,bb,cc,dt,alog,dd)
                    expected=torch.einsum('thc,hcr->thr',yy.reshape(n,48,64),rr)
                    projected_error=error(ym,expected)
                    skip_mixed=xm*dd.float()[None,:,None]
                    skip_source=xx.reshape(n,48,64)*dd.float()[None,:,None]
                    rec_expected=torch.einsum('thc,hcr->thr',yy.reshape(n,48,64)-skip_source,rr)
                    recurrent_error=error(ym-skip_mixed,rec_expected)
                    assert max(projected_error,recurrent_error)<=b['criteria']['projected_scan_relative_RMS'],('projection',site,rec['id'],projected_error,recurrent_error)
                    if rec['id']=='broad_fit_smol_magpie_ultra_022':
                        heads=np.array([0,7,13,19,25,31,37,47]);coords=np.arange(8)
                        x64=xm.cpu().numpy()[:,heads,coords].astype('f8');dt64=dt.cpu().float().numpy()[:,heads].astype('f8')
                        b64=bb.cpu().numpy().astype('f8');c64=cc.cpu().numpy().astype('f8')
                        aa=-np.exp(alog.cpu().float().numpy().astype('f8'))[heads];d64=dd.cpu().float().numpy().astype('f8')[heads]
                        state=np.zeros((8,256),dtype='f8');direct=[]
                        for t in range(n):
                            state=state*np.exp(dt64[t,:,None]*aa[:,None])+dt64[t,:,None]*x64[t,:,None]*b64[t]
                            direct.append((state*c64[t]).sum(-1)+d64*x64[t])
                        actual=ym.cpu().numpy()[:,heads,coords].astype('f8');direct=np.asarray(direct)
                        er=float(np.linalg.norm(direct-actual)/max(np.linalg.norm(actual),1e-12));assert er<=b['criteria']['scalar_F64_relative_RMS']
                        scalar.append(dict(id=rec['id'],site=site,heads=heads.tolist(),coordinates=coords.tolist(),history=n,relative_RMS=er))
                    yhat=torch.einsum('thr,hcr->thc',ym,rr).reshape(n,3072)
                    u=yy*torch.nn.functional.silu(gate);uh=yhat*torch.nn.functional.silu(gate)
                    source_inv=torch.rsqrt(u.square().mean(-1,keepdim=True)+eps)
                    source_out=torch.nn.functional.linear((u*source_inv*weight).bfloat16(),ow)
                    baseline=error(source_out,ref);assert baseline<=b['criteria']['reconstruction_relative_RMS']
                    label=torch.tensor(rec['positions'],device='cuda');arms={}
                    for mode,inv in ((b['modes'][0],source_inv),(b['modes'][1],torch.rsqrt(uh.square().mean(-1,keepdim=True)+eps))):
                        op=torch.nn.functional.linear((uh*inv*weight).bfloat16(),ow)
                        arms[mode]=dict(output_relative_RMS=error(op,ref),centered_output_relative_RMS=error(op.double()-op.double().mean(0),ref.double()-ref.double().mean(0)),
                            label_output_relative_RMS=error(op[label],ref[label]))
                    latent_path=a.directory/f"{rec['id']}.site{site:02d}.mixed.f32"
                    arr=ym.cpu().numpy().astype('<f4');assert np.isfinite(arr).all()
                    with latent_path.open('xb') as f:f.write(arr.tobytes(order='C'))
                    latent=dict(**extent(latent_path),shape=[n,48,8],dtype='F32')
                    row=dict(id=rec['id'],site=site,split=rec['split'],domain=rec['domain'],history=n,latent=latent,
                        reconstruction_relative_RMS=baseline,projected_scan_relative_RMS=projected_error,
                        projected_recurrent_only_relative_RMS=recurrent_error,ungated_reconstruction_relative_RMS=error(yhat,yy),arms=arms)
                    rows.append(row);write(a.directory/f"{rec['id']}.site{site:02d}.metrics.json",row)
                    event(stage='evaluation',site=site,id=rec['id'],complete=len(rows))
                    del xx,bb,cc,dt,yy,gate,ref,xm,ym,expected,skip_mixed,skip_source,rec_expected,yhat,u,uh,source_inv,source_out,op,inv,arr
                    torch.cuda.empty_cache()
                del alog,dd,weight,ow;torch.cuda.empty_cache()
        aggregates={};domains={};gates={};viable=[]
        for site in b['sites']:
            aggregates[str(site)]={};domains[str(site)]={}
            for split in ('FIT','DEV'):
                sr=[r for r in rows if r['site']==site and r['split']==split];assert len(sr)==24
                aggregates[str(site)][split]={};domains[str(site)][split]={}
                for mode in b['modes']:
                    aa={key:dict(mean=sum(r['arms'][mode][key] for r in sr)/24,worst=max(r['arms'][mode][key] for r in sr)) for key in sr[0]['arms'][mode]}
                    dres={d:{key:dict(mean=sum(r['arms'][mode][key] for r in sr if r['domain']==d)/2,
                        worst=max(r['arms'][mode][key] for r in sr if r['domain']==d)) for key in aa} for d in sorted({r['domain'] for r in sr})}
                    aggregates[str(site)][split][mode]=aa;domains[str(site)][split][mode]=dres
        for mode in b['modes']:
            sg={}
            for site in b['sites']:
                g=aggregates[str(site)]['DEV'][mode];d=domains[str(site)]['DEV'][mode];c=b['criteria']
                sg[str(site)]=dict(mean=g['output_relative_RMS']['mean']<=c['DEV_output_mean'],case_worst=g['output_relative_RMS']['worst']<=c['DEV_case_worst'],
                    domain_mean_worst=max(x['output_relative_RMS']['mean'] for x in d.values())<=c['DEV_domain_mean_worst'],centered_mean=g['centered_output_relative_RMS']['mean']<=c['DEV_centered_mean'])
            passed=all(all(v.values()) for v in sg.values());gates[mode]=dict(sites=sg,passed=passed)
            if passed:viable.append((max(aggregates[str(s)]['DEV'][mode]['output_relative_RMS']['mean'] for s in b['sites']),mode))
        result=dict(schema='ORIGINAL_FALCON_CHANNEL_TRANSPORT_RESULT_V1',experiment=b['experiment'],freeze=a.freeze,binding_sha256=a.binding_sha,
            decision='MIXED_HISTORY_MERITS_CONDITIONAL_DECODER_STUDY' if viable else 'MIXED_HISTORY_LINEAR_RECONSTRUCTION_BUDGET_FAIL',
            selected_diagnostic_mode=min(viable)[1] if viable else None,selections=packets,records=rows,scalar_F64=scalar,
            aggregates=aggregates,domains=domains,gates=gates,basis_head_fits=144,source_forwards=0,optimizer_updates=0,native_calls=0,
            quality_admission=False,speed_admission=False,deployable_decoder=False,
            scope='Actual source B/C/state256/gate/full nonlinear x functions;only8 fixed ungated mixtures per48heads. '
                'Both output diagnostics retain full3072 source gate/readout. No compressed generator/native/state96/P256/depth/chatbot admission.',
            actual_latent_payload_bytes=sum(r['latent']['bytes'] for r in rows),GPU_allocated_peak=torch.cuda.max_memory_allocated(),
            GPU_reserved_peak=torch.cuda.max_memory_reserved(),worker_OS_peak_snapshot=proc.memory_info().peak_wset,elapsed_seconds=time.monotonic()-start)
        assert len(rows)==144 and len(packets)==3 and len(scalar)==3
        write(a.out,result);event(stage='complete',decision=result['decision'],sha256=sha(a.out))
    except BaseException as e:
        write(a.directory/'first_fault.json',dict(stage=stage,error=repr(e),selections=packets,completed_measurements=rows,scalar_F64=scalar,
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
