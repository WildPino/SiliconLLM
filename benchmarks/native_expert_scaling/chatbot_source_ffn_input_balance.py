"""Correct input-balance orientation; reuse completed alpha=0 trials."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from chatbot_source_ffn_local import read,receipt
from chatbot_falcon_usability import SITE,sha,write
sys.path.insert(0,str(SITE))


def bind(a):
    oldpath=DOC/'chatbot_source_ffn_local_calibrate_binding_20261009.json';old=read(oldpath)
    resultpath=DOC/'chatbot_source_ffn_local_calibrate_result_20261009.json';prior=read(resultpath)
    terminal=receipt(resultpath)
    assert sha(oldpath)==prior['binding_sha256'] and prior['decision']=='SOURCE_FFN_LOCAL_BALANCE_FAIL'
    for item in old['inputs']:
        p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p)
    files=[Path(v['path']) for v in old['inputs']]
    files += [oldpath,resultpath,resultpath.with_suffix('.terminal.json'),Path(__file__),DOC/'CHATBOT_SOURCE_FFN_INPUT_BALANCE_PROTOCOL_20261009.md']
    for item in terminal['output_files']:
        p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];files.append(p)
    files=list(dict.fromkeys(p.resolve() for p in files))
    b={k:v for k,v in old.items() if k!='inputs'}
    b.update(worker_path=str(Path(__file__).resolve()),prior_result=str(resultpath),variant='MIRRORED_INPUT_ALPHA',
        runtime_binding_scope='Original local binding plus completed positive grid andoutputs/new mirrored orientation/code;not full DLL-tree.',
        inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    write(a.out,b)
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(files))),flush=True)


def worker(a):
    start=time.monotonic();assert sha(a.binding)==a.binding_sha;b=read(a.binding)
    assert b['schema']=='SOURCE_FFN_LOCAL_BINDING_V1' and b['variant']=='MIRRORED_INPUT_ALPHA' and b['phase']=='calibrate'
    a.directory.mkdir(exist_ok=False);phase='startup';completed=[];torch=None
    try:
        assert sys.version_info[:3]==(3,12,10) and Path(sys.executable).resolve()==Path(b['python']).resolve()
        assert all(os.environ.get(k)=='1' for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'))
        import numpy as np
        import psutil
        import torch
        from torch.nn import functional as F
        from safetensors import safe_open
        from chatbot_hybrid_target import aq63,row_scale
        assert (torch.__version__,np.__version__)==('2.6.0+cu124','2.4.6')
        proc=psutil.Process();proc.cpu_affinity(list(range(6)))
        torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        def guard():
            lim=b['limits'];assert time.monotonic()-start<=lim['seconds']-lim['reserve_seconds'],'time reserve'
            assert proc.memory_info().peak_wset<=lim['OS_bytes'],'OS cap'
            assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'],'GPU allocated cap'
            assert torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes'],'GPU reserved cap'
            assert not proc.children(recursive=True),'worker child'
            assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=lim['output_bytes'],'output cap'
        def event(stage,**fields):print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,**fields)),flush=True);guard()
        def raw_file(name,array,dtype):
            path=a.directory/name
            with path.open('xb') as stream:stream.write(array.tobytes());stream.flush();os.fsync(stream.fileno())
            return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path),shape=list(array.shape),dtype=dtype)
        def metric(actual,expected):
            v,t=actual.double(),expected.double();tn=t.square().sum().item();vn=v.square().sum().item();dot=(v*t).sum().item()
            amplitude=dot/max(vn,1e-30)
            return dict(relative_L2=((v-t).square().sum().item()/max(tn,1e-30))**.5,cosine=dot/max((vn*tn)**.5,1e-30),
                oracle_amplitude=amplitude,oracle_relative_L2=((v*amplitude-t).square().sum().item()/max(tn,1e-30))**.5)
        def output(identifier,value,reference):
            assert torch.isfinite(value).all().item();bf=value.to(torch.bfloat16)
            return dict(**metric(bf.float(),reference),output=raw_file(identifier+'.bf16',bf.contiguous().view(torch.uint16).cpu().numpy().astype('<u2',copy=False),'BF16'))
        capture=read(b['capture']);prior=read(b['prior_result']);cfg=read(Path(b['source'])/'config.json');ga,db=cfg['mlp_multipliers']
        assert (ga,db)==(.4419417382415922,.13020833333333331)
        def response(x,w,codes=None,scales=None):
            def linear(h,name):
                if codes is None:return F.linear(h,w[name])
                q,act=aq63(h);return F.linear(q,codes[name].float())*scales[name]*act
            return linear(linear(x,'up')*F.silu(linear(x,'gate')*ga),'down')*db
        all_sites=[];new_trials=0;reused_trials=0
        with safe_open(str(Path(b['source'])/'model.safetensors'),framework='pt',device='cpu') as tensors:
            for site in b['sites']:
                phase=f'site{site:02d}';old=next(r for r in prior['records'] if r['site']==site)
                weights={n:tensors.get_tensor(f'model.layers.{site}.feed_forward.{n}_proj.weight').to(device='cuda',dtype=torch.float32) for n in ('gate','up','down')}
                data=[]
                for rec in capture['records']:
                    packet=next(v for v in rec['sites'] if v['site']==site);row=dict(id=rec['id'],split=rec['split'])
                    for name in ('x','y'):
                        item=packet[name];bits=np.fromfile(item['path'],dtype='<u2').reshape(item['shape'])
                        row[name]=torch.from_numpy(bits.copy()).view(torch.bfloat16).to(device='cuda',dtype=torch.float32)
                    raw=old['decomposition'][rec['id']]['F32']['raw_output_F32']
                    row['original_F32']=torch.from_numpy(np.fromfile(raw['path'],dtype='<f4').reshape(raw['shape']).copy()).to('cuda')
                    data.append(row)
                fit=[r for r in data if r['split']=='FIT'];dev=[r for r in data if r['split']=='DEV']
                xrms=torch.stack([r['x'].double().square().mean(0) for r in fit]).mean(0).sqrt().clamp_min(1e-12)
                wrms=((weights['gate'].double().square().mean(0)+weights['up'].double().square().mean(0))/2).sqrt().clamp_min(1e-12)
                hidden=[(F.linear(r['x'],weights['up'])*F.silu(F.linear(r['x'],weights['gate'])*ga)).double().square().mean(0) for r in fit]
                zrms=torch.stack(hidden).mean(0).sqrt().clamp_min(1e-12);drms=weights['down'].double().square().mean(0).sqrt().clamp_min(1e-12)
                logS=xrms.log()-wrms.log();logR=drms.log()-zrms.log();logS-=logS.mean();logR-=logR.mean()
                trials=[];best_loss=float('inf');best=None
                for alpha in (0.,-.5,-1.):
                    for beta in (0.,.5,1.):
                        guard();S=(alpha*logS).exp().clamp(1/16,16).float();R=(beta*logR).exp().clamp(1/16,16).float()
                        transformed=dict(gate=weights['gate']/S,up=weights['up']*R[:,None]/S,down=weights['down']/R)
                        scales={n:row_scale(w) for n,w in transformed.items()};codes={n:torch.round(w/scales[n][:,None]).clamp(-1,1).to(torch.int8) for n,w in transformed.items()}
                        if alpha==0:
                            trial=dict(next(t for t in old['trials'] if t['alpha']==alpha and t['beta']==beta),origin='Completed positive-grid alpha0 FIT outputs reused')
                            loss=trial['FIT_case_mean_relative_L2'];reused_trials+=1
                        else:
                            trial=dict(alpha=alpha,beta=beta,FIT=[],identity_FIT=[],origin='New mirrored input response')
                            for rec in fit:
                                identifier=f"site{site:02d}.{rec['id']}.a{alpha}.b{beta}"
                                identity=response(rec['x']*S,transformed)
                                item=metric(identity,rec['original_F32']);assert item['relative_L2']<=b['criteria']['identity_relative_L2']
                                item['output_F32']=raw_file(identifier+'.identity.f32',identity.cpu().numpy().astype('<f4',copy=False),'F32')
                                trial['identity_FIT'].append(dict(id=rec['id'],**item))
                                trial['FIT'].append(dict(id=rec['id'],**output(identifier+'.quantized',response(rec['x']*S,transformed,codes,scales),rec['y'])))
                            loss=sum(r['relative_L2'] for r in trial['FIT'])/2;trial['FIT_case_mean_relative_L2']=loss;new_trials+=1
                        trials.append(trial)
                        if loss<best_loss:best_loss=loss;best=dict(alpha=alpha,beta=beta,S=S,R=R,weights=transformed,codes=codes,scales=scales)
                        event('FIT_trial',site=site,alpha=alpha,beta=beta,error=loss,origin=trial['origin'])
                selected=dict(alpha=best['alpha'],beta=best['beta'],FIT_case_mean_relative_L2=best_loss,DEV=[])
                if best['alpha']==0:
                    assert old['selected']['alpha']==0 and best['beta']==old['selected']['beta']
                    selected['DEV']=old['selected']['DEV'];selected['DEV_origin']='Completed identical selected DEV responses reused'
                else:
                    selected['DEV_origin']='New mirrored selected responses'
                    for rec in dev:
                        identifier=f"site{site:02d}.{rec['id']}.selected"
                        identity=response(rec['x']*best['S'],best['weights']);im=metric(identity,rec['original_F32']);assert im['relative_L2']<=b['criteria']['identity_relative_L2']
                        im['output_F32']=raw_file(identifier+'.identity.f32',identity.cpu().numpy().astype('<f4',copy=False),'F32')
                        item=output(identifier+'.quantized',response(rec['x']*best['S'],best['weights'],best['codes'],best['scales']),rec['y'])
                        baseline=old['decomposition'][rec['id']]['TERNARY_AQ63']['relative_L2']
                        selected['DEV'].append(dict(id=rec['id'],baseline_relative_L2=baseline,ratio=item['relative_L2']/max(baseline,1e-30),identity=im,**item))
                selected['DEV_case_mean_ratio']=sum(r['relative_L2'] for r in selected['DEV'])/max(sum(r['baseline_relative_L2'] for r in selected['DEV']),1e-30)
                selected['gates']=dict(DEV_improvement=selected['DEV_case_mean_ratio']<=b['criteria']['DEV_relative_error_ratio'],
                    every_DEV_retention=all(r['ratio']<=b['criteria']['individual_DEV_ratio'] for r in selected['DEV']))
                selected['input_scale']=raw_file(f'site{site:02d}.selected.S.f32',best['S'].cpu().numpy().astype('<f4',copy=False),'F32')
                selected['hidden_scale']=raw_file(f'site{site:02d}.selected.R.f32',best['R'].cpu().numpy().astype('<f4',copy=False),'F32')
                selected['projections']=[]
                for n in weights:
                    code,scale=best['codes'][n],best['scales'][n];pairs=((code[:,0::2]+1)*3+(code[:,1::2]+1)).T.contiguous()
                    assert torch.equal(pairs.T//3-1,code[:,0::2]) and torch.equal(pairs.T%3-1,code[:,1::2])
                    selected['projections'].append(dict(name=n,codes=raw_file(f'site{site:02d}.selected.{n}.pairs.u8',pairs.cpu().numpy().astype('u1',copy=False),'U8'),
                        scale=raw_file(f'site{site:02d}.selected.{n}.scale.f32',scale.cpu().numpy().astype('<f4',copy=False),'F32')))
                record=dict(site=site,trials=trials,selected=selected);write(a.directory/f'site{site:02d}.json',record);all_sites.append(record);completed.append(site)
                event('site_complete',site=site,alpha=best['alpha'],beta=best['beta'],DEV_ratio=selected['DEV_case_mean_ratio'],gates=selected['gates'])
                del weights,data,best,transformed,codes,scales
        assert new_trials==12 and reused_trials==6;phase='result';guard()
        decision='SOURCE_FFN_MIRROR_BALANCE_PASS' if all(all(r['selected']['gates'].values()) for r in all_sites) else 'SOURCE_FFN_MIRROR_BALANCE_FAIL'
        write(a.out,dict(schema='SOURCE_FFN_LOCAL_RESULT_V1',phase='calibrate',variant=b['variant'],decision=decision,freeze=a.freeze,binding_sha256=a.binding_sha,
            process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),cases=4,records=all_sites,new_FIT_trials=12,reused_FIT_trials=6,
            source_forwards=0,source_generations=0,optimizer_updates=0,native_runs=0,reserved_queries=0,GPU_allocated_peak=torch.cuda.max_memory_allocated(),
            GPU_reserved_peak=torch.cuda.max_memory_reserved(),elapsed_seconds=time.monotonic()-start,quality_admission=False,native_admission=False,
            scope='Corrected real-algebra input balance;two-site local response onfour captured prefixes;no whole quality/own-history/rate/n/DRAM/family admission.'))
        event('complete',decision=decision)
    except BaseException as error:
        failure=dict(fault=repr(error),phase=phase,completed=completed,elapsed_seconds=time.monotonic()-start)
        if torch is not None:failure.update(GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved())
        write(a.directory/'first_failure.json',failure);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('bind','worker'),default='worker');p.add_argument('--binding',type=Path)
    p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();bind(a) if a.mode=='bind' else worker(a)
