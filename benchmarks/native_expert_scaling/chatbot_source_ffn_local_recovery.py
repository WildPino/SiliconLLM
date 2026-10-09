"""Fixed local source-function recovery with deployable ternary/AQ63 forwards."""
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
    oldpath=DOC/'chatbot_source_ffn_input_balance_binding_20261009.json';old=read(oldpath)
    priorpath=DOC/'chatbot_source_ffn_input_balance_result_20261009.json';prior=read(priorpath);terminal=receipt(priorpath)
    assert sha(oldpath)==prior['binding_sha256'] and prior['decision']=='SOURCE_FFN_MIRROR_BALANCE_FAIL'
    for item in old['inputs']:
        p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p)
    files=[Path(v['path']) for v in old['inputs']]
    files += [oldpath,priorpath,priorpath.with_suffix('.terminal.json'),Path(__file__),DOC/'CHATBOT_SOURCE_FFN_LOCAL_RECOVERY_PROTOCOL_20261009.md']
    for item in terminal['output_files']:
        p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];files.append(p)
    files=list(dict.fromkeys(p.resolve() for p in files))
    b={k:v for k,v in old.items() if k!='inputs'}
    b.update(worker_path=str(Path(__file__).resolve()),phase='recover',variant='LOCAL_FUNCTION_QAT',prior_result=str(priorpath),
        steps=256,batch=64,lr=5e-5,criteria=dict(DEV_ratio=.90,individual_DEV_ratio=1.05,absolute_DEV_relative_L2=.10,DEV_cosine=.99),
        limits=dict(seconds=600,reserve_seconds=60,OS_bytes=8<<30,GPU_allocated_bytes=10<<30,GPU_reserved_bytes=11<<30,output_bytes=2<<30),
        runtime_binding_scope='Original operands/source/selected calibrated sectors/local functions plusfixed recovery/runtime/Python/foreign hashes;not full DLL-tree.',
        inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    write(a.out,b);print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(files))),flush=True)


def worker(a):
    start=time.monotonic();assert sha(a.binding)==a.binding_sha;b=read(a.binding)
    assert b['schema']=='SOURCE_FFN_LOCAL_BINDING_V1' and b['phase']=='recover' and b['variant']=='LOCAL_FUNCTION_QAT'
    a.directory.mkdir(exist_ok=False);phase='startup';completed=[];updates=0;torch=None
    try:
        assert sys.version_info[:3]==(3,12,10) and Path(sys.executable).resolve()==Path(b['python']).resolve()
        assert all(os.environ.get(k)=='1' for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'))
        import numpy as np
        import psutil
        import torch
        from torch import nn
        from torch.nn import functional as F
        from safetensors import safe_open
        from chatbot_hybrid_target import aq63
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
            v,t=actual.double(),expected.double();vn=v.square().sum();tn=t.square().sum();dot=(v*t).sum()
            return dict(relative_L2=float(((v-t).square().sum()/tn.clamp_min(1e-30)).sqrt()),cosine=float(dot/(vn*tn).sqrt().clamp_min(1e-30)))
        capture=read(b['capture']);prior=read(b['prior_result']);ga,db=read(Path(b['source'])/'config.json')['mlp_multipliers']
        class Student(nn.Module):
            def __init__(self,w,scales):
                super().__init__()
                for name in ('gate','up','down'):
                    self.register_parameter(name,nn.Parameter(w[name].clone()))
                    self.register_parameter(name+'_scale',nn.Parameter(scales[name].clone()))
            def linear(self,x,name):
                master=getattr(self,name);scale=getattr(self,name+'_scale');q,act=aq63(x)
                code=torch.round((master/scale[:,None]).detach()).clamp(-1,1)
                exact=F.linear(q,code)*scale*act
                if not torch.is_grad_enabled():return exact
                x_proxy=x+(q*act-x).detach()
                w_proxy=code*scale[:,None]+(master-master.detach())
                proxy=F.linear(x_proxy,w_proxy)
                # Exact finite forward, proxy derivative. Finite proxy-proxy.detach()==0.
                return exact.detach()+(proxy-proxy.detach())
            def forward(self,x):
                g=self.linear(x,'gate')*ga;u=self.linear(x,'up')
                return (self.linear(u*F.silu(g),'down')*db).to(torch.bfloat16)
        def cpu_tree(value):
            if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
            if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
            if isinstance(value,list):return [cpu_tree(v) for v in value]
            if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
            return value
        def snapshot(site,step,model,opt,history):
            state=cpu_tree(dict(schema='LOCAL_SOURCE_FFN_RECOVERY_STATE_V1',site=site,steps=step,model=model.state_dict(),optimizer=opt.state_dict(),
                CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state(),history=history,binding_sha256=a.binding_sha,freeze=a.freeze))
            assert all(torch.isfinite(v).all().item() for v in state['model'].values())
            assert len(state['optimizer']['state'])==6
            for slot in state['optimizer']['state'].values():
                assert slot['step'].item()==step and torch.isfinite(slot['exp_avg']).all().item() and torch.isfinite(slot['exp_avg_sq']).all().item()
            path=a.directory/f'site{site:02d}.state{step:03d}.pt';temp=path.with_suffix('.tmp')
            with temp.open('xb') as stream:torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
            temp.replace(path);del state
            return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path))
        records=[]
        with safe_open(str(Path(b['source'])/'model.safetensors'),framework='pt',device='cpu') as tensors:
            for site in b['sites']:
                phase=f'site{site:02d}';selected=next(r for r in prior['records'] if r['site']==site)['selected']
                S=torch.from_numpy(np.fromfile(selected['input_scale']['path'],dtype='<f4')).to('cuda')
                R=torch.from_numpy(np.fromfile(selected['hidden_scale']['path'],dtype='<f4')).to('cuda')
                assert selected['alpha']==0 and torch.equal(S,torch.ones_like(S))
                w={n:tensors.get_tensor(f'model.layers.{site}.feed_forward.{n}_proj.weight').to(device='cuda',dtype=torch.float32) for n in ('gate','up','down')}
                transformed=dict(gate=w['gate']/S,up=w['up']*R[:,None]/S,down=w['down']/R);scales={}
                for p in selected['projections']:
                    n=p['name'];item=p['scale'];scales[n]=torch.from_numpy(np.fromfile(item['path'],dtype='<f4')).to('cuda')
                    code=torch.round(transformed[n]/scales[n][:,None]).clamp(-1,1).to(torch.int8)
                    pair=((code[:,0::2]+1)*3+(code[:,1::2]+1)).T.contiguous().cpu().numpy().astype('u1',copy=False)
                    assert pair.tobytes()==Path(p['codes']['path']).read_bytes(),('initial sector',site,n)
                model=Student(transformed,scales);assert len(list(model.parameters()))==6 and sum(p.numel() for p in model.parameters())==28322816
                data=[]
                for rec in capture['records']:
                    packet=next(v for v in rec['sites'] if v['site']==site);row=dict(id=rec['id'],split=rec['split'])
                    for name in ('x','y'):
                        item=packet[name];bits=np.fromfile(item['path'],dtype='<u2').reshape(item['shape'])
                        row[name]=torch.from_numpy(bits.copy()).view(torch.bfloat16).to(device='cuda',dtype=torch.float32)
                    data.append(row)
                fit=[r for r in data if r['split']=='FIT'];dev=[r for r in data if r['split']=='DEV'];assert len(fit)==len(dev)==2
                base=next(r for r in read(DOC/'chatbot_source_ffn_local_calibrate_result_20261009.json')['records'] if r['site']==site)
                initial=[]
                for rec in data:
                    if rec['split']=='FIT':
                        trial=next(t for t in base['trials'] if t['alpha']==selected['alpha'] and t['beta']==selected['beta'])
                        expected=next(v for v in trial['FIT'] if v['id']==rec['id'])['output']
                    else:expected=next(v for v in selected['DEV'] if v['id']==rec['id'])['output']
                    bits=np.fromfile(expected['path'],dtype='<u2').reshape(expected['shape'])
                    with torch.no_grad():value=model(rec['x']*S)
                    assert np.array_equal(value.view(torch.uint16).cpu().numpy(),bits),('initial inference',site,rec['id'])
                    # Check STE finite forward on the actual captured operands before updates.
                    train_value=model(rec['x']*S)
                    assert torch.equal(train_value.view(torch.uint16),value.view(torch.uint16)),('STE forward',site,rec['id'])
                    initial.append(dict(id=rec['id'],split=rec['split'],**metric(value.float(),rec['y'])));del value,train_value
                opt=torch.optim.AdamW(model.parameters(),lr=b['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
                history=[];priced=[];price_state=None
                for step in range(1,b['steps']+1):
                    guard();case=(step-1)%2;rec=fit[case];visit=(step-1)//2
                    offset=(visit*b['batch'])%rec['x'].shape[0]
                    stop=min(offset+b['batch'],rec['x'].shape[0]);x=rec['x'][offset:stop]*S;y=rec['y'][offset:stop]
                    torch.cuda.synchronize();t0=time.monotonic();opt.zero_grad(set_to_none=True)
                    actual=model(x).float();loss=(actual-y).square().sum()/y.square().sum().clamp_min(1e-30)
                    assert torch.isfinite(loss).item();loss.backward();norms={}
                    for name,p in model.named_parameters():
                        assert p.grad is not None and torch.isfinite(p.grad).all().item(),('gradient',site,step,name)
                        norms[name]=p.grad.float().norm().item();assert norms[name]>0,('zero gradient',site,step,name)
                    gradnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True).item()
                    opt.step();updates+=1
                    with torch.no_grad():
                        for n in ('gate','up','down'):getattr(model,n+'_scale').clamp_(min=1e-8)
                    for p in model.parameters():assert torch.isfinite(p).all().item()
                    for slot in opt.state.values():
                        assert torch.isfinite(slot['exp_avg']).all().item() and torch.isfinite(slot['exp_avg_sq']).all().item() and slot['step'].item()==step
                    torch.cuda.synchronize();seconds=time.monotonic()-t0
                    row=dict(step=step,id=rec['id'],offset=offset,rows=stop-offset,loss=loss.item(),gradnorm=gradnorm,gradient_norms=norms,seconds=seconds)
                    history.append(row)
                    if step<=2:priced.append(seconds)
                    if step==2:
                        price_state=snapshot(site,step,model,opt,history)
                        remaining=b['steps']-2+(b['steps'] if site==0 else 0)
                        estimate=time.monotonic()-start+remaining*max(priced)+60
                        event('actual_price',site=site,seconds_per_step=priced,projected_total_seconds=estimate,durable_state=price_state)
                        assert estimate<=b['limits']['seconds']-b['limits']['reserve_seconds'],'priced total exceeds cap'
                    if step%32==0:event('update',site=site,step=step,loss=loss.item(),seconds=seconds)
                    del actual,loss,x,y
                final_state=snapshot(site,b['steps'],model,opt,history);after=[]
                with torch.no_grad():
                    for rec in data:
                        value=model(rec['x']*S);item=metric(value.float(),rec['y'])
                        expected=next(v for v in initial if v['id']==rec['id'])
                        item['ratio']=item['relative_L2']/max(expected['relative_L2'],1e-30)
                        item['output']=raw_file(f"site{site:02d}.{rec['id']}.after.bf16",value.contiguous().view(torch.uint16).cpu().numpy().astype('<u2',copy=False),'BF16')
                        after.append(dict(id=rec['id'],split=rec['split'],**item))
                dev_after=[v for v in after if v['split']=='DEV'];dev_before=[v for v in initial if v['split']=='DEV']
                ratio=sum(v['relative_L2'] for v in dev_after)/max(sum(v['relative_L2'] for v in dev_before),1e-30)
                gates=dict(DEV_improvement=ratio<=b['criteria']['DEV_ratio'],every_DEV_retention=all(v['ratio']<=b['criteria']['individual_DEV_ratio'] for v in dev_after),
                    absolute_DEV_error=all(v['relative_L2']<=b['criteria']['absolute_DEV_relative_L2'] for v in dev_after),
                    every_DEV_cosine=all(v['cosine']>=b['criteria']['DEV_cosine'] for v in dev_after))
                projections=[]
                for n in ('gate','up','down'):
                    scale=getattr(model,n+'_scale').detach();code=torch.round(getattr(model,n).detach()/scale[:,None]).clamp(-1,1).to(torch.int8)
                    pairs=((code[:,0::2]+1)*3+(code[:,1::2]+1)).T.contiguous()
                    assert torch.equal(pairs.T//3-1,code[:,0::2]) and torch.equal(pairs.T%3-1,code[:,1::2])
                    projections.append(dict(name=n,codes=raw_file(f'site{site:02d}.final.{n}.pairs.u8',pairs.cpu().numpy().astype('u1',copy=False),'U8'),
                        scale=raw_file(f'site{site:02d}.final.{n}.scale.f32',scale.cpu().numpy().astype('<f4',copy=False),'F32')))
                record=dict(site=site,steps=b['steps'],initial=initial,after=after,DEV_ratio=ratio,gates=gates,history=history,
                    price_state=price_state,final_state=final_state,projections=projections,input_scale=selected['input_scale'],
                    initial_sector_and_all_STE_forward_bits_equal=True,trainable_elements=28322816)
                write(a.directory/f'site{site:02d}.json',record);records.append(record);completed.append(site)
                event('site_complete',site=site,DEV_ratio=ratio,gates=gates);del model,opt,w,transformed,scales,data
        assert updates==512;phase='result';guard()
        decision='SOURCE_FFN_LOCAL_RECOVERY_PASS' if all(all(v['gates'].values()) for v in records) else 'SOURCE_FFN_LOCAL_RECOVERY_FAIL'
        write(a.out,dict(schema='SOURCE_FFN_LOCAL_RESULT_V1',phase='recover',variant=b['variant'],decision=decision,freeze=a.freeze,binding_sha256=a.binding_sha,
            process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),cases=4,records=records,optimizer_updates=updates,source_forwards=0,source_generations=0,
            native_runs=0,reserved_queries=0,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.monotonic()-start,quality_admission=False,native_admission=False,
            scope='Fixed256 updates per source FFN/two sites atsame selected rows/native arithmetic;local function evidence only,not whole source/compact chatbot/rate/n/DRAM/family admission.'))
        event('complete',decision=decision)
    except BaseException as error:
        failure=dict(fault=repr(error),phase=phase,completed=completed,optimizer_updates=updates,elapsed_seconds=time.monotonic()-start)
        if torch is not None:failure.update(GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved())
        write(a.directory/'first_failure.json',failure);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('bind','worker'),default='worker');p.add_argument('--binding',type=Path)
    p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();bind(a) if a.mode=='bind' else worker(a)
