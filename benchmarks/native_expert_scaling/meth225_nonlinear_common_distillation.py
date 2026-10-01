#!/usr/bin/env python3
"""Fixed BF16-effective H768 distillation on actual saved donor functions."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import time
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch import nn
from torch.nn import functional as F
import meth222_conditional_function_pilot as M
import meth224_common_native_feasibility as N

P,C=M.P,N.C
NATIVE=P.DOC/'meth224_common_native_feasibility_result.json'
NATIVE_SHA='bf99a82180e1757479b2eeb6d68ce432bee8ae499b88083b738fabe0d89c16fa'
PREVIOUS=P.DOC/'meth222_conditional_function_pilot_result.json'
PREVIOUS_SHA='41858ede5147f7c98821987892096acb0aa7abf3611db4a1b849663e1d0ddb34'
CAPTURE=P.ROOT/'results/native_expert_scaling/meth222_layer12_training_states.npz'
SEED,UPDATES,BATCH=225225,4096,256


class Common(nn.Module):
    def __init__(self,gate,up,down,bias):
        super().__init__()
        for name,value in (('gate',gate),('up',up),('down',down),('bias',bias)):
            setattr(self,name,nn.Parameter(value.float().clone().contiguous()))

    @staticmethod
    def effective(w):
        return w+(w.bfloat16().float()-w).detach()

    def forward(self,x):
        hidden=F.silu(F.linear(x,self.effective(self.gate)))*F.linear(x,self.effective(self.up))
        return F.linear(hidden,self.effective(self.down),self.bias)


def write_common(model,path,metadata):
    tensors={name+'.weight':getattr(model,name).detach().bfloat16().cpu().contiguous() for name in ('gate','up','down')}
    tensors['output.bias']=model.bias.detach().float().cpu().contiguous()
    save_file(tensors,str(path),metadata=metadata)
    with safe_open(str(path),framework='pt',device='cpu') as archive:
        assert set(archive.keys())==set(tensors)
        for name,value in tensors.items(): assert torch.equal(archive.get_tensor(name),value)
    return tensors


def evaluate(tensors,x,y,device,old):
    values={name:value.to(device).float() for name,value in tensors.items()}
    rows=[]
    with torch.no_grad():
        for seq in range(M.VALID):
            span=slice(seq*M.SEQ,(seq+1)*M.SEQ)
            hidden=F.silu(F.linear(x[span],values['gate.weight']))*F.linear(x[span],values['up.weight'])
            predicted=F.linear(hidden,values['down.weight'],values['output.bias'])
            energy=float(y[span].double().square().sum())
            assert energy==old['validation_rows'][seq]['energy']
            rows.append({'validation_sequence':seq,'energy':energy,
                'sse':float((predicted-y[span]).double().square().sum())})
    return {'rows':rows,'normalized_sse':sum(r['sse'] for r in rows)/sum(r['energy'] for r in rows)}


def trained_native(tensors,binary,exe,device,previous):
    fixture=P.ROOT/'results/native_expert_scaling/meth224_bf16_common_fixture.bin'
    assert P.digest(fixture)==previous['binary']['sha256'] and P.digest(exe)==previous['executable_sha256']
    shutil.copyfile(fixture,binary)
    replacement={f'model.layers.{M.LAYER}.mlp.{organ}_proj.weight':tensors[organ+'.weight'].view(torch.uint16).numpy().tobytes()
                 for organ in ('gate','up','down')}
    replacement['constructed_zero_output_bias']=tensors['output.bias'].numpy().tobytes()
    segments=[]
    with binary.open('r+b') as output:
        for row in previous['segments']:
            expected=row['sha256']
            changed=row['layer']==M.LAYER
            if changed:
                raw=replacement[row['name']]
                assert len(raw)==row['bytes']
                output.seek(row['offset']); output.write(raw)
                expected=P.M17.sha(raw)
            segments.append({**row,'expected_sha256':expected,'changed':changed})
    with binary.open('rb') as audit:
        for row in segments:
            audit.seek(row['offset']); assert P.M17.sha(audit.read(row['bytes']))==row['expected_sha256']
    assert sum(row['changed'] for row in segments)==4
    torch.cuda.synchronize(device)
    output_path=binary.with_suffix('.check.bin'); assert not output_path.exists()
    run=subprocess.run([str(exe),str(binary),str(C.VECTORS),str(output_path)],capture_output=True,text=True,check=True,timeout=600)
    timing=json.loads(run.stdout)
    assert timing['threads']==6 and timing['tokens']==256 and timing['layers']==24 and timing['hidden']==768
    raw=output_path.read_bytes()
    assert len(raw)==20+16*24*896*4 and struct.unpack_from('<8s3I',raw)==(b'M224OUT1',16,24,896)
    actual=np.frombuffer(raw,dtype='<f4',offset=20).reshape(16,24,896)
    original=np.frombuffer(fixture.with_suffix('.check.bin').read_bytes(),dtype='<f4',offset=20).reshape(16,24,896)
    other=[li for li in range(24) if li!=M.LAYER]
    assert np.array_equal(actual[:,other],original[:,other])
    states=np.frombuffer(C.VECTORS.read_bytes(),dtype='<u2',offset=24).reshape(256,24,896)
    x=torch.from_numpy(states[:16,M.LAYER].copy()).view(torch.bfloat16).to(device).float()
    values={name:value.to(device).float() for name,value in tensors.items()}
    with torch.no_grad():
        expected=F.linear(F.silu(F.linear(x,values['gate.weight']))*F.linear(x,values['up.weight']),
            values['down.weight'],values['output.bias']).cpu().numpy().astype(np.float64)
    errors=np.linalg.norm(actual[:,M.LAYER].astype(np.float64)-expected,axis=1)/np.linalg.norm(expected,axis=1)
    assert np.isfinite(actual).all() and np.isfinite(errors).all()
    return {'binary_sha256':P.digest(binary),'output_sha256':P.digest(output_path),'segments':segments,
        'other23_layers_output_bits_exact':True,'layer12_relative_l2':errors.tolist(),
        'median_relative_l2':float(np.median(errors)),'maximum_relative_l2':float(errors.max()),
        'numeric_pass':float(np.median(errors))<=1e-4 and float(errors.max())<=5e-4,
        'timing':timing,'cost_pass':statistics.median(timing['pass_ms_per_token'])<=10,
        'scope':'One trained layer plus23 untrained source fixture layers; component, not full quality/rate'}


def main():
    ap=argparse.ArgumentParser()
    for key in ('initial','checkpoint','binary','out'):
        ap.add_argument('--'+key,required=True,type=Path)
    args=ap.parse_args()
    assert all(not getattr(args,k).exists() for k in ('initial','checkpoint','binary','out'))
    for k in ('initial','checkpoint','binary','out'): getattr(args,k).parent.mkdir(parents=True,exist_ok=True)
    start,stage,logs=time.monotonic(),'bindings',[]
    try:
        assert P.digest(NATIVE)==NATIVE_SHA and P.digest(PREVIOUS)==PREVIOUS_SHA
        old=json.loads(PREVIOUS.read_text(encoding='utf-8')); native_old=json.loads(NATIVE.read_text(encoding='utf-8'))
        assert native_old['summary']['numeric_pass'] and native_old['summary']['cost_pass']
        assert P.digest(CAPTURE)==old['capture']['sha256'] and P.digest(C.VECTORS)==C.VECTORS_SHA
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        device=M.D.Q.setup(); P.MAX_SECONDS=P.M17.MAX_SECONDS=25*60
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(True)
        with np.load(CAPTURE,allow_pickle=False) as archive:
            x=torch.from_numpy(archive['x_bf16'].reshape(-1,896)).view(torch.bfloat16).to(device).float()
            y=torch.from_numpy(archive['y_bf16'].reshape(-1,896)).view(torch.bfloat16).to(device).float()
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        cut=M.FIT*M.SEQ; xf,yf=x[:cut],y[:cut]
        stage='source_prefix_and_ridge_down_initialization'
        with safe_open(str(source),framework='pt',device='cpu') as archive:
            gate=archive.get_tensor(f'model.layers.{M.LAYER}.mlp.gate_proj.weight')[:768].contiguous().to(device).float()
            up=archive.get_tensor(f'model.layers.{M.LAYER}.mlp.up_proj.weight')[:768].contiguous().to(device).float()
        with torch.no_grad():
            hidden=torch.cat([F.silu(F.linear(xf[first:first+4096],gate))*F.linear(xf[first:first+4096],up)
                              for first in range(0,cut,4096)])
            mh,my=hidden.double().mean(0),yf.double().mean(0)
            hc,yc=hidden.double()-mh,yf.double()-my
            gram=hc.T@hc/cut
            ridge=float(gram.trace()/768)*.001
            down=torch.linalg.solve(gram+ridge*torch.eye(768,dtype=torch.float64,device=device),(yc.T@hc/cut).T).T.bfloat16().float()
            bias=my.float()-down@mh.float()
        del hidden,hc,yc,gram
        model=Common(gate,up,down,bias)
        metadata={'experiment':'METH-225','layer':str(M.LAYER),'donor_sha256':P.M57.MODEL_SHA,
            'capture_sha256':old['capture']['sha256'],'compute':'BF16-effective weights, FP32 activation/control'}
        initial_tensors=write_common(model,args.initial,{**metadata,'stage':'ridge_initialized_source_prefix'})
        del gate,up,down,bias
        gc.collect(); torch.cuda.empty_cache()
        stage='fixed_4096_update_training'
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),eps=1e-8,weight_decay=0)
        rng=np.random.default_rng(SEED); order=None
        norm=float(yf.double().square().mean())
        torch.set_grad_enabled(True)
        rolling=[]
        for update in range(1,UPDATES+1):
            position=(update-1)%(cut//BATCH)
            if position==0: order=rng.permutation(cut)
            selected=torch.as_tensor(order[position*BATCH:(position+1)*BATCH],device=device)
            rate=3e-4*min(1,update/128)*(.1+.9*.5*(1+math.cos(math.pi*(update-1)/(UPDATES-1))))
            optimizer.param_groups[0]['lr']=rate
            optimizer.zero_grad(set_to_none=True)
            loss=(model(xf[selected])-yf[selected]).square().mean()/norm
            assert torch.isfinite(loss)
            loss.backward()
            gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True))
            optimizer.step(); rolling.append(float(loss.detach()))
            P.budget(start,device)
            if update%256==0:
                logs.append({'update':update,'lr':rate,'mean_training_normalized_loss':sum(rolling)/len(rolling),
                    'last_gradient_norm':gradient,'runtime':P.budget(start,device)})
                rolling.clear()
                args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'logs':logs},indent=2)+'\n',encoding='utf-8')
                print(json.dumps(logs[-1]),flush=True)
        torch.set_grad_enabled(False)
        final_tensors=write_common(model,args.checkpoint,{**metadata,'stage':'fixed_final_4096','updates':str(UPDATES)})
        decoded={name:value.to(device).float() for name,value in final_tensors.items()}
        check_x=xf[:8]
        expected=F.linear(F.silu(F.linear(check_x,decoded['gate.weight']))*F.linear(check_x,decoded['up.weight']),
            decoded['down.weight'],decoded['output.bias'])
        assert torch.equal(model(check_x),expected)
        stage='consumed_validation_once_after_training'
        initial=evaluate(initial_tensors,x[cut:],y[cut:],device,old)
        final=evaluate(final_tensors,x[cut:],y[cut:],device,old)
        rng=np.random.default_rng(SEED+1); draws=rng.integers(0,M.VALID,size=(10000,M.VALID))
        delta=np.asarray([a['sse']-b['sse'] for a,b in zip(initial['rows'],final['rows'])])
        energy=np.asarray([a['energy'] for a in initial['rows']])
        gains=delta[draws].sum(1)/energy[draws].sum(1)
        stage='trained_layer_native_parity'
        exe=P.ROOT/'benchmarks/native_expert_scaling/meth224_bf16_common_cpu.exe'
        native=trained_native(final_tensors,args.binary,exe,device,native_old)
        gates={'intermediate_common_validation_nmse_le_010':final['normalized_sse']<=.10,
            'trained_common_at_least_10_percent_better_than_ridge_initial':final['normalized_sse']<=.9*initial['normalized_sse'],
            'paired_validation_gain_p05_positive':float(np.quantile(gains,.05))>0,
            'stored_weight_forward_exact':True,'trained_native_numeric':native['numeric_pass'],
            'mixed_fixture_native_cost':native['cost_pass']}
        result={'experiment':'METH-225-nonlinear-common-output-distillation','layer':M.LAYER,'hidden':768,
            'source_sha256':P.M57.MODEL_SHA,'capture_sha256':old['capture']['sha256'],'prior_result_sha256':PREVIOUS_SHA,
            'native_fixture_result_sha256':NATIVE_SHA,'seed':SEED,'updates':UPDATES,'batch':BATCH,
            'unique_fit_token_states':cut,'fit_state_exposures':UPDATES*BATCH,'initial_ridge':ridge,'logs':logs,
            'initial_checkpoint_sha256':P.digest(args.initial),'final_checkpoint_sha256':P.digest(args.checkpoint),
            'checkpoint_bytes':args.checkpoint.stat().st_size,'every_saved_tensor_readback_equal':True,
            'initial_validation':initial,'final_validation':final,
            'bootstrap':{'gain_p05':float(np.quantile(gains,.05)),'gain_p95':float(np.quantile(gains,.95)),
                'seed':SEED+1,'draws':10000,'unit':'consumed_training_corpus_sequence'},
            'native':native,'gates':gates,'runtime':P.budget(start,device),'script_sha256':P.digest(Path(__file__)),
            'torch_version':torch.__version__,'decision':'intermediate_common_pass_freeze_parent_anchored_function_recovery' if all(gates.values()) else 'stop_this_fixed_nonlinear_common_distillation',
            'scope':'Intermediate one-layer consumed-state function distillation; full conditional .01 function gate and all final LLM quality/rate/large-n/multi-family requirements unchanged'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'decision':result['decision'],'initial_nmse':initial['normalized_sse'],'final_nmse':final['normalized_sse'],
            'gates':gates,'native_median_ms':statistics.median(native['timing']['pass_ms_per_token']),'runtime':result['runtime']}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),
            'elapsed_seconds':time.monotonic()-start,'logs':logs},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__':
    main()
