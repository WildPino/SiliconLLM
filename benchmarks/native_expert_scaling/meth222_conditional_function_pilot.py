#!/usr/bin/env python3
"""Fit common and conditional actual donor FFN functions, E16 versus E160."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import meth219_native_arithmetic_development as D
import meth136_matched_sparse_train as T

P=D.P
SEED,LAYER,RANK,PARENTS,CHILDREN=222222,12,64,16,10
FIT,VALID,SEQ=512,128,128


def assign(q,centers):
    distances=q.square().sum(1)[:,None]+centers.square().sum(1)[None]-2*q@centers.T
    return distances.argmin(1)


def kmeans(q,k,seed):
    assert len(q)>=k
    rng=np.random.default_rng(seed)
    centers=q[torch.as_tensor(rng.choice(len(q),k,replace=False),device=q.device)].clone()
    for _ in range(20):
        labels=assign(q,centers)
        for c in range(k):
            selected=q[labels==c]
            if len(selected):
                centers[c]=selected.mean(0)
    return centers,assign(q,centers)


def fit_cells(q,residual,labels,count):
    coefficients=torch.zeros((count,896,RANK+1),dtype=torch.bfloat16,device=q.device)
    counts=[]
    for c in range(count):
        mask=labels==c
        x,y=q[mask].double(),residual[mask].double()
        counts.append(len(x))
        if not len(x):
            continue
        mx,my=x.mean(0),y.mean(0)
        xc,yc=x-mx,y-my
        gram=xc.T@xc/len(x)
        ridge=max(float(gram.trace()/RANK)*.001,1e-12)
        weight=torch.linalg.solve(gram+ridge*torch.eye(RANK,dtype=torch.float64,device=q.device),xc.T@yc/len(x)).T
        bias=my-weight@mx
        coefficients[c,:,:RANK]=weight.bfloat16()
        coefficients[c,:,RANK]=bias.bfloat16()
    return coefficients,counts


def main():
    ap=argparse.ArgumentParser()
    for key in ('capture','checkpoint','out'):
        ap.add_argument('--'+key,required=True,type=Path)
    args=ap.parse_args()
    assert all(not getattr(args,k).exists() for k in ('capture','checkpoint','out'))
    for k in ('capture','checkpoint','out'):
        getattr(args,k).parent.mkdir(parents=True,exist_ok=True)
    start,stage=time.monotonic(),'bindings'
    progress={}
    def partial():
        args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'progress':progress,
            'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
    try:
        assert P.digest(T.M15.TRAIN_PATH)==T.M15.TRAIN_FILE_SHA
        excluded=set()
        for path,digest in ((T.M56.TRAIN_CHAT_PATH,T.M56.TRAIN_CHAT_SHA),
            (T.M56.DEV_PATH,T.M56.DEV_SHA),(T.M56.OLD_DEV_PATH,T.M56.OLD_DEV_SHA),
            (T.M44.PROMPT_PATH,T.M44.PROMPT_SHA),(T.M56.M47_DEV_PATH,T.M56.M47_DEV_SHA)):
            assert P.digest(path)==digest
            excluded.update(r['train_row'] for r in json.loads(Path(path).read_text(encoding='utf-8'))['rows'])
        assert len(excluded)==352
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA
        with np.load(T.M15.TRAIN_PATH,allow_pickle=False) as archive:
            ids=archive['ids'].copy()
        assert ids.shape==(31250,512) and ids.dtype==np.int32 and T.M15.sha_bytes(ids.tobytes())==T.M15.TRAIN_IDS_SHA
        rng=np.random.default_rng(SEED)
        pool=np.asarray([i for i in range(len(ids)) if i not in excluded])
        row_ids=rng.choice(pool,FIT+VALID,replace=False)
        offsets=rng.integers(0,512-SEQ+1,size=FIT+VALID)
        selected=np.stack([ids[r,o:o+SEQ] for r,o in zip(row_ids,offsets)])
        assert len(set(row_ids[:FIT])&set(row_ids[FIT:]))==0
        device=D.Q.setup()
        P.MAX_SECONDS=P.M17.MAX_SECONDS=25*60
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        stage='capture_source_layer12'
        model=AutoModelForCausalLM.from_pretrained(P.M42.MODEL,revision=P.M42.REV,
            dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to(device).eval()
        model.config.use_cache=False
        xbits=np.empty((FIT+VALID,SEQ,896),dtype=np.uint16)
        ybits=np.empty_like(xbits)
        captured={}
        def hook(module,inputs,output):
            assert inputs[0].dtype==output.dtype==torch.bfloat16
            assert inputs[0].shape==output.shape==(4,SEQ,896)
            captured['x']=inputs[0].view(torch.uint16).cpu().numpy().copy()
            captured['y']=output.view(torch.uint16).cpu().numpy().copy()
        handle=model.model.layers[LAYER].mlp.register_forward_hook(hook)
        with torch.inference_mode():
            for first in range(0,FIT+VALID,4):
                captured.clear()
                batch=torch.from_numpy(selected[first:first+4].astype(np.int64)).to(device)
                model.model(batch,use_cache=False)
                assert set(captured)=={'x','y'}
                xbits[first:first+4],ybits[first:first+4]=captured['x'],captured['y']
                P.budget(start,device)
                if (first+4)%128==0:
                    print(json.dumps({'stage':stage,'sequences':first+4,'runtime':P.budget(start,device)}),flush=True)
        handle.remove()
        del model,captured,batch
        gc.collect(); torch.cuda.empty_cache()
        np.savez(args.capture,x_bf16=xbits,y_bf16=ybits,row_ids=row_ids.astype(np.int32),
            offsets=offsets.astype(np.int32),input_ids=selected)
        with np.load(args.capture,allow_pickle=False) as audit:
            assert np.array_equal(audit['x_bf16'],xbits) and np.array_equal(audit['y_bf16'],ybits)
            assert np.array_equal(audit['input_ids'],selected)
        capture_sha=P.digest(args.capture)
        progress['capture']={'sha256':capture_sha,'bytes':args.capture.stat().st_size,
            'x_sha256':P.M17.sha(xbits.tobytes()),'y_sha256':P.M17.sha(ybits.tobytes()),
            'input_ids_sha256':P.M17.sha(selected.tobytes()),'rows':row_ids.tolist(),'offsets':offsets.tolist()}
        partial()
        x=torch.from_numpy(xbits.reshape(-1,896)).view(torch.bfloat16).to(device).float()
        y=torch.from_numpy(ybits.reshape(-1,896)).view(torch.bfloat16).to(device).float()
        del xbits,ybits,ids,selected
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        xf,yf=x[:FIT*SEQ],y[:FIT*SEQ]
        stage='fit_common_and_shared_projection'
        mx,my=xf.double().mean(0),yf.double().mean(0)
        xc,yc=xf.double()-mx,yf.double()-my
        gram=xc.T@xc/len(xf)
        cross=yc.T@xc/len(xf)
        ridge=float(gram.trace()/896)*.001
        w=torch.linalg.solve(gram+ridge*torch.eye(896,dtype=torch.float64,device=device),cross.T).T.bfloat16()
        bias=my.float()-w.float()@mx.float()
        eigenvalues,eigenvectors=torch.linalg.eigh(gram)
        assert (eigenvalues[-RANK:]>0).all()
        projection=(eigenvectors[:,-RANK:].T/eigenvalues[-RANK:].sqrt()[:,None]).bfloat16()
        qbias=-projection.float()@mx.float()
        del xc,yc,cross,gram,eigenvectors
        common=F.linear(x,w.float(),bias)
        q=F.linear(x,projection.float(),qbias)
        residual=yf-common[:FIT*SEQ]
        stage='fit_hierarchical_function_cells'
        keys16,labels16=kmeans(q[:FIT*SEQ],PARENTS,SEED)
        keys10=torch.empty((PARENTS,CHILDREN,RANK),device=device)
        labels160=torch.empty_like(labels16)
        for parent in range(PARENTS):
            mask=labels16==parent
            keys10[parent],local=kmeans(q[:FIT*SEQ][mask],CHILDREN,SEED+parent+1)
            labels160[mask]=parent*CHILDREN+local
        b16,counts16=fit_cells(q[:FIT*SEQ],residual,labels16,PARENTS)
        b160,counts160=fit_cells(q[:FIT*SEQ],residual,labels160,PARENTS*CHILDREN)
        fit_gates={'all_cells_occupied':min(counts160)>0,
            'at_least_90_percent_cells_have_65_samples':sum(n>=65 for n in counts160)>=.9*len(counts160)}
        tensors={'common.weight':w.cpu(),'common.bias':bias.cpu(),'projection.weight':projection.cpu(),
            'projection.bias':qbias.cpu(),'router.parent':keys16.cpu(),'router.children':keys10.cpu(),
            'experts.e16':b16.cpu(),'experts.e160':b160.cpu()}
        save_file(tensors,str(args.checkpoint),metadata={'experiment':'METH-222','layer':str(LAYER),
            'donor_sha256':P.M57.MODEL_SHA,'capture_sha256':capture_sha,'compute':'FP32 with BF16 weight values'})
        with safe_open(str(args.checkpoint),framework='pt',device='cpu') as archive:
            assert set(archive.keys())==set(tensors)
            for name,value in tensors.items():
                assert torch.equal(archive.get_tensor(name),value)
        progress['fit']={'counts16':counts16,'counts160':counts160,'gates':fit_gates,
            'checkpoint_sha256':P.digest(args.checkpoint),'every_tensor_readback_equal':True,
            'input_pca64_fit_variance_fraction':float(eigenvalues[-RANK:].sum()/eigenvalues.sum())}
        partial()
        stage='validation_function_screen' if all(fit_gates.values()) else 'fit_gate_stop'
        rows=[]
        gates={**fit_gates}
        summary={}
        if all(fit_gates.values()):
            # Score only the reloaded BF16-effective coefficient tensors.
            with safe_open(str(args.checkpoint),framework='pt',device='cpu') as archive:
                saved={name:archive.get_tensor(name).to(device).float() for name in archive.keys()}
            qv=F.linear(x[FIT*SEQ:],saved['projection.weight'],saved['projection.bias'])
            commonv=F.linear(x[FIT*SEQ:],saved['common.weight'],saved['common.bias'])
            pv=assign(qv,saved['router.parent'])
            localv=((qv[:,None,:]-saved['router.children'][pv]).square().sum(-1)).argmin(-1)
            cv=pv*CHILDREN+localv
            rotated=pv*CHILDREN+(localv+1)%CHILDREN
            val_sse={arm:[] for arm in ('common','e16','e160','e160_rotated')}
            energy=[]
            for seq in range(VALID):
                span=slice(seq*SEQ,(seq+1)*SEQ)
                target=y[FIT*SEQ:][span]
                prediction=commonv[span]
                features=torch.cat((qv[span],torch.ones((SEQ,1),device=device)),dim=1)
                values={'common':prediction}
                for arm,bank,labels in (('e16','experts.e16',pv),('e160','experts.e160',cv),
                                         ('e160_rotated','experts.e160',rotated)):
                    values[arm]=prediction+torch.bmm(saved[bank][labels[span]],features[:,:,None]).squeeze(-1)
                row={'validation_sequence':seq,'raw_row':int(row_ids[FIT+seq]),'raw_offset':int(offsets[FIT+seq]),
                    'energy':float(target.double().square().sum()),'sse':{}}
                for arm,value in values.items():
                    row['sse'][arm]=float((value-target).double().square().sum())
                    val_sse[arm].append(row['sse'][arm])
                energy.append(row['energy']); rows.append(row)
                P.budget(start,device)
            total_energy=sum(energy)
            summary={arm:{'sse':sum(values),'normalized_sse':sum(values)/total_energy} for arm,values in val_sse.items()}
            rng=np.random.default_rng(SEED+1)
            draws=rng.integers(0,VALID,size=(10000,VALID))
            paired=np.asarray(val_sse['e16'])-np.asarray(val_sse['e160'])
            gain=paired[draws].sum(1)/np.asarray(energy)[draws].sum(1)
            summary['paired_validation_gain']={'normalized_sse_gain_p05':float(np.quantile(gain,.05)),
                'normalized_sse_gain_p95':float(np.quantile(gain,.95)),'unit':'training_corpus_sequence','draws':10000,'seed':SEED+1}
            gates.update({'candidate_normalized_sse_le_001':summary['e160']['normalized_sse']<=.01,
                'candidate_sse_at_least_10_percent_better_than_e16':summary['e160']['sse']<=.9*summary['e16']['sse'],
                'candidate_sse_at_least_10_percent_better_than_rotated':summary['e160']['sse']<=.9*summary['e160_rotated']['sse'],
                'validation_sequence_gain_p05_positive':float(np.quantile(gain,.05))>0})
        result={'experiment':'METH-222-conditional-pretrained-function-pilot','seed':SEED,'layer':LAYER,
            'donor_sha256':P.M57.MODEL_SHA,'donor_revision':P.M42.REV,'training_file_sha256':T.M15.TRAIN_FILE_SHA,
            'fit_sequences':FIT,'validation_sequences':VALID,'sequence_tokens':SEQ,
            'rank':RANK,'parent_count':PARENTS,'children':CHILDREN,'active_cells':1,'lloyd_steps':20,
            'capture':progress['capture'],'fit':progress['fit'],'checkpoint_bytes':args.checkpoint.stat().st_size,
            'summary':summary,'gates':gates,'validation_rows':rows,
            'decision':'local_function_and_n_gain_pass_native_cost_and_composition_next' if all(gates.values()) else 'local_function_or_n_gain_fail_stop_this_fixed_geometry',
            'runtime':P.budget(start,device),'torch_version':torch.__version__,'numpy_version':np.__version__,
            'script_sha256':P.digest(Path(__file__)),
            'scope':'One source donor FFN on separate training-corpus rows; not document-held-out LLM quality, generation/task, native performance, multi-layer or 10B/general scaling'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'decision':result['decision'],'summary':summary,'gates':gates,'runtime':result['runtime']}),flush=True)
    except BaseException as error:
        partial()
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),
            'elapsed_seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__':
    main()
