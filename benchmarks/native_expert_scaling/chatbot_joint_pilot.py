"""One frozen layer12 jointly learned compact SwiGLU trial, not a whole model.

Captured original BF16 x/y are immutable targets. Geometry uses FIT x only;
novel development excludes exact shared-prefix inputs. All finite thresholds,
initialization, updates and interventions are prewritten in the pilot protocol.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once


def load_operands(adoption):
    import numpy as np
    groups={s:dict(x=[],y=[],case=[],category=[],generated=[]) for s in ('fit','development')}
    for case_id,case in enumerate(adoption['cases']):
        path=Path(case['binary_path']);assert sha(path)==case['binary_SHA256']
        assert path.stat().st_size==case['binary_bytes'] and sha(case['journal_path'])==case['journal_SHA256']
        with path.open('rb') as f:assert struct.unpack('<8s4I',f.read(24))==(b'QWCAP001',24,896,2,0)
        bits=np.memmap(path,dtype='<u2',mode='r',offset=24,shape=(case['captured_rows'],24,2,896))
        g=groups[case['split']];n=len(bits)
        g['x'].append(np.array(bits[:,12,0,:],copy=True,order='C'))
        g['y'].append(np.array(bits[:,12,1,:],copy=True,order='C'))
        g['case'].extend([case_id]*n);g['category'].extend([case['category']]*n)
        g['generated'].extend([False]*len(case['prompt_ids'])+[True]*(n-len(case['prompt_ids'])))
        del bits
    for g in groups.values():
        g['x_bits']=np.concatenate(g.pop('x'));g['y_bits']=np.concatenate(g.pop('y'))
        for name in ('x','y'):
            bits=g[name+'_bits'];assert bits.dtype==np.dtype('<u2') and bits.flags.c_contiguous
            assert not ((bits & 0x7fff)>=0x7f80).any()
            g[name]=(bits.astype('<u4')<<16).view('<f4').copy(order='C')
        for name in ('case','category','generated'):g[name]=np.asarray(g[name])
        _,g['unique_index'],g['inverse'],g['counts']=np.unique(g['x_bits'],axis=0,return_index=True,return_inverse=True,return_counts=True)
        g['weights']=1/g['counts'][g['inverse']].astype('float64')
    fit_bytes={row.tobytes() for row in groups['fit']['x_bits']}
    dev=groups['development'];dev['novel']=np.array([row.tobytes() not in fit_bytes for row in dev['x_bits']])
    assert dev['novel'].any()
    return groups


def clusters(query,count):
    """Deterministic FIT-only farthest-first + exactly20 Lloyd iterations."""
    import numpy as np
    assert query.ndim==2 and query.shape[1]==32 and len(query)>=count
    first=int(np.square(query-query.mean(0)).sum(1).argmin());chosen=[first]
    distance=np.square(query-query[first]).sum(1)
    for _ in range(count-1):
        index=int(distance.argmax());assert index not in chosen,'insufficient distinct query support'
        chosen.append(index);distance=np.minimum(distance,np.square(query-query[index]).sum(1))
    centers=query[chosen].copy();empty=[]
    for iteration in range(20):
        score=2*query@centers.T-np.square(centers).sum(1)
        assigned=score.argmax(1)
        for index in range(count):
            selected=query[assigned==index]
            if len(selected):centers[index]=selected.mean(0)
            else:empty.append([iteration,index])
    return centers.astype('float32'),dict(initial_FIT_indices=chosen,empty_iterations=empty)


def geometry(fit,device):
    import numpy as np
    import torch
    unique=fit['x'][fit['unique_index']].astype('float64');centered=unique-unique.mean(0)
    covariance=torch.from_numpy(np.ascontiguousarray(centered.T@centered/len(unique)))
    values,vectors=torch.linalg.eigh(covariance);values=values[-32:].flip(0)
    vectors=vectors[:,-32:].flip(1).T.contiguous()
    for row in vectors:
        if row[row.abs().argmax()]<0:row.neg_()
    assert torch.isfinite(values).all() and bool((values>0).all())
    projection=(vectors/values.sum().sqrt()).float().contiguous().to(device)
    unique_x=torch.from_numpy(fit['x'][fit['unique_index']].copy(order='C')).to(device)
    query=torch.nn.functional.linear(unique_x,projection).cpu().numpy().astype('float64')
    parents,receipt=clusters(query,16)
    parent=torch.from_numpy(parents).to(device)
    q=torch.nn.functional.linear(unique_x,projection)
    score=2*torch.nn.functional.linear(q,parent)-parent.square().sum(-1)
    choice=torch.argsort(score,dim=-1,descending=True,stable=True)[:,:4].cpu().numpy()
    children=[];child_receipts=[]
    for index in range(16):
        selected=query[(choice==index).any(1)]
        child,child_receipt=clusters(selected,10)
        children.append(child);child_receipts.append(child_receipt)
    return projection,parent,torch.from_numpy(np.stack(children)).to(device),dict(
        covariance='unique FIT x, F64, top32 canonical signs, scale sqrt(sum eigenvalues)',
        eigenvalues=values.tolist(),parents=receipt,children=child_receipts)


def exposure(block,groups):
    import numpy as np
    import torch
    rows={};routes={};good=True
    with torch.no_grad():
        for split,g in groups.items():
            ids,mass=block.routes(g['tx']);routes[split]=(ids,mass)
            ids_cpu=ids.cpu().numpy();mass_cpu=mass.cpu().numpy()
            eligibility=np.ones(len(ids_cpu),dtype=bool) if split=='fit' else g['novel']
            for leaf in range(len(block.leaf_g)):
                mask=(ids_cpu==leaf).any(1)&eligibility
                unique=np.unique(g['inverse'][mask]);cases=np.unique(g['case'][mask])
                item=rows.setdefault(str(leaf),{})
                item[split]=dict(selected_occurrences=int(mask.sum()),distinct_x=len(unique),
                    conversations=cases.tolist(),mass_squared_sum=float(np.square(mass_cpu[ids_cpu==leaf]).sum()))
                good &= len(unique)>=(8 if split=='fit' else 4)
                if split=='fit':good &= len(cases)>=2
    return rows,routes,bool(good)


def parameter_groups(block):
    yield 'shared',(block.shared_g,block.shared_u,block.shared_b)
    for index in range(len(block.leaf_g)):
        yield str(index),(block.leaf_g[index],block.leaf_u[index],block.leaf_b[index])


def coefficients(block):
    import torch
    result={name:getattr(block,name).detach().cpu().contiguous() for name in (
        'projection','parent_centers','child_centers','parent_norms','child_norms','shared_g','shared_u','shared_b')}
    for name in ('leaf_g','leaf_u','leaf_b'):
        result[name]=torch.stack([p.detach().cpu() for p in getattr(block,name)]).contiguous()
    return result


def save_tensors(path,value):
    import torch
    with path.open('xb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def source_features(fit,source,guard):
    import torch
    from torch.nn import functional as F
    gate,up,down=source
    x=fit['tx'][torch.as_tensor(fit['unique_index'],device=gate.device)]
    # NEW source-informed feature information, never a replacement y target.
    activation=[]
    with torch.no_grad():
        for start in range(0,len(x),256):
            activation.append((F.silu(F.linear(x[start:start+256],gate))*F.linear(x[start:start+256],up)).square())
            guard()
        energy=torch.cat(activation);column_energy=down.square().sum(0)
        global_score=energy.mean(0)*column_energy
        shared=torch.argsort(global_score,descending=True,stable=True)[:512]
    return energy,column_energy,shared


def source_initializer(block,fit,source,features,source_indices,guard):
    import torch
    gate,up,down=source
    energy,column_energy,shared=features
    x=fit['tx'][torch.as_tensor(fit['unique_index'],device=gate.device)]
    ids,mass=block.routes(x)
    with torch.no_grad():
        available=torch.ones(4864,dtype=torch.bool,device=x.device);available[shared]=False
        leaves=[]
        for leaf in range(len(block.leaf_g)):
            weight=(mass.square()*(ids==leaf)).sum(1)
            assert float(weight.sum())>0
            score=(energy*weight[:,None]).sum(0)/weight.sum()*column_energy
            score[~available]=-float('inf')
            leaves.append(torch.argsort(score,descending=True,stable=True)[:128])
        indices=torch.stack(leaves).contiguous()
        block.initialize_from_source(gate,up,down,shared.contiguous(),indices)
    source_indices.update(shared_source_indices=shared.tolist(),leaf_source_indices=indices.tolist(),
        source_feature_evaluations=len(x),source_full_response_evaluations=0,
        initialization='unscaled source rows selected by activation-squared times down-column energy')


def errors(prediction,g,novel_only=False,selected_ids=None):
    import numpy as np
    mask=g['novel'] if novel_only else np.ones(len(g['x']),dtype=bool)
    assert mask.any()
    error=np.square(prediction.astype('float64')-g['y'].astype('float64')).sum(1)
    energy=np.square(g['y'].astype('float64')).sum(1)
    def metric(selected,weighted):
        if not selected.any():return dict(rows=0,squared_ratio=None,relative_RMS=None)
        w=g['weights'][selected] if weighted else np.ones(int(selected.sum()))
        numerator=float((error[selected]*w).sum());denominator=float((energy[selected]*w).sum())
        assert denominator>0 and math.isfinite(numerator)
        return dict(rows=int(selected.sum()),error_energy=numerator,source_energy=denominator,
            squared_ratio=numerator/denominator,relative_RMS=math.sqrt(numerator/denominator))
    result=dict(occurrences=metric(mask,False),unique_x_weighted=metric(mask,True),
        all_captured_occurrences=metric(np.ones(len(mask),dtype=bool),False),
        categories={str(c):metric(mask&(g['category']==c),True) for c in sorted(set(g['category']))},
        generated=metric(mask&g['generated'],True),prompt=metric(mask&~g['generated'],True))
    if selected_ids is not None:
        result['whole_error_conditioned_on_actual_selected_leaf']={str(i):metric(mask&(selected_ids==i).any(1),True) for i in sorted(set(selected_ids.ravel()))}
    return result


def predict(block,g,routes,guard,cycled=False):
    import torch
    ids,mass=routes
    if cycled:ids=(ids//10)*10+(ids%10+1)%10
    result=[]
    with torch.no_grad():
        for start in range(0,len(g['x']),128):
            result.append(block.evaluate_selected(g['tx'][start:start+128],ids[start:start+128],mass[start:start+128]).cpu())
            guard()
    prediction=torch.cat(result).numpy();assert prediction.shape==g['y'].shape and prediction.dtype.name=='float32'
    assert prediction.flags.c_contiguous and bool(torch.isfinite(torch.from_numpy(prediction)).all())
    return prediction


def train(block,fit,routes,directory,children,guard):
    import torch
    initial={name:tuple(v.detach().clone() for v in values) for name,values in parameter_groups(block)}
    denominators={name:sum(v.square().sum() for v in values) for name,values in initial.items()}
    assert all(float(v)>0 for v in denominators.values())
    optimizer=torch.optim.Adam(block.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
    generator=torch.Generator(device='cpu').manual_seed(2601007)
    weights=torch.as_tensor(fit['weights'],dtype=torch.float32,device=fit['tx'].device)
    target=torch.from_numpy(fit['y']).to(fit['tx'].device)
    mean_energy=(weights*target.square().sum(1)).mean();assert float(mean_energy)>0
    curve=[];updates=0
    with (directory/f'E{16*children}.updates.jsonl').open('xb') as journal:
        for epoch in range(24):
            order=torch.randperm(len(target),generator=generator).to(target.device)
            total_loss=0.;seen=0
            for start in range(0,len(target),128):
                guard();indices=order[start:start+128];ids,mass=(v[indices] for v in routes)
                optimizer.zero_grad(set_to_none=True)
                prediction=block.evaluate_selected(fit['tx'][indices].contiguous(),ids,mass)
                data_loss=(weights[indices]*(prediction-target[indices]).square().sum(1)).mean()/mean_energy
                active=torch.unique(ids,sorted=True).tolist()
                def anchor(name,values):
                    return sum((v-original).square().sum() for v,original in zip(values,initial[name]))/denominators[name]
                shared_anchor=anchor('shared',(block.shared_g,block.shared_u,block.shared_b))
                leaves_anchor=sum(anchor(str(i),(block.leaf_g[i],block.leaf_u[i],block.leaf_b[i])) for i in active)/len(active)
                loss=data_loss+.001*(shared_anchor+leaves_anchor)
                assert bool(torch.isfinite(loss))
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(block.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();updates+=1
                value=float(data_loss.detach());total_loss+=value*len(indices);seen+=len(indices)
                row=dict(epoch=epoch+1,update=updates,rows=len(indices),data_loss=value,
                    anchor=float((shared_anchor+leaves_anchor).detach()),preclip_gradient_norm=float(norm))
                journal.write((json.dumps(row,separators=(',',':'),allow_nan=False)+'\n').encode('utf8'));journal.flush()
                guard()
            os.fsync(journal.fileno());curve.append(dict(epoch=epoch+1,weighted_training_minibatch_loss=total_loss/seen,updates=updates))
            print(json.dumps(dict(arm=16*children,epoch=epoch+1,updates=updates,loss=total_loss/seen)),flush=True)
    displacement={name:float(sum((v.detach()-original).square().sum() for v,original in zip(values,initial[name]))/denominators[name])
        for name,values in parameter_groups(block)}
    del optimizer,initial,denominators
    return dict(epochs=24,updates=updates,curve=curve,normalized_coefficient_displacement=displacement,
        update_journal=dict(path=str(directory/f'E{16*children}.updates.jsonl'),sha256=sha(directory/f'E{16*children}.updates.jsonl')))


def leaf_hashes(block):
    import torch
    out=[]
    for index in range(len(block.leaf_g)):
        h=hashlib.sha256()
        for value in (block.leaf_g[index],block.leaf_u[index],block.leaf_b[index]):
            h.update(value.detach().to(torch.bfloat16).cpu().contiguous().view(torch.uint16).numpy().tobytes())
        out.append(h.hexdigest())
    return out


def main(args):
    start=time.monotonic()
    import psutil
    proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_JOINT_LAYER12_PILOT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),gates={},arms=[],
        scope='COMPONENT_DEVELOPMENT_NOT_WHOLE_CHAT_QUALITY_RATE_OR_NATIVE_PARITY',new_source_full_forwards=0)
    torch=None;stage='binding';arm_start=None
    def resource():
        return dict(elapsed_seconds=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            GPU_peak_allocated=0 if torch is None or not torch.cuda.is_initialized() else torch.cuda.max_memory_allocated(),
            GPU_peak_reserved=0 if torch is None or not torch.cuda.is_initialized() else torch.cuda.max_memory_reserved())
    def guard():
        value=resource();assert value['elapsed_seconds']<=900 and value['OS_peak_snapshot']<=12<<30,'worker time/OS budget'
        assert value['GPU_peak_allocated']<=9<<30 and value['GPU_peak_reserved']<=10<<30,'GPU budget'
        assert arm_start is None or time.monotonic()-arm_start<=240,'finite arm240s budget'
        assert not proc.children(recursive=True),'unexpected descendants'
        assert not args.directory.exists() or sum(v.stat().st_size for v in args.directory.iterdir() if v.is_file())<=1<<30,'output cap'
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['schema']=='QWEN_JOINT_LAYER12_PILOT_BINDING_V1'
        for item in b['inputs']:
            assert str(Path(item['path']).resolve())==item['resolved_path']
            assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256'],item['path']
            guard()
        args.directory.mkdir();r['gates']['frozen_inputs']=True
        def no_subprocess(event,arguments):
            if event=='subprocess.Popen':raise RuntimeError('Pilot forbids subprocess: '+str(arguments[:2]))
        sys.addaudithook(no_subprocess);stage='imports'
        print(json.dumps(dict(stage=stage)),flush=True)
        import numpy as np
        import torch as torch_module
        torch=torch_module
        from safetensors import safe_open
        from chatbot_compact_geometry import build_block,validate
        assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6'
        assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
        assert torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3060'
        torch.set_num_threads(6);torch.set_num_interop_threads(1)
        torch.manual_seed(2601007);torch.cuda.manual_seed_all(2601007);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.cuda.reset_peak_memory_stats();device=torch.device('cuda:0')
        spec=validate(json.loads(Path(b['spec_path']).read_bytes()))
        adoption=json.loads(Path(b['adoption_path']).read_bytes())
        assert adoption['all_saved_frame_bytes_and_alignment_qualified'] and not adoption['missing_case_ids']
        assert (adoption['calibration_case_coverage'],adoption['fit_cases'],adoption['development_cases'],adoption['complete_requested_generations'])==(48,32,16,47)
        stage='original_operand_and_FIT_geometry'
        groups=load_operands(adoption);fit=groups['fit'];dev=groups['development']
        for g in groups.values():g['tx']=torch.from_numpy(g['x']).to(device)
        projection,parent,child,geometry_receipt=geometry(fit,device);guard()
        r['routing_coefficients_F32']=save_tensors(args.directory/'routing_F32.pt',dict(
            projection=projection.cpu(),parent_centers=parent.cpu(),child_centers=child.cpu(),
            parent_norms=parent.square().sum(-1).cpu(),child_norms=child.square().sum(-1).cpu()))
        r['data']={s:dict(rows=len(g['x']),distinct_x=len(g['unique_index']),
            generated_rows=int(g['generated'].sum()),novel_vs_FIT=int(g['novel'].sum()) if s=='development' else None) for s,g in groups.items()}
        r['geometry']=geometry_receipt
        blocks={};routes={};exposure_good=True
        for children in (1,10):
            centers=parent[:,None,:].contiguous() if children==1 else child.contiguous()
            block=build_block(spec,children,projection,parent,centers)
            receipt,arm_routes,eligible=exposure(block,groups)
            blocks[children]=block;routes[children]=arm_routes;exposure_good &= eligible
            write_once(args.directory/f'E{16*children}.exposure.json',dict(leaves=receipt,eligible=eligible))
            save_tensors(args.directory/f'E{16*children}.routes_F32.pt',
                {split:dict(ids=values[0].cpu(),mass=values[1].cpu()) for split,values in arm_routes.items()})
        r['gates']['all_leaf_FIT_and_novel_development_exposure']=exposure_good
        print(json.dumps(dict(stage=stage,data=r['data'],exposure_eligible=exposure_good)),flush=True)
        if not exposure_good:
            r['decision']='CLOSE_FINITE_PILOT_ON_EXPOSURE_NO_FIT';r['updates']=0
        else:
            stage='new_source_row_initialization'
            with safe_open(b['source_weights'],framework='pt',device='cpu') as source_file:
                source=[]
                for name,shape in (('gate_proj',(4864,896)),('up_proj',(4864,896)),('down_proj',(896,4864))):
                    value=source_file.get_tensor(f'model.layers.12.mlp.{name}.weight')
                    assert value.shape==shape and value.dtype==torch.bfloat16
                    source.append(value.to(device=device,dtype=torch.float32).contiguous())
            assert all(bool(torch.isfinite(v).all()) for v in source)
            features=source_features(fit,source,guard)
            r['source_initializer_information']=dict(unique_FIT_feature_rows=len(features[0]),
                source_full_response_replays=0,feature_matrix_reused_between_arms=True,
                feature_energy=save_tensors(args.directory/'source_FIT_feature_energy.pt',dict(energy=features[0].cpu(),column_energy=features[1].cpu(),shared_indices=features[2].cpu())))
            for children in (1,10):
                arm_start=time.monotonic();block=blocks[children]
                arm=dict(functions=16*children,gates={});r['pending_arm']=arm
                initialization={};source_initializer(block,fit,source,features,initialization,guard)
                write_once(args.directory/f'E{16*children}.initializer.json',initialization)
                initial_hashes=leaf_hashes(block)
                arm['initial_coefficients']=save_tensors(args.directory/f'E{16*children}.initial.pt',coefficients(block))
                arm['initial_error']={s:errors(predict(block,g,routes[children][s],guard),g,s=='development') for s,g in groups.items()}
                stage=f'E{16*children}_joint_fit'
                arm['training']=train(block,fit,routes[children]['fit'],args.directory,children,guard)
                stage=f'E{16*children}_final_diagnostics_export'
                final=coefficients(block);arm['final_coefficients']=save_tensors(args.directory/f'E{16*children}.final_F32.pt',final)
                final_bf16={name:(value.to(torch.bfloat16) if name not in ('parent_norms','child_norms') else value) for name,value in final.items()}
                # Norms belong to the ACTUAL serialized rounded centers.
                final_bf16['parent_norms']=final_bf16['parent_centers'].float().square().sum(-1)
                final_bf16['child_norms']=final_bf16['child_centers'].float().square().sum(-1)
                arm['proposed_BF16_coefficients']=save_tensors(args.directory/f'E{16*children}.proposed_BF16.pt',final_bf16)
                final_hashes=leaf_hashes(block)
                arm['serialized_BF16_leaf_hashes']=final_hashes
                arm['gates']['all_leaf_distinct_BF16']=len(set(final_hashes))==16*children
                arm['gates']['all_leaf_changed_from_own_initializer']=all(a!=b for a,b in zip(initial_hashes,final_hashes))
                arm['encodings']={}
                for encoding in ('F32','BF16_rounded_weights_F32_arithmetic'):
                    if encoding.startswith('BF16'):
                        with torch.no_grad():
                            for param in block.parameters():param.copy_(param.to(torch.bfloat16).float())
                            # Router matrices/centers are part of the proposed BF16
                            # payload too; recompute actual rounded routing/mass.
                            for name in ('projection','parent_centers','child_centers','parent_norms','child_norms'):
                                getattr(block,name).copy_(final_bf16[name].float())
                        arm_routes={s:block.routes(g['tx']) for s,g in groups.items()}
                    else:arm_routes=routes[children]
                    diagnostic={};prediction_files={}
                    for split,g in groups.items():
                        prediction=predict(block,g,arm_routes[split],guard)
                        prediction_path=args.directory/f'E{16*children}.{encoding}.{split}.prediction.npy'
                        with prediction_path.open('xb') as f:np.save(f,prediction,allow_pickle=False)
                        prediction_files[split]=dict(path=str(prediction_path),bytes=prediction_path.stat().st_size,sha256=sha(prediction_path))
                        diagnostic[split]=errors(prediction,g,split=='development',arm_routes[split][0].cpu().numpy())
                    if children==10:
                        prediction=predict(block,dev,arm_routes['development'],guard,True)
                        prediction_path=args.directory/f'E160.{encoding}.cycled_children_development.prediction.npy'
                        with prediction_path.open('xb') as f:np.save(f,prediction,allow_pickle=False)
                        prediction_files['cycled_children_development']=dict(path=str(prediction_path),bytes=prediction_path.stat().st_size,sha256=sha(prediction_path))
                        diagnostic['cycled_children_development']=errors(prediction,dev,True,arm_routes['development'][0].cpu().numpy())
                    diagnostic['prediction_files']=prediction_files
                    arm['encodings'][encoding]=diagnostic
                arm['elapsed_seconds']=time.monotonic()-arm_start;arm_start=None
                write_once(args.directory/f'E{16*children}.result.json',arm);r['arms'].append(arm);r.pop('pending_arm')
                del blocks[children];del block,final,final_bf16;torch.cuda.empty_cache();guard()
            small,large=r['arms']
            gates={}
            for encoding in large['encodings']:
                metric=large['encodings'][encoding]['development'];baseline=small['encodings'][encoding]['development']
                gates[encoding+'_novel_RMS_le_1pct']=metric['unique_x_weighted']['relative_RMS']<=.01
                gates[encoding+'_ALL_category_RMS_le_3pct']=all(v['relative_RMS'] is not None and v['relative_RMS']<=.03 for v in metric['categories'].values())
                gates[encoding+'_E160_SSE_le_90pct_E16']=metric['unique_x_weighted']['squared_ratio']<=.90*baseline['unique_x_weighted']['squared_ratio']
                gates[encoding+'_wrong_child_SSE_ge_110pct']=large['encodings'][encoding]['cycled_children_development']['unique_x_weighted']['squared_ratio']>=1.10*metric['unique_x_weighted']['squared_ratio']
            gates.update(large['gates']);r['eligibility']=gates
            r['decision']='ELIGIBLE_FOR_SEPARATELY_FROZEN_ALL24_CONVERSION' if all(gates.values()) else 'CLOSE_FINITE_JOINT_RECIPE_NO_WHOLE_PROMOTION'
        r['resource_before_final_serialization']=resource();r['ended_compute_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        r['runtime']=dict(torch=torch.__version__,numpy=np.__version__,GPU=torch.cuda.get_device_name(),TF32=False,deterministic=True)
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],resource=resource())),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),fault_stage=stage,resource_snapshot=resource(),ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
