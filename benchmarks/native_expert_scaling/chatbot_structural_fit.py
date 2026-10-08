"""First changed value+full-J optimization; reuse original source information."""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_joint_pilot import sha,write_once,load_operands,coefficients,save_tensors,predict,errors,leaf_hashes


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_STRUCTURAL_FIT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_source_jacobians=0,completed_updates=0,
        new_final_student_jacobians=0,new_final_student_surrogate_point_values=0,anchors=[])
    torch=None;fit_start=None;stage='binding';streams=[]
    def resource():
        return dict(elapsed_seconds=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            GPU_peak_allocated=0 if torch is None else torch.cuda.max_memory_allocated(),
            GPU_peak_reserved=0 if torch is None else torch.cuda.max_memory_reserved())
    def guard():
        v=resource();assert v['elapsed_seconds']<=360 and v['OS_peak_snapshot']<=4<<30,'worker resource cap'
        assert v['GPU_peak_allocated']<=4<<30 and v['GPU_peak_reserved']<=6<<30,'GPU cap'
        assert fit_start is None or time.monotonic()-fit_start<=240,'fit240s cap'
        assert not proc.children(recursive=True)
        if args.directory.exists():assert sum(v.stat().st_size for v in args.directory.iterdir() if v.is_file())<=512<<20
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['schema']=='QWEN_DIRECTIONAL_BINDING_V1' and b['job']['name']=='structural_fit'
        for item in b['inputs']:
            assert str(Path(item['path']).resolve())==item['resolved_path'] and Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256'];guard()
        args.directory.mkdir()
        import numpy as np
        import torch as torch_module
        torch=torch_module
        from chatbot_compact_geometry import build_block,validate
        assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6' and torch.cuda.device_count()==1
        assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3060' and os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
        torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.manual_seed(2601007);torch.cuda.manual_seed_all(2601007)
        torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.cuda.reset_peak_memory_stats();device=torch.device('cuda:0')
        r['runtime']=dict(torch=torch.__version__,numpy=np.__version__,GPU=torch.cuda.get_device_name(0),CPU_threads=6,TF32=False,deterministic=True)
        stage='adopt_existing_information'
        plan=json.loads(Path(b['plan_path']).read_bytes());old_j=json.loads(Path(b['prior_jacobian_result']).read_bytes())
        baseline=json.loads(Path(b['baseline_result']).read_bytes())['arms'][0]['encodings']['F32']
        groups=load_operands(json.loads(Path(b['adoption_path']).read_bytes()));fit=groups['fit']
        for g in groups.values():g['tx']=torch.from_numpy(g['x']).to(device)
        saved_routes=torch.load(b['saved_E16_routes'],map_location=device,weights_only=True)
        routes={split:(saved_routes[split]['ids'],saved_routes[split]['mass']) for split in groups}
        warm=torch.load(b['warm_checkpoint'],map_location=device,weights_only=True)
        reference=torch.load(b['reference_checkpoint'],map_location=device,weights_only=True)
        for name in warm:
            assert warm[name].dtype==reference[name].dtype==torch.float32 and warm[name].shape==reference[name].shape and warm[name].is_contiguous() and reference[name].is_contiguous()
            assert torch.isfinite(warm[name]).all() and torch.isfinite(reference[name]).all()
        spec=validate(json.loads(Path(b['spec_path']).read_bytes()))
        block=build_block(spec,1,warm['projection'],warm['parent_centers'],warm['child_centers'])
        with torch.no_grad():
            for name in ('projection','parent_centers','child_centers','parent_norms','child_norms','shared_g','shared_u','shared_b'):
                getattr(block,name).copy_(warm[name]);assert torch.equal(warm[name],reference[name]) if name in ('projection','parent_centers','child_centers','parent_norms','child_norms') else True
            for name in ('leaf_g','leaf_u','leaf_b'):
                for i,param in enumerate(getattr(block,name)):param.copy_(warm[name][i])
        block.initialized=True;initial_hashes=leaf_hashes(block)
        source_j=np.load(b['source_J_path'],mmap_mode='r',allow_pickle=False)
        assert source_j.shape==(32,896,896) and source_j.dtype==np.dtype('<f8') and source_j.flags.c_contiguous
        targets=np.ascontiguousarray(source_j[:16],dtype='<f4');assert np.isfinite(targets).all()
        r['source_FIT_J_F32_targets']=dict(shape=list(targets.shape),bytes=targets.nbytes,sha256=hashlib.sha256(targets.tobytes()).hexdigest(),conversion='One F64->F32 rounding; no source formula')
        tj=torch.from_numpy(targets).to(device);den_j=tj.square().sum((1,2));assert bool((den_j>0).all());del targets
        for index,a in enumerate(plan['anchors']):
            assert a['split']==('fit' if index<16 else 'development') and a['parent']==index%16
            g=groups[a['split']];row=a['split_row_index'];assert g['x_bits'][row].tobytes().hex()==a['x_BF16_HEX']
            assert routes[a['split']][0][row].tolist()==a['selected_parent_ids']
        fit_x=torch.stack([fit['tx'][a['split_row_index']] for a in plan['anchors'][:16]])
        parameter_sets={'shared':(block.shared_g,block.shared_u,block.shared_b)}
        parameter_sets.update({str(i):(block.leaf_g[i],block.leaf_u[i],block.leaf_b[i]) for i in range(16)})
        refs={'shared':tuple(reference[n] for n in ('shared_g','shared_u','shared_b'))}
        refs.update({str(i):tuple(reference[n][i] for n in ('leaf_g','leaf_u','leaf_b')) for i in range(16)})
        ref_den={n:sum(v.square().sum() for v in values) for n,values in refs.items()};assert all(float(v)>0 for v in ref_den.values())
        del reference,warm
        def function_j(x,g,u,down,need_value=False):
            gx=g@x;ux=u@x;sig=torch.sigmoid(gx);silu=gx*sig;der=sig+gx*sig*(1-sig)
            j=(down*(ux*der).unsqueeze(0))@g+(down*silu.unsqueeze(0))@u
            return j,(down@(silu*ux) if need_value else None)
        def student_j(x,ids,p,c,norm,shared,leaves):
            mass=torch.softmax(2*(c[ids]@(p@x))-norm[ids],dim=0)
            grad=2*mass.unsqueeze(1)*((c[ids]-mass@c[ids])@p)
            out,_=function_j(x,*shared)
            for slot,leaf in enumerate(ids):
                j,value=function_j(x,*leaves[leaf],True);out=out+mass[slot]*j+torch.outer(value,grad[slot])
            return out
        def prior(n):return sum((p-v).square().sum() for p,v in zip(parameter_sets[n],refs[n]))/ref_den[n]
        shared=parameter_sets['shared'];leaves=[parameter_sets[str(i)] for i in range(16)]
        optimizer=torch.optim.Adam(block.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
        generator=torch.Generator(device='cpu').manual_seed(2601007)
        weight=torch.as_tensor(fit['weights'],dtype=torch.float32,device=device);target=torch.from_numpy(fit['y']).to(device)
        mean_energy=(weight*target.square().sum(1)).mean();assert float(mean_energy)>0
        r['value_normalization_FULL_FIT_mean_energy_F32']=float(mean_energy);r['curve']=[];visits=[0]*16
        fit_start=time.monotonic();stage='first_changed_joint_fit'
        with (args.directory/'updates.jsonl').open('xb') as journal:
            for epoch in range(24):
                order=torch.randperm(len(target),generator=generator).to(device);value_sum=0.;j_sum=0.;count=0
                for offset in range(0,len(target),128):
                    guard();indices=order[offset:offset+128];ids,mass=(v[indices] for v in routes['fit'])
                    ai=r['completed_updates']%16;anchor=plan['anchors'][ai];visits[ai]+=1
                    optimizer.zero_grad(set_to_none=True)
                    prediction=block.evaluate_selected(fit['tx'][indices].contiguous(),ids,mass)
                    lv=(weight[indices]*(prediction-target[indices]).square().sum(1)).mean()/mean_energy
                    j=student_j(fit_x[ai],anchor['selected_parent_ids'],block.projection,block.parent_centers,block.parent_norms,shared,leaves)
                    lj=(j-tj[ai]).square().sum()/den_j[ai]
                    active=sorted(set(torch.unique(ids).tolist())|set(anchor['selected_parent_ids']))
                    la=prior('shared')+sum(prior(str(i)) for i in active)/len(active)
                    loss=lv+lj+.001*la;assert bool(torch.isfinite(loss))
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(block.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                    r['completed_updates']+=1;value=float(lv.detach());jvalue=float(lj.detach())
                    value_sum+=value*len(indices);j_sum+=jvalue;count+=1
                    row=dict(epoch=epoch+1,update=r['completed_updates'],rows=len(indices),anchor_index=ai,value_loss=value,J_loss=jvalue,
                        coefficient_prior=float(la.detach()),preclip_gradient_norm=float(norm),elapsed_fit_seconds=time.monotonic()-fit_start)
                    journal.write((json.dumps(row,separators=(',',':'),allow_nan=False)+'\n').encode('utf8'));journal.flush();guard()
                os.fsync(journal.fileno())
                curve=dict(epoch=epoch+1,updates=r['completed_updates'],mean_response_minibatch_loss=value_sum/len(target),mean_J_minibatch_loss=j_sum/count)
                r['curve'].append(curve);print(json.dumps(curve),flush=True)
                if epoch+1 in (6,12,18,24):
                    name='E16.final_F32.pt' if epoch==23 else f'E16.epoch{epoch+1}_F32.pt'
                    receipt=save_tensors(args.directory/name,coefficients(block));r.setdefault('checkpoints',[]).append(receipt)
        r['fit_elapsed_seconds']=time.monotonic()-fit_start;fit_start=None
        assert r['completed_updates']==744 and visits==[47]*8+[46]*8;r['anchor_visits']=visits
        r['normalized_displacement_from_original_reference']={n:float(prior(n).detach()) for n in refs}
        hashes=leaf_hashes(block);r['leaf_BF16_hashes']=hashes
        updated=all(a!=b for a,b in zip(initial_hashes,hashes));distinct=len(set(hashes))==16
        assert all(torch.isfinite(v).all() for v in block.parameters())
        r['function_gates']=dict(all_functions_distinct_BF16=distinct,all_functions_updated_BF16=updated)
        del optimizer,refs,ref_den,parameter_sets,target,weight,tj,j,prediction,loss,lv,lj,la
        stage='final_NEW_response'
        r['responses']={}
        for split,g in groups.items():
            prediction=predict(block,g,routes[split],guard);path=args.directory/f'E16.F32.{split}.prediction.npy'
            with path.open('xb') as f:np.save(f,prediction,allow_pickle=False);f.flush();os.fsync(f.fileno())
            r['responses'][split]=errors(prediction,g,split=='development',routes[split][0].cpu().numpy())
            r['responses'][split]['prediction_file']=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
        del prediction
        stage='final_NEW_student_J_and_probes'
        q=torch.from_numpy(np.array(np.load(b['Q_path'],allow_pickle=False),copy=True)).to(device)
        directions=torch.from_numpy(np.array(np.load(b['directions_path'],allow_pickle=False),copy=True)).to(device)
        p=block.projection.double();c=block.parent_centers.double();norms=block.parent_norms.double()
        shared64=tuple(v.double() for v in shared);leaves64=[tuple(v.double() for v in leaf) for leaf in leaves]
        def value(x,ids):
            mass=torch.softmax(2*(c[ids]@(p@x))-norms[ids],dim=0)
            def f(weights):g,u,down=weights;return down@(torch.nn.functional.silu(g@x)*(u@x))
            out=f(shared64)
            for slot,leaf in enumerate(ids):out=out+mass[slot]*f(leaves64[leaf])
            return out
        def output(name,shape):
            f=(args.directory/name).open('xb');streams.append(f)
            np.lib.format.write_array_header_2_0(f,dict(descr='<f8',fortran_order=False,shape=shape));return f
        jout=output('student_J_F64.npy',(32,896,896));pout=output('student_probes_F64.npy',(32,5,4,896))
        aj=(args.directory/'anchors.jsonl').open('xb');streams.append(aj)
        def append(f,tensor):
            raw=np.ascontiguousarray(tensor.cpu().numpy(),dtype='<f8').tobytes();offset=f.tell()
            f.write(raw);f.flush();os.fsync(f.fileno());return dict(path=str(Path(f.name).resolve()),byte_offset=offset,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        def decompose(a):
            aq=a@q;perp=a-aq@q.T;total=float(a.square().sum());parallel=float(aq.square().sum());null=float(perp.square().sum())
            closure=abs(total-parallel-null)/max(total,1e-300);assert closure<=1e-10
            return dict(total=total,parallel=parallel,null=null,closure_relative=closure)
        with torch.no_grad():
            for index,a in enumerate(plan['anchors']):
                guard();x=groups[a['split']]['tx'][a['split_row_index']].double();ids=a['selected_parent_ids']
                j=student_j(x,ids,p,c,norms,shared64,leaves64);assert torch.isfinite(j).all();r['new_final_student_jacobians']+=1
                sj=torch.from_numpy(np.array(source_j[index],copy=True)).to(device);se=decompose(sj);ee=decompose(j-sj)
                vectors=torch.empty((5,4,896),dtype=torch.float64,device=device);checks=[]
                for di,direction in enumerate(directions):
                    h=old_j['anchors'][index]['probes'][di]['h'];assert h>=1e-10
                    plus=value(x+h*direction,ids);r['new_final_student_surrogate_point_values']+=1
                    minus=value(x-h*direction,ids);r['new_final_student_surrogate_point_values']+=1
                    jv=j@direction;fd=(plus-minus)/(2*h);gap=float(torch.linalg.vector_norm(jv-fd));fdnorm=float(torch.linalg.vector_norm(fd));tolerance=1e-10+2e-5*fdnorm
                    vectors[di]=torch.stack((jv,fd,plus,minus));assert torch.isfinite(vectors[di]).all() and gap<=tolerance
                    checks.append(dict(direction_index=di,h=h,FD_norm=fdnorm,discrepancy_norm=gap,tolerance=tolerance,passed=True))
                record=dict(anchor_index=index,split=a['split'],x_SHA256=a['x_SHA256'],source_energy=se,error_energy=ee,
                    relative_RMS={n:math.sqrt(ee[n]/se[n]) for n in ('total','parallel','null')},
                    frames=dict(student=append(jout,j),probes=append(pout,vectors)),probes=checks)
                aj.write((json.dumps(record,separators=(',',':'),allow_nan=False)+'\n').encode('utf8'));aj.flush();os.fsync(aj.fileno());r['anchors'].append(record)
        for f in streams:f.close()
        streams=[];assert r['new_final_student_jacobians']==32 and r['new_final_student_surrogate_point_values']==320
        r['pooled']={}
        for split in groups:
            records=[a for a in r['anchors'] if a['split']==split];assert len(records)==16
            se={n:math.fsum(v['source_energy'][n] for v in records) for n in ('total','parallel','null')}
            ee={n:math.fsum(v['error_energy'][n] for v in records) for n in se}
            r['pooled'][split]=dict(source_energy=se,error_energy=ee,relative_RMS={n:math.sqrt(ee[n]/se[n]) for n in se})
        fit_metric=r['responses']['fit']['unique_x_weighted'];dev=r['responses']['development']
        mechanism=dict(novel_response_improves_10pct=dev['unique_x_weighted']['relative_RMS']<=.9*baseline['development']['unique_x_weighted']['relative_RMS'],
            FIT_response_no_more_than_5pct_worse=fit_metric['relative_RMS']<=1.05*baseline['fit']['unique_x_weighted']['relative_RMS'],
            ALL_categories_no_more_than_5pct_worse=all(v['relative_RMS']<=1.05*baseline['development']['categories'][n]['relative_RMS'] for n,v in dev['categories'].items()))
        mechanism.update({split+'_full_J_improves_10pct':r['pooled'][split]['relative_RMS']['total']<=.9*old_j['pooled'][split]['relative_RMS']['total'] for split in groups})
        absolute=dict(novel_response_RMS_le_1pct=dev['unique_x_weighted']['relative_RMS']<=.01,ALL_category_RMS_le_3pct=all(v['relative_RMS']<=.03 for v in dev['categories'].values()))
        r.update(mechanism_gates=mechanism,absolute_fidelity_gates=absolute,
            decision='LOCAL_F32_STRUCTURAL_FIDELITY_AVAILABLE_ENCODING_NOT_ADMITTED' if all(mechanism.values()) and all(absolute.values()) and all(r['function_gates'].values()) else 'CLOSE_ONE_STRUCTURAL_FIT_NO_WHOLE_PROMOTION',
            scope='Original calibration component development; no fresh whole quality/native arithmetic/count utility/rate')
        r['procedure_gates'].update(bound_inputs=True,original_source_information_adopted=True,FIT_only_J_targets=True,
            exact744_updates_and_visits=True,frozen_router_and_active_geometry=True,all_parameters_finite=True,
            ALL_final_student_derivative_checks=True,full_C_F32_predictions_and_F64_J=True)
        assert all(r['procedure_gates'].values());guard()
        r.update(resource_before_final_serialization=resource(),ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],responses={s:v['unique_x_weighted']['relative_RMS'] for s,v in r['responses'].items()},pooled=r['pooled'],resource=resource())),flush=True)
    except BaseException as error:
        for f in streams:f.close()
        r.update(fault=repr(error),fault_stage=stage,resource_snapshot=resource(),ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
