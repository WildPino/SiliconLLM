"""First directional source information; no old source/student response replay."""
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
from chatbot_interaction_launch import sha,write_once


def main(args):
    start=time.monotonic()
    import psutil
    proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_DIRECTIONAL_JACOBIAN_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_source_jacobians=0,new_student_jacobians=0,
        new_source_surrogate_point_values=0,new_student_surrogate_point_values=0,anchors=[])
    streams=[];torch=None
    def guard():
        assert time.monotonic()-start<=180,'worker deadline'
        assert proc.memory_info().peak_wset<=4<<30,'worker OS peak'
        assert not proc.children(recursive=True)
        if torch is not None:
            assert torch.cuda.max_memory_allocated()<=4<<30 and torch.cuda.max_memory_reserved()<=6<<30
        if args.directory.exists():assert sum(v.stat().st_size for v in args.directory.iterdir() if v.is_file())<=512<<20
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['schema']=='QWEN_DIRECTIONAL_BINDING_V1' and b['job']['name']=='jacobian'
        for item in b['inputs']:
            assert str(Path(item['path']).resolve())==item['resolved_path']
            assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256'],item['path'];guard()
        assert sha(b['plan_path'])==b['plan_sha256']
        plan=json.loads(Path(b['plan_path']).read_bytes());assert len(plan['anchors'])==32 and all(plan['procedure_gates'].values())
        r['plan_sha256']=b['plan_sha256'];r['arithmetic']=b['arithmetic']
        import numpy as np
        import torch as torch_module
        torch=torch_module
        from safetensors import safe_open
        assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6' and torch.cuda.is_available()
        torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.reset_peak_memory_stats();device=torch.device('cuda:0')
        r['runtime']=dict(torch=torch.__version__,numpy=np.__version__,GPU=torch.cuda.get_device_name(0),CPU_threads=6,
            TF32=False,deterministic_algorithms=True)
        shapes={'projection':(32,896),'parent_centers':(16,32),'parent_norms':(16,),
            'shared_g':(512,896),'shared_u':(512,896),'shared_b':(896,512),
            'leaf_g':(16,128,896),'leaf_u':(16,128,896),'leaf_b':(16,896,128)}
        saved=torch.load(b['student_checkpoint'],map_location='cpu',weights_only=True)
        for name,shape in shapes.items():
            v=saved[name];assert tuple(v.shape)==shape and v.dtype==torch.float32 and v.is_contiguous() and torch.isfinite(v).all()
        geometry=torch.load(b['saved_geometry'],map_location='cpu',weights_only=True)
        assert all(torch.equal(saved[n],geometry[n]) for n in ('projection','parent_centers','parent_norms'))
        p_cpu=saved['projection'].double();q_cpu=torch.linalg.qr(p_cpu.T,mode='reduced').Q.contiguous()
        orth=float(torch.linalg.matrix_norm(q_cpu.T@q_cpu-torch.eye(32,dtype=torch.float64)))
        row_error=float(torch.linalg.matrix_norm(p_cpu-(p_cpu@q_cpu)@q_cpu.T)/torch.linalg.matrix_norm(p_cpu))
        assert orth<=1e-10 and row_error<=1e-10
        r['QR']=dict(orthogonality_Frobenius=orth,projection_relative_Frobenius=row_error,exact_rank=plan['projection_rank_proof']['exact_real_row_rank'])
        q=q_cpu.to(device);p=p_cpu.to(device);centers=saved['parent_centers'].double().to(device);norms=saved['parent_norms'].double().to(device)
        weights={n:saved[n].double().to(device) for n in ('shared_g','shared_u','shared_b','leaf_g','leaf_u','leaf_b')}
        source=[]
        with safe_open(b['source_weights'],framework='pt',device='cpu') as archive:
            for name,shape in (('gate_proj',(4864,896)),('up_proj',(4864,896)),('down_proj',(896,4864))):
                v=archive.get_tensor('model.layers.12.mlp.'+name+'.weight')
                assert v.dtype==torch.bfloat16 and tuple(v.shape)==shape and v.is_contiguous() and torch.isfinite(v).all()
                source.append(v.float().double().to(device))
        sg,su,sb=source
        del saved,geometry,v
        directions_cpu=torch.zeros((5,896),dtype=torch.float64)
        for i,index in enumerate((0,31,447,895)):directions_cpu[i,index]=1
        e=directions_cpu[3];null=e-q_cpu@(q_cpu.T@e);null_norm=float(torch.linalg.vector_norm(null));assert null_norm>1e-8
        directions_cpu[4]=null/null_norm;directions=directions_cpu.to(device)
        r['directions']=dict(names=['e0','e31','e447','e895','normalized_selector_null_e895'],null_before_normalization_norm=null_norm,
            null_projection_norm=float(torch.linalg.vector_norm(p_cpu@directions_cpu[4])))
        args.directory.mkdir()
        def save_array(name,array):
            array=np.ascontiguousarray(array,dtype='<f8')
            with (args.directory/name).open('xb') as f:
                np.save(f,array,allow_pickle=False);f.flush();os.fsync(f.fileno())
        save_array('selector_rowspace_Q_F64.npy',q_cpu.numpy());save_array('directions_F64.npy',directions_cpu.numpy())
        def output(name,shape):
            f=(args.directory/name).open('xb');streams.append(f)
            np.lib.format.write_array_header_2_0(f,dict(descr='<f8',fortran_order=False,shape=shape));return f
        src_out=output('source_J_F64.npy',(32,896,896));student_out=output('student_J_F64.npy',(32,896,896))
        probe_out=output('probes_F64.npy',(32,5,2,4,896))
        journal=(args.directory/'anchors.jsonl').open('xb');streams.append(journal)
        r['array_order']=dict(J='anchor,output,input; C order; F64',probes='anchor,direction,model(source/student),role(Jv/FD/plus/minus),output; C order; F64')
        def append(f,tensor):
            array=np.ascontiguousarray(tensor.detach().cpu().numpy(),dtype='<f8');raw=array.tobytes(order='C')
            offset=f.tell();f.write(raw);f.flush();os.fsync(f.fileno())
            return dict(path=str(Path(f.name).resolve()),byte_offset=offset,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        def analytic(x,g,u,down,need_value=False):
            gx=g@x;ux=u@x;sigmoid=torch.sigmoid(gx);silu=gx*sigmoid
            derivative=sigmoid+gx*sigmoid*(1-sigmoid)
            j=(down*(ux*derivative).unsqueeze(0))@g+(down*silu.unsqueeze(0))@u
            return j,(down@(silu*ux) if need_value else None)
        def selected(x,ids):
            c=centers[ids];scores=2*(c@(p@x))-norms[ids];return torch.softmax(scores,dim=0)
        def value(x,g,u,down):
            return down@(torch.nn.functional.silu(g@x)*(u@x))
        def student_value(x,ids):
            mass=selected(x,ids)
            out=value(x,weights['shared_g'],weights['shared_u'],weights['shared_b'])
            for i,leaf in enumerate(ids):out=out+mass[i]*value(x,weights['leaf_g'][leaf],weights['leaf_u'][leaf],weights['leaf_b'][leaf])
            return out
        def energy(a):return float(torch.sum(a*a))
        def decomposition(a):
            projected=a@q;parallel=projected@q.T;perp=a-parallel
            total=energy(a);pe=energy(projected);ne=energy(perp);cross=float(torch.sum(parallel*perp))
            closure=abs(total-pe-ne)/max(total,1e-300);cross_ratio=abs(cross)/max(total,1e-300)
            assert closure<=1e-10 and cross_ratio<=1e-10
            return dict(total=total,parallel=pe,null=ne,cross=cross,closure_relative=closure,cross_relative=cross_ratio)
        with torch.no_grad():
            for index,anchor in enumerate(plan['anchors']):
                guard()
                bits=np.frombuffer(bytes.fromhex(anchor['x_BF16_HEX']),dtype='<u2').astype('<u4')<<16
                x=torch.from_numpy(bits.view('<f4').astype('<f8')).to(device)
                assert x.shape==(896,) and torch.isfinite(x).all()
                ids=anchor['selected_parent_ids'];assert len(set(ids))==4 and all(0<=i<16 for i in ids)
                mass=selected(x,ids);assert max(abs(float(a)-b) for a,b in zip(mass,anchor['selected_shadow_mass_F64']))<=1e-10
                source_j,_=analytic(x,sg,su,sb);r['new_source_jacobians']+=1
                student_j,_=analytic(x,weights['shared_g'],weights['shared_u'],weights['shared_b'])
                mass_j=torch.zeros_like(student_j);c=centers[ids];center_mean=mass@c
                grad=2*mass.unsqueeze(1)*((c-center_mean)@p)
                for i,leaf in enumerate(ids):
                    leaf_j,leaf_value=analytic(x,weights['leaf_g'][leaf],weights['leaf_u'][leaf],weights['leaf_b'][leaf],True)
                    student_j=student_j+mass[i]*leaf_j;mass_j=mass_j+torch.outer(leaf_value,grad[i])
                student_j=student_j+mass_j;r['new_student_jacobians']+=1
                assert source_j.shape==student_j.shape==(896,896) and source_j.dtype==student_j.dtype==torch.float64
                assert torch.isfinite(source_j).all() and torch.isfinite(student_j).all()
                error=student_j-source_j;se=decomposition(source_j);ee=decomposition(error);me=decomposition(mass_j)
                stats=dict(source_energy=se,error_energy=ee,mass_derivative_energy=me,
                    relative_RMS={n:math.sqrt(ee[n]/se[n]) for n in ('total','parallel','null')},source_null_energy_share=se['null']/se['total'])
                frames=dict(source=append(src_out,source_j),student=append(student_out,student_j))
                probe_vectors=torch.empty((5,2,4,896),dtype=torch.float64,device=device);probe_records=[]
                other=[i for i in range(16) if i not in ids]
                for direction_index,direction in enumerate(directions):
                    slope=2*(centers@(p@direction))
                    difference=float(torch.max(torch.abs(slope[ids].unsqueeze(1)-slope[other].unsqueeze(0))))
                    margin=anchor['interior_score_margin_F64'];h=min(2**-10,margin/(4*difference)) if difference else 2**-10
                    assert h>=1e-10
                    plus=x+h*direction;minus=x-h*direction
                    cell_margins=[]
                    for point in (plus,minus):
                        scores=2*(centers@(p@point))-norms
                        cell_margin=float(torch.min(scores[ids])-torch.max(scores[other]));assert cell_margin>0
                        cell_margins.append(cell_margin)
                    checks=[]
                    for model_index,j in enumerate((source_j,student_j)):
                        if model_index==0:
                            fplus=value(plus,sg,su,sb);r['new_source_surrogate_point_values']+=1
                            fminus=value(minus,sg,su,sb);r['new_source_surrogate_point_values']+=1
                        else:
                            fplus=student_value(plus,ids);r['new_student_surrogate_point_values']+=1
                            fminus=student_value(minus,ids);r['new_student_surrogate_point_values']+=1
                        analytic_vector=j@direction;fd=(fplus-fminus)/(2*h)
                        discrepancy=float(torch.linalg.vector_norm(analytic_vector-fd));fd_norm=float(torch.linalg.vector_norm(fd))
                        tolerance=1e-10+2e-5*fd_norm
                        probe_vectors[direction_index,model_index]=torch.stack((analytic_vector,fd,fplus,fminus))
                        checks.append(dict(model=('source','student')[model_index],FD_norm=fd_norm,discrepancy_norm=discrepancy,
                            tolerance=tolerance,passed=discrepancy<=tolerance))
                        assert torch.isfinite(probe_vectors[direction_index,model_index]).all() and discrepancy<=tolerance
                    probe_records.append(dict(direction_index=direction_index,h=h,max_score_slope_difference=difference,
                        plus_minus_cell_margins=cell_margins,checks=checks));guard()
                frames['probes']=append(probe_out,probe_vectors)
                record=dict(anchor_index=index,split=anchor['split'],parent=anchor['parent'],case_id=anchor['case_id'],
                    x_SHA256=anchor['x_SHA256'],stats=stats,frames=frames,probes=probe_records)
                journal.write((json.dumps(record,separators=(',',':'),allow_nan=False)+'\n').encode('utf8'));journal.flush();os.fsync(journal.fileno())
                r['anchors'].append(record)
                print(json.dumps(dict(anchor=index,split=anchor['split'],J_relative_RMS=stats['relative_RMS'],elapsed=time.monotonic()-start)),flush=True)
        for f in streams:f.close()
        streams=[];guard()
        assert r['new_source_jacobians']==r['new_student_jacobians']==32
        assert r['new_source_surrogate_point_values']==r['new_student_surrogate_point_values']==320
        pooled={}
        for split in ('fit','development'):
            group=[a['stats'] for a in r['anchors'] if a['split']==split];assert len(group)==16
            se={n:math.fsum(v['source_energy'][n] for v in group) for n in ('total','parallel','null')}
            ee={n:math.fsum(v['error_energy'][n] for v in group) for n in ('total','parallel','null')}
            pooled[split]=dict(source_energy=se,error_energy=ee,relative_RMS={n:math.sqrt(ee[n]/se[n]) for n in se},
                source_null_energy_share=se['null']/se['total'],anchors_total_RMS_above_10_percent=sum(v['relative_RMS']['total']>.1 for v in group),
                weighting='Equal anchors pooled Frobenius energies; not source-row/conversation prevalence')
        r['pooled']=pooled;dev=pooled['development']
        gates=dict(dev_total_J_RMS_above_10_percent=dev['relative_RMS']['total']>.1,
            dev_source_null_energy_at_least_half=dev['source_null_energy_share']>=.5,
            dev_null_J_RMS_above_10_percent=dev['relative_RMS']['null']>.1,
            dev_at_least_eight_total_RMS_above_10_percent=dev['anchors_total_RMS_above_10_percent']>=8)
        r['scientific_diagnostic_gates']=gates
        r['decision']=('DIRECTIONAL_DEFICIT_SUPPORTED_NEW_STRUCTURAL_FIT_JUSTIFIED' if all(gates.values()) else
            'DIRECTIONAL_DEFICIT_UNSUPPORTED_CHANGE_REPRESENTATION')
        r['procedure_gates'].update(exact_bound_inputs=True,original32_anchor_identity=True,source_BF16_student_F32_coefficients_exactly_promoted=True,
            full_F64_C_order_Jacobians=True,QR_roundoff=True,ALL320_directional_pairs_and_cell_margins=True,
            mass_derivative_included=True,no_unchanged_response_or_optimizer_replay=True)
        r.update(elapsed_before_final_serialization=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            GPU_peak_allocated=torch.cuda.max_memory_allocated(),GPU_peak_reserved=torch.cuda.max_memory_reserved(),
            ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],pooled=pooled)),flush=True)
    except BaseException as error:
        for f in streams:f.close()
        r.update(fault=repr(error),elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
