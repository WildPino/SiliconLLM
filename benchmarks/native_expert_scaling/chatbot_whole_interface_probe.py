"""FIRST NEW 3D BF16 adapter/gradient/KL mechanics, no transfer-quality assay."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_WHOLE_INTERFACE_PROBE_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},outputs={},new_original_BF16_full_forwards=0,new_source_core_or_old_field_responses=0,new_optimizer_updates=0)
    def guard():assert time.monotonic()-start<=180 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
    try:
        assert sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='whole_interface'
        for v in b['inputs']:
            p=Path(v['path']);assert str(p.resolve())==v['resolved_path'] and p.stat().st_size==v['bytes'] and sha(p)==v['sha256'];guard()
        import numpy as np
        import torch
        from chatbot_compact_geometry import build_block
        from chatbot_whole_transfer_model import build_mlp_adapter,install_compact_model,output_objective
        assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6'
        assert not torch.cuda.is_initialized() and 'transformers' not in sys.modules
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        args.directory.mkdir();spec=json.loads(Path(b['spec_path']).read_bytes());warm=torch.load(b['saved_block_path'],map_location='cpu',weights_only=True)
        block=build_block(spec,1,warm['projection'],warm['parent_centers'],warm['child_centers'])
        with torch.no_grad():
            for name in ('projection','parent_centers','child_centers','parent_norms','child_norms','shared_g','shared_u','shared_b'):getattr(block,name).copy_(warm[name])
            for name in ('leaf_g','leaf_u','leaf_b'):
                for e,value in enumerate(getattr(block,name)):value.copy_(warm[name][e])
        block.initialized=True
        adapter=build_mlp_adapter(spec,block,12)
        expected_parameters=3*896*(512+16*128)
        parameter_count=sum(v.numel() for v in block.parameters());buffer_count=sum(v.numel() for v in block.buffers())
        assert parameter_count==expected_parameters==6881280
        assert buffer_count==896*32+16*32+16*32+16+16==29728
        # NEW synthetic interface input, not an old donor/candidate observation.
        # Uniform exact dyadic grid avoids RNG and has a separately frozen wire.
        hidden=((torch.arange(4*896,dtype=torch.float32).reshape(1,4,896)%257)-128)/512
        hidden=hidden.to(torch.bfloat16).requires_grad_(True)
        result=adapter(hidden);assert result.shape==hidden.shape and result.dtype==torch.bfloat16
        # Verify gradients cross the BF16/F32/BF16 boundary through a fixed
        # downstream map, as needed to propagate a whole-output objective.
        head=((torch.arange(896*17,dtype=torch.float32).reshape(896,17)%31)-15)/256
        logits=result.float()@head
        logits.retain_grad();teacher=(torch.arange(4*17,dtype=torch.float32).reshape(1,4,17)%11-5)/8
        teacher.requires_grad_(True);mask=torch.tensor([[1.,0.,1.,1.]])
        loss=output_objective(logits,teacher,mask);loss.backward()
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all() and hidden.grad.abs().sum()>0
        assert teacher.grad is None and logits.grad is not None and torch.equal(logits.grad[:,1],torch.zeros_like(logits.grad[:,1]))
        selected=block.routes(hidden.detach().float().reshape(4,896))[0];selected_ids=sorted(set(selected.reshape(-1).tolist()))
        parameter_grads={}
        for name,value in block.named_parameters():
            if name.startswith('leaf_'):
                e=int(name.split('.')[1]);assert (value.grad is not None)==(e in selected_ids)
            else:assert value.grad is not None
            if value.grad is not None:
                assert torch.isfinite(value.grad).all();parameter_grads[name]=float(value.grad.double().square().sum())
        assert sum(parameter_grads.values())>0 and all(value.grad is None for value in block.buffers())
        def save(name,value):
            v=np.ascontiguousarray(value.detach().float().cpu().numpy(),dtype='<f4');p=args.directory/(name+'.npy')
            with p.open('xb') as f:np.save(f,v,allow_pickle=False)
            r['outputs'][name]=dict(path=str(p.resolve()),shape=list(v.shape),dtype=v.dtype.str,C_order=True,bytes=p.stat().st_size,sha256=sha(p))
        for name,value in (('student_logits',logits),('teacher_logits',teacher),('mask',mask),('logit_gradient',logits.grad),('input_gradient',hidden.grad)):save(name,value)
        # Scalar independent output KL/gradient certificate on stored NEW logits.
        import math
        sl=logits.detach().float().numpy();tl=teacher.detach().float().numpy();actual=logits.grad.detach().float().numpy();loss_terms=[];max_gradient_gap=0.
        for token in range(4):
            def distribution(values):
                maximum=max(float(v) for v in values);den=math.fsum(math.exp(float(v)-maximum) for v in values)
                lp=[float(v)-maximum-math.log(den) for v in values];return lp,[math.exp(v) for v in lp]
            sp,ps=distribution(sl[0,token]);tp,pt=distribution(tl[0,token]);weight=float(mask[0,token])/3
            loss_terms.append(weight*math.fsum(p*(a-c) for p,a,c in zip(pt,tp,sp)))
            max_gradient_gap=max(max_gradient_gap,max(abs(weight*(s-t)-float(g)) for s,t,g in zip(ps,pt,actual[0,token])))
        scalar_loss=math.fsum(loss_terms);loss_gap=abs(float(loss.detach())-scalar_loss)
        assert loss_gap<=2e-6 and max_gradient_gap<=2e-7
        rejected=[]
        for name,call in (
            ('uninitialized_block',lambda:build_mlp_adapter(spec,type('Missing',(),{'initialized':False})(),0)),
            ('invalid_layer',lambda:build_mlp_adapter(spec,block,24)),
            ('rank2_input',lambda:adapter(hidden.detach().reshape(4,896))),
            ('F64_input',lambda:adapter(hidden.detach().double())),
            ('empty_mask',lambda:output_objective(logits.detach(),teacher.detach(),torch.zeros_like(mask))),
            ('negative_mask',lambda:output_objective(logits.detach(),teacher.detach(),-mask))):
            try:call()
            except AssertionError:rejected.append(name)
            else:raise AssertionError('Invalid interface accepted: '+name)
        assert len(rejected)==6
        r['mechanics']=dict(parameter_count_per_block=parameter_count,buffer_count_per_block=buffer_count,
            selected_leaf_ids=selected_ids,actual_BF16_input_output=True,input_gradient_nonzero=True,
            selected_parameter_gradient_energy=parameter_grads,teacher_gradient_absent=True,masked_token_gradient_exact_zero=True,
            output_KL=float(loss.detach()),independent_scalar_KL=scalar_loss,scalar_loss_absolute_gap=loss_gap,
            maximum_logit_gradient_absolute_gap=max_gradient_gap,negative_cases_rejected=rejected,
            ALL24_installation_executed=False,full_resident_optimizer_executed=False)
        r['procedure_gates'].update(bound_existing_block_bytes_only=True,FIRST_new_3D_BF16_interface_and_gradient=True,
            actual_block_parameter_and_buffer_counts=True,independent_scalar_KL_gradient_certificate=True,
            SIX_invalid_interface_cases_rejected=True,CPU_only_no_GPU_HF_source_or_old_field_replay=True)
        r.update(decision='WHOLE_INTERFACE_AND_OUTPUT_GRADIENT_MECHANICS_QUALIFIED',elapsed_before_final_serialization=time.monotonic()-start,
            OS_peak_snapshot=proc.memory_info().peak_wset,ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            scope='Synthetic mechanics only, closed block reused; not useful capacity, ALL24 installation, donor fidelity, native chat or rate')
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],mechanics=r['mechanics'])),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset)
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
