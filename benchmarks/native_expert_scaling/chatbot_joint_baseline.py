"""First E16-only joint fit; reuse closed pilot geometry/choices verbatim."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_joint_pilot import (sha,write_once,load_operands,source_features,source_initializer,
    coefficients,save_tensors,predict,errors,train,leaf_hashes)


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_JOINT_LAYER12_BASELINE_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),gates={},arms=[],
        scope='E16_FEASIBILITY_NOT_E160_UTILITY_WHOLE_CHAT_OR_NATIVE',new_source_full_forwards=0,
        original_E160_exposure_qualified=False)
    torch=None;arm_start=None;stage='binding'
    def resource():
        return dict(elapsed_seconds=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            GPU_peak_allocated=0 if torch is None or not torch.cuda.is_initialized() else torch.cuda.max_memory_allocated(),
            GPU_peak_reserved=0 if torch is None or not torch.cuda.is_initialized() else torch.cuda.max_memory_reserved())
    def guard():
        value=resource();assert value['elapsed_seconds']<=900 and value['OS_peak_snapshot']<=12<<30,'worker budget'
        assert value['GPU_peak_allocated']<=9<<30 and value['GPU_peak_reserved']<=10<<30,'GPU budget'
        assert arm_start is None or time.monotonic()-arm_start<=240,'finite E16 arm240s budget'
        assert not proc.children(recursive=True)
        assert not args.directory.exists() or sum(p.stat().st_size for p in args.directory.iterdir() if p.is_file())<=1<<30
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['schema']=='QWEN_JOINT_LAYER12_BASELINE_BINDING_V1'
        for item in b['inputs']:
            assert str(Path(item['path']).resolve())==item['resolved_path']
            assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256'],item['path']
            guard()
        args.directory.mkdir();r['gates']['frozen_inputs']=True
        def no_subprocess(event,arguments):
            if event=='subprocess.Popen':raise RuntimeError('Baseline forbids subprocess: '+str(arguments[:2]))
        sys.addaudithook(no_subprocess);print(json.dumps(dict(stage='imports')),flush=True)
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
        adoption=json.loads(Path(b['adoption_path']).read_bytes());groups=load_operands(adoption);fit=groups['fit'];dev=groups['development']
        for g in groups.values():g['tx']=torch.from_numpy(g['x']).to(device)
        stage='adopt_saved_geometry_and_parent_control'
        geometry=torch.load(b['saved_geometry'],map_location=device,weights_only=True)
        block=build_block(spec,1,geometry['projection'],geometry['parent_centers'],geometry['parent_centers'][:,None,:].contiguous())
        saved=torch.load(b['saved_E16_routes'],map_location=device,weights_only=True)
        routes={split:(saved[split]['ids'],saved[split]['mass']) for split in groups}
        old=json.loads(Path(b['original_pilot_result']).read_bytes())
        assert old['decision']=='CLOSE_FINITE_PILOT_ON_EXPOSURE_NO_FIT'
        assert json.loads(Path(b['saved_E16_exposure']).read_bytes())['eligible']
        for split,g in groups.items():
            assert routes[split][0].shape==routes[split][1].shape==(len(g['x']),4)
            assert old['data'][split]['rows']==len(g['x']) and old['data'][split]['distinct_x']==len(g['unique_index'])
        r['data']=old['data'];r['gates']['saved_E16_geometry_control_exposure_adopted']=True
        stage='new_source_feature_initializer'
        with safe_open(b['source_weights'],framework='pt',device='cpu') as archive:
            source=[]
            for name,shape in (('gate_proj',(4864,896)),('up_proj',(4864,896)),('down_proj',(896,4864))):
                value=archive.get_tensor(f'model.layers.12.mlp.{name}.weight')
                assert value.shape==shape and value.dtype==torch.bfloat16
                source.append(value.to(device=device,dtype=torch.float32).contiguous())
        features=source_features(fit,source,guard)
        r['source_feature_energy']=save_tensors(args.directory/'source_FIT_feature_energy.pt',dict(
            energy=features[0].cpu(),column_energy=features[1].cpu(),shared_indices=features[2].cpu()))
        arm_start=time.monotonic();arm=dict(functions=16,gates={});r['pending_arm']=arm
        initializer={};source_initializer(block,fit,source,features,initializer,guard)
        write_once(args.directory/'E16.initializer.json',initializer)
        initial_hashes=leaf_hashes(block);arm['initial_coefficients']=save_tensors(args.directory/'E16.initial.pt',coefficients(block))
        arm['initial_error']={split:errors(predict(block,g,routes[split],guard),g,split=='development') for split,g in groups.items()}
        print(json.dumps(dict(stage=stage,initial_FIT_RMS=arm['initial_error']['fit']['unique_x_weighted']['relative_RMS'],
            initial_novel_development_RMS=arm['initial_error']['development']['unique_x_weighted']['relative_RMS'])),flush=True)
        stage='E16_first_joint_fit';arm['training']=train(block,fit,routes['fit'],args.directory,1,guard)
        stage='E16_final_export_and_diagnostics';final=coefficients(block)
        arm['final_coefficients']=save_tensors(args.directory/'E16.final_F32.pt',final)
        bf16={name:(value.to(torch.bfloat16) if name not in ('parent_norms','child_norms') else value) for name,value in final.items()}
        bf16['parent_norms']=bf16['parent_centers'].float().square().sum(-1)
        bf16['child_norms']=bf16['child_centers'].float().square().sum(-1)
        arm['proposed_BF16_coefficients']=save_tensors(args.directory/'E16.proposed_BF16.pt',bf16)
        hashes=leaf_hashes(block);arm['serialized_BF16_leaf_hashes']=hashes
        arm['gates'].update(all_leaf_distinct_BF16=len(set(hashes))==16,
            all_leaf_changed_from_own_initializer=all(a!=b for a,b in zip(initial_hashes,hashes)))
        arm['encodings']={}
        for encoding in ('F32','BF16_rounded_weights_F32_arithmetic'):
            if encoding.startswith('BF16'):
                with torch.no_grad():
                    for value in block.parameters():value.copy_(value.to(torch.bfloat16).float())
                    for name in ('projection','parent_centers','child_centers','parent_norms','child_norms'):getattr(block,name).copy_(bf16[name].float())
                actual_routes={split:block.routes(g['tx']) for split,g in groups.items()}
            else:actual_routes=routes
            diagnostic={};files={}
            for split,g in groups.items():
                prediction=predict(block,g,actual_routes[split],guard)
                path=args.directory/f'E16.{encoding}.{split}.prediction.npy'
                with path.open('xb') as f:np.save(f,prediction,allow_pickle=False)
                files[split]=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
                diagnostic[split]=errors(prediction,g,split=='development',actual_routes[split][0].cpu().numpy())
            diagnostic['prediction_files']=files;arm['encodings'][encoding]=diagnostic
        arm['elapsed_seconds']=time.monotonic()-arm_start;arm_start=None;r['arms'].append(arm);r.pop('pending_arm')
        gates={}
        for encoding,diagnostic in arm['encodings'].items():
            metric=diagnostic['development']
            gates[encoding+'_novel_RMS_le_1pct']=metric['unique_x_weighted']['relative_RMS']<=.01
            gates[encoding+'_ALL_category_RMS_le_3pct']=all(v['relative_RMS'] is not None and v['relative_RMS']<=.03 for v in metric['categories'].values())
        r['fidelity_criteria']=gates;r['decision']='E16_FIDELITY_AVAILABLE_NO_COUNT_OR_WHOLE_ADMISSION' if all(gates.values()) else 'CLOSE_FINITE_E16_FIT_NO_E160_OR_WHOLE_PROMOTION'
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
