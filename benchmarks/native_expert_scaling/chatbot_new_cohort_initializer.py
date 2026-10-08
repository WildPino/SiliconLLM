"""NEW-cohort E16 support and missing source-row initialization, no full forward."""
import argparse
import datetime as dt
import gc
import json
import os
from pathlib import Path
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once

BUFFERS=('projection','parent_centers','child_centers','parent_norms','child_norms')


def load_layer_x(cohort,layer):
    """NEW x-only wire; preserve manifest-order row/case maps per layer."""
    import numpy as np
    groups={s:dict(bits=[],case=[]) for s in ('fit','development')}
    for case_id,case in enumerate(cohort['conversations']):
        with Path(case['x_path']).open('rb') as stream:
            assert struct.unpack('<8s4I',stream.read(24))==(b'QWGX0001',24,896,2,0)
        assert Path(case['x_path']).stat().st_size==24+case['x_rows']*43008
        frames=np.memmap(case['x_path'],dtype='<u2',mode='r',offset=24,shape=(case['x_rows'],24,896))
        g=groups[case['split']];g['bits'].append(np.array(frames[:,layer,:],copy=True,order='C'))
        g['case'].extend([case_id]*case['x_rows']);del frames
    for split,g in groups.items():
        g['x_bits']=np.concatenate(g.pop('bits'));g['case']=np.array(g['case'],dtype='<i8')
        assert g['x_bits'].dtype==np.dtype('<u2') and g['x_bits'].flags.c_contiguous
        assert not ((g['x_bits']&0x7fff)>=0x7f80).any()
        _,g['unique_index'],g['inverse'],g['counts']=np.unique(g['x_bits'],axis=0,
            return_index=True,return_inverse=True,return_counts=True)
        g['x']=(g['x_bits'].astype('<u4')<<16).view('<f4').copy(order='C')
        assert len(g['x'])==(10263 if split=='fit' else 2570)
    fit={v.tobytes() for v in groups['fit']['x_bits']}
    groups['development']['novel']=np.array([v.tobytes() not in fit for v in groups['development']['x_bits']])
    assert groups['development']['novel'].any()
    return groups


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    torch=None;layer_start=None;stage='binding'
    r=dict(schema='QWEN_NEW_COHORT_INITIALIZER_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},layers=[],new_original_BF16_full_forwards=0,new_source_FFN_outputs=0,
        new_student_responses=0,new_optimizer_updates=0,new_endpoint_queries=0)
    def resource():
        return dict(elapsed_seconds=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            GPU_peak_allocated=0 if torch is None or not torch.cuda.is_initialized() else torch.cuda.max_memory_allocated(),
            GPU_peak_reserved=0 if torch is None or not torch.cuda.is_initialized() else torch.cuda.max_memory_reserved())
    def guard():
        value=resource();assert value['elapsed_seconds']<=900 and value['OS_peak_snapshot']<=16<<30,'worker time/OS'
        assert value['GPU_peak_allocated']<=8<<30 and value['GPU_peak_reserved']<=9<<30,'GPU envelope'
        assert layer_start is None or time.monotonic()-layer_start<=90,'per layer90s'
        assert not proc.children(recursive=True),'unexpected descendants'
        assert not args.directory.exists() or sum(p.stat().st_size for p in args.directory.iterdir() if p.is_file())<=1<<30,'output cap'
        journal_path=args.directory/'layers.jsonl'
        assert not journal_path.exists() or journal_path.stat().st_size<=2<<20,'journal cap'
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='new_cohort_initializer'
        for item in b['inputs']:
            path=Path(item['path']);assert str(path.resolve())==item['resolved_path']
            assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],item['path'];guard()
        def no_subprocess(event,arguments):
            if event=='subprocess.Popen':raise RuntimeError('Initializer forbids subprocess')
        sys.addaudithook(no_subprocess);stage='imports'
        import numpy as np
        import torch as torch_module
        torch=torch_module
        from safetensors import safe_open
        from chatbot_compact_geometry import build_block,validate
        from chatbot_joint_pilot import exposure,coefficients,save_tensors,leaf_hashes
        from chatbot_whole_initializer import parent_geometry,select_source_rows
        assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6'
        assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
        assert torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3060'
        torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.manual_seed(261008)
        torch.cuda.manual_seed_all(261008);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.cuda.reset_peak_memory_stats();device=torch.device('cuda:0')
        args.directory.mkdir();spec=validate(json.loads(Path(b['spec_path']).read_bytes()))
        cohort=json.loads(Path(b['cohort_result_path']).read_bytes())
        adoption=json.loads(Path(b['cohort_audit_path']).read_bytes())
        assert adoption['decision']=='NEW_WHOLE_TRANSFER_COHORT_BYTE_ID_QUALIFIED' and adoption['complete_200_case_cohort']
        assert len(cohort['conversations'])==200 and adoption['adopted_x_rows_PER_LAYER']==12833
        r['warm_geometry_inputs']=b['warm_geometry_inputs'];r['warm_initial_inputs']=b['warm_initial_inputs']
        def saved(path,value,cap):
            receipt=save_tensors(path,value);assert receipt['bytes']<=cap,'per-file cap';guard();return receipt
        with (args.directory/'layers.jsonl').open('xb') as journal:
            for layer in range(24):
                layer_start=time.monotonic();stage=f'layer{layer}_NEW_x_and_geometry';guard()
                groups=load_layer_x(cohort,layer)
                for g in groups.values():g['tx']=torch.from_numpy(g['x']).to(device)
                warm_geometry=b['warm_geometry_inputs'].get(str(layer))
                if warm_geometry is None:
                    projection,parents,geometry=parent_geometry(groups['fit'],device);guard()
                    child=parents[:,None,:].contiguous()
                    geometry=dict(geometry,origin='NEW_unique_FIT_geometry')
                else:
                    warm=torch.load(warm_geometry['path'],map_location='cpu',weights_only=True)
                    assert all(warm[n].dtype==torch.float32 and warm[n].is_contiguous() for n in BUFFERS)
                    projection=warm['projection'].to(device);parents=warm['parent_centers'].to(device);child=warm['child_centers'].to(device)
                    geometry=dict(origin='BYTE_reused_warm_geometry',input=warm_geometry)
                block=build_block(spec,1,projection,parents,child)
                if warm_geometry is not None:
                    with torch.no_grad():
                        for name in BUFFERS:getattr(block,name).copy_(warm[name].to(device))
                    del warm
                support,routes,eligible=exposure(block,groups);guard()
                witness={name:getattr(block,name).detach().cpu() for name in BUFFERS}
                for split,g in groups.items():
                    for name in ('unique_index','inverse','counts','case'):
                        witness[split+'_'+name]=torch.from_numpy(np.ascontiguousarray(g[name],dtype='<i8'))
                    witness[split+'_ids']=routes[split][0].cpu();witness[split+'_mass']=routes[split][1].cpu()
                witness['development_novel']=torch.from_numpy(groups['development']['novel'].copy())
                record=dict(layer=layer,geometry=geometry,support=support,support_eligible=eligible,
                    data={split:dict(rows=len(g['x']),distinct_x=len(g['unique_index']),
                        novel_occurrences=int(g['novel'].sum()) if split=='development' else None) for split,g in groups.items()},
                    new_source_feature_evaluations=0,new_source_full_FFN_response_evaluations=0)
                # Preserve NEW control/map witness before a possible source-feature fault.
                record['witness']=saved(args.directory/f'layer{layer:02d}.witness.pt',witness,3<<20)
                r['pending_layer']=record
                warm_initial=b['warm_initial_inputs'].get(str(layer))
                if eligible and warm_initial is not None:
                    record['initial_checkpoint']=warm_initial;record['initial_origin']='BYTE_reused_source_derived_warm_input'
                elif eligible:
                    stage=f'layer{layer}_NEW_FIT_source_row_selection'
                    with safe_open(b['source_weights'],framework='pt',device='cpu') as archive:
                        source=[]
                        for name,shape in (('gate_proj',(4864,896)),('up_proj',(4864,896)),('down_proj',(896,4864))):
                            value=archive.get_tensor(f'model.layers.{layer}.mlp.{name}.weight')
                            assert value.shape==shape and value.dtype==torch.bfloat16
                            source.append(value.to(device=device,dtype=torch.float32).contiguous())
                    selection,evaluations=select_source_rows(block,groups['fit'],source,guard)
                    record['new_source_feature_evaluations']=evaluations
                    record['leaf_BF16_source_copy_hashes']=leaf_hashes(block)
                    assert len(set(record['leaf_BF16_source_copy_hashes']))==16,'private coefficient copies indistinct'
                    record['unique_source_atoms_in_shared_private_union']=len(set(selection['shared_source_indices'].tolist())|
                        set(selection['leaf_source_indices'].reshape(-1).tolist()))
                    record['selection']=saved(args.directory/f'layer{layer:02d}.selection.pt',selection,1<<20)
                    record['initial_checkpoint']=saved(args.directory/f'layer{layer:02d}.initial.pt',coefficients(block),28<<20)
                    record['initial_origin']='NEW_FIT_source_rows';del source,selection,value
                record['elapsed_seconds']=time.monotonic()-layer_start;r['layers'].append(record);r.pop('pending_layer')
                journal.write((json.dumps(record,separators=(',',':'),allow_nan=False)+'\n').encode('utf8'));journal.flush();os.fsync(journal.fileno())
                print(json.dumps(dict(layer=layer,support_eligible=eligible,initialized_for_NEW_cohort=eligible,
                    elapsed_seconds=record['elapsed_seconds'],unique_FIT=record['data']['fit']['distinct_x'],
                    minimum_novel_DEV=min(v['development']['distinct_x'] for v in support.values()))),flush=True)
                guard();layer_start=None
                del block,groups,routes,witness,projection,parents,child;gc.collect();guard()
                if not eligible:break
        complete=len(r['layers'])==24 and all(v['support_eligible'] for v in r['layers'])
        supported=[v['layer'] for v in r['layers'] if v['support_eligible']]
        r['new_cohort_supported_initialized_layers']=supported
        r['source_initialized_layers_available']=sorted({0,12,*supported})
        r['all24_source_initialized_and_NEW_supported']=complete
        r['new_source_feature_evaluations']=sum(v['new_source_feature_evaluations'] for v in r['layers'])
        r['procedure_gates'].update(frozen_adopted_NEW_inputs=True,warm_geometry_0_1_12_and_initial_0_12_BYTE_reused=True,
            exact_NEW_layer_specific_x_partitions_and_saved_control=True,remaining_geometry_and_selection_NEW_FIT_only=True,
            incremental_checkpointed_support_stop=True,no_old_geometry_features_copies_or_source_student_whole_forward=True)
        r.update(decision='NEW_COHORT_ALL24_INITIALIZED_REQUIRE_FIRST_AUDIT' if complete else 'CLOSE_NEW_COHORT_ALL24_ON_SUPPORT',
            resource_before_final_serialization=resource(),ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            scope='NEW operand support/source-row initialization only; no source FFN output or transferred chatbot quality',
            runtime=dict(torch=torch.__version__,numpy=np.__version__,GPU=torch.cuda.get_device_name(),TF32=False,deterministic=True))
        assert len(json.dumps(r,separators=(',',':'),allow_nan=False).encode('utf8'))<=2<<20,'main report cap'
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],supported=supported,resource=resource())),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),fault_stage=stage,resource_snapshot=resource(),ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
