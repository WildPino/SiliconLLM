"""FIRST source-derived E16 initialization for 23 new layers; adopt old layer12.

No source/student FFN response, full model forward, fit or optimizer update.
Saved source-row selection and input/control witnesses support a separate audit.
"""
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


def load_layer_x(adoption,layer):
    """Original adopted bytes, x only; each layer owns its exact duplicate map."""
    import numpy as np
    groups={s:dict(bits=[],case=[]) for s in ('fit','development')}
    for case_id,case in enumerate(adoption['cases']):
        with Path(case['binary_path']).open('rb') as stream:
            assert struct.unpack('<8s4I',stream.read(24))==(b'QWCAP001',24,896,2,0)
        frames=np.memmap(case['binary_path'],dtype='<u2',mode='r',offset=24,
            shape=(case['captured_rows'],24,2,896))
        g=groups[case['split']];g['bits'].append(np.array(frames[:,layer,0,:],copy=True,order='C'))
        g['case'].extend([case_id]*case['captured_rows']);del frames
    for split,g in groups.items():
        g['x_bits']=np.concatenate(g.pop('bits'));g['case']=np.array(g['case'],dtype='<i8')
        assert g['x_bits'].dtype==np.dtype('<u2') and g['x_bits'].flags.c_contiguous
        assert not ((g['x_bits']&0x7fff)>=0x7f80).any()
        _,g['unique_index'],g['inverse'],g['counts']=np.unique(g['x_bits'],axis=0,
            return_index=True,return_inverse=True,return_counts=True)
        g['x']=(g['x_bits'].astype('<u4')<<16).view('<f4').copy(order='C')
        assert len(g['x'])==(3845 if split=='fit' else 1864)
    fit_bytes={v.tobytes() for v in groups['fit']['x_bits']}
    groups['development']['novel']=np.array([v.tobytes() not in fit_bytes for v in groups['development']['x_bits']])
    assert groups['development']['novel'].any()
    return groups


def parent_geometry(fit,device):
    import numpy as np
    import torch
    from chatbot_joint_pilot import clusters
    unique=fit['x'][fit['unique_index']].astype('float64');centered=unique-unique.mean(0)
    covariance=torch.from_numpy(np.ascontiguousarray(centered.T@centered/len(unique)))
    values,vectors=torch.linalg.eigh(covariance);values=values[-32:].flip(0)
    vectors=vectors[:,-32:].flip(1).T.contiguous()
    for row in vectors:
        if row[row.abs().argmax()]<0:row.neg_()
    assert torch.isfinite(values).all() and bool((values>0).all())
    projection=(vectors/values.sum().sqrt()).float().contiguous().to(device)
    x=torch.from_numpy(fit['x'][fit['unique_index']].copy(order='C')).to(device)
    query=torch.nn.functional.linear(x,projection).cpu().numpy().astype('float64')
    centers,record=clusters(query,16)
    return projection,torch.from_numpy(centers).to(device),dict(eigenvalues=values.tolist(),parents=record,
        semantics='unique FIT F64 covariance; top32 canonical signs; F32 projection;16 parents;20 Lloyd iterations; no child10')


def select_source_rows(block,fit,source,guard):
    import torch
    from torch.nn import functional as F
    gate,up,down=source;x=fit['tx'][torch.as_tensor(fit['unique_index'],device=gate.device)]
    with torch.no_grad():
        pieces=[]
        for start in range(0,len(x),256):
            part=x[start:start+256]
            pieces.append((F.silu(F.linear(part,gate))*F.linear(part,up)).square());guard()
        energy=torch.cat(pieces);del pieces
        column_energy=down.square().sum(0);global_score=energy.mean(0)*column_energy
        assert torch.isfinite(energy).all() and torch.isfinite(global_score).all()
        shared=torch.argsort(global_score,descending=True,stable=True)[:512].contiguous()
        ids,mass=block.routes(x);available=torch.ones(4864,dtype=torch.bool,device=x.device);available[shared]=False
        leaf_scores=[];leaves=[]
        for leaf in range(16):
            weight=(mass.square()*(ids==leaf)).sum(1);assert float(weight.sum())>0
            score=(energy*weight[:,None]).sum(0)/weight.sum()*column_energy
            assert torch.isfinite(score).all();score[~available]=-float('inf')
            leaf_scores.append(score.cpu());leaves.append(torch.argsort(score,descending=True,stable=True)[:128]);guard()
        private=torch.stack(leaves).contiguous();block.initialize_from_source(gate,up,down,shared,private)
    return dict(shared_source_indices=shared.cpu(),leaf_source_indices=private.cpu(),
        global_score=global_score.cpu(),leaf_scores=torch.stack(leaf_scores),column_energy=column_energy.cpu()),len(x)


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    torch=None;layer_start=None;stage='binding'
    r=dict(schema='QWEN_WHOLE_INITIALIZER_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),procedure_gates={},layers=[],
        new_original_BF16_full_forwards=0,new_source_FFN_outputs=0,new_student_responses=0,new_optimizer_updates=0)
    def resource():
        return dict(elapsed_seconds=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            GPU_peak_allocated=0 if torch is None or not torch.cuda.is_initialized() else torch.cuda.max_memory_allocated(),
            GPU_peak_reserved=0 if torch is None or not torch.cuda.is_initialized() else torch.cuda.max_memory_reserved())
    def guard():
        value=resource();assert value['elapsed_seconds']<=600 and value['OS_peak_snapshot']<=16<<30,'worker time/OS'
        assert value['GPU_peak_allocated']<=8<<30 and value['GPU_peak_reserved']<=9<<30,'GPU envelope'
        assert layer_start is None or time.monotonic()-layer_start<=90,'per NEW layer90s'
        assert not proc.children(recursive=True),'unexpected descendants'
        assert not args.directory.exists() or sum(p.stat().st_size for p in args.directory.iterdir() if p.is_file())<=1<<30,'output cap'
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='whole_initializer'
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
        assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6'
        assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
        assert torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3060'
        torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.manual_seed(261008)
        torch.cuda.manual_seed_all(261008);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.cuda.reset_peak_memory_stats();device=torch.device('cuda:0')
        args.directory.mkdir();spec=validate(json.loads(Path(b['spec_path']).read_bytes()))
        adoption=json.loads(Path(b['adoption_path']).read_bytes())
        old_exposure=json.loads(Path(b['old_exposure_path']).read_bytes());assert old_exposure['eligible']
        r['reused_layer12']=dict(layer=12,initial_path=b['old_initial_path'],initial_sha256=sha(b['old_initial_path']),
            initializer_path=b['old_indices_path'],initializer_sha256=sha(b['old_indices_path']),
            geometry_path=b['old_geometry_path'],routes_path=b['old_routes_path'],exposure_path=b['old_exposure_path'],
            recomputed_geometry_features_routes_source_copies_or_responses=False)
        with (args.directory/'layers.jsonl').open('xb') as journal:
            for layer in range(24):
                if layer==12:continue
                layer_start=time.monotonic();stage=f'layer{layer}_original_x_and_geometry';guard()
                groups=load_layer_x(adoption,layer)
                for g in groups.values():g['tx']=torch.from_numpy(g['x']).to(device)
                projection,parents,geometry=parent_geometry(groups['fit'],device);guard()
                block=build_block(spec,1,projection,parents,parents[:,None,:].contiguous())
                support,routes,eligible=exposure(block,groups);guard()
                witness={name:getattr(block,name).detach().cpu() for name in ('projection','parent_centers','child_centers','parent_norms','child_norms')}
                for split,g in groups.items():
                    for name in ('unique_index','inverse','counts','case'):
                        witness[split+'_'+name]=torch.from_numpy(np.ascontiguousarray(g[name],dtype='<i8'))
                    witness[split+'_ids']=routes[split][0].cpu();witness[split+'_mass']=routes[split][1].cpu()
                witness['development_novel']=torch.from_numpy(groups['development']['novel'].copy())
                record=dict(layer=layer,geometry=geometry,support=support,support_eligible=eligible,
                    data={split:dict(rows=len(g['x']),distinct_x=len(g['unique_index']),
                        novel_occurrences=int(g['novel'].sum()) if split=='development' else None) for split,g in groups.items()},
                    new_source_feature_evaluations=0,new_source_full_FFN_response_evaluations=0)
                if eligible:
                    stage=f'layer{layer}_source_row_selection'
                    with safe_open(b['source_weights'],framework='pt',device='cpu') as archive:
                        source=[]
                        for name,shape in (('gate_proj',(4864,896)),('up_proj',(4864,896)),('down_proj',(896,4864))):
                            value=archive.get_tensor(f'model.layers.{layer}.mlp.{name}.weight')
                            assert value.shape==shape and value.dtype==torch.bfloat16
                            source.append(value.to(device=device,dtype=torch.float32).contiguous())
                    selection,evaluations=select_source_rows(block,groups['fit'],source,guard);witness.update(selection)
                    record['new_source_feature_evaluations']=evaluations
                    record['leaf_BF16_source_copy_hashes']=leaf_hashes(block)
                    assert len(set(record['leaf_BF16_source_copy_hashes']))==16,'private coefficient copies indistinct'
                    record['unique_source_atoms_in_shared_private_union']=len(set(selection['shared_source_indices'].tolist())|
                        set(selection['leaf_source_indices'].reshape(-1).tolist()))
                    record['initial_checkpoint']=save_tensors(args.directory/f'layer{layer:02d}.initial.pt',coefficients(block))
                    del source,selection,value
                record['witness']=save_tensors(args.directory/f'layer{layer:02d}.witness.pt',witness)
                record['elapsed_seconds']=time.monotonic()-layer_start;r['layers'].append(record)
                journal.write((json.dumps(record,separators=(',',':'),allow_nan=False)+'\n').encode('utf8'));journal.flush();os.fsync(journal.fileno())
                print(json.dumps(dict(layer=layer,support_eligible=eligible,initialized=eligible,
                    elapsed_seconds=record['elapsed_seconds'],unique_FIT=record['data']['fit']['distinct_x'])),flush=True)
                guard();layer_start=None
                del block,groups,routes,witness,projection,parents;gc.collect();guard()
                if not eligible:break
        complete=len(r['layers'])==23 and all(v['support_eligible'] for v in r['layers'])
        r['initialized_layers']=sorted([12]+[v['layer'] for v in r['layers'] if v['support_eligible']])
        r['all24_source_initialized']=complete;r['new_source_feature_evaluations']=sum(v['new_source_feature_evaluations'] for v in r['layers'])
        r['procedure_gates'].update(frozen_original_inputs=True,old_layer12_only_adopted=True,
            exact_layer_specific_x_partitions_and_saved_control=True,FIT_only_geometry_and_source_selection=True,
            incremental_checkpointed_support_stop=True,no_source_output_student_fit_or_whole_forward=True)
        r.update(decision='ALL24_SOURCE_INITIALIZED_REQUIRE_FIRST_INDEPENDENT_AUDIT' if complete else 'CLOSE_ALL24_INITIALIZER_ON_LAYER_SUPPORT',
            resource_before_final_serialization=resource(),ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            scope='Source-row initialization and exposure only; approximate mass-attenuated atom sum; not transfer quality or useful capacity',
            runtime=dict(torch=torch.__version__,numpy=np.__version__,GPU=torch.cuda.get_device_name(),TF32=False,deterministic=True))
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],initialized_layers=r['initialized_layers'],resource=resource())),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),fault_stage=stage,resource_snapshot=resource(),ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
