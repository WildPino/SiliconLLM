"""FIRST independent saved-input/index/source-copy/exposure audit; no rerouting.

No source feature/output reevaluation, PCA/clustering replay, field call or fit.
Score ordering is audited on saved scores; their activation computation is not
independently recertified here. Old layer12 is adopted by original receipts only.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once


def independent_input_maps(adoption,layer):
    """Scalar file offsets and byte dictionaries, independent of memmap/unique."""
    import numpy as np
    groups={s:dict(keys=[],cases=[]) for s in ('fit','development')}
    for case_id,case in enumerate(adoption['cases']):
        with Path(case['binary_path']).open('rb') as stream:
            assert struct.unpack('<8s4I',stream.read(24))==(b'QWCAP001',24,896,2,0)
            g=groups[case['split']]
            for row in range(case['captured_rows']):
                stream.seek(24+row*86016+layer*3584);key=stream.read(1792);assert len(key)==1792
                g['keys'].append(key);g['cases'].append(case_id)
    for g in groups.values():
        positions={}
        for i,key in enumerate(g['keys']):positions.setdefault(key,[]).append(i)
        # np.unique(axis=0) orders little-endian WORDS numerically, not raw bytes.
        ordered=sorted(positions,key=lambda key:np.frombuffer(key,dtype='<u2').byteswap().tobytes())
        ranks={key:i for i,key in enumerate(ordered)}
        g['unique_index']=[positions[key][0] for key in ordered];g['counts']=[len(positions[key]) for key in ordered]
        g['inverse']=[ranks[key] for key in g['keys']];g['case']=g['cases']
    fit=set(groups['fit']['keys']);groups['development']['novel']=[key not in fit for key in groups['development']['keys']]
    return groups


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_WHOLE_INITIALIZER_AUDIT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},layers=[],new_original_BF16_full_forwards=0,new_source_features_or_outputs=0,new_routes_or_geometry=0,new_optimizer_updates=0)
    stage='binding'
    def guard():assert time.monotonic()-start<=300 and proc.memory_info().peak_wset<=8<<30 and not proc.children(recursive=True)
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='whole_initializer_audit'
        for item in b['inputs']:
            p=Path(item['path']);assert str(p.resolve())==item['resolved_path'] and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];guard()
        import numpy as np
        import torch
        assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6' and not torch.cuda.is_initialized()
        torch.set_num_threads(1);torch.set_num_interop_threads(1);args.directory.mkdir()
        raw=json.loads(Path(b['initializer_result_path']).read_bytes());adoption=json.loads(Path(b['adoption_path']).read_bytes())
        assert raw['reused_layer12']['initial_path']==b['old_initial_path'] and raw['reused_layer12']['initial_sha256']==sha(b['old_initial_path'])
        assert raw['reused_layer12']['initializer_path']==b['old_indices_path'] and raw['reused_layer12']['initializer_sha256']==sha(b['old_indices_path'])
        assert raw['reused_layer12']['recomputed_geometry_features_routes_source_copies_or_responses'] is False
        assert json.loads(Path(b['old_exposure_path']).read_bytes())['eligible']
        with Path(b['source_weights']).open('rb') as stream:
            hlen=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(hlen))
        for record in raw['layers']:
            layer=record['layer'];stage=f'layer{layer}_saved_input_support';assert layer!=12
            saved=torch.load(record['witness']['path'],map_location='cpu',weights_only=True)
            assert all(v.device.type=='cpu' and v.is_contiguous() for v in saved.values())
            maps=independent_input_maps(adoption,layer);support_eligible=True;max_mass_sum_gap=0.;max_mass_energy_gap=0.
            for split,g in maps.items():
                for name in ('unique_index','inverse','counts','case'):
                    value=saved[split+'_'+name];assert value.dtype==torch.int64 and value.tolist()==g[name],(layer,split,name)
                n=len(g['keys']);assert record['data'][split]['rows']==n and record['data'][split]['distinct_x']==len(g['unique_index'])
                ids=saved[split+'_ids'];mass=saved[split+'_mass'];assert ids.dtype==torch.int64 and mass.dtype==torch.float32
                assert tuple(ids.shape)==tuple(mass.shape)==(n,4) and torch.isfinite(mass).all()
                iid=ids.tolist();mm=mass.tolist();assert all(len(set(row))==4 and all(0<=i<16 for i in row) for row in iid)
                assert all(all(0<v<=1 for v in row) for row in mm)
                max_mass_sum_gap=max(max_mass_sum_gap,max(abs(math.fsum(row)-1) for row in mm))
                eligible=[True]*n if split=='fit' else g['novel']
                if split=='development':
                    assert saved['development_novel'].dtype==torch.bool and saved['development_novel'].tolist()==g['novel']
                    assert record['data'][split]['novel_occurrences']==sum(g['novel'])
                for leaf in range(16):
                    locations=[i for i,row in enumerate(iid) if eligible[i] and leaf in row]
                    distinct=len({g['inverse'][i] for i in locations});conversations=sorted({g['case'][i] for i in locations})
                    expected=record['support'][str(leaf)][split]
                    assert expected['selected_occurrences']==len(locations) and expected['distinct_x']==distinct and expected['conversations']==conversations
                    support_eligible &= distinct>=(8 if split=='fit' else 4) and (split!='fit' or len(conversations)>=2)
                    # Original helper's named mass_squared_sum uses ALL split
                    # occurrences, including nonnovel DEV; certify that scope.
                    exact_sum=math.fsum(value*value for row,values in zip(iid,mm) for i,value in zip(row,values) if i==leaf)
                    gap=abs(exact_sum-expected['mass_squared_sum']);max_mass_energy_gap=max(max_mass_energy_gap,gap)
                    assert gap<=2e-5*max(1,exact_sum)
            assert max_mass_sum_gap<=5e-7 and bool(support_eligible)==record['support_eligible']
            projection=saved['projection'].numpy();parents=saved['parent_centers'].numpy()
            assert projection.shape==(32,896) and parents.shape==(16,32)
            for name in ('projection','parent_centers','child_centers','parent_norms','child_norms'):
                assert saved[name].dtype==torch.float32 and torch.isfinite(saved[name]).all()
            assert np.array_equal(saved['child_centers'].numpy(),parents[:,None,:])
            assert np.array_equal(saved['parent_norms'].numpy(),saved['child_norms'].numpy().reshape(16))
            norms=np.array([math.fsum(float(v)*float(v) for v in row) for row in parents])
            assert max(abs(norms-saved['parent_norms'].numpy()))<=2e-6*max(1,float(max(norms)))
            values=record['geometry']['eigenvalues'];assert len(values)==32 and all(math.isfinite(v) and v>0 for v in values)
            assert all(a>=c for a,c in zip(values,values[1:]))
            gram=projection.astype('float64')@projection.astype('float64').T*math.fsum(values)
            geometry_gap=float(np.abs(gram-np.eye(32)).max());assert geometry_gap<=2e-5
            assert all(row[np.abs(row).argmax()]>0 for row in projection)
            item=dict(layer=layer,support_eligible=bool(support_eligible),maximum_mass_sum_gap=max_mass_sum_gap,
                maximum_F64_vs_saved_F32_mass_squared_sum_gap=max_mass_energy_gap,projection_scaled_orthogonality_max_gap=geometry_gap,
                input_duplicate_and_case_maps_exact=True,saved_control_exposure_exact=True,source_copy_elements_checked=0)
            if record['support_eligible']:
                stage=f'layer{layer}_source_indices_and_exact_copies'
                state=torch.load(record['initial_checkpoint']['path'],map_location='cpu',weights_only=True)
                assert set(state)=={'projection','parent_centers','child_centers','parent_norms','child_norms','shared_g','shared_u','shared_b','leaf_g','leaf_u','leaf_b'}
                assert all(v.dtype==torch.float32 and v.is_contiguous() and torch.isfinite(v).all() for v in state.values())
                for name in ('projection','parent_centers','child_centers','parent_norms','child_norms'):assert torch.equal(state[name],saved[name])
                shared=saved['shared_source_indices'].tolist();private=saved['leaf_source_indices'].tolist();shared_set=set(shared)
                assert saved['shared_source_indices'].dtype==saved['leaf_source_indices'].dtype==torch.int64
                assert len(shared)==len(set(shared))==512 and len(private)==16
                assert all(len(row)==len(set(row))==128 and not set(row)&set(shared) for row in private)
                assert all(0<=i<4864 for i in shared+[i for row in private for i in row])
                scores=saved['global_score'].tolist();assert len(scores)==4864 and all(math.isfinite(v) for v in scores)
                assert shared==sorted(range(4864),key=lambda i:(-scores[i],i))[:512]
                for row,score in zip(private,saved['leaf_scores'].tolist()):
                    assert len(score)==4864 and all(score[i]==-math.inf for i in shared)
                    assert all(math.isfinite(score[i]) for i in range(4864) if i not in shared_set)
                    assert row==sorted(range(4864),key=lambda i:(-score[i],i))[:128]
                hashes=[hashlib.sha256() for _ in range(16)]
                for suffix,key,shape in (('gate_proj','g',(4864,896)),('up_proj','u',(4864,896)),('down_proj','b',(896,4864))):
                    descriptor=header[f'model.layers.{layer}.mlp.{suffix}.weight']
                    assert descriptor['dtype']=='BF16' and tuple(descriptor['shape'])==shape
                    source=np.memmap(b['source_weights'],dtype='<u2',mode='r',offset=8+hlen+descriptor['data_offsets'][0],shape=shape)
                    for slot,indices in [('shared',shared)]+[(leaf,row) for leaf,row in enumerate(private)]:
                        expected=np.ascontiguousarray(source[:,indices] if key=='b' else source[indices,:])
                        actual=state['shared_'+key].numpy() if slot=='shared' else state['leaf_'+key][slot].numpy()
                        assert actual.shape==expected.shape and np.array_equal(actual.view('<u4'),expected.astype('<u4')<<16),(layer,key,slot)
                        item['source_copy_elements_checked']+=actual.size
                        if slot!='shared':hashes[slot].update(expected.tobytes(order='C'))
                    del source
                assert item['source_copy_elements_checked']==6881280
                assert [h.hexdigest() for h in hashes]==record['leaf_BF16_source_copy_hashes'] and len({h.hexdigest() for h in hashes})==16
                assert record['unique_source_atoms_in_shared_private_union']==len(set(shared)|{i for row in private for i in row})
                assert record['new_source_feature_evaluations']==len(maps['fit']['unique_index']) and record['new_source_full_FFN_response_evaluations']==0
                del state
            else:assert 'initial_checkpoint' not in record and record['new_source_feature_evaluations']==0
            r['layers'].append(item);del saved,maps;guard()
            print(json.dumps(dict(layer=layer,independently_verified=True,support_eligible=item['support_eligible'])),flush=True)
        actual_new=[v['layer'] for v in raw['layers']]
        expected_new=[i for i in range(24) if i!=12]
        assert actual_new==expected_new[:len(actual_new)] and len(actual_new)>0
        complete=len(actual_new)==23 and all(v['support_eligible'] for v in raw['layers'])
        if not complete:assert raw['layers'][-1]['support_eligible'] is False and all(v['support_eligible'] for v in raw['layers'][:-1])
        assert complete==raw['all24_source_initialized']
        assert raw['initialized_layers']==sorted([12]+[v['layer'] for v in raw['layers'] if v['support_eligible']])
        r['procedure_gates'].update(first_independent_scalar_offsets_byte_dictionary_maps=True,
            all_saved_support_counts_and_mass_scopes_verified=True,projection_buffer_sanity_without_geometry_replay=True,
            saved_score_stable_order_and_all_new_source_copies_exact=True,old_layer12_only_receipt_adopted=True,
            incremental_support_stop_and_complete_layer_counts_verified=True,no_source_feature_field_route_or_optimizer_replay=True)
        r.update(decision='ALL24_SOURCE_INITIALIZER_INDEPENDENTLY_VERIFIED' if complete else 'PARTIAL_INITIALIZER_SUPPORT_STOP_INDEPENDENTLY_VERIFIED',
            all24_source_initialized=complete,total_new_source_copy_elements_checked=sum(v['source_copy_elements_checked'] for v in r['layers']),
            elapsed_before_final_serialization=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            scope='Independent saved-input/index/byte/support audit; activation score arithmetic and field fidelity NOT independently measured; not whole quality')
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],layers=len(r['layers']))),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),fault_stage=stage,elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset)
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
