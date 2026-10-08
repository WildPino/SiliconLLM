"""FIRST independent balanced construction/query envelope/support/source bits.

No source feature/field/actual routing reexecution. Construction transcript is
independently checked; warm G/U/D retain prior byte references, new buffers win.
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

BUFFERS=('projection','parent_centers','child_centers','parent_norms','child_norms')


def independent_input_maps(cohort,layer):
    """Scalar NEW x-only offsets and byte dictionaries, no memmap/np.unique."""
    import numpy as np
    groups={s:dict(keys=[],cases=[]) for s in ('fit','development')}
    for case_id,case in enumerate(cohort['conversations']):
        with Path(case['x_path']).open('rb') as stream:
            assert struct.unpack('<8s4I',stream.read(24))==(b'QWGX0001',24,896,2,0)
            g=groups[case['split']]
            for row in range(case['x_rows']):
                stream.seek(24+row*43008+layer*1792);key=stream.read(1792);assert len(key)==1792
                g['keys'].append(key);g['cases'].append(case_id)
    for split,g in groups.items():
        assert len(g['keys'])==(10263 if split=='fit' else 2570)
        positions={}
        for i,key in enumerate(g['keys']):positions.setdefault(key,[]).append(i)
        # Sort little-endian words numerically, matching producer np.unique(axis=0).
        ordered=sorted(positions,key=lambda key:np.frombuffer(key,dtype='<u2').byteswap().tobytes())
        ranks={key:i for i,key in enumerate(ordered)}
        g['unique_index']=[positions[key][0] for key in ordered];g['counts']=[len(positions[key]) for key in ordered]
        g['inverse']=[ranks[key] for key in g['keys']];g['case']=g['cases']
    fit=set(groups['fit']['keys']);groups['development']['novel']=[key not in fit for key in groups['development']['keys']]
    return groups


def independent_distance(query,centers):
    """Coordinate-recursive F64 tree, independent of producer's packed tree."""
    def part(start,width):
        if width==1:
            delta=query[:,start,None]-centers[None,:,start]
            return delta*delta
        half=width//2
        return part(start,half)+part(start+half,half)
    return part(0,32)


def verify_construction(record,saved,maps,b,guard):
    import numpy as np
    import torch
    trace=torch.load(record['construction']['path'],map_location='cpu',weights_only=True)
    assert set(trace)=={'query_F32','quotas_int64','assignments_int64','centers_F64','assignment_costs_F64'}
    assert all(v.device.type=='cpu' and v.is_contiguous() for v in trace.values())
    assert trace['query_F32'].dtype==torch.float32 and trace['centers_F64'].dtype==trace['assignment_costs_F64'].dtype==torch.float64
    assert trace['quotas_int64'].dtype==trace['assignments_int64'].dtype==torch.int64
    g=maps['fit'];n=len(g['unique_index']);query=trace['query_F32'].numpy().astype(np.float64)
    assert query.shape==(n,32) and np.isfinite(query).all()
    quota=np.full(16,n//16,dtype=np.int64);quota[:n%16]+=1
    assert np.array_equal(quota,trace['quotas_int64'].numpy())
    assignments=trace['assignments_int64'].numpy();centers=trace['centers_F64'].numpy();costs=trace['assignment_costs_F64'].numpy()
    assert assignments.shape==(20,n) and centers.shape==(21,16,32) and costs.shape==(20,)
    assert np.isfinite(centers).all() and np.isfinite(costs).all() and quota.min()>0
    config=record['geometry']['parents']
    assert config['rows']==n and config['parents']==16 and config['iterations']==20
    assert config['quota_min']==int(quota.min()) and config['quota_max']==int(quota.max())
    assert config['constrained_optimality_or_monotonicity_claim'] is False and config['balanced_assignments_are_NOT_actual_inference_support'] is True
    bits=np.frombuffer(b''.join(g['keys'][i] for i in g['unique_index']),dtype='<u2').reshape(n,896)
    x=(bits.astype('<u4')<<16).view('<f4').astype(np.float64);projection=saved['projection'].numpy().astype(np.float64)
    minimum=2.**-126
    assert np.all((x==0)|(np.abs(x)>=minimum)) and np.all((projection==0)|(np.abs(projection)>=minimum))
    reference=x@projection.T;absolute=np.abs(x)@np.abs(projection).T
    # Conservative separate-product/add dot envelope; covers F64 reference and
    # possible product/output flush of F32 subnormals. TF32 disabled by producer.
    def gamma(unit):return np.nextafter(1792*unit/(1-1792*unit),np.inf)
    g32=gamma(2.**-24);g64=gamma(2.**-53)
    upper=np.nextafter(absolute/(1-g64),np.inf)
    envelope=np.nextafter((g32+2*g64)*upper+1792*minimum,np.inf)
    error=np.abs(query-reference);assert np.isfinite(reference).all() and np.all(error<=envelope)
    warm=b['warm_projection_and_seed_inputs'].get(str(record['layer']))
    if warm is not None:
        original=torch.load(warm['path'],map_location='cpu',weights_only=True)
        assert np.array_equal(saved['projection'].numpy().view('<u4'),original['projection'].numpy().view('<u4'))
        assert np.array_equal(centers[0].view('<u8'),original['parent_centers'].numpy().astype(np.float64).view('<u8'))
        assert record['geometry']['projection_origin']=='BYTE_warm_projection' and record['geometry']['projection_input']==warm
        assert record['geometry']['eigenvalues'] is None and record['geometry']['parents']['farthest_seed_indices'] is None
        del original
    else:
        assert record['geometry']['projection_origin']=='NEW_unique_FIT_covariance' and record['geometry']['projection_input'] is None
        mean=np.zeros((1,32),dtype=np.float64)
        for row in query:mean[0]+=row
        mean/=n
        distance=independent_distance(query,mean)[:,0];first=max(range(n),key=lambda i:(float(distance[i]),-i))
        chosen=[first];nearest=independent_distance(query,query[first:first+1])[:,0]
        for _ in range(15):
            index=max(range(n),key=lambda i:(float(nearest[i]),-i));assert index not in chosen
            chosen.append(index);nearest=np.minimum(nearest,independent_distance(query,query[index:index+1])[:,0])
        assert chosen==record['geometry']['parents']['farthest_seed_indices']
        assert np.array_equal(centers[0].view('<u8'),query[chosen].view('<u8'))
    for step in range(20):
        distance=independent_distance(query,centers[step]);flat=distance.reshape(-1)
        edges=sorted(range(n*16),key=lambda i:(float(flat[i]),i))
        expected=np.full(n,-1,dtype=np.int64);remaining=quota.tolist();done=0
        for edge in edges:
            row,parent=divmod(edge,16)
            if expected[row]<0 and remaining[parent]>0:
                expected[row]=parent;remaining[parent]-=1;done+=1
                if done==n:break
        assert done==n and remaining==[0]*16 and np.array_equal(expected,assignments[step]),(record['layer'],step,'assignments')
        actual=np.zeros((16,32),dtype=np.float64);cost=0.
        for row,parent in enumerate(expected):
            actual[parent]+=query[row];cost+=float(distance[row,parent])
        actual/=quota[:,None]
        assert np.array_equal(actual.view('<u8'),centers[step+1].view('<u8')),(record['layer'],step,'centroid means')
        assert np.float64(cost).view('<u8')==np.float64(costs[step]).view('<u8'),(record['layer'],step,'cost')
        guard()
    assert record['geometry']['origin']=='NEW_balanced_FIT_keys'
    assert np.array_equal(saved['parent_centers'].numpy().view('<u4'),centers[-1].astype(np.float32).view('<u4'))
    return dict(steps=20,quota_min=int(quota.min()),quota_max=int(quota.max()),all_greedy_assignments_means_costs_F64_bits_exact=True,
        maximum_query_F32_vs_F64_gap=float(error.max()),maximum_declared_dot_envelope=float(envelope.max()),
        warm_projection_and_starting_labels_BYTES_exact=warm is not None,query_scope='declared dot-rounding envelope, not bitwise F32 GPU dot reexecution')


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_BALANCED_INITIALIZER_AUDIT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},layers=[],new_original_BF16_full_forwards=0,new_source_features_or_outputs=0,
        new_route_or_geometry_optimization_runs=0,new_optimizer_updates=0,new_endpoint_queries=0)
    stage='binding'
    def guard():assert time.monotonic()-start<=600 and proc.memory_info().peak_wset<=8<<30 and not proc.children(recursive=True)
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='balanced_initializer_audit'
        for item in b['inputs']:
            p=Path(item['path']);assert str(p.resolve())==item['resolved_path'] and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];guard()
        import numpy as np
        import torch
        assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6' and not torch.cuda.is_initialized()
        torch.set_num_threads(1);torch.set_num_interop_threads(1);args.directory.mkdir()
        raw=json.loads(Path(b['initializer_result_path']).read_bytes());cohort=json.loads(Path(b['cohort_result_path']).read_bytes())
        assert raw['warm_projection_and_seed_inputs']==b['warm_projection_and_seed_inputs'] and raw['warm_initial_inputs']==b['warm_initial_inputs']
        assert len(cohort['conversations'])==200
        with Path(b['source_weights']).open('rb') as stream:
            hlen=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(hlen))
        journal=Path(raw['layers'][0]['witness']['path']).parent/'layers.jsonl'
        assert [json.loads(line) for line in journal.read_text(encoding='utf8').splitlines()]==raw['layers']
        for record in raw['layers']:
            layer=record['layer'];stage=f'layer{layer}_NEW_saved_input_support'
            saved=torch.load(record['witness']['path'],map_location='cpu',weights_only=True)
            assert all(v.device.type=='cpu' and v.is_contiguous() for v in saved.values())
            maps=independent_input_maps(cohort,layer);support_eligible=True;max_mass_sum_gap=0.;max_mass_energy_gap=0.
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
                    # Named mass_squared_sum is over ALL occurrences, including nonnovel DEV.
                    exact_sum=math.fsum(value*value for row,values in zip(iid,mm) for i,value in zip(row,values) if i==leaf)
                    gap=abs(exact_sum-expected['mass_squared_sum']);max_mass_energy_gap=max(max_mass_energy_gap,gap)
                    assert gap<=2e-5*max(1,exact_sum)
            assert max_mass_sum_gap<=5e-7 and bool(support_eligible)==record['support_eligible']
            for name in BUFFERS:assert saved[name].dtype==torch.float32 and torch.isfinite(saved[name]).all()
            assert tuple(saved['projection'].shape)==(32,896) and tuple(saved['parent_centers'].shape)==(16,32)
            assert tuple(saved['child_centers'].shape)==(16,1,32)
            assert tuple(saved['parent_norms'].shape)==(16,) and tuple(saved['child_norms'].shape)==(16,1)
            stage=f'layer{layer}_FIRST_balanced_construction'
            construction=verify_construction(record,saved,maps,b,guard);geometry_gap=None
            projection=saved['projection'].numpy();parents=saved['parent_centers'].numpy()
            assert np.array_equal(saved['child_centers'].numpy().view('<u4'),parents[:,None,:].view('<u4'))
            assert np.array_equal(saved['parent_norms'].numpy().view('<u4'),saved['child_norms'].numpy().reshape(16).view('<u4'))
            norms=np.array([math.fsum(float(v)*float(v) for v in row) for row in parents])
            assert max(abs(norms-saved['parent_norms'].numpy()))<=2e-6*max(1,float(max(norms)))
            if record['geometry']['projection_input'] is None:
                values=record['geometry']['eigenvalues'];assert len(values)==32 and all(math.isfinite(v) and v>0 for v in values)
                assert all(a>=c for a,c in zip(values,values[1:]))
                gram=projection.astype('float64')@projection.astype('float64').T*math.fsum(values)
                geometry_gap=float(np.abs(gram-np.eye(32)).max());assert geometry_gap<=2e-5
                assert all(row[np.abs(row).argmax()]>0 for row in projection)
            item=dict(layer=layer,support_eligible=bool(support_eligible),maximum_mass_sum_gap=max_mass_sum_gap,
                maximum_F64_vs_saved_F32_mass_squared_sum_gap=max_mass_energy_gap,NEW_projection_scaled_orthogonality_max_gap=geometry_gap,
                input_duplicate_and_case_maps_exact=True,saved_control_exposure_exact=True,source_copy_elements_checked=0,
                balanced_construction=construction,warm_GUD_reference_only=False)
            if record['support_eligible']:
                initial=b['warm_initial_inputs'].get(str(layer))
                if initial is not None:
                    assert record['initial_checkpoint']==initial and record['initial_origin']=='BYTE_warm_GUD_only_with_NEW_routing'
                    assert record['old_routing_buffers_MUST_NOT_override_NEW_geometry'] is True
                    assert record['new_source_feature_evaluations']==0 and 'selection' not in record
                    item['warm_GUD_reference_only']=True
                else:
                    stage=f'layer{layer}_NEW_source_indices_and_copies'
                    selection=torch.load(record['selection']['path'],map_location='cpu',weights_only=True)
                    state=torch.load(record['initial_checkpoint']['path'],map_location='cpu',weights_only=True)
                    assert record['initial_origin']=='NEW_FIT_source_rows'
                    assert set(state)=={*BUFFERS,'shared_g','shared_u','shared_b','leaf_g','leaf_u','leaf_b'}
                    assert all(v.dtype==torch.float32 and v.is_contiguous() and torch.isfinite(v).all() for v in state.values())
                    for name in BUFFERS:assert np.array_equal(state[name].numpy().view('<u4'),saved[name].numpy().view('<u4'))
                    shared=selection['shared_source_indices'].tolist();private=selection['leaf_source_indices'].tolist();shared_set=set(shared)
                    assert selection['shared_source_indices'].dtype==selection['leaf_source_indices'].dtype==torch.int64
                    assert len(shared)==len(set(shared))==512 and len(private)==16
                    assert all(len(row)==len(set(row))==128 and not set(row)&shared_set for row in private)
                    assert all(0<=i<4864 for i in shared+[i for row in private for i in row])
                    scores=selection['global_score'].tolist();assert len(scores)==4864 and all(math.isfinite(v) for v in scores)
                    assert shared==sorted(range(4864),key=lambda i:(-scores[i],i))[:512]
                    for row,score in zip(private,selection['leaf_scores'].tolist()):
                        assert len(score)==4864 and all(score[i]==-math.inf for i in shared)
                        assert all(math.isfinite(score[i]) for i in range(4864) if i not in shared_set)
                        assert row==sorted((i for i in range(4864) if i not in shared_set),key=lambda i:(-score[i],i))[:128]
                    energy=selection['column_energy'].tolist();assert len(energy)==4864 and all(math.isfinite(v) and v>=0 for v in energy)
                    hashes=[hashlib.sha256() for _ in range(16)]
                    for name,key,shape in (('gate_proj','g',(4864,896)),('up_proj','u',(4864,896)),('down_proj','b',(896,4864))):
                        descriptor=header[f'model.layers.{layer}.mlp.{name}.weight'];assert descriptor['dtype']=='BF16' and descriptor['shape']==list(shape)
                        source=np.memmap(b['source_weights'],dtype='<u2',mode='r',offset=8+hlen+descriptor['data_offsets'][0],shape=shape)
                        for slot,indices in [('shared',shared)]+[(leaf,row) for leaf,row in enumerate(private)]:
                            expected=np.ascontiguousarray(source[:,indices] if key=='b' else source[indices,:])
                            actual=state['shared_'+key].numpy() if slot=='shared' else state['leaf_'+key][slot].numpy()
                            assert actual.shape==expected.shape and np.array_equal(actual.view('<u4'),expected.astype('<u4')<<16),(layer,key,slot)
                            item['source_copy_elements_checked']+=actual.size
                            if slot!='shared':hashes[slot].update(expected.tobytes(order='C'))
                        del source;guard()
                    assert item['source_copy_elements_checked']==6881280
                    assert [h.hexdigest() for h in hashes]==record['leaf_BF16_source_copy_hashes'] and len({h.hexdigest() for h in hashes})==16
                    assert record['unique_source_atoms_in_shared_private_union']==len(shared_set|{i for row in private for i in row})
                    assert record['new_source_feature_evaluations']==len(maps['fit']['unique_index'])
                    del state,selection
            else:assert 'initial_checkpoint' not in record and 'selection' not in record and record['new_source_feature_evaluations']==0
            assert record['new_source_full_FFN_response_evaluations']==0
            r['layers'].append(item);del saved,maps;guard()
            print(json.dumps(dict(layer=layer,independently_verified=True,support_eligible=item['support_eligible'])),flush=True)
        entered=[v['layer'] for v in raw['layers']];assert entered==list(range(len(entered))) and 0<len(entered)<=24
        complete=len(entered)==24 and all(v['support_eligible'] for v in raw['layers'])
        if not complete:assert raw['layers'][-1]['support_eligible'] is False and all(v['support_eligible'] for v in raw['layers'][:-1])
        supported=[v['layer'] for v in raw['layers'] if v['support_eligible']]
        assert complete==raw['all24_source_initialized_and_NEW_supported']
        assert raw['new_cohort_supported_initialized_layers']==supported and raw['source_initialized_layers_available']==sorted({0,12,*supported})
        assert raw['new_source_feature_evaluations']==sum(v['new_source_feature_evaluations'] for v in raw['layers'])
        r['procedure_gates'].update(first_independent_NEW_scalar_offsets_byte_dictionary_maps=True,
            all_saved_support_counts_and_mass_scopes_verified=True,warm_projection_seed_bytes_and_GUD_references_exact=True,
            ALL_saved20_balanced_assignments_means_costs_exact_and_query_envelope=True,
            NEW_projection_sanity_saved_score_order_and_ALL_NEW_source_copies_exact=True,
            incremental_support_stop_and_complete_layer_counts_verified=True,no_source_feature_field_actual_route_old_namespace_or_optimizer_reexecution=True)
        r.update(decision='BALANCED_ALL24_INITIALIZER_INDEPENDENTLY_VERIFIED' if complete else 'BALANCED_PARTIAL_SUPPORT_STOP_INDEPENDENTLY_VERIFIED',
            all24_source_initialized_and_NEW_supported=complete,total_NEW_source_copy_elements_checked=sum(v['source_copy_elements_checked'] for v in r['layers']),
            elapsed_before_final_serialization=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            independently_verified_balanced_construction_steps=20*len(r['layers']),
            scope='FIRST construction/query-envelope/input/index/byte/support audit; source feature-energy and actual routing arithmetic NOT independently reexecuted; not whole quality')
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],layers=len(r['layers']))),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),fault_stage=stage,elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset)
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
