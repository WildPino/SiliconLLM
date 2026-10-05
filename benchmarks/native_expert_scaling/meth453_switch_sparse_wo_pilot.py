"""Compact WI plus original I8 WO with qualified exact zero-code skipping."""
import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_info,threadpool_limits
import meth451_switch_orthogonal_i4_pilot as S
import meth453_switch_sparse_wo_math as F

M,R,G,H=S.M,S.R,S.G,S.H
PROTOCOL=M.DOC/'METH_453_SWITCH_SPARSE_WO_PROTOCOL_20261005.md'
OUT=M.ROOT/'results/native_expert_scaling/meth453_switch_sparse_wo'
PARENTS={'meth451_switch_orthogonal_i4_result.json':'26e6ee00de51ea1cfc4cf6617fc69cef36950e1298ae19f153357545dd19f46f',
         'meth452_switch_operator_attribution_result.json':'857128c717284ab7796384e1ad6d3d214d64e46da43158efd127a0839080e3c7'}


class Cache:
    def __init__(self,bank):self.bank,self.cache=bank,OrderedDict()
    def get(self,identity):
        assert 0<=identity<128
        if identity not in self.cache:
            if len(self.cache)==8:self.cache.popitem(last=False)
            self.cache[identity]=(H.B.PackedOperator(self.bank['wi_packed'][identity],self.bank['wi_scales'][identity]),
                F.SparseColumns(self.bank['wo_columns'][identity],self.bank['wo_scales'][identity]))
        self.cache.move_to_end(identity);return self.cache[identity]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();numeric=None;peak=hashed=0;stage='bindings';result={'experiment':'METH-453-fixed-WI-I4-native-WO-I8-zero-only-column-layout','controls':{}}
    def guard():
        nonlocal peak
        mi=psutil.Process().memory_info();peak=max(peak,mi.rss,getattr(mi,'peak_wset',0))
        size=sum(p.stat().st_size for p in OUT.glob('*') if p.is_file()) if OUT.exists() else 0;elapsed=time.monotonic()-start
        assert peak<=4<<30 and size<=704<<20,'4GiB_704MiB'
        assert elapsed<=900 and ((numeric is None and elapsed<=300) or (numeric is not None and time.monotonic()-numeric<=600)),'admission300_numeric600_total900'
    def sha(path):
        nonlocal hashed
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            while b:=f.read(4<<20):h.update(b);hashed+=len(b);guard()
        return h.hexdigest()
    try:
        result['helper_sha256']={}
        for p in (Path(__file__),Path(F.__file__),PROTOCOL):M.committed(p);result['helper_sha256'][str(p)]=sha(p)
        parents={};files={};seen=dict(result['helper_sha256'])
        for name,expected in PARENTS.items():
            p=M.DOC/name;M.committed(p);assert sha(p)==expected;r=json.loads(p.read_bytes());parents[name]=r
            assert all(r['apparatus_gates'].values())
            for path,expected in r['helper_sha256'].items():
                if path in seen:assert seen[path]==expected;continue
                M.committed(path);assert sha(path)==expected;seen[path]=expected
            for item in r['output_inventory']:
                assert sha(item['path'])==item['sha256'];files[(name,Path(item['path']).name)]=Path(item['path'])
        parent=parents['meth451_switch_orthogonal_i4_result.json'];factorial=parents['meth452_switch_operator_attribution_result.json']
        assert all(parent['uncompressed_control_gates'].values()) and not parent['feasibility_gates']['argmax_changed_fraction_le0_01']
        assert factorial['changed_pair_counts']=={'WI_only_pair_crosses':0,'WO_only_pair_crosses':2,'both_hybrid_pairs_cross':2,'only_joint_pair_crosses':0}
        result['helper_sha256']=seen;result['retained_record_sha256']=PARENTS
        own={os.getpid(),*(p.pid for p in psutil.Process().parents())}
        for proc in psutil.process_iter(['name','cmdline']):
            if proc.pid in own:continue
            name=(proc.info['name'] or '').lower();argv=proc.info['cmdline'] or []
            if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():result.setdefault('preserved_daemons',[]).append(proc.pid);continue
            assert not(name.startswith('python') or ('meth' in name and name.endswith('.exe'))),('concurrent_job',proc.pid,name)
        psutil.Process().cpu_affinity([0]);torch.set_num_threads(1);torch.set_num_interop_threads(1)
        assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6' and psutil.disk_usage(str(M.ROOT)).free>=2<<30
        engine=M.ROOT/'benchmarks/phase60/engine.c';M.committed(engine);assert sha(engine)==parent['preserved_engine_sha256'];result['preserved_engine_sha256']=parent['preserved_engine_sha256']
        for path,expected in parent['preserved_original_binary_sha256'].items():assert sha(path)==expected
        result['preserved_original_binary_sha256']=parent['preserved_original_binary_sha256']
        name,expected=R.C.U.EXPORT[128];p=M.DOC/name;M.committed(p);assert sha(p)==expected;export=json.loads(p.read_bytes());a=export['artifact'];assert a==parent['artifacts']['128']
        assert sha(a['payload'])==a['sha256'] and sha(a['manifest'])==a['manifest_sha256'];R.B.read_manifest(a['manifest'],export['original_config'],export['tensors'],Path(a['payload']))
        result['artifacts']={'128':a};initial=(Path(a['payload']).stat().st_size,Path(a['payload']).stat().st_mtime_ns)
        mapped=np.memmap(a['payload'],mode='r',dtype='u1');entries=export['tensors'];src=S.X.ExpertCache(mapped,entries)
        ff=R.C.tensor(mapped,entries,'decoder.block.11.layer.2.layer_norm.weight');fn=R.C.tensor(mapped,entries,'decoder.final_layer_norm.weight')
        head=G.I8Operator(R.C.tensor(mapped,entries,'lm_head.weight'),R.C.tensor(mapped,entries,'lm_head.weight','scales'))
        def oldfile(name):return files[('meth451_switch_orthogonal_i4_result.json',name)]
        with np.load(oldfile('original_prefixes.npz'),allow_pickle=False) as q:rows=q['rows'].copy();scores=q['scores'].copy();books=q['books'].copy();keys=q['keys'].tolist()
        assert rows.shape==(336,) and keys==parent['data']['keys'] and books.tolist()==parent['data']['books']
        original=np.load(oldfile('original_logits.npy'),allow_pickle=False);savedWI={}
        for label in ['RI4B64_original_ID','RI4B64_ID_plus1']:
            with np.load(oldfile(label+'_states.npz'),allow_pickle=False) as q:savedWI[label]=(q['up_raw'].copy(),q['actual_function_ids'].copy())
        with np.load(oldfile('bank.npz'),allow_pickle=False) as q:bank={k:q[k].copy() for k in ('wi_packed','wi_scales','wi_signs')}
        R.C.exact(bank['wi_signs'],H.fixed_signs(768,'WI'))
        numeric=time.monotonic();OUT.mkdir(parents=True);result['admission']={'seconds':numeric-start,'bytes_hashed':hashed};print(json.dumps({'admission_complete':result['admission']}),flush=True)
        with threadpool_limits(limits=1),torch.no_grad():
            result['runtime']={'torch':torch.__version__,'numpy':np.__version__,'CPU_affinity':[0],'BLAS':threadpool_info(),'GPU':False}
            paths={p['filepath'] for p in result['runtime']['BLAS']}|{str(M.ROOT/'.venv/Lib/site-packages/torch/lib/torch_cpu.dll'),str(M.ROOT/'.venv/Lib/site-packages/torch/_C.cp312-win_amd64.pyd')}
            result['runtime']['binary_sha256']={p:sha(p) for p in sorted(paths)}
            for path,expected in parent['runtime']['binary_sha256'].items():assert result['runtime']['binary_sha256'][path]==expected
            stage='tiny_zero_code_partial_line_qualification';result['tiny_qualification']=F.tiny_qualification(R.C.quant)
            stage='all128_original_WO_exact_column_layout'
            bank['wo_columns']=np.empty((128,3072,768),np.int8);bank['wo_scales']=np.empty((128,768),np.float32);phases=[];fingerprints={}
            for identity in range(128):
                name=f'decoder.block.11.layer.2.mlp.experts.expert_{identity}.wo.weight';weights=R.C.tensor(mapped,entries,name);scales=R.C.tensor(mapped,entries,name,'scales')
                assert weights.shape==(768,3072) and np.all(weights!=-128);bank['wo_columns'][identity]=np.ascontiguousarray(weights.T);bank['wo_scales'][identity]=scales
                R.C.exact(bank['wo_columns'][identity].T,weights);R.C.exact(bank['wo_scales'][identity],scales)
                phases.append(int(entries[name]['offset']%64));h=hashlib.sha256()
                for k in ('wi_packed','wi_scales','wo_columns','wo_scales'):h.update(bank[k][identity].tobytes())
                fingerprints[str(identity)]=h.hexdigest();guard()
            np.savez(OUT/'bank.npz',**bank)
            with np.load(OUT/'bank.npz',allow_pickle=False) as q:
                assert set(q.files)==set(bank)
                for k,v in bank.items():R.C.exact(q[k],v)
            nominal=sum(v.nbytes for v in bank.values());assert nominal==472253184
            result['conversion']={'original_WO_all_coefficients_scales_transpose_inverse_byte_exact':True,'WI_bytes_exact451_no_reencoding':True,
                'per_expert_fingerprints':fingerprints,'distinct_fingerprints':len(set(fingerprints.values())),'nominal_stored_bank_bytes':nominal,
                'source_bank_bytes':605945856,'nominal_storage_ratio':float(nominal/605945856),'source_WO_rowmajor_offset_phases':phases,
                'candidate_WO_alignment':'64-byte future C layout; NPZ serialization not the physical runtime payload'}
            pairs=Cache(bank);basis=H.ActivationBasis(bank['wi_signs'],256);quantdata={};source_nnzs=[];sparse_calls=0
            stage='all336_original_complete_native_and_sparse_WO_replay'
            for i,row in enumerate(rows):
                inp=G.NativeRMS.apply(torch.from_numpy(row['pre'].copy()),ff).numpy();R.C.exact(inp,row['input']);wi0,wo0=src.get(int(row['expert']));raw=wi0.native(inp);up=np.where(raw<0,np.float32(0),raw)
                _,swo=pairs.get(int(row['expert']));down,scale,codes,indices=swo.native(up,R.C.quant);R.C.exact(down,wo0.native(up));R.C.exact(down,row['down']);R.C.exact(raw,row['up_raw']);R.C.exact(up,row['up'])
                R.C.exact(scale,row['wo_scale']);R.C.exact(codes,row['wo_codes']);p=G.NativeProbability.apply(torch.from_numpy(scores[i]),int(row['expert'])).numpy();R.C.exact(p,row['probability'])
                post=row['pre']+p*down;final=G.NativeRMS.apply(torch.from_numpy(post),fn).numpy();hi=final*np.float32(1/np.sqrt(768));R.C.exact(head.native(hi),original[i])
                for name,v in [('post',post),('final',final),('head_input',hi)]:R.C.exact(v,row[name])
                source_nnzs.append(int(len(indices)));sparse_calls+=1;guard()
            result['original_replay']={'positions':336,'full_vocab_rows':336*32128,'all_native_states_logits_WO_A16_and_sparse_original_WO_exact':True,'WO_nonzero_codes':source_nnzs}
            np.save(OUT/'original_logits.npy',original);np.savez(OUT/'original_prefixes.npz',rows=rows,scores=scores,books=books,keys=np.asarray(keys))
            target=np.stack([R.probability(z) for z in original]);logp=np.stack([S.N.log_probability(z) for z in original]);entropy=np.asarray([R.numpy_loss(z.astype(np.float64),p) for z,p in zip(original,target)])
            used=np.zeros((128,3072),bool)
            for control in ('WI4_WO8_original_ID','WI4_WO8_ID_plus1','removed'):
                stage=control;out=np.empty((336,32128),np.float32);states={k:np.empty((336,d),np.float32) for k,d in [('up_raw',3072),('down',768),('post',768),('final',768),('head_input',768)]};ids=[];footprints=[]
                scales=np.empty(336,np.float32);codes=np.empty((336,3072),np.int16)
                for i,row in enumerate(rows):
                    identity=int(row['expert']) if control=='WI4_WO8_original_ID' else (int(row['expert'])+1)%128 if control=='WI4_WO8_ID_plus1' else -1;ids.append(identity)
                    if identity<0:raw=np.zeros(3072,np.float32);down=np.zeros(768,np.float32);post=row['pre'].copy()
                    else:
                        wi,swo=pairs.get(identity);raw=wi.native(basis.apply(row['input']),R.C.quant);oldlabel='RI4B64_original_ID' if control=='WI4_WO8_original_ID' else 'RI4B64_ID_plus1';R.C.exact(raw,savedWI[oldlabel][0][i]);assert identity==int(savedWI[oldlabel][1][i])
                        up=np.where(raw<0,np.float32(0),raw);down,scales[i],codes[i],indices=swo.native(up,R.C.quant);_,wo0=src.get(identity);R.C.exact(down,wo0.native(up));sparse_calls+=1
                        footprints.append(F.footprint(indices,phases[identity]));post=row['pre']+row['probability']*down
                        if control=='WI4_WO8_original_ID':used[identity,indices]=True
                    final=G.NativeRMS.apply(torch.from_numpy(post),fn).numpy();hi=final*np.float32(1/np.sqrt(768));out[i]=head.native(hi)
                    for k,v in [('up_raw',raw),('down',down),('post',post),('final',final),('head_input',hi)]:states[k][i]=v
                    guard()
                if control=='removed':R.C.exact(out,np.load(oldfile('removed_logits.npy'),allow_pickle=False))
                else:quantdata[control+'_WO_A16_codes']=codes;quantdata[control+'_WO_A16_scales']=scales
                np.save(OUT/(control+'_logits.npy'),out);np.savez(OUT/(control+'_states.npz'),**states,actual_function_ids=np.asarray(ids,np.int64))
                ce=np.asarray([R.numpy_loss(z.astype(np.float64),p) for z,p in zip(out,target)]);kl=ce-entropy;independent=np.asarray([np.dot(p,l-S.N.log_probability(z)) for p,l,z in zip(target,logp,out)])
                assert np.isfinite(kl).all() and np.min(kl)>=-1e-10 and np.max(np.abs(kl-independent))<=1e-10
                changed=np.argmax(out,axis=1)!=np.argmax(original,axis=1)
                result['controls'][control]={'mean_source_KL':float(np.mean(kl)),'median_KL':float(np.median(kl)),'p95_KL':float(np.quantile(kl,.95)),'max_KL':float(np.max(kl)),
                    'changed_argmax':int(np.sum(changed)),'changed_fraction':float(np.mean(changed)),'changed_mask':changed.tolist(),'per_position_KL':kl.tolist(),'actual_ids':ids,
                    'per_book':{str(b):{'mean_KL':float(np.mean(kl[books==b])),'changed_argmax':int(np.sum(changed[books==b]))} for b in range(18,24)},'logical_footprints':footprints}
                print(json.dumps({'control':control,'mean_KL':float(np.mean(kl)),'changed_argmax':int(np.sum(changed))}),flush=True)
            np.savez(OUT/'WO_quant_inputs.npz',**quantdata,correct_per_expert_active_column_union=used)
            correct=result['controls']['WI4_WO8_original_ID'];wrong=result['controls']['WI4_WO8_ID_plus1'];fp=correct['logical_footprints'];ratios=np.asarray([p['ratio_to_original4733952_payload'] for p in fp]);nz=np.asarray([p['nonzero_codes'] for p in fp])
            result['identity_mean_KL_harm']=wrong['mean_source_KL']-correct['mean_source_KL']
            result['logical_screen']={'mean_ratio':float(np.mean(ratios)),'p95_ratio':float(np.quantile(ratios,.95)),'max_ratio':float(np.max(ratios)),
                'mean_nonzero_codes':float(np.mean(nz)),'p95_nonzero_codes':float(np.quantile(nz,.95)),'max_nonzero_codes':int(np.max(nz)),
                'per_book_mean_ratio':{str(b):float(np.mean(ratios[books==b])) for b in range(18,24)},
                'per_expert_union_nonzero_columns':{str(i):int(np.sum(used[i])) for i in range(128)},
                'scope':'Logical local addressed payload/workspace screening only. Core/router/head/cache/prefill, pair-LUT builder/gathers, actual C latency and hardware DRAM unmeasured.'}
            result['comparison_to_rotated_WO_source_diagnostic']={'mean_KL453_minus452_WIonly':correct['mean_source_KL']-factorial['controls']['WI1_WO0']['mean_source_KL'],
                '453_global_changes':correct['changed_argmax'],'452_WIonly_global_changes':factorial['controls']['WI1_WO0']['changed_argmax'],'scope':'different WO input basis/rounding; no inherited bit identity or quality'}
            assert sparse_calls==1008 and basis.calls==672
            result['apparatus_gates']={'all_parents_helpers_payload_complete_archives_runtime_fresh':True,'all128_WO_transposes_coefficients_scales_saved_bank_and_WI_reuse_exact':True,
                'all336_original_native_full_heads_states_WO_A16_sparse_outputs_exact':True,'all672_correct_and_wrong_WI_raw_outputs_byte_exact451':True,
                'all1008_sparse_I32_512_full_I64_and_original_native_WO_outputs_exact':True,'all672_actual_WI_basis_checks_and_block_I32_I64_F32_primal_qualified':True,
                'all_computed_full_vocab_KL_independent_finite_qualified':True,'removed_complete_heads_byte_exact451':True,'tiny_signed_bound_zero_scale_cast_line_faults_qualified':True,
                'all_coefficients_IDS_private_ReLU_original_probability_norm_head_retained':True}
            result['feasibility_gates']={'mean_KL_le0_01':correct['mean_source_KL']<=.01,'every_book_KL_le0_05':all(p['mean_KL']<=.05 for p in correct['per_book'].values()),
                'argmax_changed_fraction_le0_01':correct['changed_fraction']<=.01,'identity_mean_KL_harm_ge0_01':result['identity_mean_KL_harm']>=.01,
                'NEW_stored_bank_ratio_le0_80':nominal/605945856<=.80,'logical_mean_and_every_book_footprint_ratio_le0_60':float(np.mean(ratios))<=.60 and all(float(np.mean(ratios[books==b]))<=.60 for b in range(18,24)),
                'logical_p95_footprint_ratio_le0_65':float(np.quantile(ratios,.95))<=.65}
            result['basis_checks']={'calls':basis.calls,'max_relative_F64_explicit_error':basis.max_relative_error};result['sparse_WO_projections_qualified']=sparse_calls
        assert initial==(Path(a['payload']).stat().st_size,Path(a['payload']).stat().st_mtime_ns)
        result['data']={'keys':keys,'books':books.tolist(),'selected_ids':rows['expert'].tolist(),'consumed_positions':336}
        result['output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.glob('*')) if p.is_file()]
        result['resource']={'seconds':time.monotonic()-start,'admission_seconds':numeric-start,'numeric_seconds':time.monotonic()-numeric,'peak_bytes':peak,'streamed_file_bytes_hashed':hashed,'output_bytes':sum(p['bytes'] for p in result['output_inventory']),'optimizer_updates':0,'new_coefficient_roundings':0}
        result['decision']='local_primal_prediction_logical_screen_PASS_prepare_NEW_actual_C_cost_primal' if all(result['feasibility_gates'].values()) else 'new_WI4_native_sparse_WO8_local_gate_FAIL_reassess_before_C_export_or_timing'
        result['scope']='All128 real one-bank functions; original-coordinate I8 WO exact zero A16 skipping, WI copied451. Consumed local prefixes only. New80%storage budget disclosed before observations, old60%both-I4 failures unchanged. Logical footprint not hardware DRAM or actual C cost; no all-bank export/new routes/whole fresh quality/SAMEartifact50/useful n/family100B proof.'
        guard();M.write(args.out,result);print(json.dumps({'apparatus':result['apparatus_gates'],'feasibility':result['feasibility_gates'],'logical_screen':{k:v for k,v in result['logical_screen'].items() if k not in ('per_expert_union_nonzero_columns','scope')},'decision':result['decision'],'resource':result['resource'],'sha256':M.digest(args.out)}),flush=True)
    except BaseException as error:
        result.update(failure_stage=stage,error=repr(error),seconds=time.monotonic()-start,peak_bytes=peak,streamed_file_bytes_hashed=hashed)
        if OUT.exists():result['partial_output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':M.digest(p)} for p in sorted(OUT.glob('*')) if p.is_file()]
        M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
