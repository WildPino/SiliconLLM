"""Native2x2 WI/WO diagnostics with byte-exact451 diagonal replay first."""
import argparse
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
import meth452_switch_operator_attribution_math as F

M,R,G,H = S.M,S.R,S.G,S.H
PROTOCOL=M.DOC/'METH_452_SWITCH_OPERATOR_ATTRIBUTION_PROTOCOL_20261005.md'
OUT=M.ROOT/'results/native_expert_scaling/meth452_switch_operator_attribution'
PARENT=('meth451_switch_orthogonal_i4_result.json','26e6ee00de51ea1cfc4cf6617fc69cef36950e1298ae19f153357545dd19f46f')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();numeric=None;peak=hashed=0;stage='bindings'
    result={'experiment':'METH-452-native-WI-WO-factorial-error-attribution','controls':{}}
    def guard():
        nonlocal peak
        mi=psutil.Process().memory_info();peak=max(peak,mi.rss,getattr(mi,'peak_wset',0))
        size=sum(p.stat().st_size for p in OUT.glob('*') if p.is_file()) if OUT.exists() else 0
        elapsed=time.monotonic()-start
        assert peak<=4<<30 and size<=128<<20,'4GiB_128MiB'
        assert elapsed<=900 and ((numeric is None and elapsed<=300) or
               (numeric is not None and time.monotonic()-numeric<=600)),'admission300_numeric600_total900'
    def sha(path):
        nonlocal hashed
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            while b:=f.read(4<<20):h.update(b);hashed+=len(b);guard()
        return h.hexdigest()
    try:
        result['helper_sha256']={}
        for p in (Path(__file__),Path(F.__file__),PROTOCOL):
            M.committed(p);result['helper_sha256'][str(p)]=sha(p)
        name,expected=PARENT;p=M.DOC/name;M.committed(p);assert sha(p)==expected;parent=json.loads(p.read_bytes())
        assert all(parent['apparatus_gates'].values()) and all(parent['uncompressed_control_gates'].values())
        assert sum(parent['feasibility_gates'].values())==4 and not parent['feasibility_gates']['argmax_changed_fraction_le0_01']
        assert parent['controls']['RI4B64_original_ID']['argmax_changed_positions']==4
        for p,expected in parent['helper_sha256'].items():
            M.committed(p);assert sha(p)==expected;result['helper_sha256'][p]=expected
        for name,expected in parent['retained_record_sha256'].items():
            p=M.DOC/name;M.committed(p);assert sha(p)==expected
        result['retained_record_sha256']={PARENT[0]:PARENT[1],**parent['retained_record_sha256']}
        files={}
        for item in parent['output_inventory']:assert sha(item['path'])==item['sha256'];files[Path(item['path']).name]=Path(item['path'])
        own={os.getpid(),*(p.pid for p in psutil.Process().parents())}
        for proc in psutil.process_iter(['name','cmdline']):
            if proc.pid in own:continue
            name=(proc.info['name'] or '').lower();argv=proc.info['cmdline'] or []
            if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
                result.setdefault('preserved_daemons',[]).append(proc.pid);continue
            assert not(name.startswith('python') or ('meth' in name and name.endswith('.exe'))),('concurrent_job',proc.pid,name)
        psutil.Process().cpu_affinity([0]);torch.set_num_threads(1);torch.set_num_interop_threads(1)
        assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6' and psutil.disk_usage(str(M.ROOT)).free>=1<<30
        engine=M.ROOT/'benchmarks/phase60/engine.c';M.committed(engine);assert sha(engine)==parent['preserved_engine_sha256']
        result['preserved_engine_sha256']=parent['preserved_engine_sha256']
        for p,expected in parent['preserved_original_binary_sha256'].items():assert sha(p)==expected
        result['preserved_original_binary_sha256']=parent['preserved_original_binary_sha256']
        name,expected=R.C.U.EXPORT[128];p=M.DOC/name;M.committed(p);assert sha(p)==expected;export=json.loads(p.read_bytes())
        artifact=export['artifact'];assert artifact==parent['artifacts']['128']
        assert sha(artifact['payload'])==artifact['sha256'] and sha(artifact['manifest'])==artifact['manifest_sha256']
        R.B.read_manifest(artifact['manifest'],export['original_config'],export['tensors'],Path(artifact['payload']))
        result['artifacts']={'128':artifact};initial=(Path(artifact['payload']).stat().st_size,Path(artifact['payload']).stat().st_mtime_ns)
        mapped=np.memmap(artifact['payload'],mode='r',dtype='u1');entries=export['tensors']
        fn=R.C.tensor(mapped,entries,'decoder.final_layer_norm.weight')
        head=G.I8Operator(R.C.tensor(mapped,entries,'lm_head.weight'),R.C.tensor(mapped,entries,'lm_head.weight','scales'))
        with np.load(files['original_prefixes.npz'],allow_pickle=False) as q:
            rows=q['rows'].copy();books=q['books'].copy();keys=q['keys'].tolist()
        assert rows.shape==(336,) and keys==parent['data']['keys'] and books.tolist()==parent['data']['books']
        original=np.load(files['original_logits.npy'],allow_pickle=False)
        diagonals={};diagonal_states={}
        for label,old in [('WI0_WO0','rotated_source_original_ID'),('WI1_WO1','RI4B64_original_ID')]:
            diagonals[label]=np.load(files[old+'_logits.npy'],allow_pickle=False)
            with np.load(files[old+'_states.npz'],allow_pickle=False) as q:diagonal_states[label]={k:q[k].copy() for k in q.files}
        with np.load(files['bank.npz'],allow_pickle=False) as q:bank={k:q[k].copy() for k in q.files}
        signs={'wi':H.fixed_signs(768,'WI'),'wo':H.fixed_signs(3072,'WO')};blocks={'wi':256,'wo':1024}
        for name in ('wi','wo'):R.C.exact(signs[name],bank[name+'_signs']);assert hashlib.sha256(signs[name].tobytes()).hexdigest()==parent['conversion']['sign_sha256'][name]
        numeric=time.monotonic();OUT.mkdir(parents=True);result['admission']={'seconds':numeric-start,'bytes_hashed':hashed}
        print(json.dumps({'admission_complete':result['admission']}),flush=True)
        with threadpool_limits(limits=1),torch.no_grad():
            result['runtime']={'torch':torch.__version__,'numpy':np.__version__,'CPU_affinity':[0],'BLAS':threadpool_info(),'GPU':False}
            paths={p['filepath'] for p in result['runtime']['BLAS']}|{str(M.ROOT/'.venv/Lib/site-packages/torch/lib/torch_cpu.dll'),str(M.ROOT/'.venv/Lib/site-packages/torch/_C.cp312-win_amd64.pyd')}
            result['runtime']['binary_sha256']={p:sha(p) for p in sorted(paths)}
            for p,expected in parent['runtime']['binary_sha256'].items():assert result['runtime']['binary_sha256'][p]==expected
            stage='tiny_factorial_qualification';result['tiny_qualification']=F.tiny_qualification()
            bases={name:H.ActivationBasis(signs[name],blocks[name]) for name in ('wi','wo')}
            sourcecache=S.RotatedCache(mapped,entries,signs,blocks);packedcache=S.PackedCache(bank)
            hybrids={label:np.empty((336,32128),np.float32) for label in ('WI1_WO0','WI0_WO1')}
            states={label:{s:np.empty((336,d),np.float32) for s,d in (('up_raw',3072),('down',768),('post',768),('final',768),('head_input',768))} for label in hybrids}
            masks=[];checked_codes=set();projection_calls={'source':0,'compact':0};replayed=0
            def tail(row,raw,down):
                post=row['pre']+row['probability']*down;final=G.NativeRMS.apply(torch.from_numpy(post),fn).numpy();hi=final*np.float32(1/np.sqrt(768))
                return head.native(hi),{'up_raw':raw,'down':down,'post':post,'final':final,'head_input':hi}
            stage='rowwise_complete_diagonals_then_hybrids'
            for i,row in enumerate(rows):
                identity=int(row['expert']);wi0,wo0=sourcecache.get(identity);wi1,wo1=packedcache.get(identity)
                for name,operator in [('wi',wi0),('wo',wo0)]:
                    if (identity,name) not in checked_codes:
                        assert sha_bytes(operator.codes)==parent['conversion']['per_matrix'][f'e{identity}_{name}']['rotated_I32_sha256'];checked_codes.add((identity,name))
                xr=bases['wi'].apply(row['input']);raw0=wi0.native(xr,R.C.quant);raw1=wi1.native(xr,R.C.quant)
                up0=np.where(raw0<0,np.float32(0),raw0);up1=np.where(raw1<0,np.float32(0),raw1)
                yr0=bases['wo'].apply(up0);yr1=bases['wo'].apply(up1)
                down00=wo0.native(yr0,R.C.quant);down11=wo1.native(yr1,R.C.quant)
                for label,raw,down in [('WI0_WO0',raw0,down00),('WI1_WO1',raw1,down11)]:
                    logits,st=tail(row,raw,down);R.C.exact(logits,diagonals[label][i])
                    for field,value in st.items():R.C.exact(value,diagonal_states[label][field][i])
                    assert int(diagonal_states[label]['actual_function_ids'][i])==identity;replayed+=1
                down10=wo0.native(yr1,R.C.quant);down01=wo1.native(yr0,R.C.quant)
                for label,raw,down in [('WI1_WO0',raw1,down10),('WI0_WO1',raw0,down01)]:
                    hybrids[label][i],st=tail(row,raw,down)
                    for field,value in st.items():states[label][field][i]=value
                masks.append(int(np.count_nonzero((raw0>0)!=(raw1>0))))
                projection_calls['source']+=3;projection_calls['compact']+=3;guard()
            assert replayed==672 and projection_calls=={'source':1008,'compact':1008}
            assert bases['wi'].calls==336 and bases['wo'].calls==672
            result['diagonal_replay']={'complete_heads':replayed,'vocabulary_rows':replayed*32128,'all_logits_states_IDS_byte_exact451':True}
            result['source_transformed_code_fingerprints_checked']=len(checked_codes)
            result['projection_counts']=projection_calls;result['activation_basis_checks']={k:{'calls':v.calls,'max_relative_error':v.max_relative_error} for k,v in bases.items()}
            stage='all_diagonals_exact_hybrid_metrics_factorial_terms'
            target=np.stack([R.probability(z) for z in original]);logp=np.stack([S.N.log_probability(z) for z in original]);entropy=np.asarray([R.numpy_loss(z.astype(np.float64),p) for z,p in zip(original,target)])
            for label,logits in hybrids.items():
                ce=np.asarray([R.numpy_loss(z.astype(np.float64),p) for z,p in zip(logits,target)]);kl=ce-entropy
                independent=np.asarray([np.dot(p,l-S.N.log_probability(z)) for p,l,z in zip(target,logp,logits)])
                assert np.isfinite(kl).all() and np.min(kl)>=-1e-10 and np.max(np.abs(kl-independent))<=1e-10
                changed=np.argmax(logits,axis=1)!=np.argmax(original,axis=1)
                result['controls'][label]={'mean_source_KL':float(np.mean(kl)),'median_KL':float(np.median(kl)),'p95_KL':float(np.quantile(kl,.95)),'max_KL':float(np.max(kl)),
                    'changed_argmax':int(np.sum(changed)),'per_position_KL':kl.tolist(),'argmax_changed_mask':changed.tolist(),
                    'per_book':{str(b):{'mean_KL':float(np.mean(kl[books==b])),'argmax_changes':int(np.sum(changed[books==b]))} for b in range(18,24)}}
                np.save(OUT/(label+'_logits.npy'),logits);np.savez(OUT/(label+'_states.npz'),**states[label],actual_function_ids=rows['expert'].astype(np.int64))
            records=[];counts={k:0 for k in ['WI_only_pair_crosses','WO_only_pair_crosses','both_hybrid_pairs_cross','only_joint_pair_crosses']}
            for i in range(336):
                a=int(np.argmax(original[i]));c=int(np.argmax(diagonals['WI1_WO1'][i]));temp=original[i].copy();temp[a]=-np.inf;b=c if c!=a else int(np.argmax(temp))
                item=F.terms(original[i],diagonals['WI0_WO0'][i],hybrids['WI1_WO0'][i],hybrids['WI0_WO1'][i],diagonals['WI1_WO1'][i],a,b)
                wi=F.pair_loses(item['WI_only_pair_gap'],a,b);wo=F.pair_loses(item['WO_only_pair_gap'],a,b)
                category=('both_hybrid_pairs_cross' if wi and wo else 'WI_only_pair_crosses' if wi else 'WO_only_pair_crosses' if wo else 'only_joint_pair_crosses') if c!=a else 'compact_original_winner_retained'
                if c!=a:counts[category]+=1
                item.update(index=i,key=keys[i],original_winner=a,compact_winner=c,paired_competitor=b,
                    WI_only_global_winner=int(np.argmax(hybrids['WI1_WO0'][i])),WO_only_global_winner=int(np.argmax(hybrids['WI0_WO1'][i])),
                    compact_changed=c!=a,changed_pair_category=category,
                    explicit_interaction_needed_for_pair=bool(c!=a and not F.pair_loses(item['pair_gap_without_explicit_interaction'],a,b)),relu_mask_changes=masks[i])
                records.append(item)
            assert sum(counts.values())==4
            result['per_position_factorial']=records;result['changed_pair_counts']=counts
            result['interaction_needed_changed_pair_indices']=[p['index'] for p in records if p['explicit_interaction_needed_for_pair']]
            result['relu_mask_changes']={'per_position':masks,'mean':float(np.mean(masks)),'max':max(masks),'width':3072}
            result['apparatus_gates']={'all_parent_sources_outputs_payload_runtime_fresh_exact':True,'all672_complete_diagonal_heads_states_ids_byte_exact451':True,
                'all_actual_source_codes_match451_integer_transform_fingerprints':True,'all2016_source_I64_compact_I32_I64_native_primal_exact':True,
                'all1008_actual_basis_transforms_independent_Walsh_qualified':True,'all336_full_vocab_factorial_and_pair_margin_identities_le1e_minus10':True,
                'all_hybrid_KL_independent_and_finite_qualified':True,'known_tiny_main_interaction_baseline_tie_rules_qualified':True,
                'same_original_ID_input_probability_norm_head_no_encoding_fit':True}
        assert initial==(Path(artifact['payload']).stat().st_size,Path(artifact['payload']).stat().st_mtime_ns)
        result['data']={'keys':keys,'books':books.tolist(),'selected_ids':rows['expert'].tolist(),'consumed_positions':336}
        result['output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.glob('*')) if p.is_file()]
        result['resource']={'seconds':time.monotonic()-start,'admission_seconds':numeric-start,'numeric_seconds':time.monotonic()-numeric,
            'peak_bytes':peak,'bytes_hashed':hashed,'output_bytes':sum(a['bytes'] for a in result['output_inventory']),'optimizer_updates':0,'new_encodings':0,'full_head_forwards':1344}
        result['scope']='Native2x2 fixed-prefix diagnostic only. Higher-precision hybrids exceed60%cap, not a candidate compact model. No C export/rate/DRAM/whole fresh quality/useful-n/family100B inference. Main/joint terms exact discrete factorial decomposition, no additive quality or ReLU-only attribution.'
        guard();M.write(args.out,result);print(json.dumps({'controls':{k:{a:v[a] for a in ('mean_source_KL','changed_argmax')} for k,v in result['controls'].items()},'changed_pair_counts':counts,'interaction_needed':result['interaction_needed_changed_pair_indices'],'apparatus':result['apparatus_gates'],'resource':result['resource'],'sha256':M.digest(args.out)}),flush=True)
    except BaseException as error:
        result.update(failure_stage=stage,error=repr(error),seconds=time.monotonic()-start,peak_bytes=peak,bytes_hashed=hashed)
        if OUT.exists():result['partial_output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':M.digest(p)} for p in sorted(OUT.glob('*')) if p.is_file()]
        M.write(args.out.with_suffix('.failure.json'),result);raise


def sha_bytes(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


if __name__=='__main__':main()
