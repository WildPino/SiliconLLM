"""One calibration-free row-I4 final-bank probe on source128 native prefixes."""
import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_info, threadpool_limits
import meth443_switch_native_function_causality as N
import meth447_switch_i4_math as H

M,R,G,X = N.M,N.R,N.G,N.X
PROTOCOL = M.DOC/'METH_447_SWITCH_I4_PROTOCOL_20261005.md'
OUT = M.ROOT/'results/native_expert_scaling/meth447_switch_i4'
RECORDS = {'meth443_switch_native_causality_result.json':'b8e8614fb28ad50b91865dba7486c6e4cc6bb8c00cee3bbfe899ab495af5bf80',
           'meth446_switch_private_delta_result.json':'b6c525905cb2df1b3adb88c0131a09d34ac384328ff4208e4cd95609ed9773ee',
           'meth418_switch_function_capture_result.json':N.RECORDS['meth418_switch_function_capture_result.json'],
           'meth420_switch_function_gradient_result.failure.json':N.RECORDS['meth420_switch_function_gradient_result.failure.json']}


class PackedCache:
    def __init__(self, bank): self.bank,self.cache = bank,OrderedDict()
    def get(self, identity):
        assert 0 <= identity < 128
        if identity not in self.cache:
            if len(self.cache) == 8:self.cache.popitem(last=False)
            self.cache[identity] = tuple(H.PackedOperator(self.bank[s+'_packed'][identity],
                                        self.bank[s+'_scales'][identity]) for s in ('wi','wo'))
        self.cache.move_to_end(identity)
        return self.cache[identity]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();numeric_start=None;peak=hashed=0;stage='bindings'
    result={'experiment':'METH-447-fixed-row-I4-source128-last-bank-native-head-probe','controls':{}}
    def guard():
        nonlocal peak
        memory=psutil.Process().memory_info();peak=max(peak,memory.rss,getattr(memory,'peak_wset',0))
        size=sum(p.stat().st_size for p in OUT.glob('*') if p.is_file()) if OUT.exists() else 0
        elapsed=time.monotonic()-start
        assert peak<=3<<30 and size<=512<<20,'3GiB_512MiB'
        assert elapsed<=600 and ((numeric_start is None and elapsed<=300) or
               (numeric_start is not None and time.monotonic()-numeric_start<=300)),'admission300_numeric300_total600'
    def sha(path):
        nonlocal hashed
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block:=stream.read(4<<20):h.update(block);hashed+=len(block);guard()
        return h.hexdigest()
    try:
        result['helper_sha256']={}
        for p in (Path(__file__),Path(H.__file__),PROTOCOL,Path(N.__file__),Path(X.__file__),Path(M.__file__),
                  Path(R.__file__),Path(R.C.__file__),Path(G.__file__),Path(R.B.__file__),Path(R.C.U.__file__)):
            M.committed(p);result['helper_sha256'][str(p)]=sha(p)
        seen=dict(result['helper_sha256']);files={};records={}
        for name,expected in RECORDS.items():
            p=M.DOC/name;M.committed(p);assert sha(p)==expected;records[name]=record=json.loads(p.read_text(encoding='utf-8'))
            for p,expected in record['helper_sha256'].items():
                if p in seen:assert seen[p]==expected;continue
                M.committed(Path(p));assert sha(p)==expected;seen[p]=expected
            for a in record.get('output_inventory',[]):
                if a['path'] in files:assert files[a['path']]==a['sha256'];continue
                assert sha(a['path'])==a['sha256'];files[a['path']]=a['sha256']
        parent=records['meth443_switch_native_causality_result.json'];closed=records['meth446_switch_private_delta_result.json']
        prior=records['meth418_switch_function_capture_result.json'];baseline=records['meth420_switch_function_gradient_result.failure.json']
        assert all(parent['apparatus_gates'].values()) and parent['diagnostic_gates']['primary_native_identity_mean_KL_ge0_01']
        assert all(closed['apparatus_gates'].values()) and not closed['decision']['all_three_target_eligible_recipes']
        assert all(prior['gates'].values()) and len(baseline['baselines'])==384
        for a in baseline['baselines']:assert sha(a['archive_path'])==a['archive_sha256']
        for p,expected in parent['preserved_original_binary_sha256'].items():assert sha(p)==expected
        result['retained_record_sha256']=RECORDS;result['preserved_original_binary_sha256']=parent['preserved_original_binary_sha256']
        own={os.getpid(),*(p.pid for p in psutil.Process().parents())}
        for process in psutil.process_iter(['name','cmdline']):
            if process.pid in own:continue
            name=(process.info['name'] or '').lower();argv=process.info['cmdline'] or []
            if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
                result.setdefault('preserved_daemons',[]).append(process.pid);continue
            assert not(name.startswith('python') or ('meth' in name and name.endswith('.exe'))),('concurrent_job',process.pid,name)
        psutil.Process().cpu_affinity([0]);torch.set_num_threads(1);torch.set_num_interop_threads(1)
        assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6' and psutil.disk_usage(str(M.ROOT)).free>=2<<30
        name,expected=R.C.U.EXPORT[128];p=M.DOC/name;M.committed(p);assert sha(p)==expected
        export=json.loads(p.read_text(encoding='utf-8'));artifact=export['artifact'];assert artifact==parent['artifacts']['128']
        assert sha(artifact['payload'])==artifact['sha256'] and sha(artifact['manifest'])==artifact['manifest_sha256']
        R.B.read_manifest(artifact['manifest'],export['original_config'],export['tensors'],Path(artifact['payload']))
        initial=(Path(artifact['payload']).stat().st_size,Path(artifact['payload']).stat().st_mtime_ns)
        mapped=np.memmap(artifact['payload'],dtype='u1',mode='r');entries=export['tensors'];result['artifacts']={'128':artifact}
        ff=R.C.tensor(mapped,entries,'decoder.block.11.layer.2.layer_norm.weight');fn=R.C.tensor(mapped,entries,'decoder.final_layer_norm.weight')
        head=G.I8Operator(R.C.tensor(mapped,entries,'lm_head.weight'),R.C.tensor(mapped,entries,'lm_head.weight','scales'));cache=X.ExpertCache(mapped,entries)
        caps={a['label']:a for a in prior['captures']};old={a['label']:a for a in baseline['baselines']}
        captured=[];logits=[];scores=[];books=[];keys=[]
        for b in range(18,24):
            for c in range(4):
                item=caps[f'teacher.n128.book{b}.case{c}'];a=old[item['label']]
                assert item['prospective_split']=='validation' and not a['qualification_only_not_paired_training'] and a['source_capture_sha256']==item['capture_sha256']
                data=Path(item['capture_path']).read_bytes();assert data[:8]==b'SWFUN001' and struct.unpack_from('<6I',data,8)==(768,3072,32128,128,11,1)
                assert len(data)==32+14*R.C.DTYPE.itemsize;rows=np.frombuffer(data,dtype=R.C.DTYPE,offset=32).copy()
                assert np.array_equal(rows['position'],np.arange(14)) and np.array_equal(rows['id'],item['decoder_ids'])
                trace=Path(item['trace_path']).read_bytes();assert trace[:8]==b'SWRTA001' and struct.unpack_from('<2I',trace,8)==(128,768)
                dt=np.dtype([('index','<u4'),('phase','<u4'),('input','<f4',(768,)),('scores','<f4',(128,))]);tr=np.frombuffer(trace,dtype=dt,offset=16)[179+6*np.arange(14)]
                R.C.exact(tr['input'],rows['input']);assert np.array_equal(np.argmax(tr['scores'],axis=1),rows['expert'])
                with np.load(a['archive_path'],allow_pickle=False) as q:
                    assert np.array_equal(q['source_ids'],item['source_ids']) and np.array_equal(q['decoder_ids'],item['decoder_ids'])
                    logits.append(q['logits'].copy())
                    for field in ('input','up_raw','up','down','probability','post','final','head_input'):R.C.exact(q[field],rows[field])
                captured.append(rows);scores.append(tr['scores'].copy());books.extend([b]*14)
                keys.extend([f'book{b}.case{c}.position{j}:{item["pairing_sha256"]}' for j in range(14)])
        rows=np.concatenate(captured);scores=np.concatenate(scores);original=np.concatenate(logits);books=np.asarray(books)
        del captured,logits
        assert rows.shape==(336,) and original.shape==(336,32128) and keys==parent['data']['keys']
        numeric_start=time.monotonic();OUT.mkdir(parents=True);result['admission']={'seconds':numeric_start-start,'bytes_hashed':hashed}
        print(json.dumps({'admission_complete':result['admission']}),flush=True)
        with threadpool_limits(limits=1),torch.no_grad():
            result['runtime']={'torch':torch.__version__,'numpy':np.__version__,'CPU_affinity':[0],'BLAS':threadpool_info(),'GPU':False}
            runtime_paths={a['filepath'] for a in result['runtime']['BLAS']}|{str(M.ROOT/'.venv/Lib/site-packages/torch/lib/torch_cpu.dll'),str(M.ROOT/'.venv/Lib/site-packages/torch/_C.cp312-win_amd64.pyd')}
            result['runtime']['binary_sha256']={p:sha(p) for p in sorted(runtime_paths)}
            stage='tiny_packing_integer_qualification';result['tiny_qualification']=H.tiny_qualification(R.C.quant);guard()
            stage='all128_real_functions_fixed_I4_encoding'
            bank={'wi_packed':np.empty((128,3072,384),np.uint8),'wi_scales':np.empty((128,3072),np.float32),
                  'wo_packed':np.empty((128,768,1536),np.uint8),'wo_scales':np.empty((128,768),np.float32)}
            result['conversion']={'per_matrix':{},'per_expert_fingerprints':{}}
            for identity in range(128):
                ep=f'decoder.block.11.layer.2.mlp.experts.expert_{identity}.';fingerprint=hashlib.sha256()
                for s in ('wi','wo'):
                    weights=R.C.tensor(mapped,entries,ep+s+'.weight');scales=R.C.tensor(mapped,entries,ep+s+'.weight','scales')
                    packed,newscales,metrics=H.encode(weights,scales)
                    bank[s+'_packed'][identity]=packed;bank[s+'_scales'][identity]=newscales
                    fingerprint.update(packed.tobytes());fingerprint.update(newscales.tobytes());result['conversion']['per_matrix'][f'e{identity}_{s}']=metrics;guard()
                result['conversion']['per_expert_fingerprints'][str(identity)]=fingerprint.hexdigest()
            result['conversion']['distinct_packed_expert_fingerprints']=len(set(result['conversion']['per_expert_fingerprints'].values()))
            np.savez(OUT/'bank.npz',**bank)
            with np.load(OUT/'bank.npz',allow_pickle=False) as saved:
                assert set(saved.files)==set(bank)
                for s,v in bank.items():R.C.exact(saved[s],v)
            result['conversion']['source_bank_coefficient_and_row_scale_bytes']=128*4733952
            result['conversion']['packed_bank_coefficient_and_row_scale_bytes']=sum(v.nbytes for v in bank.values())
            result['conversion']['nominal_storage_ratio']=sum(v.nbytes for v in bank.values())/(128*4733952)
            print(json.dumps({'conversion_complete':{k:v for k,v in result['conversion'].items() if k not in ('per_matrix','per_expert_fingerprints')}}),flush=True)
            stage='all336_original_native_complete_replay';stream=hashlib.sha256()
            for i,row in enumerate(rows):
                inp=G.NativeRMS.apply(torch.from_numpy(row['pre'].copy()),ff).numpy();R.C.exact(inp,row['input'])
                wi,wo=cache.get(int(row['expert']));raw=wi.native(inp);up=np.where(raw<0,np.float32(0),raw);down=wo.native(up)
                p=G.NativeProbability.apply(torch.from_numpy(scores[i]),int(row['expert'])).numpy();post=row['pre']+p*down
                final=G.NativeRMS.apply(torch.from_numpy(post),fn).numpy();hi=final*np.float32(1/np.sqrt(768));out=head.native(hi)
                for field,v in (('up_raw',raw),('up',up),('down',down),('probability',p),('post',post),('final',final),('head_input',hi)):R.C.exact(v,row[field])
                for field,v in (('wi',inp),('wo',up),('head',hi)):
                    scale,codes=R.C.quant(v);R.C.exact(scale,row[field+'_scale']);R.C.exact(codes,row[field+'_codes'])
                R.C.exact(out,original[i]);stream.update(out.tobytes());guard()
            assert stream.hexdigest()==parent['original_replay']['logit_stream_sha256']
            result['original_replay']={'positions':336,'full_vocab_rows':336*32128,'native_states_quant_logits_byte_exact':True,'logit_stream_sha256':stream.hexdigest()}
            np.save(OUT/'original_logits.npy',original);np.savez(OUT/'original_prefixes.npz',rows=rows,scores=scores,books=books,keys=np.asarray(keys))
            targets=np.stack([R.probability(v) for v in original]);oldlogp=np.stack([N.log_probability(v) for v in original])
            entropy=np.asarray([R.numpy_loss(original[i].astype(np.float64),targets[i]) for i in range(336)])
            assert np.max(np.abs(entropy+np.sum(targets*oldlogp,axis=1)))<=1e-10
            pcache=PackedCache(bank);argmax=np.argmax(original,axis=1)
            result['data']={'keys':keys,'books':books.tolist(),'positions':rows['position'].tolist(),'selected_ids':rows['expert'].tolist(),
                'selected_probability':rows['probability'].tolist(),'natural_distinct_selected_ids':int(len(np.unique(rows['expert']))),
                'original_self_CE_mean':float(np.mean(entropy)),'original_self_CE':entropy.tolist(),'consumed_validation_only_no_training':True}
            for control in ('I4_original_ID','I4_ID_plus1','removed'):
                stage=control;out=np.empty(original.shape,np.float32)
                states={s:np.empty((336,d),np.float32) for s,d in (('up_raw',3072),('down',768),('post',768),('final',768),('head_input',768))}
                effects={s:[] for s in states};ids=[]
                for i,row in enumerate(rows):
                    identity=int(row['expert']) if control=='I4_original_ID' else (int(row['expert'])+1)%128 if control=='I4_ID_plus1' else -1;ids.append(identity)
                    if identity<0:raw=np.zeros(3072,np.float32);down=np.zeros(768,np.float32);post=row['pre'].copy()
                    else:
                        wi,wo=pcache.get(identity);raw=wi.native(row['input'],R.C.quant)
                        down=wo.native(np.where(raw<0,np.float32(0),raw),R.C.quant);post=row['pre']+row['probability']*down
                    final=G.NativeRMS.apply(torch.from_numpy(post),fn).numpy();hi=final*np.float32(1/np.sqrt(768));out[i]=head.native(hi)
                    for s,v in (('up_raw',raw),('down',down),('post',post),('final',final),('head_input',hi)):
                        states[s][i]=v;effects[s].append(R.relative(v,row[s]))
                    guard()
                if control=='removed':
                    ref=next(Path(a['path']) for a in parent['output_inventory'] if Path(a['path']).name=='removed_logits.npy')
                    R.C.exact(np.load(ref,allow_pickle=False),out)
                np.save(OUT/(control+'_logits.npy'),out);np.savez(OUT/(control+'_states.npz'),**states,actual_function_ids=np.asarray(ids))
                ce=np.asarray([R.numpy_loss(out[i].astype(np.float64),targets[i]) for i in range(336)]);kl=ce-entropy
                independent=np.asarray([np.dot(targets[i],oldlogp[i]-N.log_probability(out[i])) for i in range(336)])
                assert np.isfinite(ce).all() and np.min(kl)>=-1e-10 and np.max(np.abs(kl-independent))<=1e-10
                changed=np.argmax(out,axis=1)!=argmax
                result['controls'][control]={'mean_self_teacher_KL':float(np.mean(kl)),'median_KL':float(np.median(kl)),'p95_KL':float(np.quantile(kl,.95)),
                    'max_KL':float(np.max(kl)),'argmax_changed_positions':int(np.sum(changed)),'argmax_changed_fraction':float(np.mean(changed)),
                    'per_position_KL':kl.tolist(),'per_position_CE':ce.tolist(),'independent_KL':independent.tolist(),'argmax_changed_mask':changed.tolist(),
                    'actual_function_ids':ids,'relative_state_effects':effects,
                    'per_book':{str(b):{'mean_KL':float(np.mean(kl[books==b])),'argmax_changes':int(np.sum(changed[books==b]))} for b in range(18,24)}}
                print(json.dumps({'control_complete':control,'mean_KL':float(np.mean(kl)),'argmax_changes':int(np.sum(changed))}),flush=True)
                del out,states;guard()
            correct=result['controls']['I4_original_ID'];wrong=result['controls']['I4_ID_plus1']
            result['identity_control']={'mean_source_KL_increase_from_I4_ID_permutation':wrong['mean_self_teacher_KL']-correct['mean_self_teacher_KL'],
                'original_I8_ID_plus1_mean_source_KL':parent['controls']['permutation_plus1']['mean_self_teacher_KL'],
                'scope':'same original source posterior/input/probability; one joint bank intervention, not count of useful IDs'}
            result['apparatus_gates']={'all_sources_parent_archives_fresh_exact':True,'all336_original_native_states_A16_full_heads_byte_exact':True,
                'all128_real_functions256_matrices_packing_full_inverse_saved_bank_byte_exact':True,'all_tiny_pairs_extrema_zero_minus8_and_I32_bounds_qualified':True,
                'all1344_I4_projection_I32_and_independent_I64_primal_outputs_exact':True,'same_source_input_probability_finalnorm_head':True,
                'all_full_vocab_KL_independent_identity_le1e_minus10':True,'removed_full_heads_match_original443_byte_exact':True,
                'one_fixed_encoding_no_calibration_fit_or_data_selection':True}
            result['feasibility_gates']={'mean_source_KL_le0_01':correct['mean_self_teacher_KL']<=.01,
                'every_book_mean_source_KL_le0_05':all(v['mean_KL']<=.05 for v in correct['per_book'].values()),
                'argmax_changed_fraction_le0_01':correct['argmax_changed_fraction']<=.01,
                'I4_ID_permutation_mean_KL_increase_ge0_01':result['identity_control']['mean_source_KL_increase_from_I4_ID_permutation']>=.01,
                'nominal_bank_storage_ratio_le0_51':result['conversion']['nominal_storage_ratio']<=.51}
        assert initial==(Path(artifact['payload']).stat().st_size,Path(artifact['payload']).stat().st_mtime_ns)
        result['output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.glob('*')) if p.is_file()]
        result['resource']={'seconds':time.monotonic()-start,'admission_seconds':numeric_start-start,'numeric_and_reporting_seconds':time.monotonic()-numeric_start,
            'peak_bytes':peak,'bytes_hashed':hashed,'output_bytes':sum(a['bytes'] for a in result['output_inventory']),'optimizer_updates':0}
        result['decision']='local_fixed_I4_promising_prepare_NEW_C_format_and_cost_gate' if all(result['feasibility_gates'].values()) else 'fixed_symmetric_row_I4_local_gate_FAIL_close_before_C_export_or_timing'
        result['scope']='All128 real finalbank11 functions encoded; 336 consumed original source128 teacher prefixes. Local Python arithmetic qualified against integer reference; no C I4 kernel/export/whole quantized model, changed upstream routing/generation/task quality, useful-n scaling, actual DRAM, SAME new artifact accepted rate, other family or100B proof.'
        guard();M.write(args.out,result)
        print(json.dumps({'apparatus':result['apparatus_gates'],'feasibility':result['feasibility_gates'],'decision':result['decision'],'resource':result['resource'],'sha256':M.digest(args.out)}),flush=True)
    except BaseException as error:
        result.update(failure_stage=stage,error=repr(error),seconds=time.monotonic()-start,peak_bytes=peak,bytes_hashed=hashed)
        if OUT.exists():result['partial_output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':M.digest(p)} for p in sorted(OUT.glob('*')) if p.is_file()]
        M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
