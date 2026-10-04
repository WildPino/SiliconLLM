"""Original source128 final-function counterfactuals before foreign readouts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits, threadpool_info
import meth431_switch_additive_pilot as X
M, R, G = X.M, X.R, X.G
PROTOCOL = M.DOC/'METH_442_SWITCH_NATIVE_CAUSALITY_PROTOCOL_20261004.md'
OUT = M.ROOT/'results/native_expert_scaling/meth442_switch_native_causality'
RECORDS = {
    'meth418_switch_function_capture_result.json': '4ebb37ed40d788eb85168d028b3e0b538cd7d65e009293cd32d5fb0eb117d829',
    'meth420_switch_function_gradient_result.failure.json': '17f60d043cff598cec0d3e85dfa4d23bdad67b83635422891e92729528056427',
    'meth441_switch_fixed_function_result.json': '78784886672e484b33f510bfbdce7cbbf5dfca1f1823a15d5790c7fccc32dcac'}


def log_probability(z):
    shifted=z.astype(np.float64)-np.max(z); return shifted-np.log(np.sum(np.exp(shifted)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();peak=0;hashed=0;stage='bindings';result={'experiment':'METH-442-original128-final-bank-native-function-identity-causality','controls':{}}
    def guard():
        nonlocal peak
        mi=psutil.Process().memory_info();peak=max(peak,mi.rss,getattr(mi,'peak_wset',0));size=sum(p.stat().st_size for p in OUT.glob('*') if p.is_file()) if OUT.exists() else 0
        assert peak<=3<<30 and time.monotonic()-start<=150 and size<=160<<20,'native_150sec_3GiB_160MiB'
    def sha(p):
        nonlocal hashed
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            while block:=f.read(4<<20):h.update(block);hashed+=len(block);guard()
        return h.hexdigest()
    try:
        result['helper_sha256']={}
        for p in (Path(__file__),PROTOCOL,Path(X.__file__),Path(M.__file__),Path(R.__file__),Path(R.C.__file__),Path(G.__file__)):
            M.committed(p);result['helper_sha256'][str(p)]=sha(p)
        records={};seen={};files={}
        for name,e in RECORDS.items():
            p=M.DOC/name;M.committed(p);assert sha(p)==e;records[name]=json.loads(p.read_text(encoding='utf-8'))
            for p,e in records[name]['helper_sha256'].items():
                if p in seen:assert seen[p]==e;continue
                M.committed(Path(p));assert sha(p)==e;seen[p]=e
            for a in records[name].get('output_inventory',[]):
                if a['path'] in files:assert files[a['path']]==a['sha256'];continue
                assert sha(a['path'])==a['sha256'];files[a['path']]=a['sha256']
        prior=records['meth418_switch_function_capture_result.json'];baseline=records['meth420_switch_function_gradient_result.failure.json'];parent=records['meth441_switch_fixed_function_result.json']
        assert all(prior['gates'].values()) and len(baseline['baselines'])==384 and all(parent['apparatus_gates'].values()) and all(list(parent['diagnostic_gates'].values())[:2])
        for a in baseline['baselines']:assert sha(a['archive_path'])==a['archive_sha256']
        for p,e in parent['preserved_original_binary_sha256'].items():assert sha(p)==e
        result['retained_record_sha256']=RECORDS;result['preserved_original_binary_sha256']=parent['preserved_original_binary_sha256']
        own={os.getpid(),*(p.pid for p in psutil.Process().parents())}
        for p in psutil.process_iter(['name','cmdline']):
            if p.pid in own:continue
            name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
            if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():result.setdefault('preserved_daemons',[]).append(p.pid);continue
            assert not(name.startswith('python') or ('meth' in name and name.endswith('.exe'))),('concurrent_job',p.pid,name)
        psutil.Process().cpu_affinity([0]);torch.set_num_threads(1);torch.set_num_interop_threads(1);assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6';assert psutil.disk_usage(str(M.ROOT)).free>=2<<30
        name,e=R.C.U.EXPORT[128];p=M.DOC/name;M.committed(p);assert sha(p)==e;export=json.loads(p.read_text(encoding='utf-8'));artifact=export['artifact'];assert artifact==parent['artifacts']['128']
        assert sha(artifact['payload'])==artifact['sha256'] and sha(artifact['manifest'])==artifact['manifest_sha256'];R.B.read_manifest(artifact['manifest'],export['original_config'],export['tensors'],Path(artifact['payload']))
        initial=(Path(artifact['payload']).stat().st_size,Path(artifact['payload']).stat().st_mtime_ns);mapped=np.memmap(artifact['payload'],dtype='u1',mode='r');entries=export['tensors'];result['artifacts']={'128':artifact}
        ff=R.C.tensor(mapped,entries,'decoder.block.11.layer.2.layer_norm.weight');fn=R.C.tensor(mapped,entries,'decoder.final_layer_norm.weight');head=G.I8Operator(R.C.tensor(mapped,entries,'lm_head.weight'),R.C.tensor(mapped,entries,'lm_head.weight','scales'));cache=X.ExpertCache(mapped,entries)
        cap={a['label']:a for a in prior['captures']};old={a['label']:a for a in baseline['baselines']};rows=[];scores=[];books=[];keys=[];logits=[]
        for b in range(18,24):
            for c in range(4):
                item=cap[f'teacher.n128.book{b}.case{c}'];a=old[item['label']];assert item['prospective_split']=='validation' and not a['qualification_only_not_paired_training'] and a['source_capture_sha256']==item['capture_sha256']
                data=Path(item['capture_path']).read_bytes();assert data[:8]==b'SWFUN001' and struct.unpack_from('<6I',data,8)==(768,3072,32128,128,11,1) and len(data)==32+14*R.C.DTYPE.itemsize
                z=np.frombuffer(data,dtype=R.C.DTYPE,offset=32).copy();assert np.array_equal(z['position'],np.arange(14)) and np.array_equal(z['id'],item['decoder_ids'])
                tr=Path(item['trace_path']).read_bytes();assert tr[:8]==b'SWRTA001' and struct.unpack_from('<2I',tr,8)==(128,768);dt=np.dtype([('index','<u4'),('phase','<u4'),('input','<f4',(768,)),('scores','<f4',(128,))]);t=np.frombuffer(tr,dtype=dt,offset=16)[179+6*np.arange(14)]
                R.C.exact(t['input'],z['input']);assert np.array_equal(np.argmax(t['scores'],axis=1),z['expert']);scores.append(t['scores'].copy());rows.append(z);books.extend([b]*14);keys.extend([f'book{b}.case{c}.position{j}:{item["pairing_sha256"]}' for j in range(14)])
                with np.load(a['archive_path'],allow_pickle=False) as q:
                    assert np.array_equal(q['source_ids'],item['source_ids']) and np.array_equal(q['decoder_ids'],item['decoder_ids']);logits.append(q['logits'].copy())
                    for field in ('input','up_raw','up','down','probability','post','final','head_input'):R.C.exact(q[field],z[field])
        rows=np.concatenate(rows);scores=np.concatenate(scores);books=np.asarray(books);original=np.concatenate(logits);assert original.shape==(336,32128) and len(rows)==len(keys)==336
        assert parent['data']['development_positions']==1008 and parent['data']['validation_positions']==336
        assert keys==parent['data']['keys'][1008:]
        OUT.mkdir(parents=True)
        with threadpool_limits(limits=1),torch.no_grad():
            result['runtime']={'torch':torch.__version__,'numpy':np.__version__,'CPU_affinity':[0],'BLAS':threadpool_info(),'GPU':False};stage='complete_original128_replay';stream=hashlib.sha256()
            for i,row in enumerate(rows):
                inp=G.NativeRMS.apply(torch.from_numpy(row['pre'].copy()),ff).numpy();R.C.exact(inp,row['input']);wi,wo=cache.get(int(row['expert']));raw=wi.native(inp);up=np.where(raw<0,np.float32(0),raw);down=wo.native(up)
                p=G.NativeProbability.apply(torch.from_numpy(scores[i]),int(row['expert'])).numpy();post=row['pre']+p*down;final=G.NativeRMS.apply(torch.from_numpy(post),fn).numpy();hi=final*np.float32(1/np.sqrt(768));out=head.native(hi)
                for field,value in (('up_raw',raw),('up',up),('down',down),('probability',p),('post',post),('final',final),('head_input',hi)):R.C.exact(value,row[field])
                for field,v in (('wi',inp),('wo',up),('head',hi)):
                    scale,codes=R.C.quant(v);R.C.exact(scale,row[field+'_scale']);R.C.exact(codes,row[field+'_codes'])
                R.C.exact(out,original[i]);stream.update(out.tobytes());guard()
            result['original_replay']={'positions':336,'complete_vocab_rows':336*32128,'all_native_states_A16_codes_scales_full_logits_byte_exact':True,'logit_stream_sha256':stream.hexdigest()}
            targets=np.stack([R.probability(v) for v in original]);oldlogp=np.stack([log_probability(v) for v in original]);entropy=np.asarray([R.numpy_loss(original[i].astype(np.float64),targets[i]) for i in range(336)])
            assert np.max(np.abs(entropy+np.sum(targets*oldlogp,axis=1)))<=1e-10;original_argmax=np.argmax(original,axis=1)
            result['data']={'keys':keys,'books':books.tolist(),'selected_ids':rows['expert'].tolist(),'selected_probability':rows['probability'].tolist(),'positions':rows['position'].tolist(),'native_original_probability_sha256':hashlib.sha256(targets.tobytes()).hexdigest(),'original_self_CE_mean':float(np.mean(entropy)),'original_self_CE':entropy.tolist()}
            for control in ('fixed0','permutation_plus1','removed'):
                stage=control;out=np.empty(original.shape,np.float32);state={k:np.empty((336,768),np.float32) for k in ('down','post','head_input')};effects={k:[] for k in ('up_raw','down','post','head_input')};ids=[]
                for i,row in enumerate(rows):
                    identity=0 if control=='fixed0' else (int(row['expert'])+1)%128 if control=='permutation_plus1' else -1;ids.append(identity)
                    if identity<0:raw=np.zeros(3072,np.float32);down=np.zeros(768,np.float32);post=row['pre'].copy()
                    else:
                        wi,wo=cache.get(identity);raw=wi.native(row['input']);down=wo.native(np.where(raw<0,np.float32(0),raw));post=row['pre']+row['probability']*down
                    final=G.NativeRMS.apply(torch.from_numpy(post),fn).numpy();hi=final*np.float32(1/np.sqrt(768));out[i]=head.native(hi)
                    for k,v in (('up_raw',raw),('down',down),('post',post),('head_input',hi)):effects[k].append(R.relative(v,row[k]))
                    for k,v in (('down',down),('post',post),('head_input',hi)):state[k][i]=v
                    guard()
                np.save(OUT/(control+'_logits.npy'),out);np.savez(OUT/(control+'_states.npz'),**state,actual_function_ids=np.asarray(ids))
                ce=np.asarray([R.numpy_loss(out[i].astype(np.float64),targets[i]) for i in range(336)]);kl=ce-entropy;independent=np.asarray([np.dot(targets[i],oldlogp[i]-log_probability(out[i])) for i in range(336)]);assert np.isfinite(ce).all() and np.min(kl)>=-1e-10 and np.max(np.abs(kl-independent))<=1e-10
                changed=np.argmax(out,axis=1)!=original_argmax;result['controls'][control]={'mean_self_teacher_KL':float(np.mean(kl)),'median_KL':float(np.median(kl)),'p95_KL':float(np.quantile(kl,.95)),'max_KL':float(np.max(kl)),'self_teacher_CE':float(np.mean(ce)),'argmax_changed_positions':int(np.sum(changed)),
                    'mean_KL_ge0_01':bool(np.mean(kl)>=.01),'per_position_KL':kl.tolist(),'per_position_CE':ce.tolist(),'independent_KL':independent.tolist(),'argmax_changed_mask':changed.tolist(),'actual_function_ids':ids,'relative_state_effects':effects,
                    'per_book':{str(b):{'mean_KL':float(np.mean(kl[books==b])),'argmax_changes':int(np.sum(changed[books==b]))} for b in range(18,24)},
                    'per_selected_id':{str(e):{'positions':int(np.sum(rows['expert']==e)),'mean_KL':float(np.mean(kl[rows['expert']==e]))} for e in np.unique(rows['expert'])}}
                del out,state;guard()
            result['apparatus_gates']={'sources_and_parent_archives_fresh_exact':True,'all336_native_original_states_quant_and_complete_heads_byte_exact':True,'same_original128_input_core_and_selected_probability':True,'fixed_three_controls_no_fit_or_ID_selection':True,'independent_KL_entropy_identity_same1e_minus10':True}
        assert initial==(Path(artifact['payload']).stat().st_size,Path(artifact['payload']).stat().st_mtime_ns)
        result['output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.glob('*')) if p.is_file()];result['resource']={'seconds':time.monotonic()-start,'peak_bytes':peak,'bytes_hashed':hashed,'output_bytes':sum(a['bytes'] for a in result['output_inventory']),'optimizer_updates':0}
        result['diagnostic_gates']={'primary_native_identity_mean_KL_ge0_01':result['controls']['permutation_plus1']['mean_KL_ge0_01'],'descriptive_fixed0_mean_KL_ge0_01':result['controls']['fixed0']['mean_KL_ge0_01'],'descriptive_removal_mean_KL_ge0_01':result['controls']['removed']['mean_KL_ge0_01']}
        result['decision']='native_identity_effect_present_foreign_transfer_preserves_little_identity_effect_prepare_NEW_source_sensitive_hypothesis' if result['controls']['permutation_plus1']['mean_KL_ge0_01'] else 'native_last_bank_identity_weak_in_this_pilot_prepare_broader_native_causal_capture_before_more_transfer_fits'
        result['scope']='Source128 own captured336 teacher prefixes/finalbank11, not foreign256 core/440 readout. Original native full-head replay exact420 then pointer0/pointer+1/removal at SAME original probability. Fixed-prefix downstream last-head counterfactual only, consumed split; no whole-model rollout/task-quality/generic-bank-irrelevance or n generalization/rate/LUT/DRAM/another-family/~100B proof. Closed retargeting recipes stay closed.'
        guard();M.write(args.out,result);print(json.dumps({'apparatus':result['apparatus_gates'],'effects':{k:{q:v for q,v in x.items() if q in ('mean_self_teacher_KL','argmax_changed_positions','mean_KL_ge0_01')} for k,x in result['controls'].items()},'decision':result['decision'],'resource':result['resource'],'sha256':M.digest(args.out)}),flush=True)
    except BaseException as error:
        result.update(failure_stage=stage,error=repr(error),seconds=time.monotonic()-start,peak_bytes=peak,bytes_hashed=hashed)
        if OUT.exists():result['partial_output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':M.digest(p)} for p in sorted(OUT.glob('*')) if p.is_file()]
        M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
