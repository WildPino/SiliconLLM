"""Frozen440 readout with one source function and fixed input-alignment control."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits, threadpool_info
import meth440_switch_shared_readout_pilot as Q
X, M, R, G, H = Q.X, Q.M, Q.R, Q.G, Q.H
PROTOCOL = M.DOC/'METH_441_SWITCH_FIXED_FUNCTION_PROTOCOL_20261004.md'
OUT = M.ROOT/'results/native_expert_scaling/meth441_switch_fixed_function'
PRIOR = M.DOC/'meth440_switch_shared_readout_result.json'
PRIOR_SHA = '77eb21bb21a0c364ed30f469d370efdd8046baea51a0da2509de164754b1a455'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();peak=0;hashed=0;stage='bindings';result={'experiment':'METH-441-single-frozen-function-and-input-alignment-diagnostic','controls':{}}
    def guard():
        nonlocal peak
        mi=psutil.Process().memory_info();peak=max(peak,mi.rss,getattr(mi,'peak_wset',0));size=sum(p.stat().st_size for p in OUT.glob('*') if p.is_file()) if OUT.exists() else 0
        assert peak<=4<<30 and time.monotonic()-start<=150 and size<=96<<20,'diagnostic_150sec_4GiB_96MiB'
    def sha(p):
        nonlocal hashed
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            while block:=f.read(4<<20):h.update(block);hashed+=len(block);guard()
        return h.hexdigest()
    try:
        result['helper_sha256']={}
        for p in (Path(__file__),PROTOCOL,Path(Q.__file__),Path(H.__file__),Path(H.H.__file__),Path(H.P.__file__),Path(M.__file__),Path(R.__file__),Path(R.C.__file__),Path(G.__file__)):
            M.committed(p);result['helper_sha256'][str(p)]=sha(p)
        M.committed(PRIOR);assert sha(PRIOR)==PRIOR_SHA;parent=json.loads(PRIOR.read_text(encoding='utf-8'));assert all(parent['apparatus_gates'].values()) and len(parent['updates'])==192 and sum(parent['potential_gates_NOT_deployable_capacity'].values())==6
        for p,e in parent['helper_sha256'].items():M.committed(Path(p));assert sha(p)==e
        for a in parent['output_inventory']:assert sha(a['path'])==a['sha256']
        records={}
        for name in ('meth418_switch_function_capture_result.json','meth420_switch_function_gradient_result.failure.json'):
            p=M.DOC/name;M.committed(p);assert sha(p)==parent['retained_record_sha256'][name];records[name]=json.loads(p.read_text(encoding='utf-8'))
        prior,baseline=records.values()
        for a in prior['output_inventory']:assert sha(a['path'])==a['sha256']
        for a in baseline['baselines']:assert sha(a['archive_path'])==a['archive_sha256']
        for p,e in parent['preserved_original_binary_sha256'].items():assert sha(p)==e
        result['preserved_original_binary_sha256']=parent['preserved_original_binary_sha256'];result['retained_record_sha256']={PRIOR.name:PRIOR_SHA}|{n:parent['retained_record_sha256'][n] for n in records}
        own={os.getpid(),*(p.pid for p in psutil.Process().parents())}
        for p in psutil.process_iter(['name','cmdline']):
            if p.pid in own:continue
            name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
            if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():result.setdefault('preserved_daemons',[]).append(p.pid);continue
            assert not(name.startswith('python') or ('meth' in name and name.endswith('.exe'))),('concurrent_job',p.pid,name)
        psutil.Process().cpu_affinity([0]);torch.set_num_threads(1);torch.set_num_interop_threads(1);assert psutil.disk_usage(str(M.ROOT)).free>=2<<30
        mapped={};entries={};initial={}
        for n,(name,e) in R.C.U.EXPORT.items():
            p=M.DOC/name;M.committed(p);assert sha(p)==e;export=json.loads(p.read_text(encoding='utf-8'));a=export['artifact'];assert a==parent['artifacts'][str(n)]
            assert sha(a['payload'])==a['sha256'] and sha(a['manifest'])==a['manifest_sha256'];R.B.read_manifest(a['manifest'],export['original_config'],export['tensors'],Path(a['payload']))
            mapped[n]=np.memmap(a['payload'],dtype='u1',mode='r');entries[n]=export['tensors'];initial[n]=(Path(a['payload']).stat().st_size,Path(a['payload']).stat().st_mtime_ns)
        result['artifacts']=parent['artifacts'];OUT.mkdir(parents=True)
        with threadpool_limits(limits=1),torch.no_grad():
            result['runtime']={'torch':torch.__version__,'numpy':np.__version__,'CPU_affinity':[0],'BLAS':threadpool_info(),'GPU':False};stage='paired_data';data=X.load_pairs(prior,baseline,guard);assert data['keys']==parent['data']['keys'];result['data']=parent['data']
            for n in (128,256):assert hashlib.sha256(data[f'p{n}'].tobytes()).hexdigest()==parent['data'][f'paired_probability{n}_sha256']
            cap={a['label']:a for a in prior['captures']};rows=np.concatenate([np.frombuffer(Path(cap[f'teacher.n128.book{b}.case{c}']['capture_path']).read_bytes(),dtype=R.C.DTYPE,offset=32).copy() for b in range(24) for c in range(4)]);assert np.array_equal(rows['expert'],data['labels'])
            checkpoint=next(a for a in parent['output_inventory'] if Path(a['path']).name=='shared_readout_checkpoint.npz');result['frozen_checkpoint']=checkpoint
            with np.load(checkpoint['path'],allow_pickle=False) as z:params={k:torch.from_numpy(z['real_'+k].copy()) for k in ('C','D')};buffers=tuple(z['real_'+k].copy() for k in ('mean','std'))
            assert all(not v.requires_grad for v in params.values());fn=R.C.tensor(mapped[256],entries[256],'decoder.final_layer_norm.weight');head=G.I8Operator(R.C.tensor(mapped[256],entries[256],'lm_head.weight'),R.C.tensor(mapped[256],entries[256],'lm_head.weight','scales'));cache=X.ExpertCache(mapped[128],entries[128]);wi,wo=cache.get(0)
            val=data['val'];mask=np.asarray(parent['real_oracle_validation']['oracle_mask'],bool);old=X.losses(data['logits'][val],val,data);assert X.summarize(old)==parent['no_added_validation'];selected_losses=np.asarray(parent['real_oracle_validation']['per_position_losses']);proof_count=0
            for control,order in (('fixed0_correct_input',val),('fixed0_rotated_input168',np.roll(val,-168))):
                stage=control;out=np.empty((336,32128),np.float32)
                for j,(i,input_i) in enumerate(zip(val,order)):
                    inp=rows['input'][input_i];raw=wi.native(inp);up=np.where(raw<0,np.float32(0),raw);feature=wo.native(up)
                    if control=='fixed0_correct_input':
                        if j==0:
                            prefix='decoder.block.11.layer.2.mlp.experts.expert_0.';scale,codes=R.C.quant(inp);R.C.exact(raw,R.C.matrix(mapped[128],entries[128],prefix+'wi.weight',codes,scale));scale,codes=R.C.quant(up);R.C.exact(feature,R.C.matrix(mapped[128],entries[128],prefix+'wo.weight',codes,scale))
                        if int(data['labels'][i])==0:R.C.exact(raw,rows['up_raw'][i]);R.C.exact(up,rows['up'][i]);R.C.exact(feature,rows['down'][i]);proof_count+=1
                    out[j]=H.forward(torch.from_numpy(feature),torch.from_numpy(data['base'][i]),torch.from_numpy(data['scores'][i]),int(data['old_ids'][i]),buffers,fn,head,params)['logits'].numpy();guard()
                np.save(OUT/(control+'.npy'),out);loss=X.losses(out,val,data);del out;hard=np.where(mask[:,None],loss,old);v=X.summarize(hard)
                result['controls'][control]=v|{'per_position_forced_losses':loss.tolist(),'per_position_same_mask_losses':hard.tolist(),'same_original_real_mask_positions':int(np.sum(mask)),'actual_added_distinct_functions':1,'relative_to_selected_mixture_CE_delta':v['equal_mixture_CE']-parent['real_oracle_validation']['equal_mixture_CE'],'input_positions':order.tolist()}
            fixed=result['controls']['fixed0_correct_input'];rotated=result['controls']['fixed0_rotated_input168'];gain=parent['no_added_validation']['equal_mixture_CE']-parent['real_oracle_validation']['equal_mixture_CE'];remaining=parent['no_added_validation']['equal_mixture_CE']-fixed['equal_mixture_CE']
            result['diagnostic_gates']={'single_fixed0_keeps_at_least90percent_selected_mixture_benefit':remaining>=.9*gain,'single_fixed0_selected_gap_at_most0_001':fixed['relative_to_selected_mixture_CE_delta']<=.001,'rotating_unavailable_input_harms_fixed0_by0_01':rotated['equal_mixture_CE']-fixed['equal_mixture_CE']>=.01}
            result['apparatus_gates']={'source_and_parent_archives_fresh_exact':True,'paired_probabilities_keys_original_baseline_exact440':True,'fixed_source0_complete_first_WI_WO_integer_replay_exact':True,'same_frozen_shared_C_D_stats_probability_base_head_and_mask':True,'zero_updates_no_threshold_or_ID_choice':True};result['source0_capture_replay_positions']=proof_count;result['selected_original_real_oracle']=parent['real_oracle_validation'];result['no_added_validation']=parent['no_added_validation']
        assert sha(checkpoint['path'])==checkpoint['sha256']
        for n,a in ((n,parent['artifacts'][str(n)]) for n in (128,256)):assert initial[n]==(Path(a['payload']).stat().st_size,Path(a['payload']).stat().st_mtime_ns)
        result['output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.glob('*.npy')];result['resource']={'seconds':time.monotonic()-start,'peak_bytes':peak,'bytes_hashed':hashed,'output_bytes':sum(a['bytes'] for a in result['output_inventory']),'optimizer_updates':0}
        result['decision']='single_function_retains_benefit_do_not_interpret_teacher_ID_counts_as_useful_conditional_capacity' if all(list(result['diagnostic_gates'].values())[:2]) else 'fixed0_changes_benefit_but_original440_all9_fail_no_capacity_or_selector_fit'
        result['scope']='Frozen440/consumed336val. Added branch has ONE actual source128 function0 for all89 original oracle-mask consultations, no function chosen on outcomes. Input rotation168 shifts books3 while retaining case/token position. Both need unavailable128 core inputs, masks target-aware. Diagnostic only, not full all-single-function bound/quality/native rate/n generalization.440 fixed recipe stays closed.'
        guard();M.write(args.out,result);print(json.dumps({'apparatus':result['apparatus_gates'],'diagnostic':result['diagnostic_gates'],'controls':{k:{q:v for q,v in x.items() if q in ('equal_mixture_CE','relative_to_selected_mixture_CE_delta')} for k,x in result['controls'].items()},'decision':result['decision'],'resource':result['resource'],'sha256':M.digest(args.out)}),flush=True)
    except BaseException as error:
        result.update(failure_stage=stage,error=repr(error),seconds=time.monotonic()-start,peak_bytes=peak,bytes_hashed=hashed)
        if OUT.exists():result['partial_output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':M.digest(p)} for p in OUT.glob('*.npy')]
        M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
