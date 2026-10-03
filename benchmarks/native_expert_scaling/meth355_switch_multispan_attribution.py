"""Consumed351 fixed-state attribution, including only comparable greedy prefixes."""
import os
os.environ.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import meth324_switch_reference as M
import meth328_switch_native_contract as B

BASELINE=M.DOC/'meth351_switch_multi_span_quality_result.json'
BASELINE_SHA='6859c57d40bfc32bd80be948d1c227381cef8e165d2ef3820b0c314053e2c306'
RECOVERED=M.DOC/'meth338_switch_tensor_recovery_result.json'
RECOVERED_SHA='19ef2f987444d17396beea8e489d2fee341d64b61cff00714521ed806b407a7c'
BOUND=M.DOC/'meth327_switch_tensor_binding_result.json'
BOUND_SHA='e72fc17b5527dc34df5c00ed7c44498b2ee4fc5d338ed8b130fff101a30c20a7'
PROTOCOL=M.DOC/'METH_355_SWITCH_MULTISPAN_ATTRIBUTION_PROTOCOL_20261004.md'
INPUT=M.ROOT/'results/native_expert_scaling/meth351_switch_multi_span_quality'
OUT=M.ROOT/'results/native_expert_scaling/meth355_switch_multispan_attribution'
MASK=np.array([1,2,4,5,7,8,10,11])
CONDITIONS=('actual_native','original_state_a16','native_state_source_f32','native_state_weight_only')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();maximum=0;stage='bindings'
    result={'experiment':'METH-355-consumed351-fixed-state-and-common-prefix-attribution','cases':[]}
    def guard():
        nonlocal maximum
        rss=psutil.Process().memory_info().rss;maximum=max(maximum,rss)
        assert rss<=8<<30 and time.monotonic()-start<=1200,'attribution_resource_guard'
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block:=stream.read(4<<20):h.update(block);guard()
        return h.hexdigest()
    def array_digest(array):return hashlib.sha256(memoryview(array).cast('B')).hexdigest()
    try:
        for path in (Path(__file__),PROTOCOL,BASELINE,RECOVERED,BOUND,Path(B.__file__),Path(M.__file__)):M.committed(path)
        assert digest(BASELINE)==BASELINE_SHA and digest(RECOVERED)==RECOVERED_SHA and digest(BOUND)==BOUND_SHA
        baseline=json.loads(BASELINE.read_text(encoding='utf-8'));recovered=json.loads(RECOVERED.read_text(encoding='utf-8'));bound=json.loads(BOUND.read_text(encoding='utf-8'))
        assert not baseline['gates']['masked_original_top1_agreement_ge0p95'] and not baseline['gates']['generation_prose_edit_to_original_upper95_le0p10']
        payload=Path(recovered['artifact']['payload']);before=(payload.stat().st_size,payload.stat().st_mtime_ns)
        assert before[0]==recovered['artifact']['bytes']
        mapped=np.memmap(payload,dtype=np.uint8,mode='r');shared=recovered['tensors']['shared.weight'];head=recovered['tensors']['lm_head.weight']
        original=np.ndarray(tuple(shared['shape']),dtype='<f4',buffer=mapped,offset=shared['offset'])
        codes=np.ndarray(tuple(head['shape']),dtype=np.int8,buffer=mapped,offset=head['offset']);scales=np.ndarray((head['shape'][0],),dtype='<f4',buffer=mapped,offset=head['scale_offset'])
        assert array_digest(original)==bound['tensors']['shared.weight']['sha256']==shared['sha256']
        assert array_digest(codes)==head['sha256'] and array_digest(scales)==head['scale_sha256']
        assert original.shape==codes.shape==(32128,768) and np.all(codes!=-128)
        torch.set_num_threads(1);weights=torch.from_numpy(original.copy());codes_i64=codes.astype(np.int64);codes_f64=codes.astype(np.float64);scale_f64=scales.astype(np.float64)
        factor=np.float32(1./np.sqrt(np.float64(768)))
        def a16(x):
            maximum=np.abs(x).max(-1,keepdims=True);xscale=maximum/np.float32(32767);xscale[maximum==0]=np.float32(1.)
            q=np.clip(np.rint(x/xscale),-32767,32767).astype(np.int64);dots=q@codes_i64.T
            assert int(np.abs(dots).max())<=768*127*32767
            return ((dots.astype(np.float64)*scale_f64[None,:])*xscale.astype(np.float64)).astype(np.float32)
        def source_head(x):
            with torch.no_grad():return np.stack([F.linear(torch.from_numpy(row.copy()).reshape(1,1,768),weights).numpy()[0,0].copy() for row in x])
        totals={scope:{'positions':0,'changed_decoder_route_choices':0,'decoder_routes':0,'conditions':{name:{'changed':0,'recovered':0,'introduced':0} for name in CONDITIONS}}
                for scope in ('teacher_all','teacher_mask','generation_common_prefix')}
        def score(scope,selection,conditions,gold,current,routes):
            g=gold.argmax(-1)[selection];n=current.argmax(-1)[selection];mismatch=n!=g;details={}
            for name,values in conditions.items():
                changed=values.argmax(-1)[selection]!=g
                entry={'positions':len(g),'changed':int(changed.sum()),'recovered':int(np.sum(mismatch&~changed)),
                       'introduced':int(np.sum(~mismatch&changed)),'greedy_ids':values.argmax(-1)[selection].tolist()}
                details[name]=entry
                for key in ('changed','recovered','introduced'):totals[scope]['conditions'][name][key]+=entry[key]
            totals[scope]['positions']+=len(g)
            totals[scope]['changed_decoder_route_choices']+=int(np.sum(routes[selection,:,0]!=routes[selection,:,1]))
            totals[scope]['decoder_routes']+=len(g)*6
            return details
        def inspect_arrays(stem,kind,original_path,native_path,record,expected_original,expected_native,prefix=None):
            assert digest(original_path)==expected_original and digest(native_path)==expected_native
            with np.load(original_path) as saved:oh=saved['decoder'][:,-1].copy();gold=saved['logits'].copy();orr=saved['routes'].copy()
            native=B.read_output(native_path);nh=native[1][:,-1].copy();current=native[2];nrr=native[3]
            if prefix is None:assert oh.shape==nh.shape==(14,768);length=14
            else:length=prefix;oh=oh[:length];nh=nh[:length];gold=gold[:length];current=current[:length]
            assert length>=1 and gold.shape==current.shape==(length,32128)
            ox=(oh*factor).astype(np.float32);nx=(nh*factor).astype(np.float32)
            oracle=source_head(ox);relative=B.relative(oracle,gold)
            assert relative<=1e-6 and np.array_equal(oracle.argmax(-1),gold.argmax(-1)),('original_head_oracle',stem,kind,relative)
            native_oracle=a16(nx);assert np.array_equal(native_oracle,current),('actual_native_head_oracle',stem,kind)
            conditions={'actual_native':current,'original_state_a16':a16(ox),'native_state_source_f32':source_head(nx),
                        'native_state_weight_only':((nx.astype(np.float64)@codes_f64.T)*scale_f64[None,:]).astype(np.float32)}
            # Both route streams encode six encoder banks, then six decoder routes per cached position.
            assert len(orr)>=174+length*6 and len(nrr)>=174+length*6
            dr=np.stack((orr[174:174+length*6,0].reshape(length,6),nrr[174:174+length*6,0].reshape(length,6)),axis=-1)
            if kind=='teacher':
                assert np.array_equal(gold.argmax(-1),record['original']['greedy_teacher_forced_ids']) and np.array_equal(current.argmax(-1),record['native']['greedy_teacher_forced_ids'])
                scores={'teacher_all':score('teacher_all',np.arange(length),conditions,gold,current,dr),
                        'teacher_mask':score('teacher_mask',MASK,conditions,gold,current,dr)}
            else:
                assert np.array_equal(gold.argmax(-1),record['generation']['original']['generated_ids'][:length]) and np.array_equal(current.argmax(-1),record['generation']['native']['generated_ids'][:length])
                scores={'generation_common_prefix':score('generation_common_prefix',np.arange(length),conditions,gold,current,dr)}
            arrays=OUT/(stem+'.'+kind+'.counterfactuals.npz');np.savez(arrays,**conditions)
            return {'positions':length,'source_head_oracle_relative_l2':relative,'source_head_oracle_choices_exact':True,'native_a16_oracle_exact':True,
                    'final_norm_relative_l2':B.relative(nh,oh),'changed_encoder_route_choices':int(np.sum(orr[:174,0]!=nrr[:174,0])),
                    'changed_decoder_route_choices':int(np.sum(dr[:,:,0]!=dr[:,:,1])),'scores':scores,'arrays_sha256':digest(arrays)}
        OUT.mkdir(parents=True)
        result.update({'controller_sha256':digest(__file__),'protocol_sha256':digest(PROTOCOL),'baseline351_sha256':BASELINE_SHA,
                       'recovered338_sha256':RECOVERED_SHA,'original327_sha256':BOUND_SHA,'head_array_identities_fresh':True,
                       'payload_identity_policy':'Fresh exact head codes/scales and retained original F32 shared coefficients; whole338 qualified identity reused, size/mtime fixed; no fresh full payload/ZIP hash.',
                       'conditions':list(CONDITIONS)})
        for bi,book in enumerate(baseline['books']):
            for ci,record in enumerate(book['cases']):
                stage=f'book{bi}_case{ci}';stem=f'book{bi}.case{ci}';gen=record['generation'];oid=gen['original']['generated_ids'];nid=gen['native']['generated_ids']
                divergences=[i for i,(a,b) in enumerate(zip(oid,nid)) if a!=b];first=divergences[0] if divergences else None
                assert first is not None or oid==nid,'different_terminal_length_without_first_choice_divergence'
                length=first+1 if first is not None else len(oid)
                teacher=inspect_arrays(stem,'teacher',INPUT/(stem+'.original_reference.npz'),INPUT/(stem+'.0.bin'),record,record['original_reference_sha256'],record['native_output_sha256'])
                generation=inspect_arrays(stem,'generation',INPUT/(stem+'.original_generation.npz'),INPUT/(stem+'.generation.0.bin'),record,gen['original_generation_sha256'],gen['native_generation_sha256'],length)
                first_details={}
                if first is not None:
                    with np.load(OUT/(stem+'.generation.counterfactuals.npz')) as saved:
                        for name in CONDITIONS:first_details[name]={'choice':int(saved[name][first].argmax()),'matches_original':bool(saved[name][first].argmax()==oid[first])}
                result['cases'].append({'book':bi,'case':ci,'source_id':book['source_id'],'first_generation_divergence':first,
                                        'first_divergence_head_conditions':first_details,'teacher':teacher,'generation_common_prefix':generation})
                guard()
            print(json.dumps({'book':bi,'cases_complete':len(result['cases']),'seconds':time.monotonic()-start}),flush=True)
        assert len(result['cases'])==96 and before==(payload.stat().st_size,payload.stat().st_mtime_ns)
        assert totals['teacher_all']['positions']==1344 and totals['teacher_mask']['positions']==768
        assert totals['teacher_all']['conditions']['actual_native']['changed']==51 and totals['teacher_mask']['conditions']['actual_native']['changed']==40
        for scope,aggregate in totals.items():
            for entry in aggregate['conditions'].values():entry['agreement']=1-entry['changed']/aggregate['positions']
        divergent=[c for c in result['cases'] if c['first_generation_divergence'] is not None]
        result['aggregate']=totals
        result['first_generation_divergence']={'cases':len(divergent),'conditions':{name:{'matches_original':sum(c['first_divergence_head_conditions'][name]['matches_original'] for c in divergent)} for name in CONDITIONS}}
        result['gates']={'all96_inputs_exact351':True,'all192_original_head_oracles_closed':True,'all192_actual_native_a16_oracles_exact':True,'observed351_teacher_counts_reproduced':True}
        result['resource']={'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_rss_bytes':maximum,'end_rss_bytes':psutil.Process().memory_info().rss}
        result['decision']='diagnostic_only_select_precision_variable_then_full_numeric_cost_and_NEW_quality'
        result['scope']='Consumed351 fixed teacher states and natural common decoder-input prefixes INCLUDING first different choice; no comparison after branch histories diverge. Fixed-state head interventions are descriptive, not whole-model causal or quality/rate/n-scaling proof. All351 failures remain.'
        guard();M.write(args.out,result);print(json.dumps({'sha256':digest(args.out),'gates':result['gates'],'aggregate':totals,'first_generation_divergence':result['first_generation_divergence'],'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'stage':stage,'error':repr(error),'seconds':time.monotonic()-start,'maximum_checked_rss_bytes':maximum});M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
