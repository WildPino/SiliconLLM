"""Post-failure fixed-state head attribution; no changed whole-model qualification."""
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
import meth343_switch_fresh_prediction as P

PROTOCOL=M.DOC/'METH_344_SWITCH_HEAD_ATTRIBUTION_PROTOCOL_20261003.md'
BASELINE=M.DOC/'meth343_switch_fresh_prediction_result.json'
BASELINE_SHA='a4838903958be5f6bb567c1dedd802a17e6b27e2d97c4f490210139f296cbadf'
OUT=M.ROOT/'results/native_expert_scaling/meth344_switch_head_attribution'
CONDITIONS=('original_a8','original_a16','original_weight_only','original_f32_f64_dot',
            'native_a16','native_weight_only','native_f32_f64_dot')


def quantize(x,magnitude):
    maximum=np.abs(x).max(-1,keepdims=True);scales=maximum/np.float32(magnitude);scales[maximum==0]=np.float32(1.)
    assert np.isfinite(x).all() and np.isfinite(scales).all() and (scales>0).all()
    codes=np.clip(np.rint(x/scales),-magnitude,magnitude).astype(np.int64)
    return codes,scales


def primitive():
    entries=[]
    for cols in (8,768,4096):
        x=np.full((1,cols),32767,dtype=np.float32);x[0,1:7]=[.5,1.5,-.5,-1.5,2.5,-2.5]
        weights=np.full((2,cols),127,dtype=np.int64);weights[1,::2]=-127
        codes,scale=quantize(x,32767);actual=codes@weights.T
        expected=np.array([[sum(int(a)*int(b) for a,b in zip(codes[0],row)) for row in weights]],dtype=np.int64)
        assert np.array_equal(actual,expected) and codes[0,1:7].tolist()==[0,2,0,-2,2,-2]
        entries.append({'cols':cols,'exact_scalar_i64':True,'max_dot':int(np.abs(actual).max()),'scale':float(scale[0,0])})
    extreme=4096*127*32767;assert extreme==17045131264
    return {'cases':entries,'all4096_extreme_bound':extreme,'passed':True}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();stage='bindings';maximum=0
    result={'experiment':'METH-344-consumed-fixed-state-head-attribution','cases':[]}
    def guard():
        nonlocal maximum
        rss=psutil.Process().memory_info().rss;maximum=max(maximum,rss)
        assert rss<=8<<30 and time.monotonic()-start<=1200,'head_attribution_resource_guard'
    def file_digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block:=stream.read(4<<20):h.update(block);guard()
        return h.hexdigest()
    try:
        for path in (Path(__file__),PROTOCOL,BASELINE,P.MANIFEST,P.RECOVERED,P.BOUND,Path(P.__file__),Path(P.B.__file__),Path(M.__file__)):M.committed(path)
        assert M.digest(BASELINE)==BASELINE_SHA and M.digest(P.MANIFEST)==P.MANIFEST_SHA and M.digest(P.RECOVERED)==P.RECOVERED_SHA and M.digest(P.BOUND)==P.BOUND_SHA
        baseline=json.loads(BASELINE.read_text(encoding='utf-8'));cohort=json.loads(P.MANIFEST.read_text(encoding='utf-8'));recovered=json.loads(P.RECOVERED.read_text(encoding='utf-8'));bound=json.loads(P.BOUND.read_text(encoding='utf-8'))
        assert not baseline['gates']['original_top1_agreement_ge0p95'] and len(baseline['books'])==24
        payload=Path(recovered['artifact']['payload']);assert file_digest(payload)==recovered['artifact']['sha256']
        mapped=np.memmap(payload,dtype=np.uint8,mode='r');shared=recovered['tensors']['shared.weight'];head=recovered['tensors']['lm_head.weight']
        original=np.ndarray(tuple(shared['shape']),dtype='<f4',buffer=mapped,offset=shared['offset'])
        codes=np.ndarray(tuple(head['shape']),dtype=np.int8,buffer=mapped,offset=head['offset']);scales=np.ndarray((head['shape'][0],),dtype='<f4',buffer=mapped,offset=head['scale_offset'])
        for array,expected in ((original,bound['tensors']['shared.weight']['sha256']),(codes,head['sha256']),(scales,head['scale_sha256'])):
            assert hashlib.sha256(memoryview(array).cast('B')).hexdigest()==expected
        assert original.shape==codes.shape==(32128,768) and np.all(codes!=-128)
        torch.set_num_threads(1);weights=torch.from_numpy(original.copy());head64=original.astype(np.float64);codes64=codes.astype(np.float64);codes_i64=codes.astype(np.int64);scale64=scales.astype(np.float64)
        result.update({'controller_sha256':M.digest(__file__),'protocol_sha256':M.digest(PROTOCOL),'baseline343_sha256':BASELINE_SHA,
                       'manifest342_sha256':P.MANIFEST_SHA,'recovered338_sha256':P.RECOVERED_SHA,'original_head_from_exact_retained_f32_lookup':True,
                       'primitive':primitive(),'conditions':list(CONDITIONS)})
        OUT.mkdir(parents=True);factor=np.float32(1./np.sqrt(np.float64(768)))
        def integer_head(x,magnitude):
            quantized,xscale=quantize(x,magnitude);dots=quantized@codes_i64.T
            assert int(np.abs(dots).max())<=768*127*magnitude
            return ((dots.astype(np.float64)*scale64[None,:])*xscale.astype(np.float64)).astype(np.float32)
        aggregates={name:{'changed_to_original':0,'nll_delta':[],'span_nll_delta':[],'original_relative_l2':[],
                          'recovered_native_mismatches':0,'introduced_native_matches':0} for name in CONDITIONS}
        stage='all96_fixed_original_native_states'
        for book_index,book in enumerate(baseline['books']):
            for case_index,record in enumerate(book['cases']):
                stem=f'book{book_index}.case{case_index}';path=P.OUT/(stem+'.original_reference.npz');native_path=P.OUT/(stem+'.0.bin')
                assert file_digest(path)==record['original_reference_sha256'] and file_digest(native_path)==record['native_output_sha256']
                with np.load(path) as saved:original_h=saved['decoder'][:,-1].copy();gold=saved['logits'].copy()
                actual=P.B.read_output(native_path);native_h=actual[1][:,-1].copy();current=actual[2]
                ox=(original_h*factor).astype(np.float32);nx=(native_h*factor).astype(np.float32)
                with torch.no_grad():source_oracle=np.stack([F.linear(torch.from_numpy(row.copy()).reshape(1,1,768),weights).numpy()[0,0].copy() for row in ox])
                source_error=P.B.relative(source_oracle,gold);source_choices=bool(np.array_equal(source_oracle.argmax(-1),gold.argmax(-1)))
                assert source_error<=1e-6 and source_choices,('original_head_oracle',book_index,case_index,source_error)
                native_oracle=integer_head(nx,127);native_exact=bool(np.array_equal(native_oracle,current));assert native_exact,'actual_native_i8_head_oracle'
                candidates={'original_a8':integer_head(ox,127),'original_a16':integer_head(ox,32767),
                            'original_weight_only':((ox.astype(np.float64)@codes64.T)*scale64).astype(np.float32),
                            'original_f32_f64_dot':(ox.astype(np.float64)@head64.T).astype(np.float32),
                            'native_a16':integer_head(nx,32767),'native_weight_only':((nx.astype(np.float64)@codes64.T)*scale64).astype(np.float32),
                            'native_f32_f64_dot':(nx.astype(np.float64)@head64.T).astype(np.float32)}
                expected=cohort['items'][book_index]['cases'][case_index];original_choices=gold.argmax(-1);native_choices=current.argmax(-1)
                current_mismatch=native_choices!=original_choices;details={}
                for name,values in candidates.items():
                    choice=values.argmax(-1);changed=choice!=original_choices;entry=P.metrics(values,expected['target_ids'],expected['masked_span_ids'])
                    delta=entry['mean_nll']-record['original']['mean_nll'];span_delta=entry['mean_span_nll']-record['original']['mean_span_nll']
                    recover=int(np.sum(current_mismatch&~changed));introduce=int(np.sum(~current_mismatch&changed));error=P.B.relative(values,gold)
                    details[name]={'changed_to_original':int(changed.sum()),'recovered_native_mismatches':recover,'introduced_native_matches':introduce,
                                   'nll_delta':delta,'span_nll_delta':span_delta,'original_relative_l2':error,
                                   'logits_sha256':hashlib.sha256(values.tobytes()).hexdigest()}
                    a=aggregates[name];a['changed_to_original']+=int(changed.sum());a['recovered_native_mismatches']+=recover;a['introduced_native_matches']+=introduce
                    for key,value in (('nll_delta',delta),('span_nll_delta',span_delta),('original_relative_l2',error)):a[key].append(value)
                arrays=OUT/(stem+'.head_counterfactuals.npz');np.savez(arrays,**candidates)
                result['cases'].append({'book':book_index,'case':case_index,'source_head_oracle_relative_l2':source_error,'source_head_oracle_choices_exact':source_choices,
                                        'native_i8_head_oracle_exact':native_exact,'actual_native_changed_to_original':int(current_mismatch.sum()),
                                        'original_native_final_norm_relative_l2':P.B.relative(native_h,original_h),
                                        'native_final_norm_activation_maxabs':float(np.abs(native_h).max()),'conditions':details,'arrays_sha256':M.digest(arrays)})
                guard()
            print(json.dumps({'book':book_index,'cases_complete':len(result['cases']),'seconds':time.monotonic()-start}),flush=True)
        for name,a in aggregates.items():
            a['agreement_to_original']=1-a['changed_to_original']/1056
            for key in ('nll_delta','span_nll_delta','original_relative_l2'):a['mean_'+key]=float(np.mean(a.pop(key)))
        result['aggregate']=aggregates;result['gates']={'all96_original_head_oracle_closed':True,'all96_current_native_head_exact':True,'a16_scalar_primitive_exact':result['primitive']['passed']}
        result['resource']={'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_rss_bytes':maximum,'end_rss_bytes':psutil.Process().memory_info().rss}
        result['decision']='diagnostic_only_select_new_smallest_precision_variable_with_separate_numeric_cost_and_NEW_quality_protocol'
        result['scope']='All96 now-consumed343 fixed states. Head-only counterfactuals cannot substitute for actual upstream-changed A16 model, full native implementation, untouched quality, generation/task or accepted rate. Original34395% gate remains failed; no candidate promotion or significance from these descriptive means.'
        guard();M.write(args.out,result);print(json.dumps({'sha256':M.digest(args.out),'gates':result['gates'],'aggregate':aggregates,'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'stage':stage,'error':repr(error),'seconds':time.monotonic()-start,'maximum_checked_rss_bytes':maximum});M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
