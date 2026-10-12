"""All exported source maps and stored shadow behavior, without model replay."""
import argparse,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,tensor_reader
from chatbot_falcon_usability import write,normalized
START=time.monotonic()


def main(a):
    import numpy as np,torch,psutil
    from original_engine_chat import ChatTokenizer
    assert extent(a.report)['sha256']==a.report_sha and extent(a.holder)['sha256']==a.holder_sha
    r=json.loads(a.report.read_bytes());holder=json.loads(a.holder.read_bytes())
    assert r['schema']=='FALCON_PRETRAINED_PROJECTED_SHADOW_RESULT_V3'
    assert holder['exit_code']==0 and holder['error'] is None and holder['result']==extent(a.report)
    assert extent(holder['worker_terminal']['path'])==holder['worker_terminal']
    terminal=json.loads(Path(holder['worker_terminal']['path']).read_bytes())
    assert terminal['exit_code']==0 and terminal['error'] is None and terminal['result']==extent(a.report)
    assert extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes())
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6)
    inputs=[extent(__file__),extent(a.report),extent(a.holder),holder['worker_terminal'],r['binding'],*b['inputs'],*r['outputs']]
    def guard():
        assert time.monotonic()-START<=900,'audit deadline'
        assert proc.memory_info().peak_wset<=4<<30,'OS cap'
        assert not proc.children(recursive=True),'unexpected child'
    def check():
        for item in inputs:guard();assert extent(item['path'])==item,item['path']
    check();state=torch.load(r['model']['path'],map_location='cpu',weights_only=True,mmap=True)
    assert set(state)=={row['name'] for row in b['fields']}|{'projected_metric'}
    p=np.fromfile(b['basis']['path'],'<f4').reshape(2048,256).astype('f8');G=p.T@p
    assert np.array_equal(state['projected_metric'].numpy(),G.astype('f4'))
    getter,stream=tensor_reader(Path(b['source'])/'model.safetensors',np);maps=[]
    def compare(actual,expected,copy=False):
        assert actual.shape==expected.shape and actual.dtype==np.dtype('float32') and np.isfinite(actual).all()
        if copy:assert np.array_equal(actual,expected.astype('f4'));return 0.
        value=float(np.max(np.abs(actual.astype('f8')-expected))/max(float(np.max(np.abs(expected))),1e-300))
        assert value<=1e-5,value;return value
    try:
        for row in b['fields']:
            guard();key=row['name'];actual=state[key].numpy();assert list(actual.shape)==row['target_shape']
            if key in ('model.embed_tokens.weight','lm_head.weight'):
                gamma=getter('model.final_layernorm.weight') if key=='lm_head.weight' else None
                error=0
                for lo in range(0,65537,4096):
                    guard();hi=min(lo+4096,65537);source=getter(key,list(range(lo,hi)))
                    if gamma is not None:source=source*gamma
                    expected=source@p
                    if gamma is not None:expected=expected*math.sqrt(8)
                    error=max(error,compare(actual[lo:hi],expected))
            elif key=='model.final_layernorm.weight' or key.endswith(('input_layernorm.weight','pre_ff_layernorm.weight')):
                assert np.array_equal(actual,np.ones(256,dtype='f4'));error=0
            elif key.endswith(('mamba.in_proj.weight','self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight',
                               'feed_forward.gate_proj.weight','feed_forward.up_proj.weight')):
                prefix=key.split('.mamba.')[0] if '.mamba.' in key else key.split('.self_attn.')[0] if '.self_attn.' in key else key.split('.feed_forward.')[0]
                norm='pre_ff_layernorm' if '.feed_forward.' in key else 'input_layernorm'
                gamma=getter(prefix+'.'+norm+'.weight');source=getter(key)
                expected=(source@(gamma[:,None]*p))*math.sqrt(8);error=compare(actual,expected)
            elif key.endswith(('mamba.out_proj.weight','self_attn.o_proj.weight','feed_forward.down_proj.weight')):
                source=getter(key);expected=p.T@source;error=compare(actual,expected)
            else:
                source=getter(key);error=compare(actual,source,copy=True)
            maps.append(dict(name=key,sup_relative=error));
            if len(maps)%50==0:print(json.dumps(dict(fields=len(maps),seconds=time.monotonic()-START)),flush=True)
        stream.close();stream=None
        tokenizer=ChatTokenizer(b['source']);special=set(b['special_ids'])-set(b['eos_ids']);correct=0
        for case,row in zip(b['cases'],r['tasks']):
            guard();assert row['id']==case['id'] and row['expected']==case['expected']
            assert tokenizer.encode(row['messages'])==row['input_ids']==case['input_ids']
            scores=np.memmap(row['scores']['path'],'<f4','r',shape=(len(row['output_ids']),65537))
            assert np.isfinite(scores).all() and scores.argmax(-1).tolist()==row['output_ids']
            answer=tokenizer.decode(row['output_ids']);assert answer==row['output_text']
            assert normalized(answer)==row['normalized'];assert row['correct']==(normalized(answer)==case['expected'])
            assert row['special_leak']==any(i in special for i in row['output_ids']);correct+=row['correct']
            stopped=bool(row['output_ids']) and len(row['output_ids'])<=64 and not any(i in b['eos_ids'] for i in row['output_ids'][:-1]) and (row['output_ids'][-1] in b['eos_ids'] or len(row['output_ids'])==64)
            assert row['stop_policy']==stopped
        follow=r['own_answer_followup'];initial=next(row for row in r['tasks'] if row['id']==b['followup']['case_id'])
        expected_messages=initial['messages']+[dict(role='assistant',content=initial['output_text']),dict(role='user',content=b['followup']['prompt'])]
        assert follow['messages']==expected_messages and tokenizer.encode(expected_messages)==follow['input_ids']
        follow_scores=np.memmap(follow['scores']['path'],'<f4','r',shape=(len(follow['output_ids']),65537))
        assert np.isfinite(follow_scores).all() and follow_scores.argmax(-1).tolist()==follow['output_ids']
        assert follow['output_text']==tokenizer.decode(follow['output_ids'])
        assert follow['correct']==(normalized(follow['output_text'])==b['followup']['expected'])
        assert correct==r['correct'] and len(r['tasks'])==16
        categories={cat:sum(row['correct'] for row in r['tasks'] if row['category']==cat) for cat in sorted({row['category'] for row in r['tasks']})}
        quality=dict(correct_min=correct>=12,every_category_min=all(value>=2 for value in categories.values()),
            blank_max=sum(not row['output_text'].strip() for row in r['tasks'])<=1,
            no_special_leak=not any(row['special_leak'] for row in r['tasks']),stop_policy=all(row['stop_policy'] for row in r['tasks']))
        assert categories==r['categories'] and quality==r['quality_gates']
        max_delta=0;labels=0;dev=[]
        for rec,row in zip(b['DEV'],r['DEV']):
            guard();assert rec['id']==row['id'];n=len(rec['positions']);labels+=n
            score=np.memmap(row['scores']['path'],'<f4','r',shape=(n,65537))
            teacher=np.memmap(rec['logits']['path'],'<u2','r',shape=(n,65537));loss=[];wrong=0
            for i in range(n):
                q=(teacher[i].astype('<u4')<<16).view('<f4').astype('f8');q-=q.max()
                qq=np.exp(q);q-=math.log(math.fsum(map(float,qq)))
                v=score[i].astype('f8');assert np.isfinite(v).all();v-=v.max();v-=math.log(math.fsum(map(float,np.exp(v))))
                value=math.fsum(map(float,np.exp(q)*(q-v)));loss.append(value)
                max_delta=max(max_delta,abs(value-row['KL_per_label'][i]));wrong+=int(np.argmax(q)!=np.argmax(v))
            value=math.fsum(loss)/n;max_delta=max(max_delta,abs(value-row['KL']))
            assert wrong==row['disagreement']
            assert wrong/n==row['disagreement_rate'] and n==row['labels']
            assert len(row['trajectory'])==7
            for field,boundary in zip(row['trajectory'],(0,4,8,12,16,20,24)):
                assert field['boundary']==boundary and field['shape']==[len(rec['student_input_ids']),256]
                z=np.memmap(field['path'],'<f4','r',shape=tuple(field['shape']));assert np.isfinite(z).all()
            dev.append(dict(id=rec['id'],domain=rec['domain'],KL=value,disagreement=wrong,disagreement_rate=wrong/n));print(json.dumps(dict(DEV=len(dev),seconds=time.monotonic()-START)),flush=True)
        assert labels==4386 and len(dev)==24 and max_delta<=1e-8
        case_KL=math.fsum(x['KL'] for x in dev)/24;case_dis=math.fsum(x['disagreement_rate'] for x in dev)/24
        assert abs(case_KL-r['DEV_case_KL'])<=1e-8 and abs(case_dis-r['DEV_case_disagreement'])<=1e-12
        domains={d:dict(case_KL=math.fsum(x['KL'] for x in dev if x['domain']==d)/sum(x['domain']==d for x in dev),
            case_disagreement=math.fsum(x['disagreement_rate'] for x in dev if x['domain']==d)/sum(x['domain']==d for x in dev)) for d in sorted({x['domain'] for x in dev})}
        for d,value in domains.items():
            assert abs(value['case_KL']-r['DEV_domains'][d]['case_KL'])<=1e-8
            assert abs(value['case_disagreement']-r['DEV_domains'][d]['case_disagreement'])<=1e-12
        dev_gates=dict(DEV_case_KL=case_KL<=1.,DEV_case_disagreement=case_dis<=.20,
            all_domain_KL=all(x['case_KL']<=2. for x in domains.values()),
            all_domain_disagreement=all(x['case_disagreement']<=.35 for x in domains.values()))
        assert dev_gates==r['DEV_quality_gates']
        # Local saved F32 runtime controls versus previously qualified F64 source packets.
        parent=json.loads(Path(b['operator_control']['path']).read_bytes());directory=Path(r['model']['path']).parent
        control=[]
        for row in r['controls']:
            site=row['site'];ref=next(x for x in parent['outputs'] if Path(x['path']).name==f'site{site:02d}.frames.npz')
            with np.load(directory/f'local.site{site:02d}.npz',allow_pickle=False) as actual,np.load(ref['path'],allow_pickle=False) as original:
                local_pass=True
                for key in actual.files:
                    x=actual[key].astype('f8');y=original['source_'+key]
                    value=math.sqrt(math.fsum(map(float,(x-y).ravel()**2)))/max(math.sqrt(math.fsum(map(float,y.ravel()**2))),1e-300)
                    assert abs(value-row['relative_RMS'][key])<=1e-12
                    local_pass &= value<=b['gates']['local_operator_relative_RMS']
                assert local_pass==row['numeric_gate']
                control.append(dict(site=site,numeric_gate=row['numeric_gate']))
        check();guard()
        result=dict(schema='FALCON_PROJECTED_SHADOW_STORED_AUDIT_V1',source=extent(__file__),report=extent(a.report),
            holder=extent(a.holder),parameters=maps,max_coefficient_sup_relative=max(x['sup_relative'] for x in maps),
            weighted_coefficient_gate=1e-5,all411_parameter_and_metric_fields_checked=True,all17_greedy_score_rows_checked=True,
            all24_DEV_4386_labels_and168_trajectories_checked=True,max_metric_delta=max_delta,
            controls=control,correct=correct,categories=categories,quality_gates=quality,
            own_followup_correct=follow['correct'],DEV=dev,DEV_case_KL=case_KL,DEV_case_disagreement=case_dis,DEV_quality_gates=dev_gates,
            all_inputs_outputs_before_after_exact=True,seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
            source_or_model_or_native_or_optimizer_replay=False,scope='Complete exported coefficient relations and stored behavior audit; '
                'no independent full model replay, no native/speed/useful-n/general-converter admission.')
        write(a.out,result)
    finally:
        if stream is not None:stream.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--report-sha',required=True)
    p.add_argument('--holder',type=Path,required=True);p.add_argument('--holder-sha',required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    try:main(args)
    except BaseException as error:
        import traceback
        write(args.out.with_suffix('.fault.json'),dict(error=repr(error),traceback=traceback.format_exc(),
            source=extent(__file__),command=sys.orig_argv,seconds=time.monotonic()-START))
        raise
