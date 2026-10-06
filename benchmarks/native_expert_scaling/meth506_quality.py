"""506 actual donor APIs and retained quality rubrics; frozen before observations."""

import struct
from pathlib import Path
import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, LogitsProcessor, LogitsProcessorList
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersTop1Router

MASK=np.array([1,2,4,5,7,8,10,11])

def torch_reference(model,source,decoder):
    encoder_states=[model.shared(source).detach().numpy()[0].copy()]
    decoder_states=[];route=[];current=[];hooks=[]
    def encoder_hook(_module,_args,output):encoder_states.append(output[0].detach().numpy()[0].copy())
    def decoder_hook(_module,_args,output):current.append(output[0].detach().numpy()[0,0].copy())
    def router_hook(_module,_args,output):
        mask,prob,logits=output
        chosen=logits.argmax(dim=-1).reshape(-1).tolist();accepted=mask.any(dim=-1).reshape(-1).tolist();p=prob.reshape(-1).tolist()
        route.extend(zip(chosen,accepted,p))
    for block in model.encoder.block:hooks.append(block.register_forward_hook(encoder_hook))
    for block in model.decoder.block:hooks.append(block.register_forward_hook(decoder_hook))
    for module in model.modules():
        if type(module) is SwitchTransformersTop1Router:hooks.append(module.register_forward_hook(router_hook))
    logits=[];cache=None
    with torch.no_grad():
        encoded=model.encoder(input_ids=source,return_dict=True)
        encoder_states.append(encoded.last_hidden_state.numpy()[0].copy())
        for step in range(decoder.shape[1]):
            current.clear();current.append(model.shared(decoder[:,step:step+1]).numpy()[0,0].copy())
            output=model(encoder_outputs=encoded,decoder_input_ids=decoder[:,step:step+1],past_key_values=cache,use_cache=True,output_hidden_states=True)
            cache=output.past_key_values;current.append(output.decoder_hidden_states[-1].numpy()[0,0].copy())
            decoder_states.append(np.stack(current));logits.append(output.logits.numpy()[0,0].copy())
    for hook in hooks:hook.remove()
    return np.stack(encoder_states),np.stack(decoder_states),np.stack(logits),np.array(route,dtype=np.float64)

def read_output(path):
    data=path.read_bytes();assert data[:8]==b'SWR32O01'
    s,t,d,enc,dec,vocab,nroutes=struct.unpack_from('<7I',data,8);offset=36
    def floats(shape):
        nonlocal offset
        count=int(np.prod(shape));result=np.frombuffer(data,dtype='<f4',count=count,offset=offset).reshape(shape).copy();offset+=count*4;return result
    a=floats((enc+2,s,d));b=floats((t,dec+2,d));c=floats((t,vocab))
    routes=np.array([struct.unpack_from('<iif',data,offset+12*i) for i in range(nroutes)],dtype=np.float64)
    assert len(data)==offset+12*nroutes
    return a,b,c,routes

def relative(a,b):
    denominator=float(np.linalg.norm(b.astype(np.float64)));assert denominator>0
    return float(np.linalg.norm(a.astype(np.float64)-b.astype(np.float64))/denominator)

def cached_greedy(model, source, cap, closing):
    assert model.training is False and source.shape[0] == 1 and 0 < cap <= 64
    assert 0 < closing < model.config.vocab_size
    encoder_states=[model.shared(source).detach().numpy()[0].copy()]
    decoder_states=[];route=[];current=[];hooks=[];logits=[];choices=[]
    def encoder_hook(_module,_args,output):
        encoder_states.append(output[0].detach().numpy()[0].copy())
    def decoder_hook(_module,_args,output):
        current.append(output[0].detach().numpy()[0,0].copy())
    def router_hook(_module,_args,output):
        mask,prob,scores=output
        selected=scores.argmax(dim=-1).reshape(-1).tolist()
        accepted=mask.any(dim=-1).reshape(-1).tolist()
        route.extend(zip(selected,accepted,prob.reshape(-1).tolist()))
    for block in model.encoder.block:hooks.append(block.register_forward_hook(encoder_hook))
    for block in model.decoder.block:hooks.append(block.register_forward_hook(decoder_hook))
    for module in model.modules():
        if type(module) is SwitchTransformersTop1Router:
            hooks.append(module.register_forward_hook(router_hook))
    try:
        with torch.no_grad():
            encoded=model.encoder(input_ids=source,return_dict=True)
            encoder_states.append(encoded.last_hidden_state.numpy()[0].copy())
            cache=None;next_id=0
            for _ in range(cap):
                decoder=torch.tensor([[next_id]])
                current.clear();current.append(model.shared(decoder).numpy()[0,0].copy())
                output=model(encoder_outputs=encoded,decoder_input_ids=decoder,
                             past_key_values=cache,use_cache=True,output_hidden_states=True)
                cache=output.past_key_values
                current.append(output.decoder_hidden_states[-1].numpy()[0,0].copy())
                decoder_states.append(np.stack(current));values=output.logits.numpy()[0,0].copy()
                assert np.isfinite(values).all()
                logits.append(values);next_id=int(values.argmax());choices.append(next_id)
                if next_id in (1,closing):break
    finally:
        for hook in hooks:hook.remove()
    arrays=(np.stack(encoder_states),np.stack(decoder_states),np.stack(logits),np.array(route,dtype=np.float64))
    return choices,arrays

class ClosingSentinel(StoppingCriteria):
    def __init__(self,closing):self.closing=closing
    def __call__(self,input_ids,scores,**kwargs):
        return input_ids[:,-1]==self.closing

def official_greedy(model,source,cap,closing):
    with torch.no_grad():
        result=model.generate(input_ids=source,max_new_tokens=cap,do_sample=False,
                              num_beams=1,use_cache=True,decoder_start_token_id=0,
                              eos_token_id=1,pad_token_id=0,
                              stopping_criteria=StoppingCriteriaList([ClosingSentinel(closing)]),
                              return_dict_in_generate=True,output_scores=True)
    assert result.sequences.shape[0]==1 and result.sequences[0,0].item()==0
    choices=result.sequences[0,1:].tolist()
    scores=np.stack([v.detach().numpy()[0].copy() for v in result.scores])
    assert len(choices)==scores.shape[0] and choices==scores.argmax(-1).tolist()
    assert len(choices)<=cap and (len(choices)==cap or choices[-1] in (1,closing))
    return choices,scores

def official_teacher_cached(model, source, decoder_ids):
    assert not model.training and source.shape[0] == 1
    assert 0 < len(decoder_ids) <= 64 and decoder_ids[0] == 0
    assert all(0 <= value < model.config.vocab_size for value in decoder_ids)
    raw = []

    class CaptureTeacher(LogitsProcessor):
        def __call__(self, input_ids, scores):
            position = input_ids.shape[1] - 1
            assert 0 <= position < len(decoder_ids)
            assert input_ids[0].tolist() == decoder_ids[:position + 1]
            assert scores.shape == (1, model.config.vocab_size)
            values = scores.detach().cpu().numpy()[0].copy()
            assert np.isfinite(values).all()
            raw.append(values)
            next_id = decoder_ids[position + 1] if position + 1 < len(decoder_ids) else 0
            forced = torch.full_like(scores, -torch.inf)
            forced[:, next_id] = 0
            return forced

    with torch.no_grad():
        result = model.generate(input_ids=source, max_new_tokens=len(decoder_ids),
                                do_sample=False, num_beams=1, use_cache=True,
                                decoder_start_token_id=0, eos_token_id=None,
                                pad_token_id=0,
                                logits_processor=LogitsProcessorList([CaptureTeacher()]),
                                return_dict_in_generate=True, output_scores=False)
    assert result.sequences[0].tolist() == decoder_ids + [0]
    assert len(raw) == len(decoder_ids)
    return np.stack(raw)

def metrics(logits,target,spans):
    assert logits.shape==(14,32128) and len(target)==14 and np.asarray(spans).shape==(4,2)
    values=logits.astype(np.float64);shift=values.max(-1,keepdims=True)
    nll=np.log(np.exp(values-shift).sum(-1))+shift[:,0]-values[np.arange(14),target]
    choice=logits.argmax(-1);truth=np.asarray(spans).reshape(8)
    return {'mean_nll':float(nll.mean()),'mean_span_nll':float(nll[MASK].mean()),
            'correct_span_tokens':int(np.sum(choice[MASK]==truth)),
            'correct_fields':int(np.all((choice[MASK]==truth).reshape(4,2),axis=1).sum()),
            'all_span_teacher_forced_correct':bool(np.array_equal(choice[MASK],truth)),
            'greedy_teacher_forced_ids':choice.tolist(),'per_target_nll':nll.tolist()}

def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a):
        new=[i+1]
        for j,y in enumerate(b):new.append(min(new[-1]+1,row[j+1]+1,row[j]+(x!=y)))
        row=new
    return row[-1]

def lcs(a,b):
    row=[0]*(len(b)+1)
    for x in a:
        new=[0]
        for j,y in enumerate(b):new.append(row[j]+1 if x==y else max(new[-1],row[j+1]))
        row=new
    return row[-1]

def task(ids,truth):
    fields=[[] for _ in range(4)];expected=0;current=-1;malformed=False;closed=False
    for token in ids:
        if token==1:break
        if token==0:malformed=True;continue
        if token>=32000:
            if expected<=4 and token==32099-expected:
                if expected==4:closed=True;current=-1
                else:current=expected
                expected+=1
            else:malformed=True
        elif 0<=current<4:fields[current].append(token)
        else:malformed=True
    if malformed:fields=[[] for _ in range(4)]
    trigrams=[[(field[i],field[i+1],field[i+2]) for i in range(max(len(field)-2,0))] for field in fields]
    total=sum(len(values) for values in trigrams)
    repeated=(total-sum(len(set(values)) for values in trigrams))/max(total,1)
    healthy=closed and expected==5 and not malformed and all(fields) and ids[-1]==32095 and repeated<=.50
    exact=sum(field==gold for field,gold in zip(fields,truth))
    correct=sum(sum(i<len(field) and field[i]==gold[i] for i in range(2)) for field,gold in zip(fields,truth))
    f1=np.mean([2*lcs(field,gold)/(len(field)+len(gold)) for field,gold in zip(fields,truth)])
    prose=[token for token in ids if 1<token<32000]
    return {'healthy_complete_nonempty_fields':bool(healthy),'fields':fields,'correct_fields':exact,
            'known_token_accuracy':correct/8,'known_field_exact_accuracy':exact/4,'known_field_lcs_f1':float(f1),
            'generated_ids':ids,'generated_tokens':len(ids),'prose_tokens':len(prose),'prose_ids':prose,
            'within_field_repeated_trigram_fraction':repeated,
            'cap_reached_without_terminal':len(ids)==64 and ids[-1] not in (1,32095)}

def paired_interval(values):
    rng=np.random.default_rng(351351);values=np.asarray(values,dtype=np.float64);assert values.shape==(24,)
    draws=values[rng.integers(0,24,size=(10000,24))].mean(-1)
    return {'mean':float(values.mean()),'one_sided_lower95':float(np.quantile(draws,.05)),
            'one_sided_upper95':float(np.quantile(draws,.95)),'bootstrap_unit':'book','draws':10000,'seed':351351}
