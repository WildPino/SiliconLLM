"""Unmodified official cached greedy and generate bridge."""
import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersTop1Router


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
