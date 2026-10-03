"""Model-free current Switch forward contract, before checkpoint acquisition."""
import time
START=time.monotonic()
import argparse
import inspect
from pathlib import Path
import json
import psutil
import torch
from transformers import SwitchTransformersConfig,SwitchTransformersForConditionalGeneration
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersSparseMLP
import meth303_compact_i8_preflight as M
import meth299_gigachat_block_selection as P
PROTOCOL=M.DOC/'METH_322_SWITCH_RUNTIME_CONTRACT_PROTOCOL_20261003.md'
PRIOR=M.DOC/'meth321_switch_metadata_compatibility_result.json'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists()
    result={'experiment':'METH-322-model-free-installed-Switch-runtime-contract','cases':[]}
    try:
        for path in (Path(__file__),PROTOCOL,PRIOR):P.committed(path)
        assert M.digest(PRIOR)=='31351eae3934228468366501492c291f8a7f8b0a228bf3db80a146d797b9ebb1'
        source=Path(inspect.getfile(SwitchTransformersSparseMLP));assert M.digest(source)=='f1eab2d3e02e70d93d20d6fb07dd3acd20519fe6f88a57d7a1b214bc5b305ecd'
        torch.set_num_threads(1);torch.manual_seed(322)
        cfg=SwitchTransformersConfig(d_model=8,d_ff=16,d_kv=4,num_heads=2,num_experts=2,
            num_layers=2,num_decoder_layers=2,num_sparse_encoder_layers=1,num_sparse_decoder_layers=1,
            expert_capacity=1,dropout_rate=0.,router_jitter_noise=0.,vocab_size=32,
            pad_token_id=0,eos_token_id=1,decoder_start_token_id=0,router_dtype='float32')
        sparse=SwitchTransformersSparseMLP(cfg).eval()
        with torch.no_grad():
            sparse.router.classifier.weight.zero_();sparse.router.classifier.weight[0,0]=1.;sparse.router.classifier.weight[1,0]=-1.
        x=torch.zeros(1,4,8);x[0,:,0]=torch.tensor([1.,2.,-1.,-2.]);x[0,:,1]=1.
        with torch.no_grad():
            probability,mask,selected=sparse.router(x)
        expected=torch.tensor([[[1,0],[0,0],[0,1],[0,0]]])
        assert torch.equal(mask,expected) and probability.shape==(1,4,1) and selected.shape==(1,4,1)
        assert torch.isfinite(probability).all() and (probability>.5).all()
        assert torch.equal(probability,torch.softmax(sparse.router.classifier(x),dim=-1).max(dim=-1,keepdim=True).values)
        with torch.no_grad():assert all(torch.equal(a,b) for a,b in zip((probability,mask,selected),sparse.router(x)))
        result.update({'controller_sha256':M.digest(Path(__file__)),'protocol_sha256':M.digest(PROTOCOL),
            'source_sha256':M.digest(source),'prior_sha256':M.digest(PRIOR),'tiny_config':cfg.to_dict(),
            'direct_3D_router_control':{'exact_top1_capacity_mask':True,'F32_selected_probability_exact':True,'repeatable_eval_no_jitter':True},
            'learned_source_weights_read':False})
        for shape in ((1,1,8),(1,4,8),(2,2,8)):
            values=torch.arange(shape[0]*shape[1]*shape[2],dtype=torch.float32).reshape(shape)/32.-.5
            entry={'input_shape':list(shape),'passed':False}
            try:
                with torch.no_grad():output=sparse(values)
                entry.update({'returned_shape':list(output.shape),'finite':bool(torch.isfinite(output).all())})
                entry['passed']=output.shape==values.shape and entry['finite']
            except Exception as error:entry['error']=repr(error)
            result['cases'].append(entry)
        tiny=SwitchTransformersForConditionalGeneration(cfg).eval();head={'passed':False}
        try:
            with torch.no_grad():output=tiny(input_ids=torch.tensor([[2,3,4]]),decoder_input_ids=torch.tensor([[0]]),use_cache=True)
            head.update({'returned_shape':list(output.logits.shape),'finite':bool(torch.isfinite(output.logits).all()),'cache_present':output.past_key_values is not None})
            head['passed']=output.logits.shape==(1,1,32) and head['finite'] and head['cache_present']
        except Exception as error:head['error']=repr(error)
        result['tiny_full_head_and_cache']=head
        rss=psutil.Process().memory_info().rss;seconds=time.monotonic()-START;assert seconds<=300 and rss<=2*(1<<30)
        result['gates']={'direct_router_top1_capacity_probability':True,'every_sparse_forward_contract':all(v['passed'] for v in result['cases']),
                         'tiny_model_head_cache_contract':head['passed']}
        result['decision']='installed_reference_unsupported_before_large_checkpoint_acquisition' if not all(result['gates'].values()) else 'eligible_only_for_separate_checkpoint_binding_and_full_reference_semantics'
        result['resource']={'seconds':seconds,'end_rss_bytes':rss}
        result['scope']='Tiny synthetic operator contract, not source model quality/cost/knowledge or accepted rate. Direct router control does not qualify full expert dispatcher or checkpoint reference.'
        M.write(args.out,result);print(json.dumps({'gates':result['gates'],'cases':result['cases'],'full':head,
            'decision':result['decision'],'sha256':M.digest(args.out),'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'error':repr(error),'seconds':time.monotonic()-START});M.write(args.out.with_suffix('.failure.json'),result);raise

if __name__=='__main__':main()
