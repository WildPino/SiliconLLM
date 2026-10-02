#!/usr/bin/env python3
"""Complete stored M253 core plus effective centered E1280; no target scores."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import struct
import time
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch import nn
from torch.nn import functional as TF
from transformers import AutoConfig, AutoModelForCausalLM
import meth212_stored_core_loader as L
import meth252_private_feature_feasibility as G

P=L.P
SOURCE_RESULT=P.DOC/'meth252_private_feature_native_result.json'
SOURCE_SHA='7da69e9d44df64719fddbfb62a452a2b4afb3be7044511f1749d16305b6910ce'
NATIVE=P.DOC/'meth253_split_private_feature_native_result.json'
NATIVE_SHA='9262f98cd87021d32697bb79badc1f07d7480a558def3cce6c9e48fb7466611a'
FIXTURE=P.ROOT/'results/native_expert_scaling/meth252_private_feature_fixture.bin'
FORMAT='M253_ROWQ8_LUT_MIXED32_RES32_PRIVATE32_EFFECTIVE_E1280_V1'
BANK_FIELDS=('a','b','router','child_projection','child_keys')


def tensor_hash(t):
    return P.M17.sha(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())


class StoredFFN(nn.Module):
    def __init__(self,values):
        super().__init__();self.names=tuple(sorted(values))
        for name in self.names:self.register_buffer(name.replace('.','_'),values[name])

    def values(self):
        return {name:getattr(self,name.replace('.','_')) for name in self.names}

    def forward(self,x):
        flat=x.reshape(-1,896).float();v=self.values()
        result=G.readout(G.private_features(flat,v),v)
        return result.reshape(x.shape).to(x.dtype)


class StoredConditional(nn.Module):
    # The exact existing product-key and child-key arithmetic is reused.
    router_views=P.M55.ProductKeyExperts.router_views
    routes=P.M55.ProductKeyExperts.routes
    child_route=P.M95.HierarchicalExperts.child_route

    def __init__(self,base,values):
        super().__init__();self.base=base;self.enabled=True
        self.oracle_checks=False;self.collect_load=False
        for name in BANK_FIELDS:self.register_buffer(name,values[name])

    def forward(self,x):
        dense=self.base(x)
        if not self.enabled:return dense
        flat=x.reshape(-1,896);parents,scores=self.routes(flat)
        children=self.child_route(flat,parents);gate=TF.softmax(scores,dim=-1).to(flat.dtype)
        a=self.a[children//10].to(flat.dtype);b=self.b[children].to(flat.dtype)
        hidden=TF.silu(torch.einsum('nd,nkrd->nkr',flat,a))
        output=torch.einsum('nkr,nkdr->nkd',hidden,b)
        return dense+(output*gate.unsqueeze(-1)).sum(dim=1).reshape_as(dense)


def load_stored(path,device,expected_sha=None):
    """All executed model/bank weights and config come from this one archive."""
    if expected_sha is not None:assert P.digest(path)==expected_sha
    used=set();copied={};hashes={}
    with safe_open(str(path),framework='pt',device='cpu') as archive:
        metadata=archive.metadata();assert metadata['format']==FORMAT
        config_values=json.loads(metadata['config_json']);kind=config_values.pop('model_type')
        config=AutoConfig.for_model(kind,**config_values)
        torch.manual_seed(257257)
        model=AutoModelForCausalLM.from_config(config,dtype=torch.bfloat16,
            attn_implementation='sdpa').to(device).eval();model.config.use_cache=False
        parameters=dict(model.named_parameters());assert len(parameters)==290
        # Random constructor FFNs are deleted before any model execution.
        for layer in model.model.layers:layer.mlp=nn.Identity()
        with torch.no_grad():
            for name,param in model.named_parameters():
                organ=P.M24.classify(name,tuple(param.shape));assert organ!='ffn'
                key=name+'.bf16' if organ=='tied_head' else name
                value=archive.get_tensor(key);assert value.shape==param.shape
                param.copy_(value.to(device=device,dtype=param.dtype))
                assert torch.equal(param.detach().cpu(),value.to(param.dtype))
                used.add(key);copied[organ]=copied.get(organ,0)+1;hashes[key]=tensor_hash(value)
        del parameters,param,value
        assert copied=={'tied_head':1,'attention':96,'control':121}
        G.L.TABLE=archive.get_tensor('ffn.silu_table').to(device);used.add('ffn.silu_table')
        wrappers=[]
        for li,layer in enumerate(model.model.layers):
            prefix=f'ffn.{li}.';keys=[k for k in archive.keys() if k.startswith(prefix)]
            assert len(keys)==15
            values={k[len(prefix):]:archive.get_tensor(k).to(device) for k in keys};used.update(keys)
            bank={}
            for name in BANK_FIELDS:
                key=f'bank.{li}.{name}';value=archive.get_tensor(key)
                bank[name]=value.to(device);used.add(key);hashes[key]=tensor_hash(value)
            layer.mlp=StoredConditional(StoredFFN(values),bank);wrappers.append(layer.mlp)
        proposal={}
        for name,suffix in (('codes','.q'),('scales','.scale')):
            key=L.X.HEAD+suffix;proposal[name]=archive.get_tensor(key).to(device);used.add(key)
        assert used==set(archive.keys()) and len(used)==701
        assert model.model.embed_tokens.weight.data_ptr()==model.lm_head.weight.data_ptr()
        for parameter in model.parameters():parameter.requires_grad_(False)
    return model,wrappers,proposal,{'parameters_from_archive':218,'random_FFN_parameters_removed':72,
        'tensors_consumed':len(used),'organs':copied,'all_executed_tensors_consumed':True,
        'all_non_FFN_parameter_copies_equal':True,'tied_head_pointer_equal':True,
        'source_weights_or_conditional_checkpoints_loaded':False,'tensor_hashes':hashes}


def main():
    ap=argparse.ArgumentParser()
    for name in ('artifact','out'):ap.add_argument('--'+name,required=True,type=Path)
    args=ap.parse_args();assert not args.artifact.exists() and not args.out.exists()
    args.artifact.parent.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(args.artifact.parent).free>=4*1024**3
    start=time.monotonic();stage='bindings';segments=[];banks=[];parity=[]
    try:
        for path,sha in ((SOURCE_RESULT,SOURCE_SHA),(NATIVE,NATIVE_SHA),(L.CORE,L.CORE_SHA),
                         (L.EXPORT,L.EXPORT_SHA),(P.M122.SPECIALIZED,P.M122.SPECIALIZED_SHA),
                         (P.M122.TRAINING,P.M57.TRAINING_SHA),(G.C.VECTORS,G.C.VECTORS_SHA)):
            assert P.digest(path)==sha
        source=json.loads(SOURCE_RESULT.read_text());native=json.loads(NATIVE.read_text())
        assert all(native['gates'].values()) and P.digest(FIXTURE)==source['binary']['sha256']==native['fixture_sha256']
        parent=json.loads(P.M122.TRAINING.read_text())['checkpoints']['512']
        assert P.digest(parent['path'])==P.M57.CHECKPOINT_SHA
        device=G.Q.M.D.Q.setup();P.MAX_SECONDS=P.M17.MAX_SECONDS=20*60
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        config=AutoConfig.from_pretrained(P.M42.MODEL,revision=P.M42.REV,local_files_only=True)
        config_json=json.dumps(config.to_dict(),sort_keys=True)
        # This temporary model supplies the existing bank's exact centering
        # arithmetic. Its initialized dense weights are never executed/exported.
        torch.manual_seed(257257)
        original=AutoModelForCausalLM.from_config(config,dtype=torch.bfloat16,
            attn_implementation='sdpa').to(device).eval();original.config.use_cache=False
        L.D.H.load_centered(original,device,parent['path'])
        old_wrappers=[layer.mlp for layer in original.model.layers]
        tensors={};stage='exact_non_FFN_and_source_operator_assembly'
        with safe_open(str(L.CORE),framework='pt',device='cpu') as previous:
            for name in previous.keys():
                if '.mlp.' not in name:tensors[name]=previous.get_tensor(name).contiguous()
        assert len(tensors)==220
        with FIXTURE.open('rb') as file:
            assert file.read(32)==struct.pack('<8s6I',b'M252PF01',24,896,4864,32,32,32)
            for segment in source['segments']:
                file.seek(segment['offset']);raw=file.read(segment['bytes']);assert P.M17.sha(raw)==segment['sha256']
                name=segment['name']
                if name.startswith('private.'):
                    value=torch.from_numpy(np.frombuffer(raw,dtype='<u2').copy())
                    if name!='private.ids':value=value.reshape(32,896).view(torch.bfloat16)
                else:value=G.A.tensor_from_raw(name,raw)
                key='ffn.silu_table' if segment['layer'] is None else f"ffn.{segment['layer']}.{name}"
                tensors[key]=value.contiguous();assert tensor_hash(value)==segment['sha256']
                segments.append({'key':key,'bytes':len(raw),'sha256':segment['sha256']})
        assert len(segments)==361
        G.L.TABLE=tensors['ffn.silu_table'].to(device)
        stage='effective_BF16_centered_bank_conservation'
        for li,wrapper in enumerate(old_wrappers):
            a=wrapper.a.detach().view(128,10,8,896);assert torch.equal(a,a[:,:1].expand_as(a))
            bank={'a':a[:,0].bfloat16().cpu().contiguous(),'b':wrapper.b.detach().bfloat16().cpu().contiguous(),
                  **{name:getattr(wrapper,name).detach().cpu().contiguous() for name in BANK_FIELDS[2:]}}
            assert bank['a'].shape==(128,8,896) and bank['b'].shape==(1280,896,8)
            assert all(bank[name].dtype==torch.float32 for name in BANK_FIELDS[2:])
            expected_a=a[:,0].to(torch.bfloat16);assert torch.equal(bank['a'].to(device),expected_a)
            assert torch.equal(bank['b'].to(device),wrapper.b.detach().to(torch.bfloat16))
            groups=[len({tensor_hash(bank['b'][p*10+c]) for c in range(10)}) for p in range(128)]
            assert all(n==10 for n in groups)
            for name,value in bank.items():tensors[f'bank.{li}.{name}']=value
            banks.append({'layer':li,'private_BF16_B_groups_all10_distinct':True,'distinct_BF16_child_B':len({tensor_hash(v) for v in bank['b']}),
                'shared_parent_A_bytes':bank['a'].numel()*2,'private_B_bytes':bank['b'].numel()*2,
                'router_and_child_key_bytes':sum(bank[n].numel()*bank[n].element_size() for n in BANK_FIELDS[2:]),
                'hashes':{name:tensor_hash(value) for name,value in bank.items()}})
            P.budget(start,device)
        assert len(tensors)==701
        payload=sum(t.numel()*t.element_size() for t in tensors.values());assert payload<1_600_000_000
        metadata={'format':FORMAT,'experiment':'METH-257','model':P.M42.MODEL,'revision':P.M42.REV,
            'source_sha256':P.M57.MODEL_SHA,'source_operator_sha256':native['fixture_sha256'],
            'source_operator_result_sha256':NATIVE_SHA,'non_FFN_parent_archive_sha256':L.CORE_SHA,
            'parent_checkpoint_sha256':P.M57.CHECKPOINT_SHA,'child_checkpoint_sha256':P.M122.SPECIALIZED_SHA,
            'config_json':config_json,'bank_rule':'exact prior BF16 execution A,centered B; A shared within10 siblings',
            'source_FFN_rule':'rowQ8 gate/up plus private32 BF16,513-point LUT,mixed32 down,separateBF16 rank32',
            'dense_output_dtype':'bfloat16','conditional_activation':'exact SiLU,unchanged prior arithmetic',
            'shortlist_k':'64','probability_rule':'full exact BF16 tied head','script_sha256':P.digest(Path(__file__))}
        stage='save_and_complete_readback';save_file(tensors,str(args.artifact),metadata=metadata)
        assert args.artifact.stat().st_size<1_600_000_000
        with safe_open(str(args.artifact),framework='pt',device='cpu') as archive:
            assert archive.metadata()==metadata and set(archive.keys())==set(tensors)
            for name,value in tensors.items():
                assert torch.isfinite(value).all() and torch.equal(archive.get_tensor(name),value)
        artifact_sha=P.digest(args.artifact);print(json.dumps({'stage':stage,'payload_bytes':payload,'artifact_sha256':artifact_sha}),flush=True)
        stage='actual_stored_operator_and_conditional_component_parity'
        # Compare all256 stored original states in each of24 layers. No targets.
        raw=G.C.VECTORS.read_bytes();assert struct.unpack_from('<8s4I',raw)==(b'M125HX01',24,256,896,1280)
        states=np.frombuffer(raw,dtype='<u2',offset=24).reshape(256,24,896)
        # Keep only the reference conditional modules; free random dense organs.
        for wrapper in old_wrappers:wrapper.base=nn.Identity()
        del original;gc.collect();torch.cuda.empty_cache()
        model,wrappers,proposal,load_record=load_stored(args.artifact,device,artifact_sha)
        assert tensor_hash(proposal['codes'])==tensor_hash(tensors[L.X.HEAD+'.q'])
        assert tensor_hash(proposal['scales'])==tensor_hash(tensors[L.X.HEAD+'.scale'])
        for li,(old,new) in enumerate(zip(old_wrappers,wrappers)):
            x=torch.from_numpy(states[:,li].copy()).view(torch.bfloat16).to(device)
            values={segment['key'].split('.',2)[2]:tensors[segment['key']].to(device)
                for segment in segments if segment['key'].startswith(f'ffn.{li}.')}
            expected=G.readout(G.private_features(x.float(),values),values)
            observed=G.readout(G.private_features(x.float(),new.base.values()),new.base.values())
            assert torch.equal(expected,observed) and torch.equal(new.base(x),expected.bfloat16())
            parents,scores=old.routes(x);new_parents,new_scores=new.routes(x)
            assert torch.equal(parents,new_parents) and torch.equal(scores,new_scores)
            children=old.child_route(x,parents);assert torch.equal(children,new.child_route(x,new_parents))
            # Same qualified saved source base in both conditional arms.
            old.base=new.base;before=old(x);after=new(x)
            assert torch.equal(before,after) and torch.isfinite(after).all()
            parity.append({'layer':li,'states':256,'all_FFN_FP32_elements_bitwise_equal':True,
                'all_FFN_BF16_outputs_bitwise_equal':True,'all_parent_child_route_scores_bitwise_equal':True,
                'all_conditional_BF16_outputs_bitwise_equal':True,'full_conditional_output_sha256':tensor_hash(after)})
            P.budget(start,device)
        result={'experiment':'METH-257-complete-source-operator-core-effective-E1280-export-loader-parity',
            'source_operator_result_sha256':NATIVE_SHA,'source_segment_result_sha256':SOURCE_SHA,
            'non_FFN_archive_sha256':L.CORE_SHA,'source_sha256':P.M57.MODEL_SHA,
            'parent_checkpoint_sha256':P.M57.CHECKPOINT_SHA,'child_checkpoint_sha256':P.M122.SPECIALIZED_SHA,
            'artifact':{'path':str(args.artifact.resolve()),'bytes':args.artifact.stat().st_size,'payload_bytes':payload,'sha256':artifact_sha,'tensor_count':len(tensors)},
            'source_segments':segments,'bank_rows':banks,'component_parity':parity,'load_record':load_record,
            'gates':{'all361_source_segments_exact':True,'all220_non_FFN_and_head_proposal_fields_unchanged':True,
                'all24_effective_BF16_banks_and_centering_exact':True,'all128_sibling_groups_per_layer_10_distinct_B':True,
                'all701_snapshot_tensors_finite_readback_exact':True,'complete_stored_loader_no_source_weight_fallback':True,
                'all6144_source_and_conditional_vectors_bitwise_equal':True},
            'runtime':P.budget(start,device),'script_sha256':P.digest(Path(__file__)),
            'decision':'complete_saved_candidate_pass_freeze_full_model_development_screen',
            'scope':'Actual24-layer saved core and learned E1280 bank. No target/document/logit/generation/task screening,n-scaled DRAM/native full-model parity or accepted rate/new family/10B/100B proof.'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:result[k] for k in ('decision','artifact','gates','runtime')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),'source_segments':segments,
            'bank_rows':banks,'component_parity':parity,'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8');raise


if __name__=='__main__':main()
