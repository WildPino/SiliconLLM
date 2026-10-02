#!/usr/bin/env python3
"""Complete source core with actual unique functions and conserved route aliases."""
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
import meth258_effective_bank_diversity as Z

X=Z.X;P=X.P;G=X.G;L=X.L
AUDIT=P.DOC/'meth258_effective_bank_diversity_result.json'
AUDIT_SHA='e5069a0ed59938562ec7d995e26391d812dd7fde8c94e7d208230c98b9525311'
FORMAT='M253_COMPLETE_CORE_UNIQUE_BF16_CORRECTIONS_ROUTE_ALIAS_V1'
BANK_FIELDS=(*X.BANK_FIELDS,'leaf_map')


class MappedConditional(X.StoredConditional):
    def __init__(self,base,values):
        super().__init__(base,values);self.register_buffer('leaf_map',values['leaf_map'])

    def forward(self,x):
        dense=self.base(x)
        if not self.enabled:return dense
        flat=x.reshape(-1,896);parents,scores=self.routes(flat)
        children=self.child_route(flat,parents);gate=TF.softmax(scores,dim=-1).to(flat.dtype)
        a=self.a[children//10].to(flat.dtype)
        b=self.b[self.leaf_map[children].long()].to(flat.dtype)
        hidden=TF.silu(torch.einsum('nd,nkrd->nkr',flat,a))
        output=torch.einsum('nkr,nkdr->nkd',hidden,b)
        return dense+(output*gate.unsqueeze(-1)).sum(dim=1).reshape_as(dense)


def load_stored(path,device,expected_sha=None):
    if expected_sha is not None:assert P.digest(path)==expected_sha
    used=set();copied={}
    with safe_open(str(path),framework='pt',device='cpu') as archive:
        metadata=archive.metadata();assert metadata['format']==FORMAT
        counts=json.loads(metadata['actual_function_counts_json']);assert len(counts)==24
        config_values=json.loads(metadata['config_json']);kind=config_values.pop('model_type')
        config=AutoConfig.for_model(kind,**config_values);torch.manual_seed(259259)
        model=AutoModelForCausalLM.from_config(config,dtype=torch.bfloat16,
            attn_implementation='sdpa').to(device).eval();model.config.use_cache=False
        assert len(list(model.named_parameters()))==290
        for layer in model.model.layers:layer.mlp=nn.Identity()
        with torch.no_grad():
            for name,param in model.named_parameters():
                organ=P.M24.classify(name,tuple(param.shape));assert organ!='ffn'
                key=name+'.bf16' if organ=='tied_head' else name;value=archive.get_tensor(key)
                assert value.shape==param.shape;param.copy_(value.to(device=device,dtype=param.dtype))
                assert torch.equal(param.detach().cpu(),value.to(param.dtype))
                used.add(key);copied[organ]=copied.get(organ,0)+1
        del param,value;assert copied=={'tied_head':1,'attention':96,'control':121}
        G.L.TABLE=archive.get_tensor('ffn.silu_table').to(device);used.add('ffn.silu_table');wrappers=[]
        for li,layer in enumerate(model.model.layers):
            prefix=f'ffn.{li}.';keys=[k for k in archive.keys() if k.startswith(prefix)];assert len(keys)==15
            values={k[len(prefix):]:archive.get_tensor(k).to(device) for k in keys};used.update(keys)
            bank={}
            for name in BANK_FIELDS:
                key=f'bank.{li}.{name}';bank[name]=archive.get_tensor(key).to(device);used.add(key)
            assert bank['a'].shape==(128,8,896) and bank['a'].dtype==torch.bfloat16
            assert bank['b'].shape==(counts[li],896,8) and bank['b'].dtype==torch.bfloat16
            assert bank['leaf_map'].shape==(1280,) and bank['leaf_map'].dtype==torch.int32
            assert int(bank['leaf_map'].min())==0 and int(bank['leaf_map'].max())==counts[li]-1
            assert len(torch.unique(bank['leaf_map']))==counts[li]
            layer.mlp=MappedConditional(X.StoredFFN(values),bank);wrappers.append(layer.mlp)
        proposal={}
        for name,suffix in (('codes','.q'),('scales','.scale')):
            key=L.X.HEAD+suffix;proposal[name]=archive.get_tensor(key).to(device);used.add(key)
        assert used==set(archive.keys()) and len(used)==725
        assert model.model.embed_tokens.weight.data_ptr()==model.lm_head.weight.data_ptr()
        for parameter in model.parameters():parameter.requires_grad_(False)
    return model,wrappers,proposal,{'parameters_from_archive':218,'random_FFN_parameters_removed':72,
        'tensors_consumed':len(used),'organs':copied,'all_non_FFN_copies_exact':True,
        'all_stored_route_maps_surjective_and_in_range':True,'actual_function_counts':counts,
        'tied_head_pointer_equal':True,'source_weights_or_conditional_checkpoints_loaded':False}


def main():
    ap=argparse.ArgumentParser()
    for name in ('artifact','out'):ap.add_argument('--'+name,required=True,type=Path)
    args=ap.parse_args();assert not args.artifact.exists() and not args.out.exists()
    args.artifact.parent.mkdir(parents=True,exist_ok=True);assert shutil.disk_usage(args.artifact.parent).free>=4*1024**3
    start=time.monotonic();stage='bindings';segments=[];banks=[];parity=[]
    try:
        for path,sha in ((AUDIT,AUDIT_SHA),(X.SOURCE_RESULT,X.SOURCE_SHA),(X.NATIVE,X.NATIVE_SHA),
                         (L.CORE,L.CORE_SHA),(L.EXPORT,L.EXPORT_SHA),(P.M122.SPECIALIZED,P.M122.SPECIALIZED_SHA),
                         (P.M122.TRAINING,P.M57.TRAINING_SHA),(G.C.VECTORS,G.C.VECTORS_SHA)):
            assert P.digest(path)==sha
        audit=json.loads(AUDIT.read_text());assert all(audit['gates'].values())
        source=json.loads(X.SOURCE_RESULT.read_text());native=json.loads(X.NATIVE.read_text())
        assert all(native['gates'].values()) and P.digest(X.FIXTURE)==source['binary']['sha256']==native['fixture_sha256']
        parent=json.loads(P.M122.TRAINING.read_text())['checkpoints']['512'];assert P.digest(parent['path'])==P.M57.CHECKPOINT_SHA
        device=G.Q.M.D.Q.setup();P.MAX_SECONDS=P.M17.MAX_SECONDS=20*60
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        config=AutoConfig.from_pretrained(P.M42.MODEL,revision=P.M42.REV,local_files_only=True)
        temporary=Z.Placeholder();L.D.H.load_centered(temporary,device,parent['path'])
        old_wrappers=[layer.mlp for layer in temporary.model.layers];tensors={}
        stage='unchanged_complete_non_FFN_and_source_operator'
        with safe_open(str(L.CORE),framework='pt',device='cpu') as previous:
            for name in previous.keys():
                if '.mlp.' not in name:tensors[name]=previous.get_tensor(name).contiguous()
        assert len(tensors)==220
        with X.FIXTURE.open('rb') as file:
            assert file.read(32)==struct.pack('<8s6I',b'M252PF01',24,896,4864,32,32,32)
            for segment in source['segments']:
                file.seek(segment['offset']);raw=file.read(segment['bytes']);assert P.M17.sha(raw)==segment['sha256']
                name=segment['name']
                if name.startswith('private.'):
                    value=torch.from_numpy(np.frombuffer(raw,dtype='<u2').copy())
                    if name!='private.ids':value=value.reshape(32,896).view(torch.bfloat16)
                else:value=G.A.tensor_from_raw(name,raw)
                key='ffn.silu_table' if segment['layer'] is None else f"ffn.{segment['layer']}.{name}"
                tensors[key]=value.contiguous();assert X.tensor_hash(value)==segment['sha256']
                segments.append({'key':key,'bytes':len(raw),'sha256':segment['sha256']})
        assert len(segments)==361;stage='actual_unique_function_dictionary_and_original_leaf_aliases'
        for li,wrapper in enumerate(old_wrappers):
            a=wrapper.a.detach().view(128,10,8,896);assert torch.equal(a,a[:,:1].expand_as(a))
            original_b=wrapper.b.detach().bfloat16().cpu().contiguous();parent_a=a[:,0].bfloat16().cpu().contiguous()
            expected=[code for group in audit['layers'][li]['groups'] for code in group['BF16_signatures']]
            mapping={};entries=[];representatives=[];codes=[];leaf_map=[]
            for leaf in range(1280):
                code=Z.bank_hash(parent_a[leaf//10],original_b[leaf]);assert code==expected[leaf]
                if code not in mapping:
                    mapping[code]=len(entries);entries.append(original_b[leaf].clone());representatives.append(leaf);codes.append(code)
                leaf_map.append(mapping[code])
            count=len(entries);assert count==audit['layers'][li]['BF16_effective_function_parameter_count']
            bank={'a':parent_a,'b':torch.stack(entries),'leaf_map':torch.tensor(leaf_map,dtype=torch.int32),
                **{name:getattr(wrapper,name).detach().cpu().contiguous() for name in X.BANK_FIELDS[2:]}}
            assert len(set(codes))==count and set(leaf_map)==set(range(count))
            assert torch.equal(bank['b'][bank['leaf_map'].long()],original_b)
            assert torch.equal(bank['a'].to(device),a[:,0].bfloat16())
            for name,value in bank.items():tensors[f'bank.{li}.{name}']=value
            banks.append({'layer':li,'routing_leaves':1280,'actual_unique_function_parameters':count,
                'aliased_leaves':1280-count,'representative_original_leaf':representatives,'unique_function_signatures':codes,
                'parent_A_bytes':parent_a.numel()*2,'private_unique_B_bytes':bank['b'].numel()*2,
                'leaf_map_bytes':bank['leaf_map'].numel()*4,
                'router_and_child_key_bytes':sum(bank[n].numel()*4 for n in X.BANK_FIELDS[2:]),
                'all_original_BF16_child_B_lookup_values_exact':True,'tensor_hashes':{n:X.tensor_hash(v) for n,v in bank.items()}})
            P.budget(start,device)
        assert len(tensors)==725
        counts=[r['actual_unique_function_parameters'] for r in banks]
        payload=sum(t.numel()*t.element_size() for t in tensors.values());assert payload<1_600_000_000
        helper_hashes={str(Path(module.__file__).resolve()):P.digest(Path(module.__file__)) for module in (X,Z,G,G.A,G.L,P.M55,P.M95,L.D.H)}
        metadata={'format':FORMAT,'experiment':'METH-259','config_json':json.dumps(config.to_dict(),sort_keys=True),
            'source_sha256':P.M57.MODEL_SHA,'source_operator_sha256':native['fixture_sha256'],
            'source_operator_result_sha256':X.NATIVE_SHA,'non_FFN_archive_sha256':L.CORE_SHA,
            'parent_checkpoint_sha256':P.M57.CHECKPOINT_SHA,'child_checkpoint_sha256':P.M122.SPECIALIZED_SHA,
            'diversity_result_sha256':AUDIT_SHA,'actual_function_counts_json':json.dumps(counts),
            'dictionary_rule':'canonical BF16 A/B or exact zero,first original leaf occurrence;int32 leaf_map preserves1280 route addresses',
            'dense_output_dtype':'bfloat16','conditional_activation':'exact original SiLU,BF16 A/B arithmetic',
            'shortlist_k':'64','probability_rule':'full exact BF16 tied head','script_sha256':P.digest(Path(__file__)),
            'helper_hashes_json':json.dumps(helper_hashes,sort_keys=True)}
        stage='save_and_all725_tensor_readback';save_file(tensors,str(args.artifact),metadata=metadata)
        assert args.artifact.stat().st_size<1_600_000_000
        with safe_open(str(args.artifact),framework='pt',device='cpu') as archive:
            assert archive.metadata()==metadata and set(archive.keys())==set(tensors)
            for name,value in tensors.items():assert torch.isfinite(value).all() and torch.equal(archive.get_tensor(name),value)
        artifact_sha=P.digest(args.artifact);print(json.dumps({'stage':stage,'artifact_sha256':artifact_sha,'payload_bytes':payload}),flush=True)
        stage='complete_stored_loader_and_original_conditional_parity'
        del temporary;gc.collect();torch.cuda.empty_cache()
        model,wrappers,proposal,load_record=load_stored(args.artifact,device,artifact_sha)
        for name,suffix in (('codes','.q'),('scales','.scale')):assert X.tensor_hash(proposal[name])==X.tensor_hash(tensors[L.X.HEAD+suffix])
        raw=G.C.VECTORS.read_bytes();assert struct.unpack_from('<8s4I',raw)==(b'M125HX01',24,256,896,1280)
        states=np.frombuffer(raw,dtype='<u2',offset=24).reshape(256,24,896)
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
            assert torch.equal(new.b[new.leaf_map[children].long()],old.b[children].bfloat16())
            old.base=new.base;before=old(x);after=new(x);assert torch.equal(before,after) and torch.isfinite(after).all()
            parity.append({'layer':li,'states':256,'all_FFN_FP32_and_BF16_outputs_bitwise_equal':True,
                'all_original_parent_scores_and_leaf_routes_bitwise_equal':True,'all_selected_alias_BF16_values_exact':True,
                'all_original_centered_conditional_BF16_outputs_bitwise_equal':True,'output_sha256':X.tensor_hash(after)})
            P.budget(start,device)
        result={'experiment':'METH-259-complete-source-core-actual-unique-function-bank-route-alias-export',
            'diversity_result_sha256':AUDIT_SHA,'source_operator_result_sha256':X.NATIVE_SHA,
            'non_FFN_archive_sha256':L.CORE_SHA,'source_sha256':P.M57.MODEL_SHA,
            'parent_checkpoint_sha256':P.M57.CHECKPOINT_SHA,'child_checkpoint_sha256':P.M122.SPECIALIZED_SHA,
            'artifact':{'path':str(args.artifact.resolve()),'bytes':args.artifact.stat().st_size,'payload_bytes':payload,'sha256':artifact_sha,'tensor_count':len(tensors)},
            'source_segments':segments,'bank_rows':banks,'component_parity':parity,'load_record':load_record,
            'helper_sha256':helper_hashes,'gates':{'all361_source_segments_exact':True,'all220_non_FFN_fields_unchanged':True,
                'all24_dictionary_counts_and_signatures_match_fixed_audit':True,'all725_snapshot_fields_finite_readback_exact':True,
                'all_original1280_BF16_leaf_functions_preserved_without_duplicate_storage':True,
                'all_maps_in_range_and_surjective':True,'all6144_original_source_conditional_vectors_bitwise_equal':True,
                'complete_stored_loader_no_source_weight_or_checkpoint_fallback':True},
            'runtime':P.budget(start,device),'script_sha256':P.digest(Path(__file__)),
            'decision':'complete_unique_saved_candidate_pass_freeze_full_model_development_screen',
            'scope':'One actual saved24-layer core,1243..1280 unique effective parameter functions per layer,1280 preserved route labels. No new n-semantic benefit/RAM-scaled routing/DRAM,full document/logit/generation/task/native accepted-rate or multi-family/10B/100B proof.'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:result[k] for k in ('decision','artifact','gates','runtime')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),'source_segments':segments,
            'bank_rows':banks,'component_parity':parity,'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8');raise


if __name__=='__main__':main()
