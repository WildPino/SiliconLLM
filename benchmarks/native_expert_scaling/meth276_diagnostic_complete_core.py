#!/usr/bin/env python3
"""Diagnostic complete archive with fixed M274 arithmetic and M271 private rows."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import struct
import sys
import time

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch import nn
from torch.nn import functional as TF
from transformers import AutoConfig, AutoModelForCausalLM
import meth259_unique_bank_core_export as R
import meth274_i16_shared_input as I

P,G=R.P,R.G
FORMAT='M276_DIAGNOSTIC_I16_INPUT_PRIVATE128_UNIQUE_BF16_V1'
CORE=P.ROOT/'results/native_expert_scaling/meth259_qwen05b_unique_source_core.safetensors'
CORE_SHA='3c0949fb2b7f99dc887c6aec1041aa162b35801939349966d18e52b8807a6f71'
EXPORT=P.DOC/'meth259_unique_bank_core_export_result.json'
EXPORT_SHA='39f796651522a2af907f4123756b9c7ab1e838057debbb142db5c1e02765b663'
I16=P.DOC/'meth274_i16_shared_input_result.json'
I16_SHA='aa2acf6452031d621568cb4712c1a976587755c398af2dc03274ac844cd451cc'
PAIRED=P.DOC/'meth275_paired_i16_result.json'
PAIRED_SHA='4d2b9bcb69c8b1d39d1d6a7c1090b212b129373b11322791edd376c3eea9a0c9'
FIXTURE_SHA='ece28ea3344d9ecef674d30b4c43b067ee4290379cf9130c97c2eafee1e344d8'
NATIVE=P.ROOT/'results/native_expert_scaling/meth274_i16_shared_input.check.bin'
NATIVE_SHA='0336e15c46e80f8d4d45816525e16cce8f34de44e635c04fac584f12257ca5ba'
QUANTIZER=P.ROOT/'results/native_expert_scaling/meth274_i16_quantizer.bin'
QUANTIZER_SHA='ae70d33d3945c6ad98467eae6196931aa9da302b26d4b18377742fe2091eb70d'
PRIVATE=('private.ids','private.gate','private.up')


def record_tensor(value):
    return {'shape':list(value.shape),'dtype':str(value.dtype),
            'bytes':value.numel()*value.element_size(),'sha256':R.X.tensor_hash(value)}


class StoredI16FFN(R.X.StoredFFN):
    def fp32(self,x):
        assert x.dtype==torch.bfloat16 and x.shape[-1]==896
        flat=x.reshape(-1,896).float();v=self.values()
        codes,_,scale=I.input_codes(flat)
        return G.readout(I.private_features(flat,v,codes,scale),v).reshape(x.shape)

    def forward(self,x):
        return self.fp32(x).to(x.dtype)


def load_stored(path,device,expected_sha=None):
    """Execute only this versioned archive; no source or checkpoint fallback."""
    if expected_sha is not None:assert P.digest(path)==expected_sha
    used=set();copied={}
    with safe_open(str(path),framework='pt',device='cpu') as archive:
        metadata=archive.metadata()
        assert metadata['format']==FORMAT
        assert metadata['diagnostic_only']=='true' and metadata['native_promotion_qualified']=='false'
        assert metadata['private_units']=='128' and metadata['input_arithmetic']=='M274_I16_ties_even_FP32_scale'
        assert metadata['script_sha256']==P.digest(Path(__file__))
        for name,sha in json.loads(metadata['helper_hashes_json']).items():assert P.digest(Path(name))==sha
        inventory=json.loads(metadata['tensor_inventory_json'])
        assert len(inventory)==725 and set(inventory)==set(archive.keys())
        for name,expected in inventory.items():
            value=archive.get_tensor(name)
            assert torch.isfinite(value).all() and record_tensor(value)==expected,name
        counts=json.loads(metadata['actual_function_counts_json'])
        assert len(counts)==24 and sum(counts)==30556 and sum(1280-c for c in counts)==164
        config_values=json.loads(metadata['config_json']);kind=config_values.pop('model_type')
        config=AutoConfig.for_model(kind,**config_values);torch.manual_seed(276276)
        model=AutoModelForCausalLM.from_config(config,dtype=torch.bfloat16,
            attn_implementation='sdpa').to(device).eval();model.config.use_cache=False
        assert len(list(model.named_parameters()))==290
        for layer in model.model.layers:layer.mlp=nn.Identity()
        with torch.no_grad():
            for name,param in model.named_parameters():
                organ=P.M24.classify(name,tuple(param.shape));assert organ!='ffn'
                key=name+'.bf16' if organ=='tied_head' else name;value=archive.get_tensor(key)
                assert value.shape==param.shape
                param.copy_(value.to(device=device,dtype=param.dtype))
                assert torch.equal(param.detach().cpu(),value.to(param.dtype))
                used.add(key);copied[organ]=copied.get(organ,0)+1
        del param,value
        assert copied=={'tied_head':1,'attention':96,'control':121}
        G.L.TABLE=archive.get_tensor('ffn.silu_table').to(device);used.add('ffn.silu_table');wrappers=[]
        for li,layer in enumerate(model.model.layers):
            prefix=f'ffn.{li}.';keys=[k for k in archive.keys() if k.startswith(prefix)];assert len(keys)==15
            values={k[len(prefix):]:archive.get_tensor(k).to(device) for k in keys};used.update(keys)
            assert values['private.ids'].shape==(128,) and values['private.ids'].dtype==torch.uint16
            ids=values['private.ids'].long()
            assert len(torch.unique(ids))==128 and int(ids.min())>=0 and int(ids.max())<4864
            for name in ('private.gate','private.up'):
                assert values[name].shape==(128,896) and values[name].dtype==torch.bfloat16
            bank={}
            for name in R.BANK_FIELDS:
                key=f'bank.{li}.{name}';bank[name]=archive.get_tensor(key).to(device);used.add(key)
            assert bank['a'].shape==(128,8,896) and bank['a'].dtype==torch.bfloat16
            assert bank['b'].shape==(counts[li],896,8) and bank['b'].dtype==torch.bfloat16
            assert bank['leaf_map'].shape==(1280,) and bank['leaf_map'].dtype==torch.int32
            assert int(bank['leaf_map'].min())==0 and int(bank['leaf_map'].max())==counts[li]-1
            assert len(torch.unique(bank['leaf_map']))==counts[li]
            layer.mlp=R.MappedConditional(StoredI16FFN(values),bank);wrappers.append(layer.mlp)
        proposal={}
        for name,suffix in (('codes','.q'),('scales','.scale')):
            key=R.L.X.HEAD+suffix;proposal[name]=archive.get_tensor(key).to(device);used.add(key)
        assert used==set(archive.keys()) and len(used)==725
        assert model.model.embed_tokens.weight.data_ptr()==model.lm_head.weight.data_ptr()
        for parameter in model.parameters():parameter.requires_grad_(False)
    return model,wrappers,proposal,{'parameters_from_archive':218,'random_FFN_parameters_removed':72,
        'tensors_consumed':len(used),'organs':copied,'all725_inventory_hashes_finite_exact':True,
        'all_non_FFN_copies_exact':True,'all_maps_surjective_and_in_range':True,
        'actual_function_counts':counts,'tied_head_pointer_equal':True,
        'source_weights_or_conditional_checkpoints_loaded':False,'cache_enabled':False,
        'diagnostic_only':True,'native_promotion_qualified':False}


def conditional_controls(wrapper,x):
    """Isolate the unchanged conditional term before rounding with the dense base."""
    flat=x.reshape(-1,896);parents,scores=wrapper.routes(flat)
    children=wrapper.child_route(flat,parents)
    gate=TF.softmax(scores,dim=-1).to(flat.dtype)
    a=wrapper.a[children//10].to(flat.dtype)
    aliases=wrapper.leaf_map[children].long();b=wrapper.b[aliases].to(flat.dtype)
    hidden=TF.silu(torch.einsum('nd,nkrd->nkr',flat,a))
    output=torch.einsum('nkr,nkdr->nkd',hidden,b)
    delta=(output*gate.unsqueeze(-1)).sum(dim=1).reshape_as(x)
    return {'parents':parents,'scores':scores,'children':children,'aliases':aliases,
            'selected_a':a,'selected_b':b,'conditional_contribution':delta}


def main():
    ap=argparse.ArgumentParser()
    for name in ('artifact','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();start=time.monotonic();stage='bindings';rows=[];fields=[];gates={};device=None
    assert all(not p.exists() for p in (args.artifact,args.out,args.out.with_suffix('.partial.json'),args.out.with_suffix('.failure.json')))
    args.artifact.parent.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(args.artifact.parent).free>=3*1024**3
    def partial():
        I.A.dump(args.out.with_suffix('.partial.json'),{'stage':stage,'fields':fields,'layers':rows,
            'gates':gates,'seconds':time.monotonic()-start})
    try:
        for path,sha in ((CORE,CORE_SHA),(EXPORT,EXPORT_SHA),(I16,I16_SHA),(PAIRED,PAIRED_SHA),
                         (I.FIXTURE,FIXTURE_SHA),(NATIVE,NATIVE_SHA),(QUANTIZER,QUANTIZER_SHA),
                         (G.C.VECTORS,G.C.VECTORS_SHA),(I.PRIOR,I.PRIOR_SHA)):
            assert P.digest(path)==sha,str(path)
        export=json.loads(EXPORT.read_text(encoding='utf-8'))
        i16=json.loads(I16.read_text(encoding='utf-8'));paired=json.loads(PAIRED.read_text(encoding='utf-8'))
        prior=json.loads(I.PRIOR.read_text(encoding='utf-8'))
        assert i16['decision']=='stop_fixed_I16_shared_operator_at_native_cost'
        assert paired['decision']=='stop_fixed_paired_I16_layout_at_numeric_or_cost'
        assert not i16['gates']['native_component_median_at_most10ms']
        assert all(v for k,v in i16['gates'].items() if k!='native_component_median_at_most10ms')
        assert P.digest(Path(I.__file__))==i16['script_sha256']
        assert P.digest(Path(I.A.__file__))==prior['script_sha256']
        assert P.digest(Path(R.__file__))==export['script_sha256']
        for path,sha in export['helper_sha256'].items():assert P.digest(Path(path))==sha
        assert P.digest(I.SOURCE)==i16['source_code_sha256']
        native_raw=NATIVE.read_bytes()
        assert struct.unpack_from('<8s3I',native_raw)==(b'M274ALL1',256,24,896)
        native=np.frombuffer(native_raw,dtype='<f4',offset=20).reshape(256,24,896)
        quant_raw=QUANTIZER.read_bytes()
        assert struct.unpack_from('<8s3I',quant_raw)==(b'M274QX01',256,24,896)
        quant=np.frombuffer(quant_raw,dtype='<i2',offset=20,count=256*24*896).reshape(256,24,896)
        qoffset=20+quant.nbytes
        inverse=np.frombuffer(quant_raw,dtype='<f4',offset=qoffset,count=256*24).reshape(256,24)
        scales=np.frombuffer(quant_raw,dtype='<f4',offset=qoffset+inverse.nbytes).reshape(256,24)
        raw=G.C.VECTORS.read_bytes()
        assert struct.unpack_from('<8s4I',raw)==(b'M125HX01',24,256,896,1280)
        states=np.frombuffer(raw,dtype='<u2',offset=24).reshape(256,24,896)
        device=G.Q.M.D.Q.setup();P.MAX_SECONDS=P.M17.MAX_SECONDS=600
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        stage='construct_only72_changed_private_fields';tensors={};old_inventory={}
        with safe_open(str(CORE),framework='pt',device='cpu') as previous:
            old_metadata=previous.metadata()
            for name in previous.keys():
                value=previous.get_tensor(name).contiguous()
                tensors[name]=value;old_inventory[name]=record_tensor(value)
                assert torch.isfinite(value).all(),name
                P.budget(start,device)
        assert len(tensors)==725
        segments=prior['segments'];assert len(segments)==361
        with I.FIXTURE.open('rb') as fixture:
            assert fixture.read(32)==struct.pack('<8s6I',b'M252PF01',24,896,4864,32,32,128)
            for segment in segments:
                assert fixture.tell()==segment['offset']
                raw=fixture.read(segment['bytes']);assert P.M17.sha(raw)==segment['sha256']
                value=I.from_raw(segment['name'],raw)
                key='ffn.silu_table' if segment['layer'] is None else f"ffn.{segment['layer']}.{segment['name']}"
                if segment['name'] not in PRIVATE:
                    assert torch.equal(value,tensors[key]),key
                elif segment['name']=='private.ids':
                    assert set(tensors[key].tolist()).issubset(value.tolist()),key
                    tensors[key]=value.contiguous()
                else:
                    li=segment['layer'];old_ids=tensors[f'ffn.{li}.private.ids']
                    # The archive's old32 IDs are recovered from the immutable control.
                    with safe_open(str(CORE),framework='pt',device='cpu') as previous:
                        kept=previous.get_tensor(f'ffn.{li}.private.ids').long()
                    index={int(unit):j for j,unit in enumerate(old_ids.tolist())}
                    selected=torch.tensor([index[int(unit)] for unit in kept.tolist()])
                    assert torch.equal(value[selected],tensors[key]),key
                    tensors[key]=value.contiguous()
                P.budget(start,device)
            assert not fixture.read(1)
        inventory={name:record_tensor(value) for name,value in tensors.items()}
        changed=[name for name in inventory if inventory[name]!=old_inventory[name]]
        assert set(changed)=={f'ffn.{li}.{name}' for li in range(24) for name in PRIVATE}
        assert len(changed)==72 and len(inventory)-len(changed)==653
        for name in inventory:
            fields.append({'key':name,**inventory[name],'unchanged259':name not in changed})
        payload=sum(value['bytes'] for value in inventory.values());assert payload==1329221892
        assert payload-sum(value['bytes'] for value in old_inventory.values())==8262144
        root=(P.ROOT/'benchmarks').resolve()
        helpers=dict(export['helper_sha256'])
        for module in list(sys.modules.values()):
            location=getattr(module,'__file__',None)
            if location:
                path=Path(location).resolve()
                if root in path.parents and path.suffix=='.py' and path!=Path(__file__).resolve():
                    helpers[str(path)]=P.digest(path)
        metadata={**old_metadata,'format':FORMAT,'experiment':'METH-276',
            'diagnostic_only':'true','native_promotion_qualified':'false','private_units':'128',
            'input_arithmetic':'M274_I16_ties_even_FP32_scale','dense_output_dtype':'bfloat16',
            'original259_archive_sha256':CORE_SHA,'original259_export_sha256':EXPORT_SHA,
            'original259_metadata_json':json.dumps(old_metadata,sort_keys=True),
            'source_operator_sha256':FIXTURE_SHA,'source_operator_result_sha256':I16_SHA,
            'paired_layout_stopped_result_sha256':PAIRED_SHA,'quantizer_control_sha256':QUANTIZER_SHA,
            'actual_CPU_6144_output_sha256':NATIVE_SHA,'vectors_sha256':G.C.VECTORS_SHA,
            'changed_fields_json':json.dumps(sorted(changed)),
            'tensor_inventory_json':json.dumps(inventory,sort_keys=True),
            'script_sha256':P.digest(Path(__file__)),'helper_hashes_json':json.dumps(helpers,sort_keys=True)}
        stage='save_readback_all725';partial()
        save_file(tensors,str(args.artifact),metadata=metadata)
        assert args.artifact.stat().st_size<3*1024**3
        with safe_open(str(args.artifact),framework='pt',device='cpu') as archive:
            assert archive.metadata()==metadata and set(archive.keys())==set(tensors)
            for name,value in tensors.items():
                stored=archive.get_tensor(name)
                assert torch.isfinite(stored).all() and torch.equal(stored,value),name
                assert record_tensor(stored)==inventory[name],name
                P.budget(start,device)
        artifact_sha=P.digest(args.artifact)
        print(json.dumps({'stage':stage,'artifact_sha256':artifact_sha,'payload_bytes':payload}),flush=True)
        del tensors;gc.collect();torch.cuda.empty_cache()
        stage='standalone_versioned_loader';partial()
        model,wrappers,proposal,loaded=load_stored(args.artifact,device,artifact_sha)
        old_model,old_wrappers,old_proposal,old_loaded=R.load_stored(CORE,device,CORE_SHA)
        for name in proposal:assert torch.equal(proposal[name],old_proposal[name]),name
        assert loaded['actual_function_counts']==old_loaded['actual_function_counts']
        stage='all6144_base_route_alias_and_isolated_conditional_controls';partial()
        with torch.inference_mode():
            for li,(new,old) in enumerate(zip(wrappers,old_wrappers)):
                x=torch.from_numpy(states[:,li].copy()).view(torch.bfloat16).to(device)
                v=new.base.values();codes,inv,scale=I.input_codes(x.float())
                assert np.array_equal(codes.cpu().numpy(),quant[:,li])
                assert np.array_equal(inv,inverse[:,li]) and np.array_equal(scale,scales[:,li])
                expected=G.readout(I.private_features(x.float(),v,codes,scale),v)
                actual=new.base.fp32(x);assert torch.equal(expected,actual)
                rounded=new.base(x);assert torch.equal(rounded,expected.bfloat16())
                reference=actual.cpu().numpy().astype(np.float64)
                errors=np.linalg.norm(native[:,li].astype(np.float64)-reference,axis=1)/np.linalg.norm(reference,axis=1)
                assert np.isfinite(errors).all()
                old_controls=conditional_controls(old,x);new_controls=conditional_controls(new,x)
                for name in old_controls:assert torch.equal(new_controls[name],old_controls[name]),(li,name)
                composed=new(x)
                assert torch.equal(composed,rounded+new_controls['conditional_contribution'])
                assert torch.isfinite(composed).all()
                rows.append({'layer':li,'states':256,'loaded_FP32_equation_exact274':True,
                    'BF16_return_exact_equation_rounding':True,'quantizer_exact_actual_CPU_control':True,
                    'routes_scores_children_aliases_selected_AB_and_isolated_delta_exact259':True,
                    'composed_forward_exact_base_plus_isolated_delta':True,
                    'FP32_output_sha256':R.X.tensor_hash(actual),'BF16_output_sha256':R.X.tensor_hash(rounded),
                    'conditional_delta_sha256':R.X.tensor_hash(new_controls['conditional_contribution']),
                    'composed_output_sha256':R.X.tensor_hash(composed),
                    'native_relative_l2':errors.tolist()})
                print(json.dumps({'layer':li,'completed_states':len(rows)*256}),flush=True)
                partial();P.budget(start,device)
        errors=np.asarray([row['native_relative_l2'] for row in rows])
        numeric={'median6144':float(np.median(errors)),'max6144':float(errors.max()),
            'median384':float(np.median(errors[:,:16])),'max384':float(errors[:,:16].max())}
        assert numeric==i16['native_numeric_summary']
        gates={'all_bindings_and_stopped_cost_decisions_preserved':True,'all725_finite_inventory_readback_exact':True,
            'only72_private_fields_changed_and_all653_others_byte_exact259':True,
            'all72_private_fields_exact271_and_old32_retained':True,'payload_exact1329221892_and_extra8262144':True,
            'versioned_loader_consumes_all725_no_source_or_checkpoint_fallback':True,
            'all6144_loaded_base_equation_and_BF16_return_exact274':True,
            'all6144_quantizers_exact_frozen_actual_CPU_controls':True,
            'all6144_same_input_routes_aliases_selected_AB_and_isolated_conditional_exact259':True,
            'all6144_composed_forwards_exact_base_plus_conditional':True,
            'original384_native_numeric_limits':numeric['median384']<=1e-4 and numeric['max384']<=5e-4,
            'all6144_native_numeric_limits':numeric['median6144']<=1e-4 and numeric['max6144']<=5e-4}
        torch.cuda.synchronize(device)
        result={'experiment':'METH-276-diagnostic-complete-I16-private128-archive',
            'artifact':{'path':str(args.artifact.resolve()),'sha256':artifact_sha,
                'bytes':args.artifact.stat().st_size,'payload_bytes':payload,
                'metadata_header_bytes':args.artifact.stat().st_size-payload,'tensor_count':725},
            'original_archive_sha256':CORE_SHA,'original_export_sha256':EXPORT_SHA,
            'source_fixture_sha256':FIXTURE_SHA,'i16_result_sha256':I16_SHA,'paired_stop_result_sha256':PAIRED_SHA,
            'native_all_output_sha256':NATIVE_SHA,'quantizer_sha256':QUANTIZER_SHA,'vectors_sha256':G.C.VECTORS_SHA,
            'script_sha256':P.digest(Path(__file__)),'helper_sha256':helpers,
            'fields':fields,'layers':rows,'load_record':loaded,'native_numeric_summary':numeric,'gates':gates,
            'runtime':P.budget(start,device),'diagnostic_only':True,'native_promotion_qualified':False,
            'decision':'diagnostic_complete_archive_pass_freeze_consumed_whole_model_regression' if all(gates.values()) else 'stop_diagnostic_complete_archive_at_composition',
            'scope':'Archive composition on consumed states only. Prior cost failures and259 semantic stop retained. No whole-quality/native-rate/new-useful-n/RAM/DRAM/family transfer proof.'}
        I.A.dump(args.out,result)
        print(json.dumps({key:result[key] for key in ('decision','artifact','gates','runtime')}),flush=True)
    except BaseException as failure:
        partial();I.A.dump(args.out.with_suffix('.failure.json'),{'stage':stage,
            'error':type(failure).__name__+': '+str(failure),'seconds':time.monotonic()-start,
            'completed_layers':len(rows),'fields':fields,'gates':gates,
            'artifact_exists':args.artifact.exists(),'script_sha256':P.digest(Path(__file__))})
        raise


if __name__=='__main__':main()
