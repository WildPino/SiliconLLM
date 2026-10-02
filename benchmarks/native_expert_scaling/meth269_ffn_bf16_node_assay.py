#!/usr/bin/env python3
"""Separate source BF16 nodes,LUT,stored-weight and fixed-readout error; no fit."""
import argparse
import json
from pathlib import Path
import struct
import time
import numpy as np
import torch
from torch.nn import functional as TF
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
import meth268_generation_route_replay as A

P,L,R,Q=A.P,A.L,A.R,A.Q
G=R.G
TRACE=P.DOC/'meth268_generation_route_replay_repair1_result.json'
TRACE_SHA='ed3d8289d5e729d59d4c545bff88ab0d294056d777e0b5689766ccb012fd1abe'
METHODS=('source_fp32_exact','source_fp32_lut','source_bf16_lut','candidate_existing_fp32',
    'candidate_bf16_nodes_lut','candidate_bf16_nodes_exact_silu',
    'candidate_oracle_source_bf16_features','candidate_oracle_source_fp32_features')


def measure(reference,value):
    assert reference.shape==value.shape==(256,896) and torch.isfinite(value).all()
    error=(reference.float()-value.float()).square().sum(1);energy=reference.float().square().sum(1)
    assert (energy>0).all();ratios=error/energy
    return {'squared_error':float(error.sum()),'reference_energy':float(energy.sum()),
        'energy_normalized_squared_error':float(error.sum()/energy.sum()),
        'mean_state_relative_squared_error':float(ratios.mean()),
        'p95_state_relative_squared_error':float(torch.quantile(ratios,.95)),
        'max_state_relative_squared_error':float(ratios.max()),
        'BF16_output_values_differ':int((reference!=value).sum()),'output_values':reference.numel()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists();start=time.monotonic();stage='bindings';rows=[]
    def save():
        args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'layers':rows,
            'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
    try:
        for path,sha in ((TRACE,TRACE_SHA),(Q.D.CORE,Q.D.CORE_SHA),(Q.D.EXPORT,Q.D.EXPORT_SHA),
                         (G.C.VECTORS,G.C.VECTORS_SHA)):
            assert P.digest(path)==sha
        trace=json.loads(TRACE.read_text(encoding='utf-8'));assert all(trace['gates'].values())
        export=json.loads(Q.D.EXPORT.read_text(encoding='utf-8'));assert P.digest(Path(R.__file__))==export['script_sha256']
        for path,sha in export['helper_sha256'].items():assert P.digest(Path(path))==sha
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA
        config=AutoConfig.from_pretrained(P.M42.MODEL,revision=P.M42.REV,local_files_only=True)
        assert config.model_type=='qwen2' and config.hidden_act=='silu'
        vector_raw=G.C.VECTORS.read_bytes();assert struct.unpack_from('<8s4I',vector_raw)==(b'M125HX01',24,256,896,1280)
        states=np.frombuffer(vector_raw,dtype='<u2',offset=24).reshape(256,24,896)
        device=G.Q.M.D.Q.setup();P.MAX_SECONDS=P.M17.MAX_SECONDS=5*60
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        stage='fixed6144_source_state_arithmetic_and_codec_assay'
        with torch.inference_mode(),safe_open(str(source),framework='pt',device='cpu') as donor,safe_open(str(Q.D.CORE),framework='pt',device='cpu') as saved:
            G.L.TABLE=saved.get_tensor('ffn.silu_table').to(device)
            for li in range(24):
                x=torch.from_numpy(states[:,li].copy()).view(torch.bfloat16).to(device);xf=x.float()
                w={name:donor.get_tensor(f'model.layers.{li}.mlp.{name}_proj.weight').to(device) for name in ('gate','up','down')}
                assert all(value.dtype==torch.bfloat16 for value in w.values())
                prefix=f'ffn.{li}.';keys=[key for key in saved.keys() if key.startswith(prefix)];assert len(keys)==15
                v={key[len(prefix):]:saved.get_tensor(key).to(device) for key in keys}
                mlp=Qwen2MLP(config).to(device=device,dtype=torch.bfloat16)
                for name,value in w.items():getattr(mlp,name+'_proj').weight.copy_(value)
                gb=TF.linear(x,w['gate']);ub=TF.linear(x,w['up']);phi_b=TF.silu(gb)*ub
                reference=TF.linear(phi_b,w['down']);assert torch.equal(reference,mlp(x))
                gf=TF.linear(xf,w['gate'].float());uf=TF.linear(xf,w['up'].float());phi_f=TF.silu(gf)*uf
                outputs={
                    'source_fp32_exact':TF.linear(phi_f,w['down'].float()).bfloat16(),
                    'source_fp32_lut':TF.linear(G.L.silu_lookup(gf)*uf,w['down'].float()).bfloat16(),
                    'source_bf16_lut':TF.linear(G.L.silu_lookup(gb.float()).bfloat16()*ub,w['down']),
                    'candidate_existing_fp32':G.readout(G.private_features(xf,v),v).bfloat16(),
                    'candidate_oracle_source_bf16_features':G.readout(phi_b.float(),v).bfloat16(),
                    'candidate_oracle_source_fp32_features':G.readout(phi_f,v).bfloat16()}
                assert torch.equal(outputs['candidate_existing_fp32'],R.X.StoredFFN(v)(x))
                cg=G.L.row_linear(xf,v['gate.q'],v['gate.scale']);cu=G.L.row_linear(xf,v['up.q'],v['up.scale'])
                ids=v['private.ids'].long();cg[:,ids]=TF.linear(xf,v['private.gate'].float());cu[:,ids]=TF.linear(xf,v['private.up'].float())
                cg=cg.bfloat16();cu=cu.bfloat16()
                outputs['candidate_bf16_nodes_lut']=G.readout((G.L.silu_lookup(cg.float()).bfloat16()*cu).float(),v).bfloat16()
                outputs['candidate_bf16_nodes_exact_silu']=G.readout((TF.silu(cg)*cu).float(),v).bfloat16()
                assert set(outputs)==set(METHODS)
                rows.append({'layer':li,'states':256,'source_BF16_equation_exact_HF_MLP':True,
                    'existing_FFN_exact_archived_loader':True,
                    'methods':{name:measure(reference,outputs[name]) for name in METHODS},
                    'relative_to_source_FP32_output':{name:measure(outputs['source_fp32_exact'],outputs[name]) for name in METHODS}})
                save();print(json.dumps({'layer':li,'completed':len(rows),'runtime':P.budget(start,device)}),flush=True)
                del mlp,outputs,x,xf,w,v,gb,ub,phi_b,reference,gf,uf,phi_f,cg,cu
        summary={}
        for name in METHODS:
            cells=[r['methods'][name] for r in rows]
            summary[name]={'energy_normalized_squared_error':sum(c['squared_error'] for c in cells)/sum(c['reference_energy'] for c in cells),
                'mean_state_relative_squared_error':float(np.mean([c['mean_state_relative_squared_error'] for c in cells])),
                'maximum_layer_energy_normalized_squared_error':max(c['energy_normalized_squared_error'] for c in cells),
                'BF16_output_values_differ':sum(c['BF16_output_values_differ'] for c in cells)}
        result={'experiment':'METH-269-source-BF16-nodes-LUT-stored-FFN-consumed-assay',
            'trace_result_sha256':TRACE_SHA,'artifact_sha256':Q.D.CORE_SHA,'source_sha256':P.M57.MODEL_SHA,
            'vectors_sha256':G.C.VECTORS_SHA,'states':6144,'layers':rows,'summary':summary,
            'gates':{'all_actual_BF16_source_equations_exact_HF_MLP':True,
                'all_current_FFN_equations_exact_archived_loader':True,'all_values_finite':True},
            'runtime':P.budget(start,device),'script_sha256':P.digest(Path(__file__)),
            'decision':'consumed_precision_decomposition_complete_no_candidate_promotion',
            'scope':'Fixed previously consumed component inputs,not generation trajectories/new quality. Same encoded fields,alternative arithmetic/oracle features only,no fit/new artifact/native cost/rate/large-n/multifamily proof.267 stop unchanged.'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({key:result[key] for key in ('decision','summary','gates','runtime')}),flush=True)
    except BaseException as failure:
        save();args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':type(failure).__name__+': '+str(failure),
            'completed_layers':len(rows),'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8');raise


if __name__=='__main__':main()
