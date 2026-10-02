#!/usr/bin/env python3
"""Read-only actual FP32/BF16 bank/real-state diversity, no quality targets."""
import argparse
import json
import struct
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as TF
import meth257_full_source_core_export as X

P=X.P
FAIL=P.DOC/'meth257_full_source_core_export_result.failure.json'
FAIL_SHA='fc0fb83cb370a165b89fc3616224a47648077aefa42c13aa766e531329507fcb'


class Placeholder(nn.Module):
    def __init__(self):
        super().__init__();self.model=nn.Module()
        self.model.layers=nn.ModuleList([nn.Module() for _ in range(24)])
        for layer in self.model.layers:layer.mlp=nn.Identity()


def bank_hash(a,b):
    if not bool(b.any()):return P.M17.sha(b'exact-zero-correction')
    return P.M17.sha(bytes.fromhex(X.tensor_hash(a)+X.tensor_hash(b)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args();assert not args.out.exists();start=time.monotonic();stage='bindings';layers=[]
    try:
        for path,sha in ((FAIL,FAIL_SHA),(P.M122.SPECIALIZED,P.M122.SPECIALIZED_SHA),
                         (P.M122.TRAINING,P.M57.TRAINING_SHA),(X.G.C.VECTORS,X.G.C.VECTORS_SHA)):
            assert P.digest(path)==sha
        failure=json.loads(FAIL.read_text());assert failure['stage']=='effective_BF16_centered_bank_conservation'
        assert failure['bank_rows']==[] and len(failure['source_segments'])==361
        parent=json.loads(P.M122.TRAINING.read_text())['checkpoints']['512'];assert P.digest(parent['path'])==P.M57.CHECKPOINT_SHA
        device=X.G.Q.M.D.Q.setup();P.MAX_SECONDS=P.M17.MAX_SECONDS=20*60
        torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        temporary=Placeholder();X.L.D.H.load_centered(temporary,device,parent['path'])
        raw_child=torch.load(P.M122.SPECIALIZED,map_location='cpu',weights_only=False)
        raw_parent=torch.load(parent['path'],map_location='cpu',weights_only=False)
        raw=X.G.C.VECTORS.read_bytes();assert struct.unpack_from('<8s4I',raw)==(b'M125HX01',24,256,896,1280)
        states=np.frombuffer(raw,dtype='<u2',offset=24).reshape(256,24,896)
        stage='actual_bank_coefficients_and_fixed_real_state_probe'
        for li,layer in enumerate(temporary.model.layers):
            wrapper=layer.mlp;a=wrapper.a.detach().view(128,10,8,896);assert torch.equal(a,a[:,:1].expand_as(a))
            a32=a[:,0].cpu().contiguous();b32=wrapper.b.detach().cpu().contiguous()
            a16=a32.bfloat16();b16=b32.bfloat16();raw_b=raw_child['expert_state'][li]['b'].view(128,10,896,8)
            expected_parent=raw_parent['expert_state'][li]['b'].to(device)
            mean_error=float((wrapper.b.view(128,10,896,8).mean(1)-expected_parent).abs().max());assert mean_error<=1e-7
            # Same first64 original source states for every child of every parent.
            x=torch.from_numpy(states[:64,li].copy()).view(torch.bfloat16).to(device)
            signatures32=[];signatures16=[];groups=[];minimum=float('inf')
            for parent_id in range(128):
                raw_codes=[X.tensor_hash(v) for v in raw_b[parent_id]]
                base=parent_id*10
                codes32=[bank_hash(a32[parent_id],b32[base+c]) for c in range(10)]
                codes16=[bank_hash(a16[parent_id],b16[base+c]) for c in range(10)]
                nraw,n32,n16=len(set(raw_codes)),len(set(codes32)),len(set(codes16));assert n16<=n32<=nraw
                signatures32.extend(codes32);signatures16.extend(codes16)
                hidden=TF.silu(TF.linear(x,a16[parent_id].to(device)))
                outputs=[TF.linear(hidden,b16[base+c].to(device)) for c in range(10)]
                probe=[X.tensor_hash(v) for v in outputs]
                denominator=x.double().norm().clamp_min(1e-12)
                different=[float((outputs[i].double()-outputs[j].double()).norm()/denominator)
                    for i in range(10) for j in range(i) if codes16[i]!=codes16[j]]
                distance=min(different) if different else None
                if distance is not None:minimum=min(minimum,distance)
                groups.append({'parent':parent_id,'raw_child_B_distinct':nraw,'centered_FP32_function_parameters_distinct':n32,
                    'effective_BF16_function_parameters_distinct':n16,'fixed64_real_state_BF16_output_distinct':len(set(probe)),
                    'minimum_relative_correction_distance_between_distinct_codes':distance,
                    'raw_zero_child_B_count':sum(not bool(v.any()) for v in raw_b[parent_id]),
                    'effective_zero_child_B_count':sum(not bool(v.any()) for v in b16[base:base+10]),
                    'FP32_signatures':codes32,'BF16_signatures':codes16})
            layers.append({'layer':li,'route_labels':1280,'shared_parent_A':128,
                'FP32_actual_function_parameter_count':len(set(signatures32)),
                'BF16_effective_function_parameter_count':len(set(signatures16)),
                'within_parent_raw_distinct_sum':sum(g['raw_child_B_distinct'] for g in groups),
                'within_parent_FP32_distinct_sum':sum(g['centered_FP32_function_parameters_distinct'] for g in groups),
                'within_parent_BF16_distinct_sum':sum(g['effective_BF16_function_parameters_distinct'] for g in groups),
                'all10_distinct_BF16_parent_groups':sum(g['effective_BF16_function_parameters_distinct']==10 for g in groups),
                'groups_with_preexisting_raw_duplicates':sum(g['raw_child_B_distinct']<10 for g in groups),
                'groups_losing_distinctness_at_FP32_centering':sum(g['centered_FP32_function_parameters_distinct']<g['raw_child_B_distinct'] for g in groups),
                'groups_losing_distinctness_at_BF16_cast':sum(g['effective_BF16_function_parameters_distinct']<g['centered_FP32_function_parameters_distinct'] for g in groups),
                'minimum_distinct_code_probe_relative_l2':None if minimum==float('inf') else minimum,
                'original_parent_mean_max_abs_error':mean_error,'source_states_sha256':X.tensor_hash(x),
                'groups':groups})
            args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'layers':layers},indent=2)+'\n',encoding='utf-8')
            P.budget(start,device)
        actual=[r['BF16_effective_function_parameter_count'] for r in layers]
        result={'experiment':'METH-258-read-only-effective-centered-bank-function-diversity',
            'failed_export_sha256':FAIL_SHA,'parent_checkpoint_sha256':P.M57.CHECKPOINT_SHA,
            'child_checkpoint_sha256':P.M122.SPECIALIZED_SHA,'real_source_vectors_sha256':X.G.C.VECTORS_SHA,
            'layers':layers,'summary':{'route_labels_per_layer':1280,'actual_BF16_parameter_counts_by_layer':actual,
                'actual_BF16_parameter_count_min':min(actual),'actual_BF16_parameter_count_max':max(actual),
                'total_preexisting_duplicate_parent_groups':sum(r['groups_with_preexisting_raw_duplicates'] for r in layers),
                'total_FP32_centering_merge_groups':sum(r['groups_losing_distinctness_at_FP32_centering'] for r in layers),
                'total_BF16_cast_merge_groups':sum(r['groups_losing_distinctness_at_BF16_cast'] for r in layers)},
            'gates':{'all_source_checkpoint_vector_failure_bindings':True,'all24_complete_bank_and64_state_audits':len(layers)==24,
                'all_parent_A_sibling_equality_exact':True,'all_actual_FP32_centered_parent_means_le_1e_minus_7':True,
                'no_signature_count_increase_through_centering_or_BF16_cast':True},
            'runtime':P.budget(start,device),'script_sha256':P.digest(Path(__file__)),
            'decision':'effective_diversity_measured_require_honest_unique_function_storage',
            'scope':'Read-only fixed existing bank,not new fit/function perturbation or quality screen. Parameter counts/probes do not prove independent semantic count gain,n/RAM/general routing quality or full-model accepted speed.'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:result[k] for k in ('summary','gates','runtime','decision')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),'layers':layers,
            'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8');raise


if __name__=='__main__':main()
