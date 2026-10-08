"""New source-compatible compact SSM/SWA/ternary target budgets, integers only."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main(a):
    start=time.monotonic()
    raw=a.metadata.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==a.metadata_sha
    meta=json.loads(raw)
    source=meta['config_shape']
    assert source['D']==2048 and source['L']==24 and source['ffn_h']==4608 and source['N']==256
    targets=[]
    for layers,swa,bank in ((12,2,36),(12,2,72),(6,1,72),(6,1,144)):
        d,m,n,head,h,k,q,window,conv=512,768,256,64,128,8,32,128,4
        ssm=layers-swa
        heads=m//head
        conv_dim=m+2*n
        # Mamba2 source-compatible direct delta/convolved X,B,C; compact A/D by head.
        ssm_matrix=ssm*d*(2*m+2*n+heads+m)
        swa_matrix=swa*4*d*d
        core_matrix=ssm_matrix+swa_matrix
        ssm_scalars=ssm*(conv_dim*(conv+1)+3*heads+m)
        norms=(2*layers+1)*d
        head_elements=source['V']*d
        embedding_elements=source['V']*d
        experts=3*layers*bank*d*h
        selected=3*layers*k*d*h
        scales=layers*bank*(2*h+d)
        flat=layers*bank*(d+1)
        nodes=(bank+31)//32
        tree_internal=nodes+(1 if nodes>1 else 0)
        tree_params=layers*(d*q+tree_internal*32*(q+1))
        tree_products=layers*(d*q+32*q*(1+min(k,nodes)))
        fixed=core_matrix+ssm_scalars+norms+head_elements+embedding_elements
        dense_parameters=fixed+experts+scales+tree_params
        codes=experts//2
        active_codes=selected//2
        active_scales=layers*k*(2*h+d)
        physical_state=4*(ssm*m*n+ssm*conv_dim*conv+swa*2*window*d)
        packed_model=4*(fixed+scales+tree_params)+codes
        logical_decode=4*(core_matrix+ssm_scalars+norms+head_elements+tree_params+active_scales)+active_codes+2*physical_state
        record=dict(D=d,L=layers,SSM_layers=ssm,SWA_layers=swa,SWA_window=window,
           SSM_D=m,N=n,SSM_head_dim=head,SSM_heads=heads,gated_RMS=True,V=source['V'],expert_h=h,k=k,n_per_layer=bank,
           source_row_slots_preserved_count_only=layers*bank*h==source['L']*source['ffn_h'],
           row_slot_meaning='Same aggregate feature slot count does NOT mean preserved functions or knowledge',
           core_matrix_products=core_matrix,full_head_products=head_elements,selected_expert_products=selected,
           flat_router_products=layers*bank*d,proposed_tree_products=tree_products,
           full_decode_matrix_products_with_tree=core_matrix+head_elements+selected+tree_products,
           dense_trainable_coefficients=dense_parameters,dense_F32_Adam_array_bytes=16*dense_parameters,
           extra_teacher_BF16_file_bytes=meta['BF16_weight_file_bytes'],
           target_Adam_plus_teacher_file_lower_envelope_bytes=16*dense_parameters+meta['BF16_weight_file_bytes'],
           packed_model_coefficient_bytes=packed_model,legacy_expert_reference_extra_bytes=5*experts,
           selected_expert_codes_and_scales_bytes=active_codes+4*active_scales,
           recurrent_SWA_cache_bytes=physical_state,
           logical_decode_coefficients_and_state_bytes=logical_decode,
           expert_activation='Justified prospective ternary SwiGLU scalar gate/signed up, same packed LUT matrix machinery; gated-dReLU remains anchor, source ReLU skip identities invalid',
           required_learned_changes=['width2048->512','SSM3072->768','block24->'+str(layers),'dense parallel attention24->bounded SWA'+str(swa),
             'source functions from all layers into conditional bank','ternary coefficients/activation quantization','whole own-state output recovery'],
           implementation_status='Prospective budget; target adapter/learner/LUT scalar variant/router are missing',
           omitted_training_costs=['activations','teacher computed buffers/cache','target buffers/codes','optimizer step temporaries','checkpoint copies','runtime','CPU transfers'],
           precision='F32 core/organs baseline, byte-pair ternary experts, row scales F32, int8 activation',
           deployment_rates_quality_physical_DRAM='UNMEASURED')
        targets.append(record)
    result=dict(schema='HYBRID_COMPACT_TARGET_BUDGET_V1',freeze=a.freeze,metadata_sha256=a.metadata_sha,
      source_revision=meta['revision'],targets=targets,
      source_gating_note='Source SSM gate then RMS; require original scan plus group1 norm extension, not Tiny no-norm shortcut',
      exponential_note='Source A/delta constant within head: one exp(delta*A) per head versus original per-state exp. Hoisting is algebraically exact; float/C timing not yet qualified',
      tree_note='New prospective fanout32/beam8 count only; CPU LUT IDs plus normalized masses and useful-n remain unimplemented/unproven',
      selected_first_candidate_index=1,
      decision='D512/L12/SSM10/SWA2/n72/k8/h128 retains source aggregate FFN slot count and costs bounded active geometry; actual memory/quality need a pilot',
      elapsed_seconds=time.monotonic()-start)
    with a.out.open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(selected=targets[1],seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--metadata-sha',required=True)
    p.add_argument('--freeze',required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
