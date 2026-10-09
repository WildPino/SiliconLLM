"""Versioned compact target with two common and eight selected private functions.

The original private/core/head implementation is reused. Float32 common functions
are offline warmup scaffolding; ternary/AQ63 is the common deployment mode.
"""
import torch
from torch import nn
from torch.nn import functional as F
import chatbot_hybrid_target as base

SCHEMA='COMPACT_FALCON_SHARED_PRIVATE_TARGET_V1'
COMMON_COUNT=2
COMMON_SEED_PRIVATE_IDS=(0,36)
PARAMETERS=259669760


class CommonFunctions(nn.Module):
    def __init__(self,precision):
        super().__init__()
        self.gate=nn.Parameter(torch.empty(COMMON_COUNT,base.H,base.D))
        self.up=nn.Parameter(torch.empty(COMMON_COUNT,base.H,base.D))
        self.down=nn.Parameter(torch.empty(COMMON_COUNT,base.D,base.H))
        self.gate_scale=nn.Parameter(torch.ones(COMMON_COUNT,base.H))
        self.up_scale=nn.Parameter(torch.ones(COMMON_COUNT,base.H))
        self.down_scale=nn.Parameter(torch.ones(COMMON_COUNT,base.D))
        self.set_precision(precision)

    def set_precision(self,precision):
        assert precision in ('float32','ternary')
        self.precision=precision
        for value in (self.gate_scale,self.up_scale,self.down_scale):
            value.requires_grad_(precision=='ternary')

    def forward(self,x):
        shape=x.shape; x=x.reshape(-1,base.D); result=torch.zeros_like(x)
        for i in range(COMMON_COUNT):
            if self.precision=='float32':
                g=F.linear(x,self.gate[i]); u=F.linear(x,self.up[i])
                y=F.linear(F.silu(g)*u,self.down[i])
            else:
                g=base.ternary_linear(x,self.gate[i],self.gate_scale[i])
                u=base.ternary_linear(x,self.up[i],self.up_scale[i])
                y=base.ternary_linear(F.silu(g)*u,self.down[i],self.down_scale[i])
            result=result+y
        return result.reshape(shape)


class Functions(nn.Module):
    def __init__(self,private,precision):
        super().__init__(); self.private=private; self.common=CommonFunctions(precision)

    def forward(self,x):
        return self.private(x)+self.common(x)


class Target(base.Target):
    def __init__(self,source,common_precision='float32'):
        super().__init__(source)
        for layer in self.layers:
            layer.banks=Functions(layer.banks,common_precision)

    def set_common_precision(self,precision):
        for layer in self.layers: layer.banks.common.set_precision(precision)


def canonical_private_name(name):
    assert '.banks.common.' not in name
    return name.replace('.banks.private.','.banks.')


def mapped_model(original):
    """Copy private fields by identity; seed common features in the adapted frame."""
    assert len(original)==211
    result={name.replace('.banks.','.banks.private.'):value for name,value in original.items()}
    indices=list(COMMON_SEED_PRIVATE_IDS)
    for i in range(base.L):
        old=f'layers.{i}.banks.'; new=f'layers.{i}.banks.common.'
        for label in ('gate','up','gate_scale','up_scale'):
            result[new+label]=original[old+label][indices].clone()
        result[new+'down']=torch.zeros(COMMON_COUNT,base.D,base.H,dtype=torch.float32)
        result[new+'down_scale']=torch.full((COMMON_COUNT,base.D),1e-3,dtype=torch.float32)
    assert len(result)==283 and sum(v.numel() for v in result.values())==PARAMETERS
    return result


def mapped_optimizer(original_model,original_optimizer,new_parameter_names):
    """Preserve every original Adam slot; new common slots start lazily at step0."""
    groups=original_optimizer['param_groups']; assert len(groups)==1
    old_names=list(original_model); old_ids=groups[0]['params']
    assert len(old_names)==len(old_ids)==len(original_optimizer['state'])==211
    private_names=[canonical_private_name(name) for name in new_parameter_names if '.banks.common.' not in name]
    assert private_names==old_names,'parameter enumeration mismatch'
    old_by_name=dict(zip(old_names,old_ids,strict=True)); slots={}; mapping=[]
    for index,name in enumerate(new_parameter_names):
        if '.banks.common.' in name:
            continue
        canonical=canonical_private_name(name); old_id=old_by_name[canonical]
        slots[index]=original_optimizer['state'][old_id]
        mapping.append(dict(original_name=canonical,new_name=name,original_id=old_id,new_id=index))
    group=dict(groups[0]); group['params']=list(range(len(new_parameter_names)))
    assert len(slots)==211 and len(new_parameter_names)==283
    return dict(state=slots,param_groups=[group]),mapping


def make_optimizer(target,packet):
    names=[n for n,_ in target.named_parameters()]
    assert names==packet['optimizer_parameter_names']
    group=packet['optimizer']['param_groups'][0]
    optimizer=torch.optim.AdamW(target.parameters(),lr=group['lr'],betas=tuple(group['betas']),
                  eps=group['eps'],weight_decay=group['weight_decay'],foreach=False)
    optimizer.load_state_dict(packet['optimizer'])
    return optimizer
