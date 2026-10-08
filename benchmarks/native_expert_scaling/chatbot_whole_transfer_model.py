"""Connect initialized compact SwiGLU blocks to a complete frozen Qwen core.

No model loads or forwards at import. Installation is a conversion component;
it does not initialize ALL24 blocks or qualify transfer, export or native chat.
"""
from chatbot_compact_geometry import validate


def build_mlp_adapter(spec, block, layer_id):
    import torch
    from torch import nn
    validate(spec)
    assert isinstance(layer_id,int) and 0<=layer_id<spec['layers']
    assert block.initialized, 'Uninitialized conditional block cannot enter the whole model'
    device=block.shared_g.device;d=spec['hidden_size']
    for value in block.parameters():
        assert value.device==device and value.dtype==torch.float32 and value.is_contiguous() and torch.isfinite(value).all()
    for value in block.buffers():
        assert value.device==device and value.dtype==torch.float32 and value.is_contiguous() and torch.isfinite(value).all()
    assert (block.shared_g.shape,block.shared_u.shape,block.shared_b.shape)==((512,d),(512,d),(d,512))
    assert len(block.leaf_g)==len(block.leaf_u)==len(block.leaf_b)==16
    for gate,up,down in zip(block.leaf_g,block.leaf_u,block.leaf_b):
        assert (gate.shape,up.shape,down.shape)==((128,d),(128,d),(d,128))

    class CompactMLP(nn.Module):
        def __init__(self):
            super().__init__();self.compact=block;self.layer_id=layer_id

        def forward(self,hidden_states):
            assert hidden_states.ndim==3 and hidden_states.shape[-1]==d
            assert hidden_states.dtype in (torch.bfloat16,torch.float32) and hidden_states.device==device
            assert torch.isfinite(hidden_states).all()
            # Source attention/residual stream stays BF16. The compact nonlinear
            # field and routing run F32; the output boundary casts to source dtype.
            flat=hidden_states.reshape(-1,d).to(torch.float32).contiguous()
            result=self.compact(flat)
            assert result.shape==flat.shape and result.dtype==torch.float32 and torch.isfinite(result).all()
            return result.reshape_as(hidden_states).to(hidden_states.dtype)

    return CompactMLP()


def install_compact_model(model, blocks, spec):
    """Validate every block/core invariant, then replace ALL24 MLPs.

    Caller supplies a separate student model. No dense donor FFN fallback is
    retained by any adapter. All source attention/head/norm parameters freeze;
    all shared/private G/U/B train jointly. Router geometry stays frozen F32.
    """
    import torch
    validate(spec)
    cfg=model.config
    assert (cfg.model_type,cfg.hidden_size,cfg.num_hidden_layers,cfg.vocab_size,cfg.intermediate_size)==('qwen2',896,24,151936,4864)
    assert cfg.num_attention_heads==14 and cfg.num_key_value_heads==2 and cfg.hidden_act=='silu'
    assert cfg.tie_word_embeddings and cfg.rms_norm_eps==1e-6 and not cfg.use_sliding_window
    assert len(model.model.layers)==len(blocks)==24 and len({id(v) for v in blocks})==24
    assert all(layer.mlp.__class__.__name__=='Qwen2MLP' for layer in model.model.layers)
    assert model.model.embed_tokens.weight.data_ptr()==model.lm_head.weight.data_ptr()
    assert callable(model.gradient_checkpointing_enable)
    core={name:value for name,value in model.named_parameters() if '.mlp.' not in name}
    assert len(core)==218 and all(v.dtype==torch.bfloat16 for v in core.values())
    assert sum(v.numel() for v in core.values())==180246400
    adapters=[build_mlp_adapter(spec,block,li) for li,block in enumerate(blocks)]
    trainable=[v for block in blocks for v in block.parameters()]
    assert len({id(v) for v in trainable})==len(trainable), 'Cross-layer parameter alias is not this geometry'
    assert all(v.device==model.model.embed_tokens.weight.device for v in trainable)
    assert sum(v.numel() for v in trainable)==165150720
    for value in model.parameters():value.requires_grad_(False)
    for layer,adapter in zip(model.model.layers,adapters):layer.mlp=adapter
    for value in trainable:value.requires_grad_(True)
    actual_core={name:value for name,value in model.named_parameters() if '.mlp.' not in name}
    assert actual_core.keys()==core.keys() and all(actual_core[n] is core[n] for n in core)
    assert all(not v.requires_grad for v in actual_core.values())
    assert all('.mlp.compact.' in n for n,v in model.named_parameters() if v.requires_grad)
    assert model.model.embed_tokens.weight.data_ptr()==model.lm_head.weight.data_ptr()
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return dict(schema='QWEN_WHOLE_COMPACT_INSTALL_V1',layers=24,trainable_F32_elements=165150720,
        frozen_BF16_core_elements=180246400,all_source_non_FFN_parameter_objects_preserved=True,
        tied_head_preserved=True,source_dense_FFN_fallback=False,router='frozen_F32',
        arithmetic='BF16 source stream -> F32 compact SwiGLU/routing -> BF16 stream; no native parity claim')


def output_objective(student_logits, teacher_logits, token_mask):
    """T=1 forward KL, normalized by masked token count; teacher detached.

    This is the output objective component, not a trained method or admission
    criterion. Training cohort/updates/optimizer/whole quality must be frozen
    separately before a real fit. Teacher-forced KL does not prove own-history.
    """
    import torch
    from torch.nn import functional as F
    assert student_logits.ndim==3 and student_logits.shape==teacher_logits.shape
    assert token_mask.shape==student_logits.shape[:2] and token_mask.device==student_logits.device==teacher_logits.device
    assert student_logits.dtype in (torch.bfloat16,torch.float32) and teacher_logits.dtype in (torch.bfloat16,torch.float32)
    assert torch.isfinite(student_logits).all() and torch.isfinite(teacher_logits).all()
    assert torch.isfinite(token_mask).all() and bool((token_mask>=0).all()) and float(token_mask.sum())>0
    student_lp=F.log_softmax(student_logits.float(),dim=-1)
    teacher_lp=F.log_softmax(teacher_logits.detach().float(),dim=-1)
    per_token=(teacher_lp.exp()*(teacher_lp-student_lp)).sum(-1)
    loss=(per_token*token_mask.float()).sum()/token_mask.float().sum()
    assert torch.isfinite(loss)
    return loss


def resident_adam(model):
    """Explicit fully resident F32 Adam and dense gradient allocation for E16.

    Initialization performs no parameter update. All moments/gradient buffers
    are charged even for currently unselected leaves. A future finite fit must
    call zero_resident_gradients, never zero_grad(set_to_none=True). No n/RAM
    training scalability follows from this initial E16 implementation.
    """
    import torch
    params=[v for v in model.parameters() if v.requires_grad]
    assert sum(v.numel() for v in params)==165150720 and all(v.dtype==torch.float32 for v in params)
    optimizer=torch.optim.AdamW(params,lr=0.0001,betas=(0.9,0.999),eps=1e-8,weight_decay=0.0,foreach=False,fused=False)
    with torch.no_grad():
        for value in params:
            assert value.grad is None
            value.grad=torch.zeros_like(value)
            state=optimizer.state[value]
            state['step']=torch.tensor(0.,dtype=torch.float32,device='cpu')
            state['exp_avg']=torch.zeros_like(value);state['exp_avg_sq']=torch.zeros_like(value)
    return optimizer


def zero_resident_gradients(optimizer):
    for group in optimizer.param_groups:
        for value in group['params']:
            assert value.grad is not None and value.grad.dtype==value.dtype and value.grad.shape==value.shape
            value.grad.zero_()
