"""Isolated process configuration and exact master/moment width injection.

Old source files stay immutable. Configure globals before importing the streamed
learner: its imported DN/DT and the tensor core/specifications must agree.
"""
import torch
import original_tensor_learner as tensor

assert (tensor.DN,tensor.DT)==(512,16), 'configure once in an isolated worker'
tensor.DN,tensor.DT=1024,48
import original_falcon_learner as source
assert (source.DN,source.DT,tensor.DN,tensor.DT)==(1024,48,1024,48)
SourceLearner=source.SourceLearner
export=tensor.export
quant_weight=tensor.quant_weight


def widen(value,name,shape,moment=False):
    """Primary optimizer coordinates inject; new moments0, master replicas explicit."""
    old=value.detach().cpu()
    if tuple(old.shape)==tuple(shape):return old
    assert name.startswith('layers.') and '.organs.' in name and name.split('.')[1]!='5'
    out=torch.zeros(shape,dtype=old.dtype);organ=name.rsplit('.',1)[-1]
    if organ=='in_proj':
        out[:512]=old[:512];out[1024:1536]=old[512:]
        if not moment:out[512:1024]=old[:512];out[1536:]=old[512:]
    elif organ in ('conv_w','conv_b','A_log','Dskip','dt_b'):
        out[:512]=old
        if not moment:out[512:]=old
    elif organ=='x_proj':out[:16,:512]=old[:16];out[48:,:512]=old[16:]
    elif organ=='dt_proj':
        out[:512,:16]=old
        if not moment:out[512:,:16]=old
    elif organ=='out_proj':out[:,:512]=old
    else:raise AssertionError((name,old.shape,shape))
    return out


def verify(value,old,name,moment=False):
    """Independent primary readback/replica/zero checks with raw F32 bit equality."""
    value=value.detach().cpu();old=old.detach().cpu()
    def exact(a,b):return torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))
    if value.shape==old.shape:assert exact(value,old),name;return dict(changed_shape=False,new_coordinates=0)
    organ=name.rsplit('.',1)[-1];mask=torch.ones_like(value,dtype=torch.bool)
    if organ=='in_proj':
        assert exact(value[:512],old[:512]) and exact(value[1024:1536],old[512:]);mask[:512]=False;mask[1024:1536]=False
        if not moment:assert exact(value[512:1024],old[:512]) and exact(value[1536:],old[512:]);mask[:]=False
    elif organ in ('conv_w','conv_b','A_log','Dskip','dt_b'):
        assert exact(value[:512],old);mask[:512]=False
        if not moment:assert exact(value[512:],old);mask[:]=False
    elif organ=='x_proj':assert exact(value[:16,:512],old[:16]) and exact(value[48:,:512],old[16:]);mask[:16,:512]=False;mask[48:,:512]=False
    elif organ=='dt_proj':
        assert exact(value[:512,:16],old);mask[:512,:16]=False
        if not moment:assert exact(value[512:,:16],old);mask[512:,:16]=False
    else:assert organ=='out_proj' and exact(value[:,:512],old);mask[:,:512]=False
    assert not torch.count_nonzero(value[mask]),('nonzero new coordinate',name,moment)
    return dict(changed_shape=True,new_coordinates=value.numel()-old.numel(),primary_bit_exact=True,new_moments_zero=moment)
