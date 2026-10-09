"""Source-width FFN conversion into the existing ternary/AQ63 arithmetic.

An offline intermediate: full row coverage, no router or core transformation.
Only the final FFN result is cast back to the original source input dtype.
"""
import torch
from torch import nn
from torch.nn import functional as F
from chatbot_hybrid_target import aq63, row_scale


class EngineFFN(nn.Module):
    def __init__(self, source, site, observer=None):
        super().__init__()
        assert source.hidden_size==2048 and source.intermediate_size==4608
        assert source.config.hidden_act=='silu' and not source.config.mlp_bias
        self.site=site;self.observer=observer
        self.gate_multiplier=source.gate_multiplier
        self.down_multiplier=source.down_multiplier
        self.weight_metrics=[]
        for name in ('gate','up','down'):
            projection=getattr(source,name+'_proj');assert projection.bias is None
            w=projection.weight.detach().float()
            scale=row_scale(w)
            code=torch.round(w/scale[:,None]).clamp(-1,1).to(torch.int8)
            assert torch.isfinite(scale).all().item() and (scale>=1e-8).all().item()
            assert code.abs().max().item()<=1 and w.shape[1]%2==0
            error=(w-code.float()*scale[:,None]).double().square().sum().item()
            norm=w.double().square().sum().item()
            self.weight_metrics.append(dict(projection=name,shape=list(w.shape),
                relative_weight_L2=(error/max(norm,1e-60))**.5,
                zero_codes=int((code==0).sum().item()),elements=code.numel()))
            self.register_buffer(name+'_codes',code)
            self.register_buffer(name+'_scale',scale)

    def linear(self,x,name):
        code=getattr(self,name+'_codes');scale=getattr(self,name+'_scale')
        q,a=aq63(x)
        dot=F.linear(q,code.float())
        if self.observer is not None:self.observer(self.site,name,q,code,dot)
        return dot*scale*a

    def forward(self,x):
        dtype=x.dtype;x=x.float()
        g=self.linear(x,'gate')*self.gate_multiplier
        u=self.linear(x,'up')
        z=F.silu(g)*u
        y=self.linear(z,'down')*self.down_multiplier
        return y.to(dtype)


def packed_projection(module,name):
    code=getattr(module,name+'_codes')
    # Existing engine export: byte=(first_trit+1)*3+(second_trit+1),pair/row.
    packed=((code[:,0::2]+1)*3+(code[:,1::2]+1)).transpose(0,1).contiguous()
    decoded=packed.transpose(0,1)
    assert torch.equal(decoded//3-1,code[:,0::2])
    assert torch.equal(decoded%3-1,code[:,1::2])
    return packed,getattr(module,name+'_scale')
