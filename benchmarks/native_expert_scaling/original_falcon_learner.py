"""Source-informed original geometry with streamed first-order bank adjoint.

CPU masters and CPU ternary decisions; one expert's weights live on GPU at a time.
The custom adjoint recomputes each consulted expert and scatters a single dense
CPU gradient per bank, avoiding a full-bank zero gradient per indexed expert.
"""
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from original_tensor_learner import Learner, Bank, D, N, DN, DT, EH, K, L, aq63, quant_weight, norm

E, V = 1152, 65537


def groups(ids):
    flat=ids.detach().cpu().flatten()
    order=torch.argsort(flat,stable=True)
    values,counts=torch.unique_consecutive(flat[order],return_counts=True)
    chunks=order.split(counts.tolist())
    return [(int(e),p) for e,p in zip(values.tolist(),chunks)]


def linear(x,master,grad=False):
    # This is precisely the exporter CPU quantizer, regardless of activation device.
    q,s=quant_weight(master)
    q=q.to(x.device);s=s.to(x.device)
    xe,qx,a=aq63(x)
    exact=((qx @ q.T)*a)*s
    if not grad:return exact,None
    w=master.detach().to(x.device).requires_grad_(True)
    we=q*s[:,None]+(w-w.detach())
    surrogate=F.linear(xe,we)
    return exact.detach()+(surrogate-surrogate.detach()),w


class StreamBank(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,mass,ids,gate,up,down,progress):
        assert gate.device.type==up.device.type==down.device.type=='cpu'
        ctx.save_for_backward(x,mass,ids,gate,up,down)
        ctx.progress=progress
        pair=torch.zeros(x.shape[0]*K,D,device=x.device,dtype=x.dtype)
        grouped=groups(ids)
        for ordinal,(e,positions) in enumerate(grouped):
            pos=positions.to(x.device);token=pos//K
            inp=x.index_select(0,token)
            g,_=linear(inp,gate[e]);u,_=linear(inp,up[e])
            hidden=(F.relu(g)*F.relu(u))*mass.flatten().index_select(0,pos)[:,None]
            y,_=linear(hidden,down[e]);pair.index_copy_(0,pos,y)
            if progress is not None and (ordinal%64==0 or ordinal+1==len(grouped)):
                progress('bank_forward',ordinal+1,len(grouped))
        pair=pair.reshape(x.shape[0],K,D)
        out=torch.zeros_like(x)
        for rank in range(K):out=out+pair[:,rank]
        return out

    @staticmethod
    def backward(ctx,dy):
        x,mass,ids,gate,up,down=ctx.saved_tensors
        dxpair=torch.zeros(x.shape[0]*K,D,device=x.device,dtype=x.dtype)
        dm=torch.zeros_like(mass).flatten()
        dg,du,dd=[torch.zeros_like(w,device='cpu') for w in (gate,up,down)]
        grouped=groups(ids)
        for ordinal,(e,positions) in enumerate(grouped):
            pos=positions.to(x.device);token=pos//K
            with torch.enable_grad():
                inp=x.detach().index_select(0,token).requires_grad_(True)
                m=mass.detach().flatten().index_select(0,pos).requires_grad_(True)
                g,wg=linear(inp,gate[e],True);u,wu=linear(inp,up[e],True)
                hidden=(F.relu(g)*F.relu(u))*m[:,None]
                y,wd=linear(hidden,down[e],True)
                gi,gm,gg,gu,gd=torch.autograd.grad(y,(inp,m,wg,wu,wd),
                    dy.index_select(0,token),create_graph=False,retain_graph=False)
            dxpair.index_copy_(0,pos,gi);dm.index_copy_(0,pos,gm)
            dg[e].copy_(gg.detach().cpu());du[e].copy_(gu.detach().cpu());dd[e].copy_(gd.detach().cpu())
            del inp,m,g,u,hidden,y,wg,wu,wd,gi,gm,gg,gu,gd
            if ctx.progress is not None and (ordinal%64==0 or ordinal+1==len(grouped)):
                ctx.progress('bank_backward',ordinal+1,len(grouped))
        dxpair=dxpair.reshape(x.shape[0],K,D);dx=torch.zeros_like(x)
        for rank in range(K):dx=dx+dxpair[:,rank]
        return dx,dm.reshape_as(mass),None,dg,du,dd,None


class UnionBank(Bank):
    def forward(self,x):
        shape=x.shape;xf=x.reshape(-1,D)
        prob=torch.softmax(self.router(xf),-1)
        ids=torch.argsort(prob,dim=-1,descending=True,stable=True)[:,:K]
        mass=prob.gather(-1,ids);mass=mass/mass.sum(-1,keepdim=True)
        self.last_routes=(ids.detach().cpu(),mass.detach().cpu())
        return StreamBank.apply(xf,mass,ids,self.gate,self.up,self.down,
                                getattr(self,'progress',None)).reshape(shape)


class SourceLearner(Learner):
    def __init__(self,e=E,v=V,device='cuda'):
        super().__init__(e,v,device)
        for block in self.layers:
            block.bank.__class__=UnionBank
            block.bank.progress=None

    def forward(self,ids,positions=None):
        x=F.embedding(ids,self.embed)
        for block in self.layers:
            x=checkpoint(block,x,use_reentrant=False,preserve_rng_state=True) if self.use_checkpoint and torch.is_grad_enabled() else block(x)
        x=norm(x,self.final_norm)
        if positions is not None:x=x[:,positions]
        return F.linear(x,self.head)


@torch.no_grad()
def initialize(model,source,basis_path,seed=1710,progress=None):
    """Projection/partition initializer; fresh original core, no source-state claim."""
    import json
    from safetensors import safe_open
    source=Path(source);config=json.loads((source/'config.json').read_bytes())
    assert (config['hidden_size'],config['num_hidden_layers'],config['intermediate_size'],config['vocab_size'])==(2048,24,4608,V)
    assert model.e==E and model.v==V
    packet=torch.load(basis_path,map_location='cpu',weights_only=True)
    p=packet['P'][:,:D].float().contiguous();assert p.shape==(2048,D)
    assert float((p.T@p-torch.eye(D)).abs().max())<=1e-4
    rng=torch.Generator(device='cpu').manual_seed(seed)
    ledger=dict(schema='ORIGINAL_FALCON_INITIALIZATION_V1',seed=seed,
        basis_shape=list(p.shape),basis_sha256=hashlib.sha256(p.numpy().tobytes()).hexdigest(),
        basis_origin=str(Path(basis_path).resolve()),basis_info=packet['info'],
        core='Fresh Mamba1/SWA operators; no pretrained state transfer.',norm_factor=1.0,
        down_multiplier=E/16,expert_maps=[],source_parameters_used=[])
    del packet
    touched=set()
    def put(name,value):
        dest=model.tensor(name);assert dest.shape==value.shape and torch.isfinite(value).all(),name
        dest.copy_(value.to(dest.device));touched.add(name)
    def random(shape,std):return torch.randn(shape,generator=rng)*std
    with safe_open(source/'model.safetensors',framework='pt',device='cpu') as sf:
        def get(name):
            ledger['source_parameters_used'].append(name);return sf.get_tensor(name).float()
        for name,key,mul in [('embed','model.embed_tokens.weight',config['embedding_multiplier']),
                             ('head','lm_head.weight',config['lm_head_multiplier'])]:
            # Only a row chunk is cast to F32, not the whole65537*2048 source matrix.
            old=sf.get_slice(key);ledger['source_parameters_used'].append(key)
            for lo in range(0,V,4096):
                value=(old[lo:min(lo+4096,V)].float()@p)*mul
                model.tensor(name)[lo:lo+value.shape[0]].copy_(value.to(model.tensor(name).device))
            touched.add(name)
        put('final_norm',(p.square()*get('model.final_layernorm.weight')[:,None]).sum(0))
        for l,block in enumerate(model.layers):
            prefix=f'layers.{l}.'
            for label,key in [('norm','input_layernorm'),('ff_norm','pre_ff_layernorm')]:
                gamma=torch.stack([get(f'model.layers.{4*l+j}.{key}.weight') for j in range(4)]).mean(0)
                put(prefix+label,(p.square()*gamma[:,None]).sum(0))
            if l==5:
                put(prefix+'qkv',random((3*D,D),1/math.sqrt(D)))
                put(prefix+'o',random((D,D),.01/math.sqrt(D)))
            else:
                put(prefix+'in_proj',random((2*DN,D),1/math.sqrt(D)))
                conv=torch.zeros(DN,4);conv[:,-1]=1;put(prefix+'conv_w',conv)
                put(prefix+'conv_b',torch.zeros(DN))
                put(prefix+'x_proj',random((DT+2*N,DN),.1/math.sqrt(DN)))
                put(prefix+'dt_proj',random((DN,DT),1/math.sqrt(DT)))
                dt=torch.exp(torch.empty(DN).uniform_(math.log(.001),math.log(.05),generator=rng))
                put(prefix+'dt_b',dt+torch.log(-torch.expm1(-dt)))
                put(prefix+'A_log',torch.arange(1,N+1).float().log().expand(DN,N).clone())
                put(prefix+'Dskip',torch.ones(DN))
                put(prefix+'out_proj',random((D,DN),.01/math.sqrt(DN)))
            router=torch.empty(E,D);bias=torch.zeros(E);e=0
            for source_layer in range(4*l,4*l+4):
                base=f'model.layers.{source_layer}.feed_forward.'
                gate=(get(base+'gate_proj.weight')@p)*config['mlp_multipliers'][0]
                up=get(base+'up_proj.weight')@p
                down=(p.T@get(base+'down_proj.weight'))*config['mlp_multipliers'][1]
                for partition in ('contiguous','strided'):
                    for group in range(36):
                        rows=torch.arange(group*EH,(group+1)*EH) if partition=='contiguous' else torch.arange(group,4608,36)
                        assert rows.numel()==EH
                        for sg,su in ((1,1),(1,-1),(-1,1),(-1,-1)):
                            block.bank.gate[e].copy_(sg*gate[rows])
                            block.bank.up[e].copy_(su*up[rows])
                            block.bank.down[e].copy_(sg*su*(E/16)*down[:,rows])
                            # A signed row-mean fingerprint; top8 is learned selection, no reconstruction identity.
                            key=(sg*gate[rows].mean(0)+su*up[rows].mean(0))
                            router[e]=key/key.norm().clamp_min(1e-8)
                            ledger['expert_maps'].append(dict(site=l,expert=e,source_layer=source_layer,
                                partition=partition,group=group,rows=rows.tolist(),gate_sign=sg,up_sign=su))
                            e+=1
                del gate,up,down
                if progress is not None:progress('source_block',source_layer+1,24)
            assert e==E
            put(prefix+'router',router);put(prefix+'router_bias',bias)
            for name in ('gate','up','down'):touched.add(prefix+name)
    expected={'embed','head','final_norm'}
    for l,block in enumerate(model.layers):
        expected|={f'layers.{l}.{key}' for key in block.organs}
        expected|={f'layers.{l}.{key}' for key in ('router','router_bias','gate','up','down')}
    assert touched==expected
    ledger['source_parameters_used']=sorted(set(ledger['source_parameters_used']))
    ledger['target_fields_initialized']=sorted(touched)
    ledger['total_master_coefficients']=sum(v.numel() for v in model.parameters())
    ledger['bank_coefficients']=3*L*E*EH*D
    return ledger,p
