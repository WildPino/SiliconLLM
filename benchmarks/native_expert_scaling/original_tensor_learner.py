"""Trainable original geometry and direct E4BPv001 producer.

Torch reductions/transcendentals are not claimed bit-identical to AVX/libm.
Bank masters stay on CPU; no all-expert quantized GPU copy or E*max_count pad.
"""
import math
from pathlib import Path
import struct

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

D, N, DN, DT, H, EH, K, L, WINDOW = 256, 96, 512, 16, 8, 128, 8, 6, 128
FIELD = struct.Struct('<64s6I2Q')


def specifications(e, v):
    assert K <= e <= (2**31-32)//EH and 1 <= v <= 2**31-1
    yield 'embed', (v,D), 1
    for l in range(L):
        p = f'layers.{l}.'
        yield p+'norm', (D,), 1
        if l == 5:
            yield p+'qkv', (3*D,D), 1
            yield p+'o', (D,D), 1
        else:
            for name, shape in [('in_proj',(2*DN,D)),('conv_w',(DN,4)),('conv_b',(DN,)),
                ('x_proj',(DT+2*N,DN)),('dt_proj',(DN,DT)),('dt_b',(DN,)),
                ('A_log',(DN,N)),('Dskip',(DN,)),('out_proj',(D,DN))]:
                yield p+name, shape, 1
        yield p+'ff_norm', (D,), 1
        yield p+'router', (e,D), 1
        yield p+'router_bias', (e,), 1
    yield 'final_norm', (D,), 1
    yield 'head', (v,D), 1
    for l in range(L):
        for name in ('gate','up','down'):
            shape = (e,EH//2,D) if name == 'down' else (D//2,e*EH)
            yield f'layers.{l}.{name}_code', shape, 2
            yield f'layers.{l}.{name}_scale', (e,D if name=='down' else EH), 1


def quant_weight(w):
    scale = w.detach().abs().mean(-1).clamp_min(1e-5)
    q = (w.detach()/scale.unsqueeze(-1)).round().clamp(-1,1)
    return q, scale


def aq63(x):
    """Native multiply by reciprocal, ties-even; zero vector has zero scale."""
    scale = x.detach().abs().amax(-1,keepdim=True)/63.0
    inverse = torch.where(scale == 0, torch.zeros_like(scale), 1.0/scale)
    q = (x.detach()*inverse).round().clamp(-63,63)
    effective = x + (q*scale-x).detach()
    return effective, q, scale


def ternary_linear(x, w):
    xe, qx, a = aq63(x)
    qw, s = quant_weight(w)
    we = w + (qw*s.unsqueeze(-1)-w).detach()
    # Integer sums fit exactly in F32: <=256*63 or128*63, well below2**24.
    exact = ((qx @ qw.T)*a)*s
    surrogate = F.linear(xe,we)
    return exact.detach() + (surrogate-surrogate.detach())


class Bank(nn.Module):
    def __init__(self,e,device='cpu'):
        super().__init__()
        self.e = e
        self.router = nn.Linear(D,e,device=device,dtype=torch.float32)
        self.gate = nn.Parameter(torch.empty(e,EH,D,dtype=torch.float32))
        self.up = nn.Parameter(torch.empty(e,EH,D,dtype=torch.float32))
        self.down = nn.Parameter(torch.empty(e,D,EH,dtype=torch.float32))
        self.last_routes = None

    def forward(self,x):
        shape = x.shape
        xf = x.reshape(-1,D)
        prob = torch.softmax(self.router(xf),-1)
        ids = torch.argsort(prob,dim=-1,descending=True,stable=True)[:,:K]
        mass = prob.gather(-1,ids)
        mass = mass/mass.sum(-1,keepdim=True)
        if not torch.is_grad_enabled():
            self.last_routes = (ids.detach().cpu(),mass.detach().cpu())
        # Unique positions within a rank; fixed rank-order reduction. No index_add.
        out = torch.zeros_like(xf)
        for rank in range(K):
            part = torch.zeros_like(xf)
            for e in torch.unique(ids[:,rank],sorted=True).tolist():
                pos = torch.where(ids[:,rank] == e)[0]
                inp = xf.index_select(0,pos)
                g = ternary_linear(inp,self.gate[e].to(x.device))
                u = ternary_linear(inp,self.up[e].to(x.device))
                hidden = (F.relu(g)*F.relu(u))*mass[pos,rank,None]
                y = ternary_linear(hidden,self.down[e].to(x.device))
                part = part.index_copy(0,pos,y)
            out = out+part
        return out.reshape(shape)


def norm(x,w):
    return (x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5))*w


class Block(nn.Module):
    def __init__(self,l,e,device='cpu'):
        super().__init__()
        specs = [(name.split('.')[-1],shape) for name,shape,dtype in specifications(e,1)
                 if name.startswith(f'layers.{l}.') and dtype==1 and
                 name.split('.')[-1] not in ('router','router_bias') and not name.endswith('_scale')]
        self.organs = nn.ParameterDict({name:nn.Parameter(torch.empty(shape,device=device)) for name,shape in specs})
        self.bank = Bank(e,device)
        self.l = l

    def core(self,x):
        w = self.organs
        xn = norm(x,w['norm'])
        if self.l == 5:
            b,t,_ = xn.shape
            q,k,v = F.linear(xn,w['qkv']).chunk(3,-1)
            q,k,v = [a.reshape(b,t,H,D//H).transpose(1,2) for a in (q,k,v)]
            scores = (q @ k.transpose(-1,-2))/math.sqrt(D//H)
            at = torch.arange(t,device=x.device)
            mask = (at[:,None]>=at[None,:]) & (at[:,None]-at[None,:]<WINDOW)
            scores = scores.masked_fill(~mask,-torch.inf)
            y = (scores.softmax(-1) @ v).transpose(1,2).reshape(b,t,D)
            return F.linear(y,w['o'])
        xx,z = F.linear(xn,w['in_proj']).chunk(2,-1)
        xx = F.conv1d(xx.transpose(1,2),w['conv_w'][:,None,:],w['conv_b'],
                      padding=3,groups=DN)[...,:x.shape[1]].transpose(1,2)
        xx = xx/(1.0+torch.exp(-xx))
        dbl = F.linear(xx,w['x_proj'])
        dt,B,C = dbl.split((DT,N,N),-1)
        dt = F.softplus(F.linear(dt,w['dt_proj'],w['dt_b']),threshold=20)
        A = -torch.exp(w['A_log'])
        state = x.new_zeros(x.shape[0],DN,N)
        ys = []
        for t in range(x.shape[1]):
            d = dt[:,t,:,None]
            state = torch.exp(d*A)*state + (dt[:,t]*xx[:,t])[:,:,None]*B[:,t,None,:]
            ys.append((state*C[:,t,None,:]).sum(-1)+w['Dskip']*xx[:,t])
        y = torch.stack(ys,1)*(z/(1.0+torch.exp(-z)))
        return F.linear(y,w['out_proj'])

    def forward(self,x):
        x = x+self.core(x)
        return x+self.bank(norm(x,self.organs['ff_norm']))


class Learner(nn.Module):
    def __init__(self,e,v,device='cpu'):
        super().__init__()
        self.e,self.v = e,v
        self.embed = nn.Parameter(torch.empty(v,D,device=device))
        self.head = nn.Parameter(torch.empty(v,D,device=device))
        self.final_norm = nn.Parameter(torch.empty(D,device=device))
        self.layers = nn.ModuleList([Block(l,e,device) for l in range(L)])
        self.use_checkpoint = False

    def forward(self,ids):
        x = F.embedding(ids,self.embed)
        for block in self.layers:
            x = checkpoint(block,x,use_reentrant=False) if self.use_checkpoint and torch.is_grad_enabled() else block(x)
        return F.linear(norm(x,self.final_norm),self.head)

    def tensor(self,name):
        if '.' not in name:
            return getattr(self,name)
        _,l,key = name.split('.')
        block = self.layers[int(l)]
        if key == 'router':return block.bank.router.weight
        if key == 'router_bias':return block.bank.router.bias
        if key in ('gate','up','down'):return getattr(block.bank,key)
        return block.organs[key]

    @torch.no_grad()
    def adopt_phase57(self,packet):
        """Master weights, not dequantized legacy references. No optimizer adoption."""
        cfg,sd = packet['cfg'],packet['model']
        assert (cfg['D'],cfg['N'],cfg['L'],cfg['E'],cfg['V'],cfg['hid_e'],cfg['topk']) == (D,N,L,self.e,self.v,EH,K)
        touched = set()
        def put(dst,src):
            value = sd[src];dest=self.tensor(dst)
            assert value.numel()==dest.numel() and torch.isfinite(value).all(),src
            dest.copy_(value.reshape_as(dest));touched.add(src)
        for dst,src in [('embed','emb.weight'),('head','head.weight'),('final_norm','norm_f.w')]:put(dst,src)
        for l in range(L):
            dst=f'layers.{l}.';src=f'blocks.{l}.'
            put(dst+'norm',src+'norm.w');put(dst+'ff_norm',src+'norm2.w')
            keys = ['qkv','o'] if l==5 else ['in_proj','conv_w','conv_b','x_proj','dt_proj','dt_b','A_log','Dskip','out_proj']
            mapping={'conv_w':'conv1d.weight','conv_b':'conv1d.bias','dt_b':'dt_proj.bias'}
            for key in keys:put(dst+key,src+'mix.'+mapping.get(key,key if key in ('A_log','Dskip') else key+'.weight'))
            put(dst+'router',src+'mlp.router.weight');put(dst+'router_bias',src+'mlp.router.bias')
            for key in ('gate','up'):put(dst+key,src+'mlp.'+key+'.weight')
            put(dst+'down',src+'mlp.Wd')
        assert touched==set(sd),(set(sd)-touched)


@torch.no_grad()
def export(model,path):
    """Stream tensors in legacy-compatible field order; arbitrary runtime E/V."""
    specs = list(specifications(model.e,model.v))
    assert len(specs)==110
    cursor=80+FIELD.size*len(specs);table=[]
    for name,shape,dtype in specs:
        size=math.prod(shape)*(4 if dtype==1 else 1)
        table.append(FIELD.pack(name.encode(),dtype,len(shape),*(list(shape)+[0]*(4-len(shape))),cursor,size))
        cursor+=size
    h=struct.pack('<16I',1,model.v,D,N,H,L,DN,DT,4,WINDOW,5,model.e,EH,K,1,110)
    with Path(path).open('xb') as f:
        f.write(b'E4BPv001'+h+struct.pack('<Q',cursor)+b''.join(table))
        cached = None
        for name,shape,dtype in specs:
            if name.endswith(('_code','_scale')):
                base=name.rsplit('_',1)[0]
                if name.endswith('_code'):
                    w=model.tensor(base).detach().cpu()
                    q,s=quant_weight(w)
                    a=q.numpy().astype('i1')
                    code=((a[...,0::2].astype('i2')+1)*3+a[...,1::2]+1).astype('u1')
                    code=code.transpose(0,2,1) if base.endswith('.down') else code.reshape(model.e*EH,D//2).T
                    data=np.ascontiguousarray(code);cached=s.numpy().astype('<f4')
                    del w,q,s,a,code
                else:
                    assert cached is not None
                    data=cached;cached=None
            else:data=model.tensor(name).detach().cpu().numpy().astype('<f4',copy=False)
            assert data.shape==shape and np.all(np.isfinite(data)),name
            if name.endswith('_scale'):assert np.min(data)>0
            f.write(data.tobytes(order='C'));del data
        assert f.tell()==cursor
    return dict(bytes=cursor,e=model.e,V=model.v,expert_reference_bytes=0,
                master_coefficients=sum(p.numel() for p in model.parameters()))
