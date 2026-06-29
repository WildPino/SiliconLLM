#!/usr/bin/env python3
# Phase 56 - MQAR de-risk probe. Does the thesis (SSM + sparse-KV-recall) hold BEFORE any stage?
#   Task: sequence of (key,value) pairs spread over context, then queries; model recalls the value of the
#   queried key. Metric: recall accuracy per query, STRATIFIED by key->query distance.
#
#   Arms (param/training-matched: same SSM backbone, only the last "recall slot" layer differs):
#     ssm    : recall slot = SSM        -> should DEGRADE with context/pairs (Mamba-MQAR collapse = problem exists)
#     swa    : recall slot = sliding-window attention (win) -> recovers short-range, fails beyond window
#     sparse : recall slot = TOP-K query-aware attention over full KV (simplest sparse-recall; NOT VQ yet)
#     attn   : recall slot = full causal attention -> recall CEILING
#   Decisive: does `sparse` recover recall where `ssm` collapses, approaching `attn`?
#   (sparse and attn share the Attn module; sparse = top-k mask, attn = no mask. Perfectly matched.)
#
#   Infra: trains on T4/Linux -> use --fp16 (Turing has fp16 tensor cores, bf16 is emulated/slow). Linux-friendly.
#   Deterministic. Smoke + STOP. No commit. The CPU gather cost of `sparse` is profiled separately
#   (phase56_gather_profile.c): random top-k gather can be slower than a linear read = the hidden CPU bottleneck.
#
# Smoke: python benchmarks/phase56/phase56_mqar.py --arm all --smoke
# Real : python benchmarks/phase56/phase56_mqar.py --arm all --ctx 2048 --pairs 64 --queries 32 --steps 4000 --fp16
import argparse, math, time, os
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint as ckpt

# ---------------- MQAR synthetic data (VECTORIZED on CPU, moved to device once; deterministic) ----------------
# vocab: 0 = PAD/filler ; 1..NK = key tokens ; NK+1..NK+NV = value tokens
# gen MUST be a CPU torch.Generator (built on CPU to avoid per-element GPU scalar writes = the old 20h bottleneck).
# spread=True (R-B drift apparatus): place pairs in the first region, SPREAD queries across the rest -> query
#   absolute POSITION varies (how derived the SSM state is) DECOUPLED from key->query DISTANCE (every key precedes
#   every query). spread=False (default) = canonical tail-clustered queries (preserves all old ckpts/probes).
def make_batch(B,T,P,Q,NK,NV,gen,device,spread=False):
    if spread:
        kv_end=max(2*P+2,T//4)
        assert kv_end//2>=P and (T-kv_end)//2>=Q, f"ctx {T} too small for spread: P={P} Q={Q} kv_end={kv_end}"
        keys = torch.rand(B,NK,generator=gen).argsort(1)[:,:P]+1
        vals = torch.randint(0,NV,(B,P),generator=gen)+1+NK
        slots,_=(torch.rand(B,kv_end//2,generator=gen).argsort(1)[:,:P]*2).sort(1)         # pairs in [0,kv_end)
        x=torch.zeros(B,T,dtype=torch.long); x.scatter_(1,slots,keys); x.scatter_(1,slots+1,vals)
        qp,_=((torch.rand(B,(T-kv_end)//2,generator=gen).argsort(1)[:,:Q])*2+kv_end).sort(1)# queries SPREAD in [kv_end,T)
        qi=torch.randint(0,P,(B,Q),generator=gen)                                          # each query -> a preceding key
        qk,qv,qpos=torch.gather(keys,1,qi),torch.gather(vals,1,qi),torch.gather(slots,1,qi)
        x.scatter_(1,qp,qk); x.scatter_(1,qp+1,qv)
        tgt=torch.full((B,T),-100,dtype=torch.long); tgt.scatter_(1,qp,qv)
        dist=torch.full((B,T),-1,dtype=torch.long);  dist.scatter_(1,qp,qp-qpos)
        match=torch.full((B,T),-100,dtype=torch.long); match.scatter_(1,qp,qpos)
        return (x.to(device,non_blocking=True),tgt.to(device,non_blocking=True),
                dist.to(device,non_blocking=True),match.to(device,non_blocking=True))
    qregion=2*Q; kv_span=T-qregion
    assert kv_span>=2*P, f"context {T} too small for {P} pairs + {Q} queries"
    keys = torch.rand(B,NK,generator=gen).argsort(1)[:,:P]+1                    # (B,P) distinct keys
    vals = torch.randint(0,NV,(B,P),generator=gen)+1+NK                         # (B,P) random values
    slots,_= (torch.rand(B,kv_span//2,generator=gen).argsort(1)[:,:P]*2).sort(1)# (B,P) spread, sorted even positions
    x   = torch.zeros(B,T,dtype=torch.long)
    x.scatter_(1,slots,keys); x.scatter_(1,slots+1,vals)                        # place key@slot, value@slot+1
    qi  = torch.randint(0,P,(B,Q),generator=gen)                               # which keys get queried
    qk,qv,qpos = torch.gather(keys,1,qi),torch.gather(vals,1,qi),torch.gather(slots,1,qi)
    qp  = (kv_span+2*torch.arange(Q)).unsqueeze(0).expand(B,Q)                  # (B,Q) query positions
    x.scatter_(1,qp,qk); x.scatter_(1,qp+1,qv)                                  # teacher-forced value follows
    tgt = torch.full((B,T),-100,dtype=torch.long); tgt.scatter_(1,qp,qv)        # predict value at query-key pos
    dist= torch.full((B,T),-1,dtype=torch.long);  dist.scatter_(1,qp,qp-qpos)   # key->query distance
    match=torch.full((B,T),-100,dtype=torch.long); match.scatter_(1,qp,qpos)    # for InfoNCE: matched key POSITION per query pos
    return (x.to(device,non_blocking=True),tgt.to(device,non_blocking=True),
            dist.to(device,non_blocking=True),match.to(device,non_blocking=True))

# ---------------- mixers ----------------
SCAN_CHUNK=128   # chunked scan: exp/dBx vectorized per block (fewer kernel launches), memory-bounded, ckpt-safe
class SSM(nn.Module):                                   # minimal Mamba-1 selective diagonal SSM
    def __init__(s,D,N=16,expand=2,conv=4):
        super().__init__(); s.D=D; s.N=N; s.Dn=D*expand; s.dtr=max(8,D//16); s.conv=conv
        s.in_proj=nn.Linear(D,2*s.Dn,bias=False)
        s.conv1d=nn.Conv1d(s.Dn,s.Dn,conv,groups=s.Dn,padding=conv-1,bias=True)
        s.x_proj=nn.Linear(s.Dn,s.dtr+2*N,bias=False); s.dt_proj=nn.Linear(s.dtr,s.Dn,bias=True)
        s.A_log=nn.Parameter(torch.log(torch.arange(1,N+1,dtype=torch.float32).repeat(s.Dn,1)))
        s.Dskip=nn.Parameter(torch.ones(s.Dn)); s.out_proj=nn.Linear(s.Dn,D,bias=False)
    def forward(s,x):
        B,T,_=x.shape
        xz=s.in_proj(x); xx,z=xz.chunk(2,-1)
        xx=s.conv1d(xx.transpose(1,2))[:,:,:T].transpose(1,2); xx=F.silu(xx)
        dbl=s.x_proj(xx); dt,Bm,Cm=torch.split(dbl,[s.dtr,s.N,s.N],-1)
        dt=F.softplus(s.dt_proj(dt)); A=-torch.exp(s.A_log.float())
        with torch.autocast(device_type=x.device.type,enabled=False):
            dt=dt.float(); Bm=Bm.float(); Cm=Cm.float(); xx32=xx.float()
            h=torch.zeros(B,s.Dn,s.N,device=x.device); ys=[]
            for c0 in range(0,T,SCAN_CHUNK):                              # chunked: vectorize exp/dBx per block...
                c1=min(c0+SCAN_CHUNK,T)
                aC=torch.exp(dt[:,c0:c1].unsqueeze(-1)*A)                 # (B,k,Dn,N) one kernel for the chunk
                bC=dt[:,c0:c1].unsqueeze(-1)*Bm[:,c0:c1].unsqueeze(2)*xx32[:,c0:c1].unsqueeze(-1)
                CC=Cm[:,c0:c1]
                for t in range(c1-c0):                                    # ...sequential recurrence stays (cheap elementwise)
                    h=aC[:,t]*h+bC[:,t]; ys.append((h*CC[:,t].unsqueeze(1)).sum(-1))
            y=torch.stack(ys,1)+xx32*s.Dskip
        return s.out_proj(y.to(x.dtype)*F.silu(z))

class Attn(nn.Module):                                  # causal; mode = full | window | topk
    def __init__(s,D,H,mode='full',window=512,topk=16):
        super().__init__(); s.H=H; s.hd=D//H; s.mode=mode; s.window=window; s.topk=topk
        s.qkv=nn.Linear(D,3*D,bias=False); s.o=nn.Linear(D,D,bias=False)
    def forward(s,x):
        B,T,D=x.shape; H,hd=s.H,s.hd
        q,k,v=s.qkv(x).split(D,-1)
        q=q.view(B,T,H,hd).transpose(1,2); k=k.view(B,T,H,hd).transpose(1,2); v=v.view(B,T,H,hd).transpose(1,2)
        att=(q@k.transpose(-1,-2))/math.sqrt(hd)
        i=torch.arange(T,device=x.device); mask=(i[None,:]<=i[:,None])
        if s.mode=='window': mask=mask & (i[None,:]>i[:,None]-s.window)
        att=att.masked_fill(~mask[None,None],float('-inf'))
        if s.mode=='topk':
            kk=min(s.topk,T); thr=torch.topk(att,kk,dim=-1).values[...,-1,None]
            att=att.masked_fill(att<thr,float('-inf'))
        y=(att.softmax(-1)@v).transpose(1,2).reshape(B,T,D)
        return s.o(y)

def _hadamard_rows(n,d):                                # data-independent partition: Sylvester-Hadamard sign-pattern (ParisKV/IVF-TQ style)
    order=1
    while order<max(n,d): order*=2
    H=torch.ones(1,1)
    while H.shape[0]<order: H=torch.cat([torch.cat([H,H],1),torch.cat([H,-H],1)],0)
    return H[:n,:d].contiguous().float()               # (n atoms, d) entries in {+1,-1}

class RoutedAttn(nn.Module):    # Track A: coarse-IVF routing, refine EXACT (NO PQ). train=soft gate, eval=hard top-nprobe.
    # The recall slot routes each query to its top-nprobe codebook buckets and attends EXACTLY over the keys in those
    # buckets. R-B: arm A part='learned' (C trained), arm B part='fixed' (C = frozen Hadamard, data-independent) -> the
    # ONE variable is the PARTITION; the ENCODER stays InfoNCE in BOTH (we keep the representation, test its partition).
    def __init__(s,D,H,vatoms,nprobe,tau,learn,part='learned',qnorm=False):  # learn in {infonce,kmeans,random}; part in {learned,fixed}
        super().__init__(); s.H=H; s.hd=D//H; s.nprobe=nprobe; s.tau=tau; s.learn=learn; s.vatoms=vatoms; s.part=part; s.qnorm=qnorm
        s.qkv=nn.Linear(D,3*D,bias=False); s.o=nn.Linear(D,D,bias=False)
        if part=='fixed':                                                    # data-independent: C NOT learned (buffer), encoder still InfoNCE-trained against it
            s.register_buffer('C',F.normalize(_hadamard_rows(vatoms,D),dim=-1)); s._fixed=True
        else:
            s.C=nn.Parameter(torch.randn(vatoms,D)/math.sqrt(D),requires_grad=(learn!='random')); s._fixed=False  # random=frozen LSH baseline
        s.ql=s.kl=None; s._bal=0.0
        s.refine='exact'; s.pq_cent=None; s.pq_ds=2; s.kshort=32; s._capture=False; s._kbuf=None   # eval-only PQ-ADC (does not affect training)
    def forward(s,x):
        B,T,D=x.shape; H,hd=s.H,s.hd
        q,k,v=s.qkv(x).split(D,-1)
        Cn=F.normalize(s.C,dim=-1)
        ql=F.normalize(q,dim=-1)@Cn.t(); kl=F.normalize(k,dim=-1)@Cn.t()   # cosine routing logits (B,T,V)
        s.ql,s.kl=ql,kl                                                    # cache for aux (slot is NOT checkpointed)
        qh=q.view(B,T,H,hd).transpose(1,2); kh=k.view(B,T,H,hd).transpose(1,2); vh=v.view(B,T,H,hd).transpose(1,2)
        qs=F.normalize(qh,dim=-1) if s.qnorm else qh                       # R-B: l2-normalize query before scoring -> removes RADIAL drift (SSM ||h|| growth)
        if s._capture and not s.training: s._kbuf.append(kh.detach())      # collect per-head keys to FIT the PQ codebook
        khq=s._pq_quantize(kh) if ((not s.training) and s.refine in ('pqadc','rerank') and s.pq_cent is not None) else None
        att=(qs@(khq if s.refine=='pqadc' else kh).transpose(-1,-2))/math.sqrt(hd)  # pqadc: quantized keys; exact/rerank: exact keys
        i=torch.arange(T,device=x.device); causal=(i[None,:]<=i[:,None])
        if s.training:                                                     # SOFT differentiable gate (Gumbel-style routing)
            qa=F.softmax(ql/s.tau,-1); ka=F.softmax(kl/s.tau,-1)
            g=qa@ka.transpose(1,2)                                         # (B,T,T) prob query & key share a bucket
            att=att+torch.log(g+1e-6).unsqueeze(1)                         # bias attention toward same-bucket keys
            att=att.masked_fill(~causal[None,None],float('-inf'))
        else:                                                             # HARD IVF: top-nprobe buckets -> EXACT refine
            bk=kl.argmax(-1)                                               # key bucket (B,T)
            tp=ql.topk(min(s.nprobe,s.vatoms),-1).indices                 # query top-nprobe buckets (B,T,nprobe)
            allow=torch.zeros(B,T,s.vatoms,device=x.device,dtype=torch.bool).scatter_(2,tp,True)
            khot=F.one_hot(bk,s.vatoms).to(att.dtype)                      # (B,T,V)
            m=(allow.to(att.dtype)@khot.transpose(1,2))>0                  # (B,T,T) key j in a queried bucket
            m=m&causal[None]
            empty=~m.any(-1,keepdim=True); diag=torch.eye(T,device=x.device,dtype=torch.bool)[None]
            m=m|(empty&diag)                                              # fallback only for empty rows -> no NaN
            if s.refine=='rerank' and khq is not None:                    # 2-stage: cheap ADC shortlist -> EXACT rerank of top-kshort
                adc=(qs@khq.transpose(-1,-2)).sum(1)                      # (B,T,T) full-vector ADC score (sum over heads)
                adc=adc.masked_fill(~m,float('-inf')); ks=min(s.kshort,T)
                thr=adc.topk(ks,-1).values[...,-1,None]; m=m&(adc>=thr)   # keep only ADC top-kshort; att below is EXACT
            att=att.masked_fill(~m.unsqueeze(1),float('-inf'))
        y=(att.softmax(-1)@vh).transpose(1,2).reshape(B,T,D)
        return s.o(y)
    def _pq_quantize(s,kh):                                               # kh (B,H,T,hd) -> PQ-reconstructed (asymmetric ADC on inner product)
        B,H,T,hd=kh.shape; DS=s.pq_ds; M=hd//DS; c=s.pq_cent              # c: (H,M,16,DS) per-head per-subq centroids
        x=kh.transpose(1,2).reshape(B,T,H,M,DS)                           # (B,T,H,M,DS)
        d=((x.unsqueeze(-2)-c[None,None])**2).sum(-1)                     # (B,T,H,M,16) L2 to each centroid
        idx=d.argmin(-1)                                                  # (B,T,H,M) nearest code
        rec=torch.gather(c[None,None].expand(B,T,-1,-1,-1,-1),4,idx[...,None,None].expand(-1,-1,-1,-1,1,DS)).squeeze(4)
        return rec.reshape(B,T,H,hd).transpose(1,2)                       # (B,H,T,hd)
    @torch.no_grad()
    def fit_pq(s,kbuf,ncodes=16,iters=12,seed=0):                         # k-means per (head,subquantizer) on collected keys
        K=torch.cat(kbuf,0)                                              # (N,H,T,hd) -> flatten
        N,H,T,hd=K.shape; DS=s.pq_ds; M=hd//DS
        X=K.permute(1,0,2,3).reshape(H,N*T,M,DS).permute(0,2,1,3).contiguous()  # (H,M,Npts,DS)
        g=torch.Generator(device=X.device).manual_seed(seed)
        cent=torch.empty(H,M,ncodes,DS,device=X.device)
        for h in range(H):
            for m in range(M):
                pts=X[h,m]                                                # (Npts,DS)
                ci=pts[torch.randint(0,pts.shape[0],(ncodes,),generator=g,device=X.device)]  # init
                for _ in range(iters):
                    d=((pts[:,None]-ci[None])**2).sum(-1); a=d.argmin(1)  # assign
                    for kk in range(ncodes):
                        sel=pts[a==kk]
                        if sel.shape[0]>0: ci[kk]=sel.mean(0)
                cent[h,m]=ci
        s.pq_cent=cent; return cent

def routing_aux(rt,match,w_info,w_kmeans,w_bal,aux_scale=1.0):  # codebook-learning objective; the ONLY diff between the 3 arms
    if rt.learn=='random': return rt.ql.new_zeros(())                      # frozen codebook -> pure CE through soft gate (LSH)
    ql=rt.ql.float(); kl=rt.kl.float(); B,T,V=ql.shape
    qa=F.softmax(ql,-1)
    imp=qa.mean((0,1)); hard=F.one_hot(ql.argmax(-1),V).float().mean((0,1)); bal=V*(imp*hard).sum()  # load-balance (anti-collapse)
    rt._bal=float(bal.detach())
    if rt.learn=='infonce':                                               # query routes to its MATCHED key's bucket (task-aware)
        ka=F.softmax(kl,-1); S=torch.bmm(qa,ka.transpose(1,2))/0.1         # (B,T,T) routing similarity
        valid=(match>=0); logp=F.log_softmax(S,-1)
        picked=logp.gather(2,match.clamp(min=0).unsqueeze(-1)).squeeze(-1) # logp at the matched key position
        obj=w_info*(-(picked*valid).sum()/valid.sum().clamp(min=1))
    else:                                                                 # kmeans: VQ commitment (pull keys to nearest centroid)
        obj=w_kmeans*(1-kl.max(-1).values).mean()
    return aux_scale*obj+w_bal*bal                                        # aux_scale warms up obj; balance always on (anti-collapse)

class Layer(nn.Module):
    def __init__(s,D,mix): super().__init__(); s.norm=nn.LayerNorm(D); s.mix=mix
    def forward(s,x): return x+s.mix(s.norm(x))

MAXPOS=8192
# Track A / R-B arms -> (codebook-learning mode, partition). simvq-fixed = arm B: InfoNCE encoder + DATA-INDEPENDENT Hadamard partition.
SIMVQ={'simvq':('infonce','learned'),'simvq-kmeans':('kmeans','learned'),'simvq-random':('random','learned'),'simvq-fixed':('infonce','fixed')}
class Model(nn.Module):
    def __init__(s,vocab,D,L,arm,H,window,topk,N,vatoms=256,nprobe=4,rtau=1.0,qnorm=False):
        super().__init__(); s.emb=nn.Embedding(vocab,D); s.pos=nn.Embedding(MAXPOS,D); slot=L-1
        mds={'swa':'window','sparse':'topk','attn':'full'}
        s.slot=slot; s.routed=None; s._cap_h=False; s._hbuf=None; layers=[]      # _cap_h: mechanism gate (||h_t|| vs position)
        for i in range(L):
            if i==slot and arm in SIMVQ:
                lrn,prt=SIMVQ[arm]; rt=RoutedAttn(D,H,vatoms,nprobe,rtau,lrn,prt,qnorm); s.routed=rt; layers.append(Layer(D,rt))
            elif i==slot and arm!='ssm': layers.append(Layer(D,Attn(D,H,mds[arm],window,topk)))
            else:                        layers.append(Layer(D,SSM(D,N=N)))
        s.layers=nn.ModuleList(layers); s.norm=nn.LayerNorm(D); s.head=nn.Linear(D,vocab,bias=False); s.use_ckpt=True
    def forward(s,x):
        B,T=x.shape
        h=s.emb(x)+s.pos(torch.arange(T,device=x.device))[None]   # learned abs positions: value=token AFTER matched key needs position
        if s._cap_h: s._hbuf=[h.detach().norm(dim=-1).mean(0)]     # (T,) mean per-position L2 norm; index 0 = embedding
        for i,l in enumerate(s.layers):                     # checkpoint SSM/attn; NOT the routed slot (aux reads its cached logits)
            do=s.training and s.use_ckpt and not (s.routed is not None and i==s.slot)
            h=ckpt.checkpoint(l,h,use_reentrant=False) if do else l(h)
            if s._cap_h: s._hbuf.append(h.detach().norm(dim=-1).mean(0))
        return s.head(s.norm(h))

# ---------------- distance-stratified recall ----------------
BINS=[(0,256),(256,512),(512,1024),(1024,2048),(2048,4096),(4096,8192),(8192,1<<20)]
POSBINS=[(0,512),(512,1024),(1024,2048),(2048,4096),(4096,8192),(8192,16384),(16384,32768),(32768,1<<20)]  # ABSOLUTE query position (SSM-state derivation)
def eval_recall(model,a,NK,NV,dev,seed=777,quick=False,spread=False):
    model.eval(); gen=torch.Generator().manual_seed(seed)
    hit={b:0 for b in range(len(BINS))}; tot={b:0 for b in range(len(BINS))}; ov_h=ov_t=0
    nb=2 if quick else a.eval_batches
    with torch.no_grad():
        for _ in range(nb):
            x,tgt,dist,_=make_batch(a.batch,a.ctx,a.pairs,a.queries,NK,NV,gen,dev,spread=spread)
            with torch.autocast(dev,dtype=torch.float16 if a.fp16 else torch.bfloat16,enabled=(a.fp16 or a.bf16) and dev=='cuda'):
                logits=model(x)
            pred=logits.argmax(-1); qm=tgt!=-100
            correct=(pred==tgt)&qm
            d=dist[qm]; c=correct[qm].long()
            for bi,(lo,hi) in enumerate(BINS):
                sel=(d>=lo)&(d<hi); tot[bi]+=int(sel.sum()); hit[bi]+=int((c[sel]).sum())
            ov_t+=int(qm.sum()); ov_h+=int(correct.sum())
    model.train()
    return hit,tot,ov_h/max(ov_t,1)

# R-B apparatus metric: recall stratified by ABSOLUTE QUERY POSITION (primary) AND key->query DISTANCE (secondary).
# Drift signature = recall flat at low position, degrading at high position (where the SSM state is most derived).
def eval_strat(model,a,NK,NV,dev,seed=777,spread=True):
    model.eval(); gen=torch.Generator().manual_seed(seed)
    ph={b:0 for b in range(len(POSBINS))}; pt={b:0 for b in range(len(POSBINS))}
    dh={b:0 for b in range(len(BINS))};    dt={b:0 for b in range(len(BINS))}; ov_h=ov_t=0
    with torch.no_grad():
        for _ in range(a.eval_batches):
            x,tgt,dist,_=make_batch(a.batch,a.ctx,a.pairs,a.queries,NK,NV,gen,dev,spread=spread)
            with torch.autocast(dev,dtype=torch.float16 if a.fp16 else torch.bfloat16,enabled=(a.fp16 or a.bf16) and dev=='cuda'):
                logits=model(x)
            pred=logits.argmax(-1); qm=tgt!=-100; correct=(pred==tgt)&qm
            T=x.shape[1]; pos=torch.arange(T,device=dev).unsqueeze(0).expand(qm.shape[0],T)
            d=dist[qm]; pp=pos[qm]; c=correct[qm].long()
            for bi,(lo,hi) in enumerate(BINS):    sel=(d>=lo)&(d<hi);  dt[bi]+=int(sel.sum()); dh[bi]+=int(c[sel].sum())
            for bi,(lo,hi) in enumerate(POSBINS): sel=(pp>=lo)&(pp<hi); pt[bi]+=int(sel.sum()); ph[bi]+=int(c[sel].sum())
            ov_t+=int(qm.sum()); ov_h+=int(correct.sum())
    model.train()
    return ph,pt,dh,dt,ov_h/max(ov_t,1)

def run_arm(arm,a,vocab,NK,NV,dev,deadline,resume_ck=None):
    torch.manual_seed(a.seed)
    model=Model(vocab,a.dim,a.layers,arm,a.heads,a.window,a.topk,a.nstate,a.vatoms,a.nprobe,a.route_tau,a.qnorm).to(dev); model.use_ckpt=not a.no_ckpt
    npar=sum(p.numel() for p in model.parameters())
    rt=model.routed                                  # the routed slot (Track A simvq arms), else None
    if a.compile and rt is not None: print("  (simvq arm caches routing logits eagerly -> --compile disabled for it)");
    elif a.compile:
        try: model=torch.compile(model)
        except Exception as e: print(f"  compile failed ({e})")
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(0.9,0.95),weight_decay=0.1)
    scaler=torch.amp.GradScaler('cuda',enabled=(a.fp16 and dev=='cuda'))   # fp16 needs loss-scaling; no-op for fp32/bf16 (bf16 has fp32 exponent range -> no overflow/nan)
    gen=torch.Generator().manual_seed(a.seed)
    start=0; best=-1.0; stale=0   # plateau early-stop: only ARMED once recall>grok-thr (never cuts the pre-jump plateau)
    if resume_ck is not None:                                          # RESUME: restore weights+opt+step+rng -> continue in place
        m=(model._orig_mod if hasattr(model,'_orig_mod') else model)
        m.load_state_dict(resume_ck['model']); opt.load_state_dict(resume_ck['opt'])
        try: scaler.load_state_dict(resume_ck['scaler'])
        except Exception: pass
        try: gen.set_state(resume_ck['gen'])
        except Exception: pass
        start=resume_ck['step']; best=resume_ck['best']; stale=resume_ck['stale']
        print(f"    [{arm:8s}] RESUME from step {start}/{a.steps} (best={best*100:.1f}% stale={stale})",flush=True)
    elif a.init_from:                                                 # WARM-START: load WEIGHTS only (fresh opt/step) -> carry the grokked circuit to higher pairs/ctx
        ip=a.init_from if a.init_from.endswith('.pt') else os.path.join(a.init_from,arm,f"ckpt_{arm}.pt")
        if os.path.exists(ip):
            m=(model._orig_mod if hasattr(model,'_orig_mod') else model)
            sd=torch.load(ip,map_location=dev,weights_only=False)['model']
            if rt is not None and getattr(rt,'_fixed',False): sd={k:v for k,v in sd.items() if k!='routed.C'}  # arm B: KEEP the fixed Hadamard partition; warm-start only encoder+backbone (one variable)
            miss,unexp=m.load_state_dict(sd,strict=False)                                                      # strict=False: A and B share ONE base (e.g. simvq); partition differs
            print(f"    [{arm:8s}] WARM-START weights from {ip} (fresh optimizer/step=0; missing={len(miss)} unexpected={len(unexp)})",flush=True)
        else:
            print(f"    [{arm:8s}] WARN: --init-from has no ckpt at {ip} -> training FROM SCRATCH",flush=True)
    ckpath=os.path.join(a.out_dir,f"ckpt_{arm}.pt")
    def save_ck(step,finished,results=None):                          # atomic: write tmp then rename (no corrupt ckpt on kill)
        m=(model._orig_mod if hasattr(model,'_orig_mod') else model)
        cfg=dict(vars(a)); cfg['arm']=arm                             # store the SPECIFIC arm (not the run-level --arm like 'drift'/'ramp') -> reload rebuilds the right model
        ck={'arm':arm,'step':step,'done':finished,'best':best,'stale':stale,'npar':npar,'results':results,'cfg':cfg,
            'model':m.state_dict(),'opt':opt.state_dict(),'scaler':scaler.state_dict(),'gen':gen.get_state()}
        tmp=ckpath+'.tmp'; torch.save(ck,tmp); os.replace(tmp,ckpath)
    ev=a.eval_every if a.eval_every>0 else max(1,a.steps//8)
    model.train(); t0=time.time(); done=start; finished=False
    for step in range(start,a.steps):
        x,tgt,_,match=make_batch(a.batch,a.ctx,a.pairs,a.queries,NK,NV,gen,dev,spread=a.spread)
        opt.zero_grad()
        asc=1.0 if a.aux_warmup<=0 else (0.0 if step<a.aux_warmup else min(1.0,(step-a.aux_warmup)/a.aux_warmup))  # CE-only then ramp
        with torch.autocast(dev,dtype=torch.float16 if a.fp16 else torch.bfloat16,enabled=(a.fp16 or a.bf16) and dev=='cuda'):
            logits=model(x); loss=F.cross_entropy(logits.reshape(-1,vocab),tgt.reshape(-1),ignore_index=-100)
            if rt is not None: loss=loss+routing_aux(rt,match,a.infonce_w,a.kmeans_w,a.balance_w,asc)  # codebook-learning aux (warmup-scaled)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update(); done=step+1
        if step==start or (step+1)%ev==0:
            ov=eval_recall(model,a,NK,NV,dev,quick=True,spread=a.spread)[2]   # periodic overall recall: MQAR plateaus then JUMPS
            armed=ov>=a.grok_thr                              # only consider stopping AFTER the grok jump (high recall)
            if ov>best+0.01: best=ov; stale=0
            elif armed: stale+=1
            bal=f" bal={rt._bal:.2f}" if rt is not None else ""   # load-balance: 1.0=uniform buckets, high=collapse
            aux=f" aux={asc:.2f}" if (rt is not None and a.aux_warmup>0) else ""
            print(f"    [{arm:8s}] step {step+1:5d}/{a.steps} loss={loss.item():.4f} recall={ov*100:4.1f}% best={best*100:4.1f}% stale={stale}/{a.patience}{' [armed]' if armed else ''}{bal}{aux} ({(time.time()-t0)/max(1,step+1-start)*1000:.0f} ms/step)",flush=True)
            if armed and stale>=a.patience:
                print(f"    [{arm:8s}] CONVERGED (plateau {best*100:.1f}% above grok-thr) at step {done}/{a.steps}",flush=True); finished=True; break
        if (step+1)%a.ckpt_every==0: save_ck(done,False)              # periodic resumable checkpoint
        if time.time()>deadline:
            print(f"    [{arm:8s}] TIME BUDGET hit at step {done}/{a.steps} -> save partial (RESUMABLE, not converged)",flush=True); save_ck(done,False); break
    else:
        finished=True                                                 # loop ran all steps without break = converged-by-exhaustion
    hit,tot,ov=eval_recall(model,a,NK,NV,dev,spread=a.spread)
    res={'npar':npar,'hit':hit,'tot':tot,'ov':ov,'steps':done}
    save_ck(done,finished,results=res if finished else None)          # DONE ckpt embeds results -> resume skips it
    sd=(model._orig_mod if hasattr(model,'_orig_mod') else model).state_dict() if a.save_models else None
    return npar,hit,tot,ov,done,sd,finished

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arm",default="all",choices=["ssm","swa","sparse","attn","all","simvq","simvq-kmeans","simvq-random","simvq-fixed","simvq-all","ramp","drift"])
    ap.add_argument("--ctx",type=int,default=2048); ap.add_argument("--pairs",type=int,default=64); ap.add_argument("--queries",type=int,default=32)
    ap.add_argument("--dim",type=int,default=128); ap.add_argument("--layers",type=int,default=3); ap.add_argument("--heads",type=int,default=4)
    ap.add_argument("--nstate",type=int,default=16); ap.add_argument("--window",type=int,default=512); ap.add_argument("--topk",type=int,default=16)
    ap.add_argument("--vatoms",type=int,default=256,help="Track A: codebook/IVF atoms"); ap.add_argument("--nprobe",type=int,default=4,help="Track A: query routes to top-nprobe buckets")
    ap.add_argument("--route-tau",type=float,default=1.0,help="Track A: soft-routing softmax temperature (train gate)")
    ap.add_argument("--infonce-w",type=float,default=0.05,help="InfoNCE aux weight (was 1.0 -> at pairs>=16 the 16 contrastive negatives let it DOMINATE & block CE grokking; lowered)")
    ap.add_argument("--kmeans-w",type=float,default=1.0); ap.add_argument("--balance-w",type=float,default=0.01)
    ap.add_argument("--aux-warmup",type=int,default=0,help="CE-ONLY for first N steps, then linearly ramp aux over the next N (lets CE reach the grok scarp before the codebook aux applies; balance-reg stays on)")
    ap.add_argument("--spread",action="store_true",help="R-B drift apparatus: SPREAD queries across context (position decoupled from distance). Train+eval with it.")
    ap.add_argument("--no-qnorm",dest="qnorm",action="store_false",help="R-B: disable l2-normalize-query-before-score (default ON). Use for the radial-drift ablation.")
    ap.set_defaults(qnorm=True)
    ap.add_argument("--keys",type=int,default=512); ap.add_argument("--vals",type=int,default=512)
    ap.add_argument("--steps",type=int,default=4000); ap.add_argument("--batch",type=int,default=16); ap.add_argument("--lr",type=float,default=3e-3)
    ap.add_argument("--eval-every",type=int,default=0,help="eval recall every N steps (0=steps//8). Frequent eval catches the grok jump.")
    ap.add_argument("--grok-thr",type=float,default=0.60,help="early-stop ONLY allowed once recall exceeds this (never stops in the pre-jump low-recall plateau)")
    ap.add_argument("--patience",type=int,default=6,help="stop after this many consecutive evals with no best-recall improvement, AND only above grok-thr")
    ap.add_argument("--no-ckpt",action="store_true",help="disable gradient checkpointing (more memory, less compute)")
    ap.add_argument("--eval-batches",type=int,default=8); ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--fp16",action="store_true",help="T4/Turing: fp16 autocast"); ap.add_argument("--bf16",action="store_true",help="Ampere+: bf16 autocast")
    ap.add_argument("--compile",action="store_true"); ap.add_argument("--device",default="auto")
    ap.add_argument("--scan-chunk",type=int,default=128,help="SSM chunked-scan block (fewer kernel launches)")
    ap.add_argument("--out-dir",default="results/phase56",help="where to SAVE results (use /kaggle/working/... on Kaggle)")
    ap.add_argument("--time-budget-min",type=float,default=330,help="wall-clock budget; stop+save before the 6h T4 kill")
    ap.add_argument("--save-models",action="store_true",help="also save per-arm model state_dict")
    ap.add_argument("--ckpt-every",type=int,default=1000,help="checkpoint (weights+opt+step+rng) every N steps -> a kill loses only the last interval")
    ap.add_argument("--resume",default="",help="path to prior --out-dir OR a flat folder of ckpt_*.pt files from multiple arms (auto-detected by the 'arm' field inside each checkpoint)")
    ap.add_argument("--init-from",default="",help="WARM-START: dir (looks for <dir>/<arm>/ckpt_<arm>.pt) or a .pt; loads WEIGHTS only (fresh opt/step). Carries a grokked low-load circuit to higher pairs. Ignored if --resume finds a checkpoint.")
    ap.add_argument("--smoke",action="store_true")
    a=ap.parse_args()
    global SCAN_CHUNK; SCAN_CHUNK=a.scan_chunk
    if a.smoke: a.ctx=256; a.pairs=16; a.queries=8; a.dim=64; a.layers=2; a.steps=300; a.batch=16; a.eval_batches=4; a.vatoms=64
    # NOTE: resume no longer forces out_dir=resume. Resume dir is READ-ONLY; new progress goes to --out-dir.
    dev = a.device if a.device!="auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    NK,NV=a.keys,a.vals; vocab=1+NK+NV
    os.makedirs(a.out_dir,exist_ok=True)
    deadline=time.time()+a.time_budget_min*60
    print(f"MQAR probe | ctx={a.ctx} pairs={a.pairs} queries={a.queries} | D={a.dim} L={a.layers} H={a.heads} N={a.nstate} win={a.window} topk={a.topk} | vocab={vocab} dev={dev} fp16={a.fp16} bf16={a.bf16} | out={a.out_dir} budget={a.time_budget_min}min",flush=True)
    arms=(["ssm","swa","sparse","attn"] if a.arm=="all" else
          ["simvq","simvq-kmeans","simvq-random"] if a.arm=="simvq-all" else
          ["sparse","simvq","simvq-random"] if a.arm=="ramp" else
          ["simvq","simvq-fixed"] if a.arm=="drift" else [a.arm])   # drift = A(learned partition) vs B(data-independent Hadamard); encoder=InfoNCE both
    # --- recall-slot self-check: print the distinct mixer per arm; guard the swa==attn trap (win>=ctx = window mask is a no-op) ---
    rstr="RoutedAttn(V=%d,nprobe=%d,exact-refine,qnorm=%s)"%(a.vatoms,a.nprobe,a.qnorm)
    mds={'ssm':'SSM','swa':"Attn(mode=window,win=%d)"%a.window,'sparse':"Attn(mode=topk,k=%d)"%a.topk,'attn':'Attn(mode=full)',
         'simvq':rstr+'/InfoNCE-learnedC','simvq-kmeans':rstr+'/kmeans','simvq-random':rstr+'/random-frozen','simvq-fixed':rstr+'/InfoNCE-HadamardC(data-indep)'}
    print("  recall-slot per arm: "+" | ".join(f"{ar}->{mds[ar]}" for ar in arms),flush=True)
    if 'swa' in arms and a.window>=a.ctx:
        print(f"  !! WARNING: window({a.window}) >= ctx({a.ctx}) -> SWA window mask never bites -> swa is BIT-IDENTICAL to attn (not a real arm). Set --window < ctx (e.g. {a.ctx//4}).",flush=True)
    results={}
    def save_all():   # incremental: written after EVERY arm so a kill loses only the current arm
        import json
        active=[bi for bi in range(len(BINS)) if any(results[ar]["tot"][bi]>0 for ar in results)]
        lines=["==== RECALL @ key->query distance (accuracy) ====",
               "  arm      params  steps overall "+" ".join(f"{BINS[bi][0]}-{BINS[bi][1] if BINS[bi][1]<(1<<20) else 'inf'}".rjust(10) for bi in active)]
        for arm in results:
            r=results[arm]; cells=" ".join((f"{r['hit'][bi]/r['tot'][bi]*100:.0f}%" if r['tot'][bi]>0 else "-").rjust(10) for bi in active)
            lines.append(f"  {arm:7s} {r['npar']/1e6:5.3f}M {r['steps']:5d} {r['ov']*100:5.1f}%  {cells}")
        lines.append("read: [ssm/swa/sparse/attn] ssm collapses, sparse~attn ceiling. [simvq] gate = simvq>=88% (90% of 97.7% exact ceiling) AND simvq>kmeans>random (InfoNCE routes better).")
        txt="\n".join(lines)
        open(os.path.join(a.out_dir,"mqar_recall.txt"),"w").write(txt)
        json.dump({"config":vars(a),"results":{ar:{k:(v if not isinstance(v,dict) else {str(kk):vv for kk,vv in v.items()}) for k,v in results[ar].items()} for ar in results}},
                  open(os.path.join(a.out_dir,"mqar_results.json"),"w"),indent=1)
        return txt
    # Build resume map: scan --resume dir for checkpoint files, index by ck['arm']
    resume_map = {}  # arm_name -> loaded checkpoint dict
    if a.resume:
        rdir = a.resume
        if os.path.isdir(rdir):
            for fname in os.listdir(rdir):
                if not fname.endswith('.pt'): continue
                fpath = os.path.join(rdir, fname)
                try:
                    ck = torch.load(fpath, map_location=dev)
                    if isinstance(ck, dict) and 'arm' in ck:
                        resume_map[ck['arm']] = ck
                        print(f"  [resume-scan] {fname} -> arm='{ck['arm']}' step={ck.get('step')} done={ck.get('done')}", flush=True)
                except Exception as e:
                    print(f"  [resume-scan] skipping {fname}: {e}", flush=True)
        elif os.path.isfile(rdir):  # single checkpoint file
            try:
                ck = torch.load(rdir, map_location=dev)
                if isinstance(ck, dict) and 'arm' in ck:
                    resume_map[ck['arm']] = ck
                    print(f"  [resume-scan] {os.path.basename(rdir)} -> arm='{ck['arm']}' step={ck.get('step')} done={ck.get('done')}", flush=True)
            except Exception as e:
                print(f"  [resume-scan] failed to load {rdir}: {e}", flush=True)
        if resume_map:
            print(f"  [resume] found checkpoints for arms: {list(resume_map.keys())}", flush=True)
        else:
            print(f"  [resume] WARNING: no valid checkpoints found in {rdir}", flush=True)

    for arm in arms:
        resume_ck = resume_map.get(arm, None)
        if resume_ck is not None:
            prev_steps = resume_ck.get('cfg', {}).get('steps', 0)
            converged_early = resume_ck.get('done') and resume_ck['step'] < prev_steps
            if resume_ck.get('done') and (resume_ck['step']>=a.steps or converged_early) and resume_ck.get('results') is not None:
                results[arm]=resume_ck['results']; print(f"  == arm {arm}: DONE @step{resume_ck['step']} recall={resume_ck['results']['ov']*100:.1f}% -> skip ==",flush=True)
                print("\n"+save_all()+"\n",flush=True); continue
        if time.time()>deadline: print(f"  budget exhausted before arm {arm}; stopping.",flush=True); break
        print(f"  == arm: {arm} =={' (RESUME)' if resume_ck is not None else ''}",flush=True)
        npar,hit,tot,ov,steps,sd,finished=run_arm(arm,a,vocab,NK,NV,dev,deadline,resume_ck)
        results[arm]={"npar":npar,"hit":hit,"tot":tot,"ov":ov,"steps":steps}
        if not finished: print(f"  !! arm {arm} stopped at step {steps} NOT converged (time budget) -> rerun with --resume {a.out_dir} (raise --time-budget-min/--steps)",flush=True)
        if sd is not None: torch.save(sd,os.path.join(a.out_dir,f"mqar_{arm}.pt"))
        print("\n"+save_all()+f"\n  [saved -> {a.out_dir}/mqar_recall.txt + .json | ckpt_{arm}.pt]\n",flush=True)   # persist after each arm
    print("STOP. recall@distance saved incrementally. CPU gather cost = phase56_gather_profile. No commit.",flush=True)

if __name__=="__main__": main()
