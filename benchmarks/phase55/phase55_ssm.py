#!/usr/bin/env python3
# Phase 55 (2) - QUALITY BASELINE: pure SSM-Arch-A (~1.6M) in PyTorch, FP, CE only.
#   Arch-A = selective diagonal SSM (Mamba-1 style) x3 + 1 sliding-window-attention layer (window 128) at layer 3.
#   D=192 N=64 H=6 L=4, expand=2 (d_inner=384), dt_rank=12, conv=4, vocab=1024 (BPE-1024).
#   NO quantization, NO DAgger, NO multi-loss. Just CE. PyTorch -> gradient-check + determinism for free.
#   Measures val BPB (bits/byte, unit-invariant) + closed-loop generation -> 8-10 samples + gate-v2/byte-guard.
#
# This is the REFERENCE (quality + thread) the rest is decided against. Smoke + STOP; user launches real train.
#
# Data: results/phase55/ids.u16 (+ meta.bin) from phase55_tokdump (same BPE-1024 as the project).
# Run (smoke): python benchmarks/phase55/phase55_ssm.py --smoke
# Run (real) : python benchmarks/phase55/phase55_ssm.py --steps 20000 --seq 256 --batch 16 --lr 3e-3
import argparse, json, math, os, struct, sys, time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")        # deterministic cublas (set before torch init)
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint as ckpt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IDS  = os.path.join(ROOT, "results", "phase55", "ids.u16")
META = os.path.join(ROOT, "results", "phase55", "meta.bin")
BARS = os.path.join(ROOT, "docs", "gatev2_bars.json")
OUT  = os.path.join(ROOT, "results", "phase55")
# Creating an output directory AT IMPORT fails on a read-only mount, and this module is imported for its
# symbols far more often than it is run: on Kaggle the code ships as a dataset under /kaggle/input, so the
# import raised OSError and killed a run before it began. Writers below still create what they need.
try: os.makedirs(os.path.join(OUT, "human"), exist_ok=True)
except OSError: pass

# ---------------- data ----------------
def load_meta(path):
    with open(path,"rb") as f:
        mg,V,nt = struct.unpack("<III", f.read(12))
        assert mg==0x54444D50, "bad meta magic"
        el = np.frombuffer(f.read(V), dtype=np.uint8).astype(np.int64)
        id2bytes=[]
        for t in range(V):
            id2bytes.append(f.read(int(el[t])))
    return V, el, id2bytes

# ---------------- model ----------------
class RMSNorm(nn.Module):
    def __init__(s,d,eps=1e-5): super().__init__(); s.w=nn.Parameter(torch.ones(d)); s.eps=eps
    def forward(s,x): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+s.eps)*s.w

class SSMBlock(nn.Module):
    # Mamba-1 selective diagonal SSM
    def __init__(s,D,N,expand=2,dt_rank=12,conv=4):
        super().__init__(); s.D=D; s.N=N; s.Dn=D*expand; s.dt_rank=dt_rank; s.conv=conv
        s.in_proj=nn.Linear(D,2*s.Dn,bias=False)
        s.conv1d=nn.Conv1d(s.Dn,s.Dn,conv,groups=s.Dn,padding=conv-1,bias=True)
        s.x_proj=nn.Linear(s.Dn,dt_rank+2*N,bias=False)
        s.dt_proj=nn.Linear(dt_rank,s.Dn,bias=True)
        s.A_log=nn.Parameter(torch.log(torch.arange(1,N+1,dtype=torch.float32).repeat(s.Dn,1)))
        s.Dskip=nn.Parameter(torch.ones(s.Dn))
        s.out_proj=nn.Linear(s.Dn,D,bias=False)
    def forward(s,x):
        B,L,_=x.shape
        xz=s.in_proj(x); xx,z=xz.chunk(2,dim=-1)
        xx=s.conv1d(xx.transpose(1,2))[:,:,:L].transpose(1,2)
        xx=F.silu(xx)
        dbl=s.x_proj(xx); dt,Bm,Cm=torch.split(dbl,[s.dt_rank,s.N,s.N],dim=-1)
        dt=F.softplus(s.dt_proj(dt))                     # (B,L,Dn)
        A=-torch.exp(s.A_log.float())                    # (Dn,N)
        # SEQUENTIAL-state selective scan (the memory-safe form: only the (B,Dn,N) state is carried, so backward
        # under checkpointing stays small -> fits batch32/seq512 in 12GB). fp32 for stability. For speed, wrap the
        # whole model in torch.compile (--compile): inductor fuses this loop (official PyTorch path), no extra memory.
        with torch.autocast(device_type=x.device.type, enabled=False):
            dA=torch.exp(dt.float().unsqueeze(-1)*A)                 # (B,L,Dn,N)
            dBx=dt.float().unsqueeze(-1)*Bm.float().unsqueeze(2)*xx.float().unsqueeze(-1)
            dA_t=dA.unbind(1); dBx_t=dBx.unbind(1); C_t=Cm.float().unbind(1)
            h=torch.zeros(B,s.Dn,s.N,device=x.device,dtype=torch.float32); ys=[]
            for t in range(L):
                h=dA_t[t]*h+dBx_t[t]
                ys.append((h*C_t[t].unsqueeze(1)).sum(-1))           # (B,Dn)
            y=torch.stack(ys,1)+xx.float()*s.Dskip.float()          # (B,L,Dn)
        y=y.to(x.dtype)
        y=y*F.silu(z)
        return s.out_proj(y)

class SWABlock(nn.Module):
    def __init__(s,D,H,window=128):
        super().__init__(); s.H=H; s.hd=D//H; s.win=window
        s.qkv=nn.Linear(D,3*D,bias=False); s.o=nn.Linear(D,D,bias=False)
    def forward(s,x):
        B,L,D=x.shape; H,hd=s.H,s.hd
        q,k,v=s.qkv(x).split(D,dim=-1)
        q=q.view(B,L,H,hd).transpose(1,2); k=k.view(B,L,H,hd).transpose(1,2); v=v.view(B,L,H,hd).transpose(1,2)
        att=(q@k.transpose(-1,-2))/math.sqrt(hd)          # (B,H,L,L)
        idx=torch.arange(L,device=x.device)
        causal=idx[None,:]<=idx[:,None]
        window=idx[None,:]>idx[:,None]-s.win
        mask=causal&window
        att=att.masked_fill(~mask,float("-inf"))
        att=att.softmax(-1)
        y=(att@v).transpose(1,2).reshape(B,L,D)
        return s.o(y)

class MLP(nn.Module):
    def __init__(s,D,hid): super().__init__(); s.fc1=nn.Linear(D,hid,bias=False); s.fc2=nn.Linear(hid,D,bias=False)
    def forward(s,x): return s.fc2(F.silu(s.fc1(x)))

class Block(nn.Module):
    def __init__(s,D,N,H,is_swa,use_mlp=False,mlp_mult=4,dt_rank=12):
        super().__init__(); s.norm=RMSNorm(D)
        s.mix=SWABlock(D,H) if is_swa else SSMBlock(D,N,dt_rank=dt_rank)
        s.use_mlp=use_mlp
        if use_mlp: s.norm2=RMSNorm(D); s.mlp=MLP(D,mlp_mult*D)
    def forward(s,x):
        x=x+s.mix(s.norm(x))
        if s.use_mlp: x=x+s.mlp(s.norm2(x))
        return x

class ArchA(nn.Module):
    def __init__(s,V,D=192,N=64,H=6,L=4,swa_layer=3,use_mlp=False,mlp_mult=4,dt_rank=12):
        super().__init__(); s.emb=nn.Embedding(V,D)
        s.blocks=nn.ModuleList([Block(D,N,H,is_swa=(i==swa_layer),use_mlp=use_mlp,mlp_mult=mlp_mult,dt_rank=dt_rank) for i in range(L)])
        s.norm_f=RMSNorm(D); s.head=nn.Linear(D,V,bias=False); s.use_ckpt=True
    def forward(s,idx):
        x=s.emb(idx)
        for b in s.blocks:
            # gradient checkpointing: don't retain the 512-step scan graph (would be ~15GB at b32/seq512);
            # recompute in backward instead (~2x compute, ~O(1) activation memory). same math.
            if s.training and s.use_ckpt: x=ckpt.checkpoint(b, x, use_reentrant=False)
            else: x=b(x)
        return s.head(s.norm_f(x))

# ---------------- gate metrics (Python port of phase54.ps1 / gate-v2) ----------------
NAME_WORDS=set("lily max mom mommy mum mummy mia tim tom ben sam sue dad daddy anna lucy jack sara my spot bella leo amy".split())
def byte_guard(b):
    mWs=mCh=ws=ch=wc=nonp=0; prev=-1
    for x in b:
        if x in (32,9,10,13): wc+=1; ws+=1; mWs=max(mWs,ws); ch=0
        else:
            ws=0; ch=ch+1 if x==prev else 1; mCh=max(mCh,ch)
            if x<32 or x>126: nonp+=1
        prev=x
    n=max(len(b),1)
    return dict(wsRun=mWs,chRun=mCh,wsFrac=round(wc/n,4),nonPrint=nonp)
def word_metrics(text):
    import re
    toks=[w for w in re.sub(r'[^a-zA-Z]',' ',text).lower().split() if len(w)>=2]
    n=len(toks)
    if n<4: return None
    nameish=round(sum(1 for w in toks if w in NAME_WORDS)/n*100,1)
    run=mr=1
    for j in range(1,n): run=run+1 if toks[j]==toks[j-1] else 1; mr=max(mr,run)
    bf={}
    for j in range(n-1): k=toks[j]+' '+toks[j+1]; bf[k]=bf.get(k,0)+1
    topbi=max(bf.values()) if bf else 0
    al=am=0
    for j in range(n-2): al=al+1 if toks[j]==toks[j+2] else 0; am=max(am,al)
    return dict(topBi=topbi,altLp=am,runWst=mr,nameWst=nameish)

# ---------------- train / eval / gen ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--steps",type=int,default=20000)
    ap.add_argument("--seq",type=int,default=256)
    ap.add_argument("--batch",type=int,default=16)
    ap.add_argument("--lr",type=float,default=3e-3)
    ap.add_argument("--bf16",action="store_true")
    ap.add_argument("--eval-tok",type=int,default=200000)
    ap.add_argument("--epochs",type=float,default=0.0,help="if >0, overrides --steps to cover this many epochs of train tokens")
    ap.add_argument("--gen-bytes",type=int,default=2000)
    ap.add_argument("--gen-samples",type=int,default=4,help="samples PER temperature (0.65,0.55); 0 skips generation")
    ap.add_argument("--save",type=str,default="",help="path to save the trained state_dict (optional; not a commit)")
    ap.add_argument("--device",type=str,default="auto")
    ap.add_argument("--no-ckpt",action="store_true",help="disable gradient checkpointing (uses more memory)")
    ap.add_argument("--deterministic",action="store_true",help="force deterministic algos (slower; off by default for speed)")
    ap.add_argument("--accum",type=int,default=1,help="gradient accumulation micro-steps (effective batch = batch*accum)")
    ap.add_argument("--compile",action="store_true",help="torch.compile the model (official speedup; fuses the scan loop)")
    ap.add_argument("--load",type=str,default="",help="load a saved checkpoint and SKIP training (gen-only)")
    ap.add_argument("--arch5m",action="store_true",help="5M Arch-A: D256 N96 H8 L6 (5 SSM + 1 SWA@5) + per-block MLP 4D, dt_rank16")
    ap.add_argument("--mlp-mult",type=int,default=4,help="per-block MLP hidden multiplier (arch5m only; 4=literal ~6.7M, 2=~5.2M)")
    ap.add_argument("--top-p",type=float,default=1.0,help="nucleus sampling threshold (1.0=off); single-gen path")
    ap.add_argument("--rep-penalty",type=float,default=1.0,help="CTRL-style repetition penalty (1.0=off); single-gen path")
    ap.add_argument("--rep-window",type=int,default=128,help="how many recent tokens the repetition penalty looks at")
    ap.add_argument("--decode-sweep",action="store_true",help="with --load: regen the SAME seeds across a sampling-control sweep")
    a=ap.parse_args()
    if a.smoke: a.steps=30; a.seq=64; a.batch=4; a.eval_tok=20000
    dev = a.device if a.device!="auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dev_type = "cuda" if dev.startswith("cuda") else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    if dev_type=="cuda": torch.cuda.manual_seed_all(0)
    if a.deterministic:
        try: torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception: pass
    elif dev_type=="cuda":   # fast kernels on Ampere (3060): cuDNN autotune + TF32
        torch.backends.cudnn.benchmark=True
        torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True

    V, exp_len, id2bytes = load_meta(META)
    ids=np.fromfile(IDS,dtype=np.uint16).astype(np.int64)
    n=len(ids); ntr=int(n*0.9)
    train=ids[:ntr]; val=ids[ntr:]                       # probe-style held-out split (last 10%)
    bars=json.load(open(BARS,encoding="utf-8-sig")); WB=bars["token_word_bars"]
    el_t=torch.tensor(exp_len)                           # bytes per token id (CPU; indexed on CPU)

    eff = a.batch*a.accum
    if a.epochs>0: a.steps=max(1,int(a.epochs*ntr/(eff*a.seq)))
    if a.arch5m: AC=dict(D=256,N=96,H=8,L=6,swa_layer=5,use_mlp=True,mlp_mult=a.mlp_mult,dt_rank=16)
    else:        AC=dict(D=192,N=64,H=6,L=4,swa_layer=3,use_mlp=False,mlp_mult=4,dt_rank=12)
    model=ArchA(V,**AC).to(dev); model.use_ckpt = not a.no_ckpt
    nparam=sum(p.numel() for p in model.parameters())
    print(f"Arch-A params={nparam} ({nparam/1e6:.3f}M) | V={V} D={AC['D']} N={AC['N']} H={AC['H']} L={AC['L']} (SWA@{AC['swa_layer']},win128,expand2,mlp={AC['use_mlp']}x{AC['mlp_mult']},dt_rank{AC['dt_rank']}) | torch {torch.__version__} dev={dev} ckpt={model.use_ckpt}")
    print(f"  config: steps={a.steps} seq={a.seq} batch={a.batch} accum={a.accum} (eff {eff}, ~{eff*a.seq} tok/step, ~{a.steps*eff*a.seq/1e6:.1f}M tok = {a.steps*eff*a.seq/ntr:.2f} epochs) lr={a.lr} bf16={a.bf16} compile={a.compile}")
    def val_bpb(cap):
        model.eval(); bits=0.0; nb=0; nt=0
        with torch.no_grad():
            W=a.seq; lim=min(cap,len(val)-1); pos=0
            while pos+W+1<=lim:
                x=torch.from_numpy(val[pos:pos+W][None,:]).to(dev); y=torch.from_numpy(val[pos+1:pos+1+W][None,:]).to(dev)
                ce=F.cross_entropy(model(x).reshape(-1,V),y.reshape(-1),reduction="sum")
                bits+=ce.item()/math.log(2); nb+=int(el_t[y.reshape(-1).cpu()].sum().item()); nt+=W; pos+=W
        model.train(); return bits/max(nb,1), nt, nb
    if a.load:
        sd=torch.load(a.load,map_location=dev); msd=sd.get("model",sd) if isinstance(sd,dict) else sd
        msd={k.replace("_orig_mod.",""):v for k,v in msd.items()}   # strip torch.compile prefix if present
        model.load_state_dict(msd)
        cfgbpb=sd.get("cfg",{}).get("bpb","?") if isinstance(sd,dict) else "?"
        print(f"  loaded checkpoint <- {a.load} (cfg bpb={cfgbpb}); training skipped (decode-only)")
    else:
        opt=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(0.9,0.95),weight_decay=0.1)
        if a.compile:
            try: model=torch.compile(model); print("  torch.compile enabled (first step compiles -> slow; steady-state is the real number)")
            except Exception as e: print(f"  torch.compile failed ({e}); running eager")
        def get_batch(src):
            ix=np.random.randint(0,len(src)-a.seq-1,size=a.batch)
            x=np.stack([src[i:i+a.seq] for i in ix]); y=np.stack([src[i+1:i+1+a.seq] for i in ix])
            return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)

        print("== training (CE only; periodic val BPB = overfit watch: val UP while train DOWN => 25M tokens are the bottleneck) ==")
        print("   (ms/step = instantaneous, val/autotune excluded; step 1 is one-time compile+autotune, read step 2+)")
        model.train()
        for step in range(a.steps):
            ts=time.time()
            opt.zero_grad(); lsum=0.0
            for _ in range(a.accum):
                x,y=get_batch(train)
                if a.bf16:
                    with torch.autocast(dev_type,dtype=torch.bfloat16):
                        logits=model(x); loss=F.cross_entropy(logits.reshape(-1,V),y.reshape(-1))/a.accum
                else:
                    logits=model(x); loss=F.cross_entropy(logits.reshape(-1,V),y.reshape(-1))/a.accum
                loss.backward(); lsum+=loss.item()
            gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            if step==0 or (step+1)%max(1,a.steps//20)==0:
                if dev_type=="cuda": torch.cuda.synchronize()
                step_ms=(time.time()-ts)*1000   # pure single-step time (val/sync overhead excluded)
                vmsg=""
                if (step+1)%max(1,a.steps//10)==0 or step==0:
                    vb,_,_=val_bpb(min(a.eval_tok,16000)); vmsg=f"  | val BPB={vb:.4f}"
                print(f"  step {step+1:5d}/{a.steps}  loss={lsum:.4f}  gnorm={gn:.2f}  bits/tok={lsum/math.log(2):.4f}  ({step_ms:.0f} ms/step){vmsg}")

    # ---- val BPB (bits/byte, unit-invariant) ----
    print("== val BPB (held-out, bits/byte) ==")
    bpb,ntok_eval,nbytes=val_bpb(a.eval_tok)
    print(f"  val BPB={bpb:.4f} bits/byte  (over {ntok_eval} tok / {nbytes} bytes)")

    # ---- save checkpoint (artifact, NOT a commit) ----
    if a.save and not a.load:
        torch.save({"model":model.state_dict(),
                    "cfg":dict(V=V,**AC,win=128,expand=2,conv=4,seq=a.seq,steps=a.steps,bpb=bpb)}, a.save)
        print(f"  saved checkpoint -> {a.save}")

    # ---- closed-loop generation + gate (note: O(gen_tok x seq) recompute; fine on GPU) ----
    if a.gen_samples<=0:
        print("STOP (gen skipped). val BPB above. No commit."); return
    # sampling = raw logits -> repetition penalty -> temperature -> nucleus/top-p -> multinomial.
    # top-p and rep-penalty are SAMPLING controls (they reshape the draw), not post-hoc text filters.
    def generate(seed_ids,nbytes_target,temp,rng,top_p=1.0,rep_pen=1.0,rep_win=128):
        ctx_ids=list(int(t) for t in seed_ids); out=bytearray()
        with torch.no_grad():
            while len(out)<nbytes_target:
                xin=torch.tensor(ctx_ids[-a.seq:],dtype=torch.long,device=dev)[None,:]
                logits=model(xin)[0,-1].float()                    # raw logits (nats)
                if rep_pen!=1.0:                                   # CTRL-style: penalize recently-seen tokens
                    idx=torch.tensor(sorted(set(ctx_ids[-rep_win:])),dtype=torch.long,device=dev)
                    sel=logits[idx]; logits[idx]=torch.where(sel>0,sel/rep_pen,sel*rep_pen)
                logits=logits/temp
                p=torch.softmax(logits,-1)
                if top_p<1.0:                                      # nucleus: keep smallest set with cum-prob >= top_p
                    sp,si_=torch.sort(p,descending=True); cum=torch.cumsum(sp,0)
                    rm=cum>top_p; rm[1:]=rm[:-1].clone(); rm[0]=False
                    sp=sp.masked_fill(rm,0.0); sp=sp/sp.sum()
                    tok=int(si_[torch.multinomial(sp,1,generator=rng)].item())
                else:
                    tok=int(torch.multinomial(p,1,generator=rng).item())
                ctx_ids.append(tok); out+=id2bytes[tok]
        return bytes(out[:nbytes_target])

    def dump(txt,subdir,name,temp,cfgtag):
        bg=byte_guard(txt); wm=word_metrics(txt.decode("latin-1"))
        d=os.path.join(OUT,"human",subdir); os.makedirs(d,exist_ok=True)
        open(os.path.join(d,name),"wb").write(txt)
        wms=f"topBi={wm['topBi']} altLp={wm['altLp']} runWst={wm['runWst']} nameWst={wm['nameWst']}" if wm else "topBi=NA"
        print(f"  [{cfgtag:>16}] {name:<14} | {wms} | wsRun={bg['wsRun']} chRun={bg['chRun']} wsFrac={bg['wsFrac']} nonPrint={bg['nonPrint']}")

    print(f"  bars: wsRun<={bars['wsRun']} chRun<={bars['chRun']} wsFrac<={bars['wsFrac']} nonPrint<={bars['nonPrint']} | topBi<={WB['topBi']} altLp<={WB['altLp']} runWst<={WB['runWst']} nameWst<={WB['nameWst']}")

    if a.decode_sweep:
        # SAME seeds across every config -> controlled A/B: same prompt+rng, only the decode operator changes.
        gpos=np.random.RandomState(2024)
        seed_positions=[gpos.randint(0,len(val)-a.seq-1) for _ in range(a.gen_samples)]
        SWEEP=[("raw",1.0,1.0),("topp90",0.90,1.0),("topp92",0.92,1.0),
               ("rep110",1.0,1.10),("rep120",1.0,1.20),("topp90_rep110",0.90,1.10)]
        print(f"== decode sweep: {len(SWEEP)} configs x 2 temps x {a.gen_samples} shared seeds (rep-window={a.rep_window}) ==")
        for tag,tp,rp in SWEEP:
            for temp in (0.65,0.55):
                rng=torch.Generator(device=dev).manual_seed(1234)   # reset per config: same random stream
                for k,sp_ in enumerate(seed_positions):
                    seed=val[sp_:sp_+16]
                    txt=generate(seed,a.gen_bytes,temp,rng,top_p=tp,rep_pen=rp,rep_win=a.rep_window)
                    dump(txt,tag,f"s{k+1}_T{temp}.txt",temp,tag)
        print(f"  samples -> {os.path.join(OUT,'human')}/<config>/  (raw = grezzo a sola temperatura)")
        print("STOP. decode sweep above; architect reads the samples per config (loops gone, story coherence kept?). No commit.")
        return

    print(f"== closed-loop generation ({2*a.gen_samples} samples, gen-bytes={a.gen_bytes}, top_p={a.top_p} rep_pen={a.rep_penalty}) + gate-v2/byte-guard ==")
    rng=torch.Generator(device=dev).manual_seed(1234); si=0
    for temp in (0.65,0.55):
        for k in range(a.gen_samples):
            seedpos=np.random.randint(0,len(val)-a.seq-1); seed=val[seedpos:seedpos+16]
            txt=generate(seed,a.gen_bytes,temp,rng,top_p=a.top_p,rep_pen=a.rep_penalty,rep_win=a.rep_window); si+=1
            dump(txt,".",f"s{si}_T{temp}.txt",temp,"single")
    print(f"  samples -> {os.path.join(OUT,'human')}")
    print("STOP. raw numbers above; architect reads val BPB + the samples (does it hold the thread?). No commit.")

if __name__=="__main__":
    main()
