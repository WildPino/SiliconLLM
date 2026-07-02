// Phase 60 / E1 - fp32 REFERENCE inference core. Correctness-first, ZERO optimizations.
//   Model: 5M Arch-A (D256 N96 H8 L6, 5 SSM + SWA@5 win128, expand2 dt_rank16) with a gated-dReLU TERNARY MLP.
//   The MLP weights are loaded DEQUANTIZED (wq*scale) so the forward is bit-identical (mod fp reorder) to PyTorch's
//   inference forward. exp is EXACT (libm expf) - the AVX poly-exp is an E1-forbidden approximation; matvec is plain
//   fp32 SIMD FMA (the natural kernel, not an approximation). Loads results/phase60/e1_model.bin (magic E1M1).
//
//   Gates (each reads a PyTorch reference dump from e1_reference.py):
//     --golden   G1: per-layer residual parity on a fixed 64-tok seq (golden_trace.bin); prints max rel-err profile.
//     --logits   G2: top-1 argmax agreement vs golden_val.bin over nwin*W val tokens.
//     --bpb      G3: engine BPB on the val slice (compare to 0.8799 +/- 0.002).
//     --encode   G4: BPE-encode the corpus prefix, compare token ids to results/phase55/ids.u16.
//     --gen      G5: greedy (rep1.2/win128/no-top-p) continuations vs golden_gen.bin; report divergences.
//     --bench    perf baseline: single-thread tok/s (the "before" number; no optimization in E1).
//
// Build: clang -O3 -mavx2 -mfma -march=znver2 benchmarks/phase60/e1_engine.c -o bin/e1_engine.exe -lm
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../../archive/benchmarks/phase50/bpe_codec.h"

#define V    1024
#define D    256
#define N    96
#define H    8
#define HD   (D/H)
#define L    6
#define DN   512
#define DTR  16
#define CONV 4
#define WIN  128
#define SWA_LAYER 5
#define MLP_HID 1024
#define NLAYER (L+2)               // emb + L blocks + norm_f (matches the reference dump order)

#include <immintrin.h>
static inline float hsum256(__m256 v){ float o[8]; _mm256_storeu_ps(o,v); return o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; }
static inline float dotf(const float*a,const float*b,int n){ __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),s);
    float r=hsum256(s); for(;i<n;i++) r+=a[i]*b[i]; return r; }
static inline void matvec(const float*W,const float*x,float*y,int out,int in){ for(int o=0;o<out;o++) y[o]=dotf(W+(size_t)o*in,x,in); }
static inline float silu(float x){ return x/(1.0f+expf(-x)); }
static inline float softplus(float x){ return x>20.0f?x:log1pf(expf(x)); }
static inline float reluf(float x){ return x>0.0f?x:0.0f; }

typedef struct { float *in_proj,*conv_w,*conv_b,*x_proj,*dt_proj,*dt_b,*A,*Dskip,*out_proj,*norm; } SSML;
typedef struct { float *qkv,*o,*norm; } SWAL;
static float *emb,*head,*normf; static SSML ssm[L]; static SWAL swa; static int is_swa[L];
static float *mlp_n2[L], *mlp_gate[L], *mlp_up[L], *mlp_down[L];    // gate/up/down DEQUANT fp32
static float (*hstate)[DN][N]; static float (*convbuf)[DN][CONV]; static float *kring,*vring; static int kvpos,kvcnt;
static unsigned char* id2bytes[V]; static int id2len[V]; static uint16_t* ids; static long nids;
static int g_mlp_hid=MLP_HID;

static void* xmalloc(size_t n){ void*p=malloc(n); if(!p){fprintf(stderr,"OOM %zu\n",n);exit(1);} return p; }
static float* rd(FILE*f,size_t n){ float*p=xmalloc(n*4); if(fread(p,4,n,f)!=n){fprintf(stderr,"short read %zu\n",n);exit(1);} return p; }

static void load_weights(const char* path){
    FILE*f=fopen(path,"rb"); if(!f){fprintf(stderr,"cannot open %s\n",path);exit(1);}
    uint32_t h[16]; if(fread(h,4,16,f)!=16){fprintf(stderr,"bad header\n");exit(1);}
    if(h[0]!=0x45314D31){fprintf(stderr,"bad magic %08x (need E1M1)\n",h[0]);exit(1);}
    if((int)h[1]!=V||(int)h[2]!=D||(int)h[3]!=N||(int)h[5]!=L||(int)h[7]!=DTR||(int)h[11]!=MLP_HID){
        fprintf(stderr,"dim mismatch V%u D%u N%u L%u dtr%u hid%u vs compiled\n",h[1],h[2],h[3],h[5],h[7],h[11]);exit(1);}
    int gated=h[12],ternary=h[13],has_packed=h[14];
    g_mlp_hid=h[11];
    emb=rd(f,(size_t)V*D);
    for(int l=0;l<L;l++){ is_swa[l]=(l==SWA_LAYER);
        if(is_swa[l]){ swa.norm=rd(f,D); swa.qkv=rd(f,(size_t)3*D*D); swa.o=rd(f,(size_t)D*D); }
        else { SSML*s=&ssm[l]; s->norm=rd(f,D);
            s->in_proj=rd(f,(size_t)2*DN*D); s->conv_w=rd(f,(size_t)DN*CONV); s->conv_b=rd(f,DN);
            s->x_proj=rd(f,(size_t)(DTR+2*N)*DN); s->dt_proj=rd(f,(size_t)DN*DTR); s->dt_b=rd(f,DN);
            s->A=rd(f,(size_t)DN*N); for(int i=0;i<DN*N;i++) s->A[i]=-expf(s->A[i]);       // A = -exp(A_log)
            s->Dskip=rd(f,DN); s->out_proj=rd(f,(size_t)D*DN); }
        mlp_n2[l]=rd(f,D);
        mlp_gate[l]=rd(f,(size_t)g_mlp_hid*D); mlp_up[l]=rd(f,(size_t)g_mlp_hid*D); mlp_down[l]=rd(f,(size_t)D*g_mlp_hid);
    }
    normf=rd(f,D); head=rd(f,(size_t)V*D);
    long core_end=ftell(f); fseek(f,0,SEEK_END); long end=ftell(f);
    // packed ternary section (E2): consume it so no bogus 'trailing bytes'; E1 uses the dequant fp32 above.
    long packed=(long)L*((long)2*g_mlp_hid*D + (long)2*g_mlp_hid*4 + (long)D*g_mlp_hid + (long)D*4);
    fclose(f);
    fprintf(stderr,"weights ok: gated=%d ternary=%d has_packed=%d (core=%ld packed=%ld file=%ld)\n",
            gated,ternary,has_packed,core_end,packed,end);
    if(has_packed && end-core_end!=packed) fprintf(stderr,"WARN packed size %ld != expected %ld\n",end-core_end,packed);
    if(!has_packed && core_end!=end) fprintf(stderr,"WARN %ld trailing bytes\n",end-core_end);
}
static void load_meta(const char* path){
    FILE*f=fopen(path,"rb"); if(!f){fprintf(stderr,"cannot open %s\n",path);exit(1);}
    uint32_t mg,vv,nt; if(fread(&mg,4,1,f)!=1||fread(&vv,4,1,f)!=1||fread(&nt,4,1,f)!=1){exit(1);}
    if(mg!=0x54444D50){fprintf(stderr,"bad meta magic\n");exit(1);}
    unsigned char el[V]; if(fread(el,1,vv,f)!=vv){exit(1);}
    for(uint32_t t=0;t<vv;t++){ id2len[t]=el[t]; id2bytes[t]=xmalloc(el[t]+1);
        if(fread(id2bytes[t],1,el[t],f)!=(size_t)el[t]){exit(1);} id2bytes[t][el[t]]=0; }
    fclose(f);
}
static void load_ids(const char* path){
    FILE*f=fopen(path,"rb"); if(!f){fprintf(stderr,"cannot open %s\n",path);exit(1);}
    fseek(f,0,SEEK_END); long b=ftell(f); fseek(f,0,SEEK_SET); nids=b/2; ids=xmalloc(b);
    if(fread(ids,2,nids,f)!=(size_t)nids){exit(1);} fclose(f);
}
static void state_reset(void){
    memset(hstate,0,(size_t)L*DN*N*4); memset(convbuf,0,(size_t)L*DN*CONV*4);
    memset(kring,0,(size_t)WIN*D*4); memset(vring,0,(size_t)WIN*D*4); kvpos=0; kvcnt=0;
}
static inline void rmsnorm(const float* in,const float* w,float* out){
    float ms=0; for(int i=0;i<D;i++) ms+=in[i]*in[i]; float r=1.0f/sqrtf(ms/D+1e-5f);
    for(int i=0;i<D;i++) out[i]=in[i]*r*w[i];
}
// forward one token (O(1) state). If caps!=NULL, write per-layer residual snapshots: caps[0..D)=emb,
// caps[(l+1)*D..)=x after block l, caps[(L+1)*D..)=xn after norm_f.  logits always written.
static void forward_token(uint32_t tok,float* logits,float* caps){
    float x[D],xn[D],xz[2*DN],xx[DN],z[DN],dbl[DTR+2*N],dt[DN],y[DN],q[D],kk[D],vvv[D],att[WIN],ao[D],tmp[D];
    float gh[MLP_HID],uh[MLP_HID];
    memcpy(x,emb+(size_t)tok*D,D*4);
    if(caps) memcpy(caps,x,D*4);
    for(int l=0;l<L;l++){
        rmsnorm(x, is_swa[l]?swa.norm:ssm[l].norm, xn);
        if(is_swa[l]){
            matvec(swa.qkv,xn,xz,3*D,D); memcpy(q,xz,D*4); memcpy(kk,xz+D,D*4); memcpy(vvv,xz+2*D,D*4);
            int slot=kvpos%WIN; memcpy(kring+(size_t)slot*D,kk,D*4); memcpy(vring+(size_t)slot*D,vvv,D*4);
            kvpos++; if(kvcnt<WIN) kvcnt++;
            memset(ao,0,D*4);
            for(int hh=0;hh<H;hh++){ const float* qh=q+hh*HD; float mx=-1e30f;
                for(int j=0;j<kvcnt;j++){ int s=(kvpos-kvcnt+j)%WIN; float sc=dotf(qh,kring+(size_t)s*D+hh*HD,HD)/sqrtf((float)HD); att[j]=sc; if(sc>mx)mx=sc; }
                float Z=0; for(int j=0;j<kvcnt;j++){ att[j]=expf(att[j]-mx); Z+=att[j]; } float zi=1.0f/Z;
                for(int j=0;j<kvcnt;j++){ int s=(kvpos-kvcnt+j)%WIN; float w=att[j]*zi; const float* vh=vring+(size_t)s*D+hh*HD;
                    for(int d=0;d<HD;d++) ao[hh*HD+d]+=w*vh[d]; } }
            matvec(swa.o,ao,tmp,D,D); for(int i=0;i<D;i++) x[i]+=tmp[i];
        } else {
            SSML*s=&ssm[l];
            matvec(s->in_proj,xn,xz,2*DN,D); memcpy(xx,xz,DN*4); memcpy(z,xz+DN,DN*4);
            float (*cb)[CONV]=convbuf[l];
            for(int c=0;c<DN;c++){ for(int t=0;t<CONV-1;t++) cb[c][t]=cb[c][t+1]; cb[c][CONV-1]=xx[c];
                float acc=s->conv_b[c]; const float* w=s->conv_w+(size_t)c*CONV; for(int t=0;t<CONV;t++) acc+=w[t]*cb[c][t]; xx[c]=silu(acc); }
            matvec(s->x_proj,xx,dbl,DTR+2*N,DN);
            const float* Bm=dbl+DTR; const float* Cm=dbl+DTR+N;
            for(int c=0;c<DN;c++) dt[c]=softplus(dotf(s->dt_proj+(size_t)c*DTR,dbl,DTR)+s->dt_b[c]);
            float (*hl)[N]=hstate[l];
            for(int c=0;c<DN;c++){ const float* Ac=s->A+(size_t)c*N; float* hc=hl[c]; float dtc=dt[c],xc=xx[c],acc=0;
                for(int j=0;j<N;j++){ hc[j]=expf(dtc*Ac[j])*hc[j]+dtc*Bm[j]*xc; acc+=hc[j]*Cm[j]; }
                y[c]=acc+s->Dskip[c]*xc; }
            for(int c=0;c<DN;c++) y[c]*=silu(z[c]);
            matvec(s->out_proj,y,tmp,D,DN); for(int i=0;i<D;i++) x[i]+=tmp[i];
        }
        // gated dReLU ternary MLP:  x += down( relu(gate(norm2 x)) * relu(up(norm2 x)) )
        rmsnorm(x, mlp_n2[l], xn);
        matvec(mlp_gate[l],xn,gh,g_mlp_hid,D); matvec(mlp_up[l],xn,uh,g_mlp_hid,D);
        for(int i=0;i<g_mlp_hid;i++) gh[i]=reluf(gh[i])*reluf(uh[i]);
        matvec(mlp_down[l],gh,tmp,D,g_mlp_hid); for(int i=0;i<D;i++) x[i]+=tmp[i];
        if(caps) memcpy(caps+(size_t)(l+1)*D,x,D*4);
    }
    rmsnorm(x,normf,xn); if(caps) memcpy(caps+(size_t)(L+1)*D,xn,D*4);
    matvec(head,xn,logits,V,D);
}

static const char* MP="results/phase60/e1_model.bin";
static void bootstrap(const char* wpath){
    load_weights(wpath); load_meta("results/phase55/meta.bin"); load_ids("results/phase55/ids.u16");
    hstate=calloc(L,sizeof(*hstate)); convbuf=calloc(L,sizeof(*convbuf)); kring=calloc((size_t)WIN*D,4); vring=calloc((size_t)WIN*D,4);
}

// ---------------- G1 golden-trace ----------------
static int gate_golden(void){
    FILE*f=fopen("results/phase60/golden_trace.bin","rb"); if(!f){fprintf(stderr,"no golden_trace.bin\n");return 1;}
    uint32_t hd[4]; if(fread(hd,4,4,f)!=4||hd[0]!=0x47543031){fprintf(stderr,"bad GT01\n");return 1;}
    int T=hd[1],Dd=hd[2],nl=hd[3]; if(Dd!=D||nl!=NLAYER){fprintf(stderr,"golden dim mismatch T%d D%d nl%d\n",T,Dd,nl);return 1;}
    uint16_t* tin=xmalloc((size_t)T*2); if(fread(tin,2,T,f)!=(size_t)T){return 1;}
    float* ref=xmalloc((size_t)nl*T*D*4); if(fread(ref,4,(size_t)nl*T*D,f)!=(size_t)nl*T*D){return 1;}
    uint32_t vv; if(fread(&vv,4,1,f)!=1||(int)vv!=V){fprintf(stderr,"bad V in golden\n");return 1;}
    float* reflog=xmalloc((size_t)T*V*4); if(fread(reflog,4,(size_t)T*V,f)!=(size_t)T*V){return 1;} fclose(f);
    float* caps=xmalloc((size_t)nl*D*4); float* logits=xmalloc((size_t)V*4);
    // per-layer: max abs err, ||c-p||^2, ||p||^2  -> l2-relative error (scale-aware; the honest layer-match metric).
    double maxabs[NLAYER+1], sqerr[NLAYER+1], sqref[NLAYER+1];
    for(int i=0;i<=nl;i++){ maxabs[i]=sqerr[i]=sqref[i]=0; }
    state_reset();
    for(int t=0;t<T;t++){ forward_token(tin[t],logits,caps);
        for(int lyr=0;lyr<nl;lyr++){ const float* c=caps+(size_t)lyr*D; const float* p=ref+((size_t)lyr*T+t)*D;
            for(int d=0;d<D;d++){ double e=fabs((double)c[d]-p[d]); if(e>maxabs[lyr]) maxabs[lyr]=e;
                sqerr[lyr]+=e*e; sqref[lyr]+=(double)p[d]*p[d]; } }
        const float* pl=reflog+(size_t)t*V;
        for(int o=0;o<V;o++){ double e=fabs((double)logits[o]-pl[o]); if(e>maxabs[nl]) maxabs[nl]=e;
            sqerr[nl]+=e*e; sqref[nl]+=(double)pl[o]*pl[o]; }
    }
    const char* nm[NLAYER+1]={"emb","blk0","blk1","blk2","blk3","blk4","blk5","norm_f","logits"};
    printf("==== G1 golden-trace (fixed %d-tok seq, per-layer error vs PyTorch fp32) ====\n",T);
    printf("  layer     max_abs      l2_rel_err   %s\n","(l2_rel = ||c-p||/||p||; threshold 1e-3)");
    int pass=1;
    for(int i=0;i<=nl;i++){ double l2=sqrt(sqerr[i]/(sqref[i]+1e-30));
        printf("  %-7s  %.3e    %.3e   %s\n",nm[i],maxabs[i],l2,l2<1e-3?"":"<-- EXCEEDS 1e-3");
        if(l2>=1e-3) pass=0; }
    printf("  G1 %s (l2-relative error < 1e-3 per layer)\n",pass?"PASS":"FAIL");
    return pass?0:2;
}

// ---------------- G2 logit top-1 parity ----------------
static int gate_logits(void){
    FILE*f=fopen("results/phase60/golden_val.bin","rb"); if(!f){fprintf(stderr,"no golden_val.bin\n");return 1;}
    uint32_t hd[3]; if(fread(hd,4,3,f)!=3||hd[0]!=0x47563031){fprintf(stderr,"bad GV01\n");return 1;}
    int W=hd[1],nwin=hd[2];
    uint16_t* ref=xmalloc((size_t)nwin*W*2); if(fread(ref,2,(size_t)nwin*W,f)!=(size_t)nwin*W){return 1;} fclose(f);
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr;
    float* logits=xmalloc((size_t)V*4); long agree=0,tot=0;
    for(int w=0;w<nwin;w++){ state_reset();
        for(int t=0;t<W;t++){ forward_token(val[(long)w*W+t],logits,NULL);
            float mx=-1e30f; int am=0; for(int o=0;o<V;o++) if(logits[o]>mx){mx=logits[o];am=o;}
            if(am==ref[(long)w*W+t]) agree++; tot++; }
    }
    double pct=100.0*agree/tot;
    printf("==== G2 logit top-1 parity (%ld val tokens, %d windows x %d) ====\n",tot,nwin,W);
    printf("  top-1 agreement=%.4f%% (%ld/%ld)  G2 %s (threshold 99.9%%)\n",pct,agree,tot,pct>=99.9?"PASS":"FAIL");
    return pct>=99.9?0:2;
}

// ---------------- G3 BPB parity ----------------
static int gate_bpb(long seqW,long eval_tok){
    long ntr=(long)(nids*0.9); long nval=nids-ntr; uint16_t* val=ids+ntr;
    long lim=eval_tok<(nval-1)?eval_tok:(nval-1); double bits=0; long nbytes=0,ntok=0,pos=0;
    float* logits=xmalloc((size_t)V*4); const double LN2=0.6931471805599453;
    while(pos+seqW+1<=lim){ state_reset();
        for(long t=0;t<seqW;t++){ forward_token(val[pos+t],logits,NULL); int tgt=val[pos+t+1];
            double mx=-1e30; for(int o=0;o<V;o++) if(logits[o]>mx) mx=logits[o];
            double se=0; for(int o=0;o<V;o++) se+=exp((double)logits[o]-mx);
            bits += (-((double)logits[tgt]-mx)+log(se))/LN2; nbytes+=id2len[tgt]; ntok++; }
        pos+=seqW; }
    double bpb=bits/(nbytes>0?nbytes:1);
    printf("==== G3 BPB parity (seq=%ld eval-tok=%ld) ====\n",seqW,eval_tok);
    printf("  engine BPB=%.6f (over %ld tok / %ld bytes)  ref=0.879949  delta=%+.6f  G3 %s (|delta|<=0.002)\n",
           bpb,ntok,nbytes,bpb-0.879949,fabs(bpb-0.879949)<=0.002?"PASS":"FAIL");
    return fabs(bpb-0.879949)<=0.002?0:2;
}

// ---------------- G4 tokenizer parity ----------------
static int gate_encode(long maxb){
    Bpe B; if(!bpe_load_file(&B,"weights/bpe1024.bin")) return 1;
    FILE*fd=fopen("data/corpora/tinystories_64mb.txt","rb"); if(!fd){fprintf(stderr,"no corpus\n");return 1;}
    fseek(fd,0,SEEK_END); long fsz=ftell(fd); fseek(fd,0,SEEK_SET); if(maxb>0&&maxb<fsz) fsz=maxb;
    unsigned char* buf=xmalloc(fsz); if(fread(buf,1,fsz,fd)!=(size_t)fsz){return 1;} fclose(fd);
    uint32_t* out=xmalloc((size_t)fsz*sizeof(uint32_t));
    size_t m=bpe_encode_region(&B,buf,0,(size_t)fsz,out);
    // compare to ids.u16 prefix; drop last 2 tokens (byte-prefix cut can change the final chunk)
    long cmp=(long)m-2; if(cmp>nids) cmp=nids; long mism=0,first=-1;
    for(long i=0;i<cmp;i++){ if((uint32_t)ids[i]!=out[i]){ mism++; if(first<0) first=i; } }
    printf("==== G4 tokenizer parity (BPE-1024, first %ld corpus bytes -> %zu tok) ====\n",fsz,m);
    printf("  compared %ld tok vs ids.u16: mismatches=%ld%s  G4 %s\n",cmp,mism,
           first>=0?"":" (first: n/a)",mism==0?"PASS":"FAIL");
    if(mism) printf("  first mismatch @tok %ld: engine=%u ref=%u\n",first,out[first],ids[first]);
    return mism==0?0:2;
}

// ---------------- G5 generation parity ----------------
static int gate_gen(long seqW){
    FILE*f=fopen("results/phase60/golden_gen.bin","rb"); if(!f){fprintf(stderr,"no golden_gen.bin\n");return 1;}
    uint32_t hd[4]; if(fread(hd,4,4,f)!=4||hd[0]!=0x47473031){fprintf(stderr,"bad GG01\n");return 1;}
    int nseed=hd[1],slen=hd[2],glen=hd[3];
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr; (void)val;
    float* logits=xmalloc((size_t)V*4); float rep=1.2f; int repwin=128;
    uint16_t* seed=xmalloc((size_t)slen*2); uint16_t* refc=xmalloc((size_t)glen*2);
    uint16_t* gen=xmalloc((size_t)(slen+glen)*2);
    int total_div=0; long total=0;
    printf("==== G5 generation parity (%d seeds, greedy rep%.1f/win%d, %d gen tok) ====\n",nseed,rep,repwin,glen);
    for(int si=0;si<nseed;si++){
        if(fread(seed,2,slen,f)!=(size_t)slen||fread(refc,2,glen,f)!=(size_t)glen){return 1;}
        state_reset(); int gl=0;
        for(int i=0;i<slen;i++){ forward_token(seed[i],logits,NULL); gen[gl++]=seed[i]; }
        int div=0,firstdiv=-1;
        for(int t=0;t<glen;t++){
            long lo=gl>repwin?gl-repwin:0; static int seen[V]; static int stamp=0; stamp++;
            for(long p=lo;p<gl;p++){ int tt=gen[p]; if(seen[tt]!=stamp){ seen[tt]=stamp; float v=logits[tt]; logits[tt]= v>0? v/rep : v*rep; } }
            float mx=-1e30f; int tok=0; for(int o=0;o<V;o++) if(logits[o]>mx){mx=logits[o];tok=o;}
            if((uint16_t)tok!=refc[t]){ div++; if(firstdiv<0) firstdiv=t; }
            gen[gl++]=(uint16_t)tok; forward_token((uint16_t)tok,logits,NULL);
        }
        total_div+=div; total+=glen;
        printf("  seed %d: divergences=%d/%d%s\n",si+1,div,glen,firstdiv>=0?"":" (identical)");
        if(div) printf("           first div @gen %d: engine=%u ref=%u\n",firstdiv,gen[slen+firstdiv],refc[firstdiv]);
    }
    printf("  G5 total divergences=%d/%ld  %s (0=identical; any divergence -> Architect judges fp-reorder)\n",
           total_div,total,total_div==0?"PASS":"REVIEW");
    return 0;
}

// ---------------- perf baseline ----------------
static void bench(long bench_tok){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr;
    float* logits=xmalloc((size_t)V*4);
    state_reset(); uint32_t tk=val[0]; long warm=bench_tok/10; clock_t t0=0;
    for(long s=0;s<warm+bench_tok;s++){ if(s==warm) t0=clock(); forward_token(tk,logits,NULL);
        float mx=-1e30f; uint32_t bt=0; for(int o=0;o<V;o++) if(logits[o]>mx){mx=logits[o];bt=o;} tk=bt; }
    double el=(double)(clock()-t0)/CLOCKS_PER_SEC, tps=bench_tok/el;
    printf("==== perf baseline (single-thread fp32, exact exp, 3600X) ====\n");
    printf("  %.1f tok/s | %.2f us/tok  (the E1 'before' number; ZERO optimizations)\n",tps,1e6/tps);
}

int main(int argc,char**argv){
    int gG=0,gL=0,gB=0,gE=0,gGen=0,gAll=0,doBench=0; long seqW=512,eval_tok=200000,benchN=20000,encb=2000000;
    const char* wpath=MP;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--golden")) gG=1; else if(!strcmp(argv[i],"--logits")) gL=1;
        else if(!strcmp(argv[i],"--bpb")) gB=1; else if(!strcmp(argv[i],"--encode")) gE=1;
        else if(!strcmp(argv[i],"--gen")) gGen=1; else if(!strcmp(argv[i],"--all")) gAll=1;
        else if(!strcmp(argv[i],"--bench")) doBench=1;
        else if(!strcmp(argv[i],"--seq")&&i+1<argc) seqW=atol(argv[++i]);
        else if(!strcmp(argv[i],"--eval-tok")&&i+1<argc) eval_tok=atol(argv[++i]);
        else if(!strcmp(argv[i],"--bench-tok")&&i+1<argc) benchN=atol(argv[++i]);
        else if(!strcmp(argv[i],"--enc-bytes")&&i+1<argc) encb=atol(argv[++i]);
        else if(!strcmp(argv[i],"--weights")&&i+1<argc) wpath=argv[++i];
    }
    if(!gG&&!gL&&!gB&&!gE&&!gGen&&!doBench&&!gAll) gAll=1;
    bootstrap(wpath);
    fprintf(stderr,"E1 core loaded: ids=%ld D%d N%d L%d hid%d\n",nids,D,N,L,g_mlp_hid);
    int rc=0;
    if(gAll||gG)   rc|=gate_golden();
    if(gAll||gL)   rc|=gate_logits();
    if(gAll||gB)   rc|=gate_bpb(seqW,eval_tok);
    if(gAll||gE)   rc|=gate_encode(encb);
    if(gAll||gGen) rc|=gate_gen(seqW);
    if(gAll||doBench) bench(benchN);
    printf("STOP. E1 gates above. No commit.\n");
    return rc;
}
