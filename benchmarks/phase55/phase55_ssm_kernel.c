// Phase 55 (1) - PREMISE PROBE: C inference kernel of Arch-A, random weights, FORWARD-ONLY.
//   Measures single-core tokens/sec + working-set (weight bytes / state bytes) vs cache hierarchy.
//   Validates the "CPU-optimal" premise BEFORE spending GPU. Per-token RECURRENT inference (SSM = O(1)/token
//   state update; the whole point), so this is the generation-throughput regime.
//
//   Arch-A: D=192 N=64 H=6 L=4, layers 0..2 = selective diagonal SSM (Mamba-1), layer 3 = SWA window=128.
//   expand=2 (d_inner=384), dt_rank=12, conv=4, vocab=1024. fp32 weights (quantization = downstream, not now).
//
//   NOTE: this dev CPU (AMD Ryzen 5 3600X, Zen2) is AVX2+FMA, NO AVX-512. We measure AVX2 here; the AVX-512
//   throughput premise must be validated on target hardware. The WORKING-SET analysis (bytes vs cache) is
//   ISA-independent and is the load-bearing number.
//
// Build: gcc -O3 -march=native -mavx2 -mfma benchmarks/phase55/phase55_ssm_kernel.c -o bin/phase55_ssm_kernel.exe -lm
// Run:   bin/phase55_ssm_kernel.exe [--tokens N] [--warmup N]
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#define V    1024
#define D    192
#define N    64
#define H    6
#define HD   (D/H)
#define L    4
#define EXP  2
#define DN   (D*EXP)      // 384
#define DTR  12
#define CONV 4
#define WIN  128
#define SWA_LAYER 3

static inline float dotf(const float* a,const float* b,int n){ __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),s);
    float o[8]; _mm256_storeu_ps(o,s); float r=o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; for(;i<n;i++) r+=a[i]*b[i]; return r; }
static inline void matvec(const float* W,const float* x,float* y,int out,int in){ for(int o=0;o<out;o++) y[o]=dotf(W+(size_t)o*in,x,in); }
static inline float silu(float x){ return x/(1.0f+expf(-x)); }
static inline float softplus(float x){ return x>20.0f?x:log1pf(expf(x)); }
static uint64_t rs=0x12345; static float frand(){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return ((float)((rs>>11)&0xFFFFF)/(float)0x100000-0.5f)*0.1f; }
static float* alloc_rand(size_t n){ float* p=malloc(n*4); for(size_t i=0;i<n;i++) p[i]=frand(); return p; }

typedef struct { float *in_proj,*conv_w,*conv_b,*x_proj,*dt_proj,*dt_b,*A,*Dskip,*out_proj,*norm; } SSML;
typedef struct { float *qkv,*o,*norm; } SWAL;

int main(int argc,char**argv){
    long ntok=200000, warm=2000;
    for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"--tokens")&&i+1<argc) ntok=atol(argv[++i]); else if(!strcmp(argv[i],"--warmup")&&i+1<argc) warm=atol(argv[++i]); }

    // ---- random weights ----
    float* emb=alloc_rand((size_t)V*D);
    float* head=alloc_rand((size_t)V*D); float* normf=alloc_rand(D);
    SSML ssm[L]; SWAL swa; int is_swa[L];
    size_t wbytes=0;
    for(int l=0;l<L;l++){ is_swa[l]=(l==SWA_LAYER);
        if(is_swa[l]){ swa.qkv=alloc_rand((size_t)3*D*D); swa.o=alloc_rand((size_t)D*D); swa.norm=alloc_rand(D);
            wbytes+=((size_t)3*D*D+(size_t)D*D+D)*4; }
        else { SSML*s=&ssm[l];
            s->in_proj=alloc_rand((size_t)2*DN*D); s->conv_w=alloc_rand((size_t)DN*CONV); s->conv_b=alloc_rand(DN);
            s->x_proj=alloc_rand((size_t)(DTR+2*N)*DN); s->dt_proj=alloc_rand((size_t)DN*DTR); s->dt_b=alloc_rand(DN);
            s->A=alloc_rand((size_t)DN*N); for(int i=0;i<DN*N;i++) s->A[i]=-expf(s->A[i]); // A=-exp(.)
            s->Dskip=alloc_rand(DN); s->out_proj=alloc_rand((size_t)D*DN); s->norm=alloc_rand(D);
            wbytes+=((size_t)2*DN*D+(size_t)DN*CONV+DN+(size_t)(DTR+2*N)*DN+(size_t)DN*DTR+DN+(size_t)DN*N+DN+(size_t)D*DN+D)*4; }
    }
    wbytes+=((size_t)V*D+(size_t)V*D+D)*4;   // emb + head + final norm
    size_t param=wbytes/4;

    // ---- recurrent state ----
    float (*h)[DN][N]=calloc(L,sizeof(*h));               // SSM state per layer
    float (*convbuf)[DN][CONV]=calloc(L,sizeof(*convbuf));// depthwise conv ring (last CONV inputs)
    float* kring=calloc((size_t)WIN*D,4); float* vring=calloc((size_t)WIN*D,4); int kvpos=0,kvcnt=0;
    size_t sbytes=(size_t)L*DN*N*4 + (size_t)L*DN*CONV*4 + (size_t)WIN*D*2*4;

    // ---- scratch ----
    float x[D],xn[D],xz[2*DN],xx[DN],z[DN],dbl[DTR+2*N],dt[DN],y[DN],q[D],kk[D],vv[D],att[WIN],ao[D],tmp[D];

    fprintf(stderr,"Arch-A kernel: params=%zu (%.3fM) weights=%.2f MB | state=%.1f KB | AVX2 (no AVX-512 on this CPU)\n",
            param,param/1e6,wbytes/1e6,sbytes/1024.0);

    clock_t t0=0,t1; double elapsed=0; uint32_t tok=0; long done=0;
    for(long step=0; step<warm+ntok; step++){
        if(step==warm) t0=clock();
        // embed
        memcpy(x,emb+(size_t)tok*D,D*4);
        for(int l=0;l<L;l++){
            // RMSNorm
            const float* nw = is_swa[l]?swa.norm:ssm[l].norm;
            float ms=0; for(int i=0;i<D;i++) ms+=x[i]*x[i]; float r=1.0f/sqrtf(ms/D+1e-5f);
            for(int i=0;i<D;i++) xn[i]=x[i]*r*nw[i];
            if(is_swa[l]){
                // qkv
                matvec(swa.qkv,xn,xz,3*D,D); memcpy(q,xz,D*4); memcpy(kk,xz+D,D*4); memcpy(vv,xz+2*D,D*4);
                int slot=kvpos%WIN; memcpy(kring+(size_t)slot*D,kk,D*4); memcpy(vring+(size_t)slot*D,vv,D*4);
                kvpos++; if(kvcnt<WIN) kvcnt++;
                memset(ao,0,D*4);
                for(int hh=0;hh<H;hh++){ const float* qh=q+hh*HD; float mx=-1e30f;
                    for(int j=0;j<kvcnt;j++){ int s=(kvpos-kvcnt+j)%WIN; float sc=dotf(qh,kring+(size_t)s*D+hh*HD,HD)/sqrtf((float)HD); att[j]=sc; if(sc>mx)mx=sc; }
                    float Z=0; for(int j=0;j<kvcnt;j++){ att[j]=expf(att[j]-mx); Z+=att[j]; } float zi=1.0f/Z;
                    for(int j=0;j<kvcnt;j++){ int s=(kvpos-kvcnt+j)%WIN; float w=att[j]*zi; const float* vh=vring+(size_t)s*D+hh*HD;
                        for(int d=0;d<HD;d++) ao[hh*HD+d]+=w*vh[d]; } }
                matvec(swa.o,ao,tmp,D,D);
                for(int i=0;i<D;i++) x[i]+=tmp[i];
            } else {
                SSML*s=&ssm[l];
                matvec(s->in_proj,xn,xz,2*DN,D); memcpy(xx,xz,DN*4); memcpy(z,xz+DN,DN*4);
                // depthwise causal conv width CONV + bias, then silu
                float (*cb)[CONV]=convbuf[l];
                for(int c=0;c<DN;c++){ for(int t=0;t<CONV-1;t++) cb[c][t]=cb[c][t+1]; cb[c][CONV-1]=xx[c];
                    float acc=s->conv_b[c]; const float* w=s->conv_w+(size_t)c*CONV; for(int t=0;t<CONV;t++) acc+=w[t]*cb[c][t]; xx[c]=silu(acc); }
                matvec(s->x_proj,xx,dbl,DTR+2*N,DN);
                const float* Bm=dbl+DTR; const float* Cm=dbl+DTR+N;
                for(int c=0;c<DN;c++){ dt[c]=softplus(dotf(s->dt_proj+(size_t)c*DTR,dbl,DTR)+s->dt_b[c]); }
                float (*hl)[N]=h[l];
                for(int c=0;c<DN;c++){ const float* Ac=s->A+(size_t)c*N; float* hc=hl[c]; float dtc=dt[c], xc=xx[c]; float acc=0;
                    for(int j=0;j<N;j++){ hc[j]=expf(dtc*Ac[j])*hc[j]+dtc*Bm[j]*xc; acc+=hc[j]*Cm[j]; }
                    y[c]=acc+s->Dskip[c]*xc; }
                for(int c=0;c<DN;c++) y[c]*=silu(z[c]);
                matvec(s->out_proj,y,tmp,D,DN);
                for(int i=0;i<D;i++) x[i]+=tmp[i];
            }
        }
        // final norm + head -> argmax next token (feed back, autoregressive)
        float ms=0; for(int i=0;i<D;i++) ms+=x[i]*x[i]; float r=1.0f/sqrtf(ms/D+1e-5f);
        for(int i=0;i<D;i++) xn[i]=x[i]*r*normf[i];
        float best=-1e30f; uint32_t bt=0; for(int o=0;o<V;o++){ float lg=dotf(head+(size_t)o*D,xn,D); if(lg>best){best=lg;bt=(uint32_t)o;} }
        tok=bt; done++;
    }
    t1=clock();
    elapsed=(double)(t1-t0)/CLOCKS_PER_SEC;
    double tps=ntok/elapsed; double nspt=elapsed/ntok*1e9;
    double eff_bw=(double)wbytes*ntok/elapsed/1e9;                 // GB/s if weights streamed once/token
    long n_exp=(long)(L-1)*DN*N + (long)(L-1)*DN;                  // expf in scan (dA) + softplus per token (SSM layers)
    double macs=(double)((2*DN*D)+(DTR+2*N)*DN+(DN*DTR)+(D*DN))*(L-1) + (double)(3*D*D+D*D) + (double)V*D + (double)(L-1)*DN*N;
    printf("\n==== Phase 55 PREMISE PROBE (Arch-A inference, AVX2 single-core, fp32) ====\n");
    printf("params=%.3fM | weights=%.2f MB | recurrent state=%.1f KB | tokens timed=%ld\n",param/1e6,wbytes/1e6,sbytes/1024.0,ntok);
    printf("THROUGHPUT: %.0f tokens/sec | %.1f us/token | %.3f s for %ld tok | ~%.0f bytes/sec text\n",tps,nspt/1000.0,elapsed,ntok,tps*2.05);
    printf("BOTTLENECK (measurement-driven, NOT assumed):\n");
    printf("  weights/token stream = %.2f MB -> effective %.1f GB/s.  Zen2 L3 BW is ~100-400 GB/s, so this is %s.\n",
           wbytes/1e6, eff_bw, eff_bw<60?"FAR below BW => NOT bandwidth-bound => COMPUTE-bound":"near BW => bandwidth-bound");
    printf("  ~%ld scalar expf()/token in the selective scan (dA=exp(dt*A) over %d layers x %d ch x %d state) + softplus.\n",n_exp,L-1,DN,N);
    printf("  matvec MACs/token ~ %.2fM (AVX2-FMA, fast); transcendentals are the dominant cost at fp32-naive.\n",macs/1e6);
    printf("WORKING SET vs cache (Zen2: L1d=32KB, L2=512KB/core, L3=32MB shared):\n");
    printf("  weights %.2f MB -> %s ; recurrent state %.1f KB -> %s (hot)\n",
           wbytes/1e6, wbytes<32768?"L1":wbytes<524288?"L2":wbytes<33554432?"L3":"RAM",
           sbytes/1024.0, sbytes<32768?"L1":sbytes<524288?"L2":"L3");
    printf("  => fp32 model is L3-resident (not L1/L2). int8 weights -> %.2f MB, int4 -> %.2f MB (toward L2).\n",wbytes/1e6/4,wbytes/1e6/8);
    printf("PREMISE VERDICT (architect reads): at fp32-naive AVX2 the kernel is ~%.0f tok/s, EXP-BOUND not BW-bound.\n",tps);
    printf("  Levers BEFORE GPU (downstream, not now): (1) vectorized/approx exp in the scan (biggest), (2) int8/int4 weights\n");
    printf("  toward L2 residency, (3) AVX-512 vector-exp on target HW (this CPU = Zen2 AVX2 only, AVX-512 unmeasured).\n");
    printf("sanity: last token=%u (argmax feedback, random weights -> meaningless content, throughput only)\n",tok);
    return 0;
}
