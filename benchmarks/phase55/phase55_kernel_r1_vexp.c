// Phase 55 RUNG 1 - vectorized/approx exp in the selective scan (the measured dominant cost).
//   Same Arch-A forward as the premise probe (random weights, forward-only, argmax feedback), but the
//   scan recurrence exp() is done 8-wide with an AVX2 Cephes-style polynomial exp instead of scalar expf().
//   fp32. Goal: big tok/s speedup AT PARITY OF OUTPUT (token stream identical, logits ~1e-6).
//
//   One run reports BOTH: baseline (exact scalar expf) tok/s AND fast (vector exp) tok/s + speedup,
//   plus a parity check (run both on identical weights, compare token streams + max |logit| diff).
//
//   Dominant cost isolated: scan exp = (L-1)*DN*N = 3*384*64 = 73728 exp/token. softplus (1152/tok) and
//   SWA softmax (<=768/tok) left scalar on purpose: rung 1 == "exp in the selective scan" only.
//
// Build: gcc -O3 -march=native -mavx2 -mfma benchmarks/phase55/phase55_kernel_r1_vexp.c -o bin/phase55_kernel_r1.exe -lm
// Run:   bin/phase55_kernel_r1.exe [--tokens N] [--warmup N] [--exp fast|exact|both] [--no-parity]
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

// ---- AVX2 single-precision exp (Cephes-derived). ~1 ulp; accurate enough for output parity. ----
static inline __m256 exp256_ps(__m256 x){
    const __m256 hi = _mm256_set1_ps( 88.3762626647949f);
    const __m256 lo = _mm256_set1_ps(-88.3762626647949f);
    x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);
    __m256 fx = _mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f));
    fx = _mm256_floor_ps(fx);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f),    x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
    __m256 z = _mm256_mul_ps(x, x);
    __m256 p = _mm256_set1_ps(1.9875691500E-4f);
    p = _mm256_fmadd_ps(p, x, _mm256_set1_ps(1.3981999507E-3f));
    p = _mm256_fmadd_ps(p, x, _mm256_set1_ps(8.3334519073E-3f));
    p = _mm256_fmadd_ps(p, x, _mm256_set1_ps(4.1665795894E-2f));
    p = _mm256_fmadd_ps(p, x, _mm256_set1_ps(1.6666665459E-1f));
    p = _mm256_fmadd_ps(p, x, _mm256_set1_ps(5.0000001201E-1f));
    p = _mm256_fmadd_ps(p, z, x);
    p = _mm256_add_ps(p, _mm256_set1_ps(1.0f));
    __m256i e = _mm256_cvttps_epi32(fx);
    e = _mm256_slli_epi32(_mm256_add_epi32(e, _mm256_set1_epi32(0x7f)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}
static inline float hsum256(__m256 v){ float o[8]; _mm256_storeu_ps(o,v); return o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; }

static inline float dotf(const float* a,const float* b,int n){ __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),s);
    float r=hsum256(s); for(;i<n;i++) r+=a[i]*b[i]; return r; }
static inline void matvec(const float* W,const float* x,float* y,int out,int in){ for(int o=0;o<out;o++) y[o]=dotf(W+(size_t)o*in,x,in); }
static inline float silu(float x){ return x/(1.0f+expf(-x)); }
static inline float softplus(float x){ return x>20.0f?x:log1pf(expf(x)); }
static uint64_t rs; static float frand(){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return ((float)((rs>>11)&0xFFFFF)/(float)0x100000-0.5f)*0.1f; }
static float* alloc_rand(size_t n){ float* p=malloc(n*4); for(size_t i=0;i<n;i++) p[i]=frand(); return p; }

typedef struct { float *in_proj,*conv_w,*conv_b,*x_proj,*dt_proj,*dt_b,*A,*Dskip,*out_proj,*norm; } SSML;
typedef struct { float *qkv,*o,*norm; } SWAL;

// ---- weights (built once) ----
static float *emb,*head,*normf; static SSML ssm[L]; static SWAL swa; static int is_swa[L]; static size_t wbytes;
// ---- recurrent state ----
static float (*h)[DN][N]; static float (*convbuf)[DN][CONV]; static float *kring,*vring; static int kvpos,kvcnt;

static void state_reset(void){
    memset(h,0,(size_t)L*DN*N*4); memset(convbuf,0,(size_t)L*DN*CONV*4);
    memset(kring,0,(size_t)WIN*D*4); memset(vring,0,(size_t)WIN*D*4); kvpos=0; kvcnt=0;
}

// forward one token, return argmax. fast_exp selects vector vs scalar exp in the scan. lastlog optional copy of logits.
static uint32_t forward_token(uint32_t tok,int fast_exp,float* lastlog){
    float x[D],xn[D],xz[2*DN],xx[DN],z[DN],dbl[DTR+2*N],dt[DN],y[DN],q[D],kk[D],vv[D],att[WIN],ao[D],tmp[D];
    memcpy(x,emb+(size_t)tok*D,D*4);
    for(int l=0;l<L;l++){
        const float* nw = is_swa[l]?swa.norm:ssm[l].norm;
        float ms=0; for(int i=0;i<D;i++) ms+=x[i]*x[i]; float r=1.0f/sqrtf(ms/D+1e-5f);
        for(int i=0;i<D;i++) xn[i]=x[i]*r*nw[i];
        if(is_swa[l]){
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
            float (*cb)[CONV]=convbuf[l];
            for(int c=0;c<DN;c++){ for(int t=0;t<CONV-1;t++) cb[c][t]=cb[c][t+1]; cb[c][CONV-1]=xx[c];
                float acc=s->conv_b[c]; const float* w=s->conv_w+(size_t)c*CONV; for(int t=0;t<CONV;t++) acc+=w[t]*cb[c][t]; xx[c]=silu(acc); }
            matvec(s->x_proj,xx,dbl,DTR+2*N,DN);
            const float* Bm=dbl+DTR; const float* Cm=dbl+DTR+N;
            for(int c=0;c<DN;c++){ dt[c]=softplus(dotf(s->dt_proj+(size_t)c*DTR,dbl,DTR)+s->dt_b[c]); }
            float (*hl)[N]=h[l];
            if(fast_exp){
                for(int c=0;c<DN;c++){ const float* Ac=s->A+(size_t)c*N; float* hc=hl[c]; float dtc=dt[c], xc=xx[c]; float dbx=dtc*xc;
                    __m256 vdtc=_mm256_set1_ps(dtc), vdbx=_mm256_set1_ps(dbx), vacc=_mm256_setzero_ps();
                    for(int j=0;j<N;j+=8){                                  // N=64 -> 8 vectors, no tail
                        __m256 e=exp256_ps(_mm256_mul_ps(vdtc,_mm256_loadu_ps(Ac+j)));
                        __m256 hcj=_mm256_fmadd_ps(e,_mm256_loadu_ps(hc+j),_mm256_mul_ps(vdbx,_mm256_loadu_ps(Bm+j)));
                        _mm256_storeu_ps(hc+j,hcj);
                        vacc=_mm256_fmadd_ps(hcj,_mm256_loadu_ps(Cm+j),vacc);
                    }
                    y[c]=hsum256(vacc)+s->Dskip[c]*xc; }
            } else {
                for(int c=0;c<DN;c++){ const float* Ac=s->A+(size_t)c*N; float* hc=hl[c]; float dtc=dt[c], xc=xx[c]; float acc=0;
                    for(int j=0;j<N;j++){ hc[j]=expf(dtc*Ac[j])*hc[j]+dtc*Bm[j]*xc; acc+=hc[j]*Cm[j]; }
                    y[c]=acc+s->Dskip[c]*xc; }
            }
            for(int c=0;c<DN;c++) y[c]*=silu(z[c]);
            matvec(s->out_proj,y,tmp,D,DN);
            for(int i=0;i<D;i++) x[i]+=tmp[i];
        }
    }
    float ms=0; for(int i=0;i<D;i++) ms+=x[i]*x[i]; float r=1.0f/sqrtf(ms/D+1e-5f);
    for(int i=0;i<D;i++) xn[i]=x[i]*r*normf[i];
    float best=-1e30f; uint32_t bt=0;
    for(int o=0;o<V;o++){ float lg=dotf(head+(size_t)o*D,xn,D); if(lastlog) lastlog[o]=lg; if(lg>best){best=lg;bt=(uint32_t)o;} }
    return bt;
}

static double timed_run(int fast_exp,long warm,long ntok){
    state_reset(); uint32_t tok=0; clock_t t0=0;
    for(long step=0; step<warm+ntok; step++){ if(step==warm) t0=clock(); tok=forward_token(tok,fast_exp,NULL); }
    double el=(double)(clock()-t0)/CLOCKS_PER_SEC; (void)tok; return ntok/el;
}

int main(int argc,char**argv){
    long ntok=200000, warm=2000; int mode=2 /*both*/, parity=1;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--tokens")&&i+1<argc) ntok=atol(argv[++i]);
        else if(!strcmp(argv[i],"--warmup")&&i+1<argc) warm=atol(argv[++i]);
        else if(!strcmp(argv[i],"--exp")&&i+1<argc){ const char*m=argv[++i]; mode=!strcmp(m,"exact")?0:!strcmp(m,"fast")?1:2; }
        else if(!strcmp(argv[i],"--no-parity")) parity=0;
    }
    rs=0x12345;
    emb=alloc_rand((size_t)V*D); head=alloc_rand((size_t)V*D); normf=alloc_rand(D);
    wbytes=0;
    for(int l=0;l<L;l++){ is_swa[l]=(l==SWA_LAYER);
        if(is_swa[l]){ swa.qkv=alloc_rand((size_t)3*D*D); swa.o=alloc_rand((size_t)D*D); swa.norm=alloc_rand(D);
            wbytes+=((size_t)3*D*D+(size_t)D*D+D)*4; }
        else { SSML*s=&ssm[l];
            s->in_proj=alloc_rand((size_t)2*DN*D); s->conv_w=alloc_rand((size_t)DN*CONV); s->conv_b=alloc_rand(DN);
            s->x_proj=alloc_rand((size_t)(DTR+2*N)*DN); s->dt_proj=alloc_rand((size_t)DN*DTR); s->dt_b=alloc_rand(DN);
            s->A=alloc_rand((size_t)DN*N); for(int i=0;i<DN*N;i++) s->A[i]=-expf(s->A[i]);
            s->Dskip=alloc_rand(DN); s->out_proj=alloc_rand((size_t)D*DN); s->norm=alloc_rand(D);
            wbytes+=((size_t)2*DN*D+(size_t)DN*CONV+DN+(size_t)(DTR+2*N)*DN+(size_t)DN*DTR+DN+(size_t)DN*N+DN+(size_t)D*DN+D)*4; }
    }
    wbytes+=((size_t)V*D+(size_t)V*D+D)*4; size_t param=wbytes/4;
    h=calloc(L,sizeof(*h)); convbuf=calloc(L,sizeof(*convbuf));
    kring=calloc((size_t)WIN*D,4); vring=calloc((size_t)WIN*D,4);
    size_t sbytes=(size_t)L*DN*N*4 + (size_t)L*DN*CONV*4 + (size_t)WIN*D*2*4;

    fprintf(stderr,"Arch-A R1: params=%.3fM weights=%.2fMB state=%.1fKB | scan exp = %d/token | AVX2 (no AVX-512)\n",
            param/1e6,wbytes/1e6,sbytes/1024.0,(L-1)*DN*N);

    printf("\n==== Phase 55 RUNG 1: vectorized/approx exp in selective scan (AVX2, fp32, single-core) ====\n");
    printf("params=%.3fM | weights=%.2f MB | state=%.1f KB | timed tokens=%ld\n",param/1e6,wbytes/1e6,sbytes/1024.0,ntok);

    // ---- parity: same weights, exact vs fast, compare token streams + max |logit| diff ----
    if(parity){
        long P = ntok<4000?ntok:4000;
        float* la=malloc((size_t)V*4); float* lb=malloc((size_t)V*4);
        uint32_t *ta=malloc((size_t)P*4), *tb=malloc((size_t)P*4);
        state_reset(); { uint32_t tk=0; for(long i=0;i<P;i++){ tk=forward_token(tk,0,la); ta[i]=tk; } }
        state_reset(); { uint32_t tk=0; for(long i=0;i<P;i++){ tk=forward_token(tk,1,lb); tb[i]=tk; } }
        long div=0, firstdiv=-1; for(long i=0;i<P;i++){ if(ta[i]!=tb[i]){ if(firstdiv<0)firstdiv=i; div++; } }
        float mxabs=0,mxrel=0; for(int o=0;o<V;o++){ float d=fabsf(la[o]-lb[o]); if(d>mxabs)mxabs=d; float rr=d/(fabsf(la[o])+1e-6f); if(rr>mxrel)mxrel=rr; }
        printf("PARITY (exact vs fast, %ld tok identical weights): token mismatches=%ld/%ld%s | last-step logits max|d|=%.3e max rel=%.3e\n",
               P, div, P, div==0?" (IDENTICAL)":"", mxabs, mxrel);
        if(firstdiv>=0) printf("  first token divergence at step %ld (argmax flipped under approx exp)\n",firstdiv);
        free(la);free(lb);free(ta);free(tb);
    }

    // ---- throughput ----
    double te=0,tf=0;
    if(mode==0||mode==2){ te=timed_run(0,warm,ntok); printf("BASELINE exact expf : %.0f tok/s | %.2f us/tok | ~%.0f bytes/s text\n",te,1e6/te,te*2.05); }
    if(mode==1||mode==2){ tf=timed_run(1,warm,ntok); printf("RUNG1 vector exp    : %.0f tok/s | %.2f us/tok | ~%.0f bytes/s text\n",tf,1e6/tf,tf*2.05); }
    if(mode==2) printf("SPEEDUP: %.2fx  (target ladder: 489 -> 1200-1500 -> beyond)\n",tf/te);
    printf("STOP (rung 1). architect reads: speedup + parity. no commit.\n");
    return 0;
}
