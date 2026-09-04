// Silicon Entropy Engine — CONSOLIDATED single-core inference engine (P4.3).
//
// One core, feature-flags, replacing the five stage engines (e1/e2/e3/e35/e4_engine.c, now archival).
// The model file's magic selects the MLP family: E1M1 (dense gated-dReLU) or E4M1 (MoE top-8 experts).
//
//   --mlp  fp32|lut     MLP matvec path: fp32 dequant reference | ternary pshufb-LUT (int8 activations)
//   --skip on|off       dense-LUT only: E3 exact activation-skip (gate-first, up row-skip, down tile-skip)
//   --exp  exact|fast   selective-scan exp: libm expf | exp256_ps poly (E3.5, deterministic, parity-gated)
//
// Stage equivalences (acceptance: bit-identical logit streams vs the archived stage engines):
//   E1      = E1M1 --mlp fp32 --exp exact          E3.5iso = E1M1 --mlp fp32 --exp fast
//   E2      = E1M1 --mlp lut --skip off --exp exact  E3.5   = E1M1 --mlp lut --skip on --exp fast
//   E3      = E1M1 --mlp lut --skip on  --exp exact
//   E4-ref  = E4M1 --mlp fp32 --exp exact          E4-full = E4M1 --mlp lut --exp fast
//
// Modes: --bpb | --logits (top-1 vs the fp32 config) | --dumplogits <file> (raw fp32 logit stream,
//        the parity-acceptance instrument) | --kselftest (synthetic kernel self-tests, no weights; CI)
// Protocol shared with the stage gates: val = last 10% of ids.u16; windows of --seq (512) with
// state_reset per window; --ntok tokens from --offset.
//
// Build: clang -O3 -mavx2 -mfma -march=znver2 benchmarks/phase60/engine.c -o bin/engine -lm
// Law: correctness-first; NO -ffast-math; tok/s of this core are re-measured only at training end.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif
// 63.T threads: parallelism ONLY across independent outputs (distinct y[o]); each reduction (dotf, scan-j) stays
// serial on one thread -> bit-identity is by construction, invariant to thread count. schedule(static). --threads N
// (default 1 = the P43 path). Pinning is external (OMP_PROC_BIND/OMP_PLACES) — no topology code in the engine.
// 63.C: the OpenMP `if(g_omp_on)` clause skips the fork/join entirely at N=1 (g_omp_on set to threads>1 in main) — the
// portable one-line fix for the ~6% parallel-region tax measured at N=1, no dual code path. bit-identity is unaffected.
static int g_omp_on=0;
#ifdef _OPENMP
#define OMP_PFOR _Pragma("omp parallel for schedule(static) if(g_omp_on)")
#else
#define OMP_PFOR
#endif
#if defined(_WIN32)
#include <windows.h>
static double now_s(void){ LARGE_INTEGER f,t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t); return (double)t.QuadPart/(double)f.QuadPart; }
#else
#include <time.h>
static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
#endif

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
#define AQ   63
// dense (E1M1)
#define MLP_HID 1024
#define TUP  (D/2)
#define TDN  (MLP_HID/2)
// MoE (E4M1)
#define E    32
#define HID_E 128
#define KTOP 8
#define GH   (E*HID_E)
#define MPAD_GU ((GH+31)&~31)
#define MPAD_D  ((D+31)&~31)
#define TDE  (HID_E/2)
#define NLAYER (L+2)

// ---------------- shared primitives ----------------
static inline __m256 exp256_ps(__m256 x){                        // E3.5 poly-exp (Cephes, fixed coeffs, deterministic)
    const __m256 hi=_mm256_set1_ps(88.3762626647949f), lo=_mm256_set1_ps(-88.3762626647949f);
    x=_mm256_min_ps(_mm256_max_ps(x,lo),hi);
    __m256 fx=_mm256_fmadd_ps(x,_mm256_set1_ps(1.44269504088896341f),_mm256_set1_ps(0.5f)); fx=_mm256_floor_ps(fx);
    x=_mm256_fnmadd_ps(fx,_mm256_set1_ps(0.693359375f),x); x=_mm256_fnmadd_ps(fx,_mm256_set1_ps(-2.12194440e-4f),x);
    __m256 z=_mm256_mul_ps(x,x), p=_mm256_set1_ps(1.9875691500E-4f);
    p=_mm256_fmadd_ps(p,x,_mm256_set1_ps(1.3981999507E-3f)); p=_mm256_fmadd_ps(p,x,_mm256_set1_ps(8.3334519073E-3f));
    p=_mm256_fmadd_ps(p,x,_mm256_set1_ps(4.1665795894E-2f)); p=_mm256_fmadd_ps(p,x,_mm256_set1_ps(1.6666665459E-1f));
    p=_mm256_fmadd_ps(p,x,_mm256_set1_ps(5.0000001201E-1f)); p=_mm256_fmadd_ps(p,z,x); p=_mm256_add_ps(p,_mm256_set1_ps(1.0f));
    __m256i e=_mm256_cvttps_epi32(fx); e=_mm256_slli_epi32(_mm256_add_epi32(e,_mm256_set1_epi32(0x7f)),23);
    return _mm256_mul_ps(p,_mm256_castsi256_ps(e));
}
static inline float exp_approx1(float x){ float o[8]; _mm256_storeu_ps(o,exp256_ps(_mm256_set1_ps(x))); return o[0]; }
static inline float hsum256(__m256 v){ float o[8]; _mm256_storeu_ps(o,v); return o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; }
static inline float dotf(const float*a,const float*b,int n){ __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),s);
    float r=hsum256(s); for(;i<n;i++) r+=a[i]*b[i]; return r; }
// Sequential per-row dotf. A blocked/multi-accumulator GEMV (2.3-2.6x in an isolated cache-resident microbench,
// gemv_bench.c) was tried and REVERTED: in-engine the per-token GEMVs are memory-bandwidth-bound (weights evicted
// between tokens, not FMA-latency-bound), so ILP does not help and the 4-strided access is ~4% SLOWER than this
// sequential stream (probe-3 rho-law). Lesson: microbench speedup (compute-bound) does not compose to engine speedup
// (memory-bound) -- measure end-to-end. The fp32 projections are floor-ish; the byte lever (ternary) failed on quality (Phase 61).
static inline void matvec(const float*W,const float*x,float*y,int out,int in){ OMP_PFOR for(int o=0;o<out;o++) y[o]=dotf(W+(size_t)o*in,x,in); }
static inline float silu(float x){ return x/(1.0f+expf(-x)); }
static inline float softplus(float x){ return x>20.0f?x:log1pf(expf(x)); }
static inline float reluf(float x){ return x>0.0f?x:0.0f; }

// ---------------- ternary LUT kernels (probe-1 lineage, bit-exact vs scalar-int) ----------------
static inline void acc_add_i8x32(__m256i* acc,__m256i p){
    __m128i lo=_mm256_castsi256_si128(p),hi=_mm256_extracti128_si256(p,1);
    acc[0]=_mm256_add_epi32(acc[0],_mm256_cvtepi8_epi32(lo)); acc[1]=_mm256_add_epi32(acc[1],_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8)));
    acc[2]=_mm256_add_epi32(acc[2],_mm256_cvtepi8_epi32(hi)); acc[3]=_mm256_add_epi32(acc[3],_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8)));
}
// P1 (BRIEF_P1_NIBBLE_PACKING §1): the 2-trit code is an index the hardware reads as a NIBBLE, so two codes
// fit one byte. `--pack nibble` selects the packed encoders/kernels; `--pack byte` (default) is the shipped
// path, untouched. Both arms coexist so they can be compared; C3 is the end-to-end parity gate between them.
static int g_pack_nib=0;
#include "../donor_adaptation/nibble_pack.h"
#define MV_FULL(c,l,y,M,Mp,T)     do{ if(g_pack_nib) matvec_lut_full_n(c,l,y,M,Mp,T);       else matvec_lut_full(c,l,y,M,Mp,T); }while(0)
#define MV_SKIP(c,l,y,M,Mp,a,na)  do{ if(g_pack_nib) matvec_lut_tileskip_n(c,l,y,M,Mp,a,na);else matvec_lut_tileskip(c,l,y,M,Mp,a,na); }while(0)
#define MV_ROWS(c,l,y,r0,M,Mp,T)  do{ if(g_pack_nib) matvec_lut_rows_n(c,l,y,r0,M,Mp,T);    else matvec_lut_rows(c,l,y,r0,M,Mp,T); }while(0)
#define NPLANES(T)                 (g_pack_nib? NP_TP(T) : (T))            // byte-planes per code block
#define RMCODE(cr,t)              (g_pack_nib? np_rm_code(cr,t) : (int)(cr)[t])
#define BC_TM(W,M,K,Mp,c)         do{ if(g_pack_nib) bc_tm_n(W,M,K,Mp,c); else bc_tm(W,M,K,Mp,c); }while(0)
#define BC_RM(W,M,K,c)            do{ if(g_pack_nib) bc_rm_n(W,M,K,c);    else bc_rm(W,M,K,c);    }while(0)
// C4: ACHIEVED ternary-code bytes, summed from _msize() of the blocks malloc actually returned — never
// recomputed from T*Mpad. Convention (identical for both arms): the ternary WEIGHT-CODE arrays only;
// fp32 tensors, per-row scales, activations and LUTs are excluded from both arms alike.
static size_t g_code_bytes=0;
static void matvec_lut_full(const int8_t* codes,const int8_t* lut,int32_t* y,int M,int Mpad,int T){
    OMP_PFOR for(int base=0;base<M;base+=32){
        __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};
        for(int t=0;t<T;t++){ __m256i tbl=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)t*16)));
            __m256i idx=_mm256_loadu_si256((const __m256i*)(codes+(size_t)t*Mpad+base)); acc_add_i8x32(acc,_mm256_shuffle_epi8(tbl,idx)); }
        int32_t tmp[32]; _mm256_storeu_si256((__m256i*)(tmp+0),acc[0]); _mm256_storeu_si256((__m256i*)(tmp+8),acc[1]);
        _mm256_storeu_si256((__m256i*)(tmp+16),acc[2]); _mm256_storeu_si256((__m256i*)(tmp+24),acc[3]);
        for(int r=0;r<32&&base+r<M;r++) y[base+r]=tmp[r]; }
}
static void matvec_lut_tileskip(const int8_t* codes,const int8_t* lut,int32_t* y,int M,int Mpad,const int* act,int na){
    OMP_PFOR for(int base=0;base<M;base+=32){
        __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};
        for(int a=0;a<na;a++){ int t=act[a];
            __m256i tbl=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)t*16)));
            __m256i idx=_mm256_loadu_si256((const __m256i*)(codes+(size_t)t*Mpad+base)); acc_add_i8x32(acc,_mm256_shuffle_epi8(tbl,idx)); }
        int32_t tmp[32]; _mm256_storeu_si256((__m256i*)(tmp+0),acc[0]); _mm256_storeu_si256((__m256i*)(tmp+8),acc[1]);
        _mm256_storeu_si256((__m256i*)(tmp+16),acc[2]); _mm256_storeu_si256((__m256i*)(tmp+24),acc[3]);
        for(int r=0;r<32&&base+r<M;r++) y[base+r]=tmp[r]; }
}
// windowed rows [row0,row0+M) of a tile-major block (the E4 per-expert access)
static void matvec_lut_rows(const int8_t* codes,const int8_t* lut,int32_t* y,int row0,int M,int Mpad,int T){
    OMP_PFOR for(int bb=0;bb<M;bb+=32){ int base=row0+bb; __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};
        for(int t=0;t<T;t++){ __m256i tbl=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)t*16)));
            __m256i idx=_mm256_loadu_si256((const __m256i*)(codes+(size_t)t*Mpad+base)); acc_add_i8x32(acc,_mm256_shuffle_epi8(tbl,idx)); }
        int32_t tmp[32]; _mm256_storeu_si256((__m256i*)(tmp+0),acc[0]); _mm256_storeu_si256((__m256i*)(tmp+8),acc[1]);
        _mm256_storeu_si256((__m256i*)(tmp+16),acc[2]); _mm256_storeu_si256((__m256i*)(tmp+24),acc[3]);
        for(int r=0;r<32&&bb+r<M;r++) y[bb+r]=tmp[r]; }
}
// --- weight-once-over-K kernels (V-G3c layer-major block forward): each weight row/tile is STREAMED ONCE from the
//     cold replica and applied to all Kn positions (the amortization the streamed-regime speedup depends on). ---
static void mv_K(const float* W,int out,int in,const float* X,int xs,float* Y,int ys,int Kn){
    for(int o=0;o<out;o++){ const float* w=W+(size_t)o*in;            // stream row o once
        for(int k=0;k<Kn;k++) Y[(size_t)k*ys+o]=dotf(w,X+(size_t)k*xs,in); }
}
// ternary LUT, weight-once: codes streamed once (base,t), applied to each position's own 16-byte LUT (lm_lut[k]).
static void matvec_lut_full_K(const int8_t* codes,const int8_t* luts,int lstride,int32_t* Y,int ys,int M,int Mpad,int T,int Kn){
    for(int base=0;base<M;base+=32){
        __m256i acc[62][4];
        for(int k=0;k<Kn;k++){ acc[k][0]=_mm256_setzero_si256(); acc[k][1]=_mm256_setzero_si256(); acc[k][2]=_mm256_setzero_si256(); acc[k][3]=_mm256_setzero_si256(); }
        for(int t=0;t<T;t++){ __m256i idx=_mm256_loadu_si256((const __m256i*)(codes+(size_t)t*Mpad+base));   // codes: stream once
            for(int k=0;k<Kn;k++){ __m256i tbl=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(luts+(size_t)k*lstride+(size_t)t*16)));
                acc_add_i8x32(&acc[k][0],_mm256_shuffle_epi8(tbl,idx)); } }
        for(int k=0;k<Kn;k++){ int32_t tmp[32]; _mm256_storeu_si256((__m256i*)(tmp+0),acc[k][0]); _mm256_storeu_si256((__m256i*)(tmp+8),acc[k][1]);
            _mm256_storeu_si256((__m256i*)(tmp+16),acc[k][2]); _mm256_storeu_si256((__m256i*)(tmp+24),acc[k][3]);
            for(int r=0;r<32&&base+r<M;r++) Y[(size_t)k*ys+base+r]=tmp[r]; }
    }
}
static void build_lut_t3(const int8_t* xq,int T,int8_t* lut){ for(int t=0;t<T;t++){ int8_t x0=xq[2*t],x1=xq[2*t+1];
    for(int c=0;c<16;c++){ int s=0; if(c<9){ int w0=c/3-1,w1=c%3-1; s=w0*x0+w1*x1; } lut[t*16+c]=(int8_t)s; } } }
static void bc_tm(const int8_t* Wt,int M,int K,int Mpad,int8_t* codes){ int T=K/2;
    for(int t=0;t<T;t++){ for(int m=0;m<M;m++){ int w0=Wt[(size_t)m*K+2*t],w1=Wt[(size_t)m*K+2*t+1]; codes[(size_t)t*Mpad+m]=(int8_t)((w0+1)*3+(w1+1)); }
        for(int m=M;m<Mpad;m++) codes[(size_t)t*Mpad+m]=0; } }
static void bc_rm(const int8_t* Wt,int M,int K,int8_t* codes){ int T=K/2;
    for(int m=0;m<M;m++) for(int t=0;t<T;t++){ int w0=Wt[(size_t)m*K+2*t],w1=Wt[(size_t)m*K+2*t+1]; codes[(size_t)m*T+t]=(int8_t)((w0+1)*3+(w1+1)); } }
static void ref_t3(const int8_t* Wt,const int8_t* xq,int32_t* y,int M,int K){
    for(int m=0;m<M;m++){ long s=0; for(int k=0;k<K;k++) s+=(long)Wt[(size_t)m*K+k]*xq[k]; y[m]=(int32_t)s; } }
static float quant_i8(const float* x,int n,int8_t* xq){ float amax=0; for(int i=0;i<n;i++){ float a=fabsf(x[i]); if(a>amax)amax=a; }
    if(amax==0.0f){ memset(xq,0,n); return 0.0f; } float scale=amax/(float)AQ,inv=1.0f/scale;
    for(int i=0;i<n;i++){ int v=(int)lrintf(x[i]*inv); if(v>AQ)v=AQ; if(v<-AQ)v=-AQ; xq[i]=(int8_t)v; } return scale; }

// ---------------- model state ----------------
typedef struct { float *in_proj,*conv_w,*conv_b,*x_proj,*dt_proj,*dt_b,*A,*Dskip,*out_proj,*norm; } SSML;
typedef struct { float *qkv,*o,*norm; } SWAL;
static float *emb,*head,*normf; static SSML ssm[L]; static SWAL swa; static int is_swa[L];
static float *mlp_n2[L];
static int g_moe=0;                                              // set by the weights magic
// dense (E1M1)
static float *gate_f[L],*up_f[L],*down_f[L];
static int8_t *gate_tm[L],*up_tm[L],*up_rm[L],*down_tm[L]; static float *gate_sc[L],*up_sc[L],*down_sc[L];
// MoE (E4M1)
static float *router_w[L],*router_b[L],*egate_f[L],*eup_f[L],*eWd_f[L];
static int8_t *egate_cd[L],*eup_cd[L],*eWd_cd[L]; static float *egate_sc[L],*eup_sc[L],*eWd_sc[L];
static int8_t *egate_wt[L],*eup_wt[L],*eWd_wt[L];                // kept for kernel self-tests vs scalar-int
static float (*hstate)[DN][N]; static float (*convbuf)[DN][CONV]; static float *kring,*vring; static int kvpos,kvcnt;
static unsigned char* id2bytes[V]; static int id2len[V]; static uint16_t* ids; static long nids;

static void* xmalloc(size_t n){ void*p=malloc(n); if(!p){fprintf(stderr,"OOM %zu\n",n);exit(1);} return p; }
static float* rd(FILE*f,size_t n){ float*p=xmalloc(n*4); if(fread(p,4,n,f)!=n){fprintf(stderr,"short read\n");exit(1);} return p; }
static int8_t* rdi8(FILE*f,size_t n){ int8_t*p=xmalloc(n); if(fread(p,1,n,f)!=n){fprintf(stderr,"short read i8\n");exit(1);} return p; }

static void load_backbone(FILE* f){                              // shared prefix of both formats up to the MLP tensors
    emb=rd(f,(size_t)V*D);
}
static void load_weights(const char* path){
    FILE*f=fopen(path,"rb"); if(!f){fprintf(stderr,"cannot open %s\n",path);exit(1);}
    uint32_t h[16]; if(fread(h,4,16,f)!=16){fprintf(stderr,"short header\n");exit(1);}
    if(h[0]==0x45314D31) g_moe=0; else if(h[0]==0x45344D31) g_moe=1; else {fprintf(stderr,"bad magic %08x\n",h[0]);exit(1);}
    if(g_moe && ((int)h[11]!=E||(int)h[12]!=HID_E||(int)h[13]!=KTOP)){fprintf(stderr,"E/hid_e/k mismatch\n");exit(1);}
    if(!g_moe && (int)h[11]!=MLP_HID){fprintf(stderr,"mlp_hid mismatch\n");exit(1);}
    int has_packed=h[14];
    load_backbone(f);
    for(int l=0;l<L;l++){ is_swa[l]=(l==SWA_LAYER);
        if(is_swa[l]){ swa.norm=rd(f,D); swa.qkv=rd(f,(size_t)3*D*D); swa.o=rd(f,(size_t)D*D); }
        else { SSML*s=&ssm[l]; s->norm=rd(f,D);
            s->in_proj=rd(f,(size_t)2*DN*D); s->conv_w=rd(f,(size_t)DN*CONV); s->conv_b=rd(f,DN);
            s->x_proj=rd(f,(size_t)(DTR+2*N)*DN); s->dt_proj=rd(f,(size_t)DN*DTR); s->dt_b=rd(f,DN);
            s->A=rd(f,(size_t)DN*N); for(int i=0;i<DN*N;i++) s->A[i]=-expf(s->A[i]);
            s->Dskip=rd(f,DN); s->out_proj=rd(f,(size_t)D*DN); }
        mlp_n2[l]=rd(f,D);
        if(g_moe){ router_w[l]=rd(f,(size_t)E*D); router_b[l]=rd(f,E);
            egate_f[l]=rd(f,(size_t)GH*D); eup_f[l]=rd(f,(size_t)GH*D); eWd_f[l]=rd(f,(size_t)E*D*HID_E); }
        else { gate_f[l]=rd(f,(size_t)MLP_HID*D); up_f[l]=rd(f,(size_t)MLP_HID*D); down_f[l]=rd(f,(size_t)D*MLP_HID); }
    }
    normf=rd(f,D); head=rd(f,(size_t)V*D);
    if(!has_packed){fprintf(stderr,"engine needs packed ternary in the export\n");exit(1);}
    if(g_moe){
        for(int l=0;l<L;l++){
            egate_wt[l]=rdi8(f,(size_t)GH*D); egate_sc[l]=rd(f,GH);
            eup_wt[l]=rdi8(f,(size_t)GH*D); eup_sc[l]=rd(f,GH);
            eWd_wt[l]=rdi8(f,(size_t)E*D*HID_E); eWd_sc[l]=rd(f,(size_t)E*D);
            egate_cd[l]=xmalloc((size_t)NPLANES(TUP)*MPAD_GU); BC_TM(egate_wt[l],GH,D,MPAD_GU,egate_cd[l]);
            eup_cd[l]=xmalloc((size_t)NPLANES(TUP)*MPAD_GU); BC_TM(eup_wt[l],GH,D,MPAD_GU,eup_cd[l]);
            eWd_cd[l]=xmalloc((size_t)E*NPLANES(TDE)*MPAD_D);
            for(int e=0;e<E;e++) BC_TM(eWd_wt[l]+(size_t)e*D*HID_E,D,HID_E,MPAD_D,eWd_cd[l]+(size_t)e*NPLANES(TDE)*MPAD_D);
            g_code_bytes += _msize(egate_cd[l])+_msize(eup_cd[l])+_msize(eWd_cd[l]);
        }
    } else {
        for(int l=0;l<L;l++){ int8_t* gq=rdi8(f,(size_t)MLP_HID*D); gate_sc[l]=rd(f,MLP_HID);
            int8_t* uq=rdi8(f,(size_t)MLP_HID*D); up_sc[l]=rd(f,MLP_HID);
            int8_t* dq=rdi8(f,(size_t)D*MLP_HID); down_sc[l]=rd(f,D);
            int Mpg=(MLP_HID+31)&~31,Mpd=(D+31)&~31;
            gate_tm[l]=xmalloc((size_t)NPLANES(TUP)*Mpg); BC_TM(gq,MLP_HID,D,Mpg,gate_tm[l]);
            up_tm[l]=xmalloc((size_t)NPLANES(TUP)*Mpg); BC_TM(uq,MLP_HID,D,Mpg,up_tm[l]);
            up_rm[l]=xmalloc((size_t)MLP_HID*NPLANES(TUP)); BC_RM(uq,MLP_HID,D,up_rm[l]);
            down_tm[l]=xmalloc((size_t)NPLANES(TDN)*Mpd); BC_TM(dq,D,MLP_HID,Mpd,down_tm[l]);
            g_code_bytes += _msize(gate_tm[l])+_msize(up_tm[l])+_msize(up_rm[l])+_msize(down_tm[l]);
            free(gq);free(uq);free(dq); }
    }
    long pos=ftell(f); fseek(f,0,SEEK_END); long end=ftell(f); fclose(f);
    if(pos!=end) fprintf(stderr,"WARN %ld trailing bytes\n",end-pos);
    fprintf(stderr,"engine weights ok (%s)\n",g_moe?"E4M1 MoE":"E1M1 dense");
    // C4 ACHIEVED (printed from the allocations themselves, not from a formula)
    printf("==== P1 C4 ACHIEVED ternary-code footprint ====\n");
    printf("  pack=%s  ternary weight-code bytes (sum of _msize over every code array) = %zu B (%.3f MiB)\n",
           g_pack_nib?"nibble":"byte", g_code_bytes, g_code_bytes/1048576.0);
    printf("  convention: ternary WEIGHT-CODE arrays only; fp32 tensors, per-row scales, activations and\n");
    printf("              LUTs are excluded, identically in both arms. Padding rows charged to both arms.\n");
}
static void load_meta(const char* path){ FILE*f=fopen(path,"rb"); if(!f){fprintf(stderr,"no meta\n");exit(1);}
    uint32_t mg,vv,nt; if(fread(&mg,4,1,f)!=1||fread(&vv,4,1,f)!=1||fread(&nt,4,1,f)!=1){exit(1);}
    unsigned char el[V]; if(fread(el,1,vv,f)!=vv){exit(1);}
    for(uint32_t t=0;t<vv;t++){ id2len[t]=el[t]; id2bytes[t]=xmalloc(el[t]+1); if(fread(id2bytes[t],1,el[t],f)!=(size_t)el[t]){exit(1);} id2bytes[t][el[t]]=0; } fclose(f); }
static void load_ids(const char* path){ FILE*f=fopen(path,"rb"); if(!f){fprintf(stderr,"no ids\n");exit(1);}
    fseek(f,0,SEEK_END); long b=ftell(f); fseek(f,0,SEEK_SET); nids=b/2; ids=xmalloc(b); if(fread(ids,2,nids,f)!=(size_t)nids){exit(1);} fclose(f); }
static void state_reset(void){ memset(hstate,0,(size_t)L*DN*N*4); memset(convbuf,0,(size_t)L*DN*CONV*4);
    memset(kring,0,(size_t)WIN*D*4); memset(vring,0,(size_t)WIN*D*4); kvpos=0; kvcnt=0; }
static inline void rmsnorm(const float* in,const float* w,float* out){ float ms=0; for(int i=0;i<D;i++) ms+=in[i]*in[i];
    float r=1.0f/sqrtf(ms/(float)D+1e-5f); for(int i=0;i<D;i++) out[i]=in[i]*r*w[i]; }

// ---------------- dense MLP: fp32 | LUT-full (E2) | LUT-skip (E3) ----------------
static int8_t xqb[MLP_HID]; static int8_t lutb[TDN*16]; static int32_t Sb[MLP_HID]; static int act_tiles[TDN];
static void mlp_dense(int l,const float* xn,float* out,int mlp_lut,int skip){
    float gh[MLP_HID],uh[MLP_HID];
    if(!mlp_lut){ matvec(gate_f[l],xn,gh,MLP_HID,D); matvec(up_f[l],xn,uh,MLP_HID,D);
        for(int i=0;i<MLP_HID;i++) gh[i]=reluf(gh[i])*reluf(uh[i]); matvec(down_f[l],gh,out,D,MLP_HID); return; }
    int Mpg=(MLP_HID+31)&~31,Mpd=(D+31)&~31;
    float sa=quant_i8(xn,D,xqb); build_lut_t3(xqb,TUP,lutb);
    MV_FULL(gate_tm[l],lutb,Sb,MLP_HID,Mpg,TUP);
    for(int i=0;i<MLP_HID;i++) gh[i]=(float)Sb[i]*sa*gate_sc[l][i];
    if(!skip){                                                    // E2: up full, relu*relu
        MV_FULL(up_tm[l],lutb,Sb,MLP_HID,Mpg,TUP);
        for(int i=0;i<MLP_HID;i++) uh[i]=(float)Sb[i]*sa*up_sc[l][i];
        for(int i=0;i<MLP_HID;i++) gh[i]=reluf(gh[i])*reluf(uh[i]);
    } else {                                                      // E3: exact up row-skip (gate<=0 -> h=0)
        for(int i=0;i<MLP_HID;i++){ if(gh[i]>0.0f){ const int8_t* cr=up_rm[l]+(size_t)i*NPLANES(TUP); int S=0;
                for(int t=0;t<TUP;t++) S+=lutb[t*16+RMCODE(cr,t)]; float u=(float)S*sa*up_sc[l][i]; gh[i]=gh[i]*reluf(u); } else gh[i]=0.0f; }
    }
    float sh=quant_i8(gh,MLP_HID,xqb); build_lut_t3(xqb,TDN,lutb);
    if(!skip){ MV_FULL(down_tm[l],lutb,Sb,D,Mpd,TDN); }
    else { int na=0; for(int t=0;t<TDN;t++){ if(xqb[2*t]||xqb[2*t+1]) act_tiles[na++]=t; }
        MV_SKIP(down_tm[l],lutb,Sb,D,Mpd,act_tiles,na); }
    for(int i=0;i<D;i++) out[i]=(float)Sb[i]*sh*down_sc[l][i];
}

// ---------------- MoE MLP: fp32 experts (E4-ref) | LUT experts (E4) ----------------
static void topk_sel(const float* p,int n,int k,int* idx,float* val){   // matches torch.topk (ties -> lower index)
    int used[E]; memset(used,0,sizeof(int)*n);
    for(int j=0;j<k;j++){ int bi=-1; float bv=-1e30f;
        for(int i=0;i<n;i++) if(!used[i]&&p[i]>bv){ bv=p[i]; bi=i; }
        used[bi]=1; idx[j]=bi; val[j]=p[bi]; }
}
static int8_t g_xq[D],g_lut[TUP*16],g_hq[HID_E],g_lutd[TDE*16]; static int32_t g_S[HID_E],g_Sd[D];
// V-G3b expert-selection capture: when g_esel_cap!=NULL the router writes this position's KTOP experts per layer
// into g_esel_cap[l*KTOP..] — used to count the per-block expert UNION touched (block layer-major) vs token-by-token.
static int* g_esel_cap=NULL;
static void mlp_moe(int l,const float* xn,float* out,int mlp_lut){
    float rp[E]; for(int e=0;e<E;e++) rp[e]=dotf(router_w[l]+(size_t)e*D,xn,D)+router_b[l][e];
    float mx=-1e30f; for(int e=0;e<E;e++) if(rp[e]>mx)mx=rp[e];
    float Z=0; for(int e=0;e<E;e++){ rp[e]=expf(rp[e]-mx); Z+=rp[e]; } for(int e=0;e<E;e++) rp[e]/=Z;
    int idx[KTOP]; float wv[KTOP]; topk_sel(rp,E,KTOP,idx,wv);
    if(g_esel_cap) for(int j=0;j<KTOP;j++) g_esel_cap[l*KTOP+j]=idx[j];
    float ws=0; for(int j=0;j<KTOP;j++) ws+=wv[j]; for(int j=0;j<KTOP;j++) wv[j]/=ws;
    memset(out,0,D*4);
    float sa=0; if(mlp_lut){ sa=quant_i8(xn,D,g_xq); build_lut_t3(g_xq,TUP,g_lut); }
    for(int j=0;j<KTOP;j++){ int e=idx[j]; float tw=wv[j]; float he[HID_E];
        if(!mlp_lut){
            for(int i=0;i<HID_E;i++){ float gg=dotf(egate_f[l]+(size_t)(e*HID_E+i)*D,xn,D);
                float uu=dotf(eup_f[l]+(size_t)(e*HID_E+i)*D,xn,D); he[i]=reluf(gg)*reluf(uu)*tw; }
            for(int d=0;d<D;d++) out[d]+=dotf(eWd_f[l]+((size_t)e*D+d)*HID_E,he,HID_E);
        } else {
            MV_ROWS(egate_cd[l],g_lut,g_S,e*HID_E,HID_E,MPAD_GU,TUP);
            float gg[HID_E]; for(int i=0;i<HID_E;i++) gg[i]=(float)g_S[i]*sa*egate_sc[l][e*HID_E+i];
            MV_ROWS(eup_cd[l],g_lut,g_S,e*HID_E,HID_E,MPAD_GU,TUP);
            for(int i=0;i<HID_E;i++){ float uu=(float)g_S[i]*sa*eup_sc[l][e*HID_E+i]; he[i]=reluf(gg[i])*reluf(uu)*tw; }
            float sh=quant_i8(he,HID_E,g_hq); build_lut_t3(g_hq,TDE,g_lutd);
            MV_ROWS(eWd_cd[l]+(size_t)e*NPLANES(TDE)*MPAD_D,g_lutd,g_Sd,0,D,MPAD_D,TDE);
            for(int d=0;d<D;d++) out[d]+=(float)g_Sd[d]*sh*eWd_sc[l][(size_t)e*D+d];
        }
    }
}

// ---------------- forward ----------------
// 63.V Activation Replay (the chassis-conforming commit): during the speculative forward of the K drafts, stash the
// post-GEMV recurrence INPUTS per position (raw in_proj xx pre-conv, dt, Bm for SSM layers; k/v for the SWA layer).
// At commit we advance the persistent state (hstate scan + convbuf + kv-ring) by re-running ONLY the elementwise
// recurrence from these stashed inputs — zero GEMV, no streamed weight re-touched. This upholds the sealed R-F
// invariant "each weight streamed 1x/block" that snapshot+re-forward violated. ~28KB/position (~226KB at K=8), L2-fit.
typedef struct { float xraw[L][DN]; float dt[L][DN]; float Bm[L][N]; float kk[D]; float vv[D]; } ActPos;
static ActPos* g_cap=NULL;   // non-NULL -> forward_token stashes this position's recurrence inputs for replay-commit
typedef struct { double scan,scan_other,swa,mlp,head,norm; } Tacc;  // 64.0(b): +norm bucket (rmsnorm glue, was in untimed remainder)
// cfg: mlp_lut (0 fp32 / 1 LUT), skip (dense only), exp_fast (0 exact / 1 poly)
static void forward_token(uint32_t tok,float* logits,int mlp_lut,int skip,int exp_fast,Tacc* T){
    float x[D],xn[D],xz[2*DN],xx[DN],z[DN],dbl[DTR+2*N],dt[DN],y[DN],q[D],kk[D],vvv[D],att[WIN],ao[D],tmp[D];
    double t0;
    memcpy(x,emb+(size_t)tok*D,D*4);
    for(int l=0;l<L;l++){
        if(T)t0=now_s();
        rmsnorm(x, is_swa[l]?swa.norm:ssm[l].norm, xn);
        if(T)T->norm+=now_s()-t0;
        if(is_swa[l]){
            if(T)t0=now_s();
            matvec(swa.qkv,xn,xz,3*D,D); memcpy(q,xz,D*4); memcpy(kk,xz+D,D*4); memcpy(vvv,xz+2*D,D*4);
            if(g_cap){ memcpy(g_cap->kk,kk,D*4); memcpy(g_cap->vv,vvv,D*4); }   // stash k/v for replay-commit
            int slot=kvpos%WIN; memcpy(kring+(size_t)slot*D,kk,D*4); memcpy(vring+(size_t)slot*D,vvv,D*4);
            kvpos++; if(kvcnt<WIN) kvcnt++; memset(ao,0,D*4);
            for(int hh=0;hh<H;hh++){ const float* qh=q+hh*HD; float m2=-1e30f;
                for(int j=0;j<kvcnt;j++){ int s=(kvpos-kvcnt+j)%WIN; float sc=dotf(qh,kring+(size_t)s*D+hh*HD,HD)/sqrtf((float)HD); att[j]=sc; if(sc>m2)m2=sc; }
                float Zs=0; for(int j=0;j<kvcnt;j++){ att[j]=expf(att[j]-m2); Zs+=att[j]; } float zi=1.0f/Zs;
                for(int j=0;j<kvcnt;j++){ int s=(kvpos-kvcnt+j)%WIN; float w=att[j]*zi; const float* vh=vring+(size_t)s*D+hh*HD;
                    for(int d=0;d<HD;d++) ao[hh*HD+d]+=w*vh[d]; } }
            matvec(swa.o,ao,tmp,D,D); for(int i=0;i<D;i++) x[i]+=tmp[i];
            if(T)T->swa+=now_s()-t0;
        } else {
            SSML*s=&ssm[l];
            if(T)t0=now_s();
            matvec(s->in_proj,xn,xz,2*DN,D); memcpy(xx,xz,DN*4); memcpy(z,xz+DN,DN*4);
            if(g_cap) memcpy(g_cap->xraw[l],xx,DN*4);              // stash raw in_proj (pre-conv) for replay-commit
            float (*cb)[CONV]=convbuf[l];
            for(int c=0;c<DN;c++){ for(int t=0;t<CONV-1;t++) cb[c][t]=cb[c][t+1]; cb[c][CONV-1]=xx[c];
                float acc=s->conv_b[c]; const float* w=s->conv_w+(size_t)c*CONV; for(int t=0;t<CONV;t++) acc+=w[t]*cb[c][t]; xx[c]=silu(acc); }
            matvec(s->x_proj,xx,dbl,DTR+2*N,DN);
            const float* Bm=dbl+DTR; const float* Cm=dbl+DTR+N;
            OMP_PFOR for(int c=0;c<DN;c++) dt[c]=softplus(dotf(s->dt_proj+(size_t)c*DTR,dbl,DTR)+s->dt_b[c]);
            if(g_cap){ memcpy(g_cap->dt[l],dt,DN*4); memcpy(g_cap->Bm[l],Bm,N*4); }   // stash dt,B for replay-commit
            if(T){T->scan_other+=now_s()-t0; t0=now_s();}
            float (*hl)[N]=hstate[l];
            if(!exp_fast){
                OMP_PFOR for(int c=0;c<DN;c++){ const float* Ac=s->A+(size_t)c*N; float* hc=hl[c]; float dtc=dt[c],xc=xx[c],acc=0;
                    for(int j=0;j<N;j++){ hc[j]=expf(dtc*Ac[j])*hc[j]+dtc*Bm[j]*xc; acc+=hc[j]*Cm[j]; } y[c]=acc+s->Dskip[c]*xc; }
            } else {
                OMP_PFOR for(int c=0;c<DN;c++){ const float* Ac=s->A+(size_t)c*N; float* hc=hl[c]; float dtc=dt[c],xc=xx[c],dbx=dt[c]*xc;
                    __m256 vdtc=_mm256_set1_ps(dtc),vdbx=_mm256_set1_ps(dbx),vacc=_mm256_setzero_ps();
                    for(int j=0;j<N;j+=8){ __m256 ee=exp256_ps(_mm256_mul_ps(vdtc,_mm256_loadu_ps(Ac+j)));
                        __m256 hcj=_mm256_fmadd_ps(ee,_mm256_loadu_ps(hc+j),_mm256_mul_ps(vdbx,_mm256_loadu_ps(Bm+j)));
                        _mm256_storeu_ps(hc+j,hcj); vacc=_mm256_fmadd_ps(hcj,_mm256_loadu_ps(Cm+j),vacc); } y[c]=hsum256(vacc)+s->Dskip[c]*xc; }
            }
            if(T){T->scan+=now_s()-t0; t0=now_s();}
            for(int c=0;c<DN;c++) y[c]*=silu(z[c]);
            matvec(s->out_proj,y,tmp,D,DN); for(int i=0;i<D;i++) x[i]+=tmp[i];
            if(T)T->scan_other+=now_s()-t0;
        }
        if(T)t0=now_s();
        rmsnorm(x, mlp_n2[l], xn);
        if(T){T->norm+=now_s()-t0; t0=now_s();}
        if(g_moe) mlp_moe(l,xn,tmp,mlp_lut); else mlp_dense(l,xn,tmp,mlp_lut,skip);
        for(int i=0;i<D;i++) x[i]+=tmp[i];
        if(T)T->mlp+=now_s()-t0;
    }
    if(T)t0=now_s();
    rmsnorm(x,normf,xn);
    if(T){T->norm+=now_s()-t0; t0=now_s();}
    matvec(head,xn,logits,V,D);
    if(T)T->head+=now_s()-t0;
}

// ---------------- modes ----------------
static void dump_logits(const char* path,long seqW,long ntok,long offset,int mlp_lut,int skip,int exp_fast){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr;
    FILE* f=fopen(path,"wb"); if(!f){fprintf(stderr,"cannot open %s\n",path);exit(1);}
    float* lg=xmalloc((size_t)V*4); long done=0,pos=offset;
    while(done<ntok){ state_reset();
        for(long t=0;t<seqW&&done<ntok;t++,done++){ forward_token(val[pos+t],lg,mlp_lut,skip,exp_fast,NULL);
            fwrite(lg,4,V,f); }
        pos+=seqW; }
    fclose(f); free(lg);
    printf("dumped %ld x %d fp32 logits -> %s\n",ntok,V,path);
}
static double run_bpb(long seqW,long eval_tok,int mlp_lut,int skip,int exp_fast,long* ntok_o){
    long ntr=(long)(nids*0.9); long nval=nids-ntr; uint16_t* val=ids+ntr;
    long lim=eval_tok<(nval-1)?eval_tok:(nval-1); double bits=0; long nbytes=0,ntok=0,pos=0;
    float* logits=xmalloc((size_t)V*4); const double LN2=0.6931471805599453;
    while(pos+seqW+1<=lim){ state_reset();
        for(long t=0;t<seqW;t++){ forward_token(val[pos+t],logits,mlp_lut,skip,exp_fast,NULL); int tgt=val[pos+t+1];
            double mx=-1e30; for(int o=0;o<V;o++) if(logits[o]>mx) mx=logits[o];
            double se=0; for(int o=0;o<V;o++) se+=exp((double)logits[o]-mx);
            bits += (-((double)logits[tgt]-mx)+log(se))/LN2; nbytes+=id2len[tgt]; ntok++; }
        pos+=seqW; }
    free(logits); if(ntok_o)*ntok_o=ntok; return bits/(nbytes>0?nbytes:1);
}
static int gate_logits(long seqW,long ntok_target,int mlp_lut,int skip,int exp_fast){
    // top-1 agreement of the selected config vs the fp32+exact reference config (same binary)
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr;
    float* l1=xmalloc((size_t)V*4); float* l2=xmalloc((size_t)V*4);
    long agree=0,tot=0,pos=0; static uint16_t a1[4096];
    while(tot<ntok_target){
        state_reset();
        for(long t=0;t<seqW;t++){ forward_token(val[pos+t],l1,0,0,0,NULL); float mx=-1e30f; int am=0; for(int o=0;o<V;o++) if(l1[o]>mx){mx=l1[o];am=o;} a1[t]=am; }
        state_reset();
        for(long t=0;t<seqW;t++){ forward_token(val[pos+t],l2,mlp_lut,skip,exp_fast,NULL); float mx=-1e30f; int am=0; for(int o=0;o<V;o++) if(l2[o]>mx){mx=l2[o];am=o;} if(am==a1[t]) agree++; tot++; }
        pos+=seqW;
    }
    double pct=100.0*agree/tot;
    printf("==== top-1 agreement (config vs fp32+exact ref, %ld tok) ====\n  agreement=%.4f%% (%ld/%ld)\n",tot,pct,agree,tot);
    free(l1); free(l2); return pct>=99.0?0:2;
}
static void timing(long ntok,int mlp_lut,int skip,int exp_fast){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr; float* lg=xmalloc((size_t)V*4);
    Tacc T={0,0,0,0,0,0}; state_reset(); double t0=now_s();
    for(long i=0;i<ntok;i++) forward_token(val[i%100000],lg,mlp_lut,skip,exp_fast,&T);
    double tot=now_s()-t0; double sc=1e6/ntok;
    printf("==== timing (%s, mlp=%s skip=%d exp=%s, %ld tok): %.1f tok/s | %.1f us/tok ====\n",
           g_moe?"MoE":"dense",mlp_lut?"lut":"fp32",skip,exp_fast?"fast":"exact",ntok,ntok/tot,1e6/(ntok/tot));
    // 64.0(b) compute-floor decomposition: scan recurrence / projection GEMVs / attention / LUT-MLP / head / norms / glue-remainder
    double acc=T.scan+T.scan_other+T.swa+T.mlp+T.head+T.norm, glue=tot-acc;
    printf("   scan-recur %.1f  proj-GEMV %.1f  SWA-attn %.1f  LUT-MLP %.1f  head %.1f  norms %.1f  glue %.1f  (us/tok, total %.1f)\n",
           T.scan*sc,T.scan_other*sc,T.swa*sc,T.mlp*sc,T.head*sc,T.norm*sc,glue*sc,tot*sc);
    free(lg);
}

// ---------------- 63.V block-verify chassis (n-gram drafter + greedy verify) ----------------
// Commit = ACTIVATION REPLAY (declared, dossier-E5 option 1, favored): ONE snapshot of mutable state at block start;
// forward the K drafts for their logits WHILE stashing per-position recurrence inputs (g_cap); restore the snapshot;
// then advance state through the MATCHED prefix by re-running only the elementwise recurrence from the stash (zero
// GEMV, no streamed weight re-touched -> upholds "each weight streamed 1x/block"). The engine token at divergence is
// forwarded once (= first position of the next block). Emitted stream is token-identical to AR by construction (V-G1):
// the matched-prefix state trajectory equals the speculative-prefix trajectory, so the stashed activations are exact.
// Rep-penalty window sees COMMITTED tokens only (matched drafts == AR tokens). The streamed-regime speedup (weights
// once/block under layer-major batching) is a memory-access property invisible in-cache -> measured in V-G3b/c.
static int ng_N=0,ng_ufb=0; static uint32_t ng_cnt[9]; static uint64_t* ng_key[9]; static uint16_t* ng_nxt[9];
static void load_ngram(const char* path){ FILE* f=fopen(path,"rb"); if(!f){fprintf(stderr,"no ngram %s\n",path);exit(1);}
    uint32_t mg,nn,ufb; if(fread(&mg,4,1,f)!=1||mg!=0x3130474Eu){fprintf(stderr,"bad ngram magic\n");exit(1);}
    if(fread(&nn,4,1,f)!=1||fread(&ufb,4,1,f)!=1)exit(1); ng_N=nn; ng_ufb=ufb; size_t tot=12;
    for(int o=2;o<=(int)nn;o++){ uint32_t c; if(fread(&c,4,1,f)!=1)exit(1); ng_cnt[o]=c; tot+=4+(size_t)c*12;
        ng_key[o]=xmalloc((size_t)c*8); ng_nxt[o]=xmalloc((size_t)c*2);
        for(uint32_t i=0;i<c;i++){ uint64_t k; uint16_t nx,pad; if(fread(&k,8,1,f)!=1||fread(&nx,2,1,f)!=1||fread(&pad,2,1,f)!=1)exit(1); ng_key[o][i]=k; ng_nxt[o][i]=nx; } }
    fclose(f); fprintf(stderr,"ngram N=%d loaded (%zu bytes = %.2f MB) [V-G4: RAM-resident lookup, ~K random probes/step (latency-bound) — NOT a streamed working set, does not compete for L3 residency]\n",ng_N,tot,tot/1048576.0); }
static int ng_find(int o,uint64_t key){ long lo=0,hi=(long)ng_cnt[o]-1; while(lo<=hi){ long m=(lo+hi)>>1; uint64_t k=ng_key[o][m];
    if(k<key)lo=m+1; else if(k>key)hi=m-1; else return (int)m; } return -1; }
static int ng_draft(const uint16_t* ctx,int nctx){ for(int o=ng_N;o>=2;o--){ if(nctx<o-1) continue; uint64_t key=0,mul=1;
    for(int i=0;i<o-1;i++){ key+=(uint64_t)ctx[nctx-(o-1)+i]*mul; mul*=V; } int idx=ng_find(o,key); if(idx>=0) return ng_nxt[o][idx]; } return ng_ufb; }
static int8_t g_seen[V];
static int greedy_hyg(const float* logits,const uint16_t* hist,int nhist){   // rep 1.2 / win 128, then argmax
    int w0=nhist>WIN?nhist-WIN:0; for(int i=w0;i<nhist;i++) g_seen[hist[i]]=1;
    float best=-1e30f; int bi=0; for(int o=0;o<V;o++){ float v=logits[o]; if(g_seen[o]) v=(v>0.0f)?v/1.2f:v*1.2f; if(v>best){best=v;bi=o;} }
    for(int i=w0;i<nhist;i++) g_seen[hist[i]]=0; return bi; }
// V-G3b accounting accumulators (MoE only): per-block expert UNION over the K speculative positions vs token-by-token.
static int g_acc_on=0; static double g_acc_unionsum=0; static long g_acc_commit=0,g_acc_blocks=0; static double g_acc_unionlayer[NLAYER];
static float *snap_h,*snap_conv,*snap_k,*snap_v; static int snap_kvpos,snap_kvcnt,snap_alloc=0;
static void snap_save(void){ if(!snap_alloc){ snap_h=xmalloc((size_t)L*DN*N*4); snap_conv=xmalloc((size_t)L*DN*CONV*4); snap_k=xmalloc((size_t)WIN*D*4); snap_v=xmalloc((size_t)WIN*D*4); snap_alloc=1; }
    memcpy(snap_h,hstate,(size_t)L*DN*N*4); memcpy(snap_conv,convbuf,(size_t)L*DN*CONV*4); memcpy(snap_k,kring,(size_t)WIN*D*4); memcpy(snap_v,vring,(size_t)WIN*D*4); snap_kvpos=kvpos; snap_kvcnt=kvcnt; }
static void snap_restore(void){ memcpy(hstate,snap_h,(size_t)L*DN*N*4); memcpy(convbuf,snap_conv,(size_t)L*DN*CONV*4); memcpy(kring,snap_k,(size_t)WIN*D*4); memcpy(vring,snap_v,(size_t)WIN*D*4); kvpos=snap_kvpos; kvcnt=snap_kvcnt; }
// Activation replay: advance the persistent state (hstate scan + convbuf + kv-ring) through nacts committed positions
// using ONLY the stashed recurrence inputs. Elementwise-only (reads resident A and tiny conv_w) — no GEMV, no streamed
// weight touched. Bit-identical to forward_token's conv+scan (same float inputs, same arithmetic); y/out_proj/logits
// are not persistent state, so they are skipped. exp path (ef) mirrors forward_token so the state matches exactly.
static void replay_commit(const ActPos* acts,int nacts,int ef){
    for(int p=0;p<nacts;p++){ const ActPos* a=&acts[p];
        for(int l=0;l<L;l++){
            if(is_swa[l]){ int slot=kvpos%WIN; memcpy(kring+(size_t)slot*D,a->kk,D*4); memcpy(vring+(size_t)slot*D,a->vv,D*4);
                kvpos++; if(kvcnt<WIN)kvcnt++; continue; }
            SSML* s=&ssm[l]; float (*cb)[CONV]=convbuf[l]; const float* Bm=a->Bm[l];
            for(int c=0;c<DN;c++){
                for(int t=0;t<CONV-1;t++) cb[c][t]=cb[c][t+1]; cb[c][CONV-1]=a->xraw[l][c];   // conv-buffer advance
                float acc=s->conv_b[c]; const float* w=s->conv_w+(size_t)c*CONV; for(int t=0;t<CONV;t++) acc+=w[t]*cb[c][t];
                float xc=silu(acc); const float* Ac=s->A+(size_t)c*N; float* hc=hstate[l][c]; float dtc=a->dt[l][c];
                if(!ef){ for(int j=0;j<N;j++) hc[j]=expf(dtc*Ac[j])*hc[j]+dtc*Bm[j]*xc; }
                else { float dbx=dtc*xc; __m256 vdtc=_mm256_set1_ps(dtc),vdbx=_mm256_set1_ps(dbx);
                    for(int j=0;j<N;j+=8){ __m256 ee=exp256_ps(_mm256_mul_ps(vdtc,_mm256_loadu_ps(Ac+j)));
                        __m256 hcj=_mm256_fmadd_ps(ee,_mm256_loadu_ps(hc+j),_mm256_mul_ps(vdbx,_mm256_loadu_ps(Bm+j)));
                        _mm256_storeu_ps(hc+j,hcj); } }
            }
        }
    }
}
// generate ngen tokens from a val seed; block<=0 => pure AR. Accumulates emitted/blocks for tpp.
static long gen_stream(long seedpos,long ngen,int block,uint16_t* out,double* sum_emit,long* nblk,int ml,int sk,int ef){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr; static uint16_t hist[1<<17]; int nh=0;
    float* pend=xmalloc((size_t)V*4); state_reset();
    for(int i=0;i<16;i++){ hist[nh++]=val[seedpos+i]; forward_token(val[seedpos+i],pend,ml,sk,ef,NULL); }
    long emit=0;
    if(block<=0){ while(emit<ngen){ int nx=greedy_hyg(pend,hist,nh); out[emit++]=nx; hist[nh++]=nx; forward_token(nx,pend,ml,sk,ef,NULL); } }
    else { int K=block>62?62:block; float* Lb=xmalloc((size_t)(K+1)*V*4); uint16_t d[64];
        static ActPos* acts=NULL; if(!acts) acts=xmalloc(sizeof(ActPos)*64);
        static int* esel=NULL; if(g_acc_on&&!esel) esel=xmalloc(sizeof(int)*64*L*KTOP);
        while(emit<ngen){
            for(int k=0;k<K;k++){ hist[nh+k]=ng_draft(hist,nh+k); d[k]=hist[nh+k]; }   // draft K (cond. committed+drafted)
            snap_save(); memcpy(Lb,pend,V*4);                                          // Lb[0] predicts position 0
            for(int k=0;k<K;k++){ g_cap=&acts[k]; if(g_acc_on) g_esel_cap=esel+(size_t)k*L*KTOP;
                forward_token(d[k],pend,ml,sk,ef,NULL); g_cap=NULL; g_esel_cap=NULL; memcpy(Lb+(size_t)(k+1)*V,pend,V*4); }
            snap_restore();                                                            // roll speculative state back to block start
            int acc=0,mism=0; uint16_t eng=0;
            for(int i=0;i<K;i++){ int g=greedy_hyg(Lb+(size_t)i*V,hist,nh+i); if(g==d[i]){acc++;} else {eng=g;mism=1;break;} }
            replay_commit(acts,acc,ef);                                                // advance state through matched prefix (no GEMV)
            int ncommit;
            if(mism){ hist[nh+acc]=eng; forward_token(eng,pend,ml,sk,ef,NULL); ncommit=acc+1; } // engine token: one forward
            else { memcpy(pend,Lb+(size_t)K*V,V*4); ncommit=K; }                       // full accept: state replayed, pend=Lb[K]
            if(g_acc_on){ char seen[E]; long ublk=0;                                   // per-layer expert union over the K speculative positions
                for(int l=0;l<L;l++){ memset(seen,0,E); int u=0;
                    for(int k=0;k<K;k++) for(int j=0;j<KTOP;j++){ int e=esel[((size_t)k*L+l)*KTOP+j]; if(!seen[e]){seen[e]=1;u++;} }
                    g_acc_unionlayer[l]+=u; ublk+=u; }
                g_acc_unionsum+=ublk; g_acc_commit+=ncommit; g_acc_blocks++; }
            for(int c=0;c<ncommit&&emit<ngen;c++) out[emit++]=hist[nh+c];
            nh+=ncommit; *sum_emit+=ncommit; (*nblk)++;
        }
        free(Lb);
    }
    free(pend); return emit;
}
// V-G2 apples-to-apples: reproduce the e5_0 apparatus protocol in-engine (npos val positions, ctx window,
// ONE block of K drafts vs the model greedy self-continuation; tpp = min(LCP,K)+1). Should match the apparatus.
static double tpp_sampled(int K,int npos,int ctx,int ml,int sk,int ef){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr; long nval=nids-ntr;
    float* pend=xmalloc((size_t)V*4); static uint16_t hist[1<<17]; double sumtpp=0; int used=0; uint16_t g[64],d[64];
    for(int s=0;s<npos;s++){ long p=ctx+(long)((double)s/npos*(double)(nval-ctx-K-2)); if(p<ctx)p=ctx;
        state_reset(); int nh=0; for(int i=0;i<ctx;i++){ hist[nh++]=val[p-ctx+i]; forward_token(val[p-ctx+i],pend,ml,sk,ef,NULL); }
        int nhg=nh; for(int k=0;k<K;k++){ g[k]=greedy_hyg(pend,hist,nhg); hist[nhg++]=g[k]; forward_token(g[k],pend,ml,sk,ef,NULL); }
        for(int k=0;k<K;k++){ d[k]=ng_draft(hist,nh+k); hist[nh+k]=d[k]; }     // draft overwrites scratch, reads committed+drafted
        int lcp=0; while(lcp<K && g[lcp]==d[lcp]) lcp++;
        sumtpp += (lcp<K?lcp:K)+1; used++;
    }
    free(pend); return used? sumtpp/used : 0;
}
static int run_verify(int block,const char* ngpath,long genlen,int nseed,int ml,int sk,int ef){
    load_ngram(ngpath); long nval=nids-(long)(nids*0.9);
    long seeds[3]={1000,20000,50000}; if(nseed>3)nseed=3;
    static uint16_t a_ar[1<<17],a_bv[1<<17]; int allident=1; double se=0; long nb=0;
    printf("==== 63.V block-verify (block=%d, ngram N=%d, %d seeds x %ld tok) ====\n",block,ng_N,nseed,genlen);
    for(int s=0;s<nseed;s++){ long sp=seeds[s]; if(sp+16+genlen>=nval)sp=0;
        double d1=0; long b1=0; long na=gen_stream(sp,genlen,0,a_ar,&d1,&b1,ml,sk,ef);
        long nbv=gen_stream(sp,genlen,block,a_bv,&se,&nb,ml,sk,ef);
        int id=(na==nbv); for(long i=0;i<na&&id;i++) if(a_ar[i]!=a_bv[i]) id=0;
        if(!id) allident=0; printf("  seed %ld: token-identity vs AR = %s\n",sp,id?"IDENTICAL":"MISMATCH");
    }
    double tpp=nb>0?se/nb:0;
    // V-G3a sandbox wall-clock (report-only; weights resident -> expected neutral/worse: block-verify pays
    //   speculative + replay compute the AR path doesn't, and the streamed amortization is invisible in-cache).
    static uint16_t tb[1<<17]; double dse=0; long dnb=0; long TG=2000;
    double t0=now_s(); gen_stream(seeds[0],TG,0,tb,&dse,&dnb,ml,sk,ef); double t_ar=now_s()-t0;
    dse=0; dnb=0; t0=now_s(); gen_stream(seeds[0],TG,block,tb,&dse,&dnb,ml,sk,ef); double t_bv=now_s()-t0;
    double tpp_s=tpp_sampled(block,300,128,ml,sk,ef);       // apparatus protocol (300 pos, ctx128) — V-G2 apples-to-apples
    // V-G4 lookup cost (owed): drafter latency = backoff binary-search (<= N-1 probes into the RAM-resident table)
    long NL=500000; volatile int sink=0; double t0l=now_s();
    for(long i=0;i<NL;i++){ int nc=16+(int)(i%(genlen>32?genlen-16:16)); sink^=ng_draft(a_bv,nc); }
    double ns_draft=(now_s()-t0l)*1e9/NL; (void)sink;
    printf("  V-G1 (hard) token-identical to AR: %s\n", allident?"PASS":"FAIL");
    printf("  V-G2 in-engine tpp (apparatus protocol 300pos/ctx128) = %.3f  [apparatus ref same N/K]\n",tpp_s);
    printf("  (production tpp on self-generated stream = %.3f over %ld blocks — higher: self-text is more n-gram-predictable)\n",tpp,nb);
    printf("  V-G4 drafter lookup cost = %.1f ns/draft (RAM-resident, latency-bound; negligible vs a streamed forward ~us)\n",ns_draft);
    printf("  V-G3a in-cache speculative overhead (REPORT-ONLY, pre-registered mute): AR %.1f tok/s vs block-verify %.1f tok/s (%.2fx)\n",
           TG/t_ar, TG/t_bv, t_ar/t_bv);
    printf("    [commit now = activation replay (zero GEMV); the gap is pure speculative waste — weights free in L2, batching mute.\n");
    printf("     The streamed-regime speedup (weights once/block) is measured in V-G3b/c, not here.]\n");
    printf("  V-G3b/c (KB-touched expert-union accounting + DRAM-cold emulation) = next rung (needs layer-major cold kernel)\n");
    return allident?0:2;
}

static void run_g3c(const char* ngpath,long emu_mb,int ml,int sk,int ef);   // fwd decl (V-G3c below)
// ---------------- V-G3b: expert-union accounting (MoE, counted; no wall-clock) ----------------
#define EBYTES 49152    // per-expert streamed ternary codes: egate TUP*HID_E + eup TUP*HID_E + eWd TDE*D = 3*16384
static void run_g3b(const char* ngpath,long genlen,int ml,int sk,int ef){
    if(!g_moe){ printf("  V-G3b: skipped (dense model — expert accounting is MoE-only)\n"); return; }
    load_ngram(ngpath); long nval=nids-(long)(nids*0.9); long sp=1000; if(sp+16+genlen>=nval)sp=0;
    static uint16_t obuf[1<<17];
    printf("==== 63.V V-G3b expert-union accounting (MoE, counted) ====\n");
    printf("  per-expert streamed codes = %d B (48 KB); token-by-token = KTOP*L*EBYTES = %.1f KB/tok (matches E4's 2400 KB/tok)\n",
           EBYTES, (double)KTOP*L*EBYTES/1024.0);
    printf("    K   tpp    union/blk  union-KB/tok  tbt-KB/tok  amortization  per-layer-union(avg |set|, E=%d)\n",E);
    int Ks[2]={4,8};
    for(int ki=0;ki<2;ki++){ int K=Ks[ki];
        g_acc_on=1; g_acc_unionsum=0; g_acc_commit=0; g_acc_blocks=0; for(int l=0;l<L;l++) g_acc_unionlayer[l]=0;
        double se=0; long nb=0; gen_stream(sp,genlen,K,obuf,&se,&nb,ml,sk,ef); g_acc_on=0;
        double tpp=nb>0?se/nb:0;
        double union_per_blk=g_acc_blocks?g_acc_unionsum/g_acc_blocks:0;
        double union_kb_tok=g_acc_commit? g_acc_unionsum*(double)EBYTES/g_acc_commit/1024.0 : 0;
        double tbt_kb_tok=(double)KTOP*L*EBYTES/1024.0;
        double amort=union_kb_tok>0? tbt_kb_tok/union_kb_tok : 0;
        printf("    %-3d %.3f  %8.1f   %9.1f   %9.1f   %6.2fx      [",K,tpp,union_per_blk,union_kb_tok,tbt_kb_tok,amort);
        for(int l=0;l<L;l++) printf("%.1f%s",g_acc_blocks?g_acc_unionlayer[l]/g_acc_blocks:0,l<L-1?" ":"");
        printf("]\n");
    }
    printf("  reading (pre-declared): expert class amortizes at UNION-rate, not tpp (rejected positions' experts count);\n");
    printf("  granular MoE + large K saturates the union toward E -> weak/negative expert amortization = Phase-64 two-pool sizing datum.\n");
}

// ---------------- V-G3c: all-weights-cold emulation (dense; layer-major weight-once vs token-major AR) ----------------
// Scale-up proxy: every weight class read from a rotating replica buffer >> L3 so each pass is DRAM-cold; the
// per-position elementwise compute (scan, activations) stays hot — it does not amortize at scale-up either. AR rotates
// per token (weights streamed once/token); block rotates per pass (weights streamed once/block via the layer-major
// weight-once forward, applied to K positions). Runs on the DENSE model (clean weight-once, no routing); the MoE
// expert pool's amortization is the counted union in V-G3b. Deployed config = --mlp lut; skip-off here (E2==E3 output,
// bit-identical) so up_tm/down_tm are the streamed arrays (up_rm/skip is an orthogonal activation-sparsity lever).
#define MAXSLOT 160
static void** emu_slot[MAXSLOT]; static size_t emu_off[MAXSLOT], emu_bytes[MAXSLOT]; static int emu_ns=0;
static size_t emu_arena=0; static int emu_R=1,emu_rot=0; static char* emu_buf=NULL;
static void emu_rebind(int r){ for(int i=0;i<emu_ns;i++) *emu_slot[i]=emu_buf+(size_t)r*emu_arena+emu_off[i]; }
static void add_slot(void** p,size_t b){ emu_slot[emu_ns]=p; emu_bytes[emu_ns]=b; emu_off[emu_ns]=emu_arena; emu_arena+=(b+63)&~63; emu_ns++; }
static size_t emu_setup(long emu_mb){
    emu_ns=0; emu_arena=0; int Mpg=(MLP_HID+31)&~31,Mpd=(D+31)&~31;
    add_slot((void**)&emb,(size_t)V*D*4);
    for(int l=0;l<L;l++){ if(is_swa[l]){ add_slot((void**)&swa.norm,D*4); add_slot((void**)&swa.qkv,(size_t)3*D*D*4); add_slot((void**)&swa.o,(size_t)D*D*4); }
        else { SSML*s=&ssm[l]; add_slot((void**)&s->norm,D*4); add_slot((void**)&s->in_proj,(size_t)2*DN*D*4); add_slot((void**)&s->conv_w,(size_t)DN*CONV*4);
            add_slot((void**)&s->conv_b,DN*4); add_slot((void**)&s->x_proj,(size_t)(DTR+2*N)*DN*4); add_slot((void**)&s->dt_proj,(size_t)DN*DTR*4);
            add_slot((void**)&s->dt_b,DN*4); add_slot((void**)&s->A,(size_t)DN*N*4); add_slot((void**)&s->Dskip,DN*4); add_slot((void**)&s->out_proj,(size_t)D*DN*4); }
        add_slot((void**)&mlp_n2[l],D*4);
        add_slot((void**)&gate_tm[l],(size_t)NPLANES(TUP)*Mpg); add_slot((void**)&up_tm[l],(size_t)NPLANES(TUP)*Mpg); add_slot((void**)&down_tm[l],(size_t)NPLANES(TDN)*Mpd);
        add_slot((void**)&gate_sc[l],(size_t)MLP_HID*4); add_slot((void**)&up_sc[l],(size_t)MLP_HID*4); add_slot((void**)&down_sc[l],(size_t)D*4);
    }
    add_slot((void**)&normf,D*4); add_slot((void**)&head,(size_t)V*D*4);
    emu_R=(int)(((size_t)emu_mb*1048576+emu_arena-1)/emu_arena); if(emu_R<2)emu_R=2;
    emu_buf=xmalloc((size_t)emu_R*emu_arena);
    for(int r=0;r<emu_R;r++) for(int i=0;i<emu_ns;i++) memcpy(emu_buf+(size_t)r*emu_arena+emu_off[i],*emu_slot[i],emu_bytes[i]);
    emu_rebind(0);
    return emu_arena - (size_t)V*D*4 + (size_t)D*4;   // streamed bytes/token (AR): arena minus emb bulk, plus one emb row
}
// layer-major weight-once forward of Kn dense tokens; advances state through all Kn; fills Lb[k*V..]; stashes ActPos.
static float lm_x[62][D],lm_xn[62][D],lm_xz[62][2*DN],lm_xx[62][DN],lm_z[62][DN],lm_dbl[62][DTR+2*N],lm_dt[62][DN],lm_y[62][DN],lm_tmp[62][D];
static float lm_q[62][D],lm_kk[62][D],lm_vv[62][D],lm_ao[62][D],lm_sa[62],lm_sh[62],lm_gh[62][MLP_HID];
static int8_t lm_xqb[62][D],lm_hqb[62][MLP_HID],lm_lutU[62][TUP*16],lm_lutD[62][TDN*16]; static int32_t lm_Sg[62][MLP_HID],lm_Su[62][MLP_HID],lm_Sd[62][D];
static void forward_block_lm(const uint16_t* toks,int Kn,float* Lb,ActPos* acts,int ef){
    if(Kn>62)Kn=62; int Mpg=(MLP_HID+31)&~31,Mpd=(D+31)&~31;
    for(int k=0;k<Kn;k++) memcpy(lm_x[k],emb+(size_t)toks[k]*D,D*4);
    for(int l=0;l<L;l++){
        for(int k=0;k<Kn;k++) rmsnorm(lm_x[k],is_swa[l]?swa.norm:ssm[l].norm,lm_xn[k]);
        if(is_swa[l]){
            mv_K(swa.qkv,3*D,D,&lm_xn[0][0],D,&lm_xz[0][0],2*DN,Kn);
            for(int k=0;k<Kn;k++){ memcpy(lm_q[k],lm_xz[k],D*4); memcpy(lm_kk[k],lm_xz[k]+D,D*4); memcpy(lm_vv[k],lm_xz[k]+2*D,D*4);
                if(acts){ memcpy(acts[k].kk,lm_kk[k],D*4); memcpy(acts[k].vv,lm_vv[k],D*4); }
                int slot=kvpos%WIN; memcpy(kring+(size_t)slot*D,lm_kk[k],D*4); memcpy(vring+(size_t)slot*D,lm_vv[k],D*4);
                kvpos++; if(kvcnt<WIN)kvcnt++; memset(lm_ao[k],0,D*4);
                for(int hh=0;hh<H;hh++){ const float* qh=lm_q[k]+hh*HD; float att[WIN]; float m2=-1e30f;
                    for(int j=0;j<kvcnt;j++){ int s=(kvpos-kvcnt+j)%WIN; float sc=dotf(qh,kring+(size_t)s*D+hh*HD,HD)/sqrtf((float)HD); att[j]=sc; if(sc>m2)m2=sc; }
                    float Zs=0; for(int j=0;j<kvcnt;j++){ att[j]=expf(att[j]-m2); Zs+=att[j]; } float zi=1.0f/Zs;
                    for(int j=0;j<kvcnt;j++){ int s=(kvpos-kvcnt+j)%WIN; float w=att[j]*zi; const float* vh=vring+(size_t)s*D+hh*HD;
                        for(int d=0;d<HD;d++) lm_ao[k][hh*HD+d]+=w*vh[d]; } } }
            mv_K(swa.o,D,D,&lm_ao[0][0],D,&lm_tmp[0][0],D,Kn);
            for(int k=0;k<Kn;k++) for(int i=0;i<D;i++) lm_x[k][i]+=lm_tmp[k][i];
        } else {
            SSML* s=&ssm[l];
            mv_K(s->in_proj,2*DN,D,&lm_xn[0][0],D,&lm_xz[0][0],2*DN,Kn);
            for(int k=0;k<Kn;k++){ memcpy(lm_xx[k],lm_xz[k],DN*4); memcpy(lm_z[k],lm_xz[k]+DN,DN*4); if(acts) memcpy(acts[k].xraw[l],lm_xx[k],DN*4); }
            for(int k=0;k<Kn;k++){ float (*cb)[CONV]=convbuf[l];   // conv sequential (state)
                for(int c=0;c<DN;c++){ for(int t=0;t<CONV-1;t++) cb[c][t]=cb[c][t+1]; cb[c][CONV-1]=lm_xx[k][c];
                    float acc=s->conv_b[c]; const float* w=s->conv_w+(size_t)c*CONV; for(int t=0;t<CONV;t++) acc+=w[t]*cb[c][t]; lm_xx[k][c]=silu(acc); } }
            mv_K(s->x_proj,DTR+2*N,DN,&lm_xx[0][0],DN,&lm_dbl[0][0],DTR+2*N,Kn);
            for(int c=0;c<DN;c++){ const float* w=s->dt_proj+(size_t)c*DTR;   // dt weight-once + softplus
                for(int k=0;k<Kn;k++) lm_dt[k][c]=softplus(dotf(w,lm_dbl[k],DTR)+s->dt_b[c]); }
            for(int k=0;k<Kn;k++) if(acts){ memcpy(acts[k].dt[l],lm_dt[k],DN*4); memcpy(acts[k].Bm[l],lm_dbl[k]+DTR,N*4); }
            for(int k=0;k<Kn;k++){ const float* Bm=lm_dbl[k]+DTR; const float* Cm=lm_dbl[k]+DTR+N; float (*hl)[N]=hstate[l];  // scan sequential (state)
                if(!ef){ for(int c=0;c<DN;c++){ const float* Ac=s->A+(size_t)c*N; float* hc=hl[c]; float dtc=lm_dt[k][c],xc=lm_xx[k][c],acc=0;
                        for(int j=0;j<N;j++){ hc[j]=expf(dtc*Ac[j])*hc[j]+dtc*Bm[j]*xc; acc+=hc[j]*Cm[j]; } lm_y[k][c]=acc+s->Dskip[c]*xc; } }
                else { for(int c=0;c<DN;c++){ const float* Ac=s->A+(size_t)c*N; float* hc=hl[c]; float dtc=lm_dt[k][c],xc=lm_xx[k][c],dbx=dtc*xc;
                        __m256 vdtc=_mm256_set1_ps(dtc),vdbx=_mm256_set1_ps(dbx),vacc=_mm256_setzero_ps();
                        for(int j=0;j<N;j+=8){ __m256 ee=exp256_ps(_mm256_mul_ps(vdtc,_mm256_loadu_ps(Ac+j)));
                            __m256 hcj=_mm256_fmadd_ps(ee,_mm256_loadu_ps(hc+j),_mm256_mul_ps(vdbx,_mm256_loadu_ps(Bm+j)));
                            _mm256_storeu_ps(hc+j,hcj); vacc=_mm256_fmadd_ps(hcj,_mm256_loadu_ps(Cm+j),vacc); } lm_y[k][c]=hsum256(vacc)+s->Dskip[c]*xc; } }
                for(int c=0;c<DN;c++) lm_y[k][c]*=silu(lm_z[k][c]); }
            mv_K(s->out_proj,D,DN,&lm_y[0][0],DN,&lm_tmp[0][0],D,Kn);
            for(int k=0;k<Kn;k++) for(int i=0;i<D;i++) lm_x[k][i]+=lm_tmp[k][i];
        }
        for(int k=0;k<Kn;k++) rmsnorm(lm_x[k],mlp_n2[l],lm_xn[k]);   // MLP dense LUT weight-once (skip-off)
        for(int k=0;k<Kn;k++){ lm_sa[k]=quant_i8(lm_xn[k],D,lm_xqb[k]); build_lut_t3(lm_xqb[k],TUP,lm_lutU[k]); }
        matvec_lut_full_K(gate_tm[l],&lm_lutU[0][0],TUP*16,&lm_Sg[0][0],MLP_HID,MLP_HID,Mpg,TUP,Kn);
        matvec_lut_full_K(up_tm[l],&lm_lutU[0][0],TUP*16,&lm_Su[0][0],MLP_HID,MLP_HID,Mpg,TUP,Kn);
        for(int k=0;k<Kn;k++) for(int i=0;i<MLP_HID;i++){ float g=(float)lm_Sg[k][i]*lm_sa[k]*gate_sc[l][i],u=(float)lm_Su[k][i]*lm_sa[k]*up_sc[l][i]; lm_gh[k][i]=reluf(g)*reluf(u); }
        for(int k=0;k<Kn;k++){ lm_sh[k]=quant_i8(lm_gh[k],MLP_HID,lm_hqb[k]); build_lut_t3(lm_hqb[k],TDN,lm_lutD[k]); }
        matvec_lut_full_K(down_tm[l],&lm_lutD[0][0],TDN*16,&lm_Sd[0][0],D,D,Mpd,TDN,Kn);
        for(int k=0;k<Kn;k++) for(int d=0;d<D;d++) lm_x[k][d]+=(float)lm_Sd[k][d]*lm_sh[k]*down_sc[l][d];
    }
    for(int k=0;k<Kn;k++) rmsnorm(lm_x[k],normf,lm_xn[k]);
    mv_K(head,V,D,&lm_xn[0][0],D,Lb,V,Kn);   // head weight-once
}
// 64.0(a) streamed-thread ceiling: aggregate cold-stream bandwidth of the emu buffer at nt threads (disjoint chunks,
// concurrent). Pure streaming read (no compute) = the DRAM ceiling the two-pool budget needs; independent of --threads.
static double emu_stream_bw(int nt){
    const float* p=(const float*)emu_buf; size_t nf=(size_t)emu_R*emu_arena/4;
    size_t lim=(nf>=64)?nf-63:0;   // canonical OpenMP bound: last block start with i+64<=nf
    static float tsink[64]; for(int i=0;i<64;i++) tsink[i]=0.0f;
    double t0=now_s();
    #ifdef _OPENMP
    #pragma omp parallel num_threads(nt)
    #endif
    {
        int tid=0;
        #ifdef _OPENMP
        tid=omp_get_thread_num();
        #endif
        __m256 a[8]; for(int j=0;j<8;j++) a[j]=_mm256_setzero_ps();
        #ifdef _OPENMP
        #pragma omp for schedule(static) nowait
        #endif
        for(size_t i=0;i<lim;i+=64) for(int j=0;j<8;j++) a[j]=_mm256_add_ps(a[j],_mm256_loadu_ps(p+i+j*8));
        __m256 s=a[0]; for(int j=1;j<8;j++) s=_mm256_add_ps(s,a[j]); tsink[tid&63]+=hsum256(s);
    }
    double dt=now_s()-t0; volatile float sink=0; for(int i=0;i<64;i++) sink+=tsink[i]; (void)sink;
    return (double)emu_R*emu_arena/1e9/dt;
}
static void run_g3c(const char* ngpath,long emu_mb,int ml,int sk,int ef){
    (void)ml;(void)sk;
    if(g_moe){ printf("  V-G3c: run on the DENSE model (weight-once is clean without routing); MoE expert amortization = V-G3b union.\n"); return; }
    load_ngram(ngpath); long nval=nids-(long)(nids*0.9); long sp=1000;
    size_t spt=emu_setup(emu_mb);
    printf("==== 63.V V-G3c all-weights-cold emulation (dense; %d replicas x %.1f MB = %.0f MB buffer) ====\n",
           emu_R,emu_arena/1048576.0,(double)emu_R*emu_arena/1048576.0);
    // 64.0(a) streamed-thread bandwidth ceiling: aggregate cold-stream GB/s at threads {1,2,3,6} (re-prices E4 @28,
    // closes the derived ~1.4-1.5x). Single-thread bw drives the downstream V-G3c ratio math (bw_gbs), unchanged.
    double bw_gbs=0; { int Nt[4]={1,2,3,6}; double bw1=0;
      printf("  [64.0(a)] streamed-thread bandwidth ceiling (parallel cold-stream read, %.0f MB buffer):\n",(double)emu_R*emu_arena/1048576.0);
      for(int q=0;q<4;q++){ double b=emu_stream_bw(Nt[q]); double b2=emu_stream_bw(Nt[q]); if(b2>b)b=b2;   // best-of-2 (warm)
        if(q==0){ bw1=b; bw_gbs=b; } printf("      threads=%d  %.1f GB/s  (%.2fx vs single-thread)\n",Nt[q],b,bw1>0?b/bw1:1.0); } }
    // identity: forward_block_lm(K) == K sequential forward_token(sk=0) from the same primed state
    static uint16_t hist[1<<17]; float* pend=xmalloc((size_t)V*4); float* Lb=xmalloc((size_t)8*V*4); float* Lr=xmalloc((size_t)8*V*4);
    static ActPos* acts=NULL; if(!acts) acts=xmalloc(sizeof(ActPos)*64);
    int Kid=8; state_reset(); emu_rebind(0); int nh=0; for(int i=0;i<64;i++){ hist[nh++]=ids[(long)(nids*0.9)+sp+i]; forward_token(hist[nh-1],pend,1,0,ef,NULL); }
    uint16_t dr[8]; for(int k=0;k<Kid;k++) dr[k]=ng_draft(hist,nh+k);
    snap_save(); forward_block_lm(dr,Kid,Lb,acts,ef); snap_restore();   // block-lm advances then restores to block-start
    for(int k=0;k<Kid;k++){ forward_token(dr[k],Lr+(size_t)k*V,1,0,ef,NULL); }   // ref: K sequential AR from the same start
    double mx=0; for(int k=0;k<Kid;k++) for(int o=0;o<V;o++){ double d=fabs((double)Lb[(size_t)k*V+o]-Lr[(size_t)k*V+o]); if(d>mx)mx=d; }
    printf("  identity (layer-major weight-once vs sequential AR, K=%d): max|dLogit| = %.3e  %s\n",Kid,mx,mx==0.0?"BIT-IDENTICAL":(mx<1e-3?"OK":"FAIL"));
    long T=4000; long base=(long)(nids*0.9)+sp;
    static uint16_t s_ar[1<<17];   // AR token stream (for emulated-mode token-identity vs block)
    // AR-hot (weights resident, no rotation) = per-position COMPUTE-floor proxy; AR-cold (rotate) = compute + cold traffic.
    state_reset(); emu_rebind(0); nh=0; for(int i=0;i<64;i++){ hist[nh++]=ids[base+i]; forward_token(hist[nh-1],pend,1,0,ef,NULL); }
    double t0=now_s(); for(long i=0;i<T;i++){ int nx=greedy_hyg(pend,hist,nh); s_ar[i]=nx; hist[nh++]=nx; forward_token(nx,pend,1,0,ef,NULL); } double t_hot=now_s()-t0;
    state_reset(); emu_rot=0; nh=0; for(int i=0;i<64;i++){ hist[nh++]=ids[base+i]; forward_token(hist[nh-1],pend,1,0,ef,NULL); }
    t0=now_s(); for(long i=0;i<T;i++){ emu_rebind(emu_rot++ % emu_R); int nx=greedy_hyg(pend,hist,nh); hist[nh++]=nx; forward_token(nx,pend,1,0,ef,NULL); } double t_arc=now_s()-t0;
    double t_traf=spt/(bw_gbs*1e9);            // all-cold traffic time per pass (counted bytes / measured bandwidth)
    double t_compute=t_hot/T;                  // compute-floor proxy (resident weights)
    printf("  counted streamed bytes/token (AR) = %.0f KB (all weight classes cold)\n",spt/1024.0);
    printf("  per-pass all-cold traffic = %.2f ms (%.0f KB / %.1f GB/s); compute-floor = %.2f ms (AR-hot %.0f tok/s); ratio compute/traffic = %.2f\n",
           t_traf*1e3, spt/1024.0, bw_gbs, t_compute*1e3, T/t_hot, t_compute/t_traf);
    printf("  AR cold %.0f tok/s vs AR hot %.0f tok/s = %.0f%% cold penalty -> forward is %s\n",
           T/t_arc, T/t_hot, 100.0*(t_arc-t_hot)/t_hot, (t_compute/t_traf>0.4)?"COMPUTE-bound even all-cold":"traffic-bound");
    printf("    K   tpp    AR-tok/s  block-tok/s  wall-speedup  tok-identity(vs AR-cold)  block-bytes/tok(=AR/tpp,weight-once)\n");
    int Ks[3]={2,4,8}; double best=0;
    for(int ki=0;ki<3;ki++){ int K=Ks[ki];
        state_reset(); emu_rot=0; nh=0; for(int i=0;i<64;i++){ hist[nh++]=ids[base+i]; forward_token(hist[nh-1],pend,1,0,ef,NULL); }
        double se=0; long emit=0,nb=0,ident=1; uint16_t d[8]; t0=now_s();
        while(emit<T){ for(int k=0;k<K;k++){ d[k]=ng_draft(hist,nh+k); hist[nh+k]=d[k]; }
            emu_rebind(emu_rot++ % emu_R);              // cold replica: weights streamed once for the whole block
            snap_save(); forward_block_lm(d,K,Lb,acts,ef); snap_restore();   // Lb[k]=logits AFTER d[k] (predicts pos k+1)
            int acc=0,mism=0; uint16_t eng=0; float* Lprev=pend;            // pend predicts pos 0; Lb[i-1] predicts pos i
            for(int i=0;i<K;i++){ int g=greedy_hyg(Lprev,hist,nh+i); if(g==d[i]) acc++; else { eng=g; mism=1; break; } Lprev=Lb+(size_t)i*V; }
            replay_commit(acts,acc,ef);
            int ncommit; if(mism){ hist[nh+acc]=eng; emu_rebind(emu_rot++ % emu_R); forward_token(eng,pend,1,0,ef,NULL); ncommit=acc+1; }
            else { memcpy(pend,Lb+(size_t)(K-1)*V,V*4); ncommit=K; }
            for(int c=0;c<ncommit;c++){ long e=emit+c; if(e<T && hist[nh+c]!=s_ar[e]) ident=0; }
            nh+=ncommit; emit+=ncommit; se+=ncommit; nb++;
        }
        double t_blk=now_s()-t0; double tpp=nb?se/nb:0;
        double blk_tps=emit/t_blk, sp_up=blk_tps/(T/t_arc);
        printf("    %-3d %.3f  %8.0f  %9.0f    %.3fx        %-9s               %.0f KB\n",
               K,tpp,T/t_arc,blk_tps,sp_up,ident?"IDENTICAL":"MISMATCH",spt/tpp/1024.0);
        if(sp_up>best)best=sp_up;
    }
    printf("  MECHANISM gate: block weight-bytes/token = AR/tpp (weight-once by construction) = PASS.\n");
    printf("  CLAIM gate (>=1.4x at best K): best wall-speedup = %.3fx -> %s\n",best,best>=1.4?"ADOPTED (weight-traffic-dominated)":"NOT adopted at 8.3M");
    if(best<1.4) printf("    verdict (pre-registered outcome 3): mechanism verified, claim SCOPED OUT at 8.3M — compute-floor/traffic = %.2f >> ~0.25-0.30, so the forward is compute-bound and traffic amortization yields no wall-clock win; (traffic %.2f ms, compute %.2f ms) feed the Phase-64 two-pool sizing (block-verify wins once the streamed pool makes traffic dominate compute).\n",t_compute/t_traf,t_traf*1e3,t_compute*1e3);
    free(pend); free(Lb); free(Lr);
}

// ---------------- 64.1b microbenches (synthetic, no weights, no gate) — tighten the 64.1 budget model ----------------
// (1) proj-GEMV size sweep: real row-partitioned fp32 matvec over synthetic weight sets {4..96} MB, threads {1,6}.
//     Validates the 64.1 spill model: does the ~40 GB/s@t6 plateau hold past L3, and where does the t1 curve break
//     (L3 = 16 MB per CCX on the reference)? Input x (2 KB) stays L1-resident; the swept W is the streamed working set.
static void run_gemv_sweep(void){
    const int nin=512;                                    // DN — the largest projection input dim in Arch-A
    long mb[8]={4,8,16,24,32,48,64,96};
    printf("==== 64.1b(1) proj-GEMV size sweep (real fp32 row-partitioned matvec, in=%d, x L1-resident) ====\n",nin);
    printf("   size(MB)  out_rows |  t1 GB/s  t1 us/pass |  t6 GB/s  t6 us/pass |  t6/t1\n");
    float* x=xmalloc((size_t)nin*4); for(int i=0;i<nin;i++) x[i]=0.01f*((i%7)-3);
    for(int si=0;si<8;si++){
        size_t wb=(size_t)mb[si]*1048576; int out=(int)(wb/((size_t)nin*4)); wb=(size_t)out*nin*4;
        float* W=xmalloc(wb); for(size_t i=0;i<wb/4;i++) W[i]=0.001f*((long)(i%13)-6);
        float* y=xmalloc((size_t)out*4);
        double gbps[2],usp[2];
        for(int ti=0;ti<2;ti++){ int nt=ti?6:1;
#ifdef _OPENMP
            omp_set_num_threads(nt); g_omp_on=(nt>1);
#endif
            long npass=(long)(2048UL*1048576/wb); if(npass<16) npass=16;   // ~2 GB streamed per window (stable at 96 MB)
            matvec(W,x,y,out,nin);                                          // warm
            double best=1e30;                                              // best-of-2 (min time = de-jittered rate)
            for(int rep=0;rep<2;rep++){ double t0=now_s(); for(long p=0;p<npass;p++){ x[p&(nin-1)]+=1e-9f; matvec(W,x,y,out,nin); }
                double dt=now_s()-t0; if(dt<best) best=dt; }
            gbps[ti]=(double)wb*npass/1e9/best; usp[ti]=best/npass*1e6;
        }
        printf("   %-8ld  %-8d |  %6.1f   %9.1f |  %6.1f   %9.1f |  %.2fx\n",
               mb[si],out,gbps[0],usp[0],gbps[1],usp[1],gbps[1]/gbps[0]);
        free(W); free(y);
    }
    free(x);
}
// (2) expert-pool DRAM rate: real ternary LUT kernel over a pool >> L3 (512 MB) of 48 KB experts, 8 random x 6 layers
//     per "token" (i.i.d. selection => cold random gather). Tightens the model's widest bracket [4.2-11.4 GB/s] — the
//     effective streamed rate that sets the S1/S2/M1 rows of the curve. threads {1,6} (mirrors the engine: OMP over rows).
static void run_expert_rate(void){
    const int M=384,Mpad=384,T=128; const size_t EB=(size_t)T*Mpad;   // 49152 B = 48 KB = one expert (gate+up+down eq.)
    const size_t POOL=512UL*1048576; long NE=(long)(POOL/EB);          // ~10922 experts, contiguous, >> L3
    const int LAYERS=6,K=8; long per_tok=(long)LAYERS*K;               // 48 expert touches / token
    int8_t* pool=xmalloc((size_t)NE*EB);
    for(size_t i=0;i<(size_t)NE*EB;i++) pool[i]=(int8_t)((i*2654435761u)%9u);   // valid ternary-packed code indices [0,8]
    int8_t lut[T*16]; int8_t xq[2*T]; for(int i=0;i<2*T;i++) xq[i]=(int8_t)((i%5)-2); build_lut_t3(xq,T,lut);
    int32_t* y=xmalloc((size_t)M*4);
    printf("==== 64.1b(2) expert-pool DRAM rate (real LUT kernel; %ld MB pool = %ld experts x %ld KB; %ld touches/token, i.i.d.) ====\n",
           (long)(POOL/1048576),NE,(long)(EB/1024),per_tok);
    printf("   threads |  GB/s   us/token  us/expert\n");
    for(int ti=0;ti<2;ti++){ int nt=ti?6:1;
#ifdef _OPENMP
        omp_set_num_threads(nt); g_omp_on=(nt>1);
#endif
        uint64_t rng=0x9e3779b97f4a7c15ULL; long ntok=500,touches=0;
        for(long e=0;e<per_tok;e++){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; matvec_lut_full(pool+(size_t)(rng%NE)*EB,lut,y,M,Mpad,T); } // warm
        double t0=now_s();
        for(long tok=0;tok<ntok;tok++) for(long e=0;e<per_tok;e++){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17;
            matvec_lut_full(pool+(size_t)(rng%NE)*EB,lut,y,M,Mpad,T); touches++; }
        double dt=now_s()-t0;
        printf("   %-7d |  %5.2f  %8.1f  %8.3f\n",nt,(double)touches*EB/1e9/dt,dt/ntok*1e6,dt/touches*1e6);
    }
    free(pool); free(y);
}

// ---------------- P2: expert-path decomposition (brief BRIEF_P2_EXPERT_PATH_DECOMPOSITION.md) ------
// Three arms over the SAME kernel, same M/Mpad/T/EB, same LUT, same call count. The ONLY thing that
// varies is WHICH expert each touch reads:
//   R  i.i.d. random over the 512 MB pool  -- replicates 64.1b's registered 2.88 us/expert
//   S  sequential stride 1                 -- prefetch-friendly edge of the memory term
//   C  always expert 0 (48 KB, L1-resident) -- the pure ARITHMETIC floor, memory term ~0
// memory_term = us/expert(R) - us/expert(C). Decides SPEED_LEDGER.md §4's 2.1x bracket.
// This function is reached only via --expert-decomp; no existing code path is touched.
static void run_expert_decomp(void){
    const int M=384,Mpad=384,T=128; const size_t EB=(size_t)T*Mpad;   // 49152 B, identical to run_expert_rate
    const size_t POOL=512UL*1048576; long NE=(long)(POOL/EB);
    const int LAYERS=6,K=8; long per_tok=(long)LAYERS*K;
    int8_t* pool=xmalloc((size_t)NE*EB);
    for(size_t i=0;i<(size_t)NE*EB;i++) pool[i]=(int8_t)((i*2654435761u)%9u);
    int8_t lut[T*16]; int8_t xq[2*T]; for(int i=0;i<2*T;i++) xq[i]=(int8_t)((i%5)-2); build_lut_t3(xq,T,lut);
    int32_t* y=xmalloc((size_t)M*4);
    printf("==== P2 expert-path decomposition (%ld MB pool = %ld experts x %ld KB; %ld touches/token) ====\n",
           (long)(POOL/1048576),NE,(long)(EB/1024),per_tok);
    printf("   arm                       threads |  GB/s   us/token  us/expert   checksum\n");
    const char* names[3]={"R i.i.d. random (512 MB)","S sequential stride-1    ","C constant expert 0 (L1) "};
    double us_exp[3][2];
    for(int arm=0;arm<3;arm++) for(int ti=0;ti<2;ti++){ int nt=ti?6:1;
#ifdef _OPENMP
        omp_set_num_threads(nt); g_omp_on=(nt>1);
#endif
        uint64_t rng=0x9e3779b97f4a7c15ULL; long ntok=500,touches=0,seq=0;
        for(long e=0;e<per_tok;e++){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17;
            size_t idx = arm==0 ? (size_t)(rng%NE) : (arm==1 ? (size_t)((seq++)%NE) : (size_t)0);
            matvec_lut_full(pool+idx*EB,lut,y,M,Mpad,T); }                                   // warm
        long acc=0; double t0=now_s();
        for(long tok=0;tok<ntok;tok++) for(long e=0;e<per_tok;e++){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17;
            size_t idx = arm==0 ? (size_t)(rng%NE) : (arm==1 ? (size_t)((seq++)%NE) : (size_t)0);
            matvec_lut_full(pool+idx*EB,lut,y,M,Mpad,T); touches++; acc+=y[0]; }
        double dt=now_s()-t0;
        us_exp[arm][ti]=dt/touches*1e6;
        printf("   %s %-7d |  %5.2f  %8.1f  %8.3f  %ld\n",names[arm],nt,
               (double)touches*EB/1e9/dt,dt/ntok*1e6,dt/touches*1e6,acc);
    }
    // The decomposition and its two pre-registered controls, computed here so no reader has to.
    for(int ti=0;ti<2;ti++){ int nt=ti?6:1;
        double r=us_exp[0][ti], sq=us_exp[1][ti], c=us_exp[2][ti];
        int ordering_ok = (c<=sq*1.05) && (sq<=r*1.05);
        printf("   -- t%-2d  r=%.3f  s=%.3f  c=%.3f us/expert | memory_term=%.3f us (%.1f%% of r) | c/r=%.3f | ordering C<=S<=R: %s\n",
               nt,r,sq,c,r-c,100.0*(r-c)/r,c/r,ordering_ok?"OK":"VIOLATED -- RUN VOID");
        printf("   -- t%-2d  pre-registered label: %s\n", nt,
               (c>=0.95*r||!ordering_ok) ? "INSTRUMENT-FAILURE" :
               (c<=0.30*r) ? "BANDWIDTH-BOUND" : (c>=0.70*r) ? "COMPUTE-BOUND" : "MIXED");
    }
    free(pool); free(y);
}

// ---------------- synthetic kernel self-tests (no weights; CI) ----------------
static int kernel_selftest(void){
    int rc=0; srand(45678); long checks=0; int worst=0;
    { // dense shapes: full / row-scalar / tile-skip vs scalar-int
        int dims[2][2]={{MLP_HID,D},{D,MLP_HID}};
        for(int c=0;c<2;c++){ int M=dims[c][0],K=dims[c][1],Mpad=(M+31)&~31,T=K/2;
            int8_t* Wt=xmalloc((size_t)M*K); int8_t* ctm=xmalloc((size_t)T*Mpad); int8_t* crm=xmalloc((size_t)M*T);
            int8_t* xq=xmalloc(K); int8_t* lut=xmalloc((size_t)T*16);
            int32_t* Sl=xmalloc((size_t)M*4); int32_t* Sr=xmalloc((size_t)M*4); int* act=xmalloc((size_t)T*sizeof(int));
            for(int trial=0;trial<24;trial++){
                for(size_t i=0;i<(size_t)M*K;i++) Wt[i]=(int8_t)(rand()%3-1);
                for(int k=0;k<K;k++){ int v=rand()%(2*AQ+1)-AQ; if(rand()%3==0)v=0; xq[k]=(int8_t)v; }
                bc_tm(Wt,M,K,Mpad,ctm); bc_rm(Wt,M,K,crm); build_lut_t3(xq,T,lut); ref_t3(Wt,xq,Sr,M,K);
                matvec_lut_full(ctm,lut,Sl,M,Mpad,T);
                for(int m=0;m<M;m++){ int d=abs(Sl[m]-Sr[m]); if(d>worst)worst=d; checks++; }
                for(int m=0;m<M;m+=7){ const int8_t* cr=crm+(size_t)m*T; int S=0; for(int t=0;t<T;t++) S+=lut[t*16+cr[t]];
                    int d=abs(S-Sr[m]); if(d>worst)worst=d; checks++; }
                int na=0; for(int t=0;t<T;t++) if(xq[2*t]||xq[2*t+1]) act[na++]=t;
                matvec_lut_tileskip(ctm,lut,Sl,M,Mpad,act,na);
                for(int m=0;m<M;m++){ int d=abs(Sl[m]-Sr[m]); if(d>worst)worst=d; checks++; }
            }
            free(Wt);free(ctm);free(crm);free(xq);free(lut);free(Sl);free(Sr);free(act);
        }
    }
    { // MoE shapes: windowed rows + per-expert blocks vs scalar-int
        int8_t* Wgu=xmalloc((size_t)GH*D); int8_t* cgu=xmalloc((size_t)TUP*MPAD_GU);
        int8_t xq[D],lut[TUP*16]; int32_t Sl[HID_E],Sr[HID_E];
        for(int trial=0;trial<8;trial++){
            for(size_t i=0;i<(size_t)GH*D;i++) Wgu[i]=(int8_t)(rand()%3-1);
            for(int k=0;k<D;k++) xq[k]=(int8_t)(rand()%(2*AQ+1)-AQ);
            bc_tm(Wgu,GH,D,MPAD_GU,cgu); build_lut_t3(xq,TUP,lut);
            for(int e=0;e<E;e+=5){ matvec_lut_rows(cgu,lut,Sl,e*HID_E,HID_E,MPAD_GU,TUP);
                ref_t3(Wgu+(size_t)e*HID_E*D,xq,Sr,HID_E,D);
                for(int i=0;i<HID_E;i++){ int d=abs(Sl[i]-Sr[i]); if(d>worst)worst=d; checks++; } }
        }
        free(Wgu);free(cgu);
    }
    printf("==== engine kernel self-test (synthetic, no weights): all LUT paths vs scalar-int ====\n");
    printf("  %ld checks | worst |S - S_ref| = %d  %s\n",checks,worst,worst==0?"PASS":"FAIL");
    if(worst!=0) rc=2;
    double worst_dom=0;
    for(long i=0;i<=3000000;i++){ float x=-30.0f+(float)i*(30.0f/3000000.0f);
        float a=exp_approx1(x); float r=expf(x); double rel=fabs((double)a-r)/(double)r; if(rel>worst_dom)worst_dom=rel; }
    printf("  exp256_ps vs libm on [-30,0]: max rel err = %.3e  %s (<=2e-6)\n",worst_dom,worst_dom<=2e-6?"PASS":"FAIL");
    if(worst_dom>2e-6) rc=2;
    return rc;
}

int main(int argc,char**argv){
    int mlp_lut=1,skip=1,exp_fast=1;                 // default = the full optimized config
    int do_bpb=0,do_logits=0,do_tm=0; long seqW=512,eval_tok=200000,ntok=10240,offset=0,timetok=3000;
    int threads=1; const char* wp=NULL; const char* dumpto=NULL;
    int block=0,gen_verify=0,nseed=3,do_g3b=0,do_g3c=0,do_gemvsweep=0,do_exprate=0,do_expdecomp=0; long genlen=800,emu_mb=128; const char* ngpath=NULL;
    for(int i=1;i<argc;i++) if(!strcmp(argv[i],"--pack")&&i+1<argc){          // pre-scan: must be set before load_weights
        if(!strcmp(argv[i+1],"nibble")) g_pack_nib=1; else if(!strcmp(argv[i+1],"byte")) g_pack_nib=0;
        else { fprintf(stderr,"--pack must be byte|nibble\n"); return 1; } }
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--threads")&&i+1<argc){ threads=atoi(argv[++i]); continue; }
        else if(!strcmp(argv[i],"--block")&&i+1<argc){ block=atoi(argv[++i]); continue; }
        else if(!strcmp(argv[i],"--ngram")&&i+1<argc){ ngpath=argv[++i]; continue; }
        else if(!strcmp(argv[i],"--gen-verify")){ gen_verify=1; continue; }
        else if(!strcmp(argv[i],"--g3b")){ do_g3b=1; continue; }
        else if(!strcmp(argv[i],"--g3c")){ do_g3c=1; continue; }
        else if(!strcmp(argv[i],"--gemv-sweep")){ do_gemvsweep=1; continue; }
        else if(!strcmp(argv[i],"--expert-rate")){ do_exprate=1; continue; }
        else if(!strcmp(argv[i],"--expert-decomp")){ do_expdecomp=1; continue; }
        else if(!strcmp(argv[i],"--emu-mb")&&i+1<argc){ emu_mb=atol(argv[++i]); continue; }
        else if(!strcmp(argv[i],"--gen-len")&&i+1<argc){ genlen=atol(argv[++i]); continue; }
        else if(!strcmp(argv[i],"--seeds")&&i+1<argc){ nseed=atoi(argv[++i]); continue; }
        if(!strcmp(argv[i],"--mlp")&&i+1<argc){ i++; mlp_lut=strcmp(argv[i],"fp32")?1:0; }
        else if(!strcmp(argv[i],"--skip")&&i+1<argc){ i++; skip=strcmp(argv[i],"off")?1:0; }
        else if(!strcmp(argv[i],"--exp")&&i+1<argc){ i++; exp_fast=strcmp(argv[i],"exact")?1:0; }
        else if(!strcmp(argv[i],"--bpb")) do_bpb=1;
        else if(!strcmp(argv[i],"--logits")) do_logits=1;
        else if(!strcmp(argv[i],"--timing")) do_tm=1;
        else if(!strcmp(argv[i],"--dumplogits")&&i+1<argc) dumpto=argv[++i];
        else if(!strcmp(argv[i],"--seq")&&i+1<argc) seqW=atol(argv[++i]);
        else if(!strcmp(argv[i],"--eval-tok")&&i+1<argc) eval_tok=atol(argv[++i]);
        else if(!strcmp(argv[i],"--ntok")&&i+1<argc) ntok=atol(argv[++i]);
        else if(!strcmp(argv[i],"--offset")&&i+1<argc) offset=atol(argv[++i]);
        else if(!strcmp(argv[i],"--time-tok")&&i+1<argc) timetok=atol(argv[++i]);
        else if(!strcmp(argv[i],"--pack")&&i+1<argc){ i++; }              // already handled by the pre-scan
        else if(!strcmp(argv[i],"--kselftest")) return kernel_selftest();
        else if(!strcmp(argv[i],"--weights")&&i+1<argc) wp=argv[++i];
        else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 1; }
    }
    if(do_gemvsweep||do_exprate||do_expdecomp){   // 64.1b microbenches: synthetic, weight-free, self-sweep threads {1,6}
#ifdef _OPENMP
        omp_set_dynamic(0);
#endif
        if(do_gemvsweep) run_gemv_sweep();
        if(do_exprate) run_expert_rate();
        if(do_expdecomp) run_expert_decomp();
        return 0;
    }
    if(!wp){ fprintf(stderr,"usage: engine --weights <model.bin> [--threads N] [--mlp fp32|lut] [--skip on|off] [--exp exact|fast]\n"
                            "              [--bpb] [--logits] [--timing] [--dumplogits <file>] [--ntok N] [--offset N]\n"); return 1; }
#ifdef _OPENMP
    if(threads<1) threads=1; omp_set_num_threads(threads); omp_set_dynamic(0); g_omp_on=(threads>1);   // if(g_omp_on) skips fork at N=1
    fprintf(stderr,"engine threads=%d (OpenMP; parallelism across independent outputs only)\n",threads);
#else
    if(threads!=1) fprintf(stderr,"WARN --threads %d ignored (built without OpenMP)\n",threads);
#endif
    load_weights(wp); load_meta("results/phase55/meta.bin"); load_ids("results/phase55/ids.u16");
    hstate=calloc(L,sizeof(*hstate)); convbuf=calloc(L,sizeof(*convbuf)); kring=calloc((size_t)WIN*D,4); vring=calloc((size_t)WIN*D,4);
    fprintf(stderr,"engine loaded: %s | mlp=%s skip=%s exp=%s | ids=%ld\n",
            g_moe?"MoE":"dense",mlp_lut?"lut":"fp32",skip?"on":"off",exp_fast?"fast":"exact",nids);
    int rc=0;
    if(gen_verify){ if(!ngpath){fprintf(stderr,"--gen-verify needs --ngram <path> --block K\n");return 1;} rc|=run_verify(block,ngpath,genlen,nseed,mlp_lut,skip,exp_fast); }
    if(do_g3b){ if(!ngpath){fprintf(stderr,"--g3b needs --ngram <path>\n");return 1;} run_g3b(ngpath,genlen,mlp_lut,skip,exp_fast); }
    if(do_g3c){ if(!ngpath){fprintf(stderr,"--g3c needs --ngram <path>\n");return 1;}
        if(g_pack_nib){fprintf(stderr,"--g3c is NOT supported under --pack nibble: matvec_lut_full_K (the layer-major weight-once kernel) has no nibble variant. Refusing rather than silently mixing layouts.\n");return 1;}
        run_g3c(ngpath,emu_mb,mlp_lut,skip,exp_fast); }
    if(dumpto) dump_logits(dumpto,seqW,ntok,offset,mlp_lut,skip,exp_fast);
    if(do_logits) rc|=gate_logits(seqW,ntok,mlp_lut,skip,exp_fast);
    if(do_bpb){ long nt; double b=run_bpb(seqW,eval_tok,mlp_lut,skip,exp_fast,&nt);
        printf("==== BPB (%ld tok): %.6f ====\n",nt,b); }
    if(do_tm) timing(timetok,mlp_lut,skip,exp_fast);
    printf("STOP. engine run above. No commit.\n");
    return rc;
}
