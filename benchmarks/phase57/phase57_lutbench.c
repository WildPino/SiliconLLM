// Phase 57 / Probe-1 (1b) - VELOCITA: pshufb-LUT ternary/low-bit matvec vs fp32 on Zen2 (Ryzen 5 3600X, AVX2, NO VNNI).
//   magic: "LUT1" 0x4C555431
//
// Finding-1 to de-risk: on OUR Zen2 (no int8-VNNI) the right low-bit kernel is a pshufb byte-LUT, NOT dequant-then-int8.
//   Two signatures the Capo reads:
//     (A) LUT beats fp32 at MLP dims (anchor: bitnet.cpp 2.37-6.17x on x86).
//     (B) LUT gets FASTER at FEWER bits (the property that distinguishes it from dequant-int8, which stalls <4-bit).
//
//   Kernels (all matvec y[M] = W[M,K] . x[K], the batch-1 weight-stream the deliverable does token-by-token):
//     - f32      : AVX2 FMA reference (the thing to beat).
//     - lut_bin  : bit-serial BINARY pshufb-LUT, nbits in {4,2,1}  -> measures cost-vs-bits curve (signature B).
//                  g=4 activations/tile -> 16-entry int8 LUT (subset sums), pshufb does 32 output-rows/instr.
//     - lut_t3   : TERNARY {-1,0,+1} pshufb-LUT, g=2 base-3 -> 9-entry int8 LUT, single pass (the real 1.58 kernel).
//
//   Correctness: every LUT kernel returns the EXACT integer dot S[m]=sum_k wq[m,k]*xq[k]; we check it bit-exact
//   against a scalar integer reference (so the timing is on a kernel that provably computes the right thing).
//   Activations quantized to int8 in [-31,31] so a g=4 subset-sum (<=4*31) and a g=2 base-3 sum (<=2*31) fit int8 -> pshufb-direct.
//
//   This validates the MECHANISM on the target silicon. It is NOT a bandwidth claim (microbench is L1/L2-resident;
//   the streaming win is a scale-up property). No commit.
//
// Build (Zen2): clang -O3 -mavx2 -mfma -march=znver2 phase57_lutbench.c -o phase57_lutbench
// Run         : ./phase57_lutbench          (defaults: MLP dims fc1 1024x256 + fc2 256x1024, plus a 2048x2048 scale point)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#if defined(_WIN32)
#include <windows.h>
static double now_s(void){ LARGE_INTEGER f,t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t); return (double)t.QuadPart/(double)f.QuadPart; }
#else
static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
#endif

static void* amalloc(size_t n){ void* p=NULL;
#if defined(_WIN32)
    p=_aligned_malloc(n,64);
#else
    if(posix_memalign(&p,64,n)) p=NULL;
#endif
    if(!p){ fprintf(stderr,"alloc fail %zu\n",n); exit(1);} memset(p,0,n); return p; }
static void afree(void* p){
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

// ---------------- fp32 reference matvec (AVX2 FMA) ----------------
static void matvec_f32(const float* W, const float* x, float* y, int M, int K){
    for(int m=0;m<M;m++){
        const float* w=W+(size_t)m*K;
        __m256 acc=_mm256_setzero_ps();
        int k=0;
        for(;k+8<=K;k+=8) acc=_mm256_fmadd_ps(_mm256_loadu_ps(w+k),_mm256_loadu_ps(x+k),acc);
        __m128 lo=_mm256_castps256_ps128(acc), hi=_mm256_extractf128_ps(acc,1);
        lo=_mm_add_ps(lo,hi); lo=_mm_hadd_ps(lo,lo); lo=_mm_hadd_ps(lo,lo);
        float s=_mm_cvtss_f32(lo);
        for(;k<K;k++) s+=w[k]*x[k];
        y[m]=s;
    }
}

// ---------------- widen 32 int8 -> 4x int32, scale by (1<<sh), add into acc[4] ----------------
static inline void acc_add_i8x32(__m256i* acc, __m256i p, int sh){
    __m128i lo=_mm256_castsi256_si128(p), hi=_mm256_extracti128_si256(p,1);
    __m256i a0=_mm256_cvtepi8_epi32(lo);
    __m256i a1=_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8));
    __m256i a2=_mm256_cvtepi8_epi32(hi);
    __m256i a3=_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8));
    if(sh){ a0=_mm256_slli_epi32(a0,sh); a1=_mm256_slli_epi32(a1,sh); a2=_mm256_slli_epi32(a2,sh); a3=_mm256_slli_epi32(a3,sh); }
    acc[0]=_mm256_add_epi32(acc[0],a0); acc[1]=_mm256_add_epi32(acc[1],a1);
    acc[2]=_mm256_add_epi32(acc[2],a2); acc[3]=_mm256_add_epi32(acc[3],a3);
}
static inline void acc_store(__m256i* acc, int32_t* y, int base, int M){
    int32_t tmp[32];
    _mm256_storeu_si256((__m256i*)(tmp+0),acc[0]); _mm256_storeu_si256((__m256i*)(tmp+8),acc[1]);
    _mm256_storeu_si256((__m256i*)(tmp+16),acc[2]); _mm256_storeu_si256((__m256i*)(tmp+24),acc[3]);
    for(int r=0;r<32 && base+r<M;r++) y[base+r]=tmp[r];
}

// ---------------- bit-serial BINARY LUT matvec (nbits planes) ----------------
//  codes[plane][tile*Mpad + row] = the 4-bit (g=4) weight nibble for that plane/tile/row.
//  lut[tile*16 + c] = sum over set bits in c of xq[tile*4 + j]  (int8, |.|<=4*31=124).
//  S[m] = sum_tile sum_plane (1<<plane) * lut[tile*16 + codes[plane][tile,m]]
static void matvec_lut_bin(const int8_t* codes, const int8_t* lut, int32_t* y, int M, int Mpad, int K, int nbits){
    int T=K/4;
    for(int base=0; base<M; base+=32){
        __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};
        for(int b=0;b<nbits;b++){
            const int8_t* cb=codes+(size_t)b*T*Mpad;
            for(int t=0;t<T;t++){
                __m256i tbl=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)t*16)));
                __m256i idx=_mm256_loadu_si256((const __m256i*)(cb+(size_t)t*Mpad+base));
                __m256i p=_mm256_shuffle_epi8(tbl,idx);
                acc_add_i8x32(acc,p,b);
            }
        }
        acc_store(acc,y,base,M);
    }
}

// ---------------- TERNARY LUT matvec (g=2 base-3, single pass) ----------------
//  codes[tile*Mpad+row] = base-3 code (w0+1)*3+(w1+1) in 0..8 for ternary pair (w0,w1) in {-1,0,1}^2.
//  lut[tile*16 + c] = w0*xq[2t] + w1*xq[2t+1] decoded from c (int8, |.|<=2*31=62).
static void matvec_lut_t3(const int8_t* codes, const int8_t* lut, int32_t* y, int M, int Mpad, int K){
    int T=K/2;
    for(int base=0; base<M; base+=32){
        __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};
        for(int t=0;t<T;t++){
            __m256i tbl=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)t*16)));
            __m256i idx=_mm256_loadu_si256((const __m256i*)(codes+(size_t)t*Mpad+base));
            __m256i p=_mm256_shuffle_epi8(tbl,idx);
            acc_add_i8x32(acc,p,0);
        }
        acc_store(acc,y,base,M);
    }
}

// ---------------- LUT builders (per activation tile) ----------------
static void build_lut_bin(const int8_t* xq, int K, int8_t* lut){   // 16-entry subset sums, g=4
    int T=K/4;
    for(int t=0;t<T;t++){
        int8_t a=xq[4*t],b=xq[4*t+1],c=xq[4*t+2],d=xq[4*t+3];
        for(int code=0;code<16;code++){
            int s=0; if(code&1)s+=a; if(code&2)s+=b; if(code&4)s+=c; if(code&8)s+=d;
            lut[t*16+code]=(int8_t)s;
        }
    }
}
static void build_lut_t3(const int8_t* xq, int K, int8_t* lut){    // 9 used entries (0..8), g=2 base-3
    int T=K/2;
    for(int t=0;t<T;t++){
        int8_t x0=xq[2*t],x1=xq[2*t+1];
        for(int c=0;c<16;c++){
            int s=0;
            if(c<9){ int w0=c/3-1, w1=c%3-1; s=w0*x0+w1*x1; }
            lut[t*16+c]=(int8_t)s;
        }
    }
}

// ---------------- weight packing ----------------
static void pack_bin(const uint8_t* Wb, int M, int Mpad, int K, int nbits, int8_t* codes){
    int T=K/4;
    for(int b=0;b<nbits;b++){
        int8_t* cb=codes+(size_t)b*T*Mpad;
        for(int t=0;t<T;t++) for(int m=0;m<M;m++){
            int nib=0;
            for(int j=0;j<4;j++){ int wv=Wb[(size_t)m*K+4*t+j]; if((wv>>b)&1) nib|=(1<<j); }
            cb[(size_t)t*Mpad+m]=(int8_t)nib;
        }
        for(int t=0;t<T;t++) for(int m=M;m<Mpad;m++) cb[(size_t)t*Mpad+m]=0;
    }
}
static void pack_t3(const int8_t* Wt, int M, int Mpad, int K, int8_t* codes){  // Wt in {-1,0,1}
    int T=K/2;
    for(int t=0;t<T;t++){
        for(int m=0;m<M;m++){ int w0=Wt[(size_t)m*K+2*t], w1=Wt[(size_t)m*K+2*t+1]; codes[(size_t)t*Mpad+m]=(int8_t)((w0+1)*3+(w1+1)); }
        for(int m=M;m<Mpad;m++) codes[(size_t)t*Mpad+m]=0;
    }
}

// ---------------- scalar integer references (exact correctness gate) ----------------
static void ref_bin(const uint8_t* Wb, const int8_t* xq, int32_t* y, int M, int K){
    for(int m=0;m<M;m++){ long s=0; for(int k=0;k<K;k++) s+=(long)Wb[(size_t)m*K+k]*xq[k]; y[m]=(int32_t)s; }
}
static void ref_t3(const int8_t* Wt, const int8_t* xq, int32_t* y, int M, int K){
    for(int m=0;m<M;m++){ long s=0; for(int k=0;k<K;k++) s+=(long)Wt[(size_t)m*K+k]*xq[k]; y[m]=(int32_t)s; }
}
static int cmp_i32(const int32_t* a, const int32_t* b, int M){ for(int i=0;i<M;i++) if(a[i]!=b[i]) return i; return -1; }

static double bench(double (*fn)(void*), void* ctx, int reps, int* iters_out){
    for(int i=0;i<3;i++) fn(ctx);                 // warm caches
    double t0=now_s();
    for(int i=0;i<reps;i++){ fn(ctx); __asm__ __volatile__("" : : "r"(ctx) : "memory"); }  // barrier: defeat dead-store elision
    double dt=(now_s()-t0)/reps;
    if(iters_out)*iters_out=reps; return dt;
}

// bench contexts
typedef struct{ const float* W; const float* x; float* y; int M,K; } CtxF;
static double run_f32(void* p){ CtxF* c=p; matvec_f32(c->W,c->x,c->y,c->M,c->K); return 0; }
typedef struct{ const int8_t* codes; const int8_t* lut; int32_t* y; int M,Mpad,K,nbits; } CtxL;
static double run_bin(void* p){ CtxL* c=p; matvec_lut_bin(c->codes,c->lut,c->y,c->M,c->Mpad,c->K,c->nbits); return 0; }
static double run_t3 (void* p){ CtxL* c=p; matvec_lut_t3 (c->codes,c->lut,c->y,c->M,c->Mpad,c->K); return 0; }

static void run_dims(const char* name, int M, int K, int reps){
    srand(1234+M*7+K);
    int Mpad=(M+31)&~31;
    float* Wf=amalloc((size_t)M*K*sizeof(float));
    float* xf=amalloc((size_t)K*sizeof(float));
    uint8_t* Wb=amalloc((size_t)M*K);     // unsigned b-bit weights (for the cost-vs-bits sweep)
    int8_t* Wt=amalloc((size_t)M*K);      // ternary {-1,0,1}
    int8_t* xq=amalloc((size_t)K);
    float* yf=amalloc((size_t)M*sizeof(float));
    int32_t* yi=amalloc((size_t)M*sizeof(int32_t));
    int32_t* yref=amalloc((size_t)M*sizeof(int32_t));
    int8_t* lutB=amalloc((size_t)(K/4)*16);
    int8_t* lutT=amalloc((size_t)(K/2)*16);
    int8_t* codB=amalloc((size_t)4*(K/4)*Mpad);   // up to 4 planes
    int8_t* codT=amalloc((size_t)(K/2)*Mpad);

    for(int i=0;i<M*K;i++){ Wf[i]=((rand()/(float)RAND_MAX)-0.5f)*0.2f; Wb[i]=rand()&15; int r=rand()%3; Wt[i]=(int8_t)(r-1); }
    for(int k=0;k<K;k++){ xf[k]=((rand()/(float)RAND_MAX)-0.5f)*2.0f; xq[k]=(int8_t)((rand()%63)-31); }

    build_lut_bin(xq,K,lutB);
    build_lut_t3 (xq,K,lutT);
    pack_t3(Wt,M,Mpad,K,codT);

    printf("\n==== dims M=%d K=%d (Mpad=%d) ====\n",M,K,Mpad);

    // fp32 baseline
    CtxF cf={Wf,xf,yf,M,K};
    double tf=bench(run_f32,&cf,reps,NULL);
    printf("  f32            : %8.1f ns   (%.2f GFLOP/s)\n", tf*1e9, 2.0*M*K/tf/1e9);

    // ternary (the real 1.58 kernel)
    ref_t3(Wt,xq,yref,M,K);
    CtxL ct={codT,lutT,yi,M,Mpad,K,0};
    matvec_lut_t3(codT,lutT,yi,M,Mpad,K);
    int bad=cmp_i32(yi,yref,M);
    double tt=bench(run_t3,&ct,reps,NULL);
    printf("  lut-ternary1.58: %8.1f ns   speedup=%.2fx   [correctness %s]\n", tt*1e9, tf/tt, bad<0?"OK":"FAIL");
    if(bad>=0){ printf("    !! mismatch at row %d: %d != %d\n",bad,yi[bad],yref[bad]); }

    // bit-serial binary sweep (signature B: fewer bits -> faster)
    printf("  -- bit-serial binary LUT (cost-vs-bits) --\n");
    double tb[5]={0};
    int sweep[3]={4,2,1};
    for(int s=0;s<3;s++){
        int nb=sweep[s];
        // mask weights to nb bits for an exact ref of THIS plane-count
        uint8_t* Wm=amalloc((size_t)M*K);
        for(int i=0;i<M*K;i++) Wm[i]=Wb[i]&((1<<nb)-1);
        pack_bin(Wm,M,Mpad,K,nb,codB);
        ref_bin(Wm,xq,yref,M,K);
        CtxL cb2={codB,lutB,yi,M,Mpad,K,nb};
        matvec_lut_bin(codB,lutB,yi,M,Mpad,K,nb);
        int bd=cmp_i32(yi,yref,M);
        double t=bench(run_bin,&cb2,reps,NULL); tb[nb]=t;
        printf("     %d-bit        : %8.1f ns   speedup=%.2fx   [correctness %s]\n", nb, t*1e9, tf/t, bd<0?"OK":"FAIL");
        if(bd>=0) printf("       !! mismatch at row %d: %d != %d\n",bd,yi[bd],yref[bd]);
        afree(Wm);
    }
    printf("  -> monotonic fewer-bits-faster: %s   (4b=%.0f >= 2b=%.0f >= 1b=%.0f ns)\n",
           (tb[4]>=tb[2]&&tb[2]>=tb[1])?"YES":"NO", tb[4]*1e9, tb[2]*1e9, tb[1]*1e9);

    afree(Wf);afree(xf);afree(Wb);afree(Wt);afree(xq);afree(yf);afree(yi);afree(yref);
    afree(lutB);afree(lutT);afree(codB);afree(codT);
}

int main(int argc, char** argv){
    printf("==== Phase 57 / Probe-1 (1b): pshufb-LUT low-bit matvec vs fp32 | Zen2 AVX2 (no VNNI) ====\n");
    printf("  signatures: (A) LUT beats f32 (anchor bitnet.cpp 2.37-6.17x)   (B) fewer bits -> faster\n");
    int reps = argc>1?atoi(argv[1]):200000;
    run_dims("fc1", 1024, 256, reps);   // MLP fc1: D->4D
    run_dims("fc2", 256, 1024, reps);   // MLP fc2: 4D->D
    run_dims("scale", 2048, 2048, reps/16>1?reps/16:1);  // scale-up point (LUT win grows with size)
    printf("\nSTOP. mechanism microbench on the 3600X. No commit.\n");
    return 0;
}
