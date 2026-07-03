// Phase 60 / GEMV kernel microbench (probe-1 discipline, on the 3600X). Phase-61 double-fail -> SSM projections stay
//   fp32; the engine lever for the 52.7% scan-other bottleneck is a faster fp32 batch-1 GEMV. Hypothesis: the current
//   engine matvec (one __m256 accumulator per row + per-row hsum) is FMA-LATENCY-bound (Zen2 FMA lat ~5c, tput 2/c ->
//   a single accumulator chain runs at ~1/5 peak). Fix = multiple independent accumulators (ILP) + amortized x-load.
//   Measures GFLOP/s + speedup + max rel-err vs a float64 reference, on the REAL projection dims (+ head).
//
// Build: clang -O3 -mavx2 -mfma -march=znver2 benchmarks/phase60/gemv_bench.c -o bin/gemv_bench.exe -lm
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#if defined(_WIN32)
#include <windows.h>
static double now_s(void){ LARGE_INTEGER f,t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t); return (double)t.QuadPart/(double)f.QuadPart; }
#else
#include <time.h>
static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
#endif

static inline float hsum256(__m256 v){ __m128 lo=_mm256_castps256_ps128(v),hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi); lo=_mm_hadd_ps(lo,lo); lo=_mm_hadd_ps(lo,lo); return _mm_cvtss_f32(lo); }

// ---- current engine kernel: one accumulator per row (latency-bound) ----
static inline float dotf(const float*a,const float*b,int n){ __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),s);
    float r=hsum256(s); for(;i<n;i++) r+=a[i]*b[i]; return r; }
static void matvec_ref(const float*W,const float*x,float*y,int M,int K){ for(int o=0;o<M;o++) y[o]=dotf(W+(size_t)o*K,x,K); }

// ---- blocked: R=4 rows, 4 independent accumulators, x loaded once per 8-k for all 4 rows ----
static void matvec_blk4(const float*W,const float*x,float*y,int M,int K){
    int o=0;
    for(;o+4<=M;o+=4){
        const float*w0=W+(size_t)o*K,*w1=w0+K,*w2=w1+K,*w3=w2+K;
        __m256 a0=_mm256_setzero_ps(),a1=a0,a2=a0,a3=a0; int k=0;
        for(;k<=K-8;k+=8){ __m256 xv=_mm256_loadu_ps(x+k);
            a0=_mm256_fmadd_ps(_mm256_loadu_ps(w0+k),xv,a0); a1=_mm256_fmadd_ps(_mm256_loadu_ps(w1+k),xv,a1);
            a2=_mm256_fmadd_ps(_mm256_loadu_ps(w2+k),xv,a2); a3=_mm256_fmadd_ps(_mm256_loadu_ps(w3+k),xv,a3); }
        float s0=hsum256(a0),s1=hsum256(a1),s2=hsum256(a2),s3=hsum256(a3);
        for(;k<K;k++){ s0+=w0[k]*x[k]; s1+=w1[k]*x[k]; s2+=w2[k]*x[k]; s3+=w3[k]*x[k]; }
        y[o]=s0; y[o+1]=s1; y[o+2]=s2; y[o+3]=s3;
    }
    for(;o<M;o++) y[o]=dotf(W+(size_t)o*K,x,K);
}
// ---- blocked-2acc: single row, 2 interleaved accumulators (ILP within a row; helps tiny-M like dt_proj) ----
static void matvec_blk4x2(const float*W,const float*x,float*y,int M,int K){
    int o=0;
    for(;o+4<=M;o+=4){
        const float*w0=W+(size_t)o*K,*w1=w0+K,*w2=w1+K,*w3=w2+K;
        __m256 a0=_mm256_setzero_ps(),a1=a0,a2=a0,a3=a0,b0=a0,b1=a0,b2=a0,b3=a0; int k=0;
        for(;k<=K-16;k+=16){ __m256 x0=_mm256_loadu_ps(x+k),x1=_mm256_loadu_ps(x+k+8);
            a0=_mm256_fmadd_ps(_mm256_loadu_ps(w0+k),x0,a0); b0=_mm256_fmadd_ps(_mm256_loadu_ps(w0+k+8),x1,b0);
            a1=_mm256_fmadd_ps(_mm256_loadu_ps(w1+k),x0,a1); b1=_mm256_fmadd_ps(_mm256_loadu_ps(w1+k+8),x1,b1);
            a2=_mm256_fmadd_ps(_mm256_loadu_ps(w2+k),x0,a2); b2=_mm256_fmadd_ps(_mm256_loadu_ps(w2+k+8),x1,b2);
            a3=_mm256_fmadd_ps(_mm256_loadu_ps(w3+k),x0,a3); b3=_mm256_fmadd_ps(_mm256_loadu_ps(w3+k+8),x1,b3); }
        for(;k<=K-8;k+=8){ __m256 xv=_mm256_loadu_ps(x+k);
            a0=_mm256_fmadd_ps(_mm256_loadu_ps(w0+k),xv,a0); a1=_mm256_fmadd_ps(_mm256_loadu_ps(w1+k),xv,a1);
            a2=_mm256_fmadd_ps(_mm256_loadu_ps(w2+k),xv,a2); a3=_mm256_fmadd_ps(_mm256_loadu_ps(w3+k),xv,a3); }
        float s0=hsum256(_mm256_add_ps(a0,b0)),s1=hsum256(_mm256_add_ps(a1,b1)),s2=hsum256(_mm256_add_ps(a2,b2)),s3=hsum256(_mm256_add_ps(a3,b3));
        for(;k<K;k++){ s0+=w0[k]*x[k]; s1+=w1[k]*x[k]; s2+=w2[k]*x[k]; s3+=w3[k]*x[k]; }
        y[o]=s0; y[o+1]=s1; y[o+2]=s2; y[o+3]=s3;
    }
    for(;o<M;o++) y[o]=dotf(W+(size_t)o*K,x,K);
}

static double maxrel(const float*a,const float*b,int n){ double m=0; for(int i=0;i<n;i++){ double d=fabs((double)a[i]-b[i])/(fabs((double)b[i])+1e-9); if(d>m)m=d; } return m; }
static void ref64(const float*W,const float*x,float*y,int M,int K){ for(int o=0;o<M;o++){ double s=0; for(int k=0;k<K;k++) s+=(double)W[(size_t)o*K+k]*x[k]; y[o]=(float)s; } }

static void run(const char* nm,int M,int K,long reps){
    float*W=malloc((size_t)M*K*4),*x=malloc((size_t)K*4),*y0=malloc((size_t)M*4),*y1=malloc((size_t)M*4),*y2=malloc((size_t)M*4),*yr=malloc((size_t)M*4);
    srand(1234+M+K); for(size_t i=0;i<(size_t)M*K;i++) W[i]=((rand()/(float)RAND_MAX)-0.5f)*0.2f; for(int k=0;k<K;k++) x[k]=((rand()/(float)RAND_MAX)-0.5f)*2.0f;
    ref64(W,x,yr,M,K);
    matvec_ref(W,x,y0,M,K); matvec_blk4(W,x,y1,M,K); matvec_blk4x2(W,x,y2,M,K);
    for(int w=0;w<3;w++){ matvec_ref(W,x,y0,M,K); matvec_blk4(W,x,y1,M,K); matvec_blk4x2(W,x,y2,M,K); }
    double t0=now_s(); for(long r=0;r<reps;r++){ matvec_ref(W,x,y0,M,K); __asm__ __volatile__("":::"memory"); } double tr=(now_s()-t0)/reps;
    t0=now_s(); for(long r=0;r<reps;r++){ matvec_blk4(W,x,y1,M,K); __asm__ __volatile__("":::"memory"); } double tb=(now_s()-t0)/reps;
    t0=now_s(); for(long r=0;r<reps;r++){ matvec_blk4x2(W,x,y2,M,K); __asm__ __volatile__("":::"memory"); } double tb2=(now_s()-t0)/reps;
    double gf=2.0*M*K/1e9;
    printf("  %-9s M=%4d K=%4d | ref %6.2f GF/s | blk4 %6.2f GF/s (%.2fx) | blk4x2 %6.2f GF/s (%.2fx) | relerr blk4=%.1e vs-ref64 ref=%.1e\n",
           nm,M,K, gf/tr, gf/tb, tr/tb, gf/tb2, tr/tb2, maxrel(y1,y0,M), maxrel(y0,yr,M));
    free(W);free(x);free(y0);free(y1);free(y2);free(yr);
}

int main(int argc,char**argv){
    long reps=argc>1?atol(argv[1]):300000;
    printf("==== GEMV kernel microbench (Zen2 AVX2, batch-1 fp32) | ref=current engine matvec (1 acc/row) ====\n");
    printf("  hypothesis: ref is FMA-latency-bound (1 accumulator) -> multi-acc blocked wins; parity = fp reorder only\n");
    run("in_proj",  1024, 256, reps);
    run("x_proj",    208, 512, reps);
    run("dt_proj",   512,  16, reps*4);
    run("out_proj",  256, 512, reps);
    run("swa_qkv",   768, 256, reps);
    run("head",     1024, 256, reps);
    printf("STOP. GEMV microbench. relerr blk4-vs-ref = fp-reorder (both ~= float64 ref). No commit.\n");
    return 0;
}
