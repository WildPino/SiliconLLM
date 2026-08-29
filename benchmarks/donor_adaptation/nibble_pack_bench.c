// nibble_pack_bench.c — P1 STAGE 2 (timing), per BRIEF_P1_NIBBLE_PACKING Amendment 1 (commit e73420e).
//
// Stage 2 exists because P1 is a DIRECT discriminator: it changes the weight-code byte stream by exactly 2x,
// bit-exactly (Stage 1, C1-C4), with the per-weight pshufb count unchanged. A1.3 registers the reading:
//
//     if bandwidth-limited   -> matvecs/s roughly DOUBLES,   moved GB/s roughly INVARIANT
//     if compute/port-limited-> matvecs/s roughly INVARIANT, moved GB/s roughly HALVES
//
// Both quantities are reported for both arms. Intermediate outcomes are reported as intermediate.
//
// BYTE CONVENTION (A1.4), stated once, applied identically to both arms:
//   MOVED bytes   = bytes of the weight-code array the kernel's loads actually pull from memory in one
//                   matvec, COUNTED at runtime as 32 B per _mm256_loadu_si256 issued from `codes`.
//                   Accesses are contiguous within 64 B-aligned streams and M is a multiple of 32 at every
//                   shape measured here, so every byte of every touched line is requested: requested ==
//                   moved. This is the reporting convention; all GB/s figures headlined are moved GB/s.
//   CHARGED bytes = bytes attributable to the M*K useful weights only (M*K/2 byte arm, M*K/4 nibble arm).
//                   Reported in its own labelled column. NEVER combined with moved in a single ratio.
//
// PROTOCOL (A1.6 + coordinator): per-rep MEAN is the primary estimator, never min-of-reps (a minimum over
// reps is a maximum order statistic and biases upward by construction). Per-rep sd/CV reported. The binary
// is launched multiple times and between-invocation dispersion is aggregated outside. NO quiescence banner
// is printed and none may be cited: the machine is sampled independently, by a separate process, during
// each run.
//
// ARMS: {byte, nibble} x {stride_pad 0, +64 B} x {threads 1, 6} x 4 donor organ shapes, plus the d5cd
// resident-vs-streamed control re-fired at donor shape before any rate is quoted.
//
// Build: clang -O3 -mavx2 -mfma -march=znver2 -fopenmp benchmarks/donor_adaptation/nibble_pack_bench.c \
//              -o bin/nibble_pack_bench.exe -lm         (NO -ffast-math)
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
#include <windows.h>
static double now_s(void){ LARGE_INTEGER f,t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t); return (double)t.QuadPart/(double)f.QuadPart; }

#define AQ 63
static int g_omp_on=0;
#ifdef _OPENMP
#define OMP_PFOR _Pragma("omp parallel for schedule(static) if(g_omp_on)")
#else
#define OMP_PFOR
#endif

static void* xmalloc(size_t n){ void*p=malloc(n); if(!p){fprintf(stderr,"OOM %zu\n",n);exit(1);} return p; }

/* ---- engine.c primitives, verbatim; the only edit is the BP_CNT() accounting line ---- */
static long long g_bp_code_bytes=0; static int g_bp_count=0;
#define BP_CNT() do{ if(g_bp_count) g_bp_code_bytes+=32; }while(0)
static inline void acc_add_i8x32(__m256i* acc,__m256i p){
    __m128i lo=_mm256_castsi256_si128(p),hi=_mm256_extracti128_si256(p,1);
    acc[0]=_mm256_add_epi32(acc[0],_mm256_cvtepi8_epi32(lo)); acc[1]=_mm256_add_epi32(acc[1],_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8)));
    acc[2]=_mm256_add_epi32(acc[2],_mm256_cvtepi8_epi32(hi)); acc[3]=_mm256_add_epi32(acc[3],_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8)));
}
static void matvec_lut_full(const int8_t* codes,const int8_t* lut,int32_t* y,int M,int Mpad,int T){
    OMP_PFOR for(int base=0;base<M;base+=32){
        __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};
        for(int t=0;t<T;t++){ __m256i tbl=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)t*16)));
            __m256i idx=_mm256_loadu_si256((const __m256i*)(codes+(size_t)t*Mpad+base)); BP_CNT(); acc_add_i8x32(acc,_mm256_shuffle_epi8(tbl,idx)); }
        int32_t tmp[32]; _mm256_storeu_si256((__m256i*)(tmp+0),acc[0]); _mm256_storeu_si256((__m256i*)(tmp+8),acc[1]);
        _mm256_storeu_si256((__m256i*)(tmp+16),acc[2]); _mm256_storeu_si256((__m256i*)(tmp+24),acc[3]);
        for(int r=0;r<32&&base+r<M;r++) y[base+r]=tmp[r]; }
}
static void build_lut_t3(const int8_t* xq,int T,int8_t* lut){ for(int t=0;t<T;t++){ int8_t x0=xq[2*t],x1=xq[2*t+1];
    for(int c=0;c<16;c++){ int s=0; if(c<9){ int w0=c/3-1,w1=c%3-1; s=w0*x0+w1*x1; } lut[t*16+c]=(int8_t)s; } } }
static void bc_tm(const int8_t* Wt,int M,int K,int Mpad,int8_t* codes){ int T=K/2;
    for(int t=0;t<T;t++){ for(int m=0;m<M;m++){ int w0=Wt[(size_t)m*K+2*t],w1=Wt[(size_t)m*K+2*t+1]; codes[(size_t)t*Mpad+m]=(int8_t)((w0+1)*3+(w1+1)); }
        for(int m=M;m<Mpad;m++) codes[(size_t)t*Mpad+m]=0; } }
static void ref_t3(const int8_t* Wt,const int8_t* xq,int32_t* y,int M,int K){
    for(int m=0;m<M;m++){ long s=0; for(int k=0;k<K;k++) s+=(long)Wt[(size_t)m*K+k]*xq[k]; y[m]=(int32_t)s; } }

#include "nibble_pack.h"

static int g_ach_threads=1;
static void set_threads(int nt){
#ifdef _OPENMP
    omp_set_num_threads(nt); g_ach_threads=1;
    #pragma omp parallel if(nt>1)
    {
        #pragma omp single
        g_ach_threads=omp_get_num_threads();
    }
#else
    g_ach_threads=1;
#endif
    g_omp_on=(nt>1);
}

static uint64_t rs=0x9E3779B97F4A7C15ULL;
static inline uint32_t r32(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (uint32_t)(rs>>32); }

static int cmpd(const void* a,const void* b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:(x>y?1:0); }

typedef struct { const char* name; int M,K; } Shape;
static Shape SH[4]={ {"llama70b_kv",1024,8192}, {"llama70b_qo",8192,8192},
                     {"llama70b_gate_up",28672,8192}, {"llama70b_down",8192,28672} };

static long g_inv=0;              /* invocation id, from argv */
static double g_target_gb=12.0;   /* bytes to stream per timed cell */
static long g_pool_mb=512;

/* One timed cell. arm: 0=byte 1=nibble. stride_pad in bytes added to the plane stride.
   pool_mb<=0 selects the RESIDENT mode (single replica) for the d5cd control. */
static void cell(const char* tag,Shape s,int arm,int stride_pad,int nt,long pool_mb){
    int M=s.M,K=s.K,T=K/2;
    int Mpad=(M+31)&~31;
    int stride=Mpad+stride_pad;                       /* the ONLY thing stride_pad changes */
    int planes = arm? NP_TP(T) : T;
    size_t block=(size_t)planes*stride;

    /* build one reference block */
    int8_t* Wt=xmalloc((size_t)M*K);
    for(size_t i=0;i<(size_t)M*K;i++) Wt[i]=(int8_t)((int)(r32()%3)-1);
    int8_t* xq=xmalloc(K); for(int k=0;k<K;k++) xq[k]=(int8_t)((int)(r32()%(2*AQ+1))-AQ);
    int8_t* lut=xmalloc((size_t)T*16); build_lut_t3(xq,T,lut);
    int8_t* one=xmalloc(block);
    if(arm) bc_tm_n(Wt,M,K,stride,one); else bc_tm(Wt,M,K,stride,one);

    /* correctness guard: never time a broken kernel, and prove the stride arm is still exact */
    int32_t* yref=xmalloc((size_t)M*4); int32_t* y=xmalloc((size_t)M*4);
    ref_t3(Wt,xq,yref,M,K);
    g_omp_on=0;
    if(arm) matvec_lut_full_n(one,lut,y,M,stride,T); else matvec_lut_full(one,lut,y,M,stride,T);
    int bad=0; for(int m=0;m<M;m++) if(y[m]!=yref[m]) bad++;
    if(bad){ printf("P1S2ERR,%s,%s,arm=%d,stride_pad=%d,BITEXACT_FAIL rows=%d\n",tag,s.name,arm,stride_pad,bad); exit(2); }
    free(Wt);

    /* MOVED bytes: counted at runtime, single-threaded, OUTSIDE every timed window */
    g_bp_code_bytes=0; g_np_code_bytes=0; g_bp_count=1; g_np_count=1;
    if(arm) matvec_lut_full_n(one,lut,y,M,stride,T); else matvec_lut_full(one,lut,y,M,stride,T);
    g_bp_count=0; g_np_count=0;
    long long moved = arm? g_np_code_bytes : g_bp_code_bytes;
    long long charged = arm? ((long long)M*K)/4 : ((long long)M*K)/2;

    /* replica pool */
    int R;
    if(pool_mb<=0) R=1;                                    /* d5cd resident arm */
    else { R=(int)(((size_t)pool_mb*1048576 + block-1)/block); if(R<2)R=2; }
    char* pool=xmalloc((size_t)R*block);
    for(int r=0;r<R;r++) memcpy(pool+(size_t)r*block,one,block);
    free(one);

    long reps=(long)(g_target_gb*1e9/(double)moved); if(reps<20)reps=20; if(reps>40000)reps=40000;
    double* dt=xmalloc((size_t)reps*sizeof(double));

    set_threads(nt);
    for(int w=0;w<3;w++){ if(arm) matvec_lut_full_n((int8_t*)pool,lut,y,M,stride,T); else matvec_lut_full((int8_t*)pool,lut,y,M,stride,T); }

    for(long i=0;i<reps;i++){
        const int8_t* c=(const int8_t*)(pool+(size_t)(i%R)*block);
        double t0=now_s();
        if(arm) matvec_lut_full_n(c,lut,y,M,stride,T); else matvec_lut_full(c,lut,y,M,stride,T);
        dt[i]=now_s()-t0;
    }
    /* PRIMARY estimator = per-rep MEAN. min-of-reps is deliberately not computed. */
    double sum=0; for(long i=0;i<reps;i++) sum+=dt[i];
    double mean=sum/reps;
    double var=0; for(long i=0;i<reps;i++){ double d=dt[i]-mean; var+=d*d; }
    double sd=sqrt(var/(reps>1?reps-1:1));
    qsort(dt,reps,sizeof(double),cmpd); double med=dt[reps/2];

    double mvps = 1.0/mean;
    double moved_gbs   = (double)moved  *mvps/1e9;
    double charged_gbs = (double)charged*mvps/1e9;

    printf("P1S2CSV,%ld,%s,%s,%d,%d,%s,%d,%d,%d,%d,%d,%ld,%d,%.1f,%.4f,%.4f,%.2f,%.4f,%.2f,%lld,%.3f,%lld,%.3f\n",
           g_inv,tag,s.name,M,K, arm?"nibble":"byte", stride_pad, stride, planes, nt, g_ach_threads,
           reps, R, (double)R*block/1048576.0,
           mean*1e6, med*1e6, sd/mean*100.0, sd*1e6,
           mvps, moved, moved_gbs, charged, charged_gbs);
    fflush(stdout);
    free(pool); free(dt); free(lut); free(xq); free(y); free(yref);
}

int main(int argc,char** argv){
    int do_main=1,do_d5cd=1;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--inv")&&i+1<argc) g_inv=atol(argv[++i]);
        else if(!strcmp(argv[i],"--target-gb")&&i+1<argc) g_target_gb=atof(argv[++i]);
        else if(!strcmp(argv[i],"--pool-mb")&&i+1<argc) g_pool_mb=atol(argv[++i]);
        else if(!strcmp(argv[i],"--seed")&&i+1<argc) rs=strtoull(argv[++i],NULL,0);
        else if(!strcmp(argv[i],"--only-d5cd")) do_main=0;
        else if(!strcmp(argv[i],"--only-main")) do_d5cd=0;
        else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 1; }
    }
#ifdef _OPENMP
    omp_set_dynamic(0);
#endif
    printf("P1S2HDR,inv,tag,shape,M,K,arm,stride_pad,stride_used,planes,threads_req,threads_ach,reps,replicas,pool_MiB,mean_us,median_us,cv_pct,sd_us,matvecs_per_s,moved_B,moved_GBs,charged_B,charged_GBs\n");
    fflush(stdout);

    /* d5cd re-fired at donor shape BEFORE any rate is quoted: does the meter distinguish a block that
       fits L3 from the same block streamed from a pool >> L3? Both arms, both thread counts. */
    if(do_d5cd){
        for(int arm=0;arm<2;arm++) for(int ti=0;ti<2;ti++){ int nt=ti?6:1;
            cell("d5cd_resident",SH[0],arm,0,nt,0);
            cell("d5cd_streamed",SH[0],arm,0,nt,g_pool_mb); }
    }
    if(do_main){
        for(int si=0;si<4;si++) for(int sp=0;sp<2;sp++){ int stride_pad=sp?64:0;
            for(int arm=0;arm<2;arm++) for(int ti=0;ti<2;ti++){ int nt=ti?6:1;
                cell(sp?"main_stride64":"main",SH[si],arm,stride_pad,nt,g_pool_mb); } }
    }
    printf("P1S2DONE,%ld\n",g_inv);
    return 0;
}
