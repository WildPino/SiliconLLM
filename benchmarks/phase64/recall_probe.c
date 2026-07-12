// Phase 64.3 — recall-tier engine-side smoke (standalone, synthetic, no model weights, no engine coupling).
// Two-stage query per the sealed spec (PHASE64_DECISIONS.md §4 / P55-56 research): Hadamard-rotated space,
// IVF coarse quantizer (nlist proportional to sqrt(N)), 4-bit PQ ADC shortlist, exact fp32 top-16 rerank; dim 128,
// 128K synthetic entries. Measures: (a) us/token full query at threads {1,6}, (b) codes+index and value footprint.
// Standalone => engine.c is NOT touched: engine parity/kselftest is a separate control in the harness (zero perturbation).
// Build: clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp recall_probe.c -o bin/recall_probe.exe -lm
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(_WIN32)
#include <windows.h>
static double now_s(void){ LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart/(double)f.QuadPart; }
#else
#include <time.h>
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
#endif

#define ND   131072          // 128K entries (sealed)
#define D    128             // query/key dim (sealed)
#define NLIST 362            // ~ round(sqrt(131072)) -> nlist proportional to sqrt(N) (sealed)
#define NPROBE 4             // banked (P55 SIMVQ-IVF)
#define MSUB 16              // PQ subquantizers
#define KS   16             // 4-bit codebook (sealed: 4-bit ADC)
#define SUBD (D/MSUB)        // 8 dims / subquantizer
#define SHORT 64             // ADC shortlist size before exact rerank
#define TOPK 16              // exact rerank output (sealed)

static void* xaligned(size_t b){ void* p=_mm_malloc(b,64); if(!p){fprintf(stderr,"OOM %zu\n",b);exit(1);} return p; }
static inline float l2sq(const float* a,const float* b){ __m256 acc=_mm256_setzero_ps();
    for(int i=0;i<D;i+=8){ __m256 d=_mm256_sub_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i)); acc=_mm256_fmadd_ps(d,d,acc); }
    __m128 lo=_mm256_castps256_ps128(acc),hi=_mm256_extractf128_ps(acc,1); lo=_mm_add_ps(lo,hi);
    lo=_mm_add_ps(lo,_mm_movehl_ps(lo,lo)); lo=_mm_add_ss(lo,_mm_shuffle_ps(lo,lo,1)); return _mm_cvtss_f32(lo); }
static inline float l2sub(const float* a,const float* b){ float s=0; for(int i=0;i<SUBD;i++){ float d=a[i]-b[i]; s+=d*d; } return s; }
static void wht(float* a){ for(int len=1;len<D;len<<=1) for(int i=0;i<D;i+=len<<1) for(int j=0;j<len;j++){
    float x=a[i+j],y=a[i+j+len]; a[i+j]=x+y; a[i+j+len]=x-y; } }

static float* base;              // ND*D rotated vectors (rerank value store, RAM latency-class)
static float* cent;              // NLIST*D coarse centroids (rotated space)
static float  pqcb[MSUB][KS][SUBD];
static uint8_t* codes;           // ND*MSUB codes (one 4-bit index per byte; packed footprint reported separately)
static int* invlist;             // ND entry ids grouped by list
static int  loff[NLIST+1];       // list offsets

static uint64_t rng=0x243f6a8885a308d3ULL;
static inline float frand(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (float)((rng>>11)*(1.0/9007199254740992.0))*2.0f-1.0f; }

static void build_index(void){
    base=xaligned((size_t)ND*D*4); cent=xaligned((size_t)NLIST*D*4);
    codes=xaligned((size_t)ND*MSUB); invlist=malloc((size_t)ND*4);
    float* raw=xaligned((size_t)ND*D*4);
    for(size_t i=0;i<(size_t)ND*D;i++) raw[i]=frand();
    for(long e=0;e<ND;e++){ float* v=base+(size_t)e*D; memcpy(v,raw+(size_t)e*D,D*4); wht(v); }   // rotate all keys
    // coarse centroids = a random sample of rotated keys (partition is data-independent in intent; timing only needs balance)
    for(int c=0;c<NLIST;c++){ size_t e=((rng=rng*6364136223846793005ULL+1)>>17)%(uint64_t)ND; memcpy(cent+(size_t)c*D,base+e*D,D*4); }
    // PQ codebooks: per-subquantizer 16 centroids sampled from rotated keys' subvectors
    for(int m=0;m<MSUB;m++) for(int k=0;k<KS;k++){ size_t e=((rng=rng*6364136223846793005ULL+1)>>17)%(uint64_t)ND;
        memcpy(pqcb[m][k],base+e*D+m*SUBD,SUBD*4); }
    // assign each key to nearest centroid + PQ-encode
    int* cnt=calloc(NLIST,4); int* asg=malloc((size_t)ND*4);
    for(long e=0;e<ND;e++){ const float* v=base+(size_t)e*D; int bc=0; float bd=1e30f;
        for(int c=0;c<NLIST;c++){ float d=l2sq(v,cent+(size_t)c*D); if(d<bd){bd=d;bc=c;} } asg[e]=bc; cnt[bc]++;
        for(int m=0;m<MSUB;m++){ int bk=0; float bkd=1e30f; for(int k=0;k<KS;k++){ float d=l2sub(v+m*SUBD,pqcb[m][k]); if(d<bkd){bkd=d;bk=k;} } codes[(size_t)e*MSUB+m]=(uint8_t)bk; } }
    loff[0]=0; for(int c=0;c<NLIST;c++) loff[c+1]=loff[c]+cnt[c];
    int* cur=malloc((size_t)NLIST*4); memcpy(cur,loff,(size_t)NLIST*4);
    for(long e=0;e<ND;e++) invlist[cur[asg[e]]++]=(int)e;
    _mm_free(raw); free(cnt); free(asg); free(cur);
}

// full two-stage query; returns nothing (writes top-16 ids into out). nt = threads for the two hot loops.
static void query(const float* q,int nt,int* out){
    float qr[D]; memcpy(qr,q,D*4); wht(qr);
    float cd[NLIST];
    #pragma omp parallel for schedule(static) if(nt>1)
    for(int c=0;c<NLIST;c++) cd[c]=l2sq(qr,cent+(size_t)c*D);
    int pl[NPROBE]; float pv[NPROBE]; for(int p=0;p<NPROBE;p++){ pl[p]=-1; pv[p]=1e30f; }
    for(int c=0;c<NLIST;c++){ if(cd[c]<pv[NPROBE-1]){ int p=NPROBE-1; while(p>0&&cd[c]<pv[p-1]){ pv[p]=pv[p-1]; pl[p]=pl[p-1]; p--; } pv[p]=cd[c]; pl[p]=c; } }
    float tab[MSUB][KS];
    for(int m=0;m<MSUB;m++) for(int k=0;k<KS;k++) tab[m][k]=l2sub(qr+m*SUBD,pqcb[m][k]);
    static int   cand[ND];   // worst case; scratch (single-query serial gather)
    static float adc[ND];
    int nc=0; for(int p=0;p<NPROBE;p++){ int c=pl[p]; for(int i=loff[c];i<loff[c+1];i++) cand[nc++]=invlist[i]; }
    #pragma omp parallel for schedule(static) if(nt>1)
    for(int j=0;j<nc;j++){ const uint8_t* cc=codes+(size_t)cand[j]*MSUB; float s=0;
        for(int m=0;m<MSUB;m++) s+=tab[m][cc[m]]; adc[j]=s; }
    int sl[SHORT]; float sv[SHORT]; int ns=0;
    for(int j=0;j<nc;j++){ if(ns<SHORT){ int p=ns++; while(p>0&&adc[j]<sv[p-1]){ sv[p]=sv[p-1]; sl[p]=sl[p-1]; p--; } sv[p]=adc[j]; sl[p]=cand[j]; }
        else if(adc[j]<sv[SHORT-1]){ int p=SHORT-1; while(p>0&&adc[j]<sv[p-1]){ sv[p]=sv[p-1]; sl[p]=sl[p-1]; p--; } sv[p]=adc[j]; sl[p]=cand[j]; } }
    float tv[TOPK]; for(int t=0;t<TOPK;t++){ tv[t]=1e30f; out[t]=-1; }
    for(int s=0;s<ns;s++){ float d=l2sq(qr,base+(size_t)sl[s]*D);
        if(d<tv[TOPK-1]){ int p=TOPK-1; while(p>0&&d<tv[p-1]){ tv[p]=tv[p-1]; out[p]=out[p-1]; p--; } tv[p]=d; out[p]=sl[s]; } }
}

int main(int argc,char** argv){
    (void)argc;(void)argv;
    fprintf(stderr,"building synthetic index: N=%d dim=%d nlist=%d nprobe=%d M=%d(4-bit) ...\n",ND,D,NLIST,NPROBE,MSUB);
    build_index();
    int Q=5000; float* qs=xaligned((size_t)Q*D*4); for(size_t i=0;i<(size_t)Q*D;i++) qs[i]=frand();
    int out[TOPK];
    printf("==== 64.3 recall two-stage query smoke (IVF+Hadamard / 4-bit ADC shortlist / exact top-%d rerank; N=%d, dim=%d) ====\n",TOPK,ND,D);
    // footprint (b)
    double codes4 = (double)ND*MSUB/2.0;                 // 4-bit packed
    double idx = (double)NLIST*D*4 + (double)ND*4 + (double)(NLIST+1)*4 + sizeof(pqcb); // centroids + invlist + offsets + codebooks
    double vals = (double)ND*D*4;                        // rerank value store (RAM latency-class, ~SHORT*D*4/token touched)
    printf("  (b) footprint: codes(4-bit) %.2f MB + index %.2f MB = %.2f MB searchable | value store %.1f MB (RAM, ~%.0f KB touched/token)\n",
           codes4/1048576,idx/1048576,(codes4+idx)/1048576,vals/1048576,(double)SHORT*D*4/1024);
    printf("  (a) threads |  us/token   (queries=%d, full two-stage)\n",Q);
    for(int ti=0;ti<2;ti++){ int nt=ti?6:1;
#ifdef _OPENMP
        omp_set_num_threads(nt); omp_set_dynamic(0);
#endif
        for(int w=0;w<64;w++) query(qs+(size_t)(w%Q)*D,nt,out);            // warm
        double best=1e30; for(int rep=0;rep<3;rep++){ double t0=now_s();
            for(int i=0;i<Q;i++) query(qs+(size_t)i*D,nt,out); double dt=now_s()-t0; if(dt<best) best=dt; }
        printf("      %-7d | %8.2f\n",nt,best/Q*1e6);
    }
    printf("STOP. 64.3 recall smoke above. gate: <=50 us/token @128K + zero engine perturbation (parity control in harness).\n");
    return 0;
}
