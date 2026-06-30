// Phase 57 / Probe-3 (Finding-4) - CACHE-RESIDENCY: the step-function bandwidth curve on Zen2 (Ryzen 5 3600X).
//   magic: "CSWP" 0x43535750
//
//   WHY synthetic (no model): our 5M sits entirely in cache -> there is NO L3 crossing to observe on it. So we EMULATE
//   the batch-1 decode weight-stream: each token re-streams the SAME model weights. We stream a working-set of W bytes
//   of TERNARY codes (same pshufb-LUT layout as probe-1 phase57_lutbench.c) and REUSE the same W over many iterations:
//     - W <= L3  -> resident from iter 2 on (fast)
//     - W  > L3  -> every iteration refetches from DRAM (slow)
//   ONE VARIABLE = W. We sweep W across L1(32KB)/L2(512KB)/L3(16MB/CCX)/DRAM and read off the plateaus + the cliff.
//
//   Light per-byte work (load 64B codes + 2x pshufb into a RESIDENT LUT + accumulate) so the kernel is MEMORY-bound at
//   each level -> the curve reflects each cache level's bandwidth, not compute. This is the real weight-stream work
//   shape (pshufb-LUT), just bandwidth-limited.
//
//   Secondary (cheap, high value): same sweep with RANDOM-gather access (LCG-addressed, prefetcher-defeating) instead
//   of SEQUENTIAL -> measures the random/seq penalty-rho on our silicon = the law that killed SIMVQ / vector-codebooks
//   / graphs, now in real Zen2 numbers.
//
//   Hygiene: pin to ONE physical core (single decode thread sees its CCX's 16MB L3); warmup discards cold iters;
//   accumulate the LUT output into a volatile sink (defeat dead-code elimination); QPC high-res timer; ~2GB streamed
//   per point for stable numbers; report OBSERVED boundaries, not datasheet.
//
//   Scope (like probe-1/2): validates the MECHANISM (the step EXISTS on our silicon) + quantifies the active-slice
//   BUDGET; NOT a model speedup (no model). No commit.
//
// Build (Zen2): clang -O3 -mavx2 -mfma -march=znver2 phase57_cachesweep.c -o phase57_cachesweep
// Run         : ./phase57_cachesweep [core] [target_GB_streamed]    (default core=2, 2 GB/point; add 'smoke' for a fast pass)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <immintrin.h>
#if defined(_WIN32)
#include <windows.h>
static double now_s(void){ LARGE_INTEGER f,t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t); return (double)t.QuadPart/(double)f.QuadPart; }
static void pin_core(int c){ SetThreadAffinityMask(GetCurrentThread(),(DWORD_PTR)1<<c); SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_HIGHEST); }
static void* amalloc(size_t n){ void* p=_aligned_malloc(n,64); if(!p){fprintf(stderr,"alloc %zu fail\n",n);exit(1);} return p; }
static void afree(void* p){ _aligned_free(p); }
#else
#define _GNU_SOURCE
#include <time.h>
#include <pthread.h>
#include <sched.h>
static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static void pin_core(int c){ cpu_set_t s; CPU_ZERO(&s); CPU_SET(c,&s); sched_setaffinity(0,sizeof(s),&s); }
static void* amalloc(size_t n){ void* p=NULL; if(posix_memalign(&p,64,n)) {fprintf(stderr,"alloc fail\n");exit(1);} return p; }
static void afree(void* p){ free(p); }
#endif

#define NLUT 64                                   // 64 resident LUTs (2KB) -> L1-resident, reused across the stream
static volatile int64_t g_sink = 0;               // DCE sink

static inline int64_t reduce(__m256i a){
    __m256i s=_mm256_sad_epu8(a,_mm256_setzero_si256());
    return _mm256_extract_epi64(s,0)+_mm256_extract_epi64(s,1)+_mm256_extract_epi64(s,2)+_mm256_extract_epi64(s,3);
}

// SEQUENTIAL pass over the whole W-byte buffer (units of 64B = one cache line); resident LUT; light work -> mem-bound
static int64_t pass_seq(const int8_t* codes, const __m256i* luts, long nunits){
    __m256i acc=_mm256_setzero_si256();
    for(long u=0;u<nunits;u++){
        const int8_t* p=codes+(u<<6);
        __m256i t=luts[u&(NLUT-1)];
        acc=_mm256_add_epi8(acc,_mm256_shuffle_epi8(t,_mm256_loadu_si256((const __m256i*)p)));
        acc=_mm256_add_epi8(acc,_mm256_shuffle_epi8(t,_mm256_loadu_si256((const __m256i*)(p+32))));
    }
    return reduce(acc);
}
// RANDOM-gather pass: LCG random + Lemire range-reduction (mul+shift, no divide, any nunits) -> cache lines spread
// across the whole W range, non-strided -> defeats the HW prefetcher.
static int64_t pass_rand(const int8_t* codes, const __m256i* luts, long nunits, uint64_t seed){
    __m256i acc=_mm256_setzero_si256(); uint64_t st=seed;
    for(long u=0;u<nunits;u++){
        st=st*6364136223846793005ULL+1442695040888963407ULL;
        uint32_t rr=(uint32_t)(st>>32);
        long unit=(long)(((uint64_t)rr*(uint64_t)nunits)>>32);
        const int8_t* p=codes+(unit<<6);
        __m256i t=luts[unit&(NLUT-1)];
        acc=_mm256_add_epi8(acc,_mm256_shuffle_epi8(t,_mm256_loadu_si256((const __m256i*)p)));
        acc=_mm256_add_epi8(acc,_mm256_shuffle_epi8(t,_mm256_loadu_si256((const __m256i*)(p+32))));
    }
    return reduce(acc);
}

static void run_point(size_t W, int8_t* codes, const __m256i* luts, double target_bytes){
    long nunits=(long)(W>>6);                       // 64B units (one cache line each); W is a multiple of 64
    long iters=(long)(target_bytes/(double)W); if(iters<4) iters=4;
    long warm=iters/20; if(warm<2) warm=2;
    // ---- sequential ----
    for(long i=0;i<warm;i++) g_sink+=pass_seq(codes,luts,nunits);
    double t0=now_s(); int64_t s=0;
    for(long i=0;i<iters;i++){ s+=pass_seq(codes,luts,nunits); __asm__ __volatile__("":::"memory"); }
    double dt=now_s()-t0; g_sink+=s;
    double gb_seq=(double)W*iters/dt/1e9; double tok_seq=iters/dt;
    // ---- random ----
    for(long i=0;i<warm;i++) g_sink+=pass_rand(codes,luts,nunits,0x1234+i);
    double r0=now_s(); int64_t r=0;
    for(long i=0;i<iters;i++){ r+=pass_rand(codes,luts,nunits,0xABCDEF^i); __asm__ __volatile__("":::"memory"); }
    double rdt=now_s()-r0; g_sink+=r;
    double gb_rand=(double)W*iters/rdt/1e9; double tok_rand=iters/rdt;

    const char* unit = W<(1<<20)?"KB":"MB"; double wv = W<(1<<20)?W/1024.0:W/1048576.0;
    printf("  %6.0f%s | seq %7.1f GB/s (%9.0f tok/s) | rand %7.1f GB/s (%9.0f tok/s) | rho(seq/rand)=%4.1fx\n",
           wv,unit, gb_seq,tok_seq, gb_rand,tok_rand, gb_seq/gb_rand);
    fflush(stdout);
}

int main(int argc,char** argv){
    int core = argc>1 && strcmp(argv[1],"smoke") ? atoi(argv[1]) : 2;
    int smoke = (argc>1 && !strcmp(argv[1],"smoke")) || (argc>2 && !strcmp(argv[2],"smoke"));
    double target = smoke ? 2.5e8 : 2.0e9;          // bytes streamed per point
    pin_core(core);
    printf("==== Phase 57 / Probe-3: cache-residency working-set sweep | Zen2 Ryzen 5 3600X | core=%d %s ====\n",
           core, smoke?"(SMOKE)":"");
    printf("  pattern: re-stream W bytes of ternary codes (batch-1 decode); LIGHT pshufb-LUT work -> memory-bound.\n");
    printf("  expect: L1(32KB)/L2(512KB)/L3(16MB-per-CCX) plateaus, then the DRAM cliff past ~16MB.\n\n");

    // resident LUTs (2KB, L1)
    __m256i* luts=amalloc(NLUT*sizeof(__m256i));
    int8_t tmp[16]; for(int l=0;l<NLUT;l++){ for(int k=0;k<16;k++) tmp[k]=(int8_t)((l*7+k*3)%23-11); luts[l]=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)tmp)); }

    // biggest buffer once; sub-sweeps view a prefix (all powers of two)
    size_t Wmax = smoke ? (size_t)16<<20 : (size_t)128<<20;
    int8_t* codes=amalloc(Wmax);
    for(size_t i=0;i<Wmax;i++) codes[i]=(int8_t)((i*1103515245u+12345u)%9);   // touch every page; codes 0..8

    size_t sweep_full[]={16<<10,32<<10,64<<10,128<<10,256<<10,512<<10,
                         1u<<20,2u<<20,4u<<20,8u<<20,12u<<20,16u<<20,24u<<20,32u<<20,64u<<20,128u<<20};
    size_t sweep_smoke[]={16<<10,256<<10,1u<<20,8u<<20,16u<<20};
    size_t* sweep = smoke?sweep_smoke:sweep_full;
    int nsw = smoke? (int)(sizeof(sweep_smoke)/sizeof(size_t)) : (int)(sizeof(sweep_full)/sizeof(size_t));

    printf("    W    |   SEQUENTIAL throughput        |   RANDOM-gather throughput     | penalty\n");
    for(int i=0;i<nsw;i++){
        size_t W=sweep[i]; if(W>Wmax) continue;
        run_point(W,codes,luts,target);
    }
    printf("\n  sink=%lld (ignore)\n",(long long)g_sink);
    printf("STOP. cache-residency curve above (read the L3->DRAM cliff + rho). No model kernel, no commit.\n");
    afree(codes); afree(luts);
    return 0;
}
