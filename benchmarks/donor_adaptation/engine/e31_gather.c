// E31 -- what does a GATHERED byte cost, and at what granularity does the penalty go away?
//
// Brief: docs/research/donor_adaptation/briefs/BRIEF_E31_THE_GATHERED_BYTE.md, pushed before
// this file is ever RUN.
//
// E30 measured the dense ceiling of this box (36.30 GB/s) and closed the numerator: the only
// variable left is bytes per token, and the target is 1.45 G moved weights.  Every route to that
// target -- MoE, the carve, structured sparsity -- reads a SUBSET of each matrix.  A subset is
// not read the way a matrix is read: the hardware moves whole cache lines, the prefetcher loses
// the stream, and the USEFUL bytes per second can be far below the dense ceiling even when the
// useful byte COUNT is exactly what the budget says.
//
// So this measures the denominator's own ceiling: pick a fraction of the buffer in contiguous
// groups of G bytes, read only those, and report GB/s of the bytes actually ASKED FOR.
//
//   e31_gather.exe <threads> <reps> <buffer_MB> <div>      div = 1/fraction selected
//   e31_gather.exe --selftest                              correctness only, prints no timing
//
// Prints one line per (group size, order):  group_bytes,order,useful_MB,best_GB_s,median_GB_s
//
// THE ORDERS.  `sorted` walks the selected groups in ascending address order -- what a sane
// engine does once it knows which experts fired.  `random` walks them in a shuffled order --
// what a naive gather does.  `contig` reads the SAME VOLUME as one unbroken run and is the
// planted control: same bytes, no scatter, so it must come back at the dense ceiling.  If it
// does not, the instrument is measuring volume rather than granularity and nothing else it
// prints can be believed.
//
// Four accumulators per thread (E8's law), volatile sink, engine flags, never -ffast-math:
//   clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp e31_gather.c -o e31_gather.exe -lm
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(_WIN32)
#include <windows.h>
static double now_s(void){
    static LARGE_INTEGER f; static int init=0; LARGE_INTEGER t;
    if(!init){ QueryPerformanceFrequency(&f); init=1; }
    QueryPerformanceCounter(&t); return (double)t.QuadPart/(double)f.QuadPart;
}
#else
#include <time.h>
static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
                           return ts.tv_sec + 1e-9*ts.tv_nsec; }
#endif

static volatile uint64_t g_sink = 0;

// The granularities that matter, and why each one is in the list:
//   64     one cache line -- the floor, and the granularity of a single ternary weight fetch
//   128    two lines, the Zen 2 L2 prefetch pair
//   256    a 512-trit row
//   768    ONE S15 FFN ROW (1536 trits packed) -- E26 measured S15 suffering more than T10
//   2048   ONE T10 FFN ROW (4096 trits packed) -- E26's other shape, measured to suffer less
//   8192 .. 1048576   coarser carve groups: 4, 16, 64, 256, 512 rows of T10
static const size_t GROUPS[] = {64, 128, 256, 768, 2048, 8192, 32768, 131072, 524288, 2097152};
#define NGROUPS ((int)(sizeof(GROUPS)/sizeof(GROUPS[0])))
#define MAXREPS 16

enum { ORD_SORTED = 0, ORD_RANDOM = 1, ORD_CONTIG = 2, ORD_CONTIG_IND = 3 };
static const char* ORDNAME[4] = {"sorted", "random", "contig", "contig_ind"};
#define NORDERS 4

static int cmpd(const void* a, const void* b){
    double x=*(const double*)a, y=*(const double*)b; return (x>y)-(x<y);
}
static int cmpsz(const void* a, const void* b){
    size_t x=*(const size_t*)a, y=*(const size_t*)b; return (x>y)-(x<y);
}

static uint64_t rng_s = 0x243F6A8885A308D3ull;
static uint64_t rnd(void){                       // splitmix64
    uint64_t z = (rng_s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* Sum `nsel` groups of `gwords` uint64 each, starting at the offsets in `off`. */
static uint64_t gather_sum(const uint64_t* buf, const size_t* off, size_t nsel, size_t gwords){
    uint64_t acc = 0;
#ifdef _OPENMP
    #pragma omp parallel
    {
        uint64_t a0=0,a1=0,a2=0,a3=0;
        #pragma omp for schedule(static)
        for(long long g=0; g<(long long)nsel; g++){
            const uint64_t* p = buf + off[g];
            size_t i = 0;
            for(; i + 4 <= gwords; i += 4){ a0+=p[i]; a1+=p[i+1]; a2+=p[i+2]; a3+=p[i+3]; }
            for(; i < gwords; i++) a0 += p[i];
        }
        #pragma omp atomic
        acc += a0+a1+a2+a3;
    }
#else
    for(size_t g=0; g<nsel; g++){
        const uint64_t* p = buf + off[g];
        for(size_t i=0;i<gwords;i++) acc += p[i];
    }
#endif
    return acc;
}

/* THE REGISTERED PLANTED CONTROL: the same VOLUME as one unbroken run, read as one
   unbroken run -- no per-group indirection at all.

   RUN 1 WAS VOIDED BY G-E31A because this control was implemented as a walk through the
   offset array.  That carries the gathered arms' own per-group bookkeeping, so it sagged
   from 39.96 GB/s at 2048 B groups to 34.34 at 64 B -- it MOVED WITH THE VARIABLE IT WAS
   CONTROLLING FOR, which is not a control, and it flattered every small-group ratio it
   was the denominator of.  The gate caught it; the fix is the stricter reading of the
   brief ("one unbroken run"), not a looser gate.  `contig_ind` keeps the old behaviour as
   a DIAGNOSTIC so the bookkeeping is quantified instead of hidden. */
static uint64_t flat_sum(const uint64_t* buf, size_t nwords){
    uint64_t acc = 0;
#ifdef _OPENMP
    #pragma omp parallel
    {
        uint64_t a0=0,a1=0,a2=0,a3=0;
        #pragma omp for schedule(static)
        for(long long i=0;i<(long long)nwords;i+=4){
            a0+=buf[i]; a1+=buf[i+1]; a2+=buf[i+2]; a3+=buf[i+3];
        }
        #pragma omp atomic
        acc += a0+a1+a2+a3;
    }
#else
    for(size_t i=0;i<nwords;i++) acc += buf[i];
#endif
    return acc;
}

static int selftest(void){
    /* The gather must read EXACTLY the selected groups and nothing else.  Checked against a
       serial reference on a buffer small enough to verify by hand, at a group size that is not
       a multiple of the unrolled stride, so the tail loop is exercised too. */
    size_t n = 4096;                                  // words
    uint64_t* buf = (uint64_t*)malloc(n*8);
    for(size_t i=0;i<n;i++) buf[i] = i*2654435761u + 7;
    size_t gwords = 7;                                // 56 B, deliberately not 64 and not /4
    size_t ngrp = n/gwords, nsel = ngrp/3;
    size_t* off = (size_t*)malloc(nsel*sizeof(size_t));
    for(size_t k=0;k<nsel;k++) off[k] = (k*3+1)*gwords;
    uint64_t ref = 0;
    for(size_t k=0;k<nsel;k++) for(size_t i=0;i<gwords;i++) ref += buf[off[k]+i];
    uint64_t got = gather_sum(buf, off, nsel, gwords);
    printf("# SELFTEST gather over %zu groups of %zu words: ref %llu got %llu -> %s\n",
           nsel, gwords, (unsigned long long)ref, (unsigned long long)got,
           ref==got ? "OK" : "MISMATCH");
    size_t fw = (nsel*gwords) & ~(size_t)3;          /* flat_sum sums a multiple of 4 */
    uint64_t fref = 0; for(size_t i=0;i<fw;i++) fref += buf[i];
    uint64_t fgot = flat_sum(buf, fw);
    printf("# SELFTEST flat over %zu words: ref %llu got %llu -> %s\n",
           fw, (unsigned long long)fref, (unsigned long long)fgot,
           fref==fgot ? "OK" : "MISMATCH");
    free(off); free(buf);
    return (ref==got && fref==fgot) ? 0 : 1;
}

int main(int argc, char** argv){
    if(argc>1 && strcmp(argv[1],"--selftest")==0) return selftest();
    int threads  = argc>1 ? atoi(argv[1]) : 6;
    int reps     = argc>2 ? atoi(argv[2]) : 3;
    double bufmb = argc>3 ? atof(argv[3]) : 2048.0;
    int div      = argc>4 ? atoi(argv[4]) : 8;        // select 1 group in `div`
    if(reps>MAXREPS) reps=MAXREPS;
    if(div<1) div=1;
#ifdef _OPENMP
    omp_set_num_threads(threads); omp_set_dynamic(0);
#endif
    printf("# e31_gather  threads %d  reps %d  buffer %.0f MB  select 1/%d\n",
           threads, reps, bufmb, div);
    printf("# group_bytes,order,useful_MB,best_GB_s,median_GB_s\n");

    size_t n = (size_t)(bufmb*1048576.0/8.0);
    uint64_t* buf = (uint64_t*)malloc(n*sizeof(uint64_t));
    if(!buf){ fprintf(stderr,"allocation of %.0f MB failed\n", bufmb); return 2; }
    for(size_t i=0;i<n;i++) buf[i] = (uint64_t)i*2654435761u;      // first-touch every page

    /* ---- plan every (granularity, order) cell up front, so the TIMED loop can interleave.
       RUNS 1-3 WERE VOIDED.  Run 1: the control walked the offset array and so moved with the
       variable it controlled for.  Runs 2 and 3: the control was honest but each arm's reps sat
       in one contiguous wall-clock window, so a slow phase took a whole block and `contig` --
       identical code over identical volume at every row -- read 29.89 GB/s at one granularity
       and 40.38 at the next.  Raising reps from 3 to 15 did not help, which is the signature of
       block-correlated drift rather than per-rep noise.

       E28's registered methodology is INTERLEAVED reps so that drift is COMMON-MODE across the
       arms a ratio is taken over.  This instrument had reps innermost and arms outermost, the
       exact opposite.  Reps are now OUTERMOST.  That is the programme's own standard and it is
       stricter, not looser: the gate is unchanged. */
    size_t* offs[NGROUPS][NORDERS];
    size_t  nsels[NGROUPS], gwordss[NGROUPS];
    int     live[NGROUPS];
    for(int gi=0; gi<NGROUPS; gi++){
        size_t gwords = GROUPS[gi]/8;
        size_t ngrp = n/gwords, nsel = ngrp/(size_t)div;
        gwordss[gi] = gwords; nsels[gi] = nsel;
        live[gi] = (nsel >= 8);
        for(int o=0;o<NORDERS;o++) offs[gi][o] = NULL;
        if(!live[gi]) continue;

        size_t* idx = (size_t*)malloc(ngrp*sizeof(size_t));
        if(!idx){ fprintf(stderr,"index array for %zu B groups failed\n", GROUPS[gi]);
                  live[gi]=0; continue; }
        for(size_t i=0;i<ngrp;i++) idx[i]=i;
        for(size_t k=0;k<nsel;k++){
            size_t j = k + (size_t)(rnd() % (uint64_t)(ngrp-k));
            size_t t = idx[k]; idx[k]=idx[j]; idx[j]=t;
        }
        size_t* so = (size_t*)malloc(nsel*sizeof(size_t));      /* sorted  */
        size_t* ro = (size_t*)malloc(nsel*sizeof(size_t));      /* random  */
        size_t* co = (size_t*)malloc(nsel*sizeof(size_t));      /* contig_ind (contig needs none) */
        for(size_t k=0;k<nsel;k++){ so[k]=idx[k]; ro[k]=idx[k]*gwords; co[k]=k*gwords; }
        qsort(so, nsel, sizeof(size_t), cmpsz);
        for(size_t k=0;k<nsel;k++) so[k] *= gwords;
        free(idx);
        offs[gi][ORD_SORTED]=so; offs[gi][ORD_RANDOM]=ro;
        offs[gi][ORD_CONTIG]=co; offs[gi][ORD_CONTIG_IND]=co;   /* aliased; contig ignores it */
    }

    /* ---- the timed loop: REPS OUTERMOST, every cell visited once per rep */
    static double gbs[NGROUPS][NORDERS][MAXREPS];
    for(int r=0; r<reps; r++){
        for(int gi=0; gi<NGROUPS; gi++){
            if(!live[gi]) continue;
            size_t gwords = gwordss[gi], nsel = nsels[gi];
            for(int ord=0; ord<NORDERS; ord++){
                double t0 = now_s();
                /* ORD_CONTIG is the REGISTERED control and reads the run FLAT; ORD_CONTIG_IND
                   reads the identical addresses through the group loop, so the gap between the
                   two IS this harness's per-group bookkeeping at this granularity. */
                uint64_t acc = (ord==ORD_CONTIG)
                    ? flat_sum(buf, nsel*gwords)
                    : gather_sum(buf, offs[gi][ord], nsel, gwords);
                double dt = now_s()-t0;
                g_sink += acc;
                gbs[gi][ord][r] = (double)(nsel*gwords*8) / dt / 1e9;   /* USEFUL bytes only */
            }
        }
    }

    for(int gi=0; gi<NGROUPS; gi++){
        if(!live[gi]) continue;
        for(int ord=0; ord<NORDERS; ord++){
            double srt[MAXREPS]; memcpy(srt, gbs[gi][ord], sizeof(double)*reps);
            qsort(srt, reps, sizeof(double), cmpd);
            printf("%zu,%s,%.1f,%.3f,%.3f\n", GROUPS[gi], ORDNAME[ord],
                   (double)(nsels[gi]*gwordss[gi]*8)/1048576.0, srt[reps-1], srt[reps/2]);
        }
        fflush(stdout);
        free(offs[gi][ORD_SORTED]); free(offs[gi][ORD_RANDOM]);
        free(offs[gi][ORD_CONTIG]);          /* ORD_CONTIG_IND aliases it -- freed once */
    }
    free(buf);
    if(g_sink == 0xdeadbeef) printf("# impossible\n");
    return 0;
}
