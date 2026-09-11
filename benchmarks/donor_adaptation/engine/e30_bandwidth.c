// E30 -- how much read bandwidth does this box actually deliver to N threads?
//
// Brief: docs/research/donor_adaptation/briefs/BRIEF_E30_THE_WALL.md (63515b4), pushed before
// this file existed.
//
// The engine's byte rate at the goal's shape has to be divided by something, and a spec-sheet
// number is not a measurement.  This is that something: a read-only stream over a buffer whose
// size is swept from inside L3 to far outside DRAM's cache-friendly range, at the same thread
// count the engine uses.
//
// WHY READ-ONLY, AND WHY A SUM.  The engine's weight path READS weights and writes almost
// nothing -- one activation vector per matrix against megabytes of weights.  A STREAM-style
// triad would measure a mix of read and write bandwidth and flatter the denominator.  The sum
// is accumulated into a volatile so nothing can be elided, and the accumulator is per-thread
// so the loop is not a reduction chain.
//
// NO -ffast-math (standing law).  Built with the engine's own flags:
//   clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp e30_bandwidth.c -o e30_bandwidth.exe -lm
//
//   e30_bandwidth.exe <threads> <reps>
// prints one line per size:  bytes  GB/s(best of reps)  GB/s(median)
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

// The sizes: 4 MB sits inside this box's 32 MB L3, 8 GB is ten percent of its RAM.  The cliff
// Probe-3 measured at 16 MB has to fall inside the sweep or G-E30A cannot fire.
static const double SIZES_MB[] = {4, 8, 12, 16, 20, 24, 32, 64, 256, 1024, 2048, 4096, 8192};
#define NSIZES ((int)(sizeof(SIZES_MB)/sizeof(SIZES_MB[0])))
#define MAXREPS 16

static int cmpd(const void* a, const void* b){
    double x=*(const double*)a, y=*(const double*)b; return (x>y)-(x<y);
}

int main(int argc, char** argv){
    int threads = argc>1 ? atoi(argv[1]) : 6;
    int reps    = argc>2 ? atoi(argv[2]) : 3;
    if(reps>MAXREPS) reps=MAXREPS;
#ifdef _OPENMP
    omp_set_num_threads(threads); omp_set_dynamic(0);
#endif
    printf("# e30_bandwidth  threads %d  reps %d\n", threads, reps);
    printf("# bytes,MB,best_GB_s,median_GB_s\n");

    for(int si=0; si<NSIZES; si++){
        size_t n = (size_t)(SIZES_MB[si]*1048576.0/8.0);      // in uint64 words
        uint64_t* buf = (uint64_t*)malloc(n*sizeof(uint64_t));
        if(!buf){ fprintf(stderr,"  %.0f MB: allocation failed, stopping the sweep\n",
                          SIZES_MB[si]); break; }
        // touch every page so the timed pass measures reads and not first-touch faults
        for(size_t i=0;i<n;i++) buf[i] = (uint64_t)i*2654435761u;

        double gbs[MAXREPS];
        for(int r=0; r<reps; r++){
            double t0 = now_s();
            uint64_t acc = 0;
#ifdef _OPENMP
            #pragma omp parallel
            {
                uint64_t a0=0,a1=0,a2=0,a3=0;     // four accumulators: E8's law, so the loop is
                #pragma omp for schedule(static)  // not a latency chain instead of a byte stream
                for(long long i=0;i<(long long)n;i+=4){
                    a0+=buf[i]; a1+=buf[i+1]; a2+=buf[i+2]; a3+=buf[i+3];
                }
                #pragma omp atomic
                acc += a0+a1+a2+a3;
            }
#else
            for(size_t i=0;i<n;i++) acc += buf[i];
#endif
            double dt = now_s()-t0;
            g_sink += acc;                         // nothing above may be elided
            gbs[r] = (double)(n*sizeof(uint64_t)) / dt / 1e9;
        }
        double sorted[MAXREPS]; memcpy(sorted,gbs,sizeof(double)*reps);
        qsort(sorted,reps,sizeof(double),cmpd);
        double best = sorted[reps-1], med = sorted[reps/2];
        printf("%zu,%.0f,%.3f,%.3f\n", n*sizeof(uint64_t), SIZES_MB[si], best, med);
        fflush(stdout);
        free(buf);
    }
    if(g_sink == 0xdeadbeef) printf("# impossible\n");
    return 0;
}
