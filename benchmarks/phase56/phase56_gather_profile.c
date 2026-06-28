// Phase 56 - CPU cost of the sparse-KV-recall mechanism (arm 3's hidden bottleneck).
//   The "simplest" top-k query-aware recall over an UNCOMPRESSED KV store does, per token:
//     (a) SCORE SCAN: dot(query, key_s) for ALL s in ctx  -> O(ctx*D), sequential read of keys.
//     (b) TOP-K select over the ctx scores                -> O(ctx*k) cheap.
//     (c) GATHER: read the k selected value-vectors at RANDOM positions -> k*D random read.
//   The Researcher flagged (c): a random gather can be slower per-byte than a linear read.
//   This isolates where the cost lives. Reports us/token for: full recall, score-scan only,
//   random gather of k vs SEQUENTIAL read of k (same bytes = the pure random-access penalty),
//   and a linear read of the whole store (reference). Cross-platform (Linux T4 node + Windows dev).
//
// Build: gcc -O3 -march=native -mavx2 -mfma benchmarks/phase56/phase56_gather_profile.c -o bin/phase56_gather.exe -lm
// Run:   bin/phase56_gather.exe [--dim 64] [--k 16] [--tokens 20000]
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

static inline float dotf(const float*a,const float*b,int n){ __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),s);
    float o[8]; _mm256_storeu_ps(o,s); float r=o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; for(;i<n;i++) r+=a[i]*b[i]; return r; }
static uint64_t rs=0x9e3779b9ULL; static inline uint32_t xr(){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (uint32_t)(rs>>32); }
static inline float frand(){ return ((float)(xr()&0xFFFFF)/(float)0x100000-0.5f); }

int main(int argc,char**argv){
    int D=64,K=16; long ntok=20000;
    long ctxs[]={1024,2048,4096,8192,0};
    for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"--dim")&&i+1<argc)D=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--k")&&i+1<argc)K=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--tokens")&&i+1<argc)ntok=atol(argv[++i]); }
    long maxctx=8192;
    float* keys=malloc((size_t)maxctx*D*4); float* vals=malloc((size_t)maxctx*D*4);
    float* query=malloc((size_t)D*4); float* acc=malloc((size_t)D*4);
    float* scores=malloc((size_t)maxctx*4); int* topidx=malloc((size_t)K*4);
    for(long i=0;i<maxctx*D;i++){ keys[i]=frand(); vals[i]=frand(); }

    printf("==== Phase 56 CPU gather profile (sparse-KV-recall arm) | D=%d k=%d tokens=%ld | AVX2 ====\n",D,K,ntok);
    printf("  per-token us, broken down. score-scan is O(ctx); gather is k random reads (the flagged cost).\n");
    printf("  %6s | %10s %10s %10s | %12s %12s | %10s\n","ctx","FULL","score-scan","topk+gath","gather(rand)","seq-read(k)","linear-all");
    for(int ci=0;ci<4;ci++){ long ctx=ctxs[ci];
        // ---- FULL recall: score-scan + topk + random gather, per token ----
        clock_t t0=clock(); volatile float sink=0;
        for(long it=0;it<ntok;it++){
            for(int d=0;d<D;d++) query[d]=frand();
            for(long s=0;s<ctx;s++) scores[s]=dotf(query,keys+(size_t)s*D,D);            // (a) score scan
            for(int j=0;j<K;j++){ float best=-1e30f; int bi=0;                            // (b) top-k (simple)
                for(long s=0;s<ctx;s++){ if(scores[s]>best){best=scores[s];bi=(int)s;} } topidx[j]=bi; scores[bi]=-1e30f; }
            memset(acc,0,D*4);
            for(int j=0;j<K;j++){ const float* v=vals+(size_t)topidx[j]*D; for(int d=0;d<D;d++) acc[d]+=v[d]; } // (c) gather
            sink+=acc[it%D];
        }
        double full=(double)(clock()-t0)/CLOCKS_PER_SEC/ntok*1e6;
        // ---- score-scan only ----
        t0=clock();
        for(long it=0;it<ntok;it++){ for(int d=0;d<D;d++) query[d]=frand();
            float ss=0; for(long s=0;s<ctx;s++) ss+=dotf(query,keys+(size_t)s*D,D); sink+=ss; }
        double scan=(double)(clock()-t0)/CLOCKS_PER_SEC/ntok*1e6;
        // ---- random gather of k value-vectors (positions scattered across ctx) ----
        t0=clock();
        for(long it=0;it<ntok;it++){ memset(acc,0,D*4);
            for(int j=0;j<K;j++){ long idx=xr()%ctx; const float* v=vals+(size_t)idx*D; for(int d=0;d<D;d++) acc[d]+=v[d]; }
            sink+=acc[it%D]; }
        double gath=(double)(clock()-t0)/CLOCKS_PER_SEC/ntok*1e6;
        // ---- SEQUENTIAL read of k value-vectors (same bytes, contiguous) = the random-access baseline ----
        t0=clock();
        for(long it=0;it<ntok;it++){ memset(acc,0,D*4); long base=(xr()%(ctx-K))*D;
            for(int j=0;j<K;j++){ const float* v=vals+base+(size_t)j*D; for(int d=0;d<D;d++) acc[d]+=v[d]; }
            sink+=acc[it%D]; }
        double seq=(double)(clock()-t0)/CLOCKS_PER_SEC/ntok*1e6;
        // ---- linear read of the WHOLE value store (reference O(ctx) sequential) ----
        t0=clock();
        for(long it=0;it<ntok;it++){ memset(acc,0,D*4);
            for(long s=0;s<ctx;s++){ const float* v=vals+(size_t)s*D; for(int d=0;d<D;d++) acc[d]+=v[d]; }
            sink+=acc[it%D]; }
        double lin=(double)(clock()-t0)/CLOCKS_PER_SEC/ntok*1e6;
        double topkgath=full-scan>0?full-scan:0;
        printf("  %6ld | %10.2f %10.2f %10.2f | %12.3f %12.3f | %10.2f   (sink=%.1f)\n",
               ctx,full,scan,topkgath,gath,seq,lin,(double)sink);
    }
    printf("\nreading: FULL ~= score-scan (the O(ctx) dot scan DOMINATES; brute-force top-k re-reads all keys/token).\n");
    printf("  gather(rand) vs seq-read(k): same bytes, the delta = the random-access penalty the Researcher flagged.\n");
    printf("  => simplest sparse-recall is NOT sublinear (full scan/token); the gather itself is small. The real\n");
    printf("     CPU win needs an INDEX to skip the scan -- and THEN the random gather penalty becomes load-bearing.\n");
    printf("STOP. no commit.\n");
    free(keys);free(vals);free(query);free(acc);free(scores);free(topidx); return 0;
}
