// Phase 56 STAGE 1 - faithful CPU cost of the SIMVQ / learned-IVF recall probe (the Researcher's sketch).
//   The thesis CPU-win claim: a learned IVF index makes recall SUBLINEAR in ctx (skip the O(ctx) score-scan).
//   This measures the REAL per-token cost of the IVF probe and -- the load-bearing part -- models it at 128K.
//
//   IVF probe, per token, recall slot (random weights ok; cost is value-independent):
//     (1) INSERT new key : argmax over V centroids of dot(newkey, C_j)        -> V*d            (const in ctx)
//     (2) PROBE query    : score query vs all V centroids, pick top-nprobe    -> V*d + topk(V)  (const in ctx)
//     (3) BUCKET-GATHER  : candidates = members of the nprobe buckets ~ nprobe/V * ctx          (RANDOM reads)
//     (4) EXACT-REFINE   : dot(query, K[cand]) for each candidate, top-k      -> n_cand*d
//     (5) VALUE-GATHER   : sum top-k value vectors                            -> k random reads
//   So cost = CONST(2*V*d) + LINEAR_in_ctx( (nprobe/V)*ctx*(d refine + 1 random vec read) ).
//   vs naive sparse-recall = ctx*d score-scan (fully linear, coeff d). IVF linear coeff = (nprobe/V)*d = d/64 here.
//
//   THE TRAP (why we must MODEL 128K, not slope 1K-8K): at ctx<=8K the K-store fits in L2/L3 so the random
//   gather looks ~free; at 128K the store is 32MB (DRAM) and the (nprobe/V)*128K = 2048 random vec reads become
//   the dominant, latency-bound term. So we measure the random-gather cost AS A FUNCTION OF STORE SIZE (the
//   cache cliff) and plug the 32MB-store latency into the 128K model. Constant + refine slopes measured directly.
//
// Build: gcc -O3 -march=native -mavx2 -mfma benchmarks/phase56/phase56_ivf_profile.c -o bin/phase56_ivf.exe -lm
// Run:   bin/phase56_ivf.exe [--dim 64] [--V 256] [--nprobe 4] [--k 16] [--tokens 4000]
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
static double now(){ return (double)clock()/CLOCKS_PER_SEC; }

int main(int argc,char**argv){
    int D=64,V=256,NPROBE=4,K=16; long ntok=4000;
    for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"--dim")&&i+1<argc)D=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--V")&&i+1<argc)V=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--nprobe")&&i+1<argc)NPROBE=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--k")&&i+1<argc)K=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--tokens")&&i+1<argc)ntok=atol(argv[++i]); }
    long maxctx=131072;                                   // 128K -> store = 128K*D*4 bytes (32MB at D=64) = DRAM
    size_t vecB=(size_t)D*4;
    printf("==== Phase 56 STAGE 1 | learned-IVF probe CPU cost | D=%d V=%d nprobe=%d k=%d tok=%ld | vec=%zuB ====\n",D,V,NPROBE,K,ntok,vecB);
    printf("  store@128K = %.1f MB per tensor (K and V) -> DRAM-resident. nprobe/V=%.4f -> n_cand(ctx)=ctx*%.4f\n",
           (double)maxctx*vecB/1e6,(double)NPROBE/V,(double)NPROBE/V);

    float* C=malloc((size_t)V*D*4);                       // V learned centroids
    float* Kst=malloc((size_t)maxctx*D*4);                // key store
    float* Vst=malloc((size_t)maxctx*D*4);                // value store
    float* q=malloc(vecB); float* nk=malloc(vecB); float* acc=malloc(vecB);
    float* cs=malloc((size_t)V*4); int* cand=malloc((size_t)(maxctx)*4);
    for(long i=0;i<(long)V*D;i++) C[i]=frand();
    for(long i=0;i<maxctx*D;i++){ Kst[i]=frand(); Vst[i]=frand(); }

    // ---------- (A) RANDOM-GATHER cost vs STORE SIZE (the cache cliff). reads = independent (prefetchable, MLP) ----------
    printf("\n  -- (A) random vec-gather latency vs store size (exposes the L2/L3 -> DRAM cliff) --\n");
    printf("  %8s %10s | per-read ns\n","storeVec","storeMB");
    long sizes[]={1024,4096,16384,65536,131072,0};
    double gather_ns_at_128k=0;
    for(int si=0;sizes[si];si++){ long S=sizes[si]; long reads=4096;
        double t0=now(); volatile float sink=0;
        for(long it=0;it<ntok;it++){ float a=0;
            for(long j=0;j<reads;j++){ long idx=xr()%S; const float* v=Vst+(size_t)idx*D; for(int d=0;d<D;d++) a+=v[d]; }
            sink+=a; }
        double ns=(now()-t0)/((double)ntok*reads)*1e9;
        printf("  %8ld %10.1f | %9.2f\n",S,(double)S*vecB/1e6,ns);
        if(S==maxctx) gather_ns_at_128k=ns;
    }

    // ---------- (B) constant terms: insert argmax(V) + probe centroid-score(V) ----------
    double t0=now(); volatile float sink=0;
    for(long it=0;it<ntok;it++){ for(int d=0;d<D;d++){ q[d]=frand(); nk[d]=frand(); }
        float best=-1e30f; for(int j=0;j<V;j++){ float s=dotf(nk,C+(size_t)j*D,D); if(s>best)best=s; }   // (1) insert argmax
        for(int j=0;j<V;j++) cs[j]=dotf(q,C+(size_t)j*D,D);                                              // (2) probe scores
        for(int p=0;p<NPROBE;p++){ float bb=-1e30f; int bi=0; for(int j=0;j<V;j++) if(cs[j]>bb){bb=cs[j];bi=j;} cs[bi]=-1e30f; } // top-nprobe
        sink+=best; }
    double const_us=(now()-t0)/ntok*1e6;                  // const part, per token (independent of ctx)

    // ---------- (C) refine: per-candidate exact dot (d-dot) ----------
    t0=now();
    for(long it=0;it<ntok;it++){ for(int d=0;d<D;d++) q[d]=frand();
        float a=0; for(long j=0;j<2048;j++){ const float* kk=Kst+(size_t)(j)*D; a+=dotf(q,kk,D); } sink+=a; }
    double refine_per_cand_us=(now()-t0)/((double)ntok*2048)*1e6;  // sequential-read d-dot (refine compute, cache-friendly)

    double gather_per_read_us=gather_ns_at_128k/1e3;      // DRAM random read of one vec (the load-bearing 128K term)

    // ---------- (D) full measured pipeline at ctx 1K..8K (everything together, realistic cache for that ctx) ----------
    printf("\n  -- (D) measured per-token cost: IVF full pipeline vs naive score-scan (us/tok/layer) --\n");
    printf("  %7s %8s | %12s %12s | %10s\n","ctx","n_cand","IVF-measured","naive-scan","speedup");
    long ctxs[]={1024,2048,4096,8192,0};
    for(int ci=0;ctxs[ci];ci++){ long ctx=ctxs[ci]; long ncand=(long)NPROBE*ctx/V; if(ncand<1)ncand=1;
        // IVF full: insert + probe + candidate-pick(random in [0,ctx)) + refine + value-gather
        t0=now();
        for(long it=0;it<ntok;it++){ for(int d=0;d<D;d++){ q[d]=frand(); nk[d]=frand(); }
            float best=-1e30f; for(int j=0;j<V;j++){ float s=dotf(nk,C+(size_t)j*D,D); if(s>best)best=s; }       // insert
            for(int j=0;j<V;j++) cs[j]=dotf(q,C+(size_t)j*D,D);                                                  // probe
            for(int p=0;p<NPROBE;p++){ float bb=-1e30f; int bi=0; for(int j=0;j<V;j++) if(cs[j]>bb){bb=cs[j];bi=j;} cs[bi]=-1e30f; }
            for(long j=0;j<ncand;j++) cand[j]=xr()%ctx;                                                          // bucket members (scattered)
            float rbest=-1e30f;
            for(long j=0;j<ncand;j++){ const float* kk=Kst+(size_t)cand[j]*D; float s=dotf(q,kk,D); if(s>rbest)rbest=s; } // refine (random read+dot)
            memset(acc,0,vecB); for(int j=0;j<K&&j<ncand;j++){ const float* v=Vst+(size_t)cand[j]*D; for(int d=0;d<D;d++) acc[d]+=v[d]; } // value gather
            sink+=best+rbest+acc[it%D]; }
        double ivf_us=(now()-t0)/ntok*1e6;
        // naive: score-scan all ctx keys (= sparse arm's brute force)
        t0=now();
        for(long it=0;it<ntok;it++){ for(int d=0;d<D;d++) q[d]=frand();
            float a=0; for(long s=0;s<ctx;s++) a+=dotf(q,Kst+(size_t)s*D,D); sink+=a; }
        double naive_us=(now()-t0)/ntok*1e6;
        printf("  %7ld %8ld | %12.2f %12.2f | %9.2fx\n",ctx,ncand,ivf_us,naive_us,naive_us/ivf_us);
    }

    // ---------- (E) MODEL to 128K: const + n_cand*(refine + DRAM-gather), naive = ctx*per-dot ----------
    double perdot_us=refine_per_cand_us;                  // sequential d-dot cost (same op as naive scan)
    printf("\n  -- (E) EXPLICIT MODEL (const + linear terms; DRAM gather @32MB store) --\n");
    printf("     const term (insert+probe, 2*V dot + topk) = %.3f us/tok\n",const_us);
    printf("     refine per candidate (d-dot, cached)       = %.4f us\n",refine_per_cand_us);
    printf("     random vec gather @128K (DRAM)             = %.4f us/read  (%.1f ns)\n",gather_per_read_us,gather_ns_at_128k);
    printf("  %9s %9s | %14s %14s | %10s\n","ctx","n_cand","IVF-modeled","naive-modeled","speedup");
    long mc[]={8192,32768,131072,0};
    double ivf_128k=0,naive_128k=0;
    for(int i=0;mc[i];i++){ long ctx=mc[i]; long ncand=(long)NPROBE*ctx/V;
        double ivf = const_us + (double)ncand*(refine_per_cand_us + gather_per_read_us);
        double naive = (double)ctx*perdot_us;
        printf("  %9ld %9ld | %14.2f %14.2f | %9.2fx\n",ctx,ncand,ivf,naive,naive/ivf);
        if(ctx==131072){ ivf_128k=ivf; naive_128k=naive; }
    }

    // ---------- GATE 1 verdict ----------
    printf("\n  ==== GATE 1 (pre-registered): IVF@128K <= ~30 us AND >> margin vs naive ====\n");
    printf("     IVF@128K modeled = %.2f us/tok/layer ; naive@128K = %.2f us/tok/layer ; speedup = %.1fx\n",
           ivf_128k,naive_128k,naive_128k/ivf_128k);
    int pass = (ivf_128k <= 30.0);
    printf("     n_cand@128K = %ld random DRAM reads dominate (%.2f us = %.0f%% of IVF cost).\n",
           (long)NPROBE*131072/V, (double)((long)NPROBE*131072/V)*gather_per_read_us,
           100.0*((double)((long)NPROBE*131072/V)*gather_per_read_us)/ivf_128k);
    printf("     VERDICT: %s (threshold 30 us)\n", pass?"PASS -> proceed to STAGE 2 (quality)":"FAIL -> SIMVQ dies here, clean");
    printf("  (sink=%.1f)\nSTOP. no commit.\n",(double)sink);
    free(C);free(Kst);free(Vst);free(q);free(nk);free(acc);free(cs);free(cand); return 0;
}
