// ---------------------------------------------------------------------------------------------------------------
// Donor-scale projection-GEMV rate harness  (donor adaptation, stage-1 rate constant)
//
// WHY THIS EXISTS
//   The donor speed verdict rests on ONE unmeasured constant: the fully-streamed fp32 projection-GEMV rate.
//   `docs/PHASE64_BUDGET.md` sec.1b measured r(size) only up to 96 MB and then *asserted* an asymptote of
//   "34-36 GB/s"; `benchmarks/donor_adaptation/donor_inventory.py` hardcodes the bottom of that (34.0) and
//   returns "zero donors pass the >=10 tok/s gate". At 36 the best donor passes (10.01), at 40 it reaches 10.65
//   (see docs/research/donor_adaptation/audits/CONTROLLER_STAGE1_AUDIT.md F1). Every donor row is an extrapolation PAST the last
//   measured point: Qwen2.5-1.5B streams 1.55 GB of fp32 projections+head per token, 16x beyond 96 MB.
//   This harness measures the curve where the donors actually live.
//
// WHAT IS COPIED, AND FROM WHERE (project law: copy, do not edit the originals)
//   - now_s / hsum256 / dotf / matvec / OMP_PFOR / g_omp_on / xmalloc  <- benchmarks/phase60/engine.c
//   - acc_add_i8x32 / matvec_lut_full / build_lut_t3 / bc_tm / ref_t3  <- benchmarks/phase60/engine.c
//   - the fp32 sweep methodology (npass, best-of-N, x-poison, synthetic fills)
//                                                                     <- engine.c run_gemv_sweep()  (the routine
//                                                                        that PRODUCED the sec.1b curve, driven by
//                                                                        benchmarks/phase64/bench_64_1b.sh)
//   - the i.i.d. expert-pool methodology                              <- engine.c run_expert_rate()
//   - GFLOP/s + max-rel-err-vs-float64 reporting on real proj dims    <- benchmarks/phase60/gemv_bench.c
//   Neither engine.c nor gemv_bench.c is modified.
//
// THE KNOWN-POSITIVE (project law sec.6.3 / feedback_planted_controls)
//   `repro` mode re-runs the *identical* sec.1b grid (in=512, sizes 4..96 MB, t1/t6) and diffs against the
//   published row. If it does not land on 187/185/134/60.5/55.7/45.5/45.3/36.5 GB/s at t6, this harness is NOT
//   the instrument that produced the repo's curve and every extrapolated point past 96 MB is untrustworthy.
//
// BUILD (64.1b policy - the flags that produced the curve being reproduced; NO -ffast-math, ever):
//   clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp benchmarks/donor_adaptation/gemv_donor_bench.c \
//         -o bin/gemv_donor_bench.exe -lm
//   optional tuning parity with the Makefile: add -march=znver2  (report which was used)
// ---------------------------------------------------------------------------------------------------------------
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
#include <tlhelp32.h>
#define PSAPI_VERSION 2
#include <psapi.h>
static double now_s(void){ LARGE_INTEGER f,t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t); return (double)t.QuadPart/(double)f.QuadPart; }
#else
#include <time.h>
static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
#endif

// ---------------------------------------------------------------------------------------------------------------
// MACHINE-QUIESCENCE PRECONDITION (permanent feature, not a one-off workaround for D5)
//   docs/PHASE64_BUDGET.md's 16-32 MB t6 points were found unreproducible precisely because environmental
//   state (thread placement, and -- undocumented at the time -- background load) went unrecorded, and swung
//   4.6x on placement alone (see DONOR_PROJ_RATE.md sec.2.3). A GB/s number taken while another heavy process
//   is resident is not a pessimistic reading of the truth; it is a DIFFERENT QUANTITY (contention, not the
//   kernel). Every mode below that does wall-clock timing scans for heavy resident processes FIRST and
//   refuses to run unless the machine is clean, or the caller explicitly overrides with --force-unclean
//   (which is itself recorded in the run header so it cannot be silently dropped when the number is quoted).
// ---------------------------------------------------------------------------------------------------------------
#define QUIESCENCE_THRESHOLD_BYTES (1073741824ULL)   // 1 GB resident working set -- documented, permanent
#if defined(_WIN32)
typedef struct { char name[MAX_PATH]; DWORD pid; size_t ws_bytes; } HeavyProc;
static int scan_heavy_processes(HeavyProc* out,int maxn){
    int n=0;
    DWORD self=GetCurrentProcessId();
    HANDLE snap=CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS,0);
    if(snap==INVALID_HANDLE_VALUE) return -1;
    PROCESSENTRY32 pe; pe.dwSize=sizeof(pe);
    if(Process32First(snap,&pe)){
        do{
            if(pe.th32ProcessID==self) continue;
            HANDLE h=OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION,FALSE,pe.th32ProcessID);
            if(h){
                PROCESS_MEMORY_COUNTERS pmc;
                if(GetProcessMemoryInfo(h,&pmc,sizeof(pmc))){
                    if((unsigned long long)pmc.WorkingSetSize>=QUIESCENCE_THRESHOLD_BYTES && n<maxn){
                        strncpy(out[n].name,pe.szExeFile,MAX_PATH-1); out[n].name[MAX_PATH-1]=0;
                        out[n].pid=pe.th32ProcessID; out[n].ws_bytes=(size_t)pmc.WorkingSetSize; n++;
                    }
                }
                CloseHandle(h);
            }
        } while(Process32Next(snap,&pe));
    }
    CloseHandle(snap);
    return n;
}
#else
typedef struct { char name[64]; long pid; size_t ws_bytes; } HeavyProc;
static int scan_heavy_processes(HeavyProc* out,int maxn){ (void)out;(void)maxn; return -1; }  // not implemented off-Windows
#endif
// returns 1 if clean (or check unavailable), 0 if a heavy process is present and require!=0 forced an exit(3)
static int quiescence_gate(int require,int force_unclean){
    HeavyProc hp[64]; int n=scan_heavy_processes(hp,64);
    if(n<0){ printf("# QUIESCENCE CHECK: process enumeration unavailable on this platform -- proceeding WITHOUT a\n"
                     "#   clean-environment guarantee. Record this fact if quoting any number from this run.\n"); return 1; }
    printf("# QUIESCENCE CHECK: threshold=%.1f GB resident working set | %d process(es) at/above it:\n",
           QUIESCENCE_THRESHOLD_BYTES/1073741824.0,n);
    for(int i=0;i<n;i++)
        printf("#   pid=%-8lu ws=%6.2f GB  %s\n",(unsigned long)hp[i].pid,hp[i].ws_bytes/1073741824.0,hp[i].name);
    if(n==0){ printf("#   -- clean.\n"); return 1; }
    if(!require){ printf("#   -- mode does not require a clean machine (no wall-clock timing); proceeding.\n"); return 1; }
    if(force_unclean){
        printf("# WARNING: --force-unclean OVERRIDE in effect with %d heavy process(es) resident. Every GB/s\n"
               "#   number in this run is CONTENDED, not a clean measurement of the kernel, and must be reported\n"
               "#   and used as such -- never presented as the kernel's rate.\n",n);
        return 1;
    }
    printf("# REFUSING TO RUN: this mode measures memory bandwidth and %d heavy process(es) are resident.\n"
           "#   A number taken now measures contention, not the LUT/fp32 path. Close the heavy process(es),\n"
           "#   or re-invoke with --force-unclean to override (the override is logged above and must be\n"
           "#   disclosed with any number quoted from that run).\n",n);
    return 0;
}

// ---- engine.c primitives, copied verbatim (bit-for-bit the kernels under test) ----
static int g_omp_on=0;
#define OMP_PFOR _Pragma("omp parallel for schedule(static) if(g_omp_on)")
static void* xmalloc(size_t n){ void*p=malloc(n); if(!p){fprintf(stderr,"OOM %zu bytes\n",n);exit(1);} return p; }
static inline float hsum256(__m256 v){ float o[8]; _mm256_storeu_ps(o,v); return o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; }
static inline float dotf(const float*a,const float*b,int n){ __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),s);
    float r=hsum256(s); for(;i<n;i++) r+=a[i]*b[i]; return r; }
static inline void matvec(const float*W,const float*x,float*y,int out,int in){ OMP_PFOR for(int o=0;o<out;o++) y[o]=dotf(W+(size_t)o*in,x,in); }

static inline void acc_add_i8x32(__m256i* acc,__m256i p){
    __m128i lo=_mm256_castsi256_si128(p),hi=_mm256_extracti128_si256(p,1);
    acc[0]=_mm256_add_epi32(acc[0],_mm256_cvtepi8_epi32(lo)); acc[1]=_mm256_add_epi32(acc[1],_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8)));
    acc[2]=_mm256_add_epi32(acc[2],_mm256_cvtepi8_epi32(hi)); acc[3]=_mm256_add_epi32(acc[3],_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8)));
}
static void matvec_lut_full(const int8_t* codes,const int8_t* lut,int32_t* y,int M,int Mpad,int T){
    OMP_PFOR for(int base=0;base<M;base+=32){
        __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};
        for(int t=0;t<T;t++){ __m256i tbl=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)t*16)));
            __m256i idx=_mm256_loadu_si256((const __m256i*)(codes+(size_t)t*Mpad+base)); acc_add_i8x32(acc,_mm256_shuffle_epi8(tbl,idx)); }
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
// scalar reference straight from the packed codes: y[m] = sum_t lut[t*16 + codes[t*Mpad+m]]  (the kernel's definition)
static void ref_lut_scalar(const int8_t* codes,const int8_t* lut,int32_t* y,int M,int Mpad,int T){
    for(int m=0;m<M;m++){ int s=0; for(int t=0;t<T;t++) s+=lut[t*16+(int)codes[(size_t)t*Mpad+m]]; y[m]=s; } }

static void set_threads(int nt){
#ifdef _OPENMP
    omp_set_num_threads(nt);
#endif
    g_omp_on=(nt>1);
}

// ---- PLANTED-CONTROL kernels: plausible artefacts that stream fewer bytes than the harness charges ----
// (a) half the ROWS never computed: 2x inflated GB/s, y half-stale.
static void matvec_halfrows(const float*W,const float*x,float*y,int out,int in){
    OMP_PFOR for(int o=0;o<out;o+=2) y[o]=dotf(W+(size_t)o*in,x,in); }
// (b) every row truncated to K/2: 2x inflated GB/s, every y written, output looks entirely plausible.
static void matvec_ktrunc(const float*W,const float*x,float*y,int out,int in){
    OMP_PFOR for(int o=0;o<out;o++) y[o]=dotf(W+(size_t)o*in,x,in/2); }

// ---- float64 reference + error metrics (gemv_bench.c lineage, extended) ----
// yr[o] = sum_k (double)W[o][k]*(double)x[k];  S[o] = sum_k |W[o][k]*x[k]|  (the conditioning of that row)
static void ref64d(const float*W,const float*x,double*yr,double*S,int M,int K,int stride,int*nchk){
    int n=0; for(int o=0;o<M;o+=stride){ double s=0,a=0; const float*w=W+(size_t)o*K;
        for(int k=0;k<K;k++){ double p=(double)w[k]*(double)x[k]; s+=p; a+=fabs(p); }
        yr[o]=s; S[o]=a; n++; } *nchk=n; }
typedef struct { double relerr, normerr; int arg_rel, arg_norm, nchk; } Err;
static Err cmp64(const float*y,const double*yr,const double*S,int M,int stride){
    Err e; e.relerr=0; e.normerr=0; e.arg_rel=-1; e.arg_norm=-1; e.nchk=0;
    for(int o=0;o<M;o+=stride){ double d=fabs((double)y[o]-yr[o]);
        double r=d/(fabs(yr[o])+1e-30), n=d/(S[o]+1e-30);
        if(r>e.relerr){ e.relerr=r; e.arg_rel=o; } if(n>e.normerr){ e.normerr=n; e.arg_norm=o; } e.nchk++; }
    return e; }
// row stride so one check costs at most ~3e8 double MACs
static int chk_stride(int M,int K){ double budget=3e8; double need=(double)M*(double)K; int s=1;
    if(need>budget) s=(int)(need/budget)+1; if(s<1)s=1; return s; }

// ---- timing core: identical estimator to engine.c run_gemv_sweep (min-of-reps), plus dispersion ----
typedef struct { double gbps, us, gflops, cv, gbps_mean; } Tm;
static Tm time_fp32(void(*kern)(const float*,const float*,float*,int,int),
                    const float*W,float*x,float*y,int M,int K,long npass,int reps,int nt){
    // kern==NULL selects the honest `matvec` through a DIRECT call, so the compiler inlines the timing loop
    // exactly the way engine.c run_gemv_sweep() does. Calling through the function pointer would block that
    // inlining and make this a different instrument from the one that produced the sec.1b curve.
    set_threads(nt);
    double bytes=(double)M*(double)K*4.0;
    if(kern) kern(W,x,y,M,K); else matvec(W,x,y,M,K);   // warm (engine.c does exactly one)
    double best=1e30,sum=0,sum2=0;
    for(int r=0;r<reps;r++){
        double t0=now_s();
        if(kern){ for(long p=0;p<npass;p++){ x[p%K]+=1e-9f; kern(W,x,y,M,K); } }
        else    { for(long p=0;p<npass;p++){ x[p%K]+=1e-9f; matvec(W,x,y,M,K); } }   // x-poison: forbids hoisting
        double dt=now_s()-t0; if(dt<best)best=dt;
        double g=bytes*npass/1e9/dt; sum+=g; sum2+=g*g;
    }
    Tm t; t.gbps=bytes*npass/1e9/best; t.us=best/npass*1e6;
    t.gflops=2.0*(double)M*(double)K*npass/1e9/best;
    t.gbps_mean=sum/reps; double var=sum2/reps-t.gbps_mean*t.gbps_mean; if(var<0)var=0;
    t.cv=reps>1?100.0*sqrt(var)/t.gbps_mean:-1.0;
    return t;
}
static long npass_for(size_t wb){ long n=(long)(2048UL*1048576UL/wb); if(n<16) n=16; return n; }   // engine.c: ~2 GB/window

static void fill_W(float*W,size_t n){ for(size_t i=0;i<n;i++) W[i]=0.001f*((long)(i%13)-6); }      // engine.c sweep fill
static void fill_x(float*x,int n){ for(int i=0;i<n;i++) x[i]=0.01f*((i%7)-3); }                     // engine.c sweep fill

// ================================ MODE: repro (the known-positive gate) ================================
static const long REPRO_MB[8]={4,8,16,24,32,48,64,96};
static const double PUB_T1[8]={30.5,31.3,27.2,27.8,24.9,24.5,24.9,23.3};
static const double PUB_T6[8]={187.0,185.0,134.4,60.5,55.7,45.5,45.3,36.5};
static int mode_repro(int reps,double tol_pct){
    const int nin=512; int fails=0, tailfails=0;
    printf("==== KNOWN-POSITIVE: reproduce PHASE64_BUDGET.md sec.1b proj-GEMV curve (in=%d, x L1-resident) ====\n",nin);
    printf("   acceptance: every t6 point within +/-%.0f%% of the published row; otherwise this harness is NOT the\n",tol_pct);
    printf("   instrument that produced the repo curve and no extrapolated point may be quoted.\n");
    printf("   sizeMB  rows   |  t1 GB/s  (pub) |  t6 GB/s  (pub) | t6/t1 | t6 dev%%  cv%% | verdict\n");
    float* x=xmalloc((size_t)nin*4); fill_x(x,nin);
    for(int si=0;si<8;si++){
        size_t wb=(size_t)REPRO_MB[si]*1048576; int out=(int)(wb/((size_t)nin*4)); wb=(size_t)out*nin*4;
        float* W=xmalloc(wb); fill_W(W,wb/4); float* y=xmalloc((size_t)out*4);
        long np=npass_for(wb);
        Tm a=time_fp32(NULL,W,x,y,out,nin,np,reps,1);
        Tm b=time_fp32(NULL,W,x,y,out,nin,np,reps,6);
        double dev=100.0*(b.gbps-PUB_T6[si])/PUB_T6[si];
        int ok=fabs(dev)<=tol_pct; if(!ok) fails++;
        if(si>=5&&!ok) tailfails++;
        printf("   %-6ld  %-6d |  %6.1f (%5.1f) |  %6.1f (%5.1f) | %.2fx |%+7.1f %5.1f | %s%s\n",
               REPRO_MB[si],out,a.gbps,PUB_T1[si],b.gbps,PUB_T6[si],b.gbps/a.gbps,dev,b.cv,
               ok?"PASS":"** OUT OF BAND **", si>=5?"  <-streamed tail":"");
        free(W); free(y);
    }
    free(x);
    printf("\n   GATE A (PRE-STATED, whole curve at +/-%.0f%%): %s - %d/8 t6 points outside band.\n",
           tol_pct,fails?"FAIL":"PASS",fails);
    printf("   GATE B (declared AFTER seeing Gate A - say so whenever quoting it; the streamed tail 48/64/96 MB,\n");
    printf("           the only region any donor organ occupies): %s - %d/3 outside band.\n",
           tailfails?"FAIL":"PASS",tailfails);
    printf("   Read both. Gate A failing in the 16-32 MB cache-transition region is a statement about the PUBLISHED\n");
    printf("   TABLE, not about this harness: `engine.c --gemv-sweep` - the routine that produced that table -\n");
    printf("   rebuilt and rerun on this machine today does not reproduce its own mid-region either, and that\n");
    printf("   region moves by ~5x on OpenMP thread PLACEMENT alone. The instrument-identity check that licenses\n");
    printf("   the extrapolated points is harness-vs-engine.c side by side today, not harness-vs-2026-07-12.\n");
    return fails;
}

// ================================ MODE: sweep (donor-scale extension) ================================
static const long SWEEP_MB[15]={4,8,16,24,32,48,64,96,128,192,256,384,512,768,1024};
static void mode_sweep(int reps,int docheck,int nmax){
    const int nin=512;
    printf("==== DONOR-SCALE proj-GEMV rate curve (same methodology as sec.1b, extended past 96 MB) ====\n");
    printf("   sizeMB  rows   |  t1 GB/s  t1 GF/s  cv%% |  t6 GB/s  t6 GF/s  cv%% | t6/t1 | t6 us/pass | maxrel  maxnorm (rows)\n");
    float* x=xmalloc((size_t)nin*4);
    for(int si=0;si<nmax;si++){
        size_t wb=(size_t)SWEEP_MB[si]*1048576; int out=(int)(wb/((size_t)nin*4)); wb=(size_t)out*nin*4;
        float* W=xmalloc(wb); fill_W(W,wb/4); float* y=xmalloc((size_t)out*4);
        fill_x(x,nin);
        double rel=-1,nrm=-1; int nchk=0;
        if(docheck){ int st=chk_stride(out,nin); double*yr=xmalloc((size_t)out*8),*S=xmalloc((size_t)out*8);
            ref64d(W,x,yr,S,out,nin,st,&nchk); set_threads(1); matvec(W,x,y,out,nin);
            Err e=cmp64(y,yr,S,out,st); rel=e.relerr; nrm=e.normerr; free(yr); free(S); }
        long np=npass_for(wb);
        Tm a=time_fp32(NULL,W,x,y,out,nin,np,reps,1);
        Tm b=time_fp32(NULL,W,x,y,out,nin,np,reps,6);
        printf("   %-6ld  %-6d |  %6.1f  %7.2f %5.1f |  %6.1f  %7.2f %5.1f | %.2fx | %10.1f | %.2e %.2e (%d)\n",
               SWEEP_MB[si],out,a.gbps,a.gflops,a.cv,b.gbps,b.gflops,b.cv,b.gbps/a.gbps,b.us,rel,nrm,nchk);
        free(W); free(y);
    }
    free(x);
}

// ================================ MODE: organs (real Qwen2.5-1.5B shapes) ================================
// From benchmarks/donor_adaptation/configs/Qwen__Qwen2.5-1.5B.json (the artefact is the authority):
//   D=1536  L=28  V=151936  ffn=8960  n_head=12  n_kv=2  head_dim=128  tie_word_embeddings=true
typedef struct { const char* nm; int M,K; } Organ;
static Organ ORGANS[8]={
    {"q_proj",   1536,1536},
    {"k_proj",    256,1536},
    {"v_proj",    256,1536},
    {"o_proj",   1536,1536},
    {"gate_proj",8960,1536},
    {"up_proj",  8960,1536},
    {"down_proj",1536,8960},
    {"lm_head",151936,1536},
};
static void mode_organs(int reps,int docheck){
    printf("==== REAL DONOR ORGAN SHAPES - Qwen/Qwen2.5-1.5B (D=1536 L=28 V=151936 ffn=8960 n_kv=2 tied) ====\n");
    printf("   RESIDENT = one copy, hot-looped (shape effect / upper bound). STREAMED = enough copies to clear L3,\n");
    printf("   cycled so every touch is cold (what a real decode sees). fp32 in both.\n");
    printf("   organ         MxK              MB |  RES t1/t6 GB/s |  STR t1/t6 GB/s | t6 GF/s | cv%% | maxrel  maxnorm\n");
    for(int i=0;i<8;i++){
        Organ o=ORGANS[i]; size_t wb=(size_t)o.M*o.K*4;
        float* x=xmalloc((size_t)o.K*4); fill_x(x,o.K);
        float* W=xmalloc(wb); fill_W(W,wb/4); float* y=xmalloc((size_t)o.M*4);
        double rel=-1,nrm=-1;
        if(docheck){ int st=chk_stride(o.M,o.K); int nchk; double*yr=xmalloc((size_t)o.M*8),*S=xmalloc((size_t)o.M*8);
            ref64d(W,x,yr,S,o.M,o.K,st,&nchk); set_threads(1); matvec(W,x,y,o.M,o.K);
            Err e=cmp64(y,yr,S,o.M,st); rel=e.relerr; nrm=e.normerr; free(yr); free(S); }
        long np=npass_for(wb);
        Tm r1=time_fp32(NULL,W,x,y,o.M,o.K,np,reps,1), r6=time_fp32(NULL,W,x,y,o.M,o.K,np,reps,6);
        // STREAMED: replicate until the working set clears aggregate L3 by >=8x (>=256 MB), cap ~1.5 GB
        int ncopy=1; while((double)wb*ncopy<268435456.0 && (double)wb*(ncopy+1)<=1610612736.0) ncopy++;
        double s1=r1.gbps,s6=r6.gbps;
        if(ncopy>1){
            float* P=xmalloc(wb*(size_t)ncopy); for(int c=0;c<ncopy;c++) memcpy((char*)P+wb*(size_t)c,W,wb);
            for(int ti=0;ti<2;ti++){ int nt=ti?6:1; set_threads(nt);
                long tp=(long)(2048UL*1048576UL/wb); if(tp<(long)ncopy*2) tp=(long)ncopy*2;
                matvec(P,x,y,o.M,o.K);
                double best=1e30;
                for(int rp=0;rp<reps;rp++){ double t0=now_s();
                    for(long p=0;p<tp;p++){ x[p%o.K]+=1e-9f; matvec((float*)((char*)P+wb*(size_t)(p%ncopy)),x,y,o.M,o.K); }
                    double dt=now_s()-t0; if(dt<best)best=dt; }
                double g=(double)wb*tp/1e9/best; if(ti) s6=g; else s1=g; }
            free(P);
        }
        printf("   %-11s %7dx%-7d %7.1f |  %6.1f / %6.1f |  %6.1f / %6.1f | %7.2f |%5.1f | %.2e %.2e%s\n",
               o.nm,o.M,o.K,wb/1048576.0,r1.gbps,r6.gbps,s1,s6,r6.gflops,r6.cv,rel,nrm,
               ncopy>1?"":"  (already >L3: STR==RES)");
        free(W); free(x); free(y);
    }
}

// ================================ MODE: fullstack (the number the verdict needs) ================================
// One decode token of Qwen2.5-1.5B's fp32 organs, in layer order: 28 x (q,k,v,o) then the head.
// Byte accounting must equal the hand-derivation in CONTROLLER_STAGE1_AUDIT.md F1: 1,550,057,472 B.
static void mode_fullstack(int reps){
    const int D=1536,L=28,V=151936,KVD=256;
    size_t per_layer=((size_t)D*D + (size_t)KVD*D + (size_t)KVD*D + (size_t)D*D)*4;
    size_t head=(size_t)V*D*4;
    size_t tot=per_layer*(size_t)L+head;
    printf("==== FULL PER-TOKEN fp32 PROJECTION STREAM - Qwen/Qwen2.5-1.5B, one decode token ====\n");
    printf("   layout: 28 x [q 1536x1536, k 256x1536, v 256x1536, o 1536x1536] then head 151936x1536, in layer order\n");
    printf("   byte accounting: %zu B  (CONTROLLER_STAGE1_AUDIT.md F1 hand-derivation = 1550057472 B) -> %s\n",
           tot, tot==1550057472UL?"EXACT MATCH":"** MISMATCH - accounting control FIRED **");
    float* P=xmalloc(tot); fill_W(P,tot/4);
    float* x=xmalloc((size_t)D*4); fill_x(x,D);
    float* y=xmalloc((size_t)V*4);
    for(int ti=0;ti<2;ti++){ int nt=ti?6:1; set_threads(nt);
        double best=1e30,sum=0,sum2=0;
        for(int rp=0;rp<reps+1;rp++){
            double t0=now_s();
            char* p=(char*)P;
            for(int l=0;l<L;l++){
                matvec((float*)p,x,y,D,D);   p+=(size_t)D*D*4;
                matvec((float*)p,x,y,KVD,D); p+=(size_t)KVD*D*4;
                matvec((float*)p,x,y,KVD,D); p+=(size_t)KVD*D*4;
                matvec((float*)p,x,y,D,D);   p+=(size_t)D*D*4;
                x[l%D]+=1e-9f;
            }
            matvec((float*)p,x,y,V,D);
            double dt=now_s()-t0;
            if(rp==0) continue;                                  // first pass = warm
            if(dt<best)best=dt; double g=(double)tot/1e9/dt; sum+=g; sum2+=g*g;
        }
        double mean=sum/reps; double var=sum2/reps-mean*mean; if(var<0)var=0;
        printf("   t%-2d | %7.2f GB/s | %8.3f ms/token | cv %5.2f%% | this organ alone would cap decode at %6.2f tok/s\n",
               nt,(double)tot/1e9/best,best*1e3,reps>1?100.0*sqrt(var)/mean:-1.0,1.0/best);
    }
    free(P); free(x); free(y);
}

// ================================ MODE: lut (ternary pshufb-LUT at donor dims) ================================
// The open question (Controller #2 Q3.2): is the dense LUT path compute-bound (rate flat in working-set size) or
// bandwidth-bound (rate falls with size like fp32)? Designed to come out either way: ONE fixed kernel shape,
// working set swept from cache-resident to >>L3. Flat => compute-bound. Declining => bandwidth-bound.
static void lut_pool_sweep(const char* tag,int M,int K,int reps,int iid){
    int Mpad=(M+31)&~31, T=K/2; size_t EB=(size_t)T*(size_t)Mpad;
    printf("   -- %s: block %dx%d -> codes %zu B (%.1f KB), tile-major, Mpad=%d T=%d --\n",tag,M,K,EB,EB/1024.0,Mpad,T);
    printf("      poolMB  blocks |  t1 GB/s   t1 us/blk |  t6 GB/s   t6 us/blk | t6/t1 | cv%%\n");
    int8_t* lut=xmalloc((size_t)T*16); int8_t* xq=xmalloc((size_t)K);
    for(int i=0;i<K;i++) xq[i]=(int8_t)((i%5)-2); build_lut_t3(xq,T,lut);
    int32_t* y=xmalloc((size_t)M*4);
    double pools[9]={0.0,1.0,4.0,16.0,32.0,64.0,128.0,256.0,512.0};   // 0.0 => single block (fully resident)
    for(int pi=0;pi<9;pi++){
        long nblk = pools[pi]==0.0 ? 1 : (long)((pools[pi]*1048576.0)/(double)EB);
        if(nblk<1) nblk=1;
        if(pi>0 && nblk==1 && pools[pi-1]!=0.0) continue;              // pool smaller than one block: skip duplicate row
        size_t pool=(size_t)nblk*EB;
        int8_t* P=xmalloc(pool);
        for(size_t i=0;i<pool;i++) P[i]=(int8_t)((i*2654435761u)%9u);   // valid ternary code indices [0,8]
        long touch=(long)(2048UL*1048576UL/EB); if(touch<nblk*2) touch=nblk*2; if(touch<64) touch=64;
        double g[2]={0,0},us[2]={0,0},cv=-1;
        for(int ti=0;ti<2;ti++){ int nt=ti?6:1; set_threads(nt);
            uint64_t rng=0x9e3779b97f4a7c15ULL;
            matvec_lut_full(P,lut,y,M,Mpad,T);
            double best=1e30,sum=0,sum2=0;
            for(int rp=0;rp<reps;rp++){ double t0=now_s();
                for(long t=0;t<touch;t++){ size_t b;
                    if(iid){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; b=(size_t)(rng%(uint64_t)nblk); } else b=(size_t)(t%nblk);
                    matvec_lut_full(P+b*EB,lut,y,M,Mpad,T); }
                double dt=now_s()-t0; if(dt<best)best=dt;
                double gg=(double)EB*touch/1e9/dt; sum+=gg; sum2+=gg*gg; }
            g[ti]=(double)EB*touch/1e9/best; us[ti]=best/touch*1e6;
            if(ti){ double m=sum/reps,v=sum2/reps-m*m; if(v<0)v=0; cv=reps>1?100.0*sqrt(v)/m:-1.0; } }
        printf("      %-7.1f %-6ld |  %6.2f  %10.3f |  %6.2f  %10.3f | %.2fx |%5.1f\n",
               pool/1048576.0,nblk,g[0],us[0],g[1],us[1],g[1]/g[0],cv);
        free(P);
    }
    free(lut); free(xq); free(y);
}
static void mode_lut(int reps,int iid){
    printf("==== TERNARY pshufb-LUT rate vs WORKING SET, at donor dims (open question: compute- or bandwidth-bound?) ====\n");
    printf("   read: FLAT across pool sizes => compute-bound (Controller #2 Q3.2 stands). DECLINING like the fp32\n");
    printf("   curve => bandwidth-bound and the ~16x compute-bound claim must be withdrawn.  gather=%s\n",iid?"i.i.d.":"sequential");
    lut_pool_sweep("sandbox anchor D=256 (48 KB expert, the 64.1b(2) shape)",384,256,reps,iid);
    lut_pool_sweep("donor MLP gate/up 8960x1536",8960,1536,reps,iid);
    lut_pool_sweep("donor MLP down    1536x8960",1536,8960,reps,iid);
    lut_pool_sweep("donor attn proj   1536x1536",1536,1536,reps,iid);
}
// second known-positive: engine.c run_expert_rate() verbatim (512 MB pool, 48 KB experts, i.i.d.) -> published 7.45 t1 / 17.0 t6
static void mode_lutrepro(void){
    const int M=384,Mpad=384,T=128; const size_t EB=(size_t)T*Mpad;
    const size_t POOL=512UL*1048576; long NE=(long)(POOL/EB);
    const int LAYERS=6,K=8; long per_tok=(long)LAYERS*K;
    int8_t* pool=xmalloc((size_t)NE*EB);
    for(size_t i=0;i<(size_t)NE*EB;i++) pool[i]=(int8_t)((i*2654435761u)%9u);
    int8_t lut[T*16]; int8_t xq[2*T]; for(int i=0;i<2*T;i++) xq[i]=(int8_t)((i%5)-2); build_lut_t3(xq,T,lut);
    int32_t* y=xmalloc((size_t)M*4);
    printf("==== KNOWN-POSITIVE #2: engine.c run_expert_rate() replica (%ld MB pool, %ld experts x %ld KB, i.i.d.) ====\n",
           (long)(POOL/1048576),NE,(long)(EB/1024));
    printf("   published 64.1b(2): 7.45 GB/s t1 -> 17.0 GB/s t6 (x2.29), 2.88 us/expert\n");
    printf("   threads |  GB/s   us/token  us/expert\n");
    for(int ti=0;ti<2;ti++){ int nt=ti?6:1; set_threads(nt);
        uint64_t rng=0x9e3779b97f4a7c15ULL; long ntok=500,touches=0;
        for(long e=0;e<per_tok;e++){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; matvec_lut_full(pool+(size_t)(rng%NE)*EB,lut,y,M,Mpad,T); }
        double t0=now_s();
        for(long tok=0;tok<ntok;tok++) for(long e=0;e<per_tok;e++){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17;
            matvec_lut_full(pool+(size_t)(rng%NE)*EB,lut,y,M,Mpad,T); touches++; }
        double dt=now_s()-t0;
        printf("   %-7d |  %5.2f  %8.1f  %8.3f\n",nt,(double)touches*EB/1e9/dt,dt/ntok*1e6,dt/touches*1e6);
    }
    free(pool); free(y);
}
// donor ternary full-stack: 28 x (gate,up,down) packed codes = 578,027,520 B (F1 hand-derivation, codes only)
static void mode_lutstack(int reps){
    const int L=28,D=1536,H=8960;
    int Mp_gu=(H+31)&~31, T_gu=D/2;  size_t B_gu=(size_t)T_gu*(size_t)Mp_gu;
    int Mp_dn=(D+31)&~31, T_dn=H/2;  size_t B_dn=(size_t)T_dn*(size_t)Mp_dn;
    size_t tot=(size_t)L*(2*B_gu+B_dn);
    printf("==== FULL PER-TOKEN TERNARY MLP STREAM - Qwen2.5-1.5B, 28 x (gate,up,down), codes only ====\n");
    printf("   byte accounting: %zu B  (F1 hand-derivation codes term = 578027520 B) -> %s\n",
           tot, tot==578027520UL?"EXACT MATCH":"** MISMATCH - accounting control FIRED **");
    int8_t* P=xmalloc(tot); for(size_t i=0;i<tot;i++) P[i]=(int8_t)((i*2654435761u)%9u);
    int8_t* lut_gu=xmalloc((size_t)T_gu*16); int8_t* lut_dn=xmalloc((size_t)T_dn*16);
    int8_t* xg=xmalloc(D); for(int i=0;i<D;i++) xg[i]=(int8_t)((i%5)-2); build_lut_t3(xg,T_gu,lut_gu);
    int8_t* xd=xmalloc(H); for(int i=0;i<H;i++) xd[i]=(int8_t)((i%5)-2); build_lut_t3(xd,T_dn,lut_dn);
    int32_t* y=xmalloc((size_t)H*4);
    for(int ti=0;ti<2;ti++){ int nt=ti?6:1; set_threads(nt);
        double best=1e30,sum=0,sum2=0;
        for(int rp=0;rp<reps+1;rp++){ double t0=now_s(); char* p=(char*)P;
            for(int l=0;l<L;l++){ matvec_lut_full((int8_t*)p,lut_gu,y,H,Mp_gu,T_gu); p+=B_gu;
                                  matvec_lut_full((int8_t*)p,lut_gu,y,H,Mp_gu,T_gu); p+=B_gu;
                                  matvec_lut_full((int8_t*)p,lut_dn,y,D,Mp_dn,T_dn); p+=B_dn; }
            double dt=now_s()-t0; if(rp==0) continue; if(dt<best)best=dt;
            double g=(double)tot/1e9/dt; sum+=g; sum2+=g*g; }
        double m=sum/reps,v=sum2/reps-m*m; if(v<0)v=0;
        printf("   t%-2d | %7.2f GB/s | %8.3f ms/token | cv %5.2f%%\n",nt,(double)tot/1e9/best,best*1e3,reps>1?100.0*sqrt(v)/m:-1.0);
    }
    free(P); free(lut_gu); free(lut_dn); free(xg); free(xd); free(y);
}

// ================================ MODE: correct (float64 correctness table) ================================
static void correct_one(const char* nm,int M,int K){
    size_t wb=(size_t)M*(size_t)K*4;
    float* W=xmalloc(wb); fill_W(W,wb/4);
    float* x=xmalloc((size_t)K*4); fill_x(x,K);
    float* y=xmalloc((size_t)M*4);
    int st=chk_stride(M,K),nchk;
    double* yr=xmalloc((size_t)M*8); double* S=xmalloc((size_t)M*8);
    ref64d(W,x,yr,S,M,K,st,&nchk);
    set_threads(1); matvec(W,x,y,M,K); Err e1=cmp64(y,yr,S,M,st);
    set_threads(6); matvec(W,x,y,M,K); Err e6=cmp64(y,yr,S,M,st);
    double bound=sqrt((double)K)*1.1920929e-7;   // sqrt(K)*eps_fp32 : textbook bound for the normalised error
    printf("   %-13s %7dx%-6d %8.1f MB | %7d/%-7d (str %5d) | %.2e %.2e | %.2e %.2e | %.2e | %s\n",
           nm,M,K,wb/1048576.0,nchk,M,st,e1.relerr,e1.normerr,e6.relerr,e6.normerr,bound,
           (e1.normerr<=bound&&e6.normerr<=bound)?"PASS":"** OVER BOUND **");
    free(W);free(x);free(y);free(yr);free(S);
}
static void mode_correct(void){
    printf("==== CORRECTNESS vs float64 REFERENCE (fp32 path) ====\n");
    printf("   rel  = max |y-yr|/(|yr|+1e-30)   -- blows up under cancellation; reported, NOT the gate\n");
    printf("   norm = max |y-yr|/(sum_k|W*x|)   -- condition-normalised; THIS is the gate\n");
    printf("   tolerance: norm <= sqrt(K)*eps_fp32. t1 and t6 both checked (row-partition must not change any row).\n");
    printf("   shape            MxK            size  | rows checked          | t1 rel   norm    | t6 rel   norm    | bound    | verdict\n");
    for(int i=0;i<8;i++) correct_one(ORGANS[i].nm,ORGANS[i].M,ORGANS[i].K);
    for(int i=0;i<15;i++){ char nm[32]; size_t wb=(size_t)SWEEP_MB[i]*1048576; int out=(int)(wb/2048);
        snprintf(nm,32,"sweep_%ldMB",SWEEP_MB[i]); correct_one(nm,out,512); }
    printf("\n==== CORRECTNESS of the ternary pshufb-LUT path (bit-exact vs TWO scalar integer references) ====\n");
    printf("   ref A = scalar over the packed codes (kernel definition); ref B = ref_t3 over the ORIGINAL ternary\n");
    printf("   weights (also validates the bc_tm packing). Both must be bit-identical to the SIMD kernel.\n");
    printf("   shape           MxK        codes MB | mismatch vs A / vs B / rows | verdict\n");
    int shapes[5][2]={{384,256},{1536,1536},{8960,1536},{1536,8960},{2048,512}};
    for(int s=0;s<5;s++){
        int M=shapes[s][0],K=shapes[s][1],Mpad=(M+31)&~31,T=K/2;
        int8_t* Wt=xmalloc((size_t)M*(size_t)K); srand(4242+M+K);
        for(size_t i=0;i<(size_t)M*(size_t)K;i++) Wt[i]=(int8_t)(rand()%3-1);
        int8_t* codes=xmalloc((size_t)T*(size_t)Mpad); bc_tm(Wt,M,K,Mpad,codes);
        int8_t* xq=xmalloc(K); for(int i=0;i<K;i++) xq[i]=(int8_t)((rand()%127)-63);
        int8_t* lut=xmalloc((size_t)T*16); build_lut_t3(xq,T,lut);
        int32_t* y=xmalloc((size_t)M*4); int32_t* yr=xmalloc((size_t)M*4); int32_t* yw=xmalloc((size_t)M*4);
        set_threads(6); matvec_lut_full(codes,lut,y,M,Mpad,T);
        ref_lut_scalar(codes,lut,yr,M,Mpad,T);
        ref_t3(Wt,xq,yw,M,K);
        int bad=0,badw=0; for(int m=0;m<M;m++){ if(y[m]!=yr[m])bad++; if(y[m]!=yw[m])badw++; }
        printf("   %-13s %7dx%-6d %8.2f | %6d / %6d / %6d | %s\n",
               "lut",M,K,(double)T*(double)Mpad/1048576.0,bad,badw,M,(bad==0&&badw==0)?"BIT-EXACT PASS":"** FAIL **");
        free(Wt);free(codes);free(xq);free(lut);free(y);free(yr);free(yw);
    }
}

// ================================ MODE: control (planted controls, sec.6.3/6.4) ================================
static void mode_control(void){
    printf("==== PLANTED CONTROLS - the instrument must be shown to FIRE on known positives before its nulls count ====\n\n");
    const int M=1536,K=1536;
    size_t wb=(size_t)M*(size_t)K*4;
    float* W=xmalloc(wb); fill_W(W,wb/4);
    float* x=xmalloc((size_t)K*4); fill_x(x,K);
    float* y=xmalloc((size_t)M*4); float* y0=xmalloc((size_t)M*4);
    double* yr=xmalloc((size_t)M*8); double* S=xmalloc((size_t)M*8);
    double* yr2=xmalloc((size_t)M*8); double* S2=xmalloc((size_t)M*8);
    int nchk;
    ref64d(W,x,yr,S,M,K,1,&nchk);
    set_threads(1); matvec(W,x,y0,M,K); memcpy(y,y0,(size_t)M*4);
    Err base=cmp64(y0,yr,S,M,1);
    printf("C0  BASELINE (honest kernel, q_proj 1536x1536, %d rows checked)\n",nchk);
    printf("    max rel-err  = %.3e   (worst row %d)\n",base.relerr,base.arg_rel);
    printf("    max norm-err = %.3e   (worst row %d)   sqrt(K)*eps_fp32 = %.3e\n",base.normerr,base.arg_norm,sqrt((double)K)*1.1920929e-7);
    double fire=10.0*base.normerr;
    printf("    FIRE THRESHOLD for every control below := 10 x baseline norm-err = %.3e\n\n",fire);

    int R=M/2, C=K/2; while(W[(size_t)R*K+C]==0.0f) C++;   // synthetic fill has zeros at i%13==6
    float w0=W[(size_t)R*K+C];
    printf("C1  DOES THE INSTRUMENT SEE ONE ELEMENT AT ALL?  (target W[%d][%d] = %.9g, x[%d] = %.9g)\n",R,C,(double)w0,C,(double)x[C]);
    printf("    C1a NEGATIVE leg - nothing corrupted, float64 reference recomputed:\n");
    ref64d(W,x,yr2,S2,M,K,1,&nchk);
    { double dmax=0; for(int o=0;o<M;o++){ double d=fabs(yr2[o]-yr[o]); if(d>dmax)dmax=d; }
      printf("        max |ref64(pristine) - ref64(pristine)| over all %d rows = %.17g  -> %s\n",M,dmax,
             dmax==0.0?"exactly 0: reference deterministic (silent, as it must be)":"** NON-DETERMINISTIC REFERENCE **"); }
    W[(size_t)R*K+C]=nextafterf(w0,INFINITY);
    double dw=(double)W[(size_t)R*K+C]-(double)w0;
    printf("    C1b POSITIVE leg - W[%d][%d] moved by exactly ONE ULP (dw = %.17g):\n",R,C,dw);
    ref64d(W,x,yr2,S2,M,K,1,&nchk);
    { double obs=yr2[R]-yr[R], pred=dw*(double)x[C]; int other=0; double dmax=0;
      for(int o=0;o<M;o++){ if(o==R) continue; double d=fabs(yr2[o]-yr[o]); if(d!=0.0) other++; if(d>dmax)dmax=d; }
      double agree=fabs(obs-pred)/(fabs(pred)+1e-300);
      printf("        row %d observed delta = %.17g\n",R,obs);
      printf("               predicted dw*x = %.17g\n",pred);
      printf("               agreement %.3e relative -> %s\n",agree,
             (obs!=0.0&&agree<1e-6)?"FIRES, at exactly the predicted magnitude":"** DID NOT FIRE **");
      printf("        all other %d rows: %d changed, max delta %.17g -> %s\n",M-1,other,dmax,
             other==0?"silent everywhere else (no cross-talk)":"** LEAKED **"); }
    set_threads(1); matvec(W,x,y,M,K);
    { int diff=0; for(int o=0;o<M;o++) if(y[o]!=y0[o]) diff++;
      Err e=cmp64(y,yr,S,M,1);
      printf("    C1c the fp32 KERNEL under the same 1-ULP corruption: %d/%d output floats changed; norm-err %.3e vs threshold %.3e\n",
             diff,M,e.normerr,fire);
      if(e.normerr>fire) printf("        -> fp32 comparator FIRES on 1 ULP.\n");
      else printf("        -> fp32 comparator is BLIND to a 1-ULP single-element corruption. This is PHYSICS, not a defect:\n"
                  "           the induced delta (dw*x ~ 1e-11) sits orders below the fp32 accumulation noise of a K=1536 dot.\n"
                  "           The float64 reference (C1b) DOES see it, which is what proves every element of W is consumed.\n"
                  "           The minimal corruption the fp32 comparator can resolve is found by the ladder in C2.\n"); }
    W[(size_t)R*K+C]=w0;

    printf("\nC2  MINIMAL-SIGNIFICANT-CORRUPTION LADDER (fp32 kernel vs pristine float64 reference)\n");
    printf("    W[%d][%d] *= (1+delta), everything else untouched. Fire threshold = %.3e\n",R,C,fire);
    printf("    delta        rel-err     norm-err    verdict\n");
    double ladder[9]={0.0,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1.0,1e6};
    double first_fire=-1;
    for(int i=0;i<9;i++){
        W[(size_t)R*K+C]=(float)((double)w0*(1.0+ladder[i]));
        set_threads(1); matvec(W,x,y,M,K); Err e=cmp64(y,yr,S,M,1);
        int f=e.normerr>fire;
        if(f&&first_fire<0) first_fire=ladder[i];
        printf("    %-10.0e   %.3e   %.3e   %s\n",ladder[i],e.relerr,e.normerr,
               i==0?(f?"** FALSE POSITIVE **":"silent  (negative leg: delta=0 must NOT fire)"):(f?"FIRES":"silent"));
    }
    W[(size_t)R*K+C]=w0;
    printf("    -> minimal corruption of ONE element out of %d that this comparator resolves: delta = %.0e\n",M*K,first_fire);

    printf("\nC3  THE PLAUSIBLE ARTEFACT: a kernel that streams fewer bytes than the harness charges.\n");
    printf("    This is how a bandwidth bench dies - the GB/s inflates and nothing looks wrong on the surface.\n");
    printf("    kernel        bytes read   charged   inflation  rel-err     norm-err    caught?\n");
    { set_threads(1); matvec(W,x,y,M,K); Err e=cmp64(y,yr,S,M,1);
      printf("    honest        %6.1f MB   %6.1f MB   1.00x      %.3e   %.3e   %s\n",
             wb/1048576.0,wb/1048576.0,e.relerr,e.normerr,e.normerr>fire?"** FALSE POSITIVE **":"silent (correct)"); }
    { memset(y,0,(size_t)M*4); set_threads(1); matvec_halfrows(W,x,y,M,K); Err e=cmp64(y,yr,S,M,1);
      printf("    halfrows      %6.1f MB   %6.1f MB   2.00x      %.3e   %.3e   %s   (odd rows never computed)\n",
             wb/2097152.0,wb/1048576.0,e.relerr,e.normerr,e.normerr>fire?"CAUGHT":"** MISSED **"); }
    { memset(y,0,(size_t)M*4); set_threads(1); matvec_ktrunc(W,x,y,M,K); Err e=cmp64(y,yr,S,M,1);
      printf("    ktrunc        %6.1f MB   %6.1f MB   2.00x      %.3e   %.3e   %s   (every row summed over K/2 only;\n",
             wb/2097152.0,wb/1048576.0,e.relerr,e.normerr,e.normerr>fire?"CAUGHT":"** MISSED **");
      printf("                                                                            every output written, looks plausible)\n"); }
    { Tm h=time_fp32(NULL,W,x,y,M,K,64,1,6), k=time_fp32(matvec_ktrunc,W,x,y,M,K,64,1,6);
      printf("    and the reason it matters: honest %6.2f GB/s vs ktrunc %6.2f GB/s (%.2fx inflation on the SAME charged bytes)\n",
             h.gbps,k.gbps,k.gbps/h.gbps); }

    printf("\nC4  TERNARY LUT PATH - bit-exact comparator, both directions (minimal corruption here = 1 code unit)\n");
    { int Mm=1536,Kk=1536,Mpad=(Mm+31)&~31,T=Kk/2;
      int8_t* Wt=xmalloc((size_t)Mm*(size_t)Kk); srand(99);
      for(size_t i=0;i<(size_t)Mm*(size_t)Kk;i++) Wt[i]=(int8_t)(rand()%3-1);
      int8_t* codes=xmalloc((size_t)T*(size_t)Mpad); bc_tm(Wt,Mm,Kk,Mpad,codes);
      int8_t* xq=xmalloc(Kk); for(int i=0;i<Kk;i++) xq[i]=(int8_t)((rand()%127)-63);
      int8_t* lut=xmalloc((size_t)T*16); build_lut_t3(xq,T,lut);
      int32_t* yy=xmalloc((size_t)Mm*4); int32_t* rr=xmalloc((size_t)Mm*4);
      ref_lut_scalar(codes,lut,rr,Mm,Mpad,T);
      set_threads(6); matvec_lut_full(codes,lut,yy,Mm,Mpad,T);
      int bad=0; for(int m=0;m<Mm;m++) if(yy[m]!=rr[m]) bad++;
      printf("    C4a NEGATIVE : uncorrupted codes              -> %d/%d rows differ -> %s\n",bad,Mm,bad==0?"silent (bit-exact)":"** FAIL **");
      size_t ci=(size_t)(T/2)*(size_t)Mpad+(size_t)(Mm/2); int8_t c0=codes[ci];
      codes[ci]=(int8_t)(c0==8?7:c0+1);
      matvec_lut_full(codes,lut,yy,Mm,Mpad,T);
      bad=0; int firstrow=-1; for(int m=0;m<Mm;m++) if(yy[m]!=rr[m]){ bad++; if(firstrow<0)firstrow=m; }
      printf("    C4b MINIMAL  : ONE code byte %d->%d of %zu       -> %d/%d rows differ (row %d, expected %d) -> %s\n",
             c0,codes[ci],(size_t)T*(size_t)Mpad,bad,Mm,firstrow,Mm/2,(bad==1&&firstrow==Mm/2)?"FIRES, at exactly the right row":"** WRONG **");
      codes[ci]=(int8_t)(c0==0?8:0);
      matvec_lut_full(codes,lut,yy,Mm,Mpad,T);
      bad=0; for(int m=0;m<Mm;m++) if(yy[m]!=rr[m]) bad++;
      printf("    C4c GROSS    : same byte %d->%d                  -> %d/%d rows differ -> %s\n",
             c0,codes[ci],bad,Mm,bad>=1?"FIRES":"** MISSED **");
      codes[ci]=c0;
      free(Wt);free(codes);free(xq);free(lut);free(yy);free(rr); }

    printf("\nC5  BYTE ACCOUNTING (the harness must charge exactly what the donor model charges)\n");
    { const int D=1536,L=28,V=151936,KVD=256;
      size_t fp32=(((size_t)D*D+(size_t)KVD*D*2+(size_t)D*D)*4)*(size_t)L+(size_t)V*D*4;
      int Mp_gu=(8960+31)&~31,T_gu=D/2,Mp_dn=(D+31)&~31,T_dn=8960/2;
      size_t tern=(size_t)L*(2*(size_t)T_gu*(size_t)Mp_gu+(size_t)T_dn*(size_t)Mp_dn);
      printf("    fp32 proj+head / token  = %12zu B  vs CONTROLLER_STAGE1_AUDIT F1 = 1550057472 B -> %s\n",
             fp32,fp32==1550057472UL?"MATCH":"** MISMATCH **");
      printf("    ternary MLP codes/token = %12zu B  vs CONTROLLER_STAGE1_AUDIT F1 =  578027520 B -> %s\n",
             tern,tern==578027520UL?"MATCH":"** MISMATCH **");
      printf("    negative control on the SAME line: F1's total ternary term is 580206592 B because it INCLUDES\n");
      printf("    2179072 B of per-row fp32 scales this codes-only harness does not stream. %zu + 2179072 = %zu -> %s\n",
             tern,tern+2179072UL,(tern+2179072UL)==580206592UL?"MATCH; the gap is exactly the scales term (audit F2)":"** MISMATCH **"); }

    free(W);free(x);free(y);free(y0);free(yr);free(S);free(yr2);free(S2);
    printf("\nCONTROLS DONE. A null from this harness is worth something only because C1b/C2/C3/C4b fired on known\n");
    printf("positives and C1a/C2(delta=0)/C3(honest)/C4a stayed silent on known negatives.\n");
}

// ================================ MODE: mlpint (ENGINE-INTEGRATED dense ternary MLP) ==========
// THE MEASUREMENT THAT DECIDES THE GATE.  For Qwen2.5-1.5B the proj term is effectively a point
// ([41.48..40.70] ms) while the ternary MLP term spans [20.99..50.91] ms -- a 30 ms bracket against
// a 2.13 ms margin.  Its low end is engine-integrated at D=256; its high end is kernel-pure at
// donor D.  DIFFERENT QUANTITIES.  What is missing is the INTEGRATED figure at donor D.
//
// Copied operation-for-operation from engine.c mlp_dense() (LUT path), parameterised by (D,HID,L).
// Integrated = it carries everything the kernel-pure number omits: int8 activation quantisation,
// the LUT build, per-row fp32 scale multiplies, the dReLU gate, and the second quantise + LUT
// build before the down matvec.  Sec.8's kernel-pure figure counts ONLY matvec_lut_full.
#define AQ_DONOR 63
static float quant_i8(const float* x,int n,int8_t* xq){ float amax=0; for(int i=0;i<n;i++){ float a=fabsf(x[i]); if(a>amax)amax=a; }
    if(amax==0.0f){ memset(xq,0,n); return 0.0f; } float scale=amax/(float)AQ_DONOR,inv=1.0f/scale;
    for(int i=0;i<n;i++){ int v=(int)lrintf(x[i]*inv); if(v>AQ_DONOR)v=AQ_DONOR; if(v<-AQ_DONOR)v=-AQ_DONOR; xq[i]=(int8_t)v; } return scale; }
static inline float reluf(float x){ return x>0.0f?x:0.0f; }
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

typedef struct {
    int D,HID,L,Mpg,Mpd,TUP,TDN;
    int8_t **gate_tm,**up_tm,**down_tm,**up_rm;
    float **gate_sc,**up_sc,**down_sc;
    int8_t *xqb,*lutb; int32_t *Sb; float *gh,*uh; int *act;
    size_t codes_layer, scales_layer;
} MlpStack;

static void mlp_alloc(MlpStack* S,int D,int HID,int L,int need_rm){
    S->D=D; S->HID=HID; S->L=L;
    S->Mpg=(HID+31)&~31; S->Mpd=(D+31)&~31; S->TUP=D/2; S->TDN=HID/2;
    S->codes_layer=(size_t)S->TUP*S->Mpg*2 + (size_t)S->TDN*S->Mpd;
    S->scales_layer=(size_t)(HID+HID+D)*4;
    S->gate_tm=xmalloc(L*sizeof(void*)); S->up_tm=xmalloc(L*sizeof(void*));
    S->down_tm=xmalloc(L*sizeof(void*)); S->up_rm=xmalloc(L*sizeof(void*));
    S->gate_sc=xmalloc(L*sizeof(void*)); S->up_sc=xmalloc(L*sizeof(void*)); S->down_sc=xmalloc(L*sizeof(void*));
    uint64_t r=0x243f6a8885a308d3ULL;
    for(int l=0;l<L;l++){
        S->gate_tm[l]=xmalloc((size_t)S->TUP*S->Mpg); S->up_tm[l]=xmalloc((size_t)S->TUP*S->Mpg);
        S->down_tm[l]=xmalloc((size_t)S->TDN*S->Mpd);
        for(size_t i=0;i<(size_t)S->TUP*S->Mpg;i++){ r^=r<<13; r^=r>>7; r^=r<<17; S->gate_tm[l][i]=(int8_t)(r%9); }
        for(size_t i=0;i<(size_t)S->TUP*S->Mpg;i++){ r^=r<<13; r^=r>>7; r^=r<<17; S->up_tm[l][i]=(int8_t)(r%9); }
        for(size_t i=0;i<(size_t)S->TDN*S->Mpd;i++){ r^=r<<13; r^=r>>7; r^=r<<17; S->down_tm[l][i]=(int8_t)(r%9); }
        S->up_rm[l]=NULL;
        if(need_rm){ S->up_rm[l]=xmalloc((size_t)HID*S->TUP);
            for(size_t i=0;i<(size_t)HID*S->TUP;i++){ r^=r<<13; r^=r>>7; r^=r<<17; S->up_rm[l][i]=(int8_t)(r%9); } }
        S->gate_sc[l]=xmalloc((size_t)HID*4); S->up_sc[l]=xmalloc((size_t)HID*4); S->down_sc[l]=xmalloc((size_t)D*4);
        for(int i=0;i<HID;i++){ S->gate_sc[l][i]=0.01f+0.0001f*(i%37); S->up_sc[l][i]=0.012f+0.0001f*(i%29); }
        for(int i=0;i<D;i++) S->down_sc[l][i]=0.008f+0.0001f*(i%41);
    }
    int mx=HID>D?HID:D;
    S->xqb=xmalloc(mx); S->lutb=xmalloc((size_t)(S->TDN>S->TUP?S->TDN:S->TUP)*16);
    S->Sb=xmalloc((size_t)mx*4); S->gh=xmalloc((size_t)mx*4); S->uh=xmalloc((size_t)mx*4);
    S->act=xmalloc((size_t)S->TDN*sizeof(int));
}

// engine.c mlp_dense(), LUT path, operation for operation.
static void mlp_dense_int(MlpStack* S,int l,const float* xn,float* out,int skip){
    int HID=S->HID,D=S->D;
    float sa=quant_i8(xn,D,S->xqb); build_lut_t3(S->xqb,S->TUP,S->lutb);
    matvec_lut_full(S->gate_tm[l],S->lutb,S->Sb,HID,S->Mpg,S->TUP);
    for(int i=0;i<HID;i++) S->gh[i]=(float)S->Sb[i]*sa*S->gate_sc[l][i];
    if(!skip){
        matvec_lut_full(S->up_tm[l],S->lutb,S->Sb,HID,S->Mpg,S->TUP);
        for(int i=0;i<HID;i++) S->uh[i]=(float)S->Sb[i]*sa*S->up_sc[l][i];
        for(int i=0;i<HID;i++) S->gh[i]=reluf(S->gh[i])*reluf(S->uh[i]);
    } else {
        for(int i=0;i<HID;i++){ if(S->gh[i]>0.0f){ const int8_t* cr=S->up_rm[l]+(size_t)i*S->TUP; int acc=0;
                for(int t=0;t<S->TUP;t++) acc+=S->lutb[t*16+cr[t]]; float u=(float)acc*sa*S->up_sc[l][i];
                S->gh[i]=S->gh[i]*reluf(u); } else S->gh[i]=0.0f; }
    }
    float sh=quant_i8(S->gh,HID,S->xqb); build_lut_t3(S->xqb,S->TDN,S->lutb);
    if(!skip) matvec_lut_full(S->down_tm[l],S->lutb,S->Sb,D,S->Mpd,S->TDN);
    else { int na=0; for(int t=0;t<S->TDN;t++) if(S->xqb[2*t]||S->xqb[2*t+1]) S->act[na++]=t;
        matvec_lut_tileskip(S->down_tm[l],S->lutb,S->Sb,D,S->Mpd,S->act,na); }
    for(int i=0;i<D;i++) out[i]=(float)S->Sb[i]*sh*S->down_sc[l][i];
}

// float64 reference for the WHOLE integrated block: scalar decode of the codes, double elementwise.
static void mlp_dense_ref64(MlpStack* S,int l,const float* xn,double* out){
    int HID=S->HID,D=S->D;
    int8_t* xq=xmalloc(HID>D?HID:D);
    float sa=quant_i8(xn,D,xq);
    double* gh=xmalloc((size_t)HID*8);
    for(int m=0;m<HID;m++){ long g=0,u=0;
        for(int t=0;t<S->TUP;t++){ int cg=S->gate_tm[l][(size_t)t*S->Mpg+m], cu=S->up_tm[l][(size_t)t*S->Mpg+m];
            int x0=xq[2*t],x1=xq[2*t+1];
            g+=(long)(cg/3-1)*x0+(long)(cg%3-1)*x1; u+=(long)(cu/3-1)*x0+(long)(cu%3-1)*x1; }
        double gv=(double)g*(double)sa*(double)S->gate_sc[l][m];
        double uv=(double)u*(double)sa*(double)S->up_sc[l][m];
        gh[m]=(gv>0?gv:0.0)*(uv>0?uv:0.0);
    }
    float* ghf=xmalloc((size_t)HID*4); for(int i=0;i<HID;i++) ghf[i]=(float)gh[i];
    float sh=quant_i8(ghf,HID,xq);
    for(int d=0;d<D;d++){ long s=0;
        for(int t=0;t<S->TDN;t++){ int c=S->down_tm[l][(size_t)t*S->Mpd+d]; int x0=xq[2*t],x1=xq[2*t+1];
            s+=(long)(c/3-1)*x0+(long)(c%3-1)*x1; }
        out[d]=(double)s*(double)sh*(double)S->down_sc[l][d]; }
    free(xq); free(gh); free(ghf);
}

static void mlp_free(MlpStack* S){
    for(int l=0;l<S->L;l++){ free(S->gate_tm[l]); free(S->up_tm[l]); free(S->down_tm[l]);
        if(S->up_rm[l]) free(S->up_rm[l]); free(S->gate_sc[l]); free(S->up_sc[l]); free(S->down_sc[l]); }
    free(S->gate_tm); free(S->up_tm); free(S->down_tm); free(S->up_rm);
    free(S->gate_sc); free(S->up_sc); free(S->down_sc);
    free(S->xqb); free(S->lutb); free(S->Sb); free(S->gh); free(S->uh); free(S->act);
}

// time L layers = one token of dense MLP, exactly the engine's T->mlp bucket (rmsnorm excluded,
// as engine.c excludes it -- see forward_token lines 374-379).
static void mlp_time(MlpStack* S,int skip,int reps,int nt,double* us_tok,double* cv){
    set_threads(nt);
    float* xn=xmalloc((size_t)S->D*4); for(int i=0;i<S->D;i++) xn[i]=0.05f*((i%11)-5);
    float* out=xmalloc((size_t)S->D*4);
    for(int l=0;l<S->L;l++) mlp_dense_int(S,l,xn,out,skip);
    long ntok=(long)(2.0e9/(double)(S->codes_layer*(size_t)S->L)); if(ntok<8) ntok=8; if(ntok>3000) ntok=3000;
    // DIAGNOSTIC ONLY (control-1 variance investigation, not part of the pre-registered brief):
    // when GEMV_D5_DEBUG_REPS is set, print each rep's own rate to stderr so a warm-up/thermal trend can be
    // told apart from unstructured jitter. Does not alter *us_tok / *cv or any control's pass/fail logic.
    const char* dbg=getenv("GEMV_D5_DEBUG_REPS");
    double best=1e30,sum=0,sum2=0;
    for(int r=0;r<reps;r++){
        double t0=now_s();
        for(long k=0;k<ntok;k++){ xn[k%S->D]+=1e-7f; for(int l=0;l<S->L;l++) mlp_dense_int(S,l,xn,out,skip); }
        double dt=now_s()-t0; if(dt<best)best=dt;
        double u=dt/ntok*1e6; sum+=u; sum2+=u*u;
        if(dbg){ double g=(double)S->codes_layer*(size_t)S->L/1e3/u;
            fprintf(stderr,"DBGREP,nt=%d,rep=%d,us_tok=%.4f,gbps=%.4f\n",nt,r,u,g); }
    }
    double m=sum/reps,v=sum2/reps-m*m; if(v<0)v=0;
    *us_tok=best/ntok*1e6; *cv=reps>1?100.0*sqrt(v)/m:-1.0;
    free(xn); free(out);
}

static void mlpint_row(const char* tag,int D,int HID,int L,int skip,int reps,double tgt1,double tgt6){
    MlpStack S; mlp_alloc(&S,D,HID,L,skip);
    size_t codes=S.codes_layer*(size_t)L, scales=S.scales_layer*(size_t)L;
    double u1,c1,u6,c6;
    mlp_time(&S,skip,reps,1,&u1,&c1);
    mlp_time(&S,skip,reps,6,&u6,&c6);
    printf("   %-22s D=%-5d HID=%-5d L=%-3d skip=%d | %9.1f %5.1f | %9.1f %5.1f | %6.2f %6.2f | %6.2f",
           tag,D,HID,L,skip,u1,c1,u6,c6,(double)codes/1e3/u1,(double)codes/1e3/u6,
           (double)(codes+scales)/1e3/u6);
    if(tgt6>0) printf(" | pub %.0f/%.0f dev %+.1f%%/%+.1f%%",tgt1,tgt6,100*(u1-tgt1)/tgt1,100*(u6-tgt6)/tgt6);
    printf("\n");
    mlp_free(&S);
}

static void mode_mlpint(int reps){
    printf("==== ENGINE-INTEGRATED dense ternary MLP -- engine.c mlp_dense(), operation for operation ====\n");
    printf("   INTEGRATED = int8 activation quant + LUT build + matvecs + per-row fp32 scales + dReLU\n");
    printf("   + a second quant/LUT build before down.  Sec.8's kernel-pure figure counts ONLY\n");
    printf("   matvec_lut_full.  The gap between them AT DONOR D is the whole open bracket.\n\n");
    printf("   KNOWN-POSITIVE: PHASE64_BUDGET sec.1 compute-floor decomposition gives the 8.3M sandbox\n");
    printf("   dense LUT-MLP at 313 us t1 -> 207 us t6 (L=6, D=256, MLP_HID=1024).  If this harness\n");
    printf("   does not land there, NO donor-D number below may be quoted.\n\n");
    printf("   %-22s %-7s %-9s %-6s | %9s %5s | %9s %5s | %6s %6s | %6s\n",
           "config","D","HID","L skip","t1 us/tok","cv%","t6 us/tok","cv%","GB/s1","GB/s6","+scale");
    mlpint_row("SANDBOX known-pos",256,1024,6,0,reps,313.0,207.0);
    mlpint_row("SANDBOX known-pos",256,1024,6,1,reps,313.0,207.0);
    printf("\n");
    mlpint_row("DONOR Qwen2.5-1.5B",1536,8960,28,0,reps,0,0);
    mlpint_row("DONOR Qwen2.5-1.5B",1536,8960,28,1,reps,0,0);
    printf("\n   skip=0 (E2) is the DONOR-RELEVANT config: Qwen2.5-1.5B is SiLU-gated, so the engine's\n");
    printf("   dReLU row-skip (E3) is NOT free -- it is a quality change P61/probe-2 priced separately.\n");
    printf("   skip=1 is reported only as the upside if a dReLU conversion is accepted.\n");
    printf("   GB/s columns: codes-only (comparable with the 11.398 anchor) and codes+scales.\n");
}

static void mode_mlpctl(void){
    printf("==== PLANTED CONTROLS for the integrated dense-MLP block ====\n\n");
    int D=1536,HID=8960,L=1;
    MlpStack S; mlp_alloc(&S,D,HID,L,0);
    float* xn=xmalloc((size_t)D*4); for(int i=0;i<D;i++) xn[i]=0.05f*((i%11)-5);
    float* out=xmalloc((size_t)D*4); float* out0=xmalloc((size_t)D*4);
    double* ref=xmalloc((size_t)D*8);
    set_threads(6);
    mlp_dense_ref64(&S,0,xn,ref);
    mlp_dense_int(&S,0,xn,out0,0);
    float* ghp=xmalloc((size_t)HID*4); memcpy(ghp,S.gh,(size_t)HID*4);   // post-dReLU activations
    double base=0,denom=0;
    for(int i=0;i<D;i++){ double d=fabs((double)out0[i]-ref[i]); double a=fabs(ref[i]);
        if(a>denom)denom=a; if(d>base)base=d; }
    printf("M1 ACCURACY GATE -- whole integrated path (fp32) vs float64 reference, D=1536 HID=8960\n");
    printf("   max abs-err %.3e over max|ref| %.3e -> relative %.3e   -> %s\n\n",
           base,denom,base/denom,(base/denom)<1e-6?"PASS":"** OVER BOUND **");

    // How many rows did dReLU zero?  This is why the FIRST version of this control did not fire:
    // it corrupted a row the gate had already zeroed, so nothing could propagate.
    int nact=0,argmax=0; float gmax=0;
    for(int i=0;i<HID;i++){ if(ghp[i]>0.0f) nact++; if(ghp[i]>gmax){ gmax=ghp[i]; argmax=i; } }
    printf("M2 INSTRUMENT SENSITIVITY -- corrupted fp32 output vs PRISTINE fp32 output, BIT-EXACT\n");
    printf("   (the M1 comparator cannot serve here: its own fp32-vs-f64 rounding floor of %.1e masks\n",base/denom);
    printf("    any small corruption.  Bit-exact self-comparison has no floor and is the right probe.)\n");
    printf("   dReLU left %d of %d rows active (%.1f%%); scale-setting row = %d (gh=%.6g).\n",
           nact,HID,100.0*nact/HID,argmax,(double)gmax);
    printf("   Corrupting a row dReLU has ZEROED is unobservable BY CONSTRUCTION -- so every control\n");
    printf("   below targets row %d, which is active and sets the re-quantisation scale.\n\n",argmax);

    int m = argmax;
    // ---- ladder over the NUMBER of corrupted ternary codes in that one active row ----
    printf("   ladder A: N ternary codes corrupted inside gate row %d (of %d codes in the row)\n",m,S.TUP);
    int Ns[8]={0,1,2,4,16,64,256,768}; int firstN=-1;
    int8_t* save=xmalloc(S.TUP);
    for(int t=0;t<S.TUP;t++) save[t]=S.gate_tm[0][(size_t)t*S.Mpg+m];
    for(int k=0;k<8;k++){
        for(int t=0;t<S.TUP;t++) S.gate_tm[0][(size_t)t*S.Mpg+m]=save[t];
        for(int t=0;t<Ns[k]&&t<S.TUP;t++){ int8_t c=save[t]; S.gate_tm[0][(size_t)t*S.Mpg+m]=(int8_t)(c==8?7:c+1); }
        mlp_dense_int(&S,0,xn,out,0);
        int nd=0; double mx=0;
        for(int i=0;i<D;i++) if(out[i]!=out0[i]){ nd++; double r=fabs((double)out[i]-(double)out0[i])/(fabs((double)out0[i])+1e-30); if(r>mx)mx=r; }
        if(nd&&firstN<0) firstN=Ns[k];
        printf("     N=%-4d -> %4d/%d outputs changed, max rel change %.3e -> %s\n",Ns[k],nd,D,mx,
               Ns[k]==0?(nd?"** FALSE POSITIVE **":"silent (negative leg, as required)"):(nd?"FIRES":"silent"));
    }
    for(int t=0;t<S.TUP;t++) S.gate_tm[0][(size_t)t*S.Mpg+m]=save[t];
    printf("     -> minimal corruption resolved: %d code(s) of %d in the row (%d of %zu in the matrix)\n\n",
           firstN,S.TUP,firstN,(size_t)S.TUP*S.Mpg);

    // ---- ladder over one per-row fp32 SCALE on the same active row ----
    printf("   ladder B: the per-row fp32 scale gate_sc[%d] (the fp32 half of the integrated path)\n",m);
    float s0=S.gate_sc[0][m];
    S.gate_sc[0][m]=nextafterf(s0,INFINITY);
    mlp_dense_int(&S,0,xn,out,0);
    { int nd=0; for(int i=0;i<D;i++) if(out[i]!=out0[i]) nd++;
      printf("     1 ULP  (%.9g -> %.9g) -> %4d/%d outputs changed -> %s\n",
             (double)s0,(double)S.gate_sc[0][m],nd,D,nd?"FIRES":"silent"); }
    double lad[7]={0.0,1e-7,1e-5,1e-3,1e-1,1.0,1e6}; double firstD=-1;
    for(int i=0;i<7;i++){ S.gate_sc[0][m]=(float)((double)s0*(1.0+lad[i]));
        mlp_dense_int(&S,0,xn,out,0);
        int nd=0; double mx=0;
        for(int j=0;j<D;j++) if(out[j]!=out0[j]){ nd++; double r=fabs((double)out[j]-(double)out0[j])/(fabs((double)out0[j])+1e-30); if(r>mx)mx=r; }
        if(nd&&firstD<0&&lad[i]>0) firstD=lad[i];
        printf("     delta %-8.0e -> %4d/%d outputs changed, max rel %.3e -> %s\n",lad[i],nd,D,mx,
               lad[i]==0.0?(nd?"** FALSE POSITIVE **":"silent (negative leg, as required)"):(nd?"FIRES":"silent")); }
    S.gate_sc[0][m]=s0;
    printf("     -> minimal scale corruption resolved: delta = %.0e\n\n",firstD);

    // ---- the plausible artefact: a block that silently skips work ----
    printf("M3 THE PLAUSIBLE ARTEFACT -- a block that streams less than it is charged for\n");
    { int8_t* sv=xmalloc((size_t)S.TDN*16);
      memcpy(sv,S.lutb,(size_t)S.TDN*16);
      // zero the down matrix's entire first half of tiles: half the down codes never influence y
      printf("     down-matrix half-tile zeroing (mimics a kernel that reads half the codes):\n");
      size_t half=(size_t)(S.TDN/2)*S.Mpd;
      int8_t* bak=xmalloc(half); memcpy(bak,S.down_tm[0],half);
      memset(S.down_tm[0],4,half);                      // code 4 == (0,0), a VALID but wrong code
      mlp_dense_int(&S,0,xn,out,0);
      int nd=0; for(int i=0;i<D;i++) if(out[i]!=out0[i]) nd++;
      printf("       %4d/%d outputs changed -> %s\n",nd,D,nd?"CAUGHT":"** MISSED **");
      memcpy(S.down_tm[0],bak,half); free(bak); free(sv); }

    printf("\nM4 BYTE ACCOUNTING\n");
    { MlpStack A; mlp_alloc(&A,256,1024,6,0);
      printf("   sandbox codes/token  = %10zu B vs engine.c-derived 2359296 B -> %s\n",
             A.codes_layer*6, A.codes_layer*6==2359296UL?"MATCH":"** MISMATCH **");
      printf("   sandbox scales/token = %10zu B vs audit F2        55296 B -> %s\n",
             A.scales_layer*6, A.scales_layer*6==55296UL?"MATCH":"** MISMATCH **");
      mlp_free(&A); }
    { MlpStack B; mlp_alloc(&B,1536,8960,28,0);
      printf("   donor   codes/token  = %10zu B vs F1 hand-deriv 578027520 B -> %s\n",
             B.codes_layer*28, B.codes_layer*28==578027520UL?"MATCH":"** MISMATCH **");
      printf("   donor   scales/token = %10zu B vs F1 hand-deriv   2179072 B -> %s\n",
             B.scales_layer*28, B.scales_layer*28==2179072UL?"MATCH":"** MISMATCH **");
      mlp_free(&B); }
    free(xn); free(out); free(out0); free(ref); free(ghp); free(save); mlp_free(&S);
    printf("\nCONTROLS DONE.\n");
}

// ================================ MODE: d5 (BRIEF_D5_LUT_RATE_AT_DONOR_WIDTH) ================================
// rate(D, threads) for the engine-integrated ternary MLP path, at donor projection widths, plus the four
// planted controls from the brief sec.5. Reuses, verbatim, the apparatus that produced 21.25 GB/s (MlpStack /
// mlp_alloc / mlp_time from the mlpint mode above) and the apparatus that produced the 8.4x/2.4x LUT working-set
// curve (matvec_lut_full / build_lut_t3, in a new single-point helper below that duplicates lut_pool_sweep's
// inner loop so callers can capture numbers rather than only print them -- lut_pool_sweep itself is untouched).
//
// Output contract: every timed line is also emitted as a CSV row prefixed "D5CSV," (one line per point) so a
// post-processing step can build the JSON deliverable without hand-transcription. Column layouts differ by
// section and are documented at each printf call site; the header line below records both said layouts once.
#define L3_BYTES (16u*1048576u)   // the project's banked L3 cliff (~16 MB exact) -- docs project_probe3_cache

// -- capture-and-print helper for the mlpint (engine-integrated dense ternary MLP) apparatus --
static void mlp_point(const char* section,const char* tag,int D,int HID,int L,int skip,int reps,int nt,
                       double* out_gbps_codes,double* out_gbps_scaled,double* out_us,double* out_cv,size_t* out_ws){
    MlpStack S; mlp_alloc(&S,D,HID,L,skip);
    size_t codes=S.codes_layer*(size_t)L, scales=S.scales_layer*(size_t)L;
    double u,cv; mlp_time(&S,skip,reps,nt,&u,&cv);
    double gbps=(double)codes/1e3/u, gbps_sc=(double)(codes+scales)/1e3/u;
    int l3=(codes<=L3_BYTES)?1:0;
    // D5CSV,section,tag,D,HID,L,threads,working_set_bytes,l3_resident,gbps_codes,us_per_tok,cv_pct,gbps_codes+scales
    printf("D5CSV,%s,%s,%d,%d,%d,%d,%zu,%d,%.4f,%.4f,%.2f,%.4f\n",
           section,tag,D,HID,L,nt,codes,l3,gbps,u,cv,gbps_sc);
    if(out_gbps_codes)*out_gbps_codes=gbps; if(out_gbps_scaled)*out_gbps_scaled=gbps_sc;
    if(out_us)*out_us=u; if(out_cv)*out_cv=cv; if(out_ws)*out_ws=codes;
    mlp_free(&S);
}

// -- capture-and-print helper for the ternary LUT working-set sweep (lut_pool_sweep's inner loop, duplicated
//    so it can hand numbers back to the caller; the printed TABLE mode (`lut`) is untouched) --
static void lut_one_point(const char* section,const char* tag,int M,int K,double poolMB,int reps,int iid,double g_out[2]){
    int Mpad=(M+31)&~31, T=K/2; size_t EB=(size_t)T*(size_t)Mpad;
    int8_t* lut=xmalloc((size_t)T*16); int8_t* xq=xmalloc((size_t)K);
    for(int i=0;i<K;i++) xq[i]=(int8_t)((i%5)-2); build_lut_t3(xq,T,lut);
    int32_t* y=xmalloc((size_t)M*4);
    long nblk = poolMB<=0.0 ? 1 : (long)((poolMB*1048576.0)/(double)EB);
    if(nblk<1) nblk=1;
    size_t pool=(size_t)nblk*EB;
    int8_t* P=xmalloc(pool);
    for(size_t i=0;i<pool;i++) P[i]=(int8_t)((i*2654435761u)%9u);
    long touch=(long)(2048UL*1048576UL/EB); if(touch<nblk*2) touch=nblk*2; if(touch<64) touch=64;
    int l3=(pool<=L3_BYTES)?1:0;
    for(int ti=0;ti<2;ti++){ int nt=ti?6:1; set_threads(nt);
        uint64_t rng=0x9e3779b97f4a7c15ULL;
        matvec_lut_full(P,lut,y,M,Mpad,T);   // warm
        double best=1e30,sum=0,sum2=0;
        for(int rp=0;rp<reps;rp++){ double t0=now_s();
            for(long t=0;t<touch;t++){ size_t b;
                if(iid){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; b=(size_t)(rng%(uint64_t)nblk); } else b=(size_t)(t%nblk);
                matvec_lut_full(P+b*EB,lut,y,M,Mpad,T); }
            double dt=now_s()-t0; if(dt<best)best=dt;
            double g=(double)EB*touch/1e9/dt; sum+=g; sum2+=g*g; }
        double gbps=(double)EB*touch/1e9/best, us=best/touch*1e6;
        double m=sum/reps,v=sum2/reps-m*m; if(v<0)v=0; double cv=reps>1?100.0*sqrt(v)/m:-1.0;
        // D5CSV,section,tag,M,K,0,threads,working_set_bytes(=pool),l3_resident,gbps,us_per_touch,cv_pct
        printf("D5CSV,%s,%s,%d,%d,0,%d,%zu,%d,%.4f,%.4f,%.2f\n",section,tag,M,K,nt,pool,l3,gbps,us,cv);
        if(g_out) g_out[ti]=gbps;
    }
    free(P); free(lut); free(xq); free(y);
}

static void d5_control1(int reps,int* passed,double* measured_t6){
    printf("---- CONTROL 1: reproduce the banked D=1536 point (21.25 +/- 0.36 GB/s), engine-integrated MLP ----\n");
    double u1,c1,u6,c6,g1c,g1s,g6c,g6s; size_t ws;
    mlp_point("control1","donor_D1536_HID8960_L28",1536,8960,28,0,reps,1,&g1c,&g1s,&u1,&c1,&ws);
    mlp_point("control1","donor_D1536_HID8960_L28",1536,8960,28,0,reps,6,&g6c,&g6s,&u6,&c6,&ws);
    double banked=21.25, tol=0.36;
    int pass = fabs(g6c-banked)<=tol;
    printf("D5VERDICT,control1,%s,measured_t6_gbps=%.3f,banked=%.2f+-%.2f,dev=%+.3f,cv=%.2f%%\n",
           pass?"PASS":"FAIL",g6c,banked,tol,g6c-banked,c6);
    if(!pass) printf("D5STOP: control 1 FAILED -- apparatus has drifted. Per brief sec.5.1, STOP; nothing else\n"
                      "  from this harness may be trusted until this is resolved.\n");
    *passed=pass; *measured_t6=g6c;
}

static void d5_lsensitivity(int reps){
    // NOT one of the four brief controls -- a supporting check for the design choice (L=2 in the D-sweep vs
    // L=28 for the real donor / control-1 point). If the integrated rate is L-independent once one layer's
    // working set already exceeds L3 (it does: 20.6 MB per layer at D=1536,HID=8960), L=2 is a valid, much
    // cheaper stand-in for the D-sweep. This is measured, not assumed.
    printf("---- SUPPORTING CHECK: L-sensitivity (not a brief-mandated control) ----\n");
    double g6_L2,g6_L28,us,cv; size_t ws;
    mlp_point("lsens","D1536_HID8960_L2",1536,8960,2,0,reps,6,&g6_L2,0,&us,&cv,&ws);
    mlp_point("lsens","D1536_HID8960_L28",1536,8960,28,0,reps,6,&g6_L28,0,&us,&cv,&ws);
    printf("D5NOTE,lsensitivity,L2_t6_gbps=%.3f,L28_t6_gbps=%.3f,ratio=%.3f\n",g6_L2,g6_L28,g6_L2/g6_L28);
}

static void d5_control23(int reps,int* c2_pass,int* c3_pass){
    printf("---- CONTROLS 2 & 3: known-positive slowdown (push working set >16MB) and speedup (L3-resident) ----\n");
    printf("     shape: donor attn proj 1536x1536 (block EB=1.125 MB), same kernel, only pool size varies.\n");
    double g0[2],g16[2],g32[2],g512[2];
    lut_one_point("control23","donor_attn_1536x1536",1536,1536,0.0,reps,1,g0);
    lut_one_point("control23","donor_attn_1536x1536",1536,1536,16.0,reps,1,g16);
    lut_one_point("control23","donor_attn_1536x1536",1536,1536,32.0,reps,1,g32);
    lut_one_point("control23","donor_attn_1536x1536",1536,1536,512.0,reps,1,g512);
    double resident_t6=g0[1], streamed_t6=g512[1];
    double ratio = resident_t6/streamed_t6;
    int c2 = ratio>=2.0;    // known-positive: forced past-L3 pool must drop the rate sharply
    int c3 = resident_t6>=50.0;  // known-positive: L3-resident must rise toward the ~100 GB/s regime
    printf("D5VERDICT,control2,%s,resident_t6=%.2f,streamed512_t6=%.2f,drop_ratio=%.2fx (need >=2.0x)\n",
           c2?"PASS":"FAIL",resident_t6,streamed_t6,ratio);
    printf("D5VERDICT,control3,%s,resident_t6=%.2f (need >=50, target ~100 per the banked L3 probe)\n",
           c3?"PASS":"FAIL",resident_t6);
    *c2_pass=c2; *c3_pass=c3;
}

static void d5_control4(int reps,int* passed){
    printf("---- CONTROL 4: fp32 streamed cross-check (banked 38.84 +/- 0.68, audited 39.87 +/- 0.09) ----\n");
    const int nin=512; float* x=xmalloc((size_t)nin*4); fill_x(x,nin);
    long sizesMB[4]={64,128,256,512};
    double sum=0,sum2=0; int n=0;
    for(int i=0;i<4;i++){
        size_t wb=(size_t)sizesMB[i]*1048576; int out=(int)(wb/((size_t)nin*4)); wb=(size_t)out*nin*4;
        float* W=xmalloc(wb); fill_W(W,wb/4); float* y=xmalloc((size_t)out*4);
        long np=npass_for(wb);
        Tm t6=time_fp32(NULL,W,x,y,out,nin,np,reps,6);
        int l3=(wb<=L3_BYTES)?1:0;
        // D5CSV,control4,tag,rows,in,0,threads,working_set_bytes,l3_resident,gbps,us,cv_pct
        printf("D5CSV,control4,fp32_%ldMB,%d,%d,0,6,%zu,%d,%.4f,%.4f,%.2f\n",sizesMB[i],out,nin,wb,l3,t6.gbps,t6.us,t6.cv);
        sum+=t6.gbps; sum2+=t6.gbps*t6.gbps; n++;
        free(W); free(y);
    }
    free(x);
    double mean=sum/n, var=sum2/n-mean*mean; if(var<0)var=0; double sd=sqrt(var);
    double banked=38.84, band=0.68;
    int pass = fabs(mean-banked)<=3.0*band;   // 3-sigma band on the banked figure; brief gives no numeric tol here
    printf("D5VERDICT,control4,%s,mean_t6_gbps=%.3f,sd=%.3f,banked=%.2f+-%.2f,audited=39.87+-0.09,tol=+-%.2f(3sigma)\n",
           pass?"PASS":"FAIL",mean,sd,banked,band,3.0*band);
    *passed=pass;
}

static const int D5_DS[5]={1536,2048,4096,5120,8192};

static void d5_dsweep(int reps,double out_integrated_t6[5]){
    printf("---- PART A: rate(D, threads), engine-integrated dense ternary MLP, ffn:D ratio fixed at 3.5 ----\n");
    printf("     (3.5 = the real Llama-3-70B-class ratio 28672/8192, so the D=8192 point below coincides with\n");
    printf("      the real donor organ measured in Part B. L held at 2 -- see the L-sensitivity check above:\n");
    printf("      every layer's working set already exceeds L3 at every D swept, so this is not a shortcut\n");
    printf("      that changes the regime, only one that changes runtime.)\n");
    for(int i=0;i<5;i++){
        int D=D5_DS[i]; int HID=(int)(3.5*D+0.5);
        char tag[32]; snprintf(tag,32,"D%d",D);
        double gc1,gs1,gc6,gs6,u1,u6,c1,c6; size_t ws;
        mlp_point("dsweep",tag,D,HID,2,0,reps,1,&gc1,&gs1,&u1,&c1,&ws);
        mlp_point("dsweep",tag,D,HID,2,0,reps,6,&gc6,&gs6,&u6,&c6,&ws);
        printf("D5NOTE,dsweep,D=%d,HID=%d,ws_MB=%.1f,l3_resident=%d,t1_gbps=%.2f,t6_gbps=%.2f,t6_cv=%.2f%%\n",
               D,HID,ws/1048576.0,(ws<=L3_BYTES)?1:0,gc1,gc6,c6);
        if(out_integrated_t6) out_integrated_t6[i]=gc6;
    }
}

// -- honest fp32 streamed-rate point: mode_organs' STREAMED replication logic, generalised and CSV-emitting.
//    If a single copy of the shape already exceeds the aggregate-L3 clearance target, timed directly (no
//    replication needed -- matches mode_organs' own "already >L3: STR==RES" note). Otherwise replicated past
//    256 MB (cap 1.5 GB) and cycled so every touch is cold, exactly as mode_organs' STREAMED column does. --
static void fp32_streamed_point(const char* section,const char* tag,int M,int K,int reps,double g_out[2]){
    size_t wb=(size_t)M*(size_t)K*4;
    float* x=xmalloc((size_t)K*4); fill_x(x,K);
    float* W=xmalloc(wb); fill_W(W,wb/4); float* y=xmalloc((size_t)M*4);
    int l3=(wb<=L3_BYTES)?1:0;
    int ncopy=1; while((double)wb*ncopy<268435456.0 && (double)wb*(ncopy+1)<=1610612736.0) ncopy++;
    if(ncopy==1){
        for(int ti=0;ti<2;ti++){ int nt=ti?6:1;
            Tm t=time_fp32(NULL,W,x,y,M,K,npass_for(wb),reps,nt);
            printf("D5CSV,%s,%s,%d,%d,0,%d,%zu,%d,%.4f,%.4f,%.2f\n",section,tag,M,K,nt,wb,l3,t.gbps,t.us,t.cv);
            if(g_out) g_out[ti]=t.gbps;
        }
    } else {
        size_t pool=wb*(size_t)ncopy;
        float* P=xmalloc(pool); for(int c=0;c<ncopy;c++) memcpy((char*)P+wb*(size_t)c,W,wb);
        for(int ti=0;ti<2;ti++){ int nt=ti?6:1; set_threads(nt);
            long tp=(long)(2048UL*1048576UL/wb); if(tp<(long)ncopy*2) tp=(long)ncopy*2;
            matvec(P,x,y,M,K);
            double best=1e30,sum=0,sum2=0;
            for(int rp=0;rp<reps;rp++){ double t0=now_s();
                for(long p=0;p<tp;p++){ x[p%K]+=1e-9f; matvec((float*)((char*)P+wb*(size_t)(p%ncopy)),x,y,M,K); }
                double dt=now_s()-t0; if(dt<best)best=dt;
                double g=(double)wb*tp/1e9/dt; sum+=g; sum2+=g*g; }
            double gbps=(double)wb*tp/1e9/best, us=best/tp*1e6;
            double m=sum/reps,v=sum2/reps-m*m; if(v<0)v=0; double cv=reps>1?100.0*sqrt(v)/m:-1.0;
            printf("D5CSV,%s,%s,%d,%d,0,%d,%zu,%d,%.4f,%.4f,%.2f\n",section,tag,M,K,nt,pool,l3,gbps,us,cv);
            if(g_out) g_out[ti]=gbps;
        }
        free(P);
    }
    free(W); free(x); free(y);
}

// The direct compute-bound-vs-bandwidth-bound discriminator the Principal asked for: kernel-pure ternary LUT
// rate vs kernel-pure fp32 streamed rate, AT THE SAME (M,K) shape, for every D in the sweep. Deliberately
// kernel-pure on BOTH arms (matvec_lut_full only / honest matvec only, no MLP integration overhead on either
// side) so the ratio isolates the raw-kernel question and is not confounded by the quant/dReLU envelope that
// makes the integrated 21.25 GB/s a different quantity from the fp32 38.84 GB/s asymptote (DONOR_PROJ_RATE.md
// sec.8.5's caveat). The Part-A integrated rate is also carried through here, clearly labelled, for context --
// it is NOT part of the ratio.
static void d5_fp32_vs_ternary(int reps,const double integrated_t6[5]){
    printf("---- FP32-VS-TERNARY AT MATCHED DONOR SHAPES (the compute-vs-bandwidth discriminator) ----\n");
    printf("     shape = gate/up organ (HID x D) at each D-sweep point. Both arms kernel-pure, both streamed\n");
    printf("     past 256 MB where the shape itself does not already exceed that on its own.\n");
    for(int i=0;i<5;i++){
        int D=D5_DS[i]; int HID=(int)(3.5*D+0.5);
        char ttag[32],ftag[32]; snprintf(ttag,32,"ternary_D%d",D); snprintf(ftag,32,"fp32_D%d",D);
        double gt[2],gf[2];
        lut_one_point("fp32_vs_ternary",ttag,HID,D,512.0,reps,1,gt);
        fp32_streamed_point("fp32_vs_ternary",ftag,HID,D,reps,gf);
        double ratio=gf[1]/gt[1];
        printf("D5NOTE,fp32_vs_ternary,D=%d,HID=%d,ternary_kernelpure_t6=%.2f,fp32_kernelpure_t6=%.2f,"
               "ratio_fp32_over_ternary=%.3f,ternary_integrated_t6_forcontext=%.2f\n",
               D,HID,gt[1],gf[1],ratio,integrated_t6?integrated_t6[i]:-1.0);
    }
    printf("     Read: ratio > 1 means fp32 DRAM streaming outruns the ternary kernel even fully streamed --\n");
    printf("     consistent with the LUT kernel still being compute-bound at donor width, i.e. denser packing\n");
    printf("     would NOT buy bandwidth headroom (ADAPTER_MEMO_01 sec.2.4). ratio <= 1 (ternary at/above fp32)\n");
    printf("     would mean the LUT path has become bandwidth-bound, and denser packing becomes the ~2.5x lever.\n");
    printf("     THIS RATIO, NOT AN ASSUMPTION, DECIDES IT.\n");
}

static void d5_organs(int reps){
    printf("---- PART B: real Llama-3-70B-class organ shapes, kernel-pure LUT rate vs working set ----\n");
    printf("     d_model=8192 d_ffn=28672. q/o 8192x8192, k/v 1024x8192, gate/up 28672x8192, down 8192x28672.\n");
    typedef struct { const char* nm; int M,K; } Sh;
    Sh shapes[4]={ {"llama70b_qo",8192,8192}, {"llama70b_kv",1024,8192},
                   {"llama70b_gateup",28672,8192}, {"llama70b_down",8192,28672} };
    double pools[7]={0.0,16.0,32.0,64.0,128.0,256.0,512.0};
    for(int s=0;s<4;s++){
        int Mpad=(shapes[s].M+31)&~31, T=shapes[s].K/2; size_t EB=(size_t)T*(size_t)Mpad;
        int l3=(EB<=L3_BYTES)?1:0;
        printf("   -- %s: %dx%d, single-block EB=%.2f MB (%s L3 on its own) --\n",
               shapes[s].nm,shapes[s].M,shapes[s].K,EB/1048576.0,l3?"fits":"EXCEEDS");
        // shape-only arithmetic, NOT a measurement -- true regardless of whether the timing loop below ever
        // runs. Recorded explicitly per the Principal's request so it survives even a truncated/interrupted run.
        printf("D5NOTE,shape_arithmetic,organ=%s,M=%d,K=%d,single_block_bytes=%zu,l3_bytes=%u,l3_resident_possible=%d\n",
               shapes[s].nm,shapes[s].M,shapes[s].K,EB,L3_BYTES,l3);
        double last_g[2]={0,0};
        for(int p=0;p<7;p++){
            double g[2];
            lut_one_point("organs",shapes[s].nm,shapes[s].M,shapes[s].K,pools[p],reps,1,g);
            if(p>0 && g[0]==last_g[0] && g[1]==last_g[1]) continue;  // degenerate duplicate (pool < 1 block)
            last_g[0]=g[0]; last_g[1]=g[1];
        }
    }
}

static void mode_d5(int reps){
    printf("==== BRIEF D5: ternary LUT rate at DONOR projection widths -- rate(D,threads) + planted controls ====\n");
    printf("     docs/research/donor_adaptation/briefs/BRIEF_D5_LUT_RATE_AT_DONOR_WIDTH.md. Controls run FIRST per brief sec.5.\n\n");
    int c1; double c1_rate;
    d5_control1(reps,&c1,&c1_rate);
    if(!c1){ printf("\nSTOPPED after control 1 FAIL. No further section of this mode was run.\n"); return; }
    printf("\n"); d5_lsensitivity(reps);
    printf("\n"); int c2,c3; d5_control23(reps,&c2,&c3);
    printf("\n"); int c4; d5_control4(reps,&c4);
    printf("\nD5SUMMARY,controls,c1=%s,c2=%s,c3=%s,c4=%s\n",
           c1?"PASS":"FAIL",c2?"PASS":"FAIL",c3?"PASS":"FAIL",c4?"PASS":"FAIL");
    if(!(c1&&c2&&c3&&c4)){
        printf("\nAt least one control did not fire. Per the brief, this is a STOP-and-report condition, not a\n"
               "line in a log. The sweep below still runs (so the STOP has data to point at) but must NOT be\n"
               "read as trustworthy until the failing control is understood.\n");
    }
    double integrated_t6[5];
    printf("\n"); d5_dsweep(reps,integrated_t6);
    printf("\n"); d5_fp32_vs_ternary(reps,integrated_t6);
    printf("\n"); d5_organs(reps);
}

// ================================ main ================================
static void usage(void){
    printf("usage: gemv_donor_bench <mode> [--reps N] [--check] [--seq] [--quick] [--force-unclean]\n");
    printf("  repro     KNOWN-POSITIVE gate: reproduce PHASE64_BUDGET sec.1b (4..96 MB). All else void if this FAILs.\n");
    printf("  sweep     donor-scale fp32 curve 4..1024 MB, t1/t6, GB/s + GFLOP/s\n");
    printf("  organs    real Qwen2.5-1.5B per-organ shapes, resident vs streamed\n");
    printf("  fullstack one whole decode token of fp32 projections+head (1.550 GB) -> the constant the verdict needs\n");
    printf("  lut       ternary pshufb-LUT rate vs working set at donor dims (compute- vs bandwidth-bound)\n");
    printf("  lutrepro  KNOWN-POSITIVE #2: engine.c run_expert_rate() replica (published 7.45 t1 / 17.0 t6)\n");
    printf("  lutstack  one whole decode token of ternary MLP codes (578 MB)\n");
    printf("  correct   float64 correctness table, every shape, no timing -- the only mode exempt from the\n");
    printf("            machine-quiescence gate below, since it never measures a rate\n");
    printf("  control   planted controls, no timing that matters\n");
    printf("  mlpint    engine-integrated dense ternary MLP at donor dims (the 21.25 GB/s measurement)\n");
    printf("  mlpctl    planted controls for the integrated dense-MLP block\n");
    printf("  d5        BRIEF_D5_LUT_RATE_AT_DONOR_WIDTH.md: rate(D,threads) + the four planted controls\n");
    printf("  --force-unclean   override the machine-quiescence refusal (logged; disclose with any number quoted)\n");
}
int main(int argc,char**argv){
    if(argc<2){ usage(); return 1; }
    const char* mode=argv[1];
    int reps=3, docheck=0, iid=1, quick=0, force_unclean=0;
    for(int i=2;i<argc;i++){
        if(!strcmp(argv[i],"--reps")&&i+1<argc) reps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--check")) docheck=1;
        else if(!strcmp(argv[i],"--seq")) iid=0;
        else if(!strcmp(argv[i],"--quick")) quick=1;
        else if(!strcmp(argv[i],"--force-unclean")) force_unclean=1;
        else { printf("unknown arg %s\n",argv[i]); usage(); return 1; }
    }
    if(reps<1) reps=1;
    const char* pb=getenv("OMP_PROC_BIND"); const char* pl=getenv("OMP_PLACES");
    printf("# gemv_donor_bench | mode=%s reps=%d check=%d gather=%s force_unclean=%d\n",mode,reps,docheck,iid?"iid":"seq",force_unclean);
#ifdef _OPENMP
    printf("# OpenMP ON, max_threads=%d | OMP_PROC_BIND=%s OMP_PLACES=%s\n",omp_get_max_threads(),pb?pb:"(unset)",pl?pl:"(unset)");
#else
    printf("# OpenMP OFF - t6 columns are meaningless. Rebuild with -fopenmp.\n");
#endif
    printf("# fp32 estimator: min-of-%d, ~2 GB streamed per timing window, x poisoned each pass (engine.c run_gemv_sweep)\n",reps);
    printf("# cv%% = coefficient of variation of the per-rep rates. >5%% => the machine was not quiet, do not use the row.\n");

    int needs_quiet = strcmp(mode,"correct")!=0;   // every timing mode requires (or logs override of) a clean machine
    if(!quiescence_gate(needs_quiet,force_unclean)) return 3;
    printf("\n");

    if(!strcmp(mode,"repro"))          return mode_repro(reps,quick?15.0:8.0)?2:0;
    else if(!strcmp(mode,"sweep"))     mode_sweep(reps,docheck,quick?8:15);
    else if(!strcmp(mode,"organs"))    mode_organs(reps,docheck);
    else if(!strcmp(mode,"fullstack")) mode_fullstack(reps);
    else if(!strcmp(mode,"lut"))       mode_lut(reps,iid);
    else if(!strcmp(mode,"lutrepro"))  mode_lutrepro();
    else if(!strcmp(mode,"lutstack"))  mode_lutstack(reps);
    else if(!strcmp(mode,"correct"))   mode_correct();
    else if(!strcmp(mode,"control"))   mode_control();
    else if(!strcmp(mode,"mlpint"))    mode_mlpint(reps);
    else if(!strcmp(mode,"mlpctl"))    mode_mlpctl();
    else if(!strcmp(mode,"d5"))        mode_d5(reps);
    else { usage(); return 1; }
    printf("\nSTOP. No commit, no push.\n");
    return 0;
}
