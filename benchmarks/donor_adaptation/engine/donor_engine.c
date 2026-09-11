// donor_engine.c -- run a pretrained Qwen2.5 donor on our own runtime.
//
// This is the piece the donor-adaptation programme never had: engine.c is an SSM engine and has
// no attention organ, no RoPE and no KV cache, so no donor could ever execute on it. This is a
// transformer runtime in the same style and with the same numeric conventions -- ternary weight
// codes with one fp32 scale per output row, fp32 norms/biases/embeddings, fp32 activations.
//
// It reads the flat binary written by qwen_export.py, which supports TWO weight modes on purpose:
//   quant=0 fp32     -> proves this runtime CORRECT against PyTorch before quantization is in play
//   quant=1 ternary  -> the format engine.c actually consumes
// Building the ternary path first would leave a bug in this file indistinguishable from the cost
// of the conversion, so the parity gate runs on fp32 and only then moves to ternary.
//
// Modes:
//   --logits <ids.bin> <n>   dump fp32 logits for the first n positions      (parity gate)
//   --bpb <slice.bin>        bits per UTF-8 byte over an exported eval slice (quality)
//   --bench <n>              time n single-token decode steps with a warm KV  (speed)
//   --generate <ids.bin> <n> <prefix>
//                            greedy-argmax continuation of a prompt          (E6, the model
//                            actually speaking -- every other mode is teacher-forced)
//
// Build:
//   clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp donor_engine.c -o donor_engine.exe -lm

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(_WIN32)
#include <windows.h>
static double now_s(void){ static LARGE_INTEGER f={0}; LARGE_INTEGER t;
    if(!f.QuadPart) QueryPerformanceFrequency(&f);      // fixed for the life of the process
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart/(double)f.QuadPart; }
#else
#include <time.h>
static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec+ts.tv_nsec*1e-9; }
#endif
#include <immintrin.h>

// per-organ wall-clock accounting: profile before optimising, always.
enum { T_QKV=0, T_ROPE, T_ATTN, T_O, T_FFN, T_HEAD, T_NORM, T_N };
static double g_t[T_N]; static const char* g_tn[T_N]={"qkv_proj","rope","attention","o_proj","ffn","head","norm+glue"};
static int g_prof=0;
#define TIC double _t0=g_prof?now_s():0.0
#define TOC(k) do{ if(g_prof) g_t[k]+=now_s()-_t0; }while(0)

// ---- E8: the FFN organ, DECOMPOSED.  --profile only.  These are SUB-timers of T_FFN and are
// deliberately NOT members of g_t, so the organ table and every percentage in it are unchanged;
// they are printed separately and their sum is checked against the ffn organ (gate G-Z1).
enum { F_GU=0, F_GLUE, F_DOWN, F_RES, F_N };
static double g_ff[F_N];
static const char* g_ffn[F_N]={"gate+up","glue(silu)","down","residual"};
#define TICF double _f0=g_prof?now_s():0.0
#define TOCF(k) do{ if(g_prof) g_ff[k]+=now_s()-_f0; }while(0)

// ---- E8: independent accumulators in matvec.  Every weight organ in this engine -- qkv, o,
// ffn, head -- goes through matvec, and every matvec inner loop accumulates into ONE register,
// so consecutive FMAs are serially dependent.  On Zen2 an FMA has 5-cycle latency and 2/cycle
// throughput: a single chain runs the loop at LATENCY, not at throughput, and no amount of
// memory bandwidth can be used past it.  This is E4 s finding -- attention read as
// bandwidth-bound was latency-bound, 6.53x -- asked of the WEIGHT path.  g_mvacc==1 is a
// byte-for-byte copy of the original loop, so the baseline arm is unchanged and its logits stay
// bit-identical to every number this programme has published.
// DEFAULT 4, adopted by E8 G-Z4: +34.9% at Coder-7B, parity-gated (160/160 greedy identical,
// rel L2 3.3e-06 against the single-chain arm, itself 1.5e-05 from PyTorch).  --mvacc 1 restores
// the single-chain loop BYTE FOR BYTE and is how every rate published before E8 is reproduced.
// Defaulted rather than left as a flag because a flag a runner must remember to pass is exactly
// the defect of s9, where --attn stayed on the slow kernel for every E3 and E7 number.
static int g_mvacc=4;

// ---- CONTENTION WITNESS (E7 s10.3).  The `ffn`-invariance witness -- the FFN organ cannot
// depend on context length, so an ffn reading off its plateau means the machine was not idle --
// existed ONLY under --profile, so every un-profiled rate this programme published rested on the
// operator's BELIEF that the machine was quiet, with nothing able to contradict it.  The FFN
// block, and only it, is therefore timed on the plain --bench path too, and the reading is
// printed on the BENCH line.  ON BY DEFAULT: a witness a runner has to remember to pass is the
// same defect that left --attn on the slow kernel for every E3 and E7 number (s9).
// The claim that this does not move the rate is a GATE, not an assumption -- brief s12/s13 -- and
// it is measured at the SMALLEST/FASTEST shape, where per-token overhead is worst.
// REBUILT after G-W1 FAILED at 2L timestamps per token (paired median 0.9877 against a gate of
// 0.990).  Brief 12 named the remedy before the run: time ONE layer.  Cost is now 2 timestamps per
// token instead of 2L, and each is cheaper because now_s() no longer re-reads the timer frequency.
// The reading is reported as `ffn~`, EXTRAPOLATED from layer 0 by xL and labelled as such -- the
// layers are structurally identical, and under contention every layer is hit, so layer 0 witnesses
// what all of them see.  It is a WITNESS, not a measurement: it says whether the machine was quiet.
static int g_wit=1;
static double g_w0=0.0;              // layer-0 FFN seconds, witness path only, never the profiler's
#define TICW double _t0=(g_prof||(g_wit&&l==0))?now_s():0.0
#define TOCW(k) do{ if(g_prof) g_t[k]+=now_s()-_t0; \
                    else if(g_wit&&l==0) g_w0+=now_s()-_t0; }while(0)

static void* xmalloc(size_t n){ void* p=malloc(n); if(!p){ fprintf(stderr,"OOM %zu\n",n); exit(1);} return p; }
static void die(const char* m){ fprintf(stderr,"FATAL: %s\n",m); exit(1); }

// ------------------------------------------------------------------ weight matrix (either mode)
typedef struct mat_s {
    int out, in, packed;
    const float* f32;      // quant==0
    const int8_t* code;    // quant==1, [out, in] row-major, values in {-1,0,+1}
    const float* scale;    // quant==1, [out]
    const int8_t* tm;      // --lut only: the SAME packed bytes, tile-major [in/2][Mpad]
    int Mpad;              // out rounded up to 32, the tile width the LUT kernel writes
    // ---- E25: the FACTORED kind.  W ~= A diag(s) B with A [out,rank], B [rank,in], each of
    // them an ordinary mat_t in the SAME on-disk format as everything else here, so the two
    // halves run through the SAME kernels and nothing about the numerics is new.
    // rank==0 means "not factored"; read_mat writes 0 for quant 0/1/2, so every file written
    // before this kind existed loads and executes bit-identically.
    int rank;
    struct mat_s *fa, *fb;   // A [out, rank] and B [rank, in]
    const float* fs;         // s [rank], fp32, applied to the intermediate; NULL = identity
} mat_t;

typedef struct {
    const float* in_norm;
    mat_t q, k, v, o;
    const float *qb, *kb, *vb;
    const float* post_norm;
    mat_t gate, up, down;
    // --fuse: q|k|v and gate|up concatenated by OUTPUT ROW. They share their input, so three
    // matvecs become one and two become one. The point is not fewer instructions -- it is fewer
    // OpenMP regions and bigger ones: k_proj and v_proj are 128 output rows, which across 6
    // threads is 21 rows and ~4.6 us of real work per fork.
    mat_t qkv, gateup;
    const float* qkvb;
} layer_t;

typedef struct {
    int D, F, L, NH, NKV, HD, V, tied, quant;
    float rms_eps, rope_theta;
    const float* embed;
    layer_t* lay;
    const float* final_norm;
    mat_t head;              // only when tied==0
    char* blob;
    size_t blob_bytes;
} model_t;

// ------------------------------------------------------------------ kernels
// y[o] = sum_i W[o][i] * x[i]   (+ bias).  Both modes; ternary is codes*scale, which is exactly
// what the exporter's `deq = wq * scale` means, computed without materialising deq.
// De-interleaved activations for the packed path: byte j of a row holds the weights for input
// features 2j (low trit) and 2j+1 (high trit), so the kernel needs x split by parity. Built once
// per matvec call, outside the parallel region -- O(n_in) against the loop's O(n_out * n_in).
static float *g_xe=NULL,*g_xo=NULL; static int g_xcap=0;

// ---------------------------------------------------------------- the LUT path (--lut)
// probe-1's pshufb-LUT, which is what engine.c's dense path uses and what R1 s4.2 identified as the
// only lever left on the FFN term. Two changes at once, and they must not be conflated:
//   (a) LAYOUT. The same packed bytes, transposed to tile-major [t][Mpad], so 32 output rows share
//       one broadcast table and the shuffle result IS the partial product -- no FMA, no converts.
//   (b) NUMERICS. Activations are quantized to int8 (AQ=63) so the table entries fit a byte.
//       max |entry| = 2*63 = 126 <= 127; that bound is WHY AQ is 63 and not 127.
// (a) is exactly lossless. (b) is NOT -- it is the first activation quantization this programme has
// ever run on a donor, BRIEF_T2 s5 records it as untested. Its cost is measured, not assumed.
#define AQ 63
static int      g_lut=0, g_lutdiag=0, g_lutnodown=0, g_lutnohead=0;  // --lut* family
// ---- E13 (--lutblk): BLOCKED tile-major.  Plain tile-major stores [t][Mpad] and the kernel holds
// `base` fixed while walking t, so consecutive 32-byte reads sit `Mpad` apart -- 18.5 KB at
// gate/up, 3.5 KB at down, 299 KB at kbench's 512 MB cell.  Every read lands on a different page
// and uses 32 of a 64-byte line, which is E11's candidate explanation for the LUT kernel falling
// 4.2x across 16 MB where the packed kernel does not move at all.  E11 ASSERTED that mechanism and
// never tested it; this is the test.  Blocked stores each 32-row tile's bytes contiguously, so
// within a tile the walk is linear.  Same bytes, permuted: the t order and the accumulation tree
// are untouched, so the result is BIT-IDENTICAL and the gate is sha256, not parity.
// Default OFF so every --lut number already published reproduces byte for byte.
static int      g_lutblk=0;
// Where tile `base` starts, and how far apart consecutive t are, under each layout.
#define TM_BASE(codes,base,T,Mpad) ((g_lutblk) ? (codes)+(size_t)((base)/32)*(size_t)(T)*32 \
                                               : (codes)+(size_t)(base))
#define TM_STRIDE(Mpad)            ((g_lutblk) ? (size_t)32 : (size_t)(Mpad))
// --attn {serial,ilp4,avx1,avx4,serial2}: BRIEF_E4_ATTENTION_ACCUMULATORS.md s2.  E3 s4.6 measured
// the attention organ at ~4 cycles per FMA per thread at all twelve of its points, which is the
// signature of a SERIAL FP reduction: `d+=qh[i]*kt[i]` cannot be reassociated or vectorised without
// -ffast-math, which is forbidden here (Phase 35).  These arms separate ILP from SIMD width so the
// two explanations (latency vs load path) can be told apart instead of argued about.
// Dispatched OUTSIDE the t loop, so the branch is paid L*NH times per token, not L*NH*pos.
enum { ATTN_SERIAL=0, ATTN_ILP4=1, ATTN_AVX1=2, ATTN_AVX4=3, ATTN_SERIAL2=4,
       ATTN_SERIAL_E3=5, ATTN_SERIAL3=6 };
static int g_attn=ATTN_SERIAL;

// --attnr {none,sm2,sm3,av2,av3,fork2}: BRIEF_E5_DECOMPOSE_R.md s2.  E4 left the attention organ
// split as X (the Q.K dot loop, 2.242 ms after avx4) + R (everything else, 9.816 ms = 81.4%).
// These arms split R the way serial2/serial3 split the organ: each component is run 2x and 3x
// VALUE-PRESERVINGLY, so S and Y are solved from the 1x/2x points and then PREDICTED at 3x.
// Composes with --attn; every E5 arm is measured on top of --attn avx4.
enum { ATTNR_NONE=0, ATTNR_SM2=1, ATTNR_SM3=2, ATTNR_AV2=3, ATTNR_AV3=4, ATTNR_FORK2=5,
       ATTNR_SM1=6, ATTNR_AV1=7, ATTNR_QK1=8, ATTNR_QK2=9, ATTNR_QK3=10 };
// s10: sm1/av1 are the 1x point INSIDE the wrapped path.  Run 3 solved the component as
// (2x - none) and tested it at 3x; the two increments (av2-none) and (av3-av2) are both
// "one extra A.V pass" and disagreed by 1.28x at T10 @800, because `none` runs a DIFFERENT
// code block from the wrapped arms.  With sm1/av1 the solve and the test share a code shape
// and (1x - none) prices the code-path difference itself instead of hiding inside it.
static int g_attnr=ATTNR_NONE;
// fork2's extra region must not be elidable and must not be shrunk to its one read element,
// hence volatile on the buffer itself.
#define E5_MAXNH 1024
static volatile float g_forkbuf[E5_MAXNH];
static volatile float g_sink=0.0f;

// s11: the IN-PROCESS INTERLEAVE.  Runs 1-4 compared arms living in different processes, minutes
// apart; at T10 @800 the weight path is 301 ms, so the 1% G0 tolerance is 3.0 ms and the component
// Y is 1.9 ms -- no between-process gate can be both protective and passable.  Here an arm is a
// (attn, attnr) PAIR and the arm rotates PER TOKEN, keyed off `pos` inside forward() so the same
// rotation is active in --bpb (which makes parity a test of the arms while they are MIXED).
// The schedule is a palindrome of period 2n:  idx = pos % 2n,  arm = idx<n ? idx : 2n-1-idx.
// Each arm then holds positions b+i and b+2n-1-i, whose sum is the same for every i, so every
// arm's mean context length is EXACTLY equal -- by construction, not by averaging.  Attention cost
// grows with position and a plain round-robin would have handed arm 0 systematically shorter
// contexts than arm n-1, a bias of order 1/n on the very quantity being decomposed.
#define E5_SW_MAX 16
static int g_sw=0;                                  // arms in the sweep; 0 = off
static int g_sw_attn[E5_SW_MAX], g_sw_attnr[E5_SW_MAX];
static const char* g_sw_name[E5_SW_MAX];
static double g_sw_t[E5_SW_MAX][T_N];
static long g_sw_n[E5_SW_MAX];
static int g_sw_cur=0;
// s13.4: an EXPLICIT schedule, used by --sweepd.  With g_sw_per==0 the palindrome of period 2n
// applies and every arm's mean position is equal by symmetry; with an explicit schedule the table
// is written so that the arms it compares have equal mean position by arithmetic instead.
static int g_sw_per=0;
static int g_sw_sched[64];

static inline float hsum256(__m256 v){
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi);
    lo=_mm_add_ps(lo,_mm_movehl_ps(lo,lo));
    lo=_mm_add_ss(lo,_mm_shuffle_ps(lo,lo,1));
    return _mm_cvtss_f32(lo);
}
// A0 -- the loop exactly as it was before E4.  One accumulator, one dependency chain.
static inline float dot_serial(const float* a,const float* b,int n){
    float d=0.0f; for(int i=0;i<n;i++) d+=a[i]*b[i]; return d;
}
// G1 -- the PLANTED POSITIVE.  Twice the loads and twice the FMAs, and BIT-IDENTICAL to A0:
// d1==d2 bitwise, d1+d2 is exact (same exponent, mantissa fits), and *0.5f is exact.  If the
// attention organ does not rise ~2x under this arm, the timer is not attached to this loop.
static inline float dot_serial2(const float* a,const float* b,int n){
    float d1=0.0f,d2=0.0f;
    for(int i=0;i<n;i++) d1+=a[i]*b[i];
    for(int i=0;i<n;i++) d2+=a[i]*b[i];
    return (d1+d2)*0.5f;
}
// G1' -- the SECOND planted point.  X (dot loop) and R (softmax + A.V) were solved from the 1x and
// 2x organ times, so they fit those two by construction; 3x is a point they cannot be fitted to.
// d1==d2==d3 bitwise, so (d2-d1) and (d3-d1) are exactly +0 and the sum is exactly d1.
static inline float dot_serial3(const float* a,const float* b,int n){
    float d1=0.0f,d2=0.0f,d3=0.0f;
    for(int i=0;i<n;i++) d1+=a[i]*b[i];
    for(int i=0;i<n;i++) d2+=a[i]*b[i];
    for(int i=0;i<n;i++) d3+=a[i]*b[i];
    return d1+(d2-d1)+(d3-d1);
}
// A1 -- ILP without SIMD: four scalar chains, so the reduction is 4 deep instead of n deep.
static inline float dot_ilp4(const float* a,const float* b,int n){
    float d0=0.0f,d1=0.0f,d2=0.0f,d3=0.0f; int i=0;
    for(;i+3<n;i+=4){ d0+=a[i]*b[i]; d1+=a[i+1]*b[i+1]; d2+=a[i+2]*b[i+2]; d3+=a[i+3]*b[i+3]; }
    for(;i<n;i++) d0+=a[i]*b[i];
    return (d0+d1)+(d2+d3);
}
// A2 -- SIMD width without extra ILP: 8 lanes, still ONE chain.
static inline float dot_avx1(const float* a,const float* b,int n){
    __m256 acc=_mm256_setzero_ps(); int i=0;
    for(;i+7<n;i+=8) acc=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),acc);
    float d=hsum256(acc);
    for(;i<n;i++) d+=a[i]*b[i];
    return d;
}
// A3 -- both: 8 lanes x 4 chains = 32 FMAs in flight.
static inline float dot_avx4(const float* a,const float* b,int n){
    __m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps(),
           a2=_mm256_setzero_ps(),a3=_mm256_setzero_ps();
    int i=0;
    for(;i+31<n;i+=32){
        a0=_mm256_fmadd_ps(_mm256_loadu_ps(a+i   ),_mm256_loadu_ps(b+i   ),a0);
        a1=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+8 ),_mm256_loadu_ps(b+i+8 ),a1);
        a2=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+16),_mm256_loadu_ps(b+i+16),a2);
        a3=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+24),_mm256_loadu_ps(b+i+24),a3);
    }
    for(;i+7<n;i+=8) a0=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),a0);
    float d=hsum256(_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3)));
    for(;i<n;i++) d+=a[i]*b[i];
    return d;
}
static long     g_dn[2]={0,0}; static double g_dcrest[2]={0,0},g_drel[2]={0,0},g_dcmax[2]={0,0};
static int8_t  *g_xq=NULL,*g_lutab=NULL; static int g_lutcap=0;

static inline void acc_add_i8x32(__m256i* acc,__m256i p){
    __m128i lo=_mm256_castsi256_si128(p), hi=_mm256_extracti128_si256(p,1);
    acc[0]=_mm256_add_epi32(acc[0],_mm256_cvtepi8_epi32(lo));
    acc[1]=_mm256_add_epi32(acc[1],_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8)));
    acc[2]=_mm256_add_epi32(acc[2],_mm256_cvtepi8_epi32(hi));
    acc[3]=_mm256_add_epi32(acc[3],_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8)));
}
// x -> int8 with one scale for the whole vector; returns the scale (0 for an all-zero input).
// g_clip > 0 sets the grid from k*rms instead of amax, saturating whatever lies beyond. The donor's
// activations have a crest factor (amax/rms) of 8-70, so an amax grid spends its 63 steps covering
// outliers and leaves ~4-8 of them for the bulk. Clipping trades a few saturated features for
// resolution on the rest. k=0 keeps the amax behaviour, which is what engine.c does on a model
// TRAINED for this format and is therefore the right default there and the wrong one here.
static float g_clip=0.0f;
static float quant_i8(const float* x,int n,int8_t* xq){
    float amax=0; for(int i=0;i<n;i++){ float a=fabsf(x[i]); if(a>amax) amax=a; }
    if(amax==0.0f){ memset(xq,0,(size_t)n); return 0.0f; }
    float lim=amax;
    if(g_clip>0.0f){ double s2=0; for(int i=0;i<n;i++) s2+=(double)x[i]*x[i];
        float r=(float)sqrt(s2/n)*g_clip; if(r>0.0f&&r<lim) lim=r; }
    float s=lim/(float)AQ, inv=1.0f/s;
    for(int i=0;i<n;i++){ int v=(int)lrintf(x[i]*inv); if(v>AQ)v=AQ; if(v<-AQ)v=-AQ; xq[i]=(int8_t)v; }
    return s;
}
// One 16-byte table per input PAIR. Index is this runtime's packed byte v: low trit = v%3-1 pairs
// with input 2t, high trit = v/3-1 with 2t+1. (engine.c uses the opposite digit order; the byte
// layout here is the one qwen_export.py already writes, so the tile-major copy is a pure transpose.)
static void build_lut(const int8_t* xq,int T,int8_t* lut){
    for(int t=0;t<T;t++){ int x0=xq[2*t],x1=xq[2*t+1];
        for(int v=0;v<16;v++) lut[t*16+v]=(int8_t)(v<9 ? (v%3-1)*x0+(v/3-1)*x1 : 0); }
}
// Grouped variant: the input is cut into ng contiguous groups of gp PAIRS, each carrying its own
// activation scale, and the int32 tile accumulator is folded to float at every group boundary.
// This is what the crest numbers argue for: outlier channels in an LLM are persistent, so cutting
// the input by channel index confines them to a few groups instead of setting the grid for all of
// them. The extra cost is ng foldings of 32 lanes per tile -- O(ng * M) against the kernel's
// O(T * Mpad), i.e. nothing.
static void matvec_lut_g(const int8_t* codes,const int8_t* lut,float* y,int M,int Mpad,int T,
                         int gp,const float* gs,int ng){
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int base=0;base<Mpad;base+=32){
        const int8_t* cb=TM_BASE(codes,base,T,Mpad); const size_t cst=TM_STRIDE(Mpad);
        float f[32]; for(int r=0;r<32;r++) f[r]=0.0f;
        for(int g=0;g<ng;g++){
            int t0=g*gp, t1=t0+gp; if(t1>T) t1=T; if(t0>=t1) break;
            __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),
                            _mm256_setzero_si256(),_mm256_setzero_si256()};
            for(int t=t0;t<t1;t++){
                __m256i tbl=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)t*16)));
                __m256i idx=_mm256_loadu_si256((const __m256i*)(cb+(size_t)t*cst));
                acc_add_i8x32(acc,_mm256_shuffle_epi8(tbl,idx));
            }
            int32_t tmp[32];
            _mm256_storeu_si256((__m256i*)(tmp+0),acc[0]);  _mm256_storeu_si256((__m256i*)(tmp+8),acc[1]);
            _mm256_storeu_si256((__m256i*)(tmp+16),acc[2]); _mm256_storeu_si256((__m256i*)(tmp+24),acc[3]);
            for(int r=0;r<32;r++) f[r]+=(float)tmp[r]*gs[g];
        }
        for(int r=0;r<32&&base+r<M;r++) y[base+r]=f[r];
    }
}
static void matvec_lut(const int8_t* codes,const int8_t* lut,int32_t* y,int M,int Mpad,int T){
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int base=0;base<Mpad;base+=32){
        const int8_t* cb=TM_BASE(codes,base,T,Mpad); const size_t cst=TM_STRIDE(Mpad);
        __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),
                        _mm256_setzero_si256(),_mm256_setzero_si256()};
        for(int t=0;t<T;t++){
            __m256i tbl=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)t*16)));
            __m256i idx=_mm256_loadu_si256((const __m256i*)(cb+(size_t)t*cst));
            acc_add_i8x32(acc,_mm256_shuffle_epi8(tbl,idx));
        }
        int32_t tmp[32];
        _mm256_storeu_si256((__m256i*)(tmp+0),acc[0]);  _mm256_storeu_si256((__m256i*)(tmp+8),acc[1]);
        _mm256_storeu_si256((__m256i*)(tmp+16),acc[2]); _mm256_storeu_si256((__m256i*)(tmp+24),acc[3]);
        for(int r=0;r<32&&base+r<M;r++) y[base+r]=tmp[r];
    }
}
// Transpose the packed row-major bytes into tile-major. Pure data movement: no re-encoding, so a
// --lut model and a --quant packed model hold bit-identical WEIGHTS and differ only in activations.
// Fill the tile-major replica from row-major codes, in whichever layout g_lutblk selects. The two
// layouts hold the SAME bytes in the same tiles -- only the address of tile `b` at step `t` moves --
// so a kernel reading through TM_BASE/TM_STRIDE gets an identical byte sequence either way.
// Rows [out, Mpad) are padding and get code 4 == the (0,0) trit pair, a no-op row.
static void tm_fill(int8_t* tm,const int8_t* code,int M,int Mpad,int H){
    for(int b=0;b<Mpad;b+=32)
        for(int t=0;t<H;t++){
            int8_t* d = g_lutblk ? tm+((size_t)(b/32)*(size_t)H+(size_t)t)*32
                                 : tm+(size_t)t*(size_t)Mpad+(size_t)b;
            for(int r=0;r<32;r++){ int o=b+r; d[r]=(o<M)?code[(size_t)o*H+t]:(int8_t)4; }
        }
}
static void build_tm(mat_t* m){
    if(!m->packed||m->tm) return;
    const int H=m->in/2; m->Mpad=((m->out+31)/32)*32;
    int8_t* tm=xmalloc((size_t)H*m->Mpad);
    tm_fill(tm,m->code,m->out,m->Mpad,H);
    m->tm=tm;
}

static int32_t* g_i32b=NULL; static int g_i32cap=0;
static int32_t* g_i32(int n){ if(n>g_i32cap){ free(g_i32b); g_i32b=xmalloc((size_t)n*4); g_i32cap=n; } return g_i32b; }
static float* g_f32b=NULL; static int g_f32cap=0;
static float* g_f32(int n){ if(n>g_f32cap){ free(g_f32b); g_f32b=xmalloc((size_t)n*4); g_f32cap=n; } return g_f32b; }
static int g_group=0, g_gscap=0; static float* g_gs=NULL;   // --lut-group
static char g_grouplbl[16]="";

// ---------------------------------------------------------------- the FACTORED path (E25)
// y = A (s * (B x)) + bias, run as TWO matvecs with an intermediate of size `rank`.
// This is the first time this programme EXECUTES a low-rank form as a low-rank form.  E21 and
// E22 both computed the product A.B and installed a DENSE matrix, so every rank number
// published here priced the QUALITY of the cut and never its COST.  A rank-r cut of an
// [out,in] matrix moves r*(out+in) weights per token instead of out*in -- at D=1536, r=512
// that is 1.57 M against 2.36 M -- and that ratio is the only reason the rank axis appears in
// the 50 tok/s budget table at all.  Until this path existed, that ratio was arithmetic.
// The intermediate needs its OWN buffer: matvec's LUT path reuses g_i32b/g_f32b inside a
// single call, so the two halves may not share them.  matvec is called from forward() in
// serial (the parallelism is inside it), so one global suffices; nesting is refused at load.
static float* g_lrb=NULL; static int g_lrcap=0;
static float* g_lr(int n){ if(n>g_lrcap){ free(g_lrb); g_lrb=xmalloc((size_t)n*4); g_lrcap=n; }
                           return g_lrb; }

static void matvec(const mat_t* m, const float* x, const float* bias, float* y){
    const int n_out=m->out, n_in=m->in;   // NOT "OUT"/"IN": windows.h defines those as SAL macros
    if(m->rank){
        float* h=g_lr(m->rank);
        matvec(m->fb,x,NULL,h);
        // s multiplies the INTERMEDIATE, not either factor: folding it into B would change B's
        // per-row ternarization and folding it into A would change A's.  The whole point of
        // carrying it is that the trained scale is fp32 while the factors are not.
        if(m->fs) for(int i=0;i<m->rank;i++) h[i]*=m->fs[i];
        matvec(m->fa,h,bias,y);
        return;
    }
    if(m->tm){                                                    // --lut
        const int H=n_in/2;
        if(H>g_lutcap){ free(g_xq); free(g_lutab);
            g_xq=xmalloc((size_t)H*2); g_lutab=xmalloc((size_t)H*16); g_lutcap=H; }
        // --lut-group G: G input channels share one scale (G must be even so no PAIR straddles a
        // boundary). G=0 keeps one scale for the whole vector, which is engine.c's convention.
        int gsz = g_group>0 ? g_group : n_in;
        if(gsz>n_in) gsz=n_in;
        int ng = (n_in+gsz-1)/gsz;
        if(ng>g_gscap){ free(g_gs); g_gs=xmalloc((size_t)ng*4); g_gscap=ng; }
        for(int g=0;g<ng;g++){ int o0=g*gsz, len=(o0+gsz<=n_in)?gsz:(n_in-o0);
            g_gs[g]=quant_i8(x+o0,len,g_xq+o0); }
        float sx=g_gs[0];
        if(g_lutdiag){
            // Measure the damage on the INPUT SIDE ONLY. This is arithmetic on x and its int8
            // round-trip; the kernel is not involved, so it separates "int8 activations are
            // lossy on a donor" from "the kernel is wrong".
            // crest is measured WITHIN each group, because that is what sets the grid: with
            // --lut-group the whole-vector amax/rms no longer describes the resolution anyone gets.
            double s2=0,e2=0,cs=0,cmax=0; int nc=0;
            for(int g=0;g<ng;g++){ int o0=g*gsz, len=(o0+gsz<=n_in)?gsz:(n_in-o0);
                double gs2=0,gam=0;
                for(int i=o0;i<o0+len;i++){ double v=x[i]; gs2+=v*v; if(fabs(v)>gam) gam=fabs(v); }
                double grms=sqrt(gs2/len);
                if(grms>0){ double c=gam/grms; cs+=c; nc++; if(c>cmax) cmax=c; } }
            for(int i=0;i<n_in;i++){ double v=x[i],r=(double)g_xq[i]*g_gs[i/gsz], d=v-r;
                s2+=v*v; e2+=d*d; }
            double rel=sqrt(e2/(s2>0?s2:1));
            int b = (n_in==4864)?1:0;                     // down_proj is the only in=4864 organ
            g_dn[b]++; g_drel[b]+=rel;
            g_dcrest[b]+=(nc?cs/nc:0); if(cmax>g_dcmax[b]) g_dcmax[b]=cmax;
        }
        build_lut(g_xq,H,g_lutab);
        if(ng==1){
            int32_t* acc=(int32_t*)g_i32(n_out);
            matvec_lut(m->tm,g_lutab,acc,n_out,m->Mpad,H);
            for(int o=0;o<n_out;o++){ float t=(float)acc[o]*sx*m->scale[o]; y[o]=bias?t+bias[o]:t; }
        } else {
            float* acc=g_f32(n_out);
            matvec_lut_g(m->tm,g_lutab,acc,n_out,m->Mpad,H,gsz/2,g_gs,ng);
            for(int o=0;o<n_out;o++){ float t=acc[o]*m->scale[o]; y[o]=bias?t+bias[o]:t; }
        }
        return;
    }
    if(m->packed){
        const int H=n_in/2;
        if(H>g_xcap){ free(g_xe); free(g_xo); g_xe=xmalloc((size_t)H*4); g_xo=xmalloc((size_t)H*4); g_xcap=H; }
        for(int j=0;j<H;j++){ g_xe[j]=x[2*j]; g_xo[j]=x[2*j+1]; }
        // pshufb tables: index is the packed byte v in [0,8];  low trit = v%3 - 1, high = v/3 - 1
        const __m128i TLO=_mm_setr_epi8(-1,0,1,-1,0,1,-1,0,1,0,0,0,0,0,0,0);
        const __m128i THI=_mm_setr_epi8(-1,-1,-1,0,0,0,1,1,1,0,0,0,0,0,0,0);
        const int MA=g_mvacc;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(int o=0;o<n_out;o++){
            const int8_t* c=m->code+(size_t)o*H;
            __m256 acc=_mm256_setzero_ps(); int j=0;
            if(MA>1){
                // 2 chains (MA==2) or 4 (MA==4).  Same bytes read, same FMA count, same order
                // WITHIN a chain -- only the partition of the sum changes, so this is NOT
                // bit-identical and is gated on end-to-end parity, never on sha256.
                __m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps();
                __m256 a2=_mm256_setzero_ps(),a3=_mm256_setzero_ps();
                if(MA>2) for(;j+16<=H;j+=16){
                    __m128i b0=_mm_loadl_epi64((const __m128i*)(c+j));
                    __m128i b1=_mm_loadl_epi64((const __m128i*)(c+j+8));
                    a0=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_shuffle_epi8(TLO,b0))),
                                       _mm256_loadu_ps(g_xe+j),a0);
                    a1=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_shuffle_epi8(THI,b0))),
                                       _mm256_loadu_ps(g_xo+j),a1);
                    a2=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_shuffle_epi8(TLO,b1))),
                                       _mm256_loadu_ps(g_xe+j+8),a2);
                    a3=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_shuffle_epi8(THI,b1))),
                                       _mm256_loadu_ps(g_xo+j+8),a3);
                }
                for(;j+8<=H;j+=8){
                    __m128i b=_mm_loadl_epi64((const __m128i*)(c+j));
                    a0=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_shuffle_epi8(TLO,b))),
                                       _mm256_loadu_ps(g_xe+j),a0);
                    a1=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_shuffle_epi8(THI,b))),
                                       _mm256_loadu_ps(g_xo+j),a1);
                }
                acc=_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3));
            } else
            for(;j+8<=H;j+=8){
                __m128i b=_mm_loadl_epi64((const __m128i*)(c+j));   // 8 packed bytes
                __m128i lo=_mm_shuffle_epi8(TLO,b), hi=_mm_shuffle_epi8(THI,b);
                acc=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)),
                                    _mm256_loadu_ps(g_xe+j),acc);
                acc=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)),
                                    _mm256_loadu_ps(g_xo+j),acc);
            }
            float s[8]; _mm256_storeu_ps(s,acc);
            float t=s[0]+s[1]+s[2]+s[3]+s[4]+s[5]+s[6]+s[7];
            for(;j<H;j++){ int v=(uint8_t)c[j]; int t0=v%3-1, t1=(v/3)-1; t+=(float)t0*g_xe[j]+(float)t1*g_xo[j]; }
            t*=m->scale[o];
            y[o]=bias?t+bias[o]:t;
        }
        return;
    }
    if(m->f32){
        const int MA=g_mvacc;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(int o=0;o<n_out;o++){
            const float* w=m->f32+(size_t)o*n_in;
            __m256 acc=_mm256_setzero_ps(); int i=0;
            if(MA>1){
                __m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps();
                __m256 a2=_mm256_setzero_ps(),a3=_mm256_setzero_ps();
                if(MA>2) for(;i+32<=n_in;i+=32){
                    a0=_mm256_fmadd_ps(_mm256_loadu_ps(w+i   ),_mm256_loadu_ps(x+i   ),a0);
                    a1=_mm256_fmadd_ps(_mm256_loadu_ps(w+i+8 ),_mm256_loadu_ps(x+i+8 ),a1);
                    a2=_mm256_fmadd_ps(_mm256_loadu_ps(w+i+16),_mm256_loadu_ps(x+i+16),a2);
                    a3=_mm256_fmadd_ps(_mm256_loadu_ps(w+i+24),_mm256_loadu_ps(x+i+24),a3);
                }
                for(;i+16<=n_in;i+=16){
                    a0=_mm256_fmadd_ps(_mm256_loadu_ps(w+i  ),_mm256_loadu_ps(x+i  ),a0);
                    a1=_mm256_fmadd_ps(_mm256_loadu_ps(w+i+8),_mm256_loadu_ps(x+i+8),a1);
                }
                for(;i+8<=n_in;i+=8) a0=_mm256_fmadd_ps(_mm256_loadu_ps(w+i),_mm256_loadu_ps(x+i),a0);
                acc=_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3));
            } else
            for(;i+8<=n_in;i+=8) acc=_mm256_fmadd_ps(_mm256_loadu_ps(w+i),_mm256_loadu_ps(x+i),acc);
            float s[8]; _mm256_storeu_ps(s,acc);
            float t=s[0]+s[1]+s[2]+s[3]+s[4]+s[5]+s[6]+s[7];
            for(;i<n_in;i++) t+=w[i]*x[i];
            y[o]=bias?t+bias[o]:t;
        }
        return;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int o=0;o<n_out;o++){
        const int8_t* c=m->code+(size_t)o*n_in;
        __m256 acc=_mm256_setzero_ps(); int i=0;
        for(;i+8<=n_in;i+=8){
            // 8 ternary codes -> int32 -> float, then FMA against the activations
            __m128i c8=_mm_loadl_epi64((const __m128i*)(c+i));
            __m256i ci=_mm256_cvtepi8_epi32(c8);
            acc=_mm256_fmadd_ps(_mm256_cvtepi32_ps(ci),_mm256_loadu_ps(x+i),acc);
        }
        float s[8]; _mm256_storeu_ps(s,acc);
        float t=s[0]+s[1]+s[2]+s[3]+s[4]+s[5]+s[6]+s[7];
        for(;i<n_in;i++) t+=(float)c[i]*x[i];
        t*=m->scale[o];
        y[o]=bias?t+bias[o]:t;
    }
}

static void rmsnorm(const float* x,const float* w,int n,float eps,float* y){
    double ss=0.0; for(int i=0;i<n;i++) ss+=(double)x[i]*x[i];
    float inv=(float)(1.0/sqrt(ss/(double)n+(double)eps));
    for(int i=0;i<n;i++) y[i]=x[i]*inv*w[i];
}

static float silu(float x){ return x/(1.0f+expf(-x)); }

// HF "rotate_half": q'[j] = q[j]cos - q[j+h]sin ; q'[j+h] = q[j+h]cos + q[j]sin,  h = HD/2
// rope used to compute pow()+cosf()+sinf() PER HEAD PER LAYER, i.e. NH+NKV heads x L layers x
// HD/2 elements of transcendentals per token -- 12,288 double-precision pow() calls per token on
// this donor, for 32 distinct values. The angle depends only on (pos, j): not on the head, not on
// the layer. Hoisted to one table per token. Bit-identical by construction: the same expression,
// the same order, the same float rounding -- verified against a 64-token logit dump.
#define ROPE_MAX_HD 512
static void rope_table(float* cs,int HD,int pos,float theta){
    const int h=HD/2;
    for(int j=0;j<h;j++){
        float f=(float)(pos*pow((double)theta,-2.0*(double)j/(double)HD));
        cs[j]=cosf(f); cs[h+j]=sinf(f);
    }
}
static void rope(float* v,int n_heads,int HD,const float* cs){
    const int h=HD/2;
    for(int head=0;head<n_heads;head++){
        float* p=v+(size_t)head*HD;
        for(int j=0;j<h;j++){
            float c=cs[j], s=cs[h+j];
            float a=p[j], b=p[j+h];
            p[j]=a*c-b*s; p[j+h]=b*c+a*s;
        }
    }
}

// ------------------------------------------------------------------ model loading
static const char* rd(const char** p, size_t n){ const char* q=*p; *p+=n; return q; }

// quant==3 is the TAGGED layout: every matrix carries its own int32 kind, so one file can hold
// packed-ternary FFNs, fp32 k/v and factored q/o at the same time.  That mix is not a
// convenience -- it is exactly the configuration E22/E23 measured (k/v left fp32) and the one
// H0 trains, and a single global quant flag cannot express it.  quant 0/1/2 stay untagged and
// byte-for-byte unchanged, so every artifact already on disk still loads.
enum { MK_PACKED=0, MK_FACTORED=1, MK_F32=2 };

static void read_mat(const char** p, mat_t* m, int out, int in, int quant){
    m->out=out; m->in=in; m->packed=(quant==2); m->f32=NULL; m->code=NULL; m->scale=NULL;
    m->tm=NULL; m->Mpad=0; m->rank=0; m->fa=NULL; m->fb=NULL; m->fs=NULL;
    if(quant==3){
        int kind; memcpy(&kind,rd(p,4),4);
        if(kind==MK_PACKED){
            m->packed=1;
            if(in&1) die("tagged packed matrix with odd in_features");
            m->code=(const int8_t*)rd(p,(size_t)out*(in/2));
            m->scale=(const float*)rd(p,(size_t)out*4);
        } else if(kind==MK_F32){
            m->f32=(const float*)rd(p,(size_t)out*in*4);
        } else if(kind==MK_FACTORED){
            int r; memcpy(&r,rd(p,4),4);
            if(r<=0||(r&1)) die("factored matrix: rank must be positive and even");
            m->rank=r;
            m->fa=(mat_t*)xmalloc(sizeof(mat_t));
            m->fb=(mat_t*)xmalloc(sizeof(mat_t));
            read_mat(p,m->fa,out,r,3);
            m->fs=(const float*)rd(p,(size_t)r*4);
            read_mat(p,m->fb,r,in,3);
            // One level only.  A factor of a factor would need a second intermediate buffer and
            // nothing asks for one; refuse at LOAD rather than compute the wrong thing at run.
            if(m->fa->rank||m->fb->rank) die("factored matrix: a factor may not itself be factored");
        } else {
            die("unknown matrix kind in a quant=3 file");
        }
        return;
    }
    if(quant==0){ m->f32=(const float*)rd(p,(size_t)out*in*4); }
    else if(quant==1){ m->code=(const int8_t*)rd(p,(size_t)out*in);
                       m->scale=(const float*)rd(p,(size_t)out*4); }
    else { m->code=(const int8_t*)rd(p,(size_t)out*(in/2));   // base-3 g=2, 2 trits/byte
           m->scale=(const float*)rd(p,(size_t)out*4); }
}

// 64-bit file offsets.  `long` is 32 bits on Windows even on x64, so plain fseek/ftell
// cannot describe a file over 2 GB -- see the comment in load().
#if defined(_WIN32)
  #define XFSEEK _fseeki64
  #define XFTELL _ftelli64
  typedef long long xoff_t;
#else
  #define XFSEEK fseeko
  #define XFTELL ftello
  typedef off_t xoff_t;
#endif

static void load(model_t* M,const char* path){
    FILE* f=fopen(path,"rb"); if(!f) die("cannot open weights");
    // This used to be `fseek(...,SEEK_END); long sz=ftell(f);`.  On Windows `long` is 32 bits
    // even on x64, so for a file over 2 GB fseek(SEEK_END) FAILS (returns -1) and ftell then
    // reports 0.  The old code allocated a zero-byte blob, read zero bytes -- which equals the
    // zero it asked for, so the "short read" check passed -- and then announced
    // "bad magic -- not a QWENDON1 file" about a file whose magic was perfectly intact.
    // Measured on Qwen2.5-1.5B fp32 (6,174,857,268 bytes): fseek(END) = -1, ftell = 0.
    // The largest artifact this engine had ever been given was 1.84 GB, just under the
    // ceiling, so the ceiling had never been touched.  A 10B ternary packed model is ~5 GB.
    if(XFSEEK(f,0,SEEK_END)!=0) die("cannot seek to end of weights file (over 2 GB without 64-bit offsets?)");
    xoff_t szo=XFTELL(f);
    if(szo<=0) die("cannot determine weights file size");
    if(XFSEEK(f,0,SEEK_SET)!=0) die("cannot rewind weights file");
    size_t sz=(size_t)szo;
    M->blob=xmalloc(sz); M->blob_bytes=sz;
    // Read in chunks: not every C runtime honours a single fread larger than 2 GB.
    { char* dst=(char*)M->blob; size_t left=sz;
      while(left){ size_t want=left>((size_t)1<<30)?((size_t)1<<30):left;
                   size_t got=fread(dst,1,want,f);
                   if(got!=want) die("short read");
                   dst+=got; left-=got; } }
    fclose(f);
    const char* p=M->blob;
    if(memcmp(p,"QWENDON1",8)) die("bad magic -- not a QWENDON1 file");
    p+=8;
    const int32_t* hi=(const int32_t*)p; p+=9*4;
    M->D=hi[0]; M->F=hi[1]; M->L=hi[2]; M->NH=hi[3]; M->NKV=hi[4];
    M->HD=hi[5]; M->V=hi[6]; M->tied=hi[7]; M->quant=hi[8];
    const float* hf=(const float*)p; p+=2*4;
    M->rms_eps=hf[0]; M->rope_theta=hf[1];
    const int QO=M->NH*M->HD, KVO=M->NKV*M->HD;
    fprintf(stderr,"  D=%d F=%d L=%d heads=%d/%d hd=%d V=%d tied=%d quant=%s eps=%g theta=%g\n",
            M->D,M->F,M->L,M->NH,M->NKV,M->HD,M->V,M->tied,
            M->quant==3?"tagged(per-matrix kind)":
            M->quant==2?"packed(2 trits/byte)":M->quant?"ternary":"fp32",
            M->rms_eps,M->rope_theta);
    M->embed=(const float*)rd(&p,(size_t)M->V*M->D*4);
    M->lay=xmalloc((size_t)M->L*sizeof(layer_t));
    for(int l=0;l<M->L;l++){
        layer_t* L=&M->lay[l];
        L->in_norm=(const float*)rd(&p,(size_t)M->D*4);
        read_mat(&p,&L->q,QO,M->D,M->quant);  L->qb=(const float*)rd(&p,(size_t)QO*4);
        read_mat(&p,&L->k,KVO,M->D,M->quant); L->kb=(const float*)rd(&p,(size_t)KVO*4);
        read_mat(&p,&L->v,KVO,M->D,M->quant); L->vb=(const float*)rd(&p,(size_t)KVO*4);
        read_mat(&p,&L->o,M->D,QO,M->quant);
        L->post_norm=(const float*)rd(&p,(size_t)M->D*4);
        read_mat(&p,&L->gate,M->F,M->D,M->quant);
        read_mat(&p,&L->up,  M->F,M->D,M->quant);
        read_mat(&p,&L->down,M->D,M->F,M->quant);
    }
    M->final_norm=(const float*)rd(&p,(size_t)M->D*4);
    if(!M->tied) read_mat(&p,&M->head,M->V,M->D,M->quant);
    size_t used=(size_t)(p-M->blob);
    if(used!=M->blob_bytes){
        fprintf(stderr,"  FATAL: consumed %zu of %zu bytes -- layout mismatch\n",used,M->blob_bytes);
        exit(1);
    }
    fprintf(stderr,"  layout OK: consumed exactly %zu bytes\n",used);
}

// ---------------------------------------------------------------- --fuse
// Concatenate matrices that share an input, by output row. Pure data movement: the codes, the
// scales and the biases are copied unchanged, so a fused model computes bit-identical results
// and the only thing that changes is how many OpenMP regions a token opens (169 -> 97).
// It costs one extra copy of the fused weights in RAM (~117 MB on Qwen2.5-0.5B packed).
static int g_fuse=0;
static void fuse_mats(mat_t* dst,const mat_t* const* src,int n,int quant){
    int out=0, in=src[0]->in;
    for(int i=0;i<n;i++){
        if(src[i]->in!=in) die("--fuse: inputs differ");
        // A factored matrix has no single row space to concatenate along: q's A is [QO,r], k's
        // is [KVO,r'], and their B's are different matrices entirely.  Refuse rather than
        // silently fuse the wrong thing.
        if(src[i]->rank) die("--fuse cannot concatenate a FACTORED matrix -- run without --fuse");
        if(src[i]->packed!=src[0]->packed || (src[i]->f32!=NULL)!=(src[0]->f32!=NULL))
            die("--fuse: matrices in one group have different kinds");
        out+=src[i]->out;
    }
    (void)quant;   // the KIND now comes from the sources, so a tagged file fuses correctly too
    dst->out=out; dst->in=in; dst->packed=src[0]->packed; dst->tm=NULL; dst->Mpad=0;
    dst->f32=NULL; dst->code=NULL; dst->scale=NULL;
    dst->rank=0; dst->fa=NULL; dst->fb=NULL; dst->fs=NULL;
    if(src[0]->f32){
        float* w=xmalloc((size_t)out*in*4); size_t o=0;
        for(int i=0;i<n;i++){ memcpy(w+o,src[i]->f32,(size_t)src[i]->out*in*4); o+=(size_t)src[i]->out*in; }
        dst->f32=w;
    } else {
        size_t rb=src[0]->packed?(size_t)(in/2):(size_t)in;
        int8_t* c=xmalloc((size_t)out*rb); float* sc=xmalloc((size_t)out*4);
        size_t bo=0; int so=0;
        for(int i=0;i<n;i++){
            memcpy(c+bo,src[i]->code,(size_t)src[i]->out*rb); bo+=(size_t)src[i]->out*rb;
            memcpy(sc+so,src[i]->scale,(size_t)src[i]->out*4); so+=src[i]->out;
        }
        dst->code=c; dst->scale=sc;
    }
}
static float* cat3f(const float* a,int na,const float* b,int nb,const float* c,int nc){
    float* r=xmalloc((size_t)(na+nb+nc)*4);
    memcpy(r,a,(size_t)na*4); memcpy(r+na,b,(size_t)nb*4); memcpy(r+na+nb,c,(size_t)nc*4);
    return r;
}
static void build_fused(model_t* M){
    size_t bytes=0;
    for(int l=0;l<M->L;l++){ layer_t* L=&M->lay[l];
        const mat_t* qkv[3]={&L->q,&L->k,&L->v};
        const mat_t* gu[2]={&L->gate,&L->up};
        fuse_mats(&L->qkv,qkv,3,M->quant);
        fuse_mats(&L->gateup,gu,2,M->quant);
        L->qkvb=cat3f(L->qb,L->q.out,L->kb,L->k.out,L->vb,L->v.out);
        size_t rb=L->qkv.f32?(size_t)M->D*4:(L->qkv.packed?(size_t)(M->D/2):(size_t)M->D);
        bytes+=(size_t)(L->qkv.out+L->gateup.out)*rb;
    }
    fprintf(stderr,"  --fuse: q|k|v and gate|up concatenated, +%.1f MB, matvec calls/token %d -> %d\n",
            bytes/1048576.0, M->L*7+1, M->L*4+1);
}

// ------------------------------------------------------------------ state
typedef struct {
    float *x,*xb,*xb2,*q,*k,*v,*att,*attout,*hb,*hb2,*logits,*qkvbuf,*gubuf;
    float *kcache,*vcache;      // [L][maxseq][KVO]
    int maxseq;
} state_t;

static void state_init(state_t* s,const model_t* M,int maxseq){
    const int QO=M->NH*M->HD, KVO=M->NKV*M->HD;
    s->maxseq=maxseq;
    s->x=xmalloc((size_t)M->D*4); s->xb=xmalloc((size_t)M->D*4); s->xb2=xmalloc((size_t)M->D*4);
    s->q=xmalloc((size_t)QO*4); s->k=xmalloc((size_t)KVO*4); s->v=xmalloc((size_t)KVO*4);
    s->att=xmalloc((size_t)M->NH*maxseq*4);   // per-head scratch, no malloc in the hot loop
    s->attout=xmalloc((size_t)QO*4);         // QO need not equal D on every donor
    s->hb=xmalloc((size_t)M->F*4); s->hb2=xmalloc((size_t)M->F*4);
    s->logits=xmalloc((size_t)M->V*4);
    s->kcache=xmalloc((size_t)M->L*maxseq*KVO*4);
    s->vcache=xmalloc((size_t)M->L*maxseq*KVO*4);
    s->qkvbuf=xmalloc((size_t)(QO+2*KVO)*4);      // --fuse scratch; q|k|v land contiguous,
    s->gubuf =xmalloc((size_t)(2*M->F)*4);        // gate|up likewise, and both are read in place
}

// one token at position `pos`; logits land in s->logits
static void forward(const model_t* M,state_t* s,int token,int pos){
    if(g_sw){                                       // s11: palindrome, or s13.4 explicit table
        int per = g_sw_per ? g_sw_per : 2*g_sw;
        int idx = (int)(((unsigned)pos)%(unsigned)per);
        g_sw_cur = g_sw_per ? g_sw_sched[idx] : (idx<g_sw ? idx : 2*g_sw-1-idx);
        g_attn=g_sw_attn[g_sw_cur]; g_attnr=g_sw_attnr[g_sw_cur];
    }
    const int D=M->D, F=M->F, NH=M->NH, NKV=M->NKV, HD=M->HD;
    const int QO=NH*HD, KVO=NKV*HD, GQA=NH/NKV;
    memcpy(s->x,M->embed+(size_t)token*D,(size_t)D*4);

    float ropecs[ROPE_MAX_HD];
    { TIC;
      if(HD>ROPE_MAX_HD) die("head_dim > ROPE_MAX_HD: raise the constant");
      rope_table(ropecs,HD,pos,M->rope_theta);   // once per TOKEN, not per head per layer
      TOC(T_ROPE); }

    for(int l=0;l<M->L;l++){
        const layer_t* L=&M->lay[l];
        { TIC; rmsnorm(s->x,L->in_norm,D,M->rms_eps,s->xb); TOC(T_NORM); }
        float *qp=s->q,*kp=s->k,*vp=s->v;
        { TIC;
          if(g_fuse){
              matvec(&L->qkv,s->xb,L->qkvb,s->qkvbuf);   // one region instead of three
              qp=s->qkvbuf; kp=s->qkvbuf+QO; vp=s->qkvbuf+QO+KVO;   // read in place, no copy
          } else {
              matvec(&L->q,s->xb,L->qb,s->q);
              matvec(&L->k,s->xb,L->kb,s->k);
              matvec(&L->v,s->xb,L->vb,s->v);
          }
          TOC(T_QKV); }
        { TIC;
          // split out of T_QKV on purpose: rope was inside it, so the per-organ GB/s the
          // ledger derived for qkv charged rope's transcendentals to the weights.
          rope(qp,NH,HD,ropecs);
          rope(kp,NKV,HD,ropecs);
          TOC(T_ROPE); }

        float* kc=s->kcache+((size_t)l*s->maxseq+pos)*KVO;
        float* vc=s->vcache+((size_t)l*s->maxseq+pos)*KVO;
        memcpy(kc,kp,(size_t)KVO*4);
        memcpy(vc,vp,(size_t)KVO*4);

        const float inv=1.0f/sqrtf((float)HD);
        // loop-invariant, computed OUTSIDE the parallel region so no arm pays a branch per head
        const int avwrap=(g_attnr==ATTNR_AV1||g_attnr==ATTNR_AV2||g_attnr==ATTNR_AV3);
        const int avrep=(g_attnr==ATTNR_AV3)?3:((g_attnr==ATTNR_AV2)?2:1);
        const int smwrap=(g_attnr==ATTNR_SM1||g_attnr==ATTNR_SM2||g_attnr==ATTNR_SM3);
        const int smrep=(g_attnr==ATTNR_SM3)?3:((g_attnr==ATTNR_SM2)?2:1);
        const int qkwrap=(g_attnr==ATTNR_QK1||g_attnr==ATTNR_QK2||g_attnr==ATTNR_QK3);
        const int qkrep=(g_attnr==ATTNR_QK3)?3:((g_attnr==ATTNR_QK2)?2:1);
        TIC;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(int h=0;h<NH;h++){
            const int kvh=h/GQA;
            float* a=s->att+(size_t)h*s->maxseq;      // preallocated, disjoint per head
            const float* qh=qp+(size_t)h*HD;
            float mx=-1e30f;
            const float* kbase=s->kcache+((size_t)l*s->maxseq)*KVO+(size_t)kvh*HD;
            // one branch per head, not per (head,position): the t loop below is unbranched.
            switch(g_attn){
            case ATTN_SERIAL_E3:
                // E3's loop byte for byte, INCLUDING the per-position address arithmetic that
                // every other arm here hoists.  Gate G4' (brief s8.2): the arm that claims to
                // reproduce E3 must recompute the address the way E3 did, or the 4-7% it saves
                // is silently credited to the accumulators.
                for(int t=0;t<=pos;t++){
                    const float* kt=s->kcache+((size_t)l*s->maxseq+t)*KVO+(size_t)kvh*HD;
                    float d=0.0f; for(int i=0;i<HD;i++) d+=qh[i]*kt[i];
                    d*=inv; a[t]=d; if(d>mx) mx=d;
                } break;
            case ATTN_SERIAL3:
                for(int t=0;t<=pos;t++){ float d=dot_serial3(qh,kbase+(size_t)t*KVO,HD)*inv;
                                         a[t]=d; if(d>mx) mx=d; } break;
            case ATTN_ILP4:
                for(int t=0;t<=pos;t++){ float d=dot_ilp4(qh,kbase+(size_t)t*KVO,HD)*inv;
                                         a[t]=d; if(d>mx) mx=d; } break;
            case ATTN_AVX1:
                for(int t=0;t<=pos;t++){ float d=dot_avx1(qh,kbase+(size_t)t*KVO,HD)*inv;
                                         a[t]=d; if(d>mx) mx=d; } break;
            case ATTN_AVX4:
                if(!qkwrap){
                    for(int t=0;t<=pos;t++){ float d=dot_avx4(qh,kbase+(size_t)t*KVO,HD)*inv;
                                             a[t]=d; if(d>mx) mx=d; }
                } else {
                    // s13.4: one code shape for qkrep = 1, 2 and 3.  `keep` is exactly +0 for any
                    // finite mx, so a[t] is bit-identical to the unwrapped loop -- and 0.0f*mx is
                    // NOT foldable without -ffast-math (mx could be NaN or Inf), so pass r+1's
                    // stores genuinely depend on pass r and no pass can be elided.  It is a local,
                    // so unlike a shared sink it adds no cross-thread traffic to what it measures.
                    float keep=0.0f;
                    for(int r=0;r<qkrep;r++){
                        mx=-1e30f;
                        for(int t=0;t<=pos;t++){ float d=dot_avx4(qh,kbase+(size_t)t*KVO,HD)*inv;
                                                 a[t]=d+keep; if(d>mx) mx=d; }
                        keep=0.0f*mx;
                    }
                } break;
            case ATTN_SERIAL2:
                for(int t=0;t<=pos;t++){ float d=dot_serial2(qh,kbase+(size_t)t*KVO,HD)*inv;
                                         a[t]=d; if(d>mx) mx=d; } break;
            default:
                for(int t=0;t<=pos;t++){ float d=dot_serial(qh,kbase+(size_t)t*KVO,HD)*inv;
                                         a[t]=d; if(d>mx) mx=d; } break;
            }
            // ---- S, the softmax pass.  sm2/sm3 run it 2x/3x over the UNMODIFIED a[] and keep
            // the LAST result; the discarded sums are folded back as exact zeros (s0==s1==sum
            // bitwise, so both corrections are +0), which makes the arm bit-identical -- gate G2.
            // The discarded passes omit the store into a[] and are therefore a slight
            // UNDER-count of S by one L1 store per expf; recorded in the probe, not hidden.
            float sum=0.0f;
            if(smwrap){
                // ONE code shape for smrep = 1, 2 and 3.  The discarded passes read a[] before the
                // kept pass overwrites it, and are folded back as exact zeros (s==sum bitwise).
                float s0=0.0f,s1=0.0f;
                if(smrep>=2){ for(int t=0;t<=pos;t++) s0+=expf(a[t]-mx); }
                if(smrep>=3){ for(int t=0;t<=pos;t++) s1+=expf(a[t]-mx); }
                for(int t=0;t<=pos;t++){ a[t]=expf(a[t]-mx); sum+=a[t]; }
                if(smrep>=2) sum=sum+(s0-sum);
                if(smrep>=3) sum=sum+(s1-sum);
            } else {
                for(int t=0;t<=pos;t++){ a[t]=expf(a[t]-mx); sum+=a[t]; }
            }
            float rs=1.0f/sum;
            float* out=s->attout+(size_t)h*HD;
            // ---- Y, the A.V loop.  av2/av3 run the SAME loop 2x/3x from zero.  No scratch
            // buffer and no extra traffic: between passes `out[i]-=out[i]` is exactly +0 for any
            // finite out[i], and because it READS out[i] the previous pass cannot be dead-coded.
            // avrep==1 takes a byte-for-byte copy of the baseline loop, so `none` is unchanged.
            if(!avwrap){
                for(int i=0;i<HD;i++) out[i]=0.0f;
                for(int t=0;t<=pos;t++){
                    const float* vt=s->vcache+((size_t)l*s->maxseq+t)*KVO+(size_t)kvh*HD;
                    float w=a[t]*rs;
                    for(int i=0;i<HD;i++) out[i]+=w*vt[i];
                }
            } else {
                for(int r=0;r<avrep;r++){
                    if(r==0){ for(int i=0;i<HD;i++) out[i]=0.0f; }
                    else    { for(int i=0;i<HD;i++) out[i]-=out[i]; }
                    for(int t=0;t<=pos;t++){
                        const float* vt=s->vcache+((size_t)l*s->maxseq+t)*KVO+(size_t)kvh*HD;
                        float w=a[t]*rs;
                        for(int i=0;i<HD;i++) out[i]+=w*vt[i];
                    }
                }
            }
        }
        // ---- P, priced directly: ONE extra fork/join per layer over the same iteration space,
        // so the cost of the parallel region itself is read against the one already there.
        if(g_attnr==ATTNR_FORK2&&NH<=E5_MAXNH){
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for(int h=0;h<NH;h++) g_forkbuf[h]=(float)h;
            g_sink+=g_forkbuf[0];
        }
        TOC(T_ATTN);
        { TIC; matvec(&L->o,s->attout,NULL,s->xb2);
          for(int i=0;i<D;i++) s->x[i]+=s->xb2[i]; TOC(T_O); }

        { TIC; rmsnorm(s->x,L->post_norm,D,M->rms_eps,s->xb); TOC(T_NORM); }
        { TICW;
          { TICF;
            if(g_fuse) matvec(&L->gateup,s->xb,NULL,s->gubuf);   // one region instead of two
            else     { matvec(&L->gate,s->xb,NULL,s->hb);
                       matvec(&L->up,  s->xb,NULL,s->hb2); }
            TOCF(F_GU); }
          { TICF;
            // E9: the SwiGLU glue was the only loop in the FFN with no parallel region, while
            // every matvec on both sides of it runs on --threads.  530,432 expf per token at
            // Coder-7B, 7.5% of the token, on ONE core.  Elementwise map, no reduction: each i
            // is written once and reads only its own inputs, so splitting the range is
            // BIT-IDENTICAL -- gated on sha256, not on parity.
            if(g_fuse){ const float* g=s->gubuf; const float* u=s->gubuf+F;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                        for(int i=0;i<F;i++) s->hb[i]=silu(g[i])*u[i]; }
            else      {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                        for(int i=0;i<F;i++) s->hb[i]=silu(s->hb[i])*s->hb2[i]; }
            TOCF(F_GLUE); }
          { TICF; matvec(&L->down,s->hb,NULL,s->xb2); TOCF(F_DOWN); }
          { TICF; for(int i=0;i<D;i++) s->x[i]+=s->xb2[i]; TOCF(F_RES); }
          TOCW(T_FFN); }
    }
    { TIC; rmsnorm(s->x,M->final_norm,D,M->rms_eps,s->xb); TOC(T_NORM); }
    TIC;
    if(M->tied){
        mat_t h={M->V,D,0,M->embed,NULL,NULL};   // packed=0: the tied head reads the fp32 embedding
        matvec(&h,s->xb,NULL,s->logits);
    } else {
        matvec(&M->head,s->xb,NULL,s->logits);
    }
    TOC(T_HEAD);
}

// ---------------------------------------------------------------- --selftest-lut
// The kernel is checked against a SCALAR INTEGER reference over the same trits and the same int8
// activations, so any disagreement is the kernel and not the numerics. Case B is the planted
// control: it feeds the reference the OPPOSITE digit order and requires the comparison to FIRE.
// A gate never seen to trip is decoration (feedback_planted_controls).
static int selftest_lut(void){
    const int M=100, K=258;                       // M not a multiple of 32, K/2 odd: exercise padding
    const int H=K/2, Mpad=((M+31)/32)*32;
    int8_t* W=xmalloc((size_t)M*K); int8_t* code=xmalloc((size_t)M*H);
    int8_t* tm=xmalloc((size_t)H*Mpad); float* x=xmalloc((size_t)K*4);
    int8_t* xq=xmalloc((size_t)K); int8_t* lut=xmalloc((size_t)H*16);
    int32_t* y=xmalloc((size_t)Mpad*4); unsigned r=12345u;
    for(int i=0;i<M*K;i++){ r=r*1103515245u+12345u; W[i]=(int8_t)((int)((r>>16)%3)-1); }
    for(int i=0;i<K;i++){ r=r*1103515245u+12345u; x[i]=((float)(r>>8&0xFFFF)/32768.0f-1.0f)*3.7f; }
    for(int m=0;m<M;m++) for(int t=0;t<H;t++)                       // this runtime's byte: low=even
        code[(size_t)m*H+t]=(int8_t)((W[(size_t)m*K+2*t]+1)+3*(W[(size_t)m*K+2*t+1]+1));
    tm_fill(tm,code,M,Mpad,H);   // whichever layout g_lutblk selects: --lutblk --selftest-lut tests
                                 // the blocked one, and the order of the flags matters (arg loop).
    quant_i8(x,K,xq); build_lut(xq,H,lut); matvec_lut(tm,lut,y,M,Mpad,H);
    long bad=0, badB=0;
    for(int m=0;m<M;m++){
        long a=0,b=0;
        for(int k=0;k<K;k++) a+=(long)W[(size_t)m*K+k]*xq[k];
        for(int t=0;t<H;t++) b+=(long)W[(size_t)m*K+2*t]*xq[2*t+1]+(long)W[(size_t)m*K+2*t+1]*xq[2*t];
        if(y[m]!=(int32_t)a) bad++;
        if(y[m]!=(int32_t)b) badB++;
    }
    printf("selftest-lut  A kernel vs scalar-int : %s (%ld/%d rows differ)\n",bad?"FAIL":"PASS",bad,M);
    printf("selftest-lut  B planted (swapped digits) must DIFFER: %s (%ld/%d rows differ)\n",
           badB?"PASS":"FAIL -- the comparison cannot tell a wrong kernel from a right one",badB,M);
    // Case C: the GROUPED kernel. Three groups with three different scales, against a scalar
    // float reference built the same way. Grouping changes the fold, not the trits, so a
    // disagreement here is the fold.
    const int NG=3, GP=(H+NG-1)/NG;
    float gs[3]={0.011f,0.25f,3.0f}; float* yf=xmalloc((size_t)Mpad*4);
    matvec_lut_g(tm,lut,yf,M,Mpad,H,GP,gs,NG);
    double worstC=0;
    for(int m=0;m<M;m++){
        double ref=0;
        for(int t=0;t<H;t++){ int g=t/GP; if(g>=NG) g=NG-1;
            ref+=gs[g]*((double)W[(size_t)m*K+2*t]*xq[2*t]+(double)W[(size_t)m*K+2*t+1]*xq[2*t+1]); }
        double den=fabs(ref)>1e-6?fabs(ref):1e-6, e=fabs(yf[m]-ref)/den;
        if(e>worstC) worstC=e;
    }
    printf("selftest-lut  C grouped kernel (3 scales) vs scalar float : %s (worst rel %.2e)\n",
           worstC<1e-5?"PASS":"FAIL",worstC);
    int ok=(bad==0&&badB>0&&worstC<1e-5);
    printf("selftest-lut  VERDICT: %s\n",ok?"PASS":"FAIL");
    return ok?0:1;
}

// --lut-diag: what int8 does to the ACTIVATIONS, measured on x alone. crest = amax/rms is the
// number that decides whether an amax-scaled int8 grid is usable: the grid has AQ=63 steps
// between 0 and amax, so a vector whose typical element sits at rms gets only 63/crest of them.
static void lut_diag_report(void){
    static const char* nm[2]={"all organs with in=896 (q,k,v,o,gate,up,head)","down_proj (in=4864)"};
    fprintf(stderr,"\n  --lut-diag: activation int8 round-trip, AQ=%d, amax scale per %s\n",
            AQ, g_group>0?g_grouplbl:"whole vector");
    fprintf(stderr,"  %-46s %8s %9s %9s %10s\n","organ group","calls","crest avg","crest max","rel err");
    for(int b=0;b<2;b++){ if(!g_dn[b]) continue;
        fprintf(stderr,"  %-46s %8ld %9.1f %9.1f %9.4f\n",nm[b],g_dn[b],
                g_dcrest[b]/g_dn[b],g_dcmax[b],g_drel[b]/g_dn[b]);
        fprintf(stderr,"  %-46s %8s effective levels at rms (in-group): %.1f of %d\n","","",
                63.0/(g_dcrest[b]/g_dn[b]),AQ);
    }
}

// ------------------------------------------------------------------ modes
static int32_t* read_ids(const char* path,long* n){
    FILE* f=fopen(path,"rb"); if(!f) die("cannot open ids");
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    int32_t* v=xmalloc((size_t)sz); if(fread(v,1,(size_t)sz,f)!=(size_t)sz) die("short ids read");
    fclose(f); *n=sz/4; return v;
}

int main(int argc,char** argv){
    const char* wp=NULL; const char* mode=NULL; const char* arg2=NULL; long arg3=0;
    const char* logout=NULL;
    int threads=1, seqlen=0;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--weights")&&i+1<argc) wp=argv[++i];
        else if(!strcmp(argv[i],"--threads")&&i+1<argc) threads=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--seqlen")&&i+1<argc) seqlen=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--logits")&&i+3<argc){ mode="logits"; arg2=argv[++i]; arg3=atol(argv[++i]); logout=argv[++i]; }
        else if(!strcmp(argv[i],"--bpb")&&i+1<argc){ mode="bpb"; arg2=argv[++i]; }
        else if(!strcmp(argv[i],"--generate")&&i+3<argc){ mode="generate"; arg2=argv[++i];
            arg3=atol(argv[++i]); logout=argv[++i]; }
        else if(!strcmp(argv[i],"--sweep")){        // s11: the ten arms of run 4, in one process
            static const int SA[10]={ATTN_SERIAL,ATTN_SERIAL2,ATTN_AVX4,ATTN_AVX4,ATTN_AVX4,
                                     ATTN_AVX4,ATTN_AVX4,ATTN_AVX4,ATTN_AVX4,ATTN_AVX4};
            static const int SR[10]={ATTNR_NONE,ATTNR_NONE,ATTNR_NONE,ATTNR_SM1,ATTNR_SM2,
                                     ATTNR_SM3,ATTNR_AV1,ATTNR_AV2,ATTNR_AV3,ATTNR_FORK2};
            static const char* SN[10]={"serial","serial2","none","sm1","sm2","sm3","av1","av2",
                                       "av3","fork2"};
            g_sw=10;
            for(int a=0;a<10;a++){ g_sw_attn[a]=SA[a]; g_sw_attnr[a]=SR[a]; g_sw_name[a]=SN[a]; }
        }
        // --sweep6: run 6's ten arms.  All ten are on avx4, so unlike run 5's set they are all
        // bit-identical to `none` and G2a covers the whole sweep.  serial/serial2 are GONE: they
        // are a different code family, and X = organ(none) - R across families came out negative.
        else if(!strcmp(argv[i],"--sweep6")){
            static const int SR6[10]={ATTNR_NONE,ATTNR_SM1,ATTNR_SM2,ATTNR_SM3,ATTNR_AV1,
                                      ATTNR_AV2,ATTNR_AV3,ATTNR_QK1,ATTNR_QK2,ATTNR_QK3};
            static const char* SN6[10]={"none","sm1","sm2","sm3","av1","av2","av3",
                                        "qk1","qk2","qk3"};
            g_sw=10; g_sw_per=0;
            for(int a=0;a<10;a++){ g_sw_attn[a]=ATTN_AVX4; g_sw_attnr[a]=SR6[a];
                                   g_sw_name[a]=SN6[a]; }
        }
        // --sweepd: the price of switching arms, measured.  Four entries, THREE of which run the
        // identical `none` code and differ only in their neighbours; the fourth (sm2) is the
        // stranger that isolates them.  Period 10, schedule  F I F E H H E F I F  :
        //   `none_iso` at slots 1 and 8 -- both neighbours are sm2       mean position 4.5
        //   `none_hot` at slots 4 and 5 -- both neighbours are none      mean position 4.5
        //   `none_edge` at 3 and 6      -- one neighbour of each kind    mean position 4.5
        // so d = organ(none_iso) - organ(none_hot) is a difference between identical code at
        // identical mean context length, and nothing but the neighbourhood differs.
        else if(!strcmp(argv[i],"--sweepd")){
            static const int SRD[4]={ATTNR_NONE,ATTNR_NONE,ATTNR_NONE,ATTNR_SM2};
            static const char* SND[4]={"none_iso","none_hot","none_edge","sm2_filler"};
            static const int SCH[10]={3,0,3,2,1,1,2,3,0,3};
            g_sw=4; g_sw_per=10;
            for(int a=0;a<4;a++){ g_sw_attn[a]=ATTN_AVX4; g_sw_attnr[a]=SRD[a];
                                  g_sw_name[a]=SND[a]; }
            for(int a=0;a<10;a++) g_sw_sched[a]=SCH[a];
        }
        // --sweep8: the eight arms that are BIT-IDENTICAL to each other (all on avx4), for G2.
        // The three --attn arms differ in summation order by construction -- E4 published serial
        // at 166667.2449386003 against avx4's 166667.1361128952 -- so a run that mixes them cannot
        // be bit-identical to either, and asking it to be was a mis-specified gate, not a defect.
        else if(!strcmp(argv[i],"--sweep8")){
            static const int SR8[8]={ATTNR_NONE,ATTNR_SM1,ATTNR_SM2,ATTNR_SM3,
                                     ATTNR_AV1,ATTNR_AV2,ATTNR_AV3,ATTNR_FORK2};
            static const char* SN8[8]={"none","sm1","sm2","sm3","av1","av2","av3","fork2"};
            g_sw=8;
            for(int a=0;a<8;a++){ g_sw_attn[a]=ATTN_AVX4; g_sw_attnr[a]=SR8[a];
                                  g_sw_name[a]=SN8[a]; }
        }
        else if(!strcmp(argv[i],"--bench")&&i+1<argc){ mode="bench"; arg3=atol(argv[++i]); }
        else if(!strcmp(argv[i],"--profile")){ g_prof=1; }
        else if(!strcmp(argv[i],"--witness")){ g_wit=1; }
        else if(!strcmp(argv[i],"--no-witness")){ g_wit=0; }
        // E8. 1 = the original single-chain loop, byte for byte. 2 and 4 break the serial
        // FMA dependency; they are NOT bit-identical and are gated on end-to-end parity.
        else if(!strcmp(argv[i],"--mvacc")&&i+1<argc){ g_mvacc=atoi(argv[++i]);
            if(g_mvacc!=1&&g_mvacc!=2&&g_mvacc!=4) die("--mvacc takes 1, 2 or 4"); }
        else if(!strcmp(argv[i],"--lut")){ g_lut=1; }
        else if(!strcmp(argv[i],"--lutblk")){ g_lut=1; g_lutblk=1; }   // E13
        else if(!strcmp(argv[i],"--lut-diag")){ g_lut=1; g_lutdiag=1; atexit(lut_diag_report); }
        else if(!strcmp(argv[i],"--lut-clip")&&i+1<argc){ g_clip=(float)atof(argv[++i]); }
        else if(!strcmp(argv[i],"--lut-no-down")){ g_lutnodown=1; }
        else if(!strcmp(argv[i],"--lut-no-head")){ g_lutnohead=1; }
        else if(!strcmp(argv[i],"--fuse")){ g_fuse=1; }
        else if(!strcmp(argv[i],"--attn")&&i+1<argc){ const char* v=argv[++i];
            if(!strcmp(v,"serial")) g_attn=ATTN_SERIAL;
            else if(!strcmp(v,"ilp4")) g_attn=ATTN_ILP4;
            else if(!strcmp(v,"avx1")) g_attn=ATTN_AVX1;
            else if(!strcmp(v,"avx4")) g_attn=ATTN_AVX4;
            else if(!strcmp(v,"serial2")) g_attn=ATTN_SERIAL2;
            else if(!strcmp(v,"serial3")) g_attn=ATTN_SERIAL3;
            else if(!strcmp(v,"serial_e3")) g_attn=ATTN_SERIAL_E3;
            else { fprintf(stderr,"--attn: unknown arm %s\n",v); return 1; } }
        else if(!strcmp(argv[i],"--attnr")&&i+1<argc){ const char* v=argv[++i];
            if(!strcmp(v,"none")) g_attnr=ATTNR_NONE;
            else if(!strcmp(v,"sm2")) g_attnr=ATTNR_SM2;
            else if(!strcmp(v,"sm3")) g_attnr=ATTNR_SM3;
            else if(!strcmp(v,"av2")) g_attnr=ATTNR_AV2;
            else if(!strcmp(v,"av3")) g_attnr=ATTNR_AV3;
            else if(!strcmp(v,"fork2")) g_attnr=ATTNR_FORK2;
            else if(!strcmp(v,"sm1")) g_attnr=ATTNR_SM1;
            else if(!strcmp(v,"av1")) g_attnr=ATTNR_AV1;
            else if(!strcmp(v,"qk1")) g_attnr=ATTNR_QK1;
            else if(!strcmp(v,"qk2")) g_attnr=ATTNR_QK2;
            else if(!strcmp(v,"qk3")) g_attnr=ATTNR_QK3;
            else { fprintf(stderr,"--attnr: unknown arm %s\n",v); return 1; } }
        else if(!strcmp(argv[i],"--lut-group")&&i+1<argc){ g_group=atoi(argv[++i]);
            if(g_group&1) die("--lut-group must be even: a 2-trit LUT pair may not straddle a group");
            snprintf(g_grouplbl,sizeof g_grouplbl,"%d channels",g_group); }
        else if(!strcmp(argv[i],"--selftest-lut")){ return selftest_lut(); }
        else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 1; }
    }
    if(!wp||!mode){ fprintf(stderr,
        "usage: donor_engine --weights <bin> [--threads N] [--seqlen N]\n"
        "                    (--logits <ids.bin> <n> <out> | --bpb <ids.bin> | --bench <n>\n"
        "                     | --generate <ids.bin> <n_new> <out_prefix>)\n"); return 1; }
#ifdef _OPENMP
    if(threads<1) threads=1; omp_set_num_threads(threads); omp_set_dynamic(0);
#endif
    model_t M; load(&M,wp);
    if(g_fuse) build_fused(&M);          // BEFORE --lut: the tile-major copy must be of the
                                         // fused matrices, or the LUT path would see the old ones
    if(g_lut){
        if(M.quant!=2) die("--lut requires a --quant packed model (the tile-major copy is a transpose of those bytes)");
        size_t tmb=0; double t0=now_s();
        for(int l=0;l<M.L;l++){ layer_t* L=&M.lay[l];
            mat_t* mm[7];
            if(g_fuse){ mm[0]=&L->qkv; mm[1]=&L->o; mm[2]=&L->gateup; mm[3]=&L->down;
                        mm[4]=mm[5]=mm[6]=NULL; }
            else      { mm[0]=&L->q; mm[1]=&L->k; mm[2]=&L->v; mm[3]=&L->o;
                        mm[4]=&L->gate; mm[5]=&L->up; mm[6]=&L->down; }
            // down_proj is the last entry either way. Its input is the SwiGLU product, whose
            // crest factor (amax/rms) is 15 on average and 70 at worst -- an amax-scaled int8
            // grid leaves it ~4 usable levels of 63, and --lut-diag measures its round-trip
            // error at 2x every other organ's under a per-vector scale (--lut-group 32 removes
            // that asymmetry). --lut-no-down leaves it on the fp32-activation packed kernel.
            const int ndown = g_fuse ? 3 : 6;
            for(int j=0;j<7&&mm[j];j++){ if(j==ndown&&g_lutnodown) continue;
                                  build_tm(mm[j]); tmb+=(size_t)(mm[j]->in/2)*mm[j]->Mpad; } }
        if(!M.tied&&!g_lutnohead){ build_tm(&M.head); tmb+=(size_t)(M.head.in/2)*M.head.Mpad; }
        fprintf(stderr,"  --lut: %s replica built, %.1f MB, %.2f s (activations int8, AQ=%d)\n",
                g_lutblk?"BLOCKED tile-major (E13)":"tile-major",
                tmb/1048576.0,now_s()-t0,AQ);
    }

    if(!strcmp(mode,"bench")){
        // the palindrome balances mean position only over WHOLE periods; refuse a partial one
        // rather than quietly hand the low-numbered arms an extra short context each
        if(g_sw&&arg3%(g_sw_per?g_sw_per:2*g_sw))
            die("--sweep needs --bench divisible by the schedule period");
        state_t s; state_init(&s,&M,arg3+2);
        forward(&M,&s,1,0);                                   // warm
        double t0=now_s();
        for(long i=0;i<arg3;i++){
            double b4[T_N];
            if(g_sw&&g_prof) for(int k=0;k<T_N;k++) b4[k]=g_t[k];
            forward(&M,&s,1+(int)(i%100),(int)(i+1));
            if(g_sw&&g_prof){                       // charge this token to the arm that ran it
                for(int k=0;k<T_N;k++) g_sw_t[g_sw_cur][k]+=g_t[k]-b4[k];
                g_sw_n[g_sw_cur]++;
            }
        }
        double dt=now_s()-t0;
        printf("BENCH  %ld tokens  %.3f s  %.2f tok/s  (threads=%d, %s)",
               arg3,dt,arg3/dt,threads,
               M.quant==3?"tagged":M.quant==2?"packed":M.quant?"ternary":"fp32");
        // the witness, same convention as the profiler's own ffn row (divided by arg3, warm token
        // included) so the two are directly comparable against a plateau measured either way
        if(g_wit&&!g_prof) printf("  ffn~ %.3f ms/tok",g_w0/arg3*1e3*M.L);
        printf("\n");
        if(g_prof){
            double tot=0; for(int k=0;k<T_N;k++) tot+=g_t[k];
            printf("  organ        ms/token   %% of total\n");
            for(int k=0;k<T_N;k++)
                printf("  %-11s %9.3f   %5.1f%%\n",g_tn[k],g_t[k]/arg3*1e3,100.0*g_t[k]/tot);
            printf("  %-11s %9.3f   (organs summed; wall %.3f ms/token)\n",
                   "TOTAL",tot/arg3*1e3,dt/arg3*1e3);
            // E8 G-Z1: the ffn organ, decomposed. SUB-timers -- they are not in the table above
            // and do not enter its percentages. The residual line is the gate: these four must
            // sum to the ffn organ, or one of the brackets is holding work it does not name.
            { double fs=0; for(int k=0;k<F_N;k++) fs+=g_ff[k];
              printf("  -- ffn decomposed --\n");
              for(int k=0;k<F_N;k++)
                  printf("  %-11s %9.3f   %5.1f%% of ffn\n",
                         g_ffn[k],g_ff[k]/arg3*1e3,g_t[T_FFN]>0?100.0*g_ff[k]/g_t[T_FFN]:0.0);
              printf("  %-11s %9.3f   (sum/ffn = %.4f)\n","FFN-SUM",fs/arg3*1e3,
                     g_t[T_FFN]>0?fs/g_t[T_FFN]:0.0); }
            for(int a=0;a<g_sw;a++){                // one line per arm, ms/token, machine-readable
                printf("SWEEP %s n=%ld",g_sw_name[a],g_sw_n[a]);
                for(int k=0;k<T_N;k++)
                    printf(" %s=%.6f",g_tn[k],g_sw_n[a]?g_sw_t[a][k]/(double)g_sw_n[a]*1e3:0.0);
                printf("\n");
            }
        }
        return 0;
    }

    long n=0; int32_t* ids=read_ids(arg2,&n);

    if(!strcmp(mode,"generate")){
        // The generation path is deliberately the SAME forward() the parity gate validated in E1;
        // it differs only in where the next token comes from -- the model instead of the corpus.
        // G-P checks that by writing the prefill logits and comparing them byte-for-byte with
        // what --logits writes for the same ids.
        const long P=n, NG=arg3;
        if(P<1) die("--generate needs at least one prompt token");
        state_t s; state_init(&s,&M,(int)(P+NG+1));
        char pth[1024];
        snprintf(pth,sizeof pth,"%s.prefill.bin",logout);
        FILE* pf=fopen(pth,"wb"); if(!pf) die("cannot open prefill output");
        int32_t* out=xmalloc((size_t)(P+NG)*4);
        for(long i=0;i<P;i++) out[i]=ids[i];

        double t0=now_s();
        for(long i=0;i<P;i++){ forward(&M,&s,ids[i],(int)i);
                               fwrite(s.logits,4,(size_t)M.V,pf); }
        double tpre=now_s()-t0;
        fclose(pf);

        // greedy argmax, first index wins a tie -- no temperature, no seed, so the run is a gate
        double t1=now_s();
        for(long g=0;g<NG;g++){
            int best=0; float mx=s.logits[0];
            for(int i=1;i<M.V;i++) if(s.logits[i]>mx){ mx=s.logits[i]; best=i; }
            out[P+g]=best;
            forward(&M,&s,best,(int)(P+g));       // always: NG decode steps, timed as NG
        }
        double tdec=now_s()-t1;

        snprintf(pth,sizeof pth,"%s.ids.bin",logout);
        FILE* of=fopen(pth,"wb"); if(!of) die("cannot open ids output");
        fwrite(out,4,(size_t)(P+NG),of); fclose(of);

        printf("GEN_PROMPT_TOKENS %ld\nGEN_NEW_TOKENS %ld\n",P,NG);
        printf("GEN_PREFILL_S %.6f\nGEN_PREFILL_TOKS %.3f\n",tpre,P/tpre);
        printf("GEN_DECODE_S %.6f\nGEN_DECODE_TOKS %.3f\n",tdec,NG/tdec);
        printf("GEN_IDS");
        for(long i=0;i<P+NG;i++) printf(" %d",out[i]);
        printf("\n");
        return 0;
    }

    int SL = seqlen>0 ? seqlen : (int)n;
    if(n%SL){ fprintf(stderr,"ids length %ld not divisible by seqlen %d\n",n,SL); return 1; }
    long nseq=n/SL;
    state_t s; state_init(&s,&M,SL+1);

    if(!strcmp(mode,"logits")){
        // to a FILE, never stdout: on Windows stdout is text mode and mangles binary data
        FILE* lo=fopen(logout,"wb"); if(!lo) die("cannot open logits output");
        long m2 = arg3<n?arg3:n;
        for(long i=0;i<m2;i++){ forward(&M,&s,ids[i],(int)i); fwrite(s.logits,4,(size_t)M.V,lo); }
        fclose(lo);
        fprintf(stderr,"wrote %ld logit vectors of %d floats to %s\n",m2,M.V,logout);
        return 0;
    }

    // --bpb: total NLL over next-token prediction, per sequence
    double tot_nats=0.0; long npred=0;
    double t0=now_s();
    for(long q=0;q<nseq;q++){
        const int32_t* seq=ids+q*SL;
        for(int t=0;t<SL;t++){
            forward(&M,&s,seq[t],t);
            if(t+1<SL){
                float mx=-1e30f; for(int i=0;i<M.V;i++) if(s.logits[i]>mx) mx=s.logits[i];
                double sum=0.0; for(int i=0;i<M.V;i++) sum+=exp((double)(s.logits[i]-mx));
                tot_nats += -((double)(s.logits[seq[t+1]]-mx) - log(sum));
                npred++;
            }
        }
        fprintf(stderr,"  seq %ld/%ld  running nats/token %.6f  (%.1fs)\n",
                q+1,nseq,tot_nats/(double)npred,now_s()-t0);
    }
    printf("NATS_TOTAL %.10f\nN_PREDICTED %ld\nNATS_PER_TOKEN %.10f\n",
           tot_nats,npred,tot_nats/(double)npred);
    return 0;
}
