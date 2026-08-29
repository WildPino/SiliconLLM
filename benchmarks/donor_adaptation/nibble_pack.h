// nibble_pack.h — P1: two 2-trit codes per BYTE (2 bits/weight) with the pshufb decode path intact.
//
// BRIEF: docs/research/donor_adaptation/briefs/BRIEF_P1_NIBBLE_PACKING.md  (Stage 1, correctness only)
//
// Today the engine spends one int8_t per 2-trit code c=(w0+1)*3+(w1+1) in [0,8], and _mm256_shuffle_epi8
// selects the LUT entry using only bits [3:0] of that byte. Bits 4..7 are structurally zero and already
// ignored by the hardware. This header packs code(t) into the LOW nibble and code(t+1) into the HIGH nibble
// of the same byte, halving the weight-code stream. The alphabet (base-3, 2 trits/code), the 16-entry LUT
// and the single-pshufb decode are UNCHANGED — only the container changes.
//
// LAYOUT (tile-major, the engine's streamed layout):
//     Tp = (T+1)/2 byte-planes;  codes2[(size_t)(t/2)*Mpad + m] , low nibble = t even, high nibble = t odd
// LAYOUT (row-major, the engine's activation-skip layout):
//     codes2[(size_t)m*Tp + (t/2)] , same nibble convention, no m-padding (M rows exactly)
//
// PADDING (re-derived for this layout, NOT inherited — see brief section 2):
//   * Code 0 does NOT mean "zero weights": c=0 decodes to (w0,w1)=(-1,-1). The neutral code is 4.
//   * Padding lives on the OUTPUT axis m in [M,Mpad). Nibble packing changes only the REDUCTION axis t.
//     The store guard `for(r=0;r<32 && base+r<M;r++)` is on the output axis and is unchanged, so every
//     accumulator lane fed by a padding row is discarded before it is written. Accumulation in
//     acc_add_i8x32 is strictly lane-wise (sign-extend int8 -> int32, add), so a padding lane cannot
//     leak into a live lane. Padding content is therefore arithmetically irrelevant; we still write 0
//     (both nibbles 0) so the array is fully initialised and the encoding is deterministic.
//   * What DOES need re-deriving is the ALLOCATION BOUND, because the plane count changed T -> Tp:
//     allocation is Tp*Mpad bytes; the last 32-byte load is at (Tp-1)*Mpad + base_max, with
//     base_max = 32*((M-1)/32) and base_max+32 <= Mpad because Mpad = (M+31)&~31. In bounds.
//     Mpad MUST remain a multiple of 32 for this; it is, by construction.
//
// ODD T (brief section 2, second trap):
//   When T is odd the final byte-plane tb = Tp-1 carries a real code only in its LOW nibble.
//   The kernel loops full pairs tb in [0, T/2) and then handles the leftover plane with the LOW nibble
//   ONLY: the high nibble of that plane is NEVER LOADED INTO A SHUFFLE. It cannot contaminate the
//   accumulator because no instruction ever reads it, not because its value happens to be benign.
//   The encoder still writes 0 there so the byte is defined.
//
// K parity: T = K/2 truncates, exactly as the existing bc_tm/bc_rm do. Odd K silently drops the last
// weight in BOTH arms — a pre-existing property of the byte path, not introduced here.
//
// REQUIRES from the includer, before #include: acc_add_i8x32() and OMP_PFOR (both are engine.c primitives
// this header deliberately does NOT redefine, so the two arms share one accumulator and one threading rule).
// The scalar types and intrinsics it pulls in itself.
#ifndef NIBBLE_PACK_H
#define NIBBLE_PACK_H
#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>

#define NP_TP(T) (((T)+1)/2)          /* byte-planes needed for T tiles */

/* C4 instrument: bytes of the WEIGHT-CODE array actually fetched, counted as 32 B per _mm256_loadu_si256
   from `codes`. Off by default (zero effect on arithmetic). Single-threaded use only (plain long long). */
static long long g_np_code_bytes = 0;
static int        g_np_count     = 0;
#define NP_CNT() do{ if(g_np_count) g_np_code_bytes += 32; }while(0)

/* ---- encoders ---- */
static void bc_tm_n(const int8_t* Wt,int M,int K,int Mpad,int8_t* codes2){
    int T=K/2, Tp=NP_TP(T);
    for(int tb=0;tb<Tp;tb++){
        int t0=2*tb, t1=t0+1;
        for(int m=0;m<M;m++){
            int a0=Wt[(size_t)m*K+2*t0], a1=Wt[(size_t)m*K+2*t0+1];
            int c0=(a0+1)*3+(a1+1), c1=0;
            if(t1<T){ int b0=Wt[(size_t)m*K+2*t1], b1=Wt[(size_t)m*K+2*t1+1]; c1=(b0+1)*3+(b1+1); }
            codes2[(size_t)tb*Mpad+m]=(int8_t)(c0 | (c1<<4));      /* t1>=T -> high nibble 0, never read */
        }
        for(int m=M;m<Mpad;m++) codes2[(size_t)tb*Mpad+m]=0;       /* padding rows: killed by the store guard */
    }
}
static void bc_rm_n(const int8_t* Wt,int M,int K,int8_t* codes2){
    int T=K/2, Tp=NP_TP(T);
    for(int m=0;m<M;m++) for(int tb=0;tb<Tp;tb++){
        int t0=2*tb, t1=t0+1;
        int a0=Wt[(size_t)m*K+2*t0], a1=Wt[(size_t)m*K+2*t0+1];
        int c0=(a0+1)*3+(a1+1), c1=0;
        if(t1<T){ int b0=Wt[(size_t)m*K+2*t1], b1=Wt[(size_t)m*K+2*t1+1]; c1=(b0+1)*3+(b1+1); }
        codes2[(size_t)m*Tp+tb]=(int8_t)(c0 | (c1<<4));
    }
}
/* scalar read-back of one row-major nibble code (the activation-skip path's accessor) */
static inline int np_rm_code(const int8_t* row,int t){ int b=(unsigned char)row[t>>1]; return (t&1)? (b>>4)&0x0F : b&0x0F; }

/* ---- kernels ----
   one byte-plane -> two pshufb, exactly as today's two bytes -> two pshufb. Per-weight pshufb count unchanged. */
#define NP_PLANE(tb,t0)                                                                                  \
    __m256i b=_mm256_loadu_si256((const __m256i*)(codes2+(size_t)(tb)*Mpad+base)); NP_CNT();              \
    __m256i lo=_mm256_and_si256(b,m0f);                                                                   \
    __m256i tbl0=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)(t0)*16)));     \
    acc_add_i8x32(acc,_mm256_shuffle_epi8(tbl0,lo));

#define NP_PLANE_HI(t1)                                                                                  \
    __m256i hi=_mm256_and_si256(_mm256_srli_epi16(b,4),m0f);                                              \
    __m256i tbl1=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)(t1)*16)));     \
    acc_add_i8x32(acc,_mm256_shuffle_epi8(tbl1,hi));

static void matvec_lut_full_n(const int8_t* codes2,const int8_t* lut,int32_t* y,int M,int Mpad,int T){
    const __m256i m0f=_mm256_set1_epi8(0x0F); int npair=T/2;
    OMP_PFOR for(int base=0;base<M;base+=32){
        __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};
        for(int tb=0;tb<npair;tb++){ NP_PLANE(tb,2*tb) NP_PLANE_HI(2*tb+1) }
        if(T&1){ int tb=npair; NP_PLANE(tb,2*tb) }                 /* odd tail: LOW nibble only, high never read */
        int32_t tmp[32]; _mm256_storeu_si256((__m256i*)(tmp+0),acc[0]); _mm256_storeu_si256((__m256i*)(tmp+8),acc[1]);
        _mm256_storeu_si256((__m256i*)(tmp+16),acc[2]); _mm256_storeu_si256((__m256i*)(tmp+24),acc[3]);
        for(int r=0;r<32&&base+r<M;r++) y[base+r]=tmp[r]; }
}
/* activation tile-skip: act[] lists ACTIVE t, strictly increasing. A plane is loaded iff at least one of
   its two t is active; an inactive t inside a loaded plane is simply NOT shuffled, so the sum is identical
   to the byte path (and would be identical even if it were shuffled, since an inactive t has an all-zero LUT). */
static void matvec_lut_tileskip_n(const int8_t* codes2,const int8_t* lut,int32_t* y,int M,int Mpad,const int* act,int na){
    const __m256i m0f=_mm256_set1_epi8(0x0F);
    OMP_PFOR for(int base=0;base<M;base+=32){
        __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};
        for(int a=0;a<na;){
            int t=act[a], tb=t>>1;
            int pair = (a+1<na && (act[a+1]>>1)==tb);              /* both halves of this plane are active */
            __m256i b=_mm256_loadu_si256((const __m256i*)(codes2+(size_t)tb*Mpad+base)); NP_CNT();
            if(pair||!(t&1)){ __m256i lo=_mm256_and_si256(b,m0f);
                __m256i tt=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)(2*tb)*16)));
                acc_add_i8x32(acc,_mm256_shuffle_epi8(tt,lo)); }
            if(pair||(t&1)){ __m256i hi=_mm256_and_si256(_mm256_srli_epi16(b,4),m0f);
                __m256i tt=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)(2*tb+1)*16)));
                acc_add_i8x32(acc,_mm256_shuffle_epi8(tt,hi)); }
            a += pair?2:1;
        }
        int32_t tmp[32]; _mm256_storeu_si256((__m256i*)(tmp+0),acc[0]); _mm256_storeu_si256((__m256i*)(tmp+8),acc[1]);
        _mm256_storeu_si256((__m256i*)(tmp+16),acc[2]); _mm256_storeu_si256((__m256i*)(tmp+24),acc[3]);
        for(int r=0;r<32&&base+r<M;r++) y[base+r]=tmp[r]; }
}
/* windowed rows [row0,row0+M) of a tile-major block (the MoE per-expert access). row0 must be 32-aligned. */
static void matvec_lut_rows_n(const int8_t* codes2,const int8_t* lut,int32_t* y,int row0,int M,int Mpad,int T){
    const __m256i m0f=_mm256_set1_epi8(0x0F); int npair=T/2;
    OMP_PFOR for(int bb=0;bb<M;bb+=32){ int base=row0+bb;
        __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};
        for(int tb=0;tb<npair;tb++){ NP_PLANE(tb,2*tb) NP_PLANE_HI(2*tb+1) }
        if(T&1){ int tb=npair; NP_PLANE(tb,2*tb) }
        int32_t tmp[32]; _mm256_storeu_si256((__m256i*)(tmp+0),acc[0]); _mm256_storeu_si256((__m256i*)(tmp+8),acc[1]);
        _mm256_storeu_si256((__m256i*)(tmp+16),acc[2]); _mm256_storeu_si256((__m256i*)(tmp+24),acc[3]);
        for(int r=0;r<32&&bb+r<M;r++) y[bb+r]=tmp[r]; }
}
#endif
