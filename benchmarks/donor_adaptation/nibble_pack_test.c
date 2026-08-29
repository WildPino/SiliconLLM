// nibble_pack_test.c — P1 Stage 1 controls C1/C2/C4.  CORRECTNESS ONLY: this harness emits NO timing,
// NO GB/s, NO tok/s. Stage 2 (timing) is gated on a demonstrably clear machine and is not run here.
//
// BRIEF: docs/research/donor_adaptation/briefs/BRIEF_P1_NIBBLE_PACKING.md
// Build: clang -O3 -mavx2 -mfma -march=znver2 benchmarks/donor_adaptation/nibble_pack_test.c \
//              -o bin/nibble_pack_test.exe -lm            (NO -ffast-math, standing law)
//
// C1 BIT-EXACT   nibble kernel vs ref_t3 scalar AND vs today's byte kernel — identical int32, not close.
// C2 PLANTED     hi/lo nibble swap at one live (t,m) MUST break C1. Plus two directed demonstrations of
//                the brief's section-2 derivations (padding rows, odd-T high nibble).
// C4 BYTES       code bytes actually streamed per matvec, both arms, one stated convention.
//                Primary figure = _msize() of the real allocation. Secondary = runtime load counter.
//
// Single-threaded by construction (no -fopenmp): OMP_PFOR expands to nothing, so the plain long long
// byte counters are race-free.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <malloc.h>
#include <immintrin.h>

#define AQ 63
#define OMP_PFOR                      /* single-thread build: the pragma is deliberately absent */

/* ---------------- allocation inventory (every O(input) allocation, ACHIEVED via _msize) ---------------- */
#define INV_MAX 64
static const char* inv_tag[INV_MAX]; static size_t inv_req[INV_MAX], inv_got[INV_MAX]; static int inv_n=0;
static void inv_reset(void){ inv_n=0; }
static void* xmalloc_t(const char* tag,size_t n){
    void* p=malloc(n); if(!p){ fprintf(stderr,"OOM %s %zu\n",tag,n); exit(1); }
    if(inv_n<INV_MAX){ inv_tag[inv_n]=tag; inv_req[inv_n]=n; inv_got[inv_n]=_msize(p); inv_n++; }
    return p;
}
static size_t inv_msize(const char* tag){ for(int i=0;i<inv_n;i++) if(!strcmp(inv_tag[i],tag)) return inv_got[i]; return 0; }
static void inv_print(const char* shape){
    printf("  allocation inventory (ACHIEVED, _msize of the returned block) [%s]\n",shape);
    size_t tot=0;
    for(int i=0;i<inv_n;i++){ printf("    %-14s requested %12zu B   _msize %12zu B\n",inv_tag[i],inv_req[i],inv_got[i]); tot+=inv_got[i]; }
    printf("    %-14s %38zu B (%.2f MiB)\n","TOTAL",tot,tot/1048576.0);
}

/* ---------------- engine.c primitives, copied VERBATIM from benchmarks/phase60/engine.c ----------------
   The ONLY edit vs the engine source is the single `BP_CNT();` line added at each 32-byte code load, so
   the byte arm is charged by exactly the same instrument as the nibble arm (C4). It touches no arithmetic. */
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
static void matvec_lut_tileskip(const int8_t* codes,const int8_t* lut,int32_t* y,int M,int Mpad,const int* act,int na){
    OMP_PFOR for(int base=0;base<M;base+=32){
        __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};
        for(int a=0;a<na;a++){ int t=act[a];
            __m256i tbl=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)t*16)));
            __m256i idx=_mm256_loadu_si256((const __m256i*)(codes+(size_t)t*Mpad+base)); BP_CNT(); acc_add_i8x32(acc,_mm256_shuffle_epi8(tbl,idx)); }
        int32_t tmp[32]; _mm256_storeu_si256((__m256i*)(tmp+0),acc[0]); _mm256_storeu_si256((__m256i*)(tmp+8),acc[1]);
        _mm256_storeu_si256((__m256i*)(tmp+16),acc[2]); _mm256_storeu_si256((__m256i*)(tmp+24),acc[3]);
        for(int r=0;r<32&&base+r<M;r++) y[base+r]=tmp[r]; }
}
static void matvec_lut_rows(const int8_t* codes,const int8_t* lut,int32_t* y,int row0,int M,int Mpad,int T){
    OMP_PFOR for(int bb=0;bb<M;bb+=32){ int base=row0+bb; __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};
        for(int t=0;t<T;t++){ __m256i tbl=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)t*16)));
            __m256i idx=_mm256_loadu_si256((const __m256i*)(codes+(size_t)t*Mpad+base)); BP_CNT(); acc_add_i8x32(acc,_mm256_shuffle_epi8(tbl,idx)); }
        int32_t tmp[32]; _mm256_storeu_si256((__m256i*)(tmp+0),acc[0]); _mm256_storeu_si256((__m256i*)(tmp+8),acc[1]);
        _mm256_storeu_si256((__m256i*)(tmp+16),acc[2]); _mm256_storeu_si256((__m256i*)(tmp+24),acc[3]);
        for(int r=0;r<32&&bb+r<M;r++) y[bb+r]=tmp[r]; }
}
static void build_lut_t3(const int8_t* xq,int T,int8_t* lut){ for(int t=0;t<T;t++){ int8_t x0=xq[2*t],x1=xq[2*t+1];
    for(int c=0;c<16;c++){ int s=0; if(c<9){ int w0=c/3-1,w1=c%3-1; s=w0*x0+w1*x1; } lut[t*16+c]=(int8_t)s; } } }
static void bc_tm(const int8_t* Wt,int M,int K,int Mpad,int8_t* codes){ int T=K/2;
    for(int t=0;t<T;t++){ for(int m=0;m<M;m++){ int w0=Wt[(size_t)m*K+2*t],w1=Wt[(size_t)m*K+2*t+1]; codes[(size_t)t*Mpad+m]=(int8_t)((w0+1)*3+(w1+1)); }
        for(int m=M;m<Mpad;m++) codes[(size_t)t*Mpad+m]=0; } }
static void bc_rm(const int8_t* Wt,int M,int K,int8_t* codes){ int T=K/2;
    for(int m=0;m<M;m++) for(int t=0;t<T;t++){ int w0=Wt[(size_t)m*K+2*t],w1=Wt[(size_t)m*K+2*t+1]; codes[(size_t)m*T+t]=(int8_t)((w0+1)*3+(w1+1)); } }
static void ref_t3(const int8_t* Wt,const int8_t* xq,int32_t* y,int M,int K){
    for(int m=0;m<M;m++){ long s=0; for(int k=0;k<K;k++) s+=(long)Wt[(size_t)m*K+k]*xq[k]; y[m]=(int32_t)s; } }

#include "nibble_pack.h"

/* ---------------- rng: xorshift64*, deterministic, seed printed ---------------- */
static uint64_t rs=0;
static inline uint32_t r32(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (uint32_t)(rs>>32); }

/* ---------------- shapes ---------------- */
typedef struct { const char* name; int M,K; int trials; } Shape;
static Shape SH[]={
    /* engine shapes (Arch-A 8.3M, the model C3 runs on) */
    {"eng_dense_gate_up",    1024,   256, 8},
    {"eng_dense_down",        256,  1024, 8},
    {"eng_moe_gate_up",      4096,   256, 3},
    {"eng_moe_down",          256,   128, 8},
    /* donor width (Llama-3-70B class: d_model 8192, d_ffn 28672) */
    {"donor70b_kv",          1024,  8192, 2},
    {"donor70b_qo",          8192,  8192, 1},
    {"donor70b_gate_up",    28672,  8192, 1},
    {"donor70b_down",        8192, 28672, 1},
    /* odd / awkward: exercise the odd-T tail and every m-blocking tail */
    {"odd_T3_M37",             37,     6, 8},   /* T=3 odd, M<32 -> single partial block */
    {"odd_T513_M97",           97,  1026, 4},   /* T=513 odd, Mpad=128, M not 32-aligned */
    {"odd_T4095_M8191",      8191,  8190, 1},   /* donor-ish, T odd AND M odd */
    {"odd_T1_M1",               1,     2, 8},   /* degenerate: one tile, one row */
    {"M31_T2",                 31,     4, 8},   /* M just under one vector */
    {"M33_T2",                 33,     4, 8},   /* two blocks, second is a 1-row tail */
    {"M32_T7odd",              32,    14, 8},   /* exactly one vector, odd T */
};
#define NSHAPE ((int)(sizeof(SH)/sizeof(SH[0])))

/* fill weights: mode 0 uniform ternary, 1 all -1, 2 all +1, 3 sparse (mostly 0) */
static void fill_w(int8_t* Wt,size_t n,int mode){
    for(size_t i=0;i<n;i++){
        int v; switch(mode){ case 1: v=-1; break; case 2: v=1; break;
            case 3: v=(r32()%8==0)?((int)(r32()%3)-1):0; break; default: v=(int)(r32()%3)-1; }
        Wt[i]=(int8_t)v; }
}
static void fill_x(int8_t* xq,int K,int mode){
    for(int k=0;k<K;k++){ int v;
        switch(mode){ case 1: v=AQ; break; case 2: v=-AQ; break;
            case 3: v=(r32()%3==0)?0:((int)(r32()%(2*AQ+1))-AQ); break; default: v=(int)(r32()%(2*AQ+1))-AQ; }
        xq[k]=(int8_t)v; }
}
static int cmp_i32(const int32_t* a,const int32_t* b,int n,int* first,int32_t* av,int32_t* bv){
    int nd=0; for(int i=0;i<n;i++) if(a[i]!=b[i]){ if(!nd){ *first=i; *av=a[i]; *bv=b[i]; } nd++; }
    return nd;
}

/* ============================ C1 + C4 ============================ */
static int c1_fail=0, c1_shapes=0; static long long c1_checks=0;
static void run_shape(Shape s,int print_inv){
    int M=s.M,K=s.K,T=K/2,Mpad=(M+31)&~31,Tp=NP_TP(T);
    inv_reset();
    int8_t*  Wt   = xmalloc_t("Wt",       (size_t)M*K);
    int8_t*  xq   = xmalloc_t("xq",       (size_t)K);
    int8_t*  lut  = xmalloc_t("lut",      (size_t)T*16);
    int8_t*  cB   = xmalloc_t("codes_byte",(size_t)T*Mpad);
    int8_t*  cN   = xmalloc_t("codes_nib", (size_t)Tp*Mpad);
    int8_t*  rB   = xmalloc_t("rowm_byte", (size_t)M*T);
    int8_t*  rN   = xmalloc_t("rowm_nib",  (size_t)M*Tp);
    int32_t* yref = xmalloc_t("y_ref",    (size_t)M*4);
    int32_t* yB   = xmalloc_t("y_byte",   (size_t)M*4);
    int32_t* yN   = xmalloc_t("y_nib",    (size_t)M*4);
    int*     act  = xmalloc_t("act",      (size_t)T*sizeof(int));

    long long cntB=0,cntN=0; int shape_fail=0;
    for(int tr=0;tr<s.trials;tr++){
        int wm=tr%4, xm=(tr/2)%4;
        fill_w(Wt,(size_t)M*K,wm); fill_x(xq,K,xm);
        bc_tm(Wt,M,K,Mpad,cB); bc_tm_n(Wt,M,K,Mpad,cN);
        bc_rm(Wt,M,K,rB);      bc_rm_n(Wt,M,K,rN);
        build_lut_t3(xq,T,lut);
        ref_t3(Wt,xq,yref,M,K);

        /* --- full matvec --- */
        g_bp_count=g_np_count=1; g_bp_code_bytes=g_np_code_bytes=0;
        matvec_lut_full(cB,lut,yB,M,Mpad,T);
        matvec_lut_full_n(cN,lut,yN,M,Mpad,T);
        if(tr==0){ cntB=g_bp_code_bytes; cntN=g_np_code_bytes; }
        g_bp_count=g_np_count=0;
        int fi; int32_t av,bv;
        int d1=cmp_i32(yN,yref,M,&fi,&av,&bv);
        if(d1){ printf("    FAIL full nib-vs-ref  trial %d w%d x%d: %d rows differ, first m=%d nib=%d ref=%d\n",tr,wm,xm,d1,fi,av,bv); shape_fail=1; }
        int d2=cmp_i32(yN,yB,M,&fi,&av,&bv);
        if(d2){ printf("    FAIL full nib-vs-byte trial %d w%d x%d: %d rows differ, first m=%d nib=%d byte=%d\n",tr,wm,xm,d2,fi,av,bv); shape_fail=1; }
        int d3=cmp_i32(yB,yref,M,&fi,&av,&bv);
        if(d3){ printf("    FAIL full byte-vs-ref trial %d (BASELINE ARM BROKEN): first m=%d byte=%d ref=%d\n",tr,fi,av,bv); shape_fail=1; }
        c1_checks += 3LL*M;

        /* --- tile-skip (active t = those with a non-zero activation pair) --- */
        int na=0; for(int t=0;t<T;t++) if(xq[2*t]||xq[2*t+1]) act[na++]=t;
        matvec_lut_tileskip(cB,lut,yB,M,Mpad,act,na);
        matvec_lut_tileskip_n(cN,lut,yN,M,Mpad,act,na);
        if(cmp_i32(yN,yref,M,&fi,&av,&bv)){ printf("    FAIL skip nib-vs-ref  trial %d: first m=%d nib=%d ref=%d (na=%d/%d)\n",tr,fi,av,bv,na,T); shape_fail=1; }
        if(cmp_i32(yN,yB,M,&fi,&av,&bv)){ printf("    FAIL skip nib-vs-byte trial %d: first m=%d nib=%d byte=%d\n",tr,fi,av,bv); shape_fail=1; }
        c1_checks += 2LL*M;

        /* --- windowed rows (MoE per-expert access); row0 must be 32-aligned --- */
        if(M>=32){ int nrow=32*((M/32)>1?(M/32)/2:1); int row0=32*((M/32)/4);
            if(row0+nrow>M) nrow=M-row0;
            if(nrow>0){
                matvec_lut_rows(cB,lut,yB,row0,nrow,Mpad,T);
                matvec_lut_rows_n(cN,lut,yN,row0,nrow,Mpad,T);
                for(int i=0;i<nrow;i++) if(yN[i]!=yref[row0+i] || yN[i]!=yB[i]){
                    printf("    FAIL rows trial %d row0=%d i=%d nib=%d byte=%d ref=%d\n",tr,row0,i,yN[i],yB[i],yref[row0+i]); shape_fail=1; break; }
                c1_checks += 2LL*nrow; } }

        /* --- row-major scalar path (the engine's up-projection activation skip) --- */
        for(int m=0;m<M;m+=(M>1024?37:1)){ const int8_t* crB=rB+(size_t)m*T; const int8_t* crN=rN+(size_t)m*Tp;
            int SB=0,SN=0; for(int t=0;t<T;t++){ SB+=lut[t*16+crB[t]]; SN+=lut[t*16+np_rm_code(crN,t)]; }
            if(SN!=yref[m] || SN!=SB){ printf("    FAIL rowmajor trial %d m=%d nib=%d byte=%d ref=%d\n",tr,m,SN,SB,yref[m]); shape_fail=1; break; }
            c1_checks += 2; }
    }
    /* C4 for this shape: allocation (_msize) and counted loads, same convention both arms */
    size_t aB=inv_msize("codes_byte"), aN=inv_msize("codes_nib");
    double wpb_B=(double)cntB/((double)M*K), wpb_N=(double)cntN/((double)M*K);
    printf("  %-18s M=%-6d K=%-6d ACHIEVED T=%-6d Tp=%-6d Mpad=%-6d | %s\n",s.name,M,K,T,Tp,Mpad,shape_fail?"C1 FAIL":"C1 PASS");
    printf("      alloc _msize  byte %12zu B   nib %12zu B   ratio %.4fx\n",aB,aN,aN?(double)aB/(double)aN:0.0);
    printf("      loads counted byte %12lld B   nib %12lld B   ratio %.4fx   bits/weight  byte %.4f  nib %.4f\n",
           cntB,cntN,cntN?(double)cntB/(double)cntN:0.0, wpb_B*8.0, wpb_N*8.0);
    if(print_inv) inv_print(s.name);
    if(shape_fail) c1_fail++;
    c1_shapes++;
    free(Wt);free(xq);free(lut);free(cB);free(cN);free(rB);free(rN);free(yref);free(yB);free(yN);free(act);
}

/* ============================ C2 ============================ */
/* C2a: swap the hi/lo nibbles of ONE byte at a LIVE (t,m). C1 must FAIL, by a predicted delta.
   C2b: corrupt EVERY padding byte (m in [M,Mpad)) to 0x88 (= code 8 twice = (+1,+1), maximally non-neutral).
        C1 must still PASS — this is the empirical form of the padding derivation.
   C2c: on an odd-T shape, set the never-read high nibble of the final plane to 8 for every m.
        C1 must still PASS — this is the empirical form of the odd-T derivation. */
static int c2_verdict_ok=1; static int c2b_ran=0, c2c_ran=0;
static void c2_on(Shape s){
    int M=s.M,K=s.K,T=K/2,Mpad=(M+31)&~31,Tp=NP_TP(T);
    int8_t* Wt=malloc((size_t)M*K); int8_t* xq=malloc(K); int8_t* lut=malloc((size_t)T*16);
    int8_t* cN=malloc((size_t)Tp*Mpad); int32_t* yref=malloc((size_t)M*4); int32_t* yN=malloc((size_t)M*4);
    fill_w(Wt,(size_t)M*K,0); fill_x(xq,K,0);
    build_lut_t3(xq,T,lut); ref_t3(Wt,xq,yref,M,K);

    printf("  --- C2 on shape %s (M=%d K=%d T=%d Tp=%d Mpad=%d) ---\n",s.name,M,K,T,Tp,Mpad);

    /* C2a: plant at THREE distinct sites so the control is not merely probing byte 0:
       site 0 = first corruptible byte; site 1 = the last live row m=M-1 (the store-guard tail lane);
       site 2 = the last FULL plane tb=T/2-1 (tail-adjacent). Each plant must fire on its own. */
    int npl=0,nplok=0;
    for(int site=0;site<3;site++){
        int found=0,ftb=-1,fm=-1,flo=0,fhi=0,fdelta=0;
        int tb_lo=0, tb_hi=T/2, m_lo=0;
        if(site==1) m_lo=M-1;
        if(site==2) tb_lo=(T/2>0)?T/2-1:0;
        bc_tm_n(Wt,M,K,Mpad,cN);
        for(int tb=tb_lo;tb<tb_hi && !found;tb++) for(int m=m_lo;m<M && !found;m++){
            int b=(unsigned char)cN[(size_t)tb*Mpad+m]; int lo=b&0x0F, hi=(b>>4)&0x0F;
            if(lo==hi) continue;
            int before=lut[(2*tb)*16+lo]+lut[(2*tb+1)*16+hi];
            int after =lut[(2*tb)*16+hi]+lut[(2*tb+1)*16+lo];
            if(after==before) continue;
            found=1; ftb=tb; fm=m; flo=lo; fhi=hi; fdelta=after-before;
        }
        if(!found){ printf("      C2a site%d: no corruptible byte in this region (skipped, NOT counted as a pass)\n",site); continue; }
        npl++;
        cN[(size_t)ftb*Mpad+fm]=(int8_t)((fhi)|(flo<<4));           /* the plant: hi/lo swapped */
        matvec_lut_full_n(cN,lut,yN,M,Mpad,T);
        int nd=0,fi=-1; for(int m=0;m<M;m++) if(yN[m]!=yref[m]){ if(fi<0)fi=m; nd++; }
        int obs = (fi>=0)? (yN[fi]-yref[fi]) : 0;
        int ok = (nd==1 && fi==fm && obs==fdelta);
        printf("      C2a site%d  swap t=(%d,%d) m=%d lo=%d hi=%d predicted %+d | rows differing=%d observed %+d -> C1 %s [%s]\n",
               site,2*ftb,2*ftb+1,fm,flo,fhi,fdelta,nd,obs,
               nd?"FAILED (control FIRES)":"PASSED (CONTROL DID NOT FIRE -> C1 IS VACUOUS)", ok?"as predicted":"NOT AS PREDICTED");
        if(ok) nplok++;
    }
    if(npl==0){ printf("      C2a VERDICT: VOID - no corruptible byte exists at this shape.\n"); c2_verdict_ok=0; }
    else if(nplok!=npl){ printf("      C2a VERDICT: FAIL (%d/%d plants behaved as predicted)\n",nplok,npl); c2_verdict_ok=0; }
    else printf("      C2a VERDICT: PASS (%d/%d plants fire, each with the exact predicted delta)\n",nplok,npl);

    /* C2b: padding rows corrupted to 0x88 -> C1 must still pass */
    bc_tm_n(Wt,M,K,Mpad,cN);
    long long npad=0;
    for(int tb=0;tb<Tp;tb++) for(int m=M;m<Mpad;m++){ cN[(size_t)tb*Mpad+m]=(int8_t)0x88; npad++; }
    matvec_lut_full_n(cN,lut,yN,M,Mpad,T);
    int ndp=0; for(int m=0;m<M;m++) if(yN[m]!=yref[m]) ndp++;
    if(npad==0)
        printf("      C2b PADDING  n/a at this shape: M=%d == Mpad=%d, so NO padding byte exists. NOT counted as a pass.\n",M,Mpad);
    else{
        printf("      C2b PADDING  %lld padding bytes set to 0x88 (code 8 = w=(+1,+1), the least neutral code)\n",npad);
        printf("      C2b result   rows differing = %d  -> %s\n",ndp, ndp?"FAIL (padding LEAKS - the derivation is wrong)":"PASS (padding is discarded, as derived)");
        if(ndp) c2_verdict_ok=0; else c2b_ran++;
    }

    /* C2c: odd-T high nibble of the final plane -> C1 must still pass */
    if(T&1){
        bc_tm_n(Wt,M,K,Mpad,cN);
        int tb=Tp-1; long long nhi=0;
        for(int m=0;m<Mpad;m++){ int b=(unsigned char)cN[(size_t)tb*Mpad+m]; cN[(size_t)tb*Mpad+m]=(int8_t)((b&0x0F)|(8<<4)); nhi++; }
        matvec_lut_full_n(cN,lut,yN,M,Mpad,T);
        int ndh=0; for(int m=0;m<M;m++) if(yN[m]!=yref[m]) ndh++;
        printf("      C2c ODD-T    high nibble of plane %d (T=%d odd) set to 8 for all %lld bytes\n",tb,T,nhi);
        printf("      C2c result   rows differing = %d  -> %s\n",ndh, ndh?"FAIL (the unread nibble DOES contaminate)":"PASS (never loaded into a shuffle, as derived)");
        if(ndh) c2_verdict_ok=0; else c2c_ran++;
    } else printf("      C2c ODD-T    n/a at this shape (T=%d is even)\n",T);

    free(Wt);free(xq);free(lut);free(cN);free(yref);free(yN);
}

int main(int argc,char** argv){
    uint64_t seed=0x9E3779B97F4A7C15ULL;
    for(int i=1;i<argc;i++) if(!strcmp(argv[i],"--seed")&&i+1<argc) seed=strtoull(argv[++i],NULL,0);
    rs=seed;
    printf("==== P1 Stage 1 — nibble packing correctness (C1/C2/C4). NO timing is produced here. ====\n");
    printf("  seed=0x%016llX  shapes=%d\n",(unsigned long long)seed,NSHAPE);
    printf("\n  C4 CONVENTION (stated once, applied identically to both arms):\n");
    printf("    \"bytes streamed per matvec\" = bytes of the WEIGHT-CODE array fetched by the kernel,\n");
    printf("    counted as 32 B per _mm256_loadu_si256 from `codes`, summed over one complete matvec.\n");
    printf("    EXCLUDED from BOTH arms: the 16 B/tile LUT broadcasts (activation-derived, identical count\n");
    printf("    in both arms, L1-resident), the y output store, and the fp32 scale vectors.\n");
    printf("    bits/weight = 8 * counted_bytes / (M*K), i.e. padding rows are charged to BOTH arms alike.\n");
    printf("    The allocation figure is _msize() of the pointer malloc returned — not T*Mpad recomputed.\n");

    printf("\n==== C1 BIT-EXACT + C4 BYTES (per shape) ====\n");
    for(int i=0;i<NSHAPE;i++) run_shape(SH[i], i<2 || i==6);
    printf("\n  C1 SUMMARY: %lld exact int32 comparisons over %d shapes; shapes failing = %d -> %s\n",
           c1_checks,c1_shapes,c1_fail,c1_fail?"C1 FAIL":"C1 PASS");

    printf("\n==== C2 PLANTED (the instrument must fire on a known positive) ====\n");
    c2_on(SH[0]);                      /* engine shape, T even */
    c2_on(SH[4]);                      /* donor width kv 1024x8192, T even */
    c2_on(SH[9]);                      /* odd T=513 */
    c2_on(SH[10]);                     /* donor width, T=4095 odd AND M=8191 -> 1 padding row per plane */
    c2_on(SH[13]);                     /* M=33 -> 31 padding rows per plane, T even */
    c2_on(SH[14]);                     /* odd T=7, M=32 exactly */
    printf("\n  C2 SUMMARY: %s\n", c2_verdict_ok?"PASS":"FAIL");
    printf("              C2b padding-discard exercised on %d shapes that actually HAVE padding;\n",c2b_ran);
    printf("              C2c odd-T unread-nibble exercised on %d shapes with odd T\n",c2c_ran);
    if(c2b_ran==0||c2c_ran==0){ printf("  C2 SUMMARY: a section-2 derivation was NOT exercised -> treat as UNPROVEN.\n"); c2_verdict_ok=0; }

    int rc = (c1_fail||!c2_verdict_ok)?2:0;
    printf("\n==== STAGE 1 VERDICT: %s ====\n", rc?"STOP — a control did not behave as required":"C1 PASS, C2 PASS, C4 reported");
    printf("STOP. Stage 2 (timing) NOT run: gated on a clear machine.\n");
    return rc;
}
