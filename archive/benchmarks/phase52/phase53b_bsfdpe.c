// Phase 53.B - B-SFDPE: can a LOCAL plasticity overlay (no backprop into substrate, no BPTT) surface a slow
//   SEMANTIC variable that the static read (53.0/53.0b, flat) missed? The frozen SEE substrate stays READ-ONLY;
//   a plastic recurrent overlay h_t is added and the probe reads [x_SEE | h_t]. Plasticity = local FMA + scalar
//   gate + 2 eligibility traces, boundary-gated (PUNCT/NEWLINE), anchored to span(P) where P = top-k Oja axes of
//   the TRAIN x_SEE stream (frozen). This is a TF DECODABILITY probe, NOT a generator. magic 0x5345455E.
//   Run-order: GATE(1) A2-scalar must be flat; GATE(2) A5-scramble must collapse; then CONTENT vs FROZEN/PARITY/
//   BLIND/POS; then ablations A1/A3/A4b/A4c/A6; then kernel-check. Raw numbers only — human reads the verdict.
//   Substrate load-bearing is enforced by ablate(A1)/scramble(A5). Reuses the 53.0b state pipeline + probe + metric.
//
// Phase 53.0b - re-test of the trajectory probe with WITHIN-STORY targets (removes the cross-story confound).
//   53.0 keyed the future window at [t+200,t+400]; TinyStories are ~250 tok, so that window routinely lands in
//   the NEXT story -> FULL "fails" predicting a different story = artifact, not absence of within-story topic.
//   The corpus has NO story delimiter (no <|endoftext|>; '\n\n' marks PARAGRAPHS, which sit inside stories).
//   A paragraph never spans two stories, so within-PARAGRAPH clamping is a CONSERVATIVE within-story guarantee.
//   Targets here: REMAINDER->end-of-paragraph (well-powered "fino a fine-storia") + MID[+10,+40] within-para.
//   Same 5 controls (FULL/POINT/TOKONLY/SCRAMBLE/BLIND), same info metric. magic 0x5345455D.
//
// Phase 53.0 - is the long-range SEMANTIC signal IN THE TRAJECTORY? (supervised probe, cheap, no module)
//
// The 51->52 chain sealed the READ side: fixed-key ~3%, learned-key +0.005, token-context-key = a richer
// n-gram in disguise (52.C.A blind: GLOBAL eats the whole medium-range gain, margin -0.0467). No addressing
// scheme finds long-range structure that isn't already n-gram-able. The remaining question is upstream of
// addressing: does the COMPRESSED TRAJECTORY of the FROZEN substrate even CARRY decodable long-range
// semantics? If a small supervised probe can't read it out, no read-side module can -> rigorous STOP.
//
// INPUT (compressed history, all FIXED / NOT trained):
//   state_t (512D)  +  multiscale EMA of states {a=0.9,0.99,0.999} (3x512)  +  fixed HiPPO-surrogate memory
//   (DH fixed linear leaky integrators h_d = lam_d*h_d + (1-lam_d)*(w_d . state), fixed random w_d, lam_d
//    log-spaced in [0.9,0.9995]).  TRAJ_DIM = 512 + 3*512 + DH.   Nothing here is learned.
//
// TARGETS (supervised, predict the FAR future [t+200, t+400] in token space):
//   HEAD-A  topic/content presence: for K content tokens (freq rank band, function words skipped), does
//           each appear in the far window?  multi-label.  covers (a) story/topic + (c) far content dist.
//   HEAD-B  entity recurrence: the most-recent content token in [t-32,t] -> does it reappear in [t+200,t+400]?
//
// MANTRA CONTROLS (the line made test):
//   FULL      trajectory (state + EMA + HiPPO)                        <- the hypothesis
//   POINT     state_t only (no compressed history)                    <- does the trajectory beat one state?
//   TOKONLY   substrate ablated: fixed VSA embed of last m token-ids  <- is it the substrate or just tokens?
//   SCRAMBLE  FULL but state stream time-shuffled                     <- must collapse to ~base rate
//   BLIND     per-label recurrence logistic (recent-occurrence stats) <- the n-gram-able floor to beat
//
// METRIC: info = baseline_BCE(predict train marginal p_k) - model_BCE, in bits/label on held-out windows.
//         positive = decodable beyond the marginal. Higher = more long-range semantics readable.
//
// DECISION:  FULL >> TOKONLY  AND  FULL >> SCRAMBLE  AND  FULL > POINT  AND  FULL > BLIND
//            -> signal present -> 53.A predictive bottleneck.   Otherwise -> STOP (substrate doesn't carry it;
//            changing the substrate is the forbidden zone, so this is a hard, honest stop).
//
// DISCIPLINE: one trainer, NEW magic 0x5345455C, generator armB UNTOUCHED (this never feeds generation),
//   substrate READ-ONLY (no grad into it), smoke tiny, no commit. STOP + report per target per control.
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase52/phase53_0_probe.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase53_0_probe.exe -lm -I .
// Run:
//   bin/phase53_0_probe.exe <data> <D1_w> <bpe_merges> <outprefix> [--len N] [--epochs E] [--smoke]
//
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include "src/silicon_entropy.h"
#include "benchmarks/phase50/bpe_codec.h"

#define CLASSES   256
#define BASE_DIM  SEE_FEATURE_DIM
#define L0_DIM    SEE_L0_DIM
#define L2_DIM    64
#define D1_TOT    (BASE_DIM + L2_DIM)
#define D_EXP     128
#define GAMMA     0.25f
#define N_TS      2
#define EXP_BANDS (N_TS*D_EXP)
#define S_DIM     (D1_TOT + EXP_BANDS)     // = 512
#define PROJ_SEED 0x48B2EC0DEULL
#define N_VAL     2

// ---- probe / trajectory config ----
#define EMA_NS    3
static const float EMA_A[EMA_NS] = { 0.9f, 0.99f, 0.999f };
#define DH        128
#define TRAJ_DIM  (S_DIM + EMA_NS*S_DIM + DH)   // 512 + 1536 + 128 = 2176
#define KCONT     128       // # content-token labels (HEAD-A)
#define FUNC_SKIP 24        // skip the FUNC_SKIP most frequent tokens (function words)
#define FAR_LO    200
#define FAR_HI    400
#define RECENT    32        // window for "most recent content token" (HEAD-B)
#define BLIND_REC 200       // recent-occurrence window for BLIND features
#define DE_TOK    256       // tokonly VSA embedding dim
#define M_TOK     8         // tokonly: last M token-ids bound
#define HID       64        // probe hidden width

#define RARE_P   1e-4

enum { ST_ALL=0, ST_RECUR, ST_RR, ST_NONREC, ST_D1, ST_D2, ST_D3, ST_D4, NSTRAT };

static const float TS_ALPHA[N_TS] = { 0.90f, 0.99f };
static float Pmat[L2_DIM][BASE_DIM];
static float Omega[D_EXP][L0_DIM];
static float Bvec[D_EXP];
static float (*trigram)[CLASSES][CLASSES];
static float (*ent_table)[CLASSES];
static uint8_t* g_data; static long g_fsz;
static float g_ent_thr; static int g_ent_high=1;
static float Wd1[CLASSES][D1_TOT], Bd1[CLASSES], md1[D1_TOT], sd1[D1_TOT];
static float g_alpha=0.99f, g_l2c_d1=2.0f, g_ls_d1=0.5f;

static Bpe g_bpe;
static uint32_t* g_tok; static long* g_tokstart; static long g_ntok;
static int VTOK=1024;
static double* g_uni; static double g_Ttrain;

static inline float dot_avx(const float* w, const float* f, int n){ __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(&w[i]),_mm256_loadu_ps(&f[i]),s);
    float o[8]; _mm256_storeu_ps(o,s); float r=o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; for(;i<n;i++) r+=w[i]*f[i]; return r; }
static inline void axpy_avx(float* y, const float* x, float a, int n){ __m256 va=_mm256_set1_ps(a); int i=0;
    for(;i<=n-8;i+=8) _mm256_storeu_ps(&y[i], _mm256_fmadd_ps(va,_mm256_loadu_ps(&x[i]),_mm256_loadu_ps(&y[i])));
    for(;i<n;i++) y[i]+=a*x[i]; }

static void gen_projection(uint32_t seed){ uint64_t s=seed?(uint64_t)seed:0x9E3779B97F4A7C15ULL;
    for(int j=0;j<L2_DIM;j++) for(int k=0;k<BASE_DIM;k++){ s^=s<<13; s^=s>>7; s^=s<<17; Pmat[j][k]=(s&1ULL)?1.f:-1.f; } }
static inline double xs_u01(uint64_t* s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return (double)((*s>>11)&0x1FFFFFFFFFFFFFULL)/(double)(1ULL<<53); }
static void gen_lift(uint64_t seed){ uint64_t s=seed?seed:0xABCDEF12345ULL;
    for(int d=0;d<D_EXP;d++){
        for(int k=0;k<L0_DIM;k++){ double u1=xs_u01(&s); if(u1<1e-12) u1=1e-12; double u2=xs_u01(&s);
            Omega[d][k]=(float)(sqrt(-2.0*log(u1))*cos(6.283185307179586*u2)); }
        Bvec[d]=(float)(6.283185307179586*xs_u01(&s)); } }
static inline int ent_gate(uint8_t c1,uint8_t c2){ return g_ent_high?(ent_table[c2][c1]>g_ent_thr):(ent_table[c2][c1]<g_ent_thr); }

static int load_d1(const char* path, SiliconEntropyState* see){
    FILE* f=fopen(path,"rb"); if(!f){ fprintf(stderr,"open %s\n",path); return 0; }
    uint32_t magic; fread(&magic,4,1,f); rewind(f);
    if (magic!=0x53454540){ fprintf(stderr,"expected D1 0x53454540, got 0x%08x\n",magic); return 0; }
    uint32_t hdr4[4]; fread(hdr4,4,4,f); float hf[5]; fread(hf,4,5,f);
    float decay=hf[0], afast=hf[2], fclamp=hf[3];
    uint32_t no=0; fread(&no,4,1,f); int noja=(int)no; float ojab[SEE_N_OJA_MAX*43]; fread(ojab,4,(size_t)noja*43,f);
    uint32_t l2d=0,gt=0,eh=0,ps=0; float al=0,st=0,et=0;
    fread(&l2d,4,1,f); fread(&gt,4,1,f); fread(&al,4,1,f); fread(&st,4,1,f); fread(&et,4,1,f); fread(&eh,4,1,f); fread(&ps,4,1,f);
    g_alpha=al;
    float l2c=0,nbd=1; uint32_t cd=0,dl=0; fread(&l2c,4,1,f); fread(&nbd,4,1,f); fread(&cd,4,1,f); fread(&dl,4,1,f);
    float mx=0; fread(&mx,4,1,f); float ls=1; fread(&ls,4,1,f); float l2cap=0; fread(&l2cap,4,1,f);
    g_l2c_d1=(l2c>0)?l2c:fclamp; g_ls_d1=(ls>0)?ls:1.0f;
    size_t tn=(size_t)CLASSES*CLASSES*CLASSES; trigram=malloc(tn*sizeof(float)); fread(trigram,4,tn,f);
    g_ent_thr=et; g_ent_high=(int)eh; gen_projection(ps);
    see_init(see,42,4,decay); see->multiscale_mode=1; see->alpha_fast=afast; see->alpha_mid=0.9f; see->alpha_slow=0.99f;
    see->n_oja=noja; memcpy(see->W_oja,ojab,(size_t)noja*43*sizeof(float)); see->eta_oja=0.0f; see->plastic_blend=1.0f;
    fread(md1,4,D1_TOT,f); fread(sd1,4,D1_TOT,f);
    fread(Wd1,4,(size_t)CLASSES*D1_TOT,f); fread(Bd1,4,CLASSES,f); fclose(f);
    return 1;
}
static inline void norm_feats(const float* raw, float* out){
    for(int fi=0;fi<D1_TOT;fi++){ float x=(raw[fi]-md1[fi])/(sd1[fi]+1e-8f);
        float cl=(fi<BASE_DIM)?2.0f:g_l2c_d1; if(cl>0){ if(x>cl)x=cl; if(x<-cl)x=-cl; }
        if(fi>=BASE_DIM) x*=g_ls_d1; out[fi]=x; }
}
static inline void armb_fold(const float* feat192, float eB[N_TS][D_EXP]){
    float l0n[L0_DIM];
    for(int k=0;k<L0_DIM;k++){ float x=(feat192[k]-md1[k])/(sd1[k]+1e-8f); if(x>2.f)x=2.f; if(x<-2.f)x=-2.f; l0n[k]=x; }
    for(int d=0;d<D_EXP;d++){ float z=Bvec[d]+GAMMA*dot_avx(Omega[d],l0n,L0_DIM); float cz=cosf(z);
        for(int ts=0;ts<N_TS;ts++){ float a=TS_ALPHA[ts]; eB[ts][d]=a*eB[ts][d]+(1.0f-a)*cz; } }
}
static inline void row_bands(const float eB[N_TS][D_EXP], float* row){
    for(int ts=0;ts<N_TS;ts++) memcpy(row+D1_TOT+ts*D_EXP, eB[ts], D_EXP*4);
}
static void tokenize_corpus(void){
    g_tok=(uint32_t*)malloc((size_t)g_fsz*sizeof(uint32_t));
    g_tokstart=(long*)malloc((size_t)(g_fsz+1)*sizeof(long));
    g_ntok=(long)bpe_encode_region(&g_bpe, g_data, 0, g_fsz, g_tok);
    long off=0; for(long i=0;i<g_ntok;i++){ g_tokstart[i]=off; off+=bpe_tok_len(&g_bpe,g_tok[i]); }
    g_tokstart[g_ntok]=off;
    fprintf(stderr,"tokenized: %ld tokens over %ld bytes (%.3f b/tok)\n",g_ntok,g_fsz,(double)g_fsz/g_ntok);
}
static void build_unigram(long tok_train_end){
    g_uni=(double*)calloc(VTOK,sizeof(double)); g_Ttrain=(double)tok_train_end;
    for(long i=0;i<tok_train_end;i++) g_uni[g_tok[i]]+=1.0;
}

// ---- paragraph boundaries (the only reliable within-story unit: '\n\n'; a paragraph never spans 2 stories,
//      so within-paragraph clamping is a CONSERVATIVE within-story guarantee. no '<|endoftext|>' in corpus) ----
static long* g_brkpos; static long g_nbrk;
static void build_breaks(void){
    g_brkpos=malloc((size_t)(g_fsz/8+16)*sizeof(long)); g_nbrk=0; size_t cap=g_fsz/8+16;
    for(long b=0;b+1<g_fsz;b++){ if(g_data[b]=='\n'&&g_data[b+1]=='\n'&&(b==0||g_data[b-1]!='\n')){
        if((size_t)g_nbrk>=cap){ cap*=2; g_brkpos=realloc(g_brkpos,cap*sizeof(long)); } g_brkpos[g_nbrk++]=b; } }
    fprintf(stderr,"paragraph boundaries: %ld over %ld bytes (~%.0f tok/para)\n",g_nbrk,g_fsz,
            g_nbrk>0?(double)g_ntok/g_nbrk:0.0);
}
static long brk_at(long bytepos){ long lo=0,hi=g_nbrk; while(lo<hi){ long m=(lo+hi)/2; if(g_brkpos[m]<bytepos) lo=m+1; else hi=m; } return lo; }

// ---- extract: state rows + token at row (prev) + next token (tgt) + token index (rowtok) ----  (regime = 52.C)
static long extract(SiliconEntropyState* see, long start, long N, float* X, uint32_t* tgt, uint32_t* prev, long* rowtok){
    float L2d1[L2_DIM]={0}, pb_d1[BASE_DIM]={0}, eB[N_TS][D_EXP]; memset(eB,0,sizeof eB);
    float feat192[BASE_DIM], fa[BASE_DIM], rawd1[D1_TOT], nf[D1_TOT];
    float scale=1.0f/sqrtf((float)BASE_DIM); uint8_t cur_c2,cur_c1;
    long ti=0; { long lo=0,hi=g_ntok; while(lo<hi){ long mid=(lo+hi)/2; if(g_tokstart[mid]<start) lo=mid+1; else hi=mid; } ti=lo; }
    long bstart=g_tokstart[ti];
    see_reset(see); for(long i=0;i<bstart;i++) see_observe(see,g_data[i]);
    cur_c2=(bstart>=2)?g_data[bstart-2]:0; cur_c1=(bstart>=1)?g_data[bstart-1]:0;
    long rows=0;
    while(ti+1<g_ntok && g_tokstart[ti+1]+ (long)bpe_tok_len(&g_bpe,g_tok[ti+1]) <= start+N){
        see_extract(see,feat192);
        memcpy(rawd1,feat192,BASE_DIM*4); memcpy(rawd1+BASE_DIM,L2d1,L2_DIM*4);
        norm_feats(rawd1,nf);
        float* row=&X[(size_t)rows*S_DIM]; memcpy(row,nf,D1_TOT*4); row_bands(eB,row);
        tgt[rows]=g_tok[ti+1]; prev[rows]=g_tok[ti]; if(rowtok) rowtok[rows]=ti;
        rows++;
        int L=bpe_tok_len(&g_bpe,g_tok[ti]); const unsigned char* eb=bpe_tok_bytes(&g_bpe,g_tok[ti]);
        for(int k=0;k<L;k++){ uint8_t ob=eb[k];
            see_extract(see,feat192); armb_fold(feat192,eB);
            see_observe(see,ob); see_extract(see,fa);
            if(ent_gate(cur_c1,cur_c2)){ float src5[BASE_DIM];
                for(int kk=0;kk<BASE_DIM;kk++) src5[kk]=fa[kk]-0.5f*pb_d1[kk]; memcpy(pb_d1,fa,BASE_DIM*4);
                for(int j=0;j<L2_DIM;j++){ float p5=0; const float* pj=Pmat[j]; for(int kk=0;kk<BASE_DIM;kk++) p5+=pj[kk]*src5[kk];
                    L2d1[j]=g_alpha*L2d1[j]+(1.0f-g_alpha)*p5*scale; } }
            cur_c2=cur_c1; cur_c1=ob; }
        ti++;
    }
    return rows;
}

// ============ content-label map + fixed trajectory/tokonly assets ============
static int g_tok2lab[1<<20];   // VTOK <= 1M; -1 if not a content token
static uint32_t g_lab2tok[KCONT];
static float Hw[DH][S_DIM]; static float Hlam[DH];
static float* g_E2;  // [VTOK*DE_TOK] fixed bipolar token embeddings for TOKONLY

static int cmp_freq(const void* a,const void* b){ int x=*(const int*)a,y=*(const int*)b;
    double da=g_uni[x],db=g_uni[y]; return (da<db)-(da>db); }
static void build_content_labels(void){
    int* idx=malloc((size_t)VTOK*sizeof(int)); for(int v=0;v<VTOK;v++) idx[v]=v;
    qsort(idx,VTOK,sizeof(int),cmp_freq);
    for(int v=0;v<VTOK;v++) g_tok2lab[v]=-1;
    int got=0;
    for(int r=FUNC_SKIP; r<VTOK && got<KCONT; r++){ int v=idx[r]; if(g_uni[v]<=0) continue;
        g_tok2lab[v]=got; g_lab2tok[got]=v; got++; }
    fprintf(stderr,"content labels: %d (freq rank %d..%d)\n",got,FUNC_SKIP,FUNC_SKIP+got);
    free(idx);
}
static void hippo_init(uint64_t seed){ uint64_t s=seed?seed:0x4190C0DEULL;
    for(int d=0;d<DH;d++){ float nn=0;
        for(int k=0;k<S_DIM;k++){ s^=s<<13;s^=s>>7;s^=s<<17; Hw[d][k]=(s&1ULL)?1.f:-1.f; nn+=1.0f; }
        float inv=1.0f/sqrtf(nn); for(int k=0;k<S_DIM;k++) Hw[d][k]*=inv;
        double f=(DH>1)?(double)d/(DH-1):0.0; Hlam[d]=(float)exp(log(0.9)*(1.0-f)+log(0.9995)*f); }
}
static void tokemb_init(uint64_t seed){ g_E2=malloc((size_t)VTOK*DE_TOK*4); uint64_t s=seed?seed:0xE2C0DEULL;
    for(size_t i=0;i<(size_t)VTOK*DE_TOK;i++){ s^=s<<13;s^=s>>7;s^=s<<17; g_E2[i]=(s&1ULL)?1.f:-1.f; } }

// build the FULL/SCRAMBLE trajectory rep. order=NULL -> causal identity; else state stream consumed in
// order[] (time-shuffled) while labels stay at position ridx -> SCRAMBLE control.
static void build_traj(const float* X, long nrow, const long* order, float* T){
    float ema[EMA_NS][S_DIM]; memset(ema,0,sizeof ema);
    float h[DH]; memset(h,0,sizeof h);
    for(long ridx=0; ridx<nrow; ridx++){
        long r = order? order[ridx] : ridx;
        const float* st=&X[(size_t)r*S_DIM];
        for(int s=0;s<EMA_NS;s++){ float a=EMA_A[s]; float* e=ema[s];
            for(int k=0;k<S_DIM;k++) e[k]=a*e[k]+(1.0f-a)*st[k]; }
        for(int d=0;d<DH;d++){ float proj=dot_avx(Hw[d],st,S_DIM); h[d]=Hlam[d]*h[d]+(1.0f-Hlam[d])*proj; }
        float* o=&T[(size_t)ridx*TRAJ_DIM];
        memcpy(o, st, S_DIM*4);
        for(int s=0;s<EMA_NS;s++) memcpy(o+S_DIM+(size_t)s*S_DIM, ema[s], S_DIM*4);
        memcpy(o+S_DIM+(size_t)EMA_NS*S_DIM, h, DH*4);
    }
}
static void build_point(const float* X, long nrow, float* U){
    for(long r=0;r<nrow;r++) memcpy(&U[(size_t)r*S_DIM],&X[(size_t)r*S_DIM],S_DIM*4);
}
// TOKONLY: substrate ABLATED. rep = normalize( sum_{j<M} cyclic_shift_j( E2[prev_{r-j}] ) )
static void build_tokonly(const uint32_t* prev, long nrow, float* U){
    for(long r=0;r<nrow;r++){ float* a=&U[(size_t)r*DE_TOK]; for(int d=0;d<DE_TOK;d++) a[d]=0.f;
        for(int j=0;j<M_TOK && r-j>=0; j++){ const float* e=&g_E2[(size_t)prev[r-j]*DE_TOK];
            for(int d=0;d<DE_TOK;d++){ int dd=d+j; if(dd>=DE_TOK) dd-=DE_TOK; a[dd]+=e[d]; } }
        float nn=0; for(int d=0;d<DE_TOK;d++) nn+=a[d]*a[d]; float inv=1.0f/(sqrtf(nn)+1e-12f);
        for(int d=0;d<DE_TOK;d++) a[d]*=inv; }
}

// ============ labels (target-driven, with per-row active mask mA / mB) ============
// Tgt modes:  M_FIXED   = window [r+lo, r+hi]; if clamp, EXCLUDE rows whose window crosses a paragraph break.
//             M_REMAINDER = window [r+lo, end-of-paragraph]; require remaining >= minrem (within-story by constr.)
enum { M_FIXED=0, M_REMAINDER };
typedef struct { const char* name; int lo; int mode; int hi_or_minrem; int clamp; } Tgt;
// brk[r] = # paragraph boundaries before row r's byte start (monotone). same paragraph <=> equal brk.
static long build_labels_tgt(const Tgt* T,const uint32_t* prev,const long* brk,long nrow,
                             uint8_t* yA,uint8_t* mA,uint8_t* yB,uint8_t* mB){
    long nact=0;
    for(long r=0;r<nrow;r++){
        uint8_t* ya=&yA[(size_t)r*KCONT]; for(int k=0;k<KCONT;k++) ya[k]=0; mA[r]=0; mB[r]=0; yB[r]=0;
        long lo=r+T->lo, hi;
        if(T->mode==M_FIXED){ hi=r+T->hi_or_minrem; if(hi>=nrow) continue;
            if(T->clamp && brk[hi]!=brk[r]) continue; }
        else { long e=r; while(e+1<nrow && brk[e+1]==brk[r]) e++;   // last row of current paragraph
            hi=e; if(hi-lo < T->hi_or_minrem) continue; }
        if(lo>=nrow||hi<lo) continue; if(lo<0) lo=0;
        mA[r]=1; nact++;
        for(long j=lo;j<=hi;j++){ int lab=g_tok2lab[prev[j]]; if(lab>=0) ya[lab]=1; }
        int rctok=-1; for(long j=(r>=RECENT?r-RECENT:0);j<=r;j++){ int lab=g_tok2lab[prev[j]]; if(lab>=0) rctok=(int)prev[j]; }
        if(rctok>=0){ mB[r]=1; int pres=0; for(long j=lo;j<=hi;j++) if((int)prev[j]==rctok){ pres=1; break; } yB[r]=(uint8_t)pres; }
    }
    return nact;
}

// ============ probe: trunk H -> headA (KCONT sigmoid) + headB (1 sigmoid) ============
typedef struct { int in,H; float *W1,*b1,*WA,*bA,*Wb; float bb;
    float *mW1,*vW1,*mb1,*vb1,*mWA,*vWA,*mbA,*vbA,*mWb,*vWb; float mbb,vbb; int t; } Probe;
static void probe_init(Probe* p,int in,uint64_t seed){ p->in=in; p->H=HID; p->t=0;
    size_t s1=(size_t)HID*in, sA=(size_t)KCONT*HID;
    p->W1=calloc(s1,4);p->b1=calloc(HID,4);p->WA=calloc(sA,4);p->bA=calloc(KCONT,4);p->Wb=calloc(HID,4);p->bb=0;
    p->mW1=calloc(s1,4);p->vW1=calloc(s1,4);p->mb1=calloc(HID,4);p->vb1=calloc(HID,4);
    p->mWA=calloc(sA,4);p->vWA=calloc(sA,4);p->mbA=calloc(KCONT,4);p->vbA=calloc(KCONT,4);
    p->mWb=calloc(HID,4);p->vWb=calloc(HID,4);p->mbb=0;p->vbb=0;
    uint64_t r=seed?seed:0x53ULL; float sc1=sqrtf(2.0f/in);
    for(size_t i=0;i<s1;i++){ r^=r<<13;r^=r>>7;r^=r<<17; p->W1[i]=sc1*(((r>>11)*(1.0/(1ULL<<53)))*2-1); }
    float scA=sqrtf(2.0f/HID);
    for(size_t i=0;i<sA;i++){ r^=r<<13;r^=r>>7;r^=r<<17; p->WA[i]=scA*(((r>>11)*(1.0/(1ULL<<53)))*2-1); }
    for(int j=0;j<HID;j++){ r^=r<<13;r^=r>>7;r^=r<<17; p->Wb[j]=scA*(((r>>11)*(1.0/(1ULL<<53)))*2-1); }
}
static void probe_free(Probe* p){ free(p->W1);free(p->b1);free(p->WA);free(p->bA);free(p->Wb);
    free(p->mW1);free(p->vW1);free(p->mb1);free(p->vb1);free(p->mWA);free(p->vWA);free(p->mbA);free(p->vbA);free(p->mWb);free(p->vWb); }
static inline void probe_fwd(const Probe* p,const float* u,float* hid,float* la,float* lb){
    for(int j=0;j<p->H;j++){ float a=p->b1[j]+dot_avx(&p->W1[(size_t)j*p->in],u,p->in); hid[j]=a>0?a:0; }
    for(int k=0;k<KCONT;k++) la[k]=p->bA[k]+dot_avx(&p->WA[(size_t)k*p->H],hid,p->H);
    *lb=p->bb+dot_avx(p->Wb,hid,p->H);
}
static inline float sigf(float x){ if(x>30)return 1.f; if(x<-30)return 0.f; return 1.0f/(1.0f+expf(-x)); }
#define ADAMP(P,GG,MM,VV,NN) for(size_t z=0;z<(size_t)(NN);z++){ MM[z]=.9f*MM[z]+.1f*GG[z]; VV[z]=.999f*VV[z]+.001f*GG[z]*GG[z]; P[z]-=lt*(MM[z]/(sqrtf(VV[z])+1e-8f)+1e-5f*P[z]); }
static void probe_train(Probe* p,const float* U,const uint8_t* yA,const uint8_t* mA,const uint8_t* yB,const uint8_t* mB,
                        long used,int epochs,float lr){
    int H=p->H,in=p->in; size_t s1=(size_t)H*in, sA=(size_t)KCONT*H;
    float *gW1=malloc(s1*4),*gb1=malloc(H*4),*gWA=malloc(sA*4),*gbA=malloc(KCONT*4),*gWb=malloc(H*4),gbb;
    float *hid=malloc(H*4),*dh=malloc(H*4),*la=malloc(KCONT*4); float lb;
    int bs=256; float invn=1.0f/bs; float lamB=(float)KCONT*0.25f; // weight HEAD-B so it isn't drowned by K labels
    for(int ep=0;ep<epochs;ep++){
        memset(gW1,0,s1*4);memset(gb1,0,H*4);memset(gWA,0,sA*4);memset(gbA,0,KCONT*4);memset(gWb,0,H*4);gbb=0; long inb=0;
        for(long i=0;i<used;i++){
            if(!mA[i] && !mB[i]) continue;   // row inactive for both heads
            const float* u=&U[(size_t)i*in];
            probe_fwd(p,u,hid,la,&lb);
            memset(dh,0,H*4);
            if(mA[i]){ const uint8_t* ya=&yA[(size_t)i*KCONT];
            for(int k=0;k<KCONT;k++){ float pr=sigf(la[k]); float e=(pr-ya[k])*invn;
                gbA[k]+=e; float* gw=&gWA[(size_t)k*H]; const float* w=&p->WA[(size_t)k*H];
                for(int j=0;j<H;j++){ gw[j]+=e*hid[j]; dh[j]+=e*w[j]; } } }
            if(mB[i]){ float pr=sigf(lb); float e=(pr-yB[i])*invn*lamB;
                gbb+=e; for(int j=0;j<H;j++){ gWb[j]+=e*hid[j]; dh[j]+=e*p->Wb[j]; } }
            for(int j=0;j<H;j++) if(hid[j]>0){ gb1[j]+=dh[j]; float* gw=&gW1[(size_t)j*in]; const float g=dh[j];
                for(int k=0;k<in;k++) gw[k]+=g*u[k]; }
            inb++;
            if(inb==bs || i==used-1){ p->t++; float lt=lr*sqrtf(1-powf(.999f,p->t))/(1-powf(.9f,p->t));
                ADAMP(p->W1,gW1,p->mW1,p->vW1,s1); ADAMP(p->b1,gb1,p->mb1,p->vb1,H);
                ADAMP(p->WA,gWA,p->mWA,p->vWA,sA); ADAMP(p->bA,gbA,p->mbA,p->vbA,KCONT);
                ADAMP(p->Wb,gWb,p->mWb,p->vWb,H);
                p->mbb=.9f*p->mbb+.1f*gbb; p->vbb=.999f*p->vbb+.001f*gbb*gbb; p->bb-=lt*(p->mbb/(sqrtf(p->vbb)+1e-8f)+1e-5f*p->bb);
                memset(gW1,0,s1*4);memset(gb1,0,H*4);memset(gWA,0,sA*4);memset(gbA,0,KCONT*4);memset(gWb,0,H*4);gbb=0; inb=0; }
        }
    }
    free(gW1);free(gb1);free(gWA);free(gbA);free(gWb);free(hid);free(dh);free(la);
}
// info (bits/label) = baseline_BCE(predict pA[k]) - model_BCE, averaged over labels & rows. Also HEAD-B.
static void probe_eval(const Probe* p,const float* U,const uint8_t* yA,const uint8_t* mA,const uint8_t* yB,const uint8_t* mB,
                       long used,const double* pA,double pB,
                       double* o_baseA,double* o_modA,long* o_nA,double* o_baseB,double* o_modB,long* o_nB){
    int H=p->H,in=p->in; float *hid=malloc(H*4),*la=malloc(KCONT*4); float lb;
    double baseA=0,modA=0,baseB=0,modB=0; long nB=0,nA=0;
    double lpA[KCONT],l1pA[KCONT]; for(int k=0;k<KCONT;k++){ double q=pA[k]; if(q<1e-6)q=1e-6; if(q>1-1e-6)q=1-1e-6; lpA[k]=-log2(q); l1pA[k]=-log2(1-q); }
    double lpB=-log2((pB<1e-6?1e-6:(pB>1-1e-6?1-1e-6:pB))), l1pB=-log2((1-pB<1e-6?1e-6:(1-pB>1-1e-6?1-1e-6:1-pB)));
    for(long i=0;i<used;i++){
        if(!mA[i] && !mB[i]) continue;
        const float* u=&U[(size_t)i*in]; probe_fwd(p,u,hid,la,&lb);
        if(mA[i]){ const uint8_t* ya=&yA[(size_t)i*KCONT]; nA++;
        for(int k=0;k<KCONT;k++){ double pr=sigf(la[k]); if(pr<1e-6)pr=1e-6; if(pr>1-1e-6)pr=1-1e-6;
            if(ya[k]){ modA+=-log2(pr); baseA+=lpA[k]; } else { modA+=-log2(1-pr); baseA+=l1pA[k]; } } }
        if(mB[i]){ double pr=sigf(lb); if(pr<1e-6)pr=1e-6; if(pr>1-1e-6)pr=1-1e-6;
            if(yB[i]){ modB+=-log2(pr); baseB+=lpB; } else { modB+=-log2(1-pr); baseB+=l1pB; } nB++; }
    }
    free(hid);free(la);
    *o_baseA+=baseA; *o_modA+=modA; *o_nA+=nA; *o_baseB+=baseB; *o_modB+=modB; *o_nB+=nB;
}

// ============ BLIND: per-label recurrence logistic on [recent_seen_k, log1p(recent_count_k)] ============
// p = sigmoid(a_k + b_k*seen + c_k*lcount). The n-gram-able floor: "recently seen content token -> reappears".
static void blind_features(const uint32_t* prev, long used, float* seen, float* lcnt){
    for(long r=0;r<used;r++){
        int cnt[KCONT]; for(int k=0;k<KCONT;k++) cnt[k]=0;
        for(long j=(r>=BLIND_REC?r-BLIND_REC:0); j<=r; j++){ int lab=g_tok2lab[prev[j]]; if(lab>=0) cnt[lab]++; }
        float* se=&seen[(size_t)r*KCONT]; float* lc=&lcnt[(size_t)r*KCONT];
        for(int k=0;k<KCONT;k++){ se[k]=cnt[k]>0?1.f:0.f; lc[k]=logf(1.0f+(float)cnt[k]); }
    }
}
static void blind_run(const float* seenTr,const float* lcntTr,const uint8_t* yATr,const uint8_t* mATr,long usedTr,
                      const double* pA, float* a,float* b,float* c, int epochs,float lr){
    for(int k=0;k<KCONT;k++){ double q=pA[k]; if(q<1e-6)q=1e-6; if(q>1-1e-6)q=1-1e-6; a[k]=(float)log(q/(1-q)); b[k]=0;c[k]=0; }
    float* ma=calloc(KCONT,4);float* mb=calloc(KCONT,4);float* mc=calloc(KCONT,4);
    float* va=calloc(KCONT,4);float* vb=calloc(KCONT,4);float* vc=calloc(KCONT,4);
    int t=0; int bs=256; float invn=1.0f/bs;
    for(int ep=0;ep<epochs;ep++){
        float* ga=calloc(KCONT,4);float* gb=calloc(KCONT,4);float* gc=calloc(KCONT,4); long inb=0;
        for(long r=0;r<usedTr;r++){ if(!mATr[r]) continue; const float* se=&seenTr[(size_t)r*KCONT]; const float* lc=&lcntTr[(size_t)r*KCONT];
            const uint8_t* ya=&yATr[(size_t)r*KCONT];
            for(int k=0;k<KCONT;k++){ float z=a[k]+b[k]*se[k]+c[k]*lc[k]; float pr=sigf(z); float e=(pr-ya[k])*invn;
                ga[k]+=e; gb[k]+=e*se[k]; gc[k]+=e*lc[k]; }
            inb++;
            if(inb==bs||r==usedTr-1){ t++; float lt=lr*sqrtf(1-powf(.999f,t))/(1-powf(.9f,t));
                for(int k=0;k<KCONT;k++){ ma[k]=.9f*ma[k]+.1f*ga[k];va[k]=.999f*va[k]+.001f*ga[k]*ga[k];a[k]-=lt*ma[k]/(sqrtf(va[k])+1e-8f);
                    mb[k]=.9f*mb[k]+.1f*gb[k];vb[k]=.999f*vb[k]+.001f*gb[k]*gb[k];b[k]-=lt*mb[k]/(sqrtf(vb[k])+1e-8f);
                    mc[k]=.9f*mc[k]+.1f*gc[k];vc[k]=.999f*vc[k]+.001f*gc[k]*gc[k];c[k]-=lt*mc[k]/(sqrtf(vc[k])+1e-8f); }
                memset(ga,0,KCONT*4);memset(gb,0,KCONT*4);memset(gc,0,KCONT*4); inb=0; }
        }
        free(ga);free(gb);free(gc);
    }
    free(ma);free(mb);free(mc);free(va);free(vb);free(vc);
}
static void blind_eval(const float* seen,const float* lcnt,const uint8_t* yA,const uint8_t* mA,long used,
                       const double* pA,const float* a,const float* b,const float* c,
                       double* o_base,double* o_mod,long* o_nA){
    double base=0,mod=0; long nA=0; double lpA[KCONT],l1pA[KCONT];
    for(int k=0;k<KCONT;k++){ double q=pA[k]; if(q<1e-6)q=1e-6; if(q>1-1e-6)q=1-1e-6; lpA[k]=-log2(q); l1pA[k]=-log2(1-q); }
    for(long r=0;r<used;r++){ if(!mA[r]) continue; const float* se=&seen[(size_t)r*KCONT]; const float* lc=&lcnt[(size_t)r*KCONT];
        const uint8_t* ya=&yA[(size_t)r*KCONT]; nA++;
        for(int k=0;k<KCONT;k++){ double z=a[k]+b[k]*se[k]+c[k]*lc[k]; double pr=sigf((float)z); if(pr<1e-6)pr=1e-6;if(pr>1-1e-6)pr=1-1e-6;
            if(ya[k]){ mod+=-log2(pr); base+=lpA[k]; } else { mod+=-log2(1-pr); base+=l1pA[k]; } } }
    *o_base+=base; *o_mod+=mod; *o_nA+=nA;
}

// ============ data windows ============
static float *g_Xtr,*g_Xv[N_VAL];
static uint32_t *g_ttr,*g_ptr,*g_tv[N_VAL],*g_pv[N_VAL];
static long *g_rtTr,*g_rtV[N_VAL];        // rowtok (token index per row)
static long *g_brkTr,*g_brkV[N_VAL];      // paragraph-break count before each row
static long g_trrows,g_vrows[N_VAL], g_maxrows;
static uint8_t *g_yAtr,*g_mAtr,*g_yBtr,*g_mBtr, *g_yAv[N_VAL],*g_mAv[N_VAL],*g_yBv[N_VAL],*g_mBv[N_VAL];
static long g_nactTr, g_nactV[N_VAL];     // active HEAD-A rows for current target
static double g_pA[KCONT], g_pB;
static int g_epochs;

static void rand_perm(long* perm, long n, uint64_t seed){ for(long i=0;i<n;i++) perm[i]=i;
    uint64_t s=seed?seed:0xBADC0DEULL; for(long i=n-1;i>0;i--){ s^=s<<13;s^=s>>7;s^=s<<17; long j=(long)((s>>11)%(uint64_t)(i+1)); long t=perm[i];perm[i]=perm[j];perm[j]=t; } }

enum { V_FULL=0, V_POINT, V_TOKONLY, V_SCRAMBLE, NVARMLP };
static const char* VARN[NVARMLP]={"FULL(traj)","POINT(state)","TOKONLY(ablate)","SCRAMBLE(perm-lbl)"};
static double g_infoA[NVARMLP+1], g_infoB[NVARMLP]; // +1 slot for BLIND infoA

// SCRAMBLE = causal FULL trajectory but the trajectory<->label correspondence is broken by permuting the
// labels across rows (leak-free: no future state ever enters a row's pooled memory). Must collapse to ~0.
static void perm_labels(const uint8_t* yA,const uint8_t* mA,const uint8_t* yB,const uint8_t* mB,long used,uint64_t seed,
                        uint8_t* yA2,uint8_t* mA2,uint8_t* yB2,uint8_t* mB2){
    long* pm=malloc((size_t)used*sizeof(long)); rand_perm(pm,used,seed);
    for(long r=0;r<used;r++){ long s=pm[r]; memcpy(&yA2[(size_t)r*KCONT],&yA[(size_t)s*KCONT],KCONT);
        mA2[r]=mA[s]; yB2[r]=yB[s]; mB2[r]=mB[s]; }
    free(pm);
}
static void build_rep(int vtype,const float* X,const uint32_t* prev,long nrow,long used,int in,float* U){
    if(vtype==V_FULL||vtype==V_SCRAMBLE){ float* T=malloc((size_t)nrow*TRAJ_DIM*4); build_traj(X,nrow,NULL,T);
        for(long r=0;r<used;r++) memcpy(&U[(size_t)r*in],&T[(size_t)r*TRAJ_DIM],in*4); free(T); }
    else if(vtype==V_POINT){ for(long r=0;r<used;r++) memcpy(&U[(size_t)r*in],&X[(size_t)r*S_DIM],in*4); }
    else { float* Tk=malloc((size_t)nrow*DE_TOK*4); build_tokonly(prev,nrow,Tk);
        for(long r=0;r<used;r++) memcpy(&U[(size_t)r*in],&Tk[(size_t)r*DE_TOK],in*4); free(Tk); }
}
static void run_mlp_variant(int vtype){
    int in = (vtype==V_POINT)?S_DIM : (vtype==V_TOKONLY)?DE_TOK : TRAJ_DIM;
    float* Utr=malloc((size_t)g_trrows*in*4);
    build_rep(vtype,g_Xtr,g_ptr,g_trrows,g_trrows,in,Utr);

    // for SCRAMBLE, break the input<->label correspondence (must collapse)
    uint8_t *yAt=g_yAtr,*mAt=g_mAtr,*yBt=g_yBtr,*mBt=g_mBtr;
    if(vtype==V_SCRAMBLE){ yAt=malloc((size_t)g_trrows*KCONT); mAt=malloc(g_trrows); yBt=malloc(g_trrows); mBt=malloc(g_trrows);
        perm_labels(g_yAtr,g_mAtr,g_yBtr,g_mBtr,g_trrows,0x5C8A11ULL,yAt,mAt,yBt,mBt); }

    Probe p; probe_init(&p,in,0x53ULL^((uint64_t)vtype*0x9E37ULL));
    probe_train(&p,Utr,yAt,mAt,yBt,mBt,g_trrows,g_epochs,1e-3f);
    free(Utr); if(vtype==V_SCRAMBLE){ free(yAt);free(mAt);free(yBt);free(mBt); }

    double baseA=0,modA=0,baseB=0,modB=0; long nB=0,nA=0;
    for(int w=0;w<N_VAL;w++){ int inv=in; float* Uv=malloc((size_t)g_vrows[w]*inv*4);
        build_rep(vtype,g_Xv[w],g_pv[w],g_vrows[w],g_vrows[w],inv,Uv);
        uint8_t *yAe=g_yAv[w],*mAe=g_mAv[w],*yBe=g_yBv[w],*mBe=g_mBv[w];
        if(vtype==V_SCRAMBLE){ yAe=malloc((size_t)g_vrows[w]*KCONT); mAe=malloc(g_vrows[w]); yBe=malloc(g_vrows[w]); mBe=malloc(g_vrows[w]);
            perm_labels(g_yAv[w],g_mAv[w],g_yBv[w],g_mBv[w],g_vrows[w],0x5C8A11ULL^(uint64_t)(w+1),yAe,mAe,yBe,mBe); }
        probe_eval(&p,Uv,yAe,mAe,yBe,mBe,g_vrows[w],g_pA,g_pB,&baseA,&modA,&nA,&baseB,&modB,&nB);
        free(Uv); if(vtype==V_SCRAMBLE){ free(yAe);free(mAe);free(yBe);free(mBe); }
    }
    probe_free(&p);
    double infoA=(nA>0)?(baseA-modA)/(double)((double)nA*KCONT):0;
    double infoB=(nB>0)?(baseB-modB)/(double)nB:0;
    g_infoA[vtype]=infoA; g_infoB[vtype]=infoB;
    printf("  %-18s | HEAD-A info %+8.5f bits/lbl | HEAD-B info %+8.5f bits  (nA=%ld baseA=%.4f modA=%.4f)\n",
           VARN[vtype],infoA,infoB,nA, nA>0?baseA/((double)nA*KCONT):0, nA>0?modA/((double)nA*KCONT):0);
}

static double g_tgtFULL,g_tgtTOK,g_tgtSCR,g_tgtPOINT,g_tgtBLIND;
// ===================== B-SFDPE plastic overlay (the substrate stays READ-ONLY) =====================
#define N_OV   S_DIM        // overlay dim = 512
#define R_OV   12           // V/U/trace rank
#define K_AN   32           // anchor (Oja) dim
#define GAMMA_OV 0.5f
static float *g_W,*g_Win,*g_Vov;       // frozen: W (N*N), W_in (N*512), V (N*r)
static float g_P[K_AN][S_DIM];         // frozen Oja anchor (top-k principal axes of TRAIN x_SEE)
static float g_Prand[K_AN][S_DIM];     // A1: random isotropic anchor
static uint8_t g_isfunc[1<<20];        // top-FUNC_SKIP frequency = function word (c_t=0.3)
static uint8_t *g_puTr,*g_nlTr,*g_puV[N_VAL],*g_nlV[N_VAL];  // boundary flags per row

static void gen_gauss(float* m,size_t n,uint64_t seed,float sc){ uint64_t s=seed?seed:0x9E37ULL;
    for(size_t i=0;i<n;i++){ double u1=xs_u01(&s); if(u1<1e-12)u1=1e-12; double u2=xs_u01(&s);
        m[i]=(float)(sqrt(-2.0*log(u1))*cos(6.283185307179586*u2))*sc; } }
static void overlay_matrices_init(void){
    g_W  =malloc((size_t)N_OV*N_OV*4); gen_gauss(g_W,(size_t)N_OV*N_OV,0x5BFD0001ULL, 0.85f/sqrtf((float)N_OV)); // spectral radius ~0.85<0.9
    g_Win=malloc((size_t)N_OV*S_DIM*4);gen_gauss(g_Win,(size_t)N_OV*S_DIM,0x5BFD0002ULL,1.0f/sqrtf((float)S_DIM));
    g_Vov=malloc((size_t)N_OV*R_OV*4); gen_gauss(g_Vov,(size_t)N_OV*R_OV,0x5BFD0003ULL,1.0f/sqrtf((float)N_OV));
    // A1 random isotropic anchor (normalized rows)
    for(int c=0;c<K_AN;c++){ gen_gauss(g_Prand[c],S_DIM,0x5BFD0100ULL+c,1.0f);
        float nn=0; for(int d=0;d<S_DIM;d++) nn+=g_Prand[c][d]*g_Prand[c][d]; float inv=1.0f/(sqrtf(nn)+1e-12f);
        for(int d=0;d<S_DIM;d++) g_Prand[c][d]*=inv; }
}
// Oja/Sanger top-K subspace of TRAIN x_SEE (no val leak), one streaming pass, then row-normalize.
static void oja_anchor_fit(void){
    for(int c=0;c<K_AN;c++){ gen_gauss(g_P[c],S_DIM,0x5BFD0200ULL+c,1.0f);
        float nn=0; for(int d=0;d<S_DIM;d++) nn+=g_P[c][d]*g_P[c][d]; float inv=1.0f/(sqrtf(nn)+1e-12f);
        for(int d=0;d<S_DIM;d++) g_P[c][d]*=inv; }
    float eta=2e-3f; float y[K_AN], resid[S_DIM];
    for(int pass=0;pass<2;pass++) for(long r=0;r<g_trrows;r++){ const float* x=&g_Xtr[(size_t)r*S_DIM];
        for(int c=0;c<K_AN;c++) y[c]=dot_avx(g_P[c],x,S_DIM);
        // Sanger: dP_c = eta*y_c*(x - sum_{j<=c} y_j P_j)
        memcpy(resid,x,S_DIM*4);
        for(int c=0;c<K_AN;c++){ for(int d=0;d<S_DIM;d++) resid[d]-=y[c]*g_P[c][d];
            float yc=y[c]; for(int d=0;d<S_DIM;d++) g_P[c][d]+=eta*yc*resid[d]; }
    }
    for(int c=0;c<K_AN;c++){ float nn=0; for(int d=0;d<S_DIM;d++) nn+=g_P[c][d]*g_P[c][d]; float inv=1.0f/(sqrtf(nn)+1e-12f);
        for(int d=0;d<S_DIM;d++) g_P[c][d]*=inv; }
    fprintf(stderr,"Oja anchor fit done (K=%d over %ld train rows x2)\n",K_AN,g_trrows);
}
static void build_funcmask(void){
    int* idx=malloc((size_t)VTOK*sizeof(int)); for(int v=0;v<VTOK;v++) idx[v]=v;
    qsort(idx,VTOK,sizeof(int),cmp_freq);
    for(int v=0;v<VTOK;v++) g_isfunc[v]=0;
    for(int r=0;r<FUNC_SKIP && r<VTOK;r++) g_isfunc[idx[r]]=1;
    free(idx);
}
static void boundary_flags(const uint32_t* prev,long nrow,uint8_t* pu,uint8_t* nl){
    for(long r=0;r<nrow;r++){ int L=bpe_tok_len(&g_bpe,prev[r]); const unsigned char* b=bpe_tok_bytes(&g_bpe,prev[r]);
        int ip=0,inl=0; for(int k=0;k<L;k++){ unsigned char c=b[k]; if(c=='.'||c=='!'||c=='?') ip=1; if(c=='\n') inl=1; }
        pu[r]=(uint8_t)ip; nl[r]=(uint8_t)inl; }
}

// overlay config: variant + ablation switches
enum { OV_CONTENT=0, OV_POS, OV_PARITY };
typedef struct { const char* name; int variant; int anchor_random; int scalar_m; int boundary_off;
    int w_mode; /*0 both,1 c_only,2 wsub_only*/ int single_long; int scramble; } OCfg;

// roll the plastic overlay ONLINE over one window (U starts 0; local plasticity; substrate untouched).
// boundaries/content read from REAL positions; for scramble, x_SEE is consumed via order[] (shuffled stream).
static double overlay_roll(const OCfg* cfg,const float* Xs,const uint32_t* prev,const uint8_t* pu,const uint8_t* nl,
                           long nrow,const long* order,float* h_out){
    const float (*P)[S_DIM] = cfg->anchor_random? g_Prand : g_P;
    float* U=calloc((size_t)N_OV*R_OV,4);
    float* Qc=calloc((size_t)K_AN*K_AN,4); float* Qp=calloc((size_t)K_AN*K_AN,4);
    float h_prev[N_OV]; memset(h_prev,0,sizeof h_prev);
    float h[N_OV], xv[K_AN], xv_prev[K_AN]; memset(xv_prev,0,sizeof xv_prev);
    float xv_pPUNCT[K_AN], xv_pNEWL[K_AN]; memset(xv_pPUNCT,0,sizeof xv_pPUNCT); memset(xv_pNEWL,0,sizeof xv_pNEWL);
    float sc_[K_AN], sp_[K_AN]; memset(sc_,0,sizeof sc_); memset(sp_,0,sizeof sp_);
    float* es=calloc((size_t)N_OV*R_OV,4); float* el=calloc((size_t)N_OV*R_OV,4);
    float* incbuf=calloc((size_t)N_OV*R_OV,4);
    float p[R_OV], q[N_OV], net[N_OV];
    float ema_dxv=1e-3f, ema_dnorm=1e-3f;
    const float dec_s=expf(-1.0f/30.0f), dec_l=expf(-1.0f/100.0f);
    const float etaQ=1e-3f, eta=2e-3f, Gmax=3.0f;
    double dUF_tot=0; long dU_cnt=0;
    uint64_t prng=0x53B0FD01ULL ^ (uint64_t)cfg->variant*0x9E3779B1ULL;
    for(long r=0;r<nrow;r++){
        const float* x=&Xs[(size_t)(order?order[r]:r)*S_DIM];
        // PRE
        for(int j=0;j<R_OV;j++){ float s=0; for(int i=0;i<N_OV;i++) s+=g_Vov[(size_t)i*R_OV+j]*h_prev[i]; p[j]=s; }
        for(int i=0;i<N_OV;i++){ float s=dot_avx(&g_W[(size_t)i*N_OV],h_prev,N_OV)+dot_avx(&g_Win[(size_t)i*S_DIM],x,S_DIM);
            const float* Ui=&U[(size_t)i*R_OV]; for(int j=0;j<R_OV;j++) s+=Ui[j]*p[j]; net[i]=s; }
        for(int i=0;i<N_OV;i++){ float t=tanhf(net[i]); h[i]=(1.0f-GAMMA_OV)*h_prev[i]+GAMMA_OV*t; }
        // POST
        for(int c=0;c<K_AN;c++) xv[c]=dot_avx(P[c],x,S_DIM);
        for(int i=0;i<N_OV;i++) q[i]=1.0f-h[i]*h[i];
        float ndxv=0; for(int c=0;c<K_AN;c++){ float d=xv[c]-xv_prev[c]; ndxv+=d*d; } ndxv=sqrtf(ndxv);
        ema_dxv=0.99f*ema_dxv+0.01f*ndxv;
        float w_sub=ndxv/(ema_dxv+1e-9f); if(w_sub<0.5f)w_sub=0.5f; if(w_sub>2.0f)w_sub=2.0f;
        float c_t=g_isfunc[prev[r]]?0.3f:1.0f;
        if(cfg->w_mode==1) w_sub=1.0f;         // A4b: c_t only
        if(cfg->w_mode==2) c_t=1.0f;           // A4c: w_sub only
        float wc=c_t*w_sub;
        // traces (2 scale)
        for(int i=0;i<N_OV;i++){ float qi=q[i]; float* esi=&es[(size_t)i*R_OV]; float* eli=&el[(size_t)i*R_OV];
            for(int j=0;j<R_OV;j++){ float op=wc*qi*p[j]; esi[j]=dec_s*esi[j]+op; eli[j]=dec_l*eli[j]+op; } }
        // boundary-gated slow target -> predictor residual delta (in K_AN space)
        int trig_p = cfg->boundary_off? 1 : pu[r];
        int trig_n = cfg->boundary_off? 0 : nl[r];
        float delta[K_AN]; int have_delta=0; int is_newl=0;
        if(trig_p){ float xi[K_AN];
            if(cfg->variant==OV_POS){ for(int c=0;c<K_AN;c++){ float fr=powf(10000.0f,-2.0f*((c/2))/(float)K_AN);
                    xi[c]=(c&1)?cosf((float)r*fr):sinf((float)r*fr); } float nx=0; for(int c=0;c<K_AN;c++) nx+=xi[c]*xi[c]; nx=sqrtf(nx)+1e-9f;
                for(int c=0;c<K_AN;c++) xi[c]*=(ema_dxv/nx); }
            else for(int c=0;c<K_AN;c++) xi[c]=xv[c]-xv_pPUNCT[c];
            float dl[K_AN]; for(int c=0;c<K_AN;c++){ float pr=0; for(int e=0;e<K_AN;e++) pr+=Qc[(size_t)c*K_AN+e]*sc_[e]; dl[c]=xi[c]-pr; }
            for(int c=0;c<K_AN;c++) for(int e=0;e<K_AN;e++) Qc[(size_t)c*K_AN+e]+=etaQ*dl[c]*sc_[e];
            for(int c=0;c<K_AN;c++) sc_[c]=0.6667f*sc_[c]+0.3333f*xi[c];
            for(int c=0;c<K_AN;c++) xv_pPUNCT[c]=xv[c];
            memcpy(delta,dl,sizeof dl); have_delta=1; is_newl=0;
        }
        if(trig_n){ float xi[K_AN];
            if(cfg->variant==OV_POS){ for(int c=0;c<K_AN;c++){ float fr=powf(10000.0f,-2.0f*((c/2))/(float)K_AN);
                    xi[c]=(c&1)?cosf((float)r*fr*1.3f):sinf((float)r*fr*1.3f); } float nx=0; for(int c=0;c<K_AN;c++) nx+=xi[c]*xi[c]; nx=sqrtf(nx)+1e-9f;
                for(int c=0;c<K_AN;c++) xi[c]*=(ema_dxv/nx); }
            else for(int c=0;c<K_AN;c++) xi[c]=xv[c]-xv_pNEWL[c];
            float dl[K_AN]; for(int c=0;c<K_AN;c++){ float pr=0; for(int e=0;e<K_AN;e++) pr+=Qp[(size_t)c*K_AN+e]*sp_[e]; dl[c]=xi[c]-pr; }
            for(int c=0;c<K_AN;c++) for(int e=0;e<K_AN;e++) Qp[(size_t)c*K_AN+e]+=etaQ*dl[c]*sp_[e];
            for(int c=0;c<K_AN;c++) sp_[c]=0.6667f*sp_[c]+0.3333f*xi[c];
            for(int c=0;c<K_AN;c++) xv_pNEWL[c]=xv[c];
            if(!have_delta){ memcpy(delta,dl,sizeof dl); have_delta=1; } is_newl=1; // delta=delta_c if PUNCT else delta_p
        }
        // GATE + 3-factor VECTOR update (anchored to span(P) -> m = P^T delta). PARITY = same per-token
        // ||dU||_F but RANDOM direction (randomize at the dU-tensor level so the magnitude matches CONTENT exactly).
        if(have_delta){
            float dn=0; for(int c=0;c<K_AN;c++) dn+=delta[c]*delta[c]; dn=sqrtf(dn);
            ema_dnorm=0.95f*ema_dnorm+0.05f*dn;
            float gnov=dn/(ema_dnorm+1e-9f); if(gnov>Gmax)gnov=Gmax; if(gnov<0)gnov=0;
            float mq=0; for(int i=0;i<N_OV;i++) mq+=q[i]; mq/=N_OV;
            float g=gnov*mq;
            float m[N_OV]; for(int i=0;i<N_OV;i++){ float s=0; for(int c=0;c<K_AN;c++) s+=P[c][i]*delta[c]; m[i]=s; }
            if(cfg->scalar_m){ float mm=0; for(int i=0;i<N_OV;i++) mm+=m[i]; mm/=N_OV; for(int i=0;i<N_OV;i++) m[i]=mm; } // A2
            float sc=eta*g; double inc=0;
            for(int i=0;i<N_OV;i++){ float mi=m[i]*sc; float* INC=&incbuf[(size_t)i*R_OV];
                const float* tr_s = cfg->single_long? &el[(size_t)i*R_OV] : &es[(size_t)i*R_OV];
                for(int j=0;j<R_OV;j++){ float d=mi*tr_s[j]; INC[j]=d; inc+=(double)d*d; }
                if(is_newl){ const float* eli=&el[(size_t)i*R_OV]; for(int j=0;j<R_OV;j++){ float d=mi*eli[j]; INC[j]+=d; inc+=(double)d*d; } }
            }
            if(cfg->variant==OV_PARITY){ float fn=(float)sqrt(inc); double rn=0;
                for(size_t z=0;z<(size_t)N_OV*R_OV;z++){ double u1=xs_u01(&prng); if(u1<1e-12)u1=1e-12; double u2=xs_u01(&prng);
                    float gg=(float)(sqrt(-2.0*log(u1))*cos(6.283185307179586*u2)); incbuf[z]=gg; rn+=(double)gg*gg; }
                float s=fn/(float)(sqrt(rn)+1e-12); for(size_t z=0;z<(size_t)N_OV*R_OV;z++) incbuf[z]*=s; }
            for(size_t z=0;z<(size_t)N_OV*R_OV;z++) U[z]+=incbuf[z];
            dUF_tot+=sqrt(inc); dU_cnt++;   // inc = CONTENT-magnitude; parity matches it by construction
        }
        // STABILIZER every 50 tokens: weight-decay + Frobenius/spectral cap on U (h itself is tanh-bounded)
        if((r%50)==49){ float fn=0; for(size_t z=0;z<(size_t)N_OV*R_OV;z++){ U[z]*=0.999f; fn+=U[z]*U[z]; } fn=sqrtf(fn);
            float cap=0.3f*0.85f*sqrtf((float)N_OV); if(fn>cap){ float s=cap/fn; for(size_t z=0;z<(size_t)N_OV*R_OV;z++) U[z]*=s; } }
        memcpy(h_out+(size_t)r*N_OV,h,N_OV*4);
        memcpy(h_prev,h,sizeof h); memcpy(xv_prev,xv,sizeof xv);
    }
    free(U);free(Qc);free(Qp);free(es);free(el);free(incbuf);
    return dU_cnt? dUF_tot/(double)dU_cnt : 0.0;
}

// reusable rep buffers (in <= 1024)
static float *g_Ureptr,*g_Urepv[N_VAL]; static float *g_htr,*g_hv[N_VAL];
static void asm_xh(const float* X,const float* h,long nrow,float* U){ // [x_SEE | h] = 1024
    for(long r=0;r<nrow;r++){ memcpy(&U[(size_t)r*1024],&X[(size_t)r*S_DIM],S_DIM*4); memcpy(&U[(size_t)r*1024+S_DIM],&h[(size_t)r*N_OV],N_OV*4); } }

// train probe on a prebuilt rep + eval on val reps; returns infoA,infoB. fixed probe seed across configs.
static void probe_on(const char* name,const float* Utr,float* const Uv[N_VAL],int in,double* o_iA,double* o_iB){
    Probe p; probe_init(&p,in,0xC0FFEEULL);
    probe_train(&p,Utr,g_yAtr,g_mAtr,g_yBtr,g_mBtr,g_trrows,g_epochs,1e-3f);
    double baseA=0,modA=0,baseB=0,modB=0; long nA=0,nB=0;
    for(int w=0;w<N_VAL;w++) probe_eval(&p,Uv[w],g_yAv[w],g_mAv[w],g_yBv[w],g_mBv[w],g_vrows[w],g_pA,g_pB,&baseA,&modA,&nA,&baseB,&modB,&nB);
    probe_free(&p);
    double iA=(nA>0)?(baseA-modA)/((double)nA*KCONT):0; double iB=(nB>0)?(baseB-modB)/(double)nB:0;
    *o_iA=iA; *o_iB=iB;
    printf("  %-26s | HEAD-A %+8.5f | HEAD-B %+8.5f  (in=%d)\n",name,iA,iB,in);
}
// roll an overlay config over all windows, build [x|h] reps, run probe. dUF reported.
static double g_infoXH[16],g_infoH[16]; static int g_nXH=0;
static void run_overlay(const char* name,const OCfg* cfg,int hint_kernel,float* kern_save){
    long* ord=NULL;
    if(cfg->scramble){ ord=malloc((size_t)g_trrows*sizeof(long)); rand_perm(ord,g_trrows,0x5CA3B1EULL); }
    double dUF=overlay_roll(cfg,g_Xtr,g_ptr,g_puTr,g_nlTr,g_trrows,ord,g_htr);
    if(ord) free(ord);
    asm_xh(g_Xtr,g_htr,g_trrows,g_Ureptr);
    for(int w=0;w<N_VAL;w++){ long* ov=NULL;
        if(cfg->scramble){ ov=malloc((size_t)g_vrows[w]*sizeof(long)); rand_perm(ov,g_vrows[w],0x5CA3B1EULL^(uint64_t)(w+1)); }
        overlay_roll(cfg,g_Xv[w],g_pv[w],g_puV[w],g_nlV[w],g_vrows[w],ov,g_hv[w]);
        if(ov) free(ov);
        asm_xh(g_Xv[w],g_hv[w],g_vrows[w],g_Urepv[w]);
    }
    double iA,iB; printf("[dU_F/tok=%.4e] ",dUF);
    probe_on(name,g_Ureptr,g_Urepv,1024,&iA,&iB);
    // h-only attribution (read = overlay h ALONE, 512D): here A5/scramble must collapse to ~chance.
    { double hA,hB; float* hv[N_VAL]; for(int w=0;w<N_VAL;w++) hv[w]=g_hv[w];
      char hn[64]; snprintf(hn,sizeof hn,"  %s [h-only]",name); probe_on(hn,g_htr,hv,N_OV,&hA,&hB); g_infoH[g_nXH]=hA; }
    g_infoXH[g_nXH++]=iA;
    if(hint_kernel && kern_save){ // Gram Frobenius of read rep over a fixed val sample (val window 0)
        long M=512; if(M>g_vrows[0]) M=g_vrows[0]; double fro=0;
        for(long i=0;i<M;i++){ const float* a=&g_Urepv[0][(size_t)i*1024];
            for(long j=0;j<M;j++){ const float* b=&g_Urepv[0][(size_t)j*1024]; float d=dot_avx(a,b,1024); fro+=(double)d*d; } }
        *kern_save=(float)sqrt(fro);
    }
}

int main(int argc,char** argv){
    if(argc<5){ fprintf(stderr,"Usage: %s <data> <D1_w> <bpe_merges> <outprefix> [--len N] [--epochs E] [--smoke]\n",argv[0]); return 1; }
    setvbuf(stderr,NULL,_IONBF,0); setvbuf(stdout,NULL,_IONBF,0);
    long N=150000, maxb=0; g_epochs=6;
    for(int i=5;i<argc;i++){
        if(!strcmp(argv[i],"--len")&&i+1<argc) N=atol(argv[++i]);
        else if(!strcmp(argv[i],"--epochs")&&i+1<argc) g_epochs=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--max-bytes")&&i+1<argc) maxb=atol(argv[++i]);
        else if(!strcmp(argv[i],"--smoke")){ N=24000; g_epochs=3; }
    }
    if(S_DIM!=512){ fprintf(stderr,"S_DIM=%d != 512\n",S_DIM); return 1; }
    FILE* fd=fopen(argv[1],"rb"); if(!fd){fprintf(stderr,"data\n");return 1;} fseek(fd,0,SEEK_END); g_fsz=ftell(fd); fseek(fd,0,SEEK_SET);
    if(maxb>0 && maxb<g_fsz) g_fsz=maxb;
    g_data=malloc(g_fsz); fread(g_data,1,g_fsz,fd); fclose(fd);

    SiliconEntropyState see;
    if(!load_d1(argv[2],&see)) return 1;
    if(!bpe_load_file(&g_bpe,argv[3])) return 1; VTOK=g_bpe.vocab;
    if(VTOK> (1<<20)){ fprintf(stderr,"VTOK too big\n"); return 1; }
    gen_lift(PROJ_SEED);
    ent_table=malloc(CLASSES*CLASSES*4);
    for(int i=0;i<CLASSES;i++) for(int j=0;j<CLASSES;j++){ float m=-1e9f; for(int k=0;k<CLASSES;k++) if(trigram[i][j][k]>m)m=trigram[i][j][k];
        double se=0; for(int k=0;k<CLASSES;k++) se+=exp((double)(trigram[i][j][k]-m));
        double Hh=0; for(int k=0;k<CLASSES;k++){ double pp=exp((double)(trigram[i][j][k]-m))/se; if(pp>1e-12)Hh-=pp*log(pp); }
        ent_table[i][j]=(float)Hh; }

    // substrate read-only checksum (no grad ever flows into it)
    double sck=0; for(int i=0;i<D1_TOT;i++) sck+=md1[i]+sd1[i];
    fprintf(stderr,"substrate checksum (md1+sd1 sum) = %.6f [READ-ONLY]\n",sck);

    tokenize_corpus();
    long tok_train_end=(long)(g_ntok*0.90);
    build_unigram(tok_train_end);
    build_content_labels();
    hippo_init(0x4190C0DEULL);
    tokemb_init(0xE2C0DEULL);

    long tr_start=g_fsz/5;
    long va_start[N_VAL]={ g_fsz/2, (long)(0.78*g_fsz) };
    for(int w=0;w<N_VAL;w++) if(va_start[w]+N+3>g_fsz){ fprintf(stderr,"val window %d out of file\n",w+1); return 1; }

    build_breaks();

    g_maxrows=N+16;
    g_Xtr=malloc((size_t)g_maxrows*S_DIM*4); g_ttr=malloc((size_t)g_maxrows*4); g_ptr=malloc((size_t)g_maxrows*4); g_rtTr=malloc((size_t)g_maxrows*sizeof(long));
    g_yAtr=malloc((size_t)g_maxrows*KCONT); g_mAtr=malloc(g_maxrows); g_yBtr=malloc(g_maxrows); g_mBtr=malloc(g_maxrows); g_brkTr=malloc((size_t)g_maxrows*sizeof(long));
    for(int w=0;w<N_VAL;w++){ g_Xv[w]=malloc((size_t)g_maxrows*S_DIM*4); g_tv[w]=malloc((size_t)g_maxrows*4); g_pv[w]=malloc((size_t)g_maxrows*4); g_rtV[w]=malloc((size_t)g_maxrows*sizeof(long));
        g_yAv[w]=malloc((size_t)g_maxrows*KCONT); g_mAv[w]=malloc(g_maxrows); g_yBv[w]=malloc(g_maxrows); g_mBv[w]=malloc(g_maxrows); g_brkV[w]=malloc((size_t)g_maxrows*sizeof(long)); }
    fprintf(stderr,"extract train...\n"); g_trrows=extract(&see,tr_start,N,g_Xtr,g_ttr,g_ptr,g_rtTr);
    for(int w=0;w<N_VAL;w++){ fprintf(stderr,"extract val%d...\n",w+1); g_vrows[w]=extract(&see,va_start[w],N,g_Xv[w],g_tv[w],g_pv[w],g_rtV[w]); }
    // per-row paragraph-break counts
    for(long r=0;r<g_trrows;r++) g_brkTr[r]=brk_at(g_tokstart[g_rtTr[r]]);
    for(int w=0;w<N_VAL;w++) for(long r=0;r<g_vrows[w];r++) g_brkV[w][r]=brk_at(g_tokstart[g_rtV[w][r]]);

    // ---- overlay setup (substrate read-only; overlay assets frozen except plastic U/Q per roll) ----
    build_funcmask();
    g_puTr=malloc(g_maxrows); g_nlTr=malloc(g_maxrows); boundary_flags(g_ptr,g_trrows,g_puTr,g_nlTr);
    for(int w=0;w<N_VAL;w++){ g_puV[w]=malloc(g_maxrows); g_nlV[w]=malloc(g_maxrows); boundary_flags(g_pv[w],g_vrows[w],g_puV[w],g_nlV[w]); }
    overlay_matrices_init();
    oja_anchor_fit();   // TRAIN only, frozen
    g_Ureptr=malloc((size_t)g_maxrows*1024*4); g_htr=malloc((size_t)g_maxrows*N_OV*4);
    for(int w=0;w<N_VAL;w++){ g_Urepv[w]=malloc((size_t)g_maxrows*1024*4); g_hv[w]=malloc((size_t)g_maxrows*N_OV*4); }

    // target = FIXED MID[+10,+40] within-story (within-para). FIXED length => NO position-in-paragraph length
    // confound, so the real-position boundary-timing channel can't decode it -> A5/scramble gate stays meaningful.
    // (REMAINDER's window length = distance-to-para-end = position, which the boundary channel decodes even when
    //  the substrate is scrambled; that contaminates GATE(2). fixed window removes that.)
    Tgt T = { "MID[+10,+40] within-story (fixed len, no position confound)", 10, M_FIXED, 40, 1 };
    g_nactTr=build_labels_tgt(&T,g_ptr,g_brkTr,g_trrows,g_yAtr,g_mAtr,g_yBtr,g_mBtr);
    for(int w=0;w<N_VAL;w++) g_nactV[w]=build_labels_tgt(&T,g_pv[w],g_brkV[w],g_vrows[w],g_yAv[w],g_mAv[w],g_yBv[w],g_mBv[w]);
    for(int k=0;k<KCONT;k++) g_pA[k]=0; long pBn=0; double pBs=0,nA=0;
    for(long r=0;r<g_trrows;r++){ if(g_mAtr[r]){ const uint8_t* ya=&g_yAtr[(size_t)r*KCONT]; for(int k=0;k<KCONT;k++) g_pA[k]+=ya[k]; nA++; } if(g_mBtr[r]){ pBs+=g_yBtr[r]; pBn++; } }
    for(int k=0;k<KCONT;k++) g_pA[k]/=(nA>0?nA:1); g_pB=(pBn>0)?pBs/pBn:0.0;

    printf("\n==== Phase 53.B B-SFDPE: can LOCAL plasticity (no backprop/BPTT, substrate read-only) surface a\n");
    printf("     slow SEMANTIC variable the static read missed? TF decodability probe. target=%s\n",T.name);
    printf("N=%ld epochs=%d K=%d HID=%d  N_OV=%d R=%d K_AN=%d  actTr=%ld actV=%ld/%ld  read=[x_SEE|h]=1024D\n",
           N,g_epochs,KCONT,HID,N_OV,R_OV,K_AN,g_nactTr,g_nactV[0],g_nactV[1]);
    printf("info=baseBCE(marginal)-modelBCE bits/lbl, 2 held-out val windows. raw numbers only, no auto-verdict.\n");

    OCfg content = {"CONTENT",      OV_CONTENT,0,0,0,0,0,0};
    OCfg pos     = {"POS",          OV_POS,    0,0,0,0,0,0};
    OCfg parity  = {"PARITY",       OV_PARITY, 0,0,0,0,0,0};
    OCfg a1      = {"A1-anchorRand", OV_CONTENT,1,0,0,0,0,0};
    OCfg a2      = {"A2-scalar",     OV_CONTENT,0,1,0,0,0,0};
    OCfg a3      = {"A3-boundaryOff",OV_CONTENT,0,0,1,0,0,0};
    OCfg a4b     = {"A4b-cOnly",     OV_CONTENT,0,0,0,1,0,0};
    OCfg a4c     = {"A4c-wsubOnly",  OV_CONTENT,0,0,0,2,0,0};
    OCfg a5      = {"A5-scramble",   OV_CONTENT,0,0,0,0,0,1};
    OCfg a6      = {"A6-singleLong", OV_CONTENT,0,0,0,0,1,0};

    float kC=0,kF=0,kP=0;

    printf("\n-- GATE (1): A2-scalar MUST be flat (~ FROZEN-POINT). if not -> harness/probe broken -> STOP --\n");
    double iA_a2,iB; run_overlay("A2-scalar",&a2,0,NULL);
    iA_a2=g_infoXH[g_nXH-1]; (void)iB;

    // FROZEN-POINT baseline (read = x_SEE only, 512D) + BLIND
    double iF_A,iF_B; { float* uv[N_VAL]; for(int w=0;w<N_VAL;w++) uv[w]=g_Xv[w];
        printf("\n-- baselines --\n"); probe_on("FROZEN-POINT(x_SEE 512)",g_Xtr,uv,S_DIM,&iF_A,&iF_B); }
    double iBlind; { float* seenTr=malloc((size_t)g_trrows*KCONT*4); float* lcntTr=malloc((size_t)g_trrows*KCONT*4);
      blind_features(g_ptr,g_trrows,seenTr,lcntTr); float* a=malloc(KCONT*4),*b=malloc(KCONT*4),*c=malloc(KCONT*4);
      blind_run(seenTr,lcntTr,g_yAtr,g_mAtr,g_trrows,g_pA,a,b,c,5,1.5e-2f); free(seenTr);free(lcntTr);
      double base=0,mod=0; long nA2=0; for(int w=0;w<N_VAL;w++){ float* se=malloc((size_t)g_vrows[w]*KCONT*4); float* lc=malloc((size_t)g_vrows[w]*KCONT*4);
          blind_features(g_pv[w],g_vrows[w],se,lc); blind_eval(se,lc,g_yAv[w],g_mAv[w],g_vrows[w],g_pA,a,b,c,&base,&mod,&nA2); free(se);free(lc); }
      iBlind=(nA2>0)?(base-mod)/((double)nA2*KCONT):0; free(a);free(b);free(c);
      printf("  %-26s | HEAD-A %+8.5f | (n-gram-able recurrence floor)\n","BLIND(recur)",iBlind); }
    printf("  GATE(1) check: A2-scalar=%+.5f vs FROZEN-POINT=%+.5f  (must be ~equal/flat)\n",iA_a2,iF_A);

    printf("\n-- GATE (2): A5-scramble MUST collapse to ~chance (substrate load-bearing) --\n");
    double iA5; run_overlay("A5-scramble",&a5,0,NULL); iA5=g_infoXH[g_nXH-1];
    printf("  GATE(2) check: A5-scramble=%+.5f  (must be << CONTENT and ~FROZEN/chance)\n",iA5);

    printf("\n-- MAIN VERDICT VARIANTS (CONTENT vs POS/PARITY; baselines FROZEN/BLIND above) --\n");
    double iC; run_overlay("CONTENT",&content,1,&kC); iC=g_infoXH[g_nXH-1];
    double iPOS; run_overlay("POS",&pos,0,NULL); iPOS=g_infoXH[g_nXH-1];
    double iPAR; run_overlay("PARITY",&parity,1,&kP); iPAR=g_infoXH[g_nXH-1];
    // frozen kernel
    { long M=512; if(M>g_vrows[0]) M=g_vrows[0]; double fro=0;
      for(long i=0;i<M;i++){ const float* a=&g_Xv[0][(size_t)i*S_DIM]; for(long j=0;j<M;j++){ const float* b=&g_Xv[0][(size_t)j*S_DIM]; float d=dot_avx(a,b,S_DIM); fro+=(double)d*d; } } kF=(float)sqrt(fro); }

    printf("\n-- ABLATIONS (mechanism) --\n");
    double iA1; run_overlay("A1-anchorRand",&a1,0,NULL); iA1=g_infoXH[g_nXH-1];
    double iA3; run_overlay("A3-boundaryOff",&a3,0,NULL); iA3=g_infoXH[g_nXH-1];
    double iA4b; run_overlay("A4b-cOnly",&a4b,0,NULL); iA4b=g_infoXH[g_nXH-1];
    double iA4c; run_overlay("A4c-wsubOnly",&a4c,0,NULL); iA4c=g_infoXH[g_nXH-1];
    double iA6; run_overlay("A6-singleLong",&a6,0,NULL); iA6=g_infoXH[g_nXH-1];

    printf("\n==== 53.B RAW SUMMARY (HEAD-A info bits/lbl) ====\n");
    printf("  [x_SEE|h] read (floor = FROZEN-POINT; overlay HEADROOM = config - FROZEN):\n");
    printf("    CONTENT=%+.5f  FROZEN-POINT=%+.5f  PARITY=%+.5f  BLIND=%+.5f  POS=%+.5f\n",iC,iF_A,iPAR,iBlind,iPOS);
    printf("    headroom: CONTENT-FROZEN=%+.5f  CONTENT-PARITY=%+.5f  CONTENT-BLIND=%+.5f  CONTENT-POS=%+.5f\n",
           iC-iF_A,iC-iPAR,iC-iBlind,iC-iPOS);
    printf("  h-only read (ATTRIBUTION; here substrate-load-bearing => A5/scramble must collapse to ~0):\n");
    printf("    CONTENT_h=%+.5f  PARITY_h=%+.5f  POS_h=%+.5f  A5scramble_h=%+.5f  A1anchorRand_h=%+.5f  A2scalar_h=%+.5f\n",
           g_infoH[2],g_infoH[4],g_infoH[3],g_infoH[1],g_infoH[5],g_infoH[0]);
    printf("  GATES: (1) A2-scalar[x|h]=%+.5f vs FROZEN=%+.5f (MUST be ~equal=flat).  (2) A5-scramble headroom=%+.5f & A5_h=%+.5f (MUST be ~0=collapsed).\n",
           iA_a2,iF_A,iA5-iF_A,g_infoH[1]);
    printf("  ABLATIONS [x|h]: A1-anchorRand=%+.5f(->parity?)  A3-boundaryOff=%+.5f(->POS?)  A4b-cOnly=%+.5f  A4c-wsubOnly=%+.5f  A6-singleLong=%+.5f\n",
           iA1,iA3,iA4b,iA4c,iA6);
    printf("  KERNEL-FROB (read-rep Gram, val0 512 sample): K_CONTENT=%.4e  K_FROZEN=%.4e  K_PARITY=%.4e  (K_CONTENT~=K_PARITY => still flat)\n",kC,kF,kP);
    printf("  (interpretation is MINE to read: success iff CONTENT headroom >> PARITY/POS/0 & CONTENT > BLIND, with GATE1 flat & GATE2 collapsed.)\n");

    if(argv[4]){ char sp[512]; snprintf(sp,sizeof sp,"%s_bsfdpe.bin",argv[4]);
        FILE* f=fopen(sp,"wb"); if(f){ uint32_t mg=0x5345455E; fwrite(&mg,4,1,f);
            double v[7]={iC,iF_A,iPAR,iBlind,iPOS,iA_a2,iA5}; fwrite(v,sizeof(double),7,f); fclose(f); } }
    return 0;
}
