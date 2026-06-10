// Phase 46.B - LOWMARGIN at phrase scale (p30/p40) + PUNCT+LOWMARGIN, over D1 (train-time)
//
// 46.A was a MIXED signal: A1 PUNCT compresses (2.2511 < D1), A3 LOWMARGIN partially
// stabilizes at T=0.55 (topBi 17->11, altLp 6->2, runWst 8->4) at ~no BPB cost - but A3's L3
// fired only ~0.01% in closed-loop. 46.B0 found why: NOT a margin-distribution shift (teacher-
// forced ~= closed-loop) and NOT a mis-sited threshold per byte (~21% of bytes ARE below p20),
// but the M=4 SUSTAIN: low-margin bytes are isolated, almost never 4-in-a-row. Re-thresholding
// (inference-only) showed p40 fires ~0.74% (gap ~135, the 46.0 phrase scale); p20 was too low.
//
// 46.B raises the LOWMARGIN bar to p30/p40 of the trigram margin (the dist is ~identical
// teacher-forced vs closed-loop, so the self-calibrated quantile is valid), and adds a
// combined PUNCT+LOWMARGIN schedule. feature = [SEE 192 | L2 64 | L3 64] = 320D, L2 = D1,
// L3 = slow EMA of raw fa refreshed at the schedule. Readout linear. Warm-start C2.A. Fresh
// extract per config. The L2+L3 blocks are bit-identical to phase46_generator.c (parity).
//
//   Bd1 D1 control : L3_NONE (dead block) -> should reproduce D1 ~2.2522
//   B1 LOWMARGIN p30 : refresh on trigram margin < p30 sustained M=4
//   B2 LOWMARGIN p40 : refresh on trigram margin < p40 sustained M=4 (~phrase scale)
//   B3 PUNCT+LM p30  : refresh on (. ! ?) OR (margin < p30 sustained M=4)
// LOWMARGIN uses the TRIGRAM prob-margin (context-only -> parity-safe; no 320D readout at
// extract time). m_thr = pXX of the trigram margin, self-calibrated, written to header.
//
// Audit: BPB + L2/SEE & L3/SEE logit ratios + L3 argmax flip (trainer); L3 rel-move / update
// freq + gap + word-gate T0.65/T0.55 (ps1). Promotion needs BOTH temps (BPB alone is not enough).
//
// Weight format 0x53454544 = D1 core (... + mix + l2_scale + l2_cap) + L3 block:
//   ... + l2_cap + {u32 l3_dim, u32 l3_mode, u32 l3_K, f l3_mthr, u32 l3_M, u32 l3_refr,
//   f l3_alpha} + trigram + feat_mean[320] + feat_std[320] + W[256][320] + B[256].
//
// Build (from repo root):
//   gcc -O3 -march=native -mavx2 -mfma \
//       benchmarks/phase38-42/phase46b_l3.c \
//       src/silicon_entropy.c src/silicon_v0.c \
//       -o bin/phase46b_l3.exe -lm -I .
// Run:
//   bin/phase46b_l3.exe <dataset> <weights_prefix> <c2a_weights>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "src/silicon_entropy.h"

#define CLASSES   256
#define BASE_DIM  SEE_FEATURE_DIM
#define L2_DIM    64
#define L3_DIM    64
#define TOT_DIM   (BASE_DIM + L2_DIM + L3_DIM)   // 192 SEE | 64 L2 | 64 L3 = 320
#define FEAT_CLAMP_DEFAULT 2.0f
#define C2A_BPB   2.2593
#define P_SEED    0xB5297A4Du
#define L3_ALPHA  0.9f    // L3 slow-EMA decay (updates are already sparse: phrase-scale)

enum { G_NONE=0, G_PUNCT=1, G_WS=2, G_SURPRISE=3, G_ENTROPY=4, G_COMBINED=5 };
// Phase 46.A/B L3 phrase-memory schedule: WHEN to refresh the slow L3 = EMA of fa (SEE).
enum { L3_NONE=0, L3_PUNCT=1, L3_L2CLUST=2, L3_LOWMARGIN=3, L3_PUNCTLM=4 };

static inline float dot_simd(const float* w, const float* f, int n) {
    __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(&w[i]),_mm256_loadu_ps(&f[i]),s);
    float o[8]; _mm256_storeu_ps(o,s);
    float r=o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; for(;i<n;i++) r+=w[i]*f[i]; return r;
}
static inline void grad_simd(float* gW, const float* f, float e, int n) {
    __m256 ev=_mm256_set1_ps(e); int i=0;
    for(;i<=n-8;i+=8) _mm256_storeu_ps(&gW[i],_mm256_fmadd_ps(ev,_mm256_loadu_ps(&f[i]),_mm256_loadu_ps(&gW[i])));
    for(;i<n;i++) gW[i]+=e*f[i];
}

typedef struct { float W[CLASSES][TOT_DIM]; float B[CLASSES];
                 float mW[CLASSES][TOT_DIM]; float vW[CLASSES][TOT_DIM];
                 float mB[CLASSES]; float vB[CLASSES]; int t; } AdamState;

static uint8_t *data; static long data_size;
static int train_start, train_len, val_start, val_len;
static float *features_train, *features_val;
static uint8_t *target_train, *target_val, *ctx_train, *ctx_val, *ctx2_train, *ctx2_val;
static float (*trigram_logits)[CLASSES][CLASSES];
static float (*ent_table)[CLASSES];
static float (*margin_table)[CLASSES];   // trigram prob-margin top1-top2 per (c2,c1) context
static float feat_mean[TOT_DIM], feat_std[TOT_DIM];
static float Pmat[L2_DIM][BASE_DIM];

void gen_projection(uint32_t seed) {
    uint64_t s = seed ? (uint64_t)seed : 0x9E3779B97F4A7C15ULL;
    for (int j=0;j<L2_DIM;j++) for (int k=0;k<BASE_DIM;k++){ s^=s<<13; s^=s>>7; s^=s<<17; Pmat[j][k]=(s&1ULL)?1.f:-1.f; }
}
static inline int is_punct(uint8_t b){ return b=='.'||b=='!'||b=='?'||b=='\n'||b=='"'||b=='\''; }
static inline int is_ws(uint8_t b){ return b==' '||b=='\n'||b=='\t'||b=='.'||b==','||b=='!'||b=='?'||b==';'||b==':'; }
static inline int eval_gate(int gate, uint8_t byte, uint8_t c1, uint8_t c2, float surp_thr, float ent_thr, int ent_high) {
    switch (gate) {
        case G_NONE:   return 0;
        case G_PUNCT:  return is_punct(byte);
        case G_WS:     return is_ws(byte);
        case G_SURPRISE: return (-trigram_logits[c2][c1][byte]) > surp_thr;
        case G_ENTROPY:  return ent_high ? (ent_table[c2][c1] > ent_thr) : (ent_table[c2][c1] < ent_thr);
        case G_COMBINED: return is_punct(byte) || ((-trigram_logits[c2][c1][byte]) > surp_thr);
    }
    return 0;
}

// Phase 46.A L3 schedule - BYTE-IDENTICAL to phase46_generator.c l3_schedule_fire
// (parity-critical). Decides WHEN the slow L3 phrase-memory refreshes, from the allowed
// internal signals only. byte = the just-observed/generated byte; l2write = whether the D1
// L2 EMA just fired; trimargin = trigram prob-margin (top1-top2) at this prediction context
// (parity-safe: pure context, identical in trainer extract and generator - no 320D readout
// needed at extract time). State (clust counter, low-margin run + refractory) via pointer,
// reset per extract call (trainer) / once pre-warmup (generator).
// LOWMARGIN sub-schedule: fire on a margin<thr run of length M (refractory refr). Advances
// the lm state every call (must be invoked once per step, also inside PUNCTLM).
static inline int lm_fire(double trimargin, double m_thr, int M, int refr, int* lmrun, int* lmrefr) {
    if (trimargin < m_thr) (*lmrun)++; else *lmrun=0;
    int fire=0; if (*lmrun==M && *lmrefr<=0){ fire=1; *lmrefr=refr; }
    if (*lmrefr>0) (*lmrefr)--;
    return fire;
}
static inline int l3_schedule_fire(int mode, uint8_t byte, int l2write, double trimargin,
                                   int K, double m_thr, int M, int refr,
                                   int* clust, int* lmrun, int* lmrefr) {
    switch (mode) {
        case L3_NONE:    return 0;
        case L3_PUNCT:   return (byte=='.'||byte=='!'||byte=='?') ? 1 : 0;
        case L3_L2CLUST: if (l2write){ (*clust)++; if (*clust>=K){ *clust=0; return 1; } } return 0;
        case L3_LOWMARGIN: return lm_fire(trimargin,m_thr,M,refr,lmrun,lmrefr);
        case L3_PUNCTLM: { int p=(byte=='.'||byte=='!'||byte=='?')?1:0;
                           int l=lm_fire(trimargin,m_thr,M,refr,lmrun,lmrefr);   // always advance lm state
                           return (p||l)?1:0; }
    }
    return 0;
}

void compute_ngrams_and_entropy(void) {
    double (*tc)[CLASSES][CLASSES]=malloc(CLASSES*CLASSES*CLASSES*sizeof(double));
    double (*tt)[CLASSES]=malloc(CLASSES*CLASSES*sizeof(double));
    memset(tc,0,CLASSES*CLASSES*CLASSES*sizeof(double)); memset(tt,0,CLASSES*CLASSES*sizeof(double));
    trigram_logits=malloc(CLASSES*CLASSES*CLASSES*sizeof(float));
    ent_table=malloc(CLASSES*CLASSES*sizeof(float));
    margin_table=malloc(CLASSES*CLASSES*sizeof(float));
    for(int i=0;i<train_len;i++){uint8_t t=target_train[i],c1=ctx_train[i],c2=ctx2_train[i]; tc[c2][c1][t]++; tt[c2][c1]++;}
    for(int i=0;i<CLASSES;i++) for(int j=0;j<CLASSES;j++){
        for(int k=0;k<CLASSES;k++) trigram_logits[i][j][k]=(float)log((tc[i][j][k]+.1)/(tt[i][j]+CLASSES*.1));
        float mx=-1e9f; for(int k=0;k<CLASSES;k++) if(trigram_logits[i][j][k]>mx) mx=trigram_logits[i][j][k];
        double se=0; for(int k=0;k<CLASSES;k++) se+=exp((double)(trigram_logits[i][j][k]-mx));
        double H=0, p1=-1,p2=-1; for(int k=0;k<CLASSES;k++){ double p=exp((double)(trigram_logits[i][j][k]-mx))/se;
            if(p>1e-12) H-=p*log(p); if(p>p1){p2=p1;p1=p;} else if(p>p2)p2=p; }
        ent_table[i][j]=(float)H;
        margin_table[i][j]=(float)(p1-p2);   // trigram prob-margin top1-top2 (parity-safe L3 signal)
    }
    free(tc); free(tt);
}

void compute_thresholds(float* surp_p90, float* surp_p80, float* ent_p80) {
    const int NB=2000; const double SMAX=20.0, EMAX=6.0;
    long *hs=calloc(NB,sizeof(long)), *he=calloc(NB,sizeof(long));
    for(int i=0;i<train_len;i++){
        float s=-trigram_logits[ctx2_train[i]][ctx_train[i]][target_train[i]];
        float e=ent_table[ctx2_train[i]][ctx_train[i]];
        int bs=(int)(s/SMAX*NB); if(bs<0)bs=0; if(bs>=NB)bs=NB-1; hs[bs]++;
        int be=(int)(e/EMAX*NB); if(be<0)be=0; if(be>=NB)be=NB-1; he[be]++;
    }
    long cum=0; *surp_p90=(float)SMAX; *surp_p80=(float)SMAX; int g90=0,g80=0;
    for(int b=0;b<NB;b++){ cum+=hs[b]; double q=(double)cum/train_len;
        if(!g80&&q>=0.80){*surp_p80=(float)((b+0.5)/NB*SMAX);g80=1;}
        if(!g90&&q>=0.90){*surp_p90=(float)((b+0.5)/NB*SMAX);g90=1;} }
    cum=0; *ent_p80=(float)EMAX; int gE=0;
    for(int b=0;b<NB;b++){ cum+=he[b]; double q=(double)cum/train_len; if(!gE&&q>=0.80){*ent_p80=(float)((b+0.5)/NB*EMAX);gE=1;} }
    free(hs); free(he);
}

// Quantile q of the per-position trigram prob-margin over the training stream: the
// "low-margin" bar for LOWMARGIN. 46.B0 found p20 fires ~0.01% in closed-loop (the M=4
// sustain bottleneck); p30/p40 are needed for phrase-scale firing. Self-calibrating;
// written to the header (the margin dist is ~identical teacher-forced vs closed-loop).
float compute_margin_pq(double q) {
    const int NB=1000; long* h=calloc(NB,sizeof(long));
    for(int i=0;i<train_len;i++){ float mg=margin_table[ctx2_train[i]][ctx_train[i]];
        int b=(int)(mg*NB); if(b<0)b=0; if(b>=NB)b=NB-1; h[b]++; }
    long cum=0; float thr=0.0f; int g=0;
    for(int b=0;b<NB;b++){ cum+=h[b]; if(!g && (double)cum/train_len>=q){ thr=(float)((b+0.5)/NB); g=1; break; } }
    free(h); return thr;
}

// Extract [SEE 192 | L2 64 | L3 64] = 320D. L2 = D1 boundary memory (gated delta-mix EMA,
// mix0.5). L3 = slow phrase-memory = EMA of fa (raw SEE), refreshed ONLY at the schedule
// events. Both update blocks are bit-identical to phase46_generator.c (parity-critical).
void extract_l23(SiliconEntropyState* see, int start, int len, float* of,
                 uint8_t* ot, uint8_t* oc, uint8_t* oc2,
                 int gate, float alpha, float surp_thr, float ent_thr, int ent_high,
                 float mix, int l3_mode, int K, float m_thr, int M, int refr) {
    see_reset(see);
    for (int i=0;i<=start+1;i++) see_observe(see,data[i]);
    float L2[L2_DIM]; memset(L2,0,sizeof(L2));
    float L3[L3_DIM]; memset(L3,0,sizeof(L3));
    float prev_bound[BASE_DIM]; memset(prev_bound,0,sizeof(prev_bound));
    int clust=0, lmrun=0, lmrefr=0;   // L3 schedule state, reset per extract call
    float scale=1.0f/sqrtf((float)BASE_DIM);
    float feat192[BASE_DIM], fa[BASE_DIM], blend[BASE_DIM];
    for (int i=0;i<len;i++){
        int g=start+i;
        ot[i]=data[g+2]; oc[i]=data[g+1]; oc2[i]=data[g];
        see_extract(see, feat192);
        float* row=&of[(size_t)i*TOT_DIM];
        memcpy(row, feat192, BASE_DIM*sizeof(float));
        memcpy(row+BASE_DIM, L2, L2_DIM*sizeof(float));
        memcpy(row+BASE_DIM+L2_DIM, L3, L3_DIM*sizeof(float));
        // margin only consumed by LOWMARGIN; guarded so Pass-0 (before margin_table exists) is safe
        double trimargin = (l3_mode==L3_LOWMARGIN||l3_mode==L3_PUNCTLM) ? margin_table[oc2[i]][oc[i]] : 0.0;
        see_observe(see, ot[i]);
        see_extract(see, fa);                              // post-observe SEE (for L2 write and L3)
        int l2write=0;
        if (gate && eval_gate(gate, ot[i], oc[i], oc2[i], surp_thr, ent_thr, ent_high)) {
            const float* src=fa;
            if (mix>0.0f){ for(int k=0;k<BASE_DIM;k++) blend[k]=fa[k]-mix*prev_bound[k]; src=blend; memcpy(prev_bound,fa,sizeof(fa)); }
            float w[L2_DIM];
            for (int j=0;j<L2_DIM;j++){ float p=0; const float* pj=Pmat[j]; for(int k=0;k<BASE_DIM;k++) p+=pj[k]*src[k]; w[j]=p*scale; }
            for (int j=0;j<L2_DIM;j++) L2[j]=alpha*L2[j]+(1.0f-alpha)*w[j];
            l2write=1;
        }
        // L3 phrase-memory: refresh the slow EMA of raw fa only at the schedule events
        if (l3_schedule_fire(l3_mode, ot[i], l2write, trimargin, K, (double)m_thr, M, refr, &clust,&lmrun,&lmrefr))
            for (int j=0;j<L3_DIM;j++) L3[j]=L3_ALPHA*L3[j]+(1.0f-L3_ALPHA)*fa[j];
    }
}

// Per-dim z-normalize + clamp (base 2.0, L2 2.0). Computes feat_mean/feat_std.
void normalize_and_clamp(float base_clamp, float l2_clamp) {
    for(int f=0;f<TOT_DIM;f++){
        double m=0; for(int i=0;i<train_len;i++) m+=features_train[(size_t)i*TOT_DIM+f]; m/=train_len;
        double v=0; for(int i=0;i<train_len;i++){double d=features_train[(size_t)i*TOT_DIM+f]-m;v+=d*d;} v=sqrt(v/train_len)+1e-8;
        feat_mean[f]=(float)m; feat_std[f]=(float)v;
        float cl=(f<BASE_DIM)?base_clamp:l2_clamp;
        for(int i=0;i<train_len;i++){float x=(features_train[(size_t)i*TOT_DIM+f]-(float)m)/(float)v; if(x>cl)x=cl; if(x<-cl)x=-cl; features_train[(size_t)i*TOT_DIM+f]=x;}
        for(int i=0;i<val_len;i++){float x=(features_val[(size_t)i*TOT_DIM+f]-(float)m)/(float)v; if(x>cl)x=cl; if(x<-cl)x=-cl; features_val[(size_t)i*TOT_DIM+f]=x;}
    }
}

void apply_l2_scale(float s) {
    if (s==1.0f) return;
    for(int i=0;i<train_len;i++) for(int f=BASE_DIM;f<TOT_DIM;f++) features_train[(size_t)i*TOT_DIM+f]*=s;
    for(int i=0;i<val_len;i++)   for(int f=BASE_DIM;f<TOT_DIM;f++) features_val[(size_t)i*TOT_DIM+f]*=s;
}

double eval_model(AdamState* m) {
    double tot=0;
    for(int i=0;i<val_len;i++){
        float lg[CLASSES],mx=-1e9f;
        for(int c=0;c<CLASSES;c++){lg[c]=m->B[c]+trigram_logits[ctx2_val[i]][ctx_val[i]][c]+dot_simd(m->W[c],&features_val[(size_t)i*TOT_DIM],TOT_DIM);if(lg[c]>mx)mx=lg[c];}
        float se=0; for(int c=0;c<CLASSES;c++) se+=expf(lg[c]-mx);
        tot-=log2(fmaxf(expf(lg[target_val[i]]-mx)/se,1e-10f));}
    return tot/val_len;
}

void train_lr(AdamState* m, int eps, int bs, float lr) {
    float *gW=malloc(CLASSES*TOT_DIM*sizeof(float)),gB[CLASSES],lg[CLASSES],pr[CLASSES];
    AdamState *best=malloc(sizeof(AdamState)); double bb=eval_model(m); memcpy(best,m,sizeof(AdamState));
    printf("    ep0 %.4f\n",bb); fflush(stdout);
    for(int ep=0;ep<eps;ep++){
        memset(gW,0,CLASSES*TOT_DIM*sizeof(float)); memset(gB,0,sizeof(gB));
        for(int i=0;i<train_len;i++){
            const float* fr=&features_train[(size_t)i*TOT_DIM];
            float mx=-1e9f;
            for(int c=0;c<CLASSES;c++){lg[c]=m->B[c]+trigram_logits[ctx2_train[i]][ctx_train[i]][c]+dot_simd(m->W[c],fr,TOT_DIM);if(lg[c]>mx)mx=lg[c];}
            float se=0; for(int c=0;c<CLASSES;c++){pr[c]=expf(lg[c]-mx);se+=pr[c];} for(int c=0;c<CLASSES;c++) pr[c]/=se;
            for(int c=0;c<CLASSES;c++){float e=pr[c]-(c==target_train[i]?1.f:0.f);gB[c]+=e/bs;grad_simd(&gW[c*TOT_DIM],fr,e/bs,TOT_DIM);}
            if((i+1)%bs==0||(i+1)==train_len){
                m->t++; float lt=lr*sqrtf(1.f-powf(.999f,m->t))/(1.f-powf(.9f,m->t));
                for(int c=0;c<CLASSES;c++){m->mB[c]=.9f*m->mB[c]+.1f*gB[c];m->vB[c]=.999f*m->vB[c]+.001f*gB[c]*gB[c];m->B[c]-=lt*(m->mB[c]/(sqrtf(m->vB[c])+1e-8f)+1e-4f*m->B[c]);
                for(int f=0;f<TOT_DIM;f++){float g=gW[c*TOT_DIM+f];m->mW[c][f]=.9f*m->mW[c][f]+.1f*g;m->vW[c][f]=.999f*m->vW[c][f]+.001f*g*g;m->W[c][f]-=lt*(m->mW[c][f]/(sqrtf(m->vW[c][f])+1e-8f)+1e-4f*m->W[c][f]);}}
                memset(gW,0,CLASSES*TOT_DIM*sizeof(float)); memset(gB,0,sizeof(gB));}}
        double b=eval_model(m); printf("    ep%d %.4f\n",ep+1,b); fflush(stdout);
        if(b<bb){bb=b;memcpy(best,m,sizeof(AdamState));}}
    memcpy(m,best,sizeof(AdamState)); free(best); free(gW);
}

// Splits the readout into SEE [0:192], L2 [192:256], L3 [256:320] logit contributions.
// Reports L2/SEE and L3/SEE logit-norm ratios, and the L3 argmax flip-rate = how often
// adding L3 changes the argmax relative to SEE+L2 (isolates L3's marginal effect).
void audit_l23(AdamState* m, double* l2_ratio, double* l3_ratio, double* l2_flip, double* l3_flip) {
    double sr2=0, sr3=0; long f2=0, f3=0;
    for(int i=0;i<val_len;i++){
        const float* f=&features_val[(size_t)i*TOT_DIM];
        float see[CLASSES], d2[CLASSES], d3[CLASSES]; double ms=0, m2=0, m3=0;
        for(int c=0;c<CLASSES;c++){
            float sc=dot_simd(m->W[c],f,BASE_DIM);
            float l2=dot_simd(m->W[c]+BASE_DIM,f+BASE_DIM,L2_DIM);
            float l3=dot_simd(m->W[c]+BASE_DIM+L2_DIM,f+BASE_DIM+L2_DIM,L3_DIM);
            see[c]=m->B[c]+trigram_logits[ctx2_val[i]][ctx_val[i]][c]+sc;
            d2[c]=l2; d3[c]=l3; ms+=see[c]; m2+=l2; m3+=l3;
        }
        ms/=CLASSES; m2/=CLASSES; m3/=CLASSES;
        double vs=0,v2=0,v3=0;
        for(int c=0;c<CLASSES;c++){ double a=see[c]-ms,b=d2[c]-m2,e=d3[c]-m3; vs+=a*a; v2+=b*b; v3+=e*e; }
        double rs=sqrt(vs/CLASSES), r2=sqrt(v2/CLASSES), r3=sqrt(v3/CLASSES);
        sr2 += (rs>1e-9)? r2/rs : 0.0;  sr3 += (rs>1e-9)? r3/rs : 0.0;
        int as=0, a2=0, a3=0; float bs=-1e30f, b2=-1e30f, b3=-1e30f;
        for(int c=0;c<CLASSES;c++){ if(see[c]>bs){bs=see[c];as=c;}
            float u2=see[c]+d2[c]; if(u2>b2){b2=u2;a2=c;}
            float u3=see[c]+d2[c]+d3[c]; if(u3>b3){b3=u3;a3=c;} }
        if(as!=a2) f2++;     // SEE vs SEE+L2
        if(a2!=a3) f3++;     // SEE+L2 vs SEE+L2+L3 (L3's marginal flip)
    }
    *l2_ratio=100.0*sr2/val_len; *l3_ratio=100.0*sr3/val_len;
    *l2_flip =100.0*f2/val_len;  *l3_flip =100.0*f3/val_len;
    printf("  AUDIT: L2/SEE ratio=%.1f%% (flip %.2f%%)   L3/SEE ratio=%.1f%% (L3 marginal flip %.2f%%)\n",
           *l2_ratio,*l2_flip,*l3_ratio,*l3_flip);
    fflush(stdout);
}

int load_c2a(const char* path, SiliconEntropyState* see, float W256[CLASSES][TOT_DIM], float B[CLASSES]) {
    FILE* f=fopen(path,"rb"); if(!f) return 0;
    uint32_t magic; fread(&magic,4,1,f); rewind(f);
    if (magic!=0x53454539){ fprintf(stderr,"Expected C2.A 0x53454539, got 0x%08x\n",magic); fclose(f); return 0; }
    uint32_t hdr4[4]; fread(hdr4,4,4,f);
    float hf[5]; fread(hf,4,5,f);
    uint32_t no=0; fread(&no,4,1,f); int n_oja=(int)no;
    if (n_oja<0||n_oja>SEE_N_OJA_MAX){ fclose(f); return 0; }
    fread(see->W_oja,sizeof(float),(size_t)n_oja*43,f); see->n_oja=n_oja;
    fseek(f,(long)CLASSES*CLASSES*CLASSES*4,SEEK_CUR);
    fseek(f,BASE_DIM*4,SEEK_CUR); fseek(f,BASE_DIM*4,SEEK_CUR);
    float row[BASE_DIM];
    for(int c=0;c<CLASSES;c++){ fread(row,sizeof(float),BASE_DIM,f); memcpy(W256[c],row,BASE_DIM*sizeof(float)); for(int k=BASE_DIM;k<TOT_DIM;k++) W256[c][k]=0.f; }
    size_t rB=fread(B,sizeof(float),CLASSES,f); fclose(f);
    return (rB==CLASSES)?1:0;
}

int main(int argc, char** argv) {
    if (argc<4){printf("Usage: %s <dataset> <weights_prefix> <c2a_weights>\n",argv[0]);return 1;}
    FILE* f=fopen(argv[1],"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",argv[1]);return 1;}
    fseek(f,0,SEEK_END); data_size=ftell(f); fseek(f,0,SEEK_SET);
    data=malloc(data_size); fread(data,1,data_size,f); fclose(f);
    train_start=0; train_len=(int)(((long long)data_size*50)/100);
    val_start=(int)(((long long)data_size*50)/100); val_len=(int)(((long long)data_size*25)/100);
    printf("Dataset %ld  train=%d  val=%d  (L2_DIM=%d, TOT=%d)\n",data_size,train_len,val_len,L2_DIM,TOT_DIM); fflush(stdout);

    features_train=malloc((size_t)train_len*TOT_DIM*sizeof(float));
    features_val  =malloc((size_t)val_len  *TOT_DIM*sizeof(float));
    target_train=malloc(train_len); ctx_train=malloc(train_len); ctx2_train=malloc(train_len);
    target_val  =malloc(val_len);   ctx_val  =malloc(val_len);   ctx2_val  =malloc(val_len);
    if(!features_train||!features_val){fprintf(stderr,"OOM features (~%.1f GB)\n",((double)train_len+val_len)*TOT_DIM*4/1e9); return 1;}
    printf("RAM: features ~%.1f GB (320D single matrix; L3 schedule alters trajectory -> fresh per config)\n",
           ((double)train_len+val_len)*TOT_DIM*4/1e9); fflush(stdout);

    SiliconEntropyState see;
    see_init(&see,42,4,0.75f);
    see.multiscale_mode=1; see.alpha_fast=0.5f; see.alpha_mid=0.9f; see.alpha_slow=0.99f;
    see.eta_oja=0.0f; see.plastic_blend=1.0f;

    AdamState* base=calloc(1,sizeof(AdamState));
    if(!load_c2a(argv[3],&see,base->W,base->B)){fprintf(stderr,"Failed to load C2.A %s\n",argv[3]);return 1;}
    printf("Loaded C2.A (n_oja=%d) + warm-start readout\n",see.n_oja); fflush(stdout);
    gen_projection(P_SEED);

    printf("Pass 0: N-grams + entropy + margin tables...\n"); fflush(stdout);
    extract_l23(&see,train_start,train_len,features_train,target_train,ctx_train,ctx2_train,G_NONE,0,0,0,0,0.0f,L3_NONE,8,0.0f,4,16);
    compute_ngrams_and_entropy();
    float surp_p90,surp_p80,ent_p80; compute_thresholds(&surp_p90,&surp_p80,&ent_p80);
    float m_p30=compute_margin_pq(0.30), m_p40=compute_margin_pq(0.40);
    printf("Thresholds: surp10=%.3f surp20=%.3f ent_high(top20)=%.3f  margin_p30=%.4f p40=%.4f\n",surp_p90,surp_p80,ent_p80,m_p30,m_p40); fflush(stdout);

    // base = D1 (mix0.5 scale0.5 alpha0.99 entropy-high gate). L3 = slow EMA(fa) refreshed at
    // the schedule. 46.B0 found p20 fires ~0.01% in closed-loop (M=4 sustain) -> raise to
    // p30/p40 (p40 ~= the 46.0 phrase scale, gap ~135). Bd1 = D1 control (dead L3 ~2.2522).
    const float MIX_D1=0.5f, SCALE_D1=0.5f;
    struct { int l3; int K; float mthr; int M; int refr; const char* sfx; const char* name; } cfgs[] = {
        {L3_NONE,     8,0.0f, 4,16, "_Bd1_control.bin","46Bd1 D1 control      (no L3)"},
        {L3_LOWMARGIN,8,m_p30,4,16, "_B1_lm30.bin",    "46B1 LOWMARGIN p30    (margin<p30, M=4)"},
        {L3_LOWMARGIN,8,m_p40,4,16, "_B2_lm40.bin",    "46B2 LOWMARGIN p40    (margin<p40, M=4)"},
        {L3_PUNCTLM,  8,m_p30,4,16, "_B3_punctlm30.bin","46B3 PUNCT+LOWMARGIN  (. ! ? OR margin<p30)"},
    };
    int n_cfg=(int)(sizeof(cfgs)/sizeof(cfgs[0]));
    double results[32], a_l2r[32], a_l3r[32], a_l2f[32], a_l3f[32];
    const float ALPHA=0.99f;
    const float LRS[]={0.0005f,0.0002f,0.0001f};

    for (int ci=0; ci<n_cfg; ci++) {
        printf("\n====== %s  (mix=%.2f scale=%.2f l3=%d K=%d mthr=%.4f M=%d refr=%d) ======\n",
               cfgs[ci].name,MIX_D1,SCALE_D1,cfgs[ci].l3,cfgs[ci].K,cfgs[ci].mthr,cfgs[ci].M,cfgs[ci].refr); fflush(stdout);
        extract_l23(&see,train_start,train_len,features_train,target_train,ctx_train,ctx2_train,G_ENTROPY,ALPHA,0,ent_p80,1,MIX_D1,cfgs[ci].l3,cfgs[ci].K,cfgs[ci].mthr,cfgs[ci].M,cfgs[ci].refr);
        extract_l23(&see,val_start,val_len,features_val,target_val,ctx_val,ctx2_val,G_ENTROPY,ALPHA,0,ent_p80,1,MIX_D1,cfgs[ci].l3,cfgs[ci].K,cfgs[ci].mthr,cfgs[ci].M,cfgs[ci].refr);
        normalize_and_clamp(FEAT_CLAMP_DEFAULT, 2.0f);
        apply_l2_scale(SCALE_D1);   // scales L2 AND L3 blocks (f>=BASE_DIM)

        AdamState* m=malloc(sizeof(AdamState)); memcpy(m,base,sizeof(AdamState));
        memset(m->mW,0,sizeof(m->mW)); memset(m->vW,0,sizeof(m->vW));
        memset(m->mB,0,sizeof(m->mB)); memset(m->vB,0,sizeof(m->vB)); m->t=0;
        for(int l=0;l<3;l++){printf("  LR %.4f\n",LRS[l]);fflush(stdout);train_lr(m,2,256,LRS[l]);}
        results[ci]=eval_model(m);
        audit_l23(m,&a_l2r[ci],&a_l3r[ci],&a_l2f[ci],&a_l3f[ci]);

        char path[1024]; snprintf(path,sizeof(path),"%s%s",argv[2],cfgs[ci].sfx);
        FILE* fw=fopen(path,"wb");
        if(fw){
            uint32_t hdr[4]={0x53454544,1,BASE_DIM,4};   // 46.A format = D1 core + L3 block
            float hf[5]={0.75f,0.1f,0.5f,FEAT_CLAMP_DEFAULT,0.0f};
            uint32_t no=(uint32_t)see.n_oja;
            fwrite(hdr,sizeof(hdr),1,fw); fwrite(hf,sizeof(float),5,fw); fwrite(&no,4,1,fw);
            fwrite(see.W_oja,sizeof(float),(size_t)see.n_oja*43,fw);
            uint32_t l2d=L2_DIM, gt=(uint32_t)G_ENTROPY, eh=1u, ps=P_SEED;
            float al=ALPHA, st=0.0f, et=ent_p80;
            fwrite(&l2d,4,1,fw); fwrite(&gt,4,1,fw); fwrite(&al,4,1,fw);
            fwrite(&st,4,1,fw); fwrite(&et,4,1,fw); fwrite(&eh,4,1,fw); fwrite(&ps,4,1,fw);
            // D1 core: l2_clamp + nb_decay + cooldown + delta-flag + mix + l2_scale + l2_cap(=0)
            float l2c=2.0f, nbd=1.0f; uint32_t cd=0, dl=1u;
            float mx=MIX_D1, ls=SCALE_D1, l2cap=0.0f;
            fwrite(&l2c,4,1,fw); fwrite(&nbd,4,1,fw); fwrite(&cd,4,1,fw); fwrite(&dl,4,1,fw);
            fwrite(&mx,4,1,fw); fwrite(&ls,4,1,fw); fwrite(&l2cap,4,1,fw);
            // L3 block: l3_dim + mode + K + m_thr + M + refr + l3_alpha
            uint32_t l3d=L3_DIM, l3m=(uint32_t)cfgs[ci].l3, l3K=(uint32_t)cfgs[ci].K, l3M=(uint32_t)cfgs[ci].M, l3R=(uint32_t)cfgs[ci].refr;
            float l3mt=cfgs[ci].mthr, l3a=L3_ALPHA;
            fwrite(&l3d,4,1,fw); fwrite(&l3m,4,1,fw); fwrite(&l3K,4,1,fw); fwrite(&l3mt,4,1,fw); fwrite(&l3M,4,1,fw); fwrite(&l3R,4,1,fw); fwrite(&l3a,4,1,fw);
            fwrite(trigram_logits,sizeof(float),CLASSES*CLASSES*CLASSES,fw);
            fwrite(feat_mean,sizeof(float),TOT_DIM,fw); fwrite(feat_std,sizeof(float),TOT_DIM,fw);
            fwrite(m->W,sizeof(float),CLASSES*TOT_DIM,fw); fwrite(m->B,sizeof(float),CLASSES,fw);
            fclose(fw);
            printf("  Saved %s  BPB=%.4f\n",path,results[ci]);
        }
        free(m);
    }

    printf("\n\n====== Phase 46.B - LOWMARGIN at phrase scale (p30/p40) + PUNCT+LM ======\n");
    printf("  C2.A reference: %.4f   D1 control (Bd1) should reproduce ~2.2522.\n",C2A_BPB);
    printf("  Promotion needs BOTH T0.65 AND T0.55 word-gate (BPB alone is not enough).\n");
    printf("  %-40s  Val BPB   dC2.A    L2/SEE%%  L2flip%%  L3/SEE%%  L3flip%%\n","config");
    for(int ci=0;ci<n_cfg;ci++) printf("  %-40s  %.4f  %+.4f   %5.1f   %5.2f    %5.1f   %5.2f\n",
        cfgs[ci].name,results[ci],results[ci]-C2A_BPB,a_l2r[ci],a_l2f[ci],a_l3r[ci],a_l3f[ci]);
    printf("\n  Reading: L3/SEE ratio>0 AND L3flip>0 = the readout actually uses L3 (not dead). Then the\n");
    printf("  word-gate decides: passes T0.65 AND T0.55 -> new phrase memory (candidate). Improves BPB but\n");
    printf("  fails gate -> same story as L2. Passes gate but loses too much BPB -> stabilizer, not memory.\n");
    free(base); free(data);
    return 0;
}
