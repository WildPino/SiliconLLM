// Phase 45.C - L2 write-gate / timing tribunal (substrate-side, train-time)
//
// 45.A NEGATIVE: capping the per-event move amplitude killed the delta gain. 45.B NEGATIVE:
// no simple write GEOMETRY (blend/ortho/energy) stabilized it. But 45.B exposed the key
// fact: B0 delta has cos(write,L2_old) ~ -0.70 - the useful delta write is ANTI-aligned to
// the accumulated L2 state. It is a CORRECTION / REVERSAL dynamic, not accumulation; making
// it parallel (blend, cos->+1) or removing the axial part (ortho/novelty) destroys the gain.
// So the signal IS the strong anti-L2 reversal - but in closed loop that becomes oscillatory.
//
// 45.C does not touch amplitude or direction. It gates WHEN a write is admitted at all:
// accept or DROP the whole EMA write per boundary, on internal scale-free signals only
// (reversal angle cos(w,L2_old), candidate rel_move). Applied IDENTICALLY here and in
// phase44_generator.c l2_evolve via wtgate_accept (parity). WT_NONE = plain delta EMA.
//   C0 none      : accept every write (= delta control, reproduce ~2.2212)
//   C1 rev25     : accept iff cos(w,L2) < -0.25   (admit only reversals)
//   C2 rev50     : accept iff cos(w,L2) < -0.50   (admit only STRONG reversals)
//   C3 revmargin : accept iff cos(w,L2) < -0.50 AND candidate rel_move > 1.0 (strong+large)
//   C4 cooldown  : after a strong reversal (cos<-0.50) is admitted, block the next 3 boundaries
//   C5 thin      : deterministic 1-in-N: accept 1 delta write every N=2 boundaries
//
// Base = delta (mix1.0 scale0.5, alpha0.99, entropy-high gate). Geometry OFF (WG_NONE) for
// all configs - this is a pure timing test on the delta regime. Warm-start from C2.A. Fresh
// extraction per config (the gate alters the L2 trajectory -> no canon cache). No parallel
// trainers (ps1 single-trainer guard).
//
// Audit: BPB + L2/SEE ratio + flip (trainer); accept-rate, cos(write,L2_old) on accepted
// writes, rel_move, |Lwin| via generator telemetry (ps1). Word-gate T0.65/T0.55.
//
// Weight format 0x53454543 = 0x53454542 layout + write-gate block (after u32 wgeom):
//   ... + u32 wgeom + {u32 wt_mode, f wt_cos, f wt_stress, u32 wt_cool_k, u32 wt_thin_n} + trigram + ...
//
// Build (from repo root):
//   gcc -O3 -march=native -mavx2 -mfma \
//       benchmarks/phase38-42/phase45c_gate.c \
//       src/silicon_entropy.c src/silicon_v0.c \
//       -o bin/phase45c_gate.exe -lm -I .
// Run:
//   bin/phase45c_gate.exe <dataset> <weights_prefix> <c2a_weights>

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
#define TOT_DIM   (BASE_DIM + L2_DIM)
#define FEAT_CLAMP_DEFAULT 2.0f
#define C2A_BPB   2.2593
#define P_SEED    0xB5297A4Du
#define L2_RELMOVE_EPS 1.0f   // orthogonalization floor: only act once a state exists

enum { G_NONE=0, G_PUNCT=1, G_WS=2, G_SURPRISE=3, G_ENTROPY=4, G_COMBINED=5 };
enum { WG_NONE=0, WG_BLEND75=1, WG_BLEND50=2, WG_ORTHO=3, WG_NOVELTY=4, WG_ENERGY=5 };
enum { WT_NONE=0, WT_REV25=1, WT_REV50=2, WT_REVMARGIN=3, WT_COOLDOWN=4, WT_THIN=5 };

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

// Phase 45.B write geometry - BYTE-IDENTICAL to phase44_generator.c l2b_write_geometry
// (parity-critical). Transforms the delta write w (L2 space) in place. fa = absolute
// post-boundary features (for the H0 projection); pdnorm = ||delta_write||.
static inline void wgeom_apply(float* w, const float* L2, const float* fa, float scale, double pdnorm, int mode) {
    if (mode==WG_NONE) return;
    if (mode==WG_BLEND75 || mode==WG_BLEND50) {
        float ph[L2_DIM];
        for (int j=0;j<L2_DIM;j++){ float p=0; const float* pj=Pmat[j]; for(int k=0;k<BASE_DIM;k++) p+=pj[k]*fa[k]; ph[j]=p*scale; }
        float a=(mode==WG_BLEND75)?0.75f:0.5f, b=1.0f-a;
        double bl2=0.0; float bl[L2_DIM];
        for (int j=0;j<L2_DIM;j++){ bl[j]=a*w[j]+b*ph[j]; bl2+=(double)bl[j]*bl[j]; }
        double bln=sqrt(bl2);
        float s=(bln>1e-12)?(float)(pdnorm/bln):1.0f;
        for (int j=0;j<L2_DIM;j++) w[j]=bl[j]*s;
    } else if (mode==WG_ORTHO || mode==WG_NOVELTY) {
        double l2n2=0.0; for(int j=0;j<L2_DIM;j++) l2n2+=(double)L2[j]*L2[j];
        if (l2n2>(double)L2_RELMOVE_EPS*L2_RELMOVE_EPS){
            double dot=0.0; for(int j=0;j<L2_DIM;j++) dot+=(double)w[j]*L2[j];
            float coef=(float)(dot/l2n2);
            for (int j=0;j<L2_DIM;j++) w[j]-=coef*L2[j];
            if (mode==WG_NOVELTY){
                double wn2=0.0; for(int j=0;j<L2_DIM;j++) wn2+=(double)w[j]*w[j];
                double wn=sqrt(wn2);
                float s=(wn>1e-12)?(float)(pdnorm/wn):1.0f;
                for (int j=0;j<L2_DIM;j++) w[j]*=s;
            }
        }
    } else if (mode==WG_ENERGY) {
        float s=(pdnorm>1e-12)?(float)(1.0/pdnorm):1.0f;
        for (int j=0;j<L2_DIM;j++) w[j]*=s;
    }
}

// Phase 45.C write-gate - BYTE-IDENTICAL to phase44_generator.c wtgate_accept
// (parity-critical). Decides whether to ACCEPT this boundary's EMA write or DROP it,
// from scale-free internal signals only: cos(w,L2_old) reversal angle and candidate
// rel_move. Bootstrap (||L2||<=EPS or ||w||~0): always accept; WT_THIN still advances
// its counter so its phase stays deterministic. cool/thin persist via pointer.
static inline int wtgate_accept(const float* w, const float* L2, float alpha,
                                int mode, float cos_thr, float stress_thr,
                                int cool_k, int thin_n, int* cool, int* thin) {
    if (mode==WT_NONE) return 1;
    double l2n2=0.0, wn2=0.0, dot=0.0;
    for (int j=0;j<L2_DIM;j++){ l2n2+=(double)L2[j]*L2[j]; wn2+=(double)w[j]*w[j]; dot+=(double)w[j]*L2[j]; }
    if (l2n2 <= (double)L2_RELMOVE_EPS*L2_RELMOVE_EPS || wn2<=1e-24){
        if (mode==WT_THIN){ int a=(((*thin)%thin_n)==0)?1:0; (*thin)++; return a; }
        return 1;
    }
    double cosWL2 = dot/(sqrt(wn2)*sqrt(l2n2));
    double dn2=0.0;
    for (int j=0;j<L2_DIM;j++){ double cand=(double)alpha*L2[j]+(1.0-(double)alpha)*w[j]; double dd=cand-(double)L2[j]; dn2+=dd*dd; }
    double denom=(sqrt(l2n2)>(double)L2_RELMOVE_EPS)?sqrt(l2n2):(double)L2_RELMOVE_EPS;
    double relmove=sqrt(dn2)/denom;
    switch (mode){
        case WT_REV25:     return (cosWL2 < (double)cos_thr) ? 1 : 0;
        case WT_REV50:     return (cosWL2 < (double)cos_thr) ? 1 : 0;
        case WT_REVMARGIN: return (cosWL2 < (double)cos_thr && relmove > (double)stress_thr) ? 1 : 0;
        case WT_COOLDOWN:  if (*cool > 0){ (*cool)--; return 0; }
                           if (cosWL2 < (double)cos_thr) *cool = cool_k;
                           return 1;
        case WT_THIN:      { int a=(((*thin)%thin_n)==0)?1:0; (*thin)++; return a; }
    }
    return 1;
}

void compute_ngrams_and_entropy(void) {
    double (*tc)[CLASSES][CLASSES]=malloc(CLASSES*CLASSES*CLASSES*sizeof(double));
    double (*tt)[CLASSES]=malloc(CLASSES*CLASSES*sizeof(double));
    memset(tc,0,CLASSES*CLASSES*CLASSES*sizeof(double)); memset(tt,0,CLASSES*CLASSES*sizeof(double));
    trigram_logits=malloc(CLASSES*CLASSES*CLASSES*sizeof(float));
    ent_table=malloc(CLASSES*CLASSES*sizeof(float));
    for(int i=0;i<train_len;i++){uint8_t t=target_train[i],c1=ctx_train[i],c2=ctx2_train[i]; tc[c2][c1][t]++; tt[c2][c1]++;}
    for(int i=0;i<CLASSES;i++) for(int j=0;j<CLASSES;j++){
        for(int k=0;k<CLASSES;k++) trigram_logits[i][j][k]=(float)log((tc[i][j][k]+.1)/(tt[i][j]+CLASSES*.1));
        float mx=-1e9f; for(int k=0;k<CLASSES;k++) if(trigram_logits[i][j][k]>mx) mx=trigram_logits[i][j][k];
        double se=0; for(int k=0;k<CLASSES;k++) se+=exp((double)(trigram_logits[i][j][k]-mx));
        double H=0; for(int k=0;k<CLASSES;k++){ double p=exp((double)(trigram_logits[i][j][k]-mx))/se; if(p>1e-12) H-=p*log(p); }
        ent_table[i][j]=(float)H;
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

// Extract [SEE 192 | L2 64] with gated EMA + delta-mix write + 45.C write-gate (timing).
// The write block is bit-identical to phase44_generator.c l2_evolve (parity-critical).
// wgeom stays WG_NONE for all 45.C configs (geometry off); wt_* drive the write-gate.
void extract_l2(SiliconEntropyState* see, int start, int len, float* of,
                uint8_t* ot, uint8_t* oc, uint8_t* oc2,
                int gate, float alpha, float surp_thr, float ent_thr, int ent_high,
                float mix, int wgeom,
                int wt_mode, float wt_cos, float wt_stress, int wt_cool_k, int wt_thin_n) {
    see_reset(see);
    for (int i=0;i<=start+1;i++) see_observe(see,data[i]);
    float L2[L2_DIM]; memset(L2,0,sizeof(L2));
    float prev_bound[BASE_DIM]; memset(prev_bound,0,sizeof(prev_bound));
    int wt_cool=0, wt_thin=0;   // write-gate state, reset per extract_l2 call (parity: gen resets once pre-warmup)
    float scale=1.0f/sqrtf((float)BASE_DIM);
    float feat192[BASE_DIM], fa[BASE_DIM], blend[BASE_DIM];
    for (int i=0;i<len;i++){
        int g=start+i;
        ot[i]=data[g+2]; oc[i]=data[g+1]; oc2[i]=data[g];
        see_extract(see, feat192);
        float* row=&of[(size_t)i*TOT_DIM];
        memcpy(row, feat192, BASE_DIM*sizeof(float));
        memcpy(row+BASE_DIM, L2, L2_DIM*sizeof(float));
        see_observe(see, ot[i]);
        if (gate && eval_gate(gate, ot[i], oc[i], oc2[i], surp_thr, ent_thr, ent_high)) {
            see_extract(see, fa);
            const float* src=fa;
            if (mix>0.0f){ for(int k=0;k<BASE_DIM;k++) blend[k]=fa[k]-mix*prev_bound[k]; src=blend; memcpy(prev_bound,fa,sizeof(fa)); }
            float w[L2_DIM]; double pd2=0.0;
            for (int j=0;j<L2_DIM;j++){ float p=0; const float* pj=Pmat[j]; for(int k=0;k<BASE_DIM;k++) p+=pj[k]*src[k]; p*=scale; w[j]=p; pd2+=(double)p*p; }
            double pdnorm=sqrt(pd2);
            wgeom_apply(w, L2, fa, scale, pdnorm, wgeom);   // geometry off in 45.C (WG_NONE = no-op)
            // 45.C write-gate: admit or DROP this boundary's write (timing, not amplitude/geometry)
            if (wtgate_accept(w, L2, alpha, wt_mode, wt_cos, wt_stress, wt_cool_k, wt_thin_n, &wt_cool, &wt_thin))
                for (int j=0;j<L2_DIM;j++) L2[j]=alpha*L2[j]+(1.0f-alpha)*w[j];   // EMA, no amplitude cap
            // reject: L2 unchanged; prev_bound already advanced (delta tracks every boundary)
        }
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

void audit_l2(AdamState* m, double* out_ratio, double* out_flip) {
    double sum_ratio=0; long flips=0;
    for(int i=0;i<val_len;i++){
        const float* f=&features_val[(size_t)i*TOT_DIM];
        float see[CLASSES], d[CLASSES]; double ms=0, md=0;
        for(int c=0;c<CLASSES;c++){
            float sc=dot_simd(m->W[c],f,BASE_DIM);
            float lc=dot_simd(m->W[c]+BASE_DIM,f+BASE_DIM,L2_DIM);
            see[c]=m->B[c]+trigram_logits[ctx2_val[i]][ctx_val[i]][c]+sc;
            d[c]=lc; ms+=see[c]; md+=d[c];
        }
        ms/=CLASSES; md/=CLASSES;
        double vs=0,vd=0;
        for(int c=0;c<CLASSES;c++){ double a=see[c]-ms, b=d[c]-md; vs+=a*a; vd+=b*b; }
        double rs=sqrt(vs/CLASSES), rd=sqrt(vd/CLASSES);
        sum_ratio += (rs>1e-9)? rd/rs : 0.0;
        int as=0, af=0; float bs=-1e30f, bf=-1e30f;
        for(int c=0;c<CLASSES;c++){ if(see[c]>bs){bs=see[c];as=c;} float fu=see[c]+d[c]; if(fu>bf){bf=fu;af=c;} }
        if(as!=af) flips++;
    }
    *out_ratio=100.0*sum_ratio/val_len; *out_flip=100.0*flips/val_len;
    printf("  AUDIT L2: mean L2/SEE logit-norm ratio = %.1f%%   argmax flips = %.2f%% of val\n",*out_ratio,*out_flip);
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
    printf("RAM: features ~%.1f GB (single matrix; geometry alters trajectory -> no canon cache)\n",
           ((double)train_len+val_len)*TOT_DIM*4/1e9); fflush(stdout);

    SiliconEntropyState see;
    see_init(&see,42,4,0.75f);
    see.multiscale_mode=1; see.alpha_fast=0.5f; see.alpha_mid=0.9f; see.alpha_slow=0.99f;
    see.eta_oja=0.0f; see.plastic_blend=1.0f;

    AdamState* base=calloc(1,sizeof(AdamState));
    if(!load_c2a(argv[3],&see,base->W,base->B)){fprintf(stderr,"Failed to load C2.A %s\n",argv[3]);return 1;}
    printf("Loaded C2.A (n_oja=%d) + warm-start readout\n",see.n_oja); fflush(stdout);
    gen_projection(P_SEED);

    printf("Pass 0: N-grams + entropy table...\n"); fflush(stdout);
    extract_l2(&see,train_start,train_len,features_train,target_train,ctx_train,ctx2_train,G_NONE,0,0,0,0,0.0f,WG_NONE,WT_NONE,0,0,0,0);
    compute_ngrams_and_entropy();
    float surp_p90,surp_p80,ent_p80; compute_thresholds(&surp_p90,&surp_p80,&ent_p80);
    printf("Thresholds: surp10=%.3f surp20=%.3f ent_high(top20)=%.3f\n",surp_p90,surp_p80,ent_p80); fflush(stdout);

    // entropy-high gate, alpha 0.99, delta base (mix1 scale0.5), geometry OFF. write-gate grid.
    // cos_thr<0 = reversal; stress_thr=rel_move floor (C3); cool_k=blocked boundaries (C4); thin_n (C5).
    struct { float mix; float l2_scale; int wgeom; int wt; float wcos; float wstr; int wcool; int wthin;
             const char* sfx; const char* name; } cfgs[] = {
        {1.00f,0.50f,WG_NONE,WT_NONE,     0.00f,1.0f,3,2, "_C0_delta.bin",    "45C0 delta control      (accept all)"},
        {1.00f,0.50f,WG_NONE,WT_REV25,   -0.25f,1.0f,3,2, "_C1_rev25.bin",    "45C1 reversal gate       cos<-0.25"},
        {1.00f,0.50f,WG_NONE,WT_REV50,   -0.50f,1.0f,3,2, "_C2_rev50.bin",    "45C2 strong reversal     cos<-0.50"},
        {1.00f,0.50f,WG_NONE,WT_REVMARGIN,-0.50f,1.0f,3,2,"_C3_revmargin.bin","45C3 strong+large rev    cos<-0.50&rm>1.0"},
        {1.00f,0.50f,WG_NONE,WT_COOLDOWN,-0.50f,1.0f,3,2, "_C4_cooldown.bin", "45C4 cooldown after rev  block 3 boundaries"},
        {1.00f,0.50f,WG_NONE,WT_THIN,     0.00f,1.0f,3,2, "_C5_thin.bin",     "45C5 event-thinned       1 write per N=2"},
    };
    int n_cfg=(int)(sizeof(cfgs)/sizeof(cfgs[0]));
    double results[32], audit_ratio[32], audit_flip[32];
    const float ALPHA=0.99f;
    const float LRS[]={0.0005f,0.0002f,0.0001f};

    for (int ci=0; ci<n_cfg; ci++) {
        printf("\n====== %s  (mix=%.2f scale=%.2f wt=%d cos=%.2f rm=%.2f cool=%d thin=%d) ======\n",
               cfgs[ci].name,cfgs[ci].mix,cfgs[ci].l2_scale,cfgs[ci].wt,cfgs[ci].wcos,cfgs[ci].wstr,cfgs[ci].wcool,cfgs[ci].wthin); fflush(stdout);
        extract_l2(&see,train_start,train_len,features_train,target_train,ctx_train,ctx2_train,G_ENTROPY,ALPHA,0,ent_p80,1,cfgs[ci].mix,cfgs[ci].wgeom,cfgs[ci].wt,cfgs[ci].wcos,cfgs[ci].wstr,cfgs[ci].wcool,cfgs[ci].wthin);
        extract_l2(&see,val_start,val_len,features_val,target_val,ctx_val,ctx2_val,G_ENTROPY,ALPHA,0,ent_p80,1,cfgs[ci].mix,cfgs[ci].wgeom,cfgs[ci].wt,cfgs[ci].wcos,cfgs[ci].wstr,cfgs[ci].wcool,cfgs[ci].wthin);
        normalize_and_clamp(FEAT_CLAMP_DEFAULT, 2.0f);
        apply_l2_scale(cfgs[ci].l2_scale);

        AdamState* m=malloc(sizeof(AdamState)); memcpy(m,base,sizeof(AdamState));
        memset(m->mW,0,sizeof(m->mW)); memset(m->vW,0,sizeof(m->vW));
        memset(m->mB,0,sizeof(m->mB)); memset(m->vB,0,sizeof(m->vB)); m->t=0;
        for(int l=0;l<3;l++){printf("  LR %.4f\n",LRS[l]);fflush(stdout);train_lr(m,2,256,LRS[l]);}
        results[ci]=eval_model(m);
        audit_l2(m,&audit_ratio[ci],&audit_flip[ci]);

        char path[1024]; snprintf(path,sizeof(path),"%s%s",argv[2],cfgs[ci].sfx);
        FILE* fw=fopen(path,"wb");
        if(fw){
            uint32_t hdr[4]={0x53454543,1,BASE_DIM,4};
            float hf[5]={0.75f,0.1f,0.5f,FEAT_CLAMP_DEFAULT,0.0f};
            uint32_t no=(uint32_t)see.n_oja;
            fwrite(hdr,sizeof(hdr),1,fw); fwrite(hf,sizeof(float),5,fw); fwrite(&no,4,1,fw);
            fwrite(see.W_oja,sizeof(float),(size_t)see.n_oja*43,fw);
            uint32_t l2d=L2_DIM, gt=(uint32_t)G_ENTROPY, eh=1u, ps=P_SEED;
            float al=ALPHA, st=0.0f, et=ent_p80;
            fwrite(&l2d,4,1,fw); fwrite(&gt,4,1,fw); fwrite(&al,4,1,fw);
            fwrite(&st,4,1,fw); fwrite(&et,4,1,fw); fwrite(&eh,4,1,fw); fwrite(&ps,4,1,fw);
            // homeostasis + delta-flag + mix + l2_scale + l2_cap(=0) + relmove_cap(=0) + wgeom + write-gate
            float l2c=2.0f, nbd=1.0f; uint32_t cd=0, dl=(cfgs[ci].mix>0.0f)?1u:0u;
            float mx=cfgs[ci].mix, ls=cfgs[ci].l2_scale, l2cap=0.0f, rmc=0.0f;
            uint32_t wg=(uint32_t)cfgs[ci].wgeom;
            fwrite(&l2c,4,1,fw); fwrite(&nbd,4,1,fw); fwrite(&cd,4,1,fw); fwrite(&dl,4,1,fw);
            fwrite(&mx,4,1,fw); fwrite(&ls,4,1,fw); fwrite(&l2cap,4,1,fw); fwrite(&rmc,4,1,fw); fwrite(&wg,4,1,fw);
            // 0x53454543 write-gate block: mode + cos_thr + stress_thr + cool_k + thin_n
            uint32_t wtm=(uint32_t)cfgs[ci].wt, wck=(uint32_t)cfgs[ci].wcool, wtn=(uint32_t)cfgs[ci].wthin;
            float wcos=cfgs[ci].wcos, wstr=cfgs[ci].wstr;
            fwrite(&wtm,4,1,fw); fwrite(&wcos,4,1,fw); fwrite(&wstr,4,1,fw); fwrite(&wck,4,1,fw); fwrite(&wtn,4,1,fw);
            fwrite(trigram_logits,sizeof(float),CLASSES*CLASSES*CLASSES,fw);
            fwrite(feat_mean,sizeof(float),TOT_DIM,fw); fwrite(feat_std,sizeof(float),TOT_DIM,fw);
            fwrite(m->W,sizeof(float),CLASSES*TOT_DIM,fw); fwrite(m->B,sizeof(float),CLASSES,fw);
            fclose(fw);
            printf("  Saved %s  BPB=%.4f\n",path,results[ci]);
        }
        free(m);
    }

    printf("\n\n====== Phase 45.C - L2 write-gate / timing ======\n");
    printf("  C2.A reference: %.4f  (promote: BPB<=%.4f + topBi<=8 + altLp<=2 + nameWst<=20 + runWst<=5)\n",C2A_BPB,C2A_BPB-0.005);
    printf("  %-40s  Val BPB   Delta vs C2.A   L2/SEE%%   flip%%\n","config");
    for(int ci=0;ci<n_cfg;ci++) printf("  %-40s  %.4f    %+.4f      %5.1f   %5.2f\n",
        cfgs[ci].name,results[ci],results[ci]-C2A_BPB,audit_ratio[ci],audit_flip[ci]);
    printf("\n  Control: C0 delta should reproduce ~2.2212. Word-gate + accept-rate/cos telemetry decide promotion.\n");
    printf("  Reading: if rev-gating (C1/C2) keeps part of the delta BPB AND drops the loop -> internal timing\n");
    printf("  found (silicon wants strong reversals, but not at every boundary). If it loses everything ->\n");
    printf("  delta is a teacher-forced-only signal, archive it as non-generative.\n");
    free(base); free(data);
    return 0;
}
