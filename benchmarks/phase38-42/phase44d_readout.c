// Phase 44.D - Readout-L2 Homeostasis
//
// 44.C verdict: delta-write is REAL information, but the linear readout drinks it
// too hard and goes to attractor. Evidence: D_delta/alpha variants reach BPB
// ~2.221 (huge) yet word-gate FAILS, and self_BPB collapses as alpha slows
// (0.78 -> 0.21 -> 0.06) -> in teacher forcing the readout is nearly copying L2
// forward; in closed-loop that memory becomes a magnet. mix50 nearly passes at
// T=0.65 (topBi 9) but fails T=0.55. So: don't change the memory. Change HOW MUCH
// the readout may depend on L2 in the logits.
//
// Levers (per user):
//   - L2 block scale post-normalization (0.25/0.50/0.75) applied in BOTH train and
//     gen. NOTE: with the retrained readout + AdamW decay (1e-4*W), block scale s
//     is exactly L2-column weight decay x(1/s^2) (s=0.5->4x, s=0.25->16x). So
//     "L2-column regularization" and "block scale" are the SAME axis; we use the
//     block-scale parameterization (parity is trivial: both sides scale L2 dims).
//   - L2 feature dropout during training (0.05/0.10): a DIFFERENT mechanism
//     (breaks co-adaptation), training-only; inverted dropout so gen is unchanged.
//   (L2 logit-contribution cap = the only genuinely distinct non-linear lever;
//    held for 44.E, not in this priority grid.)
//
// Audit (per user): mean L2/SEE logit-norm ratio, argmax-flip rate, and the top
//   bytes where L2 most changes the prediction. Answers: how much does L2 weigh in
//   the logits, and where?
//
// Base: C2.A (0x53454539). Gate = entropy-high. Readout linear over [SEE|L2]=256D.
//   L2 write: src = SEE_t - mix*SEE_prev_boundary (mix0=absolute, mix1=delta).
//
// 8 configs (entropy-high, alpha 0.99):
//   H0      mix0.00  scale1.00  drop0.00   (control = reproduce 44.C D_H0    ~2.2526)
//   mix50   mix0.50  scale1.00  drop0.00   (control = reproduce 44.C D_mix50 ~2.2517)
//   delta   mix1.00  scale1.00  drop0.00   (control = reproduce 44.C D_delta ~2.2214)
//   D1      mix0.50  scale0.50  drop0.00
//   D2      mix1.00  scale0.25  drop0.00
//   D3      mix1.00  scale0.50  drop0.00
//   D4      mix0.50  scale1.00  drop0.10
//   D5      mix1.00  scale1.00  drop0.10
//
// Promotion gate (per user, unchanged): val BPB <= 2.2543 AND topBi<=8 AND
//   altLp<=2 AND nameWst<=20 AND runWst<=5 (self in [0.8,2.0]).
//
// Weight format 0x5345453F = 0x5345453E layout + f l2_scale (after mix):
//   ... + f mix + f l2_scale + trigram + mean[256] + std[256] + W[256*256] + B[256]
//   (l2_dropout is training-only; not serialized.)
//
// Build (from repo root):
//   gcc -O3 -march=native -mavx2 -mfma \
//       benchmarks/phase38-42/phase44d_readout.c \
//       src/silicon_entropy.c src/silicon_v0.c \
//       -o bin/phase44d_readout.exe -lm -I .
// Run:
//   bin/phase44d_readout.exe <dataset> <weights_prefix> <c2a_weights>

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

enum { G_NONE=0, G_PUNCT=1, G_WS=2, G_SURPRISE=3, G_ENTROPY=4, G_COMBINED=5 };

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

// Extract [SEE 192 | L2 64] with gated EMA + delta-mix write (44.C identical).
void extract_l2(SiliconEntropyState* see, int start, int len, float* of,
                uint8_t* ot, uint8_t* oc, uint8_t* oc2,
                int gate, float alpha, float surp_thr, float ent_thr, int ent_high, float mix) {
    see_reset(see);
    for (int i=0;i<=start+1;i++) see_observe(see,data[i]);
    float L2[L2_DIM]; memset(L2,0,sizeof(L2));
    float prev_bound[BASE_DIM]; memset(prev_bound,0,sizeof(prev_bound));
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
            for (int j=0;j<L2_DIM;j++){ float p=0; const float* pj=Pmat[j]; for(int k=0;k<BASE_DIM;k++) p+=pj[k]*src[k]; p*=scale; L2[j]=alpha*L2[j]+(1.0f-alpha)*p; }
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

// Block-scale L2 feature columns (post-normalization), train + val. Parity: the
// generator applies the same scale to L2 dims after its own normalize+clamp.
void apply_l2_scale(float s) {
    if (s==1.0f) return;
    for(int i=0;i<train_len;i++) for(int f=BASE_DIM;f<TOT_DIM;f++) features_train[(size_t)i*TOT_DIM+f]*=s;
    for(int i=0;i<val_len;i++)   for(int f=BASE_DIM;f<TOT_DIM;f++) features_val[(size_t)i*TOT_DIM+f]*=s;
}

// ---- L2 feature cache (per gate/mix/alpha group) ----------------------------
// Features depend on (gate,mix,alpha), NOT on readout scale/dropout. We extract +
// normalize ONCE per mix group, snapshot the post-normalization L2 columns, then
// rebuild each readout variant's L2 block as canon*scale (single multiply = exact,
// so scale=1.0 controls stay bit-identical to the non-cached path). Only ONE full
// feature matrix is ever resident; the canon buffer holds just the 64 L2 columns.
static float *l2_canon_train=NULL, *l2_canon_val=NULL;
void save_l2_canon(void) {
    for(int i=0;i<train_len;i++) memcpy(&l2_canon_train[(size_t)i*L2_DIM], &features_train[(size_t)i*TOT_DIM+BASE_DIM], L2_DIM*sizeof(float));
    for(int i=0;i<val_len;i++)   memcpy(&l2_canon_val[(size_t)i*L2_DIM],   &features_val[(size_t)i*TOT_DIM+BASE_DIM],   L2_DIM*sizeof(float));
}
void restore_l2_scaled(float s) {
    for(int i=0;i<train_len;i++){ float* d=&features_train[(size_t)i*TOT_DIM+BASE_DIM]; const float* c=&l2_canon_train[(size_t)i*L2_DIM]; for(int f=0;f<L2_DIM;f++) d[f]=c[f]*s; }
    for(int i=0;i<val_len;i++){   float* d=&features_val[(size_t)i*TOT_DIM+BASE_DIM];   const float* c=&l2_canon_val[(size_t)i*L2_DIM];   for(int f=0;f<L2_DIM;f++) d[f]=c[f]*s; }
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

// Train readout. dropout in [0,1) applies inverted dropout to L2 feature dims
// (training-only; expectation preserved so generation needs no dropout).
void train_lr(AdamState* m, int eps, int bs, float lr, float dropout, uint64_t* rng) {
    float *gW=malloc(CLASSES*TOT_DIM*sizeof(float)),gB[CLASSES],lg[CLASSES],pr[CLASSES],frow[TOT_DIM];
    float keep=1.0f-dropout, inv=(keep>0.0f)?1.0f/keep:1.0f;
    AdamState *best=malloc(sizeof(AdamState)); double bb=eval_model(m); memcpy(best,m,sizeof(AdamState));
    printf("    ep0 %.4f\n",bb); fflush(stdout);
    for(int ep=0;ep<eps;ep++){
        memset(gW,0,CLASSES*TOT_DIM*sizeof(float)); memset(gB,0,sizeof(gB));
        for(int i=0;i<train_len;i++){
            const float* brow=&features_train[(size_t)i*TOT_DIM];
            const float* fr=brow;
            if (dropout>0.0f){
                memcpy(frow,brow,TOT_DIM*sizeof(float));
                for(int f=BASE_DIM;f<TOT_DIM;f++){ *rng^=*rng<<13;*rng^=*rng>>7;*rng^=*rng<<17; double u=(*rng>>11)*(1.0/(1ULL<<53)); frow[f]=(u<dropout)?0.0f:frow[f]*inv; }
                fr=frow;
            }
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

// Audit: how much does the L2 block weigh in the logits, and where?
//   - mean over val of RMS(L2 contribution) / RMS(SEE+tri+bias contribution)
//   - argmax-flip rate: how often L2 changes the predicted byte
//   - top bytes by flip count and by mean |L2 logit contribution| at the prediction
void audit_l2(AdamState* m) {
    double sum_ratio=0; long flips=0;
    double byte_infl[CLASSES]; long byte_flip[CLASSES], byte_pred[CLASSES];
    memset(byte_infl,0,sizeof(byte_infl)); memset(byte_flip,0,sizeof(byte_flip)); memset(byte_pred,0,sizeof(byte_pred));
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
        byte_pred[af]++; byte_infl[af]+=fabs(d[af]);
        if(as!=af){ flips++; byte_flip[af]++; }
    }
    printf("  AUDIT L2: mean L2/SEE logit-norm ratio = %.1f%%   argmax flips = %.2f%% of val\n",
           100.0*sum_ratio/val_len, 100.0*flips/val_len);
    // top 8 bytes by flip count
    printf("    top bytes where L2 flips prediction:");
    for(int t=0;t<8;t++){ int bi=-1; long bv=0; for(int c=0;c<CLASSES;c++) if(byte_flip[c]>bv){bv=byte_flip[c];bi=c;} if(bi<0||bv==0) break; byte_flip[bi]=0;
        if(bi>=32&&bi<127) printf("  '%c'(%ld)",bi,bv); else printf("  0x%02x(%ld)",bi,bv); }
    printf("\n    top bytes by mean |L2 logit| at prediction:");
    for(int t=0;t<8;t++){ int bi=-1; double bv=0; for(int c=0;c<CLASSES;c++){ double mi=byte_pred[c]?byte_infl[c]/byte_pred[c]:0; if(mi>bv){bv=mi;bi=c;} } if(bi<0||bv<=0) break; byte_infl[bi]=-1; byte_pred[bi]=1;
        if(bi>=32&&bi<127) printf("  '%c'(%.2f)",bi,bv); else printf("  0x%02x(%.2f)",bi,bv); }
    printf("\n"); fflush(stdout);
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
    printf("AUDIT note: block scale s == L2-column weight decay x(1/s^2); gain remains a no-op and is absent.\n"); fflush(stdout);

    features_train=malloc((size_t)train_len*TOT_DIM*sizeof(float));
    features_val  =malloc((size_t)val_len  *TOT_DIM*sizeof(float));
    target_train=malloc(train_len); ctx_train=malloc(train_len); ctx2_train=malloc(train_len);
    target_val  =malloc(val_len);   ctx_val  =malloc(val_len);   ctx2_val  =malloc(val_len);
    if(!features_train||!features_val){fprintf(stderr,"OOM features (~%.1f GB)\n",((double)train_len+val_len)*TOT_DIM*4/1e9); return 1;}
    l2_canon_train=malloc((size_t)train_len*L2_DIM*sizeof(float));
    l2_canon_val  =malloc((size_t)val_len  *L2_DIM*sizeof(float));
    if(!l2_canon_train||!l2_canon_val){fprintf(stderr,"OOM L2 canon (~%.1f GB)\n",((double)train_len+val_len)*L2_DIM*4/1e9); return 1;}
    printf("RAM: features ~%.1f GB + L2 canon ~%.1f GB (peak resident; single matrix)\n",
           ((double)train_len+val_len)*TOT_DIM*4/1e9, ((double)train_len+val_len)*L2_DIM*4/1e9); fflush(stdout);

    SiliconEntropyState see;
    see_init(&see,42,4,0.75f);
    see.multiscale_mode=1; see.alpha_fast=0.5f; see.alpha_mid=0.9f; see.alpha_slow=0.99f;
    see.eta_oja=0.0f; see.plastic_blend=1.0f;

    AdamState* base=calloc(1,sizeof(AdamState));
    if(!load_c2a(argv[3],&see,base->W,base->B)){fprintf(stderr,"Failed to load C2.A %s\n",argv[3]);return 1;}
    printf("Loaded C2.A (n_oja=%d) + warm-start readout\n",see.n_oja); fflush(stdout);
    gen_projection(P_SEED);

    printf("Pass 0: N-grams + entropy table...\n"); fflush(stdout);
    extract_l2(&see,train_start,train_len,features_train,target_train,ctx_train,ctx2_train,G_NONE,0,0,0,0,0.0f);
    compute_ngrams_and_entropy();
    float surp_p90,surp_p80,ent_p80; compute_thresholds(&surp_p90,&surp_p80,&ent_p80);
    printf("Thresholds: surp10=%.3f surp20=%.3f ent_high(top20)=%.3f\n",surp_p90,surp_p80,ent_p80); fflush(stdout);

    // entropy-high gate, alpha 0.99. controls then interventions.
    struct { float mix; float l2_scale; float dropout; const char* sfx; const char* name; } cfgs[] = {
        {0.00f, 1.00f, 0.00f, "_H0.bin",          "44D H0 control      (mix0  s1.00 d0.00)"},
        {0.50f, 1.00f, 0.00f, "_mix50.bin",       "44D mix50 control   (mix.5 s1.00 d0.00)"},
        {1.00f, 1.00f, 0.00f, "_delta.bin",       "44D delta control   (mix1  s1.00 d0.00)"},
        {0.50f, 0.50f, 0.00f, "_D1_mix50_s50.bin","44D1 mix50 scale0.50"},
        {1.00f, 0.25f, 0.00f, "_D2_delta_s25.bin","44D2 delta scale0.25"},
        {1.00f, 0.50f, 0.00f, "_D3_delta_s50.bin","44D3 delta scale0.50"},
        {0.50f, 1.00f, 0.10f, "_D4_mix50_d10.bin","44D4 mix50 dropout0.10"},
        {1.00f, 1.00f, 0.10f, "_D5_delta_d10.bin","44D5 delta dropout0.10"},
    };
    int n_cfg=(int)(sizeof(cfgs)/sizeof(cfgs[0]));
    double results[32];
    const float ALPHA=0.99f;

    // Process by (mix,alpha) group: extract+normalize once, train all variants on
    // the cached L2 (rebuilt as canon*scale per variant). Order within a group:
    // the scale=1.0/drop=0 control comes first (it is the group's reference row).
    int done[32]; memset(done,0,sizeof(done));
    for (int gi=0; gi<n_cfg; gi++) {
        if (done[gi]) continue;
        float gmix=cfgs[gi].mix;
        printf("\n###### EXTRACT GROUP mix=%.2f alpha=%.4f (cached for readout variants) ######\n",gmix,ALPHA); fflush(stdout);
        extract_l2(&see,train_start,train_len,features_train,target_train,ctx_train,ctx2_train,G_ENTROPY,ALPHA,0,ent_p80,1,gmix);
        extract_l2(&see,val_start,val_len,features_val,target_val,ctx_val,ctx2_val,G_ENTROPY,ALPHA,0,ent_p80,1,gmix);
        normalize_and_clamp(FEAT_CLAMP_DEFAULT, 2.0f);
        save_l2_canon();

      for (int ci=gi; ci<n_cfg; ci++) {
        if (done[ci] || cfgs[ci].mix!=gmix) continue;
        done[ci]=1;
        printf("\n====== %s  (mix=%.2f scale=%.2f drop=%.2f) [cached extract] ======\n",cfgs[ci].name,cfgs[ci].mix,cfgs[ci].l2_scale,cfgs[ci].dropout); fflush(stdout);
        restore_l2_scaled(cfgs[ci].l2_scale);

        AdamState* m=malloc(sizeof(AdamState)); memcpy(m,base,sizeof(AdamState));
        memset(m->mW,0,sizeof(m->mW)); memset(m->vW,0,sizeof(m->vW));
        memset(m->mB,0,sizeof(m->mB)); memset(m->vB,0,sizeof(m->vB)); m->t=0;
        uint64_t drng = 0x44D0000000000000ULL + (uint64_t)(ci+1)*0x9E3779B97F4A7C15ULL;
        float lrs[]={0.0005f,0.0002f,0.0001f};
        for(int l=0;l<3;l++){printf("  LR %.4f\n",lrs[l]);fflush(stdout);train_lr(m,2,256,lrs[l],cfgs[ci].dropout,&drng);}
        results[ci]=eval_model(m);
        audit_l2(m);

        char path[1024]; snprintf(path,sizeof(path),"%s%s",argv[2],cfgs[ci].sfx);
        FILE* fw=fopen(path,"wb");
        if(fw){
            uint32_t hdr[4]={0x5345453F,1,BASE_DIM,4};
            float hf[5]={0.75f,0.1f,0.5f,FEAT_CLAMP_DEFAULT,0.0f};
            uint32_t no=(uint32_t)see.n_oja;
            fwrite(hdr,sizeof(hdr),1,fw); fwrite(hf,sizeof(float),5,fw); fwrite(&no,4,1,fw);
            fwrite(see.W_oja,sizeof(float),(size_t)see.n_oja*43,fw);
            uint32_t l2d=L2_DIM, gt=(uint32_t)G_ENTROPY, eh=1u, ps=P_SEED;
            float al=ALPHA, st=0.0f, et=ent_p80;
            fwrite(&l2d,4,1,fw); fwrite(&gt,4,1,fw); fwrite(&al,4,1,fw);
            fwrite(&st,4,1,fw); fwrite(&et,4,1,fw); fwrite(&eh,4,1,fw); fwrite(&ps,4,1,fw);
            // homeostasis block (no decay/cooldown) + delta-flag + mix + l2_scale
            float l2c=2.0f, nbd=1.0f; uint32_t cd=0, dl=(cfgs[ci].mix>0.0f)?1u:0u;
            float mx=cfgs[ci].mix, ls=cfgs[ci].l2_scale;
            fwrite(&l2c,4,1,fw); fwrite(&nbd,4,1,fw); fwrite(&cd,4,1,fw); fwrite(&dl,4,1,fw);
            fwrite(&mx,4,1,fw); fwrite(&ls,4,1,fw);
            fwrite(trigram_logits,sizeof(float),CLASSES*CLASSES*CLASSES,fw);
            fwrite(feat_mean,sizeof(float),TOT_DIM,fw); fwrite(feat_std,sizeof(float),TOT_DIM,fw);
            fwrite(m->W,sizeof(float),CLASSES*TOT_DIM,fw); fwrite(m->B,sizeof(float),CLASSES,fw);
            fclose(fw);
            printf("  Saved %s  BPB=%.4f\n",path,results[ci]);
        }
        free(m);
      }
    }

    printf("\n\n====== Phase 44.D - Readout-L2 Homeostasis ======\n");
    printf("  C2.A reference: %.4f  (promote: BPB<=%.4f + topBi<=8 + altLp<=2 + nameWst<=20 + runWst<=5)\n",C2A_BPB,C2A_BPB-0.005);
    printf("  %-36s  Val BPB   Delta vs C2.A\n","config");
    for(int ci=0;ci<n_cfg;ci++) printf("  %-36s  %.4f    %+.4f\n",cfgs[ci].name,results[ci],results[ci]-C2A_BPB);
    int best=0; for(int ci=1;ci<n_cfg;ci++) if(results[ci]<results[best]) best=ci;
    printf("\n  Best BPB: %s (%.4f, %+.4f vs C2.A) [word-gate decides promotion]\n",cfgs[best].name,results[best],results[best]-C2A_BPB);
    free(base); free(data);
    return 0;
}
