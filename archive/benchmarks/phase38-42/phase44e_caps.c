// Phase 44.E - Readout L2 Logit-Contribution Caps (conditional trust)
//
// 44.D verdict: static L2 block scale helps but is not enough. D1 (mix50 scale0.50)
// passes T=0.65 keeping almost all the BPB gain, but fails T=0.55. delta is archived
// (teacher-forced strong, closed-loop fragile). dropout is rejected (D4 collapses).
// Next lever = CONDITIONAL trust: cap L2's contribution to the logits ONLY when it
// tries to dominate, so it is dormant when benign and a leash when runaway.
//
// Cap (dynamic, per prediction step), using the SAME centered-RMS ratio the 44.D
// audit reports:
//   s_c = B_c + tri_c + W_SEE_c . f_SEE          (SEE logit contribution)
//   d_c = W_L2_c . f_L2                           (L2 logit contribution, scaled)
//   ratio = RMS_c(d-mean d) / RMS_c(s-mean s)
//   gamma = (ratio > cap) ? cap/ratio : 1
//   logit_c = (s_c + gamma*d_c) / temp
// "cap 2.0%" = L2 logit-RMS may be at most 0.02 * SEE logit-RMS. Fires only when L2
// over-asserts (attractor forming) -> near-zero teacher-forced BPB cost expected.
//
// INFERENCE-ONLY: the cap is applied at eval-BPB and generation on top of the
// already-trained readouts (NO retraining). The cap variants are D1/H0 weights +
// a cap field; we just re-evaluate BPB with the cap active. (Train-time co-adapted
// cap would be 44.F, only if leashing helps generation but costs too much BPB.)
//
// Bases trained (entropy-high gate, alpha 0.99): H0 (mix0 scale1.0), mix50 (mix0.5
// scale1.0), D1 (mix0.5 scale0.5). Cap files derived for free:
//   D1_cap20 = D1 + cap0.020   D1_cap15 = D1 + cap0.015   D1_cap10 = D1 + cap0.010
//   H0_cap20 = H0 + cap0.020
// Reference (generation only, via 43-gen): C2.A.
//
// Criterion (per user): NOT max BPB. Does the T=0.65 pass extend to T=0.55 without
// losing more than ~0.003-0.005 BPB? Gate unchanged: BPB<=2.2543 + topBi<=8 +
// altLp<=2 + nameWst<=20 + runWst<=5 (self in [0.8,2.0]).
//
// Weight format 0x53454540 = 0x5345453F layout + f l2_cap (after l2_scale):
//   ... + f mix + f l2_scale + f l2_cap + trigram + mean[256] + std[256] + W + B
//
// Build (from repo root):
//   gcc -O3 -march=native -mavx2 -mfma \
//       benchmarks/phase38-42/phase44e_caps.c \
//       src/silicon_entropy.c src/silicon_v0.c \
//       -o bin/phase44e_caps.exe -lm -I .
// Run:
//   bin/phase44e_caps.exe <dataset> <weights_prefix> <c2a_weights>

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

// L2 feature cache (per gate/mix/alpha group): extract+normalize once, snapshot L2
// columns, rebuild each readout variant as canon*scale (single multiply = exact).
static float *l2_canon_train=NULL, *l2_canon_val=NULL;
void save_l2_canon(void) {
    for(int i=0;i<train_len;i++) memcpy(&l2_canon_train[(size_t)i*L2_DIM], &features_train[(size_t)i*TOT_DIM+BASE_DIM], L2_DIM*sizeof(float));
    for(int i=0;i<val_len;i++)   memcpy(&l2_canon_val[(size_t)i*L2_DIM],   &features_val[(size_t)i*TOT_DIM+BASE_DIM],   L2_DIM*sizeof(float));
}
void restore_l2_scaled(float s) {
    for(int i=0;i<train_len;i++){ float* d=&features_train[(size_t)i*TOT_DIM+BASE_DIM]; const float* c=&l2_canon_train[(size_t)i*L2_DIM]; for(int f=0;f<L2_DIM;f++) d[f]=c[f]*s; }
    for(int i=0;i<val_len;i++){   float* d=&features_val[(size_t)i*TOT_DIM+BASE_DIM];   const float* c=&l2_canon_val[(size_t)i*L2_DIM];   for(int f=0;f<L2_DIM;f++) d[f]=c[f]*s; }
}

// gamma in (0,1]: cap<=0 -> 1. ratio = centered-RMS(d)/centered-RMS(s) over classes.
static inline float l2_cap_gamma(const float* s, const float* d, int n, float cap) {
    if (cap<=0.0f) return 1.0f;
    double ms=0,md=0; for(int c=0;c<n;c++){ms+=s[c];md+=d[c];} ms/=n; md/=n;
    double vs=0,vd=0; for(int c=0;c<n;c++){double a=s[c]-ms,b=d[c]-md; vs+=a*a; vd+=b*b;}
    double ns=sqrt(vs/n), nd=sqrt(vd/n);
    if (nd<=1e-12 || ns<=1e-12) return 1.0f;
    double ratio=nd/ns;
    return (ratio>cap)? (float)(cap/ratio) : 1.0f;
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

// BPB with the inference-time L2 cap active. cap<=0 -> identical to eval_model.
// Reports the fraction of val steps where the cap fired (gamma<1).
double eval_model_capped(AdamState* m, float cap, double* fire_rate) {
    double tot=0; long fires=0;
    for(int i=0;i<val_len;i++){
        const float* f=&features_val[(size_t)i*TOT_DIM];
        float s[CLASSES], d[CLASSES];
        for(int c=0;c<CLASSES;c++){
            s[c]=m->B[c]+trigram_logits[ctx2_val[i]][ctx_val[i]][c]+dot_simd(m->W[c],f,BASE_DIM);
            d[c]=dot_simd(m->W[c]+BASE_DIM,f+BASE_DIM,L2_DIM);
        }
        float gamma=l2_cap_gamma(s,d,CLASSES,cap); if(gamma<1.0f) fires++;
        float lg[CLASSES],mx=-1e9f;
        for(int c=0;c<CLASSES;c++){ lg[c]=s[c]+gamma*d[c]; if(lg[c]>mx)mx=lg[c]; }
        float se=0; for(int c=0;c<CLASSES;c++) se+=expf(lg[c]-mx);
        tot-=log2(fmaxf(expf(lg[target_val[i]]-mx)/se,1e-10f));
    }
    if(fire_rate) *fire_rate=(double)fires/val_len;
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

void audit_l2(AdamState* m) {
    double sum_ratio=0; long flips=0;
    for(int i=0;i<val_len;i++){
        const float* f=&features_val[(size_t)i*TOT_DIM];
        float see[CLASSES], d[CLASSES]; double ms=0, md=0;
        for(int c=0;c<CLASSES;c++){
            float sc=dot_simd(m->W[c],f,BASE_DIM);
            float lc=dot_simd(m->W[c]+BASE_DIM,f+BASE_DIM,L2_DIM);
            see[c]=m->B[c]+trigram_logits[ctx2_val[i]][ctx_val[i]][c]+sc; d[c]=lc; ms+=see[c]; md+=d[c];
        }
        ms/=CLASSES; md/=CLASSES;
        double vs=0,vd=0; for(int c=0;c<CLASSES;c++){ double a=see[c]-ms, b=d[c]-md; vs+=a*a; vd+=b*b; }
        double rs=sqrt(vs/CLASSES), rd=sqrt(vd/CLASSES);
        sum_ratio += (rs>1e-9)? rd/rs : 0.0;
        int as=0, af=0; float bs=-1e30f, bf=-1e30f;
        for(int c=0;c<CLASSES;c++){ if(see[c]>bs){bs=see[c];as=c;} float fu=see[c]+d[c]; if(fu>bf){bf=fu;af=c;} }
        if(as!=af) flips++;
    }
    printf("  AUDIT L2: mean L2/SEE logit-norm ratio = %.2f%%   argmax flips = %.2f%% of val\n",
           100.0*sum_ratio/val_len, 100.0*flips/val_len); fflush(stdout);
}

void save_weights(const char* prefix, const char* sfx, SiliconEntropyState* see, AdamState* m,
                  float mix, float l2_scale, float l2_cap, float ent_thr, double bpb) {
    char path[1024]; snprintf(path,sizeof(path),"%s%s",prefix,sfx);
    FILE* fw=fopen(path,"wb"); if(!fw){ fprintf(stderr,"Cannot write %s\n",path); return; }
    uint32_t hdr[4]={0x53454540,1,BASE_DIM,4};
    float hf[5]={0.75f,0.1f,0.5f,FEAT_CLAMP_DEFAULT,0.0f};
    uint32_t no=(uint32_t)see->n_oja;
    fwrite(hdr,sizeof(hdr),1,fw); fwrite(hf,sizeof(float),5,fw); fwrite(&no,4,1,fw);
    fwrite(see->W_oja,sizeof(float),(size_t)see->n_oja*43,fw);
    uint32_t l2d=L2_DIM, gt=(uint32_t)G_ENTROPY, eh=1u, ps=P_SEED;
    float al=0.99f, st=0.0f, et=ent_thr;
    fwrite(&l2d,4,1,fw); fwrite(&gt,4,1,fw); fwrite(&al,4,1,fw);
    fwrite(&st,4,1,fw); fwrite(&et,4,1,fw); fwrite(&eh,4,1,fw); fwrite(&ps,4,1,fw);
    float l2c=2.0f, nbd=1.0f; uint32_t cd=0, dl=(mix>0.0f)?1u:0u;
    float mx=mix, ls=l2_scale, cap=l2_cap;
    fwrite(&l2c,4,1,fw); fwrite(&nbd,4,1,fw); fwrite(&cd,4,1,fw); fwrite(&dl,4,1,fw);
    fwrite(&mx,4,1,fw); fwrite(&ls,4,1,fw); fwrite(&cap,4,1,fw);
    fwrite(trigram_logits,sizeof(float),CLASSES*CLASSES*CLASSES,fw);
    fwrite(feat_mean,sizeof(float),TOT_DIM,fw); fwrite(feat_std,sizeof(float),TOT_DIM,fw);
    fwrite(m->W,sizeof(float),CLASSES*TOT_DIM,fw); fwrite(m->B,sizeof(float),CLASSES,fw);
    fclose(fw);
    printf("  Saved %s  BPB=%.4f  (mix=%.2f scale=%.2f cap=%.3f)\n",path,bpb,mix,l2_scale,l2_cap);
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
    printf("AUDIT note: caps are INFERENCE-ONLY (no retrain); cap variants reuse base D1/H0 weights.\n"); fflush(stdout);

    features_train=malloc((size_t)train_len*TOT_DIM*sizeof(float));
    features_val  =malloc((size_t)val_len  *TOT_DIM*sizeof(float));
    target_train=malloc(train_len); ctx_train=malloc(train_len); ctx2_train=malloc(train_len);
    target_val  =malloc(val_len);   ctx_val  =malloc(val_len);   ctx2_val  =malloc(val_len);
    if(!features_train||!features_val){fprintf(stderr,"OOM features\n"); return 1;}
    l2_canon_train=malloc((size_t)train_len*L2_DIM*sizeof(float));
    l2_canon_val  =malloc((size_t)val_len  *L2_DIM*sizeof(float));
    if(!l2_canon_train||!l2_canon_val){fprintf(stderr,"OOM L2 canon\n"); return 1; }
    printf("RAM: features ~%.1f GB + L2 canon ~%.1f GB (single matrix)\n",
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
    const float ALPHA=0.99f;
    float lrs[]={0.0005f,0.0002f,0.0001f};

    // ===== Group mix=0.0 : train H0; derive H0_cap20 (inference-only) =====
    printf("\n###### EXTRACT GROUP mix=0.00 alpha=%.4f ######\n",ALPHA); fflush(stdout);
    extract_l2(&see,train_start,train_len,features_train,target_train,ctx_train,ctx2_train,G_ENTROPY,ALPHA,0,ent_p80,1,0.0f);
    extract_l2(&see,val_start,val_len,features_val,target_val,ctx_val,ctx2_val,G_ENTROPY,ALPHA,0,ent_p80,1,0.0f);
    normalize_and_clamp(FEAT_CLAMP_DEFAULT, 2.0f); save_l2_canon(); restore_l2_scaled(1.0f);
    AdamState* mH0=malloc(sizeof(AdamState)); memcpy(mH0,base,sizeof(AdamState));
    memset(mH0->mW,0,sizeof(mH0->mW)); memset(mH0->vW,0,sizeof(mH0->vW)); memset(mH0->mB,0,sizeof(mH0->mB)); memset(mH0->vB,0,sizeof(mH0->vB)); mH0->t=0;
    printf("\n====== H0 (mix0.00 scale1.00) ======\n"); fflush(stdout);
    for(int l=0;l<3;l++){printf("  LR %.4f\n",lrs[l]);fflush(stdout);train_lr(mH0,2,256,lrs[l]);}
    double bH0=eval_model(mH0); audit_l2(mH0);
    double fr; double bH0c20=eval_model_capped(mH0,0.020f,&fr);
    printf("  H0 base BPB=%.4f | H0+cap2.0%% BPB=%.4f (cap fired %.2f%% of val)\n",bH0,bH0c20,100.0*fr);
    save_weights(argv[2],"_H0.bin",      &see,mH0,0.0f,1.0f,0.0f,  ent_p80,bH0);
    save_weights(argv[2],"_H0_cap20.bin",&see,mH0,0.0f,1.0f,0.020f,ent_p80,bH0c20);
    free(mH0);

    // ===== Group mix=0.5 : train mix50 (scale1.0) and D1 (scale0.5); derive D1 caps =====
    printf("\n###### EXTRACT GROUP mix=0.50 alpha=%.4f ######\n",ALPHA); fflush(stdout);
    extract_l2(&see,train_start,train_len,features_train,target_train,ctx_train,ctx2_train,G_ENTROPY,ALPHA,0,ent_p80,1,0.5f);
    extract_l2(&see,val_start,val_len,features_val,target_val,ctx_val,ctx2_val,G_ENTROPY,ALPHA,0,ent_p80,1,0.5f);
    normalize_and_clamp(FEAT_CLAMP_DEFAULT, 2.0f); save_l2_canon();

    restore_l2_scaled(1.0f);
    AdamState* mM=malloc(sizeof(AdamState)); memcpy(mM,base,sizeof(AdamState));
    memset(mM->mW,0,sizeof(mM->mW)); memset(mM->vW,0,sizeof(mM->vW)); memset(mM->mB,0,sizeof(mM->mB)); memset(mM->vB,0,sizeof(mM->vB)); mM->t=0;
    printf("\n====== mix50 (mix0.50 scale1.00) ======\n"); fflush(stdout);
    for(int l=0;l<3;l++){printf("  LR %.4f\n",lrs[l]);fflush(stdout);train_lr(mM,2,256,lrs[l]);}
    double bM=eval_model(mM); audit_l2(mM);
    save_weights(argv[2],"_mix50.bin",&see,mM,0.5f,1.0f,0.0f,ent_p80,bM);
    free(mM);

    restore_l2_scaled(0.5f);
    AdamState* mD1=malloc(sizeof(AdamState)); memcpy(mD1,base,sizeof(AdamState));
    memset(mD1->mW,0,sizeof(mD1->mW)); memset(mD1->vW,0,sizeof(mD1->vW)); memset(mD1->mB,0,sizeof(mD1->mB)); memset(mD1->vB,0,sizeof(mD1->vB)); mD1->t=0;
    printf("\n====== D1 (mix0.50 scale0.50) ======\n"); fflush(stdout);
    for(int l=0;l<3;l++){printf("  LR %.4f\n",lrs[l]);fflush(stdout);train_lr(mD1,2,256,lrs[l]);}
    double bD1=eval_model(mD1); audit_l2(mD1);
    double f20,f15,f10;
    double bD1c20=eval_model_capped(mD1,0.020f,&f20);
    double bD1c15=eval_model_capped(mD1,0.015f,&f15);
    double bD1c10=eval_model_capped(mD1,0.010f,&f10);
    printf("  D1 base BPB=%.4f | cap2.0%%=%.4f(fire %.2f%%) | cap1.5%%=%.4f(fire %.2f%%) | cap1.0%%=%.4f(fire %.2f%%)\n",
           bD1,bD1c20,100.0*f20,bD1c15,100.0*f15,bD1c10,100.0*f10);
    save_weights(argv[2],"_D1.bin",      &see,mD1,0.5f,0.5f,0.0f,  ent_p80,bD1);
    save_weights(argv[2],"_D1_cap20.bin",&see,mD1,0.5f,0.5f,0.020f,ent_p80,bD1c20);
    save_weights(argv[2],"_D1_cap15.bin",&see,mD1,0.5f,0.5f,0.015f,ent_p80,bD1c15);
    save_weights(argv[2],"_D1_cap10.bin",&see,mD1,0.5f,0.5f,0.010f,ent_p80,bD1c10);
    free(mD1);

    printf("\n\n====== Phase 44.E - Readout L2 Logit-Contribution Caps ======\n");
    printf("  C2.A reference: %.4f  (promote: BPB<=%.4f + topBi<=8 + altLp<=2 + nameWst<=20 + runWst<=5)\n",C2A_BPB,C2A_BPB-0.005);
    printf("  Bases: H0=%.4f  mix50=%.4f  D1=%.4f\n",bH0,bM,bD1);
    printf("  Caps (inference-only): H0+2.0%%=%.4f  D1+2.0%%=%.4f  D1+1.5%%=%.4f  D1+1.0%%=%.4f\n",bH0c20,bD1c20,bD1c15,bD1c10);
    printf("  Criterion: does the T=0.65 pass extend to T=0.55 within ~0.003-0.005 BPB? [word-gate decides]\n");
    free(base); free(data);
    return 0;
}
