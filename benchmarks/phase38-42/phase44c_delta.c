// Phase 44.C - Entropy-high Delta Calibration
//
// 44.B answered: delta-write is the right axis. D_delta 2.2214 BPB (-0.0379 vs
// C2.A), B_delta 2.2276 (-0.0317) - by far the biggest gains in the phase. The
// silicon does NOT want to remember "what I am now" (absolute snapshot); it wants
// "what changed when something happened" (boundary delta). But generation still
// fails: D_delta T=0.65 topBi 13 / altLp 3 (FAIL); T=0.55 worse. Delta is the
// right direction but uncalibrated / too strong.
//
// 44.C calibrates the delta on the two axes that SURVIVE per-dim z-normalization:
//   alpha (EMA pole -> temporal structure) and mix (write direction).
//   NOTE: gain (scalar on the write) is a proven NO-OP here: the L2->readout path
//   is entirely linear and L2 is z-normalized per-dim, so any scalar g cancels
//   (z = (g*x - g*mu)/(g*sigma) = (x-mu)/sigma). gain configs are skipped; the
//   math is the proof. Verify gain-invariance analytically noted; gain skipped.
//
// Write rule (single knob `mix` in [0,1]):
//   src = SEE_t - mix * SEE_prev_boundary        (mix0 = absolute, mix1 = full delta)
//   L2  = alpha*L2 + (1-alpha) * (1/sqrt(192)) * P @ src     (on boundary, gate fires)
//
// Base: C2.A (0x53454539). Gate = entropy-high only (44.D_hi survivor). Readout
//   linear over [SEE 192 | L2 64] = 256D, clamp on (base 2.0, L2 2.0). Fixed P.
//
// 8 configs (entropy-high, l2_clamp=2.0, no decay, no cooldown):
//   D_H0     alpha0.99   mix0.00   (absolute control = reproduce 44.D_hi ~2.2526)
//   D_delta  alpha0.99   mix1.00   (full delta control = reproduce 44.B D_delta ~2.2214)
//   D_a995   alpha0.995  mix1.00
//   D_a9975  alpha0.9975 mix1.00
//   D_a999   alpha0.999  mix1.00
//   D_mix25  alpha0.99   mix0.25
//   D_mix50  alpha0.99   mix0.50
//   D_mix75  alpha0.99   mix0.75
//
// Promotion gate (per user): val BPB <= 2.2543 (>=0.005 vs C2.A) AND topBi<=8 AND
//   altLp<=2 AND nameWst<=20 AND runWst<=5 (self in [0.8,2.0]).
//
// Weight format 0x5345453E = 0x5345453D layout + f mix:
//   ... u32 p_seed + f l2_clamp + f nb_decay + u32 cooldown + u32 delta_flag
//       + f mix + trigram + mean[256] + std[256] + W[256*256] + B[256]
//
// Build (from repo root):
//   gcc -O3 -march=native -mavx2 -mfma \
//       benchmarks/phase38-42/phase44c_delta.c \
//       src/silicon_entropy.c src/silicon_v0.c \
//       -o bin/phase44c_delta.exe -lm -I .
// Run:
//   bin/phase44c_delta.exe <dataset> <weights_prefix> <c2a_weights>

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

// Extract [SEE 192 | L2 64] with gated EMA + delta-mix write.
//   mix in [0,1]: src = SEE_t - mix * SEE_prev_boundary  (mix0=absolute, mix1=delta)
//   nb_decay/cooldown kept for layout parity (44.C uses 1.0 / 0).
void extract_l2(SiliconEntropyState* see, int start, int len, float* of,
                uint8_t* ot, uint8_t* oc, uint8_t* oc2,
                int gate, float alpha, float surp_thr, float ent_thr, int ent_high,
                float nb_decay, int cooldown, float mix) {
    see_reset(see);
    for (int i=0;i<=start+1;i++) see_observe(see,data[i]);
    float L2[L2_DIM]; memset(L2,0,sizeof(L2));
    float prev_bound[BASE_DIM]; memset(prev_bound,0,sizeof(prev_bound));
    int cd_ctr=0;
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
        int gate_fires = gate && eval_gate(gate, ot[i], oc[i], oc2[i], surp_thr, ent_thr, ent_high);
        int do_update = gate_fires && (cd_ctr==0);
        if (do_update) {
            see_extract(see, fa);
            const float* src=fa;
            if (mix>0.0f){ for(int k=0;k<BASE_DIM;k++) blend[k]=fa[k]-mix*prev_bound[k]; src=blend; memcpy(prev_bound,fa,sizeof(fa)); }
            for (int j=0;j<L2_DIM;j++){ float p=0; const float* pj=Pmat[j]; for(int k=0;k<BASE_DIM;k++) p+=pj[k]*src[k]; p*=scale; L2[j]=alpha*L2[j]+(1.0f-alpha)*p; }
            cd_ctr=cooldown;
        } else {
            if (!gate_fires && nb_decay<1.0f) for(int j=0;j<L2_DIM;j++) L2[j]*=nb_decay;
            if (cd_ctr>0) cd_ctr--;
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
            float mx=-1e9f;
            for(int c=0;c<CLASSES;c++){lg[c]=m->B[c]+trigram_logits[ctx2_train[i]][ctx_train[i]][c]+dot_simd(m->W[c],&features_train[(size_t)i*TOT_DIM],TOT_DIM);if(lg[c]>mx)mx=lg[c];}
            float se=0; for(int c=0;c<CLASSES;c++){pr[c]=expf(lg[c]-mx);se+=pr[c];} for(int c=0;c<CLASSES;c++) pr[c]/=se;
            for(int c=0;c<CLASSES;c++){float e=pr[c]-(c==target_train[i]?1.f:0.f);gB[c]+=e/bs;grad_simd(&gW[c*TOT_DIM],&features_train[(size_t)i*TOT_DIM],e/bs,TOT_DIM);}
            if((i+1)%bs==0||(i+1)==train_len){
                m->t++; float lt=lr*sqrtf(1.f-powf(.999f,m->t))/(1.f-powf(.9f,m->t));
                for(int c=0;c<CLASSES;c++){m->mB[c]=.9f*m->mB[c]+.1f*gB[c];m->vB[c]=.999f*m->vB[c]+.001f*gB[c]*gB[c];m->B[c]-=lt*(m->mB[c]/(sqrtf(m->vB[c])+1e-8f)+1e-4f*m->B[c]);
                for(int f=0;f<TOT_DIM;f++){float g=gW[c*TOT_DIM+f];m->mW[c][f]=.9f*m->mW[c][f]+.1f*g;m->vW[c][f]=.999f*m->vW[c][f]+.001f*g*g;m->W[c][f]-=lt*(m->mW[c][f]/(sqrtf(m->vW[c][f])+1e-8f)+1e-4f*m->W[c][f]);}}
                memset(gW,0,CLASSES*TOT_DIM*sizeof(float)); memset(gB,0,sizeof(gB));}}
        double b=eval_model(m); printf("    ep%d %.4f\n",ep+1,b); fflush(stdout);
        if(b<bb){bb=b;memcpy(best,m,sizeof(AdamState));}}
    memcpy(m,best,sizeof(AdamState)); free(best); free(gW);
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
    printf("AUDIT: gain-invariance analytically noted; gain configs skipped (scalar on write cancels under per-dim z-normalization).\n"); fflush(stdout);

    features_train=malloc((size_t)train_len*TOT_DIM*sizeof(float));
    features_val  =malloc((size_t)val_len  *TOT_DIM*sizeof(float));
    target_train=malloc(train_len); ctx_train=malloc(train_len); ctx2_train=malloc(train_len);
    target_val  =malloc(val_len);   ctx_val  =malloc(val_len);   ctx2_val  =malloc(val_len);
    if(!features_train||!features_val){fprintf(stderr,"OOM features (~%.1f GB)\n",((double)train_len+val_len)*TOT_DIM*4/1e9); return 1;}

    SiliconEntropyState see;
    see_init(&see,42,4,0.75f);
    see.multiscale_mode=1; see.alpha_fast=0.5f; see.alpha_mid=0.9f; see.alpha_slow=0.99f;
    see.eta_oja=0.0f; see.plastic_blend=1.0f;

    AdamState* base=calloc(1,sizeof(AdamState));
    if(!load_c2a(argv[3],&see,base->W,base->B)){fprintf(stderr,"Failed to load C2.A %s\n",argv[3]);return 1;}
    printf("Loaded C2.A (n_oja=%d) + warm-start readout\n",see.n_oja); fflush(stdout);
    gen_projection(P_SEED);

    printf("Pass 0: N-grams + entropy table...\n"); fflush(stdout);
    extract_l2(&see,train_start,train_len,features_train,target_train,ctx_train,ctx2_train,G_NONE,0,0,0,0,1.0f,0,0.0f);
    compute_ngrams_and_entropy();
    float surp_p90,surp_p80,ent_p80; compute_thresholds(&surp_p90,&surp_p80,&ent_p80);
    printf("Thresholds: surp10=%.3f surp20=%.3f ent_high(top20)=%.3f\n",surp_p90,surp_p80,ent_p80); fflush(stdout);

    // Entropy-high gate fixed (44.D_hi survivor). Two axes: alpha (temporal), mix (direction).
    struct { float alpha; float mix; const char* sfx; const char* name; } cfgs[] = {
        {0.9900f, 0.00f, "_D_H0.bin",     "44C H0 absolute   (a0.99 mix0.00)"},
        {0.9900f, 1.00f, "_D_delta.bin",  "44C delta full    (a0.99 mix1.00)"},
        {0.9950f, 1.00f, "_D_a995.bin",   "44C delta a0.995  (mix1.00)"},
        {0.9975f, 1.00f, "_D_a9975.bin",  "44C delta a0.9975 (mix1.00)"},
        {0.9990f, 1.00f, "_D_a999.bin",   "44C delta a0.999  (mix1.00)"},
        {0.9900f, 0.25f, "_D_mix25.bin",  "44C mix0.25       (a0.99)"},
        {0.9900f, 0.50f, "_D_mix50.bin",  "44C mix0.50       (a0.99)"},
        {0.9900f, 0.75f, "_D_mix75.bin",  "44C mix0.75       (a0.99)"},
    };
    int n_cfg=(int)(sizeof(cfgs)/sizeof(cfgs[0]));
    double results[32];

    for (int ci=0; ci<n_cfg; ci++) {
        printf("\n====== %s  (alpha=%.4f mix=%.2f) ======\n",cfgs[ci].name,cfgs[ci].alpha,cfgs[ci].mix); fflush(stdout);
        extract_l2(&see,train_start,train_len,features_train,target_train,ctx_train,ctx2_train,
                   G_ENTROPY,cfgs[ci].alpha,0,ent_p80,1, 1.0f,0,cfgs[ci].mix);
        extract_l2(&see,val_start,val_len,features_val,target_val,ctx_val,ctx2_val,
                   G_ENTROPY,cfgs[ci].alpha,0,ent_p80,1, 1.0f,0,cfgs[ci].mix);
        normalize_and_clamp(FEAT_CLAMP_DEFAULT, 2.0f);

        AdamState* m=malloc(sizeof(AdamState)); memcpy(m,base,sizeof(AdamState));
        memset(m->mW,0,sizeof(m->mW)); memset(m->vW,0,sizeof(m->vW));
        memset(m->mB,0,sizeof(m->mB)); memset(m->vB,0,sizeof(m->vB)); m->t=0;
        float lrs[]={0.0005f,0.0002f,0.0001f};
        for(int l=0;l<3;l++){printf("  LR %.4f\n",lrs[l]);fflush(stdout);train_lr(m,2,256,lrs[l]);}
        results[ci]=eval_model(m);

        char path[1024]; snprintf(path,sizeof(path),"%s%s",argv[2],cfgs[ci].sfx);
        FILE* fw=fopen(path,"wb");
        if(fw){
            uint32_t hdr[4]={0x5345453E,1,BASE_DIM,4};
            float hf[5]={0.75f,0.1f,0.5f,FEAT_CLAMP_DEFAULT,0.0f};
            uint32_t no=(uint32_t)see.n_oja;
            fwrite(hdr,sizeof(hdr),1,fw); fwrite(hf,sizeof(float),5,fw); fwrite(&no,4,1,fw);
            fwrite(see.W_oja,sizeof(float),(size_t)see.n_oja*43,fw);
            uint32_t l2d=L2_DIM, gt=(uint32_t)G_ENTROPY, eh=1u, ps=P_SEED;
            float al=cfgs[ci].alpha, st=0.0f, et=ent_p80;
            fwrite(&l2d,4,1,fw); fwrite(&gt,4,1,fw); fwrite(&al,4,1,fw);
            fwrite(&st,4,1,fw); fwrite(&et,4,1,fw); fwrite(&eh,4,1,fw); fwrite(&ps,4,1,fw);
            // homeostasis block (44.C: clamp 2.0, no decay, no cooldown) + delta-flag + mix
            float l2c=2.0f, nbd=1.0f; uint32_t cd=0, dl=(cfgs[ci].mix>0.0f)?1u:0u;
            float mx=cfgs[ci].mix;
            fwrite(&l2c,4,1,fw); fwrite(&nbd,4,1,fw); fwrite(&cd,4,1,fw); fwrite(&dl,4,1,fw);
            fwrite(&mx,4,1,fw);
            fwrite(trigram_logits,sizeof(float),CLASSES*CLASSES*CLASSES,fw);
            fwrite(feat_mean,sizeof(float),TOT_DIM,fw); fwrite(feat_std,sizeof(float),TOT_DIM,fw);
            fwrite(m->W,sizeof(float),CLASSES*TOT_DIM,fw); fwrite(m->B,sizeof(float),CLASSES,fw);
            fclose(fw);
            printf("  Saved %s  BPB=%.4f\n",path,results[ci]);
        }
        free(m);
    }

    printf("\n\n====== Phase 44.C - Entropy-high Delta Calibration ======\n");
    printf("  C2.A reference: %.4f  (promote: BPB<=%.4f + topBi<=8 + altLp<=2 + nameWst<=20 + runWst<=5)\n",C2A_BPB,C2A_BPB-0.005);
    printf("  %-36s  Val BPB   Delta vs C2.A\n","config");
    for(int ci=0;ci<n_cfg;ci++) printf("  %-36s  %.4f    %+.4f\n",cfgs[ci].name,results[ci],results[ci]-C2A_BPB);
    int best=0; for(int ci=1;ci<n_cfg;ci++) if(results[ci]<results[best]) best=ci;
    printf("\n  Best BPB: %s (%.4f, %+.4f vs C2.A) [word-gate decides promotion]\n",cfgs[best].name,results[best],results[best]-C2A_BPB);
    free(base); free(data);
    return 0;
}
