// Phase 43.A2 — Fast-alpha refinement: grid over {0.3, 0.5, 0.6}
// mid=0.9 slow=0.99 fixed. Confirms whether ms_f0.5 is optimal or 0.3 is better.
//
// Build (from repo root):
//   gcc -O3 -march=native -mavx2 -mfma \
//       benchmarks/phase38-42/phase43a2_fast_grid.c \
//       src/silicon_entropy.c src/silicon_v0.c \
//       -o bin/phase43a2_fast_grid.exe -lm -I .
// Run:
//   bin/phase43a2_fast_grid.exe <dataset> <weights_prefix>

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

#define CLASSES  256
#define FEAT_DIM SEE_FEATURE_DIM

typedef struct { float W[CLASSES][FEAT_DIM]; float B[CLASSES];
                 float mW[CLASSES][FEAT_DIM]; float vW[CLASSES][FEAT_DIM];
                 float mB[CLASSES]; float vB[CLASSES]; int t; } AdamState;

static inline float dot_simd(const float* w, const float* f, int n) {
    __m256 s = _mm256_setzero_ps(); int i = 0;
    for (; i <= n-8; i+=8) s = _mm256_fmadd_ps(_mm256_loadu_ps(&w[i]), _mm256_loadu_ps(&f[i]), s);
    float o[8]; _mm256_storeu_ps(o,s);
    float r=o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7];
    for (; i<n; i++) r+=w[i]*f[i]; return r;
}
static inline void grad_simd(float* gW, const float* f, float e, int n) {
    __m256 ev = _mm256_set1_ps(e); int i=0;
    for (; i<=n-8; i+=8) _mm256_storeu_ps(&gW[i],_mm256_fmadd_ps(ev,_mm256_loadu_ps(&f[i]),_mm256_loadu_ps(&gW[i])));
    for (; i<n; i++) gW[i]+=e*f[i];
}

static uint8_t *data; static long data_size;
static int train_start, train_len, val_start, val_len;
static float *features_train, *features_val;
static uint8_t *target_train, *target_val, *ctx_train, *ctx_val, *ctx2_train, *ctx2_val;
static float bigram_logits[CLASSES][CLASSES];
static float (*trigram_logits)[CLASSES][CLASSES];
static float feat_mean[FEAT_DIM], feat_std[FEAT_DIM];

void extract(SiliconEntropyState* see, int start, int len, float* of, uint8_t* ot, uint8_t* oc, uint8_t* oc2) {
    see_reset(see);
    for (int i=0;i<=start+1;i++) see_observe(see,data[i]);
    for (int i=0;i<len;i++) { int g=start+i;
        ot[i]=data[g+2]; oc[i]=data[g+1]; oc2[i]=data[g];
        see_extract(see,&of[(size_t)i*FEAT_DIM]); see_observe(see,ot[i]); }
}
void compute_ngrams(void) {
    double bi[CLASSES][CLASSES]={0}, bt[CLASSES]={0};
    double (*tc)[CLASSES][CLASSES]=malloc(CLASSES*CLASSES*CLASSES*sizeof(double));
    double (*tt)[CLASSES]=malloc(CLASSES*CLASSES*sizeof(double));
    memset(tc,0,CLASSES*CLASSES*CLASSES*sizeof(double)); memset(tt,0,CLASSES*CLASSES*sizeof(double));
    trigram_logits=malloc(CLASSES*CLASSES*CLASSES*sizeof(float));
    for (int i=0;i<train_len;i++) {
        uint8_t t=target_train[i],c1=ctx_train[i],c2=ctx2_train[i];
        bi[c1][t]++; bt[c1]++; tc[c2][c1][t]++; tt[c2][c1]++;
    }
    for (int i=0;i<CLASSES;i++) for (int j=0;j<CLASSES;j++) {
        bigram_logits[i][j]=(float)log((bi[i][j]+0.1)/(bt[i]+CLASSES*0.1));
        for (int k=0;k<CLASSES;k++) trigram_logits[i][j][k]=(float)log((tc[i][j][k]+0.1)/(tt[i][j]+CLASSES*0.1));
    }
    free(tc); free(tt);
}
void normalize(void) {
    for (int f=0;f<FEAT_DIM;f++) {
        double m=0; for (int i=0;i<train_len;i++) m+=features_train[(size_t)i*FEAT_DIM+f]; m/=train_len;
        double v=0; for (int i=0;i<train_len;i++){double d=features_train[(size_t)i*FEAT_DIM+f]-m;v+=d*d;} v=sqrt(v/train_len)+1e-8;
        feat_mean[f]=(float)m; feat_std[f]=(float)v;
        for (int i=0;i<train_len;i++) features_train[(size_t)i*FEAT_DIM+f]=(features_train[(size_t)i*FEAT_DIM+f]-(float)m)/(float)v;
        for (int i=0;i<val_len;i++)   features_val[(size_t)i*FEAT_DIM+f]=(features_val[(size_t)i*FEAT_DIM+f]-(float)m)/(float)v;
    }
}
double eval(AdamState* m) {
    double tot=0;
    for (int i=0;i<val_len;i++) {
        float lg[CLASSES],mx=-1e9f;
        for (int c=0;c<CLASSES;c++){lg[c]=m->B[c]+trigram_logits[ctx2_val[i]][ctx_val[i]][c]+dot_simd(m->W[c],&features_val[(size_t)i*FEAT_DIM],FEAT_DIM);if(lg[c]>mx)mx=lg[c];}
        float se=0; for (int c=0;c<CLASSES;c++) se+=expf(lg[c]-mx);
        tot-=log2(fmaxf(expf(lg[target_val[i]]-mx)/se,1e-10f));
    }
    return tot/val_len;
}
void train_lr(AdamState* m, int eps, int bs, float lr) {
    float *gW=malloc(CLASSES*FEAT_DIM*sizeof(float)), gB[CLASSES], lg[CLASSES], pr[CLASSES];
    AdamState *best=malloc(sizeof(AdamState)); double bb=eval(m); memcpy(best,m,sizeof(AdamState));
    printf("  ep0 %.4f\n",bb);
    for (int ep=0;ep<eps;ep++) {
        memset(gW,0,CLASSES*FEAT_DIM*sizeof(float)); memset(gB,0,sizeof(gB));
        for (int i=0;i<train_len;i++) {
            float mx=-1e9f;
            for (int c=0;c<CLASSES;c++){lg[c]=m->B[c]+trigram_logits[ctx2_train[i]][ctx_train[i]][c]+dot_simd(m->W[c],&features_train[(size_t)i*FEAT_DIM],FEAT_DIM);if(lg[c]>mx)mx=lg[c];}
            float se=0; for(int c=0;c<CLASSES;c++){pr[c]=expf(lg[c]-mx);se+=pr[c];} for(int c=0;c<CLASSES;c++) pr[c]/=se;
            for (int c=0;c<CLASSES;c++){float e=pr[c]-(c==target_train[i]?1.f:0.f); gB[c]+=e/bs; grad_simd(&gW[c*FEAT_DIM],&features_train[(size_t)i*FEAT_DIM],e/bs,FEAT_DIM);}
            if ((i+1)%bs==0||(i+1)==train_len) {
                m->t++; float lt=lr*sqrtf(1.f-powf(.999f,m->t))/(1.f-powf(.9f,m->t));
                for (int c=0;c<CLASSES;c++){m->mB[c]=.9f*m->mB[c]+.1f*gB[c];m->vB[c]=.999f*m->vB[c]+.001f*gB[c]*gB[c];m->B[c]-=lt*(m->mB[c]/(sqrtf(m->vB[c])+1e-8f)+1e-4f*m->B[c]);
                for(int f=0;f<FEAT_DIM;f++){float g=gW[c*FEAT_DIM+f];m->mW[c][f]=.9f*m->mW[c][f]+.1f*g;m->vW[c][f]=.999f*m->vW[c][f]+.001f*g*g;m->W[c][f]-=lt*(m->mW[c][f]/(sqrtf(m->vW[c][f])+1e-8f)+1e-4f*m->W[c][f]);}}
                memset(gW,0,CLASSES*FEAT_DIM*sizeof(float)); memset(gB,0,sizeof(gB));
            }
        }
        double b=eval(m); printf("  ep%d %.4f\n",ep+1,b); fflush(stdout);
        if(b<bb){bb=b;memcpy(best,m,sizeof(AdamState));}
    }
    memcpy(m,best,sizeof(AdamState)); free(best); free(gW);
}

int main(int argc, char** argv) {
    if (argc<3){printf("Usage: %s <dataset> <weights_prefix>\n",argv[0]);return 1;}
    FILE* f=fopen(argv[1],"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",argv[1]);return 1;}
    fseek(f,0,SEEK_END); data_size=ftell(f); fseek(f,0,SEEK_SET);
    data=malloc(data_size); fread(data,1,data_size,f); fclose(f);
    train_start=0; train_len=(int)(((long long)data_size*50)/100);
    val_start=(int)(((long long)data_size*50)/100); val_len=(int)(((long long)data_size*25)/100);
    printf("Dataset %ld  train=%d  val=%d\n",data_size,train_len,val_len); fflush(stdout);

    features_train=malloc((size_t)train_len*FEAT_DIM*sizeof(float));
    features_val=malloc((size_t)val_len*FEAT_DIM*sizeof(float));
    target_train=malloc(train_len); ctx_train=malloc(train_len); ctx2_train=malloc(train_len);
    target_val=malloc(val_len);   ctx_val=malloc(val_len);   ctx2_val=malloc(val_len);

    // Fast-alpha grid: fast in {0.3, 0.5, 0.6}, mid=0.9, slow=0.99
    float fast_vals[] = {0.3f, 0.5f, 0.6f};
    const char* suffixes[] = {"_f03.bin","_f05.bin","_f06.bin"};
    int n = 3; double results[3];

    printf("\nComputing N-grams...\n"); fflush(stdout);
    SiliconEntropyState see;
    see_init(&see,42,4,0.75f); see.multiscale_mode=1; see.alpha_fast=0.5f; see.alpha_mid=0.9f; see.alpha_slow=0.99f;
    extract(&see,train_start,train_len,features_train,target_train,ctx_train,ctx2_train);
    compute_ngrams();

    for (int ci=0;ci<n;ci++) {
        printf("\n====== fast=%.1f mid=0.9 slow=0.99 ======\n",fast_vals[ci]); fflush(stdout);
        see_init(&see,42,4,0.75f); see.multiscale_mode=1;
        see.alpha_fast=fast_vals[ci]; see.alpha_mid=0.9f; see.alpha_slow=0.99f;

        extract(&see,train_start,train_len,features_train,target_train,ctx_train,ctx2_train);
        extract(&see,val_start,val_len,features_val,target_val,ctx_val,ctx2_val);
        normalize();

        AdamState* m=calloc(1,sizeof(AdamState));
        float lrs[]={0.003f,0.001f,0.0003f};
        for (int l=0;l<3;l++){printf("  LR %.4f\n",lrs[l]);fflush(stdout);train_lr(m,5,256,lrs[l]);}
        results[ci]=eval(m);

        char path[1024]; snprintf(path,sizeof(path),"%s%s",argv[2],suffixes[ci]);
        FILE* fw=fopen(path,"wb");
        if (fw) {
            uint32_t hdr[4]={0x53454535,1,FEAT_DIM,4}; float hf[3]={0.75f,0.1f,fast_vals[ci]};
            fwrite(hdr,sizeof(hdr),1,fw); fwrite(hf,sizeof(hf),1,fw);
            fwrite(trigram_logits,sizeof(float),CLASSES*CLASSES*CLASSES,fw);
            fwrite(feat_mean,sizeof(float),FEAT_DIM,fw); fwrite(feat_std,sizeof(float),FEAT_DIM,fw);
            fwrite(m->W,sizeof(float),CLASSES*FEAT_DIM,fw); fwrite(m->B,sizeof(float),CLASSES,fw);
            fclose(fw); printf("  Saved %s\n",path);
        }
        free(m);
    }

    printf("\n\n====== Phase 43.A2 — Fast-Alpha Grid ======\n");
    printf("  baseline (42.0):  2.3197 BPB\n");
    printf("  43.A ms_f0.5:     2.2757 BPB\n\n");
    printf("  %-8s  Val BPB   Delta vs f0.5\n","fast");
    for (int ci=0;ci<n;ci++)
        printf("  f=%.1f     %.4f    %+.4f\n",fast_vals[ci],results[ci],results[ci]-results[1]);

    int best=0; for(int ci=1;ci<n;ci++) if(results[ci]<results[best]) best=ci;
    printf("\nBest fast: %.1f (%.4f BPB)\n",fast_vals[best],results[best]);
    if (results[0]<results[1]-0.002f)
        printf("SIGNAL: fast=0.3 wins -> even shorter timescale helps -> consider fast=0.2\n");
    else if (results[2]<results[1]-0.002f)
        printf("SIGNAL: fast=0.6 wins -> f0.5 was too reactive\n");
    else
        printf("SIGNAL: f0.5 confirmed optimal in this range\n");
    return 0;
}
