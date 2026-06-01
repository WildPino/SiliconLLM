// Phase 43.C — Oja Plasticity Tribunal
// Base: SEE-V1S (feat_clamp=2.0, phase43h_c20.bin)
// Adds: Oja plastic cells (first SEE_N_OJA=13 of L1 fast band)
//
// Protocol:
//   Pass 1: stream with eta_oja active -> W_oja converges (unsupervised)
//   Freeze W_oja, re-extract features
//   Batch train W/B on Oja-modified + clamped features (same Adam as 43.H)
//
// Grid: eta_oja in {1e-3, 1e-4}
//
// Weight format 0x53454538:
//   {magic,ver,fdim,chunk} + float[5]{decay,0.1,alpha_fast,feat_clamp,eta_oja}
//   + W_oja[SEE_N_OJA*43] + trigram + feat_mean + feat_std + W + B
//
// Build (from repo root):
//   gcc -O3 -march=native -mavx2 -mfma \
//       benchmarks/phase38-42/phase43c_oja.c \
//       src/silicon_entropy.c src/silicon_v0.c \
//       -o bin/phase43c_oja.exe -lm -I .
// Run:
//   bin/phase43c_oja.exe <dataset> <weights_prefix> <base_weights_c20>

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
#define FEAT_CLAMP_DEFAULT 2.0f

typedef struct {
    float W[CLASSES][FEAT_DIM]; float B[CLASSES];
    float mW[CLASSES][FEAT_DIM]; float vW[CLASSES][FEAT_DIM];
    float mB[CLASSES]; float vB[CLASSES]; int t;
} AdamState;

static inline float dot_simd(const float* w, const float* f, int n) {
    __m256 s = _mm256_setzero_ps(); int i = 0;
    for (; i <= n-8; i+=8) s = _mm256_fmadd_ps(_mm256_loadu_ps(&w[i]),_mm256_loadu_ps(&f[i]),s);
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
static float (*trigram_logits)[CLASSES][CLASSES];
static float feat_mean[FEAT_DIM], feat_std[FEAT_DIM];

// Extract features with Oja running (Pass 1: Oja adapts; eta_oja > 0)
// or frozen (Pass 2: eta_oja = 0, just apply W_oja)
void extract_with_oja(SiliconEntropyState* see, int start, int len,
                      float* of, uint8_t* ot, uint8_t* oc, uint8_t* oc2) {
    see_reset(see);
    for (int i=0;i<=start+1;i++) see_observe(see,data[i]);
    for (int i=0;i<len;i++) {
        int g=start+i;
        ot[i]=data[g+2]; oc[i]=data[g+1]; oc2[i]=data[g];
        see_extract(see,&of[(size_t)i*FEAT_DIM]);
        see_observe(see,ot[i]);
    }
}

void compute_ngrams(void) {
    double (*tc)[CLASSES][CLASSES]=malloc(CLASSES*CLASSES*CLASSES*sizeof(double));
    double (*tt)[CLASSES]=malloc(CLASSES*CLASSES*sizeof(double));
    memset(tc,0,CLASSES*CLASSES*CLASSES*sizeof(double));
    memset(tt,0,CLASSES*CLASSES*sizeof(double));
    trigram_logits=malloc(CLASSES*CLASSES*CLASSES*sizeof(float));
    for (int i=0;i<train_len;i++) {
        uint8_t t=target_train[i],c1=ctx_train[i],c2=ctx2_train[i];
        tc[c2][c1][t]++; tt[c2][c1]++;
    }
    for (int i=0;i<CLASSES;i++) for (int j=0;j<CLASSES;j++)
        for (int k=0;k<CLASSES;k++)
            trigram_logits[i][j][k]=(float)log((tc[i][j][k]+0.1)/(tt[i][j]+CLASSES*0.1));
    free(tc); free(tt);
}

void normalize_and_clamp(float clamp_val) {
    for (int f=0;f<FEAT_DIM;f++) {
        double m=0;
        for (int i=0;i<train_len;i++) m+=features_train[(size_t)i*FEAT_DIM+f];
        m/=train_len;
        double v=0;
        for (int i=0;i<train_len;i++){double d=features_train[(size_t)i*FEAT_DIM+f]-m;v+=d*d;}
        v=sqrt(v/train_len)+1e-8;
        feat_mean[f]=(float)m; feat_std[f]=(float)v;
        for (int i=0;i<train_len;i++){
            float x=(features_train[(size_t)i*FEAT_DIM+f]-(float)m)/(float)v;
            if(x> clamp_val) x= clamp_val; if(x<-clamp_val) x=-clamp_val;
            features_train[(size_t)i*FEAT_DIM+f]=x;
        }
        for (int i=0;i<val_len;i++){
            float x=(features_val[(size_t)i*FEAT_DIM+f]-(float)m)/(float)v;
            if(x> clamp_val) x= clamp_val; if(x<-clamp_val) x=-clamp_val;
            features_val[(size_t)i*FEAT_DIM+f]=x;
        }
    }
}

double eval_bpb(AdamState* m) {
    double tot=0;
    for (int i=0;i<val_len;i++) {
        float lg[CLASSES],mx=-1e9f;
        for (int c=0;c<CLASSES;c++){
            lg[c]=m->B[c]+trigram_logits[ctx2_val[i]][ctx_val[i]][c]
                  +dot_simd(m->W[c],&features_val[(size_t)i*FEAT_DIM],FEAT_DIM);
            if(lg[c]>mx) mx=lg[c];
        }
        float se=0; for(int c=0;c<CLASSES;c++) se+=expf(lg[c]-mx);
        tot-=log2(fmaxf(expf(lg[target_val[i]]-mx)/se,1e-10f));
    }
    return tot/val_len;
}

void train_lr(AdamState* m, int eps, int bs, float lr) {
    float *gW=malloc(CLASSES*FEAT_DIM*sizeof(float)), gB[CLASSES], lg[CLASSES], pr[CLASSES];
    AdamState *best=malloc(sizeof(AdamState));
    double bb=eval_bpb(m); memcpy(best,m,sizeof(AdamState));
    printf("    ep0 %.4f\n",bb); fflush(stdout);
    for (int ep=0;ep<eps;ep++) {
        memset(gW,0,CLASSES*FEAT_DIM*sizeof(float)); memset(gB,0,sizeof(gB));
        for (int i=0;i<train_len;i++) {
            float mx=-1e9f;
            for(int c=0;c<CLASSES;c++){
                lg[c]=m->B[c]+trigram_logits[ctx2_train[i]][ctx_train[i]][c]
                      +dot_simd(m->W[c],&features_train[(size_t)i*FEAT_DIM],FEAT_DIM);
                if(lg[c]>mx) mx=lg[c];
            }
            float se=0;
            for(int c=0;c<CLASSES;c++){pr[c]=expf(lg[c]-mx);se+=pr[c];}
            for(int c=0;c<CLASSES;c++) pr[c]/=se;
            for(int c=0;c<CLASSES;c++){
                float e=pr[c]-(c==target_train[i]?1.f:0.f);
                gB[c]+=e/bs;
                grad_simd(&gW[c*FEAT_DIM],&features_train[(size_t)i*FEAT_DIM],e/bs,FEAT_DIM);
            }
            if ((i+1)%bs==0||(i+1)==train_len) {
                m->t++;
                float lt=lr*sqrtf(1.f-powf(.999f,m->t))/(1.f-powf(.9f,m->t));
                for(int c=0;c<CLASSES;c++){
                    m->mB[c]=.9f*m->mB[c]+.1f*gB[c];
                    m->vB[c]=.999f*m->vB[c]+.001f*gB[c]*gB[c];
                    m->B[c]-=lt*(m->mB[c]/(sqrtf(m->vB[c])+1e-8f)+1e-4f*m->B[c]);
                    for(int f=0;f<FEAT_DIM;f++){
                        float g=gW[c*FEAT_DIM+f];
                        m->mW[c][f]=.9f*m->mW[c][f]+.1f*g;
                        m->vW[c][f]=.999f*m->vW[c][f]+.001f*g*g;
                        m->W[c][f]-=lt*(m->mW[c][f]/(sqrtf(m->vW[c][f])+1e-8f)+1e-4f*m->W[c][f]);
                    }
                }
                memset(gW,0,CLASSES*FEAT_DIM*sizeof(float)); memset(gB,0,sizeof(gB));
            }
        }
        double b=eval_bpb(m); printf("    ep%d %.4f\n",ep+1,b); fflush(stdout);
        if(b<bb){bb=b;memcpy(best,m,sizeof(AdamState));}
    }
    memcpy(m,best,sizeof(AdamState)); free(best); free(gW);
}

// Load W/B from 0x53454537 (SEE-V1S) for warm start
int load_wb(const char* path, float W[CLASSES][FEAT_DIM], float B[CLASSES]) {
    FILE* f=fopen(path,"rb"); if(!f) return 0;
    uint32_t magic; fread(&magic,4,1,f); rewind(f);
    if (magic!=0x53454537 && magic!=0x53454535 && magic!=0x53454538) { fclose(f); return 0; }
    uint32_t hdr4[4]; fread(hdr4,4,4,f);
    int nf=(magic==0x53454538)?5:(magic==0x53454537)?4:3;
    fseek(f,nf*4,SEEK_CUR);
    if (magic==0x53454538) fseek(f,(long)SEE_N_OJA*43*4,SEEK_CUR); // skip W_oja
    fseek(f,(long)CLASSES*CLASSES*CLASSES*4,SEEK_CUR);
    fseek(f,FEAT_DIM*4,SEEK_CUR);
    fseek(f,FEAT_DIM*4,SEEK_CUR);
    size_t rW=fread(W,sizeof(float),CLASSES*FEAT_DIM,f);
    size_t rB=fread(B,sizeof(float),CLASSES,f);
    fclose(f);
    return (rW==CLASSES*FEAT_DIM && rB==CLASSES)?1:0;
}

int main(int argc, char** argv) {
    if (argc<4){
        printf("Usage: %s <dataset> <weights_prefix> <base_c20_weights>\n",argv[0]);
        return 1;
    }
    FILE* f=fopen(argv[1],"rb");
    if(!f){fprintf(stderr,"Cannot open %s\n",argv[1]);return 1;}
    fseek(f,0,SEEK_END); data_size=ftell(f); fseek(f,0,SEEK_SET);
    data=malloc(data_size); fread(data,1,data_size,f); fclose(f);
    train_start=0; train_len=(int)(((long long)data_size*50)/100);
    val_start=(int)(((long long)data_size*50)/100);
    val_len=(int)(((long long)data_size*25)/100);
    printf("Dataset %ld  train=%d  val=%d\n",data_size,train_len,val_len); fflush(stdout);

    features_train=malloc((size_t)train_len*FEAT_DIM*sizeof(float));
    features_val  =malloc((size_t)val_len  *FEAT_DIM*sizeof(float));
    target_train=malloc(train_len); ctx_train=malloc(train_len); ctx2_train=malloc(train_len);
    target_val  =malloc(val_len);   ctx_val  =malloc(val_len);   ctx2_val  =malloc(val_len);

    // Fixed: SEE-V1S params
    SiliconEntropyState see;
    see_init(&see,42,4,0.75f);
    see.multiscale_mode=1; see.alpha_fast=0.5f; see.alpha_mid=0.9f; see.alpha_slow=0.99f;

    // N-grams: computed once with eta_oja=0 to get stable features
    printf("Pass 0: N-grams (eta=0)...\n"); fflush(stdout);
    see.eta_oja=0.0f;
    extract_with_oja(&see,train_start,train_len,features_train,target_train,ctx_train,ctx2_train);
    compute_ngrams();

    // Keep raw features for re-use across eta variants
    float* raw_train=malloc((size_t)train_len*FEAT_DIM*sizeof(float));
    float* raw_val  =malloc((size_t)val_len  *FEAT_DIM*sizeof(float));

    // eta grid
    float eta_vals[]={1e-3f, 1e-4f};
    const char* sfx[]={"_eta1e3.bin","_eta1e4.bin"};
    int n_eta=2;
    double results[2];

    for (int ei=0;ei<n_eta;ei++) {
        float eta=eta_vals[ei];
        printf("\n====== eta_oja=%.0e ======\n",(double)eta); fflush(stdout);

        // Pass 1: Oja adapts unsupervised on train stream
        printf("  Pass 1: Oja adaptation (train stream)...\n"); fflush(stdout);
        see_oja_reset(&see);
        see.eta_oja=eta;
        extract_with_oja(&see,train_start,train_len,raw_train,target_train,ctx_train,ctx2_train);

        // Freeze W_oja, re-extract with frozen projection
        printf("  Pass 2: re-extract with frozen W_oja...\n"); fflush(stdout);
        see.eta_oja=0.0f;  // freeze
        extract_with_oja(&see,train_start,train_len,features_train,target_train,ctx_train,ctx2_train);
        extract_with_oja(&see,val_start,  val_len,  features_val,  target_val,  ctx_val,  ctx2_val);

        // Normalize + clamp ±2.0 (SEE-V1S)
        normalize_and_clamp(FEAT_CLAMP_DEFAULT);

        // Warm start W/B from base weights
        AdamState* m=calloc(1,sizeof(AdamState));
        if (load_wb(argv[3],m->W,m->B))
            printf("  Warm start W/B from %s\n",argv[3]);
        else
            printf("  Warning: cold start (base weights not loaded)\n");
        fflush(stdout);

        // Fine-tune W/B
        float lrs[]={0.001f,0.0003f,0.0001f};
        for (int l=0;l<3;l++){printf("  LR %.4f\n",lrs[l]);fflush(stdout);train_lr(m,3,256,lrs[l]);}
        results[ei]=eval_bpb(m);

        // Save: magic 0x53454538
        char path[1024]; snprintf(path,sizeof(path),"%s%s",argv[2],sfx[ei]);
        FILE* fw=fopen(path,"wb");
        if (fw) {
            uint32_t hdr[4]={0x53454538,1,FEAT_DIM,4};
            float hf[5]={0.75f,0.1f,0.5f,FEAT_CLAMP_DEFAULT,eta};
            fwrite(hdr,sizeof(hdr),1,fw);
            fwrite(hf,sizeof(hf),1,fw);
            // Save W_oja (frozen state from end of Pass 1)
            fwrite(see.W_oja,sizeof(float),SEE_N_OJA*43,fw);
            fwrite(trigram_logits,sizeof(float),CLASSES*CLASSES*CLASSES,fw);
            fwrite(feat_mean,sizeof(float),FEAT_DIM,fw);
            fwrite(feat_std, sizeof(float),FEAT_DIM,fw);
            fwrite(m->W,sizeof(float),CLASSES*FEAT_DIM,fw);
            fwrite(m->B,sizeof(float),CLASSES,fw);
            fclose(fw);
            printf("  Saved %s  BPB=%.4f\n",path,results[ei]);
        }
        free(m);
    }

    printf("\n\n====== Phase 43.C — Oja Tribunal ======\n");
    printf("  Base (SEE-V1S c20): ~2.2656 BPB\n");
    printf("  Gate: BPB < 2.26, longest <= 10, top1 <= 26%%\n\n");
    printf("  %-14s  Val BPB   Delta vs c20\n","eta_oja");
    for (int ei=0;ei<n_eta;ei++)
        printf("  %.0e         %.4f    %+.4f\n",(double)eta_vals[ei],results[ei],results[ei]-2.2656);
    int best=0; if(results[1]<results[0]) best=1;
    printf("\nBest: eta=%.0e (%.4f BPB)\n",(double)eta_vals[best],results[best]);
    if (results[best]<2.2656)
        printf("SIGNAL: Oja improves BPB -> plastic cells learned useful directions\n");
    else
        printf("SIGNAL: Oja neutral/negative -> unsupervised PCA did not help readout\n");

    free(raw_train); free(raw_val);
    return 0;
}
