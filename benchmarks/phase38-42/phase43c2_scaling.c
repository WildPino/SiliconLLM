// Phase 43.C2 - Plastic Capacity Scaling Tribunal
// Question: can SEE-V2 use MORE local plasticity (13 -> 26 cells) without
// reopening semantic attractors?  A step toward "remember more without falling
// into your own attractors."
//
// Base: SEE-V2 = SEE-V1S c20 + Oja 13 cells eta=1e-3.
// Readout stays stupid (linear residual softmax). Clamp always +/-2.0.
//
// Variants (all warm-started from c20 readout, same as how SEE-V2 was made):
//   C2.0  13 cells eta=1e-3   control / reproduce SEE-V2
//   C2.A  26 cells eta=1e-3
//   C2.B  26 cells eta=3e-4
//   C2.C  26 cells eta=1e-4   safety low-eta
//
// Weight format 0x53454539 (self-describing capacity):
//   {magic,ver,fdim,chunk} + float[5]{decay,0.1,alpha_fast,feat_clamp,eta_oja}
//   + uint32 n_oja + W_oja[n_oja*43] + trigram + feat_mean + feat_std + W + B
//
// Build (from repo root):
//   gcc -O3 -march=native -mavx2 -mfma \
//       benchmarks/phase38-42/phase43c2_scaling.c \
//       src/silicon_entropy.c src/silicon_v0.c \
//       -o bin/phase43c2_scaling.exe -lm -I .
// Run:
//   bin/phase43c2_scaling.exe <dataset> <weights_prefix> <base_c20_weights>

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
#define SEE_V2_BPB 2.2617   // reference: SEE-V2 (13 cells eta=1e-3) val BPB

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

// Load W/B for warm start. Supports 0x53454535/537/538/539 (skips any Oja blob).
int load_wb(const char* path, float W[CLASSES][FEAT_DIM], float B[CLASSES]) {
    FILE* f=fopen(path,"rb"); if(!f) return 0;
    uint32_t magic; fread(&magic,4,1,f); rewind(f);
    if (magic!=0x53454537 && magic!=0x53454535 && magic!=0x53454538 && magic!=0x53454539) { fclose(f); return 0; }
    uint32_t hdr4[4]; fread(hdr4,4,4,f);
    int nf=(magic==0x53454538||magic==0x53454539)?5:(magic==0x53454537)?4:3;
    fseek(f,nf*4,SEEK_CUR);
    if (magic==0x53454538) fseek(f,(long)SEE_N_OJA*43*4,SEEK_CUR);          // fixed 13 cells
    if (magic==0x53454539) { uint32_t no=0; fread(&no,4,1,f); fseek(f,(long)no*43*4,SEEK_CUR); }
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

    // Fixed: SEE-V1S params (multiscale f0.5/0.9/0.99)
    SiliconEntropyState see;
    see_init(&see,42,4,0.75f);
    see.multiscale_mode=1; see.alpha_fast=0.5f; see.alpha_mid=0.9f; see.alpha_slow=0.99f;

    // N-grams: byte-context only, independent of Oja/n_oja. Compute once (eta=0).
    printf("Pass 0: N-grams (eta=0)...\n"); fflush(stdout);
    see.eta_oja=0.0f; see.n_oja=SEE_N_OJA;
    extract_with_oja(&see,train_start,train_len,features_train,target_train,ctx_train,ctx2_train);
    compute_ngrams();

    float* raw_train=malloc((size_t)train_len*FEAT_DIM*sizeof(float));

    // C2 config grid
    struct { int n_oja; float eta; const char* sfx; const char* name; } cfgs[] = {
        {13, 1e-3f, "_C20.bin", "C2.0 13c eta1e-3 (reproduce SEE-V2)"},
        {26, 1e-3f, "_C2A.bin", "C2.A 26c eta1e-3"},
        {26, 3e-4f, "_C2B.bin", "C2.B 26c eta3e-4"},
        {26, 1e-4f, "_C2C.bin", "C2.C 26c eta1e-4 (safety low-eta)"},
    };
    int n_cfg = 4;
    double results[4];

    for (int ci=0; ci<n_cfg; ci++) {
        int   N   = cfgs[ci].n_oja;
        float eta = cfgs[ci].eta;
        printf("\n====== %s  (n_oja=%d, eta=%.0e) ======\n", cfgs[ci].name, N, (double)eta);
        fflush(stdout);

        // Pass 1: Oja adapts unsupervised on train stream with N plastic cells
        printf("  Pass 1: Oja adaptation (train stream)...\n"); fflush(stdout);
        see.n_oja = N;
        see_oja_reset(&see);
        see.eta_oja = eta;
        extract_with_oja(&see,train_start,train_len,raw_train,target_train,ctx_train,ctx2_train);

        // Freeze W_oja, re-extract with frozen projection
        printf("  Pass 2: re-extract with frozen W_oja...\n"); fflush(stdout);
        see.eta_oja = 0.0f;  // freeze; projection (N cells) still applied
        extract_with_oja(&see,train_start,train_len,features_train,target_train,ctx_train,ctx2_train);
        extract_with_oja(&see,val_start,  val_len,  features_val,  target_val,  ctx_val,  ctx2_val);

        // Normalize + clamp +/-2.0 (SEE-V1S homeostasis)
        normalize_and_clamp(FEAT_CLAMP_DEFAULT);

        // Warm start W/B from c20
        AdamState* m=calloc(1,sizeof(AdamState));
        if (load_wb(argv[3],m->W,m->B))
            printf("  Warm start W/B from %s\n",argv[3]);
        else
            printf("  Warning: cold start (base weights not loaded)\n");
        fflush(stdout);

        // Fine-tune W/B
        float lrs[]={0.001f,0.0003f,0.0001f};
        for (int l=0;l<3;l++){printf("  LR %.4f\n",lrs[l]);fflush(stdout);train_lr(m,3,256,lrs[l]);}
        results[ci]=eval_bpb(m);

        // Save: magic 0x53454539 (self-describing n_oja)
        char path[1024]; snprintf(path,sizeof(path),"%s%s",argv[2],cfgs[ci].sfx);
        FILE* fw=fopen(path,"wb");
        if (fw) {
            uint32_t hdr[4]={0x53454539,1,FEAT_DIM,4};
            float hf[5]={0.75f,0.1f,0.5f,FEAT_CLAMP_DEFAULT,eta};
            uint32_t n_oja_u=(uint32_t)N;
            fwrite(hdr,sizeof(hdr),1,fw);
            fwrite(hf,sizeof(hf),1,fw);
            fwrite(&n_oja_u,4,1,fw);
            fwrite(see.W_oja,sizeof(float),(size_t)N*43,fw);   // first N rows
            fwrite(trigram_logits,sizeof(float),CLASSES*CLASSES*CLASSES,fw);
            fwrite(feat_mean,sizeof(float),FEAT_DIM,fw);
            fwrite(feat_std, sizeof(float),FEAT_DIM,fw);
            fwrite(m->W,sizeof(float),CLASSES*FEAT_DIM,fw);
            fwrite(m->B,sizeof(float),CLASSES,fw);
            fclose(fw);
            printf("  Saved %s  BPB=%.4f\n",path,results[ci]);
        }
        free(m);
    }

    printf("\n\n====== Phase 43.C2 - Plastic Capacity Scaling ======\n");
    printf("  SEE-V2 reference (13c eta1e-3): %.4f BPB\n", SEE_V2_BPB);
    printf("  Promotion gate: BPB improves >= 0.002 vs SEE-V2 + word-gate (see ps1)\n\n");
    printf("  %-34s  Val BPB   Delta vs SEE-V2\n","config");
    for (int ci=0;ci<n_cfg;ci++)
        printf("  %-34s  %.4f    %+.4f\n", cfgs[ci].name, results[ci], results[ci]-SEE_V2_BPB);

    // Pick best of the 26-cell variants (ci 1..3) for the headline
    int best=1; for(int ci=2;ci<n_cfg;ci++) if(results[ci]<results[best]) best=ci;
    printf("\n  Best 26-cell: %s (%.4f BPB, %+.4f vs SEE-V2)\n",
           cfgs[best].name, results[best], results[best]-SEE_V2_BPB);
    if (results[best] <= SEE_V2_BPB - 0.002)
        printf("  SIGNAL: 26 cells improves BPB >= 0.002 -> check word-gate for attractors\n");
    else if (results[best] < SEE_V2_BPB)
        printf("  SIGNAL: 26 cells improves BPB but < 0.002 threshold -> 13 may be sweet spot\n");
    else
        printf("  SIGNAL: 26 cells does not improve BPB -> 13 is sweet spot, consider 43.D routing\n");

    // Control sanity: C2.0 should reproduce SEE-V2
    printf("\n  Control C2.0 vs SEE-V2: %.4f vs %.4f (|delta|=%.4f, expect ~0)\n",
           results[0], SEE_V2_BPB, fabs(results[0]-SEE_V2_BPB));

    free(raw_train);
    return 0;
}
