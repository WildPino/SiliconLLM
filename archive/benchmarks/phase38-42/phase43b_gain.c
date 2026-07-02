// Phase 43.B — Trainable injection byte_gain[256]
// Alternating optimization: train readout -> estimate gain gradient -> update gains -> repeat
//
// gain[b] scales how strongly byte b writes into the L1 memory bands.
// Gradient approximation (1-step): ∂loss[i]/∂gain[ctx[i]] ≈ backprop_L1[i] · (1-alpha)*l0[i]
// where see.last_l0 at extraction step i = raw L0 from ctx_train[i].
//
// Build (from repo root):
//   gcc -O3 -march=native -mavx2 -mfma \
//       benchmarks/phase38-42/phase43b_gain.c \
//       src/silicon_entropy.c src/silicon_v0.c \
//       -o bin/phase43b_gain.exe -lm -I .
// Run:
//   bin/phase43b_gain.exe <dataset> <out_weights.bin>

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
#define FEAT_DIM  SEE_FEATURE_DIM   // 192
#define N_OUTER   3                 // alternating optimization rounds
#define LR_GAIN   0.05f
#define GAIN_MIN  0.25f
#define GAIN_MAX  4.0f
#define GAIN_REG  0.02f             // pull toward 1.0 per step

typedef struct { float W[CLASSES][FEAT_DIM]; float B[CLASSES];
                 float mW[CLASSES][FEAT_DIM]; float vW[CLASSES][FEAT_DIM];
                 float mB[CLASSES]; float vB[CLASSES]; int t; } AdamState;

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
    for (int i=0;i<len;i++){int g=start+i; ot[i]=data[g+2]; oc[i]=data[g+1]; oc2[i]=data[g];
        see_extract(see,&of[(size_t)i*FEAT_DIM]); see_observe(see,ot[i]);}
}
void compute_ngrams(void) {
    double bi[CLASSES][CLASSES]={0},bt[CLASSES]={0};
    double (*tc)[CLASSES][CLASSES]=malloc(CLASSES*CLASSES*CLASSES*sizeof(double));
    double (*tt)[CLASSES]=malloc(CLASSES*CLASSES*sizeof(double));
    memset(tc,0,CLASSES*CLASSES*CLASSES*sizeof(double)); memset(tt,0,CLASSES*CLASSES*sizeof(double));
    trigram_logits=malloc(CLASSES*CLASSES*CLASSES*sizeof(float));
    for(int i=0;i<train_len;i++){uint8_t t=target_train[i],c1=ctx_train[i],c2=ctx2_train[i];
        bi[c1][t]++;bt[c1]++;tc[c2][c1][t]++;tt[c2][c1]++;}
    for(int i=0;i<CLASSES;i++) for(int j=0;j<CLASSES;j++){
        bigram_logits[i][j]=(float)log((bi[i][j]+.1)/(bt[i]+CLASSES*.1));
        for(int k=0;k<CLASSES;k++) trigram_logits[i][j][k]=(float)log((tc[i][j][k]+.1)/(tt[i][j]+CLASSES*.1));}
    free(tc); free(tt);
}
void normalize(void) {
    for(int f=0;f<FEAT_DIM;f++){
        double m=0; for(int i=0;i<train_len;i++) m+=features_train[(size_t)i*FEAT_DIM+f]; m/=train_len;
        double v=0; for(int i=0;i<train_len;i++){double d=features_train[(size_t)i*FEAT_DIM+f]-m;v+=d*d;} v=sqrt(v/train_len)+1e-8;
        feat_mean[f]=(float)m; feat_std[f]=(float)v;
        for(int i=0;i<train_len;i++) features_train[(size_t)i*FEAT_DIM+f]=(features_train[(size_t)i*FEAT_DIM+f]-(float)m)/(float)v;
        for(int i=0;i<val_len;i++)   features_val[(size_t)i*FEAT_DIM+f]=(features_val[(size_t)i*FEAT_DIM+f]-(float)m)/(float)v;}
}
double eval_model(AdamState* m) {
    double tot=0;
    for(int i=0;i<val_len;i++){
        float lg[CLASSES],mx=-1e9f;
        for(int c=0;c<CLASSES;c++){lg[c]=m->B[c]+trigram_logits[ctx2_val[i]][ctx_val[i]][c]+dot_simd(m->W[c],&features_val[(size_t)i*FEAT_DIM],FEAT_DIM);if(lg[c]>mx)mx=lg[c];}
        float se=0; for(int c=0;c<CLASSES;c++) se+=expf(lg[c]-mx);
        tot-=log2(fmaxf(expf(lg[target_val[i]]-mx)/se,1e-10f));}
    return tot/val_len;
}
void train_lr(AdamState* m, int eps, int bs, float lr) {
    float *gW=malloc(CLASSES*FEAT_DIM*sizeof(float)),gB[CLASSES],lg[CLASSES],pr[CLASSES];
    AdamState *best=malloc(sizeof(AdamState)); double bb=eval_model(m); memcpy(best,m,sizeof(AdamState));
    printf("  ep0 %.4f\n",bb); fflush(stdout);
    for(int ep=0;ep<eps;ep++){
        memset(gW,0,CLASSES*FEAT_DIM*sizeof(float)); memset(gB,0,sizeof(gB));
        for(int i=0;i<train_len;i++){
            float mx=-1e9f;
            for(int c=0;c<CLASSES;c++){lg[c]=m->B[c]+trigram_logits[ctx2_train[i]][ctx_train[i]][c]+dot_simd(m->W[c],&features_train[(size_t)i*FEAT_DIM],FEAT_DIM);if(lg[c]>mx)mx=lg[c];}
            float se=0; for(int c=0;c<CLASSES;c++){pr[c]=expf(lg[c]-mx);se+=pr[c];} for(int c=0;c<CLASSES;c++) pr[c]/=se;
            for(int c=0;c<CLASSES;c++){float e=pr[c]-(c==target_train[i]?1.f:0.f);gB[c]+=e/bs;grad_simd(&gW[c*FEAT_DIM],&features_train[(size_t)i*FEAT_DIM],e/bs,FEAT_DIM);}
            if((i+1)%bs==0||(i+1)==train_len){
                m->t++; float lt=lr*sqrtf(1.f-powf(.999f,m->t))/(1.f-powf(.9f,m->t));
                for(int c=0;c<CLASSES;c++){m->mB[c]=.9f*m->mB[c]+.1f*gB[c];m->vB[c]=.999f*m->vB[c]+.001f*gB[c]*gB[c];m->B[c]-=lt*(m->mB[c]/(sqrtf(m->vB[c])+1e-8f)+1e-4f*m->B[c]);
                for(int f=0;f<FEAT_DIM;f++){float g=gW[c*FEAT_DIM+f];m->mW[c][f]=.9f*m->mW[c][f]+.1f*g;m->vW[c][f]=.999f*m->vW[c][f]+.001f*g*g;m->W[c][f]-=lt*(m->mW[c][f]/(sqrtf(m->vW[c][f])+1e-8f)+1e-4f*m->W[c][f]);}}
                memset(gW,0,CLASSES*FEAT_DIM*sizeof(float)); memset(gB,0,sizeof(gB));}}
        double b=eval_model(m); printf("  ep%d %.4f\n",ep+1,b); fflush(stdout);
        if(b<bb){bb=b;memcpy(best,m,sizeof(AdamState));}}
    memcpy(m,best,sizeof(AdamState)); free(best); free(gW);
}

// 1-step gradient estimate for byte_gain.
// At extraction step i, see.last_l0 = raw L0 from ctx_train[i].
// gain[ctx_train[i]] scaled that L0's contribution to L1.
// grad[ctx_train[i]] += backprop_L1[i] · (1-alpha) * last_l0
void estimate_gain_gradient(SiliconEntropyState* see, AdamState* m,
                             double* grad_gain, long* count_gain) {
    memset(grad_gain, 0, CLASSES*sizeof(double));
    memset(count_gain, 0, CLASSES*sizeof(long));

    see_reset(see);
    for (int i=0;i<=train_start+1;i++) see_observe(see,data[i]);

    float af1 = 1.0f - see->alpha_fast;
    float am1 = 1.0f - see->alpha_mid;
    float as1 = 1.0f - see->alpha_slow;

    for (int i=0; i<train_len; i++) {
        // Extract and normalize features inline
        float feat[FEAT_DIM];
        see_extract(see, feat);
        for (int f=0;f<FEAT_DIM;f++) feat[f]=(feat[f]-feat_mean[f])/feat_std[f];

        // Compute softmax error
        float lg[CLASSES], pr[CLASSES], mx=-1e9f;
        for (int c=0;c<CLASSES;c++){
            lg[c]=m->B[c]+trigram_logits[ctx2_train[i]][ctx_train[i]][c]+dot_simd(m->W[c],feat,FEAT_DIM);
            if(lg[c]>mx) mx=lg[c];}
        float se=0; for(int c=0;c<CLASSES;c++){pr[c]=expf(lg[c]-mx);se+=pr[c];} for(int c=0;c<CLASSES;c++) pr[c]/=se;
        float err[CLASSES]; for(int c=0;c<CLASSES;c++) err[c]=pr[c]-(c==target_train[i]?1.f:0.f);

        // Compute backprop gradient w.r.t. L1 features [64:192]
        // backprop_L1[j] = sum_c W[c][64+j] * err[c]  (j in 0..127)
        // Then g_val = dot(backprop_L1, ema_scaled_l0) where ema_scaled_l0 uses (1-alpha) per band
        // To avoid allocating backprop_L1[128], compute g_val directly:
        //   g_val = sum_j [ (sum_c W[c][64+j]*err[c]) * (1-alpha_j)*l0[j] ]
        //         = sum_c err[c] * sum_j W[c][64+j] * (1-alpha_j) * l0[j]
        //         = dot(err, W_L1 * ema_l0)  where ema_l0 is 128D
        // Precompute ema_l0[128]:
        float ema_l0[128];
        for (int j=0;j<43;j++) ema_l0[j]    = af1 * see->last_l0[j];
        for (int j=0;j<43;j++) ema_l0[43+j] = am1 * see->last_l0[j];
        for (int j=0;j<42;j++) ema_l0[86+j] = as1 * see->last_l0[j];

        // g_val = sum_c err[c] * dot(W[c][64:192], ema_l0)
        double g_val = 0.0;
        for (int c=0;c<CLASSES;c++) {
            if (fabsf(err[c]) < 1e-6f) continue;
            g_val += err[c] * dot_simd(m->W[c]+64, ema_l0, 128);
        }

        grad_gain[ctx_train[i]] += g_val;
        count_gain[ctx_train[i]]++;

        see_observe(see, target_train[i]);
    }
}

int main(int argc, char** argv) {
    if (argc<3){printf("Usage: %s <dataset> <out_weights.bin>\n",argv[0]);return 1;}
    FILE* f=fopen(argv[1],"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",argv[1]);return 1;}
    fseek(f,0,SEEK_END); data_size=ftell(f); fseek(f,0,SEEK_SET);
    data=malloc(data_size); fread(data,1,data_size,f); fclose(f);
    train_start=0; train_len=(int)(((long long)data_size*50)/100);
    val_start=(int)(((long long)data_size*50)/100); val_len=(int)(((long long)data_size*25)/100);
    printf("Dataset %ld  train=%d  val=%d\n",data_size,train_len,val_len); fflush(stdout);

    features_train=malloc((size_t)train_len*FEAT_DIM*sizeof(float));
    features_val=malloc((size_t)val_len*FEAT_DIM*sizeof(float));
    target_train=malloc(train_len); ctx_train=malloc(train_len); ctx2_train=malloc(train_len);
    target_val=malloc(val_len);     ctx_val=malloc(val_len);     ctx2_val=malloc(val_len);

    SiliconEntropyState see;
    // Start from ms_f0.5 configuration (best from 43.A)
    see_init(&see, 42, 4, 0.75f);
    see.multiscale_mode = 1;
    see.alpha_fast = 0.5f; see.alpha_mid = 0.9f; see.alpha_slow = 0.99f;
    // byte_gain initialized to 1.0 in see_init

    printf("\nComputing N-grams (once)...\n"); fflush(stdout);
    extract(&see,train_start,train_len,features_train,target_train,ctx_train,ctx2_train);
    compute_ngrams();

    AdamState* model = calloc(1, sizeof(AdamState));
    double bpb_history[N_OUTER+1];

    // Round 0: train with all gains = 1.0
    printf("\n====== Round 0: baseline gains (all 1.0) ======\n"); fflush(stdout);
    extract(&see,train_start,train_len,features_train,target_train,ctx_train,ctx2_train);
    extract(&see,val_start,val_len,features_val,target_val,ctx_val,ctx2_val);
    normalize();
    float lrs[]={0.003f,0.001f,0.0003f};
    for(int l=0;l<3;l++){printf("  LR %.4f\n",lrs[l]);fflush(stdout);train_lr(model,5,256,lrs[l]);}
    bpb_history[0] = eval_model(model);
    printf("Round 0 Val BPB: %.4f\n", bpb_history[0]); fflush(stdout);

    // Alternating optimization
    double grad_gain[CLASSES]; long count_gain[CLASSES];
    for (int outer=0; outer<N_OUTER; outer++) {
        printf("\n====== Gain gradient pass (round %d) ======\n", outer+1); fflush(stdout);

        // Estimate gradient
        estimate_gain_gradient(&see, model, grad_gain, count_gain);

        // Update gains with SGD + regularization + clip
        int updated = 0;
        for (int b=0;b<CLASSES;b++) {
            if (count_gain[b] == 0) continue;
            float g = (float)(grad_gain[b] / count_gain[b]);
            see.byte_gain[b] -= LR_GAIN * g;
            see.byte_gain[b] += GAIN_REG * (1.0f - see.byte_gain[b]);
            if (see.byte_gain[b] < GAIN_MIN) see.byte_gain[b] = GAIN_MIN;
            if (see.byte_gain[b] > GAIN_MAX) see.byte_gain[b] = GAIN_MAX;
            updated++;
        }
        printf("  Updated %d byte gains\n", updated);

        // Print top-5 highest and lowest gains
        float sorted_g[CLASSES]; int sorted_b[CLASSES];
        for(int b=0;b<CLASSES;b++){sorted_g[b]=see.byte_gain[b];sorted_b[b]=b;}
        for(int i=0;i<5;i++) for(int j=i+1;j<CLASSES;j++)
            if(sorted_g[j]>sorted_g[i]){float t=sorted_g[i];sorted_g[i]=sorted_g[j];sorted_g[j]=t;int ti=sorted_b[i];sorted_b[i]=sorted_b[j];sorted_b[j]=ti;}
        printf("  Top gains: ");
        for(int i=0;i<5;i++) printf("'%c'(0x%02x)=%.2f ",sorted_b[i]>=' '&&sorted_b[i]<127?(char)sorted_b[i]:'?',sorted_b[i],sorted_g[i]);
        for(int i=0;i<5;i++) for(int j=i+1;j<CLASSES;j++)
            if(sorted_g[j]<sorted_g[i]){float t=sorted_g[i];sorted_g[i]=sorted_g[j];sorted_g[j]=t;int ti=sorted_b[i];sorted_b[i]=sorted_b[j];sorted_b[j]=ti;}
        printf("\n  Low gains: ");
        for(int i=0;i<5;i++) printf("'%c'(0x%02x)=%.2f ",sorted_b[i]>=' '&&sorted_b[i]<127?(char)sorted_b[i]:'?',sorted_b[i],sorted_g[i]);
        printf("\n"); fflush(stdout);

        // Re-extract features with updated gains and retrain
        printf("  Re-extracting features...\n"); fflush(stdout);
        extract(&see,train_start,train_len,features_train,target_train,ctx_train,ctx2_train);
        extract(&see,val_start,val_len,features_val,target_val,ctx_val,ctx2_val);
        normalize();

        // Fine-tune (fewer epochs — gains already warm)
        printf("  Fine-tuning readout...\n"); fflush(stdout);
        memset(model->mW,0,sizeof(model->mW)); memset(model->vW,0,sizeof(model->vW));
        memset(model->mB,0,sizeof(model->mB)); memset(model->vB,0,sizeof(model->vB));
        model->t = 0;
        for(int l=0;l<3;l++){printf("  LR %.4f\n",lrs[l]);fflush(stdout);train_lr(model,3,256,lrs[l]);}
        bpb_history[outer+1] = eval_model(model);
        printf("Round %d Val BPB: %.4f  (delta=%+.4f)\n",
               outer+1, bpb_history[outer+1], bpb_history[outer+1]-bpb_history[0]);
        fflush(stdout);
    }

    // Save final weights
    FILE* fw = fopen(argv[2], "wb");
    if (fw) {
        uint32_t hdr[4]={0x53454536,1,FEAT_DIM,4}; float hf[4]={0.75f,0.1f,0.5f,0.0f};
        fwrite(hdr,sizeof(hdr),1,fw); fwrite(hf,sizeof(hf),1,fw);
        fwrite(see.byte_gain,sizeof(float),CLASSES,fw);
        fwrite(trigram_logits,sizeof(float),CLASSES*CLASSES*CLASSES,fw);
        fwrite(feat_mean,sizeof(float),FEAT_DIM,fw); fwrite(feat_std,sizeof(float),FEAT_DIM,fw);
        fwrite(model->W,sizeof(float),CLASSES*FEAT_DIM,fw); fwrite(model->B,sizeof(float),CLASSES,fw);
        fclose(fw); printf("\nSaved -> %s\n",argv[2]);
    }

    printf("\n\n====== Phase 43.B — Byte Gain Tribunal ======\n");
    printf("  ms_f0.5 baseline: 2.2757 BPB\n");
    printf("  Round   BPB       Delta vs round0\n");
    for(int r=0;r<=N_OUTER;r++)
        printf("  %d       %.4f    %+.4f\n",r,bpb_history[r],bpb_history[r]-bpb_history[0]);

    if (bpb_history[N_OUTER] < bpb_history[0] - 0.01f)
        printf("\nSIGNAL: byte_gain helps (>0.01 BPB) -> injection identity matters\n");
    else
        printf("\nSIGNAL: byte_gain flat -> injection amplitude is not the bottleneck\n");

    free(model); free(data);
    return 0;
}
