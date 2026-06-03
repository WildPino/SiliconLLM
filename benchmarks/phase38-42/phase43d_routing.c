// Phase 43.D - Byte-to-Lane Routing Tribunal (geometry of writing)
// Hypothesis: not HOW HARD a byte writes (byte_gain 43.B failed), but WHERE it
// writes. byte_route[256][32] reshapes each byte's M4 signature (l0_out[0:32])
// for the L1 FAST-band write only; mid/slow use raw l0. Rows are mean-1.0
// (geometry, not amplitude) and clamped near 1.0, strongly regularized.
//
// base: SEE-V2 (13c Oja eta1e-3 FROZEN, clamp +/-2.0), weights phase43c_eta1e3.bin
//       -> W_oja AND readout warm-start both come from SEE-V2.
//       At route=1 (round 0) features == SEE-V2 -> round0 BPB ~ 2.2617 (control).
//
// Alternating optimization (like 43.B): train readout -> 1-step route gradient
//   -> update route (SGD + reg->1.0 + recenter mean=1.0 + clamp) -> re-extract
//   -> retrain readout. Oja stays frozen (eta=0); routing is the only new plasticity.
//
// Variants (clamp x reg):
//   D.A  clamp[0.5,1.5] reg0.10   (tight, strong reg)
//   D.B  clamp[0.25,2.0] reg0.05  (wide)
//   D.C  clamp[0.5,1.5] reg0.05   (tight, moderate reg)
//
// Weight format 0x5345453B: float[6]{decay,0.1,alpha_fast,feat_clamp,eta_oja,blend}
//   + uint32 n_oja + W_oja[n_oja*43] + byte_route[256*32]
//   + trigram + feat_mean + feat_std + W + B
//
// Build (from repo root):
//   gcc -O3 -march=native -mavx2 -mfma \
//       benchmarks/phase38-42/phase43d_routing.c \
//       src/silicon_entropy.c src/silicon_v0.c \
//       -o bin/phase43d_routing.exe -lm -I .
// Run:
//   bin/phase43d_routing.exe <dataset> <weights_prefix> <seev2_weights>

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
#define FEAT_DIM  SEE_FEATURE_DIM
#define FEAT_CLAMP_DEFAULT 2.0f
#define SEE_V2_BPB 2.2617
#define N_OUTER   3            // alternating rounds
#define LR_ROUTE  0.03f

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

typedef struct { float W[CLASSES][FEAT_DIM]; float B[CLASSES];
                 float mW[CLASSES][FEAT_DIM]; float vW[CLASSES][FEAT_DIM];
                 float mB[CLASSES]; float vB[CLASSES]; int t; } AdamState;

static uint8_t *data; static long data_size;
static int train_start, train_len, val_start, val_len;
static float *features_train, *features_val;
static uint8_t *target_train, *target_val, *ctx_train, *ctx_val, *ctx2_train, *ctx2_val;
static float (*trigram_logits)[CLASSES][CLASSES];
static float feat_mean[FEAT_DIM], feat_std[FEAT_DIM];

void extract(SiliconEntropyState* see, int start, int len, float* of, uint8_t* ot, uint8_t* oc, uint8_t* oc2) {
    see_reset(see);
    for (int i=0;i<=start+1;i++) see_observe(see,data[i]);
    for (int i=0;i<len;i++){int g=start+i; ot[i]=data[g+2]; oc[i]=data[g+1]; oc2[i]=data[g];
        see_extract(see,&of[(size_t)i*FEAT_DIM]); see_observe(see,ot[i]);}
}

void compute_ngrams(void) {
    double (*tc)[CLASSES][CLASSES]=malloc(CLASSES*CLASSES*CLASSES*sizeof(double));
    double (*tt)[CLASSES]=malloc(CLASSES*CLASSES*sizeof(double));
    memset(tc,0,CLASSES*CLASSES*CLASSES*sizeof(double)); memset(tt,0,CLASSES*CLASSES*sizeof(double));
    trigram_logits=malloc(CLASSES*CLASSES*CLASSES*sizeof(float));
    for(int i=0;i<train_len;i++){uint8_t t=target_train[i],c1=ctx_train[i],c2=ctx2_train[i]; tc[c2][c1][t]++; tt[c2][c1]++;}
    for(int i=0;i<CLASSES;i++) for(int j=0;j<CLASSES;j++) for(int k=0;k<CLASSES;k++)
        trigram_logits[i][j][k]=(float)log((tc[i][j][k]+.1)/(tt[i][j]+CLASSES*.1));
    free(tc); free(tt);
}

void normalize_and_clamp(float clamp_val) {
    for(int f=0;f<FEAT_DIM;f++){
        double m=0; for(int i=0;i<train_len;i++) m+=features_train[(size_t)i*FEAT_DIM+f]; m/=train_len;
        double v=0; for(int i=0;i<train_len;i++){double d=features_train[(size_t)i*FEAT_DIM+f]-m;v+=d*d;} v=sqrt(v/train_len)+1e-8;
        feat_mean[f]=(float)m; feat_std[f]=(float)v;
        for(int i=0;i<train_len;i++){float x=(features_train[(size_t)i*FEAT_DIM+f]-(float)m)/(float)v; if(x>clamp_val)x=clamp_val; if(x<-clamp_val)x=-clamp_val; features_train[(size_t)i*FEAT_DIM+f]=x;}
        for(int i=0;i<val_len;i++){float x=(features_val[(size_t)i*FEAT_DIM+f]-(float)m)/(float)v; if(x>clamp_val)x=clamp_val; if(x<-clamp_val)x=-clamp_val; features_val[(size_t)i*FEAT_DIM+f]=x;}
    }
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
    printf("    ep0 %.4f\n",bb); fflush(stdout);
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
        double b=eval_model(m); printf("    ep%d %.4f\n",ep+1,b); fflush(stdout);
        if(b<bb){bb=b;memcpy(best,m,sizeof(AdamState));}}
    memcpy(m,best,sizeof(AdamState)); free(best); free(gW);
}

// 1-step gradient estimate for byte_route[b][k], k in [0,SEE_ROUTE_LANES).
// route[b][k] scales last_l0[k] into the FAST band write (both the plastic Oja
// projection and the identity write). With beta=1 (h=y):
//   d l1_fast[j]/d route[b][k] = (1-af) * last_l0[k] * W_oja[j][k]   (plastic j<n_oja)
//   d l1_fast[k]/d route[b][k] = (1-af) * last_l0[k]                 (identity k>=n_oja)
// backprop_fast[j] = (sum_c err[c]*W[c][64+j]) / feat_std[64+j]
void estimate_route_gradient(SiliconEntropyState* see, AdamState* m,
                             double grad[256][SEE_ROUTE_LANES], long* count) {
    memset(grad, 0, 256*SEE_ROUTE_LANES*sizeof(double));
    memset(count, 0, 256*sizeof(long));
    float af1 = 1.0f - see->alpha_fast;
    int n_oja = see->n_oja; if (n_oja > 43) n_oja = 43;

    see_reset(see);
    for (int i=0;i<=train_start+1;i++) see_observe(see,data[i]);

    for (int i=0; i<train_len; i++) {
        float feat[FEAT_DIM];
        see_extract(see, feat);
        for (int f=0;f<FEAT_DIM;f++){
            float x=(feat[f]-feat_mean[f])/feat_std[f];
            if(x> FEAT_CLAMP_DEFAULT) x= FEAT_CLAMP_DEFAULT; if(x<-FEAT_CLAMP_DEFAULT) x=-FEAT_CLAMP_DEFAULT;
            feat[f]=x;
        }
        float lg[CLASSES], pr[CLASSES], mx=-1e9f;
        for (int c=0;c<CLASSES;c++){lg[c]=m->B[c]+trigram_logits[ctx2_train[i]][ctx_train[i]][c]+dot_simd(m->W[c],feat,FEAT_DIM);if(lg[c]>mx)mx=lg[c];}
        float se=0; for(int c=0;c<CLASSES;c++){pr[c]=expf(lg[c]-mx);se+=pr[c];} for(int c=0;c<CLASSES;c++) pr[c]/=se;
        float err[CLASSES]; for(int c=0;c<CLASSES;c++) err[c]=pr[c]-(c==target_train[i]?1.f:0.f);

        // backprop into fast-band cells [0:43] -> feature indices [64:64+43]
        float bpf[43];
        for (int j=0;j<43;j++){
            double s=0; for(int c=0;c<CLASSES;c++) s+=err[c]*m->W[c][64+j];
            bpf[j]=(float)s / feat_std[64+j];
        }

        uint8_t b = ctx_train[i];
        const float* l0 = see->last_l0;     // raw L0 (route-independent)
        for (int k=0;k<SEE_ROUTE_LANES;k++){
            float factor = af1 * l0[k];
            double plastic=0; for(int j=0;j<n_oja;j++) plastic += bpf[j]*see->W_oja[j][k];
            double ident = (k>=n_oja && k<43) ? bpf[k] : 0.0;
            grad[b][k] += factor * (plastic + ident);
        }
        count[b]++;

        see_observe(see, target_train[i]);
    }
}

// Per-round route diagnostics over ACTIVE bytes (count[b]>0); inactive bytes
// stay route=1.0 and would only dilute the stats. Called AFTER the row-mean
// recenter + clamp, so it reflects the route the next extract will use.
void route_diagnostics(SiliconEntropyState* see, long* count, float lo, float hi) {
    int n_active=0; long n_vals=0, n_at_min=0, n_at_max=0;
    double sum=0, sum2=0, sum_abs_dev=0;
    float vmin=1e9f, vmax=-1e9f, eps=1e-4f;
    for (int b=0;b<256;b++){
        if (count[b]==0) continue;
        n_active++;
        for (int k=0;k<SEE_ROUTE_LANES;k++){
            float v=see->byte_route[b][k];
            n_vals++; sum+=v; sum2+=(double)v*v; sum_abs_dev+=fabs(v-1.0f);
            if (v<vmin) vmin=v; if (v>vmax) vmax=v;
            if (v<=lo+eps) n_at_min++;
            if (v>=hi-eps) n_at_max++;
        }
    }
    if (n_vals==0){ printf("    route diag: no active bytes\n"); return; }
    double mean=sum/n_vals, var=sum2/n_vals-mean*mean, sd=var>0?sqrt(var):0;
    printf("    route diag: active_bytes=%d  mean=%.3f std=%.3f min=%.3f max=%.3f\n",
           n_active, mean, sd, vmin, vmax);
    printf("    route diag: at_min=%.1f%% at_max=%.1f%%  avg|route-1|=%.4f\n",
           100.0*n_at_min/n_vals, 100.0*n_at_max/n_vals, sum_abs_dev/n_vals);
    // top changed bytes by mean |route-1|
    float dev[256];
    for (int b=0;b<256;b++){
        if (count[b]==0){ dev[b]=-1.0f; continue; }
        float d=0; for(int k=0;k<SEE_ROUTE_LANES;k++) d+=fabsf(see->byte_route[b][k]-1.0f);
        dev[b]=d/SEE_ROUTE_LANES;
    }
    printf("    top changed bytes: ");
    for (int t=0;t<8;t++){
        int bm=-1; float dm=-1.0f;
        for (int b=0;b<256;b++) if(dev[b]>dm){ dm=dev[b]; bm=b; }
        if (bm<0 || dm<0) break;
        char c=(bm>=32&&bm<127)?(char)bm:'?';
        printf("'%c'(0x%02x)=%.3f ", c, bm, dm);
        dev[bm]=-1.0f;
    }
    printf("\n"); fflush(stdout);
}

// Load readout W/B for warm start (supports 537/538/539/53A/53B).
int load_wb(const char* path, float W[CLASSES][FEAT_DIM], float B[CLASSES]) {
    FILE* f=fopen(path,"rb"); if(!f) return 0;
    uint32_t magic; fread(&magic,4,1,f); rewind(f);
    if (magic!=0x53454537 && magic!=0x53454535 && magic!=0x53454538 &&
        magic!=0x53454539 && magic!=0x5345453A && magic!=0x5345453B) { fclose(f); return 0; }
    uint32_t hdr4[4]; fread(hdr4,4,4,f);
    int nf = (magic==0x5345453A||magic==0x5345453B)?6 : (magic==0x53454538||magic==0x53454539)?5 : (magic==0x53454537)?4 : 3;
    fseek(f,nf*4,SEEK_CUR);
    if (magic==0x53454538) fseek(f,(long)SEE_N_OJA*43*4,SEEK_CUR);
    if (magic==0x53454539 || magic==0x5345453A || magic==0x5345453B) {
        uint32_t no=0; fread(&no,4,1,f); fseek(f,(long)no*43*4,SEEK_CUR);
    }
    if (magic==0x5345453B) fseek(f,(long)256*SEE_ROUTE_LANES*4,SEEK_CUR);
    fseek(f,(long)CLASSES*CLASSES*CLASSES*4,SEEK_CUR);
    fseek(f,FEAT_DIM*4,SEEK_CUR); fseek(f,FEAT_DIM*4,SEEK_CUR);
    size_t rW=fread(W,sizeof(float),CLASSES*FEAT_DIM,f);
    size_t rB=fread(B,sizeof(float),CLASSES,f);
    fclose(f);
    return (rW==CLASSES*FEAT_DIM && rB==CLASSES)?1:0;
}

// Load frozen W_oja + n_oja from a SEE-V2/C2/C3/D weight file.
int load_oja(const char* path, SiliconEntropyState* see) {
    FILE* f=fopen(path,"rb"); if(!f) return 0;
    uint32_t magic; fread(&magic,4,1,f); rewind(f);
    uint32_t hdr4[4]; fread(hdr4,4,4,f);
    int n_oja, nf;
    if      (magic==0x53454538){ nf=5; n_oja=SEE_N_OJA; }
    else if (magic==0x53454539){ nf=5; n_oja=-1; }
    else if (magic==0x5345453A){ nf=6; n_oja=-1; }
    else if (magic==0x5345453B){ nf=6; n_oja=-1; }
    else { fclose(f); return 0; }
    fseek(f, nf*4, SEEK_CUR);
    if (n_oja<0){ uint32_t no=0; fread(&no,4,1,f); n_oja=(int)no; }
    if (n_oja<0 || n_oja>SEE_N_OJA_MAX){ fclose(f); return 0; }
    size_t r=fread(see->W_oja, sizeof(float), (size_t)n_oja*43, f);
    fclose(f);
    if (r != (size_t)n_oja*43) return 0;
    see->n_oja = n_oja;
    return 1;
}

int main(int argc, char** argv) {
    if (argc<4){printf("Usage: %s <dataset> <weights_prefix> <seev2_weights>\n",argv[0]);return 1;}
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
    see_init(&see,42,4,0.75f);
    see.multiscale_mode=1; see.alpha_fast=0.5f; see.alpha_mid=0.9f; see.alpha_slow=0.99f;
    see.eta_oja=0.0f; see.plastic_blend=1.0f;   // Oja FROZEN; pure projection

    if (!load_oja(argv[3], &see)) { fprintf(stderr,"Failed to load W_oja from %s\n",argv[3]); return 1; }
    printf("Loaded frozen W_oja from %s (n_oja=%d)\n", argv[3], see.n_oja); fflush(stdout);

    // N-grams once (route=1, frozen Oja)
    printf("Pass 0: N-grams...\n"); fflush(stdout);
    extract(&see,train_start,train_len,features_train,target_train,ctx_train,ctx2_train);
    compute_ngrams();

    // Variants: clamp x reg
    struct { float lo, hi, reg; const char* sfx; const char* name; } cfgs[] = {
        {0.50f, 1.50f, 0.10f, "_DA.bin", "D.A clamp[0.5,1.5] reg0.10"},
        {0.25f, 2.00f, 0.05f, "_DB.bin", "D.B clamp[0.25,2.0] reg0.05"},
        {0.50f, 1.50f, 0.05f, "_DC.bin", "D.C clamp[0.5,1.5] reg0.05"},
    };
    int n_cfg = 3;
    double results[3];

    double (*grad)[SEE_ROUTE_LANES] = malloc(256*SEE_ROUTE_LANES*sizeof(double));
    long count[256];

    for (int ci=0; ci<n_cfg; ci++) {
        float lo=cfgs[ci].lo, hi=cfgs[ci].hi, reg=cfgs[ci].reg;
        printf("\n====== %s ======\n", cfgs[ci].name); fflush(stdout);

        // Reset route to identity
        for (int b=0;b<256;b++) for (int k=0;k<SEE_ROUTE_LANES;k++) see.byte_route[b][k]=1.0f;

        // Round 0: features (route=1) == SEE-V2; warm-start readout from SEE-V2
        printf("  Round 0 (route=1, control)...\n"); fflush(stdout);
        extract(&see,train_start,train_len,features_train,target_train,ctx_train,ctx2_train);
        extract(&see,val_start,  val_len,  features_val,  target_val,  ctx_val,  ctx2_val);
        normalize_and_clamp(FEAT_CLAMP_DEFAULT);
        AdamState* m=calloc(1,sizeof(AdamState));
        if (load_wb(argv[3],m->W,m->B)) printf("  Warm start readout from SEE-V2\n");
        else printf("  Warning: cold readout\n");
        fflush(stdout);
        float lrs[]={0.0003f,0.0001f};
        for(int l=0;l<2;l++){printf("  LR %.4f\n",lrs[l]);fflush(stdout);train_lr(m,2,256,lrs[l]);}
        double bpb0=eval_model(m);
        const char* ctrl = (fabs(bpb0 - SEE_V2_BPB) <= 0.01) ? "OK reproduces SEE-V2" : "WARN diverges from SEE-V2";
        printf("  Round 0 BPB: %.4f (control vs SEE-V2 %.4f -> %s)\n", bpb0, SEE_V2_BPB, ctrl); fflush(stdout);

        // Alternating: route gradient -> update -> re-extract -> retrain
        for (int outer=0; outer<N_OUTER; outer++) {
            printf("  -- route round %d (clamp[%.2f,%.2f] reg%.2f) --\n", outer+1, lo, hi, reg); fflush(stdout);
            estimate_route_gradient(&see, m, grad, count);

            // Update route: SGD + reg->1.0 + recenter row mean to 1.0 + clamp
            for (int b=0;b<256;b++){
                if (count[b]==0) continue;
                for (int k=0;k<SEE_ROUTE_LANES;k++){
                    float g=(float)(grad[b][k]/count[b]);
                    see.byte_route[b][k] -= LR_ROUTE*g;
                    see.byte_route[b][k] += reg*(1.0f - see.byte_route[b][k]);
                }
                // recenter mean to 1.0 (geometry, not amplitude)
                float mean=0; for(int k=0;k<SEE_ROUTE_LANES;k++) mean+=see.byte_route[b][k];
                mean/=SEE_ROUTE_LANES;
                for(int k=0;k<SEE_ROUTE_LANES;k++){
                    float v=see.byte_route[b][k] + (1.0f-mean);
                    if(v<lo)v=lo; if(v>hi)v=hi;
                    see.byte_route[b][k]=v;
                }
            }

            route_diagnostics(&see, count, lo, hi);

            extract(&see,train_start,train_len,features_train,target_train,ctx_train,ctx2_train);
            extract(&see,val_start,  val_len,  features_val,  target_val,  ctx_val,  ctx2_val);
            normalize_and_clamp(FEAT_CLAMP_DEFAULT);
            memset(m->mW,0,sizeof(m->mW)); memset(m->vW,0,sizeof(m->vW));
            memset(m->mB,0,sizeof(m->mB)); memset(m->vB,0,sizeof(m->vB)); m->t=0;
            for(int l=0;l<2;l++){printf("  LR %.4f\n",lrs[l]);fflush(stdout);train_lr(m,2,256,lrs[l]);}
            double b=eval_model(m);
            printf("  route round %d BPB: %.4f (delta vs round0 %+.4f)\n", outer+1, b, b-bpb0); fflush(stdout);
        }
        results[ci]=eval_model(m);

        // Save 0x5345453B
        char path[1024]; snprintf(path,sizeof(path),"%s%s",argv[2],cfgs[ci].sfx);
        FILE* fw=fopen(path,"wb");
        if (fw) {
            uint32_t hdr[4]={0x5345453B,1,FEAT_DIM,4};
            float hf[6]={0.75f,0.1f,0.5f,FEAT_CLAMP_DEFAULT,0.0f,1.0f}; // eta=0 frozen, blend=1
            uint32_t n_oja_u=(uint32_t)see.n_oja;
            fwrite(hdr,sizeof(hdr),1,fw);
            fwrite(hf,sizeof(float),6,fw);
            fwrite(&n_oja_u,4,1,fw);
            fwrite(see.W_oja,sizeof(float),(size_t)see.n_oja*43,fw);
            fwrite(see.byte_route,sizeof(float),256*SEE_ROUTE_LANES,fw);
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

    printf("\n\n====== Phase 43.D - Byte-to-Lane Routing ======\n");
    printf("  SEE-V2 reference: %.4f BPB\n", SEE_V2_BPB);
    printf("  Promotion gate: BPB <= %.4f (>=0.002 better) + word-gate (see ps1)\n\n", SEE_V2_BPB-0.002);
    printf("  %-30s  Val BPB   Delta vs SEE-V2\n","config");
    for (int ci=0;ci<n_cfg;ci++)
        printf("  %-30s  %.4f    %+.4f\n", cfgs[ci].name, results[ci], results[ci]-SEE_V2_BPB);
    int best=0; for(int ci=1;ci<n_cfg;ci++) if(results[ci]<results[best]) best=ci;
    printf("\n  Best: %s (%.4f, %+.4f vs SEE-V2)\n", cfgs[best].name, results[best], results[best]-SEE_V2_BPB);
    if (results[best] <= SEE_V2_BPB - 0.002)
        printf("  SIGNAL: routing improves BPB -> 'where a byte writes' is a real axis (check word-gate)\n");
    else
        printf("  SIGNAL: routing flat -> SEE-V2 likely the substrate limit; needs structural jump\n");

    free(grad); free(data);
    return 0;
}
