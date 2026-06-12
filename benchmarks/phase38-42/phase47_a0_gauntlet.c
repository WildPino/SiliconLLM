// Phase 47.A0 - sanity gauntlet for the static nonlinear readout (NO new inference memory)
//
// 47.0 found that a 1-hidden MLP on stable [SEE|L2_D1] features hits 2.0947 val BPB,
// beating D1/delta/B3 (all linear) with NO volatile state and NO teacher. That is a huge
// jump (-0.15 vs D1), so before opening 47.A we falsify it:
//   [1] reproduce mlp-hard on 3 held-out validation windows (50%/65%/80% of file)
//   [2] linear probe in the SAME pipeline (same Adam/batching) - must land ~= D1
//   [3] controls: random-label and shuffled-time training - must show NO gain vs D1
//   [4] ablations (H=64): SEE-only / L2-only / SEE+L2 - where does the nonlinear gain live?
//   [5] hidden width ladder H=16/32/64/128 - how small can the readout be?
//   [6] telemetry per probe: pre-trigram logit RMS, output entropy (bits), max prob
// The ladder models are saved (magic 0x53454545) so the harness can word-gate the best
// SMALL one closed-loop ([7], phase47_generator.c). Criterion stays: readout admissible
// only if static, small, stateless - this gauntlet only certifies the compression claim.
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase38-42/phase47_a0_gauntlet.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase47a0_gauntlet.exe -lm -I .
// Run:
//   bin/phase47a0_gauntlet.exe <data> <D1_w> <delta_w> <B3_w> <outprefix> [--len N --epochs E]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include "src/silicon_entropy.h"

#define CLASSES   256
#define BASE_DIM  SEE_FEATURE_DIM
#define L2_DIM    64
#define L3_DIM    64
#define S_DIM     (BASE_DIM + L2_DIM)   // 256 student feature dim (D1 stable)
#define D1_TOT    (BASE_DIM + L2_DIM)
#define B3_TOT    (BASE_DIM + L2_DIM + L3_DIM)
#define N_VAL     3

enum { G_ENTROPY=4 };

static float Pmat[L2_DIM][BASE_DIM];
static float (*trigram)[CLASSES][CLASSES];
static float (*ent_table)[CLASSES];
static float (*margin_table)[CLASSES];
static uint8_t* g_data; static long g_fsz;
static float g_ent_thr; static int g_ent_high=1;

// per-model readout + normalization (D1 / delta / B3 reference baselines)
static float Wd1[CLASSES][D1_TOT], Bd1[CLASSES], md1[D1_TOT], sd1[D1_TOT];
static float Wde[CLASSES][D1_TOT], Bde[CLASSES], mde[D1_TOT], sde[D1_TOT];
static float Wb3[CLASSES][B3_TOT], Bb3[CLASSES], mb3[B3_TOT], sb3[B3_TOT];
static float g_alpha=0.99f, g_l2c_d1=2.0f, g_ls_d1=0.5f, g_l2c_de=2.0f, g_ls_de=0.5f, g_l2c_b3=2.0f, g_ls_b3=0.5f;
static float g_b3_mthr=0.1f, g_b3_alpha=0.9f; static int g_b3_M=4, g_b3_refr=16, g_b3_mode=4;

static inline float dot_avx(const float* w, const float* f, int n){ __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(&w[i]),_mm256_loadu_ps(&f[i]),s);
    float o[8]; _mm256_storeu_ps(o,s); float r=o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; for(;i<n;i++) r+=w[i]*f[i]; return r; }
static void gen_projection(uint32_t seed){ uint64_t s=seed?(uint64_t)seed:0x9E3779B97F4A7C15ULL;
    for(int j=0;j<L2_DIM;j++) for(int k=0;k<BASE_DIM;k++){ s^=s<<13; s^=s>>7; s^=s<<17; Pmat[j][k]=(s&1ULL)?1.f:-1.f; } }
static inline int ent_gate(uint8_t c1,uint8_t c2){ return g_ent_high?(ent_table[c2][c1]>g_ent_thr):(ent_table[c2][c1]<g_ent_thr); }

// same loader as phase47_0_distill.c (0x53454540 / 0x53454543 / 0x53454544)
static int load_weight(const char* path, int tot, float* W, float* B, float* mean, float* std,
                       int want_shared, SiliconEntropyState* see,
                       float* l2clamp, float* l2scale, float* mthr_out, float* a3_out, int* M_out, int* refr_out, int* mode_out) {
    FILE* f=fopen(path,"rb"); if(!f){ fprintf(stderr,"open %s\n",path); return 0; }
    uint32_t magic; fread(&magic,4,1,f); rewind(f);
    int is320=(magic==0x53454544), is_delta=(magic==0x53454543);
    uint32_t hdr4[4]; fread(hdr4,4,4,f); float hf[5]; fread(hf,4,5,f);
    float decay=hf[0], afast=hf[2], fclamp=hf[3];
    uint32_t no=0; fread(&no,4,1,f); int noja=(int)no; float ojab[SEE_N_OJA_MAX*43]; fread(ojab,4,(size_t)noja*43,f);
    uint32_t l2d=0,gt=0,eh=0,ps=0; float al=0,st=0,et=0;
    fread(&l2d,4,1,f); fread(&gt,4,1,f); fread(&al,4,1,f); fread(&st,4,1,f); fread(&et,4,1,f); fread(&eh,4,1,f); fread(&ps,4,1,f);
    g_alpha=al;
    float l2c=0,nbd=1; uint32_t cd=0,dl=0; fread(&l2c,4,1,f); fread(&nbd,4,1,f); fread(&cd,4,1,f); fread(&dl,4,1,f);
    float mx=0; fread(&mx,4,1,f); float ls=1; fread(&ls,4,1,f); float l2cap=0; fread(&l2cap,4,1,f);
    *l2clamp=(l2c>0)?l2c:fclamp; *l2scale=(ls>0)?ls:1.0f;
    if (is_delta){ float rmc=0; uint32_t wg=0; fread(&rmc,4,1,f); fread(&wg,4,1,f);
        uint32_t wm=0,wck=0,wtn=0; float wc=0,ws=0; fread(&wm,4,1,f); fread(&wc,4,1,f); fread(&ws,4,1,f); fread(&wck,4,1,f); fread(&wtn,4,1,f); }
    if (is320){ uint32_t l3d=0,l3m=0,l3K=0,l3M=0,l3R=0; float l3mt=0,l3a=0;
        fread(&l3d,4,1,f); fread(&l3m,4,1,f); fread(&l3K,4,1,f); fread(&l3mt,4,1,f); fread(&l3M,4,1,f); fread(&l3R,4,1,f); fread(&l3a,4,1,f);
        if(mthr_out)*mthr_out=l3mt; if(a3_out)*a3_out=l3a; if(M_out)*M_out=(int)l3M; if(refr_out)*refr_out=(int)l3R; if(mode_out)*mode_out=(int)l3m; }
    if (want_shared){
        size_t tn=(size_t)CLASSES*CLASSES*CLASSES; trigram=malloc(tn*sizeof(float)); fread(trigram,4,tn,f);
        g_ent_thr=et; g_ent_high=(int)eh; gen_projection(ps);
        see_init(see,42,4,decay); see->multiscale_mode=1; see->alpha_fast=afast; see->alpha_mid=0.9f; see->alpha_slow=0.99f;
        see->n_oja=noja; memcpy(see->W_oja,ojab,(size_t)noja*43*sizeof(float)); see->eta_oja=0.0f; see->plastic_blend=1.0f;
    } else {
        fseek(f,(long)CLASSES*CLASSES*CLASSES*4,SEEK_CUR);
    }
    fread(mean,4,tot,f); fread(std,4,tot,f);
    fread(W,4,(size_t)CLASSES*tot,f);
    fread(B,4,CLASSES,f); fclose(f);
    return 1;
}

static inline void norm_feats(const float* raw, int tot, const float* mean, const float* std, float l2clamp, float l2scale, float* out){
    for(int fi=0;fi<tot;fi++){ float x=(raw[fi]-mean[fi])/(std[fi]+1e-8f);
        float cl=(fi<BASE_DIM)?2.0f:l2clamp; if(cl>0){ if(x>cl)x=cl; if(x<-cl)x=-cl; }
        if(fi>=BASE_DIM) x*=l2scale; out[fi]=x; }
}
static inline double logits_bpb(const float* W, const float* B, const float* nf, int tot, uint8_t c1,uint8_t c2,uint8_t tgt){
    const float* tri=&trigram[c2][c1][0]; float lg[CLASSES],mx=-1e30f;
    for(int c=0;c<CLASSES;c++){ lg[c]=B[c]+tri[c]+dot_avx(W+(size_t)c*tot,nf,tot); if(lg[c]>mx)mx=lg[c]; }
    double Z=0; for(int c=0;c<CLASSES;c++) Z+=exp((double)(lg[c]-mx));
    double p=exp((double)(lg[tgt]-mx))/Z; return -log2(p>1e-30?p:1e-30);
}
static inline double trigram_bpb(uint8_t c1,uint8_t c2,uint8_t tgt){
    const float* tri=&trigram[c2][c1][0]; float mx=-1e30f;
    for(int c=0;c<CLASSES;c++) if(tri[c]>mx)mx=tri[c];
    double Z=0; for(int c=0;c<CLASSES;c++) Z+=exp((double)(tri[c]-mx));
    double p=exp((double)(tri[tgt]-mx))/Z; return -log2(p>1e-30?p:1e-30);
}

// ---- probe model: 1-hidden relu MLP, or LINEAR when h==0 (same Adam/batching pipeline) ----
typedef struct { int in,h; float *W1,*b1,*W2,*b2,*mW1,*vW1,*mb1,*vb1,*mW2,*vW2,*mb2,*vb2; int t; } MLP;
static void mlp_init(MLP* m,int in,int h){ m->in=in; m->h=h; m->t=0;
    int hw=(h>0)?h:in;                          // h==0: W2 maps inputs directly (linear probe)
    size_t s1=(h>0)?(size_t)h*in:0, s2=(size_t)CLASSES*hw;
    m->W1=calloc(s1?s1:1,4); m->b1=calloc(h>0?h:1,4); m->W2=calloc(s2,4); m->b2=calloc(CLASSES,4);
    m->mW1=calloc(s1?s1:1,4); m->vW1=calloc(s1?s1:1,4); m->mb1=calloc(h>0?h:1,4); m->vb1=calloc(h>0?h:1,4);
    m->mW2=calloc(s2,4); m->vW2=calloc(s2,4); m->mb2=calloc(CLASSES,4); m->vb2=calloc(CLASSES,4);
    uint64_t r=0x1234567; float sc1=sqrtf(2.0f/in);
    for(size_t i=0;i<s1;i++){ r^=r<<13;r^=r>>7;r^=r<<17; m->W1[i]=sc1*(((r>>11)*(1.0/(1ULL<<53)))*2-1); }
    float sc2=(h>0)?sqrtf(2.0f/h):0.0f;          // linear probe starts at 0 (= pure trigram)
    for(size_t i=0;i<s2;i++){ r^=r<<13;r^=r>>7;r^=r<<17; m->W2[i]=sc2*(((r>>11)*(1.0/(1ULL<<53)))*2-1); }
}
static void mlp_free(MLP* m){ free(m->W1);free(m->b1);free(m->W2);free(m->b2);
    free(m->mW1);free(m->vW1);free(m->mb1);free(m->vb1);free(m->mW2);free(m->vW2);free(m->mb2);free(m->vb2); }
static void mlp_fwd(MLP* m,const float* x,float* hid,float* lg){
    if(m->h>0){
        for(int j=0;j<m->h;j++){ float a=m->b1[j]+dot_avx(&m->W1[(size_t)j*m->in],x,m->in); hid[j]=a>0?a:0; }
        for(int c=0;c<CLASSES;c++) lg[c]=m->b2[c]+dot_avx(&m->W2[(size_t)c*m->h],hid,m->h);
    } else {
        for(int c=0;c<CLASSES;c++) lg[c]=m->b2[c]+dot_avx(&m->W2[(size_t)c*m->in],x,m->in);
    }
}
// eval on a window; feature row i = &X[(i*stride)+off], dim=m->in. Telemetry (nullable):
// o_rms = RMS of centered pre-trigram probe logits; o_ent = output entropy bits; o_maxp = mean max prob.
static double mlp_eval(MLP* m,const float* X,long stride,long off,const uint8_t* tgt,const uint8_t* c1a,const uint8_t* c2a,long n,
                       double* o_rms,double* o_ent,double* o_maxp){
    float* hid=malloc(((m->h>0)?m->h:1)*4); float lg[CLASSES]; double tot=0,arms=0,aent=0,amax=0;
    for(long i=0;i<n;i++){
        mlp_fwd(m,&X[(size_t)i*stride+off],hid,lg);
        if(o_rms){ double mu=0; for(int c=0;c<CLASSES;c++) mu+=lg[c]; mu/=CLASSES;
            double v=0; for(int c=0;c<CLASSES;c++){ double d=lg[c]-mu; v+=d*d; } arms+=sqrt(v/CLASSES); }
        const float* tri=&trigram[c2a[i]][c1a[i]][0]; for(int c=0;c<CLASSES;c++) lg[c]+=tri[c];
        float mx=-1e30f; for(int c=0;c<CLASSES;c++) if(lg[c]>mx)mx=lg[c];
        double Z=0; for(int c=0;c<CLASSES;c++) Z+=exp((double)(lg[c]-mx));
        if(o_ent){ double H=0,pm=0; for(int c=0;c<CLASSES;c++){ double p=exp((double)(lg[c]-mx))/Z; if(p>1e-12)H-=p*log2(p); if(p>pm)pm=p; } aent+=H; amax+=pm; }
        double p=exp((double)(lg[tgt[i]]-mx))/Z; tot+=-log2(p>1e-30?p:1e-30);
    }
    free(hid);
    if(o_rms)*o_rms=arms/n; if(o_ent)*o_ent=aent/n; if(o_maxp)*o_maxp=amax/n;
    return tot/n;
}
// train with Adam, hard CE. idx (nullable) permutes the FEATURE rows only (shuffled-time
// control: features at idx[i], labels/contexts at i). tgt may be a shuffled copy (random-label).
static void mlp_train(MLP* m,const float* X,long stride,long off,const long* idx,
                      const uint8_t* tgt,const uint8_t* c1a,const uint8_t* c2a,long n,int epochs,float lr){
    int H=m->h,in=m->in,lin=(H==0),hw=lin?in:H;
    size_t s1=lin?0:(size_t)H*in, s2=(size_t)CLASSES*hw;
    float* gW1=malloc((s1?s1:1)*4); float* gb1=malloc((H>0?H:1)*4); float* gW2=malloc(s2*4); float* gb2=malloc(CLASSES*4);
    float* hid=malloc((H>0?H:1)*4); float* dh=malloc((H>0?H:1)*4); float lg[CLASSES],eo[CLASSES]; int bs=512;
    for(int ep=0;ep<epochs;ep++){
        memset(gW1,0,(s1?s1:1)*4); memset(gb1,0,(H>0?H:1)*4); memset(gW2,0,s2*4); memset(gb2,0,CLASSES*4); long inb=0;
        for(long i=0;i<n;i++){
            const float* x=&X[(size_t)(idx?idx[i]:i)*stride+off];
            mlp_fwd(m,x,hid,lg);
            const float* tri=&trigram[c2a[i]][c1a[i]][0]; for(int c=0;c<CLASSES;c++) lg[c]+=tri[c];
            float mx=-1e30f; for(int c=0;c<CLASSES;c++) if(lg[c]>mx)mx=lg[c];
            float Z=0; for(int c=0;c<CLASSES;c++){ eo[c]=expf(lg[c]-mx); Z+=eo[c]; }
            for(int c=0;c<CLASSES;c++){ float y=(c==tgt[i])?1.f:0.f; eo[c]=(eo[c]/Z-y)/bs; gb2[c]+=eo[c]; }
            if(lin){
                for(int c=0;c<CLASSES;c++){ float e=eo[c]; float* gw=&gW2[(size_t)c*in]; for(int k=0;k<in;k++) gw[k]+=e*x[k]; }
            } else {
                memset(dh,0,H*4);
                for(int c=0;c<CLASSES;c++){ float e=eo[c]; float* gw=&gW2[(size_t)c*H]; const float* w2=&m->W2[(size_t)c*H];
                    for(int j=0;j<H;j++){ gw[j]+=e*hid[j]; dh[j]+=e*w2[j]; } }
                for(int j=0;j<H;j++) if(hid[j]>0){ gb1[j]+=dh[j]; float* gw=&gW1[(size_t)j*in]; const float g=dh[j];
                    for(int k=0;k<in;k++) gw[k]+=g*x[k]; }
            }
            inb++;
            if(inb==bs || i==n-1){ m->t++; float lt=lr*sqrtf(1-powf(.999f,m->t))/(1-powf(.9f,m->t));
                #define ADAM(P,G,MM,VV,NN) for(size_t z=0;z<(size_t)(NN);z++){ MM[z]=.9f*MM[z]+.1f*G[z]; VV[z]=.999f*VV[z]+.001f*G[z]*G[z]; P[z]-=lt*(MM[z]/(sqrtf(VV[z])+1e-8f)+1e-5f*P[z]); }
                if(!lin){ ADAM(m->W1,gW1,m->mW1,m->vW1,s1); ADAM(m->b1,gb1,m->mb1,m->vb1,H); }
                ADAM(m->W2,gW2,m->mW2,m->vW2,s2); ADAM(m->b2,gb2,m->mb2,m->vb2,CLASSES);
                memset(gW1,0,(s1?s1:1)*4); memset(gb1,0,(H>0?H:1)*4); memset(gW2,0,s2*4); memset(gb2,0,CLASSES*4); inb=0; }
        }
        fprintf(stderr,"    ep %d/%d done\n",ep+1,epochs);
    }
    free(gW1);free(gb1);free(gW2);free(gb2);free(hid);free(dh);
}
// save a probe for the closed-loop generator (magic 0x53454545): {magic,H,off,dim} + weights
static void mlp_save(MLP* m,long off,const char* path){
    FILE* f=fopen(path,"wb"); if(!f){ fprintf(stderr,"save %s failed\n",path); return; }
    uint32_t magic=0x53454545,H=(uint32_t)m->h,o=(uint32_t)off,d=(uint32_t)m->in;
    fwrite(&magic,4,1,f); fwrite(&H,4,1,f); fwrite(&o,4,1,f); fwrite(&d,4,1,f);
    if(m->h>0){ fwrite(m->W1,4,(size_t)m->h*m->in,f); fwrite(m->b1,4,m->h,f); fwrite(m->W2,4,(size_t)CLASSES*m->h,f); }
    else fwrite(m->W2,4,(size_t)CLASSES*m->in,f);
    fwrite(m->b2,4,CLASSES,f); fclose(f);
}

// Teacher-forced extraction (identical dynamics to phase47_0_distill.c): student feats
// X=[SEE|L2_D1] D1-normalized; window baselines trigram-only / D1 / delta / B3.
static void extract_window(SiliconEntropyState* see, long start, long N,
                           float* X, uint8_t* tgt, uint8_t* c1a, uint8_t* c2a,
                           double* oTRI, double* oD1, double* oDE, double* oB3){
    float L2d1[L2_DIM]={0}, L2de[L2_DIM]={0}, L3b3[L3_DIM]={0};
    float pb_d1[BASE_DIM]={0}, pb_de[BASE_DIM]={0}; int lmrun=0,lmrefr=0;
    float feat192[BASE_DIM], fa[BASE_DIM], rawd1[D1_TOT], rawde[D1_TOT], rawb3[B3_TOT], nf[B3_TOT];
    float scale=1.0f/sqrtf((float)BASE_DIM); double btr=0,bd1=0,bde=0,bb3=0;
    see_reset(see); for(long i=0;i<=start+1;i++) see_observe(see,g_data[i]);
    for(long i=0;i<N;i++){
        long g=start+i; uint8_t c2=g_data[g],c1=g_data[g+1],t=g_data[g+2];
        see_extract(see,feat192);
        memcpy(rawd1,feat192,BASE_DIM*4); memcpy(rawd1+BASE_DIM,L2d1,L2_DIM*4);
        memcpy(rawde,feat192,BASE_DIM*4); memcpy(rawde+BASE_DIM,L2de,L2_DIM*4);
        memcpy(rawb3,feat192,BASE_DIM*4); memcpy(rawb3+BASE_DIM,L2d1,L2_DIM*4); memcpy(rawb3+BASE_DIM+L2_DIM,L3b3,L3_DIM*4);
        btr+=trigram_bpb(c1,c2,t);
        norm_feats(rawd1,D1_TOT,md1,sd1,g_l2c_d1,g_ls_d1,nf); bd1+=logits_bpb(&Wd1[0][0],Bd1,nf,D1_TOT,c1,c2,t);
        memcpy(&X[(size_t)i*S_DIM],nf,S_DIM*4);
        norm_feats(rawde,D1_TOT,mde,sde,g_l2c_de,g_ls_de,nf); bde+=logits_bpb(&Wde[0][0],Bde,nf,D1_TOT,c1,c2,t);
        norm_feats(rawb3,B3_TOT,mb3,sb3,g_l2c_b3,g_ls_b3,nf); bb3+=logits_bpb(&Wb3[0][0],Bb3,nf,B3_TOT,c1,c2,t);
        tgt[i]=t; c1a[i]=c1; c2a[i]=c2;
        see_observe(see,t); see_extract(see,fa);
        int gate=ent_gate(c1,c2);
        if(gate){ float src5[BASE_DIM],src1[BASE_DIM];
            for(int k=0;k<BASE_DIM;k++) src5[k]=fa[k]-0.5f*pb_d1[k]; memcpy(pb_d1,fa,BASE_DIM*4);
            for(int k=0;k<BASE_DIM;k++) src1[k]=fa[k]-1.0f*pb_de[k]; memcpy(pb_de,fa,BASE_DIM*4);
            for(int j=0;j<L2_DIM;j++){ float p5=0,p1v=0; const float* pj=Pmat[j]; for(int k=0;k<BASE_DIM;k++){ p5+=pj[k]*src5[k]; p1v+=pj[k]*src1[k]; }
                L2d1[j]=g_alpha*L2d1[j]+(1.0f-g_alpha)*p5*scale; L2de[j]=g_alpha*L2de[j]+(1.0f-g_alpha)*p1v*scale; } }
        double tm=margin_table[c2][c1]; int pfire=(t=='.'||t=='!'||t=='?')?1:0;
        if(tm<g_b3_mthr) lmrun++; else lmrun=0; int lfire=0; if(lmrun==g_b3_M&&lmrefr<=0){ lfire=1; lmrefr=g_b3_refr; } if(lmrefr>0)lmrefr--;
        if(pfire||lfire) for(int j=0;j<L3_DIM;j++) L3b3[j]=g_b3_alpha*L3b3[j]+(1.0f-g_b3_alpha)*fa[j];
    }
    *oTRI=btr/N; *oD1=bd1/N; *oDE=bde/N; *oB3=bb3/N;
}

typedef struct { float* X; uint8_t *tgt,*c1,*c2; double tri,d1,de,b3; long start; } Win;

// run a probe: train on TRAIN window (with optional idx/tgt override), eval on all val windows.
static void run_probe(const char* name,int H,long off,int dim,
                      Win* tr,const long* idx,const uint8_t* tgt_ovr,
                      Win* va,int nval,long N,int EP,const char* savepath){
    fprintf(stderr,"  probe %s (H=%d off=%ld dim=%d)...\n",name,H,off,dim);
    MLP m; mlp_init(&m,dim,H);
    mlp_train(&m,tr->X,S_DIM,off,idx,tgt_ovr?tgt_ovr:tr->tgt,tr->c1,tr->c2,N,EP,0.0005f);
    double v[N_VAL],rms=0,ent=0,mxp=0;
    for(int w=0;w<nval;w++)
        v[w]=mlp_eval(&m,va[w].X,S_DIM,off,va[w].tgt,va[w].c1,va[w].c2,N,
                      w==0?&rms:NULL,w==0?&ent:NULL,w==0?&mxp:NULL);
    printf("PROBE name=%s h=%d off=%ld dim=%d val1=%.4f val2=%.4f val3=%.4f rms1=%.3f ent1=%.3f maxp1=%.3f\n",
           name,H,off,dim,v[0],v[1],v[2],rms,ent,mxp);
    if(savepath){ mlp_save(&m,off,savepath); printf("SAVED name=%s path=%s\n",name,savepath); }
    mlp_free(&m);
}

int main(int argc, char** argv){
    if(argc<6){ fprintf(stderr,"Usage: %s <data> <D1_w> <delta_w> <B3_w> <outprefix> [--len N --epochs E]\n",argv[0]); return 1; }
    setvbuf(stderr,NULL,_IONBF,0); setvbuf(stdout,NULL,_IONBF,0);
    long N=1000000; int EP=6, anchor_only=0;
    for(int i=6;i<argc;i++){ if(!strcmp(argv[i],"--len")&&i+1<argc)N=atol(argv[++i]);
        else if(!strcmp(argv[i],"--epochs")&&i+1<argc)EP=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--anchor-only")) anchor_only=1; }
    FILE* fd=fopen(argv[1],"rb"); if(!fd){fprintf(stderr,"data\n");return 1;} fseek(fd,0,SEEK_END); g_fsz=ftell(fd); fseek(fd,0,SEEK_SET);
    g_data=malloc(g_fsz); fread(g_data,1,g_fsz,fd); fclose(fd);

    SiliconEntropyState see;
    load_weight(argv[2],D1_TOT,&Wd1[0][0],Bd1,md1,sd1,1,&see,&g_l2c_d1,&g_ls_d1,0,0,0,0,0);
    load_weight(argv[3],D1_TOT,&Wde[0][0],Bde,mde,sde,0,0,&g_l2c_de,&g_ls_de,0,0,0,0,0);
    load_weight(argv[4],B3_TOT,&Wb3[0][0],Bb3,mb3,sb3,0,0,&g_l2c_b3,&g_ls_b3,&g_b3_mthr,&g_b3_alpha,&g_b3_M,&g_b3_refr,&g_b3_mode);
    ent_table=malloc(CLASSES*CLASSES*4); margin_table=malloc(CLASSES*CLASSES*4);
    for(int i=0;i<CLASSES;i++) for(int j=0;j<CLASSES;j++){ float m=-1e9f; for(int k=0;k<CLASSES;k++) if(trigram[i][j][k]>m)m=trigram[i][j][k];
        double se=0; for(int k=0;k<CLASSES;k++) se+=exp((double)(trigram[i][j][k]-m));
        double Hh=0,p1=-1,p2=-1; for(int k=0;k<CLASSES;k++){ double p=exp((double)(trigram[i][j][k]-m))/se; if(p>1e-12)Hh-=p*log(p); if(p>p1){p2=p1;p1=p;} else if(p>p2)p2=p; }
        ent_table[i][j]=(float)Hh; margin_table[i][j]=(float)(p1-p2); }

    long tr_start=g_fsz/5;
    long va_start[N_VAL]={ g_fsz/2, (long)(0.65*g_fsz), (long)(0.80*g_fsz) };
    for(int w=0;w<N_VAL;w++) if(va_start[w]+N+3>g_fsz){ fprintf(stderr,"val window %d out of file\n",w+1); return 1; }
    fprintf(stderr,"47.A0: train @%ld, val @%ld/%ld/%ld (N=%ld) ent_thr=%.3f ep=%d\n",
            tr_start,va_start[0],va_start[1],va_start[2],N,g_ent_thr,EP);

    Win tr; Win va[N_VAL];
    tr.X=malloc((size_t)N*S_DIM*4); tr.tgt=malloc(N); tr.c1=malloc(N); tr.c2=malloc(N); tr.start=tr_start;
    for(int w=0;w<N_VAL;w++){ va[w].X=malloc((size_t)N*S_DIM*4); va[w].tgt=malloc(N); va[w].c1=malloc(N); va[w].c2=malloc(N); va[w].start=va_start[w]; }
    if(!tr.X||!va[0].X||!va[1].X||!va[2].X){ fprintf(stderr,"OOM (~%.1f GB)\n",(double)N*(N_VAL+1)*S_DIM*4/1e9); return 1; }
    fprintf(stderr,"extracting train window...\n");
    extract_window(&see,tr_start,N,tr.X,tr.tgt,tr.c1,tr.c2,&tr.tri,&tr.d1,&tr.de,&tr.b3);
    for(int w=0;w<N_VAL;w++){ fprintf(stderr,"extracting val window %d...\n",w+1);
        extract_window(&see,va_start[w],N,va[w].X,va[w].tgt,va[w].c1,va[w].c2,&va[w].tri,&va[w].d1,&va[w].de,&va[w].b3); }

    printf("\n==== 47.A0 sanity gauntlet (train-win -> 3 held-out val-wins, N=%ld, ep=%d) ====\n",N,EP);
    printf("BASE win=train start=%ld tri=%.4f d1=%.4f delta=%.4f b3=%.4f\n",tr.start,tr.tri,tr.d1,tr.de,tr.b3);
    for(int w=0;w<N_VAL;w++)
        printf("BASE win=val%d start=%ld tri=%.4f d1=%.4f delta=%.4f b3=%.4f\n",w+1,va[w].start,va[w].tri,va[w].d1,va[w].de,va[w].b3);

    // ---- 47.A0b anchor repair: D1's OWN readout pushed through the probe eval path. ----
    // frozenD1 must reproduce BASE d1 (proves features/normalization/trigram/offset are
    // aligned); linD1init = train from D1's weights (does Adam-on-1M improve or DEGRADE the
    // globally-trained solution? degrade => window overfit, the "linear ~= D1 in 6ep" anchor
    // criterion was miscalibrated, NOT a feature bug).
    { MLP m; mlp_init(&m,S_DIM,0);
      for(int c=0;c<CLASSES;c++){ memcpy(&m.W2[(size_t)c*S_DIM],&Wd1[c][0],S_DIM*4); m.b2[c]=Bd1[c]; }
      double v[N_VAL],rms=0,ent=0,mxp=0;
      for(int w=0;w<N_VAL;w++) v[w]=mlp_eval(&m,va[w].X,S_DIM,0,va[w].tgt,va[w].c1,va[w].c2,N,
                                             w==0?&rms:NULL,w==0?&ent:NULL,w==0?&mxp:NULL);
      printf("PROBE name=frozenD1 h=0 off=0 dim=%d val1=%.4f val2=%.4f val3=%.4f rms1=%.3f ent1=%.3f maxp1=%.3f\n",
             S_DIM,v[0],v[1],v[2],rms,ent,mxp);
      fprintf(stderr,"  probe linD1init (train %d ep from D1 weights)...\n",EP);
      mlp_train(&m,tr.X,S_DIM,0,NULL,tr.tgt,tr.c1,tr.c2,N,EP,0.0005f);
      for(int w=0;w<N_VAL;w++) v[w]=mlp_eval(&m,va[w].X,S_DIM,0,va[w].tgt,va[w].c1,va[w].c2,N,
                                             w==0?&rms:NULL,w==0?&ent:NULL,w==0?&mxp:NULL);
      printf("PROBE name=linD1init h=0 off=0 dim=%d val1=%.4f val2=%.4f val3=%.4f rms1=%.3f ent1=%.3f maxp1=%.3f\n",
             S_DIM,v[0],v[1],v[2],rms,ent,mxp);
      mlp_free(&m); }

    // [2] linear probe from scratch, same pipeline (vs frozenD1/linD1init: training budget story)
    run_probe("linear",0,0,S_DIM,&tr,NULL,NULL,va,N_VAL,N,EP,NULL);

    if (anchor_only){
        // longer-budget scratch linear: does 3x epochs close the gap to D1? (hypothesis: the
        // anchor criterion was miscalibrated, a window-trained linear can't reach the
        // corpus-trained D1 in few epochs - or overfits the window and never will)
        run_probe("linearLong",0,0,S_DIM,&tr,NULL,NULL,va,N_VAL,N,3*EP,NULL);
        printf("\nAnchor-only run: frozenD1 must == BASE d1 (probe path integrity); linD1init vs\n");
        printf("frozenD1 = what window-Adam does to the global solution; linear/linearLong vs d1 =\n");
        printf("scratch budget story. Ladder/controls/ablations: reuse the previous full gauntlet.\n");
        return 0;
    }

    // [1]+[5] width ladder (H=128 doubles as the multi-window reproduction of 47.0's 2.09)
    char sp[512];
    int ladder[4]={16,32,64,128};
    for(int li=0;li<4;li++){
        char nm[32]; snprintf(nm,sizeof nm,"mlpH%d",ladder[li]);
        snprintf(sp,sizeof sp,"%s_mlpH%d.bin",argv[5],ladder[li]);
        run_probe(nm,ladder[li],0,S_DIM,&tr,NULL,NULL,va,N_VAL,N,EP,sp);
    }

    // [3] controls at H=128: random-label / shuffled-time (must show NO gain vs D1)
    uint8_t* tsh=malloc(N); memcpy(tsh,tr.tgt,N);
    uint64_t r=0xC0FFEE123456ULL;
    for(long i=N-1;i>0;i--){ r^=r<<13;r^=r>>7;r^=r<<17; long j=(long)(r%(uint64_t)(i+1)); uint8_t t=tsh[i]; tsh[i]=tsh[j]; tsh[j]=t; }
    run_probe("randlabel",128,0,S_DIM,&tr,NULL,tsh,va,N_VAL,N,EP,NULL); free(tsh);
    long* idx=malloc(N*sizeof(long)); for(long i=0;i<N;i++) idx[i]=i;
    r=0xBADC0DE987654ULL;
    for(long i=N-1;i>0;i--){ r^=r<<13;r^=r>>7;r^=r<<17; long j=(long)(r%(uint64_t)(i+1)); long t=idx[i]; idx[i]=idx[j]; idx[j]=t; }
    run_probe("shuftime",128,0,S_DIM,&tr,idx,NULL,va,N_VAL,N,EP,NULL); free(idx);

    // [4] ablations at H=64 (mlpH64 above is the SEE+L2 reference at the same width)
    run_probe("ablSEE",64,0,BASE_DIM,&tr,NULL,NULL,va,N_VAL,N,EP,NULL);
    run_probe("ablL2",64,BASE_DIM,L2_DIM,&tr,NULL,NULL,va,N_VAL,N,EP,NULL);

    printf("\nReading: [1] mlpH128 must beat d1 on ALL of val1/2/3 (reproduces 47.0); [2] linear must\n");
    printf("land ~= d1 (same-pipeline sanity); [3] randlabel/shuftime must NOT beat d1 (leak control);\n");
    printf("[4] ablSEE vs ablL2 vs mlpH64 locates the nonlinear gain; [5] ladder = smallest admissible\n");
    printf("readout. Saved 0x53454545 probes -> word-gate via phase47_generator (the real test).\n");
    return 0;
}
