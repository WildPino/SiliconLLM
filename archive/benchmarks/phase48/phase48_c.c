// Phase 48.C - Multiplicative-Dynamics Reservoir (TF probe)
//
// 48.B was a pre-registered negative but diagnostic: BILIN/WAVE32/MULTIBW all bought ~0.01,
// none cleared 0.015-on-3-windows-with-clean-controls, and BILIN_sp (shuffled pairing) kept
// ~85% of BILIN's gain -> the static fast.slow product is a generic quadratic feature, NOT
// gating. Three different classes, one plateau = the ceiling of the STATIC map. 48.C moves the
// multiplication from the READOUT into the DYNAMICS: the product gates the recurrent
// transition and is integrated over time, so the readout sees the accumulated gated history
// instead of an instantaneous product. Reservoir-style: gating with FIXED random projections,
// only the readout is trained (no backprop into the substrate).
//
// Base frozen = [SEE 192 | L2_D1 64 | armB cos-bands 256] = 512 (the same armB as 48.B,
// SEED_A/GAMMA_A/DA=128; armB baseline must reproduce 2.2023/2.1839/2.1980). New reservoir
// bands append to the 512. Source = L0norm (the 64 L0 dims standardized with md1/sd1, clamp
// 2.0 - includes all 32 wave dims, so WAVE32 is covered for free):
//   drive_t = A . L0norm_t            (A 64x64 gaussian fixed)
//   ctx_t   = EMA_0.99(L0norm) to t-1 (slow context)
//   g_t     = hardsig(GATE_GAIN*(B.ctx_t)), hardsig(x)=clamp(0.5+0.25x,0,1) in [0,1]  (B 64x64)
//   r^d_t   = d * r^d_{t-1} + drive_t (x) g_t,   d in {0.90, 0.99}
//   feature = [r^0.90 || r^0.99] = 128 -> probe dim 640. Controls store their own banks:
//     LIN_dyn : g_t == 0.5 (constant gate) - isolates "the multiplication" from "more dims".
//     DYN_sp  : drive[k] (x) g[perm[k]] (channel-permuted gate pairing) - gating semantics.
//     DYN_st  : DYN bands gathered from time-permuted rows (leak guard).
// Causality (zero-leak, like L2_D1/armB): row i uses r over <i; update ctx and r AFTER writing
// the row; drive/ctx from the pre-observation L0; cold-start r=0, ctx=0.
//
// PRE-REGISTERED (in the ps1): anchor==BASE; DYN beats armB >=0.015 on all 3 windows; DYN beats
// LIN_dyn >=0.008 on all 3; DYN_st does not beat armB by >0.005; (informative) DYN-DYN_sp margin.
// PASS -> DYN goes to 48.C.A (DAgger closed-loop, frozen 47 harness). TF != generative.
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase38-42/phase48_c.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase48_c.exe -lm -I .
// Run:
//   bin/phase48_c.exe <data> <D1_w> <outprefix> [--len N --epochs E --hidden H]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include "src/silicon_entropy.h"

#define CLASSES   256
#define BASE_DIM  SEE_FEATURE_DIM        // 192
#define L0_DIM    SEE_L0_DIM             // 64
#define L2_DIM    64
#define D1_TOT    (BASE_DIM + L2_DIM)    // 256
#define N_TS      2
#define DA        128                    // armB cos dims (gamma 0.25) - same as 48.B
#define ARMB_BANDS (N_TS*DA)             // 256
#define R_DIM     64                     // reservoir bank width (= L0_DIM)
#define N_BANK    2                      // decays {0.90, 0.99}
#define DYN_BANDS (N_BANK*R_DIM)         // 128
#define BASE512   (D1_TOT+ARMB_BANDS)    // 512 = the armB baseline feature
#define OFF_ARMB  (D1_TOT)               // 256
#define OFF_DYN   (BASE512)              // 512
#define OFF_LIN   (OFF_DYN+DYN_BANDS)    // 640
#define OFF_SP    (OFF_LIN+DYN_BANDS)    // 768
#define X_DIM     (OFF_SP+DYN_BANDS)     // 896 (storage row; each probe gathers base512 + one 128 bank)
#define GAMMA_A   0.25f
#define CTX_EMA   0.99f
#define GATE_GAIN 4.0f                   // hardsig slope; smoke verifies the gate is live
#define SEED_A    0x48B2EC0DEULL         // == 48.0/48.B armB (reproduces the baseline exactly)
#define SEED_DRV  0x48C1D7733ULL         // drive matrix A
#define SEED_GAT  0x48C2A9145ULL         // gate matrix B
#define SEED_SPP  0x48C3B6271ULL         // DYN_sp channel permutation
#define MAGIC_DYN 0x53454549u
#define N_VAL     3

static const float DECAY[N_BANK] = { 0.90f, 0.99f };
static const float TS_ALPHA[N_TS] = { 0.90f, 0.99f };

static float Pmat[L2_DIM][BASE_DIM];
static float OmA[DA][L0_DIM], BvA[DA];
static float MatA[R_DIM][L0_DIM], MatB[R_DIM][L0_DIM];
static int   spperm[R_DIM];
static float bmean[X_DIM], bstd[X_DIM];   // per-dim train stats for the reservoir band columns [OFF_DYN..X_DIM]
static float (*trigram)[CLASSES][CLASSES];
static float (*ent_table)[CLASSES];
static uint8_t* g_data; static long g_fsz;
static float g_ent_thr; static int g_ent_high=1;
static float Wd1[CLASSES][D1_TOT], Bd1[CLASSES], md1[D1_TOT], sd1[D1_TOT];
static float g_alpha=0.99f, g_l2c_d1=2.0f, g_ls_d1=0.5f;

static inline float dot_avx(const float* w, const float* f, int n){ __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(&w[i]),_mm256_loadu_ps(&f[i]),s);
    float o[8]; _mm256_storeu_ps(o,s); float r=o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; for(;i<n;i++) r+=w[i]*f[i]; return r; }
static void gen_projection(uint32_t seed){ uint64_t s=seed?(uint64_t)seed:0x9E3779B97F4A7C15ULL;
    for(int j=0;j<L2_DIM;j++) for(int k=0;k<BASE_DIM;k++){ s^=s<<13; s^=s>>7; s^=s<<17; Pmat[j][k]=(s&1ULL)?1.f:-1.f; } }
static inline double xs_u01(uint64_t* s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return (double)((*s>>11)&0x1FFFFFFFFFFFFFULL)/(double)(1ULL<<53); }
static void gen_lift(float Om[][L0_DIM], float* Bv, int D, uint64_t seed){ uint64_t s=seed?seed:0xABCDEF12345ULL;
    for(int d=0;d<D;d++){
        for(int k=0;k<L0_DIM;k++){ double u1=xs_u01(&s); if(u1<1e-12) u1=1e-12; double u2=xs_u01(&s);
            Om[d][k]=(float)(sqrt(-2.0*log(u1))*cos(6.283185307179586*u2)); }
        Bv[d]=(float)(6.283185307179586*xs_u01(&s)); } }
static void gen_mat(float M[][L0_DIM], int rows, uint64_t seed){ uint64_t s=seed?seed:0xCAFEBABEULL;
    for(int r=0;r<rows;r++) for(int k=0;k<L0_DIM;k++){ double u1=xs_u01(&s); if(u1<1e-12)u1=1e-12; double u2=xs_u01(&s);
        M[r][k]=(float)(sqrt(-2.0*log(u1))*cos(6.283185307179586*u2)); } }
static void gen_perm(int* p, int n, uint64_t seed){ for(int i=0;i<n;i++) p[i]=i; uint64_t r=seed?seed:0x1234ULL;
    for(int i=n-1;i>0;i--){ r^=r<<13;r^=r>>7;r^=r<<17; int j=(int)(r%(uint64_t)(i+1)); int t=p[i];p[i]=p[j];p[j]=t; } }
static inline int ent_gate(uint8_t c1,uint8_t c2){ return g_ent_high?(ent_table[c2][c1]>g_ent_thr):(ent_table[c2][c1]<g_ent_thr); }

static int load_d1(const char* path, SiliconEntropyState* see){
    FILE* f=fopen(path,"rb"); if(!f){ fprintf(stderr,"open %s\n",path); return 0; }
    uint32_t magic; fread(&magic,4,1,f); rewind(f);
    if (magic!=0x53454540){ fprintf(stderr,"expected D1 0x53454540, got 0x%08x\n",magic); return 0; }
    uint32_t hdr4[4]; fread(hdr4,4,4,f); float hf[5]; fread(hf,4,5,f);
    float decay=hf[0], afast=hf[2], fclamp=hf[3];
    uint32_t no=0; fread(&no,4,1,f); int noja=(int)no; float ojab[SEE_N_OJA_MAX*43]; fread(ojab,4,(size_t)noja*43,f);
    uint32_t l2d=0,gt=0,eh=0,ps=0; float al=0,st=0,et=0;
    fread(&l2d,4,1,f); fread(&gt,4,1,f); fread(&al,4,1,f); fread(&st,4,1,f); fread(&et,4,1,f); fread(&eh,4,1,f); fread(&ps,4,1,f);
    g_alpha=al;
    float l2c=0,nbd=1; uint32_t cd=0,dl=0; fread(&l2c,4,1,f); fread(&nbd,4,1,f); fread(&cd,4,1,f); fread(&dl,4,1,f);
    float mx=0; fread(&mx,4,1,f); float ls=1; fread(&ls,4,1,f); float l2cap=0; fread(&l2cap,4,1,f);
    g_l2c_d1=(l2c>0)?l2c:fclamp; g_ls_d1=(ls>0)?ls:1.0f;
    size_t tn=(size_t)CLASSES*CLASSES*CLASSES; trigram=malloc(tn*sizeof(float)); fread(trigram,4,tn,f);
    g_ent_thr=et; g_ent_high=(int)eh; gen_projection(ps);
    see_init(see,42,4,decay); see->multiscale_mode=1; see->alpha_fast=afast; see->alpha_mid=0.9f; see->alpha_slow=0.99f;
    see->n_oja=noja; memcpy(see->W_oja,ojab,(size_t)noja*43*sizeof(float)); see->eta_oja=0.0f; see->plastic_blend=1.0f;
    fread(md1,4,D1_TOT,f); fread(sd1,4,D1_TOT,f);
    fread(Wd1,4,(size_t)CLASSES*D1_TOT,f); fread(Bd1,4,CLASSES,f); fclose(f);
    return 1;
}
static inline void norm_feats(const float* raw, float* out){
    for(int fi=0;fi<D1_TOT;fi++){ float x=(raw[fi]-md1[fi])/(sd1[fi]+1e-8f);
        float cl=(fi<BASE_DIM)?2.0f:g_l2c_d1; if(cl>0){ if(x>cl)x=cl; if(x<-cl)x=-cl; }
        if(fi>=BASE_DIM) x*=g_ls_d1; out[fi]=x; }
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

typedef struct { long off; int len; } Seg;
static inline int gather_row(const float* X,long i,const Seg* segs,int nseg,const long* perm,long permFromOff,float* out){
    int d=0;
    for(int s=0;s<nseg;s++){
        long row=(perm && segs[s].off>=permFromOff)?perm[i]:i;
        memcpy(out+d,&X[(size_t)row*X_DIM+segs[s].off],(size_t)segs[s].len*4); d+=segs[s].len;
    }
    return d;
}

typedef struct { int in,h; float *W1,*b1,*W2,*b2,*mW1,*vW1,*mb1,*vb1,*mW2,*vW2,*mb2,*vb2; int t; } MLP;
static void mlp_init(MLP* m,int in,int h){ m->in=in; m->h=h; m->t=0;
    int hw=(h>0)?h:in; size_t s1=(h>0)?(size_t)h*in:0, s2=(size_t)CLASSES*hw;
    m->W1=calloc(s1?s1:1,4); m->b1=calloc(h>0?h:1,4); m->W2=calloc(s2,4); m->b2=calloc(CLASSES,4);
    m->mW1=calloc(s1?s1:1,4); m->vW1=calloc(s1?s1:1,4); m->mb1=calloc(h>0?h:1,4); m->vb1=calloc(h>0?h:1,4);
    m->mW2=calloc(s2,4); m->vW2=calloc(s2,4); m->mb2=calloc(CLASSES,4); m->vb2=calloc(CLASSES,4);
    uint64_t r=0x1234567; float sc1=sqrtf(2.0f/in);
    for(size_t i=0;i<s1;i++){ r^=r<<13;r^=r>>7;r^=r<<17; m->W1[i]=sc1*(((r>>11)*(1.0/(1ULL<<53)))*2-1); }
    float sc2=(h>0)?sqrtf(2.0f/h):0.0f;
    for(size_t i=0;i<s2;i++){ r^=r<<13;r^=r>>7;r^=r<<17; m->W2[i]=sc2*(((r>>11)*(1.0/(1ULL<<53)))*2-1); }
}
static void mlp_free(MLP* m){ free(m->W1);free(m->b1);free(m->W2);free(m->b2);
    free(m->mW1);free(m->vW1);free(m->mb1);free(m->vb1);free(m->mW2);free(m->vW2);free(m->mb2);free(m->vb2); }
static void mlp_fwd(MLP* m,const float* x,float* hid,float* lg){
    if(m->h>0){
        for(int j=0;j<m->h;j++){ float a=m->b1[j]+dot_avx(&m->W1[(size_t)j*m->in],x,m->in); hid[j]=a>0?a:0; }
        for(int c=0;c<CLASSES;c++) lg[c]=m->b2[c]+dot_avx(&m->W2[(size_t)c*m->h],hid,m->h);
    } else { for(int c=0;c<CLASSES;c++) lg[c]=m->b2[c]+dot_avx(&m->W2[(size_t)c*m->in],x,m->in); }
}
static double mlp_eval(MLP* m,const float* X,const Seg* segs,int nseg,const long* perm,long permFromOff,
                       const uint8_t* tgt,const uint8_t* c1a,const uint8_t* c2a,long n,double* o_rms,double* o_ent,double* o_maxp){
    float* hid=malloc(((m->h>0)?m->h:1)*4); float* xb=malloc(X_DIM*4); float lg[CLASSES];
    double tot=0,arms=0,aent=0,amax=0;
    for(long i=0;i<n;i++){
        gather_row(X,i,segs,nseg,perm,permFromOff,xb);
        mlp_fwd(m,xb,hid,lg);
        if(o_rms){ double mu=0; for(int c=0;c<CLASSES;c++) mu+=lg[c]; mu/=CLASSES;
            double v=0; for(int c=0;c<CLASSES;c++){ double d=lg[c]-mu; v+=d*d; } arms+=sqrt(v/CLASSES); }
        const float* tri=&trigram[c2a[i]][c1a[i]][0]; for(int c=0;c<CLASSES;c++) lg[c]+=tri[c];
        float mx=-1e30f; for(int c=0;c<CLASSES;c++) if(lg[c]>mx)mx=lg[c];
        double Z=0; for(int c=0;c<CLASSES;c++) Z+=exp((double)(lg[c]-mx));
        if(o_ent){ double H=0,pm=0; for(int c=0;c<CLASSES;c++){ double p=exp((double)(lg[c]-mx))/Z; if(p>1e-12)H-=p*log2(p); if(p>pm)pm=p; } aent+=H; amax+=pm; }
        double p=exp((double)(lg[tgt[i]]-mx))/Z; tot+=-log2(p>1e-30?p:1e-30);
    }
    free(hid); free(xb);
    if(o_rms)*o_rms=arms/n; if(o_ent)*o_ent=aent/n; if(o_maxp)*o_maxp=amax/n;
    return tot/n;
}
static void mlp_train(MLP* m,const float* X,const Seg* segs,int nseg,const long* perm,long permFromOff,
                      const uint8_t* tgt,const uint8_t* c1a,const uint8_t* c2a,long n,int epochs,float lr){
    int H=m->h,in=m->in; size_t s1=(size_t)H*in, s2=(size_t)CLASSES*H;
    float* gW1=malloc(s1*4); float* gb1=malloc(H*4); float* gW2=malloc(s2*4); float* gb2=malloc(CLASSES*4);
    float* hid=malloc(H*4); float* dh=malloc(H*4); float* xb=malloc(X_DIM*4); float lg[CLASSES],eo[CLASSES]; int bs=512;
    for(int ep=0;ep<epochs;ep++){
        memset(gW1,0,s1*4); memset(gb1,0,H*4); memset(gW2,0,s2*4); memset(gb2,0,CLASSES*4); long inb=0;
        for(long i=0;i<n;i++){
            gather_row(X,i,segs,nseg,perm,permFromOff,xb); const float* x=xb;
            mlp_fwd(m,x,hid,lg);
            const float* tri=&trigram[c2a[i]][c1a[i]][0]; for(int c=0;c<CLASSES;c++) lg[c]+=tri[c];
            float mx=-1e30f; for(int c=0;c<CLASSES;c++) if(lg[c]>mx)mx=lg[c];
            float Z=0; for(int c=0;c<CLASSES;c++){ eo[c]=expf(lg[c]-mx); Z+=eo[c]; }
            for(int c=0;c<CLASSES;c++){ float y=(c==tgt[i])?1.f:0.f; eo[c]=(eo[c]/Z-y)/bs; gb2[c]+=eo[c]; }
            memset(dh,0,H*4);
            for(int c=0;c<CLASSES;c++){ float e=eo[c]; float* gw=&gW2[(size_t)c*H]; const float* w2=&m->W2[(size_t)c*H];
                for(int j=0;j<H;j++){ gw[j]+=e*hid[j]; dh[j]+=e*w2[j]; } }
            for(int j=0;j<H;j++) if(hid[j]>0){ gb1[j]+=dh[j]; float* gw=&gW1[(size_t)j*in]; const float g=dh[j];
                for(int k=0;k<in;k++) gw[k]+=g*x[k]; }
            inb++;
            if(inb==bs || i==n-1){ m->t++; float lt=lr*sqrtf(1-powf(.999f,m->t))/(1-powf(.9f,m->t));
                #define ADAM(P,G,MM,VV,NN) for(size_t z=0;z<(size_t)(NN);z++){ MM[z]=.9f*MM[z]+.1f*G[z]; VV[z]=.999f*VV[z]+.001f*G[z]*G[z]; P[z]-=lt*(MM[z]/(sqrtf(VV[z])+1e-8f)+1e-5f*P[z]); }
                ADAM(m->W1,gW1,m->mW1,m->vW1,s1); ADAM(m->b1,gb1,m->mb1,m->vb1,H);
                ADAM(m->W2,gW2,m->mW2,m->vW2,s2); ADAM(m->b2,gb2,m->mb2,m->vb2,CLASSES);
                memset(gW1,0,s1*4); memset(gb1,0,H*4); memset(gW2,0,s2*4); memset(gb2,0,CLASSES*4); inb=0; }
        }
        fprintf(stderr,"    ep %d/%d done\n",ep+1,epochs);
    }
    free(gW1);free(gb1);free(gW2);free(gb2);free(hid);free(dh);free(xb);
}
// save the DYN probe (magic 0x53454549) so 48.C.A can rebuild the reservoir closed-loop.
static void mlp_save_dyn(MLP* m,const char* path){
    FILE* f=fopen(path,"wb"); if(!f){ fprintf(stderr,"save %s failed\n",path); return; }
    uint32_t magic=MAGIC_DYN,H=(uint32_t)m->h,dim=(uint32_t)m->in,base=BASE512,d_r=DYN_BANDS,nbank=N_BANK,dexp=DA,gtype=1;
    float gamma_a=GAMMA_A,ctx_ema=CTX_EMA,gate_gain=GATE_GAIN,drive_ema=0.0f;
    uint64_t sa=SEED_A,sd=SEED_DRV,sg=SEED_GAT;
    fwrite(&magic,4,1,f); fwrite(&H,4,1,f); fwrite(&dim,4,1,f); fwrite(&base,4,1,f);
    fwrite(&d_r,4,1,f); fwrite(&nbank,4,1,f); fwrite(&dexp,4,1,f); fwrite(&gtype,4,1,f);
    for(int b=0;b<N_BANK;b++){ float d=DECAY[b]; fwrite(&d,4,1,f); }
    fwrite(&gamma_a,4,1,f); fwrite(&ctx_ema,4,1,f); fwrite(&gate_gain,4,1,f); fwrite(&drive_ema,4,1,f);
    fwrite(&sa,8,1,f); fwrite(&sd,8,1,f); fwrite(&sg,8,1,f);
    fwrite(&bmean[OFF_DYN],4,DYN_BANDS,f); fwrite(&bstd[OFF_DYN],4,DYN_BANDS,f);  // band standardization (frozen train stats)
    fwrite(m->W1,4,(size_t)m->h*m->in,f); fwrite(m->b1,4,m->h,f); fwrite(m->W2,4,(size_t)CLASSES*m->h,f);
    fwrite(m->b2,4,CLASSES,f); fclose(f);
}

// Teacher-forced extraction. Row: [nf 256 | armB 256 | r_dyn 128 | r_lin 128 | r_sp 128] = 896.
// All reservoir banks causal (row i holds <i; state updated AFTER the row). Gate stats over the
// window via o_gate (mean,std,plo,phi) when non-NULL (train window).
static void extract_window(SiliconEntropyState* see, long start, long N,
                           float* X, uint8_t* tgt, uint8_t* c1a, uint8_t* c2a, double* oTRI, double* oD1, double* o_gate){
    float L2d1[L2_DIM]={0}, pb_d1[BASE_DIM]={0};
    float eA[N_TS][DA]; memset(eA,0,sizeof eA);
    float ctx[L0_DIM]={0};
    float rD[N_BANK][R_DIM], rL[N_BANK][R_DIM], rS[N_BANK][R_DIM];
    memset(rD,0,sizeof rD); memset(rL,0,sizeof rL); memset(rS,0,sizeof rS);
    float feat192[BASE_DIM], fa[BASE_DIM], rawd1[D1_TOT], nf[D1_TOT], l0n[L0_DIM];
    float drive[R_DIM], gate[R_DIM];
    float scale=1.0f/sqrtf((float)BASE_DIM); double btr=0,bd1=0;
    double gsum=0,gM2=0; long gn=0,glo=0,ghi=0;
    see_reset(see); for(long i=0;i<=start+1;i++) see_observe(see,g_data[i]);
    for(long i=0;i<N;i++){
        long g=start+i; uint8_t c2=g_data[g],c1=g_data[g+1],t=g_data[g+2];
        see_extract(see,feat192);
        memcpy(rawd1,feat192,BASE_DIM*4); memcpy(rawd1+BASE_DIM,L2d1,L2_DIM*4);
        btr+=trigram_bpb(c1,c2,t);
        norm_feats(rawd1,nf); bd1+=logits_bpb(&Wd1[0][0],Bd1,nf,D1_TOT,c1,c2,t);
        float* row=&X[(size_t)i*X_DIM];
        memcpy(row,nf,D1_TOT*4);
        for(int k=0;k<L0_DIM;k++){ float x=(feat192[k]-md1[k])/(sd1[k]+1e-8f); if(x>2.f)x=2.f; if(x<-2.f)x=-2.f; l0n[k]=x; }
        // --- write current bands (causal: integration over <i) ---
        for(int ts=0;ts<N_TS;ts++) memcpy(row+OFF_ARMB+ts*DA,eA[ts],DA*4);
        for(int b=0;b<N_BANK;b++){ memcpy(row+OFF_DYN+b*R_DIM,rD[b],R_DIM*4);
            memcpy(row+OFF_LIN+b*R_DIM,rL[b],R_DIM*4); memcpy(row+OFF_SP+b*R_DIM,rS[b],R_DIM*4); }
        tgt[i]=t; c1a[i]=c1; c2a[i]=c2;
        // --- updates (after row write) ---
        for(int d=0;d<DA;d++){ float z=BvA[d]+GAMMA_A*dot_avx(OmA[d],l0n,L0_DIM); float cz=cosf(z);
            for(int ts=0;ts<N_TS;ts++){ float a=TS_ALPHA[ts]; eA[ts][d]=a*eA[ts][d]+(1.0f-a)*cz; } }
        for(int k=0;k<R_DIM;k++){ drive[k]=dot_avx(MatA[k],l0n,L0_DIM);
            float pre=dot_avx(MatB[k],ctx,L0_DIM); float gg=0.5f+0.25f*GATE_GAIN*pre; if(gg<0)gg=0; if(gg>1)gg=1; gate[k]=gg;
            if(o_gate){ gsum+=gg; gM2+=(double)gg*gg; gn++; if(gg<0.02f)glo++; if(gg>0.98f)ghi++; } }
        for(int b=0;b<N_BANK;b++){ float dec=DECAY[b]; for(int k=0;k<R_DIM;k++){
            rD[b][k]=dec*rD[b][k]+drive[k]*gate[k];
            rL[b][k]=dec*rL[b][k]+drive[k]*0.5f;
            rS[b][k]=dec*rS[b][k]+drive[k]*gate[spperm[k]]; } }
        for(int k=0;k<L0_DIM;k++) ctx[k]=CTX_EMA*ctx[k]+(1.0f-CTX_EMA)*l0n[k];
        see_observe(see,t); see_extract(see,fa);
        if(ent_gate(c1,c2)){ float src5[BASE_DIM];
            for(int k=0;k<BASE_DIM;k++) src5[k]=fa[k]-0.5f*pb_d1[k]; memcpy(pb_d1,fa,BASE_DIM*4);
            for(int j=0;j<L2_DIM;j++){ float p5=0; const float* pj=Pmat[j]; for(int k=0;k<BASE_DIM;k++) p5+=pj[k]*src5[k];
                L2d1[j]=g_alpha*L2d1[j]+(1.0f-g_alpha)*p5*scale; } }
    }
    *oTRI=btr/N; *oD1=bd1/N;
    if(o_gate){ double mu=gsum/(gn>0?gn:1); double var=gM2/(gn>0?gn:1)-mu*mu; if(var<0)var=0;
        o_gate[0]=mu; o_gate[1]=sqrt(var); o_gate[2]=100.0*glo/(gn>0?gn:1); o_gate[3]=100.0*ghi/(gn>0?gn:1); }
}

typedef struct { float* X; uint8_t *tgt,*c1,*c2; double tri,d1; long start; long* perm; } Win;
static long* make_perm(long n,uint64_t seed){
    long* p=malloc(n*sizeof(long)); for(long i=0;i<n;i++) p[i]=i; uint64_t r=seed;
    for(long i=n-1;i>0;i--){ r^=r<<13;r^=r>>7;r^=r<<17; long j=(long)(r%(uint64_t)(i+1)); long t=p[i];p[i]=p[j];p[j]=t; } return p; }

static void run_probe(const char* name,int H,const Seg* segs,int nseg,const long* perm,long permFromOff,
                      Win* tr,Win* va,long N,int EP,const char* savepath){
    int dim=0; for(int s=0;s<nseg;s++) dim+=segs[s].len;
    fprintf(stderr,"  probe %s (H=%d dim=%d)...\n",name,H,dim);
    MLP m; mlp_init(&m,dim,H);
    mlp_train(&m,tr->X,segs,nseg,perm?tr->perm:NULL,permFromOff,tr->tgt,tr->c1,tr->c2,N,EP,0.0005f);
    double v[N_VAL],rms=0,ent=0,mxp=0;
    for(int w=0;w<N_VAL;w++)
        v[w]=mlp_eval(&m,va[w].X,segs,nseg,perm?va[w].perm:NULL,permFromOff,va[w].tgt,va[w].c1,va[w].c2,N,
                      w==0?&rms:NULL,w==0?&ent:NULL,w==0?&mxp:NULL);
    printf("PROBE name=%s h=%d dim=%d val1=%.4f val2=%.4f val3=%.4f rms1=%.3f ent1=%.3f maxp1=%.3f\n",
           name,H,dim,v[0],v[1],v[2],rms,ent,mxp);
    if(savepath){ mlp_save_dyn(&m,savepath); printf("SAVED name=%s path=%s\n",name,savepath); }
    mlp_free(&m);
}

int main(int argc, char** argv){
    if(argc<4){ fprintf(stderr,"Usage: %s <data> <D1_w> <outprefix> [--len N --epochs E --hidden H]\n",argv[0]); return 1; }
    setvbuf(stderr,NULL,_IONBF,0); setvbuf(stdout,NULL,_IONBF,0);
    long N=1000000; int EP=6, H=32;
    for(int i=4;i<argc;i++){ if(!strcmp(argv[i],"--len")&&i+1<argc)N=atol(argv[++i]);
        else if(!strcmp(argv[i],"--epochs")&&i+1<argc)EP=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--hidden")&&i+1<argc)H=atoi(argv[++i]); }
    FILE* fd=fopen(argv[1],"rb"); if(!fd){fprintf(stderr,"data\n");return 1;} fseek(fd,0,SEEK_END); g_fsz=ftell(fd); fseek(fd,0,SEEK_SET);
    g_data=malloc(g_fsz); fread(g_data,1,g_fsz,fd); fclose(fd);

    SiliconEntropyState see;
    if(!load_d1(argv[2],&see)) return 1;
    gen_lift(OmA,BvA,DA,SEED_A); gen_mat(MatA,R_DIM,SEED_DRV); gen_mat(MatB,R_DIM,SEED_GAT); gen_perm(spperm,R_DIM,SEED_SPP);
    ent_table=malloc(CLASSES*CLASSES*4);
    for(int i=0;i<CLASSES;i++) for(int j=0;j<CLASSES;j++){ float m=-1e9f; for(int k=0;k<CLASSES;k++) if(trigram[i][j][k]>m)m=trigram[i][j][k];
        double se=0; for(int k=0;k<CLASSES;k++) se+=exp((double)(trigram[i][j][k]-m));
        double Hh=0; for(int k=0;k<CLASSES;k++){ double p=exp((double)(trigram[i][j][k]-m))/se; if(p>1e-12)Hh-=p*log(p); }
        ent_table[i][j]=(float)Hh; }

    long tr_start=g_fsz/5;
    long va_start[N_VAL]={ g_fsz/2, (long)(0.65*g_fsz), (long)(0.80*g_fsz) };
    for(int w=0;w<N_VAL;w++) if(va_start[w]+N+3>g_fsz){ fprintf(stderr,"val window %d out of file\n",w+1); return 1; }
    fprintf(stderr,"48.C: train @%ld val @%ld/%ld/%ld (N=%ld) H=%d ep=%d X_DIM=%d gate_gain=%.1f (~%.1f GB feats)\n",
            tr_start,va_start[0],va_start[1],va_start[2],N,H,EP,X_DIM,GATE_GAIN,(double)N*(N_VAL+1)*X_DIM*4/1e9);

    Win tr; Win va[N_VAL];
    tr.X=malloc((size_t)N*X_DIM*4); tr.tgt=malloc(N); tr.c1=malloc(N); tr.c2=malloc(N); tr.start=tr_start; tr.perm=make_perm(N,0x48CC5EEDULL);
    for(int w=0;w<N_VAL;w++){ va[w].X=malloc((size_t)N*X_DIM*4); va[w].tgt=malloc(N); va[w].c1=malloc(N); va[w].c2=malloc(N);
        va[w].start=va_start[w]; va[w].perm=make_perm(N,0x48CC5EEDULL^(uint64_t)(w+1)*0x9E3779B9ULL); }
    if(!tr.X||!va[0].X||!va[1].X||!va[2].X){ fprintf(stderr,"OOM\n"); return 1; }
    double gate_stats[4]={0,0,0,0};
    fprintf(stderr,"extracting train window...\n");
    extract_window(&see,tr_start,N,tr.X,tr.tgt,tr.c1,tr.c2,&tr.tri,&tr.d1,gate_stats);
    for(int w=0;w<N_VAL;w++){ fprintf(stderr,"extracting val window %d...\n",w+1);
        extract_window(&see,va_start[w],N,va[w].X,va[w].tgt,va[w].c1,va[w].c2,&va[w].tri,&va[w].d1,NULL); }

    // --- standardize the reservoir band columns [OFF_DYN..X_DIM] with FROZEN train stats ---
    // The leaky integrator r is unbounded; the base feats are z-scored and armB-cos is bounded,
    // so the raw reservoir would swamp the readout. Z-score per dim (train mean/std), clamp +-4
    // (matches the base-feature clamp philosophy). Dynamics unchanged; only the readout's view.
    fprintf(stderr,"standardizing reservoir bands (train stats, clamp +-4)...\n");
    for(long col=OFF_DYN; col<X_DIM; col++){
        double s=0,s2=0; for(long i=0;i<N;i++){ float v=tr.X[(size_t)i*X_DIM+col]; s+=v; s2+=(double)v*v; }
        double mu=s/N, var=s2/N-mu*mu; if(var<0)var=0; bmean[col]=(float)mu; bstd[col]=(float)sqrt(var)+1e-8f;
    }
    { Win* all[1+N_VAL]; all[0]=&tr; for(int w=0;w<N_VAL;w++) all[1+w]=&va[w];
      for(int wi=0; wi<1+N_VAL; wi++){ float* Xp=all[wi]->X;
        for(long i=0;i<N;i++) for(long col=OFF_DYN; col<X_DIM; col++){
            float z=(Xp[(size_t)i*X_DIM+col]-bmean[col])/bstd[col]; if(z>4.f)z=4.f; if(z<-4.f)z=-4.f;
            Xp[(size_t)i*X_DIM+col]=z; } } }

    printf("\n==== 48.C multiplicative-dynamics reservoir (base=armB frozen, N=%ld, ep=%d, H=%d) ====\n",N,EP,H);
    printf("GATE_STATS mean=%.4f std=%.4f pct_lo=%.2f pct_hi=%.2f (gate live if std>~0.05 and not stuck 0/1/0.5)\n",
           gate_stats[0],gate_stats[1],gate_stats[2],gate_stats[3]);
    printf("BASE win=train tri=%.4f d1=%.4f\n",tr.tri,tr.d1);
    for(int w=0;w<N_VAL;w++) printf("BASE win=val%d tri=%.4f d1=%.4f\n",w+1,va[w].tri,va[w].d1);

    Seg s_base[1]={{0,D1_TOT}};
    Seg s_armb[1]={{0,BASE512}};                              // baseline (contiguous nf+armB)
    Seg s_dyn[2] ={{0,BASE512},{OFF_DYN,DYN_BANDS}};
    Seg s_lin[2] ={{0,BASE512},{OFF_LIN,DYN_BANDS}};
    Seg s_sp[2]  ={{0,BASE512},{OFF_SP,DYN_BANDS}};

    // frozenD1 anchor (base 256 readout) - must == BASE d1
    { double v[N_VAL]; float lg[CLASSES];
      for(int w=0;w<N_VAL;w++){ double tot=0;
        for(long i=0;i<N;i++){ const float* x=&va[w].X[(size_t)i*X_DIM];
            for(int c=0;c<CLASSES;c++) lg[c]=Bd1[c]+dot_avx(Wd1[c],x,D1_TOT);
            const float* tri=&trigram[va[w].c2[i]][va[w].c1[i]][0]; for(int c=0;c<CLASSES;c++) lg[c]+=tri[c];
            float mx=-1e30f; for(int c=0;c<CLASSES;c++) if(lg[c]>mx)mx=lg[c];
            double Z=0; for(int c=0;c<CLASSES;c++) Z+=exp((double)(lg[c]-mx));
            double pp=exp((double)(lg[va[w].tgt[i]]-mx))/Z; tot+=-log2(pp>1e-30?pp:1e-30); }
        v[w]=tot/N; }
      printf("PROBE name=frozenD1 h=0 dim=%d val1=%.4f val2=%.4f val3=%.4f rms1=0 ent1=0 maxp1=0\n",D1_TOT,v[0],v[1],v[2]); }

    char sp[512]; snprintf(sp,sizeof sp,"%s_DYN_h%d.bin",argv[3],H);
    run_probe("armB",   H,s_armb,1,NULL,X_DIM+1,&tr,va,N,EP,NULL);          // baseline (reproduces 48.0/48.B armB)
    run_probe("DYN",    H,s_dyn, 2,NULL,X_DIM+1,&tr,va,N,EP,sp);            // multiplicative reservoir (+ save)
    run_probe("LIN_dyn",H,s_lin, 2,NULL,X_DIM+1,&tr,va,N,EP,NULL);          // constant gate (multiplication isolate)
    run_probe("DYN_st", H,s_dyn, 2,(const long*)1,OFF_DYN,&tr,va,N,EP,NULL);// shuffled-time leak guard
    run_probe("DYN_sp", H,s_sp,  2,NULL,X_DIM+1,&tr,va,N,EP,NULL);          // channel-permuted gate pairing (informative)

    printf("\nPre-registered criterion (decided BEFORE the run): DYN earns 48.C.A iff frozenD1==BASE d1,\n");
    printf("DYN beats armB by >= 0.015 on ALL of val1/2/3, DYN beats LIN_dyn by >= 0.008 on ALL, and\n");
    printf("DYN_st does NOT beat armB by > 0.005. DYN-DYN_sp margin = informative (gating semantics).\n");
    printf("Gate must be live (GATE_STATS). TF-only != generative (44-47): the real verdict for a PASS\n");
    printf("is the closed-loop gate v2 + replicas + human reading at 48.C.A, exactly as armB.\n");
    return 0;
}
