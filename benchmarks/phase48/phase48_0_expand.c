// Phase 48.0 - Sonda 2 (redesigned): CAPACITY PROBE = manifold expansion (teacher-forced ONLY)
//
// TAPS (Sonda 1) came back FLAT: re-exposing past SEE states does not help because the
// reservoir already integrates recent history into SEE(t). Reading silicon_entropy.c +
// silicon_v0.c (Q0) sharpened the diagnosis:
//   * L0 is 64-D = [M4 32 (LINEAR sum of binary codes) | V0-wave 32 (the only nonlinearity,
//     from saturating wave dynamics)].
//   * The temporal reservoir (L1 fast/mid/slow bands) reads only l0_out[0:43] = all 32 linear
//     M4 dims + just 11 of 32 wave dims; the other 21 nonlinear wave dims are NEVER integrated.
//   * L1 integration is pure linear EMA; Oja's readout feature y=W.L0 is LINEAR at inference.
// => the recurrent substrate is a LINEAR reservoir over a mostly-linear 43-D input. All the
// project's nonlinearity lives in the readout MLP. Widening L0 itself is too invasive for a
// probe (breaks every frozen artifact + the anchor), so we take the light fallback: a fixed
// random NONLINEAR LIFT of L0, temporally integrated as NEW reservoir dims (substrate-side in
// spirit: deterministic fn of the byte stream, no backprop, closed-loop-reproducible).
//
// Two arms differ ONLY by a cos() (same projection, same dim, same timescales):
//   z[d]  = b[d] + GAMMA * sum_k Omega[d][k] * L0norm[k]      (standardized 64-D L0)
//   armA  (control, LINEAR):    EMA_mid(z),     EMA_slow(z)        -> 2*D_EXP new dims
//   armB  (treatment, NONLIN):  EMA_mid(cos z), EMA_slow(cos z)    -> 2*D_EXP new dims
// Both appended to the frozen base feature [SEE 192 | L2_D1 64]=256 (bit-identical => the
// frozenD1 anchor stays exact). Probes (train@20% -> val@50/65/80%, H32, 47 pipeline):
//   frozenD1  : D1 readout through the probe path (anchor, must == BASE d1)
//   linear    : scratch linear on base 256 (pipeline context)
//   notap     : H32 on base 256                  (the binding baseline, ~2.24)
//   armA      : H32 on [base | A bands]          (linear expansion control, predicted FLAT)
//   armB      : H32 on [base | B bands]          (the real test)
//   armB_shuf : H32 on [base | B bands from RANDOM rows] (leak guard: must NOT help)
//
// PRE-REGISTERED criterion (decided BEFORE the run, by the Architetto): armB earns 48.A iff
//   [1] frozenD1 == BASE d1 (|diff| <= 0.005)
//   [2] armB beats BOTH notap AND armA by >= 0.015 on ALL of val1/val2/val3
//   [3] armB_shuf does NOT beat notap by > 0.005 on any window
// Reading: A flat + B helps = the input/nonlinearity bottleneck is the lever (substrate-side,
// mantra-pure) -> 48.A. Both flat = the substrate family is the ceiling at fixed dynamics ->
// MODOJA. A helps too = informative surprise (the linear reservoir wastes L0's wave dims).
// TF-only caveat (44/45/46/47 law): compression-positive != generative. No word-gate here.
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase38-42/phase48_0_expand.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase48_0_expand.exe -lm -I .
// Run:
//   bin/phase48_0_expand.exe <data> <D1_w> <outprefix> [--len N --epochs E --hidden H]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include "src/silicon_entropy.h"

#define CLASSES   256
#define BASE_DIM  SEE_FEATURE_DIM        // 192
#define L0_DIM    SEE_L0_DIM             // 64 (the lift reads this slice of the feature)
#define L2_DIM    64
#define D1_TOT    (BASE_DIM + L2_DIM)    // 256 (frozen base feature)
#define D_EXP     128                    // genuine expansion 64 -> 128 nonlinear dims
#define GAMMA     0.25f                  // RFF-style bandwidth (z std ~2 => cos spans range)
#define N_TS      2                      // integration timescales
#define EXP_BANDS (N_TS*D_EXP)           // 256 new dims per arm
#define X_DIM     (D1_TOT + 2*EXP_BANDS) // 256 base + 256 A + 256 B = 768
#define OFF_A     (D1_TOT)               // 256
#define OFF_B     (D1_TOT + EXP_BANDS)   // 512
#define PROJ_SEED 0x48B2EC0DEULL
#define N_VAL     3

static const float TS_ALPHA[N_TS] = { 0.90f, 0.99f };   // mid, slow (matches substrate bands)

static float Pmat[L2_DIM][BASE_DIM];
static float Omega[D_EXP][L0_DIM];
static float Bvec[D_EXP];
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
// fixed random Gaussian projection (Box-Muller) + uniform phase, for the nonlinear lift
static inline double xs_u01(uint64_t* s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return (double)((*s>>11)&0x1FFFFFFFFFFFFFULL)/(double)(1ULL<<53); }
static void gen_lift(uint64_t seed){ uint64_t s=seed?seed:0xABCDEF12345ULL;
    for(int d=0;d<D_EXP;d++){
        for(int k=0;k<L0_DIM;k++){ double u1=xs_u01(&s); if(u1<1e-12) u1=1e-12; double u2=xs_u01(&s);
            Omega[d][k]=(float)(sqrt(-2.0*log(u1))*cos(6.283185307179586*u2)); }
        Bvec[d]=(float)(6.283185307179586*xs_u01(&s)); } }
static inline int ent_gate(uint8_t c1,uint8_t c2){ return g_ent_high?(ent_table[c2][c1]>g_ent_thr):(ent_table[c2][c1]<g_ent_thr); }

// same 0x53454540 loader as phase48_0_taps.c (D1 only, want_shared path)
static int load_weight(const char* path, int tot, float* W, float* B, float* mean, float* std,
                       SiliconEntropyState* see, float* l2clamp, float* l2scale) {
    FILE* f=fopen(path,"rb"); if(!f){ fprintf(stderr,"open %s\n",path); return 0; }
    uint32_t hdr4[4]; fread(hdr4,4,4,f); float hf[5]; fread(hf,4,5,f);
    float decay=hf[0], afast=hf[2], fclamp=hf[3];
    uint32_t no=0; fread(&no,4,1,f); int noja=(int)no; float ojab[SEE_N_OJA_MAX*43]; fread(ojab,4,(size_t)noja*43,f);
    uint32_t l2d=0,gt=0,eh=0,ps=0; float al=0,st=0,et=0;
    fread(&l2d,4,1,f); fread(&gt,4,1,f); fread(&al,4,1,f); fread(&st,4,1,f); fread(&et,4,1,f); fread(&eh,4,1,f); fread(&ps,4,1,f);
    g_alpha=al;
    float l2c=0,nbd=1; uint32_t cd=0,dl=0; fread(&l2c,4,1,f); fread(&nbd,4,1,f); fread(&cd,4,1,f); fread(&dl,4,1,f);
    float mx=0; fread(&mx,4,1,f); float ls=1; fread(&ls,4,1,f); float l2cap=0; fread(&l2cap,4,1,f);
    *l2clamp=(l2c>0)?l2c:fclamp; *l2scale=(ls>0)?ls:1.0f;
    size_t tn=(size_t)CLASSES*CLASSES*CLASSES; trigram=malloc(tn*sizeof(float)); fread(trigram,4,tn,f);
    g_ent_thr=et; g_ent_high=(int)eh; gen_projection(ps);
    see_init(see,42,4,decay); see->multiscale_mode=1; see->alpha_fast=afast; see->alpha_mid=0.9f; see->alpha_slow=0.99f;
    see->n_oja=noja; memcpy(see->W_oja,ojab,(size_t)noja*43*sizeof(float)); see->eta_oja=0.0f; see->plastic_blend=1.0f;
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

// ---- feature segments: probes select column ranges; expansion segments (off >= OFF_B,
// i.e. the B bands) can read from a PERMUTED row (shuffled control). ----
typedef struct { long off; int len; } Seg;
static inline int gather_row(const float* X,long i,const Seg* segs,int nseg,const long* perm,float* out){
    int d=0;
    for(int s=0;s<nseg;s++){
        long row=(segs[s].off>=OFF_B && perm)?perm[i]:i;
        memcpy(out+d,&X[(size_t)row*X_DIM+segs[s].off],(size_t)segs[s].len*4); d+=segs[s].len;
    }
    return d;
}

// ---- probe model: 1-hidden relu MLP, or LINEAR when h==0 (same Adam/batching as 47/48 taps) ----
typedef struct { int in,h; float *W1,*b1,*W2,*b2,*mW1,*vW1,*mb1,*vb1,*mW2,*vW2,*mb2,*vb2; int t; } MLP;
static void mlp_init(MLP* m,int in,int h){ m->in=in; m->h=h; m->t=0;
    int hw=(h>0)?h:in;
    size_t s1=(h>0)?(size_t)h*in:0, s2=(size_t)CLASSES*hw;
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
    } else {
        for(int c=0;c<CLASSES;c++) lg[c]=m->b2[c]+dot_avx(&m->W2[(size_t)c*m->in],x,m->in);
    }
}
static double mlp_eval(MLP* m,const float* X,const Seg* segs,int nseg,const long* perm,
                       const uint8_t* tgt,const uint8_t* c1a,const uint8_t* c2a,long n,
                       double* o_rms,double* o_ent,double* o_maxp){
    float* hid=malloc(((m->h>0)?m->h:1)*4); float* xb=malloc(X_DIM*4); float lg[CLASSES];
    double tot=0,arms=0,aent=0,amax=0;
    for(long i=0;i<n;i++){
        gather_row(X,i,segs,nseg,perm,xb);
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
static void mlp_train(MLP* m,const float* X,const Seg* segs,int nseg,const long* perm,
                      const uint8_t* tgt,const uint8_t* c1a,const uint8_t* c2a,long n,int epochs,float lr){
    int H=m->h,in=m->in,lin=(H==0),hw=lin?in:H;
    size_t s1=lin?0:(size_t)H*in, s2=(size_t)CLASSES*hw;
    float* gW1=malloc((s1?s1:1)*4); float* gb1=malloc((H>0?H:1)*4); float* gW2=malloc(s2*4); float* gb2=malloc(CLASSES*4);
    float* hid=malloc((H>0?H:1)*4); float* dh=malloc((H>0?H:1)*4); float* xb=malloc(X_DIM*4);
    float lg[CLASSES],eo[CLASSES]; int bs=512;
    for(int ep=0;ep<epochs;ep++){
        memset(gW1,0,(s1?s1:1)*4); memset(gb1,0,(H>0?H:1)*4); memset(gW2,0,s2*4); memset(gb2,0,CLASSES*4); long inb=0;
        for(long i=0;i<n;i++){
            gather_row(X,i,segs,nseg,perm,xb); const float* x=xb;
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
    free(gW1);free(gb1);free(gW2);free(gb2);free(hid);free(dh);free(xb);
}
// notap stays generator-compatible (0x53454545); armB uses NEW magic 0x53454548 carrying the
// lift reconstruction info (base_dim, d_exp, n_ts, timescales, gamma, proj_seed) so 48.A can
// rebuild the fixed projection and EMA bands closed-loop.
static void mlp_save45(MLP* m,long off,const char* path){
    FILE* f=fopen(path,"wb"); if(!f){ fprintf(stderr,"save %s failed\n",path); return; }
    uint32_t magic=0x53454545,H=(uint32_t)m->h,o=(uint32_t)off,d=(uint32_t)m->in;
    fwrite(&magic,4,1,f); fwrite(&H,4,1,f); fwrite(&o,4,1,f); fwrite(&d,4,1,f);
    if(m->h>0){ fwrite(m->W1,4,(size_t)m->h*m->in,f); fwrite(m->b1,4,m->h,f); fwrite(m->W2,4,(size_t)CLASSES*m->h,f); }
    else fwrite(m->W2,4,(size_t)CLASSES*m->in,f);
    fwrite(m->b2,4,CLASSES,f); fclose(f);
}
static void mlp_save_exp(MLP* m,const char* path){
    FILE* f=fopen(path,"wb"); if(!f){ fprintf(stderr,"save %s failed\n",path); return; }
    uint32_t magic=0x53454548,H=(uint32_t)m->h,d=(uint32_t)m->in,base=D1_TOT,dexp=D_EXP,nts=N_TS;
    float gamma=GAMMA; uint64_t pseed=PROJ_SEED;
    fwrite(&magic,4,1,f); fwrite(&H,4,1,f); fwrite(&d,4,1,f); fwrite(&base,4,1,f);
    fwrite(&dexp,4,1,f); fwrite(&nts,4,1,f); fwrite(&gamma,4,1,f); fwrite(&pseed,8,1,f);
    for(int t=0;t<N_TS;t++){ float a=TS_ALPHA[t]; fwrite(&a,4,1,f); }
    if(m->h>0){ fwrite(m->W1,4,(size_t)m->h*m->in,f); fwrite(m->b1,4,m->h,f); fwrite(m->W2,4,(size_t)CLASSES*m->h,f); }
    else fwrite(m->W2,4,(size_t)CLASSES*m->in,f);
    fwrite(m->b2,4,CLASSES,f); fclose(f);
}

// Teacher-forced extraction (D1 dynamics identical to phase48_0_taps.c). Row layout:
// [ base 256 | A bands 256 | B bands 256 ]. The lift EMAs are causal: row[i] holds the
// integration of z over steps < i (updated AFTER the row is written, like L2_D1), so no
// target leak. z is computed from the SAME pre-observation feat192 the readout predicts with.
static void extract_window(SiliconEntropyState* see, long start, long N,
                           float* X, uint8_t* tgt, uint8_t* c1a, uint8_t* c2a,
                           double* oTRI, double* oD1){
    float L2d1[L2_DIM]={0}, pb_d1[BASE_DIM]={0};
    float eA[N_TS][D_EXP], eB[N_TS][D_EXP];
    memset(eA,0,sizeof eA); memset(eB,0,sizeof eB);
    float feat192[BASE_DIM], fa[BASE_DIM], rawd1[D1_TOT], nf[D1_TOT], l0n[L0_DIM];
    float scale=1.0f/sqrtf((float)BASE_DIM); double btr=0,bd1=0;
    see_reset(see); for(long p=0;p<=start+1;p++) see_observe(see,g_data[p]);
    for(long i=0;i<N;i++){
        long g=start+i; uint8_t c2=g_data[g],c1=g_data[g+1],t=g_data[g+2];
        see_extract(see,feat192);
        memcpy(rawd1,feat192,BASE_DIM*4); memcpy(rawd1+BASE_DIM,L2d1,L2_DIM*4);
        btr+=trigram_bpb(c1,c2,t);
        norm_feats(rawd1,D1_TOT,md1,sd1,g_l2c_d1,g_ls_d1,nf); bd1+=logits_bpb(&Wd1[0][0],Bd1,nf,D1_TOT,c1,c2,t);
        float* row=&X[(size_t)i*X_DIM];
        memcpy(row,nf,D1_TOT*4);
        for(int ts=0;ts<N_TS;ts++){ memcpy(row+OFF_A+ts*D_EXP,eA[ts],D_EXP*4); memcpy(row+OFF_B+ts*D_EXP,eB[ts],D_EXP*4); }
        tgt[i]=t; c1a[i]=c1; c2a[i]=c2;
        // nonlinear lift of the standardized 64-D L0 (SEE dims 0..63), then integrate
        for(int k=0;k<L0_DIM;k++){ float x=(feat192[k]-md1[k])/(sd1[k]+1e-8f); if(x>2.f)x=2.f; if(x<-2.f)x=-2.f; l0n[k]=x; }
        for(int d=0;d<D_EXP;d++){ float z=Bvec[d]+GAMMA*dot_avx(Omega[d],l0n,L0_DIM); float cz=cosf(z);
            for(int ts=0;ts<N_TS;ts++){ float a=TS_ALPHA[ts];
                eA[ts][d]=a*eA[ts][d]+(1.0f-a)*z; eB[ts][d]=a*eB[ts][d]+(1.0f-a)*cz; } }
        see_observe(see,t); see_extract(see,fa);
        if(ent_gate(c1,c2)){ float src5[BASE_DIM];
            for(int k=0;k<BASE_DIM;k++) src5[k]=fa[k]-0.5f*pb_d1[k]; memcpy(pb_d1,fa,BASE_DIM*4);
            for(int j=0;j<L2_DIM;j++){ float p5=0; const float* pj=Pmat[j]; for(int k=0;k<BASE_DIM;k++) p5+=pj[k]*src5[k];
                L2d1[j]=g_alpha*L2d1[j]+(1.0f-g_alpha)*p5*scale; } }
    }
    *oTRI=btr/N; *oD1=bd1/N;
}

typedef struct { float* X; uint8_t *tgt,*c1,*c2; double tri,d1; long start; long* perm; } Win;

static long* make_perm(long n,uint64_t seed){
    long* p=malloc(n*sizeof(long)); for(long i=0;i<n;i++) p[i]=i;
    uint64_t r=seed;
    for(long i=n-1;i>0;i--){ r^=r<<13;r^=r>>7;r^=r<<17; long j=(long)(r%(uint64_t)(i+1)); long t=p[i]; p[i]=p[j]; p[j]=t; }
    return p;
}

static void run_probe(const char* name,int H,const Seg* segs,int nseg,int use_perm,
                      Win* tr,Win* va,long N,int EP,const char* savepath,int save_fmt){
    int dim=0; for(int s=0;s<nseg;s++) dim+=segs[s].len;
    fprintf(stderr,"  probe %s (H=%d dim=%d nseg=%d perm=%d)...\n",name,H,dim,nseg,use_perm);
    MLP m; mlp_init(&m,dim,H);
    mlp_train(&m,tr->X,segs,nseg,use_perm?tr->perm:NULL,tr->tgt,tr->c1,tr->c2,N,EP,0.0005f);
    double v[N_VAL],rms=0,ent=0,mxp=0;
    for(int w=0;w<N_VAL;w++)
        v[w]=mlp_eval(&m,va[w].X,segs,nseg,use_perm?va[w].perm:NULL,va[w].tgt,va[w].c1,va[w].c2,N,
                      w==0?&rms:NULL,w==0?&ent:NULL,w==0?&mxp:NULL);
    printf("PROBE name=%s h=%d dim=%d val1=%.4f val2=%.4f val3=%.4f rms1=%.3f ent1=%.3f maxp1=%.3f\n",
           name,H,dim,v[0],v[1],v[2],rms,ent,mxp);
    if(savepath){ if(save_fmt==48) mlp_save_exp(&m,savepath); else mlp_save45(&m,0,savepath);
        printf("SAVED name=%s path=%s\n",name,savepath); }
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
    load_weight(argv[2],D1_TOT,&Wd1[0][0],Bd1,md1,sd1,&see,&g_l2c_d1,&g_ls_d1);
    gen_lift(PROJ_SEED);
    ent_table=malloc(CLASSES*CLASSES*4);
    for(int i=0;i<CLASSES;i++) for(int j=0;j<CLASSES;j++){ float m=-1e9f; for(int k=0;k<CLASSES;k++) if(trigram[i][j][k]>m)m=trigram[i][j][k];
        double se=0; for(int k=0;k<CLASSES;k++) se+=exp((double)(trigram[i][j][k]-m));
        double Hh=0; for(int k=0;k<CLASSES;k++){ double p=exp((double)(trigram[i][j][k]-m))/se; if(p>1e-12)Hh-=p*log(p); }
        ent_table[i][j]=(float)Hh; }

    long tr_start=g_fsz/5;
    long va_start[N_VAL]={ g_fsz/2, (long)(0.65*g_fsz), (long)(0.80*g_fsz) };
    for(int w=0;w<N_VAL;w++) if(va_start[w]+N+3>g_fsz){ fprintf(stderr,"val window %d out of file\n",w+1); return 1; }
    fprintf(stderr,"48.0 EXPAND: train @%ld, val @%ld/%ld/%ld (N=%ld) H=%d ep=%d D_exp=%d gamma=%.3f xdim=%d (~%.1f GB feats)\n",
            tr_start,va_start[0],va_start[1],va_start[2],N,H,EP,D_EXP,GAMMA,X_DIM,(double)N*(N_VAL+1)*X_DIM*4/1e9);

    Win tr; Win va[N_VAL];
    tr.X=malloc((size_t)N*X_DIM*4); tr.tgt=malloc(N); tr.c1=malloc(N); tr.c2=malloc(N); tr.start=tr_start;
    tr.perm=make_perm(N,0x48AB5EEDULL);
    for(int w=0;w<N_VAL;w++){ va[w].X=malloc((size_t)N*X_DIM*4); va[w].tgt=malloc(N); va[w].c1=malloc(N); va[w].c2=malloc(N);
        va[w].start=va_start[w]; va[w].perm=make_perm(N,0x48AB5EEDULL^(uint64_t)(w+1)*0x9E3779B9ULL); }
    if(!tr.X||!va[0].X||!va[1].X||!va[2].X){ fprintf(stderr,"OOM\n"); return 1; }
    fprintf(stderr,"extracting train window...\n");
    extract_window(&see,tr_start,N,tr.X,tr.tgt,tr.c1,tr.c2,&tr.tri,&tr.d1);
    for(int w=0;w<N_VAL;w++){ fprintf(stderr,"extracting val window %d...\n",w+1);
        extract_window(&see,va_start[w],N,va[w].X,va[w].tgt,va[w].c1,va[w].c2,&va[w].tri,&va[w].d1); }

    printf("\n==== 48.0 Sonda 2: EXPAND (train-win -> 3 held-out val-wins, N=%ld, ep=%d, H=%d, D_exp=%d) ====\n",N,EP,H,D_EXP);
    printf("BASE win=train start=%ld tri=%.4f d1=%.4f\n",tr.start,tr.tri,tr.d1);
    for(int w=0;w<N_VAL;w++)
        printf("BASE win=val%d start=%ld tri=%.4f d1=%.4f\n",w+1,va[w].start,va[w].tri,va[w].d1);

    Seg s_base[1] ={{0,D1_TOT}};
    Seg s_A[2]    ={{0,D1_TOT},{OFF_A,EXP_BANDS}};
    Seg s_B[2]    ={{0,D1_TOT},{OFF_B,EXP_BANDS}};

    // anchor: D1's own readout through the probe eval path (must == BASE d1)
    { MLP m; mlp_init(&m,D1_TOT,0);
      for(int c=0;c<CLASSES;c++){ memcpy(&m.W2[(size_t)c*D1_TOT],&Wd1[c][0],D1_TOT*4); m.b2[c]=Bd1[c]; }
      double v[N_VAL],rms=0,ent=0,mxp=0;
      for(int w=0;w<N_VAL;w++) v[w]=mlp_eval(&m,va[w].X,s_base,1,NULL,va[w].tgt,va[w].c1,va[w].c2,N,
                                             w==0?&rms:NULL,w==0?&ent:NULL,w==0?&mxp:NULL);
      printf("PROBE name=frozenD1 h=0 dim=%d val1=%.4f val2=%.4f val3=%.4f rms1=%.3f ent1=%.3f maxp1=%.3f\n",
             D1_TOT,v[0],v[1],v[2],rms,ent,mxp);
      mlp_free(&m); }

    char sp[512];
    run_probe("linear",0,s_base,1,0,&tr,va,N,EP,NULL,0);
    snprintf(sp,sizeof sp,"%s_notap_h%d.bin",argv[3],H);
    run_probe("notap",H,s_base,1,0,&tr,va,N,EP,sp,45);
    run_probe("armA",H,s_A,2,0,&tr,va,N,EP,NULL,0);
    snprintf(sp,sizeof sp,"%s_armB_h%d.bin",argv[3],H);
    run_probe("armB",H,s_B,2,0,&tr,va,N,EP,sp,48);
    run_probe("armB_shuf",H,s_B,2,1,&tr,va,N,EP,NULL,0);

    printf("\nPre-registered criterion (decided BEFORE the run, by the Architetto): armB earns the\n");
    printf("48.A DAgger run iff\n");
    printf("  [1] frozenD1 == BASE d1 (|diff| <= 0.005, probe-path integrity)\n");
    printf("  [2] armB beats BOTH notap AND armA by >= 0.015 on ALL of val1/val2/val3\n");
    printf("  [3] armB_shuf does NOT beat notap by > 0.005 on any window (leak/capacity guard)\n");
    printf("Reading: A flat + B helps = input/nonlinearity bottleneck is the lever (substrate-side,\n");
    printf("mantra-pure) -> 48.A. Both flat = substrate family is the ceiling -> MODOJA. A helps =\n");
    printf("informative surprise (linear reservoir wastes L0's wave dims). TF-only != generative.\n");
    return 0;
}
