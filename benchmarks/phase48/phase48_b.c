// Phase 48.B - substrate scaling law: three new feature CLASSES on the frozen armB baseline
//
// armB (48.0/48.A) = Random Fourier Features: cos(Omega.L0) approximates a Gaussian kernel,
// so it turned the linear reservoir into a kernel machine. Raising D_EXP only approximates
// the SAME kernel better (~1/sqrt(D)). The lever is different feature CLASSES. armB taught:
// the silicon sees SIMILARITIES (kernel) but not RELATIONS - a relation is a PRODUCT (A.B =
// "A in the context of B"), which the additive EMA reservoir cannot represent. 48.B probes
// three classes, TF-only, on top of the frozen armB feature (the new baseline):
//   BILIN   : elementwise products fast_EMA.slow_EMA (recent x older) + state.input (slow.cur)
//             = multiplicative gating pulled from the dynamics. Controls: shuffled PAIRING
//             (products of uncorrelated channels - must help less) + shuffled-time leak guard.
//   WAVE32  : integrate ALL 32 wave dims (Q0: 21 of them are never integrated) vs the 11 in
//             the live reservoir. Use the silicon's OWN discarded nonlinearity. Leak: shuf-time.
//   MULTIBW : cos() at three bandwidths (gamma 0.0625 / 0.25 / 1.0). armB already has 0.25;
//             this adds 0.0625 and 1.0 = multi-scale kernel. Leak: shuf-time.
//
// Feature row: [ D1 base 256 | armB cos g0.25 256 | BILIN 128 | WAVE32 64 | MULTIBW 256 |
//               BILIN_shufpair 128 ] = 1088. The base 256 and the armB 256 reproduce the
// 48.0 armB feature EXACTLY (same PROJ_SEED/GAMMA) so the armB-baseline probe ~ 2.18-2.20 and
// frozenD1 anchor == BASE d1. Causal EMAs cold-start 0 (like L2). PRE-REGISTERED: each arm
// must beat armB-alone by >= 0.015 on ALL three windows, shuffled controls clean, anchor exact.
// Winners (stackable) -> 48.B.A DAgger closed-loop under the frozen 47 harness. TF != generative.
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase38-42/phase48_b.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase48_b.exe -lm -I .
// Run:
//   bin/phase48_b.exe <data> <D1_w> <outprefix> [--len N --epochs E --hidden H]

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
#define N_TS      2                      // EMA timescales for armB/wave/multibw
#define DA        128                    // armB cos dims (gamma 0.25) - same as 48.0
#define DB        64                     // bilin block (each product = 64); BILIN = 2 blocks
#define DW        32                     // wave dims (all 32 of the V0 wave)
#define DM        64                     // multibw dims per extra gamma
#define ARMB_BANDS (N_TS*DA)             // 256
#define BIL_BANDS  (2*DB)                // 128
#define WAV_BANDS  (N_TS*DW)             // 64
#define MB_BANDS   (2*N_TS*DM)           // 256 (2 extra gammas x N_TS)
#define OFF_ARMB   (D1_TOT)              // 256
#define OFF_BIL    (OFF_ARMB+ARMB_BANDS) // 512
#define OFF_WAV    (OFF_BIL+BIL_BANDS)   // 640
#define OFF_MB     (OFF_WAV+WAV_BANDS)   // 704
#define OFF_BILSH  (OFF_MB+MB_BANDS)     // 960
#define X_DIM      (OFF_BILSH+BIL_BANDS) // 1088
#define GAMMA_A    0.25f
#define GAMMA_M0   0.0625f
#define GAMMA_M1   1.0f
#define SEED_A     0x48B2EC0DEULL        // == 48.0 armB (reproduces the baseline exactly)
#define SEED_M0    0x48B3A1107ULL
#define SEED_M1    0x48B4CC219ULL
#define SEED_BILP  0x48B5B11A7ULL
#define N_VAL     3

static const float TS_ALPHA[N_TS] = { 0.90f, 0.99f };
static const float BIL_FAST = 0.5f, BIL_SLOW = 0.95f;

static float Pmat[L2_DIM][BASE_DIM];
static float OmA[DA][L0_DIM], BvA[DA];
static float OmM0[DM][L0_DIM], BvM0[DM], OmM1[DM][L0_DIM], BvM1[DM];
static int   bilperm[DB];
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
// segments with off >= permFromOff read from perm[i] (shuffled-time leak guard); others from i.
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
    int H=m->h,in=m->in;
    size_t s1=(size_t)H*in, s2=(size_t)CLASSES*H;
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

// Teacher-forced extraction. Builds the full 1088-D row: base | armB cos | BILIN | WAVE32 |
// MULTIBW | BILIN_shufpair. All temporally-integrated parts are causal (row i integrates <i;
// EMAs updated AFTER the row is written; products use cur=L0n_i which is known at prediction).
static void extract_window(SiliconEntropyState* see, long start, long N,
                           float* X, uint8_t* tgt, uint8_t* c1a, uint8_t* c2a, double* oTRI, double* oD1){
    float L2d1[L2_DIM]={0}, pb_d1[BASE_DIM]={0};
    float eA[N_TS][DA], eW[N_TS][DW], eM0[N_TS][DM], eM1[N_TS][DM], fastL[L0_DIM], slowL[L0_DIM];
    memset(eA,0,sizeof eA); memset(eW,0,sizeof eW); memset(eM0,0,sizeof eM0); memset(eM1,0,sizeof eM1);
    memset(fastL,0,sizeof fastL); memset(slowL,0,sizeof slowL);
    float feat192[BASE_DIM], fa[BASE_DIM], rawd1[D1_TOT], nf[D1_TOT], l0n[L0_DIM];
    float scale=1.0f/sqrtf((float)BASE_DIM); double btr=0,bd1=0;
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
        // --- write current bands (causal: hold integration over <i) ---
        for(int ts=0;ts<N_TS;ts++){ memcpy(row+OFF_ARMB+ts*DA,eA[ts],DA*4);
            memcpy(row+OFF_WAV+ts*DW,eW[ts],DW*4);
            memcpy(row+OFF_MB+(0*N_TS+ts)*DM,eM0[ts],DM*4); memcpy(row+OFF_MB+(1*N_TS+ts)*DM,eM1[ts],DM*4); }
        // BILIN: fast.slow (recent x older) | slow.cur (state x input); shufpair = same with permuted partner
        for(int k=0;k<DB;k++){ row[OFF_BIL+k]=fastL[k]*slowL[k]; row[OFF_BIL+DB+k]=slowL[k]*l0n[k];
            row[OFF_BILSH+k]=fastL[k]*slowL[bilperm[k]]; row[OFF_BILSH+DB+k]=slowL[k]*l0n[bilperm[k]]; }
        tgt[i]=t; c1a[i]=c1; c2a[i]=c2;
        // --- update EMAs/products state with this step (l0n_i) ---
        for(int d=0;d<DA;d++){ float z=BvA[d]+GAMMA_A*dot_avx(OmA[d],l0n,L0_DIM); float cz=cosf(z);
            for(int ts=0;ts<N_TS;ts++){ float a=TS_ALPHA[ts]; eA[ts][d]=a*eA[ts][d]+(1.0f-a)*cz; } }
        for(int w=0;w<DW;w++){ float v=l0n[32+w]; for(int ts=0;ts<N_TS;ts++){ float a=TS_ALPHA[ts]; eW[ts][w]=a*eW[ts][w]+(1.0f-a)*v; } }
        for(int d=0;d<DM;d++){ float z0=BvM0[d]+GAMMA_M0*dot_avx(OmM0[d],l0n,L0_DIM); float c0=cosf(z0);
            float z1=BvM1[d]+GAMMA_M1*dot_avx(OmM1[d],l0n,L0_DIM); float c1c=cosf(z1);
            for(int ts=0;ts<N_TS;ts++){ float a=TS_ALPHA[ts]; eM0[ts][d]=a*eM0[ts][d]+(1.0f-a)*c0; eM1[ts][d]=a*eM1[ts][d]+(1.0f-a)*c1c; } }
        for(int k=0;k<L0_DIM;k++){ fastL[k]=BIL_FAST*fastL[k]+(1.0f-BIL_FAST)*l0n[k]; slowL[k]=BIL_SLOW*slowL[k]+(1.0f-BIL_SLOW)*l0n[k]; }
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
    long* p=malloc(n*sizeof(long)); for(long i=0;i<n;i++) p[i]=i; uint64_t r=seed;
    for(long i=n-1;i>0;i--){ r^=r<<13;r^=r>>7;r^=r<<17; long j=(long)(r%(uint64_t)(i+1)); long t=p[i];p[i]=p[j];p[j]=t; } return p; }

static void run_probe(const char* name,int H,const Seg* segs,int nseg,const long* perm,long permFromOff,
                      Win* tr,Win* va,long N,int EP){
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
    gen_lift(OmA,BvA,DA,SEED_A); gen_lift(OmM0,BvM0,DM,SEED_M0); gen_lift(OmM1,BvM1,DM,SEED_M1); gen_perm(bilperm,DB,SEED_BILP);
    ent_table=malloc(CLASSES*CLASSES*4);
    for(int i=0;i<CLASSES;i++) for(int j=0;j<CLASSES;j++){ float m=-1e9f; for(int k=0;k<CLASSES;k++) if(trigram[i][j][k]>m)m=trigram[i][j][k];
        double se=0; for(int k=0;k<CLASSES;k++) se+=exp((double)(trigram[i][j][k]-m));
        double Hh=0; for(int k=0;k<CLASSES;k++){ double p=exp((double)(trigram[i][j][k]-m))/se; if(p>1e-12)Hh-=p*log(p); }
        ent_table[i][j]=(float)Hh; }

    long tr_start=g_fsz/5;
    long va_start[N_VAL]={ g_fsz/2, (long)(0.65*g_fsz), (long)(0.80*g_fsz) };
    for(int w=0;w<N_VAL;w++) if(va_start[w]+N+3>g_fsz){ fprintf(stderr,"val window %d out of file\n",w+1); return 1; }
    fprintf(stderr,"48.B: train @%ld val @%ld/%ld/%ld (N=%ld) H=%d ep=%d X_DIM=%d (~%.1f GB feats)\n",
            tr_start,va_start[0],va_start[1],va_start[2],N,H,EP,X_DIM,(double)N*(N_VAL+1)*X_DIM*4/1e9);

    Win tr; Win va[N_VAL];
    tr.X=malloc((size_t)N*X_DIM*4); tr.tgt=malloc(N); tr.c1=malloc(N); tr.c2=malloc(N); tr.start=tr_start; tr.perm=make_perm(N,0x48BB5EEDULL);
    for(int w=0;w<N_VAL;w++){ va[w].X=malloc((size_t)N*X_DIM*4); va[w].tgt=malloc(N); va[w].c1=malloc(N); va[w].c2=malloc(N);
        va[w].start=va_start[w]; va[w].perm=make_perm(N,0x48BB5EEDULL^(uint64_t)(w+1)*0x9E3779B9ULL); }
    if(!tr.X||!va[0].X||!va[1].X||!va[2].X){ fprintf(stderr,"OOM\n"); return 1; }
    fprintf(stderr,"extracting train window...\n");
    extract_window(&see,tr_start,N,tr.X,tr.tgt,tr.c1,tr.c2,&tr.tri,&tr.d1);
    for(int w=0;w<N_VAL;w++){ fprintf(stderr,"extracting val window %d...\n",w+1);
        extract_window(&see,va_start[w],N,va[w].X,va[w].tgt,va[w].c1,va[w].c2,&va[w].tri,&va[w].d1); }

    printf("\n==== 48.B substrate feature classes (base=armB frozen, N=%ld, ep=%d, H=%d) ====\n",N,EP,H);
    printf("BASE win=train tri=%.4f d1=%.4f\n",tr.tri,tr.d1);
    for(int w=0;w<N_VAL;w++) printf("BASE win=val%d tri=%.4f d1=%.4f\n",w+1,va[w].tri,va[w].d1);

    Seg s_base[1]={{0,D1_TOT}};
    Seg s_armb[2]={{0,D1_TOT},{OFF_ARMB,ARMB_BANDS}};                 // the NEW baseline
    Seg s_bil[3] ={{0,D1_TOT},{OFF_ARMB,ARMB_BANDS},{OFF_BIL,BIL_BANDS}};
    Seg s_bilp[3]={{0,D1_TOT},{OFF_ARMB,ARMB_BANDS},{OFF_BILSH,BIL_BANDS}};   // shuffled pairing
    Seg s_wav[3] ={{0,D1_TOT},{OFF_ARMB,ARMB_BANDS},{OFF_WAV,WAV_BANDS}};
    Seg s_mb[3]  ={{0,D1_TOT},{OFF_ARMB,ARMB_BANDS},{OFF_MB,MB_BANDS}};

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

    run_probe("armB",     H,s_armb,2,NULL,X_DIM+1,&tr,va,N,EP);                 // baseline (reproduces 48.0 armB)
    run_probe("BILIN",    H,s_bil, 3,NULL,X_DIM+1,&tr,va,N,EP);
    run_probe("BILIN_sp", H,s_bilp,3,NULL,X_DIM+1,&tr,va,N,EP);                 // shuffled-pairing control
    run_probe("BILIN_st", H,s_bil, 3,(const long*)1,OFF_BIL,&tr,va,N,EP);       // shuffled-time leak guard
    run_probe("WAVE32",   H,s_wav, 3,NULL,X_DIM+1,&tr,va,N,EP);
    run_probe("WAVE32_st",H,s_wav, 3,(const long*)1,OFF_WAV,&tr,va,N,EP);
    run_probe("MULTIBW",  H,s_mb,  3,NULL,X_DIM+1,&tr,va,N,EP);
    run_probe("MULTIBW_st",H,s_mb, 3,(const long*)1,OFF_MB,&tr,va,N,EP);

    printf("\nPre-registered criterion (decided BEFORE the run): each arm (BILIN/WAVE32/MULTIBW) earns the\n");
    printf("48.B.A DAgger run iff it beats armB-alone by >= 0.015 on ALL of val1/val2/val3, with clean\n");
    printf("controls: BILIN_sp (shuffled pairing) and *_st (shuffled-time) must NOT beat armB by > 0.005.\n");
    printf("frozenD1 must == BASE d1. Winners are stackable. TF-only != generative (44-47): the verdict\n");
    printf("for any winner is the closed-loop gate v2 + replicas + human reading, exactly as armB.\n");
    return 0;
}
