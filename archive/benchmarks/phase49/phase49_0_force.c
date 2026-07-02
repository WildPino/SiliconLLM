// Phase 49.0 - FORCE-flavored output-feedback feasibility (cheap-first)
//
// Phase 48 closed the static per-step axis (armB at the boundary; B/C/D negative with a reason
// each). Closed-loop armB = structured word-salad with short cyclic loops = a GENERATION-DYNAMICS
// problem, not a feature problem. 49.0 pulls a generator from the FIXED reservoir by closing an
// output-feedback loop and training the readout ON the loop, with NO backprop into the substrate
// and NO BPTT. It reuses the entire 47/48 DAgger+gate machinery; it changes one thing: a feedback
// reservoir h_t is appended to the armB feature, and in DAgger rollout bursts the feedback byte is
// the model's OWN sample (on-policy).
//
// FEEDBACK RESERVOIR (all FIXED random, echo-state, no backprop):
//   h_t = (1-ALPHA)*h_{t-1} + ALPHA*tanh( W_in.phi_armB(s_t) + W_fb[prev_byte] + W_rec.h_{t-1} + b )
//   - phi_armB(s_t) = the 512-D armB feature [SEE 192 | L2 64 | armB-cos 256].
//   - prev_byte = the previous emitted/observed byte (cur_c1); W_fb[prev_byte] = its fixed random
//     embedding = output-feedback (in TF prev_byte is the true byte -> h is model-independent and
//     the base windows extract once; in a rollout burst prev_byte is the sampled byte -> on-policy,
//     so h carries the model's OWN trajectory and the readout learns to correct from it).
//   - W_rec scaled by circular law (sigma = RHO/sqrt(D_H)) so the spectral radius ~ RHO < 1; the
//     leaky update + tanh add contraction. Echo-state verified empirically at startup (ECHO_STATE).
//   - NO-FB control (--nofb): identical but the W_fb term is dropped -> isolates whether the
//     output-feedback SPECIFICALLY helps, or just the extra reservoir capacity (the LIN_dyn/PARITY
//     analogue). If FORCE ~= NO-FB, the feedback is inert.
//
// STOP-GRADIENT: h_t is precomputed during extraction and STORED in the feature row, so the
// existing readout trainer sees a fixed (S_DIM + D_H)-wide input - no gradient flows into
// W_in/W_fb/W_rec, no BPTT, no backprop into the substrate. Only the H32 readout learns.
//
// Everything else (D1 base, armB lift, DAgger 47.I recipe with K16/K128-whitespace/char-flood
// bursts, premise, per-round checkpoints) is phase48a_fix verbatim. New save magic 0x53454550
// carries the feedback config so the generator rebuilds the loop.
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase38-42/phase49_0_force.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase49_0_force.exe -lm -I .
// Run:
//   bin/phase49_0_force.exe <data> <D1_w> <outprefix> [--len N] [--nofb]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include "src/silicon_entropy.h"

#define CLASSES   256
#define BASE_DIM  SEE_FEATURE_DIM
#define L0_DIM    SEE_L0_DIM
#define L2_DIM    64
#define D1_TOT    (BASE_DIM + L2_DIM)
#define D_EXP     128
#define GAMMA     0.25f
#define N_TS      2
#define EXP_BANDS (N_TS*D_EXP)
#define S_DIM     (D1_TOT + EXP_BANDS)   // 512 = phi_armB width
#define D_H       256                    // feedback reservoir width
#define FEAT      (S_DIM + D_H)          // 768 = readout input
#define ALPHA_LEAK 0.30f
#define RHO        0.90f                 // target spectral radius of W_rec (echo-state)
#define G_IN       1.0f                  // input scaling
#define G_FB       0.6f                  // output-feedback (byte embedding) scaling
#define FB_SEED_IN  0x49001A2BULL
#define FB_SEED_FB  0x49002C3DULL
#define FB_SEED_REC 0x49003E4FULL
#define MAGIC_FB   0x53454550u
#define PROJ_SEED 0x48B2EC0DEULL
#define N_VAL     3
#define T_HI      0.65f
#define RECOV     16
#define PERIOD    256
#define EP_PRE    2
#define EP_MIX    2
#define R_PRE     5
#define R_TOT     9
#define K_NEAR    16
#define K_FAR     128
#define S_CF      12
#define K_CF      24
#define CF_STAY   6

enum { G_ENTROPY=4 };
enum { X_CLEAN=0, X_CORRUPT=1 };
enum { BM_NEAR=0, BM_WS=1, BM_CF=2 };

static const float TS_ALPHA[N_TS] = { 0.90f, 0.99f };

static float Pmat[L2_DIM][BASE_DIM];
static float Omega[D_EXP][L0_DIM];
static float Bvec[D_EXP];
static float W_in[D_H][S_DIM], W_fb[D_H][CLASSES], W_rec[D_H][D_H], bh[D_H];
static int   g_use_fb=1;
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
static void gn_fill(float* dst, long n, float scale, uint64_t seed){ uint64_t s=seed?seed:0x1234ULL;
    for(long i=0;i<n;i++){ double u1=xs_u01(&s); if(u1<1e-12)u1=1e-12; double u2=xs_u01(&s);
        dst[i]=(float)(sqrt(-2.0*log(u1))*cos(6.283185307179586*u2))*scale; } }
static void gen_lift(uint64_t seed){ uint64_t s=seed?seed:0xABCDEF12345ULL;
    for(int d=0;d<D_EXP;d++){
        for(int k=0;k<L0_DIM;k++){ double u1=xs_u01(&s); if(u1<1e-12) u1=1e-12; double u2=xs_u01(&s);
            Omega[d][k]=(float)(sqrt(-2.0*log(u1))*cos(6.283185307179586*u2)); }
        Bvec[d]=(float)(6.283185307179586*xs_u01(&s)); } }
// Build the fixed feedback reservoir matrices. W_rec uses sigma=RHO/sqrt(D_H) so the circular-law
// spectral radius ~ RHO < 1 (echo-state). Deterministic from seeds -> the generator rebuilds it.
static void gen_feedback(void){
    gn_fill(&W_in[0][0], (long)D_H*S_DIM, G_IN/sqrtf((float)S_DIM), FB_SEED_IN);
    gn_fill(&W_fb[0][0], (long)D_H*CLASSES, G_FB, FB_SEED_FB);
    gn_fill(&W_rec[0][0],(long)D_H*D_H,     RHO/sqrtf((float)D_H), FB_SEED_REC);
    memset(bh,0,sizeof bh);
}
static inline void update_h(const float* phi, uint8_t prev, const float* h_prev, float* h_out){
    for(int k=0;k<D_H;k++){
        float pre=bh[k]+dot_avx(W_in[k],phi,S_DIM)+dot_avx(W_rec[k],h_prev,D_H);
        if(g_use_fb) pre+=W_fb[k][prev];
        h_out[k]=(1.0f-ALPHA_LEAK)*h_prev[k]+ALPHA_LEAK*tanhf(pre);
    }
}
// Echo-state sanity: two trajectories from h0=0 and h0=random on the SAME (synthetic, bounded)
// input must converge -> the state forgets its init. ratio = ||h_a-h_b||_T / ||h_a-h_b||_0 -> 0.
static void echo_state_test(void){
    uint64_t s=0xE5705EEDULL; float ha[D_H],hb[D_H],hna[D_H],hnb[D_H],phi[S_DIM];
    memset(ha,0,sizeof ha); for(int k=0;k<D_H;k++) hb[k]=(float)(xs_u01(&s)*2-1);
    double dinit=0; for(int k=0;k<D_H;k++){ double e=ha[k]-hb[k]; dinit+=e*e; } dinit=sqrt(dinit);
    double dT=0;
    for(int t=0;t<2000;t++){
        for(int j=0;j<S_DIM;j++) phi[j]=(float)(xs_u01(&s)*2-1);
        uint8_t pb=(uint8_t)(xs_u01(&s)*256);
        update_h(phi,pb,ha,hna); update_h(phi,pb,hb,hnb);
        memcpy(ha,hna,sizeof ha); memcpy(hb,hnb,sizeof hb);
        if(t==1999){ for(int k=0;k<D_H;k++){ double e=ha[k]-hb[k]; dT+=e*e; } dT=sqrt(dT); }
    }
    printf("ECHO_STATE use_fb=%d d0=%.4f dT=%.6e ratio=%.6e alpha=%.2f rho=%.2f D_H=%d (ratio->0 = ESP holds)\n",
           g_use_fb,dinit,dT,(dinit>0?dT/dinit:0.0),ALPHA_LEAK,RHO,D_H);
}
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
static inline void armb_bands(const float* feat192, float eB[N_TS][D_EXP], float* row){
    for(int ts=0;ts<N_TS;ts++) memcpy(row+D1_TOT+ts*D_EXP, eB[ts], D_EXP*4);
    float l0n[L0_DIM];
    for(int k=0;k<L0_DIM;k++){ float x=(feat192[k]-md1[k])/(sd1[k]+1e-8f); if(x>2.f)x=2.f; if(x<-2.f)x=-2.f; l0n[k]=x; }
    for(int d=0;d<D_EXP;d++){ float z=Bvec[d]+GAMMA*dot_avx(Omega[d],l0n,L0_DIM); float cz=cosf(z);
        for(int ts=0;ts<N_TS;ts++){ float a=TS_ALPHA[ts]; eB[ts][d]=a*eB[ts][d]+(1.0f-a)*cz; } }
}
static inline double d1_logits(const float* nf, uint8_t c1,uint8_t c2,uint8_t tgt, float* probsT, float temp){
    const float* tri=&trigram[c2][c1][0]; float lg[CLASSES],mx=-1e30f;
    for(int c=0;c<CLASSES;c++){ lg[c]=Bd1[c]+tri[c]+dot_avx(Wd1[c],nf,D1_TOT); if(lg[c]>mx)mx=lg[c]; }
    double Z=0; for(int c=0;c<CLASSES;c++) Z+=exp((double)(lg[c]-mx));
    if(probsT){ float mt=-1e30f; for(int c=0;c<CLASSES;c++){ float v=lg[c]/temp; if(v>mt)mt=v; }
        double Zt=0; for(int c=0;c<CLASSES;c++){ probsT[c]=expf(lg[c]/temp-mt); Zt+=probsT[c]; }
        for(int c=0;c<CLASSES;c++) probsT[c]=(float)(probsT[c]/Zt); }
    double p=exp((double)(lg[tgt]-mx))/Z; return -log2(p>1e-30?p:1e-30);
}
static inline double trigram_bpb(uint8_t c1,uint8_t c2,uint8_t tgt){
    const float* tri=&trigram[c2][c1][0]; float mx=-1e30f;
    for(int c=0;c<CLASSES;c++) if(tri[c]>mx)mx=tri[c];
    double Z=0; for(int c=0;c<CLASSES;c++) Z+=exp((double)(tri[c]-mx));
    double p=exp((double)(tri[tgt]-mx))/Z; return -log2(p>1e-30?p:1e-30);
}
static inline uint8_t sample_p(const float* P, uint64_t* rng){
    *rng^=*rng<<13; *rng^=*rng>>7; *rng^=*rng<<17;
    double u=(*rng>>11)*(1.0/(1ULL<<53));
    double c=0; for(int k=0;k<CLASSES;k++){ c+=P[k]; if(u<=c) return (uint8_t)k; }
    return CLASSES-1;
}

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
static double mlp_eval(MLP* m,const float* X,const uint8_t* tgt,const uint8_t* c1a,const uint8_t* c2a,long n,
                       double* o_rms,double* o_ent,double* o_maxp){
    float* hid=malloc(((m->h>0)?m->h:1)*4); float lg[CLASSES]; double tot=0,arms=0,aent=0,amax=0;
    for(long i=0;i<n;i++){
        mlp_fwd(m,&X[(size_t)i*FEAT],hid,lg);
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

typedef struct { float *gW1,*gb1,*gW2,*gb2; } Grads;
static void accum_sample(MLP* m,const float* x,uint8_t tgt,uint8_t c1,uint8_t c2,
                         const float* xpair,float lamC,float invnorm,
                         Grads* G,float* hid,float* hidP,float* dh){
    int H=m->h,in=m->in;
    float lg[CLASSES],eo[CLASSES];
    mlp_fwd(m,x,hid,lg);
    float lgp[CLASSES]; memcpy(lgp,lg,sizeof(lgp));
    const float* tri=&trigram[c2][c1][0]; for(int c=0;c<CLASSES;c++) lg[c]+=tri[c];
    float mx=-1e30f; for(int c=0;c<CLASSES;c++) if(lg[c]>mx)mx=lg[c];
    float Z=0; for(int c=0;c<CLASSES;c++){ eo[c]=expf(lg[c]-mx); Z+=eo[c]; }
    for(int c=0;c<CLASSES;c++){ float y=(c==tgt)?1.f:0.f; eo[c]=(eo[c]/Z-y)*invnorm; }
    if(xpair && lamC>0){
        float lgC[CLASSES]; mlp_fwd(m,xpair,hidP,lgC);
        float cc=lamC*invnorm/CLASSES;
        for(int c=0;c<CLASSES;c++) eo[c]+=cc*(lgp[c]-lgC[c]);
    }
    for(int c=0;c<CLASSES;c++) G->gb2[c]+=eo[c];
    memset(dh,0,H*4);
    for(int c=0;c<CLASSES;c++){ float e=eo[c]; float* gw=&G->gW2[(size_t)c*H]; const float* w2=&m->W2[(size_t)c*H];
        for(int j=0;j<H;j++){ gw[j]+=e*hid[j]; dh[j]+=e*w2[j]; } }
    for(int j=0;j<H;j++) if(hid[j]>0){ G->gb1[j]+=dh[j]; float* gw=&G->gW1[(size_t)j*in]; const float g=dh[j];
        for(int k=0;k<in;k++) gw[k]+=g*x[k]; }
}
static void train_mixed(MLP* m,const float* X0,const uint8_t* t0,const uint8_t* c10,const uint8_t* c20,long n,
                        const float* Xr,const uint8_t* tr,const uint8_t* c1r,const uint8_t* c2r,const long* posr,long npool,
                        int num,int den,float lamC,int epochs,float lr){
    int H=m->h,in=m->in;
    size_t s1=(size_t)H*in, s2=(size_t)CLASSES*H;
    Grads G; G.gW1=malloc(s1*4); G.gb1=malloc(H*4); G.gW2=malloc(s2*4); G.gb2=malloc(CLASSES*4);
    float* hid=malloc(H*4); float* hidP=malloc(H*4); float* dh=malloc(H*4); int bs=512;
    float invnorm=1.0f/bs;
    long kc=0,kr=0;
    for(int ep=0;ep<epochs;ep++){
        memset(G.gW1,0,s1*4); memset(G.gb1,0,H*4); memset(G.gW2,0,s2*4); memset(G.gb2,0,CLASSES*4); long inb=0;
        for(long s=0;s<n;s++){
            int use_roll = (npool>0) && ((s%den)<num);
            if(use_roll){
                long j=kr%npool; kr++;
                accum_sample(m,&Xr[(size_t)j*FEAT],tr[j],c1r[j],c2r[j],
                             &X0[(size_t)posr[j]*FEAT],lamC,invnorm,&G,hid,hidP,dh);
            } else {
                long k=kc%n; kc++;
                accum_sample(m,&X0[(size_t)k*FEAT],t0[k],c10[k],c20[k],NULL,0,invnorm,&G,hid,hidP,dh);
            }
            inb++;
            if(inb==bs || s==n-1){ m->t++; float lt=lr*sqrtf(1-powf(.999f,m->t))/(1-powf(.9f,m->t));
                #define ADAM(P,GG,MM,VV,NN) for(size_t z=0;z<(size_t)(NN);z++){ MM[z]=.9f*MM[z]+.1f*GG[z]; VV[z]=.999f*VV[z]+.001f*GG[z]*GG[z]; P[z]-=lt*(MM[z]/(sqrtf(VV[z])+1e-8f)+1e-5f*P[z]); }
                ADAM(m->W1,G.gW1,m->mW1,m->vW1,s1); ADAM(m->b1,G.gb1,m->mb1,m->vb1,H);
                ADAM(m->W2,G.gW2,m->mW2,m->vW2,s2); ADAM(m->b2,G.gb2,m->mb2,m->vb2,CLASSES);
                memset(G.gW1,0,s1*4); memset(G.gb1,0,H*4); memset(G.gW2,0,s2*4); memset(G.gb2,0,CLASSES*4); inb=0; }
        }
        fprintf(stderr,"    ep %d/%d done\n",ep+1,epochs);
    }
    free(G.gW1);free(G.gb1);free(G.gW2);free(G.gb2);free(hid);free(hidP);free(dh);
}
// save 0x53454550: armB lift header + feedback config (D_H, alpha, rho, scales, seeds, use_fb).
static void mlp_save(MLP* m,const char* path){
    FILE* f=fopen(path,"wb"); if(!f){ fprintf(stderr,"save %s failed\n",path); return; }
    uint32_t magic=MAGIC_FB,H=(uint32_t)m->h,d=(uint32_t)m->in,base=D1_TOT,dexp=D_EXP,nts=N_TS;
    float gamma=GAMMA; uint64_t pseed=PROJ_SEED;
    fwrite(&magic,4,1,f); fwrite(&H,4,1,f); fwrite(&d,4,1,f); fwrite(&base,4,1,f);
    fwrite(&dexp,4,1,f); fwrite(&nts,4,1,f); fwrite(&gamma,4,1,f); fwrite(&pseed,8,1,f);
    for(int t=0;t<N_TS;t++){ float a=TS_ALPHA[t]; fwrite(&a,4,1,f); }
    uint32_t dh=D_H, ufb=(uint32_t)g_use_fb;
    float alk=ALPHA_LEAK, rho=RHO, gin=G_IN, gfb=G_FB; uint64_t si=FB_SEED_IN,sf=FB_SEED_FB,sr=FB_SEED_REC;
    fwrite(&dh,4,1,f); fwrite(&ufb,4,1,f); fwrite(&alk,4,1,f); fwrite(&rho,4,1,f); fwrite(&gin,4,1,f); fwrite(&gfb,4,1,f);
    fwrite(&si,8,1,f); fwrite(&sf,8,1,f); fwrite(&sr,8,1,f);
    fwrite(m->W1,4,(size_t)m->h*m->in,f); fwrite(m->b1,4,m->h,f); fwrite(m->W2,4,(size_t)CLASSES*m->h,f);
    fwrite(m->b2,4,CLASSES,f); fclose(f);
}

// Teacher-forced (or D1-corrupt) extraction. Row = [phi_armB 512 | h 256]. h_t uses the prev
// observed byte (cur_c1) as feedback - model-independent in TF (so base windows extract once).
static void extract_window_x(SiliconEntropyState* see, long start, long N, int mode, float p, uint64_t rseed,
                             float* X, uint8_t* tgt, uint8_t* c1a, uint8_t* c2a,
                             double* oTRI, double* oD1, double* o_corr){
    float L2d1[L2_DIM]={0}, pb_d1[BASE_DIM]={0}, eB[N_TS][D_EXP];
    memset(eB,0,sizeof eB);
    float feat192[BASE_DIM], fa[BASE_DIM], rawd1[D1_TOT], nf[D1_TOT], probsT[CLASSES], h[D_H], hn[D_H];
    memset(h,0,sizeof h);
    float scale=1.0f/sqrtf((float)BASE_DIM); double btr=0,bd1=0; long ncorr=0;
    uint64_t rng=rseed?rseed:0x9E3779B97F4A7C15ULL;
    see_reset(see); for(long i=0;i<=start+1;i++) see_observe(see,g_data[i]);
    uint8_t cur_c2=g_data[start], cur_c1=g_data[start+1];
    for(long i=0;i<N;i++){
        long g=start+i; uint8_t t=g_data[g+2];
        see_extract(see,feat192);
        memcpy(rawd1,feat192,BASE_DIM*4); memcpy(rawd1+BASE_DIM,L2d1,L2_DIM*4);
        btr+=trigram_bpb(cur_c1,cur_c2,t);
        norm_feats(rawd1,nf);
        float* row=&X[(size_t)i*FEAT];
        memcpy(row,nf,D1_TOT*4);
        armb_bands(feat192,eB,row);
        update_h(row,cur_c1,h,hn); memcpy(row+S_DIM,hn,D_H*4); memcpy(h,hn,sizeof h);
        int dirty=0;
        if(mode==X_CORRUPT){ rng^=rng<<13;rng^=rng>>7;rng^=rng<<17; if(((rng>>11)*(1.0/(1ULL<<53)))<p) dirty=1; }
        bd1+=d1_logits(nf,cur_c1,cur_c2,t,dirty?probsT:NULL,T_HI);
        uint8_t ob=t;
        if(dirty){ ob=sample_p(probsT,&rng); ncorr++; }
        tgt[i]=t; c1a[i]=cur_c1; c2a[i]=cur_c2;
        see_observe(see,ob); see_extract(see,fa);
        if(ent_gate(cur_c1,cur_c2)){ float src5[BASE_DIM];
            for(int k=0;k<BASE_DIM;k++) src5[k]=fa[k]-0.5f*pb_d1[k]; memcpy(pb_d1,fa,BASE_DIM*4);
            for(int j=0;j<L2_DIM;j++){ float p5=0; const float* pj=Pmat[j]; for(int k=0;k<BASE_DIM;k++) p5+=pj[k]*src5[k];
                L2d1[j]=g_alpha*L2d1[j]+(1.0f-g_alpha)*p5*scale; } }
        cur_c2=cur_c1; cur_c1=ob;
    }
    *oTRI=btr/N; *oD1=bd1/N; if(o_corr)*o_corr=100.0*ncorr/N;
}

// Rollout extraction: in bursts the decoder samples its own bytes -> h's feedback (cur_c1) becomes
// the model's OWN sample = ON-POLICY output-feedback (the FORCE-essential bit). Three burst types.
static long extract_rollout_cf(SiliconEntropyState* see, MLP* m, long start, long N, int kfar, int kcf, uint64_t rseed,
                               float* Xr, uint8_t* tr, uint8_t* c1r, uint8_t* c2r, long* posr, long npool_max,
                               double* o_s16, double* o_s128, double* o_scf){
    float L2d1[L2_DIM]={0}, pb_d1[BASE_DIM]={0}, eB[N_TS][D_EXP];
    memset(eB,0,sizeof eB);
    float feat192[BASE_DIM], fa[BASE_DIM], rawd1[D1_TOT], nf[D1_TOT], row[FEAT], h[D_H], hn[D_H];
    memset(h,0,sizeof h);
    float* hid=malloc(((m->h>0)?m->h:1)*4); float lg[CLASSES],Pp[CLASSES];
    float scale=1.0f/sqrtf((float)BASE_DIM);
    double b16=0,b128=0,bcf=0; long n16=0,n128=0,ncf=0, np=0;
    uint64_t rng=rseed?rseed:0x9E3779B97F4A7C15ULL;
    int burst=0, bmode=BM_NEAR, cf_seed_left=0; uint8_t cf_char=0; long since=RECOV+1, bidx=0;
    see_reset(see); for(long i=0;i<=start+1;i++) see_observe(see,g_data[i]);
    uint8_t cur_c2=g_data[start], cur_c1=g_data[start+1];
    for(long i=0;i<N;i++){
        long g=start+i; uint8_t t=g_data[g+2];
        see_extract(see,feat192);
        memcpy(rawd1,feat192,BASE_DIM*4); memcpy(rawd1+BASE_DIM,L2d1,L2_DIM*4);
        norm_feats(rawd1,nf);
        memcpy(row,nf,D1_TOT*4);
        armb_bands(feat192,eB,row);
        update_h(row,cur_c1,h,hn); memcpy(row+S_DIM,hn,D_H*4); memcpy(h,hn,sizeof h);
        if(i>0 && (i%PERIOD)==0){
            int slot=(int)(bidx%8);
            if(kfar && slot==7){ burst=K_FAR; bmode=BM_WS; }
            else if(kcf && slot==3){ burst=S_CF+K_CF; bmode=BM_CF; cf_seed_left=S_CF; cf_char=cur_c1; }
            else { burst=K_NEAR; bmode=BM_NEAR; }
            bidx++;
        }
        int in_burst=(burst>0), in_recov=(!in_burst && since<RECOV);
        if((in_burst||in_recov) && np<npool_max){
            memcpy(&Xr[(size_t)np*FEAT],row,FEAT*4);
            tr[np]=t; c1r[np]=cur_c1; c2r[np]=cur_c2; posr[np]=i; np++;
        }
        uint8_t ob=t;
        if(in_burst){
            if(bmode==BM_CF && cf_seed_left>0){
                ob=cf_char; cf_seed_left--;
            } else {
                mlp_fwd(m,row,hid,lg);
                const float* tri=&trigram[cur_c2][cur_c1][0];
                float mx=-1e30f; for(int c=0;c<CLASSES;c++){ lg[c]=(lg[c]+tri[c])/T_HI; if(lg[c]>mx)mx=lg[c]; }
                float Z=0; for(int c=0;c<CLASSES;c++){ Pp[c]=expf(lg[c]-mx); Z+=Pp[c]; } for(int c=0;c<CLASSES;c++) Pp[c]/=Z;
                ob=sample_p(Pp,&rng);
                double sb=-log2((double)fmaxf(Pp[ob],1e-30f));
                if(bmode==BM_WS){ b128+=sb; n128++; } else if(bmode==BM_CF){ bcf+=sb; ncf++; } else { b16+=sb; n16++; }
            }
            burst--; if(burst==0) since=0;
        } else since++;
        see_observe(see,ob); see_extract(see,fa);
        if(ent_gate(cur_c1,cur_c2)){ float src5[BASE_DIM];
            for(int k=0;k<BASE_DIM;k++) src5[k]=fa[k]-0.5f*pb_d1[k]; memcpy(pb_d1,fa,BASE_DIM*4);
            for(int j=0;j<L2_DIM;j++){ float p5=0; const float* pj=Pmat[j]; for(int k=0;k<BASE_DIM;k++) p5+=pj[k]*src5[k];
                L2d1[j]=g_alpha*L2d1[j]+(1.0f-g_alpha)*p5*scale; } }
        cur_c2=cur_c1; cur_c1=ob;
    }
    free(hid);
    *o_s16=(n16>0)?b16/n16:0.0; *o_s128=(n128>0)?b128/n128:0.0; *o_scf=(ncf>0)?bcf/ncf:0.0;
    return np;
}

// PREMISE (NO training): seed char-floods, roll from the decoder, measure stay% (basin reachable).
static void premise_charflood(SiliconEntropyState* see, MLP* m, long start, long N, uint64_t rseed,
                              double* o_stayPct, double* o_meanCont, long* o_nprobe){
    float L2d1[L2_DIM]={0}, pb_d1[BASE_DIM]={0}, eB[N_TS][D_EXP];
    memset(eB,0,sizeof eB);
    float feat192[BASE_DIM], fa[BASE_DIM], rawd1[D1_TOT], nf[D1_TOT], row[FEAT], h[D_H], hn[D_H];
    memset(h,0,sizeof h);
    float* hid=malloc(((m->h>0)?m->h:1)*4); float lg[CLASSES],Pp[CLASSES];
    float scale=1.0f/sqrtf((float)BASE_DIM);
    uint64_t rng=rseed?rseed:0x9E3779B97F4A7C15ULL;
    long nprobe=0, nstay=0; double sumCont=0;
    int burst=0, seed_left=0;
    int lead_run=0, lead_done=0, maxrun=0, run_cur=0, prev_roll=-1;
    uint8_t cf_char=0;
    see_reset(see); for(long i=0;i<=start+1;i++) see_observe(see,g_data[i]);
    uint8_t cur_c2=g_data[start], cur_c1=g_data[start+1];
    for(long i=0;i<N;i++){
        see_extract(see,feat192);
        memcpy(rawd1,feat192,BASE_DIM*4); memcpy(rawd1+BASE_DIM,L2d1,L2_DIM*4);
        norm_feats(rawd1,nf);
        memcpy(row,nf,D1_TOT*4);
        armb_bands(feat192,eB,row);
        update_h(row,cur_c1,h,hn); memcpy(row+S_DIM,hn,D_H*4); memcpy(h,hn,sizeof h);
        if(burst==0 && i>0 && (i%PERIOD)==0){
            burst=S_CF+K_CF; seed_left=S_CF; cf_char=cur_c1;
            lead_run=0; lead_done=0; maxrun=0; run_cur=0; prev_roll=-1;
        }
        uint8_t ob;
        if(burst>0){
            if(seed_left>0){ ob=cf_char; seed_left--; }
            else {
                mlp_fwd(m,row,hid,lg);
                const float* tri=&trigram[cur_c2][cur_c1][0];
                float mx=-1e30f; for(int c=0;c<CLASSES;c++){ lg[c]=(lg[c]+tri[c])/T_HI; if(lg[c]>mx)mx=lg[c]; }
                float Z=0; for(int c=0;c<CLASSES;c++){ Pp[c]=expf(lg[c]-mx); Z+=Pp[c]; } for(int c=0;c<CLASSES;c++) Pp[c]/=Z;
                ob=sample_p(Pp,&rng);
                if(!lead_done){ if(ob==cf_char) lead_run++; else lead_done=1; }
                if((int)ob==prev_roll) run_cur++; else run_cur=1;
                if(run_cur>maxrun) maxrun=run_cur; prev_roll=(int)ob;
            }
            burst--;
            if(burst==0){ nprobe++; sumCont+=lead_run; if(maxrun>=CF_STAY) nstay++; }
        } else ob=g_data[start+i+2];
        see_observe(see,ob); see_extract(see,fa);
        if(ent_gate(cur_c1,cur_c2)){ float src5[BASE_DIM];
            for(int k=0;k<BASE_DIM;k++) src5[k]=fa[k]-0.5f*pb_d1[k]; memcpy(pb_d1,fa,BASE_DIM*4);
            for(int j=0;j<L2_DIM;j++){ float p5=0; const float* pj=Pmat[j]; for(int k=0;k<BASE_DIM;k++) p5+=pj[k]*src5[k];
                L2d1[j]=g_alpha*L2d1[j]+(1.0f-g_alpha)*p5*scale; } }
        cur_c2=cur_c1; cur_c1=ob;
    }
    free(hid);
    *o_stayPct=(nprobe>0)?100.0*nstay/nprobe:0.0; *o_meanCont=(nprobe>0)?sumCont/nprobe:0.0; *o_nprobe=nprobe;
}

typedef struct { float* X; uint8_t *tgt,*c1,*c2; double tri,d1,corr; long start; } Win;
static Win alloc_win(long N,long start){
    Win w; w.X=malloc((size_t)N*FEAT*4); w.tgt=malloc(N); w.c1=malloc(N); w.c2=malloc(N);
    w.start=start; w.tri=0; w.d1=0; w.corr=0;
    if(!w.X||!w.tgt){ fprintf(stderr,"OOM\n"); exit(1); }
    return w;
}

int main(int argc, char** argv){
    if(argc<4){ fprintf(stderr,"Usage: %s <data> <D1_w> <outprefix> [--len N] [--nofb]\n",argv[0]); return 1; }
    setvbuf(stderr,NULL,_IONBF,0); setvbuf(stdout,NULL,_IONBF,0);
    long N=1000000;
    for(int i=4;i<argc;i++){ if(!strcmp(argv[i],"--len")&&i+1<argc)N=atol(argv[++i]);
        else if(!strcmp(argv[i],"--nofb")) g_use_fb=0; }
    FILE* fd=fopen(argv[1],"rb"); if(!fd){fprintf(stderr,"data\n");return 1;} fseek(fd,0,SEEK_END); g_fsz=ftell(fd); fseek(fd,0,SEEK_SET);
    g_data=malloc(g_fsz); fread(g_data,1,g_fsz,fd); fclose(fd);

    SiliconEntropyState see;
    if(!load_d1(argv[2],&see)) return 1;
    gen_lift(PROJ_SEED);
    gen_feedback();
    ent_table=malloc(CLASSES*CLASSES*4);
    for(int i=0;i<CLASSES;i++) for(int j=0;j<CLASSES;j++){ float m=-1e9f; for(int k=0;k<CLASSES;k++) if(trigram[i][j][k]>m)m=trigram[i][j][k];
        double se=0; for(int k=0;k<CLASSES;k++) se+=exp((double)(trigram[i][j][k]-m));
        double Hh=0; for(int k=0;k<CLASSES;k++){ double p=exp((double)(trigram[i][j][k]-m))/se; if(p>1e-12)Hh-=p*log(p); }
        ent_table[i][j]=(float)Hh; }

    echo_state_test();

    long tr_start=g_fsz/5;
    long va_start[N_VAL]={ g_fsz/2, (long)(0.65*g_fsz), (long)(0.80*g_fsz) };
    for(int w=0;w<N_VAL;w++) if(va_start[w]+N+3>g_fsz){ fprintf(stderr,"val window %d out of file\n",w+1); return 1; }
    fprintf(stderr,"49.0 FORCE(use_fb=%d): train @%ld val @%ld/%ld/%ld (N=%ld) FEAT=%d (S_DIM=%d+D_H=%d) rounds %d\n",
            g_use_fb,tr_start,va_start[0],va_start[1],va_start[2],N,FEAT,S_DIM,D_H,R_TOT);

    Win tr0 = alloc_win(N,tr_start);
    Win va[N_VAL]; for(int w=0;w<N_VAL;w++) va[w]=alloc_win(N,va_start[w]);
    Win vaC = alloc_win(N,va_start[0]);
    fprintf(stderr,"extract tr_clean...\n");
    extract_window_x(&see,tr_start,N,X_CLEAN,0,0, tr0.X,tr0.tgt,tr0.c1,tr0.c2,&tr0.tri,&tr0.d1,NULL);
    for(int w=0;w<N_VAL;w++){ fprintf(stderr,"extract val%d (clean)...\n",w+1);
        extract_window_x(&see,va_start[w],N,X_CLEAN,0,0, va[w].X,va[w].tgt,va[w].c1,va[w].c2,&va[w].tri,&va[w].d1,NULL); }
    fprintf(stderr,"extract valC (D1-corrupt p=0.05)...\n");
    extract_window_x(&see,va_start[0],N,X_CORRUPT,0.05f,0x7A1C0DE5ULL,vaC.X,vaC.tgt,vaC.c1,vaC.c2,&vaC.tri,&vaC.d1,&vaC.corr);

    printf("\n==== 49.0 FORCE feedback (use_fb=%d, H32, N=%ld, FEAT=%d) ====\n",g_use_fb,N,FEAT);
    printf("BASE win=train tri=%.4f d1=%.4f\n",tr0.tri,tr0.d1);
    for(int w=0;w<N_VAL;w++) printf("BASE win=val%d tri=%.4f d1=%.4f\n",w+1,va[w].tri,va[w].d1);
    printf("BASE win=valC tri=%.4f d1=%.4f corr%%=%.2f\n",vaC.tri,vaC.d1,vaC.corr);

    { double v[N_VAL]; float lg[CLASSES];
      for(int w=0;w<N_VAL;w++){ double tot=0;
        for(long i=0;i<N;i++){ const float* x=&va[w].X[(size_t)i*FEAT];
            for(int c=0;c<CLASSES;c++) lg[c]=Bd1[c]+dot_avx(Wd1[c],x,D1_TOT);
            const float* tri=&trigram[va[w].c2[i]][va[w].c1[i]][0]; for(int c=0;c<CLASSES;c++) lg[c]+=tri[c];
            float mx=-1e30f; for(int c=0;c<CLASSES;c++) if(lg[c]>mx)mx=lg[c];
            double Z=0; for(int c=0;c<CLASSES;c++) Z+=exp((double)(lg[c]-mx));
            double pp=exp((double)(lg[va[w].tgt[i]]-mx))/Z; tot+=-log2(pp>1e-30?pp:1e-30); }
        v[w]=tot/N; }
      printf("PROBE name=frozenD1 h=0 val1=%.4f val2=%.4f val3=%.4f\n",v[0],v[1],v[2]); }

    long npool_max=N*(K_FAR+RECOV)/PERIOD + 64;
    float* Xr=malloc((size_t)npool_max*FEAT*4);
    uint8_t* trr=malloc(npool_max); uint8_t* c1r=malloc(npool_max); uint8_t* c2r=malloc(npool_max);
    long* posr=malloc(npool_max*sizeof(long));
    if(!Xr||!posr){ fprintf(stderr,"OOM pool\n"); return 1; }

    MLP mp; mlp_init(&mp,FEAT,32);
    char sp[512];
    fprintf(stderr,"   pretrain clean %dep\n",EP_PRE);
    train_mixed(&mp,tr0.X,tr0.tgt,tr0.c1,tr0.c2,N, NULL,NULL,NULL,NULL,NULL,0, 0,1,0,EP_PRE,0.0005f);
    for(int rd=1;rd<=R_PRE;rd++){
        double s16=0,s128=0,scf=0;
        uint64_t seed=0x49A00000ULL^((uint64_t)rd*0x9E37ULL)^16ULL;
        fprintf(stderr,"   round %d (K16): extract rollout...\n",rd);
        long npool=extract_rollout_cf(&see,&mp,tr_start,N,0,0,seed,Xr,trr,c1r,c2r,posr,npool_max,&s16,&s128,&scf);
        fprintf(stderr,"   round %d: pool=%ld roll=%.3f; train mixed %dep\n",rd,npool,s16,EP_MIX);
        train_mixed(&mp,tr0.X,tr0.tgt,tr0.c1,tr0.c2,N, Xr,trr,c1r,c2r,posr,npool, 1,5,0.02f,EP_MIX,0.0005f);
        double v[N_VAL],vc,rms=0,ent=0,mxp=0;
        for(int w=0;w<N_VAL;w++) v[w]=mlp_eval(&mp,va[w].X,va[w].tgt,va[w].c1,va[w].c2,N,w==0?&rms:NULL,w==0?&ent:NULL,w==0?&mxp:NULL);
        vc=mlp_eval(&mp,vaC.X,vaC.tgt,vaC.c1,vaC.c2,N,NULL,NULL,NULL);
        snprintf(sp,sizeof sp,"%s_I_h32_r%d.bin",argv[3],rd); mlp_save(&mp,sp);
        printf("PROBE name=I_r%d h=32 roll=%.3f val1=%.4f val2=%.4f val3=%.4f valC=%.4f rms1=%.3f ent1=%.3f maxp1=%.3f\n",
               rd,s16,v[0],v[1],v[2],vc,rms,ent,mxp);
        printf("SAVED name=I_r%d path=%s\n",rd,sp);
    }

    { double stayPct=0,meanCont=0; long nprobe=0;
      fprintf(stderr,"   premise char-flood (no training, r5 decoder)...\n");
      premise_charflood(&see,&mp,tr_start,N,0xCF1005EEDULL,&stayPct,&meanCont,&nprobe);
      printf("PREMISE_CF nprobe=%ld stayPct=%.2f meanCont=%.3f stayThr=%d (seed S_CF=%d roll K_CF=%d)\n",
             nprobe,stayPct,meanCont,CF_STAY,S_CF,K_CF);
      fprintf(stderr,"   premise: stayPct=%.2f%% meanCont=%.3f (n=%ld)\n",stayPct,meanCont,nprobe); }

    for(int rd=R_PRE+1;rd<=R_TOT;rd++){
        double s16=0,s128=0,scf=0;
        uint64_t seed=0x49A00000ULL^((uint64_t)rd*0x9E37ULL)^16ULL^0x11400000ULL;
        fprintf(stderr,"   round %d (K16/K128ws/CF mix): extract rollout...\n",rd);
        long npool=extract_rollout_cf(&see,&mp,tr_start,N,1,1,seed,Xr,trr,c1r,c2r,posr,npool_max,&s16,&s128,&scf);
        fprintf(stderr,"   round %d: pool=%ld rollK16=%.3f rollK128=%.3f rollCF=%.3f; train mixed %dep\n",rd,npool,s16,s128,scf,EP_MIX);
        train_mixed(&mp,tr0.X,tr0.tgt,tr0.c1,tr0.c2,N, Xr,trr,c1r,c2r,posr,npool, 1,5,0.02f,EP_MIX,0.0005f);
        double v[N_VAL],vc,rms=0,ent=0,mxp=0;
        for(int w=0;w<N_VAL;w++) v[w]=mlp_eval(&mp,va[w].X,va[w].tgt,va[w].c1,va[w].c2,N,w==0?&rms:NULL,w==0?&ent:NULL,w==0?&mxp:NULL);
        vc=mlp_eval(&mp,vaC.X,vaC.tgt,vaC.c1,vaC.c2,N,NULL,NULL,NULL);
        snprintf(sp,sizeof sp,"%s_I_h32_r%d.bin",argv[3],rd); mlp_save(&mp,sp);
        printf("PROBE name=I_r%d h=32 rollK16=%.3f rollK128=%.3f rollCF=%.3f val1=%.4f val2=%.4f val3=%.4f valC=%.4f rms1=%.3f ent1=%.3f maxp1=%.3f\n",
               rd,s16,s128,scf,v[0],v[1],v[2],vc,rms,ent,mxp);
        printf("SAVED name=I_r%d path=%s\n",rd,sp);
    }
    mlp_free(&mp);

    printf("\nReading: ECHO_STATE ratio->0 = the feedback reservoir forgets its init (ESP). use_fb=1 is\n");
    printf("FORCE (output-feedback on-policy in bursts); use_fb=0 (--nofb) is the discriminant control.\n");
    printf("FORCE is a CLOSED-LOOP method - the verdict is gate v2 (FORCE beats NO-FB on the degeneracy\n");
    printf("channels) + replicas + human reading. TF != generative; no celebration before the gate.\n");
    return 0;
}
