// Phase 48.D - MODOJA-K: error-modulated Oja tilt of the armB kernel projection (TF probe)
//
// 48.B (static readout) and 48.C (fixed-random multiplicative dynamics) closed the "free"
// directions: the SIMILARITY (kernel) the silicon gets for free from randomness (RFF is an
// unbiased estimator, any seed); the RELATION/SELECTION (which direction matters) is NOT free
// and must be learned. Without backprop into the substrate, "learned" = local, error-modulated
// plasticity. 48.D puts the error signal exactly on the lever that paid off: the armB kernel
// projection P (the +0.04), NOT on the saturated linear L0 subspace. First time the error signal
// touches the substrate.
//
// IDEA: the fixed Omega inside cos(GAMMA*Omega*L0norm) of armB becomes an error-tilted P. A
// per-row Oja rule, learning-rate modulated by the surprise of a FROZEN trigram predictor, nudges
// P toward the L0 directions that recur where the cheap model fails. Same dims as armB -> zero
// capacity confound; the ONLY difference between the probes is HOW P is obtained.
//
// Base frozen = [SEE 192 | L2_D1 64] = 256. Kernel bands = [EMA_0.90(cos(GAMMA*P*L0norm)) ||
// EMA_0.99(cos(...))] = 256 -> probe dim 512. cos-features are bounded [-1,1] -> no explosion, no
// standardization (unlike 48.C's leaky reservoir). The only difference vs armB is P != Omega.
//
// PLASTICITY (local, no backprop, deterministic), ONE pass over the train window, byte t in order:
//   a_d = P_d . L0norm_t                              (per row d)
//   dP_d = eta_t * a_d * (L0norm_t - a_d * P_d)       (Oja)
//   renormalize P_d to its ORIGINAL ||Omega_d|| after the update   [see NORM NOTE]
//   eta_t = ETA0 * m_t
//   m_t = trigram surprise normalized to EXACT mean 1 over the train window (clamp [0.2,5] then
//         rescale so mean==1). surprise_t = trigram BPB (frozen, causal, no backprop).
//   P learned ONLY on train, then FROZEN -> zero leak into the val windows.
//
// NORM NOTE (deviation from the literal "renormalize to norm 1", flagged for the Architect):
//   Omega rows have norm ~8 (64 N(0,1) entries). Renormalizing the LEARNED P to unit norm on the
//   first step would jump the kernel bandwidth far from armB and break "init == armB" in any
//   post-plasticity sense -> a SCALE confound vs the armB baseline (which must keep natural-norm
//   Omega + GAMMA 0.25 to reproduce 2.2023/2.1839/2.1980). To isolate DIRECTION (the experiment's
//   intent) we renormalize each P_d to its original ||Omega_d|| instead of 1: at init P==Omega
//   EXACTLY (bands identical to armB), plasticity only rotates the direction, kernel bandwidth
//   preserved. PARITY/SHUF use the same scheme -> all comparisons clean.
//
// CONTROLS (mean-lr IDENTICAL; only the modulator alignment differs):
//   PARITY  : m_t == 1 (unsupervised tilt, same mean lr). MODOJA-K ~= PARITY => error added nothing.
//   SHUF-MOD: m_t permuted in time (same surprise multiset, wrong positions). Must not beat armB >0.005.
//   armB    : P = Omega frozen, no plasticity (reproduces 2.2023/2.1839/2.1980).
//
// PROBE: frozenD1(256 anchor)/armB(512)/MODOJA-K(512)/PARITY(512)/SHUF-MOD(512). All 512 but anchor.
//
// P_DIVERSITY (smoke AND run, MANDATORY): P must not collapse toward a few PCs (128 rows in 64-D =
// over-complete; we want tilt-from-random, NOT PCA). Report eff_rank (participation ratio of the
// row Gram), mean/max |cos| between rows, and drift = mean cos(P_d,Omega_d) for MODOJA-K and PARITY.
//
// MEMORY: the SEE walk (base 256 + l0n) is the expensive part; the 4 cos-lift band sets are cheap
// and depend ONLY on l0n. So we extract base + cache l0n ONCE (~5 GB), then recompute each
// projection's bands into ONE reused band buffer (~4 GB) -> peak ~9 GB instead of storing all four
// band sets at once (~20 GB). No probe uses row-permutation gather (the leak guard is SHUF-MOD, a
// projection), so base/band are simply concatenated per row.
//
// PRE-REGISTERED (in the ps1): anchor==BASE; MODOJA-K beats armB >=0.015 on all 3; MODOJA-K beats
// PARITY >=0.008 on all 3; SHUF-MOD does not beat armB by >0.005; P_DIVERSITY healthy.
// PASS -> MODOJA-K goes to 48.D.A (DAgger closed-loop, frozen 47 harness). TF != generative.
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase38-42/phase48_d.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase48_d.exe -lm -I .
// Run:
//   bin/phase48_d.exe <data> <D1_w> <outprefix> [--len N --epochs E --hidden H --eta0 E0 --pdiag]

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
#define DA        128                    // kernel dims per timescale (= armB D_EXP)
#define KB        (N_TS*DA)              // 256 bands per projection
#define PROBE_DIM (D1_TOT+KB)            // 512 = base + one band set
#define GAMMA_A   0.25f                  // == armB GAMMA (reproduces the baseline exactly)
#define ETA0_DEF  1e-7f                  // Oja base lr: validated edge-of-collapse at 1M steps via --pdiag
                                         // (eff_rank ~33 of 64, drift_cos ~0.974: P tilts but the kernel stays
                                         // diverse). 1e-6 collapses to rank ~1; 1e-8 barely moves. --eta0 to override.
#define MOD_LO    0.2f
#define MOD_HI    5.0f
#define SEED_A    0x48B2EC0DEULL         // == armB PROJ_SEED (lift seed, shared by all 4 projections at init)
#define SEED_PERM 0x48D5A0317ULL         // SHUF-MOD time permutation of the modulator
#define MAGIC_MK  0x5345454Au
#define N_VAL     3

static const float TS_ALPHA[N_TS] = { 0.90f, 0.99f };

static float Pmat[L2_DIM][BASE_DIM];
static float OmA[DA][L0_DIM], BvA[DA];                 // armB (frozen Omega) + shared phases
static float Pmk[DA][L0_DIM], Ppar[DA][L0_DIM], Pshf[DA][L0_DIM]; // learned tilts (init = Omega)
static float norm0[DA];                                // original ||Omega_d|| (renorm target)
static float ETA0 = ETA0_DEF;
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
static void gen_lift(uint64_t seed){ uint64_t s=seed?seed:0xABCDEF12345ULL;
    for(int d=0;d<DA;d++){
        for(int k=0;k<L0_DIM;k++){ double u1=xs_u01(&s); if(u1<1e-12) u1=1e-12; double u2=xs_u01(&s);
            OmA[d][k]=(float)(sqrt(-2.0*log(u1))*cos(6.283185307179586*u2)); }
        BvA[d]=(float)(6.283185307179586*xs_u01(&s)); }
    for(int d=0;d<DA;d++){ float nn=0; for(int k=0;k<L0_DIM;k++) nn+=OmA[d][k]*OmA[d][k]; norm0[d]=sqrtf(nn); }
    memcpy(Pmk,OmA,sizeof OmA); memcpy(Ppar,OmA,sizeof OmA); memcpy(Pshf,OmA,sizeof OmA);
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
// gather one probe row = [base 256 | band 256] (no permutation: 48.D's leak guard is SHUF-MOD).
static inline void gather2(const float* Xbase,const float* Xband,long i,float* out){
    memcpy(out,&Xbase[(size_t)i*D1_TOT],D1_TOT*4);
    memcpy(out+D1_TOT,&Xband[(size_t)i*KB],KB*4);
}
static double mlp_eval(MLP* m,const float* Xbase,const float* Xband,
                       const uint8_t* tgt,const uint8_t* c1a,const uint8_t* c2a,long n,double* o_rms,double* o_ent,double* o_maxp){
    float* hid=malloc(((m->h>0)?m->h:1)*4); float xb[PROBE_DIM]; float lg[CLASSES];
    double tot=0,arms=0,aent=0,amax=0;
    for(long i=0;i<n;i++){
        gather2(Xbase,Xband,i,xb);
        mlp_fwd(m,xb,hid,lg);
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
static void mlp_train(MLP* m,const float* Xbase,const float* Xband,
                      const uint8_t* tgt,const uint8_t* c1a,const uint8_t* c2a,long n,int epochs,float lr){
    int H=m->h,in=m->in; size_t s1=(size_t)H*in, s2=(size_t)CLASSES*H;
    float* gW1=malloc(s1*4); float* gb1=malloc(H*4); float* gW2=malloc(s2*4); float* gb2=malloc(CLASSES*4);
    float* hid=malloc(H*4); float* dh=malloc(H*4); float xb[PROBE_DIM]; float lg[CLASSES],eo[CLASSES]; int bs=512;
    for(int ep=0;ep<epochs;ep++){
        memset(gW1,0,s1*4); memset(gb1,0,H*4); memset(gW2,0,s2*4); memset(gb2,0,CLASSES*4); long inb=0;
        for(long i=0;i<n;i++){
            gather2(Xbase,Xband,i,xb); const float* x=xb;
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
    free(gW1);free(gb1);free(gW2);free(gb2);free(hid);free(dh);
}
// save the MODOJA-K probe (magic 0x5345454A) so 48.D.A can rebuild the kernel bands closed-loop.
// Header carries the LEARNED Pmk (DA x L0_DIM) + shared phases BvA + lift hyperparams.
static void mlp_save_mk(MLP* m,const char* path){
    FILE* f=fopen(path,"wb"); if(!f){ fprintf(stderr,"save %s failed\n",path); return; }
    uint32_t magic=MAGIC_MK,H=(uint32_t)m->h,dim=(uint32_t)m->in,base=D1_TOT,dexp=DA,nts=N_TS;
    float gamma=GAMMA_A,eta0=ETA0; uint64_t pseed=SEED_A;
    fwrite(&magic,4,1,f); fwrite(&H,4,1,f); fwrite(&dim,4,1,f); fwrite(&base,4,1,f);
    fwrite(&dexp,4,1,f); fwrite(&nts,4,1,f); fwrite(&gamma,4,1,f); fwrite(&eta0,4,1,f);
    for(int t=0;t<N_TS;t++){ float a=TS_ALPHA[t]; fwrite(&a,4,1,f); }
    fwrite(&pseed,8,1,f);
    fwrite(Pmk,4,(size_t)DA*L0_DIM,f);      // the learned tilt
    fwrite(BvA,4,DA,f);                      // shared phases
    fwrite(m->W1,4,(size_t)m->h*m->in,f); fwrite(m->b1,4,m->h,f); fwrite(m->W2,4,(size_t)CLASSES*m->h,f);
    fwrite(m->b2,4,CLASSES,f); fclose(f);
}

// One Oja step over all DA rows of projection P, renormalizing each row to its original ||Omega_d||.
static inline void oja_step(float P[][L0_DIM], const float* x, float eta){
    for(int d=0;d<DA;d++){
        float* pj=P[d]; float a=dot_avx(pj,x,L0_DIM);
        for(int k=0;k<L0_DIM;k++) pj[k]+=eta*a*(x[k]-a*pj[k]);
        float nn=0; for(int k=0;k<L0_DIM;k++) nn+=pj[k]*pj[k]; nn=sqrtf(nn);
        if(nn>1e-12f){ float s=norm0[d]/nn; for(int k=0;k<L0_DIM;k++) pj[k]*=s; }
    }
}

// Plasticity pre-pass: learn Pmk / Ppar / Pshf on the TRAIN window only. Modulator m built from
// frozen trigram surprise (exact mean 1 over the window). SHUF-MOD uses a time permutation of m.
static void plasticity_pass(SiliconEntropyState* see, long start, long N){
    float* surp=malloc((size_t)N*4);              // trigram surprise (bytes only)
    double msum=0;
    for(long i=0;i<N;i++){ long g=start+i; uint8_t c2=g_data[g],c1=g_data[g+1],t=g_data[g+2];
        float s=(float)trigram_bpb(c1,c2,t); surp[i]=s; msum+=s; }
    float mean=(float)(msum/N);
    float* m=malloc((size_t)N*4); double csum=0;
    for(long i=0;i<N;i++){ float v=surp[i]/mean; if(v<MOD_LO)v=MOD_LO; if(v>MOD_HI)v=MOD_HI; m[i]=v; csum+=v; }
    float rescale=(float)(N/(csum>0?csum:1));      // exact mean 1 after the clamp
    for(long i=0;i<N;i++) m[i]*=rescale;
    long* pm=malloc((size_t)N*sizeof(long)); for(long i=0;i<N;i++) pm[i]=i;
    uint64_t r=SEED_PERM; for(long i=N-1;i>0;i--){ r^=r<<13;r^=r>>7;r^=r<<17; long j=(long)(r%(uint64_t)(i+1)); long tmp=pm[i];pm[i]=pm[j];pm[j]=tmp; }
    free(surp);
    float feat192[BASE_DIM], l0n[L0_DIM];
    see_reset(see); for(long i=0;i<=start+1;i++) see_observe(see,g_data[i]);
    for(long i=0;i<N;i++){
        long g=start+i; uint8_t t=g_data[g+2];
        see_extract(see,feat192);
        for(int k=0;k<L0_DIM;k++){ float x=(feat192[k]-md1[k])/(sd1[k]+1e-8f); if(x>2.f)x=2.f; if(x<-2.f)x=-2.f; l0n[k]=x; }
        oja_step(Pmk,  l0n, ETA0*m[i]);
        oja_step(Ppar, l0n, ETA0*1.0f);
        oja_step(Pshf, l0n, ETA0*m[pm[i]]);
        see_observe(see,t);
        if((i&0xFFFFF)==0) fprintf(stderr,"    plasticity %ld/%ld\r",i,N);
    }
    fprintf(stderr,"    plasticity %ld/%ld done\n",N,N);
    free(m); free(pm);
}

// P_DIVERSITY: eff_rank (participation ratio of the row Gram), mean/max |cos| between rows, and
// drift = mean cos(P_d, Omega_d). Collapse => eff_rank->1, mean|cos|->1, drift small/erratic.
static void diversity(const char* tag, float P[][L0_DIM]){
    float un[DA][L0_DIM];
    for(int d=0;d<DA;d++){ float nn=0; for(int k=0;k<L0_DIM;k++) nn+=P[d][k]*P[d][k]; nn=sqrtf(nn)+1e-12f;
        for(int k=0;k<L0_DIM;k++) un[d][k]=P[d][k]/nn; }
    double sumsq=0, msum=0, mx=0; long pairs=0;
    for(int a=0;a<DA;a++) for(int b=0;b<DA;b++){ double c=0; for(int k=0;k<L0_DIM;k++) c+=un[a][k]*un[b][k];
        sumsq+=c*c; if(a<b){ double ac=fabs(c); msum+=ac; if(ac>mx)mx=ac; pairs++; } }
    double effr=(double)DA*DA/sumsq;
    double drift=0; for(int d=0;d<DA;d++){ double dot=0,nP=0,nO=0; for(int k=0;k<L0_DIM;k++){ dot+=P[d][k]*OmA[d][k]; nP+=P[d][k]*P[d][k]; nO+=OmA[d][k]*OmA[d][k]; }
        drift+=dot/(sqrt(nP)*sqrt(nO)+1e-12); }  // cos(P_d, Omega_d): 1 = no tilt
    printf("P_DIVERSITY tag=%s eff_rank=%.2f (max=%d) mean_abscos=%.4f max_abscos=%.4f drift_cos=%.4f\n",
           tag, effr, L0_DIM, msum/(pairs>0?pairs:1), mx, drift/DA);
}

typedef struct { float* Xbase; float* l0nc; uint8_t *tgt,*c1,*c2; double tri,d1; long start; } Win;

// Teacher-forced base extraction: fills Xbase[D1_TOT] + caches l0n[L0_DIM] per row. The band sets
// are recomputed later from the cached l0n (compute_bands). Causal/zero-leak exactly as armB/L2_D1.
static void extract_base(SiliconEntropyState* see, long start, long N,
                         float* Xbase, float* l0nc, uint8_t* tgt, uint8_t* c1a, uint8_t* c2a, double* oTRI, double* oD1){
    float L2d1[L2_DIM]={0}, pb_d1[BASE_DIM]={0};
    float feat192[BASE_DIM], fa[BASE_DIM], rawd1[D1_TOT], nf[D1_TOT];
    float scale=1.0f/sqrtf((float)BASE_DIM); double btr=0,bd1=0;
    see_reset(see); for(long i=0;i<=start+1;i++) see_observe(see,g_data[i]);
    for(long i=0;i<N;i++){
        long g=start+i; uint8_t c2=g_data[g],c1=g_data[g+1],t=g_data[g+2];
        see_extract(see,feat192);
        memcpy(rawd1,feat192,BASE_DIM*4); memcpy(rawd1+BASE_DIM,L2d1,L2_DIM*4);
        btr+=trigram_bpb(c1,c2,t);
        norm_feats(rawd1,nf); bd1+=logits_bpb(&Wd1[0][0],Bd1,nf,D1_TOT,c1,c2,t);
        memcpy(&Xbase[(size_t)i*D1_TOT],nf,D1_TOT*4);
        float* l0n=&l0nc[(size_t)i*L0_DIM];
        for(int k=0;k<L0_DIM;k++){ float x=(feat192[k]-md1[k])/(sd1[k]+1e-8f); if(x>2.f)x=2.f; if(x<-2.f)x=-2.f; l0n[k]=x; }
        tgt[i]=t; c1a[i]=c1; c2a[i]=c2;
        see_observe(see,t); see_extract(see,fa);
        if(ent_gate(c1,c2)){ float src5[BASE_DIM];
            for(int k=0;k<BASE_DIM;k++) src5[k]=fa[k]-0.5f*pb_d1[k]; memcpy(pb_d1,fa,BASE_DIM*4);
            for(int j=0;j<L2_DIM;j++){ float p5=0; const float* pj=Pmat[j]; for(int k=0;k<BASE_DIM;k++) p5+=pj[k]*src5[k];
                L2d1[j]=g_alpha*L2d1[j]+(1.0f-g_alpha)*p5*scale; } }
    }
    *oTRI=btr/N; *oD1=bd1/N;
}

// Recompute one projection's KB cos-bands from the cached l0n (cheap; no SEE). Causal: row i holds
// the EMA over <i (cold start 0, update after writing the row) - identical to the armB lift.
static void compute_bands(const float* l0nc, long N, float P[][L0_DIM], float* Xband){
    float e[N_TS][DA]; memset(e,0,sizeof e);
    for(long i=0;i<N;i++){
        const float* l0n=&l0nc[(size_t)i*L0_DIM];
        float* row=&Xband[(size_t)i*KB];
        for(int ts=0;ts<N_TS;ts++) memcpy(row+ts*DA,e[ts],DA*4);
        for(int d=0;d<DA;d++){ float c=cosf(BvA[d]+GAMMA_A*dot_avx(P[d],l0n,L0_DIM));
            for(int ts=0;ts<N_TS;ts++){ float a=TS_ALPHA[ts]; e[ts][d]=a*e[ts][d]+(1.0f-a)*c; } }
    }
}

static void run_probe(const char* name,int H,Win* tr,Win* va,float* Xb_tr,float** Xb_va,
                      long N,int EP,const char* savepath){
    fprintf(stderr,"  probe %s (H=%d dim=%d)...\n",name,H,PROBE_DIM);
    MLP m; mlp_init(&m,PROBE_DIM,H);
    mlp_train(&m,tr->Xbase,Xb_tr,tr->tgt,tr->c1,tr->c2,N,EP,0.0005f);
    double v[N_VAL],rms=0,ent=0,mxp=0;
    for(int w=0;w<N_VAL;w++)
        v[w]=mlp_eval(&m,va[w].Xbase,Xb_va[w],va[w].tgt,va[w].c1,va[w].c2,N,
                      w==0?&rms:NULL,w==0?&ent:NULL,w==0?&mxp:NULL);
    printf("PROBE name=%s h=%d dim=%d val1=%.4f val2=%.4f val3=%.4f rms1=%.3f ent1=%.3f maxp1=%.3f\n",
           name,H,PROBE_DIM,v[0],v[1],v[2],rms,ent,mxp);
    if(savepath){ mlp_save_mk(&m,savepath); printf("SAVED name=%s path=%s\n",name,savepath); }
    mlp_free(&m);
}

int main(int argc, char** argv){
    if(argc<4){ fprintf(stderr,"Usage: %s <data> <D1_w> <outprefix> [--len N --epochs E --hidden H --eta0 E0 --pdiag]\n",argv[0]); return 1; }
    setvbuf(stderr,NULL,_IONBF,0); setvbuf(stdout,NULL,_IONBF,0);
    long N=1000000; int EP=6, H=32, pdiag=0;
    for(int i=4;i<argc;i++){ if(!strcmp(argv[i],"--len")&&i+1<argc)N=atol(argv[++i]);
        else if(!strcmp(argv[i],"--epochs")&&i+1<argc)EP=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--hidden")&&i+1<argc)H=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--eta0")&&i+1<argc)ETA0=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"--pdiag"))pdiag=1; }
    FILE* fd=fopen(argv[1],"rb"); if(!fd){fprintf(stderr,"data\n");return 1;} fseek(fd,0,SEEK_END); g_fsz=ftell(fd); fseek(fd,0,SEEK_SET);
    g_data=malloc(g_fsz); fread(g_data,1,g_fsz,fd); fclose(fd);

    SiliconEntropyState see;
    if(!load_d1(argv[2],&see)) return 1;
    gen_lift(SEED_A);
    ent_table=malloc(CLASSES*CLASSES*4);
    for(int i=0;i<CLASSES;i++) for(int j=0;j<CLASSES;j++){ float m=-1e9f; for(int k=0;k<CLASSES;k++) if(trigram[i][j][k]>m)m=trigram[i][j][k];
        double se=0; for(int k=0;k<CLASSES;k++) se+=exp((double)(trigram[i][j][k]-m));
        double Hh=0; for(int k=0;k<CLASSES;k++){ double p=exp((double)(trigram[i][j][k]-m))/se; if(p>1e-12)Hh-=p*log(p); }
        ent_table[i][j]=(float)Hh; }

    long tr_start=g_fsz/5;
    long va_start[N_VAL]={ g_fsz/2, (long)(0.65*g_fsz), (long)(0.80*g_fsz) };
    for(int w=0;w<N_VAL;w++) if(va_start[w]+N+3>g_fsz){ fprintf(stderr,"val window %d out of file\n",w+1); return 1; }
    double gb=(double)N*(N_VAL+1)*(D1_TOT+L0_DIM)*4/1e9 + (double)N*(N_VAL+1)*KB*4/1e9;
    fprintf(stderr,"48.D: train @%ld val @%ld/%ld/%ld (N=%ld) H=%d ep=%d eta0=%.2e probe_dim=%d (~%.1f GB peak)\n",
            tr_start,va_start[0],va_start[1],va_start[2],N,H,EP,ETA0,PROBE_DIM,gb);

    // --- plasticity pre-pass: learn Pmk/Ppar/Pshf on the train window, then FREEZE ---
    fprintf(stderr,"plasticity pre-pass (train window, learn P then freeze)...\n");
    plasticity_pass(&see,tr_start,N);

    // --- pdiag: plasticity-only diversity at the real step count (cheap; no feats, no MLP) ---
    if(pdiag){
        printf("\n==== 48.D PDIAG (plasticity-only diversity, N=%ld, eta0=%.2e) ====\n",N,ETA0);
        diversity("MODOJA-K",Pmk); diversity("PARITY",Ppar); diversity("SHUF-MOD",Pshf);
        return 0;
    }

    // --- base extraction (one SEE walk/window): Xbase[256] + l0n cache[64] ---
    Win tr; Win va[N_VAL];
    tr.Xbase=malloc((size_t)N*D1_TOT*4); tr.l0nc=malloc((size_t)N*L0_DIM*4);
    tr.tgt=malloc(N); tr.c1=malloc(N); tr.c2=malloc(N); tr.start=tr_start;
    for(int w=0;w<N_VAL;w++){ va[w].Xbase=malloc((size_t)N*D1_TOT*4); va[w].l0nc=malloc((size_t)N*L0_DIM*4);
        va[w].tgt=malloc(N); va[w].c1=malloc(N); va[w].c2=malloc(N); va[w].start=va_start[w]; }
    if(!tr.Xbase||!tr.l0nc||!va[0].Xbase||!va[1].Xbase||!va[2].Xbase){ fprintf(stderr,"OOM (base)\n"); return 1; }
    fprintf(stderr,"extracting train base...\n");
    extract_base(&see,tr_start,N,tr.Xbase,tr.l0nc,tr.tgt,tr.c1,tr.c2,&tr.tri,&tr.d1);
    for(int w=0;w<N_VAL;w++){ fprintf(stderr,"extracting val base %d...\n",w+1);
        extract_base(&see,va_start[w],N,va[w].Xbase,va[w].l0nc,va[w].tgt,va[w].c1,va[w].c2,&va[w].tri,&va[w].d1); }

    // reusable band buffers (one set, recomputed per projection)
    float* Xb_tr=malloc((size_t)N*KB*4); float* Xb_va[N_VAL];
    for(int w=0;w<N_VAL;w++) Xb_va[w]=malloc((size_t)N*KB*4);
    if(!Xb_tr||!Xb_va[0]||!Xb_va[1]||!Xb_va[2]){ fprintf(stderr,"OOM (bands)\n"); return 1; }

    printf("\n==== 48.D MODOJA-K error-modulated kernel tilt (base 256, N=%ld, ep=%d, H=%d, eta0=%.2e) ====\n",N,EP,H,ETA0);
    diversity("MODOJA-K",Pmk);
    diversity("PARITY",  Ppar);
    diversity("SHUF-MOD",Pshf);
    printf("BASE win=train tri=%.4f d1=%.4f\n",tr.tri,tr.d1);
    for(int w=0;w<N_VAL;w++) printf("BASE win=val%d tri=%.4f d1=%.4f\n",w+1,va[w].tri,va[w].d1);

    // frozenD1 anchor (base 256 readout) - must == BASE d1
    { double v[N_VAL]; float lg[CLASSES];
      for(int w=0;w<N_VAL;w++){ double tot=0;
        for(long i=0;i<N;i++){ const float* x=&va[w].Xbase[(size_t)i*D1_TOT];
            for(int c=0;c<CLASSES;c++) lg[c]=Bd1[c]+dot_avx(Wd1[c],x,D1_TOT);
            const float* tri=&trigram[va[w].c2[i]][va[w].c1[i]][0]; for(int c=0;c<CLASSES;c++) lg[c]+=tri[c];
            float mx=-1e30f; for(int c=0;c<CLASSES;c++) if(lg[c]>mx)mx=lg[c];
            double Z=0; for(int c=0;c<CLASSES;c++) Z+=exp((double)(lg[c]-mx));
            double pp=exp((double)(lg[va[w].tgt[i]]-mx))/Z; tot+=-log2(pp>1e-30?pp:1e-30); }
        v[w]=tot/N; }
      printf("PROBE name=frozenD1 h=0 dim=%d val1=%.4f val2=%.4f val3=%.4f rms1=0 ent1=0 maxp1=0\n",D1_TOT,v[0],v[1],v[2]); }

    char sp[512]; snprintf(sp,sizeof sp,"%s_MK_h%d.bin",argv[3],H);
    struct { const char* name; float (*P)[L0_DIM]; const char* save; } probes[4] = {
        {"armB",    OmA,  NULL},
        {"MODOJA-K",Pmk,  sp},
        {"PARITY",  Ppar, NULL},
        {"SHUF-MOD",Pshf, NULL},
    };
    for(int p=0;p<4;p++){
        fprintf(stderr,"computing bands for %s...\n",probes[p].name);
        compute_bands(tr.l0nc,N,probes[p].P,Xb_tr);
        for(int w=0;w<N_VAL;w++) compute_bands(va[w].l0nc,N,probes[p].P,Xb_va[w]);
        run_probe(probes[p].name,H,&tr,va,Xb_tr,Xb_va,N,EP,probes[p].save);
    }

    printf("\nPre-registered criterion (decided BEFORE the run): MODOJA-K earns 48.D.A iff frozenD1==BASE d1,\n");
    printf("MODOJA-K beats armB by >= 0.015 on ALL of val1/2/3, MODOJA-K beats PARITY by >= 0.008 on ALL,\n");
    printf("SHUF-MOD does NOT beat armB by > 0.005, and P_DIVERSITY is healthy (no collapse). FLAT\n");
    printf("(MODOJA-K ~= PARITY ~= armB) => error-on-substrate-via-kernel-tilt is not the lever (read by the\n");
    printf("Architect -> FORCE/RLS on the readout). TF-only != generative (44-47): a PASS is only validated\n");
    printf("by the closed-loop gate v2 + replicas + human reading at 48.D.A, exactly as armB.\n");
    return 0;
}
