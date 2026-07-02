// Phase 47.G - Last Mile H32 (execution of pre-registered branch 4 of 47.F)
//
// 47.F verdict (user): temperature-coverage hypothesis FALSIFIED cleanly (r3Tmix vs r3:
// topBi55 13=13 with 50% of the rollout pool at T0.55 -> not a dose question, archived).
// Round axis INVERTED at H64 (topBi65 9->8->11 over r2->r3->r5: r3 was the structural
// optimum; BPB keeps falling -> rounds buy compression but past r3 buy back the
// attractor). H64 wall at T0.55 (13/13/15) insensitive to rounds AND temperature ->
// capacity-sharpness story confirmed hard, H64 route closed for now.
// Best news: D16r5_h32 = 0.0072 BPB and 1 altLp unit away from the FIRST full PASS in
// project history (topBi 5/5 at both temps, BPB 2.2615 vs bar 2.2543, rounds recover
// ~0.0065 BPB/round without breaking structure). WATCH: quiet-down signals grow with
// rounds (selfBPB55 0.84->0.81, floor 0.8; ent_p50 0.378->0.351) -> the H32 risk is
// slow simplification, measurable per-checkpoint.
//
// 47.G pieces (single trainer, H32 cheap):
//   step 0  --eval <name> <path>: probe EXISTING 47.F checkpoints (r3/r4 of D16r5_h32)
//           on the same val windows -> exact BPB for the free retro-gate.
//   step 1  P branch (PLAIN): from scratch to r9, mix 20% throughout, per-round
//           checkpoint + eval. Rounds 1-5 are BIT-IDENTICAL to 47.F D16r5_h32 (same
//           rng formula 0xD47F0000^round^K, same init/pipeline) -> harness verifies
//           the prefix property via MD5 of the r5 checkpoint vs phase47f's. NOTE: a
//           true "resume" is impossible from the saved checkpoints (weights only, NO
//           Adam moments) - hence from-scratch with guaranteed prefix.
//   step 2  A branch (ANNEAL): in-memory branch - deep copy of the FULL MLP state
//           (weights + Adam moments + t) at end of round 5; restore after P finishes;
//           rounds 6-9 at mix 10%. Round-6 pool is identical across branches (same
//           model state + same seed) -> the plain-vs-anneal comparison isolates dose.
//   step 3  H48 probe (OPTIONAL, --skip-h48 to cut): D16r5_h48 from scratch, 5 rounds,
//           per-round checkpoint. Not dose-tuning: locates the basin-formation
//           threshold between 32 and 64 (complementary failures: H32 fails BPB/altLp,
//           H64 fails topBi).
// Invariants: K=16, lamC=0.02 (stop-grad clean), target ALWAYS true byte, recovery 16B,
// burst T0.65 only (Tmix dead), no label smoothing. Saves 0x53454545 -> generator
// unchanged. Forbidden: K!=16, mix 30%, 0.55-share sweeps, r7 at H64, L2/L3.
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase38-42/phase47g_lastmile.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase47g_lastmile.exe -lm -I .
// Run:
//   bin/phase47g_lastmile.exe <data> <D1_w> <outprefix> [--len N] [--skip-h48]
//                             [--eval <name> <mlp.bin>]...

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
#define S_DIM     (BASE_DIM + L2_DIM)   // 256
#define D1_TOT    (BASE_DIM + L2_DIM)
#define N_VAL     3
#define T_HI      0.65f
#define RECOV     16
#define PERIOD    256
#define EP_PRE    2
#define EP_MIX    2
#define R_PLAIN   9
#define R_BRANCH  5    // anneal branches after this round
#define R_H48     5
#define MAX_EVALS 8

enum { G_ENTROPY=4 };
enum { X_CLEAN=0, X_CORRUPT=1 };

static float Pmat[L2_DIM][BASE_DIM];
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

// ---- MLP (same family as 47.B-F; H parametric) ----
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
// deep copy of the FULL optimizer state (weights + Adam moments + step counter):
// this is what makes the in-memory anneal branch a TRUE branch (checkpoints on disk
// don't carry Adam state, so they can't).
static void mlp_copy(MLP* dst,const MLP* src){
    mlp_init(dst,src->in,src->h);
    size_t s1=(size_t)src->h*src->in, s2=(size_t)CLASSES*((src->h>0)?src->h:src->in);
    memcpy(dst->W1,src->W1,s1*4); memcpy(dst->b1,src->b1,src->h*4);
    memcpy(dst->W2,src->W2,s2*4); memcpy(dst->b2,src->b2,CLASSES*4);
    memcpy(dst->mW1,src->mW1,s1*4); memcpy(dst->vW1,src->vW1,s1*4);
    memcpy(dst->mb1,src->mb1,src->h*4); memcpy(dst->vb1,src->vb1,src->h*4);
    memcpy(dst->mW2,src->mW2,s2*4); memcpy(dst->vW2,src->vW2,s2*4);
    memcpy(dst->mb2,src->mb2,CLASSES*4); memcpy(dst->vb2,src->vb2,CLASSES*4);
    dst->t=src->t;
}
static int mlp_load(MLP* m,const char* path){
    FILE* f=fopen(path,"rb"); if(!f){ fprintf(stderr,"open %s\n",path); return 0; }
    uint32_t magic=0,H=0,o=0,d=0;
    fread(&magic,4,1,f); fread(&H,4,1,f); fread(&o,4,1,f); fread(&d,4,1,f);
    if(magic!=0x53454545){ fprintf(stderr,"expected MLP 0x53454545, got 0x%08x in %s\n",magic,path); fclose(f); return 0; }
    if(d!=S_DIM){ fprintf(stderr,"dim %u != %d in %s\n",d,S_DIM,path); fclose(f); return 0; }
    mlp_init(m,(int)d,(int)H);
    fread(m->W1,4,(size_t)H*d,f); fread(m->b1,4,H,f); fread(m->W2,4,(size_t)CLASSES*H,f);
    fread(m->b2,4,CLASSES,f); fclose(f);
    return 1;
}
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
        mlp_fwd(m,&X[(size_t)i*S_DIM],hid,lg);
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

// ---- gradient helper: accumulate one sample's CE (+optional consistency) into grads ----
typedef struct { float *gW1,*gb1,*gW2,*gb2; } Grads;
static void accum_sample(MLP* m,const float* x,uint8_t tgt,uint8_t c1,uint8_t c2,
                         const float* xpair,float lamC,float invnorm,
                         Grads* G,float* hid,float* hidP,float* dh){
    int H=m->h,in=m->in;
    float lg[CLASSES],eo[CLASSES];
    mlp_fwd(m,x,hid,lg);
    float lgp[CLASSES]; memcpy(lgp,lg,sizeof(lgp));   // pre-trigram (consistency)
    const float* tri=&trigram[c2][c1][0]; for(int c=0;c<CLASSES;c++) lg[c]+=tri[c];
    float mx=-1e30f; for(int c=0;c<CLASSES;c++) if(lg[c]>mx)mx=lg[c];
    float Z=0; for(int c=0;c<CLASSES;c++){ eo[c]=expf(lg[c]-mx); Z+=eo[c]; }
    for(int c=0;c<CLASSES;c++){ float y=(c==tgt)?1.f:0.f; eo[c]=(eo[c]/Z-y)*invnorm; }
    if(xpair && lamC>0){
        // light consistency toward the CLEAN-state logits at the same position (stop-grad
        // on the clean branch: pull the drifted state back, don't drag the clean one away)
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
// mixed training: pattern of `den` slots, first `num` from the rollout pool, rest clean.
// Rollout samples get CE on the TRUE byte + light consistency to the clean state (paired
// by window position from X0). Clean samples are plain CE. npool==0 -> pure clean.
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
                accum_sample(m,&Xr[(size_t)j*S_DIM],tr[j],c1r[j],c2r[j],
                             &X0[(size_t)posr[j]*S_DIM],lamC,invnorm,&G,hid,hidP,dh);
            } else {
                long k=kc%n; kc++;
                accum_sample(m,&X0[(size_t)k*S_DIM],t0[k],c10[k],c20[k],NULL,0,invnorm,&G,hid,hidP,dh);
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
static void mlp_save(MLP* m,const char* path){
    FILE* f=fopen(path,"wb"); if(!f){ fprintf(stderr,"save %s failed\n",path); return; }
    uint32_t magic=0x53454545,H=(uint32_t)m->h,o=0,d=(uint32_t)m->in;
    fwrite(&magic,4,1,f); fwrite(&H,4,1,f); fwrite(&o,4,1,f); fwrite(&d,4,1,f);
    fwrite(m->W1,4,(size_t)m->h*m->in,f); fwrite(m->b1,4,m->h,f); fwrite(m->W2,4,(size_t)CLASSES*m->h,f);
    fwrite(m->b2,4,CLASSES,f); fclose(f);
}

// clean / D1-corrupted extraction (47.C-F parity; corrupt only for valC continuity)
static void extract_window_x(SiliconEntropyState* see, long start, long N, int mode, float p, uint64_t rseed,
                             float* X, uint8_t* tgt, uint8_t* c1a, uint8_t* c2a,
                             double* oTRI, double* oD1, double* o_corr){
    float L2d1[L2_DIM]={0}, pb_d1[BASE_DIM]={0};
    float feat192[BASE_DIM], fa[BASE_DIM], rawd1[D1_TOT], nf[D1_TOT], probsT[CLASSES];
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
        int dirty=0;
        if(mode==X_CORRUPT){ rng^=rng<<13;rng^=rng>>7;rng^=rng<<17; if(((rng>>11)*(1.0/(1ULL<<53)))<p) dirty=1; }
        bd1+=d1_logits(nf,cur_c1,cur_c2,t,dirty?probsT:NULL,T_HI);
        uint8_t ob=t;
        if(dirty){ ob=sample_p(probsT,&rng); ncorr++; }
        memcpy(&X[(size_t)i*S_DIM],nf,S_DIM*4);
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

// TRUE rollout extraction @ T0.65 (47.F parity, tmix dead): every PERIOD bytes, K bytes
// sampled from the CURRENT decoder; pool = in-burst + RECOV recovery; target = true byte.
static long extract_rollout(SiliconEntropyState* see, MLP* m, long start, long N, int K, uint64_t rseed,
                            float* Xr, uint8_t* tr, uint8_t* c1r, uint8_t* c2r, long* posr, long npool_max,
                            double* o_selfbpb){
    float L2d1[L2_DIM]={0}, pb_d1[BASE_DIM]={0};
    float feat192[BASE_DIM], fa[BASE_DIM], rawd1[D1_TOT], nf[D1_TOT];
    float* hid=malloc(((m->h>0)?m->h:1)*4); float lg[CLASSES],Pp[CLASSES];
    float scale=1.0f/sqrtf((float)BASE_DIM); double selfbits=0; long nself=0, np=0;
    uint64_t rng=rseed?rseed:0x9E3779B97F4A7C15ULL; int burst=0; long since=RECOV+1;
    see_reset(see); for(long i=0;i<=start+1;i++) see_observe(see,g_data[i]);
    uint8_t cur_c2=g_data[start], cur_c1=g_data[start+1];
    for(long i=0;i<N;i++){
        long g=start+i; uint8_t t=g_data[g+2];
        see_extract(see,feat192);
        memcpy(rawd1,feat192,BASE_DIM*4); memcpy(rawd1+BASE_DIM,L2d1,L2_DIM*4);
        norm_feats(rawd1,nf);
        if(i>0 && (i%PERIOD)==0) burst=K;
        int in_burst=(burst>0), in_recov=(!in_burst && since<RECOV);
        if((in_burst||in_recov) && np<npool_max){
            memcpy(&Xr[(size_t)np*S_DIM],nf,S_DIM*4);
            tr[np]=t; c1r[np]=cur_c1; c2r[np]=cur_c2; posr[np]=i; np++;
        }
        uint8_t ob=t;
        if(in_burst){
            mlp_fwd(m,nf,hid,lg);
            const float* tri=&trigram[cur_c2][cur_c1][0];
            float mx=-1e30f; for(int c=0;c<CLASSES;c++){ lg[c]=(lg[c]+tri[c])/T_HI; if(lg[c]>mx)mx=lg[c]; }
            float Z=0; for(int c=0;c<CLASSES;c++){ Pp[c]=expf(lg[c]-mx); Z+=Pp[c]; } for(int c=0;c<CLASSES;c++) Pp[c]/=Z;
            ob=sample_p(Pp,&rng);
            selfbits+=-log2((double)fmaxf(Pp[ob],1e-30f)); nself++;
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
    *o_selfbpb=(nself>0)?selfbits/nself:0.0;
    return np;
}

typedef struct { float* X; uint8_t *tgt,*c1,*c2; double tri,d1,corr; long start; } Win;
static Win alloc_win(long N,long start){
    Win w; w.X=malloc((size_t)N*S_DIM*4); w.tgt=malloc(N); w.c1=malloc(N); w.c2=malloc(N);
    w.start=start; w.tri=0; w.d1=0; w.corr=0;
    if(!w.X||!w.tgt){ fprintf(stderr,"OOM\n"); exit(1); }
    return w;
}

// per-round eval + PROBE + checkpoint (one line per round -> partial runs usable)
static void probe_round(const char* name,int round,MLP* m,Win* va,Win* vaC,long N,double roll,
                        const char* outprefix,const char* fileTag){
    double v[N_VAL],vc,rms=0,ent=0,mxp=0;
    for(int w=0;w<N_VAL;w++) v[w]=mlp_eval(m,va[w].X,va[w].tgt,va[w].c1,va[w].c2,N,
                                           w==0?&rms:NULL,w==0?&ent:NULL,w==0?&mxp:NULL);
    vc=mlp_eval(m,vaC->X,vaC->tgt,vaC->c1,vaC->c2,N,NULL,NULL,NULL);
    char sp[512]; snprintf(sp,sizeof sp,"%s_%s_r%d.bin",outprefix,fileTag,round);
    mlp_save(m,sp);
    printf("PROBE name=%s_r%d h=%d roll=%.3f val1=%.4f val2=%.4f val3=%.4f valC=%.4f rms1=%.3f ent1=%.3f maxp1=%.3f\n",
           name,round,m->h,roll,v[0],v[1],v[2],vc,rms,ent,mxp);
    printf("SAVED name=%s_r%d path=%s\n",name,round,sp);
}

int main(int argc, char** argv){
    if(argc<4){ fprintf(stderr,"Usage: %s <data> <D1_w> <outprefix> [--len N] [--skip-h48] [--eval <name> <mlp.bin>]...\n",argv[0]); return 1; }
    setvbuf(stderr,NULL,_IONBF,0); setvbuf(stdout,NULL,_IONBF,0);
    long N=1000000; int skip_h48=0;
    const char* ev_name[MAX_EVALS]; const char* ev_path[MAX_EVALS]; int n_ev=0;
    for(int i=4;i<argc;i++){
        if(!strcmp(argv[i],"--len")&&i+1<argc)N=atol(argv[++i]);
        else if(!strcmp(argv[i],"--skip-h48"))skip_h48=1;
        else if(!strcmp(argv[i],"--eval")&&i+2<argc&&n_ev<MAX_EVALS){ ev_name[n_ev]=argv[i+1]; ev_path[n_ev]=argv[i+2]; n_ev++; i+=2; }
    }
    FILE* fd=fopen(argv[1],"rb"); if(!fd){fprintf(stderr,"data\n");return 1;} fseek(fd,0,SEEK_END); g_fsz=ftell(fd); fseek(fd,0,SEEK_SET);
    g_data=malloc(g_fsz); fread(g_data,1,g_fsz,fd); fclose(fd);

    SiliconEntropyState see;
    if(!load_d1(argv[2],&see)) return 1;
    ent_table=malloc(CLASSES*CLASSES*4);
    for(int i=0;i<CLASSES;i++) for(int j=0;j<CLASSES;j++){ float m=-1e9f; for(int k=0;k<CLASSES;k++) if(trigram[i][j][k]>m)m=trigram[i][j][k];
        double se=0; for(int k=0;k<CLASSES;k++) se+=exp((double)(trigram[i][j][k]-m));
        double Hh=0; for(int k=0;k<CLASSES;k++){ double p=exp((double)(trigram[i][j][k]-m))/se; if(p>1e-12)Hh-=p*log(p); }
        ent_table[i][j]=(float)Hh; }

    long tr_start=g_fsz/5;
    long va_start[N_VAL]={ g_fsz/2, (long)(0.65*g_fsz), (long)(0.80*g_fsz) };
    for(int w=0;w<N_VAL;w++) if(va_start[w]+N+3>g_fsz){ fprintf(stderr,"val window %d out of file\n",w+1); return 1; }
    fprintf(stderr,"47.G: train @%ld, val @%ld/%ld/%ld (N=%ld) P=r%d (branch A @r%d mix 1/10) H48=%s\n",
            tr_start,va_start[0],va_start[1],va_start[2],N,R_PLAIN,R_BRANCH,skip_h48?"SKIP":"r5");

    Win tr0 = alloc_win(N,tr_start);
    Win va[N_VAL]; for(int w=0;w<N_VAL;w++) va[w]=alloc_win(N,va_start[w]);
    Win vaC = alloc_win(N,va_start[0]);
    fprintf(stderr,"extract tr_clean...\n");
    extract_window_x(&see,tr_start,N,X_CLEAN,0,0, tr0.X,tr0.tgt,tr0.c1,tr0.c2,&tr0.tri,&tr0.d1,NULL);
    for(int w=0;w<N_VAL;w++){ fprintf(stderr,"extract val%d (clean)...\n",w+1);
        extract_window_x(&see,va_start[w],N,X_CLEAN,0,0, va[w].X,va[w].tgt,va[w].c1,va[w].c2,&va[w].tri,&va[w].d1,NULL); }
    fprintf(stderr,"extract valC (D1-corrupt p=0.05, continuity)...\n");
    extract_window_x(&see,va_start[0],N,X_CORRUPT,0.05f,0x7A1C0DE5ULL,vaC.X,vaC.tgt,vaC.c1,vaC.c2,&vaC.tri,&vaC.d1,&vaC.corr);

    printf("\n==== 47.G last mile H32 (N=%ld, P=r%d + anneal branch + %s) ====\n",N,R_PLAIN,skip_h48?"no H48":"H48 probe");
    printf("BASE win=train tri=%.4f d1=%.4f\n",tr0.tri,tr0.d1);
    for(int w=0;w<N_VAL;w++) printf("BASE win=val%d tri=%.4f d1=%.4f\n",w+1,va[w].tri,va[w].d1);
    printf("BASE win=valC tri=%.4f d1=%.4f corr%%=%.2f\n",vaC.tri,vaC.d1,vaC.corr);

    // frozenD1 anchor
    { MLP m; mlp_init(&m,S_DIM,0);
      for(int c=0;c<CLASSES;c++){ memcpy(&m.W2[(size_t)c*S_DIM],&Wd1[c][0],S_DIM*4); m.b2[c]=Bd1[c]; }
      double v[N_VAL],vc,rms=0,ent=0,mxp=0;
      for(int w=0;w<N_VAL;w++) v[w]=mlp_eval(&m,va[w].X,va[w].tgt,va[w].c1,va[w].c2,N,
                                             w==0?&rms:NULL,w==0?&ent:NULL,w==0?&mxp:NULL);
      vc=mlp_eval(&m,vaC.X,vaC.tgt,vaC.c1,vaC.c2,N,NULL,NULL,NULL);
      printf("PROBE name=frozenD1 h=0 val1=%.4f val2=%.4f val3=%.4f valC=%.4f rms1=%.3f ent1=%.3f maxp1=%.3f\n",
             v[0],v[1],v[2],vc,rms,ent,mxp);
      mlp_free(&m); }

    // step 0: probe existing checkpoints (47.F r3/r4 retro-gate gets exact BPB)
    for(int e=0;e<n_ev;e++){
        MLP m; if(!mlp_load(&m,ev_path[e])) return 1;
        double v[N_VAL],vc,rms=0,ent=0,mxp=0;
        for(int w=0;w<N_VAL;w++) v[w]=mlp_eval(&m,va[w].X,va[w].tgt,va[w].c1,va[w].c2,N,
                                               w==0?&rms:NULL,w==0?&ent:NULL,w==0?&mxp:NULL);
        vc=mlp_eval(&m,vaC.X,vaC.tgt,vaC.c1,vaC.c2,N,NULL,NULL,NULL);
        printf("PROBE name=%s h=%d val1=%.4f val2=%.4f val3=%.4f valC=%.4f rms1=%.3f ent1=%.3f maxp1=%.3f\n",
               ev_name[e],m.h,v[0],v[1],v[2],vc,rms,ent,mxp);
        mlp_free(&m);
    }

    long npool_max=N*(16+RECOV)/PERIOD + 64;
    float* Xr=malloc((size_t)npool_max*S_DIM*4);
    uint8_t* trr=malloc(npool_max); uint8_t* c1r=malloc(npool_max); uint8_t* c2r=malloc(npool_max);
    long* posr=malloc(npool_max*sizeof(long));
    if(!Xr||!posr){ fprintf(stderr,"OOM pool\n"); return 1; }

    // ---- step 1: P branch (plain, r1..R_PLAIN, mix 1/5) -------------------------------
    // rng formula IDENTICAL to 47.F (0xD47F0000^round^K) -> rounds 1-5 bit-identical to
    // phase47f_D16r5_h32; the harness verifies via MD5(P_r5) == MD5(47.F r5 checkpoint).
    MLP mp; mlp_init(&mp,S_DIM,32);
    MLP mbranch; int have_branch=0;
    fprintf(stderr,"  P branch (plain H32, %d rounds, mix 1/5)\n",R_PLAIN);
    fprintf(stderr,"   pretrain clean %dep\n",EP_PRE);
    train_mixed(&mp,tr0.X,tr0.tgt,tr0.c1,tr0.c2,N, NULL,NULL,NULL,NULL,NULL,0, 0,1,0,EP_PRE,0.0005f);
    for(int rd=1;rd<=R_PLAIN;rd++){
        double roll=0;
        fprintf(stderr,"   P round %d: extract rollout...\n",rd);
        long npool=extract_rollout(&see,&mp,tr_start,N,16,0xD47F0000ULL^((uint64_t)rd*0x9E37ULL)^16ULL,
                                   Xr,trr,c1r,c2r,posr,npool_max,&roll);
        fprintf(stderr,"   P round %d: pool=%ld roll=%.3f; train mixed %dep (mix 1/5)\n",rd,npool,roll,EP_MIX);
        train_mixed(&mp,tr0.X,tr0.tgt,tr0.c1,tr0.c2,N, Xr,trr,c1r,c2r,posr,npool, 1,5,0.02f,EP_MIX,0.0005f);
        probe_round("P",rd,&mp,va,&vaC,N,roll,argv[3],"P_h32");
        if(rd==R_BRANCH){ mlp_copy(&mbranch,&mp); have_branch=1;
            fprintf(stderr,"   [branch point saved in-memory at r%d: weights+Adam+t]\n",rd); }
    }
    mlp_free(&mp);

    // ---- step 2: A branch (anneal, restore r5 state, r6..R_PLAIN, mix 1/10) -----------
    if(have_branch){
        fprintf(stderr,"  A branch (anneal H32, rounds %d-%d, mix 1/10, true branch from r%d state)\n",
                R_BRANCH+1,R_PLAIN,R_BRANCH);
        for(int rd=R_BRANCH+1;rd<=R_PLAIN;rd++){
            double roll=0;
            fprintf(stderr,"   A round %d: extract rollout...\n",rd);
            long npool=extract_rollout(&see,&mbranch,tr_start,N,16,0xD47F0000ULL^((uint64_t)rd*0x9E37ULL)^16ULL,
                                       Xr,trr,c1r,c2r,posr,npool_max,&roll);
            fprintf(stderr,"   A round %d: pool=%ld roll=%.3f; train mixed %dep (mix 1/10)\n",rd,npool,roll,EP_MIX);
            train_mixed(&mbranch,tr0.X,tr0.tgt,tr0.c1,tr0.c2,N, Xr,trr,c1r,c2r,posr,npool, 1,10,0.02f,EP_MIX,0.0005f);
            probe_round("A",rd,&mbranch,va,&vaC,N,roll,argv[3],"A_h32");
        }
        mlp_free(&mbranch);
    }

    // ---- step 3 (optional): H48 probe (basin-formation threshold between 32 and 64) ---
    if(!skip_h48){
        fprintf(stderr,"  H48 probe (D16r5_h48, %d rounds, mix 1/5)\n",R_H48);
        MLP mh; mlp_init(&mh,S_DIM,48);
        fprintf(stderr,"   pretrain clean %dep\n",EP_PRE);
        train_mixed(&mh,tr0.X,tr0.tgt,tr0.c1,tr0.c2,N, NULL,NULL,NULL,NULL,NULL,0, 0,1,0,EP_PRE,0.0005f);
        for(int rd=1;rd<=R_H48;rd++){
            double roll=0;
            fprintf(stderr,"   H48 round %d: extract rollout...\n",rd);
            long npool=extract_rollout(&see,&mh,tr_start,N,16,0xD47F0000ULL^((uint64_t)rd*0x9E37ULL)^16ULL,
                                       Xr,trr,c1r,c2r,posr,npool_max,&roll);
            fprintf(stderr,"   H48 round %d: pool=%ld roll=%.3f; train mixed %dep (mix 1/5)\n",rd,npool,roll,EP_MIX);
            train_mixed(&mh,tr0.X,tr0.tgt,tr0.c1,tr0.c2,N, Xr,trr,c1r,c2r,posr,npool, 1,5,0.02f,EP_MIX,0.0005f);
            probe_round("H48",rd,&mh,va,&vaC,N,roll,argv[3],"H48");
        }
        mlp_free(&mh);
    }

    printf("\nReading: per-round PROBE = the structure-vs-round map. P_r1..r5 must reproduce\n");
    printf("47.F (prefix property, MD5-verified by harness). The race is BPB <= 2.2543 before\n");
    printf("quiet-down: watch selfBPB55 >= 0.8 and ent trend per checkpoint (pre-registered\n");
    printf("branch 3: if plain quiets down, the anneal decides). H48 locates the basin\n");
    printf("threshold between 32 and 64. Word-gate (two temps) decides; altLp55 stuck at 3\n");
    printf("with everything else green -> report to user, gate untouched (branch 5).\n");
    return 0;
}
