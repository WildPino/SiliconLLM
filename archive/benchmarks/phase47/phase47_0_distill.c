// Phase 47.0 - teacher-transfer feasibility (distillation, NO new inference memory, NO big train)
//
// Axis C, after 45/46 closed the "new volatile memory in inference" line (delta L2, L3 are
// compression-positive but generation-fragile). Question: can the useful info of the delta /
// L3 TEACHERS be distilled into the STABLE D1/SEE family WITHOUT carrying volatile state into
// inference? I.e. is the teacher's gain an instantaneous (static) function of the stable
// features, or genuinely dynamical (needs the recurrent volatile state)?
//
// All of D1 / delta / B3 share the SAME frozen C2.A SEE (oja, eta=0) and the SAME data-derived
// trigram + entropy gate; they differ only in the memory block content and the trained readout.
// So we extract ONCE teacher-forced over a val window, maintaining 3 memory states in parallel:
//   L2_D1    : gated EMA, mix0.5  (D1 stable)
//   L2_delta : gated EMA, mix1.0  (delta teacher)
//   L3_B3    : slow EMA of fa at the PUNCT+LOWMARGIN schedule (B3 teacher, on top of L2_D1)
// Student features = D1-stable [SEE 192 | L2_D1 64] (the features we'd keep at inference);
// teachers = delta readout (BPB ~2.2212) and B3 readout (~2.2509); D1 baseline ~2.2522.
//
// Probes on the student features (train 80% / test 20% of the window):
//   linear-hard  : best LINEAR readout on student feats (= D1's optimum ~2.2522, the bound)
//   mlp-hard     : 1-hidden-layer MLP on student feats (instantaneous NONLINEAR ceiling)
//   mlp-KD-delta : MLP distilling the delta teacher soft distribution
//   mlp-KD-B3    : MLP distilling the B3 teacher soft distribution
// If mlp-hard / KD beat the linear bound toward the teacher -> the gain is a static nonlinear
// function of stable features -> distillable (47.A auxiliary training viable). If they tie the
// bound -> the gain is dynamical, not in the instantaneous stable features -> need a (recurrent)
// substrate, not a readout-only distillation.
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase38-42/phase47_0_distill.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase47_0_distill.exe -lm -I .
// Run:
//   bin/phase47_0_distill.exe <data> <D1_w> <delta_w> <B3_w> [--len N --hidden H --epochs E]

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

enum { G_ENTROPY=4 };

static float Pmat[L2_DIM][BASE_DIM];
static float (*trigram)[CLASSES][CLASSES];
static float (*ent_table)[CLASSES];
static float (*margin_table)[CLASSES];
static uint8_t* g_data; static long g_fsz;
static float g_ent_thr; static int g_ent_high=1;
static uint8_t *g_c1=NULL, *g_c2=NULL;   // per-sample bigram context (for the trigram baseline in probes)

// per-model readout + normalization
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

// read a weight; fill readout W[CLASSES][tot],B,mean[tot],std[tot]; optionally trigram+oja+gate
// (only from the first/D1 call). Returns L3 params via out ptrs when present (B3).
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
    if (is_delta){ float rmc=0; uint32_t wg=0; fread(&rmc,4,1,f); fread(&wg,4,1,f);   // 0x53454543: relmove + wgeom
        uint32_t wm=0,wck=0,wtn=0; float wc=0,ws=0; fread(&wm,4,1,f); fread(&wc,4,1,f); fread(&ws,4,1,f); fread(&wck,4,1,f); fread(&wtn,4,1,f); }  // + wtgate block
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

// normalize raw [SEE|mem...] with a model's stats; mem dims (>=BASE_DIM) get clamp+scale.
static inline void norm_feats(const float* raw, int tot, const float* mean, const float* std, float l2clamp, float l2scale, float* out){
    for(int fi=0;fi<tot;fi++){ float x=(raw[fi]-mean[fi])/(std[fi]+1e-8f);
        float cl=(fi<BASE_DIM)?2.0f:l2clamp; if(cl>0){ if(x>cl)x=cl; if(x<-cl)x=-cl; }
        if(fi>=BASE_DIM) x*=l2scale; out[fi]=x; }
}
static inline double logits_bpb(const float* W, const float* B, const float* nf, int tot, uint8_t c1,uint8_t c2,uint8_t tgt, float* probs_out){
    const float* tri=&trigram[c2][c1][0]; float lg[CLASSES],mx=-1e30f;
    for(int c=0;c<CLASSES;c++){ lg[c]=B[c]+tri[c]+dot_avx(W+(size_t)c*tot,nf,tot); if(lg[c]>mx)mx=lg[c]; }
    double Z=0; for(int c=0;c<CLASSES;c++) Z+=exp((double)(lg[c]-mx));
    if(probs_out){ for(int c=0;c<CLASSES;c++) probs_out[c]=(float)(exp((double)(lg[c]-mx))/Z); }
    double p=exp((double)(lg[tgt]-mx))/Z; return -log2(p>1e-30?p:1e-30);
}

// ---- probe MLP (1 hidden, relu) trained with Adam; hard CE + optional KD to teacher probs ----
typedef struct { int in,h; float *W1,*b1,*W2,*b2,*mW1,*vW1,*mb1,*vb1,*mW2,*vW2,*mb2,*vb2; int t; } MLP;
static void mlp_init(MLP* m,int in,int h){ m->in=in; m->h=h; m->t=0;
    size_t s1=(size_t)h*in, s2=(size_t)CLASSES*h;
    m->W1=calloc(s1,4); m->b1=calloc(h,4); m->W2=calloc(s2,4); m->b2=calloc(CLASSES,4);
    m->mW1=calloc(s1,4); m->vW1=calloc(s1,4); m->mb1=calloc(h,4); m->vb1=calloc(h,4);
    m->mW2=calloc(s2,4); m->vW2=calloc(s2,4); m->mb2=calloc(CLASSES,4); m->vb2=calloc(CLASSES,4);
    uint64_t r=0x1234567; float sc1=sqrtf(2.0f/in);
    for(size_t i=0;i<s1;i++){ r^=r<<13;r^=r>>7;r^=r<<17; m->W1[i]=sc1*(((r>>11)*(1.0/(1ULL<<53)))*2-1); }
    float sc2=sqrtf(2.0f/h); for(size_t i=0;i<s2;i++){ r^=r<<13;r^=r>>7;r^=r<<17; m->W2[i]=sc2*(((r>>11)*(1.0/(1ULL<<53)))*2-1); }
}
static void mlp_fwd(MLP* m,const float* x,float* hid,float* lg){
    for(int j=0;j<m->h;j++){ float a=m->b1[j]; const float* w=&m->W1[(size_t)j*m->in]; a+=dot_avx(w,x,m->in); hid[j]=a>0?a:0; }
    for(int c=0;c<CLASSES;c++){ lg[c]=m->b2[c]+dot_avx(&m->W2[(size_t)c*m->h],hid,m->h); }
}
// returns mean hard BPB on [lo,hi)
static double mlp_eval(MLP* m,float* X,uint8_t* tgt,uint8_t* c1a,uint8_t* c2a,int dimX,long lo,long hi){
    float* hid=malloc((size_t)m->h*4); float lg[CLASSES]; double tot=0; long n=0;
    for(long i=lo;i<hi;i++){ mlp_fwd(m,&X[(size_t)i*dimX],hid,lg);
        const float* tri=&trigram[c2a[i]][c1a[i]][0]; for(int c=0;c<CLASSES;c++) lg[c]+=tri[c];
        float mx=-1e30f; for(int c=0;c<CLASSES;c++) if(lg[c]>mx)mx=lg[c];
        double Z=0; for(int c=0;c<CLASSES;c++) Z+=exp((double)(lg[c]-mx));
        double p=exp((double)(lg[tgt[i]]-mx))/Z; tot+=-log2(p>1e-30?p:1e-30); n++; }
    free(hid); return tot/n;
}
// train Adam; loss = (1-kd)*CE(hard) + kd*KL(teacher||student); tprob may be NULL (kd=0)
static void mlp_train(MLP* m,float* X,uint8_t* tgt,float* tprob,uint8_t* c1a,uint8_t* c2a,int dimX,long lo,long hi,int epochs,float lr,float kd){
    int H=m->h; size_t s1=(size_t)H*dimX, s2=(size_t)CLASSES*H;
    float* gW1=malloc(s1*4); float* gb1=malloc(H*4); float* gW2=malloc(s2*4); float* gb2=malloc(CLASSES*4);
    float* hid=malloc(H*4); float* dh=malloc(H*4); float lg[CLASSES],pr[CLASSES]; int bs=512;
    for(int ep=0;ep<epochs;ep++){
        memset(gW1,0,s1*4); memset(gb1,0,H*4); memset(gW2,0,s2*4); memset(gb2,0,CLASSES*4); long inb=0;
        for(long i=lo;i<hi;i++){
            const float* x=&X[(size_t)i*dimX]; mlp_fwd(m,x,hid,lg);
            const float* tri=&trigram[c2a[i]][c1a[i]][0]; for(int c=0;c<CLASSES;c++) lg[c]+=tri[c];
            float mx=-1e30f; for(int c=0;c<CLASSES;c++) if(lg[c]>mx)mx=lg[c];
            float Z=0; for(int c=0;c<CLASSES;c++){ pr[c]=expf(lg[c]-mx); Z+=pr[c]; } for(int c=0;c<CLASSES;c++) pr[c]/=Z;
            // output grad
            for(int c=0;c<CLASSES;c++){ float y=(c==tgt[i])?1.f:0.f; float tgtc=(kd>0&&tprob)?((1-kd)*y+kd*tprob[(size_t)i*CLASSES+c]):y;
                float e=(pr[c]-tgtc)/bs; gb2[c]+=e; float* gw=&gW2[(size_t)c*H]; for(int j=0;j<H;j++) gw[j]+=e*hid[j]; dh[c<H?c:0]=0; }
            // hidden grad
            for(int j=0;j<H;j++){ float s=0; for(int c=0;c<CLASSES;c++){ float y=(c==tgt[i])?1.f:0.f; float tgtc=(kd>0&&tprob)?((1-kd)*y+kd*tprob[(size_t)i*CLASSES+c]):y; s+=(pr[c]-tgtc)*m->W2[(size_t)c*H+j]; }
                float g=(hid[j]>0)?(s/bs):0.0f; gb1[j]+=g; float* gw=&gW1[(size_t)j*dimX]; for(int k=0;k<dimX;k++) gw[k]+=g*x[k]; }
            inb++;
            if(inb==bs || i==hi-1){ m->t++; float lt=lr*sqrtf(1-powf(.999f,m->t))/(1-powf(.9f,m->t));
                #define ADAM(P,G,MM,VV,NN) for(size_t z=0;z<(size_t)(NN);z++){ MM[z]=.9f*MM[z]+.1f*G[z]; VV[z]=.999f*VV[z]+.001f*G[z]*G[z]; P[z]-=lt*(MM[z]/(sqrtf(VV[z])+1e-8f)+1e-5f*P[z]); }
                ADAM(m->W2,gW2,m->mW2,m->vW2,s2); ADAM(m->b2,gb2,m->mb2,m->vb2,CLASSES);
                ADAM(m->W1,gW1,m->mW1,m->vW1,s1); ADAM(m->b1,gb1,m->mb1,m->vb1,H);
                memset(gW1,0,s1*4); memset(gb1,0,H*4); memset(gW2,0,s2*4); memset(gb2,0,CLASSES*4); inb=0; }
        }
    }
    free(gW1);free(gb1);free(gW2);free(gb2);free(hid);free(dh);
}

// Teacher-forced extraction over [start, start+N): student feats X=[SEE|L2_D1] (D1-normalized),
// targets/contexts, and (optional) delta/B3 teacher soft probs. Memories cold-start at 0.
static void extract_window(SiliconEntropyState* see, long start, long N,
                           float* X, uint8_t* tgt, uint8_t* c1a, uint8_t* c2a,
                           float* dprob, float* bprob, double* oD1, double* oDE, double* oB3){
    float L2d1[L2_DIM]={0}, L2de[L2_DIM]={0}, L3b3[L3_DIM]={0};
    float pb_d1[BASE_DIM]={0}, pb_de[BASE_DIM]={0}; int lmrun=0,lmrefr=0;
    float feat192[BASE_DIM], fa[BASE_DIM], rawd1[D1_TOT], rawde[D1_TOT], rawb3[B3_TOT], nf[B3_TOT];
    float scale=1.0f/sqrtf((float)BASE_DIM); double bd1=0,bde=0,bb3=0;
    see_reset(see); for(long i=0;i<=start+1;i++) see_observe(see,g_data[i]);
    for(long i=0;i<N;i++){
        long g=start+i; uint8_t c2=g_data[g],c1=g_data[g+1],t=g_data[g+2];
        see_extract(see,feat192);
        memcpy(rawd1,feat192,BASE_DIM*4); memcpy(rawd1+BASE_DIM,L2d1,L2_DIM*4);
        memcpy(rawde,feat192,BASE_DIM*4); memcpy(rawde+BASE_DIM,L2de,L2_DIM*4);
        memcpy(rawb3,feat192,BASE_DIM*4); memcpy(rawb3+BASE_DIM,L2d1,L2_DIM*4); memcpy(rawb3+BASE_DIM+L2_DIM,L3b3,L3_DIM*4);
        norm_feats(rawd1,D1_TOT,md1,sd1,g_l2c_d1,g_ls_d1,nf); bd1+=logits_bpb(&Wd1[0][0],Bd1,nf,D1_TOT,c1,c2,t,NULL);
        memcpy(&X[(size_t)i*S_DIM],nf,S_DIM*4);
        norm_feats(rawde,D1_TOT,mde,sde,g_l2c_de,g_ls_de,nf); bde+=logits_bpb(&Wde[0][0],Bde,nf,D1_TOT,c1,c2,t, dprob?&dprob[(size_t)i*CLASSES]:NULL);
        norm_feats(rawb3,B3_TOT,mb3,sb3,g_l2c_b3,g_ls_b3,nf); bb3+=logits_bpb(&Wb3[0][0],Bb3,nf,B3_TOT,c1,c2,t, bprob?&bprob[(size_t)i*CLASSES]:NULL);
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
    *oD1=bd1/N; *oDE=bde/N; *oB3=bb3/N;
}

int main(int argc, char** argv){
    if(argc<5){ fprintf(stderr,"Usage: %s <data> <D1_w> <delta_w> <B3_w> [--len N --hidden H --epochs E]\n",argv[0]); return 1; }
    setvbuf(stderr,NULL,_IONBF,0); setvbuf(stdout,NULL,_IONBF,0);
    long N=1000000; int H=128, EP=6;
    for(int i=5;i<argc;i++){ if(!strcmp(argv[i],"--len")&&i+1<argc)N=atol(argv[++i]);
        else if(!strcmp(argv[i],"--hidden")&&i+1<argc)H=atoi(argv[++i]); else if(!strcmp(argv[i],"--epochs")&&i+1<argc)EP=atoi(argv[++i]); }
    FILE* fd=fopen(argv[1],"rb"); if(!fd){fprintf(stderr,"data\n");return 1;} fseek(fd,0,SEEK_END); g_fsz=ftell(fd); fseek(fd,0,SEEK_SET);
    g_data=malloc(g_fsz); fread(g_data,1,g_fsz,fd); fclose(fd);
    long tr_start=g_fsz/5, va_start=g_fsz/2;   // train-window in train region, val-window in val
    if(tr_start+N+3>va_start) N=va_start-tr_start-3;
    if(va_start+N+3>g_fsz) N=g_fsz-va_start-3;

    SiliconEntropyState see;
    load_weight(argv[2],D1_TOT,&Wd1[0][0],Bd1,md1,sd1,1,&see,&g_l2c_d1,&g_ls_d1,0,0,0,0,0);
    load_weight(argv[3],D1_TOT,&Wde[0][0],Bde,mde,sde,0,0,&g_l2c_de,&g_ls_de,0,0,0,0,0);
    load_weight(argv[4],B3_TOT,&Wb3[0][0],Bb3,mb3,sb3,0,0,&g_l2c_b3,&g_ls_b3,&g_b3_mthr,&g_b3_alpha,&g_b3_M,&g_b3_refr,&g_b3_mode);
    ent_table=malloc(CLASSES*CLASSES*4); margin_table=malloc(CLASSES*CLASSES*4);
    for(int i=0;i<CLASSES;i++) for(int j=0;j<CLASSES;j++){ float m=-1e9f; for(int k=0;k<CLASSES;k++) if(trigram[i][j][k]>m)m=trigram[i][j][k];
        double se=0; for(int k=0;k<CLASSES;k++) se+=exp((double)(trigram[i][j][k]-m));
        double Hh=0,p1=-1,p2=-1; for(int k=0;k<CLASSES;k++){ double p=exp((double)(trigram[i][j][k]-m))/se; if(p>1e-12)Hh-=p*log(p); if(p>p1){p2=p1;p1=p;} else if(p>p2)p2=p; }
        ent_table[i][j]=(float)Hh; margin_table[i][j]=(float)(p1-p2); }
    fprintf(stderr,"47.0: train-win @%ld val-win @%ld (N=%ld each) ent_thr=%.3f B3:mthr=%.4f M=%d mode=%d H=%d ep=%d\n",
            tr_start,va_start,N,g_ent_thr,g_b3_mthr,g_b3_M,g_b3_mode,H,EP);

    // train window (with teacher soft probs for KD) + val window (held-out, generalization test)
    float* Xtr=malloc((size_t)N*S_DIM*4); uint8_t* ttr=malloc(N),*c1tr=malloc(N),*c2tr=malloc(N);
    float* dptr=malloc((size_t)N*CLASSES*4); float* bptr=malloc((size_t)N*CLASSES*4);
    float* Xva=malloc((size_t)N*S_DIM*4); uint8_t* tva=malloc(N),*c1va=malloc(N),*c2va=malloc(N);
    if(!Xtr||!dptr||!bptr||!Xva){ fprintf(stderr,"OOM (~%.1f GB)\n",(double)N*(2*S_DIM+2*CLASSES)*4/1e9); return 1; }
    double trD1,trDE,trB3, vaD1,vaDE,vaB3;
    extract_window(&see, tr_start, N, Xtr, ttr, c1tr, c2tr, dptr, bptr, &trD1,&trDE,&trB3);
    extract_window(&see, va_start, N, Xva, tva, c1va, c2va, NULL, NULL, &vaD1,&vaDE,&vaB3);
    fprintf(stderr,"baselines  TRAIN-win: D1=%.4f delta=%.4f B3=%.4f   VAL-win: D1=%.4f delta=%.4f B3=%.4f\n",trD1,trDE,trB3,vaD1,vaDE,vaB3);

    printf("\n==== 47.0 teacher-transfer feasibility (student=[SEE|L2_D1], TRAIN-win -> VAL-win generalization) ====\n");
    printf("VAL-win baselines: D1 %.4f | delta teacher %.4f | B3 teacher %.4f\n",vaD1,vaDE,vaB3);
    printf("  %-22s %10s %12s\n","probe (train->val)","val_BPB","gain_recov%");
    #define GAIN(x,teach) (100.0*(vaD1-(x))/(vaD1-(teach)))
    printf("  %-22s %10.4f %12.1f\n","linear bound (=D1)",vaD1,GAIN(vaD1,vaDE));
    { MLP m; mlp_init(&m,S_DIM,H); mlp_train(&m,Xtr,ttr,NULL,c1tr,c2tr,S_DIM,0,N,EP,0.0005f,0.0f); double b=mlp_eval(&m,Xva,tva,c1va,c2va,S_DIM,0,N);
      printf("  %-22s %10.4f %12.1f\n","mlp hard",b,GAIN(b,vaDE)); }
    { MLP m; mlp_init(&m,S_DIM,H); mlp_train(&m,Xtr,ttr,dptr,c1tr,c2tr,S_DIM,0,N,EP,0.0005f,0.5f); double b=mlp_eval(&m,Xva,tva,c1va,c2va,S_DIM,0,N);
      printf("  %-22s %10.4f %12.1f\n","mlp KD-delta",b,GAIN(b,vaDE)); }
    { MLP m; mlp_init(&m,S_DIM,H); mlp_train(&m,Xtr,ttr,bptr,c1tr,c2tr,S_DIM,0,N,EP,0.0005f,0.5f); double b=mlp_eval(&m,Xva,tva,c1va,c2va,S_DIM,0,N);
      printf("  %-22s %10.4f %12.1f\n","mlp KD-B3 (vs B3)",b,GAIN(b,vaB3)); }

    printf("\nReading: trained on a TRAIN-region window, evaluated on a held-out VAL window (mirrors how\n");
    printf("the models generalize). If mlp/KD beat the linear bound (=D1) toward the teacher -> the gain\n");
    printf("is a static nonlinear function of stable features -> DISTILLABLE (47.A viable). If they tie\n");
    printf("D1 -> the gain is dynamical, not in instantaneous stable features -> need a recurrent substrate.\n");
    return 0;
}
