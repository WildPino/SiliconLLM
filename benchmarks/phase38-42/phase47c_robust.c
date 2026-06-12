// Phase 47.C - Closed-Loop Robustness Training for the static decoder (NO new memory)
//
// 47.B closed the "it's just calibratable overconfidence" hypothesis: B2rms lowers RMS and
// keeps BPB but topBi stays high (logit scale is not the primary cause); B1ls lowers topBi
// by destroying the language; B4drop is the only "healthy" ingredient but not enough.
// New diagnosis: EXPOSURE MISMATCH - the static decoder is good on true-prefix features,
// then in generation it meets states produced by ITSELF and cannot re-enter.
//
// 47.C trains the same small MLP (H32/H64) on DIRTIED states, targets stay TRUE bytes:
//   C1 byte-corrupt p=0.02   - during extraction the observed byte is sometimes replaced by
//   C2 byte-corrupt p=0.05     a D1-sample @ T0.65 (substrate+contexts follow the corrupted
//                              stream; the training target remains the true next byte)
//   C3 scheduled bursts      - every 256 bytes, K=12 bytes driven by D1 self-samples, then
//                              re-anchor to the corpus (learn to come back)
//   C4 consistency loss      - paired clean/corrupt features; CE on both + lamC*MSE between
//                              PRE-trigram logits (close, not uniform)
//   C0see/C5see              - SEE-only (dim 192) clean vs corrupt: the A0 ablation says the
//                              nonlinear gain lives in SEE, not in D1 memory
// Corruption source = D1 readout sample (stable, precomputable -> extract once, train after;
// using the in-training MLP would make extraction online and non-reproducible).
// Eval: 3 CLEAN val windows (comparable to A0/47.B) + 1 CORRUPTED val window (p=0.05) as a
// robustness transfer diagnostic. frozenD1 anchor always evaluated (A0b lesson).
// No L2/L3, no inference caps, no word counters in the policy. Saves 0x53454545 (generator
// unchanged; SEE-only probes save dim=192 and the generator slices accordingly).
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase38-42/phase47c_robust.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase47c_robust.exe -lm -I .
// Run:
//   bin/phase47c_robust.exe <data> <D1_w> <outprefix> [--len N --epochs E]

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
#define CORR_TEMP 0.65f

enum { G_ENTROPY=4 };
enum { X_CLEAN=0, X_CORRUPT=1, X_BURST=2 };

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
// D1 logits on nf with contexts (c1,c2): returns BPB of tgt; fills probs at CORR_TEMP if asked
static inline double d1_logits(const float* nf, uint8_t c1,uint8_t c2,uint8_t tgt, float* probsT){
    const float* tri=&trigram[c2][c1][0]; float lg[CLASSES],mx=-1e30f;
    for(int c=0;c<CLASSES;c++){ lg[c]=Bd1[c]+tri[c]+dot_avx(Wd1[c],nf,D1_TOT); if(lg[c]>mx)mx=lg[c]; }
    double Z=0; for(int c=0;c<CLASSES;c++) Z+=exp((double)(lg[c]-mx));
    if(probsT){ float mt=-1e30f; for(int c=0;c<CLASSES;c++){ float v=lg[c]/CORR_TEMP; if(v>mt)mt=v; }
        double Zt=0; for(int c=0;c<CLASSES;c++){ probsT[c]=expf(lg[c]/CORR_TEMP-mt); Zt+=probsT[c]; }
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

// ---- MLP probe, plain CE (47.B dose sweeps closed) + optional paired consistency ----
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
// train: CE on (X,c1,c2,tgt). If Xb!=NULL (paired clean/corrupt at the same positions),
// adds CE on the pair + lamC * MSE between PRE-trigram logits (consistency, not uniformity).
static void mlp_train(MLP* m,const float* X,const uint8_t* c1a,const uint8_t* c2a,
                      const float* Xb,const uint8_t* c1b,const uint8_t* c2b,float lamC,
                      const uint8_t* tgt,long n,int epochs,float lr){
    int H=m->h,in=m->in,lin=(H==0),hw=lin?in:H,paired=(Xb!=NULL);
    size_t s1=lin?0:(size_t)H*in, s2=(size_t)CLASSES*hw;
    float* gW1=malloc((s1?s1:1)*4); float* gb1=malloc((H>0?H:1)*4); float* gW2=malloc(s2*4); float* gb2=malloc(CLASSES*4);
    float* hidA=malloc((H>0?H:1)*4); float* hidB=malloc((H>0?H:1)*4); float* dh=malloc((H>0?H:1)*4);
    float lgA[CLASSES],lgB[CLASSES],eoA[CLASSES],eoB[CLASSES]; int bs=512;
    float norm=paired?(2.0f*bs):(float)bs;
    for(int ep=0;ep<epochs;ep++){
        memset(gW1,0,(s1?s1:1)*4); memset(gb1,0,(H>0?H:1)*4); memset(gW2,0,s2*4); memset(gb2,0,CLASSES*4); long inb=0;
        for(long i=0;i<n;i++){
            const float* xA=&X[(size_t)i*S_DIM];
            mlp_fwd(m,xA,hidA,lgA);
            const float* xB=NULL;
            if(paired){ xB=&Xb[(size_t)i*S_DIM]; mlp_fwd(m,xB,hidB,lgB); }
            // CE grads branch A (pre-trigram logits kept in lgA for consistency term)
            float ce[CLASSES]; memcpy(ce,lgA,sizeof(ce));
            { const float* tri=&trigram[c2a[i]][c1a[i]][0]; for(int c=0;c<CLASSES;c++) ce[c]+=tri[c];
              float mx=-1e30f; for(int c=0;c<CLASSES;c++) if(ce[c]>mx)mx=ce[c];
              float Z=0; for(int c=0;c<CLASSES;c++){ eoA[c]=expf(ce[c]-mx); Z+=eoA[c]; }
              for(int c=0;c<CLASSES;c++){ float y=(c==tgt[i])?1.f:0.f; eoA[c]=(eoA[c]/Z-y)/norm; } }
            if(paired){
              memcpy(ce,lgB,sizeof(ce));
              const float* tri=&trigram[c2b[i]][c1b[i]][0]; for(int c=0;c<CLASSES;c++) ce[c]+=tri[c];
              float mx=-1e30f; for(int c=0;c<CLASSES;c++) if(ce[c]>mx)mx=ce[c];
              float Z=0; for(int c=0;c<CLASSES;c++){ eoB[c]=expf(ce[c]-mx); Z+=eoB[c]; }
              for(int c=0;c<CLASSES;c++){ float y=(c==tgt[i])?1.f:0.f; eoB[c]=(eoB[c]/Z-y)/norm; }
              // consistency: d/dlg lamC/(2C) * ||lgA-lgB||^2
              float cc=lamC/(CLASSES*norm);
              for(int c=0;c<CLASSES;c++){ float d=cc*(lgA[c]-lgB[c]); eoA[c]+=d; eoB[c]-=d; }
            }
            for(int c=0;c<CLASSES;c++) gb2[c]+=eoA[c]+(paired?eoB[c]:0.0f);
            if(lin){
                for(int c=0;c<CLASSES;c++){ float* gw=&gW2[(size_t)c*in];
                    float eA=eoA[c]; for(int k=0;k<in;k++) gw[k]+=eA*xA[k];
                    if(paired){ float eB=eoB[c]; for(int k=0;k<in;k++) gw[k]+=eB*xB[k]; } }
            } else {
                memset(dh,0,H*4);
                for(int c=0;c<CLASSES;c++){ float e=eoA[c]; float* gw=&gW2[(size_t)c*H]; const float* w2=&m->W2[(size_t)c*H];
                    for(int j=0;j<H;j++){ gw[j]+=e*hidA[j]; dh[j]+=e*w2[j]; } }
                for(int j=0;j<H;j++) if(hidA[j]>0){ gb1[j]+=dh[j]; float* gw=&gW1[(size_t)j*in]; const float g=dh[j];
                    for(int k=0;k<in;k++) gw[k]+=g*xA[k]; }
                if(paired){
                    memset(dh,0,H*4);
                    for(int c=0;c<CLASSES;c++){ float e=eoB[c]; float* gw=&gW2[(size_t)c*H]; const float* w2=&m->W2[(size_t)c*H];
                        for(int j=0;j<H;j++){ gw[j]+=e*hidB[j]; dh[j]+=e*w2[j]; } }
                    for(int j=0;j<H;j++) if(hidB[j]>0){ gb1[j]+=dh[j]; float* gw=&gW1[(size_t)j*in]; const float g=dh[j];
                        for(int k=0;k<in;k++) gw[k]+=g*xB[k]; }
                }
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
    free(gW1);free(gb1);free(gW2);free(gb2);free(hidA);free(hidB);free(dh);
}
static void mlp_save(MLP* m,const char* path){
    FILE* f=fopen(path,"wb"); if(!f){ fprintf(stderr,"save %s failed\n",path); return; }
    uint32_t magic=0x53454545,H=(uint32_t)m->h,o=0,d=(uint32_t)m->in;
    fwrite(&magic,4,1,f); fwrite(&H,4,1,f); fwrite(&o,4,1,f); fwrite(&d,4,1,f);
    if(m->h>0){ fwrite(m->W1,4,(size_t)m->h*m->in,f); fwrite(m->b1,4,m->h,f); fwrite(m->W2,4,(size_t)CLASSES*m->h,f); }
    else fwrite(m->W2,4,(size_t)CLASSES*m->in,f);
    fwrite(m->b2,4,CLASSES,f); fclose(f);
}

// extraction with optional dirtied stream. Substrate (SEE+L2) and trigram contexts follow
// the OBSERVED (possibly corrupted) stream; the training target is always the TRUE byte.
// mode X_CLEAN reproduces the 47.A0/47.B extraction byte-identically (ob==t always).
static void extract_window_x(SiliconEntropyState* see, long start, long N, int mode, float p, int period, int K, uint64_t rseed,
                             float* X, uint8_t* tgt, uint8_t* c1a, uint8_t* c2a,
                             double* oTRI, double* oD1, double* o_corr){
    float L2d1[L2_DIM]={0}, pb_d1[BASE_DIM]={0};
    float feat192[BASE_DIM], fa[BASE_DIM], rawd1[D1_TOT], nf[D1_TOT], probsT[CLASSES];
    float scale=1.0f/sqrtf((float)BASE_DIM); double btr=0,bd1=0; long ncorr=0;
    uint64_t rng=rseed?rseed:0x9E3779B97F4A7C15ULL; int burst=0;
    see_reset(see); for(long i=0;i<=start+1;i++) see_observe(see,g_data[i]);
    uint8_t cur_c2=g_data[start], cur_c1=g_data[start+1];
    for(long i=0;i<N;i++){
        long g=start+i; uint8_t t=g_data[g+2];
        see_extract(see,feat192);
        memcpy(rawd1,feat192,BASE_DIM*4); memcpy(rawd1+BASE_DIM,L2d1,L2_DIM*4);
        btr+=trigram_bpb(cur_c1,cur_c2,t);
        norm_feats(rawd1,nf);
        // decide the observed byte: clean=t; corrupt: D1-sample with prob p; burst: K-byte
        // D1-driven stretches every `period` bytes, then re-anchor to the corpus
        int dirty=0;
        if(mode==X_CORRUPT){ rng^=rng<<13;rng^=rng>>7;rng^=rng<<17; if(((rng>>11)*(1.0/(1ULL<<53)))<p) dirty=1; }
        else if(mode==X_BURST){ if(i>0 && (i%period)==0) burst=K; if(burst>0){ dirty=1; burst--; } }
        bd1+=d1_logits(nf,cur_c1,cur_c2,t,dirty?probsT:NULL);
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

typedef struct { float* X; uint8_t *tgt,*c1,*c2; double tri,d1,corr; long start; const char* name; } Win;

static Win alloc_win(long N,long start,const char* name){
    Win w; w.X=malloc((size_t)N*S_DIM*4); w.tgt=malloc(N); w.c1=malloc(N); w.c2=malloc(N);
    w.start=start; w.name=name; w.tri=0; w.d1=0; w.corr=0;
    if(!w.X||!w.tgt){ fprintf(stderr,"OOM win %s\n",name); exit(1); }
    return w;
}

int main(int argc, char** argv){
    if(argc<4){ fprintf(stderr,"Usage: %s <data> <D1_w> <outprefix> [--len N --epochs E]\n",argv[0]); return 1; }
    setvbuf(stderr,NULL,_IONBF,0); setvbuf(stdout,NULL,_IONBF,0);
    long N=1000000; int EP=6;
    for(int i=4;i<argc;i++){ if(!strcmp(argv[i],"--len")&&i+1<argc)N=atol(argv[++i]);
        else if(!strcmp(argv[i],"--epochs")&&i+1<argc)EP=atoi(argv[++i]); }
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
    fprintf(stderr,"47.C: train @%ld, val @%ld/%ld/%ld (N=%ld) ep=%d corrT=%.2f\n",
            tr_start,va_start[0],va_start[1],va_start[2],N,EP,CORR_TEMP);

    // train-side windows (same position, different exposure) + clean vals + corrupted val
    Win tr0 = alloc_win(N,tr_start,"tr_clean");
    Win tc2 = alloc_win(N,tr_start,"tr_corr02");
    Win tc5 = alloc_win(N,tr_start,"tr_corr05");
    Win tbu = alloc_win(N,tr_start,"tr_burst");
    Win va[N_VAL]; for(int w=0;w<N_VAL;w++){ char nm[8]; snprintf(nm,8,"val%d",w+1); va[w]=alloc_win(N,va_start[w],"val"); }
    Win vaC = alloc_win(N,va_start[0],"valC");
    fprintf(stderr,"~%.1f GB feature RAM\n",(double)N*8*S_DIM*4/1e9);

    fprintf(stderr,"extract tr_clean...\n");  extract_window_x(&see,tr_start,N,X_CLEAN,0,0,0,0,        tr0.X,tr0.tgt,tr0.c1,tr0.c2,&tr0.tri,&tr0.d1,NULL);
    fprintf(stderr,"extract tr_corr02...\n"); extract_window_x(&see,tr_start,N,X_CORRUPT,0.02f,0,0,0xC0DE0002ULL,tc2.X,tc2.tgt,tc2.c1,tc2.c2,&tc2.tri,&tc2.d1,&tc2.corr);
    fprintf(stderr,"extract tr_corr05...\n"); extract_window_x(&see,tr_start,N,X_CORRUPT,0.05f,0,0,0xC0DE0005ULL,tc5.X,tc5.tgt,tc5.c1,tc5.c2,&tc5.tri,&tc5.d1,&tc5.corr);
    fprintf(stderr,"extract tr_burst...\n");  extract_window_x(&see,tr_start,N,X_BURST,0,256,12,0xC0DEB057ULL,tbu.X,tbu.tgt,tbu.c1,tbu.c2,&tbu.tri,&tbu.d1,&tbu.corr);
    for(int w=0;w<N_VAL;w++){ fprintf(stderr,"extract val%d (clean)...\n",w+1);
        extract_window_x(&see,va_start[w],N,X_CLEAN,0,0,0,0, va[w].X,va[w].tgt,va[w].c1,va[w].c2,&va[w].tri,&va[w].d1,NULL); }
    fprintf(stderr,"extract valC (corrupt p=0.05)...\n");
    extract_window_x(&see,va_start[0],N,X_CORRUPT,0.05f,0,0,0x7A1C0DE5ULL,vaC.X,vaC.tgt,vaC.c1,vaC.c2,&vaC.tri,&vaC.d1,&vaC.corr);

    printf("\n==== 47.C closed-loop robustness training (H32/H64, N=%ld, ep=%d) ====\n",N,EP);
    printf("BASE win=train tri=%.4f d1=%.4f\n",tr0.tri,tr0.d1);
    printf("BASE win=tr_corr02 d1=%.4f corr%%=%.2f\n",tc2.d1,tc2.corr);
    printf("BASE win=tr_corr05 d1=%.4f corr%%=%.2f\n",tc5.d1,tc5.corr);
    printf("BASE win=tr_burst d1=%.4f corr%%=%.2f\n",tbu.d1,tbu.corr);
    for(int w=0;w<N_VAL;w++) printf("BASE win=val%d tri=%.4f d1=%.4f\n",w+1,va[w].tri,va[w].d1);
    printf("BASE win=valC tri=%.4f d1=%.4f corr%%=%.2f\n",vaC.tri,vaC.d1,vaC.corr);

    // frozenD1 anchor through the probe path (clean val) - A0b lesson
    { MLP m; mlp_init(&m,S_DIM,0);
      for(int c=0;c<CLASSES;c++){ memcpy(&m.W2[(size_t)c*S_DIM],&Wd1[c][0],S_DIM*4); m.b2[c]=Bd1[c]; }
      double v[N_VAL],vc,rms=0,ent=0,mxp=0;
      for(int w=0;w<N_VAL;w++) v[w]=mlp_eval(&m,va[w].X,va[w].tgt,va[w].c1,va[w].c2,N,
                                             w==0?&rms:NULL,w==0?&ent:NULL,w==0?&mxp:NULL);
      vc=mlp_eval(&m,vaC.X,vaC.tgt,vaC.c1,vaC.c2,N,NULL,NULL,NULL);
      printf("PROBE name=frozenD1 h=0 dim=%d val1=%.4f val2=%.4f val3=%.4f valC=%.4f rms1=%.3f ent1=%.3f maxp1=%.3f\n",
             S_DIM,v[0],v[1],v[2],vc,rms,ent,mxp);
      mlp_free(&m); }

    // the matrix: train window varies (exposure), eval is ALWAYS clean val1-3 + corrupted valC
    struct { const char* name; int H,dim; Win* trw; Win* pair; float lamC; } cfgs[] = {
        { "C1corr02_h32", 32, S_DIM,    &tc2, NULL, 0.0f },
        { "C1corr02_h64", 64, S_DIM,    &tc2, NULL, 0.0f },
        { "C2corr05_h32", 32, S_DIM,    &tc5, NULL, 0.0f },
        { "C2corr05_h64", 64, S_DIM,    &tc5, NULL, 0.0f },
        { "C3burst_h32",  32, S_DIM,    &tbu, NULL, 0.0f },
        { "C3burst_h64",  64, S_DIM,    &tbu, NULL, 0.0f },
        { "C4cons_h32",   32, S_DIM,    &tr0, &tc2, 0.05f },
        { "C4cons_h64",   64, S_DIM,    &tr0, &tc2, 0.05f },
        { "C0see_h64",    64, BASE_DIM, &tr0, NULL, 0.0f },
        { "C5see_h64",    64, BASE_DIM, &tc2, NULL, 0.0f },
    };
    int ncfg=(int)(sizeof(cfgs)/sizeof(cfgs[0]));
    char sp[512];
    for(int ci=0;ci<ncfg;ci++){
        fprintf(stderr,"  probe %s (H=%d dim=%d train=%s%s)...\n",cfgs[ci].name,cfgs[ci].H,cfgs[ci].dim,
                cfgs[ci].trw==&tr0?"clean":(cfgs[ci].trw==&tc2?"corr02":(cfgs[ci].trw==&tc5?"corr05":"burst")),
                cfgs[ci].pair?"+pair":"");
        MLP m; mlp_init(&m,cfgs[ci].dim,cfgs[ci].H);
        Win* T=cfgs[ci].trw; Win* P=cfgs[ci].pair;
        mlp_train(&m,T->X,T->c1,T->c2, P?P->X:NULL,P?P->c1:NULL,P?P->c2:NULL,cfgs[ci].lamC,
                  T->tgt,N,EP,0.0005f);
        double v[N_VAL],vc,rms=0,ent=0,mxp=0;
        for(int w=0;w<N_VAL;w++) v[w]=mlp_eval(&m,va[w].X,va[w].tgt,va[w].c1,va[w].c2,N,
                                               w==0?&rms:NULL,w==0?&ent:NULL,w==0?&mxp:NULL);
        vc=mlp_eval(&m,vaC.X,vaC.tgt,vaC.c1,vaC.c2,N,NULL,NULL,NULL);
        printf("PROBE name=%s h=%d dim=%d val1=%.4f val2=%.4f val3=%.4f valC=%.4f rms1=%.3f ent1=%.3f maxp1=%.3f\n",
               cfgs[ci].name,cfgs[ci].H,cfgs[ci].dim,v[0],v[1],v[2],vc,rms,ent,mxp);
        snprintf(sp,sizeof sp,"%s_%s.bin",argv[3],cfgs[ci].name);
        mlp_save(&m,sp); printf("SAVED name=%s path=%s\n",cfgs[ci].name,sp);
        mlp_free(&m);
    }

    printf("\nReading: valC = transfer onto a dirtied stream (robustness proxy); the REAL verdict\n");
    printf("is the closed-loop word-gate (mini first: topBi must drop vs the A0 unregularized refs\n");
    printf("at the same H without selfBPB exploding, then full). Train exposure changes ONLY the\n");
    printf("training features: same decoder, same generator, no inference-side changes.\n");
    return 0;
}
