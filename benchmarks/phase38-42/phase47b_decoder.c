// Phase 47.B - Regularized Static Decoder (H32/H64, NO new memory, NO inference hacks)
//
// 47.A0b diagnosis: the static nonlinear readout reproduces compression on 3 windows with
// clean controls, and in closed-loop it CURES nameWst/runWst/altLp - it fails almost only
// on topBi + OOD overconfidence (ent_p10 ~ 0, closed-loop logit RMS >> teacher-forced RMS).
// That is a DIFFERENT wall than the L2/L3 volatile-memory attractors: the decoder is not
// destroying structure, it is becoming too certain. 47.B regularizes the decoder at TRAIN
// time (trained properties, nothing capped at inference):
//   B1  label smoothing            - forbid near-zero output entropy
//   B2  logit RMS penalty          - keep pre-trigram RMS near the D1/TF range (~4.8), as a
//                                    trained property, not an inference cap
//   B3  weight decay (H64)        - smoother surface, not a BPB play
//   B4  hidden dropout train-time - don't let the decoder become a sharp nonlinear lookup
//   B5  combined (moderate doses)
// H32/H64 only: H128 is the compression champion but already too hungry. Success criterion
// (user): NOT 2.08 - even 2.23-2.24 is a win if topBi<=8, altLp<=2 and name/run stay low.
// The win is closing the loop. frozenD1 anchor always evaluated (47.A0b lesson).
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase38-42/phase47b_decoder.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase47b_decoder.exe -lm -I .
// Run:
//   bin/phase47b_decoder.exe <data> <D1_w> <outprefix> [--len N --epochs E]

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

enum { G_ENTROPY=4 };

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

// D1 loader (0x53454540 only - 47.B needs no delta/B3 teachers)
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
static inline double d1_bpb(const float* nf, uint8_t c1,uint8_t c2,uint8_t tgt){
    const float* tri=&trigram[c2][c1][0]; float lg[CLASSES],mx=-1e30f;
    for(int c=0;c<CLASSES;c++){ lg[c]=Bd1[c]+tri[c]+dot_avx(Wd1[c],nf,D1_TOT); if(lg[c]>mx)mx=lg[c]; }
    double Z=0; for(int c=0;c<CLASSES;c++) Z+=exp((double)(lg[c]-mx));
    double p=exp((double)(lg[tgt]-mx))/Z; return -log2(p>1e-30?p:1e-30);
}
static inline double trigram_bpb(uint8_t c1,uint8_t c2,uint8_t tgt){
    const float* tri=&trigram[c2][c1][0]; float mx=-1e30f;
    for(int c=0;c<CLASSES;c++) if(tri[c]>mx)mx=tri[c];
    double Z=0; for(int c=0;c<CLASSES;c++) Z+=exp((double)(tri[c]-mx));
    double p=exp((double)(tri[tgt]-mx))/Z; return -log2(p>1e-30?p:1e-30);
}

// ---- MLP probe (h==0 = linear), 0x53454545 save format (phase47_generator-compatible) ----
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
// regularized training: Adam + hard CE with
//   ls       label smoothing (target = (1-ls)*onehot + ls/256)
//   rms_tau/rms_lam  penalty lam*max(0, rms_pre_trigram - tau)^2 (trained property)
//   wd       decoupled weight decay inside Adam step
//   drop     inverted dropout on hidden, TRAIN ONLY, deterministic RNG
static void mlp_train(MLP* m,const float* X,const uint8_t* tgt,const uint8_t* c1a,const uint8_t* c2a,
                      long n,int epochs,float lr,float ls,float rms_tau,float rms_lam,float wd,float drop){
    int H=m->h,in=m->in,lin=(H==0),hw=lin?in:H;
    size_t s1=lin?0:(size_t)H*in, s2=(size_t)CLASSES*hw;
    float* gW1=malloc((s1?s1:1)*4); float* gb1=malloc((H>0?H:1)*4); float* gW2=malloc(s2*4); float* gb2=malloc(CLASSES*4);
    float* hid=malloc((H>0?H:1)*4); float* dh=malloc((H>0?H:1)*4);
    float lg[CLASSES],lgp[CLASSES],eo[CLASSES]; int bs=512;
    uint64_t drng=0xD20DC0DE12345ULL; float keep=1.0f-drop, kinv=(keep>0)?1.0f/keep:0.0f;
    for(int ep=0;ep<epochs;ep++){
        memset(gW1,0,(s1?s1:1)*4); memset(gb1,0,(H>0?H:1)*4); memset(gW2,0,s2*4); memset(gb2,0,CLASSES*4); long inb=0;
        for(long i=0;i<n;i++){
            const float* x=&X[(size_t)i*S_DIM];
            mlp_fwd(m,x,hid,lg);
            if(drop>0 && H>0) for(int j=0;j<H;j++){ drng^=drng<<13;drng^=drng>>7;drng^=drng<<17;
                if(((drng>>11)*(1.0/(1ULL<<53)))<drop) hid[j]=0; else hid[j]*=kinv; }
            if(drop>0 && H>0) for(int c=0;c<CLASSES;c++) lg[c]=m->b2[c]+dot_avx(&m->W2[(size_t)c*H],hid,H);  // re-fwd output on dropped hidden
            memcpy(lgp,lg,sizeof(lgp));                      // pre-trigram logits (for RMS penalty)
            double mu=0; for(int c=0;c<CLASSES;c++) mu+=lgp[c]; mu/=CLASSES;
            double var=0; for(int c=0;c<CLASSES;c++){ double d=lgp[c]-mu; var+=d*d; }
            double rms=sqrt(var/CLASSES);
            float pcoef=0.0f;
            if(rms_lam>0 && rms>rms_tau) pcoef=(float)(rms_lam*2.0*(rms-rms_tau)/(CLASSES*(rms>1e-9?rms:1e-9)));
            const float* tri=&trigram[c2a[i]][c1a[i]][0]; for(int c=0;c<CLASSES;c++) lg[c]+=tri[c];
            float mx=-1e30f; for(int c=0;c<CLASSES;c++) if(lg[c]>mx)mx=lg[c];
            float Z=0; for(int c=0;c<CLASSES;c++){ eo[c]=expf(lg[c]-mx); Z+=eo[c]; }
            float lsu=ls/CLASSES;
            for(int c=0;c<CLASSES;c++){ float y=((c==tgt[i])?(1.0f-ls):0.0f)+lsu;
                eo[c]=(eo[c]/Z-y)/bs + pcoef*(float)(lgp[c]-mu)/bs; gb2[c]+=eo[c]; }
            if(lin){
                for(int c=0;c<CLASSES;c++){ float e=eo[c]; float* gw=&gW2[(size_t)c*in]; for(int k=0;k<in;k++) gw[k]+=e*x[k]; }
            } else {
                memset(dh,0,H*4);
                for(int c=0;c<CLASSES;c++){ float e=eo[c]; float* gw=&gW2[(size_t)c*H]; const float* w2=&m->W2[(size_t)c*H];
                    for(int j=0;j<H;j++){ gw[j]+=e*hid[j]; dh[j]+=e*w2[j]; } }
                // dropped units have hid=0 -> no grad; kept units carry the inverted-dropout
                // scale kinv through relu' (hid = relu(a)*kinv on kept units)
                for(int j=0;j<H;j++) if(hid[j]>0){ float gj=(drop>0)?dh[j]*kinv:dh[j];
                    gb1[j]+=gj; float* gw=&gW1[(size_t)j*in];
                    for(int k=0;k<in;k++) gw[k]+=gj*x[k]; }
            }
            inb++;
            if(inb==bs || i==n-1){ m->t++; float lt=lr*sqrtf(1-powf(.999f,m->t))/(1-powf(.9f,m->t));
                #define ADAM(P,G,MM,VV,NN) for(size_t z=0;z<(size_t)(NN);z++){ MM[z]=.9f*MM[z]+.1f*G[z]; VV[z]=.999f*VV[z]+.001f*G[z]*G[z]; P[z]-=lt*(MM[z]/(sqrtf(VV[z])+1e-8f)+wd*P[z]); }
                if(!lin){ ADAM(m->W1,gW1,m->mW1,m->vW1,s1); ADAM(m->b1,gb1,m->mb1,m->vb1,H); }
                ADAM(m->W2,gW2,m->mW2,m->vW2,s2); ADAM(m->b2,gb2,m->mb2,m->vb2,CLASSES);
                memset(gW1,0,(s1?s1:1)*4); memset(gb1,0,(H>0?H:1)*4); memset(gW2,0,s2*4); memset(gb2,0,CLASSES*4); inb=0; }
        }
        fprintf(stderr,"    ep %d/%d done\n",ep+1,epochs);
    }
    free(gW1);free(gb1);free(gW2);free(gb2);free(hid);free(dh);
}
static void mlp_save(MLP* m,const char* path){
    FILE* f=fopen(path,"wb"); if(!f){ fprintf(stderr,"save %s failed\n",path); return; }
    uint32_t magic=0x53454545,H=(uint32_t)m->h,o=0,d=(uint32_t)m->in;
    fwrite(&magic,4,1,f); fwrite(&H,4,1,f); fwrite(&o,4,1,f); fwrite(&d,4,1,f);
    if(m->h>0){ fwrite(m->W1,4,(size_t)m->h*m->in,f); fwrite(m->b1,4,m->h,f); fwrite(m->W2,4,(size_t)CLASSES*m->h,f); }
    else fwrite(m->W2,4,(size_t)CLASSES*m->in,f);
    fwrite(m->b2,4,CLASSES,f); fclose(f);
}

// teacher-forced extraction (parity with phase47_a0_gauntlet, D1-only branch)
static void extract_window(SiliconEntropyState* see, long start, long N,
                           float* X, uint8_t* tgt, uint8_t* c1a, uint8_t* c2a, double* oTRI, double* oD1){
    float L2d1[L2_DIM]={0}, pb_d1[BASE_DIM]={0};
    float feat192[BASE_DIM], fa[BASE_DIM], rawd1[D1_TOT], nf[D1_TOT];
    float scale=1.0f/sqrtf((float)BASE_DIM); double btr=0,bd1=0;
    see_reset(see); for(long i=0;i<=start+1;i++) see_observe(see,g_data[i]);
    for(long i=0;i<N;i++){
        long g=start+i; uint8_t c2=g_data[g],c1=g_data[g+1],t=g_data[g+2];
        see_extract(see,feat192);
        memcpy(rawd1,feat192,BASE_DIM*4); memcpy(rawd1+BASE_DIM,L2d1,L2_DIM*4);
        btr+=trigram_bpb(c1,c2,t);
        norm_feats(rawd1,nf); bd1+=d1_bpb(nf,c1,c2,t);
        memcpy(&X[(size_t)i*S_DIM],nf,S_DIM*4);
        tgt[i]=t; c1a[i]=c1; c2a[i]=c2;
        see_observe(see,t); see_extract(see,fa);
        if(ent_gate(c1,c2)){ float src5[BASE_DIM];
            for(int k=0;k<BASE_DIM;k++) src5[k]=fa[k]-0.5f*pb_d1[k]; memcpy(pb_d1,fa,BASE_DIM*4);
            for(int j=0;j<L2_DIM;j++){ float p5=0; const float* pj=Pmat[j]; for(int k=0;k<BASE_DIM;k++) p5+=pj[k]*src5[k];
                L2d1[j]=g_alpha*L2d1[j]+(1.0f-g_alpha)*p5*scale; } }
    }
    *oTRI=btr/N; *oD1=bd1/N;
}

typedef struct { float* X; uint8_t *tgt,*c1,*c2; double tri,d1; long start; } Win;
typedef struct { const char* name; int H; float ls,tau,lam,wd,drop; } Cfg;

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
    fprintf(stderr,"47.B: train @%ld, val @%ld/%ld/%ld (N=%ld) ep=%d\n",tr_start,va_start[0],va_start[1],va_start[2],N,EP);

    Win tr; Win va[N_VAL];
    tr.X=malloc((size_t)N*S_DIM*4); tr.tgt=malloc(N); tr.c1=malloc(N); tr.c2=malloc(N); tr.start=tr_start;
    for(int w=0;w<N_VAL;w++){ va[w].X=malloc((size_t)N*S_DIM*4); va[w].tgt=malloc(N); va[w].c1=malloc(N); va[w].c2=malloc(N); va[w].start=va_start[w]; }
    if(!tr.X||!va[0].X||!va[1].X||!va[2].X){ fprintf(stderr,"OOM (~%.1f GB)\n",(double)N*(N_VAL+1)*S_DIM*4/1e9); return 1; }
    fprintf(stderr,"extracting train window...\n");
    extract_window(&see,tr_start,N,tr.X,tr.tgt,tr.c1,tr.c2,&tr.tri,&tr.d1);
    for(int w=0;w<N_VAL;w++){ fprintf(stderr,"extracting val window %d...\n",w+1);
        extract_window(&see,va_start[w],N,va[w].X,va[w].tgt,va[w].c1,va[w].c2,&va[w].tri,&va[w].d1); }

    printf("\n==== 47.B regularized static decoder (H32/H64 matrix, N=%ld, ep=%d) ====\n",N,EP);
    printf("BASE win=train start=%ld tri=%.4f d1=%.4f\n",tr.start,tr.tri,tr.d1);
    for(int w=0;w<N_VAL;w++) printf("BASE win=val%d start=%ld tri=%.4f d1=%.4f\n",w+1,va[w].start,va[w].tri,va[w].d1);

    // frozenD1 anchor (47.A0b lesson: every run self-verifies the probe path)
    { MLP m; mlp_init(&m,S_DIM,0);
      for(int c=0;c<CLASSES;c++){ memcpy(&m.W2[(size_t)c*S_DIM],&Wd1[c][0],S_DIM*4); m.b2[c]=Bd1[c]; }
      double v[N_VAL],rms=0,ent=0,mxp=0;
      for(int w=0;w<N_VAL;w++) v[w]=mlp_eval(&m,va[w].X,va[w].tgt,va[w].c1,va[w].c2,N,
                                             w==0?&rms:NULL,w==0?&ent:NULL,w==0?&mxp:NULL);
      printf("PROBE name=frozenD1 h=0 val1=%.4f val2=%.4f val3=%.4f rms1=%.3f ent1=%.3f maxp1=%.3f\n",
             v[0],v[1],v[2],rms,ent,mxp);
      mlp_free(&m); }

    // ---- the matrix: one axis per config + combined, H32/H64 (B3 wd-only at H64) ----
    // RMS target tau=4.5: D1/TF range is ~4.8 (frozenD1 rms1 4.83, mlpH64 TF 4.80) - keep the
    // decoder's trained logit scale at/below the scale D1 generates stably with.
    Cfg cfgs[] = {
        { "B1ls_h32",   32, 0.10f, 0.0f, 0.00f, 1e-5f, 0.0f },
        { "B1ls_h64",   64, 0.10f, 0.0f, 0.00f, 1e-5f, 0.0f },
        { "B2rms_h32",  32, 0.00f, 4.5f, 0.02f, 1e-5f, 0.0f },
        { "B2rms_h64",  64, 0.00f, 4.5f, 0.02f, 1e-5f, 0.0f },
        { "B3wd_h64",   64, 0.00f, 0.0f, 0.00f, 1e-3f, 0.0f },
        { "B4drop_h32", 32, 0.00f, 0.0f, 0.00f, 1e-5f, 0.2f },
        { "B4drop_h64", 64, 0.00f, 0.0f, 0.00f, 1e-5f, 0.2f },
        { "B5all_h32",  32, 0.05f, 4.5f, 0.02f, 3e-4f, 0.1f },
        { "B5all_h64",  64, 0.05f, 4.5f, 0.02f, 3e-4f, 0.1f },
    };
    int ncfg=(int)(sizeof(cfgs)/sizeof(cfgs[0]));
    char sp[512];
    for(int ci=0;ci<ncfg;ci++){
        Cfg* c=&cfgs[ci];
        fprintf(stderr,"  probe %s (H=%d ls=%.2f tau=%.1f lam=%.2f wd=%.0e drop=%.2f)...\n",
                c->name,c->H,c->ls,c->tau,c->lam,c->wd,c->drop);
        MLP m; mlp_init(&m,S_DIM,c->H);
        mlp_train(&m,tr.X,tr.tgt,tr.c1,tr.c2,N,EP,0.0005f,c->ls,c->tau,c->lam,c->wd,c->drop);
        double v[N_VAL],rms=0,ent=0,mxp=0;
        for(int w=0;w<N_VAL;w++) v[w]=mlp_eval(&m,va[w].X,va[w].tgt,va[w].c1,va[w].c2,N,
                                               w==0?&rms:NULL,w==0?&ent:NULL,w==0?&mxp:NULL);
        printf("PROBE name=%s h=%d ls=%.2f tau=%.1f lam=%.2f wd=%.0e drop=%.2f val1=%.4f val2=%.4f val3=%.4f rms1=%.3f ent1=%.3f maxp1=%.3f\n",
               c->name,c->H,c->ls,c->tau,c->lam,c->wd,c->drop,v[0],v[1],v[2],rms,ent,mxp);
        snprintf(sp,sizeof sp,"%s_%s.bin",argv[3],c->name);
        mlp_save(&m,sp); printf("SAVED name=%s path=%s\n",c->name,sp);
        mlp_free(&m);
    }

    printf("\nReading: per config compare val BPB AND rms1/ent1/maxp1 vs the unregularized A0 probes\n");
    printf("(mlpH32 rms 4.11/ent 2.16, mlpH64 rms 4.80/ent 2.11). The win condition is NOT BPB:\n");
    printf("it is closing the closed-loop (topBi<=8, altLp<=2, name/run low) at BPB <= ~2.25.\n");
    printf("Word-gate + closed-loop telemetry via phase47b.ps1 / phase47_generator (unchanged).\n");
    return 0;
}
