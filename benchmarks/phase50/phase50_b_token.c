// Phase 50.B - DAgger coverage of the TOKEN-REPEAT (to close topBi).
//
// 50.A breakthrough: recognizable English, byte-fidelity solved (all byte-guards pass), BPB 1.69;
// the only full-gate fail is topBi (word/token repeat: "the the", "to the to the"). That is the
// token-flood, the coarse-grained analog of the char-flood already tamed by K128 far-field
// coverage. The K16/K128 bursts are tuned for byte-floods and do not touch it.
//
// One targeted adjustment (hard cap): in rounds 6-9 (branched from the clean r5), alongside
// K16/K128, add a REPEAT burst SEEDED from a token-repeat state -- force a short run of a common
// token (sampled from the most frequent), then roll out from the decoder itself, target = the TRUE
// corpus token -> the model learns to EXIT the token-loop, exactly as it learned to exit the
// char-flood. Aim T0.55 (binding temperature). This is COVERAGE, not an inference hack (same
// mechanism as K128; the generation path is untouched -> the 0x53454554 generator is reused).
//
// A mandatory PREMISE-CHECK (--premise, no training) verifies on the r5 decoder that it actually
// ENTERS/STAYS in token-repeat from seeded states (else seeding teaches nothing -> report).
//
// Also fixes a 50.A train/infer nit: the bigram context of a recorded rollout row is now the LAST
// EMITTED token (what the generator sees), not the true corpus token.
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase38-42/phase50_b_token.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase50_b_token.exe -lm -I .
// Run (train, branch from r5):
//   bin/phase50_b_token.exe <data> <D1_w> <bpe_merges> <outprefix> --resume-from <r5.bin> [--len N]
// Run (premise check, no train):
//   bin/phase50_b_token.exe <data> <D1_w> <bpe_merges> <outprefix> --premise <r5.bin> [--len N]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include "src/silicon_entropy.h"
#include "benchmarks/phase38-42/bpe_codec.h"

#define CLASSES   256
#define BASE_DIM  SEE_FEATURE_DIM
#define L0_DIM    SEE_L0_DIM
#define L2_DIM    64
#define D1_TOT    (BASE_DIM + L2_DIM)
#define D_EXP     128
#define GAMMA     0.25f
#define N_TS      2
#define EXP_BANDS (N_TS*D_EXP)
#define S_DIM     (D1_TOT + EXP_BANDS)
#define PROJ_SEED 0x48B2EC0DEULL
#define N_VAL     3
#define T_HI      0.65f
#define T_REP     0.55f          // repeat coverage targets the binding (confident) temperature
#define RECOV     16
#define EP_PRE    2
#define EP_MIX    2
#define R_PRE     5
#define R_TOT     9
#define K_NEAR    16
#define K_FAR     128
#define PERIOD    256
#define K_REPSEED 6              // forced repeat run length
#define K_REPRECOV 16           // decoder recovery after the seed
#define REP_TOPN  16            // sample the repeat token among the top-N most frequent

static const float TS_ALPHA[N_TS] = { 0.90f, 0.99f };

static float Pmat[L2_DIM][BASE_DIM];
static float Omega[D_EXP][L0_DIM];
static float Bvec[D_EXP];
static float (*trigram)[CLASSES][CLASSES];
static float (*ent_table)[CLASSES];
static uint8_t* g_data; static long g_fsz;
static float g_ent_thr; static int g_ent_high=1;
static float Wd1[CLASSES][D1_TOT], Bd1[CLASSES], md1[D1_TOT], sd1[D1_TOT];
static float g_alpha=0.99f, g_l2c_d1=2.0f, g_ls_d1=0.5f;

static Bpe g_bpe;
static uint32_t* g_tok; static long* g_tokstart; static long g_ntok;
static int VTOK=1024;
static float* g_tbig;
static uint32_t g_freqtok[REP_TOPN]; static int g_nfreq=0;
static uint32_t g_altpairs[8][2]; static int g_naltp=0;     // top alternation bigrams (A!=B)
static uint32_t g_contenttok[8]; static int g_ncontent=0;   // mid-freq alphabetic content words

static inline float dot_avx(const float* w, const float* f, int n){ __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(&w[i]),_mm256_loadu_ps(&f[i]),s);
    float o[8]; _mm256_storeu_ps(o,s); float r=o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; for(;i<n;i++) r+=w[i]*f[i]; return r; }
static void gen_projection(uint32_t seed){ uint64_t s=seed?(uint64_t)seed:0x9E3779B97F4A7C15ULL;
    for(int j=0;j<L2_DIM;j++) for(int k=0;k<BASE_DIM;k++){ s^=s<<13; s^=s>>7; s^=s<<17; Pmat[j][k]=(s&1ULL)?1.f:-1.f; } }
static inline double xs_u01(uint64_t* s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return (double)((*s>>11)&0x1FFFFFFFFFFFFFULL)/(double)(1ULL<<53); }
static void gen_lift(uint64_t seed){ uint64_t s=seed?seed:0xABCDEF12345ULL;
    for(int d=0;d<D_EXP;d++){
        for(int k=0;k<L0_DIM;k++){ double u1=xs_u01(&s); if(u1<1e-12) u1=1e-12; double u2=xs_u01(&s);
            Omega[d][k]=(float)(sqrt(-2.0*log(u1))*cos(6.283185307179586*u2)); }
        Bvec[d]=(float)(6.283185307179586*xs_u01(&s)); } }
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
static inline void armb_fold(const float* feat192, float eB[N_TS][D_EXP]){
    float l0n[L0_DIM];
    for(int k=0;k<L0_DIM;k++){ float x=(feat192[k]-md1[k])/(sd1[k]+1e-8f); if(x>2.f)x=2.f; if(x<-2.f)x=-2.f; l0n[k]=x; }
    for(int d=0;d<D_EXP;d++){ float z=Bvec[d]+GAMMA*dot_avx(Omega[d],l0n,L0_DIM); float cz=cosf(z);
        for(int ts=0;ts<N_TS;ts++){ float a=TS_ALPHA[ts]; eB[ts][d]=a*eB[ts][d]+(1.0f-a)*cz; } }
}
static inline void row_bands(const float eB[N_TS][D_EXP], float* row){
    for(int ts=0;ts<N_TS;ts++) memcpy(row+D1_TOT+ts*D_EXP, eB[ts], D_EXP*4);
}
static inline double trigram_bpb(uint8_t c1,uint8_t c2,uint8_t tgt){
    const float* tri=&trigram[c2][c1][0]; float mx=-1e30f;
    for(int c=0;c<CLASSES;c++) if(tri[c]>mx)mx=tri[c];
    double Z=0; for(int c=0;c<CLASSES;c++) Z+=exp((double)(tri[c]-mx));
    double p=exp((double)(tri[tgt]-mx))/Z; return -log2(p>1e-30?p:1e-30);
}
static inline uint32_t sample_tok(const float* P, uint64_t* rng){
    *rng^=*rng<<13; *rng^=*rng>>7; *rng^=*rng<<17;
    double u=(*rng>>11)*(1.0/(1ULL<<53));
    double c=0; for(int k=0;k<VTOK;k++){ c+=P[k]; if(u<=c) return (uint32_t)k; }
    return (uint32_t)(VTOK-1);
}

static void tokenize_corpus(void){
    g_tok=(uint32_t*)malloc((size_t)g_fsz*sizeof(uint32_t));
    g_tokstart=(long*)malloc((size_t)(g_fsz+1)*sizeof(long));
    g_ntok=(long)bpe_encode_region(&g_bpe, g_data, 0, g_fsz, g_tok);
    long off=0; for(long i=0;i<g_ntok;i++){ g_tokstart[i]=off; off+=bpe_tok_len(&g_bpe,g_tok[i]); }
    g_tokstart[g_ntok]=off;
    fprintf(stderr,"tokenized: %ld tokens over %ld bytes (%.3f b/tok)\n",g_ntok,g_fsz,(double)g_fsz/g_ntok);
}
static int tok_is_word(uint32_t id){ int L=bpe_tok_len(&g_bpe,id); const unsigned char* b=bpe_tok_bytes(&g_bpe,id);
    int s=(L>0 && b[0]==' ')?1:0; int letters=0;
    for(int k=s;k<L;k++){ unsigned char c=b[k]; if((c>='a'&&c<='z')||(c>='A'&&c<='Z')) letters++; else return 0; }
    return letters>=3; }
static void build_tbig(long tok_train_end){
    g_tbig=(float*)malloc((size_t)VTOK*VTOK*sizeof(float));
    double* uni=(double*)calloc(VTOK,sizeof(double));
    double* ctx=(double*)calloc(VTOK,sizeof(double));
    uint32_t* bg=(uint32_t*)calloc((size_t)VTOK*VTOK,sizeof(uint32_t));
    for(long i=0;i<tok_train_end;i++) uni[g_tok[i]]+=1.0;
    for(long i=0;i+1<tok_train_end;i++){ uint32_t a=g_tok[i],b=g_tok[i+1]; bg[(size_t)a*VTOK+b]++; ctx[a]+=1.0; }
    double T=(double)tok_train_end; const double K=0.5;
    for(int a=0;a<VTOK;a++){ double ca=ctx[a];
        for(int b=0;b<VTOK;b++){ double pb=(uni[b]+K)/(T+K*VTOK); double p;
            if(ca>0){ double cab=bg[(size_t)a*VTOK+b]; p=(cab+K*VTOK*pb)/(ca+K*VTOK); } else p=pb;
            g_tbig[(size_t)a*VTOK+b]=(float)log((p>1e-30)?p:1e-30); } }
    // top-N frequent tokens (for repeat seeding) - skip pure single-space token? keep all, they are
    // exactly the repeaters. selection by raw unigram count.
    g_nfreq=0;
    for(int n=0;n<REP_TOPN;n++){ int best=-1; double bc=-1;
        for(int b=0;b<VTOK;b++){ int used=0; for(int q=0;q<g_nfreq;q++) if(g_freqtok[q]==(uint32_t)b){used=1;break;}
            if(!used && uni[b]>bc){ bc=uni[b]; best=b; } }
        if(best<0) break; g_freqtok[g_nfreq++]=(uint32_t)best; }
    fprintf(stderr,"top repeat tokens:"); for(int q=0;q<g_nfreq;q++) fprintf(stderr," %u",g_freqtok[q]); fprintf(stderr,"\n");
    // top alternation pairs (A!=B) by bigram count -> the "to the to the" mode
    g_naltp=0;
    for(int n=0;n<8;n++){ long bestc=-1; int ba=-1,bb=-1;
        for(int a=0;a<VTOK;a++) for(int b=0;b<VTOK;b++){ if(a==b) continue; long c=bg[(size_t)a*VTOK+b];
            int used=0; for(int q=0;q<g_naltp;q++) if(g_altpairs[q][0]==(uint32_t)a && g_altpairs[q][1]==(uint32_t)b){used=1;break;}
            if(!used && c>bestc){ bestc=c; ba=a; bb=b; } }
        if(ba<0) break; g_altpairs[g_naltp][0]=ba; g_altpairs[g_naltp][1]=bb; g_naltp++; }
    fprintf(stderr,"top alt pairs:"); for(int q=0;q<g_naltp;q++) fprintf(stderr," (%u,%u)",g_altpairs[q][0],g_altpairs[q][1]); fprintf(stderr,"\n");
    // content words (alphabetic, len>=3, NOT in top-freq) by frequency -> the "named named" mode
    g_ncontent=0;
    for(int n=0;n<8;n++){ double bestc=-1; int best=-1;
        for(int b=0;b<VTOK;b++){ if(!tok_is_word((uint32_t)b)) continue;
            int used=0; for(int q=0;q<g_ncontent;q++) if(g_contenttok[q]==(uint32_t)b){used=1;break;}
            int isf=0; for(int q=0;q<g_nfreq;q++) if(g_freqtok[q]==(uint32_t)b){isf=1;break;}
            if(!used && !isf && uni[b]>bestc){ bestc=uni[b]; best=b; } }
        if(best<0) break; g_contenttok[g_ncontent++]=(uint32_t)best; }
    fprintf(stderr,"content words:"); for(int q=0;q<g_ncontent;q++) fprintf(stderr," %u",g_contenttok[q]); fprintf(stderr,"\n");
    free(uni); free(ctx); free(bg);
}

typedef struct { int in,h; float *W1,*b1,*W2,*b2,*mW1,*vW1,*mb1,*vb1,*mW2,*vW2,*mb2,*vb2; int t; } MLP;
static void mlp_init(MLP* m,int in,int h){ m->in=in; m->h=h; m->t=0;
    size_t s1=(size_t)h*in, s2=(size_t)VTOK*h;
    m->W1=calloc(s1,4); m->b1=calloc(h,4); m->W2=calloc(s2,4); m->b2=calloc(VTOK,4);
    m->mW1=calloc(s1,4); m->vW1=calloc(s1,4); m->mb1=calloc(h,4); m->vb1=calloc(h,4);
    m->mW2=calloc(s2,4); m->vW2=calloc(s2,4); m->mb2=calloc(VTOK,4); m->vb2=calloc(VTOK,4);
    uint64_t r=0x1234567; float sc1=sqrtf(2.0f/in);
    for(size_t i=0;i<s1;i++){ r^=r<<13;r^=r>>7;r^=r<<17; m->W1[i]=sc1*(((r>>11)*(1.0/(1ULL<<53)))*2-1); }
    float sc2=sqrtf(2.0f/h);
    for(size_t i=0;i<s2;i++){ r^=r<<13;r^=r>>7;r^=r<<17; m->W2[i]=sc2*(((r>>11)*(1.0/(1ULL<<53)))*2-1); }
}
static void mlp_free(MLP* m){ free(m->W1);free(m->b1);free(m->W2);free(m->b2);
    free(m->mW1);free(m->vW1);free(m->mb1);free(m->vb1);free(m->mW2);free(m->vW2);free(m->mb2);free(m->vb2); }
static void mlp_fwd(MLP* m,const float* x,float* hid,float* lg){
    for(int j=0;j<m->h;j++){ float a=m->b1[j]+dot_avx(&m->W1[(size_t)j*m->in],x,m->in); hid[j]=a>0?a:0; }
    for(int c=0;c<VTOK;c++) lg[c]=m->b2[c]+dot_avx(&m->W2[(size_t)c*m->h],hid,m->h);
}
// load just the W1/b1/W2/b2 from a 0x53454554 checkpoint (merges/tbig rebuilt deterministically).
static int mlp_load(MLP* m,const char* path){
    FILE* f=fopen(path,"rb"); if(!f){ fprintf(stderr,"mlp_load open %s\n",path); return 0; }
    uint32_t mm=0,mh=0,md=0,mb=0,mde=0,mn=0; fread(&mm,4,1,f);
    if(mm!=0x53454554){ fprintf(stderr,"mlp_load bad magic 0x%08x\n",mm); fclose(f); return 0; }
    fread(&mh,4,1,f); fread(&md,4,1,f); fread(&mb,4,1,f); fread(&mde,4,1,f); fread(&mn,4,1,f);
    float gm; uint64_t ps; fread(&gm,4,1,f); fread(&ps,8,1,f);
    float ts[N_TS]; for(int t=0;t<N_TS;t++) fread(&ts[t],4,1,f);
    uint32_t vt=0,nm=0; fread(&vt,4,1,f); fread(&nm,4,1,f);
    if((int)mh!=m->h || (int)md!=m->in || (int)vt!=VTOK){ fprintf(stderr,"mlp_load shape mismatch H%u dim%u VTOK%u\n",mh,md,vt); fclose(f); return 0; }
    fseek(f,(long)nm*8,SEEK_CUR);                     // skip merges
    fseek(f,(long)VTOK*VTOK*4,SEEK_CUR);              // skip tbig
    fread(m->W1,4,(size_t)m->h*m->in,f); fread(m->b1,4,m->h,f);
    fread(m->W2,4,(size_t)VTOK*m->h,f); fread(m->b2,4,VTOK,f); fclose(f);
    return 1;
}
static double mlp_eval(MLP* m,const float* X,const uint32_t* tgt,const uint32_t* prev,long n,double winbytes,double* o_bpu){
    float* hid=malloc(m->h*4); float* lg=malloc((size_t)VTOK*4); double tot=0;
    for(long i=0;i<n;i++){
        mlp_fwd(m,&X[(size_t)i*S_DIM],hid,lg);
        const float* tb=&g_tbig[(size_t)prev[i]*VTOK]; for(int c=0;c<VTOK;c++) lg[c]+=tb[c];
        float mx=-1e30f; for(int c=0;c<VTOK;c++) if(lg[c]>mx)mx=lg[c];
        double Z=0; for(int c=0;c<VTOK;c++) Z+=exp((double)(lg[c]-mx));
        double p=exp((double)(lg[tgt[i]]-mx))/Z; tot+=-log2(p>1e-30?p:1e-30);
    }
    free(hid); free(lg); if(o_bpu)*o_bpu=tot/n; return tot/winbytes;
}

typedef struct { float *gW1,*gb1,*gW2,*gb2; } Grads;
static void accum_sample(MLP* m,const float* x,uint32_t tgt,uint32_t prev,float invnorm,
                         Grads* G,float* hid,float* dh,float* lg,float* eo){
    int H=m->h,in=m->in;
    mlp_fwd(m,x,hid,lg);
    const float* tb=&g_tbig[(size_t)prev*VTOK]; for(int c=0;c<VTOK;c++) lg[c]+=tb[c];
    float mx=-1e30f; for(int c=0;c<VTOK;c++) if(lg[c]>mx)mx=lg[c];
    float Z=0; for(int c=0;c<VTOK;c++){ eo[c]=expf(lg[c]-mx); Z+=eo[c]; }
    for(int c=0;c<VTOK;c++){ float y=(c==(int)tgt)?1.f:0.f; eo[c]=(eo[c]/Z-y)*invnorm; }
    for(int c=0;c<VTOK;c++) G->gb2[c]+=eo[c];
    memset(dh,0,H*4);
    for(int c=0;c<VTOK;c++){ float e=eo[c]; float* gw=&G->gW2[(size_t)c*H]; const float* w2=&m->W2[(size_t)c*H];
        for(int j=0;j<H;j++){ gw[j]+=e*hid[j]; dh[j]+=e*w2[j]; } }
    for(int j=0;j<H;j++) if(hid[j]>0){ G->gb1[j]+=dh[j]; float* gw=&G->gW1[(size_t)j*in]; const float g=dh[j];
        for(int k=0;k<in;k++) gw[k]+=g*x[k]; }
}
static void train_mixed(MLP* m,const float* X0,const uint32_t* t0,const uint32_t* p0,long n,
                        const float* Xr,const uint32_t* tr,const uint32_t* pr,long npool,
                        int num,int den,int epochs,float lr){
    int H=m->h,in=m->in;
    size_t s1=(size_t)H*in, s2=(size_t)VTOK*H;
    Grads G; G.gW1=malloc(s1*4); G.gb1=malloc(H*4); G.gW2=malloc(s2*4); G.gb2=malloc((size_t)VTOK*4);
    float* hid=malloc(H*4); float* dh=malloc(H*4); float* lg=malloc((size_t)VTOK*4); float* eo=malloc((size_t)VTOK*4);
    int bs=512; float invnorm=1.0f/bs; long kc=0,kr=0;
    for(int ep=0;ep<epochs;ep++){
        memset(G.gW1,0,s1*4); memset(G.gb1,0,H*4); memset(G.gW2,0,s2*4); memset(G.gb2,0,(size_t)VTOK*4); long inb=0;
        for(long s=0;s<n;s++){
            int use_roll = (npool>0) && ((s%den)<num);
            if(use_roll){ long j=kr%npool; kr++; accum_sample(m,&Xr[(size_t)j*S_DIM],tr[j],pr[j],invnorm,&G,hid,dh,lg,eo); }
            else { long k=kc%n; kc++; accum_sample(m,&X0[(size_t)k*S_DIM],t0[k],p0[k],invnorm,&G,hid,dh,lg,eo); }
            inb++;
            if(inb==bs || s==n-1){ m->t++; float lt=lr*sqrtf(1-powf(.999f,m->t))/(1-powf(.9f,m->t));
                #define ADAM(P,GG,MM,VV,NN) for(size_t z=0;z<(size_t)(NN);z++){ MM[z]=.9f*MM[z]+.1f*GG[z]; VV[z]=.999f*VV[z]+.001f*GG[z]*GG[z]; P[z]-=lt*(MM[z]/(sqrtf(VV[z])+1e-8f)+1e-5f*P[z]); }
                ADAM(m->W1,G.gW1,m->mW1,m->vW1,s1); ADAM(m->b1,G.gb1,m->mb1,m->vb1,H);
                ADAM(m->W2,G.gW2,m->mW2,m->vW2,s2); ADAM(m->b2,G.gb2,m->mb2,m->vb2,VTOK);
                memset(G.gW1,0,s1*4); memset(G.gb1,0,H*4); memset(G.gW2,0,s2*4); memset(G.gb2,0,(size_t)VTOK*4); inb=0; }
        }
        fprintf(stderr,"    ep %d/%d done\n",ep+1,epochs);
    }
    free(G.gW1);free(G.gb1);free(G.gW2);free(G.gb2);free(hid);free(dh);free(lg);free(eo);
}
static void mlp_save(MLP* m,const char* path){
    FILE* f=fopen(path,"wb"); if(!f){ fprintf(stderr,"save %s failed\n",path); return; }
    uint32_t magic=0x53454554,H=(uint32_t)m->h,d=(uint32_t)m->in,base=D1_TOT,dexp=D_EXP,nts=N_TS;
    float gamma=GAMMA; uint64_t pseed=PROJ_SEED;
    fwrite(&magic,4,1,f); fwrite(&H,4,1,f); fwrite(&d,4,1,f); fwrite(&base,4,1,f);
    fwrite(&dexp,4,1,f); fwrite(&nts,4,1,f); fwrite(&gamma,4,1,f); fwrite(&pseed,8,1,f);
    for(int t=0;t<N_TS;t++){ float a=TS_ALPHA[t]; fwrite(&a,4,1,f); }
    uint32_t vt=(uint32_t)VTOK, nm=(uint32_t)g_bpe.nmerge; fwrite(&vt,4,1,f); fwrite(&nm,4,1,f);
    for(int r=0;r<g_bpe.nmerge;r++){ fwrite(&g_bpe.mA[r],4,1,f); fwrite(&g_bpe.mB[r],4,1,f); }
    fwrite(g_tbig,4,(size_t)VTOK*VTOK,f);
    fwrite(m->W1,4,(size_t)m->h*m->in,f); fwrite(m->b1,4,m->h,f);
    fwrite(m->W2,4,(size_t)VTOK*m->h,f); fwrite(m->b2,4,VTOK,f); fclose(f);
}

// feed one byte through the substrate (per-byte armB fold + observe + L2), advance ctx.
static inline void feed_byte_x(SiliconEntropyState* see, uint8_t ob, float* L2d1, float eB[N_TS][D_EXP],
                               float* pb_d1, uint8_t* cur_c1, uint8_t* cur_c2, float scale){
    float feat192[BASE_DIM], fa[BASE_DIM];
    see_extract(see,feat192); armb_fold(feat192,eB);
    see_observe(see,ob); see_extract(see,fa);
    if(ent_gate(*cur_c1,*cur_c2)){ float src5[BASE_DIM];
        for(int kk=0;kk<BASE_DIM;kk++) src5[kk]=fa[kk]-0.5f*pb_d1[kk]; memcpy(pb_d1,fa,BASE_DIM*4);
        for(int j=0;j<L2_DIM;j++){ float p5=0; const float* pj=Pmat[j]; for(int kk=0;kk<BASE_DIM;kk++) p5+=pj[kk]*src5[kk];
            L2d1[j]=g_alpha*L2d1[j]+(1.0f-g_alpha)*p5*scale; } }
    *cur_c2=*cur_c1; *cur_c1=ob;
}

static long extract_window_tok(SiliconEntropyState* see, long start, long N, double* oBYTES,
                               float* X, uint32_t* tgt, uint32_t* prev){
    float L2d1[L2_DIM]={0}, pb_d1[BASE_DIM]={0}, eB[N_TS][D_EXP]; memset(eB,0,sizeof eB);
    float feat192[BASE_DIM], rawd1[D1_TOT], nf[D1_TOT]; float scale=1.0f/sqrtf((float)BASE_DIM);
    long ti=0; { long lo=0,hi=g_ntok; while(lo<hi){ long mid=(lo+hi)/2; if(g_tokstart[mid]<start) lo=mid+1; else hi=mid; } ti=lo; }
    long bstart=g_tokstart[ti];
    see_reset(see); for(long i=0;i<bstart;i++) see_observe(see,g_data[i]);
    uint8_t cur_c2=(bstart>=2)?g_data[bstart-2]:0, cur_c1=(bstart>=1)?g_data[bstart-1]:0;
    long rows=0, wbytes=0;
    while(ti+1<g_ntok && g_tokstart[ti+1]+(long)bpe_tok_len(&g_bpe,g_tok[ti+1]) <= start+N){
        see_extract(see,feat192);
        memcpy(rawd1,feat192,BASE_DIM*4); memcpy(rawd1+BASE_DIM,L2d1,L2_DIM*4); norm_feats(rawd1,nf);
        float* row=&X[(size_t)rows*S_DIM]; memcpy(row,nf,D1_TOT*4); row_bands(eB,row);
        tgt[rows]=g_tok[ti+1]; prev[rows]=g_tok[ti]; rows++;
        int Lt=bpe_tok_len(&g_bpe,g_tok[ti+1]); const unsigned char* tb=bpe_tok_bytes(&g_bpe,g_tok[ti+1]); wbytes+=Lt;
        for(int k=0;k<Lt;k++) feed_byte_x(see,tb[k],L2d1,eB,pb_d1,&cur_c1,&cur_c2,scale);
        ti++;
    }
    *oBYTES=(double)wbytes; return rows;
}

// rollout with K16/K128 bursts + (repeat_mode) REPEAT bursts. row bigram-context = last EMITTED token.
static long extract_rollout_tok(SiliconEntropyState* see, MLP* m, long start, long N, int kfar, int repeat_mode,
                                uint64_t rseed, float* Xr, uint32_t* tr, uint32_t* pr, long npool_max,
                                double* o_self16, double* o_self128, double* o_selfRep){
    float L2d1[L2_DIM]={0}, pb_d1[BASE_DIM]={0}, eB[N_TS][D_EXP]; memset(eB,0,sizeof eB);
    float feat192[BASE_DIM], rawd1[D1_TOT], nf[D1_TOT], row[S_DIM];
    float* hid=malloc(m->h*4); float* lg=malloc((size_t)VTOK*4); float* Pp=malloc((size_t)VTOK*4);
    float scale=1.0f/sqrtf((float)BASE_DIM);
    double bits16=0,bits128=0,bitsR=0; long n16=0,n128=0,nR=0, np=0;
    uint64_t rng=rseed?rseed:0x9E3779B97F4A7C15ULL;
    int burst=0,cur_far=0, rep_seed=0,rep_recov=0; uint32_t rep_tok=0;
    long since=RECOV+1, bidx=0, step=0;
    long ti=0; { long lo=0,hi=g_ntok; while(lo<hi){ long mid=(lo+hi)/2; if(g_tokstart[mid]<start) lo=mid+1; else hi=mid; } ti=lo; }
    long bstart=g_tokstart[ti];
    see_reset(see); for(long i=0;i<bstart;i++) see_observe(see,g_data[i]);
    uint8_t cur_c2=(bstart>=2)?g_data[bstart-2]:0, cur_c1=(bstart>=1)?g_data[bstart-1]:0;
    uint32_t prev_emit=(ti>0)?g_tok[ti-1]:0;
    while(ti+1<g_ntok && g_tokstart[ti+1] <= start+N){
        see_extract(see,feat192);
        memcpy(rawd1,feat192,BASE_DIM*4); memcpy(rawd1+BASE_DIM,L2d1,L2_DIM*4); norm_feats(rawd1,nf);
        memcpy(row,nf,D1_TOT*4); row_bands(eB,row);
        uint32_t target=g_tok[ti+1];
        if(step>0 && (step%PERIOD)==0){
            int cr = repeat_mode && ((bidx%8)==3);
            cur_far = (kfar && (bidx%8)==7);
            if(cr){ rng^=rng<<13;rng^=rng>>7;rng^=rng<<17; rep_tok=g_freqtok[(int)((rng>>11)%(uint64_t)(g_nfreq>0?g_nfreq:1))];
                    rep_seed=K_REPSEED; rep_recov=K_REPRECOV; burst=0; }
            else burst = cur_far?K_FAR:K_NEAR;
            bidx++;
        }
        int in_rseed=(rep_seed>0), in_rrecov=(!in_rseed && rep_recov>0);
        int in_burst=(burst>0), in_recov=(!in_burst && !in_rseed && !in_rrecov && since<RECOV);
        if((in_burst||in_recov||in_rseed||in_rrecov) && np<npool_max){
            memcpy(&Xr[(size_t)np*S_DIM],row,S_DIM*4); tr[np]=target; pr[np]=prev_emit; np++;
        }
        uint32_t emit=target; float tempe=in_rseed||in_rrecov?T_REP:T_HI;
        if(in_rseed){ emit=rep_tok; rep_seed--; }
        else if(in_burst || in_rrecov){
            mlp_fwd(m,row,hid,lg); const float* tb=&g_tbig[(size_t)prev_emit*VTOK];
            float mx=-1e30f; for(int c=0;c<VTOK;c++){ lg[c]=(lg[c]+tb[c])/tempe; if(lg[c]>mx)mx=lg[c]; }
            float Z=0; for(int c=0;c<VTOK;c++){ Pp[c]=expf(lg[c]-mx); Z+=Pp[c]; } for(int c=0;c<VTOK;c++) Pp[c]/=Z;
            emit=sample_tok(Pp,&rng);
            double sb=-log2((double)fmaxf(Pp[emit],1e-30f));
            if(in_rrecov){ bitsR+=sb; nR++; rep_recov--; }
            else if(cur_far){ bits128+=sb; n128++; } else { bits16+=sb; n16++; }
            if(in_burst){ burst--; if(burst==0) since=0; }
        } else since++;
        int L=bpe_tok_len(&g_bpe,emit); const unsigned char* eb=bpe_tok_bytes(&g_bpe,emit);
        for(int k=0;k<L;k++) feed_byte_x(see,eb[k],L2d1,eB,pb_d1,&cur_c1,&cur_c2,scale);
        prev_emit=emit; ti++; step++;
    }
    free(hid);free(lg);free(Pp);
    *o_self16=(n16>0)?bits16/n16:0.0; *o_self128=(n128>0)?bits128/n128:0.0; *o_selfRep=(nR>0)?bitsR/nR:0.0;
    return np;
}

// PREMISE (no train): from a seeded state of one of the REAL topBi modes, does the decoder STAY in
// that pattern? mode 0=single (frequent token), 1=alt (A B A B), 2=dup (content-word duplication).
// reports seeded persistence vs a natural-state control of the SAME pattern.
enum { PM_SINGLE=0, PM_ALT=1, PM_DUP=2 };
static const char* PM_NAME[3]={"single","alt","dup"};
// run K_PROBE samples from the CURRENT (already-positioned) decoder state; expected[s] = the token
// that would continue the pattern. returns frac matching + longest matching run. feeds the SAMPLED
// token (on-policy). mutates the substrate -> caller must snapshot/restore.
static double probe_pattern(SiliconEntropyState* see, MLP* m, float* L2d1, float eB[N_TS][D_EXP], float* pb_d1,
                            uint8_t* c1, uint8_t* c2, uint32_t* prev_emit, float scale, uint64_t* rng,
                            const uint32_t* expected, int K_PROBE, double* o_maxrun){
    float feat192[BASE_DIM], rawd1[D1_TOT], nf[D1_TOT], row[S_DIM];
    float* hid=malloc(m->h*4); float* lg=malloc((size_t)VTOK*4); float* Pp=malloc((size_t)VTOK*4);
    int hit=0, run=0, maxrun=0;
    for(int s=0;s<K_PROBE;s++){
        see_extract(see,feat192); memcpy(rawd1,feat192,BASE_DIM*4); memcpy(rawd1+BASE_DIM,L2d1,L2_DIM*4);
        norm_feats(rawd1,nf); memcpy(row,nf,D1_TOT*4); row_bands(eB,row);
        mlp_fwd(m,row,hid,lg); const float* tb=&g_tbig[(size_t)*prev_emit*VTOK];
        float mx=-1e30f; for(int c=0;c<VTOK;c++){ lg[c]=(lg[c]+tb[c])/T_REP; if(lg[c]>mx)mx=lg[c]; }
        float Z=0; for(int c=0;c<VTOK;c++){ Pp[c]=expf(lg[c]-mx); Z+=Pp[c]; } for(int c=0;c<VTOK;c++) Pp[c]/=Z;
        uint32_t e=sample_tok(Pp,rng);
        if(e==expected[s]){ hit++; run++; if(run>maxrun)maxrun=run; } else run=0;
        int L=bpe_tok_len(&g_bpe,e); const unsigned char* eb=bpe_tok_bytes(&g_bpe,e);
        for(int k=0;k<L;k++) feed_byte_x(see,eb[k],L2d1,eB,pb_d1,c1,c2,scale); *prev_emit=e;
    }
    free(hid);free(lg);free(Pp);
    if(o_maxrun)*o_maxrun=maxrun;
    return (double)hit/K_PROBE;
}
static void premise_check(SiliconEntropyState* see, MLP* m, long start, long N, int mode){
    float L2d1[L2_DIM]={0}, pb_d1[BASE_DIM]={0}, eB[N_TS][D_EXP]; memset(eB,0,sizeof eB);
    float scale=1.0f/sqrtf((float)BASE_DIM); uint64_t rng=0x50B09E3779B97F4AULL+(uint64_t)mode*0x1000;
    long ti=0; { long lo=0,hi=g_ntok; while(lo<hi){ long mid=(lo+hi)/2; if(g_tokstart[mid]<start) lo=mid+1; else hi=mid; } ti=lo; }
    long bstart=g_tokstart[ti];
    see_reset(see); for(long i=0;i<bstart;i++) see_observe(see,g_data[i]);
    uint8_t cur_c2=(bstart>=2)?g_data[bstart-2]:0, cur_c1=(bstart>=1)?g_data[bstart-1]:0;
    uint32_t prev_emit=(ti>0)?g_tok[ti-1]:0;
    const int K_PROBE=20; long step=0, nprobe=0; double seeded_persist=0, natural_persist=0, seeded_run=0;
    int fi=0;
    uint32_t expected[64];
    while(ti+1<g_ntok && g_tokstart[ti+1] <= start+N && nprobe<300){
        if(step>0 && (step%PERIOD)==0){
            // choose the pattern + the expected continuation for this mode
            uint32_t A=0,B=0;
            if(mode==PM_SINGLE){ A=B=(g_nfreq>0)?g_freqtok[fi%g_nfreq]:32; }
            else if(mode==PM_DUP){ A=B=(g_ncontent>0)?g_contenttok[fi%g_ncontent]:32; }
            else { int p=(g_naltp>0)?fi%g_naltp:0; A=g_altpairs[p][0]; B=g_altpairs[p][1]; }
            fi++;
            // expected continuation after the seed (seed ends on the 2nd element for alt, on A for single/dup)
            for(int s=0;s<K_PROBE;s++) expected[s] = (mode==PM_ALT)? ((s%2==0)?A:B) : A;
            // snapshot
            SiliconEntropyState snap=*see; float sL2[L2_DIM]; memcpy(sL2,L2d1,sizeof sL2);
            float sEB[N_TS][D_EXP]; memcpy(sEB,eB,sizeof sEB); float sPB[BASE_DIM]; memcpy(sPB,pb_d1,sizeof sPB);
            uint8_t sc1=cur_c1,sc2=cur_c2; uint32_t spe=prev_emit;
            // --- seeded: force the pattern ---
            for(int s=0;s<K_REPSEED;s++){
                uint32_t t0 = (mode==PM_ALT)? A : A; int L0=bpe_tok_len(&g_bpe,t0); const unsigned char* e0=bpe_tok_bytes(&g_bpe,t0);
                for(int k=0;k<L0;k++) feed_byte_x(see,e0[k],L2d1,eB,pb_d1,&cur_c1,&cur_c2,scale); prev_emit=t0;
                if(mode==PM_ALT){ int L1=bpe_tok_len(&g_bpe,B); const unsigned char* e1=bpe_tok_bytes(&g_bpe,B);
                    for(int k=0;k<L1;k++) feed_byte_x(see,e1[k],L2d1,eB,pb_d1,&cur_c1,&cur_c2,scale); prev_emit=B; }
            }
            double mr=0; double sp=probe_pattern(see,m,L2d1,eB,pb_d1,&cur_c1,&cur_c2,&prev_emit,scale,&rng,expected,K_PROBE,&mr);
            seeded_persist += sp; seeded_run += mr;
            // restore
            *see=snap; memcpy(L2d1,sL2,sizeof sL2); memcpy(eB,sEB,sizeof sEB); memcpy(pb_d1,sPB,sizeof sPB); cur_c1=sc1; cur_c2=sc2; prev_emit=spe;
            // --- natural control: same expected pattern, no seed ---
            double np=probe_pattern(see,m,L2d1,eB,pb_d1,&cur_c1,&cur_c2,&prev_emit,scale,&rng,expected,K_PROBE,NULL);
            natural_persist += np;
            *see=snap; memcpy(L2d1,sL2,sizeof sL2); memcpy(eB,sEB,sizeof sEB); memcpy(pb_d1,sPB,sizeof sPB); cur_c1=sc1; cur_c2=sc2; prev_emit=spe;
            nprobe++;
        }
        int L=bpe_tok_len(&g_bpe,g_tok[ti+1]); const unsigned char* eb=bpe_tok_bytes(&g_bpe,g_tok[ti+1]);
        for(int k=0;k<L;k++) feed_byte_x(see,eb[k],L2d1,eB,pb_d1,&cur_c1,&cur_c2,scale);
        prev_emit=g_tok[ti+1]; ti++; step++;
    }
    double sp=(nprobe>0)?seeded_persist/nprobe:0, np2=(nprobe>0)?natural_persist/nprobe:0, sr=(nprobe>0)?seeded_run/nprobe:0;
    printf("PREMISE mode=%s probes=%ld seeded_persist=%.3f natural_persist=%.3f seeded_maxrun=%.2f ratio=%.2f\n",
           PM_NAME[mode],nprobe,sp,np2,sr,(np2>1e-6)?sp/np2:999.0);
}

typedef struct { float* X; uint32_t *tgt,*prev; double bytes; long rows; } Win;
static Win alloc_win(long maxrows){ Win w; w.X=malloc((size_t)maxrows*S_DIM*4); w.tgt=malloc((size_t)maxrows*4);
    w.prev=malloc((size_t)maxrows*4); w.bytes=0; w.rows=0; if(!w.X||!w.tgt){ fprintf(stderr,"OOM win\n"); exit(1);} return w; }

int main(int argc, char** argv){
    if(argc<5){ fprintf(stderr,"Usage: %s <data> <D1_w> <bpe_merges> <outprefix> [--len N] [--resume-from ckpt] [--premise ckpt] [--max-bytes N]\n",argv[0]); return 1; }
    setvbuf(stderr,NULL,_IONBF,0); setvbuf(stdout,NULL,_IONBF,0);
    long N=1000000, maxb=0; const char* resume=0; const char* premise=0;
    for(int i=5;i<argc;i++){ if(!strcmp(argv[i],"--len")&&i+1<argc)N=atol(argv[++i]);
        else if(!strcmp(argv[i],"--max-bytes")&&i+1<argc)maxb=atol(argv[++i]);
        else if(!strcmp(argv[i],"--resume-from")&&i+1<argc)resume=argv[++i];
        else if(!strcmp(argv[i],"--premise")&&i+1<argc)premise=argv[++i]; }
    FILE* fd=fopen(argv[1],"rb"); if(!fd){fprintf(stderr,"data\n");return 1;} fseek(fd,0,SEEK_END); g_fsz=ftell(fd); fseek(fd,0,SEEK_SET);
    if(maxb>0 && maxb<g_fsz) g_fsz=maxb;
    g_data=malloc(g_fsz); fread(g_data,1,g_fsz,fd); fclose(fd);

    SiliconEntropyState see;
    if(!load_d1(argv[2],&see)) return 1;
    if(!bpe_load_file(&g_bpe,argv[3])) return 1; VTOK=g_bpe.vocab;
    gen_lift(PROJ_SEED);
    ent_table=malloc(CLASSES*CLASSES*4);
    for(int i=0;i<CLASSES;i++) for(int j=0;j<CLASSES;j++){ float m=-1e9f; for(int k=0;k<CLASSES;k++) if(trigram[i][j][k]>m)m=trigram[i][j][k];
        double se=0; for(int k=0;k<CLASSES;k++) se+=exp((double)(trigram[i][j][k]-m));
        double Hh=0; for(int k=0;k<CLASSES;k++){ double p=exp((double)(trigram[i][j][k]-m))/se; if(p>1e-12)Hh-=p*log(p); }
        ent_table[i][j]=(float)Hh; }

    tokenize_corpus();
    long tok_train_end=(long)(g_ntok*0.90);
    fprintf(stderr,"building token-bigram prior + freq tokens (train tokens %ld)...\n",tok_train_end);
    build_tbig(tok_train_end);

    long tr_start=g_fsz/5;
    long va_start[N_VAL]={ g_fsz/2, (long)(0.65*g_fsz), (long)(0.80*g_fsz) };
    for(int w=0;w<N_VAL;w++) if(va_start[w]+N+3>g_fsz){ fprintf(stderr,"val window %d out of file\n",w+1); return 1; }

    // ---- PREMISE mode: load decoder, probe, exit ----
    if(premise){
        MLP mp; mlp_init(&mp,S_DIM,32); if(!mlp_load(&mp,premise)){ fprintf(stderr,"premise load fail\n"); return 1; }
        fprintf(stderr,"PREMISE check on %s (train window @%ld) - all 3 modes\n",premise,tr_start);
        printf("PREMISE ckpt=%s\n",premise);
        for(int mode=0;mode<3;mode++) premise_check(&see,&mp,tr_start,N,mode);
        printf("PREMISE read: per mode, seeded_persist >> natural (ratio>=2 AND seeded>=0.30) => that mode\n");
        printf("              is a coverable attractor. seeded ~= natural => NOT coverable (charter-question).\n");
        mlp_free(&mp); return 0;
    }

    fprintf(stderr,"50.B token+repeat-cov: train @%ld val @%ld/%ld/%ld N=%ld VTOK=%d resume=%s\n",
            tr_start,va_start[0],va_start[1],va_start[2],N,VTOK,resume?resume:"(none)");
    long maxrows=N+16;
    Win tr0=alloc_win(maxrows); Win va[N_VAL]; for(int w=0;w<N_VAL;w++) va[w]=alloc_win(maxrows);
    fprintf(stderr,"extract tr_clean...\n");
    tr0.rows=extract_window_tok(&see,tr_start,N,&tr0.bytes,tr0.X,tr0.tgt,tr0.prev);
    for(int w=0;w<N_VAL;w++){ fprintf(stderr,"extract val%d...\n",w+1);
        va[w].rows=extract_window_tok(&see,va_start[w],N,&va[w].bytes,va[w].X,va[w].tgt,va[w].prev); }

    printf("\n==== 50.B token + REPEAT coverage (H32, N=%ld bytes, VTOK=%d) ====\n",N,VTOK);
    printf("BASE win=train rows=%ld bytes=%.0f\n",tr0.rows,tr0.bytes);
    for(int w=0;w<N_VAL;w++) printf("BASE win=val%d rows=%ld bytes=%.0f\n",w+1,va[w].rows,va[w].bytes);

    long npool_max=N*(K_FAR+RECOV)/PERIOD + 64;
    float* Xr=malloc((size_t)npool_max*S_DIM*4);
    uint32_t* trr=malloc((size_t)npool_max*4); uint32_t* prr=malloc((size_t)npool_max*4);
    if(!Xr||!trr){ fprintf(stderr,"OOM pool\n"); return 1; }

    MLP mp; mlp_init(&mp,S_DIM,32);
    int start_rd=1;
    if(resume){ if(!mlp_load(&mp,resume)){ fprintf(stderr,"resume load fail\n"); return 1; }
        start_rd=R_PRE+1; mp.t=0; fprintf(stderr,"resumed from %s -> starting round %d (REPEAT coverage on)\n",resume,start_rd); }
    else { char sp0[512]; fprintf(stderr,"   pretrain clean %dep\n",EP_PRE);
        train_mixed(&mp,tr0.X,tr0.tgt,tr0.prev,tr0.rows, NULL,NULL,NULL,0, 0,1,EP_PRE,0.0005f); (void)sp0; }

    char sp[512];
    for(int rd=start_rd;rd<=R_TOT;rd++){
        int far=(rd>R_PRE); int repmode=(rd>R_PRE);   // REPEAT coverage in rounds 6-9
        double s16=0,s128=0,sRep=0;
        uint64_t seed = 0x50B70000ULL^((uint64_t)rd*0x9E37ULL)^16ULL ^ (far?0x11400000ULL:0ULL);
        fprintf(stderr,"   round %d (%s%s): rollout...\n",rd,far?"K16/K128":"K16",repmode?"+REPEAT":"");
        long npool=extract_rollout_tok(&see,&mp,tr_start,N,far,repmode,seed,Xr,trr,prr,npool_max,&s16,&s128,&sRep);
        fprintf(stderr,"   round %d: pool=%ld rollK16=%.3f rollK128=%.3f rollRep=%.3f; train %dep\n",rd,npool,s16,s128,sRep,EP_MIX);
        train_mixed(&mp,tr0.X,tr0.tgt,tr0.prev,tr0.rows, Xr,trr,prr,npool, 1,5,EP_MIX,0.0005f);
        double v[N_VAL],bpu[N_VAL];
        for(int w=0;w<N_VAL;w++) v[w]=mlp_eval(&mp,va[w].X,va[w].tgt,va[w].prev,va[w].rows,va[w].bytes,&bpu[w]);
        snprintf(sp,sizeof sp,"%s_I_h32_r%d.bin",argv[4],rd); mlp_save(&mp,sp);
        printf("PROBE name=I_r%d h=32 rollK16=%.3f rollK128=%.3f rollRep=%.3f val1=%.4f val2=%.4f val3=%.4f bpu1=%.3f\n",
               rd,s16,s128,sRep,v[0],v[1],v[2],bpu[0]);
        printf("SAVED name=I_r%d path=%s\n",rd,sp);
    }
    mlp_free(&mp);
    printf("\nReading: rollRep = self-BPB of the decoder recovery after a forced token-repeat (should fall\n");
    printf("with rounds = learning to exit). val* = bits/byte. Gate is closed-loop: byte-guards stay clean\n");
    printf("+ topBi UNDER bar + BPB < bar + human read. TF != generative: no celebration pre-gate.\n");
    return 0;
}
