// Phase 50.A - BPE-1024 token readout over the BYTE-DRIVEN armB substrate.
//
// 50.0 picked BPE-1024 (bpb-o2 << byte, repeat/flood dissolved). 50.A wires the unit into the
// readout WITHOUT touching the substrate: armB stays byte-driven and INVARIATO (load_d1 + the
// same Omega/Bvec lift + per-byte L2 EMA). At every TOKEN boundary the readout reads phi_armB(s)
// and predicts the NEXT token (1024-way softmax, H32), with a frozen token-BIGRAM prior added to
// the logits (the direct analog of the byte-trigram prior in the 47/48 byte readout). Training =
// the 47.I-final DAgger recipe, lifted to tokens: TF clean rounds + rollout bursts where the
// model EMITS a sampled token, feeds ITS bytes to the substrate one by one (on-policy), and the
// target is always the TRUE next corpus token. Generation: predict token dist -> sample -> emit
// the token's bytes -> substrate advances -> next boundary.
//
// The byte-guards (chRun/wsRun/...) are exactly the channels BPE should make structurally
// impossible; BPB is reported in bits-per-BYTE (unit-invariant). Checkpoint 0x53454554 embeds the
// BPE merges + token-bigram + armB header so the generator reconstructs the whole pipeline.
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase38-42/phase50_a_token.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase50_a_token.exe -lm -I .
// Run:
//   bin/phase50_a_token.exe <data> <D1_w> <bpe_merges> <outprefix> [--len N]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include "src/silicon_entropy.h"
#include "benchmarks/phase38-42/bpe_codec.h"

#define CLASSES   256                    // byte classes (substrate/anchor only)
#define BASE_DIM  SEE_FEATURE_DIM        // 192
#define L0_DIM    SEE_L0_DIM             // 64
#define L2_DIM    64
#define D1_TOT    (BASE_DIM + L2_DIM)    // 256
#define D_EXP     128
#define GAMMA     0.25f
#define N_TS      2
#define EXP_BANDS (N_TS*D_EXP)           // 256
#define S_DIM     (D1_TOT + EXP_BANDS)   // 512
#define PROJ_SEED 0x48B2EC0DEULL
#define N_VAL     3
#define T_HI      0.65f
#define RECOV     16
#define EP_PRE    2
#define EP_MIX    2
#define R_PRE     5
#define R_TOT     9
#define K_NEAR    16
#define K_FAR     128
#define PERIOD    256                    // bursts every PERIOD tokens

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

// tokenized corpus
static Bpe g_bpe;
static uint32_t* g_tok; static long* g_tokstart; static long g_ntok;
static int VTOK=1024;
static float* g_tbig;     // [VTOK*VTOK] additive log-prob prior (bigram)

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
// armB band fold (per byte): update eB from current feat192. (No row write.)
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

// ---- tokenize corpus + build token-bigram prior ----
static void tokenize_corpus(void){
    g_tok=(uint32_t*)malloc((size_t)g_fsz*sizeof(uint32_t));
    g_tokstart=(long*)malloc((size_t)(g_fsz+1)*sizeof(long));
    g_ntok=(long)bpe_encode_region(&g_bpe, g_data, 0, g_fsz, g_tok);
    long off=0; for(long i=0;i<g_ntok;i++){ g_tokstart[i]=off; off+=bpe_tok_len(&g_bpe,g_tok[i]); }
    g_tokstart[g_ntok]=off;
    fprintf(stderr,"tokenized: %ld tokens over %ld bytes (%.3f b/tok)\n",g_ntok,g_fsz,(double)g_fsz/g_ntok);
}
static void build_tbig(long tok_train_end){
    // bigram counts over train tokens [0,tok_train_end); additive log-prob (Laplace k=0.5), backoff
    // to smoothed unigram for unseen context. Stored as additive logit prior.
    g_tbig=(float*)malloc((size_t)VTOK*VTOK*sizeof(float));
    double* uni=(double*)calloc(VTOK,sizeof(double));
    double* ctx=(double*)calloc(VTOK,sizeof(double));   // ctx[a]=sum_b count(a,b)
    uint32_t* bg=(uint32_t*)calloc((size_t)VTOK*VTOK,sizeof(uint32_t));
    for(long i=0;i<tok_train_end;i++) uni[g_tok[i]]+=1.0;
    for(long i=0;i+1<tok_train_end;i++){ uint32_t a=g_tok[i],b=g_tok[i+1]; bg[(size_t)a*VTOK+b]++; ctx[a]+=1.0; }
    double T=(double)tok_train_end; const double K=0.5;
    for(int a=0;a<VTOK;a++){
        double ca=ctx[a];
        for(int b=0;b<VTOK;b++){
            double pb=(uni[b]+K)/(T+K*VTOK);                       // smoothed unigram
            double p;
            if(ca>0){ double cab=bg[(size_t)a*VTOK+b]; p=(cab+K*VTOK*pb)/(ca+K*VTOK); }
            else p=pb;
            g_tbig[(size_t)a*VTOK+b]=(float)log((p>1e-30)?p:1e-30);
        }
    }
    free(uni); free(ctx); free(bg);
}

// ---- MLP H32 -> VTOK with token-bigram prior ----
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
// eval: bits per BYTE over a window (token CE / window bytes). prev = bigram context per row.
static double mlp_eval(MLP* m,const float* X,const uint32_t* tgt,const uint32_t* prev,long n,double winbytes,
                       double* o_bpu){
    float* hid=malloc(m->h*4); float* lg=malloc((size_t)VTOK*4); double tot=0;
    for(long i=0;i<n;i++){
        mlp_fwd(m,&X[(size_t)i*S_DIM],hid,lg);
        const float* tb=&g_tbig[(size_t)prev[i]*VTOK]; for(int c=0;c<VTOK;c++) lg[c]+=tb[c];
        float mx=-1e30f; for(int c=0;c<VTOK;c++) if(lg[c]>mx)mx=lg[c];
        double Z=0; for(int c=0;c<VTOK;c++) Z+=exp((double)(lg[c]-mx));
        double p=exp((double)(lg[tgt[i]]-mx))/Z; tot+=-log2(p>1e-30?p:1e-30);
    }
    free(hid); free(lg);
    if(o_bpu)*o_bpu=tot/n;
    return tot/winbytes;     // bits per BYTE
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
    float* hid=malloc(H*4); float* dh=malloc(H*4);
    float* lg=malloc((size_t)VTOK*4); float* eo=malloc((size_t)VTOK*4);
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
// save 0x53454554: armB header + VTOK + merges + tbig + MLP weights
static void mlp_save(MLP* m,const char* path){
    FILE* f=fopen(path,"wb"); if(!f){ fprintf(stderr,"save %s failed\n",path); return; }
    uint32_t magic=0x53454554,H=(uint32_t)m->h,d=(uint32_t)m->in,base=D1_TOT,dexp=D_EXP,nts=N_TS;
    float gamma=GAMMA; uint64_t pseed=PROJ_SEED;
    fwrite(&magic,4,1,f); fwrite(&H,4,1,f); fwrite(&d,4,1,f); fwrite(&base,4,1,f);
    fwrite(&dexp,4,1,f); fwrite(&nts,4,1,f); fwrite(&gamma,4,1,f); fwrite(&pseed,8,1,f);
    for(int t=0;t<N_TS;t++){ float a=TS_ALPHA[t]; fwrite(&a,4,1,f); }
    uint32_t vt=(uint32_t)VTOK, nm=(uint32_t)g_bpe.nmerge;
    fwrite(&vt,4,1,f); fwrite(&nm,4,1,f);
    for(int r=0;r<g_bpe.nmerge;r++){ fwrite(&g_bpe.mA[r],4,1,f); fwrite(&g_bpe.mB[r],4,1,f); }
    fwrite(g_tbig,4,(size_t)VTOK*VTOK,f);
    fwrite(m->W1,4,(size_t)m->h*m->in,f); fwrite(m->b1,4,m->h,f);
    fwrite(m->W2,4,(size_t)VTOK*m->h,f); fwrite(m->b2,4,VTOK,f); fclose(f);
}

// ---------- token-stepped extraction over byte window [start, start+N) ----------
// fires the readout at each token boundary fully inside the window. mode: CLEAN feeds true token
// bytes; CORRUPT samples a token from the CURRENT model (mp!=NULL) for valC continuity.
static long extract_window_tok(SiliconEntropyState* see, long start, long N, MLP* mp, float pcorr, uint64_t rseed,
                               float* X, uint32_t* tgt, uint32_t* prev, double* oTRI, double* oBYTES){
    float L2d1[L2_DIM]={0}, pb_d1[BASE_DIM]={0}, eB[N_TS][D_EXP]; memset(eB,0,sizeof eB);
    float feat192[BASE_DIM], fa[BASE_DIM], rawd1[D1_TOT], nf[D1_TOT];
    float* hid=mp?malloc(mp->h*4):NULL; float* lg=mp?malloc((size_t)VTOK*4):NULL; float* Pp=mp?malloc((size_t)VTOK*4):NULL;
    float scale=1.0f/sqrtf((float)BASE_DIM); double btr=0; uint64_t rng=rseed?rseed:0x9E3779B97F4A7C15ULL;
    // first token fully inside window
    long ti=0; { long lo=0,hi=g_ntok; while(lo<hi){ long mid=(lo+hi)/2; if(g_tokstart[mid]<start) lo=mid+1; else hi=mid; } ti=lo; }
    long bstart=g_tokstart[ti];
    see_reset(see); for(long i=0;i<bstart;i++) see_observe(see,g_data[i]);
    uint8_t cur_c2=(bstart>=2)?g_data[bstart-2]:0, cur_c1=(bstart>=1)?g_data[bstart-1]:0;
    long rows=0; long wbytes=0;
    while(ti+1<g_ntok && g_tokstart[ti+1]+ (long)bpe_tok_len(&g_bpe,g_tok[ti+1]) <= start+N){
        // boundary: predict token ti+1 with context ti
        see_extract(see,feat192);
        memcpy(rawd1,feat192,BASE_DIM*4); memcpy(rawd1+BASE_DIM,L2d1,L2_DIM*4);
        norm_feats(rawd1,nf);
        float* row=&X[(size_t)rows*S_DIM]; memcpy(row,nf,D1_TOT*4); row_bands(eB,row);
        uint32_t target=g_tok[ti+1], ctx=g_tok[ti];
        tgt[rows]=target; prev[rows]=ctx; rows++;
        // bits proxy (byte trigram over the token's bytes) for TRI baseline
        // choose emitted token
        uint32_t emit=target;
        if(mp){ rng^=rng<<13;rng^=rng>>7;rng^=rng<<17; double u=(rng>>11)*(1.0/(1ULL<<53));
            if(u<pcorr){ mlp_fwd(mp,row,hid,lg); const float* tb=&g_tbig[(size_t)ctx*VTOK];
                float mxx=-1e30f; for(int c=0;c<VTOK;c++){ lg[c]=(lg[c]+tb[c])/T_HI; if(lg[c]>mxx)mxx=lg[c]; }
                float Z=0; for(int c=0;c<VTOK;c++){ Pp[c]=expf(lg[c]-mxx); Z+=Pp[c]; } for(int c=0;c<VTOK;c++) Pp[c]/=Z;
                emit=sample_tok(Pp,&rng); } }
        // feed emitted token's bytes (per byte: armB fold, observe, L2 EMA)
        int L=bpe_tok_len(&g_bpe,emit); const unsigned char* eb=bpe_tok_bytes(&g_bpe,emit);
        // TRI baseline: byte trigram bits over the TRUE token's bytes
        { int Lt=bpe_tok_len(&g_bpe,target); const unsigned char* tb2=bpe_tok_bytes(&g_bpe,target);
          uint8_t a=cur_c1,b=cur_c2; for(int k=0;k<Lt;k++){ btr+=trigram_bpb(a,b,tb2[k]); b=a; a=tb2[k]; } wbytes+=Lt; }
        for(int k=0;k<L;k++){ uint8_t ob=eb[k];
            see_extract(see,feat192); armb_fold(feat192,eB);
            see_observe(see,ob); see_extract(see,fa);
            if(ent_gate(cur_c1,cur_c2)){ float src5[BASE_DIM];
                for(int kk=0;kk<BASE_DIM;kk++) src5[kk]=fa[kk]-0.5f*pb_d1[kk]; memcpy(pb_d1,fa,BASE_DIM*4);
                for(int j=0;j<L2_DIM;j++){ float p5=0; const float* pj=Pmat[j]; for(int kk=0;kk<BASE_DIM;kk++) p5+=pj[kk]*src5[kk];
                    L2d1[j]=g_alpha*L2d1[j]+(1.0f-g_alpha)*p5*scale; } }
            cur_c2=cur_c1; cur_c1=ob; }
        ti++;
    }
    if(hid){free(hid);free(lg);free(Pp);}
    *oTRI=(wbytes>0)?btr/wbytes:0.0; *oBYTES=(double)wbytes;
    return rows;
}

// rollout: token bursts (per-burst K via bidx%8==7 far). emits sampled tokens on-policy, feeds
// their bytes, target = true next token. self-bits/token per K subset.
static long extract_rollout_tok(SiliconEntropyState* see, MLP* m, long start, long N, int kfar, uint64_t rseed,
                                float* Xr, uint32_t* tr, uint32_t* pr, long* posr, long npool_max,
                                double* o_self16, double* o_self128){
    float L2d1[L2_DIM]={0}, pb_d1[BASE_DIM]={0}, eB[N_TS][D_EXP]; memset(eB,0,sizeof eB);
    float feat192[BASE_DIM], fa[BASE_DIM], rawd1[D1_TOT], nf[D1_TOT], row[S_DIM];
    float* hid=malloc(m->h*4); float* lg=malloc((size_t)VTOK*4); float* Pp=malloc((size_t)VTOK*4);
    float scale=1.0f/sqrtf((float)BASE_DIM);
    double bits16=0,bits128=0; long n16=0,n128=0, np=0;
    uint64_t rng=rseed?rseed:0x9E3779B97F4A7C15ULL; int burst=0,cur_far=0; long since=RECOV+1, bidx=0, step=0;
    long ti=0; { long lo=0,hi=g_ntok; while(lo<hi){ long mid=(lo+hi)/2; if(g_tokstart[mid]<start) lo=mid+1; else hi=mid; } ti=lo; }
    long bstart=g_tokstart[ti];
    see_reset(see); for(long i=0;i<bstart;i++) see_observe(see,g_data[i]);
    uint8_t cur_c2=(bstart>=2)?g_data[bstart-2]:0, cur_c1=(bstart>=1)?g_data[bstart-1]:0;
    while(ti+1<g_ntok && g_tokstart[ti+1] <= start+N){
        see_extract(see,feat192);
        memcpy(rawd1,feat192,BASE_DIM*4); memcpy(rawd1+BASE_DIM,L2d1,L2_DIM*4);
        norm_feats(rawd1,nf); memcpy(row,nf,D1_TOT*4); row_bands(eB,row);
        uint32_t target=g_tok[ti+1], ctx=g_tok[ti];
        if(step>0 && (step%PERIOD)==0){ cur_far=(kfar && (bidx%8)==7); burst=cur_far?K_FAR:K_NEAR; bidx++; }
        int in_burst=(burst>0), in_recov=(!in_burst && since<RECOV);
        if((in_burst||in_recov) && np<npool_max){
            memcpy(&Xr[(size_t)np*S_DIM],row,S_DIM*4); tr[np]=target; pr[np]=ctx; posr[np]=-1; np++;
        }
        uint32_t emit=target;
        if(in_burst){
            mlp_fwd(m,row,hid,lg); const float* tb=&g_tbig[(size_t)ctx*VTOK];
            float mx=-1e30f; for(int c=0;c<VTOK;c++){ lg[c]=(lg[c]+tb[c])/T_HI; if(lg[c]>mx)mx=lg[c]; }
            float Z=0; for(int c=0;c<VTOK;c++){ Pp[c]=expf(lg[c]-mx); Z+=Pp[c]; } for(int c=0;c<VTOK;c++) Pp[c]/=Z;
            emit=sample_tok(Pp,&rng);
            double sb=-log2((double)fmaxf(Pp[emit],1e-30f));
            if(cur_far){ bits128+=sb; n128++; } else { bits16+=sb; n16++; }
            burst--; if(burst==0) since=0;
        } else since++;
        int L=bpe_tok_len(&g_bpe,emit); const unsigned char* eb=bpe_tok_bytes(&g_bpe,emit);
        for(int k=0;k<L;k++){ uint8_t ob=eb[k];
            see_extract(see,feat192); armb_fold(feat192,eB);
            see_observe(see,ob); see_extract(see,fa);
            if(ent_gate(cur_c1,cur_c2)){ float src5[BASE_DIM];
                for(int kk=0;kk<BASE_DIM;kk++) src5[kk]=fa[kk]-0.5f*pb_d1[kk]; memcpy(pb_d1,fa,BASE_DIM*4);
                for(int j=0;j<L2_DIM;j++){ float p5=0; const float* pj=Pmat[j]; for(int kk=0;kk<BASE_DIM;kk++) p5+=pj[kk]*src5[kk];
                    L2d1[j]=g_alpha*L2d1[j]+(1.0f-g_alpha)*p5*scale; } }
            cur_c2=cur_c1; cur_c1=ob; }
        ti++; step++;
    }
    free(hid);free(lg);free(Pp);
    *o_self16=(n16>0)?bits16/n16:0.0; *o_self128=(n128>0)?bits128/n128:0.0;
    return np;
}

typedef struct { float* X; uint32_t *tgt,*prev; double tri,bytes; long start,rows; } Win;
static Win alloc_win(long maxrows,long start){
    Win w; w.X=malloc((size_t)maxrows*S_DIM*4); w.tgt=malloc((size_t)maxrows*4); w.prev=malloc((size_t)maxrows*4);
    w.start=start; w.tri=0; w.bytes=0; w.rows=0; if(!w.X||!w.tgt){ fprintf(stderr,"OOM win\n"); exit(1); } return w;
}

int main(int argc, char** argv){
    if(argc<5){ fprintf(stderr,"Usage: %s <data> <D1_w> <bpe_merges> <outprefix> [--len N]\n",argv[0]); return 1; }
    setvbuf(stderr,NULL,_IONBF,0); setvbuf(stdout,NULL,_IONBF,0);
    long N=1000000, maxb=0;
    for(int i=5;i<argc;i++){ if(!strcmp(argv[i],"--len")&&i+1<argc)N=atol(argv[++i]);
        else if(!strcmp(argv[i],"--max-bytes")&&i+1<argc)maxb=atol(argv[++i]); }
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
    fprintf(stderr,"building token-bigram prior (train tokens %ld)...\n",tok_train_end);
    build_tbig(tok_train_end);

    long tr_start=g_fsz/5;
    long va_start[N_VAL]={ g_fsz/2, (long)(0.65*g_fsz), (long)(0.80*g_fsz) };
    for(int w=0;w<N_VAL;w++) if(va_start[w]+N+3>g_fsz){ fprintf(stderr,"val window %d out of file\n",w+1); return 1; }
    fprintf(stderr,"50.A token: train @%ld, val @%ld/%ld/%ld (N=%ld bytes) S_DIM=%d VTOK=%d\n",
            tr_start,va_start[0],va_start[1],va_start[2],N,S_DIM,VTOK);

    long maxrows=N+16;     // <= N tokens
    Win tr0=alloc_win(maxrows,tr_start);
    Win va[N_VAL]; for(int w=0;w<N_VAL;w++) va[w]=alloc_win(maxrows,va_start[w]);
    Win vaC=alloc_win(maxrows,va_start[0]);
    fprintf(stderr,"extract tr_clean...\n");
    tr0.rows=extract_window_tok(&see,tr_start,N,NULL,0,0, tr0.X,tr0.tgt,tr0.prev,&tr0.tri,&tr0.bytes);
    for(int w=0;w<N_VAL;w++){ fprintf(stderr,"extract val%d...\n",w+1);
        va[w].rows=extract_window_tok(&see,va_start[w],N,NULL,0,0, va[w].X,va[w].tgt,va[w].prev,&va[w].tri,&va[w].bytes); }

    printf("\n==== 50.A BPE-1024 token readout (H32, N=%ld bytes, VTOK=%d, prefix r1-%d + mix %d-%d) ====\n",N,VTOK,R_PRE,R_PRE+1,R_TOT);
    printf("BASE win=train rows=%ld bytes=%.0f byteTRI=%.4f\n",tr0.rows,tr0.bytes,tr0.tri);
    for(int w=0;w<N_VAL;w++) printf("BASE win=val%d rows=%ld bytes=%.0f byteTRI=%.4f\n",w+1,va[w].rows,va[w].bytes,va[w].tri);

    // anchor: token-bigram-only bits/byte (no MLP) on val1 -> a fixed reference the generator
    // must reproduce; also proves tokenize+tbig path is aligned. (Not the byte frozenD1 anchor;
    // this head is token-level.)
    { double tot=0; for(long i=0;i<va[0].rows;i++){ const float* tb=&g_tbig[(size_t)va[0].prev[i]*VTOK];
        float mx=-1e30f; for(int c=0;c<VTOK;c++) if(tb[c]>mx)mx=tb[c];
        double Z=0; for(int c=0;c<VTOK;c++) Z+=exp((double)(tb[c]-mx));
        double p=exp((double)(tb[va[0].tgt[i]]-mx))/Z; tot+=-log2(p>1e-30?p:1e-30); }
      printf("PROBE name=tbigOnly h=0 val1bpb=%.4f (token-bigram prior, bits/byte)\n",tot/va[0].bytes); }

    long npool_max=N*(K_FAR+RECOV)/PERIOD + 64;
    float* Xr=malloc((size_t)npool_max*S_DIM*4);
    uint32_t* trr=malloc((size_t)npool_max*4); uint32_t* prr=malloc((size_t)npool_max*4);
    long* posr=malloc(npool_max*sizeof(long));
    if(!Xr||!posr){ fprintf(stderr,"OOM pool\n"); return 1; }

    MLP mp; mlp_init(&mp,S_DIM,32);
    char sp[512];
    fprintf(stderr,"   pretrain clean %dep\n",EP_PRE);
    train_mixed(&mp,tr0.X,tr0.tgt,tr0.prev,tr0.rows, NULL,NULL,NULL,0, 0,1,EP_PRE,0.0005f);
    for(int rd=1;rd<=R_TOT;rd++){
        int far=(rd>R_PRE); double s16=0,s128=0;
        uint64_t seed = 0x50A70000ULL^((uint64_t)rd*0x9E37ULL)^16ULL ^ (far?0x11400000ULL:0ULL);
        fprintf(stderr,"   round %d (%s): rollout...\n",rd,far?"K16/K128":"K16");
        long npool=extract_rollout_tok(&see,&mp,tr_start,N,far,seed,Xr,trr,prr,posr,npool_max,&s16,&s128);
        fprintf(stderr,"   round %d: pool=%ld rollK16=%.3f rollK128=%.3f; train %dep\n",rd,npool,s16,s128,EP_MIX);
        train_mixed(&mp,tr0.X,tr0.tgt,tr0.prev,tr0.rows, Xr,trr,prr,npool, 1,5,EP_MIX,0.0005f);
        double v[N_VAL],bpu[N_VAL];
        for(int w=0;w<N_VAL;w++) v[w]=mlp_eval(&mp,va[w].X,va[w].tgt,va[w].prev,va[w].rows,va[w].bytes,&bpu[w]);
        snprintf(sp,sizeof sp,"%s_I_h32_r%d.bin",argv[4],rd);
        mlp_save(&mp,sp);
        printf("PROBE name=I_r%d h=32 rollK16=%.3f rollK128=%.3f val1=%.4f val2=%.4f val3=%.4f bpu1=%.3f\n",
               rd,s16,s128,v[0],v[1],v[2],bpu[0]);
        printf("SAVED name=I_r%d path=%s\n",rd,sp);
    }
    mlp_free(&mp);
    printf("\nReading: val* = bits per BYTE (unit-invariant, bar 2.2543). byteTRI = byte-trigram baseline\n");
    printf("over the same windows. CLOSED-LOOP verdict is the gate (byte-guards now structurally easy +\n");
    printf("topBi on tokens + BPB + human read). TF != generative (44-49 law): no celebration pre-gate.\n");
    return 0;
}
