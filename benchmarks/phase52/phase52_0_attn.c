// Phase 52.0 - O(t) keep-all-keys SINGLE FIXED-KEY ATTENTION HEAD (TF-first, stratified gate).
//
// Phase 51 proved the O(1)-superpose ceiling: relevance is load-bearing but crosstalk caps long-range
// recall (~4% of oracle), and sparse bucketing can't escape it (LSH==RAND). Coherence costs O(t).
// 52.0 wires ONE fixed-key attention head under the amended charter:
//   key_tau   = normalize(W_k * state_tau)   FIXED random projection of the frozen-substrate trajectory
//   value_tau = atom(token_{tau+1})          predictive (validated 51.B), the silicon's own trajectory
//   q_p       = W_q * state_p                 LEARNED (init = W_k)
//   a_{p,tau} = softmax_tau( q_p . key_tau )  over tau in [p-cap, p-2]   (no temp; W_q learns sharpness)
//   r_p       = sum_tau a_{p,tau} value_tau ;  logits = readout([state_p | r_p])
// Only W_q + readout are trained. Write = append (NO learned write-selection). No backprop into the
// substrate, no BPTT. Keys/values are the trajectory the silicon already computed; we learn to READ it.
//
// Causality (no-leak assert): value=next token => predicting token_p, attend only tau <= p-2. Context
// bound = safety window cap (TinyStories has NO distinct story delimiter -- stories and paragraphs both
// split on blank lines -- so a cap, not a story-reset, is the operative O(t) bound; cap >= 1024 covers
// the >200 stratum).
//
// Arms: NO-MEM (r=0 = 50.A baseline) / FULL-ATTN (cap=CAP_FULL) / RECENT-WINDOW (cap=32, isolates
// genuine long-range from local recency: FULL>RECENT proves the long range works) / PRED-SHUF (FULL,
// value permuted = content control, must collapse) / ORACLE (r=atom(next) = USE ceiling).
//
// Gate: stratified long-distance (>200 primary, 33-200 secondary), bar 0.05 fixed. KEY CHECK: in O(1)
// FULL capped at ~4% of oracle; without crosstalk a well-learned query should approach it -> report the
// FRACTION of the oracle ceiling captured per arm.
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase52/phase52_0_attn.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase52_0_attn.exe -lm -I .
// Run:
//   bin/phase52_0_attn.exe <data> <D1_w> <bpe_merges> <outprefix> [--len N] [--dk 1024] [--dv 256]
//        [--epochs 4] [--cap 1024] [--smoke]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include "src/silicon_entropy.h"
#include "benchmarks/phase50/bpe_codec.h"

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
#define HID       32

#define DWIN_S   0.05
#define DSHUF_S  0.03
#define RARE_P   1e-4
#define CAP_RECENT 32

enum { ST_ALL=0, ST_RECUR, ST_RR, ST_NONREC, ST_D1, ST_D2, ST_D3, ST_D4, NSTRAT };
static const char* STRN[NSTRAT]={"ALL","RECUR","RECUR&RARE","NONREC","dist1-4","dist5-32","dist33-200","dist>200"};

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
static double* g_uni; static double g_Ttrain;

static inline float dot_avx(const float* w, const float* f, int n){ __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(&w[i]),_mm256_loadu_ps(&f[i]),s);
    float o[8]; _mm256_storeu_ps(o,s); float r=o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; for(;i<n;i++) r+=w[i]*f[i]; return r; }
static inline void axpy_avx(float* y, const float* x, float a, int n){ __m256 va=_mm256_set1_ps(a); int i=0;
    for(;i<=n-8;i+=8) _mm256_storeu_ps(&y[i], _mm256_fmadd_ps(va,_mm256_loadu_ps(&x[i]),_mm256_loadu_ps(&y[i])));
    for(;i<n;i++) y[i]+=a*x[i]; }

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
static void tokenize_corpus(void){
    g_tok=(uint32_t*)malloc((size_t)g_fsz*sizeof(uint32_t));
    g_tokstart=(long*)malloc((size_t)(g_fsz+1)*sizeof(long));
    g_ntok=(long)bpe_encode_region(&g_bpe, g_data, 0, g_fsz, g_tok);
    long off=0; for(long i=0;i<g_ntok;i++){ g_tokstart[i]=off; off+=bpe_tok_len(&g_bpe,g_tok[i]); }
    g_tokstart[g_ntok]=off;
    fprintf(stderr,"tokenized: %ld tokens over %ld bytes (%.3f b/tok)\n",g_ntok,g_fsz,(double)g_fsz/g_ntok);
}
static void build_tbig(long tok_train_end){
    g_tbig=(float*)malloc((size_t)VTOK*VTOK*sizeof(float));
    g_uni=(double*)calloc(VTOK,sizeof(double)); g_Ttrain=(double)tok_train_end;
    double* ctx=(double*)calloc(VTOK,sizeof(double));
    uint32_t* bg=(uint32_t*)calloc((size_t)VTOK*VTOK,sizeof(uint32_t));
    for(long i=0;i<tok_train_end;i++) g_uni[g_tok[i]]+=1.0;
    for(long i=0;i+1<tok_train_end;i++){ uint32_t a=g_tok[i],b=g_tok[i+1]; bg[(size_t)a*VTOK+b]++; ctx[a]+=1.0; }
    double T=(double)tok_train_end; const double K=0.5;
    for(int a=0;a<VTOK;a++){ double ca=ctx[a];
        for(int b=0;b<VTOK;b++){ double pb=(g_uni[b]+K)/(T+K*VTOK); double p;
            if(ca>0){ double cab=bg[(size_t)a*VTOK+b]; p=(cab+K*VTOK*pb)/(ca+K*VTOK); } else p=pb;
            g_tbig[(size_t)a*VTOK+b]=(float)log((p>1e-30)?p:1e-30); } }
    free(ctx); free(bg);
}
static long extract(SiliconEntropyState* see, long start, long N, int want_strata,
                    float* X, uint32_t* tgt, uint32_t* prev, double* oBYTES,
                    uint8_t* recur, int* dist, uint8_t* rare){
    float L2d1[L2_DIM]={0}, pb_d1[BASE_DIM]={0}, eB[N_TS][D_EXP]; memset(eB,0,sizeof eB);
    float feat192[BASE_DIM], fa[BASE_DIM], rawd1[D1_TOT], nf[D1_TOT];
    float scale=1.0f/sqrtf((float)BASE_DIM); uint8_t cur_c2,cur_c1;
    long ti=0; { long lo=0,hi=g_ntok; while(lo<hi){ long mid=(lo+hi)/2; if(g_tokstart[mid]<start) lo=mid+1; else hi=mid; } ti=lo; }
    long bstart=g_tokstart[ti];
    see_reset(see); for(long i=0;i<bstart;i++) see_observe(see,g_data[i]);
    cur_c2=(bstart>=2)?g_data[bstart-2]:0; cur_c1=(bstart>=1)?g_data[bstart-1]:0;
    long rows=0, wbytes=0;
    long* lastpos = want_strata? (long*)malloc((size_t)VTOK*sizeof(long)) : NULL;
    if(lastpos) for(int v=0;v<VTOK;v++) lastpos[v]=-1;
    while(ti+1<g_ntok && g_tokstart[ti+1]+ (long)bpe_tok_len(&g_bpe,g_tok[ti+1]) <= start+N){
        see_extract(see,feat192);
        memcpy(rawd1,feat192,BASE_DIM*4); memcpy(rawd1+BASE_DIM,L2d1,L2_DIM*4);
        norm_feats(rawd1,nf);
        float* row=&X[(size_t)rows*S_DIM]; memcpy(row,nf,D1_TOT*4); row_bands(eB,row);
        uint32_t target=g_tok[ti+1];
        tgt[rows]=target; prev[rows]=g_tok[ti];
        if(want_strata){
            long lp=lastpos[target]; uint8_t rc=(lp>=0)?1:0; recur[rows]=rc;
            dist[rows]= rc? (int)(rows-lp) : -1;
            double pu=g_uni[target]/g_Ttrain; rare[rows]=(pu<RARE_P)?1:0;
            lastpos[target]=rows;
        }
        rows++; wbytes += bpe_tok_len(&g_bpe,g_tok[ti]);
        int L=bpe_tok_len(&g_bpe,g_tok[ti]); const unsigned char* eb=bpe_tok_bytes(&g_bpe,g_tok[ti]);
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
    if(lastpos) free(lastpos);
    *oBYTES=(double)wbytes;
    return rows;
}

// ============ head assets ============
static int g_Dk=1024, g_Dv=256;
static float* g_cb;    // [VTOK*Dv] unit bipolar value atoms
static float* g_Wk;    // [Dk*512] fixed key projection
static void store_assets_init(uint64_t seed){
    g_cb=(float*)realloc(g_cb,(size_t)VTOK*g_Dv*4); uint64_t s=seed?seed:0x520C0DEULL;
    float invv=1.0f/sqrtf((float)g_Dv);
    for(size_t i=0;i<(size_t)VTOK*g_Dv;i++){ s^=s<<13;s^=s>>7;s^=s<<17; g_cb[i]=(s&1ULL)?invv:-invv; }
    g_Wk=(float*)realloc(g_Wk,(size_t)g_Dk*512*4);
    for(size_t i=0;i<(size_t)g_Dk*512;i++){ s^=s<<13;s^=s>>7;s^=s<<17; g_Wk[i]=(s&1ULL)?1.f:-1.f; }
}
static void precompute_keys(const float* X, long nrow, float* khat){
    for(long i=0;i<nrow;i++){ const float* st=&X[(size_t)i*S_DIM]; float* k=&khat[(size_t)i*g_Dk];
        for(int a=0;a<g_Dk;a++) k[a]=dot_avx(&g_Wk[(size_t)a*512],st,512);
        float nn=0; for(int a=0;a<g_Dk;a++) nn+=k[a]*k[a]; float inv=1.0f/(sqrtf(nn)+1e-12f);
        for(int a=0;a<g_Dk;a++) k[a]*=inv; }
}

// ============ readout MLP (+ learned query W_q) ============
typedef struct { int in,H; float *W1,*b1,*W2,*b2,*mW1,*vW1,*mb1,*vb1,*mW2,*vW2,*mb2,*vb2;
                 float *Wq,*mWq,*vWq; int learn; int t; } Net;
static void net_init(Net* n,int in,int H,int learn,uint64_t seed){ n->in=in;n->H=H;n->t=0;n->learn=learn;
    size_t s1=(size_t)H*in,s2=(size_t)VTOK*H;
    n->W1=calloc(s1,4);n->b1=calloc(H,4);n->W2=calloc(s2,4);n->b2=calloc(VTOK,4);
    n->mW1=calloc(s1,4);n->vW1=calloc(s1,4);n->mb1=calloc(H,4);n->vb1=calloc(H,4);
    n->mW2=calloc(s2,4);n->vW2=calloc(s2,4);n->mb2=calloc(VTOK,4);n->vb2=calloc(VTOK,4);
    uint64_t r=seed?seed:0x1234567ULL; float sc1=sqrtf(2.0f/in);
    for(size_t i=0;i<s1;i++){ r^=r<<13;r^=r>>7;r^=r<<17; n->W1[i]=sc1*(((r>>11)*(1.0/(1ULL<<53)))*2-1); }
    float sc2=sqrtf(2.0f/H);
    for(size_t i=0;i<s2;i++){ r^=r<<13;r^=r>>7;r^=r<<17; n->W2[i]=sc2*(((r>>11)*(1.0/(1ULL<<53)))*2-1); }
    if(learn){ size_t sq=(size_t)g_Dk*512; n->Wq=malloc(sq*4); memcpy(n->Wq,g_Wk,sq*4);  // init W_q = W_k
        n->mWq=calloc(sq,4); n->vWq=calloc(sq,4); } else { n->Wq=n->mWq=n->vWq=NULL; }
}
static void net_free(Net* n){ free(n->W1);free(n->b1);free(n->W2);free(n->b2);
    free(n->mW1);free(n->vW1);free(n->mb1);free(n->vb1);free(n->mW2);free(n->vW2);free(n->mb2);free(n->vb2);
    if(n->Wq){free(n->Wq);free(n->mWq);free(n->vWq);} }
static inline void assemble(float* u,const float* state,const float* Rrow,int mem){
    memcpy(u,state,512*4); if(mem) memcpy(u+512,Rrow,g_Dv*4); }
static void net_fwd(const Net* n,const float* u,float* hid,float* lg){
    for(int j=0;j<n->H;j++){ float a=n->b1[j]+dot_avx(&n->W1[(size_t)j*n->in],u,n->in); hid[j]=a>0?a:0; }
    for(int c=0;c<VTOK;c++) lg[c]=n->b2[c]+dot_avx(&n->W2[(size_t)c*n->H],hid,n->H);
}
#define ADAM(P,GG,MM,VV,NN) for(size_t z=0;z<(size_t)(NN);z++){ MM[z]=.9f*MM[z]+.1f*GG[z]; VV[z]=.999f*VV[z]+.001f*GG[z]*GG[z]; P[z]-=lt*(MM[z]/(sqrtf(VV[z])+1e-8f)+1e-5f*P[z]); }
// OFFLINE clean training on precomputed [state|R] (NO-MEM mem=0; ORACLE r=atom(next)).
static void train_clean(Net* n,const float* X,const float* R,const uint32_t* tgt,const uint32_t* prev,
                        long nrow,int mem,int epochs,float lr){
    int H=n->H,in=n->in; size_t s1=(size_t)H*in,s2=(size_t)VTOK*H;
    float *gW1=malloc(s1*4),*gb1=malloc(H*4),*gW2=malloc(s2*4),*gb2=malloc((size_t)VTOK*4);
    float *u=malloc((size_t)in*4),*hid=malloc(H*4),*dh=malloc(H*4),*lg=malloc((size_t)VTOK*4),*eo=malloc((size_t)VTOK*4);
    int bs=512; float invn=1.0f/bs;
    for(int ep=0;ep<epochs;ep++){
        memset(gW1,0,s1*4);memset(gb1,0,H*4);memset(gW2,0,s2*4);memset(gb2,0,(size_t)VTOK*4); long inb=0;
        for(long i=0;i<nrow;i++){
            assemble(u,&X[(size_t)i*S_DIM], mem?&R[(size_t)i*g_Dv]:NULL, mem);
            net_fwd(n,u,hid,lg);
            const float* tb=&g_tbig[(size_t)prev[i]*VTOK]; for(int c=0;c<VTOK;c++) lg[c]+=tb[c];
            float mx=-1e30f; for(int c=0;c<VTOK;c++) if(lg[c]>mx)mx=lg[c];
            float Z=0; for(int c=0;c<VTOK;c++){ eo[c]=expf(lg[c]-mx); Z+=eo[c]; }
            for(int c=0;c<VTOK;c++){ float y=(c==(int)tgt[i])?1.f:0.f; eo[c]=(eo[c]/Z-y)*invn; }
            for(int c=0;c<VTOK;c++) gb2[c]+=eo[c];
            memset(dh,0,H*4);
            for(int c=0;c<VTOK;c++){ float e=eo[c]; float* gw=&gW2[(size_t)c*H]; const float* w2=&n->W2[(size_t)c*H];
                for(int j=0;j<H;j++){ gw[j]+=e*hid[j]; dh[j]+=e*w2[j]; } }
            for(int j=0;j<H;j++) if(hid[j]>0){ gb1[j]+=dh[j]; float* gw=&gW1[(size_t)j*in]; const float g=dh[j];
                for(int k=0;k<in;k++) gw[k]+=g*u[k]; }
            inb++;
            if(inb==bs || i==nrow-1){ n->t++; float lt=lr*sqrtf(1-powf(.999f,n->t))/(1-powf(.9f,n->t));
                ADAM(n->W1,gW1,n->mW1,n->vW1,s1); ADAM(n->b1,gb1,n->mb1,n->vb1,H);
                ADAM(n->W2,gW2,n->mW2,n->vW2,s2); ADAM(n->b2,gb2,n->mb2,n->vb2,VTOK);
                memset(gW1,0,s1*4);memset(gb1,0,H*4);memset(gW2,0,s2*4);memset(gb2,0,(size_t)VTOK*4); inb=0; }
        }
    }
    free(gW1);free(gb1);free(gW2);free(gb2);free(u);free(hid);free(dh);free(lg);free(eo);
}
// offline eval on precomputed [state|R] -> strata (for NO-MEM / ORACLE).
static void eval_clean(const Net* n,const float* X,const float* R,const uint32_t* tgt,const uint32_t* prev,
                       long nrow,int mem,const uint8_t* recur,const int* dist,const uint8_t* rare,
                       double* sbits,long* scnt){
    int H=n->H,in=n->in; float *u=malloc((size_t)in*4),*hid=malloc(H*4),*lg=malloc((size_t)VTOK*4);
    for(int st=0;st<NSTRAT;st++){ sbits[st]=0; scnt[st]=0; }
    for(long i=0;i<nrow;i++){
        assemble(u,&X[(size_t)i*S_DIM], mem?&R[(size_t)i*g_Dv]:NULL, mem);
        net_fwd(n,u,hid,lg);
        const float* tb=&g_tbig[(size_t)prev[i]*VTOK]; for(int c=0;c<VTOK;c++) lg[c]+=tb[c];
        float mx=-1e30f; for(int c=0;c<VTOK;c++) if(lg[c]>mx)mx=lg[c];
        double Z=0; for(int c=0;c<VTOK;c++) Z+=exp((double)(lg[c]-mx));
        double p=exp((double)(lg[tgt[i]]-mx))/Z; double b=-log2(p>1e-30?p:1e-30);
        sbits[ST_ALL]+=b; scnt[ST_ALL]++;
        if(recur[i]){ sbits[ST_RECUR]+=b; scnt[ST_RECUR]++; if(rare[i]){ sbits[ST_RR]+=b; scnt[ST_RR]++; }
            int d=dist[i]; int sb=(d<=4)?ST_D1:(d<=32)?ST_D2:(d<=200)?ST_D3:ST_D4; sbits[sb]+=b; scnt[sb]++;
        } else { sbits[ST_NONREC]+=b; scnt[ST_NONREC]++; }
    }
    free(u);free(hid);free(lg);
}

// ===== attention head: r_p = sum_{tau in [p-cap, p-2]} softmax(q_p . khat_tau) atom(vtok_tau).
// do_train: SGD on W_q+readout. else: strata eval. vtok = tgt[valperm?valperm[tau]:tau].
static long g_capmax;
static int attn_run(Net* n,const float* X,const float* khat,const uint32_t* tgt,const uint32_t* prev,
                    long nrow,int cap,const long* valperm,int do_train,int epochs,float lr,
                    const uint8_t* recur,const int* dist,const uint8_t* rare,double* sbits,long* scnt){
    int H=n->H,in=n->in,Dk=g_Dk,Dv=g_Dv; size_t s1=(size_t)H*in,s2=(size_t)VTOK*H,sq=(size_t)Dk*512;
    float *gW1=NULL,*gb1=NULL,*gW2=NULL,*gb2=NULL,*gWq=NULL;
    if(do_train){ gW1=calloc(s1,4);gb1=calloc(H,4);gW2=calloc(s2,4);gb2=calloc(VTOK,4);gWq=calloc(sq,4); }
    float *q=malloc((size_t)Dk*4),*r=malloc((size_t)Dv*4);
    float *aw=malloc((size_t)g_capmax*4),*da=malloc((size_t)g_capmax*4);   // softmax weights, value-grads
    float *u=malloc((size_t)in*4),*hid=malloc(H*4),*dh=malloc(H*4),*lg=malloc((size_t)VTOK*4),*eo=malloc((size_t)VTOK*4);
    float *du=malloc((size_t)in*4),*dLdq=malloc((size_t)Dk*4);
    int bs=512; float invn=1.0f/bs; long inb=0; int ok=1; int E=do_train?epochs:1;
    if(scnt){ for(int st=0;st<NSTRAT;st++){ sbits[st]=0; scnt[st]=0; } }
    for(int ep=0;ep<E;ep++){
        if(do_train){ memset(gW1,0,s1*4);memset(gb1,0,H*4);memset(gW2,0,s2*4);memset(gb2,0,(size_t)VTOK*4);memset(gWq,0,sq*4); inb=0; }
        for(long p=0;p<nrow;p++){
            const float* state=&X[(size_t)p*S_DIM];
            long hi=p-2, lo=hi-(cap-1); if(lo<0) lo=0;
            int m = (hi>=lo)? (int)(hi-lo+1) : 0;
            for(int a2=0;a2<Dk;a2++) q[a2]=dot_avx(&n->Wq[(size_t)a2*512],state,512);
            for(int b=0;b<Dv;b++) r[b]=0.f;
            if(m>0){
                float mx=-1e30f;
                for(int j=0;j<m;j++){ long tau=lo+j; float s=dot_avx(q,&khat[(size_t)tau*Dk],Dk); aw[j]=s; if(s>mx)mx=s; }
                float Z=0; for(int j=0;j<m;j++){ aw[j]=expf(aw[j]-mx); Z+=aw[j]; }
                float zinv=1.0f/Z; for(int j=0;j<m;j++) aw[j]*=zinv;
                for(int j=0;j<m;j++){ long tau=lo+j; const float* v=&g_cb[(size_t)tgt[valperm?valperm[tau]:tau]*Dv]; axpy_avx(r,v,aw[j],Dv); }
            }
            assemble(u,state,r,1);
            net_fwd(n,u,hid,lg);
            const float* tb=&g_tbig[(size_t)prev[p]*VTOK]; for(int c=0;c<VTOK;c++) lg[c]+=tb[c];
            float mx2=-1e30f; for(int c=0;c<VTOK;c++) if(lg[c]>mx2)mx2=lg[c];
            double Zd=0; for(int c=0;c<VTOK;c++){ double e=exp((double)(lg[c]-mx2)); eo[c]=(float)e; Zd+=e; }
            if(scnt){ double pp=eo[tgt[p]]/Zd; double b=-log2(pp>1e-30?pp:1e-30);
                sbits[ST_ALL]+=b; scnt[ST_ALL]++;
                if(recur[p]){ sbits[ST_RECUR]+=b; scnt[ST_RECUR]++; if(rare[p]){ sbits[ST_RR]+=b; scnt[ST_RR]++; }
                    int d=dist[p]; int sb=(d<=4)?ST_D1:(d<=32)?ST_D2:(d<=200)?ST_D3:ST_D4; sbits[sb]+=b; scnt[sb]++;
                } else { sbits[ST_NONREC]+=b; scnt[ST_NONREC]++; } }
            if(do_train){
                for(int c=0;c<VTOK;c++){ float y=(c==(int)tgt[p])?1.f:0.f; eo[c]=(float)(eo[c]/Zd-y)*invn; }
                for(int c=0;c<VTOK;c++) gb2[c]+=eo[c];
                memset(dh,0,H*4);
                for(int c=0;c<VTOK;c++){ float e=eo[c]; float* gw=&gW2[(size_t)c*H]; const float* w2=&n->W2[(size_t)c*H];
                    for(int j=0;j<H;j++){ gw[j]+=e*hid[j]; dh[j]+=e*w2[j]; } }
                memset(du,0,in*4);
                for(int j=0;j<H;j++) if(hid[j]>0){ gb1[j]+=dh[j]; float gg=dh[j]; float* gw=&gW1[(size_t)j*in]; const float* w1=&n->W1[(size_t)j*in];
                    for(int k=0;k<in;k++){ gw[k]+=gg*u[k]; du[k]+=gg*w1[k]; } }
                if(m>0){
                    const float* dur=du+512;                                   // dL/dr (Dv)
                    float asum=0;
                    for(int j=0;j<m;j++){ long tau=lo+j; const float* v=&g_cb[(size_t)tgt[valperm?valperm[tau]:tau]*Dv];
                        da[j]=dot_avx(dur,v,Dv); asum+=aw[j]*da[j]; }
                    for(int b=0;b<Dk;b++) dLdq[b]=0.f;
                    for(int j=0;j<m;j++){ long tau=lo+j; float g=aw[j]*(da[j]-asum);     // dL/ds_j
                        if(g!=0.f) axpy_avx(dLdq,&khat[(size_t)tau*Dk],g,Dk); }
                    for(int a2=0;a2<Dk;a2++){ float dq=dLdq[a2]; if(dq!=0.f){ float* gw=&gWq[(size_t)a2*512]; axpy_avx(gw,state,dq,512); } }
                }
                inb++;
                if(inb==bs || p==nrow-1){ n->t++; float lt=lr*sqrtf(1-powf(.999f,n->t))/(1-powf(.9f,n->t));
                    ADAM(n->W1,gW1,n->mW1,n->vW1,s1); ADAM(n->b1,gb1,n->mb1,n->vb1,H);
                    ADAM(n->W2,gW2,n->mW2,n->vW2,s2); ADAM(n->b2,gb2,n->mb2,n->vb2,VTOK);
                    ADAM(n->Wq,gWq,n->mWq,n->vWq,sq);
                    memset(gW1,0,s1*4);memset(gb1,0,H*4);memset(gW2,0,s2*4);memset(gb2,0,(size_t)VTOK*4);memset(gWq,0,sq*4); inb=0; }
            }
        }
        if(do_train) fprintf(stderr,"    attn ep %d/%d (cap=%d)\n",ep+1,E,cap);
    }
    if(do_train){ free(gW1);free(gb1);free(gW2);free(gb2);free(gWq); }
    free(q);free(r);free(aw);free(da);free(u);free(hid);free(dh);free(lg);free(eo);free(du);free(dLdq);
    return ok;
}

static double bpt(double bits,long cnt){ return cnt? bits/cnt : 0.0; }
static double substrate_checksum(void){ double s=0; for(int i=0;i<D1_TOT;i++) s+=md1[i]+sd1[i];
    for(int d=0;d<D_EXP;d++){ s+=Bvec[d]; for(int k=0;k<L0_DIM;k++) s+=Omega[d][k]; } return s; }

// windows + scratch globals
static float *g_Xtr,*g_Xv[N_VAL]; static uint32_t *g_ttr,*g_ptr,*g_tv[N_VAL],*g_pv[N_VAL];
static long g_trrows,g_vrows[N_VAL]; static uint8_t *g_vrec[N_VAL],*g_vrare[N_VAL]; static int *g_vdist[N_VAL];
static long g_maxrows; static long* g_perm; static int g_epochs; static double g_nm[NSTRAT];
static float *g_khat_tr, *g_khat_v[N_VAL];   // precomputed keys

// run an attention arm; aggregate strata across val windows. returns d_NOMEM[D3/D4], d for caller.
static void run_attn(const char* label,int cap,int shuf,const char* outprefix,double* o_pr){
    Net net; net_init(&net,512+g_Dv,HID,1,0xB1ULL^(uint64_t)cap^(shuf?0x9999:0));
    long* perm = shuf? g_perm : NULL;
    if(shuf){ for(long i=0;i<g_trrows;i++) g_perm[i]=i;
        uint64_t s=0x5C0FF1E0ULL; for(long i=g_trrows-1;i>0;i--){ s^=s<<13;s^=s>>7;s^=s<<17; long j=(long)((s>>11)%(uint64_t)(i+1)); long t=g_perm[i];g_perm[i]=g_perm[j];g_perm[j]=t; } }
    double ck0=substrate_checksum();
    attn_run(&net,g_Xtr,g_khat_tr,g_ttr,g_ptr,g_trrows,cap,perm,1,g_epochs,5e-4f,NULL,NULL,NULL,NULL,NULL);
    double ck1=substrate_checksum();
    double b[NSTRAT]={0}; long c[NSTRAT]={0};
    for(int w=0;w<N_VAL;w++){ double sb[NSTRAT]; long sc[NSTRAT]; long* vperm=NULL;
        if(shuf){ vperm=g_perm; for(long i=0;i<g_vrows[w];i++) g_perm[i]=i;
            uint64_t s=0x5C0FF1E0ULL^(uint64_t)(w+1); for(long i=g_vrows[w]-1;i>0;i--){ s^=s<<13;s^=s>>7;s^=s<<17; long j=(long)((s>>11)%(uint64_t)(i+1)); long t=g_perm[i];g_perm[i]=g_perm[j];g_perm[j]=t; } }
        attn_run(&net,g_Xv[w],g_khat_v[w],g_tv[w],g_pv[w],g_vrows[w],cap,vperm,0,1,0,g_vrec[w],g_vdist[w],g_vrare[w],sb,sc);
        for(int st=0;st<NSTRAT;st++){ b[st]+=sb[st]; c[st]+=sc[st]; } }
    double pr[NSTRAT]; for(int st=0;st<NSTRAT;st++) pr[st]=bpt(b[st],c[st]);
    if(outprefix){ char sp[512]; snprintf(sp,sizeof sp,"%s_%s.bin",outprefix,label);
        FILE* f=fopen(sp,"wb"); if(f){ uint32_t mg=0x53454559; int dk=g_Dk,dv=g_Dv,cp=cap; fwrite(&mg,4,1,f);fwrite(&dk,4,1,f);fwrite(&dv,4,1,f);fwrite(&cp,4,1,f);
            fwrite(net.W1,4,(size_t)net.H*net.in,f);fwrite(net.b1,4,net.H,f);fwrite(net.W2,4,(size_t)VTOK*net.H,f);fwrite(net.b2,4,VTOK,f);fwrite(net.Wq,4,(size_t)g_Dk*512,f); fclose(f);} }
    net_free(&net);
    printf("==== %s (cap=%d%s) | no-substrate-grad %s ====\n",label,cap,shuf?" SHUF":"",(fabs(ck1-ck0)<1e-3)?"PASS":"*** FAIL ***");
    printf("  %-12s %9s %9s | %8s\n","stratum","NO-MEM",label,"d_NOMEM");
    int order[4]={ST_D4,ST_D3,ST_D2,ST_RECUR};
    for(int oi=0;oi<4;oi++){ int st=order[oi]; printf("  %-12s %9.4f %9.4f | %+8.4f\n",STRN[st],g_nm[st],pr[st],g_nm[st]-pr[st]); }
    if(o_pr) for(int st=0;st<NSTRAT;st++) o_pr[st]=pr[st];
}

int main(int argc,char** argv){
    if(argc<5){ fprintf(stderr,"Usage: %s <data> <D1_w> <bpe_merges> <outprefix> [--len N] [--dk D] [--dv D] [--epochs E] [--cap C] [--smoke]\n",argv[0]); return 1; }
    setvbuf(stderr,NULL,_IONBF,0); setvbuf(stdout,NULL,_IONBF,0);
    long N=150000, maxb=0; g_epochs=4; int cap_full=1024;
    for(int i=5;i<argc;i++){
        if(!strcmp(argv[i],"--len")&&i+1<argc) N=atol(argv[++i]);
        else if(!strcmp(argv[i],"--dk")&&i+1<argc) g_Dk=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--dv")&&i+1<argc) g_Dv=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--epochs")&&i+1<argc) g_epochs=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--cap")&&i+1<argc) cap_full=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--max-bytes")&&i+1<argc) maxb=atol(argv[++i]);
        else if(!strcmp(argv[i],"--smoke")){ N=20000; g_Dk=256; g_Dv=64; g_epochs=2; cap_full=256; }
    }
    g_capmax = cap_full>CAP_RECENT? cap_full : CAP_RECENT;
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
        double Hh=0; for(int k=0;k<CLASSES;k++){ double pp=exp((double)(trigram[i][j][k]-m))/se; if(pp>1e-12)Hh-=pp*log(pp); }
        ent_table[i][j]=(float)Hh; }

    tokenize_corpus();
    long tok_train_end=(long)(g_ntok*0.90);
    fprintf(stderr,"building token-bigram prior + unigram...\n");
    build_tbig(tok_train_end);
    store_assets_init(0x520C0DEULL);

    long tr_start=g_fsz/5;
    long va_start[N_VAL]={ g_fsz/2, (long)(0.65*g_fsz), (long)(0.80*g_fsz) };
    for(int w=0;w<N_VAL;w++) if(va_start[w]+N+3>g_fsz){ fprintf(stderr,"val window %d out of file\n",w+1); return 1; }

    g_maxrows=N+16;
    g_Xtr=malloc((size_t)g_maxrows*S_DIM*4); g_ttr=malloc((size_t)g_maxrows*4); g_ptr=malloc((size_t)g_maxrows*4);
    for(int w=0;w<N_VAL;w++){ g_Xv[w]=malloc((size_t)g_maxrows*S_DIM*4); g_tv[w]=malloc((size_t)g_maxrows*4); g_pv[w]=malloc((size_t)g_maxrows*4);
        g_vrec[w]=malloc(g_maxrows); g_vrare[w]=malloc(g_maxrows); g_vdist[w]=malloc((size_t)g_maxrows*sizeof(int)); }
    double trb; fprintf(stderr,"extract train...\n");
    g_trrows=extract(&see,tr_start,N,0,g_Xtr,g_ttr,g_ptr,&trb,NULL,NULL,NULL);
    for(int w=0;w<N_VAL;w++){ double vb; fprintf(stderr,"extract val%d...\n",w+1);
        g_vrows[w]=extract(&see,va_start[w],N,1,g_Xv[w],g_tv[w],g_pv[w],&vb,g_vrec[w],g_vdist[w],g_vrare[w]); }
    g_perm=malloc((size_t)g_maxrows*sizeof(long));
    fprintf(stderr,"precompute keys...\n");
    g_khat_tr=malloc((size_t)g_maxrows*g_Dk*4); precompute_keys(g_Xtr,g_trrows,g_khat_tr);
    for(int w=0;w<N_VAL;w++){ g_khat_v[w]=malloc((size_t)g_maxrows*g_Dk*4); precompute_keys(g_Xv[w],g_vrows[w],g_khat_v[w]); }

    printf("\n==== 52.0 O(t) single fixed-key attention head (Dk=%d Dv=%d H=%d N=%ld cap=%d epochs=%d) ====\n",g_Dk,g_Dv,HID,N,cap_full,g_epochs);
    printf("Gate on dist>200, bar %.2f fixed. KEY CHECK: fraction of oracle ceiling captured (O(1) capped ~4%%).\n",DWIN_S);
    printf("Arms: NO-MEM / ORACLE(ceiling) / FULL-ATTN(cap=%d) / RECENT-WINDOW(cap=%d) / PRED-SHUF.\n\n",cap_full,CAP_RECENT);

    // NO-MEM
    { fprintf(stderr,"=== NO-MEM ===\n"); double b[NSTRAT]={0}; long c[NSTRAT]={0};
      Net net; net_init(&net,512,HID,0,0xA0ULL);
      train_clean(&net,g_Xtr,NULL,g_ttr,g_ptr,g_trrows,0,g_epochs,5e-4f);
      for(int w=0;w<N_VAL;w++){ double sb[NSTRAT]; long sc[NSTRAT];
          eval_clean(&net,g_Xv[w],NULL,g_tv[w],g_pv[w],g_vrows[w],0,g_vrec[w],g_vdist[w],g_vrare[w],sb,sc);
          for(int st=0;st<NSTRAT;st++){ b[st]+=sb[st]; c[st]+=sc[st]; } }
      net_free(&net); for(int st=0;st<NSTRAT;st++) g_nm[st]=bpt(b[st],c[st]); }
    printf("NO-MEM: dist>200=%.4f dist33-200=%.4f RECUR=%.4f ALL=%.4f\n\n",g_nm[ST_D4],g_nm[ST_D3],g_nm[ST_RECUR],g_nm[ST_ALL]);

    // ORACLE ceiling: r = atom(next token) (perfect retrieval)
    double orc[NSTRAT];
    { fprintf(stderr,"=== ORACLE ===\n"); double b[NSTRAT]={0}; long c[NSTRAT]={0};
      float* Roa=malloc((size_t)g_maxrows*g_Dv*4);
      Net net; net_init(&net,512+g_Dv,HID,0,0xDEAD1ULL);
      for(long i=0;i<g_trrows;i++) memcpy(&Roa[(size_t)i*g_Dv],&g_cb[(size_t)g_ttr[i]*g_Dv],g_Dv*4);
      train_clean(&net,g_Xtr,Roa,g_ttr,g_ptr,g_trrows,1,g_epochs,5e-4f);
      for(int w=0;w<N_VAL;w++){ double sb[NSTRAT]; long sc[NSTRAT];
          for(long i=0;i<g_vrows[w];i++) memcpy(&Roa[(size_t)i*g_Dv],&g_cb[(size_t)g_tv[w][i]*g_Dv],g_Dv*4);
          eval_clean(&net,g_Xv[w],Roa,g_tv[w],g_pv[w],g_vrows[w],1,g_vrec[w],g_vdist[w],g_vrare[w],sb,sc);
          for(int st=0;st<NSTRAT;st++){ b[st]+=sb[st]; c[st]+=sc[st]; } }
      net_free(&net); free(Roa); for(int st=0;st<NSTRAT;st++) orc[st]=bpt(b[st],c[st]); }
    printf("ORACLE ceiling: dist>200 d_NOMEM=%+.4f | dist33-200=%+.4f (USE upper bound = perfect retrieval)\n\n",
           g_nm[ST_D4]-orc[ST_D4],g_nm[ST_D3]-orc[ST_D3]);

    double full[NSTRAT],recent[NSTRAT],shufv[NSTRAT];
    fprintf(stderr,"=== FULL-ATTN ===\n");   run_attn("FULL-ATTN",cap_full,0,argv[4],full);
    fprintf(stderr,"=== RECENT-WINDOW ===\n"); run_attn("RECENT-WIN",CAP_RECENT,0,NULL,recent);
    fprintf(stderr,"=== PRED-SHUF ===\n");    run_attn("PRED-SHUF",cap_full,1,NULL,shufv);

    // ---- verdict ----
    printf("\n==== 52.0 VERDICT (gate on dist>200, bar %.2f) ====\n",DWIN_S);
    printf("  %-12s | NO-MEM   FULL    RECENT  SHUF   | d_NOMEM(FULL) d_RECENT d_SHUF  frac_oracle\n","stratum");
    int gst[2]={ST_D4,ST_D3};
    for(int gi=0; gi<2; gi++){ int st=gst[gi];
        double dN=g_nm[st]-full[st], dR=recent[st]-full[st], dS=shufv[st]-full[st];
        double frac=(g_nm[st]-orc[st])>1e-6?dN/(g_nm[st]-orc[st]):0;
        printf("  %-12s | %7.4f %7.4f %7.4f %7.4f | %+11.4f %+8.4f %+7.4f  %.0f%%\n",
               STRN[st],g_nm[st],full[st],recent[st],shufv[st],dN,dR,dS,100*frac); }
    int st=ST_D4; double dN=g_nm[st]-full[st], dR=recent[st]-full[st], dS=shufv[st]-full[st];
    int pass=(dN>=DWIN_S)&&(dS>=DSHUF_S)&&(dR>=0);
    printf("  -> GATE(dist>200): %s  d_NOMEM=%+.4f(bar %.2f) d_SHUF=%+.4f(bar %.2f) FULL>RECENT=%+.4f\n",
           pass?"PASS (-> 52.A closed-loop)":"fail",dN,DWIN_S,dS,DSHUF_S,dR);
    printf("\nReading: FULL captures frac_oracle of the ceiling (O(1) capped ~4%%); FULL>RECENT = genuine\n");
    printf("long-range (not recency); FULL>SHUF = real content. USE-test stays GENERATIVE (52.A).\n");
    return 0;
}
