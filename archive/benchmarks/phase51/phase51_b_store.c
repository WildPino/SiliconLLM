// Phase 51.B - PREDICTIVE / induction store + STRATIFIED TF gate.
//
// 51.A proved two things with clean controls: the TF MEAN is blind to memory utility, and
// value=previous-token is redundant with the reservoir+bigram. 51.B fixes BOTH:
//   VALUE  = atom of the NEXT token (induction: "what followed a similar context before")
//   KEY=QUERY = phi(W_k * context_state), W_k FIXED random (shared) -> pure content-address,
//               r_t is PARAMETER-INDEPENDENT and PRECOMPUTABLE (so the readout trains offline).
//   GATE   = STRATIFIED BPB (recurrence x rarity x distance), not the blind mean.
//
// Store (fast-weight, Hebbian, no backprop): M <- decay*M + key_t (x) value_t ; r_t = norm(M^T q_t).
// Only the readout (MLP H32) + token-bigram prior are trained; armB substrate and W_k frozen.
//
// CAUSALITY (mandatory no-leakage): predicting y_p, M must hold only writes for tau <= p-2
// (exclude the current target's pair). Implemented as a ONE-STEP-DELAYED write: at boundary i we
// retrieve from M, then flush boundary (i-1)'s pair. A write-counter assert enforces it; --leak-test
// deliberately breaks it (write-before-predict) to prove the metric catches leakage (BPB -> ~0).
//
// Arms: NO-MEM (control) / PREDICT / PREDICT-SHUF (value permuted -> key->value broken = isolates
// retrieval from width). Decay sweep (0.99 vs a low value = target the distant).
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase38-42/phase51_b_store.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase51_b_store.exe -lm -I .
// Run:
//   bin/phase51_b_store.exe <data> <D1_w> <bpe_merges> <outprefix>
//        [--len N] [--dk 1024] [--dv 256] [--epochs 6] [--smoke] [--leak-test]

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
#define S_DIM     (D1_TOT + EXP_BANDS)        // 512
#define PROJ_SEED 0x48B2EC0DEULL
#define N_VAL     3
#define HID       32

// stratified gate thresholds (bits/byte) on the recurrence strata (pre-registered)
#define DWIN_S   0.05
#define DSHUF_S  0.03
#define RARE_P   1e-4    // unigram prob below this = "rare"

// strata
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
static double* g_uni; static double g_Ttrain;   // unigram probs source

static inline float dot_avx(const float* w, const float* f, int n){ __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(&w[i]),_mm256_loadu_ps(&f[i]),s);
    float o[8]; _mm256_storeu_ps(o,s); float r=o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; for(;i<n;i++) r+=w[i]*f[i]; return r; }
static inline void axpy_avx(float* y, const float* x, float a, int n){ __m256 va=_mm256_set1_ps(a); int i=0;
    for(;i<=n-8;i+=8) _mm256_storeu_ps(&y[i], _mm256_fmadd_ps(va,_mm256_loadu_ps(&x[i]),_mm256_loadu_ps(&y[i])));
    for(;i<n;i++) y[i]+=a*x[i]; }
static inline void scal_avx(float* y, float a, int n){ __m256 va=_mm256_set1_ps(a); int i=0;
    for(;i<=n-8;i+=8) _mm256_storeu_ps(&y[i], _mm256_mul_ps(va,_mm256_loadu_ps(&y[i])));
    for(;i<n;i++) y[i]*=a; }

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

// ---- substrate extraction + per-row strata metadata ----
// X[row*S_DIM]=state; tgt=next token; prev=ctx token; for val also: recur, dist, rare flags.
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

// ============ store assets + predictive replay (r_t precompute) ============
static int g_Dk=1024, g_Dv=256;
static float* g_cb;    // [VTOK*Dv] unit bipolar value atoms
static float* g_Wk;    // [Dk*512] fixed key/query projection (+-1)
static void store_assets_init(uint64_t seed){
    g_cb=(float*)malloc((size_t)VTOK*g_Dv*4); uint64_t s=seed?seed:0x51B0C0DEULL;
    float invv=1.0f/sqrtf((float)g_Dv);
    for(size_t i=0;i<(size_t)VTOK*g_Dv;i++){ s^=s<<13;s^=s>>7;s^=s<<17; g_cb[i]=(s&1ULL)?invv:-invv; }
    g_Wk=(float*)malloc((size_t)g_Dk*512*4);
    for(size_t i=0;i<(size_t)g_Dk*512;i++){ s^=s<<13;s^=s>>7;s^=s<<17; g_Wk[i]=(s&1ULL)?1.f:-1.f; }
}
static inline void key_of(const float* state,float* phi){
    for(int a=0;a<g_Dk;a++) phi[a]=dot_avx(&g_Wk[(size_t)a*512],state,512);
    float nn=0; for(int a=0;a<g_Dk;a++) nn+=phi[a]*phi[a]; nn=1.0f/(sqrtf(nn)+1e-12f);
    for(int a=0;a<g_Dk;a++) phi[a]*=nn;
}
// Precompute r_t into Rout[row*Dv]. valperm: if non-NULL, value uses tgt[valperm[i]] (SHUF).
// leak!=0 => write BEFORE retrieve (canary; breaks causality on purpose).
// Returns 0 if a causality assert trips.
static int replay(const float* X,const uint32_t* tgt,long nrow,float decay,
                  const long* valperm,int leak,float* Rout,float* M){
    float* q=malloc((size_t)g_Dk*4); float* r=malloc((size_t)g_Dv*4);
    float* pend_key=malloc((size_t)g_Dk*4); int pend_have=0; const float* pend_val=NULL;
    long writes=0; int ok=1;
    memset(M,0,(size_t)g_Dk*g_Dv*4);
    for(long i=0;i<nrow;i++){
        const float* state=&X[(size_t)i*S_DIM];
        key_of(state,q);
        const float* vi = &g_cb[(size_t)tgt[valperm?valperm[i]:i]*g_Dv];
        if(leak){ // canary: write current pair first (leaks the answer)
            scal_avx(M,decay,g_Dk*g_Dv);
            for(int a=0;a<g_Dk;a++){ float c=q[a]; if(c!=0.f) axpy_avx(&M[(size_t)a*g_Dv],vi,c,g_Dv); }
            writes++;
        }
        // retrieve r = normalize(M^T q)
        for(int b=0;b<g_Dv;b++) r[b]=0.f;
        for(int a=0;a<g_Dk;a++){ float qa=q[a]; if(qa!=0.f) axpy_avx(r,&M[(size_t)a*g_Dv],qa,g_Dv); }
        float nn=0; for(int b=0;b<g_Dv;b++) nn+=r[b]*r[b]; float inv=1.0f/(sqrtf(nn)+1e-12f);
        for(int b=0;b<g_Dv;b++) Rout[(size_t)i*g_Dv+b]=r[b]*inv;
        if(!leak){
            // causal assert: before flushing pair (i-1), M must hold exactly i-1 writes (tau<=i-2)
            if(writes != (i>0?i-1:0)){ fprintf(stderr,"LEAKAGE ASSERT FAIL i=%ld writes=%ld\n",i,writes); ok=0; }
            if(pend_have){ scal_avx(M,decay,g_Dk*g_Dv);
                for(int a=0;a<g_Dk;a++){ float c=pend_key[a]; if(c!=0.f) axpy_avx(&M[(size_t)a*g_Dv],pend_val,c,g_Dv); }
                writes++; }
            memcpy(pend_key,q,g_Dk*4); pend_val=vi; pend_have=1;
        }
    }
    free(q);free(r);free(pend_key);
    return ok;
}

// ============ readout MLP (H32) over [state | r] with token-bigram prior ============
typedef struct { int in,H; float *W1,*b1,*W2,*b2,*mW1,*vW1,*mb1,*vb1,*mW2,*vW2,*mb2,*vb2; int t; } Net;
static void net_init(Net* n,int in,int H,uint64_t seed){ n->in=in;n->H=H;n->t=0;
    size_t s1=(size_t)H*in,s2=(size_t)VTOK*H;
    n->W1=calloc(s1,4);n->b1=calloc(H,4);n->W2=calloc(s2,4);n->b2=calloc(VTOK,4);
    n->mW1=calloc(s1,4);n->vW1=calloc(s1,4);n->mb1=calloc(H,4);n->vb1=calloc(H,4);
    n->mW2=calloc(s2,4);n->vW2=calloc(s2,4);n->mb2=calloc(VTOK,4);n->vb2=calloc(VTOK,4);
    uint64_t r=seed?seed:0x1234567ULL; float sc1=sqrtf(2.0f/in);
    for(size_t i=0;i<s1;i++){ r^=r<<13;r^=r>>7;r^=r<<17; n->W1[i]=sc1*(((r>>11)*(1.0/(1ULL<<53)))*2-1); }
    float sc2=sqrtf(2.0f/H);
    for(size_t i=0;i<s2;i++){ r^=r<<13;r^=r>>7;r^=r<<17; n->W2[i]=sc2*(((r>>11)*(1.0/(1ULL<<53)))*2-1); }
}
static void net_free(Net* n){ free(n->W1);free(n->b1);free(n->W2);free(n->b2);
    free(n->mW1);free(n->vW1);free(n->mb1);free(n->vb1);free(n->mW2);free(n->vW2);free(n->mb2);free(n->vb2); }
static inline void assemble(float* u,const float* state,const float* Rrow,int mem){
    memcpy(u,state,512*4); if(mem) memcpy(u+512,Rrow,g_Dv*4);
}
static void net_fwd(const Net* n,const float* u,float* hid,float* lg){
    for(int j=0;j<n->H;j++){ float a=n->b1[j]+dot_avx(&n->W1[(size_t)j*n->in],u,n->in); hid[j]=a>0?a:0; }
    for(int c=0;c<VTOK;c++) lg[c]=n->b2[c]+dot_avx(&n->W2[(size_t)c*n->H],hid,n->H);
}
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
                #define ADAM(P,GG,MM,VV,NN) for(size_t z=0;z<(size_t)(NN);z++){ MM[z]=.9f*MM[z]+.1f*GG[z]; VV[z]=.999f*VV[z]+.001f*GG[z]*GG[z]; P[z]-=lt*(MM[z]/(sqrtf(VV[z])+1e-8f)+1e-5f*P[z]); }
                ADAM(n->W1,gW1,n->mW1,n->vW1,s1); ADAM(n->b1,gb1,n->mb1,n->vb1,H);
                ADAM(n->W2,gW2,n->mW2,n->vW2,s2); ADAM(n->b2,gb2,n->mb2,n->vb2,VTOK);
                memset(gW1,0,s1*4);memset(gb1,0,H*4);memset(gW2,0,s2*4);memset(gb2,0,(size_t)VTOK*4); inb=0; }
        }
    }
    free(gW1);free(gb1);free(gW2);free(gb2);free(u);free(hid);free(dh);free(lg);free(eo);
}
// stratified eval: accumulate bits and counts per stratum.
static void eval_strat(const Net* n,const float* X,const float* R,const uint32_t* tgt,const uint32_t* prev,
                       long nrow,int mem,const uint8_t* recur,const int* dist,const uint8_t* rare,
                       double* sbits,long* scnt){
    int H=n->H,in=n->in;
    float *u=malloc((size_t)in*4),*hid=malloc(H*4),*lg=malloc((size_t)VTOK*4);
    for(int st=0;st<NSTRAT;st++){ sbits[st]=0; scnt[st]=0; }
    for(long i=0;i<nrow;i++){
        assemble(u,&X[(size_t)i*S_DIM], mem?&R[(size_t)i*g_Dv]:NULL, mem);
        net_fwd(n,u,hid,lg);
        const float* tb=&g_tbig[(size_t)prev[i]*VTOK]; for(int c=0;c<VTOK;c++) lg[c]+=tb[c];
        float mx=-1e30f; for(int c=0;c<VTOK;c++) if(lg[c]>mx)mx=lg[c];
        double Z=0; for(int c=0;c<VTOK;c++) Z+=exp((double)(lg[c]-mx));
        double p=exp((double)(lg[tgt[i]]-mx))/Z; double b=-log2(p>1e-30?p:1e-30);
        sbits[ST_ALL]+=b; scnt[ST_ALL]++;
        if(recur[i]){ sbits[ST_RECUR]+=b; scnt[ST_RECUR]++;
            if(rare[i]){ sbits[ST_RR]+=b; scnt[ST_RR]++; }
            int d=dist[i]; int sb=(d<=4)?ST_D1:(d<=32)?ST_D2:(d<=200)?ST_D3:ST_D4;
            sbits[sb]+=b; scnt[sb]++;
        } else { sbits[ST_NONREC]+=b; scnt[ST_NONREC]++; }
    }
    free(u);free(hid);free(lg);
}
static double bpt(double bits,long cnt){ return cnt? bits/cnt : 0.0; } // bits per TOKEN (stratum-internal)

int main(int argc,char** argv){
    if(argc<5){ fprintf(stderr,"Usage: %s <data> <D1_w> <bpe_merges> <outprefix> [--len N] [--dk D] [--dv D] [--epochs E] [--smoke] [--leak-test]\n",argv[0]); return 1; }
    setvbuf(stderr,NULL,_IONBF,0); setvbuf(stdout,NULL,_IONBF,0);
    long N=400000, maxb=0; int epochs=6, smoke=0, leak=0;
    int ndec=2; float decays[4]={0.99f,0.95f,0,0};
    for(int i=5;i<argc;i++){
        if(!strcmp(argv[i],"--len")&&i+1<argc) N=atol(argv[++i]);
        else if(!strcmp(argv[i],"--dk")&&i+1<argc) g_Dk=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--dv")&&i+1<argc) g_Dv=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--epochs")&&i+1<argc) epochs=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--max-bytes")&&i+1<argc) maxb=atol(argv[++i]);
        else if(!strcmp(argv[i],"--leak-test")) leak=1;
        else if(!strcmp(argv[i],"--smoke")){ smoke=1; N=20000; g_Dk=256; g_Dv=64; epochs=2; ndec=1; }
    }
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
    fprintf(stderr,"building token-bigram prior + unigram (train tokens %ld)...\n",tok_train_end);
    build_tbig(tok_train_end);
    store_assets_init(0x51B0C0DEULL);

    long tr_start=g_fsz/5;
    long va_start[N_VAL]={ g_fsz/2, (long)(0.65*g_fsz), (long)(0.80*g_fsz) };
    for(int w=0;w<N_VAL;w++) if(va_start[w]+N+3>g_fsz){ fprintf(stderr,"val window %d out of file\n",w+1); return 1; }
    fprintf(stderr,"51.B store: train@%ld N=%ld Dk=%d Dv=%d epochs=%d decays=%d leak=%d\n",
            tr_start,N,g_Dk,g_Dv,epochs,ndec,leak);

    long maxrows=N+16;
    float *Xtr=malloc((size_t)maxrows*S_DIM*4); uint32_t *ttr=malloc((size_t)maxrows*4),*ptr=malloc((size_t)maxrows*4);
    float *Xv[N_VAL]; uint32_t *tv[N_VAL],*pv[N_VAL]; double vbytes[N_VAL]; long vrows[N_VAL];
    uint8_t *vrec[N_VAL],*vrare[N_VAL]; int *vdist[N_VAL];
    for(int w=0;w<N_VAL;w++){ Xv[w]=malloc((size_t)maxrows*S_DIM*4); tv[w]=malloc((size_t)maxrows*4); pv[w]=malloc((size_t)maxrows*4);
        vrec[w]=malloc(maxrows); vrare[w]=malloc(maxrows); vdist[w]=malloc((size_t)maxrows*sizeof(int)); }
    double trbytes; long trrows;
    fprintf(stderr,"extract train...\n");
    trrows=extract(&see,tr_start,N,0,Xtr,ttr,ptr,&trbytes,NULL,NULL,NULL);
    for(int w=0;w<N_VAL;w++){ fprintf(stderr,"extract val%d...\n",w+1);
        vrows[w]=extract(&see,va_start[w],N,1,Xv[w],tv[w],pv[w],&vbytes[w],vrec[w],vdist[w],vrare[w]); }
    // recurrence rate report
    for(int w=0;w<N_VAL;w++){ long rc=0,rr=0; for(long i=0;i<vrows[w];i++){ rc+=vrec[w][i]; if(vrec[w][i]&&vrare[w][i])rr++; }
        fprintf(stderr,"val%d rows=%ld recur=%.1f%% recur&rare=%.1f%%\n",w+1,vrows[w],100.0*rc/vrows[w],100.0*rr/vrows[w]); }

    // value-shuffle permutation per window (fixed seed), and store M scratch
    long* perm=malloc((size_t)maxrows*sizeof(long));
    float* M=malloc((size_t)g_Dk*g_Dv*4); if(!M){ fprintf(stderr,"OOM M\n"); return 1; }
    float* Rbuf=malloc((size_t)maxrows*g_Dv*4); if(!Rbuf){ fprintf(stderr,"OOM R\n"); return 1; }

    printf("\n==== 51.B PREDICTIVE store + STRATIFIED gate (Dk=%d Dv=%d H=%d N=%ld) ====\n",g_Dk,g_Dv,HID,N);
    printf("PRE-REGISTERED STRATIFIED GATE: PREDICT WINS iff on RECUR and RECUR&RARE strata\n");
    printf("  (NO-MEM - PREDICT) >= %.3f bits/tok AND (PREDICT-SHUF - PREDICT) >= %.3f bits/tok.\n",DWIN_S,DSHUF_S);
    printf("  Necessary-not-sufficient: the true USE test stays generative (44-49 law).\n\n");

    // ---- NO-MEM (decay-independent) ----
    double nomem_sb[N_VAL][NSTRAT]; long nomem_sc[N_VAL][NSTRAT];
    { fprintf(stderr,"=== NO-MEM ===\n"); Net net; net_init(&net,512,HID,0xA0ULL);
      train_clean(&net,Xtr,NULL,ttr,ptr,trrows,0,epochs,5e-4f);
      for(int w=0;w<N_VAL;w++) eval_strat(&net,Xv[w],NULL,tv[w],pv[w],vrows[w],0,vrec[w],vdist[w],vrare[w],nomem_sb[w],nomem_sc[w]);
      net_free(&net);
    }
    // aggregate NO-MEM strata across windows
    double nm[NSTRAT]; long nmc[NSTRAT];
    for(int st=0;st<NSTRAT;st++){ double b=0; long c=0; for(int w=0;w<N_VAL;w++){ b+=nomem_sb[w][st]; c+=nomem_sc[w][st]; } nm[st]=bpt(b,c); nmc[st]=c; }

    // ---- ORACLE-LEAK ceiling: write-before-predict (leaks the exact next-token atom). NOT a model;
    // it is the USE CEILING = "if retrieval were PERFECT, how much can THIS readout/value even use?".
    // It also proves the metric's leakage-sensitivity. The rigorous no-leakage guarantee is the causal
    // assert in replay() (writes==i-1), which the real PREDICT arm passes.
    double lk[NSTRAT];
    { fprintf(stderr,"=== ORACLE-LEAK ceiling (write-before-predict) ===\n");
      double b[NSTRAT]={0}; long c[NSTRAT]={0};
      Net net; net_init(&net,512+g_Dv,HID,0xDEAD1ULL);
      replay(Xtr,ttr,trrows,0.99f,NULL,1,Rbuf,M);   // leak=1
      train_clean(&net,Xtr,Rbuf,ttr,ptr,trrows,1,epochs,5e-4f);
      for(int w=0;w<N_VAL;w++){ double sb[NSTRAT]; long sc[NSTRAT];
          replay(Xv[w],tv[w],vrows[w],0.99f,NULL,1,Rbuf,M);
          eval_strat(&net,Xv[w],Rbuf,tv[w],pv[w],vrows[w],1,vrec[w],vdist[w],vrare[w],sb,sc);
          for(int st=0;st<NSTRAT;st++){ b[st]+=sb[st]; c[st]+=sc[st]; } }
      net_free(&net); for(int st=0;st<NSTRAT;st++) lk[st]=bpt(b[st],c[st]);
      printf("ORACLE-LEAK ceiling (perfect-retrieval upper bound for this readout/value):\n");
      printf("  ALL=%.4f (d_NOMEM %+.4f) | RECUR=%.4f (%+.4f) | RECUR&RARE=%.4f (%+.4f)\n",
             lk[ST_ALL],nm[ST_ALL]-lk[ST_ALL],lk[ST_RECUR],nm[ST_RECUR]-lk[ST_RECUR],lk[ST_RR],nm[ST_RR]-lk[ST_RR]);
      printf("  (if this ceiling is flat vs NO-MEM, the bottleneck is USE/value-decodability, not ADDRESS.)\n\n");
      (void)leak;
    }

    // ---- decay sweep: PREDICT and PREDICT-SHUF ----
    for(int di=0; di<ndec; di++){
        float dc=decays[di];
        fprintf(stderr,"=== decay %.3f ===\n",dc);
        // PREDICT
        double pr[NSTRAT]; { double b[NSTRAT]={0}; long c[NSTRAT]={0};
            Net net; net_init(&net,512+g_Dv,HID,0xB1ULL+di);
            int okc=replay(Xtr,ttr,trrows,dc,NULL,0,Rbuf,M); if(!okc){ printf("CAUSALITY FAIL (predict train)\n"); return 2; }
            train_clean(&net,Xtr,Rbuf,ttr,ptr,trrows,1,epochs,5e-4f);
            for(int w=0;w<N_VAL;w++){ double sb[NSTRAT]; long sc[NSTRAT];
                replay(Xv[w],tv[w],vrows[w],dc,NULL,0,Rbuf,M);
                eval_strat(&net,Xv[w],Rbuf,tv[w],pv[w],vrows[w],1,vrec[w],vdist[w],vrare[w],sb,sc);
                for(int st=0;st<NSTRAT;st++){ b[st]+=sb[st]; c[st]+=sc[st]; } }
            char sp[512]; snprintf(sp,sizeof sp,"%s_predict_d%02d.bin",argv[4],(int)(dc*100));
            { FILE* f=fopen(sp,"wb"); if(f){ uint32_t mg=0x53454556; fwrite(&mg,4,1,f); fwrite(&dc,4,1,f);
                fwrite(net.W1,4,(size_t)net.H*net.in,f); fwrite(net.b1,4,net.H,f); fwrite(net.W2,4,(size_t)VTOK*net.H,f); fwrite(net.b2,4,VTOK,f); fclose(f);} }
            net_free(&net); for(int st=0;st<NSTRAT;st++) pr[st]=bpt(b[st],c[st]); }
        // PREDICT-SHUF
        double ps[NSTRAT]; { double b[NSTRAT]={0}; long c[NSTRAT]={0};
            Net net; net_init(&net,512+g_Dv,HID,0xC1ULL+di);
            for(long i=0;i<trrows;i++) perm[i]=i;
            { uint64_t s=0x5C0FF1E0ULL^(uint64_t)di; for(long i=trrows-1;i>0;i--){ s^=s<<13;s^=s>>7;s^=s<<17; long j=(long)((s>>11)%(uint64_t)(i+1)); long t=perm[i];perm[i]=perm[j];perm[j]=t; } }
            replay(Xtr,ttr,trrows,dc,perm,0,Rbuf,M);
            train_clean(&net,Xtr,Rbuf,ttr,ptr,trrows,1,epochs,5e-4f);
            for(int w=0;w<N_VAL;w++){ double sb[NSTRAT]; long sc[NSTRAT];
                for(long i=0;i<vrows[w];i++) perm[i]=i;
                { uint64_t s=0x5C0FF1E0ULL^(uint64_t)(di*7+w+1); for(long i=vrows[w]-1;i>0;i--){ s^=s<<13;s^=s>>7;s^=s<<17; long j=(long)((s>>11)%(uint64_t)(i+1)); long t=perm[i];perm[i]=perm[j];perm[j]=t; } }
                replay(Xv[w],tv[w],vrows[w],dc,perm,0,Rbuf,M);
                eval_strat(&net,Xv[w],Rbuf,tv[w],pv[w],vrows[w],1,vrec[w],vdist[w],vrare[w],sb,sc);
                for(int st=0;st<NSTRAT;st++){ b[st]+=sb[st]; c[st]+=sc[st]; } }
            net_free(&net); for(int st=0;st<NSTRAT;st++) ps[st]=bpt(b[st],c[st]); }

        // ---- stratified table for this decay ----
        printf("==== decay=%.3f stratified bits/token (lower=better; count in parens) ====\n",dc);
        printf("  %-12s %10s %10s %10s | %10s %10s\n","stratum","NO-MEM","PREDICT","PRED-SHUF","d_NOMEM","d_SHUF");
        for(int st=0;st<NSTRAT;st++){
            double dN=nm[st]-pr[st], dS=ps[st]-pr[st];
            printf("  %-12s %10.4f %10.4f %10.4f | %+9.4f %+9.4f  (n=%ld)\n",STRN[st],nm[st],pr[st],ps[st],dN,dS,nmc[st]);
        }
        int pass = (nm[ST_RECUR]-pr[ST_RECUR]>=DWIN_S) && (ps[ST_RECUR]-pr[ST_RECUR]>=DSHUF_S)
                && (nm[ST_RR]-pr[ST_RR]>=DWIN_S)       && (ps[ST_RR]-pr[ST_RR]>=DSHUF_S);
        printf("  -> decay=%.3f STRATIFIED GATE: %s (RECUR d_NOMEM=%+.4f d_SHUF=%+.4f | RR d_NOMEM=%+.4f d_SHUF=%+.4f)\n\n",
               dc, pass?"PASS (-> 51.B.A closed-loop)":"fail",
               nm[ST_RECUR]-pr[ST_RECUR], ps[ST_RECUR]-pr[ST_RECUR], nm[ST_RR]-pr[ST_RR], ps[ST_RR]-pr[ST_RR]);
    }
    printf("Reading: bits/TOKEN within each stratum (NOT bits/byte; strata are token-conditioned).\n");
    printf("NO-MEM=discriminant, PRED-SHUF isolates retrieval from width. Flat on RECUR&RARE -> addressing\n");
    printf("broken (random key under-separates contexts) -> learned key-proj or bigger Dk. Flat everywhere\n");
    printf("-> memory not expressible at this scale. A real win earns the GENERATIVE 51.B.A closed-loop.\n");
    return 0;
}
