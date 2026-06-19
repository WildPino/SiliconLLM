// Phase 51.A - associative STORE wired into the token readout (TF-PROBE first).
//
// 51.0 proved an O(1) fixed-key store HOLDS ~200 tokens. 51.A is the first real wiring into the
// model, to test the ADDRESS and USE gates - cheap, TF only, NO DAgger, NO generator. The
// armB substrate stays BYTE-DRIVEN and FROZEN; the store keys are FIXED; only the query-proj Wq
// and the readout (MLP + output) are trained by SGD. No BPTT, no attention module.
//
// Store = fast-weight / linear-attention matrix M[Dk x Dv]:
//     write (per token, Hebbian, param-free):  M <- decay*M + phi(key_i) (x) value_i
//     value_i  = unit atom of the just-seen token (codebook, fixed random)
//     retrieve (before writing token i):        r_i = normalize( M^T q_i ),  q_i = Wq * state_i
//     readout logits = MLP([ state_i | r_i ]) + token-bigram prior
//
// TRIBUNAL (3 arms, same precomputed substrate, only the KEY differs):
//   NO-MEM   : r_i := 0                                   (discriminant: does memory help at all)
//   POS-KEY  : phi(key_i) = cyclic-shift^i(p0)            (the 51.0 positional store)
//   CONTENT  : phi(key_i) = normalize(R * state_i)        (true content-addressing, R fixed random)
// Goodhart control (per mem arm): SHUF = destroy the key<->value correspondence (write each value
//   at a SHUFFLED key) -> same params/width, retrieval made uninformative. If the gain survives
//   SHUF it was readout-width, not retrieval.
//
// Pre-registered TF gate (48.0-style, printed before results; the Architect reads the deltas):
//   a mem arm WINS consideration iff  mean(NO-MEM) - mean(arm) >= DWIN  on >=2/3 windows
//   AND  mean(arm_shuf) - mean(arm) >= DSHUF   (the retrieval, not the width).
//   TF != generative (44-49 law): a win here only earns the closed-loop 51.A.A, not promotion.
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase38-42/phase51_a_store.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase51_a_store.exe -lm -I .
// Run:
//   bin/phase51_a_store.exe <data> <D1_w> <bpe_merges> <outprefix> [--len N] [--dk 1024] [--dv 256]
//                            [--epochs 4] [--decay 0.99] [--smoke]

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
#define S_DIM     (D1_TOT + EXP_BANDS)        // 512 reservoir state row
#define PROJ_SEED 0x48B2EC0DEULL
#define N_VAL     3
#define HID       32

// pre-registered TF gate thresholds (bits per byte)
#define DWIN   0.02
#define DSHUF  0.015

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
static float* g_tbig;     // [VTOK*VTOK] additive log-prob bigram prior

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
static inline double trigram_bpb(uint8_t c1,uint8_t c2,uint8_t tgt){
    const float* tri=&trigram[c2][c1][0]; float mx=-1e30f;
    for(int c=0;c<CLASSES;c++) if(tri[c]>mx)mx=tri[c];
    double Z=0; for(int c=0;c<CLASSES;c++) Z+=exp((double)(tri[c]-mx));
    double p=exp((double)(tri[tgt]-mx))/Z; return -log2(p>1e-30?p:1e-30);
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
    free(uni); free(ctx); free(bg);
}

// ---- frozen substrate extraction (CLEAN only): per token boundary -> state row, target, ctx ----
static long extract_clean(SiliconEntropyState* see, long start, long N,
                          float* X, uint32_t* tgt, uint32_t* prev, double* oBYTES){
    float L2d1[L2_DIM]={0}, pb_d1[BASE_DIM]={0}, eB[N_TS][D_EXP]; memset(eB,0,sizeof eB);
    float feat192[BASE_DIM], fa[BASE_DIM], rawd1[D1_TOT], nf[D1_TOT];
    float scale=1.0f/sqrtf((float)BASE_DIM);
    uint8_t cur_c2,cur_c1;
    long ti=0; { long lo=0,hi=g_ntok; while(lo<hi){ long mid=(lo+hi)/2; if(g_tokstart[mid]<start) lo=mid+1; else hi=mid; } ti=lo; }
    long bstart=g_tokstart[ti];
    see_reset(see); for(long i=0;i<bstart;i++) see_observe(see,g_data[i]);
    cur_c2=(bstart>=2)?g_data[bstart-2]:0; cur_c1=(bstart>=1)?g_data[bstart-1]:0;
    long rows=0, wbytes=0;
    while(ti+1<g_ntok && g_tokstart[ti+1]+ (long)bpe_tok_len(&g_bpe,g_tok[ti+1]) <= start+N){
        see_extract(see,feat192);
        memcpy(rawd1,feat192,BASE_DIM*4); memcpy(rawd1+BASE_DIM,L2d1,L2_DIM*4);
        norm_feats(rawd1,nf);
        float* row=&X[(size_t)rows*S_DIM]; memcpy(row,nf,D1_TOT*4); row_bands(eB,row);
        tgt[rows]=g_tok[ti+1]; prev[rows]=g_tok[ti]; rows++;
        wbytes += bpe_tok_len(&g_bpe,g_tok[ti]);
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
    *oBYTES=(double)wbytes;
    return rows;
}

// ================= associative store + trained query/readout =================
static int g_Dk=1024, g_Dv=256; static float g_decayM=0.99f;
static float* g_cb;    // [VTOK*Dv] unit bipolar value atoms
static float* g_R;     // [Dk*512] content-key projection (+-1)
static float* g_p0;    // [Dk] positional base (unit)

static void store_assets_init(uint64_t seed){
    g_cb=(float*)malloc((size_t)VTOK*g_Dv*4);
    uint64_t s=seed?seed:0x51A0C0DEULL;
    float invv=1.0f/sqrtf((float)g_Dv);
    for(size_t i=0;i<(size_t)VTOK*g_Dv;i++){ s^=s<<13;s^=s>>7;s^=s<<17; g_cb[i]=(s&1ULL)?invv:-invv; }
    g_R=(float*)malloc((size_t)g_Dk*512*4);
    for(size_t i=0;i<(size_t)g_Dk*512;i++){ s^=s<<13;s^=s>>7;s^=s<<17; g_R[i]=(s&1ULL)?1.f:-1.f; }
    g_p0=(float*)malloc((size_t)g_Dk*4); float invk=1.0f/sqrtf((float)g_Dk);
    for(int a=0;a<g_Dk;a++){ s^=s<<13;s^=s>>7;s^=s<<17; g_p0[a]=(s&1ULL)?invk:-invk; }
}
// phi(key) for token at iteration i using source row src512 (content) / shift idx (pos).
// arm: 1=pos, 2=content. writes Dk-vector into phi (unit-normalized for content).
static void key_phi(int arm, long shift_idx, const float* src512, float* phi){
    if(arm==1){ int s=(int)(shift_idx % g_Dk); // cyclic-shift^s(p0): phi[a]=p0[(a-s) mod Dk]
        for(int a=0;a<g_Dk;a++){ int j=a-s; if(j<0) j+=g_Dk; phi[a]=g_p0[j]; } }
    else { for(int a=0;a<g_Dk;a++) phi[a]=dot_avx(&g_R[(size_t)a*512],src512,512);
        float nn=0; for(int a=0;a<g_Dk;a++) nn+=phi[a]*phi[a]; nn=1.0f/(sqrtf(nn)+1e-12f);
        for(int a=0;a<g_Dk;a++) phi[a]*=nn; }
}

typedef struct {
    int in,H; // in = 512 (nomem) or 512+Dv
    float *W1,*b1,*W2,*b2, *mW1,*vW1,*mb1,*vb1,*mW2,*vW2,*mb2,*vb2;
    float *Wq,*mWq,*vWq; // [Dk*512], NULL for nomem
    int t;
} Net;
static void net_init(Net* n,int in,int H,int usemem,uint64_t seed){
    n->in=in; n->H=H; n->t=0;
    size_t s1=(size_t)H*in, s2=(size_t)VTOK*H;
    n->W1=calloc(s1,4); n->b1=calloc(H,4); n->W2=calloc(s2,4); n->b2=calloc(VTOK,4);
    n->mW1=calloc(s1,4); n->vW1=calloc(s1,4); n->mb1=calloc(H,4); n->vb1=calloc(H,4);
    n->mW2=calloc(s2,4); n->vW2=calloc(s2,4); n->mb2=calloc(VTOK,4); n->vb2=calloc(VTOK,4);
    uint64_t r=seed?seed:0x1234567ULL; float sc1=sqrtf(2.0f/in);
    for(size_t i=0;i<s1;i++){ r^=r<<13;r^=r>>7;r^=r<<17; n->W1[i]=sc1*(((r>>11)*(1.0/(1ULL<<53)))*2-1); }
    float sc2=sqrtf(2.0f/H);
    for(size_t i=0;i<s2;i++){ r^=r<<13;r^=r>>7;r^=r<<17; n->W2[i]=sc2*(((r>>11)*(1.0/(1ULL<<53)))*2-1); }
    if(usemem){ size_t sq=(size_t)g_Dk*512; n->Wq=calloc(sq,4); n->mWq=calloc(sq,4); n->vWq=calloc(sq,4);
        float scq=sqrtf(1.0f/512.0f);
        for(size_t i=0;i<sq;i++){ r^=r<<13;r^=r>>7;r^=r<<17; n->Wq[i]=scq*(((r>>11)*(1.0/(1ULL<<53)))*2-1); } }
    else { n->Wq=n->mWq=n->vWq=NULL; }
}
static void net_free(Net* n){ free(n->W1);free(n->b1);free(n->W2);free(n->b2);
    free(n->mW1);free(n->vW1);free(n->mb1);free(n->vb1);free(n->mW2);free(n->vW2);free(n->mb2);free(n->vb2);
    if(n->Wq){ free(n->Wq);free(n->mWq);free(n->vWq);} }

// retrieve r = normalize(M^T q), q = Wq*state. returns r (Dv), q (Dk), rnorm (pre-norm L2).
static void retrieve(const Net* n,const float* M,const float* state,float* q,float* r,float* rnorm){
    for(int a=0;a<g_Dk;a++) q[a]=dot_avx(&n->Wq[(size_t)a*512],state,512);
    for(int b=0;b<g_Dv;b++) r[b]=0.f;
    for(int a=0;a<g_Dk;a++){ float qa=q[a]; if(qa!=0.f) axpy_avx(r,&M[(size_t)a*g_Dv],qa,g_Dv); }
    float nn=0; for(int b=0;b<g_Dv;b++) nn+=r[b]*r[b]; *rnorm=sqrtf(nn);
    float inv=1.0f/(*rnorm+1e-12f); for(int b=0;b<g_Dv;b++) r[b]*=inv;
}
// One full TF pass over a sequence: arm/shuf, optional training. Returns mean bits/byte if eval.
// perm: shuffled token-index map (used only when shuf!=0).
typedef struct { float *gW1,*gb1,*gW2,*gb2,*gWq; } G;
static double run_pass(Net* n,int arm,int shuf,const long* perm,
                       const float* X,const uint32_t* tgt,const uint32_t* prev,long nrow,
                       float* M, int train, double winbytes, float lr){
    int in=n->in,H=n->H, mem=(arm!=0);
    float *q=mem?malloc((size_t)g_Dk*4):NULL, *r=mem?malloc((size_t)g_Dv*4):NULL;
    float *phi=mem?malloc((size_t)g_Dk*4):NULL, *v;
    float *u=malloc((size_t)in*4), *hid=malloc(H*4), *dh=malloc(H*4);
    float *lg=malloc((size_t)VTOK*4), *eo=malloc((size_t)VTOK*4), *du=malloc((size_t)in*4);
    float *dLdr=mem?malloc((size_t)g_Dv*4):NULL, *dLdq=mem?malloc((size_t)g_Dk*4):NULL;
    G g; size_t s1=(size_t)H*in,s2=(size_t)VTOK*H,sq=(size_t)g_Dk*512;
    if(train){ g.gW1=calloc(s1,4);g.gb1=calloc(H,4);g.gW2=calloc(s2,4);g.gb2=calloc(VTOK,4); g.gWq=mem?calloc(sq,4):NULL; }
    int bs=256; float invn=1.0f/bs; long inb=0; double tot=0;
    // reset store
    memset(M,0,(size_t)g_Dk*g_Dv*4);
    for(long i=0;i<nrow;i++){
        const float* state=&X[(size_t)i*S_DIM];
        float rnorm=0;
        if(mem){ retrieve(n,M,state,q,r,&rnorm);
            memcpy(u,state,512*4); memcpy(u+512,r,g_Dv*4); }
        else memcpy(u,state,512*4);
        // forward MLP
        for(int j=0;j<H;j++){ float a=n->b1[j]+dot_avx(&n->W1[(size_t)j*in],u,in); hid[j]=a>0?a:0; }
        for(int c=0;c<VTOK;c++) lg[c]=n->b2[c]+dot_avx(&n->W2[(size_t)c*H],hid,H);
        const float* tb=&g_tbig[(size_t)prev[i]*VTOK]; for(int c=0;c<VTOK;c++) lg[c]+=tb[c];
        float mx=-1e30f; for(int c=0;c<VTOK;c++) if(lg[c]>mx)mx=lg[c];
        double Z=0; for(int c=0;c<VTOK;c++){ eo[c]=expf(lg[c]-mx); Z+=eo[c]; }
        double p=eo[tgt[i]]/Z; tot += -log2(p>1e-30?p:1e-30);
        if(train){
            for(int c=0;c<VTOK;c++){ float y=(c==(int)tgt[i])?1.f:0.f; eo[c]=(float)(eo[c]/Z - y)*invn; }
            for(int c=0;c<VTOK;c++) g.gb2[c]+=eo[c];
            memset(dh,0,H*4);
            for(int c=0;c<VTOK;c++){ float e=eo[c]; float* gw=&g.gW2[(size_t)c*H]; const float* w2=&n->W2[(size_t)c*H];
                for(int j=0;j<H;j++){ gw[j]+=e*hid[j]; dh[j]+=e*w2[j]; } }
            memset(du,0,in*4);
            for(int j=0;j<H;j++) if(hid[j]>0){ g.gb1[j]+=dh[j]; float gg=dh[j];
                float* gw=&g.gW1[(size_t)j*in]; const float* w1=&n->W1[(size_t)j*in];
                for(int k=0;k<in;k++){ gw[k]+=gg*u[k]; du[k]+=gg*w1[k]; } }
            if(mem){ // backprop through r-normalize -> q -> Wq
                const float* dur=du+512; // dL/d rn
                float dot=0; for(int b=0;b<g_Dv;b++) dot+=dur[b]*r[b]; // r is normalized here
                float inv=1.0f/(rnorm+1e-12f);
                for(int b=0;b<g_Dv;b++) dLdr[b]=(dur[b]-dot*r[b])*inv;
                for(int a=0;a<g_Dk;a++) dLdq[a]=dot_avx(&M[(size_t)a*g_Dv],dLdr,g_Dv);
                for(int a=0;a<g_Dk;a++){ float dq=dLdq[a]; if(dq!=0.f){ float* gw=&g.gWq[(size_t)a*512]; axpy_avx(gw,state,dq,512);} }
            }
            inb++;
            if(inb==bs || i==nrow-1){ n->t++; float lt=lr*sqrtf(1-powf(.999f,n->t))/(1-powf(.9f,n->t));
                #define ADAM(P,GG,MM,VV,NN) for(size_t z=0;z<(size_t)(NN);z++){ MM[z]=.9f*MM[z]+.1f*GG[z]; VV[z]=.999f*VV[z]+.001f*GG[z]*GG[z]; P[z]-=lt*(MM[z]/(sqrtf(VV[z])+1e-8f)+1e-5f*P[z]); }
                ADAM(n->W1,g.gW1,n->mW1,n->vW1,s1); ADAM(n->b1,g.gb1,n->mb1,n->vb1,H);
                ADAM(n->W2,g.gW2,n->mW2,n->vW2,s2); ADAM(n->b2,g.gb2,n->mb2,n->vb2,VTOK);
                if(mem){ ADAM(n->Wq,g.gWq,n->mWq,n->vWq,sq); }
                memset(g.gW1,0,s1*4);memset(g.gb1,0,H*4);memset(g.gW2,0,s2*4);memset(g.gb2,0,VTOK*4);
                if(mem) memset(g.gWq,0,sq*4); inb=0; }
        }
        // write token i into M (decay then add phi(key)(x)v). value = atom(prev[i]).
        if(mem){
            long src_i = (shuf&&perm)?perm[i]:i;
            const float* src_state = (arm==2)? &X[(size_t)src_i*S_DIM] : NULL;
            long shift = (arm==1)? src_i : i;
            key_phi(arm, shift, src_state, phi);
            v=&g_cb[(size_t)prev[i]*g_Dv];
            scal_avx(M,g_decayM,(int)((size_t)g_Dk*g_Dv));
            for(int a=0;a<g_Dk;a++){ float c=phi[a]; if(c!=0.f) axpy_avx(&M[(size_t)a*g_Dv],v,c,g_Dv); }
        }
    }
    if(train){ free(g.gW1);free(g.gb1);free(g.gW2);free(g.gb2); if(g.gWq)free(g.gWq); }
    if(mem){free(q);free(r);free(phi);free(dLdr);free(dLdq);}
    free(u);free(hid);free(dh);free(lg);free(eo);free(du);
    return tot/winbytes; // bits per byte (only meaningful when called as eval)
}

static void save_ckpt(const Net* n,const char* path,int arm){
    FILE* f=fopen(path,"wb"); if(!f){ fprintf(stderr,"save %s fail\n",path); return; }
    uint32_t magic=0x53454555, H=(uint32_t)n->H, in=(uint32_t)n->in, dk=(uint32_t)g_Dk, dv=(uint32_t)g_Dv, a=(uint32_t)arm;
    float decay=g_decayM; uint32_t vt=(uint32_t)VTOK, nm=(uint32_t)g_bpe.nmerge;
    fwrite(&magic,4,1,f); fwrite(&a,4,1,f); fwrite(&H,4,1,f); fwrite(&in,4,1,f); fwrite(&dk,4,1,f); fwrite(&dv,4,1,f);
    fwrite(&decay,4,1,f); fwrite(&vt,4,1,f); fwrite(&nm,4,1,f);
    for(int r=0;r<g_bpe.nmerge;r++){ fwrite(&g_bpe.mA[r],4,1,f); fwrite(&g_bpe.mB[r],4,1,f); }
    fwrite(g_tbig,4,(size_t)VTOK*VTOK,f);
    fwrite(n->W1,4,(size_t)n->H*n->in,f); fwrite(n->b1,4,n->H,f);
    fwrite(n->W2,4,(size_t)VTOK*n->H,f); fwrite(n->b2,4,VTOK,f);
    if(n->Wq) fwrite(n->Wq,4,(size_t)g_Dk*512,f);
    fclose(f);
}

int main(int argc,char** argv){
    if(argc<5){ fprintf(stderr,"Usage: %s <data> <D1_w> <bpe_merges> <outprefix> [--len N] [--dk D] [--dv D] [--epochs E] [--decay d] [--smoke]\n",argv[0]); return 1; }
    setvbuf(stderr,NULL,_IONBF,0); setvbuf(stdout,NULL,_IONBF,0);
    long N=300000, maxb=0; int epochs=4, smoke=0;
    for(int i=5;i<argc;i++){
        if(!strcmp(argv[i],"--len")&&i+1<argc) N=atol(argv[++i]);
        else if(!strcmp(argv[i],"--dk")&&i+1<argc) g_Dk=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--dv")&&i+1<argc) g_Dv=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--epochs")&&i+1<argc) epochs=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--decay")&&i+1<argc) g_decayM=atof(argv[++i]);
        else if(!strcmp(argv[i],"--max-bytes")&&i+1<argc) maxb=atol(argv[++i]);
        else if(!strcmp(argv[i],"--smoke")){ smoke=1; N=20000; g_Dk=256; g_Dv=64; epochs=1; }
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
    fprintf(stderr,"building token-bigram prior (train tokens %ld)...\n",tok_train_end);
    build_tbig(tok_train_end);
    store_assets_init(0x51A0C0DEULL);

    long tr_start=g_fsz/5;
    long va_start[N_VAL]={ g_fsz/2, (long)(0.65*g_fsz), (long)(0.80*g_fsz) };
    for(int w=0;w<N_VAL;w++) if(va_start[w]+N+3>g_fsz){ fprintf(stderr,"val window %d out of file\n",w+1); return 1; }
    fprintf(stderr,"51.A store: train@%ld val@%ld/%ld/%ld N=%ld Dk=%d Dv=%d decay=%.3f epochs=%d\n",
            tr_start,va_start[0],va_start[1],va_start[2],N,g_Dk,g_Dv,g_decayM,epochs);

    long maxrows=N+16;
    float *Xtr=malloc((size_t)maxrows*S_DIM*4); uint32_t *ttr=malloc((size_t)maxrows*4),*ptr=malloc((size_t)maxrows*4);
    float *Xv[N_VAL]; uint32_t *tv[N_VAL],*pv[N_VAL]; double vbytes[N_VAL]; long vrows[N_VAL];
    for(int w=0;w<N_VAL;w++){ Xv[w]=malloc((size_t)maxrows*S_DIM*4); tv[w]=malloc((size_t)maxrows*4); pv[w]=malloc((size_t)maxrows*4); }
    if(!Xtr){ fprintf(stderr,"OOM\n"); return 1; }
    double trbytes; long trrows;
    fprintf(stderr,"extract train...\n");
    trrows=extract_clean(&see,tr_start,N,Xtr,ttr,ptr,&trbytes);
    for(int w=0;w<N_VAL;w++){ fprintf(stderr,"extract val%d...\n",w+1);
        vrows[w]=extract_clean(&see,va_start[w],N,Xv[w],tv[w],pv[w],&vbytes[w]); }
    fprintf(stderr,"rows: train=%ld val=%ld/%ld/%ld\n",trrows,vrows[0],vrows[1],vrows[2]);

    // shuffled token-index permutation (for SHUF controls), fixed seed
    long* perm=malloc((size_t)maxrows*sizeof(long)); for(long i=0;i<trrows;i++) perm[i]=i;
    { uint64_t s=0x5C0FF1E0ULL; for(long i=trrows-1;i>0;i--){ s^=s<<13;s^=s>>7;s^=s<<17; long j=(long)((s>>11)%(uint64_t)(i+1)); long t=perm[i];perm[i]=perm[j];perm[j]=t; } }

    float* M=malloc((size_t)g_Dk*g_Dv*4); if(!M){ fprintf(stderr,"OOM M\n"); return 1; }

    printf("\n==== 51.A associative-store TF probe (Dk=%d Dv=%d decay=%.3f H=%d N=%ld) ====\n",g_Dk,g_Dv,g_decayM,HID,N);
    printf("PRE-REGISTERED TF GATE: mem arm WINS iff (NO-MEM - arm) >= %.3f on >=2/3 windows AND (arm_shuf - arm) >= %.3f.\n",DWIN,DSHUF);
    printf("TF != generative (44-49 law). A win earns 51.A.A closed-loop, not promotion.\n\n");

    // token-bigram-only anchor on each val
    for(int w=0;w<N_VAL;w++){ double tot=0; for(long i=0;i<vrows[w];i++){ const float* tb=&g_tbig[(size_t)pv[w][i]*VTOK];
        float mx=-1e30f; for(int c=0;c<VTOK;c++) if(tb[c]>mx)mx=tb[c]; double Z=0; for(int c=0;c<VTOK;c++) Z+=exp((double)(tb[c]-mx));
        double p=exp((double)(tb[tv[w][i]]-mx))/Z; tot+=-log2(p>1e-30?p:1e-30); }
        printf("ANCHOR tbigOnly val%d=%.4f bits/byte\n",w+1,tot/vbytes[w]); }
    printf("\n");

    struct { const char* name; int arm; int shuf; } arms[5] = {
        {"NO-MEM",0,0},{"POS",1,0},{"CONTENT",2,0},{"POS-SHUF",1,1},{"CONTENT-SHUF",2,1} };
    double meanbpb[5], vb[5][N_VAL];
    for(int ai=0; ai<5; ai++){
        int arm=arms[ai].arm, shuf=arms[ai].shuf, mem=(arm!=0);
        int in = mem? (512+g_Dv) : 512;
        Net net; net_init(&net,in,HID,mem, 0xA1B2C300ULL+ai);
        fprintf(stderr,"=== arm %s (in=%d) ===\n",arms[ai].name,in);
        for(int ep=0;ep<epochs;ep++){ run_pass(&net,arm,shuf,perm, Xtr,ttr,ptr,trrows, M,1,trbytes,5e-4f);
            fprintf(stderr,"   %s ep %d/%d\n",arms[ai].name,ep+1,epochs); }
        double mb=0; for(int w=0;w<N_VAL;w++){ vb[ai][w]=run_pass(&net,arm,shuf,perm, Xv[w],tv[w],pv[w],vrows[w], M,0,vbytes[w],0); mb+=vb[ai][w]; }
        meanbpb[ai]=mb/N_VAL;
        printf("PROBE arm=%-12s val1=%.4f val2=%.4f val3=%.4f mean=%.4f\n",arms[ai].name,vb[ai][0],vb[ai][1],vb[ai][2],meanbpb[ai]);
        char sp[512]; snprintf(sp,sizeof sp,"%s_%s.bin",argv[4],arms[ai].name);
        save_ckpt(&net,sp,arm);
        net_free(&net);
    }

    // ---- pre-registered verdict ----
    printf("\n==== TF GATE VERDICT (pre-registered DWIN=%.3f DSHUF=%.3f) ====\n",DWIN,DSHUF);
    double nomem=meanbpb[0];
    int idx[2]={1,2}, shufidx[2]={3,4}; const char* nm[2]={"POS","CONTENT"};
    int any=0;
    for(int t=0;t<2;t++){ int a=idx[t], s=shufidx[t];
        int winw=0; for(int w=0;w<N_VAL;w++) if((vb[0][w]-vb[a][w])>=DWIN) winw++;
        double dwin=nomem-meanbpb[a], dshuf=meanbpb[s]-meanbpb[a];
        int pass = (winw>=2) && (dshuf>=DSHUF);
        printf("  %-8s mean=%.4f  d_vs_NOMEM=%+.4f (win %d/3 windows)  d_vs_SHUF=%+.4f  -> %s\n",
               nm[t],meanbpb[a],dwin,winw,dshuf, pass?"PASS (-> 51.A.A closed-loop)":"fail");
        if(pass) any=1;
    }
    if(!any) printf("  NO arm passes: NO-MEM ~ POS ~ CONTENT -> memory is a TF no-op; rethink query/value (state-as-value vs token-as-value) per the pre-registered tree.\n");
    printf("\nReading: bits/byte (unit-invariant). NO-MEM is the discriminant; SHUF isolates retrieval from readout-width.\n");
    return 0;
}
