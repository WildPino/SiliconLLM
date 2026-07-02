// Phase 53.0 - is the long-range SEMANTIC signal IN THE TRAJECTORY? (supervised probe, cheap, no module)
//
// The 51->52 chain sealed the READ side: fixed-key ~3%, learned-key +0.005, token-context-key = a richer
// n-gram in disguise (52.C.A blind: GLOBAL eats the whole medium-range gain, margin -0.0467). No addressing
// scheme finds long-range structure that isn't already n-gram-able. The remaining question is upstream of
// addressing: does the COMPRESSED TRAJECTORY of the FROZEN substrate even CARRY decodable long-range
// semantics? If a small supervised probe can't read it out, no read-side module can -> rigorous STOP.
//
// INPUT (compressed history, all FIXED / NOT trained):
//   state_t (512D)  +  multiscale EMA of states {a=0.9,0.99,0.999} (3x512)  +  fixed HiPPO-surrogate memory
//   (DH fixed linear leaky integrators h_d = lam_d*h_d + (1-lam_d)*(w_d . state), fixed random w_d, lam_d
//    log-spaced in [0.9,0.9995]).  TRAJ_DIM = 512 + 3*512 + DH.   Nothing here is learned.
//
// TARGETS (supervised, predict the FAR future [t+200, t+400] in token space):
//   HEAD-A  topic/content presence: for K content tokens (freq rank band, function words skipped), does
//           each appear in the far window?  multi-label.  covers (a) story/topic + (c) far content dist.
//   HEAD-B  entity recurrence: the most-recent content token in [t-32,t] -> does it reappear in [t+200,t+400]?
//
// MANTRA CONTROLS (the line made test):
//   FULL      trajectory (state + EMA + HiPPO)                        <- the hypothesis
//   POINT     state_t only (no compressed history)                    <- does the trajectory beat one state?
//   TOKONLY   substrate ablated: fixed VSA embed of last m token-ids  <- is it the substrate or just tokens?
//   SCRAMBLE  FULL but state stream time-shuffled                     <- must collapse to ~base rate
//   BLIND     per-label recurrence logistic (recent-occurrence stats) <- the n-gram-able floor to beat
//
// METRIC: info = baseline_BCE(predict train marginal p_k) - model_BCE, in bits/label on held-out windows.
//         positive = decodable beyond the marginal. Higher = more long-range semantics readable.
//
// DECISION:  FULL >> TOKONLY  AND  FULL >> SCRAMBLE  AND  FULL > POINT  AND  FULL > BLIND
//            -> signal present -> 53.A predictive bottleneck.   Otherwise -> STOP (substrate doesn't carry it;
//            changing the substrate is the forbidden zone, so this is a hard, honest stop).
//
// DISCIPLINE: one trainer, NEW magic 0x5345455C, generator armB UNTOUCHED (this never feeds generation),
//   substrate READ-ONLY (no grad into it), smoke tiny, no commit. STOP + report per target per control.
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase52/phase53_0_probe.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase53_0_probe.exe -lm -I .
// Run:
//   bin/phase53_0_probe.exe <data> <D1_w> <bpe_merges> <outprefix> [--len N] [--epochs E] [--smoke]
//
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
#define S_DIM     (D1_TOT + EXP_BANDS)     // = 512
#define PROJ_SEED 0x48B2EC0DEULL
#define N_VAL     2

// ---- probe / trajectory config ----
#define EMA_NS    3
static const float EMA_A[EMA_NS] = { 0.9f, 0.99f, 0.999f };
#define DH        128
#define TRAJ_DIM  (S_DIM + EMA_NS*S_DIM + DH)   // 512 + 1536 + 128 = 2176
#define KCONT     128       // # content-token labels (HEAD-A)
#define FUNC_SKIP 24        // skip the FUNC_SKIP most frequent tokens (function words)
#define FAR_LO    200
#define FAR_HI    400
#define RECENT    32        // window for "most recent content token" (HEAD-B)
#define BLIND_REC 200       // recent-occurrence window for BLIND features
#define DE_TOK    256       // tokonly VSA embedding dim
#define M_TOK     8         // tokonly: last M token-ids bound
#define HID       64        // probe hidden width

#define RARE_P   1e-4

enum { ST_ALL=0, ST_RECUR, ST_RR, ST_NONREC, ST_D1, ST_D2, ST_D3, ST_D4, NSTRAT };

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
static void build_unigram(long tok_train_end){
    g_uni=(double*)calloc(VTOK,sizeof(double)); g_Ttrain=(double)tok_train_end;
    for(long i=0;i<tok_train_end;i++) g_uni[g_tok[i]]+=1.0;
}

// ---- extract: state rows + token at row (prev) + next token (tgt) ----  (identical regime to 52.C)
static long extract(SiliconEntropyState* see, long start, long N, float* X, uint32_t* tgt, uint32_t* prev){
    float L2d1[L2_DIM]={0}, pb_d1[BASE_DIM]={0}, eB[N_TS][D_EXP]; memset(eB,0,sizeof eB);
    float feat192[BASE_DIM], fa[BASE_DIM], rawd1[D1_TOT], nf[D1_TOT];
    float scale=1.0f/sqrtf((float)BASE_DIM); uint8_t cur_c2,cur_c1;
    long ti=0; { long lo=0,hi=g_ntok; while(lo<hi){ long mid=(lo+hi)/2; if(g_tokstart[mid]<start) lo=mid+1; else hi=mid; } ti=lo; }
    long bstart=g_tokstart[ti];
    see_reset(see); for(long i=0;i<bstart;i++) see_observe(see,g_data[i]);
    cur_c2=(bstart>=2)?g_data[bstart-2]:0; cur_c1=(bstart>=1)?g_data[bstart-1]:0;
    long rows=0;
    while(ti+1<g_ntok && g_tokstart[ti+1]+ (long)bpe_tok_len(&g_bpe,g_tok[ti+1]) <= start+N){
        see_extract(see,feat192);
        memcpy(rawd1,feat192,BASE_DIM*4); memcpy(rawd1+BASE_DIM,L2d1,L2_DIM*4);
        norm_feats(rawd1,nf);
        float* row=&X[(size_t)rows*S_DIM]; memcpy(row,nf,D1_TOT*4); row_bands(eB,row);
        tgt[rows]=g_tok[ti+1]; prev[rows]=g_tok[ti];
        rows++;
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
    return rows;
}

// ============ content-label map + fixed trajectory/tokonly assets ============
static int g_tok2lab[1<<20];   // VTOK <= 1M; -1 if not a content token
static uint32_t g_lab2tok[KCONT];
static float Hw[DH][S_DIM]; static float Hlam[DH];
static float* g_E2;  // [VTOK*DE_TOK] fixed bipolar token embeddings for TOKONLY

static int cmp_freq(const void* a,const void* b){ int x=*(const int*)a,y=*(const int*)b;
    double da=g_uni[x],db=g_uni[y]; return (da<db)-(da>db); }
static void build_content_labels(void){
    int* idx=malloc((size_t)VTOK*sizeof(int)); for(int v=0;v<VTOK;v++) idx[v]=v;
    qsort(idx,VTOK,sizeof(int),cmp_freq);
    for(int v=0;v<VTOK;v++) g_tok2lab[v]=-1;
    int got=0;
    for(int r=FUNC_SKIP; r<VTOK && got<KCONT; r++){ int v=idx[r]; if(g_uni[v]<=0) continue;
        g_tok2lab[v]=got; g_lab2tok[got]=v; got++; }
    fprintf(stderr,"content labels: %d (freq rank %d..%d)\n",got,FUNC_SKIP,FUNC_SKIP+got);
    free(idx);
}
static void hippo_init(uint64_t seed){ uint64_t s=seed?seed:0x4190C0DEULL;
    for(int d=0;d<DH;d++){ float nn=0;
        for(int k=0;k<S_DIM;k++){ s^=s<<13;s^=s>>7;s^=s<<17; Hw[d][k]=(s&1ULL)?1.f:-1.f; nn+=1.0f; }
        float inv=1.0f/sqrtf(nn); for(int k=0;k<S_DIM;k++) Hw[d][k]*=inv;
        double f=(DH>1)?(double)d/(DH-1):0.0; Hlam[d]=(float)exp(log(0.9)*(1.0-f)+log(0.9995)*f); }
}
static void tokemb_init(uint64_t seed){ g_E2=malloc((size_t)VTOK*DE_TOK*4); uint64_t s=seed?seed:0xE2C0DEULL;
    for(size_t i=0;i<(size_t)VTOK*DE_TOK;i++){ s^=s<<13;s^=s>>7;s^=s<<17; g_E2[i]=(s&1ULL)?1.f:-1.f; } }

// build the FULL/SCRAMBLE trajectory rep. order=NULL -> causal identity; else state stream consumed in
// order[] (time-shuffled) while labels stay at position ridx -> SCRAMBLE control.
static void build_traj(const float* X, long nrow, const long* order, float* T){
    float ema[EMA_NS][S_DIM]; memset(ema,0,sizeof ema);
    float h[DH]; memset(h,0,sizeof h);
    for(long ridx=0; ridx<nrow; ridx++){
        long r = order? order[ridx] : ridx;
        const float* st=&X[(size_t)r*S_DIM];
        for(int s=0;s<EMA_NS;s++){ float a=EMA_A[s]; float* e=ema[s];
            for(int k=0;k<S_DIM;k++) e[k]=a*e[k]+(1.0f-a)*st[k]; }
        for(int d=0;d<DH;d++){ float proj=dot_avx(Hw[d],st,S_DIM); h[d]=Hlam[d]*h[d]+(1.0f-Hlam[d])*proj; }
        float* o=&T[(size_t)ridx*TRAJ_DIM];
        memcpy(o, st, S_DIM*4);
        for(int s=0;s<EMA_NS;s++) memcpy(o+S_DIM+(size_t)s*S_DIM, ema[s], S_DIM*4);
        memcpy(o+S_DIM+(size_t)EMA_NS*S_DIM, h, DH*4);
    }
}
static void build_point(const float* X, long nrow, float* U){
    for(long r=0;r<nrow;r++) memcpy(&U[(size_t)r*S_DIM],&X[(size_t)r*S_DIM],S_DIM*4);
}
// TOKONLY: substrate ABLATED. rep = normalize( sum_{j<M} cyclic_shift_j( E2[prev_{r-j}] ) )
static void build_tokonly(const uint32_t* prev, long nrow, float* U){
    for(long r=0;r<nrow;r++){ float* a=&U[(size_t)r*DE_TOK]; for(int d=0;d<DE_TOK;d++) a[d]=0.f;
        for(int j=0;j<M_TOK && r-j>=0; j++){ const float* e=&g_E2[(size_t)prev[r-j]*DE_TOK];
            for(int d=0;d<DE_TOK;d++){ int dd=d+j; if(dd>=DE_TOK) dd-=DE_TOK; a[dd]+=e[d]; } }
        float nn=0; for(int d=0;d<DE_TOK;d++) nn+=a[d]*a[d]; float inv=1.0f/(sqrtf(nn)+1e-12f);
        for(int d=0;d<DE_TOK;d++) a[d]*=inv; }
}

// ============ labels ============
// usable rows: r in [0, nrow-FAR_HI).  yA[r*KCONT], yB[r], mB[r] (1=row has a HEAD-B target)
static long build_labels(const uint32_t* prev, long nrow, uint8_t* yA, uint8_t* yB, uint8_t* mB){
    long used = nrow - FAR_HI; if(used<0) used=0;
    for(long r=0;r<used;r++){
        uint8_t* ya=&yA[(size_t)r*KCONT]; for(int k=0;k<KCONT;k++) ya[k]=0;
        for(long j=r+FAR_LO; j<=r+FAR_HI; j++){ int lab=g_tok2lab[prev[j]]; if(lab>=0) ya[lab]=1; }
        // HEAD-B: most recent content token in [r-RECENT, r]
        int rctok=-1; for(long j=(r>=RECENT?r-RECENT:0); j<=r; j++){ int lab=g_tok2lab[prev[j]]; if(lab>=0) rctok=(int)prev[j]; }
        if(rctok<0){ mB[r]=0; yB[r]=0; }
        else { mB[r]=1; int pres=0; for(long j=r+FAR_LO; j<=r+FAR_HI; j++) if((int)prev[j]==rctok){ pres=1; break; } yB[r]=(uint8_t)pres; }
    }
    return used;
}

// ============ probe: trunk H -> headA (KCONT sigmoid) + headB (1 sigmoid) ============
typedef struct { int in,H; float *W1,*b1,*WA,*bA,*Wb; float bb;
    float *mW1,*vW1,*mb1,*vb1,*mWA,*vWA,*mbA,*vbA,*mWb,*vWb; float mbb,vbb; int t; } Probe;
static void probe_init(Probe* p,int in,uint64_t seed){ p->in=in; p->H=HID; p->t=0;
    size_t s1=(size_t)HID*in, sA=(size_t)KCONT*HID;
    p->W1=calloc(s1,4);p->b1=calloc(HID,4);p->WA=calloc(sA,4);p->bA=calloc(KCONT,4);p->Wb=calloc(HID,4);p->bb=0;
    p->mW1=calloc(s1,4);p->vW1=calloc(s1,4);p->mb1=calloc(HID,4);p->vb1=calloc(HID,4);
    p->mWA=calloc(sA,4);p->vWA=calloc(sA,4);p->mbA=calloc(KCONT,4);p->vbA=calloc(KCONT,4);
    p->mWb=calloc(HID,4);p->vWb=calloc(HID,4);p->mbb=0;p->vbb=0;
    uint64_t r=seed?seed:0x53ULL; float sc1=sqrtf(2.0f/in);
    for(size_t i=0;i<s1;i++){ r^=r<<13;r^=r>>7;r^=r<<17; p->W1[i]=sc1*(((r>>11)*(1.0/(1ULL<<53)))*2-1); }
    float scA=sqrtf(2.0f/HID);
    for(size_t i=0;i<sA;i++){ r^=r<<13;r^=r>>7;r^=r<<17; p->WA[i]=scA*(((r>>11)*(1.0/(1ULL<<53)))*2-1); }
    for(int j=0;j<HID;j++){ r^=r<<13;r^=r>>7;r^=r<<17; p->Wb[j]=scA*(((r>>11)*(1.0/(1ULL<<53)))*2-1); }
}
static void probe_free(Probe* p){ free(p->W1);free(p->b1);free(p->WA);free(p->bA);free(p->Wb);
    free(p->mW1);free(p->vW1);free(p->mb1);free(p->vb1);free(p->mWA);free(p->vWA);free(p->mbA);free(p->vbA);free(p->mWb);free(p->vWb); }
static inline void probe_fwd(const Probe* p,const float* u,float* hid,float* la,float* lb){
    for(int j=0;j<p->H;j++){ float a=p->b1[j]+dot_avx(&p->W1[(size_t)j*p->in],u,p->in); hid[j]=a>0?a:0; }
    for(int k=0;k<KCONT;k++) la[k]=p->bA[k]+dot_avx(&p->WA[(size_t)k*p->H],hid,p->H);
    *lb=p->bb+dot_avx(p->Wb,hid,p->H);
}
static inline float sigf(float x){ if(x>30)return 1.f; if(x<-30)return 0.f; return 1.0f/(1.0f+expf(-x)); }
#define ADAMP(P,GG,MM,VV,NN) for(size_t z=0;z<(size_t)(NN);z++){ MM[z]=.9f*MM[z]+.1f*GG[z]; VV[z]=.999f*VV[z]+.001f*GG[z]*GG[z]; P[z]-=lt*(MM[z]/(sqrtf(VV[z])+1e-8f)+1e-5f*P[z]); }
static void probe_train(Probe* p,const float* U,const uint8_t* yA,const uint8_t* yB,const uint8_t* mB,
                        long used,int epochs,float lr){
    int H=p->H,in=p->in; size_t s1=(size_t)H*in, sA=(size_t)KCONT*H;
    float *gW1=malloc(s1*4),*gb1=malloc(H*4),*gWA=malloc(sA*4),*gbA=malloc(KCONT*4),*gWb=malloc(H*4),gbb;
    float *hid=malloc(H*4),*dh=malloc(H*4),*la=malloc(KCONT*4); float lb;
    int bs=256; float invn=1.0f/bs; float lamB=(float)KCONT*0.25f; // weight HEAD-B so it isn't drowned by K labels
    for(int ep=0;ep<epochs;ep++){
        memset(gW1,0,s1*4);memset(gb1,0,H*4);memset(gWA,0,sA*4);memset(gbA,0,KCONT*4);memset(gWb,0,H*4);gbb=0; long inb=0;
        for(long i=0;i<used;i++){
            const float* u=&U[(size_t)i*in];
            probe_fwd(p,u,hid,la,&lb);
            memset(dh,0,H*4);
            const uint8_t* ya=&yA[(size_t)i*KCONT];
            for(int k=0;k<KCONT;k++){ float pr=sigf(la[k]); float e=(pr-ya[k])*invn;
                gbA[k]+=e; float* gw=&gWA[(size_t)k*H]; const float* w=&p->WA[(size_t)k*H];
                for(int j=0;j<H;j++){ gw[j]+=e*hid[j]; dh[j]+=e*w[j]; } }
            if(mB[i]){ float pr=sigf(lb); float e=(pr-yB[i])*invn*lamB;
                gbb+=e; for(int j=0;j<H;j++){ gWb[j]+=e*hid[j]; dh[j]+=e*p->Wb[j]; } }
            for(int j=0;j<H;j++) if(hid[j]>0){ gb1[j]+=dh[j]; float* gw=&gW1[(size_t)j*in]; const float g=dh[j];
                for(int k=0;k<in;k++) gw[k]+=g*u[k]; }
            inb++;
            if(inb==bs || i==used-1){ p->t++; float lt=lr*sqrtf(1-powf(.999f,p->t))/(1-powf(.9f,p->t));
                ADAMP(p->W1,gW1,p->mW1,p->vW1,s1); ADAMP(p->b1,gb1,p->mb1,p->vb1,H);
                ADAMP(p->WA,gWA,p->mWA,p->vWA,sA); ADAMP(p->bA,gbA,p->mbA,p->vbA,KCONT);
                ADAMP(p->Wb,gWb,p->mWb,p->vWb,H);
                p->mbb=.9f*p->mbb+.1f*gbb; p->vbb=.999f*p->vbb+.001f*gbb*gbb; p->bb-=lt*(p->mbb/(sqrtf(p->vbb)+1e-8f)+1e-5f*p->bb);
                memset(gW1,0,s1*4);memset(gb1,0,H*4);memset(gWA,0,sA*4);memset(gbA,0,KCONT*4);memset(gWb,0,H*4);gbb=0; inb=0; }
        }
    }
    free(gW1);free(gb1);free(gWA);free(gbA);free(gWb);free(hid);free(dh);free(la);
}
// info (bits/label) = baseline_BCE(predict pA[k]) - model_BCE, averaged over labels & rows. Also HEAD-B.
static void probe_eval(const Probe* p,const float* U,const uint8_t* yA,const uint8_t* yB,const uint8_t* mB,
                       long used,const double* pA,double pB,
                       double* o_baseA,double* o_modA,double* o_baseB,double* o_modB,long* o_nB){
    int H=p->H,in=p->in; float *hid=malloc(H*4),*la=malloc(KCONT*4); float lb;
    double baseA=0,modA=0,baseB=0,modB=0; long nB=0;
    double lpA[KCONT],l1pA[KCONT]; for(int k=0;k<KCONT;k++){ double q=pA[k]; if(q<1e-6)q=1e-6; if(q>1-1e-6)q=1-1e-6; lpA[k]=-log2(q); l1pA[k]=-log2(1-q); }
    double lpB=-log2((pB<1e-6?1e-6:(pB>1-1e-6?1-1e-6:pB))), l1pB=-log2((1-pB<1e-6?1e-6:(1-pB>1-1e-6?1-1e-6:1-pB)));
    for(long i=0;i<used;i++){
        const float* u=&U[(size_t)i*in]; probe_fwd(p,u,hid,la,&lb);
        const uint8_t* ya=&yA[(size_t)i*KCONT];
        for(int k=0;k<KCONT;k++){ double pr=sigf(la[k]); if(pr<1e-6)pr=1e-6; if(pr>1-1e-6)pr=1-1e-6;
            if(ya[k]){ modA+=-log2(pr); baseA+=lpA[k]; } else { modA+=-log2(1-pr); baseA+=l1pA[k]; } }
        if(mB[i]){ double pr=sigf(lb); if(pr<1e-6)pr=1e-6; if(pr>1-1e-6)pr=1-1e-6;
            if(yB[i]){ modB+=-log2(pr); baseB+=lpB; } else { modB+=-log2(1-pr); baseB+=l1pB; } nB++; }
    }
    free(hid);free(la);
    *o_baseA+=baseA; *o_modA+=modA; *o_baseB+=baseB; *o_modB+=modB; *o_nB+=nB;
}

// ============ BLIND: per-label recurrence logistic on [recent_seen_k, log1p(recent_count_k)] ============
// p = sigmoid(a_k + b_k*seen + c_k*lcount). The n-gram-able floor: "recently seen content token -> reappears".
static void blind_features(const uint32_t* prev, long used, float* seen, float* lcnt){
    for(long r=0;r<used;r++){
        int cnt[KCONT]; for(int k=0;k<KCONT;k++) cnt[k]=0;
        for(long j=(r>=BLIND_REC?r-BLIND_REC:0); j<=r; j++){ int lab=g_tok2lab[prev[j]]; if(lab>=0) cnt[lab]++; }
        float* se=&seen[(size_t)r*KCONT]; float* lc=&lcnt[(size_t)r*KCONT];
        for(int k=0;k<KCONT;k++){ se[k]=cnt[k]>0?1.f:0.f; lc[k]=logf(1.0f+(float)cnt[k]); }
    }
}
static void blind_run(const float* seenTr,const float* lcntTr,const uint8_t* yATr,long usedTr,
                      const double* pA, float* a,float* b,float* c, int epochs,float lr){
    for(int k=0;k<KCONT;k++){ double q=pA[k]; if(q<1e-6)q=1e-6; if(q>1-1e-6)q=1-1e-6; a[k]=(float)log(q/(1-q)); b[k]=0;c[k]=0; }
    float* ma=calloc(KCONT,4);float* mb=calloc(KCONT,4);float* mc=calloc(KCONT,4);
    float* va=calloc(KCONT,4);float* vb=calloc(KCONT,4);float* vc=calloc(KCONT,4);
    int t=0; int bs=256; float invn=1.0f/bs;
    for(int ep=0;ep<epochs;ep++){
        float* ga=calloc(KCONT,4);float* gb=calloc(KCONT,4);float* gc=calloc(KCONT,4); long inb=0;
        for(long r=0;r<usedTr;r++){ const float* se=&seenTr[(size_t)r*KCONT]; const float* lc=&lcntTr[(size_t)r*KCONT];
            const uint8_t* ya=&yATr[(size_t)r*KCONT];
            for(int k=0;k<KCONT;k++){ float z=a[k]+b[k]*se[k]+c[k]*lc[k]; float pr=sigf(z); float e=(pr-ya[k])*invn;
                ga[k]+=e; gb[k]+=e*se[k]; gc[k]+=e*lc[k]; }
            inb++;
            if(inb==bs||r==usedTr-1){ t++; float lt=lr*sqrtf(1-powf(.999f,t))/(1-powf(.9f,t));
                for(int k=0;k<KCONT;k++){ ma[k]=.9f*ma[k]+.1f*ga[k];va[k]=.999f*va[k]+.001f*ga[k]*ga[k];a[k]-=lt*ma[k]/(sqrtf(va[k])+1e-8f);
                    mb[k]=.9f*mb[k]+.1f*gb[k];vb[k]=.999f*vb[k]+.001f*gb[k]*gb[k];b[k]-=lt*mb[k]/(sqrtf(vb[k])+1e-8f);
                    mc[k]=.9f*mc[k]+.1f*gc[k];vc[k]=.999f*vc[k]+.001f*gc[k]*gc[k];c[k]-=lt*mc[k]/(sqrtf(vc[k])+1e-8f); }
                memset(ga,0,KCONT*4);memset(gb,0,KCONT*4);memset(gc,0,KCONT*4); inb=0; }
        }
        free(ga);free(gb);free(gc);
    }
    free(ma);free(mb);free(mc);free(va);free(vb);free(vc);
}
static void blind_eval(const float* seen,const float* lcnt,const uint8_t* yA,long used,
                       const double* pA,const float* a,const float* b,const float* c,
                       double* o_base,double* o_mod){
    double base=0,mod=0; double lpA[KCONT],l1pA[KCONT];
    for(int k=0;k<KCONT;k++){ double q=pA[k]; if(q<1e-6)q=1e-6; if(q>1-1e-6)q=1-1e-6; lpA[k]=-log2(q); l1pA[k]=-log2(1-q); }
    for(long r=0;r<used;r++){ const float* se=&seen[(size_t)r*KCONT]; const float* lc=&lcnt[(size_t)r*KCONT];
        const uint8_t* ya=&yA[(size_t)r*KCONT];
        for(int k=0;k<KCONT;k++){ double z=a[k]+b[k]*se[k]+c[k]*lc[k]; double pr=sigf((float)z); if(pr<1e-6)pr=1e-6;if(pr>1-1e-6)pr=1-1e-6;
            if(ya[k]){ mod+=-log2(pr); base+=lpA[k]; } else { mod+=-log2(1-pr); base+=l1pA[k]; } } }
    *o_base+=base; *o_mod+=mod;
}

// ============ data windows ============
static float *g_Xtr,*g_Xv[N_VAL];
static uint32_t *g_ttr,*g_ptr,*g_tv[N_VAL],*g_pv[N_VAL];
static long g_trrows,g_vrows[N_VAL], g_maxrows;
static uint8_t *g_yAtr,*g_yBtr,*g_mBtr, *g_yAv[N_VAL],*g_yBv[N_VAL],*g_mBv[N_VAL];
static long g_usedTr, g_usedV[N_VAL];
static double g_pA[KCONT], g_pB;
static int g_epochs;

static void rand_perm(long* perm, long n, uint64_t seed){ for(long i=0;i<n;i++) perm[i]=i;
    uint64_t s=seed?seed:0xBADC0DEULL; for(long i=n-1;i>0;i--){ s^=s<<13;s^=s>>7;s^=s<<17; long j=(long)((s>>11)%(uint64_t)(i+1)); long t=perm[i];perm[i]=perm[j];perm[j]=t; } }

enum { V_FULL=0, V_POINT, V_TOKONLY, V_SCRAMBLE, NVARMLP };
static const char* VARN[NVARMLP]={"FULL(traj)","POINT(state)","TOKONLY(ablate)","SCRAMBLE(perm-lbl)"};
static double g_infoA[NVARMLP+1], g_infoB[NVARMLP]; // +1 slot for BLIND infoA

// SCRAMBLE = causal FULL trajectory but the trajectory<->label correspondence is broken by permuting the
// labels across rows (leak-free: no future state ever enters a row's pooled memory). Must collapse to ~0.
static void perm_labels(const uint8_t* yA,const uint8_t* yB,const uint8_t* mB,long used,uint64_t seed,
                        uint8_t* yA2,uint8_t* yB2,uint8_t* mB2){
    long* pm=malloc((size_t)used*sizeof(long)); rand_perm(pm,used,seed);
    for(long r=0;r<used;r++){ long s=pm[r]; memcpy(&yA2[(size_t)r*KCONT],&yA[(size_t)s*KCONT],KCONT);
        yB2[r]=yB[s]; mB2[r]=mB[s]; }
    free(pm);
}
static void build_rep(int vtype,const float* X,const uint32_t* prev,long nrow,long used,int in,float* U){
    if(vtype==V_FULL||vtype==V_SCRAMBLE){ float* T=malloc((size_t)nrow*TRAJ_DIM*4); build_traj(X,nrow,NULL,T);
        for(long r=0;r<used;r++) memcpy(&U[(size_t)r*in],&T[(size_t)r*TRAJ_DIM],in*4); free(T); }
    else if(vtype==V_POINT){ for(long r=0;r<used;r++) memcpy(&U[(size_t)r*in],&X[(size_t)r*S_DIM],in*4); }
    else { float* Tk=malloc((size_t)nrow*DE_TOK*4); build_tokonly(prev,nrow,Tk);
        for(long r=0;r<used;r++) memcpy(&U[(size_t)r*in],&Tk[(size_t)r*DE_TOK],in*4); free(Tk); }
}
static void run_mlp_variant(int vtype){
    int in = (vtype==V_POINT)?S_DIM : (vtype==V_TOKONLY)?DE_TOK : TRAJ_DIM;
    float* Utr=malloc((size_t)g_usedTr*in*4);
    build_rep(vtype,g_Xtr,g_ptr,g_trrows,g_usedTr,in,Utr);

    // for SCRAMBLE, break the input<->label correspondence (must collapse)
    uint8_t *yAt=g_yAtr,*yBt=g_yBtr,*mBt=g_mBtr;
    if(vtype==V_SCRAMBLE){ yAt=malloc((size_t)g_usedTr*KCONT); yBt=malloc(g_usedTr); mBt=malloc(g_usedTr);
        perm_labels(g_yAtr,g_yBtr,g_mBtr,g_usedTr,0x5C8A11ULL,yAt,yBt,mBt); }

    Probe p; probe_init(&p,in,0x53ULL^((uint64_t)vtype*0x9E37ULL));
    probe_train(&p,Utr,yAt,yBt,mBt,g_usedTr,g_epochs,1e-3f);
    free(Utr); if(vtype==V_SCRAMBLE){ free(yAt);free(yBt);free(mBt); }

    double baseA=0,modA=0,baseB=0,modB=0; long nB=0;
    for(int w=0;w<N_VAL;w++){ int inv=in; float* Uv=malloc((size_t)g_usedV[w]*inv*4);
        build_rep(vtype,g_Xv[w],g_pv[w],g_vrows[w],g_usedV[w],inv,Uv);
        uint8_t *yAe=g_yAv[w],*yBe=g_yBv[w],*mBe=g_mBv[w];
        if(vtype==V_SCRAMBLE){ yAe=malloc((size_t)g_usedV[w]*KCONT); yBe=malloc(g_usedV[w]); mBe=malloc(g_usedV[w]);
            perm_labels(g_yAv[w],g_yBv[w],g_mBv[w],g_usedV[w],0x5C8A11ULL^(uint64_t)(w+1),yAe,yBe,mBe); }
        probe_eval(&p,Uv,yAe,yBe,mBe,g_usedV[w],g_pA,g_pB,&baseA,&modA,&baseB,&modB,&nB);
        free(Uv); if(vtype==V_SCRAMBLE){ free(yAe);free(yBe);free(mBe); }
    }
    probe_free(&p);
    long totL=0; for(int w=0;w<N_VAL;w++) totL+=g_usedV[w]; totL*=KCONT;
    double infoA=(baseA-modA)/(double)totL;
    double infoB=(nB>0)?(baseB-modB)/(double)nB:0;
    g_infoA[vtype]=infoA; g_infoB[vtype]=infoB;
    printf("  %-16s | HEAD-A info %+8.5f bits/lbl | HEAD-B info %+8.5f bits  (baseA=%.4f modA=%.4f)\n",
           VARN[vtype],infoA,infoB, baseA/totL, modA/totL);
}

int main(int argc,char** argv){
    if(argc<5){ fprintf(stderr,"Usage: %s <data> <D1_w> <bpe_merges> <outprefix> [--len N] [--epochs E] [--smoke]\n",argv[0]); return 1; }
    setvbuf(stderr,NULL,_IONBF,0); setvbuf(stdout,NULL,_IONBF,0);
    long N=150000, maxb=0; g_epochs=6;
    for(int i=5;i<argc;i++){
        if(!strcmp(argv[i],"--len")&&i+1<argc) N=atol(argv[++i]);
        else if(!strcmp(argv[i],"--epochs")&&i+1<argc) g_epochs=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--max-bytes")&&i+1<argc) maxb=atol(argv[++i]);
        else if(!strcmp(argv[i],"--smoke")){ N=24000; g_epochs=3; }
    }
    if(S_DIM!=512){ fprintf(stderr,"S_DIM=%d != 512\n",S_DIM); return 1; }
    FILE* fd=fopen(argv[1],"rb"); if(!fd){fprintf(stderr,"data\n");return 1;} fseek(fd,0,SEEK_END); g_fsz=ftell(fd); fseek(fd,0,SEEK_SET);
    if(maxb>0 && maxb<g_fsz) g_fsz=maxb;
    g_data=malloc(g_fsz); fread(g_data,1,g_fsz,fd); fclose(fd);

    SiliconEntropyState see;
    if(!load_d1(argv[2],&see)) return 1;
    if(!bpe_load_file(&g_bpe,argv[3])) return 1; VTOK=g_bpe.vocab;
    if(VTOK> (1<<20)){ fprintf(stderr,"VTOK too big\n"); return 1; }
    gen_lift(PROJ_SEED);
    ent_table=malloc(CLASSES*CLASSES*4);
    for(int i=0;i<CLASSES;i++) for(int j=0;j<CLASSES;j++){ float m=-1e9f; for(int k=0;k<CLASSES;k++) if(trigram[i][j][k]>m)m=trigram[i][j][k];
        double se=0; for(int k=0;k<CLASSES;k++) se+=exp((double)(trigram[i][j][k]-m));
        double Hh=0; for(int k=0;k<CLASSES;k++){ double pp=exp((double)(trigram[i][j][k]-m))/se; if(pp>1e-12)Hh-=pp*log(pp); }
        ent_table[i][j]=(float)Hh; }

    // substrate read-only checksum (no grad ever flows into it)
    double sck=0; for(int i=0;i<D1_TOT;i++) sck+=md1[i]+sd1[i];
    fprintf(stderr,"substrate checksum (md1+sd1 sum) = %.6f [READ-ONLY]\n",sck);

    tokenize_corpus();
    long tok_train_end=(long)(g_ntok*0.90);
    build_unigram(tok_train_end);
    build_content_labels();
    hippo_init(0x4190C0DEULL);
    tokemb_init(0xE2C0DEULL);

    long tr_start=g_fsz/5;
    long va_start[N_VAL]={ g_fsz/2, (long)(0.78*g_fsz) };
    for(int w=0;w<N_VAL;w++) if(va_start[w]+N+3>g_fsz){ fprintf(stderr,"val window %d out of file\n",w+1); return 1; }

    g_maxrows=N+16;
    g_Xtr=malloc((size_t)g_maxrows*S_DIM*4); g_ttr=malloc((size_t)g_maxrows*4); g_ptr=malloc((size_t)g_maxrows*4);
    for(int w=0;w<N_VAL;w++){ g_Xv[w]=malloc((size_t)g_maxrows*S_DIM*4); g_tv[w]=malloc((size_t)g_maxrows*4); g_pv[w]=malloc((size_t)g_maxrows*4); }
    fprintf(stderr,"extract train...\n"); g_trrows=extract(&see,tr_start,N,g_Xtr,g_ttr,g_ptr);
    for(int w=0;w<N_VAL;w++){ fprintf(stderr,"extract val%d...\n",w+1); g_vrows[w]=extract(&see,va_start[w],N,g_Xv[w],g_tv[w],g_pv[w]); }

    // labels
    g_yAtr=malloc((size_t)g_maxrows*KCONT); g_yBtr=malloc(g_maxrows); g_mBtr=malloc(g_maxrows);
    g_usedTr=build_labels(g_ptr,g_trrows,g_yAtr,g_yBtr,g_mBtr);
    for(int w=0;w<N_VAL;w++){ g_yAv[w]=malloc((size_t)g_maxrows*KCONT); g_yBv[w]=malloc(g_maxrows); g_mBv[w]=malloc(g_maxrows);
        g_usedV[w]=build_labels(g_pv[w],g_vrows[w],g_yAv[w],g_yBv[w],g_mBv[w]); }

    // base rates from TRAIN
    for(int k=0;k<KCONT;k++) g_pA[k]=0; long pBn=0; double pBs=0;
    for(long r=0;r<g_usedTr;r++){ const uint8_t* ya=&g_yAtr[(size_t)r*KCONT]; for(int k=0;k<KCONT;k++) g_pA[k]+=ya[k];
        if(g_mBtr[r]){ pBs+=g_yBtr[r]; pBn++; } }
    for(int k=0;k<KCONT;k++) g_pA[k]/=(double)g_usedTr; g_pB=(pBn>0)?pBs/pBn:0.0;
    double pAm=0; for(int k=0;k<KCONT;k++) pAm+=g_pA[k]; pAm/=KCONT;
    fprintf(stderr,"usedTr=%ld usedV=%ld/%ld  base-rate: HEAD-A mean p=%.4f  HEAD-B p=%.4f (n=%ld)\n",
            g_usedTr,g_usedV[0],g_usedV[1],pAm,g_pB,pBn);

    printf("\n==== Phase 53.0 PROBE: is long-range semantics IN THE TRAJECTORY of the frozen substrate? ====\n");
    printf("N=%ld epochs=%d  far=[t+%d,t+%d]  K=%d content labels  HID=%d  TRAJ_DIM=%d\n",
           N,g_epochs,FAR_LO,FAR_HI,KCONT,HID,TRAJ_DIM);
    printf("info = baseline_BCE(predict marginal) - model_BCE, bits/label on held-out windows. higher=more decodable.\n\n");

    run_mlp_variant(V_FULL);
    run_mlp_variant(V_POINT);
    run_mlp_variant(V_TOKONLY);
    run_mlp_variant(V_SCRAMBLE);

    // BLIND (HEAD-A only): per-label recurrence logistic
    { float* seenTr=malloc((size_t)g_usedTr*KCONT*4); float* lcntTr=malloc((size_t)g_usedTr*KCONT*4);
      blind_features(g_ptr,g_usedTr,seenTr,lcntTr);
      float* a=malloc(KCONT*4),*b=malloc(KCONT*4),*c=malloc(KCONT*4);
      blind_run(seenTr,lcntTr,g_yAtr,g_usedTr,g_pA,a,b,c,5,1.5e-2f);
      free(seenTr);free(lcntTr);
      double base=0,mod=0; long totL=0;
      for(int w=0;w<N_VAL;w++){ float* se=malloc((size_t)g_usedV[w]*KCONT*4); float* lc=malloc((size_t)g_usedV[w]*KCONT*4);
          blind_features(g_pv[w],g_usedV[w],se,lc); blind_eval(se,lc,g_yAv[w],g_usedV[w],g_pA,a,b,c,&base,&mod);
          totL+=g_usedV[w]; free(se);free(lc); }
      totL*=KCONT; g_infoA[NVARMLP]=(base-mod)/(double)totL;
      free(a);free(b);free(c);
      printf("  %-16s | HEAD-A info %+8.5f bits/lbl | (n-gram-able recurrence floor)\n","BLIND(recur)",g_infoA[NVARMLP]);
    }

    // ---- verdict ----
    double full=g_infoA[V_FULL], point=g_infoA[V_POINT], toko=g_infoA[V_TOKONLY], scr=g_infoA[V_SCRAMBLE], blind=g_infoA[NVARMLP];
    printf("\n==== 53.0 VERDICT (HEAD-A topic/content presence, info bits/label) ====\n");
    printf("  FULL=%+.5f  POINT=%+.5f  TOKONLY=%+.5f  SCRAMBLE=%+.5f  BLIND=%+.5f\n",full,point,toko,scr,blind);
    printf("  FULL>>TOKONLY: %+.5f | FULL>>SCRAMBLE: %+.5f | FULL>POINT: %+.5f | FULL>BLIND: %+.5f\n",
           full-toko, full-scr, full-point, full-blind);
    int c1=(full-toko)>=0.0030, c2=(full-scr)>=0.0030, c3=(full-point)>0.0, c4=(full-blind)>0.0;
    printf("  HEAD-B info: FULL=%+.5f POINT=%+.5f TOKONLY=%+.5f SCRAMBLE=%+.5f\n",
           g_infoB[V_FULL],g_infoB[V_POINT],g_infoB[V_TOKONLY],g_infoB[V_SCRAMBLE]);
    printf("  -> [FULL>>TOK:%d][FULL>>SCR:%d][FULL>POINT:%d][FULL>BLIND:%d]  =>  %s\n",
           c1,c2,c3,c4,(c1&&c2&&c3&&c4)?"SIGNAL PRESENT -> 53.A predictive bottleneck":"SIGNAL ABSENT/WEAK -> STOP (substrate doesn't carry it)");
    printf("\nReading: if FULL collapses to TOKONLY/SCRAMBLE -> the frozen substrate's trajectory adds no long-range\n");
    printf("semantics beyond the tokens themselves -> read-side is exhausted, the bottleneck is the substrate (forbidden zone).\n");

    if(argv[4]){ char sp[512]; snprintf(sp,sizeof sp,"%s_probe.bin",argv[4]);
        FILE* f=fopen(sp,"wb"); if(f){ uint32_t mg=0x5345455C; fwrite(&mg,4,1,f);
            double v[5]={full,point,toko,scr,blind}; fwrite(v,sizeof(double),5,f); fclose(f); } }
    return 0;
}
