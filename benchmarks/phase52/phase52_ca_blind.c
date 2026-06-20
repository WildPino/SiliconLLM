// Phase 52.C - induction head / kNN-LM FAITHFUL: key on the TOKEN-CONTEXT, not on W*state.
//
// 52.B sealed: the reservoir STATE does not encode long-range identity in a linearly-addressable form
// (it entangles token identity with context). All of 51-52 keyed on W*state. 52.C changes what the key
// READS: the clean token-context, not the murky state.
//
// HEAD (everything fixed except the readout, V1):
//   E[tok]   : fixed random bipolar embedding per token-id (Dk)
//   addr_i   = normalize( sum_{j=0..k-1} cyclic_shift_j( E[ prev_{i-j} ] ) )   VSA-bind of last k token-ids
//              (ties to 51.0: shift = binding-by-position; equal k-grams -> ~identical addr, partial overlap
//               -> graded soft similarity). Key AND query use the SAME per-row encoding (context ending at i).
//   value_i  = atom(tgt_i) = atom(token following the context ending at i)   FIXED (oracle clears value)
//   a_{p,tau}= softmax_{tau in [p-cap,p-2]}( addr_p . addr_tau / temp ) ; r_p = sum a*value
//   logits   = readout([ state_p | r_p ])      ONLY readout trained (V1).
// Charter-pure: token-ids are CLEAN INPUT, substrate frozen, no value learned, no W on state, no BPTT.
// Because key/query/value are all fixed, r_p is PRECOMPUTED -> each arm = train_clean on [state|r] (same
// regime as NO-MEM/ORACLE); FIXED-STATE-KEY vs TOKKEY differ ONLY in what the addr encodes.
//
// Arms: NO-MEM / ORACLE / FIXED-STATE-KEY(addr=normalize(W_k*state), =52.0/B ~3% ref) /
//       TOKKEY-k{2,3,4} / TOKKEY-SHUF(value permuted, must collapse) / [GLOBAL-NGRAM if --global].
// PRIMARY CHECK: does TOKKEY beat FIXED-STATE-KEY and push frac_oracle well past 52.B's ~2%? If yes ->
// the murky state was the problem, token-keying bypasses it. Gate stratified (>33,>200) bar 0.05.
// SECONDARY (only if TOKKEY wins): GLOBAL-NGRAM (same k-gram->next stats pooled globally over train) vs
// TOKKEY-intra-window -> TOKKEY>GLOBAL = intra-document memory, not just a richer n-gram model.
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase52/phase52_c_attn.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase52_c_attn.exe -lm -I .
// Run:
//   bin/phase52_c_attn.exe <data> <D1_w> <bpe_merges> <outprefix> [--len N] [--dk 1024] [--dv 256]
//        [--epochs 6] [--cap 1024] [--temp 0.1] [--global] [--smoke]
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
#define S_DIM     (D1_TOT + EXP_BANDS)
#define PROJ_SEED 0x48B2EC0DEULL
#define N_VAL     3
#define HID       32

#define DWIN_S   0.05
#define DSHUF_S  0.03
#define RARE_P   1e-4

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
static float* g_Wk;    // [Dk*512] fixed state-key projection (FIXED-STATE-KEY reference)
static float* g_E;     // [VTOK*Dk] fixed random bipolar token embeddings (TOKKEY)
static void store_assets_init(uint64_t seed){
    g_cb=(float*)realloc(g_cb,(size_t)VTOK*g_Dv*4); uint64_t s=seed?seed:0x520C0DEULL;
    float invv=1.0f/sqrtf((float)g_Dv);
    for(size_t i=0;i<(size_t)VTOK*g_Dv;i++){ s^=s<<13;s^=s>>7;s^=s<<17; g_cb[i]=(s&1ULL)?invv:-invv; }
    g_Wk=(float*)realloc(g_Wk,(size_t)g_Dk*512*4);
    for(size_t i=0;i<(size_t)g_Dk*512;i++){ s^=s<<13;s^=s>>7;s^=s<<17; g_Wk[i]=(s&1ULL)?1.f:-1.f; }
    g_E=(float*)realloc(g_E,(size_t)VTOK*g_Dk*4);
    for(size_t i=0;i<(size_t)VTOK*g_Dk;i++){ s^=s<<13;s^=s>>7;s^=s<<17; g_E[i]=(s&1ULL)?1.f:-1.f; }
}
// addr_i = normalize(W_k*state_i)  (FIXED-STATE-KEY: the 52.0/B reference, only readout trained)
static void enc_state(const float* X, long nrow, float* addr){
    for(long i=0;i<nrow;i++){ const float* st=&X[(size_t)i*S_DIM]; float* k=&addr[(size_t)i*g_Dk];
        for(int a=0;a<g_Dk;a++) k[a]=dot_avx(&g_Wk[(size_t)a*512],st,512);
        float nn=0; for(int a=0;a<g_Dk;a++) nn+=k[a]*k[a]; float inv=1.0f/(sqrtf(nn)+1e-12f);
        for(int a=0;a<g_Dk;a++) k[a]*=inv; }
}
// addr_i = normalize( sum_{j=0..k-1} cyclic_shift_j( E[prev_{i-j}] ) )   VSA-bind of last k token-ids
static void enc_tokkey(const uint32_t* prev, long nrow, int k, float* addr){
    int Dk=g_Dk;
    for(long i=0;i<nrow;i++){ float* a=&addr[(size_t)i*Dk]; for(int d=0;d<Dk;d++) a[d]=0.f;
        for(int j=0;j<k && i-j>=0; j++){ const float* e=&g_E[(size_t)prev[i-j]*Dk];
            for(int d=0;d<Dk;d++){ int dd=d+j; if(dd>=Dk) dd-=Dk; a[dd]+=e[d]; } }   // cyclic shift right by j
        float nn=0; for(int d=0;d<Dk;d++) nn+=a[d]*a[d]; float inv=1.0f/(sqrtf(nn)+1e-12f);
        for(int d=0;d<Dk;d++) a[d]*=inv; }
}
// r_p = sum_{tau in [p-cap, p-2]} softmax(addr_p . addr_tau / temp) atom(tgt[valperm?valperm[tau]:tau])
static long g_capmax;
static void attn_precompute_r(const float* addr, const uint32_t* tgt, const long* valperm,
                              long nrow, int cap, float temp, float* R){
    int Dk=g_Dk,Dv=g_Dv; float* aw=malloc((size_t)g_capmax*4); float invt=1.0f/temp;
    for(long p=0;p<nrow;p++){ float* r=&R[(size_t)p*Dv]; for(int b=0;b<Dv;b++) r[b]=0.f;
        long hi=p-2, lo=hi-(cap-1); if(lo<0) lo=0; int m=(hi>=lo)?(int)(hi-lo+1):0;
        if(m<=0) continue;
        const float* qp=&addr[(size_t)p*Dk]; float mx=-1e30f;
        for(int j=0;j<m;j++){ long tau=lo+j; float s=dot_avx(qp,&addr[(size_t)tau*Dk],Dk)*invt; aw[j]=s; if(s>mx)mx=s; }
        float Z=0; for(int j=0;j<m;j++){ aw[j]=expf(aw[j]-mx); Z+=aw[j]; }
        float zinv=1.0f/Z; for(int j=0;j<m;j++) aw[j]*=zinv;
        for(int j=0;j<m;j++){ long tau=lo+j; const float* v=&g_cb[(size_t)tgt[valperm?valperm[tau]:tau]*Dv]; axpy_avx(r,v,aw[j],Dv); }
    }
    free(aw);
}

// ============ readout MLP ([state | r]) ============
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
    memcpy(u,state,512*4); if(mem) memcpy(u+512,Rrow,g_Dv*4); }
static void net_fwd(const Net* n,const float* u,float* hid,float* lg){
    for(int j=0;j<n->H;j++){ float a=n->b1[j]+dot_avx(&n->W1[(size_t)j*n->in],u,n->in); hid[j]=a>0?a:0; }
    for(int c=0;c<VTOK;c++) lg[c]=n->b2[c]+dot_avx(&n->W2[(size_t)c*n->H],hid,n->H);
}
#define ADAM(P,GG,MM,VV,NN) for(size_t z=0;z<(size_t)(NN);z++){ MM[z]=.9f*MM[z]+.1f*GG[z]; VV[z]=.999f*VV[z]+.001f*GG[z]*GG[z]; P[z]-=lt*(MM[z]/(sqrtf(VV[z])+1e-8f)+1e-5f*P[z]); }
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
static double bpt(double bits,long cnt){ return cnt? bits/cnt : 0.0; }


// ============ GLOBAL-NGRAM (Step-0 honesty control): k-gram->next expected value-atom, pooled GLOBALLY ====
// TOKKEY = intra-window retrieval; this = SAME k-gram->next stats but over the WHOLE train stream.
// r_global(ctx) = sum_next Pglobal(next|ctx) atom(next) = (1/cnt) sum_occurrences atom(next).
typedef struct { uint64_t* slot; uint32_t* idx; size_t mask; uint32_t ndist; uint32_t* cnt; float* sum; } GMap;
static inline uint64_t packk(const uint32_t* p, long i, int k){
    if(i-k+1<0) return ~0ULL; uint64_t key=0; for(int j=0;j<k;j++) key|=((uint64_t)p[i-j])<<(11*j); return key; }
static void gmap_build(GMap* g, int k, long tend){
    size_t ns=1; while(ns < (size_t)tend*2) ns<<=1; g->mask=ns-1;
    g->slot=calloc(ns,8); g->idx=malloc(ns*4); g->ndist=0;
    const uint32_t* T=g_tok;
    for(long t=k-1;t+1<tend;t++){ uint64_t kk=packk(T,t,k); if(kk==~0ULL) continue; uint64_t s=(kk<<1)|1;
        size_t h=(size_t)((kk*0x9E3779B97F4A7C15ULL)&g->mask);
        while(g->slot[h]&&g->slot[h]!=s) h=(h+1)&g->mask;
        if(!g->slot[h]){ g->slot[h]=s; g->idx[h]=g->ndist++; } }
    g->cnt=calloc(g->ndist,4); g->sum=calloc((size_t)g->ndist*g_Dv,4);
    for(long t=k-1;t+1<tend;t++){ uint64_t kk=packk(T,t,k); if(kk==~0ULL) continue; uint64_t s=(kk<<1)|1;
        size_t h=(size_t)((kk*0x9E3779B97F4A7C15ULL)&g->mask);
        while(g->slot[h]!=s) h=(h+1)&g->mask;
        uint32_t id=g->idx[h]; g->cnt[id]++; axpy_avx(&g->sum[(size_t)id*g_Dv], &g_cb[(size_t)T[t+1]*g_Dv], 1.0f, g_Dv); }
    fprintf(stderr,"  gmap k=%d: %u distinct contexts over %ld train tokens\n",k,g->ndist,tend);
}
static void gmap_free(GMap* g){ free(g->slot);free(g->idx);free(g->cnt);free(g->sum); }
static void gmap_R(const GMap* g, const uint32_t* prev, long nrow, int k, float* R){
    int Dv=g_Dv;
    for(long i=0;i<nrow;i++){ float* r=&R[(size_t)i*Dv]; for(int b=0;b<Dv;b++) r[b]=0.f;
        uint64_t kk=packk(prev,i,k); if(kk==~0ULL) continue; uint64_t s=(kk<<1)|1;
        size_t h=(size_t)((kk*0x9E3779B97F4A7C15ULL)&g->mask);
        while(g->slot[h]&&g->slot[h]!=s) h=(h+1)&g->mask;
        if(g->slot[h]==s){ uint32_t id=g->idx[h]; float sc=1.0f/(float)g->cnt[id];
            const float* sv=&g->sum[(size_t)id*Dv]; for(int b=0;b<Dv;b++) r[b]=sv[b]*sc; } }
}

// windows + scratch
static float *g_Xtr,*g_Xv[N_VAL]; static uint32_t *g_ttr,*g_ptr,*g_tv[N_VAL],*g_pv[N_VAL];
static long g_trrows,g_vrows[N_VAL]; static uint8_t *g_vrec[N_VAL],*g_vrare[N_VAL]; static int *g_vdist[N_VAL];
static long g_maxrows; static int g_epochs; static double g_nm[NSTRAT];
static float *g_addr,*g_Rtr,*g_Rv[N_VAL];

// train readout on [state|Rtr], eval on the 3 windows with their Rv[w] -> aggregated strata in out[].
static void run_R(const char* label, double* out){
    Net net; net_init(&net,512+g_Dv,HID,0xC1ULL^(uint64_t)label[0]^((uint64_t)label[1]<<8));
    train_clean(&net,g_Xtr,g_Rtr,g_ttr,g_ptr,g_trrows,1,g_epochs,5e-4f);
    double b[NSTRAT]={0}; long c[NSTRAT]={0};
    for(int w=0;w<N_VAL;w++){ double sb[NSTRAT]; long sc[NSTRAT];
        eval_clean(&net,g_Xv[w],g_Rv[w],g_tv[w],g_pv[w],g_vrows[w],1,g_vrec[w],g_vdist[w],g_vrare[w],sb,sc);
        for(int st=0;st<NSTRAT;st++){ b[st]+=sb[st]; c[st]+=sc[st]; } }
    net_free(&net);
    for(int st=0;st<NSTRAT;st++) out[st]=bpt(b[st],c[st]);
    printf("  %-16s | >200 %+.4f | 33-200 %+.4f | 5-32 %+.4f | RECUR %+.4f\n",
        label, g_nm[ST_D4]-out[ST_D4], g_nm[ST_D3]-out[ST_D3], g_nm[ST_D2]-out[ST_D2], g_nm[ST_RECUR]-out[ST_RECUR]);
}
static void arm_tokkey(const char* label,int k,float temp,double* out){
    enc_tokkey(g_ptr,g_trrows,k,g_addr); attn_precompute_r(g_addr,g_ttr,NULL,g_trrows,g_capmax,temp,g_Rtr);
    for(int w=0;w<N_VAL;w++){ enc_tokkey(g_pv[w],g_vrows[w],k,g_addr); attn_precompute_r(g_addr,g_tv[w],NULL,g_vrows[w],g_capmax,temp,g_Rv[w]); }
    run_R(label,out);
}
static void arm_state(const char* label,double* out){
    enc_state(g_Xtr,g_trrows,g_addr); attn_precompute_r(g_addr,g_ttr,NULL,g_trrows,g_capmax,0.10f,g_Rtr);
    for(int w=0;w<N_VAL;w++){ enc_state(g_Xv[w],g_vrows[w],g_addr); attn_precompute_r(g_addr,g_tv[w],NULL,g_vrows[w],g_capmax,0.10f,g_Rv[w]); }
    run_R(label,out);
}
static void arm_global(const char* label,int k,long tend,double* out){
    GMap g; gmap_build(&g,k,tend);
    gmap_R(&g,g_ptr,g_trrows,k,g_Rtr);
    for(int w=0;w<N_VAL;w++) gmap_R(&g,g_pv[w],g_vrows[w],k,g_Rv[w]);
    gmap_free(&g);
    run_R(label,out);
}

int main(int argc,char** argv){
    if(argc<5){ fprintf(stderr,"Usage: %s <data> <D1_w> <bpe_merges> <outprefix> [--len N] [--dk D] [--dv D] [--epochs E] [--cap C] [--smoke]\n",argv[0]); return 1; }
    setvbuf(stderr,NULL,_IONBF,0); setvbuf(stdout,NULL,_IONBF,0);
    long N=150000, maxb=0; g_epochs=6; int cap_full=1024;
    for(int i=5;i<argc;i++){
        if(!strcmp(argv[i],"--len")&&i+1<argc) N=atol(argv[++i]);
        else if(!strcmp(argv[i],"--dk")&&i+1<argc) g_Dk=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--dv")&&i+1<argc) g_Dv=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--epochs")&&i+1<argc) g_epochs=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--cap")&&i+1<argc) cap_full=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--max-bytes")&&i+1<argc) maxb=atol(argv[++i]);
        else if(!strcmp(argv[i],"--smoke")){ N=20000; g_Dk=256; g_Dv=64; g_epochs=4; cap_full=256; }
    }
    g_capmax=cap_full;
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
    g_addr=malloc((size_t)g_maxrows*g_Dk*4);
    g_Rtr=malloc((size_t)g_maxrows*g_Dv*4); for(int w=0;w<N_VAL;w++) g_Rv[w]=malloc((size_t)g_maxrows*g_Dv*4);

    printf("\n==== 52.C.A STEP-0 BLIND: TOKKEY (intra-window) vs GLOBAL-NGRAM (pooled) + temp sweep ====\n");
    printf("Dk=%d Dv=%d N=%ld cap=%d epochs=%d. DECISION on dist33-200 & 5-32: TOKKEY >> GLOBAL = intra-doc memory.\n",g_Dk,g_Dv,N,cap_full,g_epochs);
    printf("(numbers = d_NOMEM, higher=better; the global bigram is ALREADY in NO-MEM, so these are NET of it)\n\n");

    { fprintf(stderr,"=== NO-MEM ===\n"); double b[NSTRAT]={0}; long c[NSTRAT]={0};
      Net net; net_init(&net,512,HID,0xA0ULL);
      train_clean(&net,g_Xtr,NULL,g_ttr,g_ptr,g_trrows,0,g_epochs,5e-4f);
      for(int w=0;w<N_VAL;w++){ double sb[NSTRAT]; long sc[NSTRAT];
          eval_clean(&net,g_Xv[w],NULL,g_tv[w],g_pv[w],g_vrows[w],0,g_vrec[w],g_vdist[w],g_vrare[w],sb,sc);
          for(int st=0;st<NSTRAT;st++){ b[st]+=sb[st]; c[st]+=sc[st]; } }
      net_free(&net); for(int st=0;st<NSTRAT;st++) g_nm[st]=bpt(b[st],c[st]); }
    printf("NO-MEM abs: >200=%.4f 33-200=%.4f 5-32=%.4f RECUR=%.4f\n\n",g_nm[ST_D4],g_nm[ST_D3],g_nm[ST_D2],g_nm[ST_RECUR]);

    double orc[NSTRAT];
    { fprintf(stderr,"=== ORACLE ===\n"); double b[NSTRAT]={0}; long c[NSTRAT]={0};
      float* Roa=malloc((size_t)g_maxrows*g_Dv*4);
      Net net; net_init(&net,512+g_Dv,HID,0xDEAD1ULL);
      for(long i=0;i<g_trrows;i++) memcpy(&Roa[(size_t)i*g_Dv],&g_cb[(size_t)g_ttr[i]*g_Dv],g_Dv*4);
      train_clean(&net,g_Xtr,Roa,g_ttr,g_ptr,g_trrows,1,g_epochs,5e-4f);
      for(int w=0;w<N_VAL;w++){ double sb[NSTRAT]; long sc[NSTRAT];
          for(long i=0;i<g_vrows[w];i++) memcpy(&Roa[(size_t)i*g_Dv],&g_cb[(size_t)g_tv[w][i]*g_Dv],g_Dv*4);
          eval_clean(&net,g_Xv[w],Roa,g_tv[w],g_pv[w],g_vrows[w],1,g_vrec[w],g_vdist[w],g_vrare[w],sb,sc);
          for(int st=0;st<NSTRAT;st++){ b[st]+=sb[st]; c[st]+=sc[st]; } }
      net_free(&net); free(Roa); for(int st=0;st<NSTRAT;st++) orc[st]=bpt(b[st],c[st]); }
    printf("ORACLE ceiling: >200 %+.4f | 33-200 %+.4f | 5-32 %+.4f\n\n",g_nm[ST_D4]-orc[ST_D4],g_nm[ST_D3]-orc[ST_D3],g_nm[ST_D2]-orc[ST_D2]);

    double fix[NSTRAT];
    printf("-- reference --\n"); fprintf(stderr,"=== FIXED-STATE ===\n"); arm_state("FIXED-STATE",fix);

    float temps[3]={0.05f,0.10f,0.20f};
    double tk2[3][NSTRAT], tk3[3][NSTRAT];
    printf("-- TOKKEY-k2 (winner) temp sweep --\n");
    for(int ti=0;ti<3;ti++){ char lab[40]; snprintf(lab,sizeof lab,"TOKKEY-k2 t%.2f",temps[ti]);
        fprintf(stderr,"=== %s ===\n",lab); arm_tokkey(lab,2,temps[ti],tk2[ti]); }
    printf("-- TOKKEY-k3 temp sweep --\n");
    for(int ti=0;ti<3;ti++){ char lab[40]; snprintf(lab,sizeof lab,"TOKKEY-k3 t%.2f",temps[ti]);
        fprintf(stderr,"=== %s ===\n",lab); arm_tokkey(lab,3,temps[ti],tk3[ti]); }

    double g2[NSTRAT], g3[NSTRAT];
    printf("-- GLOBAL-NGRAM (pooled k-gram->next stats over train) --\n");
    fprintf(stderr,"=== GLOBAL k2 ===\n"); arm_global("GLOBAL-k2",2,tok_train_end,g2);
    fprintf(stderr,"=== GLOBAL k3 ===\n"); arm_global("GLOBAL-k3",3,tok_train_end,g3);

    int bi2=0,bi3=0; for(int ti=1;ti<3;ti++){ if((g_nm[ST_D3]-tk2[ti][ST_D3])>(g_nm[ST_D3]-tk2[bi2][ST_D3])) bi2=ti;
                                               if((g_nm[ST_D3]-tk3[ti][ST_D3])>(g_nm[ST_D3]-tk3[bi3][ST_D3])) bi3=ti; }
    printf("\n==== STEP-0 BLIND VERDICT (decision strata: 33-200, 5-32) ====\n");
    printf("  best TOKKEY-k2 temp=%.2f | 33-200 %+.4f | 5-32 %+.4f\n",temps[bi2],g_nm[ST_D3]-tk2[bi2][ST_D3],g_nm[ST_D2]-tk2[bi2][ST_D2]);
    printf("  best TOKKEY-k3 temp=%.2f | 33-200 %+.4f | 5-32 %+.4f\n",temps[bi3],g_nm[ST_D3]-tk3[bi3][ST_D3],g_nm[ST_D2]-tk3[bi3][ST_D2]);
    printf("  GLOBAL-k2              | 33-200 %+.4f | 5-32 %+.4f\n",g_nm[ST_D3]-g2[ST_D3],g_nm[ST_D2]-g2[ST_D2]);
    printf("  GLOBAL-k3              | 33-200 %+.4f | 5-32 %+.4f\n",g_nm[ST_D3]-g3[ST_D3],g_nm[ST_D2]-g3[ST_D2]);
    printf("  FIXED-STATE           | 33-200 %+.4f | 5-32 %+.4f\n",g_nm[ST_D3]-fix[ST_D3],g_nm[ST_D2]-fix[ST_D2]);
    double tkbest=g_nm[ST_D3]-tk2[bi2][ST_D3]; double gbest=g_nm[ST_D3]-g2[ST_D3]; if((g_nm[ST_D3]-g3[ST_D3])>gbest) gbest=g_nm[ST_D3]-g3[ST_D3];
    double margin=tkbest-gbest;
    printf("  -> TOKKEY_best(33-200)=%+.4f  GLOBAL_best=%+.4f  MARGIN=%+.4f\n",tkbest,gbest,margin);
    printf("  DECISION: %s\n", margin>=0.02 ? "TOKKEY >> GLOBAL = intra-doc MEMORY -> proceed to Step-1 closed-loop"
                                            : (margin>0 ? "TOKKEY > GLOBAL but THIN -> judgment call (sweep/closed-loop risk)"
                                                        : "GLOBAL eats the gain = just a richer n-gram -> STOP, report"));
    return 0;
}
