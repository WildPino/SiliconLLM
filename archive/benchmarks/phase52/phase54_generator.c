// Phase 54 generator - 50.A token readout + GLOBAL-k3 n-gram PRIOR (medium-range honest cash-in).
//
// Identical to phase50_a_generator EXCEPT a THIRD additive log-prob term in the token logits:
//   mlg[c] = (mlg[c] + tb[c] + lambda_k3 * k3logp[last3][c]) / temp
// where k3logp = log of an interpolated/backoff-gated trigram-context LM P(next | last 3 tokens)
// built at startup from the TRAIN split of the corpus (k3 -> bigram(1-tok) -> unigram, count-gated).
// MEDIUM-range fluency/compression lever (52.C.A GLOBAL-k3 +0.40 BPB teacher-forced); NOT a long-range
// fix. Promoted only if it lowers held-out BPB AND does not regress the read (gate-v2 word + Phase-47
// byte-guards). Substrate + token MLP UNCHANGED. k3 deterministic (counts only).
//
// Two modes:
//   (default) closed-loop generation with the k3 term wired in; prints self_BPB + byte-guards.
//   --tf-eval  teacher-forced held-out BPB sweep over lambda_k3 x count-thr, stratified by distance.
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase52/phase54_generator.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase54_generator.exe -lm -I .
// Run (gen): bin/phase54_generator.exe <data> <D1_0x53454540> <tok_mlp_0x53454554> [--gen-len N
//      --temp F --warmup N --seed-start N --rng-seed N --mode argmax --lambda-k3 F --k3-thr N --k3-train-frac F]
// Run (TF):  bin/phase54_generator.exe <data> <D1> <tok_mlp> --tf-eval [--tf-len N --tf-start-frac F --k3-train-frac F]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include "src/silicon_entropy.h"
#include "benchmarks/phase50/bpe_codec.h"

#define CLASSES   256
#define BASE_DIM  SEE_FEATURE_DIM
#define L0_DIM    SEE_L0_DIM
#define L2_DIM    64
#define D1_TOT    (BASE_DIM + L2_DIM)
#define D_EXP_MAX 256
#define N_TS_MAX  4
#define FEAT_MAX  (D1_TOT + N_TS_MAX*D_EXP_MAX)

enum { G_NONE=0, G_PUNCT=1, G_WS=2, G_SURPRISE=3, G_ENTROPY=4, G_COMBINED=5 };

static float Pmat[L2_DIM][BASE_DIM];
static float (*trigram)[CLASSES][CLASSES];
static float (*ent_table)[CLASSES];
static float feat_mean[D1_TOT], feat_std[D1_TOT];
static int   g_gate; static float g_alpha, g_surp_thr, g_ent_thr; static int g_ent_high;
static float g_l2_clamp=2.0f, g_nb_decay=1.0f, g_mix=0.5f, g_l2_scale=0.5f; static int g_cooldown=0;
static int   cd_ctr=0; static float prev_bound[BASE_DIM];

static int g_H=0, g_dim=D1_TOT, g_dexp=0, g_nts=0; static float g_gamma=0.25f; static uint64_t g_pseed=0;
static float g_ts[N_TS_MAX];
static float *W1=NULL,*b1=NULL,*W2=NULL,*b2=NULL;
static float Omega[D_EXP_MAX][L0_DIM], Bvec[D_EXP_MAX];
static int VTOK=1024; static float* g_tbig=NULL;
static Bpe g_bpe;

// ===================== GLOBAL-k3 prior: P(next | last 3 tokens), backoff k3->bigram(1tok)->unigram =====================
// CSR over contexts: a sorted distinct-context array + per-context slice of reduced (next,count) + total.
// Interpolated/count-gated: p(d)=U(d); if ctx1 total>=thr blend bigram; if ctx3 total>=thr blend trigram.
static double* g_uprob=NULL;                 // [VTOK] unigram prob (add-K)
static uint32_t *g3_ctx=NULL,*g3_off=NULL,*g3_tot=NULL; static long g3_n=0;     // distinct ctx3 (30-bit), CSR
static uint16_t *g3_next=NULL; static uint32_t *g3_cnt=NULL;
static uint32_t *g3_hkey=NULL,*g3_hidx=NULL; static long g3_hmask=0;             // open-addr hash ctx3->index
static uint16_t *g1_ctx=NULL; static uint32_t *g1_off=NULL,*g1_tot=NULL; static long g1_n=0;  // distinct ctx1 (10-bit)
static uint16_t *g1_next=NULL; static uint32_t *g1_cnt=NULL;
static int g1_index[1024];                                                       // ctx1 (<1024) -> index, -1 none
static float g_w1=0.5f, g_w3=0.6f;           // backoff blend weights (fixed)

static int cmp_u64(const void* a,const void* b){ uint64_t x=*(const uint64_t*)a,y=*(const uint64_t*)b; return (x>y)-(x<y); }
static long nextpow2(long x){ long p=1; while(p<x) p<<=1; return p; }
static void g3_build_hash(void);

// build CSR for a packed (ctx<<10|next) observation array (already malloc'd, len no). ctxbits = #ctx bits.
// returns distinct-ctx count; fills *octx,*ooff(+1 sentinel),*otot,*onext,*ocnt.
static long build_csr(uint64_t* obs, long no, int ctxbits,
                      uint32_t** octx, uint32_t** ooff, uint32_t** otot, uint16_t** onext, uint32_t** ocnt){
    if(no<=0){ *octx=NULL;*ooff=NULL;*otot=NULL;*onext=NULL;*ocnt=NULL; return 0; }
    qsort(obs,no,sizeof(uint64_t),cmp_u64);
    // reduce consecutive equal keys -> count; count distinct (ctx,next) pairs and distinct ctx
    long npair=0, nctx=0; uint64_t prevk=~0ULL; uint32_t prevctx=0xFFFFFFFFu;
    for(long i=0;i<no;i++){ uint64_t k=obs[i]; if(i==0||k!=prevk){ npair++; uint32_t ctx=(uint32_t)(k>>10);
            if(npair==1||ctx!=prevctx){ nctx++; prevctx=ctx; } prevk=k; } }
    uint32_t* ctx=malloc((size_t)nctx*4); uint32_t* off=malloc((size_t)(nctx+1)*4); uint32_t* tot=malloc((size_t)nctx*4);
    uint16_t* nxt=malloc((size_t)npair*2); uint32_t* cnt=malloc((size_t)npair*4);
    long pi=-1, ci=-1; prevk=~0ULL; prevctx=0xFFFFFFFFu;
    for(long i=0;i<no;i++){ uint64_t k=obs[i]; uint32_t cx=(uint32_t)(k>>10); uint16_t nx=(uint16_t)(k&1023);
        if(i==0||k!=prevk){ pi++; nxt[pi]=nx; cnt[pi]=1;
            if(ci<0||cx!=prevctx){ ci++; ctx[ci]=cx; off[ci]=(uint32_t)pi; tot[ci]=0; prevctx=cx; }
            tot[ci]+=1; prevk=k;
        } else { cnt[pi]++; tot[ci]+=1; }
    }
    off[nctx]=(uint32_t)npair;
    (void)ctxbits;
    *octx=ctx;*ooff=off;*otot=tot;*onext=nxt;*ocnt=cnt; return nctx;
}
static void k3_build(const uint32_t* tok, long ntrain, double addK){
    // unigram
    double* uc=calloc(VTOK,sizeof(double)); for(long i=0;i<ntrain;i++) uc[tok[i]]+=1.0;
    g_uprob=malloc((size_t)VTOK*sizeof(double)); double T=(double)ntrain;
    for(int v=0;v<VTOK;v++) g_uprob[v]=(uc[v]+addK)/(T+addK*VTOK); free(uc);
    // trigram observations: ctx3=(t[i-3]<<20|t[i-2]<<10|t[i-1]), next=t[i]
    long n3=(ntrain>3)?(ntrain-3):0; uint64_t* o3=malloc((size_t)n3*8);
    for(long i=3;i<ntrain;i++){ uint64_t ctx=((uint64_t)tok[i-3]<<20)|((uint64_t)tok[i-2]<<10)|tok[i-1];
        o3[i-3]=(ctx<<10)|tok[i]; }
    g3_n=build_csr(o3,n3,30,&g3_ctx,&g3_off,&g3_tot,&g3_next,&g3_cnt); free(o3);
    g3_build_hash();
    // bigram (1-token ctx): ctx1=t[i-1], next=t[i]
    long n1=(ntrain>1)?(ntrain-1):0; uint64_t* o1=malloc((size_t)n1*8);
    for(long i=1;i<ntrain;i++){ uint64_t ctx=tok[i-1]; o1[i-1]=(ctx<<10)|tok[i]; }
    uint32_t* c1ctx32=NULL; g1_n=build_csr(o1,n1,10,&c1ctx32,&g1_off,&g1_tot,&g1_next,&g1_cnt); free(o1);
    g1_ctx=malloc((size_t)g1_n*2); for(int v=0;v<1024;v++) g1_index[v]=-1;
    for(long i=0;i<g1_n;i++){ g1_ctx[i]=(uint16_t)c1ctx32[i]; g1_index[c1ctx32[i]]=(int)i; } free(c1ctx32);
    fprintf(stderr,"k3 built: ntrain=%ld  distinct ctx3=%ld  ctx1=%ld\n",ntrain,g3_n,g1_n);
}
static void g3_build_hash(void){
    g3_hmask=nextpow2(g3_n*2+16)-1; g3_hkey=malloc((size_t)(g3_hmask+1)*4); g3_hidx=malloc((size_t)(g3_hmask+1)*4);
    for(long i=0;i<=g3_hmask;i++){ g3_hkey[i]=0xFFFFFFFFu; g3_hidx[i]=0xFFFFFFFFu; }
    for(long i=0;i<g3_n;i++){ uint32_t k=g3_ctx[i]; uint64_t h=(k*2654435761ULL)&g3_hmask;
        while(g3_hkey[h]!=0xFFFFFFFFu) h=(h+1)&g3_hmask; g3_hkey[h]=k; g3_hidx[h]=(uint32_t)i; }
}
static int k3_save(const char* path){ FILE* f=fopen(path,"wb"); if(!f) return 0; uint32_t mg=0x4B330001; int vt=VTOK;
    fwrite(&mg,4,1,f); fwrite(&vt,4,1,f); fwrite(&g_w1,4,1,f); fwrite(&g_w3,4,1,f);
    fwrite(g_uprob,sizeof(double),VTOK,f);
    fwrite(&g3_n,sizeof(long),1,f); long np3=g3_off?g3_off[g3_n]:0;
    fwrite(g3_ctx,4,g3_n,f); fwrite(g3_off,4,g3_n+1,f); fwrite(g3_tot,4,g3_n,f);
    fwrite(&np3,sizeof(long),1,f); fwrite(g3_next,2,np3,f); fwrite(g3_cnt,4,np3,f);
    fwrite(&g1_n,sizeof(long),1,f); long np1=g1_off?g1_off[g1_n]:0;
    fwrite(g1_ctx,2,g1_n,f); fwrite(g1_off,4,g1_n+1,f); fwrite(g1_tot,4,g1_n,f);
    fwrite(&np1,sizeof(long),1,f); fwrite(g1_next,2,np1,f); fwrite(g1_cnt,4,np1,f);
    fwrite(g1_index,sizeof(int),1024,f); fclose(f); return 1; }
static int k3_load(const char* path){ FILE* f=fopen(path,"rb"); if(!f) return 0; uint32_t mg=0,vt=0;
    fread(&mg,4,1,f); if(mg!=0x4B330001){ fclose(f); return 0; } fread(&vt,4,1,f); if((int)vt!=VTOK){ fclose(f); return 0; }
    fread(&g_w1,4,1,f); fread(&g_w3,4,1,f);
    g_uprob=malloc((size_t)VTOK*sizeof(double)); fread(g_uprob,sizeof(double),VTOK,f);
    fread(&g3_n,sizeof(long),1,f); long np3=0;
    g3_ctx=malloc((size_t)g3_n*4); g3_off=malloc((size_t)(g3_n+1)*4); g3_tot=malloc((size_t)g3_n*4);
    fread(g3_ctx,4,g3_n,f); fread(g3_off,4,g3_n+1,f); fread(g3_tot,4,g3_n,f);
    fread(&np3,sizeof(long),1,f); g3_next=malloc((size_t)np3*2); g3_cnt=malloc((size_t)np3*4);
    fread(g3_next,2,np3,f); fread(g3_cnt,4,np3,f);
    fread(&g1_n,sizeof(long),1,f); long np1=0;
    g1_ctx=malloc((size_t)g1_n*2); g1_off=malloc((size_t)(g1_n+1)*4); g1_tot=malloc((size_t)g1_n*4);
    fread(g1_ctx,2,g1_n,f); fread(g1_off,4,g1_n+1,f); fread(g1_tot,4,g1_n,f);
    fread(&np1,sizeof(long),1,f); g1_next=malloc((size_t)np1*2); g1_cnt=malloc((size_t)np1*4);
    fread(g1_next,2,np1,f); fread(g1_cnt,4,np1,f);
    fread(g1_index,sizeof(int),1024,f); fclose(f); g3_build_hash();
    fprintf(stderr,"k3 loaded: ctx3=%ld ctx1=%ld\n",g3_n,g1_n); return 1; }
static inline long g3_lookup(uint32_t ctx3){ uint64_t h=(ctx3*2654435761ULL)&g3_hmask;
    while(g3_hkey[h]!=0xFFFFFFFFu){ if(g3_hkey[h]==ctx3) return (long)g3_hidx[h]; h=(h+1)&g3_hmask; } return -1; }
// build dense log-prob vector for context (a,b,c)=last 3 tokens (c=most recent). count-gate thr.
static void k3_logp(uint32_t a,uint32_t b,uint32_t c,int thr,float* out){
    static double* p=NULL; if(!p) p=malloc((size_t)VTOK*sizeof(double));
    for(int d=0;d<VTOK;d++) p[d]=g_uprob[d];
    int i1=(c<1024)?g1_index[c]:-1;                 // bigram backoff on last token
    if(i1>=0 && g1_tot[i1]>=(uint32_t)thr){ double inv=1.0/g1_tot[i1]; double s=1.0-g_w1;
        for(int d=0;d<VTOK;d++) p[d]*=s; uint32_t s0=g1_off[i1],s1=g1_off[i1+1];
        for(uint32_t z=s0;z<s1;z++) p[g1_next[z]]+=g_w1*g1_cnt[z]*inv; }
    long i3=g3_lookup(((uint32_t)a<<20)|((uint32_t)b<<10)|c);
    if(i3>=0 && g3_tot[i3]>=(uint32_t)thr){ double inv=1.0/g3_tot[i3]; double s=1.0-g_w3;
        for(int d=0;d<VTOK;d++) p[d]*=s; uint32_t s0=g3_off[i3],s1=g3_off[i3+1];
        for(uint32_t z=s0;z<s1;z++) p[g3_next[z]]+=g_w3*g3_cnt[z]*inv; }
    for(int d=0;d<VTOK;d++){ double pp=p[d]; out[d]=(float)log(pp>1e-30?pp:1e-30); }
}

static inline float dot_avx(const float* w, const float* f, int n) {
    __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(&w[i]),_mm256_loadu_ps(&f[i]),s);
    float o[8]; _mm256_storeu_ps(o,s);
    float r=o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; for(;i<n;i++) r+=w[i]*f[i]; return r;
}
static uint32_t sample_tok(const float* P, uint64_t* rng) {
    *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
    double u = (*rng >> 11) * (1.0 / (1ULL << 53));
    double c=0; for(int k=0;k<VTOK;k++){ c+=P[k]; if(u<=c) return (uint32_t)k; }
    return (uint32_t)(VTOK-1);
}
static void gen_projection(uint32_t seed) {
    uint64_t s = seed ? (uint64_t)seed : 0x9E3779B97F4A7C15ULL;
    for (int j=0;j<L2_DIM;j++) for (int k=0;k<BASE_DIM;k++){ s^=s<<13; s^=s>>7; s^=s<<17; Pmat[j][k]=(s&1ULL)?1.f:-1.f; }
}
static inline double xs_u01(uint64_t* s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return (double)((*s>>11)&0x1FFFFFFFFFFFFFULL)/(double)(1ULL<<53); }
static void gen_lift(uint64_t seed){ uint64_t s=seed?seed:0xABCDEF12345ULL;
    for(int d=0;d<g_dexp;d++){
        for(int k=0;k<L0_DIM;k++){ double u1=xs_u01(&s); if(u1<1e-12) u1=1e-12; double u2=xs_u01(&s);
            Omega[d][k]=(float)(sqrt(-2.0*log(u1))*cos(6.283185307179586*u2)); }
        Bvec[d]=(float)(6.283185307179586*xs_u01(&s)); } }
static inline int eval_gate(uint8_t byte, uint8_t c1, uint8_t c2) {
    switch (g_gate) {
        case G_NONE:   return 0;
        case G_PUNCT:  return byte=='.'||byte=='!'||byte=='?'||byte=='\n'||byte=='"'||byte=='\'';
        case G_WS:     return byte==' '||byte=='\n'||byte=='\t'||byte=='.'||byte==','||byte=='!'||byte=='?'||byte==';'||byte==':';
        case G_SURPRISE: return (-trigram[c2][c1][byte]) > g_surp_thr;
        case G_ENTROPY:  return g_ent_high ? (ent_table[c2][c1] > g_ent_thr) : (ent_table[c2][c1] < g_ent_thr);
        case G_COMBINED: return (byte=='.'||byte=='!'||byte=='?'||byte=='\n'||byte=='"'||byte=='\'') || ((-trigram[c2][c1][byte]) > g_surp_thr);
    }
    return 0;
}
static inline void l2_evolve(float* L2, const float* fa, int gate) {
    int upd = gate && (cd_ctr==0);
    if (upd) {
        const float* src=fa; static float blend[BASE_DIM];
        if (g_mix>0.0f){ for(int k=0;k<BASE_DIM;k++) blend[k]=fa[k]-g_mix*prev_bound[k]; src=blend; memcpy(prev_bound,fa,BASE_DIM*sizeof(float)); }
        float scale=1.0f/sqrtf((float)BASE_DIM); float w[L2_DIM];
        for (int j=0;j<L2_DIM;j++){ float p=0; const float* pj=Pmat[j]; for(int k=0;k<BASE_DIM;k++) p+=pj[k]*src[k]; w[j]=p*scale; }
        for (int j=0;j<L2_DIM;j++) L2[j]=g_alpha*L2[j]+(1.0f-g_alpha)*w[j];
        cd_ctr=g_cooldown;
    } else {
        if (!gate && g_nb_decay<1.0f) for(int j=0;j<L2_DIM;j++) L2[j]*=g_nb_decay;
        if (cd_ctr>0) cd_ctr--;
    }
}
static inline void norm_feats(const float* raw, float* out){
    for(int fi=0;fi<D1_TOT;fi++){ float x=(raw[fi]-feat_mean[fi])/(feat_std[fi]+1e-8f);
        float cl=(fi<BASE_DIM)?2.0f:g_l2_clamp; if(cl>0){ if(x>cl)x=cl; if(x<-cl)x=-cl; }
        if(fi>=BASE_DIM) x*=g_l2_scale; out[fi]=x; }
}
static inline void armb_fold(const float* feat192, float eB[N_TS_MAX][D_EXP_MAX]){
    float l0n[L0_DIM];
    for(int k=0;k<L0_DIM;k++){ float x=(feat192[k]-feat_mean[k])/(feat_std[k]+1e-8f); if(x>2.f)x=2.f; if(x<-2.f)x=-2.f; l0n[k]=x; }
    for(int d=0;d<g_dexp;d++){ float z=Bvec[d]+g_gamma*dot_avx(Omega[d],l0n,L0_DIM); float cz=cosf(z);
        for(int ts=0;ts<g_nts;ts++){ float a=g_ts[ts]; eB[ts][d]=a*eB[ts][d]+(1.0f-a)*cz; } }
}
static inline void row_bands(const float eB[N_TS_MAX][D_EXP_MAX], float* row){
    for(int ts=0;ts<g_nts;ts++) memcpy(row+D1_TOT+ts*g_dexp, eB[ts], g_dexp*4);
}
static inline void mlp_logits(const float* feat, float* hid, float* lg){
    for(int j=0;j<g_H;j++){ float a=b1[j]+dot_avx(&W1[(size_t)j*g_dim],feat,g_dim); hid[j]=a>0?a:0; }
    for(int c=0;c<VTOK;c++) lg[c]=b2[c]+dot_avx(&W2[(size_t)c*g_H],hid,g_H);
}
// feed one byte through the substrate (armB fold + observe + L2), advance ctx.
static inline void feed_byte(SiliconEntropyState* see, uint8_t b, float* L2, float eB[N_TS_MAX][D_EXP_MAX],
                             uint8_t* ctx1, uint8_t* ctx2){
    float pre[BASE_DIM], fa[BASE_DIM];
    see_extract(see,pre); armb_fold(pre,eB);
    int gate=eval_gate(b,*ctx1,*ctx2);
    see_observe(see,b); see_extract(see,fa);
    l2_evolve(L2,fa,gate);
    *ctx2=*ctx1; *ctx1=b;
}

int main(int argc, char** argv) {
    if (argc<4){ fprintf(stderr,"Usage: %s <data> <D1_w> <tok_mlp> [opts]\n",argv[0]); return 1; }
    int gen_len=2000, warmup=5000, seed_start=0, argmax=0, rng_time=0;
    float temp=0.65f; uint64_t rng_seed=0x243F6A8885A308D3ULL;
    float lambda_k3=0.0f; int k3_thr=5; float k3_train_frac=0.90f; long k3_max_bytes=0;
    int tf_eval=0; long tf_len=80000; float tf_start_frac=0.93f;
    const char* k3_save_path=NULL; const char* k3_load_path=NULL;
    for (int i=4;i<argc;i++){
        if      (!strcmp(argv[i],"--gen-len")   && i+1<argc) gen_len=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--temp")      && i+1<argc) temp=(float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--warmup")    && i+1<argc) warmup=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--seed-start")&& i+1<argc) seed_start=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--rng-seed")  && i+1<argc){ const char* v=argv[++i]; if(!strcmp(v,"-1")) rng_time=1; else rng_seed=strtoull(v,NULL,0); }
        else if (!strcmp(argv[i],"--mode")      && i+1<argc) argmax=!strcmp(argv[++i],"argmax");
        else if (!strcmp(argv[i],"--lambda-k3") && i+1<argc) lambda_k3=(float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--k3-thr")    && i+1<argc) k3_thr=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--k3-train-frac")&& i+1<argc) k3_train_frac=(float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--k3-max-bytes")&& i+1<argc) k3_max_bytes=atol(argv[++i]);
        else if (!strcmp(argv[i],"--tf-eval"))  tf_eval=1;
        else if (!strcmp(argv[i],"--tf-len")    && i+1<argc) tf_len=atol(argv[++i]);
        else if (!strcmp(argv[i],"--tf-start-frac")&& i+1<argc) tf_start_frac=(float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--k3-save")   && i+1<argc) k3_save_path=argv[++i];
        else if (!strcmp(argv[i],"--k3-load")   && i+1<argc) k3_load_path=argv[++i];
    }
    int k3_on = (tf_eval || lambda_k3>0.0f || k3_save_path);

    FILE* fd=fopen(argv[1],"rb"); if(!fd){fprintf(stderr,"Cannot open %s\n",argv[1]);return 1;}
    fseek(fd,0,SEEK_END); long fsz=ftell(fd); fseek(fd,0,SEEK_SET);
    long need=(long)seed_start+warmup+512; if(need>fsz){need=fsz; warmup=(int)(fsz-seed_start-512); if(warmup<0)warmup=0;}
    long load_bytes = need;
    if(k3_on){ load_bytes = (k3_max_bytes>0 && k3_max_bytes<fsz)? k3_max_bytes : fsz; if(load_bytes<need) load_bytes=need; }
    uint8_t* fdat=malloc(load_bytes); fread(fdat,1,load_bytes,fd); fclose(fd);
    long corp_bytes=load_bytes;

    // ---- D1 substrate (0x53454540) ----
    FILE* fw=fopen(argv[2],"rb"); if(!fw){fprintf(stderr,"Cannot open weights %s\n",argv[2]);return 1;}
    uint32_t magic; fread(&magic,4,1,fw); rewind(fw);
    if (magic!=0x53454540){ fprintf(stderr,"Expected D1 0x53454540, got 0x%08x\n",magic); return 1; }
    uint32_t hdr4[4]; fread(hdr4,4,4,fw);
    float hf[5]; fread(hf,4,5,fw);
    float decay=hf[0], alpha_fast=hf[2], feat_clamp=hf[3];
    uint32_t no=0; fread(&no,4,1,fw); int n_oja=(int)no;
    if (n_oja<0||n_oja>SEE_N_OJA_MAX){ fprintf(stderr,"bad n_oja %d\n",n_oja); return 1; }
    float W_oja_buf[SEE_N_OJA_MAX*43]; fread(W_oja_buf,sizeof(float),(size_t)n_oja*43,fw);
    uint32_t l2d=0,gt=0,eh=0,ps=0; float al=0,st=0,et=0;
    fread(&l2d,4,1,fw); fread(&gt,4,1,fw); fread(&al,4,1,fw);
    fread(&st,4,1,fw); fread(&et,4,1,fw); fread(&eh,4,1,fw); fread(&ps,4,1,fw);
    if ((int)l2d!=L2_DIM){ fprintf(stderr,"L2_DIM mismatch %u\n",l2d); return 1; }
    g_gate=(int)gt; g_alpha=al; g_surp_thr=st; g_ent_thr=et; g_ent_high=(int)eh;
    float l2c=0,nbd=1.0f; uint32_t cd=0,dl=0;
    fread(&l2c,4,1,fw); fread(&nbd,4,1,fw); fread(&cd,4,1,fw); fread(&dl,4,1,fw);
    g_l2_clamp=(l2c>0.0f)?l2c:feat_clamp; g_nb_decay=nbd; g_cooldown=(int)cd;
    float mx0=0.5f; fread(&mx0,4,1,fw); g_mix=mx0;
    float ls=0.5f; fread(&ls,4,1,fw); g_l2_scale=(ls>0.0f)?ls:1.0f;
    float l2cap=0; fread(&l2cap,4,1,fw);
    size_t tri_n=(size_t)CLASSES*CLASSES*CLASSES;
    trigram=malloc(tri_n*sizeof(float)); fread(trigram,sizeof(float),tri_n,fw);
    fread(feat_mean,sizeof(float),D1_TOT,fw);
    fread(feat_std, sizeof(float),D1_TOT,fw);
    fclose(fw);

    // ---- token MLP (0x53454554): armB header + VTOK + merges + tbig + weights ----
    FILE* fm=fopen(argv[3],"rb"); if(!fm){fprintf(stderr,"Cannot open mlp %s\n",argv[3]);return 1;}
    uint32_t mm=0,mh=0,mdim=0,mbase=0,mdexp=0,mnts=0; fread(&mm,4,1,fm);
    if (mm!=0x53454554){ fprintf(stderr,"Expected token 0x53454554, got 0x%08x\n",mm); return 1; }
    fread(&mh,4,1,fm); fread(&mdim,4,1,fm); fread(&mbase,4,1,fm); fread(&mdexp,4,1,fm); fread(&mnts,4,1,fm);
    fread(&g_gamma,4,1,fm); fread(&g_pseed,8,1,fm);
    g_H=(int)mh; g_dim=(int)mdim; g_dexp=(int)mdexp; g_nts=(int)mnts;
    if (mbase!=D1_TOT){ fprintf(stderr,"base dim %u != %d\n",mbase,D1_TOT); return 1; }
    if (g_dexp>D_EXP_MAX||g_nts>N_TS_MAX){ fprintf(stderr,"lift too big\n"); return 1; }
    if (g_dim!=D1_TOT+g_nts*g_dexp){ fprintf(stderr,"dim mismatch\n"); return 1; }
    for(int t=0;t<g_nts;t++) fread(&g_ts[t],4,1,fm);
    uint32_t vt=0,nm=0; fread(&vt,4,1,fm); fread(&nm,4,1,fm); VTOK=(int)vt;
    uint32_t *mA=malloc((size_t)nm*4), *mB=malloc((size_t)nm*4);
    for(uint32_t r=0;r<nm;r++){ fread(&mA[r],4,1,fm); fread(&mB[r],4,1,fm); }
    bpe_load_arrays(&g_bpe,VTOK,(int)nm,mA,mB); free(mA); free(mB);
    g_tbig=malloc((size_t)VTOK*VTOK*4); fread(g_tbig,4,(size_t)VTOK*VTOK,fm);
    W1=malloc((size_t)g_H*g_dim*4); b1=malloc(g_H*4); W2=malloc((size_t)VTOK*g_H*4); b2=malloc((size_t)VTOK*4);
    fread(W1,4,(size_t)g_H*g_dim,fm); fread(b1,4,g_H,fm); fread(W2,4,(size_t)VTOK*g_H,fm); fread(b2,4,VTOK,fm);
    fclose(fm);

    gen_projection(ps);
    gen_lift(g_pseed);
    ent_table=malloc(CLASSES*CLASSES*sizeof(float));
    for(int i=0;i<CLASSES;i++) for(int j=0;j<CLASSES;j++){
        float m=-1e9f; for(int k=0;k<CLASSES;k++) if(trigram[i][j][k]>m) m=trigram[i][j][k];
        double se=0; for(int k=0;k<CLASSES;k++) se+=exp((double)(trigram[i][j][k]-m));
        double H=0; for(int k=0;k<CLASSES;k++){ double p=exp((double)(trigram[i][j][k]-m))/se; if(p>1e-12) H-=p*log(p); }
        ent_table[i][j]=(float)H;
    }

    // ---- GLOBAL-k3: load cached tables, or tokenize corpus + build counts on the TRAIN split ----
    uint32_t* g_tok=NULL; long g_ntok=0; long* g_tokstart=NULL; long k3_train_end=0;
    if(k3_load_path && !tf_eval){
        if(!k3_load(k3_load_path)){ fprintf(stderr,"k3 load failed: %s\n",k3_load_path); return 1; }
    } else if(k3_on){
        g_tok=malloc((size_t)corp_bytes*sizeof(uint32_t));
        g_ntok=(long)bpe_encode_region(&g_bpe,fdat,0,corp_bytes,g_tok);
        g_tokstart=malloc((size_t)(g_ntok+1)*sizeof(long));
        long off=0; for(long i=0;i<g_ntok;i++){ g_tokstart[i]=off; off+=bpe_tok_len(&g_bpe,g_tok[i]); } g_tokstart[g_ntok]=off;
        k3_train_end=(long)(g_ntok*k3_train_frac);
        fprintf(stderr,"corpus tokenized: %ld tokens / %ld bytes; k3 train_end=%ld (frac %.2f)\n",
                g_ntok,corp_bytes,k3_train_end,k3_train_frac);
        k3_build(g_tok,k3_train_end,0.5);
        if(k3_save_path){ if(k3_save(k3_save_path)) fprintf(stderr,"k3 saved -> %s\n",k3_save_path); }
    }
    if(k3_save_path && !lambda_k3 && !tf_eval) return 0;   // build-and-save-only mode

    // ============================ TF-EVAL MODE (teacher-forced held-out BPB sweep) ============================
    if(tf_eval){
        long tf_start=(long)(g_ntok*tf_start_frac);
        if(tf_start < k3_train_end) tf_start=k3_train_end;          // held-out only
        long tf_end=tf_start+tf_len; if(tf_end> g_ntok-1) tf_end=g_ntok-1;
        long byte_start=g_tokstart[tf_start];
        fprintf(stderr,"TF-eval: tokens [%ld,%ld) (held-out, start_frac %.2f), byte_start=%ld\n",tf_start,tf_end,tf_start_frac,byte_start);
        // run the substrate forward ONCE over the window; cache base logits (mlp+tb) + context + true-next + stratum.
        SiliconEntropyState see; see_init(&see,42,4,decay); see.multiscale_mode=1; see.alpha_fast=alpha_fast;
        see.alpha_mid=0.9f; see.alpha_slow=0.99f; see.n_oja=n_oja; memcpy(see.W_oja,W_oja_buf,(size_t)n_oja*43*sizeof(float));
        see.eta_oja=0.0f; see.plastic_blend=1.0f;
        float L2[L2_DIM]; memset(L2,0,sizeof L2); static float eB[N_TS_MAX][D_EXP_MAX]; memset(eB,0,sizeof eB);
        memset(prev_bound,0,sizeof prev_bound); cd_ctr=0; uint8_t c1=0,c2=0;
        see_reset(&see);
        long warm0=byte_start-200000; if(warm0<0) warm0=0;                  // warm the substrate before the window
        for(long i=warm0;i<byte_start;i++) feed_byte(&see,fdat[i],L2,eB,&c1,&c2);
        long W=tf_end-tf_start;
        float* base=malloc((size_t)W*VTOK*4);    // mlp+tb per token
        uint32_t* a3=malloc((size_t)W*4),*b3=malloc((size_t)W*4),*c3=malloc((size_t)W*4),*tn=malloc((size_t)W*4);
        int* strat=malloc((size_t)W*4); int* nby=malloc((size_t)W*4);
        long* lastpos=malloc((size_t)VTOK*sizeof(long)); for(int v=0;v<VTOK;v++) lastpos[v]=-1;
        float raw[D1_TOT], feat[FEAT_MAX]; float* hid=malloc(g_H*4); float* mlg=malloc((size_t)VTOK*4);
        long ti=tf_start; long w=0;
        while(ti<tf_end){
            see_extract(&see,raw); memcpy(raw+BASE_DIM,L2,L2_DIM*sizeof(float));
            norm_feats(raw,feat); row_bands(eB,feat); mlp_logits(feat,hid,mlg);
            const float* tb=&g_tbig[(size_t)g_tok[ti]*VTOK];
            float* bp=&base[(size_t)w*VTOK]; for(int cc=0;cc<VTOK;cc++) bp[cc]=mlg[cc]+tb[cc];
            a3[w]=(ti>=3)?g_tok[ti-2]:0; b3[w]=(ti>=2)?g_tok[ti-1]:0; c3[w]=g_tok[ti];   // last3 ctx for predicting tok[ti+1]
            uint32_t nt=g_tok[ti+1]; tn[w]=nt;
            long lp=lastpos[nt]; int d=(lp>=0)?(int)(w-lp):-1;
            strat[w]= (d<0)?0 : (d<=4)?1 : (d<=32)?2 : (d<=200)?3 : 4;          // 0=NONREC,1..4 dist bins
            lastpos[nt]=w;
            nby[w]=bpe_tok_len(&g_bpe,nt);
            int L=bpe_tok_len(&g_bpe,g_tok[ti]); const unsigned char* eb=bpe_tok_bytes(&g_bpe,g_tok[ti]);
            for(int k=0;k<L;k++) feed_byte(&see,eb[k],L2,eB,&c1,&c2);
            ti++; w++;
        }
        long WN=w; free(hid);free(mlg);
        // sweep lambda x thr over the cached window (cheap)
        float lambdas[]={0.0f,0.25f,0.5f,1.0f,2.0f}; int nl=5;
        int thrs[]={2,5,10}; int nthr=3;
        const char* SN[5]={"NONREC","d1-4","d5-32","d33-200","d>200"};
        printf("\n==== Phase 54 TF-EVAL: held-out token BPB with GLOBAL-k3 prior (window=%ld tok) ====\n",WN);
        printf("bits/token by distance stratum + overall bits/BYTE. lambda=0 = baseline (mlp+bigram, no k3).\n");
        float* lg=malloc((size_t)VTOK*4); float* k3v=malloc((size_t)VTOK*4);
        for(int it=0; it<nthr; it++){
            // precompute k3 logp per token once per thr (depends on thr only), cache compactly? recompute inline.
            for(int il=0; il<nl; il++){ float lam=lambdas[il];
                double sbits[5]={0,0,0,0,0}; long scnt[5]={0,0,0,0,0}; double tot_bits=0; long tot_bytes=0;
                for(long i=0;i<WN;i++){
                    const float* bp=&base[(size_t)i*VTOK];
                    if(lam>0.0f){ k3_logp(a3[i],b3[i],c3[i],thrs[it],k3v);
                        for(int cc=0;cc<VTOK;cc++) lg[cc]=bp[cc]+lam*k3v[cc]; }
                    else for(int cc=0;cc<VTOK;cc++) lg[cc]=bp[cc];
                    float mx=-1e30f; for(int cc=0;cc<VTOK;cc++) if(lg[cc]>mx)mx=lg[cc];
                    double Z=0; for(int cc=0;cc<VTOK;cc++) Z+=exp((double)(lg[cc]-mx));
                    double pt=exp((double)(lg[tn[i]]-mx))/Z; double b=-log2(pt>1e-30?pt:1e-30);
                    int s=strat[i]; sbits[s]+=b; scnt[s]++; tot_bits+=b; tot_bytes+=nby[i];
                }
                double bpb=tot_bits/(double)tot_bytes;
                printf("  thr=%2d lam=%.2f | BPB=%.4f |", thrs[it],lam,bpb);
                for(int s=0;s<5;s++) printf(" %s=%.4f", SN[s], scnt[s]?sbits[s]/scnt[s]:0.0);
                printf("\n");
                if(il==0) printf("  ----- (lam=0 baseline above; deltas = baseline - k3, positive=k3 helps) -----\n");
            }
            printf("\n");
        }
        free(lg);free(k3v);free(base);free(a3);free(b3);free(c3);free(tn);free(strat);free(nby);free(lastpos);
        printf("(read: medium-range strata d33-200/d>200 should DROP with k3; report the curve. lower BPB = compression win.)\n");
        return 0;
    }

    SiliconEntropyState see;
    see_init(&see, 42, 4, decay);
    see.multiscale_mode=1; see.alpha_fast=alpha_fast; see.alpha_mid=0.9f; see.alpha_slow=0.99f;
    see.n_oja=n_oja; memcpy(see.W_oja,W_oja_buf,(size_t)n_oja*43*sizeof(float));
    see.eta_oja=0.0f; see.plastic_blend=1.0f;
    fprintf(stderr,"50.A gen: D1 alpha=%.2f | armB H=%d dim=%d dexp=%d nts=%d | TOK VTOK=%d nmerge=%d temp=%.2f\n",
            g_alpha,g_H,g_dim,g_dexp,g_nts,VTOK,(int)nm,temp);

    float L2[L2_DIM]; memset(L2,0,sizeof(L2));
    static float eB[N_TS_MAX][D_EXP_MAX]; memset(eB,0,sizeof(eB));
    memset(prev_bound,0,sizeof(prev_bound)); cd_ctr=0;
    uint8_t ctx1=0, ctx2=0;

    see_reset(&see);
    for (int i=seed_start;i<seed_start+warmup;i++) feed_byte(&see,fdat[i],L2,eB,&ctx1,&ctx2);
    int soff=seed_start+warmup; int slen=(int)(need-soff); if(slen>512) slen=512;
    for (int i=0;i<slen;i++) feed_byte(&see,fdat[soff+i],L2,eB,&ctx1,&ctx2);

    // initial context = last 3 tokens of the seed region (prevtok=last, pt2,pt3 = before)
    uint32_t prevtok=0, pt2=0, pt3=0;
    if(slen>0){ uint32_t* st=malloc((size_t)slen*4); size_t ns=bpe_encode_region(&g_bpe,fdat+soff,0,slen,st);
        if(ns>0) prevtok=st[ns-1]; if(ns>1) pt2=st[ns-2]; if(ns>2) pt3=st[ns-3]; free(st); }

    fprintf(stderr,"--- generated ---\n");
    uint64_t rng = rng_time ? ((uint64_t)time(NULL)^0xdeadbeefcafeULL^(uint64_t)seed_start)
                            : (rng_seed ^ ((uint64_t)seed_start*0x9E3779B97F4A7C15ULL));
    if (rng==0) rng=0x9E3779B97F4A7C15ULL;

    uint8_t* gen=malloc(gen_len+64); long gn=0; double self_bits=0; long ntok=0;
    float raw[D1_TOT], feat[FEAT_MAX]; float* hid=malloc(g_H*4); float* mlg=malloc((size_t)VTOK*4); float* Pp=malloc((size_t)VTOK*4);
    float* k3v = k3_on ? malloc((size_t)VTOK*4) : NULL;
    while (gn<gen_len){
        see_extract(&see, raw); memcpy(raw+BASE_DIM,L2,L2_DIM*sizeof(float));
        norm_feats(raw,feat); row_bands(eB,feat);          // base + armB bands (eB BEFORE folding next token)
        mlp_logits(feat,hid,mlg);
        const float* tb=&g_tbig[(size_t)prevtok*VTOK];
        if(lambda_k3>0.0f){ k3_logp(pt3,pt2,prevtok,k3_thr,k3v);
            float mxl=-1e30f; for(int c=0;c<VTOK;c++){ mlg[c]=(mlg[c]+tb[c]+lambda_k3*k3v[c])/temp; if(mlg[c]>mxl)mxl=mlg[c]; }
            float Z=0; for(int c=0;c<VTOK;c++){ Pp[c]=expf(mlg[c]-mxl); Z+=Pp[c]; } for(int c=0;c<VTOK;c++) Pp[c]/=Z;
        } else {
            float mxl=-1e30f; for(int c=0;c<VTOK;c++){ mlg[c]=(mlg[c]+tb[c])/temp; if(mlg[c]>mxl)mxl=mlg[c]; }
            float Z=0; for(int c=0;c<VTOK;c++){ Pp[c]=expf(mlg[c]-mxl); Z+=Pp[c]; } for(int c=0;c<VTOK;c++) Pp[c]/=Z;
        }
        uint32_t tok;
        if(argmax){ float bb=-1; tok=0; for(int c=0;c<VTOK;c++) if(Pp[c]>bb){bb=Pp[c];tok=(uint32_t)c;} }
        else tok=sample_tok(Pp,&rng);
        self_bits += -log2((double)fmaxf(Pp[tok],1e-30f)); ntok++;
        // emit token bytes
        int L=bpe_tok_len(&g_bpe,tok); const unsigned char* eb=bpe_tok_bytes(&g_bpe,tok);
        for(int k=0;k<L && gn<gen_len;k++){ gen[gn++]=eb[k]; feed_byte(&see,eb[k],L2,eB,&ctx1,&ctx2); }
        pt3=pt2; pt2=prevtok; prevtok=tok;
    }
    fwrite(gen,1,gen_len,stdout);
    // byte-guard counters on the emitted bytes (same definitions as gate-v2 / Phase 47)
    { int maxWs=0,maxCh=0,ws=0,ch=0,wsCnt=0,nonPrint=0,prev=-1;
      for(long i=0;i<gn;i++){ int b=gen[i];
        if(b==32||b==9||b==10||b==13){ wsCnt++; ws++; if(ws>maxWs)maxWs=ws; ch=0; }
        else { ws=0; if(b==prev)ch++; else ch=1; if(ch>maxCh)maxCh=ch; if(b<32||b>126)nonPrint++; }
        prev=b; }
      fprintf(stderr,"\n--- stats ---\nself_BPB: %.4f\n", self_bits/(double)gn);
      fprintf(stderr,"byte_guard: wsRun=%d chRun=%d wsFrac=%.4f nonPrint=%d\n",maxWs,maxCh,(double)wsCnt/(double)gn,nonPrint);
      fprintf(stderr,"k3: lambda=%.3f thr=%d | tokens=%ld bytes=%ld bytes/tok=%.3f\n",lambda_k3,k3_thr,ntok,gn,(double)gn/ntok);
    }
    free(trigram); free(ent_table); free(fdat); free(gen);
    return 0;
}
