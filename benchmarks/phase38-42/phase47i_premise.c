// Phase 47.I step 2a - Premise Probe: do LONG bursts enter the wasteland? (NO training)
//
// Human read of P_r7 samples (user, 2026-06-12) found a GOODHART failure: three byte-level
// pathologies invisible to all word-level gate metrics — whitespace floods (10-50+ spaces),
// the "wasteland" (diffuse attractor: template fragments like "..nhat sh"/"..nke go" with
// surface variation, so no single bigram dominates and topBi stays low), char floods
// ("Jaaaaa...", 30-50 chars = one weird word, runWst silent) + non-printable bytes.
// Temporal pattern: wasteland appears in the SECOND HALF of samples, after ~500-1000 bytes
// of self-generation -> FAR-FIELD attractor. Training bursts K=16 always start from
// corpus-anchored context and never get there: the decoder never learned to exit a region
// it never entered during training. DAgger cured the near-field; optimization pushed the
// residue where the proxies don't look.
//
// This probe tests the COVERAGE hypothesis without spending any training: pure extraction
// with the P_r7 decoder, bursts K=128 (vs K=16 control) every 256 bytes. For each burst,
// the TAIL (last min(64,K) bytes) is scored with the byte-guards:
//   wasteland-entry := wsRun>8 OR charRun>8 OR wsFrac>30% in the tail.
// If long bursts NEVER enter the wasteland, the coverage hypothesis dies here and 47.I
// closes without training (pre-registered rule in phase47i.ps1: proceed iff K128 rate >= 1%).
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase38-42/phase47i_premise.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase47i_premise.exe -lm -I .
// Run:
//   bin/phase47i_premise.exe <data> <D1_w> <mlp_w> [--len N]

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
#define S_DIM     (BASE_DIM + L2_DIM)
#define D1_TOT    (BASE_DIM + L2_DIM)
#define T_GEN     0.65f
#define PERIOD    256
#define TAILMAX   64

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
static inline uint8_t sample_p(const float* P, uint64_t* rng){
    *rng^=*rng<<13; *rng^=*rng>>7; *rng^=*rng<<17;
    double u=(*rng>>11)*(1.0/(1ULL<<53));
    double c=0; for(int k=0;k<CLASSES;k++){ c+=P[k]; if(u<=c) return (uint8_t)k; }
    return CLASSES-1;
}

typedef struct { int in,h; float *W1,*b1,*W2,*b2; } MLP;
static int mlp_load(MLP* m,const char* path){
    FILE* f=fopen(path,"rb"); if(!f){ fprintf(stderr,"open %s\n",path); return 0; }
    uint32_t magic=0,H=0,o=0,d=0;
    fread(&magic,4,1,f); fread(&H,4,1,f); fread(&o,4,1,f); fread(&d,4,1,f);
    if(magic!=0x53454545||d!=S_DIM){ fprintf(stderr,"bad mlp %s\n",path); fclose(f); return 0; }
    m->in=(int)d; m->h=(int)H;
    m->W1=malloc((size_t)H*d*4); m->b1=malloc(H*4); m->W2=malloc((size_t)CLASSES*H*4); m->b2=malloc(CLASSES*4);
    fread(m->W1,4,(size_t)H*d,f); fread(m->b1,4,H,f); fread(m->W2,4,(size_t)CLASSES*H,f); fread(m->b2,4,CLASSES,f);
    fclose(f); return 1;
}
static void mlp_fwd(MLP* m,const float* x,float* hid,float* lg){
    for(int j=0;j<m->h;j++){ float a=m->b1[j]+dot_avx(&m->W1[(size_t)j*m->in],x,m->in); hid[j]=a>0?a:0; }
    for(int c=0;c<CLASSES;c++) lg[c]=m->b2[c]+dot_avx(&m->W2[(size_t)c*m->h],hid,m->h);
}

// byte-guards on a buffer (same definitions as gate v2 in phase47i.ps1)
static inline int is_ws(uint8_t b){ return b==32||b==9||b==10||b==13; }
static void byte_guards(const uint8_t* buf,int n,int* o_wsrun,int* o_chrun,double* o_wsfrac,int* o_nonprint){
    int mws=0,mch=0,ws=0,ch=0,wsc=0,np=0; int prev=-1;
    for(int i=0;i<n;i++){ uint8_t b=buf[i];
        if(is_ws(b)){ wsc++; ws++; if(ws>mws)mws=ws; ch=0; }
        else { ws=0; if(b==prev) ch++; else ch=1; if(ch>mch)mch=ch; }
        if(!(b==9||b==10||b==13||(b>=32&&b<=126))) np++;
        prev=b; }
    *o_wsrun=mws; *o_chrun=mch; *o_wsfrac=(double)wsc/n; *o_nonprint=np;
}

// burst extraction with per-burst tail scoring (substrate follows the dirty trail, 47.D parity)
static void probe_K(SiliconEntropyState* see, MLP* m, long start, long N, int K, uint64_t rseed, const char* tag){
    float L2d1[L2_DIM]={0}, pb_d1[BASE_DIM]={0};
    float feat192[BASE_DIM], fa[BASE_DIM], rawd1[D1_TOT], nf[D1_TOT];
    float* hid=malloc(m->h*4); float lg[CLASSES],Pp[CLASSES];
    float scale=1.0f/sqrtf((float)BASE_DIM);
    uint64_t rng=rseed?rseed:0x9E3779B97F4A7C15ULL; int burst=0;
    uint8_t bbuf[1024]; int blen=0;
    long nburst=0,nwaste=0,nwsr=0,nchr=0,nwsf=0; double selfbits=0; long nself=0;
    int wsr_max=0,chr_max=0; double wsf_max=0; long npn=0;
    see_reset(see); for(long i=0;i<=start+1;i++) see_observe(see,g_data[i]);
    uint8_t cur_c2=g_data[start], cur_c1=g_data[start+1];
    for(long i=0;i<N;i++){
        long g=start+i; uint8_t t=g_data[g+2];
        see_extract(see,feat192);
        memcpy(rawd1,feat192,BASE_DIM*4); memcpy(rawd1+BASE_DIM,L2d1,L2_DIM*4);
        norm_feats(rawd1,nf);
        if(i>0 && (i%PERIOD)==0){ burst=K; blen=0; }
        uint8_t ob=t;
        if(burst>0){
            mlp_fwd(m,nf,hid,lg);
            const float* tri=&trigram[cur_c2][cur_c1][0];
            float mx=-1e30f; for(int c=0;c<CLASSES;c++){ lg[c]=(lg[c]+tri[c])/T_GEN; if(lg[c]>mx)mx=lg[c]; }
            float Z=0; for(int c=0;c<CLASSES;c++){ Pp[c]=expf(lg[c]-mx); Z+=Pp[c]; } for(int c=0;c<CLASSES;c++) Pp[c]/=Z;
            ob=sample_p(Pp,&rng);
            selfbits+=-log2((double)fmaxf(Pp[ob],1e-30f)); nself++;
            if(blen<(int)sizeof(bbuf)) bbuf[blen++]=ob;
            burst--;
            if(burst==0){
                int tail=(blen<TAILMAX)?blen:TAILMAX;
                int wsr,chr,np; double wsf;
                byte_guards(bbuf+blen-tail,tail,&wsr,&chr,&wsf,&np);
                nburst++;
                int waste=(wsr>8)||(chr>8)||(wsf>0.30);
                if(waste) nwaste++;
                if(wsr>8) nwsr++; if(chr>8) nchr++; if(wsf>0.30) nwsf++;
                if(wsr>wsr_max)wsr_max=wsr; if(chr>chr_max)chr_max=chr; if(wsf>wsf_max)wsf_max=wsf;
                npn+=np;
            }
        }
        see_observe(see,ob); see_extract(see,fa);
        if(ent_gate(cur_c1,cur_c2)){ float src5[BASE_DIM];
            for(int k=0;k<BASE_DIM;k++) src5[k]=fa[k]-0.5f*pb_d1[k]; memcpy(pb_d1,fa,BASE_DIM*4);
            for(int j=0;j<L2_DIM;j++){ float p5=0; const float* pj=Pmat[j]; for(int k=0;k<BASE_DIM;k++) p5+=pj[k]*src5[k];
                L2d1[j]=g_alpha*L2d1[j]+(1.0f-g_alpha)*p5*scale; } }
        cur_c2=cur_c1; cur_c1=ob;
    }
    free(hid);
    printf("PREMISE win=%s K=%d bursts=%ld waste%%=%.2f wsr%%=%.2f chr%%=%.2f wsf%%=%.2f wsrMax=%d chrMax=%d wsfMax=%.2f nonPrint=%ld selfBPB=%.3f\n",
           tag,K,nburst,100.0*nwaste/(nburst?nburst:1),100.0*nwsr/(nburst?nburst:1),100.0*nchr/(nburst?nburst:1),
           100.0*nwsf/(nburst?nburst:1),wsr_max,chr_max,wsf_max,npn,(nself>0)?selfbits/nself:0.0);
}

int main(int argc, char** argv){
    if(argc<4){ fprintf(stderr,"Usage: %s <data> <D1_w> <mlp_w> [--len N]\n",argv[0]); return 1; }
    setvbuf(stdout,NULL,_IONBF,0);
    long N=1000000;
    for(int i=4;i<argc;i++){ if(!strcmp(argv[i],"--len")&&i+1<argc)N=atol(argv[++i]); }
    FILE* fd=fopen(argv[1],"rb"); if(!fd){fprintf(stderr,"data\n");return 1;} fseek(fd,0,SEEK_END); g_fsz=ftell(fd); fseek(fd,0,SEEK_SET);
    g_data=malloc(g_fsz); fread(g_data,1,g_fsz,fd); fclose(fd);
    SiliconEntropyState see;
    if(!load_d1(argv[2],&see)) return 1;
    MLP m; if(!mlp_load(&m,argv[3])) return 1;
    ent_table=malloc(CLASSES*CLASSES*4);
    for(int i=0;i<CLASSES;i++) for(int j=0;j<CLASSES;j++){ float mm=-1e9f; for(int k=0;k<CLASSES;k++) if(trigram[i][j][k]>mm)mm=trigram[i][j][k];
        double se=0; for(int k=0;k<CLASSES;k++) se+=exp((double)(trigram[i][j][k]-mm));
        double Hh=0; for(int k=0;k<CLASSES;k++){ double p=exp((double)(trigram[i][j][k]-mm))/se; if(p>1e-12)Hh-=p*log(p); }
        ent_table[i][j]=(float)Hh; }
    long tr_start=g_fsz/5, v1_start=g_fsz/2;
    if(v1_start+N+3>g_fsz){ fprintf(stderr,"window out of file\n"); return 1; }
    printf("==== 47.I premise probe (P_r7 decoder, wasteland-entry of burst tails) ====\n");
    printf("wasteland := tail(last %d B) has wsRun>8 OR charRun>8 OR wsFrac>30%%\n",TAILMAX);
    // K16 control vs K128, two windows; same seed family per (window,K)
    probe_K(&see,&m,tr_start,N, 16,0x147A0000ULL^16ULL, "train");
    probe_K(&see,&m,tr_start,N,128,0x147A0000ULL^128ULL,"train");
    probe_K(&see,&m,v1_start,N, 16,0x147A1111ULL^16ULL, "val1");
    probe_K(&see,&m,v1_start,N,128,0x147A1111ULL^128ULL,"val1");
    printf("\nReading: if K=128 waste%% ~ 0 the coverage hypothesis is DEAD (the far-field is\n");
    printf("not reachable from anchored bursts even at 128 bytes) -> 47.I closes without\n");
    printf("training. Pre-registered proceed rule (phase47i.ps1): K128 waste%% >= 1.0.\n");
    return 0;
}
