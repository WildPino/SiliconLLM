// Phase 46.B0 - LOWMARGIN firing diagnostic (closed-loop, NO training)
//
// 46.A is a MIXED signal: A1 PUNCT compresses (BPB 2.2511 < D1), A3 LOWMARGIN partially
// stabilizes at T=0.55 (topBi 17->11, altLp 6->2, runWst 8->4) at ~no BPB cost. But A3's L3
// update_freq in closed-loop is only ~0.02% - far below the 46.0 teacher-forced expectation
// (LOWMARGIN gap p50~130). 46.B0 asks: bug, or does the trigram-margin p20 (calibrated on
// teacher-forced text) almost never fire in closed-loop at T=0.55?
//
// Loads a 0x53454544 weight (A3 lowmargin, or A0 as the D1 proxy). Runs N closed-loop seeds,
// recording per step the trigram prob-margin margin_table[ctx2][ctx1] and the actual L3
// firing. Also scans the teacher-forced val margin distribution (context-only). Prints:
//   - per-seed L3 update count + total/freq
//   - real gap p50/p90 (only seeds with >=2 events)
//   - trigram-margin quantiles teacher-forced vs closed-loop (p10/p20/p30/p40/p50)
//   - inference-only re-threshold: LOWMARGIN firing count at p30/p40/p50 of the closed-loop
//     margin (re-run the schedule state machine on the recorded margin seq, no regeneration)
//   - first 10 L3 events of the loopiest seed (lowest self-BPB) with generated context
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase38-42/phase46_b0_margin.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase46_b0_margin.exe -lm -I .
// Run:
//   bin/phase46_b0_margin.exe <data> <weights_0x53454544> [--seeds N --gen-len N --warmup N --temp F]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <immintrin.h>
#include "src/silicon_entropy.h"

#define CLASSES   256
#define BASE_DIM  SEE_FEATURE_DIM
#define L2_DIM    64
#define L3_DIM    64
#define TOT_DIM   (BASE_DIM + L2_DIM + L3_DIM)

enum { G_NONE=0, G_PUNCT=1, G_WS=2, G_SURPRISE=3, G_ENTROPY=4, G_COMBINED=5 };
enum { L3_NONE=0, L3_PUNCT=1, L3_L2CLUST=2, L3_LOWMARGIN=3 };

static float Pmat[L2_DIM][BASE_DIM];
static float (*trigram)[CLASSES][CLASSES];
static float (*ent_table)[CLASSES];
static float (*margin_table)[CLASSES];
static float feat_mean[TOT_DIM], feat_std[TOT_DIM];
static float Wm[CLASSES][TOT_DIM], Bv[CLASSES];

static int   g_gate; static float g_alpha, g_surp_thr, g_ent_thr; static int g_ent_high;
static float g_l2_clamp=2.0f, g_nb_decay=1.0f, g_mix=0.5f, g_l2_scale=0.5f; static int g_cooldown=0;
static int   g_l3_mode=L3_NONE, g_l3_K=8, g_l3_M=4, g_l3_refr=16;
static float g_l3_mthr=0.0f, g_l3_alpha=0.9f;
static int   cd_ctr=0; static float prev_bound[BASE_DIM];
static int   g_clust=0, g_lmrun=0, g_lmrefr=0;
static int   g_l2write=0, g_l3_wrote=0;
// SEE config captured from the weight header (read by run_seed)
static float g_see_decay=0.75f, g_see_afast=0.5f; static int g_see_noja=0; static float g_see_woja[SEE_N_OJA_MAX*43];

static inline float dot_avx(const float* w, const float* f, int n) {
    __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(&w[i]),_mm256_loadu_ps(&f[i]),s);
    float o[8]; _mm256_storeu_ps(o,s);
    float r=o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; for(;i<n;i++) r+=w[i]*f[i]; return r;
}
static uint8_t sample_from(const float P[CLASSES], uint64_t* rng) {
    *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
    double u = (*rng >> 11) * (1.0 / (1ULL << 53));
    double c=0; for(int k=0;k<CLASSES;k++){ c+=P[k]; if(u<=c) return (uint8_t)k; }
    return CLASSES-1;
}
static void gen_projection(uint32_t seed) {
    uint64_t s = seed ? (uint64_t)seed : 0x9E3779B97F4A7C15ULL;
    for (int j=0;j<L2_DIM;j++) for (int k=0;k<BASE_DIM;k++){ s^=s<<13; s^=s>>7; s^=s<<17; Pmat[j][k]=(s&1ULL)?1.f:-1.f; }
}
static inline int is_punct(uint8_t b){ return b=='.'||b=='!'||b=='?'||b=='\n'||b=='"'||b=='\''; }
static inline int is_ws(uint8_t b){ return b==' '||b=='\n'||b=='\t'||b=='.'||b==','||b=='!'||b=='?'||b==';'||b==':'; }
static inline int eval_gate(uint8_t byte, uint8_t c1, uint8_t c2) {
    switch (g_gate) {
        case G_NONE: return 0; case G_PUNCT: return is_punct(byte); case G_WS: return is_ws(byte);
        case G_SURPRISE: return (-trigram[c2][c1][byte]) > g_surp_thr;
        case G_ENTROPY:  return g_ent_high ? (ent_table[c2][c1] > g_ent_thr) : (ent_table[c2][c1] < g_ent_thr);
        case G_COMBINED: return is_punct(byte) || ((-trigram[c2][c1][byte]) > g_surp_thr);
    }
    return 0;
}
// LOWMARGIN schedule firing (one step). Matches phase46a_l3.c / phase46_generator.c.
static inline int lm_fire(double trimargin, double m_thr, int M, int refr, int* lmrun, int* lmrefr) {
    if (trimargin < m_thr) (*lmrun)++; else *lmrun=0;
    int fire=0; if (*lmrun==M && *lmrefr<=0){ fire=1; *lmrefr=refr; }
    if (*lmrefr>0) (*lmrefr)--; return fire;
}
static inline int l3_schedule_fire(int mode, uint8_t byte, int l2write, double trimargin,
                                   int K, double m_thr, int M, int refr, int* clust, int* lmrun, int* lmrefr) {
    switch (mode) {
        case L3_NONE: return 0;
        case L3_PUNCT: return (byte=='.'||byte=='!'||byte=='?')?1:0;
        case L3_L2CLUST: if(l2write){ (*clust)++; if(*clust>=K){ *clust=0; return 1; } } return 0;
        case L3_LOWMARGIN: return lm_fire(trimargin,m_thr,M,refr,lmrun,lmrefr);
    }
    return 0;
}
static inline void l2_evolve(float* L2, const float* fa, int gate) {
    int upd = gate && (cd_ctr==0); g_l2write=upd;
    if (upd) {
        const float* src=fa; static float blend[BASE_DIM];
        if (g_mix>0.0f){ for(int k=0;k<BASE_DIM;k++) blend[k]=fa[k]-g_mix*prev_bound[k]; src=blend; memcpy(prev_bound,fa,BASE_DIM*sizeof(float)); }
        float scale=1.0f/sqrtf((float)BASE_DIM); float w[L2_DIM];
        for (int j=0;j<L2_DIM;j++){ float p=0; const float* pj=Pmat[j]; for(int k=0;k<BASE_DIM;k++) p+=pj[k]*src[k]; w[j]=p*scale; }
        for (int j=0;j<L2_DIM;j++) L2[j]=g_alpha*L2[j]+(1.0f-g_alpha)*w[j];
        cd_ctr=g_cooldown;
    } else { if(!gate && g_nb_decay<1.0f) for(int j=0;j<L2_DIM;j++) L2[j]*=g_nb_decay; if(cd_ctr>0)cd_ctr--; }
}
static inline void l3_evolve(float* L3, const float* fa, uint8_t byte, int l2write, double trimargin) {
    g_l3_wrote=0;
    if (l3_schedule_fire(g_l3_mode,byte,l2write,trimargin,g_l3_K,(double)g_l3_mthr,g_l3_M,g_l3_refr,&g_clust,&g_lmrun,&g_lmrefr)) {
        for (int j=0;j<L3_DIM;j++) L3[j]=g_l3_alpha*L3[j]+(1.0f-g_l3_alpha)*fa[j];
        g_l3_wrote=1;
    }
}

static int cmp_f(const void* a, const void* b){ float x=*(const float*)a,y=*(const float*)b; return (x<y)?-1:((x>y)?1:0); }
static float q_of(float* a, long n, double q){ if(n<1) return 0; long i=(long)(q*(n-1)); if(i<0)i=0; if(i>=n)i=n-1; return a[i]; }

static uint8_t* g_data; static long g_fsz;

// One closed-loop seed: warmup(online stats) + seed + generate. Records per gen step the
// trigram-margin and whether L3 fired. Returns L3 update count; sets *self_bpb.
static int run_seed(int seed_start, int warmup, int gen_len, float temp, uint64_t rng_seed,
                    float* margin_out, uint8_t* gen_out, int* fired_out, double* self_bpb) {
    float feat192[BASE_DIM], fa[BASE_DIM], raw[TOT_DIM];
    float L2[L2_DIM]; memset(L2,0,sizeof(L2));
    float L3[L3_DIM]; memset(L3,0,sizeof(L3));
    memset(prev_bound,0,sizeof(prev_bound)); cd_ctr=0; g_clust=0; g_lmrun=0; g_lmrefr=0;
    uint8_t ctx1=0,ctx2=0;
    SiliconEntropyState see;
    see_init(&see,42,4,g_see_decay);
    see.multiscale_mode=1; see.alpha_fast=g_see_afast; see.alpha_mid=0.9f; see.alpha_slow=0.99f;
    see.n_oja=g_see_noja; memcpy(see.W_oja,g_see_woja,(size_t)g_see_noja*43*sizeof(float));
    see.eta_oja=0.0f; see.plastic_blend=1.0f;

    double omean[TOT_DIM],oM2[TOT_DIM]; memset(omean,0,sizeof(omean)); memset(oM2,0,sizeof(oM2));
    long on=0; int cold=64; see_reset(&see);
    for (int i=seed_start;i<seed_start+warmup;i++){ uint8_t b=g_data[i];
        double tm=margin_table[ctx2][ctx1]; int gate=eval_gate(b,ctx1,ctx2);
        see_observe(&see,b); see_extract(&see,fa); l2_evolve(L2,fa,gate); l3_evolve(L3,fa,b,g_l2write,tm);
        ctx2=ctx1; ctx1=b;
        if (i-seed_start>=cold){ memcpy(raw,fa,BASE_DIM*4); memcpy(raw+BASE_DIM,L2,L2_DIM*4); memcpy(raw+BASE_DIM+L2_DIM,L3,L3_DIM*4);
            on++; for(int f=0;f<TOT_DIM;f++){ double d=raw[f]-omean[f]; omean[f]+=d/on; oM2[f]+=d*(raw[f]-omean[f]); } }
    }
    for(int f=0;f<TOT_DIM;f++){ feat_mean[f]=(float)omean[f]; feat_std[f]=(on>1)?(float)sqrt(oM2[f]/(on-1))+1e-8f:1.0f; }
    int soff=seed_start+warmup; int slen=(int)(g_fsz-soff); if(slen>512)slen=512;
    for(int i=0;i<slen;i++){ uint8_t b=g_data[soff+i];
        double tm=margin_table[ctx2][ctx1]; int gate=eval_gate(b,ctx1,ctx2);
        see_observe(&see,b); see_extract(&see,fa); l2_evolve(L2,fa,gate); l3_evolve(L3,fa,b,g_l2write,tm); ctx2=ctx1; ctx1=b; }

    uint64_t rng = rng_seed ^ ((uint64_t)seed_start*0x9E3779B97F4A7C15ULL); if(rng==0) rng=0x9E3779B97F4A7C15ULL;
    double bits=0; int updates=0; float Pp[CLASSES];
    for (int i=0;i<gen_len;i++){
        see_extract(&see,feat192);
        memcpy(raw,feat192,BASE_DIM*4); memcpy(raw+BASE_DIM,L2,L2_DIM*4); memcpy(raw+BASE_DIM+L2_DIM,L3,L3_DIM*4);
        for(int f=0;f<TOT_DIM;f++){ float x=(raw[f]-feat_mean[f])/(feat_std[f]+1e-8f);
            float cl=(f<BASE_DIM)?2.0f:g_l2_clamp; if(cl>0){ if(x>cl)x=cl; if(x<-cl)x=-cl; } if(f>=BASE_DIM)x*=g_l2_scale; raw[f]=x; }
        const float* tri=&trigram[ctx2][ctx1][0];
        float lg[CLASSES],mx=-1e30f;
        for(int c=0;c<CLASSES;c++){ lg[c]=(tri[c]+Bv[c]+dot_avx(Wm[c],raw,TOT_DIM))/temp; if(lg[c]>mx)mx=lg[c]; }
        float Z=0; for(int c=0;c<CLASSES;c++){ Pp[c]=expf(lg[c]-mx); Z+=Pp[c]; } for(int c=0;c<CLASSES;c++) Pp[c]/=Z;
        uint8_t next=sample_from(Pp,&rng);
        bits += -log2((double)fmaxf(Pp[next],1e-30f));
        margin_out[i]=margin_table[ctx2][ctx1]; gen_out[i]=next;
        double tm=margin_table[ctx2][ctx1];
        see_observe(&see,next); see_extract(&see,fa);
        int gate=eval_gate(next,ctx1,ctx2); l2_evolve(L2,fa,gate); l3_evolve(L3,fa,next,g_l2write,tm);
        fired_out[i]=g_l3_wrote; if(g_l3_wrote) updates++;
        ctx2=ctx1; ctx1=next;
    }
    *self_bpb = bits/gen_len;
    return updates;
}

int main(int argc, char** argv) {
    if (argc<3){ fprintf(stderr,"Usage: %s <data> <weights_0x53454544> [--seeds N --gen-len N --warmup N --temp F]\n",argv[0]); return 1; }
    int nseeds=8, gen_len=2000, warmup=5000; float temp=0.55f; uint64_t rng_seed=12345;
    for (int i=3;i<argc;i++){
        if(!strcmp(argv[i],"--seeds")&&i+1<argc) nseeds=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--gen-len")&&i+1<argc) gen_len=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--warmup")&&i+1<argc) warmup=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--temp")&&i+1<argc) temp=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"--rng-seed")&&i+1<argc) rng_seed=strtoull(argv[++i],NULL,0);
    }
    FILE* fd=fopen(argv[1],"rb"); if(!fd){fprintf(stderr,"Cannot open %s\n",argv[1]);return 1;}
    fseek(fd,0,SEEK_END); g_fsz=ftell(fd); fseek(fd,0,SEEK_SET);
    g_data=malloc(g_fsz); fread(g_data,1,g_fsz,fd); fclose(fd);

    FILE* fw=fopen(argv[2],"rb"); if(!fw){fprintf(stderr,"Cannot open weights %s\n",argv[2]);return 1;}
    uint32_t magic; fread(&magic,4,1,fw); rewind(fw);
    if (magic!=0x53454544){ fprintf(stderr,"Expected 0x53454544, got 0x%08x\n",magic); return 1; }
    uint32_t hdr4[4]; fread(hdr4,4,4,fw); float hf[5]; fread(hf,4,5,fw);
    g_see_decay=hf[0]; g_see_afast=hf[2];
    uint32_t no=0; fread(&no,4,1,fw); g_see_noja=(int)no; fread(g_see_woja,sizeof(float),(size_t)g_see_noja*43,fw);
    uint32_t l2d=0,gt=0,eh=0,ps=0; float al=0,st=0,et=0;
    fread(&l2d,4,1,fw); fread(&gt,4,1,fw); fread(&al,4,1,fw); fread(&st,4,1,fw); fread(&et,4,1,fw); fread(&eh,4,1,fw); fread(&ps,4,1,fw);
    g_gate=(int)gt; g_alpha=al; g_surp_thr=st; g_ent_thr=et; g_ent_high=(int)eh;
    float l2c=0,nbd=1; uint32_t cd=0,dl=0; fread(&l2c,4,1,fw); fread(&nbd,4,1,fw); fread(&cd,4,1,fw); fread(&dl,4,1,fw);
    g_l2_clamp=(l2c>0)?l2c:2.0f; g_nb_decay=nbd; g_cooldown=(int)cd;
    float mx=0.5f; fread(&mx,4,1,fw); g_mix=mx; float ls=0.5f; fread(&ls,4,1,fw); g_l2_scale=(ls>0)?ls:1.0f;
    float l2cap=0; fread(&l2cap,4,1,fw);
    uint32_t l3d=0,l3m=0,l3K=8,l3M=4,l3R=16; float l3mt=0,l3a=0.9f;
    fread(&l3d,4,1,fw); fread(&l3m,4,1,fw); fread(&l3K,4,1,fw); fread(&l3mt,4,1,fw); fread(&l3M,4,1,fw); fread(&l3R,4,1,fw); fread(&l3a,4,1,fw);
    g_l3_mode=(int)l3m; g_l3_K=(int)l3K; g_l3_mthr=l3mt; g_l3_M=(int)l3M; g_l3_refr=(int)l3R; g_l3_alpha=l3a;
    size_t tn=(size_t)CLASSES*CLASSES*CLASSES; trigram=malloc(tn*sizeof(float)); fread(trigram,sizeof(float),tn,fw);
    fread(feat_mean,sizeof(float),TOT_DIM,fw); fread(feat_std,sizeof(float),TOT_DIM,fw);
    fread(Wm,sizeof(float),CLASSES*TOT_DIM,fw); fread(Bv,sizeof(float),CLASSES,fw); fclose(fw);

    gen_projection(ps);
    ent_table=malloc(CLASSES*CLASSES*sizeof(float)); margin_table=malloc(CLASSES*CLASSES*sizeof(float));
    for(int i=0;i<CLASSES;i++) for(int j=0;j<CLASSES;j++){
        float m=-1e9f; for(int k=0;k<CLASSES;k++) if(trigram[i][j][k]>m)m=trigram[i][j][k];
        double se=0; for(int k=0;k<CLASSES;k++) se+=exp((double)(trigram[i][j][k]-m));
        double H=0,p1=-1,p2=-1; for(int k=0;k<CLASSES;k++){ double p=exp((double)(trigram[i][j][k]-m))/se; if(p>1e-12)H-=p*log(p); if(p>p1){p2=p1;p1=p;} else if(p>p2)p2=p; }
        ent_table[i][j]=(float)H; margin_table[i][j]=(float)(p1-p2);
    }
    const char* l3n[]={"none","punct","l2clust","lowmargin"};
    fprintf(stderr,"46.B0: weight=%s L3=%s header_mthr(p20_tf)=%.4f M=%d refr=%d | seeds=%d gen=%d warmup=%d temp=%.2f\n",
            argv[2],l3n[g_l3_mode],g_l3_mthr,g_l3_M,g_l3_refr,nseeds,gen_len,warmup,temp);

    // ---- teacher-forced margin distribution (context-only over val) ----
    long vstart=g_fsz/2, vlen=2000000; if(vstart+vlen+2>g_fsz) vlen=g_fsz-vstart-2;
    float* mtf=malloc(vlen*sizeof(float));
    for(long i=0;i<vlen;i++){ uint8_t c2=g_data[vstart+i], c1=g_data[vstart+i+1]; mtf[i]=margin_table[c2][c1]; }
    qsort(mtf,vlen,sizeof(float),cmp_f);

    // ---- N closed-loop seeds ----
    long fszm=g_fsz-(warmup+1024);
    float* mcl=malloc((long)nseeds*gen_len*sizeof(float));
    float** seedm=malloc(nseeds*sizeof(float*)); uint8_t** seedg=malloc(nseeds*sizeof(uint8_t*)); int** seedf=malloc(nseeds*sizeof(int*));
    int* upc=malloc(nseeds*sizeof(int)); double* sbpb=malloc(nseeds*sizeof(double));
    long mcl_n=0;
    fprintf(stderr,"per-seed: ");
    for(int s=0;s<nseeds;s++){
        int seed_start=(int)((long)s*fszm/nseeds);
        seedm[s]=malloc(gen_len*sizeof(float)); seedg[s]=malloc(gen_len); seedf[s]=malloc(gen_len*sizeof(int));
        upc[s]=run_seed(seed_start,warmup,gen_len,temp,rng_seed,seedm[s],seedg[s],seedf[s],&sbpb[s]);
        for(int i=0;i<gen_len;i++) mcl[mcl_n++]=seedm[s][i];
        fprintf(stderr,"s%d(up=%d bpb=%.3f) ",s,upc[s],sbpb[s]);
    }
    fprintf(stderr,"\n");

    float* mcls=malloc(mcl_n*sizeof(float)); memcpy(mcls,mcl,mcl_n*sizeof(float)); qsort(mcls,mcl_n,sizeof(float),cmp_f);

    // ---- update counts + gaps ----
    long tot_up=0; for(int s=0;s<nseeds;s++) tot_up+=upc[s];
    float* gaps=malloc(mcl_n*sizeof(float)); long ng=0;
    for(int s=0;s<nseeds;s++){ int last=-1; for(int i=0;i<gen_len;i++) if(seedf[s][i]){ if(last>=0) gaps[ng++]=(float)(i-last); last=i; } }
    qsort(gaps,ng,sizeof(float),cmp_f);

    printf("\n==== 46.B0 LOWMARGIN firing diagnostic ====\n");
    printf("L3 updates: total=%ld over %d seeds x %d steps = %.4f%% (per-seed mean %.1f)\n",
           tot_up,nseeds,gen_len,100.0*tot_up/((long)nseeds*gen_len),(double)tot_up/nseeds);
    printf("real gaps between firings (n=%ld): p50=%.0f p90=%.0f  (teacher-forced 46.0 expected p50~130)\n",
           ng, q_of(gaps,ng,0.5), q_of(gaps,ng,0.9));

    printf("\ntrigram-margin distribution (lower = less confident = should fire LOWMARGIN):\n");
    printf("  %-14s %8s %8s %8s %8s %8s\n","quantile","p10","p20","p30","p40","p50");
    printf("  %-14s %8.4f %8.4f %8.4f %8.4f %8.4f\n","teacher-forced",q_of(mtf,vlen,.1),q_of(mtf,vlen,.2),q_of(mtf,vlen,.3),q_of(mtf,vlen,.4),q_of(mtf,vlen,.5));
    printf("  %-14s %8.4f %8.4f %8.4f %8.4f %8.4f\n","closed-loop",q_of(mcls,mcl_n,.1),q_of(mcls,mcl_n,.2),q_of(mcls,mcl_n,.3),q_of(mcls,mcl_n,.4),q_of(mcls,mcl_n,.5));
    double frac_below=0; for(long i=0;i<mcl_n;i++) if(mcls[i]<g_l3_mthr) frac_below+=1; frac_below/=mcl_n;
    printf("  header m_thr=%.4f sits at closed-loop fraction-below = %.2f%% (if << 20%%, p20_tf is too low for closed-loop)\n",
           g_l3_mthr, 100.0*frac_below);

    // ---- inference-only re-threshold on the recorded closed-loop margin sequences ----
    double thrs[4]={ g_l3_mthr, q_of(mcls,mcl_n,.3), q_of(mcls,mcl_n,.4), q_of(mcls,mcl_n,.5) };
    const char* tn2[4]={"p20_tf(header)","p30_cl","p40_cl","p50_cl"};
    printf("\ninference-only re-threshold (re-run LOWMARGIN M=%d refr=%d on recorded margins, no regen):\n",g_l3_M,g_l3_refr);
    printf("  %-16s %10s %10s\n","threshold","value","fires%");
    for(int t=0;t<4;t++){ long fires=0;
        for(int s=0;s<nseeds;s++){ int lmrun=0,lmrefr=0;
            for(int i=0;i<gen_len;i++) if(lm_fire(seedm[s][i],thrs[t],g_l3_M,g_l3_refr,&lmrun,&lmrefr)) fires++; }
        printf("  %-16s %10.4f %10.4f\n",tn2[t],thrs[t],100.0*fires/((long)nseeds*gen_len));
    }

    // ---- first 10 L3 events of the loopiest seed (lowest self-BPB) ----
    int loopy=0; for(int s=1;s<nseeds;s++) if(sbpb[s]<sbpb[loopy]) loopy=s;
    printf("\nfirst 10 L3 events of loopiest seed s%d (self-BPB %.3f), '|' marks the firing byte:\n",loopy,sbpb[loopy]);
    int shown=0;
    for(int i=0;i<gen_len && shown<10;i++) if(seedf[loopy][i]){ shown++;
        printf("  ev%-2d @step %4d margin=%.4f  [",shown,i,seedm[loopy][i]);
        for(int t=-12;t<=2;t++){ int q=i+t; if(q<0||q>=gen_len) continue; uint8_t b=seedg[loopy][q];
            if(t==0) printf("|"); if(b=='\n')printf("\\n"); else if(isprint(b))putchar(b); else printf("\\x%02x",b); }
        printf("]\n");
    }
    if(shown==0) printf("  (no L3 events in the loopiest seed - LOWMARGIN never fired here)\n");

    free(g_data); free(trigram); free(ent_table); free(margin_table); free(mtf); free(mcl); free(mcls); free(gaps);
    return 0;
}
