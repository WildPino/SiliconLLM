// Phase 46.A generator - closed-loop [SEE 192 | L2 64 | L3 64] = 320D over D1 + L3 phrase
// memory. Reads the 0x53454544 weight (D1 core + L3 block). Maintains:
//   L2 = D1 boundary memory (gated delta-mix EMA, mix0.5, alpha0.99, entropy-high gate),
//   L3 = slow EMA of raw fa (SEE), refreshed only at the phrase schedule (PUNCT / L2CLUST /
//        LOWMARGIN). The L2 + L3 update blocks are byte-identical to phase46a_l3.c (parity).
// Linear readout, no L2 cap / geometry / write-gate (those were the 45.A-C dead ends).
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase38-42/phase46_generator.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase46_generator.exe -lm -I .
// Run:
//   bin/phase46_generator.exe <data> <weights_0x53454544> [--gen-len N --temp F --warmup N
//                              --seed-start N --rng-seed N --mode argmax --telemetry f.tsv]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include "src/silicon_entropy.h"

#define CLASSES   256
#define BASE_DIM  SEE_FEATURE_DIM
#define L2_DIM    64
#define L3_DIM    64
#define TOT_DIM   (BASE_DIM + L2_DIM + L3_DIM)   // 320

enum { G_NONE=0, G_PUNCT=1, G_WS=2, G_SURPRISE=3, G_ENTROPY=4, G_COMBINED=5 };
enum { L3_NONE=0, L3_PUNCT=1, L3_L2CLUST=2, L3_LOWMARGIN=3, L3_PUNCTLM=4 };

static float Pmat[L2_DIM][BASE_DIM];
static float (*trigram)[CLASSES][CLASSES];
static float (*ent_table)[CLASSES];
static float (*margin_table)[CLASSES];
static float feat_mean[TOT_DIM], feat_std[TOT_DIM];
static float Wm[CLASSES][TOT_DIM], Bv[CLASSES];

static int   g_gate; static float g_alpha, g_surp_thr, g_ent_thr; static int g_ent_high;
static float g_l2_clamp=2.0f, g_nb_decay=1.0f, g_mix=0.5f, g_l2_scale=0.5f; static int g_cooldown=0;
// L3 phrase schedule (from header)
static int   g_l3_mode=L3_NONE, g_l3_K=8, g_l3_M=4, g_l3_refr=16;
static float g_l3_mthr=0.0f, g_l3_alpha=0.9f;
// L2/L3 evolve state (reset once pre-warmup)
static int   cd_ctr=0; static float prev_bound[BASE_DIM];
static int   g_clust=0, g_lmrun=0, g_lmrefr=0;
// telemetry
static int   g_l2write=0, g_l3_wrote=0; static double g_l3_relmove=0.0;

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
        case G_NONE:   return 0;
        case G_PUNCT:  return is_punct(byte);
        case G_WS:     return is_ws(byte);
        case G_SURPRISE: return (-trigram[c2][c1][byte]) > g_surp_thr;
        case G_ENTROPY:  return g_ent_high ? (ent_table[c2][c1] > g_ent_thr) : (ent_table[c2][c1] < g_ent_thr);
        case G_COMBINED: return is_punct(byte) || ((-trigram[c2][c1][byte]) > g_surp_thr);
    }
    return 0;
}
// LOWMARGIN sub-schedule: fire on a margin<thr run of length M (refractory refr). Advances
// the lm state every call (must be invoked once per step, also inside PUNCTLM).
static inline int lm_fire(double trimargin, double m_thr, int M, int refr, int* lmrun, int* lmrefr) {
    if (trimargin < m_thr) (*lmrun)++; else *lmrun=0;
    int fire=0; if (*lmrun==M && *lmrefr<=0){ fire=1; *lmrefr=refr; }
    if (*lmrefr>0) (*lmrefr)--;
    return fire;
}
// L3 schedule - BYTE-IDENTICAL to phase46b_l3.c l3_schedule_fire (parity-critical).
static inline int l3_schedule_fire(int mode, uint8_t byte, int l2write, double trimargin,
                                   int K, double m_thr, int M, int refr,
                                   int* clust, int* lmrun, int* lmrefr) {
    switch (mode) {
        case L3_NONE:    return 0;
        case L3_PUNCT:   return (byte=='.'||byte=='!'||byte=='?') ? 1 : 0;
        case L3_L2CLUST: if (l2write){ (*clust)++; if (*clust>=K){ *clust=0; return 1; } } return 0;
        case L3_LOWMARGIN: return lm_fire(trimargin,m_thr,M,refr,lmrun,lmrefr);
        case L3_PUNCTLM: { int p=(byte=='.'||byte=='!'||byte=='?')?1:0;
                           int l=lm_fire(trimargin,m_thr,M,refr,lmrun,lmrefr);   // always advance lm state
                           return (p||l)?1:0; }
    }
    return 0;
}
// D1 L2 evolve: gated delta-mix EMA (no cap / geometry / write-gate). Sets g_l2write.
static inline void l2_evolve(float* L2, const float* fa, int gate) {
    int upd = gate && (cd_ctr==0);
    g_l2write = upd;
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
// L3 phrase-memory: refresh the slow EMA of raw fa at schedule events. Called EVERY step
// (the schedule state machine must advance every step). Sets g_l3_wrote / g_l3_relmove.
static inline void l3_evolve(float* L3, const float* fa, uint8_t byte, int l2write, double trimargin) {
    g_l3_wrote=0; g_l3_relmove=0.0;
    if (l3_schedule_fire(g_l3_mode, byte, l2write, trimargin, g_l3_K, (double)g_l3_mthr, g_l3_M, g_l3_refr, &g_clust,&g_lmrun,&g_lmrefr)) {
        double o2=0.0, d2=0.0;
        for (int j=0;j<L3_DIM;j++){ float nl=g_l3_alpha*L3[j]+(1.0f-g_l3_alpha)*fa[j]; double dd=(double)nl-L3[j]; d2+=dd*dd; o2+=(double)L3[j]*L3[j]; L3[j]=nl; }
        g_l3_wrote=1; double on=sqrt(o2); g_l3_relmove=sqrt(d2)/(on>1e-6?on:1e-6);
    }
}

int main(int argc, char** argv) {
    if (argc<3){ fprintf(stderr,"Usage: %s <data> <weights_0x53454544> [opts]\n",argv[0]); return 1; }
    int gen_len=2000, warmup=5000, seed_start=0, argmax=0, rng_time=0;
    float temp=0.65f; uint64_t rng_seed=0x243F6A8885A308D3ULL; FILE* tel=NULL;
    for (int i=3;i<argc;i++){
        if      (!strcmp(argv[i],"--gen-len")   && i+1<argc) gen_len=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--temp")      && i+1<argc) temp=(float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--warmup")    && i+1<argc) warmup=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--seed-start")&& i+1<argc) seed_start=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--rng-seed")  && i+1<argc){ const char* v=argv[++i]; if(!strcmp(v,"-1")) rng_time=1; else rng_seed=strtoull(v,NULL,0); }
        else if (!strcmp(argv[i],"--mode")      && i+1<argc) argmax=!strcmp(argv[++i],"argmax");
        else if (!strcmp(argv[i],"--telemetry") && i+1<argc) tel=fopen(argv[++i],"w");
    }

    FILE* fd=fopen(argv[1],"rb"); if(!fd){fprintf(stderr,"Cannot open %s\n",argv[1]);return 1;}
    fseek(fd,0,SEEK_END); long fsz=ftell(fd); fseek(fd,0,SEEK_SET);
    long need=(long)seed_start+warmup+512; if(need>fsz){need=fsz; warmup=(int)(fsz-seed_start-512); if(warmup<0)warmup=0;}
    uint8_t* fdat=malloc(need); fread(fdat,1,need,fd); fclose(fd);

    FILE* fw=fopen(argv[2],"rb"); if(!fw){fprintf(stderr,"Cannot open weights %s\n",argv[2]);return 1;}
    uint32_t magic; fread(&magic,4,1,fw); rewind(fw);
    if (magic!=0x53454544){ fprintf(stderr,"Expected 0x53454544 (46.A L3), got 0x%08x\n",magic); return 1; }
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
    // D1 core
    float l2c=0,nbd=1.0f; uint32_t cd=0,dl=0;
    fread(&l2c,4,1,fw); fread(&nbd,4,1,fw); fread(&cd,4,1,fw); fread(&dl,4,1,fw);
    g_l2_clamp=(l2c>0.0f)?l2c:feat_clamp; g_nb_decay=nbd; g_cooldown=(int)cd;
    float mx=0.5f; fread(&mx,4,1,fw); g_mix=mx;
    float ls=0.5f; fread(&ls,4,1,fw); g_l2_scale=(ls>0.0f)?ls:1.0f;
    float l2cap=0; fread(&l2cap,4,1,fw);   // D1 has no L2 cap (ignored)
    // L3 block
    uint32_t l3d=0,l3m=0,l3K=8,l3M=4,l3R=16; float l3mt=0.0f, l3a=0.9f;
    fread(&l3d,4,1,fw); fread(&l3m,4,1,fw); fread(&l3K,4,1,fw); fread(&l3mt,4,1,fw); fread(&l3M,4,1,fw); fread(&l3R,4,1,fw); fread(&l3a,4,1,fw);
    if ((int)l3d!=L3_DIM){ fprintf(stderr,"L3_DIM mismatch %u\n",l3d); return 1; }
    g_l3_mode=(int)l3m; g_l3_K=(int)l3K; g_l3_mthr=l3mt; g_l3_M=(int)l3M; g_l3_refr=(int)l3R; g_l3_alpha=l3a;
    size_t tri_n=(size_t)CLASSES*CLASSES*CLASSES;
    trigram=malloc(tri_n*sizeof(float));
    fread(trigram,sizeof(float),tri_n,fw);
    fread(feat_mean,sizeof(float),TOT_DIM,fw);   // read past (overwritten by online warmup stats)
    fread(feat_std, sizeof(float),TOT_DIM,fw);
    fread(Wm,sizeof(float),CLASSES*TOT_DIM,fw);
    fread(Bv,sizeof(float),CLASSES,fw);
    fclose(fw);

    gen_projection(ps);
    ent_table=malloc(CLASSES*CLASSES*sizeof(float));
    margin_table=malloc(CLASSES*CLASSES*sizeof(float));
    for(int i=0;i<CLASSES;i++) for(int j=0;j<CLASSES;j++){
        float m=-1e9f; for(int k=0;k<CLASSES;k++) if(trigram[i][j][k]>m) m=trigram[i][j][k];
        double se=0; for(int k=0;k<CLASSES;k++) se+=exp((double)(trigram[i][j][k]-m));
        double H=0,p1=-1,p2=-1; for(int k=0;k<CLASSES;k++){ double p=exp((double)(trigram[i][j][k]-m))/se;
            if(p>1e-12) H-=p*log(p); if(p>p1){p2=p1;p1=p;} else if(p>p2)p2=p; }
        ent_table[i][j]=(float)H; margin_table[i][j]=(float)(p1-p2);
    }

    SiliconEntropyState see;
    see_init(&see, 42, 4, decay);
    see.multiscale_mode=1; see.alpha_fast=alpha_fast; see.alpha_mid=0.9f; see.alpha_slow=0.99f;
    see.n_oja=n_oja; memcpy(see.W_oja,W_oja_buf,(size_t)n_oja*43*sizeof(float));
    see.eta_oja=0.0f; see.plastic_blend=1.0f;
    const char* gn[]={"none","punct","whitespace","surprise","entropy","combined"};
    const char* l3n[]={"none","punct","l2clust","lowmargin","punct+lm"};
    fprintf(stderr,"46 gen: magic=0x%08x gate=%s alpha=%.2f mix=%.2f scale=%.2f l2_clamp=%.2f | L3=%s K=%d mthr=%.4f M=%d refr=%d a3=%.2f\n",
            magic,gn[g_gate],g_alpha,g_mix,g_l2_scale,g_l2_clamp,(g_l3_mode>=0&&g_l3_mode<=4)?l3n[g_l3_mode]:"?",g_l3_K,g_l3_mthr,g_l3_M,g_l3_refr,g_l3_alpha);

    float feat192[BASE_DIM], fa[BASE_DIM], raw[TOT_DIM];
    float L2[L2_DIM]; memset(L2,0,sizeof(L2));
    float L3[L3_DIM]; memset(L3,0,sizeof(L3));
    memset(prev_bound,0,sizeof(prev_bound)); cd_ctr=0; g_clust=0; g_lmrun=0; g_lmrefr=0;
    uint8_t ctx1=0, ctx2=0;

    // ---- warmup: online 320D stats over the closed-loop feature distribution ----
    double omean[TOT_DIM], oM2[TOT_DIM]; memset(omean,0,sizeof(omean)); memset(oM2,0,sizeof(oM2));
    long on=0; int cold=64;
    see_reset(&see);
    for (int i=seed_start;i<seed_start+warmup;i++){
        uint8_t b=fdat[i];
        double trimargin=margin_table[ctx2][ctx1];
        int gate=eval_gate(b,ctx1,ctx2);
        see_observe(&see,b); see_extract(&see,fa);
        l2_evolve(L2,fa,gate);
        l3_evolve(L3,fa,b,g_l2write,trimargin);
        ctx2=ctx1; ctx1=b;
        if (i-seed_start>=cold){
            memcpy(raw,fa,BASE_DIM*sizeof(float)); memcpy(raw+BASE_DIM,L2,L2_DIM*sizeof(float)); memcpy(raw+BASE_DIM+L2_DIM,L3,L3_DIM*sizeof(float));
            on++; for(int f=0;f<TOT_DIM;f++){ double d=raw[f]-omean[f]; omean[f]+=d/on; oM2[f]+=d*(raw[f]-omean[f]); }
        }
    }
    for(int f=0;f<TOT_DIM;f++){ feat_mean[f]=(float)omean[f]; feat_std[f]=(on>1)?(float)sqrt(oM2[f]/(on-1))+1e-8f:1.0f; }
    fprintf(stderr,"Online stats from %ld samples\n",on);

    // ---- seed: condition SEE+L2+L3 on 512 bytes ----
    int soff=seed_start+warmup; int slen=(int)(need-soff); if(slen>512) slen=512;
    for (int i=0;i<slen;i++){ uint8_t b=fdat[soff+i];
        double trimargin=margin_table[ctx2][ctx1];
        int gate=eval_gate(b,ctx1,ctx2);
        see_observe(&see,b); see_extract(&see,fa);
        l2_evolve(L2,fa,gate); l3_evolve(L3,fa,b,g_l2write,trimargin);
        ctx2=ctx1; ctx1=b;
    }
    fprintf(stderr,"--- generated ---\n");

    uint64_t rng = rng_time ? ((uint64_t)time(NULL)^0xdeadbeefcafeULL^(uint64_t)seed_start)
                            : (rng_seed ^ ((uint64_t)seed_start*0x9E3779B97F4A7C15ULL));
    if (rng==0) rng=0x9E3779B97F4A7C15ULL;

    if (tel) fprintf(tel,"step\tbyte\tbpb\tl3ratio\tl3flip\tl3wrote\tl3relmove\tgate\n");

    double self_bits=0; uint8_t* gen=malloc(gen_len); float Pp[CLASSES];
    for (int i=0;i<gen_len;i++){
        see_extract(&see, feat192);
        memcpy(raw,feat192,BASE_DIM*sizeof(float)); memcpy(raw+BASE_DIM,L2,L2_DIM*sizeof(float)); memcpy(raw+BASE_DIM+L2_DIM,L3,L3_DIM*sizeof(float));
        for (int f=0;f<TOT_DIM;f++){ float x=(raw[f]-feat_mean[f])/(feat_std[f]+1e-8f);
            float cl=(f<BASE_DIM)?feat_clamp:g_l2_clamp; if(cl>0){ if(x>cl)x=cl; if(x<-cl)x=-cl; }
            if (f>=BASE_DIM) x*=g_l2_scale; raw[f]=x; }
        const float* tri=&trigram[ctx2][ctx1][0];
        float logits[CLASSES], mx=-1e30f;
        for (int c=0;c<CLASSES;c++){ logits[c]=(tri[c]+Bv[c]+dot_avx(Wm[c],raw,TOT_DIM))/temp; if(logits[c]>mx)mx=logits[c]; }
        float Z=0; for(int c=0;c<CLASSES;c++){ Pp[c]=expf(logits[c]-mx); Z+=Pp[c]; } for(int c=0;c<CLASSES;c++) Pp[c]/=Z;
        uint8_t next;
        if (argmax){ float b=-1; next=0; for(int c=0;c<CLASSES;c++) if(Pp[c]>b){b=Pp[c];next=(uint8_t)c;} }
        else next=sample_from(Pp,&rng);
        self_bits += -log2((double)fmaxf(Pp[next],1e-30f));
        gen[i]=next;

        double trimargin=margin_table[ctx2][ctx1];

        if (tel){
            // L3 contribution: SEE+L2 logit vs L3 logit (isolates L3)
            float sl2[CLASSES], l3v[CLASSES]; double ms=0,m3=0;
            for(int c=0;c<CLASSES;c++){ sl2[c]=tri[c]+Bv[c]+dot_avx(Wm[c],raw,BASE_DIM+L2_DIM); l3v[c]=dot_avx(Wm[c]+BASE_DIM+L2_DIM,raw+BASE_DIM+L2_DIM,L3_DIM); ms+=sl2[c]; m3+=l3v[c]; }
            ms/=CLASSES; m3/=CLASSES; double vs=0,v3=0;
            for(int c=0;c<CLASSES;c++){ double a=sl2[c]-ms,b=l3v[c]-m3; vs+=a*a; v3+=b*b; }
            double rs=sqrt(vs/CLASSES), r3=sqrt(v3/CLASSES); double ratio=(rs>1e-9)?r3/rs:0.0;
            int as=0,af=0; float bs=-1e30f,bf=-1e30f;
            for(int c=0;c<CLASSES;c++){ if(sl2[c]>bs){bs=sl2[c];as=c;} float u=sl2[c]+l3v[c]; if(u>bf){bf=u;af=c;} }
            fprintf(tel,"%d\t%d\t%.4f\t%.4f\t%d\t%d\t%.4f\t%d\n", i,(int)next,self_bits/(i+1),ratio,(as!=af)?1:0,g_l3_wrote,g_l3_relmove,eval_gate(next,ctx1,ctx2));
        }

        see_observe(&see,next); see_extract(&see,fa);
        int gate=eval_gate(next,ctx1,ctx2);
        l2_evolve(L2,fa,gate);
        l3_evolve(L3,fa,next,g_l2write,trimargin);
        ctx2=ctx1; ctx1=next;
    }
    fwrite(gen,1,gen_len,stdout);
    fprintf(stderr,"\n--- stats ---\nself_BPB: %.4f\n", self_bits/gen_len);
    if (tel) fclose(tel);
    free(trigram); free(ent_table); free(margin_table); free(fdat); free(gen);
    return 0;
}
