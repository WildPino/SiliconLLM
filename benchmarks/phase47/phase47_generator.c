// Phase 47.A0 generator - closed-loop D1 substrate [SEE 192 | L2 64] + STATIC NONLINEAR
// readout (MLP probe, magic 0x53454545, saved by phase47a0_gauntlet). No volatile memory
// beyond D1's own L2: the only change vs the D1 generator is the readout lens.
//
// Parity with training (phase47_a0_gauntlet extract_window):
//   - L2 evolve: entropy-high gated delta-mix EMA, mix0.5 alpha0.99 (cooldown/decay from
//     header; trivial for D1/F0) - byte-identical update;
//   - normalization: D1's STORED mean/std (the stats the MLP was trained on), base clamp
//     2.0, mem clamp l2_clamp, mem scale l2_scale. NOT online warmup stats: a nonlinear
//     readout is sensitive to its input distribution. --online-stats for diagnosis only.
//   - logits = (trigram + mlp(nf))/temp.
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma benchmarks/phase38-42/phase47_generator.c \
//       src/silicon_entropy.c src/silicon_v0.c -o bin/phase47_generator.exe -lm -I .
// Run:
//   bin/phase47_generator.exe <data> <D1_w_0x53454540> <mlp_0x53454545> [--gen-len N --temp F
//       --warmup N --seed-start N --rng-seed N --mode argmax --telemetry f.tsv --online-stats]

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
#define TOT_DIM   (BASE_DIM + L2_DIM)   // 256

enum { G_NONE=0, G_PUNCT=1, G_WS=2, G_SURPRISE=3, G_ENTROPY=4, G_COMBINED=5 };

static float Pmat[L2_DIM][BASE_DIM];
static float (*trigram)[CLASSES][CLASSES];
static float (*ent_table)[CLASSES];
static float feat_mean[TOT_DIM], feat_std[TOT_DIM];

static int   g_gate; static float g_alpha, g_surp_thr, g_ent_thr; static int g_ent_high;
static float g_l2_clamp=2.0f, g_nb_decay=1.0f, g_mix=0.5f, g_l2_scale=0.5f; static int g_cooldown=0;
static int   cd_ctr=0; static float prev_bound[BASE_DIM]; static int g_l2write=0;

// MLP probe (0x53454545): H==0 means linear
static int g_H=0; static long g_off=0; static int g_dim=TOT_DIM;
static float *W1=NULL,*b1=NULL,*W2=NULL,*b2=NULL;

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
// D1 L2 evolve - same as phase44/46 generators; for D1/F0 (cooldown=0, nb_decay=1) this is
// byte-identical to the gauntlet's extract_window update.
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
// normalization - byte-identical to the gauntlet's norm_feats (base clamp 2.0 hardcoded)
static inline void norm_feats(const float* raw, float* out){
    for(int fi=0;fi<TOT_DIM;fi++){ float x=(raw[fi]-feat_mean[fi])/(feat_std[fi]+1e-8f);
        float cl=(fi<BASE_DIM)?2.0f:g_l2_clamp; if(cl>0){ if(x>cl)x=cl; if(x<-cl)x=-cl; }
        if(fi>=BASE_DIM) x*=g_l2_scale; out[fi]=x; }
}
static inline void mlp_logits(const float* nf, float* hid, float* lg){
    const float* x=nf+g_off;
    if(g_H>0){
        for(int j=0;j<g_H;j++){ float a=b1[j]+dot_avx(&W1[(size_t)j*g_dim],x,g_dim); hid[j]=a>0?a:0; }
        for(int c=0;c<CLASSES;c++) lg[c]=b2[c]+dot_avx(&W2[(size_t)c*g_H],hid,g_H);
    } else {
        for(int c=0;c<CLASSES;c++) lg[c]=b2[c]+dot_avx(&W2[(size_t)c*g_dim],x,g_dim);
    }
}

int main(int argc, char** argv) {
    if (argc<4){ fprintf(stderr,"Usage: %s <data> <D1_w> <mlp_w> [opts]\n",argv[0]); return 1; }
    int gen_len=2000, warmup=5000, seed_start=0, argmax=0, rng_time=0, online_stats=0;
    float temp=0.65f; uint64_t rng_seed=0x243F6A8885A308D3ULL; FILE* tel=NULL;
    for (int i=4;i<argc;i++){
        if      (!strcmp(argv[i],"--gen-len")   && i+1<argc) gen_len=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--temp")      && i+1<argc) temp=(float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--warmup")    && i+1<argc) warmup=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--seed-start")&& i+1<argc) seed_start=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--rng-seed")  && i+1<argc){ const char* v=argv[++i]; if(!strcmp(v,"-1")) rng_time=1; else rng_seed=strtoull(v,NULL,0); }
        else if (!strcmp(argv[i],"--mode")      && i+1<argc) argmax=!strcmp(argv[++i],"argmax");
        else if (!strcmp(argv[i],"--telemetry") && i+1<argc) tel=fopen(argv[++i],"w");
        else if (!strcmp(argv[i],"--online-stats")) online_stats=1;
    }

    FILE* fd=fopen(argv[1],"rb"); if(!fd){fprintf(stderr,"Cannot open %s\n",argv[1]);return 1;}
    fseek(fd,0,SEEK_END); long fsz=ftell(fd); fseek(fd,0,SEEK_SET);
    long need=(long)seed_start+warmup+512; if(need>fsz){need=fsz; warmup=(int)(fsz-seed_start-512); if(warmup<0)warmup=0;}
    uint8_t* fdat=malloc(need); fread(fdat,1,need,fd); fclose(fd);

    // ---- D1 substrate (0x53454540): SEE + gate + L2 params + trigram + stored stats ----
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
    float mx=0.5f; fread(&mx,4,1,fw); g_mix=mx;
    float ls=0.5f; fread(&ls,4,1,fw); g_l2_scale=(ls>0.0f)?ls:1.0f;
    float l2cap=0; fread(&l2cap,4,1,fw);
    size_t tri_n=(size_t)CLASSES*CLASSES*CLASSES;
    trigram=malloc(tri_n*sizeof(float)); fread(trigram,sizeof(float),tri_n,fw);
    fread(feat_mean,sizeof(float),TOT_DIM,fw);   // STORED training stats: the MLP's input frame
    fread(feat_std, sizeof(float),TOT_DIM,fw);
    fclose(fw);                                  // D1's own linear readout not needed
    if (feat_clamp!=2.0f) fprintf(stderr,"WARN: feat_clamp=%.2f but probe trained with base clamp 2.0\n",feat_clamp);

    // ---- MLP probe (0x53454545) ----
    FILE* fm=fopen(argv[3],"rb"); if(!fm){fprintf(stderr,"Cannot open mlp %s\n",argv[3]);return 1;}
    uint32_t mm=0,mh=0,mo=0,md=0; fread(&mm,4,1,fm); fread(&mh,4,1,fm); fread(&mo,4,1,fm); fread(&md,4,1,fm);
    if (mm!=0x53454545){ fprintf(stderr,"Expected MLP 0x53454545, got 0x%08x\n",mm); return 1; }
    g_H=(int)mh; g_off=(long)mo; g_dim=(int)md;
    if (g_off+g_dim>TOT_DIM){ fprintf(stderr,"probe slice off=%ld dim=%d > %d\n",g_off,g_dim,TOT_DIM); return 1; }
    if (g_H>0){ W1=malloc((size_t)g_H*g_dim*4); b1=malloc(g_H*4); W2=malloc((size_t)CLASSES*g_H*4);
        fread(W1,4,(size_t)g_H*g_dim,fm); fread(b1,4,g_H,fm); fread(W2,4,(size_t)CLASSES*g_H,fm); }
    else { W2=malloc((size_t)CLASSES*g_dim*4); fread(W2,4,(size_t)CLASSES*g_dim,fm); }
    b2=malloc(CLASSES*4); fread(b2,4,CLASSES,fm); fclose(fm);

    gen_projection(ps);
    ent_table=malloc(CLASSES*CLASSES*sizeof(float));
    for(int i=0;i<CLASSES;i++) for(int j=0;j<CLASSES;j++){
        float m=-1e9f; for(int k=0;k<CLASSES;k++) if(trigram[i][j][k]>m) m=trigram[i][j][k];
        double se=0; for(int k=0;k<CLASSES;k++) se+=exp((double)(trigram[i][j][k]-m));
        double H=0; for(int k=0;k<CLASSES;k++){ double p=exp((double)(trigram[i][j][k]-m))/se; if(p>1e-12) H-=p*log(p); }
        ent_table[i][j]=(float)H;
    }

    SiliconEntropyState see;
    see_init(&see, 42, 4, decay);
    see.multiscale_mode=1; see.alpha_fast=alpha_fast; see.alpha_mid=0.9f; see.alpha_slow=0.99f;
    see.n_oja=n_oja; memcpy(see.W_oja,W_oja_buf,(size_t)n_oja*43*sizeof(float));
    see.eta_oja=0.0f; see.plastic_blend=1.0f;
    fprintf(stderr,"47 gen: D1 substrate alpha=%.2f mix=%.2f scale=%.2f clamp=%.2f | MLP H=%d off=%ld dim=%d | stats=%s\n",
            g_alpha,g_mix,g_l2_scale,g_l2_clamp,g_H,g_off,g_dim,online_stats?"online":"stored");

    float fa[BASE_DIM], raw[TOT_DIM], nf[TOT_DIM];
    float L2[L2_DIM]; memset(L2,0,sizeof(L2));
    memset(prev_bound,0,sizeof(prev_bound)); cd_ctr=0;
    uint8_t ctx1=0, ctx2=0;

    // ---- warmup: condition SEE+L2 (and optionally re-derive stats online) ----
    double omean[TOT_DIM], oM2[TOT_DIM]; memset(omean,0,sizeof(omean)); memset(oM2,0,sizeof(oM2));
    long on=0; int cold=64;
    see_reset(&see);
    for (int i=seed_start;i<seed_start+warmup;i++){
        uint8_t b=fdat[i];
        int gate=eval_gate(b,ctx1,ctx2);
        see_observe(&see,b); see_extract(&see,fa);
        l2_evolve(L2,fa,gate);
        ctx2=ctx1; ctx1=b;
        if (online_stats && i-seed_start>=cold){
            memcpy(raw,fa,BASE_DIM*sizeof(float)); memcpy(raw+BASE_DIM,L2,L2_DIM*sizeof(float));
            on++; for(int f=0;f<TOT_DIM;f++){ double d=raw[f]-omean[f]; omean[f]+=d/on; oM2[f]+=d*(raw[f]-omean[f]); }
        }
    }
    if (online_stats && on>1){ for(int f=0;f<TOT_DIM;f++){ feat_mean[f]=(float)omean[f]; feat_std[f]=(float)sqrt(oM2[f]/(on-1))+1e-8f; }
        fprintf(stderr,"Online stats from %ld samples\n",on); }

    // ---- seed: condition on 512 bytes ----
    int soff=seed_start+warmup; int slen=(int)(need-soff); if(slen>512) slen=512;
    for (int i=0;i<slen;i++){ uint8_t b=fdat[soff+i];
        int gate=eval_gate(b,ctx1,ctx2);
        see_observe(&see,b); see_extract(&see,fa);
        l2_evolve(L2,fa,gate);
        ctx2=ctx1; ctx1=b;
    }
    fprintf(stderr,"--- generated ---\n");

    uint64_t rng = rng_time ? ((uint64_t)time(NULL)^0xdeadbeefcafeULL^(uint64_t)seed_start)
                            : (rng_seed ^ ((uint64_t)seed_start*0x9E3779B97F4A7C15ULL));
    if (rng==0) rng=0x9E3779B97F4A7C15ULL;

    if (tel) fprintf(tel,"step\tbyte\tbpb\tmlprms\tent\tmaxp\tgate\n");

    double self_bits=0; uint8_t* gen=malloc(gen_len); float Pp[CLASSES];
    float* hid=malloc(((g_H>0)?g_H:1)*4); float mlg[CLASSES];
    for (int i=0;i<gen_len;i++){
        see_extract(&see, raw);                     // raw[0..191] = SEE
        memcpy(raw+BASE_DIM,L2,L2_DIM*sizeof(float));
        norm_feats(raw,nf);
        mlp_logits(nf,hid,mlg);
        const float* tri=&trigram[ctx2][ctx1][0];
        float logits[CLASSES], mxl=-1e30f;
        for (int c=0;c<CLASSES;c++){ logits[c]=(tri[c]+mlg[c])/temp; if(logits[c]>mxl)mxl=logits[c]; }
        float Z=0; for(int c=0;c<CLASSES;c++){ Pp[c]=expf(logits[c]-mxl); Z+=Pp[c]; } for(int c=0;c<CLASSES;c++) Pp[c]/=Z;
        uint8_t next;
        if (argmax){ float bb=-1; next=0; for(int c=0;c<CLASSES;c++) if(Pp[c]>bb){bb=Pp[c];next=(uint8_t)c;} }
        else next=sample_from(Pp,&rng);
        self_bits += -log2((double)fmaxf(Pp[next],1e-30f));
        gen[i]=next;

        if (tel){
            double mu=0; for(int c=0;c<CLASSES;c++) mu+=mlg[c]; mu/=CLASSES;
            double v=0; for(int c=0;c<CLASSES;c++){ double d=mlg[c]-mu; v+=d*d; } double rms=sqrt(v/CLASSES);
            double H=0,pm=0; for(int c=0;c<CLASSES;c++){ double p=Pp[c]; if(p>1e-12)H-=p*log2(p); if(p>pm)pm=p; }
            fprintf(tel,"%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%d\n", i,(int)next,self_bits/(i+1),rms,H,pm,eval_gate(next,ctx1,ctx2));
        }

        see_observe(&see,next); see_extract(&see,fa);
        int gate=eval_gate(next,ctx1,ctx2);
        l2_evolve(L2,fa,gate);
        ctx2=ctx1; ctx1=next;
    }
    fwrite(gen,1,gen_len,stdout);
    fprintf(stderr,"\n--- stats ---\nself_BPB: %.4f\n", self_bits/gen_len);
    if (tel) fclose(tel);
    free(trigram); free(ent_table); free(fdat); free(gen);
    return 0;
}
