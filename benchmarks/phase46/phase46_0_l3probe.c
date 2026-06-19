// Phase 46.0 - slow phrase/episode L3 boundary probe above D1 (diagnostic, NO training)
//
// Axis B (chosen after the 45.A-D substrate-side L2-write dead end): instead of controlling
// the volatile per-event L2 delta write, add a SLOWER, higher-level memory (L3) on top of the
// near-stable D1 regime - a memory that refreshes RARELY (phrase / episode scale), not every
// L2 boundary. 46.0 does not build L3 yet. It MEASURES candidate slow-boundary definitions on
// the real D1 trajectory and estimates how often L3 should update, so 46.A can be designed.
//
// Base = D1 (or D1cap20): entropy-high gate, mix0.5, scale0.5, alpha0.99. The readout stays
// linear (we use D1's own co-adapted readout to score loss / margin). Teacher-forced over a
// val window. No linguistic parser, no bigram counters, no generative hacks.
//
// Candidate L3-event definitions (ONLY the allowed signals: entropy/surprise, punct/ws byte
// class, L2-gate clusters, low-margin episodes, L2 state-change summaries):
//   PUNCT      sentence-end byte  . ! ?                         (byte class)
//   NEWLINE    \n                                               (byte class, paragraph/story)
//   L2CLUST    every K-th L2 gate firing                        (L2-gate clusters, K=8)
//   ENT_EPISODE rising edge of entropy above a high bar + refr  (entropy episode onset)
//   LOWMARGIN  prob margin < m_thr sustained >= M bytes         (low-margin episode onset)
//   STATECHANGE cumulative ||L2 - L2_anchor||/||anchor|| >= s   (L2 state-change summary)
//
// For each definition: event frequency, inter-event gap mean/p50/p90, a candidate L3 feature
// (slow EMA of L2 refreshed only at those events) and its stability (rel-move p50/p90, ||L3||
// range), a cheap loss-reduction signal (mean loss in [p,p+W) minus [p-W,p)), and onset text
// examples. Then a proposed 46.A grid (3-4 variants in the L3-scale band).
//
// Build (from repo root):
//   gcc -O3 -march=native -mavx2 -mfma \
//       benchmarks/phase38-42/phase46_0_l3probe.c \
//       src/silicon_entropy.c src/silicon_v0.c \
//       -o bin/phase46_0_l3probe.exe -lm -I .
// Run:
//   bin/phase46_0_l3probe.exe <dataset> <D1_weights> [--start N --len N --k 8 --refr 16
//                                                      --margin 0.10 --msteps 4 --statechg 0.5
//                                                      --enthi 0 --lwin 32 --out f.tsv]

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
#define TOT_DIM   (BASE_DIM + L2_DIM)
#define P_SEED    0xB5297A4Du
#define L3_ALPHA  0.9f       // candidate L3 EMA decay (slow; updates are already sparse)

enum { G_NONE=0, G_PUNCT=1, G_WS=2, G_SURPRISE=3, G_ENTROPY=4, G_COMBINED=5 };
enum { D_PUNCT=0, D_NEWLINE=1, D_L2CLUST=2, D_ENT=3, D_LOWMARGIN=4, D_STATECHG=5, NDEF=6 };
static const char* DEFN[NDEF]={"PUNCT","NEWLINE","L2CLUST","ENT_EPISODE","LOWMARGIN","STATECHANGE"};

static float (*trigram)[CLASSES][CLASSES];
static float (*ent_table)[CLASSES];
static float feat_mean[TOT_DIM], feat_std[TOT_DIM];
static float Wm[CLASSES][TOT_DIM], Bv[CLASSES];
static float Pmat[L2_DIM][BASE_DIM];
static uint8_t* data; static long fsz;

static int   g_gate=G_ENTROPY, g_ent_high=1, g_cooldown=0, g_delta=0;
static float g_surp_thr=0.0f, g_ent_thr=0.0f, g_nb_decay=1.0f;
static float g_base_clamp=2.0f, g_l2_clamp=2.0f, g_l2_scale=1.0f, g_mix=0.5f, g_alpha=0.99f;

static inline float dot_simd(const float* w, const float* f, int n) {
    __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(&w[i]),_mm256_loadu_ps(&f[i]),s);
    float o[8]; _mm256_storeu_ps(o,s);
    float r=o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; for(;i<n;i++) r+=w[i]*f[i]; return r;
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
static int cmp_f(const void* a, const void* b){ float x=*(const float*)a,y=*(const float*)b; return (x<y)?-1:((x>y)?1:0); }
static float pctl(float* a, long n, double q){ if(n<1) return 0.0f; long i=(long)floor(q*(n-1)); if(i<0)i=0; if(i>=n)i=n-1; return a[i]; }

int main(int argc, char** argv) {
    if (argc<3){ fprintf(stderr,"Usage: %s <dataset> <D1_weights> [opts]\n",argv[0]); return 1; }
    long mstart=-1, mlen=2000000, K=8, refr=16, msteps=4, lwin=32;
    double m_thr=0.10, s_thr=0.5, ent_hi=0.0;   // ent_hi=0 -> auto (gate ent_thr + 0.5)
    const char* outp="results/phase46_0/l3events.tsv";
    for (int i=3;i<argc;i++){
        if      (!strcmp(argv[i],"--start")   && i+1<argc) mstart=atol(argv[++i]);
        else if (!strcmp(argv[i],"--len")     && i+1<argc) mlen=atol(argv[++i]);
        else if (!strcmp(argv[i],"--k")       && i+1<argc) K=atol(argv[++i]);
        else if (!strcmp(argv[i],"--refr")    && i+1<argc) refr=atol(argv[++i]);
        else if (!strcmp(argv[i],"--margin")  && i+1<argc) m_thr=atof(argv[++i]);
        else if (!strcmp(argv[i],"--msteps")  && i+1<argc) msteps=atol(argv[++i]);
        else if (!strcmp(argv[i],"--statechg")&& i+1<argc) s_thr=atof(argv[++i]);
        else if (!strcmp(argv[i],"--enthi")   && i+1<argc) ent_hi=atof(argv[++i]);
        else if (!strcmp(argv[i],"--lwin")    && i+1<argc) lwin=atol(argv[++i]);
        else if (!strcmp(argv[i],"--out")     && i+1<argc) outp=argv[++i];
    }

    FILE* fd=fopen(argv[1],"rb"); if(!fd){fprintf(stderr,"Cannot open %s\n",argv[1]);return 1;}
    fseek(fd,0,SEEK_END); fsz=ftell(fd); fseek(fd,0,SEEK_SET);
    data=malloc(fsz); fread(data,1,fsz,fd); fclose(fd);
    if (mstart<0) mstart=(fsz*50)/100;
    if (mstart+mlen+3 > fsz) mlen=fsz-mstart-3;
    if (mlen<2000){ fprintf(stderr,"window too small\n"); return 1; }

    // ---- generic weight loader (0x5345453C .. 0x53454543) ----
    FILE* fw=fopen(argv[2],"rb"); if(!fw){fprintf(stderr,"Cannot open weights %s\n",argv[2]);return 1;}
    uint32_t magic; fread(&magic,4,1,fw); rewind(fw);
    if (magic<0x5345453C || magic>0x53454543){ fprintf(stderr,"Unexpected magic 0x%08x\n",magic); return 1; }
    int has_homeo=(magic>=0x5345453D), has_calib=(magic>=0x5345453E), has_scale=(magic>=0x5345453F);
    int has_cap=(magic>=0x53454540), has_relmove=(magic>=0x53454541), has_wgeom=(magic>=0x53454542), has_wtgate=(magic>=0x53454543);
    uint32_t hdr4[4]; fread(hdr4,4,4,fw);
    float hf[5]; fread(hf,4,5,fw);
    float decay=hf[0], alpha_fast=hf[2], feat_clamp=hf[3]; g_base_clamp=feat_clamp;
    uint32_t no=0; fread(&no,4,1,fw); int n_oja=(int)no;
    if (n_oja<0||n_oja>SEE_N_OJA_MAX){ fprintf(stderr,"bad n_oja %d\n",n_oja); return 1; }
    float W_oja_buf[SEE_N_OJA_MAX*43]; fread(W_oja_buf,sizeof(float),(size_t)n_oja*43,fw);
    uint32_t l2d=0,gt=0,eh=0,ps=0; float al=0,st=0,et=0;
    fread(&l2d,4,1,fw); fread(&gt,4,1,fw); fread(&al,4,1,fw);
    fread(&st,4,1,fw); fread(&et,4,1,fw); fread(&eh,4,1,fw); fread(&ps,4,1,fw);
    if ((int)l2d!=L2_DIM){ fprintf(stderr,"L2_DIM mismatch %u\n",l2d); return 1; }
    g_gate=(int)gt; g_alpha=al; g_surp_thr=st; g_ent_thr=et; g_ent_high=(int)eh;
    g_l2_clamp=feat_clamp; g_nb_decay=1.0f; g_cooldown=0; g_delta=0; g_mix=0.0f; g_l2_scale=1.0f;
    if (has_homeo){ float l2c=0,nbd=1.0f; uint32_t cd=0,dl=0;
        fread(&l2c,4,1,fw); fread(&nbd,4,1,fw); fread(&cd,4,1,fw); fread(&dl,4,1,fw);
        g_l2_clamp=(l2c>0.0f)?l2c:feat_clamp; g_nb_decay=nbd; g_cooldown=(int)cd; g_delta=(int)dl;
        g_mix=g_delta?1.0f:0.0f; }
    if (has_calib){ float mx=0; fread(&mx,4,1,fw); g_mix=mx; }
    if (has_scale){ float ls=1.0f; fread(&ls,4,1,fw); g_l2_scale=(ls>0.0f)?ls:1.0f; }
    if (has_cap){ float cp=0; fread(&cp,4,1,fw); /* l2_cap: logit-side, no effect on L2 trajectory */ }
    if (has_relmove){ float rc=0; fread(&rc,4,1,fw); }
    if (has_wgeom){ uint32_t wg=0; fread(&wg,4,1,fw); }
    if (has_wtgate){ uint32_t a; float b; fread(&a,4,1,fw); fread(&b,4,1,fw); fread(&b,4,1,fw); fread(&a,4,1,fw); fread(&a,4,1,fw); }
    size_t tri_n=(size_t)CLASSES*CLASSES*CLASSES;
    trigram=malloc(tri_n*sizeof(float));
    fread(trigram,sizeof(float),tri_n,fw);
    fread(feat_mean,sizeof(float),TOT_DIM,fw);
    fread(feat_std, sizeof(float),TOT_DIM,fw);
    fread(Wm,sizeof(float),CLASSES*TOT_DIM,fw);
    fread(Bv,sizeof(float),CLASSES,fw);
    fclose(fw);

    gen_projection(ps);
    ent_table=malloc(CLASSES*CLASSES*sizeof(float));
    for(int i=0;i<CLASSES;i++) for(int j=0;j<CLASSES;j++){
        float m=-1e9f; for(int k=0;k<CLASSES;k++) if(trigram[i][j][k]>m) m=trigram[i][j][k];
        double se=0; for(int k=0;k<CLASSES;k++) se+=exp((double)(trigram[i][j][k]-m));
        double H=0; for(int k=0;k<CLASSES;k++){ double p=exp((double)(trigram[i][j][k]-m))/se; if(p>1e-12) H-=p*log(p); }
        ent_table[i][j]=(float)H;
    }
    if (ent_hi<=0.0) ent_hi = (double)g_ent_thr + 0.5;   // auto high bar above the L2 gate

    SiliconEntropyState see;
    see_init(&see, 42, 4, decay);
    see.multiscale_mode=1; see.alpha_fast=alpha_fast; see.alpha_mid=0.9f; see.alpha_slow=0.99f;
    see.n_oja=n_oja; memcpy(see.W_oja,W_oja_buf,(size_t)n_oja*43*sizeof(float));
    see.eta_oja=0.0f; see.plastic_blend=1.0f;

    fprintf(stderr,"46.0 L3 probe: magic=0x%08x gate=%d ent_thr=%.3f ent_hi=%.3f mix=%.2f scale=%.2f alpha=%.2f cd=%d nbd=%.3f\n",
            magic,g_gate,g_ent_thr,ent_hi,g_mix,g_l2_scale,g_alpha,g_cooldown,g_nb_decay);
    fprintf(stderr,"window start=%ld len=%ld | K=%ld refr=%ld margin=%.2f msteps=%ld statechg=%.2f lwin=%ld\n",
            mstart,mlen,K,refr,m_thr,msteps,s_thr,lwin);
    if (g_mix>0.6f) fprintf(stderr,"WARNING: mix=%.2f looks like DELTA, not D1 (expected ~0.5)\n",g_mix);

    // align SEE to mstart+1 (L2 cold-starts at 0, mirrors trainer val extract)
    see_reset(&see);
    for (long i=0;i<=mstart+1;i++) see_observe(&see, data[i]);

    float L2[L2_DIM]; memset(L2,0,sizeof(L2));
    float prevb[BASE_DIM]; memset(prevb,0,sizeof(prevb));
    float feat192[BASE_DIM], fa[BASE_DIM], blend[BASE_DIM], nf[TOT_DIM];
    float scale=1.0f/sqrtf((float)BASE_DIM);
    int cd=0;

    float* loss=malloc(mlen*sizeof(float));
    // per-definition event bookkeeping
    long  *evpos[NDEF]; float *evgap[NDEF]; float *evrm[NDEF];
    long   evn[NDEF]; long lastpos[NDEF];
    float  L3[NDEF][L2_DIM]; int L3init[NDEF]; double L3nmin[NDEF], L3nmax[NDEF];
    for(int d=0;d<NDEF;d++){ evpos[d]=malloc(mlen*sizeof(long)); evgap[d]=malloc(mlen*sizeof(float)); evrm[d]=malloc(mlen*sizeof(float));
        evn[d]=0; lastpos[d]=-1; L3init[d]=0; memset(L3[d],0,sizeof(L3[d])); L3nmin[d]=1e30; L3nmax[d]=0; }
    long l2gate_total=0, l2clust_ctr=0;
    int  prev_ent_hi=0;                 // for ENT rising edge
    long lowmargin_run=0; long lowmargin_refr=0;  // LOWMARGIN episode state
    float anchor[L2_DIM]; memset(anchor,0,sizeof(anchor)); int anchor_init=0;  // STATECHANGE

    FILE* fo=fopen(outp,"w");
    if(!fo){ fprintf(stderr,"Cannot open out %s (mkdir results/phase46_0)\n",outp); return 1; }
    fprintf(fo,"# def\tpos\tgap\tbyte\tentropy\tmargin\tl2gate\tL3relmove\n");

    for (long i=0;i<mlen;i++){
        long g=mstart+i;
        uint8_t c2=data[g], c1=data[g+1], tgt=data[g+2];
        see_extract(&see, feat192);
        // normalize [feat192|L2] and score loss + margin on the true next byte
        for (int f=0;f<BASE_DIM;f++){ float x=(feat192[f]-feat_mean[f])/(feat_std[f]+1e-8f);
            if(g_base_clamp>0){ if(x>g_base_clamp)x=g_base_clamp; if(x<-g_base_clamp)x=-g_base_clamp; } nf[f]=x; }
        for (int j=0;j<L2_DIM;j++){ int f=BASE_DIM+j; float x=(L2[j]-feat_mean[f])/(feat_std[f]+1e-8f);
            if(g_l2_clamp>0){ if(x>g_l2_clamp)x=g_l2_clamp; if(x<-g_l2_clamp)x=-g_l2_clamp; } nf[f]=x*g_l2_scale; }
        const float* tri=&trigram[c2][c1][0];
        float lg[CLASSES], mx=-1e30f;
        for (int c=0;c<CLASSES;c++){ lg[c]=Bv[c]+tri[c]+dot_simd(Wm[c],nf,TOT_DIM); if(lg[c]>mx)mx=lg[c]; }
        double Z=0; for(int c=0;c<CLASSES;c++) Z+=exp((double)(lg[c]-mx));
        double p1=-1,p2=-1; for(int c=0;c<CLASSES;c++){ double p=exp((double)(lg[c]-mx))/Z; if(p>p1){p2=p1;p1=p;} else if(p>p2)p2=p; }
        double ptgt=exp((double)(lg[tgt]-mx))/Z;
        loss[i]=(float)(-log2(ptgt>1e-30?ptgt:1e-30));
        double margin=p1-p2;
        double entropy=ent_table[c2][c1];

        see_observe(&see, tgt);
        int l2gate = eval_gate(tgt, c1, c2);
        int wrote=0;
        if (l2gate && cd==0){            // D1 L2 evolve (gated delta-mix EMA, WG/WT none)
            see_extract(&see, fa);
            const float* src=fa;
            if (g_mix>0.0f){ for(int k=0;k<BASE_DIM;k++) blend[k]=fa[k]-g_mix*prevb[k]; src=blend; memcpy(prevb,fa,sizeof(fa)); }
            float w[L2_DIM]; for(int j=0;j<L2_DIM;j++){ float pp=0; const float* pj=Pmat[j]; for(int k=0;k<BASE_DIM;k++) pp+=pj[k]*src[k]; w[j]=pp*scale; }
            for (int j=0;j<L2_DIM;j++) L2[j]=g_alpha*L2[j]+(1.0f-g_alpha)*w[j];
            cd=g_cooldown; wrote=1; l2gate_total++;
        } else { if(!l2gate && g_nb_decay<1.0f) for(int j=0;j<L2_DIM;j++) L2[j]*=g_nb_decay; if(cd>0)cd--; }

        // ---- evaluate the NDEF candidate L3-event detectors at this byte ----
        int fire[NDEF]; for(int d=0;d<NDEF;d++) fire[d]=0;
        if (tgt=='.'||tgt=='!'||tgt=='?') fire[D_PUNCT]=1;
        if (tgt=='\n') fire[D_NEWLINE]=1;
        if (wrote){ l2clust_ctr++; if(l2clust_ctr>=K){ fire[D_L2CLUST]=1; l2clust_ctr=0; } }
        { int hi=(entropy>ent_hi)?1:0;
          if (hi && !prev_ent_hi && (lastpos[D_ENT]<0 || (g-lastpos[D_ENT])>=refr)) fire[D_ENT]=1;
          prev_ent_hi=hi; }
        { if (margin<m_thr){ lowmargin_run++; } else { lowmargin_run=0; }
          if (lowmargin_run==msteps && (lowmargin_refr<=0)){ fire[D_LOWMARGIN]=1; lowmargin_refr=refr; }
          if (lowmargin_refr>0) lowmargin_refr--; }
        { if(!anchor_init){ memcpy(anchor,L2,sizeof(anchor)); anchor_init=1; }
          else { double an2=0,dd2=0; for(int j=0;j<L2_DIM;j++){ an2+=(double)anchor[j]*anchor[j]; double q=(double)L2[j]-anchor[j]; dd2+=q*q; }
            double denom=(sqrt(an2)>1.0)?sqrt(an2):1.0; if(sqrt(dd2)/denom>=s_thr){ fire[D_STATECHG]=1; memcpy(anchor,L2,sizeof(anchor)); } } }

        for (int d=0;d<NDEF;d++) if (fire[d]){
            long gap = (lastpos[d]>=0)?(g-lastpos[d]):-1;
            // candidate L3 feature: slow EMA of the current L2 state, refreshed only here
            double rm=0;
            if (!L3init[d]){ memcpy(L3[d],L2,sizeof(L3[d])); L3init[d]=1; rm=0; }
            else { double o2=0,d2=0; float nl[L2_DIM];
                for(int j=0;j<L2_DIM;j++){ nl[j]=L3_ALPHA*L3[d][j]+(1.0f-L3_ALPHA)*L2[j]; double dd=(double)nl[j]-L3[d][j]; d2+=dd*dd; o2+=(double)L3[d][j]*L3[d][j]; }
                double den=(sqrt(o2)>1e-6)?sqrt(o2):1e-6; rm=sqrt(d2)/den; memcpy(L3[d],nl,sizeof(nl)); }
            double n2=0; for(int j=0;j<L2_DIM;j++) n2+=(double)L3[d][j]*L3[d][j]; double ln=sqrt(n2);
            if(ln<L3nmin[d])L3nmin[d]=ln; if(ln>L3nmax[d])L3nmax[d]=ln;
            evpos[d][evn[d]]=g; if(gap>=0) evgap[d][evn[d]]=(float)gap; else evgap[d][evn[d]]=-1;
            evrm[d][evn[d]]=(float)rm; evn[d]++;
            lastpos[d]=g;
            if (evn[d]<=20000) fprintf(fo,"%s\t%ld\t%ld\t%d\t%.3f\t%.4f\t%d\t%.4f\n",DEFN[d],g,gap,(int)tgt,entropy,margin,l2gate,rm);
        }
    }
    fclose(fo);

    double mean_loss=0; for(long i=0;i<mlen;i++) mean_loss+=loss[i]; mean_loss/=mlen;
    fprintf(stderr,"\n--- 46.0 L3 boundary survey (D1, %ld bytes, mean loss %.4f bpb, L2 gates=%ld = %.1f%%) ---\n",
            mlen, mean_loss, l2gate_total, 100.0*l2gate_total/mlen);
    fprintf(stderr,"%-12s %8s %9s %7s %7s %7s %9s %9s %9s\n",
            "def","events","per1k","gapMean","gapP50","gapP90","L3rm_p50","L3rm_p90","dLoss(W)");
    fprintf(stderr,"  %s\n","--------------------------------------------------------------------------------------");
    float* tmp=malloc(mlen*sizeof(float));
    for (int d=0;d<NDEF;d++){
        long n=evn[d];
        if (n<2){ fprintf(stderr,"%-12s %8ld   (too few)\n",DEFN[d],n); continue; }
        // gap stats (skip the first -1)
        long ng=0; for(long e=0;e<n;e++) if(evgap[d][e]>=0) tmp[ng++]=evgap[d][e];
        double gm=0; for(long e=0;e<ng;e++) gm+=tmp[e]; gm/=(ng?ng:1);
        qsort(tmp,ng,sizeof(float),cmp_f); float gp50=pctl(tmp,ng,0.5), gp90=pctl(tmp,ng,0.9);
        // L3 rel-move stats
        long nr=0; for(long e=0;e<n;e++) if(evrm[d][e]>0) tmp[nr++]=evrm[d][e];
        qsort(tmp,nr,sizeof(float),cmp_f); float rp50=pctl(tmp,nr,0.5), rp90=pctl(tmp,nr,0.9);
        // cheap loss-reduction: mean loss in [p,p+W) minus [p-W,p), averaged over events
        double dl=0; long nl=0;
        for(long e=0;e<n;e++){ long r=evpos[d][e]-mstart; if(r-lwin<0||r+lwin>mlen) continue;
            double bef=0,aft=0; for(long t=1;t<=lwin;t++){ bef+=loss[r-t]; aft+=loss[r+t-1]; } dl+=(aft-bef)/lwin; nl++; }
        double dlm=(nl?dl/nl:0);
        fprintf(stderr,"%-12s %8ld %9.2f %7.0f %7.0f %7.0f %9.4f %9.4f %+9.4f\n",
                DEFN[d], n, 1000.0*n/mlen, gm, gp50, gp90, rp50, rp90, dlm);
    }
    free(tmp);

    // onset examples (first 4 events of each def, +-12 bytes, escaped)
    fprintf(stderr,"\n--- onset examples (teacher-forced text, '|' marks the event byte) ---\n");
    for (int d=0;d<NDEF;d++){
        if (evn[d]<1) continue;
        fprintf(stderr,"  %-12s ",DEFN[d]);
        for (int e=0; e<4 && e<evn[d]; e++){ long p=evpos[d][e]; fprintf(stderr,"[");
            for (long t=-10;t<=2;t++){ long q=p+t; if(q<0||q>=fsz) continue; uint8_t b=data[q];
                if(t==0) fprintf(stderr,"|");
                if(b=='\n') fprintf(stderr,"\\n"); else if(b=='\t') fprintf(stderr,"\\t"); else if(isprint(b)) fputc(b,stderr); else fprintf(stderr,"\\x%02x",b); }
            fprintf(stderr,"] "); }
        fprintf(stderr,"\n");
    }

    // proposed 46.A grid: defs in the L3-scale band (p50 gap in [20,500]) with structure
    fprintf(stderr,"\n--- proposed 46.A grid (L3-scale = gap p50 in [20,500]; prefer dLoss<0) ---\n");
    int picked=0;
    for (int d=0;d<NDEF && picked<4;d++){
        long n=evn[d]; if(n<2) continue;
        // recompute p50 from this def's gaps
        float* gg=malloc(n*sizeof(float)); long m2=0; for(long e=0;e<n;e++) if(evgap[d][e]>=0) gg[m2++]=evgap[d][e];
        qsort(gg,m2,sizeof(float),cmp_f); float gp50=pctl(gg,m2,0.5); free(gg);
        if (gp50>=20 && gp50<=500){ fprintf(stderr,"  [%d] %-12s  L3-update schedule, gap p50~%.0f bytes\n",++picked,DEFN[d],gp50); }
    }
    if (picked==0) fprintf(stderr,"  (no definition landed in the L3-scale band - widen thresholds: --k, --refr, --statechg, --enthi)\n");
    fprintf(stderr,"  -> 46.A: build L3 = slow EMA(L2) refreshed at the chosen schedule, concat [SEE|L2|L3], retrain linear readout.\n");
    fprintf(stderr,"     Keep <=3-4 variants. Readout stays linear. L3 updates rarely (see gaps above).\n");

    fprintf(stderr,"\nTSV: %s\n", outp);
    free(loss); for(int d=0;d<NDEF;d++){ free(evpos[d]); free(evgap[d]); free(evrm[d]); }
    free(trigram); free(ent_table); free(data);
    return 0;
}
