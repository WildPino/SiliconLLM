// Phase 45.D - per-event causal attribution of delta L2 writes (diagnostic, NO training)
//
// 45.A/B/C closed the substrate-side WRITE knobs (amplitude / geometry / hard gate) negative:
// hard gates on cos/reversal and deterministic thinning SWITCH L2 OFF (they drop the write
// instead of preserving it). Before declaring the delta signal "teacher-forced-only / non
// generative", we ask the prior question CAUSALLY: which delta writes actually HELP the next
// byte, which HURT it, and is that split SEPARABLE by internal signals?
//
// Teacher-forced over a val window, using the co-adapted C0_delta readout (no training). The
// L2 trajectory is the real delta one (mix1 scale0.5 alpha0.99 entropy-high gate, WG_NONE,
// WT_NONE). At each WRITE event we snapshot L2_old (pre-EMA); at the NEXT prediction step we
// compute the marginal counterfactual: loss(next byte | L2_new) vs loss(next byte | L2_old),
// everything else fixed. dloss = loss_without - loss_with  (>0 => the write HELPED).
//
// Per event we log: cos(write,L2_old), rel_move, write_norm, surprise, entropy, punct/ws
// boundary flags, loss_with, loss_without, dloss, useful. A TSV; phase45d_attribution.ps1
// does the separability analysis (correlations, quantile bins, threshold retention).
//
// This reads a 0x53454543 weight (write-gate format); WT_NONE/WG_NONE fields are ignored.
//
// Build (from repo root):
//   gcc -O3 -march=native -mavx2 -mfma \
//       benchmarks/phase38-42/phase45d_attribution.c \
//       src/silicon_entropy.c src/silicon_v0.c \
//       -o bin/phase45d_attribution.exe -lm -I .
// Run:
//   bin/phase45d_attribution.exe <dataset> <C0_delta_weights> [--start N --len N --out f.tsv]

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
#define TOT_DIM   (BASE_DIM + L2_DIM)
#define P_SEED    0xB5297A4Du

enum { G_NONE=0, G_PUNCT=1, G_WS=2, G_SURPRISE=3, G_ENTROPY=4, G_COMBINED=5 };

static float (*trigram)[CLASSES][CLASSES];   // [c2][c1][k]
static float (*ent_table)[CLASSES];          // [c2][c1]
static float feat_mean[TOT_DIM], feat_std[TOT_DIM];
static float Wm[CLASSES][TOT_DIM], Bv[CLASSES];
static float Pmat[L2_DIM][BASE_DIM];
static uint8_t* data; static long fsz;

// homeostasis/config read from the weight
static int   g_gate=G_ENTROPY, g_ent_high=1;
static float g_surp_thr=0.0f, g_ent_thr=0.0f;
static float g_base_clamp=2.0f, g_l2_clamp=2.0f, g_l2_scale=1.0f, g_mix=1.0f, g_alpha=0.99f;

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

// normalized+clamped+scaled feature row for a given [feat192 | L2] raw vector
static inline void normalize_row(const float* feat192, const float* L2, float* nf) {
    for (int f=0;f<BASE_DIM;f++){
        float x=(feat192[f]-feat_mean[f])/(feat_std[f]+1e-8f);
        if (g_base_clamp>0.0f){ if(x>g_base_clamp)x=g_base_clamp; if(x<-g_base_clamp)x=-g_base_clamp; }
        nf[f]=x;
    }
    for (int j=0;j<L2_DIM;j++){ int f=BASE_DIM+j;
        float x=(L2[j]-feat_mean[f])/(feat_std[f]+1e-8f);
        if (g_l2_clamp>0.0f){ if(x>g_l2_clamp)x=g_l2_clamp; if(x<-g_l2_clamp)x=-g_l2_clamp; }
        nf[f]=x*g_l2_scale;
    }
}
// index-sort by an external key (for quantile bins / threshold sweeps)
static const float* g_sortkey;
static int cmp_idx(const void* a, const void* b){
    float x=g_sortkey[*(const int*)a], y=g_sortkey[*(const int*)b];
    return (x<y)?-1:((x>y)?1:0);
}
static double pearson(const float* x, const float* d, long n){
    if(n<3) return 0.0;
    double mx=0,md=0; for(long i=0;i<n;i++){ mx+=x[i]; md+=d[i]; } mx/=n; md/=n;
    double sxy=0,sxx=0,sdd=0;
    for(long i=0;i<n;i++){ double a=x[i]-mx, b=d[i]-md; sxy+=a*b; sxx+=a*a; sdd+=b*b; }
    double den=sqrt(sxx*sdd); return (den>1e-12)?sxy/den:0.0;
}
// One signal's separability report: Pearson r, decile-mean(dloss), and the best single
// threshold's KEPT net dloss (keep the side with higher mean) vs the oracle (keep dloss>0).
static void analyze_signal(const char* name, const float* sig, const float* d, long n,
                           double oracle_pos){
    double r=pearson(sig,d,n);
    int* idx=malloc(n*sizeof(int)); for(long i=0;i<n;i++) idx[i]=(int)i;
    g_sortkey=sig; qsort(idx,n,sizeof(int),cmp_idx);
    // prefix sums of dloss in signal-sorted order -> any threshold split is O(1)
    // best split keeping the higher-mean side
    double total=0; for(long i=0;i<n;i++) total+=d[i];
    double pre=0, best_net=-1e30; double best_thr=0; const char* best_side="";
    long best_keep=0;
    for(long i=1;i<n;i++){ pre+=d[idx[i-1]];
        double lo_net=pre;                 // keep events with sig <= thr
        double hi_net=total-pre;           // keep events with sig >  thr
        if(lo_net>best_net){ best_net=lo_net; best_thr=sig[idx[i]]; best_side="<="; best_keep=i; }
        if(hi_net>best_net){ best_net=hi_net; best_thr=sig[idx[i]]; best_side=">";  best_keep=n-i; }
    }
    fprintf(stderr,"  %-11s r=%+.3f   best-thr %s %-8.3f  kept_net=%+.2f bits (n=%ld)  vs oracle +%.2f\n",
            name, r, best_side, best_thr, best_net, best_keep, oracle_pos);
    // decile means
    fprintf(stderr,"    decile mean(dloss): ");
    for(int q=0;q<10;q++){ long a=(long)((double)q*n/10), b=(long)((double)(q+1)*n/10);
        double s=0; long c=0; for(long i=a;i<b;i++){ s+=d[idx[i]]; c++; }
        fprintf(stderr,"%+.4f ", c?s/c:0); }
    fprintf(stderr,"\n");
    free(idx);
}

// -log2 P(target) under the readout, given a normalized row and bigram context
static inline double loss_of(const float* nf, uint8_t c1, uint8_t c2, uint8_t tgt) {
    const float* tri=&trigram[c2][c1][0];
    float lg[CLASSES], mx=-1e30f;
    for (int c=0;c<CLASSES;c++){ lg[c]=Bv[c]+tri[c]+dot_simd(Wm[c],nf,TOT_DIM); if(lg[c]>mx)mx=lg[c]; }
    double Z=0; for(int c=0;c<CLASSES;c++) Z+=exp((double)(lg[c]-mx));
    double p=exp((double)(lg[tgt]-mx))/Z;
    return -log2(p>1e-30?p:1e-30);
}

int main(int argc, char** argv) {
    if (argc<3){ fprintf(stderr,"Usage: %s <dataset> <C0_delta_weights> [--start N --len N --out f.tsv]\n",argv[0]); return 1; }
    long mstart=-1, mlen=2000000; const char* outp="results/phase45d/events.tsv";
    for (int i=3;i<argc;i++){
        if      (!strcmp(argv[i],"--start") && i+1<argc) mstart=atol(argv[++i]);
        else if (!strcmp(argv[i],"--len")   && i+1<argc) mlen=atol(argv[++i]);
        else if (!strcmp(argv[i],"--out")   && i+1<argc) outp=argv[++i];
    }

    FILE* fd=fopen(argv[1],"rb"); if(!fd){fprintf(stderr,"Cannot open %s\n",argv[1]);return 1;}
    fseek(fd,0,SEEK_END); fsz=ftell(fd); fseek(fd,0,SEEK_SET);
    data=malloc(fsz); fread(data,1,fsz,fd); fclose(fd);
    if (mstart<0) mstart=(fsz*50)/100;   // default: start of val region
    if (mstart+mlen+3 > fsz) mlen=fsz-mstart-3;
    if (mlen<1000){ fprintf(stderr,"window too small\n"); return 1; }

    // ---- read weight (0x53454543 layout) ----
    FILE* fw=fopen(argv[2],"rb"); if(!fw){fprintf(stderr,"Cannot open weights %s\n",argv[2]);return 1;}
    uint32_t magic; fread(&magic,4,1,fw); rewind(fw);
    if (magic!=0x53454543){ fprintf(stderr,"Expected 0x53454543 (C0_delta), got 0x%08x\n",magic); return 1; }
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
    // homeostasis + mix + l2_scale + l2_cap + relmove_cap + wgeom + write-gate
    float l2c=0,nbd=1.0f; uint32_t cd=0,dl=0;
    fread(&l2c,4,1,fw); fread(&nbd,4,1,fw); fread(&cd,4,1,fw); fread(&dl,4,1,fw);
    g_l2_clamp=(l2c>0.0f)?l2c:feat_clamp;
    float mx=0; fread(&mx,4,1,fw); g_mix=mx;
    float ls=1.0f; fread(&ls,4,1,fw); g_l2_scale=(ls>0.0f)?ls:1.0f;
    float l2cap=0,rmc=0; uint32_t wg=0,wtm=0,wck=0,wtn=0; float wcos=0,wstr=0;
    fread(&l2cap,4,1,fw); fread(&rmc,4,1,fw); fread(&wg,4,1,fw);
    fread(&wtm,4,1,fw); fread(&wcos,4,1,fw); fread(&wstr,4,1,fw); fread(&wck,4,1,fw); fread(&wtn,4,1,fw);
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

    SiliconEntropyState see;
    see_init(&see, 42, 4, decay);
    see.multiscale_mode=1; see.alpha_fast=alpha_fast; see.alpha_mid=0.9f; see.alpha_slow=0.99f;
    see.n_oja=n_oja; memcpy(see.W_oja,W_oja_buf,(size_t)n_oja*43*sizeof(float));
    see.eta_oja=0.0f; see.plastic_blend=1.0f;

    fprintf(stderr,"45.D attribution: magic=0x%08x gate=%d ent_thr=%.3f mix=%.2f l2_scale=%.2f l2_clamp=%.2f base_clamp=%.2f alpha=%.2f\n",
            magic,g_gate,g_ent_thr,g_mix,g_l2_scale,g_l2_clamp,g_base_clamp,g_alpha);
    fprintf(stderr,"window: start=%ld len=%ld (val=%ld..%ld)\n", mstart, mlen, (fsz*50)/100, fsz);

    // align SEE up to mstart+1 (mirror trainer extract_l2 start), L2 cold-starts at 0
    see_reset(&see);
    for (long i=0;i<=mstart+1;i++) see_observe(&see, data[i]);

    float L2[L2_DIM]; memset(L2,0,sizeof(L2));
    float prevb[BASE_DIM]; memset(prevb,0,sizeof(prevb));
    float feat192[BASE_DIM], fa[BASE_DIM], blend[BASE_DIM];
    float scale=1.0f/sqrtf((float)BASE_DIM);

    // pending counterfactual from the previous step's write (resolved at the NEXT prediction)
    int pend=0; float pL2old[L2_DIM];
    double pcos=0,prel=0,pwn=0,psurp=0,pent=0; int ppun=0,pws=0; uint8_t pbyte=0;

    FILE* fo=fopen(outp,"w");
    if(!fo){ fprintf(stderr,"Cannot open out %s (mkdir results/phase45d first)\n",outp); return 1; }
    fprintf(fo,"# C0_delta per-event attribution: dloss = loss_without - loss_with (>0 = write helped next byte)\n");
    fprintf(fo,"idx\tpos\tbyte\tcos_wL2\trel_move\twrite_norm\tsurprise\tentropy\tpunct\tws\tloss_with\tloss_without\tdloss\tuseful\n");

    long nev=0, nuse=0, nharm=0; double sum_pos=0, sum_neg=0, sum_d=0;
    float nf[TOT_DIM];
    // per-event arrays for the separability analysis (upper bound = mlen)
    float *aD=malloc(mlen*sizeof(float)), *aCos=malloc(mlen*sizeof(float)), *aRel=malloc(mlen*sizeof(float));
    float *aWn=malloc(mlen*sizeof(float)), *aSurp=malloc(mlen*sizeof(float)), *aEnt=malloc(mlen*sizeof(float));

    for (long i=0;i<mlen;i++){
        long g=mstart+i;
        uint8_t c2=data[g], c1=data[g+1], tgt=data[g+2];
        see_extract(&see, feat192);

        // resolve a pending write's counterfactual on THIS byte (features use L2 = L2_new)
        if (pend){
            normalize_row(feat192, L2, nf);             // L2 here is post-write (L2_new)
            double lw = loss_of(nf, c1, c2, tgt);
            normalize_row(feat192, pL2old, nf);         // counterfactual: write suppressed
            double lo = loss_of(nf, c1, c2, tgt);
            double dl = lo - lw;                         // >0 => the write reduced next-byte loss
            int useful = (dl>0.0)?1:0;
            fprintf(fo,"%ld\t%ld\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t%d\t%.5f\t%.5f\t%.5f\t%d\n",
                    nev,g-1,(int)pbyte,pcos,prel,pwn,psurp,pent,ppun,pws,lw,lo,dl,useful);
            aD[nev]=(float)dl; aCos[nev]=(float)pcos; aRel[nev]=(float)prel; aWn[nev]=(float)pwn;
            aSurp[nev]=(float)psurp; aEnt[nev]=(float)pent;
            nev++; sum_d+=dl; if(dl>0){nuse++; sum_pos+=dl;} else {nharm++; sum_neg+=dl;}
            pend=0;
        }

        see_observe(&see, tgt);
        if (eval_gate(tgt, c1, c2)) {                    // boundary write on tgt
            see_extract(&see, fa);
            const float* src=fa;
            if (g_mix>0.0f){ for(int k=0;k<BASE_DIM;k++) blend[k]=fa[k]-g_mix*prevb[k]; src=blend; memcpy(prevb,fa,sizeof(fa)); }
            float w[L2_DIM]; double pd2=0.0;
            for (int j=0;j<L2_DIM;j++){ float p=0; const float* pj=Pmat[j]; for(int k=0;k<BASE_DIM;k++) p+=pj[k]*src[k]; p*=scale; w[j]=p; pd2+=(double)p*p; }
            double wn=sqrt(pd2);
            // internal signals vs L2_old (current L2, pre-EMA)
            double l2n2=0.0, dot=0.0; for(int j=0;j<L2_DIM;j++){ l2n2+=(double)L2[j]*L2[j]; dot+=(double)w[j]*L2[j]; }
            double l2n=sqrt(l2n2);
            double cosv=(wn>1e-12 && l2n>1e-12)?dot/(wn*l2n):0.0;
            // candidate EMA + rel_move
            double dn2=0.0; float cand[L2_DIM];
            for (int j=0;j<L2_DIM;j++){ cand[j]=g_alpha*L2[j]+(1.0f-g_alpha)*w[j]; double dd=(double)cand[j]-L2[j]; dn2+=dd*dd; }
            double denom=(l2n>1.0)?l2n:1.0; double relmv=sqrt(dn2)/denom;
            // snapshot L2_old, commit the write, arm the counterfactual for the next byte
            memcpy(pL2old, L2, sizeof(L2));
            for (int j=0;j<L2_DIM;j++) L2[j]=cand[j];
            pcos=cosv; prel=relmv; pwn=wn; psurp=-(double)trigram[c2][c1][tgt]; pent=ent_table[c2][c1];
            ppun=is_punct(tgt); pws=is_ws(tgt); pbyte=tgt; pend=1;
        }
    }
    fclose(fo);

    double net=sum_pos+sum_neg;
    fprintf(stderr,"\n--- 45.D summary (%ld events) ---\n", nev);
    fprintf(stderr,"useful (dloss>0): %ld (%.1f%%)   harmful: %ld (%.1f%%)\n",
            nuse,(nev?100.0*nuse/nev:0), nharm,(nev?100.0*nharm/nev:0));
    fprintf(stderr,"sum +dloss = %.2f bits   sum -dloss = %.2f bits   NET = %.2f bits   mean dloss = %.5f\n",
            sum_pos, sum_neg, net, (nev?sum_d/nev:0));

    fprintf(stderr,"\n--- separability: r(signal,dloss), best single-threshold KEPT net, decile means ---\n");
    fprintf(stderr,"  (oracle = keep exactly dloss>0 -> +%.2f bits. A signal separates iff its best\n",sum_pos);
    fprintf(stderr,"   kept_net approaches oracle and well exceeds keep-all NET=%.2f.)\n", net);
    if (nev>=10){
        analyze_signal("cos_wL2",  aCos,  aD, nev, sum_pos);
        analyze_signal("rel_move", aRel,  aD, nev, sum_pos);
        analyze_signal("write_norm",aWn,  aD, nev, sum_pos);
        analyze_signal("surprise", aSurp, aD, nev, sum_pos);
        analyze_signal("entropy",  aEnt,  aD, nev, sum_pos);
    }
    fprintf(stderr,"\nTSV: %s\n", outp);
    free(aD);free(aCos);free(aRel);free(aWn);free(aSurp);free(aEnt);
    free(trigram); free(ent_table); free(data);
    return 0;
}
