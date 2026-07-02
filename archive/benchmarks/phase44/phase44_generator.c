// Phase 44 Generator - boundary-gated L2 memory (magic 0x5345453C)
//
// Reconstructs the harness-level L2 memory during generation, identically to
// phase44a_boundary.c: SEE (C2.A frozen) -> 192D; L2 (64D) = gated EMA of a
// fixed random projection of the current SEE summary, updated only on boundary
// bytes; readout linear over [SEE 192 | L2 64] = 256D with online-normalized,
// clamped features.
//
// Build (from repo root):
//   gcc -O3 -march=native -mavx2 -mfma \
//       benchmarks/phase38-42/phase44_generator.c \
//       src/silicon_entropy.c src/silicon_v0.c \
//       -o bin/phase44_generator.exe -lm -I .
// Run:
//   phase44_generator.exe <data> <weights_0x5345453C> [--gen-len N] [--temp F]
//                         [--warmup N] [--seed-start N] [--rng-seed N] [--mode argmax]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "src/silicon_entropy.h"

#define CLASSES  256
#define BASE_DIM SEE_FEATURE_DIM   // 192
#define L2_DIM   64
#define TOT_DIM  (BASE_DIM + L2_DIM)

enum { G_NONE=0, G_PUNCT=1, G_WS=2, G_SURPRISE=3, G_ENTROPY=4, G_COMBINED=5 };

static float Pmat[L2_DIM][BASE_DIM];
static float (*trigram)[CLASSES][CLASSES];
static float (*ent_table)[CLASSES];
static float feat_mean[TOT_DIM], feat_std[TOT_DIM];
static float Wm[CLASSES][TOT_DIM], Bv[CLASSES];

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
    for (int j=0;j<L2_DIM;j++) for (int k=0;k<BASE_DIM;k++){
        s^=s<<13; s^=s>>7; s^=s<<17;
        Pmat[j][k] = (s & 1ULL) ? 1.0f : -1.0f;
    }
}
static inline int is_punct(uint8_t b){ return b=='.'||b=='!'||b=='?'||b=='\n'||b=='"'||b=='\''; }
static inline int is_ws(uint8_t b){ return b==' '||b=='\n'||b=='\t'||b=='.'||b==','||b=='!'||b=='?'||b==';'||b==':'; }

static int g_gate; static float g_alpha, g_surp_thr, g_ent_thr; static int g_ent_high;
// Phase 44.B homeostasis (defaults = off, i.e. 0x5345453C behaviour)
static float g_l2_clamp = 2.0f, g_nb_decay = 1.0f; static int g_cooldown = 0, g_delta = 0;
// Phase 44.C delta calibration: single write knob `mix` in [0,1]
//   src = SEE_t - mix * SEE_prev_boundary   (mix0=absolute, mix1=full delta).
//   Subsumes 44.B delta flag: g_delta=1 -> mix=1.0, g_delta=0 -> mix=0.0.
static float g_mix = 0.0f;
// Phase 44.D readout homeostasis: L2 block scale applied to L2 feature dims after
//   normalize+clamp (parity with trainer apply_l2_scale). 1.0 = no scaling.
static float g_l2_scale = 1.0f;
// Phase 44.E conditional trust: dynamic cap on the L2 logit contribution. Per step
//   gamma = (RMS_c(d-mean d)/RMS_c(s-mean s) > cap) ? cap/ratio : 1; logit=s+gamma*d.
//   cap<=0 = disabled. Fires only when L2 over-asserts (closed-loop attractor).
static float g_l2_cap = 0.0f;
// Phase 45.A substrate-side relative-move homeostasis: cap the per-event fractional
//   reorientation of the L2 memory. After the candidate EMA update L2_new, if
//   rel_move = ||L2_new-L2_old|| / max(||L2_old||,eps) > cap, scale the move down to
//   cap: L2 = L2_old + (cap/rel_move)*(L2_new-L2_old). cap<=0 = disabled (bit-identical
//   to the plain EMA). This is memory-side, NOT readout-side: it asks L2 not to swing
//   more than its natural per-event budget. Applied identically in trainer extract_l2.
#define L2_RELMOVE_EPS 1.0f
static float g_relmove_cap = 0.0f;
static float g_relmove_cap_override = -2.0f;   // diagnostic CLI override (sentinel: unset)
// Phase 45.B substrate-side L2 WRITE GEOMETRY: 45.A proved capping the move amplitude
//   destroys the delta compression - the useful signal lives in the strong reorientation
//   itself. 45.B instead changes the DIRECTION of the write (or what part of it is
//   admitted), without an amplitude cap. The write vector p (projection of the delta src
//   into L2 space) is transformed before the EMA. Applied identically in trainer
//   extract_l2 (parity-critical). WG_NONE = bit-identical to the plain delta write.
enum { WG_NONE=0, WG_BLEND75=1, WG_BLEND50=2, WG_ORTHO=3, WG_NOVELTY=4, WG_ENERGY=5 };
static int   g_wgeom = WG_NONE;
static int   g_wgeom_override = -1;   // diagnostic CLI override (sentinel: unset)
static float g_lastwrite[L2_DIM]; static int g_have_lastwrite = 0;   // previous write (for cosine audit)
static double g_l2_cos_wL2 = 0.0, g_l2_cos_wprev = 0.0;      // diagnostics: cos(write,L2_old), cos(write,prev_write)
// Phase 45.C substrate-side L2 WRITE-GATE (timing): 45.A proved capping amplitude kills
//   the delta gain; 45.B proved no simple geometry stabilizes (and revealed the key fact:
//   the useful delta write is ANTI-aligned to L2, cos~-0.70 - a correction/reversal, not
//   accumulation). 45.C asks WHEN to admit those strong reversals: accept or DROP the whole
//   EMA write per boundary, on internal scale-free signals only (reversal angle cos(w,L2),
//   candidate rel_move). No amplitude cap, no geometry. Applied identically in trainer
//   extract_l2 (parity-critical). WT_NONE = bit-identical to the plain delta write.
enum { WT_NONE=0, WT_REV25=1, WT_REV50=2, WT_REVMARGIN=3, WT_COOLDOWN=4, WT_THIN=5 };
static int   g_wt_mode = WT_NONE;
static float g_wt_cos_thr = 0.0f;      // reversal threshold (accept when cos(w,L2) < thr)
static float g_wt_stress_thr = 1.0f;   // C3 margin: also require candidate rel_move > thr
static int   g_wt_cool_k = 3;          // C4: boundary events to block after a strong reversal
static int   g_wt_thin_n = 2;          // C5: accept 1 delta write every N boundaries
static int   g_wt_cool = 0, g_wt_thin = 0;   // persistent write-gate state (reset before warmup)
static int   g_wt_mode_override = -1;        // diagnostic CLI override (sentinel: unset)
// Phase 44.G closed-loop telemetry (diagnostic; emits a per-step TSV when set).
// Same generation path as production - the log reflects the real trajectory.
#define TEL_PMAX 32
static FILE* g_tel = NULL;
static float g_prevL2[L2_DIM]; static int g_have_prevL2 = 0;
static int g_runp[TEL_PMAX+1];   // per-period run lengths for byte-cycle loop detection
// Phase 45.0 event-conditioned write diagnostics: last l2_evolve's raw projection
// norm ||p|| (pre-EMA, the magnitude of the incoming write), whether it actually
// updated (gate fired AND cooldown clear), and the gate decision it saw. Set inside
// l2_evolve; aligned with l2delta (both describe the most-recent evolve). Diagnostic
// only - reading these never alters L2 / RNG / logits, so generation stays bit-identical.
static double g_l2_wnorm = 0.0; static int g_l2_wrote = 0; static int g_l2_gatefired = 0;

// ---- Phase 44.H targeted attractor forensics (diagnostic; off unless armed) ----
// Word-level bigram tracking matching the tribunal's topBi metric: tokens are
// maximal [a-zA-Z] runs lowercased, length>=2, bigrams over adjacent surviving
// tokens. A "warning" fires the first time any bigram count reaches g_warn_bigram.
// Ablations apply from the warning step onward (causal: detect the loop forming ->
// intervene -> does it break?). Forensic emits a per-step rich row for windowing.
enum { ABL_NONE=0, ABL_L2ZERO=1, ABL_L2FREEZE=2, ABL_L2RESET=3, ABL_CAP=4 };
static int   g_ablate=ABL_NONE; static float g_ablate_cap=0.010f;
static int   g_warn_bigram=0, g_warned=0, g_warn_step=-1, g_reset_done=0;
static FILE* g_forensic=NULL;
#define BHASH 16384
static char  g_bkey[BHASH][40]; static int g_bcnt[BHASH];
static char  g_wcur[40]; static int g_wcurlen=0; static char g_wprev[40]; static int g_wprevok=0;
static char  g_curbi[84]=""; static int g_curbi_cnt=0; static int g_topbi_cnt=0;
static int bigram_inc(const char* key){
    uint32_t h=2166136261u; for(const char* p=key;*p;p++){ h^=(uint8_t)*p; h*=16777619u; }
    int idx=(int)(h&(BHASH-1));
    for(int probe=0;probe<BHASH;probe++){ int j=(idx+probe)&(BHASH-1);
        if(g_bcnt[j]==0){ strncpy(g_bkey[j],key,39); g_bkey[j][39]=0; g_bcnt[j]=1; return 1; }
        if(strcmp(g_bkey[j],key)==0) return ++g_bcnt[j];
    }
    return 0;
}
static void track_byte(uint8_t b, int step){
    if (isalpha(b)){ if(g_wcurlen<39) g_wcur[g_wcurlen++]=(char)tolower(b); return; }
    if (g_wcurlen>=2){ g_wcur[g_wcurlen]=0;
        if (g_wprevok){ char key[84]; snprintf(key,sizeof(key),"%s %s",g_wprev,g_wcur);
            int cnt=bigram_inc(key); strncpy(g_curbi,key,83); g_curbi[83]=0; g_curbi_cnt=cnt;
            if(cnt>g_topbi_cnt) g_topbi_cnt=cnt;
            if(g_warn_bigram>0 && !g_warned && cnt>=g_warn_bigram){ g_warned=1; g_warn_step=step;
                fprintf(stderr,"WARN_STEP: %d bigram='%s' count=%d\n",step,key,cnt); }
        }
        strncpy(g_wprev,g_wcur,39); g_wprev[39]=0; g_wprevok=1;
    }
    g_wcurlen=0;
}

// ---- Phase 44.I endogenous L2 trust fatigue (inference-only policy) ----------
// Triggers from INTERNAL signals only (no word counters - those are 44.H sentinels):
//   L2/SEE disagreement (flip), low prob-margin, gamma saturation, L2-state stress.
//   FLIP/LOWMARGIN reduce L2's logit trust for a window; STATE damps the L2 memory
//   after sustained internal stress. All off unless --fatigue is set.
enum { FAT_NONE=0, FAT_FLIP=1, FAT_LOWMARGIN=2, FAT_STATE=3 };
static int   g_fat_mode=FAT_NONE, g_fat_window=16, g_fat_stress=6;
static float g_fat_margin=0.15f, g_fat_trust=0.5f, g_fat_cap=0.005f, g_fat_gammalow=0.85f, g_fat_damp=0.85f;
static int   cd_ctr = 0; static float prev_bound[BASE_DIM];

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
// Phase 45.B write geometry: transform the delta write w (in L2 space) in place before
// the EMA. WG_NONE is a no-op (bit-identical to the plain delta write). feat192 is the
// post-boundary SEE extract (used to recompute the absolute/H0 projection for blends);
// pdnorm = ||delta_write|| is the target energy for direction-preserving renorms. This
// must be byte-identical to the trainer's copy (parity-critical, closed-loop FP-sensitive).
static inline void l2b_write_geometry(float* w, const float* L2, const float* feat192, float scale, double pdnorm) {
    int mode = g_wgeom;
    if (mode==WG_NONE) return;
    if (mode==WG_BLEND75 || mode==WG_BLEND50) {
        // H0 (absolute) write = projection of the post-boundary features, no delta subtraction
        float ph[L2_DIM];
        for (int j=0;j<L2_DIM;j++){ float p=0; const float* pj=Pmat[j]; for(int k=0;k<BASE_DIM;k++) p+=pj[k]*feat192[k]; ph[j]=p*scale; }
        float a=(mode==WG_BLEND75)?0.75f:0.5f, b=1.0f-a;
        double bl2=0.0; float bl[L2_DIM];
        for (int j=0;j<L2_DIM;j++){ bl[j]=a*w[j]+b*ph[j]; bl2+=(double)bl[j]*bl[j]; }
        double bln=sqrt(bl2);
        float s=(bln>1e-12)?(float)(pdnorm/bln):1.0f;   // direction blend, delta-write energy preserved
        for (int j=0;j<L2_DIM;j++) w[j]=bl[j]*s;
    } else if (mode==WG_ORTHO || mode==WG_NOVELTY) {
        double l2n2=0.0; for(int j=0;j<L2_DIM;j++) l2n2+=(double)L2[j]*L2[j];
        if (l2n2>(double)L2_RELMOVE_EPS*L2_RELMOVE_EPS){   // only orthogonalize once a state exists
            double dot=0.0; for(int j=0;j<L2_DIM;j++) dot+=(double)w[j]*L2[j];
            float coef=(float)(dot/l2n2);                  // remove the component parallel to L2
            for (int j=0;j<L2_DIM;j++) w[j]-=coef*L2[j];
            if (mode==WG_NOVELTY){
                double wn2=0.0; for(int j=0;j<L2_DIM;j++) wn2+=(double)w[j]*w[j];
                double wn=sqrt(wn2);
                float s=(wn>1e-12)?(float)(pdnorm/wn):1.0f; // novelty direction at full delta energy
                for (int j=0;j<L2_DIM;j++) w[j]*=s;
            }
        }
    } else if (mode==WG_ENERGY) {
        float s=(pdnorm>1e-12)?(float)(1.0/pdnorm):1.0f;    // unit-norm write (equalize per-event energy)
        for (int j=0;j<L2_DIM;j++) w[j]*=s;
    }
}

// Phase 45.C write-gate: decide whether to ACCEPT this boundary's EMA write or DROP it.
// Pure function of the candidate write w, current state L2, alpha, and a small persistent
// state (cooldown / thinning counters, by pointer). Scale-free INTERNAL signals only:
//   cosWL2  = cos(w, L2_old)              (reversal angle; <0 = anti-aligned correction)
//   relmove = ||cand-L2|| / max(||L2||,EPS)  (45.0 discriminant; reversal-strength proxy)
// Bootstrap (||L2||<=EPS or ||w||~0): always accept (no state to measure an angle against);
// WT_THIN still advances its counter so its phase stays deterministic. This MUST be
// byte-identical to the trainer's copy (parity-critical, closed-loop FP-sensitive).
static inline int wtgate_accept(const float* w, const float* L2, float alpha,
                                int mode, float cos_thr, float stress_thr,
                                int cool_k, int thin_n, int* cool, int* thin) {
    if (mode==WT_NONE) return 1;
    double l2n2=0.0, wn2=0.0, dot=0.0;
    for (int j=0;j<L2_DIM;j++){ l2n2+=(double)L2[j]*L2[j]; wn2+=(double)w[j]*w[j]; dot+=(double)w[j]*L2[j]; }
    if (l2n2 <= (double)L2_RELMOVE_EPS*L2_RELMOVE_EPS || wn2<=1e-24){
        if (mode==WT_THIN){ int a=(((*thin)%thin_n)==0)?1:0; (*thin)++; return a; }
        return 1;   // cold start: admit the bootstrap writes that build the state
    }
    double cosWL2 = dot/(sqrt(wn2)*sqrt(l2n2));
    double dn2=0.0;
    for (int j=0;j<L2_DIM;j++){ double cand=(double)alpha*L2[j]+(1.0-(double)alpha)*w[j]; double dd=cand-(double)L2[j]; dn2+=dd*dd; }
    double denom=(sqrt(l2n2)>(double)L2_RELMOVE_EPS)?sqrt(l2n2):(double)L2_RELMOVE_EPS;
    double relmove=sqrt(dn2)/denom;
    switch (mode){
        case WT_REV25:     return (cosWL2 < (double)cos_thr) ? 1 : 0;
        case WT_REV50:     return (cosWL2 < (double)cos_thr) ? 1 : 0;
        case WT_REVMARGIN: return (cosWL2 < (double)cos_thr && relmove > (double)stress_thr) ? 1 : 0;
        case WT_COOLDOWN:  if (*cool > 0){ (*cool)--; return 0; }
                           if (cosWL2 < (double)cos_thr) *cool = cool_k;
                           return 1;
        case WT_THIN:      { int a=(((*thin)%thin_n)==0)?1:0; (*thin)++; return a; }
    }
    return 1;
}

// Evolve L2 by one observed byte (post-observe SEE summary feat192), with the
// gate decision and homeostasis brakes (cooldown, non-boundary decay, delta-write).
static inline void l2_evolve(float* L2, const float* feat192, int gate_fires) {
    int do_update = gate_fires && (cd_ctr==0);
    g_l2_gatefired = gate_fires; g_l2_wrote = do_update; g_l2_wnorm = 0.0;   // 45.0 write diag
    if (do_update) {
        const float* src=feat192;
        static float blend[BASE_DIM];
        if (g_mix>0.0f){ for(int k=0;k<BASE_DIM;k++) blend[k]=feat192[k]-g_mix*prev_bound[k]; src=blend; memcpy(prev_bound,feat192,BASE_DIM*sizeof(float)); }
        float scale = 1.0f / sqrtf((float)BASE_DIM);
        // delta write w = projection of the delta src into L2 space (the base write).
        float w[L2_DIM]; double pd2=0.0;
        for (int j=0;j<L2_DIM;j++){ float p=0; const float* pj=Pmat[j]; for (int k=0;k<BASE_DIM;k++) p+=pj[k]*src[k]; p*=scale; w[j]=p; pd2+=(double)p*p; }
        double pdnorm=sqrt(pd2);   // ||delta_write|| (target energy for the geometry blends)
        l2b_write_geometry(w, L2, feat192, scale, pdnorm);   // 45.B: transform w in place (WG_NONE = no-op)
        // effective write norm after geometry, for the 45.0 write diagnostics
        double wn2=0.0; for(int j=0;j<L2_DIM;j++) wn2+=(double)w[j]*w[j]; g_l2_wnorm=sqrt(wn2);
        // cosine diagnostics: write vs current L2 state, and write vs previous write
        { double wnn=g_l2_wnorm, l2n2=0.0, dwl=0.0; for(int j=0;j<L2_DIM;j++){ l2n2+=(double)L2[j]*L2[j]; dwl+=(double)w[j]*L2[j]; }
          double l2n=sqrt(l2n2); g_l2_cos_wL2=(wnn>1e-12&&l2n>1e-12)?dwl/(wnn*l2n):0.0;
          if (g_have_lastwrite){ double pn2=0.0,dwp=0.0; for(int j=0;j<L2_DIM;j++){ pn2+=(double)g_lastwrite[j]*g_lastwrite[j]; dwp+=(double)w[j]*g_lastwrite[j]; }
              double pn=sqrt(pn2); g_l2_cos_wprev=(wnn>1e-12&&pn>1e-12)?dwp/(wnn*pn):0.0; } else g_l2_cos_wprev=0.0; }
        // 45.C write-gate (timing): admit or DROP this boundary's write on internal signals
        // (reversal angle / rel_move). WT_NONE -> always accept (bit-identical to 45.B).
        int accept = wtgate_accept(w, L2, g_alpha, g_wt_mode, g_wt_cos_thr, g_wt_stress_thr,
                                   g_wt_cool_k, g_wt_thin_n, &g_wt_cool, &g_wt_thin);
        if (accept) {
            // candidate EMA update (no amplitude cap in 45.B/C; the 45.A rel-move cap stays
            // available for back-compat with relmove weights, off when g_relmove_cap<=0)
            float cand[L2_DIM];
            for (int j=0;j<L2_DIM;j++) cand[j]=g_alpha*L2[j]+(1.0f-g_alpha)*w[j];
            if (g_relmove_cap>0.0f){
                double dn=0.0, on=0.0;
                for (int j=0;j<L2_DIM;j++){ double dd=(double)cand[j]-L2[j]; dn+=dd*dd; on+=(double)L2[j]*L2[j]; }
                dn=sqrt(dn); on=sqrt(on);
                double denom=(on>L2_RELMOVE_EPS)?on:(double)L2_RELMOVE_EPS;
                double rel=dn/denom;
                if (rel>g_relmove_cap && dn>0.0){ float s=(float)(g_relmove_cap/rel); for(int j=0;j<L2_DIM;j++) L2[j]=L2[j]+s*(cand[j]-L2[j]); }
                else { for(int j=0;j<L2_DIM;j++) L2[j]=cand[j]; }
            } else {
                for(int j=0;j<L2_DIM;j++) L2[j]=cand[j];
            }
            memcpy(g_lastwrite,w,sizeof(w)); g_have_lastwrite=1;
            cd_ctr=g_cooldown;
        } else {
            g_l2_wrote=0;   // dropped by the write-gate: L2 unchanged, no cooldown set
        }
    } else {
        if (!gate_fires && g_nb_decay<1.0f) for(int j=0;j<L2_DIM;j++) L2[j]*=g_nb_decay;
        if (cd_ctr>0) cd_ctr--;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,"Usage: %s <data> <weights_0x5345453C> [opts]\n",argv[0]);
        fprintf(stderr,"  --gen-len N  --temp F  --warmup N  --seed-start N  --rng-seed N  --mode argmax\n");
        return 1;
    }
    int gen_len=2000, warmup=5000, seed_start=0, argmax=0, rng_seed_time=0;
    float temp=0.65f;
    uint64_t rng_seed=0x243F6A8885A308D3ULL;
    for (int i=3;i<argc;i++){
        if      (!strcmp(argv[i],"--gen-len")   && i+1<argc) gen_len=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--temp")      && i+1<argc) temp=(float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--warmup")    && i+1<argc) warmup=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--seed-start")&& i+1<argc) seed_start=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--rng-seed")  && i+1<argc){ const char* v=argv[++i]; if(!strcmp(v,"-1")) rng_seed_time=1; else rng_seed=strtoull(v,NULL,0); }
        else if (!strcmp(argv[i],"--mode")      && i+1<argc) argmax=!strcmp(argv[++i],"argmax");
        else if (!strcmp(argv[i],"--telemetry") && i+1<argc) g_tel=fopen(argv[++i],"w");
        else if (!strcmp(argv[i],"--warn-bigram")&& i+1<argc) g_warn_bigram=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--ablate")    && i+1<argc){ const char* v=argv[++i];
            if(!strcmp(v,"l2zero")) g_ablate=ABL_L2ZERO; else if(!strcmp(v,"l2freeze")) g_ablate=ABL_L2FREEZE;
            else if(!strcmp(v,"l2reset")) g_ablate=ABL_L2RESET; else if(!strcmp(v,"cap")) g_ablate=ABL_CAP; else g_ablate=ABL_NONE; }
        else if (!strcmp(argv[i],"--ablate-cap")&& i+1<argc) g_ablate_cap=(float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--relmove-cap")&& i+1<argc) g_relmove_cap_override=(float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--wgeom")     && i+1<argc) g_wgeom_override=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--wtgate")    && i+1<argc) g_wt_mode_override=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--wt-cos")    && i+1<argc) g_wt_cos_thr=(float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--wt-stress") && i+1<argc) g_wt_stress_thr=(float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--wt-cool")   && i+1<argc) g_wt_cool_k=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--wt-thin")   && i+1<argc){ g_wt_thin_n=atoi(argv[++i]); if(g_wt_thin_n<1)g_wt_thin_n=1; }
        else if (!strcmp(argv[i],"--forensic")  && i+1<argc) g_forensic=fopen(argv[++i],"w");
        else if (!strcmp(argv[i],"--fatigue")   && i+1<argc){ const char* v=argv[++i];
            if(!strcmp(v,"flip")) g_fat_mode=FAT_FLIP; else if(!strcmp(v,"lowmargin")) g_fat_mode=FAT_LOWMARGIN;
            else if(!strcmp(v,"state")) g_fat_mode=FAT_STATE; else g_fat_mode=FAT_NONE; }
        else if (!strcmp(argv[i],"--fat-window")  && i+1<argc) g_fat_window=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--fat-margin")  && i+1<argc) g_fat_margin=(float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--fat-trust")   && i+1<argc) g_fat_trust=(float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--fat-cap")     && i+1<argc) g_fat_cap=(float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--fat-gammalow")&& i+1<argc) g_fat_gammalow=(float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--fat-damp")    && i+1<argc) g_fat_damp=(float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--fat-stress")  && i+1<argc) g_fat_stress=atoi(argv[++i]);
    }

    FILE* fd=fopen(argv[1],"rb"); if(!fd){fprintf(stderr,"Cannot open %s\n",argv[1]);return 1;}
    fseek(fd,0,SEEK_END); long fsz=ftell(fd); fseek(fd,0,SEEK_SET);
    long total_needed=(long)seed_start+warmup+512;
    if (total_needed>fsz){ total_needed=fsz; warmup=(int)(fsz-seed_start-512); if(warmup<0)warmup=0; }
    uint8_t* file_data=malloc(total_needed); fread(file_data,1,total_needed,fd); fclose(fd);

    FILE* fw=fopen(argv[2],"rb"); if(!fw){fprintf(stderr,"Cannot open weights %s\n",argv[2]);return 1;}
    uint32_t magic; fread(&magic,4,1,fw); rewind(fw);
    if (magic!=0x5345453C && magic!=0x5345453D && magic!=0x5345453E && magic!=0x5345453F && magic!=0x53454540 && magic!=0x53454541 && magic!=0x53454542 && magic!=0x53454543){ fprintf(stderr,"Expected magic 0x5345453C..0x53454543, got 0x%08x\n",magic); fclose(fw); return 1; }
    int has_homeo = (magic>=0x5345453D);
    int has_calib = (magic>=0x5345453E);   // 44.C: extra f mix after homeostasis block
    int has_scale = (magic>=0x5345453F);   // 44.D: extra f l2_scale after mix
    int has_cap   = (magic>=0x53454540);   // 44.E: extra f l2_cap after l2_scale
    int has_relmove = (magic>=0x53454541);   // 45.A: extra f relmove_cap after l2_cap
    int has_wgeom   = (magic>=0x53454542);   // 45.B: extra u32 wgeom after relmove_cap
    int has_wtgate  = (magic>=0x53454543);   // 45.C: extra {u32 mode, f cos, f stress, u32 cool_k, u32 thin_n}

    uint32_t hdr4[4]; fread(hdr4,4,4,fw);
    float hf[5]; fread(hf,4,5,fw);
    float decay=hf[0], alpha_fast=hf[2], feat_clamp=hf[3];
    uint32_t no=0; fread(&no,4,1,fw); int n_oja=(int)no;
    if (n_oja<0||n_oja>SEE_N_OJA_MAX){ fprintf(stderr,"bad n_oja %d\n",n_oja); return 1; }
    float W_oja_buf[SEE_N_OJA_MAX*43];
    fread(W_oja_buf,sizeof(float),(size_t)n_oja*43,fw);
    uint32_t l2d=0,gt=0,eh=0,ps=0; float al=0,st=0,et=0;
    fread(&l2d,4,1,fw); fread(&gt,4,1,fw); fread(&al,4,1,fw);
    fread(&st,4,1,fw); fread(&et,4,1,fw); fread(&eh,4,1,fw); fread(&ps,4,1,fw);
    if ((int)l2d != L2_DIM){ fprintf(stderr,"L2_DIM mismatch: file %u, binary %d\n",l2d,L2_DIM); return 1; }
    g_gate=(int)gt; g_alpha=al; g_surp_thr=st; g_ent_thr=et; g_ent_high=(int)eh;
    g_l2_clamp=feat_clamp; g_nb_decay=1.0f; g_cooldown=0; g_delta=0; g_mix=0.0f; g_l2_scale=1.0f;   // 0x5345453C defaults
    if (has_homeo) {
        float l2c=0,nbd=1.0f; uint32_t cd=0,dl=0;
        fread(&l2c,4,1,fw); fread(&nbd,4,1,fw); fread(&cd,4,1,fw); fread(&dl,4,1,fw);
        g_l2_clamp=(l2c>0.0f)?l2c:feat_clamp; g_nb_decay=nbd; g_cooldown=(int)cd; g_delta=(int)dl;
        g_mix = g_delta ? 1.0f : 0.0f;   // 0x5345453D: delta flag -> mix
    }
    if (has_calib) {
        float mx=0.0f; fread(&mx,4,1,fw); g_mix=mx;   // 0x5345453E: explicit mix overrides
    }
    if (has_scale) {
        float ls=1.0f; fread(&ls,4,1,fw); g_l2_scale=(ls>0.0f)?ls:1.0f;   // 0x5345453F: L2 block scale
    }
    if (has_cap) {
        float cp=0.0f; fread(&cp,4,1,fw); g_l2_cap=cp;   // 0x53454540: L2 logit-contribution cap
    }
    if (has_relmove) {
        float rc=0.0f; fread(&rc,4,1,fw); g_relmove_cap=rc;   // 0x53454541: L2 relative-move cap
    }
    if (g_relmove_cap_override > -1.5f) g_relmove_cap=g_relmove_cap_override;   // diagnostic override (breaks header parity by design)
    if (has_wgeom) {
        uint32_t wg=0; fread(&wg,4,1,fw); g_wgeom=(int)wg;   // 0x53454542: L2 write geometry mode
    }
    if (g_wgeom_override >= 0) g_wgeom=g_wgeom_override;   // diagnostic override
    if (has_wtgate) {                                       // 0x53454543: L2 write-gate (timing)
        uint32_t wtm=0,ck=3,tn=2; float ct=0.0f,st45=1.0f;
        fread(&wtm,4,1,fw); fread(&ct,4,1,fw); fread(&st45,4,1,fw); fread(&ck,4,1,fw); fread(&tn,4,1,fw);
        g_wt_mode=(int)wtm; g_wt_cos_thr=ct; g_wt_stress_thr=st45; g_wt_cool_k=(int)ck; g_wt_thin_n=(int)tn;
        if (g_wt_thin_n<1) g_wt_thin_n=1;
    }
    if (g_wt_mode_override >= 0) g_wt_mode=g_wt_mode_override;   // diagnostic override

    size_t tri_n=(size_t)CLASSES*CLASSES*CLASSES;
    trigram=malloc(tri_n*sizeof(float));
    fread(trigram,sizeof(float),tri_n,fw);
    fread(feat_mean,sizeof(float),TOT_DIM,fw);
    fread(feat_std, sizeof(float),TOT_DIM,fw);
    fread(Wm,sizeof(float),CLASSES*TOT_DIM,fw);
    fread(Bv,sizeof(float),CLASSES,fw);
    fclose(fw);

    gen_projection(ps);
    // entropy table from trigram (for entropy gate)
    ent_table=malloc(CLASSES*CLASSES*sizeof(float));
    for(int i=0;i<CLASSES;i++) for(int j=0;j<CLASSES;j++){
        float mx=-1e9f; for(int k=0;k<CLASSES;k++) if(trigram[i][j][k]>mx) mx=trigram[i][j][k];
        double se=0; for(int k=0;k<CLASSES;k++) se+=exp((double)(trigram[i][j][k]-mx));
        double H=0; for(int k=0;k<CLASSES;k++){ double p=exp((double)(trigram[i][j][k]-mx))/se; if(p>1e-12) H-=p*log(p); }
        ent_table[i][j]=(float)H;
    }

    SiliconEntropyState see;
    see_init(&see, 42, 4, decay);
    see.multiscale_mode=1; see.alpha_fast=alpha_fast; see.alpha_mid=0.9f; see.alpha_slow=0.99f;
    see.n_oja=n_oja; memcpy(see.W_oja,W_oja_buf,(size_t)n_oja*43*sizeof(float));
    see.eta_oja=0.0f; see.plastic_blend=1.0f;
    const char* gname[]={"none","punct","whitespace","surprise","entropy","combined"};
    fprintf(stderr,"Format 0x%08x  n_oja=%d  L2_DIM=%d  gate=%s  alpha=%.2f  surp_thr=%.3f  ent_thr=%.3f(high=%d)\n",
            magic,n_oja,L2_DIM,gname[g_gate],g_alpha,g_surp_thr,g_ent_thr,g_ent_high);
    const char* wgn[]={"none","blend75","blend50","ortho","novelty","energy"};
    const char* wtn[]={"none","rev25","rev50","revmargin","cooldown","thin"};
    fprintf(stderr,"Homeostasis: l2_clamp=%.2f nb_decay=%.3f cooldown=%d delta=%d mix=%.2f l2_scale=%.2f l2_cap=%.3f relmove_cap=%.3f wgeom=%s\n",
            g_l2_clamp,g_nb_decay,g_cooldown,g_delta,g_mix,g_l2_scale,g_l2_cap,g_relmove_cap,
            (g_wgeom>=0&&g_wgeom<=5)?wgn[g_wgeom]:"?");
    fprintf(stderr,"WriteGate: mode=%s cos_thr=%.2f stress_thr=%.2f cool_k=%d thin_n=%d\n",
            (g_wt_mode>=0&&g_wt_mode<=5)?wtn[g_wt_mode]:"?",g_wt_cos_thr,g_wt_stress_thr,g_wt_cool_k,g_wt_thin_n);
    fprintf(stderr,"Warmup %d | Seed 512 | Gen %d | Temp %.2f | base_clamp %.2f\n",warmup,gen_len,temp,feat_clamp);
    memset(prev_bound,0,sizeof(prev_bound)); cd_ctr=0;
    g_wt_cool=0; g_wt_thin=0;   // 45.C write-gate state (persists warmup->seed->generate)

    float feat192[BASE_DIM], fa[BASE_DIM], raw[TOT_DIM];
    float L2[L2_DIM]; memset(L2,0,sizeof(L2));
    uint8_t ctx1=0, ctx2=0;

    // ---- Warmup: online stats over [SEE|L2], 256D ----
    double omean[TOT_DIM], oM2[TOT_DIM]; memset(omean,0,sizeof(omean)); memset(oM2,0,sizeof(oM2));
    long on=0; int cold_skip=64;
    see_reset(&see);
    for (int i=seed_start; i<seed_start+warmup; i++){
        uint8_t b=file_data[i];
        int gate=eval_gate(b, ctx1, ctx2);
        see_observe(&see, b);
        see_extract(&see, feat192);
        l2_evolve(L2, feat192, gate);
        ctx2=ctx1; ctx1=b;
        if (i-seed_start >= cold_skip){
            memcpy(raw, feat192, BASE_DIM*sizeof(float));
            memcpy(raw+BASE_DIM, L2, L2_DIM*sizeof(float));
            on++;
            for (int f=0;f<TOT_DIM;f++){ double d=raw[f]-omean[f]; omean[f]+=d/on; oM2[f]+=d*(raw[f]-omean[f]); }
        }
    }
    for (int f=0;f<TOT_DIM;f++){ feat_mean[f]=(float)omean[f]; feat_std[f]=(on>1)?(float)sqrt(oM2[f]/(on-1))+1e-8f:1.0f; }
    fprintf(stderr,"Online stats from %ld samples\n", on);

    // ---- Seed: condition SEE + L2 on 512 bytes (no reset) ----
    int seed_offset=seed_start+warmup;
    int seed_len=(int)(total_needed-seed_offset); if(seed_len>512) seed_len=512;
    for (int i=0;i<seed_len;i++){
        uint8_t b=file_data[seed_offset+i];
        int gate=eval_gate(b, ctx1, ctx2);
        see_observe(&see, b);
        see_extract(&see, feat192);
        l2_evolve(L2, feat192, gate);
        ctx2=ctx1; ctx1=b;
    }
    fprintf(stderr,"--- generated ---\n");

    // ---- Generate ----
    uint64_t rng = rng_seed_time
        ? ((uint64_t)time(NULL) ^ 0xdeadbeefcafeULL ^ (uint64_t)seed_start)
        : (rng_seed ^ ((uint64_t)seed_start * 0x9E3779B97F4A7C15ULL));
    if (rng==0) rng=0x9E3779B97F4A7C15ULL;
    fprintf(stderr,"RNG: %s seed=0x%016llx\n", rng_seed_time?"time":"fixed",(unsigned long long)rng);

    if (g_tel){
        fprintf(g_tel,"# cap=%.4f mix=%.2f l2_scale=%.2f temp=%.2f seed_start=%d rng=0x%016llx\n",
                g_l2_cap,g_mix,g_l2_scale,temp,seed_start,(unsigned long long)rng);
        fprintf(g_tel,"step\tsamp\tpredF\tpredN\tflip\tratio\tgamma\tfired\tl2norm\tl2delta\tgate\tLwin\tloop_run\tloop_p\twnorm\twrote\tgfired\tcos_wL2\tcos_wprev\n");
        memset(g_runp,0,sizeof(g_runp)); g_have_prevL2=0;
        g_l2_wnorm=0.0; g_l2_wrote=0; g_l2_gatefired=0;
        g_l2_cos_wL2=0.0; g_l2_cos_wprev=0.0;
    }
    if (g_forensic){
        const char* abn[]={"none","l2zero","l2freeze","l2reset","cap"};
        fprintf(g_forensic,"# cap=%.4f mix=%.2f l2_scale=%.2f temp=%.2f seed_start=%d warn_bigram=%d ablate=%s ablate_cap=%.4f\n",
                g_l2_cap,g_mix,g_l2_scale,temp,seed_start,g_warn_bigram,abn[g_ablate],g_ablate_cap);
        fprintf(g_forensic,"step\tbyte\tbigram\tbicnt\ttopbicnt\tseeTop\tseeLgt\tl2Top\tl2Lgt\targF\tratio\tgamma\tflip\tmargin\tgate\twarned\n");
    }

    double self_bits=0; uint8_t* generated=malloc(gen_len); float Pp[CLASSES];
    int fat_timer=0, stress_ctr=0;   // 44.I endogenous fatigue state
    for (int i=0;i<gen_len;i++){
        see_extract(&see, feat192);
        memcpy(raw, feat192, BASE_DIM*sizeof(float));
        memcpy(raw+BASE_DIM, L2, L2_DIM*sizeof(float));
        for (int f=0;f<TOT_DIM;f++){
            float x=(raw[f]-feat_mean[f])/(feat_std[f]+1e-8f);
            float cl=(f<BASE_DIM)?feat_clamp:g_l2_clamp;   // L2 dims use homeostatic clamp
            if (cl>0.0f){ if(x>cl)x=cl; if(x<-cl)x=-cl; }
            if (f>=BASE_DIM) x*=g_l2_scale;                // 44.D: L2 block scale (post-clamp)
            raw[f]=x;
        }
        const float* tri=&trigram[0][0][0] + ((size_t)ctx2*CLASSES+ctx1)*CLASSES;
        // Split SEE vs L2 logit contributions for the conditional-trust cap.
        float svec[CLASSES], dvec[CLASSES];
        for (int c=0;c<CLASSES;c++){ svec[c]=tri[c]+Bv[c]+dot_avx(Wm[c],raw,BASE_DIM); dvec[c]=dot_avx(Wm[c]+BASE_DIM,raw+BASE_DIM,L2_DIM); }
        float gamma=1.0f;
        float capnow = (g_warned && g_ablate==ABL_CAP) ? g_ablate_cap : g_l2_cap;  // 44.H: tighten cap after warning
        double l2ratio=0.0; int have_ratio=0;
        if (capnow!=0.0f || g_fat_mode==FAT_LOWMARGIN){   // need the raw L2/SEE ratio
            double ms=0,md=0; for(int c=0;c<CLASSES;c++){ms+=svec[c];md+=dvec[c];} ms/=CLASSES; md/=CLASSES;
            double vs=0,vd=0; for(int c=0;c<CLASSES;c++){double a=svec[c]-ms,b=dvec[c]-md; vs+=a*a; vd+=b*b;}
            double ns=sqrt(vs/CLASSES), nd=sqrt(vd/CLASSES);
            if (nd>1e-12 && ns>1e-12){ l2ratio=nd/ns; have_ratio=1; }
        }
        if (capnow!=0.0f && have_ratio){   // cap>0 hard, cap<0 soft (tanh) - identical to trainer l2_cap_gamma
            if (capnow<0.0f){ double capv=-(double)capnow; double eff=capv*tanh(l2ratio/capv); gamma=(float)(eff/l2ratio); }
            else if (l2ratio>capnow){ gamma=(float)(capnow/l2ratio); }
        }
        if (g_warned && g_ablate==ABL_L2ZERO) gamma=0.0f;   // 44.H: zero L2 logit contribution after warning
        float base_gamma=gamma;   // 44.I: gamma after cap, before endogenous fatigue
        if (g_fat_mode==FAT_FLIP && fat_timer>0) gamma*=g_fat_trust;   // reduced L2 trust during fatigue
        else if (g_fat_mode==FAT_LOWMARGIN && fat_timer>0 && have_ratio && l2ratio>g_fat_cap){ float gt=(float)(g_fat_cap/l2ratio); if(gt<gamma) gamma=gt; }
        float logits[CLASSES], mx=-1e30f;
        for (int c=0;c<CLASSES;c++){ logits[c]=(svec[c]+gamma*dvec[c])/temp; if(logits[c]>mx)mx=logits[c]; }
        float Z=0; for(int c=0;c<CLASSES;c++){ Pp[c]=expf(logits[c]-mx); Z+=Pp[c]; }
        for(int c=0;c<CLASSES;c++) Pp[c]/=Z;
        uint8_t next;
        if (argmax){ float bst=-1; next=0; for(int c=0;c<CLASSES;c++) if(Pp[c]>bst){bst=Pp[c];next=(uint8_t)c;} }
        else next=sample_from(Pp,&rng);
        self_bits += -log2((double)fmaxf(Pp[next],1e-30f));
        generated[i]=next;

        if (g_warn_bigram>0 || g_forensic) track_byte(next, i);   // 44.H: word/bigram tracking + warning
        if (g_warned && g_ablate==ABL_L2RESET && !g_reset_done){ memset(L2,0,sizeof(L2)); g_reset_done=1; }  // 44.H: one-time L2 reset

        // 44.I endogenous fatigue: triggers from internal signals (disagreement, low
        // margin, gamma saturation, state stress). No word counters in the policy.
        if (g_fat_mode!=FAT_NONE){
            int af=0,as=0; float b1=-1e30f,b2=-1e30f,bs=-1e30f;
            for(int c=0;c<CLASSES;c++){ float pv=Pp[c]; if(pv>b1){b2=b1;b1=pv;af=c;} else if(pv>b2) b2=pv; if(svec[c]>bs){bs=svec[c];as=c;} }
            float pm=b1-b2; int flip=(af!=as);   // prob-margin and L2/SEE disagreement
            if (g_fat_mode==FAT_FLIP){ if(flip && pm<g_fat_margin) fat_timer=g_fat_window; else if(fat_timer>0) fat_timer--; }
            else if (g_fat_mode==FAT_LOWMARGIN){ if(base_gamma<g_fat_gammalow && pm<g_fat_margin) fat_timer=g_fat_window; else if(fat_timer>0) fat_timer--; }
            else if (g_fat_mode==FAT_STATE){ if(flip && pm<g_fat_margin) stress_ctr++; else if(stress_ctr>0) stress_ctr--;
                if(stress_ctr>=g_fat_stress){ for(int j=0;j<L2_DIM;j++) L2[j]*=g_fat_damp; stress_ctr=0; } }   // damp L2 state, not zero
        }

        int gate=eval_gate(next, ctx1, ctx2);

        if (g_forensic){
            double ms=0,md=0; for(int c=0;c<CLASSES;c++){ms+=svec[c];md+=dvec[c];} ms/=CLASSES; md/=CLASSES;
            double vs=0,vd=0; for(int c=0;c<CLASSES;c++){double a=svec[c]-ms,b=dvec[c]-md; vs+=a*a; vd+=b*b;}
            double ns=sqrt(vs/CLASSES), nd=sqrt(vd/CLASSES); double ratio=(ns>1e-12)?nd/ns:0.0;
            int st=0,lt=0,af=0; float sb=-1e30f,lb=-1e30f;
            for(int c=0;c<CLASSES;c++){ if(svec[c]>sb){sb=svec[c];st=c;} if(dvec[c]>lb){lb=dvec[c];lt=c;} }
            float t1=-1e30f,t2=-1e30f; for(int c=0;c<CLASSES;c++){ float v=logits[c]; if(v>t1){t2=t1;t1=v;af=c;} else if(v>t2)t2=v; }
            fprintf(g_forensic,"%d\t%d\t%s\t%d\t%d\t%d\t%.3f\t%d\t%.3f\t%d\t%.4f\t%.4f\t%d\t%.4f\t%d\t%d\n",
                    i,(int)next,(g_curbi[0]?g_curbi:"-"),g_curbi_cnt,g_topbi_cnt,
                    st,(double)svec[st],lt,(double)dvec[lt],af,ratio,(double)gamma,(af!=st)?1:0,(double)(t1-t2),gate,g_warned?1:0);
        }

        if (g_tel){
            // byte-cycle loop detector: run length of periodicity at each period p
            for (int p=1;p<=TEL_PMAX;p++){ if (i>=p && generated[i]==generated[i-p]) g_runp[p]++; else g_runp[p]=0; }
            int lrun=0, lp=0; for(int p=1;p<=TEL_PMAX;p++) if(g_runp[p]>lrun){lrun=g_runp[p];lp=p;}
            // raw L2/SEE logit-norm ratio (pre-cap), and argmax with/without L2
            double ms=0,md=0; for(int c=0;c<CLASSES;c++){ms+=svec[c];md+=dvec[c];} ms/=CLASSES; md/=CLASSES;
            double vs=0,vd=0; for(int c=0;c<CLASSES;c++){double a=svec[c]-ms,b=dvec[c]-md; vs+=a*a; vd+=b*b;}
            double ns=sqrt(vs/CLASSES), nd=sqrt(vd/CLASSES); double ratio=(ns>1e-12)?nd/ns:0.0;
            int pf=0,pn=0; float bf=-1e30f,bn=-1e30f;
            for(int c=0;c<CLASSES;c++){ if(logits[c]>bf){bf=logits[c];pf=c;} if(svec[c]>bn){bn=svec[c];pn=c;} }
            // L2 memory-state norm and its step-to-step change
            double l2n=0; for(int j=0;j<L2_DIM;j++) l2n+=(double)L2[j]*L2[j]; l2n=sqrt(l2n);
            double l2d=0; if(g_have_prevL2){ for(int j=0;j<L2_DIM;j++){ double dd=(double)L2[j]-g_prevL2[j]; l2d+=dd*dd; } l2d=sqrt(l2d); }
            for(int j=0;j<L2_DIM;j++) g_prevL2[j]=L2[j]; g_have_prevL2=1;
            float Lwin=gamma*dvec[next];   // L2's signed contribution to the emitted byte's logit
            // wnorm/wrote/gfired/cos_* describe the SAME evolve that produced l2delta
            // (col 10), i.e. the previous step's end-of-loop l2_evolve - aligned for
            // event-conditioning. cos_wL2/cos_wprev are the 45.B write-geometry diagnostics.
            fprintf(g_tel,"%d\t%d\t%d\t%d\t%d\t%.4f\t%.4f\t%d\t%.4f\t%.4f\t%d\t%.4f\t%d\t%d\t%.4f\t%d\t%d\t%.4f\t%.4f\n",
                    i,(int)next,pf,pn,(pf!=pn)?1:0,ratio,gamma,(gamma<0.999999f)?1:0,l2n,l2d,gate,Lwin,lrun,lp,
                    g_l2_wnorm,g_l2_wrote,g_l2_gatefired,g_l2_cos_wL2,g_l2_cos_wprev);
        }

        see_observe(&see, next);
        see_extract(&see, fa);
        if (!(g_warned && g_ablate==ABL_L2FREEZE)) l2_evolve(L2, fa, gate);   // 44.H: freeze L2 state after warning
        ctx2=ctx1; ctx1=next;

        if ((i+1)%100==0) fprintf(stderr,"bpb_at_%04d: %.4f\n", i+1, self_bits/(i+1));
    }
    fwrite(generated,1,gen_len,stdout);
    fprintf(stderr,"\n--- stats ---\nself_BPB: %.4f\n", self_bits/gen_len);
    fprintf(stderr,"TOPBI_FINAL: %d  WARNED: %d  WARN_STEP: %d\n", g_topbi_cnt, g_warned, g_warn_step);
    if (g_tel) fclose(g_tel);
    if (g_forensic) fclose(g_forensic);
    free(trigram); free(ent_table); free(file_data); free(generated);
    return 0;
}
