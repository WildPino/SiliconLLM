#include "see_codec.h"
#include "lz_topn.h"
#include "tok_lz.h"
#include "span_lz.h"
#include "moe_engine.h"
#include "silicon_entropy.h"
#include "range_coder.h"
#include "regime_prior.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include <immintrin.h>
#ifdef _MSC_VER
#  include <intrin.h>
#else
#  include <x86intrin.h>
#endif

// ── Constants ─────────────────────────────────────────────────────────────────
#define CLASSES       256
#define CDF_SCALE     16384
#define CDF_SCALE_BITS 14

// ── Weights file on-disk header (unchanged from Phase 21) ────────────────────
typedef struct {
    uint32_t magic;        // 0x53454531 ("SEE1")
    uint32_t version;
    uint32_t feature_dim;
    uint32_t chunk_size;
    float    decay;
    uint32_t codebook_seed;
    float    alpha;
} WeightsFileHeader;

#define WEIGHTS_MAGIC 0x53454531u

// ── Loaded weights context ────────────────────────────────────────────────────
typedef struct {
    WeightsFileHeader hdr;
    float* trigram_logits;              // [CLASSES][CLASSES][CLASSES] flat
    float  feature_means[SEE_FEATURE_DIM];
    float  feature_stds[SEE_FEATURE_DIM];
    float  W[CLASSES][SEE_FEATURE_DIM]; // linear classifier weights
    float  B[CLASSES];                  // linear classifier biases
    float* A_proj;                      // [lr_rank * SEE_FEATURE_DIM] or NULL
    float* B_proj;                      // [CLASSES * lr_rank] or NULL
    uint32_t lr_rank;
    uint8_t* topk_indices;             // [CLASSES][CLASSES][CLASSES] flat
    float*   base_probs;               // [CLASSES][CLASSES][CLASSES] for tail_mode
    float*   Z_base;                   // [CLASSES][CLASSES] for tail_mode
} WeightsCtx;

// ── Internal mode enum ───────────────────────────────────────────────────────
typedef enum { MODE_ENCODE, MODE_DECODE, MODE_AUDIT } RunMode;

// ── SHA-256 (portable, no external lib) ──────────────────────────────────────
// Condensed public-domain SHA-256 implementation.
typedef struct { uint32_t s[8]; uint8_t buf[64]; uint64_t len; int buflen; } Sha256Ctx;

static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
#define ROR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define CH(e,f,g)  (((e)&(f))^(~(e)&(g)))
#define MAJ(a,b,c) (((a)&(b))^((a)&(c))^((b)&(c)))
#define EP0(a)     (ROR32(a,2)^ROR32(a,13)^ROR32(a,22))
#define EP1(e)     (ROR32(e,6)^ROR32(e,11)^ROR32(e,25))
#define SIG0(x)    (ROR32(x,7)^ROR32(x,18)^((x)>>3))
#define SIG1(x)    (ROR32(x,17)^ROR32(x,19)^((x)>>10))

static void sha256_block(Sha256Ctx* c) {
    uint32_t w[64], a,b,d,e,f,g,h,t1,t2;
    uint32_t* s = c->s;
    for (int i=0;i<16;i++) w[i]=((uint32_t)c->buf[i*4]<<24)|((uint32_t)c->buf[i*4+1]<<16)|((uint32_t)c->buf[i*4+2]<<8)|c->buf[i*4+3];
    for (int i=16;i<64;i++) w[i]=SIG1(w[i-2])+w[i-7]+SIG0(w[i-15])+w[i-16];
    a=s[0];b=s[1];uint32_t cc=s[2];d=s[3];e=s[4];f=s[5];g=s[6];h=s[7];
    for(int i=0;i<64;i++){t1=h+EP1(e)+CH(e,f,g)+K256[i]+w[i];t2=EP0(a)+MAJ(a,b,cc);h=g;g=f;f=e;e=d+t1;d=cc;cc=b;b=a;a=t1+t2;}
    s[0]+=a;s[1]+=b;s[2]+=cc;s[3]+=d;s[4]+=e;s[5]+=f;s[6]+=g;s[7]+=h;
}
static void sha256_init(Sha256Ctx* c){c->s[0]=0x6a09e667;c->s[1]=0xbb67ae85;c->s[2]=0x3c6ef372;c->s[3]=0xa54ff53a;c->s[4]=0x510e527f;c->s[5]=0x9b05688c;c->s[6]=0x1f83d9ab;c->s[7]=0x5be0cd19;c->len=0;c->buflen=0;}
static void sha256_feed(Sha256Ctx* c,const uint8_t* d,size_t n){for(size_t i=0;i<n;i++){c->buf[c->buflen++]=d[i];if(c->buflen==64){sha256_block(c);c->buflen=0;}c->len++;}}
static void sha256_done(Sha256Ctx* c,uint8_t out[32]){c->buf[c->buflen++]=0x80;while(c->buflen!=56){if(c->buflen==64){sha256_block(c);c->buflen=0;}else c->buf[c->buflen++]=0;}uint64_t bits=c->len*8;for(int i=7;i>=0;i--){c->buf[56+i]=(uint8_t)(bits&0xff);bits>>=8;}sha256_block(c);for(int i=0;i<8;i++){out[i*4]=(c->s[i]>>24);out[i*4+1]=(c->s[i]>>16);out[i*4+2]=(c->s[i]>>8);out[i*4+3]=c->s[i]&0xff;}}

static void sha256_file(const char* path, char hex_out[65]) {
    FILE* f = fopen(path, "rb");
    if (!f) { hex_out[0] = 0; return; }
    Sha256Ctx c; sha256_init(&c);
    uint8_t buf[65536]; size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) sha256_feed(&c, buf, n);
    fclose(f);
    uint8_t digest[32]; sha256_done(&c, digest);
    for (int i = 0; i < 32; i++) sprintf(hex_out + i*2, "%02x", digest[i]);
    hex_out[64] = 0;
}

// ── SIMD dot product ──────────────────────────────────────────────────────────
static inline float dot_avx(const float* w, const float* f, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(w+i), _mm256_loadu_ps(f+i), acc);
    float tmp[8]; _mm256_storeu_ps(tmp, acc);
    float r = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    for (; i < n; i++) r += w[i] * f[i];
    return r;
}

// ── Weights loading ───────────────────────────────────────────────────────────
static WeightsCtx* weights_load(const char* path, int tail_mode) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "see: cannot open weights: %s\n", path); return NULL; }

    WeightsCtx* wc = (WeightsCtx*)calloc(1, sizeof(WeightsCtx));
    if (!wc) { fclose(f); return NULL; }

    if (fread(&wc->hdr, sizeof(WeightsFileHeader), 1, f) != 1 ||
        wc->hdr.magic != WEIGHTS_MAGIC) {
        fprintf(stderr, "see: invalid weights magic in %s\n", path);
        goto fail;
    }
    if (wc->hdr.feature_dim != SEE_FEATURE_DIM) {
        fprintf(stderr, "see: feature_dim mismatch (weights=%u expected=%d)\n",
                wc->hdr.feature_dim, SEE_FEATURE_DIM);
        goto fail;
    }

    wc->trigram_logits = (float*)malloc((size_t)CLASSES*CLASSES*CLASSES*sizeof(float));
    if (!wc->trigram_logits) goto fail;
    fread(wc->trigram_logits, sizeof(float), (size_t)CLASSES*CLASSES*CLASSES, f);
    fread(wc->feature_means, sizeof(float), SEE_FEATURE_DIM, f);
    fread(wc->feature_stds,  sizeof(float), SEE_FEATURE_DIM, f);
    fread(wc->W,             sizeof(float), CLASSES*SEE_FEATURE_DIM, f);
    fread(wc->B,             sizeof(float), CLASSES, f);

    if (fread(&wc->lr_rank, sizeof(uint32_t), 1, f) == 1 && wc->lr_rank > 0) {
        wc->A_proj = (float*)malloc(wc->lr_rank * SEE_FEATURE_DIM * sizeof(float));
        wc->B_proj = (float*)malloc(CLASSES * wc->lr_rank * sizeof(float));
        if (!wc->A_proj || !wc->B_proj) goto fail;
        fread(wc->A_proj, sizeof(float), wc->lr_rank * SEE_FEATURE_DIM, f);
        fread(wc->B_proj, sizeof(float), CLASSES * wc->lr_rank, f);
    }
    fclose(f); f = NULL;

    // Pre-sort topk_indices[ctx2][ctx1][rank] = byte with rank-th highest logit+B
    wc->topk_indices = (uint8_t*)malloc((size_t)CLASSES*CLASSES*CLASSES);
    if (!wc->topk_indices) goto fail;
    typedef struct { uint8_t idx; float logit; } LP;
    LP* pairs = (LP*)malloc(CLASSES * sizeof(LP));
    if (!pairs) goto fail;
    for (int c2 = 0; c2 < CLASSES; c2++) {
        for (int c1 = 0; c1 < CLASSES; c1++) {
            for (int c = 0; c < CLASSES; c++) {
                pairs[c].idx   = (uint8_t)c;
                pairs[c].logit = wc->trigram_logits[(c2*CLASSES+c1)*CLASSES+c] + wc->B[c];
            }
            // insertion sort (CLASSES=256, small enough)
            for (int i = 1; i < CLASSES; i++) {
                LP key = pairs[i]; int j = i-1;
                while (j >= 0 && pairs[j].logit < key.logit) { pairs[j+1]=pairs[j]; j--; }
                pairs[j+1] = key;
            }
            for (int k = 0; k < CLASSES; k++)
                wc->topk_indices[(c2*CLASSES+c1)*CLASSES+k] = pairs[k].idx;
        }
    }
    free(pairs);

    // Pre-compute softmax base_probs for tail_mode
    if (tail_mode > 0) {
        wc->base_probs = (float*)malloc((size_t)CLASSES*CLASSES*CLASSES*sizeof(float));
        wc->Z_base     = (float*)malloc((size_t)CLASSES*CLASSES*sizeof(float));
        if (!wc->base_probs || !wc->Z_base) goto fail;
        for (int c2 = 0; c2 < CLASSES; c2++) {
            for (int c1 = 0; c1 < CLASSES; c1++) {
                float sum = 0;
                for (int c = 0; c < CLASSES; c++) {
                    float lg = wc->trigram_logits[(c2*CLASSES+c1)*CLASSES+c];
                    float p  = (tail_mode == 1) ? expf(lg) : expf(lg + wc->B[c]);
                    wc->base_probs[(c2*CLASSES+c1)*CLASSES+c] = p;
                    sum += p;
                }
                wc->Z_base[c2*CLASSES+c1] = sum;
            }
        }
    }
    return wc;

fail:
    if (f) fclose(f);
    if (wc) {
        free(wc->trigram_logits);
        free(wc->topk_indices);
        free(wc->base_probs);
        free(wc->Z_base);
        free(wc->A_proj);
        free(wc->B_proj);
        free(wc);
    }
    return NULL;
}

static void weights_free(WeightsCtx* wc) {
    if (!wc) return;
    free(wc->trigram_logits);
    free(wc->topk_indices);
    free(wc->base_probs);
    free(wc->Z_base);
    free(wc->A_proj);
    free(wc->B_proj);
    free(wc);
}

// ── Core streaming loop (shared by encode / decode / audit) ──────────────────
typedef struct {
    WeightsCtx*         wc;
    const SeeCodecConfig* cfg;
    RunMode             mode;
    uint8_t*            data;        // input bytes (encode/audit)
    size_t              data_size;
    int                 eval_start;
    int                 eval_len;
    // Range coder handles (one is non-NULL depending on mode)
    FILE*               f_enc;
    FILE*               f_dec;
    FILE*               f_dump;
    FILE*               f_tel;
    // Output
    SeeAuditResult*     result;      // filled when mode==MODE_AUDIT
} RunCtx;

static int run_loop(RunCtx* rc) {
    WeightsCtx*         wc  = rc->wc;
    const SeeCodecConfig* cfg = rc->cfg;
    int no_lz        = cfg->no_lz;
    int lz_mute      = cfg->lz_mute;
    int use_moe      = cfg->use_moe;
    int req_topk     = cfg->req_topk;
    int req_topm     = cfg->req_topm;
    int tail_mode    = cfg->tail_mode;
    int lz_key_bytes = (cfg->lz_key_bytes > 0) ? cfg->lz_key_bytes : 4;
    int lz_dual      = cfg->lz_dual;
    int tok_prefix   = cfg->tok_prefix;
    int tok_prev     = cfg->tok_prev;
    int tok_prev_mute= cfg->tok_prev_mute;
    int tok_prev_elig= cfg->tok_prev_elig;
    int span_pfx     = cfg->span_pfx;
    int span_pfx_mute= cfg->span_pfx_mute;

    // Key masks for primary LZ expert (4, 6, or 8 bytes)
    uint64_t lz_key_mask = (lz_key_bytes == 8) ? 0xFFFFFFFFFFFFFFFFULL
                         : (lz_key_bytes == 6) ? 0xFFFFFFFFFFFFULL
                         :                       0xFFFFFFFFULL; // 4 bytes

    // ── Allocate per-session state ────────────────────────────────────────────
    LzEntry* lz_table  = NULL;
    LzEntry* lz_table8 = NULL;  // second LZ table for dual mode (always 8-byte key)
    LzEntry* tok_table = NULL;  // tok-prefix expert table (reuses LZ8 MoE slot)
    LzEntry* tok_prev_table = NULL; // tok-prev expert table
    LzEntry* span_table     = NULL; // span-prefix expert table
    TokLzState tok_state;
    tok_lz_init(&tok_state);
    SpanLzState span_state;
    span_lz_init(&span_state);
    if (!no_lz && !lz_mute) {
        lz_table = lz_table_alloc();
        if (!lz_table) { fprintf(stderr, "see: OOM allocating LZ table\n"); return -1; }
        if (lz_dual) {
            lz_table8 = lz_table_alloc();
            if (!lz_table8) { lz_table_free(lz_table); fprintf(stderr, "see: OOM allocating LZ8 table\n"); return -1; }
        } else if (tok_prefix) {
            tok_table = lz_table_alloc();
            if (!tok_table) { lz_table_free(lz_table); lz_table_free(lz_table8); fprintf(stderr, "see: OOM allocating tok-prefix table\n"); return -1; }
        }
        if (tok_prev && !tok_prev_mute) {
            tok_prev_table = lz_table_alloc();
            if (!tok_prev_table) { lz_table_free(lz_table); lz_table_free(lz_table8); lz_table_free(tok_table); fprintf(stderr, "see: OOM allocating tok-prev table\n"); return -1; }
        }
        if (span_pfx && !span_pfx_mute) {
            span_table = lz_table_alloc();
            if (!span_table) { lz_table_free(lz_table); lz_table_free(lz_table8); lz_table_free(tok_table); lz_table_free(tok_prev_table); fprintf(stderr, "see: OOM allocating span table\n"); return -1; }
        }
    }

    SiliconEntropyState see_state;
    see_init(&see_state, (int)wc->hdr.codebook_seed,
             (int)wc->hdr.chunk_size, wc->hdr.decay);
    if (cfg->history_tokens > 0)
        see_state.history_tokens = cfg->history_tokens;

    // Dynamic n-gram counters
    uint32_t dyn_uni[CLASSES]          = {0};
    uint16_t dyn_bi[CLASSES][CLASSES]  = {{0}};
    uint32_t dyn_total                 = 0;

    // MoE state
    int n_active = 0;
    int abs_slots[7];
    abs_slots[n_active++] = 0; // SEE
    abs_slots[n_active++] = 1; // UNI
    abs_slots[n_active++] = 2; // BI
    if (!no_lz) {
        abs_slots[n_active++] = 3; // LZ
        if (lz_dual || tok_prefix) {
            abs_slots[n_active++] = 4; // LZ8 or TOKPFX
        }
        if (tok_prev) {
            abs_slots[n_active++] = 5; // TOK_PREV
        }
        if (span_pfx) {
            abs_slots[n_active++] = 6; // SPAN_PFX
        }
    }
    MoeState moe;
    moe_init(&moe, cfg->moe_eta, cfg->moe_share, n_active);

    RegimePriorState rp;
    int use_regime_prior = cfg->regime_prior;
    if (use_regime_prior)
        regime_prior_init(&rp, cfg->regime_prior_mute);

    RangeEncoder re; RangeDecoder rd;
    if (rc->f_enc) rc_encoder_init(&re, rc->f_enc);
    if (rc->f_dec) rc_decoder_init(&rd, rc->f_dec);

    // ── Warmup: feed bytes before eval_start ─────────────────────────────────
    uint8_t ctx2 = 0, ctx1 = 0;
    uint64_t lz_ctx = 0;  // rolling 64-bit context; callers mask to 32/48/64 bits

    if (rc->mode != MODE_DECODE) {
        // Feed up to eval_start+1 bytes into state (but not the 2 we use as ctx)
        for (int i = 0; i <= rc->eval_start + 1 && i < (int)rc->data_size; i++) {
            see_observe(&see_state, rc->data[i]);
            dyn_uni[rc->data[i]]++;
            dyn_bi[ctx1][rc->data[i]]++;
            dyn_total++;
            ctx2 = ctx1; ctx1 = rc->data[i];
        }
    } else {
        // Decode mode: seed from archive header (already written to f_dump)
        uint8_t s0, s1;
        // The caller has read seed bytes from the archive header before calling us.
        // They are stored as first bytes in data[] (borrowed storage trick).
        s0 = rc->data[0]; s1 = rc->data[1];
        // lz_ctx intentionally stays 0 — mirrors encoder warmup which never updates it
        see_observe(&see_state, s0); dyn_uni[s0]++; dyn_bi[ctx1][s0]++; dyn_total++;
        ctx2 = ctx1; ctx1 = s0;
        see_observe(&see_state, s1); dyn_uni[s1]++; dyn_bi[ctx1][s1]++; dyn_total++;
        ctx2 = ctx1; ctx1 = s1;
    }

    // ── Unigram baseline (for reporting) ─────────────────────────────────────
    double uni_loss_baseline = 0;
    if (rc->mode != MODE_DECODE) {
        double uc[CLASSES] = {0}; double utotal = 0;
        for (int i = 0; i < rc->eval_len; i++) { uc[rc->data[rc->eval_start+2+i]]++; utotal++; }
        for (int i = 0; i < rc->eval_len; i++) {
            double p = uc[rc->data[rc->eval_start+2+i]] / utotal;
            uni_loss_baseline -= log2(p > 0 ? p : 1e-10);
        }
    }

    // ── Per-expert probability arrays (stack) ─────────────────────────────────
    float p_see[CLASSES], p_uni[CLASSES], p_bi[CLASSES], p_lz[CLASSES], p_lz8[CLASSES], p_tok_prev[CLASSES], p_span[CLASSES];
    float probs[CLASSES];
    const float* all_ptrs[7] = { p_see, p_uni, p_bi, p_lz, p_lz8, p_tok_prev, p_span };
    const float* expert_ptrs[7];
    for (int i = 0; i < n_active; i++) expert_ptrs[i] = all_ptrs[abs_slots[i]];

    double total_loss = 0, total_qloss = 0;
    double expert_loss_sum[7] = {0};
    uint64_t total_cyc = 0;
    uint64_t n_elig_bytes = 0;              // tok_prev eligible steps
    uint64_t n_span_elig_bytes = 0;         // span_pfx eligible steps
    double   tokprev_alnum_start_loss = 0;  // tok_prev loss on ALNUM_START bytes
    uint64_t n_alnum_start_bytes = 0;       // total ALNUM_START bytes observed

    if (rc->f_tel)
        fprintf(rc->f_tel, "i,target,loss_see,loss_uni,loss_bi,loss_lz,loss_lz8,loss_tokprev,"
                "loss_actual,w_see,w_uni,w_bi,w_lz,w_lz8,w_tokprev,loss_span,w_span\n");

    // ── Main per-byte loop ────────────────────────────────────────────────────
    for (int i = 0; i < rc->eval_len; i++) {
        if (i % 10000 == 0) { printf("  byte %d / %d\r", i, rc->eval_len); fflush(stdout); }

        int global_idx = rc->eval_start + 2 + i;
        uint8_t target = 0;
        if (rc->mode != MODE_DECODE) target = rc->data[global_idx];

        uint64_t t0 = __rdtsc();

        // 1. Extract SEE features
        float features[SEE_FEATURE_DIM];
        see_extract(&see_state, features);
        for (int fj = 0; fj < SEE_FEATURE_DIM; fj++)
            features[fj] = (features[fj] - wc->feature_means[fj]) / wc->feature_stds[fj];

        // 2. Build candidate list (topk or full)
        int num_cands = (tail_mode == 0) ? CLASSES : req_topk;
        int cand_idx[CLASSES];
        float cand_logit[CLASSES];

        if (tail_mode > 0 && wc->lr_rank > 0) {
            // Low-rank pre-filter
            float z[256];
            for (uint32_t r = 0; r < wc->lr_rank; r++)
                z[r] = dot_avx(&wc->A_proj[r * SEE_FEATURE_DIM], features, SEE_FEATURE_DIM);
            for (int k = 0; k < req_topk; k++) cand_logit[k] = -1e9f;
            for (int m = 0; m < req_topm; m++) {
                int c = wc->topk_indices[(ctx2*CLASSES+ctx1)*CLASSES+m];
                float lr_val = 0;
                for (uint32_t r = 0; r < wc->lr_rank; r++) lr_val += wc->B_proj[c*wc->lr_rank+r]*z[r];
                float score = wc->trigram_logits[(ctx2*CLASSES+ctx1)*CLASSES+c] + wc->B[c] + lr_val;
                if (score > cand_logit[req_topk-1]) {
                    int pos = req_topk-1;
                    while (pos > 0 && score > cand_logit[pos-1]) { cand_logit[pos]=cand_logit[pos-1]; cand_idx[pos]=cand_idx[pos-1]; pos--; }
                    cand_logit[pos] = score; cand_idx[pos] = c;
                }
            }
        } else {
            for (int k = 0; k < num_cands; k++)
                cand_idx[k] = (tail_mode == 0) ? k : wc->topk_indices[(ctx2*CLASSES+ctx1)*CLASSES+k];
        }

        // 3. Full residual dot products for candidates
        float full_logits[CLASSES]; float max_l = -1e9f;
        for (int k = 0; k < num_cands; k++) {
            int c = cand_idx[k];
            full_logits[k] = wc->trigram_logits[(ctx2*CLASSES+ctx1)*CLASSES+c] + wc->B[c]
                           + dot_avx(wc->W[c], features, SEE_FEATURE_DIM);
            if (full_logits[k] > max_l) max_l = full_logits[k];
        }

        // 4. Build p_see via softmax (+ tail if enabled)
        float tail_mass = 0;
        if (tail_mode > 0) {
            tail_mass = wc->Z_base[ctx2*CLASSES+ctx1];
            for (int k = 0; k < num_cands; k++) tail_mass -= wc->base_probs[(ctx2*CLASSES+ctx1)*CLASSES+cand_idx[k]];
            if (tail_mass < 0) tail_mass = 0;
        }
        if (max_l < 0) max_l = 0;
        float Z_K = 0;
        for (int k = 0; k < num_cands; k++) Z_K += expf(full_logits[k] - max_l);
        float Z_total = Z_K + tail_mass * expf(-max_l);
        float tail_scale = (tail_mode > 0) ? (expf(-max_l) / Z_total) : 0;

        for (int c = 0; c < CLASSES; c++)
            p_see[c] = (tail_mode > 0) ? wc->base_probs[(ctx2*CLASSES+ctx1)*CLASSES+c] * tail_scale : 0;
        for (int k = 0; k < num_cands; k++)
            p_see[cand_idx[k]] = expf(full_logits[k] - max_l) / Z_total;

        // 5. Build p_uni and p_bi (Laplace-smoothed dynamic counters)
        float alpha = 1.0f;
        float sum_uni = (float)dyn_total + CLASSES * alpha;
        float sum_bi  = CLASSES * alpha;
        for (int c = 0; c < CLASSES; c++) sum_bi += dyn_bi[ctx1][c];
        for (int c = 0; c < CLASSES; c++) {
            p_uni[c] = ((float)dyn_uni[c] + alpha) / sum_uni;
            p_bi[c]  = ((float)dyn_bi[ctx1][c] + alpha) / sum_bi;
        }

        // 6. Build p_lz (primary LZ, masked to lz_key_bytes)
        LzEntry* lz_ent  = NULL;
        LzEntry* lz_ent8 = NULL;
        uint64_t tok_key = 0;
        uint64_t prev_key = 0;
        if (no_lz || lz_mute) {
            for (int c = 0; c < CLASSES; c++) p_lz[c]  = 1.0f / 256.0f;
            for (int c = 0; c < CLASSES; c++) p_lz8[c] = 1.0f / 256.0f;
            for (int c = 0; c < CLASSES; c++) p_tok_prev[c] = 1.0f / 256.0f;
        } else {
            lz_ent = lz_lookup(lz_table, lz_ctx & lz_key_mask);
            lz_build_probs(lz_ent, LZ_K_DEFAULT, p_lz);
            if (lz_dual) {
                lz_ent8 = lz_lookup(lz_table8, lz_ctx);  // full 8-byte context
                lz_build_probs(lz_ent8, LZ_K_DEFAULT, p_lz8);
            } else if (tok_prefix && tok_table) {
                tok_key = tok_lz_key(&tok_state);
                LzEntry* tok_ent = lz_lookup(tok_table, tok_key);
                lz_build_probs(tok_ent, LZ_K_DEFAULT, p_lz8);
            } else {
                for (int c = 0; c < CLASSES; c++) p_lz8[c] = 1.0f / 256.0f;
            }
            
            if (tok_prev && tok_prev_table) {
                if (tok_state.tok_type == TOKTYPE_ALNUM || tok_state.tok_type == TOKTYPE_MACRO) {
                    for (int c = 0; c < CLASSES; c++) p_tok_prev[c] = 1.0f / 256.0f;
                } else {
                    prev_key = tok_prev_key(&tok_state);
                    LzEntry* p_ent = lz_lookup(tok_prev_table, prev_key);
                    lz_build_probs(p_ent, LZ_K_DEFAULT, p_tok_prev);
                }
            } else {
                for (int c = 0; c < CLASSES; c++) p_tok_prev[c] = 1.0f / 256.0f;
            }

            if (span_pfx && !span_pfx_mute && span_table && span_lz_eligible(&span_state)) {
                uint64_t span_key = span_lz_key(&span_state);
                LzEntry* sp_ent  = lz_lookup(span_table, span_key);
                lz_build_probs(sp_ent, LZ_K_DEFAULT, p_span);
            } else {
                for (int c = 0; c < CLASSES; c++) p_span[c] = 1.0f / 256.0f;
            }
        }

        // 7. Mix experts
        // Compute eligibility: tok_prev is armed when we are outside ALNUM/MACRO
        // and last_tok_hash != 0 (i.e. at least one ALNUM/MACRO has been seen).
        int prev_armed = (tok_prev_elig &&
                          tok_state.tok_type != TOKTYPE_ALNUM &&
                          tok_state.tok_type != TOKTYPE_MACRO &&
                          tok_state.last_tok_hash != 0);
        int span_armed = (span_pfx && span_lz_eligible(&span_state));
        uint8_t eligible[7] = {1, 1, 1, 1, 1, 1, 1};
        if (tok_prev_elig) {
            for (int i = 0; i < n_active; i++) {
                if (abs_slots[i] == 5) eligible[i] = prev_armed ? 1 : 0;
            }
            if (prev_armed) n_elig_bytes++;
        }
        if (span_pfx) {
            for (int i = 0; i < n_active; i++) {
                if (abs_slots[i] == 6) eligible[i] = span_armed ? 1 : 0;
            }
            if (span_armed) n_span_elig_bytes++;
        }

        if (use_moe) {
            moe_mix_gated(&moe, expert_ptrs, eligible, probs);
        } else {
            float lam = cfg->blend_lambda;
            for (int c = 0; c < CLASSES; c++) {
                float p_dyn = 0.5f * p_uni[c] + 0.5f * p_bi[c];
                probs[c] = (1.0f - lam) * p_see[c] + lam * p_dyn;
            }
        }

        // 8. Build CDF
        uint16_t freq[CLASSES]; uint32_t assigned = 0;
        float scale = (float)(CDF_SCALE - CLASSES);
        int max_c = 0; float max_p = -1;
        for (int c = 0; c < CLASSES; c++) {
            int fv = (int)(probs[c] * scale); if (fv < 0) fv = 0;
            freq[c] = (uint16_t)(fv + 1); assigned += freq[c];
            if (probs[c] > max_p) { max_p = probs[c]; max_c = c; }
        }
        int diff = CDF_SCALE - (int)assigned;
        if (diff > 0) freq[max_c] += (uint16_t)diff;

        // 9. Range coder I/O
        if (rc->f_enc) {
            uint32_t cum = 0;
            for (int c = 0; c < target; c++) cum += freq[c];
            rc_encode(&re, cum, freq[target], CDF_SCALE);
        } else if (rc->f_dec) {
            uint64_t fval = rc_get_freq(&rd, CDF_SCALE);
            uint32_t cum = 0; target = 0;
            for (int c = 0; c < CLASSES; c++) {
                if (fval >= cum && fval < cum + freq[c]) { target = (uint8_t)c; break; }
                cum += freq[c];
            }
            rc_decode(&rd, cum, freq[target], CDF_SCALE);
            if (rc->f_dump) { uint8_t t = target; fwrite(&t, 1, 1, rc->f_dump); }
        }

        uint64_t t1 = __rdtsc();
        total_cyc += (t1 - t0);

        // 10. Compute per-expert losses and mixture loss
        float fp = probs[target]; if (fp < 1e-10f) fp = 1e-10f;
        double sample_loss = -log2(fp);
        total_loss += sample_loss;

        double expert_loss[7];
        float ep[7] = { p_see[target], p_uni[target], p_bi[target], p_lz[target], p_lz8[target], p_tok_prev[target], p_span[target] };
        double oracle = 1e18;
        for (int e = 0; e < 7; e++) {
            if (ep[e] < 1e-10f) ep[e] = 1e-10f;
            expert_loss[e] = -log2(ep[e]);
            expert_loss_sum[e] += expert_loss[e];
        }

        double moe_losses[7];
        for (int i = 0; i < n_active; i++) {
            moe_losses[i] = expert_loss[abs_slots[i]];
            if (moe_losses[i] < oracle) oracle = moe_losses[i];
        }

        double qp = (double)freq[target] / CDF_SCALE; if (qp < 1e-10) qp = 1e-10;
        total_qloss += -log2(qp);

        // 11. Telemetry
        if (rc->f_tel) {
            double out_w[7] = {0};
            for (int ii = 0; ii < n_active; ii++) out_w[abs_slots[ii]] = moe.w[ii];
            fprintf(rc->f_tel, "%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
                    i, target,
                    expert_loss[0], expert_loss[1], expert_loss[2], expert_loss[3], expert_loss[4], expert_loss[5],
                    sample_loss, out_w[0], out_w[1], out_w[2], out_w[3], out_w[4], out_w[5],
                    expert_loss[6], out_w[6]);
        }

        // 12. Update MoE weights
        if (use_moe) moe_update_gated(&moe, moe_losses, eligible);

        // 12b. Regime prior: observe byte, blend prior into MoE for next prediction
        if (use_regime_prior) {
            regime_prior_observe(&rp, target);
            regime_prior_apply(&rp, &moe, abs_slots, n_active);
        }

        // 13. Update n-gram counters and SEE state
        see_observe(&see_state, target);
        dyn_uni[target]++; dyn_bi[ctx1][target]++; dyn_total++;

        // 14. Update LZ table(s)
        if (!no_lz && !lz_mute) {
            if (lz_ent)  lz_update(lz_ent,  lz_ctx & lz_key_mask, target);
            if (lz_dual && lz_ent8) lz_update(lz_ent8, lz_ctx, target);
            if (tok_prefix && tok_table) {
                LzEntry* tok_ent = lz_lookup(tok_table, tok_key);
                lz_update(tok_ent, tok_key, target);
            }
            if (tok_prev || tok_prefix) {
                int was_outside = (tok_state.tok_type != TOKTYPE_ALNUM && tok_state.tok_type != TOKTYPE_MACRO);
                uint64_t update_key = prev_key;
                TokLzType next_type = tok_lz_advance(&tok_state, target);
                // Track ALNUM_START: byte lands as first byte of a new ALNUM/MACRO token
                int is_alnum_start = was_outside && (next_type == TOKTYPE_ALNUM || next_type == TOKTYPE_MACRO);
                if (is_alnum_start) {
                    n_alnum_start_bytes++;
                    if (tok_prev) tokprev_alnum_start_loss += expert_loss[5];
                }
                if (tok_prev && tok_prev_table) {
                    if (is_alnum_start) {
                        LzEntry* p_ent = lz_lookup(tok_prev_table, update_key);
                        lz_update(p_ent, update_key, target);
                    }
                }
            }
        }
        // 15. Update span-prefix table
        if (span_pfx && !span_pfx_mute && span_table) {
            int was_elig = span_armed;  // eligibility before this byte
            if (was_elig) {
                uint64_t sp_key = span_lz_key(&span_state);
                LzEntry* sp_ent = lz_lookup(span_table, sp_key);
                lz_update(sp_ent, sp_key, target);
            }
        }
        span_lz_advance(&span_state, target);

        lz_ctx = (lz_ctx << 8) | target;

        ctx2 = ctx1; ctx1 = target;
    }
    printf("  byte %d / %d\n", rc->eval_len, rc->eval_len);

    if (rc->f_enc) { rc_encoder_flush(&re); }

    // ── Fill result (audit mode) ──────────────────────────────────────────────
    if (rc->result && rc->eval_len > 0) {
        SeeAuditResult* r = rc->result;
        double n = rc->eval_len;
        r->bpb             = total_loss  / n;
        r->quant_bpb       = total_qloss / n;
        r->see_bpb         = expert_loss_sum[0] / n;
        r->uni_bpb         = expert_loss_sum[1] / n;
        r->bi_bpb          = expert_loss_sum[2] / n;
        r->lz_bpb          = expert_loss_sum[3] / n;
        r->lz8_bpb         = (lz_dual || tok_prefix) ? (expert_loss_sum[4] / n) : 0.0;
        r->tok_prev_bpb    = tok_prev ? (expert_loss_sum[5] / n) : 0.0;
        r->span_bpb        = span_pfx ? (expert_loss_sum[6] / n) : 0.0;
        r->tok_prefix      = tok_prefix;
        r->tok_prev        = tok_prev;
        r->span_pfx        = span_pfx;
        r->unigram_bpb     = uni_loss_baseline / n;
        r->cycles_per_byte = (double)total_cyc / n;
        r->bytes_evaluated = (uint64_t)rc->eval_len;
        r->lz_key_bytes    = lz_key_bytes;
        r->lz_dual         = lz_dual;
        for (int i = 0; i < n_active; i++) {
            int a = abs_slots[i];
            r->avg_w[a]          = moe_avg_weight(&moe, i);
            r->avg_w_when_elig[a]= moe_avg_weight_elig(&moe, i);
            r->final_w[a]        = moe.w[i];
            r->wins[a]           = moe.wins[i];
        }
        r->n_elig_bytes             = n_elig_bytes;
        r->n_span_elig_bytes        = n_span_elig_bytes;
        r->tokprev_alnum_start_bpb  = (n_alnum_start_bytes > 0 && tok_prev)
                                        ? tokprev_alnum_start_loss / n_alnum_start_bytes
                                        : 0.0;
        r->n_alnum_start_bytes      = n_alnum_start_bytes;
        // oracle: recompute
        double oracle_sum = 0;
        // already accumulated implicitly; re-derive from per-expert sums
        // (true oracle requires per-sample min; we approximate via sum of expert bpbs is not oracle)
        // We stored oracle per sample in run — but we didn't accumulate it above.
        // Set to NaN to signal it's not computed in this pass.
        r->oracle_bpb = 0.0 / 0.0; // NaN
    }

    lz_table_free(lz_table);
    lz_table_free(lz_table8);
    lz_table_free(tok_table);
    lz_table_free(tok_prev_table);
    lz_table_free(span_table);
    return 0;
}

// ── Public API ────────────────────────────────────────────────────────────────

void see_codec_config_defaults(SeeCodecConfig* cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->moe_eta        = 0.03f;
    cfg->moe_share      = 0.001f;
    cfg->req_topk       = 256;
    cfg->req_topm       = 256;
    cfg->eval_start_pct = 0;
    cfg->eval_len_pct   = 100;
    cfg->lz_key_bytes   = 6;    // LZ6 default: Phase 26 tribunal verdict
}

static void apply_profile(SeeCodecConfig* cfg) {
    // Resolve speed: speed_profile takes precedence, then legacy profile field.
    const char* sp = cfg->speed_profile ? cfg->speed_profile : cfg->profile;
    if (!sp) return;
    if      (strcmp(sp, "full")     == 0) { cfg->req_topk = 256; cfg->tail_mode = 0; }
    else if (strcmp(sp, "accurate") == 0) { cfg->req_topk = 64;  cfg->tail_mode = 2; }
    else if (strcmp(sp, "fast")     == 0) { cfg->req_topk = 48;  cfg->tail_mode = 2; }
}

static void apply_expert_profile(SeeCodecConfig* cfg) {
    if (!cfg->expert_profile) return;
    if (strcmp(cfg->expert_profile, "general") == 0) {
        cfg->tok_prefix    = 1;
        cfg->tok_prev      = 0;
        cfg->tok_prev_elig = 0;
    } else if (strcmp(cfg->expert_profile, "prose") == 0 ||
               strcmp(cfg->expert_profile, "text")  == 0) {
        /* "text" kept as backward-compat alias for "prose" */
        cfg->tok_prefix    = 1;
        cfg->tok_prev      = 1;
        cfg->tok_prev_elig = 1;
    }
    // "experimental": manual flags used as-is, no override
}

int see_codec_encode_file(const char* input_path,
                          const char* archive_path,
                          const SeeCodecConfig* cfg_in) {
    SeeCodecConfig cfg = *cfg_in;
    apply_expert_profile(&cfg);
    apply_profile(&cfg);

    uint8_t* data = NULL; size_t data_size = 0;
    FILE* fi = fopen(input_path, "rb");
    if (!fi) { fprintf(stderr, "see: cannot open input: %s\n", input_path); return -1; }
    fseek(fi, 0, SEEK_END); data_size = ftell(fi); fseek(fi, 0, SEEK_SET);
    data = (uint8_t*)malloc(data_size);
    if (!data) { fclose(fi); return -1; }
    fread(data, 1, data_size, fi); fclose(fi);

    WeightsCtx* wc = weights_load(cfg.weights_path, cfg.tail_mode);
    if (!wc) { free(data); return -1; }

    FILE* fo = fopen(archive_path, "wb");
    if (!fo) { weights_free(wc); free(data); return -1; }
    setvbuf(fo, NULL, _IOFBF, 65536);

    // Write SEE3 header
    SeeArchiveHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic            = SEE3_MAGIC;
    hdr.header_size      = sizeof(SeeArchiveHeader);
    hdr.archive_flags    = SEE_FLAG_MOE_ACTIVE;
    if (!cfg.no_lz && !cfg.lz_mute) hdr.archive_flags |= SEE_FLAG_LZ_ENABLED;
    if (cfg.lz_dual) hdr.archive_flags |= SEE_FLAG_LZ_DUAL;
    if (cfg.tok_prefix)    hdr.archive_flags |= SEE_FLAG_TOK_PREFIX;
    if (cfg.tok_prev)      hdr.archive_flags |= SEE_FLAG_TOK_PREV;
    if (cfg.tok_prev_elig) hdr.archive_flags |= SEE_FLAG_TOK_PREV_ELIG;
    if (cfg.span_pfx)      hdr.archive_flags |= SEE_FLAG_SPAN_PFX;
    if (cfg.regime_prior)  hdr.archive_flags |= SEE_FLAG_REGIME_PRIOR;
    hdr.original_size    = (uint32_t)data_size;
    hdr.chunk_size       = wc->hdr.chunk_size;
    hdr.codebook_seed    = wc->hdr.codebook_seed;
    hdr.lz_hash_size     = LZ_HASH_SIZE;
    hdr.req_topk         = (uint16_t)cfg.req_topk;
    hdr.tail_mode        = (uint16_t)cfg.tail_mode;
    hdr.coder_scale_bits = CDF_SCALE_BITS;
    hdr.lz_top_n         = LZ_TOP_N;
    hdr.profile_id       = 0;
    hdr.decay            = wc->hdr.decay;
    hdr.blend_lambda     = cfg.use_moe ? -2.0f : cfg.blend_lambda;
    hdr.lz_K             = LZ_K_DEFAULT;
    hdr.moe_eta          = cfg.moe_eta;
    hdr.moe_share        = cfg.moe_share;
    hdr.seed_byte0       = data_size > 0 ? data[0] : 0;
    hdr.seed_byte1       = data_size > 1 ? data[1] : 0;
    hdr.lz_key_bytes     = (uint8_t)cfg.lz_key_bytes;

    // Bind archive to the exact weights file used at encode time.
    // sha256_file returns empty string on failure (non-fatal: archive still valid).
    char wsha_hex[65] = {0};
    sha256_file(cfg.weights_path, wsha_hex);
    for (int i = 0; i < 32; i++) {
        unsigned v = 0;
        sscanf(wsha_hex + i*2, "%02x", &v);
        hdr.weights_sha256[i] = (uint8_t)v;
    }

    fwrite(&hdr, sizeof(hdr), 1, fo);

    RunCtx rc = {0};
    rc.wc         = wc;
    rc.cfg        = &cfg;
    rc.mode       = MODE_ENCODE;
    rc.data       = data;
    rc.data_size  = data_size;
    rc.eval_start = 0;
    rc.eval_len   = data_size >= 2 ? (int)data_size - 2 : 0;
    rc.f_enc      = fo;

    int ret = run_loop(&rc);
    fclose(fo);
    weights_free(wc);
    free(data);
    return ret;
}

int see_codec_decode_file(const char* archive_path,
                          const char* output_path,
                          const char* weights_path) {
    FILE* fi = fopen(archive_path, "rb");
    if (!fi) { fprintf(stderr, "see: cannot open archive: %s\n", archive_path); return -1; }
    setvbuf(fi, NULL, _IOFBF, 65536);

    SeeArchiveHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, fi) != 1) {
        fprintf(stderr, "see: truncated archive header\n"); fclose(fi); return -1;
    }
    if (hdr.magic != SEE3_MAGIC) {
        fprintf(stderr, "see: bad magic 0x%08X (expected 0x%08X \"SEE3\")\n",
                hdr.magic, SEE3_MAGIC);
        fclose(fi); return -1;
    }
    if (hdr.header_size != sizeof(SeeArchiveHeader)) {
        fprintf(stderr, "see: header_size mismatch (%u vs %zu) — version incompatible\n",
                hdr.header_size, sizeof(SeeArchiveHeader));
        fclose(fi); return -1;
    }

    // Verify the weights file matches what was used at encode time.
    {
        char wsha_hex[65] = {0};
        sha256_file(weights_path, wsha_hex);
        uint8_t wsha[32] = {0};
        for (int i = 0; i < 32; i++) {
            unsigned v = 0;
            sscanf(wsha_hex + i*2, "%02x", &v);
            wsha[i] = (uint8_t)v;
        }
        // All-zero header hash means weights were unavailable at encode — skip check.
        int hdr_nonzero = 0;
        for (int i = 0; i < 32; i++) if (hdr.weights_sha256[i]) { hdr_nonzero = 1; break; }
        if (hdr_nonzero && memcmp(hdr.weights_sha256, wsha, 32) != 0) {
            char ahex[65]; ahex[64] = 0;
            for (int i = 0; i < 32; i++) sprintf(ahex + i*2, "%02x", hdr.weights_sha256[i]);
            fprintf(stderr,
                "see: weights mismatch — archive was encoded with different weights\n"
                "     archive sha256: %.16s...\n"
                "     current sha256: %.16s...\n",
                ahex, wsha_hex);
            fclose(fi); return -1;
        }
    }

    // Reconstruct config from header
    SeeCodecConfig cfg;
    see_codec_config_defaults(&cfg);
    cfg.weights_path  = weights_path;
    cfg.use_moe       = (hdr.archive_flags & SEE_FLAG_MOE_ACTIVE) != 0;
    cfg.no_lz         = (hdr.archive_flags & SEE_FLAG_LZ_ENABLED) == 0;
    cfg.lz_dual       = (hdr.archive_flags & SEE_FLAG_LZ_DUAL) != 0;
    cfg.tok_prefix    = (hdr.archive_flags & SEE_FLAG_TOK_PREFIX) != 0;
    cfg.tok_prev      = (hdr.archive_flags & SEE_FLAG_TOK_PREV) != 0;
    cfg.tok_prev_elig = (hdr.archive_flags & SEE_FLAG_TOK_PREV_ELIG) != 0;
    cfg.span_pfx      = (hdr.archive_flags & SEE_FLAG_SPAN_PFX) != 0;
    cfg.regime_prior  = (hdr.archive_flags & SEE_FLAG_REGIME_PRIOR) != 0;
    cfg.req_topk      = hdr.req_topk;
    cfg.tail_mode     = hdr.tail_mode;
    cfg.blend_lambda  = hdr.blend_lambda == -2.0f ? 0.0f : hdr.blend_lambda;
    cfg.moe_eta       = hdr.moe_eta;
    cfg.moe_share     = hdr.moe_share;
    cfg.lz_key_bytes  = hdr.lz_key_bytes;
    cfg.eval_start_pct = 0;
    cfg.eval_len_pct   = 100;

    WeightsCtx* wc = weights_load(weights_path, cfg.tail_mode);
    if (!wc) { fclose(fi); return -1; }

    FILE* fo = fopen(output_path, "wb");
    if (!fo) { weights_free(wc); fclose(fi); return -1; }
    setvbuf(fo, NULL, _IOFBF, 65536);

    // Write seed bytes first
    if (hdr.original_size > 0) fwrite(&hdr.seed_byte0, 1, 1, fo);
    if (hdr.original_size > 1) fwrite(&hdr.seed_byte1, 1, 1, fo);

    // Borrow a small buffer to pass seed bytes to run_loop
    uint8_t seed_buf[2] = { hdr.seed_byte0, hdr.seed_byte1 };

    RunCtx rc = {0};
    rc.wc        = wc;
    rc.cfg       = &cfg;
    rc.mode      = MODE_DECODE;
    rc.data      = seed_buf;
    rc.data_size = hdr.original_size;
    rc.eval_start = 0;
    rc.eval_len  = hdr.original_size >= 2 ? (int)hdr.original_size - 2 : 0;
    rc.f_dec     = fi;
    rc.f_dump    = fo;

    int ret = run_loop(&rc);
    fclose(fo);
    fclose(fi);
    weights_free(wc);
    return ret;
}

int see_codec_audit_file(const char* input_path,
                         const SeeCodecConfig* cfg_in,
                         SeeAuditResult* result) {
    SeeCodecConfig cfg = *cfg_in;
    apply_expert_profile(&cfg);
    apply_profile(&cfg);

    uint8_t* data = NULL; size_t data_size = 0;
    FILE* fi = fopen(input_path, "rb");
    if (!fi) { fprintf(stderr, "see: cannot open input: %s\n", input_path); return -1; }
    fseek(fi, 0, SEEK_END); data_size = ftell(fi); fseek(fi, 0, SEEK_SET);
    data = (uint8_t*)malloc(data_size);
    if (!data) { fclose(fi); return -1; }
    fread(data, 1, data_size, fi); fclose(fi);

    sha256_file(input_path, result->input_sha256);

    WeightsCtx* wc = weights_load(cfg.weights_path, cfg.tail_mode);
    if (!wc) { free(data); return -1; }

    FILE* f_tel = NULL;
    if (cfg.telemetry_path) {
        f_tel = fopen(cfg.telemetry_path, "w");
    }

    int eval_start = (int)((data_size * cfg.eval_start_pct) / 100);
    int eval_len   = (int)((data_size * cfg.eval_len_pct)   / 100);
    if (eval_start + eval_len + 2 > (int)data_size)
        eval_len = (int)data_size - eval_start - 2;

    RunCtx rc = {0};
    rc.wc         = wc;
    rc.cfg        = &cfg;
    rc.mode       = MODE_AUDIT;
    rc.data       = data;
    rc.data_size  = data_size;
    rc.eval_start = eval_start;
    rc.eval_len   = eval_len;
    rc.f_tel      = f_tel;
    rc.result     = result;

    int ret = run_loop(&rc);

    if (f_tel) fclose(f_tel);
    weights_free(wc);
    free(data);
    return ret;
}

void see_audit_result_print(const SeeAuditResult* r, const char* label) {
    printf("\n=== SEE Audit: %s ===\n", label ? label : "");
    printf("Input SHA-256:   %s\n", r->input_sha256);
    printf("Bytes evaluated: %llu\n", (unsigned long long)r->bytes_evaluated);
    printf("Unigram BPB:     %.4f\n", r->unigram_bpb);
    printf("Model BPB:       %.4f\n", r->bpb);
    printf("Quantized BPB:   %.4f\n", r->quant_bpb);
    printf("Cycles/byte:     %.1f\n",  r->cycles_per_byte);
    printf("LZ key width:    %d bytes%s%s%s\n", r->lz_key_bytes,
           r->lz_dual    ? " + LZ8 dual"         : "",
           r->tok_prefix ? " + tok-prefix expert" : "",
           r->tok_prev   ? " + tok-prev expert"   : "");
    printf("--- Per-expert BPB ---\n");
    printf("SEE Only:        %.4f\n", r->see_bpb);
    printf("UNI Only:        %.4f\n", r->uni_bpb);
    printf("BI  Only:        %.4f\n", r->bi_bpb);
    printf("LZ  Only:        %.4f  (key=%d bytes)\n", r->lz_bpb, r->lz_key_bytes);
    if (r->lz_dual || r->tok_prefix)
        printf("%-16s %.4f\n",
               r->tok_prefix ? "TOKPFX Only:    " : "LZ8 Only:       ",
               r->lz8_bpb);
    if (r->tok_prev)
        printf("TOKPREV Only:    %.4f\n", r->tok_prev_bpb);
    if (r->span_pfx)
        printf("SPANPFX Only:    %.4f\n", r->span_bpb);

    printf("--- MoE weights ---\n");
    char label_buf[128] = "SEE UNI BI  LZ ";
    if (r->lz_dual) strcat(label_buf, " LZ8   ");
    else if (r->tok_prefix) strcat(label_buf, " TOKPFX");
    if (r->tok_prev) strcat(label_buf, " TKPREV");
    if (r->span_pfx) strcat(label_buf, " SPANPFX");

    printf("Avg  [%s]: %.4f %.4f %.4f %.4f", label_buf, r->avg_w[0], r->avg_w[1], r->avg_w[2], r->avg_w[3]);
    if (r->lz_dual || r->tok_prefix) printf(" %.4f", r->avg_w[4]);
    if (r->tok_prev) printf(" %.4f", r->avg_w[5]);
    if (r->span_pfx) printf(" %.4f", r->avg_w[6]);
    printf("\n");

    printf("Final[%s]: %.4f %.4f %.4f %.4f", label_buf, r->final_w[0], r->final_w[1], r->final_w[2], r->final_w[3]);
    if (r->lz_dual || r->tok_prefix) printf(" %.4f", r->final_w[4]);
    if (r->tok_prev) printf(" %.4f", r->final_w[5]);
    if (r->span_pfx) printf(" %.4f", r->final_w[6]);
    printf("\n");

    if (r->span_pfx && r->n_span_elig_bytes > 0) {
        double pct = 100.0 * r->n_span_elig_bytes / r->bytes_evaluated;
        printf("--- SPANPFX Eligibility ---\n");
        printf("Eligible bytes:  %llu / %llu (%.1f%%)\n",
               (unsigned long long)r->n_span_elig_bytes,
               (unsigned long long)r->bytes_evaluated, pct);
        printf("W_SPANPFX global:%.4f   when-eligible: %.4f\n",
               r->avg_w[6], r->avg_w_when_elig[6]);
    }
    if (r->tok_prev && r->n_elig_bytes > 0) {
        double pct_elig = 100.0 * r->n_elig_bytes / r->bytes_evaluated;
        printf("--- TOK_PREV Eligibility ---\n");
        printf("Eligible bytes:  %llu / %llu (%.1f%%)\n",
               (unsigned long long)r->n_elig_bytes,
               (unsigned long long)r->bytes_evaluated,
               pct_elig);
        printf("W_TOKPREV global:%.4f   when-eligible: %.4f\n",
               r->avg_w[5], r->avg_w_when_elig[5]);
        if (r->n_alnum_start_bytes > 0) {
            double pct_start = 100.0 * r->n_alnum_start_bytes / r->bytes_evaluated;
            printf("ALNUM_START:     %llu bytes (%.1f%%)\n",
                   (unsigned long long)r->n_alnum_start_bytes, pct_start);
            printf("TOKPREV BPB@START: %.4f\n", r->tokprev_alnum_start_bpb);
        }
    }
}
