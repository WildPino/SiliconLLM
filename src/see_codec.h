#ifndef SEE_CODEC_H
#define SEE_CODEC_H

#include <stdint.h>

// ── SEE3 Archive Format ───────────────────────────────────────────────────────
// Magic: 0x33454553 ("SEE3").
// The decoder reads all parameters from this header — no external flags needed.

#define SEE3_MAGIC          0x33454553u

// archive_flags bitmask
#define SEE_FLAG_LZ_ENABLED 0x01u   // LZ Top-N expert is active
#define SEE_FLAG_MOE_ACTIVE 0x02u   // Fixed-share MoE credit assignment
#define SEE_FLAG_LZ_DUAL    0x04u
#define SEE_FLAG_TOK_PREFIX 0x08u
#define SEE_FLAG_TOK_PREV      0x10u
#define SEE_FLAG_TOK_PREV_ELIG 0x20u
#define SEE_FLAG_SPAN_PFX      0x40u  // inline-span delimiter expert active
#define SEE_FLAG_REGIME_PRIOR  0x80u  // Phase 33 regime prior router active
#define SEE_FLAG_REGIME_CREDIT 0x100u // Phase 34 credit-only regime router (research)
#define SEE_FLAG_REGIME_DUAL   0x200u // Phase 34B dual-EMA drift router (research)

typedef struct {
    uint32_t magic;             // SEE3_MAGIC
    uint32_t header_size;       // sizeof(SeeArchiveHeader) — forward-compat sentinel
    uint32_t archive_flags;     // SEE_FLAG_* bitmask
    uint32_t original_size;     // byte count of the original (uncompressed) file
    uint32_t chunk_size;        // SEE engine chunk size (from weights file)
    uint32_t codebook_seed;     // SEE codebook seed (from weights file)
    uint32_t lz_hash_size;      // LZ table entry count (power of 2)
    uint16_t req_topk;          // SEE top-k candidate count (256 = full)
    uint16_t tail_mode;         // 0=none, 1=ngram, 2=ngram+bias
    uint16_t coder_scale_bits;  // log2(CDF_SCALE); typically 14 → 16384
    uint8_t  lz_top_n;          // Top-N slots per LZ entry
    uint8_t  profile_id;        // 0=full, 1=accurate, 2=fast
    float    decay;             // SEE decay parameter (from weights file)
    float    blend_lambda;      // static blend: 0=SEE-only, −2=MoE
    float    lz_K;              // LZ smoothing constant (mass_lz = total/(total+K))
    float    moe_eta;           // MoE exponentiated gradient rate
    float    moe_share;         // MoE fixed-share redistribution coefficient
    uint8_t  seed_byte0;        // first byte of original (seeded into decoder)
    uint8_t  seed_byte1;        // second byte of original
    uint8_t  lz_key_bytes;      // LZ key width in bytes
    uint8_t  _pad1;             // alignment padding — must be zero
    uint8_t  weights_sha256[32]; // SHA-256 of the weights .bin file used at encode time
} SeeArchiveHeader;             // sizeof = 92 bytes

// ── Codec Configuration ───────────────────────────────────────────────────────

typedef struct {
    const char* weights_path;   // path to weights .bin file (required)

    // Expert selection
    int   no_lz;                // 1 → 3-expert mode (SEE+UNI+BI), no LZ table
    int   lz_mute;              // 1 → LZ always uniform (ablation only)
    int   lz_key_bytes;         // context width: 4 (default), 6, or 8
    int   lz_dual;              // 1 → 5-expert mode: LZ4 + LZ8 as separate experts
    int   tok_prefix;           // 1 → replace LZ8 slot with inside-token prefix expert
    int   tok_prev;             // 1 → token transition expert
    int   tok_prev_mute;        // 1 → token transition expert stays uniform (ablation)
    int   tok_prev_elig;        // 1 → token transition expert is dynamically eligible
    int   span_pfx;             // 1 → inline-span delimiter expert (backtick/dollar)
    int   span_pfx_mute;        // 1 → span expert stays uniform (ablation)

    // MoE
    int   use_moe;              // 1 → fixed-share credit assignment
    float moe_eta;              // default 0.03
    float moe_share;            // default 0.001

    // SEE prediction
    int   req_topk;             // top-k candidates, 256 = full (default)
    int   req_topm;             // top-m for low-rank pre-filter (default 256)
    int   tail_mode;            // 0=none, 1=ngram, 2=ngram+bias
    float blend_lambda;         // static blend; overridden when use_moe=1

    // Speed shortcut (overrides topk/topm/tail_mode when set)
    // NULL, "full", "accurate", or "fast"
    const char* profile;   // legacy alias for speed_profile
    const char* speed_profile;  // "full" | "accurate" | "fast"

    // Expert-set shortcut (overrides individual expert flags when set)
    // NULL:        manual flags used as-is
    // "general":   LZ6 + TOKPFX
    // "text":      LZ6 + TOKPFX + TOK_PREV_ELIG
    const char* expert_profile;

    // Audit-mode window (percentages, 0–100)
    int   eval_start_pct;       // default 0  (0 = from byte 0)
    int   eval_len_pct;         // default 100 (100 = entire file)

    // Optional telemetry CSV path (NULL = no telemetry)
    const char* telemetry_path;

    // Phase 33: Regime Prior Router
    int   regime_prior;      // 1 = enable regime prior blending
    int   regime_prior_mute; // 1 = active but neutral prior (ablation)

    // Phase 34 (research): Credit-Only Regime Router
    int   regime_credit;     // 1 = enable credit-based prior (replaces regime_prior)

    // Phase 34B (research): Dual-EMA Drift Router
    int   regime_dual;       // 1 = enable dual-EMA drift-gated prior
} SeeCodecConfig;

// ── Audit Results ─────────────────────────────────────────────────────────────

typedef struct {
    double   bpb;               // model BPB (floating-point mixture)
    double   quant_bpb;         // quantized BPB (as range-coded)
    double   see_bpb;           // SEE-only BPB
    double   uni_bpb;           // UNI-only BPB
    double   bi_bpb;            // BI-only BPB
    double   lz_bpb;            // LZ-only BPB (primary LZ, key=lz_key_bytes)
    double   lz8_bpb;           // LZ8-only BPB (dual mode) or TOKPFX-only BPB (tok_prefix mode)
    double   tok_prev_bpb;      // TOKPREV-only BPB
    double   span_bpb;          // SPANPFX-only BPB
    double   oracle_bpb;        // oracle (best expert per byte)
    double   unigram_bpb;       // i.i.d. unigram lower-bound on this segment
    double   cycles_per_byte;   // total CPU cycles / evaluated bytes
    double   avg_w[7];          // average MoE weights [SEE,UNI,BI,LZ,TOKPFX,TOKPREV,SPANPFX]
    double   avg_w_when_elig[7]; // average weight only on eligible steps (0 if not gated)
    double   final_w[7];        // final MoE weights at end of segment
    uint64_t wins[7];           // per-expert "best predictor" counts
    uint64_t n_elig_bytes;      // bytes where tok_prev was eligible
    uint64_t n_span_elig_bytes; // bytes where span_pfx was eligible (inside span)
    uint64_t bytes_evaluated;
    // Per-tok-category loss for TOKPREV (global, all bytes)
    double   tokprev_alnum_start_bpb;  // TOKPREV loss on bytes that turned out to be ALNUM_START
    uint64_t n_alnum_start_bytes;      // count of ALNUM_START bytes in segment
    char     input_sha256[65];  // hex SHA-256 of the input file
    int      lz_key_bytes;      // recorded from config (for display)
    int      lz_dual;           // 1 if dual-LZ mode was active
    int      tok_prefix;        // 1 if tok-prefix expert occupied the LZ8 slot
    int      tok_prev;          // 1 if tok-prev expert was active
    int      span_pfx;          // 1 if span-prefix expert was active
} SeeAuditResult;

// ── Public API ────────────────────────────────────────────────────────────────

// Initialize cfg to safe defaults. Call before filling individual fields.
void see_codec_config_defaults(SeeCodecConfig* cfg);

// Encode input_path → archive_path using SEE3 format.
// Returns 0 on success, non-zero on error.
int see_codec_encode_file(const char* input_path,
                          const char* archive_path,
                          const SeeCodecConfig* cfg);

// Decode archive_path → output_path.
// All codec parameters are read from the SEE3 header; weights_path is required.
// Returns 0 on success, non-zero on error.
int see_codec_decode_file(const char* archive_path,
                          const char* output_path,
                          const char* weights_path);

// Evaluate input_path without encoding/decoding. Fills *result.
// Returns 0 on success, non-zero on error.
int see_codec_audit_file(const char* input_path,
                         const SeeCodecConfig* cfg,
                         SeeAuditResult* result);

// Print a formatted result summary to stdout.
void see_audit_result_print(const SeeAuditResult* r, const char* label);

#endif // SEE_CODEC_H
