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
    uint8_t  _pad[2];           // alignment padding — must be zero
} SeeArchiveHeader;             // sizeof = 60 bytes

// ── Codec Configuration ───────────────────────────────────────────────────────

typedef struct {
    const char* weights_path;   // path to weights .bin file (required)

    // Expert selection
    int   no_lz;                // 1 → 3-expert mode (SEE+UNI+BI), no LZ table
    int   lz_mute;              // 1 → LZ always uniform (ablation only)
    int   lz_key_bytes;         // context width: 4 (default), 6, or 8
    int   lz_dual;              // 1 → 5-expert mode: LZ4 + LZ8 as separate experts
    int   tok_prefix;           // 1 → replace LZ8 slot with inside-token prefix expert

    // MoE
    int   use_moe;              // 1 → fixed-share credit assignment
    float moe_eta;              // default 0.03
    float moe_share;            // default 0.001

    // SEE prediction
    int   req_topk;             // top-k candidates, 256 = full (default)
    int   req_topm;             // top-m for low-rank pre-filter (default 256)
    int   tail_mode;            // 0=none, 1=ngram, 2=ngram+bias
    float blend_lambda;         // static blend; overridden when use_moe=1

    // Profile shortcut (overrides topk/topm/tail_mode when set)
    // NULL, "full", "accurate", or "fast"
    const char* profile;

    // Audit-mode window (percentages, 0–100)
    int   eval_start_pct;       // default 0  (0 = from byte 0)
    int   eval_len_pct;         // default 100 (100 = entire file)

    // Optional telemetry CSV path (NULL = no telemetry)
    const char* telemetry_path;
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
    double   oracle_bpb;        // oracle (best expert per byte)
    double   unigram_bpb;       // i.i.d. unigram lower-bound on this segment
    double   cycles_per_byte;   // total CPU cycles / evaluated bytes
    double   avg_w[5];          // average MoE weights [SEE, UNI, BI, LZ, LZ8]
    double   final_w[5];        // final MoE weights at end of segment
    uint64_t wins[5];           // per-expert "best predictor" counts
    uint64_t bytes_evaluated;
    char     input_sha256[65];  // hex SHA-256 of the input file
    int      lz_key_bytes;      // recorded from config (for display)
    int      lz_dual;           // 1 if dual-LZ mode was active
    int      tok_prefix;        // 1 if tok-prefix expert occupied the LZ8 slot
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
