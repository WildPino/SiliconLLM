"""
gen_corpus_29a.py  —  Generate Phase 29A synthetic corpus files.

Outputs (all in data/):
  json_synth.json          ~120 KB  structured JSON (API/config-like)
  c_header_synth.h         ~120 KB  C/C++ header declarations
  project_notes_it.txt     ~150 KB  Italian text (promessi_sposi slice + walkthrough docs)
  repo_markdown_mixed.md   ~110 KB  diverse markdown (README + API + changelog style)
  log_synth.log            ~110 KB  synthetic server/application log

All RNG uses seed=42 for reproducibility.
Run from the repo root or from scripts/.
"""

import os
import re
import random
import json
import hashlib
import struct
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

rng = random.Random(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def target_path(name):
    return os.path.join(DATA, name)

def already_exists(name):
    p = target_path(name)
    if os.path.exists(p):
        print(f"  [SKIP] {name} already exists ({os.path.getsize(p)//1024} KB)")
        return True
    return False

def write_file(name, content, encoding="utf-8"):
    p = target_path(name)
    if isinstance(content, str):
        content = content.encode(encoding)
    with open(p, "wb") as f:
        f.write(content)
    sha = hashlib.sha256(content).hexdigest()[:16]
    print(f"  [OK]   {name}  {len(content)//1024} KB  sha={sha}...")


# ---------------------------------------------------------------------------
# 1. json_synth.json  —  Structured JSON with realistic patterns
# ---------------------------------------------------------------------------

FIRST_NAMES = ["Alice", "Bruno", "Carla", "Diego", "Elena", "Fabio", "Giulia",
               "Hector", "Irene", "Joel", "Katia", "Luca", "Marco", "Nadia",
               "Oscar", "Paola", "Quentin", "Rosa", "Simone", "Tania"]
LAST_NAMES  = ["Rossi", "Bianchi", "Ferrari", "Colombo", "Ricci", "Marino",
               "Greco", "Bruno", "Gallo", "Conti", "De Luca", "Costa",
               "Giordano", "Mancini", "Rizzo", "Lombardi", "Moreno", "Barbieri"]
DOMAINS     = ["gmail.com", "hotmail.com", "company.io", "example.org", "dev.net"]
ROLES       = ["admin", "editor", "viewer", "developer", "analyst", "operator"]
COUNTRIES   = ["IT", "FR", "DE", "US", "BR", "ES", "PL", "NL", "PT", "SE"]
STATUS      = ["active", "inactive", "pending", "suspended"]

EVENT_TYPES = ["user.login", "user.logout", "file.upload", "file.delete",
               "config.update", "api.call", "error.raised", "session.expire",
               "auth.failure", "rate.limit"]
ENDPOINTS   = ["/api/v1/users", "/api/v1/files", "/api/v2/config",
               "/api/v2/metrics", "/api/v1/auth/token", "/api/v2/events",
               "/api/v1/export", "/api/v2/search", "/health", "/metrics"]
HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"]
HTTP_STATUS  = [200, 200, 200, 201, 204, 400, 401, 403, 404, 422, 500, 503]

CONFIG_KEYS = {
    "database": {
        "host": "db.internal.company.io",
        "port": 5432,
        "name": "entropy_engine",
        "pool_min": 4,
        "pool_max": 32,
        "timeout_ms": 5000,
        "ssl": True,
        "retry_count": 3,
    },
    "cache": {
        "backend": "redis",
        "host": "cache.internal.company.io",
        "port": 6379,
        "ttl_seconds": 3600,
        "max_memory_mb": 512,
        "eviction_policy": "allkeys-lru",
    },
    "api": {
        "rate_limit_per_min": 1000,
        "max_payload_bytes": 10485760,
        "timeout_ms": 30000,
        "cors_origins": ["https://app.company.io", "https://dev.company.io"],
        "auth_providers": ["jwt", "api_key"],
        "version": "2.4.1",
    },
    "storage": {
        "provider": "s3_compatible",
        "bucket": "entropy-artifacts",
        "region": "eu-west-1",
        "max_object_size_mb": 256,
        "compression": "zstd",
        "encryption": "aes-256-gcm",
    },
    "logging": {
        "level": "INFO",
        "format": "json",
        "sink": "stdout",
        "include_trace": True,
        "redact_fields": ["password", "token", "secret", "api_key"],
    },
    "feature_flags": {
        "new_compression_pipeline": True,
        "experimental_moe_v3": False,
        "enable_telemetry": True,
        "beta_ui": False,
        "async_encode": True,
    },
}

METRIC_NAMES = ["cpu_usage_pct", "mem_usage_mb", "disk_io_mbps",
                "api_latency_p50_ms", "api_latency_p95_ms", "api_latency_p99_ms",
                "requests_per_sec", "errors_per_sec", "active_connections",
                "queue_depth", "cache_hit_rate", "compression_ratio"]


def make_user(i):
    fn = rng.choice(FIRST_NAMES)
    ln = rng.choice(LAST_NAMES)
    return {
        "id": f"usr_{i:06d}",
        "username": f"{fn.lower()}.{ln.lower().replace(' ', '_')}_{i}",
        "email": f"{fn.lower()}{i}@{rng.choice(DOMAINS)}",
        "full_name": f"{fn} {ln}",
        "role": rng.choice(ROLES),
        "country": rng.choice(COUNTRIES),
        "status": rng.choice(STATUS),
        "created_at": (datetime(2022, 1, 1) + timedelta(days=rng.randint(0, 900))).isoformat() + "Z",
        "last_login": (datetime(2024, 1, 1) + timedelta(days=rng.randint(0, 500))).isoformat() + "Z",
        "score": round(rng.uniform(0.0, 100.0), 2),
        "tags": rng.sample(["power_user", "beta", "vip", "trial", "enterprise",
                             "internal", "verified", "legacy"], k=rng.randint(0, 3)),
        "quota_used_bytes": rng.randint(0, 10 * 1024 * 1024),
        "quota_limit_bytes": rng.choice([1, 5, 10, 50, 100]) * 1024 * 1024,
        "mfa_enabled": rng.random() > 0.4,
        "preferences": {
            "theme": rng.choice(["light", "dark", "auto"]),
            "language": rng.choice(["en", "it", "fr", "de", "es"]),
            "notifications": rng.random() > 0.3,
            "timezone": rng.choice(["UTC", "Europe/Rome", "America/New_York",
                                    "Asia/Tokyo", "Europe/Berlin"]),
        },
    }


def make_event(i):
    t = datetime(2024, 1, 1) + timedelta(seconds=rng.randint(0, 365 * 86400))
    et = rng.choice(EVENT_TYPES)
    ep = rng.choice(ENDPOINTS)
    return {
        "event_id": f"evt_{i:08x}",
        "type": et,
        "timestamp": t.isoformat() + "Z",
        "user_id": f"usr_{rng.randint(0, 199):06d}",
        "session_id": f"sess_{rng.randint(0, 9999):06x}",
        "endpoint": ep,
        "method": rng.choice(HTTP_METHODS),
        "status_code": rng.choice(HTTP_STATUS),
        "duration_ms": rng.randint(1, 5000),
        "bytes_sent": rng.randint(64, 65536),
        "bytes_recv": rng.randint(0, 131072),
        "ip": f"10.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}",
        "user_agent": rng.choice([
            "Mozilla/5.0 (compatible; EntropyBot/1.0)",
            "SEE-Client/2.4.1 (Linux; x86_64)",
            "curl/7.88.1",
            "python-requests/2.31.0",
            "axios/1.4.0",
        ]),
        "error": (f"Error code {rng.randint(1000, 9999)}: {rng.choice(['timeout', 'auth_failed', 'not_found', 'rate_limited'])}"
                  if rng.random() < 0.12 else None),
    }


def make_metric_snapshot(ts):
    return {
        "timestamp": ts.isoformat() + "Z",
        "node_id": f"node-{rng.randint(1, 8):02d}",
        "values": {k: round(rng.uniform(0, 100 if "pct" in k or "ratio" in k or "rate" in k
                                        else (500 if "ms" in k
                                              else (1024 if "mb" in k
                                                    else 500))), 3)
                   for k in METRIC_NAMES},
    }


def gen_json_synth():
    doc = {
        "schema_version": "1.3.0",
        "generated_at": "2026-05-25T12:00:00Z",
        "generator": "gen_corpus_29a.py",
        "config": CONFIG_KEYS,
        "users": [make_user(i) for i in range(90)],
        "events": [make_event(i) for i in range(150)],
        "metrics": {
            "snapshots": [
                make_metric_snapshot(datetime(2026, 5, 25) - timedelta(hours=h))
                for h in range(0, 48, 1)
            ],
            "summary": {
                k: {
                    "mean": round(rng.uniform(0, 100), 3),
                    "p50": round(rng.uniform(0, 100), 3),
                    "p95": round(rng.uniform(0, 100), 3),
                    "p99": round(rng.uniform(0, 100), 3),
                    "min": round(rng.uniform(0, 10), 3),
                    "max": round(rng.uniform(90, 100), 3),
                }
                for k in METRIC_NAMES
            },
        },
    }
    return json.dumps(doc, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# 2. c_header_synth.h  —  Realistic C/C++ header
# ---------------------------------------------------------------------------

C_HEADER_TEMPLATE = """\
/*
 * c_header_synth.h  —  Silicon Entropy Engine: synthetic header corpus
 *
 * Includes: type aliases, integer constants, flag sets, struct definitions,
 * function prototypes, and inline utility macros.
 *
 * Generated by gen_corpus_29a.py (seed=42) for Phase 29A corpus tests.
 */

#ifndef SEE_SYNTH_CORPUS_H
#define SEE_SYNTH_CORPUS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <limits.h>

/* =========================================================================
 * Basic integer typedefs
 * ========================================================================= */

typedef uint8_t   u8;
typedef uint16_t  u16;
typedef uint32_t  u32;
typedef uint64_t  u64;
typedef int8_t    i8;
typedef int16_t   i16;
typedef int32_t   i32;
typedef int64_t   i64;
typedef float     f32;
typedef double    f64;

typedef u32 see_flags_t;
typedef u64 see_handle_t;
typedef u32 see_version_t;
typedef i32 see_error_t;

/* =========================================================================
 * Version and capabilities
 * ========================================================================= */

#define SEE_VERSION_MAJOR      3
#define SEE_VERSION_MINOR      2
#define SEE_VERSION_PATCH      1
#define SEE_VERSION_BUILD      20260525
#define SEE_VERSION_STRING     "3.2.1-20260525"

#define SEE_MAKE_VERSION(maj, min, pat) \\
    (((u32)(maj) << 24) | ((u32)(min) << 16) | ((u32)(pat)))

#define SEE_VERSION SEE_MAKE_VERSION(SEE_VERSION_MAJOR, \\
                                      SEE_VERSION_MINOR, \\
                                      SEE_VERSION_PATCH)

#define SEE_MIN_COMPATIBLE_VERSION SEE_MAKE_VERSION(2, 0, 0)

/* =========================================================================
 * Error codes
 * ========================================================================= */

#define SEE_OK                  ((see_error_t)  0)
#define SEE_ERR_NOMEM           ((see_error_t) -1)
#define SEE_ERR_INVAL           ((see_error_t) -2)
#define SEE_ERR_OVERFLOW        ((see_error_t) -3)
#define SEE_ERR_UNDERFLOW       ((see_error_t) -4)
#define SEE_ERR_IO              ((see_error_t) -5)
#define SEE_ERR_CORRUPT         ((see_error_t) -6)
#define SEE_ERR_MISMATCH        ((see_error_t) -7)
#define SEE_ERR_UNSUPPORTED     ((see_error_t) -8)
#define SEE_ERR_TIMEOUT         ((see_error_t) -9)
#define SEE_ERR_BUSY            ((see_error_t) -10)
#define SEE_ERR_AGAIN           ((see_error_t) -11)
#define SEE_ERR_NOSPACE         ((see_error_t) -12)
#define SEE_ERR_PERMISSION      ((see_error_t) -13)
#define SEE_ERR_NOTFOUND        ((see_error_t) -14)
#define SEE_ERR_BADSTATE        ((see_error_t) -15)

static inline const char *see_strerror(see_error_t e) {
    switch (e) {
        case SEE_OK:             return "OK";
        case SEE_ERR_NOMEM:      return "out of memory";
        case SEE_ERR_INVAL:      return "invalid argument";
        case SEE_ERR_OVERFLOW:   return "buffer overflow";
        case SEE_ERR_UNDERFLOW:  return "buffer underflow";
        case SEE_ERR_IO:         return "I/O error";
        case SEE_ERR_CORRUPT:    return "data corruption";
        case SEE_ERR_MISMATCH:   return "SHA-256 mismatch";
        case SEE_ERR_UNSUPPORTED:return "unsupported operation";
        case SEE_ERR_TIMEOUT:    return "operation timed out";
        case SEE_ERR_BUSY:       return "resource busy";
        case SEE_ERR_AGAIN:      return "try again";
        case SEE_ERR_NOSPACE:    return "no space left";
        case SEE_ERR_PERMISSION: return "permission denied";
        case SEE_ERR_NOTFOUND:   return "not found";
        case SEE_ERR_BADSTATE:   return "bad state";
        default:                 return "unknown error";
    }
}

/* =========================================================================
 * Codec flags (bitfield)
 * ========================================================================= */

#define SEE_FLAG_NONE           0x00000000u
#define SEE_FLAG_LZ_ENABLED     0x00000001u
#define SEE_FLAG_TOK_PFX        0x00000002u
#define SEE_FLAG_TOK_PREV       0x00000004u
#define SEE_FLAG_TOK_PREV_ELIG  0x00000020u
#define SEE_FLAG_SPAN_PFX       0x00000040u
#define SEE_FLAG_LZ_DUAL        0x00000080u
#define SEE_FLAG_TAIL_COMP      0x00000100u
#define SEE_FLAG_FAST_MODE      0x00000200u
#define SEE_FLAG_ACCURATE_MODE  0x00000400u
#define SEE_FLAG_FULL_MODE      0x00000800u
#define SEE_FLAG_TELEMETRY      0x00001000u
#define SEE_FLAG_SHA256_CHECK   0x00002000u
#define SEE_FLAG_STREAM_MODE    0x00004000u
#define SEE_FLAG_DICT_STATIC    0x00008000u
#define SEE_FLAG_COMPRESSED_HDR 0x00010000u
#define SEE_FLAG_ALIGN_64       0x00020000u
#define SEE_FLAG_ALIGN_256      0x00040000u

#define SEE_FLAGS_PROFILE_GENERAL  (SEE_FLAG_LZ_ENABLED | SEE_FLAG_TOK_PFX)
#define SEE_FLAGS_PROFILE_TEXT     (SEE_FLAGS_PROFILE_GENERAL | SEE_FLAG_TOK_PREV_ELIG)
#define SEE_FLAGS_PROFILE_EXPERI   (SEE_FLAGS_PROFILE_GENERAL | SEE_FLAG_SPAN_PFX)

/* =========================================================================
 * Limits and buffer sizes
 * ========================================================================= */

#define SEE_MAX_BLOCK_SIZE      (1u << 26)   /* 64 MiB */
#define SEE_MIN_BLOCK_SIZE      (1u << 10)   /* 1 KiB  */
#define SEE_DEFAULT_BLOCK_SIZE  (1u << 17)   /* 128 KiB */
#define SEE_LZ_TABLE_SIZE       (1u << 20)
#define SEE_LZ_KEY_WIDTH_MIN    4u
#define SEE_LZ_KEY_WIDTH_MAX    8u
#define SEE_LZ_KEY_WIDTH_DEF    6u
#define SEE_TOPK_MIN            16u
#define SEE_TOPK_MAX            512u
#define SEE_TOPK_DEFAULT        256u
#define SEE_MOE_MAX_EXPERTS     8u
#define SEE_WEIGHT_COUNT        256u
#define SEE_SHA256_LEN          32u
#define SEE_HEADER_MAGIC        0x53454533u   /* "SEE3" */
#define SEE_HEADER_VERSION      0x00010000u

/* =========================================================================
 * Expert identifiers
 * ========================================================================= */

typedef enum see_expert_id {
    SEE_EXPERT_SEE     = 0,
    SEE_EXPERT_UNI     = 1,
    SEE_EXPERT_BI      = 2,
    SEE_EXPERT_LZ      = 3,
    SEE_EXPERT_TOKPFX  = 4,
    SEE_EXPERT_TOKPREV = 5,
    SEE_EXPERT_SPAN    = 6,
    SEE_EXPERT_LZ2     = 7,
    SEE_EXPERT_COUNT   = 8,
} see_expert_id_t;

static const char *const see_expert_names[SEE_EXPERT_COUNT] = {
    "SEE", "UNI", "BI", "LZ", "TOKPFX", "TOKPREV", "SPAN", "LZ2"
};

/* =========================================================================
 * MoE weight vector
 * ========================================================================= */

typedef struct see_weights {
    float    avg[SEE_EXPERT_COUNT];
    float    final_[SEE_EXPERT_COUNT];
    u32      n_experts;
    u32      n_bytes;
    double   total_cross_entropy;
    double   model_bpb;
    double   quant_bpb;
    double   unigram_bpb;
    u64      cycles_total;
    u32      cycles_per_kbyte;
} see_weights_t;

static inline u32 see_weights_dominant(const see_weights_t *w) {
    u32 best = 0;
    for (u32 i = 1; i < w->n_experts; ++i)
        if (w->avg[i] > w->avg[best]) best = i;
    return best;
}

static inline double see_weights_entropy_gain(const see_weights_t *w) {
    return w->unigram_bpb - w->quant_bpb;
}

/* =========================================================================
 * Range coder state
 * ========================================================================= */

#define RC_TOP      (1u << 24)
#define RC_BOT      (1u << 16)
#define RC_RANGE_MAX UINT32_MAX

typedef struct see_rc_enc {
    u64      low;
    u32      range;
    u32      cache;
    i32      carry_bytes;
    u8      *buf;
    size_t   pos;
    size_t   cap;
} see_rc_enc_t;

typedef struct see_rc_dec {
    u64      low;
    u32      range;
    u32      code;
    const u8 *buf;
    size_t    pos;
    size_t    len;
} see_rc_dec_t;

/* =========================================================================
 * LZ context
 * ========================================================================= */

typedef struct see_lz_entry {
    u32  hash;
    u32  offset;
    u16  len;
    u16  freq;
} see_lz_entry_t;

typedef struct see_lz_ctx {
    see_lz_entry_t  *table;
    u32              mask;
    u32              key_width;
    u32              n_hits;
    u32              n_miss;
    u64              total_bits_saved;
    u8               key_buf[8];
    u8               key_pos;
} see_lz_ctx_t;

typedef struct see_lz_stats {
    u32   hits;
    u32   misses;
    float hit_rate;
    float avg_bits_saved;
    u64   total_saves;
} see_lz_stats_t;

see_error_t  see_lz_init(see_lz_ctx_t *ctx, u32 key_width);
void         see_lz_reset(see_lz_ctx_t *ctx);
void         see_lz_free(see_lz_ctx_t *ctx);
float        see_lz_predict(see_lz_ctx_t *ctx, const u8 *key, u8 byte);
void         see_lz_update(see_lz_ctx_t *ctx, const u8 *key, u8 byte);
void         see_lz_get_stats(const see_lz_ctx_t *ctx, see_lz_stats_t *out);

/* =========================================================================
 * Token-prefix expert
 * ========================================================================= */

#define TOKPFX_TABLE_SIZE  (1u << 18)
#define TOKPFX_KEY_MASK    (TOKPFX_TABLE_SIZE - 1u)
#define TOKPFX_ALPHA       0.125f
#define TOKPFX_MIN_FREQ    2u

typedef struct see_tokpfx_ctx {
    float   *table;      /* [256][TOKPFX_TABLE_SIZE]: prediction[256] per key */
    u32      key;
    u8       key_len;
    bool     in_token;
    u32      n_predictions;
    double   cumulative_bpb;
} see_tokpfx_ctx_t;

see_error_t see_tokpfx_init(see_tokpfx_ctx_t *ctx);
void        see_tokpfx_free(see_tokpfx_ctx_t *ctx);
void        see_tokpfx_predict(see_tokpfx_ctx_t *ctx, float out[256]);
void        see_tokpfx_update(see_tokpfx_ctx_t *ctx, u8 byte);

/* =========================================================================
 * Span expert
 * ========================================================================= */

typedef enum see_span_type {
    SEE_SPAN_NONE      = 0,
    SEE_SPAN_BACKTICK  = 1,
    SEE_SPAN_DOLLAR    = 2,
    SEE_SPAN_DOUBLE_BT = 3,
    SEE_SPAN_TRIPLE_BT = 4,
} see_span_type_t;

typedef struct see_span_ctx {
    see_span_type_t  active;
    u32              depth;
    u32              n_spans_seen;
    u32              n_bytes_in_span;
    bool             eligible;
    u8               last_bytes[4];
    float           *prefix_table;
    u32              pfx_key;
} see_span_ctx_t;

see_error_t see_span_init(see_span_ctx_t *ctx);
void        see_span_free(see_span_ctx_t *ctx);
bool        see_span_is_eligible(const see_span_ctx_t *ctx);
void        see_span_predict(see_span_ctx_t *ctx, float out[256]);
void        see_span_update(see_span_ctx_t *ctx, u8 byte);

/* =========================================================================
 * MoE engine
 * ========================================================================= */

typedef struct see_moe_config {
    float    eta;
    float    share;
    float    init_weight;
    u32      n_experts;
    u32      topk;
    see_flags_t flags;
} see_moe_config_t;

typedef struct see_moe_state {
    float    w[SEE_MOE_MAX_EXPERTS];
    float    w_sum;
    float    w_sum_sq;
    u32      n_experts;
    u64      n_updates;
    float    eta;
    float    share;
} see_moe_state_t;

void see_moe_init(see_moe_state_t *s, const see_moe_config_t *cfg);
void see_moe_predict(const see_moe_state_t *s,
                     const float *experts[],
                     float out[256]);
void see_moe_update(see_moe_state_t *s,
                    const float *experts[],
                    u8 observed);
void see_moe_update_gated(see_moe_state_t *s,
                          const float *experts[],
                          const bool   eligible[],
                          u8 observed);
float see_moe_entropy(const see_moe_state_t *s,
                      const float *experts[],
                      u8 byte);

/* =========================================================================
 * Archive header (on-disk format)
 * ========================================================================= */

#define SEE_ARCHIVE_HDR_SIZE  64u

typedef struct __attribute__((packed)) see_archive_hdr {
    u32  magic;          /* SEE_HEADER_MAGIC = 0x53454533 */
    u32  version;        /* SEE_HEADER_VERSION */
    u32  flags;          /* SEE_FLAG_* bitfield */
    u32  lz_key_width;
    u32  topk;
    u32  reserved_a;
    u64  original_size;
    u64  compressed_size;
    u8   sha256[SEE_SHA256_LEN];  /* SHA-256 of original data */
    u8   reserved_b[4];
} see_archive_hdr_t;

/* static_assert ensures no padding slips in */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(see_archive_hdr_t) == SEE_ARCHIVE_HDR_SIZE,
               "see_archive_hdr_t must be exactly 64 bytes");
#endif

see_error_t see_hdr_write(see_archive_hdr_t *hdr, u8 buf[SEE_ARCHIVE_HDR_SIZE]);
see_error_t see_hdr_read(see_archive_hdr_t *hdr, const u8 buf[SEE_ARCHIVE_HDR_SIZE]);
see_error_t see_hdr_validate(const see_archive_hdr_t *hdr);

/* =========================================================================
 * Top-level codec API
 * ========================================================================= */

typedef struct see_codec_ctx see_codec_ctx_t;

see_error_t see_codec_create(see_codec_ctx_t       **ctx,
                              const see_moe_config_t *cfg);
void        see_codec_destroy(see_codec_ctx_t *ctx);

see_error_t see_encode(see_codec_ctx_t  *ctx,
                       const u8         *src,
                       size_t            src_len,
                       u8               *dst,
                       size_t           *dst_len);

see_error_t see_decode(see_codec_ctx_t  *ctx,
                       const u8         *src,
                       size_t            src_len,
                       u8               *dst,
                       size_t            dst_len);

see_error_t see_encode_file(see_codec_ctx_t *ctx,
                             const char      *src_path,
                             const char      *dst_path);

see_error_t see_decode_file(see_codec_ctx_t *ctx,
                             const char      *src_path,
                             const char      *dst_path);

see_error_t see_audit(see_codec_ctx_t  *ctx,
                      const u8         *data,
                      size_t            len,
                      see_weights_t    *out_weights);

/* =========================================================================
 * Utility: fast integer log2
 * ========================================================================= */

static inline u32 see_log2_u32(u32 x) {
#if defined(__GNUC__) || defined(__clang__)
    return x ? (31u - (u32)__builtin_clz(x)) : 0u;
#elif defined(_MSC_VER)
    unsigned long idx;
    return _BitScanReverse(&idx, x) ? (u32)idx : 0u;
#else
    u32 r = 0;
    while (x >>= 1) ++r;
    return r;
#endif
}

static inline u64 see_log2_u64(u64 x) {
#if defined(__GNUC__) || defined(__clang__)
    return x ? (63u - (u32)__builtin_clzll(x)) : 0u;
#elif defined(_MSC_VER)
    unsigned long idx;
    return _BitScanReverse64(&idx, x) ? (u64)idx : 0u;
#else
    u64 r = 0;
    while (x >>= 1) ++r;
    return r;
#endif
}

/* =========================================================================
 * Utility: aligned allocation
 * ========================================================================= */

#if defined(_WIN32)
#  include <malloc.h>
#  define see_aligned_alloc(align, size)  _aligned_malloc((size), (align))
#  define see_aligned_free(ptr)           _aligned_free(ptr)
#else
#  include <stdlib.h>
static inline void *see_aligned_alloc_impl(size_t align, size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, align, size) != 0) return NULL;
    return ptr;
}
#  define see_aligned_alloc(align, size)  see_aligned_alloc_impl((align), (size))
#  define see_aligned_free(ptr)           free(ptr)
#endif

/* =========================================================================
 * Utility: min / max / clamp
 * ========================================================================= */

#define SEE_MIN(a, b)       ((a) < (b) ? (a) : (b))
#define SEE_MAX(a, b)       ((a) > (b) ? (a) : (b))
#define SEE_CLAMP(v, lo, hi) SEE_MIN(SEE_MAX((v), (lo)), (hi))
#define SEE_ALIGN_UP(v, a)  (((v) + (a) - 1u) & ~((a) - 1u))
#define SEE_ALIGN_DN(v, a)  ((v) & ~((a) - 1u))
#define SEE_IS_POW2(x)      ((x) && !((x) & ((x) - 1)))
#define SEE_ARRAY_LEN(a)    (sizeof(a) / sizeof((a)[0]))
#define SEE_UNUSED(x)       ((void)(x))

/* =========================================================================
 * Utility: byte-order swaps
 * ========================================================================= */

static inline u32 see_bswap32(u32 x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap32(x);
#elif defined(_MSC_VER)
    return _byteswap_ulong(x);
#else
    return ((x >> 24) & 0xFFu)
         | ((x >>  8) & 0xFF00u)
         | ((x <<  8) & 0xFF0000u)
         | ((x << 24) & 0xFF000000u);
#endif
}

static inline u64 see_bswap64(u64 x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap64(x);
#elif defined(_MSC_VER)
    return _byteswap_uint64(x);
#else
    return ((u64)see_bswap32((u32)(x      ))  << 32)
         | ((u64)see_bswap32((u32)(x >> 32)));
#endif
}

/* =========================================================================
 * Telemetry / per-byte CSV
 * ========================================================================= */

typedef struct see_telemetry_entry {
    u64     byte_idx;
    u8      byte_val;
    float   bits_used;
    float   w[SEE_MOE_MAX_EXPERTS];
    u8      dominant_expert;
    bool    tok_eligible;
    bool    span_eligible;
} see_telemetry_entry_t;

typedef struct see_telemetry_ctx {
    see_telemetry_entry_t  *entries;
    size_t                  cap;
    size_t                  len;
    FILE                   *csv_file;
    bool                    enabled;
} see_telemetry_ctx_t;

see_error_t see_telemetry_init(see_telemetry_ctx_t *t, const char *csv_path);
void        see_telemetry_record(see_telemetry_ctx_t *t,
                                 const see_telemetry_entry_t *e);
void        see_telemetry_flush(see_telemetry_ctx_t *t);
void        see_telemetry_free(see_telemetry_ctx_t *t);

/* =========================================================================
 * Benchmark / profiling helpers
 * ========================================================================= */

typedef struct see_bench_result {
    double   bpb;
    double   compress_ratio;
    u64      cycles_total;
    double   cycles_per_byte;
    double   throughput_mbps;
    u64      wall_ns;
    size_t   original_bytes;
    size_t   compressed_bytes;
} see_bench_result_t;

see_error_t see_bench_run(see_codec_ctx_t     *ctx,
                           const u8            *data,
                           size_t               len,
                           int                  n_warmup,
                           int                  n_runs,
                           see_bench_result_t  *out);

void see_bench_print(const see_bench_result_t *r, const char *label);

#ifdef __cplusplus
}
#endif

#endif /* SEE_SYNTH_CORPUS_H */
"""


# ---------------------------------------------------------------------------
# 3. project_notes_it.txt  —  Italian text from docs + promessi_sposi slice
# ---------------------------------------------------------------------------

IT_DOCS_PATHS = [
    "docs/phases/walkthrough_phase22.md",
    "docs/phases/walkthrough_phase22b.md",
    "docs/phases/walkthrough_phase21.md",
    "docs/phases/walkthrough_phase20C.md",
    "docs/phases/phase20b_walkthrough.md",
    "docs/phases/phase20_walkthrough.md",
    "docs/phases/phase19_walkthrough.md",
    "docs/phases/phase18_walkthrough.md",
]

# ---------------------------------------------------------------------------
# 2b. gen_c_header_synth — programmatic C/C++ header to reach ~100 KB
# ---------------------------------------------------------------------------

_C_SUBSYSTEMS = [
    "ring_buf", "arena", "hash_map", "skip_list", "bloom_filter",
    "bit_vec", "heap", "trie", "graph", "matrix",
    "thread_pool", "lock_free_queue", "rcu", "epoch_gc", "hazard_ptr",
    "raft_log", "wal", "snapshot", "compaction", "bloom_table",
    "codec_ctx", "stream_enc", "stream_dec", "checksum", "crc32",
    "simd_util", "avx2_ops", "entropy_coder", "range_coder", "ans_coder",
    "model_ctx", "predictor", "mixer", "weight_vec", "softmax",
    "lz_chain", "lz_table", "lz_match", "lz_emit", "lz_decode",
]
_C_TYPES = ["uint8_t", "uint16_t", "uint32_t", "uint64_t",
            "int32_t", "int64_t", "size_t", "float", "double", "bool"]
_C_QUALS = ["const", "volatile", "restrict", "", "", ""]
_C_OP_SUFFIXES = ["init", "create", "destroy", "reset", "clone",
                  "push", "pop", "peek", "insert", "remove",
                  "lookup", "find", "update", "flush", "drain",
                  "encode", "decode", "compress", "decompress",
                  "read", "write", "seek", "tell", "size",
                  "lock", "unlock", "trylock", "wait", "signal",
                  "serialize", "deserialize", "validate", "hash", "cmp"]


def gen_c_header_synth():
    rng2 = random.Random(42)
    parts = [C_HEADER_TEMPLATE]

    for sub in _C_SUBSYSTEMS:
        SU = sub.upper()
        parts.append(f"\n/* {'='*70}")
        parts.append(f" * Subsystem: {sub}")
        parts.append(f" * {'='*70} */\n")

        # Macros
        n_macros = rng2.randint(4, 10)
        for i in range(n_macros):
            name  = f"{SU}_{rng2.choice(['MAX','MIN','DEFAULT','LIMIT','SIZE','CAP','SHIFT','MASK','BITS','ALIGN'])}_{i}"
            val   = rng2.choice([
                f"(1u << {rng2.randint(8,24)})",
                f"{rng2.randint(4,256)}",
                f"0x{rng2.randint(0,0xFFFF):04X}u",
                f"({rng2.randint(1,32)} * 1024 * 1024)",
            ])
            parts.append(f"#define {name:<44} {val}")
        parts.append("")

        # Error codes
        n_errs = rng2.randint(3, 7)
        for i in range(n_errs):
            ename = f"{SU}_ERR_{rng2.choice(['NOMEM','INVAL','OVERFLOW','CORRUPT','IO','NOTFOUND','BUSY','AGAIN','BADSTATE','LIMIT'])}_{i}"
            parts.append(f"#define {ename:<44} ((see_error_t) {-i-1})")
        parts.append("")

        # Enum
        parts.append(f"typedef enum {sub}_state {{")
        states = [f"    {SU}_STATE_{s.upper()}" for s in
                  ["idle", "running", "paused", "error", "draining", "flushing"]]
        for s in states:
            parts.append(f"{s},")
        parts.append(f"    {SU}_STATE_COUNT,")
        parts.append(f"}} {sub}_state_t;\n")

        # Main struct
        n_fields = rng2.randint(6, 14)
        parts.append(f"typedef struct {sub} {{")
        for _ in range(n_fields):
            ftype = rng2.choice(_C_TYPES)
            fname = rng2.choice(["buf", "ptr", "len", "cap", "pos", "idx",
                                  "count", "mask", "flags", "state",
                                  "head", "tail", "next", "prev",
                                  "size", "stride", "offset", "align",
                                  "n_items", "n_alloc", "n_free", "n_err"])
            suffix = rng2.choice(["", "_a", "_b", f"_{_}"])
            qual   = rng2.choice(_C_QUALS)
            stars  = rng2.choice(["", "*", "*"])
            qpart  = (qual + " ") if qual else ""
            parts.append(f"    {qpart}{ftype}{stars:<10} {fname}{suffix};")
        parts.append(f"    {sub}_state_t  state;")
        parts.append(f"    uint64_t       serial;")
        parts.append(f"}} {sub}_t;\n")

        # Stats struct
        parts.append(f"typedef struct {sub}_stats {{")
        for stat in ["n_ops", "n_errors", "n_alloc", "n_free",
                     "bytes_in", "bytes_out", "cycles_total"]:
            parts.append(f"    uint64_t   {stat};")
        parts.append(f"    double     throughput_mbps;")
        parts.append(f"    float      hit_rate;")
        parts.append(f"}} {sub}_stats_t;\n")

        # Function prototypes
        for op in rng2.sample(_C_OP_SUFFIXES, k=rng2.randint(6, 12)):
            ret   = rng2.choice(["see_error_t", "see_error_t", "void",
                                  "uint32_t", "uint64_t", "bool", "size_t"])
            args  = []
            args.append(f"{sub}_t *ctx")
            if op not in ("init", "create", "destroy"):
                t2 = rng2.choice(_C_TYPES)
                q2 = rng2.choice(["const ", ""])
                args.append(f"{q2}{t2} *data")
                args.append(f"size_t len")
            if ret != "void":
                args.append(f"{ret} *out")
            arg_str = ", ".join(args[:rng2.randint(1, len(args))])
            parts.append(f"{ret:<14} {sub}_{op}({arg_str});")
        parts.append("")

    parts.append("\n#endif /* SEE_SYNTH_CORPUS_EXTENDED_H */\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------

def strip_markdown(text):
    text = re.sub(r"```[^`]*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"^\s*#+ ", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s*[-*>|]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\$\$[^$]+\$\$", "", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$]+\$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def gen_project_notes_it():
    parts = []
    for rel in IT_DOCS_PATHS:
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                raw = f.read()
            parts.append(strip_markdown(raw))

    project_docs = "\n\n".join(parts)

    # Fill to ~150 KB with a promessi_sposi slice (bytes 350000-500000)
    ps_path = os.path.join(DATA, "promessi_sposi.txt")
    ps_slice = ""
    if os.path.exists(ps_path):
        with open(ps_path, encoding="utf-8", errors="replace") as f:
            ps_text = f.read()
        # Use a middle slice that doesn't overlap with existing corpora
        start = 350000
        need  = max(0, 150000 - len(project_docs))
        ps_slice = ps_text[start : start + need]

    combined = project_docs + "\n\n" + ps_slice
    return combined


# ---------------------------------------------------------------------------
# 4. repo_markdown_mixed.md  —  Diverse markdown (not like markdown_docs.md)
# ---------------------------------------------------------------------------

REPO_MARKDOWN = """\
# Silicon Entropy Engine — Project Overview

> A streaming Mixture-of-Experts entropy coder for CPU-native lossless compression.
> Status: **Phase 29A in progress** · Build: passing · Version: 0.9-dev

## Table of Contents

- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Expert Profiles](#expert-profiles)
- [Benchmark Results](#benchmark-results)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Changelog](#changelog)
- [Contributing](#contributing)

---

## Introduction

The Silicon Entropy Engine (SEE) is a research-grade lossless data compressor
that routes each byte through a dynamic mixture of specialized prediction experts.
Unlike block-based compressors (zstd, brotli, lz4), SEE operates in a single
streaming pass with no look-ahead, making it suitable for real-time encoding
pipelines.

### Core idea

Each byte $x_t$ is predicted by $N$ experts. A softmax-weighted mixture
$P_{mix} = \\sum_i w_i P_i$ produces the final probability. The weights $w_i$
are updated online via exponential loss:

$$w_i \\leftarrow w_i \\cdot \\exp(-\\eta \\cdot \\ell_i)$$

where $\\ell_i = -\\log_2 P_i(x_t)$ is the coding cost of expert $i$ on the
observed byte.

### Why streaming?

- **Zero latency**: the encoder never buffers beyond a single byte.
- **Adaptive**: weights converge to the right expert within a few hundred bytes.
- **Composable**: new experts can be added without retraining existing ones.
- **Transparent**: the weight vector is a live readout of domain structure.

---

## Quick Start

### Prerequisites

```
gcc >= 12.0 or clang >= 15.0
Windows: MSVC 2022 or mingw64
Python >= 3.9  (for benchmark scripts)
make (optional, build.bat on Windows)
```

### Build

```bash
# Linux / macOS
make

# Windows
build.bat

# Manual (single TU)
gcc -O3 -march=native -o see see.c src/see_codec.c -lm
```

### Encode a file

```bash
./see encode input.txt output.see --weights weights/entropy_weights_factors_r16.bin
./see decode output.see restored.txt --weights weights/entropy_weights_factors_r16.bin
```

### Run an audit (BPB + weight diagnostics)

```bash
./see audit input.txt --weights weights/entropy_weights_factors_r16.bin \\
    --blend moe --expert-profile general
```

Expected output for a natural-language file:

```
=== SEE Audit: input.txt ===
Bytes evaluated: 218553
Unigram BPB:     3.9827
Model BPB:       2.4656
Quantized BPB:   2.4741
Cycles/byte:     58751.9
LZ key width:    6 bytes + tok-prefix expert
--- Per-expert BPB ---
SEE Only:        5.1674
UNI Only:        3.9949
BI  Only:        2.9462
LZ  Only:        4.2525  (key=6 bytes)
TOKPFX Only:     2.6219
--- MoE weights ---
Avg  [SEE UNI BI  LZ  TOKPFX]: 0.0031 0.0126 0.0282 0.0054 0.9507
Final[SEE UNI BI  LZ  TOKPFX]: 0.0021 0.0041 0.0260 0.0084 0.9594
```

---

## Architecture

### Expert taxonomy

| Expert   | Type        | Key signal                    | Default profile |
|----------|-------------|-------------------------------|-----------------|
| SEE      | Neural/static | Pre-trained byte distribution | all             |
| UNI      | Unigram     | Per-byte frequency            | all             |
| BI       | Bigram      | Byte-pair frequency           | all             |
| LZ       | LZ-hash     | Repeated substrings (key=6B)  | general, text   |
| TOKPFX   | Token-prefix| Inside-token byte prediction  | general, text   |
| TOKPREV  | Token-trans | After-token transition (elig.)| text            |
| SPAN     | Span-prefix | Backtick/dollar spans (exp.)  | experimental    |

### MoE update rule

The fixed-share update prevents expert weight collapse:

```c
// Normalize + fixed-share
float sum = 0.0f;
for (int i = 0; i < n; ++i) {
    w[i] *= expf(-eta * loss[i]);
    sum  += w[i];
}
for (int i = 0; i < n; ++i) {
    w[i] = (1.0f - share) * (w[i] / sum) + share / (float)n;
}
```

### Range coder

SEE uses a standard carry-less range coder (Schindler 1998) with 24-bit top,
16-bit bottom, and a carry-byte buffer. The codec is symmetric: encode and
decode share the same probability tables and update steps.

```
RC state: { low: u64, range: u32, code: u32 }
Encode: low += range * cum_freq / total
        range = range * freq / total
Renormalize when range < RC_BOT (2^16)
```

---

## Expert Profiles

### `--expert-profile general` (default)

Best for: all domains. Recommended for unknown input.

Experts active: SEE + UNI + BI + LZ6 + TOKPFX

```
Corpus            BPB
──────────────────────
natural_text     2.474
markdown_docs    3.887
c_code           1.940
shuffled         5.012
```

### `--expert-profile text`

Best for: natural language prose (Italian, English, French…).
Adds: `TOK_PREV_ELIG` — gated token-transition expert, fires only when
the previous byte ended an alphanumeric/macro token.

```
Corpus            BPB    Δ vs general
──────────────────────────────────────
natural_text     2.429   -0.045  ✓
markdown_docs    3.890   +0.003  (MoE silences TOKPREV on MD)
c_code           1.942   +0.002  (same)
shuffled         5.012   +0.000
```

> **Rule**: use `text` profile only when input is confirmed prose.
> For mixed or unknown content, prefer `general`.

### `--span-pfx` (experimental)

Adds: span-prefix expert, event-driven (fires only inside backtick/dollar
spans). Phase 28B tribunal result: **rejected** for markdown gain
(+0.0002 BPB vs criterion 0.025). Remains available as a flag for
future investigation.

---

## Benchmark Results

Phase 29A robustness matrix (in progress):

| Corpus              | general | text  | Δ text  | no-token | Δ no-tok |
|---------------------|---------|-------|---------|----------|----------|
| natural_text.txt    | 2.474   | 2.429 | -0.045  | TBD      | TBD      |
| markdown_docs.md    | 3.887   | 3.890 | +0.003  | TBD      | TBD      |
| c_code.c            | 1.940   | 1.942 | +0.002  | TBD      | TBD      |
| shuffled.bin        | 5.012   | 5.012 | 0.000   | TBD      | TBD      |
| json_synth.json     | TBD     | TBD   | TBD     | TBD      | TBD      |
| c_header_synth.h    | TBD     | TBD   | TBD     | TBD      | TBD      |
| project_notes_it    | TBD     | TBD   | TBD     | TBD      | TBD      |
| repo_markdown_mixed | TBD     | TBD   | TBD     | TBD      | TBD      |
| log_synth.log       | TBD     | TBD   | TBD     | TBD      | TBD      |

_Table will be filled in by phase29a_tribunal.py output._

---

## API Reference

### `see_encode`

```c
see_error_t see_encode(
    see_codec_ctx_t  *ctx,
    const uint8_t    *src,
    size_t            src_len,
    uint8_t          *dst,
    size_t           *dst_len   /* in: capacity, out: bytes written */
);
```

Encodes `src_len` bytes from `src` into the pre-allocated buffer `dst`.
`dst_len` must be at least `src_len + SEE_ARCHIVE_HDR_SIZE + 256` on entry.
Returns `SEE_OK` on success, `SEE_ERR_OVERFLOW` if output buffer is too small.

### `see_decode`

```c
see_error_t see_decode(
    see_codec_ctx_t  *ctx,
    const uint8_t    *src,
    size_t            src_len,
    uint8_t          *dst,
    size_t            dst_len   /* must match original size from header */
);
```

Decodes a SEE3 archive. The original size is embedded in the archive header;
`dst_len` must match exactly or `SEE_ERR_MISMATCH` is returned.

### `see_audit`

```c
see_error_t see_audit(
    see_codec_ctx_t  *ctx,
    const uint8_t    *data,
    size_t            len,
    see_weights_t    *out_weights  /* optional, may be NULL */
);
```

Performs a dry-run encode (no output file), collecting BPB statistics and
MoE weight trajectories into `out_weights`.

---

## Configuration

### CLI options

```
--blend moe|<lambda>       Blend mode. moe = dynamic, float = fixed mix.
--eta   <float>            MoE learning rate (default 0.03).
--share <float>            Fixed-share coefficient (default 0.001).
--topk  <int>              SEE top-k candidates (default 256).
--speed full|accurate|fast Quantizer speed/accuracy tradeoff.
--expert-profile <name>    Expert set: general | text | experimental.
--lz-key <4|6|8>           LZ context key width in bytes.
--tok-prefix               Enable TOKPFX expert manually.
--tok-prev-elig            Enable gated TOK_PREV expert manually.
--span-pfx                 Enable experimental span-prefix expert.
--no-lz                    Disable LZ (3-expert mode: SEE+UNI+BI).
```

### Environment variables

| Variable              | Default | Description                        |
|-----------------------|---------|------------------------------------|
| `SEE_WEIGHTS_PATH`    | (none)  | Default weights file path          |
| `SEE_MAX_THREADS`     | 1       | Number of worker threads (future)  |
| `SEE_LOG_LEVEL`       | INFO    | Log verbosity (DEBUG/INFO/WARN)    |
| `SEE_TELEMETRY_DIR`   | (none)  | Directory for per-byte CSV output  |

---

## Changelog

### v0.8 — Phase 28 (2026-05-25)

- **Added** `--span-pfx` expert (event-driven, backtick/dollar gated)
- **Added** `--span-pfx-mute` ablation flag
- **Tribunal result**: SPANPFX rejected — markdown gain +0.0002 BPB
  (criterion 0.025). Flag remains available as experimental.
- **Fixed** SHA-256 symmetry bug: `SEE_FLAG_TOK_PREV_ELIG` was missing
  from archive header encoding, causing decoder state mismatch.

### v0.7 — Phase 27F (2026-05-25)

- **Added** `--expert-profile text` with `TOK_PREV_ELIG` gated expert
- **Validated** eligibility semantics: W_when_elig = 0.8030,
  natural_text gain −0.045 BPB, shuffled correctly muted
- **Fixed** SHA-256 header bug (same session as v0.8 hotfix)

### v0.6 — Phase 27C/27E (2026-05-24)

- **Cartography**: ALNUM_START is the natural_text gap (0.696 BPB, 22.9%)
- **Cartography**: markdown gap = MoE convergence lag +1.3 BPB uniform
- **Added** `--tok-prev-elig` manual flag

### v0.5 — Phase 24 (2026-05-20)

- **Sealed** hybrid architecture: LZ6 + TOKPFX as stable base
- **Removed** experimental LSTM and PPM paths
- **Added** `--expert-profile` CLI flag replacing ad-hoc combinations

### v0.4 — Phase 23 (2026-05-18)

- **Added** compact LZ implementation with micro-audit harness
- **Benchmarked**: LZ6 +0.5 BPB over bigram on c_code domain

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b phase-NN-description`.
3. Run the full tribunal before opening a PR:

```bash
python scripts/phase29a_tribunal.py
```

4. Ensure no historical domain regresses beyond **0.005 BPB**.
5. Update `docs/phases/` with a walkthrough of your changes.
6. Open a PR with the tribunal output attached.

### Code style

- C11, no VLAs, no alloca.
- All functions return `see_error_t`; no bare `void` that can fail.
- Every header is self-contained (no implicit includes).
- No `goto` except cleanup paths.
- Comment only non-obvious invariants.

### Commit format

```
Silicon Entropy Engine VX.Y — Phase NN: short description

Body: what changed and why. Reference tribunal results if relevant.
```

---

_Last updated: 2026-05-25 — Phase 29A corpus generation_
"""

# Pad to ~110 KB by appending a second technical section
REPO_MARKDOWN_EXTRA = """
---

## Appendix A: Mathematical Background

### Arithmetic coding fundamentals

Given a probability model $P$ over the alphabet $\\{0, \\ldots, 255\\}$, the
arithmetic coder assigns to a sequence $x_1, x_2, \\ldots, x_n$ a code of
length $\\lceil -\\log_2 \\prod_t P(x_t | x_{<t}) \\rceil + 1$ bits. The
per-symbol average is the cross-entropy:

$$H_P(X) = -\\frac{1}{n} \\sum_{t=1}^n \\log_2 P(x_t | x_{<t}) \\quad [\\text{bits per byte, BPB}]$$

The true entropy of the source is a lower bound: $H_P(X) \\geq H(X)$, with
equality iff $P$ matches the true distribution.

### Mixture of experts bound

If the true distribution is a mixture $P^* = \\sum_i \\alpha_i P_i$ with
$\\alpha_i > 0$, then the online MoE achieves:

$$H_{\\text{MoE}}(X) \\leq H_{P^*}(X) + \\frac{\\ln(1/\\alpha_{\\min})}{n} + O(1/n)$$

The second term is the *price of ignorance* about the mixing coefficients.
With $n = 100{,}000$ bytes and $\\alpha_{\\min} = 0.001$, this is $0.0001$
BPB — negligible. This is why online MoE is competitive with optimal
offline mixture at corpus scale.

### Fixed-share and the sleeping expert regret

The standard Hedge algorithm (Freund & Schapire 1997) minimizes regret
against the best single expert. Fixed-share (Herbster & Warmuth 1998)
minimizes regret against the best *sequence* of experts with at most $s$
switches. The regret bound is:

$$R_T \\leq \\sqrt{\\frac{T \\ln N}{2}} + O(s \\ln T)$$

For compression, this means SEE adapts to domain switches within a file
(e.g., mixed code and prose) with bounded overhead.

---

## Appendix B: LZ Hashing Details

The LZ expert uses a rolling hash over the last $k$ bytes (default $k=6$):

```c
static inline uint32_t lz_hash(const uint8_t *key, uint32_t k) {
    uint32_t h = 2166136261u;  // FNV-1a offset basis
    for (uint32_t i = 0; i < k; ++i) {
        h ^= key[i];
        h *= 16777619u;        // FNV prime
    }
    return h;
}
```

The table stores the *predicted next byte distribution*: an array of 256
counts, incremented on each observation. At prediction time, the counts
are normalised to a probability vector. The table has $2^{20}$ slots
(~4 MB for 32-bit counts).

Collision handling: open addressing with linear probing. Eviction uses
a frequency-weighted LRU approximation — entries with low `freq` field
are overwritten by new entries that hash to the same slot.

Key width selection:
- $k=4$: fast hashing, high collision rate, useful only for very small files
- $k=6$: default; good balance of coverage and collision rate
- $k=8$: best for highly repetitive structured data (binary logs, compressed
  inner layers); higher memory footprint and slower hash

---

## Appendix C: Token-Prefix Expert Design

The token-prefix (TOKPFX) expert exploits a regularity in natural text and
source code: within a token (a contiguous run of alphanumeric characters or
underscore), the continuation probability is very different from the
transition probability.

**Example**: given the prefix `uint8`, the next byte is highly likely to be
`_` (completing `uint8_t`) or a digit/letter (completing a longer token) —
not a space or newline. This is a sharp conditional distribution that the
bigram expert struggles to capture because it only sees one byte of context.

The TOKPFX key is the current token prefix (up to 4 bytes), hashed to a
$2^{18}$-slot table. A separate table tracks whether the byte ended a token,
which is used by the downstream TOK_PREV_ELIG expert.

Learned weights on natural text:

```
Avg  [SEE UNI BI  LZ  TOKPFX]: 0.003 0.013 0.028 0.005 0.951
```

TOKPFX carries 95% of the predictive weight. The other experts contribute
a small regularizing prior that prevents pathological behaviour on
out-of-distribution bytes.

---

## Appendix D: Evaluation Methodology

### Why BPB and not compression ratio?

BPB (bits per byte) measures *information-theoretic efficiency* independently
of file size. Compression ratio conflates codec overhead (headers, checksums)
with actual prediction quality. For audit purposes, we always compare
quantized BPB because it includes the range-coder quantization step.

### Tribunal protocol

Each phase introduces a candidate expert. The tribunal runs three
configurations:

1. **BASE** — current best profile (no new expert)
2. **MUTE** — new expert added but weight is frozen at uniform (measures
   the MoE tax of adding a new arm)
3. **ACTIVE** — new expert with full online learning

Promotion criteria (from Phase 27 onwards):
- Primary gain ≥ 0.025 BPB on target domain
- Regression ≤ 0.005 BPB on all other domains
- Shuffled (random) domain: weight stays at floor (MoE ignores it)
- SHA-256 encode/decode symmetry: PASS on all test files

### Corpus selection

The five historical corpora cover the main structural categories:

| File              | Domain            | Key structure           |
|-------------------|-------------------|-------------------------|
| natural_text.txt  | English prose     | Word-level patterns     |
| markdown_docs.md  | Markdown tech doc | Mixed prose + code      |
| c_code.c          | C source          | Symbol-heavy, low entropy|
| shuffled.bin      | Random permutation| Zero structure (oracle) |
| promessi_sposi.txt| Italian literary  | Long-range coherence    |

Phase 29A adds five new corpora to test generalization beyond the
training distribution.
"""


def _gen_markdown_faq():
    """Generates a FAQ section with enough volume to pad markdown to ~100KB."""
    rng2 = random.Random(99)
    questions = [
        ("What is BPB?",
         "BPB stands for Bits Per Byte. It measures how many bits the coder "
         "uses on average to represent each byte of the input. Lower is better. "
         "The theoretical minimum is the Shannon entropy of the source."),
        ("Why not use zstd or brotli instead?",
         "zstd and brotli are excellent general-purpose compressors optimized "
         "for throughput and compression ratio on known workloads. SEE is a "
         "research platform for studying online mixture-of-experts prediction. "
         "The goal is understanding *why* certain byte patterns are predictable, "
         "not shipping the smallest archive."),
        ("What does 'streaming' mean in this context?",
         "Streaming means the encoder processes one byte at a time without "
         "buffering future input. At each step, the current probability model "
         "depends only on bytes seen so far. This constrains the design: no "
         "block-level statistics, no two-pass preprocessing, no dictionary "
         "pre-building from the full input."),
        ("Is SEE lossless?",
         "Yes. The range coder is exact: decode(encode(x)) = x for all inputs. "
         "SHA-256 symmetry is verified after every tribunal as a non-negotiable "
         "correctness criterion."),
        ("Can SEE compress random data?",
         "No. Random data has entropy equal to 8 BPB (one bit per bit). SEE "
         "achieves 5.01 BPB on shuffled.bin — above 5, worse than uncompressed "
         "in practice due to header overhead. This is expected and correct: "
         "the MoE assigns near-uniform weights to all experts, and the "
         "quantization overhead adds ~0.01 BPB."),
        ("What is the 'shuffled' corpus?",
         "shuffled.bin is a byte-level permutation of c_code.c. The marginal "
         "distribution is identical to c_code (same byte frequencies), but all "
         "sequential correlations are destroyed. It serves as an oracle: "
         "any gain a sequential model claims on shuffled.bin is a false positive."),
        ("Why does TOKPFX help so much on natural text?",
         "English and Italian prose consists largely of known words from a "
         "limited vocabulary. Within a word (a token), the next character is "
         "highly predictable from the current prefix: given 'uniq', 'u' is "
         "almost certainly next ('unique', 'uniqueness'). The TOKPFX table "
         "stores a separate 256-slot probability vector for each observed "
         "within-token prefix, updated online. On natural_text.txt, it captures "
         "~95% of the total predictive weight."),
        ("Why did SPANPFX fail on markdown?",
         "The content inside backtick and dollar spans in technical markdown "
         "is extremely varied: _mm256i, uint8_t, \\\\mathbb{R}^n, sin(x), "
         "my_func(), etc. A prefix-hash LZ dictionary needs to see the same "
         "prefix multiple times to build a useful distribution. On a 126KB "
         "markdown file, each distinct span prefix appears fewer than 3 times "
         "on average. The table stays sparse and the MoE correctly assigns it "
         "near-zero weight (W_SPAN = 0.0145 when eligible)."),
        ("What is the fixed-share coefficient?",
         "The fixed-share coefficient (default 0.001) redistributes a small "
         "fraction of the total weight uniformly among all experts after each "
         "update. Without it, a losing expert's weight decays exponentially "
         "and can reach numerical zero, preventing recovery. With share=0.001, "
         "even a fully silenced expert retains 0.1% / N weight at minimum."),
        ("How is 'dominant expert' defined?",
         "The dominant expert is the one with the highest average weight over "
         "the entire audit run. Average weight is more informative than final "
         "weight because final weight reflects only the last few bytes. "
         "On natural text, TOKPFX dominates with avg~0.95. On c_code, TOKPFX "
         "still dominates but with a lower avg~0.87 due to more non-token bytes."),
        ("Does SEE support parallel encoding?",
         "Not currently. The streaming MoE state (weight vector, LZ table, "
         "token context) is sequential by design. Parallel encoding would "
         "require either blocking (fixed-size chunks with independent state) "
         "or a checkpoint protocol to synchronize weight vectors across threads."),
        ("What is the range coder's symbol precision?",
         "SEE uses 12-bit probability quantization: probabilities are rounded "
         "to the nearest multiple of 1/4096. The quantization loss is "
         "approximately 0.001-0.003 BPB depending on the sharpness of the "
         "distribution (sharper = more loss from rounding peaks). This is "
         "the gap between 'Model BPB' and 'Quantized BPB' in audit output."),
        ("What happened to the PPM and LSTM experiments?",
         "Both were tested in Phases 18-21 and discarded. PPM order-5 "
         "performed well on c_code but consumed O(|alphabet|^5) memory and "
         "was impractical. LSTM required offline training and could not adapt "
         "online without catastrophic forgetting. The MoE approach gives "
         "similar gains with no training and O(1) memory per expert."),
        ("Why is cycles/byte so high compared to modern compressors?",
         "SEE prioritizes compression ratio over speed. The MoE update involves "
         "floating-point exponentials for each expert on each byte. A typical "
         "run costs ~60,000 cycles/byte vs ~50 cycles/byte for LZ4. This is "
         "a research tool, not a production codec. A hardware-accelerated "
         "implementation could reduce this by 100-1000x."),
        ("What does 'MoE tax' mean?",
         "Adding a new expert to the mixture has a cost even if the expert "
         "carries no useful signal: the softmax normalization slightly dilutes "
         "the weights of existing experts, and the fixed-share mechanism "
         "permanently steals a tiny fraction of weight. Phase 28B measured "
         "this: adding a fully muted SPAN expert costs +0.0000 BPB on most "
         "domains, confirming the tax is negligible at 5-expert scale."),
    ]

    sections = ["\n\n---\n\n## Frequently Asked Questions\n"]
    for i, (q, a) in enumerate(questions):
        sections.append(f"\n### Q{i+1}. {q}\n\n{a}\n")

    # Phase-by-phase narrative (adds bulk)
    sections.append("\n\n---\n\n## Phase-by-Phase History\n")
    phase_summaries = [
        (18, "Wave Engine", "Replaced boolean bitwise CA with AVX2 saturating arithmetic. "
             "Established ESN (Echo State Network) paradigm: fixed reservoir, trained readout. "
             "Key result: temporal damping (`>> 1`) is necessary for echo state property."),
        (19, "Reservoir Readout", "Added M4 circular buffer as direct readout input. "
             "LMS training converged to 100% accuracy on Echo-5 with M4 channels. "
             "Confirmed that the wave is a nonlinear mixer; M4 provides the linear component."),
        (20, "Multi-Channel Readout", "Extracted K=32 channels via regional summation. "
             "Demonstrated that spatial gradients are preserved through summation. "
             "First working end-to-end encode/decode pipeline."),
        (21, "Online Expert Mixing", "Introduced 3-expert MoE (SEE, UNI, BI) with "
             "fixed-share Hedge update. First system to beat any fixed lambda on all "
             "domains simultaneously. SHA-256 roundtrip verified."),
        (22, "MoE Validation", "Full cross-domain audit. MoE outperforms every fixed "
             "lambda on every dataset. Natural language compression: 2.47 BPB. "
             "Code: 1.94 BPB. Shuffled: 5.01 BPB (correct behavior confirmed)."),
        (23, "LZ Expert", "Added LZ6 expert (FNV-1a hash, 1M-slot table). "
             "BPB improvements: c_code -0.5, markdown -0.3, natural_text -0.1. "
             "Introduced micro-audit harness for per-expert attribution."),
        (24, "Consolidation", "Sealed hybrid architecture: LZ6 + base MoE. "
             "Removed LSTM and PPM experiments. Established --expert-profile CLI. "
             "Defined 'default operational' vs 'experimental' tier."),
        (25, "Benchmark", "Cross-platform benchmark: Linux / Windows / MSVC / GCC. "
             "Confirmed cycles/byte consistency within 5% across compilers."),
        (26, "LZ Tribunal", "LZ6 promoted to default. LZ8 and LZ-dual archived. "
             "First use of tribunal protocol: BASE / MUTE / ACTIVE three-config pattern."),
        (27, "Token-LZ Expert", "TOKPFX expert introduced. natural_text gain -0.47 BPB. "
             "MoE assigns 95% weight to TOKPFX on prose. Shuffled correctly muted."),
        ("27C", "Cartography", "ALNUM_START is the natural_text gap (0.696 BPB, 22.9% bytes). "
             "Markdown gap confirmed as uniform MoE convergence lag +1.3 BPB, not byte-class specific."),
        ("27F", "Eligibility Audit", "TOK_PREV_ELIG promoted to `text` profile. "
             "SHA-256 bug found and fixed (missing flag in archive header). "
             "natural_text gain: -0.045 BPB. W_when_elig = 0.8030."),
        (28, "SPAN Tribunal", "BOL rejected (volume). SPANPFX rejected (dictionary too sparse). "
             "Markdown gap accepted as structural limit of current streaming approach on 126KB corpus. "
             "Key learning: the gap is uniform across the file, not concentrated in delimiters."),
        ("29A", "Robustness Matrix", "Current phase. 5 new synthetic corpora. "
             "4-profile matrix. TOKPFX value measurement via no-token baseline. "
             "Objective: confirm general/text profiles generalize beyond historical 4 files."),
    ]
    for ph, name, summary in phase_summaries:
        sections.append(f"\n### Phase {ph}: {name}\n\n{summary}\n")

    return "\n".join(sections)


def gen_repo_markdown():
    base = REPO_MARKDOWN + REPO_MARKDOWN_EXTRA
    extra = _gen_markdown_faq()
    # Repeat the appendices with slight variation if still under target
    full = base + extra
    while len(full.encode("utf-8")) < 100_000:
        full += "\n\n<!-- filler -->\n" + extra
    return full


# ---------------------------------------------------------------------------
# 5. log_synth.log  —  Synthetic server/application log
# ---------------------------------------------------------------------------

LOG_LEVELS    = ["INFO", "INFO", "INFO", "WARN", "ERROR", "DEBUG"]
LOG_SERVICES  = ["api-gateway", "encoder-worker", "decoder-worker",
                 "moe-engine", "lz-cache", "telemetry", "health-check",
                 "scheduler", "storage-driver", "auth-service"]
LOG_MSGS_INFO = [
    "Request completed successfully",
    "Worker thread started",
    "Cache entry evicted (LRU)",
    "Weight vector updated: TOKPFX=0.9507",
    "Audit pass completed: 218553 bytes, BPB=2.4741",
    "Encode job queued: file_id={}",
    "SHA-256 verified: {}",
    "Connection pool: {}/{} active",
    "MoE weights converged after {} bytes",
    "LZ table miss rate: {:.1f}%",
    "Expert profile: general (LZ6+TOKPFX)",
    "Throughput: {:.1f} KB/s",
    "Health check passed: all {} services healthy",
    "Session started: user_id=usr_{:06d}",
    "Scheduled audit job dispatched: phase29a",
    "Config reloaded from disk",
    "Rate limit check passed: {}/{} requests",
    "Corpus loaded: {} bytes, domain={}",
]
LOG_MSGS_WARN = [
    "High memory usage: {:.0f} MB / 512 MB",
    "LZ collision detected at slot 0x{:06x}",
    "Slow encode detected: {} ms > threshold 1000 ms",
    "Weight divergence: expert {} dropped below 0.001",
    "Rate limit approaching: {}/{} requests per minute",
    "Cache miss rate elevated: {:.1f}% in last 60s",
    "Decode retry #{}: possible data corruption",
    "Worker backlog: {} items queued",
]
LOG_MSGS_ERR = [
    "SHA-256 mismatch: expected={}, got={}",
    "Encode failed: SEE_ERR_NOMEM (requested {} bytes)",
    "LZ table full: eviction failed after {} retries",
    "Connection timeout after {}ms",
    "Expert weight NaN detected at byte {}",
    "Archive header magic mismatch: 0x{:08x}",
    "Decode aborted: corrupt bitstream at offset {}",
    "Worker panic: unhandled exception in moe_engine",
]

def fmt_log_msg(msg, rng):
    fillers = [
        rng.randint(10000, 99999),
        f"{rng.randint(0, 0xFFFFFF):06x}",
        f"{rng.randint(100, 9999)}",
        f"{rng.randint(0, 255)}/{rng.randint(256, 512)}",
        f"{rng.uniform(0, 100):.1f}",
        f"usr_{rng.randint(0, 9999):06d}",
        rng.choice(["natural_text", "markdown", "c_code", "json", "binary"]),
        f"{rng.randint(1, 8)}",
    ]
    try:
        return msg.format(*fillers[:msg.count("{}")])
    except (IndexError, ValueError):
        return msg

def gen_log_synth():
    lines = []
    base_ts = datetime(2026, 5, 25, 0, 0, 0)
    t = base_ts
    for _ in range(1500):
        t += timedelta(seconds=rng.uniform(0.01, 2.5))
        level   = rng.choice(LOG_LEVELS)
        service = rng.choice(LOG_SERVICES)
        req_id  = f"{rng.randint(0, 0xFFFFFFFF):08x}"
        pid     = rng.randint(10000, 99999)
        if level == "INFO":
            msg = fmt_log_msg(rng.choice(LOG_MSGS_INFO), rng)
        elif level == "WARN":
            msg = fmt_log_msg(rng.choice(LOG_MSGS_WARN), rng)
        elif level == "ERROR":
            msg = fmt_log_msg(rng.choice(LOG_MSGS_ERR), rng)
        else:  # DEBUG
            msg = f"trace={req_id} op=predict byte=0x{rng.randint(0,255):02x} bpb={rng.uniform(1,8):.4f}"

        line = (f"{t.strftime('%Y-%m-%dT%H:%M:%S.') + str(rng.randint(0,999)).zfill(3) + 'Z'}"
                f" [{level:<5}] [{service}] pid={pid} req={req_id} {msg}")
        lines.append(line)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(DATA, exist_ok=True)
    print(f"Generating Phase 29A corpus into {DATA}/\n")

    print("1. json_synth.json")
    if not already_exists("json_synth.json"):
        write_file("json_synth.json", gen_json_synth())

    print("2. c_header_synth.h")
    if not already_exists("c_header_synth.h"):
        write_file("c_header_synth.h", gen_c_header_synth())

    print("3. project_notes_it.txt")
    if not already_exists("project_notes_it.txt"):
        write_file("project_notes_it.txt", gen_project_notes_it())

    print("4. repo_markdown_mixed.md")
    if not already_exists("repo_markdown_mixed.md"):
        write_file("repo_markdown_mixed.md", gen_repo_markdown())

    print("5. log_synth.log")
    if not already_exists("log_synth.log"):
        write_file("log_synth.log", gen_log_synth())

    print("\nDone. Summary:")
    for name in ["json_synth.json", "c_header_synth.h", "project_notes_it.txt",
                 "repo_markdown_mixed.md", "log_synth.log"]:
        p = target_path(name)
        if os.path.exists(p):
            with open(p, "rb") as f: content = f.read()
            sha = hashlib.sha256(content).hexdigest()[:16]
            print(f"  {name:<30} {len(content)//1024:4d} KB  sha={sha}...")


if __name__ == "__main__":
    main()
