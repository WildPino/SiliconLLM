#include "tok_lz.h"

// Per-type hash seeds: distinguish "start of ALNUM" from "start of MACRO" etc.
// Chosen to be well-separated in 64-bit space.
static const uint64_t TYPE_SEEDS[6] = {
    0x0000000000000000ULL,  // NONE
    0xAAAAAAAAAAAAAAAAULL,  // ALNUM
    0x5555555555555555ULL,  // MACRO (backslash-alnum)
    0xCCCCCCCCCCCCCCCCULL,  // SPACE
    0x3333333333333333ULL,  // NL
    0xF0F0F0F0F0F0F0F0ULL,  // DELIM
};

// LCG-style hash mix: cheap and sufficient for a Top-N table with Murmur
// finalizer at lookup time.
static inline uint64_t hash_mix(uint64_t h, uint8_t b) {
    return h * 6364136223846793005ULL ^ (uint64_t)b;
}

// Token position bucket → 3-bit value for high bits of key.
// Encodes "how many bytes we've seen in this token so far".
static inline uint32_t tok_pos_bucket(uint32_t pos) {
    if (pos == 0) return 0;
    if (pos <= 2) return 1;
    if (pos <= 5) return 2;
    if (pos <= 11) return 3;
    return 4;
}

// Classify a byte given the previous token type.
// Returns the new token type.
static TokLzType classify_byte(uint8_t b, TokLzType prev) {
    int alnum = ((b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') ||
                 (b >= '0' && b <= '9'));

    if (b == '\\') return TOKTYPE_MACRO;

    if (alnum) {
        // Continue an existing ALNUM or MACRO run.
        if (prev == TOKTYPE_ALNUM || prev == TOKTYPE_MACRO) return prev;
        return TOKTYPE_ALNUM;
    }

    if (b == '\n' || b == '\r') return TOKTYPE_NL;
    if (b == ' '  || b == '\t') return TOKTYPE_SPACE;
    return TOKTYPE_DELIM;
}

// A boundary occurs when the token type changes, or when the current token
// is a single-byte type (SPACE, NL, DELIM — each byte is its own token).
static inline int is_token_boundary(TokLzType new_type, TokLzType prev) {
    if (new_type != prev) return 1;
    // Runs are only allowed for ALNUM and MACRO.
    if (prev == TOKTYPE_SPACE || prev == TOKTYPE_NL || prev == TOKTYPE_DELIM) return 1;
    return 0;
}


void tok_lz_init(TokLzState* s) {
    s->prefix_hash   = 0;
    s->tok_pos       = 0;
    s->tok_type      = TOKTYPE_NONE;
    s->last_tok_hash = 0;
}

uint64_t tok_lz_key(const TokLzState* s) {
    uint32_t bucket = tok_pos_bucket(s->tok_pos);
    // High 4 bits: position bucket.  Remaining 60 bits: prefix content.
    return s->prefix_hash ^ ((uint64_t)bucket << 60);
}

TokLzType tok_lz_advance(TokLzState* s, uint8_t b) {
    TokLzType new_type = classify_byte(b, s->tok_type);

    if (is_token_boundary(new_type, s->tok_type)) {
        // Token boundary: seal current token, start new one.
        s->last_tok_hash = s->prefix_hash;
        s->prefix_hash   = hash_mix(TYPE_SEEDS[new_type], b);
        s->tok_pos       = 1;
        s->tok_type      = new_type;
    } else {
        // Continue current token.
        s->prefix_hash = hash_mix(s->prefix_hash, b);
        s->tok_pos++;
    }

    return new_type;
}
