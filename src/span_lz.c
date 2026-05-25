#include "span_lz.h"

// Distinct seeds per span type, well-separated in 64-bit space.
// Chosen to differ from tok_lz seeds (0xAAAA..., 0x5555...) to avoid collisions
// if both experts share a hash table (they don't, but belt-and-suspenders).
static const uint64_t SPAN_SEEDS[3] = {
    0x0000000000000000ULL,  // NONE (unused in key computation)
    0xD1E2F3A4B5C60718ULL,  // BACKTICK
    0x3C4D5E6F7A8B9CA0ULL,  // DOLLAR
};

// LCG-style mix: same as tok_lz for consistency.
static inline uint64_t hash_mix(uint64_t h, uint8_t b) {
    return h * 6364136223846793005ULL ^ (uint64_t)b;
}

// Position bucket: how many interior bytes have been seen so far.
static inline uint64_t span_pos_bucket(uint32_t pos) {
    if (pos == 0) return 0;
    if (pos <= 2) return 1;
    if (pos <= 5) return 2;
    if (pos <= 11) return 3;
    return 4;
}

void span_lz_init(SpanLzState* s) {
    s->span_type   = SPAN_NONE;
    s->prefix_hash = 0;
    s->span_pos    = 0;
}

int span_lz_eligible(const SpanLzState* s) {
    return s->span_type != SPAN_NONE;
}

uint64_t span_lz_key(const SpanLzState* s) {
    uint64_t pb = span_pos_bucket(s->span_pos);
    return SPAN_SEEDS[s->span_type] ^ s->prefix_hash ^ (pb << 60);
}

SpanType span_lz_advance(SpanLzState* s, uint8_t b) {
    // Newline terminates any open inline span (they cannot cross lines).
    if (b == '\n') {
        s->span_type   = SPAN_NONE;
        s->prefix_hash = 0;
        s->span_pos    = 0;
        return SPAN_NONE;
    }

    if (s->span_type == SPAN_NONE) {
        if (b == '`') {
            s->span_type   = SPAN_BACKTICK;
            s->prefix_hash = 0;
            s->span_pos    = 0;
        } else if (b == '$') {
            s->span_type   = SPAN_DOLLAR;
            s->prefix_hash = 0;
            s->span_pos    = 0;
        }
        return s->span_type;  // opening delimiter itself is not interior
    }

    // Closing delimiter: seal the span.
    if ((s->span_type == SPAN_BACKTICK && b == '`') ||
        (s->span_type == SPAN_DOLLAR   && b == '$')) {
        s->span_type   = SPAN_NONE;
        s->prefix_hash = 0;
        s->span_pos    = 0;
        return SPAN_NONE;
    }

    // Interior byte: extend prefix hash and bump position.
    s->prefix_hash = hash_mix(s->prefix_hash, b);
    s->span_pos++;
    return s->span_type;
}
