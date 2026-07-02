#ifndef SPAN_LZ_H
#define SPAN_LZ_H

#include <stdint.h>

// Inline-span expert for MoE.
//
// Tracks markdown inline delimiter spans in a single-pass, streaming-compatible
// state machine:
//   BACKTICK — content inside `...`  (inline code)
//   DOLLAR   — content inside $...$  (inline math / LaTeX)
//   NONE     — everything else
//
// The expert is INELIGIBLE when span_type == NONE: its weight is frozen and it
// contributes nothing to the mixture (gated MoE).
//
// Deliberate simplifications (v1 — first tribunal):
//   - Single-backtick toggle only (no ``, no code fences)
//   - Single-dollar toggle only (no $$, no escaped \$)
//   - Newline resets any open span (inline spans cannot cross lines)
//   - No escape processing (\` or \$ treated as plain bytes)
//
// Key:
//   seed[span_type] ^ prefix_hash ^ (pos_bucket << 60)
//
// Reuses lz_topn infrastructure (LzEntry, lz_lookup, lz_build_probs, lz_update).

typedef enum {
    SPAN_NONE     = 0,
    SPAN_BACKTICK = 1,
    SPAN_DOLLAR   = 2,
} SpanType;

typedef struct {
    SpanType  span_type;    // current inline span state
    uint64_t  prefix_hash;  // hash of bytes seen so far inside the span
    uint32_t  span_pos;     // interior byte count (0 at first byte after opening delim)
} SpanLzState;

// Initialize to stream start (outside any span).
void span_lz_init(SpanLzState* s);

// Returns 1 if currently inside a span (BACKTICK or DOLLAR).
// Call BEFORE span_lz_advance on each byte.
int span_lz_eligible(const SpanLzState* s);

// Lookup key for the current span state. Only meaningful when eligible.
// Call BEFORE span_lz_advance.
uint64_t span_lz_key(const SpanLzState* s);

// Advance state after observing byte b. Returns the NEW SpanType.
// Call AFTER prediction and LZ update.
SpanType span_lz_advance(SpanLzState* s, uint8_t b);

#endif // SPAN_LZ_H
