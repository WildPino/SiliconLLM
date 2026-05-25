#ifndef TOK_LZ_H
#define TOK_LZ_H

#include <stdint.h>
#include "lz_topn.h"

// Token-LZ: inside-token prefix expert for MoE.
//
// Observes the stream as a sequence of TOKENS (not individual bytes):
//   ALNUM  — maximal alphanumeric run: [a-zA-Z0-9]+
//   MACRO  — backslash followed by alphanumeric: \mathbf, \begin, \end ...
//   SPACE  — single ' ' or '\t'
//   NL     — single '\n' or '\r'
//   DELIM  — any other byte (each its own single-byte token)
//
// Context key (TOK_PREFIX mode):
//   key = prefix_hash ^ (tok_pos_bucket << 60)
//
// Where:
//   prefix_hash = TYPE_SEED[tok_type] mixed with bytes seen so far in this token
//   tok_pos_bucket encodes position within token (0,1,2-3,4-7,8+)
//
// This lets the expert learn:
//   "given I am N bytes into an ALNUM/MACRO token whose prefix is H,
//    what is the next byte?"
//
// Reuses LzEntry and lz_build_probs() from lz_topn — same Top-N structure,
// same smoothing.  Caller must allocate a separate hash table via lz_table_alloc().

typedef enum {
    TOKTYPE_NONE  = 0,
    TOKTYPE_ALNUM = 1,
    TOKTYPE_MACRO = 2,  // '\' + alphanum (LaTeX/escape sequences)
    TOKTYPE_SPACE = 3,
    TOKTYPE_NL    = 4,
    TOKTYPE_DELIM = 5,
} TokLzType;

typedef struct {
    uint64_t  prefix_hash;    // hash of bytes seen so far in current token
    uint32_t  tok_pos;        // bytes accumulated in current token (0 at stream start)
    TokLzType tok_type;       // type of the current token
    uint64_t  last_tok_hash;  // hash of last completed token (for future TOK_PREV)
} TokLzState;

// Initialize to stream start.
void tok_lz_init(TokLzState* s);

// Compute the lookup key from current state (call BEFORE tok_lz_advance).
// Use this key to call lz_lookup() and lz_build_probs() for prediction.
uint64_t tok_lz_key(const TokLzState* s);

// Advance state after observing byte b (call AFTER prediction and lz_update).
// Returns the new tok_type (caller can inspect but usually ignores it).
TokLzType tok_lz_advance(TokLzState* s, uint8_t b);

#endif // TOK_LZ_H
