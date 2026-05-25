#ifndef LZ_TOPN_H
#define LZ_TOPN_H

#include <stdint.h>

// Compact Top-N LZ dictionary.
// Each context key maps to the N most-observed successor bytes.
// Probability: p_lz(c) = mass_lz*p_topN(c) + (1-mass_lz*cov)/256
// where mass_lz = total/(total+K), cov = sum(top_counts)/total.
// Normalized by construction; degrades to uniform on unknown contexts.

#define LZ_HASH_SIZE  262144u   // must be a power of 2
#define LZ_TOP_N      4
#define LZ_K_DEFAULT  16.0f     // smoothing: confidence grows with observations

typedef struct {
    uint8_t  byte;
    uint16_t count;
} LzTopSlot;

typedef struct {
    uint32_t  key;
    uint32_t  total;
    LzTopSlot top[LZ_TOP_N];
} LzEntry;

// Allocate and zero-initialize the hash table. Returns NULL on OOM.
LzEntry* lz_table_alloc(void);

// Release table memory.
void lz_table_free(LzEntry* table);

// Look up (or claim an empty slot for) the given key.
LzEntry* lz_lookup(LzEntry* table, uint32_t key);

// Fill p_out[256] with the smoothed LZ probability distribution for entry e.
void lz_build_probs(const LzEntry* e, float lz_K, float* p_out);

// Record that byte `target` was observed under context key `ctx_key`.
void lz_update(LzEntry* e, uint32_t ctx_key, uint8_t target);

#endif // LZ_TOPN_H
