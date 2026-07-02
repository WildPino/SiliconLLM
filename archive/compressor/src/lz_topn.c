#include "lz_topn.h"
#include <stdlib.h>
#include <string.h>

LzEntry* lz_table_alloc(void) {
    return (LzEntry*)calloc(LZ_HASH_SIZE, sizeof(LzEntry));
}

void lz_table_free(LzEntry* table) {
    free(table);
}

LzEntry* lz_lookup(LzEntry* table, uint64_t key) {
    // 64-bit finalizer (MurmurHash3-style)
    uint64_t h = key;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    uint32_t idx = (uint32_t)(h & (LZ_HASH_SIZE - 1));
    for (int i = 0; i < 16; i++) {
        if (table[idx].total == 0) return &table[idx];
        if (table[idx].key  == key)  return &table[idx];
        idx = (idx + 1) & (LZ_HASH_SIZE - 1);
    }
    return &table[idx]; // evict on full probe window
}

void lz_build_probs(const LzEntry* e, float lz_K, float* p_out) {
    if (e->total == 0) {
        for (int c = 0; c < 256; c++) p_out[c] = 1.0f / 256.0f;
        return;
    }
    float mass_lz = (float)e->total / ((float)e->total + lz_K);
    float cov_counts = 0;
    for (int n = 0; n < LZ_TOP_N; n++) cov_counts += e->top[n].count;
    float cov = cov_counts / (float)e->total;
    // uniform_part = remaining mass after top-N (tail + prior)
    float uniform_part = (1.0f - mass_lz * cov) / 256.0f;
    for (int c = 0; c < 256; c++) p_out[c] = uniform_part;
    float inv_total = mass_lz / (float)e->total;
    for (int n = 0; n < LZ_TOP_N; n++) {
        if (e->top[n].count > 0)
            p_out[(uint8_t)e->top[n].byte] += inv_total * e->top[n].count;
    }
}

void lz_update(LzEntry* e, uint64_t ctx_key, uint8_t target) {
    if (e->key != ctx_key) {
        e->key   = ctx_key;
        e->total = 0;
        memset(e->top, 0, sizeof(e->top));
    }
    e->total++;
    for (int n = 0; n < LZ_TOP_N; n++) {
        if (e->top[n].count > 0 && e->top[n].byte == target) {
            e->top[n].count++;
            return;
        }
    }
    for (int n = 0; n < LZ_TOP_N; n++) {
        if (e->top[n].count == 0) {
            e->top[n].byte  = target;
            e->top[n].count = 1;
            return;
        }
    }
    // All slots occupied and byte not found: total increments, coverage decreases naturally
}
