/*
 * Minimal read-only GGUF (v2/v3) reader for the C50 engine: memory-maps the file, parses
 * the key/value header and the tensor table, and exposes typed lookups.  Format as written
 * by llama.cpp gguf-py at the pinned commit 5b335f4 (general.alignment, default 32).
 * Windows only (CreateFileMapping); the engine is a Windows/x86-64-v3 build.
 */
#ifndef C50_GGUF_H
#define C50_GGUF_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

enum { GG_U8, GG_I8, GG_U16, GG_I16, GG_U32, GG_I32, GG_F32, GG_BOOL, GG_STR, GG_ARR, GG_U64, GG_I64, GG_F64 };
/* ggml tensor types used by the artifacts of this campaign */
enum { GT_F32 = 0, GT_F16 = 1, GT_Q8_0 = 8, GT_Q4_K = 12, GT_Q5_K = 13, GT_Q6_K = 14, GT_IQ4_XS = 23, GT_BF16 = 30 };

typedef struct { const char *name; uint32_t name_len, n_dims, type; uint64_t dims[4], offset; const uint8_t *data; } gg_tensor;
typedef struct { const char *key; uint32_t key_len, type; const uint8_t *val; } gg_kv;

typedef struct {
    const uint8_t *base;
    uint64_t size, n_tensors, n_kv, alignment;
    gg_kv *kv;
    gg_tensor *t;
    HANDLE file, map;
} gguf;

static void gg_fail(const char *m) { fprintf(stderr, "gguf: %s\n", m); exit(2); }

static uint64_t gg_scalar_size(uint32_t type) {
    switch (type) {
    case GG_U8: case GG_I8: case GG_BOOL: return 1;
    case GG_U16: case GG_I16: return 2;
    case GG_U32: case GG_I32: case GG_F32: return 4;
    case GG_U64: case GG_I64: case GG_F64: return 8;
    default: return 0;
    }
}

/* bytes occupied by one value of `type` starting at p (strings and arrays included) */
static uint64_t gg_value_size(const uint8_t *p, uint32_t type) {
    if (type == GG_STR) { uint64_t n; memcpy(&n, p, 8); return 8 + n; }
    if (type == GG_ARR) {
        uint32_t et; uint64_t n, off = 12, i;
        memcpy(&et, p, 4); memcpy(&n, p + 4, 8);
        if (et == GG_STR || et == GG_ARR) { for (i = 0; i < n; ++i) off += gg_value_size(p + off, et); return off; }
        return off + n * gg_scalar_size(et);
    }
    if (!gg_scalar_size(type)) gg_fail("unknown kv type");
    return gg_scalar_size(type);
}

static uint64_t gg_type_row_bytes(uint32_t type, uint64_t n) {
    switch (type) {
    case GT_F32: return 4 * n;
    case GT_F16: case GT_BF16: return 2 * n;
    case GT_Q8_0: return n / 32 * 34;
    case GT_Q4_K: return n / 256 * 144;
    case GT_Q5_K: return n / 256 * 176;
    case GT_Q6_K: return n / 256 * 210;
    case GT_IQ4_XS: return n / 256 * 136;
    default: return 0;
    }
}

static const char *gg_type_name(uint32_t type) {
    switch (type) {
    case GT_F32: return "F32"; case GT_F16: return "F16"; case GT_BF16: return "BF16"; case GT_Q8_0: return "Q8_0";
    case GT_Q4_K: return "Q4_K"; case GT_Q5_K: return "Q5_K"; case GT_Q6_K: return "Q6_K"; case GT_IQ4_XS: return "IQ4_XS";
    default: return "?";
    }
}

static void gguf_open(gguf *g, const char *path) {
    LARGE_INTEGER sz;
    const uint8_t *p;
    uint32_t magic, version;
    uint64_t i, off;
    memset(g, 0, sizeof *g);
    g->file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (g->file == INVALID_HANDLE_VALUE) gg_fail("cannot open file");
    GetFileSizeEx(g->file, &sz);
    g->size = (uint64_t)sz.QuadPart;
    g->map = CreateFileMappingA(g->file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!g->map) gg_fail("CreateFileMapping failed");
    g->base = MapViewOfFile(g->map, FILE_MAP_READ, 0, 0, 0);
    if (!g->base) gg_fail("MapViewOfFile failed");
    p = g->base;
    memcpy(&magic, p, 4); memcpy(&version, p + 4, 4);
    if (magic != 0x46554747U || version < 2) gg_fail("not a GGUF v2+ file");
    memcpy(&g->n_tensors, p + 8, 8); memcpy(&g->n_kv, p + 16, 8);
    off = 24;
    g->kv = calloc(g->n_kv, sizeof *g->kv);
    g->alignment = 32;
    for (i = 0; i < g->n_kv; ++i) {
        uint64_t kl; memcpy(&kl, p + off, 8);
        g->kv[i].key = (const char *)p + off + 8; g->kv[i].key_len = (uint32_t)kl; off += 8 + kl;
        memcpy(&g->kv[i].type, p + off, 4); off += 4;
        g->kv[i].val = p + off;
        if (kl == 17 && !memcmp(g->kv[i].key, "general.alignment", 17)) { uint32_t a; memcpy(&a, p + off, 4); g->alignment = a; }
        off += gg_value_size(p + off, g->kv[i].type);
    }
    g->t = calloc(g->n_tensors, sizeof *g->t);
    for (i = 0; i < g->n_tensors; ++i) {
        gg_tensor *t = &g->t[i];
        uint64_t nl, d;
        memcpy(&nl, p + off, 8); t->name = (const char *)p + off + 8; t->name_len = (uint32_t)nl; off += 8 + nl;
        memcpy(&t->n_dims, p + off, 4); off += 4;
        if (t->n_dims > 4) gg_fail("tensor rank > 4");
        for (d = 0; d < 4; ++d) t->dims[d] = 1;
        for (d = 0; d < t->n_dims; ++d) { memcpy(&t->dims[d], p + off, 8); off += 8; }
        memcpy(&t->type, p + off, 4); off += 4;
        memcpy(&t->offset, p + off, 8); off += 8;
    }
    off = (off + g->alignment - 1) / g->alignment * g->alignment;
    for (i = 0; i < g->n_tensors; ++i) {
        g->t[i].data = g->base + off + g->t[i].offset;
        if (off + g->t[i].offset > g->size) gg_fail("tensor data outside the file");
    }
}

static const gg_kv *gguf_kv(const gguf *g, const char *key) {
    uint64_t i, n = strlen(key);
    for (i = 0; i < g->n_kv; ++i) if (g->kv[i].key_len == n && !memcmp(g->kv[i].key, key, n)) return &g->kv[i];
    return NULL;
}

static uint64_t gguf_u(const gguf *g, const char *key) {
    const gg_kv *k = gguf_kv(g, key);
    uint64_t v = 0;
    if (!k) { fprintf(stderr, "gguf: missing key %s\n", key); exit(2); }
    switch (k->type) {
    case GG_U8: case GG_BOOL: v = k->val[0]; break;
    case GG_U16: { uint16_t x; memcpy(&x, k->val, 2); v = x; break; }
    case GG_U32: case GG_I32: { uint32_t x; memcpy(&x, k->val, 4); v = x; break; }
    case GG_U64: case GG_I64: memcpy(&v, k->val, 8); break;
    default: gg_fail("key is not an integer");
    }
    return v;
}

static float gguf_f(const gguf *g, const char *key) {
    const gg_kv *k = gguf_kv(g, key);
    float v;
    if (!k || k->type != GG_F32) { fprintf(stderr, "gguf: missing f32 key %s\n", key); exit(2); }
    memcpy(&v, k->val, 4);
    return v;
}

static const gg_tensor *gguf_tensor(const gguf *g, const char *name) {
    uint64_t i, n = strlen(name);
    for (i = 0; i < g->n_tensors; ++i) if (g->t[i].name_len == n && !memcmp(g->t[i].name, name, n)) return &g->t[i];
    fprintf(stderr, "gguf: missing tensor %s\n", name);
    exit(2);
}

#endif
