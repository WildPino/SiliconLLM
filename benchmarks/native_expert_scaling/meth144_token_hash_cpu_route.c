// METH-144: actual-state cost of a token-context hash third routing tier.
#define main meth124_reference_main
#include "meth124_centered_factor_cpu.c"
#undef main

typedef struct { char magic[8]; uint32_t layers, tokens, width, experts; } VectorHeader;
typedef struct { char magic[8]; uint32_t tokens; } ContextHeader;
typedef struct { uint32_t token, previous, position; } TokenContext;

static const uint64_t C1 = UINT64_C(0x9E3779B97F4A7C15);
static const uint64_t C2 = UINT64_C(0xBF58476D1CE4E5B9);
static const uint64_t C3 = UINT64_C(0x94D049BB133111EB);
static const uint64_t C4 = UINT64_C(0xD6E8FEB86659FD93);
static const uint64_t C5 = UINT64_C(0xA5A3564E27F8862D);

static int hash_grandchild(uint32_t token, uint32_t previous,
                           uint32_t position, int child, int layer) {
    uint64_t z = UINT64_C(142142) ^ ((uint64_t)token * C1) ^
                 ((uint64_t)previous * C2) ^ ((uint64_t)position * C3) ^
                 ((uint64_t)child * C4) ^ ((uint64_t)layer * C5);
    z += C1;
    z = (z ^ (z >> 30)) * C2;
    z = (z ^ (z >> 27)) * C3;
    z ^= z >> 31;
    return child * 10 + (int)(z % UINT64_C(10));
}

static void select_base(const Layer *layer, const float *x, const Header *h,
                        int *children, float *gates) {
    int parents[4];
    route_parent(layer, x, h, parents, gates);
    route_child(layer, x, h, parents, children);
}

static double median5(const double *values) {
    double sorted[5];
    memcpy(sorted, values, sizeof sorted);
    qsort(sorted, 5, sizeof(double), cmp_double);
    return sorted[2];
}

static double time_routes(const Layer *layers, const Header *header,
                          const float *states, const TokenContext *contexts,
                          int expanded, volatile double *checksum) {
    double started = now_s();
    for (int token = 0; token < 256; ++token)
        for (int li = 0; li < 24; ++li) {
            const float *x = states + ((size_t)token * 24 + li) * 896;
            int child[4];
            float gate[4];
            select_base(&layers[li], x, header, child, gate);
            if (expanded)
                for (int k = 0; k < 4; ++k)
                    child[k] = hash_grandchild(contexts[token].token,
                        contexts[token].previous, contexts[token].position,
                        child[k], li);
            *checksum += child[0] + gate[0];
        }
    return (now_s() - started) * 1000.0 / 256.0;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s EXACT_BANK VECTORS TOKEN_CONTEXT\n", argv[0]);
        return 2;
    }
    double started = now_s();
    size_t bank_bytes, vector_bytes, context_bytes;
    const uint8_t *bank = read_all(argv[1], &bank_bytes);
    const uint8_t *vectors = read_all(argv[2], &vector_bytes);
    const uint8_t *fixture = read_all(argv[3], &context_bytes);
    Header h;
    VectorHeader v;
    ContextHeader t;
    if (bank_bytes < sizeof h || vector_bytes < sizeof v ||
        context_bytes < sizeof t) return 2;
    memcpy(&h, bank, sizeof h);
    memcpy(&v, vectors, sizeof v);
    memcpy(&t, fixture, sizeof t);
    if (memcmp(h.magic, "M126FB01", 8) || h.l != 24 || h.d != 896 ||
        h.rank != 64 || h.na != 8 || h.nb != 16 || h.child_rank != 32 ||
        h.children != 10 || h.r != 8 ||
        memcmp(v.magic, "M125HX01", 8) || v.layers != 24 ||
        v.tokens != 256 || v.width != 896 || v.experts != 1280 ||
        memcmp(t.magic, "M144TX01", 8) || t.tokens != 256) {
        fprintf(stderr, "invalid input header\n"); return 2;
    }
    size_t parent_bytes = ((size_t)h.rank * h.d + (h.na + h.nb) * h.rank) * 4;
    size_t child_projection_bytes = (size_t)h.child_rank * h.d * 4;
    size_t child_key_bytes = (size_t)1280 * h.child_rank * 4;
    size_t router_bytes = parent_bytes + child_projection_bytes + child_key_bytes;
    size_t a_bytes = (size_t)128 * h.r * h.d * 2;
    size_t b_bytes = (size_t)1280 * h.r * h.d * 2;
    size_t layer_bytes = router_bytes + a_bytes + b_bytes;
    if (bank_bytes != sizeof h + 24 * layer_bytes ||
        vector_bytes != sizeof v + (size_t)256 * 24 * 896 * 2 ||
        context_bytes != sizeof t + (size_t)256 * sizeof(TokenContext)) {
        fprintf(stderr, "invalid input lengths\n"); return 2;
    }
    const TokenContext *contexts = (const TokenContext *)(fixture + sizeof t);
    for (int token = 0; token < 256; ++token) {
        if (contexts[token].position != (uint32_t)token ||
            (token == 0 ? contexts[token].previous != 0 :
             contexts[token].previous != contexts[token-1].token)) {
            fprintf(stderr, "invalid token context\n"); return 2;
        }
    }
    if (hash_grandchild(123, 45, 67, 89, 3) != 899) {
        fprintf(stderr, "hash golden mismatch\n"); return 1;
    }
    Layer layers[24];
    for (int li = 0; li < 24; ++li) {
        const uint8_t *base = bank + sizeof h + (size_t)li * layer_bytes;
        layers[li].p = (const float *)base;
        layers[li].a = layers[li].p + (size_t)h.rank * h.d;
        layers[li].b = layers[li].a + (size_t)h.na * h.rank;
        layers[li].child_projection = (const float *)(base + parent_bytes);
        layers[li].child_keys = (const float *)(base + parent_bytes + child_projection_bytes);
        layers[li].fa = (const uint16_t *)(base + router_bytes);
        layers[li].fb = (const uint16_t *)(base + router_bytes + a_bytes);
    }
    size_t input_count = (size_t)256 * 24 * 896;
    float *x = malloc(input_count * sizeof(float));
    if (!x) return 2;
    const uint16_t *bf = (const uint16_t *)(vectors + sizeof v);
    for (size_t i = 0; i < input_count; ++i) x[i] = bf_float(bf[i]);
    uint8_t seen[24][12800] = {{0}};
    int coverage[24] = {0};
    volatile double checksum = 0;
    for (int token = 0; token < 256; ++token)
        for (int li = 0; li < 24; ++li) {
            const float *input = x + ((size_t)token * 24 + li) * 896;
            int child[4], repeated[4], grand[4];
            float gate[4], gate_repeat[4];
            select_base(&layers[li], input, &h, child, gate);
            select_base(&layers[li], input, &h, repeated, gate_repeat);
            for (int k = 0; k < 4; ++k) {
                grand[k] = hash_grandchild(contexts[token].token,
                    contexts[token].previous, contexts[token].position,
                    child[k], li);
                if (child[k] != repeated[k] ||
                    memcmp(&gate[k], &gate_repeat[k], sizeof(float)) ||
                    child[k] < 0 || child[k] >= 1280 ||
                    grand[k] < 0 || grand[k] >= 12800 ||
                    grand[k] / 10 != child[k] || !isfinite(gate[k])) {
                    fprintf(stderr, "route/gate identity failure\n"); return 1;
                }
                if (!seen[li][grand[k]]) {
                    seen[li][grand[k]] = 1; coverage[li]++;
                }
            }
            printf("HASH_ROUTE token=%d layer=%d children=%d,%d,%d,%d grandchildren=%d,%d,%d,%d\n",
                   token, li, child[0], child[1], child[2], child[3],
                   grand[0], grand[1], grand[2], grand[3]);
        }
    double base_ms[5], expanded_ms[5];
    for (int rep = 0; rep < 5; ++rep) {
        if (rep % 2 == 0) {
            base_ms[rep] = time_routes(layers, &h, x, contexts, 0, &checksum);
            expanded_ms[rep] = time_routes(layers, &h, x, contexts, 1, &checksum);
        } else {
            expanded_ms[rep] = time_routes(layers, &h, x, contexts, 1, &checksum);
            base_ms[rep] = time_routes(layers, &h, x, contexts, 0, &checksum);
        }
        printf("HASH_REP rep=%d base_ms=%.9f hash_ms=%.9f\n",
               rep, base_ms[rep], expanded_ms[rep]);
        if (now_s() - started > 120 || rss_bytes() > 3ull * (1ull << 30)) {
            fprintf(stderr, "METH-144 native resource stop\n"); return 1;
        }
    }
    int selected_total = 0;
    for (int li = 0; li < 24; ++li) {
        printf("HASH_LAYER layer=%d coverage=%d\n", li, coverage[li]);
        selected_total += coverage[li];
    }
    double base = median5(base_ms), expanded = median5(expanded_ms);
    int pass = selected_total > 0 && expanded <= 3.0 && expanded <= 2.0 * base;
    printf("HASH_SUMMARY states=6144 selections=24576 unique_total=%d base_median_ms=%.9f hash_median_ms=%.9f ratio=%.6f hash_addressed_bytes_per_token=%zu fixture_bytes=%zu rss_bytes=%zu elapsed_seconds=%.6f checksum=%.9f gate=%s\n",
           selected_total, base, expanded, expanded/base,
           (size_t)24 * 4 * (sizeof(TokenContext) + sizeof(uint32_t)),
           context_bytes, rss_bytes(), now_s() - started, checksum,
           pass ? "PASS" : "FAIL");
    free(x); free((void *)fixture); free((void *)vectors); free((void *)bank);
    return pass ? 0 : 1;
}
