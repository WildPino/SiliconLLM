// METH-125: varied-token cost for exact E128 and centered E1280 factor banks.
#define main meth124_reference_main
#include "meth124_centered_factor_cpu.c"
#undef main

typedef struct { char magic[8]; uint32_t layers, tokens, width, experts; } VHeader;

static double median5(double *values) {
    double ordered[5];
    memcpy(ordered, values, sizeof ordered);
    qsort(ordered, 5, sizeof(double), cmp_double);
    return ordered[2];
}

static void route_selected(const Layer *layer, const float *x, const Header *h,
                           int *ids, float *gates) {
    int parents[4];
    route_parent(layer, x, h, parents, gates);
    if (h->children == 1) memcpy(ids, parents, sizeof parents);
    else route_child(layer, x, h, parents, ids);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s BANK VECTORS\n", argv[0]); return 2;
    }
    double started = now_s();
    size_t bank_size, vector_size;
    const uint8_t *bank = read_all(argv[1], &bank_size);
    const uint8_t *vector_data = read_all(argv[2], &vector_size);
    Header h;
    VHeader v;
    if (bank_size < sizeof h || vector_size < sizeof v) return 2;
    memcpy(&h, bank, sizeof h);
    memcpy(&v, vector_data, sizeof v);
    int parent_only = !memcmp(h.magic, "M125PR01", 8);
    int hierarchical = !memcmp(h.magic, "M124FB01", 8);
    if ((!parent_only && !hierarchical) || h.l != 24 || h.d != 896 || h.r != 8 ||
        h.rank != 64 || h.na != 8 || h.nb != 16 ||
        (parent_only && (h.children != 1 || h.child_rank != 0)) ||
        (hierarchical && (h.children != 10 || h.child_rank != 32))) {
        fprintf(stderr, "invalid bank header\n"); return 2;
    }
    uint32_t experts = h.na * h.nb * h.children;
    if (memcmp(v.magic, "M125HX01", 8) || v.layers != 24 ||
        v.tokens != 256 || v.width != 896 || v.experts != experts) {
        fprintf(stderr, "invalid vector header\n"); return 2;
    }
    size_t parent_bytes = ((size_t)h.rank * h.d + (h.na + h.nb) * h.rank) * sizeof(float);
    size_t projection_bytes = (size_t)h.child_rank * h.d * sizeof(float);
    size_t key_bytes = (size_t)experts * h.child_rank * sizeof(float);
    size_t router_bytes = parent_bytes + projection_bytes + key_bytes;
    size_t factor_bytes = (size_t)experts * h.r * h.d * sizeof(uint16_t);
    size_t layer_bytes = router_bytes + 2 * factor_bytes;
    if (bank_size != sizeof h + h.l * layer_bytes ||
        vector_size != sizeof v + (size_t)v.tokens * v.layers * v.width * sizeof(uint16_t)) {
        fprintf(stderr, "invalid input length\n"); return 2;
    }
    Layer layers[24];
    for (uint32_t l = 0; l < h.l; ++l) {
        const uint8_t *base = bank + sizeof h + l * layer_bytes;
        layers[l].p = (const float *)base;
        layers[l].a = layers[l].p + (size_t)h.rank * h.d;
        layers[l].b = layers[l].a + (size_t)h.na * h.rank;
        layers[l].child_projection = hierarchical ? (const float *)(base + parent_bytes) : NULL;
        layers[l].child_keys = hierarchical ? (const float *)(base + parent_bytes + projection_bytes) : NULL;
        layers[l].fa = (const uint16_t *)(base + router_bytes);
        layers[l].fb = (const uint16_t *)(base + router_bytes + factor_bytes);
    }
    const uint16_t *xbf = (const uint16_t *)(vector_data + sizeof v);
    size_t elements = (size_t)v.tokens * v.layers * v.width;
    float *x = malloc(elements * sizeof(float));
    if (!x) return 2;
    for (size_t i = 0; i < elements; ++i) x[i] = bf_float(xbf[i]);
    int selected[256][24][4];
    float gates[256][24][4];
    uint8_t seen[24][1280] = {{0}};
    int unique[24] = {0};
    volatile double checksum = 0;
    for (int token = 0; token < 256; ++token)
        for (int l = 0; l < 24; ++l) {
            const float *input = x + ((size_t)token * 24 + l) * 896;
            route_selected(&layers[l], input, &h, selected[token][l], gates[token][l]);
            for (int k = 0; k < 4; ++k) {
                int id = selected[token][l][k];
                if (id < 0 || (uint32_t)id >= experts || !isfinite(gates[token][l][k])) {
                    fprintf(stderr, "invalid route/gate\n"); return 1;
                }
                if (!seen[l][id]) { seen[l][id] = 1; unique[l]++; }
            }
            float out[896];
            residual(&layers[l], input, &h, selected[token][l], gates[token][l], out);
            for (int d = 0; d < 896; ++d) if (!isfinite(out[d])) {
                fprintf(stderr, "nonfinite residual\n"); return 1;
            }
            checksum += out[(token + l) % 896];
        }
    double route_ms[5], factor_ms[5];
    for (int rep = 0; rep < 5; ++rep) {
        double begin = now_s();
        for (int token = 0; token < 256; ++token)
            for (int l = 0; l < 24; ++l) {
                const float *input = x + ((size_t)token * 24 + l) * 896;
                int ids[4]; float weights[4];
                route_selected(&layers[l], input, &h, ids, weights);
                checksum += ids[0] + weights[0];
            }
        route_ms[rep] = (now_s() - begin) * 1000.0 / 256.0;
        begin = now_s();
        for (int token = 0; token < 256; ++token)
            for (int l = 0; l < 24; ++l) {
                const float *input = x + ((size_t)token * 24 + l) * 896;
                float out[896];
                residual(&layers[l], input, &h,
                         selected[token][l], gates[token][l], out);
                checksum += out[(token + l) % 896];
            }
        factor_ms[rep] = (now_s() - begin) * 1000.0 / 256.0;
        printf("VARIED_REP experts=%u rep=%d route_ms_per_token=%.9f factor_ms_per_token=%.9f\n",
               experts, rep, route_ms[rep], factor_ms[rep]);
    }
    int minimum = unique[0], maximum = unique[0], total = 0;
    for (int l = 0; l < 24; ++l) {
        if (unique[l] < minimum) minimum = unique[l];
        if (unique[l] > maximum) maximum = unique[l];
        total += unique[l];
        printf("VARIED_LAYER layer=%d unique_selected=%d\n", l, unique[l]);
    }
    printf("VARIED_SUMMARY experts=%u tokens=256 repetitions=5 route_median_ms_per_token=%.9f factor_median_ms_per_token=%.9f combined_median_sum_ms_per_token=%.9f unique_min=%d unique_max=%d unique_total=%d unique_factor_bytes=%zu selected_factor_bytes_per_token=%zu router_bytes_per_token=%zu bank_bytes=%zu rss_bytes=%zu elapsed_seconds=%.6f checksum=%.9f\n",
           experts, median5(route_ms), median5(factor_ms),
           median5(route_ms) + median5(factor_ms),
           minimum, maximum, total,
           (size_t)total * 2 * h.r * h.d * sizeof(uint16_t),
           (size_t)4 * 2 * h.r * h.d * sizeof(uint16_t) * h.l,
           router_bytes * h.l,
           bank_size, rss_bytes(), now_s() - started, checksum);
    free(x);
    free((void *)vector_data);
    free((void *)bank);
    return 0;
}
