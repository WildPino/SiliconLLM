// METH-135: actual-state cost of an untrained third routing tier.
#define main meth124_reference_main
#include "meth124_centered_factor_cpu.c"
#undef main

typedef struct {
    char magic[8];
    uint32_t layers, width, rank, children, grandchildren, seed;
} ThirdHeader;

typedef struct { const float *projection, *keys; } ThirdLayer;
typedef struct { char magic[8]; uint32_t layers, tokens, width, experts; } VectorHeader;

static double median5(const double *numbers) {
    double sorted[5];
    memcpy(sorted, numbers, sizeof sorted);
    qsort(sorted, 5, sizeof(double), cmp_double);
    return sorted[2];
}

static void select_base(const Layer *layer, const float *x, const Header *h,
                        int *ids, float *gates) {
    int parents[4];
    route_parent(layer, x, h, parents, gates);
    route_child(layer, x, h, parents, ids);
}

static int select_grandchildren(const ThirdLayer *third, const float *x,
                                const int *children, int *grandchildren) {
    float query[32];
    for (int r = 0; r < 32; ++r) {
        float sum = 0;
        const float *row = third->projection + (size_t)r * 896;
        for (int d = 0; d < 896; ++d) sum += row[d] * x[d];
        if (!isfinite(sum)) return 0;
        query[r] = sum;
    }
    for (int k = 0; k < 4; ++k) {
        int best = -1;
        float maximum = 0;
        for (int c = 0; c < 10; ++c) {
            const float *key = third->keys +
                ((size_t)children[k] * 10 + c) * 32;
            float score = 0;
            for (int r = 0; r < 32; ++r) score += query[r] * key[r];
            if (!isfinite(score)) return 0;
            if (best < 0 || score > maximum) { best = c; maximum = score; }
        }
        grandchildren[k] = children[k] * 10 + best;
    }
    return 1;
}

static double time_routes(const Layer *layers, const ThirdLayer *third,
                          const Header *h, const float *x, int expanded,
                          volatile double *checksum) {
    double started = now_s();
    for (int token = 0; token < 256; ++token)
        for (int li = 0; li < 24; ++li) {
            const float *input = x + ((size_t)token * 24 + li) * 896;
            int children[4], grandchildren[4];
            float gates[4];
            select_base(&layers[li], input, h, children, gates);
            if (expanded && !select_grandchildren(&third[li], input,
                                                    children, grandchildren)) {
                fprintf(stderr, "nonfinite third-tier route\n"); exit(1);
            }
            *checksum += (expanded ? grandchildren[0] : children[0]) + gates[0];
        }
    return (now_s() - started) * 1000.0 / 256.0;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s EXACT_BANK VECTORS THIRD_ROUTER\n", argv[0]);
        return 2;
    }
    double started = now_s();
    size_t bank_bytes, vector_bytes, third_bytes;
    const uint8_t *bank = read_all(argv[1], &bank_bytes);
    const uint8_t *vectors = read_all(argv[2], &vector_bytes);
    const uint8_t *sidecar = read_all(argv[3], &third_bytes);
    Header h;
    VectorHeader v;
    ThirdHeader t;
    if (bank_bytes < sizeof h || vector_bytes < sizeof v ||
        third_bytes < sizeof t) return 2;
    memcpy(&h, bank, sizeof h);
    memcpy(&v, vectors, sizeof v);
    memcpy(&t, sidecar, sizeof t);
    if (memcmp(h.magic, "M126FB01", 8) || h.l != 24 || h.d != 896 ||
        h.rank != 64 || h.na != 8 || h.nb != 16 || h.child_rank != 32 ||
        h.children != 10 || h.r != 8 ||
        memcmp(v.magic, "M125HX01", 8) || v.layers != 24 ||
        v.tokens != 256 || v.width != 896 || v.experts != 1280) {
        fprintf(stderr, "invalid exact bank/vector header\n"); return 2;
    }
    if (memcmp(t.magic, "M135RT01", 8) || t.layers != 24 ||
        t.width != 896 || t.rank != 32 || t.children != 1280 ||
        t.grandchildren != 10 || t.seed != 134134) {
        fprintf(stderr, "invalid third-tier sidecar header\n"); return 2;
    }
    size_t parent_bytes = ((size_t)h.rank * h.d + (h.na + h.nb) * h.rank) * 4;
    size_t child_projection_bytes = (size_t)h.child_rank * h.d * 4;
    size_t child_key_bytes = (size_t)1280 * h.child_rank * 4;
    size_t router_bytes = parent_bytes + child_projection_bytes + child_key_bytes;
    size_t a_bytes = (size_t)128 * h.r * h.d * 2;
    size_t b_bytes = (size_t)1280 * h.r * h.d * 2;
    size_t bank_layer_bytes = router_bytes + a_bytes + b_bytes;
    size_t third_projection_bytes = (size_t)32 * 896 * 4;
    size_t third_key_bytes = (size_t)1280 * 10 * 32 * 4;
    size_t third_layer_bytes = third_projection_bytes + third_key_bytes;
    if (bank_bytes != sizeof h + 24 * bank_layer_bytes ||
        vector_bytes != sizeof v + (size_t)256 * 24 * 896 * 2 ||
        third_bytes != sizeof t + 24 * third_layer_bytes) {
        fprintf(stderr, "invalid input lengths\n"); return 2;
    }
    Layer layers[24];
    ThirdLayer third[24];
    for (int li = 0; li < 24; ++li) {
        const uint8_t *base = bank + sizeof h + (size_t)li * bank_layer_bytes;
        layers[li].p = (const float *)base;
        layers[li].a = layers[li].p + (size_t)h.rank * h.d;
        layers[li].b = layers[li].a + (size_t)h.na * h.rank;
        layers[li].child_projection = (const float *)(base + parent_bytes);
        layers[li].child_keys = (const float *)(base + parent_bytes + child_projection_bytes);
        layers[li].fa = (const uint16_t *)(base + router_bytes);
        layers[li].fb = (const uint16_t *)(base + router_bytes + a_bytes);
        const uint8_t *extra = sidecar + sizeof t + (size_t)li * third_layer_bytes;
        third[li].projection = (const float *)extra;
        third[li].keys = (const float *)(extra + third_projection_bytes);
    }
    size_t input_count = (size_t)256 * 24 * 896;
    float *x = malloc(input_count * sizeof(float));
    if (!x) return 2;
    const uint16_t *bf = (const uint16_t *)(vectors + sizeof v);
    for (size_t i = 0; i < input_count; ++i) x[i] = bf_float(bf[i]);
    uint8_t seen_child[24][1280] = {{0}};
    uint8_t seen_grand[24][12800] = {{0}};
    int unique_child[24] = {0}, unique_grand[24] = {0};
    double max_residual_error = 0;
    volatile double checksum = 0;
    for (int token = 0; token < 256; ++token)
        for (int li = 0; li < 24; ++li) {
            const float *input = x + ((size_t)token * 24 + li) * 896;
            int children[4], repeated_children[4], grandchildren[4];
            float gates[4], repeated_gates[4];
            select_base(&layers[li], input, &h, children, gates);
            select_base(&layers[li], input, &h, repeated_children, repeated_gates);
            if (!select_grandchildren(&third[li], input, children, grandchildren)) {
                fprintf(stderr, "nonfinite third-tier route\n"); return 1;
            }
            for (int k = 0; k < 4; ++k) {
                int child = children[k], grand = grandchildren[k];
                if (child != repeated_children[k] ||
                    memcmp(&gates[k], &repeated_gates[k], sizeof(float)) ||
                    child < 0 || child >= 1280 || grand < 0 || grand >= 12800 ||
                    grand / 10 != child || !isfinite(gates[k])) {
                    fprintf(stderr, "route/gate identity failure\n"); return 1;
                }
                if (!seen_child[li][child]) { seen_child[li][child] = 1; unique_child[li]++; }
                if (!seen_grand[li][grand]) { seen_grand[li][grand] = 1; unique_grand[li]++; }
            }
            float exact[896], clone[896];
            residual(&layers[li], input, &h, children, gates, exact);
            for (int k = 0; k < 4; ++k) repeated_children[k] = grandchildren[k] / 10;
            residual(&layers[li], input, &h, repeated_children, gates, clone);
            for (int d = 0; d < 896; ++d) {
                if (!isfinite(exact[d]) || !isfinite(clone[d])) {
                    fprintf(stderr, "nonfinite factor residual\n"); return 1;
                }
                double difference = fabs((double)exact[d] - clone[d]);
                if (difference > max_residual_error) max_residual_error = difference;
            }
            checksum += exact[(token + li) % 896];
        }
    if (max_residual_error != 0) {
        fprintf(stderr, "clone residual mismatch\n"); return 1;
    }
    double base_ms[5], expanded_ms[5];
    for (int rep = 0; rep < 5; ++rep) {
        if (rep % 2 == 0) {
            base_ms[rep] = time_routes(layers, third, &h, x, 0, &checksum);
            expanded_ms[rep] = time_routes(layers, third, &h, x, 1, &checksum);
        } else {
            expanded_ms[rep] = time_routes(layers, third, &h, x, 1, &checksum);
            base_ms[rep] = time_routes(layers, third, &h, x, 0, &checksum);
        }
        printf("THIRD_REP rep=%d base_ms=%.9f e12800_ms=%.9f\n",
               rep, base_ms[rep], expanded_ms[rep]);
        if (now_s() - started > 120 || rss_bytes() > 3ull * (1ull << 30)) {
            fprintf(stderr, "METH-135 resource stop\n"); return 1;
        }
    }
    int child_total = 0, grand_total = 0;
    for (int li = 0; li < 24; ++li) {
        child_total += unique_child[li]; grand_total += unique_grand[li];
        printf("THIRD_LAYER layer=%d unique_children=%d unique_grandchildren=%d\n",
               li, unique_child[li], unique_grand[li]);
    }
    if (grand_total <= child_total) {
        fprintf(stderr, "nontrivial grandchild coverage failed\n"); return 1;
    }
    double base_median = median5(base_ms), expanded_median = median5(expanded_ms);
    int pass = expanded_median <= 3.0 && expanded_median <= 2.0 * base_median;
    printf("THIRD_SUMMARY states=6144 selections=24576 child_unique_total=%d grand_unique_total=%d base_median_ms=%.9f e12800_median_ms=%.9f ratio=%.6f third_key_bytes_per_token=%zu sidecar_bytes=%zu max_residual_abs=%.9g rss_bytes=%zu elapsed_seconds=%.6f checksum=%.9f gate=%s\n",
           child_total, grand_total, base_median, expanded_median,
           expanded_median / base_median,
           (size_t)24 * (third_projection_bytes + (size_t)4 * 10 * 32 * 4),
           third_bytes, max_residual_error, rss_bytes(), now_s() - started,
           checksum, pass ? "PASS" : "FAIL");
    free(x); free((void *)sidecar); free((void *)vectors); free((void *)bank);
    return pass ? 0 : 1;
}
