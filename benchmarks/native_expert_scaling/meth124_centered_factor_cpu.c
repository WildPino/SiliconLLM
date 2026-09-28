// METH-124: native centered E1280 router and selected-factor parity/cost.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <psapi.h>

typedef struct { char magic[8]; uint32_t l, d, rank, na, nb, child_rank, children, r; } Header;
typedef struct { const float *p, *a, *b, *child_projection, *child_keys; const uint16_t *fa, *fb; } Layer;

static double now_s(void) {
    LARGE_INTEGER frequency, tick;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&tick);
    return (double)tick.QuadPart / (double)frequency.QuadPart;
}

static size_t rss_bytes(void) {
    PROCESS_MEMORY_COUNTERS counters;
    return GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof counters)
           ? (size_t)counters.WorkingSetSize : 0;
}

static float bf_float(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof result);
    return result;
}

static uint16_t float_bf(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    if ((bits & 0x7f800000u) == 0x7f800000u) return (uint16_t)(bits >> 16);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float round_bf(float value) { return bf_float(float_bf(value)); }

static void *read_all(const char *path, size_t *size) {
    FILE *file = fopen(path, "rb");
    if (!file) { perror(path); exit(2); }
    if (fseek(file, 0, SEEK_END) != 0) exit(2);
    long length = ftell(file);
    if (length < 0 || fseek(file, 0, SEEK_SET) != 0) exit(2);
    void *data = malloc((size_t)length);
    if (!data || fread(data, 1, (size_t)length, file) != (size_t)length) exit(2);
    fclose(file);
    *size = (size_t)length;
    return data;
}

static int better(float left, int li, float right, int ri) {
    return left > right || (left == right && li < ri);
}

static void top4(const float *scores, int count, int *ids) {
    int used[16] = {0};
    if (count > 16) { fprintf(stderr, "axis too wide\n"); exit(2); }
    for (int k = 0; k < 4; ++k) {
        int best = -1;
        for (int i = 0; i < count; ++i)
            if (!used[i] && (best < 0 || better(scores[i], i, scores[best], best))) best = i;
        ids[k] = best;
        used[best] = 1;
    }
}

static void route_parent(const Layer *layer, const float *x, const Header *h,
                         int *ids, float *gates) {
    float q[64], as[8], bs[16], candidate_scores[16];
    int ai[4], bi[4], candidate_ids[16], top[4];
    for (uint32_t r = 0; r < h->rank; ++r) {
        float sum = 0;
        const float *row = layer->p + (size_t)r * h->d;
        for (uint32_t d = 0; d < h->d; ++d) sum += row[d] * x[d];
        q[r] = sum;
    }
    for (uint32_t a = 0; a < h->na; ++a) {
        float sum = 0;
        for (uint32_t r = 0; r < h->rank; ++r)
            sum += layer->a[(size_t)a * h->rank + r] * q[r];
        as[a] = sum;
    }
    for (uint32_t b = 0; b < h->nb; ++b) {
        float sum = 0;
        for (uint32_t r = 0; r < h->rank; ++r)
            sum += layer->b[(size_t)b * h->rank + r] * q[r];
        bs[b] = sum;
    }
    top4(as, (int)h->na, ai);
    top4(bs, (int)h->nb, bi);
    for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) {
        int index = 4 * a + b;
        candidate_scores[index] = as[ai[a]] + bs[bi[b]];
        candidate_ids[index] = ai[a] * (int)h->nb + bi[b];
    }
    top4(candidate_scores, 16, top);
    float maximum = candidate_scores[top[0]], sum = 0;
    for (int k = 0; k < 4; ++k) {
        ids[k] = candidate_ids[top[k]];
        gates[k] = expf(candidate_scores[top[k]] - maximum);
        sum += gates[k];
    }
    for (int k = 0; k < 4; ++k) gates[k] = round_bf(gates[k] / sum);
}

static void route_child(const Layer *layer, const float *x, const Header *h,
                        const int *parents, int *children) {
    float q[32];
    for (uint32_t r = 0; r < h->child_rank; ++r) {
        float sum = 0;
        const float *row = layer->child_projection + (size_t)r * h->d;
        for (uint32_t d = 0; d < h->d; ++d) sum += row[d] * x[d];
        q[r] = sum;
    }
    for (int k = 0; k < 4; ++k) {
        int best = -1;
        float maximum = 0;
        for (uint32_t c = 0; c < h->children; ++c) {
            const float *key = layer->child_keys +
                               ((size_t)parents[k] * h->children + c) * h->child_rank;
            float score = 0;
            for (uint32_t r = 0; r < h->child_rank; ++r) score += q[r] * key[r];
            if (best < 0 || score > maximum) { best = (int)c; maximum = score; }
        }
        children[k] = parents[k] * (int)h->children + best;
    }
}

static void residual(const Layer *layer, const float *x, const Header *h,
                     const int *ids, const float *gates, float *out) {
    float hidden[4][8];
    for (int k = 0; k < 4; ++k) for (uint32_t r = 0; r < h->r; ++r) {
        size_t a_id = !memcmp(h->magic, "M126FB01", 8)
                      ? (size_t)ids[k] / h->children : (size_t)ids[k];
        const uint16_t *row = layer->fa + (a_id * h->r + r) * h->d;
        float sum = 0;
        for (uint32_t d = 0; d < h->d; ++d) sum += bf_float(row[d]) * x[d];
        sum = round_bf(sum);
        hidden[k][r] = round_bf(sum / (1.0f + expf(-sum)));
    }
    for (uint32_t d = 0; d < h->d; ++d) {
        float sum = 0;
        for (int k = 0; k < 4; ++k) {
            const uint16_t *row = layer->fb + ((size_t)ids[k] * h->d + d) * h->r;
            float part = 0;
            for (uint32_t r = 0; r < h->r; ++r) part += hidden[k][r] * bf_float(row[r]);
            part = round_bf(part);
            sum += round_bf(part * gates[k]);
        }
        out[d] = round_bf(sum);
    }
}

static int compare_route(const int *parent_ids, const int *child_ids,
                         const float *gates, const uint32_t *expected_parent,
                         const uint32_t *expected_child, const float *expected_gate,
                         float *worst_gate) {
    for (int k = 0; k < 4; ++k) {
        int matched = -1;
        for (int j = 0; j < 4; ++j)
            if ((uint32_t)parent_ids[k] == expected_parent[j]) matched = j;
        if (matched < 0 || (uint32_t)child_ids[k] != expected_child[matched]) return 0;
        float difference = fabsf(gates[k] - expected_gate[matched]);
        if (difference > *worst_gate) *worst_gate = difference;
    }
    return 1;
}

static int cmp_double(const void *left, const void *right) {
    double a = *(const double *)left, b = *(const double *)right;
    return (a > b) - (a < b);
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s BANK FIXTURES\n", argv[0]); return 2; }
    size_t bank_size;
    const uint8_t *bank = read_all(argv[1], &bank_size);
    if (bank_size < sizeof(Header)) return 2;
    Header h;
    memcpy(&h, bank, sizeof h);
    int shared_a = !memcmp(h.magic, "M126FB01", 8);
    if ((!shared_a && memcmp(h.magic, "M124FB01", 8)) || h.l != 24 || h.d != 896 ||
        h.rank != 64 || h.na != 8 || h.nb != 16 ||
        h.child_rank != 32 || h.children != 10 || h.r != 8) {
        fprintf(stderr, "invalid bank header\n"); return 2;
    }
    size_t parent_bytes = ((size_t)h.rank * h.d + (h.na + h.nb) * h.rank) * 4;
    size_t projection_bytes = (size_t)h.child_rank * h.d * 4;
    size_t key_bytes = (size_t)h.na * h.nb * h.children * h.child_rank * 4;
    size_t router_bytes = parent_bytes + projection_bytes + key_bytes;
    size_t factor_bytes = (size_t)h.na * h.nb * h.children * h.r * h.d * sizeof(uint16_t);
    size_t a_bytes = shared_a ? factor_bytes / h.children : factor_bytes;
    size_t layer_bytes = router_bytes + a_bytes + factor_bytes;
    if (bank_size != sizeof(Header) + h.l * layer_bytes) {
        fprintf(stderr, "invalid bank length\n"); return 2;
    }
    Layer layers[24];
    for (uint32_t l = 0; l < h.l; ++l) {
        const uint8_t *start = bank + sizeof(Header) + l * layer_bytes;
        layers[l].p = (const float *)start;
        layers[l].a = layers[l].p + (size_t)h.rank * h.d;
        layers[l].b = layers[l].a + (size_t)h.na * h.rank;
        layers[l].child_projection = (const float *)(start + parent_bytes);
        layers[l].child_keys = (const float *)(start + parent_bytes + projection_bytes);
        layers[l].fa = (const uint16_t *)(start + router_bytes);
        layers[l].fb = (const uint16_t *)(start + router_bytes + a_bytes);
    }
    FILE *file = fopen(argv[2], "rb");
    if (!file) { perror(argv[2]); return 2; }
    Header fh;
    if (fread(&fh, sizeof fh, 1, file) != 1 || memcmp(fh.magic, "M124FX01", 8) ||
        memcmp((const uint8_t *)&fh + 8, (const uint8_t *)&h + 8, sizeof h - 8)) {
        fprintf(stderr, "invalid fixture header\n"); return 2;
    }
    int cases = 0, failures = 0;
    float worst_gate = 0, worst_residual = 0;
    float first_x[24][896];
    for (;;) {
        uint32_t li;
        if (fread(&li, sizeof li, 1, file) != 1) break;
        uint16_t xbf[896];
        uint32_t expected_parent[4], expected_child[4];
        float expected_gate[4], expected_out[896];
        if (li >= h.l || li != (uint32_t)(cases / 4) ||
            fread(xbf, sizeof(uint16_t), h.d, file) != h.d ||
            fread(expected_parent, sizeof(uint32_t), 4, file) != 4 ||
            fread(expected_child, sizeof(uint32_t), 4, file) != 4 ||
            fread(expected_gate, sizeof(float), 4, file) != 4 ||
            fread(expected_out, sizeof(float), h.d, file) != h.d) {
            fprintf(stderr, "truncated fixture\n"); return 2;
        }
        float x[896], gates[4];
        int parent_ids[4], child_ids[4];
        for (uint32_t d = 0; d < h.d; ++d) x[d] = bf_float(xbf[d]);
        if (cases == (int)li * 4) memcpy(first_x[li], x, sizeof x);
        route_parent(&layers[li], x, &h, parent_ids, gates);
        route_child(&layers[li], x, &h, parent_ids, child_ids);
        float gate_error = 0, residual_error = 0;
        int okay = compare_route(parent_ids, child_ids, gates, expected_parent,
                                 expected_child, expected_gate, &gate_error);
        if (okay) {
            float actual_out[896];
            residual(&layers[li], x, &h, child_ids, gates, actual_out);
            for (uint32_t d = 0; d < h.d; ++d) {
                float difference = fabsf(actual_out[d] - expected_out[d]);
                if (difference > residual_error) residual_error = difference;
            }
        }
        if (!okay || gate_error > 1e-5f || residual_error > 0.03f) ++failures;
        if (gate_error > worst_gate) worst_gate = gate_error;
        if (residual_error > worst_residual) worst_residual = residual_error;
        ++cases;
        printf("case=%d layer=%u route_match=%d gate_max_abs=%.9g residual_max_abs=%.9g\n",
               cases - 1, li, okay, gate_error, residual_error);
    }
    fclose(file);
    printf("SUMMARY cases=%d failures=%d gate_max_abs=%.9g residual_max_abs=%.9g\n",
           cases, failures, worst_gate, worst_residual);
    if (cases != 96 || failures) { free((void *)bank); return 1; }
    double parent_ms[5], hierarchy_ms[5], residual_ms[5];
    volatile double checksum = 0;
    for (int rep = 0; rep < 5; ++rep) {
        double begin = now_s();
        for (int token = 0; token < 1024; ++token)
            for (int l = 0; l < 24; ++l) {
                int parent_ids[4]; float gates[4];
                route_parent(&layers[l], first_x[l], &h, parent_ids, gates);
                checksum += parent_ids[0] + gates[0];
            }
        parent_ms[rep] = (now_s() - begin) * 1000.0 / 1024.0;
        begin = now_s();
        for (int token = 0; token < 1024; ++token)
            for (int l = 0; l < 24; ++l) {
                int parent_ids[4], child_ids[4]; float gates[4];
                route_parent(&layers[l], first_x[l], &h, parent_ids, gates);
                route_child(&layers[l], first_x[l], &h, parent_ids, child_ids);
                checksum += child_ids[0] + gates[0];
            }
        hierarchy_ms[rep] = (now_s() - begin) * 1000.0 / 1024.0;
        begin = now_s();
        for (int token = 0; token < 1024; ++token)
            for (int l = 0; l < 24; ++l) {
                int parent_ids[4], child_ids[4]; float gates[4], out[896];
                route_parent(&layers[l], first_x[l], &h, parent_ids, gates);
                route_child(&layers[l], first_x[l], &h, parent_ids, child_ids);
                residual(&layers[l], first_x[l], &h, child_ids, gates, out);
                checksum += out[(token + l) % 896];
            }
        residual_ms[rep] = (now_s() - begin) * 1000.0 / 1024.0;
        printf("rep=%d parent_route_ms=%.6f hierarchy_route_ms=%.6f combined_ms=%.6f\n",
               rep, parent_ms[rep], hierarchy_ms[rep], residual_ms[rep]);
    }
    qsort(parent_ms, 5, sizeof(double), cmp_double);
    qsort(hierarchy_ms, 5, sizeof(double), cmp_double);
    qsort(residual_ms, 5, sizeof(double), cmp_double);
    printf("TIMING parent_route_median_ms=%.6f parent_route_min_ms=%.6f parent_route_max_ms=%.6f "
           "hierarchy_route_median_ms=%.6f hierarchy_route_min_ms=%.6f hierarchy_route_max_ms=%.6f "
           "combined_median_ms=%.6f combined_min_ms=%.6f combined_max_ms=%.6f "
           "route_ratio=%.6f rss_bytes=%zu bank_bytes=%zu selected_factor_bytes_per_token=%zu checksum=%.6f\n",
           parent_ms[2], parent_ms[0], parent_ms[4],
           hierarchy_ms[2], hierarchy_ms[0], hierarchy_ms[4],
           residual_ms[2], residual_ms[0], residual_ms[4],
           hierarchy_ms[2] / parent_ms[2], rss_bytes(), bank_size,
           (size_t)4 * 2 * h.r * h.d * sizeof(uint16_t) * h.l, (double)checksum);
    free((void *)bank);
    return 0;
}
