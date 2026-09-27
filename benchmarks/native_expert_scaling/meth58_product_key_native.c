// Native component parity reader for the learned METH-56 E128 product-key bank.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char magic[8]; uint32_t l, d, r, e, rank, na, nb; } Header;
typedef struct { const float *p, *a, *b; const uint16_t *fa, *fb; } Layer;

static float bf_float(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t float_bf(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
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
    for (int k = 0; k < 4; ++k) {
        int best = -1;
        for (int i = 0; i < count; ++i)
            if (!used[i] && (best < 0 || better(scores[i], i, scores[best], best))) best = i;
        ids[k] = best;
        used[best] = 1;
    }
}

static void route(const Layer *layer, const float *x, const Header *h,
                  int *ids, float *gates) {
    float q[64], as[8], bs[16], cand_score[16];
    int ai[4], bi[4], cand_id[16], top[4];
    for (uint32_t r = 0; r < h->rank; ++r) {
        float sum = 0;
        const float *row = layer->p + (size_t)r * h->d;
        for (uint32_t d = 0; d < h->d; ++d) sum += row[d] * x[d];
        q[r] = sum;
    }
    for (uint32_t a = 0; a < h->na; ++a) {
        float sum = 0;
        for (uint32_t r = 0; r < h->rank; ++r) sum += layer->a[(size_t)a * h->rank + r] * q[r];
        as[a] = sum;
    }
    for (uint32_t b = 0; b < h->nb; ++b) {
        float sum = 0;
        for (uint32_t r = 0; r < h->rank; ++r) sum += layer->b[(size_t)b * h->rank + r] * q[r];
        bs[b] = sum;
    }
    top4(as, (int)h->na, ai);
    top4(bs, (int)h->nb, bi);
    for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) {
        int index = 4 * a + b;
        cand_score[index] = as[ai[a]] + bs[bi[b]];
        cand_id[index] = ai[a] * (int)h->nb + bi[b];
    }
    top4(cand_score, 16, top);
    float maximum = cand_score[top[0]], sum = 0;
    for (int k = 0; k < 4; ++k) {
        ids[k] = cand_id[top[k]];
        gates[k] = expf(cand_score[top[k]] - maximum);
        sum += gates[k];
    }
    for (int k = 0; k < 4; ++k) gates[k] = round_bf(gates[k] / sum);
}

static void residual(const Layer *layer, const float *x, const Header *h,
                     const int *ids, const float *gates, float *out) {
    float hidden[4][8];
    for (int k = 0; k < 4; ++k) for (uint32_t r = 0; r < h->r; ++r) {
        const uint16_t *row = layer->fa + ((size_t)ids[k] * h->r + r) * h->d;
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

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s BANK FIXTURES\n", argv[0]); return 2; }
    size_t bank_size;
    const uint8_t *bank = read_all(argv[1], &bank_size);
    if (bank_size < sizeof(Header)) return 2;
    Header h;
    memcpy(&h, bank, sizeof(h));
    if (memcmp(h.magic, "M58PK001", 8) || h.l != 24 || h.d != 896 || h.r != 8 ||
        h.e != 128 || h.rank != 64 || h.na != 8 || h.nb != 16) {
        fprintf(stderr, "invalid bank header\n"); return 2;
    }
    size_t router_bytes = ((size_t)h.rank * h.d + (h.na + h.nb) * h.rank) * sizeof(float);
    size_t factor_bytes = (size_t)h.e * h.r * h.d * sizeof(uint16_t);
    size_t layer_bytes = router_bytes + 2 * factor_bytes;
    if (bank_size != sizeof(Header) + h.l * layer_bytes) {
        fprintf(stderr, "invalid bank length\n"); return 2;
    }
    Layer layers[24];
    for (uint32_t l = 0; l < h.l; ++l) {
        const uint8_t *start = bank + sizeof(Header) + l * layer_bytes;
        layers[l].p = (const float *)start;
        layers[l].a = layers[l].p + (size_t)h.rank * h.d;
        layers[l].b = layers[l].a + (size_t)h.na * h.rank;
        layers[l].fa = (const uint16_t *)(start + router_bytes);
        layers[l].fb = (const uint16_t *)(start + router_bytes + factor_bytes);
    }
    FILE *file = fopen(argv[2], "rb");
    if (!file) { perror(argv[2]); return 2; }
    Header fh;
    if (fread(&fh, sizeof(fh), 1, file) != 1 || memcmp(fh.magic, "M58FX001", 8) ||
        memcmp((const uint8_t *)&fh + 8, (const uint8_t *)&h + 8, sizeof(Header) - 8)) {
        fprintf(stderr, "invalid fixture header\n"); return 2;
    }
    int cases = 0, misses = 0, gate_failures = 0, residual_failures = 0;
    float worst_gate = 0, worst_residual = 0;
    for (;;) {
        uint32_t li;
        if (fread(&li, sizeof(li), 1, file) != 1) break;
        uint16_t xbf[896]; uint32_t expected_id[4]; float expected_gate[4], expected_out[896];
        if (li >= h.l || fread(xbf, sizeof(uint16_t), h.d, file) != h.d ||
            fread(expected_id, sizeof(uint32_t), 4, file) != 4 ||
            fread(expected_gate, sizeof(float), 4, file) != 4 ||
            fread(expected_out, sizeof(float), h.d, file) != h.d) {
            fprintf(stderr, "truncated fixture\n"); return 2;
        }
        float x[896], actual_out[896], gates[4]; int ids[4], route_miss = 0;
        for (uint32_t d = 0; d < h.d; ++d) x[d] = bf_float(xbf[d]);
        route(&layers[li], x, &h, ids, gates);
        for (int k = 0; k < 4; ++k) {
            int found = 0;
            for (int j = 0; j < 4; ++j) if ((uint32_t)ids[j] == expected_id[k]) found = 1;
            if (!found) route_miss = 1;
        }
        float gate_error = 0, residual_error = 0;
        if (!route_miss) {
            for (int k = 0; k < 4; ++k) for (int j = 0; j < 4; ++j)
                if ((uint32_t)ids[k] == expected_id[j]) {
                    float diff = fabsf(gates[k] - expected_gate[j]);
                    if (diff > gate_error) gate_error = diff;
                }
            residual(&layers[li], x, &h, ids, gates, actual_out);
            for (uint32_t d = 0; d < h.d; ++d) {
                float diff = fabsf(actual_out[d] - expected_out[d]);
                if (diff > residual_error) residual_error = diff;
            }
        }
        cases++; misses += route_miss;
        gate_failures += gate_error > 1e-5f;
        residual_failures += residual_error > 0.03f;
        if (gate_error > worst_gate) worst_gate = gate_error;
        if (residual_error > worst_residual) worst_residual = residual_error;
        printf("case=%d layer=%u route_miss=%d gate_max_abs=%.9g residual_max_abs=%.9g\n",
               cases - 1, li, route_miss, gate_error, residual_error);
    }
    fclose(file);
    free((void *)bank);
    printf("SUMMARY cases=%d route_misses=%d gate_failures=%d residual_failures=%d gate_max_abs=%.9g residual_max_abs=%.9g\n",
           cases, misses, gate_failures, residual_failures, worst_gate, worst_residual);
    return (cases == 96 && !misses && !gate_failures && !residual_failures) ? 0 : 1;
}
