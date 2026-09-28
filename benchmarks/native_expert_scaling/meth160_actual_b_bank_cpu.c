// METH-160: mapped learned E1280/E12800 B-bank selected-path cost.
#define M151_RUNTIME_ONLY
#include "meth151_shared_route_cpu.c"
#undef M151_RUNTIME_ONLY

typedef struct { char magic[8]; uint32_t layers, width, rank, rows; } BHeader;
typedef struct {
    HANDLE file, mapping;
    const uint8_t *data;
    size_t bytes;
    uint32_t rows;
    const uint16_t *layer[24];
} BBank;

static BBank map_b_bank(const char *path, uint32_t rows) {
    BBank result = {0};
    result.file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (result.file == INVALID_HANDLE_VALUE) { fprintf(stderr, "open B bank failed: %s\n", path); exit(2); }
    LARGE_INTEGER length;
    if (!GetFileSizeEx(result.file, &length) || length.QuadPart < 0) exit(2);
    result.bytes = (size_t)length.QuadPart;
    size_t stride = (size_t)rows * 896 * 8 * sizeof(uint16_t);
    if (result.bytes != sizeof(BHeader) + 24 * stride) {
        fprintf(stderr, "invalid B bank length: %s\n", path); exit(2);
    }
    result.mapping = CreateFileMappingA(result.file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!result.mapping) { fprintf(stderr, "map B bank failed: %s\n", path); exit(2); }
    result.data = (const uint8_t *)MapViewOfFile(result.mapping, FILE_MAP_READ, 0, 0, 0);
    if (!result.data) { fprintf(stderr, "view B bank failed: %s\n", path); exit(2); }
    BHeader header;
    memcpy(&header, result.data, sizeof header);
    if (memcmp(header.magic, "M136BF01", 8) || header.layers != 24 ||
        header.width != 896 || header.rank != 8 || header.rows != rows) {
        fprintf(stderr, "invalid B bank header: %s\n", path); exit(2);
    }
    result.rows = rows;
    for (int li = 0; li < 24; ++li)
        result.layer[li] = (const uint16_t *)(result.data + sizeof header + (size_t)li * stride);
    return result;
}

static void unmap_b_bank(BBank *bank) {
    UnmapViewOfFile(bank->data);
    CloseHandle(bank->mapping);
    CloseHandle(bank->file);
}

static uint32_t page_faults(void) {
    PROCESS_MEMORY_COUNTERS counters;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof counters)) exit(2);
    return counters.PageFaultCount;
}

// Same BF16 arithmetic/rounding order as METH-124 residual(), with shared
// A addressed by the original parent and B by the selected grandchild.
static void residual_grand(const Layer *layer, const uint16_t *grand_b,
                           const float *x, const Header *h,
                           const int *grand, const float *gates, float *out) {
    float hidden[4][8];
    for (int k = 0; k < 4; ++k) for (uint32_t r = 0; r < h->r; ++r) {
        size_t parent = (size_t)grand[k] / 100;
        const uint16_t *row = layer->fa + (parent * h->r + r) * h->d;
        float sum = 0;
        for (uint32_t d = 0; d < h->d; ++d) sum += bf_float(row[d]) * x[d];
        sum = round_bf(sum);
        hidden[k][r] = round_bf(sum / (1.0f + expf(-sum)));
    }
    for (uint32_t d = 0; d < h->d; ++d) {
        float sum = 0;
        for (int k = 0; k < 4; ++k) {
            const uint16_t *row = grand_b + ((size_t)grand[k] * h->d + d) * h->r;
            float part = 0;
            for (uint32_t r = 0; r < h->r; ++r) part += hidden[k][r] * bf_float(row[r]);
            part = round_bf(part);
            sum += round_bf(part * gates[k]);
        }
        out[d] = round_bf(sum);
    }
}

static void choose_grand(const int *children, int *grand, int structural,
                         const TokenContext *ctx, int layer) {
    for (int k = 0; k < 4; ++k) {
        grand[k] = structural ? children[k] * 10 :
                   hash_content(ctx->token, ctx->previous, ctx->position,
                                children[k], layer);
        if (grand[k] / 10 != children[k] || grand[k] < 0 || grand[k] >= 12800 ||
            (grand[k] % 10 == 0) != structural) {
            fprintf(stderr, "grandchild route identity failure\n"); exit(1);
        }
    }
}

static double time_route_factor(const Layer *layers, const Header *h,
                                const float *states, const TokenContext *contexts,
                                const TokenContext *table, uint32_t table_count,
                                const BBank *bank, int expanded,
                                volatile double *checksum, uint32_t *fault_delta) {
    uint32_t faults_before = page_faults();
    double started = now_s();
    for (int token = 0; token < 256; ++token) {
        int structural = expanded && is_shared(table, table_count, &contexts[token]);
        for (int li = 0; li < 24; ++li) {
            const float *x = states + ((size_t)token * 24 + li) * 896;
            int child[4], grand[4];
            float gates[4], output[896];
            select_base(&layers[li], x, h, child, gates);
            if (expanded) {
                choose_grand(child, grand, structural, &contexts[token], li);
                residual_grand(&layers[li], bank->layer[li], x, h, grand, gates, output);
            } else {
                Layer control = layers[li];
                control.fb = bank->layer[li];
                residual(&control, x, h, child, gates, output);
            }
            *checksum += output[(token + li) % 896];
        }
    }
    double elapsed = (now_s() - started) * 1000.0 / 256.0;
    *fault_delta = page_faults() - faults_before;
    return elapsed;
}

static int selftest(void) {
    const size_t row = (size_t)896 * 8;
    uint16_t *a = calloc((size_t)11 * 8 * 896, sizeof(uint16_t));
    uint16_t *source_b = calloc((size_t)102 * row, sizeof(uint16_t));
    uint16_t *grand_b = calloc((size_t)1011 * row, sizeof(uint16_t));
    if (!a || !source_b || !grand_b) return 2;
    for (size_t i = 0; i < (size_t)11 * 8 * 896; ++i)
        a[i] = float_bf(((int)(i % 17) - 8) * 0.002f);
    for (size_t i = 0; i < (size_t)102 * row; ++i)
        source_b[i] = float_bf(((int)(i % 23) - 11) * 0.001f);
    int child[4] = {89, 90, 100, 101};
    int grand[4] = {899, 900, 1009, 1010};
    for (int k = 0; k < 4; ++k)
        memcpy(grand_b + (size_t)grand[k] * row,
               source_b + (size_t)child[k] * row, row * sizeof(uint16_t));
    float x[896], gates[4] = {0.125f, 0.25f, 0.375f, 0.25f};
    float control[896], expanded[896];
    for (int i = 0; i < 896; ++i) x[i] = bf_float(float_bf(((i % 29) - 14) * 0.02f));
    Header h = {0};
    memcpy(h.magic, "M126FB01", 8);
    h.d = 896; h.r = 8; h.children = 10;
    Layer layer = {0};
    layer.fa = a; layer.fb = source_b;
    residual(&layer, x, &h, child, gates, control);
    residual_grand(&layer, grand_b, x, &h, grand, gates, expanded);
    int exact = memcmp(control, expanded, sizeof control) == 0;
    printf("M160_SELFTEST cases=4 exact_residual=%d a_parent_ids=8,9,10,10\n", exact);
    free(grand_b); free(source_b); free(a);
    return exact ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--selftest") == 0) return selftest();
    if (argc != 7) {
        fprintf(stderr, "usage: %s EXACT_BANK VECTORS CONTEXT TABLE CONTROL_B CANDIDATE_B\n", argv[0]);
        return 2;
    }
    double start = now_s();
    size_t bank_bytes, vector_bytes, context_bytes, table_bytes;
    const uint8_t *bank = read_all(argv[1], &bank_bytes);
    const uint8_t *vectors = read_all(argv[2], &vector_bytes);
    const uint8_t *fixture = read_all(argv[3], &context_bytes);
    const uint8_t *table_data = read_all(argv[4], &table_bytes);
    Header h; VectorHeader v; ContextHeader t; TableHeader th;
    if (bank_bytes < sizeof h || vector_bytes < sizeof v ||
        context_bytes < sizeof t || table_bytes < sizeof th) return 2;
    memcpy(&h, bank, sizeof h);
    memcpy(&v, vectors, sizeof v);
    memcpy(&t, fixture, sizeof t);
    memcpy(&th, table_data, sizeof th);
    if (memcmp(h.magic, "M126FB01", 8) || h.l != 24 || h.d != 896 ||
        h.rank != 64 || h.na != 8 || h.nb != 16 || h.child_rank != 32 ||
        h.children != 10 || h.r != 8 || memcmp(v.magic, "M125HX01", 8) ||
        v.layers != 24 || v.tokens != 256 || v.width != 896 || v.experts != 1280 ||
        memcmp(t.magic, "M151TX01", 8) || t.tokens != 256 ||
        memcmp(th.magic, "M151TB01", 8) || th.count != 75 || th.threshold != 32) {
        fprintf(stderr, "invalid input header\n"); return 2;
    }
    size_t parent_bytes = ((size_t)h.rank * h.d + (h.na + h.nb) * h.rank) * 4;
    size_t projection_bytes = (size_t)h.child_rank * h.d * 4;
    size_t key_bytes = (size_t)1280 * h.child_rank * 4;
    size_t router_bytes = parent_bytes + projection_bytes + key_bytes;
    size_t a_bytes = (size_t)128 * h.r * h.d * 2;
    size_t b_bytes = (size_t)1280 * h.r * h.d * 2;
    size_t layer_bytes = router_bytes + a_bytes + b_bytes;
    if (bank_bytes != sizeof h + 24 * layer_bytes ||
        vector_bytes != sizeof v + (size_t)256 * 24 * 896 * 2 ||
        context_bytes != sizeof t + (size_t)256 * sizeof(TokenContext) ||
        table_bytes != sizeof th + (size_t)75 * sizeof(TokenContext)) {
        fprintf(stderr, "invalid input length\n"); return 2;
    }
    const TokenContext *contexts = (const TokenContext *)(fixture + sizeof t);
    const TokenContext *table = (const TokenContext *)(table_data + sizeof th);
    for (uint32_t i = 1; i < th.count; ++i)
        if (compare_context(&table[i-1], &table[i]) >= 0) {
            fprintf(stderr, "invalid structural table order\n"); return 2;
        }
    TokenContext recurrent = {11, 16948, 7};
    TokenContext content = {123, 45, 67};
    if (hash_content(123, 45, 67, 89, 3) != 899 ||
        !is_shared(table, th.count, &recurrent) ||
        is_shared(table, th.count, &content)) {
        fprintf(stderr, "hash/table golden mismatch\n"); return 1;
    }
    Layer layers[24];
    for (int li = 0; li < 24; ++li) {
        const uint8_t *base = bank + sizeof h + (size_t)li * layer_bytes;
        layers[li].p = (const float *)base;
        layers[li].a = layers[li].p + (size_t)h.rank * h.d;
        layers[li].b = layers[li].a + (size_t)h.na * h.rank;
        layers[li].child_projection = (const float *)(base + parent_bytes);
        layers[li].child_keys = (const float *)(base + parent_bytes + projection_bytes);
        layers[li].fa = (const uint16_t *)(base + router_bytes);
        layers[li].fb = (const uint16_t *)(base + router_bytes + a_bytes);
    }
    BBank control = map_b_bank(argv[5], 1280);
    BBank candidate = map_b_bank(argv[6], 12800);
    size_t elements = (size_t)256 * 24 * 896;
    float *states = malloc(elements * sizeof(float));
    if (!states) return 2;
    const uint16_t *bits = (const uint16_t *)(vectors + sizeof v);
    for (size_t i = 0; i < elements; ++i) states[i] = bf_float(bits[i]);
    const uint16_t *source_b[24];
    for (int li = 0; li < 24; ++li) source_b[li] = layers[li].fb;
    int control_unique[24] = {0}, candidate_unique[24] = {0};
    uint8_t control_seen[24][1280] = {{0}};
    uint8_t candidate_seen[24][12800] = {{0}};
    int structural_tokens = 0;
    volatile double checksum = 0;
    for (int token = 0; token < 256; ++token) {
        int structural = is_shared(table, th.count, &contexts[token]);
        structural_tokens += structural;
        for (int li = 0; li < 24; ++li) {
            const float *x = states + ((size_t)token * 24 + li) * 896;
            int child[4], grand[4]; float gates[4], source_out[896], control_out[896], candidate_out[896];
            select_base(&layers[li], x, &h, child, gates);
            choose_grand(child, grand, structural, &contexts[token], li);
            Layer source = layers[li], continued = layers[li];
            source.fb = source_b[li]; continued.fb = control.layer[li];
            residual(&source, x, &h, child, gates, source_out);
            residual(&continued, x, &h, child, gates, control_out);
            residual_grand(&layers[li], candidate.layer[li], x, &h,
                           grand, gates, candidate_out);
            for (int k = 0; k < 4; ++k) {
                if (child[k] < 0 || child[k] >= 1280 || !isfinite(gates[k])) return 1;
                if (!control_seen[li][child[k]]) { control_seen[li][child[k]] = 1; control_unique[li]++; }
                if (!candidate_seen[li][grand[k]]) { candidate_seen[li][grand[k]] = 1; candidate_unique[li]++; }
            }
            for (int d = 0; d < 896; ++d)
                if (!isfinite(source_out[d]) || !isfinite(control_out[d]) ||
                    !isfinite(candidate_out[d])) {
                    fprintf(stderr, "nonfinite residual\n"); return 1;
                }
            checksum += source_out[(token + li) % 896] +
                        control_out[(token + li) % 896] + candidate_out[(token + li) % 896];
        }
    }
    double control_ms[5], candidate_ms[5], control_route_ms[5], candidate_route_ms[5];
    uint32_t control_faults[5], candidate_faults[5];
    for (int rep = 0; rep < 5; ++rep) {
        if (rep % 2 == 0) {
            control_ms[rep] = time_route_factor(layers, &h, states, contexts, table,
                                                 th.count, &control, 0, &checksum, &control_faults[rep]);
            candidate_ms[rep] = time_route_factor(layers, &h, states, contexts, table,
                                                   th.count, &candidate, 1, &checksum, &candidate_faults[rep]);
        } else {
            candidate_ms[rep] = time_route_factor(layers, &h, states, contexts, table,
                                                   th.count, &candidate, 1, &checksum, &candidate_faults[rep]);
            control_ms[rep] = time_route_factor(layers, &h, states, contexts, table,
                                                 th.count, &control, 0, &checksum, &control_faults[rep]);
        }
        control_route_ms[rep] = time_routes(layers, &h, states, contexts,
                                             table, th.count, 0, &checksum);
        candidate_route_ms[rep] = time_routes(layers, &h, states, contexts,
                                               table, th.count, 1, &checksum);
        printf("M160_REP rep=%d control_ms=%.9f candidate_ms=%.9f control_route_ms=%.9f candidate_route_ms=%.9f control_faults=%u candidate_faults=%u\n",
               rep, control_ms[rep], candidate_ms[rep], control_route_ms[rep],
               candidate_route_ms[rep], control_faults[rep], candidate_faults[rep]);
        fflush(stdout);
        if (now_s() - start > 300 || rss_bytes() > 10ull * (1ull << 30)) {
            fprintf(stderr, "METH-160 resource stop\n"); return 1;
        }
    }
    int control_total = 0, candidate_total = 0;
    for (int li = 0; li < 24; ++li) {
        control_total += control_unique[li]; candidate_total += candidate_unique[li];
        printf("M160_LAYER layer=%d control_unique=%d candidate_unique=%d\n",
               li, control_unique[li], candidate_unique[li]);
    }
    double base = median5(control_ms), expanded = median5(candidate_ms);
    int pass = expanded <= 5.0 && expanded <= 2.0 * base;
    size_t row_bytes = (size_t)896 * 8 * 2;
    printf("M160_SUMMARY tokens=256 states=6144 structural_tokens=%d control_unique=%d candidate_unique=%d control_unique_B_bytes=%zu candidate_unique_B_bytes=%zu selected_B_bytes_per_token=%zu control_bank_bytes=%zu candidate_bank_bytes=%zu control_median_ms=%.9f candidate_median_ms=%.9f ratio=%.6f control_route_median_ms=%.9f candidate_route_median_ms=%.9f rss_bytes=%zu elapsed_seconds=%.6f checksum=%.9f gate=%s\n",
           structural_tokens, control_total, candidate_total,
           (size_t)control_total * row_bytes, (size_t)candidate_total * row_bytes,
           (size_t)4 * 24 * row_bytes, control.bytes, candidate.bytes,
           base, expanded, expanded / base, median5(control_route_ms),
           median5(candidate_route_ms), rss_bytes(), now_s() - start,
           checksum, pass ? "PASS" : "FAIL");
    free(states);
    unmap_b_bank(&candidate); unmap_b_bank(&control);
    free((void *)table_data); free((void *)fixture);
    free((void *)vectors); free((void *)bank);
    return pass ? 0 : 1;
}
