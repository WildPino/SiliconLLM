// METH-174: paired 138/1685 shared-table lookup on actual E1280 states.
#define M151_RUNTIME_ONLY
#include "meth151_shared_route_cpu.c"
#undef M151_RUNTIME_ONLY

static int shared174(const TokenContext *table, uint32_t count,
                     const TokenContext *context) {
    return context->token == 151644 || context->token == 151645 ||
           is_shared(table, count, context);
}

static double time_routes174(const Layer *layers, const Header *header,
                             const float *states, const TokenContext *contexts,
                             const TokenContext *table, uint32_t table_count,
                             volatile double *checksum) {
    double started = now_s();
    for (int token = 0; token < 256; ++token) {
        int structural = shared174(table, table_count, &contexts[token]);
        for (int li = 0; li < 24; ++li) {
            const float *x = states + ((size_t)token * 24 + li) * 896;
            int child[4];
            float gate[4];
            select_base(&layers[li], x, header, child, gate);
            for (int k = 0; k < 4; ++k)
                child[k] = structural ? child[k] * 10 : hash_content(
                    contexts[token].token, contexts[token].previous,
                    contexts[token].position, child[k], li);
            *checksum += child[0] + gate[0];
        }
    }
    return (now_s() - started) * 1000.0 / 256.0;
}

static uint32_t page_faults174(void) {
    PROCESS_MEMORY_COUNTERS counters;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof counters)) exit(2);
    return counters.PageFaultCount;
}

static void verify_fixture(const TokenContext *contexts,
                           const TokenContext *old_table, uint32_t base_count,
                           const TokenContext *new_table, uint32_t candidate_count,
                           int old_expected, int new_expected) {
    for (int i = 0; i < 256; ++i) {
        int old_hit = shared174(old_table, base_count, &contexts[i]);
        int new_hit = shared174(new_table, candidate_count, &contexts[i]);
        if (old_hit != old_expected || new_hit != new_expected) {
            fprintf(stderr, "fixture shared-classification mismatch token=%d\n", i);
            exit(1);
        }
        int old_grand = old_hit ? 890 : hash_content(contexts[i].token,
            contexts[i].previous, contexts[i].position, 89, 3);
        int new_grand = new_hit ? 890 : hash_content(contexts[i].token,
            contexts[i].previous, contexts[i].position, 89, 3);
        if (old_grand / 10 != 89 || new_grand / 10 != 89 ||
            ((old_grand % 10) == 0) != old_hit ||
            ((new_grand % 10) == 0) != new_hit) {
            fprintf(stderr, "fixture grandchild identity mismatch token=%d\n", i);
            exit(1);
        }
    }
}

static int compare_cell(const char *name, const Layer *layers, const Header *h,
                        const float *states, const TokenContext *contexts,
                        const TokenContext *old_table, uint32_t base_count,
                        const TokenContext *new_table, uint32_t candidate_count,
                        int old_expected, int new_expected,
                        volatile double *checksum, double process_started) {
    verify_fixture(contexts, old_table, base_count, new_table, candidate_count,
                   old_expected, new_expected);
    int first_old = old_expected ? 890 : hash_content(contexts[0].token,
        contexts[0].previous, contexts[0].position, 89, 3);
    int first_new = new_expected ? 890 : hash_content(contexts[0].token,
        contexts[0].previous, contexts[0].position, 89, 3);
    printf("M174_GOLDEN cell=%s token=%u previous=%u position=%u old_grand=%d new_grand=%d\n",
           name, contexts[0].token, contexts[0].previous,
           contexts[0].position, first_old, first_new);
    // Both inputs are read and routed once before timed repetitions.
    time_routes174(layers, h, states, contexts, old_table, base_count, checksum);
    time_routes174(layers, h, states, contexts, new_table, candidate_count, checksum);
    double old_ms[5], new_ms[5];
    uint32_t faults_old[5], faults_new[5];
    for (int rep = 0; rep < 5; ++rep) {
        if (rep % 2 == 0) {
            uint32_t before = page_faults174();
            old_ms[rep] = time_routes174(layers, h, states, contexts,
                                         old_table, base_count, checksum);
            faults_old[rep] = page_faults174() - before;
            before = page_faults174();
            new_ms[rep] = time_routes174(layers, h, states, contexts,
                                         new_table, candidate_count, checksum);
            faults_new[rep] = page_faults174() - before;
        } else {
            uint32_t before = page_faults174();
            new_ms[rep] = time_routes174(layers, h, states, contexts,
                                         new_table, candidate_count, checksum);
            faults_new[rep] = page_faults174() - before;
            before = page_faults174();
            old_ms[rep] = time_routes174(layers, h, states, contexts,
                                         old_table, base_count, checksum);
            faults_old[rep] = page_faults174() - before;
        }
        printf("M174_REP cell=%s rep=%d old_ms=%.9f new_ms=%.9f old_faults=%u new_faults=%u\n",
               name, rep, old_ms[rep], new_ms[rep], faults_old[rep], faults_new[rep]);
        fflush(stdout);
        if (now_s() - process_started > 300.0 || rss_bytes() > 4ull * (1ull << 30)) {
            fprintf(stderr, "METH-174 resource stop\n"); exit(1);
        }
    }
    double old_median = median5(old_ms), new_median = median5(new_ms);
    int pass = new_median <= 3.5 && new_median <= 1.25 * old_median;
    printf("M174_CELL cell=%s tokens=256 states=6144 old_shared=%d new_shared=%d old_median_ms=%.9f new_median_ms=%.9f ratio=%.6f rss_bytes=%zu checksum=%.9f gate=%s\n",
           name, old_expected, new_expected, old_median, new_median,
           new_median / old_median, rss_bytes(), *checksum, pass ? "PASS" : "FAIL");
    fflush(stdout);
    return pass ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc != 9) {
        fprintf(stderr, "usage: %s EXACT_BANK VECTORS BASE_TABLE CANDIDATE_TABLE BASE_HIT CANDIDATE_HIT DELIMITER MISS\n", argv[0]);
        return 2;
    }
    double started = now_s();
    size_t bank_bytes, vector_bytes, old_bytes, new_bytes;
    const uint8_t *bank = read_all(argv[1], &bank_bytes);
    const uint8_t *vectors = read_all(argv[2], &vector_bytes);
    const uint8_t *old_data = read_all(argv[3], &old_bytes);
    const uint8_t *new_data = read_all(argv[4], &new_bytes);
    Header h; VectorHeader v; TableHeader old_header, new_header;
    if (bank_bytes < sizeof h || vector_bytes < sizeof v ||
        old_bytes < sizeof old_header || new_bytes < sizeof new_header) return 2;
    memcpy(&h, bank, sizeof h);
    memcpy(&v, vectors, sizeof v);
    memcpy(&old_header, old_data, sizeof old_header);
    memcpy(&new_header, new_data, sizeof new_header);
    if (memcmp(h.magic, "M126FB01", 8) || h.l != 24 || h.d != 896 ||
        h.rank != 64 || h.na != 8 || h.nb != 16 || h.child_rank != 32 ||
        h.children != 10 || h.r != 8 ||
        memcmp(v.magic, "M125HX01", 8) || v.layers != 24 ||
        v.tokens != 256 || v.width != 896 || v.experts != 1280 ||
        memcmp(old_header.magic, "M151TB01", 8) || old_header.count != 138 ||
        old_header.threshold != 32 ||
        memcmp(new_header.magic, "M151TB01", 8) || new_header.count != 1685 ||
        new_header.threshold != 8) {
        fprintf(stderr, "invalid bank/vector/table header\n"); return 2;
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
        old_bytes != sizeof old_header + (size_t)138 * sizeof(TokenContext) ||
        new_bytes != sizeof new_header + (size_t)1685 * sizeof(TokenContext)) {
        fprintf(stderr, "invalid bank/vector/table length\n"); return 2;
    }
    const TokenContext *old_table = (const TokenContext *)(old_data + sizeof old_header);
    const TokenContext *new_table = (const TokenContext *)(new_data + sizeof new_header);
    for (uint32_t i = 1; i < old_header.count; ++i)
        if (compare_context(&old_table[i-1], &old_table[i]) >= 0) return 2;
    for (uint32_t i = 1; i < new_header.count; ++i)
        if (compare_context(&new_table[i-1], &new_table[i]) >= 0) return 2;
    TokenContext golden_shared = {11, 16948, 7};
    TokenContext golden_content = {123, 45, 67};
    TokenContext golden_candidate_only = {11, 198, 56};
    TokenContext golden_delimiter = {151644, 123, 67};
    if (!shared174(old_table, 138, &golden_shared) ||
        !shared174(new_table, 1685, &golden_shared) ||
        shared174(old_table, 138, &golden_content) ||
        shared174(new_table, 1685, &golden_content) ||
        shared174(old_table, 138, &golden_candidate_only) ||
        !shared174(new_table, 1685, &golden_candidate_only) ||
        !shared174(old_table, 138, &golden_delimiter) ||
        !shared174(new_table, 1685, &golden_delimiter) ||
        hash_content(123, 45, 67, 89, 3) != 899) return 1;
    for (uint32_t i = 0; i < new_header.count; ++i)
        if (!shared174(new_table, new_header.count, &new_table[i])) return 1;
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
    size_t n_states = (size_t)256 * 24 * 896;
    float *states = malloc(n_states * sizeof(float));
    if (!states) return 2;
    const uint16_t *bits = (const uint16_t *)(vectors + sizeof v);
    for (size_t i = 0; i < n_states; ++i) states[i] = bf_float(bits[i]);
    const char *names[4] = {"base_hit", "candidate_hit", "delimiter", "miss"};
    int old_expected[4] = {1, 0, 1, 0};
    int new_expected[4] = {1, 1, 1, 0};
    int failures = 0;
    volatile double checksum = 0;
    for (int cell = 0; cell < 4; ++cell) {
        size_t fixture_bytes;
        const uint8_t *fixture = read_all(argv[5 + cell], &fixture_bytes);
        ContextHeader fh;
        if (fixture_bytes < sizeof fh) return 2;
        memcpy(&fh, fixture, sizeof fh);
        if (memcmp(fh.magic, "M151TX01", 8) || fh.tokens != 256 ||
            fixture_bytes != sizeof fh + (size_t)256 * sizeof(TokenContext)) return 2;
        const TokenContext *contexts = (const TokenContext *)(fixture + sizeof fh);
        failures += compare_cell(names[cell], layers, &h, states, contexts,
                                 old_table, 138, new_table, 1685,
                                 old_expected[cell], new_expected[cell],
                                 &checksum, started);
        free((void *)fixture);
    }
    printf("M174_SUMMARY cells=4 old_table_bytes=%zu new_table_bytes=%zu rss_bytes=%zu elapsed_seconds=%.6f checksum=%.9f gate=%s\n",
           old_bytes, new_bytes, rss_bytes(), now_s() - started,
           checksum, failures ? "FAIL" : "PASS");
    free(states);
    free((void *)new_data); free((void *)old_data);
    free((void *)vectors); free((void *)bank);
    return failures ? 1 : 0;
}

