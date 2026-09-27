// CPU cost of trained BF16-effective rank-8 selected factors.
// Expanded E values repeat E128 rows: timing only, no expanded quality claim.
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <psapi.h>

#define LAYERS 24
#define SEED_E 128
#define DIM 896
#define RANK 8
#define KEEP 4
#define FACTOR_ELEMENTS (2 * RANK * DIM)
#define FACTOR_BYTES (FACTOR_ELEMENTS * 2)
#define WARMUP 64
#define TOKENS 256
#define REPS 4

typedef struct {
    uint16_t *factors;
    int experts;
    size_t bytes;
} Pool;
typedef struct {
    double dispatch_s, projection_s, output_s;
    uint64_t checksum;
} Timing;

static double now_s(void) {
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
}

static size_t rss_bytes(void) {
    PROCESS_MEMORY_COUNTERS info;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof info)) return 0;
    return (size_t)info.WorkingSetSize;
}

static void *alloc_or_die(size_t bytes) {
    void *p = malloc(bytes);
    if (!p) { fprintf(stderr, "allocation failed: %zu bytes\n", bytes); exit(2); }
    return p;
}

static void read_exact(FILE *f, void *p, size_t bytes) {
    if (fread(p, 1, bytes, f) != bytes) { fputs("short seed read\n", stderr); exit(2); }
}

static float bf16_scalar(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof result);
    return result;
}

static float dot_scalar(const uint16_t *w, const float *x, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) acc += bf16_scalar(w[i]) * x[i];
    return acc;
}

static float dot_avx2(const uint16_t *w, const float *x, int n) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) {
        __m128i eight = _mm_loadu_si128((const __m128i *)(w + i));
        __m256i bits = _mm256_slli_epi32(_mm256_cvtepu16_epi32(eight), 16);
        acc = _mm256_fmadd_ps(_mm256_castsi256_ps(bits), _mm256_loadu_ps(x + i), acc);
    }
    float lanes[8]; _mm256_storeu_ps(lanes, acc);
    float result = 0.0f;
    for (int i = 0; i < 8; i++) result += lanes[i];
    return result;
}

static uint64_t next_random(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *state = x;
    return x;
}

static Pool load_expand(const char *path, int experts) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open seed: %s\n", path); exit(2); }
    char magic[8]; uint32_t shape[4];
    read_exact(f, magic, 8); read_exact(f, shape, sizeof shape);
    if (memcmp(magic, "M52BF16\0", 8) != 0 ||
        shape[0] != LAYERS || shape[1] != SEED_E ||
        shape[2] != DIM || shape[3] != RANK) {
        fputs("seed header mismatch\n", stderr); exit(2);
    }
    size_t seed_bytes = (size_t)LAYERS * SEED_E * FACTOR_BYTES;
    uint16_t *seed = alloc_or_die(seed_bytes);
    read_exact(f, seed, seed_bytes);
    if (fgetc(f) != EOF) { fputs("seed has trailing bytes\n", stderr); exit(2); }
    fclose(f);
    Pool pool = {0}; pool.experts = experts;
    pool.bytes = (size_t)LAYERS * (size_t)experts * FACTOR_BYTES;
    if (pool.bytes > 32ULL * 1024 * 1024 * 1024) {
        fputs("pool cap exceeded\n", stderr); exit(2);
    }
    pool.factors = alloc_or_die(pool.bytes);
    for (int li = 0; li < LAYERS; li++)
        for (int e = 0; e < experts; e++) {
            int source = (e * 73 + li * 7) & (SEED_E - 1);
            memcpy(pool.factors + ((size_t)li * experts + e) * FACTOR_ELEMENTS,
                   seed + ((size_t)li * SEED_E + source) * FACTOR_ELEMENTS,
                   FACTOR_BYTES);
        }
    free(seed);
    if (rss_bytes() > 32ULL * 1024 * 1024 * 1024) {
        fputs("RSS cap exceeded after pool expansion\n", stderr); exit(2);
    }
    return pool;
}

static uint32_t *make_routes(int experts) {
    size_t count = (size_t)(WARMUP + TOKENS * REPS) * LAYERS * KEEP;
    uint32_t *routes = alloc_or_die(count * sizeof(uint32_t));
    uint64_t rng = UINT64_C(0x9e3779b97f4a7c15);
    for (int t = 0; t < WARMUP + TOKENS * REPS; t++)
        for (int li = 0; li < LAYERS; li++) {
            uint32_t *chosen = routes + ((size_t)t * LAYERS + li) * KEEP;
            for (int j = 0; j < KEEP; j++) {
                uint32_t id; int duplicate;
                do {
                    id = (uint32_t)(next_random(&rng) % (uint64_t)experts);
                    duplicate = 0;
                    for (int prior = 0; prior < j; prior++)
                        if (chosen[prior] == id) duplicate = 1;
                } while (duplicate);
                chosen[j] = id;
            }
        }
    return routes;
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

static void one_token(const Pool *pool, const uint32_t *routes,
                      int token, int threads, Timing *timing) {
    const float gates[KEEP] = {0.4f, 0.3f, 0.2f, 0.1f};
    for (int li = 0; li < LAYERS; li++) {
        float x[DIM], h[KEEP][RANK], output[DIM];
        const uint16_t *selected[KEEP];
        double t0 = now_s();
        for (int d = 0; d < DIM; d++) {
            uint32_t value = (uint32_t)(token * 131 + li * 313 + d * 23);
            x[d] = ((float)(value % 1009) - 504.0f) / 504.0f;
        }
        const uint32_t *chosen = routes + ((size_t)token * LAYERS + li) * KEEP;
        for (int j = 0; j < KEEP; j++) {
            if (chosen[j] >= (uint32_t)pool->experts) { fputs("bad route ID\n", stderr); exit(2); }
            selected[j] = pool->factors + ((size_t)li * pool->experts + chosen[j]) * FACTOR_ELEMENTS;
        }
        double t1 = now_s();
        for (int j = 0; j < KEEP; j++)
            for (int r = 0; r < RANK; r++)
                h[j][r] = silu(dot_avx2(selected[j] + (size_t)r * DIM, x, DIM));
        double t2 = now_s();
        #pragma omp parallel for if(threads > 1) num_threads(threads) schedule(static)
        for (int d = 0; d < DIM; d++) {
            float sum = 0.0f;
            for (int j = 0; j < KEEP; j++) {
                const uint16_t *b = selected[j] + RANK * DIM + (size_t)d * RANK;
                sum += gates[j] * dot_avx2(b, h[j], RANK);
            }
            output[d] = sum;
        }
        double t3 = now_s();
        uint32_t bits;
        memcpy(&bits, &output[(token + li * 137) % DIM], sizeof bits);
        timing->checksum ^= bits;
        timing->checksum *= UINT64_C(1099511628211);
        double t4 = now_s();
        timing->dispatch_s += (t1 - t0) + (t4 - t3);
        timing->projection_s += t2 - t1;
        timing->output_s += t3 - t2;
    }
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static double median4(const double *values) {
    double sorted[REPS]; memcpy(sorted, values, sizeof sorted);
    qsort(sorted, REPS, sizeof(double), cmp_double);
    return 0.5 * (sorted[1] + sorted[2]);
}

int main(int argc, char **argv) {
    const char *seed_path = NULL, *out_path = NULL;
    int experts = 0, threads = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed_path = argv[++i];
        else if (!strcmp(argv[i], "--experts") && i + 1 < argc) experts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else { fputs("bad argument\n", stderr); return 2; }
    }
    if (!seed_path || !out_path || (experts != 128 && experts != 2735 && experts != 27355) ||
        (threads != 1 && threads != 6)) {
        fputs("usage: --seed FILE --experts 128|2735|27355 --threads 1|6 --out FILE\n", stderr);
        return 2;
    }
    omp_set_dynamic(0);
    double job_start = now_s();
    Pool pool = load_expand(seed_path, experts);
    uint32_t *routes = make_routes(experts);
    float x[DIM], h[RANK];
    for (int i = 0; i < DIM; i++) x[i] = ((float)(i % 17) - 8.0f) / 8.0f;
    for (int i = 0; i < RANK; i++) h[i] = ((float)(i % 5) - 2.0f) / 3.0f;
    float max_error = 0.0f;
    for (int e = 0; e < SEED_E; e++) {
        const uint16_t *factor = pool.factors + (size_t)e * FACTOR_ELEMENTS;
        for (int r = 0; r < RANK; r++) {
            const uint16_t *row = factor + (size_t)r * DIM;
            float reference = dot_scalar(row, x, DIM), got = dot_avx2(row, x, DIM);
            float error = fabsf(reference - got);
            if (error > max_error) max_error = error;
            if (error > 0.001f * (1.0f + fabsf(reference))) {
                fprintf(stderr, "A dot selftest failed e=%d r=%d\n", e, r);
                return 2;
            }
        }
        for (int d = 0; d < DIM; d += 7) {
            const uint16_t *row = factor + RANK * DIM + (size_t)d * RANK;
            float reference = dot_scalar(row, h, RANK), got = dot_avx2(row, h, RANK);
            float error = fabsf(reference - got);
            if (error > max_error) max_error = error;
            if (error > 0.001f * (1.0f + fabsf(reference))) {
                fprintf(stderr, "B dot selftest failed e=%d d=%d\n", e, d);
                return 2;
            }
        }
    }
    Timing warm = {0}; warm.checksum = UINT64_C(1469598103934665603);
    for (int t = 0; t < WARMUP; t++) one_token(&pool, routes, t, threads, &warm);
    double total[REPS], dispatch[REPS], projection[REPS], output[REPS];
    uint64_t checksums[REPS];
    for (int rep = 0; rep < REPS; rep++) {
        Timing timing = {0}; timing.checksum = UINT64_C(1469598103934665603);
        for (int t = 0; t < TOKENS; t++)
            one_token(&pool, routes, WARMUP + rep * TOKENS + t, threads, &timing);
        dispatch[rep] = timing.dispatch_s * 1000.0 / TOKENS;
        projection[rep] = timing.projection_s * 1000.0 / TOKENS;
        output[rep] = timing.output_s * 1000.0 / TOKENS;
        total[rep] = dispatch[rep] + projection[rep] + output[rep];
        checksums[rep] = timing.checksum;
        printf("rep %d/%d E=%d threads=%d total=%.3f ms/token\n",
               rep + 1, REPS, experts, threads, total[rep]); fflush(stdout);
        if (now_s() - job_start > 15 * 60) { fputs("METH-52 wall stop\n", stderr); return 2; }
        if (rss_bytes() > 32ULL * 1024 * 1024 * 1024) { fputs("RSS cap\n", stderr); return 2; }
    }
    FILE *f = fopen(out_path, "wb");
    if (!f) { fputs("cannot open output\n", stderr); return 2; }
    fprintf(f, "{\n\"experiment\":\"METH-52\",\"experts\":%d,\"threads\":%d,", experts, threads);
    fprintf(f, "\"layers\":%d,\"dimension\":%d,\"rank\":%d,\"selected_experts\":%d,",
            LAYERS, DIM, RANK, KEEP);
    fprintf(f, "\"warmup_tokens\":%d,\"measured_tokens_per_rep\":%d,\"repetitions\":%d,",
            WARMUP, TOKENS, REPS);
    fprintf(f, "\"pool_bytes\":%llu,\"selected_factor_bytes_per_token\":%llu,",
            (unsigned long long)pool.bytes, (unsigned long long)((size_t)LAYERS * KEEP * FACTOR_BYTES));
    fprintf(f, "\"selftest_max_abs_error\":%.9g,\"rss_end_bytes\":%llu,",
            max_error, (unsigned long long)rss_bytes());
    fprintf(f, "\"repetition_checksums\":[");
    for (int r = 0; r < REPS; r++) fprintf(f, "%s\"%016llx\"", r ? "," : "", (unsigned long long)checksums[r]);
    fprintf(f, "],\"repetitions_ms_per_token\":[");
    for (int r = 0; r < REPS; r++) fprintf(f, "%s%.9g", r ? "," : "", total[r]);
    fprintf(f, "],\"dispatch_ms_per_token\":[");
    for (int r = 0; r < REPS; r++) fprintf(f, "%s%.9g", r ? "," : "", dispatch[r]);
    fprintf(f, "],\"projection_ms_per_token\":[");
    for (int r = 0; r < REPS; r++) fprintf(f, "%s%.9g", r ? "," : "", projection[r]);
    fprintf(f, "],\"output_ms_per_token\":[");
    for (int r = 0; r < REPS; r++) fprintf(f, "%s%.9g", r ? "," : "", output[r]);
    fprintf(f, "],\"median_total_ms_per_token\":%.9g,", median4(total));
    fprintf(f, "\"median_dispatch_ms_per_token\":%.9g,", median4(dispatch));
    fprintf(f, "\"median_projection_ms_per_token\":%.9g,", median4(projection));
    fprintf(f, "\"median_output_ms_per_token\":%.9g,", median4(output));
    fprintf(f, "\"process_seconds\":%.9g}\n", now_s() - job_start);
    fclose(f);
    printf("median E=%d threads=%d: %.3f ms/token, checksum=%016llx\n",
           experts, threads, median4(total), (unsigned long long)checksums[0]);
    free(routes); free(pool.factors);
    return 0;
}
