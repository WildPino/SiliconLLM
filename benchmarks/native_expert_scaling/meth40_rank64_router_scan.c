// METH-40: CPU component cost of the stored rank-64 int8 router scan.
// Synthetic expansion of E128 seed rows measures traffic, not model quality.
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <omp.h>

#define LAYERS 24
#define SEED_E 128
#define DIM 896
#define RANK 64
#define KEEP 96
#define WARMUP 8
#define REPEATS 4
#define MAX_THREADS 16

typedef struct { float score; int id; } Entry;
typedef struct { Entry entries[KEEP]; int count; } Heap;
typedef struct {
    float *basis;
    int8_t *codes;
    float *scales;
    size_t pool_bytes;
    int experts;
} Pool;
typedef struct {
    double projection_s, scan_s, merge_s;
    uint64_t checksum;
} Timing;

static double now_s(void) {
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
}

static void *alloc_or_die(size_t bytes) {
    void *p = malloc(bytes);
    if (!p) { fprintf(stderr, "allocation failed: %zu bytes\n", bytes); exit(2); }
    return p;
}

static void read_exact(FILE *f, void *p, size_t bytes) {
    if (fread(p, 1, bytes, f) != bytes) { fputs("short seed read\n", stderr); exit(2); }
}

static int worse(Entry a, Entry b) {
    return a.score < b.score || (a.score == b.score && a.id > b.id);
}

static void heap_push(Heap *h, Entry x) {
    int n = h->count;
    if (n < KEEP) {
        h->entries[n] = x;
        h->count = n + 1;
        while (n > 0) {
            int parent = (n - 1) / 2;
            if (!worse(h->entries[n], h->entries[parent])) break;
            Entry tmp = h->entries[n]; h->entries[n] = h->entries[parent];
            h->entries[parent] = tmp; n = parent;
        }
        return;
    }
    if (!worse(h->entries[0], x)) return;
    h->entries[0] = x;
    int i = 0;
    for (;;) {
        int left = 2 * i + 1, right = left + 1;
        if (left >= KEEP) break;
        int worst = left;
        if (right < KEEP && worse(h->entries[right], h->entries[left])) worst = right;
        if (!worse(h->entries[worst], h->entries[i])) break;
        Entry tmp = h->entries[i]; h->entries[i] = h->entries[worst];
        h->entries[worst] = tmp; i = worst;
    }
}

static float dot_scalar(const int8_t *q, const float *x) {
    float acc = 0.0f;
    for (int i = 0; i < RANK; i++) acc += (float)q[i] * x[i];
    return acc;
}

static float dot_avx2(const int8_t *q, const float *x) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < RANK; i += 8) {
        __m128i eight = _mm_loadl_epi64((const __m128i *)(q + i));
        __m256i ints = _mm256_cvtepi8_epi32(eight);
        __m256 vf = _mm256_cvtepi32_ps(ints);
        acc = _mm256_fmadd_ps(vf, _mm256_loadu_ps(x + i), acc);
    }
    float lanes[8]; _mm256_storeu_ps(lanes, acc);
    float sum = 0.0f;
    for (int i = 0; i < 8; i++) sum += lanes[i];
    return sum;
}

static void project(const float *basis, const float *x, float *out) {
    __m256 acc[8];
    for (int j = 0; j < 8; j++) acc[j] = _mm256_setzero_ps();
    for (int d = 0; d < DIM; d++) {
        __m256 xx = _mm256_set1_ps(x[d]);
        const float *row = basis + (size_t)d * RANK;
        for (int j = 0; j < 8; j++)
            acc[j] = _mm256_fmadd_ps(xx, _mm256_loadu_ps(row + j * 8), acc[j]);
    }
    for (int j = 0; j < 8; j++) _mm256_storeu_ps(out + j * 8, acc[j]);
}

static Pool load_expand(const char *path, int experts) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open seed: %s\n", path); exit(2); }
    char magic[8]; uint32_t shape[4];
    read_exact(f, magic, 8); read_exact(f, shape, sizeof shape);
    if (memcmp(magic, "M40R64\0\0", 8) != 0 ||
        shape[0] != LAYERS || shape[1] != SEED_E ||
        shape[2] != DIM || shape[3] != RANK) {
        fputs("seed header mismatch\n", stderr); exit(2);
    }
    size_t basis_n = (size_t)LAYERS * DIM * RANK;
    size_t codes_n = (size_t)LAYERS * (size_t)experts * RANK;
    size_t scales_n = (size_t)LAYERS * (size_t)experts;
    Pool p = {0}; p.experts = experts;
    p.basis = alloc_or_die(basis_n * sizeof(float));
    p.codes = alloc_or_die(codes_n);
    p.scales = alloc_or_die(scales_n * sizeof(float));
    p.pool_bytes = basis_n * sizeof(float) + codes_n + scales_n * sizeof(float);
    int8_t seed_codes[SEED_E * RANK]; float seed_scales[SEED_E];
    for (int layer = 0; layer < LAYERS; layer++) {
        read_exact(f, p.basis + (size_t)layer * DIM * RANK,
                   (size_t)DIM * RANK * sizeof(float));
        read_exact(f, seed_codes, sizeof seed_codes);
        read_exact(f, seed_scales, sizeof seed_scales);
        for (int e = 0; e < experts; e++) {
            int src = (e * 73 + layer * 7) & (SEED_E - 1);
            memcpy(p.codes + ((size_t)layer * experts + e) * RANK,
                   seed_codes + src * RANK, RANK);
            int block = (e / SEED_E) % 17;
            p.scales[(size_t)layer * experts + e] =
                seed_scales[src] * (1.0f + (float)(block - 8) * 0.001f);
        }
    }
    if (fgetc(f) != EOF) { fputs("seed has trailing bytes\n", stderr); exit(2); }
    fclose(f);
    return p;
}

static float *make_inputs(int tokens) {
    size_t count = (size_t)tokens * LAYERS * DIM;
    float *x = alloc_or_die(count * sizeof(float));
    for (int t = 0; t < tokens; t++)
        for (int l = 0; l < LAYERS; l++)
            for (int d = 0; d < DIM; d++) {
                uint32_t v = (uint32_t)(t * 131 + l * 313 + d * 23);
                x[((size_t)t * LAYERS + l) * DIM + d] =
                    ((float)(v % 1009) - 504.0f) / 504.0f;
            }
    return x;
}

static int compare_entries(const void *pa, const void *pb) {
    Entry a = *(const Entry *)pa, b = *(const Entry *)pb;
    if (a.score > b.score) return -1;
    if (a.score < b.score) return 1;
    return a.id < b.id ? -1 : a.id > b.id;
}

static uint64_t hash_top(const Heap *h, uint64_t prior) {
    Entry sorted[KEEP];
    memcpy(sorted, h->entries, KEEP * sizeof(Entry));
    qsort(sorted, KEEP, sizeof(Entry), compare_entries);
    uint64_t state = prior;
    for (int i = 0; i < KEEP; i++) {
        uint32_t bits; memcpy(&bits, &sorted[i].score, sizeof bits);
        state ^= (uint32_t)sorted[i].id; state *= UINT64_C(1099511628211);
        state ^= bits; state *= UINT64_C(1099511628211);
    }
    return state;
}

static void one_token(const Pool *p, const float *inputs, int token,
                      int threads, Timing *times) {
    Heap locals[MAX_THREADS];
    for (int layer = 0; layer < LAYERS; layer++) {
        const float *basis = p->basis + (size_t)layer * DIM * RANK;
        const float *x = inputs + ((size_t)token * LAYERS + layer) * DIM;
        float xp[RANK];
        double t0 = now_s(); project(basis, x, xp); double t1 = now_s();
        const int8_t *codes = p->codes + (size_t)layer * p->experts * RANK;
        const float *scales = p->scales + (size_t)layer * p->experts;
        memset(locals, 0, (size_t)threads * sizeof(Heap));
        #pragma omp parallel num_threads(threads)
        {
            int tid = omp_get_thread_num();
            int first = (int)((int64_t)p->experts * tid / threads);
            int last = (int)((int64_t)p->experts * (tid + 1) / threads);
            Heap *h = &locals[tid];
            for (int e = first; e < last; e++) {
                float score = dot_avx2(codes + (size_t)e * RANK, xp) * scales[e];
                heap_push(h, (Entry){score, e});
            }
        }
        double t2 = now_s();
        Heap merged = {0};
        for (int tid = 0; tid < threads; tid++) {
            if (locals[tid].count != KEEP) { fputs("incomplete local heap\n", stderr); exit(2); }
            for (int i = 0; i < KEEP; i++) heap_push(&merged, locals[tid].entries[i]);
        }
        if (merged.count != KEEP) { fputs("incomplete merged heap\n", stderr); exit(2); }
        times->checksum = hash_top(&merged, times->checksum);
        double t3 = now_s();
        times->projection_s += t1 - t0;
        times->scan_s += t2 - t1;
        times->merge_s += t3 - t2;
    }
}

static int double_cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static double median4(const double *x) {
    double y[REPEATS]; memcpy(y, x, sizeof y);
    qsort(y, REPEATS, sizeof(double), double_cmp);
    return 0.5 * (y[1] + y[2]);
}

int main(int argc, char **argv) {
    const char *seed = NULL, *out = NULL;
    int experts = 0, threads = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = argv[++i];
        else if (!strcmp(argv[i], "--experts") && i + 1 < argc) experts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else { fputs("bad argument\n", stderr); return 2; }
    }
    if (!seed || !out || (experts != 27355 && experts != 273547) ||
        (threads != 1 && threads != 6) || threads > MAX_THREADS) {
        fputs("usage: --seed FILE --experts 27355|273547 --threads 1|6 --out FILE\n", stderr);
        return 2;
    }
    omp_set_dynamic(0);
    double job_start = now_s();
    Pool p = load_expand(seed, experts);
    int measured_tokens = experts == 27355 ? 64 : 16;
    float *inputs = make_inputs(measured_tokens);
    float xp[RANK]; project(p.basis, inputs, xp);
    float max_error = 0.0f;
    for (int e = 0; e < SEED_E; e++) {
        const int8_t *row = p.codes + (size_t)e * RANK;
        float ref = dot_scalar(row, xp), got = dot_avx2(row, xp);
        float err = fabsf(ref - got);
        if (err > max_error) max_error = err;
        if (err > 0.001f * (1.0f + fabsf(ref))) {
            fprintf(stderr, "AVX2 dot selftest failed at row %d: %.8g vs %.8g\n", e, ref, got);
            return 2;
        }
    }
    Timing warm = {0}; warm.checksum = UINT64_C(1469598103934665603);
    for (int t = 0; t < WARMUP; t++) one_token(&p, inputs, t, threads, &warm);
    double proj[REPEATS], scan[REPEATS], merge[REPEATS], total[REPEATS];
    uint64_t checksums[REPEATS];
    for (int rep = 0; rep < REPEATS; rep++) {
        Timing a = {0}; a.checksum = UINT64_C(1469598103934665603);
        for (int t = 0; t < measured_tokens; t++)
            one_token(&p, inputs, t, threads, &a);
        proj[rep] = a.projection_s * 1000.0 / measured_tokens;
        scan[rep] = a.scan_s * 1000.0 / measured_tokens;
        merge[rep] = a.merge_s * 1000.0 / measured_tokens;
        total[rep] = proj[rep] + scan[rep] + merge[rep];
        checksums[rep] = a.checksum;
        if (now_s() - job_start > 15 * 60) { fputs("METH-40 wall stop\n", stderr); return 2; }
        printf("rep %d/%d E=%d threads=%d total=%.3f ms/token\n",
               rep + 1, REPEATS, experts, threads, total[rep]); fflush(stdout);
    }
    for (int rep = 1; rep < REPEATS; rep++)
        if (checksums[rep] != checksums[0]) { fputs("checksum differs across repetitions\n", stderr); return 2; }
    FILE *f = fopen(out, "wb"); if (!f) { fputs("cannot open output\n", stderr); return 2; }
    fprintf(f, "{\n\"experiment\":\"METH-40\",\"experts\":%d,\"threads\":%d,",
            experts, threads);
    fprintf(f, "\"layers\":%d,\"dimension\":%d,\"rank\":%d,\"candidates\":%d,",
            LAYERS, DIM, RANK, KEEP);
    fprintf(f, "\"warmup_tokens\":%d,\"measured_tokens_per_rep\":%d,\"repetitions\":%d,",
            WARMUP, measured_tokens, REPEATS);
    fprintf(f, "\"pool_bytes\":%llu,\"code_bytes_per_token\":%llu,",
            (unsigned long long)p.pool_bytes,
            (unsigned long long)((size_t)LAYERS * experts * RANK));
    fprintf(f, "\"scale_bytes_per_token\":%llu,\"basis_bytes_per_token\":%llu,",
            (unsigned long long)((size_t)LAYERS * experts * sizeof(float)),
            (unsigned long long)((size_t)LAYERS * DIM * RANK * sizeof(float)));
    fprintf(f, "\"selftest_max_abs_error\":%.9g,\"checksum\":\"%016llx\",",
            max_error, (unsigned long long)checksums[0]);
    fprintf(f, "\"repetitions_ms_per_token\":[");
    for (int r = 0; r < REPEATS; r++) fprintf(f, "%s%.9g", r ? "," : "", total[r]);
    fprintf(f, "],\"projection_ms_per_token\":[");
    for (int r = 0; r < REPEATS; r++) fprintf(f, "%s%.9g", r ? "," : "", proj[r]);
    fprintf(f, "],\"scan_ms_per_token\":[");
    for (int r = 0; r < REPEATS; r++) fprintf(f, "%s%.9g", r ? "," : "", scan[r]);
    fprintf(f, "],\"merge_ms_per_token\":[");
    for (int r = 0; r < REPEATS; r++) fprintf(f, "%s%.9g", r ? "," : "", merge[r]);
    fprintf(f, "],\"median_total_ms_per_token\":%.9g,",
            median4(total));
    fprintf(f, "\"median_projection_ms_per_token\":%.9g,",
            median4(proj));
    fprintf(f, "\"median_scan_ms_per_token\":%.9g,",
            median4(scan));
    fprintf(f, "\"median_merge_ms_per_token\":%.9g,",
            median4(merge));
    fprintf(f, "\"process_seconds\":%.9g}\n", now_s() - job_start);
    fclose(f);
    printf("median E=%d threads=%d: %.3f ms/token, checksum=%016llx\n",
           experts, threads, median4(total), (unsigned long long)checksums[0]);
    free(inputs); free(p.basis); free(p.codes); free(p.scales);
    return 0;
}
