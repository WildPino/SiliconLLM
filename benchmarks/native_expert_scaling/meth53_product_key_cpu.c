// Synthetic exact product-key top-4 router. Measures CPU cost, not donor quality.
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
#define DIM 896
#define KEEP 4
#define REPS 4
#define WARMUP 8

typedef struct { int id; float score; } Choice;
typedef struct { int a, b; float *keys, *scores; size_t bytes; } Pool;
typedef struct { double dot, select, total; uint64_t checksum; } Timed;

static double now_s(void) {
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
}
static size_t rss_bytes(void) {
    PROCESS_MEMORY_COUNTERS m;
    return GetProcessMemoryInfo(GetCurrentProcess(), &m, sizeof m) ? (size_t)m.WorkingSetSize : 0;
}
static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "allocation failed: %zu\n", n); exit(2); }
    return p;
}
static uint64_t mix(uint64_t z) {
    z += UINT64_C(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
static float unit(uint64_t n) {
    return (float)((int)((mix(n) >> 40) & 0xffff) - 32768) * (1.0f / 65536.0f);
}
static void input(float *x, int token, int layer) {
    uint64_t seed = (uint64_t)(token + 1) * 100003U + (uint64_t)(layer + 1) * 900001U;
    for (int i = 0; i < DIM; i++) x[i] = unit(seed + (uint64_t)i);
}
static float dot_scalar(const float *w, const float *x) {
    float s = 0;
    for (int i = 0; i < DIM; i++) s += w[i] * x[i];
    return s;
}
static float dot_avx2(const float *w, const float *x) {
    __m256 s = _mm256_setzero_ps();
    for (int i = 0; i < DIM; i += 8)
        s = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(x + i), s);
    float lanes[8]; _mm256_storeu_ps(lanes, s);
    float out = 0;
    for (int i = 0; i < 8; i++) out += lanes[i];
    return out;
}
static Pool make_pool(int a, int b) {
    Pool p = {0}; p.a = a; p.b = b;
    size_t rows = (size_t)LAYERS * (size_t)(a + b);
    p.bytes = rows * DIM * sizeof(float);
    p.keys = xmalloc(p.bytes);
    p.scores = xmalloc(rows * sizeof(float));
    for (size_t row = 0; row < rows; row++) {
        float *w = p.keys + row * DIM;
        uint64_t seed = (uint64_t)(row + 1) * 1000003U;
        for (int k = 0; k < DIM; k++) w[k] = unit(seed + (uint64_t)k) * 0.1f;
    }
    return p;
}
static int better(Choice a, Choice b) {
    return a.score > b.score || (a.score == b.score && a.id < b.id);
}
static void insert(Choice best[KEEP], Choice c) {
    for (int k = 0; k < KEEP; k++) {
        if (better(c, best[k])) {
            for (int j = KEEP - 1; j > k; j--) best[j] = best[j - 1];
            best[k] = c;
            return;
        }
    }
}
static void empty(Choice best[KEEP]) {
    for (int k = 0; k < KEEP; k++) best[k] = (Choice){INT32_MAX, -INFINITY};
}
static void axis_top4(const float *scores, int n, Choice out[KEEP]) {
    empty(out);
    for (int i = 0; i < n; i++) insert(out, (Choice){i, scores[i]});
}
static void pair_top4(const float *scores, int a, int b, Choice out[KEEP]) {
    Choice as[KEEP], bs[KEEP];
    axis_top4(scores, a, as);
    axis_top4(scores + a, b, bs);
    empty(out);
    for (int i = 0; i < KEEP; i++) for (int j = 0; j < KEEP; j++)
        insert(out, (Choice){as[i].id * b + bs[j].id, as[i].score + bs[j].score});
}
static void oracle(const float *scores, int a, int b, Choice out[KEEP]) {
    empty(out);
    for (int i = 0; i < a; i++) for (int j = 0; j < b; j++)
        insert(out, (Choice){i * b + j, scores[i] + scores[a + j]});
}
static uint64_t route_hash(uint64_t h, const Choice route[KEEP]) {
    for (int k = 0; k < KEEP; k++) h = mix(h ^ ((uint64_t)route[k].id + (uint64_t)(k + 1) * 10000019U));
    return h;
}
static void score_layer(const Pool *p, int layer, const float *x) {
    int rows = p->a + p->b;
    size_t base = (size_t)layer * rows;
    #pragma omp parallel for schedule(static) if(rows >= 128)
    for (int r = 0; r < rows; r++)
        p->scores[base + r] = dot_avx2(p->keys + (base + r) * DIM, x);
}
static void verify(Pool *p, double *max_dot_err, int *oracle_cases) {
    float x[DIM];
    int inputs = p->a == 8 ? 16 : 1;
    *max_dot_err = 0; *oracle_cases = 0;
    for (int t = 0; t < inputs; t++) for (int layer = 0; layer < LAYERS; layer++) {
        input(x, t, layer);
        score_layer(p, layer, x);
        int rows = p->a + p->b;
        size_t base = (size_t)layer * rows;
        for (int r = 0; r < rows; r += rows > 32 ? rows / 31 : 1) {
            float s = dot_scalar(p->keys + (base + r) * DIM, x);
            double d = fabs((double)s - p->scores[base + r]);
            if (d > *max_dot_err) *max_dot_err = d;
        }
        Choice fast[KEEP], exact[KEEP];
        pair_top4(p->scores + base, p->a, p->b, fast);
        oracle(p->scores + base, p->a, p->b, exact);
        for (int k = 0; k < KEEP; k++) if (fast[k].id != exact[k].id || fast[k].score != exact[k].score) {
            fprintf(stderr, "oracle mismatch token=%d layer=%d k=%d fast=%d exact=%d\n",
                    t, layer, k, fast[k].id, exact[k].id); exit(3);
        }
        (*oracle_cases)++;
    }
    if (*max_dot_err > 1e-4) { fprintf(stderr, "dot error %.9g\n", *max_dot_err); exit(3); }
}
static Timed run(Pool *p, int tokens, int offset) {
    Timed out = {0}; out.checksum = UINT64_C(0x123456789abcdef0);
    float x[DIM];
    double start = now_s();
    for (int t = 0; t < tokens; t++) for (int layer = 0; layer < LAYERS; layer++) {
        input(x, t + offset, layer);
        double d0 = now_s();
        score_layer(p, layer, x);
        double d1 = now_s();
        Choice top[KEEP];
        pair_top4(p->scores + (size_t)layer * (p->a + p->b), p->a, p->b, top);
        out.checksum = route_hash(out.checksum, top);
        double d2 = now_s();
        out.dot += d1 - d0;
        out.select += d2 - d1;
    }
    out.total = now_s() - start;
    return out;
}
static double median4(const double v[REPS]) {
    double x[REPS]; memcpy(x, v, sizeof x);
    for (int i = 1; i < REPS; i++) {
        double c = x[i]; int j = i - 1;
        while (j >= 0 && x[j] > c) { x[j + 1] = x[j]; j--; }
        x[j + 1] = c;
    }
    return (x[1] + x[2]) * 0.5;
}
int main(int argc, char **argv) {
    if (argc != 5) { fputs("usage: meth53_product_key_cpu A B THREADS TOKENS\n", stderr); return 2; }
    int a = atoi(argv[1]), b = atoi(argv[2]), threads = atoi(argv[3]), tokens = atoi(argv[4]);
    if (a < KEEP || b < KEEP || a > 1024 || b > 1024 || (threads != 1 && threads != 6) || tokens < 1 || tokens > 256) return 2;
    omp_set_dynamic(0); omp_set_num_threads(threads);
    Pool p = make_pool(a, b);
    double dot_error; int oracle_cases;
    verify(&p, &dot_error, &oracle_cases);
    run(&p, WARMUP, 0);
    double dot[REPS], select[REPS], total[REPS];
    uint64_t checksum = 0;
    for (int r = 0; r < REPS; r++) {
        Timed v = run(&p, tokens, WARMUP);
        dot[r] = v.dot * 1000.0 / tokens;
        select[r] = v.select * 1000.0 / tokens;
        total[r] = v.total * 1000.0 / tokens;
        if (r && checksum != v.checksum) { fputs("repeat checksum mismatch\n", stderr); return 3; }
        checksum = v.checksum;
    }
    printf("{\"method\":\"METH53\",\"a\":%d,\"b\":%d,\"experts\":%d,\"threads\":%d,\"tokens\":%d,\"layers\":%d,\"dim\":%d,\"key_pool_bytes\":%zu,\"addressed_bytes_per_token\":%zu,\"rss_bytes\":%zu,\"dot_max_abs_error\":%.9g,\"oracle_cases\":%d,\"checksum\":\"%016llx\",\"dot_ms_median\":%.9f,\"pair_ms_median\":%.9f,\"total_ms_median\":%.9f,\"total_ms_reps\":[%.9f,%.9f,%.9f,%.9f]}\n",
           a,b,a*b,threads,tokens,LAYERS,DIM,p.bytes,p.bytes,rss_bytes(),dot_error,oracle_cases,
           (unsigned long long)checksum,median4(dot),median4(select),median4(total),total[0],total[1],total[2],total[3]);
    free(p.scores); free(p.keys);
    return 0;
}
