/*
 * C50-S1 arm M: memory-bank lookup bench.  Roadmap §5 S1; protocol
 * docs/research/CHATBOT50_ROADMAP_20261016/C50_S1_CAPACITY_INVARIANCE_PROTOCOL_20261018.md.
 *
 * A synthetic table of --gb GiB in private committed memory, rows of 4,096 bytes (2,048 BF16 values).
 * Per token, --k rows at deterministic addresses splitmix64(seed ^ token * K1 ^ j * K2) mod rows; each row's dot
 * with a fixed F32 vector, in the engine's BF16 router kernel order (one 8-lane FMA chain on rows expanded to
 * F32), on 6 threads pinned one per physical core like the engine.  Row j of a token goes to thread j mod 6.
 * --warmup tokens are run first, then --tokens timed one by one.  Prints a CONFIG line and one JSON line.
 * Numbers only; it decides nothing.
 *
 * Usage: c50_s1_membank --gb 16 --k 32 [--tokens 2000] [--warmup 200] [--seed 20261016]
 * Build: clang -O2 -mavx2 -mfma -o bin/c50_s1_membank.exe c50_s1_membank.c
 */
#define _WIN32_WINNT 0x0A00
#include <windows.h>
#include <psapi.h>

#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NT 6
#define COLS 2048
#define ROW_BYTES (COLS * 2)

static volatile LONG g_gen, g_done;
static volatile int g_stop;
static DWORD_PTR g_mask[NT];
static int g_cpu[NT];
static const uint16_t *g_table;
static uint64_t g_rows, g_seed;
static unsigned g_k;
static volatile uint64_t g_token;
static float g_x[COLS];
static float g_part[NT][16];

static uint64_t splitmix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static float hsum8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

static float dot_bf16(const uint16_t *w) {
    __m256 s = _mm256_setzero_ps();
    unsigned c;
    for (c = 0; c < COLS; c += 8) {
        const __m256 wf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(w + c))), 16));
        s = _mm256_fmadd_ps(wf, _mm256_loadu_ps(g_x + c), s);
    }
    return hsum8(s);
}

static void work(int tid) {
    unsigned j;
    float acc = 0.0f;
    for (j = (unsigned)tid; j < g_k; j += NT) {
        uint64_t r = splitmix64(g_seed ^ (g_token * 0xD1B54A32D192ED03ull) ^ ((uint64_t)j * 0xABC98388FB8FAC03ull)) % g_rows;
        acc += dot_bf16(g_table + r * COLS);
    }
    g_part[tid][0] = acc;
}

static DWORD WINAPI worker_main(LPVOID p) {
    int tid = (int)(intptr_t)p;
    LONG seen = 0;
    SetThreadAffinityMask(GetCurrentThread(), g_mask[tid]);
    for (;;) {
        while (g_gen == seen) _mm_pause();
        seen = g_gen;
        if (g_stop) return 0;
        work(tid);
        InterlockedIncrement(&g_done);
    }
}

static float run_token(void) {
    int t;
    float s = 0.0f;
    g_done = 0;
    MemoryBarrier();
    InterlockedIncrement(&g_gen);
    work(0);
    while (g_done < NT - 1) _mm_pause();
    for (t = 0; t < NT; ++t) s += g_part[t][0];
    return s;
}

static void pick_cores(void) {
    DWORD len = 0;
    char *buf, *ptr;
    int n = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    buf = malloc(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf, &len)) {
        fprintf(stderr, "topology query failed\n"); exit(2);
    }
    for (ptr = buf; ptr < buf + len && n < NT;) {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)ptr;
        KAFFINITY m = info->Processor.GroupMask[0].Mask;
        int bit = 0;
        while (!((m >> bit) & 1)) ++bit;
        g_mask[n] = (DWORD_PTR)1 << bit;
        g_cpu[n] = bit;
        ++n;
        ptr += info->Size;
    }
    free(buf);
    if (n < NT) { fprintf(stderr, "fewer than %d physical cores\n", NT); exit(2); }
}

static double now_s(void) {
    static LARGE_INTEGER f;
    LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    double gb = 0, *t, t_fill, med, sum = 0;
    unsigned tokens = 2000, warmup = 200, i, c;
    uint64_t r, nbytes;
    uint16_t *table;
    float check = 0.0f;
    PROCESS_MEMORY_COUNTERS pmc;
    g_seed = 20261016;
    for (i = 1; i < (unsigned)argc; ++i) {
        if (!strcmp(argv[i], "--gb") && i + 1 < (unsigned)argc) gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--k") && i + 1 < (unsigned)argc) g_k = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens") && i + 1 < (unsigned)argc) tokens = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < (unsigned)argc) warmup = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < (unsigned)argc) g_seed = _strtoui64(argv[++i], NULL, 10);
        else { fprintf(stderr, "usage: c50_s1_membank --gb G --k K [--tokens N] [--warmup W] [--seed S]\n"); return 2; }
    }
    if (gb <= 0 || g_k == 0 || tokens == 0) { fprintf(stderr, "--gb and --k are required\n"); return 2; }
    g_rows = (uint64_t)(gb * 1073741824.0) / ROW_BYTES;
    nbytes = g_rows * ROW_BYTES;
    pick_cores();
    SetThreadAffinityMask(GetCurrentThread(), g_mask[0]);
    printf("CONFIG engine=c50_s1_membank gb=%.3f rows=%llu row_bytes=%d k=%u tokens=%u warmup=%u seed=%llu threads=%d "
           "cores=%d,%d,%d,%d,%d,%d mem=private\n", gb, (unsigned long long)g_rows, ROW_BYTES, g_k, tokens, warmup,
           (unsigned long long)g_seed, NT, g_cpu[0], g_cpu[1], g_cpu[2], g_cpu[3], g_cpu[4], g_cpu[5]);
    fflush(stdout);
    table = VirtualAlloc(NULL, (SIZE_T)nbytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!table) { fprintf(stderr, "table allocation of %llu bytes failed\n", (unsigned long long)nbytes); return 4; }
    t_fill = now_s();
    for (r = 0; r < g_rows; ++r) {   /* synthetic BF16 values in [1, 2): exponent 127, a row-dependent mantissa */
        uint16_t *row = table + r * COLS;
        uint16_t base = (uint16_t)(0x3F80 | (splitmix64(r) & 0x7F));
        for (c = 0; c < COLS; ++c) row[c] = (uint16_t)(base ^ (c & 0x7F));
    }
    t_fill = now_s() - t_fill;
    for (c = 0; c < COLS; ++c) g_x[c] = 1.0f / (float)(1 + (c & 15));
    g_table = table;
    for (i = 1; i < NT; ++i) CreateThread(NULL, 0, worker_main, (LPVOID)(intptr_t)i, 0, NULL);
    t = malloc(sizeof(double) * tokens);
    for (i = 0; i < warmup; ++i) { g_token = i; check += run_token(); }
    for (i = 0; i < tokens; ++i) {
        double t0 = now_s();
        g_token = warmup + i;
        check += run_token();
        t[i] = now_s() - t0;
        sum += t[i];
    }
    g_stop = 1;
    InterlockedIncrement(&g_gen);
    qsort(t, tokens, sizeof *t, cmpd);
    med = tokens % 2 ? t[tokens / 2] : 0.5 * (t[tokens / 2 - 1] + t[tokens / 2]);
    memset(&pmc, 0, sizeof pmc);
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc);
    printf("{\"mode\":\"membank\",\"gb\":%.3f,\"k\":%u,\"tokens\":%u,\"fill_s\":%.3f,\"median_token_ns\":%.1f,"
           "\"mean_token_ns\":%.1f,\"min_token_ns\":%.1f,\"max_token_ns\":%.1f,\"median_row_ns\":%.2f,"
           "\"effective_gb_s\":%.3f,\"peak_working_set_bytes\":%llu,\"checksum\":%.6e}\n",
           gb, g_k, tokens, t_fill, med * 1e9, sum / tokens * 1e9, t[0] * 1e9, t[tokens - 1] * 1e9, med * 1e9 / g_k,
           (double)g_k * ROW_BYTES / med / 1e9, (unsigned long long)pmc.PeakWorkingSetSize, (double)check);
    return 0;
}
