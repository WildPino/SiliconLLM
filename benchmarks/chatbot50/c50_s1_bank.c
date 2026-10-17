/*
 * C50-S1 arm M: the conditional-memory lookup microbenchmark.  A table of synthetic rows (deterministic bytes) is
 * read the way a memory bank would be read in decode: per token, per layer, k rows at deterministic addresses
 * (splitmix64 of token, layer and index, modulo the row count), split over the engine's 6 pinned threads with a
 * spin barrier per layer, as the engine runs one job per organ.  Each row is reduced (byte sums) so that every
 * byte is loaded.  Per token the wall time of all layers is taken; one JSON line reports the median token time,
 * the effective GB/s and ns per row.  Rates need a clean box (ROADMAP §5 S1, arm M).
 *
 *   c50_s1_bank --table-gb G --row-bytes B --k K [--layers 19] [--tokens 256] [--warm 16] [--mem pageable|locked]
 *
 * Build:
 *   clang -O2 -mavx2 -mfma -o bin/c50_s1_bank.exe c50_s1_bank.c
 */
#define _WIN32_WINNT 0x0A00
#include <windows.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NT 6
#define MAXK 4096

static DWORD_PTR g_mask[NT];
static int g_cpu[NT];
static volatile LONG g_gen, g_done;
static volatile int g_quit;

typedef struct {
    const uint8_t *table;
    uint64_t nrows, row_bytes;
    unsigned k, token, layer;
    uint64_t sum[NT][4];   /* per-thread reduction, kept so the loads are not dead */
} job_t;
static job_t g_job;

static uint64_t mix(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void work(int tid) {
    job_t *j = &g_job;
    unsigned i;
    __m256i acc = _mm256_setzero_si256();
    for (i = tid; i < j->k; i += NT) {
        const uint64_t r = mix(((uint64_t)j->token << 32) ^ ((uint64_t)j->layer << 16) ^ i) % j->nrows;
        const uint8_t *p = j->table + r * j->row_bytes;
        uint64_t b;
        for (b = 0; b < j->row_bytes; b += 32)
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(_mm256_loadu_si256((const __m256i *)(p + b)), _mm256_setzero_si256()));
    }
    _mm256_storeu_si256((__m256i *)j->sum[tid], _mm256_add_epi64(_mm256_loadu_si256((const __m256i *)j->sum[tid]), acc));
}

static DWORD WINAPI worker(LPVOID p) {
    int tid = (int)(intptr_t)p;
    LONG seen = 0;
    SetThreadAffinityMask(GetCurrentThread(), g_mask[tid]);
    for (;;) {
        while (g_gen == seen) _mm_pause();
        seen = g_gen;
        if (g_quit) return 0;
        work(tid);
        InterlockedIncrement(&g_done);
    }
}

static void run_layer(void) {
    g_done = 0;
    MemoryBarrier();
    InterlockedIncrement(&g_gen);
    work(0);
    while (g_done < NT - 1) _mm_pause();
}

static void pick_cores(void) {   /* the engine's rule: the first logical processor of each physical core */
    DWORD len = 0;
    char *buf, *ptr;
    int n = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    buf = malloc(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf, &len)) exit(2);
    for (ptr = buf; ptr < buf + len && n < NT;) {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)ptr;
        KAFFINITY m = info->Processor.GroupMask[0].Mask;
        int bit = 0;
        while (!((m >> bit) & 1)) ++bit;
        g_mask[n] = (DWORD_PTR)1 << bit;
        g_cpu[n++] = bit;
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
    return x < y ? -1 : x > y;
}

int main(int argc, char **argv) {
    double table_gb = 0, *t, fill_s, med;
    uint64_t row_bytes = 0, nrows, size, i, total = 0;
    unsigned k = 0, layers = 19, tokens = 256, warm = 16, tok, l;
    int locked = 0, a, tid;
    uint8_t *table;
    for (a = 1; a < argc; ++a) {
        if (!strcmp(argv[a], "--table-gb") && a + 1 < argc) table_gb = atof(argv[++a]);
        else if (!strcmp(argv[a], "--row-bytes") && a + 1 < argc) row_bytes = strtoull(argv[++a], NULL, 10);
        else if (!strcmp(argv[a], "--k") && a + 1 < argc) k = (unsigned)atoi(argv[++a]);
        else if (!strcmp(argv[a], "--layers") && a + 1 < argc) layers = (unsigned)atoi(argv[++a]);
        else if (!strcmp(argv[a], "--tokens") && a + 1 < argc) tokens = (unsigned)atoi(argv[++a]);
        else if (!strcmp(argv[a], "--warm") && a + 1 < argc) warm = (unsigned)atoi(argv[++a]);
        else if (!strcmp(argv[a], "--mem") && a + 1 < argc) {
            ++a;
            locked = !strcmp(argv[a], "pageable") ? 0 : !strcmp(argv[a], "locked") ? 1 : -1;
            if (locked < 0) { fprintf(stderr, "--mem must be pageable or locked\n"); return 2; }
        } else { fprintf(stderr, "unknown argument %s\n", argv[a]); return 2; }
    }
    if (table_gb <= 0 || !row_bytes || row_bytes % 32 || !k || k > MAXK || !tokens) {
        fprintf(stderr, "usage: c50_s1_bank --table-gb G --row-bytes B (multiple of 32) --k K (<= %d) [--layers 19] "
                        "[--tokens 256] [--warm 16] [--mem pageable|locked]\n", MAXK);
        return 2;
    }
    nrows = (uint64_t)(table_gb * 1e9) / row_bytes;
    size = nrows * row_bytes;
    table = VirtualAlloc(NULL, (SIZE_T)size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!table) { fprintf(stderr, "table allocation failed (%lu)\n", (unsigned long)GetLastError()); return 4; }
    if (locked && (!SetProcessWorkingSetSizeEx(GetCurrentProcess(), (SIZE_T)size + ((SIZE_T)1 << 30),
                                               (SIZE_T)size + ((SIZE_T)2 << 30), QUOTA_LIMITS_HARDWS_MIN_ENABLE) ||
                   !VirtualLock(table, (SIZE_T)size))) {
        fprintf(stderr, "working-set lock failed (%lu)\n", (unsigned long)GetLastError());
        return 4;
    }
    pick_cores();
    printf("CONFIG engine=c50_s1_bank table_bytes=%llu rows=%llu row_bytes=%llu k=%u layers=%u tokens=%u warm=%u "
           "threads=%d cores=%d,%d,%d,%d,%d,%d mem=%s\n", (unsigned long long)size, (unsigned long long)nrows,
           (unsigned long long)row_bytes, k, layers, tokens, warm, NT, g_cpu[0], g_cpu[1], g_cpu[2], g_cpu[3], g_cpu[4],
           g_cpu[5], locked ? "locked" : "pageable");
    fflush(stdout);
    fill_s = now_s();
    for (i = 0; i < size; i += 8) { uint64_t v = mix(i); memcpy(table + i, &v, 8); }   /* deterministic synthetic values */
    fill_s = now_s() - fill_s;
    SetThreadAffinityMask(GetCurrentThread(), g_mask[0]);
    for (tid = 1; tid < NT; ++tid) CreateThread(NULL, 0, worker, (LPVOID)(intptr_t)tid, 0, NULL);
    g_job.table = table; g_job.nrows = nrows; g_job.row_bytes = row_bytes; g_job.k = k;
    t = malloc(sizeof(double) * tokens);
    for (tok = 0; tok < warm + tokens; ++tok) {
        double t0 = now_s();
        g_job.token = tok;
        for (l = 0; l < layers; ++l) { g_job.layer = l; run_layer(); }
        if (tok >= warm) t[tok - warm] = now_s() - t0;
    }
    for (tid = 0; tid < NT; ++tid) total += g_job.sum[tid][0] + g_job.sum[tid][1] + g_job.sum[tid][2] + g_job.sum[tid][3];
    qsort(t, tokens, sizeof *t, cmpd);
    med = tokens % 2 ? t[tokens / 2] : 0.5 * (t[tokens / 2 - 1] + t[tokens / 2]);
    printf("{\"mode\":\"bank\",\"table_bytes\":%llu,\"row_bytes\":%llu,\"k\":%u,\"layers\":%u,\"tokens\":%u,\"fill_s\":%.3f,"
           "\"median_token_s\":%.7f,\"min_token_s\":%.7f,\"max_token_s\":%.7f,\"bytes_per_token\":%llu,\"gb_s\":%.3f,"
           "\"ns_per_row\":%.2f,\"checksum\":%llu}\n", (unsigned long long)size, (unsigned long long)row_bytes, k, layers,
           tokens, fill_s, med, t[0], t[tokens - 1], (unsigned long long)((uint64_t)layers * k * row_bytes),
           (double)layers * k * row_bytes / med / 1e9, med / ((double)layers * k) * 1e9, (unsigned long long)total);
    g_quit = 1;
    InterlockedIncrement(&g_gen);
    return 0;
}
