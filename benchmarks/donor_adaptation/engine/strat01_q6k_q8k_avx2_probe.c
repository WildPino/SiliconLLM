#include "../../phase60/strat01_q6k_q8k_avx2.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_exact(const char *path, void *data, size_t bytes) {
    FILE *stream = fopen(path, "rb");
    int ok;
    if (!stream) return 0;
    ok = fread(data, 1, bytes, stream) == bytes && fgetc(stream) == EOF && !ferror(stream);
    if (fclose(stream) != 0) ok = 0;
    return ok;
}

static int write_exact(const char *path, const void *data, size_t bytes) {
    FILE *stream = fopen(path, "wb");
    int ok;
    if (!stream) return 0;
    ok = fwrite(data, 1, bytes, stream) == bytes && fclose(stream) == 0;
    return ok;
}

int main(int argc, char **argv) {
    unsigned long parsed;
    unsigned count, blocks;
    uint8_t *q6 = NULL;
    strat01_q8_k_block *q8 = NULL;
    float dot;
    int active, result = 2;
    char *end = NULL;

    if (argc != 6 || (strcmp(argv[1], "active") && strcmp(argv[1], "generic"))) {
        fprintf(stderr, "usage: q6_avx2_probe <active|generic> <n> <q6.bin> <q8.bin> <dot.f32le>\n");
        return 2;
    }
    active = !strcmp(argv[1], "active");
    errno = 0;
    parsed = strtoul(argv[2], &end, 10);
    if (errno || !end || *end || !parsed || parsed > UINT32_MAX || parsed % STRAT01_QK_K) {
        fprintf(stderr, "invalid vector length\n");
        return 2;
    }
    count = (unsigned)parsed;
    blocks = count / STRAT01_QK_K;
    q6 = (uint8_t *)malloc((size_t)blocks * STRAT01_Q6_K_BLOCK_BYTES);
    q8 = (strat01_q8_k_block *)malloc((size_t)blocks * sizeof(*q8));
    if (!q6 || !q8 ||
        !read_exact(argv[3], q6, (size_t)blocks * STRAT01_Q6_K_BLOCK_BYTES) ||
        !read_exact(argv[4], q8, (size_t)blocks * sizeof(*q8))) {
        fprintf(stderr, "stored-block input failure\n");
        goto cleanup;
    }
    dot = active ? strat01_q6k_q8k_dot_avx2(q6, q8, count)
                 : strat01_q6k_q8k_dot_generic(q6, q8, count);
    if (!isfinite(dot) || !write_exact(argv[5], &dot, sizeof(dot))) {
        fprintf(stderr, "stored-block output failure\n");
        goto cleanup;
    }
    result = 0;
cleanup:
    free(q8);
    free(q6);
    return result;
}
