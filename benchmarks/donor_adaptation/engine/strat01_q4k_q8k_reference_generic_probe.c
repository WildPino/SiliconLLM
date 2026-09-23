#include "../../phase60/strat01_q4k_q8k.h"

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
    uint8_t *q4 = NULL;
    strat01_q8_k_block *q8 = NULL;
    float dot;
    int mode, result = 2;
    char *end = NULL;

    if (argc != 6) goto usage;
    if (!strcmp(argv[1], "candidate")) mode = 0;
    else if (!strcmp(argv[1], "avx-generic")) mode = 1;
    else if (!strcmp(argv[1], "active-avx2")) mode = 2;
    else goto usage;
    errno = 0;
    parsed = strtoul(argv[2], &end, 10);
    if (errno || !end || *end || !parsed || parsed > UINT32_MAX || parsed % STRAT01_QK_K) {
        fprintf(stderr, "invalid vector length\n");
        return 2;
    }
    count = (unsigned)parsed;
    blocks = count / STRAT01_QK_K;
    q4 = (uint8_t *)malloc((size_t)blocks * STRAT01_Q4_K_BLOCK_BYTES);
    q8 = (strat01_q8_k_block *)malloc((size_t)blocks * sizeof(*q8));
    if (!q4 || !q8 ||
        !read_exact(argv[3], q4, (size_t)blocks * STRAT01_Q4_K_BLOCK_BYTES) ||
        !read_exact(argv[4], q8, (size_t)blocks * sizeof(*q8))) {
        fprintf(stderr, "stored-block input failure\n");
        goto cleanup;
    }
    dot = mode == 0 ? strat01_q4k_q8k_dot_reference_generic(q4, q8, count)
          : mode == 1 ? strat01_q4k_q8k_dot_generic(q4, q8, count)
                      : strat01_q4k_q8k_dot_avx2(q4, q8, count);
    if (!isfinite(dot) || !write_exact(argv[5], &dot, sizeof(dot))) {
        fprintf(stderr, "stored-block output failure\n");
        goto cleanup;
    }
    result = 0;
cleanup:
    free(q8);
    free(q4);
    return result;
usage:
    fprintf(stderr, "usage: reference_generic_probe <candidate|avx-generic|active-avx2> <n> <q4.bin> <q8.bin> <dot.f32le>\n");
    return 2;
}
