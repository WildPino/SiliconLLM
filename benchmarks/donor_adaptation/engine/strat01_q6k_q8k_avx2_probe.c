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

static int matrix_mode(int argc, char **argv) {
    unsigned long parsed_count, parsed_rows, parsed_batch;
    unsigned count, rows, batch, blocks;
    size_t row_bytes, input_count, output_count;
    uint8_t *q6 = NULL;
    float *input = NULL, *output = NULL;
    strat01_q8_k_block *q8 = NULL;
    char *end = NULL;
    int result = 2;

    if (argc != 9) {
        fprintf(stderr, "usage: q6_avx2_probe matrix <n> <rows> <batch> <matrix> <input.f32le> <q8.bin> <output.f32le>\n");
        return 2;
    }
#define PARSE_POSITIVE(text, destination) do { \
    errno = 0; end = NULL; destination = strtoul((text), &end, 10); \
    if (errno || !end || *end || !(destination) || (destination) > UINT32_MAX) { \
        fprintf(stderr, "invalid matrix dimension\n"); return 2; \
    } \
} while (0)
    PARSE_POSITIVE(argv[2], parsed_count);
    PARSE_POSITIVE(argv[3], parsed_rows);
    PARSE_POSITIVE(argv[4], parsed_batch);
#undef PARSE_POSITIVE
    count = (unsigned)parsed_count; rows = (unsigned)parsed_rows; batch = (unsigned)parsed_batch;
    if (count % STRAT01_QK_K) {
        fprintf(stderr, "invalid vector length\n");
        return 2;
    }
    blocks = count / STRAT01_QK_K;
    row_bytes = (size_t)blocks * STRAT01_Q6_K_BLOCK_BYTES;
    input_count = (size_t)batch * count;
    output_count = (size_t)batch * rows;
    q6 = (uint8_t *)malloc((size_t)rows * row_bytes);
    input = (float *)malloc(input_count * sizeof(*input));
    q8 = (strat01_q8_k_block *)malloc((size_t)batch * blocks * sizeof(*q8));
    output = (float *)malloc(output_count * sizeof(*output));
    if (!q6 || !input || !q8 || !output ||
        !read_exact(argv[5], q6, (size_t)rows * row_bytes) ||
        !read_exact(argv[6], input, input_count * sizeof(*input))) {
        fprintf(stderr, "matrix input failure\n");
        goto cleanup;
    }
    for (unsigned item = 0; item < batch; ++item) {
        if (!strat01_quantize_q8_k_row(
                input + (size_t)item * count, count, q8 + (size_t)item * blocks)) {
            fprintf(stderr, "matrix quantization failure\n");
            goto cleanup;
        }
    }
    for (unsigned row = 0; row < rows; ++row) {
        for (unsigned item = 0; item < batch; ++item) {
            float dot = strat01_q6k_q8k_dot_avx2(
                q6 + (size_t)row * row_bytes, q8 + (size_t)item * blocks, count);
            if (!isfinite(dot)) {
                fprintf(stderr, "matrix reduction failure\n");
                goto cleanup;
            }
            output[(size_t)item * rows + row] = dot;
        }
    }
    if (!write_exact(argv[7], q8, (size_t)batch * blocks * sizeof(*q8)) ||
        !write_exact(argv[8], output, output_count * sizeof(*output))) {
        fprintf(stderr, "matrix output failure\n");
        goto cleanup;
    }
    result = 0;
cleanup:
    free(output); free(q8); free(input); free(q6);
    return result;
}

int main(int argc, char **argv) {
    unsigned long parsed;
    unsigned count, blocks;
    uint8_t *q6 = NULL;
    strat01_q8_k_block *q8 = NULL;
    float dot;
    int active, result = 2;
    char *end = NULL;

    if (argc > 1 && !strcmp(argv[1], "matrix")) return matrix_mode(argc, argv);
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
