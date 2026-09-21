#include "../../phase60/strat01_q4k_q8k.h"

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
    float input[STRAT01_QK_K], dot;
    uint8_t q4[STRAT01_Q4_K_BLOCK_BYTES];
    strat01_q8_k_block q8;
    if (argc == 5 && !strcmp(argv[1], "--q8-dot")) {
        if (!read_exact(argv[2], q4, sizeof(q4)) ||
            !read_exact(argv[3], &q8, sizeof(q8))) {
            fprintf(stderr, "project probe stored-block input failure\n");
            return 2;
        }
        dot = strat01_q4k_q8k_dot(q4, &q8, STRAT01_QK_K);
        if (!isfinite(dot) || !write_exact(argv[4], &dot, sizeof(dot))) {
            fprintf(stderr, "project probe stored-block output failure\n");
            return 2;
        }
        return 0;
    }
    if (argc != 5) {
        fprintf(stderr, "usage: project_probe <input.f32le> <q4.bin> <q8.bin> <dot.f32le>\n");
        return 2;
    }
    if (!read_exact(argv[1], input, sizeof(input)) ||
        !read_exact(argv[2], q4, sizeof(q4)) ||
        !strat01_quantize_q8_k_block(input, &q8)) {
        fprintf(stderr, "project probe input or quantization failure\n");
        return 2;
    }
    dot = strat01_q4k_q8k_dot(q4, &q8, STRAT01_QK_K);
    if (!isfinite(dot) || !write_exact(argv[3], &q8, sizeof(q8)) ||
        !write_exact(argv[4], &dot, sizeof(dot))) {
        fprintf(stderr, "project probe output failure\n");
        return 2;
    }
    return 0;
}
