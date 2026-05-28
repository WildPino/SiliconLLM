/*
 * Phase 38: Online LMS Readout Adaptation
 *
 * Mirrors exactly the SEE expert from see_codec.c:
 *   logit[c] = trigram[ctx2][ctx1][c] + B[c] + W[c] · features
 *
 * Baseline (--no-update): matches see.exe "SEE Only" BPB.
 * Online mode: updates only W (the residual readout) after each byte via SGD.
 *
 * Usage:
 *   phase38_online_lms.exe <data_file> <weights_file> [--lr <float>] [--window <int>] [--no-update]
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "../src/silicon_entropy.h"

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#define CLASSES 256

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t feature_dim;
    uint32_t chunk_size;
    float    decay;
    uint32_t codebook_seed;
    float    alpha;
} WeightsFileHeader;

static float* trigram;   /* [256*256*256] */
static float  W[CLASSES][SEE_FEATURE_DIM];
static float  B[CLASSES];
static float  feature_means[SEE_FEATURE_DIM];
static float  feature_stds[SEE_FEATURE_DIM];

static WeightsFileHeader g_hdr;

static int load_weights(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open weights: %s\n", path); return 0; }

    fread(&g_hdr, sizeof(WeightsFileHeader), 1, f);

    size_t tri_count = (size_t)CLASSES * CLASSES * CLASSES;
    trigram = (float*)malloc(tri_count * sizeof(float));
    if (!trigram) { fprintf(stderr, "OOM\n"); fclose(f); return 0; }
    fread(trigram, sizeof(float), tri_count, f);

    fread(feature_means, sizeof(float), SEE_FEATURE_DIM, f);
    fread(feature_stds,  sizeof(float), SEE_FEATURE_DIM, f);
    fread(W, sizeof(float), CLASSES * SEE_FEATURE_DIM, f);
    fread(B, sizeof(float), CLASSES, f);

    fclose(f);
    return 1;
}

static inline float dot_avx(const float* w, const float* f, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(w+i), _mm256_loadu_ps(f+i), acc);
    float tmp[8]; _mm256_storeu_ps(tmp, acc);
    float s = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    for (; i < n; i++) s += w[i]*f[i];
    return s;
}

/* Compute SEE prediction for target byte. Returns bits (-log2 P[target]). */
static double predict_see(const float* feat, uint8_t ctx1, uint8_t ctx2,
                           uint8_t target, float P_out[CLASSES]) {
    const float* tri_base = &trigram[(ctx2 * CLASSES + ctx1) * CLASSES];

    float max_l = -1e30f;
    float logits[CLASSES];
    for (int c = 0; c < CLASSES; c++) {
        logits[c] = tri_base[c] + B[c] + dot_avx(W[c], feat, SEE_FEATURE_DIM);
        if (logits[c] > max_l) max_l = logits[c];
    }

    float Z = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        P_out[c] = expf(logits[c] - max_l);
        Z += P_out[c];
    }
    for (int c = 0; c < CLASSES; c++) P_out[c] /= Z;

    float p = P_out[target];
    if (p < 1e-30f) p = 1e-30f;
    return -log2((double)p);
}

/* Online SGD: update only W (residual readout), not the trigram */
static void lms_update_residual(const float* feat, uint8_t target,
                                const float P[CLASSES], float lr) {
    for (int c = 0; c < CLASSES; c++) {
        float err = P[c] - (c == target ? 1.0f : 0.0f);
        if (fabsf(err) < 1e-7f) continue;
        float scaled = lr * err;
        float* wrow = W[c];
        for (int f = 0; f < SEE_FEATURE_DIM; f++)
            wrow[f] -= scaled * feat[f];
        B[c] -= scaled;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <data_file> <weights_file> [--lr <f>] [--window <n>] [--no-update]\n", argv[0]);
        return 1;
    }

    const char* data_path    = argv[1];
    const char* weights_path = argv[2];
    float lr      = 0.001f;
    int   window  = 20000;
    int   no_update = 0;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--lr") && i+1 < argc)      lr = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--window") && i+1 < argc) window = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-update"))          no_update = 1;
    }

    FILE* fd = fopen(data_path, "rb");
    if (!fd) { fprintf(stderr, "Cannot open data: %s\n", data_path); return 1; }
    fseek(fd, 0, SEEK_END); long data_size = ftell(fd); fseek(fd, 0, SEEK_SET);
    uint8_t* data = (uint8_t*)malloc(data_size);
    fread(data, 1, data_size, fd); fclose(fd);

    if (!load_weights(weights_path)) return 1;

    SiliconEntropyState see;
    see_init(&see, (int)g_hdr.codebook_seed, (int)g_hdr.chunk_size, g_hdr.decay);

    printf("=== Phase 38: Online LMS Readout Adaptation ===\n");
    printf("Data:    %s (%ld bytes)\n", data_path, data_size);
    printf("Weights: %s\n", weights_path);
    printf("LR: %.5f  Window: %d  Update: %s\n\n",
           lr, window, no_update ? "OFF (baseline)" : "ON");
    printf("%-12s  %-10s  %-10s\n", "byte_offset", "window_bpb", "cumul_bpb");
    printf("%-12s  %-10s  %-10s\n", "-----------", "----------", "---------");

    double cumul_bits  = 0.0;
    double window_bits = 0.0;
    long   window_start = 0;

    float feat[SEE_FEATURE_DIM];
    float P[CLASSES];

    /* Need 2 bytes of context before predicting */
    uint8_t ctx2 = 0, ctx1 = 0;
    see_observe(&see, data[0]); ctx2 = data[0];
    if (data_size > 1) { see_observe(&see, data[1]); ctx1 = data[1]; }

    for (long i = 2; i < data_size; i++) {
        uint8_t byte = data[i];

        see_extract(&see, feat);
        for (int fj = 0; fj < SEE_FEATURE_DIM; fj++)
            feat[fj] = (feat[fj] - feature_means[fj]) / (feature_stds[fj] + 1e-8f);

        double bits = predict_see(feat, ctx1, ctx2, byte, P);
        window_bits += bits;
        cumul_bits  += bits;

        if (!no_update)
            lms_update_residual(feat, byte, P, lr);

        see_observe(&see, byte);
        ctx2 = ctx1;
        ctx1 = byte;

        long pos = i - 1;  /* 0-based count of bytes predicted so far */
        if ((pos + 1) % window == 0 || i + 1 == data_size) {
            long w_len = (pos + 1) - window_start;
            printf("%-12ld  %-10.4f  %-10.4f\n",
                   pos + 1,
                   window_bits / (double)w_len,
                   cumul_bits  / (double)(pos + 1));
            window_bits  = 0.0;
            window_start = pos + 1;
        }
    }

    printf("\nFinal cumulative BPB: %.4f  (%ld bytes evaluated)\n",
           cumul_bits / (double)(data_size - 2), data_size - 2);
    free(data);
    free(trigram);
    return 0;
}
