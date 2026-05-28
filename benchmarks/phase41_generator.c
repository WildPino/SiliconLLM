/*
 * Phase 41: First Byte-Level Generator
 *
 * Generates text from the SEE model by autoregressively sampling from
 *   P(next | ctx) = softmax(trigram[ctx2][ctx1] + B[c] + W[c] · features)
 *
 * Key metrics:
 *   - self_bpb: the model's cross-entropy loss on its OWN generated output
 *     (high = incoherent; low = internally consistent structure)
 *   - byte distribution vs. reference (KL divergence)
 *
 * Usage:
 *   phase41_generator.exe <seed_file> <weights_file> [options]
 *
 * Options:
 *   --seed-len <int>     bytes of seed to feed before generating (default 512)
 *   --gen-len  <int>     bytes to generate (default 2000)
 *   --mode argmax|sample sampling mode (default: sample)
 *   --temp <float>       softmax temperature for sample mode (default 1.0)
 *   --online-lms         apply LMS update during generation (risky: hallucination loop)
 *   --lr <float>         LMS learning rate (default 0.003)
 *   --ref-file <path>    reference file for byte distribution KL (optional)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../src/silicon_entropy.h"

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#define CLASSES 256

typedef struct {
    uint32_t magic, version, feature_dim, chunk_size;
    float decay;
    uint32_t codebook_seed;
    float alpha;
} WeightsFileHeader;

static float* trigram;
static float  W[CLASSES][SEE_FEATURE_DIM];
static float  B[CLASSES];
static float  feature_means[SEE_FEATURE_DIM];
static float  feature_stds[SEE_FEATURE_DIM];
static WeightsFileHeader g_hdr;

static int load_weights(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open weights: %s\n", path); return 0; }
    fread(&g_hdr, sizeof(WeightsFileHeader), 1, f);
    size_t tri_n = (size_t)CLASSES * CLASSES * CLASSES;
    trigram = (float*)malloc(tri_n * sizeof(float));
    fread(trigram, sizeof(float), tri_n, f);
    fread(feature_means, sizeof(float), SEE_FEATURE_DIM, f);
    fread(feature_stds,  sizeof(float), SEE_FEATURE_DIM, f);
    fread(W, sizeof(float), CLASSES * SEE_FEATURE_DIM, f);
    fread(B, sizeof(float), CLASSES, f);
    fclose(f);
    return 1;
}

static inline float dot_avx(const float* a, const float* b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n-8; i+=8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
    float tmp[8]; _mm256_storeu_ps(tmp, acc);
    float s = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    for (; i < n; i++) s += a[i]*b[i];
    return s;
}

static void predict(const float* feat, uint8_t ctx1, uint8_t ctx2,
                    float P[CLASSES], float temp) {
    const float* tri = &trigram[(ctx2*CLASSES+ctx1)*CLASSES];
    float max_l = -1e30f, logits[CLASSES];
    for (int c = 0; c < CLASSES; c++) {
        logits[c] = (tri[c] + B[c] + dot_avx(W[c], feat, SEE_FEATURE_DIM)) / temp;
        if (logits[c] > max_l) max_l = logits[c];
    }
    float Z = 0;
    for (int c = 0; c < CLASSES; c++) { P[c] = expf(logits[c]-max_l); Z += P[c]; }
    for (int c = 0; c < CLASSES; c++) P[c] /= Z;
}

static void lms_update(const float* feat, uint8_t target, const float P[CLASSES], float lr) {
    for (int c = 0; c < CLASSES; c++) {
        float err = P[c] - (c == target ? 1.0f : 0.0f);
        if (fabsf(err) < 1e-7f) continue;
        float s = lr * err;
        for (int f = 0; f < SEE_FEATURE_DIM; f++) W[c][f] -= s * feat[f];
        B[c] -= s;
    }
}

static uint8_t sample_from(const float P[CLASSES], uint64_t* rng) {
    /* xorshift64 */
    *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
    double u = (*rng >> 11) * (1.0 / (1ULL << 53));
    double cum = 0;
    for (int c = 0; c < CLASSES; c++) {
        cum += P[c];
        if (u < cum) return (uint8_t)c;
    }
    return 255;
}

static uint8_t argmax_from(const float P[CLASSES]) {
    int best = 0;
    for (int c = 1; c < CLASSES; c++) if (P[c] > P[best]) best = c;
    return (uint8_t)best;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <seed_file> <weights_file> [options]\n", argv[0]);
        printf("  --seed-len <int>    seed bytes (default 512)\n");
        printf("  --gen-len  <int>    bytes to generate (default 2000)\n");
        printf("  --mode argmax|sample  (default: sample)\n");
        printf("  --temp <float>      temperature (default 1.0)\n");
        printf("  --online-lms        apply LMS during generation\n");
        printf("  --lr <float>        LMS lr (default 0.003)\n");
        printf("  --ref-file <path>   reference for byte distribution KL\n");
        return 1;
    }

    const char* seed_path    = argv[1];
    const char* weights_path = argv[2];
    int    seed_len   = 512;
    int    gen_len    = 2000;
    int    argmax     = 0;
    float  temp       = 1.0f;
    int    online_lms = 0;
    float  lr         = 0.003f;
    const char* ref_path = NULL;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i],"--seed-len")  && i+1<argc) seed_len   = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--gen-len")   && i+1<argc) gen_len    = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--mode")       && i+1<argc) argmax = !strcmp(argv[++i],"argmax");
        else if (!strcmp(argv[i],"--temp")       && i+1<argc) temp       = (float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--online-lms"))             online_lms = 1;
        else if (!strcmp(argv[i],"--lr")         && i+1<argc) lr         = (float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--ref-file")   && i+1<argc) ref_path   = argv[++i];
    }

    FILE* fd = fopen(seed_path, "rb");
    if (!fd) { fprintf(stderr, "Cannot open seed: %s\n", seed_path); return 1; }
    fseek(fd, 0, SEEK_END); long fsz = ftell(fd); fseek(fd, 0, SEEK_SET);
    int actual_seed = (seed_len < (int)fsz) ? seed_len : (int)fsz;
    uint8_t* seed_data = (uint8_t*)malloc(actual_seed);
    fread(seed_data, 1, actual_seed, fd); fclose(fd);

    if (!load_weights(weights_path)) return 1;

    /* Reference distribution for KL */
    double ref_freq[CLASSES] = {0};
    if (ref_path) {
        FILE* rf = fopen(ref_path, "rb");
        if (rf) {
            int ch; long rtotal = 0;
            while ((ch = fgetc(rf)) != EOF) { ref_freq[ch]++; rtotal++; }
            fclose(rf);
            for (int c = 0; c < CLASSES; c++) ref_freq[c] = (ref_freq[c]+1)/(rtotal+CLASSES);
        }
    }

    SiliconEntropyState see;
    see_init(&see, (int)g_hdr.codebook_seed, (int)g_hdr.chunk_size, g_hdr.decay);

    uint64_t rng = (uint64_t)time(NULL) ^ 0xdeadbeefcafe;

    /* Feed seed */
    uint8_t ctx2 = 0, ctx1 = 0;
    for (int i = 0; i < actual_seed; i++) {
        see_observe(&see, seed_data[i]);
        if (i > 0) ctx2 = ctx1;
        ctx1 = seed_data[i];
    }

    fprintf(stderr, "=== Phase 41: Generator ===\n");
    fprintf(stderr, "Seed: %d bytes from %s\n", actual_seed, seed_path);
    fprintf(stderr, "Weights: %s\n", weights_path);
    fprintf(stderr, "Mode: %s | Temp: %.2f | Online-LMS: %s\n\n",
            argmax ? "argmax" : "sample", temp, online_lms ? "ON" : "OFF");

    /* Stats */
    double self_bits = 0;
    long   gen_count[CLASSES] = {0};
    uint8_t* generated = (uint8_t*)malloc(gen_len);

    float feat[SEE_FEATURE_DIM], P[CLASSES];

    for (int i = 0; i < gen_len; i++) {
        see_extract(&see, feat);
        for (int fj = 0; fj < SEE_FEATURE_DIM; fj++)
            feat[fj] = (feat[fj] - feature_means[fj]) / (feature_stds[fj] + 1e-8f);

        predict(feat, ctx1, ctx2, P, temp);

        uint8_t next = argmax ? argmax_from(P) : sample_from(P, &rng);

        /* Self-BPB: model's confidence in its own choice */
        float p_self = P[next];
        if (p_self < 1e-30f) p_self = 1e-30f;
        self_bits += -log2((double)p_self);

        if (online_lms) lms_update(feat, next, P, lr);

        see_observe(&see, next);
        ctx2 = ctx1;
        ctx1 = next;
        generated[i] = next;
        gen_count[next]++;
    }

    /* Print generated bytes as text to stdout */
    fwrite(generated, 1, gen_len, stdout);

    double self_bpb = self_bits / gen_len;

    /* Byte distribution stats */
    double gen_freq[CLASSES];
    for (int c = 0; c < CLASSES; c++) gen_freq[c] = (gen_count[c]+1.0)/(gen_len+CLASSES);

    double kl = 0;
    if (ref_path) {
        for (int c = 0; c < CLASSES; c++)
            if (ref_freq[c] > 0)
                kl += gen_freq[c] * log2(gen_freq[c] / ref_freq[c]);
    }

    fprintf(stderr, "\n--- Generation Stats ---\n");
    fprintf(stderr, "Generated:    %d bytes\n", gen_len);
    fprintf(stderr, "Self-BPB:     %.4f  (lower = more internally coherent)\n", self_bpb);
    if (ref_path) fprintf(stderr, "KL(gen||ref): %.4f bits  (lower = distribution closer to training)\n", kl);

    /* Top-10 most frequent generated bytes */
    fprintf(stderr, "Top generated bytes: ");
    int order[CLASSES]; for (int c=0;c<CLASSES;c++) order[c]=c;
    for (int i=0;i<10;i++) for (int j=i+1;j<CLASSES;j++) if (gen_count[order[j]]>gen_count[order[i]]){int t=order[i];order[i]=order[j];order[j]=t;}
    for (int i = 0; i < 10; i++) {
        uint8_t c = order[i];
        if (c >= 32 && c < 127)
            fprintf(stderr, "'%c':%ld ", c, gen_count[c]);
        else
            fprintf(stderr, "0x%02x:%ld ", c, gen_count[c]);
    }
    fprintf(stderr, "\n");

    free(generated); free(seed_data); free(trigram);
    return 0;
}
