// Phase 42.C — Reservoir² (Second ESN)
// Augments SEE's 192D features with 64D from a second leaky-integrator reservoir.
// Combined feature vector: [192D SEE | 64D R2] = 256D
// Trains softmax readout on 256D. R2 captures temporal structure between SEE states.
//
// Build:
//   gcc -O3 -march=native -mavx2 -mfma \
//       benchmarks/phase42c_reservoir2.c src/silicon_entropy.c src/silicon_v0.c src/silicon_r2.c \
//       -o bin/phase42c_r2.exe -lm -I .
// Run:
//   ./bin/phase42c_r2.exe <dataset.bin> <weights_out.bin>
//
// Compare val BPB against Phase 42.0 baseline weights.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include "../src/silicon_entropy.h"
#include "../src/silicon_r2.h"

#define CLASSES    256
#define FEAT_SEE   SEE_FEATURE_DIM  // 192
#define FEAT_R2    R2_DIM           // 64
#define FEAT_DIM_C (FEAT_SEE + FEAT_R2)  // 256

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t feature_dim;
    uint32_t chunk_size;
    float    decay;
    uint32_t codebook_seed;
    float    alpha_ngram;
    float    r2_alpha;
    uint32_t r2_seed;
} WeightsFileHeader42C;

typedef struct {
    float W[CLASSES][FEAT_DIM_C];
    float B[CLASSES];
    float mW[CLASSES][FEAT_DIM_C];
    float vW[CLASSES][FEAT_DIM_C];
    float mB[CLASSES];
    float vB[CLASSES];
    int   t;
} AdamState42C;

static inline float dot_product_simd(const float* w, const float* f, int n) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8)
        sum = _mm256_fmadd_ps(_mm256_loadu_ps(&w[i]), _mm256_loadu_ps(&f[i]), sum);
    float out[8]; _mm256_storeu_ps(out, sum);
    float res = out[0]+out[1]+out[2]+out[3]+out[4]+out[5]+out[6]+out[7];
    for (; i < n; i++) res += w[i] * f[i];
    return res;
}

static inline void grad_update_simd(float* gradW, const float* f, float e_div_b, int n) {
    __m256 ev = _mm256_set1_ps(e_div_b);
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 gw = _mm256_loadu_ps(&gradW[i]);
        gw = _mm256_fmadd_ps(ev, _mm256_loadu_ps(&f[i]), gw);
        _mm256_storeu_ps(&gradW[i], gw);
    }
    for (; i < n; i++) gradW[i] += e_div_b * f[i];
}

// --------------------------------------------------------------------------

static uint8_t* data;
static long     data_size;
static int      train_start, train_len, val_start, val_len;

static float*   features_train;
static float*   features_val;
static uint8_t* target_train;
static uint8_t* target_val;
static uint8_t* ctx_train;
static uint8_t* ctx_val;
static uint8_t* ctx2_train;
static uint8_t* ctx2_val;

static float bigram_logits[CLASSES][CLASSES];
static float (*trigram_logits)[CLASSES][CLASSES];
static float feature_means[FEAT_DIM_C];
static float feature_stds[FEAT_DIM_C];

void extract_dataset_r2(SiliconEntropyState* see, SiliconR2* r2,
                        int start, int len,
                        float* out_feat, uint8_t* out_tgt,
                        uint8_t* out_ctx, uint8_t* out_ctx2) {
    see_reset(see);
    silicon_r2_reset(r2);
    for (int i = 0; i <= start + 1; i++) {
        see_observe(see, data[i]);
        float tmp[FEAT_SEE];
        see_extract(see, tmp);
        silicon_r2_update(r2, tmp);
    }
    for (int i = 0; i < len; i++) {
        int g = start + i;
        out_tgt[i]  = data[g + 2];
        out_ctx[i]  = data[g + 1];
        out_ctx2[i] = data[g];

        float see_feats[FEAT_SEE];
        see_extract(see, see_feats);

        float r2_feats[FEAT_R2];
        silicon_r2_extract(r2, r2_feats);

        size_t base = (size_t)i * FEAT_DIM_C;
        memcpy(&out_feat[base],           see_feats, FEAT_SEE * sizeof(float));
        memcpy(&out_feat[base + FEAT_SEE], r2_feats, FEAT_R2  * sizeof(float));

        see_observe(see, out_tgt[i]);
        float updated[FEAT_SEE];
        see_extract(see, updated);
        silicon_r2_update(r2, updated);
    }
}

void compute_ngrams(double alpha) {
    double bi_counts[CLASSES][CLASSES] = {0};
    double bi_totals[CLASSES] = {0};
    double (*tri_counts)[CLASSES][CLASSES] = malloc(CLASSES*CLASSES*CLASSES*sizeof(double));
    double (*tri_totals)[CLASSES]          = malloc(CLASSES*CLASSES*sizeof(double));
    memset(tri_counts, 0, CLASSES*CLASSES*CLASSES*sizeof(double));
    memset(tri_totals, 0, CLASSES*CLASSES*sizeof(double));
    trigram_logits = malloc(CLASSES*CLASSES*CLASSES*sizeof(float));
    for (int i = 0; i < train_len; i++) {
        uint8_t t = target_train[i], c1 = ctx_train[i], c2 = ctx2_train[i];
        bi_counts[c1][t]++; bi_totals[c1]++;
        tri_counts[c2][c1][t]++; tri_totals[c2][c1]++;
    }
    for (int i = 0; i < CLASSES; i++)
        for (int j = 0; j < CLASSES; j++) {
            bigram_logits[i][j] = (float)log((bi_counts[i][j]+alpha)/(bi_totals[i]+CLASSES*alpha));
            for (int k = 0; k < CLASSES; k++)
                trigram_logits[i][j][k] = (float)log(
                    (tri_counts[i][j][k]+alpha)/(tri_totals[i][j]+CLASSES*alpha));
        }
    free(tri_counts); free(tri_totals);
}

void normalize_features(void) {
    for (int f = 0; f < FEAT_DIM_C; f++) {
        double mean = 0.0;
        for (int i = 0; i < train_len; i++) mean += features_train[(size_t)i*FEAT_DIM_C+f];
        mean /= train_len;
        double var = 0.0;
        for (int i = 0; i < train_len; i++) {
            double d = features_train[(size_t)i*FEAT_DIM_C+f] - mean;
            var += d*d;
        }
        double std_dev = sqrt(var/train_len) + 1e-8;
        feature_means[f] = (float)mean;
        feature_stds[f]  = (float)std_dev;
        for (int i = 0; i < train_len; i++)
            features_train[(size_t)i*FEAT_DIM_C+f] = (features_train[(size_t)i*FEAT_DIM_C+f] - mean) / std_dev;
        for (int i = 0; i < val_len; i++)
            features_val[(size_t)i*FEAT_DIM_C+f] = (features_val[(size_t)i*FEAT_DIM_C+f] - mean) / std_dev;
    }
}

double evaluate_model(AdamState42C* model) {
    double total = 0.0;
    for (int i = 0; i < val_len; i++) {
        float logits[CLASSES], max_l = -1e9f;
        for (int c = 0; c < CLASSES; c++) {
            logits[c] = model->B[c]
                      + trigram_logits[ctx2_val[i]][ctx_val[i]][c]
                      + dot_product_simd(model->W[c], &features_val[(size_t)i*FEAT_DIM_C], FEAT_DIM_C);
            if (logits[c] > max_l) max_l = logits[c];
        }
        float sum_e = 0.0f;
        for (int c = 0; c < CLASSES; c++) sum_e += expf(logits[c] - max_l);
        total -= log2(fmaxf(expf(logits[target_val[i]] - max_l) / sum_e, 1e-10f));
    }
    return total / val_len;
}

void train_lr(AdamState42C* model, int epochs, int batch_size, float lr) {
    float* gradW = malloc(CLASSES * FEAT_DIM_C * sizeof(float));
    float  gradB[CLASSES], logits[CLASSES], probs[CLASSES];
    AdamState42C* best = malloc(sizeof(AdamState42C));
    double best_bpb = evaluate_model(model);
    memcpy(best, model, sizeof(AdamState42C));
    printf("  Epoch  0 | Val BPB %.4f\n", best_bpb);

    for (int ep = 0; ep < epochs; ep++) {
        memset(gradW, 0, CLASSES*FEAT_DIM_C*sizeof(float));
        memset(gradB, 0, sizeof(gradB));
        for (int i = 0; i < train_len; i++) {
            float max_l = -1e9f;
            for (int c = 0; c < CLASSES; c++) {
                logits[c] = model->B[c]
                          + trigram_logits[ctx2_train[i]][ctx_train[i]][c]
                          + dot_product_simd(model->W[c], &features_train[(size_t)i*FEAT_DIM_C], FEAT_DIM_C);
                if (logits[c] > max_l) max_l = logits[c];
            }
            float sum_e = 0.0f;
            for (int c = 0; c < CLASSES; c++) { probs[c] = expf(logits[c]-max_l); sum_e += probs[c]; }
            for (int c = 0; c < CLASSES; c++) probs[c] /= sum_e;
            for (int c = 0; c < CLASSES; c++) {
                float err = probs[c] - (c == target_train[i] ? 1.0f : 0.0f);
                gradB[c] += err / batch_size;
                grad_update_simd(&gradW[c*FEAT_DIM_C], &features_train[(size_t)i*FEAT_DIM_C], err/batch_size, FEAT_DIM_C);
            }
            if ((i+1) % batch_size == 0 || (i+1) == train_len) {
                model->t++;
                float lr_t = lr * sqrtf(1.0f - powf(0.999f, model->t)) / (1.0f - powf(0.9f, model->t));
                for (int c = 0; c < CLASSES; c++) {
                    model->mB[c] = 0.9f*model->mB[c] + 0.1f*gradB[c];
                    model->vB[c] = 0.999f*model->vB[c] + 0.001f*gradB[c]*gradB[c];
                    model->B[c] -= lr_t*(model->mB[c]/(sqrtf(model->vB[c])+1e-8f) + 1e-4f*model->B[c]);
                    for (int ff = 0; ff < FEAT_DIM_C; ff++) {
                        model->mW[c][ff] = 0.9f*model->mW[c][ff] + 0.1f*gradW[c*FEAT_DIM_C+ff];
                        model->vW[c][ff] = 0.999f*model->vW[c][ff] + 0.001f*gradW[c*FEAT_DIM_C+ff]*gradW[c*FEAT_DIM_C+ff];
                        model->W[c][ff] -= lr_t*(model->mW[c][ff]/(sqrtf(model->vW[c][ff])+1e-8f) + 1e-4f*model->W[c][ff]);
                    }
                }
                memset(gradW, 0, CLASSES*FEAT_DIM_C*sizeof(float));
                memset(gradB, 0, sizeof(gradB));
            }
        }
        double bpb = evaluate_model(model);
        printf("  Epoch %2d | Val BPB %.4f\n", ep+1, bpb);
        if (bpb < best_bpb) { best_bpb = bpb; memcpy(best, model, sizeof(AdamState42C)); }
    }
    memcpy(model, best, sizeof(AdamState42C));
    free(best); free(gradW);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <dataset.bin> <out_weights.bin> [--r2-alpha <float>] [--r2-seed <int>]\n", argv[0]);
        return 1;
    }
    const char* dataset_path = argv[1];
    const char* out_weights  = argv[2];

    float r2_alpha = 0.9f;
    int   r2_seed  = 123;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--r2-alpha") == 0 && i+1 < argc) r2_alpha = (float)atof(argv[++i]);
        if (strcmp(argv[i], "--r2-seed")  == 0 && i+1 < argc) r2_seed  = atoi(argv[++i]);
    }

    FILE* f = fopen(dataset_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", dataset_path); return 1; }
    fseek(f, 0, SEEK_END);
    data_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    data = malloc(data_size);
    fread(data, 1, data_size, f);
    fclose(f);

    train_start = (int)(((long long)data_size * 0)  / 100);
    train_len   = (int)(((long long)data_size * 50) / 100);
    val_start   = (int)(((long long)data_size * 50) / 100);
    val_len     = (int)(((long long)data_size * 25) / 100);

    printf("Dataset: %ld bytes\n", data_size);
    printf("Train: len=%d   Val: len=%d\n", train_len, val_len);
    printf("R2 alpha=%.2f  seed=%d\n", r2_alpha, r2_seed);
    printf("Feature dim: %d SEE + %d R2 = %d total\n", FEAT_SEE, FEAT_R2, FEAT_DIM_C);

    features_train = malloc((size_t)train_len * FEAT_DIM_C * sizeof(float));
    features_val   = malloc((size_t)val_len   * FEAT_DIM_C * sizeof(float));
    target_train   = malloc(train_len); ctx_train  = malloc(train_len); ctx2_train = malloc(train_len);
    target_val     = malloc(val_len);   ctx_val    = malloc(val_len);   ctx2_val   = malloc(val_len);

    SiliconEntropyState see;
    SiliconR2 r2;
    silicon_r2_init(&r2, r2_seed, r2_alpha);

    printf("\nExtracting train features (SEE+R2)...\n");
    see_init(&see, 42, 4, 0.75f);
    extract_dataset_r2(&see, &r2, train_start, train_len,
                       features_train, target_train, ctx_train, ctx2_train);

    printf("Extracting val features (SEE+R2)...\n");
    see_init(&see, 42, 4, 0.75f);
    silicon_r2_reset(&r2);
    extract_dataset_r2(&see, &r2, val_start, val_len,
                       features_val, target_val, ctx_val, ctx2_val);

    printf("Normalizing %dD features...\n", FEAT_DIM_C);
    normalize_features();

    printf("Computing N-grams...\n");
    compute_ngrams(0.1);

    printf("Training softmax (256 x %d)...\n", FEAT_DIM_C);
    AdamState42C* model = calloc(1, sizeof(AdamState42C));
    float lrs[] = { 0.003f, 0.001f, 0.0003f };
    for (int l = 0; l < 3; l++) {
        printf("\nLR %.4f\n", lrs[l]);
        train_lr(model, 10, 256, lrs[l]);
    }

    double final_bpb = evaluate_model(model);
    printf("\nFinal Val BPB: %.4f\n", final_bpb);

    printf("Saving weights to %s...\n", out_weights);
    FILE* fw = fopen(out_weights, "wb");
    WeightsFileHeader42C hdr = {0x53454532, 1, FEAT_DIM_C, 4, 0.75f, 42, 0.1f, r2_alpha, (uint32_t)r2_seed};
    fwrite(&hdr,          sizeof(hdr), 1, fw);
    fwrite(trigram_logits, sizeof(float), CLASSES*CLASSES*CLASSES, fw);
    fwrite(feature_means,  sizeof(float), FEAT_DIM_C, fw);
    fwrite(feature_stds,   sizeof(float), FEAT_DIM_C, fw);
    fwrite(model->W,       sizeof(float), CLASSES*FEAT_DIM_C, fw);
    fwrite(model->B,       sizeof(float), CLASSES, fw);
    fclose(fw);
    printf("Done.\n");
    return 0;
}
