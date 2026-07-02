// Phase 43.A — Multi-Timescale L1 Tribunal
// Tests whether replacing the single-alpha chunk EMA with 3 per-byte EMA bands
// improves representation quality at a fixed 192D readout.
//
// Configurations:
//   0 = legacy   : multiscale_mode=0, decay=0.75 (chunk mean+last, baseline)
//   1 = ms_fast05: multiscale_mode=1, fast=0.5,  mid=0.9, slow=0.99
//   2 = ms_fast07: multiscale_mode=1, fast=0.7,  mid=0.9, slow=0.99
//
// L1 bands (mode 1 or 2):
//   l1_state[0:42]   alpha_fast of L0[0:42]   -- morpheme / character scale
//   l1_state[43:85]  alpha_mid  of L0[0:42]   -- word / syntagma scale
//   l1_state[86:127] alpha_slow of L0[0:41]   -- phrase / local context scale
//
// Build (from repo root):
//   gcc -O3 -march=native -mavx2 -mfma \
//       benchmarks/phase38-42/phase43a_multiscale.c \
//       src/silicon_entropy.c src/silicon_v0.c \
//       -o bin/phase43a_multiscale.exe -lm -I .
// Run:
//   bin/phase43a_multiscale.exe <dataset> <weights_prefix>

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

#include "src/silicon_entropy.h"

#define CLASSES   256
#define FEAT_DIM  SEE_FEATURE_DIM  // 192

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t feature_dim;
    uint32_t chunk_size;
    float    decay;
    uint32_t codebook_seed;
    float    ngram_alpha;
    uint32_t multiscale_mode;
    float    alpha_fast;
    float    alpha_mid;
    float    alpha_slow;
} WeightsFileHeader43A;

typedef struct {
    float W[CLASSES][FEAT_DIM];
    float B[CLASSES];
    float mW[CLASSES][FEAT_DIM];
    float vW[CLASSES][FEAT_DIM];
    float mB[CLASSES];
    float vB[CLASSES];
    int   t;
} AdamState;

// ------------------------------------------------------------

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

// ------------------------------------------------------------

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

static float feature_means[FEAT_DIM];
static float feature_stds[FEAT_DIM];

void extract_dataset(SiliconEntropyState* see, int start, int len,
                     float* out_feat, uint8_t* out_tgt,
                     uint8_t* out_ctx, uint8_t* out_ctx2) {
    see_reset(see);
    for (int i = 0; i <= start + 1; i++) see_observe(see, data[i]);
    for (int i = 0; i < len; i++) {
        int g = start + i;
        out_tgt[i]  = data[g + 2];
        out_ctx[i]  = data[g + 1];
        out_ctx2[i] = data[g];
        see_extract(see, &out_feat[(size_t)i * FEAT_DIM]);
        see_observe(see, out_tgt[i]);
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
            bigram_logits[i][j] = (float)log(
                (bi_counts[i][j]+alpha)/(bi_totals[i]+CLASSES*alpha));
            for (int k = 0; k < CLASSES; k++)
                trigram_logits[i][j][k] = (float)log(
                    (tri_counts[i][j][k]+alpha)/(tri_totals[i][j]+CLASSES*alpha));
        }
    free(tri_counts); free(tri_totals);
}

void normalize_features(void) {
    for (int f = 0; f < FEAT_DIM; f++) {
        double mean = 0.0;
        for (int i = 0; i < train_len; i++) mean += features_train[(size_t)i*FEAT_DIM+f];
        mean /= train_len;
        double var = 0.0;
        for (int i = 0; i < train_len; i++) {
            double d = features_train[(size_t)i*FEAT_DIM+f] - mean; var += d*d;
        }
        double std_dev = sqrt(var/train_len) + 1e-8;
        feature_means[f] = (float)mean;
        feature_stds[f]  = (float)std_dev;
        for (int i = 0; i < train_len; i++)
            features_train[(size_t)i*FEAT_DIM+f] =
                (features_train[(size_t)i*FEAT_DIM+f] - (float)mean) / (float)std_dev;
        for (int i = 0; i < val_len; i++)
            features_val[(size_t)i*FEAT_DIM+f] =
                (features_val[(size_t)i*FEAT_DIM+f] - (float)mean) / (float)std_dev;
    }
}

double evaluate_model(AdamState* model) {
    double total = 0.0;
    for (int i = 0; i < val_len; i++) {
        float logits[CLASSES], max_l = -1e9f;
        for (int c = 0; c < CLASSES; c++) {
            logits[c] = model->B[c]
                      + trigram_logits[ctx2_val[i]][ctx_val[i]][c]
                      + dot_product_simd(model->W[c], &features_val[(size_t)i*FEAT_DIM], FEAT_DIM);
            if (logits[c] > max_l) max_l = logits[c];
        }
        float sum_e = 0.0f;
        for (int c = 0; c < CLASSES; c++) sum_e += expf(logits[c] - max_l);
        total -= log2(fmaxf(expf(logits[target_val[i]] - max_l) / sum_e, 1e-10f));
    }
    return total / val_len;
}

void train_lr(AdamState* model, int epochs, int batch_size, float lr) {
    float* gradW = malloc(CLASSES * FEAT_DIM * sizeof(float));
    float  gradB[CLASSES], logits[CLASSES], probs[CLASSES];
    AdamState* best = malloc(sizeof(AdamState));
    double best_bpb = evaluate_model(model);
    memcpy(best, model, sizeof(AdamState));
    printf("  Epoch  0 | Val BPB %.4f\n", best_bpb);

    for (int ep = 0; ep < epochs; ep++) {
        memset(gradW, 0, CLASSES*FEAT_DIM*sizeof(float));
        memset(gradB, 0, sizeof(gradB));
        for (int i = 0; i < train_len; i++) {
            float max_l = -1e9f;
            for (int c = 0; c < CLASSES; c++) {
                logits[c] = model->B[c]
                          + trigram_logits[ctx2_train[i]][ctx_train[i]][c]
                          + dot_product_simd(model->W[c], &features_train[(size_t)i*FEAT_DIM], FEAT_DIM);
                if (logits[c] > max_l) max_l = logits[c];
            }
            float sum_e = 0.0f;
            for (int c = 0; c < CLASSES; c++) { probs[c] = expf(logits[c]-max_l); sum_e += probs[c]; }
            for (int c = 0; c < CLASSES; c++) probs[c] /= sum_e;
            for (int c = 0; c < CLASSES; c++) {
                float err = probs[c] - (c == target_train[i] ? 1.0f : 0.0f);
                gradB[c] += err / batch_size;
                grad_update_simd(&gradW[c*FEAT_DIM], &features_train[(size_t)i*FEAT_DIM],
                                 err/batch_size, FEAT_DIM);
            }
            if ((i+1) % batch_size == 0 || (i+1) == train_len) {
                model->t++;
                float lr_t = lr * sqrtf(1.0f - powf(0.999f, model->t))
                                / (1.0f - powf(0.9f, model->t));
                for (int c = 0; c < CLASSES; c++) {
                    model->mB[c] = 0.9f*model->mB[c] + 0.1f*gradB[c];
                    model->vB[c] = 0.999f*model->vB[c] + 0.001f*gradB[c]*gradB[c];
                    model->B[c] -= lr_t*(model->mB[c]/(sqrtf(model->vB[c])+1e-8f)
                                        + 1e-4f*model->B[c]);
                    for (int f = 0; f < FEAT_DIM; f++) {
                        float g = gradW[c*FEAT_DIM+f];
                        model->mW[c][f] = 0.9f*model->mW[c][f] + 0.1f*g;
                        model->vW[c][f] = 0.999f*model->vW[c][f] + 0.001f*g*g;
                        model->W[c][f] -= lr_t*(model->mW[c][f]/(sqrtf(model->vW[c][f])+1e-8f)
                                               + 1e-4f*model->W[c][f]);
                    }
                }
                memset(gradW, 0, CLASSES*FEAT_DIM*sizeof(float));
                memset(gradB, 0, sizeof(gradB));
            }
        }
        double bpb = evaluate_model(model);
        printf("  Epoch %2d | Val BPB %.4f\n", ep+1, bpb); fflush(stdout);
        if (bpb < best_bpb) { best_bpb = bpb; memcpy(best, model, sizeof(AdamState)); }
    }
    memcpy(model, best, sizeof(AdamState));
    free(best); free(gradW);
}

void save_weights(const char* path, AdamState* model,
                  int ms_mode, float af, float am, float as_) {
    FILE* fw = fopen(path, "wb");
    if (!fw) { fprintf(stderr, "Cannot write %s\n", path); return; }
    WeightsFileHeader43A hdr = {
        0x53454535, 1, FEAT_DIM, 4, 0.75f, 42, 0.1f,
        (uint32_t)ms_mode, af, am, as_
    };
    fwrite(&hdr, sizeof(hdr), 1, fw);
    fwrite(trigram_logits, sizeof(float), CLASSES*CLASSES*CLASSES, fw);
    fwrite(feature_means,  sizeof(float), FEAT_DIM, fw);
    fwrite(feature_stds,   sizeof(float), FEAT_DIM, fw);
    fwrite(model->W, sizeof(float), CLASSES*FEAT_DIM, fw);
    fwrite(model->B, sizeof(float), CLASSES, fw);
    fclose(fw);
}

typedef struct {
    const char* name;
    const char* suffix;
    int   ms_mode;
    float fast, mid, slow;
} Config;

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <dataset> <weights_prefix>\n", argv[0]);
        printf("Outputs: <prefix>_legacy.bin  <prefix>_ms05.bin  <prefix>_ms07.bin\n");
        return 1;
    }
    const char* dataset_path   = argv[1];
    const char* weights_prefix = argv[2];

    FILE* f = fopen(dataset_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", dataset_path); return 1; }
    fseek(f, 0, SEEK_END);
    data_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    data = malloc(data_size);
    fread(data, 1, data_size, f);
    fclose(f);

    train_start = (int)(((long long)data_size *  0) / 100);
    train_len   = (int)(((long long)data_size * 50) / 100);
    val_start   = (int)(((long long)data_size * 50) / 100);
    val_len     = (int)(((long long)data_size * 25) / 100);

    printf("Dataset: %ld bytes\n", data_size);
    printf("Train: [%d, %d)  len=%d\n", train_start, train_start+train_len, train_len);
    printf("Val:   [%d, %d)  len=%d\n", val_start,   val_start+val_len,     val_len);
    fflush(stdout);

    features_train = malloc((size_t)train_len * FEAT_DIM * sizeof(float));
    features_val   = malloc((size_t)val_len   * FEAT_DIM * sizeof(float));
    target_train   = malloc(train_len); ctx_train  = malloc(train_len); ctx2_train = malloc(train_len);
    target_val     = malloc(val_len);   ctx_val    = malloc(val_len);   ctx2_val   = malloc(val_len);

    // Configurations
    Config configs[] = {
        { "legacy",   "_legacy.bin", 0, 0.0f, 0.0f, 0.0f   },
        { "ms_f0.5",  "_ms05.bin",   1, 0.5f, 0.9f, 0.99f  },
        { "ms_f0.7",  "_ms07.bin",   1, 0.7f, 0.9f, 0.99f  },
    };
    int n_configs = 3;
    double results[3];

    // N-grams computed once (mode-independent)
    printf("\nComputing N-grams (alpha=0.1)...\n"); fflush(stdout);
    SiliconEntropyState see;
    see_init(&see, 42, 4, 0.75f);
    extract_dataset(&see, train_start, train_len, features_train,
                    target_train, ctx_train, ctx2_train);
    compute_ngrams(0.1);

    for (int ci = 0; ci < n_configs; ci++) {
        Config* cfg = &configs[ci];
        printf("\n====== Config %d (%s) ======\n", ci, cfg->name);
        printf("  multiscale_mode=%d  fast=%.2f  mid=%.2f  slow=%.3f\n",
               cfg->ms_mode, cfg->fast, cfg->mid, cfg->slow);
        fflush(stdout);

        see_init(&see, 42, 4, 0.75f);
        see.multiscale_mode = cfg->ms_mode;
        if (cfg->ms_mode == 1) {
            see.alpha_fast = cfg->fast;
            see.alpha_mid  = cfg->mid;
            see.alpha_slow = cfg->slow;
        }

        printf("Extracting train features...\n"); fflush(stdout);
        extract_dataset(&see, train_start, train_len, features_train,
                        target_train, ctx_train, ctx2_train);
        printf("Extracting val features...\n"); fflush(stdout);
        extract_dataset(&see, val_start, val_len, features_val,
                        target_val, ctx_val, ctx2_val);
        printf("Normalizing...\n"); fflush(stdout);
        normalize_features();

        AdamState* model = calloc(1, sizeof(AdamState));
        float lrs[] = { 0.003f, 0.001f, 0.0003f };
        for (int l = 0; l < 3; l++) {
            printf("\n  LR %.4f\n", lrs[l]); fflush(stdout);
            train_lr(model, 5, 256, lrs[l]);
        }
        results[ci] = evaluate_model(model);

        char out_path[1024];
        snprintf(out_path, sizeof(out_path), "%s%s", weights_prefix, cfg->suffix);
        save_weights(out_path, model, cfg->ms_mode, cfg->fast, cfg->mid, cfg->slow);
        printf("  Saved -> %s\n", out_path); fflush(stdout);
        free(model);
    }

    printf("\n\n========================================\n");
    printf("Phase 43.A -- Multi-Timescale Tribunal\n");
    printf("========================================\n");
    printf("  Baseline (42.0):  2.3197 BPB  (reference from train_entropy_readout.c)\n");
    printf("%-12s  Val BPB   Delta vs legacy\n", "Config");
    for (int ci = 0; ci < n_configs; ci++)
        printf("%-12s  %.4f    %+.4f\n",
               configs[ci].name, results[ci], results[ci] - results[0]);

    double best_bpb = results[0]; int best_ci = 0;
    for (int ci = 1; ci < n_configs; ci++)
        if (results[ci] < best_bpb) { best_bpb = results[ci]; best_ci = ci; }

    printf("\nBest: %s (%.4f BPB, delta=%+.4f vs legacy)\n",
           configs[best_ci].name, best_bpb, best_bpb - results[0]);

    if (best_bpb < results[0] - 0.03)
        printf("SIGNAL: multi-timescale improves > 0.03 BPB -> temporal scale IS a bottleneck\n");
    else
        printf("SIGNAL: flat -> temporal scale not the main bottleneck at this architecture\n");

    return 0;
}
