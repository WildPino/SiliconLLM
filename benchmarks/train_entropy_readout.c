#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "../src/silicon_entropy.h"

#include <immintrin.h>
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
    float decay;
    uint32_t codebook_seed;
    float alpha;
} WeightsFileHeader;

uint8_t* data;
long data_size = 0;

int train_start, train_len;
int val_start, val_len;

float* features_train;
float* features_val;

uint8_t* target_train;
uint8_t* target_val;
uint8_t* ctx_train;
uint8_t* ctx_val;
uint8_t* ctx2_train;
uint8_t* ctx2_val;

float bigram_logits[CLASSES][CLASSES];
float (*trigram_logits)[CLASSES][CLASSES];

static inline float dot_product_simd(const float* w, const float* f, int n) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8) {
        sum = _mm256_fmadd_ps(_mm256_loadu_ps(&w[i]), _mm256_loadu_ps(&f[i]), sum);
    }
    float out[8];
    _mm256_storeu_ps(out, sum);
    float res = out[0] + out[1] + out[2] + out[3] + out[4] + out[5] + out[6] + out[7];
    for (; i < n; i++) res += w[i] * f[i];
    return res;
}

static inline void grad_update_simd(float* gradW, const float* f, float err_div_batch, int n) {
    __m256 err_vec = _mm256_set1_ps(err_div_batch);
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 gw = _mm256_loadu_ps(&gradW[i]);
        __m256 fv = _mm256_loadu_ps(&f[i]);
        gw = _mm256_fmadd_ps(err_vec, fv, gw);
        _mm256_storeu_ps(&gradW[i], gw);
    }
    for (; i < n; i++) gradW[i] += err_div_batch * f[i];
}

typedef struct {
    float W[CLASSES][SEE_FEATURE_DIM];
    float B[CLASSES];
    float mW[CLASSES][SEE_FEATURE_DIM];
    float vW[CLASSES][SEE_FEATURE_DIM];
    float mB[CLASSES];
    float vB[CLASSES];
    int t;
} AdamState;

void adam_init(AdamState* adam) {
    memset(adam, 0, sizeof(AdamState));
}

void adam_step(AdamState* adam, float* gradW, float gradB[CLASSES], int num_features, float lr, float beta1, float beta2, float eps, float wd) {
    adam->t++;
    float lr_t = lr * sqrtf(1.0f - powf(beta2, adam->t)) / (1.0f - powf(beta1, adam->t));
    for (int c = 0; c < CLASSES; c++) {
        adam->mB[c] = beta1 * adam->mB[c] + (1.0f - beta1) * gradB[c];
        adam->vB[c] = beta2 * adam->vB[c] + (1.0f - beta2) * gradB[c] * gradB[c];
        adam->B[c] -= lr_t * (adam->mB[c] / (sqrtf(adam->vB[c]) + eps) + wd * adam->B[c]);
        for (int f = 0; f < num_features; f++) {
            adam->mW[c][f] = beta1 * adam->mW[c][f] + (1.0f - beta1) * gradW[c * SEE_FEATURE_DIM + f];
            adam->vW[c][f] = beta2 * adam->vW[c][f] + (1.0f - beta2) * gradW[c * SEE_FEATURE_DIM + f] * gradW[c * SEE_FEATURE_DIM + f];
            adam->W[c][f] -= lr_t * (adam->mW[c][f] / (sqrtf(adam->vW[c][f]) + eps) + wd * adam->W[c][f]);
        }
    }
}

void compute_ngrams(double alpha) {
    double bi_counts[CLASSES][CLASSES] = {0};
    double bi_totals[CLASSES] = {0};
    double (*tri_counts)[CLASSES][CLASSES] = malloc(CLASSES * CLASSES * CLASSES * sizeof(double));
    double (*tri_totals)[CLASSES] = malloc(CLASSES * CLASSES * sizeof(double));
    memset(tri_counts, 0, CLASSES * CLASSES * CLASSES * sizeof(double));
    memset(tri_totals, 0, CLASSES * CLASSES * sizeof(double));

    trigram_logits = malloc(CLASSES * CLASSES * CLASSES * sizeof(float));
    
    for (int i = 0; i < train_len; i++) {
        uint8_t t = target_train[i];
        uint8_t c1 = ctx_train[i];
        uint8_t c2 = ctx2_train[i];
        bi_counts[c1][t]++; bi_totals[c1]++;
        tri_counts[c2][c1][t]++; tri_totals[c2][c1]++;
    }
    
    for (int i = 0; i < CLASSES; i++) {
        for (int j = 0; j < CLASSES; j++) {
            double b_prob = (bi_counts[i][j] + alpha) / (bi_totals[i] + CLASSES * alpha);
            bigram_logits[i][j] = (float)log(b_prob);
            for (int k = 0; k < CLASSES; k++) {
                double t_prob = (tri_counts[i][j][k] + alpha) / (tri_totals[i][j] + CLASSES * alpha);
                trigram_logits[i][j][k] = (float)log(t_prob);
            }
        }
    }
    free(tri_counts);
    free(tri_totals);
}

void extract_dataset(SiliconEntropyState* see, int start_idx, int len, float* out_features, uint8_t* out_targets, uint8_t* out_ctx, uint8_t* out_ctx2) {
    see_reset(see);
    // Observe initial bytes before the evaluation window starts to build context.
    // We observe up to `start_idx + 1` (inclusive) because to predict `data[start_idx + 2]`, we need context up to `data[start_idx + 1]`.
    for (int i = 0; i <= start_idx + 1; i++) {
        see_observe(see, data[i]);
    }

    for (int i = 0; i < len; i++) {
        int global_idx = start_idx + i;
        out_targets[i] = data[global_idx + 2];
        out_ctx[i] = data[global_idx + 1];
        out_ctx2[i] = data[global_idx];
        
        see_extract(see, &out_features[(size_t)i * SEE_FEATURE_DIM]);
        // Observe actual target to update state for the next step
        see_observe(see, out_targets[i]);
    }
}

float feature_means[SEE_FEATURE_DIM];
float feature_stds[SEE_FEATURE_DIM];

void normalize_features() {
    for (int f = 0; f < SEE_FEATURE_DIM; f++) {
        double mean = 0.0;
        for (int i = 0; i < train_len; i++) mean += features_train[(size_t)i * SEE_FEATURE_DIM + f];
        mean /= train_len;
        double var = 0.0;
        for (int i = 0; i < train_len; i++) var += (features_train[(size_t)i * SEE_FEATURE_DIM + f] - mean) * (features_train[(size_t)i * SEE_FEATURE_DIM + f] - mean);
        var /= train_len;
        double std_dev = sqrt(var) + 1e-8;

        feature_means[f] = (float)mean;
        feature_stds[f] = (float)std_dev;

        for (int i = 0; i < train_len; i++) features_train[(size_t)i * SEE_FEATURE_DIM + f] = (features_train[(size_t)i * SEE_FEATURE_DIM + f] - mean) / std_dev;
        for (int i = 0; i < val_len; i++) features_val[(size_t)i * SEE_FEATURE_DIM + f] = (features_val[(size_t)i * SEE_FEATURE_DIM + f] - mean) / std_dev;
    }
}

double evaluate_model(AdamState* model) {
    double total_loss = 0.0;
    for (int i = 0; i < val_len; i++) {
        float logits[CLASSES];
        float max_l = -1e9;
        for (int c = 0; c < CLASSES; c++) {
            logits[c] = model->B[c] + trigram_logits[ctx2_val[i]][ctx_val[i]][c];
            logits[c] += dot_product_simd(model->W[c], &features_val[(size_t)i * SEE_FEATURE_DIM], SEE_FEATURE_DIM);
            if (logits[c] > max_l) max_l = logits[c];
        }
        float sum_e = 0.0f;
        for (int c = 0; c < CLASSES; c++) {
            sum_e += expf(logits[c] - max_l);
        }
        float prob = fmaxf(expf(logits[target_val[i]] - max_l) / sum_e, 1e-10f);
        total_loss -= log2(prob);
    }
    return total_loss / val_len;
}

void train_logistic_regression(AdamState* model, int epochs, int batch_size, float lr) {
    float* gradW = (float*)malloc(CLASSES * SEE_FEATURE_DIM * sizeof(float));
    float gradB[CLASSES];
    float logits[CLASSES];
    float probs[CLASSES];
    
    AdamState* best_adam = (AdamState*)malloc(sizeof(AdamState));
    double best_bpb = evaluate_model(model);
    memcpy(best_adam, model, sizeof(AdamState));
    
    printf("Epoch %2d | Val BPB: %.4f\n", 0, best_bpb);
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        memset(gradW, 0, CLASSES * SEE_FEATURE_DIM * sizeof(float));
        memset(gradB, 0, sizeof(gradB));
        
        for (int i = 0; i < train_len; i++) {
            float max_l = -1e9;
            for (int c = 0; c < CLASSES; c++) {
                logits[c] = model->B[c] + trigram_logits[ctx2_train[i]][ctx_train[i]][c];
                logits[c] += dot_product_simd(model->W[c], &features_train[(size_t)i * SEE_FEATURE_DIM], SEE_FEATURE_DIM);
                if (logits[c] > max_l) max_l = logits[c];
            }
            float sum_e = 0.0f;
            for (int c = 0; c < CLASSES; c++) {
                probs[c] = expf(logits[c] - max_l);
                sum_e += probs[c];
            }
            for (int c = 0; c < CLASSES; c++) probs[c] /= sum_e;
            for (int c = 0; c < CLASSES; c++) {
                float err = probs[c] - (c == target_train[i] ? 1.0f : 0.0f);
                gradB[c] += err / batch_size;
                grad_update_simd(&gradW[c * SEE_FEATURE_DIM], &features_train[(size_t)i * SEE_FEATURE_DIM], err / batch_size, SEE_FEATURE_DIM);
            }
            if ((i + 1) % batch_size == 0 || (i + 1) == train_len) {
                adam_step(model, gradW, gradB, SEE_FEATURE_DIM, lr, 0.9f, 0.999f, 1e-8f, 1e-4f);
                memset(gradW, 0, CLASSES * SEE_FEATURE_DIM * sizeof(float));
                memset(gradB, 0, sizeof(gradB));
            }
        }
        double val_bpb = evaluate_model(model);
        printf("Epoch %2d | Val BPB: %.4f\n", epoch + 1, val_bpb);
        if (val_bpb < best_bpb) {
            best_bpb = val_bpb;
            memcpy(best_adam, model, sizeof(AdamState));
        }
    }
    memcpy(model, best_adam, sizeof(AdamState));
    free(best_adam);
    free(gradW);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <dataset_path> <out_weights.bin> [--train-start %%] [--train-len %%] [--val-start %%] [--val-len %%] [--t3 <int>]\n", argv[0]);
        return 1;
    }

    const char* dataset_path = argv[1];
    const char* out_weights = argv[2];

    int train_start_pct = 0, train_len_pct = 50;
    int val_start_pct = 50, val_len_pct = 25;
    int t3_tokens = 4;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--train-start") == 0 && i + 1 < argc) train_start_pct = atoi(argv[++i]);
        if (strcmp(argv[i], "--train-len") == 0 && i + 1 < argc) train_len_pct = atoi(argv[++i]);
        if (strcmp(argv[i], "--val-start") == 0 && i + 1 < argc) val_start_pct = atoi(argv[++i]);
        if (strcmp(argv[i], "--val-len") == 0 && i + 1 < argc) val_len_pct = atoi(argv[++i]);
        if (strcmp(argv[i], "--t3") == 0 && i + 1 < argc) t3_tokens = atoi(argv[++i]);
    }

    FILE* f = fopen(dataset_path, "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END);
    data_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    data = malloc(data_size);
    fread(data, 1, data_size, f);
    fclose(f);

    train_start = (int)(((long long)data_size * train_start_pct) / 100);
    train_len   = (int)(((long long)data_size * train_len_pct)   / 100);
    val_start   = (int)(((long long)data_size * val_start_pct)   / 100);
    val_len     = (int)(((long long)data_size * val_len_pct)     / 100);

    printf("Dataset Size: %ld\n", data_size);
    printf("Train Split: %d to %d (len %d)\n", train_start, train_start + train_len, train_len);
    printf("Val Split: %d to %d (len %d)\n", val_start, val_start + val_len, val_len);
    printf("t3_tokens: %d\n", t3_tokens);

    features_train = (float*)malloc((size_t)train_len * SEE_FEATURE_DIM * sizeof(float));
    features_val = (float*)malloc((size_t)val_len * SEE_FEATURE_DIM * sizeof(float));
    target_train = malloc(train_len); ctx_train = malloc(train_len); ctx2_train = malloc(train_len);
    target_val = malloc(val_len); ctx_val = malloc(val_len); ctx2_val = malloc(val_len);

    printf("Extracting features...\n");
    SiliconEntropyState see;
    see_init(&see, 42, 4, 0.75f);
    see.history_tokens = t3_tokens;
    
    extract_dataset(&see, train_start, train_len, features_train, target_train, ctx_train, ctx2_train);
    extract_dataset(&see, val_start, val_len, features_val, target_val, ctx_val, ctx2_val);
    
    printf("Normalizing features...\n");
    normalize_features();
    
    printf("Computing N-Grams (alpha=0.1)...\n");
    compute_ngrams(0.1f);
    
    printf("Training Softmax Residual...\n");
    AdamState* model = (AdamState*)malloc(sizeof(AdamState));
    adam_init(model);
    
    float lrs[] = {0.003f, 0.001f, 0.0003f};
    for (int l = 0; l < 3; l++) {
        printf("\n--- LR: %.4f ---\n", lrs[l]);
        train_logistic_regression(model, 10, 256, lrs[l]);
    }
    
    printf("\nSaving Weights to %s...\n", out_weights);
    FILE* fw = fopen(out_weights, "wb");
    WeightsFileHeader header = {0x53454531, 1, SEE_FEATURE_DIM, 4, 0.75f, 42, 0.1f};
    fwrite(&header, sizeof(WeightsFileHeader), 1, fw);
    
    // Save N-Grams
    fwrite(trigram_logits, sizeof(float), CLASSES * CLASSES * CLASSES, fw);
    
    // Save Normalization Stats
    fwrite(feature_means, sizeof(float), SEE_FEATURE_DIM, fw);
    fwrite(feature_stds, sizeof(float), SEE_FEATURE_DIM, fw);
    
    // Save Softmax Weights and Biases
    fwrite(model->W, sizeof(float), CLASSES * SEE_FEATURE_DIM, fw);
    fwrite(model->B, sizeof(float), CLASSES, fw);
    
    fclose(fw);
    printf("Done.\n");
    return 0;
}
