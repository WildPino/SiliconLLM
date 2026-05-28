#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "../src/silicon_v0.h"

#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#define MAX_SAMPLES 500000
#define MAX_L0_FEATURES 64
#define MAX_L1_FEATURES 512
#define MAX_FEATURES (MAX_L0_FEATURES + MAX_L1_FEATURES)
#define CLASSES 256

uint8_t data[MAX_SAMPLES];
int data_size = 0;

float* features_train;
float* features_val;
float* features_test;

uint8_t target_train[MAX_SAMPLES];
uint8_t target_val[MAX_SAMPLES];
uint8_t target_test[MAX_SAMPLES];

uint8_t ctx_train[MAX_SAMPLES];
uint8_t ctx_val[MAX_SAMPLES];
uint8_t ctx_test[MAX_SAMPLES];

uint8_t ctx2_train[MAX_SAMPLES];
uint8_t ctx2_val[MAX_SAMPLES];
uint8_t ctx2_test[MAX_SAMPLES];

int train_size = 0;
int val_size = 0;
int test_size = 0;

double unigram_probs[CLASSES];
double bigram_probs[CLASSES][CLASSES];
float bigram_logits[CLASSES][CLASSES];

// Allocated dynamically
double (*trigram_probs)[CLASSES][CLASSES];
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
    float W[CLASSES][MAX_FEATURES];
    float B[CLASSES];
    float mW[CLASSES][MAX_FEATURES];
    float vW[CLASSES][MAX_FEATURES];
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
            adam->mW[c][f] = beta1 * adam->mW[c][f] + (1.0f - beta1) * gradW[c * MAX_FEATURES + f];
            adam->vW[c][f] = beta2 * adam->vW[c][f] + (1.0f - beta2) * gradW[c * MAX_FEATURES + f] * gradW[c * MAX_FEATURES + f];
            adam->W[c][f] -= lr_t * (adam->mW[c][f] / (sqrtf(adam->vW[c][f]) + eps) + wd * adam->W[c][f]);
        }
    }
}

void normalize_features(int num_features) {
    for (int f = 0; f < num_features; f++) {
        double mean = 0.0;
        for (int i = 0; i < train_size; i++) mean += features_train[i * num_features + f];
        mean /= train_size;
        double var = 0.0;
        for (int i = 0; i < train_size; i++) var += (features_train[i * num_features + f] - mean) * (features_train[i * num_features + f] - mean);
        var /= train_size;
        double std_dev = sqrt(var) + 1e-8;
        for (int i = 0; i < train_size; i++) features_train[i * num_features + f] = (features_train[i * num_features + f] - mean) / std_dev;
        for (int i = 0; i < val_size; i++) features_val[i * num_features + f] = (features_val[i * num_features + f] - mean) / std_dev;
        for (int i = 0; i < test_size; i++) features_test[i * num_features + f] = (features_test[i * num_features + f] - mean) / std_dev;
    }
}

void _evaluate_model(AdamState* model, int num_features, int use_residual, int use_trigram, double* out_bpb, float* features, uint8_t* ctx, uint8_t* ctx2, uint8_t* target, int size, uint64_t* fwd_cyc) {
    double total_loss = 0.0;
    for (int i = 0; i < size; i++) {
        uint64_t s = __rdtsc();
        float logits[CLASSES];
        float max_l = -1e9;
        for (int c = 0; c < CLASSES; c++) {
            logits[c] = model->B[c];
            if (use_residual) {
                if (use_trigram) logits[c] += trigram_logits[ctx2[i]][ctx[i]][c];
                else logits[c] += bigram_logits[ctx[i]][c];
            }
            if (num_features > 0) logits[c] += dot_product_simd(model->W[c], &features[i * num_features], num_features);
            if (logits[c] > max_l) max_l = logits[c];
        }
        float sum_e = 0.0f;
        for (int c = 0; c < CLASSES; c++) {
            sum_e += expf(logits[c] - max_l);
        }
        float prob = fmaxf(expf(logits[target[i]] - max_l) / sum_e, 1e-10f);
        total_loss -= log2(prob);
        uint64_t e = __rdtsc();
        if (fwd_cyc) *fwd_cyc += (e - s);
    }
    *out_bpb = total_loss / size;
}

void train_logistic_regression(AdamState* model, int num_features, int epochs, int batch_size, float lr, int use_residual, int use_trigram) {
    float* gradW = (float*)malloc(CLASSES * MAX_FEATURES * sizeof(float));
    float gradB[CLASSES];
    float logits[CLASSES];
    float probs[CLASSES];
    
    AdamState* best_adam = (AdamState*)malloc(sizeof(AdamState));
    double best_bpb = 1e9;
    _evaluate_model(model, num_features, use_residual, use_trigram, &best_bpb, features_val, ctx_val, ctx2_val, target_val, val_size, NULL);
    memcpy(best_adam, model, sizeof(AdamState));
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        memset(gradW, 0, CLASSES * MAX_FEATURES * sizeof(float));
        memset(gradB, 0, sizeof(gradB));
        
        for (int i = 0; i < train_size; i++) {
            float max_l = -1e9;
            for (int c = 0; c < CLASSES; c++) {
                logits[c] = model->B[c];
                if (use_residual) {
                    if (use_trigram) logits[c] += trigram_logits[ctx2_train[i]][ctx_train[i]][c];
                    else logits[c] += bigram_logits[ctx_train[i]][c];
                }
                if (num_features > 0) logits[c] += dot_product_simd(model->W[c], &features_train[i * num_features], num_features);
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
                if (num_features > 0) grad_update_simd(&gradW[c * MAX_FEATURES], &features_train[i * num_features], err / batch_size, num_features);
            }
            if ((i + 1) % batch_size == 0 || (i + 1) == train_size) {
                adam_step(model, gradW, gradB, num_features, lr, 0.9f, 0.999f, 1e-8f, 1e-4f);
                memset(gradW, 0, CLASSES * MAX_FEATURES * sizeof(float));
                memset(gradB, 0, sizeof(gradB));
            }
        }
        double val_bpb;
        _evaluate_model(model, num_features, use_residual, use_trigram, &val_bpb, features_val, ctx_val, ctx2_val, target_val, val_size, NULL);
        if (val_bpb < best_bpb) {
            best_bpb = val_bpb;
            memcpy(best_adam, model, sizeof(AdamState));
        }
    }
    memcpy(model, best_adam, sizeof(AdamState));
    free(best_adam);
    free(gradW);
}

void compute_ngrams(double alpha) {
    double uni_counts[CLASSES] = {0};
    double bi_counts[CLASSES][CLASSES] = {0};
    double bi_totals[CLASSES] = {0};
    double (*tri_counts)[CLASSES][CLASSES] = malloc(CLASSES * CLASSES * CLASSES * sizeof(double));
    double (*tri_totals)[CLASSES] = malloc(CLASSES * CLASSES * sizeof(double));
    memset(tri_counts, 0, CLASSES * CLASSES * CLASSES * sizeof(double));
    memset(tri_totals, 0, CLASSES * CLASSES * sizeof(double));

    trigram_probs = malloc(CLASSES * CLASSES * CLASSES * sizeof(double));
    trigram_logits = malloc(CLASSES * CLASSES * CLASSES * sizeof(float));
    
    double uni_total = 0;
    for (int i = 0; i < train_size; i++) {
        uint8_t t = target_train[i];
        uint8_t c1 = ctx_train[i];
        uint8_t c2 = ctx2_train[i];
        uni_counts[t]++; uni_total++;
        bi_counts[c1][t]++; bi_totals[c1]++;
        tri_counts[c2][c1][t]++; tri_totals[c2][c1]++;
    }
    for (int i = 0; i < CLASSES; i++) {
        unigram_probs[i] = (uni_counts[i] + alpha) / (uni_total + CLASSES * alpha);
        for (int j = 0; j < CLASSES; j++) {
            bigram_probs[i][j] = (bi_counts[i][j] + alpha) / (bi_totals[i] + CLASSES * alpha);
            bigram_logits[i][j] = (float)log2(bigram_probs[i][j]) * 0.6931471805599453f;
            for (int k = 0; k < CLASSES; k++) {
                trigram_probs[i][j][k] = (tri_counts[i][j][k] + alpha) / (tri_totals[i][j] + CLASSES * alpha);
                trigram_logits[i][j][k] = (float)log2(trigram_probs[i][j][k]) * 0.6931471805599453f;
            }
        }
    }
    free(tri_counts);
    free(tri_totals);
}

// -------------------------------------------------------------
// Engine Implementations (L0)
// -------------------------------------------------------------

__m256i m4_cb[256];
uint8_t hist[256];
SiliconV0 v0;

void l0_init(int seed) {
    srand(seed);
    for(int b = 0; b < 256; b++) {
        uint8_t vec[32];
        for(int i = 0; i < 32; i++) vec[i] = (rand() % 2) ? 255 : 0;
        m4_cb[b] = _mm256_loadu_si256((__m256i*)vec);
    }
    silicon_v0_init(&v0, seed);
}

void l0_extract(int idx, uint8_t byte, float* out) {
    int history_tokens = 4; // Fissato a T=4 da Phase 13
    
    // M4 (32D)
    if (idx == 0) memset(hist, 0, sizeof(hist));
    memmove(hist + 1, hist, 255);
    hist[0] = byte;
    memset(out, 0, 32 * sizeof(float));
    for (int t = 0; t < history_tokens; t++) {
        uint8_t vec[32];
        _mm256_storeu_si256((__m256i*)vec, m4_cb[hist[t]]);
        for (int i = 0; i < 32; i++) out[i] += vec[i];
    }
    
    // V0 (32D)
    if (idx == 0) silicon_v0_reset(&v0);
    silicon_v0_tick_t(&v0, byte, history_tokens);
    double d_out[32];
    silicon_v0_extract_32d(&v0, d_out);
    for (int i = 0; i < 32; i++) out[32 + i] = (float)d_out[i];
}

// -------------------------------------------------------------
// L1 Bottleneck Chunking (Exponential Decay)
// -------------------------------------------------------------
float l1_state[128];
float current_chunk_mean[64];
uint64_t total_l0_cyc = 0;
uint64_t total_l1_cyc = 0;

void l1_reset() {
    memset(l1_state, 0, sizeof(l1_state));
    memset(current_chunk_mean, 0, sizeof(current_chunk_mean));
    total_l0_cyc = 0;
    total_l1_cyc = 0;
}

void extract_ema(int global_idx, uint8_t byte, float* out, int chunk_size, float decay) {
    uint64_t s_l0 = __rdtsc();
    // 1. Estrai L0 per questo byte (64D)
    l0_extract(global_idx, byte, out);
    uint64_t e_l0 = __rdtsc();
    total_l0_cyc += (e_l0 - s_l0);
    
    uint64_t s_l1 = __rdtsc();
    if (chunk_size > 0) {
        // Accumula running mean
        for(int i = 0; i < 64; i++) {
            current_chunk_mean[i] += out[i];
        }
        
        // Output features: [L0_64, l1_state_128]
        memcpy(out + 64, l1_state, 128 * sizeof(float));
        
        // If we close the chunk, update l1_state
        if ((global_idx + 1) % chunk_size == 0) {
            float mean_last[128];
            for(int i=0; i<64; i++) {
                mean_last[i] = current_chunk_mean[i] / (float)chunk_size; // Mean
                mean_last[64 + i] = out[i];                     // Last
            }
            
            for(int i=0; i<128; i++) {
                l1_state[i] = (l1_state[i] * decay) + mean_last[i];
            }
            
            memset(current_chunk_mean, 0, sizeof(current_chunk_mean));
        }
    }
    uint64_t e_l1 = __rdtsc();
    total_l1_cyc += (e_l1 - s_l1);
}

double run_eval(const char* name, int chunk_size, float decay, int use_trigram, int print, double base_bpb) {
    l1_reset();
    l0_init(42);
    
    int num_features = (chunk_size > 0) ? (64 + 128) : (chunk_size == 0 ? 64 : 0);
    if (chunk_size == -1) num_features = 0; // Pure Baseline
    
    if (num_features > 0) {
        for (int i = 0; i < train_size; i++) extract_ema(i, data[i+1], &features_train[i * num_features], chunk_size, decay);
        l1_reset(); l0_init(42);
        for (int i = 0; i < val_size; i++) extract_ema(i, data[train_size + i + 1], &features_val[i * num_features], chunk_size, decay);
        l1_reset(); l0_init(42);
        for (int i = 0; i < test_size; i++) extract_ema(i, data[train_size + val_size + i + 1], &features_test[i * num_features], chunk_size, decay);
        normalize_features(num_features);
    }
    
    double best_res_bpb = 1e9;
    uint64_t best_fwd_cyc = 0;
    float lrs[] = {0.003f, 0.001f, 0.0003f, 0.0001f};
    
    for (int l = 0; l < 4; l++) {
        AdamState* adam = (AdamState*)malloc(sizeof(AdamState));
        adam_init(adam);
        if (num_features > 0 || base_bpb == 0) {
            train_logistic_regression(adam, num_features, 10, 256, lrs[l], 1, use_trigram);
        }
        double bpb;
        uint64_t fwd_cyc = 0;
        _evaluate_model(adam, num_features, 1, use_trigram, &bpb, features_test, ctx_test, ctx2_test, target_test, test_size, &fwd_cyc);
        if (bpb < best_res_bpb) {
            best_res_bpb = bpb;
            best_fwd_cyc = fwd_cyc;
        }
        free(adam);
    }
    
    if (print) {
        double dBPB = (base_bpb > 0) ? (base_bpb - best_res_bpb) : 0;
        double l0_c_p_b = (double)total_l0_cyc / test_size;
        double l1_c_p_b = (double)total_l1_cyc / test_size;
        double fwd_c_p_b = (double)best_fwd_cyc / test_size;
        double total_c = l0_c_p_b + l1_c_p_b + fwd_c_p_b;
        double dbpb_1kc = (dBPB > 0) ? (dBPB / (total_c / 1000.0)) : 0;
        if (chunk_size == -1) dbpb_1kc = 0;
        
        printf("| %-22s | %3d | %5.3f | %4d | %.4f | %7.1f | %7.1f | %7.1f | %9.5f |\n", 
            name, chunk_size, decay, num_features, best_res_bpb, l0_c_p_b, l1_c_p_b, fwd_c_p_b, dbpb_1kc);
        fflush(stdout);
    }
    return best_res_bpb;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <dataset_path>\n", argv[0]);
        return 1;
    }
    const char* filename = argv[1];
    printf("Starting Phase 14C Shuffled Control Test...\n");
    printf("Dataset: %s\n", filename);
    
    FILE* f = fopen(filename, "rb");
    if (!f) return 1;
    data_size = fread(data, 1, MAX_SAMPLES, f);
    fclose(f);
    
    // --- GLOBAL BYTE SHUFFLE ---
    printf("Applying global byte shuffle (seed=42)...\n");
    srand(42);
    for (int i = data_size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        uint8_t t = data[i];
        data[i] = data[j];
        data[j] = t;
    }
    printf("Shuffle complete.\n");
    
    train_size = data_size / 2;
    val_size = data_size / 4;
    test_size = data_size - train_size - val_size - 2; 
    
    for (int i = 0; i < train_size; i++) {
        target_train[i] = data[i+2];
        ctx_train[i] = data[i+1];
        ctx2_train[i] = data[i];
    }
    for (int i = 0; i < val_size; i++) {
        target_val[i] = data[train_size + i + 2];
        ctx_val[i] = data[train_size + i + 1];
        ctx2_val[i] = data[train_size + i];
    }
    for (int i = 0; i < test_size; i++) {
        target_test[i] = data[train_size + val_size + i + 2];
        ctx_test[i] = data[train_size + val_size + i + 1];
        ctx2_test[i] = data[train_size + val_size + i];
    }
    
    compute_ngrams(0.1);
    
    double uni_loss = 0.0;
    for (int i = 0; i < test_size; i++) {
        uni_loss -= log2(unigram_probs[target_test[i]]);
    }
    double unigram_bpb = uni_loss / test_size;
    printf("\n=== Unigram Base (Shuffled) ===\n");
    printf("Unigram BPB: %.4f\n", unigram_bpb);
    
    features_train = (float*)malloc(train_size * MAX_FEATURES * sizeof(float));
    features_val = (float*)malloc(val_size * MAX_FEATURES * sizeof(float));
    features_test = (float*)malloc(test_size * MAX_FEATURES * sizeof(float));

    for(int use_trigram = 0; use_trigram <= 1; use_trigram++) {
        printf("\n=== %s Residual Base ===\n", use_trigram ? "Trigram" : "Bigram");
        printf("| %-22s | %3s | %5s | %4s | %6s | %7s | %7s | %7s | %9s |\n", 
            "Model", "C", "Decay", "Dim", "BPB", "L0 Cyc", "L1 Cyc", "Fwd Cyc", "dBPB/1kC");
        printf("|------------------------|-----|-------|------|--------|---------|---------|---------|-----------|\n");
        
        double base_bpb = run_eval(use_trigram ? "Trigram Base" : "Bigram Base", -1, 0, use_trigram, 1, 0);
        double l0_bpb = run_eval("L0 Only (T=4)", 0, 0, use_trigram, 1, base_bpb);
        run_eval("L0+L1(C=4, d=0.500)", 4, 0.500f, use_trigram, 1, base_bpb);
    }
    
    return 0;
}
