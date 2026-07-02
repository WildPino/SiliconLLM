#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../src/silicon_v0.h"

#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

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

#define MAX_SAMPLES 500000
#define MAX_FEATURES 512
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

int train_size = 0;
int val_size = 0;
int test_size = 0;

double unigram_probs[CLASSES];
double bigram_probs[CLASSES][CLASSES];
float bigram_logits[CLASSES][CLASSES];

double global_bigram_bpb = 0.0;

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

void _evaluate_model(AdamState* model, int num_features, int use_residual, double* out_bpb, double* out_acc, float* features, uint8_t* ctx, uint8_t* target, int size, double* out_cycles) {
    double total_loss = 0.0;
    int correct = 0;
    
    uint64_t start_cycles = __rdtsc();
    
    for (int i = 0; i < size; i++) {
        float logits[CLASSES];
        float max_l = -1e9;
        
        for (int c = 0; c < CLASSES; c++) {
            logits[c] = model->B[c];
            if (use_residual) {
                logits[c] += bigram_logits[ctx[i]][c];
            }
            logits[c] += dot_product_simd(model->W[c], &features[i * num_features], num_features);
            if (logits[c] > max_l) max_l = logits[c];
        }
        
        float sum_e = 0.0f;
        int best_c = 0;
        float best_l = -1e9;
        
        for (int c = 0; c < CLASSES; c++) {
            float p = expf(logits[c] - max_l);
            sum_e += p;
            if (logits[c] > best_l) {
                best_l = logits[c];
                best_c = c;
            }
        }
        
        float prob = expf(logits[target[i]] - max_l) / sum_e;
        total_loss -= log2(prob);
        if (best_c == target[i]) correct++;
    }
    
    uint64_t end_cycles = __rdtsc();
    
    *out_bpb = total_loss / size;
    *out_acc = (double)correct / size * 100.0;
    if (out_cycles) {
        *out_cycles = (double)(end_cycles - start_cycles) / size;
    }
}

void evaluate_model_on_val(AdamState* model, int num_features, int use_residual, double* out_bpb, double* out_acc) {
    _evaluate_model(model, num_features, use_residual, out_bpb, out_acc, features_val, ctx_val, target_val, val_size, NULL);
}

void evaluate_model_on_test(AdamState* model, int num_features, int use_residual, double* out_bpb, double* out_acc, double* out_cycles) {
    _evaluate_model(model, num_features, use_residual, out_bpb, out_acc, features_test, ctx_test, target_test, test_size, out_cycles);
}

void train_logistic_regression(AdamState* model, int num_features, int epochs, int batch_size, float lr, int use_residual) {
    float* gradW = (float*)malloc(CLASSES * MAX_FEATURES * sizeof(float));
    float gradB[CLASSES];
    float logits[CLASSES];
    float probs[CLASSES];
    
    AdamState* best_adam = (AdamState*)malloc(sizeof(AdamState));
    double best_bpb = 1e9, acc = 0;
    evaluate_model_on_val(model, num_features, use_residual, &best_bpb, &acc);
    memcpy(best_adam, model, sizeof(AdamState));
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        memset(gradW, 0, CLASSES * MAX_FEATURES * sizeof(float));
        memset(gradB, 0, sizeof(gradB));
        
        for (int i = 0; i < train_size; i++) {
            float max_l = -1e9;
            for (int c = 0; c < CLASSES; c++) {
                logits[c] = model->B[c];
                if (use_residual) {
                    logits[c] += bigram_logits[ctx_train[i]][c];
                }
                logits[c] += dot_product_simd(model->W[c], &features_train[i * num_features], num_features);
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
                grad_update_simd(&gradW[c * MAX_FEATURES], &features_train[i * num_features], err / batch_size, num_features);
            }
            
            if ((i + 1) % batch_size == 0 || (i + 1) == train_size) {
                adam_step(model, gradW, gradB, num_features, lr, 0.9f, 0.999f, 1e-8f, 1e-4f);
                memset(gradW, 0, CLASSES * MAX_FEATURES * sizeof(float));
                memset(gradB, 0, sizeof(gradB));
            }
        }
        double val_bpb, val_acc;
        evaluate_model_on_val(model, num_features, use_residual, &val_bpb, &val_acc);
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
    double uni_total = 0;
    
    for (int i = 0; i < train_size; i++) {
        uint8_t t = target_train[i];
        uint8_t c1 = ctx_train[i];
        uni_counts[t]++;
        uni_total++;
        bi_counts[c1][t]++;
        bi_totals[c1]++;
    }
    
    for (int i = 0; i < CLASSES; i++) {
        unigram_probs[i] = (uni_counts[i] + alpha) / (uni_total + CLASSES * alpha);
        for (int j = 0; j < CLASSES; j++) {
            bigram_probs[i][j] = (bi_counts[i][j] + alpha) / (bi_totals[i] + CLASSES * alpha);
            bigram_logits[i][j] = (float)log2(bigram_probs[i][j]) * 0.6931471805599453f;
        }
    }
}

void eval_ngrams(double* uni_bpb, double* bi_bpb) {
    double u_loss = 0;
    double b_loss = 0;
    
    for (int i = 0; i < test_size; i++) {
        uint8_t t = target_test[i];
        uint8_t c1 = ctx_test[i];
        
        u_loss -= log2(unigram_probs[t]);
        b_loss -= log2(bigram_probs[c1][t]);
    }
    *uni_bpb = u_loss / test_size;
    *bi_bpb = b_loss / test_size;
}

// -------------------------------------------------------------
// Engine Implementations
// -------------------------------------------------------------

void run_eval(const char* name, int num_features, void (*extract_fn)(int idx, uint8_t byte, float* out)) {
    features_train = (float*)malloc(train_size * num_features * sizeof(float));
    features_val = (float*)malloc(val_size * num_features * sizeof(float));
    features_test = (float*)malloc(test_size * num_features * sizeof(float));

    for (int i = 0; i < train_size; i++) extract_fn(i, data[i], &features_train[i * num_features]);
    for (int i = 0; i < val_size; i++) extract_fn(i + train_size, data[train_size + i], &features_val[i * num_features]);
    
    uint64_t start_ext = __rdtsc();
    for (int i = 0; i < test_size; i++) extract_fn(i + train_size + val_size, data[train_size + val_size + i], &features_test[i * num_features]);
    uint64_t end_ext = __rdtsc();
    double ext_cycles = (double)(end_ext - start_ext) / test_size;
    
    normalize_features(num_features);
    
    double best_res_bpb = 1e9, best_res_acc = 0;
    double best_fwd_cycles = 0;
    float lrs[] = {0.01f, 0.003f, 0.001f};
    
    for (int l = 0; l < 3; l++) {
        AdamState* adam = (AdamState*)malloc(sizeof(AdamState));
        adam_init(adam);
        train_logistic_regression(adam, num_features, 10, 256, lrs[l], 1);
        
        double bpb, acc, fwd_cyc;
        evaluate_model_on_test(adam, num_features, 1, &bpb, &acc, &fwd_cyc);
        if (bpb < best_res_bpb) { 
            best_res_bpb = bpb; 
            best_res_acc = acc; 
            best_fwd_cycles = fwd_cyc;
        }
        free(adam);
    }
    
    double delta_bpb = global_bigram_bpb - best_res_bpb;
    double delta_bpb_per_feat = delta_bpb / num_features;
    double total_cycles = ext_cycles + best_fwd_cycles;
    double delta_bpb_per_1000_cyc = (delta_bpb / total_cycles) * 1000.0;
    
    printf("| %-20s | %4d | %.4f | %.2f%% | %7.1f | %7.1f | %9.1f | %.6f | %.5f |\n", 
           name, num_features, best_res_bpb, best_res_acc, ext_cycles, best_fwd_cycles, total_cycles, delta_bpb_per_feat, delta_bpb_per_1000_cyc);
    fflush(stdout);

    free(features_train);
    free(features_val);
    free(features_test);
}

// M4 Simulation Codebook
__m256i m4_cb[256];
uint8_t hist[16];

void m4_extract_512(int idx, uint8_t byte, float* out) {
    if (idx == 0) memset(hist, 0, 16);
    memmove(hist + 1, hist, 15);
    hist[0] = byte;
    
    for (int t = 0; t < 16; t++) {
        uint8_t vec[32];
        _mm256_storeu_si256((__m256i*)vec, m4_cb[hist[t]]);
        for (int i = 0; i < 32; i++) out[t * 32 + i] = vec[i];
    }
}

void m4_extract_32(int idx, uint8_t byte, float* out) {
    if (idx == 0) memset(hist, 0, 16);
    memmove(hist + 1, hist, 15);
    hist[0] = byte;
    
    memset(out, 0, 32 * sizeof(float));
    for (int t = 0; t < 16; t++) {
        uint8_t vec[32];
        _mm256_storeu_si256((__m256i*)vec, m4_cb[hist[t]]);
        for (int i = 0; i < 32; i++) out[i] += vec[i];
    }
}

SiliconV0 v0;
void v0_extract_32(int idx, uint8_t byte, float* out) {
    if (idx == 0) silicon_v0_reset(&v0);
    silicon_v0_tick(&v0, byte);
    double d_out[32];
    silicon_v0_extract_32d(&v0, d_out);
    for (int i = 0; i < 32; i++) out[i] = (float)d_out[i];
}

void concat_64(int idx, uint8_t byte, float* out) {
    m4_extract_32(idx, byte, out);
    v0_extract_32(idx, byte, out + 32);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <dataset_path>\n", argv[0]);
        return 1;
    }
    
    const char* filename = argv[1];
    printf("Starting Phase 12 Hybrid Benchmark...\n");
    printf("Dataset: %s\n", filename);
    
    FILE* f = fopen(filename, "rb");
    if (!f) { printf("Failed to open file %s\n", filename); return 1; }
    data_size = fread(data, 1, MAX_SAMPLES, f);
    fclose(f);
    
    train_size = data_size / 2;
    val_size = data_size / 4;
    test_size = data_size - train_size - val_size - 1; 
    
    for (int i = 0; i < train_size; i++) {
        target_train[i] = data[i+1];
        ctx_train[i] = data[i];
    }
    for (int i = 0; i < val_size; i++) {
        target_val[i] = data[train_size + i + 1];
        ctx_val[i] = data[train_size + i];
    }
    for (int i = 0; i < test_size; i++) {
        target_test[i] = data[train_size + val_size + i + 1];
        ctx_test[i] = data[train_size + val_size + i];
    }
    
    srand(42);
    for(int b = 0; b < 256; b++) {
        uint8_t vec[32];
        for(int i = 0; i < 32; i++) vec[i] = (rand() % 2) ? 255 : 0;
        m4_cb[b] = _mm256_loadu_si256((__m256i*)vec);
    }
    silicon_v0_init(&v0, 42);
    
    compute_ngrams(0.1);
    double u_bpb, b_bpb;
    eval_ngrams(&u_bpb, &b_bpb);
    global_bigram_bpb = b_bpb;
    
    printf("\n| %-20s | %4s | %6s | %7s | %7s | %7s | %9s | %8s | %8s |\n", 
           "Model", "Dim", "BPB", "Acc", "Ext Cyc", "Fwd Cyc", "Total Cyc", "dBPB/Fea", "dBPB/1kC");
    printf("|----------------------|------|--------|---------|---------|---------|-----------|----------|----------|\n");
    printf("| %-20s | %4d | %.4f | %7s | %7s | %7s | %9s | %8s | %8s |\n", 
           "Bigram (Baseline)", 0, b_bpb, "-", "-", "-", "-", "-", "-");
    
    run_eval("M4 Pooled", 32, m4_extract_32);
    run_eval("V0 Pooled", 32, v0_extract_32);
    run_eval("Concat", 64, concat_64);
    run_eval("M4 Full", 512, m4_extract_512);
    
    return 0;
}
