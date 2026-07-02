#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#ifdef _WIN32
#include <windows.h>
#endif

// ============================================================================
// TIMING UTILS
// ============================================================================
static inline uint64_t get_cycles() {
    unsigned int dummy;
    return __rdtscp(&dummy);
}

// ============================================================================
// DATASET GENERATORS
// ============================================================================

uint8_t* create_block_shuffled(uint8_t* src, int len, int block_size) {
    uint8_t* dst = malloc(len);
    int num_blocks = len / block_size;
    int* perm = malloc(num_blocks * sizeof(int));
    for(int i=0; i<num_blocks; i++) perm[i] = i;
    for(int i=num_blocks-1; i>0; i--) {
        int j = rand() % (i+1);
        int temp = perm[i];
        perm[i] = perm[j];
        perm[j] = temp;
    }
    for(int i=0; i<num_blocks; i++) {
        memcpy(dst + i*block_size, src + perm[i]*block_size, block_size);
    }
    int remainder = len % block_size;
    if (remainder > 0) memcpy(dst + num_blocks*block_size, src + num_blocks*block_size, remainder);
    free(perm);
    return dst;
}

// ============================================================================
// CHOLESKY RIDGE REGRESSION
// ============================================================================

int cholesky_decompose(const double* A, int n, double* L) {
    memset(L, 0, n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0;
            for (int k = 0; k < j; k++) sum += L[i * n + k] * L[j * n + k];
            if (i == j) {
                double diag = A[i * n + i] - sum;
                if (diag <= 0.0) return 0;
                L[i * n + j] = sqrt(diag);
            } else {
                L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
            }
        }
    }
    return 1;
}

void cholesky_solve(const double* L, int n, const double* b, double* x) {
    double* y = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int k = 0; k < i; k++) sum += L[i * n + k] * y[k];
        y[i] = (b[i] - sum) / L[i * n + i];
    }
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        for (int k = i + 1; k < n; k++) sum += L[k * n + i] * x[k];
        x[i] = (y[i] - sum) / L[i * n + i];
    }
    free(y);
}

typedef struct {
    double* X;
    double* y;
    int N;
    int num_features;
} Dataset;

Dataset* create_dataset(int N, int num_features) {
    Dataset* d = (Dataset*)malloc(sizeof(Dataset));
    d->N = N;
    d->num_features = num_features;
    d->X = (double*)calloc(N * num_features, sizeof(double));
    d->y = (double*)calloc(N, sizeof(double));
    return d;
}

void free_dataset(Dataset* d) {
    free(d->X);
    free(d->y);
    free(d);
}

double fit_ridge(Dataset* train_data, double* w_out, double* b_out) {
    int n = train_data->num_features;
    int N = train_data->N;
    
    double* x_mean = (double*)calloc(n, sizeof(double));
    double y_mean = 0;
    
    for(int i=0; i<N; i++) {
        y_mean += train_data->y[i];
        for(int j=0; j<n; j++) x_mean[j] += train_data->X[i*n + j];
    }
    y_mean /= N;
    for(int j=0; j<n; j++) x_mean[j] /= N;
    
    double* XtX = (double*)calloc(n * n, sizeof(double));
    double* Xty = (double*)calloc(n, sizeof(double));
    
    for(int i=0; i<N; i++) {
        double y_c = train_data->y[i] - y_mean;
        for(int j=0; j<n; j++) {
            double xj_c = train_data->X[i*n + j] - x_mean[j];
            Xty[j] += xj_c * y_c;
            for(int k=0; k<=j; k++) {
                double xk_c = train_data->X[i*n + k] - x_mean[k];
                XtX[j*n + k] += xj_c * xk_c;
            }
        }
    }
    for(int j=0; j<n; j++) {
        for(int k=j+1; k<n; k++) XtX[j*n + k] = XtX[k*n + j];
    }
    
    double* L = (double*)calloc(n * n, sizeof(double));
    double* XtX_reg = (double*)calloc(n * n, sizeof(double));
    double lambda = 1e-4;
    int success = 0;
    
    for(int attempt=0; attempt<15; attempt++) {
        memcpy(XtX_reg, XtX, n*n*sizeof(double));
        for(int j=0; j<n; j++) XtX_reg[j*n + j] += lambda * N;
        if (cholesky_decompose(XtX_reg, n, L)) { success = 1; break; }
        lambda *= 10.0;
    }
    
    if (!success) {
        memset(w_out, 0, n * sizeof(double));
        *b_out = y_mean;
    } else {
        cholesky_solve(L, n, Xty, w_out);
        *b_out = y_mean;
        for(int j=0; j<n; j++) *b_out -= w_out[j] * x_mean[j];
    }
    
    free(x_mean); free(XtX); free(Xty); free(L); free(XtX_reg);
    return lambda;
}

double predict(double* features, double* w, double b, int n) {
    double score = b;
    for(int i=0; i<n; i++) score += features[i] * w[i];
    return score;
}

// ============================================================================
// HARDWARE TEMPLATES
// ============================================================================

#define DECLARE_ENGINE(GRID_VECTORS) \
typedef struct { \
    __m256i state[GRID_VECTORS]; \
    __m256i m4_buf[256]; \
    int m4_head; \
    int grid_vectors; \
} SiliconEngine_##GRID_VECTORS; \
\
static inline void engine_init_##GRID_VECTORS(SiliconEngine_##GRID_VECTORS* e) { \
    memset(e->state, 0, sizeof(e->state)); \
    memset(e->m4_buf, 0, sizeof(e->m4_buf)); \
    e->m4_head = 0; \
    e->grid_vectors = GRID_VECTORS; \
} \
\
static inline void engine_tick_##GRID_VECTORS(SiliconEngine_##GRID_VECTORS* e, uint8_t input_byte, int t3_tokens) { \
    __m256i m4_mem = _mm256_set1_epi8(input_byte); \
    e->m4_buf[e->m4_head] = m4_mem; \
    e->m4_head = (e->m4_head + 1) % 256; \
    int blocks = GRID_VECTORS; \
    int spacing = blocks / t3_tokens; \
    if (spacing == 0) spacing = 1; \
    for(int slot=0; slot<t3_tokens; slot++) { \
        int hist_idx = (e->m4_head - 1 - slot + 256) % 256; \
        __m256i t = e->m4_buf[hist_idx]; \
        int idx = slot * spacing; \
        if (idx < blocks) e->state[idx] = _mm256_adds_epu8(e->state[idx], t); \
    } \
    __m256i const_128 = _mm256_set1_epi8(-128); \
    __m256i zero = _mm256_setzero_si256(); \
    __m256i mask_7F = _mm256_set1_epi8(0x7F); \
    _Pragma("GCC unroll 4") \
    for(int i=0; i<blocks; i++) { \
        e->state[i] = _mm256_and_si256(_mm256_srli_epi16(e->state[i], 1), mask_7F); \
    } \
    __m256i new_state[GRID_VECTORS]; \
    for(int w=0; w<4; w++) { \
        new_state[0] = e->state[0]; \
        new_state[blocks-1] = e->state[blocks-1]; \
        _Pragma("GCC unroll 4") \
        for (int i=1; i<blocks-1; i++) { \
            __m256i L = e->state[i-1]; \
            __m256i C = e->state[i]; \
            __m256i R = e->state[i+1]; \
            __m256i r0 = _mm256_adds_epu8(_mm256_avg_epu8(L, R), _mm256_subs_epu8(C, const_128)); \
            __m256i r1 = _mm256_adds_epu8(_mm256_subs_epu8(L, C), R); \
            __m256i l_half = _mm256_avg_epu8(L, zero); \
            __m256i c_half = _mm256_avg_epu8(C, zero); \
            __m256i r_half = _mm256_avg_epu8(R, zero); \
            __m256i r2 = _mm256_adds_epu8(l_half, _mm256_adds_epu8(r_half, c_half)); \
            __m256i r3 = _mm256_subs_epu8(_mm256_adds_epu8(L, R), C); \
            __m256i m0 = _mm256_set1_epi8(0xAA); \
            __m256i m1 = _mm256_set1_epi8(0xCC); \
            __m256i sel01 = _mm256_blendv_epi8(r0, r1, m0); \
            __m256i sel23 = _mm256_blendv_epi8(r2, r3, m0); \
            new_state[i] = _mm256_blendv_epi8(sel01, sel23, m1); \
        } \
        memcpy(e->state, new_state, blocks * sizeof(__m256i)); \
    } \
} \
static inline void extract_features_##GRID_VECTORS(SiliconEngine_##GRID_VECTORS* e, double* f_out, int channels) { \
    int32_t wave_sums[128] = {0}; \
    int blocks_per_channel = e->grid_vectors / channels; \
    if (blocks_per_channel == 0) blocks_per_channel = 1; \
    for (int k=0; k<channels; k++) { \
        for (int i=0; i<blocks_per_channel; i++) { \
            int block_idx = k * blocks_per_channel + i; \
            if (block_idx >= e->grid_vectors) break; \
            uint8_t bytes[32]; \
            _mm256_storeu_si256((__m256i*)bytes, e->state[block_idx]); \
            for(int eng=0; eng<32; eng++) wave_sums[k] += bytes[eng]; \
        } \
    } \
    for(int k=0; k<channels; k++) f_out[k] = (double)wave_sums[k]; \
    for(int i=0; i<16; i++) { \
        int idx = (e->m4_head - 1 - i + 256) % 256; \
        uint8_t bytes[32]; \
        _mm256_storeu_si256((__m256i*)bytes, e->m4_buf[idx]); \
        f_out[channels + i] = (double)bytes[0]; \
    } \
}

DECLARE_ENGINE(128)
DECLARE_ENGINE(256)
DECLARE_ENGINE(512)
DECLARE_ENGINE(1024)

// ============================================================================
// EVALUATION CONTEXT
// ============================================================================

typedef struct {
    Dataset* train_bits[8];
    Dataset* val_bits[8];
    int WARMUP;
    int N_train;
    int N_val;
} SweepContext;

SweepContext* create_sweep_context(int N_train, int N_val) {
    SweepContext* ctx = malloc(sizeof(SweepContext));
    ctx->WARMUP = 2000;
    ctx->N_train = N_train;
    ctx->N_val = N_val;
    for(int b=0; b<8; b++) {
        ctx->train_bits[b] = create_dataset(N_train, 128 + 16); // max features
        ctx->val_bits[b] = create_dataset(N_val, 128 + 16);
    }
    return ctx;
}

void free_sweep_context(SweepContext* ctx) {
    for(int b=0; b<8; b++) {
        free_dataset(ctx->train_bits[b]);
        free_dataset(ctx->val_bits[b]);
    }
    free(ctx);
}

// Global baseline variables
double baseline_unigram_bpb = 0;
double baseline_m4_bpb = 0;

typedef struct {
    double mean_acc;
    double std_acc;
    double mean_bpb;
} MetricResult;

void eval_and_report(SweepContext* ctx, int n_features, int feature_offset, const char* label, double cycles_per_byte, MetricResult* res_out) {
    double w[8][128+16], b[8];
    uint64_t t_ridge_start = get_cycles();
    
    for(int bit=0; bit<8; bit++) {
        Dataset* ds = create_dataset(ctx->N_train, n_features);
        for(int i=0; i<ctx->N_train; i++) {
            for(int j=0; j<n_features; j++) ds->X[i*n_features + j] = ctx->train_bits[bit]->X[i*(128+16) + feature_offset + j];
            ds->y[i] = ctx->train_bits[bit]->y[i];
        }
        fit_ridge(ds, w[bit], &b[bit]);
        free_dataset(ds);
    }
    uint64_t t_ridge_end = get_cycles();
    double ridge_ms = (t_ridge_end - t_ridge_start) / 3000000.0; // Assume ~3GHz

    int correct_bits = 0;
    double brier_sum = 0;
    double bpb_sum = 0;
    
    for(int i=0; i<ctx->N_val; i++) {
        for(int bit=0; bit<8; bit++) {
            double* f = &ctx->val_bits[bit]->X[i * (128+16) + feature_offset];
            double score = predict(f, w[bit], b[bit], n_features);
            
            // Linear Probability Model: Ridge predicts E[y] directly since targets are 0/1.
            double prob = score;
            if (prob < 1e-4) prob = 1e-4;
            if (prob > 1.0 - 1e-4) prob = 1.0 - 1e-4;
            
            int target = (int)ctx->val_bits[bit]->y[i];
            int pred = (score > 0.5) ? 1 : 0;
            if (pred == target) correct_bits++;
            double p_target = target ? prob : (1.0 - prob);
            brier_sum += (prob - target)*(prob - target);
            bpb_sum += -log2(p_target);
        }
    }
    
    double bit_acc = 100.0 * correct_bits / (ctx->N_val * 8.0);
    double bpb = bpb_sum / ctx->N_val;
    
    if (res_out) {
        res_out->mean_acc = bit_acc;
        res_out->mean_bpb = bpb;
    }
    
    if (label) {
        double gain_m4 = (baseline_m4_bpb - bpb) / (cycles_per_byte / 1000.0);
        double gain_uni = (baseline_unigram_bpb - bpb) / (cycles_per_byte / 1000.0);
        if (strcmp(label, "M4-only") == 0) {
            baseline_m4_bpb = bpb;
            printf("  %-15s | Acc: %5.2f%% | BPB: %5.2f | C/B: %4.0f | Gain vs Uni/1k: %+5.2f\n", 
                   label, bit_acc, bpb, cycles_per_byte, gain_uni);
        } else {
            printf("  %-15s | Acc: %5.2f%% | BPB: %5.2f | C/B: %4.0f | Gain vs M4/1k: %+5.2f\n", 
                   label, bit_acc, bpb, cycles_per_byte, gain_m4);
        }
    }
}

// Macro to generate sweep combinations
#define RUN_SWEEP(GRID) \
{ \
    SiliconEngine_##GRID e; \
    engine_init_##GRID(&e); \
    uint64_t t_start = get_cycles(); \
    for(int t=0; t < ctx->WARMUP + ctx->N_train + ctx->N_val - 1; t++) { \
        engine_tick_##GRID(&e, text[t], t3_tokens); \
        if (t >= ctx->WARMUP) { \
            int idx = t - ctx->WARMUP; \
            Dataset** dst_arr = (idx < ctx->N_train) ? ctx->train_bits : ctx->val_bits; \
            int offset = (idx < ctx->N_train) ? idx : idx - ctx->N_train; \
            double features[128+16] = {0}; \
            extract_features_##GRID(&e, features, channels); \
            for(int bit=0; bit<8; bit++) { \
                for(int i=0; i<channels+16; i++) dst_arr[bit]->X[offset * (128+16) + i] = features[i]; \
                dst_arr[bit]->y[offset] = (text[t+1] >> bit) & 1; \
            } \
        } \
    } \
    uint64_t t_end = get_cycles(); \
    double cycles_per_byte = (double)(t_end - t_start) / (ctx->WARMUP + ctx->N_train + ctx->N_val); \
    char label[64]; \
    sprintf(label, "G%d_T%d_C%d", GRID, t3_tokens, channels); \
    eval_and_report(ctx, channels, 0, label, cycles_per_byte, NULL); \
}

void compute_baselines(uint8_t* text, SweepContext* ctx) {
    // Unigram
    int byte_counts[256] = {0};
    for(int t=ctx->WARMUP; t < ctx->WARMUP + ctx->N_train; t++) byte_counts[text[t+1]]++;
    double unigram_bpb = 0;
    int unigram_correct = 0;
    for(int t=ctx->WARMUP + ctx->N_train; t < ctx->WARMUP + ctx->N_train + ctx->N_val - 1; t++) {
        uint8_t target = text[t+1];
        for(int bit=0; bit<8; bit++) {
            int bit_target = (target >> bit) & 1;
            int count_1 = 0;
            for(int c=0; c<256; c++) if ((c >> bit) & 1) count_1 += byte_counts[c];
            double p1 = (double)count_1 / ctx->N_train;
            if (p1 > 0.5 == bit_target) unigram_correct++;
            double p_target = bit_target ? p1 : (1.0 - p1);
            if (p_target < 1e-7) p_target = 1e-7;
            unigram_bpb += -log2(p_target);
        }
    }
    baseline_unigram_bpb = unigram_bpb / ctx->N_val;
    printf("  %-15s | Acc: %5.2f%% | BPB: %5.2f | C/B:    0 | Gain vs Uni/1k:  0.00\n", 
           "Unigram", 100.0 * unigram_correct / (ctx->N_val * 8.0), baseline_unigram_bpb);
           
    // M4-only
    // Run G128 just to extract M4 features efficiently
    SiliconEngine_128 e;
    engine_init_128(&e);
    uint64_t t_start = get_cycles();
    for(int t=0; t < ctx->WARMUP + ctx->N_train + ctx->N_val - 1; t++) {
        engine_tick_128(&e, text[t], 0); // No T3 needed for pure M4
        if (t >= ctx->WARMUP) {
            int idx = t - ctx->WARMUP;
            Dataset** dst_arr = (idx < ctx->N_train) ? ctx->train_bits : ctx->val_bits;
            int offset = (idx < ctx->N_train) ? idx : idx - ctx->N_train;
            double features[128+16] = {0};
            extract_features_128(&e, features, 0); // extracts only M4 at feature[0..15]
            for(int bit=0; bit<8; bit++) {
                for(int i=0; i<16; i++) dst_arr[bit]->X[offset * (128+16) + 128 + i] = features[i];
                dst_arr[bit]->y[offset] = (text[t+1] >> bit) & 1;
            }
        }
    }
    uint64_t t_end = get_cycles();
    double cycles_per_byte = (double)(t_end - t_start) / (ctx->WARMUP + ctx->N_train + ctx->N_val);
    eval_and_report(ctx, 16, 128, "M4-only", cycles_per_byte, NULL);
}

// ============================================================================
// PHASE 8B RUNNER
// ============================================================================

void run_phase_8b_dataset(uint8_t* text, long fsize, const char* name, int is_shuffled) {
    printf("\n=== PHASE 8B: %s ===\n", name);
    int N_train = 5000;
    int N_val = 2000;
    int NUM_SPLITS = 5;
    
    if (fsize < (N_train + N_val) * NUM_SPLITS + 2000) {
        printf("  [Warning] File too small for 5 separate splits, overlapping offsets.\n");
    }
    
    double bigram_acc_sum = 0;
    double unigram_acc_sum = 0;
    double m4_acc_sum = 0;
    double wave_acc_sum = 0;
    
    double m4_accs[5], wave_accs[5], uni_accs[5], bi_accs[5];
    
    for(int s=0; s<NUM_SPLITS; s++) {
        int warmup = 2000 + s * (N_train + N_val);
        if (warmup + N_train + N_val >= fsize) warmup = 2000 + (s * 500); // overlap
        if (warmup + N_train + N_val >= fsize) warmup = 0;
        
        SweepContext* ctx = create_sweep_context(N_train, N_val);
        ctx->WARMUP = warmup;
        
        // 1. Bigram
        double alpha = 0.1;
        double bigram_matrix[256][256] = {0};
        double bigram_sums[256] = {0};
        for(int i=0; i<256; i++) {
            for(int j=0; j<256; j++) bigram_matrix[i][j] = alpha;
            bigram_sums[i] = 256 * alpha;
        }
        for(int t=warmup; t < warmup + N_train - 1; t++) {
            bigram_matrix[text[t]][text[t+1]] += 1.0;
            bigram_sums[text[t]] += 1.0;
        }
        int bi_correct = 0;
        for(int t=warmup + N_train; t < warmup + N_train + N_val - 1; t++) {
            uint8_t current = text[t];
            uint8_t target = text[t+1];
            int best_byte = 0; double best_p = -1;
            for(int c=0; c<256; c++) {
                if (bigram_matrix[current][c] > best_p) { best_p = bigram_matrix[current][c]; best_byte = c; }
            }
            for(int bit=0; bit<8; bit++) {
                if (((best_byte >> bit) & 1) == ((target >> bit) & 1)) bi_correct++;
            }
        }
        bi_accs[s] = 100.0 * bi_correct / (N_val * 8.0);
        bigram_acc_sum += bi_accs[s];
        
        // 2. Unigram
        int byte_counts[256] = {0};
        for(int t=warmup; t < warmup + N_train; t++) byte_counts[text[t+1]]++;
        int uni_correct = 0;
        for(int t=warmup + N_train; t < warmup + N_train + N_val - 1; t++) {
            uint8_t target = text[t+1];
            for(int bit=0; bit<8; bit++) {
                int bit_target = (target >> bit) & 1;
                int count_1 = 0;
                for(int c=0; c<256; c++) if ((c >> bit) & 1) count_1 += byte_counts[c];
                double p1 = (double)count_1 / N_train;
                if ((p1 > 0.5) == bit_target) uni_correct++;
            }
        }
        uni_accs[s] = 100.0 * uni_correct / (N_val * 8.0);
        unigram_acc_sum += uni_accs[s];
        
        // 3. M4 Extraction
        SiliconEngine_128 e;
        engine_init_128(&e);
        for(int t=0; t < warmup + N_train + N_val - 1; t++) {
            engine_tick_128(&e, text[t], 16); // G128_T16_C16 sweet spot
            if (t >= warmup) {
                int idx = t - warmup;
                Dataset** dst_arr = (idx < N_train) ? ctx->train_bits : ctx->val_bits;
                int offset = (idx < N_train) ? idx : idx - N_train;
                double features[128+16] = {0};
                extract_features_128(&e, features, 16);
                for(int bit=0; bit<8; bit++) {
                    for(int i=0; i<16+16; i++) dst_arr[bit]->X[offset * (128+16) + i] = features[i];
                    dst_arr[bit]->y[offset] = (text[t+1] >> bit) & 1;
                }
            }
        }
        
        MetricResult m4_res;
        eval_and_report(ctx, 16, 16, NULL, 0, &m4_res); // features[16..31] are M4
        m4_accs[s] = m4_res.mean_acc;
        m4_acc_sum += m4_accs[s];
        
        MetricResult wave_res;
        eval_and_report(ctx, 16, 0, NULL, 0, &wave_res); // features[0..15] are Wave
        wave_accs[s] = wave_res.mean_acc;
        wave_acc_sum += wave_accs[s];
        
        free_sweep_context(ctx);
    }
    
    double bi_mean = bigram_acc_sum / NUM_SPLITS;
    double uni_mean = unigram_acc_sum / NUM_SPLITS;
    double m4_mean = m4_acc_sum / NUM_SPLITS;
    double wave_mean = wave_acc_sum / NUM_SPLITS;
    
    double bi_std=0, uni_std=0, m4_std=0, wave_std=0;
    for(int s=0; s<NUM_SPLITS; s++) {
        bi_std += (bi_accs[s]-bi_mean)*(bi_accs[s]-bi_mean);
        uni_std += (uni_accs[s]-uni_mean)*(uni_accs[s]-uni_mean);
        m4_std += (m4_accs[s]-m4_mean)*(m4_accs[s]-m4_mean);
        wave_std += (wave_accs[s]-wave_mean)*(wave_accs[s]-wave_mean);
    }
    bi_std = sqrt(bi_std/NUM_SPLITS);
    uni_std = sqrt(uni_std/NUM_SPLITS);
    m4_std = sqrt(m4_std/NUM_SPLITS);
    wave_std = sqrt(wave_std/NUM_SPLITS);
    
    printf("  Bigram (Markov-1) : %5.2f%% ± %4.2f\n", bi_mean, bi_std);
    printf("  Unigram Marginal  : %5.2f%% ± %4.2f\n", uni_mean, uni_std);
    printf("  M4-only           : %5.2f%% ± %4.2f\n", m4_mean, m4_std);
    printf("  Wave (G128_T16)   : %5.2f%% ± %4.2f\n", wave_mean, wave_std);
    printf("  Wave Gain vs M4   : %+5.2f%%\n", wave_mean - m4_mean);
}

void phase_8b_sweep() {
    printf("\n=======================================================================\n");
    printf("PHASE 8B: GENERALIZATION SWEEP\n");
    printf("=======================================================================\n");
    
    const char* files[] = {"benchmark6_text.c", "DOCS/architecture_decisions.md", "data/promessi_sposi.txt"};
    
    for(int i=0; i<3; i++) {
        FILE* f = fopen(files[i], "rb");
        if(!f) { printf("Error opening %s\n", files[i]); continue; }
        fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
        uint8_t* text = (uint8_t*)malloc(fsize);
        fread(text, 1, fsize, f); fclose(f);
        
        run_phase_8b_dataset(text, fsize, files[i], 0);
        
        if (strcmp(files[i], "data/promessi_sposi.txt") == 0) {
            uint8_t* shuffled = create_block_shuffled(text, fsize, 16);
            run_phase_8b_dataset(shuffled, fsize, "data/promessi_sposi.txt (SHUFFLED 16-byte blocks)", 1);
            free(shuffled);
        }
        free(text);
    }
}

void phase_8a_sweep(uint8_t* text, int N_train, int N_val) {
    printf("\n=== PHASE 8A: HARDWARE GEOMETRY SWEEP ===\n");
    SweepContext* ctx = create_sweep_context(N_train, N_val);
    
    compute_baselines(text, ctx);
    
    int t3_configs[] = {8, 16, 32, 64};
    int ch_configs[] = {16, 32, 64};
    
    for(int i_t3=0; i_t3<4; i_t3++) {
        for(int i_ch=0; i_ch<3; i_ch++) {
            int t3_tokens = t3_configs[i_t3];
            int channels = ch_configs[i_ch];
            
            RUN_SWEEP(128);
            RUN_SWEEP(256);
            RUN_SWEEP(512);
            RUN_SWEEP(1024);
        }
    }
    
    free_sweep_context(ctx);
}

int main() {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 1);
#endif
    
    // Uncomment to run 8A
    // const char* filename = "benchmark6_text.c";
    // FILE* f = fopen(filename, "rb");
    // if(f) {
    //     fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    //     uint8_t* text = (uint8_t*)malloc(fsize);
    //     fread(text, 1, fsize, f); fclose(f);
    //     phase_8a_sweep(text, 5000, 2000);
    //     free(text);
    // }
    
    phase_8b_sweep();
    
    return 0;
}
