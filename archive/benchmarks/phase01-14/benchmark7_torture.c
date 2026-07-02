#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

// ============================================================================
// SILICONLLM ENGINE V1
// ============================================================================

#define GRID_VECTORS 256
#define GRID_CELLS (GRID_VECTORS * 32)
#define HISTORY_SIZE 64
#define NUM_WAVE_FEATURES 32
#define NUM_M4_FEATURES 16
#define NUM_TOTAL_FEATURES (NUM_WAVE_FEATURES + NUM_M4_FEATURES)

typedef struct {
    __m256i state[GRID_VECTORS];
    __m256i m4_buf[HISTORY_SIZE];
    int m4_head;
    int grid_vectors;
    int t3_enabled;
} SiliconEngine;

void engine_init(SiliconEngine* e, int t3_enabled) {
    memset(e->state, 0, sizeof(e->state));
    memset(e->m4_buf, 0, sizeof(e->m4_buf));
    e->m4_head = 0;
    e->grid_vectors = GRID_VECTORS;
    e->t3_enabled = t3_enabled;
}

void engine_tick(SiliconEngine* e, uint8_t input_byte) {
    __m256i m4_mem = _mm256_set1_epi8(input_byte);
    e->m4_buf[e->m4_head] = m4_mem;
    e->m4_head = (e->m4_head + 1) % HISTORY_SIZE;

    __m256i t0 = e->m4_buf[(e->m4_head - 1 + HISTORY_SIZE) % HISTORY_SIZE];
    __m256i t1 = e->m4_buf[(e->m4_head - 2 + HISTORY_SIZE) % HISTORY_SIZE];
    __m256i t2 = e->m4_buf[(e->m4_head - 3 + HISTORY_SIZE) % HISTORY_SIZE];

    int blocks = GRID_VECTORS;
    if (e->t3_enabled) {
        for(int i=0; i<blocks; i+=12) {
            if (i < blocks) e->state[i] = _mm256_adds_epu8(e->state[i], t0);
            if (i+4 < blocks) e->state[i+4] = _mm256_adds_epu8(e->state[i+4], t1);
            if (i+8 < blocks) e->state[i+8] = _mm256_adds_epu8(e->state[i+8], t2);
        }
    } else {
        // Only current byte (t0) at the center
        e->state[blocks/2] = _mm256_adds_epu8(e->state[blocks/2], t0);
    }
    
    __m256i const_128 = _mm256_set1_epi8(-128);
    __m256i zero = _mm256_setzero_si256();
    __m256i mask_7F = _mm256_set1_epi8(0x7F);
    
    // Global Damping
    for(int i=0; i<blocks; i++) {
        e->state[i] = _mm256_and_si256(_mm256_srli_epi16(e->state[i], 1), mask_7F);
    }
    
    __m256i new_state[GRID_VECTORS];
    
    for(int w=0; w<4; w++) {
        new_state[0] = e->state[0];
        new_state[blocks-1] = e->state[blocks-1];
        
        for (int i=1; i<blocks-1; i++) {
            __m256i L = e->state[i-1];
            __m256i C = e->state[i];
            __m256i R = e->state[i+1];
            
            __m256i r0 = _mm256_adds_epu8(_mm256_avg_epu8(L, R), _mm256_subs_epu8(C, const_128));
            __m256i r1 = _mm256_adds_epu8(_mm256_subs_epu8(L, C), R);
            __m256i l_half = _mm256_avg_epu8(L, zero);
            __m256i c_half = _mm256_avg_epu8(C, zero);
            __m256i r_half = _mm256_avg_epu8(R, zero);
            __m256i r2 = _mm256_adds_epu8(l_half, _mm256_adds_epu8(r_half, c_half));
            __m256i r3 = _mm256_subs_epu8(_mm256_adds_epu8(L, R), C);
            
            __m256i m0 = _mm256_set1_epi8(0xAA);
            __m256i m1 = _mm256_set1_epi8(0xCC);
            __m256i sel01 = _mm256_blendv_epi8(r0, r1, m0);
            __m256i sel23 = _mm256_blendv_epi8(r2, r3, m0);
            new_state[i] = _mm256_blendv_epi8(sel01, sel23, m1);
        }
        memcpy(e->state, new_state, blocks * sizeof(__m256i));
    }
}

void extract_features(SiliconEngine* e, double* f_out) {
    int32_t wave_sums[32] = {0};
    int offset = e->grid_vectors - 256; 
    if (offset < 0) offset = 0;
    
    for(int i = 0; i < 256; i+=32) {
        for(int k=0; k<32; k++) {
            uint8_t bytes[32];
            _mm256_storeu_si256((__m256i*)bytes, e->state[offset + i + k]);
            for(int eng=0; eng<32; eng++) wave_sums[k] += bytes[eng];
        }
    }
    for(int k=0; k<32; k++) f_out[k] = (double)wave_sums[k];
    
    // 16 M4 Channels
    for(int i=0; i<16; i++) {
        int idx = (e->m4_head - 1 - i + HISTORY_SIZE) % HISTORY_SIZE;
        uint8_t bytes[32];
        _mm256_storeu_si256((__m256i*)bytes, e->m4_buf[idx]);
        f_out[32 + i] = (double)bytes[0];
    }
}

// ============================================================================
// CHOLESKY RIDGE REGRESSION
// ============================================================================

int cholesky_decompose(const double* A, int n, double* L) {
    memset(L, 0, n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0;
            for (int k = 0; k < j; k++) {
                sum += L[i * n + k] * L[j * n + k];
            }
            if (i == j) {
                double diag = A[i * n + i] - sum;
                if (diag <= 0.0) return 0; // Not positive definite
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
        for(int k=j+1; k<n; k++) {
            XtX[j*n + k] = XtX[k*n + j];
        }
    }
    
    double* L = (double*)calloc(n * n, sizeof(double));
    double* XtX_reg = (double*)calloc(n * n, sizeof(double));
    
    double lambda = 1e-4;
    int success = 0;
    
    for(int attempt=0; attempt<15; attempt++) {
        memcpy(XtX_reg, XtX, n*n*sizeof(double));
        for(int j=0; j<n; j++) XtX_reg[j*n + j] += lambda * N;
        
        if (cholesky_decompose(XtX_reg, n, L)) {
            success = 1;
            break;
        }
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
    
    free(x_mean);
    free(XtX);
    free(Xty);
    free(L);
    free(XtX_reg);
    return lambda;
}

double predict(double* features, double* w, double b, int n) {
    double score = b;
    for(int i=0; i<n; i++) score += features[i] * w[i];
    return score;
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

uint8_t* create_intra_block_shuffled(uint8_t* src, int len, int block_size) {
    uint8_t* dst = malloc(len);
    int num_blocks = len / block_size;
    for(int b=0; b<num_blocks; b++) {
        uint8_t block[256];
        memcpy(block, src + b*block_size, block_size);
        for(int i=block_size-1; i>0; i--) {
            int j = rand() % (i+1);
            uint8_t temp = block[i];
            block[i] = block[j];
            block[j] = temp;
        }
        memcpy(dst + b*block_size, block, block_size);
    }
    int remainder = len % block_size;
    if (remainder > 0) memcpy(dst + num_blocks*block_size, src + num_blocks*block_size, remainder);
    return dst;
}

// ============================================================================
// METRICS EVALUATION
// ============================================================================

void eval_text_metrics(Dataset** test_bits, double w[8][NUM_TOTAL_FEATURES], double b[8], int n_features, int feature_offset, const char* name) {
    int N = test_bits[0]->N;
    int correct_bits = 0;
    double brier_sum = 0;
    double bpb_sum = 0;
    
    for(int i=0; i<N; i++) {
        for(int bit=0; bit<8; bit++) {
            double* f = &test_bits[bit]->X[i * NUM_TOTAL_FEATURES + feature_offset];
            double score = predict(f, w[bit], b[bit], n_features);
            
            double prob = 1.0 / (1.0 + exp(-score * 5.0));
            if (prob < 1e-7) prob = 1e-7;
            if (prob > 1.0 - 1e-7) prob = 1.0 - 1e-7;
            
            int target = (int)test_bits[bit]->y[i];
            int pred = (score > 0.5) ? 1 : 0;
            
            if (pred == target) correct_bits++;
            
            double p_target = target ? prob : (1.0 - prob);
            brier_sum += (prob - target)*(prob - target);
            bpb_sum += -log2(p_target);
        }
    }
    
    double bit_acc = 100.0 * correct_bits / (N * 8.0);
    double mse = brier_sum / (N * 8.0);
    double bpb = bpb_sum / N;
    
    printf("  %-30s | Bit Acc: %5.2f%% | Factorized BPB: %5.2f\n", name, bit_acc, mse, bpb);
}

// ============================================================================
// PHASE 7 BENCHMARK
// ============================================================================

typedef struct {
    Dataset* train_bits[8];
    Dataset* val_bits[8];
    int WARMUP;
    int N_train;
    int N_val;
    int m4_random_map[32][32];
} TortureContext;

void extract_to_datasets(TortureContext* ctx, uint8_t* source_text, int t3_enabled) {
    SiliconEngine e;
    engine_init(&e, t3_enabled);
    for(int t=0; t < ctx->WARMUP + ctx->N_train + ctx->N_val - 1; t++) {
        uint8_t input_sym = source_text[t];
        uint8_t target_sym = source_text[t+1];
        engine_tick(&e, input_sym);
        if (t >= ctx->WARMUP) {
            int idx = t - ctx->WARMUP;
            Dataset** dst_arr = (idx < ctx->N_train) ? ctx->train_bits : ctx->val_bits;
            int offset = (idx < ctx->N_train) ? idx : idx - ctx->N_train;
            double features[NUM_TOTAL_FEATURES];
            extract_features(&e, features);
            for(int bit=0; bit<8; bit++) {
                for(int i=0; i<NUM_TOTAL_FEATURES; i++) {
                    dst_arr[bit]->X[offset * NUM_TOTAL_FEATURES + i] = features[i];
                }
                dst_arr[bit]->y[offset] = (target_sym >> bit) & 1;
            }
        }
    }
}

void train_and_eval(TortureContext* ctx, int n_features, int feature_offset, const char* name, int apply_random_proj) {
    double w[8][NUM_TOTAL_FEATURES], b[8];
    for(int bit=0; bit<8; bit++) {
        Dataset* ds = create_dataset(ctx->N_train, n_features);
        for(int i=0; i<ctx->N_train; i++) {
            if (apply_random_proj) {
                for(int j=0; j<32; j++) {
                    double sum = 0;
                    for(int r=0; r<32; r++) {
                        sum += ctx->train_bits[bit]->X[i*NUM_TOTAL_FEATURES + NUM_WAVE_FEATURES + ctx->m4_random_map[j][r]];
                    }
                    ds->X[i*n_features + j] = sum;
                }
            } else {
                for(int j=0; j<n_features; j++) ds->X[i*n_features + j] = ctx->train_bits[bit]->X[i*NUM_TOTAL_FEATURES + feature_offset + j];
            }
            ds->y[i] = ctx->train_bits[bit]->y[i];
        }
        fit_ridge(ds, w[bit], &b[bit]);
        free_dataset(ds);
    }
    
    if (apply_random_proj) {
        int correct_bits = 0;
        double brier_sum = 0;
        double bpb_sum = 0;
        for(int i=0; i<ctx->N_val; i++) {
            for(int bit=0; bit<8; bit++) {
                double f[32];
                for(int j=0; j<32; j++) {
                    double sum = 0;
                    for(int r=0; r<32; r++) {
                        sum += ctx->val_bits[bit]->X[i*NUM_TOTAL_FEATURES + NUM_WAVE_FEATURES + ctx->m4_random_map[j][r]];
                    }
                    f[j] = sum;
                }
                double score = predict(f, w[bit], b[bit], n_features);
                double prob = 1.0 / (1.0 + exp(-score * 5.0));
                if (prob < 1e-7) prob = 1e-7;
                if (prob > 1.0 - 1e-7) prob = 1.0 - 1e-7;
                int target = (int)ctx->val_bits[bit]->y[i];
                int pred = (score > 0.5) ? 1 : 0;
                if (pred == target) correct_bits++;
                double p_target = target ? prob : (1.0 - prob);
                brier_sum += (prob - target)*(prob - target);
                bpb_sum += -log2(p_target);
            }
        }
        printf("  %-30s | Bit Acc: %5.2f%% | MSE: %6.4f | Factorized BPB: %5.2f\n", name, 100.0 * correct_bits / (ctx->N_val * 8.0), brier_sum / (ctx->N_val * 8.0), bpb_sum / ctx->N_val);
    } else {
        eval_text_metrics(ctx->val_bits, w, b, n_features, feature_offset, name);
    }
}

void run_torture_chamber(const char* filename, int N_train, int N_val) {
    printf("\n=== PHASE 7: TORTURE CHAMBER (%s) ===\n", filename);
    
    FILE* f = fopen(filename, "rb");
    if(!f) {
        printf("Error opening %s\n", filename);
        return;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* text = (uint8_t*)malloc(fsize);
    fread(text, 1, fsize, f);
    fclose(f);
    
    int WARMUP = 2000;
    if (fsize < WARMUP + N_train + N_val) {
        printf("File too small for requested N_train/N_val.\n");
        free(text);
        return;
    }
    
    printf("Data Split: %d Train, %d Validation\n\n", N_train, N_val);
    
    // --- Baseline: Bigram (True Byte-Level BPB) ---
    double alpha = 0.1;
    double bigram_matrix[256][256] = {0};
    double bigram_sums[256] = {0};
    
    for(int i=0; i<256; i++) {
        for(int j=0; j<256; j++) bigram_matrix[i][j] = alpha;
        bigram_sums[i] = 256 * alpha;
    }
    for(int t=WARMUP; t < WARMUP + N_train - 1; t++) {
        bigram_matrix[text[t]][text[t+1]] += 1.0;
        bigram_sums[text[t]] += 1.0;
    }
    
    double bigram_bpb = 0;
    int bigram_correct_bits = 0;
    for(int t=WARMUP + N_train; t < WARMUP + N_train + N_val - 1; t++) {
        uint8_t current = text[t];
        uint8_t target = text[t+1];
        double p = bigram_matrix[current][target] / bigram_sums[current];
        bigram_bpb += -log2(p);
        
        int best_byte = 0; double best_p = -1;
        for(int c=0; c<256; c++) {
            if (bigram_matrix[current][c] > best_p) { best_p = bigram_matrix[current][c]; best_byte = c; }
        }
        for(int bit=0; bit<8; bit++) {
            if (((best_byte >> bit) & 1) == ((target >> bit) & 1)) bigram_correct_bits++;
        }
    }
    printf("  %-30s | Bit Acc: %5.2f%% | True Byte BPB:  %5.2f\n", "Bigram (alpha=0.1)", 
           100.0 * bigram_correct_bits / (N_val * 8.0), bigram_bpb / N_val);
           
    // Marginal Unigram Bitwise Baseline (for Global Target Shuffle)
    double unigram_bpb = 0;
    int unigram_correct = 0;
    int byte_counts[256] = {0};
    for(int t=WARMUP; t < WARMUP + N_train; t++) byte_counts[text[t+1]]++;
    for(int t=WARMUP + N_train; t < WARMUP + N_train + N_val - 1; t++) {
        uint8_t target = text[t+1];
        for(int bit=0; bit<8; bit++) {
            int bit_target = (target >> bit) & 1;
            int count_1 = 0;
            for(int c=0; c<256; c++) if ((c >> bit) & 1) count_1 += byte_counts[c];
            double p1 = (double)count_1 / N_train;
            int pred = p1 > 0.5 ? 1 : 0;
            if (pred == bit_target) unigram_correct++;
            double p_target = bit_target ? p1 : (1.0 - p1);
            if (p_target < 1e-7) p_target = 1e-7;
            unigram_bpb += -log2(p_target);
        }
    }
    printf("  %-30s | Bit Acc: %5.2f%% | Factorized BPB: %5.2f\n", "Marginal Unigram Bitwise", 
           100.0 * unigram_correct / (N_val * 8.0), unigram_bpb / N_val);
           
    TortureContext ctx;
    ctx.WARMUP = WARMUP;
    ctx.N_train = N_train;
    ctx.N_val = N_val;
    for(int b=0; b<8; b++) {
        ctx.train_bits[b] = create_dataset(N_train, NUM_TOTAL_FEATURES);
        ctx.val_bits[b] = create_dataset(N_val, NUM_TOTAL_FEATURES);
    }
    for(int i=0; i<32; i++) {
        for(int j=0; j<32; j++) ctx.m4_random_map[i][j] = rand() % 16;
    }
    
    // --- 1. NORMAL REAL TEXT ---
    printf("\n--- Test: Real Text (T3 Enabled) ---\n");
    extract_to_datasets(&ctx, text, 1);
    train_and_eval(&ctx, NUM_M4_FEATURES, NUM_WAVE_FEATURES, "M4-readout", 0);
    train_and_eval(&ctx, NUM_WAVE_FEATURES, 0, "Wave-readout-T3", 0);
    train_and_eval(&ctx, NUM_WAVE_FEATURES, 0, "Wave+M4-readout-T3", 0);
    train_and_eval(&ctx, NUM_WAVE_FEATURES, 0, "Random M4 Dimension-Matched", 1);
    
    // --- 2. GLOBAL TARGET SHUFFLE ---
    printf("\n--- Test: Global Target Shuffle ---\n");
    for(int bit=0; bit<8; bit++) {
        for(int i=N_val-1; i>0; i--) {
            int j = rand() % (i+1);
            double temp = ctx.val_bits[bit]->y[i];
            ctx.val_bits[bit]->y[i] = ctx.val_bits[bit]->y[j];
            ctx.val_bits[bit]->y[j] = temp;
        }
    }
    train_and_eval(&ctx, NUM_WAVE_FEATURES, 0, "Wave-readout-T3 (Shuffled y)", 0);
    
    // --- 3. INTRA-BLOCK SHUFFLE ---
    printf("\n--- Test: Intra-Block Shuffle (16-bytes) ---\n");
    uint8_t* intra_shuffled = create_intra_block_shuffled(text, fsize, 16);
    extract_to_datasets(&ctx, intra_shuffled, 1);
    train_and_eval(&ctx, NUM_M4_FEATURES, NUM_WAVE_FEATURES, "M4-readout", 0);
    train_and_eval(&ctx, NUM_WAVE_FEATURES, 0, "Wave-readout-T3", 0);
    free(intra_shuffled);
    
    // --- 4. BLOCK SHUFFLE ---
    printf("\n--- Test: Block Shuffle (16-bytes) ---\n");
    uint8_t* block_shuffled = create_block_shuffled(text, fsize, 16);
    extract_to_datasets(&ctx, block_shuffled, 1);
    train_and_eval(&ctx, NUM_M4_FEATURES, NUM_WAVE_FEATURES, "M4-readout", 0);
    train_and_eval(&ctx, NUM_WAVE_FEATURES, 0, "Wave-readout-T3", 0);
    free(block_shuffled);
    
    // --- 5. T3 ABLATION (Current-byte only) ---
    printf("\n--- Test: T3 Ablation ---\n");
    extract_to_datasets(&ctx, text, 0);
    train_and_eval(&ctx, NUM_WAVE_FEATURES, 0, "Wave-current-only", 0);
    
    for(int b=0; b<8; b++) {
        free_dataset(ctx.train_bits[b]);
        free_dataset(ctx.val_bits[b]);
    }
    free(text);
}

int main() {
    srand(42);
    
    // Run size sweep on benchmark6_text.c
    printf("=======================================================================\n");
    printf("PHASE 7A/7B: TORTURE CHAMBER & COMPRESSION (benchmark6_text.c)\n");
    printf("=======================================================================\n");
    
    // Train Size Sweep
    run_torture_chamber("benchmark6_text.c", 2000, 2000);
    run_torture_chamber("benchmark6_text.c", 5000, 2000);
    run_torture_chamber("benchmark6_text.c", 10000, 5000);
    
    return 0;
}
