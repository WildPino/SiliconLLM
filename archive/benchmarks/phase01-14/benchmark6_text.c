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
} SiliconEngine;

void engine_init(SiliconEngine* e) {
    memset(e->state, 0, sizeof(e->state));
    memset(e->m4_buf, 0, sizeof(e->m4_buf));
    e->m4_head = 0;
    e->grid_vectors = GRID_VECTORS;
}

void engine_tick(SiliconEngine* e, uint8_t input_byte) {
    __m256i current_state[GRID_VECTORS];
    memcpy(current_state, e->state, sizeof(current_state));

    __m256i m4_mem = _mm256_set1_epi8(input_byte);
    e->m4_buf[e->m4_head] = m4_mem;
    e->m4_head = (e->m4_head + 1) % HISTORY_SIZE;

    __m256i t0 = e->m4_buf[(e->m4_head - 1 + HISTORY_SIZE) % HISTORY_SIZE];
    __m256i t1 = e->m4_buf[(e->m4_head - 2 + HISTORY_SIZE) % HISTORY_SIZE];
    __m256i t2 = e->m4_buf[(e->m4_head - 3 + HISTORY_SIZE) % HISTORY_SIZE];

    int blocks = GRID_VECTORS;
    for(int i=0; i<blocks; i+=12) {
        if (i < blocks) e->state[i] = _mm256_adds_epu8(e->state[i], t0);
        if (i+4 < blocks) e->state[i+4] = _mm256_adds_epu8(e->state[i+4], t1);
        if (i+8 < blocks) e->state[i+8] = _mm256_adds_epu8(e->state[i+8], t2);
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
            
            // A1
            __m256i r0 = _mm256_adds_epu8(_mm256_avg_epu8(L, R), _mm256_subs_epu8(C, const_128));
            // A2
            __m256i r1 = _mm256_adds_epu8(_mm256_subs_epu8(L, C), R);
            // A3
            __m256i l_half = _mm256_avg_epu8(L, zero);
            __m256i c_half = _mm256_avg_epu8(C, zero);
            __m256i r_half = _mm256_avg_epu8(R, zero);
            __m256i r2 = _mm256_adds_epu8(l_half, _mm256_adds_epu8(r_half, c_half));
            // A4
            __m256i r3 = _mm256_subs_epu8(_mm256_adds_epu8(L, R), C);
            
            // Simplified blend (we use all rules, here we just cycle or mix)
            // Let's use a static blend mask for simplicity since we don't have rule_select array here
            __m256i m0 = _mm256_set1_epi8(0xAA); // Alternating
            __m256i m1 = _mm256_set1_epi8(0xCC);
            __m256i sel01 = _mm256_blendv_epi8(r0, r1, m0);
            __m256i sel23 = _mm256_blendv_epi8(r2, r3, m0);
            new_state[i] = _mm256_blendv_epi8(sel01, sel23, m1);
        }
        memcpy(e->state, new_state, blocks * sizeof(__m256i));
    }
}

void extract_features(SiliconEngine* e, double* f_out) {
    // 32 Wave Channels (Sum over 8 cells per channel, equivalent to benchmark5)
    int32_t wave_sums[32] = {0};
    int offset = e->grid_vectors - 256; // benchmark5 extracted from last 256 blocks
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

// A is n*n, symmetric positive definite. L is output (n*n, lower triangular).
// Returns 1 if successful, 0 if matrix is not positive definite.
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

// Solve L * L^T * x = b
void cholesky_solve(const double* L, int n, const double* b, double* x) {
    double* y = (double*)malloc(n * sizeof(double));
    // Forward substitution: L * y = b
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int k = 0; k < i; k++) sum += L[i * n + k] * y[k];
        y[i] = (b[i] - sum) / L[i * n + i];
    }
    // Backward substitution: L^T * x = y
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        for (int k = i + 1; k < n; k++) sum += L[k * n + i] * x[k];
        x[i] = (y[i] - sum) / L[i * n + i];
    }
    free(y);
}

// ============================================================================
// DATASET MANAGER & ABLATION RUNNER
// ============================================================================

typedef struct {
    double* X; // [N * num_features]
    double* y; // [N]
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

// Fit Ridge Regression: W = (X^T X + lambda I)^-1 X^T Y
// X must be [N * num_features]. Y is [N].
// Returns dynamic lambda used.
double fit_ridge(Dataset* train_data, double* w_out, double* b_out) {
    int n = train_data->num_features;
    int N = train_data->N;
    
    // Mean centering
    double* x_mean = (double*)calloc(n, sizeof(double));
    double y_mean = 0;
    
    for(int i=0; i<N; i++) {
        y_mean += train_data->y[i];
        for(int j=0; j<n; j++) x_mean[j] += train_data->X[i*n + j];
    }
    y_mean /= N;
    for(int j=0; j<n; j++) x_mean[j] /= N;
    
    // Covariance X^T X
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
    // Mirror symmetric
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
        for(int j=0; j<n; j++) XtX_reg[j*n + j] += lambda * N; // lambda is per-sample
        
        if (cholesky_decompose(XtX_reg, n, L)) {
            success = 1;
            break;
        }
        lambda *= 10.0;
    }
    
    if (!success) {
        printf("ERROR: Cholesky failed even with massive regularization.\n");
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
// PHASE 6A: XOR-2 RANDOM INDIPENDENTE
// ============================================================================

void run_phase_6a() {
    printf("\n=== PHASE 6A: XOR-2 RANDOM INDEPENDENT ===\n");
    int N_train = 50000;
    int N_val = 10000;
    
    Dataset* train_data = create_dataset(N_train, NUM_TOTAL_FEATURES);
    Dataset* val_data = create_dataset(N_val, NUM_TOTAL_FEATURES);
    
    SiliconEngine e;
    engine_init(&e);
    
    // Generatore random indipendente per i due stream
    // stream 1 a t-1, stream 2 a t-2
    uint8_t history[10] = {0};
    
    for(int t=0; t<N_train + N_val + 1000; t++) {
        uint8_t a = (rand() % 2) ? 255 : 0;
        uint8_t b = (rand() % 2) ? 255 : 0;
        
        // Input: multiplexing the streams
        // Per testare XOR-2 pulito, l'input t è 'a'. L'input t-1 era 'a_prev', l'input t-2 era 'b_prev'.
        // But for a standard RC task: input_sym is random. Target is XOR(t-1, t-2).
        uint8_t input_sym = (rand() % 2) ? 255 : 0;
        
        // Shift history
        for(int i=9; i>0; i--) history[i] = history[i-1];
        history[0] = input_sym;
        
        uint8_t target_sym = (history[1] > 0) ^ (history[2] > 0) ? 1 : 0;
        
        engine_tick(&e, input_sym);
        
        if (t >= 1000) {
            int idx = t - 1000;
            Dataset* dst = (idx < N_train) ? train_data : val_data;
            int offset = (idx < N_train) ? idx : idx - N_train;
            
            double f[NUM_TOTAL_FEATURES];
            extract_features(&e, f);
            
            for(int i=0; i<NUM_TOTAL_FEATURES; i++) dst->X[offset * NUM_TOTAL_FEATURES + i] = f[i];
            dst->y[offset] = (double)target_sym;
        }
    }
    
    double w[NUM_TOTAL_FEATURES];
    double b;
    
    // --- Ablation: M4 Only ---
    Dataset* m4_train = create_dataset(N_train, NUM_M4_FEATURES);
    for(int i=0; i<N_train; i++) {
        for(int j=0; j<NUM_M4_FEATURES; j++) m4_train->X[i*NUM_M4_FEATURES + j] = train_data->X[i*NUM_TOTAL_FEATURES + NUM_WAVE_FEATURES + j];
        m4_train->y[i] = train_data->y[i];
    }
    fit_ridge(m4_train, w, &b);
    int correct = 0;
    for(int i=0; i<N_val; i++) {
        double* f = &val_data->X[i*NUM_TOTAL_FEATURES + NUM_WAVE_FEATURES];
        double p = predict(f, w, b, NUM_M4_FEATURES);
        if ((p > 0.5 && val_data->y[i] == 1) || (p <= 0.5 && val_data->y[i] == 0)) correct++;
    }
    printf("  M4-Only   Accuracy: %5.2f%%\n", 100.0 * correct / N_val);
    free_dataset(m4_train);
    
    // --- Ablation: Wave Only ---
    Dataset* wave_train = create_dataset(N_train, NUM_WAVE_FEATURES);
    for(int i=0; i<N_train; i++) {
        for(int j=0; j<NUM_WAVE_FEATURES; j++) wave_train->X[i*NUM_WAVE_FEATURES + j] = train_data->X[i*NUM_TOTAL_FEATURES + j];
        wave_train->y[i] = train_data->y[i];
    }
    fit_ridge(wave_train, w, &b);
    correct = 0;
    for(int i=0; i<N_val; i++) {
        double* f = &val_data->X[i*NUM_TOTAL_FEATURES];
        double p = predict(f, w, b, NUM_WAVE_FEATURES);
        if ((p > 0.5 && val_data->y[i] == 1) || (p <= 0.5 && val_data->y[i] == 0)) correct++;
    }
    printf("  Wave-Only Accuracy: %5.2f%%\n", 100.0 * correct / N_val);
    free_dataset(wave_train);
    
    // --- Full: Wave + M4 ---
    fit_ridge(train_data, w, &b);
    correct = 0;
    for(int i=0; i<N_val; i++) {
        double p = predict(&val_data->X[i*NUM_TOTAL_FEATURES], w, b, NUM_TOTAL_FEATURES);
        if ((p > 0.5 && val_data->y[i] == 1) || (p <= 0.5 && val_data->y[i] == 0)) correct++;
    }
    printf("  Wave+M4   Accuracy: %5.2f%%\n", 100.0 * correct / N_val);
    
    free_dataset(train_data);
    free_dataset(val_data);
}

// ============================================================================
// PHASE 6B: TEXT BENCHMARK
// ============================================================================

void eval_text_metrics(Dataset** test_bits, double w[8][NUM_TOTAL_FEATURES], double b[8], int n_features, int feature_offset, const char* name) {
    int N = test_bits[0]->N;
    int correct_bits = 0;
    double brier_sum = 0;
    double bpb_sum = 0;
    
    for(int i=0; i<N; i++) {
        double p_byte = 1.0;
        for(int bit=0; bit<8; bit++) {
            double* f = &test_bits[bit]->X[i * NUM_TOTAL_FEATURES + feature_offset];
            double score = predict(f, w[bit], b[bit], n_features);
            
            // Simple Platt scaling/Sigmoid calibration
            // score is directly used as logit. Often Ridge scores need scaling, 
            // but for a proxy BPB we just clamp and sigmoid.
            // A true calibration would fit a scalar 'alpha' on a holdout set: p = 1 / (1 + exp(-alpha * score)).
            // For now, we use a fixed steepness to get valid probabilities.
            double prob = 1.0 / (1.0 + exp(-score * 5.0)); // scale factor 5 to push confidently to 0/1
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
    
    printf("  %-15s | Bit Acc: %5.2f%% | MSE: %6.4f | BPB: %5.2f\n", name, bit_acc, mse, bpb);
}

void run_phase_6b(const char* filename) {
    printf("\n=== PHASE 6B: TEXT BENCHMARK (%s) ===\n", filename);
    
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
    int TRAIN_MAX = 50000;
    int VAL_MAX = 10000;
    
    if (fsize < WARMUP + 2000) {
        printf("File too small.\n");
        free(text);
        return;
    }
    
    int N_train = (fsize - WARMUP > TRAIN_MAX) ? TRAIN_MAX : (fsize - WARMUP) / 2;
    int N_val = (fsize - WARMUP - N_train > VAL_MAX) ? VAL_MAX : (fsize - WARMUP - N_train);
    
    printf("Data Split: %d Train, %d Validation (No Shuffle)\n", N_train, N_val);
    
    // 8 independent binary classification datasets
    Dataset* train_bits[8];
    Dataset* val_bits[8];
    for(int b=0; b<8; b++) {
        train_bits[b] = create_dataset(N_train, NUM_TOTAL_FEATURES);
        val_bits[b] = create_dataset(N_val, NUM_TOTAL_FEATURES);
    }
    
    SiliconEngine e;
    engine_init(&e);
    
    int byte_counts[256] = {0};
    
    for(int t=0; t < WARMUP + N_train + N_val - 1; t++) {
        uint8_t input_sym = text[t];
        uint8_t target_sym = text[t+1];
        
        if (t >= WARMUP && t < WARMUP + N_train) byte_counts[target_sym]++;
        
        engine_tick(&e, input_sym);
        
        if (t >= WARMUP) {
            int idx = t - WARMUP;
            Dataset** dst_arr = (idx < N_train) ? train_bits : val_bits;
            int offset = (idx < N_train) ? idx : idx - N_train;
            
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
    
    // --- Baseline: Random ---
    printf("  %-15s | Bit Acc: 50.00%% | MSE: 0.2500 | BPB:  8.00\n", "Random");
    
    // --- Baseline: Unigram ---
    double unigram_bpb = 0;
    double unigram_brier = 0;
    int unigram_correct = 0;
    for(int t=WARMUP + N_train; t < WARMUP + N_train + N_val - 1; t++) {
        uint8_t target = text[t+1];
        for(int bit=0; bit<8; bit++) {
            int bit_target = (target >> bit) & 1;
            
            // Marginal probability of this bit being 1 in training data
            int count_1 = 0;
            for(int c=0; c<256; c++) if ((c >> bit) & 1) count_1 += byte_counts[c];
            double p1 = (double)count_1 / N_train;
            
            int pred = p1 > 0.5 ? 1 : 0;
            if (pred == bit_target) unigram_correct++;
            
            double p_target = bit_target ? p1 : (1.0 - p1);
            if (p_target < 1e-7) p_target = 1e-7;
            
            unigram_brier += (p1 - bit_target)*(p1 - bit_target);
            unigram_bpb += -log2(p_target);
        }
    }
    printf("  %-15s | Bit Acc: %5.2f%% | MSE: %6.4f | BPB: %5.2f\n", "Unigram", 
           100.0 * unigram_correct / (N_val * 8.0), unigram_brier / (N_val * 8.0), unigram_bpb / N_val);
    
    // --- Baseline: Previous Byte ---
    int prev_correct = 0;
    for(int t=WARMUP + N_train; t < WARMUP + N_train + N_val - 1; t++) {
        uint8_t input = text[t];
        uint8_t target = text[t+1];
        for(int bit=0; bit<8; bit++) {
            if (((input >> bit)&1) == ((target >> bit)&1)) prev_correct++;
        }
    }
    printf("  %-15s | Bit Acc: %5.2f%% | MSE: N/A    | BPB: N/A\n", "Previous-Byte", 100.0 * prev_correct / (N_val * 8.0));
    
    // --- Ridge Regression Training ---
    double w_m4[8][NUM_TOTAL_FEATURES], b_m4[8];
    double w_wave[8][NUM_TOTAL_FEATURES], b_wave[8];
    double w_full[8][NUM_TOTAL_FEATURES], b_full[8];
    
    for(int bit=0; bit<8; bit++) {
        // M4 Only
        Dataset* m4_train = create_dataset(N_train, NUM_M4_FEATURES);
        for(int i=0; i<N_train; i++) {
            for(int j=0; j<NUM_M4_FEATURES; j++) m4_train->X[i*NUM_M4_FEATURES + j] = train_bits[bit]->X[i*NUM_TOTAL_FEATURES + NUM_WAVE_FEATURES + j];
            m4_train->y[i] = train_bits[bit]->y[i];
        }
        fit_ridge(m4_train, w_m4[bit], &b_m4[bit]);
        free_dataset(m4_train);
        
        // Wave Only
        Dataset* wave_train = create_dataset(N_train, NUM_WAVE_FEATURES);
        for(int i=0; i<N_train; i++) {
            for(int j=0; j<NUM_WAVE_FEATURES; j++) wave_train->X[i*NUM_WAVE_FEATURES + j] = train_bits[bit]->X[i*NUM_TOTAL_FEATURES + j];
            wave_train->y[i] = train_bits[bit]->y[i];
        }
        fit_ridge(wave_train, w_wave[bit], &b_wave[bit]);
        free_dataset(wave_train);
        
        // Wave + M4
        fit_ridge(train_bits[bit], w_full[bit], &b_full[bit]);
    }
    
    eval_text_metrics(val_bits, w_m4, b_m4, NUM_M4_FEATURES, NUM_WAVE_FEATURES, "M4-only Ridge");
    eval_text_metrics(val_bits, w_wave, b_wave, NUM_WAVE_FEATURES, 0, "Wave-only Ridge");
    eval_text_metrics(val_bits, w_full, b_full, NUM_TOTAL_FEATURES, 0, "Wave+M4 Ridge");
    
    for(int b=0; b<8; b++) {
        free_dataset(train_bits[b]);
        free_dataset(val_bits[b]);
    }
    free(text);
}

int main() {
    srand(42);
    
    run_phase_6a();
    
    run_phase_6b("benchmark6_text.c");
    run_phase_6b("DOCS/architecture_decisions.md");
    
    return 0;
}
