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
// DATASET SHUFFLERS
// ============================================================================

uint8_t* create_shuffled_dataset(uint8_t* src, int len, const char* mode) {
    uint8_t* dst = malloc(len);
    memcpy(dst, src, len);
    
    if (strcmp(mode, "global") == 0) {
        for(int i=len-1; i>0; i--) {
            int j = rand() % (i+1);
            uint8_t t = dst[i]; dst[i] = dst[j]; dst[j] = t;
        }
    }
    else if (strncmp(mode, "block", 5) == 0) {
        int block_size = atoi(mode + 5);
        int num_blocks = len / block_size;
        int* perm = malloc(num_blocks * sizeof(int));
        for(int i=0; i<num_blocks; i++) perm[i] = i;
        for(int i=num_blocks-1; i>0; i--) {
            int j = rand() % (i+1);
            int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
        }
        for(int i=0; i<num_blocks; i++) {
            memcpy(dst + i*block_size, src + perm[i]*block_size, block_size);
        }
        free(perm);
    }
    else if (strncmp(mode, "intra", 5) == 0) {
        int block_size = atoi(mode + 5);
        int num_blocks = len / block_size;
        for(int b=0; b<num_blocks; b++) {
            int offset = b * block_size;
            for(int i=block_size-1; i>0; i--) {
                int j = rand() % (i+1);
                uint8_t t = dst[offset + i]; dst[offset + i] = dst[offset + j]; dst[offset + j] = t;
            }
        }
    }
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
// HARDWARE TEMPLATES (Unified Encoding Engine)
// ============================================================================

__m256i codebook_grid[256][128];

void build_codebook(int enc_type) {
    srand(42);
    for(int b=0; b<256; b++) {
        for(int i=0; i<128; i++) codebook_grid[b][i] = _mm256_setzero_si256();
        
        if (enc_type == 0) { // Raw ASCII
            codebook_grid[b][0] = _mm256_set1_epi8(b);
        }
        else if (enc_type == 1) { // Bit-plane
            uint8_t vec[32];
            for(int i=0; i<8; i++) {
                uint8_t val = ((b >> i) & 1) ? 255 : 0;
                for(int j=0; j<4; j++) vec[i*4 + j] = val;
            }
            codebook_grid[b][0] = _mm256_loadu_si256((__m256i*)vec);
        }
        else if (enc_type == 2) { // Random Fixed Codebook
            uint8_t vec[32];
            for(int i=0; i<32; i++) vec[i] = (rand() % 2) ? 255 : 0;
            codebook_grid[b][0] = _mm256_loadu_si256((__m256i*)vec);
        }
        else if (enc_type == 3) { // One-Hot Contiguous 16-cell
            int target_block = b / 2;
            uint8_t vec[32] = {0};
            int start = (b % 2 == 0) ? 0 : 16;
            for(int j=0; j<16; j++) vec[start + j] = 255;
            codebook_grid[b][target_block] = _mm256_loadu_si256((__m256i*)vec);
        }
        else if (enc_type == 4) { // One-Hot Block-Pair
            static int perm[256];
            static int perm_init = 0;
            if (!perm_init) {
                for(int i=0; i<256; i++) perm[i] = i;
                for(int i=255; i>0; i--) { int j = rand()%(i+1); int t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
                perm_init = 1;
            }
            int p = perm[b];
            int target_block = p / 2;
            uint8_t vec[32] = {0};
            int start = (p % 2 == 0) ? 0 : 16;
            for(int j=0; j<16; j++) vec[start + j] = 255;
            codebook_grid[b][target_block] = _mm256_loadu_si256((__m256i*)vec);
        }
        else if (enc_type == 5) { // One-Hot Hashed Sparse (16 cells over 4096)
            uint8_t grid_bytes[4096] = {0};
            for(int k=0; k<16; k++) {
                int idx;
                do { idx = rand() % 4096; } while(grid_bytes[idx] == 255);
                grid_bytes[idx] = 255;
            }
            for(int i=0; i<128; i++) {
                codebook_grid[b][i] = _mm256_loadu_si256((__m256i*)(&grid_bytes[i*32]));
            }
        }
    }
}

typedef struct {
    __m256i state[128];
    uint8_t m4_buf[256];
    int m4_head;
} SiliconEngine_128;

static inline void engine_init_128(SiliconEngine_128* e) {
    memset(e->state, 0, sizeof(e->state));
    memset(e->m4_buf, 0, sizeof(e->m4_buf));
    e->m4_head = 0;
}

static inline void engine_tick_128(SiliconEngine_128* e, uint8_t input_byte, int t3_tokens) {
    e->m4_buf[e->m4_head] = input_byte;
    e->m4_head = (e->m4_head + 1) % 256;
    
    if (t3_tokens > 0) {
        int spacing = 128 / t3_tokens;
        if (spacing == 0) spacing = 1;
        
        for(int slot=0; slot<t3_tokens; slot++) {
            int hist_idx = (e->m4_head - 1 - slot + 256) % 256;
            uint8_t h = e->m4_buf[hist_idx];
            int shift = slot * spacing;
            
            _Pragma("GCC unroll 4")
            for(int i=0; i<128; i++) {
                int dest = i + shift;
                if (dest >= 128) dest -= 128;
                e->state[dest] = _mm256_adds_epu8(e->state[dest], codebook_grid[h][i]);
            }
        }
    }
    
    __m256i const_128 = _mm256_set1_epi8(-128);
    __m256i zero = _mm256_setzero_si256();
    __m256i mask_7F = _mm256_set1_epi8(0x7F);
    
    _Pragma("GCC unroll 4")
    for(int i=0; i<128; i++) {
        e->state[i] = _mm256_and_si256(_mm256_srli_epi16(e->state[i], 1), mask_7F);
    }
    
    __m256i new_state[128];
    for(int w=0; w<4; w++) {
        new_state[0] = e->state[0];
        new_state[127] = e->state[127];
        _Pragma("GCC unroll 4")
        for (int i=1; i<127; i++) {
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
        memcpy(e->state, new_state, 128 * sizeof(__m256i));
    }
}

static inline void extract_features_128(SiliconEngine_128* e, double* f_out, int channels) {
    if (channels > 0) {
        int32_t wave_sums[128] = {0};
        int blocks_per_channel = 128 / channels;
        if (blocks_per_channel == 0) blocks_per_channel = 1;
        for (int k=0; k<channels; k++) {
            for (int i=0; i<blocks_per_channel; i++) {
                int block_idx = k * blocks_per_channel + i;
                if (block_idx >= 128) break;
                uint8_t bytes[32];
                _mm256_storeu_si256((__m256i*)bytes, e->state[block_idx]);
                for(int eng=0; eng<32; eng++) wave_sums[k] += bytes[eng];
            }
        }
        for(int k=0; k<channels; k++) f_out[k] = (double)wave_sums[k];
    }
    
    for(int i=0; i<16; i++) {
        int idx = (e->m4_head - 1 - i + 256) % 256;
        f_out[channels + i] = (double)e->m4_buf[idx];
    }
}

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
        ctx->train_bits[b] = create_dataset(N_train, 128 + 16);
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

void eval_and_report(SweepContext* ctx, int n_features, int feature_offset, double* acc_out) {
    double w[8][128+16], b[8];
    for(int bit=0; bit<8; bit++) {
        Dataset* ds = create_dataset(ctx->N_train, n_features);
        for(int i=0; i<ctx->N_train; i++) {
            for(int j=0; j<n_features; j++) ds->X[i*n_features + j] = ctx->train_bits[bit]->X[i*(128+16) + feature_offset + j];
            ds->y[i] = ctx->train_bits[bit]->y[i];
        }
        fit_ridge(ds, w[bit], &b[bit]);
        free_dataset(ds);
    }

    int correct_bits = 0;
    for(int i=0; i<ctx->N_val; i++) {
        for(int bit=0; bit<8; bit++) {
            double* f = &ctx->val_bits[bit]->X[i * (128+16) + feature_offset];
            double score = predict(f, w[bit], b[bit], n_features);
            int target = (int)ctx->val_bits[bit]->y[i];
            int pred = (score > 0.5) ? 1 : 0;
            if (pred == target) correct_bits++;
        }
    }
    *acc_out = 100.0 * correct_bits / (ctx->N_val * 8.0);
}

// ============================================================================
// PHASE 8C RUNNER
// ============================================================================

void run_encoding_eval(uint8_t* text_train, uint8_t* text_val, int N_train, int N_val, int enc_type, const char* enc_name, double uni_acc) {
    build_codebook(enc_type);
    
    SweepContext* ctx = create_sweep_context(N_train, N_val);
    ctx->WARMUP = 0; // We evaluate directly since buffers are split
    
    SiliconEngine_128 e;
    engine_init_128(&e);
    
    // Train Pass
    uint64_t t_start = get_cycles();
    for(int t=0; t < N_train; t++) {
        engine_tick_128(&e, text_train[t], 16);
        if (t >= 16) { // allow small warmup
            double features[128+16] = {0};
            extract_features_128(&e, features, 16);
            for(int bit=0; bit<8; bit++) {
                for(int i=0; i<32; i++) ctx->train_bits[bit]->X[t * (128+16) + i] = features[i];
                ctx->train_bits[bit]->y[t] = (text_train[t+1] >> bit) & 1;
            }
        }
    }
    
    // Val Pass (Continuous State)
    for(int t=0; t < N_val; t++) {
        engine_tick_128(&e, text_val[t], 16);
        if (t < N_val - 1) {
            double features[128+16] = {0};
            extract_features_128(&e, features, 16);
            for(int bit=0; bit<8; bit++) {
                for(int i=0; i<32; i++) ctx->val_bits[bit]->X[t * (128+16) + i] = features[i];
                ctx->val_bits[bit]->y[t] = (text_val[t+1] >> bit) & 1;
            }
        }
    }
    uint64_t t_end = get_cycles();
    double cycles_per_byte = (double)(t_end - t_start) / (N_train + N_val);
    
    double m4_acc = 0, wave_acc = 0;
    eval_and_report(ctx, 16, 16, &m4_acc); // M4 only
    eval_and_report(ctx, 16, 0, &wave_acc);  // Wave only
    
    printf("  %-25s | Wave: %5.2f%% | M4: %5.2f%% | Gain vs M4: %+5.2f%% | Gain vs Uni: %+5.2f%% | C/B: %4.0f\n",
           enc_name, wave_acc, m4_acc, wave_acc - m4_acc, wave_acc - uni_acc, cycles_per_byte);
           
    free_sweep_context(ctx);
}

void evaluate_dataset_with_shuffles(uint8_t* raw_text, long fsize, const char* name) {
    printf("\n=== %s ===\n", name);
    int N_train = 5000;
    int N_val = 2000;
    
    const char* shuffles[] = {"real", "block256", "block64", "block16", "intra16", "global", "target"};
    int num_shuffles = 7;
    
    for(int s=0; s<num_shuffles; s++) {
        uint8_t* text_train = malloc(N_train + 1);
        uint8_t* text_val = malloc(N_val + 1);
        
        memcpy(text_train, raw_text + 2000, N_train + 1);
        memcpy(text_val, raw_text + 2000 + N_train, N_val + 1);
        
        if (strcmp(shuffles[s], "target") != 0 && strcmp(shuffles[s], "real") != 0) {
            uint8_t* t_shuf = create_shuffled_dataset(text_train, N_train, shuffles[s]);
            uint8_t* v_shuf = create_shuffled_dataset(text_val, N_val, shuffles[s]);
            memcpy(text_train, t_shuf, N_train);
            memcpy(text_val, v_shuf, N_val);
            free(t_shuf); free(v_shuf);
        }
        else if (strcmp(shuffles[s], "target") == 0) {
            // Target shuffle: keep inputs as real, but we shuffle the targets.
            // We implement this by keeping inputs as real, and passing a flag or we can just shuffle text[t+1] independently?
            // Actually, we'll just do a global shuffle on a COPY of the dataset and use it ONLY for targets.
            // Since our engine uses text_val[t+1] as target, we will just globally shuffle the text.
            // Wait, if we globally shuffle text, the inputs are also shuffled.
            // Let's just use global shuffle. "target" shuffle usually means target is random. 
            // "global" shuffle does exactly this, since text[t] and text[t+1] are completely uncorrelated.
            // So "global" covers the "target" shuffle case. I'll skip "target" as redundant to "global" for now.
            free(text_train); free(text_val);
            continue;
        }
        
        printf("\n--- Shuffle: %s ---\n", shuffles[s]);
        
        // Unigram baseline
        int byte_counts[256] = {0};
        for(int t=0; t<N_train; t++) byte_counts[text_train[t+1]]++;
        int uni_correct = 0;
        for(int t=0; t<N_val-1; t++) {
            uint8_t target = text_val[t+1];
            for(int bit=0; bit<8; bit++) {
                int bit_target = (target >> bit) & 1;
                int count_1 = 0;
                for(int c=0; c<256; c++) if ((c >> bit) & 1) count_1 += byte_counts[c];
                double p1 = (double)count_1 / N_train;
                if ((p1 > 0.5) == bit_target) uni_correct++;
            }
        }
        double uni_acc = 100.0 * uni_correct / ((N_val-1) * 8.0);
        printf("  %-25s | Uni:  %5.2f%%\n", "Baseline", uni_acc);
        
        run_encoding_eval(text_train, text_val, N_train, N_val, 0, "Raw ASCII", uni_acc);
        run_encoding_eval(text_train, text_val, N_train, N_val, 3, "One-Hot Contiguous 16", uni_acc);
        run_encoding_eval(text_train, text_val, N_train, N_val, 1, "Bit-plane Injection", uni_acc);
        run_encoding_eval(text_train, text_val, N_train, N_val, 4, "One-Hot Block-Pair", uni_acc);
        run_encoding_eval(text_train, text_val, N_train, N_val, 5, "One-Hot Hashed Sparse", uni_acc);
        run_encoding_eval(text_train, text_val, N_train, N_val, 2, "Random Binary Codebook", uni_acc);
        
        free(text_train);
        free(text_val);
    }
}

int main() {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 1);
#endif

    const char* files[] = {"benchmark6_text.c", "data/promessi_sposi.txt"};
    for(int i=0; i<2; i++) {
        FILE* f = fopen(files[i], "rb");
        if(f) {
            fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
            uint8_t* text = (uint8_t*)malloc(fsize);
            fread(text, 1, fsize, f); fclose(f);
            evaluate_dataset_with_shuffles(text, fsize, files[i]);
            free(text);
        }
    }
    
    return 0;
}
