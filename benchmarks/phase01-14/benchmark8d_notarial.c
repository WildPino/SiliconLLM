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
uint8_t codebook_32[256][32];
double proj_matrix[16][512];

void build_codebook(int seed) {
    srand(seed);
    for(int b=0; b<256; b++) {
        for(int i=0; i<32; i++) codebook_32[b][i] = (rand() % 2) ? 255 : 0;
        codebook_grid[b][0] = _mm256_loadu_si256((__m256i*)codebook_32[b]);
        for(int i=1; i<128; i++) codebook_grid[b][i] = _mm256_setzero_si256();
    }
    for(int i=0; i<16; i++) {
        for(int j=0; j<512; j++) {
            proj_matrix[i][j] = (rand() % 2) ? 1.0 : -1.0;
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

static inline void engine_tick_128(SiliconEngine_128* e, uint8_t input_byte, int reset_state) {
    e->m4_buf[e->m4_head] = input_byte;
    e->m4_head = (e->m4_head + 1) % 256;
    
    if (reset_state) {
        memset(e->state, 0, sizeof(e->state));
    }
    
    int t3_tokens = 16;
    int spacing = 128 / t3_tokens;
    
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

static inline void extract_all_features(SiliconEngine_128* e, double* f_wave, double* f_m4_raw, double* f_m4_full, double* f_m4_pooled, double* f_m4_proj) {
    // 1. Wave 16D
    int channels = 16;
    int32_t wave_sums[128] = {0};
    int blocks_per_channel = 128 / channels;
    for (int k=0; k<channels; k++) {
        for (int i=0; i<blocks_per_channel; i++) {
            int block_idx = k * blocks_per_channel + i;
            if (block_idx >= 128) break;
            uint8_t bytes[32];
            _mm256_storeu_si256((__m256i*)bytes, e->state[block_idx]);
            for(int eng=0; eng<32; eng++) wave_sums[k] += bytes[eng];
        }
    }
    for(int k=0; k<channels; k++) f_wave[k] = (double)wave_sums[k];
    
    // 2. M4 Raw 16D, M4 Full 512D, M4 Pooled 32D
    for(int k=0; k<32; k++) f_m4_pooled[k] = 0;
    
    for(int slot=0; slot<16; slot++) {
        int idx = (e->m4_head - 1 - slot + 256) % 256;
        uint8_t h = e->m4_buf[idx];
        f_m4_raw[slot] = (double)h;
        for(int k=0; k<32; k++) {
            double v = (double)codebook_32[h][k];
            f_m4_full[slot*32 + k] = v;
            f_m4_pooled[k] += v;
        }
    }
    
    // 3. M4 Projected 16D
    for(int i=0; i<16; i++) {
        f_m4_proj[i] = 0;
        for(int j=0; j<512; j++) {
            f_m4_proj[i] += f_m4_full[j] * proj_matrix[i][j];
        }
    }
}

// ============================================================================
// EVALUATION CONTEXT
// ============================================================================

typedef struct {
    Dataset* train_bits[8];
    Dataset* val_bits[8];
    int N_train;
    int N_val;
} SweepContext;

SweepContext* create_sweep_context(int N_train, int N_val, int num_features) {
    SweepContext* ctx = malloc(sizeof(SweepContext));
    ctx->N_train = N_train;
    ctx->N_val = N_val;
    for(int b=0; b<8; b++) {
        ctx->train_bits[b] = create_dataset(N_train, num_features);
        ctx->val_bits[b] = create_dataset(N_val, num_features);
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

double eval_and_report(SweepContext* ctx) {
    int n_features = ctx->train_bits[0]->num_features;
    double* w = malloc(8 * n_features * sizeof(double));
    double b[8];
    for(int bit=0; bit<8; bit++) {
        fit_ridge(ctx->train_bits[bit], &w[bit * n_features], &b[bit]);
    }

    int correct_bits = 0;
    for(int i=0; i<ctx->N_val; i++) {
        for(int bit=0; bit<8; bit++) {
            double* f = &ctx->val_bits[bit]->X[i * n_features];
            double score = predict(f, &w[bit * n_features], b[bit], n_features);
            int target = (int)ctx->val_bits[bit]->y[i];
            int pred = (score > 0.5) ? 1 : 0;
            if (pred == target) correct_bits++;
        }
    }
    free(w);
    return 100.0 * correct_bits / (ctx->N_val * 8.0);
}

// ============================================================================
// PHASE 8D RUNNER
// ============================================================================

typedef struct {
    double m4_raw;
    double m4_full;
    double m4_pooled;
    double m4_proj;
    double wave_pers;
    double wave_reset;
    double cb_wave_pers;
    double cb_wave_reset;
    double bigram;
} ResultMetrics;

void evaluate_split(uint8_t* text_train, uint8_t* text_val, int N_train, int N_val, int seed, ResultMetrics* out) {
    build_codebook(seed);
    
    SweepContext* ctx_m4_raw = create_sweep_context(N_train, N_val, 16);
    SweepContext* ctx_m4_full = create_sweep_context(N_train, N_val, 512);
    SweepContext* ctx_m4_pooled = create_sweep_context(N_train, N_val, 32);
    SweepContext* ctx_m4_proj = create_sweep_context(N_train, N_val, 16);
    SweepContext* ctx_wave_pers = create_sweep_context(N_train, N_val, 16);
    SweepContext* ctx_wave_reset = create_sweep_context(N_train, N_val, 16);
    
    SiliconEngine_128 e_pers; engine_init_128(&e_pers);
    SiliconEngine_128 e_reset; engine_init_128(&e_reset);
    
    for(int t=0; t < N_train + N_val; t++) {
        int is_val = (t >= N_train);
        int local_t = is_val ? t - N_train : t;
        uint8_t byte_in = is_val ? text_val[local_t] : text_train[local_t];
        uint8_t target_byte = is_val ? text_val[local_t+1] : text_train[local_t+1];
        
        engine_tick_128(&e_pers, byte_in, 0); // 0 means false for reset
        engine_tick_128(&e_reset, byte_in, 1); // 1 means true for reset
        
        if ((!is_val && local_t >= 16) || (is_val && local_t < N_val-1)) {
            double f_wave_p[16], f_wave_r[16], f_raw[16], f_full[512], f_pooled[32], f_proj[16];
            extract_all_features(&e_pers, f_wave_p, f_raw, f_full, f_pooled, f_proj);
            extract_all_features(&e_reset, f_wave_r, f_raw, f_full, f_pooled, f_proj); // throwaway redundant M4
            
            SweepContext* c_raw = is_val ? ctx_m4_raw : ctx_m4_raw; // just mapping
            SweepContext* cr = is_val ? ctx_m4_raw : ctx_m4_raw; 
            
            for(int bit=0; bit<8; bit++) {
                int bit_val = (target_byte >> bit) & 1;
                
                SweepContext* tgts[] = {ctx_m4_raw, ctx_m4_full, ctx_m4_pooled, ctx_m4_proj, ctx_wave_pers, ctx_wave_reset};
                double* feats[] = {f_raw, f_full, f_pooled, f_proj, f_wave_p, f_wave_r};
                int dims[] = {16, 512, 32, 16, 16, 16};
                
                for(int m=0; m<6; m++) {
                    Dataset* ds = is_val ? tgts[m]->val_bits[bit] : tgts[m]->train_bits[bit];
                    for(int d=0; d<dims[m]; d++) ds->X[local_t * dims[m] + d] = feats[m][d];
                    ds->y[local_t] = bit_val;
                }
            }
        }
    }
    
    out->m4_raw = eval_and_report(ctx_m4_raw);
    out->m4_full = eval_and_report(ctx_m4_full);
    out->m4_pooled = eval_and_report(ctx_m4_pooled);
    out->m4_proj = eval_and_report(ctx_m4_proj);
    out->wave_pers = eval_and_report(ctx_wave_pers);
    out->wave_reset = eval_and_report(ctx_wave_reset);
    
    // Bigramma
    int byte_counts[256][256] = {0};
    for(int t=0; t<N_train; t++) byte_counts[text_train[t]][text_train[t+1]]++;
    int uni_correct = 0;
    for(int t=0; t<N_val-1; t++) {
        uint8_t context = text_val[t];
        uint8_t target = text_val[t+1];
        for(int bit=0; bit<8; bit++) {
            int bit_target = (target >> bit) & 1;
            int count_1 = 0;
            int total = 0;
            for(int c=0; c<256; c++) {
                if ((c >> bit) & 1) count_1 += byte_counts[context][c];
                total += byte_counts[context][c];
            }
            double p1 = (total > 0) ? (double)count_1 / total : 0.5;
            if ((p1 > 0.5) == bit_target) uni_correct++;
        }
    }
    out->bigram = 100.0 * uni_correct / ((N_val-1) * 8.0);
    
    free_sweep_context(ctx_m4_raw); free_sweep_context(ctx_m4_full);
    free_sweep_context(ctx_m4_pooled); free_sweep_context(ctx_m4_proj);
    free_sweep_context(ctx_wave_pers); free_sweep_context(ctx_wave_reset);
}

void evaluate_dataset(uint8_t* raw_text, long fsize, const char* name) {
    printf("\n=========================================================\n");
    printf("DATASET: %s\n", name);
    printf("=========================================================\n");
    
    int N_train = 5000;
    int N_val = 2000;
    int num_splits = 5;
    int seeds[] = {42, 123, 999};
    int num_seeds = 3;
    
    double sum_m4_raw = 0, sum_m4_full = 0, sum_m4_pooled = 0, sum_m4_proj = 0;
    double sum_wave_p = 0, sum_wave_r = 0, sum_bigram = 0;
    
    for(int s=0; s<num_splits; s++) {
        int offset = 2000 + s * (N_train + N_val);
        if (offset + N_train + N_val > fsize) break;
        
        for(int seed_idx=0; seed_idx<num_seeds; seed_idx++) {
            ResultMetrics rm = {0};
            evaluate_split(raw_text + offset, raw_text + offset + N_train, N_train, N_val, seeds[seed_idx], &rm);
            
            sum_m4_raw += rm.m4_raw;
            sum_m4_full += rm.m4_full;
            sum_m4_pooled += rm.m4_pooled;
            sum_m4_proj += rm.m4_proj;
            sum_wave_p += rm.wave_pers;
            sum_wave_r += rm.wave_reset;
            sum_bigram += rm.bigram;
        }
    }
    
    int total_runs = num_splits * num_seeds;
    printf("  %-25s : %5.2f%%\n", "Bigram (Markov-1)", sum_bigram / total_runs);
    printf("  %-25s : %5.2f%%\n", "Raw M4 (16D)", sum_m4_raw / total_runs);
    printf("  %-25s : %5.2f%%\n", "Codebook M4 Full (512D)", sum_m4_full / total_runs);
    printf("  %-25s : %5.2f%%\n", "Codebook M4 Pooled (32D)", sum_m4_pooled / total_runs);
    printf("  %-25s : %5.2f%%\n", "Codebook M4 Proj (16D)", sum_m4_proj / total_runs);
    printf("  %-25s : %5.2f%%\n", "Codebook Wave Reset (16D)", sum_wave_r / total_runs);
    printf("  %-25s : %5.2f%%\n", "Codebook Wave Pers (16D)", sum_wave_p / total_runs);
}

int main() {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 1);
#endif

    const char* files[] = {"benchmark6_text.c", "DOCS/architecture_decisions.md", "data/promessi_sposi.txt"};
    for(int i=0; i<3; i++) {
        FILE* f = fopen(files[i], "rb");
        if(f) {
            fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
            if (fsize < 2000 + 5*(5000+2000)) {
                printf("[Warning] %s is small, evaluating first split only.\n", files[i]);
            }
            uint8_t* text = (uint8_t*)malloc(fsize);
            fread(text, 1, fsize, f); fclose(f);
            evaluate_dataset(text, fsize, files[i]);
            free(text);
        }
    }
    return 0;
}
