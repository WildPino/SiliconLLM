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
// HARDWARE TEMPLATES & CODEBOOKS
// ============================================================================

__m256i codebook_grid[256][128];
double codebook_32_collapsed[256][32];
double proj_matrix[16][512];

void generate_balanced_vector(uint8_t* vec) {
    int ones = 16, zeros = 16;
    for(int i=0; i<32; i++) {
        if (ones > 0 && zeros > 0) {
            if (rand() % 2) { vec[i] = 255; ones--; }
            else { vec[i] = 0; zeros--; }
        } else if (ones > 0) { vec[i] = 255; ones--; }
        else { vec[i] = 0; zeros--; }
    }
}

void build_codebook(int type, int seed) {
    srand(seed);
    
    for(int b=0; b<256; b++) {
        for(int i=0; i<128; i++) codebook_grid[b][i] = _mm256_setzero_si256();
        
        if (type == 0) { // Single-block Random
            uint8_t vec[32];
            for(int i=0; i<32; i++) vec[i] = (rand() % 2) ? 255 : 0;
            codebook_grid[b][0] = _mm256_loadu_si256((__m256i*)vec);
        }
        else if (type == 1) { // Single-block Balanced
            uint8_t vec[32];
            generate_balanced_vector(vec);
            codebook_grid[b][0] = _mm256_loadu_si256((__m256i*)vec);
        }
        else if (type == 2) { // Distributed Random (4 blocks)
            for(int k=0; k<4; k++) {
                int blk = rand() % 128;
                uint8_t vec[32];
                for(int i=0; i<32; i++) vec[i] = (rand() % 2) ? 255 : 0;
                codebook_grid[b][blk] = _mm256_loadu_si256((__m256i*)vec);
            }
        }
        else if (type == 3) { // Distributed Balanced (4 blocks)
            for(int k=0; k<4; k++) {
                int blk = rand() % 128;
                uint8_t vec[32];
                generate_balanced_vector(vec);
                codebook_grid[b][blk] = _mm256_loadu_si256((__m256i*)vec);
            }
        }
        
        // Compute the collapsed 32D vector for M4 metrics
        for(int lane=0; lane<32; lane++) codebook_32_collapsed[b][lane] = 0;
        for(int i=0; i<128; i++) {
            uint8_t bytes[32];
            _mm256_storeu_si256((__m256i*)bytes, codebook_grid[b][i]);
            for(int lane=0; lane<32; lane++) codebook_32_collapsed[b][lane] += bytes[lane];
        }
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

// ============================================================================
// FEATURE EXTRACTION (Readout Tribunal)
// ============================================================================

typedef struct {
    double m4_full[512];
    double m4_pooled[32];
    double m4_proj[16];
    
    double wave_sum[16];
    double wave_full[512];
    double wave_pooled[32];
    double wave_local[32];
    double wave_proj[16];
} FeaturesOut;

static inline void extract_features_out(SiliconEngine_128* e, FeaturesOut* out) {
    // Wave extraction
    double wave_channel_lanes[16][32] = {0};
    int blocks_per_channel = 128 / 16;
    for(int k=0; k<16; k++) {
        for(int i=0; i<blocks_per_channel; i++) {
            uint8_t bytes[32];
            _mm256_storeu_si256((__m256i*)bytes, e->state[k*blocks_per_channel + i]);
            for(int lane=0; lane<32; lane++) {
                wave_channel_lanes[k][lane] += bytes[lane];
            }
        }
    }
    
    memset(out->wave_pooled, 0, sizeof(out->wave_pooled));
    memset(out->wave_sum, 0, sizeof(out->wave_sum));
    memset(out->wave_proj, 0, sizeof(out->wave_proj));
    
    for(int k=0; k<16; k++) {
        for(int lane=0; lane<32; lane++) {
            double v = wave_channel_lanes[k][lane];
            out->wave_full[k*32 + lane] = v;
            out->wave_pooled[lane] += v;
            out->wave_sum[k] += v;
            if (k == 0) out->wave_local[lane] = v;
        }
    }
    for(int i=0; i<16; i++) {
        for(int j=0; j<512; j++) {
            out->wave_proj[i] += out->wave_full[j] * proj_matrix[i][j];
        }
    }

    // M4 extraction
    memset(out->m4_pooled, 0, sizeof(out->m4_pooled));
    memset(out->m4_proj, 0, sizeof(out->m4_proj));
    
    for(int slot=0; slot<16; slot++) {
        int idx = (e->m4_head - 1 - slot + 256) % 256;
        uint8_t h = e->m4_buf[idx];
        for(int lane=0; lane<32; lane++) {
            double v = codebook_32_collapsed[h][lane];
            out->m4_full[slot*32 + lane] = v;
            out->m4_pooled[lane] += v;
        }
    }
    for(int i=0; i<16; i++) {
        for(int j=0; j<512; j++) {
            out->m4_proj[i] += out->m4_full[j] * proj_matrix[i][j];
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
// PHASE 8E RUNNER
// ============================================================================

typedef struct {
    double m4_full, m4_pooled, m4_proj;
    double wp_sum, wp_full, wp_pooled, wp_local, wp_proj;
    double wr_sum, wr_full, wr_pooled, wr_local, wr_proj;
} RunMetrics;

void evaluate_split(uint8_t* text_train, uint8_t* text_val, int N_train, int N_val, int cb_type, int seed, RunMetrics* out) {
    build_codebook(cb_type, seed);
    
    // Arrays of contexts to avoid massive variable lists
    SweepContext* ctx_m4[3];
    ctx_m4[0] = create_sweep_context(N_train, N_val, 512); // Full
    ctx_m4[1] = create_sweep_context(N_train, N_val, 32);  // Pooled
    ctx_m4[2] = create_sweep_context(N_train, N_val, 16);  // Proj
    
    SweepContext* ctx_wp[5]; // Pers
    ctx_wp[0] = create_sweep_context(N_train, N_val, 16);  // Sum
    ctx_wp[1] = create_sweep_context(N_train, N_val, 512); // Full
    ctx_wp[2] = create_sweep_context(N_train, N_val, 32);  // Pooled
    ctx_wp[3] = create_sweep_context(N_train, N_val, 32);  // Local
    ctx_wp[4] = create_sweep_context(N_train, N_val, 16);  // Proj
    
    SweepContext* ctx_wr[5]; // Reset
    ctx_wr[0] = create_sweep_context(N_train, N_val, 16);  // Sum
    ctx_wr[1] = create_sweep_context(N_train, N_val, 512); // Full
    ctx_wr[2] = create_sweep_context(N_train, N_val, 32);  // Pooled
    ctx_wr[3] = create_sweep_context(N_train, N_val, 32);  // Local
    ctx_wr[4] = create_sweep_context(N_train, N_val, 16);  // Proj
    
    SiliconEngine_128 e_pers; engine_init_128(&e_pers);
    SiliconEngine_128 e_reset; engine_init_128(&e_reset);
    
    for(int t=0; t < N_train + N_val; t++) {
        int is_val = (t >= N_train);
        int local_t = is_val ? t - N_train : t;
        uint8_t byte_in = is_val ? text_val[local_t] : text_train[local_t];
        uint8_t target_byte = is_val ? text_val[local_t+1] : text_train[local_t+1];
        
        engine_tick_128(&e_pers, byte_in, 0); // persistent
        engine_tick_128(&e_reset, byte_in, 1); // reset
        
        if ((!is_val && local_t >= 16) || (is_val && local_t < N_val-1)) {
            FeaturesOut f_pers, f_reset;
            extract_features_out(&e_pers, &f_pers);
            extract_features_out(&e_reset, &f_reset);
            
            for(int bit=0; bit<8; bit++) {
                int bit_val = (target_byte >> bit) & 1;
                
                // Helper to populate datasets
                #define POP(ctx, feats, dim) do { \
                    Dataset* ds = is_val ? ctx->val_bits[bit] : ctx->train_bits[bit]; \
                    for(int d=0; d<dim; d++) ds->X[local_t * dim + d] = feats[d]; \
                    ds->y[local_t] = bit_val; \
                } while(0)
                
                POP(ctx_m4[0], f_pers.m4_full, 512);
                POP(ctx_m4[1], f_pers.m4_pooled, 32);
                POP(ctx_m4[2], f_pers.m4_proj, 16);
                
                POP(ctx_wp[0], f_pers.wave_sum, 16);
                POP(ctx_wp[1], f_pers.wave_full, 512);
                POP(ctx_wp[2], f_pers.wave_pooled, 32);
                POP(ctx_wp[3], f_pers.wave_local, 32);
                POP(ctx_wp[4], f_pers.wave_proj, 16);
                
                POP(ctx_wr[0], f_reset.wave_sum, 16);
                POP(ctx_wr[1], f_reset.wave_full, 512);
                POP(ctx_wr[2], f_reset.wave_pooled, 32);
                POP(ctx_wr[3], f_reset.wave_local, 32);
                POP(ctx_wr[4], f_reset.wave_proj, 16);
            }
        }
    }
    
    out->m4_full = eval_and_report(ctx_m4[0]);
    out->m4_pooled = eval_and_report(ctx_m4[1]);
    out->m4_proj = eval_and_report(ctx_m4[2]);
    
    out->wp_sum = eval_and_report(ctx_wp[0]);
    out->wp_full = eval_and_report(ctx_wp[1]);
    out->wp_pooled = eval_and_report(ctx_wp[2]);
    out->wp_local = eval_and_report(ctx_wp[3]);
    out->wp_proj = eval_and_report(ctx_wp[4]);
    
    out->wr_sum = eval_and_report(ctx_wr[0]);
    out->wr_full = eval_and_report(ctx_wr[1]);
    out->wr_pooled = eval_and_report(ctx_wr[2]);
    out->wr_local = eval_and_report(ctx_wr[3]);
    out->wr_proj = eval_and_report(ctx_wr[4]);
    
    for(int i=0; i<3; i++) free_sweep_context(ctx_m4[i]);
    for(int i=0; i<5; i++) { free_sweep_context(ctx_wp[i]); free_sweep_context(ctx_wr[i]); }
}

void evaluate_dataset(uint8_t* raw_text, long fsize, const char* name) {
    printf("\n=========================================================\n");
    printf("DATASET: %s\n", name);
    printf("=========================================================\n");
    
    int N_train = 5000, N_val = 2000;
    int num_splits = 5, num_seeds = 3;
    int seeds[] = {42, 123, 999};
    const char* cb_names[] = {"Single Random", "Single Balanced", "Dist. Random", "Dist. Balanced"};
    
    for(int cb_type=0; cb_type<4; cb_type++) {
        printf("\n--- Codebook: %s ---\n", cb_names[cb_type]);
        RunMetrics sum_rm = {0};
        
        for(int s=0; s<num_splits; s++) {
            int offset = 2000 + s * (N_train + N_val);
            if (offset + N_train + N_val > fsize) break;
            
            for(int seed_idx=0; seed_idx<num_seeds; seed_idx++) {
                RunMetrics rm = {0};
                evaluate_split(raw_text + offset, raw_text + offset + N_train, N_train, N_val, cb_type, seeds[seed_idx], &rm);
                
                sum_rm.m4_full += rm.m4_full; sum_rm.m4_pooled += rm.m4_pooled; sum_rm.m4_proj += rm.m4_proj;
                sum_rm.wp_sum += rm.wp_sum; sum_rm.wp_full += rm.wp_full; sum_rm.wp_pooled += rm.wp_pooled; sum_rm.wp_local += rm.wp_local; sum_rm.wp_proj += rm.wp_proj;
                sum_rm.wr_sum += rm.wr_sum; sum_rm.wr_full += rm.wr_full; sum_rm.wr_pooled += rm.wr_pooled; sum_rm.wr_local += rm.wr_local; sum_rm.wr_proj += rm.wr_proj;
            }
        }
        
        int runs = num_splits * num_seeds;
        printf("  [M4 Base]   Full 512D: %5.2f%% | Pooled 32D: %5.2f%% | Proj 16D: %5.2f%%\n", 
               sum_rm.m4_full/runs, sum_rm.m4_pooled/runs, sum_rm.m4_proj/runs);
        printf("  [Wave Pers] Full 512D: %5.2f%% | Pooled 32D: %5.2f%% | Proj 16D: %5.2f%% | Sum 16D: %5.2f%% | Local 32D: %5.2f%%\n", 
               sum_rm.wp_full/runs, sum_rm.wp_pooled/runs, sum_rm.wp_proj/runs, sum_rm.wp_sum/runs, sum_rm.wp_local/runs);
        printf("  [Wave Rset] Full 512D: %5.2f%% | Pooled 32D: %5.2f%% | Proj 16D: %5.2f%% | Sum 16D: %5.2f%% | Local 32D: %5.2f%%\n", 
               sum_rm.wr_full/runs, sum_rm.wr_pooled/runs, sum_rm.wr_proj/runs, sum_rm.wr_sum/runs, sum_rm.wr_local/runs);
    }
}

int main() {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 1);
#endif
    const char* file = "data/promessi_sposi.txt";
    FILE* f = fopen(file, "rb");
    if(f) {
        fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
        uint8_t* text = (uint8_t*)malloc(fsize);
        fread(text, 1, fsize, f); fclose(f);
        evaluate_dataset(text, fsize, file);
        free(text);
    }
    return 0;
}
