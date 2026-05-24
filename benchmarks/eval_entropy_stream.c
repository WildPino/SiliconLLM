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
int data_size = 0;

float (*trigram_logits)[CLASSES][CLASSES];
float feature_means[SEE_FEATURE_DIM];
float feature_stds[SEE_FEATURE_DIM];
float W[CLASSES][SEE_FEATURE_DIM];
float B[CLASSES];

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

typedef struct {
    uint8_t index;
    float logit;
} LogitPair;

int cmp_logit(const void* a, const void* b) {
    float va = ((LogitPair*)a)->logit;
    float vb = ((LogitPair*)b)->logit;
    if (va > vb) return -1;
    if (va < vb) return 1;
    return 0;
}

typedef struct {
    uint8_t index;
    float logit;
} CandPair;

int cmp_cand(const void* a, const void* b) {
    float va = ((CandPair*)a)->logit;
    float vb = ((CandPair*)b)->logit;
    if (va > vb) return -1;
    if (va < vb) return 1;
    return 0;
}

uint8_t (*topk_indices)[CLASSES][CLASSES];

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <dataset_path> <weights.bin> [--eval-start %%] [--eval-len %%] [--chunk-size N] [--decay N] [--seed N] [--shuffled]\n", argv[0]);
        return 1;
    }
    
    const char* dataset_path = argv[1];
    const char* weights_path = argv[2];
    
    int eval_start_pct = 75, eval_len_pct = 25;
    
    int req_chunk_size = -1;
    float req_decay = -1.0f;
    int req_seed = -1;
    int do_shuffle = 0;
    int req_topk = 256;
    int req_topm = 256;
    int tail_mode = 0; // 0=none, 1=ngram, 2=ngram+bias
    
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--eval-start") == 0 && i + 1 < argc) eval_start_pct = atoi(argv[++i]);
        if (strcmp(argv[i], "--eval-len") == 0 && i + 1 < argc) eval_len_pct = atoi(argv[++i]);
        if (strcmp(argv[i], "--chunk-size") == 0 && i + 1 < argc) req_chunk_size = atoi(argv[++i]);
        if (strcmp(argv[i], "--decay") == 0 && i + 1 < argc) req_decay = (float)atof(argv[++i]);
        if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) req_seed = atoi(argv[++i]);
        if (strcmp(argv[i], "--shuffled") == 0) do_shuffle = 1;
        if (strcmp(argv[i], "--topk") == 0 && i + 1 < argc) req_topk = atoi(argv[++i]);
        if (strcmp(argv[i], "--topm") == 0 && i + 1 < argc) req_topm = atoi(argv[++i]);
        if (strcmp(argv[i], "--tail-mode") == 0 && i + 1 < argc) tail_mode = atoi(argv[++i]);
    }

    
    FILE* fw = fopen(weights_path, "rb");
    if (!fw) {
        printf("Error: Could not open weights file %s\n", weights_path);
        return 1;
    }
    
    WeightsFileHeader header;
    fread(&header, sizeof(WeightsFileHeader), 1, fw);
    if (header.magic != 0x53454531) {
        printf("Error: Invalid magic number in weights file.\n");
        return 1;
    }
    if (header.feature_dim != SEE_FEATURE_DIM) {
        printf("Error: Feature dimension mismatch.\n");
        return 1;
    }
    
    // Runtime Integrity Validation
    if (req_chunk_size != -1 && (uint32_t)req_chunk_size != header.chunk_size) {
        printf("Error: Requested chunk size (%d) does not match weights file (%d).\n", req_chunk_size, header.chunk_size);
        return 1;
    }
    if (req_decay != -1.0f && fabsf(req_decay - header.decay) > 1e-5f) {
        printf("Error: Requested decay (%.4f) does not match weights file (%.4f).\n", req_decay, header.decay);
        return 1;
    }
    if (req_seed != -1 && (uint32_t)req_seed != header.codebook_seed) {
        printf("Error: Requested codebook seed (%d) does not match weights file (%d).\n", req_seed, header.codebook_seed);
        return 1;
    }
    
    trigram_logits = malloc(CLASSES * CLASSES * CLASSES * sizeof(float));
    fread(trigram_logits, sizeof(float), CLASSES * CLASSES * CLASSES, fw);
    fread(feature_means, sizeof(float), SEE_FEATURE_DIM, fw);
    fread(feature_stds, sizeof(float), SEE_FEATURE_DIM, fw);
    fread(W, sizeof(float), CLASSES * SEE_FEATURE_DIM, fw);
    fread(B, sizeof(float), CLASSES, fw);
    
    uint32_t lr_rank = 0;
    float* A_proj = NULL;
    float* B_proj = NULL;
    if (fread(&lr_rank, sizeof(uint32_t), 1, fw) == 1) {
        A_proj = malloc(lr_rank * SEE_FEATURE_DIM * sizeof(float));
        B_proj = malloc(CLASSES * lr_rank * sizeof(float));
        fread(A_proj, sizeof(float), lr_rank * SEE_FEATURE_DIM, fw);
        fread(B_proj, sizeof(float), CLASSES * lr_rank, fw);
        printf("Loaded Low-Rank factors with Rank = %d\n", lr_rank);
    }
    fclose(fw);

    
    
    float (*base_probs)[CLASSES][CLASSES] = NULL;
    float (*Z_base)[CLASSES] = NULL;
    if (tail_mode > 0) {
        base_probs = malloc(CLASSES * CLASSES * CLASSES * sizeof(float));
        Z_base = malloc(CLASSES * CLASSES * sizeof(float));
        for (int c2 = 0; c2 < CLASSES; c2++) {
            for (int c1 = 0; c1 < CLASSES; c1++) {
                float sum = 0;
                for (int c = 0; c < CLASSES; c++) {
                    float p = 0;
                    if (tail_mode == 1) p = expf(trigram_logits[c2][c1][c]);
                    else if (tail_mode == 2) p = expf(trigram_logits[c2][c1][c] + B[c]);
                    base_probs[c2][c1][c] = p;
                    sum += p;
                }
                Z_base[c2][c1] = sum;
            }
        }
    }

    topk_indices = malloc(CLASSES * CLASSES * CLASSES * sizeof(uint8_t));
    for (int c2 = 0; c2 < CLASSES; c2++) {
        for (int c1 = 0; c1 < CLASSES; c1++) {
            LogitPair pairs[CLASSES];
            for (int c = 0; c < CLASSES; c++) {
                pairs[c].index = c;
                pairs[c].logit = trigram_logits[c2][c1][c] + B[c];
            }
            qsort(pairs, CLASSES, sizeof(LogitPair), cmp_logit);
            for (int k = 0; k < CLASSES; k++) {
                topk_indices[c2][c1][k] = pairs[k].index;
            }
        }
    }
    
    FILE* f = fopen(dataset_path, "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END);
    data_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    data = malloc(data_size);
    fread(data, 1, data_size, f);
    fclose(f);
    
    if (do_shuffle) {
        printf("Applying global byte shuffle (seed=42) to evaluation data...\n");
        srand(42);
        for(int i = data_size - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            uint8_t temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }
    
    int eval_start = (data_size * eval_start_pct) / 100;
    int eval_len = (data_size * eval_len_pct) / 100;
    if (eval_start + eval_len + 2 > data_size) {
        eval_len = data_size - eval_start - 2;
    }
    
    printf("Evaluating on %s\n", dataset_path);
    printf("Eval Split: %d to %d (len %d)\n", eval_start, eval_start + eval_len, eval_len);
    
    SiliconEntropyState see;
    see_init(&see, header.codebook_seed, header.chunk_size, header.decay);
    
    uint8_t ctx2 = 0, ctx1 = 0;
    
    // Warmup: Observe bytes to build state up to eval_start + 1
    for (int i = 0; i <= eval_start + 1; i++) {
        see_observe(&see, data[i]);
        ctx2 = ctx1;
        ctx1 = data[i];
    }
    
    double total_loss = 0;
    uint64_t total_correct = 0;
    uint64_t total_topk_hits = 0;
    
    // Performance metrics
    uint64_t total_extract_cyc = 0;
    uint64_t total_norm_cyc = 0;
    uint64_t total_predict_cyc = 0;
    uint64_t total_observe_cyc = 0;

    double total_loss_out_pool = 0;
    double total_loss_in_pool_out_topk = 0;
    uint64_t count_target_in_pool = 0;

    uint64_t total_cand_cyc = 0;
    uint64_t total_res_cyc = 0;
    uint64_t total_soft_cyc = 0;
    double total_tail_mass = 0;
    
    uint64_t total_residual_dots = 0;
    uint64_t total_lowrank_cands = 0;
    uint64_t total_A_projs = 0;

    uint64_t total_cyc = 0;
    
    // Calculate unigram to provide baseline
    double uni_counts[CLASSES] = {0};
    double total_uni = 0;
    for(int i=0; i<eval_len; i++) {
        uni_counts[data[eval_start + 2 + i]]++;
        total_uni++;
    }
    double uni_loss = 0;
    for(int i=0; i<eval_len; i++) {
        double p = uni_counts[data[eval_start + 2 + i]] / total_uni;
        uni_loss -= log2(p > 0 ? p : 1e-10);
    }
    
    for (int i = 0; i < eval_len; i++) {
        int global_idx = eval_start + 2 + i;
        uint8_t target = data[global_idx];
        
        uint64_t t0 = __rdtsc();
        
        // 1. Extract Features
        float features[SEE_FEATURE_DIM];
        see_extract(&see, features);
        uint64_t t1 = __rdtsc();
        
        // 2. Normalize
        for (int f = 0; f < SEE_FEATURE_DIM; f++) {
            features[f] = (features[f] - feature_means[f]) / feature_stds[f];
        }
        uint64_t t2 = __rdtsc();
        
        // 3. Predict Logits
        uint64_t t_cand0 = __rdtsc();
        int cand_indices[CLASSES];
        float cand_logits[CLASSES];
        int num_cands = (tail_mode == 0) ? CLASSES : req_topk;
        int target_in_pool = 0;
        
        if (tail_mode > 0 && lr_rank > 0) {
            float z[256]; // max rank 256
            for (uint32_t r = 0; r < lr_rank; r++) {
                z[r] = dot_product_simd(&A_proj[r * SEE_FEATURE_DIM], features, SEE_FEATURE_DIM);
            }
            total_A_projs++;
            total_lowrank_cands += req_topm;
            
            for(int i=0; i<req_topk; i++) { cand_logits[i] = -1e9f; cand_indices[i] = 0; }
            for (int m = 0; m < req_topm; m++) {
                int c = topk_indices[ctx2][ctx1][m];
                if (c == target) target_in_pool = 1;
                float lr_val = 0;
                if (lr_rank == 16) {
                    __m256 b1 = _mm256_loadu_ps(&B_proj[c * 16]);
                    __m256 b2 = _mm256_loadu_ps(&B_proj[c * 16 + 8]);
                    __m256 z1 = _mm256_loadu_ps(&z[0]);
                    __m256 z2 = _mm256_loadu_ps(&z[8]);
                    __m256 sum = _mm256_fmadd_ps(b1, z1, _mm256_setzero_ps());
                    sum = _mm256_fmadd_ps(b2, z2, sum);
                    float out[8];
                    _mm256_storeu_ps(out, sum);
                    lr_val = out[0] + out[1] + out[2] + out[3] + out[4] + out[5] + out[6] + out[7];
                } else {
                    lr_val = dot_product_simd(&B_proj[c * lr_rank], z, lr_rank);
                }
                
                float l = B[c] + trigram_logits[ctx2][ctx1][c] + lr_val;
                
                if (l > cand_logits[req_topk-1]) {
                    int pos = req_topk - 1;
                    while (pos > 0 && l > cand_logits[pos-1]) {
                        cand_logits[pos] = cand_logits[pos-1];
                        cand_indices[pos] = cand_indices[pos-1];
                        pos--;
                    }
                    cand_logits[pos] = l;
                    cand_indices[pos] = c;
                }
            }
        } else {
            // Default top-k from trigram or full evaluation
            for (int i = 0; i < num_cands; i++) {
                cand_indices[i] = (tail_mode == 0) ? i : topk_indices[ctx2][ctx1][i];
            }
        }
        uint64_t t_cand1 = __rdtsc();
        total_cand_cyc += (t_cand1 - t_cand0);
        
        uint64_t t_res0 = __rdtsc();
        float full_logits[CLASSES];
        int target_in_topk = 0;
        int target_idx = -1;
        float max_l = -1e9f;
        
        for (int i = 0; i < num_cands; i++) {
            int c = cand_indices[i];
            if (c == target) {
                target_in_topk = 1;
                target_idx = i;
            }
            full_logits[i] = trigram_logits[ctx2][ctx1][c] + B[c] + dot_product_simd(W[c], features, SEE_FEATURE_DIM);
            if (full_logits[i] > max_l) max_l = full_logits[i];
        }
        float actual_max_l = max_l;
        if (max_l < 0) max_l = 0; // Don't shift up, only shift down to prevent overflow
        
        if (lr_rank == 0) target_in_pool = target_in_topk;
        
        uint64_t t_res1 = __rdtsc();
        total_res_cyc += (t_res1 - t_res0);
        
        uint64_t t_soft0 = __rdtsc();
        float Z_K = 0;
        for (int i = 0; i < num_cands; i++) {
            Z_K += expf(full_logits[i] - max_l);
        }
        
        float tail_mass = 0;
        if (tail_mode > 0) {
            tail_mass = Z_base[ctx2][ctx1];
            for (int i = 0; i < num_cands; i++) {
                tail_mass -= base_probs[ctx2][ctx1][cand_indices[i]];
            }
            if (tail_mass < 0) tail_mass = 0;
        }
        
        float Z_total = Z_K + tail_mass * expf(-max_l);
        float prob = 0;
        if (target_in_topk) {
            prob = expf(full_logits[target_idx] - max_l) / Z_total;
        } else if (tail_mode > 0) {
            prob = (base_probs[ctx2][ctx1][target] * expf(-max_l)) / Z_total;
        } else {
            prob = 1e-10f; // target not in topk and no tail mode (should not happen if tail_mode=0 as K=256)
        }
        if (prob < 1e-10f) prob = 1e-10f;
        
        int best_c = cand_indices[0]; 
        for(int i=0; i<num_cands; i++) {
            if(full_logits[i] == actual_max_l) { best_c = cand_indices[i]; break; }
        }
        
        total_tail_mass += tail_mass;
        
        uint64_t t_soft1 = __rdtsc();
        total_soft_cyc += (t_soft1 - t_soft0);
        
        uint64_t t3 = t_soft1; // for total_predict_cyc
        
        if (best_c == target) total_correct++;
        if (target_in_topk) total_topk_hits++;
        
        total_residual_dots += req_topk;
        
        // 4. Update Metrics
        double sample_loss = -log2(prob);
        total_loss += sample_loss;
        
        if (target_in_pool || tail_mode == 0) count_target_in_pool++;
        if (!target_in_pool && tail_mode > 0) total_loss_out_pool += sample_loss;
        else if (!target_in_topk && tail_mode > 0) total_loss_in_pool_out_topk += sample_loss;
        
        // 5. Observe actual target to update state
        see_observe(&see, target);
        ctx2 = ctx1;
        ctx1 = target;
        uint64_t t4 = __rdtsc();
        
        total_extract_cyc += (t1 - t0);
        total_norm_cyc += (t2 - t1);
        total_predict_cyc += (t3 - t2);
        total_observe_cyc += (t4 - t3);
        total_cyc += (t4 - t0);
    }
    
    double bpb = total_loss / eval_len;
    double unigram_bpb = uni_loss / eval_len;
    
    printf("\n=== Streaming Inference Results ===\n");
    printf("Unigram BPB:     %.4f\n", unigram_bpb);
    printf("Model BPB:       %.4f\n", bpb);
    printf("Accuracy:        %.2f%%\n", (double)total_correct / eval_len * 100.0);
    if (lr_rank == 0) {
        printf("Target in Pool:  N/A (Static K)\n");
    } else {
        printf("Target in Pool:  %.2f%% (M=%d)\n", (double)count_target_in_pool / eval_len * 100.0, req_topm);
    }
    printf("Top-K Hit Rate:  %.2f%% (K=%d)\n", (double)total_topk_hits / eval_len * 100.0, req_topk);
    if (lr_rank > 0) printf("Loss out pool:   %.4f BPB\n", total_loss_out_pool / eval_len);
    printf("Loss out K:      %.4f BPB\n", total_loss_in_pool_out_topk / eval_len);
    
    printf("\n--- Dot Products / Byte ---\n");
    printf("A Projections:   %.1f\n", (double)total_A_projs / eval_len);
    printf("LowRank Cands:   %.1f\n", (double)total_lowrank_cands / eval_len);
    printf("Full Residuals:  %.1f\n", (double)total_residual_dots / eval_len);
    printf("\n--- Mean Cycles / Byte ---\n");
    printf("Extract:         %.1f\n", (double)total_extract_cyc / eval_len);
    printf("Normalize:       %.1f\n", (double)total_norm_cyc / eval_len);
    printf("Predict:         %.1f\n", (double)total_predict_cyc / eval_len);
    printf("  - Candidate Gen: %.1f\n", (double)total_cand_cyc / eval_len);
    printf("  - Full Residual: %.1f\n", (double)total_res_cyc / eval_len);
    printf("  - Softmax+Tail:  %.1f\n", (double)total_soft_cyc / eval_len);
    printf("Avg Tail Mass:   %.4f\n", total_tail_mass / eval_len);
    printf("Observe:         %.1f\n", (double)total_observe_cyc / eval_len);
    printf("Total:           %.1f\n", (double)total_cyc / eval_len);
    
    return 0;
}
