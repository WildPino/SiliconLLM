#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "../src/silicon_entropy.h"
#include "../src/range_coder.h"
#define CDF_SCALE 16384

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

enum Mode { MODE_NONE, MODE_ENCODE, MODE_DECODE, MODE_EVAL };

typedef struct {
    uint32_t magic; // "SEE2" (0x32454553)
    uint32_t original_size;
    int32_t req_topk;
    int32_t tail_mode;
    float blend_lambda;
    uint32_t chunk_size;
    float decay;
    uint32_t codebook_seed;
    uint8_t seed_byte0;
    uint8_t seed_byte1;
} ArchiveHeader;

uint8_t (*topk_indices)[CLASSES][CLASSES];

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage:\n");
        printf("  %s --encode <input_file> <archive.bin> --weights <weights.bin> [options]\n", argv[0]);
        printf("  %s --decode <archive.bin> <output_file> --weights <weights.bin>\n", argv[0]);
        printf("  %s --eval <input_file> --weights <weights.bin> [options]\n", argv[0]);
        return 1;
    }
    
    enum Mode mode = MODE_NONE;
    const char* input_path = NULL;
    const char* archive_path = NULL;
    const char* output_path = NULL;
    const char* weights_path = NULL;
    
    int eval_start_pct = 75, eval_len_pct = 25;
    
    int req_chunk_size = -1;
    float req_decay = -1.0f;
    int req_seed = -1;
    int do_shuffle = 0;
    int req_topk = 256;
    int req_topm = 256;
    int tail_mode = 0; // 0=none, 1=ngram, 2=ngram+bias
    const char* profile = NULL;
    float blend_lambda = 0.0f;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--encode") == 0 && i + 2 < argc) {
            mode = MODE_ENCODE;
            input_path = argv[++i];
            archive_path = argv[++i];
        } else if (strcmp(argv[i], "--decode") == 0 && i + 2 < argc) {
            mode = MODE_DECODE;
            archive_path = argv[++i];
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--eval") == 0 && i + 1 < argc) {
            mode = MODE_EVAL;
            input_path = argv[++i];
        } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights_path = argv[++i];
        } else if (strcmp(argv[i], "--eval-start") == 0 && i + 1 < argc) {
            eval_start_pct = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--eval-len") == 0 && i + 1 < argc) {
            eval_len_pct = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--chunk-size") == 0 && i + 1 < argc) {
            req_chunk_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--decay") == 0 && i + 1 < argc) {
            req_decay = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            req_seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--shuffled") == 0) {
            do_shuffle = 1;
        } else if (strcmp(argv[i], "--topk") == 0 && i + 1 < argc) {
            req_topk = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--topm") == 0 && i + 1 < argc) {
            req_topm = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tail-mode") == 0 && i + 1 < argc) {
            tail_mode = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--profile") == 0 && i + 1 < argc) {
            profile = argv[++i];
        } else if (strcmp(argv[i], "--telemetry") == 0 && i + 1 < argc) {
            telemetry_path = argv[++i];
        } else if (strcmp(argv[i], "--blend") == 0 && i + 1 < argc) {
            blend_lambda = (float)atof(argv[++i]);
        }
    }
    
    if (mode == MODE_NONE || !weights_path) {
        printf("Error: Missing mode or --weights.\n");
        return 1;
    }

    
    
    if (profile) {
        if (strcmp(profile, "full") == 0) {
            req_topk = 256; tail_mode = 0; req_topm = 256;
        } else if (strcmp(profile, "accurate") == 0) {
            req_topk = 64; tail_mode = 2; req_topm = 256;
        } else if (strcmp(profile, "fast") == 0) {
            req_topk = 48; tail_mode = 2; req_topm = 256;
        } else {
            printf("Error: Unknown profile '%s'\n", profile);
            return 1;
        }
    }

    
    if (telemetry_path) {
        f_telemetry = fopen(telemetry_path, "w");
        if (f_telemetry) {
            fprintf(f_telemetry, "i,target,loss_see,loss_dyn,loss_actual,H_see,H_dyn,count_bi,tail_mass,prob_see,prob_dyn,lambda_local\n");
        }
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

    
    
    
    if (profile && strcmp(profile, "full") != 0) {
        lr_rank = 0;
        printf("Profile '%s' selected: forcing lr_rank=0\n", profile);
    }

    uint8_t* data = NULL;
    size_t data_size = 0;
    
    ArchiveHeader arc_hdr = {0};
    FILE* f_enc = NULL;
    FILE* f_dec = NULL;
    FILE* f_dump = NULL;
    RangeEncoder re;
    RangeDecoder rd;
    
    int eval_start = 0;
    int eval_len = 0;
    
    if (mode == MODE_ENCODE || mode == MODE_EVAL) {
        FILE* f = fopen(input_path, "rb");
        if (!f) {
            printf("Error: Could not open %s\n", input_path);
            return 1;
        }
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
        
        if (mode == MODE_EVAL) {
            eval_start = (data_size * eval_start_pct) / 100;
            eval_len = (data_size * eval_len_pct) / 100;
            if (eval_start + eval_len + 2 > data_size) {
                eval_len = data_size - eval_start - 2;
            }
        } else { // ENCODE
            eval_start = 0;
            eval_len = data_size >= 2 ? data_size - 2 : 0;
            
            arc_hdr.magic = 0x32454553; // "SEE2"
            arc_hdr.original_size = data_size;
            arc_hdr.req_topk = req_topk;
            arc_hdr.tail_mode = tail_mode;
            arc_hdr.blend_lambda = blend_lambda;
            arc_hdr.chunk_size = header.chunk_size;
            arc_hdr.decay = header.decay;
            arc_hdr.codebook_seed = header.codebook_seed;
            arc_hdr.seed_byte0 = data_size > 0 ? data[0] : 0;
            arc_hdr.seed_byte1 = data_size > 1 ? data[1] : 0;
            
            f_enc = fopen(archive_path, "wb");
            if (!f_enc) return 1;
            setvbuf(f_enc, NULL, _IOFBF, 65536);
            fwrite(&arc_hdr, sizeof(ArchiveHeader), 1, f_enc);
            rc_encoder_init(&re, f_enc);
        }
    } else if (mode == MODE_DECODE) {
        f_dec = fopen(archive_path, "rb");
        if (!f_dec) return 1;
        setvbuf(f_dec, NULL, _IOFBF, 65536);
        if (fread(&arc_hdr, sizeof(ArchiveHeader), 1, f_dec) != 1) {
            printf("Error reading archive header.\n");
            return 1;
        }
        if (arc_hdr.magic != 0x32454553) {
            printf("Error: Invalid archive magic.\n");
            return 1;
        }
        
        req_topk = arc_hdr.req_topk;
        tail_mode = arc_hdr.tail_mode;
        blend_lambda = arc_hdr.blend_lambda;
        
        if (output_path) {
            f_dump = fopen(output_path, "wb");
            if (!f_dump) return 1;
            setvbuf(f_dump, NULL, _IOFBF, 65536);
            if (arc_hdr.original_size > 0) fwrite(&arc_hdr.seed_byte0, 1, 1, f_dump);
            if (arc_hdr.original_size > 1) fwrite(&arc_hdr.seed_byte1, 1, 1, f_dump);
        }
        
        data_size = arc_hdr.original_size;
        eval_start = 0;
        eval_len = data_size >= 2 ? data_size - 2 : 0;
        
        rc_decoder_init(&rd, f_dec);
    }
    
    
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
    
    
    SiliconEntropyState see;
    see_init(&see, header.codebook_seed, header.chunk_size, header.decay);
    
    uint64_t total_cdf_cyc = 0;
    uint64_t total_rc_cyc = 0;

    uint8_t ctx2 = 0, ctx1 = 0;
    
    // Dynamic counts for Phase 20
    uint32_t dyn_uni_counts[CLASSES] = {0};
    uint16_t dyn_bi_counts[CLASSES][CLASSES] = {{0}};
    uint32_t dyn_total_count = 0;
    
    // Warmup
    if (mode == MODE_EVAL || mode == MODE_ENCODE) {
        for (int i = 0; i <= eval_start + 1 && i < data_size; i++) {
            see_observe(&see, data[i]);
            dyn_uni_counts[data[i]]++;
            dyn_bi_counts[ctx1][data[i]]++;
            dyn_total_count++;
            
            ctx2 = ctx1;
            ctx1 = data[i];
        }
    } else if (mode == MODE_DECODE) {
        if (data_size > 0) {
            see_observe(&see, arc_hdr.seed_byte0);
            dyn_uni_counts[arc_hdr.seed_byte0]++;
            dyn_bi_counts[ctx1][arc_hdr.seed_byte0]++;
            dyn_total_count++;
            ctx2 = ctx1;
            ctx1 = arc_hdr.seed_byte0;
        }
        if (data_size > 1) {
            see_observe(&see, arc_hdr.seed_byte1);
            dyn_uni_counts[arc_hdr.seed_byte1]++;
            dyn_bi_counts[ctx1][arc_hdr.seed_byte1]++;
            dyn_total_count++;
            ctx2 = ctx1;
            ctx1 = arc_hdr.seed_byte1;
        }
    }
    

    
    double total_loss_see_only = 0;
    double total_loss_dyn_only = 0;
    double total_loss_oracle = 0;
    double total_H_see = 0;
    double total_H_dyn = 0;
    
    FILE* f_telemetry = NULL;
    char* telemetry_path = NULL;
    float lambda_state = 0.0f; // For EMA
double total_loss = 0;
    double total_quantized_loss = 0;
    
    double startup_loss = 0;
    double startup_quantized_loss = 0;
    int startup_count = 0;
    
    double stable_loss = 0;
    double stable_quantized_loss = 0;
    int stable_count = 0;
    
    uint64_t total_correct = 0;
    uint64_t total_topk_hits = 0;
    
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

    double uni_loss = 0;
    if (mode == MODE_EVAL || mode == MODE_ENCODE) {
        double uni_counts[CLASSES] = {0};
        double total_uni = 0;
        for(int i=0; i<eval_len; i++) {
            uni_counts[data[eval_start + 2 + i]]++;
            total_uni++;
        }
        for(int i=0; i<eval_len; i++) {
            double p = uni_counts[data[eval_start + 2 + i]] / total_uni;
            uni_loss -= log2(p > 0 ? p : 1e-10);
        }
    }
    
for (int i = 0; i < eval_len; i++) {
        if (i % 5000 == 0) { printf("Byte %d\n", i); fflush(stdout); }
        int global_idx = eval_start + 2 + i;
        uint8_t original_target = 0;
        if (mode == MODE_EVAL || mode == MODE_ENCODE) {
            original_target = data[global_idx];
        }
        uint8_t target = original_target; // Will be overwritten if decoding
        
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
        
        if (tail_mode > 0 && lr_rank > 0) {
            float z[256]; 
            for (uint32_t r = 0; r < lr_rank; r++) {
                z[r] = dot_product_simd(&A_proj[r * SEE_FEATURE_DIM], features, SEE_FEATURE_DIM);
            }
            total_A_projs++;
            total_lowrank_cands += req_topm;
            
            for(int k=0; k<req_topk; k++) { cand_logits[k] = -1e9f; cand_indices[k] = 0; }
            for (int m = 0; m < req_topm; m++) {
                int c = topk_indices[ctx2][ctx1][m];
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
                    lr_val = out[0]+out[1]+out[2]+out[3]+out[4]+out[5]+out[6]+out[7];
                } else {
                    for (uint32_t r = 0; r < lr_rank; r++) lr_val += B_proj[c * lr_rank + r] * z[r];
                }
                float score = trigram_logits[ctx2][ctx1][c] + B[c] + lr_val;
                
                if (score > cand_logits[req_topk - 1]) {
                    int pos = req_topk - 1;
                    while (pos > 0 && score > cand_logits[pos - 1]) {
                        cand_logits[pos] = cand_logits[pos - 1];
                        cand_indices[pos] = cand_indices[pos - 1];
                        pos--;
                    }
                    cand_logits[pos] = score;
                    cand_indices[pos] = c;
                }
            }
        } else {
            for (int k = 0; k < num_cands; k++) {
                cand_indices[k] = (tail_mode == 0) ? k : topk_indices[ctx2][ctx1][k];
            }
        }
        
        uint64_t t_cand1 = __rdtsc();
        total_cand_cyc += (t_cand1 - t_cand0);
        
        uint64_t t_res0 = __rdtsc();
        float full_logits[CLASSES];
        float max_l = -1e9f;
        
        for (int k = 0; k < num_cands; k++) {
            int c = cand_indices[k];
            full_logits[k] = trigram_logits[ctx2][ctx1][c] + B[c] + dot_product_simd(W[c], features, SEE_FEATURE_DIM);
            if (full_logits[k] > max_l) max_l = full_logits[k];
        }
        float actual_max_l = max_l;
        if (max_l < 0) max_l = 0;
        
        uint64_t t_res1 = __rdtsc();
        total_res_cyc += (t_res1 - t_res0);
        
        uint64_t t_soft0 = __rdtsc();
        float Z_K = 0;
        for (int k = 0; k < num_cands; k++) Z_K += expf(full_logits[k] - max_l);
        
        float tail_mass = 0;
        if (tail_mode > 0) {
            tail_mass = Z_base[ctx2][ctx1];
            for (int k = 0; k < num_cands; k++) tail_mass -= base_probs[ctx2][ctx1][cand_indices[k]];
            if (tail_mass < 0) tail_mass = 0;
        }
        
        float Z_total = Z_K + tail_mass * expf(-max_l);
        float probs_tmp[CLASSES];
        
        uint64_t t_soft1 = __rdtsc();
        total_soft_cyc += (t_soft1 - t_soft0);
        
        // 4. Build CDF
        uint64_t t_cdf0 = __rdtsc();
        uint16_t freq[CLASSES];
        uint32_t total_assigned = 0;
        
        float tail_scale = (tail_mode > 0) ? (expf(-max_l) / Z_total) : 0;
        float cdf_scale_f = (float)(CDF_SCALE - CLASSES);
        
        float p_see[CLASSES];
        float p_dyn[CLASSES];
        
        for (int c = 0; c < CLASSES; c++) {
            if (tail_mode > 0) p_see[c] = base_probs[ctx2][ctx1][c] * tail_scale;
            else p_see[c] = 0;
        }
        for (int k = 0; k < num_cands; k++) {
            int c = cand_indices[k];
            p_see[c] = expf(full_logits[k] - max_l) / Z_total;
        }
        
        float alpha = 1.0f;
        float sum_uni = dyn_total_count + CLASSES * alpha;
        float sum_bi = 0;
        for(int c=0; c<CLASSES; c++) sum_bi += dyn_bi_counts[ctx1][c];
        sum_bi += CLASSES * alpha;
        
        for (int c = 0; c < CLASSES; c++) {
            float p_uni = (dyn_uni_counts[c] + alpha) / sum_uni;
            float p_bi = (dyn_bi_counts[ctx1][c] + alpha) / sum_bi;
            p_dyn[c] = 0.5f * p_uni + 0.5f * p_bi;
        }
        
        float H_see = 0;
        float H_dyn = 0;
        for (int c = 0; c < CLASSES; c++) {
            if (p_see[c] > 1e-10f) H_see -= p_see[c] * log2(p_see[c]);
            if (p_dyn[c] > 1e-10f) H_dyn -= p_dyn[c] * log2(p_dyn[c]);
        }
        
        float current_lambda = blend_lambda;
        if (blend_lambda < 0.0f) { // Auto
            float lambda_raw = 0.0f;
            float actual_bi = sum_bi - CLASSES * alpha;
            if (actual_bi > 50 && H_dyn < H_see) {
                lambda_raw = 0.9f;
            } else if (tail_mass > 0.4f) {
                lambda_raw = 0.5f;
            } else {
                lambda_raw = 0.1f;
            }
            lambda_state = 0.9f * lambda_state + 0.1f * lambda_raw;
            current_lambda = lambda_state;
        }
        
        for (int c = 0; c < CLASSES; c++) {
            probs_tmp[c] = (1.0f - current_lambda) * p_see[c] + current_lambda * p_dyn[c];
        }
        
        for (int c = 0; c < CLASSES; c++) {
            int f = (int)(probs_tmp[c] * cdf_scale_f);
            if (f < 0) f = 0;
            freq[c] = (uint16_t)(f + 1);
            total_assigned += freq[c];
        }
        
        int diff = CDF_SCALE - total_assigned;
        if (diff > 0) {
            int max_c = 0; float max_p = -1;
            for (int c = 0; c < CLASSES; c++) {
                if (probs_tmp[c] > max_p) { max_p = probs_tmp[c]; max_c = c; }
            }
            freq[max_c] += diff;
        }
        
        uint64_t t_cdf1 = __rdtsc();
        total_cdf_cyc += (t_cdf1 - t_cdf0);
        
        // 5. Encode / Decode
        uint64_t t_rc0 = __rdtsc();
        if (f_enc) {
            uint32_t cum = 0;
            for (int c = 0; c < target; c++) cum += freq[c];
            rc_encode(&re, cum, freq[target], CDF_SCALE);
        } else if (f_dec) {
            uint64_t f_val = rc_get_freq(&rd, CDF_SCALE);
            uint32_t cum = 0;
            target = 0;
            for (int c = 0; c < CLASSES; c++) {
                if (f_val >= cum && f_val < cum + freq[c]) {
                    target = c;
                    break;
                }
                cum += freq[c];
            }
            rc_decode(&rd, cum, freq[target], CDF_SCALE);
            
            
                if (f_telemetry) {
        fclose(f_telemetry);
    }
if (f_dump) {
                uint8_t t = (uint8_t)target;
                fwrite(&t, 1, 1, f_dump);
            }
        }
        uint64_t t_rc1 = __rdtsc();
        total_rc_cyc += (t_rc1 - t_rc0);
        
        // Metrics & Loss (using final resolved target)
        uint64_t t3 = t_rc1;
        float final_prob = probs_tmp[target];
        if (final_prob < 1e-10f) final_prob = 1e-10f;
        double sample_loss = -log2(final_prob);
        total_loss += sample_loss;
        
        float prob_see = p_see[target];
        if (prob_see < 1e-10f) prob_see = 1e-10f;
        double loss_see = -log2(prob_see);
        
        float prob_dyn = p_dyn[target];
        if (prob_dyn < 1e-10f) prob_dyn = 1e-10f;
        double loss_dyn = -log2(prob_dyn);
        
        total_loss_see_only += loss_see;
        total_loss_dyn_only += loss_dyn;
        total_loss_oracle += (loss_see < loss_dyn) ? loss_see : loss_dyn;
        
        total_H_see += H_see;
        total_H_dyn += H_dyn;
        
        if (f_telemetry) {
            fprintf(f_telemetry, "%d,%d,%f,%f,%f,%f,%f,%d,%f,%f,%f,%f\n", 
                    i, target, loss_see, loss_dyn, sample_loss, H_see, H_dyn, 
                    (int)(sum_bi - CLASSES * alpha), tail_mass, prob_see, prob_dyn, current_lambda);
        }

        
        double quant_prob = (double)freq[target] / CDF_SCALE;
        if (quant_prob < 1e-10) quant_prob = 1e-10;
        double quant_loss = -log2(quant_prob);
        total_quantized_loss += quant_loss;
        int is_startup = (i < 4096);
        if (is_startup) {
            startup_loss += sample_loss;
            startup_quantized_loss += quant_loss;
            startup_count++;
        } else {
            stable_loss += sample_loss;
            stable_quantized_loss += quant_loss;
            stable_count++;
        }
        
        int best_c = cand_indices[0]; 
        for(int k=0; k<num_cands; k++) {
            if(full_logits[k] == actual_max_l) { best_c = cand_indices[k]; break; }
        }
        if (best_c == target) total_correct++;
        
        // 6. Observe actual target to update state
        see_observe(&see, target);
        
        dyn_uni_counts[target]++;
        dyn_bi_counts[ctx1][target]++;
        dyn_total_count++;
        
        ctx2 = ctx1;
        ctx1 = target;
        uint64_t t4 = __rdtsc();
        
        total_extract_cyc += (t1 - t0);
        total_norm_cyc += (t2 - t1);
        total_predict_cyc += (t_cdf0 - t2); // predict includes cand+res+soft
        total_observe_cyc += (t4 - t3);
        total_cyc += (t4 - t0);
    }

    if (f_enc) {
        rc_encoder_flush(&re);
        fclose(f_enc);
    }
    if (f_dec) {
        fclose(f_dec);
    }
    if (f_dump) {
        fclose(f_dump);
    }

    if (mode == MODE_EVAL || mode == MODE_ENCODE) {
        double bpb = total_loss / eval_len;
        double quant_bpb = total_quantized_loss / eval_len;
        double startup_bpb = startup_count > 0 ? startup_quantized_loss / startup_count : 0;
        double stable_bpb = stable_count > 0 ? stable_quantized_loss / stable_count : 0;
        
        printf("\n=== Streaming Inference Results ===\n");
        printf("Unigram BPB:     %.4f\n", uni_loss / eval_len);
        printf("Model BPB:       %.4f\n", bpb);
        printf("Quantized BPB:   %.4f\n", quant_bpb);
        printf("  - Startup BPB: %.4f (first 4KB)\n", startup_bpb);
        printf("  - Stable BPB:  %.4f\n", stable_bpb);
        printf("Accuracy:        %.2f%%\n", 100.0 * total_correct / eval_len);

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
        printf("  - CDF Build:     %.1f\n", (double)total_cdf_cyc / eval_len);
        if (mode == MODE_ENCODE || mode == MODE_DECODE) {
            printf("  - Range Coder:   %.1f\n", (double)total_rc_cyc / eval_len);
        }
        printf("Avg Tail Mass:   %.4f\n", total_tail_mass / eval_len);
        printf("Observe:         %.1f\n", (double)total_observe_cyc / eval_len);
        
        double rc = (mode == MODE_ENCODE || mode == MODE_DECODE) ? (double)total_rc_cyc / eval_len : 0;
        printf("Total:           %.1f\n", (double)total_cyc / eval_len + rc);
    } else {
        printf("\nDecode Complete! Length: %d bytes.\n", eval_len);
    }
return 0;
}
        printf("--- Phase 21 Oracle / Telemetry ---\n");
        printf("SEE Only BPB:    %.4f\n", total_loss_see_only / eval_len);
        printf("Dyn Only BPB:    %.4f\n", total_loss_dyn_only / eval_len);
        printf("Oracle BPB:      %.4f\n", total_loss_oracle / eval_len);
        printf("Avg H_see:       %.4f\n", total_H_see / eval_len);
        printf("Avg H_dyn:       %.4f\n", total_H_dyn / eval_len);

