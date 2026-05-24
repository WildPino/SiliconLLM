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

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <dataset_path> <weights.bin> [--eval-start %%] [--eval-len %%]\n", argv[0]);
        return 1;
    }
    
    const char* dataset_path = argv[1];
    const char* weights_path = argv[2];
    
    int eval_start_pct = 75, eval_len_pct = 25;
    
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--eval-start") == 0 && i + 1 < argc) eval_start_pct = atoi(argv[++i]);
        if (strcmp(argv[i], "--eval-len") == 0 && i + 1 < argc) eval_len_pct = atoi(argv[++i]);
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
    
    trigram_logits = malloc(CLASSES * CLASSES * CLASSES * sizeof(float));
    fread(trigram_logits, sizeof(float), CLASSES * CLASSES * CLASSES, fw);
    fread(feature_means, sizeof(float), SEE_FEATURE_DIM, fw);
    fread(feature_stds, sizeof(float), SEE_FEATURE_DIM, fw);
    fread(W, sizeof(float), CLASSES * SEE_FEATURE_DIM, fw);
    fread(B, sizeof(float), CLASSES, fw);
    fclose(fw);
    
    FILE* f = fopen(dataset_path, "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END);
    data_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    data = malloc(data_size);
    fread(data, 1, data_size, f);
    fclose(f);
    
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
    // Target is data[eval_start + 2], so we observe up to data[eval_start + 1].
    for (int i = 0; i <= eval_start + 1; i++) {
        see_observe(&see, data[i]);
        ctx2 = ctx1;
        ctx1 = data[i];
    }
    
    double total_loss = 0;
    uint64_t total_cycles = 0;
    
    for (int i = 0; i < eval_len; i++) {
        int global_idx = eval_start + 2 + i;
        uint8_t target = data[global_idx];
        
        uint64_t start_tsc = __rdtsc();
        
        // 1. Extract Features
        float features[SEE_FEATURE_DIM];
        see_extract(&see, features);
        
        // 2. Normalize
        for (int f = 0; f < SEE_FEATURE_DIM; f++) {
            features[f] = (features[f] - feature_means[f]) / feature_stds[f];
        }
        
        // 3. Predict Logits
        float logits[CLASSES];
        float max_l = -1e9f;
        for (int c = 0; c < CLASSES; c++) {
            logits[c] = B[c] + trigram_logits[ctx2][ctx1][c];
            logits[c] += dot_product_simd(W[c], features, SEE_FEATURE_DIM);
            if (logits[c] > max_l) max_l = logits[c];
        }
        
        float sum_e = 0.0f;
        for (int c = 0; c < CLASSES; c++) {
            sum_e += expf(logits[c] - max_l);
        }
        float prob = fmaxf(expf(logits[target] - max_l) / sum_e, 1e-10f);
        
        // 4. Update Metrics
        total_loss -= log2(prob);
        
        // 5. Observe actual target to update state
        see_observe(&see, target);
        ctx2 = ctx1;
        ctx1 = target;
        
        total_cycles += (__rdtsc() - start_tsc);
    }
    
    double bpb = total_loss / eval_len;
    double cycles_per_byte = (double)total_cycles / eval_len;
    
    printf("\n=== Streaming Inference Results ===\n");
    printf("BPB:             %.4f\n", bpb);
    printf("Cycles / Byte:   %.1f\n", cycles_per_byte);
    
    return 0;
}
