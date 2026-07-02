#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define READOUT_SIZE 64
#define NUM_PATTERNS 1000
#define R3_NUM_TEMPLATES 16

static inline uint64_t get_cycles() {
    unsigned int dummy;
    return __rdtscp(&dummy);
}

// -----------------------------------------------------------------------------
// CAPACITY & ROBUSTNESS BENCHMARKS (RIGOROUS)
// -----------------------------------------------------------------------------
typedef struct {
    uint32_t v[4];
} Vec4;

typedef struct {
    uint32_t v[8];
} Vec8;

typedef struct {
    uint32_t v[R3_NUM_TEMPLATES];
} Vec16;

int cmp_vec4(Vec4 a, Vec4 b) { return memcmp(&a, &b, sizeof(Vec4)) == 0; }
int cmp_vec8(Vec8 a, Vec8 b) { return memcmp(&a, &b, sizeof(Vec8)) == 0; }
int cmp_vec16(Vec16 a, Vec16 b) { return memcmp(&a, &b, sizeof(Vec16)) == 0; }

Vec4 get_r2_k4(uint8_t* rz) {
    Vec4 out = {0};
    for(int i=0; i<READOUT_SIZE; i+=4) {
        out.v[0] += rz[i]; out.v[1] += rz[i+1];
        out.v[2] += rz[i+2]; out.v[3] += rz[i+3];
    }
    return out;
}

Vec8 get_r2_k8(uint8_t* rz) {
    Vec8 out = {0};
    for(int i=0; i<READOUT_SIZE; i+=8) {
        for(int k=0; k<8; k++) out.v[k] += rz[i+k];
    }
    return out;
}

// AVX2 lookup table for popcount
__m256i popcnt_lookup;
__m256i low_mask;
__m256i templates_0[R3_NUM_TEMPLATES];
__m256i templates_1[R3_NUM_TEMPLATES];

void init_r3() {
    popcnt_lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    );
    low_mask = _mm256_set1_epi8(0x0F);
    
    for(int i=0; i<R3_NUM_TEMPLATES; i++) {
        uint8_t t[64];
        for(int j=0; j<64; j++) t[j] = rand() % 256;
        templates_0[i] = _mm256_loadu_si256((__m256i*)t);
        templates_1[i] = _mm256_loadu_si256((__m256i*)(t+32));
    }
}

Vec16 get_r3(uint8_t* rz) {
    Vec16 out = {0};
    __m256i v0 = _mm256_loadu_si256((__m256i*)rz);
    __m256i v1 = _mm256_loadu_si256((__m256i*)(rz+32));
    __m256i zero = _mm256_setzero_si256();
    
    for(int i=0; i<R3_NUM_TEMPLATES; i++) {
        __m256i x0 = _mm256_xor_si256(v0, templates_0[i]);
        __m256i x1 = _mm256_xor_si256(v1, templates_1[i]);
        
        __m256i lo0 = _mm256_and_si256(x0, low_mask);
        __m256i hi0 = _mm256_and_si256(_mm256_srli_epi16(x0, 4), low_mask);
        __m256i pc0 = _mm256_add_epi8(_mm256_shuffle_epi8(popcnt_lookup, lo0), _mm256_shuffle_epi8(popcnt_lookup, hi0));
        
        __m256i lo1 = _mm256_and_si256(x1, low_mask);
        __m256i hi1 = _mm256_and_si256(_mm256_srli_epi16(x1, 4), low_mask);
        __m256i pc1 = _mm256_add_epi8(_mm256_shuffle_epi8(popcnt_lookup, lo1), _mm256_shuffle_epi8(popcnt_lookup, hi1));
        
        __m256i sum_sad = _mm256_sad_epu8(_mm256_add_epi8(pc0, pc1), zero);
        out.v[i] = _mm256_extract_epi32(sum_sad, 0) + _mm256_extract_epi32(sum_sad, 4);
    }
    return out;
}

void bench_rigorous_capacity() {
    printf("--- RIGOROUS CHANNEL CAPACITY TEST ---\n");
    init_r3();
    
    uint8_t patterns[NUM_PATTERNS][READOUT_SIZE];
    for(int i=0; i<NUM_PATTERNS; i++) {
        for(int j=0; j<READOUT_SIZE; j++) patterns[i][j] = rand() % 256;
    }
    
    int col_r2_k4 = 0, col_r2_k8 = 0, col_r3 = 0;
    
    Vec4 outs_k4[NUM_PATTERNS];
    Vec8 outs_k8[NUM_PATTERNS];
    Vec16 outs_r3[NUM_PATTERNS];
    
    for(int i=0; i<NUM_PATTERNS; i++) {
        outs_k4[i] = get_r2_k4(patterns[i]);
        outs_k8[i] = get_r2_k8(patterns[i]);
        outs_r3[i] = get_r3(patterns[i]);
    }
    
    // O(N^2) collision check
    for(int i=0; i<NUM_PATTERNS; i++) {
        for(int j=i+1; j<NUM_PATTERNS; j++) {
            if(cmp_vec4(outs_k4[i], outs_k4[j])) col_r2_k4++;
            if(cmp_vec8(outs_k8[i], outs_k8[j])) col_r2_k8++;
            if(cmp_vec16(outs_r3[i], outs_r3[j])) col_r3++;
        }
    }
    
    printf("Testing %d completely random patterns for collisions:\n", NUM_PATTERNS);
    printf("  R2 (K=4) : %d collisions\n", col_r2_k4);
    printf("  R2 (K=8) : %d collisions\n", col_r2_k8);
    printf("  R3 (16T) : %d collisions\n", col_r3);
    printf("\n");
    
    printf("--- SENSITIVITY TEST (Flipping Bytes) ---\n");
    uint8_t base[READOUT_SIZE];
    for(int i=0; i<READOUT_SIZE; i++) base[i] = rand() % 256;
    
    Vec4 b_k4 = get_r2_k4(base);
    Vec8 b_k8 = get_r2_k8(base);
    Vec16 b_r3 = get_r3(base);
    
    int flip_amounts[] = {1, 2, 4, 8, 16};
    
    for(int f=0; f<5; f++) {
        int flips = flip_amounts[f];
        int fail_k4=0, fail_k8=0, fail_r3=0;
        int trials = 1000;
        for(int t=0; t<trials; t++) {
            uint8_t variant[READOUT_SIZE];
            memcpy(variant, base, READOUT_SIZE);
            for(int k=0; k<flips; k++) variant[rand()%READOUT_SIZE] ^= 0xFF; // flip full bytes
            
            Vec4 v_k4 = get_r2_k4(variant);
            Vec8 v_k8 = get_r2_k8(variant);
            Vec16 v_r3 = get_r3(variant);
            
            if(cmp_vec4(b_k4, v_k4)) fail_k4++;
            if(cmp_vec8(b_k8, v_k8)) fail_k8++;
            if(cmp_vec16(b_r3, v_r3)) fail_r3++;
        }
        printf("Flipping %2d bytes | Indistinguishable -> R2(K=4): %4d/1000 | R2(K=8): %4d/1000 | R3(16T): %4d/1000\n", 
            flips, fail_k4, fail_k8, fail_r3);
    }
}

int main() {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 1);
#endif

    printf("=== PHASE 4.B: READOUT CAPACITY TESTS ===\n\n");
    bench_rigorous_capacity();

    return 0;
}
