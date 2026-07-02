#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>

#ifdef _WIN32
#include <windows.h>
#endif

static inline uint64_t get_cycles() {
    unsigned int dummy;
    return __rdtscp(&dummy);
}

// 1. Dependency Chains
void bench_dependency_chains(size_t iterations) {
    uint64_t start, end;

    volatile int a = 1;
    start = get_cycles();
    for(size_t i=0; i<iterations; ++i) {
        a = (a * 3) + 1;
    }
    end = get_cycles();
    printf("Serial Dependency Chain       | Cycles/op: %.3f\n", (double)(end - start)/iterations);

    volatile int a1=1, a2=2, a3=3, a4=4;
    start = get_cycles();
    for(size_t i=0; i<iterations; ++i) {
        a1 = (a1 * 3) + 1;
        a2 = (a2 * 3) + 1;
        a3 = (a3 * 3) + 1;
        a4 = (a4 * 3) + 1;
    }
    end = get_cycles();
    printf("OoO Independent Chains (4x)   | Cycles/loop: %.3f (per op: %.3f)\n", 
           (double)(end - start)/iterations, (double)(end - start)/(iterations * 4));
}

// 2 & 3. SIMD Occupancy and Popcount Scalability
void bench_simd_popcount(size_t iterations) {
    uint64_t start, end;
    
    uint64_t v1 = 0xAAAAAAAAAAAAAAAAULL;
    uint64_t v2 = 0x5555555555555555ULL;
    volatile int sum_scalar = 0;
    start = get_cycles();
    for(size_t i=0; i<iterations; ++i) {
        sum_scalar += __builtin_popcountll(~(v1 ^ (v2 + i)));
    }
    end = get_cycles();
    printf("Scalar XNOR+Popcnt (64-bit)   | Cycles/op: %.3f\n", (double)(end-start)/iterations);

    __m256i avx1 = _mm256_set1_epi64x(0xAAAAAAAAAAAAAAAAULL);
    __m256i avx2 = _mm256_set1_epi64x(0x5555555555555555ULL);
    volatile __m256i res;
    start = get_cycles();
    for(size_t i=0; i<iterations; ++i) {
        avx2 = _mm256_add_epi64(avx2, _mm256_set1_epi64x(1));
        res = _mm256_xor_si256(avx1, avx2);
    }
    end = get_cycles();
    printf("AVX2 XOR+Add (256-bit)        | Cycles/op: %.3f\n", (double)(end-start)/iterations);

    // Packing/Unpacking
    __m256i vec = _mm256_set1_epi32(0xFFFFFFFF);
    volatile int sum_pack = 0;
    start = get_cycles();
    for(size_t i=0; i<iterations; i++) {
        vec = _mm256_add_epi32(vec, _mm256_set1_epi32(1));
        sum_pack += _mm256_extract_epi32(vec, 0);
    }
    end = get_cycles();
    printf("AVX2 Add + Extract (to Scalar) | Cycles/op: %.3f\n", (double)(end-start)/iterations);
}

// 4. Tiny Tiles and Branching
void bench_branching(size_t iterations) {
    uint64_t start, end;
    volatile int sum1 = 0, sum2 = 0;
    
    int* data = malloc(iterations * sizeof(int));
    for(size_t i=0; i<iterations; i++) {
        data[i] = rand() % 2; 
    }

    start = get_cycles();
    for(size_t i=0; i<iterations; ++i) {
        if(data[i]) {
            sum1 += 5;
        } else {
            sum1 -= 3;
        }
    }
    end = get_cycles();
    printf("Branching (if/else unpred.)   | Cycles/op: %.3f\n", (double)(end-start)/iterations);

    start = get_cycles();
    for(size_t i=0; i<iterations; ++i) {
        sum2 += data[i]*8 - 3;
    }
    end = get_cycles();
    printf("Branchless (math/bit-masking) | Cycles/op: %.3f\n", (double)(end-start)/iterations);
    
    free(data);
}

int main() {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 1);
#endif

    printf("=== SILICON LLM - PHASE 1.5 BENCHMARKS ===\n");
    printf("CPU: Testing Active Computation & Pipeline Stalls\n\n");

    size_t iterations = 50000000; // 50M ops

    printf("--- 1. Dependency Chains (Serial vs OoO) ---\n");
    bench_dependency_chains(iterations);
    printf("\n");

    printf("--- 2 & 3. SIMD Occupancy & Popcount Scalability ---\n");
    bench_simd_popcount(iterations);
    printf("\n");

    printf("--- 4. Tiny Tiles & Branching (Predicted vs Branchless) ---\n");
    bench_branching(iterations);
    printf("\n");

    return 0;
}
