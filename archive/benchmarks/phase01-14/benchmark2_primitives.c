#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>

#ifdef _WIN32
#include <windows.h>
#endif

static inline uint64_t get_cycles() {
    unsigned int dummy;
    return __rdtscp(&dummy);
}

// Global dummy to prevent optimization
volatile int g_dummy = 0;

void bench_avx2_popcount(size_t iterations) {
    uint64_t start, end;
    
    __m256i popcnt_lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    );
    __m256i low_mask = _mm256_set1_epi8(0x0F);
    __m256i zero = _mm256_setzero_si256();

    __m256i A = _mm256_set1_epi32(0xAAAAAAAA);
    __m256i B = _mm256_set1_epi32(0x55555555);

    int local_sum = 0;
    
    start = get_cycles();
    for (size_t i = 0; i < iterations; i++) {
        __m256i xnor_res = _mm256_xor_si256(A, B);
        
        __m256i lo = _mm256_and_si256(xnor_res, low_mask);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(xnor_res, 4), low_mask);
        __m256i popcnt1 = _mm256_shuffle_epi8(popcnt_lookup, lo);
        __m256i popcnt2 = _mm256_shuffle_epi8(popcnt_lookup, hi);
        __m256i total_popcnt = _mm256_add_epi8(popcnt1, popcnt2);
        
        __m256i sum_sad = _mm256_sad_epu8(total_popcnt, zero);
        
        A = _mm256_add_epi32(A, sum_sad); 
        
        // Extract 32-bit to force dependency
        local_sum += _mm256_extract_epi32(A, 0);
    }
    end = get_cycles();
    g_dummy += local_sum;
    printf("AVX2 XNOR+Popcount (256-bit) | Cycles/op: %.3f\n", (double)(end - start)/iterations);
}

void bench_hadamard_mixing(size_t iterations) {
    uint64_t start, end;
    
    __m256i v0 = _mm256_set1_epi32(1);
    __m256i v1 = _mm256_set1_epi32(2);
    __m256i v2 = _mm256_set1_epi32(3);
    __m256i v3 = _mm256_set1_epi32(4);

    int local_sum = 0;
    start = get_cycles();
    for (size_t i = 0; i < iterations; i++) {
        __m256i t0 = _mm256_add_epi32(v0, v1);
        __m256i t1 = _mm256_sub_epi32(v0, v1);
        __m256i t2 = _mm256_add_epi32(v2, v3);
        __m256i t3 = _mm256_sub_epi32(v2, v3);

        v0 = _mm256_add_epi32(t0, t2);
        v1 = _mm256_add_epi32(t1, t3);
        v2 = _mm256_sub_epi32(t0, t2);
        v3 = _mm256_sub_epi32(t1, t3);
        
        v0 = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(0, 3, 2, 1));
        
        // Force evaluation
        local_sum += _mm256_extract_epi32(v0, 0);
        // Perturb to avoid complete loop unrolling / hoisting
        v0 = _mm256_add_epi32(v0, _mm256_set1_epi32(1));
    }
    end = get_cycles();
    g_dummy += local_sum;
    printf("AVX2 Butterfly Mixing (4x256-bit) | Cycles/op: %.3f (per 256-bit vector: %.3f)\n", 
           (double)(end - start)/iterations, (double)(end - start)/(iterations * 4));
}

void bench_local_propagation(size_t iterations) {
    size_t grid_size = 1024;
    int *state = (int*)_mm_malloc(grid_size * sizeof(int), 32);
    int *new_state = (int*)_mm_malloc(grid_size * sizeof(int), 32);
    for (size_t i = 0; i < grid_size; i++) state[i] = i;

    uint64_t start, end;
    int local_sum = 0;

    // STEP 1: Baseline Deterministic (Linear XOR)
    start = get_cycles();
    for (size_t iter = 0; iter < iterations/grid_size; iter++) {
        for (size_t i = 1; i < grid_size - 1; i++) {
            new_state[i] = state[i] ^ state[i-1];
        }
        int* temp = state; state = new_state; new_state = temp;
    }
    end = get_cycles();
    local_sum += state[10];
    printf("Local Prop Step 1 (XOR baseline)   | Cycles/element: %.3f\n", 
           (double)(end - start)/(iterations));

    // STEP 2: Controlled Mixing (+ and -)
    start = get_cycles();
    for (size_t iter = 0; iter < iterations/grid_size; iter++) {
        for (size_t i = 1; i < grid_size - 1; i++) {
            new_state[i] = (state[i-1] + state[i+1]) - state[i];
        }
        int* temp = state; state = new_state; new_state = temp;
    }
    end = get_cycles();
    local_sum += state[10];
    printf("Local Prop Step 2 (Mix +/-)        | Cycles/element: %.3f\n", 
           (double)(end - start)/(iterations));

    // STEP 3: Quasi-Rule 30 (Masked Non-linear)
    start = get_cycles();
    for (size_t iter = 0; iter < iterations/grid_size; iter++) {
        for (size_t i = 1; i < grid_size - 1; i++) {
            new_state[i] = (state[i-1] & state[i+1]) ^ state[i];
        }
        int* temp = state; state = new_state; new_state = temp;
    }
    end = get_cycles();
    local_sum += state[10];
    printf("Local Prop Step 3 (Quasi-Rule 30)  | Cycles/element: %.3f\n", 
           (double)(end - start)/(iterations));

    // STEP 4: SIMD Tile Version (32 elements at a time = 256 bits)
    start = get_cycles();
    for (size_t iter = 0; iter < iterations/grid_size; iter++) {
        for (size_t i = 0; i < grid_size; i += 8) {
            __m256i s = _mm256_load_si256((__m256i*)&state[i]);
            __m256i s_left = _mm256_permutevar8x32_epi32(s, _mm256_setr_epi32(7,0,1,2,3,4,5,6));
            __m256i s_right = _mm256_permutevar8x32_epi32(s, _mm256_setr_epi32(1,2,3,4,5,6,7,0));
            
            __m256i new_s = _mm256_xor_si256(_mm256_and_si256(s_left, s_right), s);
            _mm256_store_si256((__m256i*)&new_state[i], new_s);
        }
        int* temp = state; state = new_state; new_state = temp;
    }
    end = get_cycles();
    local_sum += state[10];
    g_dummy += local_sum;
    printf("Local Prop Step 4 (SIMD Tile 8-el) | Cycles/element: %.3f\n", 
           (double)(end - start)/(iterations));

    _mm_free(state);
    _mm_free(new_state);
}

void bench_tile_compute(size_t iterations) {
    uint64_t start, end;
    
    __m256i state = _mm256_setzero_si256();
    __m256i input = _mm256_set1_epi32(5);
    __m256i mask = _mm256_setr_epi32(0, -1, 0, -1, 0, -1, 0, -1);

    int local_sum = 0;
    start = get_cycles();
    for (size_t i = 0; i < iterations; i++) {
        __m256i masked_input = _mm256_and_si256(input, mask);
        state = _mm256_add_epi32(state, masked_input);
        
        mask = _mm256_permutevar8x32_epi32(mask, _mm256_setr_epi32(1,2,3,4,5,6,7,0));
        input = _mm256_add_epi32(input, _mm256_set1_epi32(1));
        
        local_sum += _mm256_extract_epi32(state, 0);
    }
    end = get_cycles();
    g_dummy += local_sum;
    printf("Tile 8x1 Branchless Mask-Acc (256-bit) | Cycles/op: %.3f\n", (double)(end - start)/iterations);
}

void bench_packing(size_t iterations) {
    uint64_t start, end;
    
    __m256i data = _mm256_set1_epi32(0x01020304);
    __m256i packed_res = _mm256_setzero_si256();
    
    int local_sum = 0;
    start = get_cycles();
    for(size_t i = 0; i < iterations; i++) {
        __m256i shift = _mm256_srli_epi16(data, 4);
        __m256i mask = _mm256_set1_epi8(0x0F);
        
        __m256i p1 = _mm256_and_si256(data, mask);
        __m256i p2 = _mm256_and_si256(shift, _mm256_set1_epi8((char)0xF0));
        
        packed_res = _mm256_or_si256(p1, p2);
        data = _mm256_add_epi32(data, _mm256_set1_epi32(1));
        
        local_sum += _mm256_extract_epi32(packed_res, 0);
    }
    end = get_cycles();
    g_dummy += local_sum;
    printf("Packing 8-bit to 4-bit (256-bit) | Cycles/op: %.3f\n", (double)(end - start)/iterations);
}

int main() {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 1);
#endif

    printf("=== SILICON LLM - PHASE 2 PRIMITIVES ===\n");
    printf("CPU: Isolating AVX2, ILP, and Locality\n\n");

    size_t iterations = 50000000;

    printf("--- Test 1: AVX2 XNOR+POPCOUNT ---\n");
    bench_avx2_popcount(iterations);
    printf("\n");

    printf("--- Test 2: Hadamard / Butterfly Mixing ---\n");
    bench_hadamard_mixing(iterations);
    printf("\n");

    printf("--- Test 3: Local Cellular Propagation ---\n");
    bench_local_propagation(iterations); 
    printf("\n");

    printf("--- Test 4: Tile Branchless Compute ---\n");
    bench_tile_compute(iterations);
    printf("\n");

    printf("--- Test 5: Packing ---\n");
    bench_packing(iterations);
    printf("\n");

    // Print dummy to ensure it's not optimized out completely
    if (g_dummy == 42) printf("dummy = %d\n", g_dummy);

    return 0;
}
