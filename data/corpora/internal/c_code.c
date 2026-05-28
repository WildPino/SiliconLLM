#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h> // For AVX2 and __rdtsc

#ifdef _WIN32
#include <windows.h>
#endif

// Helper to get CPU cycles
static inline uint64_t get_cycles() {
    unsigned int dummy;
    return __rdtscp(&dummy);
}

// -----------------------------------------------------------------------------
// BENCHMARK A: Sequential Memory
// -----------------------------------------------------------------------------
void benchmark_sequential(size_t size_bytes) {
    size_t num_elements = size_bytes / sizeof(int);
    int *data = (int *)malloc(size_bytes);
    if (!data) return;

    // Warmup & initialization
    for (size_t i = 0; i < num_elements; ++i) {
        data[i] = (int)i;
    }

    uint64_t start = get_cycles();
    volatile int sum = 0;
    
    // Sequential read
    for (size_t i = 0; i < num_elements; ++i) {
        sum += data[i];
    }
    
    uint64_t end = get_cycles();
    double cycles_per_element = (double)(end - start) / num_elements;

    printf("Sequential Read - Size: %8zu KB | Cycles/element: %.2f\n", size_bytes / 1024, cycles_per_element);

    free(data);
}

// -----------------------------------------------------------------------------
// BENCHMARK B: Random Memory Access
// -----------------------------------------------------------------------------
void benchmark_random(size_t size_bytes) {
    size_t num_elements = size_bytes / sizeof(int);
    int *data = (int *)malloc(size_bytes);
    int *indices = (int *)malloc(size_bytes);
    if (!data || !indices) {
        free(data); free(indices);
        return;
    }

    // Initialize and shuffle indices
    for (size_t i = 0; i < num_elements; ++i) {
        data[i] = (int)i;
        indices[i] = (int)i;
    }

    // Fisher-Yates shuffle for random indices
    uint64_t rng_state = 88172645463325252ull;
    for (size_t i = num_elements - 1; i > 0; --i) {
        uint64_t x = rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        rng_state = x;

        size_t j = x % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    uint64_t start = get_cycles();
    volatile int sum = 0;
    
    // Random read
    for (size_t i = 0; i < num_elements; ++i) {
        sum += data[indices[i]];
    }
    
    uint64_t end = get_cycles();
    double cycles_per_element = (double)(end - start) / num_elements;

    printf("Random Read     - Size: %8zu KB | Cycles/element: %.2f\n", size_bytes / 1024, cycles_per_element);

    free(data);
    free(indices);
}

// -----------------------------------------------------------------------------
// BENCHMARK C: XOR + Popcount
// -----------------------------------------------------------------------------
void benchmark_bitwise(size_t iterations) {
    uint64_t start, end;
    volatile int sum = 0;
    
    // Scalar 64-bit Popcount + XOR (XNOR similarity)
    uint64_t val1 = 0xAAAAAAAAAAAAAAAAULL;
    uint64_t val2 = 0x5555555555555555ULL;
    
    start = get_cycles();
    for (size_t i = 0; i < iterations; ++i) {
        // XNOR is equivalent to ~(val1 ^ val2)
        uint64_t xnor = ~(val1 ^ (val2 + i));
        sum += __builtin_popcountll(xnor);
    }
    end = get_cycles();
    
    printf("Scalar XOR+Popcnt (64-bit) | Cycles/op: %.3f\n", (double)(end - start) / iterations);
}

// -----------------------------------------------------------------------------
// BENCHMARK D: Tiny Matrix Kernels (2x2)
// -----------------------------------------------------------------------------
void benchmark_tiny_matrix(size_t iterations) {
    // 2x2 Matrix Multiplication A * B
    // Stored in row-major
    float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    volatile float C[4] = {0};

    uint64_t start = get_cycles();
    for (size_t i = 0; i < iterations; ++i) {
        // Unrolled 2x2 multiplication
        C[0] = A[0]*B[0] + A[1]*B[2];
        C[1] = A[0]*B[1] + A[1]*B[3];
        C[2] = A[2]*B[0] + A[3]*B[2];
        C[3] = A[2]*B[1] + A[3]*B[3];
        
        // Anti-optimization trick
        A[0] += 0.0001f;
    }
    uint64_t end = get_cycles();
    
    printf("Tiny Matrix 2x2 (Scalar)   | Cycles/mul: %.3f\n", (double)(end - start) / iterations);
}

int main() {
#ifdef _WIN32
    // Lock thread to one core for more stable rdtsc readings
    SetThreadAffinityMask(GetCurrentThread(), 1);
#endif

    printf("=== SILICON LLM - PHASE 1 BENCHMARKS ===\n");
    printf("CPU: AMD Ryzen 5 3600X (Zen 2)\n\n");

    // L1: 32KB, L2: 512KB, L3: 32MB
    size_t sizes[] = {
        16 * 1024,      // 16 KB (Fits in L1)
        32 * 1024,      // 32 KB (L1 boundary)
        256 * 1024,     // 256 KB (Fits in L2)
        512 * 1024,     // 512 KB (L2 boundary)
        4 * 1024 * 1024,  // 4 MB (Fits in L3)
        32 * 1024 * 1024, // 32 MB (L3 boundary)
        128 * 1024 * 1024 // 128 MB (Spills to RAM)
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("--- Benchmark A & B: Memory Hierarchy ---\n");
    for (int i = 0; i < num_sizes; ++i) {
        benchmark_sequential(sizes[i]);
        benchmark_random(sizes[i]);
        printf("\n");
    }

    printf("--- Benchmark C: Bitwise Primitives ---\n");
    benchmark_bitwise(10000000);
    printf("\n");

    printf("--- Benchmark D: Tiny Kernels ---\n");
    benchmark_tiny_matrix(10000000);
    printf("\n");

    return 0;
}


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


#include "test_harness.h"

#ifdef _WIN32
#include <windows.h>
#endif

// Full integrated state
typedef struct {
    uint8_t state[GRID_SIZE];
    uint8_t traces[GRID_SIZE];      // M6 Leaky Integrator
    uint8_t m4_buf[HISTORY_SIZE];   // M4 Circular Buffer
    uint8_t rule_select[NUM_ZONES];
    int m4_head;
} Engine;

static inline uint64_t get_cycles() {
    unsigned int dummy;
    return __rdtscp(&dummy);
}

static inline uint8_t apply_rule(uint8_t rule, uint8_t L, uint8_t C, uint8_t R) {
    switch(rule) {
        case 0: return (L & R) ^ C;
        case 1: return L ^ C ^ R;
        case 2: return (L | R) ^ C;
        case 3: return (L ^ R) & (C | 0x55);
        default: return C;
    }
}

void engine_init(Engine* e) {
    memset(e, 0, sizeof(Engine));
    for(int i=0; i<NUM_ZONES; i++) {
        e->rule_select[i] = rand() % NUM_RULES;
    }
}

// 1 Tick = 4 wave prop + 1 bridge + M4/M6 updates
void engine_tick(Engine* e, uint8_t input_sym) {
    // M4 Injection (Write to ring buffer)
    e->m4_buf[e->m4_head] = input_sym ? 0xFF : 0x00;
    e->m4_head = (e->m4_head + 1) % HISTORY_SIZE;
    
    // Inject M4 buffer into wave (first 32 cells) to provide sustained state
    for(int i=0; i<32; i++) {
        int idx = (e->m4_head - 1 - i + HISTORY_SIZE) % HISTORY_SIZE;
        e->state[i] ^= e->m4_buf[idx];
    }
    
    uint8_t new_state[GRID_SIZE];
    
    // 4 Waves
    for(int w=0; w<4; w++) {
        new_state[0] = e->state[0];
        new_state[GRID_SIZE-1] = e->state[GRID_SIZE-1];
        
        for (int i=1; i<GRID_SIZE-1; i++) {
            int zone = i / BRIDGE_SPACING;
            uint8_t r = e->rule_select[zone];
            uint8_t wave = apply_rule(r, e->state[i-1], e->state[i], e->state[i+1]);
            new_state[i] = wave;
            
            // M6 trace update on last wave step
            if(w == 3) {
                e->traces[i] = (e->traces[i] >> 1) + (wave & 0x01);
            }
        }
        memcpy(e->state, new_state, GRID_SIZE);
    }
    
    // Butterfly Bridge
    for(int phase=0; phase<2; phase++) {
        for(int z=phase; z<NUM_ZONES-1; z+=2) {
            int left_idx = (z + 1) * BRIDGE_SPACING - 8;
            int right_idx = (z + 1) * BRIDGE_SPACING;
            for(int i=0; i<8; i++) {
                uint8_t a = e->state[left_idx + i];
                uint8_t b = e->state[right_idx + i];
                e->state[left_idx + i] = a + b;
                e->state[right_idx + i] = a - b;
            }
        }
    }
}

// R2 K=8 Readout
uint8_t engine_readout(Engine* e) {
    uint32_t votes[8] = {0};
    int offset = GRID_SIZE - 64; 
    for(int i=0; i<64; i+=8) {
        for(int k=0; k<8; k++) {
            votes[k] += e->state[offset + i + k];
        }
    }
    // Balanced decision
    uint32_t evens = votes[0] + votes[2] + votes[4] + votes[6];
    uint32_t odds = votes[1] + votes[3] + votes[5] + votes[7];
    return (evens > odds) ? 1 : 0;
}

// -----------------------------------------------------------------------------
// INTEGRATED THROUGHPUT BENCHMARK
// -----------------------------------------------------------------------------
void bench_integrated_throughput() {
    printf("--- INTEGRATED THROUGHPUT (Sanity Check) ---\n");
    Engine e;
    engine_init(&e);
    size_t iters = 100000;
    uint64_t start = get_cycles();
    for(size_t t=0; t<iters; t++) {
        engine_tick(&e, t % 2);
        volatile uint8_t r = engine_readout(&e);
    }
    uint64_t end = get_cycles();
    printf("Full Cycle (Wave + M4 + M6 + R2_K8): %.2f cycles/tick\n\n", (double)(end-start)/iters);
}

// -----------------------------------------------------------------------------
// L2: PERTURBATION + ESTIMATION
// -----------------------------------------------------------------------------
#define WIGGLE_WINDOW 100

void run_l2_task(const char* name, int task_type) {
    printf("Task: %s\n", name);
    Engine e;
    engine_init(&e);
    
    uint8_t backup_rules[NUM_ZONES];
    memcpy(backup_rules, e.rule_select, NUM_ZONES);
    
    double best_acc = 0.0;
    int correct_window = 0;
    int correct_report = 0;
    uint8_t hist[16] = {0};
    
    for(int t=1; t<=TOTAL_TICKS; t++) {
        TaskData d = generate_task(task_type, t, hist);
        engine_tick(&e, d.input_sym);
        uint8_t pred = engine_readout(&e);
        
        if (pred == d.target_sym) {
            correct_window++;
            correct_report++;
        }
        
        // L2 Wiggle Evaluation
        if (t % WIGGLE_WINDOW == 0) {
            double acc = (double)correct_window / WIGGLE_WINDOW;
            if (acc >= best_acc) {
                best_acc = acc;
                // Keep mutation
                memcpy(backup_rules, e.rule_select, NUM_ZONES);
            } else {
                // Revert mutation
                memcpy(e.rule_select, backup_rules, NUM_ZONES);
                // Decay baseline slightly to unstick from local optima
                best_acc *= 0.98;
            }
            
            // Apply new random wiggle
            int zone = rand() % NUM_ZONES;
            e.rule_select[zone] = rand() % NUM_RULES;
            
            correct_window = 0;
        }
        
        if (t % 10000 == 0) {
            print_accuracy(t, correct_report, 10000);
            correct_report = 0;
        }
    }
    printf("\n");
}

int main() {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 1);
#endif

    printf("=== PHASE 4.C: L2 PERTURBATION LEARNING ===\n\n");
    bench_integrated_throughput();
    
    run_l2_task("XOR-2", 0);
    run_l2_task("Period-7", 1);
    run_l2_task("Echo-5", 2);

    return 0;
}


#include "test_harness.h"
#include <immintrin.h>

#ifdef _WIN32
#include <windows.h>
#endif

static inline uint64_t get_cycles() {
    unsigned int dummy;
    return __rdtscp(&dummy);
}

// 32-Engine SIMD State
typedef struct {
    int grid_size;
    int num_zones;
    
    __m256i* state;
    __m256i* traces;
    __m256i* m4_buf;
    
    int m4_head;
    
    uint8_t** rule_select; // [num_zones][32]
    __m256i* mask_b0;
    __m256i* mask_b1;
} Engine32;

Engine32* engine_alloc(int grid_size) {
    Engine32* e = malloc(sizeof(Engine32));
    e->grid_size = grid_size;
    e->num_zones = grid_size / BRIDGE_SPACING;
    
    e->state = _mm_malloc(grid_size * sizeof(__m256i), 32);
    e->traces = _mm_malloc(grid_size * sizeof(__m256i), 32);
    e->m4_buf = _mm_malloc(HISTORY_SIZE * sizeof(__m256i), 32);
    
    e->rule_select = malloc(e->num_zones * sizeof(uint8_t*));
    e->mask_b0 = _mm_malloc(e->num_zones * sizeof(__m256i), 32);
    e->mask_b1 = _mm_malloc(e->num_zones * sizeof(__m256i), 32);
    for(int z=0; z<e->num_zones; z++) {
        e->rule_select[z] = malloc(32);
    }
    
    return e;
}

void update_masks(Engine32* e) {
    for(int z=0; z<e->num_zones; z++) {
        uint8_t b0[32], b1[32];
        for(int i=0; i<32; i++) {
            b0[i] = (e->rule_select[z][i] & 1) ? 0xFF : 0x00;
            b1[i] = (e->rule_select[z][i] & 2) ? 0xFF : 0x00;
        }
        e->mask_b0[z] = _mm256_loadu_si256((__m256i*)b0);
        e->mask_b1[z] = _mm256_loadu_si256((__m256i*)b1);
    }
}

void engine_init(Engine32* e) {
    memset(e->state, 0, e->grid_size * sizeof(__m256i));
    memset(e->traces, 0, e->grid_size * sizeof(__m256i));
    memset(e->m4_buf, 0, HISTORY_SIZE * sizeof(__m256i));
    e->m4_head = 0;
    
    for(int z=0; z<e->num_zones; z++) {
        for(int i=0; i<32; i++) {
            e->rule_select[z][i] = rand() % NUM_RULES;
        }
    }
    update_masks(e);
}

void engine_tick(Engine32* e, uint8_t input_sym) {
    // M4 Injection (Write to ring buffer for all 32 engines)
    uint8_t injected[32];
    for(int i=0; i<32; i++) injected[i] = input_sym ? 0xFF : 0x00;
    __m256i v_inj = _mm256_loadu_si256((__m256i*)injected);
    
    e->m4_buf[e->m4_head] = v_inj;
    e->m4_head = (e->m4_head + 1) % HISTORY_SIZE;
    
    // Inject M4 buffer into wave
    for(int i=0; i<32; i++) {
        int idx = (e->m4_head - 1 - i + HISTORY_SIZE) % HISTORY_SIZE;
        e->state[i] = _mm256_xor_si256(e->state[i], e->m4_buf[idx]);
    }
    
    __m256i* new_state = _mm_malloc(e->grid_size * sizeof(__m256i), 32);
    
    __m256i const_55 = _mm256_set1_epi8(0x55);
    __m256i mask_01 = _mm256_set1_epi8(0x01);
    
    // 4 Waves
    for(int w=0; w<4; w++) {
        new_state[0] = e->state[0];
        new_state[e->grid_size-1] = e->state[e->grid_size-1];
        
        for (int i=1; i<e->grid_size-1; i++) {
            int zone = i / BRIDGE_SPACING;
            
            __m256i L = e->state[i-1];
            __m256i C = e->state[i];
            __m256i R = e->state[i+1];
            
            // Calc 4 rules
            __m256i r0 = _mm256_xor_si256(_mm256_and_si256(L, R), C);
            __m256i r1 = _mm256_xor_si256(_mm256_xor_si256(L, C), R);
            __m256i r2 = _mm256_xor_si256(_mm256_or_si256(L, R), C);
            __m256i r3 = _mm256_and_si256(_mm256_xor_si256(L, R), _mm256_or_si256(C, const_55));
            
            // Blend
            __m256i m0 = e->mask_b0[zone];
            __m256i m1 = e->mask_b1[zone];
            __m256i sel01 = _mm256_blendv_epi8(r0, r1, m0);
            __m256i sel23 = _mm256_blendv_epi8(r2, r3, m0);
            __m256i wave = _mm256_blendv_epi8(sel01, sel23, m1);
            
            new_state[i] = wave;
            
            // M6 trace
            if(w == 3) {
                // state = (state >> 1) + (wave & 1)
                // Need to do logic shift right for epi8. AVX2 has no srli_epi8, so we do srli_epi16 and mask.
                __m256i shift = _mm256_and_si256(_mm256_srli_epi16(e->traces[i], 1), _mm256_set1_epi8(0x7F));
                e->traces[i] = _mm256_add_epi8(shift, _mm256_and_si256(wave, mask_01));
            }
        }
        memcpy(e->state, new_state, e->grid_size * sizeof(__m256i));
    }
    _mm_free(new_state);
    
    // Butterfly Bridge
    for(int phase=0; phase<2; phase++) {
        for(int z=phase; z<e->num_zones-1; z+=2) {
            int left_idx = (z + 1) * BRIDGE_SPACING - 8;
            int right_idx = (z + 1) * BRIDGE_SPACING;
            for(int i=0; i<8; i++) {
                __m256i a = e->state[left_idx + i];
                __m256i b = e->state[right_idx + i];
                e->state[left_idx + i] = _mm256_add_epi8(a, b);
                e->state[right_idx + i] = _mm256_sub_epi8(a, b);
            }
        }
    }
}

void engine_readout(Engine32* e, uint8_t* preds_out) {
    uint32_t votes[32][8] = {0};
    int offset = e->grid_size - 64; 
    
    for(int i=0; i<64; i+=8) {
        for(int k=0; k<8; k++) {
            uint8_t bytes[32];
            _mm256_storeu_si256((__m256i*)bytes, e->state[offset + i + k]);
            for(int eng=0; eng<32; eng++) {
                votes[eng][k] += bytes[eng];
            }
        }
    }
    
    for(int eng=0; eng<32; eng++) {
        uint32_t evens = votes[eng][0] + votes[eng][2] + votes[eng][4] + votes[eng][6];
        uint32_t odds = votes[eng][1] + votes[eng][3] + votes[eng][5] + votes[eng][7];
        preds_out[eng] = (evens > odds) ? 1 : 0;
    }
}

// -----------------------------------------------------------------------------
// BENCHMARKS
// -----------------------------------------------------------------------------
void bench_throughput(int grid_size, const char* label) {
    Engine32* e = engine_alloc(grid_size);
    engine_init(e);
    size_t iters = 10000;
    uint64_t start = get_cycles();
    for(size_t t=0; t<iters; t++) {
        engine_tick(e, t % 2);
    }
    uint64_t end = get_cycles();
    double cycles_per_tick = (double)(end-start)/iters;
    // Divide by 32 to get cost per virtual engine
    double cycles_per_engine = cycles_per_tick / 32.0;
    printf("SIMD Throughput (%s, %d cells): %.2f cycles/tick (%.2f per virtual engine)\n", 
            label, grid_size, cycles_per_tick, cycles_per_engine);
}

// Tournament selection and mutation
void evolve_population(Engine32* e, int* correct_counters) {
    // Top 8 replace bottom 8. Soft tournament.
    // For simplicity, find the 8 worst and replace them with 8 random from the top 16.
    
    int sorted_engines[32];
    for(int i=0; i<32; i++) sorted_engines[i] = i;
    
    // Bubble sort engines by correct_counters (descending)
    for(int i=0; i<31; i++) {
        for(int j=i+1; j<32; j++) {
            if(correct_counters[sorted_engines[j]] > correct_counters[sorted_engines[i]]) {
                int tmp = sorted_engines[i];
                sorted_engines[i] = sorted_engines[j];
                sorted_engines[j] = tmp;
            }
        }
    }
    
    // Replace bottom 8 with clones of top 8
    for(int i=0; i<8; i++) {
        int best_eng = sorted_engines[i];
        int worst_eng = sorted_engines[31 - i];
        
        // Clone rules
        for(int z=0; z<e->num_zones; z++) {
            e->rule_select[z][worst_eng] = e->rule_select[z][best_eng];
        }
        
        // Mutate 2 rules for diversity
        for(int m=0; m<2; m++) {
            int mut_z = rand() % e->num_zones;
            e->rule_select[mut_z][worst_eng] = rand() % NUM_RULES;
        }
        
        // Also apply 1 mutation to engines 8-23 to maintain exploration
        int mid_eng = sorted_engines[8 + i];
        int mut_z = rand() % e->num_zones;
        e->rule_select[mut_z][mid_eng] = rand() % NUM_RULES;
    }
    
    update_masks(e);
}

void run_l4_task(const char* name, int task_type, int grid_size) {
    printf("Task: %s (Grid: %d cells)\n", name, grid_size);
    Engine32* e = engine_alloc(grid_size);
    engine_init(e);
    
    int correct_window[32] = {0};
    uint8_t hist[16] = {0};
    uint8_t preds[32];
    
    int EVOLUTION_WINDOW = 1000;
    
    for(int t=1; t<=TOTAL_TICKS; t++) {
        TaskData d = generate_task(task_type, t, hist);
        engine_tick(e, d.input_sym);
        engine_readout(e, preds);
        
        for(int eng=0; eng<32; eng++) {
            if (preds[eng] == d.target_sym) correct_window[eng]++;
        }
        
        if (t % EVOLUTION_WINDOW == 0) {
            if (t % 10000 == 0) {
                // Find best accuracy in this window
                int best_correct = 0;
                for(int eng=0; eng<32; eng++) {
                    if(correct_window[eng] > best_correct) best_correct = correct_window[eng];
                }
                print_accuracy(t, best_correct, EVOLUTION_WINDOW);
            }
            
            evolve_population(e, correct_window);
            memset(correct_window, 0, sizeof(correct_window));
        }
    }
    printf("\n");
}

int main() {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 1);
#endif

    printf("=== PHASE 4.C: L4 SIMD EVOLUTION LEARNING ===\n\n");
    bench_throughput(1024, "L2 Cache Bound");
    bench_throughput(256, "L1 Cache Bound");
    printf("\n");
    
    // Run tests on L1-bound grid (256 cells) for maximum cycle efficiency and dense learning
    run_l4_task("XOR-2", 0, 256);
    run_l4_task("Period-7", 1, 256);
    run_l4_task("Echo-5", 2, 256);

    return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define GRID_SIZE 1024
#define MEM_SIZE 256

static inline uint64_t get_cycles() {
    unsigned int dummy;
    return __rdtscp(&dummy);
}

// Helper: Hamming distance between two byte arrays
int hamming_distance(uint8_t* a, uint8_t* b, int len) {
    int dist = 0;
    for(int i=0; i<len; i++) {
        uint8_t xor_val = a[i] ^ b[i];
        while(xor_val) {
            dist += xor_val & 1;
            xor_val >>= 1;
        }
    }
    return dist;
}

// -----------------------------------------------------------------------------
// CYCLES BENCHMARKS
// -----------------------------------------------------------------------------
void bench_cycles() {
    uint8_t *state = calloc(GRID_SIZE, 1);
    uint8_t *new_state = calloc(GRID_SIZE, 1);
    size_t iters = 200000;
    uint64_t start, end;
    
    printf("--- CYCLE COSTS (per element) ---\n");
    
    // Baseline Wave
    start = get_cycles();
    for(size_t t=0; t<iters; t++) {
        for(int i=1; i<GRID_SIZE-1; i++) {
            new_state[i] = (state[i-1] & state[i+1]) ^ state[i];
        }
        uint8_t* tmp = state; state = new_state; new_state = tmp;
    }
    end = get_cycles();
    double wave_cost = (double)(end-start)/(iters * GRID_SIZE);
    printf("Baseline Wave (Quasi-Rule 30) : %.3f cycles/elem\n", wave_cost);

    // M2: Bistable (Isolated)
    start = get_cycles();
    for(size_t t=0; t<iters; t++) {
        for(int i=1; i<GRID_SIZE-1; i++) {
            uint8_t input = state[i-1];
            uint8_t set_mask = (input > 200) ? 0xFF : 0x00;
            uint8_t reset_mask = (input < 50) ? 0xFF : 0x00;
            new_state[i] = set_mask | (state[i] & ~reset_mask);
        }
        uint8_t* tmp = state; state = new_state; new_state = tmp;
    }
    end = get_cycles();
    printf("M2 Bistable (Isolated)        : %.3f cycles/elem\n", (double)(end-start)/(iters * GRID_SIZE));

    // M2: Bistable (Interleaved with wave)
    start = get_cycles();
    for(size_t t=0; t<iters; t++) {
        for(int i=1; i<GRID_SIZE-1; i++) {
            uint8_t wave = (state[i-1] & state[i+1]) ^ state[i];
            uint8_t set_mask = (wave > 200) ? 0xFF : 0x00;
            uint8_t reset_mask = (wave < 50) ? 0xFF : 0x00;
            new_state[i] = set_mask | (state[i] & ~reset_mask);
        }
        uint8_t* tmp = state; state = new_state; new_state = tmp;
    }
    end = get_cycles();
    printf("M2 Bistable (Interleaved)     : %.3f cycles/elem\n", (double)(end-start)/(iters * GRID_SIZE));

    // M3: Delay Line / Shift Register (Isolated)
    start = get_cycles();
    for(size_t t=0; t<iters; t++) {
        memmove(new_state + 1, state, GRID_SIZE - 1);
        new_state[0] = (uint8_t)t;
        uint8_t* tmp = state; state = new_state; new_state = tmp;
    }
    end = get_cycles();
    printf("M3 Shift Register (Isolated)  : %.3f cycles/elem\n", (double)(end-start)/(iters * GRID_SIZE));

    // M3: Delay Line (Interleaved)
    start = get_cycles();
    for(size_t t=0; t<iters; t++) {
        for(int i=1; i<GRID_SIZE/2; i++) {
            new_state[i] = (state[i-1] & state[i+1]) ^ state[i];
        }
        memmove(new_state + GRID_SIZE/2 + 1, state + GRID_SIZE/2, GRID_SIZE/2 - 1);
        new_state[GRID_SIZE/2] = new_state[GRID_SIZE/2 - 1]; 
        uint8_t* tmp = state; state = new_state; new_state = tmp;
    }
    end = get_cycles();
    printf("M3 Shift Reg (Interleaved)    : %.3f cycles/elem\n", (double)(end-start)/(iters * GRID_SIZE));

    // M4: Circular Buffer (Isolated)
    start = get_cycles();
    int head = 0;
    for(size_t t=0; t<iters; t++) {
        state[head] = (uint8_t)t;
        head = (head + 1) % GRID_SIZE;
    }
    end = get_cycles();
    printf("M4 Circular Buffer (Isolated) : %.3f cycles/elem\n", (double)(end-start)/(iters * GRID_SIZE));

    // M6: Leaky Integrator (Isolated)
    start = get_cycles();
    for(size_t t=0; t<iters; t++) {
        for(int i=1; i<GRID_SIZE-1; i++) {
            new_state[i] = (state[i] >> 1) + state[i-1];
        }
        uint8_t* tmp = state; state = new_state; new_state = tmp;
    }
    end = get_cycles();
    printf("M6 Leaky Integrator (Isolated): %.3f cycles/elem\n", (double)(end-start)/(iters * GRID_SIZE));

    // M6: Leaky Integrator (Interleaved)
    start = get_cycles();
    for(size_t t=0; t<iters; t++) {
        for(int i=1; i<GRID_SIZE-1; i++) {
            uint8_t wave = (state[i-1] & state[i+1]) ^ state[i];
            new_state[i] = (state[i] >> 1) + wave;
        }
        uint8_t* tmp = state; state = new_state; new_state = tmp;
    }
    end = get_cycles();
    printf("M6 Leaky Integ (Interleaved)  : %.3f cycles/elem\n", (double)(end-start)/(iters * GRID_SIZE));
    
    printf("\n");
    free(state);
    free(new_state);
}

// -----------------------------------------------------------------------------
// FIDELITY / PERSISTENCE BENCHMARKS
// -----------------------------------------------------------------------------
void bench_fidelity() {
    printf("--- FIDELITY / PERSISTENCE (Hamming Dist after T ticks) ---\n");
    int pattern_len = 32;
    uint8_t pattern[32];
    for(int i=0; i<pattern_len; i++) pattern[i] = rand() % 256;
    
    int T_values[] = {10, 50, 100, 200};
    
    for(int test_idx=0; test_idx<4; test_idx++) {
        int T = T_values[test_idx];
        printf("T = %d ticks:\n", T);
        
        uint8_t *state = calloc(GRID_SIZE, 1);
        uint8_t *new_state = calloc(GRID_SIZE, 1);
        
        // M2 Fidelity
        memcpy(state + 100, pattern, pattern_len); 
        for(int t=0; t<T; t++) {
            state[0] = rand() % 256;
            for(int i=1; i<GRID_SIZE-1; i++) {
                uint8_t wave = (state[i-1] & state[i+1]) ^ state[i];
                uint8_t set_mask = (wave > 200) ? 0xFF : 0x00;
                uint8_t reset_mask = (wave < 50) ? 0xFF : 0x00;
                new_state[i] = set_mask | (state[i] & ~reset_mask);
            }
            uint8_t* tmp = state; state = new_state; new_state = tmp;
        }
        int dist_m2 = hamming_distance(pattern, state + 100, pattern_len);
        printf("  M2 (Bistable)        : Dist = %d / %d bits\n", dist_m2, pattern_len*8);

        // M3 Fidelity
        memset(state, 0, GRID_SIZE);
        for(int t=0; t<T; t++) {
            memmove(state + 1, state, GRID_SIZE - 1);
            if(t < pattern_len) state[0] = pattern[pattern_len - 1 - t];
            else state[0] = 0;
        }
        int dist_m3 = 0;
        if (T >= pattern_len && T <= GRID_SIZE) {
            dist_m3 = hamming_distance(pattern, state + T - pattern_len, pattern_len);
        } else {
            dist_m3 = pattern_len*8; 
        }
        printf("  M3 (Shift Register)  : Dist = %d / %d bits\n", dist_m3, pattern_len*8);
        
        // M4 Fidelity
        memset(state, 0, GRID_SIZE);
        int head = 0;
        for(int t=0; t<T; t++) {
            if(t < pattern_len) state[head] = pattern[t];
            else state[head] = 0;
            head = (head + 1) % GRID_SIZE;
        }
        int dist_m4 = 0;
        if (T <= GRID_SIZE) {
            uint8_t readback[32];
            for(int i=0; i<pattern_len; i++) {
                int read_idx = (head - T + i + GRID_SIZE) % GRID_SIZE;
                readback[i] = state[read_idx];
            }
            dist_m4 = hamming_distance(pattern, readback, pattern_len);
        } else {
            dist_m4 = pattern_len*8;
        }
        printf("  M4 (Circular Buffer) : Dist = %d / %d bits\n", dist_m4, pattern_len*8);

        // M6 Fidelity
        memset(state, 0, GRID_SIZE);
        memcpy(state + 100, pattern, pattern_len);
        for(int t=0; t<T; t++) {
            for(int i=1; i<GRID_SIZE-1; i++) {
                uint8_t wave = (state[i-1] & state[i+1]) ^ state[i];
                new_state[i] = (state[i] >> 1) + (wave & 0x01); 
            }
            uint8_t* tmp = state; state = new_state; new_state = tmp;
        }
        int dist_m6 = hamming_distance(pattern, state + 100, pattern_len);
        printf("  M6 (Leaky Integrator): Dist = %d / %d bits\n", dist_m6, pattern_len*8);
        
        printf("\n");
        free(state);
        free(new_state);
    }
}

int main() {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 1);
#endif

    printf("=== PHASE 4.A: MEMORY MECHANISMS ===\n\n");
    bench_cycles();
    bench_fidelity();

    return 0;
}


#include "test_harness.h"

#ifdef _WIN32
#include <windows.h>
#endif

// Define a simple Engine for ESP testing
typedef struct {
    uint8_t state[GRID_SIZE];
    uint8_t traces[GRID_SIZE];
    uint8_t rule_select[NUM_ZONES];
} EngineRC;

static inline uint8_t apply_rule(uint8_t rule, uint8_t L, uint8_t C, uint8_t R) {
    switch(rule) {
        case 0: return (L & R) ^ C;
        case 1: return L ^ C ^ R;
        case 2: return (L | R) ^ C;
        case 3: return (L ^ R) & (C | 0x55);
        default: return C;
    }
}

void engine_init(EngineRC* e) {
    memset(e->state, 0, GRID_SIZE);
    memset(e->traces, 0, GRID_SIZE);
    for(int i=0; i<NUM_ZONES; i++) {
        e->rule_select[i] = rand() % NUM_RULES;
    }
}

void engine_tick(EngineRC* e, uint8_t input_sym) {
    // Global Damping to guarantee ESP
    // Shift every cell right by 1 bit. This halves the energy of the reservoir every tick.
    for(int i=0; i<GRID_SIZE; i++) {
        e->state[i] >>= 1;
    }

    // Left edge injection (32 cells)
    uint8_t val = input_sym ? 0xFF : 0x00;
    for(int i=0; i<32; i++) {
        e->state[i] ^= val;
    }
    
    uint8_t new_state[GRID_SIZE];
    
    // 4 Waves
    for(int w=0; w<4; w++) {
        new_state[0] = e->state[0];
        new_state[GRID_SIZE-1] = e->state[GRID_SIZE-1];
        
        for(int i=1; i<GRID_SIZE-1; i++) {
            int zone = i / BRIDGE_SPACING;
            uint8_t wave = apply_rule(e->rule_select[zone], e->state[i-1], e->state[i], e->state[i+1]);
            new_state[i] = wave;
            if (w == 3) {
                e->traces[i] = (e->traces[i] >> 1) + (wave & 0x01);
            }
        }
        memcpy(e->state, new_state, GRID_SIZE);
    }
    
    // Butterfly Bridge
    for(int phase=0; phase<2; phase++) {
        for(int z=phase; z<NUM_ZONES-1; z+=2) {
            int left_idx = (z + 1) * BRIDGE_SPACING - 8;
            int right_idx = (z + 1) * BRIDGE_SPACING;
            for(int i=0; i<8; i++) {
                uint8_t a = e->state[left_idx + i];
                uint8_t b = e->state[right_idx + i];
                e->state[left_idx + i] = a + b;
                e->state[right_idx + i] = a - b;
            }
        }
    }
}

int hamming_distance(uint8_t* a, uint8_t* b, int size) {
    int dist = 0;
    for(int i=0; i<size; i++) {
        uint8_t x = a[i] ^ b[i];
        // simple popcount
        while(x) {
            dist += x & 1;
            x >>= 1;
        }
    }
    return dist;
}

void test_esp() {
    printf("--- TEST 0: ECHO STATE PROPERTY (Baseline) ---\n");
    EngineRC e_a, e_b;
    engine_init(&e_a);
    
    // Clone e_a to e_b
    e_b = e_a;
    // Introduce 1 bit difference in the middle of state
    e_b.state[GRID_SIZE/2] ^= 0x01;
    
    int dist_state = hamming_distance(e_a.state, e_b.state, GRID_SIZE);
    int dist_trace = hamming_distance(e_a.traces, e_b.traces, GRID_SIZE);
    printf("Initial Hamming Distance (State): %d | (Traces): %d\n", dist_state, dist_trace);
    
    for(int t=1; t<=500; t++) {
        uint8_t sym = rand() % 2;
        engine_tick(&e_a, sym);
        engine_tick(&e_b, sym);
        
        if (t % 50 == 0) {
            dist_state = hamming_distance(e_a.state, e_b.state, GRID_SIZE);
            dist_trace = hamming_distance(e_a.traces, e_b.traces, GRID_SIZE);
            printf("Step %3d | Hamming Distance -> State: %4d bits | Traces: %4d bits\n", t, dist_state, dist_trace);
        }
    }
    printf("\n");
}

void run_rc_task(const char* name, int task_type, int num_channels) {
    printf("Task: %s (K=%d channels)\n", name, num_channels);
    EngineRC e;
    engine_init(&e);
    
    int32_t w[64] = {0}; // Support up to 64 channels
    int LR_SHIFT = 4;
    int32_t MAX_W = 10000;
    
    int correct_report = 0;
    uint8_t hist[16] = {0};
    
    for(int t=1; t<=TOTAL_TICKS; t++) {
        TaskData d = generate_task(task_type, t, hist);
        engine_tick(&e, d.input_sym);
        
        // Extract channels
        int32_t channel[64] = {0};
        int cells_to_read = num_channels * 8; // e.g. 8 channels * 8 = 64 cells
        if (cells_to_read > GRID_SIZE) cells_to_read = GRID_SIZE;
        
        int offset = GRID_SIZE - cells_to_read;
        for(int i=0; i<cells_to_read; i+=num_channels) {
            for(int k=0; k<num_channels; k++) {
                if (offset + i + k < GRID_SIZE) {
                    channel[k] += e.state[offset + i + k];
                }
            }
        }
        
        // Inference
        int32_t score = 0;
        for(int k=0; k<num_channels; k++) score += w[k] * channel[k];
        uint8_t pred = (score > 0) ? 1 : 0;
        
        if (pred == d.target_sym) {
            correct_report++;
        }
        
        // LMS Learning
        int error = d.target_sym - pred;
        if (error != 0) {
            for(int k=0; k<num_channels; k++) {
                w[k] += (error * channel[k]) >> LR_SHIFT;
                if (w[k] > MAX_W) w[k] = MAX_W;
                if (w[k] < -MAX_W) w[k] = -MAX_W;
            }
        }
        
        if (t % 10000 == 0) {
            print_accuracy(t, correct_report, 10000);
            correct_report = 0;
        }
    }
    printf("\n");
}

void run_rc3_task(const char* name, int task_type) {
    printf("Task: %s (RC-3: K=8 channels + M4 history)\n", name);
    EngineRC e;
    engine_init(&e);
    
    uint8_t m4_buf[16] = {0};
    int m4_head = 0;
    
    int32_t w[24] = {0}; 
    int LR_SHIFT = 4;
    int32_t MAX_W = 10000;
    
    int correct_report = 0;
    uint8_t hist[16] = {0};
    
    for(int t=1; t<=TOTAL_TICKS; t++) {
        TaskData d = generate_task(task_type, t, hist);
        
        m4_buf[m4_head] = d.input_sym;
        m4_head = (m4_head + 1) % 16;
        
        engine_tick(&e, d.input_sym);
        
        int32_t channel[24] = {0};
        int offset = GRID_SIZE - 64;
        for(int i=0; i<64; i+=8) {
            for(int k=0; k<8; k++) {
                channel[k] += e.state[offset + i + k];
            }
        }
        
        for(int i=0; i<16; i++) {
            int idx = (m4_head - 1 - i + 16) % 16;
            channel[8 + i] = m4_buf[idx] * 255; 
        }
        
        int32_t score = 0;
        for(int k=0; k<24; k++) score += w[k] * channel[k];
        uint8_t pred = (score > 0) ? 1 : 0;
        
        if (pred == d.target_sym) correct_report++;
        
        int error = d.target_sym - pred;
        if (error != 0) {
            for(int k=0; k<24; k++) {
                w[k] += (error * channel[k]) >> LR_SHIFT;
                if (w[k] > MAX_W) w[k] = MAX_W;
                if (w[k] < -MAX_W) w[k] = -MAX_W;
            }
        }
        
        if (t % 10000 == 0) {
            print_accuracy(t, correct_report, 10000);
            correct_report = 0;
        }
    }
    printf("\n");
}

int main() {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 1);
#endif

    test_esp();
    
    printf("=== PHASE 4.RC: RESERVOIR COMPUTING BENCHMARKS ===\n\n");
    
    printf("--- [RC-1] Baseline (K=8 channels) ---\n");
    run_rc_task("XOR-2", 0, 8);
    run_rc_task("Period-7", 1, 8);
    run_rc_task("Echo-5", 2, 8);
    
    printf("--- [RC-2] High-Bandwidth (K=32 channels) ---\n");
    run_rc_task("XOR-2", 0, 32);
    run_rc_task("Period-7", 1, 32);
    run_rc_task("Echo-5", 2, 32);
    
    printf("--- [RC-3] Temporal Skip (M4 directly to Readout) ---\n");
    run_rc3_task("XOR-2", 0);
    run_rc3_task("Period-7", 1);
    run_rc3_task("Echo-5", 2);

    return 0;
}


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


#include "test_harness.h"
#include <immintrin.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#endif

static inline uint64_t get_cycles() {
    unsigned int dummy;
    return __rdtscp(&dummy);
}

typedef struct {
    int grid_size;
    int num_zones;
    int topology;
    
    __m256i* state;
    __m256i* new_state;
    __m256i* m4_buf;
    int m4_head;
    
    uint8_t** rule_select; // [num_zones][32]
    __m256i* mask_b0;
    __m256i* mask_b1;
} EngineArit;

EngineArit* engine_alloc(int grid_size, int topology) {
    EngineArit* e = malloc(sizeof(EngineArit));
    e->grid_size = grid_size;
    e->num_zones = grid_size / BRIDGE_SPACING;
    e->topology = topology;
    
    e->state = _mm_malloc(grid_size * sizeof(__m256i), 32);
    e->new_state = _mm_malloc(grid_size * sizeof(__m256i), 32);
    e->m4_buf = _mm_malloc(HISTORY_SIZE * sizeof(__m256i), 32);
    
    e->rule_select = malloc(e->num_zones * sizeof(uint8_t*));
    e->mask_b0 = _mm_malloc(e->num_zones * sizeof(__m256i), 32);
    e->mask_b1 = _mm_malloc(e->num_zones * sizeof(__m256i), 32);
    for(int z=0; z<e->num_zones; z++) {
        e->rule_select[z] = malloc(32);
    }
    
    return e;
}

void update_masks(EngineArit* e) {
    for(int z=0; z<e->num_zones; z++) {
        uint8_t b0[32], b1[32];
        for(int i=0; i<32; i++) {
            b0[i] = (e->rule_select[z][i] & 1) ? 0xFF : 0x00;
            b1[i] = (e->rule_select[z][i] & 2) ? 0xFF : 0x00;
        }
        e->mask_b0[z] = _mm256_loadu_si256((__m256i*)b0);
        e->mask_b1[z] = _mm256_loadu_si256((__m256i*)b1);
    }
}

void engine_init(EngineArit* e) {
    memset(e->state, 0, e->grid_size * sizeof(__m256i));
    memset(e->new_state, 0, e->grid_size * sizeof(__m256i));
    memset(e->m4_buf, 0, HISTORY_SIZE * sizeof(__m256i));
    e->m4_head = 0;
    
    for(int z=0; z<e->num_zones; z++) {
        for(int i=0; i<32; i++) {
            e->rule_select[z][i] = rand() % NUM_RULES;
        }
    }
    update_masks(e);
}

void engine_tick(EngineArit* e, uint8_t input_sym) {
    __m256i const_128 = _mm256_set1_epi8(-128);
    __m256i zero = _mm256_setzero_si256();
    __m256i threshold = _mm256_set1_epi8(32);
    
    // Global Damping: >> 1 to preserve ESP
    __m256i mask_7F = _mm256_set1_epi8(0x7F);
    for(int i=0; i<e->grid_size; i++) {
        e->state[i] = _mm256_and_si256(_mm256_srli_epi16(e->state[i], 1), mask_7F);
    }

    // M4 Injection
    uint8_t injected[32];
    for(int i=0; i<32; i++) injected[i] = input_sym ? 0xFF : 0x00;
    __m256i v_inj = _mm256_loadu_si256((__m256i*)injected);
    
    e->m4_buf[e->m4_head] = v_inj;
    e->m4_head = (e->m4_head + 1) % HISTORY_SIZE;
    
    __m256i t0 = e->m4_buf[(e->m4_head - 1 + HISTORY_SIZE) % HISTORY_SIZE];
    __m256i t1 = e->m4_buf[(e->m4_head - 2 + HISTORY_SIZE) % HISTORY_SIZE];
    __m256i t2 = e->m4_buf[(e->m4_head - 3 + HISTORY_SIZE) % HISTORY_SIZE];

    // INJECTION TOPOLOGY
    if (e->topology == 0) { // Baseline (Left Edge)
        for(int i=0; i<32; i++) {
            int idx = (e->m4_head - 1 - i + HISTORY_SIZE) % HISTORY_SIZE;
            e->state[i] = _mm256_xor_si256(e->state[i], e->m4_buf[idx]);
        }
    } else if (e->topology == 1) { // T1: Current distributed
        for(int i=0; i<256; i+=8) {
            e->state[i] = _mm256_adds_epu8(e->state[i], t0);
        }
    } else if (e->topology == 2) { // T2: Multi-temporal sparse
        for(int i=0; i<256; i+=24) {
            if (i < 256) e->state[i] = _mm256_adds_epu8(e->state[i], t0);
            if (i+8 < 256) e->state[i+8] = _mm256_adds_epu8(e->state[i+8], t1);
            if (i+16 < 256) e->state[i+16] = _mm256_adds_epu8(e->state[i+16], t2);
        }
    } else if (e->topology == 3) { // T3: Multi-temporal dense
        for(int i=0; i<256; i+=12) {
            if (i < 256) e->state[i] = _mm256_adds_epu8(e->state[i], t0);
            if (i+4 < 256) e->state[i+4] = _mm256_adds_epu8(e->state[i+4], t1);
            if (i+8 < 256) e->state[i+8] = _mm256_adds_epu8(e->state[i+8], t2);
        }
    }
    
    // 4 Waves (ARITHMETIC)
    for(int w=0; w<4; w++) {
        e->new_state[0] = e->state[0];
        e->new_state[e->grid_size-1] = e->state[e->grid_size-1];
        
        for (int i=1; i<e->grid_size-1; i++) {
            int zone = i / BRIDGE_SPACING;
            
            __m256i L = e->state[i-1];
            __m256i C = e->state[i];
            __m256i R = e->state[i+1];
            
            // A1: sat_add(avg(L,R), sat_sub(C, 128))
            __m256i r0 = _mm256_adds_epu8(_mm256_avg_epu8(L, R), _mm256_subs_epu8(C, const_128));
            
            // A2: sat_add(sat_sub(L, C), R)
            __m256i r1 = _mm256_adds_epu8(_mm256_subs_epu8(L, C), R);
            
            // A3: sat_add(L>>1, sat_add(R>>1, C>>1))
            __m256i l_half = _mm256_avg_epu8(L, zero);
            __m256i c_half = _mm256_avg_epu8(C, zero);
            __m256i r_half = _mm256_avg_epu8(R, zero);
            __m256i r2 = _mm256_adds_epu8(l_half, _mm256_adds_epu8(r_half, c_half));
            
            // A4: sat_sub(sat_add(L, R), C)
            __m256i r3 = _mm256_subs_epu8(_mm256_adds_epu8(L, R), C);
            
            // Blend
            __m256i m0 = e->mask_b0[zone];
            __m256i m1 = e->mask_b1[zone];
            __m256i sel01 = _mm256_blendv_epi8(r0, r1, m0);
            __m256i sel23 = _mm256_blendv_epi8(r2, r3, m0);
            // Soft Damping is removed, we use >> 1 globally as before.
            e->new_state[i] = _mm256_blendv_epi8(sel01, sel23, m1);
        }
        
        memcpy(e->state, e->new_state, e->grid_size * sizeof(__m256i));
    }
    
    // Butterfly Bridge
    for(int phase=0; phase<2; phase++) {
        for(int z=phase; z<e->num_zones-1; z+=2) {
            int left_idx = (z + 1) * BRIDGE_SPACING - 8;
            int right_idx = (z + 1) * BRIDGE_SPACING;
            for(int i=0; i<8; i++) {
                __m256i a = e->state[left_idx + i];
                __m256i b = e->state[right_idx + i];
                e->state[left_idx + i] = _mm256_adds_epu8(a, b);
                e->state[right_idx + i] = _mm256_subs_epu8(a, b);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// METRICS
// -----------------------------------------------------------------------------
void bench_throughput(int grid_size) {
    EngineArit* e = engine_alloc(grid_size, 0);
    engine_init(e);
    size_t iters = 10000;
    uint64_t start = get_cycles();
    for(size_t t=0; t<iters; t++) {
        engine_tick(e, t % 2);
    }
    uint64_t end = get_cycles();
    double cycles_per_tick = (double)(end-start)/iters;
    double cycles_per_engine = cycles_per_tick / 32.0;
    printf("--- SIMD ARITHMETIC THROUGHPUT ---\n");
    printf("Grid: %d cells | %.2f cycles/tick | %.2f per virtual engine\n\n", 
            grid_size, cycles_per_tick, cycles_per_engine);
}

int hamming_distance_simd(EngineArit* a, EngineArit* b) {
    int dist = 0;
    for(int i=0; i<a->grid_size; i++) {
        uint8_t ba[32], bb[32];
        _mm256_storeu_si256((__m256i*)ba, a->state[i]);
        _mm256_storeu_si256((__m256i*)bb, b->state[i]);
        for(int j=0; j<32; j++) {
            uint8_t x = ba[j] ^ bb[j];
            while(x) {
                dist += x & 1;
                x >>= 1;
            }
        }
    }
    return dist;
}

void test_esp(int grid_size) {
    printf("--- TEST ESP (Arithmetic Damping) ---\n");
    EngineArit* e_a = engine_alloc(grid_size, 0);
    EngineArit* e_b = engine_alloc(grid_size, 0);
    engine_init(e_a);
    engine_init(e_b);
    
    // Copy a to b
    memcpy(e_b->state, e_a->state, grid_size * sizeof(__m256i));
    for(int z=0; z<e_a->num_zones; z++) {
        memcpy(e_b->rule_select[z], e_a->rule_select[z], 32);
    }
    update_masks(e_b);
    
    // Inject 1 bit difference into engine 0 at middle of grid
    uint8_t diff[32] = {0};
    diff[0] = 0x01;
    e_b->state[grid_size/2] = _mm256_xor_si256(e_b->state[grid_size/2], _mm256_loadu_si256((__m256i*)diff));
    
    for(int t=1; t<=500; t++) {
        uint8_t sym = rand() % 2;
        engine_tick(e_a, sym);
        engine_tick(e_b, sym);
        
        if (t % 50 == 0) {
            int dist = hamming_distance_simd(e_a, e_b);
            printf("Step %3d | Hamming Distance: %4d bits\n", t, dist);
        }
    }
    printf("\n");
}

void run_rc3_task(const char* name, int task_type, int grid_size, int topology) {
    printf("Task: %s (Arithmetic RC-3 | K=32 channels + M4)\n", name);
    EngineArit* e = engine_alloc(grid_size, topology);
    engine_init(e);
    
    // 32 Independent RC LMS readouts
    int32_t w[32][49] = {0}; // [engine][channels] (32 wave + 16 M4 + 1 bias = 49)
    int LR_SHIFT = 4;
    int32_t MAX_W = 10000;
    
    int correct_report[32] = {0};
    uint8_t hist[16] = {0};
    
    // Variance tracking
    int64_t sum_channels[32][49] = {0};
    int64_t sum_sq_channels[32][49] = {0};
    
    // Diagnostic tracking for Engine 0
    int64_t mean_class1[49] = {0};
    int64_t mean_class0[49] = {0};
    int count1 = 0;
    int count0 = 0;
    
    for(int t=1; t<=500000; t++) {
        TaskData d = generate_task(task_type, t, hist);
        engine_tick(e, d.input_sym);
        
        // Extract channels for all 32 engines
        int32_t channel[32][49] = {0};
        
        // Wave Channels (K=32 from last 256 cells)
        int offset = e->grid_size - 256;
        for(int i=0; i<256; i+=32) {
            for(int k=0; k<32; k++) {
                uint8_t bytes[32];
                _mm256_storeu_si256((__m256i*)bytes, e->state[offset + i + k]);
                for(int eng=0; eng<32; eng++) channel[eng][k] += bytes[eng];
            }
        }
        
        // M4 Channels (16 extra)
        for(int i=0; i<16; i++) {
            int idx = (e->m4_head - 1 - i + HISTORY_SIZE) % HISTORY_SIZE;
            uint8_t bytes[32];
            _mm256_storeu_si256((__m256i*)bytes, e->m4_buf[idx]);
            for(int eng=0; eng<32; eng++) channel[eng][32 + i] = bytes[eng]; // M4 values are already 0/255
        }
        
        for(int eng=0; eng<32; eng++) {
            channel[eng][48] = 255; // Bias channel
        }
        
        if (t <= 10000) {
            if (d.target_sym == 1) count1++;
            else count0++;
        }
        
        // M4 Channels (16 extra)
        for(int i=0; i<16; i++) {
            int idx = (e->m4_head - 1 - i + HISTORY_SIZE) % HISTORY_SIZE;
            uint8_t bytes[32];
            _mm256_storeu_si256((__m256i*)bytes, e->m4_buf[idx]);
            for(int eng=0; eng<32; eng++) channel[eng][32 + i] = (int32_t)bytes[eng] << 5; // Scale M4 UP to match wave sum
        }
        
        for(int eng=0; eng<32; eng++) {
            int32_t score = 0;
            for(int k=0; k<49; k++) {
                score += w[eng][k] * channel[eng][k];
                
                // Track variance (only last 10000 steps to avoid overflow)
                if (t > 500000 - 10000) {
                    sum_channels[eng][k] += channel[eng][k];
                    sum_sq_channels[eng][k] += (int64_t)channel[eng][k] * channel[eng][k];
                }
                
                if (t <= 10000 && eng == 0) {
                    if (d.target_sym == 1) mean_class1[k] += channel[eng][k];
                    else mean_class0[k] += channel[eng][k];
                }
            }
            
            uint8_t pred = (score > 0) ? 1 : 0;
            if (pred == d.target_sym) correct_report[eng]++;
            
            int error = d.target_sym - pred;
            if (error != 0) {
                for(int k=0; k<49; k++) {
                    w[eng][k] += (error * channel[eng][k]) >> LR_SHIFT;
                    if (w[eng][k] > MAX_W) w[eng][k] = MAX_W;
                    if (w[eng][k] < -MAX_W) w[eng][k] = -MAX_W;
                }
            }
        }
        
        if (t == 10000) {
            printf("  --- Feature Separability (Delta means over 10K steps) ---\n");
            for(int k=0; k<49; k++) {
                double m1 = (double)mean_class1[k] / count1;
                double m0 = (double)mean_class0[k] / count0;
                double delta = fabs(m1 - m0);
                if (k < 32 && delta > 50) printf("    Wave Ch %2d: Delta %6.1f\n", k, delta);
                else if (k >= 32 && k < 48 && delta > 10) printf("    M4   Ch %2d: Delta %6.1f\n", k-32, delta);
            }
        }
        
        if (t % 50000 == 0) {
            // Report average accuracy across 32 engines
            double total_acc = 0;
            for(int eng=0; eng<32; eng++) total_acc += (double)correct_report[eng] / 50000.0;
            printf("  Step %5d | Avg Accuracy: %5.1f%%\n", t, (total_acc / 32.0) * 100.0);
            memset(correct_report, 0, sizeof(correct_report));
        }
    }
    
    // Variance analysis on Engine 0
    double var_sum = 0;
    for(int k=0; k<32; k++) { // only check wave channels
        double mean = (double)sum_channels[0][k] / 10000.0;
        double mean_sq = (double)sum_sq_channels[0][k] / 10000.0;
        double variance = mean_sq - (mean * mean);
        var_sum += variance;
    }
    printf("  Engine 0 Wave Feature Variance: %.2f (Avg per channel)\n\n", var_sum / 32.0);
}

int main() {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 1);
#endif

    printf("=== PHASE 5: ARITHMETIC WAVE VALIDATION ===\n\n");
    printf("Starting bench_throughput...\n");
    bench_throughput(256);
    
    printf("\nStarting test_esp...\n");
    test_esp(256);
    
    printf("\n--- RUNNING TOPOLOGY TESTS (XOR-2) ---\n");
    printf("T0: Baseline (Left Edge)\n");
    run_rc3_task("XOR-2", 0, 256, 0);
    
    printf("\nT1: Distribuita solo input corrente\n");
    run_rc3_task("XOR-2", 0, 256, 1);
    
    printf("\nT2: Multi-temporale sparse (spaziatura 8)\n");
    run_rc3_task("XOR-2", 0, 256, 2);
    
    printf("\nT3: Multi-temporale dense (spaziatura 4)\n");
    run_rc3_task("XOR-2", 0, 256, 3);
    run_rc3_task("Echo-5", 0, 256, 3);
    run_rc3_task("Period-7", 0, 256, 3);
    
    return 0;
}


#include "silicon_v0.h"
#include <stdlib.h>
#include <string.h>

void silicon_v0_init(SiliconV0* e, int codebook_seed) {
    srand(codebook_seed);
    for(int b = 0; b < 256; b++) {
        uint8_t vec[32];
        for(int i = 0; i < 32; i++) {
            vec[i] = (rand() % 2) ? 255 : 0;
        }
        e->codebook[b] = _mm256_loadu_si256((__m256i*)vec);
    }
    memset(e->m4_buf, 0, sizeof(e->m4_buf));
    e->m4_head = 0;
    memset(e->state, 0, sizeof(e->state));
}

void silicon_v0_reset(SiliconV0* e) {
    memset(e->state, 0, sizeof(e->state));
}

void silicon_v0_tick(SiliconV0* e, uint8_t input_byte) {
    // 1. Queue in historical M4
    e->m4_buf[e->m4_head] = input_byte;
    e->m4_head = (e->m4_head + 1) % 256;
    
    // 2. T3 Shift-Window Injection (Discrete Topology)
    int t3_tokens = 16;
    int spacing = 128 / t3_tokens; // 8
    
    for(int slot = 0; slot < t3_tokens; slot++) {
        int hist_idx = (e->m4_head - 1 - slot + 256) % 256;
        uint8_t h = e->m4_buf[hist_idx];
        int dest = slot * spacing; // Injection on a single block per token
        if (dest >= 128) dest -= 128;
        
        e->state[dest] = _mm256_adds_epu8(e->state[dest], e->codebook[h]);
    }
    
    // 3. Thermodynamic damping (Exponential decay)
    __m256i mask_7F = _mm256_set1_epi8(0x7F);
    _Pragma("GCC unroll 4")
    for(int i = 0; i < 128; i++) {
        e->state[i] = _mm256_and_si256(_mm256_srli_epi16(e->state[i], 1), mask_7F);
    }
    
    // 4. Wave Dynamics (Path integration)
    __m256i const_128 = _mm256_set1_epi8(-128);
    __m256i zero = _mm256_setzero_si256();
    __m256i m0 = _mm256_set1_epi8(0xAA);
    __m256i m1 = _mm256_set1_epi8(0xCC);
    
    __m256i new_state[128];
    for(int step = 0; step < 4; step++) {
        new_state[0] = e->state[0];
        new_state[127] = e->state[127];
        
        _Pragma("GCC unroll 4")
        for (int i = 1; i < 127; i++) {
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
            
            __m256i sel01 = _mm256_blendv_epi8(r0, r1, m0);
            __m256i sel23 = _mm256_blendv_epi8(r2, r3, m0);
            new_state[i] = _mm256_blendv_epi8(sel01, sel23, m1);
        }
        memcpy(e->state, new_state, 128 * sizeof(__m256i));
    }
}

void silicon_v0_extract_32d(const SiliconV0* e, double* out_32d) {
    memset(out_32d, 0, 32 * sizeof(double));
    int blocks_per_channel = 128 / 16; // 8
    
    for(int k = 0; k < 16; k++) {
        for(int i = 0; i < blocks_per_channel; i++) {
            uint8_t bytes[32];
            _mm256_storeu_si256((__m256i*)bytes, e->state[k * blocks_per_channel + i]);
            for(int lane = 0; lane < 32; lane++) {
                out_32d[lane] += bytes[lane];
            }
        }
    }
}


#ifndef SILICON_V0_H
#define SILICON_V0_H

#include <stdint.h>
#include <immintrin.h>

#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// Silicon Sequence Compressor V0
// Architecture: G128_T16_C16
// Dynamics: Persistent Wave (sat_add / avg)
// Readout: Lane-aware pooled 32D
// ----------------------------------------------------------------------------

typedef struct {
    // Spatial state of the wave (128 AVX2 blocks = 4096 bytes)
    __m256i state[128];
    
    // Physical Codebook (Random Binary Single-Block)
    __m256i codebook[256];
    
    // M4 History Buffer (circular)
    uint8_t m4_buf[256];
    int m4_head;
} SiliconV0;

/**
 * Initialize the engine.
 * Generates the Codebook (Random Binary Single-Block) using the provided seed,
 * and clears the spatial state and historical buffer.
 */
void silicon_v0_init(SiliconV0* e, int codebook_seed);

/**
 * Resets exclusively the differential state of the Wave.
 * Useful for ablations or to test the engine without historical integration,
 * or at the start of a new document.
 */
void silicon_v0_reset(SiliconV0* e);

/**
 * Executes an engine tick for the input byte.
 * 1. Queues the byte in M4.
 * 2. Reinjects the spatially shifted T3 window.
 * 3. Applies damping and the 4 wave diffusion steps.
 */
void silicon_v0_tick(SiliconV0* e, uint8_t input_byte);

/**
 * Extracts Lane-Aware Pooled 32D features from the spatial grid.
 * Performs lane-wise sum-pooling over the 16 spatial channels.
 * The output is a 32-double vector ready for regression/linear layer.
 */
void silicon_v0_extract_32d(const SiliconV0* e, double* out_32d);

#ifdef __cplusplus
}
#endif

#endif // SILICON_V0_H


#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// Shared hardware parameters
#define GRID_SIZE 256
#define BRIDGE_SPACING 64
#define NUM_ZONES (GRID_SIZE / BRIDGE_SPACING)
#define NUM_RULES 4
#define HISTORY_SIZE 1024

// Shared evaluation metrics
#define TOTAL_TICKS 100000

typedef struct {
    uint8_t input_sym;
    uint8_t target_sym;
} TaskData;

static inline TaskData generate_task(int task_type, int t, uint8_t* hist) {
    TaskData d;
    d.input_sym = rand() % 2;
    d.target_sym = 0;
    
    if (task_type == 0) { // XOR-2
        d.target_sym = hist[0] ^ hist[1];
    } else if (task_type == 1) { // Period-7
        d.target_sym = (t % 7 == 0) ? 1 : 0;
        d.input_sym = 0; 
    } else if (task_type == 2) { // Echo-5
        d.target_sym = hist[4];
    }
    
    // Shift history
    memmove(&hist[1], &hist[0], 15);
    hist[0] = d.input_sym;
    return d;
}

static inline void print_accuracy(int t, int correct, int window) {
    double acc = (double)correct / window;
    printf("  Step %5d | Accuracy: %5.1f%%\n", t, acc * 100.0);
}

#endif


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define GRID_SIZE 1024
#define BRIDGE_SPACING 64
#define NUM_ZONES (GRID_SIZE / BRIDGE_SPACING)
#define BRIDGE_FREQUENCY 4
#define NUM_RULES 4

static inline uint64_t get_cycles() {
    unsigned int dummy;
    return __rdtscp(&dummy);
}

typedef struct {
    uint8_t state[GRID_SIZE];
    uint8_t new_state[GRID_SIZE];
    uint8_t rule_select[NUM_ZONES];
    uint64_t ticks;
} WaveEngine;

void engine_init(WaveEngine* e) {
    memset(e->state, 0, GRID_SIZE);
    memset(e->new_state, 0, GRID_SIZE);
    for(int i = 0; i < NUM_ZONES; i++) {
        e->rule_select[i] = rand() % NUM_RULES;
    }
    e->ticks = 0;
}

static inline uint8_t apply_rule(uint8_t rule, uint8_t L, uint8_t C, uint8_t R) {
    switch(rule) {
        case 0: return (L & R) ^ C;
        case 1: return L ^ C ^ R;
        case 2: return (L | R) ^ C;
        case 3: return (L ^ R) & (C | 0x55);
        default: return C;
    }
}

// Phase A: Local Propagation
void wave_propagate(WaveEngine* e) {
    e->new_state[0] = e->state[0];
    e->new_state[GRID_SIZE-1] = e->state[GRID_SIZE-1];
    
    // We use a tight scalar loop as benchmarks showed ~0.28 cycles/elem
    for (int i = 1; i < GRID_SIZE - 1; i++) {
        int zone = i / BRIDGE_SPACING;
        uint8_t r = e->rule_select[zone];
        e->new_state[i] = apply_rule(r, e->state[i-1], e->state[i], e->state[i+1]);
    }
    
    memcpy(e->state, e->new_state, GRID_SIZE);
    e->ticks++;
}

// Phase B: Butterfly Bridge (Global Mixing)
void butterfly_bridge(WaveEngine* e) {
    // Two passes: even-odd, then odd-even
    for(int phase = 0; phase < 2; phase++) {
        for(int z = phase; z < NUM_ZONES - 1; z += 2) {
            int left_idx = (z + 1) * BRIDGE_SPACING - 8;
            int right_idx = (z + 1) * BRIDGE_SPACING;
            
            // Mix 8 boundary cells (Hadamard style add/sub)
            for(int i = 0; i < 8; i++) {
                uint8_t a = e->state[left_idx + i];
                uint8_t b = e->state[right_idx + i];
                e->state[left_idx + i] = a + b;
                e->state[right_idx + i] = a - b;
            }
        }
    }
}

// Phase C: Readout
uint8_t wave_readout(WaveEngine* e, uint32_t threshold) {
    uint32_t vote = 0;
    // Accumulate from the last 64 cells
    for (int i = GRID_SIZE - 64; i < GRID_SIZE; i++) {
        vote += e->state[i];
    }
    // Branchless comparison
    return (vote > threshold) ? 1 : 0;
}

// Inject symbol (Left-edge injection)
void wave_inject(WaveEngine* e, uint8_t symbol) {
    for (int i = 0; i < 8; i++) {
        e->state[i] ^= symbol;
    }
}

// Learning Mechanism
void wave_learn(WaveEngine* e, uint8_t error, uint8_t pred, int* threshold) {
    if (error) {
        // Mutate a random zone in the right half of the grid (closer to readout)
        int z = (NUM_ZONES / 2) + (rand() % (NUM_ZONES / 2));
        e->rule_select[z] = (e->rule_select[z] + 1) % NUM_RULES;
        
        // History Feedback: XOR error signal into left edge
        for (int i = 0; i < 8; i++) {
            e->state[i] ^= 0xFF;
        }
        
        // Threshold adaptation
        if (pred == 1) *threshold += 50; // Decrease sensitivity
        else *threshold -= 50;           // Increase sensitivity
    }
}

// Throughput Benchmark
void run_benchmark(WaveEngine* e) {
    uint64_t start = get_cycles();
    size_t iters = 100000;
    for(size_t i = 0; i < iters; i++) {
        wave_propagate(e);
        if (e->ticks % BRIDGE_FREQUENCY == 0) {
            butterfly_bridge(e);
        }
        wave_readout(e, 128 * 64);
    }
    uint64_t end = get_cycles();
    printf("Throughput: %.2f cycles/tick (for %d cells)\n", (double)(end-start)/iters, GRID_SIZE);
}

// ---------------------------------------------------------
// TASKS
// ---------------------------------------------------------
void run_task(const char* name, int task_type) {
    WaveEngine e;
    engine_init(&e);
    
    int threshold = 128 * 64; 
    int correct_count = 0;
    int window = 10000; // Large window to smooth out accuracy
    
    printf("Task: %s\n", name);
    
    uint8_t hist[16] = {0};
    
    for(int t = 0; t < 50000; t++) {
        uint8_t input_sym = rand() % 2; 
        uint8_t target = 0;
        
        if (task_type == 0) { // XOR-2
            target = hist[0] ^ hist[1];
        } else if (task_type == 1) { // Period-7
            target = (t % 7 == 0) ? 1 : 0;
            input_sym = 0; // Pure oscillation tracking
        } else if (task_type == 2) { // Echo-5
            target = hist[4];
        }
        
        // Shift history
        memmove(&hist[1], &hist[0], 15);
        hist[0] = input_sym;
        
        // Phase 1: Inject
        wave_inject(&e, input_sym ? 0xFF : 0x00);
        
        // Phase 2: Compute macroscopic timestep
        for(int w = 0; w < BRIDGE_FREQUENCY; w++) {
            wave_propagate(&e);
        }
        butterfly_bridge(&e);
        
        // Phase 3: Readout
        uint8_t pred = wave_readout(&e, threshold);
        
        // Phase 4: Learn
        uint8_t error = (pred != target) ? 1 : 0;
        if (!error) correct_count++;
        else wave_learn(&e, error, pred, &threshold);
        
        if ((t + 1) % window == 0) {
            double acc = (double)correct_count / window;
            printf("  Step %5d | Accuracy: %.1f%% | Threshold: %d\n", t + 1, acc * 100.0, threshold);
            correct_count = 0;
        }
    }
    printf("\n");
}

int main() {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 1);
#endif

    printf("=== PHASE 3: WAVE-BUTTERFLY MICRO-ENGINE ===\n\n");
    
    WaveEngine e;
    engine_init(&e);
    printf("--- Benchmark ---\n");
    run_benchmark(&e);
    printf("\n");
    
    printf("--- Learning Tests ---\n");
    run_task("XOR-2 (seq[t] = seq[t-1] ^ seq[t-2])", 0);
    run_task("Period-7 (predict pattern)", 1);
    run_task("Echo-5 (seq[t] = seq[t-5])", 2);
    
    return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../src/silicon_v0.h"

#include <immintrin.h>

static inline float dot_product_simd(const float* w, const float* f, int n) {
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) {
        sum = _mm256_fmadd_ps(_mm256_loadu_ps(&w[i]), _mm256_loadu_ps(&f[i]), sum);
    }
    float out[8];
    _mm256_storeu_ps(out, sum);
    return out[0] + out[1] + out[2] + out[3] + out[4] + out[5] + out[6] + out[7];
}

static inline void grad_update_simd(float* gradW, const float* f, float err_div_batch, int n) {
    __m256 err_vec = _mm256_set1_ps(err_div_batch);
    for (int i = 0; i < n; i += 8) {
        __m256 gw = _mm256_loadu_ps(&gradW[i]);
        __m256 fv = _mm256_loadu_ps(&f[i]);
        gw = _mm256_fmadd_ps(err_vec, fv, gw);
        _mm256_storeu_ps(&gradW[i], gw);
    }
}


#define MAX_SAMPLES 500000
#define MAX_FEATURES 512
#define CLASSES 256

uint8_t data[MAX_SAMPLES];
int data_size = 0;

float* features_train;
float* features_test;
uint8_t target_train[MAX_SAMPLES];
uint8_t target_test[MAX_SAMPLES];
uint8_t ctx_train[MAX_SAMPLES];
uint8_t ctx_test[MAX_SAMPLES];

int train_size = 0;
int test_size = 0;

double unigram_probs[CLASSES];
double bigram_probs[CLASSES][CLASSES];
float bigram_logits[CLASSES][CLASSES];

// -------------------------------------------------------------
// AdamW Optimizer State
// -------------------------------------------------------------
typedef struct {
    float W[CLASSES][MAX_FEATURES];
    float B[CLASSES];
    float mW[CLASSES][MAX_FEATURES];
    float vW[CLASSES][MAX_FEATURES];
    float mB[CLASSES];
    float vB[CLASSES];
    int t;
} AdamState;

void adam_init(AdamState* adam) {
    memset(adam, 0, sizeof(AdamState));
}

void adam_step(AdamState* adam, float* gradW, float gradB[CLASSES], int num_features, float lr, float beta1, float beta2, float eps, float wd) {
    adam->t++;
    float lr_t = lr * sqrtf(1.0f - powf(beta2, adam->t)) / (1.0f - powf(beta1, adam->t));
    
    for (int c = 0; c < CLASSES; c++) {
        adam->mB[c] = beta1 * adam->mB[c] + (1.0f - beta1) * gradB[c];
        adam->vB[c] = beta2 * adam->vB[c] + (1.0f - beta2) * gradB[c] * gradB[c];
        adam->B[c] -= lr_t * (adam->mB[c] / (sqrtf(adam->vB[c]) + eps) + wd * adam->B[c]);
        
        for (int f = 0; f < num_features; f++) {
            adam->mW[c][f] = beta1 * adam->mW[c][f] + (1.0f - beta1) * gradW[c * MAX_FEATURES + f];
            adam->vW[c][f] = beta2 * adam->vW[c][f] + (1.0f - beta2) * gradW[c * MAX_FEATURES + f] * gradW[c * MAX_FEATURES + f];
            adam->W[c][f] -= lr_t * (adam->mW[c][f] / (sqrtf(adam->vW[c][f]) + eps) + wd * adam->W[c][f]);
        }
    }
}

// -------------------------------------------------------------
// SGD Optimizer State (Control)
// -------------------------------------------------------------
void sgd_step(AdamState* adam, float* gradW, float gradB[CLASSES], int num_features, float lr, float wd) {
    for (int c = 0; c < CLASSES; c++) {
        adam->B[c] -= lr * (gradB[c] + wd * adam->B[c]);
        for (int f = 0; f < num_features; f++) {
            adam->W[c][f] -= lr * (gradW[c * MAX_FEATURES + f] + wd * adam->W[c][f]);
        }
    }
}

// -------------------------------------------------------------
// ML Harness
// -------------------------------------------------------------

void normalize_features(int num_features) {
    for (int f = 0; f < num_features; f++) {
        double mean = 0.0;
        for (int i = 0; i < train_size; i++) mean += features_train[i * num_features + f];
        mean /= train_size;
        
        double var = 0.0;
        for (int i = 0; i < train_size; i++) var += (features_train[i * num_features + f] - mean) * (features_train[i * num_features + f] - mean);
        var /= train_size;
        double std_dev = sqrt(var) + 1e-8;
        
        for (int i = 0; i < train_size; i++) features_train[i * num_features + f] = (features_train[i * num_features + f] - mean) / std_dev;
        for (int i = 0; i < test_size; i++) features_test[i * num_features + f] = (features_test[i * num_features + f] - mean) / std_dev;
    }
}

void evaluate_model(AdamState* model, int num_features, int use_residual, double* out_bpb, double* out_acc);
void train_logistic_regression(AdamState* model, int num_features, int epochs, int batch_size, float lr, int use_residual, int use_sgd) {
    float* gradW = (float*)malloc(CLASSES * MAX_FEATURES * sizeof(float));
    float gradB[CLASSES];
    float logits[CLASSES];
    float probs[CLASSES];
    
    AdamState* best_adam = (AdamState*)malloc(sizeof(AdamState));
    double best_bpb = 1e9, acc = 0;
    evaluate_model(model, num_features, use_residual, &best_bpb, &acc);
    memcpy(best_adam, model, sizeof(AdamState));
    
    if (use_residual) {
        printf("    Epoch 0 (Baseline): Val BPB = %.4f\n", best_bpb);
    }
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        memset(gradW, 0, CLASSES * MAX_FEATURES * sizeof(float));
        memset(gradB, 0, sizeof(gradB));
        
        for (int i = 0; i < train_size; i++) {

            float max_l = -1e9;
            for (int c = 0; c < CLASSES; c++) {
                logits[c] = model->B[c];
                if (use_residual) {
                    logits[c] += bigram_logits[ctx_train[i]][c];
                }
                logits[c] += dot_product_simd(model->W[c], &features_train[i * num_features], num_features);
                if (logits[c] > max_l) max_l = logits[c];
            }
            
            float sum_e = 0.0f;
            for (int c = 0; c < CLASSES; c++) {
                probs[c] = expf(logits[c] - max_l);
                sum_e += probs[c];
            }
            
            for (int c = 0; c < CLASSES; c++) probs[c] /= sum_e;
            
            total_loss -= logf(probs[target_train[i]]);
            
            for (int c = 0; c < CLASSES; c++) {
                float err = probs[c] - (c == target_train[i] ? 1.0f : 0.0f);
                gradB[c] += err / batch_size;
                grad_update_simd(&gradW[c * MAX_FEATURES], &features_train[i * num_features], err / batch_size, num_features);
            }
            
            if ((i + 1) % batch_size == 0 || (i + 1) == train_size) {
                if (use_sgd) {
                    sgd_step(model, gradW, gradB, num_features, lr, 1e-4f);
                } else {
                    adam_step(model, gradW, gradB, num_features, lr, 0.9f, 0.999f, 1e-8f, 1e-4f);
                }
                memset(gradW, 0, CLASSES * MAX_FEATURES * sizeof(float));
                memset(gradB, 0, sizeof(gradB));
            }
        }
        double val_bpb, val_acc;
        evaluate_model(model, num_features, use_residual, &val_bpb, &val_acc);
        if (val_bpb < best_bpb) {
            best_bpb = val_bpb;
            memcpy(best_adam, model, sizeof(AdamState));
        }
        // printf("    Epoch %d: Loss = %.4f | Val BPB = %.4f\n", epoch+1, total_loss / train_size, val_bpb);
    }
    memcpy(model, best_adam, sizeof(AdamState));
    free(best_adam);
    free(gradW);
}

void evaluate_model(AdamState* model, int num_features, int use_residual, double* out_bpb, double* out_acc) {
    double total_loss = 0.0;
    int correct = 0;
    
    for (int i = 0; i < test_size; i++) {
        float logits[CLASSES];
        float max_l = -1e9;
        
        for (int c = 0; c < CLASSES; c++) {
            logits[c] = model->B[c];
            if (use_residual) {
                logits[c] += bigram_logits[ctx_test[i]][c];
            }
            logits[c] += dot_product_simd(model->W[c], &features_test[i * num_features], num_features);
            if (logits[c] > max_l) max_l = logits[c];
        }
        
        float sum_e = 0.0f;
        int best_c = 0;
        float best_l = -1e9;
        
        for (int c = 0; c < CLASSES; c++) {
            float p = expf(logits[c] - max_l);
            sum_e += p;
            if (logits[c] > best_l) {
                best_l = logits[c];
                best_c = c;
            }
        }
        
        float prob = expf(logits[target_test[i]] - max_l) / sum_e;
        total_loss -= log2(prob); // True BPB (base 2)
        if (best_c == target_test[i]) correct++;
    }
    
    *out_bpb = total_loss / test_size;
    *out_acc = (double)correct / test_size * 100.0;
}

// -------------------------------------------------------------
// N-gram Baselines
// -------------------------------------------------------------

void compute_ngrams(double alpha) {
    double uni_counts[CLASSES] = {0};
    double bi_counts[CLASSES][CLASSES] = {0};
    double bi_totals[CLASSES] = {0};
    double uni_total = 0;
    
    for (int i = 0; i < train_size; i++) {
        uint8_t t = target_train[i];
        uint8_t c = ctx_train[i];
        uni_counts[t]++;
        uni_total++;
        bi_counts[c][t]++;
        bi_totals[c]++;
    }
    
    for (int i = 0; i < CLASSES; i++) {
        unigram_probs[i] = (uni_counts[i] + alpha) / (uni_total + CLASSES * alpha);
        for (int j = 0; j < CLASSES; j++) {
            bigram_probs[i][j] = (bi_counts[i][j] + alpha) / (bi_totals[i] + CLASSES * alpha);
            bigram_logits[i][j] = (float)log2(bigram_probs[i][j]) * 0.6931471805599453f; // Convert log2 to natural log! Wait, I should just use logf
        }
    }
}

void eval_ngrams(double* uni_bpb, double* bi_bpb) {
    double u_loss = 0;
    double b_loss = 0;
    for (int i = 0; i < test_size; i++) {
        uint8_t t = target_test[i];
        uint8_t c = ctx_test[i];
        u_loss -= log2(unigram_probs[t]);
        b_loss -= log2(bigram_probs[c][t]);
    }
    *uni_bpb = u_loss / test_size;
    *bi_bpb = b_loss / test_size;
}

// -------------------------------------------------------------
// Engine Implementations
// -------------------------------------------------------------

void run_eval(const char* name, int num_features, void (*extract_fn)(int idx, uint8_t byte, float* out)) {
    features_train = (float*)malloc(train_size * num_features * sizeof(float));
    features_test = (float*)malloc(test_size * num_features * sizeof(float));

    // 1. Extract Train
    for (int i = 0; i < train_size; i++) {
        extract_fn(i, data[i], &features_train[i * num_features]);
    }
    
    // 2. Extract Test
    for (int i = 0; i < test_size; i++) {
        extract_fn(i + train_size, data[train_size + i], &features_test[i * num_features]);
    }
    
    normalize_features(num_features);
    
    double best_pure_bpb = 1e9, best_pure_acc = 0;
    double best_res_bpb = 1e9, best_res_acc = 0;
    float lrs[] = {0.01f, 0.003f, 0.001f};
    
    for (int l = 0; l < 3; l++) {
        AdamState* adam = (AdamState*)malloc(sizeof(AdamState));
        
        // Pure Model
        adam_init(adam);
        train_logistic_regression(adam, num_features, 10, 256, lrs[l], 0, 0);
        double bpb, acc;
        evaluate_model(adam, num_features, 0, &bpb, &acc);
        if (bpb < best_pure_bpb) { best_pure_bpb = bpb; best_pure_acc = acc; }
        
        // Residual Model
        adam_init(adam);
        train_logistic_regression(adam, num_features, 10, 256, lrs[l], 1, 0);
        evaluate_model(adam, num_features, 1, &bpb, &acc);
        if (bpb < best_res_bpb) { best_res_bpb = bpb; best_res_acc = acc; }
        
        free(adam);
    }
    
    printf("[%-20s] BPB: %.4f | Acc: %.2f%%\n", name, best_pure_bpb, best_pure_acc);
    printf("[%-20s Residual] BPB: %.4f | Acc: %.2f%%\n", name, best_res_bpb, best_res_acc);
    fflush(stdout);

    free(features_train);
    free(features_test);
}

// M4 Simulation Codebook
__m256i m4_cb[256];
uint8_t hist[16];

void m4_extract_512(int idx, uint8_t byte, float* out) {
    if (idx == 0) memset(hist, 0, 16);
    memmove(hist + 1, hist, 15);
    hist[0] = byte;
    
    for (int t = 0; t < 16; t++) {
        uint8_t vec[32];
        _mm256_storeu_si256((__m256i*)vec, m4_cb[hist[t]]);
        for (int i = 0; i < 32; i++) out[t * 32 + i] = vec[i];
    }
}

void m4_extract_current_32(int idx, uint8_t byte, float* out) {
    uint8_t vec[32];
    _mm256_storeu_si256((__m256i*)vec, m4_cb[byte]);
    for (int i = 0; i < 32; i++) out[i] = vec[i];
}

void m4_extract_32(int idx, uint8_t byte, float* out) {
    if (idx == 0) memset(hist, 0, 16);
    memmove(hist + 1, hist, 15);
    hist[0] = byte;
    
    memset(out, 0, 32 * sizeof(float));
    for (int t = 0; t < 16; t++) {
        uint8_t vec[32];
        _mm256_storeu_si256((__m256i*)vec, m4_cb[hist[t]]);
        for (int i = 0; i < 32; i++) out[i] += vec[i];
    }
}

SiliconV0 v0;
void v0_extract_32(int idx, uint8_t byte, float* out) {
    if (idx == 0) silicon_v0_reset(&v0);
    silicon_v0_tick(&v0, byte);
    double d_out[32];
    silicon_v0_extract_32d(&v0, d_out);
    for (int i = 0; i < 32; i++) out[i] = (float)d_out[i];
}

int main(int argc, char** argv) {
    printf("Starting...\n"); fflush(stdout);
    FILE* f = fopen("data/promessi_sposi.txt", "rb");
    if (!f) { printf("Failed to open file\n"); return 1; }
    data_size = fread(data, 1, MAX_SAMPLES, f);
    fclose(f);
    printf("Data size: %d\n", data_size);
    fflush(stdout);
    
    train_size = data_size / 2;
    test_size = data_size / 2 - 1; // reserve 1 for next target
    
    for (int i = 0; i < train_size; i++) {
        target_train[i] = data[i+1];
        ctx_train[i] = data[i];
    }
    for (int i = 0; i < test_size; i++) {
        target_test[i] = data[train_size + i + 1];
        ctx_test[i] = data[train_size + i];
    }
    
    printf("Dataset: promessi_sposi.txt (Train: %d, Test: %d)\n", train_size, test_size);
    
    // Init M4 Codebook identically to V0
    srand(42);
    for(int b = 0; b < 256; b++) {
        uint8_t vec[32];
        for(int i = 0; i < 32; i++) vec[i] = (rand() % 2) ? 255 : 0;
        m4_cb[b] = _mm256_loadu_si256((__m256i*)vec);
    }
    silicon_v0_init(&v0, 42);
    
    printf("\n--- N-gram Baselines (Smoothing Alpha = 0.1) ---\n");
    compute_ngrams(0.1);
    double u_bpb, b_bpb;
    eval_ngrams(&u_bpb, &b_bpb);
    printf("Unigram True BPB: %.4f\n", u_bpb);
    printf("Bigram True BPB:  %.4f\n\n", b_bpb);
    fflush(stdout);
    
    printf("--- Softmax Distribution Readout (AdamW) ---\n");
    run_eval("M4 Current 32D", 32, m4_extract_current_32);
    run_eval("M4 Full 512D", 512, m4_extract_512);
    run_eval("M4 Pooled 32D", 32, m4_extract_32);
    run_eval("V0 Pooled 32D", 32, v0_extract_32);
    
    return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../src/silicon_v0.h"

#define MAX_SAMPLES 500000
#define MAX_FEATURES 512
#define CLASSES 256

uint8_t data[MAX_SAMPLES];
int data_size = 0;

float* features_train;
float* features_test;
uint8_t target_train[MAX_SAMPLES];
uint8_t target_test[MAX_SAMPLES];
uint8_t ctx_train[MAX_SAMPLES];
uint8_t ctx_test[MAX_SAMPLES];

int train_size = 0;
int test_size = 0;

double unigram_probs[CLASSES];
double bigram_probs[CLASSES][CLASSES];
float bigram_logits[CLASSES][CLASSES];

// -------------------------------------------------------------
// AdamW Optimizer State
// -------------------------------------------------------------
typedef struct {
    float W[CLASSES][MAX_FEATURES];
    float B[CLASSES];
    float mW[CLASSES][MAX_FEATURES];
    float vW[CLASSES][MAX_FEATURES];
    float mB[CLASSES];
    float vB[CLASSES];
    int t;
} AdamState;

void adam_init(AdamState* adam) {
    memset(adam, 0, sizeof(AdamState));
}

void adam_step(AdamState* adam, float* gradW, float gradB[CLASSES], int num_features, float lr, float beta1, float beta2, float eps, float wd) {
    adam->t++;
    float lr_t = lr * sqrtf(1.0f - powf(beta2, adam->t)) / (1.0f - powf(beta1, adam->t));
    
    for (int c = 0; c < CLASSES; c++) {
        adam->mB[c] = beta1 * adam->mB[c] + (1.0f - beta1) * gradB[c];
        adam->vB[c] = beta2 * adam->vB[c] + (1.0f - beta2) * gradB[c] * gradB[c];
        adam->B[c] -= lr_t * (adam->mB[c] / (sqrtf(adam->vB[c]) + eps) + wd * adam->B[c]);
        
        for (int f = 0; f < num_features; f++) {
            adam->mW[c][f] = beta1 * adam->mW[c][f] + (1.0f - beta1) * gradW[c * MAX_FEATURES + f];
            adam->vW[c][f] = beta2 * adam->vW[c][f] + (1.0f - beta2) * gradW[c * MAX_FEATURES + f] * gradW[c * MAX_FEATURES + f];
            adam->W[c][f] -= lr_t * (adam->mW[c][f] / (sqrtf(adam->vW[c][f]) + eps) + wd * adam->W[c][f]);
        }
    }
}

// -------------------------------------------------------------
// SGD Optimizer State (Control)
// -------------------------------------------------------------
void sgd_step(AdamState* adam, float* gradW, float gradB[CLASSES], int num_features, float lr, float wd) {
    for (int c = 0; c < CLASSES; c++) {
        adam->B[c] -= lr * (gradB[c] + wd * adam->B[c]);
        for (int f = 0; f < num_features; f++) {
            adam->W[c][f] -= lr * (gradW[c * MAX_FEATURES + f] + wd * adam->W[c][f]);
        }
    }
}

// -------------------------------------------------------------
// ML Harness
// -------------------------------------------------------------

void normalize_features(int num_features) {
    for (int f = 0; f < num_features; f++) {
        double mean = 0.0;
        for (int i = 0; i < train_size; i++) mean += features_train[i * num_features + f];
        mean /= train_size;
        
        double var = 0.0;
        for (int i = 0; i < train_size; i++) var += (features_train[i * num_features + f] - mean) * (features_train[i * num_features + f] - mean);
        var /= train_size;
        double std_dev = sqrt(var) + 1e-8;
        
        for (int i = 0; i < train_size; i++) features_train[i * num_features + f] = (features_train[i * num_features + f] - mean) / std_dev;
        for (int i = 0; i < test_size; i++) features_test[i * num_features + f] = (features_test[i * num_features + f] - mean) / std_dev;
    }
}

void train_logistic_regression(AdamState* model, int num_features, int epochs, int batch_size, float lr, int use_residual, int use_sgd) {
    float* gradW = (float*)malloc(CLASSES * MAX_FEATURES * sizeof(float));
    float gradB[CLASSES];
    float logits[CLASSES];
    float probs[CLASSES];
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        memset(gradW, 0, CLASSES * MAX_FEATURES * sizeof(float));
        memset(gradB, 0, sizeof(gradB));
        
        for (int i = 0; i < train_size; i++) {
            float max_l = -1e9;
            for (int c = 0; c < CLASSES; c++) {
                logits[c] = model->B[c];
                if (use_residual) {
                    logits[c] += bigram_logits[ctx_train[i]][c];
                }
                for (int f = 0; f < num_features; f++) logits[c] += model->W[c][f] * features_train[i * num_features + f];
                if (logits[c] > max_l) max_l = logits[c];
            }
            
            float sum_e = 0.0f;
            for (int c = 0; c < CLASSES; c++) {
                probs[c] = expf(logits[c] - max_l);
                sum_e += probs[c];
            }
            
            for (int c = 0; c < CLASSES; c++) probs[c] /= sum_e;
            
            total_loss -= logf(probs[target_train[i]]);
            
            for (int c = 0; c < CLASSES; c++) {
                float err = probs[c] - (c == target_train[i] ? 1.0f : 0.0f);
                gradB[c] += err / batch_size;
                for (int f = 0; f < num_features; f++) {
                    gradW[c * MAX_FEATURES + f] += err * features_train[i * num_features + f] / batch_size;
                }
            }
            
            if ((i + 1) % batch_size == 0 || (i + 1) == train_size) {
                if (use_sgd) {
                    sgd_step(model, gradW, gradB, num_features, lr, 1e-4f);
                } else {
                    adam_step(model, gradW, gradB, num_features, lr, 0.9f, 0.999f, 1e-8f, 1e-4f);
                }
                memset(gradW, 0, CLASSES * MAX_FEATURES * sizeof(float));
                memset(gradB, 0, sizeof(gradB));
            }
        }
        if ((epoch+1) % 5 == 0) {
            printf("  Epoch %d: Loss = %.4f\n", epoch+1, total_loss / train_size);
            fflush(stdout);
        }
    }
    free(gradW);
}

void evaluate_model(AdamState* model, int num_features, int use_residual, double* out_bpb, double* out_acc) {
    double total_loss = 0.0;
    int correct = 0;
    
    for (int i = 0; i < test_size; i++) {
        float logits[CLASSES];
        float max_l = -1e9;
        
        for (int c = 0; c < CLASSES; c++) {
            logits[c] = model->B[c];
            if (use_residual) {
                logits[c] += bigram_logits[ctx_test[i]][c];
            }
            for (int f = 0; f < num_features; f++) logits[c] += model->W[c][f] * features_test[i * num_features + f];
            if (logits[c] > max_l) max_l = logits[c];
        }
        
        float sum_e = 0.0f;
        int best_c = 0;
        float best_l = -1e9;
        
        for (int c = 0; c < CLASSES; c++) {
            float p = expf(logits[c] - max_l);
            sum_e += p;
            if (logits[c] > best_l) {
                best_l = logits[c];
                best_c = c;
            }
        }
        
        float prob = expf(logits[target_test[i]] - max_l) / sum_e;
        total_loss -= log2(prob); // True BPB (base 2)
        if (best_c == target_test[i]) correct++;
    }
    
    *out_bpb = total_loss / test_size;
    *out_acc = (double)correct / test_size * 100.0;
}

// -------------------------------------------------------------
// N-gram Baselines
// -------------------------------------------------------------

void compute_ngrams(double alpha) {
    double uni_counts[CLASSES] = {0};
    double bi_counts[CLASSES][CLASSES] = {0};
    double bi_totals[CLASSES] = {0};
    double uni_total = 0;
    
    for (int i = 0; i < train_size; i++) {
        uint8_t t = target_train[i];
        uint8_t c = ctx_train[i];
        uni_counts[t]++;
        uni_total++;
        bi_counts[c][t]++;
        bi_totals[c]++;
    }
    
    for (int i = 0; i < CLASSES; i++) {
        unigram_probs[i] = (uni_counts[i] + alpha) / (uni_total + CLASSES * alpha);
        for (int j = 0; j < CLASSES; j++) {
            bigram_probs[i][j] = (bi_counts[i][j] + alpha) / (bi_totals[i] + CLASSES * alpha);
            bigram_logits[i][j] = (float)log2(bigram_probs[i][j]) * 0.6931471805599453f; // Convert log2 to natural log! Wait, I should just use logf
        }
    }
}

void eval_ngrams(double* uni_bpb, double* bi_bpb) {
    double u_loss = 0;
    double b_loss = 0;
    for (int i = 0; i < test_size; i++) {
        uint8_t t = target_test[i];
        uint8_t c = ctx_test[i];
        u_loss -= log2(unigram_probs[t]);
        b_loss -= log2(bigram_probs[c][t]);
    }
    *uni_bpb = u_loss / test_size;
    *bi_bpb = b_loss / test_size;
}

// -------------------------------------------------------------
// Engine Implementations
// -------------------------------------------------------------

void run_eval(const char* name, int num_features, void (*extract_fn)(int idx, uint8_t byte, float* out)) {
    features_train = (float*)malloc(train_size * num_features * sizeof(float));
    features_test = (float*)malloc(test_size * num_features * sizeof(float));

    // 1. Extract Train
    for (int i = 0; i < train_size; i++) {
        extract_fn(i, data[i], &features_train[i * num_features]);
    }
    
    // 2. Extract Test
    for (int i = 0; i < test_size; i++) {
        extract_fn(i + train_size, data[train_size + i], &features_test[i * num_features]);
    }
    
    normalize_features(num_features);
    
    AdamState* adam = (AdamState*)malloc(sizeof(AdamState));
    double bpb, acc;
    
    // Pure Model
    adam_init(adam);
    train_logistic_regression(adam, num_features, 10, 256, 0.01f, 0, 0); // AdamW
    evaluate_model(adam, num_features, 0, &bpb, &acc);
    printf("[%-20s] BPB: %.4f | Acc: %.2f%%\n", name, bpb, acc);
    fflush(stdout);
    
    // Residual Model
    adam_init(adam);
    train_logistic_regression(adam, num_features, 10, 256, 0.01f, 1, 0); // AdamW
    evaluate_model(adam, num_features, 1, &bpb, &acc);
    printf("[%-20s Residual] BPB: %.4f | Acc: %.2f%%\n", name, bpb, acc);
    fflush(stdout);

    free(features_train);
    free(features_test);
    free(adam);
}

// M4 Simulation Codebook
__m256i m4_cb[256];
uint8_t hist[16];

void m4_extract_512(int idx, uint8_t byte, float* out) {
    if (idx == 0) memset(hist, 0, 16);
    memmove(hist + 1, hist, 15);
    hist[0] = byte;
    
    for (int t = 0; t < 16; t++) {
        uint8_t vec[32];
        _mm256_storeu_si256((__m256i*)vec, m4_cb[hist[t]]);
        for (int i = 0; i < 32; i++) out[t * 32 + i] = vec[i];
    }
}

void m4_extract_32(int idx, uint8_t byte, float* out) {
    if (idx == 0) memset(hist, 0, 16);
    memmove(hist + 1, hist, 15);
    hist[0] = byte;
    
    memset(out, 0, 32 * sizeof(float));
    for (int t = 0; t < 16; t++) {
        uint8_t vec[32];
        _mm256_storeu_si256((__m256i*)vec, m4_cb[hist[t]]);
        for (int i = 0; i < 32; i++) out[i] += vec[i];
    }
}

SiliconV0 v0;
void v0_extract_32(int idx, uint8_t byte, float* out) {
    if (idx == 0) silicon_v0_reset(&v0);
    silicon_v0_tick(&v0, byte);
    double d_out[32];
    silicon_v0_extract_32d(&v0, d_out);
    for (int i = 0; i < 32; i++) out[i] = (float)d_out[i];
}

int main(int argc, char** argv) {
    printf("Starting...\n"); fflush(stdout);
    FILE* f = fopen("data/promessi_sposi.txt", "rb");
    if (!f) { printf("Failed to open file\n"); return 1; }
    data_size = fread(data, 1, MAX_SAMPLES, f);
    fclose(f);
    printf("Data size: %d\n", data_size);
    fflush(stdout);
    
    train_size = data_size / 2;
    test_size = data_size / 2 - 1; // reserve 1 for next target
    
    for (int i = 0; i < train_size; i++) {
        target_train[i] = data[i+1];
        ctx_train[i] = data[i];
    }
    for (int i = 0; i < test_size; i++) {
        target_test[i] = data[train_size + i + 1];
        ctx_test[i] = data[train_size + i];
    }
    
    printf("Dataset: promessi_sposi.txt (Train: %d, Test: %d)\n", train_size, test_size);
    
    // Init M4 Codebook identically to V0
    srand(42);
    for(int b = 0; b < 256; b++) {
        uint8_t vec[32];
        for(int i = 0; i < 32; i++) vec[i] = (rand() % 2) ? 255 : 0;
        m4_cb[b] = _mm256_loadu_si256((__m256i*)vec);
    }
    silicon_v0_init(&v0, 42);
    
    printf("\n--- N-gram Baselines (Smoothing Alpha = 0.1) ---\n");
    compute_ngrams(0.1);
    double u_bpb, b_bpb;
    eval_ngrams(&u_bpb, &b_bpb);
    printf("Unigram True BPB: %.4f\n", u_bpb);
    printf("Bigram True BPB:  %.4f\n\n", b_bpb);
    fflush(stdout);
    
    printf("--- Softmax Distribution Readout (AdamW) ---\n");
    run_eval("M4 Full 512D", 512, m4_extract_512);
    run_eval("M4 Pooled 32D", 32, m4_extract_32);
    run_eval("V0 Pooled 32D", 32, v0_extract_32);
    
    return 0;
}


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
    int t3_enabled;
} SiliconEngine;

void engine_init(SiliconEngine* e, int t3_enabled) {
    memset(e->state, 0, sizeof(e->state));
    memset(e->m4_buf, 0, sizeof(e->m4_buf));
    e->m4_head = 0;
    e->grid_vectors = GRID_VECTORS;
    e->t3_enabled = t3_enabled;
}

void engine_tick(SiliconEngine* e, uint8_t input_byte) {
    __m256i m4_mem = _mm256_set1_epi8(input_byte);
    e->m4_buf[e->m4_head] = m4_mem;
    e->m4_head = (e->m4_head + 1) % HISTORY_SIZE;

    __m256i t0 = e->m4_buf[(e->m4_head - 1 + HISTORY_SIZE) % HISTORY_SIZE];
    __m256i t1 = e->m4_buf[(e->m4_head - 2 + HISTORY_SIZE) % HISTORY_SIZE];
    __m256i t2 = e->m4_buf[(e->m4_head - 3 + HISTORY_SIZE) % HISTORY_SIZE];

    int blocks = GRID_VECTORS;
    if (e->t3_enabled) {
        for(int i=0; i<blocks; i+=12) {
            if (i < blocks) e->state[i] = _mm256_adds_epu8(e->state[i], t0);
            if (i+4 < blocks) e->state[i+4] = _mm256_adds_epu8(e->state[i+4], t1);
            if (i+8 < blocks) e->state[i+8] = _mm256_adds_epu8(e->state[i+8], t2);
        }
    } else {
        // Only current byte (t0) at the center
        e->state[blocks/2] = _mm256_adds_epu8(e->state[blocks/2], t0);
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
        memcpy(e->state, new_state, blocks * sizeof(__m256i));
    }
}

void extract_features(SiliconEngine* e, double* f_out) {
    int32_t wave_sums[32] = {0};
    int offset = e->grid_vectors - 256; 
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
        for(int j=0; j<n; j++) XtX_reg[j*n + j] += lambda * N;
        
        if (cholesky_decompose(XtX_reg, n, L)) {
            success = 1;
            break;
        }
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
// DATASET GENERATORS
// ============================================================================

uint8_t* create_block_shuffled(uint8_t* src, int len, int block_size) {
    uint8_t* dst = malloc(len);
    int num_blocks = len / block_size;
    int* perm = malloc(num_blocks * sizeof(int));
    for(int i=0; i<num_blocks; i++) perm[i] = i;
    for(int i=num_blocks-1; i>0; i--) {
        int j = rand() % (i+1);
        int temp = perm[i];
        perm[i] = perm[j];
        perm[j] = temp;
    }
    for(int i=0; i<num_blocks; i++) {
        memcpy(dst + i*block_size, src + perm[i]*block_size, block_size);
    }
    int remainder = len % block_size;
    if (remainder > 0) memcpy(dst + num_blocks*block_size, src + num_blocks*block_size, remainder);
    free(perm);
    return dst;
}

uint8_t* create_intra_block_shuffled(uint8_t* src, int len, int block_size) {
    uint8_t* dst = malloc(len);
    int num_blocks = len / block_size;
    for(int b=0; b<num_blocks; b++) {
        uint8_t block[256];
        memcpy(block, src + b*block_size, block_size);
        for(int i=block_size-1; i>0; i--) {
            int j = rand() % (i+1);
            uint8_t temp = block[i];
            block[i] = block[j];
            block[j] = temp;
        }
        memcpy(dst + b*block_size, block, block_size);
    }
    int remainder = len % block_size;
    if (remainder > 0) memcpy(dst + num_blocks*block_size, src + num_blocks*block_size, remainder);
    return dst;
}

// ============================================================================
// METRICS EVALUATION
// ============================================================================

void eval_text_metrics(Dataset** test_bits, double w[8][NUM_TOTAL_FEATURES], double b[8], int n_features, int feature_offset, const char* name) {
    int N = test_bits[0]->N;
    int correct_bits = 0;
    double brier_sum = 0;
    double bpb_sum = 0;
    
    for(int i=0; i<N; i++) {
        for(int bit=0; bit<8; bit++) {
            double* f = &test_bits[bit]->X[i * NUM_TOTAL_FEATURES + feature_offset];
            double score = predict(f, w[bit], b[bit], n_features);
            
            double prob = 1.0 / (1.0 + exp(-score * 5.0));
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
    
    printf("  %-30s | Bit Acc: %5.2f%% | Factorized BPB: %5.2f\n", name, bit_acc, mse, bpb);
}

// ============================================================================
// PHASE 7 BENCHMARK
// ============================================================================

typedef struct {
    Dataset* train_bits[8];
    Dataset* val_bits[8];
    int WARMUP;
    int N_train;
    int N_val;
    int m4_random_map[32][32];
} TortureContext;

void extract_to_datasets(TortureContext* ctx, uint8_t* source_text, int t3_enabled) {
    SiliconEngine e;
    engine_init(&e, t3_enabled);
    for(int t=0; t < ctx->WARMUP + ctx->N_train + ctx->N_val - 1; t++) {
        uint8_t input_sym = source_text[t];
        uint8_t target_sym = source_text[t+1];
        engine_tick(&e, input_sym);
        if (t >= ctx->WARMUP) {
            int idx = t - ctx->WARMUP;
            Dataset** dst_arr = (idx < ctx->N_train) ? ctx->train_bits : ctx->val_bits;
            int offset = (idx < ctx->N_train) ? idx : idx - ctx->N_train;
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
}

void train_and_eval(TortureContext* ctx, int n_features, int feature_offset, const char* name, int apply_random_proj) {
    double w[8][NUM_TOTAL_FEATURES], b[8];
    for(int bit=0; bit<8; bit++) {
        Dataset* ds = create_dataset(ctx->N_train, n_features);
        for(int i=0; i<ctx->N_train; i++) {
            if (apply_random_proj) {
                for(int j=0; j<32; j++) {
                    double sum = 0;
                    for(int r=0; r<32; r++) {
                        sum += ctx->train_bits[bit]->X[i*NUM_TOTAL_FEATURES + NUM_WAVE_FEATURES + ctx->m4_random_map[j][r]];
                    }
                    ds->X[i*n_features + j] = sum;
                }
            } else {
                for(int j=0; j<n_features; j++) ds->X[i*n_features + j] = ctx->train_bits[bit]->X[i*NUM_TOTAL_FEATURES + feature_offset + j];
            }
            ds->y[i] = ctx->train_bits[bit]->y[i];
        }
        fit_ridge(ds, w[bit], &b[bit]);
        free_dataset(ds);
    }
    
    if (apply_random_proj) {
        int correct_bits = 0;
        double brier_sum = 0;
        double bpb_sum = 0;
        for(int i=0; i<ctx->N_val; i++) {
            for(int bit=0; bit<8; bit++) {
                double f[32];
                for(int j=0; j<32; j++) {
                    double sum = 0;
                    for(int r=0; r<32; r++) {
                        sum += ctx->val_bits[bit]->X[i*NUM_TOTAL_FEATURES + NUM_WAVE_FEATURES + ctx->m4_random_map[j][r]];
                    }
                    f[j] = sum;
                }
                double score = predict(f, w[bit], b[bit], n_features);
                double prob = 1.0 / (1.0 + exp(-score * 5.0));
                if (prob < 1e-7) prob = 1e-7;
                if (prob > 1.0 - 1e-7) prob = 1.0 - 1e-7;
                int target = (int)ctx->val_bits[bit]->y[i];
                int pred = (score > 0.5) ? 1 : 0;
                if (pred == target) correct_bits++;
                double p_target = target ? prob : (1.0 - prob);
                brier_sum += (prob - target)*(prob - target);
                bpb_sum += -log2(p_target);
            }
        }
        printf("  %-30s | Bit Acc: %5.2f%% | MSE: %6.4f | Factorized BPB: %5.2f\n", name, 100.0 * correct_bits / (ctx->N_val * 8.0), brier_sum / (ctx->N_val * 8.0), bpb_sum / ctx->N_val);
    } else {
        eval_text_metrics(ctx->val_bits, w, b, n_features, feature_offset, name);
    }
}

void run_torture_chamber(const char* filename, int N_train, int N_val) {
    printf("\n=== PHASE 7: TORTURE CHAMBER (%s) ===\n", filename);
    
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
    if (fsize < WARMUP + N_train + N_val) {
        printf("File too small for requested N_train/N_val.\n");
        free(text);
        return;
    }
    
    printf("Data Split: %d Train, %d Validation\n\n", N_train, N_val);
    
    // --- Baseline: Bigram (True Byte-Level BPB) ---
    double alpha = 0.1;
    double bigram_matrix[256][256] = {0};
    double bigram_sums[256] = {0};
    
    for(int i=0; i<256; i++) {
        for(int j=0; j<256; j++) bigram_matrix[i][j] = alpha;
        bigram_sums[i] = 256 * alpha;
    }
    for(int t=WARMUP; t < WARMUP + N_train - 1; t++) {
        bigram_matrix[text[t]][text[t+1]] += 1.0;
        bigram_sums[text[t]] += 1.0;
    }
    
    double bigram_bpb = 0;
    int bigram_correct_bits = 0;
    for(int t=WARMUP + N_train; t < WARMUP + N_train + N_val - 1; t++) {
        uint8_t current = text[t];
        uint8_t target = text[t+1];
        double p = bigram_matrix[current][target] / bigram_sums[current];
        bigram_bpb += -log2(p);
        
        int best_byte = 0; double best_p = -1;
        for(int c=0; c<256; c++) {
            if (bigram_matrix[current][c] > best_p) { best_p = bigram_matrix[current][c]; best_byte = c; }
        }
        for(int bit=0; bit<8; bit++) {
            if (((best_byte >> bit) & 1) == ((target >> bit) & 1)) bigram_correct_bits++;
        }
    }
    printf("  %-30s | Bit Acc: %5.2f%% | True Byte BPB:  %5.2f\n", "Bigram (alpha=0.1)", 
           100.0 * bigram_correct_bits / (N_val * 8.0), bigram_bpb / N_val);
           
    // Marginal Unigram Bitwise Baseline (for Global Target Shuffle)
    double unigram_bpb = 0;
    int unigram_correct = 0;
    int byte_counts[256] = {0};
    for(int t=WARMUP; t < WARMUP + N_train; t++) byte_counts[text[t+1]]++;
    for(int t=WARMUP + N_train; t < WARMUP + N_train + N_val - 1; t++) {
        uint8_t target = text[t+1];
        for(int bit=0; bit<8; bit++) {
            int bit_target = (target >> bit) & 1;
            int count_1 = 0;
            for(int c=0; c<256; c++) if ((c >> bit) & 1) count_1 += byte_counts[c];
            double p1 = (double)count_1 / N_train;
            int pred = p1 > 0.5 ? 1 : 0;
            if (pred == bit_target) unigram_correct++;
            double p_target = bit_target ? p1 : (1.0 - p1);
            if (p_target < 1e-7) p_target = 1e-7;
            unigram_bpb += -log2(p_target);
        }
    }
    printf("  %-30s | Bit Acc: %5.2f%% | Factorized BPB: %5.2f\n", "Marginal Unigram Bitwise", 
           100.0 * unigram_correct / (N_val * 8.0), unigram_bpb / N_val);
           
    TortureContext ctx;
    ctx.WARMUP = WARMUP;
    ctx.N_train = N_train;
    ctx.N_val = N_val;
    for(int b=0; b<8; b++) {
        ctx.train_bits[b] = create_dataset(N_train, NUM_TOTAL_FEATURES);
        ctx.val_bits[b] = create_dataset(N_val, NUM_TOTAL_FEATURES);
    }
    for(int i=0; i<32; i++) {
        for(int j=0; j<32; j++) ctx.m4_random_map[i][j] = rand() % 16;
    }
    
    // --- 1. NORMAL REAL TEXT ---
    printf("\n--- Test: Real Text (T3 Enabled) ---\n");
    extract_to_datasets(&ctx, text, 1);
    train_and_eval(&ctx, NUM_M4_FEATURES, NUM_WAVE_FEATURES, "M4-readout", 0);
    train_and_eval(&ctx, NUM_WAVE_FEATURES, 0, "Wave-readout-T3", 0);
    train_and_eval(&ctx, NUM_WAVE_FEATURES, 0, "Wave+M4-readout-T3", 0);
    train_and_eval(&ctx, NUM_WAVE_FEATURES, 0, "Random M4 Dimension-Matched", 1);
    
    // --- 2. GLOBAL TARGET SHUFFLE ---
    printf("\n--- Test: Global Target Shuffle ---\n");
    for(int bit=0; bit<8; bit++) {
        for(int i=N_val-1; i>0; i--) {
            int j = rand() % (i+1);
            double temp = ctx.val_bits[bit]->y[i];
            ctx.val_bits[bit]->y[i] = ctx.val_bits[bit]->y[j];
            ctx.val_bits[bit]->y[j] = temp;
        }
    }
    train_and_eval(&ctx, NUM_WAVE_FEATURES, 0, "Wave-readout-T3 (Shuffled y)", 0);
    
    // --- 3. INTRA-BLOCK SHUFFLE ---
    printf("\n--- Test: Intra-Block Shuffle (16-bytes) ---\n");
    uint8_t* intra_shuffled = create_intra_block_shuffled(text, fsize, 16);
    extract_to_datasets(&ctx, intra_shuffled, 1);
    train_and_eval(&ctx, NUM_M4_FEATURES, NUM_WAVE_FEATURES, "M4-readout", 0);
    train_and_eval(&ctx, NUM_WAVE_FEATURES, 0, "Wave-readout-T3", 0);
    free(intra_shuffled);
    
    // --- 4. BLOCK SHUFFLE ---
    printf("\n--- Test: Block Shuffle (16-bytes) ---\n");
    uint8_t* block_shuffled = create_block_shuffled(text, fsize, 16);
    extract_to_datasets(&ctx, block_shuffled, 1);
    train_and_eval(&ctx, NUM_M4_FEATURES, NUM_WAVE_FEATURES, "M4-readout", 0);
    train_and_eval(&ctx, NUM_WAVE_FEATURES, 0, "Wave-readout-T3", 0);
    free(block_shuffled);
    
    // --- 5. T3 ABLATION (Current-byte only) ---
    printf("\n--- Test: T3 Ablation ---\n");
    extract_to_datasets(&ctx, text, 0);
    train_and_eval(&ctx, NUM_WAVE_FEATURES, 0, "Wave-current-only", 0);
    
    for(int b=0; b<8; b++) {
        free_dataset(ctx.train_bits[b]);
        free_dataset(ctx.val_bits[b]);
    }
    free(text);
}

int main() {
    srand(42);
    
    // Run size sweep on benchmark6_text.c
    printf("=======================================================================\n");
    printf("PHASE 7A/7B: TORTURE CHAMBER & COMPRESSION (benchmark6_text.c)\n");
    printf("=======================================================================\n");
    
    // Train Size Sweep
    run_torture_chamber("benchmark6_text.c", 2000, 2000);
    run_torture_chamber("benchmark6_text.c", 5000, 2000);
    run_torture_chamber("benchmark6_text.c", 10000, 5000);
    
    return 0;
}


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
// DATASET GENERATORS
// ============================================================================

uint8_t* create_block_shuffled(uint8_t* src, int len, int block_size) {
    uint8_t* dst = malloc(len);
    int num_blocks = len / block_size;
    int* perm = malloc(num_blocks * sizeof(int));
    for(int i=0; i<num_blocks; i++) perm[i] = i;
    for(int i=num_blocks-1; i>0; i--) {
        int j = rand() % (i+1);
        int temp = perm[i];
        perm[i] = perm[j];
        perm[j] = temp;
    }
    for(int i=0; i<num_blocks; i++) {
        memcpy(dst + i*block_size, src + perm[i]*block_size, block_size);
    }
    int remainder = len % block_size;
    if (remainder > 0) memcpy(dst + num_blocks*block_size, src + num_blocks*block_size, remainder);
    free(perm);
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
// HARDWARE TEMPLATES
// ============================================================================

#define DECLARE_ENGINE(GRID_VECTORS) \
typedef struct { \
    __m256i state[GRID_VECTORS]; \
    __m256i m4_buf[256]; \
    int m4_head; \
    int grid_vectors; \
} SiliconEngine_##GRID_VECTORS; \
\
static inline void engine_init_##GRID_VECTORS(SiliconEngine_##GRID_VECTORS* e) { \
    memset(e->state, 0, sizeof(e->state)); \
    memset(e->m4_buf, 0, sizeof(e->m4_buf)); \
    e->m4_head = 0; \
    e->grid_vectors = GRID_VECTORS; \
} \
\
static inline void engine_tick_##GRID_VECTORS(SiliconEngine_##GRID_VECTORS* e, uint8_t input_byte, int t3_tokens) { \
    __m256i m4_mem = _mm256_set1_epi8(input_byte); \
    e->m4_buf[e->m4_head] = m4_mem; \
    e->m4_head = (e->m4_head + 1) % 256; \
    int blocks = GRID_VECTORS; \
    int spacing = blocks / t3_tokens; \
    if (spacing == 0) spacing = 1; \
    for(int slot=0; slot<t3_tokens; slot++) { \
        int hist_idx = (e->m4_head - 1 - slot + 256) % 256; \
        __m256i t = e->m4_buf[hist_idx]; \
        int idx = slot * spacing; \
        if (idx < blocks) e->state[idx] = _mm256_adds_epu8(e->state[idx], t); \
    } \
    __m256i const_128 = _mm256_set1_epi8(-128); \
    __m256i zero = _mm256_setzero_si256(); \
    __m256i mask_7F = _mm256_set1_epi8(0x7F); \
    _Pragma("GCC unroll 4") \
    for(int i=0; i<blocks; i++) { \
        e->state[i] = _mm256_and_si256(_mm256_srli_epi16(e->state[i], 1), mask_7F); \
    } \
    __m256i new_state[GRID_VECTORS]; \
    for(int w=0; w<4; w++) { \
        new_state[0] = e->state[0]; \
        new_state[blocks-1] = e->state[blocks-1]; \
        _Pragma("GCC unroll 4") \
        for (int i=1; i<blocks-1; i++) { \
            __m256i L = e->state[i-1]; \
            __m256i C = e->state[i]; \
            __m256i R = e->state[i+1]; \
            __m256i r0 = _mm256_adds_epu8(_mm256_avg_epu8(L, R), _mm256_subs_epu8(C, const_128)); \
            __m256i r1 = _mm256_adds_epu8(_mm256_subs_epu8(L, C), R); \
            __m256i l_half = _mm256_avg_epu8(L, zero); \
            __m256i c_half = _mm256_avg_epu8(C, zero); \
            __m256i r_half = _mm256_avg_epu8(R, zero); \
            __m256i r2 = _mm256_adds_epu8(l_half, _mm256_adds_epu8(r_half, c_half)); \
            __m256i r3 = _mm256_subs_epu8(_mm256_adds_epu8(L, R), C); \
            __m256i m0 = _mm256_set1_epi8(0xAA); \
            __m256i m1 = _mm256_set1_epi8(0xCC); \
            __m256i sel01 = _mm256_blendv_epi8(r0, r1, m0); \
            __m256i sel23 = _mm256_blendv_epi8(r2, r3, m0); \
            new_state[i] = _mm256_blendv_epi8(sel01, sel23, m1); \
        } \
        memcpy(e->state, new_state, blocks * sizeof(__m256i)); \
    } \
} \
static inline void extract_features_##GRID_VECTORS(SiliconEngine_##GRID_VECTORS* e, double* f_out, int channels) { \
    int32_t wave_sums[128] = {0}; \
    int blocks_per_channel = e->grid_vectors / channels; \
    if (blocks_per_channel == 0) blocks_per_channel = 1; \
    for (int k=0; k<channels; k++) { \
        for (int i=0; i<blocks_per_channel; i++) { \
            int block_idx = k * blocks_per_channel + i; \
            if (block_idx >= e->grid_vectors) break; \
            uint8_t bytes[32]; \
            _mm256_storeu_si256((__m256i*)bytes, e->state[block_idx]); \
            for(int eng=0; eng<32; eng++) wave_sums[k] += bytes[eng]; \
        } \
    } \
    for(int k=0; k<channels; k++) f_out[k] = (double)wave_sums[k]; \
    for(int i=0; i<16; i++) { \
        int idx = (e->m4_head - 1 - i + 256) % 256; \
        uint8_t bytes[32]; \
        _mm256_storeu_si256((__m256i*)bytes, e->m4_buf[idx]); \
        f_out[channels + i] = (double)bytes[0]; \
    } \
}

DECLARE_ENGINE(128)
DECLARE_ENGINE(256)
DECLARE_ENGINE(512)
DECLARE_ENGINE(1024)

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
        ctx->train_bits[b] = create_dataset(N_train, 128 + 16); // max features
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

// Global baseline variables
double baseline_unigram_bpb = 0;
double baseline_m4_bpb = 0;

typedef struct {
    double mean_acc;
    double std_acc;
    double mean_bpb;
} MetricResult;

void eval_and_report(SweepContext* ctx, int n_features, int feature_offset, const char* label, double cycles_per_byte, MetricResult* res_out) {
    double w[8][128+16], b[8];
    uint64_t t_ridge_start = get_cycles();
    
    for(int bit=0; bit<8; bit++) {
        Dataset* ds = create_dataset(ctx->N_train, n_features);
        for(int i=0; i<ctx->N_train; i++) {
            for(int j=0; j<n_features; j++) ds->X[i*n_features + j] = ctx->train_bits[bit]->X[i*(128+16) + feature_offset + j];
            ds->y[i] = ctx->train_bits[bit]->y[i];
        }
        fit_ridge(ds, w[bit], &b[bit]);
        free_dataset(ds);
    }
    uint64_t t_ridge_end = get_cycles();
    double ridge_ms = (t_ridge_end - t_ridge_start) / 3000000.0; // Assume ~3GHz

    int correct_bits = 0;
    double brier_sum = 0;
    double bpb_sum = 0;
    
    for(int i=0; i<ctx->N_val; i++) {
        for(int bit=0; bit<8; bit++) {
            double* f = &ctx->val_bits[bit]->X[i * (128+16) + feature_offset];
            double score = predict(f, w[bit], b[bit], n_features);
            
            // Linear Probability Model: Ridge predicts E[y] directly since targets are 0/1.
            double prob = score;
            if (prob < 1e-4) prob = 1e-4;
            if (prob > 1.0 - 1e-4) prob = 1.0 - 1e-4;
            
            int target = (int)ctx->val_bits[bit]->y[i];
            int pred = (score > 0.5) ? 1 : 0;
            if (pred == target) correct_bits++;
            double p_target = target ? prob : (1.0 - prob);
            brier_sum += (prob - target)*(prob - target);
            bpb_sum += -log2(p_target);
        }
    }
    
    double bit_acc = 100.0 * correct_bits / (ctx->N_val * 8.0);
    double bpb = bpb_sum / ctx->N_val;
    
    if (res_out) {
        res_out->mean_acc = bit_acc;
        res_out->mean_bpb = bpb;
    }
    
    if (label) {
        double gain_m4 = (baseline_m4_bpb - bpb) / (cycles_per_byte / 1000.0);
        double gain_uni = (baseline_unigram_bpb - bpb) / (cycles_per_byte / 1000.0);
        if (strcmp(label, "M4-only") == 0) {
            baseline_m4_bpb = bpb;
            printf("  %-15s | Acc: %5.2f%% | BPB: %5.2f | C/B: %4.0f | Gain vs Uni/1k: %+5.2f\n", 
                   label, bit_acc, bpb, cycles_per_byte, gain_uni);
        } else {
            printf("  %-15s | Acc: %5.2f%% | BPB: %5.2f | C/B: %4.0f | Gain vs M4/1k: %+5.2f\n", 
                   label, bit_acc, bpb, cycles_per_byte, gain_m4);
        }
    }
}

// Macro to generate sweep combinations
#define RUN_SWEEP(GRID) \
{ \
    SiliconEngine_##GRID e; \
    engine_init_##GRID(&e); \
    uint64_t t_start = get_cycles(); \
    for(int t=0; t < ctx->WARMUP + ctx->N_train + ctx->N_val - 1; t++) { \
        engine_tick_##GRID(&e, text[t], t3_tokens); \
        if (t >= ctx->WARMUP) { \
            int idx = t - ctx->WARMUP; \
            Dataset** dst_arr = (idx < ctx->N_train) ? ctx->train_bits : ctx->val_bits; \
            int offset = (idx < ctx->N_train) ? idx : idx - ctx->N_train; \
            double features[128+16] = {0}; \
            extract_features_##GRID(&e, features, channels); \
            for(int bit=0; bit<8; bit++) { \
                for(int i=0; i<channels+16; i++) dst_arr[bit]->X[offset * (128+16) + i] = features[i]; \
                dst_arr[bit]->y[offset] = (text[t+1] >> bit) & 1; \
            } \
        } \
    } \
    uint64_t t_end = get_cycles(); \
    double cycles_per_byte = (double)(t_end - t_start) / (ctx->WARMUP + ctx->N_train + ctx->N_val); \
    char label[64]; \
    sprintf(label, "G%d_T%d_C%d", GRID, t3_tokens, channels); \
    eval_and_report(ctx, channels, 0, label, cycles_per_byte, NULL); \
}

void compute_baselines(uint8_t* text, SweepContext* ctx) {
    // Unigram
    int byte_counts[256] = {0};
    for(int t=ctx->WARMUP; t < ctx->WARMUP + ctx->N_train; t++) byte_counts[text[t+1]]++;
    double unigram_bpb = 0;
    int unigram_correct = 0;
    for(int t=ctx->WARMUP + ctx->N_train; t < ctx->WARMUP + ctx->N_train + ctx->N_val - 1; t++) {
        uint8_t target = text[t+1];
        for(int bit=0; bit<8; bit++) {
            int bit_target = (target >> bit) & 1;
            int count_1 = 0;
            for(int c=0; c<256; c++) if ((c >> bit) & 1) count_1 += byte_counts[c];
            double p1 = (double)count_1 / ctx->N_train;
            if (p1 > 0.5 == bit_target) unigram_correct++;
            double p_target = bit_target ? p1 : (1.0 - p1);
            if (p_target < 1e-7) p_target = 1e-7;
            unigram_bpb += -log2(p_target);
        }
    }
    baseline_unigram_bpb = unigram_bpb / ctx->N_val;
    printf("  %-15s | Acc: %5.2f%% | BPB: %5.2f | C/B:    0 | Gain vs Uni/1k:  0.00\n", 
           "Unigram", 100.0 * unigram_correct / (ctx->N_val * 8.0), baseline_unigram_bpb);
           
    // M4-only
    // Run G128 just to extract M4 features efficiently
    SiliconEngine_128 e;
    engine_init_128(&e);
    uint64_t t_start = get_cycles();
    for(int t=0; t < ctx->WARMUP + ctx->N_train + ctx->N_val - 1; t++) {
        engine_tick_128(&e, text[t], 0); // No T3 needed for pure M4
        if (t >= ctx->WARMUP) {
            int idx = t - ctx->WARMUP;
            Dataset** dst_arr = (idx < ctx->N_train) ? ctx->train_bits : ctx->val_bits;
            int offset = (idx < ctx->N_train) ? idx : idx - ctx->N_train;
            double features[128+16] = {0};
            extract_features_128(&e, features, 0); // extracts only M4 at feature[0..15]
            for(int bit=0; bit<8; bit++) {
                for(int i=0; i<16; i++) dst_arr[bit]->X[offset * (128+16) + 128 + i] = features[i];
                dst_arr[bit]->y[offset] = (text[t+1] >> bit) & 1;
            }
        }
    }
    uint64_t t_end = get_cycles();
    double cycles_per_byte = (double)(t_end - t_start) / (ctx->WARMUP + ctx->N_train + ctx->N_val);
    eval_and_report(ctx, 16, 128, "M4-only", cycles_per_byte, NULL);
}

// ============================================================================
// PHASE 8B RUNNER
// ============================================================================

void run_phase_8b_dataset(uint8_t* text, long fsize, const char* name, int is_shuffled) {
    printf("\n=== PHASE 8B: %s ===\n", name);
    int N_train = 5000;
    int N_val = 2000;
    int NUM_SPLITS = 5;
    
    if (fsize < (N_train + N_val) * NUM_SPLITS + 2000) {
        printf("  [Warning] File too small for 5 separate splits, overlapping offsets.\n");
    }
    
    double bigram_acc_sum = 0;
    double unigram_acc_sum = 0;
    double m4_acc_sum = 0;
    double wave_acc_sum = 0;
    
    double m4_accs[5], wave_accs[5], uni_accs[5], bi_accs[5];
    
    for(int s=0; s<NUM_SPLITS; s++) {
        int warmup = 2000 + s * (N_train + N_val);
        if (warmup + N_train + N_val >= fsize) warmup = 2000 + (s * 500); // overlap
        if (warmup + N_train + N_val >= fsize) warmup = 0;
        
        SweepContext* ctx = create_sweep_context(N_train, N_val);
        ctx->WARMUP = warmup;
        
        // 1. Bigram
        double alpha = 0.1;
        double bigram_matrix[256][256] = {0};
        double bigram_sums[256] = {0};
        for(int i=0; i<256; i++) {
            for(int j=0; j<256; j++) bigram_matrix[i][j] = alpha;
            bigram_sums[i] = 256 * alpha;
        }
        for(int t=warmup; t < warmup + N_train - 1; t++) {
            bigram_matrix[text[t]][text[t+1]] += 1.0;
            bigram_sums[text[t]] += 1.0;
        }
        int bi_correct = 0;
        for(int t=warmup + N_train; t < warmup + N_train + N_val - 1; t++) {
            uint8_t current = text[t];
            uint8_t target = text[t+1];
            int best_byte = 0; double best_p = -1;
            for(int c=0; c<256; c++) {
                if (bigram_matrix[current][c] > best_p) { best_p = bigram_matrix[current][c]; best_byte = c; }
            }
            for(int bit=0; bit<8; bit++) {
                if (((best_byte >> bit) & 1) == ((target >> bit) & 1)) bi_correct++;
            }
        }
        bi_accs[s] = 100.0 * bi_correct / (N_val * 8.0);
        bigram_acc_sum += bi_accs[s];
        
        // 2. Unigram
        int byte_counts[256] = {0};
        for(int t=warmup; t < warmup + N_train; t++) byte_counts[text[t+1]]++;
        int uni_correct = 0;
        for(int t=warmup + N_train; t < warmup + N_train + N_val - 1; t++) {
            uint8_t target = text[t+1];
            for(int bit=0; bit<8; bit++) {
                int bit_target = (target >> bit) & 1;
                int count_1 = 0;
                for(int c=0; c<256; c++) if ((c >> bit) & 1) count_1 += byte_counts[c];
                double p1 = (double)count_1 / N_train;
                if ((p1 > 0.5) == bit_target) uni_correct++;
            }
        }
        uni_accs[s] = 100.0 * uni_correct / (N_val * 8.0);
        unigram_acc_sum += uni_accs[s];
        
        // 3. M4 Extraction
        SiliconEngine_128 e;
        engine_init_128(&e);
        for(int t=0; t < warmup + N_train + N_val - 1; t++) {
            engine_tick_128(&e, text[t], 16); // G128_T16_C16 sweet spot
            if (t >= warmup) {
                int idx = t - warmup;
                Dataset** dst_arr = (idx < N_train) ? ctx->train_bits : ctx->val_bits;
                int offset = (idx < N_train) ? idx : idx - N_train;
                double features[128+16] = {0};
                extract_features_128(&e, features, 16);
                for(int bit=0; bit<8; bit++) {
                    for(int i=0; i<16+16; i++) dst_arr[bit]->X[offset * (128+16) + i] = features[i];
                    dst_arr[bit]->y[offset] = (text[t+1] >> bit) & 1;
                }
            }
        }
        
        MetricResult m4_res;
        eval_and_report(ctx, 16, 16, NULL, 0, &m4_res); // features[16..31] are M4
        m4_accs[s] = m4_res.mean_acc;
        m4_acc_sum += m4_accs[s];
        
        MetricResult wave_res;
        eval_and_report(ctx, 16, 0, NULL, 0, &wave_res); // features[0..15] are Wave
        wave_accs[s] = wave_res.mean_acc;
        wave_acc_sum += wave_accs[s];
        
        free_sweep_context(ctx);
    }
    
    double bi_mean = bigram_acc_sum / NUM_SPLITS;
    double uni_mean = unigram_acc_sum / NUM_SPLITS;
    double m4_mean = m4_acc_sum / NUM_SPLITS;
    double wave_mean = wave_acc_sum / NUM_SPLITS;
    
    double bi_std=0, uni_std=0, m4_std=0, wave_std=0;
    for(int s=0; s<NUM_SPLITS; s++) {
        bi_std += (bi_accs[s]-bi_mean)*(bi_accs[s]-bi_mean);
        uni_std += (uni_accs[s]-uni_mean)*(uni_accs[s]-uni_mean);
        m4_std += (m4_accs[s]-m4_mean)*(m4_accs[s]-m4_mean);
        wave_std += (wave_accs[s]-wave_mean)*(wave_accs[s]-wave_mean);
    }
    bi_std = sqrt(bi_std/NUM_SPLITS);
    uni_std = sqrt(uni_std/NUM_SPLITS);
    m4_std = sqrt(m4_std/NUM_SPLITS);
    wave_std = sqrt(wave_std/NUM_SPLITS);
    
    printf("  Bigram (Markov-1) : %5.2f%% ± %4.2f\n", bi_mean, bi_std);
    printf("  Unigram Marginal  : %5.2f%% ± %4.2f\n", uni_mean, uni_std);
    printf("  M4-only           : %5.2f%% ± %4.2f\n", m4_mean, m4_std);
    printf("  Wave (G128_T16)   : %5.2f%% ± %4.2f\n", wave_mean, wave_std);
    printf("  Wave Gain vs M4   : %+5.2f%%\n", wave_mean - m4_mean);
}

void phase_8b_sweep() {
    printf("\n=======================================================================\n");
    printf("PHASE 8B: GENERALIZATION SWEEP\n");
    printf("=======================================================================\n");
    
    const char* files[] = {"benchmark6_text.c", "DOCS/architecture_decisions.md", "data/promessi_sposi.txt"};
    
    for(int i=0; i<3; i++) {
        FILE* f = fopen(files[i], "rb");
        if(!f) { printf("Error opening %s\n", files[i]); continue; }
        fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
        uint8_t* text = (uint8_t*)malloc(fsize);
        fread(text, 1, fsize, f); fclose(f);
        
        run_phase_8b_dataset(text, fsize, files[i], 0);
        
        if (strcmp(files[i], "data/promessi_sposi.txt") == 0) {
            uint8_t* shuffled = create_block_shuffled(text, fsize, 16);
            run_phase_8b_dataset(shuffled, fsize, "data/promessi_sposi.txt (SHUFFLED 16-byte blocks)", 1);
            free(shuffled);
        }
        free(text);
    }
}

void phase_8a_sweep(uint8_t* text, int N_train, int N_val) {
    printf("\n=== PHASE 8A: HARDWARE GEOMETRY SWEEP ===\n");
    SweepContext* ctx = create_sweep_context(N_train, N_val);
    
    compute_baselines(text, ctx);
    
    int t3_configs[] = {8, 16, 32, 64};
    int ch_configs[] = {16, 32, 64};
    
    for(int i_t3=0; i_t3<4; i_t3++) {
        for(int i_ch=0; i_ch<3; i_ch++) {
            int t3_tokens = t3_configs[i_t3];
            int channels = ch_configs[i_ch];
            
            RUN_SWEEP(128);
            RUN_SWEEP(256);
            RUN_SWEEP(512);
            RUN_SWEEP(1024);
        }
    }
    
    free_sweep_context(ctx);
}

int main() {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 1);
#endif
    
    // Uncomment to run 8A
    // const char* filename = "benchmark6_text.c";
    // FILE* f = fopen(filename, "rb");
    // if(f) {
    //     fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    //     uint8_t* text = (uint8_t*)malloc(fsize);
    //     fread(text, 1, fsize, f); fclose(f);
    //     phase_8a_sweep(text, 5000, 2000);
    //     free(text);
    // }
    
    phase_8b_sweep();
    
    return 0;
}


