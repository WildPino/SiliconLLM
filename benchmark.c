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
