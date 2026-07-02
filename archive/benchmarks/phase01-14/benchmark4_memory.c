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
