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
