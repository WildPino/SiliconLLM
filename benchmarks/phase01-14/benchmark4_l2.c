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
