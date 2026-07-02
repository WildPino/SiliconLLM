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
