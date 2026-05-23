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
