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
