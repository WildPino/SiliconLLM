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
    // 1. Accoda in M4 storco
    e->m4_buf[e->m4_head] = input_byte;
    e->m4_head = (e->m4_head + 1) % 256;
    
    // 2. Iniezione T3 Shift-Window (Topologia Discreta)
    int t3_tokens = 16;
    int spacing = 128 / t3_tokens; // 8
    
    for(int slot = 0; slot < t3_tokens; slot++) {
        int hist_idx = (e->m4_head - 1 - slot + 256) % 256;
        uint8_t h = e->m4_buf[hist_idx];
        int dest = slot * spacing; // Iniezione su un singolo blocco per token
        if (dest >= 128) dest -= 128;
        
        e->state[dest] = _mm256_adds_epu8(e->state[dest], e->codebook[h]);
    }
    
    // 3. Damping termodinamico (Decadimento esponenziale)
    __m256i mask_7F = _mm256_set1_epi8(0x7F);
    _Pragma("GCC unroll 4")
    for(int i = 0; i < 128; i++) {
        e->state[i] = _mm256_and_si256(_mm256_srli_epi16(e->state[i], 1), mask_7F);
    }
    
    // 4. Wave Dynamics (Integrazione di percorso)
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
