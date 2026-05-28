#include "silicon_entropy.h"
#include <string.h>
#include <stdlib.h>

void see_reset(SiliconEntropyState* see) {
    silicon_v0_reset(&see->v0);
    memset(see->hist, 0, sizeof(see->hist));
    memset(see->last_l0, 0, sizeof(see->last_l0));
    memset(see->l1_state, 0, sizeof(see->l1_state));
    memset(see->current_chunk_mean, 0, sizeof(see->current_chunk_mean));
    see->bytes_in_chunk = 0;
}

void see_init(SiliconEntropyState* see, int seed, int chunk_size, float decay) {
    see->chunk_size = chunk_size;
    see->decay = decay;
    see->history_tokens = 4;
    
    // Initialize Codebook for M4 (32D)
    srand(seed);
    for(int b = 0; b < 256; b++) {
        uint8_t vec[32];
        for(int i = 0; i < 32; i++) vec[i] = (rand() % 2) ? 255 : 0;
        see->m4_cb[b] = _mm256_loadu_si256((__m256i*)vec);
    }
    
    silicon_v0_init(&see->v0, seed);
    see_reset(see);
}

void see_observe(SiliconEntropyState* see, uint8_t byte) {
    // 1. Update L0 state
    memmove(see->hist + 1, see->hist, 255);
    see->hist[0] = byte;
    
    int history_tokens = see->history_tokens;
    silicon_v0_tick_t(&see->v0, byte, history_tokens);

    // 2. Extract current L0 features
    float l0_out[SEE_L0_DIM];
    memset(l0_out, 0, sizeof(l0_out));

    // M4 (32D)
    for (int t = 0; t < history_tokens; t++) {
        uint8_t vec[32];
        _mm256_storeu_si256((__m256i*)vec, see->m4_cb[see->hist[t]]);
        for (int i = 0; i < 32; i++) l0_out[i] += vec[i];
    }
    
    // V0 (32D)
    double d_out[32];
    silicon_v0_extract_32d(&see->v0, d_out);
    for (int i = 0; i < 32; i++) l0_out[32 + i] = (float)d_out[i];
    
    // Save for next extraction
    memcpy(see->last_l0, l0_out, SEE_L0_DIM * sizeof(float));
    
    // 3. Accumulate L1 Chunk Mean
    for(int i = 0; i < SEE_L0_DIM; i++) {
        see->current_chunk_mean[i] += l0_out[i];
    }
    see->bytes_in_chunk++;
    
    // 4. Close chunk and update L1 state via EMA
    if (see->bytes_in_chunk == see->chunk_size) {
        float mean_last[SEE_L1_DIM];
        for(int i = 0; i < SEE_L0_DIM; i++) {
            mean_last[i] = see->current_chunk_mean[i] / (float)see->chunk_size; // mean
            mean_last[SEE_L0_DIM + i] = l0_out[i];                              // last
        }
        
        for(int i = 0; i < SEE_L1_DIM; i++) {
            see->l1_state[i] = (see->l1_state[i] * see->decay) + mean_last[i];
        }
        
        memset(see->current_chunk_mean, 0, sizeof(see->current_chunk_mean));
        see->bytes_in_chunk = 0;
    }
}

void see_extract(const SiliconEntropyState* see, float out_features[SEE_FEATURE_DIM]) {
    memcpy(out_features, see->last_l0, SEE_L0_DIM * sizeof(float));
    memcpy(out_features + SEE_L0_DIM, see->l1_state, SEE_L1_DIM * sizeof(float));
}
