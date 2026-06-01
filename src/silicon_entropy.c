#include "silicon_entropy.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

void see_reset(SiliconEntropyState* see) {
    silicon_v0_reset(&see->v0);
    memset(see->hist, 0, sizeof(see->hist));
    memset(see->last_l0, 0, sizeof(see->last_l0));
    memset(see->l1_state, 0, sizeof(see->l1_state));
    memset(see->current_chunk_mean, 0, sizeof(see->current_chunk_mean));
    see->bytes_in_chunk = 0;
}

void see_oja_reset(SiliconEntropyState* see) {
    for (int j = 0; j < SEE_N_OJA; j++) {
        memset(see->W_oja[j], 0, 43 * sizeof(float));
        if (j < 43) see->W_oja[j][j] = 1.0f;  // identity init
    }
}

void see_init(SiliconEntropyState* see, int seed, int chunk_size, float decay) {
    see->chunk_size = chunk_size;
    see->decay = decay;
    see->history_tokens = 4;
    see->pooling_mode = 0;
    see->multiscale_mode = 0;
    see->alpha_fast = 0.7f;
    see->alpha_mid  = 0.9f;
    see->alpha_slow = 0.99f;
    see->eta_oja    = 0.0f;
    for (int i = 0; i < 256; i++) see->byte_gain[i] = 1.0f;
    see_oja_reset(see);
    
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
    silicon_v0_extract_32d_mode(&see->v0, d_out, see->pooling_mode);
    for (int i = 0; i < 32; i++) l0_out[32 + i] = (float)d_out[i];
    
    // Save for next extraction
    memcpy(see->last_l0, l0_out, SEE_L0_DIM * sizeof(float));
    
    // 3. Update L1 state
    if (see->multiscale_mode == 1) {
        float g   = see->byte_gain[byte];
        float af  = (1.0f - see->alpha_fast) * g;
        float am  = (1.0f - see->alpha_mid)  * g;
        float as_ = (1.0f - see->alpha_slow) * g;

        // Oja plastic cells: first SEE_N_OJA of fast band.
        // The learned projection W_oja is ALWAYS applied. Since W_oja is
        // identity-initialized, this is bit-identical to SEE-V1S when untrained
        // (y == l0_out[j]). The Oja *update* only runs when eta_oja > 0, so a
        // frozen-but-trained W_oja still shapes features during Pass-2
        // re-extraction and generation (where eta_oja == 0). This decoupling is
        // what lets the readout actually see the learned directions.
        for (int j = 0; j < SEE_N_OJA; j++) {
            float y = 0;
            for (int k = 0; k < 43; k++) y += see->W_oja[j][k] * l0_out[k];
            see->l1_state[j] = see->alpha_fast * see->l1_state[j] + (1.0f - see->alpha_fast) * y;
            if (see->eta_oja > 0.0f) {
                // Oja update (no byte_gain for plastic cells)
                for (int k = 0; k < 43; k++)
                    see->W_oja[j][k] += see->eta_oja * y * (l0_out[k] - y * see->W_oja[j][k]);
                // Norm clamp: |w_j| <= 2.0
                float n2 = 0;
                for (int k = 0; k < 43; k++) n2 += see->W_oja[j][k] * see->W_oja[j][k];
                if (n2 > 4.0f) {
                    float s = 2.0f / sqrtf(n2 + 1e-8f);
                    for (int k = 0; k < 43; k++) see->W_oja[j][k] *= s;
                }
            }
        }
        for (int i = SEE_N_OJA; i < 43; i++)
            see->l1_state[i] = see->alpha_fast * see->l1_state[i] + af * l0_out[i];
        for (int i = 0; i < 43; i++)
            see->l1_state[43 + i] = see->alpha_mid  * see->l1_state[43 + i] + am  * l0_out[i];
        for (int i = 0; i < 42; i++)
            see->l1_state[86 + i] = see->alpha_slow * see->l1_state[86 + i] + as_ * l0_out[i];
    } else {
        // Legacy: chunk mean + last via single EMA
        for(int i = 0; i < SEE_L0_DIM; i++)
            see->current_chunk_mean[i] += l0_out[i];
        see->bytes_in_chunk++;

        if (see->bytes_in_chunk == see->chunk_size) {
            float mean_last[SEE_L1_DIM];
            for(int i = 0; i < SEE_L0_DIM; i++) {
                mean_last[i]              = see->current_chunk_mean[i] / (float)see->chunk_size;
                mean_last[SEE_L0_DIM + i] = l0_out[i];
            }
            for(int i = 0; i < SEE_L1_DIM; i++)
                see->l1_state[i] = (see->l1_state[i] * see->decay) + mean_last[i];
            memset(see->current_chunk_mean, 0, sizeof(see->current_chunk_mean));
            see->bytes_in_chunk = 0;
        }
    }
}

void see_extract(const SiliconEntropyState* see, float out_features[SEE_FEATURE_DIM]) {
    memcpy(out_features, see->last_l0, SEE_L0_DIM * sizeof(float));
    memcpy(out_features + SEE_L0_DIM, see->l1_state, SEE_L1_DIM * sizeof(float));
}
