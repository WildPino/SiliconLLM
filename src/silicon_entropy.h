#ifndef SILICON_ENTROPY_H
#define SILICON_ENTROPY_H

#include <stdint.h>
#include "silicon_v0.h"

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

// The dimensionality of the Silicon Entropy Engine output
#define SEE_FEATURE_DIM 192
#define SEE_L0_DIM 64
#define SEE_L1_DIM 128

typedef struct {
    // L0 State
    SiliconV0 v0;
    __m256i m4_cb[256];
    uint8_t hist[256];
    float last_l0[SEE_L0_DIM];

    // L1 State
    float l1_state[SEE_L1_DIM];
    float current_chunk_mean[SEE_L0_DIM];
    int bytes_in_chunk;
    
    // Configs
    int chunk_size;
    float decay;
    int history_tokens;  // t3_tokens for silicon_v0_tick_t (default 4)
    int pooling_mode;    // V0 pooling: 0=sum 1=max 2=range 3=threshold
} SiliconEntropyState;

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the Silicon Entropy Engine
void see_init(SiliconEntropyState* see, int seed, int chunk_size, float decay);

// Reset the internal state (useful for epoch boundaries or new files)
void see_reset(SiliconEntropyState* see);

// Observe a true byte, update internal representations (L0 and L1)
void see_observe(SiliconEntropyState* see, uint8_t byte);

// Extract the 192D feature vector to predict the NEXT byte
void see_extract(const SiliconEntropyState* see, float out_features[SEE_FEATURE_DIM]);

#ifdef __cplusplus
}
#endif

#endif // SILICON_ENTROPY_H
