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

    // Multi-timescale L1 (Phase 43.A)
    // When multiscale_mode=1, l1_state is populated per-byte as 3 EMA bands:
    //   l1_state[0:42]   alpha_fast, L0[0:42]
    //   l1_state[43:85]  alpha_mid,  L0[0:42]
    //   l1_state[86:127] alpha_slow, L0[0:41]
    // Legacy chunk-based path (multiscale_mode=0) is unchanged.
    int   multiscale_mode; // 0=legacy, 1=3-band per-byte EMA
    float alpha_fast;      // default 0.7
    float alpha_mid;       // default 0.9
    float alpha_slow;      // default 0.99

    // Trainable injection gain (Phase 43.B)
    // Applied only when multiscale_mode=1: scales L0 contribution to L1 update.
    // gain[b] in [0.25, 4.0]; initialized to 1.0 (no-op).
    float byte_gain[256];

    // Oja plastic cells (Phase 43.C)
    // When eta_oja > 0: first SEE_N_OJA cells of L1 fast band use learned
    // projection W_oja[j] instead of identity (L0[j]).
    // Oja rule: w_j += eta * y_j * (L0 - y_j * w_j)  where y_j = W_oja[j] · L0
    // W_oja identity-initialized; norm-clamped to |w| <= 2.0 per step.
    // W_oja persists across see_reset (survives document boundaries).
    // To reset W_oja to identity, call see_oja_reset().
#define SEE_N_OJA 13   // ~10% of SEE_L1_DIM (128)
    float W_oja[SEE_N_OJA][43];
    float eta_oja;           // Oja learning rate (0.0 = disabled)
} SiliconEntropyState;

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the Silicon Entropy Engine
void see_init(SiliconEntropyState* see, int seed, int chunk_size, float decay);

// Reset the internal state (useful for epoch boundaries or new files).
// NOTE: does NOT reset W_oja — Oja weights survive document boundaries.
void see_reset(SiliconEntropyState* see);

// Reset Oja weights to identity (call only when you want to restart unsupervised learning)
void see_oja_reset(SiliconEntropyState* see);

// Observe a true byte, update internal representations (L0 and L1)
void see_observe(SiliconEntropyState* see, uint8_t byte);

// Extract the 192D feature vector to predict the NEXT byte
void see_extract(const SiliconEntropyState* see, float out_features[SEE_FEATURE_DIM]);

#ifdef __cplusplus
}
#endif

#endif // SILICON_ENTROPY_H
