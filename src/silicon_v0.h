#ifndef SILICON_V0_H
#define SILICON_V0_H

#include <stdint.h>
#include <immintrin.h>

#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// Silicon Sequence Compressor V0
// Architecture: G128_T16_C16
// Dynamics: Persistent Wave (sat_add / avg)
// Readout: Lane-aware pooled 32D
// ----------------------------------------------------------------------------

typedef struct {
    // Spatial state of the wave (128 AVX2 blocks = 4096 bytes)
    __m256i state[128];
    
    // Physical Codebook (Random Binary Single-Block)
    __m256i codebook[256];
    
    // M4 History Buffer (circular)
    uint8_t m4_buf[256];
    int m4_head;
} SiliconV0;

/**
 * Initializes the engine.
 * Generates the Codebook (Random Binary Single-Block) using the provided seed,
 * and clears the spatial state and historical buffer.
 */
void silicon_v0_init(SiliconV0* e, int codebook_seed);

/**
 * Resets exclusively the differential state of the Wave.
 * Useful for ablations or to test the engine without historical integration,
 * or at the start of a new document.
 */
void silicon_v0_reset(SiliconV0* e);

/**
 * Executes an engine tick for the input byte.
 * 1. Queues the byte in M4.
 * 2. Reinjects the spatially shifted T3 window.
 * 3. Applies damping and the 4 wave diffusion steps.
 */
void silicon_v0_tick(SiliconV0* engine, uint8_t input_byte);

/**
 * Executes an engine tick with explicit T3 token injection.
 */
void silicon_v0_tick_t(SiliconV0* engine, uint8_t input_byte, int t3_tokens);

/**
 * Extracts Lane-Aware Pooled 32D features from the spatial grid.
 * Performs lane-wise sum-pooling over the 16 spatial channels.
 * The output is a 32-double vector ready for regression/linear layer.
 */
void silicon_v0_extract_32d(const SiliconV0* engine, double* out_32d);

#ifdef __cplusplus
}
#endif

#endif // SILICON_V0_H
