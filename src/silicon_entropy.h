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

    // Oja plastic cells (Phase 43.C / 43.C2)
    // The first n_oja cells of the L1 fast band use a learned projection
    // W_oja[j] (dot with L0[0:43]) instead of plain identity L0[j]. The
    // projection is ALWAYS applied (identity by default => bit-identical to
    // SEE-V1S); the Oja *update* only runs when eta_oja > 0:
    //   w_j += eta * y_j * (L0 - y_j * w_j)  where y_j = W_oja[j] · L0
    // W_oja identity-initialized; norm-clamped to |w| <= 2.0 per step.
    // W_oja persists across see_reset (survives document boundaries).
    // To reset W_oja to identity, call see_oja_reset().
    //
    // n_oja is a RUNTIME field (Phase 43.C2): plastic capacity can scale
    // (13 -> 26 ...) without recompiling. Storage is sized to SEE_N_OJA_MAX
    // (full fast band) and the active count is carried in the weight-file
    // header (self-describing, magic 0x53454539).
    //
    // plastic_blend (Phase 43.C3, homeostasis): the injected plastic feature is
    //   h = h_base + beta * (h_oja - h_base) = (1-beta)*L0[j] + beta*(W_oja[j].L0)
    // beta=1.0 => pure Oja projection (43.C2 behavior, default, backward-compat).
    // beta<1.0 brakes the learned projection toward the raw identity base so more
    // plastic cells can add signal without drifting OOD into semantic tunnels.
    // The Oja *update* still uses the raw projection y (learning is unconstrained;
    // only the readout-facing feature is tamed). Carried in header magic 0x5345453A.
#define SEE_N_OJA 13       // default plastic cell count (SEE-V2, ~10% of fast band)
#define SEE_N_OJA_MAX 43   // max plastic cells = full L1 fast band; sizes W_oja storage
    float W_oja[SEE_N_OJA_MAX][43];
    float eta_oja;           // Oja learning rate (0.0 = disabled, projection still applied)
    int   n_oja;             // active plastic cells (runtime; default SEE_N_OJA, <= SEE_N_OJA_MAX)
    float plastic_blend;     // homeostatic blend beta in [0,1]; default 1.0 = pure Oja

    // Byte-to-lane routing (Phase 43.D) — geometry of writing.
    // byte_route[b][k] reshapes WHERE byte b deposits its M4 codebook signature
    // (input lanes l0_out[0:32]) into the L1 FAST band only. The routed input
    //   l0_fast_in[k] = (k<32 ? byte_route[b][k] : 1) * l0_out[k]
    // feeds both the plastic (Oja) projection and the identity fast-band write;
    // mid/slow bands use raw l0_out. Rows are mean-1.0 (geometry, not amplitude)
    // and clamped near 1.0. Default all 1.0 => no-op (backward compatible).
    // Carried in header magic 0x5345453B.
#define SEE_ROUTE_LANES 32   // M4 codebook lanes routed (l0_out[0:32])
    float byte_route[256][SEE_ROUTE_LANES];
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
