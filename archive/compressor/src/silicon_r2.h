#ifndef SILICON_R2_H
#define SILICON_R2_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// Silicon R2 — Second Reservoir (Phase 42.C)
// Architecture: leaky integrator ESN, fixed random binary W_in
// Input:  SEE_FEATURE_DIM (192) float features from the primary wave
// Output: R2_DIM (64) float values — temporal summary of feature dynamics
// ----------------------------------------------------------------------------

#define R2_DIM 64
#define R2_IN  192

typedef struct {
    float   h[R2_DIM];
    int8_t  W_in[R2_DIM][R2_IN];  // fixed random {-1, +1}
    float   alpha;                  // leaky decay (default 0.9)
} SiliconR2;

// Initialise W_in from seed and set alpha. Clears h.
void silicon_r2_init(SiliconR2* r2, int seed, float alpha);

// Zero the recurrent state without touching W_in.
void silicon_r2_reset(SiliconR2* r2);

// Feed one feature vector (192D) and update the reservoir state.
void silicon_r2_update(SiliconR2* r2, const float* features_192d);

// Copy current state to out_64d.
void silicon_r2_extract(const SiliconR2* r2, float* out_64d);

#ifdef __cplusplus
}
#endif

#endif // SILICON_R2_H
