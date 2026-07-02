#include "silicon_r2.h"
#include <stdlib.h>
#include <string.h>

void silicon_r2_init(SiliconR2* r2, int seed, float alpha) {
    r2->alpha = alpha;
    srand(seed);
    for (int j = 0; j < R2_DIM; j++)
        for (int i = 0; i < R2_IN; i++)
            r2->W_in[j][i] = (rand() & 1) ? 1 : -1;
    silicon_r2_reset(r2);
}

void silicon_r2_reset(SiliconR2* r2) {
    memset(r2->h, 0, R2_DIM * sizeof(float));
}

void silicon_r2_update(SiliconR2* r2, const float* features_192d) {
    float scale = 1.0f / R2_IN;
    for (int j = 0; j < R2_DIM; j++) {
        float inj = 0.0f;
        const int8_t* w = r2->W_in[j];
        for (int i = 0; i < R2_IN; i++) inj += w[i] * features_192d[i];
        r2->h[j] = r2->alpha * r2->h[j] + inj * scale;
    }
}

void silicon_r2_extract(const SiliconR2* r2, float* out_64d) {
    memcpy(out_64d, r2->h, R2_DIM * sizeof(float));
}
