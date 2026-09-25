#include <cstddef>

extern "C" float strat01_layer1_f32_router_oracle(
        const float *x, const float *y, unsigned count) {
    double sum = 0.0;
    for (unsigned i = 0; i < count; ++i) {
        sum += (double) (x[i] * y[i]);
    }
    return (float) sum;
}
