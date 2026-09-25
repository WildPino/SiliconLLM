#include <cstddef>

extern "C" void strat01_router_weight_norm_oracle(
        const float weights[4], float normalized[4]) {
    double sum = 0.0;
    for (std::size_t i = 0; i < 4U; ++i) sum += static_cast<double>(weights[i]);
    float denominator = static_cast<float>(sum);
    if (denominator < 6.103515625e-5f) denominator = 6.103515625e-5f;
    for (std::size_t i = 0; i < 4U; ++i) normalized[i] = weights[i] / denominator;
}
