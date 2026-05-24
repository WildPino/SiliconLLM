import re

with open('benchmarks/benchmark10_audit.c', 'r') as f:
    code = f.read()

simd_code = '''
#include <immintrin.h>

static inline float dot_product_simd(const float* w, const float* f, int n) {
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) {
        sum = _mm256_fmadd_ps(_mm256_loadu_ps(&w[i]), _mm256_loadu_ps(&f[i]), sum);
    }
    float out[8];
    _mm256_storeu_ps(out, sum);
    return out[0] + out[1] + out[2] + out[3] + out[4] + out[5] + out[6] + out[7];
}

static inline void grad_update_simd(float* gradW, const float* f, float err_div_batch, int n) {
    __m256 err_vec = _mm256_set1_ps(err_div_batch);
    for (int i = 0; i < n; i += 8) {
        __m256 gw = _mm256_loadu_ps(&gradW[i]);
        __m256 fv = _mm256_loadu_ps(&f[i]);
        gw = _mm256_fmadd_ps(err_vec, fv, gw);
        _mm256_storeu_ps(&gradW[i], gw);
    }
}
'''

code = code.replace('#include "../src/silicon_v0.h"', '#include "../src/silicon_v0.h"\n' + simd_code)

# Replace dot product in train_logistic_regression
dot_train = '''
                for (int f = 0; f < num_features; f++) logits[c] += model->W[c][f] * features_train[i * num_features + f];
'''
code = code.replace(dot_train, '''
                logits[c] += dot_product_simd(model->W[c], &features_train[i * num_features], num_features);
''')

# Replace grad update in train_logistic_regression
grad_train = '''
                for (int f = 0; f < num_features; f++) {
                    gradW[c * MAX_FEATURES + f] += err * features_train[i * num_features + f] / batch_size;
                }
'''
code = code.replace(grad_train, '''
                grad_update_simd(&gradW[c * MAX_FEATURES], &features_train[i * num_features], err / batch_size, num_features);
''')

# Replace dot product in evaluate_model
dot_eval = '''
            for (int f = 0; f < num_features; f++) logits[c] += model->W[c][f] * features_test[i * num_features + f];
'''
code = code.replace(dot_eval, '''
            logits[c] += dot_product_simd(model->W[c], &features_test[i * num_features], num_features);
''')

with open('benchmarks/benchmark10_audit.c', 'w') as f:
    f.write(code)
print("Vectorized successfully.")
