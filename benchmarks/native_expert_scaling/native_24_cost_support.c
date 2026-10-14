/* Cache-state control only; never interpreted as a DRAM traffic counter. */
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#define API __declspec(dllexport)
API void cost_evict(const void *p, size_t n) {
    if (!n) return;
    uintptr_t lo = (uintptr_t)p & ~(uintptr_t)63;
    uintptr_t hi = (uintptr_t)p + n;
    for (uintptr_t a = lo; a < hi; a += 64) _mm_clflush((const void *)a);
    _mm_mfence();
}
/* Same signature as the qualified row ABI, for wrapper overhead measurement. */
API int cost_noop(void *ctx, int n, const void *tags, const void *rows,
    const void *w, const void *conv, const void *q, const void *scale,
    const void *f, const void *bits, const void *r, const void *dots,
    const void *a, const void *raw) {
    return n == 1 && ctx && tags && rows && w && conv && q && scale &&
        f && bits && r && dots && a && raw ? 0 : 1;
}
