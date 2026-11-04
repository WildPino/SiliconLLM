// Stage 2 test 2b: the head in two passes, as the panel tools apply it.  The preview's logits come from a second
// context on a GGUF that differs from the model only in output.weight, so the hidden state is the same and both heads
// run on llama.cpp's own kernels.  The candidates are the K largest preview logits; they take the model's exact logits.
// The tail keeps the preview's logits, plus the offset mean(exact - preview) over the candidates (--tail-offset 1).
// The token is chosen among the candidates only: a tail token can hold probability, never win the greedy choice.
#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace s2t {

struct TwoPass {
    int k = 0;
    bool tail_offset = true;
    uint64_t positions = 0, missed = 0;   // the exact head's argmax outside the candidates
    std::vector<int> order;
    std::vector<float> z;

    // Fills z; returns the greedy token: the candidates' largest exact logit, exact ties to the lowest id.
    int compose(const float * exact, const float * preview, int n_vocab) {
        if (static_cast<int>(order.size()) != n_vocab) {
            order.resize(n_vocab);
            z.resize(n_vocab);
        }
        std::iota(order.begin(), order.end(), 0);
        std::nth_element(order.begin(), order.begin() + (k - 1), order.end(), [&](int x, int y) {
            return preview[x] > preview[y] || (preview[x] == preview[y] && x < y);
        });
        double diff = 0.0;
        for (int i = 0; i < k; ++i) diff += static_cast<double>(exact[order[i]]) - preview[order[i]];
        const float off = tail_offset ? static_cast<float>(diff / k) : 0.0f;
        for (int v = 0; v < n_vocab; ++v) z[v] = preview[v] + off;
        int arg = order[0];
        for (int i = 0; i < k; ++i) {
            const int c = order[i];
            z[c] = exact[c];
            if (exact[c] > exact[arg] || (exact[c] == exact[arg] && c < arg)) arg = c;
        }
        int full = 0;
        for (int v = 1; v < n_vocab; ++v) if (exact[v] > exact[full]) full = v;
        positions += 1;
        missed += full != arg;
        return arg;
    }
};

}  // namespace s2t
