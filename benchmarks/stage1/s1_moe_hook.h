// Stage 1 (a): fewer routed experts per token, applied inside pinned llama.cpp through the graph
// evaluation callback.  The callback observes every "ffn_moe_weights-<layer>" tensor: the routing
// weights of the n_expert_used slots of each token, [1, n_expert_used, n_tokens], F32, taken from the
// unbiased probabilities in slot order.  Slot j is rank j of the BIASED selection score
// (ggml_argsort_top_k sorts descending; the bias balances expert load), so keeping slots 0..k-1 is the
// native top-k.  A dropped slot gets weight 0 before the norm_topk_prob division, so the kept slots are
// renormalized as the native top-k would be, and the dropped expert's output enters the sum as an
// exact zero.
//
// Modes (treatment and control share this one code path):
//   none     observe only; nothing changes
//   top      keep slots 0..k-1 (the native top-k by biased score)
//   topw     keep the k slots with the highest UNBIASED weight (top-k by weight, not by biased score)
//   random   keep k slots drawn uniformly without replacement (the trivial control of top and topw)
//   topp     Top-P (Huang et al. 2024) on the pretrained top-8: sort the slots by UNBIASED weight,
//            descending, and keep the shortest prefix whose normalized weight sums to >= p
//   randp    keep as many slots as topp would, drawn uniformly (the trivial control of topp)
//   pgrid    observe only; accumulate, for each p of a grid, the slots topp would keep (calibration)
// decode_only: the mode acts only while the caller has set in_decode (generated tokens); prompt
// positions keep all slots.
// The random draws come from a splitmix64 stream keyed by the seed, the layer and a per-layer token
// counter, so a rerun on the same inputs in the same order draws the same slots.
// Counters: tensors, distinct layers, slots, kept, the same for decode tokens, and the tokens whose
// weights are not non-increasing in slot order (biased order differs from weight order).
//
// Returning false from the callback only stops the current split, and llama_decode still reports
// success: every failure is recorded in MoeHook::error, which the caller checks after every decode.
#pragma once

#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace s1 {

constexpr int MAX_LAYERS = 128;
constexpr int MAX_USED = 64;
constexpr int GRID = 66;  // p = 0.30, 0.31, ..., 0.95

struct MoeHook {
    std::string mode = "none";
    int k = 0;
    double p = 0.0;
    uint64_t seed = 0;
    bool decode_only = false;
    bool in_decode = false;               // set by the caller around generation decodes
    int n_used = 0;                       // n_expert_used seen in the tensors (must stay constant)
    uint64_t tensors = 0, slots = 0, kept = 0, dslots = 0, dkept = 0, tokens_seen = 0, misordered = 0;
    uint64_t layer_tokens[MAX_LAYERS] = {};
    std::vector<double> grid_kept = std::vector<double>(GRID, 0.0);
    uint64_t grid_tokens = 0;
    std::string error;
};

inline bool valid_mode(const std::string & m) {
    return m == "none" || m == "top" || m == "topw" || m == "random" || m == "topp" || m == "randp" || m == "pgrid";
}

inline uint64_t splitmix(uint64_t & s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// slot indices by unbiased weight, descending, ties to the lower slot
inline void weight_order(const float * w, int n, int * ord) {
    for (int j = 0; j < n; ++j) ord[j] = j;
    std::stable_sort(ord, ord + n, [&](int a, int b) { return w[a] > w[b]; });
}

inline int topp_count(const float * w, const int * ord, int n, double p) {
    double sum = 0.0;
    for (int j = 0; j < n; ++j) sum += w[j];
    if (!(sum > 0.0)) return n;
    double c = 0.0;
    for (int j = 0; j < n; ++j) {
        c += w[ord[j]] / sum;
        if (c >= p) return j + 1;
    }
    return n;
}

// keep m slots drawn uniformly without replacement (partial Fisher-Yates); zero the others
inline void keep_random(float * w, int n, int m, uint64_t & state) {
    int idx[MAX_USED];
    for (int j = 0; j < n; ++j) idx[j] = j;
    for (int j = 0; j < m; ++j) {
        const int r = j + static_cast<int>(splitmix(state) % static_cast<uint64_t>(n - j));
        std::swap(idx[j], idx[r]);
    }
    bool keep[MAX_USED] = {};
    for (int j = 0; j < m; ++j) keep[idx[j]] = true;
    for (int j = 0; j < n; ++j) if (!keep[j]) w[j] = 0.0f;
}

inline int layers_seen(const MoeHook & h) {
    int c = 0;
    for (int l = 0; l < MAX_LAYERS; ++l) c += h.layer_tokens[l] ? 1 : 0;
    return c;
}

inline bool moe_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    static const char PREFIX[] = "ffn_moe_weights-";
    const char * name = ggml_get_name(t);
    if (std::strncmp(name, PREFIX, sizeof(PREFIX) - 1) != 0) return !ask;
    const char * tail = name + sizeof(PREFIX) - 1;   // the layer number only, not "_norm-..."
    if (!*tail) return !ask;
    for (const char * c = tail; *c; ++c) if (*c < '0' || *c > '9') return !ask;
    if (ask) return true;
    auto * h = static_cast<MoeHook *>(user_data);
    auto fail = [&](const char * why) { if (h->error.empty()) h->error = why; return false; };
    if (t->type != GGML_TYPE_F32 || t->ne[0] != 1 || t->ne[3] != 1 || !ggml_is_contiguous(t))
        return fail("ffn_moe_weights: unexpected layout");
    const int n = static_cast<int>(t->ne[1]);
    const int64_t n_tok = t->ne[2];
    const int layer = std::atoi(tail);
    if (layer < 0 || layer >= MAX_LAYERS || n < 1 || n > MAX_USED) return fail("ffn_moe_weights: layer or width out of range");
    if (h->n_used == 0) h->n_used = n;
    if (h->n_used != n) return fail("ffn_moe_weights: n_expert_used changed");
    if (!valid_mode(h->mode)) return fail("unknown --moe-mode");
    if ((h->mode == "top" || h->mode == "topw" || h->mode == "random") && (h->k < 1 || h->k > n))
        return fail("--moe-k outside [1, n_expert_used]");
    if (h->in_decode && n_tok != 1) return fail("decode phase with more than one token");
    const bool act = !h->decode_only || h->in_decode;
    float * w = static_cast<float *>(t->data);
    h->tensors += 1;
    int ord[MAX_USED];
    for (int64_t i = 0; i < n_tok; ++i) {
        float * row = w + i * n;
        uint64_t state = h->seed ^ (static_cast<uint64_t>(layer) << 48) ^ h->layer_tokens[layer];
        splitmix(state);
        h->layer_tokens[layer] += 1;
        h->tokens_seen += 1;
        for (int j = 1; j < n; ++j) if (row[j] > row[j - 1]) { h->misordered += 1; break; }
        int m = n;
        if (act && h->mode == "top") {
            m = h->k;
            for (int j = m; j < n; ++j) row[j] = 0.0f;
        } else if (act && h->mode == "topw") {
            m = h->k;
            weight_order(row, n, ord);
            for (int j = m; j < n; ++j) row[ord[j]] = 0.0f;
        } else if (act && h->mode == "random") {
            m = h->k;
            keep_random(row, n, m, state);
        } else if (act && h->mode == "topp") {
            weight_order(row, n, ord);
            m = topp_count(row, ord, n, h->p);
            for (int j = m; j < n; ++j) row[ord[j]] = 0.0f;
        } else if (act && h->mode == "randp") {
            weight_order(row, n, ord);
            m = topp_count(row, ord, n, h->p);
            keep_random(row, n, m, state);
        } else if (h->mode == "pgrid") {
            weight_order(row, n, ord);
            for (int g = 0; g < GRID; ++g) h->grid_kept[g] += topp_count(row, ord, n, 0.30 + 0.01 * g);
            h->grid_tokens += 1;
        }
        h->slots += n;
        h->kept += m;
        if (h->in_decode) {
            h->dslots += n;
            h->dkept += m;
        }
    }
    return true;
}

// one line the caller asserts: the treatment must have touched every layer it claims to change
inline void print_moe(const MoeHook & h, uint64_t input_tokens) {
    auto mean = [&](uint64_t kept, uint64_t slots) { return slots ? static_cast<double>(kept) * h.n_used / slots : 0.0; };
    printf("MOE mode=%s k=%d p=%.4f seed=%llu decode_only=%d n_expert_used=%d layers=%d input_tokens=%llu tensors=%llu "
           "slots=%llu kept=%llu mean_kept=%.4f decode_slots=%llu decode_mean_kept=%.4f misordered=%llu tokens_seen=%llu\n",
           h.mode.c_str(), h.k, h.p, static_cast<unsigned long long>(h.seed), h.decode_only ? 1 : 0, h.n_used,
           layers_seen(h), static_cast<unsigned long long>(input_tokens), static_cast<unsigned long long>(h.tensors),
           static_cast<unsigned long long>(h.slots), static_cast<unsigned long long>(h.kept), mean(h.kept, h.slots),
           static_cast<unsigned long long>(h.dslots), mean(h.dkept, h.dslots),
           static_cast<unsigned long long>(h.misordered), static_cast<unsigned long long>(h.tokens_seen));
    if (h.mode == "pgrid" && h.grid_tokens) {
        printf("PGRID tokens=%llu", static_cast<unsigned long long>(h.grid_tokens));
        for (int g = 0; g < GRID; ++g) printf(" %.2f:%.4f", 0.30 + 0.01 * g, h.grid_kept[g] / h.grid_tokens);
        printf("\n");
    }
    fflush(stdout);
}

}  // namespace s1
