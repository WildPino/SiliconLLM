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
//
// mass_k (stage 1 (a), part 1b; arXiv 2609.04575, Eq. 2): 0 keeps the behaviour above (the kept slots
// renormalized among themselves, k2 = k1).  mass_k >= n_expert_used moves the hook after the
// norm_topk_prob division, onto "ffn_moe_weights_norm-<layer>" ([n_expert_used, n_tokens], each row
// p_sel / S, S = the mass of the selected slots).  The same selection zeroes the dropped slots there,
// and the kept slots are multiplied by S / D, where D is S plus the (mass_k - n_expert_used) largest
// unbiased probabilities among the experts not selected.  So mass_k = n_expert_used keeps each kept
// expert's weight at its top-8 value (k2 = the trained k), and mass_k = 16 divides by a top-16 mass.
// For this the hook also copies "ffn_moe_probs-<layer>" ([n_expert, n_tokens], the unbiased
// probabilities) and "ffn_moe_topk-<layer>" (the selected ids, a strided view).  Every row is checked:
// it must sum to 1 and equal probs[id] / S slot by slot, or the hook records an error.
#pragma once

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
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
    int mass_k = 0;                       // 0: renormalize over the kept slots; >= n_used: see above
    std::vector<float> probs;             // copied from ffn_moe_probs-<probs_layer>
    std::vector<int32_t> ids;             // copied from ffn_moe_topk-<ids_layer>
    int probs_layer = -1, ids_layer = -1, n_expert = 0;
    int64_t probs_tok = 0, ids_tok = 0;
    double mass_factor_sum = 0.0;         // sum over rows of S / D
    uint64_t mass_rows = 0;
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

// the layer number if name is PREFIX followed by digits only, else -1
inline int layer_of(const char * name, const char * prefix) {
    const size_t n = std::strlen(prefix);
    if (std::strncmp(name, prefix, n) != 0 || !name[n]) return -1;
    for (const char * c = name + n; *c; ++c) if (*c < '0' || *c > '9') return -1;
    return std::atoi(name + n);
}

// S / D for one token: S = the mass of the selected slots, D = S plus the (mass_k - n) largest
// unbiased probabilities among the experts not selected
inline double mass_factor(const MoeHook & h, const float * p, const int32_t * id, int n, double s) {
    const int extra = h.mass_k - n;
    if (extra <= 0) return 1.0;
    std::vector<float> rest;
    rest.reserve(h.n_expert);
    std::vector<bool> sel(h.n_expert, false);
    for (int j = 0; j < n; ++j) sel[id[j]] = true;
    for (int e = 0; e < h.n_expert; ++e) if (!sel[e]) rest.push_back(p[e]);
    const int m = std::min<int>(extra, static_cast<int>(rest.size()));
    std::nth_element(rest.begin(), rest.begin() + m, rest.end(), std::greater<float>());
    double d = s;
    for (int j = 0; j < m; ++j) d += rest[j];
    return d > 0.0 ? s / d : 1.0;
}

inline bool moe_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    const char * name = ggml_get_name(t);
    auto * h = static_cast<MoeHook *>(user_data);
    const bool mass = h->mass_k != 0;
    // 1: the weights before the norm (mass_k = 0); 2, 3: copies for the mass; 4: the weights after the norm
    int layer = layer_of(name, "ffn_moe_weights-"), kind = 0;
    if (layer >= 0) kind = mass ? 0 : 1;
    else if (mass && (layer = layer_of(name, "ffn_moe_probs-")) >= 0) kind = 2;
    else if (mass && (layer = layer_of(name, "ffn_moe_topk-")) >= 0) kind = 3;
    else if (mass && (layer = layer_of(name, "ffn_moe_weights_norm-")) >= 0) kind = 4;
    if (kind == 0) return !ask;
    if (ask) return true;
    auto fail = [&](const char * why) { if (h->error.empty()) h->error = why; return false; };
    if (layer >= MAX_LAYERS) return fail("layer out of range");
    if (kind == 2) {
        if (t->type != GGML_TYPE_F32 || t->ne[2] != 1 || t->ne[3] != 1 || t->nb[0] != sizeof(float))
            return fail("ffn_moe_probs: unexpected layout");
        h->n_expert = static_cast<int>(t->ne[0]);
        h->probs_tok = t->ne[1];
        h->probs.resize(static_cast<size_t>(h->n_expert) * h->probs_tok);
        for (int64_t i = 0; i < h->probs_tok; ++i)
            std::memcpy(&h->probs[i * h->n_expert], static_cast<const char *>(t->data) + i * t->nb[1], h->n_expert * sizeof(float));
        h->probs_layer = layer;
        return true;
    }
    if (kind == 3) {
        if (t->type != GGML_TYPE_I32 || t->ne[2] != 1 || t->ne[3] != 1 || t->ne[0] > MAX_USED)
            return fail("ffn_moe_topk: unexpected layout");
        const int64_t n_id = t->ne[0];
        h->ids_tok = t->ne[1];
        h->ids.resize(static_cast<size_t>(n_id) * h->ids_tok);
        for (int64_t i = 0; i < h->ids_tok; ++i)
            for (int64_t j = 0; j < n_id; ++j)
                h->ids[i * n_id + j] = *reinterpret_cast<const int32_t *>(static_cast<const char *>(t->data) + j * t->nb[0] + i * t->nb[1]);
        h->ids_layer = layer;
        return true;
    }
    int n;
    int64_t n_tok;
    if (kind == 1) {
        if (t->type != GGML_TYPE_F32 || t->ne[0] != 1 || t->ne[3] != 1 || !ggml_is_contiguous(t))
            return fail("ffn_moe_weights: unexpected layout");
        n = static_cast<int>(t->ne[1]);
        n_tok = t->ne[2];
    } else {
        if (t->type != GGML_TYPE_F32 || t->ne[2] != 1 || t->ne[3] != 1 || !ggml_is_contiguous(t))
            return fail("ffn_moe_weights_norm: unexpected layout");
        n = static_cast<int>(t->ne[0]);
        n_tok = t->ne[1];
        if (h->probs_layer != layer || h->ids_layer != layer || h->probs_tok != n_tok || h->ids_tok != n_tok
            || static_cast<int64_t>(h->ids.size()) != n * n_tok)
            return fail("ffn_moe_weights_norm: probs or ids of this layer were not seen first");
        if (h->mass_k < n) return fail("--moe-mass below n_expert_used");
    }
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
        double f = 1.0;
        if (kind == 4) {
            const float * p = &h->probs[i * h->n_expert];
            const int32_t * id = &h->ids[i * n];
            double s = 0.0, rs = 0.0;
            for (int j = 0; j < n; ++j) {
                if (id[j] < 0 || id[j] >= h->n_expert) return fail("ffn_moe_topk: id out of range");
                s += p[id[j]];
                rs += row[j];
            }
            if (std::fabs(rs - 1.0) > 1e-4) return fail("ffn_moe_weights_norm: a row does not sum to 1");
            for (int j = 0; j < n; ++j)
                if (std::fabs(row[j] - p[id[j]] / s) > 1e-5) return fail("ffn_moe_weights_norm: row differs from probs[id] / S");
            f = mass_factor(*h, p, id, n, s);
        }
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
        if (kind == 4) {   // the factor goes with the selection: not on the prompt rows of a decode-only run
            if (!act) f = 1.0;
            if (f != 1.0) for (int j = 0; j < n; ++j) row[j] = static_cast<float>(row[j] * f);
            h->mass_factor_sum += f;
            h->mass_rows += 1;
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
    if (h.mass_k)
        printf("MASS mass_k=%d rows=%llu mean_factor=%.6f\n", h.mass_k, static_cast<unsigned long long>(h.mass_rows),
               h.mass_rows ? h.mass_factor_sum / h.mass_rows : 0.0);
    if (h.mode == "pgrid" && h.grid_tokens) {
        printf("PGRID tokens=%llu", static_cast<unsigned long long>(h.grid_tokens));
        for (int g = 0; g < GRID; ++g) printf(" %.2f:%.4f", 0.30 + 0.01 * g, h.grid_kept[g] / h.grid_tokens);
        printf("\n");
    }
    fflush(stdout);
}

}  // namespace s1
