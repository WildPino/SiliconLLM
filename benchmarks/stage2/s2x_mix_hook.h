// Stage 2 test 3b: experts read at two precisions in one token, inside pinned llama.cpp, through the graph evaluation
// callback.  The model in the context has ternary experts.  After "ffn_moe_down-<layer>" ([n_embd, n_used, n_tokens], each
// slot's expert output before the routing weight), the chosen slots are recomputed from another GGUF's experts (the
// original Q4_K_M) on the same input, "ffn_norm-<layer>", and written over the slot.  The router, the weights and every
// other tensor are the context's.
//
// Modes (treatment and controls share this path):
//   top     the slot with the highest routing weight, the weight after the norm, ties to the lower slot
//   random  one slot with a non-zero weight, drawn from a splitmix64 stream keyed by seed, layer and token counter
//   all     every slot with a non-zero weight: on the model the file came from, this replays it
//   wrong   the top slot, read from the next expert's weights: the planted control
// Layers outside --exact-layers keep the context's experts.  The slots run ggml's own row products (the type's vec_dot, the
// activations in its vec_dot format), so a slot read from the context's own file equals the kernel's up to the sum's order.
//
// Returning false from the callback only stops the split: failures go to MoeHook::error, which the caller checks.
#pragma once

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "s1_moe_hook.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace s2x {

struct Weights {
    std::vector<uint8_t> bytes;
    ggml_type type = GGML_TYPE_COUNT;
    int64_t row = 0, rows = 0, expert_bytes = 0, row_bytes = 0;
};

struct Mix {
    s1::MoeHook * hook = nullptr;
    std::string mode = "top";
    uint64_t seed = 0;
    bool layer_on[s1::MAX_LAYERS] = {};
    Weights gate[s1::MAX_LAYERS], up[s1::MAX_LAYERS], down[s1::MAX_LAYERS];
    int n_expert = 0;
    int n_threads = 6;
    std::vector<float> x;                 // ffn_norm of x_layer
    std::vector<int32_t> ids;             // ffn_moe_topk of ids_layer
    std::vector<float> w;                 // ffn_moe_weights_norm of w_layer
    int x_layer = -1, ids_layer = -1, w_layer = -1;
    int64_t x_tok = 0, ids_tok = 0, w_tok = 0;
    int n_used = 0;
    uint64_t counter[s1::MAX_LAYERS] = {};
    uint64_t tokens = 0, replaced = 0;
    double top_weight = 0.0, read_weight = 0.0;   // sums over token-layers: the highest weight, the weight of the slots read
    double diff2 = 0.0, old2 = 0.0;       // sum of squares: new slot - context's slot, and the context's slot
    double rel_sum = 0.0;                 // sum over slots of each slot's own relative error: massive tokens do not dominate it
    std::vector<std::vector<uint8_t>> scratch;
};

inline bool valid_mode(const std::string & m) { return m == "top" || m == "random" || m == "all" || m == "wrong"; }

// "1-19" or "1-6,10": the layers in which slots are read exact
inline void parse_layers(Mix & m, const std::string & s) {
    size_t i = 0;
    while (i < s.size()) {
        size_t e = s.find(',', i);
        if (e == std::string::npos) e = s.size();
        const std::string part = s.substr(i, e - i);
        const size_t d = part.find('-');
        const int a = std::stoi(part.substr(0, d)), b = d == std::string::npos ? a : std::stoi(part.substr(d + 1));
        if (a < 0 || b >= s1::MAX_LAYERS || b < a) throw std::runtime_error("--exact-layers out of range");
        for (int l = a; l <= b; ++l) m.layer_on[l] = true;
        i = e + 1;
    }
}

// Reads the chosen layers' experts of a GGUF into memory.
inline void load(Mix & m, const std::string & path) {
    ggml_context * meta = nullptr;
    gguf_init_params ip{true, &meta};
    gguf_context * g = gguf_init_from_file(path.c_str(), ip);
    if (!g) throw std::runtime_error("cannot read --exact-from");
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open --exact-from");
    const size_t base = gguf_get_data_offset(g);
    for (int l = 0; l < s1::MAX_LAYERS; ++l) {
        if (!m.layer_on[l]) continue;
        const std::pair<const char *, Weights *> parts[3] = {{"gate", &m.gate[l]}, {"up", &m.up[l]}, {"down", &m.down[l]}};
        for (const auto & p : parts) {
            const std::string name = "blk." + std::to_string(l) + ".ffn_" + p.first + "_exps.weight";
            const int64_t id = gguf_find_tensor(g, name.c_str());
            if (id < 0) throw std::runtime_error("--exact-from has no " + name);
            ggml_tensor * t = ggml_get_tensor(meta, name.c_str());
            Weights & w = *p.second;
            w.type = t->type;
            w.row = t->ne[0];
            w.rows = t->ne[1];
            w.row_bytes = static_cast<int64_t>(ggml_row_size(t->type, t->ne[0]));
            w.expert_bytes = w.row_bytes * w.rows;
            m.n_expert = static_cast<int>(t->ne[2]);
            w.bytes.resize(ggml_nbytes(t));
            _fseeki64(f, static_cast<long long>(base + gguf_get_tensor_offset(g, id)), SEEK_SET);
            if (fread(w.bytes.data(), 1, w.bytes.size(), f) != w.bytes.size()) throw std::runtime_error("short read of " + name);
        }
    }
    fclose(f);
    gguf_free(g);
    ggml_free(meta);
}

struct Job { int64_t token; int slot; };

// One expert over its tokens: silu(gate x) * (up x), then down, on ggml's own row products: the activations go through
// the type's vec_dot format, as in mul_mat_id.
inline void forward(Mix & m, int l, int e, const std::vector<Job> & jobs, const float * x, int64_t n_embd, char * out,
                    size_t nb1, size_t nb2, std::vector<uint8_t> & buf, bool inner) {
    const Weights & G = m.gate[l], & U = m.up[l], & D = m.down[l];
    const auto * tg = ggml_get_type_traits_cpu(G.type), * tu = ggml_get_type_traits_cpu(U.type), * td = ggml_get_type_traits_cpu(D.type);
    const int64_t ff = G.rows;
    const size_t sg = ggml_row_size(tg->vec_dot_type, n_embd), su = ggml_row_size(tu->vec_dot_type, n_embd),
                 sd = ggml_row_size(td->vec_dot_type, ff);
    const size_t pad = (sg + su + sd + 63) & ~size_t(63);
    buf.resize(pad + ff * sizeof(float) + 64);
    uint8_t * qg = buf.data(), * qu = qg + sg, * qa = qu + su;
    float * act = reinterpret_cast<float *>(buf.data() + pad);
    const uint8_t * bg = G.bytes.data() + static_cast<size_t>(e) * G.expert_bytes;
    const uint8_t * bu = U.bytes.data() + static_cast<size_t>(e) * U.expert_bytes;
    const uint8_t * bd = D.bytes.data() + static_cast<size_t>(e) * D.expert_bytes;
    for (const Job & j : jobs) {
        const float * in = x + j.token * n_embd;
        ggml_get_type_traits_cpu(tg->vec_dot_type)->from_float(in, qg, n_embd);
        ggml_get_type_traits_cpu(tu->vec_dot_type)->from_float(in, qu, n_embd);
#pragma omp parallel for if (inner)
        for (int64_t r = 0; r < ff; ++r) {
            float a = 0.0f, b = 0.0f;
            tg->vec_dot(static_cast<int>(n_embd), &a, 0, bg + r * G.row_bytes, 0, qg, 0, 1);
            tu->vec_dot(static_cast<int>(n_embd), &b, 0, bu + r * U.row_bytes, 0, qu, 0, 1);
            act[r] = a / (1.0f + std::exp(-a)) * b;
        }
        ggml_get_type_traits_cpu(td->vec_dot_type)->from_float(act, qa, ff);
        float * y = reinterpret_cast<float *>(out + j.slot * nb1 + j.token * nb2);
        double e2 = 0.0, o2 = 0.0;
#pragma omp parallel for if (inner) reduction(+ : e2, o2)
        for (int64_t r = 0; r < D.rows; ++r) {
            float v = 0.0f;
            td->vec_dot(static_cast<int>(ff), &v, 0, bd + r * D.row_bytes, 0, qa, 0, 1);
            e2 += static_cast<double>(v - y[r]) * (v - y[r]);
            o2 += static_cast<double>(y[r]) * y[r];
            y[r] = v;
        }
#pragma omp atomic
        m.diff2 += e2;
#pragma omp atomic
        m.old2 += o2;
        const double rel = o2 > 0.0 ? std::sqrt(e2 / o2) : 0.0;
#pragma omp atomic
        m.rel_sum += rel;
    }
}

inline bool replace(Mix & m, int l, ggml_tensor * t) {
    s1::MoeHook * h = m.hook;
    auto fail = [&](const char * why) { if (h->error.empty()) h->error = why; return false; };
    const int64_t n_embd = t->ne[0], n = t->ne[1], n_tok = t->ne[2];
    if (t->type != GGML_TYPE_F32 || t->ne[3] != 1 || t->nb[0] != sizeof(float)) return fail("ffn_moe_down: unexpected layout");
    if (m.x_layer != l || m.ids_layer != l || m.w_layer != l || m.x_tok != n_tok || m.ids_tok != n_tok || m.w_tok != n_tok
        || m.n_used != n || static_cast<int64_t>(m.x.size()) != n_embd * n_tok)
        return fail("ffn_moe_down: input, ids or weights of this layer were not seen first");
    std::map<int, std::vector<Job>> groups;
    for (int64_t i = 0; i < n_tok; ++i) {
        const float * w = &m.w[i * n];
        const int32_t * id = &m.ids[i * n];
        int top = 0, live = 0;
        for (int j = 0; j < n; ++j) {
            if (w[j] > w[top]) top = j;
            live += w[j] > 0.0f;
        }
        uint64_t state = m.seed ^ (static_cast<uint64_t>(l) << 48) ^ m.counter[l]++;
        s1::splitmix(state);
        m.tokens += 1;
        m.top_weight += w[top];
        if (m.mode == "all") {
            for (int j = 0; j < n; ++j) if (w[j] > 0.0f) {
                groups[id[j]].push_back({i, j});
                m.replaced += 1;
                m.read_weight += w[j];
            }
            continue;
        }
        int slot = top;
        if (m.mode == "random") {
            int pick = static_cast<int>(s1::splitmix(state) % static_cast<uint64_t>(live));
            for (int j = 0; j < n; ++j) if (w[j] > 0.0f && pick-- == 0) { slot = j; break; }
        }
        const int e = m.mode == "wrong" ? (id[slot] + 1) % m.n_expert : id[slot];
        groups[e].push_back({i, slot});
        m.replaced += 1;
        m.read_weight += w[slot];
    }
    std::vector<std::pair<int, std::vector<Job> *>> list;
    for (auto & kv : groups) list.push_back({kv.first, &kv.second});
    char * out = static_cast<char *>(t->data);
    if (static_cast<int>(list.size()) >= m.n_threads) {
        m.scratch.resize(m.n_threads);
#pragma omp parallel for schedule(dynamic, 1) num_threads(m.n_threads)
        for (int i = 0; i < static_cast<int>(list.size()); ++i)
            forward(m, l, list[i].first, *list[i].second, m.x.data(), n_embd, out, t->nb[1], t->nb[2],
                    m.scratch[omp_get_thread_num()], false);
    } else {
        m.scratch.resize(std::max<size_t>(m.scratch.size(), 1));
        for (auto & p : list) forward(m, l, p.first, *p.second, m.x.data(), n_embd, out, t->nb[1], t->nb[2], m.scratch[0], true);
    }
    return true;
}

inline bool mix_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * m = static_cast<Mix *>(user_data);
    const char * name = ggml_get_name(t);
    int l, kind = 0;
    if ((l = s1::layer_of(name, "ffn_norm-")) >= 0) kind = 1;
    else if ((l = s1::layer_of(name, "ffn_moe_topk-")) >= 0) kind = 2;
    else if ((l = s1::layer_of(name, "ffn_moe_weights_norm-")) >= 0) kind = 3;
    else if ((l = s1::layer_of(name, "ffn_moe_down-")) >= 0) kind = 4;
    const bool mine = kind && l < s1::MAX_LAYERS && m->layer_on[l];
    if (ask) return s1::moe_cb(t, true, m->hook) || mine;
    if (!s1::moe_cb(t, false, m->hook)) return false;
    if (!mine) return true;
    const char * base = static_cast<const char *>(t->data);
    if (kind == 1) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) { m->hook->error = "ffn_norm: unexpected layout"; return false; }
        m->x_layer = l;
        m->x_tok = t->ne[1];
        m->x.assign(reinterpret_cast<const float *>(base), reinterpret_cast<const float *>(base) + ggml_nelements(t));
    } else if (kind == 2) {
        m->ids_layer = l;
        m->ids_tok = t->ne[1];
        m->n_used = static_cast<int>(t->ne[0]);
        m->ids.resize(static_cast<size_t>(t->ne[0]) * t->ne[1]);
        for (int64_t i = 0; i < t->ne[1]; ++i)
            for (int64_t j = 0; j < t->ne[0]; ++j)
                m->ids[i * t->ne[0] + j] = *reinterpret_cast<const int32_t *>(base + j * t->nb[0] + i * t->nb[1]);
    } else if (kind == 3) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) { m->hook->error = "ffn_moe_weights_norm: unexpected layout"; return false; }
        m->w_layer = l;
        m->w_tok = t->ne[1];
        m->w.assign(reinterpret_cast<const float *>(base), reinterpret_cast<const float *>(base) + ggml_nelements(t));
    } else {
        return replace(*m, l, t);
    }
    return true;
}

inline void print_mix(const Mix & m) {
    int layers = 0;
    for (bool on : m.layer_on) layers += on;
    printf("MIX mode=%s layers=%d token_layers=%llu replaced=%llu mean_top_weight=%.5f mean_read_weight=%.5f slot_rel_err=%.6f mean_slot_rel_err=%.6f\n",
           m.mode.c_str(), layers, static_cast<unsigned long long>(m.tokens), static_cast<unsigned long long>(m.replaced),
           m.tokens ? m.top_weight / m.tokens : 0.0, m.tokens ? m.read_weight / m.tokens : 0.0,
           m.old2 > 0.0 ? std::sqrt(m.diff2 / m.old2) : 0.0, m.replaced ? m.rel_sum / m.replaced : 0.0);
}

}  // namespace s2x
