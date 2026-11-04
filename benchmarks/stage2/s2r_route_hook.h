// Stage 2, routing by conversation: writes which experts each token chose, through the graph evaluation callback.  The
// model and the graph are unchanged.  After s1::moe_cb has acted, "ffn_moe_topk-<layer>" (the selected ids, in biased order)
// and "ffn_moe_weights_norm-<layer>" (the weights, a dropped slot at 0) go to a file, one record per layer and ubatch:
//   int32 layer, int32 n_tokens, int32 n_used, int16 ids[n_tokens * n_used], float weights[n_tokens * n_used]
#pragma once

#include "ggml.h"
#include "s1_moe_hook.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace s2r {

struct Route {
    s1::MoeHook * hook = nullptr;
    FILE * f = nullptr;
    std::vector<int16_t> ids;
    int ids_layer = -1;
    int64_t ids_tok = 0, n_used = 0;
    uint64_t records = 0, tokens = 0;
    ~Route() { if (f) fclose(f); }
};

inline bool route_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * r = static_cast<Route *>(user_data);
    const char * name = ggml_get_name(t);
    const int lt = s1::layer_of(name, "ffn_moe_topk-"), lw = s1::layer_of(name, "ffn_moe_weights_norm-");
    if (ask) return lt >= 0 || lw >= 0 || s1::moe_cb(t, true, r->hook);
    if (!s1::moe_cb(t, false, r->hook)) return false;
    auto fail = [&](const char * why) { if (r->hook->error.empty()) r->hook->error = why; return false; };
    if (lt >= 0) {
        if (t->type != GGML_TYPE_I32 || t->ne[2] != 1 || t->ne[3] != 1) return fail("route: ffn_moe_topk layout");
        r->n_used = t->ne[0];
        r->ids_tok = t->ne[1];
        r->ids.resize(static_cast<size_t>(r->n_used * r->ids_tok));
        for (int64_t i = 0; i < r->ids_tok; ++i)
            for (int64_t j = 0; j < r->n_used; ++j)
                r->ids[i * r->n_used + j] = static_cast<int16_t>(*reinterpret_cast<const int32_t *>(
                    static_cast<const char *>(t->data) + j * t->nb[0] + i * t->nb[1]));
        r->ids_layer = lt;
    } else if (lw >= 0) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || t->ne[0] != r->n_used || t->ne[1] != r->ids_tok || r->ids_layer != lw)
            return fail("route: ffn_moe_weights_norm does not follow its ids");
        const int32_t head[3] = {lw, static_cast<int32_t>(r->ids_tok), static_cast<int32_t>(r->n_used)};
        fwrite(head, sizeof head, 1, r->f);
        fwrite(r->ids.data(), sizeof(int16_t), r->ids.size(), r->f);
        fwrite(t->data, sizeof(float), r->ids.size(), r->f);
        r->records += 1;
        r->tokens += static_cast<uint64_t>(r->ids_tok);
    }
    return true;
}

inline void print_route(const Route & r) {
    printf("ROUTE records=%llu layer_tokens=%llu\n", static_cast<unsigned long long>(r.records), static_cast<unsigned long long>(r.tokens));
}

}  // namespace s2r
