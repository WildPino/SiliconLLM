// Stage 1 scorer: c50_llama_score (per-position next-token statistics from pinned llama.cpp on
// source-tokenizer IDs) with the routed-expert modes of s1_moe_hook.h and a top-K output for KL.
//
// Input  (--ids): one document per line, "<doc_index>\t<id>,<id>,...", as c50_llama_score.
// Output (--out PREFIX):
//   PREFIX.pos.bin   as c50_llama_score: 32 bytes per scored position p = 1..n-1 of each document
//                    (int32 doc, pos, target, top1; float logp_target, logp_top1; double logsumexp)
//   PREFIX.top.bin   (only with --top K) per scored position a 32-byte head (int32 doc, pos, target,
//                    top1; float logp_target; int32 K; double logsumexp) and K x (int32 id, float logp).
//                    Without --ref-top the ids are this run's own top-K, by logp descending, ties by
//                    id.  With --ref-top FILE the ids are those of FILE at the same position (doc, pos
//                    and target must match), and logp is this run's: the reference's top-K partition
//                    on which the caller computes KL(reference || this run).
//   PREFIX.json      runtime record (CONFIG and MOE fields, positions, wall seconds)
// Expert modes (--moe-mode none|top|random|topp|randp|pgrid, --moe-k, --moe-p, --moe-seed): see
// s1_moe_hook.h.  Every position here is a prompt position, so there is no decode-only mode.  --no-hook runs without the callback (the c50 path); --override-experts K sets the
// GGUF key <arch>.expert_used_count to K (llama.cpp's own top-K, for the parity check only).
// Numerics as c50_llama_score: logits for every position, log-sum-exp in double, KV F16, flash
// attention disabled, context cleared per document.  Prints one CONFIG and one MOE line.
//
// Build: benchmarks/stage1/s1_build.sh s1_llama_score

#include "llama.h"
#include "s1_moe_hook.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string model, ids, out, ref_top, mode = "none";
    int threads = 6, ubatch = 512, ctx = 4096, top = 0, moe_k = 0, override_experts = 0;
    double moe_p = 0.0;
    unsigned long long moe_seed = 0;
    bool no_hook = false;
    double max_projected_s = 0.0;
};

struct Rec {
    int32_t doc, pos, target, top1;
    float logp_target, logp_top1;
    double lse;
};
static_assert(sizeof(Rec) == 32, "record layout");

struct TopHead {
    int32_t doc, pos, target, top1;
    float logp_target;
    int32_t k;
    double lse;
};
static_assert(sizeof(TopHead) == 32, "record layout");

struct Entry {
    int32_t id;
    float logp;
};
static_assert(sizeof(Entry) == 8, "record layout");

void quiet(enum ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

std::vector<std::pair<int, std::vector<llama_token>>> read_ids(const std::string & path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open ids file");
    std::vector<std::pair<int, std::vector<llama_token>>> docs;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        size_t tab = line.find('\t');
        if (tab == std::string::npos) throw std::runtime_error("ids line without tab");
        int doc = std::stoi(line.substr(0, tab));
        std::vector<llama_token> ids;
        std::stringstream ss(line.substr(tab + 1));
        std::string tok;
        while (std::getline(ss, tok, ',')) ids.push_back(static_cast<llama_token>(std::stol(tok)));
        docs.emplace_back(doc, std::move(ids));
    }
    return docs;
}

}  // namespace

int main(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--model") o.model = next();
        else if (a == "--ids") o.ids = next();
        else if (a == "--out") o.out = next();
        else if (a == "--threads") o.threads = std::stoi(next());
        else if (a == "--ubatch") o.ubatch = std::stoi(next());
        else if (a == "--ctx") o.ctx = std::stoi(next());
        else if (a == "--top") o.top = std::stoi(next());
        else if (a == "--ref-top") o.ref_top = next();
        else if (a == "--moe-mode") o.mode = next();
        else if (a == "--moe-k") o.moe_k = std::stoi(next());
        else if (a == "--moe-p") o.moe_p = std::stod(next());
        else if (a == "--moe-seed") o.moe_seed = std::stoull(next());
        else if (a == "--no-hook") o.no_hook = true;
        else if (a == "--override-experts") o.override_experts = std::stoi(next());
        else if (a == "--max-projected-s") o.max_projected_s = std::stod(next());
        else throw std::runtime_error("unknown argument " + a);
    }
    if (o.model.empty() || o.ids.empty() || o.out.empty() || !s1::valid_mode(o.mode) || o.top < 0 || o.top > 4096
        || (!o.ref_top.empty() && o.top == 0) || (o.no_hook && o.mode != "none")) {
        fprintf(stderr, "usage: --model GGUF --ids FILE --out PREFIX [--top K [--ref-top FILE]] [--moe-mode M --moe-k K "
                        "--moe-p P --moe-seed S | --no-hook] [--override-experts K] [--threads N] [--ubatch N] [--ctx N]\n");
        return 2;
    }
    auto docs = read_ids(o.ids);
    llama_log_set(quiet, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    std::vector<llama_model_kv_override> kvo;
    std::string override_key;
    if (o.override_experts > 0) {
        override_key = "bailingmoe2.expert_used_count";
        llama_model_kv_override ov{};
        ov.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        std::snprintf(ov.key, sizeof ov.key, "%s", override_key.c_str());
        ov.val_i64 = o.override_experts;
        kvo.push_back(ov);
        kvo.push_back(llama_model_kv_override{});  // terminator: key[0] == 0
        mp.kv_overrides = kvo.data();
    }
    llama_model * model = llama_model_load_from_file(o.model.c_str(), mp);
    if (!model) throw std::runtime_error("model load failed");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (o.top > n_vocab) throw std::runtime_error("--top beyond the vocabulary");
    s1::MoeHook hook;
    hook.mode = o.mode;
    hook.k = o.moe_k;
    hook.p = o.moe_p;
    hook.seed = o.moe_seed;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = static_cast<uint32_t>(o.ctx);
    cp.n_batch = static_cast<uint32_t>(o.ubatch);
    cp.n_ubatch = static_cast<uint32_t>(o.ubatch);
    cp.n_seq_max = 1;
    cp.n_threads = o.threads;
    cp.n_threads_batch = o.threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    cp.no_perf = true;
    if (!o.no_hook) {
        cp.cb_eval = s1::moe_cb;
        cp.cb_eval_user_data = &hook;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) throw std::runtime_error("context init failed");

    size_t positions = 0;
    for (auto & d : docs) {
        if (d.second.size() < 2 || static_cast<int>(d.second.size()) > o.ctx)
            throw std::runtime_error("document length outside [2, ctx]");
        for (llama_token t : d.second)
            if (t < 0 || t >= n_vocab) throw std::runtime_error("token id outside the GGUF vocabulary");
        positions += d.second.size() - 1;
    }
    printf("CONFIG model=%s n_vocab=%d docs=%zu positions=%zu threads=%d ubatch=%d ctx=%d flash_attn=disabled "
           "kv=f16 top=%d ref_top=%s hook=%d moe_mode=%s moe_k=%d moe_p=%.4f moe_seed=%llu override_experts=%d "
           "max_projected_s=%.0f system_info=[%s]\n", o.model.c_str(), n_vocab, docs.size(), positions, o.threads,
           o.ubatch, o.ctx, o.top, o.ref_top.empty() ? "-" : o.ref_top.c_str(), o.no_hook ? 0 : 1, o.mode.c_str(),
           o.moe_k, o.moe_p, o.moe_seed, o.override_experts, o.max_projected_s, llama_print_system_info());
    fflush(stdout);

    FILE * fpos = fopen((o.out + ".pos.bin").c_str(), "wb");
    FILE * ftop = o.top ? fopen((o.out + ".top.bin").c_str(), "wb") : nullptr;
    FILE * fref = o.ref_top.empty() ? nullptr : fopen(o.ref_top.c_str(), "rb");
    if (!fpos || (o.top && !ftop) || (!o.ref_top.empty() && !fref)) throw std::runtime_error("cannot open outputs or --ref-top");
    std::vector<Entry> ent(o.top);
    std::vector<int> order(n_vocab);
    llama_batch batch = llama_batch_init(o.ubatch, 0, 1);
    auto t0 = std::chrono::steady_clock::now();
    bool first_doc = true;
    for (auto & d : docs) {
        const auto & ids = d.second;
        const int n = static_cast<int>(ids.size());
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int s = 0; s < n - 1; s += o.ubatch) {
            const int len = std::min(o.ubatch, n - 1 - s);  // the last token is never an input
            batch.n_tokens = len;
            for (int j = 0; j < len; ++j) {
                batch.token[j] = ids[s + j];
                batch.pos[j] = s + j;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j] = 1;
            }
            if (llama_decode(ctx, batch) != 0) throw std::runtime_error("llama_decode failed");
            if (!hook.error.empty()) throw std::runtime_error(hook.error);
            for (int j = 0; j < len; ++j) {
                const float * lg = llama_get_logits_ith(ctx, j);
                double mx = -INFINITY;
                int arg = 0;
                for (int v = 0; v < n_vocab; ++v)
                    if (lg[v] > mx) { mx = lg[v]; arg = v; }
                double sum = 0.0;
                for (int v = 0; v < n_vocab; ++v) sum += std::exp(static_cast<double>(lg[v]) - mx);
                const double lse = mx + std::log(sum);
                const int p = s + j + 1;
                Rec r{d.first, p, ids[p], arg, static_cast<float>(lg[ids[p]] - lse),
                      static_cast<float>(lg[arg] - lse), lse};
                fwrite(&r, sizeof r, 1, fpos);
                if (ftop) {
                    TopHead h{d.first, p, ids[p], arg, r.logp_target, o.top, lse};
                    if (fref) {
                        TopHead rh;
                        if (fread(&rh, sizeof rh, 1, fref) != 1 || rh.doc != h.doc || rh.pos != h.pos
                            || rh.target != h.target || rh.k != o.top
                            || fread(ent.data(), sizeof(Entry), o.top, fref) != static_cast<size_t>(o.top))
                            throw std::runtime_error("--ref-top does not match this position");
                        for (auto & e : ent) e.logp = static_cast<float>(lg[e.id] - lse);
                    } else {
                        std::iota(order.begin(), order.end(), 0);
                        std::partial_sort(order.begin(), order.begin() + o.top, order.end(), [&](int x, int y) {
                            return lg[x] > lg[y] || (lg[x] == lg[y] && x < y);
                        });
                        for (int q = 0; q < o.top; ++q) ent[q] = Entry{order[q], static_cast<float>(lg[order[q]] - lse)};
                    }
                    fwrite(&h, sizeof h, 1, ftop);
                    fwrite(ent.data(), sizeof(Entry), o.top, ftop);
                }
            }
        }
        if (first_doc) {
            first_doc = false;
            const double t1 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double projected = t1 * static_cast<double>(positions) / static_cast<double>(ids.size() - 1);
            printf("FIRST_DOC positions=%zu wall_s=%.1f projected_s=%.0f\n", ids.size() - 1, t1, projected);
            fflush(stdout);
            if (o.max_projected_s > 0.0 && projected > o.max_projected_s) {
                printf("STOP projected_s=%.0f exceeds max_projected_s=%.0f\n", projected, o.max_projected_s);
                return 3;
            }
        }
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (fref) {
        TopHead extra;
        if (fread(&extra, sizeof extra, 1, fref) == 1) throw std::runtime_error("--ref-top has more positions than this run");
        fclose(fref);
    }
    fclose(fpos);
    if (ftop) fclose(ftop);
    llama_batch_free(batch);
    s1::print_moe(hook, positions);  // every scored position is one input token
    std::ofstream js(o.out + ".json");
    js << "{\"model\":\"" << o.model << "\",\"n_vocab\":" << n_vocab << ",\"docs\":" << docs.size()
       << ",\"positions\":" << positions << ",\"threads\":" << o.threads << ",\"ubatch\":" << o.ubatch
       << ",\"ctx\":" << o.ctx << ",\"flash_attn\":\"disabled\",\"kv\":\"f16\",\"top\":" << o.top
       << ",\"hook\":" << (o.no_hook ? 0 : 1) << ",\"moe_mode\":\"" << o.mode << "\",\"moe_k\":" << o.moe_k
       << ",\"moe_p\":" << o.moe_p << ",\"moe_seed\":" << o.moe_seed << ",\"override_experts\":" << o.override_experts
       << ",\"moe_tensors\":" << hook.tensors << ",\"moe_slots\":" << hook.slots << ",\"moe_kept\":" << hook.kept
       << ",\"wall_s\":" << wall << "}\n";
    printf("DONE positions=%zu wall_s=%.1f\n", positions, wall);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
