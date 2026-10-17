// C50 reference scorer: per-position next-token statistics from pinned llama.cpp on
// source-tokenizer IDs (roadmap A2b; reused as the reference side of A3/A4).
//
// Input  (--ids): one document per line, "<doc_index>\t<id>,<id>,...", IDs from the
//         source tokenizer (c50_holdout.py); no special token is added.
// Output (--out PREFIX):
//   PREFIX.pos.bin   one 32-byte record per scored position p = 1..n-1 of each document:
//                    int32 doc, int32 pos, int32 target, int32 top1,
//                    float logp_target, float logp_top1, double logsumexp
//   PREFIX.logp.f32  (only with --dump-logprobs) the full log-softmax row, n_vocab float32
//                    per scored position, same order as PREFIX.pos.bin
//   PREFIX.json      runtime record (CONFIG fields, positions, wall seconds)
// --max-projected-s S: after the first document, stop (exit 3) if the projected wall time of
//   all positions exceeds S seconds (protocol stop rule).
// Position p is predicted from the logits after ids[0..p-1]; the first token of each
// document is context only.  A6 variant of c50_llama_score: KV cache type (--kv f16|q8_0) and
// flash attention (--fa 0|1) are options printed on the CONFIG line (llama.cpp needs flash
// attention for a quantized V cache); context cleared per document.  Prints and the caller
// asserts one CONFIG line.
//
// Build: benchmarks/chatbot50/c50_build_scorer.sh (links the pinned static libraries).

#include "llama.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string model, ids, out;
    int threads = 6, ubatch = 512, ctx = 4096;
    bool dump = false;
    std::string kv = "f16";
    int fa = 0;
    double max_projected_s = 0.0;  // 0 = no stop rule
};

struct Rec {
    int32_t doc, pos, target, top1;
    float logp_target, logp_top1;
    double lse;
};
static_assert(sizeof(Rec) == 32, "record layout");

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
        else if (a == "--dump-logprobs") o.dump = true;
        else if (a == "--kv") o.kv = next();
        else if (a == "--fa") o.fa = std::stoi(next());
        else if (a == "--max-projected-s") o.max_projected_s = std::stod(next());
        else throw std::runtime_error("unknown argument " + a);
    }
    if (o.model.empty() || o.ids.empty() || o.out.empty()) {
        fprintf(stderr, "usage: --model GGUF --ids FILE --out PREFIX [--threads N] [--ubatch N] [--ctx N] [--dump-logprobs]\n");
        return 2;
    }
    auto docs = read_ids(o.ids);
    llama_log_set(quiet, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(o.model.c_str(), mp);
    if (!model) throw std::runtime_error("model load failed");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = static_cast<uint32_t>(o.ctx);
    cp.n_batch = static_cast<uint32_t>(o.ubatch);
    cp.n_ubatch = static_cast<uint32_t>(o.ubatch);
    cp.n_seq_max = 1;
    cp.n_threads = o.threads;
    cp.n_threads_batch = o.threads;
    if (o.kv != "f16" && o.kv != "q8_0") throw std::runtime_error("--kv must be f16 or q8_0");
    if (o.kv == "q8_0" && !o.fa) throw std::runtime_error("--kv q8_0 needs --fa 1");
    cp.flash_attn_type = o.fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = o.kv == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    cp.type_v = cp.type_k;
    cp.no_perf = true;
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
    printf("CONFIG model=%s n_vocab=%d docs=%zu positions=%zu threads=%d ubatch=%d ctx=%d flash_attn=%s "
           "kv=%s dump=%d max_projected_s=%.0f system_info=[%s]\n", o.model.c_str(), n_vocab, docs.size(),
           positions, o.threads, o.ubatch, o.ctx, o.fa ? "enabled" : "disabled", o.kv.c_str(), o.dump ? 1 : 0, o.max_projected_s, llama_print_system_info());
    fflush(stdout);

    FILE * fpos = fopen((o.out + ".pos.bin").c_str(), "wb");
    FILE * flp = o.dump ? fopen((o.out + ".logp.f32").c_str(), "wb") : nullptr;
    if (!fpos || (o.dump && !flp)) throw std::runtime_error("cannot open outputs");
    std::vector<float> row(n_vocab);
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
                if (flp) {
                    for (int v = 0; v < n_vocab; ++v) row[v] = static_cast<float>(lg[v] - lse);
                    fwrite(row.data(), sizeof(float), n_vocab, flp);
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
                fclose(fpos);
                if (flp) fclose(flp);
                return 3;
            }
        }
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fclose(fpos);
    if (flp) fclose(flp);
    llama_batch_free(batch);
    std::ofstream js(o.out + ".json");
    js << "{\"model\":\"" << o.model << "\",\"n_vocab\":" << n_vocab << ",\"docs\":" << docs.size()
       << ",\"positions\":" << positions << ",\"threads\":" << o.threads << ",\"ubatch\":" << o.ubatch
       << ",\"ctx\":" << o.ctx << ",\"flash_attn\":\"" << (o.fa ? "enabled" : "disabled") << "\",\"kv\":\"" << o.kv << "\",\"dump\":" << (o.dump ? 1 : 0)
       << ",\"wall_s\":" << wall << "}\n";
    printf("DONE positions=%zu wall_s=%.1f\n", positions, wall);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
