// C50 reference generator: greedy continuations from pinned llama.cpp on source-tokenizer IDs
// (roadmap A4: document rollout and generated task answers, both arms through the same binary).
//
// Input  (--prompts): one prompt per line, "<item_index>\t<id>,<id>,...", IDs from the source
//         tokenizer; no special token is added by this program.
// Output (--out PREFIX):
//   PREFIX.jsonl  one line per item: {"item":i,"prompt_tokens":n,"stop":"EOS"|"MAX","gen":[ids]}
//   PREFIX.json   runtime record (CONFIG fields, items, generated tokens, wall seconds)
// Decoding: the prompt is evaluated in micro-batches of --ubatch, then one token per decode.
// Greedy: argmax of the logits, exact ties to the lowest ID.  Stops on any ID of --stop (that ID
// is recorded in "gen") or after --max-new tokens.  KV F16, flash attention disabled, context
// cleared per item.  --max-projected-s S: after the first item, stop (exit 3) if the projected
// wall time of all items exceeds S seconds.  Prints and the caller asserts one CONFIG line.
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
    std::string model, prompts, out;
    int threads = 6, ubatch = 512, ctx = 4096, max_new = 256;
    std::vector<llama_token> stop;
    double max_projected_s = 0.0;
};

void quiet(enum ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

std::vector<llama_token> parse_ids(const std::string & s) {
    std::vector<llama_token> ids;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) ids.push_back(static_cast<llama_token>(std::stol(tok)));
    return ids;
}

std::vector<std::pair<int, std::vector<llama_token>>> read_prompts(const std::string & path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open prompts file");
    std::vector<std::pair<int, std::vector<llama_token>>> items;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        size_t tab = line.find('\t');
        if (tab == std::string::npos) throw std::runtime_error("prompt line without tab");
        items.emplace_back(std::stoi(line.substr(0, tab)), parse_ids(line.substr(tab + 1)));
    }
    return items;
}

std::string json_escape(const std::string & s) {
    std::string o;
    for (char c : s) {
        if (c == '\\' || c == '"') { o += '\\'; o += c; }
        else if (static_cast<unsigned char>(c) < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += c;
    }
    return o;
}

llama_token greedy(const float * lg, int n_vocab) {
    int arg = 0;
    float mx = lg[0];
    for (int v = 1; v < n_vocab; ++v)
        if (lg[v] > mx) { mx = lg[v]; arg = v; }  // strict: exact ties keep the lowest ID
    return static_cast<llama_token>(arg);
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
        else if (a == "--prompts") o.prompts = next();
        else if (a == "--out") o.out = next();
        else if (a == "--threads") o.threads = std::stoi(next());
        else if (a == "--ubatch") o.ubatch = std::stoi(next());
        else if (a == "--ctx") o.ctx = std::stoi(next());
        else if (a == "--max-new") o.max_new = std::stoi(next());
        else if (a == "--stop") o.stop = parse_ids(next());
        else if (a == "--max-projected-s") o.max_projected_s = std::stod(next());
        else throw std::runtime_error("unknown argument " + a);
    }
    if (o.model.empty() || o.prompts.empty() || o.out.empty() || o.max_new < 1) {
        fprintf(stderr, "usage: --model GGUF --prompts FILE --out PREFIX [--max-new N] [--stop id,id] "
                        "[--threads N] [--ubatch N] [--ctx N] [--max-projected-s S]\n");
        return 2;
    }
    auto items = read_prompts(o.prompts);
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
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) throw std::runtime_error("context init failed");

    size_t prompt_tokens = 0;
    for (auto & it : items) {
        if (it.second.empty() || static_cast<int>(it.second.size()) + o.max_new > o.ctx)
            throw std::runtime_error("prompt empty or prompt + max_new beyond ctx");
        for (llama_token t : it.second)
            if (t < 0 || t >= n_vocab) throw std::runtime_error("token id outside the GGUF vocabulary");
        prompt_tokens += it.second.size();
    }
    for (llama_token t : o.stop)
        if (t < 0 || t >= n_vocab) throw std::runtime_error("stop id outside the GGUF vocabulary");
    std::string stop_s;
    for (size_t k = 0; k < o.stop.size(); ++k) stop_s += (k ? "," : "") + std::to_string(o.stop[k]);
    printf("CONFIG model=%s n_vocab=%d items=%zu prompt_tokens=%zu max_new=%d stop=%s greedy=lowest_id_ties "
           "threads=%d ubatch=%d ctx=%d flash_attn=disabled kv=f16 max_projected_s=%.0f system_info=[%s]\n",
           o.model.c_str(), n_vocab, items.size(), prompt_tokens, o.max_new, stop_s.c_str(), o.threads, o.ubatch,
           o.ctx, o.max_projected_s, llama_print_system_info());
    fflush(stdout);

    FILE * fo = fopen((o.out + ".jsonl").c_str(), "wb");
    if (!fo) throw std::runtime_error("cannot open output");
    llama_batch batch = llama_batch_init(o.ubatch, 0, 1);
    auto t0 = std::chrono::steady_clock::now();
    size_t generated = 0;
    for (size_t k = 0; k < items.size(); ++k) {
        const auto & ids = items[k].second;
        const int n = static_cast<int>(ids.size());
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int s = 0; s < n; s += o.ubatch) {
            const int len = std::min(o.ubatch, n - s);
            batch.n_tokens = len;
            for (int j = 0; j < len; ++j) {
                batch.token[j] = ids[s + j];
                batch.pos[j] = s + j;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j] = (s + j == n - 1) ? 1 : 0;
            }
            if (llama_decode(ctx, batch) != 0) throw std::runtime_error("llama_decode failed (prompt)");
        }
        std::vector<llama_token> gen;
        const char * stop_reason = "MAX";
        llama_token t = greedy(llama_get_logits_ith(ctx, -1), n_vocab);
        for (int step = 0; step < o.max_new; ++step) {
            gen.push_back(t);
            bool is_stop = false;
            for (llama_token st : o.stop) if (t == st) is_stop = true;
            if (is_stop) { stop_reason = "EOS"; break; }
            if (step + 1 == o.max_new) break;
            batch.n_tokens = 1;
            batch.token[0] = t;
            batch.pos[0] = n + step;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            if (llama_decode(ctx, batch) != 0) throw std::runtime_error("llama_decode failed (generation)");
            t = greedy(llama_get_logits_ith(ctx, -1), n_vocab);
        }
        generated += gen.size();
        fprintf(fo, "{\"item\":%d,\"prompt_tokens\":%d,\"stop\":\"%s\",\"gen\":[", items[k].first, n, stop_reason);
        for (size_t j = 0; j < gen.size(); ++j) fprintf(fo, "%s%d", j ? "," : "", gen[j]);
        fprintf(fo, "]}\n");
        fflush(fo);
        if (k == 0) {
            const double t1 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double projected = t1 * static_cast<double>(items.size());
            printf("FIRST_ITEM generated=%zu wall_s=%.1f projected_s=%.0f\n", gen.size(), t1, projected);
            fflush(stdout);
            if (o.max_projected_s > 0.0 && projected > o.max_projected_s) {
                printf("STOP projected_s=%.0f exceeds max_projected_s=%.0f\n", projected, o.max_projected_s);
                fclose(fo);
                return 3;
            }
        }
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fclose(fo);
    llama_batch_free(batch);
    std::ofstream js(o.out + ".json");
    js << "{\"model\":\"" << json_escape(o.model) << "\",\"n_vocab\":" << n_vocab << ",\"items\":" << items.size()
       << ",\"prompt_tokens\":" << prompt_tokens << ",\"generated_tokens\":" << generated << ",\"max_new\":" << o.max_new
       << ",\"stop\":\"" << stop_s << "\",\"threads\":" << o.threads << ",\"ubatch\":" << o.ubatch << ",\"ctx\":" << o.ctx
       << ",\"wall_s\":" << wall << "}\n";
    printf("DONE items=%zu generated=%zu wall_s=%.1f\n", items.size(), generated, wall);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
