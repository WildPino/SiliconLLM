// C50-S3 teacher signal (option A): per-position top-k log-probabilities from pinned llama.cpp on
// source-tokenizer IDs, written in shards so that a run can be stopped at any moment and resumed.
//
// Input  (--ids): one document per line, "<doc_index>\t<id>,<id>,...", as c50_llama_score.
// Output (--out DIR):
//   DIR/run.json            the run identity (model, ids file size and FNV-1a 64, topk, shard size,
//                           ctx, ubatch, offload); a resume with another identity is refused
//   DIR/shard_NNNNNN.bin    shard i = documents [i*S, (i+1)*S) in file order; per scored position
//                           p = 1..n-1 one record of 24 + 8*K bytes:
//                             int32 doc, int32 pos, int32 target, float logp_target, double lse,
//                             K x (int32 id, float logp), sorted by logp descending, ties by id
//   DIR/manifest.jsonl      one line per finished shard (index, documents, positions, bytes,
//                           FNV-1a 64 of the shard, wall seconds); written after the shard is
//                           renamed into place, so a line exists only for a complete shard
// Resume: shards listed in the manifest whose file has the listed size are skipped; any other
// shard is (re)computed.  A stop (Ctrl+C, power) loses at most the shard in progress.
// Numerics as c50_llama_score: logits for every position, log-sum-exp in double, KV cache F16,
// flash attention disabled, context cleared per document.  The top-k and log-sum-exp of the
// positions of a batch run on --threads workers (each position independently; deterministic).
// Offload: --ngl N puts N layers on the GPU (needs a GPU build); --cpu-moe keeps the routed
// expert tensors in host memory.  With --ngl 0 a GPU build still offloads large-batch operations
// (llama.cpp op offload).  Prints one CONFIG line; the caller asserts it.
//
// Build: c50_build_scorer.sh c50_s3_teacher (CPU, pinned static libraries) or
//        c50_build_teacher_gpu.sh (the pinned commit's official CUDA release, b11040).

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Options {
    std::string model, ids, out;
    int threads = 6, ubatch = 512, ctx = 4096, topk = 32, shard_docs = 16, ngl = 0, limit_docs = 0;
    bool cpu_moe = false;
};

struct Head {
    int32_t doc, pos, target;
    float logp_target;
    double lse;
};
static_assert(sizeof(Head) == 24, "record layout");

struct Entry {
    int32_t id;
    float logp;
};
static_assert(sizeof(Entry) == 8, "record layout");

void quiet(enum ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

uint64_t fnv1a(const void * data, size_t n, uint64_t h = 1469598103934665603ULL) {
    const auto * p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

std::string hex64(uint64_t v) {
    char b[17];
    snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
    return b;
}

std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<std::pair<int, std::vector<llama_token>>> parse_ids(const std::string & text) {
    std::vector<std::pair<int, std::vector<llama_token>>> docs;
    std::stringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
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

// Reads "shard":N and "bytes":N from each manifest line (the tool writes the lines itself).
std::map<int, long long> read_manifest(const std::string & path) {
    std::map<int, long long> done;
    std::ifstream in(path);
    std::string line;
    auto field = [](const std::string & l, const std::string & key) -> long long {
        size_t k = l.find("\"" + key + "\":");
        if (k == std::string::npos) return -1;
        return std::stoll(l.substr(k + key.size() + 3));
    };
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        long long s = field(line, "shard"), b = field(line, "bytes");
        if (s >= 0 && b >= 0) done[static_cast<int>(s)] = b;
    }
    return done;
}

long long file_size(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return -1;
    _fseeki64(f, 0, SEEK_END);
    long long n = _ftelli64(f);
    fclose(f);
    return n;
}

// One position: log-sum-exp in double over the full row, then the top-k by logit (ties: lower id).
void score_row(const float * lg, int n_vocab, int topk, llama_token target, Head & h, Entry * out) {
    double mx = -INFINITY;
    for (int v = 0; v < n_vocab; ++v) mx = std::max(mx, static_cast<double>(lg[v]));
    double sum = 0.0;
    for (int v = 0; v < n_vocab; ++v) sum += std::exp(static_cast<double>(lg[v]) - mx);
    const double lse = mx + std::log(sum);
    auto better = [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); };
    std::vector<int> heap;  // min-heap of the current top-k under `better`
    heap.reserve(topk + 1);
    for (int v = 0; v < n_vocab; ++v) {
        if (static_cast<int>(heap.size()) < topk) {
            heap.push_back(v);
            std::push_heap(heap.begin(), heap.end(), better);
        } else if (better(v, heap.front())) {
            std::pop_heap(heap.begin(), heap.end(), better);
            heap.back() = v;
            std::push_heap(heap.begin(), heap.end(), better);
        }
    }
    std::sort(heap.begin(), heap.end(), better);
    h.target = target;
    h.logp_target = static_cast<float>(lg[target] - lse);
    h.lse = lse;
    for (int k = 0; k < topk; ++k) out[k] = Entry{heap[k], static_cast<float>(lg[heap[k]] - lse)};
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
        else if (a == "--topk") o.topk = std::stoi(next());
        else if (a == "--shard-docs") o.shard_docs = std::stoi(next());
        else if (a == "--ngl") o.ngl = std::stoi(next());
        else if (a == "--cpu-moe") o.cpu_moe = true;
        else if (a == "--limit-docs") o.limit_docs = std::stoi(next());
        else throw std::runtime_error("unknown argument " + a);
    }
    if (o.model.empty() || o.ids.empty() || o.out.empty() || o.topk < 1 || o.shard_docs < 1) {
        fprintf(stderr, "usage: --model GGUF --ids FILE --out DIR [--threads N] [--ubatch N] [--ctx N] [--topk K] "
                        "[--shard-docs S] [--ngl N] [--cpu-moe] [--limit-docs N]\n");
        return 2;
    }
    const std::string ids_text = read_file(o.ids);
    if (ids_text.empty()) throw std::runtime_error("cannot read ids file");
    auto docs = parse_ids(ids_text);
    if (o.limit_docs > 0 && static_cast<int>(docs.size()) > o.limit_docs) docs.resize(o.limit_docs);

    // run identity: a resume must match it exactly
    std::ostringstream id;
    id << "{\"model\":\"" << o.model << "\",\"ids_bytes\":" << ids_text.size() << ",\"ids_fnv1a64\":\""
       << hex64(fnv1a(ids_text.data(), ids_text.size())) << "\",\"docs\":" << docs.size() << ",\"topk\":" << o.topk
       << ",\"shard_docs\":" << o.shard_docs << ",\"ctx\":" << o.ctx << ",\"ubatch\":" << o.ubatch
       << ",\"ngl\":" << o.ngl << ",\"cpu_moe\":" << (o.cpu_moe ? 1 : 0)
       << ",\"record_bytes\":" << (24 + 8 * o.topk) << "}\n";
    const std::string run_path = o.out + "/run.json", man_path = o.out + "/manifest.jsonl";
    const std::string prev = read_file(run_path);
    if (!prev.empty() && prev != id.str()) {
        fprintf(stderr, "run.json differs from this invocation; refusing to mix runs:\n%s%s", prev.c_str(), id.str().c_str());
        return 4;
    }
    if (prev.empty()) {
        std::ofstream rj(run_path, std::ios::binary);
        if (!rj) throw std::runtime_error("cannot write run.json (does --out exist?)");
        rj << id.str();
    }
    const std::map<int, long long> done = read_manifest(man_path);
    const int n_shards = static_cast<int>((docs.size() + o.shard_docs - 1) / o.shard_docs);
    auto shard_path = [&](int s) {
        char b[32];
        snprintf(b, sizeof b, "/shard_%06d.bin", s);
        return o.out + b;
    };
    std::vector<int> todo;
    for (int s = 0; s < n_shards; ++s) {
        auto it = done.find(s);
        if (it == done.end() || file_size(shard_path(s)) != it->second) todo.push_back(s);
    }

    llama_log_set(quiet, nullptr);
    llama_backend_init();
    ggml_backend_load_all();  // a GPU build loads its backends from the executable's directory; no-op when static
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = o.ngl;
    std::vector<llama_model_tensor_buft_override> ov;
    if (o.cpu_moe) {
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu) throw std::runtime_error("no CPU backend device");
        ov.push_back({"\\.ffn_(up|down|gate)_exps", ggml_backend_dev_buffer_type(cpu)});
        ov.push_back({nullptr, nullptr});
        mp.tensor_buft_overrides = ov.data();
    }
    llama_model * model = llama_model_load_from_file(o.model.c_str(), mp);
    if (!model) throw std::runtime_error("model load failed");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (o.topk > n_vocab) throw std::runtime_error("topk above the vocabulary");
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

    size_t positions = 0, todo_positions = 0;
    for (size_t i = 0; i < docs.size(); ++i) {
        const auto & d = docs[i];
        if (d.second.size() < 2 || static_cast<int>(d.second.size()) > o.ctx)
            throw std::runtime_error("document length outside [2, ctx]");
        for (llama_token t : d.second)
            if (t < 0 || t >= n_vocab) throw std::runtime_error("token id outside the GGUF vocabulary");
        positions += d.second.size() - 1;
        if (std::find(todo.begin(), todo.end(), static_cast<int>(i / o.shard_docs)) != todo.end())
            todo_positions += d.second.size() - 1;
    }
    std::string devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dv = ggml_backend_dev_get(i);
        devs += std::string(devs.empty() ? "" : ",") + ggml_backend_dev_name(dv);
    }
    printf("CONFIG model=%s n_vocab=%d docs=%zu positions=%zu shards=%d todo_shards=%zu todo_positions=%zu topk=%d "
           "shard_docs=%d threads=%d ubatch=%d ctx=%d ngl=%d cpu_moe=%d flash_attn=disabled kv=f16 devices=[%s] "
           "system_info=[%s]\n", o.model.c_str(), n_vocab, docs.size(), positions, n_shards, todo.size(),
           todo_positions, o.topk, o.shard_docs, o.threads, o.ubatch, o.ctx, o.ngl, o.cpu_moe ? 1 : 0, devs.c_str(),
           llama_print_system_info());
    fflush(stdout);

    const size_t rec_bytes = 24 + 8 * static_cast<size_t>(o.topk);
    llama_batch batch = llama_batch_init(o.ubatch, 0, 1);
    std::vector<unsigned char> buf;
    size_t run_positions = 0;
    for (int s : todo) {
        const auto t0 = std::chrono::steady_clock::now();
        const size_t first = static_cast<size_t>(s) * o.shard_docs;
        const size_t last = std::min(docs.size(), first + o.shard_docs);
        buf.clear();
        size_t shard_positions = 0;
        for (size_t di = first; di < last; ++di) {
            const auto & ids = docs[di].second;
            const int n = static_cast<int>(ids.size());
            llama_memory_clear(llama_get_memory(ctx), true);
            for (int st = 0; st < n - 1; st += o.ubatch) {
                const int len = std::min(o.ubatch, n - 1 - st);  // the last token is never an input
                batch.n_tokens = len;
                for (int j = 0; j < len; ++j) {
                    batch.token[j] = ids[st + j];
                    batch.pos[j] = st + j;
                    batch.n_seq_id[j] = 1;
                    batch.seq_id[j][0] = 0;
                    batch.logits[j] = 1;
                }
                if (llama_decode(ctx, batch) != 0) throw std::runtime_error("llama_decode failed");
                const size_t base = buf.size();
                buf.resize(base + rec_bytes * len);
                std::vector<const float *> rows(len);  // fetched on this thread; the workers only read them
                for (int j = 0; j < len; ++j) rows[j] = llama_get_logits_ith(ctx, j);
                std::vector<std::thread> pool;
                const int nw = std::max(1, std::min(o.threads, len));
                for (int w = 0; w < nw; ++w) {
                    pool.emplace_back([&, w]() {
                        for (int j = w; j < len; j += nw) {
                            unsigned char * r = buf.data() + base + rec_bytes * j;
                            Head h{docs[di].first, st + j + 1, 0, 0.0f, 0.0};
                            score_row(rows[j], n_vocab, o.topk, ids[st + j + 1], h,
                                      reinterpret_cast<Entry *>(r + sizeof(Head)));
                            memcpy(r, &h, sizeof h);
                        }
                    });
                }
                for (auto & t : pool) t.join();
                shard_positions += len;
            }
        }
        const std::string path = shard_path(s), tmp = path + ".tmp";
        FILE * f = fopen(tmp.c_str(), "wb");
        if (!f || fwrite(buf.data(), 1, buf.size(), f) != buf.size() || fclose(f) != 0)
            throw std::runtime_error("cannot write " + tmp);
        remove(path.c_str());
        if (rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("cannot rename " + tmp);
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        FILE * m = fopen(man_path.c_str(), "ab");
        if (!m) throw std::runtime_error("cannot append manifest");
        fprintf(m, "{\"shard\":%d,\"bytes\":%zu,\"docs\":[%d,%d],\"positions\":%zu,\"fnv1a64\":\"%s\",\"wall_s\":%.1f}\n", s,
                buf.size(), docs[first].first, docs[last - 1].first, shard_positions,
                hex64(fnv1a(buf.data(), buf.size())).c_str(), wall);
        fclose(m);
        run_positions += shard_positions;
        printf("SHARD %d docs=%zu positions=%zu wall_s=%.1f tok_s=%.2f run_positions=%zu/%zu\n", s, last - first,
               shard_positions, wall, shard_positions / wall, run_positions, todo_positions);
        fflush(stdout);
    }
    llama_batch_free(batch);
    printf("DONE shards=%d positions=%zu\n", n_shards, positions);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
