// C50-T2 stitching cell: dumps named intermediate tensors of a pinned llama.cpp forward, per position, through the
// scheduler's eval callback (no change to the graph or the numerics).
//
// Input  (--ids): one document per line, "<doc_index>\t<id>,<id>,...", as c50_llama_score; every token is an input,
//                 each document is one batch (length <= --ubatch), context cleared per document.
// Output (--out DIR):
//   DIR/<name>-<layer>.bin  for each requested base name and layer: one row per position, in file order, of the
//                           tensor's ne[0] elements, F32 tensors as float32 and I32 tensors as int32, little endian
//   DIR/run.json            model, ids file size and FNV-1a 64, names, layers, threads, ubatch, ctx, and per tensor
//                           file its dtype, row width and rows; written at the end, so it exists only for a whole run
// Every position is an output (logits for all), so the last layer is not cut to the output rows.
// Numerics as c50_llama_score: KV cache F16, flash attention disabled.  Prints one CONFIG line; the caller asserts it.
//
// Build: c50_build_scorer.sh c50_llama_dump (CPU, pinned static libraries).

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string model, ids, out, names;
    std::string layers;  // "a-b" or comma list
    int threads = 6, ubatch = 512, ctx = 1024;
};

struct Sink {
    FILE * f = nullptr;
    ggml_type type = GGML_TYPE_F32;
    int64_t width = 0;
    long long rows = 0;
};

struct State {
    std::set<std::string> want;
    std::map<std::string, Sink> sinks;
    std::string out;
    int expect_rows = 0;  // tokens in the current batch
    std::vector<unsigned char> tmp;
    std::string error;  // set by the callback, which stops the compute instead of throwing through ggml
};

std::string read_file(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

std::string json_str(const std::string & s) {  // a JSON string literal: quotes, backslashes, control bytes escaped
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += static_cast<char>(c); }
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += static_cast<char>(c);
    }
    return o + "\"";
}

uint64_t fnv1a(const unsigned char * p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

std::vector<std::pair<int, std::vector<llama_token>>> parse_ids(const std::string & text) {
    std::vector<std::pair<int, std::vector<llama_token>>> docs;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) throw std::runtime_error("ids line without a tab");
        std::vector<llama_token> ids;
        std::stringstream ss(line.substr(tab + 1));
        std::string tok;
        while (std::getline(ss, tok, ',')) ids.push_back(static_cast<llama_token>(std::stol(tok)));
        docs.emplace_back(std::stoi(line.substr(0, tab)), ids);
    }
    return docs;
}

std::vector<int> parse_layers(const std::string & s) {
    std::vector<int> v;
    const size_t dash = s.find('-');
    if (dash != std::string::npos) {
        for (int i = std::stoi(s.substr(0, dash)); i <= std::stoi(s.substr(dash + 1)); ++i) v.push_back(i);
        return v;
    }
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) v.push_back(std::stoi(t));
    return v;
}

void take(ggml_tensor * t, State * st);

bool on_tensor(ggml_tensor * t, bool ask, void * ud) {
    auto * st = static_cast<State *>(ud);
    if (ask) return st->want.count(ggml_get_name(t)) > 0;
    try {
        take(t, st);
    } catch (const std::exception & e) {
        st->error = e.what();
        return false;
    }
    return true;
}

void take(ggml_tensor * t, State * st) {
    const std::string name = ggml_get_name(t);
    if (!st->want.count(name)) return;
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32) throw std::runtime_error("unsupported dtype: " + name);
    if (t->ne[2] != 1 || t->ne[3] != 1) throw std::runtime_error("tensor is not 2-D: " + name);
    if (t->ne[1] != st->expect_rows) throw std::runtime_error("rows differ from the batch: " + name);
    Sink & s = st->sinks[name];
    if (!s.f) {
        s.f = fopen((st->out + "/" + name + ".bin").c_str(), "wb");
        if (!s.f) throw std::runtime_error("cannot open the dump of " + name);
        s.type = t->type;
        s.width = t->ne[0];
    }
    if (s.type != t->type || s.width != t->ne[0]) throw std::runtime_error("shape changed: " + name);
    const size_t es = ggml_type_size(t->type), row = es * static_cast<size_t>(t->ne[0]);
    st->tmp.resize(row * static_cast<size_t>(t->ne[1]));
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, st->tmp.data(), 0, st->tmp.size());
    } else {
        if (!t->buffer || !ggml_backend_buffer_is_host(t->buffer)) throw std::runtime_error("strided non-host: " + name);
        const auto * base = static_cast<const unsigned char *>(t->data);
        for (int64_t r = 0; r < t->ne[1]; ++r)
            for (int64_t c = 0; c < t->ne[0]; ++c)
                memcpy(st->tmp.data() + r * row + c * es, base + r * t->nb[1] + c * t->nb[0], es);
    }
    if (fwrite(st->tmp.data(), 1, st->tmp.size(), s.f) != st->tmp.size()) throw std::runtime_error("write failed");
    s.rows += t->ne[1];
}

void quiet(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

}  // namespace

int run(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--model") o.model = next();
        else if (a == "--ids") o.ids = next();
        else if (a == "--out") o.out = next();
        else if (a == "--names") o.names = next();
        else if (a == "--layers") o.layers = next();
        else if (a == "--threads") o.threads = std::stoi(next());
        else if (a == "--ubatch") o.ubatch = std::stoi(next());
        else if (a == "--ctx") o.ctx = std::stoi(next());
        else throw std::runtime_error("unknown option " + a);
    }
    if (o.model.empty() || o.ids.empty() || o.out.empty() || o.names.empty() || o.layers.empty())
        throw std::runtime_error("need --model --ids --out --names --layers");
    if (!read_file(o.out + "/run.json").empty()) throw std::runtime_error("run.json exists; refusing to overwrite");

    const std::string text = read_file(o.ids);
    const auto docs = parse_ids(text);
    State st;
    st.out = o.out;
    std::vector<std::string> names;
    {
        std::stringstream ss(o.names);
        std::string n;
        while (std::getline(ss, n, ',')) names.push_back(n);
    }
    const std::vector<int> layers = parse_layers(o.layers);
    for (const auto & n : names)
        for (int l : layers) st.want.insert(n + "-" + std::to_string(l));

    llama_log_set(quiet, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(o.model.c_str(), mp);
    if (!model) throw std::runtime_error("model load failed");
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
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
    cp.cb_eval = on_tensor;
    cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) throw std::runtime_error("context init failed");

    size_t positions = 0;
    for (const auto & d : docs) {
        if (d.second.empty() || static_cast<int>(d.second.size()) > o.ubatch)
            throw std::runtime_error("document length outside [1, ubatch]");
        for (llama_token t : d.second)
            if (t < 0 || t >= n_vocab) throw std::runtime_error("token id outside the GGUF vocabulary");
        positions += d.second.size();
    }
    printf("CONFIG model=%s docs=%zu positions=%zu names=%s layers=%s tensors=%zu threads=%d ubatch=%d ctx=%d "
           "flash_attn=disabled kv=f16 system_info=[%s]\n", o.model.c_str(), docs.size(), positions, o.names.c_str(),
           o.layers.c_str(), st.want.size(), o.threads, o.ubatch, o.ctx, llama_print_system_info());
    fflush(stdout);

    llama_batch batch = llama_batch_init(o.ubatch, 0, 1);
    size_t done = 0;
    for (size_t di = 0; di < docs.size(); ++di) {
        const auto & ids = docs[di].second;
        const int n = static_cast<int>(ids.size());
        llama_memory_clear(llama_get_memory(ctx), true);
        batch.n_tokens = n;
        for (int j = 0; j < n; ++j) {
            batch.token[j] = ids[j];
            batch.pos[j] = j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = 1;
        }
        st.expect_rows = n;
        if (llama_decode(ctx, batch) != 0 || !st.error.empty())
            throw std::runtime_error("llama_decode failed: " + st.error);
        done += n;
        if ((di + 1) % 8 == 0 || di + 1 == docs.size()) {
            printf("DOCS %zu/%zu positions=%zu\n", di + 1, docs.size(), done);
            fflush(stdout);
        }
    }
    llama_batch_free(batch);

    std::ostringstream rj;
    rj << "{\"model\":" << json_str(o.model) << ",\"ids_bytes\":" << text.size() << ",\"ids_fnv1a64\":\"";
    char hx[17];
    snprintf(hx, sizeof hx, "%016llx", static_cast<unsigned long long>(fnv1a(
        reinterpret_cast<const unsigned char *>(text.data()), text.size())));
    rj << hx << "\",\"docs\":" << docs.size() << ",\"positions\":" << positions << ",\"names\":" << json_str(o.names)
       << ",\"layers\":" << json_str(o.layers) << ",\"threads\":" << o.threads << ",\"ubatch\":" << o.ubatch
       << ",\"ctx\":" << o.ctx << ",\"tensors\":{";
    bool first = true;
    for (const auto & w : st.want) {
        auto it = st.sinks.find(w);
        if (it == st.sinks.end()) throw std::runtime_error("never seen: " + w);
        fclose(it->second.f);
        if (it->second.rows != static_cast<long long>(positions)) throw std::runtime_error("row count: " + w);
        rj << (first ? "" : ",") << "\"" << w << "\":{\"dtype\":\"" << (it->second.type == GGML_TYPE_F32 ? "f32" : "i32")
           << "\",\"width\":" << it->second.width << ",\"rows\":" << it->second.rows << "}";
        first = false;
    }
    rj << "}}\n";
    std::ofstream(o.out + "/run.json", std::ios::binary) << rj.str();
    printf("DONE docs=%zu positions=%zu tensors=%zu\n", docs.size(), positions, st.want.size());
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "ERROR %s\n", e.what());
        return 1;
    }
}
