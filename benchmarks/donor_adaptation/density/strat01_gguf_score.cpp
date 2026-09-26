// Bounded, per-document GGUF BPB scorer for STRAT-01.
//
// This program intentionally does not parse corpus JSON.  The companion Python
// wrapper supplies a private, tab-separated exchange file containing base64
// identifiers/text, raw UTF-8 byte counts, and the Python tokenizer IDs.  The
// scorer independently calls llama_tokenize() on every document.  In explicit
// source-ID mode the companion wrapper must first verify the complete source
// tokenizer ID-to-token mapping against GGUF; llama.cpp's GigaChat text-to-ID
// implementation is diagnosed but not used.  The score JSONL contains only
// the five fields accepted by strat02_score.DocumentScore.
//
// Build through the pinned llama.cpp source with the companion helper:
//   python benchmarks/donor_adaptation/density/test_strat01_gguf_score.py --build
// Run the no-model deterministic checks:
//   strat01_gguf_score.exe --self-test

#include "llama.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr int kDefaultContext = 4096;
constexpr int kDefaultBatch = 128;
constexpr size_t kMaxDocuments = 256;
constexpr size_t kMaxDocumentBytes = 1U << 20;
constexpr size_t kMaxInputLineBytes = kMaxDocumentBytes * 3;

struct Options {
    bool self_test = false;
    bool source_token_ids = false;
    std::string model;
    std::string input;
    std::string output;
    std::string metadata_output;
    std::string context_marker = "bos";
    int n_ctx = kDefaultContext;
    int n_batch = kDefaultBatch;
    // -1 means preserve llama.cpp's default established by
    // llama_context_default_params().
    int n_threads = -1;
};

struct InputDocument {
    std::string source_document_id;
    std::string category;
    std::string text;
    uint64_t byte_count = 0;
    std::vector<llama_token> python_ids;
};

struct Score {
    std::string source_document_id;
    std::string category;
    uint64_t tokens = 0;
    uint64_t bytes = 0;
    double bits = 0.0;
};

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

void silent_log(enum ggml_log_level, const char *, void *) {}

bool is_decimal(std::string_view value) {
    return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char ch) {
        return ch >= '0' && ch <= '9';
    });
}

uint64_t parse_u64(std::string_view value, const char * name) {
    if (!is_decimal(value)) {
        fail(std::string(name) + " must be an unsigned decimal integer");
    }
    uint64_t parsed = 0;
    for (const unsigned char ch : value) {
        if (parsed > (std::numeric_limits<uint64_t>::max() - (ch - '0')) / 10U) {
            fail(std::string(name) + " overflows uint64");
        }
        parsed = parsed * 10U + (ch - '0');
    }
    return parsed;
}

int parse_positive_int(std::string_view value, const char * name, int maximum) {
    const uint64_t parsed = parse_u64(value, name);
    if (parsed == 0 || parsed > static_cast<uint64_t>(maximum)) {
        fail(std::string(name) + " is outside the allowed range");
    }
    return static_cast<int>(parsed);
}

std::vector<std::string_view> split_exact(std::string_view value, char delimiter, size_t expected) {
    std::vector<std::string_view> fields;
    size_t begin = 0;
    while (true) {
        const size_t end = value.find(delimiter, begin);
        fields.push_back(value.substr(begin, end == std::string_view::npos ? value.size() - begin : end - begin));
        if (end == std::string_view::npos) {
            break;
        }
        begin = end + 1;
    }
    if (fields.size() != expected) {
        fail("private input line has the wrong number of fields");
    }
    return fields;
}

int base64_value(unsigned char ch) {
    if (ch >= 'A' && ch <= 'Z') return ch - 'A';
    if (ch >= 'a' && ch <= 'z') return ch - 'a' + 26;
    if (ch >= '0' && ch <= '9') return ch - '0' + 52;
    if (ch == '+') return 62;
    if (ch == '/') return 63;
    return -1;
}

std::string base64_decode(std::string_view encoded) {
    if (encoded.empty() || encoded.size() % 4 != 0) {
        fail("invalid base64 field");
    }
    std::string decoded;
    decoded.reserve(encoded.size() / 4 * 3);
    for (size_t index = 0; index < encoded.size(); index += 4) {
        const unsigned char a = encoded[index];
        const unsigned char b = encoded[index + 1];
        const unsigned char c = encoded[index + 2];
        const unsigned char d = encoded[index + 3];
        const int va = base64_value(a);
        const int vb = base64_value(b);
        const bool c_pad = c == '=';
        const bool d_pad = d == '=';
        const int vc = c_pad ? 0 : base64_value(c);
        const int vd = d_pad ? 0 : base64_value(d);
        if (va < 0 || vb < 0 || vc < 0 || vd < 0 || (c_pad && !d_pad) || ((c_pad || d_pad) && index + 4 != encoded.size())) {
            fail("invalid base64 field");
        }
        const uint32_t packed = (static_cast<uint32_t>(va) << 18U) |
                                (static_cast<uint32_t>(vb) << 12U) |
                                (static_cast<uint32_t>(vc) << 6U) |
                                static_cast<uint32_t>(vd);
        decoded.push_back(static_cast<char>((packed >> 16U) & 0xffU));
        if (!c_pad) decoded.push_back(static_cast<char>((packed >> 8U) & 0xffU));
        if (!d_pad) decoded.push_back(static_cast<char>(packed & 0xffU));
    }
    return decoded;
}

std::vector<llama_token> parse_ids(std::string_view csv) {
    if (csv.empty()) {
        fail("payload token IDs must not be empty");
    }
    std::vector<llama_token> ids;
    size_t begin = 0;
    while (true) {
        const size_t end = csv.find(',', begin);
        const std::string_view item = csv.substr(begin, end == std::string_view::npos ? csv.size() - begin : end - begin);
        const uint64_t value = parse_u64(item, "token ID");
        if (value > static_cast<uint64_t>(std::numeric_limits<llama_token>::max())) {
            fail("token ID exceeds llama_token range");
        }
        ids.push_back(static_cast<llama_token>(value));
        if (end == std::string_view::npos) {
            break;
        }
        begin = end + 1;
    }
    return ids;
}

std::vector<InputDocument> read_input(const std::string & path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        fail("cannot open private input file");
    }
    std::vector<InputDocument> documents;
    std::string line;
    while (std::getline(stream, line)) {
        if (line.size() > kMaxInputLineBytes) {
            fail("private input line exceeds its bounded size");
        }
        if (line.empty()) {
            fail("private input contains a blank line");
        }
        if (documents.size() == kMaxDocuments) {
            fail("private input exceeds the bounded document count");
        }
        const auto fields = split_exact(line, '\t', 5);
        InputDocument document;
        document.source_document_id = base64_decode(fields[0]);
        document.category = base64_decode(fields[1]);
        document.byte_count = parse_u64(fields[2], "byte count");
        document.text = base64_decode(fields[3]);
        document.python_ids = parse_ids(fields[4]);
        if (document.source_document_id.empty() || document.category.empty()) {
            fail("private input contains an empty document identity field");
        }
        if (document.text.empty() || document.text.size() > kMaxDocumentBytes || document.byte_count != document.text.size()) {
            fail("private input has an invalid raw UTF-8 byte count");
        }
        documents.push_back(std::move(document));
    }
    if (!stream.eof()) {
        fail("cannot read private input file");
    }
    if (documents.empty()) {
        fail("private input has no documents");
    }
    return documents;
}

std::string json_quote(std::string_view value) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (ch < 0x20U) {
                    output << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned int>(ch)
                           << std::dec << std::setfill(' ');
                } else {
                    output << static_cast<char>(ch);
                }
        }
    }
    output << '"';
    return output.str();
}

void ensure_new_path(const std::string & path, const char * label) {
    if (path.empty()) {
        return;
    }
    std::ifstream existing(path, std::ios::binary);
    if (existing.good()) {
        fail(std::string(label) + " already exists; refusing to overwrite it");
    }
}

void write_text_new(const std::string & path, const std::string & content, const char * label) {
    ensure_new_path(path, label);
    std::ofstream stream(path, std::ios::binary | std::ios::out);
    if (!stream) {
        fail(std::string("cannot create ") + label);
    }
    stream.write(content.data(), static_cast<std::streamsize>(content.size()));
    stream.flush();
    if (!stream) {
        fail(std::string("cannot write ") + label);
    }
}

std::vector<llama_token> tokenize_with_llama(const llama_vocab * vocab, const std::string & text) {
    if (text.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        fail("document is too large for llama_tokenize");
    }
    const int32_t needed_negative = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), nullptr, 0, false, false);
    if (needed_negative >= 0 || needed_negative == std::numeric_limits<int32_t>::min()) {
        fail("llama_tokenize did not report a bounded required size");
    }
    const int32_t needed = -needed_negative;
    if (needed <= 0) {
        fail("llama_tokenize produced no payload tokens");
    }
    std::vector<llama_token> ids(static_cast<size_t>(needed));
    const int32_t actual = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), ids.data(), needed, false, false);
    if (actual != needed) {
        fail("llama_tokenize changed its required token count");
    }
    return ids;
}

llama_token select_context_marker(const llama_vocab * vocab, const std::string & policy, int32_t n_vocab) {
    const llama_token bos = llama_vocab_bos(vocab);
    const llama_token eos = llama_vocab_eos(vocab);
    const auto valid = [n_vocab](llama_token token) {
        return token >= 0 && token < n_vocab;
    };
    if (policy == "auto") {
        if (llama_vocab_get_add_bos(vocab) && valid(bos)) return bos;
        if (valid(eos)) return eos;
        if (valid(bos)) return bos;
        fail("model metadata exposes neither a usable BOS nor EOS marker");
    }
    if (policy == "bos") {
        if (!valid(bos)) fail("model metadata exposes no usable BOS marker");
        return bos;
    }
    if (policy == "eos") {
        if (!valid(eos)) fail("model metadata exposes no usable EOS marker");
        return eos;
    }
    const uint64_t explicit_id = parse_u64(policy, "context marker");
    if (explicit_id >= static_cast<uint64_t>(n_vocab)) {
        fail("explicit context marker is outside model vocabulary");
    }
    return static_cast<llama_token>(explicit_id);
}

double stable_logsumexp(const float * logits, int32_t n_vocab, llama_token target) {
    if (logits == nullptr || target < 0 || target >= n_vocab) {
        fail("invalid logits row or target token");
    }
    float maximum = -std::numeric_limits<float>::infinity();
    for (int32_t token = 0; token < n_vocab; ++token) {
        const float value = logits[token];
        if (!std::isfinite(value)) {
            fail("non-finite model logit");
        }
        maximum = std::max(maximum, value);
    }
    long double sum = 0.0L;
    for (int32_t token = 0; token < n_vocab; ++token) {
        sum += std::exp(static_cast<long double>(logits[token] - maximum));
    }
    if (!(sum > 0.0L) || !std::isfinite(static_cast<double>(sum))) {
        fail("non-finite logsumexp accumulator");
    }
    const long double nll = static_cast<long double>(maximum) + std::log(sum) - static_cast<long double>(logits[target]);
    if (!(nll >= 0.0L) || !std::isfinite(static_cast<double>(nll))) {
        fail("non-finite negative log likelihood");
    }
    return static_cast<double>(nll);
}

struct ScoringPosition {
    llama_token input;
    llama_token target;
};

ScoringPosition scoring_position(const std::vector<llama_token> & ids, size_t index, llama_token context_marker) {
    if (index >= ids.size()) fail("scoring position exceeds the payload");
    return {index == 0 ? context_marker : ids[index - 1], ids[index]};
}

Score score_document(
        llama_context * context,
        const InputDocument & document,
        const llama_vocab * vocab,
        int32_t n_vocab,
        llama_token context_marker,
        int n_ctx,
        int n_batch,
        bool source_token_ids,
        uint64_t & runtime_tokenizer_mismatch_documents) {
    const std::vector<llama_token> llama_ids = tokenize_with_llama(vocab, document.text);
    if (llama_ids != document.python_ids) {
        size_t first = 0;
        while (first < llama_ids.size() && first < document.python_ids.size() && llama_ids[first] == document.python_ids[first]) {
            ++first;
        }
        std::ostringstream reason;
        reason << "Python and llama.cpp token IDs differ: python_count=" << document.python_ids.size()
               << " llama_count=" << llama_ids.size() << " first_mismatch=" << first;
        if (first < llama_ids.size() && first < document.python_ids.size()) {
            reason << " python_id=" << document.python_ids[first] << " llama_id=" << llama_ids[first];
        }
        if (!source_token_ids) {
            fail(reason.str());
        }
        ++runtime_tokenizer_mismatch_documents;
    }
    const std::vector<llama_token> & ids = document.python_ids;
    if (ids.size() + 1U > static_cast<size_t>(n_ctx)) {
        fail("document exceeds the no-truncation context bound");
    }
    for (const llama_token token : ids) {
        if (token < 0 || token >= n_vocab) {
            fail("payload token is outside the GGUF vocabulary");
        }
    }

    // This clears all sequence metadata before every document.  The old KV
    // buffer bytes may remain allocated, but no position is addressable by the
    // following decode, so no cross-document attention state is available.
    llama_memory_clear(llama_get_memory(context), false);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    long double nll = 0.0L;
    try {
        for (size_t start = 0; start < ids.size(); start += static_cast<size_t>(n_batch)) {
            const size_t count = std::min(static_cast<size_t>(n_batch), ids.size() - start);
            batch.n_tokens = static_cast<int32_t>(count);
            for (size_t offset = 0; offset < count; ++offset) {
                const size_t index = start + offset;
                batch.token[offset] = scoring_position(ids, index, context_marker).input;
                batch.pos[offset] = static_cast<llama_pos>(index);
                batch.n_seq_id[offset] = 1;
                batch.seq_id[offset][0] = 0;
                batch.logits[offset] = 1;
            }
            const int32_t status = llama_decode(context, batch);
            if (status != 0) {
                fail("llama_decode failed while scoring a document");
            }
            for (size_t offset = 0; offset < count; ++offset) {
                const llama_token target = scoring_position(ids, start + offset, context_marker).target;
                nll += stable_logsumexp(llama_get_logits_ith(context, static_cast<int32_t>(offset)), n_vocab, target);
            }
        }
    } catch (...) {
        llama_batch_free(batch);
        llama_memory_clear(llama_get_memory(context), false);
        throw;
    }
    llama_batch_free(batch);
    llama_memory_clear(llama_get_memory(context), false);
    const double bits = static_cast<double>(nll / std::log(2.0L));
    if (!(bits >= 0.0) || !std::isfinite(bits)) {
        fail("non-finite total bits");
    }
    return Score{document.source_document_id, document.category, static_cast<uint64_t>(ids.size()), document.byte_count, bits};
}

std::string scores_jsonl(const std::vector<Score> & scores) {
    std::ostringstream output;
    output << std::setprecision(17);
    for (const Score & score : scores) {
        output << "{\"source_document_id\":" << json_quote(score.source_document_id)
               << ",\"category\":" << json_quote(score.category)
               << ",\"tokens\":" << score.tokens
               << ",\"bytes\":" << score.bytes
               << ",\"bits\":" << score.bits << "}\n";
    }
    return output.str();
}

std::string metadata_json(
        const Options & options,
        const std::vector<Score> & scores,
        const llama_vocab * vocab,
        int32_t n_vocab,
        llama_token marker,
        uint64_t runtime_tokenizer_mismatch_documents) {
    uint64_t bytes = 0;
    uint64_t tokens = 0;
    for (const Score & score : scores) {
        bytes += score.bytes;
        tokens += score.tokens;
    }
    std::ostringstream output;
    output << "{\"schema\":\"strat01_gguf_score_runtime_v1\""
           << ",\"documents\":" << scores.size()
           << ",\"payload_tokens\":" << tokens
           << ",\"payload_bytes\":" << bytes
           << ",\"n_ctx\":" << options.n_ctx
           << ",\"n_batch\":" << options.n_batch
           << ",\"n_threads\":" << options.n_threads
           << ",\"context_marker_policy\":" << json_quote(options.context_marker)
           << ",\"context_marker_id\":" << marker
           << ",\"model_bos_id\":" << llama_vocab_bos(vocab)
           << ",\"model_eos_id\":" << llama_vocab_eos(vocab)
           << ",\"model_add_bos\":" << (llama_vocab_get_add_bos(vocab) ? "true" : "false")
           << ",\"model_add_eos\":" << (llama_vocab_get_add_eos(vocab) ? "true" : "false")
           << ",\"model_vocab_size\":" << n_vocab
           << ",\"source_token_ids_mode\":" << (options.source_token_ids ? "true" : "false")
           << ",\"runtime_tokenizer_mismatch_documents\":" << runtime_tokenizer_mismatch_documents
           << ",\"runtime_tokenizer_id_parity_all_documents\":" << (runtime_tokenizer_mismatch_documents == 0 ? "true" : "false")
           << ",\"kv_reset_per_document\":true"
           << ",\"payload_eos_appended\":false}"
           << "\n";
    return output.str();
}

Options parse_options(int argc, char ** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--self-test") {
            options.self_test = true;
            continue;
        }
        if (argument == "--source-token-ids") {
            options.source_token_ids = true;
            continue;
        }
        if (index + 1 >= argc) {
            fail("missing value for command-line option");
        }
        const std::string value = argv[++index];
        if (argument == "--model") options.model = value;
        else if (argument == "--input") options.input = value;
        else if (argument == "--output") options.output = value;
        else if (argument == "--metadata-out") options.metadata_output = value;
        else if (argument == "--context-marker") options.context_marker = value;
        else if (argument == "--n-ctx") options.n_ctx = parse_positive_int(value, "n-ctx", kDefaultContext);
        else if (argument == "--n-batch") options.n_batch = parse_positive_int(value, "n-batch", kDefaultContext);
        else if (argument == "--threads") options.n_threads = parse_positive_int(value, "threads", 1024);
        else fail("unknown command-line option");
    }
    if (options.self_test) {
        if (argc != 2) fail("--self-test accepts no other options");
        return options;
    }
    if (options.model.empty() || options.input.empty() || options.output.empty() || options.metadata_output.empty()) {
        fail("--model, --input, --output, and --metadata-out are required");
    }
    if (options.n_batch > options.n_ctx) {
        fail("n-batch must not exceed n-ctx");
    }
    return options;
}

void self_test() {
    const std::array<float, 2> equal = {0.0F, 0.0F};
    const std::array<float, 3> large = {1000.0F, 999.0F, 998.0F};
    const double first = stable_logsumexp(equal.data(), static_cast<int32_t>(equal.size()), 0);
    const double second = stable_logsumexp(large.data(), static_cast<int32_t>(large.size()), 0);
    if (std::abs(first - std::log(2.0)) > 1e-14 || !std::isfinite(second) || second <= 0.0) {
        fail("deterministic logsumexp self-test failed");
    }
    const std::vector<llama_token> payload = {1, 2, 0};
    const std::array<llama_token, 3> expected_inputs = {3, 1, 2};
    const std::array<std::array<float, 3>, 3> synthetic_logits = {{{0.0F, 2.0F, 0.0F},
                                                                   {0.0F, 0.0F, 3.0F},
                                                                   {4.0F, 0.0F, 0.0F}}};
    double measured_nll = 0.0;
    double expected_nll = 0.0;
    for (size_t index = 0; index < payload.size(); ++index) {
        const ScoringPosition position = scoring_position(payload, index, 3);
        if (position.input != expected_inputs[index] || position.target != payload[index]) {
            fail("synthetic next-token alignment self-test failed");
        }
        measured_nll += stable_logsumexp(synthetic_logits[index].data(), 3, position.target);
        const double numerator = std::exp(static_cast<double>(synthetic_logits[index][payload[index]]));
        const double denominator = std::exp(static_cast<double>(synthetic_logits[index][0]))
                                 + std::exp(static_cast<double>(synthetic_logits[index][1]))
                                 + std::exp(static_cast<double>(synthetic_logits[index][2]));
        expected_nll -= std::log(numerator / denominator);
    }
    if (std::abs(measured_nll - expected_nll) > 1e-12) {
        fail("synthetic known-probability self-test failed");
    }
    if (base64_decode("YQ==") != "a" || parse_ids("0,17,23") != std::vector<llama_token>({0, 17, 23})) {
        fail("deterministic exchange-format self-test failed");
    }
}

int run(const Options & options) {
    if (options.self_test) {
        self_test();
        return 0;
    }
    ensure_new_path(options.output, "score output");
    ensure_new_path(options.metadata_output, "metadata output");
    const std::vector<InputDocument> documents = read_input(options.input);
    llama_log_set(silent_log, nullptr);
    llama_backend_init();
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    try {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model_params.load_mtp = false;
        model = llama_model_load_from_file(options.model.c_str(), model_params);
        if (model == nullptr) {
            fail("cannot load GGUF model");
        }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        if (vocab == nullptr || n_vocab <= 0) {
            fail("GGUF exposes no usable vocabulary");
        }
        const llama_token marker = select_context_marker(vocab, options.context_marker, n_vocab);
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<uint32_t>(options.n_ctx);
        context_params.n_batch = static_cast<uint32_t>(options.n_batch);
        context_params.n_ubatch = static_cast<uint32_t>(options.n_batch);
        context_params.n_seq_max = 1;
        if (options.n_threads > 0) {
            context_params.n_threads = options.n_threads;
            context_params.n_threads_batch = options.n_threads;
        }
        context_params.no_perf = true;
        context = llama_init_from_model(model, context_params);
        if (context == nullptr || llama_n_ctx(context) < static_cast<uint32_t>(options.n_ctx)) {
            fail("cannot create the requested bounded llama context");
        }
        std::vector<Score> scores;
        scores.reserve(documents.size());
        uint64_t runtime_tokenizer_mismatch_documents = 0;
        for (const InputDocument & document : documents) {
            scores.push_back(score_document(context, document, vocab, n_vocab, marker, options.n_ctx, options.n_batch,
                                            options.source_token_ids, runtime_tokenizer_mismatch_documents));
        }
        write_text_new(options.output, scores_jsonl(scores), "score output");
        write_text_new(options.metadata_output, metadata_json(options, scores, vocab, n_vocab, marker,
                                                             runtime_tokenizer_mismatch_documents), "metadata output");
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (...) {
        if (context != nullptr) llama_free(context);
        if (model != nullptr) llama_model_free(model);
        llama_backend_free();
        throw;
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception & error) {
        // Deliberately never include document text, token pieces, or logits.
        std::fputs("strat01_gguf_score: ", stderr);
        std::fputs(error.what(), stderr);
        std::fputc('\n', stderr);
        return 2;
    }
}
