// Bounded GGUF PIQA scorer for STRAT-01 GigaChat.
//
// The private exchange contains token IDs only:
//   index<TAB>label<TAB>prefix_ids<TAB>option0_suffix_ids<TAB>option1_suffix_ids
// No benchmark text is emitted by this program.  The main path decodes the
// common [BOS] + prefix once, copies its KV sequence, and scores both suffixes.
// --replay-prefix is the slower apparatus reference: it replays the full
// prefix independently for each option.

// Build through the pinned llama.cpp source with:
//   python test_strat01_gguf_piqa.py --build
// Run deterministic no-model checks with:
//   strat01_gguf_piqa.exe --self-test

#include "llama.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
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
constexpr size_t kExpectedItems = 1838;
constexpr size_t kMaxInputLineBytes = 1U << 18;

struct Options {
    bool self_test = false;
    bool replay_prefix = false;
    std::string model;
    std::string input;
    std::string output;
    std::string metadata_output;
    int n_ctx = kDefaultContext;
    int n_batch = kDefaultBatch;
    int n_threads = -1;
};

struct Item {
    uint64_t index = 0;
    int label = 0;
    std::vector<llama_token> prefix;
    std::array<std::vector<llama_token>, 2> suffixes;
};

struct OptionScore {
    uint64_t tokens = 0;
    double nll = 0.0;
};

struct Result {
    uint64_t index = 0;
    int label = 0;
    uint64_t prefix_tokens = 0;
    std::array<OptionScore, 2> options;
    int choice_mean = 0;
    int choice_total = 0;
    bool correct_mean = false;
    bool correct_total = false;
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
    if (!is_decimal(value)) fail(std::string(name) + " must be an unsigned decimal integer");
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
        if (end == std::string_view::npos) break;
        begin = end + 1;
    }
    if (fields.size() != expected) fail("private PIQA line has the wrong number of fields");
    return fields;
}

std::vector<llama_token> parse_ids(std::string_view csv, const char * label) {
    if (csv.empty()) fail(std::string(label) + " token IDs must not be empty");
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
        if (end == std::string_view::npos) break;
        begin = end + 1;
    }
    return ids;
}

std::vector<Item> read_input(const std::string & path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) fail("cannot open private PIQA input file");
    std::vector<Item> items;
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) fail("private PIQA input contains a blank line");
        if (line.size() > kMaxInputLineBytes) fail("private PIQA line exceeds its bounded size");
        if (items.size() == kExpectedItems) fail("private PIQA input exceeds 1838 items");
        const auto fields = split_exact(line, '\t', 5);
        Item item;
        item.index = parse_u64(fields[0], "item index");
        if (item.index != items.size()) fail("PIQA item indices must be contiguous from zero");
        const uint64_t label = parse_u64(fields[1], "PIQA label");
        if (label > 1) fail("PIQA label must be zero or one");
        item.label = static_cast<int>(label);
        item.prefix = parse_ids(fields[2], "prefix");
        item.suffixes[0] = parse_ids(fields[3], "option 0 suffix");
        item.suffixes[1] = parse_ids(fields[4], "option 1 suffix");
        items.push_back(std::move(item));
    }
    if (!stream.eof()) fail("cannot read private PIQA input file");
    if (items.empty()) fail("private PIQA input has no items");
    return items;
}

void ensure_new_path(const std::string & path, const char * label) {
    if (path.empty()) return;
    std::ifstream existing(path, std::ios::binary);
    if (existing.good()) fail(std::string(label) + " already exists; refusing to overwrite it");
}

void write_text_new(const std::string & path, const std::string & content, const char * label) {
    ensure_new_path(path, label);
    std::ofstream stream(path, std::ios::binary | std::ios::out);
    if (!stream) fail(std::string("cannot create ") + label);
    stream.write(content.data(), static_cast<std::streamsize>(content.size()));
    stream.flush();
    if (!stream) fail(std::string("cannot write ") + label);
}

double stable_nll(const float * logits, int32_t n_vocab, llama_token target) {
    if (logits == nullptr || target < 0 || target >= n_vocab) fail("invalid logits row or target token");
    float maximum = -std::numeric_limits<float>::infinity();
    for (int32_t token = 0; token < n_vocab; ++token) {
        if (!std::isfinite(logits[token])) fail("non-finite model logit");
        maximum = std::max(maximum, logits[token]);
    }
    long double sum = 0.0L;
    for (int32_t token = 0; token < n_vocab; ++token) {
        sum += std::exp(static_cast<long double>(logits[token] - maximum));
    }
    const long double nll = static_cast<long double>(maximum) + std::log(sum) - logits[target];
    if (!(sum > 0.0L) || !(nll >= 0.0L) || !std::isfinite(static_cast<double>(nll))) {
        fail("non-finite negative log likelihood");
    }
    return static_cast<double>(nll);
}

void validate_ids(const Item & item, int32_t n_vocab, int n_ctx) {
    const auto validate = [n_vocab](const std::vector<llama_token> & ids) {
        for (const llama_token token : ids) {
            if (token < 0 || token >= n_vocab) fail("PIQA token is outside the GGUF vocabulary");
        }
    };
    validate(item.prefix);
    for (const auto & suffix : item.suffixes) {
        validate(suffix);
        if (1U + item.prefix.size() + suffix.size() > static_cast<size_t>(n_ctx)) {
            fail("PIQA option exceeds the no-truncation context bound");
        }
    }
}

// Decode one token vector into a single sequence.  Only the final row may be
// requested as logits, which is sufficient for the first suffix token.
const float * decode_prefix(
        llama_context * context,
        llama_batch & batch,
        const std::vector<llama_token> & prefix_inputs,
        llama_seq_id sequence,
        int n_batch) {
    const float * final_logits = nullptr;
    for (size_t start = 0; start < prefix_inputs.size(); start += static_cast<size_t>(n_batch)) {
        const size_t count = std::min(static_cast<size_t>(n_batch), prefix_inputs.size() - start);
        batch.n_tokens = static_cast<int32_t>(count);
        for (size_t offset = 0; offset < count; ++offset) {
            const size_t index = start + offset;
            batch.token[offset] = prefix_inputs[index];
            batch.pos[offset] = static_cast<llama_pos>(index);
            batch.n_seq_id[offset] = 1;
            batch.seq_id[offset][0] = sequence;
            batch.logits[offset] = index + 1 == prefix_inputs.size() ? 1 : 0;
        }
        if (llama_decode(context, batch) != 0) fail("llama_decode failed on PIQA prefix");
        if (start + count == prefix_inputs.size()) {
            final_logits = llama_get_logits_ith(context, static_cast<int32_t>(count - 1));
            if (final_logits == nullptr) fail("PIQA prefix produced no final logits");
        }
    }
    return final_logits;
}

double score_suffix_tail(
        llama_context * context,
        llama_batch & batch,
        const std::vector<llama_token> & suffix,
        size_t prefix_tokens,
        llama_seq_id sequence,
        int32_t n_vocab,
        int n_batch) {
    long double nll = 0.0L;
    // suffix[j-1] is the input whose logits predict suffix[j].  The prefix
    // final row, handled by the caller, predicts suffix[0].
    for (size_t first_target = 1; first_target < suffix.size();) {
        const size_t count = std::min(static_cast<size_t>(n_batch), suffix.size() - first_target);
        batch.n_tokens = static_cast<int32_t>(count);
        for (size_t offset = 0; offset < count; ++offset) {
            const size_t target_index = first_target + offset;
            batch.token[offset] = suffix[target_index - 1];
            batch.pos[offset] = static_cast<llama_pos>(prefix_tokens + target_index);
            batch.n_seq_id[offset] = 1;
            batch.seq_id[offset][0] = sequence;
            batch.logits[offset] = 1;
        }
        if (llama_decode(context, batch) != 0) fail("llama_decode failed on PIQA suffix");
        for (size_t offset = 0; offset < count; ++offset) {
            nll += stable_nll(llama_get_logits_ith(context, static_cast<int32_t>(offset)), n_vocab,
                              suffix[first_target + offset]);
        }
        first_target += count;
    }
    return static_cast<double>(nll);
}

std::vector<llama_token> make_prefix_inputs(llama_token bos, const std::vector<llama_token> & prefix) {
    std::vector<llama_token> inputs;
    inputs.reserve(prefix.size() + 1U);
    inputs.push_back(bos);
    inputs.insert(inputs.end(), prefix.begin(), prefix.end());
    return inputs;
}

OptionScore score_option_replay(
        llama_context * context,
        llama_batch & batch,
        const Item & item,
        const std::vector<llama_token> & prefix_inputs,
        int option,
        int32_t n_vocab,
        int n_batch) {
    llama_memory_clear(llama_get_memory(context), false);
    const float * logits = decode_prefix(context, batch, prefix_inputs, 0, n_batch);
    const auto & suffix = item.suffixes[option];
    long double nll = stable_nll(logits, n_vocab, suffix[0]);
    nll += score_suffix_tail(context, batch, suffix, item.prefix.size(), 0, n_vocab, n_batch);
    llama_memory_clear(llama_get_memory(context), false);
    return {static_cast<uint64_t>(suffix.size()), static_cast<double>(nll)};
}

Result score_item(
        llama_context * context,
        llama_batch & batch,
        const Item & item,
        llama_token bos,
        int32_t n_vocab,
        int n_batch,
        bool replay_prefix) {
    const std::vector<llama_token> prefix_inputs = make_prefix_inputs(bos, item.prefix);
    std::array<OptionScore, 2> scores;
    if (replay_prefix) {
        scores[0] = score_option_replay(context, batch, item, prefix_inputs, 0, n_vocab, n_batch);
        scores[1] = score_option_replay(context, batch, item, prefix_inputs, 1, n_vocab, n_batch);
    } else {
        llama_memory_clear(llama_get_memory(context), false);
        const float * prefix_logits = decode_prefix(context, batch, prefix_inputs, 0, n_batch);
        scores[0] = {static_cast<uint64_t>(item.suffixes[0].size()),
                     stable_nll(prefix_logits, n_vocab, item.suffixes[0][0])};
        scores[1] = {static_cast<uint64_t>(item.suffixes[1].size()),
                     stable_nll(prefix_logits, n_vocab, item.suffixes[1][0])};
        llama_memory_seq_cp(llama_get_memory(context), 0, 1, -1, -1);
        scores[0].nll += score_suffix_tail(context, batch, item.suffixes[0], item.prefix.size(), 0, n_vocab, n_batch);
        scores[1].nll += score_suffix_tail(context, batch, item.suffixes[1], item.prefix.size(), 1, n_vocab, n_batch);
        llama_memory_clear(llama_get_memory(context), false);
    }
    const double mean0 = scores[0].nll / static_cast<double>(scores[0].tokens);
    const double mean1 = scores[1].nll / static_cast<double>(scores[1].tokens);
    if (!std::isfinite(mean0) || !std::isfinite(mean1)) fail("non-finite PIQA mean NLL");
    const int choice_mean = mean0 <= mean1 ? 0 : 1;
    const int choice_total = scores[0].nll <= scores[1].nll ? 0 : 1;
    return {item.index, item.label, static_cast<uint64_t>(item.prefix.size()), scores,
            choice_mean, choice_total, choice_mean == item.label, choice_total == item.label};
}

std::string results_jsonl(const std::vector<Result> & results) {
    std::ostringstream output;
    output << std::setprecision(17);
    for (const Result & result : results) {
        output << "{\"index\":" << result.index
               << ",\"label\":" << result.label
               << ",\"prefix_tokens\":" << result.prefix_tokens
               << ",\"option0_tokens\":" << result.options[0].tokens
               << ",\"option0_nll\":" << result.options[0].nll
               << ",\"option0_mean_nll\":" << result.options[0].nll / result.options[0].tokens
               << ",\"option1_tokens\":" << result.options[1].tokens
               << ",\"option1_nll\":" << result.options[1].nll
               << ",\"option1_mean_nll\":" << result.options[1].nll / result.options[1].tokens
               << ",\"choice_mean_nll\":" << result.choice_mean
               << ",\"choice_total_nll\":" << result.choice_total
               << ",\"correct_mean_nll\":" << (result.correct_mean ? "true" : "false")
               << ",\"correct_total_nll\":" << (result.correct_total ? "true" : "false") << "}\n";
    }
    return output.str();
}

std::string metadata_json(
        const Options & options,
        const std::vector<Result> & results,
        const llama_vocab * vocab,
        int32_t n_vocab,
        uint32_t physical_n_ctx) {
    uint64_t correct_mean = 0;
    uint64_t correct_total = 0;
    uint64_t prefix_tokens = 0;
    uint64_t suffix_tokens = 0;
    for (const auto & result : results) {
        correct_mean += result.correct_mean ? 1U : 0U;
        correct_total += result.correct_total ? 1U : 0U;
        prefix_tokens += result.prefix_tokens;
        suffix_tokens += result.options[0].tokens + result.options[1].tokens;
    }
    std::ostringstream output;
    output << "{\"schema\":\"strat01_gguf_piqa_runtime_v1\""
           << ",\"items\":" << results.size()
           << ",\"correct_mean_nll\":" << correct_mean
           << ",\"correct_total_nll\":" << correct_total
           << ",\"prefix_tokens\":" << prefix_tokens
           << ",\"suffix_tokens\":" << suffix_tokens
           << ",\"logical_n_ctx\":" << options.n_ctx
           << ",\"physical_n_ctx\":" << physical_n_ctx
           << ",\"n_batch\":" << options.n_batch
           << ",\"n_threads\":" << options.n_threads
           << ",\"model_bos_id\":" << llama_vocab_bos(vocab)
           << ",\"model_eos_id\":" << llama_vocab_eos(vocab)
           << ",\"model_vocab_size\":" << n_vocab
           << ",\"source_token_ids_mode\":true"
           << ",\"prefix_cache_shared\":" << (options.replay_prefix ? "false" : "true")
           << ",\"replay_prefix_reference\":" << (options.replay_prefix ? "true" : "false")
           << ",\"kv_reset_per_item\":true"
           << ",\"choice_metric\":\"mean_suffix_nll\""
           << ",\"tie_policy\":\"option_0\"}\n";
    return output.str();
}

Options parse_options(int argc, char ** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--self-test") { options.self_test = true; continue; }
        if (argument == "--replay-prefix") { options.replay_prefix = true; continue; }
        if (index + 1 >= argc) fail("missing value for command-line option");
        const std::string value = argv[++index];
        if (argument == "--model") options.model = value;
        else if (argument == "--input") options.input = value;
        else if (argument == "--output") options.output = value;
        else if (argument == "--metadata-out") options.metadata_output = value;
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
    if (options.n_batch > options.n_ctx) fail("n-batch must not exceed n-ctx");
    return options;
}

void self_test() {
    const std::array<float, 3> logits = {2.0F, 0.0F, -1.0F};
    const double observed = stable_nll(logits.data(), 3, 0);
    const double expected = -std::log(std::exp(2.0) / (std::exp(2.0) + 1.0 + std::exp(-1.0)));
    if (std::abs(observed - expected) > 1e-12) fail("known-probability NLL self-test failed");
    const std::vector<llama_token> prefix = {10, 11};
    const std::vector<llama_token> suffix = {20, 21, 22};
    if (make_prefix_inputs(1, prefix) != std::vector<llama_token>({1, 10, 11})) {
        fail("PIQA BOS/prefix alignment self-test failed");
    }
    const size_t prefix_tokens = prefix.size();
    const std::array<llama_token, 2> expected_inputs = {20, 21};
    const std::array<llama_token, 2> expected_targets = {21, 22};
    const std::array<size_t, 2> expected_positions = {3, 4};
    for (size_t target_index = 1; target_index < suffix.size(); ++target_index) {
        const size_t offset = target_index - 1;
        if (suffix[target_index - 1] != expected_inputs[offset] || suffix[target_index] != expected_targets[offset]
                || prefix_tokens + target_index != expected_positions[offset]) {
            fail("PIQA suffix next-token alignment self-test failed");
        }
    }
    if (parse_ids("0,17,23", "test") != std::vector<llama_token>({0, 17, 23})) {
        fail("PIQA exchange-format self-test failed");
    }
}

int run(const Options & options) {
    if (options.self_test) { self_test(); return 0; }
    ensure_new_path(options.output, "PIQA score output");
    ensure_new_path(options.metadata_output, "PIQA metadata output");
    const std::vector<Item> items = read_input(options.input);
    llama_log_set(silent_log, nullptr);
    llama_backend_init();
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    try {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model_params.load_mtp = false;
        model = llama_model_load_from_file(options.model.c_str(), model_params);
        if (model == nullptr) fail("cannot load GGUF model");
        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (vocab == nullptr) fail("GGUF exposes no vocabulary");
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        const llama_token bos = llama_vocab_bos(vocab);
        if (n_vocab <= 0 || bos != 1) fail("GGUF does not expose pinned GigaChat BOS=1");
        for (const Item & item : items) validate_ids(item, n_vocab, options.n_ctx);

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<uint32_t>(options.n_ctx * 2);
        context_params.n_batch = static_cast<uint32_t>(options.n_batch);
        context_params.n_ubatch = static_cast<uint32_t>(options.n_batch);
        context_params.n_seq_max = 2;
        if (options.n_threads > 0) {
            context_params.n_threads = options.n_threads;
            context_params.n_threads_batch = options.n_threads;
        }
        context_params.no_perf = true;
        context = llama_init_from_model(model, context_params);
        if (context == nullptr || llama_n_ctx(context) < static_cast<uint32_t>(options.n_ctx * 2)) {
            fail("cannot create the requested two-sequence llama context");
        }
        llama_batch batch = llama_batch_init(options.n_batch, 0, 2);
        std::vector<Result> results;
        results.reserve(items.size());
        try {
            for (const Item & item : items) {
                results.push_back(score_item(context, batch, item, bos, n_vocab, options.n_batch, options.replay_prefix));
            }
        } catch (...) {
            llama_batch_free(batch);
            throw;
        }
        llama_batch_free(batch);
        write_text_new(options.output, results_jsonl(results), "PIQA score output");
        write_text_new(options.metadata_output, metadata_json(options, results, vocab, n_vocab, llama_n_ctx(context)),
                       "PIQA metadata output");
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

}  // namespace

int main(int argc, char ** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception & error) {
        std::cerr << "strat01_gguf_piqa: " << error.what() << '\n';
        return 2;
    }
}
