// Mechanical GGUF document-rollout runner for STRAT-01 GigaChat 3.1.
//
// This program deliberately accepts source-tokenizer IDs only.  The Python
// wrapper owns source selection and source-tokenizer text decoding; this core
// binds the GGUF execution to [BOS] + 256 supplied payload IDs and emits IDs
// only.  It writes both artifacts only after every document has completed, so
// a numerical fault cannot become a partial scientific record.
//
// Build against the same pinned llama.cpp source as strat01_gguf_piqa.cpp.
// Run no-model apparatus checks with:
//   strat01_gguf_rollout.exe --self-test

#include "llama.h"

#include <algorithm>
#include <array>
#include <chrono>
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
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr int kDefaultContext = 4096;
constexpr int kDefaultBatch = 128;
constexpr int kPinnedBos = 1;
constexpr int kPinnedEos = 2;
constexpr size_t kExpectedDocuments = 96;
constexpr size_t kPromptTokens = 257;  // BOS plus the first 256 payload IDs.
constexpr size_t kMaxNewTokens = 256;
constexpr size_t kMaxInputLineBytes = 1U << 20;

struct Options {
    bool self_test = false;
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
    std::string item_id;
    std::vector<llama_token> prompt_ids;
};

struct Result {
    uint64_t index = 0;
    std::string item_id;
    std::vector<llama_token> prompt_ids;
    std::vector<llama_token> generated_ids;
    std::string stop_reason;
    double elapsed_seconds = 0.0;
    bool run32 = false;
    bool loop8x3 = false;
    bool empty = false;
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

// Small strict JSON reader for the private exchange.  It accepts arbitrary
// additional JSON members, but requires exactly one usable index, item_id,
// and prompt_ids member per line.  Keeping the parser here avoids a new JSON
// dependency in the pinned C++17 apparatus.
class JsonCursor {
public:
    explicit JsonCursor(const std::string & text) : text_(text) {}

    void skip_ws() {
        while (pos_ < text_.size()) {
            const unsigned char ch = static_cast<unsigned char>(text_[pos_]);
            if (ch != ' ' && ch != '\n' && ch != '\r' && ch != '\t') break;
            ++pos_;
        }
    }

    bool consume(char expected) {
        skip_ws();
        if (pos_ < text_.size() && text_[pos_] == expected) {
            ++pos_;
            return true;
        }
        return false;
    }

    void expect(char expected, const char * what) {
        if (!consume(expected)) fail(std::string("malformed rollout JSON: expected ") + what);
    }

    std::string parse_string() {
        skip_ws();
        if (pos_ >= text_.size() || text_[pos_] != '"') fail("malformed rollout JSON string");
        ++pos_;
        std::string result;
        while (pos_ < text_.size()) {
            const unsigned char ch = static_cast<unsigned char>(text_[pos_++]);
            if (ch == '"') return result;
            if (ch < 0x20U) fail("malformed rollout JSON control character in string");
            if (ch != '\\') {
                result.push_back(static_cast<char>(ch));
                continue;
            }
            if (pos_ >= text_.size()) fail("malformed rollout JSON escape");
            const char escaped = text_[pos_++];
            switch (escaped) {
            case '"': result.push_back('"'); break;
            case '\\': result.push_back('\\'); break;
            case '/': result.push_back('/'); break;
            case 'b': result.push_back('\b'); break;
            case 'f': result.push_back('\f'); break;
            case 'n': result.push_back('\n'); break;
            case 'r': result.push_back('\r'); break;
            case 't': result.push_back('\t'); break;
            case 'u': append_codepoint(result, parse_hex4()); break;
            default: fail("malformed rollout JSON escape");
            }
        }
        fail("unterminated rollout JSON string");
    }

    uint64_t parse_u64_number(const char * label) {
        skip_ws();
        const size_t begin = pos_;
        if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') {
            fail(std::string(label) + " must be a non-negative JSON integer");
        }
        if (text_[pos_] == '0') {
            ++pos_;
            if (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
                fail(std::string(label) + " has a non-canonical leading zero");
            }
        } else {
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        }
        if (pos_ < text_.size() && (text_[pos_] == '.' || text_[pos_] == 'e' || text_[pos_] == 'E')) {
            fail(std::string(label) + " must be an integer");
        }
        return parse_u64(std::string_view(text_).substr(begin, pos_ - begin), label);
    }

    void skip_value() {
        skip_ws();
        if (pos_ >= text_.size()) fail("malformed rollout JSON value");
        const char lead = text_[pos_];
        if (lead == '"') { static_cast<void>(parse_string()); return; }
        if (lead == '{') {
            ++pos_;
            skip_ws();
            if (consume('}')) return;
            while (true) {
                static_cast<void>(parse_string());
                expect(':', "':'");
                skip_value();
                if (consume('}')) return;
                expect(',', "','");
            }
        }
        if (lead == '[') {
            ++pos_;
            if (consume(']')) return;
            while (true) {
                skip_value();
                if (consume(']')) return;
                expect(',', "','");
            }
        }
        if (text_.compare(pos_, 4, "true") == 0) { pos_ += 4; return; }
        if (text_.compare(pos_, 5, "false") == 0) { pos_ += 5; return; }
        if (text_.compare(pos_, 4, "null") == 0) { pos_ += 4; return; }
        skip_number_literal();
    }

    void expect_end() {
        skip_ws();
        if (pos_ != text_.size()) fail("trailing data in rollout JSON line");
    }

private:
    uint32_t parse_hex4() {
        if (pos_ + 4U > text_.size()) fail("truncated rollout JSON unicode escape");
        uint32_t value = 0;
        for (int count = 0; count < 4; ++count) {
            const unsigned char ch = static_cast<unsigned char>(text_[pos_++]);
            value <<= 4U;
            if (ch >= '0' && ch <= '9') value |= ch - '0';
            else if (ch >= 'a' && ch <= 'f') value |= ch - 'a' + 10U;
            else if (ch >= 'A' && ch <= 'F') value |= ch - 'A' + 10U;
            else fail("invalid rollout JSON unicode escape");
        }
        return value;
    }

    void append_codepoint(std::string & result, uint32_t codepoint) {
        if (codepoint >= 0xd800U && codepoint <= 0xdbffU) {
            if (pos_ + 2U > text_.size() || text_[pos_] != '\\' || text_[pos_ + 1U] != 'u') {
                fail("unpaired high surrogate in rollout JSON string");
            }
            pos_ += 2U;
            const uint32_t low = parse_hex4();
            if (low < 0xdc00U || low > 0xdfffU) fail("invalid low surrogate in rollout JSON string");
            codepoint = 0x10000U + ((codepoint - 0xd800U) << 10U) + (low - 0xdc00U);
        } else if (codepoint >= 0xdc00U && codepoint <= 0xdfffU) {
            fail("unpaired low surrogate in rollout JSON string");
        }
        if (codepoint <= 0x7fU) {
            result.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ffU) {
            result.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
            result.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else if (codepoint <= 0xffffU) {
            result.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
            result.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
            result.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else {
            result.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
            result.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
            result.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
            result.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        }
    }

    void skip_number_literal() {
        skip_ws();
        const size_t begin = pos_;
        if (pos_ < text_.size() && text_[pos_] == '-') ++pos_;
        if (pos_ >= text_.size()) fail("malformed rollout JSON number");
        if (text_[pos_] == '0') ++pos_;
        else {
            if (text_[pos_] < '1' || text_[pos_] > '9') fail("malformed rollout JSON number");
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        }
        if (pos_ < text_.size() && text_[pos_] == '.') {
            ++pos_;
            const size_t fraction = pos_;
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
            if (pos_ == fraction) fail("malformed rollout JSON number fraction");
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
            const size_t exponent = pos_;
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
            if (pos_ == exponent) fail("malformed rollout JSON number exponent");
        }
        if (begin == pos_) fail("malformed rollout JSON number");
    }

    const std::string & text_;
    size_t pos_ = 0;
};
std::string json_escape(const std::string & value) {
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
                output << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                       << static_cast<unsigned int>(ch) << std::dec << std::setfill(' ');
            } else {
                output << static_cast<char>(ch);
            }
        }
    }
    output << '"';
    return output.str();
}

std::vector<llama_token> parse_prompt_ids(JsonCursor & cursor) {
    cursor.expect('[', "'['");
    std::vector<llama_token> ids;
    if (cursor.consume(']')) return ids;
    while (true) {
        const uint64_t value = cursor.parse_u64_number("prompt token ID");
        if (value > static_cast<uint64_t>(std::numeric_limits<llama_token>::max())) {
            fail("prompt token ID exceeds llama_token range");
        }
        ids.push_back(static_cast<llama_token>(value));
        if (cursor.consume(']')) return ids;
        cursor.expect(',', "','");
    }
}

Item parse_item_line(const std::string & line) {
    JsonCursor cursor(line);
    cursor.expect('{', "'{'");
    Item item;
    bool have_index = false;
    bool have_item_id = false;
    bool have_prompt_ids = false;
    if (!cursor.consume('}')) {
        while (true) {
            const std::string key = cursor.parse_string();
            cursor.expect(':', "':'");
            if (key == "index") {
                if (have_index) fail("rollout JSON has duplicate index member");
                item.index = cursor.parse_u64_number("index");
                have_index = true;
            } else if (key == "item_id") {
                if (have_item_id) fail("rollout JSON has duplicate item_id member");
                item.item_id = cursor.parse_string();
                if (item.item_id.empty()) fail("rollout item_id must be non-empty");
                have_item_id = true;
            } else if (key == "prompt_ids") {
                if (have_prompt_ids) fail("rollout JSON has duplicate prompt_ids member");
                item.prompt_ids = parse_prompt_ids(cursor);
                have_prompt_ids = true;
            } else {
                cursor.skip_value();
            }
            if (cursor.consume('}')) break;
            cursor.expect(',', "','");
        }
    }
    cursor.expect_end();
    if (!have_index || !have_item_id || !have_prompt_ids) {
        fail("rollout JSON requires index, item_id, and prompt_ids");
    }
    return item;
}

void validate_item_shape(const Item & item, int n_ctx) {
    if (item.item_id.empty()) fail("rollout item_id must be non-empty");
    if (item.prompt_ids.size() != kPromptTokens) {
        fail("rollout prompt_ids must contain exactly [BOS] plus 256 payload IDs");
    }
    if (item.prompt_ids.front() != kPinnedBos) fail("rollout prompt_ids must begin with pinned BOS=1");
    if (item.prompt_ids.size() + kMaxNewTokens > static_cast<size_t>(n_ctx)) {
        fail("rollout prompt plus fixed generation cap exceeds the no-truncation context bound");
    }
}

void validate_unique_items(const std::vector<Item> & items) {
    std::unordered_set<std::string> seen_item_ids;
    std::unordered_set<uint64_t> seen_indices;
    for (const Item & item : items) {
        if (!seen_item_ids.insert(item.item_id).second) fail("rollout input contains a duplicate item_id");
        if (!seen_indices.insert(item.index).second) fail("rollout input contains a duplicate index");
    }
}

std::vector<Item> read_input(const std::string & path, int n_ctx) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) fail("cannot open private rollout input file");
    std::vector<Item> items;
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) fail("private rollout input contains a blank line");
        if (line.size() > kMaxInputLineBytes) fail("private rollout JSON line exceeds its bounded size");
        if (items.size() == kExpectedDocuments) fail("private rollout input exceeds 96 documents");
        Item item = parse_item_line(line);
        if (item.index != items.size()) fail("rollout indices must be contiguous from zero in frozen source order");
        validate_item_shape(item, n_ctx);
        items.push_back(std::move(item));
    }
    if (!stream.eof()) fail("cannot read private rollout input file");
    if (items.empty()) fail("rollout input must contain at least one frozen heldout document");
    validate_unique_items(items);
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

void validate_ids(const Item & item, int32_t n_vocab) {
    for (const llama_token token : item.prompt_ids) {
        if (token < 0 || token >= n_vocab) fail("rollout token is outside the GGUF vocabulary");
    }
}

llama_token stable_argmax(const float * logits, int32_t n_vocab) {
    if (logits == nullptr || n_vocab <= kPinnedEos) fail("invalid rollout logits or vocabulary");
    float maximum = -std::numeric_limits<float>::infinity();
    llama_token choice = -1;
    for (int32_t token = 0; token < n_vocab; ++token) {
        const float value = logits[token];
        if (!std::isfinite(value)) fail("non-finite generation logit");
        // Strictly-greater keeps the first (therefore lowest-ID) maximum.
        if (choice < 0 || value > maximum) {
            maximum = value;
            choice = static_cast<llama_token>(token);
        }
    }
    if (choice < 0 || choice >= n_vocab) fail("stable argmax selected an invalid token ID");
    return choice;
}

std::array<bool, 3> rollout_flags(const std::vector<llama_token> & ids) {
    if (std::find(ids.begin(), ids.end(), static_cast<llama_token>(kPinnedEos)) != ids.end()) {
        fail("rollout flags require EOS to be absent from generated IDs");
    }
    bool run32 = false;
    for (size_t start = 0; start + 32U <= ids.size(); ++start) {
        bool all_same = true;
        for (size_t offset = 1; offset < 32U; ++offset) {
            if (ids[start + offset] != ids[start]) { all_same = false; break; }
        }
        if (all_same) { run32 = true; break; }
    }
    bool loop8x3 = false;
    for (size_t start = 0; start + 24U <= ids.size(); ++start) {
        if (std::equal(ids.begin() + static_cast<std::ptrdiff_t>(start),
                       ids.begin() + static_cast<std::ptrdiff_t>(start + 8U),
                       ids.begin() + static_cast<std::ptrdiff_t>(start + 8U))
                && std::equal(ids.begin() + static_cast<std::ptrdiff_t>(start),
                              ids.begin() + static_cast<std::ptrdiff_t>(start + 8U),
                              ids.begin() + static_cast<std::ptrdiff_t>(start + 16U))) {
            loop8x3 = true;
            break;
        }
    }
    return {run32, loop8x3, ids.empty()};
}

const float * decode_prefill(
        llama_context * context,
        llama_batch & batch,
        const std::vector<llama_token> & prompt,
        int n_batch) {
    const float * final_logits = nullptr;
    for (size_t start = 0; start < prompt.size(); start += static_cast<size_t>(n_batch)) {
        const size_t count = std::min(static_cast<size_t>(n_batch), prompt.size() - start);
        batch.n_tokens = static_cast<int32_t>(count);
        for (size_t offset = 0; offset < count; ++offset) {
            const size_t position = start + offset;
            batch.token[offset] = prompt[position];
            batch.pos[offset] = static_cast<llama_pos>(position);
            batch.n_seq_id[offset] = 1;
            batch.seq_id[offset][0] = 0;
            batch.logits[offset] = position + 1U == prompt.size() ? 1 : 0;
        }
        if (llama_decode(context, batch) != 0) fail("llama_decode failed on rollout prompt");
        if (start + count == prompt.size()) {
            final_logits = llama_get_logits_ith(context, static_cast<int32_t>(count - 1U));
            if (final_logits == nullptr) fail("rollout prompt produced no final logits");
        }
    }
    return final_logits;
}

const float * decode_generated_token(
        llama_context * context,
        llama_batch & batch,
        llama_token token,
        size_t position) {
    batch.n_tokens = 1;
    batch.token[0] = token;
    batch.pos[0] = static_cast<llama_pos>(position);
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;
    if (llama_decode(context, batch) != 0) fail("llama_decode failed on cached rollout token");
    const float * logits = llama_get_logits_ith(context, 0);
    if (logits == nullptr) fail("cached rollout token produced no logits");
    return logits;
}

double checked_elapsed(std::chrono::steady_clock::duration duration, const char * label) {
    const double elapsed = std::chrono::duration<double>(duration).count();
    if (!std::isfinite(elapsed) || elapsed < 0.0) fail(std::string("non-finite ") + label);
    return elapsed;
}

Result rollout_item(
        llama_context * context,
        llama_batch & batch,
        const Item & item,
        int32_t n_vocab,
        int n_batch) {
    const auto started = std::chrono::steady_clock::now();
    // Exactly one fresh KV state is established at the beginning of every
    // document.  The following document resets it before its own prefill.
    llama_memory_clear(llama_get_memory(context), false);
    const float * logits = decode_prefill(context, batch, item.prompt_ids, n_batch);
    std::vector<llama_token> generated;
    generated.reserve(kMaxNewTokens);
    std::string stop_reason = "CAP";
    while (generated.size() < kMaxNewTokens) {
        const llama_token token = stable_argmax(logits, n_vocab);
        if (token == kPinnedEos) {
            stop_reason = "EOS";
            break;
        }
        generated.push_back(token);
        if (generated.size() == kMaxNewTokens) break;
        const size_t position = item.prompt_ids.size() + generated.size() - 1U;
        logits = decode_generated_token(context, batch, token, position);
    }
    const double elapsed = checked_elapsed(std::chrono::steady_clock::now() - started, "document elapsed_seconds");
    const auto flags = rollout_flags(generated);
    return {item.index, item.item_id, item.prompt_ids, std::move(generated), stop_reason, elapsed,
            flags[0], flags[1], flags[2]};
}

void append_ids(std::ostringstream & output, const std::vector<llama_token> & ids) {
    output << '[';
    for (size_t index = 0; index < ids.size(); ++index) {
        if (index != 0) output << ',';
        output << ids[index];
    }
    output << ']';
}

std::string results_jsonl(const std::vector<Result> & results) {
    std::ostringstream output;
    output << std::setprecision(17);
    for (const Result & result : results) {
        if (!std::isfinite(result.elapsed_seconds) || result.elapsed_seconds < 0.0) {
            fail("attempted to serialize invalid document elapsed_seconds");
        }
        output << "{\"index\":" << result.index
               << ",\"item_id\":" << json_escape(result.item_id)
               << ",\"prompt_ids\":";
        append_ids(output, result.prompt_ids);
        output << ",\"generated_ids\":";
        append_ids(output, result.generated_ids);
        output << ",\"stop_reason\":\"" << result.stop_reason
               << "\",\"generated_count\":" << result.generated_ids.size()
               << ",\"elapsed_seconds\":" << result.elapsed_seconds
               << ",\"run32\":" << (result.run32 ? "true" : "false")
               << ",\"loop8x3\":" << (result.loop8x3 ? "true" : "false")
               << ",\"empty\":" << (result.empty ? "true" : "false") << "}\n";
    }
    return output.str();
}

std::string metadata_json(
        const Options & options,
        const std::vector<Result> & results,
        const llama_vocab * vocab,
        int32_t n_vocab,
        uint32_t physical_n_ctx,
        double total_elapsed_seconds) {
    if (!std::isfinite(total_elapsed_seconds) || total_elapsed_seconds < 0.0) {
        fail("attempted to serialize invalid total_elapsed_seconds");
    }
    uint64_t prompt_tokens = 0;
    uint64_t generated_tokens = 0;
    for (const Result & result : results) {
        prompt_tokens += result.prompt_ids.size();
        generated_tokens += result.generated_ids.size();
    }
    std::ostringstream output;
    output << std::setprecision(17)
           << "{\"schema\":\"strat01_gguf_rollout_runtime_v1\""
           << ",\"items\":" << results.size()
           << ",\"prompt_tokens\":" << prompt_tokens
           << ",\"generated_tokens\":" << generated_tokens
           << ",\"total_tokens\":" << prompt_tokens + generated_tokens
           << ",\"total_elapsed_seconds\":" << total_elapsed_seconds
           << ",\"logical_n_ctx\":" << options.n_ctx
           << ",\"physical_n_ctx\":" << physical_n_ctx
           << ",\"n_batch\":" << options.n_batch
           << ",\"n_threads\":" << options.n_threads
           << ",\"model_bos_id\":" << llama_vocab_bos(vocab)
           << ",\"model_eos_id\":" << llama_vocab_eos(vocab)
           << ",\"model_vocab_size\":" << n_vocab
           << ",\"source_token_ids_mode\":true"
           << ",\"greedy\":true"
           << ",\"tie_policy\":\"lowest_id\""
           << ",\"kv_reset_per_document\":true"
           << ",\"generation_max_new_tokens\":" << kMaxNewTokens
           << ",\"flag_definitions\":{\"run32\":\"at least 32 consecutive identical non-EOS generated IDs\""
           << ",\"loop8x3\":\"an 8-ID non-EOS sequence repeated three consecutive times\""
           << ",\"empty\":\"zero non-EOS generated IDs\"}}\n";
    return output.str();
}

Options parse_options(int argc, char ** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--self-test") { options.self_test = true; continue; }
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
    if (options.output == options.metadata_output) fail("--output and --metadata-out must be distinct paths");
    if (options.n_batch > options.n_ctx) fail("n-batch must not exceed n-ctx");
    if (kPromptTokens + kMaxNewTokens > static_cast<size_t>(options.n_ctx)) {
        fail("n-ctx is too small for the frozen 257-token prompt and 256-token cap");
    }
    return options;
}

template <typename Function>
void expect_failure(Function && function, const char * label) {
    try {
        function();
    } catch (const std::runtime_error &) {
        return;
    }
    fail(std::string("self-test accepted invalid case: ") + label);
}

std::string selftest_record(uint64_t index, const std::string & item_id, size_t prompt_count = kPromptTokens) {
    std::ostringstream output;
    output << "{\"index\":" << index << ",\"item_id\":" << json_escape(item_id) << ",\"prompt_ids\":[";
    for (size_t token = 0; token < prompt_count; ++token) {
        if (token != 0) output << ',';
        output << (token == 0 ? kPinnedBos : 3);
    }
    output << "]}";
    return output.str();
}

void self_test() {
    if (stable_argmax(std::array<float, 4>{-1.0F, 3.0F, 3.0F, 2.0F}.data(), 4) != 1) {
        fail("lowest-ID stable argmax self-test failed");
    }
    const std::array<float, 3> non_finite = {0.0F, std::numeric_limits<float>::quiet_NaN(), 1.0F};
    expect_failure([&]() { static_cast<void>(stable_argmax(non_finite.data(), 3)); }, "non-finite logits");

    const auto empty = rollout_flags({});
    if (empty != std::array<bool, 3>{false, false, true}) fail("empty rollout flag self-test failed");
    const std::vector<llama_token> run32_ids(32, 9);
    const auto run32 = rollout_flags(run32_ids);
    if (run32 != std::array<bool, 3>{true, true, false}) fail("run32 rollout flag self-test failed");
    std::vector<llama_token> loop8_ids;
    for (int copy = 0; copy < 3; ++copy) {
        for (int token = 0; token < 8; ++token) loop8_ids.push_back(static_cast<llama_token>(10 + token));
    }
    const auto loop8 = rollout_flags(loop8_ids);
    if (loop8 != std::array<bool, 3>{false, true, false}) fail("loop8x3 rollout flag self-test failed");
    if (rollout_flags({3, 4, 5}) != std::array<bool, 3>{false, false, false}) {
        fail("non-degenerate rollout flag self-test failed");
    }
    expect_failure([&]() { static_cast<void>(rollout_flags({kPinnedEos})); }, "retained EOS");

    const Item valid = parse_item_line(selftest_record(0, "document-0"));
    validate_item_shape(valid, kDefaultContext);
    expect_failure([&]() { static_cast<void>(parse_item_line("{\"index\":-1,\"item_id\":\"x\",\"prompt_ids\":[1]}")); },
                   "negative index");
    expect_failure([&]() { static_cast<void>(parse_item_line("{\"index\":0,\"item_id\":\"x\"}")); },
                   "missing prompt_ids");
    const Item too_short = parse_item_line(selftest_record(0, "short", kPromptTokens - 1U));
    expect_failure([&]() { validate_item_shape(too_short, kDefaultContext); }, "short prompt");
    Item duplicate = valid;
    duplicate.index = 1;
    expect_failure([&]() { validate_unique_items({valid, duplicate}); }, "duplicate item_id");
    expect_failure([&]() { validate_item_shape(valid, static_cast<int>(kPromptTokens + kMaxNewTokens - 1U)); },
                   "context overflow");
}

int run(const Options & options) {
    if (options.self_test) { self_test(); return 0; }
    ensure_new_path(options.output, "rollout output");
    ensure_new_path(options.metadata_output, "rollout metadata output");
    const std::vector<Item> items = read_input(options.input, options.n_ctx);
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
        if (n_vocab <= 0 || llama_vocab_bos(vocab) != kPinnedBos || llama_vocab_eos(vocab) != kPinnedEos) {
            fail("GGUF does not expose pinned GigaChat BOS=1 and EOS=2");
        }
        for (const Item & item : items) validate_ids(item, n_vocab);

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
            fail("cannot create the requested rollout llama context");
        }
        llama_batch batch = llama_batch_init(options.n_batch, 0, 1);
        std::vector<Result> results;
        results.reserve(items.size());
        const auto rollout_started = std::chrono::steady_clock::now();
        try {
            for (const Item & item : items) {
                results.push_back(rollout_item(context, batch, item, n_vocab, options.n_batch));
            }
        } catch (...) {
            llama_batch_free(batch);
            throw;
        }
        llama_batch_free(batch);
        const double total_elapsed = checked_elapsed(std::chrono::steady_clock::now() - rollout_started,
                                                     "total_elapsed_seconds");
        // Both serializations happen only after all selected documents and all logits
        // have been accepted.  In particular, non-finite logits never produce
        // a rollout JSONL that could be mistaken for a valid arm.
        const std::string output_jsonl = results_jsonl(results);
        const std::string runtime_json = metadata_json(options, results, vocab, n_vocab, llama_n_ctx(context), total_elapsed);
        write_text_new(options.output, output_jsonl, "rollout output");
        write_text_new(options.metadata_output, runtime_json, "rollout metadata output");
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
        std::cerr << "strat01_gguf_rollout: " << error.what() << '\n';
        return 2;
    }
}
