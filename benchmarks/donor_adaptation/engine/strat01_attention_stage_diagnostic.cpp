#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kTokens = 8;
constexpr size_t kHeads = 32;
constexpr size_t kQk = 576;
constexpr size_t kLatent = 512;
constexpr size_t kScores = kTokens * kHeads * kTokens;
constexpr size_t kLatents = kTokens * kHeads * kLatent;

struct Options {
    std::filesystem::path qcur, kcur, vcur, captured_kq, captured_softmax, output;
    bool selftest = false;
};

[[noreturn]] void fail(const std::string &message) { throw std::runtime_error(message); }

Options parse(int argc, char **argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto value = [&](const char *name) -> std::filesystem::path {
            if (++i >= argc) fail(std::string("missing value for ") + name);
            return argv[i];
        };
        if (argument == "--qcur") result.qcur = value("--qcur");
        else if (argument == "--kcur") result.kcur = value("--kcur");
        else if (argument == "--vcur") result.vcur = value("--vcur");
        else if (argument == "--captured-kq") result.captured_kq = value("--captured-kq");
        else if (argument == "--captured-softmax") result.captured_softmax = value("--captured-softmax");
        else if (argument == "--output-dir") result.output = value("--output-dir");
        else if (argument == "--selftest") result.selftest = true;
        else fail("unknown argument: " + argument);
    }
    return result;
}

std::vector<float> read_f32(const std::filesystem::path &path, size_t count) {
    if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) != count * sizeof(float)) {
        fail("float payload byte count mismatch: " + path.string());
    }
    std::vector<float> values(count);
    std::ifstream stream(path, std::ios::binary);
    if (!stream.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)))) {
        fail("cannot read float payload: " + path.string());
    }
    for (float value : values) if (!std::isfinite(value)) fail("non-finite float input");
    return values;
}

void write_f32(const std::filesystem::path &path, const std::vector<float> &values) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)))) {
        fail("cannot write output: " + path.string());
    }
}

float kq_scale() {
    const float m = 1.0f + 0.1f * std::log(64.0f);
    return m * m / std::sqrt(192.0f);
}

std::vector<float> qk_dots(
    const std::vector<float> &qcur,
    const std::vector<float> &kcur,
    bool f16_cache,
    bool mutate_q) {
    std::vector<ggml_fp16_t> cache(kTokens * kQk);
    if (f16_cache) {
        for (size_t i = 0; i < cache.size(); ++i) cache[i] = ggml_fp32_to_fp16(kcur[i]);
    }
    std::vector<float> result(kScores);
    for (size_t token = 0; token < kTokens; ++token) {
        for (size_t head = 0; head < kHeads; ++head) {
            const float *query = qcur.data() + (token * kHeads + head) * kQk;
            for (size_t slot = 0; slot < kTokens; ++slot) {
                float dot = 0.0f;
                for (size_t i = 0; i < kQk; ++i) {
                    float q = query[i];
                    if (mutate_q && token == 7 && head == 31 && i == 0) q += 1.0f;
                    const float key = f16_cache ? ggml_fp16_to_fp32(cache[slot * kQk + i]) : kcur[slot * kQk + i];
                    dot += q * key;
                }
                result[(token * kHeads + head) * kTokens + slot] = dot;
            }
        }
    }
    return result;
}

std::vector<float> causal_softmax(const std::vector<float> &raw) {
    std::vector<float> result(kScores, 0.0f);
    const float scale = kq_scale();
    for (size_t token = 0; token < kTokens; ++token) {
        for (size_t head = 0; head < kHeads; ++head) {
            const size_t base = (token * kHeads + head) * kTokens;
            float maximum = -INFINITY;
            for (size_t slot = 0; slot <= token; ++slot) maximum = std::max(maximum, raw[base + slot] * scale);
            float sum = 0.0f;
            for (size_t slot = 0; slot <= token; ++slot) {
                result[base + slot] = std::exp(raw[base + slot] * scale - maximum);
                sum += result[base + slot];
            }
            for (size_t slot = 0; slot <= token; ++slot) result[base + slot] /= sum;
        }
    }
    return result;
}

std::vector<float> reduce_values(
    const std::vector<float> &probabilities,
    const std::vector<float> &kcur,
    const std::vector<float> &vcur,
    bool f16_cache,
    bool swap_probabilities) {
    std::vector<ggml_fp16_t> cache(kTokens * kQk);
    if (f16_cache) {
        for (size_t i = 0; i < cache.size(); ++i) cache[i] = ggml_fp32_to_fp16(kcur[i]);
    }
    std::vector<float> result(kLatents);
    for (size_t token = 0; token < kTokens; ++token) {
        for (size_t head = 0; head < kHeads; ++head) {
            const size_t base = (token * kHeads + head) * kTokens;
            for (size_t i = 0; i < kLatent; ++i) {
                float value = 0.0f;
                for (size_t slot = 0; slot <= token; ++slot) {
                    size_t probability_slot = slot;
                    if (swap_probabilities && token >= 1) {
                        if (slot == 0) probability_slot = 1;
                        else if (slot == 1) probability_slot = 0;
                    }
                    const float cached = f16_cache ? ggml_fp16_to_fp32(cache[slot * kQk + i]) : vcur[slot * kLatent + i];
                    value += probabilities[base + probability_slot] * cached;
                }
                result[(token * kHeads + head) * kLatent + i] = value;
            }
        }
    }
    return result;
}

int selftest() {
    std::vector<float> raw(kScores, 0.0f);
    const auto probabilities = causal_softmax(raw);
    for (size_t token = 0; token < kTokens; ++token) {
        for (size_t head = 0; head < kHeads; ++head) {
            const size_t base = (token * kHeads + head) * kTokens;
            float sum = 0.0f;
            for (size_t slot = 0; slot < kTokens; ++slot) sum += probabilities[base + slot];
            if (std::abs(sum - 1.0f) > 1e-6f) fail("selftest softmax normalization failed");
        }
    }
    std::cout << "attention-stage diagnostic selftest passed\n";
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse(argc, argv);
        if (options.selftest) return selftest();
        if (options.qcur.empty() || options.kcur.empty() || options.vcur.empty() ||
            options.captured_kq.empty() || options.captured_softmax.empty() || options.output.empty()) {
            fail("qcur, kcur, vcur, captured-kq, captured-softmax and output-dir are required");
        }
        std::filesystem::create_directories(options.output);
        const auto qcur = read_f32(options.qcur, kTokens * kHeads * kQk);
        const auto kcur = read_f32(options.kcur, kTokens * kQk);
        const auto vcur = read_f32(options.vcur, kTokens * kLatent);
        const auto captured_kq = read_f32(options.captured_kq, kScores);
        const auto captured_softmax = read_f32(options.captured_softmax, kScores);
        for (size_t token = 0; token < kTokens; ++token) {
            for (size_t i = 0; i < kLatent; ++i) {
                if (kcur[token * kQk + i] != vcur[token * kLatent + i]) fail("Kcur latent prefix differs from Vcur");
            }
        }
        const auto raw_qk = qk_dots(qcur, kcur, true, false);
        const auto mutated_qk = qk_dots(qcur, kcur, true, true);
        const auto f32_cache_qk = qk_dots(qcur, kcur, false, false);
        const auto project_softmax = causal_softmax(raw_qk);
        const auto captured_qk_softmax = causal_softmax(captured_kq);
        const auto project_latent = reduce_values(project_softmax, kcur, vcur, true, false);
        const auto captured_softmax_latent = reduce_values(captured_softmax, kcur, vcur, true, false);
        const auto swapped_probability_latent = reduce_values(captured_softmax, kcur, vcur, true, true);
        const auto f32_cache_latent = reduce_values(captured_softmax, kcur, vcur, false, false);
        write_f32(options.output / "raw_qk.f32le", raw_qk);
        write_f32(options.output / "mutated_qk.f32le", mutated_qk);
        write_f32(options.output / "f32_cache_qk.f32le", f32_cache_qk);
        write_f32(options.output / "project_softmax.f32le", project_softmax);
        write_f32(options.output / "captured_qk_softmax.f32le", captured_qk_softmax);
        write_f32(options.output / "project_latent.f32le", project_latent);
        write_f32(options.output / "captured_softmax_latent.f32le", captured_softmax_latent);
        write_f32(options.output / "swapped_probability_latent.f32le", swapped_probability_latent);
        write_f32(options.output / "f32_cache_latent.f32le", f32_cache_latent);
        std::cout << "wrote attention-stage diagnostic payloads\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "attention-stage diagnostic: " << error.what() << '\n';
        return 2;
    }
}
