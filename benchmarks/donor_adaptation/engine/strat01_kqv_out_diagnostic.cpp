#include "ggml-cpu.h"
#include "ggml-cpu/quants.h"
#include "ggml-quants.h"
#include "../../phase60/strat01_q4k_q8k.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
constexpr size_t kValueHead = 192;
constexpr size_t kOutput = kHeads * kValueHead;
constexpr uint64_t kVbOffset = 291089280;
constexpr size_t kBlocks = kLatent / QK_K;
constexpr size_t kRowBytes = kBlocks * sizeof(block_q4_K);
constexpr size_t kRows = kHeads * kValueHead;

struct options {
    std::filesystem::path model, qcur, kcur, vcur, latent, output;
    bool selftest = false;
};

[[noreturn]] void fail(const std::string &message) { throw std::runtime_error(message); }

options parse(int argc, char **argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto value = [&](const char *name) -> std::filesystem::path {
            if (i + 1 >= argc) fail(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (argument == "--model") result.model = value("--model");
        else if (argument == "--qcur") result.qcur = value("--qcur");
        else if (argument == "--kcur") result.kcur = value("--kcur");
        else if (argument == "--vcur") result.vcur = value("--vcur");
        else if (argument == "--latent") result.latent = value("--latent");
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

std::vector<uint8_t> read_vb(const std::filesystem::path &model) {
    std::vector<uint8_t> weights(kRows * kRowBytes);
    std::ifstream stream(model, std::ios::binary);
    stream.seekg(static_cast<std::streamoff>(kVbOffset));
    if (!stream || !stream.read(reinterpret_cast<char *>(weights.data()), static_cast<std::streamsize>(weights.size()))) {
        fail("cannot read exact V-B span");
    }
    return weights;
}

template <typename T>
void write_vector(const std::filesystem::path &path, const std::vector<T> &values) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)))) {
        fail("cannot write output: " + path.string());
    }
}

float kq_scale() {
    const float m = 1.0f + 0.1f * std::log(64.0f);
    return m * m / std::sqrt(192.0f);
}

std::vector<float> reconstruct_attention(
    const std::vector<float> &qcur,
    const std::vector<float> &kcur,
    const std::vector<float> &vcur) {
    std::vector<ggml_fp16_t> cache(kTokens * kQk);
    std::vector<float> output(kTokens * kHeads * kLatent);
    for (size_t token = 0; token < kTokens; ++token) {
        for (size_t i = 0; i < kQk; ++i) cache[token * kQk + i] = ggml_fp32_to_fp16(kcur[token * kQk + i]);
        for (size_t i = 0; i < kLatent; ++i) {
            if (kcur[token * kQk + i] != vcur[token * kLatent + i]) fail("Kcur latent prefix differs from Vcur");
        }
    }
    const float scale = kq_scale();
    for (size_t token = 0; token < kTokens; ++token) {
        for (size_t head = 0; head < kHeads; ++head) {
            float scores[kTokens], maximum = -INFINITY, sum = 0.0f;
            const float *query = qcur.data() + (token * kHeads + head) * kQk;
            for (size_t slot = 0; slot <= token; ++slot) {
                float dot = 0.0f;
                for (size_t i = 0; i < kQk; ++i) dot += query[i] * ggml_fp16_to_fp32(cache[slot * kQk + i]);
                scores[slot] = dot * scale;
                maximum = std::max(maximum, scores[slot]);
            }
            for (size_t slot = 0; slot <= token; ++slot) {
                scores[slot] = std::exp(scores[slot] - maximum);
                sum += scores[slot];
            }
            for (size_t i = 0; i < kLatent; ++i) {
                float value = 0.0f;
                for (size_t slot = 0; slot <= token; ++slot) {
                    value += (scores[slot] / sum) * ggml_fp16_to_fp32(cache[slot * kQk + i]);
                }
                output[(token * kHeads + head) * kLatent + i] = value;
            }
        }
    }
    return output;
}

struct products {
    std::vector<float> d32, project, pinned, mutated, transposed, wrong_scales;
    std::vector<strat01_q8_k_block> project_q8;
    std::vector<block_q8_K> pinned_q8;
};

products evaluate(const std::vector<uint8_t> &weights, const std::vector<float> &latent) {
    products result;
    result.d32.resize(kTokens * kOutput);
    result.project.resize(kTokens * kOutput);
    result.pinned.resize(kTokens * kOutput);
    result.mutated.resize(kTokens * kOutput);
    result.transposed.resize(kTokens * kOutput);
    result.wrong_scales.resize(kTokens * kOutput);
    const size_t activation_rows = kTokens * kHeads;
    result.project_q8.resize(activation_rows * kBlocks);
    result.pinned_q8.resize(activation_rows * kBlocks);
    std::vector<float> decoded(kLatent);

    for (size_t token = 0; token < kTokens; ++token) {
        for (size_t head = 0; head < kHeads; ++head) {
            const size_t activation_index = token * kHeads + head;
            const float *input = latent.data() + activation_index * kLatent;
            auto *project_q8 = result.project_q8.data() + activation_index * kBlocks;
            auto *pinned_q8 = result.pinned_q8.data() + activation_index * kBlocks;
            if (!strat01_quantize_q8_k_row(input, kLatent, project_q8)) fail("project Q8_K quantization failed");
            quantize_row_q8_K_ref(input, pinned_q8, kLatent);
            if (std::memcmp(project_q8, pinned_q8, kBlocks * sizeof(block_q8_K)) != 0) fail("project and pinned Q8_K bytes differ");
            for (size_t value = 0; value < kValueHead; ++value) {
                const size_t row = head * kValueHead + value;
                const uint8_t *raw = weights.data() + row * kRowBytes;
                float d32 = 0.0f, pinned = 0.0f;
                dequantize_row_q4_K(reinterpret_cast<const block_q4_K *>(raw), decoded.data(), kLatent);
                for (size_t i = 0; i < kLatent; ++i) d32 += decoded[i] * input[i];
                const float project = strat01_q4k_q8k_dot(raw, project_q8, kLatent);
                ggml_vec_dot_q4_K_q8_K(kLatent, &pinned, sizeof(float), raw, kRowBytes, pinned_q8, kBlocks * sizeof(block_q8_K), 1);

                strat01_q8_k_block mutated_blocks[kBlocks];
                std::memcpy(mutated_blocks, project_q8, sizeof(mutated_blocks));
                reinterpret_cast<uint8_t *>(&mutated_blocks[0].d)[2] ^= 0x80;
                const float mutated = strat01_q4k_q8k_dot(raw, mutated_blocks, kLatent);

                const size_t wrong_head = (head + 1U) % kHeads;
                const auto *wrong_q8 = result.project_q8.data() + (token * kHeads + wrong_head) * kBlocks;
                float transposed = NAN;
                if (wrong_head < head) transposed = strat01_q4k_q8k_dot(raw, wrong_q8, kLatent);

                uint8_t wrong_raw[kRowBytes];
                std::memcpy(wrong_raw, raw, kRowBytes);
                for (size_t block = 0; block < kBlocks; ++block) std::swap(wrong_raw[block * 144U + 4U], wrong_raw[block * 144U + 8U]);
                const float wrong_scale = strat01_q4k_q8k_dot(wrong_raw, project_q8, kLatent);

                const size_t output = token * kOutput + row;
                result.d32[output] = d32;
                result.project[output] = project;
                result.pinned[output] = pinned;
                result.mutated[output] = mutated;
                result.transposed[output] = transposed;
                result.wrong_scales[output] = wrong_scale;
            }
        }
        /* Fill forward head-shift cases after every head's Q8 blocks exist. */
        for (size_t head = 0; head < kHeads; ++head) {
            const size_t wrong_head = (head + 1U) % kHeads;
            const auto *wrong_q8 = result.project_q8.data() + (token * kHeads + wrong_head) * kBlocks;
            for (size_t value = 0; value < kValueHead; ++value) {
                const size_t row = head * kValueHead + value;
                const uint8_t *raw = weights.data() + row * kRowBytes;
                result.transposed[token * kOutput + row] = strat01_q4k_q8k_dot(raw, wrong_q8, kLatent);
            }
        }
    }
    return result;
}

int selftest() {
    float input[QK_K];
    for (size_t i = 0; i < QK_K; ++i) input[i] = std::sin(static_cast<float>(i) * 0.071f);
    strat01_q8_k_block project{};
    block_q8_K pinned{};
    if (!strat01_quantize_q8_k_block(input, &project)) fail("selftest quantization failed");
    quantize_row_q8_K_ref(input, &pinned, QK_K);
    if (std::memcmp(&project, &pinned, sizeof(pinned)) != 0) fail("selftest Q8_K bytes differ");
    std::cout << "kqv diagnostic selftest passed\n";
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        ggml_cpu_init();
        const options opts = parse(argc, argv);
        if (opts.selftest) return selftest();
        if (opts.model.empty() || opts.output.empty()) {
            fail("model and output-dir are required");
        }
        const bool captured_latent = !opts.latent.empty();
        const bool reconstructed_latent = !opts.qcur.empty() && !opts.kcur.empty() && !opts.vcur.empty();
        if (captured_latent == reconstructed_latent) {
            fail("provide exactly one of latent or qcur/kcur/vcur");
        }
        std::filesystem::create_directories(opts.output);
        const auto weights = read_vb(opts.model);
        std::vector<float> latent;
        if (captured_latent) {
            latent = read_f32(opts.latent, kTokens * kHeads * kLatent);
        } else {
            const auto qcur = read_f32(opts.qcur, kTokens * kHeads * kQk);
            const auto kcur = read_f32(opts.kcur, kTokens * kQk);
            const auto vcur = read_f32(opts.vcur, kTokens * kLatent);
            latent = reconstruct_attention(qcur, kcur, vcur);
        }
        const auto outputs = evaluate(weights, latent);
        write_vector(opts.output / "latent.f32le", latent);
        write_vector(opts.output / "d32.f32le", outputs.d32);
        write_vector(opts.output / "project_q8.f32le", outputs.project);
        write_vector(opts.output / "pinned_q8.f32le", outputs.pinned);
        write_vector(opts.output / "mutated_q8.f32le", outputs.mutated);
        write_vector(opts.output / "transposed_head.f32le", outputs.transposed);
        write_vector(opts.output / "wrong_scales.f32le", outputs.wrong_scales);
        write_vector(opts.output / "project_q8.bin", outputs.project_q8);
        write_vector(opts.output / "pinned_q8.bin", outputs.pinned_q8);
        std::cout << "wrote kqv_out diagnostic payloads\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "kqv_out diagnostic: " << error.what() << "\n";
        return 2;
    }
}
