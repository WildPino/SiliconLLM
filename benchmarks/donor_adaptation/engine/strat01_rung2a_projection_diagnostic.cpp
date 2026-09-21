#include "ggml-cpu/quants.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int64_t kInput = 1536;
constexpr int64_t kTokens = 8;
constexpr int64_t kQRows = 6144;
constexpr int64_t kKvRows = 576;
constexpr uint64_t kQOffset = 285780864;
constexpr uint64_t kKvOffset = 279966592;

struct options {
    std::filesystem::path model;
    std::filesystem::path input;
    std::filesystem::path output;
    std::string tensor;
    std::string mode;
    bool mutate_q8 = false;
    bool selftest = false;
};

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

options parse(int argc, char **argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char *name) -> std::string {
            if (i + 1 >= argc) {
                fail(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--model") result.model = value("--model");
        else if (arg == "--input") result.input = value("--input");
        else if (arg == "--output") result.output = value("--output");
        else if (arg == "--tensor") result.tensor = value("--tensor");
        else if (arg == "--mode") result.mode = value("--mode");
        else if (arg == "--mutate-q8") result.mutate_q8 = true;
        else if (arg == "--selftest") result.selftest = true;
        else fail("unknown argument: " + arg);
    }
    return result;
}

std::vector<float> read_f32(const std::filesystem::path &path, size_t count) {
    if (std::filesystem::file_size(path) != count * sizeof(float)) {
        fail("input float payload has wrong byte count");
    }
    std::vector<float> data(count);
    std::ifstream stream(path, std::ios::binary);
    if (!stream.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(count * sizeof(float)))) {
        fail("cannot read input float payload");
    }
    for (float value : data) {
        if (!std::isfinite(value)) fail("input float payload contains non-finite value");
    }
    return data;
}

std::vector<uint8_t> read_weights(const std::filesystem::path &path, uint64_t offset, int64_t rows) {
    static_assert(QK_K == 256);
    static_assert(sizeof(block_q4_K) == 144);
    const size_t row_bytes = static_cast<size_t>(kInput / QK_K) * sizeof(block_q4_K);
    std::vector<uint8_t> data(static_cast<size_t>(rows) * row_bytes);
    std::ifstream stream(path, std::ios::binary);
    stream.seekg(static_cast<std::streamoff>(offset));
    if (!stream || !stream.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size()))) {
        fail("cannot read exact Q4_K weight span");
    }
    return data;
}

void write_f32(const std::filesystem::path &path, const std::vector<float> &data) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)))) {
        fail("cannot write output float payload");
    }
}

std::vector<float> compute_d32(
    const std::vector<uint8_t> &weights,
    const std::vector<float> &input,
    int64_t rows,
    bool use_f64) {
    const size_t row_bytes = static_cast<size_t>(kInput / QK_K) * sizeof(block_q4_K);
    std::vector<float> output(static_cast<size_t>(kTokens * rows));
    std::vector<float> decoded(static_cast<size_t>(kInput));
    for (int64_t row = 0; row < rows; ++row) {
        const auto *blocks = reinterpret_cast<const block_q4_K *>(weights.data() + static_cast<size_t>(row) * row_bytes);
        dequantize_row_q4_K(blocks, decoded.data(), kInput);
        for (int64_t token = 0; token < kTokens; ++token) {
            const float *x = input.data() + token * kInput;
            if (use_f64) {
                double sum = 0.0;
                for (int64_t j = 0; j < kInput; ++j) sum += static_cast<double>(decoded[j]) * x[j];
                output[static_cast<size_t>(token * rows + row)] = static_cast<float>(sum);
            } else {
                float sum = 0.0f;
                for (int64_t j = 0; j < kInput; ++j) sum += decoded[j] * x[j];
                output[static_cast<size_t>(token * rows + row)] = sum;
            }
        }
    }
    return output;
}

std::vector<float> compute_q8k(
    const std::vector<uint8_t> &weights,
    const std::vector<float> &input,
    int64_t rows,
    bool mutate) {
    const size_t row_bytes = static_cast<size_t>(kInput / QK_K) * sizeof(block_q4_K);
    std::vector<float> output(static_cast<size_t>(kTokens * rows));
    std::vector<block_q8_K> quantized(static_cast<size_t>(kInput / QK_K));
    for (int64_t token = 0; token < kTokens; ++token) {
        quantize_row_q8_K(input.data() + token * kInput, quantized.data(), kInput);
        if (mutate && token == 0) {
            float baseline = 0.0f;
            ggml_vec_dot_q4_K_q8_K(kInput, &baseline, sizeof(float), weights.data(), row_bytes, quantized.data(), quantized.size() * sizeof(block_q8_K), 1);
            bool fired = false;
            for (size_t block = 0; block < quantized.size() && !fired; ++block) {
                auto *scale_bytes = reinterpret_cast<uint8_t *>(&quantized[block].d);
                const uint8_t saved = scale_bytes[2];
                scale_bytes[2] ^= 0x80;
                float changed = 0.0f;
                ggml_vec_dot_q4_K_q8_K(kInput, &changed, sizeof(float), weights.data(), row_bytes, quantized.data(), quantized.size() * sizeof(block_q8_K), 1);
                if (std::isfinite(changed) && changed != baseline) {
                    std::cout << "mutated Q8_K activation scale byte block=" << block << " byte=2\n";
                    fired = true;
                } else {
                    scale_bytes[2] = saved;
                }
            }
            if (!fired) fail("no one-byte Q8_K mutation changed the first stored-row dot");
        }
        for (int64_t row = 0; row < rows; ++row) {
            const void *weight_row = weights.data() + static_cast<size_t>(row) * row_bytes;
            float sum = 0.0f;
            ggml_vec_dot_q4_K_q8_K(kInput, &sum, sizeof(float), weight_row, row_bytes, quantized.data(), quantized.size() * sizeof(block_q8_K), 1);
            if (!std::isfinite(sum)) fail("projection produced non-finite value");
            output[static_cast<size_t>(token * rows + row)] = sum;
        }
    }
    return output;
}

int selftest() {
    std::vector<float> weights(static_cast<size_t>(kInput));
    std::vector<float> input(static_cast<size_t>(kInput));
    for (int64_t i = 0; i < kInput; ++i) {
        weights[static_cast<size_t>(i)] = std::sin(static_cast<float>(i) * 0.017f) * 0.25f;
        input[static_cast<size_t>(i)] = std::cos(static_cast<float>(i) * 0.013f);
    }
    std::vector<block_q4_K> q4(static_cast<size_t>(kInput / QK_K));
    std::vector<block_q8_K> q8(static_cast<size_t>(kInput / QK_K));
    quantize_row_q4_K_ref(weights.data(), q4.data(), kInput);
    quantize_row_q8_K(input.data(), q8.data(), kInput);
    float baseline = 0.0f;
    ggml_vec_dot_q4_K_q8_K(kInput, &baseline, sizeof(float), q4.data(), q4.size() * sizeof(block_q4_K), q8.data(), q8.size() * sizeof(block_q8_K), 1);
    std::vector<float> decoded(static_cast<size_t>(kInput));
    dequantize_row_q4_K(q4.data(), decoded.data(), kInput);
    float decoded_dot = 0.0f;
    for (int64_t i = 0; i < kInput; ++i) decoded_dot += decoded[static_cast<size_t>(i)] * input[static_cast<size_t>(i)];
    std::cout << "selftest baseline=" << baseline << " decoded_dot=" << decoded_dot << " q8_d0=" << q8[0].d << "\n";
    float mutated = baseline;
    bool fired = false;
    for (size_t block = 0; block < q8.size() && !fired; ++block) {
        auto *scale_bytes = reinterpret_cast<uint8_t *>(&q8[block].d);
        const uint8_t saved = scale_bytes[2];
        scale_bytes[2] ^= 0x80;
        float candidate = 0.0f;
        ggml_vec_dot_q4_K_q8_K(kInput, &candidate, sizeof(float), q4.data(), q4.size() * sizeof(block_q4_K), q8.data(), q8.size() * sizeof(block_q8_K), 1);
        if (std::isfinite(candidate) && candidate != baseline) {
            mutated = candidate;
            fired = true;
        } else {
            scale_bytes[2] = saved;
        }
    }
    if (!fired || !std::isfinite(baseline) || !std::isfinite(mutated) || baseline == mutated) {
        fail("Q8_K mutation self-test did not fire");
    }
    std::cout << "projection diagnostic selftest passed\n";
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        ggml_cpu_init();
        const options opts = parse(argc, argv);
        if (opts.selftest) return selftest();
        if (opts.model.empty() || opts.input.empty() || opts.output.empty()) fail("model, input, and output are required");
        const bool is_q = opts.tensor == "q";
        if (!is_q && opts.tensor != "kv") fail("tensor must be q or kv");
        if (opts.mode != "d32" && opts.mode != "q8k" && opts.mode != "f64") fail("mode must be d32, q8k, or f64");
        if (opts.mutate_q8 && opts.mode != "q8k") fail("--mutate-q8 requires --mode q8k");
        const int64_t rows = is_q ? kQRows : kKvRows;
        const uint64_t offset = is_q ? kQOffset : kKvOffset;
        const auto input = read_f32(opts.input, static_cast<size_t>(kTokens * kInput));
        const auto weights = read_weights(opts.model, offset, rows);
        const auto output = opts.mode == "q8k"
            ? compute_q8k(weights, input, rows, opts.mutate_q8)
            : compute_d32(weights, input, rows, opts.mode == "f64");
        write_f32(opts.output, output);
        std::cout << "wrote " << output.size() << " float32 values\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "projection diagnostic: " << error.what() << "\n";
        return 2;
    }
}
