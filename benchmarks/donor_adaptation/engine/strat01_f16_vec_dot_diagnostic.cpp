#include "ggml.h"
#include "ggml-cpu/vec.h"

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

constexpr size_t kTokens = 8, kHeads = 32, kSlots = 256, kQk = 576, kLatent = 512;
constexpr size_t kScores = kTokens * kHeads * 8;
constexpr size_t kLatents = kTokens * kHeads * kLatent;

struct Options {
    std::filesystem::path qcur, kcur, padded_softmax, output;
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
        else if (argument == "--padded-softmax") result.padded_softmax = value("--padded-softmax");
        else if (argument == "--output-dir") result.output = value("--output-dir");
        else if (argument == "--selftest") result.selftest = true;
        else fail("unknown argument: " + argument);
    }
    return result;
}

uint32_t f32_bits(float value) { uint32_t bits; std::memcpy(&bits, &value, 4); return bits; }

uint16_t project_f32_to_f16(float value) {
    const uint32_t bits = f32_bits(value), sign = (bits >> 16) & 0x8000U, mantissa = bits & 0x7fffffU;
    int exponent = static_cast<int>((bits >> 23) & 0xffU);
    if (exponent == 255) return static_cast<uint16_t>(sign | (mantissa ? 0x7e00U : 0x7c00U));
    exponent -= 127;
    if (exponent > 15) return static_cast<uint16_t>(sign | 0x7c00U);
    if (exponent < -24) return static_cast<uint16_t>(sign);
    if (exponent < -14) {
        const unsigned shift = static_cast<unsigned>(-exponent - 14), rshift = shift + 13U;
        const uint32_t m = mantissa | 0x800000U, tie = UINT32_C(1) << (rshift - 1U);
        uint32_t half = m >> rshift, remainder = m & ((UINT32_C(1) << rshift) - 1U);
        if (remainder > tie || (remainder == tie && (half & 1U))) ++half;
        return static_cast<uint16_t>(sign | half);
    }
    uint32_t half_exponent = static_cast<uint32_t>(exponent + 15) << 10, half = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffU;
    if (remainder > 0x1000U || (remainder == 0x1000U && (half & 1U))) {
        ++half;
        if (half == 0x400U) { half = 0; half_exponent += 0x400U; }
    }
    return static_cast<uint16_t>(sign | half_exponent | half);
}

std::vector<float> read_f32(const std::filesystem::path &path, size_t count) {
    if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) != count * sizeof(float)) fail("float payload size mismatch");
    std::vector<float> values(count);
    std::ifstream stream(path, std::ios::binary);
    if (!stream.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(count * sizeof(float)))) fail("cannot read float payload");
    for (float value : values) if (!std::isfinite(value)) fail("non-finite float input");
    return values;
}

template <typename T>
void write_values(const std::filesystem::path &path, const std::vector<T> &values) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)))) fail("cannot write output");
}

struct Products {
    std::vector<float> qk_scalar, qk_vec, qk_mutated, value_scalar, value_vec, value_mutated;
    std::vector<uint16_t> project_f16, pinned_f16;
};

Products evaluate(const std::vector<float> &qcur, const std::vector<float> &kcur, const std::vector<float> &softmax) {
    Products result;
    result.qk_scalar.resize(kScores); result.qk_vec.resize(kScores); result.qk_mutated.resize(kScores);
    result.value_scalar.resize(kLatents); result.value_vec.resize(kLatents); result.value_mutated.resize(kLatents);
    std::vector<ggml_fp16_t> key_cache(kTokens * kQk), query(kQk), probabilities(kSlots), values(kSlots);
    const size_t audit_count = kcur.size() + qcur.size() + softmax.size();
    result.project_f16.reserve(audit_count); result.pinned_f16.reserve(audit_count);
    auto convert = [&](float value) -> ggml_fp16_t {
        const uint16_t project = project_f32_to_f16(value);
        const ggml_fp16_t pinned = ggml_fp32_to_fp16(value);
        result.project_f16.push_back(project); result.pinned_f16.push_back(pinned);
        if (project != pinned) fail("project and pinned F16 conversion differ");
        return pinned;
    };
    for (size_t i = 0; i < kcur.size(); ++i) key_cache[i] = convert(kcur[i]);
    for (size_t token = 0; token < kTokens; ++token) {
        for (size_t head = 0; head < kHeads; ++head) {
            const float *q = qcur.data() + (token * kHeads + head) * kQk;
            for (size_t i = 0; i < kQk; ++i) query[i] = convert(q[i]);
            for (size_t slot = 0; slot < kTokens; ++slot) {
                float scalar = 0.0f, pinned = 0.0f, mutated = 0.0f;
                const auto *key = key_cache.data() + slot * kQk;
                for (size_t i = 0; i < kQk; ++i) scalar += ggml_fp16_to_fp32(key[i]) * ggml_fp16_to_fp32(query[i]);
                ggml_vec_dot_f16(static_cast<int>(kQk), &pinned, 0, const_cast<ggml_fp16_t *>(key), 0, query.data(), 0, 1);
                const ggml_fp16_t saved = query[0];
                if (token == 7 && head == 31) query[0] = static_cast<ggml_fp16_t>(query[0] ^ 0x0100U);
                ggml_vec_dot_f16(static_cast<int>(kQk), &mutated, 0, const_cast<ggml_fp16_t *>(key), 0, query.data(), 0, 1);
                query[0] = saved;
                const size_t out = (token * kHeads + head) * kTokens + slot;
                result.qk_scalar[out] = scalar; result.qk_vec[out] = pinned; result.qk_mutated[out] = mutated;
            }
            const float *p = softmax.data() + (token * kHeads + head) * kSlots;
            for (size_t slot = 0; slot < kSlots; ++slot) probabilities[slot] = convert(p[slot]);
            for (size_t feature = 0; feature < kLatent; ++feature) {
                for (size_t slot = 0; slot < kSlots; ++slot) values[slot] = slot < kTokens ? key_cache[slot * kQk + feature] : 0;
                float scalar = 0.0f, pinned = 0.0f, mutated = 0.0f;
                for (size_t slot = 0; slot < kSlots; ++slot) scalar += ggml_fp16_to_fp32(values[slot]) * ggml_fp16_to_fp32(probabilities[slot]);
                ggml_vec_dot_f16(static_cast<int>(kSlots), &pinned, 0, values.data(), 0, probabilities.data(), 0, 1);
                const ggml_fp16_t saved = probabilities[0];
                if (token == 7 && head == 31) probabilities[0] = static_cast<ggml_fp16_t>(probabilities[0] ^ 0x0100U);
                ggml_vec_dot_f16(static_cast<int>(kSlots), &mutated, 0, values.data(), 0, probabilities.data(), 0, 1);
                probabilities[0] = saved;
                const size_t out = (token * kHeads + head) * kLatent + feature;
                result.value_scalar[out] = scalar; result.value_vec[out] = pinned; result.value_mutated[out] = mutated;
            }
        }
    }
    return result;
}

int selftest() {
    const float cases[] = {0.0f, -0.0f, 1.0f, -2.0f, 65504.0f, 0x1p-24f, 1.00048828125f};
    for (float value : cases) if (project_f32_to_f16(value) != ggml_fp32_to_fp16(value)) fail("selftest F16 conversion mismatch");
    std::cout << "F16 vec-dot diagnostic selftest passed\n";
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse(argc, argv);
        if (options.selftest) return selftest();
        if (options.qcur.empty() || options.kcur.empty() || options.padded_softmax.empty() || options.output.empty()) fail("qcur, kcur, padded-softmax and output-dir are required");
        std::filesystem::create_directories(options.output);
        const auto products = evaluate(
            read_f32(options.qcur, kTokens * kHeads * kQk),
            read_f32(options.kcur, kTokens * kQk),
            read_f32(options.padded_softmax, kTokens * kHeads * kSlots));
        write_values(options.output / "qk_f16_scalar.f32le", products.qk_scalar);
        write_values(options.output / "qk_f16_vec_dot.f32le", products.qk_vec);
        write_values(options.output / "qk_mutated.f32le", products.qk_mutated);
        write_values(options.output / "value_f16_scalar.f32le", products.value_scalar);
        write_values(options.output / "value_f16_vec_dot.f32le", products.value_vec);
        write_values(options.output / "value_mutated.f32le", products.value_mutated);
        write_values(options.output / "project_f16.bin", products.project_f16);
        write_values(options.output / "pinned_f16.bin", products.pinned_f16);
        std::cout << "wrote F16 vec-dot diagnostic payloads\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "F16 vec-dot diagnostic: " << error.what() << '\n';
        return 2;
    }
}
