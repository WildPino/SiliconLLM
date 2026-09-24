#include "ggml-cpu.h"
#include "ggml-cpu/quants.h"
#include "ggml-quants.h"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<std::uint8_t> read_exact(const std::string &path, std::size_t bytes) {
    std::vector<std::uint8_t> value(bytes);
    std::ifstream stream(path, std::ios::binary);
    if (!stream.read(reinterpret_cast<char *>(value.data()), static_cast<std::streamsize>(bytes)) ||
        stream.peek() != EOF)
        throw std::runtime_error("oracle input byte count mismatch: " + path);
    return value;
}
void write_exact(const std::string &path, float value) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream.write(reinterpret_cast<const char *>(&value), sizeof(value)))
        throw std::runtime_error("oracle output failure: " + path);
}
void write_exact(const std::string &path, const void *data, std::size_t bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(bytes)))
        throw std::runtime_error("oracle output failure: " + path);
}
unsigned long parse_positive(const char *text, const char *label) {
    char *end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (errno || !end || *end || !value || value > UINT32_MAX)
        throw std::runtime_error(std::string("invalid ") + label);
    return value;
}
std::uint64_t parse_offset(const char *text) {
    char *end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno || !end || *end) throw std::runtime_error("invalid matrix offset");
    return static_cast<std::uint64_t>(value);
}

int full_matrix(int argc, char **argv) {
    if (argc != 10) {
        throw std::runtime_error(
            "usage: q6_reference_generic_oracle matrix <n> <rows> <batch> <offset> "
            "<model> <input.f32le> <q8.bin> <output.f32le>"
        );
    }
    const unsigned count = static_cast<unsigned>(parse_positive(argv[2], "vector length"));
    const unsigned rows = static_cast<unsigned>(parse_positive(argv[3], "row count"));
    const unsigned batch = static_cast<unsigned>(parse_positive(argv[4], "batch count"));
    const std::uint64_t offset = parse_offset(argv[5]);
    if (count % QK_K) throw std::runtime_error("vector length is not QK_K aligned");
    const std::size_t blocks = count / QK_K;
    const std::size_t row_bytes = blocks * sizeof(block_q6_K);
    const auto input_bytes = read_exact(
        argv[7], static_cast<std::size_t>(batch) * count * sizeof(float));
    std::vector<float> input(static_cast<std::size_t>(batch) * count);
    std::memcpy(input.data(), input_bytes.data(), input_bytes.size());
    std::vector<block_q8_K> q8(static_cast<std::size_t>(batch) * blocks);
    std::vector<std::uint8_t> q6(row_bytes);
    std::vector<float> output(static_cast<std::size_t>(batch) * rows);
    std::ifstream model(argv[6], std::ios::binary);
    if (!model.seekg(static_cast<std::streamoff>(offset)))
        throw std::runtime_error("oracle matrix seek failure");
    ggml_cpu_init();
    for (unsigned item = 0; item < batch; ++item) {
        quantize_row_q8_K(input.data() + static_cast<std::size_t>(item) * count,
                          q8.data() + static_cast<std::size_t>(item) * blocks, count);
    }
    for (unsigned row = 0; row < rows; ++row) {
        if (!model.read(reinterpret_cast<char *>(q6.data()), static_cast<std::streamsize>(row_bytes)))
            throw std::runtime_error("oracle matrix short read");
        for (unsigned item = 0; item < batch; ++item) {
            float dot = 0.0f;
            ggml_vec_dot_q6_K_q8_K(
                static_cast<int>(count), &dot, sizeof(dot), q6.data(), sizeof(block_q6_K),
                q8.data() + static_cast<std::size_t>(item) * blocks,
                sizeof(block_q8_K), 1);
            if (!std::isfinite(dot))
                throw std::runtime_error("oracle produced non-finite matrix output");
            output[static_cast<std::size_t>(item) * rows + row] = dot;
        }
    }
    write_exact(argv[8], q8.data(), q8.size() * sizeof(block_q8_K));
    write_exact(argv[9], output.data(), output.size() * sizeof(float));
    return 0;
}
}

int main(int argc, char **argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "matrix") return full_matrix(argc, argv);
        if (argc != 5)
            throw std::runtime_error("usage: q6_reference_generic_oracle <n> <q6.bin> <q8.bin> <dot.f32le>");
        static_assert(QK_K == 256 && sizeof(block_q6_K) == 210 && sizeof(block_q8_K) == 292);
        char *end = nullptr;
        errno = 0;
        const unsigned long parsed = std::strtoul(argv[1], &end, 10);
        if (errno || !end || *end || !parsed || parsed > UINT32_MAX || parsed % QK_K)
            throw std::runtime_error("invalid vector length");
        const int count = static_cast<int>(parsed);
        const std::size_t blocks = static_cast<std::size_t>(count / QK_K);
        const auto q6 = read_exact(argv[2], blocks * sizeof(block_q6_K));
        const auto q8 = read_exact(argv[3], blocks * sizeof(block_q8_K));
        float dot = 0.0f;
        ggml_cpu_init();
        ggml_vec_dot_q6_K_q8_K(count, &dot, sizeof(dot), q6.data(), sizeof(block_q6_K),
                              q8.data(), sizeof(block_q8_K), 1);
        if (!std::isfinite(dot)) throw std::runtime_error("oracle produced non-finite dot");
        write_exact(argv[4], dot);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "Q6_K/Q8_K reference-generic oracle: %s\n", error.what());
        return 2;
    }
}
