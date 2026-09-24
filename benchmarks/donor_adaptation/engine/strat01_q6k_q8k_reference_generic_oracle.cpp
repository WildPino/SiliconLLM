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
}

int main(int argc, char **argv) {
    try {
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
