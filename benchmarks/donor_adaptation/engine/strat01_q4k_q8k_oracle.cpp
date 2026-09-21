#include "ggml-cpu.h"
#include "ggml-cpu/quants.h"
#include "ggml-quants.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

template <typename T>
T read_exact(const std::string &path) {
    T value{};
    std::ifstream stream(path, std::ios::binary);
    if (!stream.read(reinterpret_cast<char *>(&value), sizeof(value)) || stream.peek() != EOF) {
        throw std::runtime_error("oracle input byte count mismatch: " + path);
    }
    return value;
}

template <typename T>
void write_exact(const std::string &path, const T &value) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream.write(reinterpret_cast<const char *>(&value), sizeof(value))) {
        throw std::runtime_error("oracle output failure: " + path);
    }
}

struct input_block { float values[QK_K]; };

}  // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 5) {
            throw std::runtime_error("usage: oracle <input.f32le> <q4.bin> <q8.bin> <dot.f32le>");
        }
        static_assert(QK_K == 256);
        static_assert(sizeof(block_q4_K) == 144);
        static_assert(sizeof(block_q8_K) == 292);
        ggml_cpu_init();
        const input_block input = read_exact<input_block>(argv[1]);
        const block_q4_K q4 = read_exact<block_q4_K>(argv[2]);
        block_q8_K q8{};
        float dot = 0.0f;
        quantize_row_q8_K_ref(input.values, &q8, QK_K);
        ggml_vec_dot_q4_K_q8_K(QK_K, &dot, sizeof(dot), &q4, sizeof(q4), &q8, sizeof(q8), 1);
        if (!std::isfinite(dot)) throw std::runtime_error("oracle produced non-finite dot");
        write_exact(argv[3], q8);
        write_exact(argv[4], dot);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "Q4_K/Q8_K oracle: %s\n", error.what());
        return 2;
    }
}
