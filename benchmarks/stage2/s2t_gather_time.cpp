// Stage 2 test 2b, the two desk numbers of the two-pass head: the selection of the K largest of 157,184 logits, and the
// exact rows, K random Q6_K rows of Ling-flash's width against one activation, on ggml's own row product.  Timing only: it
// needs an idle machine.  The head (528 MB) is cold: a 1 GB stream runs between repetitions, as the other organs would.
// Prints one line per cell with the median, p10 and p90 in milliseconds over the repetitions.
//
// Usage: s2t_gather_time_avx2.exe --threads 6 --reps 200 --json FILE
#include "ggml.h"
#include "ggml-cpu.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int V = 157184, D = 4096;
const int KS[] = {256, 1024, 4096};

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Stat { double p10, p50, p90; };

// The K largest by a sampled threshold: a strided sample names a value near the 2K-th, a parallel scan keeps what is above
// it, and nth_element runs on those few.  Falls back to the whole vocabulary if the sample's threshold kept too few.
void select_fast(const std::vector<float> & z, int k, int threads, std::vector<float> & sample, std::vector<std::vector<int>> & local,
                 std::vector<int> & kept, std::vector<int> & all) {
    constexpr int STRIDE = 37;
    sample.clear();
    for (int i = 0; i < V; i += STRIDE) sample.push_back(z[i]);
    const int at = std::min<int>(static_cast<int>(sample.size()) - 1, 2 * k / STRIDE + 8);
    std::nth_element(sample.begin(), sample.begin() + at, sample.end(), std::greater<float>());
    const float thr = sample[at];
#pragma omp parallel num_threads(threads)
    {
        auto & mine = local[omp_get_thread_num()];
        mine.clear();
#pragma omp for schedule(static)
        for (int i = 0; i < V; ++i) if (z[i] >= thr) mine.push_back(i);
    }
    kept.clear();
    for (auto & l : local) kept.insert(kept.end(), l.begin(), l.end());
    auto cmp = [&](int x, int y) { return z[x] > z[y] || (z[x] == z[y] && x < y); };
    if (static_cast<int>(kept.size()) < k) {
        kept = all;
    }
    std::nth_element(kept.begin(), kept.begin() + (k - 1), kept.end(), cmp);
    kept.resize(k);
}

Stat stat(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return {v[v.size() / 10], v[v.size() / 2], v[v.size() * 9 / 10]};
}

}  // namespace

int main(int argc, char ** argv) {
    int threads = 6, reps = 200;
    std::string json;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--threads") threads = std::stoi(argv[i + 1]);
        else if (a == "--reps") reps = std::stoi(argv[i + 1]);
        else if (a == "--json") json = argv[i + 1];
    }
    ggml_cpu_init();
    const auto * q6 = ggml_get_type_traits_cpu(GGML_TYPE_Q6_K);
    const auto * q8 = ggml_get_type_traits_cpu(q6->vec_dot_type);
    const size_t row = ggml_row_size(GGML_TYPE_Q6_K, D);
    std::mt19937_64 rng(20261104);
    // real Q6_K rows: quantized once from noise, then repeated to fill the head
    std::vector<float> f(D);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<uint8_t> head(row * V);
    for (int r = 0; r < 64; ++r) {
        for (float & x : f) x = nd(rng);
        q6->from_float(f.data(), head.data() + r * row, D);
    }
    for (int r = 64; r < V; ++r) std::memcpy(head.data() + r * row, head.data() + (r % 64) * row, row);
    std::vector<uint8_t> act(ggml_row_size(q6->vec_dot_type, D));
    for (float & x : f) x = nd(rng) * 50.0f;
    q8->from_float(f.data(), act.data(), D);
    std::vector<uint8_t> flush(size_t(1) << 30, 1);
    volatile uint64_t sink = 0;
    auto cold = [&]() {
        uint64_t s = 0;
#pragma omp parallel for reduction(+ : s) num_threads(threads)
        for (int64_t i = 0; i < static_cast<int64_t>(flush.size() / 8); ++i) s += reinterpret_cast<const uint64_t *>(flush.data())[i];
        sink = sink + s;
    };
    std::vector<float> logits(V);
    std::normal_distribution<float> nl(0.0f, 3.0f);
    std::vector<int> order(V), ids;
    std::vector<float> out(4096);
    printf("CONFIG v=%d d=%d row_bytes=%zu threads=%d reps=%d\n", V, D, row, threads, reps);
    std::string body;
    std::vector<float> sample;
    std::vector<std::vector<int>> local(threads);
    std::vector<int> kept, all(V);
    std::iota(all.begin(), all.end(), 0);
    for (int k : KS) {
        std::vector<double> sel, fast, gat;
        int same = 0;
        for (int rep = 0; rep < reps; ++rep) {
            for (float & x : logits) x = nl(rng);
            std::iota(order.begin(), order.end(), 0);
            cold();
            double t0 = now_ms();
            std::nth_element(order.begin(), order.begin() + (k - 1), order.end(), [&](int x, int y) {
                return logits[x] > logits[y] || (logits[x] == logits[y] && x < y);
            });
            sel.push_back(now_ms() - t0);
            ids.assign(order.begin(), order.begin() + k);
            cold();
            t0 = now_ms();
            select_fast(logits, k, threads, sample, local, kept, all);
            fast.push_back(now_ms() - t0);
            std::vector<int> a = ids, b = kept;
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            same += a == b;
            cold();
            t0 = now_ms();
#pragma omp parallel for num_threads(threads)
            for (int i = 0; i < k; ++i) q6->vec_dot(D, &out[i], 0, head.data() + static_cast<size_t>(ids[i]) * row, 0, act.data(), 0, 1);
            gat.push_back(now_ms() - t0);
        }
        const Stat a = stat(sel), b = stat(gat), c = stat(fast);
        const double gbs = k * static_cast<double>(row) / 1e6 / b.p50;
        printf("CELL k=%d nth_element_ms p10=%.3f p50=%.3f p90=%.3f select_ms p10=%.3f p50=%.3f p90=%.3f same_set=%d/%d "
               "gather_ms p10=%.3f p50=%.3f p90=%.3f gather_gb_s=%.2f\n", k, a.p10, a.p50, a.p90, c.p10, c.p50, c.p90, same, reps,
               b.p10, b.p50, b.p90, gbs);
        char line[400];
        snprintf(line, sizeof line, "%s{\"k\":%d,\"nth_element_ms\":[%.4f,%.4f,%.4f],\"select_ms\":[%.4f,%.4f,%.4f],\"same_set\":%d,"
                 "\"gather_ms\":[%.4f,%.4f,%.4f],\"gather_gb_s\":%.3f}",
                 body.empty() ? "" : ",", k, a.p10, a.p50, a.p90, c.p10, c.p50, c.p90, same, b.p10, b.p50, b.p90, gbs);
        body += line;
    }
    if (!json.empty()) {
        std::ofstream js(json);
        js << "{\"threads\":" << threads << ",\"reps\":" << reps << ",\"row_bytes\":" << row << ",\"cells\":[" << body << "]}\n";
    }
    return sink == 12345 ? 1 : 0;
}
