/* Read-only routed input capture for the pinned source-ID BF16 graph. */
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <string>

static int m298_chunk = -1, m298_batch = -1;
static FILE * m298_stream = nullptr;
static uint64_t m298_rows = 0, m298_bytes = 0;

static void m298_write(const void * data, size_t bytes) {
    if (!m298_stream || fwrite(data, 1, bytes, m298_stream) != bytes) {
        fprintf(stderr, "M298 capture write failed\n"); exit(81);
    }
    m298_bytes += bytes;
    if (m298_bytes > (uint64_t(2) << 30)) {
        fprintf(stderr, "M298 capture exceeds 2 GiB\n"); exit(82);
    }
}

static void m298_capture(const std::string & name, int expert,
                         int slot, int token, int width, const float * x) {
    if (expert != 0 && expert != 32 && expert != 63) return;
    int layer = -1, organ = -1;
    const int layers[] = {1, 13, 25};
    const char * organs[] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
    for (int l : layers) for (int o = 0; o < 3; ++o) {
        if (name == "blk." + std::to_string(l) + "." + organs[o] + ".weight") {
            layer = l; organ = o;
        }
    }
    if (layer < 0) return;
    if (m298_chunk < 0 || m298_batch < 0 || m298_batch >= 4 ||
        slot < 0 || slot >= 4 || token < 0 || token >= 128 ||
        width != (organ == 2 ? 1280 : 1536)) {
        fprintf(stderr, "M298 capture geometry mismatch\n"); exit(83);
    }
    for (int j = 0; j < width; ++j) if (!std::isfinite(x[j])) {
        fprintf(stderr, "M298 nonfinite input\n"); exit(84);
    }
    if (!m298_stream) {
        const char * path = std::getenv("SILICON_M298_CAPTURE");
        if (!path || !*path) { fprintf(stderr,"M298 missing output path\n"); exit(85); }
        FILE * existing = fopen(path, "rb");
        if (existing) { fclose(existing); fprintf(stderr,"M298 output already exists\n"); exit(86); }
        m298_stream = fopen(path, "wb");
        m298_write("M298ACT1", 8);
    }
    const uint32_t meta[] = {uint32_t(layer), uint32_t(organ), uint32_t(expert),
        uint32_t(m298_chunk), uint32_t(m298_batch), uint32_t(slot),
        uint32_t(token), uint32_t(width)};
    m298_write(meta, sizeof(meta));
    m298_write(x, size_t(width) * sizeof(float));
    ++m298_rows;
}

static void m298_finish() {
    if (!m298_stream || !m298_rows) { fprintf(stderr,"M298 empty capture\n"); exit(87); }
    const uint32_t end[] = {UINT32_MAX, 0, 0, 0, 0, 0, 0, 0};
    m298_write(end, sizeof(end));
    m298_write(&m298_rows, sizeof(m298_rows));
    if (fclose(m298_stream) != 0) { fprintf(stderr,"M298 close failed\n"); exit(88); }
    m298_stream = nullptr;
    fprintf(stderr,"M298_CAPTURE_COMPLETE rows=%llu bytes=%llu\n",
        (unsigned long long)m298_rows, (unsigned long long)m298_bytes);
}
