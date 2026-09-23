// Pinned llama.cpp reference trace apparatus for STRAT-01 engine rung 2A.
// This program intentionally contains no phase60 engine code and does not
// adjudicate C-engine parity.  It creates the independently checkable
// reference side required by the frozen Rung-2A protocol.
// STRAT01_RUNG2B and STRAT01_RUNG2C extend only the requested callback
// boundary; the default build and every historical artifact remain unchanged.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::array<llama_token, 8> kTokens = {1, 72, 14, 14129, 14, 2135, 1512, 2015};
constexpr std::array<llama_pos, 8> kPositions = {0, 1, 2, 3, 4, 5, 6, 7};
constexpr const char * kArtifactSha256 = "68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb";
constexpr uintmax_t kArtifactBytes = 6474702976ULL;
#if defined(STRAT01_RUNG2C)
constexpr const char * kSchema = "strat01_engine_rung2c_reference_manifest_v1";
#elif defined(STRAT01_RUNG2B)
constexpr const char * kSchema = "strat01_engine_rung2b_reference_manifest_v1";
#else
constexpr const char * kSchema = "strat01_engine_rung2a_reference_manifest_v1";
#endif
constexpr const char * kLlamaCommit = "5b335f413e4f73b0809c4fe39af894efbcc6a0d2";
constexpr uint32_t kRequestedCtx = 8;
constexpr uint32_t kResolvedCtx = 256;
constexpr uint32_t kBatch = 8;
constexpr uint32_t kUbatch = 8;

struct Error : std::runtime_error { using std::runtime_error::runtime_error; };

void validate_resolved_context_dimensions(uint32_t n_ctx, uint32_t n_batch, uint32_t n_ubatch) {
    if (n_ctx != kResolvedCtx || n_batch != kBatch || n_ubatch != kUbatch) {
        std::ostringstream message;
        message << "VOID: llama.cpp context dimensions differ from frozen requested/resolved contract"
                << " (requested_n_ctx=" << kRequestedCtx
                << ", expected_resolved_n_ctx=" << kResolvedCtx
                << ", actual_n_ctx=" << n_ctx
                << ", expected_n_batch=" << kBatch
                << ", actual_n_batch=" << n_batch
                << ", expected_n_ubatch=" << kUbatch
                << ", actual_n_ubatch=" << n_ubatch << ')';
        throw Error(message.str());
    }
}

// Small self-contained SHA-256 so model identity and payload manifests do not
// depend on an external executable or a platform crypto provider.
class Sha256 {
public:
    Sha256() { reset(); }
    void reset() {
        h_ = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
              0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
        bits_ = 0; used_ = 0;
    }
    void update(const uint8_t * p, size_t n) {
        bits_ += static_cast<uint64_t>(n) * 8;
        while (n) {
            const size_t take = std::min(n, block_.size() - used_);
            std::memcpy(block_.data() + used_, p, take);
            used_ += take; p += take; n -= take;
            if (used_ == block_.size()) { transform(block_.data()); used_ = 0; }
        }
    }
    std::string finish_hex() {
        const uint64_t original_bits = bits_;
        const uint8_t one = 0x80;
        update(&one, 1);
        const uint8_t zero = 0;
        while (used_ != 56) update(&zero, 1);
        uint8_t len[8];
        for (int i = 0; i != 8; ++i) len[7 - i] = static_cast<uint8_t>(original_bits >> (8 * i));
        update(len, sizeof(len));
        std::ostringstream out;
        for (uint32_t v : h_) for (int i = 3; i >= 0; --i)
            out << std::hex << std::setw(2) << std::setfill('0') << ((v >> (8 * i)) & 0xffU);
        return out.str();
    }
private:
    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    static uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
    static uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
    static uint32_t bs0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
    static uint32_t bs1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
    static uint32_t ss0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
    static uint32_t ss1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }
    void transform(const uint8_t * p) {
        static constexpr uint32_t k[64] = {
            0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
            0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
            0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
            0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
            0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
            0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
            0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
            0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U};
        uint32_t w[64];
        for (int i = 0; i != 16; ++i) w[i] = (uint32_t(p[4*i]) << 24) | (uint32_t(p[4*i+1]) << 16) | (uint32_t(p[4*i+2]) << 8) | p[4*i+3];
        for (int i = 16; i != 64; ++i) w[i] = ss1(w[i-2]) + w[i-7] + ss0(w[i-15]) + w[i-16];
        uint32_t a=h_[0],b=h_[1],c=h_[2],d=h_[3],e=h_[4],f=h_[5],g=h_[6],hh=h_[7];
        for (int i = 0; i != 64; ++i) { uint32_t t1=hh+bs1(e)+ch(e,f,g)+k[i]+w[i], t2=bs0(a)+maj(a,b,c); hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2; }
        h_[0]+=a;h_[1]+=b;h_[2]+=c;h_[3]+=d;h_[4]+=e;h_[5]+=f;h_[6]+=g;h_[7]+=hh;
    }
    std::array<uint32_t, 8> h_{}; std::array<uint8_t, 64> block_{}; uint64_t bits_ = 0; size_t used_ = 0;
};

std::string sha256_bytes(const std::vector<uint8_t> & bytes) { Sha256 h; h.update(bytes.data(), bytes.size()); return h.finish_hex(); }
std::string sha256_file(const fs::path & path) {
    std::ifstream in(path, std::ios::binary); if (!in) throw Error("VOID: cannot open model for SHA-256");
    // A 1 MiB automatic array exhausts the default 1 MiB Windows stack before
    // model loading.  Keep the streaming buffer on the heap; this identity
    // gate must work under the same executable stack settings as production.
    std::vector<uint8_t> buf(1 << 20); Sha256 h;
    while (in) { in.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(buf.size())); const auto n=in.gcount(); if (n > 0) h.update(buf.data(), static_cast<size_t>(n)); }
    if (!in.eof()) throw Error("VOID: cannot read model for SHA-256");
    return h.finish_hex();
}

bool host_is_little_endian() { const uint16_t x = 1; return *reinterpret_cast<const uint8_t *>(&x) == 1; }
void append_le_u32(std::vector<uint8_t> & out, uint32_t v) { for (int i=0;i<4;++i) out.push_back(static_cast<uint8_t>(v >> (8*i))); }
std::vector<uint8_t> encode_f32_le(const std::vector<float> & values) {
    if (!host_is_little_endian()) throw Error("VOID: canonical float codec requires an explicit big-endian implementation");
    std::vector<uint8_t> out; out.reserve(values.size()*4);
    for (float x : values) { uint32_t u; std::memcpy(&u, &x, sizeof(u)); append_le_u32(out, u); }
    return out;
}

std::string json_quote(std::string_view in) {
    std::ostringstream out; out << '"';
    for (unsigned char c : in) {
        switch (c) { case '\\': out << "\\\\"; break; case '"': out << "\\\""; break; case '\n': out << "\\n"; break; case '\r': out << "\\r"; break; case '\t': out << "\\t"; break;
        default: if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec; else out << char(c); }
    }
    return out << '"', out.str();
}
bool simple_file_component(std::string_view s) {
    return !s.empty() && s.find("..") == std::string_view::npos && s.find_first_of("\\/:\0") == std::string_view::npos;
}

enum class Arm { Prefill8, Cached7p1, All };
struct Cli { bool self_test=false; fs::path model, out_dir; Arm arm=Arm::All; };
std::optional<Arm> parse_arm(std::string_view text) {
    if (text == "prefill8") return Arm::Prefill8;
    if (text == "cached7p1") return Arm::Cached7p1;
    return std::nullopt;
}
Cli parse_cli(int argc, char ** argv) {
    Cli cli; int i=1;
    if (argc == 2 && std::string_view(argv[1]) == "--self-test") { cli.self_test = true; return cli; }
    while (i < argc) {
        const std::string_view a(argv[i++]);
        if (a == "--model" && i < argc) cli.model = argv[i++];
        else if (a == "--out-dir" && i < argc) cli.out_dir = argv[i++];
        else if (a == "--arm" && i < argc) { auto arm=parse_arm(argv[i++]); if (!arm) throw Error("VOID: --arm must be prefill8 or cached7p1"); cli.arm=*arm; }
        else if (a == "--all") { if (cli.arm != Arm::All) throw Error("VOID: --all and --arm are mutually exclusive"); cli.arm=Arm::All; }
        else throw Error("VOID: malformed CLI (expected --model PATH --out-dir PATH [--all|--arm prefill8|cached7p1])");
    }
    if (cli.model.empty() || cli.out_dir.empty()) throw Error("VOID: --model and --out-dir are required");
    if (!fs::is_regular_file(cli.model)) throw Error("VOID: model path is not a regular file");
    if (fs::exists(cli.out_dir)) throw Error("VOID: output directory already exists; refusing overwrite");
    return cli;
}

struct Event {
    std::string phase, name, op, type; int ordinal=0; std::array<int64_t,4> shape{}; int rank=0;
    std::vector<float> values; std::vector<int32_t> integer_values;
};
#if defined(STRAT01_RUNG2C)
constexpr std::array<const char *, 33> kNames = {
    "Kcur-0", "l_out-0",
    "attn_norm-1", "q-1", "kv_cmpr_pe-1", "k_pe-1", "kv_cmpr-1", "q_pe-1",
    "q_nope_absorbed_perm-1", "Qcur-1", "Kcur-1", "Vcur-1", "kqv_out-1", "ffn_inp-1",
    "ffn_norm-1", "ffn_moe_logits-1", "ffn_moe_probs-1", "ffn_moe_probs_biased-1",
    "ffn_moe_topk-1", "ffn_moe_weights-1", "ffn_moe_weights_norm-1",
    "ffn_moe_up-1", "ffn_moe_gate-1", "ffn_moe_swiglu-1", "ffn_moe_down-1",
    "ffn_moe_weighted-1", "ffn_moe_out-1", "ffn_up-1", "ffn_gate-1", "ffn_swiglu-1",
    "ffn_shexp-1", "ffn_out-1", "l_out-1"};
#elif defined(STRAT01_RUNG2B)
constexpr std::array<const char *, 7> kNames = {"ffn_inp-0", "ffn_norm-0", "ffn_up-0", "ffn_gate-0", "ffn_swiglu-0", "ffn_out-0", "l_out-0"};
#else
constexpr std::array<const char *, 16> kNames = {"attn_norm-0", "q-0", "kv_cmpr_pe-0", "k_pe-0", "kv_cmpr-0", "q_pe-0", "q_nope_absorbed_perm-0", "Qcur-0", "Kcur-0", "Vcur-0", "kq-0", "kq_soft_max-0", "kqv-0", "kqv_mla-0", "kqv_out-0", "ffn_inp-0"};
#endif
bool wanted_name(std::string_view name) { return std::find_if(kNames.begin(), kNames.end(), [&](const char * p){ return name == p; }) != kNames.end(); }
int protocol_rank(std::string_view name) {
    if (name == "q-0" || name == "k_pe-0" || name == "q_pe-0" || name == "q_nope_absorbed_perm-0" || name == "Qcur-0" || name == "Kcur-0" || name == "Vcur-0" || name == "kq-0" || name == "kq_soft_max-0" || name == "kqv-0" || name == "kqv_mla-0" ||
        name == "q-1" || name == "k_pe-1" || name == "q_pe-1" || name == "q_nope_absorbed_perm-1" || name == "Qcur-1" || name == "Kcur-1" || name == "Vcur-1" ||
        name == "ffn_moe_up-1" || name == "ffn_moe_gate-1" || name == "ffn_moe_swiglu-1" || name == "ffn_moe_down-1" || name == "ffn_moe_weighted-1") return 3;
    if (wanted_name(name)) return 2;
    throw Error("VOID: callback name lacks a frozen logical rank");
}

std::vector<float> tensor_f32(const ggml_tensor * t) {
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) throw Error("VOID: required callback tensor has unsupported non-float type");
    const size_t nbytes = ggml_nbytes(t); std::vector<uint8_t> raw(nbytes); ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    int rank=0; for (int i=0;i<GGML_MAX_DIMS;++i) if (t->ne[i] > 1 || i == 0) rank=i+1;
    size_t count=1; for (int i=0;i<rank;++i) { if (t->ne[i] <= 0) throw Error("VOID: callback tensor has invalid shape"); count *= static_cast<size_t>(t->ne[i]); }
    std::vector<float> out; out.reserve(count);
    for (int64_t i3=0;i3<(rank>3?t->ne[3]:1);++i3) for (int64_t i2=0;i2<(rank>2?t->ne[2]:1);++i2)
        for (int64_t i1=0;i1<(rank>1?t->ne[1]:1);++i1) for (int64_t i0=0;i0<t->ne[0];++i0) {
            const size_t off=static_cast<size_t>(i0*t->nb[0]+i1*t->nb[1]+i2*t->nb[2]+i3*t->nb[3]);
            if (t->type == GGML_TYPE_F32) { if (off+4>raw.size()) throw Error("VOID: callback tensor stride exceeds dump"); float x; std::memcpy(&x,raw.data()+off,4); out.push_back(x); }
            else { if (off+2>raw.size()) throw Error("VOID: callback tensor stride exceeds dump"); ggml_fp16_t h; std::memcpy(&h,raw.data()+off,2); out.push_back(ggml_fp16_to_fp32(h)); }
        }
    return out;
}

std::vector<int32_t> tensor_i32(const ggml_tensor * t) {
    if (t->type != GGML_TYPE_I32) throw Error("VOID: required integer callback tensor is not I32");
    const size_t nbytes=ggml_nbytes(t); std::vector<uint8_t> raw(nbytes); ggml_backend_tensor_get(t,raw.data(),0,raw.size());
    int rank=0; for(int i=0;i<GGML_MAX_DIMS;++i) if(t->ne[i]>1||i==0) rank=i+1;
    std::vector<int32_t> out; size_t count=1;for(int i=0;i<rank;++i)count*=static_cast<size_t>(t->ne[i]);out.reserve(count);
    for(int64_t i3=0;i3<(rank>3?t->ne[3]:1);++i3)for(int64_t i2=0;i2<(rank>2?t->ne[2]:1);++i2)for(int64_t i1=0;i1<(rank>1?t->ne[1]:1);++i1)for(int64_t i0=0;i0<t->ne[0];++i0){const size_t off=static_cast<size_t>(i0*t->nb[0]+i1*t->nb[1]+i2*t->nb[2]+i3*t->nb[3]);if(off+4>raw.size())throw Error("VOID: I32 callback tensor stride exceeds dump");int32_t v;std::memcpy(&v,raw.data()+off,4);out.push_back(v);}
    return out;
}

struct Collector {
    std::string phase; std::map<std::string,int> next_ordinal; std::vector<Event> events;
    void begin(std::string next_phase) { phase=std::move(next_phase); next_ordinal.clear(); events.clear(); }
    static bool callback(ggml_tensor * t, bool ask, void * user) {
        auto & self=*static_cast<Collector *>(user); const std::string name(t->name);
        if (ask) return wanted_name(name); // Request only block-0 protocol candidates.
        if (!wanted_name(name)) return true;
        Event e; e.phase=self.phase; e.name=name; e.ordinal=self.next_ordinal[name]++; e.op=ggml_op_name(t->op); e.type=ggml_type_name(t->type);
        for (int i=0;i<4;++i) e.shape[i]=t->ne[i];
        // ggml tensors do not retain a declared rank once trailing dimensions
        // become singleton.  The frozen protocol rank preserves the token axis
        // for the one-token cached decode instead of silently collapsing it.
        e.rank=protocol_rank(name);
        if((name=="ffn_moe_weights-1"||name=="ffn_moe_weights_norm-1")&&t->ne[0]==1){e.shape[0]=t->ne[1];e.shape[1]=t->ne[2];e.shape[2]=1;}
        if(t->type==GGML_TYPE_I32)e.integer_values=tensor_i32(t);else e.values=tensor_f32(t);
        self.events.push_back(std::move(e)); return true;
    }
};

const Event & one(const std::vector<Event> & events, std::string_view name, std::string_view required_op = {}) {
    const Event * found=nullptr;
    for (const Event & e : events) if (e.name==name && (required_op.empty() || e.op==required_op)) {
        if (found) throw Error("VOID: ambiguous required callback selection for " + std::string(name)); found=&e;
    }
    if (!found) throw Error("VOID: missing required callback selection for " + std::string(name)); return *found;
}
const Event & last(const std::vector<Event> & events, std::string_view name, std::string_view required_op) {
    const Event * found=nullptr;
    for (const Event & e : events) if (e.name==name && e.op==required_op && (!found || e.ordinal>found->ordinal)) found=&e;
    if (!found) throw Error("VOID: missing required post-operation callback selection for " + std::string(name)); return *found;
}
std::map<std::string,const Event *> select_logical(const std::vector<Event> & e) {
    // Selection is tied to the clean pinned source call order, never to an
    // unqualified callback name.  q after RESHAPE is explicitly resolved.
#if defined(STRAT01_RUNG2C)
    return {
        {"Kcur-0", &one(e,"Kcur-0")}, {"l_out-0", &one(e,"l_out-0")},
        {"attn_norm-1", &one(e,"attn_norm-1")}, {"q-1", &one(e,"q-1","RESHAPE")},
        {"kv_cmpr_pe-1", &one(e,"kv_cmpr_pe-1")}, {"k_pe-1", &last(e,"k_pe-1","ROPE")},
        {"kv_cmpr-1", &last(e,"kv_cmpr-1","MUL")}, {"q_pe-1", &last(e,"q_pe-1","ROPE")},
        {"q_nope_absorbed_perm-1", &one(e,"q_nope_absorbed_perm-1")}, {"Qcur-1", &one(e,"Qcur-1")},
        {"Kcur-1", &one(e,"Kcur-1")}, {"Vcur-1", &one(e,"Vcur-1")},
        {"kqv_out-1", &one(e,"kqv_out-1")}, {"ffn_inp-1", &one(e,"ffn_inp-1")},
        {"ffn_norm-1", &one(e,"ffn_norm-1")}, {"ffn_moe_logits-1", &one(e,"ffn_moe_logits-1")},
        {"ffn_moe_probs-1", &one(e,"ffn_moe_probs-1")}, {"ffn_moe_probs_biased-1", &one(e,"ffn_moe_probs_biased-1")},
        {"ffn_moe_topk-1", &one(e,"ffn_moe_topk-1")}, {"ffn_moe_weights-1", &one(e,"ffn_moe_weights-1")},
        {"ffn_moe_weights_norm-1", &one(e,"ffn_moe_weights_norm-1")}, {"ffn_moe_up-1", &one(e,"ffn_moe_up-1")},
        {"ffn_moe_gate-1", &one(e,"ffn_moe_gate-1")}, {"ffn_moe_swiglu-1", &one(e,"ffn_moe_swiglu-1")},
        {"ffn_moe_down-1", &one(e,"ffn_moe_down-1")}, {"ffn_moe_weighted-1", &one(e,"ffn_moe_weighted-1")},
        {"ffn_moe_out-1", &last(e,"ffn_moe_out-1","ADD")}, {"ffn_up-1", &one(e,"ffn_up-1")},
        {"ffn_gate-1", &one(e,"ffn_gate-1")}, {"ffn_swiglu-1", &one(e,"ffn_swiglu-1")},
        {"ffn_shexp-1", &one(e,"ffn_shexp-1")}, {"ffn_out-1", &one(e,"ffn_out-1")},
        {"l_out-1", &one(e,"l_out-1")}};
#elif defined(STRAT01_RUNG2B)
    return {{"ffn_inp-0", &one(e,"ffn_inp-0")},
            {"ffn_norm-0", &one(e,"ffn_norm-0")},
            {"ffn_up-0", &one(e,"ffn_up-0")},
            {"ffn_gate-0", &one(e,"ffn_gate-0")},
            {"ffn_swiglu-0", &one(e,"ffn_swiglu-0")},
            {"ffn_out-0", &one(e,"ffn_out-0")},
            {"l_out-0", &one(e,"l_out-0")}};
#else
    std::map<std::string,const Event *> result = {{"attn_norm-0", &one(e,"attn_norm-0")}, {"q-0", &one(e,"q-0","RESHAPE")},
            {"kv_cmpr_pe-0", &one(e,"kv_cmpr_pe-0")}, {"k_pe-0", &last(e,"k_pe-0","ROPE")},
            {"kv_cmpr-0", &last(e,"kv_cmpr-0","MUL")}, {"q_pe-0", &last(e,"q_pe-0","ROPE")},
            {"q_nope_absorbed_perm-0", &one(e,"q_nope_absorbed_perm-0")}, {"Qcur-0", &one(e,"Qcur-0")},
            {"Kcur-0", &one(e,"Kcur-0")}, {"Vcur-0", &one(e,"Vcur-0")},
            {"kq-0", &one(e,"kq-0")}, {"kq_soft_max-0", &one(e,"kq_soft_max-0")},
            {"kqv-0", &one(e,"kqv-0")}, {"kqv_mla-0", &one(e,"kqv_mla-0")},
            {"kqv_out-0", &one(e,"kqv_out-0")}, {"ffn_inp-0", &one(e,"ffn_inp-0")}};
    return result;
#endif
}

Event stitch_cached_event(const Event & prefix, const Event & final, std::string_view logical) {
    if (prefix.name != final.name || prefix.op != final.op || prefix.type != final.type || prefix.rank != final.rank || prefix.rank < 2) {
        throw Error("VOID: cached7p1 callback identity differs between prefix and final for " + std::string(logical));
    }
    const int token_axis = prefix.rank - 1;
    for (int i = 0; i < prefix.rank; ++i) {
        const int64_t expected = i == token_axis ? 7 : prefix.shape[i];
        const int64_t actual = i == token_axis ? 1 : final.shape[i];
        if (prefix.shape[i] != expected || final.shape[i] != actual || (i != token_axis && prefix.shape[i] != final.shape[i])) {
            throw Error("VOID: cached7p1 callback shape cannot be composed for " + std::string(logical));
        }
    }
    size_t per_token = 1;
    for (int i = 0; i < token_axis; ++i) per_token *= static_cast<size_t>(prefix.shape[i]);
    Event combined = prefix;
    combined.phase = "cached7p1_composed";
    combined.ordinal = -1; // no invented callback occurrence: the manifest records both sources below
    combined.shape[token_axis] = 8;
    if(prefix.type=="I32"){
        if(prefix.integer_values.size()!=7*per_token||final.integer_values.size()!=per_token)throw Error("VOID: cached7p1 I32 payload size cannot be composed for "+std::string(logical));
        combined.integer_values.insert(combined.integer_values.end(),final.integer_values.begin(),final.integer_values.end());
    }else{
        if(prefix.values.size()!=7*per_token||final.values.size()!=per_token)throw Error("VOID: cached7p1 float payload size cannot be composed for "+std::string(logical));
        combined.values.insert(combined.values.end(),final.values.begin(),final.values.end());
    }
    return combined;
}

void validate_fixed_identity(const llama_token * tokens, const llama_pos * positions, size_t n) {
    if (n != kTokens.size()) throw Error("VOID: protocol token/position count differs from fixed eight-token input");
    for (size_t i=0;i<n;++i) if (tokens[i] != kTokens[i] || positions[i] != kPositions[i]) throw Error("VOID: token ID or position differs from frozen protocol");
}
void validate_fixed_slice(const llama_token * tokens, const llama_pos * positions, size_t n, size_t offset) {
    if (offset + n > kTokens.size()) throw Error("VOID: protocol token/position slice is out of bounds");
    for (size_t i=0;i<n;++i) if (tokens[i] != kTokens[offset+i] || positions[i] != kPositions[offset+i]) throw Error("VOID: token ID or position differs from frozen protocol");
}
void decode_exact(llama_context * ctx, Collector & collector, std::string phase, const llama_token * tokens, const llama_pos * positions, int n, size_t protocol_offset) {
    // The public batch API is used so positions cannot silently be inferred.
    validate_fixed_slice(tokens, positions, static_cast<size_t>(n), protocol_offset);
    llama_batch batch=llama_batch_init(n,0,1); batch.n_tokens=n;
    for (int i=0;i<n;++i) { batch.token[i]=tokens[i]; batch.pos[i]=positions[i]; batch.n_seq_id[i]=1; batch.seq_id[i][0]=0; batch.logits[i]=0; }
    collector.begin(std::move(phase));
    if (llama_decode(ctx,batch) != 0) { llama_batch_free(batch); throw Error("VOID: llama_decode failed"); }
    llama_batch_free(batch);
}

struct Trace { std::string arm; std::vector<Event> prefix_events, final_events; };
std::map<std::string,Event> payload_events(const Trace & trace) {
    std::map<std::string,Event> result;
    const auto final = select_logical(trace.final_events);
    if (trace.arm != "cached7p1") {
        for (const auto & [logical,event] : final) result.emplace(logical,*event);
        return result;
    }
    const auto prefix = select_logical(trace.prefix_events);
    for (const char * logical : kNames) {
        result.emplace(logical,stitch_cached_event(*prefix.at(logical),*final.at(logical),logical));
    }
    return result;
}
Trace run_arm(llama_context * ctx, Collector & collector, Arm arm) {
    llama_memory_clear(llama_get_memory(ctx), true);
    if (arm == Arm::Prefill8) {
        validate_fixed_identity(kTokens.data(),kPositions.data(),kTokens.size());
        decode_exact(ctx,collector,"prefill8",kTokens.data(),kPositions.data(),8,0);
        return {"prefill8",{},collector.events};
    }
    if (arm != Arm::Cached7p1) throw Error("VOID: internal non-arm passed to run_arm");
    decode_exact(ctx,collector,"cached7p1_prefix",kTokens.data(),kPositions.data(),7,0);
    const auto prefix=collector.events;
    // The one-token call keeps the same context and memory: this is the cache
    // continuity arm, not a replay or reconstructed prefix.
    decode_exact(ctx,collector,"cached7p1_final",kTokens.data()+7,kPositions.data()+7,1,7);
    return {"cached7p1",prefix,collector.events};
}

template<class T> std::vector<T> token7_slice(const Event & e,const std::vector<T> & values) {
    if (e.rank < 2) return values;
    // Pinned DeepSeek2 callback tensors carry the token dimension last.  A
    // one-token cached final already has index zero; an 8-token prefill uses 7.
    const int64_t n_tok=e.shape[e.rank-1]; if (n_tok < 1) throw Error("VOID: selected tensor lacks token dimension");
    const size_t per=values.size()/static_cast<size_t>(n_tok); if (per*n_tok != values.size()) throw Error("VOID: non-integral token slice");
    const size_t idx=n_tok==8?7:0; return {values.begin()+static_cast<std::ptrdiff_t>(idx*per), values.begin()+static_cast<std::ptrdiff_t>((idx+1)*per)};
}

struct Payload { std::string logical, kind, path, sha256; size_t byte_count=0; };
Payload write_payload(const fs::path & root, const std::string & arm, const std::string & logical, const std::string & kind, const std::vector<float> & values) {
    if (!simple_file_component(arm) || !simple_file_component(logical) || !simple_file_component(kind)) throw Error("VOID: malformed manifest payload component");
    const fs::path relative=fs::path(arm)/(logical+"."+kind+".f32le"); const fs::path path=root/relative;
    fs::create_directories(path.parent_path()); const auto bytes=encode_f32_le(values);
    std::ofstream out(path,std::ios::binary|std::ios::trunc); if (!out) throw Error("VOID: cannot create payload"); out.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size())); if (!out) throw Error("VOID: cannot write payload");
    return {logical,kind,relative.generic_string(),sha256_bytes(bytes),bytes.size()};
}
Payload write_i32_payload(const fs::path & root,const std::string & arm,const std::string & logical,const std::string & kind,const std::vector<int32_t> & values) {
    if(!simple_file_component(arm)||!simple_file_component(logical)||!simple_file_component(kind))throw Error("VOID: malformed I32 payload component");
    const fs::path relative=fs::path(arm)/(logical+"."+kind+".i32le");const fs::path path=root/relative;fs::create_directories(path.parent_path());std::vector<uint8_t> bytes;bytes.reserve(values.size()*4);for(int32_t x:values)append_le_u32(bytes,static_cast<uint32_t>(x));std::ofstream out(path,std::ios::binary|std::ios::trunc);if(!out)throw Error("VOID: cannot create I32 payload");out.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));if(!out)throw Error("VOID: cannot write I32 payload");return{logical,kind,relative.generic_string(),sha256_bytes(bytes),bytes.size()};
}
Payload write_cache_f16_roundtrip_payload(const fs::path & root,const std::string & arm,const std::string & logical,const std::string & kind,const std::vector<float> & values) {
    std::vector<float> rounded;rounded.reserve(values.size());for(float x:values)rounded.push_back(ggml_fp16_to_fp32(ggml_fp32_to_fp16(x)));return write_payload(root,arm,logical,kind,rounded);
}
void write_shape(std::ostream & out, const Event & e) { out << '['; for(int i=0;i<e.rank;++i) { if(i) out << ','; out << e.shape[i]; } out << ']'; }
void write_event(std::ostream & out, const Event & e) {
    out << "{\"phase\":" << json_quote(e.phase) << ",\"name\":" << json_quote(e.name) << ",\"ordinal\":" << e.ordinal << ",\"op\":" << json_quote(e.op) << ",\"type\":" << json_quote(e.type) << ",\"shape\":"; write_shape(out,e); out << '}';
}
void write_trace_json(const fs::path & root, const Trace & trace) {
    const fs::path arm_dir=root/trace.arm; fs::create_directories(arm_dir);
    const auto selected=payload_events(trace);
    std::vector<Payload> payloads;
    for (const auto & [logical,event] : selected) {
        if(event.type=="I32") { payloads.push_back(write_i32_payload(root,trace.arm,logical,"full",event.integer_values));payloads.push_back(write_i32_payload(root,trace.arm,logical,"token7",token7_slice(event,event.integer_values))); }
        else { payloads.push_back(write_payload(root,trace.arm,logical,"full",event.values));payloads.push_back(write_payload(root,trace.arm,logical,"token7",token7_slice(event,event.values))); }
    }
#if defined(STRAT01_RUNG2C)
    for(const char *logical:{"Kcur-0","Kcur-1"}){
        const Event &full=selected.at(logical);payloads.push_back(write_cache_f16_roundtrip_payload(root,trace.arm,logical,"cache_f16_roundtrip",full.values));
        if(trace.arm=="cached7p1"){const Event &prefix=one(trace.prefix_events,logical);payloads.push_back(write_cache_f16_roundtrip_payload(root,trace.arm,logical,"prefix_cache_f16_roundtrip",prefix.values));}
    }
#elif !defined(STRAT01_RUNG2B)
    // In cached7p1, prefix Kcur is the logical cache witness.  This does not
    // claim access to llama.cpp's private physical cache bytes.
    if (trace.arm=="cached7p1") { const Event & k=one(trace.prefix_events,"Kcur-0"); payloads.push_back(write_payload(root,trace.arm,"Kcur-0","prefix_logical_rows",k.values)); }
#endif
    const fs::path manifest=arm_dir/"manifest.json"; std::ofstream out(manifest,std::ios::binary|std::ios::trunc); if(!out) throw Error("VOID: cannot create arm manifest");
#if defined(STRAT01_RUNG2C)
    out << "{\n\"schema\":" << json_quote(kSchema) << ",\n\"arm\":" << json_quote(trace.arm) << ",\n\"logical_cache_contract\":{\"extraction\":\"Kcur callback values rounded through F16 by producer\",\"physical_bytes_claimed\":false,\"storage_type\":\"F16\",\"row_length\":576,\"layers\":[0,1],\"separate_v_cache\":false},\n\"callback_records\":[";
#elif defined(STRAT01_RUNG2B)
    out << "{\n\"schema\":" << json_quote(kSchema) << ",\n\"arm\":" << json_quote(trace.arm) << ",\n\"logical_cache_contract\":{\"inherited_rung2a_not_remeasured\":true},\n\"callback_records\":[";
#else
    out << "{\n\"schema\":" << json_quote(kSchema) << ",\n\"arm\":" << json_quote(trace.arm) << ",\n\"logical_cache_contract\":{\"extraction\":\"logical_Kcur_callback_values_only\",\"physical_bytes_claimed\":false,\"storage_type\":\"F16\",\"row_length\":576,\"separate_v_cache\":false,\"checkpoints\":[";
    if (trace.arm == "cached7p1") {
        out << "{\"phase\":\"cached7p1_prefix\",\"occupied_positions\":[0,1,2,3,4,5,6],\"logical_rows_payload\":\"Kcur-0.prefix_logical_rows.f32le\"},";
    }
    out << "{\"phase\":" << json_quote(trace.arm == "cached7p1" ? "cached7p1_final" : "prefill8") << ",\"occupied_positions\":[0,1,2,3,4,5,6,7],\"token7_position\":7}],\"note\":\"logical values only; no physical cache-byte extraction is claimed\"},\n\"callback_records\":[";
#endif
    bool first=true; for(const Event & e:trace.prefix_events){if(!first)out<<',';first=false;write_event(out,e);} for(const Event & e:trace.final_events){if(!first)out<<',';first=false;write_event(out,e);} out << "],\n\"logical_selection\":{\n";
    const auto final_selected=select_logical(trace.final_events);
    const auto prefix_selected=trace.arm=="cached7p1"?select_logical(trace.prefix_events):std::map<std::string,const Event *>{};
    first=true; for(const auto &[logical,event]:selected){
        if(!first)out<<",\n";first=false;out<<json_quote(logical)<<":{\"logical_shape\":";write_shape(out,event);
        if(trace.arm=="cached7p1"){
            out<<",\"composition\":\"prefix7_then_final1\",\"prefix_source\":";write_event(out,*prefix_selected.at(logical));out<<",\"final_source\":";write_event(out,*final_selected.at(logical));
        }else{
            out<<",\"composition\":\"single_prefill8_callback\",\"source\":";write_event(out,*final_selected.at(logical));
        }
        out<<'}';
    } out << "\n},\n\"payloads\":[";
    for(size_t i=0;i<payloads.size();++i){const auto&p=payloads[i];if(i)out<<',';out<<"{\"logical\":"<<json_quote(p.logical)<<",\"kind\":"<<json_quote(p.kind)<<",\"path\":"<<json_quote(p.path)<<",\"byte_count\":"<<p.byte_count<<",\"sha256\":"<<json_quote(p.sha256)<<'}';}
    out << "]\n}\n"; if(!out) throw Error("VOID: cannot write arm manifest");
}
void write_root_manifest(const fs::path & root, const fs::path & model, const std::vector<Trace> & traces) {
    std::ofstream out(root/"manifest.json",std::ios::binary|std::ios::trunc); if(!out) throw Error("VOID: cannot create root manifest");
    out << "{\n\"schema\":"<<json_quote(kSchema)<<",\n\"state\":\"REFERENCE_TRACE_READY_PENDING_C_ENGINE\",\n\"llama_cpp_commit\":"<<json_quote(kLlamaCommit)<<",\n\"model\":{\"path\":"<<json_quote(model.generic_string())<<",\"bytes\":"<<kArtifactBytes<<",\"sha256\":"<<json_quote(kArtifactSha256)<<"},\n\"config\":{\"n_gpu_layers\":0,\"n_threads\":1,\"n_threads_batch\":1,\"requested_n_ctx\":"<<kRequestedCtx<<",\"resolved_n_ctx\":"<<kResolvedCtx<<",\"n_batch\":"<<kBatch<<",\"n_ubatch\":"<<kUbatch<<",\"flash_attn\":\"disabled\",\"offload_kqv\":false,\"op_offload\":false,\"type_k\":\"F16\",\"type_v\":\"F16\"},\n\"fixed_tokens\":[";
    for(size_t i=0;i<kTokens.size();++i){if(i)out<<',';out<<kTokens[i];}out<<"],\n\"fixed_positions\":[";for(size_t i=0;i<kPositions.size();++i){if(i)out<<',';out<<kPositions[i];}out<<"],\n\"arms\":[";
    for(size_t i=0;i<traces.size();++i){if(i)out<<',';out<<"{\"arm\":"<<json_quote(traces[i].arm)<<",\"manifest\":"<<json_quote((fs::path(traces[i].arm)/"manifest.json").generic_string())<<"}";}out<<"]\n}\n";
}

void require_accepted_model(const fs::path & model) { if(fs::file_size(model)!=kArtifactBytes) throw Error("VOID: artifact size differs from frozen accepted donor"); if(sha256_file(model)!=kArtifactSha256) throw Error("VOID: artifact SHA-256 differs from frozen accepted donor"); }
void run_production(const Cli & cli) {
    require_accepted_model(cli.model); fs::create_directories(cli.out_dir);
    llama_backend_init();
    try {
        llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=0;
        llama_model * model=llama_model_load_from_file(cli.model.string().c_str(),mp); if(!model) throw Error("VOID: llama.cpp could not load accepted model");
        Collector collector; llama_context_params cp=llama_context_default_params(); cp.n_ctx=kRequestedCtx;cp.n_batch=kBatch;cp.n_ubatch=kUbatch;cp.n_threads=1;cp.n_threads_batch=1;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=false;cp.op_offload=false;cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.cb_eval=&Collector::callback;cp.cb_eval_user_data=&collector;
        llama_context * ctx=llama_init_from_model(model,cp); if(!ctx) {llama_model_free(model);throw Error("VOID: llama.cpp could not create frozen CPU context");}
        try { validate_resolved_context_dimensions(llama_n_ctx(ctx),llama_n_batch(ctx),llama_n_ubatch(ctx)); }
        catch (...) { llama_free(ctx);llama_model_free(model);throw; }
        std::vector<Trace> traces; if(cli.arm==Arm::All||cli.arm==Arm::Prefill8) traces.push_back(run_arm(ctx,collector,Arm::Prefill8)); if(cli.arm==Arm::All||cli.arm==Arm::Cached7p1) traces.push_back(run_arm(ctx,collector,Arm::Cached7p1));
        for(const Trace&t:traces)write_trace_json(cli.out_dir,t); write_root_manifest(cli.out_dir,cli.model,traces); llama_free(ctx);llama_model_free(model);
    } catch (...) { llama_backend_free(); throw; }
    llama_backend_free();
}

bool self_tests() {
    bool ok=true; auto check=[&](bool x,const char * name){if(!x){std::cerr<<"self-test failed: "<<name<<'\n';ok=false;}};
    check(host_is_little_endian(),"little endian host");
    const std::vector<float> vals={1.0f,-2.5f}; const auto bytes=encode_f32_le(vals); check(bytes.size()==8&&bytes[0]==0&&bytes[1]==0&&bytes[2]==0x80&&bytes[3]==0x3f,"canonical f32le");
    check(sha256_bytes(bytes)=="48943f7a0ea247f8e3c9386d0c5822fe181d323a9289980426638cc4e72a43e1","canonical float SHA-256");
    {
        std::ostringstream leaf; leaf << "strat01-rung2a-sha-selftest-" << reinterpret_cast<uintptr_t>(&ok) << ".bin";
        const fs::path fixture=fs::temp_directory_path()/leaf.str();
        try {
            { std::ofstream out(fixture,std::ios::binary|std::ios::trunc); out << "abc"; if(!out) throw Error("self-test fixture write failed"); }
            check(sha256_file(fixture)=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","streaming file SHA-256");
            fs::remove(fixture);
        } catch (...) { std::error_code ignored; fs::remove(fixture,ignored); throw; }
    }
    try { auto bad=kTokens; bad[3]++; validate_fixed_identity(bad.data(),kPositions.data(),bad.size()); check(false,"token identity refusal"); } catch(const Error&) {}
    try { auto bad=kPositions; bad[4]++; validate_fixed_identity(kTokens.data(),bad.data(),bad.size()); check(false,"position identity refusal"); } catch(const Error&) {}
    try { validate_resolved_context_dimensions(kResolvedCtx,kBatch,kUbatch); } catch(const Error&) { check(false,"resolved context acceptance"); }
    try { validate_resolved_context_dimensions(kRequestedCtx,kBatch,kUbatch); check(false,"unpadded context refusal"); } catch(const Error&) {}
    try { validate_resolved_context_dimensions(kResolvedCtx,kBatch-1,kUbatch); check(false,"batch mismatch refusal"); } catch(const Error&) {}
    std::vector<Event> fixture={{"x","q-0","MUL_MAT","F32",0,{1,1,1,1},1,{}},{"x","q-0","RESHAPE","F32",1,{1,1,1,1},1,{}},{"x","k_pe-0","VIEW","F32",0,{1,1,1,1},1,{}},{"x","k_pe-0","ROPE","F32",1,{1,1,1,1},1,{}}};
    check(one(fixture,"q-0","RESHAPE").ordinal==1&&last(fixture,"k_pe-0","ROPE").ordinal==1,"occurrence disambiguation");
    try { (void)one(fixture,"q-0"); check(false,"ambiguous occurrence refusal"); } catch(const Error&) {}
    { Event p{"prefix","q-0","RESHAPE","F32",1,{2,3,7,1},3,std::vector<float>(42,1.0f)}; Event f{"final","q-0","RESHAPE","F32",1,{2,3,1,1},3,std::vector<float>(6,2.0f)}; const Event c=stitch_cached_event(p,f,"q-0"); check(c.shape[2]==8&&c.values.size()==48&&c.values[41]==1.0f&&c.values[42]==2.0f,"cached7p1 payload composition"); }
    { Event p{"prefix","ffn_moe_topk-1","VIEW","I32",0,{4,7,1,1},2,{},std::vector<int32_t>(28,3)};Event f{"final","ffn_moe_topk-1","VIEW","I32",0,{4,1,1,1},2,{},std::vector<int32_t>(4,9)};const Event c=stitch_cached_event(p,f,"ffn_moe_topk-1");const auto t7=token7_slice(c,c.integer_values);check(c.shape[1]==8&&c.integer_values.size()==32&&t7.size()==4&&t7[0]==9,"cached7p1 I32 composition"); }
    { const float x=1.0001f;check(ggml_fp16_to_fp32(ggml_fp32_to_fp16(x))!=x,"cache F16 roundtrip witness"); }
    check(!simple_file_component("../bad")&&!simple_file_component("a/b")&&simple_file_component("Kcur-0"),"malformed manifest component refusal");
    try { char a0[]="x",a1[]="--arm",a2[]="wrong"; char *a[]={a0,a1,a2}; (void)parse_cli(3,a); check(false,"malformed CLI refusal"); } catch(const Error&) {}
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    try { const Cli cli=parse_cli(argc,argv); if(cli.self_test) return self_tests()?0:1; run_production(cli); return 0; }
    catch(const std::exception & e) {
#if defined(STRAT01_RUNG2C)
        std::cerr << "strat01_engine_rung2c_reference: " << e.what() << '\n';
#elif defined(STRAT01_RUNG2B)
        std::cerr << "strat01_engine_rung2b_reference: " << e.what() << '\n';
#else
        std::cerr << "strat01_engine_rung2a_reference: " << e.what() << '\n';
#endif
        return 2;
    }
}
