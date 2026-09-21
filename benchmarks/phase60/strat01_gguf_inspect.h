/*
 * STRAT-01 GGUF inspection boundary (header-only on purpose).
 *
 * This is a metadata/layout verifier, not a GGUF loader or inference path.
 * It deliberately keeps only tensor descriptors in memory and streams payload
 * bytes solely for SHA-256 identity verification.
 */
#ifndef STRAT01_GGUF_INSPECT_H
#define STRAT01_GGUF_INSPECT_H

#include <errno.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define STRAT01_FSEEK _fseeki64
#define STRAT01_FTELL _ftelli64
#else
#define STRAT01_FSEEK fseeko
#define STRAT01_FTELL ftello
#endif

#define STRAT01_GGUF_MAGIC UINT32_C(0x46554747) /* little-endian "GGUF" */
#define STRAT01_GGUF_VERSION 3U
#define STRAT01_DEFAULT_ALIGNMENT UINT64_C(32)
#define STRAT01_EXPECTED_SIZE UINT64_C(6474702976)
#define STRAT01_EXPECTED_SHA256 "68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb"
#define STRAT01_MAX_TENSORS UINT64_C(65536)
#define STRAT01_MAX_METADATA UINT64_C(65536)
#define STRAT01_MAX_NAME UINT64_C(4096)

/* GGUF v3 values from pinned llama.cpp ggml/include/gguf.h. */
enum strat01_gguf_value_type {
    STRAT01_GGUF_UINT8 = 0, STRAT01_GGUF_INT8 = 1,
    STRAT01_GGUF_UINT16 = 2, STRAT01_GGUF_INT16 = 3,
    STRAT01_GGUF_UINT32 = 4, STRAT01_GGUF_INT32 = 5,
    STRAT01_GGUF_FLOAT32 = 6, STRAT01_GGUF_BOOL = 7,
    STRAT01_GGUF_STRING = 8, STRAT01_GGUF_ARRAY = 9,
    STRAT01_GGUF_UINT64 = 10, STRAT01_GGUF_INT64 = 11,
    STRAT01_GGUF_FLOAT64 = 12, STRAT01_GGUF_VALUE_TYPE_COUNT = 13
};

/* GGML codes and block layouts from pinned llama.cpp ggml/include/ggml.h
 * and ggml/src/ggml-common.h. Only the four frozen STRAT-01 types are
 * admissible: the final type census makes this a strict identity contract. */
enum strat01_ggml_type {
    STRAT01_GGML_F32 = 0,
    STRAT01_GGML_Q5_0 = 6,
    STRAT01_GGML_Q4_K = 12,
    STRAT01_GGML_Q6_K = 14
};

typedef struct {
    uint32_t h[8];
    uint64_t bits;
    uint8_t block[64];
    size_t used;
} strat01_sha256;

typedef struct {
    FILE *f;
    uint64_t size;
    uint64_t pos;
    char error[256];
} strat01_reader;

typedef struct {
    char *name;
    uint32_t rank;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;
    uint64_t span;
    uint64_t file_offset;
} strat01_tensor;

typedef struct {
    const char *key;
    uint64_t expected;
    uint64_t value;
    int seen;
} strat01_required_scalar;

typedef struct {
    uint64_t metadata_count;
    uint64_t tensor_count;
    uint64_t alignment;
    uint64_t data_offset;
    uint64_t type_counts[4]; /* F32, Q5_0, Q4_K, Q6_K */
    int architecture_seen;
    int architecture_ok;
    int mtp_excluded;
    strat01_required_scalar required[20];
    strat01_tensor *tensors;
} strat01_inventory;

static uint32_t strat01_rotr32(uint32_t x, unsigned n) { return (x >> n) | (x << (32U - n)); }
static uint32_t strat01_be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}
static void strat01_put_be32(uint8_t *p, uint32_t x) {
    p[0] = (uint8_t)(x >> 24); p[1] = (uint8_t)(x >> 16); p[2] = (uint8_t)(x >> 8); p[3] = (uint8_t)x;
}
static void strat01_sha256_transform(strat01_sha256 *s, const uint8_t b[64]) {
    static const uint32_t k[64] = {
        0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
        0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
        0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
        0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
        0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
        0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
        0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
        0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U
    };
    uint32_t w[64], a,bv,c,d,e,f,g,h,t1,t2;
    int i;
    for (i = 0; i < 16; ++i) w[i] = strat01_be32(b + 4*i);
    for (; i < 64; ++i) {
        uint32_t s0 = strat01_rotr32(w[i-15],7) ^ strat01_rotr32(w[i-15],18) ^ (w[i-15] >> 3);
        uint32_t s1 = strat01_rotr32(w[i-2],17) ^ strat01_rotr32(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    a=s->h[0]; bv=s->h[1]; c=s->h[2]; d=s->h[3]; e=s->h[4]; f=s->h[5]; g=s->h[6]; h=s->h[7];
    for (i = 0; i < 64; ++i) {
        uint32_t s1 = strat01_rotr32(e,6) ^ strat01_rotr32(e,11) ^ strat01_rotr32(e,25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t s0 = strat01_rotr32(a,2) ^ strat01_rotr32(a,13) ^ strat01_rotr32(a,22);
        uint32_t maj = (a & bv) ^ (a & c) ^ (bv & c);
        t1 = h + s1 + ch + k[i] + w[i]; t2 = s0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=bv; bv=a; a=t1+t2;
    }
    s->h[0]+=a; s->h[1]+=bv; s->h[2]+=c; s->h[3]+=d;
    s->h[4]+=e; s->h[5]+=f; s->h[6]+=g; s->h[7]+=h;
}
static void strat01_sha256_init(strat01_sha256 *s) {
    static const uint32_t init[8] = { 0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U };
    memcpy(s->h, init, sizeof(init)); s->bits = 0; s->used = 0;
}
static void strat01_sha256_update(strat01_sha256 *s, const uint8_t *p, size_t n) {
    size_t take;
    if (n > (UINT64_MAX - s->bits) / 8U) return; /* caller has bounded files; avoid wrap defensively */
    s->bits += (uint64_t)n * 8U;
    while (n) {
        take = 64U - s->used; if (take > n) take = n;
        memcpy(s->block + s->used, p, take); s->used += take; p += take; n -= take;
        if (s->used == 64U) { strat01_sha256_transform(s, s->block); s->used = 0; }
    }
}
static void strat01_sha256_final(strat01_sha256 *s, uint8_t out[32]) {
    uint8_t pad[128]; size_t n = s->used; uint64_t bits = s->bits; int i;
    memset(pad, 0, sizeof(pad)); pad[0] = 0x80;
    strat01_sha256_update(s, pad, n < 56U ? 56U-n : 120U-n);
    for (i = 0; i < 8; ++i) pad[i] = (uint8_t)(bits >> (56 - 8*i));
    strat01_sha256_update(s, pad, 8);
    for (i = 0; i < 8; ++i) strat01_put_be32(out + 4*i, s->h[i]);
}
static void strat01_sha256_hex(const uint8_t in[32], char out[65]) {
    static const char hx[] = "0123456789abcdef"; int i;
    for (i = 0; i < 32; ++i) { out[2*i] = hx[in[i] >> 4]; out[2*i+1] = hx[in[i] & 15]; } out[64] = 0;
}
static int strat01_sha256_file(const char *path, char hex[65], uint64_t *bytes, char err[256]) {
    FILE *f = fopen(path, "rb"); uint8_t buf[65536], digest[32]; size_t n; strat01_sha256 s; uint64_t total = 0;
    if (!f) { snprintf(err, 256, "cannot open for SHA-256: %s", strerror(errno)); return 0; }
    strat01_sha256_init(&s);
    while ((n = fread(buf, 1, sizeof(buf), f)) != 0) {
        if (total > UINT64_MAX - (uint64_t)n) { fclose(f); snprintf(err,256,"SHA-256 byte count overflow"); return 0; }
        total += (uint64_t)n; strat01_sha256_update(&s, buf, n);
    }
    if (ferror(f)) { fclose(f); snprintf(err, 256, "SHA-256 read failure"); return 0; }
    fclose(f); strat01_sha256_final(&s, digest); strat01_sha256_hex(digest, hex); *bytes = total; return 1;
}

static void strat01_fail(strat01_reader *r, const char *what) {
    if (!r->error[0]) snprintf(r->error, sizeof(r->error), "%s at byte %" PRIu64, what, r->pos);
}
static int strat01_read(strat01_reader *r, void *dst, size_t n) {
    if ((uint64_t)n > r->size - r->pos || fread(dst, 1, n, r->f) != n) { strat01_fail(r, "truncated GGUF"); return 0; }
    r->pos += (uint64_t)n; return 1;
}
static int strat01_skip(strat01_reader *r, uint64_t n) {
    uint8_t buf[4096]; size_t take;
    if (n > r->size - r->pos) { strat01_fail(r, "length exceeds GGUF bounds"); return 0; }
    while (n) { take = n > sizeof(buf) ? sizeof(buf) : (size_t)n; if (!strat01_read(r, buf, take)) return 0; n -= take; }
    return 1;
}
static int strat01_u32(strat01_reader *r, uint32_t *v) { uint8_t b[4]; if(!strat01_read(r,b,4)) return 0; *v=(uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24); return 1; }
static int strat01_u64(strat01_reader *r, uint64_t *v) { uint8_t b[8]; int i; if(!strat01_read(r,b,8)) return 0; *v=0; for(i=7;i>=0;--i)*v=(*v<<8)|b[i]; return 1; }
static int strat01_add_u64(uint64_t a, uint64_t b, uint64_t *o) { if (a > UINT64_MAX-b) return 0; *o=a+b; return 1; }
static int strat01_mul_u64(uint64_t a, uint64_t b, uint64_t *o) { if (a && b > UINT64_MAX/a) return 0; *o=a*b; return 1; }
static int strat01_round_up(uint64_t v, uint64_t a, uint64_t *o) { uint64_t rem; if (!a) return 0; rem=v%a; return rem ? strat01_add_u64(v,a-rem,o) : (*o=v,1); }
static int strat01_power_of_two(uint64_t v) { return v && !(v & (v-1)); }

static int strat01_read_name(strat01_reader *r, char **out, const char *label) {
    uint64_t n; char *s;
    if (!strat01_u64(r, &n)) return 0;
    if (n == 0 || n > STRAT01_MAX_NAME || n > r->size-r->pos || n >= SIZE_MAX) { strat01_fail(r, label); return 0; }
    s=(char *)malloc((size_t)n+1U); if (!s) { strat01_fail(r,"out of memory for GGUF name"); return 0; }
    if (!strat01_read(r,s,(size_t)n)) { free(s); return 0; }
    if (memchr(s,0,(size_t)n)) { free(s); strat01_fail(r,"NUL in GGUF name"); return 0; }
    s[n]=0; *out=s; return 1;
}
static int strat01_skip_string(strat01_reader *r) { uint64_t n; if(!strat01_u64(r,&n)) return 0; return strat01_skip(r,n); }
static int strat01_type_scalar_size(uint32_t t, uint64_t *sz) {
    switch(t) {
        case STRAT01_GGUF_UINT8: case STRAT01_GGUF_INT8: case STRAT01_GGUF_BOOL: *sz=1; return 1;
        case STRAT01_GGUF_UINT16: case STRAT01_GGUF_INT16: *sz=2; return 1;
        case STRAT01_GGUF_UINT32: case STRAT01_GGUF_INT32: case STRAT01_GGUF_FLOAT32: *sz=4; return 1;
        case STRAT01_GGUF_UINT64: case STRAT01_GGUF_INT64: case STRAT01_GGUF_FLOAT64: *sz=8; return 1;
        default: return 0;
    }
}
static int strat01_read_numeric(strat01_reader *r, uint32_t type, uint64_t *out) {
    uint8_t b[8]; uint64_t n=0; uint64_t sz;
    if (!strat01_type_scalar_size(type,&sz) || type==STRAT01_GGUF_FLOAT32 || type==STRAT01_GGUF_FLOAT64 || type==STRAT01_GGUF_BOOL) return 0;
    if (!strat01_read(r,b,(size_t)sz)) return -1;
    while (sz) { --sz; n=(n<<8)|b[sz]; } *out=n; return 1;
}
static void strat01_init_required(strat01_inventory *in) {
    static const char *keys[20] = {
        "general.quantization_version", "general.file_type",
        "deepseek2.vocab_size", "deepseek2.embedding_length", "deepseek2.block_count",
        "deepseek2.feed_forward_length", "deepseek2.expert_feed_forward_length",
        "deepseek2.expert_count", "deepseek2.expert_used_count", "deepseek2.expert_shared_count",
        "deepseek2.attention.head_count", "deepseek2.attention.head_count_kv",
        "deepseek2.attention.key_length", "deepseek2.attention.value_length",
        "deepseek2.attention.key_length_mla", "deepseek2.attention.value_length_mla",
        "deepseek2.attention.kv_lora_rank", "deepseek2.rope.dimension_count",
        "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"
    };
    static const uint64_t values[20] = { 2,15,128256,1536,26,8960,1280,64,4,1,32,1,576,512,192,192,512,64,1,2 };
    int i; memset(in,0,sizeof(*in)); in->alignment=STRAT01_DEFAULT_ALIGNMENT;
    for(i=0;i<20;++i){in->required[i].key=keys[i];in->required[i].expected=values[i];}
}
static int strat01_record_required(strat01_inventory *in, const char *key, uint32_t type, uint64_t value, strat01_reader *r) {
    int i;
    for(i=0;i<20;++i) if(!strcmp(key,in->required[i].key)) {
        if (in->required[i].seen || type != STRAT01_GGUF_UINT32 || value != in->required[i].expected) { strat01_fail(r,"frozen scalar metadata mismatch"); return 0; }
        in->required[i].seen=1; in->required[i].value=value;
        return 1;
    }
    return 1;
}
static int strat01_read_metadata_value(strat01_reader *r, strat01_inventory *in, const char *key, uint32_t type) {
    uint64_t n, sz, value; uint32_t elem; int rc;
    if (type >= STRAT01_GGUF_VALUE_TYPE_COUNT) { strat01_fail(r,"unknown GGUF metadata value type"); return 0; }
    if (type == STRAT01_GGUF_ARRAY) {
        if(!strat01_u32(r,&elem)||!strat01_u64(r,&n)) return 0;
        if (elem >= STRAT01_GGUF_VALUE_TYPE_COUNT || elem == STRAT01_GGUF_ARRAY) { strat01_fail(r,"unknown/nested GGUF array type"); return 0; }
        if (elem == STRAT01_GGUF_STRING) {
            if (n > (r->size-r->pos)/8U) { strat01_fail(r,"GGUF string array count exceeds bounds"); return 0; }
            while(n--) if(!strat01_skip_string(r)) return 0;
            return 1;
        }
        if (!strat01_type_scalar_size(elem,&sz) || !strat01_mul_u64(n,sz,&value)) { strat01_fail(r,"GGUF array byte-size overflow"); return 0; }
        return strat01_skip(r,value);
    }
    if (type == STRAT01_GGUF_STRING) {
        if (!strcmp(key,"general.architecture")) {
            char *architecture=NULL;
            if(!strat01_read_name(r,&architecture,"invalid architecture string")) return 0;
            in->architecture_seen=1; in->architecture_ok=!strcmp(architecture,"deepseek2"); free(architecture);
            if(!in->architecture_ok) { strat01_fail(r,"architecture is not frozen deepseek2"); return 0; }
            return 1;
        }
        return strat01_skip_string(r);
    }
    /* general.alignment is optional in GGUF; absence means the v3 default 32.
       If serialized explicitly, it must still equal the frozen effective value. */
    if (!strcmp(key,"general.alignment")) {
        if (type != STRAT01_GGUF_UINT32 || strat01_read_numeric(r,type,&value) != 1 || value != STRAT01_DEFAULT_ALIGNMENT) {
            strat01_fail(r,"frozen GGUF alignment mismatch"); return 0;
        }
        in->alignment=value; return 1;
    }
    /* Known floating-point and bool metadata occur in the accepted artifact
       (YaRN, RMS epsilon, expert norm). They are structurally parsed here but
       are not integer members of rung 0's frozen scalar table. */
    if (type == STRAT01_GGUF_FLOAT32 || type == STRAT01_GGUF_FLOAT64 || type == STRAT01_GGUF_BOOL) {
        if (!strat01_type_scalar_size(type,&sz)) { strat01_fail(r,"unsupported GGUF scalar type"); return 0; }
        return strat01_skip(r,sz);
    }
    rc=strat01_read_numeric(r,type,&value);
    if (rc == -1) return 0;
    if (rc == 0) { strat01_fail(r,"unsupported GGUF scalar type"); return 0; }
    return strat01_record_required(in,key,type,value,r);
}
static int strat01_tensor_type(uint32_t type, uint64_t *block, uint64_t *bytes, unsigned *which) {
    switch(type) {
        case STRAT01_GGML_F32: *block=1; *bytes=4; *which=0; return 1;
        case STRAT01_GGML_Q5_0: *block=32; *bytes=22; *which=1; return 1;
        case STRAT01_GGML_Q4_K: *block=256; *bytes=144; *which=2; return 1;
        case STRAT01_GGML_Q6_K: *block=256; *bytes=210; *which=3; return 1;
        default: return 0;
    }
}
static const char *strat01_type_name(uint32_t t) {
    switch(t) { case STRAT01_GGML_F32:return "F32"; case STRAT01_GGML_Q5_0:return "Q5_0"; case STRAT01_GGML_Q4_K:return "Q4_K"; case STRAT01_GGML_Q6_K:return "Q6_K"; default:return "UNKNOWN"; }
}
static int strat01_parse_gguf(const char *path, strat01_inventory *in, uint64_t *file_size, char error[256]) {
    FILE *f; int64_t end; strat01_reader r; uint32_t magic,version,type,rank; uint64_t n_tensors,n_meta,i,j,block,bytes,elements,span,endoff; unsigned which;
    strat01_init_required(in); memset(error,0,256);
    f=fopen(path,"rb"); if(!f){snprintf(error,256,"cannot open GGUF: %s",strerror(errno));return 0;}
    if(STRAT01_FSEEK(f,0,SEEK_END)!=0 || (end=(int64_t)STRAT01_FTELL(f))<0 || STRAT01_FSEEK(f,0,SEEK_SET)!=0){fclose(f);snprintf(error,256,"cannot determine GGUF size");return 0;}
    r.f=f; r.size=(uint64_t)end; r.pos=0; r.error[0]=0; *file_size=r.size;
    if(!strat01_u32(&r,&magic)||!strat01_u32(&r,&version)||!strat01_u64(&r,&n_tensors)||!strat01_u64(&r,&n_meta)) goto fail;
    if(magic!=STRAT01_GGUF_MAGIC||version!=STRAT01_GGUF_VERSION){strat01_fail(&r,"GGUF magic/version mismatch");goto fail;}
    if(!n_tensors||n_tensors>STRAT01_MAX_TENSORS||n_meta>STRAT01_MAX_METADATA){strat01_fail(&r,"GGUF count exceeds bounded parser limit");goto fail;}
    in->metadata_count=n_meta; in->tensor_count=n_tensors;
    for(i=0;i<n_meta;++i){char *key=NULL; if(!strat01_read_name(&r,&key,"invalid metadata key")||!strat01_u32(&r,&type)){free(key);goto fail;} if(!strat01_read_metadata_value(&r,in,key,type)){free(key);goto fail;} free(key);}
    if(!in->alignment || !strat01_power_of_two(in->alignment) || in->alignment!=STRAT01_DEFAULT_ALIGNMENT){strat01_fail(&r,"frozen GGUF alignment mismatch");goto fail;}
    in->tensors=(strat01_tensor *)calloc((size_t)n_tensors,sizeof(*in->tensors)); if(!in->tensors){strat01_fail(&r,"out of memory for tensor descriptors");goto fail;}
    for(i=0;i<n_tensors;++i){
        strat01_tensor *t=&in->tensors[i];
        if(!strat01_read_name(&r,&t->name,"invalid tensor name")||!strat01_u32(&r,&rank)){goto fail;}
        if(!rank||rank>4){strat01_fail(&r,"invalid tensor rank");goto fail;} t->rank=rank; elements=1;
        for(j=0;j<rank;++j){if(!strat01_u64(&r,&t->dims[j]))goto fail; if(!t->dims[j]||!strat01_mul_u64(elements,t->dims[j],&elements)){strat01_fail(&r,"tensor dimension product overflow");goto fail;}}
        if(!strat01_u32(&r,&type)||!strat01_u64(&r,&t->offset)){goto fail;} t->type=type;
        if(!strat01_tensor_type(type,&block,&bytes,&which)||t->dims[0]%block||elements%block||!strat01_mul_u64(elements/block,bytes,&span)){strat01_fail(&r,"unsupported tensor type or block divisibility");goto fail;}
        t->span=span; in->type_counts[which]++;
        if (strstr(t->name,"blk.26.") || strstr(t->name,".nextn.") || strstr(t->name,"mtp")) { strat01_fail(&r,"MTP tensor present in base-only GGUF"); goto fail; }
    }
    if(!strat01_round_up(r.pos,in->alignment,&in->data_offset)||in->data_offset>r.size){strat01_fail(&r,"invalid aligned GGUF data offset");goto fail;}
    for(i=0;i<n_tensors;++i){
        strat01_tensor *a=&in->tensors[i];
        if(a->offset%in->alignment||!strat01_add_u64(in->data_offset,a->offset,&a->file_offset)||!strat01_add_u64(a->file_offset,a->span,&endoff)||endoff>r.size){strat01_fail(&r,"tensor payload is out of bounds or unaligned");goto fail;}
        for(j=0;j<i;++j){strat01_tensor *b=&in->tensors[j]; uint64_t bend; if(!strcmp(a->name,b->name)){strat01_fail(&r,"duplicate tensor name");goto fail;} if(!strat01_add_u64(b->file_offset,b->span,&bend)){strat01_fail(&r,"tensor span overflow");goto fail;} if(a->file_offset<bend && b->file_offset<endoff){strat01_fail(&r,"overlapping tensor payload spans");goto fail;}}
    }
    in->mtp_excluded=1;
    if(!in->architecture_seen||!in->architecture_ok||n_meta!=47||n_tensors!=414||in->type_counts[0]!=129||in->type_counts[1]!=26||in->type_counts[2]!=233||in->type_counts[3]!=26){strat01_fail(&r,"frozen GGUF metadata/tensor census mismatch");goto fail;}
    for(i=0;i<20;++i) if(!in->required[i].seen){strat01_fail(&r,"missing frozen scalar metadata");goto fail;}
    /* The pinned architecture gives 192 MLA Q/K values and 64 RoPE values. */
    if(in->required[14].value-in->required[17].value!=128){strat01_fail(&r,"frozen non-RoPE head dimension mismatch");goto fail;}
    fclose(f); return 1;
fail:
    if(!r.error[0])strat01_fail(&r,"GGUF parse failure"); snprintf(error,256,"%s",r.error); fclose(f); return 0;
}
static void strat01_free_inventory(strat01_inventory *in) { uint64_t i; if(in->tensors){for(i=0;i<in->tensor_count;++i)free(in->tensors[i].name);free(in->tensors);} memset(in,0,sizeof(*in)); }
static void strat01_json_string(FILE *o, const char *s) { const unsigned char *p=(const unsigned char *)s; fputc('"',o); for(;*p;++p){if(*p=='"'||*p=='\\')fprintf(o,"\\%c",*p);else if(*p<0x20||*p>=0x80)fprintf(o,"\\u%04x",(unsigned)*p);else fputc(*p,o);} fputc('"',o); }
static int strat01_write_failure_json(const char *out, const char *path, const char *error) { FILE *o=fopen(out,"wb"); if(!o)return 0; fputs("{\"command\":\"--strat01-gguf-inspect\",\"input_path\":",o);strat01_json_string(o,path);fputs(",\"gate_status\":\"FAIL_ENGINE_RUNG0\",\"error\":",o);strat01_json_string(o,error);fputs("}\n",o);fclose(o);return 1; }
static int strat01_write_success_json(const char *out, const char *path, const strat01_inventory *in, uint64_t size, const char *hash, const char *engine_source_hash, const char *inspector_source_hash) {
    FILE *o=fopen(out,"wb"); uint64_t i,j;
    if(!o)return 0;
    fputs("{\n  \"engine_source_sha256\": ",o);strat01_json_string(o,engine_source_hash);
    fputs(",\n  \"inspector_source_sha256\": ",o);strat01_json_string(o,inspector_source_hash);
    fputs(",\n  \"command\": \"--strat01-gguf-inspect\",\n  \"input_path\": ",o);strat01_json_string(o,path);
    fprintf(o,",\n  \"byte_size\": %" PRIu64 ",\n  \"sha256\": ",size);strat01_json_string(o,hash);
    fprintf(o,",\n  \"gguf\": {\"magic\": \"GGUF\", \"version\": %u, \"alignment\": %" PRIu64 ", \"data_offset\": %" PRIu64 "},",STRAT01_GGUF_VERSION,in->alignment,in->data_offset);
    fprintf(o,"\n  \"metadata_count\": %" PRIu64 ",\n  \"tensor_count\": %" PRIu64 ",",in->metadata_count,in->tensor_count);
    fprintf(o,"\n  \"required_metadata\": {\"general.architecture\": \"deepseek2\", \"general.alignment\": %" PRIu64,in->alignment); for(i=0;i<20;++i){fputs(", ",o);strat01_json_string(o,in->required[i].key);fprintf(o,": %" PRIu64,in->required[i].value);} fputs("},",o);
    fprintf(o,"\n  \"type_census\": {\"F32\": %" PRIu64 ", \"Q5_0\": %" PRIu64 ", \"Q4_K\": %" PRIu64 ", \"Q6_K\": %" PRIu64 "},",in->type_counts[0],in->type_counts[1],in->type_counts[2],in->type_counts[3]);
    fprintf(o,"\n  \"mtp_excluded\": %s,\n  \"tensors\": [",in->mtp_excluded?"true":"false");
    for(i=0;i<in->tensor_count;++i){const strat01_tensor*t=&in->tensors[i];fputs(i?",\n    {":"\n    {",o);fputs("\"name\": ",o);strat01_json_string(o,t->name);fprintf(o,", \"rank\": %u, \"dims\": [",t->rank);for(j=0;j<t->rank;++j)fprintf(o,"%s%" PRIu64,j?", ":"",t->dims[j]);fprintf(o,"], \"type\": \"");fputs(strat01_type_name(t->type),o);fprintf(o,"\", \"type_id\": %u, \"offset\": %" PRIu64 ", \"byte_span\": %" PRIu64 ", \"file_offset\": %" PRIu64 "}",t->type,t->offset,t->span,t->file_offset);}
    fputs("\n  ],\n  \"gate_status\": \"PASS_ENGINE_RUNG0\"\n}\n",o);fclose(o);return 1;
}
static int strat01_gguf_inspect_cli(const char *path, const char *json_path, const char *engine_source_path) {
    strat01_inventory in; uint64_t parsed_size=0,hashed_size=0,expected_size=STRAT01_EXPECTED_SIZE; char err[256]={0},hash[65]={0},engine_source_hash[65]={0},inspector_source_hash[65]={0}; const char *expected_hash=STRAT01_EXPECTED_SHA256;
#if defined(STRAT01_ENABLE_TEST_IDENTITY_OVERRIDE)
#if !defined(STRAT01_TEST_EXPECTED_SIZE) || !defined(STRAT01_TEST_EXPECTED_SHA256)
#error STRAT01 test identity override requires both compile-time size and SHA-256
#endif
    expected_size=(uint64_t)STRAT01_TEST_EXPECTED_SIZE; expected_hash=STRAT01_TEST_EXPECTED_SHA256;
#endif
    if(!strat01_parse_gguf(path,&in,&parsed_size,err)){strat01_write_failure_json(json_path,path,err);fprintf(stderr,"STRAT-01 GGUF inspect refused: %s\n",err);return 1;}
    if(parsed_size!=expected_size){snprintf(err,sizeof(err),"frozen byte size mismatch: got %" PRIu64,parsed_size);strat01_free_inventory(&in);strat01_write_failure_json(json_path,path,err);fprintf(stderr,"STRAT-01 GGUF inspect refused: %s\n",err);return 1;}
    if(!strat01_sha256_file(path,hash,&hashed_size,err)||hashed_size!=parsed_size||strcmp(hash,expected_hash)){if(!err[0])snprintf(err,sizeof(err),"frozen SHA-256 mismatch");strat01_free_inventory(&in);strat01_write_failure_json(json_path,path,err);fprintf(stderr,"STRAT-01 GGUF inspect refused: %s\n",err);return 1;}
    if(!strat01_sha256_file(engine_source_path,engine_source_hash,&hashed_size,err))strcpy(engine_source_hash,"unavailable");
    err[0]=0;
    if(!strat01_sha256_file(__FILE__,inspector_source_hash,&hashed_size,err))strcpy(inspector_source_hash,"unavailable");
    if(!strat01_write_success_json(json_path,path,&in,parsed_size,hash,engine_source_hash,inspector_source_hash)){strat01_free_inventory(&in);fprintf(stderr,"STRAT-01 GGUF inspect refused: cannot write JSON %s\n",json_path);return 1;}
    strat01_free_inventory(&in); fprintf(stderr,"STRAT-01 GGUF inspect: PASS_ENGINE_RUNG0\n"); return 0;
}

#endif /* STRAT01_GGUF_INSPECT_H */
