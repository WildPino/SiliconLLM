/*
 * STRAT-01 GGUF numerical-parity rung 1 (header-only alongside rung 0).
 *
 * This is deliberately a small, scalar, bounded-memory apparatus.  It is
 * neither a general GGUF loader nor an inference implementation: it consumes
 * exactly the three frozen descriptors after rung 0 has accepted the complete
 * artifact identity.  Layouts are from llama.cpp 5b335f413e4f73b0809c4fe39af894efbcc6a0d2.
 */
#ifndef STRAT01_GGUF_RUNG1_H
#define STRAT01_GGUF_RUNG1_H

#include <float.h>

#define STRAT01_RUNG1_REFERENCE_REVISION "llama.cpp 5b335f413e4f73b0809c4fe39af894efbcc6a0d2 (gguf-py/gguf/quants.py)"
#define STRAT01_RUNG1_Q4_NAME "blk.0.attn_q.weight"
#define STRAT01_RUNG1_Q5_NAME "blk.0.attn_k_b.weight"
#define STRAT01_RUNG1_Q6_NAME "blk.0.ffn_down.weight"

typedef struct {
    const char *cell;
    const char *name;
    uint32_t type;
    uint32_t rank;
    uint64_t dims[4];
    uint64_t offset;
    uint64_t span;
    uint64_t file_offset;
} strat01_rung1_expected;

typedef struct {
    const strat01_rung1_expected *expected;
    uint64_t block_count;
    uint64_t decoded_count;
    uint64_t finite_count;
    uint64_t output_finite_count;
    float decoded_min;
    float decoded_max;
    float output_min;
    float output_max;
    char dequant_sha256[65];
    char output_sha256[65];
    char output_path[1024];
} strat01_rung1_cell_result;

static const strat01_rung1_expected strat01_rung1_expected_cells[3] = {
    { "G-R1-Q4", STRAT01_RUNG1_Q4_NAME, STRAT01_GGML_Q4_K, 2,
      { 1536, 6144, 0, 0 }, UINT64_C(279677952), UINT64_C(5308416), UINT64_C(285780864) },
    { "G-R1-Q5", STRAT01_RUNG1_Q5_NAME, STRAT01_GGML_Q5_0, 3,
      { 128, 512, 32, 0 }, UINT64_C(272421888), UINT64_C(1441792), UINT64_C(278524800) },
    { "G-R1-Q6", STRAT01_RUNG1_Q6_NAME, STRAT01_GGML_Q6_K, 2,
      { 8960, 1536, 0, 0 }, UINT64_C(286755840), UINT64_C(11289600), UINT64_C(292858752) }
};

static uint16_t strat01_rung1_u16le(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
static uint32_t strat01_rung1_u32le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
/* GGML stores binary16 bytes little-endian.  This conversion is intentionally
 * bit-based rather than a host half type/padded struct interpretation. */
static float strat01_rung1_fp16le(const uint8_t *p) {
    uint16_t h = strat01_rung1_u16le(p);
    uint32_t sign = (uint32_t)(h >> 15), exponent = (h >> 10) & 31U, fraction = h & 1023U;
    float value;
    if (exponent == 0) value = (float)fraction * 0x1p-24f;
    else if (exponent == 31) value = fraction ? NAN : INFINITY;
    else value = (float)(fraction + 1024U) * ldexpf(1.0f, (int)exponent - 25);
    return sign ? -value : value;
}
static void strat01_rung1_put_f32le(uint8_t out[4], float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    out[0] = (uint8_t)bits; out[1] = (uint8_t)(bits >> 8);
    out[2] = (uint8_t)(bits >> 16); out[3] = (uint8_t)(bits >> 24);
}
static void strat01_rung1_sha_f32le(strat01_sha256 *sha, float value) {
    uint8_t bytes[4]; strat01_rung1_put_f32le(bytes, value); strat01_sha256_update(sha, bytes, sizeof(bytes));
}
static float strat01_rung1_input(uint64_t i) {
    return (float)((int64_t)((i * UINT64_C(73) + UINT64_C(19)) % UINT64_C(257)) - INT64_C(128)) / 128.0f;
}
static void strat01_rung1_get_scale_min_k4(unsigned j, const uint8_t scales[12], uint8_t *scale, uint8_t *minimum) {
    if (j < 4U) { *scale = scales[j] & 63U; *minimum = scales[j + 4U] & 63U; }
    else { *scale = (scales[j + 4U] & 15U) | ((scales[j - 4U] >> 6) << 4);
           *minimum = (scales[j + 4U] >> 4) | ((scales[j] >> 6) << 4); }
}
static void strat01_rung1_decode_q4_k(const uint8_t block[144], float out[256]) {
    const float d = strat01_rung1_fp16le(block), dmin = strat01_rung1_fp16le(block + 2);
    const uint8_t *scales = block + 4, *qs = block + 16;
    unsigned pair;
    for (pair = 0; pair < 4U; ++pair) {
        uint8_t sc0, mn0, sc1, mn1; unsigned l, base = pair * 64U;
        volatile float d0, m0, d1, m1;
        strat01_rung1_get_scale_min_k4(pair * 2U, scales, &sc0, &mn0);
        strat01_rung1_get_scale_min_k4(pair * 2U + 1U, scales, &sc1, &mn1);
        /* Preserve the pinned float32 operation sequence.  Volatile
         * intermediates prevent -mfma from changing the canonical digest. */
        d0 = d * (float)sc0; m0 = dmin * (float)mn0;
        d1 = d * (float)sc1; m1 = dmin * (float)mn1;
        for (l = 0; l < 32U; ++l) {
            volatile float product = d0 * (float)(qs[pair * 32U + l] & 15U);
            out[base + l] = product - m0;
        }
        for (l = 0; l < 32U; ++l) {
            volatile float product = d1 * (float)(qs[pair * 32U + l] >> 4);
            out[base + 32U + l] = product - m1;
        }
    }
}
static void strat01_rung1_decode_q5_0(const uint8_t block[22], float out[32]) {
    const float d = strat01_rung1_fp16le(block); const uint32_t qh = strat01_rung1_u32le(block + 2); const uint8_t *qs = block + 6;
    unsigned j;
    for (j = 0; j < 16U; ++j) {
        uint8_t hi0 = (uint8_t)(((qh >> j) << 4) & 16U), hi1 = (uint8_t)((qh >> (j + 12U)) & 16U);
        out[j] = (float)((int)((qs[j] & 15U) | hi0) - 16) * d;
        out[j + 16U] = (float)((int)((qs[j] >> 4) | hi1) - 16) * d;
    }
}
static void strat01_rung1_decode_q6_k(const uint8_t block[210], float out[256]) {
    const uint8_t *ql = block, *qh = block + 128; const int8_t *scales = (const int8_t *)(block + 192); const float d = strat01_rung1_fp16le(block + 208);
    unsigned n;
    for (n = 0; n < 256U; n += 128U) {
        const uint8_t *qln = ql + n / 2U, *qhn = qh + n / 4U; const int8_t *sc = scales + n / 16U; unsigned l;
        for (l = 0; l < 32U; ++l) {
            unsigned is = l / 16U;
            volatile float ds1, ds2, ds3, ds4;
            int q1 = (int)((qln[l] & 15U) | (((qhn[l] >> 0) & 3U) << 4)) - 32;
            int q2 = (int)((qln[l + 32U] & 15U) | (((qhn[l] >> 2) & 3U) << 4)) - 32;
            int q3 = (int)((qln[l] >> 4) | (((qhn[l] >> 4) & 3U) << 4)) - 32;
            int q4 = (int)((qln[l + 32U] >> 4) | (((qhn[l] >> 6) & 3U) << 4)) - 32;
            ds1 = d * (float)sc[is]; ds2 = d * (float)sc[is + 2U];
            ds3 = d * (float)sc[is + 4U]; ds4 = d * (float)sc[is + 6U];
            out[n + l] = ds1 * (float)q1;
            out[n + l + 32U] = ds2 * (float)q2;
            out[n + l + 64U] = ds3 * (float)q3;
            out[n + l + 96U] = ds4 * (float)q4;
        }
    }
}
static int strat01_rung1_decode_block(uint32_t type, const uint8_t *block, float out[256], unsigned *count) {
    if (type == STRAT01_GGML_Q4_K) { strat01_rung1_decode_q4_k(block, out); *count = 256; return 1; }
    if (type == STRAT01_GGML_Q5_0) { strat01_rung1_decode_q5_0(block, out); *count = 32; return 1; }
    if (type == STRAT01_GGML_Q6_K) { strat01_rung1_decode_q6_k(block, out); *count = 256; return 1; }
    return 0;
}
static const strat01_tensor *strat01_rung1_find_tensor(const strat01_inventory *in, const char *name) {
    uint64_t i; for (i = 0; i < in->tensor_count; ++i) if (!strcmp(in->tensors[i].name, name)) return &in->tensors[i]; return NULL;
}
static int strat01_rung1_check_descriptor(const strat01_inventory *in, const strat01_rung1_expected *e, const strat01_tensor **out, char error[256]) {
    const strat01_tensor *t = strat01_rung1_find_tensor(in, e->name); unsigned i;
    if (!t) { snprintf(error, 256, "frozen rung-1 tensor is absent: %s", e->name); return 0; }
    if (t->type != e->type || t->rank != e->rank || t->offset != e->offset || t->span != e->span || t->file_offset != e->file_offset) {
        snprintf(error, 256, "frozen rung-1 descriptor mismatch: %s", e->name); return 0;
    }
    for (i = 0; i < e->rank; ++i) if (t->dims[i] != e->dims[i]) { snprintf(error, 256, "frozen rung-1 dimensions mismatch: %s", e->name); return 0; }
    *out = t; return 1;
}
static int strat01_rung1_path(char out[1024], const char *dir, const char *leaf) {
    int n = snprintf(out, 1024, "%s/%s", dir, leaf); return n > 0 && n < 1024;
}
static int strat01_rung1_process_cell(const char *gguf_path, const strat01_tensor *t, strat01_rung1_cell_result *result, const char *out_dir, char error[256]) {
    FILE *input = NULL, *output = NULL; uint64_t elements = 1, rows, blocks_per_row, block_values, block_bytes, row, i;
    uint8_t raw[210]; float decoded[256]; strat01_sha256 dequant_sha, output_sha; uint8_t digest[32]; unsigned decoded_n, which;
    char leaf[128];
    if (!strat01_tensor_type(t->type, &block_values, &block_bytes, &which) || t->type == STRAT01_GGML_F32) { snprintf(error,256,"unsupported rung-1 tensor type"); return 0; }
    for (i = 0; i < t->rank; ++i) if (!strat01_mul_u64(elements, t->dims[i], &elements)) { snprintf(error,256,"rung-1 element count overflow"); return 0; }
    rows = 1; for (i = 1; i < t->rank; ++i) if (!strat01_mul_u64(rows, t->dims[i], &rows)) { snprintf(error,256,"rung-1 row count overflow"); return 0; }
    if (t->dims[0] % block_values) { snprintf(error,256,"rung-1 row block divisibility failure"); return 0; }
    blocks_per_row = t->dims[0] / block_values;
    if (snprintf(leaf, sizeof(leaf), "strat01_rung1_%s.f32", result->expected->cell + 5) <= 0 || !strat01_rung1_path(result->output_path,out_dir,leaf)) { snprintf(error,256,"rung-1 output path overflow"); return 0; }
    input = fopen(gguf_path,"rb"); output = fopen(result->output_path,"wb");
    if (!input || !output) { if(input)fclose(input);if(output)fclose(output);snprintf(error,256,"cannot open rung-1 payload/output");return 0; }
    if (STRAT01_FSEEK(input, (int64_t)t->file_offset, SEEK_SET) != 0) { fclose(input); fclose(output); snprintf(error,256,"cannot seek rung-1 payload"); return 0; }
    strat01_sha256_init(&dequant_sha); strat01_sha256_init(&output_sha);
    for (row = 0; row < rows; ++row) {
        float accumulator = 0.0f;
        for (i = 0; i < blocks_per_row; ++i) {
            uint64_t j;
            if (fread(raw, 1, (size_t)block_bytes, input) != (size_t)block_bytes || !strat01_rung1_decode_block(t->type, raw, decoded, &decoded_n) || decoded_n != block_values) {
                fclose(input); fclose(output); snprintf(error,256,"rung-1 payload read/decode failure"); return 0;
            }
            for (j = 0; j < block_values; ++j) {
                float value = decoded[j]; volatile float product;
                if (!isfinite(value)) { fclose(input); fclose(output); snprintf(error,256,"non-finite decoded rung-1 value"); return 0; }
                if (!result->finite_count || value < result->decoded_min) result->decoded_min = value;
                if (!result->finite_count || value > result->decoded_max) result->decoded_max = value;
                ++result->finite_count; ++result->decoded_count; strat01_rung1_sha_f32le(&dequant_sha,value);
                product = value * strat01_rung1_input(i * block_values + j); accumulator = accumulator + product;
            }
            ++result->block_count;
        }
        if (!isfinite(accumulator)) { fclose(input); fclose(output); snprintf(error,256,"non-finite rung-1 row output"); return 0; }
        if (!result->output_finite_count || accumulator < result->output_min) result->output_min=accumulator;
        if (!result->output_finite_count || accumulator > result->output_max) result->output_max=accumulator;
        ++result->output_finite_count; strat01_rung1_sha_f32le(&output_sha,accumulator);
        { uint8_t bytes[4]; strat01_rung1_put_f32le(bytes,accumulator); if (fwrite(bytes,1,4,output)!=4) { fclose(input);fclose(output);snprintf(error,256,"rung-1 output write failure");return 0; } }
    }
    { int input_error = ferror(input), input_close = fclose(input), output_close = fclose(output);
      if (input_error || input_close != 0 || output_close != 0) { snprintf(error,256,"rung-1 payload/output close failure"); return 0; } }
    if (result->decoded_count != elements || result->output_finite_count != rows) { snprintf(error,256,"rung-1 decoded/output count mismatch"); return 0; }
    strat01_sha256_final(&dequant_sha,digest); strat01_sha256_hex(digest,result->dequant_sha256);
    strat01_sha256_final(&output_sha,digest); strat01_sha256_hex(digest,result->output_sha256); return 1;
}
static int strat01_rung1_write_failure(const char *out_dir, const char *path, const char *error) {
    char report[1024]; FILE *o; if (!strat01_rung1_path(report,out_dir,"strat01_rung1.json") || !(o=fopen(report,"wb"))) return 0;
    fputs("{\"command\":\"--strat01-gguf-rung1\",\"input_path\":",o);strat01_json_string(o,path);fputs(",\"c_state\":\"ENGINE_RUNG1_C_FAILURE\",\"error\":",o);strat01_json_string(o,error);fputs("}\n",o);fclose(o);return 1;
}
static int strat01_rung1_write_success(const char *out_dir, const char *path, uint64_t byte_size, const char *artifact_sha, const char *engine_sha, const char *rung1_sha, const strat01_rung1_cell_result cells[3]) {
    char report[1024]; FILE *o; unsigned c, d;
    if(!strat01_rung1_path(report,out_dir,"strat01_rung1.json") || !(o=fopen(report,"wb"))) return 0;
    fputs("{\n  \"command\": \"--strat01-gguf-rung1\",\n  \"c_state\": \"ENGINE_OUTPUT_READY_PENDING_REFERENCE\",\n  \"input_path\": ",o);strat01_json_string(o,path);
    fprintf(o,",\n  \"byte_size\": %" PRIu64 ",\n  \"sha256\": ",byte_size);strat01_json_string(o,artifact_sha);
    fputs(",\n  \"engine_source_sha256\": ",o);strat01_json_string(o,engine_sha);fputs(",\n  \"rung1_source_sha256\": ",o);strat01_json_string(o,rung1_sha);
    fputs(",\n  \"reference_revision\": ",o);strat01_json_string(o,STRAT01_RUNG1_REFERENCE_REVISION);fputs(",\n  \"cells\": [",o);
    for(c=0;c<3U;++c){const strat01_rung1_cell_result*r=&cells[c];const strat01_rung1_expected*e=r->expected;fprintf(o,c?",\n    {":"\n    {");fputs("\"cell\": ",o);strat01_json_string(o,e->cell);fputs(", \"tensor\": ",o);strat01_json_string(o,e->name);fprintf(o,", \"type\": \"");fputs(strat01_type_name(e->type),o);fprintf(o,"\", \"rank\": %u, \"dims\": [",e->rank);for(d=0;d<e->rank;++d)fprintf(o,"%s%" PRIu64,d?", ":"",e->dims[d]);fprintf(o,"], \"offset\": %" PRIu64 ", \"byte_span\": %" PRIu64 ", \"file_offset\": %" PRIu64,e->offset,e->span,e->file_offset);fprintf(o,", \"block_count\": %" PRIu64 ", \"decoded_value_count\": %" PRIu64 ", \"decoded_finite_count\": %" PRIu64 ", \"decoded_min\": %.9g, \"decoded_max\": %.9g",r->block_count,r->decoded_count,r->finite_count,r->decoded_min,r->decoded_max);fputs(", \"dequant_f32le_sha256\": ",o);strat01_json_string(o,r->dequant_sha256);fputs(", \"output_path\": ",o);strat01_json_string(o,r->output_path);fputs(", \"output_f32le_sha256\": ",o);strat01_json_string(o,r->output_sha256);fprintf(o,", \"output_finite_count\": %" PRIu64 ", \"output_min\": %.9g, \"output_max\": %.9g}",r->output_finite_count,r->output_min,r->output_max);}
    fputs("\n  ]\n}\n",o);fclose(o);return 1;
}
static int strat01_gguf_rung1_cli(const char *path, const char *out_dir, const char *engine_source_path) {
    strat01_inventory in; uint64_t parsed_size=0,hashed_size=0,expected_size=STRAT01_EXPECTED_SIZE; char error[256]={0}, artifact_sha[65]={0}, engine_sha[65]={0}, rung1_sha[65]={0}; const char *expected_sha=STRAT01_EXPECTED_SHA256; strat01_rung1_cell_result cells[3]; unsigned i;
#if FLT_RADIX != 2
    fprintf(stderr,"STRAT-01 rung-1 refused: non-binary floating point host\n"); return 1;
#endif
    if(sizeof(float)!=4U){fprintf(stderr,"STRAT-01 rung-1 refused: host float is not binary32-sized\n");return 1;}
#if defined(STRAT01_ENABLE_TEST_IDENTITY_OVERRIDE)
    expected_size=(uint64_t)STRAT01_TEST_EXPECTED_SIZE; expected_sha=STRAT01_TEST_EXPECTED_SHA256;
#endif
    memset(&in,0,sizeof(in));
    if(!strat01_parse_gguf(path,&in,&parsed_size,error) || parsed_size!=expected_size || !strat01_sha256_file(path,artifact_sha,&hashed_size,error) || hashed_size!=parsed_size || strcmp(artifact_sha,expected_sha)) {
        if(!error[0])snprintf(error,sizeof(error),"frozen rung-1 artifact identity mismatch");strat01_free_inventory(&in);strat01_rung1_write_failure(out_dir,path,error);fprintf(stderr,"STRAT-01 rung-1 refused: %s\n",error);return 1;
    }
    memset(cells,0,sizeof(cells));
    for(i=0;i<3U;++i){const strat01_tensor*t=NULL;cells[i].expected=&strat01_rung1_expected_cells[i];if(!strat01_rung1_check_descriptor(&in,cells[i].expected,&t,error)||!strat01_rung1_process_cell(path,t,&cells[i],out_dir,error)){strat01_free_inventory(&in);strat01_rung1_write_failure(out_dir,path,error);fprintf(stderr,"STRAT-01 rung-1 refused: %s\n",error);return 1;}}
    if(!strat01_sha256_file(engine_source_path,engine_sha,&hashed_size,error))strcpy(engine_sha,"unavailable");error[0]=0;
    if(!strat01_sha256_file(__FILE__,rung1_sha,&hashed_size,error))strcpy(rung1_sha,"unavailable");
    if(!strat01_rung1_write_success(out_dir,path,parsed_size,artifact_sha,engine_sha,rung1_sha,cells)){strat01_free_inventory(&in);fprintf(stderr,"STRAT-01 rung-1 refused: cannot write report\n");return 1;}
    strat01_free_inventory(&in);fprintf(stderr,"STRAT-01 rung-1: ENGINE_OUTPUT_READY_PENDING_REFERENCE\n");return 0;
}
/* A no-file regression instrument for the three byte layouts.  Its expected
 * points are selected to hit both nibbles, Q5 high-bit positions, Q4 packed
 * high scale planes, and all Q6 high-bit planes. */
static int strat01_gguf_rung1_selftest(void) {
    uint8_t q4[144]={0},q5[22]={0},q6[210]={0};float out[256];int bad=0;
    strat01_inventory inventory; strat01_tensor tensor; const strat01_tensor *found=NULL; char error[256]={0};
    q4[0]=0; q4[1]=0x3c; q4[2]=0; q4[3]=0x38; q4[4]=0xc1; q4[5]=2; q4[6]=3; q4[7]=4; q4[8]=0xc5; q4[9]=6; q4[10]=7; q4[11]=8; q4[12]=0xc9; q4[13]=0xda; q4[14]=0xeb; q4[15]=0xfc; q4[16]=0x1f; q4[48]=0xe2; q4[80]=0x0f; q4[112]=0x4b; strat01_rung1_decode_q4_k(q4,out); if(out[0]!=12.5f||out[32]!=-1.0f||out[64]!=2.5f||out[128]!=825.0f||out[192]!=114.0f)++bad;
    q5[1]=0x3c;q5[2]=1;q5[3]=2;q5[4]=4;q5[5]=0x80;q5[6]=0x10;q5[7]=0x2f;q5[21]=0xf1;strat01_rung1_decode_q5_0(q5,out);if(out[0]!=0.0f||out[1]!=-1.0f||out[16]!=-15.0f||out[31]!=15.0f)++bad;
    q6[0]=0x10;q6[32]=0x32;q6[128]=0xe4;q6[192]=1;q6[194]=2;q6[196]=3;q6[198]=4;q6[208]=0;q6[209]=0x3c;strat01_rung1_decode_q6_k(q6,out);if(out[0]!=-32.0f||out[32]!=-28.0f||out[64]!=3.0f||out[96]!=76.0f)++bad;
    memset(&inventory,0,sizeof(inventory)); memset(&tensor,0,sizeof(tensor));
    inventory.tensor_count=1; inventory.tensors=&tensor; tensor.name=(char *)STRAT01_RUNG1_Q4_NAME;
    tensor.type=strat01_rung1_expected_cells[0].type; tensor.rank=strat01_rung1_expected_cells[0].rank;
    memcpy(tensor.dims,strat01_rung1_expected_cells[0].dims,sizeof(tensor.dims));
    tensor.offset=strat01_rung1_expected_cells[0].offset; tensor.span=strat01_rung1_expected_cells[0].span; tensor.file_offset=strat01_rung1_expected_cells[0].file_offset;
    if(!strat01_rung1_check_descriptor(&inventory,&strat01_rung1_expected_cells[0],&found,error))++bad;
    tensor.type=STRAT01_GGML_Q6_K; error[0]=0; if(strat01_rung1_check_descriptor(&inventory,&strat01_rung1_expected_cells[0],&found,error))++bad;
    tensor.type=strat01_rung1_expected_cells[0].type; ++tensor.dims[0]; error[0]=0; if(strat01_rung1_check_descriptor(&inventory,&strat01_rung1_expected_cells[0],&found,error))++bad;
    --tensor.dims[0]; tensor.name=(char *)"wrong.tensor"; error[0]=0; if(strat01_rung1_check_descriptor(&inventory,&strat01_rung1_expected_cells[0],&found,error))++bad;
    fprintf(stderr,"STRAT-01 rung-1 codec selftest: %s\n",bad?"FAIL":"PASS");return bad?1:0;
}

#endif /* STRAT01_GGUF_RUNG1_H */
