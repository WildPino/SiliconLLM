#ifndef STRAT01_GGUF_POST_F16_BLOCK0_SWIGLU_SSE2_SEMANTICS_H
#define STRAT01_GGUF_POST_F16_BLOCK0_SWIGLU_SSE2_SEMANTICS_H

/*
 * Diagnostic consumer of the shared pinned no-FMA four-lane SSE2 SwiGLU
 * primitive. Frozen source hashes and the scientific scope are recorded in
 * the 2026-09-24 protocol.
 */
#define STRAT01_SSE2_REF_REVISION "5b335f413e4f73b0809c4fe39af894efbcc6a0d2"
#define STRAT01_SSE2_VEC_CPP_SHA "a946fee202dfe4528453865a6402d13a004e31ea73e88c3586793bbcc05994f7"
#define STRAT01_SSE2_VEC_H_SHA "8817801355b20079de39fd4c67c7453ca2033cdb69fc5bd1f71bb66f12f57318"
#define STRAT01_SSE2_PREDECESSOR_SHA "ad03099c59a2bb1ee30bb0063c2b4b9f393b1551ec0e19931d0bf518dd1095eb"
#define STRAT01_SSE2_REF_GATE_SHA STRAT01_P16FFN_REF_GATE_SHA
#define STRAT01_SSE2_REF_UP_SHA STRAT01_P16FFN_REF_UP_SHA
#define STRAT01_SSE2_REF_SW_SHA STRAT01_P16FFN_REF_SW_SHA
#define STRAT01_SSE2_REF_INP_SHA STRAT01_P16FFN_REF_INP_SHA
#define STRAT01_SSE2_SCALAR_SW_SHA STRAT01_P16FFN_EXPR_SW_SHA
#define STRAT01_SSE2_SCALAR_OUT_SHA STRAT01_P16FFN_REPLAY_OUT_SHA
#define STRAT01_SSE2_SCALAR_SUM_SHA STRAT01_P16FFN_REPLAY_SUM_SHA

typedef struct {
    const char *name;
    float *swiglu;
    float *out;
    float *sum;
    strat01_r2a_arm attention;
    strat01_r2c_moe_arm moe;
    strat01_r2c_dump dumps[STRAT01_R2C_DUMPS];
    char sw_path[1024], sw_sha[65];
    char out_path[1024], out_sha[65];
    char sum_path[1024], sum_sha[65];
} strat01_sse2_arm;

static int strat01_sse2_write_failure(const char *out_dir, const char *error) {
    char path[1024];
    FILE *f;
    if (!strat01_r2a_path(path, out_dir, "strat01_post_f16_block0_swiglu_sse2_semantics.json") ||
        (f = fopen(path, "wb")) == NULL) {
        return 0;
    }
    fputs("{\"command\":\"--strat01-post-f16-block0-swiglu-sse2-semantics\",\"state\":\"DIAGNOSTIC_FAILURE\",\"error\":", f);
    strat01_json_string(f, error);
    fputs(",\"donor_graph_executions\":0,\"reference_graph_executions\":0}\n", f);
    fclose(f);
    return 1;
}

static int strat01_sse2_run_arm(
        const char *model, const strat01_tensor *down, const strat01_tensor *attn[7],
        const strat01_tensor *moe[9], const float *r_inp, const char *out_dir,
        strat01_sse2_arm *arm, char error[256]) {
    char leaf[192];
    if (!strat01_r2b_q6_matmul_batch(model, down, arm->swiglu, 8U, arm->out, error)) return 0;
    strat01_down_sha(arm->swiglu, STRAT01_DOWN_SWIGLU_COUNT, arm->sw_sha);
    strat01_down_sha(arm->out, STRAT01_R2C_L1START_INPUT_COUNT, arm->out_sha);
    strat01_down_sum(r_inp, arm->out, arm->sum);
    strat01_down_sha(arm->sum, STRAT01_R2C_L1START_INPUT_COUNT, arm->sum_sha);
    if (!strat01_postf16_run_full(model, attn, moe, arm->sum, &arm->attention, &arm->moe, error) ||
        !strat01_postf16_write_arm(out_dir, arm->name, arm->sum, &arm->attention, &arm->moe, arm->dumps, error)) return 0;
    snprintf(leaf, sizeof(leaf), "%s_ffn_swiglu-0.f32le", arm->name);
    if (!strat01_r2a_path(arm->sw_path, out_dir, leaf) ||
        !strat01_q4q8_write_output(arm->sw_path, arm->swiglu, STRAT01_DOWN_SWIGLU_COUNT, arm->sw_sha, error)) return 0;
    snprintf(leaf, sizeof(leaf), "%s_ffn_out-0.f32le", arm->name);
    if (!strat01_r2a_path(arm->out_path, out_dir, leaf) ||
        !strat01_q4q8_write_output(arm->out_path, arm->out, STRAT01_R2C_L1START_INPUT_COUNT, arm->out_sha, error)) return 0;
    snprintf(leaf, sizeof(leaf), "%s_l_out-0.f32le", arm->name);
    return strat01_r2a_path(arm->sum_path, out_dir, leaf) &&
           strat01_q4q8_write_output(arm->sum_path, arm->sum, STRAT01_R2C_L1START_INPUT_COUNT, arm->sum_sha, error);
}

static void strat01_sse2_report_payload(
        FILE *o, const char *path, uint64_t bytes, const char *sha) {
    fputs("{\"path\":", o);
    strat01_json_string(o, path);
    fprintf(o, ",\"bytes\":%" PRIu64 ",\"sha256\":", bytes);
    strat01_json_string(o, sha);
    fputc('}', o);
}

static void strat01_sse2_report_arm(FILE *o, const strat01_sse2_arm *arm) {
    fputs("{\"swiglu\":", o);
    strat01_sse2_report_payload(o, arm->sw_path, STRAT01_DOWN_SWIGLU_BYTES, arm->sw_sha);
    fputs(",\"ffn_out\":", o);
    strat01_sse2_report_payload(o, arm->out_path, STRAT01_R2C_L1START_INPUT_BYTES, arm->out_sha);
    fputs(",\"l_out\":", o);
    strat01_sse2_report_payload(o, arm->sum_path, STRAT01_R2C_L1START_INPUT_BYTES, arm->sum_sha);
    fputs(",\"checkpoints\":", o);
    strat01_postf16_report_arm(o, (strat01_r2c_dump *) arm->dumps);
    fputc('}', o);
}

static int strat01_sse2_cli(
        const char *model, const char *r_gate_path, const char *r_up_path,
        const char *r_sw_path, const char *r_inp_path, const char *out_dir,
        const char *engine_source_path) {
    strat01_inventory inv;
    const strat01_tensor *down = NULL, *attn[7] = {0}, *moe[9] = {0};
    float *r_gate = NULL, *r_up = NULL, *r_sw = NULL, *r_inp = NULL;
    float *ctl_gate = NULL, *ctl_up = NULL;
    strat01_sse2_arm arms[2] = {
        {"scalar_libm_replay", NULL, NULL, NULL, {0}, {0}, {{0}}, {0}, {0}, {0}, {0}, {0}, {0}},
        {"pinned_sse2_candidate", NULL, NULL, NULL, {0}, {0}, {{0}}, {0}, {0}, {0}, {0}, {0}, {0}}
    };
    strat01_sse2_arm controls[2] = {
        {"sse2_gate_token6_negated", NULL, NULL, NULL, {0}, {0}, {{0}}, {0}, {0}, {0}, {0}, {0}, {0}},
        {"sse2_up_rows0_7_swapped", NULL, NULL, NULL, {0}, {0}, {{0}}, {0}, {0}, {0}, {0}, {0}, {0}}
    };
    const char *paths[4] = {r_gate_path, r_up_path, r_sw_path, r_inp_path};
    const char *names[4] = {"reference_gate", "reference_up", "reference_swiglu", "reference_ffn_inp"};
    const char *expected[4] = {STRAT01_SSE2_REF_GATE_SHA, STRAT01_SSE2_REF_UP_SHA, STRAT01_SSE2_REF_SW_SHA, STRAT01_SSE2_REF_INP_SHA};
    uint64_t counts[4] = {STRAT01_DOWN_SWIGLU_COUNT, STRAT01_DOWN_SWIGLU_COUNT, STRAT01_DOWN_SWIGLU_COUNT, STRAT01_R2C_L1START_INPUT_COUNT};
    float **dst[4] = {&r_gate, &r_up, &r_sw, &r_inp};
    char input_sha[4][65] = {{0}};
    char error[256] = {0}, model_sha[65] = {0}, engine_sha[65] = {0}, header_sha[65] = {0}, report_path[1024];
    uint64_t hashed = 0, parsed = 0, tmp = 0;
    FILE *report = NULL;
    int ok = 0;

    memset(&inv, 0, sizeof(inv));
    strat01_f16vec_reset_counts();
#if !defined(__clang__) || !defined(__SSE2__)
    snprintf(error, 256, "SSE2 semantics diagnostic requires Clang SSE2");
    goto finish;
#endif
    if (!strat01_sha256_file(model, model_sha, &hashed, error) || hashed != STRAT01_EXPECTED_SIZE || strcmp(model_sha, STRAT01_EXPECTED_SHA256)) {
        if (!error[0]) snprintf(error, 256, "SSE2 semantics artifact mismatch");
        goto finish;
    }
    if (!strat01_parse_gguf(model, &inv, &parsed, error) || parsed != hashed) {
        if (!error[0]) snprintf(error, 256, "SSE2 semantics GGUF parse mismatch");
        goto finish;
    }
    if (!strat01_r2a_check_spec(&inv, &strat01_r2b_specs[3], &down, error)) goto finish;
    for (unsigned i = 0; i < 7U; ++i) if (!strat01_r2a_check_spec(&inv, &strat01_r2c_attn_specs[i], &attn[i], error)) goto finish;
    for (unsigned i = 0; i < 9U; ++i) if (!strat01_r2a_check_spec(&inv, &strat01_r2c_moe_specs[i], &moe[i], error)) goto finish;
    for (unsigned i = 0; i < 4U; ++i) {
        *dst[i] = strat01_r2a_alloc((size_t) counts[i], error);
        if (!*dst[i] || !strat01_down_read(paths[i], expected[i], counts[i], *dst[i], input_sha[i], error)) goto finish;
    }
    ctl_gate = strat01_r2a_alloc(STRAT01_DOWN_SWIGLU_COUNT, error);
    ctl_up = strat01_r2a_alloc(STRAT01_DOWN_SWIGLU_COUNT, error);
    if (!ctl_gate || !ctl_up) goto finish;
    for (unsigned group = 0; group < 2U; ++group) for (unsigned i = 0; i < 2U; ++i) {
        strat01_sse2_arm *a = group ? &controls[i] : &arms[i];
        a->swiglu = strat01_r2a_alloc(STRAT01_DOWN_SWIGLU_COUNT, error);
        a->out = strat01_r2a_alloc(STRAT01_R2C_L1START_INPUT_COUNT, error);
        a->sum = strat01_r2a_alloc(STRAT01_R2C_L1START_INPUT_COUNT, error);
        if (!a->swiglu || !a->out || !a->sum || !strat01_r2a_arm_alloc(&a->attention, error) || !strat01_r2c_moe_alloc(&a->moe, error)) goto finish;
    }
    if (!strat01_sw_compute(r_gate, r_up, arms[0].swiglu, STRAT01_DOWN_SWIGLU_COUNT, error) ||
        !strat01_sse2_swiglu_compute(r_gate, r_up, arms[1].swiglu, STRAT01_DOWN_SWIGLU_COUNT, error)) goto finish;
    memcpy(ctl_gate, r_gate, STRAT01_DOWN_SWIGLU_BYTES);
    memcpy(ctl_up, r_up, STRAT01_DOWN_SWIGLU_BYTES);
    for (size_t i = 6U * STRAT01_R2B_FFN; i < 7U * STRAT01_R2B_FFN; ++i) ctl_gate[i] = -ctl_gate[i];
    for (unsigned i = 0; i < STRAT01_R2B_FFN; ++i) {
        float x = ctl_up[i];
        ctl_up[i] = ctl_up[7U * STRAT01_R2B_FFN + i];
        ctl_up[7U * STRAT01_R2B_FFN + i] = x;
    }
    if (!strat01_sse2_swiglu_compute(ctl_gate, r_up, controls[0].swiglu, STRAT01_DOWN_SWIGLU_COUNT, error) ||
        !strat01_sse2_swiglu_compute(r_gate, ctl_up, controls[1].swiglu, STRAT01_DOWN_SWIGLU_COUNT, error)) goto finish;
    for (unsigned i = 0; i < 2U; ++i) if (!strat01_sse2_run_arm(model, down, attn, moe, r_inp, out_dir, &arms[i], error)) goto finish;
    for (unsigned i = 0; i < 2U; ++i) if (!strat01_sse2_run_arm(model, down, attn, moe, r_inp, out_dir, &controls[i], error)) goto finish;
    if (strcmp(arms[0].sw_sha, STRAT01_SSE2_SCALAR_SW_SHA) ||
        strcmp(arms[0].out_sha, STRAT01_SSE2_SCALAR_OUT_SHA) ||
        strcmp(arms[0].sum_sha, STRAT01_SSE2_SCALAR_SUM_SHA)) {
        snprintf(error, 256, "scalar-libm replay mismatch");
        goto finish;
    }
    if (!strat01_f16v_write_counts(out_dir, error) ||
        !strat01_sha256_file(engine_source_path, engine_sha, &tmp, error) ||
        !strat01_sha256_file(__FILE__, header_sha, &tmp, error)) goto finish;
    if (!strat01_r2a_path(report_path, out_dir, "strat01_post_f16_block0_swiglu_sse2_semantics.json") ||
        (report = fopen(report_path, "wb")) == NULL) {
        snprintf(error, 256, "cannot write SSE2 semantics report");
        goto finish;
    }
    fputs("{\n\"command\":\"--strat01-post-f16-block0-swiglu-sse2-semantics\",\n\"state\":\"OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\",\n\"self_certifies_pass\":false,\n\"model\":{\"path\":", report);
    strat01_json_string(report, model);
    fprintf(report, ",\"bytes\":%" PRIu64 ",\"sha256\":", hashed);
    strat01_json_string(report, model_sha);
    fputs("},\n\"inputs\":{", report);
    for (unsigned i = 0; i < 4U; ++i) {
        fprintf(report, "%s\"%s\":{\"path\":", i ? "," : "", names[i]);
        strat01_json_string(report, paths[i]);
        fprintf(report, ",\"bytes\":%" PRIu64 ",\"sha256\":", counts[i] * 4U);
        strat01_json_string(report, input_sha[i]);
        fputc('}', report);
    }
    fputs("},\n\"semantics\":{\"reference_revision\":\"" STRAT01_SSE2_REF_REVISION "\",\"vec_cpp_sha256\":\"" STRAT01_SSE2_VEC_CPP_SHA "\",\"vec_h_sha256\":\"" STRAT01_SSE2_VEC_H_SHA "\",\"lane_width\":4,\"fma\":false,\"scalar_tail\":false},\n\"down_tensor\":{\"name\":", report);
    strat01_json_string(report, down->name);
    fprintf(report, ",\"type\":%u,\"offset\":%" PRIu64 ",\"file_offset\":%" PRIu64 ",\"span\":%" PRIu64 "},\n\"layer1_tensors\":[", down->type, down->offset, down->file_offset, down->span);
    for (unsigned i = 0; i < 16U; ++i) {
        const strat01_tensor *t = i < 7U ? attn[i] : moe[i - 7U];
        fprintf(report, "%s{\"name\":", i ? "," : "");
        strat01_json_string(report, t->name);
        fprintf(report, ",\"type\":%u,\"offset\":%" PRIu64 ",\"file_offset\":%" PRIu64 ",\"span\":%" PRIu64 "}", t->type, t->offset, t->file_offset, t->span);
    }
    fputs("],\n\"arms\":{\"scalar_libm_replay\":", report);
    strat01_sse2_report_arm(report, &arms[0]);
    fputs(",\"pinned_sse2_candidate\":", report);
    strat01_sse2_report_arm(report, &arms[1]);
    fputs("},\n\"controls\":{\"sse2_gate_token6_negated\":", report);
    strat01_sse2_report_arm(report, &controls[0]);
    fputs(",\"sse2_up_rows0_7_swapped\":", report);
    strat01_sse2_report_arm(report, &controls[1]);
    fputs("},\n\"q6_down_arms\":4,\n\"engine_source_sha256\":", report);
    strat01_json_string(report, engine_sha);
    fputs(",\n\"diagnostic_source_sha256\":", report);
    strat01_json_string(report, header_sha);
    fputs(",\n\"compiler_family\":\"clang\",\n\"donor_graph_executions\":0,\n\"reference_graph_executions\":0,\n\"timing_or_rate_claim\":null\n}\n", report);
    if (fclose(report) != 0) {
        report = NULL;
        snprintf(error, 256, "SSE2 semantics report close failure");
        goto finish;
    }
    report = NULL;
    ok = 1;

finish:
    if (report) fclose(report);
    for (unsigned i = 0; i < 4U; ++i) free(*dst[i]);
    free(ctl_gate);
    free(ctl_up);
    for (unsigned group = 0; group < 2U; ++group) for (unsigned i = 0; i < 2U; ++i) {
        strat01_sse2_arm *a = group ? &controls[i] : &arms[i];
        free(a->swiglu); free(a->out); free(a->sum);
        strat01_r2a_arm_free(&a->attention);
        strat01_r2c_moe_free(&a->moe);
    }
    strat01_free_inventory(&inv);
    if (!ok) {
        if (!error[0]) snprintf(error, 256, "unspecified SSE2 semantics failure");
        strat01_sse2_write_failure(out_dir, error);
        fprintf(stderr, "STRAT-01 post-F16 SwiGLU SSE2 semantics refused: %s\n", error);
        return 1;
    }
    fprintf(stderr, "STRAT-01 post-F16 SwiGLU SSE2 semantics: OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\n");
    return 0;
}

static int strat01_sse2_selftest(void) {
    int bad = 0, checks = 0;
    float gate[8] = {0.1f, 1.2345f, -3.21f, 7.8f, -0.25f, 0.75f, 2.5f, -6.0f};
    float up[8] = {1.0f, -2.0f, 3.0f, 0.5f, 4.0f, -1.0f, 0.25f, 2.0f};
    float sse2[8] = {0}, scalar[8] = {0}, mutated[8] = {0};
    char error[256] = {0};
#define SSE2_CHECK(x) do { ++checks; if (!(x)) ++bad; } while (0)
    SSE2_CHECK(strlen(STRAT01_SSE2_REF_REVISION) == 40U);
    SSE2_CHECK(strlen(STRAT01_SSE2_VEC_CPP_SHA) == 64U);
    SSE2_CHECK(strlen(STRAT01_SSE2_VEC_H_SHA) == 64U);
    SSE2_CHECK(!strat01_sse2_swiglu_compute(gate, up, sse2, 7U, error));
    error[0] = '\0';
    SSE2_CHECK(strat01_sse2_swiglu_compute(gate, up, sse2, 8U, error));
    SSE2_CHECK(strat01_sw_compute(gate, up, scalar, 8U, error));
    SSE2_CHECK(memcmp(sse2, scalar, sizeof(sse2)) != 0);
    memcpy(mutated, gate, sizeof(gate)); mutated[3] = -mutated[3];
    SSE2_CHECK(strat01_sse2_swiglu_compute(mutated, up, mutated, 8U, error));
    SSE2_CHECK(memcmp(sse2, mutated, sizeof(sse2)) != 0);
    SSE2_CHECK(STRAT01_DOWN_SWIGLU_COUNT % 4U == 0U);
    fprintf(stderr, "STRAT-01 post-F16 SwiGLU SSE2 semantics selftest: %s (%d checks)\n", bad ? "FAIL" : "PASS", checks);
#undef SSE2_CHECK
    return bad ? 1 : 0;
}

#endif
