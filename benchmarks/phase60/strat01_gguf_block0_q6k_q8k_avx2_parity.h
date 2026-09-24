/* Block-0 Q6_K/Q8_K full-matrix parity diagnostic; zero model graphs. */
#ifndef STRAT01_GGUF_BLOCK0_Q6K_Q8K_AVX2_PARITY_H
#define STRAT01_GGUF_BLOCK0_Q6K_Q8K_AVX2_PARITY_H

#define STRAT01_Q6P_INPUT_SHA "de9245a2f6f60e8d7511d03fe981e6b55ef21a8634d5071dd36727a1a5950cef"
#define STRAT01_Q6P_FFN_INPUT_SHA "baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1"
#define STRAT01_Q6P_BATCH 8U
#define STRAT01_Q6P_ROWS 1536U
#define STRAT01_Q6P_WIDTH 8960U

static int strat01_q6p_descriptor_ok(const strat01_tensor *tensor) {
    return tensor && !strcmp(tensor->name, "blk.0.ffn_down.weight") &&
           tensor->type == STRAT01_GGML_Q6_K && tensor->rank == 2U &&
           tensor->dims[0] == STRAT01_Q6P_WIDTH && tensor->dims[1] == STRAT01_Q6P_ROWS &&
           tensor->offset == 286755840ULL && tensor->file_offset == 292858752ULL &&
           tensor->span == 11289600ULL;
}

static int strat01_q6p_write_bytes(
        const char *dir, const char *leaf, const void *data, size_t bytes,
        char path[1024], char sha[65], char error[256]) {
    FILE *stream;
    uint64_t observed = 0;
    if (!strat01_r2a_path(path, dir, leaf) || (stream = fopen(path, "wb")) == NULL) {
        snprintf(error, 256, "cannot create Q6 parity output");
        return 0;
    }
    if (fwrite(data, 1, bytes, stream) != bytes || fclose(stream) != 0) {
        snprintf(error, 256, "cannot write Q6 parity output");
        return 0;
    }
    if (!strat01_sha256_file(path, sha, &observed, error) || observed != bytes) return 0;
    return 1;
}

static int strat01_q6p_write_f32(
        const char *dir, const char *leaf, const float *data, size_t count,
        char path[1024], char sha[65], char error[256]) {
    if (!strat01_r2a_path(path, dir, leaf)) return 0;
    return strat01_q4q8_write_output(path, data, count, sha, error);
}

static int strat01_q6p_write_failure(
        const char *dir, const char *error, int reference_generic) {
    char path[1024];
    FILE *stream;
    const char *leaf = reference_generic
        ? "strat01_block0_q6k_q8k_reference_generic_parity.json"
        : "strat01_block0_q6k_q8k_avx2_parity.json";
    const char *command = reference_generic
        ? "--strat01-block0-q6k-q8k-reference-generic-parity"
        : "--strat01-block0-q6k-q8k-avx2-parity";
    if (!strat01_r2a_path(path, dir, leaf) ||
        (stream = fopen(path, "wb")) == NULL) return 0;
    fputs("{\"command\":", stream); strat01_json_string(stream, command);
    fputs(",\"state\":\"DIAGNOSTIC_FAILURE\",\"error\":", stream);
    strat01_json_string(stream, error);
    fputs(",\"q8_populations_completed\":0,\"q6_matrix_reads\":0,"
          "\"diagnostic_executions\":1,\"donor_graph_executions\":0,"
          "\"reference_graph_executions\":0}\n", stream);
    fclose(stream);
    return 1;
}

static int strat01_q6p_cli_mode(
        const char *model, const char *input_path, const char *ffn_input_path,
        const char *topk_path, const char *ref_kqv_path, const char *ref_layer1_ffn_path,
        const char *ref_gate_path, const char *ref_q_path, const char *ref_k_path,
        const char *ref_shared_path, const char *ref_weights_path,
        const char *out_dir, const char *engine_source_path,
        int reference_generic) {
    strat01_inventory inventory;
    const strat01_tensor *matrix = NULL, *wo = NULL, *ffn_norm = NULL;
    const strat01_tensor *up_experts = NULL, *down_experts = NULL;
    const strat01_tensor *attn_norm = NULL, *kv_a = NULL, *kv_a_norm = NULL, *v_b = NULL;
    const unsigned blocks = STRAT01_Q6P_WIDTH / STRAT01_QK_K;
    const size_t row_bytes = (size_t)blocks * STRAT01_Q6_K_BLOCK_BYTES;
    const size_t q8_count = (size_t)STRAT01_Q6P_BATCH * blocks;
    const size_t output_count = (size_t)STRAT01_Q6P_BATCH * STRAT01_Q6P_ROWS;
    float *input = NULL, *ffn_input = NULL, *generic = NULL, *candidate = NULL;
    float *active_control = NULL;
    float *generic_lout = NULL, *candidate_lout = NULL;
    float *ref_kqv = NULL, *ref_layer1_ffn = NULL, *ref_gate = NULL, *ref_q = NULL;
    float *ref_k = NULL, *ref_shared = NULL, *ref_weights = NULL;
    float *projection = NULL, *layer1_ffn = NULL, *norm_weight = NULL, *norm = NULL;
    float *up = NULL, *swiglu = NULL, *down = NULL, *weighted = NULL, *moe = NULL;
    float *ffn_out = NULL, *layer1_lout = NULL, *attn_weight = NULL, *layer2_norm = NULL;
    float *kv_projection = NULL, *kv_norm_weight = NULL, *prefix = NULL, *k = NULL;
    float *downstream = NULL;
    int32_t topk[32];
    strat01_q8_k_block *q8 = NULL;
    uint8_t *raw = NULL, *mutated = NULL;
    float mutation[STRAT01_Q6P_BATCH] = {0};
    FILE *weights = NULL, *report = NULL;
    char error[256] = {0}, model_sha[65] = {0}, input_sha[65] = {0};
    char ffn_input_sha[65] = {0}, engine_sha[65] = {0}, header_sha[65] = {0};
    char q8_path[1024], q8_sha[65], generic_path[1024], generic_sha[65];
    char candidate_path[1024], candidate_sha[65], generic_lout_path[1024], generic_lout_sha[65];
    char candidate_lout_path[1024], candidate_lout_sha[65], mutation_path[1024], mutation_sha[65];
    char active_path[1024], active_sha[65], downstream_path[1024], downstream_sha[65], topk_sha[65];
    char extra_sha[7][65] = {{0}};
    char report_path[1024];
    uint64_t hashed = 0, parsed = 0, temporary = 0;
    int ok = 0;
    const char *command_name = reference_generic
        ? "--strat01-block0-q6k-q8k-reference-generic-parity"
        : "--strat01-block0-q6k-q8k-avx2-parity";
    const char *report_leaf = reference_generic
        ? "strat01_block0_q6k_q8k_reference_generic_parity.json"
        : "strat01_block0_q6k_q8k_avx2_parity.json";
    const char *generic_name = reference_generic
        ? "historical_avx_tu_generic" : "current_generic_replay";
    const char *generic_leaf = reference_generic
        ? "historical_avx_tu_generic.f32le" : "current_generic_replay.f32le";
    const char *candidate_name = reference_generic
        ? "reference_generic_candidate" : "pinned_avx2_candidate";
    const char *candidate_leaf = reference_generic
        ? "reference_generic_candidate.f32le" : "pinned_avx2_candidate.f32le";
    const char *generic_lout_name = reference_generic
        ? "historical_avx_tu_generic_lout" : "current_generic_lout";
    const char *generic_lout_leaf = reference_generic
        ? "historical_avx_tu_generic_lout.f32le" : "current_generic_lout.f32le";
    const char *candidate_lout_name = reference_generic
        ? "reference_generic_candidate_lout" : "pinned_avx2_candidate_lout";
    const char *candidate_lout_leaf = reference_generic
        ? "reference_generic_candidate_lout.f32le" : "pinned_avx2_candidate_lout.f32le";
    const char *downstream_name = reference_generic
        ? "reference_generic_candidate_downstream" : "pinned_avx2_candidate_downstream";
    const char *downstream_leaf = reference_generic
        ? "reference_generic_candidate_downstream.f32le" : "pinned_avx2_candidate_downstream.f32le";
    memset(&inventory, 0, sizeof(inventory));
#if !defined(__clang__) || !defined(__AVX2__) || !defined(__FMA__)
    snprintf(error, 256, "Q6 parity diagnostic requires Clang AVX2/FMA");
    goto finish;
#endif
    if (!strat01_sha256_file(model, model_sha, &hashed, error) ||
        hashed != STRAT01_EXPECTED_SIZE || strcmp(model_sha, STRAT01_EXPECTED_SHA256)) {
        if (!error[0]) snprintf(error, 256, "Q6 parity artifact identity mismatch");
        goto finish;
    }
    if (!strat01_parse_gguf(model, &inventory, &parsed, error) || parsed != hashed) goto finish;
    matrix = strat01_rung1_find_tensor(&inventory, "blk.0.ffn_down.weight");
    if (!strat01_q6p_descriptor_ok(matrix)) {
        snprintf(error, 256, "Q6 parity matrix descriptor mismatch");
        goto finish;
    }
    wo = strat01_rung1_find_tensor(&inventory, "blk.1.attn_output.weight");
    ffn_norm = strat01_rung1_find_tensor(&inventory, "blk.1.ffn_norm.weight");
    attn_norm = strat01_rung1_find_tensor(&inventory, "blk.2.attn_norm.weight");
    kv_a = strat01_rung1_find_tensor(&inventory, "blk.2.attn_kv_a_mqa.weight");
    kv_a_norm = strat01_rung1_find_tensor(&inventory, "blk.2.attn_kv_a_norm.weight");
    v_b = strat01_rung1_find_tensor(&inventory, "blk.2.attn_v_b.weight");
    if (!strat01_l1ao_descriptor_ok(wo) || !strat01_l1fr_norm_descriptor_ok(ffn_norm) ||
        !strat01_r2a_check_spec(&inventory, &strat01_r2c_moe_specs[3], &up_experts, error) ||
        !strat01_r2a_check_spec(&inventory, &strat01_r2c_moe_specs[5], &down_experts, error) ||
        !strat01_l2an_norm_descriptor_ok(attn_norm) ||
        !strat01_l2kva_projection_descriptor_ok(kv_a) ||
        !strat01_l2rn_norm_descriptor_ok(kv_a_norm) ||
        !strat01_l2rn_vb_descriptor_ok(v_b)) {
        if (!error[0]) snprintf(error, 256, "Q6 parity downstream descriptor mismatch");
        goto finish;
    }
#define Q6P_ALLOC(pointer, count) do { pointer = strat01_r2a_alloc((count), error); if (!pointer) goto finish; } while (0)
    Q6P_ALLOC(input, STRAT01_Q6P_BATCH * STRAT01_Q6P_WIDTH);
    Q6P_ALLOC(ffn_input, output_count);
    Q6P_ALLOC(generic, output_count);
    Q6P_ALLOC(candidate, output_count);
    if (reference_generic) Q6P_ALLOC(active_control, output_count);
    Q6P_ALLOC(generic_lout, output_count);
    Q6P_ALLOC(candidate_lout, output_count);
    Q6P_ALLOC(ref_kqv, STRAT01_Q6P_BATCH * 6144U);
    Q6P_ALLOC(ref_layer1_ffn, output_count);
    Q6P_ALLOC(ref_gate, STRAT01_L1SW_COUNT);
    Q6P_ALLOC(ref_q, STRAT01_Q6P_BATCH * 32U * 576U);
    Q6P_ALLOC(ref_k, STRAT01_Q6P_BATCH * 576U);
    Q6P_ALLOC(ref_shared, output_count);
    Q6P_ALLOC(ref_weights, STRAT01_Q6P_BATCH * 4U);
    Q6P_ALLOC(projection, output_count);
    Q6P_ALLOC(layer1_ffn, output_count);
    Q6P_ALLOC(norm_weight, 1536U);
    Q6P_ALLOC(norm, output_count);
    Q6P_ALLOC(up, STRAT01_L1SW_COUNT);
    Q6P_ALLOC(swiglu, STRAT01_L1SW_COUNT);
    Q6P_ALLOC(down, STRAT01_Q6P_BATCH * 4U * 1536U);
    Q6P_ALLOC(weighted, STRAT01_Q6P_BATCH * 4U * 1536U);
    Q6P_ALLOC(moe, output_count);
    Q6P_ALLOC(ffn_out, output_count);
    Q6P_ALLOC(layer1_lout, output_count);
    Q6P_ALLOC(attn_weight, 1536U);
    Q6P_ALLOC(layer2_norm, output_count);
    Q6P_ALLOC(kv_projection, STRAT01_Q6P_BATCH * 576U);
    Q6P_ALLOC(kv_norm_weight, 512U);
    Q6P_ALLOC(prefix, STRAT01_Q6P_BATCH * 512U);
    Q6P_ALLOC(k, STRAT01_Q6P_BATCH * 576U);
    Q6P_ALLOC(downstream, STRAT01_Q6P_BATCH * 6144U);
#undef Q6P_ALLOC
    q8 = (strat01_q8_k_block *)malloc(q8_count * sizeof(*q8));
    raw = (uint8_t *)malloc(row_bytes);
    mutated = (uint8_t *)malloc(row_bytes);
    if (!q8 || !raw || !mutated) {
        snprintf(error, 256, "Q6 parity allocation failed");
        goto finish;
    }
    if (!strat01_r2c_cross_read_f32(
            input_path, STRAT01_Q6P_BATCH * STRAT01_Q6P_WIDTH * 4ULL,
            STRAT01_Q6P_INPUT_SHA, input, input_sha, error) ||
        !strat01_r2c_cross_read_f32(
            ffn_input_path, output_count * 4ULL, STRAT01_Q6P_FFN_INPUT_SHA,
            ffn_input, ffn_input_sha, error) ||
        !strat01_r2c_cross_read_f32(ref_kqv_path, 196608U, STRAT01_L1AO_KQV_SHA,
            ref_kqv, extra_sha[0], error) ||
        !strat01_r2c_cross_read_f32(ref_layer1_ffn_path, 49152U, STRAT01_L1FR_REF_INPUT_SHA,
            ref_layer1_ffn, extra_sha[1], error) ||
        !strat01_r2c_cross_read_f32(ref_gate_path, 163840U, STRAT01_L1SW_REF_GATE_SHA,
            ref_gate, extra_sha[2], error) ||
        !strat01_r2c_cross_read_f32(ref_q_path, 589824U, STRAT01_L2RN_REF_Q_SHA,
            ref_q, extra_sha[3], error) ||
        !strat01_r2c_cross_read_f32(ref_k_path, 18432U, STRAT01_L2RN_REF_K_SHA,
            ref_k, extra_sha[4], error) ||
        !strat01_r2c_cross_read_f32(ref_shared_path, 49152U, STRAT01_L1FO_REF_SHARED_SHA,
            ref_shared, extra_sha[5], error) ||
        !strat01_r2c_cross_read_f32(ref_weights_path, 128U, STRAT01_L1RM_REF_WEIGHTS_SHA,
            ref_weights, extra_sha[6], error) ||
        !strat01_l1q6_read_topk(topk_path, topk, topk_sha, error)) goto finish;
    for (unsigned item = 0; item < STRAT01_Q6P_BATCH; ++item) {
        if (!strat01_quantize_q8_k_row(
                input + (size_t)item * STRAT01_Q6P_WIDTH, STRAT01_Q6P_WIDTH,
                q8 + (size_t)item * blocks)) {
            snprintf(error, 256, "Q6 parity Q8_K quantization failed");
            goto finish;
        }
    }
    weights = fopen(model, "rb");
    if (!weights || !strat01_r2a_seek(weights, matrix->file_offset, error)) goto finish;
    for (unsigned row = 0; row < STRAT01_Q6P_ROWS; ++row) {
        if (fread(raw, 1, row_bytes, weights) != row_bytes) {
            snprintf(error, 256, "Q6 parity matrix short read");
            goto finish;
        }
        for (unsigned item = 0; item < STRAT01_Q6P_BATCH; ++item) {
            const strat01_q8_k_block *activation = q8 + (size_t)item * blocks;
            const size_t index = (size_t)item * STRAT01_Q6P_ROWS + row;
            generic[index] = strat01_q6k_q8k_dot_generic(raw, activation, STRAT01_Q6P_WIDTH);
            candidate[index] = reference_generic
                ? strat01_q6k_q8k_dot_reference_generic(raw, activation, STRAT01_Q6P_WIDTH)
                : strat01_q6k_q8k_dot_avx2(raw, activation, STRAT01_Q6P_WIDTH);
            if (reference_generic)
                active_control[index] = strat01_q6k_q8k_dot_avx2(
                    raw, activation, STRAT01_Q6P_WIDTH);
            if (!isfinite(generic[index]) || !isfinite(candidate[index]) ||
                (reference_generic && !isfinite(active_control[index]))) {
                snprintf(error, 256, "Q6 parity non-finite output");
                goto finish;
            }
        }
        if (row == 0U) {
            memcpy(mutated, raw, row_bytes);
            mutated[17] ^= 1U;
            for (unsigned item = 0; item < STRAT01_Q6P_BATCH; ++item)
                mutation[item] = reference_generic
                    ? strat01_q6k_q8k_dot_reference_generic(
                        mutated, q8 + (size_t)item * blocks, STRAT01_Q6P_WIDTH)
                    : strat01_q6k_q8k_dot_avx2(
                        mutated, q8 + (size_t)item * blocks, STRAT01_Q6P_WIDTH);
        }
    }
    if (ferror(weights) || fclose(weights) != 0) {
        weights = NULL;
        snprintf(error, 256, "Q6 parity matrix I/O failure");
        goto finish;
    }
    weights = NULL;
    for (size_t index = 0; index < output_count; ++index) {
        generic_lout[index] = ffn_input[index] + generic[index];
        candidate_lout[index] = ffn_input[index] + candidate[index];
    }
    if (!strat01_r2a_read_f32_vector(model, ffn_norm, norm_weight, 1536U, error) ||
        !strat01_r2a_read_f32_vector(model, attn_norm, attn_weight, 1536U, error) ||
        !strat01_r2a_read_f32_vector(model, kv_a_norm, kv_norm_weight, 512U, error) ||
        !strat01_r2a_matmul_batch(model, wo, ref_kqv, STRAT01_Q6P_BATCH, 6144U,
            projection, 1536U, error)) goto finish;
    strat01_l1ao_add(projection, candidate_lout, layer1_ffn);
    strat01_r2a_rmsnorm_pinned(
        layer1_ffn, norm_weight, norm, STRAT01_Q6P_BATCH, 1536U, STRAT01_R2A_RMS_EPS);
    if (!strat01_l1up_project(model, up_experts, topk, norm, up, error) ||
        !strat01_sse2_swiglu_compute(ref_gate, up, swiglu, STRAT01_L1SW_COUNT, error) ||
        !strat01_l1q6_routed(model, down_experts, topk, swiglu, down, error)) goto finish;
    strat01_l1rm_reduce(down, ref_weights, weighted, moe);
    strat01_l1fo_add(moe, ref_shared, ffn_out);
    strat01_l1tc_add(ref_layer1_ffn, ffn_out, layer1_lout);
    strat01_r2a_rmsnorm_pinned(
        layer1_lout, attn_weight, layer2_norm, STRAT01_Q6P_BATCH, 1536U,
        STRAT01_R2A_RMS_EPS);
    if (!strat01_r2a_matmul_batch(model, kv_a, layer2_norm, STRAT01_Q6P_BATCH,
            1536U, kv_projection, 576U, error)) goto finish;
    strat01_l2rn_compute_prefix(kv_projection, kv_norm_weight, prefix, 0, 0);
    strat01_l2rn_compose_k(k, ref_k, prefix);
    if (!strat01_r2c_cross_run_arm(model, v_b, ref_q, k, 0, 0, downstream, error)) goto finish;
    if (!strat01_q6p_write_bytes(out_dir, "q8_population.bin", q8,
            q8_count * sizeof(*q8), q8_path, q8_sha, error) ||
        !strat01_q6p_write_f32(out_dir, generic_leaf, generic,
            output_count, generic_path, generic_sha, error) ||
        !strat01_q6p_write_f32(out_dir, candidate_leaf, candidate,
            output_count, candidate_path, candidate_sha, error) ||
        !strat01_q6p_write_f32(out_dir, generic_lout_leaf, generic_lout,
            output_count, generic_lout_path, generic_lout_sha, error) ||
        !strat01_q6p_write_f32(out_dir, candidate_lout_leaf, candidate_lout,
            output_count, candidate_lout_path, candidate_lout_sha, error) ||
        !strat01_q6p_write_f32(out_dir, "control_mutated_q6_row.f32le", mutation,
            STRAT01_Q6P_BATCH, mutation_path, mutation_sha, error) ||
        !strat01_q6p_write_f32(out_dir, downstream_leaf,
            downstream, STRAT01_Q6P_BATCH * 6144U, downstream_path, downstream_sha, error)) goto finish;
    if (reference_generic &&
        !strat01_q6p_write_f32(out_dir, "closed_active_avx2_control.f32le", active_control,
            output_count, active_path, active_sha, error)) goto finish;
    if (!strat01_sha256_file(engine_source_path, engine_sha, &temporary, error) ||
        !strat01_sha256_file(__FILE__, header_sha, &temporary, error)) goto finish;
    if (!strat01_r2a_path(report_path, out_dir, report_leaf) ||
        (report = fopen(report_path, "wb")) == NULL) {
        snprintf(error, 256, "cannot write Q6 parity report");
        goto finish;
    }
    fputs("{\n\"command\":", report); strat01_json_string(report, command_name);
    fputs(",\n\"state\":\"OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\","
          "\n\"self_certifies_pass\":false,\n\"model\":{\"path\":", report);
    strat01_json_string(report, model);
    fprintf(report, ",\"bytes\":%" PRIu64 ",\"sha256\":", hashed);
    strat01_json_string(report, model_sha);
    fputs("},\n\"inputs\":{\"swiglu\":{\"path\":", report);
    strat01_json_string(report, input_path);
    fprintf(report, ",\"bytes\":%u,\"sha256\":", STRAT01_Q6P_BATCH * STRAT01_Q6P_WIDTH * 4U);
    strat01_json_string(report, input_sha);
    fputs("},\"ffn_input\":{\"path\":", report);
    strat01_json_string(report, ffn_input_path);
    fprintf(report, ",\"bytes\":%u,\"sha256\":", STRAT01_Q6P_BATCH * STRAT01_Q6P_ROWS * 4U);
    strat01_json_string(report, ffn_input_sha);
    fputs("},\"topk\":{\"path\":", report); strat01_json_string(report, topk_path);
    fputs(",\"bytes\":128,\"sha256\":", report); strat01_json_string(report, topk_sha);
#define Q6P_JSON_INPUT(name, path, bytes, sha) do { \
    fputs("},\"" name "\":{\"path\":", report); strat01_json_string(report, path); \
    fprintf(report, ",\"bytes\":%u,\"sha256\":", (unsigned)(bytes)); strat01_json_string(report, sha); \
} while (0)
    Q6P_JSON_INPUT("ref_kqv", ref_kqv_path, 196608U, extra_sha[0]);
    Q6P_JSON_INPUT("ref_layer1_ffn", ref_layer1_ffn_path, 49152U, extra_sha[1]);
    Q6P_JSON_INPUT("ref_gate", ref_gate_path, 163840U, extra_sha[2]);
    Q6P_JSON_INPUT("ref_q", ref_q_path, 589824U, extra_sha[3]);
    Q6P_JSON_INPUT("ref_k", ref_k_path, 18432U, extra_sha[4]);
    Q6P_JSON_INPUT("ref_shared", ref_shared_path, 49152U, extra_sha[5]);
    Q6P_JSON_INPUT("ref_weights", ref_weights_path, 128U, extra_sha[6]);
#undef Q6P_JSON_INPUT
    fputs("}},\n\"matrix\":{\"name\":\"blk.0.ffn_down.weight\",\"type\":\"Q6_K\","
          "\"shape\":[8960,1536],\"offset\":286755840,\"file_offset\":292858752,"
          "\"span\":11289600},\n\"outputs\":{", report);
#define Q6P_JSON_OUTPUT(name, path, bytes, sha) do { \
    fputc('\"', report); fputs((name), report); fputs("\":{\"path\":", report); strat01_json_string(report, path); \
    fprintf(report, ",\"bytes\":%zu,\"sha256\":", (size_t)(bytes)); strat01_json_string(report, sha); fputc('}', report); \
} while (0)
    Q6P_JSON_OUTPUT("q8_population", q8_path, q8_count * sizeof(*q8), q8_sha); fputc(',', report);
    Q6P_JSON_OUTPUT(generic_name, generic_path, output_count * 4U, generic_sha); fputc(',', report);
    Q6P_JSON_OUTPUT(candidate_name, candidate_path, output_count * 4U, candidate_sha); fputc(',', report);
    Q6P_JSON_OUTPUT(generic_lout_name, generic_lout_path, output_count * 4U, generic_lout_sha); fputc(',', report);
    Q6P_JSON_OUTPUT(candidate_lout_name, candidate_lout_path, output_count * 4U, candidate_lout_sha); fputc(',', report);
    Q6P_JSON_OUTPUT("control_mutated_q6_row", mutation_path, sizeof(mutation), mutation_sha); fputc(',', report);
    if (reference_generic) {
        Q6P_JSON_OUTPUT("closed_active_avx2_control", active_path, output_count * 4U, active_sha);
        fputc(',', report);
    }
    Q6P_JSON_OUTPUT(downstream_name, downstream_path,
        STRAT01_Q6P_BATCH * 6144U * 4U, downstream_sha);
#undef Q6P_JSON_OUTPUT
    fputs("},\n\"engine_source_sha256\":", report); strat01_json_string(report, engine_sha);
    fputs(",\n\"diagnostic_source_sha256\":", report); strat01_json_string(report, header_sha);
    fputs(",\n\"compiler_family\":\"clang\",\"q8_populations_completed\":1,"
          "\"q6_matrix_reads\":1,\"diagnostic_executions\":1,"
          "\"donor_graph_executions\":0,\"reference_graph_executions\":0,"
          "\"timing_or_rate_claim\":null\n}\n", report);
    if (fclose(report) != 0) {
        report = NULL;
        snprintf(error, 256, "Q6 parity report close failure");
        goto finish;
    }
    report = NULL;
    ok = 1;
finish:
    if (report) fclose(report);
    if (weights) fclose(weights);
    free(input); free(ffn_input); free(generic); free(candidate); free(active_control);
    free(generic_lout); free(candidate_lout); free(q8); free(raw); free(mutated);
    free(ref_kqv); free(ref_layer1_ffn); free(ref_gate); free(ref_q); free(ref_k);
    free(ref_shared); free(ref_weights); free(projection); free(layer1_ffn); free(norm_weight);
    free(norm); free(up); free(swiglu); free(down); free(weighted); free(moe); free(ffn_out);
    free(layer1_lout); free(attn_weight); free(layer2_norm); free(kv_projection);
    free(kv_norm_weight); free(prefix); free(k); free(downstream);
    strat01_free_inventory(&inventory);
    if (!ok) {
        strat01_q6p_write_failure(
            out_dir, error[0] ? error : "Q6 parity diagnostic failure", reference_generic);
        fprintf(stderr, "STRAT-01 block-0 Q6 parity: %s\n", error[0] ? error : "failed");
        return 2;
    }
    fprintf(stderr, "STRAT-01 block-0 Q6 parity: OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\n");
    return 0;
}

static int strat01_q6p_cli(
        const char *model, const char *input_path, const char *ffn_input_path,
        const char *topk_path, const char *ref_kqv_path, const char *ref_layer1_ffn_path,
        const char *ref_gate_path, const char *ref_q_path, const char *ref_k_path,
        const char *ref_shared_path, const char *ref_weights_path,
        const char *out_dir, const char *engine_source_path) {
    return strat01_q6p_cli_mode(
        model, input_path, ffn_input_path, topk_path, ref_kqv_path, ref_layer1_ffn_path,
        ref_gate_path, ref_q_path, ref_k_path, ref_shared_path, ref_weights_path,
        out_dir, engine_source_path, 0);
}

static int strat01_q6rg_cli(
        const char *model, const char *input_path, const char *ffn_input_path,
        const char *topk_path, const char *ref_kqv_path, const char *ref_layer1_ffn_path,
        const char *ref_gate_path, const char *ref_q_path, const char *ref_k_path,
        const char *ref_shared_path, const char *ref_weights_path,
        const char *out_dir, const char *engine_source_path) {
    return strat01_q6p_cli_mode(
        model, input_path, ffn_input_path, topk_path, ref_kqv_path, ref_layer1_ffn_path,
        ref_gate_path, ref_q_path, ref_k_path, ref_shared_path, ref_weights_path,
        out_dir, engine_source_path, 1);
}

static int strat01_q6p_selftest(void) {
    strat01_tensor tensor = {0};
    float residual[4] = {1.0f, -2.0f, 3.0f, -4.0f};
    float output[4] = {0.5f, 0.25f, -0.75f, 1.0f};
    float sum[4];
    int bad = 0, checks = 0;
#define Q6P_CHECK(value) do { ++checks; if (!(value)) ++bad; } while (0)
    tensor.name = (char *)"blk.0.ffn_down.weight";
    tensor.type = STRAT01_GGML_Q6_K; tensor.rank = 2U;
    tensor.dims[0] = 8960U; tensor.dims[1] = 1536U;
    tensor.offset = 286755840ULL; tensor.file_offset = 292858752ULL; tensor.span = 11289600ULL;
    Q6P_CHECK(strat01_q6p_descriptor_ok(&tensor));
    tensor.span++;
    Q6P_CHECK(!strat01_q6p_descriptor_ok(&tensor));
    for (unsigned i = 0; i < 4U; ++i) sum[i] = residual[i] + output[i];
    Q6P_CHECK(sum[0] == 1.5f && sum[1] == -1.75f && sum[2] == 2.25f && sum[3] == -3.0f);
    Q6P_CHECK(sizeof(strat01_q8_k_block) == 292U && STRAT01_Q6_K_BLOCK_BYTES == 210U);
    Q6P_CHECK(STRAT01_Q6P_WIDTH % STRAT01_QK_K == 0U && STRAT01_Q6P_WIDTH / STRAT01_QK_K == 35U);
#undef Q6P_CHECK
    fprintf(stderr, "STRAT-01 block-0 Q6 parity selftest: %s (%d checks)\n", bad ? "FAIL" : "PASS", checks);
    return bad ? 1 : 0;
}

#endif /* STRAT01_GGUF_BLOCK0_Q6K_Q8K_AVX2_PARITY_H */
