import os

def patch_file():
    with open("benchmarks/benchmark18_coder.c", "r") as f:
        content = f.read()
        
    if "LZ_HASH_SIZE" in content:
        print("Already patched!")
        return

    # 1. Struct
    struct_code = """
#define LZ_HASH_SIZE 262144
typedef struct {
    uint32_t key;
    uint32_t total;
    uint16_t counts[256];
} LzEntry;
LzEntry* lz_table = NULL;
uint32_t lz_hash(uint32_t key) {
    key ^= key >> 16; key *= 0x85ebca6b; key ^= key >> 13; key *= 0xc2b2ae35; key ^= key >> 16;
    return key & (LZ_HASH_SIZE - 1);
}
LzEntry* lz_lookup(uint32_t key) {
    uint32_t idx = lz_hash(key);
    for (int i = 0; i < 16; i++) {
        if (lz_table[idx].total == 0) return &lz_table[idx];
        if (lz_table[idx].key == key) return &lz_table[idx];
        idx = (idx + 1) & (LZ_HASH_SIZE - 1);
    }
    return &lz_table[idx];
}
"""
    content = content.replace("#define CLASSES 256", "#define CLASSES 256\n" + struct_code)

    # 2. Variables
    content = content.replace("double total_loss_bi_only = 0;", "double total_loss_bi_only = 0;\n    double total_loss_lz_only = 0;")
    content = content.replace("double w_see = 1.0 / 3.0;\n    double w_uni = 1.0 / 3.0;\n    double w_bi = 1.0 / 3.0;", 
                              "double w_see = 1.0 / 4.0;\n    double w_uni = 1.0 / 4.0;\n    double w_bi = 1.0 / 4.0;\n    double w_lz = 1.0 / 4.0;")
    content = content.replace("uint64_t win_bi = 0;", "uint64_t win_bi = 0;\n    uint64_t win_lz = 0;")
    content = content.replace("double sum_w_bi = 0;", "double sum_w_bi = 0;\n    double sum_w_lz = 0;")

    # 3. CSV header
    content = content.replace('fprintf(f_telemetry, "i,target,loss_see,loss_uni,loss_bi,loss_actual,w_see,w_uni,w_bi\\n");',
                              'fprintf(f_telemetry, "i,target,loss_see,loss_uni,loss_bi,loss_lz,loss_actual,w_see,w_uni,w_bi,w_lz\\n");')

    # 4. Alloc
    content = content.replace("topk_indices = malloc(CLASSES * CLASSES * CLASSES * sizeof(uint8_t));",
                              "topk_indices = malloc(CLASSES * CLASSES * CLASSES * sizeof(uint8_t));\n    lz_table = calloc(LZ_HASH_SIZE, sizeof(LzEntry));\n    uint32_t lz_ctx = 0;\n    float p_lz_arr[CLASSES];")

    # 5. Build probs
    prob_build = """        LzEntry* lz_ent = lz_lookup(lz_ctx);
        float sum_lz = lz_ent->total + 256.0f * 1.0f;
        for (int c = 0; c < CLASSES; c++) {
            p_uni_arr[c] = (dyn_uni_counts[c] + alpha) / sum_uni;
            p_bi_arr[c] = (dyn_bi_counts[ctx1][c] + alpha) / sum_bi;
            if (lz_ent->total > 0 && lz_ent->key == lz_ctx) {
                p_lz_arr[c] = (lz_ent->counts[c] + 1.0f) / sum_lz;
            } else {
                p_lz_arr[c] = 1.0f / 256.0f;
            }"""
    content = content.replace("        for (int c = 0; c < CLASSES; c++) {\n            p_uni_arr[c] = (dyn_uni_counts[c] + alpha) / sum_uni;\n            p_bi_arr[c] = (dyn_bi_counts[ctx1][c] + alpha) / sum_bi;", prob_build)

    # 6. Mix
    content = content.replace("probs_tmp[c] = w_see * p_see[c] + w_uni * p_uni_arr[c] + w_bi * p_bi_arr[c];",
                              "probs_tmp[c] = w_see * p_see[c] + w_uni * p_uni_arr[c] + w_bi * p_bi_arr[c] + w_lz * p_lz_arr[c];")

    # 7. Loss
    content = content.replace("double loss_bi = -log2(prob_bi);",
                              "double loss_bi = -log2(prob_bi);\n        float prob_lz = p_lz_arr[target];\n        if(prob_lz < 1e-10f) prob_lz = 1e-10f;\n        double loss_lz = -log2(prob_lz);")
    content = content.replace("total_loss_bi_only += loss_bi;", "total_loss_bi_only += loss_bi;\n        total_loss_lz_only += loss_lz;")
    content = content.replace("if (loss_bi < min_loss) min_loss = loss_bi;", "if (loss_bi < min_loss) min_loss = loss_bi;\n        if (loss_lz < min_loss) min_loss = loss_lz;")

    # 8. Win
    content = content.replace("if (loss_see <= loss_uni && loss_see <= loss_bi) win_see++;\n        else if (loss_uni <= loss_bi) win_uni++;\n        else win_bi++;",
                              "if (loss_see <= loss_uni && loss_see <= loss_bi && loss_see <= loss_lz) win_see++;\n        else if (loss_uni <= loss_bi && loss_uni <= loss_lz) win_uni++;\n        else if (loss_bi <= loss_lz) win_bi++;\n        else win_lz++;")

    content = content.replace("sum_w_bi += w_bi;", "sum_w_bi += w_bi;\n        sum_w_lz += w_lz;")

    # 9. Print telemetry
    content = content.replace('fprintf(f_telemetry, "%d,%d,%f,%f,%f,%f,%f,%f,%f\\n", \n                    i, target, loss_see, loss_uni, loss_bi, sample_loss, w_see, w_uni, w_bi);',
                              'fprintf(f_telemetry, "%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\\n", \n                    i, target, loss_see, loss_uni, loss_bi, loss_lz, sample_loss, w_see, w_uni, w_bi, w_lz);')

    # 10. Update MoE
    content = content.replace("w_bi *= expf(-moe_eta * loss_bi);", "w_bi *= expf(-moe_eta * loss_bi);\n            w_lz *= expf(-moe_eta * loss_lz);")
    content = content.replace("double sum_w = w_see + w_uni + w_bi;", "double sum_w = w_see + w_uni + w_bi + w_lz;")
    content = content.replace("w_see = 1.0/3.0; w_uni = 1.0/3.0; w_bi = 1.0/3.0;", "w_see = 1.0/4.0; w_uni = 1.0/4.0; w_bi = 1.0/4.0; w_lz = 1.0/4.0;")
    content = content.replace("w_see /= sum_w; w_uni /= sum_w; w_bi /= sum_w;", "w_see /= sum_w; w_uni /= sum_w; w_bi /= sum_w; w_lz /= sum_w;")
    
    content = content.replace("w_see = (1.0f - moe_share) * w_see + moe_share / 3.0f;", "w_see = (1.0f - moe_share) * w_see + moe_share / 4.0f;")
    content = content.replace("w_uni = (1.0f - moe_share) * w_uni + moe_share / 3.0f;", "w_uni = (1.0f - moe_share) * w_uni + moe_share / 4.0f;")
    content = content.replace("w_bi = (1.0f - moe_share) * w_bi + moe_share / 3.0f;", "w_bi = (1.0f - moe_share) * w_bi + moe_share / 4.0f;\n            w_lz = (1.0f - moe_share) * w_lz + moe_share / 4.0f;")

    # 11. Context Update
    update_code = """        if (lz_ent->key != lz_ctx) {
            lz_ent->key = lz_ctx;
            lz_ent->total = 0;
            memset(lz_ent->counts, 0, sizeof(lz_ent->counts));
        }
        lz_ent->counts[target]++;
        lz_ent->total++;
        lz_ctx = (lz_ctx << 8) | target;
        ctx2 = ctx1;"""
    content = content.replace("        ctx2 = ctx1;\n        ctx1 = target;\n        uint64_t t4 = __rdtsc();", update_code + "\n        ctx1 = target;\n        uint64_t t4 = __rdtsc();")

    # 12. Final Stats Print
    content = content.replace('printf("Bi  Only BPB:    %.4f\\n", total_loss_bi_only / eval_len);', 'printf("Bi  Only BPB:    %.4f\\n", total_loss_bi_only / eval_len);\n    printf("LZ  Only BPB:    %.4f\\n", total_loss_lz_only / eval_len);')
    content = content.replace('printf("Avg W_BI:        %.4f\\n", sum_w_bi / eval_len);', 'printf("Avg W_BI:        %.4f\\n", sum_w_bi / eval_len);\n    printf("Avg W_LZ:        %.4f\\n", sum_w_lz / eval_len);')
    content = content.replace('printf("Final W_BI:      %.4f\\n", w_bi);', 'printf("Final W_BI:      %.4f\\n", w_bi);\n    printf("Final W_LZ:      %.4f\\n", w_lz);')
    content = content.replace('printf("Win %% BI:        %.1f%%\\n", 100.0 * win_bi / eval_len);', 'printf("Win %% BI:        %.1f%%\\n", 100.0 * win_bi / eval_len);\n    printf("Win %% LZ:        %.1f%%\\n", 100.0 * win_lz / eval_len);')

    with open("benchmarks/benchmark18_coder.c", "w") as f:
        f.write(content)
    print("Patched successfully!")

if __name__ == "__main__":
    patch_file()
