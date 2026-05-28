import os

with open("benchmarks/benchmark18_coder.c", "r") as f:
    content = f.read()

idx = content.find('for (int i = 0; i < eval_len; i++) {')

vars_decl = """
    double total_loss = 0;
    double total_quantized_loss = 0;
    
    double startup_loss = 0;
    double startup_quantized_loss = 0;
    int startup_count = 0;
    
    double stable_loss = 0;
    double stable_quantized_loss = 0;
    int stable_count = 0;
    
    uint64_t total_correct = 0;
    uint64_t total_topk_hits = 0;
    
    uint64_t total_extract_cyc = 0;
    uint64_t total_norm_cyc = 0;
    uint64_t total_predict_cyc = 0;
    uint64_t total_observe_cyc = 0;

    double total_loss_out_pool = 0;
    double total_loss_in_pool_out_topk = 0;
    uint64_t count_target_in_pool = 0;

    uint64_t total_cand_cyc = 0;
    uint64_t total_res_cyc = 0;
    uint64_t total_soft_cyc = 0;
    double total_tail_mass = 0;
    
    uint64_t total_residual_dots = 0;
    uint64_t total_lowrank_cands = 0;
    uint64_t total_A_projs = 0;

    uint64_t total_cyc = 0;

    double uni_loss = 0;
    if (mode == MODE_EVAL || mode == MODE_ENCODE) {
        double uni_counts[CLASSES] = {0};
        double total_uni = 0;
        for(int i=0; i<eval_len; i++) {
            uni_counts[data[eval_start + 2 + i]]++;
            total_uni++;
        }
        for(int i=0; i<eval_len; i++) {
            double p = uni_counts[data[eval_start + 2 + i]] / total_uni;
            uni_loss -= log2(p > 0 ? p : 1e-10);
        }
    }
    
"""

content = content[:idx] + vars_decl + content[idx:]

with open("benchmarks/benchmark18_coder.c", "w") as f:
    f.write(content)
print("Variables restored")
