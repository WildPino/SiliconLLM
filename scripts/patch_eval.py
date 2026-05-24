import re

with open(r"d:\_THINGS\Progetti\SiliconLLM\benchmarks\eval_entropy_stream.c", "r") as f:
    code = f.read()

# Add tail_mode arg
code = code.replace("int req_topk = 256;", "int req_topk = 256;\n    int tail_mode = 0; // 0=none, 1=ngram, 2=ngram+bias")
code = code.replace('if (strcmp(argv[i], "--topk") == 0 && i + 1 < argc) req_topk = atoi(argv[++i]);',
                    'if (strcmp(argv[i], "--topk") == 0 && i + 1 < argc) req_topk = atoi(argv[++i]);\n        if (strcmp(argv[i], "--tail-mode") == 0 && i + 1 < argc) tail_mode = atoi(argv[++i]);')

# Add A_proj/B_proj loading after fclose(fw)
load_code = """
    uint32_t lr_rank = 0;
    float* A_proj = NULL;
    float* B_proj = NULL;
    if (fread(&lr_rank, sizeof(uint32_t), 1, fw) == 1) {
        A_proj = malloc(lr_rank * SEE_FEATURE_DIM * sizeof(float));
        B_proj = malloc(CLASSES * lr_rank * sizeof(float));
        fread(A_proj, sizeof(float), lr_rank * SEE_FEATURE_DIM, fw);
        fread(B_proj, sizeof(float), CLASSES * lr_rank, fw);
        printf("Loaded Low-Rank factors with Rank = %d\\n", lr_rank);
    }
    fclose(fw);
"""
code = code.replace("fclose(fw);", load_code)

# Add base_probs calculation
base_probs_code = """
    float (*base_probs)[CLASSES][CLASSES] = NULL;
    float (*Z_base)[CLASSES] = NULL;
    if (tail_mode > 0) {
        base_probs = malloc(CLASSES * CLASSES * CLASSES * sizeof(float));
        Z_base = malloc(CLASSES * CLASSES * sizeof(float));
        for (int c2 = 0; c2 < CLASSES; c2++) {
            for (int c1 = 0; c1 < CLASSES; c1++) {
                float sum = 0;
                for (int c = 0; c < CLASSES; c++) {
                    float p = 0;
                    if (tail_mode == 1) p = expf(trigram_logits[c2][c1][c]);
                    else if (tail_mode == 2) p = expf(trigram_logits[c2][c1][c] + B[c]);
                    base_probs[c2][c1][c] = p;
                    sum += p;
                }
                Z_base[c2][c1] = sum;
            }
        }
    }
"""
code = code.replace("topk_indices = malloc(", base_probs_code + "\n    topk_indices = malloc(")

# Replace prediction loop variables
metrics = """
    uint64_t total_cand_cyc = 0;
    uint64_t total_res_cyc = 0;
    uint64_t total_soft_cyc = 0;
    double total_tail_mass = 0;
"""
code = code.replace("uint64_t total_observe_cyc = 0;", "uint64_t total_observe_cyc = 0;\n" + metrics)

# Replace the entire prediction loop (from `float logits[CLASSES];` to `float prob = ...`)
# I will use regex to find and replace it
import sys

new_predict_loop = """
        // 3. Predict Logits
        uint64_t t_cand0 = __rdtsc();
        int cand_indices[CLASSES];
        float cand_logits[CLASSES];
        int num_cands = (tail_mode == 0) ? CLASSES : req_topk;
        
        if (tail_mode > 0 && lr_rank > 0) {
            float z[256]; // max rank 256
            for (uint32_t r = 0; r < lr_rank; r++) {
                z[r] = dot_product_simd(&A_proj[r * SEE_FEATURE_DIM], features, SEE_FEATURE_DIM);
            }
            
            for(int i=0; i<req_topk; i++) { cand_logits[i] = -1e9f; cand_indices[i] = 0; }
            for (int c = 0; c < CLASSES; c++) {
                float l = B[c] + trigram_logits[ctx2][ctx1][c];
                float lr_val = 0;
                for(uint32_t r = 0; r < lr_rank; r++) {
                    lr_val += B_proj[c * lr_rank + r] * z[r];
                }
                l += lr_val;
                
                if (l > cand_logits[req_topk-1]) {
                    int pos = req_topk - 1;
                    while (pos > 0 && l > cand_logits[pos-1]) {
                        cand_logits[pos] = cand_logits[pos-1];
                        cand_indices[pos] = cand_indices[pos-1];
                        pos--;
                    }
                    cand_logits[pos] = l;
                    cand_indices[pos] = c;
                }
            }
        } else {
            // Default top-k from trigram or full evaluation
            for (int i = 0; i < num_cands; i++) {
                cand_indices[i] = (tail_mode == 0) ? i : topk_indices[ctx2][ctx1][i];
            }
        }
        uint64_t t_cand1 = __rdtsc();
        total_cand_cyc += (t_cand1 - t_cand0);
        
        uint64_t t_res0 = __rdtsc();
        float full_logits[CLASSES];
        int target_in_topk = 0;
        int target_idx = -1;
        float max_l = -1e9f;
        
        for (int i = 0; i < num_cands; i++) {
            int c = cand_indices[i];
            if (c == target) {
                target_in_topk = 1;
                target_idx = i;
            }
            full_logits[i] = trigram_logits[ctx2][ctx1][c] + B[c] + dot_product_simd(W[c], features, SEE_FEATURE_DIM);
            if (full_logits[i] > max_l) max_l = full_logits[i];
        }
        if (max_l < 0) max_l = 0; // Don't shift up, only shift down to prevent overflow
        
        uint64_t t_res1 = __rdtsc();
        total_res_cyc += (t_res1 - t_res0);
        
        uint64_t t_soft0 = __rdtsc();
        float Z_K = 0;
        for (int i = 0; i < num_cands; i++) {
            Z_K += expf(full_logits[i] - max_l);
        }
        
        float tail_mass = 0;
        if (tail_mode > 0) {
            tail_mass = Z_base[ctx2][ctx1];
            for (int i = 0; i < num_cands; i++) {
                tail_mass -= base_probs[ctx2][ctx1][cand_indices[i]];
            }
            if (tail_mass < 0) tail_mass = 0;
        }
        
        float Z_total = Z_K + tail_mass * expf(-max_l);
        float prob = 0;
        if (target_in_topk) {
            prob = expf(full_logits[target_idx] - max_l) / Z_total;
        } else if (tail_mode > 0) {
            prob = (base_probs[ctx2][ctx1][target] * expf(-max_l)) / Z_total;
        } else {
            prob = 1e-10f; // target not in topk and no tail mode (should not happen if tail_mode=0 as K=256)
        }
        if (prob < 1e-10f) prob = 1e-10f;
        
        int best_c = cand_indices[0]; // because full_logits might not be sorted, but cand_indices is sorted by lr
        // Wait, cand_indices is sorted by lr, not by full_logits! So best_c is just the top candidate.
        // For accurate accuracy, we should find the max of full_logits.
        for(int i=0; i<num_cands; i++) {
            if(full_logits[i] == max_l) { best_c = cand_indices[i]; break; }
        }
        
        total_tail_mass += tail_mass;
        
        uint64_t t_soft1 = __rdtsc();
        total_soft_cyc += (t_soft1 - t_soft0);
        
        uint64_t t3 = t_soft1; // for total_predict_cyc
"""

start_str = "        // 3. Predict Logits"
end_str = "        uint64_t t3 = __rdtsc();"

start_idx = code.find(start_str)
end_idx = code.find(end_str) + len(end_str)

code = code[:start_idx] + new_predict_loop.strip() + code[end_idx:]

# Also update the print block
code = code.replace('printf("Predict:         %.1f\\n", (double)total_predict_cyc / eval_len);', 
"""printf("Predict:         %.1f\\n", (double)total_predict_cyc / eval_len);
    printf("  - Candidate Gen: %.1f\\n", (double)total_cand_cyc / eval_len);
    printf("  - Full Residual: %.1f\\n", (double)total_res_cyc / eval_len);
    printf("  - Softmax+Tail:  %.1f\\n", (double)total_soft_cyc / eval_len);
    printf("Avg Tail Mass:   %.4f\\n", total_tail_mass / eval_len);""")


with open(r"d:\_THINGS\Progetti\SiliconLLM\benchmarks\eval_entropy_stream.c", "w") as f:
    f.write(code)

print("eval_entropy_stream.c patched successfully.")
