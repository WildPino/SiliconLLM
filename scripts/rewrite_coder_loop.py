import os

with open("benchmarks/benchmark18_coder.c", "r") as f:
    content = f.read()

uni_idx = content.find('// Calculate unigram to provide baseline')
uni_end = content.find('for (int i = 0; i < eval_len; i++) {')

new_uni = """
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
content = content[:uni_idx] + new_uni + content[uni_end:]

loop_idx = content.find('for (int i = 0; i < eval_len; i++) {')
loop_end = content.find('uint64_t t0 = __rdtsc();')

new_loop_start = """for (int i = 0; i < eval_len; i++) {
        int global_idx = eval_start + 2 + i;
        uint8_t original_target = 0;
        if (mode == MODE_EVAL || mode == MODE_ENCODE) {
            original_target = data[global_idx];
        }
        uint8_t target = original_target; // Will be overwritten if decoding
        
        """
content = content[:loop_idx] + new_loop_start + content[loop_end:]

loss_idx = content.find('total_loss -= log2(prob);')
total_corr_end = content.find('total_correct++;', loss_idx) + 16
total_corr_block_end = content.find('}', total_corr_end) + 1

new_loss = """if (mode == MODE_EVAL || mode == MODE_ENCODE) {
            total_loss -= log2(prob);
            total_quantized_loss += quantized_loss;
            int is_startup = (i < 4096);
            if (is_startup) {
                startup_loss -= log2(prob);
                startup_quantized_loss += quantized_loss;
                startup_count++;
            } else {
                stable_loss -= log2(prob);
                stable_quantized_loss += quantized_loss;
                stable_count++;
            }
            if (target == topk_indices[ctx2][ctx1][0]) {
                total_correct++;
            }
        }"""
content = content[:loss_idx] + new_loss + content[total_corr_block_end:]

dec_err_idx = content.find('if (target != original_target) {')
if dec_err_idx != -1:
    dec_err_end = content.find('}', dec_err_idx) + 1
    content = content[:dec_err_idx] + content[dec_err_end:]

print_idx = content.find('double bpb = total_loss / eval_len;')
print_end = content.find('return 0;', print_idx)

new_print = """
    if (mode == MODE_EVAL || mode == MODE_ENCODE) {
        double bpb = total_loss / eval_len;
        double quant_bpb = total_quantized_loss / eval_len;
        double startup_bpb = startup_count > 0 ? startup_quantized_loss / startup_count : 0;
        double stable_bpb = stable_count > 0 ? stable_quantized_loss / stable_count : 0;
        
        printf("\\n=== Streaming Inference Results ===\\n");
        printf("Unigram BPB:     %.4f\\n", uni_loss / eval_len);
        printf("Model BPB:       %.4f\\n", bpb);
        printf("Quantized BPB:   %.4f\\n", quant_bpb);
        printf("  - Startup BPB: %.4f (first 4KB)\\n", startup_bpb);
        printf("  - Stable BPB:  %.4f\\n", stable_bpb);
        printf("Accuracy:        %.2f%%\\n", 100.0 * total_correct / eval_len);

        printf("\\n--- Dot Products / Byte ---\\n");
        printf("A Projections:   %.1f\\n", (double)total_A_projs / eval_len);
        printf("LowRank Cands:   %.1f\\n", (double)total_lowrank_cands / eval_len);
        printf("Full Residuals:  %.1f\\n", (double)total_residual_dots / eval_len);

        printf("\\n--- Mean Cycles / Byte ---\\n");
        printf("Extract:         %.1f\\n", (double)total_extract_cyc / eval_len);
        printf("Normalize:       %.1f\\n", (double)total_norm_cyc / eval_len);
        printf("Predict:         %.1f\\n", (double)total_predict_cyc / eval_len);
        printf("  - Candidate Gen: %.1f\\n", (double)total_cand_cyc / eval_len);
        printf("  - Full Residual: %.1f\\n", (double)total_res_cyc / eval_len);
        printf("  - Softmax+Tail:  %.1f\\n", (double)total_soft_cyc / eval_len);
        printf("  - CDF Build:     %.1f\\n", (double)total_cdf_cyc / eval_len);
        if (mode == MODE_ENCODE || mode == MODE_DECODE) {
            printf("  - Range Coder:   %.1f\\n", (double)total_rc_cyc / eval_len);
        }
        printf("Avg Tail Mass:   %.4f\\n", total_tail_mass / eval_len);
        printf("Observe:         %.1f\\n", (double)total_observe_cyc / eval_len);
        
        double rc = (mode == MODE_ENCODE || mode == MODE_DECODE) ? (double)total_rc_cyc / eval_len : 0;
        printf("Total:           %.1f\\n", (double)total_cyc / eval_len + rc);
    } else {
        printf("\\nDecode Complete! Length: %d bytes.\\n", eval_len);
    }
"""
content = content[:print_idx] + new_print + content[print_end:]

with open("benchmarks/benchmark18_coder.c", "w") as f:
    f.write(content)
print("Updated successfully")
