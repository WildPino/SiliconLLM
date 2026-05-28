import os

with open("benchmarks/benchmark18_coder.c", "r") as f:
    content = f.read()

# 1. Add globals for telemetry
metrics_vars = """
    double total_loss_see_only = 0;
    double total_loss_dyn_only = 0;
    double total_loss_oracle = 0;
    double total_H_see = 0;
    double total_H_dyn = 0;
    
    FILE* f_telemetry = NULL;
    char* telemetry_path = NULL;
    float lambda_state = 0.0f; // For EMA
"""
vars_idx = content.find('double total_loss = 0;')
content = content[:vars_idx] + metrics_vars + content[vars_idx:]

# 2. Add CLI parsing for telemetry
cli_parse = """} else if (strcmp(argv[i], "--telemetry") == 0 && i + 1 < argc) {
            telemetry_path = argv[++i];
        """
cli_idx = content.find('} else if (strcmp(argv[i], "--blend") == 0 && i + 1 < argc) {')
content = content[:cli_idx] + cli_parse + content[cli_idx:]

# 3. Open telemetry file
telemetry_open = """
    if (telemetry_path) {
        f_telemetry = fopen(telemetry_path, "w");
        if (f_telemetry) {
            fprintf(f_telemetry, "i,target,loss_see,loss_dyn,loss_actual,H_see,H_dyn,count_bi,tail_mass,prob_see,prob_dyn,lambda_local\\n");
        }
    }
"""
weights_idx = content.find('FILE* fw = fopen(weights_path, "rb");')
content = content[:weights_idx] + telemetry_open + content[weights_idx:]

# 4. Replace CDF Build
cdf_build_start = content.find('// 4. Build CDF')
cdf_build_end = content.find('int diff = CDF_SCALE - total_assigned;')

new_cdf_build = """// 4. Build CDF
        uint64_t t_cdf0 = __rdtsc();
        uint16_t freq[CLASSES];
        uint32_t total_assigned = 0;
        
        float tail_scale = (tail_mode > 0) ? (expf(-max_l) / Z_total) : 0;
        float cdf_scale_f = (float)(CDF_SCALE - CLASSES);
        
        float p_see[CLASSES];
        float p_dyn[CLASSES];
        
        for (int c = 0; c < CLASSES; c++) {
            if (tail_mode > 0) p_see[c] = base_probs[ctx2][ctx1][c] * tail_scale;
            else p_see[c] = 0;
        }
        for (int k = 0; k < num_cands; k++) {
            int c = cand_indices[k];
            p_see[c] = expf(full_logits[k] - max_l) / Z_total;
        }
        
        float alpha = 1.0f;
        float sum_uni = dyn_total_count + CLASSES * alpha;
        float sum_bi = 0;
        for(int c=0; c<CLASSES; c++) sum_bi += dyn_bi_counts[ctx1][c];
        sum_bi += CLASSES * alpha;
        
        for (int c = 0; c < CLASSES; c++) {
            float p_uni = (dyn_uni_counts[c] + alpha) / sum_uni;
            float p_bi = (dyn_bi_counts[ctx1][c] + alpha) / sum_bi;
            p_dyn[c] = 0.5f * p_uni + 0.5f * p_bi;
        }
        
        float H_see = 0;
        float H_dyn = 0;
        for (int c = 0; c < CLASSES; c++) {
            if (p_see[c] > 1e-10f) H_see -= p_see[c] * log2(p_see[c]);
            if (p_dyn[c] > 1e-10f) H_dyn -= p_dyn[c] * log2(p_dyn[c]);
        }
        
        float current_lambda = blend_lambda;
        if (blend_lambda < 0.0f) { // Auto
            float lambda_raw = 0.0f;
            float actual_bi = sum_bi - CLASSES * alpha;
            if (actual_bi > 50 && H_dyn < H_see) {
                lambda_raw = 0.9f;
            } else if (tail_mass > 0.4f) {
                lambda_raw = 0.5f;
            } else {
                lambda_raw = 0.1f;
            }
            lambda_state = 0.9f * lambda_state + 0.1f * lambda_raw;
            current_lambda = lambda_state;
        }
        
        for (int c = 0; c < CLASSES; c++) {
            probs_tmp[c] = (1.0f - current_lambda) * p_see[c] + current_lambda * p_dyn[c];
        }
        
        for (int c = 0; c < CLASSES; c++) {
            int f = (int)(probs_tmp[c] * cdf_scale_f);
            if (f < 0) f = 0;
            freq[c] = (uint16_t)(f + 1);
            total_assigned += freq[c];
        }
        
        """
content = content[:cdf_build_start] + new_cdf_build + content[cdf_build_end:]

# 5. Add metrics logic
metrics_start = content.find('double sample_loss = -log2(final_prob);')
metrics_end = content.find('total_loss += sample_loss;') + len('total_loss += sample_loss;')

new_metrics = """double sample_loss = -log2(final_prob);
        total_loss += sample_loss;
        
        float prob_see = p_see[target];
        if (prob_see < 1e-10f) prob_see = 1e-10f;
        double loss_see = -log2(prob_see);
        
        float prob_dyn = p_dyn[target];
        if (prob_dyn < 1e-10f) prob_dyn = 1e-10f;
        double loss_dyn = -log2(prob_dyn);
        
        total_loss_see_only += loss_see;
        total_loss_dyn_only += loss_dyn;
        total_loss_oracle += (loss_see < loss_dyn) ? loss_see : loss_dyn;
        
        total_H_see += H_see;
        total_H_dyn += H_dyn;
        
        if (f_telemetry) {
            fprintf(f_telemetry, "%d,%d,%f,%f,%f,%f,%f,%d,%f,%f,%f,%f\\n", 
                    i, target, loss_see, loss_dyn, sample_loss, H_see, H_dyn, 
                    (int)(sum_bi - CLASSES * alpha), tail_mass, prob_see, prob_dyn, current_lambda);
        }
"""
content = content[:metrics_start] + new_metrics + content[metrics_end:]

# 6. Close telemetry
close_telemetry = """    if (f_telemetry) {
        fclose(f_telemetry);
    }
"""
close_idx = content.find('if (f_dump) {')
content = content[:close_idx] + close_telemetry + content[close_idx:]

# 7. Print final metrics
print_metrics = """
        printf("--- Phase 21 Oracle / Telemetry ---\\n");
        printf("SEE Only BPB:    %.4f\\n", total_loss_see_only / eval_len);
        printf("Dyn Only BPB:    %.4f\\n", total_loss_dyn_only / eval_len);
        printf("Oracle BPB:      %.4f\\n", total_loss_oracle / eval_len);
        printf("Avg H_see:       %.4f\\n", total_H_see / eval_len);
        printf("Avg H_dyn:       %.4f\\n", total_H_dyn / eval_len);
"""
print_idx = content.find('printf("\\\\n--- Mean Cycles / Byte ---");')
if print_idx == -1: print_idx = content.find('printf("\\n--- Mean Cycles / Byte ---");')
content = content[:print_idx] + print_metrics + content[print_idx:]

with open("scripts/phase21_instrument.py.out.c", "w") as f:
    f.write(content)
print("File prepared.")
