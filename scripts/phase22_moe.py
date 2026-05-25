import os

with open("benchmarks/benchmark18_coder.c", "r") as f:
    content = f.read()

# 1. Add MoE hyperparameters and weights to global vars in main
vars_text = """
    double total_loss_see_only = 0;
    double total_loss_dyn_only = 0;
    double total_loss_oracle = 0;
    double total_H_see = 0;
    double total_H_dyn = 0;
    
    FILE* f_telemetry = NULL;
    char* telemetry_path = NULL;
    float lambda_state = 0.0f; // For EMA
"""

new_vars = """
    double total_loss_see_only = 0;
    double total_loss_uni_only = 0;
    double total_loss_bi_only = 0;
    double total_loss_oracle = 0;
    
    FILE* f_telemetry = NULL;
    char* telemetry_path = NULL;
    
    float moe_eta = 0.03f;
    float moe_share = 0.001f;
    int use_moe = 0;
    double w_see = 1.0 / 3.0;
    double w_uni = 1.0 / 3.0;
    double w_bi = 1.0 / 3.0;
    
    uint64_t win_see = 0;
    uint64_t win_uni = 0;
    uint64_t win_bi = 0;
    double sum_w_see = 0;
    double sum_w_uni = 0;
    double sum_w_bi = 0;
"""
content = content.replace(vars_text, new_vars)

# 2. Add CLI arguments for eta, share, and moe
cli_text = """} else if (strcmp(argv[i], "--blend") == 0 && i + 1 < argc) {
            blend_lambda = (float)atof(argv[++i]);
        }"""
new_cli = """} else if (strcmp(argv[i], "--blend") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            if (strcmp(arg, "moe") == 0) {
                use_moe = 1;
            } else {
                blend_lambda = (float)atof(arg);
            }
        } else if (strcmp(argv[i], "--eta") == 0 && i + 1 < argc) {
            moe_eta = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--share") == 0 && i + 1 < argc) {
            moe_share = (float)atof(argv[++i]);
        }"""
content = content.replace(cli_text, new_cli)

# 3. Telemetry header
tel_head_old = 'fprintf(f_telemetry, "i,target,loss_see,loss_dyn,loss_actual,H_see,H_dyn,count_bi,tail_mass,prob_see,prob_dyn,lambda_local\\n");'
tel_head_new = 'fprintf(f_telemetry, "i,target,loss_see,loss_uni,loss_bi,loss_actual,w_see,w_uni,w_bi\\n");'
content = content.replace(tel_head_old, tel_head_new)

# 4. CDF Build logic
cdf_build_start = content.find('float p_see[CLASSES];\n        float p_dyn[CLASSES];')
if cdf_build_start == -1: cdf_build_start = content.find('float p_see[CLASSES];\r\n        float p_dyn[CLASSES];')

cdf_build_end = content.find('int diff = CDF_SCALE - total_assigned;')

new_cdf_build = """float p_see[CLASSES];
        float p_uni_arr[CLASSES];
        float p_bi_arr[CLASSES];
        
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
            p_uni_arr[c] = (dyn_uni_counts[c] + alpha) / sum_uni;
            p_bi_arr[c] = (dyn_bi_counts[ctx1][c] + alpha) / sum_bi;
        }
        
        if (use_moe) {
            for (int c = 0; c < CLASSES; c++) {
                probs_tmp[c] = w_see * p_see[c] + w_uni * p_uni_arr[c] + w_bi * p_bi_arr[c];
            }
        } else {
            float current_lambda = blend_lambda;
            for (int c = 0; c < CLASSES; c++) {
                float p_dyn = 0.5f * p_uni_arr[c] + 0.5f * p_bi_arr[c];
                probs_tmp[c] = (1.0f - current_lambda) * p_see[c] + current_lambda * p_dyn;
            }
        }
        
        for (int c = 0; c < CLASSES; c++) {
            int f = (int)(probs_tmp[c] * cdf_scale_f);
            if (f < 0) f = 0;
            freq[c] = (uint16_t)(f + 1);
            total_assigned += freq[c];
        }
        
        """
content = content[:cdf_build_start] + new_cdf_build + content[cdf_build_end:]

# 5. The stray fclose(f_telemetry) block around 687
stray_fclose = """            
                if (f_telemetry) {
        fclose(f_telemetry);
    }
"""
if stray_fclose in content:
    content = content.replace(stray_fclose, "            \n")

# 6. Metrics logic
metrics_start = content.find('float prob_see = p_see[target];')
metrics_end = content.find('double quant_prob = (double)freq[target] / CDF_SCALE;')

new_metrics = """float prob_see = p_see[target];
        if (prob_see < 1e-10f) prob_see = 1e-10f;
        double loss_see = -log2(prob_see);
        
        float prob_uni = p_uni_arr[target];
        if (prob_uni < 1e-10f) prob_uni = 1e-10f;
        double loss_uni = -log2(prob_uni);
        
        float prob_bi = p_bi_arr[target];
        if (prob_bi < 1e-10f) prob_bi = 1e-10f;
        double loss_bi = -log2(prob_bi);
        
        total_loss_see_only += loss_see;
        total_loss_uni_only += loss_uni;
        total_loss_bi_only += loss_bi;
        
        double min_loss = loss_see;
        if (loss_uni < min_loss) min_loss = loss_uni;
        if (loss_bi < min_loss) min_loss = loss_bi;
        total_loss_oracle += min_loss;
        
        if (loss_see <= loss_uni && loss_see <= loss_bi) win_see++;
        else if (loss_uni <= loss_bi) win_uni++;
        else win_bi++;
        
        sum_w_see += w_see;
        sum_w_uni += w_uni;
        sum_w_bi += w_bi;
        
        if (f_telemetry) {
            fprintf(f_telemetry, "%d,%d,%f,%f,%f,%f,%f,%f,%f\\n", 
                    i, target, loss_see, loss_uni, loss_bi, sample_loss, w_see, w_uni, w_bi);
        }
        
        // Fixed-Share Weight Update
        if (use_moe) {
            w_see *= expf(-moe_eta * loss_see);
            w_uni *= expf(-moe_eta * loss_uni);
            w_bi *= expf(-moe_eta * loss_bi);
            
            // Normalize
            double sum_w = w_see + w_uni + w_bi;
            if (sum_w < 1e-30) {
                w_see = 1.0/3.0; w_uni = 1.0/3.0; w_bi = 1.0/3.0;
            } else {
                w_see /= sum_w; w_uni /= sum_w; w_bi /= sum_w;
            }
            
            // Share
            w_see = (1.0f - moe_share) * w_see + moe_share / 3.0f;
            w_uni = (1.0f - moe_share) * w_uni + moe_share / 3.0f;
            w_bi = (1.0f - moe_share) * w_bi + moe_share / 3.0f;
        }
        
        """
content = content[:metrics_start] + new_metrics + content[metrics_end:]

# 7. Print logic
printf_block = """        printf("--- Phase 21 Oracle / Telemetry ---\\n");
        printf("SEE Only BPB:    %.4f\\n", total_loss_see_only / eval_len);
        printf("Dyn Only BPB:    %.4f\\n", total_loss_dyn_only / eval_len);
        printf("Oracle BPB:      %.4f\\n", total_loss_oracle / eval_len);
        printf("Avg H_see:       %.4f\\n", total_H_see / eval_len);
        printf("Avg H_dyn:       %.4f\\n", total_H_dyn / eval_len);"""

new_printf = """        printf("--- Phase 22 Online Mixture of Experts ---\\n");
        printf("SEE Only BPB:    %.4f\\n", total_loss_see_only / eval_len);
        printf("Uni Only BPB:    %.4f\\n", total_loss_uni_only / eval_len);
        printf("Bi  Only BPB:    %.4f\\n", total_loss_bi_only / eval_len);
        printf("Oracle BPB:      %.4f\\n", total_loss_oracle / eval_len);
        if (use_moe) {
            printf("\\n--- MoE Stats ---\\n");
            printf("Avg W_SEE:       %.4f\\n", sum_w_see / eval_len);
            printf("Avg W_UNI:       %.4f\\n", sum_w_uni / eval_len);
            printf("Avg W_BI:        %.4f\\n", sum_w_bi / eval_len);
            printf("Final W_SEE:     %.4f\\n", w_see);
            printf("Final W_UNI:     %.4f\\n", w_uni);
            printf("Final W_BI:      %.4f\\n", w_bi);
            printf("Win %% SEE:       %.1f%%\\n", 100.0 * win_see / eval_len);
            printf("Win %% UNI:       %.1f%%\\n", 100.0 * win_uni / eval_len);
            printf("Win %% BI:        %.1f%%\\n", 100.0 * win_bi / eval_len);
        }"""
content = content.replace(printf_block, new_printf)

# Also fix the f_dump fclose problem
fclose_dump_str = """if (f_dump) {
                uint8_t t = (uint8_t)target;
                fwrite(&t, 1, 1, f_dump);
            }"""
if fclose_dump_str in content:
    pass

# We also need to add fclose(f_telemetry) to the end of main loop
end_loop = """
    if (f_dump) {
        fclose(f_dump);
    }
"""
if end_loop in content:
    content = content.replace(end_loop, end_loop + """    if (f_telemetry) {
        fclose(f_telemetry);
    }\n""")

with open("benchmarks/benchmark18_coder.c", "w") as f:
    f.write(content)
print("Phase 22 MoE injected.")
