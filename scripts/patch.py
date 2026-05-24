import re

with open('benchmarks/benchmark10_distribution.c', 'r') as f:
    code = f.read()

# 1. Add m4_extract_current_32
code = code.replace('void m4_extract_32(int idx, uint8_t byte, float* out) {',
'''void m4_extract_current_32(int idx, uint8_t byte, float* out) {
    uint8_t vec[32];
    _mm256_storeu_si256((__m256i*)vec, m4_cb[byte]);
    for (int i = 0; i < 32; i++) out[i] = vec[i];
}

void m4_extract_32(int idx, uint8_t byte, float* out) {''')

# 2. Add evaluate_model prototype
code = code.replace('void train_logistic_regression(', 'void evaluate_model(AdamState* model, int num_features, int use_residual, double* out_bpb, double* out_acc);\nvoid train_logistic_regression(')

# 3. Modify train_logistic_regression for early stopping
train_loop = '''    AdamState* best_adam = (AdamState*)malloc(sizeof(AdamState));
    double best_bpb = 1e9, acc = 0;
    evaluate_model(model, num_features, use_residual, &best_bpb, &acc);
    memcpy(best_adam, model, sizeof(AdamState));
    
    // if (use_residual) printf("    Epoch 0 (Baseline): Val BPB = %.4f\\n", best_bpb);
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        memset(gradW, 0, CLASSES * MAX_FEATURES * sizeof(float));
        memset(gradB, 0, sizeof(gradB));
        
        for (int i = 0; i < train_size; i++) {
'''
code = re.sub(r'    for \(int epoch = 0; epoch < epochs; epoch\+\+\) \{\s+double total_loss = 0.0;\s+memset\(gradW, 0, CLASSES \* MAX_FEATURES \* sizeof\(float\)\);\s+memset\(gradB, 0, sizeof\(gradB\)\);\s+for \(int i = 0; i < train_size; i\+\+\) \{', train_loop, code)

train_end = '''        double val_bpb, val_acc;
        evaluate_model(model, num_features, use_residual, &val_bpb, &val_acc);
        if (val_bpb < best_bpb) {
            best_bpb = val_bpb;
            memcpy(best_adam, model, sizeof(AdamState));
        }
        // printf("    Epoch %d: Loss = %.4f | Val BPB = %.4f\\n", epoch+1, total_loss / train_size, val_bpb);
    }
    memcpy(model, best_adam, sizeof(AdamState));
    free(best_adam);
    free(gradW);'''
code = re.sub(r'        if \(\(epoch\+1\) % 5 == 0\) \{\s+printf\("  Epoch %d: Loss = %\.4f\\n", epoch\+1, total_loss / train_size\);\s+fflush\(stdout\);\s+\}\s+\}\s+free\(gradW\);', train_end, code)

# 4. Modify run_eval for LR sweep
run_eval_code = '''    double best_pure_bpb = 1e9, best_pure_acc = 0;
    double best_res_bpb = 1e9, best_res_acc = 0;
    float lrs[] = {0.01f, 0.003f, 0.001f};
    
    for (int l = 0; l < 3; l++) {
        AdamState* adam = (AdamState*)malloc(sizeof(AdamState));
        
        // Pure Model
        adam_init(adam);
        train_logistic_regression(adam, num_features, 10, 256, lrs[l], 0, 0);
        double bpb, acc;
        evaluate_model(adam, num_features, 0, &bpb, &acc);
        if (bpb < best_pure_bpb) { best_pure_bpb = bpb; best_pure_acc = acc; }
        
        // Residual Model
        adam_init(adam);
        train_logistic_regression(adam, num_features, 10, 256, lrs[l], 1, 0);
        evaluate_model(adam, num_features, 1, &bpb, &acc);
        if (bpb < best_res_bpb) { best_res_bpb = bpb; best_res_acc = acc; }
        
        free(adam);
    }
    printf("[%-20s] BPB: %.4f | Acc: %.2f%%\\n", name, best_pure_bpb, best_pure_acc);
    printf("[%-20s Residual] BPB: %.4f | Acc: %.2f%%\\n", name, best_res_bpb, best_res_acc);
    fflush(stdout);'''
code = re.sub(r'    AdamState\* adam = \(AdamState\*\)malloc\(sizeof\(AdamState\)\);\s+double bpb, acc;.*?fflush\(stdout\);\s*', run_eval_code + '\n', code, flags=re.DOTALL)

# 5. Add M4 Current-Only to main
code = code.replace('    run_eval("M4 Full 512D", 512, m4_extract_512);', '    run_eval("M4 Current 32D", 32, m4_extract_current_32);\n    run_eval("M4 Full 512D", 512, m4_extract_512);')

with open('benchmarks/benchmark10_audit.c', 'w') as f:
    f.write(code)

print("File written successfully.")
