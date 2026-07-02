import os

with open("benchmarks/benchmark18_coder.c", "r") as f:
    content = f.read()

# Remove the vars from around 426
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
content = content.replace(vars_text, "")

# Insert at the beginning of main
main_start = content.find('    enum Mode mode = MODE_NONE;')
content = content[:main_start] + vars_text + content[main_start:]

# Fix the printf at the end of the file. It's currently at line 817.
# Let's locate the printf block.
printf_block = """
        printf("--- Phase 21 Oracle / Telemetry ---\\n");
        printf("SEE Only BPB:    %.4f\\n", total_loss_see_only / eval_len);
        printf("Dyn Only BPB:    %.4f\\n", total_loss_dyn_only / eval_len);
        printf("Oracle BPB:      %.4f\\n", total_loss_oracle / eval_len);
        printf("Avg H_see:       %.4f\\n", total_H_see / eval_len);
        printf("Avg H_dyn:       %.4f\\n", total_H_dyn / eval_len);
"""
content = content.replace(printf_block, "")

# Find where to put the printf block. It should be inside `if (mode == MODE_EVAL || mode == MODE_ENCODE) {` block where it prints metrics.
# We'll put it right after `printf("Total:           %.1f\\n", (double)total_cyc / eval_len + rc);`
target_print = 'printf("Total:           %.1f\\n", (double)total_cyc / eval_len + rc);'
if target_print in content:
    idx = content.find(target_print) + len(target_print)
    content = content[:idx] + "\\n" + printf_block + content[idx:]

with open("benchmarks/benchmark18_coder.c", "w") as f:
    f.write(content)
print("Syntax fixed.")
