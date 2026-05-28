import sys

with open("benchmarks/eval_entropy_stream.c", "r") as f:
    code = f.read()

# Add include
code = code.replace('#include "../src/silicon_entropy.h"', '#include "../src/silicon_entropy.h"\n#include "../src/range_coder.h"\n#define CDF_SCALE 16384')

# Add args
args_search = 'int tail_mode = 0; // 0=none, 1=ngram, 2=ngram+bias'
args_replace = '''int tail_mode = 0; // 0=none, 1=ngram, 2=ngram+bias
    const char* encode_path = NULL;
    const char* decode_path = NULL;
    const char* profile = NULL;'''
code = code.replace(args_search, args_replace)

args_parse = 'if (strcmp(argv[i], "--tail-mode") == 0 && i + 1 < argc) tail_mode = atoi(argv[++i]);'
args_parse_rep = '''if (strcmp(argv[i], "--tail-mode") == 0 && i + 1 < argc) tail_mode = atoi(argv[++i]);
        if (strcmp(argv[i], "--encode") == 0 && i + 1 < argc) encode_path = argv[++i];
        if (strcmp(argv[i], "--decode") == 0 && i + 1 < argc) decode_path = argv[++i];
        if (strcmp(argv[i], "--profile") == 0 && i + 1 < argc) profile = argv[++i];'''
code = code.replace(args_parse, args_parse_rep)

profile_setup = '''
    if (profile) {
        if (strcmp(profile, "full") == 0) {
            req_topk = 256; tail_mode = 0; req_topm = 256;
        } else if (strcmp(profile, "accurate") == 0) {
            req_topk = 64; tail_mode = 2; req_topm = 256;
        } else if (strcmp(profile, "fast") == 0) {
            req_topk = 48; tail_mode = 2; req_topm = 256;
        } else {
            printf("Error: Unknown profile '%s'\\n", profile);
            return 1;
        }
    }
'''
code = code.replace('FILE* fw = fopen(weights_path, "rb");', profile_setup + '\n    FILE* fw = fopen(weights_path, "rb");')

rank_override = '''
    if (profile && strcmp(profile, "full") != 0) {
        lr_rank = 0;
        printf("Profile '%s' selected: forcing lr_rank=0\\n", profile);
    }
'''
code = code.replace('float (*base_probs)[CLASSES][CLASSES] = NULL;', rank_override + '\n    float (*base_probs)[CLASSES][CLASSES] = NULL;')

# Set up Range Encoder/Decoder
setup_rc = '''
    RangeEncoder re;
    RangeDecoder rd;
    FILE* f_enc = NULL;
    FILE* f_dec = NULL;
    if (encode_path) {
        f_enc = fopen(encode_path, "wb");
        if (!f_enc) return 1;
        rc_encoder_init(&re, f_enc);
        printf("Encoding to %s\\n", encode_path);
    }
    if (decode_path) {
        f_dec = fopen(decode_path, "rb");
        if (!f_dec) return 1;
        rc_decoder_init(&rd, f_dec);
        printf("Decoding from %s\\n", decode_path);
    }
    
    uint64_t total_cdf_cyc = 0;
    uint64_t total_rc_cyc = 0;
'''
code = code.replace('uint8_t ctx2 = 0, ctx1 = 0;', setup_rc + '\n    uint8_t ctx2 = 0, ctx1 = 0;')

flush_rc = '''
    if (f_enc) {
        rc_encoder_flush(&re);
        fclose(f_enc);
    }
    if (f_dec) {
        fclose(f_dec);
    }
'''
code = code.replace('double bpb = total_loss / eval_len;', flush_rc + '\n    double bpb = total_loss / eval_len;')

# Replace the inner loop logic
# We need to find the `uint64_t t0 = __rdtsc();` block and rewrite it entirely.
loop_start = 'for (int i = 0; i < eval_len; i++) {'
loop_end = 'printf("\\n--- Dot Products / Byte ---'
inner_loop = code[code.find(loop_start):code.find(loop_end)]

new_inner_loop = '''for (int i = 0; i < eval_len; i++) {
        int global_idx = eval_start + 2 + i;
        uint8_t original_target = data[global_idx];
        uint8_t target = original_target; // Will be overwritten if decoding
        
        uint64_t t0 = __rdtsc();
        
        // 1. Extract Features
        float features[SEE_FEATURE_DIM];
        see_extract(&see, features);
        uint64_t t1 = __rdtsc();
        
        // 2. Normalize
        for (int f = 0; f < SEE_FEATURE_DIM; f++) {
            features[f] = (features[f] - feature_means[f]) / feature_stds[f];
        }
        uint64_t t2 = __rdtsc();
        
        // 3. Predict Logits
        uint64_t t_cand0 = __rdtsc();
        int cand_indices[CLASSES];
        float cand_logits[CLASSES];
        int num_cands = (tail_mode == 0) ? CLASSES : req_topk;
        
        if (tail_mode > 0 && lr_rank > 0) {
            float z[256]; 
            for (uint32_t r = 0; r < lr_rank; r++) {
                z[r] = dot_product_simd(&A_proj[r * SEE_FEATURE_DIM], features, SEE_FEATURE_DIM);
            }
            total_A_projs++;
            total_lowrank_cands += req_topm;
            
            for(int k=0; k<req_topk; k++) { cand_logits[k] = -1e9f; cand_indices[k] = 0; }
            for (int m = 0; m < req_topm; m++) {
                int c = topk_indices[ctx2][ctx1][m];
                float lr_val = 0;
                if (lr_rank == 16) {
                    __m256 b1 = _mm256_loadu_ps(&B_proj[c * 16]);
                    __m256 b2 = _mm256_loadu_ps(&B_proj[c * 16 + 8]);
                    __m256 z1 = _mm256_loadu_ps(&z[0]);
                    __m256 z2 = _mm256_loadu_ps(&z[8]);
                    __m256 sum = _mm256_fmadd_ps(b1, z1, _mm256_setzero_ps());
                    sum = _mm256_fmadd_ps(b2, z2, sum);
                    float out[8];
                    _mm256_storeu_ps(out, sum);
                    lr_val = out[0]+out[1]+out[2]+out[3]+out[4]+out[5]+out[6]+out[7];
                } else {
                    for (uint32_t r = 0; r < lr_rank; r++) lr_val += B_proj[c * lr_rank + r] * z[r];
                }
                float score = trigram_logits[ctx2][ctx1][c] + B[c] + lr_val;
                
                if (score > cand_logits[req_topk - 1]) {
                    int pos = req_topk - 1;
                    while (pos > 0 && score > cand_logits[pos - 1]) {
                        cand_logits[pos] = cand_logits[pos - 1];
                        cand_indices[pos] = cand_indices[pos - 1];
                        pos--;
                    }
                    cand_logits[pos] = score;
                    cand_indices[pos] = c;
                }
            }
        } else {
            for (int k = 0; k < num_cands; k++) {
                cand_indices[k] = (tail_mode == 0) ? k : topk_indices[ctx2][ctx1][k];
            }
        }
        
        uint64_t t_cand1 = __rdtsc();
        total_cand_cyc += (t_cand1 - t_cand0);
        
        uint64_t t_res0 = __rdtsc();
        float full_logits[CLASSES];
        float max_l = -1e9f;
        
        for (int k = 0; k < num_cands; k++) {
            int c = cand_indices[k];
            full_logits[k] = trigram_logits[ctx2][ctx1][c] + B[c] + dot_product_simd(W[c], features, SEE_FEATURE_DIM);
            if (full_logits[k] > max_l) max_l = full_logits[k];
        }
        float actual_max_l = max_l;
        if (max_l < 0) max_l = 0;
        
        uint64_t t_res1 = __rdtsc();
        total_res_cyc += (t_res1 - t_res0);
        
        uint64_t t_soft0 = __rdtsc();
        float Z_K = 0;
        for (int k = 0; k < num_cands; k++) Z_K += expf(full_logits[k] - max_l);
        
        float tail_mass = 0;
        if (tail_mode > 0) {
            tail_mass = Z_base[ctx2][ctx1];
            for (int k = 0; k < num_cands; k++) tail_mass -= base_probs[ctx2][ctx1][cand_indices[k]];
            if (tail_mass < 0) tail_mass = 0;
        }
        
        float Z_total = Z_K + tail_mass * expf(-max_l);
        float probs_tmp[CLASSES];
        
        uint64_t t_soft1 = __rdtsc();
        total_soft_cyc += (t_soft1 - t_soft0);
        
        // 4. Build CDF
        uint64_t t_cdf0 = __rdtsc();
        uint16_t freq[CLASSES];
        uint32_t total_assigned = 0;
        
        for (int c = 0; c < CLASSES; c++) {
            float p = 0;
            int idx = -1;
            for (int k = 0; k < num_cands; k++) {
                if (cand_indices[k] == c) { idx = k; break; }
            }
            if (idx != -1) {
                p = expf(full_logits[idx] - max_l) / Z_total;
            } else if (tail_mode > 0) {
                p = (base_probs[ctx2][ctx1][c] * expf(-max_l)) / Z_total;
            } else {
                p = 1e-10f;
            }
            probs_tmp[c] = p;
            
            int f = (int)(p * CDF_SCALE);
            if (f < 1) f = 1;
            freq[c] = (uint16_t)f;
            total_assigned += f;
        }
        
        // Normalize to exactly CDF_SCALE
        while (total_assigned > CDF_SCALE) {
            uint16_t max_f = 0; int max_c = 0;
            for (int c = 0; c < CLASSES; c++) {
                if (freq[c] > max_f) { max_f = freq[c]; max_c = c; }
            }
            freq[max_c]--;
            total_assigned--;
        }
        while (total_assigned < CDF_SCALE) {
            uint16_t max_f = 0; int max_c = 0;
            for (int c = 0; c < CLASSES; c++) {
                if (freq[c] > max_f) { max_f = freq[c]; max_c = c; }
            }
            freq[max_c]++;
            total_assigned++;
        }
        
        uint64_t t_cdf1 = __rdtsc();
        total_cdf_cyc += (t_cdf1 - t_cdf0);
        
        // 5. Encode / Decode
        uint64_t t_rc0 = __rdtsc();
        if (f_enc) {
            uint32_t cum = 0;
            for (int c = 0; c < target; c++) cum += freq[c];
            rc_encode(&re, cum, freq[target], CDF_SCALE);
        } else if (f_dec) {
            uint64_t f_val = rc_get_freq(&rd, CDF_SCALE);
            uint32_t cum = 0;
            target = 0;
            for (int c = 0; c < CLASSES; c++) {
                if (f_val >= cum && f_val < cum + freq[c]) {
                    target = c;
                    break;
                }
                cum += freq[c];
            }
            rc_decode(&rd, cum, freq[target], CDF_SCALE);
            
            if (target != original_target) {
                printf("DECODE ERROR at byte %d: expected %d, got %d\\n", i, original_target, target);
                break; // Stop decoding
            }
        }
        uint64_t t_rc1 = __rdtsc();
        total_rc_cyc += (t_rc1 - t_rc0);
        
        // Metrics & Loss (using final resolved target)
        uint64_t t3 = t_rc1;
        float final_prob = probs_tmp[target];
        if (final_prob < 1e-10f) final_prob = 1e-10f;
        double sample_loss = -log2(final_prob);
        total_loss += sample_loss;
        
        int best_c = cand_indices[0]; 
        for(int k=0; k<num_cands; k++) {
            if(full_logits[k] == actual_max_l) { best_c = cand_indices[k]; break; }
        }
        if (best_c == target) total_correct++;
        
        // 6. Observe actual target to update state
        see_observe(&see, target);
        ctx2 = ctx1;
        ctx1 = target;
        uint64_t t4 = __rdtsc();
        
        total_extract_cyc += (t1 - t0);
        total_norm_cyc += (t2 - t1);
        total_predict_cyc += (t_cdf0 - t2); // predict includes cand+res+soft
        total_observe_cyc += (t4 - t3);
        total_cyc += (t4 - t0);
    }
'''

code = code.replace(inner_loop, new_inner_loop + '\n    ')

# Add CDF / RC cycles to prints
metrics_search = 'printf("  - Softmax+Tail:  %.1f\\n", (double)total_soft_cyc / eval_len);'
metrics_rep = '''printf("  - Softmax+Tail:  %.1f\\n", (double)total_soft_cyc / eval_len);
    printf("  - CDF Build:     %.1f\\n", (double)total_cdf_cyc / eval_len);
    printf("  - Range Coder:   %.1f\\n", (double)total_rc_cyc / eval_len);'''
code = code.replace(metrics_search, metrics_rep)

with open("benchmarks/benchmark18_coder.c", "w") as f:
    f.write(code)
print("benchmark18_coder.c generated successfully!")
