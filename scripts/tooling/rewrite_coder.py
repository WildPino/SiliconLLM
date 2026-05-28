import os

with open("benchmarks/benchmark18_coder.c", "r") as f:
    content = f.read()

# We need to replace the data loading logic and everything down to the evaluation loop
# First, let's find the `FILE* f = fopen(dataset_path, "rb");`
data_load_idx = content.find('FILE* f = fopen(dataset_path, "rb");')
if data_load_idx == -1:
    print("Cannot find dataset_path fopen")
    exit(1)
    
# Find the end of warmup loop
main_loop_idx = content.find('for (int i = 0; i < eval_len; i++) {')

if main_loop_idx == -1:
    print("Cannot find main loop")
    exit(1)

replacement = """
    uint8_t* data = NULL;
    size_t data_size = 0;
    
    ArchiveHeader arc_hdr = {0};
    FILE* f_enc = NULL;
    FILE* f_dec = NULL;
    FILE* f_dump = NULL;
    RangeEncoder re;
    RangeDecoder rd;
    
    int eval_start = 0;
    int eval_len = 0;
    
    if (mode == MODE_ENCODE || mode == MODE_EVAL) {
        FILE* f = fopen(input_path, "rb");
        if (!f) {
            printf("Error: Could not open %s\\n", input_path);
            return 1;
        }
        fseek(f, 0, SEEK_END);
        data_size = ftell(f);
        fseek(f, 0, SEEK_SET);
        data = malloc(data_size);
        fread(data, 1, data_size, f);
        fclose(f);
        
        if (do_shuffle) {
            printf("Applying global byte shuffle (seed=42) to evaluation data...\\n");
            srand(42);
            for(int i = data_size - 1; i > 0; i--) {
                int j = rand() % (i + 1);
                uint8_t temp = data[i];
                data[i] = data[j];
                data[j] = temp;
            }
        }
        
        if (mode == MODE_EVAL) {
            eval_start = (data_size * eval_start_pct) / 100;
            eval_len = (data_size * eval_len_pct) / 100;
            if (eval_start + eval_len + 2 > data_size) {
                eval_len = data_size - eval_start - 2;
            }
        } else { // ENCODE
            eval_start = 0;
            eval_len = data_size >= 2 ? data_size - 2 : 0;
            
            arc_hdr.magic = 0x32454553; // "SEE2"
            arc_hdr.original_size = data_size;
            arc_hdr.req_topk = req_topk;
            arc_hdr.tail_mode = tail_mode;
            arc_hdr.blend_lambda = blend_lambda;
            arc_hdr.chunk_size = header.chunk_size;
            arc_hdr.decay = header.decay;
            arc_hdr.codebook_seed = header.codebook_seed;
            arc_hdr.seed_byte0 = data_size > 0 ? data[0] : 0;
            arc_hdr.seed_byte1 = data_size > 1 ? data[1] : 0;
            
            f_enc = fopen(archive_path, "wb");
            if (!f_enc) return 1;
            setvbuf(f_enc, NULL, _IOFBF, 65536);
            fwrite(&arc_hdr, sizeof(ArchiveHeader), 1, f_enc);
            rc_encoder_init(&re, f_enc);
        }
    } else if (mode == MODE_DECODE) {
        f_dec = fopen(archive_path, "rb");
        if (!f_dec) return 1;
        setvbuf(f_dec, NULL, _IOFBF, 65536);
        if (fread(&arc_hdr, sizeof(ArchiveHeader), 1, f_dec) != 1) {
            printf("Error reading archive header.\\n");
            return 1;
        }
        if (arc_hdr.magic != 0x32454553) {
            printf("Error: Invalid archive magic.\\n");
            return 1;
        }
        
        req_topk = arc_hdr.req_topk;
        tail_mode = arc_hdr.tail_mode;
        blend_lambda = arc_hdr.blend_lambda;
        
        if (output_path) {
            f_dump = fopen(output_path, "wb");
            if (!f_dump) return 1;
            setvbuf(f_dump, NULL, _IOFBF, 65536);
            if (arc_hdr.original_size > 0) fwrite(&arc_hdr.seed_byte0, 1, 1, f_dump);
            if (arc_hdr.original_size > 1) fwrite(&arc_hdr.seed_byte1, 1, 1, f_dump);
        }
        
        data_size = arc_hdr.original_size;
        eval_start = 0;
        eval_len = data_size >= 2 ? data_size - 2 : 0;
        
        rc_decoder_init(&rd, f_dec);
    }
    
    SiliconEntropyState see;
    see_init(&see, header.codebook_seed, header.chunk_size, header.decay);
    
    uint64_t total_cdf_cyc = 0;
    uint64_t total_rc_cyc = 0;

    uint8_t ctx2 = 0, ctx1 = 0;
    
    // Dynamic counts for Phase 20
    uint32_t dyn_uni_counts[CLASSES] = {0};
    uint16_t dyn_bi_counts[CLASSES][CLASSES] = {{0}};
    uint32_t dyn_total_count = 0;
    
    // Warmup
    if (mode == MODE_EVAL || mode == MODE_ENCODE) {
        for (int i = 0; i <= eval_start + 1 && i < data_size; i++) {
            see_observe(&see, data[i]);
            dyn_uni_counts[data[i]]++;
            dyn_bi_counts[ctx1][data[i]]++;
            dyn_total_count++;
            
            ctx2 = ctx1;
            ctx1 = data[i];
        }
    } else if (mode == MODE_DECODE) {
        if (data_size > 0) {
            see_observe(&see, arc_hdr.seed_byte0);
            dyn_uni_counts[arc_hdr.seed_byte0]++;
            dyn_bi_counts[ctx1][arc_hdr.seed_byte0]++;
            dyn_total_count++;
            ctx2 = ctx1;
            ctx1 = arc_hdr.seed_byte0;
        }
        if (data_size > 1) {
            see_observe(&see, arc_hdr.seed_byte1);
            dyn_uni_counts[arc_hdr.seed_byte1]++;
            dyn_bi_counts[ctx1][arc_hdr.seed_byte1]++;
            dyn_total_count++;
            ctx2 = ctx1;
            ctx1 = arc_hdr.seed_byte1;
        }
    }
    
"""

content = content[:data_load_idx] + replacement + content[main_loop_idx:]

with open("benchmarks/benchmark18_coder.c", "w") as f:
    f.write(content)
print("Updated successfully")
