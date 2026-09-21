/*
 * Diagnostic-only cross-input Q4_K path for STRAT-01 Rung 2B.
 * Reads two frozen ffn_norm payloads and the accepted block-0 up/gate
 * matrices. It never executes the donor graph and never self-adjudicates.
 */
#ifndef STRAT01_GGUF_RUNG2B_CROSS_INPUT_H
#define STRAT01_GGUF_RUNG2B_CROSS_INPUT_H

#define STRAT01_R2B_CROSS_INPUT_COUNT (8U*1536U)
#define STRAT01_R2B_CROSS_OUTPUT_COUNT (8U*STRAT01_R2B_FFN)
#define STRAT01_R2B_CROSS_Q8_BLOCKS (STRAT01_R2B_CROSS_INPUT_COUNT/STRAT01_QK_K)
#define STRAT01_R2B_CROSS_REF_INPUT_SHA "7bbfd9c2f0ed17c12a429b5dff83f75d47a45dd97008ad88251a7ec8d64dc588"
#define STRAT01_R2B_CROSS_C_INPUT_SHA "ea5d60791c4404fbb98e7839fa73ff8112c54abba6060a7fbc3d5e83b951e3e0"

static int strat01_r2b_cross_descriptor_ok(const strat01_tensor *t,const char *name,uint64_t offset,uint64_t file_offset) {
    return t&&!strcmp(t->name,name)&&t->type==STRAT01_GGML_Q4_K&&t->rank==2&&
           t->dims[0]==1536U&&t->dims[1]==STRAT01_R2B_FFN&&
           t->offset==offset&&t->span==UINT64_C(7741440)&&t->file_offset==file_offset;
}

static int strat01_r2b_cross_read_f32(const char *path,const char *expected_sha,float *out,char actual_sha[65],char error[256]) {
    FILE *f=NULL;uint64_t bytes=0;uint8_t raw[4];
    if(!strat01_sha256_file(path,actual_sha,&bytes,error)||bytes!=UINT64_C(49152)||strcmp(actual_sha,expected_sha)){
        if(!error[0])snprintf(error,256,"cross-input payload identity mismatch");return 0;
    }
    f=fopen(path,"rb");if(!f){snprintf(error,256,"cannot open cross-input payload");return 0;}
    for(size_t i=0;i<STRAT01_R2B_CROSS_INPUT_COUNT;++i){
        if(fread(raw,1,4,f)!=4){snprintf(error,256,"cross-input payload short read");fclose(f);return 0;}
        out[i]=strat01_r2a_f32le(raw);if(!isfinite(out[i])){snprintf(error,256,"cross-input payload non-finite");fclose(f);return 0;}
    }
    if(ferror(f)||fclose(f)!=0){snprintf(error,256,"cross-input payload I/O failure");return 0;}return 1;
}

static int strat01_r2b_cross_quantize(const float *input,strat01_q8_k_block *q8) {
    for(unsigned token=0;token<8U;++token)
        if(!strat01_quantize_q8_k_row(input+(size_t)token*1536U,1536U,q8+(size_t)token*6U))return 0;
    return 1;
}

static void strat01_r2b_cross_census(const strat01_q8_k_block *a,const strat01_q8_k_block *b,size_t blocks,size_t *changed_blocks,size_t *changed_bytes) {
    const uint8_t *pa=(const uint8_t *)a,*pb=(const uint8_t *)b;*changed_blocks=0;*changed_bytes=0;
    for(size_t block=0;block<blocks;++block){
        const size_t begin=block*sizeof(*a);int changed=0;
        for(size_t j=0;j<sizeof(*a);++j)if(pa[begin+j]!=pb[begin+j]){++*changed_bytes;changed=1;}
        if(changed)++*changed_blocks;
    }
}

static int strat01_r2b_cross_write_q8(const char *path,const strat01_q8_k_block *q8,size_t blocks,char sha[65],char error[256]) {
    FILE *f=fopen(path,"wb");uint64_t bytes=0;size_t written;int close_rc;
    if(!f){snprintf(error,256,"cannot create cross-input Q8 payload");return 0;}
    written=fwrite(q8,sizeof(*q8),blocks,f);close_rc=fclose(f);
    if(written!=blocks||close_rc!=0){snprintf(error,256,"cross-input Q8 write failure");return 0;}
    if(!strat01_sha256_file(path,sha,&bytes,error)||bytes!=blocks*sizeof(*q8)){if(!error[0])snprintf(error,256,"cross-input Q8 hash failure");return 0;}return 1;
}

static int strat01_r2b_cross_write_failure(const char *out_dir,const char *error) {
    char path[1024];FILE *f;if(!strat01_r2a_path(path,out_dir,"strat01_rung2b_cross_input.json")||(f=fopen(path,"wb"))==NULL)return 0;
    fputs("{\"command\":\"--strat01-rung2b-cross-input\",\"state\":\"DIAGNOSTIC_FAILURE\",\"error\":",f);strat01_json_string(f,error);fputs(",\"donor_graph_executions\":0}\n",f);fclose(f);return 1;
}

static int strat01_r2b_cross_input_cli(const char *model,const char *ref_input_path,const char *c_input_path,const char *out_dir,const char *engine_source_path) {
    strat01_inventory inv;const strat01_tensor *gate=NULL,*up=NULL;float *ref_input=NULL,*c_input=NULL,*ref_up=NULL,*ref_gate=NULL,*c_up=NULL,*c_gate=NULL;
    strat01_q8_k_block *ref_q8=NULL,*c_q8=NULL;char error[256]={0},model_sha[65]={0},ref_input_sha[65]={0},c_input_sha[65]={0},engine_sha[65]={0},header_sha[65]={0};
    char ref_up_path[1024],ref_gate_path[1024],c_up_path[1024],c_gate_path[1024],ref_q8_path[1024],c_q8_path[1024],report_path[1024];
    char ref_up_sha[65]={0},ref_gate_sha[65]={0},c_up_sha[65]={0},c_gate_sha[65]={0},ref_q8_sha[65]={0},c_q8_sha[65]={0};
    uint64_t hashed=0,parsed=0,tmp=0;size_t changed_blocks=0,changed_bytes=0;FILE *report=NULL;int ok=0;
    memset(&inv,0,sizeof(inv));
#if !defined(__clang__)
    snprintf(error,256,"cross-input diagnostic requires Clang");goto finish;
#endif
    if(sizeof(float)!=4||sizeof(strat01_q8_k_block)!=292U){snprintf(error,256,"unsupported cross-input host layout");goto finish;}
    if(!strat01_sha256_file(model,model_sha,&hashed,error)||hashed!=STRAT01_EXPECTED_SIZE||strcmp(model_sha,STRAT01_EXPECTED_SHA256)){if(!error[0])snprintf(error,256,"cross-input artifact identity mismatch");goto finish;}
    if(!strat01_parse_gguf(model,&inv,&parsed,error)||parsed!=hashed){if(!error[0])snprintf(error,256,"cross-input GGUF parse/size mismatch");goto finish;}
    gate=strat01_rung1_find_tensor(&inv,"blk.0.ffn_gate.weight");up=strat01_rung1_find_tensor(&inv,"blk.0.ffn_up.weight");
    if(!strat01_r2b_cross_descriptor_ok(gate,"blk.0.ffn_gate.weight",UINT64_C(298045440),UINT64_C(304148352))||
       !strat01_r2b_cross_descriptor_ok(up,"blk.0.ffn_up.weight",UINT64_C(305793024),UINT64_C(311895936))){snprintf(error,256,"cross-input frozen matrix descriptor mismatch");goto finish;}
    ref_input=strat01_r2a_alloc(STRAT01_R2B_CROSS_INPUT_COUNT,error);c_input=strat01_r2a_alloc(STRAT01_R2B_CROSS_INPUT_COUNT,error);
    ref_up=strat01_r2a_alloc(STRAT01_R2B_CROSS_OUTPUT_COUNT,error);ref_gate=strat01_r2a_alloc(STRAT01_R2B_CROSS_OUTPUT_COUNT,error);
    c_up=strat01_r2a_alloc(STRAT01_R2B_CROSS_OUTPUT_COUNT,error);c_gate=strat01_r2a_alloc(STRAT01_R2B_CROSS_OUTPUT_COUNT,error);
    ref_q8=(strat01_q8_k_block *)malloc(STRAT01_R2B_CROSS_Q8_BLOCKS*sizeof(*ref_q8));c_q8=(strat01_q8_k_block *)malloc(STRAT01_R2B_CROSS_Q8_BLOCKS*sizeof(*c_q8));
    if(!ref_input||!c_input||!ref_up||!ref_gate||!c_up||!c_gate||!ref_q8||!c_q8){if(!error[0])snprintf(error,256,"cross-input allocation failed");goto finish;}
    if(!strat01_r2b_cross_read_f32(ref_input_path,STRAT01_R2B_CROSS_REF_INPUT_SHA,ref_input,ref_input_sha,error)||
       !strat01_r2b_cross_read_f32(c_input_path,STRAT01_R2B_CROSS_C_INPUT_SHA,c_input,c_input_sha,error)||
       !strat01_r2b_cross_quantize(ref_input,ref_q8)||!strat01_r2b_cross_quantize(c_input,c_q8)){if(!error[0])snprintf(error,256,"cross-input Q8 quantization failed");goto finish;}
    strat01_r2b_cross_census(ref_q8,c_q8,STRAT01_R2B_CROSS_Q8_BLOCKS,&changed_blocks,&changed_bytes);
    if(!strat01_r2a_matmul_batch(model,up,ref_input,8U,1536U,ref_up,STRAT01_R2B_FFN,error)||
       !strat01_r2a_matmul_batch(model,gate,ref_input,8U,1536U,ref_gate,STRAT01_R2B_FFN,error)||
       !strat01_r2a_matmul_batch(model,up,c_input,8U,1536U,c_up,STRAT01_R2B_FFN,error)||
       !strat01_r2a_matmul_batch(model,gate,c_input,8U,1536U,c_gate,STRAT01_R2B_FFN,error))goto finish;
    if(!strat01_r2a_path(ref_up_path,out_dir,"reference_input_up.f32le")||!strat01_r2a_path(ref_gate_path,out_dir,"reference_input_gate.f32le")||
       !strat01_r2a_path(c_up_path,out_dir,"c_input_up.f32le")||!strat01_r2a_path(c_gate_path,out_dir,"c_input_gate.f32le")||
       !strat01_r2a_path(ref_q8_path,out_dir,"reference_input.q8k")||!strat01_r2a_path(c_q8_path,out_dir,"c_input.q8k")||
       !strat01_r2a_path(report_path,out_dir,"strat01_rung2b_cross_input.json")){snprintf(error,256,"cross-input output path overflow");goto finish;}
    if(!strat01_q4q8_write_output(ref_up_path,ref_up,STRAT01_R2B_CROSS_OUTPUT_COUNT,ref_up_sha,error)||
       !strat01_q4q8_write_output(ref_gate_path,ref_gate,STRAT01_R2B_CROSS_OUTPUT_COUNT,ref_gate_sha,error)||
       !strat01_q4q8_write_output(c_up_path,c_up,STRAT01_R2B_CROSS_OUTPUT_COUNT,c_up_sha,error)||
       !strat01_q4q8_write_output(c_gate_path,c_gate,STRAT01_R2B_CROSS_OUTPUT_COUNT,c_gate_sha,error)||
       !strat01_r2b_cross_write_q8(ref_q8_path,ref_q8,STRAT01_R2B_CROSS_Q8_BLOCKS,ref_q8_sha,error)||
       !strat01_r2b_cross_write_q8(c_q8_path,c_q8,STRAT01_R2B_CROSS_Q8_BLOCKS,c_q8_sha,error))goto finish;
    if(!strat01_sha256_file(engine_source_path,engine_sha,&tmp,error)||!strat01_sha256_file(__FILE__,header_sha,&tmp,error))goto finish;
    report=fopen(report_path,"wb");if(!report){snprintf(error,256,"cannot create cross-input report");goto finish;}
    fputs("{\n  \"command\": \"--strat01-rung2b-cross-input\",\n  \"state\": \"OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\",\n  \"self_certifies_pass\": false,\n  \"model\": {\"path\": ",report);strat01_json_string(report,model);fprintf(report,", \"bytes\": %" PRIu64 ", \"sha256\": ",hashed);strat01_json_string(report,model_sha);
    fputs("},\n  \"inputs\": {\"reference\": {\"path\": ",report);strat01_json_string(report,ref_input_path);fputs(", \"bytes\": 49152, \"sha256\": ",report);strat01_json_string(report,ref_input_sha);fputs("}, \"c_engine\": {\"path\": ",report);strat01_json_string(report,c_input_path);fputs(", \"bytes\": 49152, \"sha256\": ",report);strat01_json_string(report,c_input_sha);
    fputs("}},\n  \"matrices\": [{\"name\":\"blk.0.ffn_up.weight\",\"offset\":305793024,\"file_offset\":311895936},{\"name\":\"blk.0.ffn_gate.weight\",\"offset\":298045440,\"file_offset\":304148352}],\n  \"outputs\": {",report);
#define CROSS_OUTPUT_JSON(key,path,sha,comma) do{fputs(comma "\"" key "\":{\"path\":",report);strat01_json_string(report,path);fputs(",\"bytes\":286720,\"sha256\":",report);strat01_json_string(report,sha);fputc('}',report);}while(0)
    CROSS_OUTPUT_JSON("reference_input_up",ref_up_path,ref_up_sha,"");CROSS_OUTPUT_JSON("reference_input_gate",ref_gate_path,ref_gate_sha,",");CROSS_OUTPUT_JSON("c_input_up",c_up_path,c_up_sha,",");CROSS_OUTPUT_JSON("c_input_gate",c_gate_path,c_gate_sha,",");
#undef CROSS_OUTPUT_JSON
    fputs("},\n  \"q8\": {\"reference\":{\"path\":",report);strat01_json_string(report,ref_q8_path);fputs(",\"bytes\":14016,\"sha256\":",report);strat01_json_string(report,ref_q8_sha);fputs("},\"c_engine\":{\"path\":",report);strat01_json_string(report,c_q8_path);fputs(",\"bytes\":14016,\"sha256\":",report);strat01_json_string(report,c_q8_sha);fprintf(report,"},\"changed_blocks\":%zu,\"total_blocks\":48,\"changed_bytes\":%zu,\"total_bytes\":14016},\n",changed_blocks,changed_bytes);
    fputs("  \"engine_source_sha256\": ",report);strat01_json_string(report,engine_sha);fputs(",\n  \"diagnostic_source_sha256\": ",report);strat01_json_string(report,header_sha);fputs(",\n  \"compiler_family\": \"clang\",\n  \"donor_graph_executions\": 0,\n  \"timing_or_rate_claim\": null\n}\n",report);
    if(fclose(report)!=0){report=NULL;snprintf(error,256,"cross-input report close failure");goto finish;}report=NULL;ok=1;
finish:
    if(report)fclose(report);free(ref_input);free(c_input);free(ref_up);free(ref_gate);free(c_up);free(c_gate);free(ref_q8);free(c_q8);strat01_free_inventory(&inv);
    if(!ok){if(!error[0])snprintf(error,256,"unspecified cross-input failure");strat01_r2b_cross_write_failure(out_dir,error);fprintf(stderr,"STRAT-01 Rung-2B cross-input refused: %s\n",error);return 1;}
    fprintf(stderr,"STRAT-01 Rung-2B cross-input: OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\n");return 0;
}

static int strat01_r2b_cross_input_selftest(void) {
    int bad=0,checks=0;strat01_tensor t;strat01_q8_k_block a[2],b[2];size_t cb=0,cy=0;
#define CROSS_CHECK(x) do{++checks;if(!(x))++bad;}while(0)
    CROSS_CHECK(strat01_gguf_rung2b_selftest()==0);memset(&t,0,sizeof(t));t.name=(char *)"blk.0.ffn_up.weight";t.type=STRAT01_GGML_Q4_K;t.rank=2;t.dims[0]=1536;t.dims[1]=8960;t.offset=305793024;t.span=7741440;t.file_offset=311895936;
    CROSS_CHECK(strat01_r2b_cross_descriptor_ok(&t,"blk.0.ffn_up.weight",305793024,311895936));t.file_offset++;CROSS_CHECK(!strat01_r2b_cross_descriptor_ok(&t,"blk.0.ffn_up.weight",305793024,311895936));
    memset(a,0,sizeof(a));memset(b,0,sizeof(b));strat01_r2b_cross_census(a,b,2,&cb,&cy);CROSS_CHECK(cb==0&&cy==0);b[1].qs[3]=1;strat01_r2b_cross_census(a,b,2,&cb,&cy);CROSS_CHECK(cb==1&&cy==1);
#undef CROSS_CHECK
    fprintf(stderr,"STRAT-01 Rung-2B cross-input selftest: %s (%d checks)\n",bad?"FAIL":"PASS",checks);return bad?1:0;
}

#endif /* STRAT01_GGUF_RUNG2B_CROSS_INPUT_H */
