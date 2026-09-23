/* Post-F16 block-0 terminal components -> complete layer-1 diagnostic. */
#ifndef STRAT01_GGUF_POST_F16_BLOCK0_TERMINAL_COMPONENT_CROSS_INPUT_H
#define STRAT01_GGUF_POST_F16_BLOCK0_TERMINAL_COMPONENT_CROSS_INPUT_H

#define STRAT01_P16TERM_REF_INP_SHA "baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1"
#define STRAT01_P16TERM_REF_OUT_SHA "f5514fe59d64b517a685ec70004a75152a5f4d511417148f34e4d13b3134bef4"
#define STRAT01_P16TERM_REF_SUM_SHA "385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa"
#define STRAT01_P16TERM_CUR_SUM_SHA "7fb5fe52df684e4af82413004f6845133528241e3b030530d41595f74298bdc4"

typedef struct {
    const char *name,*inp_origin,*out_origin;float *sum;char sum_sha[65];
    strat01_r2a_arm attention;strat01_r2c_moe_arm moe;strat01_r2c_dump dumps[STRAT01_R2C_DUMPS];
} strat01_p16term_arm;

static void strat01_p16term_add(const float *a,const float *b,float *out) {
    for(size_t i=0;i<STRAT01_R2C_L1START_INPUT_COUNT;++i)out[i]=a[i]+b[i];
}

static void strat01_p16term_sha(const float *values,char hex[65]) {
    strat01_sha256 h;uint8_t digest[32];strat01_sha256_init(&h);
    for(size_t i=0;i<STRAT01_R2C_L1START_INPUT_COUNT;++i){uint8_t raw[4];strat01_rung1_put_f32le(raw,values[i]);strat01_sha256_update(&h,raw,4);}strat01_sha256_final(&h,digest);strat01_sha256_hex(digest,hex);
}

static int strat01_p16term_write_failure(const char *out_dir,const char *error) {
    char path[1024];FILE *f;if(!strat01_r2a_path(path,out_dir,"strat01_post_f16_block0_terminal_component_cross_input.json")||(f=fopen(path,"wb"))==NULL)return 0;
    fputs("{\"command\":\"--strat01-post-f16-block0-terminal-component-cross-input\",\"state\":\"DIAGNOSTIC_FAILURE\",\"error\":",f);strat01_json_string(f,error);fputs(",\"donor_graph_executions\":0,\"reference_graph_executions\":0}\n",f);fclose(f);return 1;
}

static int strat01_p16term_cli(const char *model,const char *r_inp_path,const char *r_out_path,const char *r_sum_path,const char *c_sum_path,const char *out_dir,const char *engine_source_path) {
    strat01_inventory inv;const strat01_tensor *base[8]={0},*b0ffn[4]={0},*attn[7]={0},*moe[9]={0};strat01_r2a_arm block0,n6,s7;strat01_r2b_arm dense;
    float *r_inp=NULL,*r_out=NULL,*r_sum=NULL,*c_sum=NULL,*r_rebuilt=NULL;strat01_p16term_arm arms[2]={{"current_attention_reference_ffn","current","reference",NULL,{0},{0},{0},{{0}}},{"reference_attention_current_ffn","reference","current",NULL,{0},{0},{0},{{0}}}};
    char error[256]={0},model_sha[65]={0},input_sha[4][65]={{0}},cur_inp_path[1024],cur_out_path[1024],cur_sum_path[1024],cur_inp_sha[65]={0},cur_out_sha[65]={0},cur_sum_sha[65]={0},n6_path[1024],s7_path[1024],n6_sha[65]={0},s7_sha[65]={0},engine_sha[65]={0},header_sha[65]={0},report_path[1024];
    uint64_t hashed=0,parsed=0,tmp=0;FILE *report=NULL;int ok=0;const char *input_paths[4]={r_inp_path,r_out_path,r_sum_path,c_sum_path};const char *input_names[4]={"reference_ffn_inp","reference_ffn_out","reference_l_out","current_l_out"};const char *input_expected[4]={STRAT01_P16TERM_REF_INP_SHA,STRAT01_P16TERM_REF_OUT_SHA,STRAT01_P16TERM_REF_SUM_SHA,STRAT01_P16TERM_CUR_SUM_SHA};float **input_dst[4]={&r_inp,&r_out,&r_sum,&c_sum};
    memset(&inv,0,sizeof(inv));memset(&block0,0,sizeof(block0));memset(&dense,0,sizeof(dense));memset(&n6,0,sizeof(n6));memset(&s7,0,sizeof(s7));strat01_f16vec_reset_counts();
#if !defined(__clang__)
    snprintf(error,256,"post-F16 terminal-component diagnostic requires Clang");goto finish;
#endif
    if(!strat01_sha256_file(model,model_sha,&hashed,error)||hashed!=STRAT01_EXPECTED_SIZE||strcmp(model_sha,STRAT01_EXPECTED_SHA256)){if(!error[0])snprintf(error,256,"post-F16 terminal-component artifact mismatch");goto finish;}
    if(!strat01_parse_gguf(model,&inv,&parsed,error)||parsed!=hashed){if(!error[0])snprintf(error,256,"post-F16 terminal-component GGUF parse mismatch");goto finish;}
    for(unsigned i=0;i<8U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2a_specs[i],&base[i],error))goto finish;for(unsigned i=0;i<4U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2b_specs[i],&b0ffn[i],error))goto finish;for(unsigned i=0;i<7U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2c_attn_specs[i],&attn[i],error))goto finish;for(unsigned i=0;i<9U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2c_moe_specs[i],&moe[i],error))goto finish;
    for(unsigned i=0;i<4U;++i){*input_dst[i]=strat01_r2a_alloc(STRAT01_R2C_L1START_INPUT_COUNT,error);if(!*input_dst[i]||!strat01_r2c_l1start_read(input_paths[i],input_expected[i],*input_dst[i],input_sha[i],error))goto finish;}r_rebuilt=strat01_r2a_alloc(STRAT01_R2C_L1START_INPUT_COUNT,error);if(!r_rebuilt)goto finish;
    if(!strat01_r2a_arm_alloc(&block0,error)||!strat01_r2b_arm_alloc(&dense,error)||!strat01_r2a_arm_alloc(&n6,error)||!strat01_r2a_arm_alloc(&s7,error))goto finish;
    for(unsigned i=0;i<2U;++i){arms[i].sum=strat01_r2a_alloc(STRAT01_R2C_L1START_INPUT_COUNT,error);if(!arms[i].sum||!strat01_r2a_arm_alloc(&arms[i].attention,error)||!strat01_r2c_moe_alloc(&arms[i].moe,error))goto finish;}
    if(!strat01_r2a_build_upstream_range(model,base,&block0,0,8,error)||!strat01_r2a_run_schedule(model,base[6],base[7],&block0,0,error)||!strat01_r2b_run(model,b0ffn[0],b0ffn[1],b0ffn[2],b0ffn[3],&block0,&dense,error))goto finish;
    strat01_p16term_add(r_inp,r_out,r_rebuilt);if(memcmp(r_rebuilt,r_sum,STRAT01_R2C_L1START_INPUT_BYTES)){snprintf(error,256,"post-F16 reference homogeneous reconstruction mismatch");goto finish;}if(memcmp(dense.l_out,c_sum,STRAT01_R2C_L1START_INPUT_BYTES)){snprintf(error,256,"post-F16 current homogeneous reconstruction mismatch");goto finish;}
    strat01_p16term_add(block0.ffn_inp,r_out,arms[0].sum);strat01_p16term_add(r_inp,dense.out,arms[1].sum);for(unsigned i=0;i<2U;++i)strat01_p16term_sha(arms[i].sum,arms[i].sum_sha);
    for(unsigned i=0;i<2U;++i)if(!strat01_postf16_run_full(model,attn,moe,arms[i].sum,&arms[i].attention,&arms[i].moe,error)||!strat01_postf16_write_arm(out_dir,arms[i].name,arms[i].sum,&arms[i].attention,&arms[i].moe,arms[i].dumps,error))goto finish;
    if(!strat01_postf16_run_attention(model,attn,arms[0].sum,1,0,&n6,error)||!strat01_postf16_run_attention(model,attn,arms[0].sum,0,1,&s7,error))goto finish;
    if(!strat01_r2a_path(cur_inp_path,out_dir,"current_ffn_inp-0.f32le")||!strat01_r2a_path(cur_out_path,out_dir,"current_ffn_out-0.f32le")||!strat01_r2a_path(cur_sum_path,out_dir,"current_l_out-0.f32le")||!strat01_q4q8_write_output(cur_inp_path,block0.ffn_inp,STRAT01_R2C_L1START_INPUT_COUNT,cur_inp_sha,error)||!strat01_q4q8_write_output(cur_out_path,dense.out,STRAT01_R2C_L1START_INPUT_COUNT,cur_out_sha,error)||!strat01_q4q8_write_output(cur_sum_path,dense.l_out,STRAT01_R2C_L1START_INPUT_COUNT,cur_sum_sha,error))goto finish;
    if(strcmp(cur_sum_sha,STRAT01_P16TERM_CUR_SUM_SHA)){snprintf(error,256,"post-F16 current sum SHA mismatch");goto finish;}
    if(!strat01_r2a_path(n6_path,out_dir,"control_current_attention_reference_ffn_token6_negated_kqv_out-1.f32le")||!strat01_r2a_path(s7_path,out_dir,"control_current_attention_reference_ffn_rows0_7_swapped_kqv_out-1.f32le")||!strat01_q4q8_write_output(n6_path,n6.kqv_out,8U*6144U,n6_sha,error)||!strat01_q4q8_write_output(s7_path,s7.kqv_out,8U*6144U,s7_sha,error)||!strat01_f16v_write_counts(out_dir,error))goto finish;
    if(!strat01_sha256_file(engine_source_path,engine_sha,&tmp,error)||!strat01_sha256_file(__FILE__,header_sha,&tmp,error))goto finish;
    if(!strat01_r2a_path(report_path,out_dir,"strat01_post_f16_block0_terminal_component_cross_input.json")||(report=fopen(report_path,"wb"))==NULL){snprintf(error,256,"cannot write post-F16 terminal-component report");goto finish;}
    fputs("{\n\"command\":\"--strat01-post-f16-block0-terminal-component-cross-input\",\n\"state\":\"OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\",\n\"self_certifies_pass\":false,\n\"model\":{\"path\":",report);strat01_json_string(report,model);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",hashed);strat01_json_string(report,model_sha);fputs("},\n\"inputs\":{",report);
    for(unsigned i=0;i<4U;++i){fprintf(report,"%s\"%s\":{\"path\":",i?",":"",input_names[i]);strat01_json_string(report,input_paths[i]);fputs(",\"bytes\":49152,\"sha256\":",report);strat01_json_string(report,input_sha[i]);fputc('}',report);}fputs("},\n\"tensors\":[",report);
    for(unsigned i=0;i<28U;++i){const strat01_tensor *t=i<8U?base[i]:(i<12U?b0ffn[i-8U]:(i<19U?attn[i-12U]:moe[i-19U]));fprintf(report,"%s{\"name\":",i?",":"");strat01_json_string(report,t->name);fprintf(report,",\"type\":%u,\"offset\":%" PRIu64 ",\"file_offset\":%" PRIu64 ",\"span\":%" PRIu64 "}",t->type,t->offset,t->file_offset,t->span);}fputs("],\n\"current_components\":{\"ffn_inp-0\":{\"path\":",report);strat01_json_string(report,cur_inp_path);fputs(",\"bytes\":49152,\"sha256\":",report);strat01_json_string(report,cur_inp_sha);fputs("},\"ffn_out-0\":{\"path\":",report);strat01_json_string(report,cur_out_path);fputs(",\"bytes\":49152,\"sha256\":",report);strat01_json_string(report,cur_out_sha);fputs("},\"l_out-0\":{\"path\":",report);strat01_json_string(report,cur_sum_path);fputs(",\"bytes\":49152,\"sha256\":",report);strat01_json_string(report,cur_sum_sha);fputs("}},\n\"compositions\":{",report);
    for(unsigned a=0;a<2U;++a){fprintf(report,"%s\"%s\":{\"ffn_inp_origin\":",a?",":"",arms[a].name);strat01_json_string(report,arms[a].inp_origin);fputs(",\"ffn_out_origin\":",report);strat01_json_string(report,arms[a].out_origin);fputs(",\"bytes\":49152,\"sha256\":",report);strat01_json_string(report,arms[a].sum_sha);fputc('}',report);}fputs("},\n\"outputs\":{\"current_attention_reference_ffn\":",report);strat01_postf16_report_arm(report,arms[0].dumps);fputs(",\"reference_attention_current_ffn\":",report);strat01_postf16_report_arm(report,arms[1].dumps);
    fputs("},\n\"controls\":{\"hybrid_token6_negated\":{\"path\":",report);strat01_json_string(report,n6_path);fputs(",\"bytes\":196608,\"sha256\":",report);strat01_json_string(report,n6_sha);fputs("},\"hybrid_rows0_7_swapped\":{\"path\":",report);strat01_json_string(report,s7_path);fputs(",\"bytes\":196608,\"sha256\":",report);strat01_json_string(report,s7_sha);fputs("}},\n\"homogeneous_replay\":{\"reference\":true,\"current\":true},\n\"engine_source_sha256\":",report);strat01_json_string(report,engine_sha);fputs(",\n\"diagnostic_source_sha256\":",report);strat01_json_string(report,header_sha);fputs(",\n\"compiler_family\":\"clang\",\n\"donor_graph_executions\":0,\n\"reference_graph_executions\":0,\n\"timing_or_rate_claim\":null\n}\n",report);
    if(fclose(report)!=0){report=NULL;snprintf(error,256,"post-F16 terminal-component report close failure");goto finish;}report=NULL;ok=1;
finish:
    if(report)fclose(report);for(unsigned i=0;i<4U;++i)free(*input_dst[i]);free(r_rebuilt);strat01_r2a_arm_free(&block0);strat01_r2b_arm_free(&dense);strat01_r2a_arm_free(&n6);strat01_r2a_arm_free(&s7);for(unsigned i=0;i<2U;++i){free(arms[i].sum);strat01_r2a_arm_free(&arms[i].attention);strat01_r2c_moe_free(&arms[i].moe);}strat01_free_inventory(&inv);
    if(!ok){if(!error[0])snprintf(error,256,"unspecified post-F16 terminal-component failure");strat01_p16term_write_failure(out_dir,error);fprintf(stderr,"STRAT-01 post-F16 terminal-component refused: %s\n",error);return 1;}fprintf(stderr,"STRAT-01 post-F16 terminal-component: OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\n");return 0;
}

static int strat01_p16term_selftest(void) {
    int bad=0,checks=0;float a[STRAT01_R2C_L1START_INPUT_COUNT]={0},b[STRAT01_R2C_L1START_INPUT_COUNT]={0},s[STRAT01_R2C_L1START_INPUT_COUNT]={0};char sha[65]={0};a[0]=1;a[1]=-2;a[2]=.25f;a[3]=16777216;b[0]=2;b[1]=.5f;b[2]=.75f;b[3]=1;
#define P16TERM_CHECK(z) do{++checks;if(!(z))++bad;}while(0)
    P16TERM_CHECK(strlen(STRAT01_P16TERM_REF_INP_SHA)==64U);P16TERM_CHECK(strlen(STRAT01_P16TERM_REF_OUT_SHA)==64U);P16TERM_CHECK(strlen(STRAT01_P16TERM_CUR_SUM_SHA)==64U);strat01_p16term_add(a,b,s);P16TERM_CHECK(s[0]==3&&s[1]==-1.5f&&s[2]==1&&s[3]==16777216);strat01_p16term_sha(s,sha);P16TERM_CHECK(strlen(sha)==64U);
#undef P16TERM_CHECK
    fprintf(stderr,"STRAT-01 post-F16 terminal-component selftest: %s (%d checks)\n",bad?"FAIL":"PASS",checks);return bad?1:0;
}

#endif
