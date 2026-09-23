/* Frozen block-0 terminal-addend -> layer-1 propagation diagnostic. */
#ifndef STRAT01_GGUF_BLOCK0_TERMINAL_COMPONENT_CROSS_INPUT_H
#define STRAT01_GGUF_BLOCK0_TERMINAL_COMPONENT_CROSS_INPUT_H

#define STRAT01_TERM_C_INP_SHA  "13721a425bf93e87aa313b9ea7718fdcc2d2ce25f9ec38403eee9fc6cce9847e"
#define STRAT01_TERM_C_OUT_SHA  "8433164833fe4daecd24a840bb4cd70c41d390a15655abe479ef058248a6fb4f"
#define STRAT01_TERM_C_SUM_SHA  STRAT01_R2C_L1START_C_SHA
#define STRAT01_TERM_R_INP_SHA  "baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1"
#define STRAT01_TERM_R_OUT_SHA  "f5514fe59d64b517a685ec70004a75152a5f4d511417148f34e4d13b3134bef4"
#define STRAT01_TERM_R_SUM_SHA  STRAT01_R2C_L1START_REF_SHA

typedef struct {
    const char *name;
    const char *inp_origin;
    const char *out_origin;
    float *sum;
    strat01_r2a_arm state;
    strat01_r2c_l1start_dump dumps[11];
    char sum_sha[65];
} strat01_term_arm;

static void strat01_term_compose(const float *inp,const float *out,float *sum) {
    for(size_t i=0;i<STRAT01_R2C_L1START_INPUT_COUNT;++i)sum[i]=inp[i]+out[i];
}

static void strat01_term_sha_f32(const float *values,char hex[65]) {
    strat01_sha256 sha;uint8_t digest[32];strat01_sha256_init(&sha);
    for(size_t i=0;i<STRAT01_R2C_L1START_INPUT_COUNT;++i){uint8_t b[4];strat01_rung1_put_f32le(b,values[i]);strat01_sha256_update(&sha,b,4);}
    strat01_sha256_final(&sha,digest);strat01_sha256_hex(digest,hex);
}

static int strat01_term_write_failure(const char *out_dir,const char *error) {
    char path[1024];FILE *f;if(!strat01_r2a_path(path,out_dir,"strat01_block0_terminal_component_cross_input.json")||(f=fopen(path,"wb"))==NULL)return 0;
    fputs("{\"command\":\"--strat01-block0-terminal-component-cross-input\",\"state\":\"DIAGNOSTIC_FAILURE\",\"error\":",f);strat01_json_string(f,error);fputs(",\"donor_graph_executions\":0,\"reference_graph_executions\":0}\n",f);fclose(f);return 1;
}

static int strat01_term_cli(const char *model,const char *c_inp_path,const char *c_out_path,const char *c_sum_path,const char *r_inp_path,const char *r_out_path,const char *r_sum_path,const char *out_dir,const char *engine_source_path) {
    strat01_inventory inv;const strat01_tensor *t[7]={0};float *c_inp=NULL,*c_out=NULL,*c_sum_target=NULL,*r_inp=NULL,*r_out=NULL,*r_sum_target=NULL;
    strat01_term_arm arms[4]={{"reference_reference","reference","reference",NULL,{0},{{0}},{0}},{"c_c","c","c",NULL,{0},{{0}},{0}},{"c_attention_reference_ffn","c","reference",NULL,{0},{{0}},{0}},{"reference_attention_c_ffn","reference","c",NULL,{0},{{0}},{0}}};
    strat01_r2a_arm n7,s7;FILE *report=NULL;int ok=0;char error[256]={0},model_sha[65]={0},input_sha[6][65]={{0}},engine_sha[65]={0},header_sha[65]={0},n7_path[1024],s7_path[1024],n7_sha[65]={0},s7_sha[65]={0},report_path[1024];uint64_t hashed=0,parsed=0,tmp=0;
    const char *input_paths[6]={c_inp_path,c_out_path,c_sum_path,r_inp_path,r_out_path,r_sum_path};const char *input_names[6]={"c_ffn_inp","c_ffn_out","c_l_out","reference_ffn_inp","reference_ffn_out","reference_l_out"};const char *input_expected[6]={STRAT01_TERM_C_INP_SHA,STRAT01_TERM_C_OUT_SHA,STRAT01_TERM_C_SUM_SHA,STRAT01_TERM_R_INP_SHA,STRAT01_TERM_R_OUT_SHA,STRAT01_TERM_R_SUM_SHA};float **input_dst[6]={&c_inp,&c_out,&c_sum_target,&r_inp,&r_out,&r_sum_target};
    memset(&inv,0,sizeof(inv));memset(&n7,0,sizeof(n7));memset(&s7,0,sizeof(s7));
#if !defined(__clang__)
    snprintf(error,256,"terminal-component diagnostic requires Clang");goto finish;
#endif
    if(!strat01_sha256_file(model,model_sha,&hashed,error)||hashed!=STRAT01_EXPECTED_SIZE||strcmp(model_sha,STRAT01_EXPECTED_SHA256)){if(!error[0])snprintf(error,256,"terminal-component artifact mismatch");goto finish;}
    if(!strat01_parse_gguf(model,&inv,&parsed,error)||parsed!=hashed){if(!error[0])snprintf(error,256,"terminal-component GGUF parse mismatch");goto finish;}
    for(unsigned i=0;i<7U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2c_attn_specs[i],&t[i],error))goto finish;
    for(unsigned i=0;i<6U;++i){*input_dst[i]=strat01_r2a_alloc(STRAT01_R2C_L1START_INPUT_COUNT,error);if(!*input_dst[i]||!strat01_r2c_l1start_read(input_paths[i],input_expected[i],*input_dst[i],input_sha[i],error))goto finish;}
    for(unsigned i=0;i<4U;++i){arms[i].sum=strat01_r2a_alloc(STRAT01_R2C_L1START_INPUT_COUNT,error);if(!arms[i].sum||!strat01_r2a_arm_alloc(&arms[i].state,error))goto finish;}
    if(!strat01_r2a_arm_alloc(&n7,error)||!strat01_r2a_arm_alloc(&s7,error))goto finish;
    strat01_term_compose(r_inp,r_out,arms[0].sum);strat01_term_compose(c_inp,c_out,arms[1].sum);strat01_term_compose(c_inp,r_out,arms[2].sum);strat01_term_compose(r_inp,c_out,arms[3].sum);
    for(unsigned i=0;i<4U;++i)strat01_term_sha_f32(arms[i].sum,arms[i].sum_sha);
    if(memcmp(arms[0].sum,r_sum_target,STRAT01_R2C_L1START_INPUT_BYTES)||strcmp(arms[0].sum_sha,STRAT01_TERM_R_SUM_SHA)){snprintf(error,256,"reference homogeneous reconstruction mismatch");goto finish;}
    if(memcmp(arms[1].sum,c_sum_target,STRAT01_R2C_L1START_INPUT_BYTES)||strcmp(arms[1].sum_sha,STRAT01_TERM_C_SUM_SHA)){snprintf(error,256,"C homogeneous reconstruction mismatch");goto finish;}
    for(unsigned i=0;i<4U;++i){if(!strat01_r2c_l1start_run(model,t,arms[i].sum,0,0,&arms[i].state,error))goto finish;strat01_r2c_l1start_make_dumps(&arms[i].state,arms[i].dumps);if(!strat01_r2c_l1start_write_arm(out_dir,arms[i].name,arms[i].dumps,error))goto finish;}
    if(!strat01_r2c_l1start_run(model,t,arms[2].sum,1,0,&n7,error)||!strat01_r2c_l1start_run(model,t,arms[2].sum,0,1,&s7,error))goto finish;
    if(!strat01_r2a_path(n7_path,out_dir,"control_hybrid_token7_negated_kqv_out-1.f32le")||!strat01_r2a_path(s7_path,out_dir,"control_hybrid_rows0_7_swapped_kqv_out-1.f32le")||!strat01_q4q8_write_output(n7_path,n7.kqv_out,8U*6144U,n7_sha,error)||!strat01_q4q8_write_output(s7_path,s7.kqv_out,8U*6144U,s7_sha,error))goto finish;
    if(!strat01_sha256_file(engine_source_path,engine_sha,&tmp,error)||!strat01_sha256_file(__FILE__,header_sha,&tmp,error))goto finish;
    if(!strat01_r2a_path(report_path,out_dir,"strat01_block0_terminal_component_cross_input.json")||(report=fopen(report_path,"wb"))==NULL){snprintf(error,256,"cannot write terminal-component report");goto finish;}
    fputs("{\n\"command\":\"--strat01-block0-terminal-component-cross-input\",\n\"state\":\"OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\",\n\"self_certifies_pass\":false,\n\"model\":{\"path\":",report);strat01_json_string(report,model);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",hashed);strat01_json_string(report,model_sha);fputs("},\n\"inputs\":{",report);
    for(unsigned i=0;i<6U;++i){fprintf(report,"%s\"%s\":{\"path\":",i?",":"",input_names[i]);strat01_json_string(report,input_paths[i]);fputs(",\"bytes\":49152,\"sha256\":",report);strat01_json_string(report,input_sha[i]);fputc('}',report);}fputs("},\n\"tensors\":[",report);
    for(unsigned i=0;i<7U;++i){fprintf(report,"%s{\"name\":",i?",":"");strat01_json_string(report,t[i]->name);fprintf(report,",\"type\":%u,\"offset\":%" PRIu64 ",\"file_offset\":%" PRIu64 ",\"span\":%" PRIu64 "}",t[i]->type,t[i]->offset,t[i]->file_offset,t[i]->span);}fputs("],\n\"compositions\":{",report);
    for(unsigned a=0;a<4U;++a){fprintf(report,"%s\"%s\":{\"ffn_inp_origin\":",a?",":"",arms[a].name);strat01_json_string(report,arms[a].inp_origin);fputs(",\"ffn_out_origin\":",report);strat01_json_string(report,arms[a].out_origin);fputs(",\"bytes\":49152,\"sha256\":",report);strat01_json_string(report,arms[a].sum_sha);fprintf(report,",\"homogeneous_replay\":%s}",a<2U?"true":"false");}fputs("},\n\"outputs\":{",report);
    for(unsigned a=0;a<4U;++a){fprintf(report,"%s\"%s\":{",a?",":"",arms[a].name);for(unsigned i=0;i<11U;++i){strat01_r2c_l1start_dump *d=&arms[a].dumps[i];fprintf(report,"%s\"%s\":{\"path\":",i?",":"",d->name);strat01_json_string(report,d->path);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",d->count*4U);strat01_json_string(report,d->sha);fputc('}',report);}fputc('}',report);}fputs("},\n\"controls\":{\"hybrid_token7_negated\":{\"path\":",report);strat01_json_string(report,n7_path);fputs(",\"bytes\":196608,\"sha256\":",report);strat01_json_string(report,n7_sha);fputs("},\"hybrid_rows0_7_swapped\":{\"path\":",report);strat01_json_string(report,s7_path);fputs(",\"bytes\":196608,\"sha256\":",report);strat01_json_string(report,s7_sha);fputs("}},\n\"engine_source_sha256\":",report);strat01_json_string(report,engine_sha);fputs(",\n\"diagnostic_source_sha256\":",report);strat01_json_string(report,header_sha);fputs(",\n\"compiler_family\":\"clang\",\n\"donor_graph_executions\":0,\n\"reference_graph_executions\":0,\n\"timing_or_rate_claim\":null\n}\n",report);
    if(fclose(report)!=0){report=NULL;snprintf(error,256,"terminal-component report close failure");goto finish;}report=NULL;ok=1;
finish:
    if(report)fclose(report);for(unsigned i=0;i<6U;++i)free(*input_dst[i]);for(unsigned i=0;i<4U;++i){free(arms[i].sum);strat01_r2a_arm_free(&arms[i].state);}strat01_r2a_arm_free(&n7);strat01_r2a_arm_free(&s7);strat01_free_inventory(&inv);
    if(!ok){if(!error[0])snprintf(error,256,"unspecified terminal-component failure");strat01_term_write_failure(out_dir,error);fprintf(stderr,"STRAT-01 terminal-component refused: %s\n",error);return 1;}fprintf(stderr,"STRAT-01 terminal-component: OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\n");return 0;
}

static int strat01_term_selftest(void) {
    int bad=0,checks=0;float a[4]={1.0f,-2.0f,0.25f,16777216.0f},b[4]={2.0f,0.5f,0.75f,1.0f},s[4]={0};char sha[65]={0};
#define TERM_CHECK(z) do{++checks;if(!(z))++bad;}while(0)
    TERM_CHECK(strlen(STRAT01_TERM_C_INP_SHA)==64U);TERM_CHECK(strlen(STRAT01_TERM_C_OUT_SHA)==64U);TERM_CHECK(strlen(STRAT01_TERM_R_INP_SHA)==64U);TERM_CHECK(strlen(STRAT01_TERM_R_OUT_SHA)==64U);
    for(unsigned i=0;i<4U;++i)s[i]=a[i]+b[i];TERM_CHECK(s[0]==3.0f);TERM_CHECK(s[1]==-1.5f);TERM_CHECK(s[2]==1.0f);TERM_CHECK(s[3]==16777216.0f);
    {strat01_sha256 h;uint8_t digest[32];strat01_sha256_init(&h);for(unsigned i=0;i<4U;++i){uint8_t raw[4];strat01_rung1_put_f32le(raw,s[i]);strat01_sha256_update(&h,raw,4);}strat01_sha256_final(&h,digest);strat01_sha256_hex(digest,sha);}TERM_CHECK(strlen(sha)==64U);
#undef TERM_CHECK
    fprintf(stderr,"STRAT-01 terminal-component selftest: %s (%d checks)\n",bad?"FAIL":"PASS",checks);return bad?1:0;
}

#endif
