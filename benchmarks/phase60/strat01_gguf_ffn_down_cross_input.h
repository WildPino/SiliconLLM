/* Frozen ffn_swiglu-0 -> Q6_K down -> layer-1 propagation diagnostic. */
#ifndef STRAT01_GGUF_FFN_DOWN_CROSS_INPUT_H
#define STRAT01_GGUF_FFN_DOWN_CROSS_INPUT_H

#define STRAT01_DOWN_SWIGLU_COUNT (8U*STRAT01_R2B_FFN)
#define STRAT01_DOWN_SWIGLU_BYTES UINT64_C(286720)
#define STRAT01_DOWN_C_SWIGLU_SHA "e4073a39ca0d6f904dab90b22f8fac9c9516c219b265611a102da8dd3597bc8d"
#define STRAT01_DOWN_R_SWIGLU_SHA "de9245a2f6f60e8d7511d03fe981e6b55ef21a8634d5071dd36727a1a5950cef"
#define STRAT01_DOWN_C_OUT_SHA STRAT01_TERM_C_OUT_SHA
#define STRAT01_DOWN_R_OUT_SHA STRAT01_TERM_R_OUT_SHA
#define STRAT01_DOWN_R_INP_SHA STRAT01_TERM_R_INP_SHA
#define STRAT01_DOWN_C_SUM_SHA "b822e425fb6c88faa62cc59dec8011a7869ebb9d64c3731e54beb585f66cb541"

typedef struct { const char *name;const char *source;float *out,*sum;strat01_r2a_arm state;strat01_r2c_l1start_dump dumps[11];char out_path[1024],out_sha[65],sum_sha[65]; } strat01_down_arm;

static int strat01_down_read(const char *path,const char *expected,uint64_t count,float *out,char actual[65],char error[256]) {
    FILE *f=NULL;uint64_t bytes=0;uint8_t raw[4];if(!strat01_sha256_file(path,actual,&bytes,error)||bytes!=count*4U||strcmp(actual,expected)){if(!error[0])snprintf(error,256,"FFN-down input identity mismatch");return 0;}
    f=fopen(path,"rb");if(!f){snprintf(error,256,"cannot open FFN-down input");return 0;}for(uint64_t i=0;i<count;++i){if(fread(raw,1,4,f)!=4){fclose(f);snprintf(error,256,"FFN-down input short read");return 0;}out[i]=strat01_r2a_f32le(raw);if(!isfinite(out[i])){fclose(f);snprintf(error,256,"FFN-down input non-finite");return 0;}}if(ferror(f)||fclose(f)!=0){snprintf(error,256,"FFN-down input I/O failure");return 0;}return 1;
}

static void strat01_down_sha(const float *values,uint64_t count,char hex[65]) { strat01_sha256 sha;uint8_t digest[32];strat01_sha256_init(&sha);for(uint64_t i=0;i<count;++i){uint8_t b[4];strat01_rung1_put_f32le(b,values[i]);strat01_sha256_update(&sha,b,4);}strat01_sha256_final(&sha,digest);strat01_sha256_hex(digest,hex); }
static void strat01_down_sum(const float *inp,const float *out,float *sum) { for(size_t i=0;i<STRAT01_R2C_L1START_INPUT_COUNT;++i)sum[i]=inp[i]+out[i]; }

static int strat01_down_write_failure(const char *out_dir,const char *error) { char path[1024];FILE *f;if(!strat01_r2a_path(path,out_dir,"strat01_ffn_down_cross_input.json")||(f=fopen(path,"wb"))==NULL)return 0;fputs("{\"command\":\"--strat01-ffn-down-cross-input\",\"state\":\"DIAGNOSTIC_FAILURE\",\"error\":",f);strat01_json_string(f,error);fputs(",\"donor_graph_executions\":0,\"reference_graph_executions\":0}\n",f);fclose(f);return 1; }

static int strat01_down_write_control(const char *out_dir,const char *name,const float *out,const strat01_r2a_arm *state,char out_path[1024],char out_sha[65],char kqv_path[1024],char kqv_sha[65],char error[256]) {
    char leaf[192];snprintf(leaf,sizeof(leaf),"control_%s_ffn_out-0.f32le",name);if(!strat01_r2a_path(out_path,out_dir,leaf)||!strat01_q4q8_write_output(out_path,out,8U*1536U,out_sha,error))return 0;snprintf(leaf,sizeof(leaf),"control_%s_kqv_out-1.f32le",name);return strat01_r2a_path(kqv_path,out_dir,leaf)&&strat01_q4q8_write_output(kqv_path,state->kqv_out,8U*6144U,kqv_sha,error);
}

static int strat01_down_cli(const char *model,const char *c_swiglu_path,const char *r_swiglu_path,const char *c_out_path,const char *r_out_path,const char *r_inp_path,const char *out_dir,const char *engine_source_path) {
    strat01_inventory inv;const strat01_tensor *down=NULL,*t[7]={0};float *c_sw=NULL,*r_sw=NULL,*c_target=NULL,*r_target=NULL,*r_inp=NULL,*ctl_sw=NULL,*ctl_out[2]={NULL,NULL};
    strat01_down_arm arms[3]={{"reference_captured_control","captured_reference",NULL,NULL,{0},{{0}},{0},{0},{0}},{"reference_swiglu_current_q6","reference_swiglu_current_q6",NULL,NULL,{0},{{0}},{0},{0},{0}},{"c_swiglu_current_q6","c_swiglu_current_q6",NULL,NULL,{0},{{0}},{0},{0},{0}}};strat01_r2a_arm ctl_state[2];
    const char *paths[5]={c_swiglu_path,r_swiglu_path,c_out_path,r_out_path,r_inp_path};const char *names[5]={"c_swiglu","reference_swiglu","c_ffn_out","reference_ffn_out","reference_ffn_inp"};const char *expected[5]={STRAT01_DOWN_C_SWIGLU_SHA,STRAT01_DOWN_R_SWIGLU_SHA,STRAT01_DOWN_C_OUT_SHA,STRAT01_DOWN_R_OUT_SHA,STRAT01_DOWN_R_INP_SHA};uint64_t counts[5]={STRAT01_DOWN_SWIGLU_COUNT,STRAT01_DOWN_SWIGLU_COUNT,STRAT01_R2C_L1START_INPUT_COUNT,STRAT01_R2C_L1START_INPUT_COUNT,STRAT01_R2C_L1START_INPUT_COUNT};float **dst[5]={&c_sw,&r_sw,&c_target,&r_target,&r_inp};char input_sha[5][65]={{0}};
    FILE *report=NULL;int ok=0;char error[256]={0},model_sha[65]={0},engine_sha[65]={0},header_sha[65]={0},report_path[1024],ctl_out_path[2][1024],ctl_out_sha[2][65]={{0}},ctl_kqv_path[2][1024],ctl_kqv_sha[2][65]={{0}};uint64_t hashed=0,parsed=0,tmp=0;
    memset(&inv,0,sizeof(inv));memset(ctl_state,0,sizeof(ctl_state));
#if !defined(__clang__)
    snprintf(error,256,"FFN-down diagnostic requires Clang");goto finish;
#endif
    if(!strat01_sha256_file(model,model_sha,&hashed,error)||hashed!=STRAT01_EXPECTED_SIZE||strcmp(model_sha,STRAT01_EXPECTED_SHA256)){if(!error[0])snprintf(error,256,"FFN-down artifact mismatch");goto finish;}if(!strat01_parse_gguf(model,&inv,&parsed,error)||parsed!=hashed){if(!error[0])snprintf(error,256,"FFN-down GGUF parse mismatch");goto finish;}
    if(!strat01_r2a_check_spec(&inv,&strat01_r2b_specs[3],&down,error))goto finish;for(unsigned i=0;i<7U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2c_attn_specs[i],&t[i],error))goto finish;
    for(unsigned i=0;i<5U;++i){*dst[i]=strat01_r2a_alloc((size_t)counts[i],error);if(!*dst[i]||!strat01_down_read(paths[i],expected[i],counts[i],*dst[i],input_sha[i],error))goto finish;}
    for(unsigned i=0;i<3U;++i){arms[i].out=strat01_r2a_alloc(STRAT01_R2C_L1START_INPUT_COUNT,error);arms[i].sum=strat01_r2a_alloc(STRAT01_R2C_L1START_INPUT_COUNT,error);if(!arms[i].out||!arms[i].sum||!strat01_r2a_arm_alloc(&arms[i].state,error))goto finish;}ctl_sw=strat01_r2a_alloc(STRAT01_DOWN_SWIGLU_COUNT,error);for(unsigned i=0;i<2U;++i){ctl_out[i]=strat01_r2a_alloc(STRAT01_R2C_L1START_INPUT_COUNT,error);if(!ctl_out[i]||!strat01_r2a_arm_alloc(&ctl_state[i],error))goto finish;}
    memcpy(arms[0].out,r_target,STRAT01_R2C_L1START_INPUT_BYTES);if(!strat01_r2b_q6_matmul_batch(model,down,r_sw,8U,arms[1].out,error)||!strat01_r2b_q6_matmul_batch(model,down,c_sw,8U,arms[2].out,error))goto finish;if(memcmp(arms[2].out,c_target,STRAT01_R2C_L1START_INPUT_BYTES)){snprintf(error,256,"C SwiGLU Q6 output replay mismatch");goto finish;}
    for(unsigned i=0;i<3U;++i){strat01_down_sha(arms[i].out,STRAT01_R2C_L1START_INPUT_COUNT,arms[i].out_sha);strat01_down_sum(r_inp,arms[i].out,arms[i].sum);strat01_down_sha(arms[i].sum,STRAT01_R2C_L1START_INPUT_COUNT,arms[i].sum_sha);}if(strcmp(arms[0].sum_sha,STRAT01_TERM_R_SUM_SHA)||strcmp(arms[2].sum_sha,STRAT01_DOWN_C_SUM_SHA)){snprintf(error,256,"FFN-down terminal replay mismatch");goto finish;}
    for(unsigned i=0;i<3U;++i){if(!strat01_r2c_l1start_run(model,t,arms[i].sum,0,0,&arms[i].state,error))goto finish;strat01_r2c_l1start_make_dumps(&arms[i].state,arms[i].dumps);if(!strat01_r2c_l1start_write_arm(out_dir,arms[i].name,arms[i].dumps,error))goto finish;{char leaf[192];snprintf(leaf,sizeof(leaf),"%s_ffn_out-0.f32le",arms[i].name);if(!strat01_r2a_path(arms[i].out_path,out_dir,leaf)||!strat01_q4q8_write_output(arms[i].out_path,arms[i].out,8U*1536U,arms[i].out_sha,error))goto finish;}}
    memcpy(ctl_sw,r_sw,STRAT01_DOWN_SWIGLU_BYTES);for(size_t i=0;i<STRAT01_DOWN_SWIGLU_COUNT;++i)ctl_sw[i]=-ctl_sw[i];if(!strat01_r2b_q6_matmul_batch(model,down,ctl_sw,8U,ctl_out[0],error))goto finish;strat01_down_sum(r_inp,ctl_out[0],arms[1].sum);if(!strat01_r2c_l1start_run(model,t,arms[1].sum,0,0,&ctl_state[0],error))goto finish;
    memcpy(ctl_sw,r_sw,STRAT01_DOWN_SWIGLU_BYTES);for(unsigned i=0;i<STRAT01_R2B_FFN;++i){float x=ctl_sw[i];ctl_sw[i]=ctl_sw[7U*STRAT01_R2B_FFN+i];ctl_sw[7U*STRAT01_R2B_FFN+i]=x;}if(!strat01_r2b_q6_matmul_batch(model,down,ctl_sw,8U,ctl_out[1],error))goto finish;strat01_down_sum(r_inp,ctl_out[1],arms[1].sum);if(!strat01_r2c_l1start_run(model,t,arms[1].sum,0,0,&ctl_state[1],error))goto finish;
    if(!strat01_down_write_control(out_dir,"reference_swiglu_negated",ctl_out[0],&ctl_state[0],ctl_out_path[0],ctl_out_sha[0],ctl_kqv_path[0],ctl_kqv_sha[0],error)||!strat01_down_write_control(out_dir,"reference_swiglu_rows0_7_swapped",ctl_out[1],&ctl_state[1],ctl_out_path[1],ctl_out_sha[1],ctl_kqv_path[1],ctl_kqv_sha[1],error))goto finish;
    if(!strat01_sha256_file(engine_source_path,engine_sha,&tmp,error)||!strat01_sha256_file(__FILE__,header_sha,&tmp,error))goto finish;if(!strat01_r2a_path(report_path,out_dir,"strat01_ffn_down_cross_input.json")||(report=fopen(report_path,"wb"))==NULL){snprintf(error,256,"cannot write FFN-down report");goto finish;}
    fputs("{\n\"command\":\"--strat01-ffn-down-cross-input\",\n\"state\":\"OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\",\n\"self_certifies_pass\":false,\n\"model\":{\"path\":",report);strat01_json_string(report,model);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",hashed);strat01_json_string(report,model_sha);fputs("},\n\"inputs\":{",report);for(unsigned i=0;i<5U;++i){fprintf(report,"%s\"%s\":{\"path\":",i?",":"",names[i]);strat01_json_string(report,paths[i]);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",counts[i]*4U);strat01_json_string(report,input_sha[i]);fputc('}',report);}fputs("},\n\"down_tensor\":{\"name\":",report);strat01_json_string(report,down->name);fprintf(report,",\"type\":%u,\"offset\":%" PRIu64 ",\"file_offset\":%" PRIu64 ",\"span\":%" PRIu64 "},\n\"layer1_tensors\":[",down->type,down->offset,down->file_offset,down->span);
    for(unsigned i=0;i<7U;++i){fprintf(report,"%s{\"name\":",i?",":"");strat01_json_string(report,t[i]->name);fprintf(report,",\"type\":%u,\"offset\":%" PRIu64 ",\"file_offset\":%" PRIu64 ",\"span\":%" PRIu64 "}",t[i]->type,t[i]->offset,t[i]->file_offset,t[i]->span);}fputs("],\n\"arms\":{",report);
    for(unsigned a=0;a<3U;++a){fprintf(report,"%s\"%s\":{\"source\":",a?",":"",arms[a].name);strat01_json_string(report,arms[a].source);fputs(",\"ffn_out\":{\"path\":",report);strat01_json_string(report,arms[a].out_path);fputs(",\"bytes\":49152,\"sha256\":",report);strat01_json_string(report,arms[a].out_sha);fputs("},\"l_out_sha256\":",report);strat01_json_string(report,arms[a].sum_sha);fputs(",\"checkpoints\":{",report);for(unsigned i=0;i<11U;++i){strat01_r2c_l1start_dump *d=&arms[a].dumps[i];fprintf(report,"%s\"%s\":{\"path\":",i?",":"",d->name);strat01_json_string(report,d->path);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",d->count*4U);strat01_json_string(report,d->sha);fputc('}',report);}fputs("}}",report);}fputs("},\n\"controls\":{",report);
    for(unsigned i=0;i<2U;++i){const char *n=i?"reference_swiglu_rows0_7_swapped":"reference_swiglu_negated";fprintf(report,"%s\"%s\":{\"ffn_out\":{\"path\":",i?",":"",n);strat01_json_string(report,ctl_out_path[i]);fputs(",\"bytes\":49152,\"sha256\":",report);strat01_json_string(report,ctl_out_sha[i]);fputs("},\"kqv_out-1\":{\"path\":",report);strat01_json_string(report,ctl_kqv_path[i]);fputs(",\"bytes\":196608,\"sha256\":",report);strat01_json_string(report,ctl_kqv_sha[i]);fputs("}}",report);}fputs("},\n\"engine_source_sha256\":",report);strat01_json_string(report,engine_sha);fputs(",\n\"diagnostic_source_sha256\":",report);strat01_json_string(report,header_sha);fputs(",\n\"compiler_family\":\"clang\",\n\"donor_graph_executions\":0,\n\"reference_graph_executions\":0,\n\"timing_or_rate_claim\":null\n}\n",report);
    if(fclose(report)!=0){report=NULL;snprintf(error,256,"FFN-down report close failure");goto finish;}report=NULL;ok=1;
finish:
    if(report)fclose(report);for(unsigned i=0;i<5U;++i)free(*dst[i]);free(ctl_sw);for(unsigned i=0;i<3U;++i){free(arms[i].out);free(arms[i].sum);strat01_r2a_arm_free(&arms[i].state);}for(unsigned i=0;i<2U;++i){free(ctl_out[i]);strat01_r2a_arm_free(&ctl_state[i]);}strat01_free_inventory(&inv);
    if(!ok){if(!error[0])snprintf(error,256,"unspecified FFN-down failure");strat01_down_write_failure(out_dir,error);fprintf(stderr,"STRAT-01 FFN-down refused: %s\n",error);return 1;}fprintf(stderr,"STRAT-01 FFN-down: OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\n");return 0;
}

static int strat01_down_selftest(void) { int bad=0,checks=0;float inp[4]={1,-2,3,-4},out[4]={2,3,-1,-2},sum[4]={0};
#define DOWN_CHECK(z) do{++checks;if(!(z))++bad;}while(0)
    DOWN_CHECK(strlen(STRAT01_DOWN_C_SWIGLU_SHA)==64U);DOWN_CHECK(strlen(STRAT01_DOWN_R_SWIGLU_SHA)==64U);for(unsigned i=0;i<4U;++i)sum[i]=inp[i]+out[i];DOWN_CHECK(sum[0]==3);DOWN_CHECK(sum[1]==1);DOWN_CHECK(sum[2]==2);DOWN_CHECK(sum[3]==-6);{strat01_tensor t={0};char e[256]={0};t.type=STRAT01_GGML_F32;t.rank=2;t.dims[0]=8960;t.dims[1]=1536;DOWN_CHECK(!strat01_r2b_q6_matmul_batch("missing",&t,inp,1,out,e));}
#undef DOWN_CHECK
    fprintf(stderr,"STRAT-01 FFN-down selftest: %s (%d checks)\n",bad?"FAIL":"PASS",checks);return bad?1:0; }

#endif
