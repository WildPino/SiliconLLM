/* Post-F16 block-0 SwiGLU-expression versus Q6 -> complete layer-1 diagnostic. */
#ifndef STRAT01_GGUF_POST_F16_BLOCK0_FFN_OPERATOR_CROSS_INPUT_H
#define STRAT01_GGUF_POST_F16_BLOCK0_FFN_OPERATOR_CROSS_INPUT_H

#define STRAT01_P16FFN_REF_GATE_SHA "5c30c0ada2e96ce43a92b22057b4d593b0b9ca342feade7092bfec084e8c6c2a"
#define STRAT01_P16FFN_REF_UP_SHA "2b608af95db90fcde83c29946ba2ac680fb6465cbaaf518fc12874ab510b29e4"
#define STRAT01_P16FFN_REF_SW_SHA "de9245a2f6f60e8d7511d03fe981e6b55ef21a8634d5071dd36727a1a5950cef"
#define STRAT01_P16FFN_EXPR_SW_SHA "ac2b46108a7e18ddebdb1b3a26462bca0faebea536361a5ca4069c4fa7ddcf40"
#define STRAT01_P16FFN_REF_INP_SHA "baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1"
#define STRAT01_P16FFN_Q6_ONLY_OUT_SHA "f52e0e5bdb22a17ae4a9e0ee9e0c6ffc6bb485b3e61e3f821039eae38697d6ce"
#define STRAT01_P16FFN_Q6_ONLY_SUM_SHA "a71c814dc43360f73b49cb165e7cdf7128f2501015685ff16e7e524dd1f05f11"
#define STRAT01_P16FFN_REPLAY_OUT_SHA "7d0037ab06440493b146e9839d5c89ffa8fdea2b952389f12a38ccea2a94e684"
#define STRAT01_P16FFN_REPLAY_SUM_SHA "7fb5fe52df684e4af82413004f6845133528241e3b030530d41595f74298bdc4"

typedef struct {
    const char *name;float *swiglu,*out,*sum;strat01_r2a_arm attention;strat01_r2c_moe_arm moe;strat01_r2c_dump dumps[STRAT01_R2C_DUMPS];
    char sw_path[1024],sw_sha[65],out_path[1024],out_sha[65],sum_path[1024],sum_sha[65];
} strat01_p16ffn_arm;

static int strat01_p16ffn_write_failure(const char *out_dir,const char *error) {
    char path[1024];FILE *f;if(!strat01_r2a_path(path,out_dir,"strat01_post_f16_block0_ffn_operator_cross_input.json")||(f=fopen(path,"wb"))==NULL)return 0;
    fputs("{\"command\":\"--strat01-post-f16-block0-ffn-operator-cross-input\",\"state\":\"DIAGNOSTIC_FAILURE\",\"error\":",f);strat01_json_string(f,error);fputs(",\"donor_graph_executions\":0,\"reference_graph_executions\":0}\n",f);fclose(f);return 1;
}

static int strat01_p16ffn_run_arm(const char *model,const strat01_tensor *down,const strat01_tensor *attn[7],const strat01_tensor *moe[9],const float *r_inp,const char *out_dir,strat01_p16ffn_arm *arm,char error[256]) {
    char leaf[192];
    if(!strat01_r2b_q6_matmul_batch(model,down,arm->swiglu,8U,arm->out,error))return 0;
    strat01_down_sha(arm->swiglu,STRAT01_DOWN_SWIGLU_COUNT,arm->sw_sha);strat01_down_sha(arm->out,STRAT01_R2C_L1START_INPUT_COUNT,arm->out_sha);strat01_down_sum(r_inp,arm->out,arm->sum);strat01_down_sha(arm->sum,STRAT01_R2C_L1START_INPUT_COUNT,arm->sum_sha);
    if(!strat01_postf16_run_full(model,attn,moe,arm->sum,&arm->attention,&arm->moe,error)||!strat01_postf16_write_arm(out_dir,arm->name,arm->sum,&arm->attention,&arm->moe,arm->dumps,error))return 0;
    snprintf(leaf,sizeof(leaf),"%s_ffn_swiglu-0.f32le",arm->name);if(!strat01_r2a_path(arm->sw_path,out_dir,leaf)||!strat01_q4q8_write_output(arm->sw_path,arm->swiglu,STRAT01_DOWN_SWIGLU_COUNT,arm->sw_sha,error))return 0;
    snprintf(leaf,sizeof(leaf),"%s_ffn_out-0.f32le",arm->name);if(!strat01_r2a_path(arm->out_path,out_dir,leaf)||!strat01_q4q8_write_output(arm->out_path,arm->out,STRAT01_R2C_L1START_INPUT_COUNT,arm->out_sha,error))return 0;
    snprintf(leaf,sizeof(leaf),"%s_l_out-0.f32le",arm->name);return strat01_r2a_path(arm->sum_path,out_dir,leaf)&&strat01_q4q8_write_output(arm->sum_path,arm->sum,STRAT01_R2C_L1START_INPUT_COUNT,arm->sum_sha,error);
}

static void strat01_p16ffn_report_payload(FILE *o,const char *path,uint64_t bytes,const char *sha) { fputs("{\"path\":",o);strat01_json_string(o,path);fprintf(o,",\"bytes\":%" PRIu64 ",\"sha256\":",bytes);strat01_json_string(o,sha);fputc('}',o); }

static void strat01_p16ffn_report_arm(FILE *o,const strat01_p16ffn_arm *arm) {
    fputs("{\"swiglu\":",o);strat01_p16ffn_report_payload(o,arm->sw_path,STRAT01_DOWN_SWIGLU_BYTES,arm->sw_sha);fputs(",\"ffn_out\":",o);strat01_p16ffn_report_payload(o,arm->out_path,STRAT01_R2C_L1START_INPUT_BYTES,arm->out_sha);fputs(",\"l_out\":",o);strat01_p16ffn_report_payload(o,arm->sum_path,STRAT01_R2C_L1START_INPUT_BYTES,arm->sum_sha);fputs(",\"checkpoints\":",o);strat01_postf16_report_arm(o,(strat01_r2c_dump *)arm->dumps);fputc('}',o);
}

static int strat01_p16ffn_cli(const char *model,const char *r_gate_path,const char *r_up_path,const char *r_sw_path,const char *r_inp_path,const char *out_dir,const char *engine_source_path) {
    strat01_inventory inv;const strat01_tensor *down=NULL,*attn[7]={0},*moe[9]={0};float *r_gate=NULL,*r_up=NULL,*r_sw=NULL,*r_inp=NULL;strat01_p16ffn_arm arms[2]={{"q6_only",NULL,NULL,NULL,{0},{0},{{0}},{0},{0},{0},{0},{0},{0}}, {"expression_q6_replay",NULL,NULL,NULL,{0},{0},{{0}},{0},{0},{0},{0},{0},{0}}};strat01_p16ffn_arm controls[2]={{"control_reference_swiglu_token6_negated",NULL,NULL,NULL,{0},{0},{{0}},{0},{0},{0},{0},{0},{0}}, {"control_reference_swiglu_rows0_7_swapped",NULL,NULL,NULL,{0},{0},{{0}},{0},{0},{0},{0},{0},{0}}};
    const char *paths[4]={r_gate_path,r_up_path,r_sw_path,r_inp_path};const char *names[4]={"reference_gate","reference_up","reference_swiglu","reference_ffn_inp"};const char *expected[4]={STRAT01_P16FFN_REF_GATE_SHA,STRAT01_P16FFN_REF_UP_SHA,STRAT01_P16FFN_REF_SW_SHA,STRAT01_P16FFN_REF_INP_SHA};uint64_t counts[4]={STRAT01_DOWN_SWIGLU_COUNT,STRAT01_DOWN_SWIGLU_COUNT,STRAT01_DOWN_SWIGLU_COUNT,STRAT01_R2C_L1START_INPUT_COUNT};float **dst[4]={&r_gate,&r_up,&r_sw,&r_inp};char input_sha[4][65]={{0}};
    char error[256]={0},model_sha[65]={0},engine_sha[65]={0},header_sha[65]={0},report_path[1024];uint64_t hashed=0,parsed=0,tmp=0;FILE *report=NULL;int ok=0;
    memset(&inv,0,sizeof(inv));strat01_f16vec_reset_counts();
#if !defined(__clang__)
    snprintf(error,256,"post-F16 FFN operator diagnostic requires Clang");goto finish;
#endif
    if(!strat01_sha256_file(model,model_sha,&hashed,error)||hashed!=STRAT01_EXPECTED_SIZE||strcmp(model_sha,STRAT01_EXPECTED_SHA256)){if(!error[0])snprintf(error,256,"post-F16 FFN operator artifact mismatch");goto finish;}
    if(!strat01_parse_gguf(model,&inv,&parsed,error)||parsed!=hashed){if(!error[0])snprintf(error,256,"post-F16 FFN operator GGUF parse mismatch");goto finish;}
    if(!strat01_r2a_check_spec(&inv,&strat01_r2b_specs[3],&down,error))goto finish;for(unsigned i=0;i<7U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2c_attn_specs[i],&attn[i],error))goto finish;for(unsigned i=0;i<9U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2c_moe_specs[i],&moe[i],error))goto finish;
    for(unsigned i=0;i<4U;++i){*dst[i]=strat01_r2a_alloc((size_t)counts[i],error);if(!*dst[i]||!strat01_down_read(paths[i],expected[i],counts[i],*dst[i],input_sha[i],error))goto finish;}
    for(unsigned group=0;group<2U;++group)for(unsigned i=0;i<2U;++i){strat01_p16ffn_arm *a=group?&controls[i]:&arms[i];a->swiglu=strat01_r2a_alloc(STRAT01_DOWN_SWIGLU_COUNT,error);a->out=strat01_r2a_alloc(STRAT01_R2C_L1START_INPUT_COUNT,error);a->sum=strat01_r2a_alloc(STRAT01_R2C_L1START_INPUT_COUNT,error);if(!a->swiglu||!a->out||!a->sum||!strat01_r2a_arm_alloc(&a->attention,error)||!strat01_r2c_moe_alloc(&a->moe,error))goto finish;}
    memcpy(arms[0].swiglu,r_sw,STRAT01_DOWN_SWIGLU_BYTES);if(!strat01_sw_compute(r_gate,r_up,arms[1].swiglu,STRAT01_DOWN_SWIGLU_COUNT,error))goto finish;
    memcpy(controls[0].swiglu,r_sw,STRAT01_DOWN_SWIGLU_BYTES);for(size_t i=6U*STRAT01_R2B_FFN;i<7U*STRAT01_R2B_FFN;++i)controls[0].swiglu[i]=-controls[0].swiglu[i];
    memcpy(controls[1].swiglu,r_sw,STRAT01_DOWN_SWIGLU_BYTES);for(unsigned i=0;i<STRAT01_R2B_FFN;++i){float x=controls[1].swiglu[i];controls[1].swiglu[i]=controls[1].swiglu[7U*STRAT01_R2B_FFN+i];controls[1].swiglu[7U*STRAT01_R2B_FFN+i]=x;}
    for(unsigned i=0;i<2U;++i)if(!strat01_p16ffn_run_arm(model,down,attn,moe,r_inp,out_dir,&arms[i],error))goto finish;for(unsigned i=0;i<2U;++i)if(!strat01_p16ffn_run_arm(model,down,attn,moe,r_inp,out_dir,&controls[i],error))goto finish;
    if(strcmp(arms[0].sw_sha,STRAT01_P16FFN_REF_SW_SHA)||strcmp(arms[0].out_sha,STRAT01_P16FFN_Q6_ONLY_OUT_SHA)||strcmp(arms[0].sum_sha,STRAT01_P16FFN_Q6_ONLY_SUM_SHA)){snprintf(error,256,"post-F16 Q6-only replay mismatch");goto finish;}
    if(strcmp(arms[1].sw_sha,STRAT01_P16FFN_EXPR_SW_SHA)||strcmp(arms[1].out_sha,STRAT01_P16FFN_REPLAY_OUT_SHA)||strcmp(arms[1].sum_sha,STRAT01_P16FFN_REPLAY_SUM_SHA)){snprintf(error,256,"post-F16 expression+Q6 replay mismatch");goto finish;}
    if(!strat01_f16v_write_counts(out_dir,error)||!strat01_sha256_file(engine_source_path,engine_sha,&tmp,error)||!strat01_sha256_file(__FILE__,header_sha,&tmp,error))goto finish;
    if(!strat01_r2a_path(report_path,out_dir,"strat01_post_f16_block0_ffn_operator_cross_input.json")||(report=fopen(report_path,"wb"))==NULL){snprintf(error,256,"cannot write post-F16 FFN operator report");goto finish;}
    fputs("{\n\"command\":\"--strat01-post-f16-block0-ffn-operator-cross-input\",\n\"state\":\"OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\",\n\"self_certifies_pass\":false,\n\"model\":{\"path\":",report);strat01_json_string(report,model);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",hashed);strat01_json_string(report,model_sha);fputs("},\n\"inputs\":{",report);
    for(unsigned i=0;i<4U;++i){fprintf(report,"%s\"%s\":{\"path\":",i?",":"",names[i]);strat01_json_string(report,paths[i]);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",counts[i]*4U);strat01_json_string(report,input_sha[i]);fputc('}',report);}fputs("},\n\"down_tensor\":{\"name\":",report);strat01_json_string(report,down->name);fprintf(report,",\"type\":%u,\"offset\":%" PRIu64 ",\"file_offset\":%" PRIu64 ",\"span\":%" PRIu64 "},\n\"layer1_tensors\":[",down->type,down->offset,down->file_offset,down->span);
    for(unsigned i=0;i<16U;++i){const strat01_tensor *t=i<7U?attn[i]:moe[i-7U];fprintf(report,"%s{\"name\":",i?",":"");strat01_json_string(report,t->name);fprintf(report,",\"type\":%u,\"offset\":%" PRIu64 ",\"file_offset\":%" PRIu64 ",\"span\":%" PRIu64 "}",t->type,t->offset,t->file_offset,t->span);}fputs("],\n\"arms\":{\"q6_only\":",report);strat01_p16ffn_report_arm(report,&arms[0]);fputs(",\"expression_q6_replay\":",report);strat01_p16ffn_report_arm(report,&arms[1]);fputs("},\n\"controls\":{\"reference_swiglu_token6_negated\":",report);strat01_p16ffn_report_arm(report,&controls[0]);fputs(",\"reference_swiglu_rows0_7_swapped\":",report);strat01_p16ffn_report_arm(report,&controls[1]);
    fputs("},\n\"q6_down_arms\":4,\n\"engine_source_sha256\":",report);strat01_json_string(report,engine_sha);fputs(",\n\"diagnostic_source_sha256\":",report);strat01_json_string(report,header_sha);fputs(",\n\"compiler_family\":\"clang\",\n\"donor_graph_executions\":0,\n\"reference_graph_executions\":0,\n\"timing_or_rate_claim\":null\n}\n",report);
    if(fclose(report)!=0){report=NULL;snprintf(error,256,"post-F16 FFN operator report close failure");goto finish;}report=NULL;ok=1;
finish:
    if(report)fclose(report);for(unsigned i=0;i<4U;++i)free(*dst[i]);for(unsigned group=0;group<2U;++group)for(unsigned i=0;i<2U;++i){strat01_p16ffn_arm *a=group?&controls[i]:&arms[i];free(a->swiglu);free(a->out);free(a->sum);strat01_r2a_arm_free(&a->attention);strat01_r2c_moe_free(&a->moe);}strat01_free_inventory(&inv);
    if(!ok){if(!error[0])snprintf(error,256,"unspecified post-F16 FFN operator failure");strat01_p16ffn_write_failure(out_dir,error);fprintf(stderr,"STRAT-01 post-F16 FFN operator refused: %s\n",error);return 1;}fprintf(stderr,"STRAT-01 post-F16 FFN operator: OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\n");return 0;
}

static int strat01_p16ffn_selftest(void) {
    int bad=0,checks=0;float g[4]={0,1,-1,2},u[4]={2,2,2,2},s[4]={0};char error[256]={0},sha[65]={0};
#define P16FFN_CHECK(z) do{++checks;if(!(z))++bad;}while(0)
    P16FFN_CHECK(strlen(STRAT01_P16FFN_REF_SW_SHA)==64U);P16FFN_CHECK(strlen(STRAT01_P16FFN_REPLAY_OUT_SHA)==64U);P16FFN_CHECK(strat01_sw_compute(g,u,s,4,error));P16FFN_CHECK(s[0]==0.0f&&s[1]>0.0f&&s[2]<0.0f);strat01_down_sha(s,4,sha);P16FFN_CHECK(strlen(sha)==64U);P16FFN_CHECK(strcmp(STRAT01_P16FFN_Q6_ONLY_OUT_SHA,STRAT01_P16FFN_REPLAY_OUT_SHA));
#undef P16FFN_CHECK
    fprintf(stderr,"STRAT-01 post-F16 FFN operator selftest: %s (%d checks)\n",bad?"FAIL":"PASS",checks);return bad?1:0;
}

#endif
