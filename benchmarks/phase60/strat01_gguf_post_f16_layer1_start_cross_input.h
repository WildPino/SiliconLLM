/* Post-F16 exact l_out-0 -> complete layer-1 cross-input diagnostic. */
#ifndef STRAT01_GGUF_POST_F16_LAYER1_START_CROSS_INPUT_H
#define STRAT01_GGUF_POST_F16_LAYER1_START_CROSS_INPUT_H

#define STRAT01_POSTF16_REF_SHA "385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa"
#define STRAT01_POSTF16_CUR_SHA "7fb5fe52df684e4af82413004f6845133528241e3b030530d41595f74298bdc4"

static int strat01_postf16_run_attention(const char *model,const strat01_tensor *attn[7],const float *input,int negate6,int swap07,strat01_r2a_arm *a,char error[256]) {
    memcpy(a->embd,input,STRAT01_R2C_L1START_INPUT_BYTES);
    if(negate6)for(size_t i=(size_t)6U*1536U;i<(size_t)7U*1536U;++i)a->embd[i]=-a->embd[i];
    if(swap07)for(unsigned i=0;i<1536U;++i){float x=a->embd[i];a->embd[i]=a->embd[7U*1536U+i];a->embd[7U*1536U+i]=x;}
    return strat01_r2c_build_attention_range(model,attn,a,0,8,error)&&strat01_r2a_run_schedule(model,attn[5],attn[6],a,0,error);
}

static int strat01_postf16_run_full(const char *model,const strat01_tensor *attn[7],const strat01_tensor *moe[9],const float *input,strat01_r2a_arm *a,strat01_r2c_moe_arm *m,char error[256]) {
    return strat01_postf16_run_attention(model,attn,input,0,0,a,error)&&strat01_r2c_run_moe(model,moe,a,m,error);
}

static int strat01_postf16_write_arm(const char *out_dir,const char *arm,const float *input,const strat01_r2a_arm *a,const strat01_r2c_moe_arm *m,strat01_r2c_dump d[STRAT01_R2C_DUMPS],char error[256]) {
    strat01_r2b_arm start;memset(&start,0,sizeof(start));start.l_out=(float *)input;
    if(strat01_r2c_make_dumps(&start,a,m,d)!=STRAT01_R2C_DUMPS){snprintf(error,256,"post-F16 dump count mismatch");return 0;}
    for(unsigned i=0;i<STRAT01_R2C_DUMPS;++i){d[i].ordinal=i;if(!strat01_r2c_dump_payload(&d[i],out_dir,arm,error))return 0;}
    return 1;
}

static void strat01_postf16_report_arm(FILE *o,strat01_r2c_dump d[STRAT01_R2C_DUMPS]) {
    fputc('{',o);for(unsigned i=0;i<STRAT01_R2C_DUMPS;++i){fprintf(o,"%s\"%s\":{\"path\":",i?",":"",d[i].logical_name);strat01_json_string(o,d[i].path);fprintf(o,",\"bytes\":%" PRIu64 ",\"sha256\":",d[i].count*4U);strat01_json_string(o,d[i].sha256);fputc('}',o);}fputc('}',o);
}

static int strat01_postf16_write_failure(const char *out_dir,const char *error) {
    char path[1024];FILE *f;if(!strat01_r2a_path(path,out_dir,"strat01_post_f16_layer1_start_cross_input.json")||(f=fopen(path,"wb"))==NULL)return 0;
    fputs("{\"command\":\"--strat01-post-f16-layer1-start-cross-input\",\"state\":\"DIAGNOSTIC_FAILURE\",\"error\":",f);strat01_json_string(f,error);fputs(",\"donor_graph_executions\":0,\"reference_graph_executions\":0}\n",f);fclose(f);return 1;
}

static int strat01_postf16_cli(const char *model,const char *ref_path,const char *cur_path,const char *out_dir,const char *engine_source_path) {
    strat01_inventory inv;const strat01_tensor *attn[7]={0},*moe[9]={0};float *ri=NULL,*ci=NULL;strat01_r2a_arm ra,ca,n6,s7;strat01_r2c_moe_arm rm,cm;strat01_r2c_dump rd[STRAT01_R2C_DUMPS],cd[STRAT01_R2C_DUMPS];
    char error[256]={0},model_sha[65]={0},ref_sha[65]={0},cur_sha[65]={0},engine_sha[65]={0},header_sha[65]={0},n6_path[1024],s7_path[1024],n6_sha[65]={0},s7_sha[65]={0},report_path[1024];uint64_t hashed=0,parsed=0,tmp=0;FILE *report=NULL;int ok=0;
    memset(&inv,0,sizeof(inv));memset(&ra,0,sizeof(ra));memset(&ca,0,sizeof(ca));memset(&n6,0,sizeof(n6));memset(&s7,0,sizeof(s7));memset(&rm,0,sizeof(rm));memset(&cm,0,sizeof(cm));strat01_f16vec_reset_counts();
#if !defined(__clang__)
    snprintf(error,256,"post-F16 diagnostic requires Clang");goto finish;
#endif
    if(!strat01_sha256_file(model,model_sha,&hashed,error)||hashed!=STRAT01_EXPECTED_SIZE||strcmp(model_sha,STRAT01_EXPECTED_SHA256)){if(!error[0])snprintf(error,256,"post-F16 artifact mismatch");goto finish;}
    if(!strat01_parse_gguf(model,&inv,&parsed,error)||parsed!=hashed){if(!error[0])snprintf(error,256,"post-F16 GGUF parse mismatch");goto finish;}
    for(unsigned i=0;i<7U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2c_attn_specs[i],&attn[i],error))goto finish;
    for(unsigned i=0;i<9U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2c_moe_specs[i],&moe[i],error))goto finish;
    ri=strat01_r2a_alloc(STRAT01_R2C_L1START_INPUT_COUNT,error);ci=strat01_r2a_alloc(STRAT01_R2C_L1START_INPUT_COUNT,error);if(!ri||!ci)goto finish;
    if(!strat01_r2c_l1start_read(ref_path,STRAT01_POSTF16_REF_SHA,ri,ref_sha,error)||!strat01_r2c_l1start_read(cur_path,STRAT01_POSTF16_CUR_SHA,ci,cur_sha,error))goto finish;
    if(!strat01_r2a_arm_alloc(&ra,error)||!strat01_r2a_arm_alloc(&ca,error)||!strat01_r2a_arm_alloc(&n6,error)||!strat01_r2a_arm_alloc(&s7,error)||!strat01_r2c_moe_alloc(&rm,error)||!strat01_r2c_moe_alloc(&cm,error))goto finish;
    if(!strat01_postf16_run_full(model,attn,moe,ri,&ra,&rm,error)||!strat01_postf16_run_full(model,attn,moe,ci,&ca,&cm,error)||!strat01_postf16_run_attention(model,attn,ri,1,0,&n6,error)||!strat01_postf16_run_attention(model,attn,ri,0,1,&s7,error))goto finish;
    if(!strat01_postf16_write_arm(out_dir,"reference_start",ri,&ra,&rm,rd,error)||!strat01_postf16_write_arm(out_dir,"current_start",ci,&ca,&cm,cd,error))goto finish;
    if(!strat01_r2a_path(n6_path,out_dir,"control_reference_token6_negated_kqv_out-1.f32le")||!strat01_r2a_path(s7_path,out_dir,"control_reference_rows0_7_swapped_kqv_out-1.f32le")||!strat01_q4q8_write_output(n6_path,n6.kqv_out,8U*6144U,n6_sha,error)||!strat01_q4q8_write_output(s7_path,s7.kqv_out,8U*6144U,s7_sha,error)||!strat01_f16v_write_counts(out_dir,error))goto finish;
    if(!strat01_sha256_file(engine_source_path,engine_sha,&tmp,error)||!strat01_sha256_file(__FILE__,header_sha,&tmp,error))goto finish;
    if(!strat01_r2a_path(report_path,out_dir,"strat01_post_f16_layer1_start_cross_input.json")||(report=fopen(report_path,"wb"))==NULL){snprintf(error,256,"cannot write post-F16 report");goto finish;}
    fputs("{\n\"command\":\"--strat01-post-f16-layer1-start-cross-input\",\n\"state\":\"OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\",\n\"self_certifies_pass\":false,\n\"model\":{\"path\":",report);strat01_json_string(report,model);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",hashed);strat01_json_string(report,model_sha);
    fputs("},\n\"inputs\":{\"reference\":{\"path\":",report);strat01_json_string(report,ref_path);fputs(",\"bytes\":49152,\"sha256\":",report);strat01_json_string(report,ref_sha);fputs("},\"current\":{\"path\":",report);strat01_json_string(report,cur_path);fputs(",\"bytes\":49152,\"sha256\":",report);strat01_json_string(report,cur_sha);fputs("}},\n\"tensors\":[",report);
    for(unsigned i=0;i<16U;++i){const strat01_tensor *t=i<7U?attn[i]:moe[i-7U];fprintf(report,"%s{\"name\":",i?",":"");strat01_json_string(report,t->name);fprintf(report,",\"type\":%u,\"offset\":%" PRIu64 ",\"file_offset\":%" PRIu64 ",\"span\":%" PRIu64 "}",t->type,t->offset,t->file_offset,t->span);}fputs("],\n\"outputs\":{\"reference_start\":",report);strat01_postf16_report_arm(report,rd);fputs(",\"current_start\":",report);strat01_postf16_report_arm(report,cd);
    fputs("},\n\"controls\":{\"reference_token6_negated\":{\"path\":",report);strat01_json_string(report,n6_path);fputs(",\"bytes\":196608,\"sha256\":",report);strat01_json_string(report,n6_sha);fputs("},\"reference_rows0_7_swapped\":{\"path\":",report);strat01_json_string(report,s7_path);fputs(",\"bytes\":196608,\"sha256\":",report);strat01_json_string(report,s7_sha);fputs("}},\n\"engine_source_sha256\":",report);strat01_json_string(report,engine_sha);fputs(",\n\"diagnostic_source_sha256\":",report);strat01_json_string(report,header_sha);fputs(",\n\"compiler_family\":\"clang\",\n\"donor_graph_executions\":0,\n\"reference_graph_executions\":0,\n\"timing_or_rate_claim\":null\n}\n",report);
    if(fclose(report)!=0){report=NULL;snprintf(error,256,"post-F16 report close failure");goto finish;}report=NULL;ok=1;
finish:
    if(report)fclose(report);free(ri);free(ci);strat01_r2a_arm_free(&ra);strat01_r2a_arm_free(&ca);strat01_r2a_arm_free(&n6);strat01_r2a_arm_free(&s7);strat01_r2c_moe_free(&rm);strat01_r2c_moe_free(&cm);strat01_free_inventory(&inv);
    if(!ok){if(!error[0])snprintf(error,256,"unspecified post-F16 failure");strat01_postf16_write_failure(out_dir,error);fprintf(stderr,"STRAT-01 post-F16 layer1-start refused: %s\n",error);return 1;}fprintf(stderr,"STRAT-01 post-F16 layer1-start: OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\n");return 0;
}

static int strat01_postf16_selftest(void) {
    int bad=0,checks=0;float x[8U*1536U]={0};
#define P16_CHECK(z) do{++checks;if(!(z))++bad;}while(0)
    P16_CHECK(strlen(STRAT01_POSTF16_REF_SHA)==64U);P16_CHECK(strlen(STRAT01_POSTF16_CUR_SHA)==64U);P16_CHECK(strcmp(STRAT01_POSTF16_REF_SHA,STRAT01_POSTF16_CUR_SHA));
    x[6U*1536U]=2.0f;for(size_t i=6U*1536U;i<7U*1536U;++i)x[i]=-x[i];P16_CHECK(x[6U*1536U]==-2.0f);P16_CHECK(STRAT01_R2C_DUMPS==32U);
#undef P16_CHECK
    fprintf(stderr,"STRAT-01 post-F16 layer1-start selftest: %s (%d checks)\n",bad?"FAIL":"PASS",checks);return bad?1:0;
}

#endif
