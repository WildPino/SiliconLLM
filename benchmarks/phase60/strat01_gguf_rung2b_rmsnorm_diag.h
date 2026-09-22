/* Diagnostic-only float-vs-double RMSNorm accumulator comparison. */
#ifndef STRAT01_GGUF_RUNG2B_RMSNORM_DIAG_H
#define STRAT01_GGUF_RUNG2B_RMSNORM_DIAG_H

#pragma STDC FP_CONTRACT OFF

#define STRAT01_R2B_RMS_REF_INP_SHA "baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1"
#define STRAT01_R2B_RMS_C_INP_SHA "bd00c9c5b01726980ab0865e27860f73b7c4186940362919d16dee7e75ee8d82"

static void strat01_r2b_rmsnorm_double_sum(const float *x,const float *weight,float *y,unsigned rows,unsigned n,float eps) {
    strat01_r2a_rmsnorm_pinned(x,weight,y,rows,n,eps);
}

static int strat01_r2b_rms_descriptor_ok(const strat01_tensor *t) {
    return t&&!strcmp(t->name,"blk.0.ffn_norm.weight")&&t->type==STRAT01_GGML_F32&&t->rank==1&&t->dims[0]==1536U&&
           t->offset==UINT64_C(305786880)&&t->span==UINT64_C(6144)&&t->file_offset==UINT64_C(311889792);
}

static int strat01_r2b_rms_write_failure(const char *out_dir,const char *error) {
    char path[1024];FILE *f;if(!strat01_r2a_path(path,out_dir,"strat01_rung2b_rmsnorm_diag.json")||(f=fopen(path,"wb"))==NULL)return 0;
    fputs("{\"command\":\"--strat01-rung2b-rmsnorm-diagnostic\",\"state\":\"DIAGNOSTIC_FAILURE\",\"error\":",f);strat01_json_string(f,error);fputs(",\"donor_graph_executions\":0}\n",f);fclose(f);return 1;
}

static int strat01_r2b_rmsnorm_diag_cli(const char *model,const char *ref_input_path,const char *c_input_path,const char *out_dir,const char *engine_source_path) {
    strat01_inventory inv;const strat01_tensor *norm=NULL,*gate=NULL,*up=NULL;float *weight=NULL,*ref_input=NULL,*c_input=NULL;
    float *ref_float=NULL,*ref_double=NULL,*c_float=NULL,*c_double=NULL,*c_float_up=NULL,*c_float_gate=NULL,*c_double_up=NULL,*c_double_gate=NULL;
    strat01_q8_k_block *float_q8=NULL,*double_q8=NULL;char error[256]={0},model_sha[65]={0},ref_input_sha[65]={0},c_input_sha[65]={0},engine_sha[65]={0},header_sha[65]={0};
    char paths[8][1024],shas[8][65],float_q8_path[1024],double_q8_path[1024],float_q8_sha[65]={0},double_q8_sha[65]={0},report_path[1024];
    uint64_t hashed=0,parsed=0,tmp=0;size_t changed_blocks=0,changed_bytes=0;FILE *report=NULL;int ok=0;
    const char *leaves[8]={"reference_float_norm.f32le","reference_double_norm.f32le","c_float_norm.f32le","c_double_norm.f32le","c_float_up.f32le","c_float_gate.f32le","c_double_up.f32le","c_double_gate.f32le"};
    float *values[8];size_t counts[8]={STRAT01_R2B_CROSS_INPUT_COUNT,STRAT01_R2B_CROSS_INPUT_COUNT,STRAT01_R2B_CROSS_INPUT_COUNT,STRAT01_R2B_CROSS_INPUT_COUNT,STRAT01_R2B_CROSS_OUTPUT_COUNT,STRAT01_R2B_CROSS_OUTPUT_COUNT,STRAT01_R2B_CROSS_OUTPUT_COUNT,STRAT01_R2B_CROSS_OUTPUT_COUNT};
    memset(&inv,0,sizeof(inv));memset(shas,0,sizeof(shas));
#if !defined(__clang__)
    snprintf(error,256,"RMSNorm diagnostic requires Clang");goto finish;
#endif
    if(sizeof(float)!=4||sizeof(double)!=8||sizeof(strat01_q8_k_block)!=292U){snprintf(error,256,"unsupported RMSNorm diagnostic host layout");goto finish;}
    if(!strat01_sha256_file(model,model_sha,&hashed,error)||hashed!=STRAT01_EXPECTED_SIZE||strcmp(model_sha,STRAT01_EXPECTED_SHA256)){if(!error[0])snprintf(error,256,"RMSNorm diagnostic artifact identity mismatch");goto finish;}
    if(!strat01_parse_gguf(model,&inv,&parsed,error)||parsed!=hashed){if(!error[0])snprintf(error,256,"RMSNorm diagnostic GGUF parse mismatch");goto finish;}
    norm=strat01_rung1_find_tensor(&inv,"blk.0.ffn_norm.weight");gate=strat01_rung1_find_tensor(&inv,"blk.0.ffn_gate.weight");up=strat01_rung1_find_tensor(&inv,"blk.0.ffn_up.weight");
    if(!strat01_r2b_rms_descriptor_ok(norm)||!strat01_r2b_cross_descriptor_ok(gate,"blk.0.ffn_gate.weight",UINT64_C(298045440),UINT64_C(304148352))||!strat01_r2b_cross_descriptor_ok(up,"blk.0.ffn_up.weight",UINT64_C(305793024),UINT64_C(311895936))){snprintf(error,256,"RMSNorm diagnostic descriptor mismatch");goto finish;}
    weight=strat01_r2a_alloc(1536U,error);ref_input=strat01_r2a_alloc(STRAT01_R2B_CROSS_INPUT_COUNT,error);c_input=strat01_r2a_alloc(STRAT01_R2B_CROSS_INPUT_COUNT,error);
    ref_float=strat01_r2a_alloc(STRAT01_R2B_CROSS_INPUT_COUNT,error);ref_double=strat01_r2a_alloc(STRAT01_R2B_CROSS_INPUT_COUNT,error);c_float=strat01_r2a_alloc(STRAT01_R2B_CROSS_INPUT_COUNT,error);c_double=strat01_r2a_alloc(STRAT01_R2B_CROSS_INPUT_COUNT,error);
    c_float_up=strat01_r2a_alloc(STRAT01_R2B_CROSS_OUTPUT_COUNT,error);c_float_gate=strat01_r2a_alloc(STRAT01_R2B_CROSS_OUTPUT_COUNT,error);c_double_up=strat01_r2a_alloc(STRAT01_R2B_CROSS_OUTPUT_COUNT,error);c_double_gate=strat01_r2a_alloc(STRAT01_R2B_CROSS_OUTPUT_COUNT,error);
    float_q8=(strat01_q8_k_block *)malloc(STRAT01_R2B_CROSS_Q8_BLOCKS*sizeof(*float_q8));double_q8=(strat01_q8_k_block *)malloc(STRAT01_R2B_CROSS_Q8_BLOCKS*sizeof(*double_q8));
    if(!weight||!ref_input||!c_input||!ref_float||!ref_double||!c_float||!c_double||!c_float_up||!c_float_gate||!c_double_up||!c_double_gate||!float_q8||!double_q8){if(!error[0])snprintf(error,256,"RMSNorm diagnostic allocation failure");goto finish;}
    if(!strat01_r2a_read_f32_vector(model,norm,weight,1536U,error)||!strat01_r2b_cross_read_f32(ref_input_path,STRAT01_R2B_RMS_REF_INP_SHA,ref_input,ref_input_sha,error)||!strat01_r2b_cross_read_f32(c_input_path,STRAT01_R2B_RMS_C_INP_SHA,c_input,c_input_sha,error))goto finish;
    strat01_r2a_rmsnorm(ref_input,weight,ref_float,8U,1536U,STRAT01_R2A_RMS_EPS);strat01_r2b_rmsnorm_double_sum(ref_input,weight,ref_double,8U,1536U,STRAT01_R2A_RMS_EPS);
    strat01_r2a_rmsnorm(c_input,weight,c_float,8U,1536U,STRAT01_R2A_RMS_EPS);strat01_r2b_rmsnorm_double_sum(c_input,weight,c_double,8U,1536U,STRAT01_R2A_RMS_EPS);
    if(!strat01_r2a_matmul_batch(model,up,c_float,8U,1536U,c_float_up,STRAT01_R2B_FFN,error)||!strat01_r2a_matmul_batch(model,gate,c_float,8U,1536U,c_float_gate,STRAT01_R2B_FFN,error)||
       !strat01_r2a_matmul_batch(model,up,c_double,8U,1536U,c_double_up,STRAT01_R2B_FFN,error)||!strat01_r2a_matmul_batch(model,gate,c_double,8U,1536U,c_double_gate,STRAT01_R2B_FFN,error))goto finish;
    if(!strat01_r2b_cross_quantize(c_float,float_q8)||!strat01_r2b_cross_quantize(c_double,double_q8)){snprintf(error,256,"RMSNorm diagnostic Q8 quantization failure");goto finish;}
    strat01_r2b_cross_census(float_q8,double_q8,STRAT01_R2B_CROSS_Q8_BLOCKS,&changed_blocks,&changed_bytes);
    values[0]=ref_float;values[1]=ref_double;values[2]=c_float;values[3]=c_double;values[4]=c_float_up;values[5]=c_float_gate;values[6]=c_double_up;values[7]=c_double_gate;
    for(unsigned i=0;i<8U;++i)if(!strat01_r2a_path(paths[i],out_dir,leaves[i])||!strat01_q4q8_write_output(paths[i],values[i],counts[i],shas[i],error))goto finish;
    if(!strat01_r2a_path(float_q8_path,out_dir,"c_float_norm.q8k")||!strat01_r2a_path(double_q8_path,out_dir,"c_double_norm.q8k")||!strat01_r2a_path(report_path,out_dir,"strat01_rung2b_rmsnorm_diag.json")||
       !strat01_r2b_cross_write_q8(float_q8_path,float_q8,STRAT01_R2B_CROSS_Q8_BLOCKS,float_q8_sha,error)||!strat01_r2b_cross_write_q8(double_q8_path,double_q8,STRAT01_R2B_CROSS_Q8_BLOCKS,double_q8_sha,error))goto finish;
    if(!strat01_sha256_file(engine_source_path,engine_sha,&tmp,error)||!strat01_sha256_file(__FILE__,header_sha,&tmp,error))goto finish;
    report=fopen(report_path,"wb");if(!report){snprintf(error,256,"cannot create RMSNorm diagnostic report");goto finish;}
    fputs("{\n\"command\":\"--strat01-rung2b-rmsnorm-diagnostic\",\n\"state\":\"OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\",\n\"self_certifies_pass\":false,\n\"model\":{\"path\":",report);strat01_json_string(report,model);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",hashed);strat01_json_string(report,model_sha);
    fputs("},\n\"inputs\":{\"reference\":{\"path\":",report);strat01_json_string(report,ref_input_path);fputs(",\"bytes\":49152,\"sha256\":",report);strat01_json_string(report,ref_input_sha);fputs("},\"c_engine\":{\"path\":",report);strat01_json_string(report,c_input_path);fputs(",\"bytes\":49152,\"sha256\":",report);strat01_json_string(report,c_input_sha);fputs("}},\n\"outputs\":{",report);
    for(unsigned i=0;i<8U;++i){if(i)fputc(',',report);strat01_json_string(report,leaves[i]);fputs(":{\"path\":",report);strat01_json_string(report,paths[i]);fprintf(report,",\"bytes\":%zu,\"sha256\":",counts[i]*4U);strat01_json_string(report,shas[i]);fputc('}',report);}fputs("},\n\"q8\":{\"float\":{\"path\":",report);strat01_json_string(report,float_q8_path);fputs(",\"bytes\":14016,\"sha256\":",report);strat01_json_string(report,float_q8_sha);fputs("},\"double\":{\"path\":",report);strat01_json_string(report,double_q8_path);fputs(",\"bytes\":14016,\"sha256\":",report);strat01_json_string(report,double_q8_sha);fprintf(report,"},\"changed_blocks\":%zu,\"total_blocks\":48,\"changed_bytes\":%zu,\"total_bytes\":14016},\n",changed_blocks,changed_bytes);
    fputs("\"engine_source_sha256\":",report);strat01_json_string(report,engine_sha);fputs(",\n\"diagnostic_source_sha256\":",report);strat01_json_string(report,header_sha);fputs(",\n\"compiler_family\":\"clang\",\n\"donor_graph_executions\":0,\n\"timing_or_rate_claim\":null\n}\n",report);
    if(fclose(report)!=0){report=NULL;snprintf(error,256,"RMSNorm diagnostic report close failure");goto finish;}report=NULL;ok=1;
finish:
    if(report)fclose(report);free(weight);free(ref_input);free(c_input);free(ref_float);free(ref_double);free(c_float);free(c_double);free(c_float_up);free(c_float_gate);free(c_double_up);free(c_double_gate);free(float_q8);free(double_q8);strat01_free_inventory(&inv);
    if(!ok){if(!error[0])snprintf(error,256,"unspecified RMSNorm diagnostic failure");strat01_r2b_rms_write_failure(out_dir,error);fprintf(stderr,"STRAT-01 Rung-2B RMSNorm diagnostic refused: %s\n",error);return 1;}
    fprintf(stderr,"STRAT-01 Rung-2B RMSNorm diagnostic: OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\n");return 0;
}

static int strat01_r2b_rmsnorm_diag_selftest(void) {
    int bad=0,checks=0;float *x=(float *)malloc(1536U*sizeof(float)),*w=(float *)malloc(1536U*sizeof(float)),*a=(float *)malloc(1536U*sizeof(float)),*b=(float *)malloc(1536U*sizeof(float));strat01_tensor t;
#define RMS_CHECK(z) do{++checks;if(!(z))++bad;}while(0)
    RMS_CHECK(strat01_r2b_cross_input_selftest()==0);RMS_CHECK(x&&w&&a&&b);if(x&&w&&a&&b){for(unsigned i=0;i<1536U;++i){x[i]=i?1.0f:10000.0f;w[i]=1.0f;}strat01_r2a_rmsnorm(x,w,a,1,1536,1e-6f);strat01_r2b_rmsnorm_double_sum(x,w,b,1,1536,1e-6f);RMS_CHECK(memcmp(a,b,1536U*sizeof(float))!=0);}
    memset(&t,0,sizeof(t));t.name=(char *)"blk.0.ffn_norm.weight";t.type=STRAT01_GGML_F32;t.rank=1;t.dims[0]=1536;t.offset=305786880;t.span=6144;t.file_offset=311889792;RMS_CHECK(strat01_r2b_rms_descriptor_ok(&t));t.span++;RMS_CHECK(!strat01_r2b_rms_descriptor_ok(&t));free(x);free(w);free(a);free(b);
#undef RMS_CHECK
    fprintf(stderr,"STRAT-01 Rung-2B RMSNorm diagnostic selftest: %s (%d checks)\n",bad?"FAIL":"PASS",checks);return bad?1:0;
}

#pragma STDC FP_CONTRACT ON
#endif /* STRAT01_GGUF_RUNG2B_RMSNORM_DIAG_H */
