/* Frozen layer-1 routed/shared SwiGLU -> Q6_K down cross-input diagnostic. */
#ifndef STRAT01_GGUF_LAYER1_Q6_CROSS_INPUT_H
#define STRAT01_GGUF_LAYER1_Q6_CROSS_INPUT_H

#define STRAT01_L1Q6_TOPK_COUNT 32U
#define STRAT01_L1Q6_ROUTED_IN_COUNT (8U*4U*1280U)
#define STRAT01_L1Q6_ROUTED_OUT_COUNT (8U*4U*1536U)
#define STRAT01_L1Q6_SHARED_IN_COUNT (8U*1280U)
#define STRAT01_L1Q6_SHARED_OUT_COUNT (8U*1536U)
#define STRAT01_L1Q6_TOPK_SHA "557f502a4cc9f24ec866d11e63b857b9c60e40cc0cef43bfeb95c6ee59648e95"
#define STRAT01_L1Q6_C_ROUTED_IN_SHA "ba62cdfd1689cd299dd476e3ac8a31d1b5e374a4028596c8472ef2b7345cf8fa"
#define STRAT01_L1Q6_R_ROUTED_IN_SHA "5097dc8599d0635477ad83f63f7e2de21b659775abc345dc933ce1e173be29e8"
#define STRAT01_L1Q6_C_SHARED_IN_SHA "30ec7ceafc08c3d30b73c21919a7d31619f69b9cd8045421e00c2e38af81a6b8"
#define STRAT01_L1Q6_R_SHARED_IN_SHA "fa0d1b4fe5da3f561ac30b8b8cb95520f359232528ea66c1d60e9244642c6003"

typedef struct { char path[1024],sha[65]; } strat01_l1q6_output;

static int strat01_l1q6_read_f32(const char *path,const char *expected,uint64_t count,float *out,char actual[65],char error[256]) {
    FILE *f=NULL;uint64_t bytes=0;uint8_t raw[4];
    if(!strat01_sha256_file(path,actual,&bytes,error)||bytes!=count*4U||strcmp(actual,expected)){if(!error[0])snprintf(error,256,"layer-1 Q6 F32 input identity mismatch");return 0;}
    f=fopen(path,"rb");if(!f){snprintf(error,256,"cannot open layer-1 Q6 F32 input");return 0;}
    for(uint64_t i=0;i<count;++i){if(fread(raw,1,4,f)!=4){fclose(f);snprintf(error,256,"layer-1 Q6 F32 input short read");return 0;}out[i]=strat01_r2a_f32le(raw);if(!isfinite(out[i])){fclose(f);snprintf(error,256,"layer-1 Q6 F32 input non-finite");return 0;}}
    if(ferror(f)||fclose(f)!=0){snprintf(error,256,"layer-1 Q6 F32 input I/O failure");return 0;}return 1;
}

static int strat01_l1q6_read_topk(const char *path,int32_t ids[STRAT01_L1Q6_TOPK_COUNT],char actual[65],char error[256]) {
    FILE *f=NULL;uint64_t bytes=0;uint8_t raw[4];
    if(!strat01_sha256_file(path,actual,&bytes,error)||bytes!=STRAT01_L1Q6_TOPK_COUNT*4U||strcmp(actual,STRAT01_L1Q6_TOPK_SHA)){if(!error[0])snprintf(error,256,"layer-1 Q6 top-k identity mismatch");return 0;}
    f=fopen(path,"rb");if(!f){snprintf(error,256,"cannot open layer-1 Q6 top-k input");return 0;}
    for(unsigned i=0;i<STRAT01_L1Q6_TOPK_COUNT;++i){if(fread(raw,1,4,f)!=4){fclose(f);snprintf(error,256,"layer-1 Q6 top-k short read");return 0;}ids[i]=(int32_t)strat01_rung1_u32le(raw);if(ids[i]<0||ids[i]>=64){fclose(f);snprintf(error,256,"layer-1 Q6 expert ID out of range");return 0;}}
    if(ferror(f)||fclose(f)!=0){snprintf(error,256,"layer-1 Q6 top-k I/O failure");return 0;}return 1;
}

static int strat01_l1q6_routed(const char *model,const strat01_tensor *experts,const int32_t ids[STRAT01_L1Q6_TOPK_COUNT],const float *input,float *output,char error[256]) {
    for(unsigned tok=0;tok<8U;++tok)for(unsigned slot=0;slot<4U;++slot){size_t item=(size_t)tok*4U+slot;strat01_tensor view;
        if(!strat01_r2c_expert_view(experts,(unsigned)ids[item],1280U,1536U,&view,error)||
           !strat01_r2c_q6_matmul_batch(model,&view,input+item*1280U,1U,1280U,output+item*1536U,error))return 0;
    }return 1;
}

static int strat01_l1q6_write(const char *out_dir,const char *leaf,const float *values,uint64_t count,strat01_l1q6_output *out,char error[256]) {
    return strat01_r2a_path(out->path,out_dir,leaf)&&strat01_q4q8_write_output(out->path,values,count,out->sha,error);
}

static int strat01_l1q6_write_failure(const char *out_dir,const char *error) {
    char path[1024];FILE *f;if(!strat01_r2a_path(path,out_dir,"strat01_layer1_q6_cross_input.json")||(f=fopen(path,"wb"))==NULL)return 0;
    fputs("{\"command\":\"--strat01-layer1-q6-cross-input\",\"state\":\"DIAGNOSTIC_FAILURE\",\"error\":",f);strat01_json_string(f,error);fputs(",\"donor_graph_executions\":0,\"reference_graph_executions\":0}\n",f);fclose(f);return 1;
}

static int strat01_l1q6_cli(const char *model,const char *topk_path,const char *c_routed_path,const char *r_routed_path,const char *c_shared_path,const char *r_shared_path,const char *out_dir,const char *engine_source_path) {
    strat01_inventory inv;const strat01_tensor *experts=NULL,*shared=NULL;int32_t ids[STRAT01_L1Q6_TOPK_COUNT],mutated_ids[STRAT01_L1Q6_TOPK_COUNT];
    float *c_routed=NULL,*r_routed=NULL,*c_shared=NULL,*r_shared=NULL,*tmp_routed=NULL,*tmp_shared=NULL;
    float *outputs[7]={NULL,NULL,NULL,NULL,NULL,NULL,NULL};const uint64_t output_counts[7]={STRAT01_L1Q6_ROUTED_OUT_COUNT,STRAT01_L1Q6_ROUTED_OUT_COUNT,STRAT01_L1Q6_SHARED_OUT_COUNT,STRAT01_L1Q6_SHARED_OUT_COUNT,STRAT01_L1Q6_ROUTED_OUT_COUNT,STRAT01_L1Q6_SHARED_OUT_COUNT,STRAT01_L1Q6_ROUTED_OUT_COUNT};
    const char *output_names[7]={"reference_routed_current_q6","current_routed_current_q6","reference_shared_current_q6","current_shared_current_q6","control_negated_reference_routed","control_negated_reference_shared","control_mutated_expert_id"};
    const char *output_leaves[7]={"reference_routed_current_q6.f32le","current_routed_current_q6.f32le","reference_shared_current_q6.f32le","current_shared_current_q6.f32le","control_negated_reference_routed.f32le","control_negated_reference_shared.f32le","control_mutated_expert_id.f32le"};
    strat01_l1q6_output records[7];char topk_sha[65]={0},input_sha[4][65]={{0}},model_sha[65]={0},engine_sha[65]={0},header_sha[65]={0},report_path[1024],error[256]={0};uint64_t hashed=0,parsed=0,tmp=0;FILE *report=NULL;int ok=0;
    const char *input_paths[4]={c_routed_path,r_routed_path,c_shared_path,r_shared_path};const char *input_names[4]={"current_routed_swiglu","reference_routed_swiglu","current_shared_swiglu","reference_shared_swiglu"};const char *input_expected[4]={STRAT01_L1Q6_C_ROUTED_IN_SHA,STRAT01_L1Q6_R_ROUTED_IN_SHA,STRAT01_L1Q6_C_SHARED_IN_SHA,STRAT01_L1Q6_R_SHARED_IN_SHA};const uint64_t input_counts[4]={STRAT01_L1Q6_ROUTED_IN_COUNT,STRAT01_L1Q6_ROUTED_IN_COUNT,STRAT01_L1Q6_SHARED_IN_COUNT,STRAT01_L1Q6_SHARED_IN_COUNT};float **input_dst[4]={&c_routed,&r_routed,&c_shared,&r_shared};
    memset(&inv,0,sizeof(inv));memset(records,0,sizeof(records));
#if !defined(__clang__)
    snprintf(error,256,"layer-1 Q6 diagnostic requires Clang");goto finish;
#endif
    if(!strat01_sha256_file(model,model_sha,&hashed,error)||hashed!=STRAT01_EXPECTED_SIZE||strcmp(model_sha,STRAT01_EXPECTED_SHA256)){if(!error[0])snprintf(error,256,"layer-1 Q6 artifact mismatch");goto finish;}
    if(!strat01_parse_gguf(model,&inv,&parsed,error)||parsed!=hashed){if(!error[0])snprintf(error,256,"layer-1 Q6 GGUF parse mismatch");goto finish;}
    if(!strat01_r2a_check_spec(&inv,&strat01_r2c_moe_specs[5],&experts,error)||!strat01_r2a_check_spec(&inv,&strat01_r2c_moe_specs[8],&shared,error))goto finish;
    if(!strat01_l1q6_read_topk(topk_path,ids,topk_sha,error))goto finish;
    for(unsigned i=0;i<4U;++i){*input_dst[i]=strat01_r2a_alloc((size_t)input_counts[i],error);if(!*input_dst[i]||!strat01_l1q6_read_f32(input_paths[i],input_expected[i],input_counts[i],*input_dst[i],input_sha[i],error))goto finish;}
    for(unsigned i=0;i<7U;++i){outputs[i]=strat01_r2a_alloc((size_t)output_counts[i],error);if(!outputs[i])goto finish;}
    if(!strat01_l1q6_routed(model,experts,ids,r_routed,outputs[0],error)||!strat01_l1q6_routed(model,experts,ids,c_routed,outputs[1],error)||
       !strat01_r2c_q6_matmul_batch(model,shared,r_shared,8U,1280U,outputs[2],error)||!strat01_r2c_q6_matmul_batch(model,shared,c_shared,8U,1280U,outputs[3],error))goto finish;
    tmp_routed=strat01_r2a_alloc(STRAT01_L1Q6_ROUTED_IN_COUNT,error);tmp_shared=strat01_r2a_alloc(STRAT01_L1Q6_SHARED_IN_COUNT,error);if(!tmp_routed||!tmp_shared)goto finish;
    for(size_t i=0;i<STRAT01_L1Q6_ROUTED_IN_COUNT;++i)tmp_routed[i]=-r_routed[i];for(size_t i=0;i<STRAT01_L1Q6_SHARED_IN_COUNT;++i)tmp_shared[i]=-r_shared[i];
    if(!strat01_l1q6_routed(model,experts,ids,tmp_routed,outputs[4],error)||!strat01_r2c_q6_matmul_batch(model,shared,tmp_shared,8U,1280U,outputs[5],error))goto finish;
    memcpy(mutated_ids,ids,sizeof(ids));mutated_ids[0]=(mutated_ids[0]+1)%64;if(!strat01_l1q6_routed(model,experts,mutated_ids,r_routed,outputs[6],error)||!memcmp(outputs[0],outputs[6],STRAT01_L1Q6_ROUTED_OUT_COUNT*4U)){snprintf(error,256,"layer-1 Q6 expert-ID control did not change output");goto finish;}
    for(unsigned i=0;i<7U;++i)if(!strat01_l1q6_write(out_dir,output_leaves[i],outputs[i],output_counts[i],&records[i],error))goto finish;
    if(!strat01_sha256_file(engine_source_path,engine_sha,&tmp,error)||!strat01_sha256_file(__FILE__,header_sha,&tmp,error))goto finish;
    if(!strat01_r2a_path(report_path,out_dir,"strat01_layer1_q6_cross_input.json")||(report=fopen(report_path,"wb"))==NULL){snprintf(error,256,"cannot write layer-1 Q6 report");goto finish;}
    fputs("{\n\"command\":\"--strat01-layer1-q6-cross-input\",\n\"state\":\"OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\",\n\"self_certifies_pass\":false,\n\"model\":{\"path\":",report);strat01_json_string(report,model);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",hashed);strat01_json_string(report,model_sha);fputs("},\n\"topk\":{\"path\":",report);strat01_json_string(report,topk_path);fputs(",\"bytes\":128,\"sha256\":",report);strat01_json_string(report,topk_sha);fputs(",\"ids\":[",report);for(unsigned i=0;i<STRAT01_L1Q6_TOPK_COUNT;++i)fprintf(report,"%s%d",i?",":"",ids[i]);fputs("]},\n\"inputs\":{",report);
    for(unsigned i=0;i<4U;++i){fprintf(report,"%s\"%s\":{\"path\":",i?",":"",input_names[i]);strat01_json_string(report,input_paths[i]);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",input_counts[i]*4U);strat01_json_string(report,input_sha[i]);fputc('}',report);}fputs("},\n\"tensors\":{\"routed\":{\"name\":",report);strat01_json_string(report,experts->name);fprintf(report,",\"type\":%u,\"rank\":%u,\"dims\":[%" PRIu64 ",%" PRIu64 ",%" PRIu64 "],\"file_offset\":%" PRIu64 ",\"span\":%" PRIu64 "},\"shared\":{\"name\":",experts->type,experts->rank,experts->dims[0],experts->dims[1],experts->dims[2],experts->file_offset,experts->span);strat01_json_string(report,shared->name);fprintf(report,",\"type\":%u,\"rank\":%u,\"dims\":[%" PRIu64 ",%" PRIu64 "],\"file_offset\":%" PRIu64 ",\"span\":%" PRIu64 "}},\n\"outputs\":{",shared->type,shared->rank,shared->dims[0],shared->dims[1],shared->file_offset,shared->span);
    for(unsigned i=0;i<7U;++i){fprintf(report,"%s\"%s\":{\"path\":",i?",":"",output_names[i]);strat01_json_string(report,records[i].path);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",output_counts[i]*4U);strat01_json_string(report,records[i].sha);fputc('}',report);}fputs("},\n\"controls\":{\"q8_deterministic\":true,\"mutated_expert_id\":",report);fprintf(report,"%d},\n\"engine_source_sha256\":",mutated_ids[0]);strat01_json_string(report,engine_sha);fputs(",\n\"diagnostic_source_sha256\":",report);strat01_json_string(report,header_sha);fputs(",\n\"compiler_family\":\"clang\",\n\"donor_graph_executions\":0,\n\"reference_graph_executions\":0,\n\"timing_or_rate_claim\":null\n}\n",report);
    if(fclose(report)!=0){report=NULL;snprintf(error,256,"layer-1 Q6 report close failure");goto finish;}report=NULL;ok=1;
finish:
    if(report)fclose(report);for(unsigned i=0;i<4U;++i)free(*input_dst[i]);for(unsigned i=0;i<7U;++i)free(outputs[i]);free(tmp_routed);free(tmp_shared);strat01_free_inventory(&inv);
    if(!ok){if(!error[0])snprintf(error,256,"unspecified layer-1 Q6 failure");strat01_l1q6_write_failure(out_dir,error);fprintf(stderr,"STRAT-01 layer-1 Q6 refused: %s\n",error);return 1;}fprintf(stderr,"STRAT-01 layer-1 Q6: OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\n");return 0;
}

static int strat01_l1q6_selftest(void) { int bad=0,checks=0;float row[1280];strat01_q8_k_block a[5],b[5];strat01_tensor src={0},view={0};char error[256]={0};
#define L1Q6_CHECK(z) do{++checks;if(!(z))++bad;}while(0)
    L1Q6_CHECK(strlen(STRAT01_L1Q6_TOPK_SHA)==64U);L1Q6_CHECK(STRAT01_L1Q6_ROUTED_IN_COUNT==40960U);L1Q6_CHECK(STRAT01_L1Q6_ROUTED_OUT_COUNT==49152U);
    for(unsigned i=0;i<1280U;++i)row[i]=(float)((int)(i%37U)-18)/19.0f;L1Q6_CHECK(strat01_quantize_q8_k_row(row,1280U,a));L1Q6_CHECK(strat01_quantize_q8_k_row(row,1280U,b));L1Q6_CHECK(!memcmp(a,b,sizeof(a)));
    src.type=STRAT01_GGML_Q6_K;src.rank=3;src.dims[0]=1280;src.dims[1]=1536;src.dims[2]=64;src.file_offset=4096;L1Q6_CHECK(strat01_r2c_expert_view(&src,7,1280,1536,&view,error));L1Q6_CHECK(view.rank==2&&view.dims[2]==0&&view.file_offset>src.file_offset);L1Q6_CHECK(!strat01_r2c_expert_view(&src,64,1280,1536,&view,error));
#undef L1Q6_CHECK
    fprintf(stderr,"STRAT-01 layer-1 Q6 cross-input selftest: %s (%d checks)\n",bad?"FAIL":"PASS",checks);return bad?1:0;
}

#endif
