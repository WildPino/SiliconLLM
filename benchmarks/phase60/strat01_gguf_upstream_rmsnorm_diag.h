/* Diagnostic-only propagation of pinned double RMSNorm semantics. */
#ifndef STRAT01_GGUF_UPSTREAM_RMSNORM_DIAG_H
#define STRAT01_GGUF_UPSTREAM_RMSNORM_DIAG_H

#pragma STDC FP_CONTRACT OFF

#define STRAT01_UPSTREAM_ATTN_BASELINE_SHA "f746d41ff1d909d70d09b241c2a3f2d66fd837b51dbeec5ca95ba618d3456d9e"
#define STRAT01_UPSTREAM_FFN_BASELINE_SHA "ea5d60791c4404fbb98e7839fa73ff8112c54abba6060a7fbc3d5e83b951e3e0"

typedef struct { float *norm,*up,*gate; } strat01_upstream_ffn_arm;

static int strat01_upstream_read_f32(const char *path,const char *expected,float *values,size_t count,char actual[65],char error[256]) {
    FILE *f=NULL;uint64_t bytes=0;uint8_t raw[4];
    if(!strat01_sha256_file(path,actual,&bytes,error)||bytes!=count*4U||strcmp(actual,expected)){if(!error[0])snprintf(error,256,"upstream RMSNorm baseline identity mismatch");return 0;}
    f=fopen(path,"rb");if(!f){snprintf(error,256,"cannot open upstream RMSNorm baseline");return 0;}
    for(size_t i=0;i<count;++i){if(fread(raw,1,4,f)!=4){snprintf(error,256,"upstream RMSNorm baseline short read");fclose(f);return 0;}values[i]=strat01_r2a_f32le(raw);if(!isfinite(values[i])){snprintf(error,256,"upstream RMSNorm baseline non-finite");fclose(f);return 0;}}
    if(ferror(f)||fclose(f)!=0){snprintf(error,256,"upstream RMSNorm baseline I/O failure");return 0;}return 1;
}

static int strat01_upstream_build_range(const char *path,const strat01_tensor *tensors[8],strat01_r2a_arm *a,unsigned start,unsigned count,char error[256]) {
    float *attn_w=NULL,*kv_w=NULL;
    if(!count||start>=8||count>8-start){snprintf(error,256,"invalid upstream double schedule range");return 0;}
    attn_w=strat01_r2a_alloc(1536,error);kv_w=strat01_r2a_alloc(512,error);if(!attn_w||!kv_w)goto fail;
    if(!strat01_r2a_embedding(path,tensors[0],strat01_r2a_tokens+start,count,a->embd+(size_t)start*1536,error)||!strat01_r2a_read_f32_vector(path,tensors[1],attn_w,1536,error))goto fail;
    strat01_r2b_rmsnorm_double_sum(a->embd+(size_t)start*1536,attn_w,a->attn_norm+(size_t)start*1536,count,1536,STRAT01_R2A_RMS_EPS);
    if(!strat01_r2a_matmul_batch(path,tensors[2],a->attn_norm+(size_t)start*1536,count,1536,a->q+(size_t)start*6144,6144,error)||!strat01_r2a_matmul_batch(path,tensors[3],a->attn_norm+(size_t)start*1536,count,1536,a->kv_cmpr_pe+(size_t)start*576,576,error)||!strat01_r2a_read_f32_vector(path,tensors[4],kv_w,512,error))goto fail;
    for(unsigned tok=start;tok<start+count;++tok){memcpy(a->kv_cmpr+(size_t)tok*512,a->kv_cmpr_pe+(size_t)tok*576,512*4);memcpy(a->k_pe+(size_t)tok*64,a->kv_cmpr_pe+(size_t)tok*576+512,64*4);strat01_r2a_rope64(a->k_pe+(size_t)tok*64,strat01_r2a_positions[tok]);}
    {float *tmp=strat01_r2a_alloc((size_t)count*512U,error);if(!tmp)goto fail;strat01_r2b_rmsnorm_double_sum(a->kv_cmpr+(size_t)start*512,kv_w,tmp,count,512,STRAT01_R2A_RMS_EPS);memcpy(a->kv_cmpr+(size_t)start*512,tmp,(size_t)count*512U*4U);free(tmp);}
    for(unsigned tok=start;tok<start+count;++tok)for(unsigned h=0;h<32;++h){float *qp=a->q_pe+((size_t)tok*32+h)*64;memcpy(qp,a->q+(size_t)tok*6144+(size_t)h*192+128,64*4);strat01_r2a_rope64(qp,strat01_r2a_positions[tok]);}
    if(!strat01_r2a_kb_batch(path,tensors[5],a->q+(size_t)start*6144,count,a->q_abs+(size_t)start*32*512,error))goto fail;
    for(unsigned tok=start;tok<start+count;++tok){memcpy(a->kcur+(size_t)tok*576,a->kv_cmpr+(size_t)tok*512,512*4);memcpy(a->kcur+(size_t)tok*576+512,a->k_pe+(size_t)tok*64,64*4);memcpy(a->vcur+(size_t)tok*512,a->kv_cmpr+(size_t)tok*512,512*4);for(unsigned h=0;h<32;++h){float *qc=a->qcur+((size_t)tok*32+h)*576;memcpy(qc,a->q_abs+((size_t)tok*32+h)*512,512*4);memcpy(qc+512,a->q_pe+((size_t)tok*32+h)*64,64*4);}}
    free(attn_w);free(kv_w);return 1;
fail: free(attn_w);free(kv_w);return 0;
}

static int strat01_upstream_ffn_alloc(strat01_upstream_ffn_arm *a,char error[256]) {
    memset(a,0,sizeof(*a));a->norm=strat01_r2a_alloc(8U*1536U,error);a->up=strat01_r2a_alloc(8U*STRAT01_R2B_FFN,error);a->gate=strat01_r2a_alloc(8U*STRAT01_R2B_FFN,error);return a->norm&&a->up&&a->gate;
}

static void strat01_upstream_ffn_free(strat01_upstream_ffn_arm *a) { free(a->norm);free(a->up);free(a->gate);memset(a,0,sizeof(*a)); }

static int strat01_upstream_ffn_run(const char *path,const strat01_tensor *norm,const strat01_tensor *gate,const strat01_tensor *up,const strat01_r2a_arm *input,strat01_upstream_ffn_arm *out,char error[256]) {
    float *weight=strat01_r2a_alloc(1536U,error);if(!weight)return 0;
    if(!strat01_r2a_read_f32_vector(path,norm,weight,1536U,error)){free(weight);return 0;}
    strat01_r2b_rmsnorm_double_sum(input->ffn_inp,weight,out->norm,8U,1536U,STRAT01_R2A_RMS_EPS);free(weight);
    return strat01_r2a_matmul_batch(path,up,out->norm,8U,1536U,out->up,STRAT01_R2B_FFN,error)&&strat01_r2a_matmul_batch(path,gate,out->norm,8U,1536U,out->gate,STRAT01_R2B_FFN,error);
}

static int strat01_upstream_dump_ffn(strat01_upstream_ffn_arm *a,const char *out_dir,const char *arm,char shas[3][65],char error[256]) {
    strat01_r2a_dump d[3];const char *names[3]={"ffn_norm-0","ffn_up-0","ffn_gate-0"};const char *ops[3]={"MUL","MUL_MAT","MUL_MAT"};uint64_t widths[3]={1536,8960,8960};float *values[3]={a->norm,a->up,a->gate};
    for(unsigned i=0;i<3;++i){memset(&d[i],0,sizeof(d[i]));d[i].logical_name=names[i];d[i].selection="upstream double RMSNorm diagnostic";d[i].op=ops[i];d[i].rank=2;d[i].shape[0]=widths[i];d[i].shape[1]=8;d[i].payload_order="token,feature";d[i].data=values[i];d[i].count=8U*widths[i];if(!strat01_r2a_dump_f32(&d[i],out_dir,arm,error))return 0;strcpy(shas[i],d[i].sha256);}return 1;
}

static int strat01_upstream_q8_write(const char *out_dir,const char *leaf,const float *values,unsigned rows,unsigned width,char path[1024],char sha[65],strat01_q8_k_block **blocks_out,size_t *count_out,char error[256]) {
    const unsigned per=width/STRAT01_QK_K;size_t count=(size_t)rows*per;FILE *f=NULL;uint64_t bytes=0;strat01_q8_k_block *blocks=NULL;
    if(width%STRAT01_QK_K||!strat01_r2a_path(path,out_dir,leaf)){snprintf(error,256,"upstream Q8 path/shape mismatch");return 0;}
    blocks=(strat01_q8_k_block *)malloc(count*sizeof(*blocks));if(!blocks){snprintf(error,256,"upstream Q8 allocation failed");return 0;}
    for(unsigned r=0;r<rows;++r)if(!strat01_quantize_q8_k_row(values+(size_t)r*width,width,blocks+(size_t)r*per)){snprintf(error,256,"upstream Q8 quantization failed");free(blocks);return 0;}
    f=fopen(path,"wb");if(!f){snprintf(error,256,"upstream Q8 open failure");free(blocks);return 0;}if(fwrite(blocks,sizeof(*blocks),count,f)!=count){fclose(f);snprintf(error,256,"upstream Q8 write failure");free(blocks);return 0;}if(fclose(f)!=0){snprintf(error,256,"upstream Q8 close failure");free(blocks);return 0;}
    if(!strat01_sha256_file(path,sha,&bytes,error)||bytes!=count*sizeof(*blocks)){free(blocks);return 0;}*blocks_out=blocks;*count_out=count;return 1;
}

static void strat01_upstream_q8_census(const strat01_q8_k_block *a,const strat01_q8_k_block *b,size_t count,size_t *changed_blocks,size_t *changed_bytes) {
    const uint8_t *aa=(const uint8_t *)a,*bb=(const uint8_t *)b;*changed_blocks=0;*changed_bytes=0;for(size_t i=0;i<count;++i)if(memcmp(a+i,b+i,sizeof(*a)))++*changed_blocks;for(size_t i=0;i<count*sizeof(*a);++i)if(aa[i]!=bb[i])++*changed_bytes;
}

static int strat01_upstream_write_failure(const char *out_dir,const char *error) {
    char path[1024];FILE *f;if(!strat01_r2a_path(path,out_dir,"strat01_upstream_rmsnorm_diag.json")||(f=fopen(path,"wb"))==NULL)return 0;fputs("{\"command\":\"--strat01-upstream-rmsnorm-diagnostic\",\"state\":\"DIAGNOSTIC_FAILURE\",\"error\":",f);strat01_json_string(f,error);fputs(",\"donor_graph_executions\":0}\n",f);fclose(f);return 1;
}

static int strat01_upstream_rmsnorm_diag_cli(const char *model,const char *baseline_attn_path,const char *baseline_ffn_path,const char *out_dir,const char *engine_source_path) {
    strat01_inventory inv;const strat01_tensor *base[8]={0},*ffn[3]={0};strat01_r2a_arm pre,cache;strat01_upstream_ffn_arm pre_ffn,cache_ffn;strat01_r2a_dump dumps[12];
    float *baseline_attn=NULL,*baseline_ffn=NULL;strat01_q8_k_block *ba=NULL,*ca=NULL,*bf=NULL,*cf=NULL;size_t ba_n=0,ca_n=0,bf_n=0,cf_n=0,attn_cb=0,attn_cy=0,ffn_cb=0,ffn_cy=0;
    char error[256]={0},model_sha[65]={0},engine_sha[65]={0},header_sha[65]={0},attn_input_sha[65]={0},ffn_input_sha[65]={0};char ffn_shas[2][3][65];
    char q8_paths[4][1024],q8_shas[4][65],pf_cache_path[1024],pf_cache_sha[65],c7_prefix_path[1024],c7_prefix_sha[65],c7_final_path[1024],c7_final_sha[65],report_path[1024];
    uint64_t hashed=0,parsed=0,tmp=0;FILE *report=NULL;int ok=0;memset(&inv,0,sizeof(inv));memset(&pre,0,sizeof(pre));memset(&cache,0,sizeof(cache));memset(&pre_ffn,0,sizeof(pre_ffn));memset(&cache_ffn,0,sizeof(cache_ffn));memset(ffn_shas,0,sizeof(ffn_shas));memset(q8_shas,0,sizeof(q8_shas));
#if !defined(__clang__)
    snprintf(error,256,"upstream RMSNorm diagnostic requires Clang");goto finish;
#endif
    if(sizeof(float)!=4||sizeof(double)!=8||sizeof(strat01_q8_k_block)!=292U){snprintf(error,256,"unsupported upstream RMSNorm host layout");goto finish;}
    if(!strat01_sha256_file(model,model_sha,&hashed,error)||hashed!=STRAT01_EXPECTED_SIZE||strcmp(model_sha,STRAT01_EXPECTED_SHA256)){if(!error[0])snprintf(error,256,"upstream RMSNorm artifact identity mismatch");goto finish;}
    if(!strat01_parse_gguf(model,&inv,&parsed,error)||parsed!=hashed){if(!error[0])snprintf(error,256,"upstream RMSNorm GGUF parse mismatch");goto finish;}
    for(unsigned i=0;i<8U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2a_specs[i],&base[i],error))goto finish;
    for(unsigned i=0;i<3U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2b_specs[i],&ffn[i],error))goto finish;
    baseline_attn=strat01_r2a_alloc(8U*1536U,error);baseline_ffn=strat01_r2a_alloc(8U*1536U,error);if(!baseline_attn||!baseline_ffn)goto finish;
    if(!strat01_upstream_read_f32(baseline_attn_path,STRAT01_UPSTREAM_ATTN_BASELINE_SHA,baseline_attn,8U*1536U,attn_input_sha,error)||!strat01_upstream_read_f32(baseline_ffn_path,STRAT01_UPSTREAM_FFN_BASELINE_SHA,baseline_ffn,8U*1536U,ffn_input_sha,error))goto finish;
    if(!strat01_r2a_arm_alloc(&pre,error)||!strat01_r2a_arm_alloc(&cache,error)||!strat01_upstream_ffn_alloc(&pre_ffn,error)||!strat01_upstream_ffn_alloc(&cache_ffn,error))goto finish;
    if(!strat01_upstream_build_range(model,base,&pre,0,8,error)||!strat01_r2a_run_schedule(model,base[6],base[7],&pre,0,error))goto finish;
    if(!strat01_upstream_build_range(model,base,&cache,0,7,error))goto finish;for(unsigned t=0;t<7;++t)strat01_r2a_cache_write(&cache,t,cache.kcur+(size_t)t*576);
    if(!strat01_r2a_write_cache_dump(out_dir,"cached7p1_cache_prefix7.f32",cache.cache,7,c7_prefix_path,c7_prefix_sha,error))goto finish;memset(cache.cache,0,sizeof(cache.cache));cache.occupied=0;
    if(!strat01_upstream_build_range(model,base,&cache,7,1,error)||!strat01_r2a_run_schedule(model,base[6],base[7],&cache,1,error))goto finish;
    if(!strat01_upstream_ffn_run(model,ffn[0],ffn[1],ffn[2],&pre,&pre_ffn,error)||!strat01_upstream_ffn_run(model,ffn[0],ffn[1],ffn[2],&cache,&cache_ffn,error))goto finish;
    strat01_r2a_make_dumps(&pre,dumps);for(unsigned i=0;i<12U;++i)if(!strat01_r2a_dump_f32(&dumps[i],out_dir,"prefill8",error))goto finish;
    if(!strat01_r2a_write_cache_dump(out_dir,"prefill8_cache_final.f32",pre.cache,8,pf_cache_path,pf_cache_sha,error)||!strat01_r2a_write_manifest(out_dir,"prefill8",dumps,pf_cache_path,pf_cache_sha,NULL,NULL,error))goto finish;
    strat01_r2a_make_dumps(&cache,dumps);for(unsigned i=0;i<12U;++i)if(!strat01_r2a_dump_f32(&dumps[i],out_dir,"cached7p1",error))goto finish;
    if(!strat01_r2a_write_cache_dump(out_dir,"cached7p1_cache_final.f32",cache.cache,8,c7_final_path,c7_final_sha,error)||!strat01_r2a_write_manifest(out_dir,"cached7p1",dumps,c7_final_path,c7_final_sha,c7_prefix_path,c7_prefix_sha,error))goto finish;
    if(!strat01_upstream_dump_ffn(&pre_ffn,out_dir,"prefill8_double",ffn_shas[0],error)||!strat01_upstream_dump_ffn(&cache_ffn,out_dir,"cached7p1_double",ffn_shas[1],error))goto finish;
    if(!strat01_upstream_q8_write(out_dir,"baseline_attn_norm.q8k",baseline_attn,8,1536,q8_paths[0],q8_shas[0],&ba,&ba_n,error)||!strat01_upstream_q8_write(out_dir,"candidate_attn_norm.q8k",pre.attn_norm,8,1536,q8_paths[1],q8_shas[1],&ca,&ca_n,error)||!strat01_upstream_q8_write(out_dir,"baseline_ffn_norm.q8k",baseline_ffn,8,1536,q8_paths[2],q8_shas[2],&bf,&bf_n,error)||!strat01_upstream_q8_write(out_dir,"candidate_ffn_norm.q8k",pre_ffn.norm,8,1536,q8_paths[3],q8_shas[3],&cf,&cf_n,error))goto finish;
    if(ba_n!=ca_n||bf_n!=cf_n){snprintf(error,256,"upstream Q8 census size mismatch");goto finish;}strat01_upstream_q8_census(ba,ca,ba_n,&attn_cb,&attn_cy);strat01_upstream_q8_census(bf,cf,bf_n,&ffn_cb,&ffn_cy);
    if(!strat01_sha256_file(engine_source_path,engine_sha,&tmp,error))strcpy(engine_sha,"unavailable");error[0]=0;if(!strat01_sha256_file(__FILE__,header_sha,&tmp,error))strcpy(header_sha,"unavailable");error[0]=0;
    if(!strat01_r2a_path(report_path,out_dir,"strat01_upstream_rmsnorm_diag.json")||(report=fopen(report_path,"wb"))==NULL){snprintf(error,256,"cannot write upstream RMSNorm report");goto finish;}
    fputs("{\n  \"command\": \"--strat01-upstream-rmsnorm-diagnostic\",\n  \"state\": \"OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\",\n  \"self_certifies_pass\": false,\n  \"model\": {\"path\": ",report);strat01_json_string(report,model);fprintf(report,", \"bytes\": %" PRIu64 ", \"sha256\": ",hashed);strat01_json_string(report,model_sha);
    fputs("},\n  \"baseline_inputs\": {\"attn_norm\": {\"path\": ",report);strat01_json_string(report,baseline_attn_path);fputs(", \"sha256\": ",report);strat01_json_string(report,attn_input_sha);fputs("}, \"ffn_norm\": {\"path\": ",report);strat01_json_string(report,baseline_ffn_path);fputs(", \"sha256\": ",report);strat01_json_string(report,ffn_input_sha);fputs("}},\n  \"engine_source_sha256\": ",report);strat01_json_string(report,engine_sha);fputs(",\n  \"diagnostic_source_sha256\": ",report);strat01_json_string(report,header_sha);
    fprintf(report,",\n  \"ffn_output_sha256\": {\"prefill8\": {\"norm\": \"%s\", \"up\": \"%s\", \"gate\": \"%s\"}, \"cached7p1\": {\"norm\": \"%s\", \"up\": \"%s\", \"gate\": \"%s\"}},\n",ffn_shas[0][0],ffn_shas[0][1],ffn_shas[0][2],ffn_shas[1][0],ffn_shas[1][1],ffn_shas[1][2]);
    fprintf(report,"  \"q8_census\": {\"attention\": {\"changed_blocks\": %zu, \"total_blocks\": %zu, \"changed_bytes\": %zu, \"total_bytes\": %zu}, \"ffn\": {\"changed_blocks\": %zu, \"total_blocks\": %zu, \"changed_bytes\": %zu, \"total_bytes\": %zu}},\n",attn_cb,ba_n,attn_cy,ba_n*sizeof(*ba),ffn_cb,bf_n,ffn_cy,bf_n*sizeof(*bf));
    fputs("  \"compiler_family\": \"clang\",\n  \"donor_graph_executions\": 1,\n  \"timing_or_rate_claim\": null\n}\n",report);if(fclose(report)!=0){report=NULL;snprintf(error,256,"upstream RMSNorm report close failure");goto finish;}report=NULL;ok=1;
finish:
    if(report)fclose(report);free(baseline_attn);free(baseline_ffn);free(ba);free(ca);free(bf);free(cf);strat01_free_inventory(&inv);strat01_r2a_arm_free(&pre);strat01_r2a_arm_free(&cache);strat01_upstream_ffn_free(&pre_ffn);strat01_upstream_ffn_free(&cache_ffn);
    if(!ok){if(!error[0])snprintf(error,256,"unspecified upstream RMSNorm diagnostic failure");strat01_upstream_write_failure(out_dir,error);fprintf(stderr,"STRAT-01 upstream RMSNorm diagnostic refused: %s\n",error);return 1;}
    fprintf(stderr,"STRAT-01 upstream RMSNorm diagnostic: OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\n");return 0;
}

static int strat01_upstream_rmsnorm_diag_selftest(void) {
    int bad=0,checks=0;float *x=(float *)malloc(1536U*sizeof(float)),*w=(float *)malloc(1536U*sizeof(float)),*a=(float *)malloc(1536U*sizeof(float)),*b=(float *)malloc(1536U*sizeof(float));
#define UP_CHECK(z) do{++checks;if(!(z))++bad;}while(0)
    UP_CHECK(strat01_r2b_rmsnorm_diag_selftest()==0);UP_CHECK(x&&w&&a&&b);if(x&&w&&a&&b){for(unsigned i=0;i<1536U;++i){x[i]=i?1.0f:10000.0f;w[i]=1.0f;}strat01_r2a_rmsnorm(x,w,a,1,1536,1e-6f);strat01_r2b_rmsnorm_double_sum(x,w,b,1,1536,1e-6f);UP_CHECK(memcmp(a,b,1536U*sizeof(float))!=0);UP_CHECK(isfinite(b[0])&&isfinite(b[1535]));}UP_CHECK(strlen(STRAT01_UPSTREAM_ATTN_BASELINE_SHA)==64);UP_CHECK(strlen(STRAT01_UPSTREAM_FFN_BASELINE_SHA)==64);free(x);free(w);free(a);free(b);
#undef UP_CHECK
    fprintf(stderr,"STRAT-01 upstream RMSNorm diagnostic selftest: %s (%d checks)\n",bad?"FAIL":"PASS",checks);return bad?1:0;
}

#pragma STDC FP_CONTRACT ON
#endif /* STRAT01_GGUF_UPSTREAM_RMSNORM_DIAG_H */
