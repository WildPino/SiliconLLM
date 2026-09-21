/* Diagnostic-only composition of double RMSNorm and K-B Q5_0 x Q8_0. */
#ifndef STRAT01_GGUF_COMBINED_RMS_Q5Q8_H
#define STRAT01_GGUF_COMBINED_RMS_Q5Q8_H

#define STRAT01_COMBINED_KB_Q8_SHA "5dae7c9874e8ca79fdccc9ceda45e652f729a2176de0bbebb24453ae818213f2"
#define STRAT01_COMBINED_KB_OUT_SHA "cf69d34fa506ee356ab5fb49982e7f872505e47312c7863540d1e0464036d51a"

static int strat01_combined_kb_range(const uint8_t *weights,const float *q,unsigned batch,float *out,char error[256]) {
    for(unsigned tok=0;tok<batch;++tok)for(unsigned head=0;head<32U;++head){
        strat01_kb_q8_0_block q8[4];strat01_kb_quantize_q8_0(q+(size_t)tok*6144U+(size_t)head*192U,q8);
        for(unsigned feature=0;feature<512U;++feature){const uint8_t *row=weights+((size_t)head*512U+feature)*88U;float value=strat01_kb_q5q8_dot(row,q8,0);if(!isfinite(value)){snprintf(error,256,"combined K-B non-finite output");return 0;}out[((size_t)tok*32U+head)*512U+feature]=value;}
    }
    return 1;
}

static int strat01_combined_build_range(const char *path,const strat01_tensor *tensors[8],const uint8_t *kb_weights,strat01_r2a_arm *a,unsigned start,unsigned count,char error[256]) {
    float *attn_w=NULL,*kv_w=NULL;
    if(!count||start>=8U||count>8U-start){snprintf(error,256,"invalid combined schedule range");return 0;}
    attn_w=strat01_r2a_alloc(1536U,error);kv_w=strat01_r2a_alloc(512U,error);if(!attn_w||!kv_w)goto fail;
    if(!strat01_r2a_embedding(path,tensors[0],strat01_r2a_tokens+start,count,a->embd+(size_t)start*1536U,error)||!strat01_r2a_read_f32_vector(path,tensors[1],attn_w,1536U,error))goto fail;
    strat01_r2b_rmsnorm_double_sum(a->embd+(size_t)start*1536U,attn_w,a->attn_norm+(size_t)start*1536U,count,1536U,STRAT01_R2A_RMS_EPS);
    if(!strat01_r2a_matmul_batch(path,tensors[2],a->attn_norm+(size_t)start*1536U,count,1536U,a->q+(size_t)start*6144U,6144U,error)||!strat01_r2a_matmul_batch(path,tensors[3],a->attn_norm+(size_t)start*1536U,count,1536U,a->kv_cmpr_pe+(size_t)start*576U,576U,error)||!strat01_r2a_read_f32_vector(path,tensors[4],kv_w,512U,error))goto fail;
    for(unsigned tok=start;tok<start+count;++tok){memcpy(a->kv_cmpr+(size_t)tok*512U,a->kv_cmpr_pe+(size_t)tok*576U,512U*4U);memcpy(a->k_pe+(size_t)tok*64U,a->kv_cmpr_pe+(size_t)tok*576U+512U,64U*4U);strat01_r2a_rope64(a->k_pe+(size_t)tok*64U,strat01_r2a_positions[tok]);}
    {float *tmp=strat01_r2a_alloc((size_t)count*512U,error);if(!tmp)goto fail;strat01_r2b_rmsnorm_double_sum(a->kv_cmpr+(size_t)start*512U,kv_w,tmp,count,512U,STRAT01_R2A_RMS_EPS);memcpy(a->kv_cmpr+(size_t)start*512U,tmp,(size_t)count*512U*4U);free(tmp);}
    for(unsigned tok=start;tok<start+count;++tok)for(unsigned head=0;head<32U;++head){float *qp=a->q_pe+((size_t)tok*32U+head)*64U;memcpy(qp,a->q+(size_t)tok*6144U+(size_t)head*192U+128U,64U*4U);strat01_r2a_rope64(qp,strat01_r2a_positions[tok]);}
    if(!strat01_combined_kb_range(kb_weights,a->q+(size_t)start*6144U,count,a->q_abs+(size_t)start*32U*512U,error))goto fail;
    for(unsigned tok=start;tok<start+count;++tok){memcpy(a->kcur+(size_t)tok*576U,a->kv_cmpr+(size_t)tok*512U,512U*4U);memcpy(a->kcur+(size_t)tok*576U+512U,a->k_pe+(size_t)tok*64U,64U*4U);memcpy(a->vcur+(size_t)tok*512U,a->kv_cmpr+(size_t)tok*512U,512U*4U);for(unsigned head=0;head<32U;++head){float *qc=a->qcur+((size_t)tok*32U+head)*576U;memcpy(qc,a->q_abs+((size_t)tok*32U+head)*512U,512U*4U);memcpy(qc+512U,a->q_pe+((size_t)tok*32U+head)*64U,64U*4U);}}
    free(attn_w);free(kv_w);return 1;
fail: free(attn_w);free(kv_w);return 0;
}

static int strat01_combined_write_kb_checkpoints(const char *out_dir,const char *arm,const strat01_r2a_arm *a,char q8_path[1024],char q8_sha[65],char out_path[1024],char out_sha[65],char error[256]) {
    strat01_kb_q8_0_block *q8=(strat01_kb_q8_0_block *)malloc(STRAT01_KB_Q5Q8_Q8_BLOCKS*sizeof(*q8));char leaf[128];int ok=0;if(!q8){snprintf(error,256,"combined K-B checkpoint allocation failed");return 0;}
    for(unsigned tok=0;tok<8U;++tok)for(unsigned head=0;head<32U;++head)strat01_kb_quantize_q8_0(a->q+(size_t)tok*6144U+(size_t)head*192U,q8+((size_t)tok*32U+head)*4U);
    snprintf(leaf,sizeof(leaf),"%s_kb_q8.bin",arm);if(!strat01_kb_write_payload(out_dir,leaf,q8,STRAT01_KB_Q5Q8_Q8_BLOCKS*sizeof(*q8),q8_path,q8_sha,error))goto finish;
    snprintf(leaf,sizeof(leaf),"%s_kb_q_abs.f32le",arm);if(!strat01_kb_write_payload(out_dir,leaf,a->q_abs,STRAT01_KB_Q5Q8_OUTPUT_COUNT*4U,out_path,out_sha,error))goto finish;
    if(strcmp(q8_sha,STRAT01_COMBINED_KB_Q8_SHA)||strcmp(out_sha,STRAT01_COMBINED_KB_OUT_SHA)){snprintf(error,256,"combined K-B frozen checkpoint mismatch");goto finish;}ok=1;
finish: free(q8);return ok;
}

static int strat01_combined_write_failure(const char *out_dir,const char *error) { char path[1024];FILE *f;if(!strat01_r2a_path(path,out_dir,"strat01_combined_rms_q5q8.json")||(f=fopen(path,"wb"))==NULL)return 0;fputs("{\"command\":\"--strat01-combined-rms-q5q8\",\"state\":\"DIAGNOSTIC_FAILURE\",\"error\":",f);strat01_json_string(f,error);fputs(",\"donor_graph_executions\":0}\n",f);fclose(f);return 1; }

static int strat01_combined_rms_q5q8_cli(const char *model,const char *baseline_attn_path,const char *baseline_ffn_path,const char *out_dir,const char *engine_source_path) {
    strat01_inventory inv;const strat01_tensor *base[8]={0},*ffn[3]={0};strat01_r2a_arm pre,cache;strat01_upstream_ffn_arm pre_ffn,cache_ffn;strat01_r2a_dump dumps[12];uint8_t *kb_weights=NULL;
    float *baseline_attn=NULL,*baseline_ffn=NULL;strat01_q8_k_block *ba=NULL,*ca=NULL,*bf=NULL,*cf=NULL;size_t ba_n=0,ca_n=0,bf_n=0,cf_n=0,attn_cb=0,attn_cy=0,ffn_cb=0,ffn_cy=0;
    char error[256]={0},model_sha[65]={0},engine_sha[65]={0},header_sha[65]={0},attn_input_sha[65]={0},ffn_input_sha[65]={0};char ffn_shas[2][3][65],q8_paths[4][1024],q8_shas[4][65],kb_q8_paths[2][1024],kb_q8_shas[2][65],kb_out_paths[2][1024],kb_out_shas[2][65],pf_cache_path[1024],pf_cache_sha[65],c7_prefix_path[1024],c7_prefix_sha[65],c7_final_path[1024],c7_final_sha[65],report_path[1024];
    uint64_t hashed=0,parsed=0,tmp=0;FILE *report=NULL;int ok=0;memset(&inv,0,sizeof(inv));memset(&pre,0,sizeof(pre));memset(&cache,0,sizeof(cache));memset(&pre_ffn,0,sizeof(pre_ffn));memset(&cache_ffn,0,sizeof(cache_ffn));memset(ffn_shas,0,sizeof(ffn_shas));
#if !defined(__clang__)
    snprintf(error,256,"combined diagnostic requires Clang");goto finish;
#endif
    if(sizeof(float)!=4||sizeof(double)!=8||sizeof(strat01_q8_k_block)!=292U||sizeof(strat01_kb_q8_0_block)!=34U){snprintf(error,256,"combined host layout mismatch");goto finish;}
    if(!strat01_sha256_file(model,model_sha,&hashed,error)||hashed!=STRAT01_EXPECTED_SIZE||strcmp(model_sha,STRAT01_EXPECTED_SHA256)){if(!error[0])snprintf(error,256,"combined artifact identity mismatch");goto finish;}
    if(!strat01_parse_gguf(model,&inv,&parsed,error)||parsed!=hashed){if(!error[0])snprintf(error,256,"combined GGUF parse mismatch");goto finish;}
    for(unsigned i=0;i<8U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2a_specs[i],&base[i],error))goto finish;for(unsigned i=0;i<3U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2b_specs[i],&ffn[i],error))goto finish;
    if(!strat01_kb_q5q8_descriptor_ok(base[5])){snprintf(error,256,"combined K-B descriptor mismatch");goto finish;}kb_weights=(uint8_t *)malloc((size_t)base[5]->span);if(!kb_weights||!strat01_kb_load_weights(model,base[5],kb_weights,error)){if(!error[0])snprintf(error,256,"combined K-B weight load failed");goto finish;}
    baseline_attn=strat01_r2a_alloc(8U*1536U,error);baseline_ffn=strat01_r2a_alloc(8U*1536U,error);if(!baseline_attn||!baseline_ffn)goto finish;if(!strat01_upstream_read_f32(baseline_attn_path,STRAT01_UPSTREAM_ATTN_BASELINE_SHA,baseline_attn,8U*1536U,attn_input_sha,error)||!strat01_upstream_read_f32(baseline_ffn_path,STRAT01_UPSTREAM_FFN_BASELINE_SHA,baseline_ffn,8U*1536U,ffn_input_sha,error))goto finish;
    if(!strat01_r2a_arm_alloc(&pre,error)||!strat01_r2a_arm_alloc(&cache,error)||!strat01_upstream_ffn_alloc(&pre_ffn,error)||!strat01_upstream_ffn_alloc(&cache_ffn,error))goto finish;
    if(!strat01_combined_build_range(model,base,kb_weights,&pre,0,8,error)||!strat01_r2a_run_schedule(model,base[6],base[7],&pre,0,error))goto finish;
    if(!strat01_combined_build_range(model,base,kb_weights,&cache,0,7,error))goto finish;for(unsigned t=0;t<7U;++t)strat01_r2a_cache_write(&cache,t,cache.kcur+(size_t)t*576U);
    if(!strat01_r2a_write_cache_dump(out_dir,"cached7p1_cache_prefix7.f32",cache.cache,7,c7_prefix_path,c7_prefix_sha,error))goto finish;memset(cache.cache,0,sizeof(cache.cache));cache.occupied=0;
    if(!strat01_combined_build_range(model,base,kb_weights,&cache,7,1,error)||!strat01_r2a_run_schedule(model,base[6],base[7],&cache,1,error))goto finish;
    if(!strat01_combined_write_kb_checkpoints(out_dir,"prefill8",&pre,kb_q8_paths[0],kb_q8_shas[0],kb_out_paths[0],kb_out_shas[0],error)||!strat01_combined_write_kb_checkpoints(out_dir,"cached7p1",&cache,kb_q8_paths[1],kb_q8_shas[1],kb_out_paths[1],kb_out_shas[1],error))goto finish;
    if(!strat01_upstream_ffn_run(model,ffn[0],ffn[1],ffn[2],&pre,&pre_ffn,error)||!strat01_upstream_ffn_run(model,ffn[0],ffn[1],ffn[2],&cache,&cache_ffn,error))goto finish;
    strat01_r2a_make_dumps(&pre,dumps);for(unsigned i=0;i<12U;++i)if(!strat01_r2a_dump_f32(&dumps[i],out_dir,"prefill8",error))goto finish;if(!strat01_r2a_write_cache_dump(out_dir,"prefill8_cache_final.f32",pre.cache,8,pf_cache_path,pf_cache_sha,error)||!strat01_r2a_write_manifest(out_dir,"prefill8",dumps,pf_cache_path,pf_cache_sha,NULL,NULL,error))goto finish;
    strat01_r2a_make_dumps(&cache,dumps);for(unsigned i=0;i<12U;++i)if(!strat01_r2a_dump_f32(&dumps[i],out_dir,"cached7p1",error))goto finish;if(!strat01_r2a_write_cache_dump(out_dir,"cached7p1_cache_final.f32",cache.cache,8,c7_final_path,c7_final_sha,error)||!strat01_r2a_write_manifest(out_dir,"cached7p1",dumps,c7_final_path,c7_final_sha,c7_prefix_path,c7_prefix_sha,error))goto finish;
    if(!strat01_upstream_dump_ffn(&pre_ffn,out_dir,"prefill8_double",ffn_shas[0],error)||!strat01_upstream_dump_ffn(&cache_ffn,out_dir,"cached7p1_double",ffn_shas[1],error))goto finish;
    if(!strat01_upstream_q8_write(out_dir,"baseline_attn_norm.q8k",baseline_attn,8,1536,q8_paths[0],q8_shas[0],&ba,&ba_n,error)||!strat01_upstream_q8_write(out_dir,"candidate_attn_norm.q8k",pre.attn_norm,8,1536,q8_paths[1],q8_shas[1],&ca,&ca_n,error)||!strat01_upstream_q8_write(out_dir,"baseline_ffn_norm.q8k",baseline_ffn,8,1536,q8_paths[2],q8_shas[2],&bf,&bf_n,error)||!strat01_upstream_q8_write(out_dir,"candidate_ffn_norm.q8k",pre_ffn.norm,8,1536,q8_paths[3],q8_shas[3],&cf,&cf_n,error))goto finish;
    if(ba_n!=ca_n||bf_n!=cf_n){snprintf(error,256,"combined Q8_K census size mismatch");goto finish;}strat01_upstream_q8_census(ba,ca,ba_n,&attn_cb,&attn_cy);strat01_upstream_q8_census(bf,cf,bf_n,&ffn_cb,&ffn_cy);
    if(!strat01_sha256_file(engine_source_path,engine_sha,&tmp,error))goto finish;if(!strat01_sha256_file(__FILE__,header_sha,&tmp,error))goto finish;
    if(!strat01_r2a_path(report_path,out_dir,"strat01_combined_rms_q5q8.json")||(report=fopen(report_path,"wb"))==NULL){snprintf(error,256,"cannot write combined report");goto finish;}
    fputs("{\n  \"command\": \"--strat01-combined-rms-q5q8\",\n  \"state\": \"OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\",\n  \"self_certifies_pass\": false,\n  \"model\": {\"path\": ",report);strat01_json_string(report,model);fprintf(report,", \"bytes\": %" PRIu64 ", \"sha256\": ",hashed);strat01_json_string(report,model_sha);
    fputs("},\n  \"baseline_inputs\": {\"attn_norm\": {\"path\": ",report);strat01_json_string(report,baseline_attn_path);fputs(", \"sha256\": ",report);strat01_json_string(report,attn_input_sha);fputs("}, \"ffn_norm\": {\"path\": ",report);strat01_json_string(report,baseline_ffn_path);fputs(", \"sha256\": ",report);strat01_json_string(report,ffn_input_sha);fputs("}},\n  \"engine_source_sha256\": ",report);strat01_json_string(report,engine_sha);fputs(",\n  \"diagnostic_source_sha256\": ",report);strat01_json_string(report,header_sha);
    fprintf(report,",\n  \"kb_checkpoints\": {\"prefill8\": {\"q8_sha256\": \"%s\", \"output_sha256\": \"%s\"}, \"cached7p1\": {\"q8_sha256\": \"%s\", \"output_sha256\": \"%s\"}},\n",kb_q8_shas[0],kb_out_shas[0],kb_q8_shas[1],kb_out_shas[1]);
    fprintf(report,"  \"ffn_output_sha256\": {\"prefill8\": {\"norm\": \"%s\", \"up\": \"%s\", \"gate\": \"%s\"}, \"cached7p1\": {\"norm\": \"%s\", \"up\": \"%s\", \"gate\": \"%s\"}},\n",ffn_shas[0][0],ffn_shas[0][1],ffn_shas[0][2],ffn_shas[1][0],ffn_shas[1][1],ffn_shas[1][2]);
    fprintf(report,"  \"q8_census\": {\"attention\": {\"changed_blocks\": %zu, \"total_blocks\": %zu, \"changed_bytes\": %zu, \"total_bytes\": %zu}, \"ffn\": {\"changed_blocks\": %zu, \"total_blocks\": %zu, \"changed_bytes\": %zu, \"total_bytes\": %zu}},\n",attn_cb,ba_n,attn_cy,ba_n*sizeof(*ba),ffn_cb,bf_n,ffn_cy,bf_n*sizeof(*bf));
    fputs("  \"compiler_family\": \"clang\",\n  \"donor_graph_executions\": 1,\n  \"timing_or_rate_claim\": null\n}\n",report);if(fclose(report)!=0){report=NULL;snprintf(error,256,"combined report close failure");goto finish;}report=NULL;ok=1;
finish:
    if(report)fclose(report);free(kb_weights);free(baseline_attn);free(baseline_ffn);free(ba);free(ca);free(bf);free(cf);strat01_free_inventory(&inv);strat01_r2a_arm_free(&pre);strat01_r2a_arm_free(&cache);strat01_upstream_ffn_free(&pre_ffn);strat01_upstream_ffn_free(&cache_ffn);
    if(!ok){if(!error[0])snprintf(error,256,"unspecified combined diagnostic failure");strat01_combined_write_failure(out_dir,error);fprintf(stderr,"STRAT-01 combined RMS/Q5Q8 refused: %s\n",error);return 2;}fprintf(stderr,"STRAT-01 combined RMS/Q5Q8: OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\n");return 0;
}

static int strat01_combined_rms_q5q8_selftest(void) { int bad=0,checks=0;
#define COMBINED_CHECK(x) do{++checks;if(!(x))++bad;}while(0)
    COMBINED_CHECK(strat01_upstream_rmsnorm_diag_selftest()==0);COMBINED_CHECK(strat01_kb_q5q8_diag_selftest()==0);COMBINED_CHECK(strlen(STRAT01_COMBINED_KB_Q8_SHA)==64U);COMBINED_CHECK(strlen(STRAT01_COMBINED_KB_OUT_SHA)==64U);
#undef COMBINED_CHECK
    fprintf(stderr,"STRAT-01 combined RMS/Q5Q8 selftest: %s (%d checks)\n",bad?"FAIL":"PASS",checks);return bad?1:0; }

#endif
